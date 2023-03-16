#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Generate SCF response functions
These are based on scf/_response_functions.py
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, rohf, uhf, ghf, dhf

def pdft_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None):
    '''Generate a function to compute the product of RHF response function and
    RHF density matrices.

    Kwargs:
        singlet (None or boolean) : If singlet is None, response function for
            orbital hessian or CPHF will be generated. If singlet is boolean,
            it is used in TDPDFT response kernel.
    '''
    assert(not isinstance(mf, (uhf.UHF, rohf.ROHF)))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if _is_dft_object(mf):
        from pyscf.dft import numint
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = abs(hyb) > 1e-10
        if(omega> 1e-10 and alpha>1e-10):
          hybrid = True 
        phybrid = False
        if(hasattr(mf,'phyb')):
          if(abs(sum(mf.phyb)))>1e-10:
            phybrid = True 
        pxc = mf.xc 
        if(phybrid):
         if(mf.allc>0):
          pxc=(pxc.split(','))[0] + ','
        print(' Fraction HFX ',hyb, mf.phyb,hybrid,phybrid)
        print(' hermi ',hermi)
        print(' singlet ',singlet)

        pmo_coeff = [] 
        if(phybrid):
         for ip in range(len(mf.phyb)):
          pmo_coeff_this = numpy.zeros(mo_coeff.shape)
          pmo_coeff_this = numpy.dot(mf.QS[ip],mo_coeff)
          pmo_coeff.append(pmo_coeff_this)
   

        # mf can be pbc.dft.RKS object with multigrid
        if (not hybrid and not phybrid and 
            'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
            print('MULTIGRID') 
            from pyscf.pbc.dft import multigrid
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        if singlet is None:
            # for ground state orbital hessian
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                mo_coeff, mo_occ, 0)
            if(phybrid): # BGJ 
              rho0p = [] 
              vxcp  = [] 
              fxcp  = [] 
              for ip in range(len(mf.phyb)):
                rho0pt, vxcpt, fxcpt = ni.cache_xc_kernel(mol, mf.grids, pxc,   
                                                  pmo_coeff[ip], mo_occ, 0)
                rho0p.append(rho0pt)
                vxcp.append(vxcpt)
                fxcp.append(fxcpt)
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                [mo_coeff]*2, [mo_occ*.5]*2, spin=1)
            if(phybrid):
              rho0p = [] 
              vxcp  = [] 
              fxcp  = [] 
              for ip in range(len(mf.phyb)):
                rho0pt, vxcpt, fxcpt = ni.cache_xc_kernel(mol, mf.grids, pxc,  
                                                  [pmo_coeff[ip]]*2, [mo_occ*.5]*2, spin=1)
                rho0p.append(rho0pt)
                vxcp.append(vxcpt)
                fxcp.append(fxcpt)
        dm0 = None  #mf.make_rdm1(mo_coeff, mo_occ)

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:

            # Without specify singlet, used in ground state orbital hessian
            def vind(dm1):

                # BGJ project the density matrices 
                dm0p = []
                dm1p = [] 
                for ip in range(len(mf.phyb)):
                  dm0pt = None 
                  if(dm0 is not None):
                    dm0pt = numpy.zeros(dm0.shape)
                  dm1pt = numpy.zeros(dm1.shape)
                  if(phybrid):
                    if(dm0 is not None):
                      dm0pt = numpy.einsum('ik,kj->ij',mf.QS[ip],numpy.einsum('ik,kj->ij',dm0,mf.SQ[ip]))
                    for i in range(dm1.shape[0]):
                      dm1pt[i] =  numpy.einsum('ik,kj->ij',mf.QS[ip],numpy.einsum('ik,kj->ij',dm1[i],mf.SQ[ip]))
                  dm0p.append(dm0pt)
                  dm1p.append(dm1pt)

                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, max_memory=max_memory)
                    #print('v1[0] A2:\n',v1[0])

                    if phybrid: # Subtract off core projected DFT 
                      for ip in range(len(mf.phyb)):
                        v1p0 = numint.nr_rks_fxc(ni, mol, mf.grids, pxc, dm0p[ip], dm1p[ip], 0,
                                            hermi, rho0p[ip], vxcp[ip], fxcp[ip],
                                            max_memory=max_memory)
                        for i in range(dm1.shape[0]):
                          v1p = numpy.dot(mf.SQ[ip],numpy.dot(v1p0[i],mf.QS[ip]))
                          v1[i] -= mf.phyb[ip] *  v1p 
                
                if phybrid: # Add core projected HFX 
                  for ip in range(len(mf.phyb)):
                    dm1pt = dm1p[ip]
                    for i in range(v1.shape[0]):
                      vxxp0 =  mf.get_k(mol, dm1pt[i], hermi=hermi)
                      vxxp = numpy.dot(mf.SQ[ip],numpy.dot(vxxp0,mf.QS[ip]))
                      v1[i] -= .5* mf.phyb[ip] * vxxp 
               
                #print('v1[0] A:\n',v1[0])
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if omega > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk

                        if(phybrid):
                          for ip in range(len(mf.phyb)):
                            dm1pt = dm1p[ip]
                            for i in range(v1.shape[0]):
                              vk0 = mf.get_k(mol, dm1pt[i], hermi=hermi)
                              vk0 *= hyb
                              if omega > 1e-10:  # For range separated Coulomb
                                vk0 += mf.get_k(mol, dm1pt[i], hermi, omega) * (alpha-hyb)
                              v1[i] += .5* mf.phyb[ip] *  numpy.dot(mf.SQ[ip],numpy.dot(vk0,mf.QS[ip]))

                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                        if(phybrid):
                          for ip in range(len(mf.phyb)):
                            dm1pt = dm1p[ip]
                            for i in range(v1.shape[0]):
                              vk0 = mf.get_k(mol, dm1pt[i], hermi=hermi)
                              v1[i] += .5* mf.phyb[ip] *  numpy.dot(mf.SQ[ip],numpy.dot(vk0,mf.QS[ip]))

                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                #print('v1[0] B:\n',v1[0])
                return v1

        elif singlet:
            def vind(dm1):

                # BGJ 
                dm0p = []
                dm1p = [] 
                for ip in range(len(mf.phyb)):
                  dm0pt = None 
                  if(dm0 is not None):
                    dm0pt = numpy.zeros(dm0.shape)
                  dm1pt = numpy.zeros(dm1.shape)
                  if(phybrid):
                    if(dm0 is not None):
                      dm0pt = numpy.einsum('ik,kj->ij',mf.QS[ip],numpy.einsum('ik,kj->ij',dm0,mf.SQ[ip]))
                    for i in range(dm1.shape[0]):
                      dm1pt[i] =  numpy.einsum('ik,kj->ij',mf.QS[ip],numpy.einsum('ik,kj->ij',dm1[i],mf.SQ[ip]))
                  dm0p.append(dm0pt)
                  dm1p.append(dm1pt)
                
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = numint.nr_rks_fxc_st(ni,mol, mf.grids, mf.xc, dm0, dm1, 0, True,
                                          rho0, vxc, fxc, max_memory=max_memory)
                    v1 *= .5
                    #print('v1[0] C2:\n',v1[0])
                
                    if phybrid: # Subtract off core projected DFT 
                      for ip in range(len(mf.phyb)):
                        v1p0 = numint.nr_rks_fxc_st(ni,mol, mf.grids, pxc, dm0p[ip], dm1p[ip], 0, True,
                                          rho0p[ip], vxcp[ip], fxcp[ip], max_memory=max_memory)
                        for i in range(dm1.shape[0]):
                          v1p =  .5* numpy.dot(mf.SQ[ip],numpy.dot(v1p0[i],mf.QS[ip]))
                          v1[i] -= mf.phyb[ip] *  v1p 
                
                if phybrid: # Add core projected HFX,same factor of -.5 as below 
                  for ip in range(len(mf.phyb)):
                    dm1pt = dm1p[ip]
                    vxxp0 =  mf.get_k(mol, dm1pt, hermi=hermi)
                    for i in range(dm1.shape[0]):
                      vxxp = numpy.dot(mf.SQ[ip],numpy.dot(vxxp0[i],mf.QS[ip]))
                      v1[i] -= .5* mf.phyb[ip] * vxxp 
                
                if hybrid:
                    if hermi != 2:
                        #print('Here comes the hybrid')
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if omega > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk

                        if phybrid: # Subtract off the projected part 
                          for ip in range(len(mf.phyb)):
                            dm1pt = dm1p[ip]
                            for i in range(v1.shape[0]):
                              vk0 = mf.get_k(mol, dm1pt[i], hermi=hermi)
                              vk0 *= hyb
                              if omega > 1e-10:  # For range separated Coulomb
                                vk0 += mf.get_k(mol, dm1pt[i], hermi, omega) * (alpha-hyb)
                              v1[i] += .5* mf.phyb[ip] *  numpy.dot(mf.SQ[ip],numpy.dot(vk0,mf.QS[ip]))
                    else:
                        vk = mf.get_k(mol, dm1, hermi=hermi)
                        v1 -= .5 * hyb * vk 
                        if phybrid: # Subtract off the projected part 
                          for ip in range(len(mf.phyb)):
                            dm1pt = dm1p[ip]
                            for i in range(v1.shape[0]):
                              vk0 = hyb*mf.get_k(mol, dm1pt[i], hermi=hermi)
                              v1[i] += .5* mf.phyb[ip] *  numpy.dot(mf.SQ[ip],numpy.dot(vk0,mf.QS[ip]))

                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)

                #print('v1[0] D:\n',v1[0])
                return v1
        else:  # triplet
            def vind(dm1):

                # BGJ
                dm0p = []
                dm1p = [] 
                for ip in range(len(mf.phyb)):
                  dm0pt = None 
                  if(dm0 is not None):
                    dm0pt = numpy.zeros(dm0.shape)
                  dm1pt = numpy.zeros(dm1.shape)
                  if phybrid: 
                    if(dm0 is not None):
                      dm0pt = numpy.einsum('ik,kj->ij',mf.QS[ip],numpy.einsum('ik,kj->ij',dm0,mf.SQ[ip]))
                    for i in range(dm1.shape[0]):
                      dm1pt[i] =  numpy.einsum('ik,kj->ij',mf.QS[ip],numpy.einsum('ik,kj->ij',dm1[i],mf.SQ[ip]))
                  dm0p.append(dm0pt)
                  dm1p.append(dm1pt)
               
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = numint.nr_rks_fxc_st(ni, mol, mf.grids, mf.xc, dm0, dm1, 0,
                                              False, rho0, vxc, fxc,
                                              max_memory=max_memory)
                    v1 *= .5

                    # BGJ 
                    if phybrid: 
                      for ip in range(len(mf.phyb)):
                        v1p0 = numint.nr_rks_fxc_st(ni, mol, mf.grids, pxc, dm0p[ip], dm1p[ip], 0,
                                            False, rho0p[ip], vxcp[ip], fxcp[ip],
                                            max_memory=max_memory)
                        for i in range(dm1.shape[0]):
                          v1p =  .5* numpy.dot(mf.SQ[ip],numpy.dot(v1p0[i],mf.QS[ip]))
                          v1[i] -= mf.phyb[ip] *  v1p 
                
                         # BGJ add projected HF exchange ,same factor of -.5 as below 
                        dm1pt = dm1p[ip]
                        for i in range(v1.shape[0]):
                          vxxp0 =  mf.get_k(mol, dm1pt[i], hermi=hermi)
                          vxxp = numpy.dot(mf.SQ[ip],numpy.dot(vxxp0,mf.QS[ip]))
                          v1[i] -= .5* mf.phyb[ip] * vxxp 

                if hybrid:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += -.5 * vk

                    if phybrid:
                      for ip in range(len(mf.phyb)):
                        dm1pt = dm1p[ip]
                        for i in range(v1.shape[0]):
                          vk0 = mf.get_k(mol, dm1pt[i], hermi=hermi)
                          vk0 *= hyb
                          if omega > 1e-10:  # For range separated Coulomb
                            vk0 += mf.get_k(mol, dm1pt[i], hermi, omega) * (alpha-hyb)
                          v1[i] += .5* mf.phyb[ip] *  numpy.dot(mf.SQ[ip],numpy.dot(vk0,mf.QS[ip]))

                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(mol, dm1, hermi=hermi)

    return vind


def pdft_uhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None):
    '''Generate a function to compute the product of UHF response function and
    UHF density matrices.
    '''
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol

    if _is_dft_object(mf):
        from pyscf.dft import numint
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = abs(hyb) > 1e-10
        if(omega> 1e-10 and alpha>1e-10):
          hybrid = True 

        phybrid = False
        if(hasattr(mf,'phyb')):
          if(abs(sum(mf.phyb)))>1e-10:
            phybrid = True 
        pxc = mf.xc 
        if(phybrid):
         if(mf.allc>0):
          pxc=(pxc.split(','))[0] + ','

        if(phybrid):
          print(' Fraction HFX ',hyb, mf.phyb,hybrid,phybrid)

        # BGJ projected MO coefficients
        pmo_coeff = [] 
        if(phybrid):
          for ip in range(len(mf.phyb)):
            if(isinstance(mo_coeff,list) or isinstance(mo_coeff,tuple)):
              pmo_coeff_this = [] 
              for jp in range(len(mo_coeff)):
                pmo_coeff_this.append(numpy.dot(mf.QS[ip],mo_coeff[jp]))
              pmo_coeff.append(pmo_coeff_this)
            else:
              pmo_coeff_this = numpy.zeros(mo_coeff.shape)
              for ii in range((mo_coeff.shape)[0]):
                pmo_coeff_this[ii] = numpy.dot(mf.QS[ip],mo_coeff[ii])
              pmo_coeff.append(pmo_coeff_this)

        # mf can be pbc.dft.UKS object with multigrid
        if (not hybrid and
            'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
            from pyscf.pbc.dft import multigrid
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_uhf_response(mf, dm0, with_j, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1)
        if phybrid:
          rho0p = [] 
          vxcp  = [] 
          fxcp  = [] 
          for ip in range(len(mf.phyb)):
            rho0pt, vxcpt, fxcpt = ni.cache_xc_kernel(mol, mf.grids, pxc,   
                                              pmo_coeff[ip], mo_occ, 1)
            rho0p.append(rho0pt)
            vxcp.append(vxcpt)
            fxcp.append(fxcpt)

        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1):

            # BGJ projected density matrices 
            dm0p = []
            dm1p = [] 
            if(phybrid):
              for ip in range(len(mf.phyb)):
                dm0pt = None 
                if(dm0 is not None):
                  dm0pt = numpy.zeros(dm0.shape)
                dm1pt = numpy.zeros(dm1.shape)
                if(dm0 is not None):
                  for i in range(dm0.shape[0]):
                    for j in range(dm0.shape[1]):
                      dm0pt[i,j] =  numpy.dot(mf.QS[ip],numpy.dot(dm0[i,j],mf.SQ[ip]))
                for i in range(dm1.shape[0]):
                  for j in range(dm1.shape[1]):
                    dm1pt[i,j] =  numpy.dot(mf.QS[ip],numpy.dot(dm1[i,j],mf.SQ[ip]))
                dm0p.append(dm0pt)
                dm1p.append(dm1pt)

            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, max_memory=max_memory)

                if phybrid: # Subtract off core projected DFT 
                  for ip in range(len(mf.phyb)):
                    v1p0 = numint.nr_uks_fxc(ni,mol, mf.grids, pxc,   dm0p[ip], dm1p[ip], 0,
                                            hermi, rho0p[ip], vxcp[ip], fxcp[ip],
                                            max_memory=max_memory)
                    for i in range(dm1.shape[0]):
                      for j in range(dm1.shape[1]):
                        v1p = numpy.dot(mf.SQ[ip],numpy.dot(v1p0[i,j],mf.QS[ip]))
                        v1[i,j] -= mf.phyb[ip] *  v1p 

            if phybrid: # Add core projected HFX , factor of -1 matches the hybrid - vk below 
               for ip in range(len(mf.phyb)):
                 dm1pthis = dm1p[ip]
                 #vxxp0 =  mf.get_k(mol, dm1pthis, hermi=hermi)
                 for i in range(v1.shape[0]):
                  for j in range(v1.shape[1]):
                   vxxp0 =  mf.get_k(mol, dm1pt[i,j], hermi=hermi)
                   #vxxp = numpy.dot(mf.SQ[ip],numpy.dot(vxxp0[i,j],mf.QS[ip]))
                   vxxp = numpy.dot(mf.SQ[ip],numpy.dot(vxxp0,mf.QS[ip]))
                   v1[i,j] -= mf.phyb[ip] * vxxp 

            if not hybrid:
                if with_j:
                    vj = mf.get_j(mol, dm1, hermi=hermi)
                    v1 += vj[0] + vj[1]
            else:
                if with_j:
                    vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += vj[0] + vj[1] - vk
                    if phybrid: # Subtract off unprojected HFX from hybrid 
                      for ip in range(len(mf.phyb)):
                         dm1pthis = dm1p[ip]
                         for i in range(v1.shape[0]):
                           for j in range(v1.shape[1]):
                             vk0 = mf.get_k(mol, dm1pthis[i,j], hermi=hermi)
                             vk0 *= hyb
                             if omega > 1e-10:  # For range separated Coulomb
                               vk0 += mf.get_k(mol, dm1pthis[i,j], hermi, omega) * (alpha-hyb)
                             v1[i,j] += mf.phyb[ip] *  numpy.dot(mf.SQ[ip],numpy.dot(vk0,mf.QS[ip]))
                else:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 -= vk
                    if phybrid: # Subtract off unprojected HFX from hybrid 
                      for ip in range(len(mf.phyb)):
                         dm1pthis = dm1p[ip]
                         for i in range(v1.shape[0]):
                           for j in range(v1.shape[1]):
                             vk0 = mf.get_k(mol, dm1pthis[i,j], hermi=hermi)
                             vk0 *= hyb
                             if omega > 1e-10:  # For range separated Coulomb
                               vk0 += mf.get_k(mol, dm1pthis[i,j], hermi, omega) * (alpha-hyb)
                             v1[i,j] += mf.phyb[ip] *  numpy.dot(mf.SQ[ip],numpy.dot(vk0,mf.QS[ip]))

            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _is_dft_object(mf):
    return getattr(mf, 'xc', None) is not None and hasattr(mf, '_numint')
