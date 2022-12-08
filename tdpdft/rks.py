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
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

import time
import numpy
from pyscf import lib
from pyscf import symm
from pyscf import scf
#from pyscf.tdscf import rhf
from tdpdft import rhf
from pyscf.scf import hf_symm
from pyscf.scf import _response_functions  # noqa
from pyscf.data import nist
from pyscf.dft.rks import KohnShamDFT
from pyscf import __config__
from tdpdft import pdft_response_functions 

# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDPDFT_positive_eig_threshold', 1e-3)


class TDA(rhf.TDA):
    def nuc_grad_method(self):
        from pyscf.grad import tdrks
        return tdrks.Gradients(self)

class TDPDFT(rhf.TDHF):
    def nuc_grad_method(self):
        from pyscf.grad import tdrks
        return tdrks.Gradients(self)

RPA = TDRKS = TDPDFT

class TDPDFTNoHybrid(TDA):
    ''' Solve (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''
    def gen_vind(self, mf):
        wfnsym = self.wfnsym
        singlet = self.singlet

        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ==2)[0]
        if(mf.rew is not None): 
          occidx = numpy.where(numpy.logical_and(mf.rew[0]<mo_energy,mo_energy<mf.rew[1]))[0]
        viridx = numpy.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:,viridx]
        orbo = mo_coeff[:,occidx]

        if wfnsym is not None and mol.symmetry:
            if isinstance(wfnsym, str):
                wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
            wfnsym = wfnsym % 10  # convert to D2h subgroup
            orbsym = hf_symm.get_orbsym(mol, mo_coeff)
            orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
            sym_forbid = (orbsym_in_d2h[occidx,None] ^ orbsym_in_d2h[viridx]) != wfnsym

        e_ia = (mo_energy[viridx].reshape(-1,1) - mo_energy[occidx]).T
        if wfnsym is not None and mol.symmetry:
            e_ia[sym_forbid] = 0
        d_ia = numpy.sqrt(e_ia)
        ed_ia = e_ia * d_ia
        hdiag = e_ia.ravel() ** 2

        # BGJ 
        # vresp = mf.gen_response(singlet=singlet, hermi=1)
        vresp = pdft_response_functions.pdft_rhf_response(mf, singlet=singlet, hermi=1)

        def vind(zs):
            zs = numpy.asarray(zs).reshape(-1,nocc,nvir)
            # *2 for double occupancy
            dmov = lib.einsum('xov,ov,po,qv->xpq', zs, d_ia*2, orbo, orbv.conj())
            # +cc for A+B and K_{ai,jb} in A == K_{ai,bj} in B
            dmov = dmov + dmov.conj().transpose(0,2,1)

            v1ao = vresp(dmov)
            v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo.conj(), orbv)

            # numpy.sqrt(e_ia) * (e_ia*d_ia*z + v1ov)
            v1ov += numpy.einsum('xov,ov->xov', zs, ed_ia)
            v1ov *= d_ia
            return v1ov.reshape(v1ov.shape[0],-1)

        return vind, hdiag

    def kernel(self, x0=None, nstates=None):
        '''TDPDFT diagonalization solver
        '''
        #cpu0 = (time.clock(), time.time())
        mf = self._scf
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            raise RuntimeError('%s cannot be used with hybrid functional'
                               % self.__class__)
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = lib.logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > POSTIVE_EIG_THRESHOLD**2)[0]
            return w[idx], v[:,idx], idx

        self.converged, w2, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_space=self.max_space, pick=pickeig,
                              verbose=log)

        mo_energy = self._scf.mo_energy
        mo_occ = self._scf.mo_occ
        occidx = numpy.where(mo_occ==2)[0]
        if(mf.rew is not None): 
          occidx = numpy.where(numpy.logical_and(mf.rew[0]<mo_energy,mo_energy<mf.rew[1]))[0]
        viridx = numpy.where(mo_occ==0)[0]
        e_ia = (mo_energy[viridx,None] - mo_energy[occidx]).T
        e_ia = numpy.sqrt(e_ia)
        def norm_xy(w, z):
            zp = e_ia * z.reshape(e_ia.shape)
            zm = w/e_ia * z.reshape(e_ia.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
            return (x*norm, y*norm)

        idx = numpy.where(w2 > POSTIVE_EIG_THRESHOLD**2)[0]
        self.e = numpy.sqrt(w2[idx])
        self.xy = [norm_xy(self.e[i], x1[i]) for i in idx]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tdpdft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tdpdft/xy', self.xy)

        #log.timer('TDPDFT', *cpu0)
        log.note('Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self.e, self.xy

    def nuc_grad_method(self):
        from pyscf.grad import tdrks
        return tdrks.Gradients(self)


class dRPA(TDPDFTNoHybrid):
    def __init__(self, mf):
        if not isinstance(mf, KohnShamDFT):
            raise RuntimeError("direct RPA can only be applied with DFT; for HF+dRPA, use .xc='hf'")
        mf = scf.addons.convert_to_rhf(mf)
        # commit fc8d1967995b7e033b60d4428ddcca87aac78e4f handles xc='' .
        # xc='0*LDA' is equivalent to xc=''
        #mf.xc = '0.0*LDA'
        mf.xc = ''
        TDPDFTNoHybrid.__init__(self, mf)

TDH = dRPA

class dTDA(TDA):
    def __init__(self, mf):
        if not isinstance(mf, KohnShamDFT):
            raise RuntimeError("direct TDA can only be applied with DFT; for HF+dTDA, use .xc='hf'")
        mf = scf.addons.convert_to_rhf(mf)
        # commit fc8d1967995b7e033b60d4428ddcca87aac78e4f handles xc='' .
        # xc='0*LDA' is equivalent to xc=''
        #mf.xc = '0.0*LDA'
        mf.xc = ''
        TDA.__init__(self, mf)


def tdpdft(mf):
    '''Driver to create TDPDFT or TDPDFTNoHybrid object'''
    #print('Now in tdpdft rks tddft with ',mf.phyb)
    if(mf._numint.libxc.is_hybrid_xc(mf.xc) or mf.phyb>0):
        return TDPDFT(mf)
    else:
        return TDPDFTNoHybrid(mf)

from pyscf import dft
dft.rks.RKS.TDA           = dft.rks_symm.RKS.TDA           = lib.class_as_method(TDA)
dft.rks.RKS.TDHF          = dft.rks_symm.RKS.TDHF          = None
#dft.rks.RKS.TDPDFT         = dft.rks_symm.RKS.TDPDFT         = lib.class_as_method(TDPDFT)
dft.rks.RKS.TDPDFTNoHybrid = dft.rks_symm.RKS.TDPDFTNoHybrid = lib.class_as_method(TDPDFTNoHybrid)
dft.rks.RKS.TDPDFT         = dft.rks_symm.RKS.TDPDFT         = tdpdft
dft.rks.RKS.dTDA          = dft.rks_symm.RKS.dTDA          = lib.class_as_method(dTDA)
dft.rks.RKS.dRPA          = dft.rks_symm.RKS.dRPA          = lib.class_as_method(dRPA)
dft.roks.ROKS.TDA           = dft.rks_symm.ROKS.TDA           = None
dft.roks.ROKS.TDHF          = dft.rks_symm.ROKS.TDHF          = None
dft.roks.ROKS.TDPDFT         = dft.rks_symm.ROKS.TDPDFT         = None
dft.roks.ROKS.TDPDFTNoHybrid = dft.rks_symm.ROKS.TDPDFTNoHybrid = None
dft.roks.ROKS.dTDA          = dft.rks_symm.ROKS.dTDA          = None
dft.roks.ROKS.dRPA          = dft.rks_symm.ROKS.dRPA          = None

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'lda, vwn_rpa'
    mf.scf()
    td = mf.TDPDFTNoHybrid()
    #td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [  9.74227238   9.74227238  14.85153818  30.35019348  30.35019348]
    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [  9.08754045   9.08754045  12.48375957  29.66870808  29.66870808]

    mf = dft.RKS(mol)
    mf.xc = 'b88,p86'
    mf.scf()
    td = mf.TDPDFT()
    td.nstates = 5
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.82204435   9.82204435  15.0410193   30.01373062  30.01373062]
    td.singlet = False
    print(td.kernel()[0] * 27.2114)
# [  9.09322358   9.09322358  12.29843139  29.26731075  29.26731075]

    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = mf.TDA()
    print(td.kernel()[0] * 27.2114)
# [  9.68872769   9.68872769  15.07122478]
    td.singlet = False
    #td.verbose = 5
    print(td.kernel()[0] * 27.2114)
# [  9.0139312    9.0139312   12.42444659]

    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = dRPA(mf)
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [ 10.00343861  10.00343861  15.62586305  30.69238874  30.69238874]

    mf = dft.RKS(mol)
    mf.xc = 'lda,vwn'
    mf.scf()
    td = dTDA(mf)
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
# [ 10.05245288  10.05245288  16.03497655  30.7120363   30.7120363 ]

