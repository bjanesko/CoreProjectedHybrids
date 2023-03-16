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

'''
Projected Hybrid Density functional theory
=========================

Simple usage::

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='def2-tzvp')
    >>> mf = dft.RKS(mol,paos='CoreAOs',phyb=1)
    >>> mf.xc = 'pbe,pbe'
    >>> mf.run()
'''

try:
    from pyscf.dft import libxc
    XC = libxc.XC
except (ImportError, OSError):
    pass
try:
    from pyscf.dft import xcfun
    XC = xcfun.XC
except (ImportError, OSError):
    pass
#from pyscf.dft import xc
#from pyscf.dft import rks
from pdft import rks
#from pyscf.dft import roks
from pdft import roks 
#from pyscf.dft import uks
from pdft import uks
from pyscf.dft import gks
from pyscf.dft import rks_symm
from pyscf.dft import uks_symm
from pyscf.dft import gks_symm
from pyscf.dft import dks
from pyscf.dft import gen_grid as grid
from pyscf.dft import radi
from pyscf.df import density_fit
from pyscf.dft.gen_grid import sg1_prune, nwchem_prune, treutler_prune, \
        stratmann, original_becke, Grids
from pyscf.dft.radi import BRAGG_RADII, COVALENT_RADII, \
        delley, mura_knowles, gauss_chebyshev, treutler, treutler_ahlrichs, \
        treutler_atomic_radii_adjust, becke_atomic_radii_adjust


def KS(mol, xc='LDA,VWN',paos='',phyb=0, rew=None, allc=0):
    if mol.spin == 0:
        return RKS(mol, xc, paos, phyb, rew, allc)
    else:
        return UKS(mol, xc, paos, phyb, rew, allc)
KS.__doc__ = '''
A wrap function to create DFT object (RKS or UKS).\n
''' + rks.RKS.__doc__
DFT = KS

def RKS(mol, xc='LDA,VWN', paos='', phyb=0, rew=None, allc=0):
    if mol.nelectron == 1:
        return uks.UKS(mol, xc, paos, phyb, rew, allc)
    elif not mol.symmetry or mol.groupname == 'C1':
        if mol.spin > 0:
            return roks.ROKS(mol, xc, paos, phyb, rew, allc)
        else:
            return rks.RKS(mol, xc, paos, phyb, rew, allc)
    else:
        if mol.spin > 0:
            return rks_symm.ROKS(mol, xc, paos, phyb, rew, allc)
        else:
            return rks_symm.RKS(mol, xc, paos, phyb, rew, allc)
RKS.__doc__ = rks.RKS.__doc__

def ROKS(mol, xc='LDA,VWN', paos='', phyb=0, rew=None, allc=0):
    if mol.nelectron == 1:
        return uks.UKS(mol, xc, paos, phyb, rew, allc)
    elif not mol.symmetry or mol.groupname == 'C1':
        return roks.ROKS(mol, xc, paos, phyb, rew, allc)
    else:
        return rks_symm.ROKS(mol, xc, paos, phyb, rew, allc)
ROKS.__doc__ = roks.ROKS.__doc__

#def UKS(mol, xc='LDA,VWN'):
def UKS(mol, xc='LDA,VWN', paos='', phyb=0, rew=None, allc=0):
    if not mol.symmetry or mol.groupname == 'C1':
        return uks.UKS(mol, xc, paos, phyb, rew, allc)
    else:
        return uks_symm.UKS(mol, xc)
UKS.__doc__ = uks.UKS.__doc__

def GKS(mol, xc='LDA,VWN'):
    if not mol.symmetry or mol.groupname == 'C1':
        return gks.GKS(mol, xc)
    else:
        return gks_symm.GKS(mol, xc)
GKS.__doc__ = gks.GKS.__doc__

def DKS(mol, xc='LDA,VWN'):
    from pyscf.scf import dhf
    if dhf.zquatev and mol.spin == 0:
        return dks.RDKS(mol, xc=xc)
    else:
        return dks.UDKS(mol, xc=xc)
