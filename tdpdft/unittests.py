from pyscf import gto,scf,dft,tddft
import sys
sys.path.append('../')
import pdft, tdpdft

m=gto.Mole(atom='He 0.0 0.0 0.0',basis='3-21g') 
m.build()

print('PBE monomer ')
md=dft.RKS(m,xc='pbe,pbe')
md.kernel() 
print(md.mo_energy)
mt = tddft.TDDFT(md)
mt.kernel() 

print('HF monomer ')
md=dft.RKS(m,xc='hf,pbe')
md.kernel() 
print(md.mo_energy)
mt = tddft.TDDFT(md)
mt.kernel() 

m=gto.Mole(atom='He 0.0 0.0 0.0; He 0.0 0.0 30.0',basis='3-21g') 
m.build()

print('PBE ')
md1=dft.RKS(m,xc='pbe,pbe')
md1.kernel() 
print(md1.mo_energy)
mt1 = tddft.TDDFT(md1)
mt1.nstates=4 
mt1.kernel() 

print('HF ')
md2=dft.RKS(m,xc='hf,pbe')
md2.kernel() 
print(md2.mo_energy)
mt2 = tddft.TDDFT(md2)
mt2.nstates=4 
mt2.kernel() 

# Total energies should be midway between HF and PBE, and both kinds of orbital energy should be present
print('Half-And-Half')
m.verbose = 3 
md3=pdft.RKS(m,xc='pbe,pbe',phyb=[1.0],paos=[[0,1]],allc=1)
md3.kernel() 
print(md3.mo_energy)
print('Total energy test: %12.6f ' % (md1.e_tot+md2.e_tot-2*md3.e_tot))
mt3 = tdpdft.TDPDFT(md3)
mt3.nstates=8
mt3.kernel() 

# Recheck for unrestricted, should be identical 
print('Half-And-Half UHF')
m.verbose = 3 
md3=pdft.UKS(m,xc='pbe,pbe',phyb=[1.0],paos=[[0,1]],allc=1)
md3.kernel() 
print(md3.mo_energy)
print('Total energy test: %12.6f ' % (md1.e_tot+md2.e_tot-2*md3.e_tot))
mt3 = tdpdft.TDPDFT(md3)
mt3.nstates=8
mt3.kernel() 

