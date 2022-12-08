from pyscf import gto,scf,dft
import sys
sys.path.append('../')
import pdft 

m=gto.Mole(atom='He 0.0 0.0 0.0; He 0.0 0.0 30.0',basis='3-21g') 
m.build()

print('PBE ')
md1=dft.RKS(m,xc='pbe,')
md1.kernel() 
print(md1.mo_energy)

print('HF ')
md2=dft.RKS(m,xc='hf,')
md2.kernel() 
print(md2.mo_energy)

# Total energies should be midway between HF and PBE, and both kinds of orbital energy should be present
print('Half-And-Half')
md3=pdft.RKS(m,xc='pbe,',phyb=1.0,paos=[0,1])
md3.kernel() 
print(md3.mo_energy)
print('Total energy test: %12.6f ' % (md1.e_tot+md2.e_tot-2*md3.e_tot))

