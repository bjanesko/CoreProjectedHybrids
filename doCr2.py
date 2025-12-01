from pyscf import scf,gto,dft,cc
import numpy ,scipy, sys, os.path 
import pdft
from scipy import linalg
from pdft.projwork import euci , euci5, eci , epzlh, build_proj, P_1to2
import dftd3.pyscf as dftd3

# B4LYP-UCI four-parameter hybrid 
fullbasis='def2tzvp'

# Some dummy terms 
names=['Dummy','H',          'He',
'Li','Be',   'B','C','N','O','F','Ne',
'Na','Mg',   'Al','Si','P','S','Cl','Ar',
'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr']
 
def readChk(file):

  # Read header information 
  NAO=0 
  NMO=0 
  Nat=0 
  b=''
  charge=0
  spin=0
  f = open(file,'r') 
  lines = f.readlines()
  b = lines[1].split().pop()
  if(b=='Gen'):
    b = 'def2tzvp'
  for l in lines:
    if('Number of atoms') in l:
      Nat= int(l.split().pop())
    if('Charge ') in l:
      charge= int(l.split().pop())
    if('Multiplicity ') in l:
     spin= int(l.split().pop())-1 
    if('Number of basis functions') in l:
      NAO = int(l.split().pop())
    if('Number of independent functions') in l:
      NMO = int(l.split().pop())
      break 
  print('Your file has basis ',b,' with ',Nat,' atoms and ',NAO,' ',NMO,' basis functions')
  print('Charge ',charge,' spin ',spin)
  print('Basis ',b)

  # Read lists of atom numbers and cartesin coordinates
  iats=[]
  cart=[]
  N = 0 
  r0= 0 
  rs= 0 
  for l in lines:
    if(len(iats)>=Nat):
      r0= 0 
    if(len(cart)>=N):
      rs= 0 
    if(r0>0):
      for x in l.split():
       if(len(iats)<Nat):
         iats.append(int(x))
    if(rs>0):
      for x in l.split():
       if(len(cart)<N):
         cart.append(float(x))
    if('Atomic numbers' in l):
      r0= 1 
    if('Current cartesian coordinates' in l):
      rs= 1 
      N =int(l.split().pop())

  # Repackage these into a PySCF molecule
  geom=''
  ind = -1 
  for iat in range(Nat):
    geom+= ' %4s ' %(names[iats[iat]])
    for i in range(3):
      ind = ind+1
      geom+=' %12.6f ' %(cart[ind])
    geom+='\n'

  #print('Your geometry is:\n ',geom)
  m = gto.Mole(atom=geom,charge=charge,spin=spin,basis=b)
  m.unit='B' # Gaussian uses Bohr units for geometries 
  #m.cart=True # I hate this 
  m.build() 
  NAO = m.nao 
  labs=m.ao_labels()

  # Read the basis functions in Gaussian order
  # PySCF reorders them as 
  # (1) atoms, (2) angular momentum, (3) shells, (4) spherical harmonics 
  ipy=[]
  for i in range(NAO):
    ipy.append(i)
  for i in range(NAO):
    if('dxy' in labs[i]): # Swap d subshells 
      ipy[i  ] = i+2
      ipy[i+1] = i+3
      ipy[i+2] = i+1
      ipy[i+3] = i+4
      ipy[i+4] = i+0
    if('f-3' in labs[i]): # Swap f subshells 
      ipy[i  ] = i+3
      ipy[i+1] = i+4
      ipy[i+2] = i+2
      ipy[i+3] = i+5
      ipy[i+4] = i+1
      ipy[i+5] = i+6
      ipy[i+6] = i+0
    if('g-4' in labs[i]): # Swap g subshells 
      ipy[i  ] = i+4
      ipy[i+1] = i+5
      ipy[i+2] = i+3
      ipy[i+3] = i+6
      ipy[i+4] = i+2
      ipy[i+5] = i+7
      ipy[i+6] = i+1
      ipy[i+7] = i+8
      ipy[i+8] = i+0

  # Read lists of total and spin density matrices 
  pdm0 = [] 
  pdms = [] 
  N = 0 
  r0= 0 
  rs= 0 
  for l in lines:
    if(len(pdm0)>=N):
      r0= 0 
    if(len(pdms)>=N):
      rs= 0 
    if(r0>0):
      for x in l.split():
       if(len(pdm0)<N):
         pdm0.append(float(x))
    if(rs>0):
      for x in l.split():
       if(len(pdms)<N):
         pdms.append(float(x))
    if('Total SCF Density' in l):
      r0= 1 
      N =int(l.split().pop())
    if('Spin SCF Density' in l):
      rs= 1 
  #print('You read in total 1PDM \n',pdm0,'\n and spin 1PDM\n',pdms)

  # Repackage these into a PySCF density matrix 
  P=numpy.zeros((2,NAO,NAO))
  ind = -1 
  for i in range(NAO):
    for j in range(i+1):
      ind = ind + 1 
      sp = 0 
      if(len(pdms)>0):
        sp = pdms[ind]
      v0 =  (pdm0[ind]+sp)/2
      v1 =  (pdm0[ind]-sp)/2
      P[0,ipy[i],ipy[j]] = v0
      P[0,ipy[j],ipy[i]] = v0
      P[1,ipy[i],ipy[j]] = v1
      P[1,ipy[j],ipy[i]] = v1

  # Read lists of alpha and beta orbital coefficients
  coefa= [] 
  coefb= [] 
  N = 0 
  ra= 0 
  rb= 0 
  for l in lines:
    if(len(coefa)>=N):
      ra= 0 
    if(len(coefb)>=N):
      rb= 0 
    if(ra>0):
      for x in l.split():
       if(len(coefa)<N):
         coefa.append(float(x))
    if(rb>0):
      for x in l.split():
       if(len(coefb)<N):
         coefb.append(float(x))
    if('Alpha MO coefficie' in l):
      ra= 1 
      N =int(l.split().pop())
    if('Beta MO coefficie' in l):
      rb= 1 
      N =int(l.split().pop())

  # Repackage these into a PySCF MO coefficient list [spin,ao,mo] 
  print('Coef len: ',len(coefa),len(coefb))
  mo_coeff=numpy.zeros((2,NAO,NMO))
  ind = -1 
  for imo in range(NMO):
    for iao in range(NAO):
      ind = ind + 1 
      v0 = coefa[ind]
      v1 = v0
      if(len(coefb)>0):
        v1 = coefb[ind]
      mo_coeff[0,ipy[iao],imo] = v0
      mo_coeff[1,ipy[iao],imo] = v1

  # Read lists of orbital energies
  aorb = [] 
  borb = [] 
  N = 0 
  ra= 0 
  rb= 0 
  for l in lines:
    if(len(aorb)>=N):
      ra= 0 
    if(len(borb)>=N):
      rb= 0 
    if(ra>0):
      for x in l.split():
       if(len(aorb)<N):
         aorb.append(float(x))
    if(rb>0):
      for x in l.split():
       if(len(borb)<N):
         borb.append(float(x))
    if('Alpha Orbital Energies' in l):
      ra= 1 
      N =int(l.split().pop())
    if('Beta Orbital Energies' in l):
      rb= 1 
  if(len(borb)<1):
    borb = aorb 
  print('Orbital array lengths ',len(aorb),len(borb))
  return(m,P,mo_coeff,aorb,borb)
  

if __name__=='__main__':
    tehFile=sys.argv[1]
    if(os.path.isfile(tehFile)):

        # Get the monomor alpha HOMO as phi 
        mmon,Pmon,mo_coeff_mon,aorbmon,borbmon=readChk('roCr.fchk')
        (NAmon,NBmon)=mmon.nelec
        natmon = mmon.natm
        phi=mo_coeff_mon[0,:,NAmon-6:NAmon].T

        # Read this molecule 
        m,P,mo_coeff,aorb,borb = readChk(tehFile)
        Na,Nb=m.nelec
        NAO = m.nao
        nat = m.natm
        nmon = int(nat/natmon)
        mo_occ=numpy.zeros((2,NAO))
        mo_occ[0,:Na]=1
        mo_occ[1,:Nb]=1
        print('This molecule contains ',nmon,' monomers')

        S = m.intor_symmetric('int1e_ovlp')
        Sm = numpy.linalg.pinv(S)
        (vals,vecs) = linalg.eigh(S)
        Smhalf0 = numpy.zeros((NAO,NAO))
        for i in range(NAO):
          if(vals[i]>0.00000001):
            Smhalf0[i,i] = ((vals[i]).real)**(-0.5)
        Smhalf = numpy.dot(vecs,numpy.dot(Smhalf0,vecs.T))
        test = numpy.dot(Smhalf,numpy.dot(S,Smhalf))-numpy.identity(NAO)
        print('TEST: ',numpy.sum(test*test))

        # Set up frag AOs for each monomer 
        fragstarts=[]
        fragorbs=[]
        funcsets=[]
        fragid0 = 0 
        nphi = phi.shape[0]
        for imon in range(nmon):
          fthis = [x+fragid0 for x in range(nphi)]
          funcsets.append(fthis)
          fragid0 = fragid0 + nphi 
          fragstarts.append(imon*natmon)
          fragorbs.append(phi)
        fragments = [fragstarts,fragorbs,fullbasis]
        # Pairs of adjacent fragments 
        funcsets2 = [] 
        for i in range(nmon-1):   
         ff = funcsets[i]
         for f in funcsets[i+1]: 
           ff.append(f)
         funcsets2.append(ff)

        # Build the UKS object from the projected MOs and density
        md=pdft.UKS(m,xc='hf,',phyb=[0],paos='FragAOs')
        md.fragments=fragments
        md.allc=3
        md.addMP2=False 
        md.lhlam=1
        mp2lam = 0.6 
        md.mp2lam = mp2lam 
        md.mo_occ=mo_occ
        build_proj(md)

        # Do a single-shot Fock build to generate large-basis MOs, MO energies, and HF energy 
        mo_energy = numpy.zeros((2,NAO))
        mo_coeff = numpy.zeros((2,NAO,NAO))
        h0=md.get_hcore()
        Jmat2 = md.get_j(dm=P)
        Jmat = Jmat2[0] + Jmat2[1] 
        Eother = numpy.einsum('sij,ij->',P,h0) + numpy.einsum('sij,ji->',P,Jmat)/2. +m.get_enuc()
        K=md.get_k(dm=P)
        EX=-0.5*numpy.einsum('sij,sij->',P,K)
        EHF = Eother + EX 
        for ispin in range(2):
          f = h0+Jmat-K[ispin]
          ft=numpy.dot(Smhalf,numpy.dot(f,Smhalf))
          (vals,vecs) = linalg.eigh(ft)
          mo_energy[ispin]=vals 
          mo_coeff[ispin] = numpy.dot(Smhalf,vecs) # should be indexed AO,MO
        md.mo_energy = mo_energy 
        md.mo_coeff = mo_coeff 
        print('Shapes: \nMO ',md.mo_coeff.shape,' occ ',md.mo_occ.shape,' energy ',md.mo_energy.shape)

        print('Normal termination')
        print('SCF Done: E = %12.6f Hartree HF '%(EHF))

        EC1,EC2,EC3,EC4,EX,EXSL=euci(md,hl=0,Pin=P)
        EC5,ECMP25,EX5,EC52,ECMP252,ECMP253=euci5(md,Pin=P,hl=2,funcsets=funcsets2)
        print('SCF Done: E = %12.6f Hartree PCI1-noit '%(EHF+EC1))
        print('SCF Done: E = %12.6f Hartree PCI1 '%(EHF+EC5))
        print('SCF Done: E = %12.6f Hartree PCI2 '%(EHF+EC52))
      
        # Regularized MP2 correlation 
        hf_mo_energy = numpy.copy(md.mo_energy)
        for i in range(Na,NAO):
          if((hf_mo_energy[0,i]-hf_mo_energy[0,Na-1])<mp2lam):
            hf_mo_energy[0,i] = hf_mo_energy[0,Na-1] +mp2lam 
        for i in range(Nb,NAO):
          if((hf_mo_energy[1,i]-hf_mo_energy[1,Nb-1])<mp2lam):
            hf_mo_energy[1,i] = hf_mo_energy[1,Nb-1] +mp2lam 
      
        mp = md.MP2()
        teheris = mp.ao2mo(mo_coeff=md.mo_coeff)
        mp.mo_energy = hf_mo_energy
        ECregMP2= mp.init_amps(mo_energy=hf_mo_energy, mo_coeff=md.mo_coeff,eris=teheris,with_t2=False)[0]
        EMP2 = ECregMP2 + EHF 

        print('SCF Done: E = %12.6f Hartree HF-rMP2 '%(EMP2))
        print('SCF Done: E = %12.6f Hartree PCI1-rMP2 '%(EMP2+EC5-ECMP25))
        print('SCF Done: E = %12.6f Hartree PCI2-rMP2 '%(EMP2+EC52-ECMP253))
        print('SCF Done: E = %12.6f Hartree PCI2-rMP2b '%(EMP2+EC52-ECMP253+ECMP252))


