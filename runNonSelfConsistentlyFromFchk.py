from pyscf import scf,gto,dft,cc
import numpy ,scipy, sys, os.path 
import pdft
from scipy import linalg
from pdft.projwork import euci , eci , epzlh, build_proj, P_1to2
import dftd3.pyscf as dftd3

# Compute PiFCI+HF and PiFCI+DFT using SCF orbitals from a small basis Gaussian formatted checkpoint file
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
        ms,Ps,mo_coeff_s,aorb,borb = readChk(tehFile)
        naos = ms.nao

        # Update mo_energy as 25% hybrid
        mms=scf.UHF(ms)
        Ss = mms.get_ovlp() 
        (vals,vecs) = linalg.eigh(Ss)
        Ssmhalf= numpy.zeros((naos,naos))
        for i in range(naos):
          if(vals[i]>0.00000001):
           Ssmhalf[i,i] = ((vals[i]).real)**(-0.5)
        Ssm = numpy.dot(vecs,numpy.dot(Ssmhalf,numpy.transpose(vecs)))
        h0s=mms.get_hcore() 
        Js2=mms.get_j(dm=Ps)
        Js = Js2[0]+Js2[1]
        Ks=mms.get_k(dm=Ps)
        fs=numpy.zeros_like(Ps)
        fs[0] = h0s+Js-Ks[0]
        fs[1] = h0s+Js-Ks[1]
        aorb2=numpy.zeros((naos))
        borb2=numpy.zeros((naos))
        for i in range(naos):
            aorb2[i] = numpy.dot(mo_coeff_s[0,:,i],numpy.dot(fs[0],mo_coeff_s[0,:,i]))  # indexed as spin, AO, MO
            borb2[i] = numpy.dot(mo_coeff_s[1,:,i],numpy.dot(fs[1],mo_coeff_s[1,:,i]))  # indexed as spin, AO, MO
        #fst=numpy.zeros_like(Ps)
        #fst[0]=numpy.dot(Ssm,numpy.dot(fs[0],Ssm))
        #fst[1]=numpy.dot(Ssm,numpy.dot(fs[1],Ssm))
        #(aorb3,dum) = linalg.eigh(fst[0])
        #(borb3,dum) = linalg.eigh(fst[1])
        #print('Input orb energies and HF orb energies \n',aorb,'\n',aorb2,'\n',aorb3,'\n',borb,'\n',borb2,'\n',borb3)
        aorb=numpy.array(aorb)
        borb=numpy.array(borb)
        aorb = 0.75*aorb + 0.25*aorb2
        borb = 0.75*borb + 0.25*borb2
        mo_energy = numpy.array((aorb,borb))
        

        # Build a full molecule in the full basis set 
        m = gto.Mole(atom=ms.atom,charge=ms.charge,spin=ms.spin,basis=fullbasis,unit=ms.unit)
        m.build()
        Na,Nb=m.nelec
        NAO = m.nao
        print('Your calculation is projected into a basis size ',NAO)

        # Translate the small molecule MOs and density matrix into the full basis set 
        S = m.intor_symmetric('int1e_ovlp')
        Sm = numpy.linalg.pinv(S)
        SX = gto.intor_cross('int1e_ovlp',m,ms)
        P=numpy.zeros((2,NAO,NAO)) # indexed as spin, AO, MO
        mo_coeff=numpy.zeros((2,NAO,naos)) # indexed as spin, AO, MO
        mo_coeff[0]=numpy.dot(numpy.dot(Sm,SX),mo_coeff_s[0])
        mo_coeff[1]=numpy.dot(numpy.dot(Sm,SX),mo_coeff_s[1])
        for ispin in range(2):
            SMO = numpy.dot(numpy.transpose(mo_coeff[ispin]),numpy.dot(S,mo_coeff[ispin]))
            (vals,vecs)=linalg.eigh(SMO)
            vals[vals<0.00000001]=0.000001
            ss =  numpy.diag(1 / numpy.sqrt(vals))
            tt = numpy.dot(vecs,numpy.dot(ss,vecs.T))
            mo_coeff[ispin]=mo_coeff[ispin] @ tt
            test = numpy.identity(naos) - numpy.dot(numpy.transpose(mo_coeff[ispin]),numpy.dot(S,mo_coeff[ispin]))
            print('TEST ',numpy.trace(numpy.dot(test,test)))
        P[0]=numpy.einsum('mi,ni->mn',mo_coeff[0,:,:Na],mo_coeff[0,:,:Na])
        P[1]=numpy.einsum('mi,ni->mn',mo_coeff[1,:,:Nb],mo_coeff[1,:,:Nb])
        print('Nelec Test ',numpy.einsum('sij,ij->s',P,S))



        mo_occ=numpy.zeros((2,NAO))
        mo_occ[0,:Na]=1
        mo_occ[1,:Nb]=1

        md=pdft.UKS(m,xc='pbe,pbe',phyb=[1],paos='NewVAOs',allc=0)
        md.mo_coeff=mo_coeff
        md.mo_occ=mo_occ
        md.mo_energy=mo_energy 
        build_proj(md)
        
        EC1,EC2,EC3,EXU,EXSLU=euci(md,hl=0,Pin=P)
        print('EUCI ',EC1)
        md.xc='b88,lyp'
        Eother,EX,EXP,EXSL,ECSL,EXSLP = epzlh(md,P,allc=1)
        md.xc='lda,vwn5'
        Eother,EX,EXP,EXSLL,ECSLL,EXSLPL = epzlh(md,P,allc=1)
        EXSLG=EXSL-EXSLL
        ECSLG=ECSL-ECSLL
        EXSLPG=EXSLP-EXSLPL

        # HF and PiFCI+HF 
        EHF = Eother  + EX 
        print('SCF Done: E = %12.6f Hartree         HF '%(EHF))
        print('SCF Done: E = %12.6f Hartree   PiFCI+HF '%(EHF+EC1))

        # Final parameter set 
        s8=0.0
        a=0.8 
        b=0.0 
        c=0.8 
        a1=0.2 
        a2=5.0
        d3 = dftd3.DFTD3Dispersion(m, param={'s6':1.0,'s8':s8,'a1':a1,'a2':a2}, version="d3bj")
        v=d3.kernel()[0]
        E0 = Eother+EX+ECSLL+ EC1  + v 
        E1 = E0 + (1-a)*(EXSLPL+b*EXSLPG-EXP) + c*ECSLG
        print('SCF Done: E = %12.6f Hartree   a %.1f b %.1f c %.1f a1 %.1f a2 %.1f '%(E1,a,b,c,a1,a2))

