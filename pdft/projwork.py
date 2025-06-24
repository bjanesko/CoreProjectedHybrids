#!/usr/bin/env python
# Work routines for projected DFT 
import time
import numpy, sys 
from scipy import linalg
from scipy.special import erf 
from pyscf import gto ,ao2mo, mcscf , lib , scf, cc 

def get_d(m):
  DAOs = [] 

  # Find the transition metal atoms 
  tms=[]
  for iat in range(m.natm):
    iq = m.atom_charge(iat)
    if( (iq>=21 and iq<=30) or (iq>=39 and iq<48) ):
      tms.append(iat)
  ntm = len(tms)
  print(' We have ',ntm,' transition metal atoms')

  # Each transition metal has five sets of d orbitals 
  labs = m.ao_labels()
  for iat in tms:
     for ityp in ('dxy','dyz','dz^2','dxz','dx2-y2'):
        vals=[]
        for iao in range(m.nao):
          icen = int(labs[iao].split()[0])
          #print('Comparing label ',labs[iao],' to type ',ityp)
          if((icen==iat) and (ityp in labs[iao])):
            DAOs.append(iao)
            vals.append(iao)
  DAOs = [*set(DAOs)] # Remove duplicates 
  DAOs = [DAOs] # one set  
  return(DAOs)

def assign_cores(m):
  # Maximum valence and minimum core exponent for atoms H-Ar, STO-2G basis set 
  MaxValence=[0, 0, 0.246, 0.508, 0.864, 1.136, 1.461, 1.945, 2.498, 3.187, 0.594, 0.561, 0.561, 0.594, 0.7, 0.815, 0.855, 1.053]
  MinCore=[0, 0, 1.097, 2.053, 3.321, 4.874, 6.745, 8.896, 11.344, 14.09, 1.18, 1.482, 1.852, 2.273, 2.747, 3.267, 3.819, 4.427]

  # Compute each AO's radial extent in terms of kinetic energy integrals 
  CoreAOs = [] 
  T=m.intor_symmetric('int1e_kin')
  labs = m.ao_labels()
  for iao in range(m.nao):
    icen = int(labs[iao].split()[0])
    iat = m.atom_charge(icen)
    acut = MinCore[iat-1]
    Tcut = acut*1.5 * 1.0
    Tval = T[iao,iao]
    #print("AO ",iao," ",labs[iao]," center ",icen," atom charge ",iat," T ",Tval," Tcut ",Tcut)
    if(Tval>Tcut and iat>2 and ('s ' in labs[iao] )):
      CoreAOs.append(iao)
      #print(labs[iao])

  CoreAOs = [*set(CoreAOs)] # Remove duplicates 
  return(CoreAOs)

def mo_proj(ks):
    ''' Build a single projection operator from an input list of orthonormal MOs 
    Args:
        ks : an instance of :class:`RKS`
    '''  
    orbs = ks.paos[0]
    m = ks.mol
    N = m.nao
    S = ks.get_ovlp() 
    (NAO,NC) = orbs.shape
    #print('N ',N)
    #print('Orbs: ',orbs.shape)
    if(NAO != N):
      die('Failure in mo_proj with ',N,' and ',orbs.shape)
    Q =numpy.zeros((N,N))
    for i in range(NC):
      v = orbs[:,i]
      Q = Q + numpy.outer(v,v)
    ks.QS=[numpy.einsum('ik,kj->ij',Q,S)]
    ks.SQ=[numpy.einsum('ik,kj->ij',S,Q)]
    # DEBUG TEST 
    test = numpy.dot(Q,numpy.dot(S,Q))-Q
    print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q))
   
gsspins=[1,0, 
1,0,  1,2,3,2,1,0, 
1,0,  1,2,3,2,1,0,  
1,0, 1,2,3,4,5,4,3,2,1,0, 1,2,3,2,1,0,   
1,0, 1,2,3,4,5,4,3,2,1,0, 1,2,3,2,1,0,
1,0, 
     1,2,3,4,5,6,7,6,5,4,3,2,1,0,
     1,2,3,4,5,4,3,2,1,0, 1,2,3,2,1,0,
1,0, 
     1,2,3,4,5,6,7,6,5,4,3,2,1,0,
     1,2,3,4,5,4,3,2,1,0, 1,2,3,2,1,0,
] 
valencetag=['1s','1s',
'2s','2s',  '2p','2p','2p','2p','2p','2p',
'3s','3s',  '3p','3p','3p','3p','3p','3p',
'4s','4s',  '3d','3d','3d','3d','3d','3d','3d','3d','3d','3d',   '4p','4p','4p','4p','4p','4p',
'5s','5s',  '4d','4d','4d','4d','4d','4d','4d','4d','4d','4d',   '5p','5p','5p','5p','5p','5p'] 

def build_mbproj_2(ks):
  # June 2024, this version builds a set of orthogonalized projected atomic
  # orbitals (opAOs) on each atom, from the STO-3G minimal basis, then builds projector
  # Q from orthogonalizing all of those. The opAOs are saved in ks.proj for use
  # in euci. This sets up pAOs in current basis from projection onto minimal
  # basis, then calls pao_proj 
  # Note this requires PROJECTING the minimal basis onto the current AO basis, thus it requires Sinv 
  # It also requires a tag for the 'valence' AOs of each atom 

  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  m = ks.mol
  thepaos= [] 
  theVeepAOs= [] 
  print('Projected minimal AOs in build_mbproj_2')
  mmin =gto.Mole(atom=m.atom,basis='sto-3g',ecp='sto-3g',charge=m.charge,spin=m.spin)
  mmin.unit=m.unit
  mmin.build()
  SX = gto.intor_cross('int1e_ovlp',m,mmin)
  labs = mmin.ao_labels()
  #print('Searching in labels \n',labs)
  for iat in range(m.natm):
    thisatpaos=[] 
    attype = m.atom_charge(iat)
    tag = valencetag[attype]
    keptaos=[]
    for iao in range(mmin.nao):
      icen = int(labs[iao].split()[0])
      #if(tag in labs[iao]):
        #if(icen == iat): 
        #  print(iat,': ',labs[iao])
        #  keptaos.append(iao)
        #  thisatpaos.append(numpy.dot(Sm,SX[:,iao]))
      if(icen == iat): 
        keep=1
        if(attype>2 and ('1s' in labs[iao])):
          keep=0
        if(attype>10 and ('2s' in labs[iao] or '2p' in labs[iao])):
          keep=0
        if(attype>18 and ('3s' in labs[iao] or '3p' in labs[iao])):
          keep=0
        if(keep):
          print(iat,': ',labs[iao])
          thisatpaos.append(numpy.dot(Sm,SX[:,iao]))
    thisatpaos2=numpy.transpose(numpy.array(thisatpaos))
    thepaos.append(thisatpaos2)

    # September 2024, use this atom's minimal basis twoelecints as the pAO
    # twoelecints 
    attype = m.atom_charge(iat)-1 
    matmin=gto.Mole(atom=m.atom_symbol(iat),basis='sto-3g',ecp='sto-3g',charge=0,spin=gsspins[attype])
    matmin.build() 
    keptataos=[]
    atlabs = matmin.ao_labels()
    for iao in range(matmin.nao):
        keep=1
        if(attype>=2 and ('1s' in atlabs[iao])):
          keep=0
        if(attype>=10 and ('2s' in atlabs[iao] or '2p' in atlabs[iao])):
          keep=0
        if(attype>=18 and ('3s' in atlabs[iao] or '3p' in atlabs[iao])):
          keep=0
        if(keep):
          print(atlabs[iao])
          keptataos.append(iao)
    #  if(tag in atlabs[iao]):
    #    keptataos.append(iao)
    eri0=matmin.intor('int2e')
    eri0=eri0.reshape(matmin.nao,matmin.nao,matmin.nao,matmin.nao)
    npao = len(keptataos)
    VeepAO=numpy.zeros((npao,npao,npao,npao))
    print('TEST SIZES ',npao,matmin.nao,VeepAO.shape,eri0.shape)
    for i in range(npao):
      for j in range(npao):
        for k in range(npao):
          for l in range(npao):
            VeepAO[i,j,k,l]=eri0[keptataos[i],keptataos[j],keptataos[k],keptataos[l]]
    theVeepAOs.append(VeepAO)

  ks.paos = thepaos
  ks.VeepAOs= theVeepAOs
  pao_proj(ks) 
  print('LOOK HERE IS KS.QS ',len(ks.QS),len(ks.SQ))
  #print('LOOK HERE IS KS.QS[0] ',ks.QS[0])
  return
       
def build_mbproj_3(ks):
  # August 2024, this version builds a set of orthogonalized projected atomic
  # orbitals (opAOs) on each atom, from the 3-21g basis, then builds projector
  # Q from orthogonalizing all of those. The opAOs are saved in ks.proj for use
  # in euci. This sets up pAOs in current basis from projection onto minimal
  # basis, then calls pao_proj 
  # Note this requires PROJECTING the minimal basis onto the current AO basis, thus it requires Sinv 
  # It also requires a tag for the 'valence' AOs of each atom 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  m = ks.mol
  thepaos= [] 
  theVeepAOs= [] 
  print('Projected AOs in build_mbproj_3')
  mmin =gto.Mole(atom=m.atom,basis='3-21g',ecp='3-21g',charge=m.charge,spin=m.spin)
  mmin.unit=m.unit
  mmin.build()
  SX = gto.intor_cross('int1e_ovlp',m,mmin)
  labs = mmin.ao_labels()
  for iat in range(m.natm):
    thisatpaos=[] 
    attype = m.atom_charge(iat)
    for iao in range(mmin.nao):
      icen = int(labs[iao].split()[0])
      if(icen == iat): 
        keep=1
        if(attype>2 and ('1s' in labs[iao])):
          keep=0
        if(attype>10 and ('2s' in labs[iao] or '2p' in labs[iao])):
          keep=0
        if(attype>18 and ('3s' in labs[iao] or '3p' in labs[iao])):
          keep=0
        if(keep):
          print(iat,': ',labs[iao])
          thisatpaos.append(numpy.dot(Sm,SX[:,iao]))
    thisatpaos2=numpy.transpose(numpy.array(thisatpaos))
    thepaos.append(thisatpaos2)

    # December 2024 use this atom's 3-21g basis twoelecints as the pAO
    # twoelecints to accelerate 
    attype = m.atom_charge(iat)-1 
    print('Getting twoelecints for atom ',attype,gsspins[attype])
    matmin=gto.Mole(atom=m.atom_symbol(iat),basis='3-21g',ecp='3-21g',charge=0,spin=gsspins[attype])
    matmin.build() 
    keptataos=[]
    atlabs = matmin.ao_labels()
    for iao in range(matmin.nao):
        keep=1
        if(attype>=2 and ('1s' in atlabs[iao])):
          keep=0
        if(attype>=10 and ('2s' in atlabs[iao] or '2p' in atlabs[iao])):
          keep=0
        if(attype>=18 and ('3s' in atlabs[iao] or '3p' in atlabs[iao])):
          keep=0
        if(keep):
          print(atlabs[iao])
          keptataos.append(iao)
      #if(tag in atlabs[iao]):
      #  keptataos.append(iao)
    eri0=matmin.intor('int2e')
    eri0=eri0.reshape(matmin.nao,matmin.nao,matmin.nao,matmin.nao)
    npao = len(keptataos)
    VeepAO=numpy.zeros((npao,npao,npao,npao))
    print('TEST SIZES ',npao,matmin.nao,VeepAO.shape,eri0.shape)
    for i in range(npao):
      for j in range(npao):
        for k in range(npao):
          for l in range(npao):
            VeepAO[i,j,k,l]=eri0[keptataos[i],keptataos[j],keptataos[k],keptataos[l]]
    theVeepAOs.append(VeepAO)

  ks.paos = thepaos
  ks.VeepAOs= theVeepAOs
  pao_proj(ks) 
  print('LOOK HERE IS KS.QS ',len(ks.QS),len(ks.SQ))
  return
       
  

def build_mbproj(ks,daos=False,faos=False,vaos=False,dum=False):
    ''' Build projection operators from an existing AO basis 
    This version builds two projection operators: one for second-period
    atoms Li-Ne, one for third-period atoms Na-Ar. 
    With daos=True, we get one projection operator for transition metal atoms 
    With vaos=True, we get one projection operator containing minimal basis AOs in this basis 
    Args:
        ks : an instance of :class:`RKS`
    June 2024, modify paos to include a list of matrices of pAOs from each shell 
    '''  
    if(ks.QS is None):
      m = ks.mol
      S = ks.get_ovlp() 
      Sm = linalg.inv(S)
      N=(S.shape)[0]
      Q2 = numpy.zeros((N,N))
      Q3 = numpy.zeros((N,N))
      ks.QS=[]
      ks.SQ=[] 

      # Prepare dummy molecule with minimal basis set 
      md = m.copy()
      md.basis='sto3g'   
      md.build() 
      SM = md.intor_symmetric('int1e_ovlp')
      
      # Build cross-overlap matrix between current and minimal core 
      SX = gto.intor_cross('int1e_ovlp',m,md)
      
      # Find minimal basis AOs used for this projection 
      coreaos=[]
      coreassign=[]
      coreatind=[]
      labs = md.ao_labels()
      for iao in range(md.nao):
        icen = int(labs[iao].split()[0])
        iat = md.atom_charge(icen) + md.atom_nelec_core(icen)
        if(daos):
          if( ((iat>20 and iat<31) or (iat>38 and iat<49)) and   ('d' in labs[iao]) ):
            coreaos.append(iao)
            coreassign.append(2)
            print(labs[iao],coreassign[-1])
        elif(faos):
          if( ((iat>56 and iat<72) or (iat>88 and iat<104)) and   ('f' in labs[iao]) ):
            coreaos.append(iao)
            coreassign.append(2)
            print('Proj AO',labs[iao],coreassign[-1])
        elif(vaos):
          if( ((iat<3) and   ('1s' in labs[iao])) 
or ((iat>2 and iat<5) and  ('2s' in labs[iao])) 
or ((iat>4 and iat<11) and  ('2p' in labs[iao])) 
or ((iat>10 and iat<13) and  ('3s' in labs[iao])) 
or ((iat>12 and iat<19) and  ('3p' in labs[iao])) 
or ((iat>18 and iat<21) and  ('4s' in labs[iao])) 
or ((iat>20 and iat<31) and  ('3d' in labs[iao])) 
or ((iat>20 and iat<31) and  ('4s' in labs[iao])) 
or ((iat>30 and iat<37) and  ('4p' in labs[iao])) 
or ((iat>36 and iat<39) and  ('5s' in labs[iao])) 
or ((iat>48 and iat<55) and  ('5p' in labs[iao])) 
):
            coreaos.append(iao)
            coreatind.append(icen) # BGJ label each AO with the center index 
            coreassign.append(2) 
            print('Proj AO',labs[iao],coreassign[-1])
        elif(dum):
          #print('+++ ',iao,labs[iao])
          if(iat>2 and   (('py' in labs[iao]) or ('pz' in labs[iao]) or ('px' in labs[iao])) ):
            coreaos.append(iao)
            coreassign.append(2)
            print(labs[iao],coreassign[-1])
        else:
          #print('YOU ARE THERE WITH IAT ',iat,' daos ',daos,' faos ',faos)
          if(iat>2 and   (' 1s' in labs[iao]) ):
             coreaos.append(iao)
             if(iat>10):
               coreassign.append(3)
             else: 
               coreassign.append(2)
             print(labs[iao],coreassign[-1])
          if(iat>10 and   (' 2s' in labs[iao] or ' 2p' in labs[iao]) ):
             coreaos.append(iao)
             coreassign.append(3)
             print(labs[iao],coreassign[-1])

      # Project kept core AOs into this basis
      NC = len(coreaos)
      if(NC>0):
        SC = numpy.zeros((NC,NC))
        SXC = numpy.zeros((N,NC))
        for ic in range(NC):
          SXC[:,ic] = SX[:,coreaos[ic]]
          for jc in range(NC):
             SC[ic,jc] = SM[coreaos[ic],coreaos[jc]]

        # New June 2024 put projected core AOs from each atom into paos 
        # paos will be a list of Nsite rectangular matrices
        # Does not work! 
        if(vaos):
          projats = numpy.unique(coreatind)
          paosthis = [] 
          for iat in range(len(projats)):
            thisatpaos=[]
            for ic in range(NC):
              if(coreatind[ic] == projats[iat]):
                thisatpaos.append(SXC[:,ic])
          thisatpaos2 = numpy.transpose(numpy.array(thisatpaos))
          print('TEST SHAPE ',thisatpaos2.shape)
          paosthis.append(thisatpaos2)
          ks.paos = paosthis

        # Prepare two SC^(-1) operators, one for 2nd period elements and one for 3rd period elements 
        # Assume that the ith orthogonalized core AO equals the ith core AO 
        #SCm= linalg.inv(SC)
        (vals,vecs) = linalg.eigh(SC)
        SCm2 = numpy.zeros((NC,NC))
        SCm3 = numpy.zeros((NC,NC))
        SCmhalf = numpy.zeros((NC,NC))
        for i in range(NC):
          if(vals[i]>0.00000001):
            SCmhalf[i,i] = ((vals[i]).real)**(-0.5)
          else:
            print('Eliminating overlap eigenvalue ',i,vals[i])
        QCbig = numpy.dot(vecs,numpy.dot(SCmhalf,numpy.transpose(vecs)))
        for ic in range(NC):
          v = QCbig[ic]
          Qset = numpy.outer(v,v)
          if(coreassign[ic] == 3):
            SCm3 = SCm3 + Qset 
          else:
            SCm2 = SCm2 + Qset 
      
        # Core AO projection operators in current basis set 
        #Q = numpy.einsum('ia,ab,bc,dc,dj->ij',Sm,SXC,SCm,SXC,Sm)
        SmSXC= numpy.dot(Sm,SXC)
        t2 = numpy.dot(SmSXC,SCm2)
        Q2 = numpy.dot(t2,numpy.transpose(SmSXC))
        t2 = numpy.dot(SmSXC,SCm3)
        Q3 = numpy.dot(t2,numpy.transpose(SmSXC))
        ks.QS=[numpy.einsum('ik,kj->ij',Q2,S),numpy.einsum('ik,kj->ij',Q3,S)]
        ks.SQ=[numpy.einsum('ik,kj->ij',S,Q2),numpy.einsum('ik,kj->ij',S,Q3)]

      # DEBUG TEST 
      test = numpy.dot(Q2,numpy.dot(S,Q2))-Q2
      test+= numpy.dot(Q3,numpy.dot(S,Q3))-Q3
      test+= numpy.dot(Q2,numpy.dot(S,Q3))
      print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q2),numpy.einsum('ij,ji->',S,Q3))


def old_build_mbproj(ks):
    ''' Build the projection operators QS and SQ from existing basis core AOs 
    Args:
        ks : an instance of :class:`RKS`
    '''  
    if(ks.QS is None):
      m = ks.mol
      S = ks.get_ovlp() 
      N=(S.shape)[0]
      Q    =numpy.zeros((N,N))
      ks.QS=numpy.zeros((N,N))
      ks.SQ=numpy.zeros((N,N))
      # Prepare dummy molecule with minimal basis set 
      md = m.copy()
      md.basis='uncccpcvtz'   
      md.build() 
      SM = md.intor_symmetric('int1e_ovlp')
      
      # Build cross-overlap matrix between current and minimal core 
      SX = gto.intor_cross('int1e_ovlp',m,md)
      #print('Cross- overlap \n',SX.shape)
      
      # Find core minimal basis AOs 
      coreaos=[]
      labs = md.ao_labels()
      for iao in range(md.nao):
        icen = int(labs[iao].split()[0])
        iat = md.atom_charge(icen)
        if(iat>2 and   (' 1s' in labs[iao]) ):
           coreaos.append(iao)
           #print(labs[iao])

      # Build core minimal basis 
      NC = len(coreaos)
      #print(' Keeping ',NC,' core minimal AOs') 
      SC = numpy.zeros((NC,NC))
      SXC = numpy.zeros((N,NC))
      #print('Transfer Shape: ',SXC.shape)
      for ic in range(NC):
        SXC[:,ic] = SX[:,coreaos[ic]]
        for jc in range(NC):
           SC[ic,jc] = SM[coreaos[ic],coreaos[jc]]
      #print('SC ',SC)
      #print('SXC ',SXC)
      SXCT = numpy.transpose(SXC)
      
      # Build and orthogonalize projected core minimal basis set 
      SPC = numpy.dot(SXCT,numpy.dot(S,SXC))
      #print('Projected core min basis: ',SPC.shape)
      #print('SPC ',SPC)
      (vals,vecs) = linalg.eigh(SPC)
      SPCmhalf = numpy.zeros((NC,NC))
      for i in range(NC):
        if(vals[i]>0.00000001):
          SPCmhalf[i,i] = ((vals[i]).real)**(-0.5)
        else:
           print('Eliminating overlap eigenvalue ',i,vals[i])
      QPC = numpy.dot(vecs,numpy.dot(SPCmhalf,numpy.transpose(vecs)))
      
      # Build projector 
      Q0 = numpy.zeros((NC,NC))
      for i in range(NC):
        v = QPC[i]
        Q0 = Q0 + numpy.outer(v,v)
      Q = numpy.dot(SXC,numpy.dot(Q0,SXCT))
      ks.QS=numpy.einsum('ik,kj->ij',Q,S)
      ks.SQ=numpy.einsum('ik,kj->ij',S,Q)

      # DEBUG TEST 
      test = numpy.dot(Q,numpy.dot(S,Q))-Q
      #print('Q \n',Q)
      print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q))

def build_proj(ks):
    ''' Build the projection operators QS and SQ from list of integers paos 
    Args:
        ks : an instance of :class:`RKS`
    '''
    if(ks.paos is None):
      return 
    if(ks.QS is None):
      if(isinstance(ks.paos,str)):
        if('NewCoreAOs' in ks.paos):
          build_mbproj(ks)
          return 
        if('NewDAOs' in ks.paos):
          build_mbproj(ks,daos=True)
          return 
        if('NewFAOs' in ks.paos):
          build_mbproj(ks,faos=True)
          return 
        if('NewVAOs' in ks.paos):
          #build_mbproj(ks,vaos=True)
          build_mbproj_2(ks)
          return 
        if('NewDZVAOs' in ks.paos):
          #build_mbproj(ks,vaos=True)
          build_mbproj_3(ks)
          return 
        if('Dum' in ks.paos):
          build_mbproj(ks,dum=True)
          return 
        aoss = [] 
        if('CoreAOs' in ks.paos):
          aoss = assign_cores(ks.mol)
        elif('DAOs' in ks.paos):
          aoss = get_d(ks.mol)
        elif('AllAOs' in ks.paos):
          aoss = [] 
          aoss2 = []
          for i in range(ks.mol.nao):
            aoss2.append(i) 
          aoss.append(aoss2)
      #elif(isinstance(ks.paos,list) and len(ks.paos[0].shape)==2 ):
      elif(isinstance(ks.paos,list) ): 
        pao_proj(ks) # New function May 2024, list of shells of AOs 
        return 
      elif(isinstance(ks.paos,list)):
        aoss = ks.paos 
      else:
        if(len(ks.paos[0].shape) == 2):
          mo_proj(ks)
          return 
        else:
          raise Exception('Not sure what paos is')

      # If we are still here, aoss contains a list of lists of AOs
      # to project onto. Orthogonalize and project. 
      print('AOs to project ',aoss)
      S = ks.get_ovlp() 
      N=(S.shape)[0]
      ks.QS=[]
      ks.SQ=[]

      # Do a symmetric orthogonalization, then assign orthogonalized vectors
      # to sets based on maximum overlap. We'll *assume* that the ith
      # orthogonalized AO equals the ith AO. 
      (vals,vecs) = linalg.eigh(S)
      Smhalf = numpy.zeros((N,N))
      for i in range(N):
        if(vals[i]>0.00000001):
          Smhalf[i,i] = ((vals[i]).real)**(-0.5)
        else:
           print('Eliminating overlap eigenvalue ',i,vals[i])
      Qbig = numpy.dot(vecs,numpy.dot(Smhalf,numpy.transpose(vecs)))
      for iset in range(len(aoss)):
        Qset = numpy.zeros((N,N))
        for i in aoss[iset]:
          v = Qbig[i]
          Qset = Qset + numpy.outer(v,v)
        ks.QS.append(numpy.einsum('ik,kj->ij',Qset,S))
        ks.SQ.append(numpy.einsum('ik,kj->ij',S,Qset))
        test = numpy.dot(Qset,numpy.dot(S,Qset))-Qset
        print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Qset))

######  May 2024 new functions for pDFT+UCI in projected atomic orbitals 
def makeOPAOs(SAO,pAOs,VeepAOs=None):
  opAOs=[]
  SopAOs=[]
  SAOopAOs=[]
  VeeopAOs=[]
  for ishell in range(len(pAOs)):
     pAO = pAOs[ishell]
     #print('Shell ',ishell,' pAOs \n',pAO)
     nproj=pAO.shape[1]
     #print('Shell ',ishell,' nproj ',nproj)
     opAO = numpy.zeros_like(pAO)
     #Sshell = numpy.einsum('mp,mn,nq->pq',pAO,SAO,pAO)
     Sshell = numpy.dot(numpy.transpose(pAO),numpy.dot(SAO,pAO))
     #print('Shell ',ishell,' pAO overlap \n',Sshell)
     (vals,vecs)=numpy.linalg.eigh(Sshell)
     print('PAO shell ',ishell,' overlap eigenvalues \n',vals)

     # Do sinular value decomposition inverse 
     for i in range(len(vals)):
      if(vals[i]>0.000001):
        vals[i]=vals[i]**(-0.5)
      else:
        vals[i]=0
      vecs[:,i]=vecs[:,i]*vals[i]
     opAO = numpy.dot(pAO,vecs)
     #opAO = numpy.einsum('mi,ij,j->mj',pAO,vecs,vals)
     #print('Shell ',ishell,' opAOs \n',opAO)
     opAOs.append(opAO) 
     #SopAOs.append(numpy.einsum('mi,mn,nj->ij',opAO,SAO,opAO))
     SopAOs.append(numpy.dot(numpy.transpose(opAO),numpy.dot(SAO,opAO)))
     #SAOopAOs.append(numpy.einsum('mi,mn->ni',opAO,SAO))
     SAOopAOs.append(numpy.dot(SAO,opAO))

     # Transform VeepAOs
     if(VeepAOs is not None):
       VeepAO = VeepAOs[ishell]
       temp1=numpy.einsum('pqrs,sl->pqrl',VeepAO,vecs)
       temp2=numpy.einsum('pqrl,rk->pqkl',temp1,vecs)
       temp1=numpy.einsum('pqkl,qj->pjkl',temp2,vecs)
       VeeopAO=numpy.einsum('pjkl,pi->ijkl',temp1,vecs)
       VeeopAOs.append(VeeopAO)
  return(opAOs,SopAOs,SAOopAOs,VeeopAOs)

def makeallOPAOs(SAO,opAOs):
  # Return the rectangular (ao,opao) matrix of all the block-orthogonalized
  # opAOs expressed in the AO basis, the square matrix of opAO-opAO overlaps,
  # and the rectangular matrix of AO-opAO overlaps. 
  nshell = len(opAOs)
  nao = opAOs[0].shape[0]
  ntot=0
  for ishell in range(nshell):
    ntot = ntot + opAOs[ishell].shape[1]
  allopAO=numpy.zeros((nao,ntot))
  j=0
  for ishell in range(nshell):
   for iao in range(opAOs[ishell].shape[1]):
     allopAO[:,j]=(opAOs[ishell])[:,iao]
     j=j+1
  #SAOallopAO = numpy.einsum('mi,mn->ni',allopAO,SAO)
  SAOallopAO = numpy.dot(SAO,allopAO)
  #SallopAO = numpy.einsum('mi,mn,nj->ij',allopAO,SAO,allopAO)
  SallopAO = numpy.dot(numpy.transpose(allopAO),numpy.dot(SAO,allopAO))
  return(allopAO,SallopAO,SAOallopAO)

def pao_proj(ks):
    ''' 
    Build a projection operator Q from an input list of blocks of nonorthogonal
    functions expanded in the current AO basis set
    Args:
        ks : an instance of :class:`RKS`
    '''  
    pAOs = ks.paos 
    print('Now in pao_proj with ',len(pAOs),' shells of pAOs ')
    m = ks.mol
    nao = m.nao
    S = ks.get_ovlp() 
    opAOs,SopAOs,SAOopAOs,VeeopAOs = makeOPAOs(S,pAOs)
    opAO,SopAO,SAOopAO = makeallOPAOs(S,opAOs)
    SopAOm = numpy.linalg.inv(SopAO)
    Sm = numpy.linalg.inv(S)
    #Q = numpy.einsum('mn,ni,ij,oj,op->mp',Sm,SAOopAO,SopAOm,SAOopAO,Sm)
    temp=numpy.dot(Sm,SAOopAO)
    Q = numpy.dot(numpy.dot(temp,SopAOm),numpy.transpose(temp))
    ks.QS=[numpy.dot(Q,S)]
    ks.SQ=[numpy.dot(S,Q)]
    # DEBUG TEST 
    test = numpy.dot(Q,numpy.dot(S,Q))-Q
    print('PAO_PROJ TEST: ',numpy.sum(test*test))

def V_1to2(V1,S12,Sm2):
  # General function to convert vector v from basis 1 to basis 2 
  # given their overlap S12 and basis 2 inverse 
  temp=numpy.dot(S12,Sm2)
  print('TEST ',S12.shape,Sm2.shape,temp.shape)
  ret = numpy.dot(V1,temp)
  return(ret)

def P_1to2(P1,S12,Sm2):
  # General function to convert density matrix P from basis 1 to basis 2 
  # given their overlap S12 and basis 2 inverse 
  #P2 = numpy.einsum('pr,mr,smn,nt,tq->spq',Sm2,S12,P1,S12,Sm2)
  temp=numpy.dot(S12,Sm2)
  P2a = numpy.dot(numpy.transpose(temp),numpy.dot(P1[0],temp))
  P2b = numpy.dot(numpy.transpose(temp),numpy.dot(P1[1],temp))
  P2 = numpy.asarray([P2a,P2b])
  return(P2)

def O1_1to2(O1,S12,Sm1):
  # General function to convert one-electron operator O1 from basis 1 to basis 2 
  # given their overlap S12 and basis 1 inverse 
  #O2 = numpy.einsum('mi,mn,no,op,pj->ij',S12,Sm1,O1,Sm1,S12)
  temp=numpy.dot(Sm1,S12)
  O2 = numpy.dot(numpy.transpose(temp),numpy.dot(O1,temp))
  return(O2)

def O2_1to2(O1,S12,Sm1):
  # General function to convert two-electron operator O1 from basis 1 to basis 2 
  # given their overlap S12 and basis 1 inverse 
  tt = numpy.einsum('mi,mn->ni',S12,Sm1)
  tmp =numpy.einsum('mi,nj,mnop->ijop',tt,tt,O1) 
  O2 =numpy.einsum('ok,pl,ijop->ijkl',tt,tt,tmp) 
  return(O2)


def eci(ks,QS=None,Pin=None):
  # Generate the pDFT+CI correlation energy with projection Q defaulting to that in ks 
  if(QS is None):
    QS = ks.QS[0]
  m = ks.mol
  nao = m.nao
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  if(Pin is None):
    P = ks.make_rdm1() 
  else:
    P = Pin 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  Q = numpy.dot(QS,Sm) 
  SQ = numpy.dot(S,Q)
  SQS = numpy.dot(S,QS)

  if(len(P.shape)<3): # Convert RHF to UHF 
    print('Converting RHF to UHF') 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
    mo_a=ks.mo_coeff
    mo_b=mo_a
    e_a = ks.mo_energy 
    e_b=e_a
  else:
    if(len(ks.mo_energy.shape)<2): # Convert ROHF to UHF 
      mo_a=ks.mo_coeff
      mo_b=mo_a
      e_a = ks.mo_energy 
      e_b=e_a
    else:
      (mo_a,mo_b) = ks.mo_coeff
      (e_a,e_b) = ks.mo_energy
  (Na,Nb)=m.nelec

  # Define function to build reference system's effective onelectron Hamiltonian and projected energy 
  def getReferenceSystem(Pin):
    h0 =  ks.get_hcore()
    Enuc = ks.energy_nuc()
    Pp = numpy.einsum('ij,sjk,kl->sil',QS,Pin,SQ)
    print('Getting reference potential \nTotal and projected electrons: ',numpy.einsum('sij,ij->',Pin,S),numpy.einsum('sij,ij->',Pp,S))
    j = ks.get_j(dm=P)
    J = j[0]+j[1]
    j = ks.get_j(dm=Pp)
    Jp0 = j[0]+j[1]
    K  = -1.0*ks.get_k(dm=P)
    Kp0 = -1.0*ks.get_k(dm=Pp)
    n, EXC, VXC = ks._numint.nr_uks(m,ks.grids,ks.xc,P)
    pxc = ks.xc 
    if(ks.allc>0):
      pxc=(pxc.split(','))[0] + ','
    np,EXCp,VXCp0= ks._numint.nr_uks(m,ks.grids,pxc,Pp)
 
    # Project the potentials as well, so full 4-index (ab|cd) is projected 
    Jp= numpy.einsum('ik,kj->ij',SQ,numpy.einsum('ik,kj->ij',Jp0,QS))
    Kp= numpy.einsum('ik,skj->sij',SQ,numpy.einsum('sik,kj->sij',Kp0,QS))
    VXCp= numpy.einsum('ik,skj->sij',SQ,numpy.einsum('sik,kj->sij',VXCp0,QS))
    #Jp = Jp0
    #Kp = Kp0
    #VXCp = VXCp0
    #print('Full J\n',J,'\nProj J\n',Jp)
    #print('Full VXC\n',VXC,'\nProj VXC\n',VXCp)

    # Terms 'inside' and 'outside' the projected region 
    Jout   = J-Jp 
    VXCout = (VXC-VXCp) + hyb*(K-Kp) 
    #print('Js\n',J,'\n',Jp,'\n',Jout)
    #print('Kss\n',(VXC+hyb*K)[0],'\n',(VXCp+hyb*(K-Kp))[0],'\n',VXCout[0])

    # Mean-field effective Hamiltonian, 'outside' and 'inside' 
    # NOTE this assumes 100% projected Vee 'inside' reference system 
    h1aoout = numpy.zeros_like(P) 
    h1aoout[0] = h0 + Jout + VXCout[0]
    h1aoout[1] = h0 + Jout + VXCout[1] 
    
    EH = numpy.einsum('sij,ji->',P,0.5*J)
    EHp = numpy.einsum('sij,ji->',Pp,0.5*Jp0)
    EX = numpy.einsum('sij,sji->',P,0.5*K)
    EXp = numpy.einsum('sij,sji->',Pp,0.5*Kp0)
  
    # Reference system mean-field two-electron energy and HXC operator 
    EEref = EHp + EXp 
    vhxcin  = numpy.zeros_like(P) 
    vhxcin[0] = Jp + Kp[0]
    vhxcin[1] = Jp + Kp[1] 

    # TEST 
    #print('Focka\n',h0+J+K[0],'\n',ks.get_fock()[0],'\n',h1aoout[0],'\n',vhxcin[0])
   
    # Projected energy
    Eproj0 = numpy.einsum('sij,ij->',P,h0) + Enuc 
    Eproj1 = Eproj0 + EH + EXp   + (EXC - EXCp) + hyb*(EX-EXp) -EEref 

    # Reference system energy, single-determinant HF calc 
    ErefHF = EEref +numpy.einsum('sij,sij->',P,h1aoout)
    print('Reference system HF energy ',ErefHF)
  
    return(h1aoout,vhxcin,Eproj1,EEref,ErefHF)
  
  # Get embedding potential from SCF density  
  h1aoout,vhxcin,Eproj1,EEref,ErefHF = getReferenceSystem(P)
  EPDFT1 = Eproj1+EEref # Total energy of the real system, single-determiant HF calc 
  fao = h1aoout + vhxcin 
  fmoa=numpy.einsum('mi,mn,nj->ij',mo_a,fao[0],mo_a)
  fmob=numpy.einsum('mi,mn,nj->ij',mo_b,fao[1],mo_b)
  
  t1=(fmoa-numpy.diag(e_a))**2 + (fmob-numpy.diag(e_b))**2
  print('MORE ',EPDFT1,ks.e_tot)
  print('CI TESTS: ',(EPDFT1-ks.e_tot)*2,numpy.einsum('ij->',t1)) # Test ref sys energy and Fock operator 
  #print('MO energies')
  #for i in range(nao):
  #   print('%12.6f %12.6f %12.6f %12.6f'%(fmoa[i,i],e_a[i],fmob[i,i],e_b[i]))

  # Transform the MOs, only do CI with transformed MOs that have non-negligible projection onto Q. 

  # Transform occ alpha, virt alpha, occ beta, virt beta blocks separately 
  thresh=0.02
  ncas = 0 
  nelecas = 0 
  froz_a=[]
  act_a=[]
  virt_a=[]
  froz_b=[]
  act_b=[]
  virt_b=[]
  for ttype in range(4): 
    mo=mo_a[:,:Na]
    if(ttype == 1): mo = mo_a[:,Na:]
    if(ttype == 2): mo = mo_b[:,:Nb]
    if(ttype == 3): mo = mo_b[:,Nb:]
    if(mo.shape[1]>0):
      B = numpy.einsum('mi,mn,nj->ij',mo,SQS,mo)
      val,vec= linalg.eigh(B)
      print(' ttype ',ttype,' eigenvals \n',val)
      tt= numpy.einsum('mi,ij->mj',mo,vec) 
      froz=[]
      act=[]
      # TEST add beta virts to keep act_a and act_b the same length 
      if(ttype==3):
        nact=len(act_a) - len(act_b)
        print('Adding ',nact,' of ',B.shape[0],' beta virts to ',len(act_b))
        nvir= B.shape[0]-nact
        for i in range(B.shape[0]):
           vv = tt[:,i]
           if(i<nvir):
             froz.append(vv)
           else:       
             act.append(vv)
      else:           
        for i in range(B.shape[0]):
          #vv = numpy.einsum('p,mp->m',vec[:,i],mo)
          vv = tt[:,i]
          if(val[i]>thresh):
            act.append(vv)
          else:
            froz.append(vv)

      if(ttype == 0): 
        froz_a = froz 
        act_a = act 
      if(ttype == 1): 
        virt_a = froz 
        act_a = act_a + act 
      if(ttype == 2): 
        froz_b = froz 
        act_b = act 
      if(ttype == 3): 
        virt_b = froz 
        act_b = act_b + act 

  # Do we need these to be the same size? What if all our frozen orbitals are the same ? 
  #if(len(froz_a) != len(froz_b)):
  #  err = 'froz_a %s neq froz_b %s'%(froz_a,froz_b)
  #  sys.exit(err)
  #if(len(act_a) != len(act_b)):
  #  err = 'act_a %s neq act_b %s'%(act_a,act_b)
  #  sys.exit(err)

  ncas = len(act_a)
  #nelecas=Na+Nb - len(froz_a)  - len(froz_b) 
  nelecas=[Na-len(froz_a),Nb-len(froz_b)]
  # Active orbital resize
  diff=len(act_b)-len(act_a)
  if(diff>0):
    act_b=act_b[diff:]
  tmo_a = numpy.transpose(numpy.array(froz_a + act_a + virt_a))
  atmo_a = numpy.transpose(numpy.array(act_a ))
  tmo_b = numpy.transpose(numpy.array(froz_b + act_b + virt_b))
  atmo_b = numpy.transpose(numpy.array(act_b ))
  # TEST halfass the beta spins
  #tmo_b = tmo_a
  #atmo_b = atmo_a 
  print('act_a len',len(act_a))
  print('atmo_a shape ',atmo_a.shape)

  # TEST. Full CI, no transform, for testing purposes only. 
  #ncas = nao
  #nelecas=[Na,Nb]
  #tmo_a = mo_a 
  #tmo_b = mo_b 
  #atmo_a = tmo_a
  #atmo_b = tmo_b

  print('Doing CI with nao ',nao,' nelec ',Na+Nb,' ncas ',ncas,' nelecas ',nelecas)
  if(nelecas[0]+nelecas[1]<2):
    return(0) 
  if(nelecas[0]+nelecas[1]>ncas):
    return(0) 
  print('Transformed MO shape ',tmo_a.shape)
  print('Active transformed MO shape ',atmo_a.shape,atmo_b.shape)
  

  # Construct MO-basis reference system h1 and h2 
  #h1mo=numpy.zeros_like(P)
  h1mo = numpy.zeros((2,ncas,ncas))
  h1mo[0]=numpy.einsum('mi,mn,nj->ij',atmo_a,h1aoout[0],atmo_a)
  h1mo[1]=numpy.einsum('mi,mn,nj->ij',atmo_b,h1aoout[1],atmo_b)
  pmo_a=numpy.dot(QS,atmo_a) # fck I think this is wrong 
  pmo_b=numpy.dot(QS,atmo_b)
  porbs = numpy.array([pmo_a,pmo_b])
  apmo_a=numpy.dot(QS,atmo_a)
  apmo_b=numpy.dot(QS,atmo_b)
  aporbs = numpy.array([apmo_a,apmo_b])

  mc=mcscf.UCASCI(ks,ncas,nelecas)
  max_memory = max(400, mc.max_memory-lib.current_memory()[0])
  #h2mo = mc.get_h2eff(porbs)
  h2mo = mc.get_h2eff(aporbs) # Only need in the active projected MOs , right? 
  print('h1mo shape ',h1mo.shape)
  print('h2mo[0] shape ',h2mo[0].shape)

  # Do full CI 
  ErefCI, fcivec = mc.fcisolver.kernel(h1mo, h2mo, mc.ncas, mc.nelecas,
                                      ci0=None,verbose=10, max_memory=max_memory,ecore=0)
  print('FCI vector shape ',fcivec.shape)
  print(fcivec)
  hfvec=numpy.zeros_like(fcivec)
  hfvec[0,0]=1
  eci = mc.fcisolver.energy(h1mo,h2mo,fcivec,mc.ncas,mc.nelecas)
  ehf = mc.fcisolver.energy(h1mo,h2mo,hfvec,mc.ncas,mc.nelecas)
  print('FCI vector energies',eci,ehf)
  Ecorr = eci - ehf 

  # Do CCSD following the PySCF example 42-as_casci_fcisolver.py 
  #mc.fcisolver = AsFCISolver()
  #print('Doing with cas nelecas ',mc.ncas,mc.nelecas)
  #Ecorr = mc.fcisolver.kernel(h1mo, h2mo, mc.ncas, mc.nelecas,
  #                            ci0=None,verbose=10, max_memory=max_memory,ecore=0)

  print('Ecorr ',Ecorr)
  return(Ecorr) 

# Older version 
def epzlh(ks,P,allc=0):
  # Generate the pDFT projected XC energy using local-hybrid-PZ ansatz
  m = ks.mol
  nao = m.nao
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  #P = ks.make_rdm1() 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  if(len(P.shape)<3): # Convert RHF to UHF 
    print('Converting RHF to UHF') 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
  (Na,Nb)=m.nelec
  (Pa,Pb)=P

  PP=numpy.zeros_like(P)
  PP[0] = numpy.dot(ks.QS[0],numpy.dot(P[0],ks.SQ[0]))
  PP[1] = numpy.dot(ks.QS[0],numpy.dot(P[1],ks.SQ[0]))
  h0=ks.get_hcore()
  Jmat2 = ks.get_j(dm=P)
  Jmat = Jmat2[0] + Jmat2[1] 
  Eother = numpy.einsum('sij,ij->',P,h0) + numpy.einsum('sij,ji->',P,Jmat)/2. +m.get_enuc()
  K=ks.get_k(dm=P)
  EX=-0.5*numpy.einsum('sij,sij->',P,K)
  KP=ks.get_k(dm=PP)
  EXP=-0.5*numpy.einsum('sij,sij->',PP,KP)
  print('EPZSL N ',numpy.einsum('sij,ji->s',P,S),numpy.einsum('sij,ji->s',PP,S))
  #print('EPZSL EX ',EX,EXP)

  EXC1 = EXP+hyb*(EX-EXP)
  EXC2 = EXC1

  R=numpy.zeros_like(P)
  R[0]=0.5*(numpy.dot(Sm,numpy.dot(K[0],P[0])) + numpy.dot(P[0],numpy.dot(K[0],Sm)))
  R[1]=0.5*(numpy.dot(Sm,numpy.dot(K[1],P[1])) + numpy.dot(P[1],numpy.dot(K[1],Sm)))
  RP=numpy.zeros_like(P)
  RP[0]=0.5*(numpy.dot(Sm,numpy.dot(KP[0],PP[0])) + numpy.dot(PP[0],numpy.dot(KP[0],Sm)))
  RP[1]=0.5*(numpy.dot(Sm,numpy.dot(KP[1],PP[1])) + numpy.dot(PP[1],numpy.dot(KP[1],Sm)))

  # Do the numerical integrals 
  EX2=0
  EXP2=0
  ESL=0
  EXSL=0
  ECSL=0
  EPSL=0
  EPSL2=0
  NA=0
  NPA=0

  hermi=1 
  xc=ks.xc
  pxc=xc
  if(allc>0):
   pxc=(pxc.split(','))[0] + ','
  ni = ks._numint
  ao_deriv=0
  xctype=ni._xc_type(xc)
  if xctype == 'GGA':
    ao_deriv=1
  elif xctype == 'MGGA':
    ao_deriv=1
  nao=m.nao
  tiny = 0.00000001
  make_rhoa, nset = ni._gen_rho_evaluator(m, [P[0]], hermi, False, ks.grids)[:2]
  make_rhob       = ni._gen_rho_evaluator(m, [P[1]], hermi, False, ks.grids)[0]
  make_rhpa, nset = ni._gen_rho_evaluator(m, [PP[0]], hermi, False, ks.grids)[:2]
  make_rhpb       = ni._gen_rho_evaluator(m, [PP[1]], hermi, False, ks.grids)[0]
  for aos, mask, weight, coords in ni.block_loop(m, ks.grids, nao, ao_deriv, max_memory=2000):


      # SL XC 
      rho_a = make_rhoa(0, aos, mask, xctype)
      rho_b = make_rhob(0, aos, mask, xctype)
      if(len(rho_b.shape)>1):
        rho_b[0,rho_b[0]<tiny]=tiny
        rhob = rho_b[0]
      else:
        rho_b[rho_b<tiny]=tiny
        rhob = rho_b
      if(len(rho_a.shape)>1):
        rho_a[0,rho_a[0]<tiny]=tiny
        rhoa = rho_a[0]
      else:
        rho_a[rho_a<tiny]=tiny
        rhoa = rho_a
      rho = (rho_a, rho_b)
      exc, vxc = ni.eval_xc_eff(xc, rho, deriv=1, xctype=xctype)[:2]
      exc = exc*(rhoa+rhob) 
      exsl = exc
      if(allc>0):
        if(pxc == 'hf,'):
          exsl = exsl * 0.0 
        else:   
          exsl, vxc = ni.eval_xc_eff(pxc, rho, deriv=1, xctype=xctype)[:2]
          exsl = exsl*(rhoa+rhob) 
         
      # Projected SL XC 
      rhp_a = make_rhpa(0, aos, mask, xctype)
      rhp_b = make_rhpb(0, aos, mask, xctype)
      if(len(rhp_b.shape)>0):
        rhp_b[0,rhp_b[0]<tiny]=tiny
        rhpb = rhp_b[0]
      else:
        rhp_b[rhp_b<tiny]=tiny
        rhpb = rhp_b
      if(len(rhp_a.shape)>0):
        rhp_a[0,rhp_a[0]<tiny]=tiny
        rhpa = rhp_a[0]
      else:
        rhp_a[rhp_a<tiny]=tiny
        rhpa = rhp_a
      rhp = (rhp_a, rhp_b)
      if(pxc == 'hf,'):
        epxc = 0.0 * exc
      else: 
        epxc, vxc = ni.eval_xc_eff(pxc, rhp, deriv=1, xctype=xctype)[:2]
        epxc = epxc*(rhpa+rhpb)

      # Local hybrid type weighting 
      aos = m.eval_gto("GTOval_sph",coords)
      rhoa= numpy.einsum('ri,ij,rj->r',aos,Pa,aos)
      rhob= numpy.einsum('ri,ij,rj->r',aos,Pb,aos)
      ex=numpy.einsum('ri,sij,rj->r',aos,R,aos)
      ex[ex<tiny]=tiny
      expr=numpy.einsum('ri,sij,rj->r',aos,RP,aos)
      wt = expr/ex
      #wt[wt>1]=1 We will allow the weight to go above 1 I guess? 
      #wt[wt>10]=10 # compromise 12/20/24 to avoid artifacts in alkali
      wt[wt>5]=5 # compromise 12/20/24 to avoid artifacts in alkali
      wt[wt<0]=0
      epxc2 = exsl*wt

      EX2 = EX2 -0.5*numpy.dot(ex,weight)
      EXP2 = EXP2 -0.5*numpy.dot(expr,weight)
      ESL = ESL + numpy.dot(exc,weight)
      EXSL = EXSL + numpy.dot(exsl,weight)
      ECSL = ECSL + numpy.dot(exc-exsl,weight)
      EPSL = EPSL + numpy.dot(epxc,weight)
      EPSL2 = EPSL2 + numpy.dot(epxc2,weight)
      NA = NA + numpy.dot(rhoa,weight)
      NPA= NPA+ numpy.dot(rhoa*wt,weight)
  print('EPZLH TEST: ',Eother,EX,EX2,EXP,EXP2,NA,NPA)
  print('EPZLH ESL: ',ESL,EXSL,ECSL,EPSL,EPSL2)
  EXC1 = EXC1+(ESL-EPSL2)
  EXC2 = EXC2+(ESL-EPSL)
  #return(Eother,EXC1,EXC2)
  return(Eother,EX,EXP,EXSL,ECSL,EPSL2)

# Newer version 
#def epzlh(ks,P,lhexp=20):
#  # Generate the pDFT projected XC energy using local-hybrid-PZ ansatz
#  m = ks.mol
#  nao = m.nao
#  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
#  #P = ks.make_rdm1() 
#  S = ks.get_ovlp() 
#  Sm = numpy.linalg.inv(S)
#  if(len(P.shape)<3): # Convert RHF to UHF 
#    print('Converting RHF to UHF') 
#    Ptemp=P
#    P=numpy.zeros((2,nao,nao))
#    P[0]=Ptemp/2
#    P[1]=Ptemp/2
#  (Na,Nb)=m.nelec
#  (Pa,Pb)=P
#
#  PP=numpy.zeros_like(P)
#  PP[0] = numpy.dot(ks.QS[0],numpy.dot(P[0],ks.SQ[0]))
#  PP[1] = numpy.dot(ks.QS[0],numpy.dot(P[1],ks.SQ[0]))
#  h0=ks.get_hcore()
#  Jmat2 = ks.get_j(dm=P)
#  Jmat = Jmat2[0] + Jmat2[1] 
#  Eother = numpy.einsum('sij,ij->',P,h0) + numpy.einsum('sij,ji->',P,Jmat)/2. +m.get_enuc()
#  K=ks.get_k(dm=P)
#  EX=-0.5*numpy.einsum('sij,sij->',P,K)
#  KP=ks.get_k(dm=PP)
#  EXP=-0.5*numpy.einsum('sij,sij->',PP,KP)
#  print('EPZSL N ',numpy.einsum('sij,ji->s',P,S),numpy.einsum('sij,ji->s',PP,S))
#  #print('EPZSL EX ',EX,EXP)
#
#  EXC1 = EXP+hyb*(EX-EXP)
#  EXC2 = EXC1
#
#  R=numpy.zeros_like(P)
#  R[0]=0.5*(numpy.dot(Sm,numpy.dot(K[0],P[0])) + numpy.dot(P[0],numpy.dot(K[0],Sm)))
#  R[1]=0.5*(numpy.dot(Sm,numpy.dot(K[1],P[1])) + numpy.dot(P[1],numpy.dot(K[1],Sm)))
#  RP=numpy.zeros_like(P)
#  RP[0]=0.5*(numpy.dot(Sm,numpy.dot(KP[0],PP[0])) + numpy.dot(PP[0],numpy.dot(KP[0],Sm)))
#  RP[1]=0.5*(numpy.dot(Sm,numpy.dot(KP[1],PP[1])) + numpy.dot(PP[1],numpy.dot(KP[1],Sm)))
#
#  # Do the numerical integrals 
#  EX2=0
#  EXP2=0
#  EXSL=0
#  EXLH=0
#  ECSL=0
#  ECLH=0
#  EXPSL=0
#  ECPSL=0
#  EXPLH=0
#  ECPLH=0
#  NA=0
#  NPA=0
#
#  hermi=1 
#  xc=ks.xc
#  pxc = xc 
#  pxc=(pxc.split(','))[0] + ','
#
#  ni = ks._numint
#  ao_deriv=0
#  xctype=ni._xc_type(xc)
#  if xctype == 'GGA':
#    ao_deriv=1
#  elif xctype == 'MGGA':
#    ao_deriv=1
#  nao=m.nao
#  tiny = 0.00000001
#  make_rhoa, nset = ni._gen_rho_evaluator(m, [P[0]], hermi, False, ks.grids)[:2]
#  make_rhob       = ni._gen_rho_evaluator(m, [P[1]], hermi, False, ks.grids)[0]
#  make_rhpa, nset = ni._gen_rho_evaluator(m, [PP[0]], hermi, False, ks.grids)[:2]
#  make_rhpb       = ni._gen_rho_evaluator(m, [PP[1]], hermi, False, ks.grids)[0]
#  for aos, mask, weight, coords in ni.block_loop(m, ks.grids, nao, ao_deriv, max_memory=2000):
#
#
#      # Density and projected density 
#      rho_a = make_rhoa(0, aos, mask, xctype)
#      rho_b = make_rhob(0, aos, mask, xctype)
#      if(len(rho_b.shape)>1):
#        rho_b[0,rho_b[0]<tiny]=tiny
#        rhob = rho_b[0]
#      else:
#        rho_b[rho_b<tiny]=tiny
#        rhob = rho_b
#      if(len(rho_a.shape)>1):
#        rho_a[0,rho_a[0]<tiny]=tiny
#        rhoa = rho_a[0]
#      else:
#        rho_a[rho_a<tiny]=tiny
#        rhoa = rho_a
#      rho = (rho_a, rho_b)
#      rhp_a = make_rhpa(0, aos, mask, xctype)
#      rhp_b = make_rhpb(0, aos, mask, xctype)
#      if(len(rhp_b.shape)>0):
#        rhp_b[0,rhp_b[0]<tiny]=tiny
#        rhpb = rhp_b[0]
#      else:
#        rhp_b[rhp_b<tiny]=tiny
#        rhpb = rhp_b
#      if(len(rhp_a.shape)>0):
#        rhp_a[0,rhp_a[0]<tiny]=tiny
#        rhpa = rhp_a[0]
#      else:
#        rhp_a[rhp_a<tiny]=tiny
#        rhpa = rhp_a
#      rhp = (rhp_a, rhp_b)
#
#      # Exact exchange 
#      #aos = m.eval_gto("GTOval_sph",coords)
#      aos = m.eval_gto("GTOval_cart",coords)
#      print('TEST shapes ',m.nao,aos.shape,Pa.shape)
#      rhoa= numpy.einsum('ri,ij,rj->r',aos,Pa,aos)
#      rhob= numpy.einsum('ri,ij,rj->r',aos,Pb,aos)
#      rhoa[rhoa<tiny]=tiny
#      rhob[rhob<tiny]=tiny
#      ex=numpy.einsum('ri,sij,rj->r',aos,R,aos)
#      ex[ex<tiny]=tiny
#      ex= -0.5*ex
#      expr=numpy.einsum('ri,sij,rj->r',aos,RP,aos)
#      expr[expr<tiny]=tiny
#      expr= -0.5*expr
#
#      # SL XC  
#      exsl=numpy.zeros_like(rhoa)
#      ecsl=numpy.zeros_like(rhoa)
#      if(xc!='hf'):
#        excsl, vxc = ni.eval_xc_eff(xc, rho, deriv=1, xctype=xctype)[:2]
#        excsl = excsl*(rhoa+rhob) 
#        if(pxc == 'hf,'):
#          exsl = numpy.zeros_like(excsl)
#        else:
#          exsl, vxc = ni.eval_xc_eff(pxc, rho, deriv=1, xctype=xctype)[:2]
#        exsl = exsl*(rhoa+rhob) 
#        ecsl = excsl - exsl 
#
#      # No XC where projected exact exchange is near total exact exchange. 
#      fac = expr/ex
#      fac[fac>1]=1
#      fac[fac<0]=0
#
#      # Local hybrid weights: No SL XC where fac is near 0
#      # Note that exsl must be multiplied by (1-hyb) for global hybrids. 
#      ahflh=numpy.zeros_like(ex)
#      if(pxc != 'hf,'):
#        exslf = exsl
#        if(hyb>0.000001):
#          exslf = exsl/(1-hyb)
#        z=(exslf/ex)-1
#        z[z<tiny]=tiny
#        ahflh=erf(lhexp*z) 
#        # October 24. Local hybrid admixture of full HF exchange near projected regions where fac is 1
#        ahflh = fac**0.1
#        #ahflh = ahflh*0.0 # turh off 
#        ahflh[ahflh>1]=1
#        ahflh[ahflh<0]=0
#      # Projected SL XC 
#      #epxcsl, vxc = ni.eval_xc_eff(xc, rhp, deriv=1, xctype=xctype)[:2]
#      #epxcsl = epxcsl*(rhpa+rhpb)
#      #epxlh, vxc = ni.eval_xc_eff(pxc, rhp, deriv=1, xctype=xctype)[:2]
#      #epxlh = epxlh*(rhpa+rhpb)
#      #epcsl = epxcsl - epxlh 
#      exlh = ahflh*ex+(1-ahflh)*(hyb*ex+exsl) # exsl is already scaled by (1-hyb), this is a local hybrid of a global hybrid 
#      eclh = (1-ahflh)*ecsl
#      epxlh = fac*exlh
#      epclh = fac*eclh
#      epxsl = fac*exsl
#      epcsl = fac*ecsl
#
#      EX2   = EX2  + numpy.dot(ex,weight)
#      EXP2  = EXP2 + numpy.dot(expr,weight)
#      EXSL  = EXSL + numpy.dot(exsl,weight) 
#      ECSL  = ECSL + numpy.dot(ecsl,weight)
#      EXPSL = EXPSL + numpy.dot(epxsl,weight) 
#      ECPSL = ECPSL + numpy.dot(epcsl,weight) 
#      EXLH  = EXLH + numpy.dot(exlh,weight) 
#      EXPLH = EXPLH + numpy.dot(epxlh,weight) 
#      ECLH  = ECLH + numpy.dot(eclh,weight)
#      ECPLH = ECPLH + numpy.dot(epclh,weight) 
#      NA = NA + numpy.dot(rhoa+rhob,weight)
#      NPA= NPA+ numpy.dot((rhoa+rhob)*fac,weight)
#  if(hyb>0.000001):
#     EXSL = EXSL/(1-hyb) # exsl is already scaled by (1-hyb), this is a global hybrid 
#  print('EPZLH Ntot,Nact',NA,NPA)
#  print('EPZLH TEST: ',Eother,EX,EX2,EXP,EXP2,NA,NPA)
#  print('EPZSL ESL: ',EXSL,ECSL,EXPSL,ECPSL)
#  print('EPZLH ELH: ',EXLH,ECLH,EXPLH,ECPLH)
#  #return(Eother,EX,EXP, EXSL,EXLH,ECSL,ECLH, EXPLH,ECPLH)
#  return(Eother,EX,EXP, EXLH,ECSL,EXPLH,ECPSL)
#
def euci(ks,hl=0,Pin=None):
  # Generate the pDFT+UCI correlation energy 

  # Set up 
  pAOs = ks.paos 
  m = ks.mol
  nao = m.nao
  omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=m.spin)
  if(Pin is None):
    P = ks.make_rdm1() 
  else:
    P = Pin 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  if(len(P.shape)<3): # Convert RHF to UHF 
    print('Converting RHF to UHF') 
    Ptemp=P
    P=numpy.zeros((2,nao,nao))
    P[0]=Ptemp/2
    P[1]=Ptemp/2
    mo_a=ks.mo_coeff
    mo_b=mo_a
    e_a = ks.mo_energy 
    e_b=e_a
  else:
    if(len(ks.mo_energy.shape)<2): # ROHF to UHF 
      print('Converting ROHF to UHF') 
      mo_a=ks.mo_coeff
      mo_b=mo_a
      e_a = ks.mo_energy 
      e_b=e_a
    else:
      (mo_a,mo_b) = ks.mo_coeff
      (e_a,e_b) = ks.mo_energy
  (Na,Nb)=m.nelec

  opAOs,SopAOs,SAOopAOs,VeeopAOs= makeOPAOs(S,pAOs,ks.VeepAOs)
  opAOf,SopAOf,SAOopAOf = makeallOPAOs(S,opAOs) 


  # Energy weighted density matrices 
  Pv=numpy.zeros_like(P)
  Pv[0] = numpy.dot(mo_a[:,Na:],numpy.transpose(mo_a[:,Na:]))
  Pv[1] = numpy.dot(mo_b[:,Nb:],numpy.transpose(mo_b[:,Nb:]))
  PE=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,:Na],e_a[:Na])
  PE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,:Na]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,:Nb],e_b[:Nb])
  PE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,:Nb]))
  PvE=numpy.zeros_like(P)
  temp=numpy.einsum('mi,i->mi',mo_a[:,Na:],e_a[Na:])
  PvE[0] = numpy.dot(temp,numpy.transpose(mo_a[:,Na:]))
  temp=numpy.einsum('mi,i->mi',mo_b[:,Nb:],e_b[Nb:])
  PvE[1] = numpy.dot(temp,numpy.transpose(mo_b[:,Nb:]))

  # Indexing for projected natural orbitals 
  ntot=0
  shellstarts = []
  shellends = []
  for ishell in range(len(opAOs)):
    shellstarts.append(ntot)
    ntot = ntot + opAOs[ishell].shape[1]
    shellends.append(ntot)

  exs=numpy.zeros(ntot)
  exsls=numpy.zeros(ntot)
  ecs=numpy.zeros(ntot)
  ecs2=numpy.zeros(ntot)
  wts=numpy.zeros(ntot)

  # Loop over shells 
  itot=-1
  for ishell in range(len(opAOs)):
    SopAOm=numpy.linalg.inv(SopAOs[ishell])


    # This step is a timesink for large systems, as it's an AO-MO transform for
    # the entire molecule not just for the shell of interest.  The good side is
    # there's a small and asymptotically constant number of 'mos' on this shell
    # Might be smarter to compute opAO Vees from just the AOs on this atom? 
    ####Vees = O2_1to2(VeeAO,SAOopAOs[ishell],Sm)
    #tt = numpy.dot(Sm,SAOopAOs[ishell])
    #print('TEST TT ',tt.shape)
    #print('TEST2 TT ',numpy.einsum('mi,mn,nj->ij',tt,S,tt))
    #Vees2 = m.ao2mo(tt) 
    #Vees = ao2mo.restore(1,numpy.asarray(Vees2),tt.shape[1])
    Vees = VeeopAOs[ishell]
    print('TEST VEEs ',Vees.shape)
    print('VEES FIRST ',Vees[0,0,0,0])

    # Density matrices in this shell 
    Ps=P_1to2(P,SAOopAOs[ishell],SopAOm)
    Pvs=P_1to2(Pv,SAOopAOs[ishell],SopAOm)
    PEs=P_1to2(PE,SAOopAOs[ishell],SopAOm)
    PvEs=P_1to2(PvE,SAOopAOs[ishell],SopAOm)
    print('Shell ',ishell,' alpha density \n',Ps[0])
    print('Shell ',ishell,' projected 1PDM diagonal ')
    for iproj in range(Ps.shape[1]):
      print('%6.3f %6.3f '%(Ps[0,iproj,iproj],Ps[1,iproj,iproj]))

    # Loop over atomic natural orbitals in this shell 
    (vals,vecs)=numpy.linalg.eigh(Ps[0])
    print('Shell ',ishell,' alpha density eigenvalues ',vals)
    for iproj in range(len(vals)):
       itot = itot+1 
       v = vecs[:,iproj]

       # Do FCI on just this ANO 
       if(hl>0): 
         vf = V_1to2(v,numpy.transpose(SAOopAOs[ishell]),Sm)
         q=numpy.einsum('m,n->mn',vf,vf)
         test = (numpy.dot(q,numpy.dot(S,q))-q)**2
         print('HL 1 TEST: ',numpy.einsum('ij->',test))
         qs = numpy.dot(q,S)
         ecs2[itot]  = eci(ks=ks,QS=qs,Pin=Pin)

       #w0 = numpy.einsum('p,pq->q',v,SopAOf[shellstarts[ishell]:shellends[ishell]])
       w0 = numpy.dot(v,SopAOf[shellstarts[ishell]:shellends[ishell]])
       wt = numpy.dot(w0,w0)
       wts[itot]=wt
       print('Total Weight Projection ',wt)
       print('Weight ',1/wt)

       J=numpy.dot(v,numpy.dot(v,numpy.dot(v,numpy.dot(v,Vees))))


       # Assemble the 2x2 Hamiltonian 
       noa=numpy.dot(v,numpy.dot(Ps[0],v))
       nob=numpy.dot(v,numpy.dot(Ps[1],v))
       nva=numpy.dot(v,numpy.dot(Pvs[0],v))
       nvb=numpy.dot(v,numpy.dot(Pvs[1],v))
       eoa=numpy.dot(v,numpy.dot(PEs[0],v))
       eob=numpy.dot(v,numpy.dot(PEs[1],v))
       eva=numpy.dot(v,numpy.dot(PvEs[0],v))
       evb=numpy.dot(v,numpy.dot(PvEs[1],v))
       #(noa,nob) = numpy.einsum('m,smn,n->s',v,Ps,v)
       #(nva,nvb) = numpy.einsum('m,smn,n->s',v,Pvs,v)
       #(eoa,eob) = numpy.einsum('m,smn,n->s',v,PEs,v)
       #(eva,evb) = numpy.einsum('m,smn,n->s',v,PvEs,v)
       print('Occupancies ',ishell,iproj,noa,nob)
       print('Virt occs   ',nva,nvb)
       print('Self energy ',J)

       # Keep DFT+U-type exchange pieces 
       exs[itot]   = -J*(noa**2+nob**2)/2
       exsls[itot] = -J*(noa+nob)/2

       # Diagonalize the 2x2 Hamiltonian 
       if(noa>0.000001 and nob>0.000001 and nva>0.000001 and nvb>0.000001):
          eoa = eoa/(noa +0.00000001)
          eob = eob/(nob +0.00000001)
          eva = eva/(nva +0.00000001)
          evb = evb/(nvb +0.00000001)
          print('Proj Occ Energies    ',eoa,eob)
          print('Proj Virt Energies',eva,evb)

          # Diagonalize 
          o = J*(noa*nob*nva*nvb)**0.5 # <Phi_0|Vp(r)|Phi_oo^vv> 
          o = numpy.maximum(o,1e-10)
          ee=(eva+evb-eoa-eob)          
          d = (ee + J*(noa*nob+nva*nvb-noa*nvb-nob*nva))/2
          # Very little difference d = ee/2 # TEST 
          ec = d-(d**2+o**2)**0.5 
          print('Hamiltonian o and d and ec',o,d,ec)
          ecs[itot]=ec

  wts1 = 1.0/wts
  wts1 = wts1**2
  print('ECS ',ecs,ecs2)
  print('WEIGHTS ',wts1)
  EC1 = numpy.dot(ecs  ,wts1)
  EC2 = numpy.dot(ecs2 ,wts1)
  EX  = numpy.dot(exs  ,wts1)
  EXSL= numpy.dot(exsls,wts1)

  ### Projected CI in the full active space 
  EC3 = 0 
  if(hl>1): 
    EC3 = eci(ks,Pin=Pin)
  return(EC1,EC2,EC3,EX,EXSL)


##### 
class AsFCISolver(object):
    def __init__(self):
        self.mycc = None

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        fakemol = gto.M(verbose=0)
        fakemol.spin=nelec[0]-nelec[1]
        nelec = numpy.sum(nelec)
        fakemol.nelectron = nelec
        fake_hf = scf.UHF(fakemol)
        print('Look we made a fake hf ')
        print('h1 shape ',h1.shape)
        #fake_hf._eri = ao2mo.restore(8, h2, norb)
        fake_hf._eri = ao2mo.restore(8, h2[0], norb) # bgj use alpha-alpha eris here 
        fake_hf.get_hcore = lambda *args: h1
        fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
        fake_hf.kernel()
        print('Fake HF energy ',fake_hf.e_tot)
        self.mycc = cc.UCCSD(fake_hf)
        eris = self.mycc.ao2mo() # This is a _ChemistsERIs object not just a list of ERIs. It has a bunch of crap like Fock operators, OOOO and OVOV blocks, and other horseshit in it 
        e_corr, t1, t2 = self.mycc.kernel(eris=eris)
        return(e_corr)
        #print('Fake correlation energy ',e_corr)
        #l1, l2 = self.mycc.solve_lambda(t1, t2, eris=eris)
        #print('Fake total energy ',self.mycc.e_tot)
        #e_tot = self.mycc.e_tot + ecore
        #return e_tot, CCSDAmplitudesAsCIWfn([t1, t2, l1, l2])

    def make_rdm1(self, fake_ci, norb, nelec):
        t1, t2, l1, l2 = fake_ci.cc_amplitues
        dm1 = self.mycc.make_rdm1(t1, t2, l1, l2, ao_repr=True)
        return dm1

    def make_rdm12(self, fake_ci, norb, nelec):
        t1, t2, l1, l2 = fake_ci.cc_amplitues
        dm2 = self.mycc.make_rdm2(t1, t2, l1, l2, ao_repr=True)
        return self.make_rdm1(fake_ci, norb, nelec), dm2

    def spin_square(self, fake_ci, norb, nelec):
        return 0, 1
class CCSDAmplitudesAsCIWfn:
    def __init__(self, cc_amplitues):
        self.cc_amplitues = cc_amplitues
