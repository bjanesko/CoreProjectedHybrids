#!/usr/bin/env python
# Work routines for projected DFT 
import time
import numpy
from scipy import linalg
from pyscf import gto 

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
    orbs = ks.paos 
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

def build_mbproj(ks,daos=False,faos=False,vaos=False,dum=False):
    ''' Build projection operators from the core AOs of an existing AO basis 
    This version builds two projection operators: one for second-period
    atoms Li-Ne, one for third-period atoms Na-Ar. 
    With daos=True, we get one projection operator for transition metal atoms 
    With vaos=True, we get one projection operator per shell
    Args:
        ks : an instance of :class:`RKS`
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
      
      # Find second- and third-period atom core minimal basis AOs 
      coreaos=[]
      coreassign=[]
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
            coreassign.append(2)
            print('Proj AO',labs[iao],coreassign[-1])
        elif(dum):
          #print('+++ ',iao,labs[iao])
          if(iat>2 and   (('py' in labs[iao]) or ('pz' in labs[iao]) or ('px' in labs[iao])) ):
            coreaos.append(iao)
            coreassign.append(2)
            print(labs[iao],coreassign[-1])
        else:
          print('YOU ARE THERE WITH IAT ',iat,' daos ',daos,' faos ',faos)
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

      # Build core minimal basis 
      NC = len(coreaos)
      if(NC>0):
        SC = numpy.zeros((NC,NC))
        SXC = numpy.zeros((N,NC))
        for ic in range(NC):
          SXC[:,ic] = SX[:,coreaos[ic]]
          for jc in range(NC):
             SC[ic,jc] = SM[coreaos[ic],coreaos[jc]]

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
          build_mbproj(ks,vaos=True)
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
      elif(isinstance(ks.paos,list) and len(ks.paos[0].shape)==2 ):
        pao_proj(ks) # New function May 2024, list of shells of AOs 
        return 
      elif(isinstance(ks.paos,list)):
        aoss = ks.paos 
      else:
        if(len(ks.paos.shape) == 2):
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
def makeOPAOs(SAO,pAOs):
  # Make block-orthogonalized pAOs from blocks of pAOs 
  opAOs=[]
  SopAOs=[]
  SAOopAOs=[]
  for ishell in range(len(pAOs)):
     pAO = pAOs[ishell]
     print('Shell ',ishell,' pAOs \n',pAO)
     nproj=pAO.shape[1]
     opAO = numpy.zeros_like(pAO)
     Sshell = numpy.einsum('mp,mn,nq->pq',pAO,SAO,pAO)
     print('Shell ',ishell,' pAO overlap \n',Sshell)
     (vals,vecs)=numpy.linalg.eigh(Sshell)
     for i in range(len(vals)):
      if(vals[i]>0.000001):
        vals[i]=vals[i]**(-0.5)
      else:
        vals[i]=0
     opAO = numpy.einsum('mi,ij,j->mj',pAO,vecs,vals)
     opAOs.append(opAO) 
     SopAOs.append(numpy.einsum('mi,mn,nj->ij',opAO,SAO,opAO))
     SAOopAOs.append(numpy.einsum('mi,mn->ni',opAO,SAO))
  return(opAOs,SopAOs,SAOopAOs)

def makeallOPAOs(SAO,opAOs):
  # Return a single (ao,opao) matrix of all the block-orthogonalized opAOs. 
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
  SAOallopAO = numpy.einsum('mi,mn->ni',allopAO,SAO)
  SallopAO = numpy.einsum('mi,mn,nj->ij',allopAO,SAO,allopAO)
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
    opAOs,SopAOs,SAOopAOs = makeOPAOs(S,pAOs)
    opAO,SopAO,SAOopAO = makeallOPAOs(S,opAOs)
    SopAOm = numpy.linalg.inv(SopAO)
    Sm = numpy.linalg.inv(S)
    Q = numpy.einsum('mn,ni,ij,oj,op->mp',Sm,SAOopAO,SopAOm,SAOopAO,Sm)
    ks.QS=[numpy.einsum('ik,kj->ij',Q,S)]
    ks.SQ=[numpy.einsum('ik,kj->ij',S,Q)]
    # DEBUG TEST 
    test = numpy.dot(Q,numpy.dot(S,Q))-Q
    print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q))

def P_1to2(P1,S12,Sm2):
  # General function to convert density matrix P from basis 1 to basis 2 
  # given their overlap S12 and basis 2 inverse 
  P2 = numpy.einsum('pr,mr,smn,nt,tq->spq',Sm2,S12,P1,S12,Sm2)
  return(P2)

def O1_1to2(O1,S12,Sm1):
  # General function to convert one-electron operator O1 from basis 1 to basis 2 
  # given their overlap S12 and basis 1 inverse 
  #O2 = numpy.einsum('mi,mn,no,op,pj->ij',S12,Sm1,O1,Sm1,S12)
  tt = numpy.einsum('mi,mn->ni',S12,Sm1)
  O2 = numpy.einsum('ni,np,pj->ij',tt,O1,tt)
  return(O2)

def O2_1to2(O1,S12,Sm1):
  # General function to convert two-electron operator O1 from basis 1 to basis 2 
  # given their overlap S12 and basis 1 inverse 
  tt = numpy.einsum('mi,mn->ni',S12,Sm1)
  tmp =numpy.einsum('mi,nj,mnop->ijop',tt,tt,O1) 
  O2 =numpy.einsum('ok,pl,ijop->ijkl',tt,tt,tmp) 
  return(O2)

def euci(ks):
  # Generate the pDFT+UCI correlation energy 

  # Set up 
  pAOs = ks.paos 
  m = ks.mol
  P = ks.make_rdm1() 
  S = ks.get_ovlp() 
  Sm = numpy.linalg.inv(S)
  (mo_a,mo_b) = ks.mo_coeff
  (e_a,e_b) = ks.mo_energy
  nao = m.nao
  (Na,Nb)=m.nelec
  opAOs,SopAOs,SAOopAOs = makeOPAOs(S,pAOs)
  opAOf,SopAOf,SAOopAOf = makeallOPAOs(S,opAOs) 

  # Twoelec integrals in AO basis 
  VeeAO = m.intor("int2e") 

  # Energy weighted density matrices 
  Pv=numpy.zeros_like(P)
  Pv[0] = numpy.einsum('mi,ni->mn',mo_a[:,Na:],mo_a[:,Na:])
  Pv[1] = numpy.einsum('mi,ni->mn',mo_b[:,Nb:],mo_b[:,Nb:])
  PE=numpy.zeros_like(P)
  PE[0] = numpy.einsum('mi,i,ni->mn',mo_a[:,:Na],e_a[:Na],mo_a[:,:Na])
  PE[1] = numpy.einsum('mi,i,ni->mn',mo_b[:,:Nb],e_b[:Nb],mo_b[:,:Nb])
  PvE=numpy.zeros_like(P)
  PvE[0] = numpy.einsum('mi,i,ni->mn',mo_a[:,Na:],e_a[Na:],mo_a[:,Na:])
  PvE[1] = numpy.einsum('mi,i,ni->mn',mo_b[:,Nb:],e_b[Nb:],mo_b[:,Nb:])

  # Indexing for projected natural orbitals 
  ntot=0
  shellstarts = []
  shellends = []
  for ishell in range(len(opAOs)):
    shellstarts.append(ntot)
    ntot = ntot + opAOs[ishell].shape[1]
    shellends.append(ntot)

  ecs=numpy.zeros(ntot)
  wts=numpy.zeros(ntot)
  # Loop over shells 
  itot=-1
  for ishell in range(len(opAOs)):
    SopAOm=numpy.linalg.inv(SopAOs[ishell])
    Vees = O2_1to2(VeeAO,SAOopAOs[ishell],Sm)

    # Density matrices in this shell 
    Ps=P_1to2(P,SAOopAOs[ishell],SopAOm)
    Pvs=P_1to2(Pv,SAOopAOs[ishell],SopAOm)
    PEs=P_1to2(PE,SAOopAOs[ishell],SopAOm)
    PvEs=P_1to2(PvE,SAOopAOs[ishell],SopAOm)

    # Loop over atomic natural orbitals in this shell 
    (vals,vecs)=numpy.linalg.eigh(Ps[0])
    for iproj in range(len(vals)):
       itot = itot+1 
       v = vecs[:,iproj]

       w0 = numpy.einsum('p,pq->q',v,SopAOf[shellstarts[ishell]:shellends[ishell]])
       wt = numpy.dot(w0,w0)
       wts[itot]=wt
       print('Weight ',1/wt)

       # This should be sped up 
       J = numpy.einsum('i,j,k,l,ijkl->',v,v,v,v,Vees)
       #PforU=numpy.zeros_like(Ps)
       #PforU[0]=numpy.einsum('m,n->mn',v,v)
       #PforUAO = P_1to2(PforU,numpy.transpose(SAOopAOs[ishell]),Sm)
       #JforUAO=ks.get_j(dm=PforUAO)
       #J=numpy.einsum('ij,ji->',PforUAO[0],JforUAO[0])

       # Assemble the 2x2 Hamiltonian 
       (noa,nob) = numpy.einsum('m,smn,n->s',v,Ps,v)
       (nva,nvb) = numpy.einsum('m,smn,n->s',v,Pvs,v)
       (eoa,eob) = numpy.einsum('m,smn,n->s',v,PEs,v)
       (eva,evb) = numpy.einsum('m,smn,n->s',v,PvEs,v)
       if(noa>0.000001 and nob>0.000001 and nva>0.000001 and nvb>0.000001):
          eoa = eoa/(noa +0.00000001)
          eob = eob/(nob +0.00000001)
          eva = eva/(nva +0.00000001)
          evb = evb/(nvb +0.00000001)
          print('Occupancies ',ishell,iproj,noa,nob)
          print('Virt occs   ',nva,nvb)
          print('Proj Occ Energies    ',eoa,eob)
          print('Proj Virt Energies',eva,evb)
          print('Self energy ',J)

          # Diagonalize 
          o = J*(noa*nob*nva*nvb)**0.5 # <Phi_0|Vp(r)|Phi_oo^vv> 
          o = numpy.maximum(o,1e-10)
          ee=(eva+evb-eoa-eob)          
          d = (ee + J*(noa*nob+nva*nvb-noa*nvb-nob*nva))/2
          ec = d-(d**2+o**2)**0.5 
          print('Hamiltonian o and d and ec',o,d,ec)
          ecs[itot]=ec
  wts1 = 1.0/wts
  wts2 = 1.0/(ecs*wts)
  EC1 = numpy.einsum('s,s->',ecs,wts1) 
  EC2 = numpy.einsum('s,s->',ecs**2,wts2) 
  return(EC1,EC2)
     
