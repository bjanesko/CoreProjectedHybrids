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
    ''' Build the projection operators QS and SQ from an input list of orthonormal MOs 
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
    ks.QS=numpy.einsum('ik,kj->ij',Q,S)
    ks.SQ=numpy.einsum('ik,kj->ij',S,Q)
    # DEBUG TEST 
    test = numpy.dot(Q,numpy.dot(S,Q))-Q
    print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q))

def build_mbproj(ks):
    ''' Build the projection operators QS and SQ from existing basis core AOs 
    Args:
        ks : an instance of :class:`RKS`
    '''  
    if(ks.QS is None):
      m = ks.mol
      S = ks.get_ovlp() 
      Sm = linalg.inv(S)
      N=(S.shape)[0]
      Q    =numpy.zeros((N,N))
      ks.QS=numpy.zeros((N,N))
      ks.SQ=numpy.zeros((N,N))
      # Prepare dummy molecule with minimal basis set 
      md = m.copy()
      md.basis='321g'   
      md.build() 
      SM = md.intor_symmetric('int1e_ovlp')
      
      # Build cross-overlap matrix between current and minimal core 
      SX = gto.intor_cross('int1e_ovlp',m,md)
      
      # Find core minimal basis AOs 
      coreaos=[]
      labs = md.ao_labels()
      for iao in range(md.nao):
        icen = int(labs[iao].split()[0])
        iat = md.atom_charge(icen)
        if(iat>2 and   (' 1s' in labs[iao]) ):
           coreaos.append(iao)
           print(labs[iao])
        if(iat>10 and   (' 2s' in labs[iao] or ' 2p' in labs[iao]) ):
           coreaos.append(iao)
           print(labs[iao])

      # Build core minimal basis 
      NC = len(coreaos)
      if(NC>0):
        SC = numpy.zeros((NC,NC))
        SXC = numpy.zeros((N,NC))
        for ic in range(NC):
          SXC[:,ic] = SX[:,coreaos[ic]]
          for jc in range(NC):
             SC[ic,jc] = SM[coreaos[ic],coreaos[jc]]
        SCm= linalg.inv(SC)
      
        # Core AO projection operator in current basis set 
        #Q = numpy.einsum('ia,ab,bc,dc,dj->ij',Sm,SXC,SCm,SXC,Sm)
        SmSXC= numpy.dot(Sm,SXC)
        t2 = numpy.dot(SmSXC,SCm)
        Q = numpy.dot(t2,numpy.transpose(SmSXC))
        ks.QS=numpy.einsum('ik,kj->ij',Q,S)
        ks.SQ=numpy.einsum('ik,kj->ij',S,Q)

      # DEBUG TEST 
      test = numpy.dot(Q,numpy.dot(S,Q))-Q
      print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q))

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
           print(labs[iao])

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
      print('SC ',SC)
      print('SXC ',SXC)
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
        aoss = [] 
        if('CoreAOs' in ks.paos):
          aoss = assign_cores(ks.mol)
        elif('DAOs' in ks.paos):
          aoss = get_d(ks.mol)
        elif('AllAOs' in ks.paos):
          for i in range(ks.mol.nao):
            aoss.append(i) 
      elif(isinstance(ks.paos,list)):
        aoss = ks.paos 
      else:
        if(len(ks.paos.shape) == 2):
          mo_proj(ks)
          return 
        else:
          raise Exception('Not sure what paos is')

      # If we are still here, aoss contains a list of AOs
      # to project onto. Orthogonalize and project. 
      print('AOs to project ',aoss)
      S = ks.get_ovlp() 
      N=(S.shape)[0]
      Q    =numpy.zeros((N,N))
      ks.QS=numpy.zeros((N,N))
      ks.SQ=numpy.zeros((N,N))

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
      for i in aoss:
        v = Qbig[i]
        Q = Q + numpy.outer(v,v)
      ks.QS=numpy.einsum('ik,kj->ij',Q,S)
      ks.SQ=numpy.einsum('ik,kj->ij',S,Q)
 
      # DEBUG TEST 
      test = numpy.dot(Q,numpy.dot(S,Q))-Q
      print('TEST: ',numpy.sum(test*test),numpy.einsum('ij,ji->',S,Q))
