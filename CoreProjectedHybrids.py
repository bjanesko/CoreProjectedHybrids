from pyscf import scf,gto,dft
import numpy 
from scipy import linalg 
import sys 

# This file implements core-projected hybrids, and tests them for a dataset of
# core IP and valence properties 

# Construct projection operator P onto atomic orbitals AOs 
# and its complement Q
def AOProj(mf,aos): 
  S = mf.get_ovlp()
  N=(S.shape)[0]
  n=len(aos)
  # Build the set of complement AOs 
  val=[]
  for i in range(N):
    if(not (i in aos)):
      val.append(i) 
  nv=len(val)
  P = numpy.zeros((N,N))
  Q = numpy.zeros((N,N))

  # Build the projector P 
  P = numpy.zeros((N,N))
  s =  numpy.zeros((n,n))
  for i in range(n):
    for j in range(n):
      s[i,j]=S[aos[i],aos[j]]
  sm=linalg.inv(s)
  for i in range(n):
    for j in range(n):
      P[aos[i],aos[j]]=sm[i,j]

  if(nv>0):
    # Build the nv x nv overlap of the projected valence AOs
    # Note SPSPS = SPS 
    Qt = numpy.zeros((N,N))
    SPS = numpy.dot(S,numpy.dot(P,S))
    St = S -SPS 
    stv = numpy.zeros((nv,nv))
    for i in range(nv):
      for j in range(nv):
        stv[i,j]=St[val[i],val[j]]

    # Inverse of this matrix gives projector Qt
    # in the basis of projected valence AOs 
    stvm=linalg.inv(stv)
    for i in range(nv):
      for j in range(nv):
        Qt[val[i],val[j]]=stvm[i,j]

    # Convert from projected valence AOs to unprojected AOs 
    PSQ = numpy.dot(P,numpy.dot(S,Qt))
    QSP = numpy.dot(Qt,numpy.dot(S,P))
    PSQSP = numpy.dot(P,numpy.dot(S,QSP))
    Q = Qt - PSQ - QSP + PSQSP 
  
  # Debug 
  #print('P',P)
  #print('Q',Q)
  #print('P.S.P-P',numpy.dot(P,numpy.dot(S,P))-P)
  #print('Q.S.Q-Q',numpy.dot(Q,numpy.dot(S,Q))-Q)
  #print('P.S.Q',numpy.dot(P,numpy.dot(S,Q)))
  return((P,Q))

# Project AO-basis density matrices PA,PB using projection operator Q
# Return projected density matrices and exchange energies 
def ProjDM2(mf,PA,PB,Q): 
  S = mf.get_ovlp()
  PpA = numpy.dot(Q,numpy.dot(S,numpy.dot(PA,numpy.dot(S,Q))))
  PpB = numpy.dot(Q,numpy.dot(S,numpy.dot(PB,numpy.dot(S,Q))))
  NA = numpy.einsum('mn,nm->',PpA,S)
  NB = numpy.einsum('mn,nm->',PpB,S)
  KA = -mf.get_k(dm=PpA)
  KB = -mf.get_k(dm=PpB)
  ExA = 0.5*numpy.einsum('mn,nm->',PpA,KA)
  ExB = 0.5*numpy.einsum('mn,nm->',PpB,KB)
  # Updated 9/10/2022 to include second projection in potential, from the second 1PDM entering exchange energy 
  KpA = numpy.dot(S,numpy.dot(Q,numpy.dot(KA,numpy.dot(Q,S))))
  KpB = numpy.dot(S,numpy.dot(Q,numpy.dot(KB,numpy.dot(Q,S))))
  ExA2 = 0.5*numpy.einsum('mn,nm->',PA,KpA)
  ExB2 = 0.5*numpy.einsum('mn,nm->',PB,KpB)
  print('Proj: ',NA,NB,ExA,ExB,ExA2,ExB2)
  return(PpA,PpB,ExA,ExB,KpA,KpB)

def AssignCores(m):
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
    #Tcut = acut*1.5 * 0.9 
    Tcut = acut*1.5 * 1.4
    #Tcut = acut*1.5 * 0.60
    Tval = T[iao,iao]
    print("AO ",iao," ",labs[iao]," center ",icen," atom charge ",iat," T ",Tval," Tcut ",Tcut)
    #if(Tval>Tcut and iat>2 and ('s ' in labs[iao] or 'px ' in labs[iao] or 'py ' in labs[iao] or 'pz ' in labs[iao]) ):
    if(Tval>Tcut and iat>2 and ('s ' in labs[iao] )):
      CoreAOs.append(iao)

  CoreAOs = [*set(CoreAOs)] # Remove duplicates 
  return(CoreAOs)

# Core and HOMO IP, HOMO-LUMO gap, and total (atomization) energies 
# 14 molecules from Tu2007 
# For each, compare HF, B3LYP, and B3LYP-valence/HF-core
# Generalize to do 100% 'dynamical' correlation and a hybrid in exchange 
# Total and Orbital energies computed with HF densities 
# 6-311G(2d,2p) basis set, ONE core AO per atom. 
# B3LYP/6-311G(2d,2p) geometries from Gaussian 
# All molecules are closed shell singlets 
# Erefs are B3LYP/6-311G(2d,2p) from Gaussian
# Refs are the N lowest orbital energies converted to core IP (eV) 
# 15 Second row molecules are MP2 references from Besley2021
#b='6-311G**'
a2eV=27.211 
names=['CO','H2O','CH4','CH3CN','CH3COOH','Glycine','MBO','PhCH3','PhNH2','PhOH','PhF','C2H2','C2H4','C2H6',
'AlH3','AlH2Cl','AlH2F','SiH4','H3SiOH','H3SiCl','PH3','H3PO','H2POOH','CH3SH','H2CS','H2S','CH3Cl','HCOCl','HCl',
'H','C','N','O','F',  'Al','Si','P','S','Cl']
spins=[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,
0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 
1,2,3,2,1, 1,2,3,2,1]

geoms=['''O 0.0 0.0 0.0; C 0.0 0.0 1.12696''', # CO 
'''O 0.0 0.0 0.11869; H 0. -0.75705 -0.47476;  H 0.  0.75705 -0.47476''', # H2O 
'''C 0.0 0.0 0.0; H  0.6296  0.6296  0.6296;  H -0.6296 -0.6296  0.6296;  H -0.6296  0.6296 -0.6296;  H  0.6296 -0.6296 -0.6296''', # CH4
'''N 1.432840    0.000015   -0.000024;C -1.175750    0.000009   -0.000008;H -1.553340    0.286693   -0.983563;H -1.553539    0.708400    0.739996;H -1.553369   -0.995131    0.243417;C 0.280811   -0.000019    0.000061''', #CH3CN
'''C -1.393668   -0.118188    0.000008; H  -1.919443    0.833667    0.000001; H  -1.672621   -0.701695    0.880179; H  -1.672632   -0.701710   -0.880150; C   0.091790    0.128184   -0.000008; O   0.633780    1.202292   -0.000043; O   0.785247   -1.038745    0.000036; H  1.723754   -0.798614    0.000025''', # CH3COOH
''' N   -1.971034    0.011897    0.000043; H   -2.000667    0.626204   -0.808013; C  -0.726308   -0.732679   -0.000066; C   0.541944    0.111991   -0.000003; H  -0.684243   -1.393994    0.871220; H  -0.684284   -1.393714   -0.871574; O   0.578903    1.315451   -0.000026; H  -2.000694    0.625966    0.808279; O   1.653572   -0.660940    0.000046; H   2.413502   -0.059704    0.000046''', # Glycine 
''' C  0.573668    0.728182    0.00; C      0.543534   -0.665892    0.00; C     -1.584362    0.013880    0.00; H      1.813402    2.502221    0.00; C     1.776425    1.420079    0.00; C      1.686679   -1.437440    0.00; C      2.902759   -0.745331    0.00; C      2.944245    0.651654    0.00; H      1.635301   -2.518075    0.00; H      3.829390   -1.305887    0.00; H      3.903896    1.154169    0.00; O     -0.766871   -1.089621    0.00; N     -0.762824    1.109223    0.00; H     -1.134224    2.045299    0.00; S     -3.226920   -0.044756    0.00 ''', # MBO 
''' C  -1.199361    1.202807    0.002100; C      0.193649    1.200180   -0.008753; C      0.912392    0.000254   -0.011649; C      0.193938   -1.200014   -0.008756; C     -1.198933   -1.203040    0.002102; C     -1.901741   -0.000165    0.008322; H     -1.736086    2.145218    0.001706; H      0.731476    2.143088   -0.017391; H      0.732068   -2.142769   -0.017414; H     -1.735446   -2.145573    0.001718; H     -2.985868   -0.000363    0.013715; C      2.422586    0.000096    0.009301; H      2.801610   -0.013179    1.037342; H      2.828607   -0.877753   -0.499128; H      2.828462    0.890616   -0.476551 ''' , # PhCH3 
''' C     1.170277    1.200008    0.003309; C    -0.220222    1.205814   -0.004299; C    -0.938460    0.000264   -0.007323; C    -0.220103   -1.205759   -0.004224; C     1.170033   -1.200226    0.003217; C     1.879087    0.000024    0.007219; H     1.703130    2.144911    0.007092; H    -0.759369    2.148217   -0.012114; H    -0.759870   -2.147805   -0.012598; H     1.703090   -2.145013    0.007339; H     2.962460   -0.000144    0.013719; N    -2.333326   -0.000098   -0.076203; H    -2.779568   -0.836081    0.271643; H    -2.780256    0.835845    0.270944''' , # PhNH2
''' C  -1.169340   -1.188361    0.000029; C     0.220214   -1.221027    0.000019; C     0.940081   -0.024008    0.000003; C     0.262538    1.196887    0.000019; C    -1.130827    1.216460   -0.000015; C    -1.854571    0.027313   -0.000025; H    -1.721693   -2.121496    0.000045; H     0.765282   -2.157151   -0.000048; H     0.822567    2.127888    0.000113; H    -1.648422    2.169253   -0.000039; H    -2.937777    0.045685   -0.000040; H     2.677119    0.776944    0.000157; O     2.304295   -0.110589   -0.000046''' , # PhOH 
''' C   -1.134161   -1.206127    0.000004; C     0.259288   -1.214601    0.000003; C     0.928779   -0.000015   -0.000029; C     0.259290    1.214589   -0.000010; C    -1.134137    1.206147    0.000018; C    -1.833038    0.000003   -0.000008; H    -1.672281   -2.147023    0.000011; H     0.826617   -2.136896    0.000004; H     0.826674    2.136853   -0.000012; H    -1.672274    2.147031    0.000026; H    -2.916451    0.000024   -0.000013; F     2.281288    0.000003    0.000013 ''' , # PhF 
''' C   0.000000    0.000000    0.599138; C   0.000000    0.000000   -0.599138; H   0.000000    0.000000    1.662069; H   0.000000    0.000000   -1.662069''' , # C2H2
'''C       0.000000    0.000000    0.663456; C     0.000000    0.000000   -0.663456; H     0.000000    0.922494    1.234601; H     0.000000    0.922494   -1.234601; H     0.000000   -0.922494    1.234601; H     0.000000   -0.922494   -1.234601 ''' , # C2H4
'''C    0.000000    0.000000    0.765291; C     0.000000    0.000000   -0.765291; H     0.000000    1.018566    1.163701; H     0.882104    0.509283   -1.163701; H    -0.882104   -0.509283    1.163701; H     0.000000   -1.018566   -1.163701; H     0.882104   -0.509283    1.163701; H    -0.882104    0.509283   -1.163701 ''', # C2H6

''' Al        0.000000    0.000000    0.000000; H        0.000000    1.584152    0.000000; H       -1.371916   -0.792076    0.000000; H        1.371916   -0.792076    0.000000 ''', # AlH3 
'''Al        0.000000    0.000000   -1.079249; H        0.000000    1.398161   -1.797451; H        0.000000   -1.398161   -1.797451; Cl         0.000000    0.000000    1.036773''', # AlH2Cl 
'''Al    0.000000    0.000000    0.568932; H        0.000000    1.401918    1.281386; H        0.000000   -1.401918    1.281386; F       0.000000    0.000000   -1.106543  ''' , # AlH2F 

''' Si   0.000    0.000000    0.000000; H        0.856732    0.856732    0.856732; H       -0.856732   -0.856732    0.856732; H       -0.856732    0.856732   -0.856732; H        0.856732   -0.856732   -0.856732 ''' , # SiH4 
''' Si       0.534717   -0.008479    0.000000; H        1.039108    1.381821   -0.000002; H        1.050915   -0.725341    1.198630; H        1.050915   -0.725344   -1.198629; O       -1.122708    0.110209    0.000000; H       -1.645309   -0.694105    0.000000 ''' , # H3SiOH
'''Si     0.000000    0.000000   -1.000901; H        0.000000    1.405397   -1.459597; H        1.217110   -0.702699   -1.459597; H       -1.217110   -0.702699   -1.459597; Cl        0.000000    0.000000    1.081848 ''' , # H3SiCl 

'''P        0.000000    0.128161    0.000000; H        0.599079   -0.640924    1.037429; H       -1.198158   -0.640570    0.000000; H        0.599079   -0.640924   -1.037429 ''' , # PH3 
''' P     0.000000    0.000000    0.382254; H        0.000000    1.258672    1.041022; H       -1.090042   -0.629336    1.041022; H        1.090042   -0.629336    1.041022; O        0.000000    0.000000   -1.107109 ''' , # H3PO 
''' P      0.124683    0.362264    0.019251; H        0.052677    1.151385    1.193312; H       -0.035276    1.297757   -1.023525; O        1.296603   -0.540361   -0.033848; O       -1.352623   -0.305364   -0.077963; H       -1.439486   -1.117300    0.435928 ''' , # H2POOH 

'''S       0.667204   -0.087078    0.000000; H        0.908439    1.239663    0.000000; C       -1.165195    0.019886    0.000000; H       -1.527512   -1.007801   -0.000011; H       -1.532512    0.521024    0.894596; H       -1.532511    0.521042   -0.894586 ''', # CH3SH 
'''S        0.000000    0.000000    0.586602; C        0.000000    0.000000   -1.028472; H        0.000000    0.923556   -1.607400; H        0.000000   -0.923556   -1.607400 ''' , # H2CS 
'''S        0.000000    0.000000    0.103531; H     0.000000   -0.973611   -0.828251; H     0.000000    0.973611   -0.828251; ''', # H2S 
''' C       0.000000    0.000000   -1.141992; H        0.000000    1.032732   -1.481893; H        0.894372   -0.516366   -1.481893; H       -0.894372   -0.516366   -1.481893; Cl        0.000000    0.000000    0.664567 ''', # CH3Cl 
''' C      0.696974    0.423981    0.000000; H    0.750322    1.518957    0.000001; Cl   -1.039202   -0.079470    0.000000; O    1.591784   -0.338982    0.000000 ''' , # HCOCl 
'''     Cl 0.000000    0.000000    0.071493; H   000000    0.000000   -1.215382 ''' # HCl
]

refs=[ 
[542.10,295.50], # CO 
[539.93], # H2O 
[290.83], # CH4
[405.60,292.98,292.44], # CH3CN 
[540.09,538.36,295.38,291.55], # CH3COOH 
[540.20,538.40,405.40,295.30,295.20], # Glycine(note reorder)
[2446.92,540.58,407.01,295.71,293.91,293.01,291.67,291.45], # MBO (note reorder)
[290.90,290.10], # PhCH3 
[405.30,291.20], # PhNH2
[538.90,292.00], # PhOH 
[693.30,292.90], # PhF
[291.20], #C2H2 
[290.70], #C2H4 
[290.60],  #C2H6 

[1565.07], # AlH3 
[-1,1565.83], # AlH2Cl skip the Cl 
[1566.01], # AlH2F 
[1843.19], # SiH4 
[1843.96], # SiH3OH
[-1,1844.30], # SiH3Cl skopt the Cl 
[2145.78], # PH3 
[2148.33], # H3PO
[2149.06], # H2POOH 
[2471.02], # CH3SH 
[2471.15], # H2CS 
[2471.73], # H2S 
[2820.31,1.00], # CH3Cl
[2820.60], # HCOCl 
[2821.44]  # HCl 
]

def myAddHFX(m,ni,EXCSL,FSL,xcstr,PA,PB,KA,KB):
  omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xcstr, spin=m.spin)
  if(abs(hyb)>1e-10):
    print('Hybrid coeff: ',hyb)
    #print('Hybrid EXA',.5*hyb*numpy.einsum('ij,ji->',PA,KA))
    EXCSL +=.5*hyb*numpy.einsum('ij,ji->',PA,KA)
    EXCSL +=.5*hyb*numpy.einsum('ij,ji->',PB,KB)
    FSL[0] += hyb*KA
    FSL[1] += hyb*KB
  return(EXCSL,FSL) 

def CoreProj(myhf,Fother,Eother,PA,PB,CoreAOs,xcstr,acore): 
  # Return four total energies and four sets of orbital energies: 
  # Reference, semilocal, and two variants of
  # reference-core/semilocal-valence 
  # Eother is one-electron, Hartree, and nuclear energy, Fother is the
  # corresponding Fock matrix contributions
  m = myhf.mol
  S = myhf.get_ovlp() 
  df=dft.UKS(m) 
  df.grids.level=6
  ni = df._numint

  # Terms from the full density matrices 
  KA = -myhf.get_k(dm=PA)
  KB = -myhf.get_k(dm=PB)
  EXHF = 0.5*(numpy.einsum('ij,ji->',PA,KA)+numpy.einsum('ij,ji->',PB,KB))
  FHF = numpy.array(([Fother+KA,Fother+KB]))
  (valsHF,vecsHF)=myhf.eig(FHF,S)
  #print('Real HF MO energies: ',myhf.mo_energy[0])
  #print('Test HF MO energies: ',valsHF[0])

  # The projection should subtract off only the semilocal exchange piece, and
  # keep 100% of the semilocal correlation.
  N,EXSL,VXSL = ni.nr_uks(m,df.grids,xcstr,(PA,PB))
  (EXSL,VXSL) = myAddHFX(m,ni,EXSL,VXSL,xcstr,PA,PB,KA,KB)  # Add global hybrid 
  ECSL = 0.0 
  VCSL = 0.0*VXSL 
  xstr = xcstr 
  if(',' in xcstr):
    [xstr,cstr]= xcstr.split(',')
    xstr = xstr + ','
    cstr = ','+cstr 
    N,EXSL,VXSL = ni.nr_uks(m,df.grids,xstr,(PA,PB))
    (EXSL,VCSL) = myAddHFX(m,ni,EXSL,VXSL,xstr,PA,PB,KA,KB)  # Add global hybrid 
    N,ECSL,VCSL = ni.nr_uks(m,df.grids,cstr,(PA,PB))
    print('Projecting with ',xstr,' projected exchange and full ',cstr,' correlation ')
    print('Full exchange: ',EXSL,' Full correlation: ',ECSL)

  FSL = numpy.array(([Fother+VXSL[0]+VCSL[0],Fother+VXSL[1]+VCSL[1]]))
  (valsSL,vecsSL)=myhf.eig(FSL,S)
  print('HF  energy: ',Eother+EXHF)
  print('DFT energy: ',Eother+EXSL+ECSL)
  print('DFT MO energies: ',valsSL[0])

  # Projection into core AOs and onto valence AOs
  (P,Q) = AOProj(myhf,CoreAOs) 

  # Project into core AOs. SL in core-valence 
  (PAproj,PBproj,EXAproj,EXBproj,KAproj,KBproj) = ProjDM2(myhf,PA,PB,P)
  Nproj,EXSLproj,VXSLproj = ni.nr_uks(m,df.grids,xstr,(PAproj,PBproj)) # Semilocal in projected 
  VXSLproj[0] = numpy.dot(S,numpy.dot(P,numpy.dot(VXSLproj[0],numpy.dot(P,S))))
  VXSLproj[1] = numpy.dot(S,numpy.dot(P,numpy.dot(VXSLproj[1],numpy.dot(P,S))))
  (EXSLproj,VXSLproj) = myAddHFX(m,ni,EXSLproj,VXSLproj,xstr,PAproj,PBproj,KAproj,KBproj)  # Add global hybrid in projected 
  #print('KA',KA)
  #print('KAproj',KAproj)
  #print('VXSL',VXSL[0])
  #print('VXSLproj',VXSLproj[0])

  # Replace a factor of acore of SLproj with HFproj 
  Fproj=numpy.array(([Fother+VXSL[0]+VCSL[0]+acore*(KAproj-VXSLproj[0]),Fother+VXSL[1]+VCSL[1]+acore*(KBproj-VXSLproj[1]) ]))
  Eproj1 = Eother + EXSL+ECSL+ acore*(EXAproj+EXBproj-EXSLproj)
  (valsproj1,vecsproj)=myhf.eig(Fproj,S)
  print('Projected MO energies: ',valsproj1[0])

  # Project into non-core AOs. HF in core-valence
  valsproj2=0.0 * valsproj1
  Eproj2 = 0.0 
  (PAproj,PBproj,EXAproj,EXBproj,KAproj,KBproj) = ProjDM2(myhf,PA,PB,Q)
  Nproj,EXSLproj,VXSLproj = ni.nr_uks(m,df.grids,xstr,(PAproj,PBproj)) # SL in projected 
  VXSLproj[0] = numpy.dot(S,numpy.dot(Q,numpy.dot(VXSLproj[0],numpy.dot(Q,S))))
  VXSLproj[1] = numpy.dot(S,numpy.dot(Q,numpy.dot(VXSLproj[1],numpy.dot(Q,S))))
  (EXSLproj,VXSLproj) = myAddHFX(m,ni,EXSLproj,VXSLproj,xstr,PAproj,PBproj,KAproj,KBproj)  # Add global hybrid in projected 

  # Replace SL with a factor of acore* HF out of projected
  Eproj2 = Eother + ECSL +EXSL + acore*(EXHF-EXAproj-EXBproj) - acore*(EXSL-EXSLproj)
  Fproj=numpy.array(([Fother+VXSL[0]+VCSL[0]+acore*(KA-KAproj)-acore*(VXSL[0]-VXSLproj[0]),Fother+VXSL[1]+VCSL[1]+acore*(KB-KBproj)-acore*(VXSL[1]-VXSLproj[1])]))
  (valsproj2,vecsproj)=myhf.eig(Fproj,S)
  print('Projected MO energies: ',valsproj2[0])

  ret = [Eother+EXHF,Eother+EXSL+ECSL,Eproj1,Eproj2,valsHF[0],valsSL[0],valsproj1[0],valsproj2[0]]
  #print("Here are the returned values from core : \n",ret)
  return(ret)  


if __name__=='__main__':
  imol = int(sys.argv[1])
  xcstr=sys.argv[2]
  b=sys.argv[3]
  xcstr = xcstr.replace("COMMA",",")
  b = b.replace("COMMA",",")
  acore=float(sys.argv[4]) # Fraction of exact exchage in cores 
  name=names[imol]
  geom = name 
  sp = spins[imol]
  ref=[0]
  print("Testing Molecule ",name)
  if(imol<len(geoms)):
    geom = geoms[imol]
    ref = refs[imol]
  print("Geometry: ",geom)

  # Build molecule 
  m=gto.Mole(atom=geom,spin=sp,basis=b)
  m.build()

  # Find the core AOs
  AllAOs = range(m.nao)
  CoreAOs= AssignCores(m)
  labs = m.ao_labels()
  print(" All AOs: ",labs)
  print(" Core AOs: ",[labs[i] for i in CoreAOs])
  print(" Core AOs: ",CoreAOs)

  # Do UHF 
  NA = m.nelec[0]
  myhf=scf.UHF(m)
  myhf.kernel() 
  hcore = myhf.get_hcore()
  S=myhf.get_ovlp()

  # Ground state densities 
  (PA0,PB0) = myhf.make_rdm1() 
  Fother0 = hcore+myhf.get_j(dm=PA0+PB0)
  Eother0= numpy.einsum('ij,ji->',PA0+PB0,hcore+myhf.get_j(dm=PA0+PB0)/2) + myhf.energy_nuc() 
  
  # Do the projection 
  (EHF,ESL,Eproj1,Eproj2,eHF,eSL,eproj1,eproj2) = CoreProj(myhf,Fother0,Eother0,PA0,PB0,CoreAOs,xcstr,acore)
  (EHF,ESL,Eproj3,Eproj4,eHF,eSL,eproj3,eproj4) = CoreProj(myhf,Fother0,Eother0,PA0,PB0,AllAOs,xcstr,acore)
  #Eproj3=0
  #eproj3=0*eproj2
  #print("HF: ",eHF)
  #$print("SL: ",eSL)
  #print("P1: ",eproj1)
  #print("P2: ",eproj2) 
  
  # Print the results 
  res = "Res,%s,%s,%s,%.2f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f" %(names[imol],xcstr,b,acore,m.nao,len(CoreAOs),EHF,ESL,Eproj1,Eproj2,Eproj3)
  res= res+",%.2f,%.2f,%.2f,%.2f,%.2f" %  (-a2eV*eHF[NA-1],-a2eV*eSL[NA-1],-a2eV*eproj1[NA-1],-a2eV*eproj2[NA-1],-a2eV*eproj3[NA-1])
  # a2eV*(eHF[NA]-eHF[NA-1]), a2eV*(eSL[NA]-eSL[NA-1]), a2eV*(eproj1[NA]-eproj1[NA-1]), a2eV*(eproj2[NA]-eproj2[NA-1]) ) 
  for i in range(len(ref)):
     if(ref[i]>0):
       res = res + ",%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % ( ref[i], -a2eV*eHF[i], -a2eV*eSL[i], -a2eV*eproj1[i], -a2eV*eproj2[i], -a2eV*eproj3[i] )
  print(res)
  
