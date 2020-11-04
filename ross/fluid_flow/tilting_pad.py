"""Tilting-pad hydrodynamic bearing properties calculation module

In this module, the tilting-pad bearing properties can be estimated and
simulated for a given set of properties, until the model reaches a steady-
-state of numerical convergence. These properties can then be exported to
external files, and used in the bearing_seal_element.py routine inside the
ROSS framework. This routine can also be used for bearing properties esti-
mation for general purposes.
"""


import os
import warnings

import numpy as np
import toml
from plotly import graph_objects as go
from scipy import interpolate as interpolate

from ross.element import Element
from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow.fluid_flow_coefficients import (
    calculate_short_damping_matrix, calculate_short_stiffness_matrix)
from ross.units import Q_, check_units
from ross.utils import read_table_file



# __all__ = [
# ]

# FOR THE MOMENT, THIS CODE IS BEING TRANSLATED FROM MATLAB TO PYTHON, AND
# THEREFORE THIS PROCESS WILL BE MADE IN LINEAR PROGRAMMING STANDARDS. ONCE
# THIS INITIAL TRANSLATION IS DONE, AND THE CODE VALIDATED, THE MODULARIZATION 
# AND CONVERSION TO OBJECT ORIENTED STANDARDS WILL BE MADE.


#class _TiltingPad:
#    """Tilting-pad hydrodynamic bearing properties calculation class.
#
#    This class takes the tilting-pad bearing geometrical and physical properties, to
#    determine it's influence in the rotor system's dynamic response. It also generates
#    the bearings stiffness depending on the rotor rotation speed.
#
#    STILL HAS TO CHANGE THE REST FROM HERE DOWN
#    """
#
#    def __init__(self, coefficient, frequency=None):
        


"""
P1
T1
XH
YH
E
phi
# Cr
# Tcuba
Xtheta
XZdim
ntheta
Fhx
Fhy
Pmax
hmin
Tmax
"""

# optim values from legacy codes
# x=[0.000321298907440948 0.000218101024776208 0.000241891712348458 0.000385504446042090 0.000516992650533115 0.000460227890390222];

psi_pad = np.array([x[1], x[2], x[3], x[4], x[5], x[6]])
npad = 6

# Radial clearance
Cr = 250e-6

# Oil tank temperature
Tcuba = 40

# Rotor center position
xx = E * Cr * np.cos(phi)
yy = E * Cr * np.sin(phi)
alphapt = 0 * (2 * pi * 5) * alpha


# Geometric parameters for the bearing --------------------------------------------

# Journal radius
R = 0.5 * 930e-3

# Pad radius
Rs=0.5*934e-3 # [m]

# Pad thickness
esp=67e-3 # [m]

# Pad arc
betha_s=25 # [degree]
betha_s=betha_s*(np.pi/180) # [rad]

# Pivot position (arc pivot/arc pad)
rp_pad=0.6

# Bength of bearing
L=197e-3 # [m]

# Angular position of the pivot
sigma = np.array([0,300,60]) # [degree]
sigma= sigma*(np.pi/180) # [rad]

# Bearing loading 
fR=90.6e3 # [N]

# Rotor speed
wa=300 # [rpm]
war=wa*(np.pi/30) # rad/s

# Reference temperature
T_ref=Tcuba # [Celsius]

# Thermal properties for the oil ----------------------------------------------------

# Thermal conductivity
kt=0.07031*np.exp(484.1/(Tcuba+273.15+474)) # [J/s.m.C]

# Specific heat
Cp=(16.5*np.exp(-2442/(Tcuba+273.15+829.1)))*1e3 # [J/kgC]

# Specific mass
rho=0.04514*np.exp(9103/(Tcuba+273.15+2766))*1e3 # [kg/m**2]

# Reference viscosity
#mi_ref=0.0752
mi_ref=5.506e-09*np.exp(5012/(Tcuba+273.15+0.1248)) # [N.s/m**2]

# Bearing Position ---------------------------------------------------------

# Rotor center velocity
xpt=-(2*np.pi*5)*yy
ypt=(2*np.pi*5)*xx

#  Discretized Mesh ------------------------------------------------------

# Number of volumes in theta direction
ntheta=48

# Number of volumes in x direction
nX=ntheta

# Number of volumes in z direction
nZ=48

# Number of volumes in neta direction
nN=30

Z1=0 # initial coordinate z dimensionless
Z2=1 # final coordinate z dimensionless
dZ=1/(nZ) # differential z dimensionless
dz=dZ*L # differential z dimensional: [m]
XZ[1]=Z1
XZ[nZ+2]=Z2

XZ[1:nZ] = Z1+0.5* np.array([dz, Z2-0.5*dZ, dz]) # vector z dimensionless

XZdim=XZ*L # vector z dimensional [m]

N1=0 # initial coordinate netha dimensionless
N2=1 # final coordinate netha dimensionless
dN=1/(nN) # differential netha dimensionless
netha[1]=N1
netha[nN+2]=N2

netha[1:nN]=N1+0.5* np.array([dN, N2-0.5*dN, dN]) # vector netha dimensionless

theta1=-(rp_pad)*betha_s # initial coordinate theta [rad]
theta2=(1-rp_pad)*betha_s # final coordinate theta [rad]
dtheta=betha_s/(ntheta) # differential theta [rad]
Xtheta[0]=theta1
Xtheta[ntheta+1]=theta2

Xtheta[1:ntheta]=theta1+0.5* np.array([dtheta, theta2-0.5*dtheta, dtheta]) # vector theta [rad]

dX=1/nX # differential x dimensionless
dx=dX*(betha_s*Rs) # differential x dimensional: [m]
XX=Xtheta*Rs # vector x dimensional: [m

# Pad recess
len_betha=0.39*betha_s # Pad Angle with recess
len_L=0.71*L # Bearing length with recess
center_pos_L=L/2
start_pos_betha=0*betha_s
drop_pressure_pos_L=np.array([center_pos_L-len_L/2, center_pos_L+len_L/2])
drop_pressure_pos_betha=np.array([start_pos_betha, start_pos_betha+len_betha])

#drop_pressure_Ele_nZ=find(XZdim>drop_pressure_pos_L(1) & XZdim<drop_pressure_pos_L(2))
#drop_pressure_Ele_ntetha=find(Xtheta>=drop_pressure_pos_betha(1)+theta1 & Xtheta<=drop_pressure_pos_betha(2)+theta1)

# Initial Parameters ---------------------------------------------------------------

Tmist=T_ref*np.ones((nN+2))
minovo=mi_ref*np.ones((nZ,ntheta,nN)) # Initial viscosity field

# Velocity field  - 3D
vu=np.zeros((nZ,ntheta,nN))
vv=np.zeros((nZ,ntheta,nN))
vw=np.zeros((nZ,ntheta,nN))

# Velocity field - 2D
Vu=np.zeros((nN,ntheta))
Vv=np.zeros((nN,ntheta))
Vw=np.zeros((nN,ntheta))

YH = np.zeros[nN+2,nX+2,npad]

XH = np.zeros[nN+2,nX+2]


for ii in range(1,nX+2+1):
    XH[:,ii] = Xtheta[ii]

# Loop on the pads ==========================================================================
for n_p in range(1,npad+1): 
    alpha=psi_pad[n_p]

    # transformation of coordinates
    xryr=np.array([[np.cos(sigma(n_p))  , np.sin(sigma(n_p))]; 
                   [-np.sin(sigma(n_p)) , np.cos(sigma(n_p))]]) * np.array([[xx]; [yy]])

    xryrpt=np.array([[np.cos(sigma(n_p))  , np.sin(sigma(n_p))]; 
                     [-np.sin(sigma(n_p)) , np.cos(sigma(n_p))]]) * np.array([[xpt]; [ypt]])

    xr=xryr[0]
    yr=xryr[1]
    
    xrpt=xryrpt[0]
    yrpt=xryrpt[1]
    
    # Temperature matrix with boundary conditions ====================================
    T_novo=T_ref*np.ones(nN+2,ntheta+2)
    
    Tcomp=1.2*T_novo
    
    # Oil temperature field loop ================================================================
    while (np.linalg.norm((T_novo-Tcomp) / np.linalg.norm(Tcomp)) > 0.01):
    
        mi=minovo
        Tcomp=T_novo
        
        nk=nZ*ntheta
        K_null=np.zeros[0,nk]
        Kij_null=np.zeros[nZ,ntheta]
        ki=1
        kj=1
        k=0; # indice utilizado na vetoriza��o da press�o
        nn=1
        
        Mat_coef=np.zeros[nk,nk]
        b=np.zeros[nk]
        
        # Mesh loop in Z direction ====================================================
        for ii in range((Z1+0.5*dZ), dZ, (Z2-0.5*dZ)):
            
            # Mesh loop in THETA direction ====================================================
            for jj in range((theta1+0.5*dtheta), dtheta, (theta2-0.5*dtheta)):
                
                if kj==1 :
                    vector_mi[0,]=mi[ki,kj,]
                    vector_mi[1,]=mi[ki,kj+1,]
                    vector_mi[2,]=mi[ki,kj,]
                
                if kj==ntheta :
                    vector_mi[0,]=mi[ki,kj,]
                    vector_mi[1,]=mi[ki,kj,]
                    vector_mi[2,]=mi[ki,kj-1,]
                
                if (kj>1 and kj<ntheta) :
                    vector_mi[0,]=mi[ki,kj,]
                    vector_mi[1,]=mi[ki,kj+1,]
                    vector_mi[2,]=mi[ki,kj-1,]
                
                # Loop in N 
                for kk in range(N1+0.5*dN, dN, N2-0.5*dN):
                    
                    mi_adP=vector_mi(1,nN+1-nn)/mi_ref
                    mi_adE=vector_mi(2,nN+1-nn)/mi_ref
                    mi_adW=vector_mi(3,nN+1-nn)/mi_ref
                    
                    auxFF0P[nn+1]=(1/mi_adP)
                    auxFF1P[nn+1]=(kk/mi_adP)
                    auxFF0E[nn+1]=(1/mi_adE)
                    auxFF1E[nn+1]=(kk/mi_adE)
                    auxFF0W[nn+1]=(1/mi_adW)
                    auxFF1W[nn+1]=(kk/mi_adW)
                    
                    nn=nn+1
                

                nn=1
                
                auxFF0P[0]=auxFF0P[1]
                auxFF0P[nN+1]=auxFF0P[nN]
                
                auxFF1P[0]=0
                auxFF1P[nN+1]=(N2/(vector_mi[0,nN]/mi_ref))
                
                auxFF0E[0]=auxFF0E[1]
                auxFF0E[nN+1]=auxFF0E[nN]

                auxFF1E[0]=0
                auxFF1E[nN+1]=(N2/(vector_mi[1,nN]/mi_ref))
                
                auxFF0W[0]=auxFF0W[1]
                auxFF0W[nN+1]=auxFF0W[nN]
                
                auxFF1W[0]=0
                auxFF1W[nN+1]=(N2/(vector_mi[2,nN]/mi_ref))
                
                FF0P=0.5*np.sum((netha[1:]-netha[0:-2]).*(auxFF0P[1:]+auxFF0P[0:-2]))
                FF1P=0.5*np.sum((netha[1:]-netha[0:-2]).*(auxFF1P[1:]+auxFF1P[0:-2]))
                FF0E=0.5*np.sum((netha[1:]-netha[0:-2]).*(auxFF0E[1:]+auxFF0E[0:-2]))
                FF1E=0.5*np.sum((netha[1:]-netha[0:-2]).*(auxFF1E[1:]+auxFF1E[0:-2]))
                FF0W=0.5*np.sum((netha[1:]-netha[0:-2]).*(auxFF0W[1:]+auxFF0W[0:-2]))
                FF1W=0.5*np.sum((netha[1:]-netha[0:-2]).*(auxFF1W[1:]+auxFF1W[0:-2]))
                
                FF0e=0.5*(FF0P+FF0E)
                FF0w=0.5*(FF0P+FF0W)
                FF1e=0.5*(FF1P+FF1E)
                FF1w=0.5*(FF1P+FF1W)
                
                # Loop in N 
                for kk in range(N1+0.5*dN, dN, N2-0.5*dN):
                    
                    mi_adP=vector_mi[0,nN-nn]/mi_ref
                    mi_adE=vector_mi[1,nN-nn]/mi_ref
                    mi_adW=vector_mi[2,nN-nn]/mi_ref
                    
                    auxFF2P[nn]=(kk/mi_adP)*(kk-FF1P/FF0P)
                    auxFF2E[nn]=(kk/mi_adE)*(kk-FF1E/FF0E)
                    auxFF2W[nn]=(kk/mi_adW)*(kk-FF1W/FF0W)
                    nn=nn+1
                
                nn=1
                
                auxFF2P[0]=0
                auxFF2P[nN+1]=(N2/(vector_mi[0,nN]/mi_ref))*(N2-FF1P/FF0P)
                
                auxFF2E[0]=0
                auxFF2E[nN+1]=(N2/(vector_mi[1,nN]/mi_ref))*(N2-FF1P/FF0P)
                
                auxFF2W[0]=0
                auxFF2W[nN+1]=(N2/(vector_mi[2,nN]/mi_ref))*(N2-FF1P/FF0P)
                
                # integration process ===================================================
                FF2P=0.5*np.sum((netha[1:]-netha[0:-2]).*(auxFF2P[1:]+auxFF2P[0:-2]))
                FF2E=0.5*np.sum((netha[1:]-netha[0:-2]).*(auxFF2E[1:]+auxFF2E[0:-2]))
                FF2W=0.5*np.sum((netha[1:]-netha[0:-2]).*(auxFF2W[1:]+auxFF2W[0:-2]))
                
                FF2e=0.5*(FF2P+FF2E)
                FF2w=0.5*(FF2P+FF2W)
                FF2n=FF2P
                FF2s=FF2n
                

                # Admensional oil film thickness ========================================
                hP=(Rs-R-(np.sin(jj)*(yr+alpha*(Rs+esp))+np.cos(jj)*(xr+Rs-R-Cr)))/Cr
                he=(Rs-R-(np.sin(jj+0.5*dtheta)*(yr+alpha*(Rs+esp))+np.cos(jj+0.5*dtheta)*(xr+Rs-R-Cr)))/Cr
                hw=(Rs-R-(np.sin(jj-0.5*dtheta)*(yr+alpha*(Rs+esp))+np.cos(jj-0.5*dtheta)*(xr+Rs-R-Cr)))/Cr
                hn=hP
                hs=hn
                hpt=-(1/(Cr*war))*(np.cos(jj)*xrpt+np.sin(jj)*yrpt+np.sin(jj)*(Rs+esp)*alphapt) # admensional
                
                CE=1/(betha_s)**2*(FF2e*he**3)*dZ/dX
                CW=1/(betha_s)**2*(FF2w*hw**3)*dZ/dX
                CN=(FF2n*hn**3)*(dX/dZ)*(Rs/L)**2
                CS=(FF2s*hs**3)*(dX/dZ)*(Rs/L)**2
                CP=-(CE+CW+CN+CS)
                B=(R/(Rs*betha_s))*dZ*(he*(1-FF1e/FF0e)-hw*(1-FF1w/FF0w))+hpt*dX*dZ
                k=k+1
                b[k,0]=B
                hhh[k,n_p]=hP*Cr

                if (ki==1 and kj==1):
                    Mat_coef[k,k]=CP-CN-CW
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k+ntheta]=CS
                
                
                if (ki==1 and kj>1 and kj<nX):
                    Mat_coef[k,k]=CP-CN
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+ntheta]=CS
                
                
                if (ki==1 and kj==nX):
                    Mat_coef[k,k]=CP-CE-CN
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+ntheta]=CS
                
                
                if (kj==1 and ki>1 and ki<nZ):
                    Mat_coef[k,k]=CP-CW
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-ntheta]=CN
                    Mat_coef[k,k+ntheta]=CS
                
                
                if (ki>1 and ki<nZ and kj>1 and kj<nX):
                    Mat_coef[k,k]=CP
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k-ntheta]=CN
                    Mat_coef[k,k+ntheta]=CS
                    Mat_coef[k,k+1]=CE
                
                
                if (kj==nX and ki>1 and ki<nZ):
                    Mat_coef[k,k]=CP-CE
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k-ntheta]=CN
                    Mat_coef[k,k+ntheta]=CS
                
                
                if (kj==1 and ki==nZ):
                    Mat_coef[k,k]=CP-CS-CW
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-ntheta]=CN
                
                
                if (ki==nZ and kj>1 and kj<nX):
                    Mat_coef[k,k]=CP-CS
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k-ntheta]=CN
                
                
                if (ki==nZ and kj==nX):
                    Mat_coef[k,k]=CP-CE-CS
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k-ntheta]=CN
                
                
                if (isempty(find(drop_pressure_Ele_nZ==ki+1))==0 & isempty(find(drop_pressure_Ele_ntetha==kj+1))==0):
                    K_null[k]=k
                    Kij_null[ki,kj]=1
                
                
                kj=kj+1
            # loop end
            
            kj=1
            ki=ki+1
        # loop end

        
        
        # Pressure field solution ==============================================================
        
        cc=find(K_null==0)
        p=Mat_coef(cc,cc)\b(cc,cc) # verificar matriz b
        
        cont=0
        
        # Matrix form of the pressure field ====================================================
        for i in range(1, nZ):  # Loop in Z
            for j in range(1, ntheta):  # Loop in THETA
                if isempty(find(drop_pressure_Ele_nZ==i+1))==0 & isempty(find(drop_pressure_Ele_ntetha==j+1))==0
                    P[i,j]=0
                else
                    
                    cont=cont+1
                    P(i,j)=p(cont)
                    
                    if P(i,j) < 0
                        P(i,j)=0
                    

        # Pressure border conditions ====================================================
        for i in range(1, nZ):  # Loop in Z
            for j in range(1, ntheta):  # Loop in THETA
                if (P[i,j]<0):
                    P[i,j]=0
                
        
        # Dimmensional pressure determination in Pascals
        Pdim=P*mi_ref*war*Rs^2/Cr^2
        
        # Full pressure field with borders
        PPdim=zeros[nZ+2,ntheta+2]
        
        for i in range(2, nZ+1):  # Loop in Z
            for j in range(2, ntheta+1):  # Loop in THETA
                PPdim[i,j]=Pdim[i-1,j-1]
            
        


        %%%%%%%%%%%%%%%%%%% Solu��o do Campo de Temperatura %%%%%%%%%%%%%%%%%%%
        
