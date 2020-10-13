"""Tilting-pad hydrodynamic bearing properties calculation module

In this module, the tilting-pad bearing properties can be estimated and
simulated for a given set of properties, until the model reaches a steady-
-state of numerical convergence. These properties can then be exported to
external files, and used in the bearing_seal_element.py routine inside the
ROSS framework. This routine can also be used for bearing properties esti-
mation for other purposes.
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


# x=[0.000321298907440948 0.000218101024776208 0.000241891712348458 0.000385504446042090 0.000516992650533115 0.000460227890390222];

psi_pad = np.array([x[1], x[2], x[3], x[4], x[5], x[6]])
npad = 6

# Radial clearance
Cr = 250e-6

# Temperature of oil tank
Tcuba = 40

# Rotor center position
xx = E * Cr * np.cos(phi)
yy = E * Cr * np.sin(phi)
alphapt = 0 * (2 * pi * 5) * alpha


# Geometric parameters of bearing --------------------------------------------

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

# Angular position of pivot
#sigma=0:60:300 # [degree]
sigma = np.array([0,300,60]) # [degree]
sigma= sigma*(np.pi/180) # [rad]

# Loading bearing
fR=90.6e3 # [N]

# Rotor speed
wa=300 # [rpm]
war=wa*(np.pi/30) # rad/s

# Reference temperature
T_ref=Tcuba # [Celsius]

# Thermal Properties Oil ----------------------------------------------------

# Thermal conductivity
kt=0.07031*math.exp(484.1/(Tcuba+273.15+474)) # [J/s.m.C]

# Specific heat
Cp=(16.5*math.exp(-2442/(Tcuba+273.15+829.1)))*1e3 # [J/kgC]

# Specific mass
rho=0.04514*math.exp(9103/(Tcuba+273.15+2766))*1e3 # [kg/m**2]

# Reference viscosity
#mi_ref=0.0752
mi_ref=5.506e-09*math.exp(5012/(Tcuba+273.15+0.1248)) # [N.s/m**2]

# Bearing Position ---------------------------------------------------------

# Rotor center velocity
xpt=-(2*np.pi*5)*yy
ypt=(2*np.pi*5)*xx

#  Discretizated Mesh ------------------------------------------------------

# Number of volumes in theta direction
ntheta=48

# Number of volumes in x direction
nX=ntheta

# Number of volumes in z direction
nZ=48

# Number of volumes in netha direction
nN=30

Z1=0 # initial coordinate z dimensionless
Z2=1 # final coordinate z dimensionless
dZ=1/(nZ) # differential z dimensionless
dz=dZ*L # differential z dimensional: [m]
XZ(1)=Z1
XZ(nZ+2)=Z2
# XZ(2:nZ+1)=Z1+0.5*dZ:dZ:Z2-0.5*dZ # vector z dimensionless
XZ(2:nZ+1)=Z1+0.5* np.array([dz, Z2-0.5*dZ, dz]) # vector z dimensionless
XZdim=XZ*L # vector z dimensional [m]

N1=0 # initial coordinate netha dimensionless
N2=1 # final coordinate netha dimensionless
dN=1/(nN) # differential netha dimensionless
netha(1)=N1
netha(nN+2)=N2
# netha(2:nN+1)=N1+0.5*dN:dN:N2-0.5*dN # vector netha dimensionless
netha(2:nN+1)=N1+0.5* np.array([dN, N2-0.5*dN, dN]) # vector netha dimensionless

theta1=-(rp_pad)*betha_s # initial coordinate theta [rad]
theta2=(1-rp_pad)*betha_s # final coordinate theta [rad]
dtheta=betha_s/(ntheta) # differential theta [rad]
Xtheta(1)=theta1
Xtheta(ntheta+2)=theta2
# Xtheta(2:ntheta+1)=theta1+0.5*dtheta:dtheta:theta2-0.5*dtheta # vector theta [rad]
Xtheta(2:ntheta+1)=theta1+0.5* np.array([dtheta, theta2-0.5*dtheta, dtheta]) # vector theta [rad]

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





