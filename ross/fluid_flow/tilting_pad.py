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
    calculate_short_damping_matrix,
    calculate_short_stiffness_matrix,
)
from ross.units import Q_, check_units
from ross.utils import read_table_file


# __all__ = [
# ]

# FOR THE MOMENT, THIS CODE IS BEING TRANSLATED FROM MATLAB TO PYTHON, AND
# THEREFORE THIS PROCESS WILL BE MADE IN LINEAR PROGRAMMING STANDARDS. ONCE
# THIS INITIAL TRANSLATION IS DONE, AND THE CODE VALIDATED, THE MODULARIZATION
# AND CONVERSION TO OBJECT ORIENTED STANDARDS WILL BE MADE.


# class _TiltingPad:
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

phi = 30 * np.pi / 180
E = 0.5


# optim values from legacy codes
x = np.array(
    [
        0.000401905634685165,
        0.000210288009621476,
        0.000136772568561263,
        0.000273217426488742,
        0.000537108459033454,
        0.000574699109242178,
    ]
)

psi_pad = np.array([x[0], x[1], x[2], x[3], x[4], x[5]])
npad = 6

# Radial clearance
Cr = 250e-6

# Oil tank temperature
Tcuba = 40

# Rotor center position
xx = E * Cr * np.cos(phi)
yy = E * Cr * np.sin(phi)
alphapt = 0  # * (2 * np.pi * 5) * alpha


# Geometric parameters for the bearing --------------------------------------------

# Journal radius
R = 0.5 * 930e-3

# Pad radius
Rs = 0.5 * 934e-3  # [m]

# Pad thickness
esp = 67e-3  # [m]

# Pad arc
betha_s = 25  # [degree]
betha_s = betha_s * (np.pi / 180)  # [rad]

# Pivot position (arc pivot/arc pad)
rp_pad = 0.6

# Bength of bearing
L = 197e-3  # [m]

# Angular position of the pivot
sigma = np.array([0, 300, 60])  # [degree]
sigma = sigma * (np.pi / 180)  # [rad]

# Bearing loading
fR = 90.6e3  # [N]

# Rotor speed
wa = 300  # [rpm]
war = wa * (np.pi / 30)  # rad/s

# Reference temperature
T_ref = Tcuba  # [Celsius]

# Thermal properties for the oil ----------------------------------------------------

# Thermal conductivity
kt = 0.07031 * np.exp(484.1 / (Tcuba + 273.15 + 474))  # [J/s.m.C]

# Specific heat
Cp = (16.5 * np.exp(-2442 / (Tcuba + 273.15 + 829.1))) * 1e3  # [J/kgC]

# Specific mass
rho = 0.04514 * np.exp(9103 / (Tcuba + 273.15 + 2766)) * 1e3  # [kg/m**2]

# Reference viscosity
# mi_ref=0.0752
mi_ref = 5.506e-09 * np.exp(5012 / (Tcuba + 273.15 + 0.1248))  # [N.s/m**2]

# Bearing Position ---------------------------------------------------------

# Rotor center velocity
xpt = -(2 * np.pi * 5) * yy
ypt = (2 * np.pi * 5) * xx

#  Discretized Mesh ------------------------------------------------------

# Number of volumes in theta direction
ntheta = 48

# Number of volumes in x direction
nX = ntheta

# Number of volumes in z direction
nZ = 48

# Number of volumes in neta direction
nN = 30

Z1 = 0  # initial coordinate z dimensionless
Z2 = 1  # final coordinate z dimensionless
dZ = 1 / (nZ)  # differential z dimensionless
dz = dZ * L  # differential z dimensional: [m]
XZ = np.zeros([nZ + 2])
XZ[0] = Z1
XZ[nZ + 1] = Z2
XZ[1 : nZ + 1] = Z1 + np.arange(0.5 * dZ, Z2, dZ)  # vector z dimensionless

XZdim = XZ * L  # vector z dimensional [m]

N1 = 0  # initial coordinate netha dimensionless
N2 = 1  # final coordinate netha dimensionless
dN = 1 / (nN)  # differential netha dimensionless

netha = np.zeros([nN + 2])
netha[0] = N1
netha[nN + 1] = N2
netha[1 : nN + 1] = N1 + np.arange(0.5 * dN, N2, dN)  # vector netha dimensionless

theta1 = -(rp_pad) * betha_s  # initial coordinate theta [rad]
theta2 = (1 - rp_pad) * betha_s  # final coordinate theta [rad]

dtheta = betha_s / (ntheta)  # differential theta [rad]
Xtheta = np.zeros([ntheta + 2])
Xtheta[0] = theta1
Xtheta[ntheta + 1] = theta2
Xtheta[1 : ntheta + 1] = np.arange(
    theta1 + 0.5 * dtheta, theta2, dtheta
)  # vector theta [rad]

dX = 1 / nX  # differential x dimensionless
dx = dX * (betha_s * Rs)  # differential x dimensional: [m]
XX = Xtheta * Rs  # vector x dimensional: [m]

# Pad recess
len_betha = 0.39 * betha_s  # Pad Angle with recess
len_L = 0.71 * L  # Bearing length with recess
center_pos_L = L / 2
start_pos_betha = 0 * betha_s
drop_pressure_pos_L = np.array([center_pos_L - len_L / 2, center_pos_L + len_L / 2])
drop_pressure_pos_betha = np.array([start_pos_betha, start_pos_betha + len_betha])

drop_pressure_Ele_nZ = np.intersect1d(
    np.where(XZdim > drop_pressure_pos_L[0]), np.where(XZdim < drop_pressure_pos_L[1])
)
drop_pressure_Ele_ntetha = np.intersect1d(
    np.where(Xtheta >= drop_pressure_pos_betha[0] + theta1),
    np.where(Xtheta <= drop_pressure_pos_betha[1] + theta1),
)

# Initial Parameters ---------------------------------------------------------------

# Startup
npad = npad - 1
Tmist = T_ref * np.ones((nN + 2))

# Initial viscosity field
minovo = mi_ref * np.ones((nZ, ntheta, nN))

# Velocity field  - 3D
vu = np.zeros((nZ, ntheta, nN))
vv = np.zeros((nZ, ntheta, nN))
vw = np.zeros((nZ, ntheta, nN))

# Velocity field - 2D
Vu = np.zeros((nN, ntheta))
Vv = np.zeros((nN, ntheta))
Vw = np.zeros((nN, ntheta))

# Pressure field
P = np.zeros((ntheta, ntheta))
P1 = np.zeros((ntheta, ntheta, npad))

# Temperature field
T = np.zeros((nN, ntheta))
T1 = np.zeros((nN, ntheta, npad))

# Field derivatives
dudx = np.zeros((1, ntheta))
dwdz = np.zeros((1, ntheta))

# Other variables declarations
Mi = np.zeros((nZ, nN))
YH = np.zeros((nN + 2, nX + 2, npad))
XH = np.zeros((nN + 2, nX + 2))

fxj = np.zeros((npad))
My = np.zeros((npad))

for ii in range(0, nX + 2):
    XH[:, ii] = Xtheta[ii]

# Loop on the pads ==========================================================================
for n_p in range(0, npad):
    alpha = psi_pad[n_p]

    # transformation of coordinates - inertial to pivot referential
    xryr = np.dot(
        [
            [np.cos(sigma[n_p]), np.sin(sigma[n_p])],
            [-np.sin(sigma[n_p]), np.cos(sigma[n_p])],
        ],
        [[xx], [yy]],
    )

    xryrpt = np.dot(
        [
            [np.cos(sigma[n_p]), np.sin(sigma[n_p])],
            [-np.sin(sigma[n_p]), np.cos(sigma[n_p])],
        ],
        [[xpt], [ypt]],
    )

    xr = xryr[0, 0]
    yr = xryr[1, 0]

    xrpt = xryrpt[0, 0]
    yrpt = xryrpt[1, 0]

    # Temperature matrix with boundary conditions ====================================
    T_novo = T_ref * np.ones((nN + 2, ntheta + 2))
    Tcomp = 1.2 * T_novo

    # Oil temperature field loop ================================================================
    while np.linalg.norm((T_novo - Tcomp) / np.linalg.norm(Tcomp)) > 0.01:

        nk = nZ * ntheta
        vector_mi = np.zeros((3, nN))
        auxFF0P = np.zeros((nN + 2))
        auxFF1P = np.zeros((nN + 2))
        auxFF0E = np.zeros((nN + 2))
        auxFF1E = np.zeros((nN + 2))
        auxFF0W = np.zeros((nN + 2))
        auxFF1W = np.zeros((nN + 2))
        auxFF2P = np.zeros((nN + 2))
        auxFF2E = np.zeros((nN + 2))
        auxFF2W = np.zeros((nN + 2))
        hhh = np.zeros((nk, npad))  # oil film thickness
        K_null = np.zeros((1, nk))
        Kij_null = np.zeros((nZ, ntheta))

        mi = minovo
        Tcomp = T_novo

        ki = 0
        kj = 0
        k = 0  # pressure vectorization index
        nn = 0

        Mat_coef = np.zeros((nk, nk))
        b = np.zeros((nk))

        # Mesh loop in Z direction ====================================================
        # for ii in range((Z1 + 0.5 * dZ), dZ, (Z2 - 0.5 * dZ)):
        for ii in range(0, nZ):

            # Mesh loop in THETA direction ====================================================
            # for jj in range((theta1 + 0.5 * dtheta), dtheta, (theta2 - 0.5 * dtheta)):
            for jj in range(0, ntheta):

                if kj == 0:
                    vector_mi[0, :] = mi[ki, kj, :]
                    vector_mi[1, :] = mi[ki, kj + 1, :]
                    vector_mi[2, :] = mi[ki, kj, :]

                if kj == ntheta - 1:
                    vector_mi[0, :] = mi[ki, kj, :]
                    vector_mi[1, :] = mi[ki, kj, :]
                    vector_mi[2, :] = mi[ki, kj - 1, :]

                if kj > 0 and kj < ntheta - 1:
                    vector_mi[0, :] = mi[ki, kj, :]
                    vector_mi[1, :] = mi[ki, kj + 1, :]
                    vector_mi[2, :] = mi[ki, kj - 1, :]

                for kk in range(1, nN + 1):

                    mi_adP = vector_mi[0, nN - nn - 1] / mi_ref
                    mi_adE = vector_mi[1, nN - nn - 1] / mi_ref
                    mi_adW = vector_mi[2, nN - nn - 1] / mi_ref

                    auxFF0P[nn + 1] = 1 / mi_adP
                    auxFF1P[nn + 1] = (dN * (-0.5 + kk)) / mi_adP
                    auxFF0E[nn + 1] = 1 / mi_adE
                    auxFF1E[nn + 1] = (dN * (-0.5 + kk)) / mi_adE
                    auxFF0W[nn + 1] = 1 / mi_adW
                    auxFF1W[nn + 1] = (dN * (-0.5 + kk)) / mi_adW

                    nn = nn + 1

                nn = 0

                auxFF0P[0] = auxFF0P[1]
                auxFF0P[nN + 1] = auxFF0P[nN]

                auxFF1P[0] = 0
                auxFF1P[nN + 1] = N2 / (vector_mi[0, nN - 1] / mi_ref)

                auxFF0E[0] = auxFF0E[1]
                auxFF0E[nN + 1] = auxFF0E[nN]

                auxFF1E[0] = 0
                auxFF1E[nN + 1] = N2 / (vector_mi[1, nN - 1] / mi_ref)

                auxFF0W[0] = auxFF0W[1]
                auxFF0W[nN + 1] = auxFF0W[nN]

                auxFF1W[0] = 0
                auxFF1W[nN + 1] = N2 / (vector_mi[2, nN - 1] / mi_ref)

                # Numerical integration

                # FF0P=0.5*sum((netha(2:end)-netha(1:end-1)).*(auxFF0P(2:end)+auxFF0P(1:end-1)));

                FF0P = 0.5 * np.sum(
                    (netha[1:] - netha[0:-1]) * (auxFF0P[1:] + auxFF0P[0:-1])
                )
                FF1P = 0.5 * np.sum(
                    (netha[1:] - netha[0:-1]) * (auxFF1P[1:] + auxFF1P[0:-1])
                )
                FF0E = 0.5 * np.sum(
                    (netha[1:] - netha[0:-1]) * (auxFF0E[1:] + auxFF0E[0:-1])
                )
                FF1E = 0.5 * np.sum(
                    (netha[1:] - netha[0:-1]) * (auxFF1E[1:] + auxFF1E[0:-1])
                )
                FF0W = 0.5 * np.sum(
                    (netha[1:] - netha[0:-1]) * (auxFF0W[1:] + auxFF0W[0:-1])
                )
                FF1W = 0.5 * np.sum(
                    (netha[1:] - netha[0:-1]) * (auxFF1W[1:] + auxFF1W[0:-1])
                )

                FF0e = 0.5 * (FF0P + FF0E)
                FF0w = 0.5 * (FF0P + FF0W)
                FF1e = 0.5 * (FF1P + FF1E)
                FF1w = 0.5 * (FF1P + FF1W)

                # Loop in N
                # for kk in range(N1 + 0.5 * dN, dN, N2 - 0.5 * dN):
                for kk in range(0, nN + 1):

                    mi_adP = vector_mi[0, nN - nn - 1] / mi_ref
                    mi_adE = vector_mi[1, nN - nn - 1] / mi_ref
                    mi_adW = vector_mi[2, nN - nn - 1] / mi_ref

                    auxFF2P[nn] = ((dN * (-0.5 + kk)) / mi_adP) * (
                        (dN * (-0.5 + kk)) - FF1P / FF0P
                    )
                    auxFF2E[nn] = ((dN * (-0.5 + kk)) / mi_adE) * (
                        (dN * (-0.5 + kk)) - FF1E / FF0E
                    )
                    auxFF2W[nn] = ((dN * (-0.5 + kk)) / mi_adW) * (
                        (dN * (-0.5 + kk)) - FF1W / FF0W
                    )
                    nn = nn + 1

                nn = 0

                auxFF2P[0] = 0
                auxFF2P[nN + 1] = (N2 / (vector_mi[0, nN - 1] / mi_ref)) * (
                    N2 - FF1P / FF0P
                )

                auxFF2E[0] = 0
                auxFF2E[nN + 1] = (N2 / (vector_mi[1, nN - 1] / mi_ref)) * (
                    N2 - FF1P / FF0P
                )

                auxFF2W[0] = 0
                auxFF2W[nN + 1] = (N2 / (vector_mi[2, nN - 1] / mi_ref)) * (
                    N2 - FF1P / FF0P
                )

                # integration process ===================================================
                FF2P = 0.5 * np.sum(
                    (netha[1:] - netha[0:-1]) * (auxFF2P[1:] + auxFF2P[0:-1])
                )
                FF2E = 0.5 * np.sum(
                    (netha[1:] - netha[0:-1]) * (auxFF2E[1:] + auxFF2E[0:-1])
                )
                FF2W = 0.5 * np.sum(
                    (netha[1:] - netha[0:-1]) * (auxFF2W[1:] + auxFF2W[0:-1])
                )

                FF2e = 0.5 * (FF2P + FF2E)
                FF2w = 0.5 * (FF2P + FF2W)
                FF2n = FF2P
                FF2s = FF2n

                # Admensional oil film thickness ========================================

                hP = (
                    Rs
                    - R
                    - (
                        np.sin(Xtheta[jj + 1]) * (yr + alpha * (Rs + esp))
                        + np.cos(Xtheta[jj + 1]) * (xr + Rs - R - Cr)
                    )
                ) / Cr
                he = (
                    Rs
                    - R
                    - (
                        np.sin(Xtheta[jj + 1] + 0.5 * dtheta)
                        * (yr + alpha * (Rs + esp))
                        + np.cos(Xtheta[jj + 1] + 0.5 * dtheta) * (xr + Rs - R - Cr)
                    )
                ) / Cr
                hw = (
                    Rs
                    - R
                    - (
                        np.sin(Xtheta[jj + 1] - 0.5 * dtheta)
                        * (yr + alpha * (Rs + esp))
                        + np.cos(Xtheta[jj + 1] - 0.5 * dtheta) * (xr + Rs - R - Cr)
                    )
                ) / Cr
                hn = hP
                hs = hn
                hpt = -(1 / (Cr * war)) * (
                    np.cos(Xtheta[jj + 1]) * xrpt
                    + np.sin(Xtheta[jj + 1]) * yrpt
                    + np.sin(Xtheta[jj + 1]) * (Rs + esp) * alphapt
                )  # admensional

                # Finite volume frontiers
                CE = 1 / (betha_s) ** 2 * (FF2e * he ** 3) * dZ / dX
                CW = 1 / (betha_s) ** 2 * (FF2w * hw ** 3) * dZ / dX
                CN = (FF2n * hn ** 3) * (dX / dZ) * (Rs / L) ** 2
                CS = (FF2s * hs ** 3) * (dX / dZ) * (Rs / L) ** 2
                CP = -(CE + CW + CN + CS)

                B = (R / (Rs * betha_s)) * dZ * (
                    he * (1 - FF1e / FF0e) - hw * (1 - FF1w / FF0w)
                ) + hpt * dX * dZ

                b[k] = B
                hhh[k, n_p] = hP * Cr

                # Mat_coef determination depending on its mesh localization
                if ki == 0 and kj == 0:
                    Mat_coef[k, k] = CP - CN - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + ntheta] = CS

                if ki == 0 and kj > 0 and kj < nX - 1:
                    Mat_coef[k, k] = CP - CN
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + ntheta] = CS

                if ki == 0 and kj == nX - 1:
                    Mat_coef[k, k] = CP - CE - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + ntheta] = CS

                if kj == 0 and ki > 0 and ki < nZ - 1:
                    Mat_coef[k, k] = CP - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - ntheta] = CN
                    Mat_coef[k, k + ntheta] = CS

                if ki > 0 and ki < nZ - 1 and kj > 0 and kj < nX - 1:
                    Mat_coef[k, k] = CP
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - ntheta] = CN
                    Mat_coef[k, k + ntheta] = CS
                    Mat_coef[k, k + 1] = CE

                if kj == nX - 1 and ki > 0 and ki < nZ - 1:
                    Mat_coef[k, k] = CP - CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - ntheta] = CN
                    Mat_coef[k, k + ntheta] = CS

                if kj == 0 and ki == nZ - 1:
                    Mat_coef[k, k] = CP - CS - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - ntheta] = CN

                if ki == nZ - 1 and kj > 0 and kj < nX - 1:
                    Mat_coef[k, k] = CP - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - ntheta] = CN

                if ki == nZ - 1 and kj == nX - 1:
                    Mat_coef[k, k] = CP - CE - CS
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - ntheta] = CN

                if (
                    len(np.where(drop_pressure_Ele_nZ == ki + 1)[0]) == 0
                    and len(np.where(drop_pressure_Ele_ntetha == kj + 1)[0]) == 0
                ):
                    K_null[0, k] = k
                    Kij_null[ki, kj - 2] = 1

                kj = kj + 1
                k = k + 1
            # loop end

            kj = 0
            ki = ki + 1
        # loop end

        # Pressure field solution ==============================================================

        # cc = (K_null == 0).nonzero()  # cc = find(K_null == 0)
        cc_aux = np.where(K_null == 0)  # cc = find(K_null == 0)
        cc = cc_aux[1]

        lalala = Mat_coef[[cc, cc]]

        lalala2 = b[cc]

        p = np.linalg.solve(Mat_coef[[cc, cc]], b[cc])

        cont = 0

        # Matrix form of the pressure field ====================================================
        for i in range(0, nZ - 1):  # Loop in Z
            for j in range(0, ntheta - 1):  # Loop in THETA
                if (
                    len(np.where(drop_pressure_Ele_nZ == i + 1)[0]) == 0
                    and len(np.where(drop_pressure_Ele_ntetha == j + 1)[0]) == 0
                ):
                    P[i, j] = 0
                else:

                    cont = cont + 1
                    P[i, j] = p(cont)

                    if P[i, j] < 0:
                        P[i, j] = 0

        # Pressure border conditions ====================================================
        for i in range(0, nZ - 1):  # Loop in Z
            for j in range(0, ntheta - 1):  # Loop in THETA
                if P[i, j] < 0:
                    P[i, j] = 0

        # Dimmensional pressure determination in Pascals
        Pdim = P * mi_ref * war * Rs ^ 2 / Cr ^ 2

        # Full pressure field with borders
        PPdim = np.zeros((nZ + 1, ntheta + 1))

        for i in range(0, nZ - 1):  # Loop in Z
            for j in range(0, ntheta - 1):  # Loop in THETA
                PPdim[i, j] = Pdim[i - 1, j - 1]

        # %%%%%%%%%%%%%%%%%%% Temperature field solution %%%%%%%%%%%%%%%%%%%

        # Velocity field calculation
        ki = 0
        kj = 0
        kk = 0
        nn = 0

        # Dimensionless Netha loop ====================================================
        for ky in range((N1 + 0.5 * dN), dN, (N2 - 0.5 * dN)):
            # Mesh loop in Z direction ====================================================
            for ii in range((Z1 + 0.5 * dZ), dZ, (Z2 - 0.5 * dZ)):
                # Mesh loop in THETA direction ====================================================
                for jj in range(
                    (theta1 + 0.5 * dtheta), dtheta, (theta2 - 0.5 * dtheta)
                ):

                    # Pressure gradients calculation
                    if ki == 0 and kj == 0:
                        dPdx = Pdim[ki, kj] / (0.5 * dx)
                        dPdz = Pdim[ki, kj] / (0.5 * dz)

                    if ki == 0 and kj > 0:
                        dPdx = (Pdim[ki, kj] - Pdim[ki, kj - 1]) / dx
                        dPdz = Pdim[ki, kj] / (0.5 * dz)

                    if ki > 0 and kj == 0:
                        dPdx = Pdim[ki, kj] / (0.5 * dx)
                        dPdz = (Pdim[ki, kj] - Pdim[ki - 1, kj]) / dz

                    if ki > 0 and kj > 0:
                        dPdx = (Pdim[ki, kj] - Pdim[ki, kj - 1]) / dx
                        dPdz = (Pdim[ki, kj] - Pdim[ki - 1, kj]) / dz

                    # Dimensional film thickness in Meters
                    h = (
                        Rs
                        - R
                        - (
                            np.sin(jj) * (yr + alpha * (Rs + esp))
                            + np.cos(jj) * (xr + Rs - R - Cr)
                        )
                    )

                    auxFF0 = np.zeros((1, netha.size))
                    auxFF1 = np.zeros((1, netha.size))

                    for contk in range(
                        ((N1 + 0.5 * dN) * h), (dN * h), ((N2 - (0.5 * dN)) * h)
                    ):
                        auxFF0[nn + 1] = 1 / mi[ki, kj, nN + 1 - nn]
                        auxFF1[nn + 1] = contk / mi[ki, kj, nN + 1 - nn]
                        nn = nn + 1
                    nn = 0

                    auxFF0[0] = auxFF0[1]
                    auxFF0[nN + 2] = auxFF0[nN + 1]

                    auxFF1[0] = 0
                    auxFF1[nN + 2] = (N2 * h) / mi[ki, kj, 1]

                    ydim1 = h * netha
                    FF0 = 0.5 * np.sum(
                        (ydim1[1:] - ydim1[0:-1]) * (auxFF0[1:] + auxFF0[0:-1])
                    )
                    FF1 = 0.5 * np.sum(
                        (ydim1[1:] - ydim1[0:-1]) * (auxFF1[1:] + auxFF1[0:-1])
                    )

                    aux_var_1 = np.arange(N1, ky + 1, dN)  # N1:dN:ky
                    auxG0 = np.zeros[1, aux_var_1.size]
                    auxG1 = np.zeros[1, aux_var_1.size]
                    ydim2 = np.zeros[1, aux_var_1.size]

                    for contk in range(((N1 + 0.5 * dN) * h), (dN * h), (ky * h)):
                        auxG0[nn + 1] = 1 / mi[ki, kj, nN + 1 - nn]
                        auxG1[nn + 1] = contk / mi[ki, kj, nN + 1 - nn]
                        ydim2[nn + 1] = contk
                        nn = nn + 1
                    nn = 0

                    auxG0[0] = auxG0[1]
                    auxG1[0] = 0
                    ydim2[0] = N1 * h

                    G0 = (
                        0.5
                        * np.sum[(ydim2[1:] - ydim2[0:-1]) * (auxG0[1:] + auxG0[0:-1])]
                    )
                    G1 = (
                        0.5
                        * np.sum[(ydim2[1:] - ydim2[0:-1]) * (auxG1[1:] + auxG1[0:-1])]
                    )

                    vu[ki, kj, kk] = dPdx * G1 + (war * R / FF0 - FF1 / FF0 * dPdx) * G0
                    vw[ki, kj, kk] = dPdz * G1 - (FF1 / FF0 * dPdz) * G0

                    kj = kj + 1

                kj = 0
                ki = ki + 1

            ki = 0
            kk = kk + 1

        # Radial speed calculation start ----------------------------------------------------
        nn = 0
        ki = 0
        kj = 0
        kk = 0

        # Mesh loop in Z direction ====================================================
        for ii in range((Z1 + 0.5 * dZ), dZ, (Z2 - 0.5 * dZ)):
            # Mesh loop in THETA direction ====================================================
            for jj in range((theta1 + 0.5 * dtheta), dtheta, (theta2 - 0.5 * dtheta)):

                hpt = -(
                    np.cos[dtheta * (0.5 + jj)] * xrpt
                    + np.sin[dtheta * (0.5 + jj)] * yrpt
                    + np.sin[dtheta * (0.5 + jj)] * (Rs + esp) * alphapt
                )

                if ki == 1 and kj == 1:
                    for contk in range(N1 + 0.5 * dN, dN, N2 - 0.5 * dN):
                        dudx[0, nn] = 0
                        dwdz[0, nn] = 0
                        nn = nn + 1
                    nn = 0

                if ki == 1 and kj > 1:
                    for contk in range(N1 + 0.5 * dN, dN, N2 - 0.5 * dN):
                        dudx[0, nn + 1] = (vu[ki, kj, nn] - vu[ki, kj - 1, nn]) / dx
                        dwdz[0, nn + 1] = 0
                        nn = nn + 1
                    nn = 0

                if ki > 1 and kj == 1:
                    for contk in range(N1 + 0.5 * dN, dN, N2 - 0.5 * dN):
                        dudx[0, nn + 1] = 0
                        dwdz[0, nn + 1] = (vw[ki, kj, nn] - vw[ki - 1, kj, nn]) / dz
                        nn = nn + 1
                    nn = 0

                if ki > 1 and ki < nN and kj > 1 and kj < nX:
                    for contk in range(N1 + 0.5 * dN, dN, N2 - 0.5 * dN):
                        dudx[0, nn + 1] = (vu[ki, kj, nn] - vu[ki, kj - 1, nn]) / dx
                        dwdz[0, nn + 1] = (vw[ki, kj, nn] - vw[ki - 1, kj, nn]) / dz
                        nn = nn + 1
                    nn = 0

                dudx[0, 0] = dudx[0, 1]
                dwdz[0, 0] = dwdz[0, 1]
                dudx[0, nN + 2] = dudx[0, nN + 1]
                dwdz[0, nN + 2] = dwdz[0, nN + 1]

                auxD = dudx + dwdz
                intv = 0.5 * np.sum((ydim1[1:] - ydim1[0:-1]) * (auxD[1:] + auxD[0:-1]))
                vv[ki, kj,] = (
                    -intv + hpt
                )
                kj = kj + 1

            kj = 0
            ki = ki + 1

        ki = 0
        ki = nN
        for ii in range(1, nN):
            for jj in range(1, ntheta):
                Vu[ii, jj] = vu[:, jj, ki].mean
                Vv[ii, jj] = vv[:, jj, ki].mean
                Vw[ii, jj] = vw[:, jj, ki].mean

            ki = ki - 1

        # Radial velociti calculation ending ------------------------------
        ksi1 = 0
        ksi2 = 1

        ki = 0
        kj = 0
        dksi = dX

        for ii in range(1, nZ):
            for jj in range(1, nN):
                Mi[jj, ii] = mi[0, ii, jj]

        nk = nN * ntheta
        Mat_coef = np.zeros[nk, nk]
        b = np.zeros[nk, 1]
        k = 0

        for ii in range(N1 + 0.5 * dN, dN, N2 - 0.5 * dN):

            for jj in range(ksi1 + 0.5 * dksi, dksi, ksi2 - 0.5 * dksi):

                theta = (-0.5 + dtheta * (0.5 + jj)) * betha_s
                HP = (
                    Rs
                    - R
                    - (
                        np.sin[theta] * (yr + alpha * (Rs + esp))
                        + np.cos[theta] * (xr + Rs - R - Cr)
                    )
                )
                He = (
                    Rs
                    - R
                    - (
                        np.sin[theta + 0.5 * dtheta] * (yr + alpha * (Rs + esp))
                        + np.cos[theta + 0.5 * dtheta] * (xr + Rs - R - Cr)
                    )
                )
                Hw = (
                    Rs
                    - R
                    - (
                        np.sin[theta - 0.5 * dtheta] * (yr + alpha * (Rs + esp))
                        + np.cos[theta - 0.5 * dtheta] * (xr + Rs - R - Cr)
                    )
                )
                Hn = HP
                Hs = Hn
                Hnw = Hw
                Hsw = Hnw

                yh[kj] = HP
                xh[kj] = theta

                JP = 1 / (betha_s * Rs * HP)
                Je = 1 / (betha_s * Rs * He)
                Jw = 1 / (betha_s * Rs * Hw)
                Jn = 1 / (betha_s * Rs * Hn)
                Js = 1 / (betha_s * Rs * Hs)

                if ki == 1 and kj == 1:

                    uP = Vu[ki, kj]
                    uE = Vu[ki, kj + 1]
                    uW = Vu[ki, kj]
                    uN = Vu[ki, kj]
                    uS = Vu[ki + 1, kj]

                    vP = Vv[ki, kj]
                    vE = Vv[ki, kj + 1]
                    vW = Vv[ki, kj]
                    vN = Vv[ki, kj]
                    vS = Vv[ki + 1, kj]

                    wP = Vw[ki, kj]
                    wE = Vw[ki, kj + 1]
                    wW = Vw[ki, kj]
                    wN = Vw[ki, kj]
                    wS = Vw[ki + 1, kj]

                if ki == 1 and kj > 1 and kj < nX:
                    uP = Vu[ki, kj]
                    uE = Vu[ki, kj + 1]
                    uW = Vu[ki, kj - 1]
                    uN = Vu[ki, kj]
                    uS = Vu[ki + 1, kj]

                if ki > 1 and ki < nN and kj == 1:
                    uP = Vu[ki, kj]
                    uE = Vu[ki, kj + 1]
                    uW = Vu[ki, kj]
                    uN = Vu[ki - 1, kj]
                    uS = Vu[ki + 1, kj]

                if ki == nN and kj == 1:
                    uP = Vu[ki, kj]
                    uE = Vu[ki, kj + 1]
                    uW = Vu[ki, kj]
                    uN = Vu[ki - 1, kj]
                    uS = Vu[ki, kj]

                if ki == 1 and kj == nX:
                    uP = Vu[ki, kj]
                    uE = Vu[ki, kj]
                    uW = Vu[ki, kj - 1]
                    uN = Vu[ki, kj]
                    uS = Vu[ki + 1, kj]

                if ki > 1 and ki < nN and kj == nX:
                    uP = Vu[ki, kj]
                    uE = Vu[ki, kj]
                    uW = Vu[ki, kj - 1]
                    uN = Vu[ki - 1, kj]
                    uS = Vu[ki + 1, kj]

                if ki == nN and kj == nX:
                    uP = Vu[ki, kj]
                    uE = Vu[ki, kj]
                    uW = Vu[ki, kj - 1]
                    uN = Vu[ki - 1, kj]
                    uS = Vu[ki, kj]

                if ki == nN and kj > 1 and kj < nX:
                    uP = Vu[ki, kj]
                    uE = Vu[ki, kj + 1]
                    uW = Vu[ki, kj - 1]
                    uN = Vu[ki - 1, kj]
                    uS = Vu[ki, kj]

                if ki > 1 and ki < nN and kj > 1 and kj < nX:
                    uP = Vu[ki, kj]
                    uE = Vu[ki, kj + 1]
                    uW = Vu[ki, kj - 1]
                    uN = Vu[ki - 1, kj]
                    uS = Vu[ki + 1, kj]

                    vP = Vv[ki, kj]
                    vE = Vv[ki, kj + 1]
                    vW = Vv[ki, kj - 1]
                    vN = Vv[ki - 1, kj]
                    vS = Vv[ki + 1, kj]

                    wP = Vw[ki, kj]
                    wE = Vw[ki, kj + 1]
                    wW = Vw[ki, kj - 1]
                    wN = Vw[ki - 1, kj]
                    wS = Vw[ki + 1, kj]

                ue = 0.5 * [uP + uE]
                uw = 0.5 * [uP + uW]
                un = 0.5 * [uP + uN]
                us = 0.5 * [uP + uS]

                ve = 0.5 * [vP + vE]
                vw = 0.5 * [vP + vW]
                vn = 0.5 * [vP + vN]
                vs = 0.5 * [vP + vS]

                we = 0.5 * [wP + wE]
                ww = 0.5 * [wP + wW]
                wn = 0.5 * [wP + wN]
                ws = 0.5 * [wP + wS]

                UP = hP * uP
                Ue = He * ue
                Uw = Hw * uw

                np = 1 - ii
                ne = np
                nw = np
                nn = np + dN
                ns = np - dN

                dhdksi_p = -betha_s * (
                    np.cos[theta] * (yr + alpha * (Rs + esp))
                    - np.sin[theta] * (xr + Rs - R - Cr)
                )
                dhdksi_e = -betha_s * (
                    np.cos[theta + 0.5 * dtheta] * (yr + alpha * (Rs + esp))
                    - np.sin[theta + 0.5 * dtheta] * (xr + Rs - R - Cr)
                )
                dhdksi_w = -betha_s * (
                    np.cos[theta - 0.5 * dtheta] * (yr + alpha * (Rs + esp))
                    - np.sin[theta - 0.5 * dtheta] * (xr + Rs - R - Cr)
                )
                dhdksi_n = dhdksi_p
                dhdksi_s = dhdksi_n

                VP = betha_s * Rs * vP - np * dhdksi_p * uP
                Vn = betha_s * Rs * vn - nn * dhdksi_n * un
                Vs = betha_s * Rs * vs - ns * dhdksi_s * us

                alpha11P = HP ** 2
                alpha11e = He ** 2
                alpha11w = Hw ** 2

                alpha12P = -np * HP * dhdksi_p
                alpha12e = -ne * He * dhdksi_e
                alpha12w = -nw * Hw * dhdksi_w

                alpha21P = alpha12P
                alpha21n = -nn * Hn * dhdksi_n
                alpha21s = -ns * Hs * dhdksi_s

                alpha22P = (betha_s * Rs) ** 2 + (np * dhdksi_p) ** 2
                alpha22n = (betha_s * Rs) ** 2 + (nn * dhdksi_n) ** 2
                alpha22s = (betha_s * Rs) ** 2 + (ns * dhdksi_s) ** 2

                Me = rho * Ue * dN
                Mw = rho * Uw * dN
                Mn = rho * Vn * dksi
                Ms = rho * Vs * dksi

                D11 = kt / Cp * JP * alpha11P * dN
                D11e = kt / Cp * Je * alpha11e * dN
                D11w = kt / Cp * Jw * alpha11w * dN

                D12 = kt / Cp * JP * alpha12P * dN
                D12e = kt / Cp * Je * alpha12e * dN
                D12w = kt / Cp * Jw * alpha12w * dN

                D21 = kt / Cp * JP * alpha21P * dksi
                D21n = kt / Cp * Jn * alpha21n * dksi
                D21s = kt / Cp * Js * alpha21s * dksi

                D22 = kt / Cp * JP * alpha22P * dksi
                D22n = kt / Cp * Jn * alpha22n * dksi
                D22s = kt / Cp * Js * alpha22s * dksi

                # Interpolation coefficients

                Pee = rho * uE * Cp * dtheta * Rs / kt  # Peclet's number
                Pew = rho * uW * Cp * dtheta * Rs / kt

                Pen = rho * uN * Cp * dtheta * Rs / kt
                Pes = rho * uS * Cp * dtheta * Rs / kt

                a_pe = Pee ** 2 / (10 + 2 * Pee ** 2)
                b_pe = (1 + 0.005 * Pee ** 2) / (1 + 0.05 * Pee ** 2)

                a_pw = Pew ** 2 / (10 + 2 * Pew ** 2)
                b_pw = (1 + 0.005 * Pew ** 2) / (1 + 0.05 * Pew ** 2)

                a_sw = Pes ** 2 / (10 + 2 * Pes ** 2)
                b_sw = (1 + 0.005 * Pes ** 2) / (1 + 0.05 * Pes ** 2)

                a_nw = Pen ** 2 / (10 + 2 * Pen ** 2)
                b_nw = (1 + 0.005 * Pen ** 2) / (1 + 0.05 * Pen ** 2)

                a_pn = 0  # Central differences
                b_pn = 1

                a_ps = 0
                b_ps = 1

                Ae = Me * (0.5 - a_pe) - D11e / dksi * b_pe - (D21n - D21s) / (4 * dksi)
                Aw = (
                    -Mw * (0.5 + a_pw) - D11w / dksi * b_pw + (D21n - D21s) / (4 * dksi)
                )
                An = Mn * (0.5 - a_pn) - D22n / dN * b_pn - (D12e - D12w) / (4 * dN)
                As = -Ms * (0.5 + a_ps) - D22s / dN * b_ps - (D12w - D12e) / (4 * dN)
                Ane = -D12e / (4 * dN) - D21n / (4 * dksi)
                Ase = D12e / (4 * dN) + D21s / (4 * dksi)
                Anw = D12w / (4 * dN) + D21n / (4 * dksi)
                Asw = -D12w / (4 * dN) - D21s / (4 * dksi)
                Ap = -(Ae + Aw + An + As + Ane + Ase + Anw + Asw)

                up_a = uP / (R * war)
                uw_a = uW / (R * war)
                ue_a = uE / (R * war)
                us_a = uS / (R * war)
                un_a = uN / (R * war)

                vp_a = vP / (R * war)
                vw_a = vW / (R * war)
                ve_a = vE / (R * war)
                vs_a = vS / (R * war)
                vn_a = vN / (R * war)

                wp_a = wP / (R * war)
                ww_a = wW / (R * war)
                we_a = wE / (R * war)
                ws_a = wS / (R * war)
                wn_a = wN / (R * war)

                fdiss = 2 * (
                    (
                        Cr * hP * (up_a - uw_a) / dksi
                        - np * Cr * dhdksi_p * (up_a - us_a) / dN
                    )
                    ** 2
                    + (betha_s * Rs * (vn_a - vp_a) / dN) ** 2
                    + (
                        betha_s * Rs * (up_a - us_a) / dN
                        + Cr * hP * (vp_a - vw_a) / dksi
                        - np * Cr * dhdksi_p * (vp_a - vs_a) / dksi
                    )
                    ** 2
                    + (
                        Cr * hP * (wp_a - ww_a) / dksi
                        - np * Cr * dhdksi_p * (wp_a - ws_a) / dN
                    )
                    ** 2
                    + (betha_s * Rs * (wp_a - ws_a) / dksi) ** 2
                )

                # Source term ----------------
                Bp = JP * (war * R) ** 2 * Mi(ki, kj) / Cp * dN * dksi * fdiss

                # Vectorizing
                k = k + 1

                b[k, 0] = Bp

                if ki == 1 and kj == 1:
                    Mat_coef[k, k] = Ap + An - Aw
                    Mat_coef[k, k + 1] = Ae + Ane
                    Mat_coef[k, k + ntheta] = As - Asw
                    Mat_coef[k, k + ntheta + 1] = Ase
                    b[k, 0] = (
                        b[k, 0]
                        - 2 * (Aw * Tmist[ki + 1] + Asw * Tmist[ki + 2])
                        - Anw * (Tmist[ki])
                    )

                if ki == 1 and kj > 1 and kj < nX:
                    Mat_coef[k, k] = Ap + An
                    Mat_coef[k, k + 1] = Ae + Ane
                    Mat_coef[k, k - 1] = Aw + Anw
                    Mat_coef[k, k + ntheta] = As
                    Mat_coef[k, k + ntheta + 1] = Ase
                    Mat_coef[k, k + ntheta - 1] = Asw

                if ki == 1 and kj == nX:
                    Mat_coef[k, k] = Ap + An + Ane + Ae
                    Mat_coef[k, k - 1] = Aw + Anw
                    Mat_coef[k, k + ntheta] = As + Ase
                    Mat_coef[k, k + ntheta - 1] = Asw

                if kj == 1 and ki > 1 and ki < nN:
                    Mat_coef[k, k] = Ap - Aw
                    Mat_coef[k, k + 1] = Ae
                    Mat_coef[k, k + ntheta] = As - Asw
                    Mat_coef[k, k - ntheta] = An - Anw
                    Mat_coef[k, k + ntheta + 1] = Ase
                    Mat_coef[k, k - ntheta + 1] = Ane
                    b[k, 0] = (
                        b[k, 0]
                        - 2 * Tmist[ki - 1] * Anw
                        - 2 * Tmist[ki + 1] * Aw
                        - 2 * Tmist[ki + 2] * Asw
                    )

                if ki > 1 and ki < nN and kj > 1 and kj < nX:
                    Mat_coef[k, k] = Ap
                    Mat_coef[k, k + 1] = Ae
                    Mat_coef[k, k - 1] = Aw
                    Mat_coef[k, k + ntheta] = As
                    Mat_coef[k, k - ntheta] = An
                    Mat_coef[k, k + ntheta + 1] = Ase
                    Mat_coef[k, k + ntheta - 1] = Asw
                    Mat_coef[k, k - ntheta + 1] = Ane
                    Mat_coef[k, k - ntheta - 1] = Anw

                if kj == nX and ki > 1 and ki < nN:
                    Mat_coef[k, k] = Ap + Ae
                    Mat_coef[k, k - 1] = Aw
                    Mat_coef[k, k + ntheta] = As + Ase
                    Mat_coef[k, k - ntheta] = An + Ane
                    Mat_coef[k, k + ntheta - 1] = Asw
                    Mat_coef[k, k - ntheta - 1] = Anw

                if kj == 1 and ki == nN:
                    Mat_coef[k, k] = Ap + As - Aw
                    Mat_coef[k, k + 1] = Ae + Ase
                    Mat_coef[k, k - ntheta] = An - Anw
                    Mat_coef[k, k - ntheta + 1] = Ane
                    b[k, 0] = (
                        b[k, 0]
                        - 2 * Tmist[ki + 1] * Aw
                        - 2 * Tmist[ki] * Anw
                        - Tmist[ki + 2] * Asw
                    )

                if ki == nN and kj > 1 and kj < nX:
                    Mat_coef[k, k] = Ap + As
                    Mat_coef[k, k + 1] = Ae + Ase
                    Mat_coef[k, k - 1] = Aw + Asw
                    Mat_coef[k, k - ntheta] = An
                    Mat_coef[k, k - ntheta + 1] = Ane
                    Mat_coef[k, k - ntheta - 1] = Anw

                if ki == nN and kj == nX:
                    Mat_coef[k, k] = Ap + As + Ae + Ase
                    Mat_coef[k, k - 1] = Aw + Asw
                    Mat_coef[k, k - ntheta] = An + Ane
                    Mat_coef[k, k - ntheta - 1] = Anw

                kj = kj + 1

            kj = 0
            ki = ki + 1

        ki = 0
        nn = 0
        t = np.linalg.solve(Mat_coef, b)  # calculo da temperatura vetorizada

        # Temperature matrix ----------------------
        cont = 0

        for i in range(0, nN):
            for j in range(0, nX):
                T[i, j] = t[cont]
                cont = cont + 1

        # Viscosity equation ========================================================================
        # VG68 - Polynomial adjustment using predetermined values
        #
        # Via Oil_Regression_Analyses.m and propvalue.m the equation coefficients
        # are obtained for the regression on the viscosity determination as a
        # function of the temperature.

        # 3D temperature field -----------------------------------------------------
        TT = np.zeros(nZ, nX, nN)
        for k in range(0, nN):
            for j in range(0, nX):
                TT[:, j, k] = T[k, j]

        # Regression equation coefficients
        a = 5.506e-09
        b = 5012
        c = 0.1248
        minovo = a * np.exp(b / (TT + 273.15 + c))
        k = 0

        # Full temperature matrix, including borders
        for i in range(1, nN):
            for j in range(1, ntheta):
                T_novo[i, j] = T[i - 1, j - 1]

        T_novo[0,] = T_novo[
            1,
        ]
        T_novo[nN + 1,] = T_novo[
            nN,
        ]
        T_novo[1:nN, 0] = Tmist[1, nN]
        T_novo[:, nX + 1] = T_novo[:, nX]

    # WHILE ENDS HERE ==========================================================

    T1[:, :, n_p] = T_novo
    P1[:, :, n_p] = PPdim

    yh = (
        Rs
        - R
        - (
            np.sin[Xtheta] * (yr + alpha * (Rs + esp))
            + np.cos[Xtheta] * (xr + Rs - R - Cr)
        )
    )
    for jj in range(0, nX + 1):
        YH[:, jj, n_p] = np.fliplr(np.linspace(0, yh(jj), nN + 2))

    # Integration of pressure field - HydroForces
    auxF = np.array([np.cos[Xtheta[0:-1]], np.sin[Xtheta[0:-1]]])
    dA = dx * dz

    auxP = P1[1:-1, 1:-1, n_p] * dA

    vector_auxF_x = auxF[
        0,
    ]
    vector_auxF_y = auxF[
        1,
    ]

    auxFx = auxP * vector_auxF_x.T
    auxFy = auxP * vector_auxF_y.T

    fxj[n_p] = -np.sum(auxFx)
    fyj = -np.sum(auxFy)

    My[n_p] = fyj * (Rs + esp)

    if fxj(n_p) >= -1:
        My[n_p] = 10e6


# END PADS FOR LOOP ===============================================================

score[0] = My[0]
score[1] = My[1]
score[2] = My[2]
score[3] = My[3]
score[4] = My[4]
score[5] = My[5]

# hydrodynamic forces
Fhx = (
    fxj[0] * np.cos[psi_pad[0] + sigma[0]]
    + fxj[1] * np.cos[psi_pad[1] + sigma[1]]
    + fxj[2] * np.cos[psi_pad[2] + sigma[2]]
    + fxj[3] * np.cos[psi_pad[3] + sigma[3]]
    + fxj[4] * np.cos[psi_pad[4] + sigma[4]]
    + fxj[5] * np.cos[psi_pad[5] + sigma[5]]
)
Fhy = (
    fxj[0] * np.sin[psi_pad[0] + sigma[0]]
    + fxj[1] * np.sin[psi_pad[1] + sigma[1]]
    + fxj[2] * np.sin[psi_pad[2] + sigma[2]]
    + fxj[3] * np.sin[psi_pad[3] + sigma[3]]
    + fxj[4] * np.sin[psi_pad[4] + sigma[4]]
    + fxj[5] * np.sin[psi_pad[5] + sigma[5]]
)

# Maximum pressure, maximum temperature and minimum thickness
Pmax = np.max[np.max[np.max[P1]]]
Tmax = np.max[np.max[np.max[T1]]]
hmin = np.min[np.min[hhh]]
