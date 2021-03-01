import numpy as np
from scipy.optimize import minimize
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from matplotlib import cm
from ross.units import Q_, check_units
import time
import math
import sys


class THDCylindrical:
    """This class calculates the pressure and temperature field in oil film of a cylindrical bearing, with two (2) pads. It is also possible to obtain the stiffness and damping coefficients.

    Parameters
    ----------
    Bearing Geometry
    ^^^^^^^^^^^^^^^^
    Describes the geometric characteristics.
    L : float, pint.Quantity
        Bearing length. Default unit is meter.
    R : float
        Rotor radius. The unit is meter.
    c_r : float
        Radial clearence between rotor and bearing. The unit is meter.
    betha_s : float
        Arc length of each pad. The unit is degree.


    Operation conditions
    ^^^^^^^^^^^^^^^^^^^^
    Describes the operation conditions of the bearing.
    speed : float, pint.Quantity
        Rotor rotational speed. Default unit is rad/s.
    Wx : Float
        Load in X direction. The unit is newton.
    Wy : Float
        Load in Y direction. The unit is newton.

    Fluid propierties
    ^^^^^^^^^^^^^^^^^
    Describes the fluid characteristics.
    mu_ref : float
        Fluid reference viscosity. The unit is Pa*s.
    rho : float, pint.Quantity
        Fluid density. Default unit is kg/m^3.
    k_t :  Float
        Fluid thermal conductivity. The unit is J/(s*m*°C).
    Cp : float
        Fluid specific heat. The unit is J/(kg*°C).
    Treserv : float
        Oil reservoir temperature. The unit is celsius.
    fat_mixt : float
        Ratio of oil in Treserv temperature that mixes with the circulating oil.

    Mesh discretization
    ^^^^^^^^^^^^^^^^^^^
    Describes the discretization of the bearing.
    ntheta : int
        Number of volumes along the direction theta (direction of flow).
    nz : int
        Number of volumes along the Z direction (axial direction).
    n_gap : int
        Number of volumes in recess zone.


    Returns
    -------
    A THDCylindrical object.

    References
    ----------
    .. [1] BARBOSA, J. S.; LOBATO, FRAN S.; CAMPANINE SICCHIERI, LEONARDO;CAVALINI JR, ALDEMIR AP. ; STEFFEN JR, VALDER. Determinação da Posição de Equilíbrio em Mancais Hidrodinâmicos Cilíndricos usando o Algoritmo de Evolução Diferencial. REVISTA CEREUS, v. 10, p. 224-239, 2018. ..
    .. [2] DANIEL, G.B. Desenvolvimento de um Modelo Termohidrodinâmico para Análise em Mancais Segmentados. Campinas: Faculdade de Engenharia Mecânica, Universidade Estadual de Campinas, 2012. Tese (Doutorado). ..
    .. [3] NICOLETTI, R., Efeitos Térmicos em Mancais Segmentados Híbridos – Teoria e Experimento. 1999. Dissertação de Mestrado. Universidade Estadual de Campinas, Campinas. ..

    Attributes
    ----------
    Pdim : array
        Dimensional pressure field. The unit is pascal.
    dPdz : array
        Differential pressure field in z direction.
    dPdy : array
        Differential pressure field in theta direction.
    Tdim : array
        Dimensional temperature field. The unit is celsius.
    Fhx : float
        Force in X direction. The unit is newton.
    Fhy : float
        Force in Y direction. The unit is newton.

    Examples
    --------
    >>> from ross.fluid_flow.cylindrical import cylindrical_bearing_example
    >>> x0 = [0.1,-0.1]
    >>> bearing = cylindrical_bearing_example()
    >>> bearing.run()
    >>> bearing.Fhy
    112814.91
    """
    @check_units
    def __init__(
        self,
        L,
        R,
        c_r,
        n_theta,
        n_z,
        n_y,
        n_gap,
        n_pad,
        betha_s,
        mu_ref,
        speed,
        Wx,
        Wy,
        k_t,
        Cp,
        rho,
        T_reserv,
        fat_mixt,
        sommerfeld_type=2,
    ):

        self.L = L
        self.R = R
        self.c_r = c_r
        self.n_theta = n_theta
        self.n_z = n_z
        self.n_y = n_y
        self.n_gap = n_gap
        self.n_pad = n_pad
        self.mu_ref = mu_ref
        self.speed = speed
        self.Wx = Wx
        self.Wy = Wy
        self.k_t = k_t
        self.Cp = Cp
        self.rho = rho
        self.T_reserv = T_reserv
        self.fat_mixt = fat_mixt
        self.equilibrium_pos = None
        self.sommerfeld_type = sommerfeld_type

        if self.n_y == None:
            self.n_y = self.n_theta

        self.betha_s = betha_s * np.pi / 180

        self.thetaI = 0
        self.thetaF = self.betha_s
        self.dtheta = (self.thetaF - self.thetaI) / (self.n_theta)

        ## Plot ## Remove later

        self.Ytheta = np.zeros(2 * (self.n_theta + self.n_gap) + 2)

        self.Ytheta[1:-1] = np.arange(0.5 * self.dtheta, 2 * np.pi, self.dtheta)
        self.Ytheta[0] = 0
        self.Ytheta[-1] = 2 * np.pi

        ##
        # Dimensionless discretization variables

        self.dY = 1 / self.n_y
        self.dZ = 1 / self.n_z

        # Z-axis direction

        self.Z_I = 0
        self.Z_F = 1
        Z = np.zeros((self.n_z + 2))

        Z[0] = self.Z_I
        Z[self.n_z + 1] = self.Z_F
        Z[1 : self.n_z + 1] = np.arange(self.Z_I + 0.5 * self.dZ, self.Z_F, self.dZ)
        self.Z = Z

        # Dimensionalization

        self.dz = self.dZ * self.L
        self.dy = self.dY * self.betha_s * self.R

        self.Zdim = self.Z * L

    def _forces(self, x0, y0, xpt0, ypt0):
        """Calculates the forces in Y and X direction.

        Parameters
        ----------
        x0 : array, float
            If the other parameters are None, x0 is an array with eccentricity ratio and attitude angle.
            Else, x0 is the position of the center of the rotor in the x-axis.
        y0 : float
            The position of the center of the rotor in the y-axis.
        xpt0 : float
            The speed of the center of the rotor in the x-axis.
        ypt0 : float
            The speed of the center of the rotor in the y-axis.


        Returns
        -------
        Fhx : float
            Force in X direction. The unit is newton.
        Fhy : float
            Force in Y direction. The unit is newton.
        """
        if y0 is None and xpt0 is None and ypt0 is None:
            self.x0 = x0

            xr = self.x0[0] * self.c_r * np.cos(self.x0[1])
            yr = self.x0[0] * self.c_r * np.sin(self.x0[1])
            self.Y = yr / self.c_r
            self.X = xr / self.c_r

            self.Xpt = 0
            self.Ypt = 0
        else:
            self.X = x0 / self.c_r
            self.Y = y0 / self.c_r

            self.Xpt = xpt0 / (self.c_r * self.speed)
            self.Ypt = ypt0 / (self.c_r * self.speed)

        for i in range(4):

            P = np.zeros((self.n_z, self.n_theta, self.n_pad))
            dPdy = np.zeros((self.n_z, self.n_theta, self.n_pad))
            dPdz = np.zeros((self.n_z, self.n_theta, self.n_pad))
            T = np.ones((self.n_z, self.n_theta, self.n_pad))
            T_new = np.ones((self.n_z, self.n_theta, self.n_pad)) * 1.2

            if i == 0:
                T_mist = self.T_reserv * np.ones(self.n_pad)

            mi_new = 1.1 * np.ones((self.n_z, self.n_theta, self.n_pad))
            PPlot = np.zeros(((self.n_z + 2), (len(self.Ytheta))))
            auxF = np.zeros((2, len(self.Ytheta)))

            nk = (self.n_z) * (self.n_theta)

            Mat_coef = np.zeros((nk, nk))
            Mat_coef_T = np.zeros((nk, nk))
            b = np.zeros((nk, 1))
            b_T = np.zeros((nk, 1))

            for n_p in np.arange(self.n_pad):

                self.thetaI = (
                    n_p * self.betha_s
                    + self.dtheta * self.n_gap / 2
                    + (n_p * self.dtheta * self.n_gap)
                )

                self.thetaF = self.thetaI + self.betha_s

                self.dtheta = (self.thetaF - self.thetaI) / (self.n_theta)

                T_ref = T_mist[n_p - 1]

                # Temperature convergence while

                while (
                    np.linalg.norm(T_new[:, :, n_p] - T[:, :, n_p])
                    / np.linalg.norm(T[:, :, n_p])
                    >= 1e-3
                ):

                    T_ref = T_mist[n_p - 1]

                    mi = mi_new

                    T[:, :, n_p] = T_new[:, :, n_p]

                    ki = 0
                    kj = 0
                    k = 0

                    # Solution of pressure field initialization

                    for ii in np.arange((self.Z_I + 0.5 * self.dZ), self.Z_F, self.dZ):
                        for jj in np.arange(
                            self.thetaI + (self.dtheta / 2), self.thetaF, self.dtheta
                        ):

                            hP = 1 - self.X * np.cos(jj) - self.Y * np.sin(jj)
                            he = (
                                1
                                - self.X * np.cos(jj + 0.5 * self.dtheta)
                                - self.Y * np.sin(jj + 0.5 * self.dtheta)
                            )
                            hw = (
                                1
                                - self.X * np.cos(jj - 0.5 * self.dtheta)
                                - self.Y * np.sin(jj - 0.5 * self.dtheta)
                            )
                            hn = hP
                            hs = hn

                            if kj == 0 and ki == 0:
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = mi[ki, kj]
                                MI_s = mi[ki, kj]
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])

                            if kj == 0 and ki > 0 and ki < self.n_z - 1:
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = mi[ki, kj]
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])

                            if kj == 0 and ki == self.n_z - 1:
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = mi[ki, kj]
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = mi[ki, kj]

                            if ki == 0 and kj > 0 and kj < self.n_theta - 1:
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = mi[ki, kj]
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])

                            if (
                                kj > 0
                                and kj < self.n_theta - 1
                                and ki > 0
                                and ki < self.n_z - 1
                            ):
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])

                            if ki == self.n_z - 1 and kj > 0 and kj < self.n_theta - 1:
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = mi[ki, kj]

                            if ki == 0 and kj == self.n_theta - 1:
                                MI_e = mi[ki, kj]
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = mi[ki, kj]
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])

                            if kj == self.n_theta - 1 and ki > 0 and ki < self.n_z - 1:
                                MI_e = mi[ki, kj]
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])

                            if kj == self.n_theta - 1 and ki == self.n_z - 1:
                                MI_e = mi[ki, kj]
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = mi[ki, kj]

                            CE = (self.dZ * he ** 3) / (
                                12 * MI_e[n_p] * self.dY * self.betha_s ** 2
                            )
                            CW = (self.dZ * hw ** 3) / (
                                12 * MI_w[n_p] * self.dY * self.betha_s ** 2
                            )
                            CN = (self.dY * (self.R ** 2) * hn ** 3) / (
                                12 * MI_n[n_p] * self.dZ * self.L ** 2
                            )
                            CS = (self.dY * (self.R ** 2) * hs ** 3) / (
                                12 * MI_s[n_p] * self.dZ * self.L ** 2
                            )
                            CP = -(CE + CW + CN + CS)

                            B = (self.dZ / (2 * self.betha_s)) * (he - hw) - (
                                (self.Ypt * np.cos(jj) + self.Xpt * np.sin(jj))
                                * self.dy
                                * self.dZ
                            )

                            k = k + 1
                            b[k - 1, 0] = B

                            if ki == 0 and kj == 0:
                                Mat_coef[k - 1, k - 1] = CP - CS - CW
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k + self.n_theta - 1] = CN

                            elif kj == 0 and ki > 0 and ki < self.n_z - 1:
                                Mat_coef[k - 1, k - 1] = CP - CW
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - self.n_theta - 1] = CS
                                Mat_coef[k - 1, k + self.n_theta - 1] = CN

                            elif kj == 0 and ki == self.n_z - 1:
                                Mat_coef[k - 1, k - 1] = CP - CN - CW
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - self.n_theta - 1] = CS

                            elif ki == 0 and kj > 0 and kj < self.n_y - 1:
                                Mat_coef[k - 1, k - 1] = CP - CS
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k + self.n_theta - 1] = CN

                            elif (
                                ki > 0
                                and ki < self.n_z - 1
                                and kj > 0
                                and kj < self.n_y - 1
                            ):
                                Mat_coef[k - 1, k - 1] = CP
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k - self.n_theta - 1] = CS
                                Mat_coef[k - 1, k + self.n_theta - 1] = CN
                                Mat_coef[k - 1, k] = CE

                            elif ki == self.n_z - 1 and kj > 0 and kj < self.n_y - 1:
                                Mat_coef[k - 1, k - 1] = CP - CN
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k - self.n_theta - 1] = CS

                            elif ki == 0 and kj == self.n_y - 1:
                                Mat_coef[k - 1, k - 1] = CP - CE - CS
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k + self.n_theta - 1] = CN

                            elif kj == self.n_y - 1 and ki > 0 and ki < self.n_z - 1:
                                Mat_coef[k - 1, k - 1] = CP - CE
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k - self.n_theta - 1] = CS
                                Mat_coef[k - 1, k + self.n_theta - 1] = CN

                            elif ki == self.n_z - 1 and kj == self.n_y - 1:
                                Mat_coef[k - 1, k - 1] = CP - CE - CN
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k - self.n_theta - 1] = CS

                            kj = kj + 1

                        kj = 0
                        ki = ki + 1

                    # Solution of pressure field end

                    p = np.dot(pinv(Mat_coef), b)

                    cont = 0

                    for i in np.arange(self.n_z):
                        for j in np.arange(self.n_theta):

                            P[i, j, n_p] = p[cont]
                            cont = cont + 1

                            if P[i, j, n_p] < 0:
                                P[i, j, n_p] = 0

                    # Dimensional pressure fied

                    Pdim = (P * self.mu_ref * self.speed * (self.R ** 2)) / (
                        self.c_r ** 2
                    )

                    ki = 0
                    kj = 0
                    k = 0

                    # Solution of temperature field initialization

                    for ii in np.arange(
                        (self.Z_I + 0.5 * self.dZ), (self.Z_F), self.dZ
                    ):
                        for jj in np.arange(
                            self.thetaI + (self.dtheta / 2), self.thetaF, self.dtheta
                        ):

                            # Pressure gradients

                            if kj == 0 and ki == 0:
                                dPdy[ki, kj, n_p] = (P[ki, kj + 1, n_p] - 0) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (P[ki + 1, kj, n_p] - 0) / (
                                    2 * self.dZ
                                )

                            if kj == 0 and ki > 0 and ki < self.n_z - 1:
                                dPdy[ki, kj, n_p] = (P[ki, kj + 1, n_p] - 0) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (
                                    P[ki + 1, kj, n_p] - P[ki - 1, kj, n_p]
                                ) / (2 * self.dZ)

                            if kj == 0 and ki == self.n_z - 1:
                                dPdy[ki, kj, n_p] = (P[ki, kj + 1, n_p] - 0) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (0 - P[ki - 1, kj, n_p]) / (
                                    2 * self.dZ
                                )

                            if ki == 0 and kj > 0 and kj < self.n_theta - 1:
                                dPdy[ki, kj, n_p] = (
                                    P[ki, kj + 1, n_p] - P[ki, kj - 1, n_p]
                                ) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (P[ki + 1, kj, n_p] - 0) / (
                                    2 * self.dZ
                                )

                            if (
                                kj > 0
                                and kj < self.n_theta - 1
                                and ki > 0
                                and ki < self.n_z - 1
                            ):
                                dPdy[ki, kj, n_p] = (
                                    P[ki, kj + 1, n_p] - P[ki, kj - 1, n_p]
                                ) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (
                                    P[ki + 1, kj, n_p] - P[ki - 1, kj, n_p]
                                ) / (2 * self.dZ)

                            if ki == self.n_z - 1 and kj > 0 and kj < self.n_theta - 1:
                                dPdy[ki, kj, n_p] = (
                                    P[ki, kj + 1, n_p] - P[ki, kj - 1, n_p]
                                ) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (0 - P[ki - 1, kj, n_p]) / (
                                    2 * self.dZ
                                )

                            if ki == 0 and kj == self.n_theta - 1:
                                dPdy[ki, kj, n_p] = (0 - P[ki, kj - 1, n_p]) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (P[ki + 1, kj, n_p] - 0) / (
                                    2 * self.dZ
                                )

                            if kj == self.n_theta - 1 and ki > 0 and ki < self.n_z - 1:
                                dPdy[ki, kj, n_p] = (0 - P[ki, kj - 1, n_p]) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (
                                    P[ki + 1, kj, n_p] - P[ki - 1, kj, n_p]
                                ) / (2 * self.dZ)

                            if kj == self.n_theta - 1 and ki == self.n_z - 1:
                                dPdy[ki, kj, n_p] = (0 - P[ki, kj - 1, n_p]) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (0 - P[ki - 1, kj, n_p]) / (
                                    2 * self.dZ
                                )

                            HP = 1 - self.X * np.cos(jj) - self.Y * np.sin(jj)
                            hpt = -self.Ypt * np.cos(jj) + self.Xpt * np.sin(jj)

                            mi_p = mi[ki, kj, n_p]

                            AE = -(self.k_t * HP * self.dZ) / (
                                self.rho
                                * self.Cp
                                * self.speed
                                * ((self.betha_s * R) ** 2)
                                * self.dY
                            )
                            AW = (
                                (
                                    ((HP ** 3) * dPdy[ki, kj, n_p] * self.dZ)
                                    / (12 * mi_p * (self.betha_s ** 2))
                                )
                                - ((HP) * self.dZ / (2 * self.betha_s))
                                - (
                                    (self.k_t * HP * self.dZ)
                                    / (
                                        self.rho
                                        * self.Cp
                                        * self.speed
                                        * ((self.betha_s * R) ** 2)
                                        * self.dY
                                    )
                                )
                            )
                            AN = -(self.k_t * HP * self.dY) / (
                                self.rho
                                * self.Cp
                                * self.speed
                                * (self.L ** 2)
                                * self.dZ
                            )
                            AS = (
                                (
                                    (self.R ** 2)
                                    * (HP ** 3)
                                    * dPdz[ki, kj, n_p]
                                    * self.dY
                                )
                                / (12 * (self.L ** 2) * mi_p)
                            ) - (
                                (self.k_t * HP * self.dY)
                                / (
                                    self.rho
                                    * self.Cp
                                    * self.speed
                                    * (self.L ** 2)
                                    * self.dZ
                                )
                            )
                            AP = -(AE + AW + AN + AS)

                            auxb_T = (self.speed * self.mu_ref) / (
                                self.rho * self.Cp * self.T_reserv * self.c_r
                            )
                            b_TG = (
                                self.mu_ref
                                * self.speed
                                * (R ** 2)
                                * self.dY
                                * self.dZ
                                * P[ki, kj, n_p]
                                * hpt
                            ) / (self.rho * self.Cp * self.T_reserv * (self.c_r ** 2))
                            b_TH = (
                                self.speed
                                * self.mu_ref
                                * (hpt ** 2)
                                * 4
                                * mi_p
                                * self.dY
                                * self.dZ
                            ) / (self.rho * self.Cp * self.T_reserv * 3 * HP)
                            b_TI = (
                                auxb_T
                                * (mi_p * (self.R ** 2) * self.dY * self.dZ)
                                / (HP * self.c_r)
                            )
                            b_TJ = (
                                auxb_T
                                * (
                                    (self.R ** 2)
                                    * (HP ** 3)
                                    * (dPdy[ki, kj, n_p] ** 2)
                                    * self.dY
                                    * self.dZ
                                )
                                / (12 * self.c_r * (self.betha_s ** 2) * mi_p)
                            )
                            b_TK = (
                                auxb_T
                                * (
                                    (self.R ** 4)
                                    * (HP ** 3)
                                    * (dPdz[ki, kj, n_p] ** 2)
                                    * self.dY
                                    * self.dZ
                                )
                                / (12 * self.c_r * (self.L ** 2) * mi_p)
                            )

                            B_T = b_TG + b_TH + b_TI + b_TJ + b_TK

                            k = k + 1

                            b_T[k - 1, 0] = B_T

                            if ki == 0 and kj == 0:
                                Mat_coef_T[k - 1, k - 1] = AP + AS - AW
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[k - 1, k + self.n_theta - 1] = AN
                                b_T[k - 1, 0] = b_T[k - 1, 0] - 2 * AW * (
                                    T_ref / self.T_reserv
                                )

                            elif kj == 0 and ki > 0 and ki < self.n_z - 1:
                                Mat_coef_T[k - 1, k - 1] = AP - AW
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[k - 1, k - self.n_theta - 1] = AS
                                Mat_coef_T[k - 1, k + self.n_theta - 1] = AN
                                b_T[k - 1, 0] = b_T[k - 1, 0] - 2 * AW * (
                                    T_ref / self.T_reserv
                                )

                            elif kj == 0 and ki == self.n_z - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AN - AW
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[k - 1, k - self.n_theta - 1] = AS
                                b_T[k - 1, 0] = b_T[k - 1, 0] - 2 * AW * (
                                    T_ref / self.T_reserv
                                )

                            elif ki == 0 and kj > 0 and kj < self.n_y - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AS
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[k - 1, k + self.n_theta - 1] = AN

                            elif (
                                ki > 0
                                and ki < self.n_z - 1
                                and kj > 0
                                and kj < self.n_y - 1
                            ):
                                Mat_coef_T[k - 1, k - 1] = AP
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[k - 1, k - self.n_theta - 1] = AS
                                Mat_coef_T[k - 1, k + self.n_theta - 1] = AN
                                Mat_coef_T[k - 1, k] = AE

                            elif ki == self.n_z - 1 and kj > 0 and kj < self.n_y - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AN
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[k - 1, k - self.n_theta - 1] = AS

                            elif ki == 0 and kj == self.n_y - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AE + AS
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[k - 1, k + self.n_theta - 1] = AN

                            elif kj == self.n_y - 1 and ki > 0 and ki < self.n_z - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AE
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[k - 1, k - self.n_theta - 1] = AS
                                Mat_coef_T[k - 1, k + self.n_theta - 1] = AN

                            elif ki == self.n_z - 1 and kj == self.n_y - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AE + AN
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[k - 1, k - self.n_theta - 1] = AS

                            kj = kj + 1

                        kj = 0
                        ki = ki + 1

                    # Solution of temperature field end

                    t = np.dot(pinv(Mat_coef_T), b_T)

                    cont = 0

                    for i in np.arange(self.n_z):
                        for j in np.arange(self.n_theta):

                            T_new[i, j, n_p] = t[cont]
                            cont = cont + 1

                    Tdim = T_new * self.T_reserv

                    T_end = np.sum(Tdim[:, -1, n_p]) / self.n_z

                    T_mist[n_p] = (
                        self.fat_mixt * self.T_reserv + (1 - self.fat_mixt) * T_end
                    )

                    for i in np.arange(self.n_z):
                        for j in np.arange(self.n_theta):

                            mi_new[i, j, n_p] = (
                                6.4065 * (Tdim[i, j, n_p]) ** -1.475
                            ) / self.mu_ref

        ## Plot  ## Remove later

        cont = 0
        for n_p in np.arange(self.n_pad):
            for ii in np.arange(1, self.n_z + 1):
                cont = (
                    1
                    + (n_p) * (self.n_gap / 2)
                    + (n_p) * (self.n_theta + self.n_gap / 2)
                )
                for jj in np.arange(1, self.n_theta + 1):

                    PPlot[ii, int(cont)] = Pdim[int(ii - 1), int(jj - 1), int(n_p)]
                    cont = cont + 1

        self.PPlot = PPlot

        ##

        auxF = np.zeros((2, len(self.Ytheta[1:-1])))

        auxF[0, :] = np.cos(self.Ytheta[1:-1])
        auxF[1, :] = np.sin(self.Ytheta[1:-1])

        dA = self.dy * self.dz

        auxP = PPlot[1:-1, 1:-1] * dA

        vector_auxF_x = auxF[0, :]
        vector_auxF_y = auxF[1, :]

        auxFx = auxP * vector_auxF_x
        auxFy = auxP * vector_auxF_y

        fxj = -np.sum(auxFx)
        fyj = -np.sum(auxFy)

        Fhx = fxj
        Fhy = fyj

        self.Fhx = Fhx
        self.Fhy = Fhy

        return Fhx, Fhy

    def run(self, x, plot_pressure=False, print_progress=False):
        """This method runs the optimization to find the equilibrium position of the rotor's center.

        Parameters
        ----------
        x : array
            Array with eccentricity ratio and attitude angle
        print_progress : bool
            Set it True to print the score and forces on each iteration.
            False by default.
        """
        args = print_progress
        t1 = time.time()
        res = minimize(
            self._score,
            x,
            args,
            method="Nelder-Mead",
            tol=10e-3,
            options={"maxiter": 1000},
        )
        self.equilibrium_pos = res.x
        t2 = time.time()
        print(res)
        print(f"Time Spent: {t2-t1} seconds")

        if plot_pressure:
            self._plotPressure()

    def coefficients(self, show_coef=False):
        """Calculates the dynamic coefficients of stiffness "k" and damping "c". The formulation is based in application of virtual displacements and speeds on the rotor from its equilibrium position to determine the bearing stiffness and damping coefficients.

        Parameters
        ----------
        show_coef : bool
            Set it True, to print the calculated coefficients.
            False by default.

        Returns
        -------
        coefs : tuple
            Bearing stiffness and damping coefficients. 
            Its shape is: ((kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy))

        """
        if self.equilibrium_pos is None:
            self.run([0.1, -0.1], True, True)
            self.coefficients()
        else:
            xeq = self.equilibrium_pos[0] * self.c_r * np.cos(self.equilibrium_pos[1])
            yeq = self.equilibrium_pos[0] * self.c_r * np.sin(self.equilibrium_pos[1])

            dE = 0.001
            epix = np.abs(dE * self.c_r * np.cos(self.equilibrium_pos[1]))
            epiy = np.abs(dE * self.c_r * np.sin(self.equilibrium_pos[1]))

            Va = self.speed * (self.R)
            epixpt = 0.000001 * np.abs(Va * np.sin(self.equilibrium_pos[1]))
            epiypt = 0.000001 * np.abs(Va * np.cos(self.equilibrium_pos[1]))

            Aux01 = self._forces(xeq + epix, yeq, 0, 0)
            Aux02 = self._forces(xeq - epix, yeq, 0, 0)
            Aux03 = self._forces(xeq, yeq + epiy, 0, 0)
            Aux04 = self._forces(xeq, yeq - epiy, 0, 0)

            Aux05 = self._forces(xeq, yeq, epixpt, 0)
            Aux06 = self._forces(xeq, yeq, -epixpt, 0)
            Aux07 = self._forces(xeq, yeq, 0, epiypt)
            Aux08 = self._forces(xeq, yeq, 0, -epiypt)

            # Ss = self.sommerfeld(Aux08[0],Aux08[1])

            Kxx = -self.sommerfeld(Aux01[0], Aux02[1]) * (
                (Aux01[0] - Aux02[0]) / (epix / self.c_r)
            )
            Kxy = -self.sommerfeld(Aux03[0], Aux04[1]) * (
                (Aux03[0] - Aux04[0]) / (epiy / self.c_r)
            )
            Kyx = -self.sommerfeld(Aux01[1], Aux02[1]) * (
                (Aux01[1] - Aux02[1]) / (epix / self.c_r)
            )
            Kyy = -self.sommerfeld(Aux03[1], Aux04[1]) * (
                (Aux03[1] - Aux04[1]) / (epiy / self.c_r)
            )

            Cxx = -self.sommerfeld(Aux05[0], Aux06[0]) * (
                (Aux05[0] - Aux06[0]) / (epixpt / self.c_r / self.speed)
            )
            Cxy = -self.sommerfeld(Aux07[0], Aux08[0]) * (
                (Aux07[0] - Aux08[0]) / (epiypt / self.c_r / self.speed)
            )
            Cyx = -self.sommerfeld(Aux05[1], Aux06[1]) * (
                (Aux05[1] - Aux06[1]) / (epixpt / self.c_r / self.speed)
            )
            Cyy = -self.sommerfeld(Aux07[1], Aux08[1]) * (
                (Aux07[1] - Aux08[1]) / (epiypt / self.c_r / self.speed)
            )

            kxx = (np.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / self.c_r) * Kxx
            kxy = (np.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / self.c_r) * Kxy
            kyx = (np.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / self.c_r) * Kyx
            kyy = (np.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / self.c_r) * Kyy

            cxx = (
                np.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / (self.c_r * self.speed)
            ) * Cxx
            cxy = (
                np.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / (self.c_r * self.speed)
            ) * Cxy
            cyx = (
                np.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / (self.c_r * self.speed)
            ) * Cyx
            cyy = (
                np.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / (self.c_r * self.speed)
            ) * Cyy

            if show_coef:
                print(f"kxx = {kxx}")
                print(f"kxy = {kxy}")
                print(f"kyx = {kyx}")
                print(f"kyy = {kyy}")

                print(f"cxx = {cxx}")
                print(f"cxy = {cxy}")
                print(f"cyx = {cyx}")
                print(f"cyy = {cyy}")

            coefs = ((kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy))

            return coefs

    def _score(self, x, print_progress=False):
        """This method used to set the objective function of minimize optimization.

        Parameters
        ==========
        score: float
           Balanced Force expression between the load aplied in bearing and the
           resultant force provide by oil film.

        Returns
        ========
        Score coefficient.

        """
        Fhx, Fhy = self._forces(x, None, None, None)
        score = np.sqrt(((self.Wx + Fhx) ** 2) + ((self.Wy + Fhy) ** 2))
        if print_progress:
            print(f"Score: ", score)
            print("============================================")
            print(f"Força na direção x: ", Fhx)
            print("============================================")
            print(f"Força na direção y: ", Fhy)
            print("")

        return score

    def sommerfeld(self, force_x, force_y):
        """Calculate the sommerfeld number. This dimensionless number is used to calculate the dynamic coeficients.

        Parameters
        ----------
        force_x : float
            Force in x direction. The unit is newton.
        force_y : float
            Force in y direction. The unit is newton.

        Returns
        -------
        Ss : float
            Sommerfeld number.
        """
        if self.sommerfeld_type == 1:
            S = (self.mu_ref * ((self.R) ** 3) * self.L * self.speed) / (
                np.pi * (self.c_r ** 2) * np.sqrt((self.Wx ** 2) + (self.Wy ** 2))
            )

        elif self.sommerfeld_type == 2:
            S = 1 / (
                2
                * ((self.L / (2 * self.R)) ** 2)
                * (np.sqrt((force_x ** 2) + (force_y ** 2)))
            )

        Ss = S * ((self.L / (2 * self.R)) ** 2)

        return Ss

    def _plotPressure(self):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        Ydim, Zdim = np.meshgrid(self.Ytheta, self.Zdim)
        surf = ax.plot_surface(
            Ydim, Zdim, self.PPlot, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        plt.show()

def cylindrical_bearing_example():
    """[summary]

    Returns:
        [type]: [description]
    """

    bearing = THDCylindrical(
        L=0.263144,
        R=0.2,
        c_r=1.945e-4,
        n_theta=41,
        n_z=10,
        n_gap=1,
        betha_s=176,
        mu_ref=0.02,
        speed=Q_(900,"RPM"),
        Wx=0,
        Wy=-112814.91,
        k_t=0.15327,
        Cp=1915.5,
        rho=854.952,
        Treserv=50,
        fat_mixt=0.52,
    )
    
    return bearing

if __name__ == "__main__":

    x0 = [0.1, -0.1]
    L = float(0.263144)  # [metros]
    R = float(0.2)  # [metros]
    Cr = float(1.945e-4)  # [metros]
    nTheta = int(41)
    nZ = int(10)
    nY = None

    mu = float(0.02)  # [Ns/m²]
    speed = Q_(900,"RPM")  # [RPM]
    Wx = float(0)  # [N]
    Wy = float(-112814.91)  # [N]
    k = float(0.15327)  # Thermal conductivity [J/s.m.°C]
    Cp = float(1915.5)  # Specific heat [J/kg°C]
    rho = float(854.952)  # Specific mass [kg/m³]
    Treserv = float(50)  # Temperature of oil tank [ºC]
    mix = float(0.52)  # Mixing factor. Used because the oil supply flow is not known.
    nGap = int(1)  #    Number of volumes in recess zone
    nPad = int(2)  #    Number of pads
    betha_s = 176

    mancal = THDCylindrical(
        L,
        R,
        Cr,
        nTheta,
        nZ,
        nY,
        nGap,
        nPad,
        betha_s,
        mu,
        speed,
        Wx,
        Wy,
        k,
        Cp,
        rho,
        Treserv,
        mix,
    )
    mancal.run(
        x0, print_progress=True, plot_pressure=True, method="Riman", max_iter=1000
    )
    # mancal.coefficients()