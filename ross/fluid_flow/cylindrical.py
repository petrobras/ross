
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv
import scipy as sci
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import math
import sys

class THDCylindrical:
    def __init__(
        self,
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
        Tref,
        mix,
    ):

        self.L = L  # L = float(0.263144)      # [metros]
        self.R = R  # R = float(0.2)           # [metros]
        self.Cr = Cr  # Cr = float(1.945e-4)     # [metros]
        self.ntheta = nTheta  # ntheta = int(38)
        self.nZ = nZ  # nZ = int(30)
        self.nY = nY  # nY = ntheta
        self.ngap = nGap  # ngap= int(2) #    Number of volumes in recess zone
        self.n_pad = nPad  # n_pad=int(2) #    Number of pads
        # self.betha_s = betha_s # betha_s = 170
        self.mi_ref = mu  # mi_ref = float(0.02)     # [Ns/m²]
        self.wa = speed  # wa = float(900)     # [RPM]
        self.Wx = Wx  # Wx = float(0)    # [N]
        self.Wy = Wy  # Wy = float(-112814.91)    # [N]
        self.kt = k  # kt=float(0.15327)     #Thermal conductivity [J/s.m.°C]
        self.Cp = Cp  # Cp=float(1800.24)      #Specific heat [J/kg°C]
        self.rho = rho  # rho=float(880)    #Specific mass [kg/m³]
        self.Treserv = Tref  # Treserv=float(50)      #Temperature of oil tank [ºC]
        self.fat_mist = mix  # fat_mist=float(0.8)    # Mixing factor. Used because the oil supply flow is not known.

        # self.T_ref = self.Treserv  # Reference temperature [ºC]

        if self.nY == None:
            self.nY = self.ntheta

        self.war = (self.wa * np.pi) / 30  # Transforma de rpm para rad/s
        self.betha_s = betha_s * np.pi / 180  # [rad]

        self.theta1 = 0  # initial coordinate theta [rad]
        self.theta2 = self.betha_s  # final coordinate theta [rad]
        self.dtheta = (self.theta2 - self.theta1) / (self.ntheta)

        self.dY = 1 / self.nY
        self.dZ = 1 / self.nZ

        self.Ytheta = np.zeros(2 * (self.ntheta + self.ngap) + 2)

        self.Ytheta[1:-1] = np.arange(0.5 * self.dtheta, 2 * np.pi, self.dtheta)
        self.Ytheta[0] = 0
        self.Ytheta[-1] = 2 * np.pi

        Z1 = 0  # initial coordinate z dimensionless
        Z2 = 1
        Z = np.zeros((self.nZ + 2))
        Z[0] = Z1
        Z[self.nZ + 1] = Z2
        Z[1 : self.nZ + 1] = np.arange(
            Z1 + 0.5 * self.dZ, Z2, self.dZ
        )  # vector z dimensionless
        self.Z = Z
        self.Zdim = self.Z * L

        # def THDEquilibrio(x0):
        #    global p
        # Dimensioless

        # h=Cr-(yr*np.cos(theta))-(xr*np.sin(theta))

        self.dz = self.dZ * self.L
        self.dy = self.dY * self.betha_s * self.R

        self.equilibrium_pos = None

    def _forces(self, x0, y0, xpt0, ypt0):
        if y0 is None and xpt0 is None and ypt0 is None:
            self.x0 = x0

            xr = (
                self.x0[0] * self.Cr * np.sin(self.x0[1])
            )  # Representa a posição do centro do eixo ao longo da direção "Y"
            yr = (
                self.x0[0] * self.Cr * np.cos(self.x0[1])
            )  # Representa a posição do centro do eixo ao longo da direção "X"
            self.Y = yr / self.Cr  # Representa a posição em x adimensional
            self.X = xr / self.Cr

            self.Xpt = 0
            self.Ypt = 0
        else:
            self.X = x0 / self.Cr
            self.Y = y0 / self.Cr

            self.Xpt = xpt0 / (self.Cr * self.war)
            self.Ypt = ypt0 / (self.Cr * self.war)
            
        for i in range(4):
            P = np.zeros((self.nZ, self.ntheta, self.n_pad))
            dPdy = np.zeros((self.nZ, self.ntheta, self.n_pad))
            dPdz = np.zeros((self.nZ, self.ntheta, self.n_pad))
            T = np.ones((self.nZ, self.ntheta, self.n_pad))
            T_new = np.ones((self.nZ, self.ntheta, self.n_pad)) * 1.2
            
            if i == 0:
                T_mist_aux = self.Treserv * np.ones(self.n_pad)
                
            mi_new = 1.1*np.ones((self.nZ, self.ntheta, self.n_pad))
            PPlot = np.zeros(((self.nZ + 2), (len(self.Ytheta))))
            auxF = np.zeros((2, len(self.Ytheta)))
    
            nk = (self.nZ) * (self.ntheta)
    
            Mat_coef = np.zeros((nk, nk))
            Mat_coef_t = np.zeros((nk, nk))
            b = np.zeros((nk, 1))
            b_t = np.zeros((nk, 1))
    
            for n_p in np.arange(self.n_pad):
    
                T_ref = T_mist_aux[n_p-1]
    
                self.theta1 = (
                    n_p * self.betha_s
                    + self.dtheta * self.ngap / 2
                    + (n_p * self.dtheta * self.ngap)
                )
    
                self.theta2 = self.theta1 + self.betha_s
    
                # self.dtheta = (self.theta2 - self.theta1) / (self.ntheta)
    
                Z1 = 0  # initial coordinate z dimensionless
                Z2 = 1  # final coordinate z dimensionless
    
                while (
                    np.linalg.norm(T_new[:, :, n_p] - T[:, :, n_p])
                    / np.linalg.norm(T[:, :, n_p])
                    >= 1e-3
                ):
                    # print(
                    #     np.linalg.norm(T_new[:, :, n_p] - T[:, :, n_p])
                    #     / np.linalg.norm(T[:, :, n_p])
                    # )
    
                    ki = 0
                    kj = 0
                    
                    T_ref = T_mist_aux[n_p-1]
                    
                    mi = mi_new
    
                    T[:, :, n_p] = T_new[:, :, n_p]
    
                    k = 0  # vectorization pressure index
    
                    for ii in np.arange((Z1 + 0.5 * self.dZ), Z2, self.dZ):
                        for jj in np.arange(
                            self.theta1 + (self.dtheta / 2), self.theta2, self.dtheta
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
    
                            if kj == 0 and ki > 0 and ki < self.nZ - 1:
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = mi[ki, kj]
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])
    
                            if kj == 0 and ki == self.nZ - 1:
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = mi[ki, kj]
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = mi[ki, kj]
    
                            if ki == 0 and kj > 0 and kj < self.ntheta - 1:
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = mi[ki, kj]
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])
    
                            if (
                                kj > 0
                                and kj < self.ntheta - 1
                                and ki > 0
                                and ki < self.nZ - 1
                            ):
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])
    
                            if ki == self.nZ - 1 and kj > 0 and kj < self.ntheta - 1:
                                MI_e = 0.5 * (mi[ki, kj] + mi[ki, kj + 1])
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = mi[ki, kj]
    
                            if ki == 0 and kj == self.ntheta - 1:
                                MI_e = mi[ki, kj]
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = mi[ki, kj]
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])
    
                            if kj == self.ntheta - 1 and ki > 0 and ki < self.nZ - 1:
                                MI_e = mi[ki, kj]
                                MI_w = 0.5 * (mi[ki, kj] + mi[ki, kj - 1])
                                MI_s = 0.5 * (mi[ki, kj] + mi[ki - 1, kj])
                                MI_n = 0.5 * (mi[ki, kj] + mi[ki + 1, kj])
    
                            if kj == self.ntheta - 1 and ki == self.nZ - 1:
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
                                Mat_coef[k - 1, k + self.ntheta - 1] = CN
    
                            elif kj == 0 and ki > 0 and ki < self.nZ - 1:
                                Mat_coef[k - 1, k - 1] = CP - CW
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - self.ntheta - 1] = CS
                                Mat_coef[k - 1, k + self.ntheta - 1] = CN
    
                            elif kj == 0 and ki == self.nZ - 1:
                                Mat_coef[k - 1, k - 1] = CP - CN - CW
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - self.ntheta - 1] = CS
    
                            elif ki == 0 and kj > 0 and kj < self.nY - 1:
                                Mat_coef[k - 1, k - 1] = CP - CS
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k + self.ntheta - 1] = CN
    
                            elif (
                                ki > 0 and ki < self.nZ - 1 and kj > 0 and kj < self.nY - 1
                            ):
                                Mat_coef[k - 1, k - 1] = CP
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k - self.ntheta - 1] = CS
                                Mat_coef[k - 1, k + self.ntheta - 1] = CN
                                Mat_coef[k - 1, k] = CE
    
                            elif ki == self.nZ - 1 and kj > 0 and kj < self.nY - 1:
                                Mat_coef[k - 1, k - 1] = CP - CN
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k - self.ntheta - 1] = CS
    
                            elif ki == 0 and kj == self.nY - 1:
                                Mat_coef[k - 1, k - 1] = CP - CE - CS
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k + self.ntheta - 1] = CN
    
                            elif kj == self.nY - 1 and ki > 0 and ki < self.nZ - 1:
                                Mat_coef[k - 1, k - 1] = CP - CE
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k - self.ntheta - 1] = CS
                                Mat_coef[k - 1, k + self.ntheta - 1] = CN
    
                            elif ki == self.nZ - 1 and kj == self.nY - 1:
                                Mat_coef[k - 1, k - 1] = CP - CE - CN
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[k - 1, k - self.ntheta - 1] = CS
    
                            kj = kj + 1
    
                        kj = 0
                        ki = ki + 1
    
                    #    %%%%%%%%%%%%%%%%%%%%%% Solution of pressure field %%%%%%%%%%%%%%%%%%%%
    
                    p = np.linalg.solve(Mat_coef, b)
    
                    cont = 0
    
                    for i in np.arange(self.nZ):
                        for j in np.arange(self.ntheta):
    
                            P[i, j, n_p] = p[cont]
                            cont = cont + 1
    
                    #    %Boundary condiction of pressure
    
                    for i in np.arange(self.nZ):
                        for j in np.arange(self.ntheta):
                            if P[i, j, n_p] < 0:
                                P[i, j, n_p] = 0
    
                    #    % Dimensional pressure fied [Pa]
    
                    Pdim = (P * self.mi_ref * self.war * (self.R ** 2)) / (self.Cr ** 2)
    
                    #    %%%%%%%%%%%%%%%%%%%%%% Solution of temperature field %%%%%%%%%%%%%%%%%%%%
    
                    ki = 0
                    kj = 0
                    k = 0  # vectorization pressure index
    
                    for ii in np.arange((Z1 + 0.5 * self.dZ), (Z2), self.dZ):
                        for jj in np.arange(
                            self.theta1 + (self.dtheta / 2), self.theta2, self.dtheta
                        ):
    
                            #                  Pressure gradients
    
                            if kj == 0 and ki == 0:
                                dPdy[ki, kj, n_p] = (P[ki, kj + 1, n_p] - 0) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (P[ki + 1, kj, n_p] - 0) / (2 * self.dZ)
    
                            if kj == 0 and ki > 0 and ki < self.nZ - 1:
                                dPdy[ki, kj, n_p] = (P[ki, kj + 1, n_p] - 0) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (
                                    P[ki + 1, kj, n_p] - P[ki - 1, kj, n_p]
                                ) / (2 * self.dZ)
    
                            if kj == 0 and ki == self.nZ - 1:
                                dPdy[ki, kj, n_p] = (P[ki, kj + 1, n_p] - 0) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (0 - P[ki - 1, kj, n_p]) / (2 * self.dZ)
    
                            if ki == 0 and kj > 0 and kj < self.ntheta - 1:
                                dPdy[ki, kj, n_p] = (
                                    P[ki, kj + 1, n_p] - P[ki, kj - 1, n_p]
                                ) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (P[ki + 1, kj, n_p] - 0) / (2 * self.dZ)
    
                            if (
                                kj > 0
                                and kj < self.ntheta - 1
                                and ki > 0
                                and ki < self.nZ - 1
                            ):
                                dPdy[ki, kj, n_p] = (
                                    P[ki, kj + 1, n_p] - P[ki, kj - 1, n_p]
                                ) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (
                                    P[ki + 1, kj, n_p] - P[ki - 1, kj, n_p]
                                ) / (2 * self.dZ)
    
                            if ki == self.nZ - 1 and kj > 0 and kj < self.ntheta - 1:
                                dPdy[ki, kj, n_p] = (
                                    P[ki, kj + 1, n_p] - P[ki, kj - 1, n_p]
                                ) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (0 - P[ki - 1, kj, n_p]) / (2 * self.dZ)
    
                            if ki == 0 and kj == self.ntheta - 1:
                                dPdy[ki, kj, n_p] = (0 - P[ki, kj - 1, n_p]) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (P[ki + 1, kj, n_p] - 0) / (2 * self.dZ)
    
                            if kj == self.ntheta - 1 and ki > 0 and ki < self.nZ - 1:
                                dPdy[ki, kj, n_p] = (0 - P[ki, kj - 1, n_p]) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (
                                    P[ki + 1, kj, n_p] - P[ki - 1, kj, n_p]
                                ) / (2 * self.dZ)
    
                            if kj == self.ntheta - 1 and ki == self.nZ - 1:
                                dPdy[ki, kj, n_p] = (0 - P[ki, kj - 1, n_p]) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (0 - P[ki - 1, kj, n_p]) / (2 * self.dZ)
    
                            HP = 1 - self.X * np.cos(jj) - self.Y * np.sin(jj)
                            hpt = -self.Ypt * np.cos(jj) + self.Xpt * np.sin(jj)
    
                            mi_p = mi[ki, kj, n_p]
    
                            AE = -(self.kt * HP * self.dZ) / (
                                self.rho
                                * self.Cp
                                * self.war
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
                                    (self.kt * HP * self.dZ)
                                    / (
                                        self.rho
                                        * self.Cp
                                        * self.war
                                        * ((self.betha_s * R) ** 2)
                                        * self.dY
                                    )
                                )
                            )
                            AN = -(self.kt * HP * self.dY) / (
                                self.rho * self.Cp * self.war * (self.L ** 2) * self.dZ
                            )
                            AS = (
                                ((self.R ** 2) * (HP ** 3) * dPdz[ki, kj, n_p] * self.dY)
                                / (12 * (self.L ** 2) * mi_p)
                            ) - (
                                (self.kt * HP * self.dY)
                                / (self.rho * self.Cp * self.war * (self.L ** 2) * self.dZ)
                            )
                            AP = -(AE + AW + AN + AS)
    
                            auxB_t = (self.war * self.mi_ref) / (
                                self.rho * self.Cp * self.Treserv * self.Cr
                            )
                            B_tG = (
                                self.mi_ref
                                * self.war
                                * (R ** 2)
                                * self.dY
                                * self.dZ
                                * P[ki, kj, n_p]
                                * hpt
                            ) / (self.rho * self.Cp * self.Treserv * (self.Cr ** 2))
                            B_tH = (
                                self.war
                                * self.mi_ref
                                * (hpt ** 2)
                                * 4
                                * mi_p
                                * self.dY
                                * self.dZ
                            ) / (self.rho * self.Cp * self.Treserv * 3 * HP)
                            B_tI = (
                                auxB_t
                                * (mi_p * (self.R ** 2) * self.dY * self.dZ)
                                / (HP * self.Cr)
                            )
                            B_tJ = (
                                auxB_t
                                * (
                                    (self.R ** 2)
                                    * (HP ** 3)
                                    * (dPdy[ki, kj, n_p] ** 2)
                                    * self.dY
                                    * self.dZ
                                )
                                / (12 * self.Cr * (self.betha_s ** 2) * mi_p)
                            )
                            B_tK = (
                                auxB_t
                                * (
                                    (self.R ** 4)
                                    * (HP ** 3)
                                    * (dPdz[ki, kj, n_p] ** 2)
                                    * self.dY
                                    * self.dZ
                                )
                                / (12 * self.Cr * (self.L ** 2) * mi_p)
                            )
    
                            B_t = B_tG + B_tH + B_tI + B_tJ + B_tK
    
                            k = k + 1
    
                            b_t[k - 1, 0] = B_t
    
                            if ki == 0 and kj == 0:
                                Mat_coef_t[k - 1, k - 1] = AP + AS - AW
                                Mat_coef_t[k - 1, k] = AE
                                Mat_coef_t[k - 1, k + self.ntheta - 1] = AN
                                b_t[k - 1, 0] = b_t[k - 1, 0] - 2 * AW * (
                                    T_ref / self.Treserv
                                )
    
                            elif kj == 0 and ki > 0 and ki < self.nZ - 1:
                                Mat_coef_t[k - 1, k - 1] = AP - AW
                                Mat_coef_t[k - 1, k] = AE
                                Mat_coef_t[k - 1, k - self.ntheta - 1] = AS
                                Mat_coef_t[k - 1, k + self.ntheta - 1] = AN
                                b_t[k - 1, 0] = b_t[k - 1, 0] - 2 * AW * (
                                    T_ref / self.Treserv
                                )
    
                            elif kj == 0 and ki == self.nZ - 1:
                                Mat_coef_t[k - 1, k - 1] = AP + AN - AW
                                Mat_coef_t[k - 1, k] = AE
                                Mat_coef_t[k - 1, k - self.ntheta - 1] = AS
                                b_t[k - 1, 0] = b_t[k - 1, 0] - 2 * AW * (
                                    T_ref / self.Treserv
                                )
    
                            elif ki == 0 and kj > 0 and kj < self.nY - 1:
                                Mat_coef_t[k - 1, k - 1] = AP + AS
                                Mat_coef_t[k - 1, k] = AE
                                Mat_coef_t[k - 1, k - 2] = AW
                                Mat_coef_t[k - 1, k + self.ntheta - 1] = AN
    
                            elif (
                                ki > 0 and ki < self.nZ - 1 and kj > 0 and kj < self.nY - 1
                            ):
                                Mat_coef_t[k - 1, k - 1] = AP
                                Mat_coef_t[k - 1, k - 2] = AW
                                Mat_coef_t[k - 1, k - self.ntheta - 1] = AS
                                Mat_coef_t[k - 1, k + self.ntheta - 1] = AN
                                Mat_coef_t[k - 1, k] = AE
    
                            elif ki == self.nZ - 1 and kj > 0 and kj < self.nY - 1:
                                Mat_coef_t[k - 1, k - 1] = AP + AN
                                Mat_coef_t[k - 1, k] = AE
                                Mat_coef_t[k - 1, k - 2] = AW
                                Mat_coef_t[k - 1, k - self.ntheta - 1] = AS
    
                            elif ki == 0 and kj == self.nY - 1:
                                Mat_coef_t[k - 1, k - 1] = AP + AE + AS
                                Mat_coef_t[k - 1, k - 2] = AW
                                Mat_coef_t[k - 1, k + self.ntheta - 1] = AN
    
                            elif kj == self.nY - 1 and ki > 0 and ki < self.nZ - 1:
                                Mat_coef_t[k - 1, k - 1] = AP + AE
                                Mat_coef_t[k - 1, k - 2] = AW
                                Mat_coef_t[k - 1, k - self.ntheta - 1] = AS
                                Mat_coef_t[k - 1, k + self.ntheta - 1] = AN
    
                            elif ki == self.nZ - 1 and kj == self.nY - 1:
                                Mat_coef_t[k - 1, k - 1] = AP + AE + AN
                                Mat_coef_t[k - 1, k - 2] = AW
                                Mat_coef_t[k - 1, k - self.ntheta - 1] = AS
    
                            kj = kj + 1
    
                        kj = 0
                        ki = ki + 1
    
                    #    %%%%%%%%%%%%%%%%%%%%%% Solution of temperature field %%%%%%%%%%%%%%%%%%%%
    
                    t = np.linalg.solve(Mat_coef_t, b_t)
    
                    cont = 0
    
                    for i in np.arange(self.nZ):
                        for j in np.arange(self.ntheta):
    
                            T_new[i, j, n_p] = t[cont]
                            cont = cont + 1
    
                    #    % Dimensional Temperature fied [Pa]
    
                    Tdim = T_new * self.Treserv
    
                    T_end = np.sum(Tdim[:, -1, n_p]) / self.nZ
    
                    T_mist_aux[n_p] = (
                        self.fat_mist * self.Treserv + (1 - self.fat_mist) * T_end
                    )
    
    
                    for i in np.arange(self.nZ):
                        for j in np.arange(self.ntheta):
    
                            mi_new[i, j, n_p] = (
                                6.4065 * (Tdim[i, j, n_p]) ** -1.475
                            ) / self.mi_ref

        cont = 0
        for n_p in np.arange(self.n_pad):
            for ii in np.arange(1, self.nZ + 1):
                cont = (
                    1 + (n_p) * (self.ngap / 2) + (n_p) * (self.ntheta + self.ngap / 2)
                )
                for jj in np.arange(1, self.ntheta + 1):

                    PPlot[ii, int(cont)] = Pdim[int(ii - 1), int(jj - 1), int(n_p)]
                    cont = cont + 1

        self.PPlot = PPlot

        auxF=np.zeros((2,len(self.Ytheta[1:-1])))
    
        auxF[0,:]=np.cos(self.Ytheta[1:-1])
        auxF[1,:]=np.sin(self.Ytheta[1:-1])
        
        dA=self.dy*self.dz 
        
        auxP=PPlot[1:-1,1:-1]*dA
        
        vector_auxF_x=auxF[0,:]
        vector_auxF_y=auxF[1,:]
        
        auxFx=auxP*vector_auxF_x
        auxFy=auxP*vector_auxF_y
        
        fxj=-np.sum(auxFx)
        fyj=-np.sum(auxFy)
        
        Fhx=fxj
        Fhy=fyj

        return Fhx, Fhy

    def run(self, x0, plot_pressure=False, print_progress=False):

        args = print_progress
        t1 = time.time()
        res = minimize(
            self._score,
            x0,
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
        if self.equilibrium_pos is None:
            self.run([0.1, -0.1], True, True)
            self.coefficients()
        else:
            xeq = self.equilibrium_pos[0] * self.Cr * np.cos(self.equilibrium_pos[1])
            yeq = self.equilibrium_pos[0] * self.Cr * np.sin(self.equilibrium_pos[1])

            dE = 0.001
            epix = np.abs(dE * self.Cr * np.cos(self.equilibrium_pos[1]))
            epiy = np.abs(dE * self.Cr * np.sin(self.equilibrium_pos[1]))

            Va = self.war * (self.R)
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

            # Coeficientes Adimensionais de Rigidez e Amortecimento dos Mancais
            
            # S=(mi_ref*((R)**3)*L*war)/(np.pi*(Cr**2)*math.sqrt((Wx**2)+(Wy**2)))
            S = 1 / (2 * ((self.L / (2 * self.R)) ** 2) * (np.sqrt((Aux08[0] ** 2) + (Aux08[1] ** 2))))
            Ss = S * ((self.L / (2 * self.R)) ** 2)

            Kxx = -Ss * ((Aux01[0] - Aux02[0]) / (epix / self.Cr))
            Kxy = -Ss * ((Aux03[0] - Aux04[0]) / (epiy / self.Cr))
            Kyx = -Ss * ((Aux01[1] - Aux02[1]) / (epix / self.Cr))
            Kyy = -Ss * ((Aux03[1] - Aux04[1]) / (epiy / self.Cr))

            Cxx = -Ss * ((Aux05[0] - Aux06[0]) / (epixpt / self.Cr / self.war))
            Cxy = -Ss * ((Aux07[0] - Aux08[0]) / (epiypt / self.Cr / self.war))
            Cyx = -Ss * ((Aux05[1] - Aux06[1]) / (epixpt / self.Cr / self.war))
            Cyy = -Ss * ((Aux07[1] - Aux08[1]) / (epiypt / self.Cr / self.war))

            kxx = (math.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / self.Cr) * Kxx
            kxy = (math.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / self.Cr) * Kxy
            kyx = (math.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / self.Cr) * Kyx
            kyy = (math.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / self.Cr) * Kyy

            cxx = (math.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / (self.Cr * self.war)) * Cxx
            cxy = (math.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / (self.Cr * self.war)) * Cxy
            cyx = (math.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / (self.Cr * self.war)) * Cyx
            cyy = (math.sqrt((self.Wx ** 2) + (self.Wy ** 2)) / (self.Cr * self.war)) * Cyy
            
            if show_coef:
                print(f"kxx = {kxx}")
                print(f"kxy = {kxy}")
                print(f"kyx = {kyx}")
                print(f"kyy = {kyy}")

                print(f"cxx = {cxx}")
                print(f"cxy = {cxy}")
                print(f"cyx = {cyx}")
                print(f"cyy = {cyy}")

            coefs = ((kxx,kxy,kyx,kyy),(cxx,cxy,cyx,cyy))

            return coefs

    def _score(self, x, print_progress=False):
        Fhx, Fhy = self._forces(x,None,None,None)
        score = np.sqrt(((self.Wx + Fhx) ** 2) + ((self.Wy + Fhy) ** 2))
        if print_progress:
            print(f"Score: ", score)
            print("============================================")
            print(f"Força na direção x: ", Fhx)
            print("============================================")
            print(f"Força na direção y: ", Fhy)
            print("")

        return score

    def _plotPressure(self):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        Ydim, Zdim = np.meshgrid(self.Ytheta, self.Zdim)
        surf = ax.plot_surface(
            Ydim, Zdim, self.PPlot, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        plt.show()


if __name__ == "__main__":

    x0 = [0.1, -0.1]
    L = float(0.263144)  # [metros]
    R = float(0.2)  # [metros]
    Cr = float(1.945e-4)  # [metros]
    nTheta = int(41)
    nZ = int(10)
    nY = None

    mu = float(0.02)  # [Ns/m²]
    speed = float(900)  # [RPM]
    Wx = float(0)  # [N]
    Wy = float(-112814.91)  # [N]
    k = float(0.15327)  # Thermal conductivity [J/s.m.°C]
    Cp = float(1915.5)  # Specific heat [J/kg°C]
    rho = float(854.952)  # Specific mass [kg/m³]
    Treserv = float(50)  # Temperature of oil tank [ºC]
    mix = float(0.5)  # Mixing factor. Used because the oil supply flow is not known.
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
    mancal.run(x0,print_progress=True,plot_pressure=True)
    # mancal.coefficients()