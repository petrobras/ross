import numpy as np
from numpy.linalg import pinv
from scipy.linalg import solve
from decimal import Decimal


class Tilting:
    def __init__(
        self,
        R,
        Rs,
        esp,
        betha_s,
        rp_pad,
        L,
        sigma,
        fR,
        wa,
        kt,
        Cp,
        rho,
        mi_ref,
        E,
        phi,
        psi_pad,
        npad,
        Cr,
        alpha,
        Tcuba,
        ntheta,
        nX,
        nZ,
        nN,
    ):
        self.R = R
        self.Rs = Rs
        self.esp = esp
        self.rp_pad = rp_pad
        self.L = L
        self.fR = fR
        self.wa = wa
        self.kt = kt
        self.Cp = Cp
        self.rho = rho
        self.mi_ref = mi_ref
        self.E = E
        self.phi = phi
        self.psi_pad = psi_pad
        self.npad = npad
        self.Cr = Cr
        self.Tcuba = Tcuba
        self.ntheta = ntheta
        self.nX = nX
        self.nZ = nZ
        self.nN = nN

        self.war = wa * (np.pi / 30)

        self.betha_s = betha_s * (np.pi / 180)
        self.sigma = sigma * (np.pi / 180)
        self.xx = E * Cr * np.cos(phi)
        self.yy = E * Cr * np.sin(phi)
        self.alphapt = alpha  # * (2 * np.pi * 5) * alpha

        self.xpt = -(2 * np.pi * 5) * self.yy
        self.ypt = (2 * np.pi * 5) * self.xx

        self.Z1 = 0  # initial coordinate z dimensionless
        self.Z2 = 1  # final coordinate z dimensionless
        self.dZ = 1 / (nZ)  # differential z dimensionless
        self.dz = self.dZ * L  # differential z dimensional: [m]
        XZ = np.zeros([nZ + 2])
        XZ[0] = self.Z1
        XZ[nZ + 1] = self.Z2
        XZ[1 : nZ + 1] = self.Z1 + np.arange(
            0.5 * self.dZ, self.Z2, self.dZ
        )  # vector z dimensionless

        self.XZ = XZ

        self.XZdim = XZ * L  # vector z dimensional [m]

        self.N1 = 0  # initial coordinate netha dimensionless
        self.N2 = 1  # final coordinate netha dimensionless
        self.dN = 1 / (nN)  # differential netha dimensionless

        netha = np.zeros([nN + 2])
        netha[0] = self.N1
        netha[nN + 1] = self.N2
        netha[1 : nN + 1] = self.N1 + np.arange(
            0.5 * self.dN, self.N2, self.dN
        )  # vector netha dimensionless

        self.netha = netha

        self.theta1 = -(rp_pad) * self.betha_s  # initial coordinate theta [rad]
        self.theta2 = (1 - rp_pad) * self.betha_s  # final coordinate theta [rad]

        self.dtheta = self.betha_s / (ntheta)  # differential theta [rad]
        Xtheta = np.zeros([ntheta + 2])
        Xtheta[0] = self.theta1
        Xtheta[ntheta + 1] = self.theta2
        Xtheta[1 : ntheta + 1] = np.arange(
            self.theta1 + 0.5 * self.dtheta, self.theta2, self.dtheta
        )  # vector theta [rad]
        self.Xtheta = Xtheta

        self.dX = 1 / nX  # differential x dimensionless
        self.dx = self.dX * (self.betha_s * Rs)  # differential x dimensional: [m]
        self.XX = Xtheta * Rs  # vector x dimensional: [m]

        # Pad recess
        self.len_betha = 0.39 * self.betha_s  # Pad Angle with recess
        self.len_L = 0.71 * L  # Bearing length with recess
        self.center_pos_L = L / 2
        self.start_pos_betha = 0 * self.betha_s
        self.drop_pressure_pos_L = np.array(
            [self.center_pos_L - self.len_L / 2, self.center_pos_L + self.len_L / 2]
        )
        self.drop_pressure_pos_betha = np.array(
            [self.start_pos_betha, self.start_pos_betha + self.len_betha]
        )

        self.drop_pressure_Ele_nZ = np.intersect1d(
            np.where(self.XZdim > self.drop_pressure_pos_L[0]),
            np.where(self.XZdim < self.drop_pressure_pos_L[1]),
        )
        self.drop_pressure_Ele_ntetha = np.intersect1d(
            np.where(Xtheta >= self.drop_pressure_pos_betha[0] + self.theta1),
            np.where(Xtheta <= self.drop_pressure_pos_betha[1] + self.theta1),
        )

    def _forces(self):

        T_ref = self.Tcuba

        # Startup
        npad = self.npad - 1
        Tmist = T_ref * np.ones((self.nN + 2))

        # Initial viscosity field
        minovo = self.mi_ref * np.ones((self.nZ, self.ntheta, self.nN))

        # Velocity field  - 3D
        vu = np.zeros((self.nZ, self.ntheta, self.nN))
        vv = np.zeros((self.nZ, self.ntheta, self.nN))
        vw = np.zeros((self.nZ, self.ntheta, self.nN))

        # Velocity field - 2D
        Vu = np.zeros((self.nN, self.ntheta))
        Vv = np.zeros((self.nN, self.ntheta))
        Vw = np.zeros((self.nN, self.ntheta))

        # Pressure field
        P = np.zeros((self.ntheta, self.ntheta))
        P1 = np.zeros((self.ntheta, self.ntheta, npad))

        # Temperature field
        T = np.zeros((self.nN, self.ntheta))
        T1 = np.zeros((self.nN + 2, self.ntheta + 2, npad))

        # Field derivatives
        dudx = np.zeros((self.nN + 2))
        dwdz = np.zeros((self.nN + 2))

        # Other variables declarations
        Mi = np.zeros((self.nN, self.nZ))
        YH = np.zeros((self.nN + 2, self.nX + 2, npad))
        XH = np.zeros((self.nN + 2, self.nX + 2))

        fxj = np.zeros((npad))
        My = np.zeros((npad))

        for ii in range(0, self.nX + 2):
            XH[:, ii] = self.Xtheta[ii]

        # Loop on the pads ==========================================================================
        for n_p in range(0, npad):
            alpha = self.psi_pad[n_p]

            # transformation of coordinates - inertial to pivot referential
            xryr = np.dot(
                [
                    [np.cos(self.sigma[n_p]), np.sin(self.sigma[n_p])],
                    [-np.sin(self.sigma[n_p]), np.cos(self.sigma[n_p])],
                ],
                [[self.xx], [self.yy]],
            )

            xryrpt = np.dot(
                [
                    [np.cos(self.sigma[n_p]), np.sin(self.sigma[n_p])],
                    [-np.sin(self.sigma[n_p]), np.cos(self.sigma[n_p])],
                ],
                [[self.xpt], [self.ypt]],
            )

            xr = xryr[0, 0]
            yr = xryr[1, 0]

            xrpt = xryrpt[0, 0]
            yrpt = xryrpt[1, 0]

            # Temperature matrix with boundary conditions ====================================
            T_novo = T_ref * np.ones((self.nN + 2, self.ntheta + 2))
            Tcomp = 1.2 * T_novo

            # Oil temperature field loop ================================================================
            while np.linalg.norm((T_novo - Tcomp) / np.linalg.norm(Tcomp)) > 0.01:

                nk = self.nZ * self.ntheta
                vector_mi = np.zeros((3, self.nN))
                auxFF0P = np.zeros((self.nN + 2))
                auxFF1P = np.zeros((self.nN + 2))
                auxFF0E = np.zeros((self.nN + 2))
                auxFF1E = np.zeros((self.nN + 2))
                auxFF0W = np.zeros((self.nN + 2))
                auxFF1W = np.zeros((self.nN + 2))
                auxFF2P = np.zeros((self.nN + 2))
                auxFF2E = np.zeros((self.nN + 2))
                auxFF2W = np.zeros((self.nN + 2))
                xh = np.zeros((self.nX))
                yh = np.zeros((self.nX))
                hhh = np.zeros((nk, npad))  # oil film thickness
                K_null = np.zeros((1, nk))
                Kij_null = np.zeros((self.nZ, self.ntheta))

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
                for ii in range(0, self.nZ):

                    # Mesh loop in THETA direction ====================================================
                    # for jj in range((theta1 + 0.5 * dtheta), dtheta, (theta2 - 0.5 * dtheta)):
                    for jj in range(0, self.ntheta):

                        if kj == 0:
                            vector_mi[0, :] = mi[ki, kj, :]
                            vector_mi[1, :] = mi[ki, kj + 1, :]
                            vector_mi[2, :] = mi[ki, kj, :]

                        if kj == self.ntheta - 1:
                            vector_mi[0, :] = mi[ki, kj, :]
                            vector_mi[1, :] = mi[ki, kj, :]
                            vector_mi[2, :] = mi[ki, kj - 1, :]

                        if kj > 0 and kj < self.ntheta - 1:
                            vector_mi[0, :] = mi[ki, kj, :]
                            vector_mi[1, :] = mi[ki, kj + 1, :]
                            vector_mi[2, :] = mi[ki, kj - 1, :]

                        for kk in range(1, self.nN + 1):

                            mi_adP = vector_mi[0, self.nN - nn - 1] / self.mi_ref
                            mi_adE = vector_mi[1, self.nN - nn - 1] / self.mi_ref
                            mi_adW = vector_mi[2, self.nN - nn - 1] / self.mi_ref

                            auxFF0P[nn + 1] = 1 / mi_adP
                            auxFF1P[nn + 1] = (self.dN * (-0.5 + kk)) / mi_adP
                            auxFF0E[nn + 1] = 1 / mi_adE
                            auxFF1E[nn + 1] = (self.dN * (-0.5 + kk)) / mi_adE
                            auxFF0W[nn + 1] = 1 / mi_adW
                            auxFF1W[nn + 1] = (self.dN * (-0.5 + kk)) / mi_adW

                            nn = nn + 1

                        nn = 0

                        auxFF0P[0] = auxFF0P[1]
                        auxFF0P[self.nN + 1] = auxFF0P[self.nN]

                        auxFF1P[0] = 0
                        auxFF1P[self.nN + 1] = self.N2 / (
                            vector_mi[0, self.nN - 1] / self.mi_ref
                        )

                        auxFF0E[0] = auxFF0E[1]
                        auxFF0E[self.nN + 1] = auxFF0E[self.nN]

                        auxFF1E[0] = 0
                        auxFF1E[self.nN + 1] = self.N2 / (
                            vector_mi[1, self.nN - 1] / self.mi_ref
                        )

                        auxFF0W[0] = auxFF0W[1]
                        auxFF0W[self.nN + 1] = auxFF0W[self.nN]

                        auxFF1W[0] = 0
                        auxFF1W[self.nN + 1] = self.N2 / (
                            vector_mi[2, self.nN - 1] / self.mi_ref
                        )

                        # Numerical integration
                        FF0P = 0.5 * np.sum(
                            (self.netha[1:] - self.netha[0:-1])
                            * (auxFF0P[1:] + auxFF0P[0:-1])
                        )
                        FF1P = 0.5 * np.sum(
                            (self.netha[1:] - self.netha[0:-1])
                            * (auxFF1P[1:] + auxFF1P[0:-1])
                        )
                        FF0E = 0.5 * np.sum(
                            (self.netha[1:] - self.netha[0:-1])
                            * (auxFF0E[1:] + auxFF0E[0:-1])
                        )
                        FF1E = 0.5 * np.sum(
                            (self.netha[1:] - self.netha[0:-1])
                            * (auxFF1E[1:] + auxFF1E[0:-1])
                        )
                        FF0W = 0.5 * np.sum(
                            (self.netha[1:] - self.netha[0:-1])
                            * (auxFF0W[1:] + auxFF0W[0:-1])
                        )
                        FF1W = 0.5 * np.sum(
                            (self.netha[1:] - self.netha[0:-1])
                            * (auxFF1W[1:] + auxFF1W[0:-1])
                        )

                        FF0e = 0.5 * (FF0P + FF0E)
                        FF0w = 0.5 * (FF0P + FF0W)
                        FF1e = 0.5 * (FF1P + FF1E)
                        FF1w = 0.5 * (FF1P + FF1W)

                        # Loop in N
                        # for kk in range(N1 + 0.5 * self.dN, self.dN, self.N2 - 0.5 * self.dN):
                        for kk in range(0, self.nN + 1):

                            mi_adP = vector_mi[0, self.nN - nn - 1] / self.mi_ref
                            mi_adE = vector_mi[1, self.nN - nn - 1] / self.mi_ref
                            mi_adW = vector_mi[2, self.nN - nn - 1] / self.mi_ref

                            auxFF2P[nn] = ((self.dN * (-0.5 + kk)) / mi_adP) * (
                                (self.dN * (-0.5 + kk)) - FF1P / FF0P
                            )
                            auxFF2E[nn] = ((self.dN * (-0.5 + kk)) / mi_adE) * (
                                (self.dN * (-0.5 + kk)) - FF1E / FF0E
                            )
                            auxFF2W[nn] = ((self.dN * (-0.5 + kk)) / mi_adW) * (
                                (self.dN * (-0.5 + kk)) - FF1W / FF0W
                            )
                            nn = nn + 1

                        nn = 0

                        auxFF2P[0] = 0
                        auxFF2P[self.nN + 1] = (
                            self.N2 / (vector_mi[0, self.nN - 1] / self.mi_ref)
                        ) * (self.N2 - FF1P / FF0P)

                        auxFF2E[0] = 0
                        auxFF2E[self.nN + 1] = (
                            self.N2 / (vector_mi[1, self.nN - 1] / self.mi_ref)
                        ) * (self.N2 - FF1P / FF0P)

                        auxFF2W[0] = 0
                        auxFF2W[self.nN + 1] = (
                            self.N2 / (vector_mi[2, self.nN - 1] / self.mi_ref)
                        ) * (self.N2 - FF1P / FF0P)

                        # integration process ===================================================
                        FF2P = 0.5 * np.sum(
                            (self.netha[1:] - self.netha[0:-1])
                            * (auxFF2P[1:] + auxFF2P[0:-1])
                        )
                        FF2E = 0.5 * np.sum(
                            (self.netha[1:] - self.netha[0:-1])
                            * (auxFF2E[1:] + auxFF2E[0:-1])
                        )
                        FF2W = 0.5 * np.sum(
                            (self.netha[1:] - self.netha[0:-1])
                            * (auxFF2W[1:] + auxFF2W[0:-1])
                        )

                        FF2e = 0.5 * (FF2P + FF2E)
                        FF2w = 0.5 * (FF2P + FF2W)
                        FF2n = FF2P
                        FF2s = FF2n

                        # Admensional oil film thickness ========================================
                        hP = (
                            self.Rs
                            - self.R
                            - (
                                np.sin(self.Xtheta[jj + 1])
                                * (yr + alpha * (self.Rs + self.esp))
                                + np.cos(self.Xtheta[jj + 1])
                                * (xr + self.Rs - self.R - self.Cr)
                            )
                        ) / self.Cr
                        he = (
                            self.Rs
                            - self.R
                            - (
                                np.sin(self.Xtheta[jj + 1] + 0.5 * self.dtheta)
                                * (yr + alpha * (self.Rs + self.esp))
                                + np.cos(self.Xtheta[jj + 1] + 0.5 * self.dtheta)
                                * (xr + self.Rs - self.R - self.Cr)
                            )
                        ) / self.Cr
                        hw = (
                            self.Rs
                            - self.R
                            - (
                                np.sin(self.Xtheta[jj + 1] - 0.5 * self.dtheta)
                                * (yr + alpha * (self.Rs + self.esp))
                                + np.cos(self.Xtheta[jj + 1] - 0.5 * self.dtheta)
                                * (xr + self.Rs - self.R - self.Cr)
                            )
                        ) / self.Cr
                        hn = hP
                        hs = hn
                        hpt = -(1 / (self.Cr * self.war)) * (
                            np.cos(self.Xtheta[jj + 1]) * xrpt
                            + np.sin(self.Xtheta[jj + 1]) * yrpt
                            + np.sin(self.Xtheta[jj + 1])
                            * (self.Rs + self.esp)
                            * self.alphapt
                        )  # admensional

                        # Finite volume frontiers
                        CE = (
                            1
                            / (self.betha_s) ** 2
                            * (FF2e * he ** 3)
                            * self.dZ
                            / self.dX
                        )
                        CW = (
                            1
                            / (self.betha_s) ** 2
                            * (FF2w * hw ** 3)
                            * self.dZ
                            / self.dX
                        )
                        CN = (
                            (FF2n * hn ** 3)
                            * (self.dX / self.dZ)
                            * (self.Rs / self.L) ** 2
                        )
                        CS = (
                            (FF2s * hs ** 3)
                            * (self.dX / self.dZ)
                            * (self.Rs / self.L) ** 2
                        )
                        CP = -(CE + CW + CN + CS)

                        B = (self.R / (self.Rs * self.betha_s)) * self.dZ * (
                            he * (1 - FF1e / FF0e) - hw * (1 - FF1w / FF0w)
                        ) + hpt * self.dX * self.dZ

                        b[k] = B
                        hhh[k, n_p] = hP * self.Cr

                        # Mat_coef determination depending on its mesh localization
                        if ki == 0 and kj == 0:
                            Mat_coef[k, k] = CP - CN - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + self.ntheta] = CS

                        if ki == 0 and kj > 0 and kj < self.nX - 1:
                            Mat_coef[k, k] = CP - CN
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + self.ntheta] = CS

                        if ki == 0 and kj == self.nX - 1:
                            Mat_coef[k, k] = CP - CE - CN
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + self.ntheta] = CS

                        if kj == 0 and ki > 0 and ki < self.nZ - 1:
                            Mat_coef[k, k] = CP - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - self.ntheta] = CN
                            Mat_coef[k, k + self.ntheta] = CS

                        if ki > 0 and ki < self.nZ - 1 and kj > 0 and kj < self.nX - 1:
                            Mat_coef[k, k] = CP
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.ntheta] = CN
                            Mat_coef[k, k + self.ntheta] = CS
                            Mat_coef[k, k + 1] = CE

                        if kj == self.nX - 1 and ki > 0 and ki < self.nZ - 1:
                            Mat_coef[k, k] = CP - CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.ntheta] = CN
                            Mat_coef[k, k + self.ntheta] = CS

                        if kj == 0 and ki == self.nZ - 1:
                            Mat_coef[k, k] = CP - CS - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - self.ntheta] = CN

                        if ki == self.nZ - 1 and kj > 0 and kj < self.nX - 1:
                            Mat_coef[k, k] = CP - CS
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.ntheta] = CN

                        if ki == self.nZ - 1 and kj == self.nX - 1:
                            Mat_coef[k, k] = CP - CE - CS
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.ntheta] = CN

                        kj = kj + 1
                        k = k + 1
                    # loop end

                    kj = 0
                    ki = ki + 1
                # loop end

                # Pressure field solution ==============================================================
                p = np.linalg.solve(Mat_coef, b)

                cont = 0

                for i in np.arange(self.nZ):
                    for j in np.arange(self.ntheta):

                        P[i, j] = p[cont]
                        cont = cont + 1

                        if P[i, j] < 0:
                            P[i, j] = 0

                # Pressure border conditions ====================================================
                for i in range(0, self.nZ - 1):  # Loop in Z
                    for j in range(0, self.ntheta - 1):  # Loop in THETA
                        if P[i, j] < 0:
                            P[i, j] = 0

                # Dimmensional pressure determination in Pascals
                Pdim = (P * self.mi_ref * self.war * (self.Rs ** 2)) / (self.Cr ** 2)

                # Full pressure field with borders
                PPdim = np.zeros((self.nZ + 1, self.ntheta + 1))

                for i in range(0, self.nZ - 1):  # Loop in Z
                    for j in range(0, self.ntheta - 1):  # Loop in THETA
                        PPdim[i, j] = Pdim[i - 1, j - 1]

                # %%%%%%%%%%%%%%%%%%% Temperature field solution %%%%%%%%%%%%%%%%%%%

                # Velocity field calculation
                ki = 0
                kj = 0
                kk = 0
                nn = 0

                # Dimensionless Netha loop ====================================================
                for ky in np.arange(
                    (self.N1 + 0.5 * self.dN),
                    (self.N2 - 0.5 * self.dN) + self.dN,
                    self.dN,
                ):
                    # for ky in range(0, self.nN - 1):
                    # Mesh loop in Z direction ====================================================
                    for ii in np.arange(
                        (self.Z1 + 0.5 * self.dZ),
                        (self.Z2 - 0.5 * self.dZ) + self.dZ,
                        self.dZ,
                    ):
                        # for ii in range(0, self.nZ):
                        # Mesh loop in THETA direction ====================================================
                        for jj in np.arange(
                            (self.theta1 + 0.5 * self.dtheta),
                            (self.theta2 - 0 * self.dtheta),
                            self.dtheta,
                        ):
                            # for jj in range(0, self.ntheta):

                            # Pressure gradients calculation
                            if ki == 0 and kj == 0:
                                dPdx = Pdim[ki, kj] / (0.5 * self.dx)
                                dPdz = Pdim[ki, kj] / (0.5 * self.dz)

                            if ki == 0 and kj > 0:
                                dPdx = (Pdim[ki, kj] - Pdim[ki, kj - 1]) / self.dx
                                dPdz = Pdim[ki, kj] / (0.5 * self.dz)

                            if ki > 0 and kj == 0:
                                dPdx = Pdim[ki, kj] / (0.5 * self.dx)
                                dPdz = (Pdim[ki, kj] - Pdim[ki - 1, kj]) / self.dz

                            if ki > 0 and kj > 0:
                                dPdx = (Pdim[ki, kj] - Pdim[ki, kj - 1]) / self.dx
                                dPdz = (Pdim[ki, kj] - Pdim[ki - 1, kj]) / self.dz

                            # Dimensional oil film thickness in Meters
                            h = (
                                self.Rs
                                - self.R
                                - (
                                    np.sin(jj) * (yr + alpha * (self.Rs + self.esp))
                                    + np.cos(jj) * (xr + self.Rs - self.R - self.Cr)
                                )
                            )

                            auxFF0 = np.zeros((self.netha.size))
                            auxFF1 = np.zeros((self.netha.size))

                            # for contk in range(((N1 + 0.5 * self.dN) * h), (self.dN * h), ((self.N2 - (0.5 * self.dN)) * h)):
                            for contk in range(1, self.nN + 1):
                                nn = nn + 1
                                auxFF0[nn] = 1 / mi[ki, kj, self.nN - 1 - nn]
                                auxFF1[nn] = (self.dN * (-0.5 + contk) * h) / mi[
                                    ki, kj, self.nN - 1 - nn
                                ]

                            nn = 0

                            auxFF0[0] = auxFF0[1]
                            auxFF0[self.nN + 1] = auxFF0[self.nN]

                            auxFF1[0] = 0
                            auxFF1[self.nN + 1] = (self.N2 * h) / mi[ki, kj, 1]

                            ydim1 = h * self.netha
                            FF0 = 0.5 * np.sum(
                                (ydim1[1:] - ydim1[0:-1]) * (auxFF0[1:] + auxFF0[0:-1])
                            )
                            FF1 = 0.5 * np.sum(
                                (ydim1[1:] - ydim1[0:-1]) * (auxFF1[1:] + auxFF1[0:-1])
                            )

                            # Auxilary variables declaration/reset
                            aux_size = np.arange(self.N1, ky + self.dN, self.dN)
                            auxG0 = np.zeros(self.netha.size)
                            auxG1 = np.zeros(self.netha.size)
                            ydim2 = np.zeros(self.netha.size)

                            for contk in np.arange(
                                ((self.N1 + 0.5 * self.dN) * h),
                                ((ky + self.dN) * h),
                                (self.dN * h),
                            ):
                                nn = nn + 1

                                auxG0[nn] = 1 / mi[ki, kj, self.nN - 1 - nn]
                                auxG1[nn] = contk / mi[ki, kj, self.nN - 1 - nn]
                                ydim2[nn] = contk

                            # Counter reset
                            nn = 0

                            # auxG0 = auxG0[::-1]
                            auxG0 = auxG0[: int(len(aux_size))]
                            auxG0[0] = auxG0[1]

                            # auxG1 = auxG1[::-1]
                            auxG1 = auxG1[: int(len(aux_size))]
                            auxG1[0] = 0

                            # ydim2 = ydim2[::-1]
                            ydim2 = ydim2[: int(len(aux_size))]
                            ydim2[0] = self.N1 * h

                            G0 = 0.5 * np.sum(
                                (ydim2[1:] - ydim2[0:-1]) * (auxG0[1:] + auxG0[0:-1])
                            )
                            G1 = 0.5 * np.sum(
                                (ydim2[1:] - ydim2[0:-1]) * (auxG1[1:] + auxG1[0:-1])
                            )

                            # vu(ki,kj,kk)=dPdx*G1+(self.war*R/FF0-FF1/FF0*dPdx)*G0;
                            vu[ki, kj, kk] = (
                                dPdx * G1
                                + (self.war * self.R / FF0 - FF1 / FF0 * dPdx) * G0
                            )

                            # vw(ki,kj,kk)=dPdz*G1-(FF1/FF0*dPdz)*G0;
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
                # for ii in range((Z1 + 0.5 * self.dZ), self.dZ, (Z2 - 0.5 * self.dZ)):
                for ii in range(0, self.nZ):
                    # Mesh loop in THETA direction ====================================================
                    # for jj in range((theta1 + 0.5 * self.dtheta), self.dtheta, (theta2 - 0.5 * self.dtheta)):
                    for jj in range(0, self.ntheta):

                        hpt = -(
                            np.cos(self.Xtheta[jj + 1]) * xrpt
                            + np.sin(self.Xtheta[jj + 1]) * yrpt
                            + np.sin(self.Xtheta[jj + 1])
                            * (self.Rs + self.esp)
                            * self.alphapt
                        )

                        if ki == 0 and kj == 0:
                            # for contk in range(N1 + 0.5 * self.dN, self.dN, self.N2 - 0.5 * self.dN):
                            for contk in range(0, self.nN + 1):
                                dudx[nn] = 0
                                dwdz[nn] = 0
                                nn = nn + 1
                            nn = 0

                        if ki == 0 and kj > 0:
                            # for contk in range(N1 + 0.5 * self.dN, self.dN, self.N2 - 0.5 * self.dN):
                            for contk in range(0, self.nN + 1):
                                dudx[nn] = (
                                    vu[ki, kj, nn - 1] - vu[ki, kj - 1, nn - 1]
                                ) / self.dx
                                dwdz[nn] = 0
                                nn = nn + 1
                            nn = 0

                        if ki > 0 and kj == 0:
                            # for contk in range(N1 + 0.5 * self.dN, self.dN, self.N2 - 0.5 * self.dN):
                            for contk in range(0, self.nN + 1):
                                dudx[nn] = 0
                                dwdz[nn] = (
                                    vw[ki, kj, nn - 1] - vw[ki - 1, kj, nn - 1]
                                ) / self.dz
                                nn = nn + 1
                            nn = 0

                        if ki > 0 and ki < self.nN - 1 and kj > 0 and kj < self.nX - 1:
                            # for contk in range(N1 + 0.5 * self.dN, self.dN, self.N2 - 0.5 * self.dN):
                            for contk in range(0, self.nN + 1):
                                dudx[nn] = (
                                    vu[ki, kj, nn - 1] - vu[ki, kj - 1, nn - 1]
                                ) / self.dx
                                dwdz[nn] = (
                                    vw[ki, kj, nn - 1] - vw[ki - 1, kj, nn - 1]
                                ) / self.dz
                                nn = nn + 1
                            nn = 0

                        dudx[0] = dudx[1]
                        dwdz[0] = dwdz[1]
                        dudx[self.nN + 1] = dudx[self.nN]
                        dwdz[self.nN + 1] = dwdz[self.nN]

                        auxD = dudx + dwdz

                        # intv=0.5*sum((ydim1(2:end)-ydim1(1:end-1)).*(auxD(2:end)+auxD(1:end-1)));
                        intv = 0.5 * np.sum(
                            (ydim1[1:] - ydim1[0:-1]) * (auxD[1:] + auxD[0:-1])
                        )

                        vv[ki, kj,] = (
                            -intv + hpt
                        )
                        kj = kj + 1

                    kj = 0
                    ki = ki + 1

                ki = 0
                ki = self.nN - 1
                for ii in range(0, self.nN):
                    for jj in range(0, self.ntheta):
                        Vu[ii, jj] = np.mean(vu[:, jj, ki])
                        Vv[ii, jj] = np.mean(vv[:, jj, ki])
                        Vw[ii, jj] = np.mean(vw[:, jj, ki])

                    ki = ki - 1

                # Radial velocity calculation ending ------------------------------
                ksi1 = 0
                ksi2 = 1

                ki = 0
                kj = 0
                dksi = self.dX

                for ii in range(0, self.nZ):
                    for jj in range(0, self.nN):
                        Mi[jj, ii] = mi[0, ii, jj]

                nk = self.nN * self.ntheta
                Mat_coef = np.zeros((nk, nk))
                b = np.zeros((nk))
                k = 0

                # for ii in range(N1 + 0.5 * self.dN, self.dN, self.N2 - 0.5 * self.dN):
                for ii in range(0, self.nN):

                    # for jj in range(ksi1 + 0.5 * dksi, dksi, ksi2 - 0.5 * dksi):
                    # jj is equivalent to: dksi * (-0.5 + jj)
                    for jj in range(0, self.nX):

                        #
                        theta = (-0.5 + (dksi * (+0.5 + jj))) * self.betha_s
                        HP = (
                            self.Rs
                            - self.R
                            - (
                                np.sin(theta) * (yr + alpha * (self.Rs + self.esp))
                                + np.cos(theta) * (xr + self.Rs - self.R - self.Cr)
                            )
                        )
                        He = (
                            self.Rs
                            - self.R
                            - (
                                np.sin(theta + 0.5 * self.dtheta)
                                * (yr + alpha * (self.Rs + self.esp))
                                + np.cos(theta + 0.5 * self.dtheta)
                                * (xr + self.Rs - self.R - self.Cr)
                            )
                        )
                        Hw = (
                            self.Rs
                            - self.R
                            - (
                                np.sin(theta - 0.5 * self.dtheta)
                                * (yr + alpha * (self.Rs + self.esp))
                                + np.cos(theta - 0.5 * self.dtheta)
                                * (xr + self.Rs - self.R - self.Cr)
                            )
                        )
                        Hn = HP
                        Hs = Hn
                        Hnw = Hw
                        Hsw = Hnw

                        yh[kj] = HP
                        xh[kj] = theta

                        JP = 1 / (self.betha_s * self.Rs * HP)
                        Je = 1 / (self.betha_s * self.Rs * He)
                        Jw = 1 / (self.betha_s * self.Rs * Hw)
                        Jn = 1 / (self.betha_s * self.Rs * Hn)
                        Js = 1 / (self.betha_s * self.Rs * Hs)

                        if ki == 0 and kj == 0:

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

                        if ki == 0 and kj > 0 and kj < self.nX - 1:
                            uP = Vu[ki, kj]
                            uE = Vu[ki, kj + 1]
                            uW = Vu[ki, kj - 1]
                            uN = Vu[ki, kj]
                            uS = Vu[ki + 1, kj]

                        if ki > 0 and ki < self.nN - 1 and kj == 0:
                            uP = Vu[ki, kj]
                            uE = Vu[ki, kj + 1]
                            uW = Vu[ki, kj]
                            uN = Vu[ki - 1, kj]
                            uS = Vu[ki + 1, kj]

                        if ki == self.nN - 1 and kj == 0:
                            uP = Vu[ki, kj]
                            uE = Vu[ki, kj + 1]
                            uW = Vu[ki, kj]
                            uN = Vu[ki - 1, kj]
                            uS = Vu[ki, kj]

                        if ki == 0 and kj == self.nX - 1:
                            uP = Vu[ki, kj]
                            uE = Vu[ki, kj]
                            uW = Vu[ki, kj - 1]
                            uN = Vu[ki, kj]
                            uS = Vu[ki + 1, kj]

                        if ki > 0 and ki < self.nN - 1 and kj == self.nX - 1:
                            uP = Vu[ki, kj]
                            uE = Vu[ki, kj]
                            uW = Vu[ki, kj - 1]
                            uN = Vu[ki - 1, kj]
                            uS = Vu[ki + 1, kj]

                        if ki == self.nN - 1 and kj == self.nX - 1:
                            uP = Vu[ki, kj]
                            uE = Vu[ki, kj]
                            uW = Vu[ki, kj - 1]
                            uN = Vu[ki - 1, kj]
                            uS = Vu[ki, kj]

                        if ki == self.nN - 1 and kj > 0 and kj < self.nX - 1:
                            uP = Vu[ki, kj]
                            uE = Vu[ki, kj + 1]
                            uW = Vu[ki, kj - 1]
                            uN = Vu[ki - 1, kj]
                            uS = Vu[ki, kj]

                        if ki > 0 and ki < self.nN - 1 and kj > 0 and kj < self.nX - 1:
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

                        ue = 0.5 * (uP + uE)
                        uw = 0.5 * (uP + uW)
                        un = 0.5 * (uP + uN)
                        us = 0.5 * (uP + uS)

                        ve = 0.5 * (vP + vE)
                        vw = 0.5 * (vP + vW)
                        vn = 0.5 * (vP + vN)
                        vs = 0.5 * (vP + vS)

                        we = 0.5 * (wP + wE)
                        ww = 0.5 * (wP + wW)
                        wn = 0.5 * (wP + wN)
                        ws = 0.5 * (wP + wS)

                        UP = hP * uP
                        Ue = He * ue
                        Uw = Hw * uw

                        np2 = 1 - ((ii + 0.5) * self.dN)
                        ne = np2
                        nw = np2
                        nn = np2 + self.dN
                        ns = np2 - self.dN

                        dhdksi_p = -self.betha_s * (
                            np.cos(theta) * (yr + alpha * (self.Rs + self.esp))
                            - np.sin(theta) * (xr + self.Rs - self.R - self.Cr)
                        )
                        dhdksi_e = -self.betha_s * (
                            np.cos(theta + 0.5 * self.dtheta)
                            * (yr + alpha * (self.Rs + self.esp))
                            - np.sin(theta + 0.5 * self.dtheta)
                            * (xr + self.Rs - self.R - self.Cr)
                        )
                        dhdksi_w = -self.betha_s * (
                            np.cos(theta - 0.5 * self.dtheta)
                            * (yr + alpha * (self.Rs + self.esp))
                            - np.sin(theta - 0.5 * self.dtheta)
                            * (xr + self.Rs - self.R - self.Cr)
                        )
                        dhdksi_n = dhdksi_p
                        dhdksi_s = dhdksi_n

                        VP = self.betha_s * self.Rs * vP - np2 * dhdksi_p * uP
                        Vn = self.betha_s * self.Rs * vn - nn * dhdksi_n * un
                        Vs = self.betha_s * self.Rs * vs - ns * dhdksi_s * us

                        alpha11P = HP ** 2
                        alpha11e = He ** 2
                        alpha11w = Hw ** 2

                        alpha12P = -np2 * HP * dhdksi_p
                        alpha12e = -ne * He * dhdksi_e
                        alpha12w = -nw * Hw * dhdksi_w

                        alpha21P = alpha12P
                        alpha21n = -nn * Hn * dhdksi_n
                        alpha21s = -ns * Hs * dhdksi_s

                        alpha22P = (self.betha_s * self.Rs) ** 2 + (np2 * dhdksi_p) ** 2
                        alpha22n = (self.betha_s * self.Rs) ** 2 + (nn * dhdksi_n) ** 2
                        alpha22s = (self.betha_s * self.Rs) ** 2 + (ns * dhdksi_s) ** 2

                        Me = self.rho * Ue * self.dN
                        Mw = self.rho * Uw * self.dN
                        Mn = self.rho * Vn * dksi
                        Ms = self.rho * Vs * dksi

                        D11 = self.kt / self.Cp * JP * alpha11P * self.dN
                        D11e = self.kt / self.Cp * Je * alpha11e * self.dN
                        D11w = self.kt / self.Cp * Jw * alpha11w * self.dN

                        D12 = self.kt / self.Cp * JP * alpha12P * self.dN
                        D12e = self.kt / self.Cp * Je * alpha12e * self.dN
                        D12w = self.kt / self.Cp * Jw * alpha12w * self.dN

                        D21 = self.kt / self.Cp * JP * alpha21P * dksi
                        D21n = self.kt / self.Cp * Jn * alpha21n * dksi
                        D21s = self.kt / self.Cp * Js * alpha21s * dksi

                        D22 = self.kt / self.Cp * JP * alpha22P * dksi
                        D22n = self.kt / self.Cp * Jn * alpha22n * dksi
                        D22s = self.kt / self.Cp * Js * alpha22s * dksi

                        # Interpolation coefficients
                        Pee = (
                            self.rho * uE * self.Cp * self.dtheta * self.Rs / self.kt
                        )  # Peclet's number
                        Pew = self.rho * uW * self.Cp * self.dtheta * self.Rs / self.kt

                        Pen = self.rho * uN * self.Cp * self.dtheta * self.Rs / self.kt
                        Pes = self.rho * uS * self.Cp * self.dtheta * self.Rs / self.kt

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

                        Ae = (
                            Me * (0.5 - a_pe)
                            - D11e / dksi * b_pe
                            - (D21n - D21s) / (4 * dksi)
                        )
                        Aw = (
                            -Mw * (0.5 + a_pw)
                            - D11w / dksi * b_pw
                            + (D21n - D21s) / (4 * dksi)
                        )
                        An = (
                            Mn * (0.5 - a_pn)
                            - D22n / self.dN * b_pn
                            - (D12e - D12w) / (4 * self.dN)
                        )
                        As = (
                            -Ms * (0.5 + a_ps)
                            - D22s / self.dN * b_ps
                            - (D12w - D12e) / (4 * self.dN)
                        )
                        Ane = -D12e / (4 * self.dN) - D21n / (4 * dksi)
                        Ase = D12e / (4 * self.dN) + D21s / (4 * dksi)
                        Anw = D12w / (4 * self.dN) + D21n / (4 * dksi)
                        Asw = -D12w / (4 * self.dN) - D21s / (4 * dksi)
                        Ap = -(Ae + Aw + An + As + Ane + Ase + Anw + Asw)

                        up_a = uP / (self.R * self.war)
                        uw_a = uW / (self.R * self.war)
                        ue_a = uE / (self.R * self.war)
                        us_a = uS / (self.R * self.war)
                        un_a = uN / (self.R * self.war)

                        vp_a = vP / (self.R * self.war)
                        vw_a = vW / (self.R * self.war)
                        ve_a = vE / (self.R * self.war)
                        vs_a = vS / (self.R * self.war)
                        vn_a = vN / (self.R * self.war)

                        wp_a = wP / (self.R * self.war)
                        ww_a = wW / (self.R * self.war)
                        we_a = wE / (self.R * self.war)
                        ws_a = wS / (self.R * self.war)
                        wn_a = wN / (self.R * self.war)

                        fdiss = 2 * (
                            (
                                self.Cr * hP * (up_a - uw_a) / dksi
                                - np2 * self.Cr * dhdksi_p * (up_a - us_a) / self.dN
                            )
                            ** 2
                            + (self.betha_s * self.Rs * (vn_a - vp_a) / self.dN) ** 2
                            + (
                                self.betha_s * self.Rs * (up_a - us_a) / self.dN
                                + self.Cr * hP * (vp_a - vw_a) / dksi
                                - np2 * self.Cr * dhdksi_p * (vp_a - vs_a) / dksi
                            )
                            ** 2
                            + (
                                self.Cr * hP * (wp_a - ww_a) / dksi
                                - np2 * self.Cr * dhdksi_p * (wp_a - ws_a) / self.dN
                            )
                            ** 2
                            + (self.betha_s * self.Rs * (wp_a - ws_a) / dksi) ** 2
                        )

                        # Source term ----------------
                        Bp = (
                            JP
                            * ((self.war * self.R) ** 2)
                            * Mi[ki, kj]
                            / self.Cp
                            * self.dN
                            * dksi
                            * fdiss
                        )

                        k = k + 1

                        b[k - 1] = Bp

                        if ki == 0 and kj == 0:
                            Mat_coef[k - 1, k - 1] = Ap + An - Aw
                            Mat_coef[k - 1, k] = Ae + Ane
                            Mat_coef[k - 1, k - 1 + self.ntheta] = As - Asw
                            Mat_coef[k - 1, k + self.ntheta] = Ase
                            b[k - 1] = (
                                b[k - 1]
                                - 2 * (Aw * Tmist[ki] + Asw * Tmist[ki + 1])
                                - Anw * (Tmist[ki - 1])
                            )

                        if ki == 0 and kj > 0 and kj < self.nX - 1:
                            Mat_coef[k - 1, k - 1] = Ap + An
                            Mat_coef[k - 1, k] = Ae + Ane
                            Mat_coef[k - 1, k - 2] = Aw + Anw
                            Mat_coef[k - 1, k + self.ntheta - 1] = As
                            Mat_coef[k - 1, k + self.ntheta] = Ase
                            Mat_coef[k - 1, k + self.ntheta - 2] = Asw

                        if ki == 0 and kj == self.nX:
                            Mat_coef[k - 1, k - 1] = Ap + An + Ane + Ae
                            Mat_coef[k - 1, k - 2] = Aw + Anw
                            Mat_coef[k - 1, k + self.ntheta - 1] = As + Ase
                            Mat_coef[k - 1, k + self.ntheta - 2] = Asw

                        if kj == 0 and ki > 0 and ki < self.nN - 1:
                            Mat_coef[k - 1, k - 1] = Ap - Aw
                            Mat_coef[k - 1, k] = Ae
                            Mat_coef[k - 1, k + self.ntheta - 1] = As - Asw
                            Mat_coef[k - 1, k - self.ntheta - 1] = An - Anw
                            Mat_coef[k - 1, k + self.ntheta] = Ase
                            Mat_coef[k - 1, k - self.ntheta] = Ane
                            b[k - 1] = (
                                b[k - 1]
                                - 2 * Tmist[ki - 2] * Anw
                                - 2 * Tmist[ki] * Aw
                                - 2 * Tmist[ki + 1] * Asw
                            )

                        if ki > 0 and ki < self.nN - 1 and kj > 0 and kj < self.nX - 1:
                            Mat_coef[k - 1, k - 1] = Ap
                            Mat_coef[k - 1, k] = Ae
                            Mat_coef[k - 1, k - 2] = Aw
                            Mat_coef[k - 1, k + self.ntheta - 1] = As
                            Mat_coef[k - 1, k - self.ntheta - 1] = An
                            Mat_coef[k - 1, k + self.ntheta] = Ase
                            Mat_coef[k - 1, k + self.ntheta - 2] = Asw
                            Mat_coef[k - 1, k - self.ntheta] = Ane
                            Mat_coef[k - 1, k - self.ntheta - 2] = Anw

                        if kj == self.nX - 1 and ki > 0 and ki < self.nN - 1:
                            Mat_coef[k - 1, k - 1] = Ap + Ae
                            Mat_coef[k - 1, k - 2] = Aw
                            Mat_coef[k - 1, k + self.ntheta - 1] = As + Ase
                            Mat_coef[k - 1, k - self.ntheta - 1] = An + Ane
                            Mat_coef[k - 1, k + self.ntheta - 2] = Asw
                            Mat_coef[k - 1, k - self.ntheta - 2] = Anw

                        if kj == 0 and ki == self.nN - 1:
                            Mat_coef[k - 1, k - 1] = Ap + As - Aw
                            Mat_coef[k - 1, k] = Ae + Ase
                            Mat_coef[k - 1, k - self.ntheta - 1] = An - Anw
                            Mat_coef[k - 1, k - self.ntheta] = Ane
                            b[k - 1] = (
                                b[k - 1]
                                - 2 * Tmist[ki] * Aw
                                - 2 * Tmist[ki - 1] * Anw
                                - Tmist[ki + 1] * Asw
                            )

                        if ki == self.nN - 1 and kj > 0 and kj < self.nX - 1:
                            Mat_coef[k - 1, k - 1] = Ap + As
                            Mat_coef[k - 1, k] = Ae + Ase
                            Mat_coef[k - 1, k - 2] = Aw + Asw
                            Mat_coef[k - 1, k - self.ntheta - 1] = An
                            Mat_coef[k - 1, k - self.ntheta] = Ane
                            Mat_coef[k - 1, k - self.ntheta - 2] = Anw

                        if ki == self.nN - 1 and kj == self.nX - 1:
                            Mat_coef[k - 1, k - 1] = Ap + As + Ae + Ase
                            Mat_coef[k - 1, k - 2] = Aw + Asw
                            Mat_coef[k - 1, k - self.ntheta - 1] = An + Ane
                            Mat_coef[k - 1, k - self.ntheta - 2] = Anw

                        kj = kj + 1

                    kj = 0
                    ki = ki + 1

                ki = 0
                nn = 0

                # Linear system solution via pseudoinverse for robustness
                t = np.dot(pinv(Mat_coef), b)

                # Temperature matrix ----------------------
                cont = 0

                for i in range(0, self.nN):
                    for j in range(0, self.nX):
                        T[i, j] = t[cont]
                        cont = cont + 1

                # Viscosity equation ========================================================================
                # VG68 - Polynomial adjustment using predetermined values
                #
                # Via Oil_Regression_Analyses.m and propvalue.m the equation coefficients
                # are obtained for the regression on the viscosity determination as a
                # function of the temperature.

                # 3D temperature field -----------------------------------------------------
                TT = np.zeros((self.nZ, self.nX, self.nN))
                for k in range(0, self.nN):
                    for j in range(0, self.nX):
                        TT[:, j, k] = T[k, j]

                # Regression equation coefficients
                a = 5.506e-09
                b = 5012
                c = 0.1248
                minovo = a * np.exp(b / (TT + 273.15 + c))
                k = 0

                # Full temperature matrix, including borders
                for i in range(1, self.nN):
                    for j in range(1, self.ntheta):
                        T_novo[i, j] = T[i - 1, j - 1]

                T_novo[0,] = T_novo[
                    1,
                ]
                T_novo[self.nN + 1, :] = T_novo[self.nN, :]
                T_novo[1 : self.nN, 0] = Tmist[1 : self.nN]
                T_novo[:, self.nX + 1] = T_novo[:, self.nX]

            # WHILE ENDS HERE ==========================================================

            T1[:, :, n_p] = T_novo[:, :]
            P1[:, :, n_p] = PPdim

            yh = (
                self.Rs
                - self.R
                - (
                    np.sin(self.Xtheta) * (yr + alpha * (self.Rs + self.esp))
                    + np.cos(self.Xtheta) * (xr + self.Rs - self.R - self.Cr)
                )
            )
            for jj in range(0, self.nX + 1):
                YH[:, jj, n_p] = np.fliplr(np.linspace(0, yh[jj], self.nN + 2))

            # Integration of pressure field - HydroForces
            auxF = np.array([np.cos(self.Xtheta[0:-1]), np.sin(self.Xtheta[0:-1])])
            dA = self.dx * self.dz

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

            My[n_p] = fyj * (self.Rs + self.esp)

            if fxj[n_p] >= -1:
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
            fxj[0] * np.cos(self.psi_pad[0] + self.sigma[0])
            + fxj[1] * np.cos(self.psi_pad[1] + self.sigma[1])
            + fxj[2] * np.cos(self.psi_pad[2] + self.sigma[2])
            + fxj[3] * np.cos(self.psi_pad[3] + self.sigma[3])
            + fxj[4] * np.cos(self.psi_pad[4] + self.sigma[4])
            + fxj[5] * np.cos(self.psi_pad[5] + self.sigma[5])
        )
        Fhy = (
            fxj[0] * np.sin(self.psi_pad[0] + self.sigma[0])
            + fxj[1] * np.sin(self.psi_pad[1] + self.sigma[1])
            + fxj[2] * np.sin(self.psi_pad[2] + self.sigma[2])
            + fxj[3] * np.sin(self.psi_pad[3] + self.sigma[3])
            + fxj[4] * np.sin(self.psi_pad[4] + self.sigma[4])
            + fxj[5] * np.sin(self.psi_pad[5] + self.sigma[5])
        )

        return Fhx, Fhy

    def run(self):
        Fhx, Fhy = self._forces()
        print(f"Fhx = {Fhx}\nFhy = {Fhy}\n")


if __name__ == "__main__":

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
    # psi_pad = x
    psi_pad = np.array([x[0], x[1], x[2], x[3], x[4], x[5]])
    npad = 6

    # Radial clearance
    Cr = 250e-6

    # Oil tank temperature
    Tcuba = 40

    alpha = 0  # * (2 * np.pi * 5) * alpha

    # Geometric parameters for the bearing --------------------------------------------

    # Journal radius
    R = 0.5 * 930e-3

    # Pad radius
    Rs = 0.5 * 934e-3  # [m]

    # Pad thickness
    esp = 67e-3  # [m]

    # Pad arc
    betha_s = 25  # [degree]

    # Pivot position (arc pivot/arc pad)
    rp_pad = 0.6

    # Bength of bearing
    L = 197e-3  # [m]

    # Angular position of the pivot
    sigma = np.array([0, 300, 60])  # [degree]

    # Bearing loading
    fR = 90.6e3  # [N]

    # Rotor speed
    wa = 300  # [rpm]

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

    #  Discretized Mesh ------------------------------------------------------

    # Number of volumes in theta direction
    ntheta = 48

    # Number of volumes in x direction
    nX = ntheta

    # Number of volumes in z direction
    nZ = 48

    # Number of volumes in neta direction
    nN = 30

    mancal = Tilting(
        R,
        Rs,
        esp,
        betha_s,
        rp_pad,
        L,
        sigma,
        fR,
        wa,
        kt,
        Cp,
        rho,
        mi_ref,
        E,
        phi,
        psi_pad,
        npad,
        Cr,
        alpha,
        Tcuba,
        ntheta,
        nX,
        nZ,
        nN,
    )
    mancal.run()
    # mancal.coefficients()