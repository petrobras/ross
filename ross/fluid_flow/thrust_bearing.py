import numpy as np
from numpy.linalg import pinv
from scipy.linalg import solve
from scipy.optimize import fmin
from decimal import Decimal


class Thrust:
    def __init__(
        r1,
        r2,
        rp,
        teta0,
        tetap,
        TC,
        Tin,
        T0,
        rho,
        cp,
        kt,
        k1,
        k2,
        k3,
        mi0,
        fz,
        Npad,
        NTETA,
        NR,
        war,
        R1,
        R2,
        TETA1,
        TETA2,
        Rp,
        TETAp,
        dR,
        dTETA,
        Ti,
        x0,
    ):
        self.r1 = r1
        self.r2 = r2
        self.rp = rp
        self.teta0 = teta0
        self.tetap = tetap
        self.TC = TC
        self.Tin = Tin
        self.T0 = T0
        self.rho = rho
        self.cp = cp
        self.kt = kt
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.mi0 = (1e-3) * k1 * np.exp(k2 / (T0 - k3))
        self.fz = fz
        self.Npad = Npad
        self.NTETA = NTETA
        self.NR = NR
        self.war = wa * (np.pi / 30)
        self.R1 = R1
        self.R2 = R2
        self.TETA1 = TETA1
        self.TETA2 = TETA2
        self.Rp = Rp
        self.TETAp = TETAp
        self.dR = dR
        self.dTETA = dTETA
        self.Ti = T0 * (1 + np.zeros(NR, NTETA))
        self.x0 = x0

        # --------------------------------------------------------------------------
        # PRE-PROCESSING

        # loop counters for ease of understanding
        vec_R = np.arange((R1 + 0.5 * dR), (R2 - 0.5 * dR), dR)
        vec_TETA = np.arange((TETA1 + 0.5 * dTETA), (TETA2 - 0.5 * dTETA), dTETA)

        # --------------------------------------------------------------------------
        # WHILE LOOP INITIALIZATION
        ResFM = 1
        tolFM = 1e-8
        while ResFM >= tolFM:
            # --------------------------------------------------------------------------
            # Equilibrium position optimization [h0,ar,ap]
            x = scipy.optimize.fmin(
                ArAsh0Equilibrium,
                x0,
                args=(),
                xtol=tolFM,
                ftol=tolFM,
                maxiter=100000,
                maxfun=100000,
                full_output=0,
                disp=1,
                retall=0,
                callback=None,
                initial_simplex=None,
            )
            a_r = x[0]  # [rad]
            a_s = x[1]  # [rad]
            h0 = x[2]  # [m]

            # --------------------------------------------------------------------------
            #  Temperature field
            tolMI = 1e-6
            [T, resMx, resMy, resFre] = TEMPERATURE(h0, a_r, a_s, tolMI)
            Ti = T * T0
            ResFM = np.norm(resMx, resMy, resFre)
            xo = x

        # --------------------------------------------------------------------------
        # Full temperature field
        TT = 1 + np.zeros(NR + 1, NTETA + 1)
        TT[1:NR, 1:NTETA] = np.fliplr(Ti)
        TT[:, 0] = T0
        TT[0, :] = TT[1, :]
        TT[NR + 1, :] = TT[NR, :]
        TT[:, NTETA + 1] = TT[:, NTETA]
        TT = TT - 273.15

        # --------------------------------------------------------------------------
        # Viscosity field
        for ii in range(0, NR):
            for jj in range(0, NTETA):
                mi[ii, jj] = (1e-3) * k1 * np.exp(k2 / (Ti[ii, jj] - k3))  # [Pa.s]

        # ==========================================================================
        # PRESSURE =================================================================
        # STARTS HERE ==============================================================
        # ==========================================================================

        [P0, H0, H0ne, H0nw, H0se, H0sw] = PRESSURE(a_r, a_s, h0, mi)

        # ==========================================================================
        # PRESSURE =================================================================
        # ENDS HERE ================================================================
        # ==========================================================================

        # --------------------------------------------------------------------------
        # Stiffness and Damping Coefficients
        wp = war  # perturbation frequency [rad/s]
        WP = wp / war

        # ==========================================================================
        # HYDROCOEFF_z =============================================================
        # STARTS HERE ==============================================================
        # ==========================================================================

        MI = (1 / mi0) * mi

        kR = 0
        kTETA = 0
        k = -1  # pressure vectorization index
        nk = NR * NTETA  # volumes number

        # coefficients matrix
        Mat_coef = np.zeros(nk, nk)
        b = np.zeros(nk, 1)
        cont = 0

        for R in vec_R:
            for TETA in vec_TETA:

                cont = cont + 1
                TETAe = TETA + 0.5 * dTETA
                TETAw = TETA - 0.5 * dTETA
                Rn = R + 0.5 * dR
                Rs = R - 0.5 * dR

                if kTETA == 0 and kR == 0:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / dTETA
                    dPdTETAw = P0[kR, kTETA] / (0.5 * dTETA)
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = P0[kR, kTETA] / (0.5 * dR)

                if kTETA == 0 and kR > 0 and kR < NR - 1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / dTETA
                    dPdTETAw = P0[kR, kTETA] / (0.5 * dTETA)
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / dR

                if kTETA == 0 and kR == NR - 1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / dTETA
                    dPdTETAw = P0[kR, kTETA] / (0.5 * dTETA)
                    dPdRn = -P0[kR, kTETA] / (0.5 * dR)
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / dR

                if kR == 0 and kTETA > 0 and kTETA < NTETA:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / dTETA
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = P0[kR, kTETA] / (0.5 * dR)

                if kTETA > 0 and kTETA < NTETA and kR > 0 and kR < NR:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / dTETA
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / dR

                if kR == NR and kTETA > 0 and kTETA < NTETA:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / dTETA
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = -P0[kR, kTETA] / (0.5 * dR)
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / dR

                if kR == 0 and kTETA == NTETA:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]
                    dPdTETAe = -P0[kR, kTETA] / (0.5 * dTETA)
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = P0[kR, kTETA] / (0.5 * dR)

                if kTETA == NTETA and kR > 0 and kR < NR:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = -P0[kR, kTETA] / (0.5 * dTETA)
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / dR

                if kTETA == NTETA and kR == NR:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = -P0[kR, kTETA] / (0.5 * dTETA)
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = -P0[kR, kTETA] / (0.5 * dR)
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / dR

                As_ne = 1
                As_nw = 1
                As_se = 1
                As_sw = 1

                # G1=dhpivotdR=0
                G1_ne = 0
                G1_nw = 0
                G1_se = 0
                G1_sw = 0

                # Gs=dhpivotdTETA=0
                G2_ne = 0
                G2_nw = 0
                G2_se = 0
                G2_sw = 0

                # Coefficients for solving the Reynolds equation
                CE_1 = (
                    1
                    / (24 * teta0 ** 2 * MI_e)
                    * (dR / dTETA)
                    * (
                        As_ne * H0ne[kR, kTETA] ** 3 / Rn
                        + As_se * H0se[kR, kTETA] ** 3 / Rs
                    )
                )
                CE_2 = (
                    dR
                    / (48 * teta0 ** 2 * MI_e)
                    * (
                        G2_ne * H0ne[kR, kTETA] ** 3 / Rn
                        + G2_se * H0se[kR, kTETA] ** 3 / Rs
                    )
                )
                CE = CE_1 + CE_2

                CW_1 = (
                    1
                    / (24 * teta0 ** 2 * MI_w)
                    * (dR / dTETA)
                    * (
                        As_nw * H0nw[kR, kTETA] ** 3 / Rn
                        + As_sw * H0sw[kR, kTETA] ** 3 / Rs
                    )
                )
                CW_2 = (
                    -dR
                    / (48 * teta0 ** 2 * MI_w)
                    * (
                        G2_nw * H0nw[kR, kTETA] ** 3 / Rn
                        + G2_sw * H0sw[kR, kTETA] ** 3 / Rs
                    )
                )
                CW = CW_1 + CW_2

                CN_1 = (
                    Rn
                    / (24 * MI_n)
                    * (dTETA / dR)
                    * (As_ne * H0ne[kR, kTETA] ** 3 + As_nw * H0nw[kR, kTETA] ** 3)
                )
                CN_2 = (
                    Rn
                    / (48 * MI_n)
                    * (dTETA)
                    * (G1_ne * H0ne[kR, kTETA] ** 3 + G1_nw * H0nw[kR, kTETA] ** 3)
                )
                CN = CN_1 + CN_2

                CS_1 = (
                    Rs
                    / (24 * MI_s)
                    * (dTETA / dR)
                    * (As_se * H0se[kR, kTETA] ** 3 + As_sw * H0sw[kR, kTETA] ** 3)
                )
                CS_2 = (
                    -Rs
                    / (48 * MI_s)
                    * (dTETA)
                    * (G1_se * H0se[kR, kTETA] ** 3 + G1_sw * H0sw[kR, kTETA] ** 3)
                )
                CS = CS_1 + CS_2

                CP = -(CE_1 + CW_1 + CN_1 + CS_1) + (CE_2 + CW_2 + CN_2 + CS_2)

                B_1 = (Rn * dTETA / (8 * MI_n)) * dPdRn * (
                    As_ne * H0ne[kR, kTETA] ** 2 + As_nw * H0nw[kR, kTETA] ** 2
                ) - (Rs * dTETA / (8 * MI_s)) * dPdRs * (
                    As_se * H0se[kR, kTETA] ** 2 + As_sw * H0sw[kR, kTETA] ** 2
                )
                B_2 = (dR / (8 * teta0 ** 2 * MI_e)) * dPdTETAe * (
                    As_ne * H0ne[kR, kTETA] ** 2 / Rn
                    + As_se * H0se[kR, kTETA] ** 2 / Rs
                ) - (dR / (8 * teta0 ** 2 * MI_w)) * dPdTETAw * (
                    As_nw * H0nw[kR, kTETA] ** 2 / Rn
                    + As_sw * H0sw[kR, kTETA] ** 2 / Rs
                )
                B_3 = dR / (4 * teta0) * (As_ne * Rn + As_se * Rs) - dR / (
                    4 * teta0
                ) * (As_nw * Rn + As_sw * Rs)
                B_4 = (
                    complex(0, 1)
                    * WP
                    * dR
                    * dTETA
                    / 4
                    * (Rn * As_ne + Rn * As_nw + Rs * As_se + Rs * As_sw)
                )

                # vectorization index
                k = k + 1

                b[k, 0] = -(B_1 + B_2) + B_3 + B_4

                if kTETA == 0 and kR == 0:
                    Mat_coef[k, k] = CP - CW - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == 0 and kR > 0 and kR < NR:
                    Mat_coef[k, k] = CP - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + NTETA] = CN
                    Mat_coef[k, k - NTETA] = CS

                if kTETA == 0 and kR == NR:
                    Mat_coef[k, k] = CP - CW - CN
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - NTETA] = CS

                if kR == 0 and kTETA > 0 and kTETA < NTETA:
                    Mat_coef[k, k] = CP - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN

                if kTETA > 0 and kTETA < NTETA and kR > 0 and kR < NR:
                    Mat_coef[k, k] = CP
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN
                    Mat_coef[k, k - NTETA] = CS
                    Mat_coef[k, k + 1] = CE

                if kR == NR and kTETA > 0 and kTETA < NTETA:
                    Mat_coef[k, k] = CP - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - NTETA] = CS

                if kR == 0 and kTETA == NTETA:
                    Mat_coef[k, k] = CP - CE - CS
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == NTETA and kR > 0 and kR < NR:
                    Mat_coef[k, k] = CP - CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - NTETA] = CS
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == NTETA and kR == NR:
                    Mat_coef[k, k] = CP - CE - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - NTETA] = CS

                kTETA = kTETA + 1

            kR = kR + 1
            kTETA = 0

        # vectorized pressure field solution
        p = np.linalg.solve(Mat_coef, b)
        cont = 0

        # pressure matrix
        for ii in range(0, NR):
            for jj in range(0, NTETA):
                cont = cont + 1
                P[ii, jj] = p[cont]

        # dimensional pressure
        Pdim = P * (r1 ** 2) * war * mi0 / (h0 ** 3)

        # RESULTING FORCE AND MOMENTUM: Equilibrium position
        XR = r1 * vec_R
        XTETA = teta0 * vec_TETA
        Xrp = rp * (1 + np.zeros(XR, XR))

        for ii in range(0, NTETA):
            Mxr[:, ii] = (Pdim[:, ii] * (np.transpose(XR) ** 2)) * np.sin(
                XTETA(ii) - tetap
            )
            Myr[:, ii] = (
                -Pdim[:, ii]
                * np.transpose(XR)
                * np.transpose(XR * np.cos(XTETA(ii) - tetap) - Xrp)
            )
            Frer[:, ii] = Pdim[:, ii] * np.transpose(XR)

        mxr = np.trapz[XR, Mxr]
        myr = np.trapz[XR, Myr]
        frer = np.trapz[XR, Frer]

        mx = -np.trapz[XTETA, mxr]
        my = -np.trapz[XTETA, myr]
        fre = -np.trapz[XTETA, frer]

        # ==========================================================================
        # HYDROCOEFF_z =============================================================
        # ENDS HERE ================================================================
        # ==========================================================================

        K = Npad * np.real(kk_zz)  # Stiffness Coefficient
        C = Npad * 1 / wp * np.imag(kk_zz)  # Damping Coefficient

        # --------------------------------------------------------------------------
        # Output values - Pmax [Pa]- hmax[m] - hmin[m] - h0[m]
        Pmax = np.max(PPdim)
        hmax = np.max(h0 * H0)
        hmin = np.min(h0 * H0)
        Tmax = np.max(TT)
        h0


def thrust_bearing_example():
    """Create an example of a thrust bearing with hydrodynamic effects. 
    This function returns pressure field and dynamic coefficient. The 
    purpose is to make available a simple model so that a doctest can be 
    written using it.

    Returns
    -------
    Thrust : ross.Thrust Object
        An instance of a hydrodynamic thrust bearing model object.
    Examples
    --------
    >>> bearing = thrust_bearing_example()
    >>> bearing.L
    0.263144
    """

    bearing = Thrust(
        r1=0.5 * 90e-3,  # pad inner radius [m]
        r2=0.5 * 160e-3,  # pad outer radius [m]
        rp=(r2 - r1) * 0.5 + r1,  # pad pivot radius [m]
        teta0=35 * pi / 180,  # pad complete angle [rad]
        tetap=19.5 * pi / 180,  # pad pivot angle [rad]
        TC=40 + 273.15,  # Collar temperature [K]
        Tin=40 + 273.15,  # Cold oil temperature [K]
        T0=0.5 * (TC + Tin),  # Reference temperature [K]
        rho=870,  # Oil density [kg/m³]
        cp=1850,  # Oil thermal capacity [J/kg/K]
        kt=0.15,  # Oil thermal conductivity [W/m/K]
        k1=0.06246,  # Coefficient for ISO VG 32 turbine oil - Vogel's equation
        k2=868.8,  # Coefficient for ISO VG 32 turbine oil - Vogel's equation
        k3=170.4,  # Coefficient for ISO VG 32 turbine oil - Vogel's equation
        mi0=1e-6 * rho * 22,  # Oil VG 22
        fz=370 * 9.81,  # Loading in Y direction [N]
        Npad=3,  # Number of PADs
        NTETA=40,  # TETA direction N volumes
        NR=40,  # R direction N volumes
        war=(1200 * pi) / 30,  # Shaft rotation speed [RPM]
        R1=1,  # Inner pad FEM radius
        R2=r2 / r1,  # Outer pad FEM radius
        TETA1=0,  # Initial angular coordinate
        TETA2=1,  # Final angular coordinate
        Rp=rp / r1,  # Radial pivot position
        TETAp=tetap / teta0,  # Angular pivot position
        dR=(R2 - R1) / (NR),  # R direction volumes length
        dTETA=(TETA2 - TETA1) / (NTETA),  # TETA direction volumes length
        Ti=T0 * ones(NR, NTETA),  # Initial temperature field [°C]
        x0=np.array(
            -2.251004554793839e-04, -1.332796067467349e-04, 2.152552477569639e-05
        ),  # Initial equilibrium position
    )

    return bearing
