import numpy as np
import scipy

# import tensorflow as tf
import mpmath as fp

from numpy.linalg import pinv
from scipy.linalg import solve
from scipy.optimize import fmin
from decimal import Decimal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
class Thrust:
    """ This class calculates the pressure and temperature fields, equilibrium position of a tilting-pad thrust bearing. It is also possible to obtain the stiffness and damping coefficients.
    
    Parameters
    ----------
    Bearing Geometry
    ^^^^^^^^^^^^^^^^
    Describes the geometric characteristics.
    r1 : float
        Inner pad radius. Default unit is meter.
    r2 : float
        Outer pad radius. Default unit is meter.
    rp : float
        Pivot pad radius. Default unit is meter.
    teta0 : float
          Arc length of each pad. The unit is degree.
    tetap : float
          Angular pivot position. The unit is degree.
    Npad : integer
         Number of pads
    
    Operating conditions
    ^^^^^^^^^^^^^^^^^^^^
    Describes the operating conditions of the bearing
    speed : float
        Rotor rotating speed. Default unit is rad/s
    fz : Float
        Axial load. The unit is Newton.
    Tcub : Float
        Oil bath temperature. The unit is °C
    
    Fluid properties
    ^^^^^^^^^^^^^^^^
    Describes the fluid characteristics.
    mi0 : float
        Reference fluid viscosity. The unit is Pa*s.
    rho : float
        Fluid specific mass. Default unit is kg/m^3.
    kt : float
        Fluid thermal conductivity. The unit is J/(s*m*°C).
    cp : float
        Fluid specific heat. The unit is J/(kg*°C).
    T0 : float
        Reference fluid temperature. The unit is °C.
    k1, k2, and k3 : float
        Oil coefficients for the viscosity interpolation.

    Mesh discretization
    ^^^^^^^^^^^^^^^^^^^
    Describes the discretization of the fluid film
    NR : int
        Number of volumes along the R direction.
    NTETA : int
        Number of volumes along the TETA direction. 
    
    
    Returns
    -------
    Pmax : float
          Maximum pressure. The unit is Pa.
    Tmax : float
          Maximum temperature. The unit is °C.
    h0 : float
          oil film thickness at the pivot point. The unit is m.
    hmax : float
         maximum oil film thickness. The unit is m.
    hmin : float
         minimum oil film thickness. The unit is m.
    K : float
         bearing stiffness coefficient. The unit is N/m.
    C : float
         bearing damping coefficient. The unit is N.s/m.
    PPdim : array
         pressure field. The unit is Pa.
    XH,YH : array
         mesh grid. The uni is m.
    TT : array
         temperature field. The unit is °C.

    References
    ----------
    .. [1] BARBOSA, J.S. Analise de Modelos Termohidrodinamicos para Mancais de unidades geradoras Francis. 2016. Dissertacao de Mestrado. Universidade Federal de Uberlandia, Uberlandia. ..
    .. [2] HEINRICHSON, N.; SANTOS, I. F.; FUERST, A., The Influence of Injection Pockets on the Performance of Tilting Pad Thrust Bearings Part I Theory. Journal of Tribology, 2007. .. 
    .. [3] NICOLETTI, R., Efeitos Termicos em Mancais Segmentados Hibridos Teoria e Experimento. 1999. Dissertacao de Mestrado. Universidade Estadual de Campinas, Campinas. ..
    .. [4] LUND, J. W.; THOMSEN, K. K. A calculation method and data for the dynamic coefficients of oil lubricated journal bearings. Topics in fluid film bearing and rotor bearing system design and optimization, n. 1000118, 1978. ..
    Attributes
    ----------
    dPdR : array 
    """
    def __init__(
        self,
        r1,
        r2,
        rp,
        teta0,
        tetap,
        Tcub,
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
        wa,
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
        # E_pad,
        # tpad,
        # v_pad,
        # alpha_pad,
        # mi,
        # P0,
        # MI,
        # H0,
        # H0ne,
        # H0se,
        # H0nw,
        # H0sw,
    ):
        self.r1 = r1
        self.r2 = r2
        self.rp = rp
        self.teta0 = teta0
        self.tetap = tetap
        self.Tcub = Tcub
        self.TC = TC
        self.Tin = Tin
        self.T0 = T0
        self.rho = rho
        self.cp = cp
        self.kt = kt
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.mi0 = k1 * np.exp(k2 / (T0 + k3))
        self.fz = fz
        self.Npad = Npad
        self.NTETA = NTETA
        self.NR = NR
        self.wa = wa
        self.war = wa * (np.pi / 30)
        self.R1 = R1
        self.R2 = R2
        self.TETA1 = TETA1
        self.TETA2 = TETA2
        self.Rp = Rp
        self.TETAp = TETAp
        self.dR = dR
        self.dTETA = dTETA
        self.Ti = T0 * (np.ones((NR, NTETA)))
        self.x0 = x0
        # self.E_pad = E_pad
        # self.tpad = tpad
        # self.v_pad = v_pad
        # self.alpha_pad = alpha_pad
        # self.mi = mi
        # self.P0 = P0
        # self.MI = MI
        # self.H0 = H0
        # self.H0ne = H0ne
        # self.H0se = H0se
        # self.H0nw = H0nw
        # self.H0sw = H0sw

        # --------------------------------------------------------------------------
        # Pre-processing loop counters for ease of understanding
        vec_R = np.zeros(NR)
        vec_R[0] = R1 + 0.5 * dR

        vec_TETA = np.zeros(NTETA)
        vec_TETA[0] = TETA1 + 0.5 * dTETA


        for ii in range(1, NR):
            vec_R[ii] = vec_R[ii-1] + dR

        for jj in range(1, NTETA):
            vec_TETA[jj] = vec_TETA[jj-1] + dTETA

        fp.mp.dps = 800  # numerical solver precision setting

        # --------------------------------------------------------------------------
        # WHILE LOOP INITIALIZATION
        ResFM = 1
        tolFM = 1e-8

        while ResFM >= tolFM:
            print(ResFM)######################################################################################################################################
            # --------------------------------------------------------------------------
            # Equilibrium position optimization [h0,ar,ap]
            
            entrada= tuple((r1,rp,teta0,tetap,mi0,fz,Npad,NTETA,NR,war,R1,R2,TETA1,TETA2,Rp,dR,dTETA,k1,k2,k3,TETAp,Ti,x0))
            
            x = scipy.optimize.fmin(
                ArAsh0Equilibrium,                  
                x0,
                args=entrada,
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

            # TEMPERATURE ==============================================================
            # STARTS HERE ==============================================================

            dHdT = 0
            mi_i = np.zeros((NR, NTETA))

            # initial temperature field
            T_i = Ti

            for ii in range(0, NR):
                for jj in range(0, NTETA):
                    mi_i[ii, jj] = (
                        k1 * np.exp(k2 / (T_i[ii, jj] + k3))
                    )  # [Pa.s]

            MI_new = (1 / mi0) * mi_i
            MI = 0.2 * MI_new

            # TEMPERATURE FIELD - Solution of ENERGY equation
            for ii in range(0, NR):
                for jj in range(0, NTETA):
                    varMI = np.abs((MI_new[ii, jj] - MI[ii, jj]) / MI[ii, jj])
            aux1=1
            
            while aux1 >= tolMI:

                MI = np.array(MI_new)

                # PRESSURE_THD =============================================================
                # STARTS HERE ==============================================================

                Ar = a_r * r1 / h0
                As = a_s * r1 / h0

                # PRESSURE FIELD - Solution of Reynolds equation
                kR = 0
                kTETA = 0

                # pressure vectorization index
                k = -1

                # volumes number
                nk = (NR) * (NTETA)

                # Variable initialization
                Mat_coef = np.zeros((nk, nk))
                b = np.zeros((nk, 1))
                H0 = np.zeros((NR, NTETA))
                H0ne = np.zeros((NR, NTETA))
                H0nw = np.zeros((NR, NTETA))
                H0se = np.zeros((NR, NTETA))
                H0sw = np.zeros((NR, NTETA))
                dP0dR = np.zeros((NR, NTETA))
                dP0dTETA = np.zeros((NR, NTETA))
                T_new = np.zeros((NR, NTETA))
                Mxr = np.zeros((NR, NTETA))
                Myr = np.zeros((NR, NTETA))
                Frer = np.zeros((NR, NTETA))
                P0 = np.ones((NR, NTETA))
                P = np.zeros((NR, NTETA))
                mi = np.zeros((NR, NTETA))

                PPdim = np.zeros((NR + 2, NTETA + 2))
                

                cont = -1

                for R in vec_R:
                    for TETA in vec_TETA:

                        cont = cont + 1
                        TETAe = TETA + 0.5 * dTETA
                        TETAw = TETA - 0.5 * dTETA
                        Rn = R + 0.5 * dR
                        Rs = R - 0.5 * dR

                        # oil film thickness
                        H0[kR, kTETA] = (
                            (h0 / h0)
                            + As * (Rp - R * np.cos(teta0 * (TETA - TETAp)))
                            + Ar * R * np.sin(teta0 * (TETA - TETAp))
                        )
                        H0ne[kR, kTETA] = (
                            (h0 / h0)
                            + As * (Rp - Rn * np.cos(teta0 * (TETAe - TETAp)))
                            + Ar * Rn * np.sin(teta0 * (TETAe - TETAp))
                        )
                        H0nw[kR, kTETA] = (
                            (h0 / h0)
                            + As * (Rp - Rn * np.cos(teta0 * (TETAw - TETAp)))
                            + Ar * Rn * np.sin(teta0 * (TETAw - TETAp))
                        )
                        H0se[kR, kTETA] = (
                            (h0 / h0)
                            + As * (Rp - Rs * np.cos(teta0 * (TETAe - TETAp)))
                            + Ar * Rs * np.sin(teta0 * (TETAe - TETAp))
                        )
                        H0sw[kR, kTETA] = (
                            (h0 / h0)
                            + As * (Rp - Rs * np.cos(teta0 * (TETAw - TETAp)))
                            + Ar * Rs * np.sin(teta0 * (TETAw - TETAp))
                        )

                        if kTETA == 0 and kR == 0:
                            MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                            MI_w = MI[kR, kTETA]
                            MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                            MI_s = MI[kR, kTETA]

                        if kTETA == 0 and kR > 0 and kR < NR-1:
                            MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                            MI_w = MI[kR, kTETA]
                            MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                            MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                        if kTETA == 0 and kR == NR-1:
                            MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                            MI_w = MI[kR, kTETA]
                            MI_n = MI[kR, kTETA]
                            MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                        if kR == 0 and kTETA > 0 and kTETA < NTETA-1:
                            MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                            MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                            MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                            MI_s = MI[kR, kTETA]

                        if kTETA > 0 and kTETA < NTETA-1 and kR > 0 and kR < NR-1:
                            MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                            MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                            MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                            MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                        if kR == NR-1 and kTETA > 0 and kTETA < NTETA-1:
                            MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                            MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                            MI_n = MI[kR, kTETA]
                            MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                        if kR == 0 and kTETA == NTETA-1:
                            MI_e = MI[kR, kTETA]
                            MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                            MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                            MI_s = MI[kR, kTETA]

                        if kTETA == NTETA-1 and kR > 0 and kR < NR-1:
                            MI_e = MI[kR, kTETA]
                            MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                            MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                            MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                        if kTETA == NTETA-1 and kR == NR-1:
                            MI_e = MI[kR, kTETA]
                            MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                            MI_n = MI[kR, kTETA]
                            MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                        # Coefficients for solving the Reynolds equation
                        CE = (
                            1
                            / (24 * teta0 ** 2 * MI_e)
                            * (dR / dTETA)
                            * (H0ne[kR, kTETA] ** 3 / Rn + H0se[kR, kTETA] ** 3 / Rs)
                        )
                        CW = (
                            1
                            / (24 * teta0 ** 2 * MI_w)
                            * (dR / dTETA)
                            * (H0nw[kR, kTETA] ** 3 / Rn + H0sw[kR, kTETA] ** 3 / Rs)
                        )
                        CN = (
                            Rn
                            / (24 * MI_n)
                            * (dTETA / dR)
                            * (H0ne[kR, kTETA] ** 3 + H0nw[kR, kTETA] ** 3)
                        )
                        CS = (
                            Rs
                            / (24 * MI_s)
                            * (dTETA / dR)
                            * (H0se[kR, kTETA] ** 3 + H0sw[kR, kTETA] ** 3)
                        )
                        CP = -(CE + CW + CN + CS)

                        # vectorization index
                        k = k + 1
                        b[k, 0] = (
                            dR
                            / (4 * teta0)
                            * (
                                Rn * H0ne[kR, kTETA]
                                + Rs * H0se[kR, kTETA]
                                - Rn * H0nw[kR, kTETA]
                                - Rs * H0sw[kR, kTETA]
                            )
                        )

                        if kTETA == 0 and kR == 0:
                            Mat_coef[k, k] = CP - CS - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + NTETA] = CN

                        if kTETA == 0 and kR > 0 and kR < NR-1:
                            Mat_coef[k, k] = CP - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + NTETA] = CN
                            Mat_coef[k, k - NTETA] = CS

                        if kTETA == 0 and kR == NR-1:
                            Mat_coef[k, k] = CP - CW - CN
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - NTETA] = CS

                        if kR == 0 and kTETA > 0 and kTETA < NTETA-1:
                            Mat_coef[k, k] = CP - CS
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + NTETA] = CN

                        if kTETA > 0 and kTETA < NTETA-1 and kR > 0 and kR < NR-1:
                            Mat_coef[k, k] = CP
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + NTETA] = CN
                            Mat_coef[k, k - NTETA] = CS
                            Mat_coef[k, k + 1] = CE

                        if kR == NR-1 and kTETA > 0 and kTETA < NTETA-1:
                            Mat_coef[k, k] = CP - CN
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - NTETA] = CS

                        if kR == 0 and kTETA == NTETA-1:
                            Mat_coef[k, k] = CP - CE - CS
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + NTETA] = CN

                        if kTETA == NTETA-1 and kR > 0 and kR < NR-1:
                            Mat_coef[k, k] = CP - CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - NTETA] = CS
                            Mat_coef[k, k + NTETA] = CN

                        if kTETA == NTETA-1 and kR == NR-1:
                            Mat_coef[k, k] = CP - CE - CN
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - NTETA] = CS

                        kTETA = kTETA + 1

                    kR = kR + 1
                    kTETA = 0

                # Pressure field solution
                p = np.linalg.solve(Mat_coef, b)

                cont = 0

                # pressure matrix
                for ii in range(0, NR):
                    for jj in range(0, NTETA):
                        
                        P0[ii, jj] = p[cont]
                        cont = cont + 1
              
                # pressure boundary conditions
                for ii in range(0, NR):
                    for jj in range(0, NTETA):
                        if P0[ii, jj] < 0:
                            P0[ii, jj] = 0


                PPdim[1:-1, 1:-1] = (r1 ** 2 * war * mi0 / h0 ** 2) * np.flipud(P0)
                
                
                # Pressure derivatives on the faces: dPdR dPdTETA dP2dR2 dP2dTETA2
                kR = 0
                kTETA = 0
                cont = 0  # this one here really is 0 and not -1

                for R in vec_R:
                    for TETA in vec_TETA:

                        if kTETA == 0 and kR == 0:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / dTETA
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR

                        if kTETA == 0 and kR > 0 and kR < NR-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / dTETA
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                dR
                            )

                        if kTETA == 0 and kR == NR-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / dTETA
                            dP0dR[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * dR)

                        if kR == 0 and kTETA > 0 and kTETA < NTETA-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / (dTETA)
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                dR
                            )

                        if kTETA > 0 and kTETA < NTETA-1 and kR > 0 and kR < NR-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / (dTETA)
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                dR
                            )

                        if kR == NR-1 and kTETA > 0 and kTETA < NTETA-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / (dTETA)
                            dP0dR[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * dR)

                        if kR == 0 and kTETA == NTETA-1:
                            dP0dTETA[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * dTETA)
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                dR
                            )

                        if kTETA == NTETA-1 and kR > 0 and kR < NR-1:
                            dP0dTETA[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * dTETA)
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                dR
                            )

                        if kTETA == NTETA-1 and kR == NR-1:
                            dP0dTETA[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * dTETA)
                            dP0dR[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * dR)

                        kTETA = kTETA + 1

                    kR = kR + 1
                    kTETA = 0

                # PRESSURE_THD =============================================================
                # ENDS HERE ================================================================

                kR = 0
                kTETA = 0

                # pressure vectorization index
                k = -1

                # volumes number
                nk = (NR) * (NTETA)

                # Coefficients Matrix
                Mat_coef = np.zeros((nk, nk))
                b = np.zeros((nk, 1))
                cont = -1

                for R in vec_R:
                    for TETA in vec_TETA:

                        cont = cont + 1
                        TETAe = TETA + 0.5 * dTETA
                        TETAw = TETA - 0.5 * dTETA
                        Rn = R + 0.5 * dR
                        Rs = R - 0.5 * dR

                        # Coefficients for solving the energy equation
                        aux_n = (
                            dTETA
                            / 12
                            * (
                                R
                                * H0[kR, kTETA] ** 3
                                / MI[kR, kTETA]
                                * dP0dR[kR, kTETA]
                            )
                        )
                        CN_1 = 0.5 * aux_n
                        CS_1 = -0.5 * aux_n
                        CP_1 = -(CS_1 + CN_1)
                        aux_e = (
                            dR
                            / (12 * teta0 ** 2)
                            * (
                                H0[kR, kTETA] ** 3
                                / (R * MI[kR, kTETA])
                                * dP0dTETA[kR, kTETA]
                            )
                            - dR / (2 * teta0) * H0[kR, kTETA] * R
                        )
                        CE_1 = 0 * aux_e
                        CW_1 = -1 * aux_e
                        CP_2 = -(CE_1 + CW_1)

                        # difusive terms - central differences
                        CN_2 = (
                            kt
                            / (rho * cp * war * r1 ** 2)
                            * (dTETA * Rn)
                            / (dR)
                            * H0[kR, kTETA]
                        )
                        CS_2 = (
                            kt
                            / (rho * cp * war * r1 ** 2)
                            * (dTETA * Rs)
                            / (dR)
                            * H0[kR, kTETA]
                        )
                        CP_3 = -(CN_2 + CS_2)
                        CE_2 = (
                            kt
                            / (rho * cp * war * r1 ** 2)
                            * dR
                            / (teta0 ** 2 * dTETA)
                            * H0[kR, kTETA]
                            / R
                        )
                        CW_2 = (
                            kt
                            / (rho * cp * war * r1 ** 2)
                            * dR
                            / (teta0 ** 2 * dTETA)
                            * H0[kR, kTETA]
                            / R
                        )
                        CP_4 = -(CE_2 + CW_2)

                        CW = CW_1 + CW_2
                        CS = CS_1 + CS_2
                        CN = CN_1 + CN_2
                        CE = CE_1 + CE_2
                        CP = CP_1 + CP_2 + CP_3 + CP_4

                        B_F = 0
                        B_G = 0
                        B_H = (
                            dR
                            * dTETA
                            / (12 * teta0 ** 2)
                            * (
                                H0[kR, kTETA] ** 3
                                / (MI[kR, kTETA] * R)
                                * dP0dTETA[kR, kTETA] ** 2
                            )
                        )
                        B_I = MI[kR, kTETA] * R ** 3 / (H0[kR, kTETA]) * dR * dTETA
                        B_J = (
                            dR
                            * dTETA
                            / 12
                            * (R * H0[kR, kTETA] ** 3 / MI[kR, kTETA])
                            * dP0dR[kR, kTETA] ** 2
                        )
                        B_K = (
                            dR
                            * dTETA
                            / (12 * teta0)
                            * (H0[kR, kTETA] ** 3 / R)
                            * dP0dTETA[kR, kTETA]
                        )
                        B_L = (
                            dR
                            * dTETA
                            / 60
                            * (H0[kR, kTETA] ** 5 / (MI[kR, kTETA] * R))
                            * dP0dR[kR, kTETA] ** 2
                        )
                        B_M = (
                            2
                            * dR
                            * dTETA
                            * (R * MI[kR, kTETA] / H0[kR, kTETA])
                            * (dHdT) ** 2
                        )
                        B_N = dR * dTETA / 3 * R * MI[kR, kTETA] * H0[kR, kTETA]
                        B_O = (
                            dR
                            * dTETA
                            / (120 * teta0 ** 2)
                            * (H0[kR, kTETA] ** 5 / (MI[kR, kTETA] * R ** 3))
                            * dP0dTETA[kR, kTETA] ** 2
                        )

                        # vectorization index
                        k = k + 1

                        b[k, 0] = (
                            -B_F
                            + (war * mi0 * r1 ** 2 / (rho * cp * h0 ** 2 * T0))
                            * (B_G - B_H - B_I - B_J)
                            + (mi0 * war / (rho * cp * T0))
                            * (B_K - B_L - B_M - B_N - B_O)
                        )

                        if kTETA == 0 and kR == 0:
                            Mat_coef[k, k] = CP + CS
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + NTETA] = CN
                            b[k, 0] = b[k, 0] - 1 * CW

                        if kTETA == 0 and kR > 0 and kR < NR-1:
                            Mat_coef[k, k] = CP
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + NTETA] = CN
                            Mat_coef[k, k - NTETA] = CS
                            b[k, 0] = b[k, 0] - 1 * CW

                        if kTETA == 0 and kR == NR-1:
                            Mat_coef[k, k] = CP + CN
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - NTETA] = CS
                            b[k, 0] = b[k, 0] - 1 * CW

                        if kR == 0 and kTETA > 0 and kTETA < NTETA-1:
                            Mat_coef[k, k] = CP + CS
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + NTETA] = CN

                        if kTETA > 0 and kTETA < NTETA-1 and kR > 0 and kR < NR-1:
                            Mat_coef[k, k] = CP
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + NTETA] = CN
                            Mat_coef[k, k - NTETA] = CS
                            Mat_coef[k, k + 1] = CE

                        if kR == NR-1 and kTETA > 0 and kTETA < NTETA-1:
                            Mat_coef[k, k] = CP + CN
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - NTETA] = CS

                        if kR == 0 and kTETA == NTETA-1:
                            Mat_coef[k, k] = CP + CE + CS
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + NTETA] = CN

                        if kTETA == NTETA-1 and kR > 0 and kR < NR-1:
                            Mat_coef[k, k] = CP + CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - NTETA] = CS
                            Mat_coef[k, k + NTETA] = CN

                        if kTETA == NTETA-1 and kR == NR-1:
                            Mat_coef[k, k] = CP + CN + CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - NTETA] = CS

                        kTETA = kTETA + 1

                    kR = kR + 1
                    kTETA = 0

                # Temperature field solution
                t = np.linalg.solve(Mat_coef, b)
                cont = -1

                # Temperature matrix
                for ii in range(0, NR):
                    for jj in range(0, NTETA):
                        cont = cont + 1
                        T_new[ii, jj] = t[cont]

                # viscosity field
                varMI=np.zeros((NR, NTETA))
                for ii in range(0, NR):
                    for jj in range(0, NTETA):
                        MI_new[ii, jj] = (
                             (1 / mi0)
                            * k1
                            * np.exp(k2 / (T0 * T_new[ii, jj] + k3))
                        )
                        varMI[ii, jj] = abs((MI_new[ii, jj] - MI[ii, jj]) / MI[ii, jj])

                T = T_new
                aux1=np.max(varMI)

            
            # RESULTING FORCE AND MOMENTUM: Equilibrium position

            # dimensional pressure
            Pdim = P0 * (r1 ** 2) * war * mi0 / (h0 ** 2)

            # RESULTING FORCE AND MOMENTUM: Equilibrium position
            XR = r1 * vec_R
            XTETA = teta0 * vec_TETA
            Xrp = rp * (np.ones((np.size(XR))))

            for ii in range(0, NTETA):
                Mxr[:, ii] = (Pdim[:, ii] * (np.transpose(XR) ** 2)) * np.sin(
                    XTETA[ii] - tetap
                )
                Myr[:, ii] = (
                    -Pdim[:, ii]
                    * np.transpose(XR)
                    * np.transpose(XR * np.cos(XTETA[ii] - tetap) - Xrp)
                )
                Frer[:, ii] = Pdim[:, ii] * np.transpose(XR)

            ######################################################################
            mxr = np.trapz( Mxr, XTETA)
            myr = np.trapz( Myr, XTETA)
            frer = np.trapz( Frer, XTETA)

            mx = np.trapz(mxr, XR)
            my = np.trapz( myr, XR)
            fre = -np.trapz( frer, XR) + fz / Npad 
            ######################################################################

            resMx = mx
            resMy = my
            resFre = fre

            # TEMPERATURE ==============================================================
            # ENDS HERE ================================================================

            Ti = T * T0
            ResFM = np.linalg.norm(np.array([resMx, resMy, resFre]))
            x0 = x

        # --------------------------------------------------------------------------
        # Full temperature field
        TT = np.ones((NR + 2, NTETA + 2))
        TT[1:NR+1, 1:NTETA+1] = np.flipud(Ti)
        TT[:, 0] = T0
        TT[0, :] = TT[1, :]
        TT[NR + 1, :] = TT[NR, :]
        TT[:, NTETA + 1] = TT[:, NTETA]
        TT = TT - 273.15

        # --------------------------------------------------------------------------
        # Viscosity field
        for ii in range(0, NR):
            for jj in range(0, NTETA):
                mi[ii, jj] =  k1 * np.exp(k2 / (Ti[ii, jj] + k3))  # [Pa.s]

        # PRESSURE =================================================================
        # STARTS HERE ==============================================================

        Ar = a_r * r1 / h0
        As = a_s * r1 / h0
        MI = 1 / mi0 * mi

        # PRESSURE FIELD - Solution of Reynolds equation
        kR = 0
        kTETA = 0

        # index using for pressure vectorization
        k = -1

        # number of volumes
        nk = (NR) * (NTETA)

        # Coefficients Matrix
        Mat_coef = np.zeros((nk, nk))
        b = np.zeros((nk, 1))
        cont = -1

        for R in vec_R:
            for TETA in vec_TETA:

                cont = cont + 1
                TETAe = TETA + 0.5 * dTETA
                TETAw = TETA - 0.5 * dTETA
                Rn = R + 0.5 * dR
                Rs = R - 0.5 * dR

                H0[kR, kTETA] = (
                    h0 / h0
                    + As * (Rp - R * np.cos(teta0 * (TETA - TETAp)))
                    + Ar * R * np.sin(teta0 * (TETA - TETAp))
                )
                H0ne[kR, kTETA] = (
                    h0 / h0
                    + As * (Rp - Rn * np.cos(teta0 * (TETAe - TETAp)))
                    + Ar * Rn * np.sin(teta0 * (TETAe - TETAp))
                )
                H0nw[kR, kTETA] = (
                    h0 / h0
                    + As * (Rp - Rn * np.cos(teta0 * (TETAw - TETAp)))
                    + Ar * Rn * np.sin(teta0 * (TETAw - TETAp))
                )
                H0se[kR, kTETA] = (
                    h0 / h0
                    + As * (Rp - Rs * np.cos(teta0 * (TETAe - TETAp)))
                    + Ar * Rs * np.sin(teta0 * (TETAe - TETAp))
                )
                H0sw[kR, kTETA] = (
                    h0 / h0
                    + As * (Rp - Rs * np.cos(teta0 * (TETAw - TETAp)))
                    + Ar * Rs * np.sin(teta0 * (TETAw - TETAp))
                )

                if kTETA == 0 and kR == 0:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA == 0 and kR > 0 and kR < NR-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kTETA == 0 and kR == NR-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == 0 and kTETA > 0 and kTETA < NTETA-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA > 0 and kTETA < NTETA-1 and kR > 0 and kR < NR-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == NR-1 and kTETA > 0 and kTETA < NTETA-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == 0 and kTETA == NTETA-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA == NTETA-1 and kR > 0 and kR < NR-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kTETA == NTETA-1 and kR == NR-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                # Coefficients for solving the Reynolds equation
                CE = (
                    1
                    / (24 * teta0 ** 2 * MI_e)
                    * (dR / dTETA)
                    * (H0ne[kR, kTETA] ** 3 / Rn + H0se[kR, kTETA] ** 3 / Rs)
                )
                CW = (
                    1
                    / (24 * teta0 ** 2 * MI_w)
                    * (dR / dTETA)
                    * (H0nw[kR, kTETA] ** 3 / Rn + H0sw[kR, kTETA] ** 3 / Rs)
                )
                CN = (
                    Rn
                    / (24 * MI_n)
                    * (dTETA / dR)
                    * (H0ne[kR, kTETA] ** 3 + H0nw[kR, kTETA] ** 3)
                )
                CS = (
                    Rs
                    / (24 * MI_s)
                    * (dTETA / dR)
                    * (H0se[kR, kTETA] ** 3 + H0sw[kR, kTETA] ** 3)
                )
                CP = -(CE + CW + CN + CS)

                # vectorization index
                k = k + 1

                b[k, 0] = (
                    dR
                    / (4 * teta0)
                    * (
                        Rn * H0ne[kR, kTETA]
                        + Rs * H0se[kR, kTETA]
                        - Rn * H0nw[kR, kTETA]
                        - Rs * H0sw[kR, kTETA]
                    )
                )

                if kTETA == 0 and kR == 0:
                    Mat_coef[k, k] = CP - CS - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == 0 and kR > 0 and kR < NR-1:
                    Mat_coef[k, k] = CP - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + NTETA] = CN
                    Mat_coef[k, k - NTETA] = CS

                if kTETA == 0 and kR == NR-1:
                    Mat_coef[k, k] = CP - CW - CN
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - NTETA] = CS

                if kR == 0 and kTETA > 0 and kTETA < NTETA-1:
                    Mat_coef[k, k] = CP - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN

                if kTETA > 0 and kTETA < NTETA-1 and kR > 0 and kR < NR-1:
                    Mat_coef[k, k] = CP
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN
                    Mat_coef[k, k - NTETA] = CS
                    Mat_coef[k, k + 1] = CE

                if kR == NR-1 and kTETA > 0 and kTETA < NTETA-1:
                    Mat_coef[k, k] = CP - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - NTETA] = CS

                if kR == 0 and kTETA == NTETA-1:
                    Mat_coef[k, k] = CP - CE - CS
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == NTETA-1 and kR > 0 and kR < NR-1:
                    Mat_coef[k, k] = CP - CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - NTETA] = CS
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == NTETA-1 and kR == NR-1:
                    Mat_coef[k, k] = CP - CE - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - NTETA] = CS

                kTETA = kTETA + 1

            kR = kR + 1
            kTETA = 0

        # Pressure field solution
        p = np.linalg.solve(Mat_coef, b)
        cont = -1

        # pressure matrix
        for ii in range(0, NR):
            for jj in range(0, NTETA):
                cont = cont + 1
                P0[ii, jj] = p[cont]

        # pressure boundary conditions
        for ii in range(0, NR):
            for jj in range(0, NTETA):
                if P0[ii, jj] < 0:
                    P0[ii, jj] = 0

        PPdim = np.zeros((NR + 2, NTETA + 2))
        PPdim[1:-1, 1:-1] = (r1 ** 2 * war * mi0 / h0 ** 2) * np.flipud(P0)
      
        # PRESSURE =================================================================
        # ENDS HERE ================================================================

        # --------------------------------------------------------------------------
        # Stiffness and Damping Coefficients
        wp = war  # perturbation frequency [rad/s]
        WP = wp / war

        # HYDROCOEFF_z =============================================================
        # STARTS HERE ==============================================================

        MI = (1 / mi0) * mi

        kR = 0
        kTETA = 0
        k = -1  # pressure vectorization index
        nk = (NR) * (NTETA)  # volumes number

        # coefficients matrix
        Mat_coef = np.zeros((nk, nk))
        b_coef = np.zeros((nk, 1),dtype=complex)
        p_coef = np.zeros((nk, 1),dtype=complex)
        P_coef = np.zeros((NR, NTETA),dtype=complex)
        P_dim_coef = np.zeros((NR, NTETA),dtype=complex)
        Mxr_coef = np.zeros((NR, NTETA),dtype=complex)
        Myr_coef = np.zeros((NR, NTETA),dtype=complex)
        Frer_coef = np.zeros((NR, NTETA),dtype=complex)

        cont = -1

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

                if kR == 0 and kTETA > 0 and kTETA < NTETA-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / dTETA
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = P0[kR, kTETA] / (0.5 * dR)

                if kTETA > 0 and kTETA < NTETA-1 and kR > 0 and kR < NR-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / dTETA
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / dR

                if kR == NR-1 and kTETA > 0 and kTETA < NTETA -1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / dTETA
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = -P0[kR, kTETA] / (0.5 * dR)
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / dR

                if kR == 0 and kTETA == NTETA-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]
                    dPdTETAe = -P0[kR, kTETA] / (0.5 * dTETA)
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = P0[kR, kTETA] / (0.5 * dR)

                if kTETA == NTETA-1 and kR > 0 and kR < NR-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = -P0[kR, kTETA] / (0.5 * dTETA)
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / dR
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / dR

                if kTETA == NTETA-1 and kR == NR-1:
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

                b_coef[k, 0] = -(B_1 + B_2) + B_3 + B_4

                if kTETA == 0 and kR == 0:
                    Mat_coef[k, k] = CP - CW - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == 0 and kR > 0 and kR < NR-1:
                    Mat_coef[k, k] = CP - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + NTETA] = CN
                    Mat_coef[k, k - NTETA] = CS

                if kTETA == 0 and kR == NR-1:
                    Mat_coef[k, k] = CP - CW - CN
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - NTETA] = CS

                if kR == 0 and kTETA > 0 and kTETA < NTETA-1:
                    Mat_coef[k, k] = CP - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN

                if kTETA > 0 and kTETA < NTETA-1 and kR > 0 and kR < NR-1:
                    Mat_coef[k, k] = CP
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN
                    Mat_coef[k, k - NTETA] = CS
                    Mat_coef[k, k + 1] = CE

                if kR == NR-1 and kTETA > 0 and kTETA < NTETA-1:
                    Mat_coef[k, k] = CP - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - NTETA] = CS

                if kR == 0 and kTETA == NTETA-1:
                    Mat_coef[k, k] = CP - CE - CS
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == NTETA-1 and kR > 0 and kR < NR-1:
                    Mat_coef[k, k] = CP - CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - NTETA] = CS
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == NTETA-1 and kR == NR-1:
                    Mat_coef[k, k] = CP - CE - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - NTETA] = CS

                kTETA = kTETA + 1

            kR = kR + 1
            kTETA = 0

        # vectorized pressure field solution
        p_coef = np.linalg.solve(Mat_coef, b_coef)
        cont = -1

        # pressure matrix
        for ii in range(0, NR):
            for jj in range(0, NTETA):
                cont = cont + 1
                P_coef[ii, jj] = p_coef[cont]

        # dimensional pressure
        Pdim_coef = P_coef * (r1 ** 2) * war * mi0 / (h0 ** 3)

        # RESULTING FORCE AND MOMENTUM: Equilibrium position
        XR = r1 * vec_R
        XTETA = teta0 * vec_TETA
        Xrp = rp * (np.ones((np.size(XR))))

        for ii in range(0, NTETA):
            Mxr_coef[:, ii] = (Pdim_coef[:, ii] * (np.transpose(XR) ** 2)) * np.sin(
                XTETA[ii] - tetap
            )
            Myr_coef[:, ii] = (
                -Pdim_coef[:, ii]
                * np.transpose(XR)
                * np.transpose(XR * np.cos(XTETA[ii] - tetap) - Xrp)
            )
            Frer_coef[:, ii] = Pdim_coef[:, ii] * np.transpose(XR)

        ######################################################################
        mxr_coef = np.trapz( Mxr_coef, XTETA)
        myr_coef = np.trapz( Myr_coef, XTETA)
        frer_coef = np.trapz( Frer_coef, XTETA)

        mx_coef = np.trapz(mxr_coef, XR)
        my_coef = np.trapz( myr_coef, XR)
        fre_coef = -np.trapz( frer_coef, XR) 
        ######################################################################

        # HYDROCOEFF_z =============================================================
        # ENDS HERE ================================================================

        K = Npad * np.real(fre_coef)  # Stiffness Coefficient
        C = Npad * 1 / wp * np.imag(fre_coef)  # Damping Coefficient

        # THERMO-ELASTO-HYDRODYNAMIC SOLUTION ======================================
        # STARTS HERE ==============================================================
        
        # Variable startup
        # DZ = np.zeros((NR, NTETA))
        # D_P = np.zeros((NR, NTETA))
        # D_T = np.zeros((NR, NTETA))
        # b_p = np.zeros((nk, 1))
        # Mat_coef = np.zeros((nk, nk))

        # Tback = T0 # Temperature in the bottom pad surface

        # D = (E_pad * tpad ** 3) / (12 * (1 - v_pad ** 2))

        # YM = 4/(3 * teta0) * np.sin(teta0/2) * (r2 ** 2 + r1 * r2 + r1 ** 2)/(r2 + r1)

        # kR=0
        # kTETA=0
        # k=-1

        # #PAD DEFORMATION: EFFECTS OF PRESSURE AND TEMPERATURE

        # for R in vec_R:
        #     for TETA in vec_TETA:

        #         yx = np.dot( 
        #             [
        #                 [-np.cos(teta0/2), -np.sin(teta0/2)],
        #                 [np.sin(teta0/2), -np.cos(teta0/2)],
        #             ],   
        #             [
        #                 [-R * r1 * np.sin(TETA * teta0)],
        #                 [R * r1 * np.cos(TETA * teta0)],
        #             ]
        #         )

        #         yx = yx +[[0],[YM]]

        #         Y=yx[0]
        #         X=yx[1]

        #         M_1 = tpad ** 2 * E_pad * alpha_pad / (12 * (1 - v_pad)) * (Ti[0 , kTETA] - Tback)
        #         M_2 = tpad ** 2 * E_pad * alpha_pad / (12 * (1 - v_pad)) * (Ti[kR , NTETA-1] - Tback)
        #         M_3 = tpad ** 2 * E_pad * alpha_pad / (12 * (1 - v_pad)) * (Ti[NR-1 , kTETA] - Tback)
        #         M_4 = tpad ** 2 * E_pad * alpha_pad / (12 * (1 - v_pad)) * (Ti[kR , 0] - Tback)

        #         DZ[kR , kTETA] = -1/(2 * D * (1+v_pad ** 2)) * (((M_1 + M_3) - v_pad * (M_2 + M_4)) * X ** 2 +
        #                         ((M_2 + M_4) - v_pad * (M_1 + M_3)) * Y ** 2)
                
        #         #Mechanical deformation
        #         AE = 1/(R * r1) ** 2 * (1/ (dTETA * teta0) ** 2)
        #         AW = 1/(R * r1) ** 2 * (1/ (dTETA * teta0) ** 2)
        #         AN = 1/(dR * r1) ** 2 + 1/(R * r1) * (1/(2 * dR * r1))
        #         AS = 1/(dR * r1) ** 2 - 1/(R * r1) * (1/(2 * dR * r1))
        #         AP = -(AE + AW + AN + AS)

        #         k = k + 1 #vectorization index

        #         b_p[k , 0] = tpad ** 2/(6 * (1 - v_pad)) * (r1 ** 2 * war * mi0 / h0 ** 2) * P0[kR , kTETA] / D

        #         if kTETA == 0 and kR == 0:
        #             Mat_coef[k, k] = AP - AW - AS
        #             Mat_coef[k, k + 1] = AE
        #             Mat_coef[k, k + NTETA] = AN

        #         if kTETA == 0 and kR > 0 and kR < NR-1:
        #             Mat_coef[k, k] = AP - AW
        #             Mat_coef[k, k + 1] = AE
        #             Mat_coef[k, k + NTETA] = AN
        #             Mat_coef[k, k - NTETA] = AS

        #         if kTETA == 0 and kR == NR-1:
        #             Mat_coef[k, k] = AP - AW - AN
        #             Mat_coef[k, k + 1] = AE
        #             Mat_coef[k, k - NTETA] = AS

        #         if kR == 0 and kTETA > 0 and kTETA < NTETA-1:
        #             Mat_coef[k, k] = AP - AS
        #             Mat_coef[k, k + 1] = AE
        #             Mat_coef[k, k - 1] = AW
        #             Mat_coef[k, k + NTETA] = AN

        #         if kTETA > 0 and kTETA < NTETA-1 and kR > 0 and kR < NR-1:
        #             Mat_coef[k, k] = AP
        #             Mat_coef[k, k - 1] = AW
        #             Mat_coef[k, k + NTETA] = AN
        #             Mat_coef[k, k - NTETA] = AS
        #             Mat_coef[k, k + 1] = AE

        #         if kR == NR-1 and kTETA > 0 and kTETA < NTETA-1:
        #             Mat_coef[k, k] = AP - AN
        #             Mat_coef[k, k - 1] = AW
        #             Mat_coef[k, k + 1] = AE
        #             Mat_coef[k, k - NTETA] = AS

        #         if kR == 0 and kTETA == NTETA-1:
        #             Mat_coef[k, k] = AP - AE - AS
        #             Mat_coef[k, k - 1] = AW
        #             Mat_coef[k, k + NTETA] = AN

        #         if kTETA == NTETA-1 and kR > 0 and kR < NR-1:
        #             Mat_coef[k, k] = AP - AE
        #             Mat_coef[k, k - 1] = AW
        #             Mat_coef[k, k - NTETA] = AS
        #             Mat_coef[k, k + NTETA] = AN

        #         if kTETA == NTETA-1 and kR == NR-1:
        #             Mat_coef[k, k] = AP - AE - AN
        #             Mat_coef[k, k - 1] = AW
        #             Mat_coef[k, k - NTETA] = AS

        #         kTETA = kTETA + 1

        #     kR = kR + 1
        #     kTETA = 0
        # # PAD DEFORMATION - PRESSURE EFFECTS 
        
        # d_p = np.linalg.solve(Mat_coef, b_p) # solve deformation vectorized

        # cont=-1

        # # deformation matrix
        # for ii in range(0, NR):
        #     for jj in range(0, NTETA):
        #         cont = cont + 1
        #         D_P[ii, jj] = d_p[cont]
        
        # # TOTAL PAD DEFORMATION 
        # D_T=DZ+D_P

        # THERMO-ELASTO-HYDRODYNAMIC SOLUTION ======================================
        # ENDS HERE ==============================================================

        # print(PPdim)

        # yraio1 = np.arange(r1 * np.sin((np.pi/2 - teta0/2),(np.pi/2 + teta0/2), dTETA * teta0))
        # xraio1 = np.arange(r1 * np.cos(np.pi/2 - teta0/2, np.pi/2 + teta0/2, dTETA * teta0))
        # yraio2 = np.arange(r2 * np.sin((np.pi/2 - teta0/2), (np.pi/2 + teta0/2), dTETA*teta0))
        # xraio2 = np.arange(r2 * np.cos(np.pi/2 - teta0/2, np.pi/2 +teta0/2, dTETA*teta0))

        # dx = (xraio1[0] - xraio2[0]/(NTETA - 2))
        # xreta1 = np.arange(xraio2[0], xraio1[0], dx)
        # yreta1 = np.arange(yraio2[0], yraio1[0], dx * np.tan(np.pi/2 -teta0/2))
        # xreta2 = np.arange(xraio2[-1], xraio1[-1], -dx)
        # yreta2 = np.arange(yraio2[0], yraio1[0], dx * np.tan(np.pi/2 - teta0/2))
        # np.savetxt('YH.txt', YH)
        # np.savetxt('XH.txt', XH)

        # fig1 = plt.figure(1)
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(XH[1:NR+1,1:NTETA+1], YH[1:NR+1,1:NTETA+1], 1e6*D_T, rstride=1, cstride=1, cmap='jet', edgecolor='none')
        # # plt.savefig("pressao.png", bbox_inches="tight", dpi=600)
        # plt.grid()
        # ax.set_title('Deflected middle plane')
        # ax.set_xlim([np.min(XH), np.max(XH)])
        # ax.set_ylim([np.min(YH), np.max(YH)])
        # ax.set_xlabel('x direction [m]')
        # ax.set_ylabel('y direction [m]')
        # ax.set_zlabel('Directional deformation [$\mu$m]')
        # plt.show()

        # fig2 = plt.figure(2)
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(XH[1:NR+1,1:NTETA+1], YH[1:NR+1,1:NTETA+1], 1e6*D_P, rstride=1, cstride=1, cmap='jet', edgecolor='none')
        # # plt.savefig("pressao.png", bbox_inches="tight", dpi=600)
        # plt.grid()
        # ax.set_title('Mechanical Deflection')
        # ax.set_xlim([np.min(XH), np.max(XH)])
        # ax.set_ylim([np.min(YH), np.max(YH)])
        # ax.set_xlabel('x direction [m]')
        # ax.set_ylabel('y direction [m]')
        # ax.set_zlabel('Directional deformation [$\mu$m]')
        # plt.show()

        # fig3 = plt.figure(3)
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(XH[1:NR+1,1:NTETA+1], YH[1:NR+1,1:NTETA+1], 1e6*DZ, rstride=1, cstride=1, cmap='jet', edgecolor='none')
        # # plt.savefig("pressao.png", bbox_inches="tight", dpi=600)
        # plt.grid()
        # ax.set_title('Thermal Deflection')
        # ax.set_xlim([np.min(XH), np.max(XH)])
        # ax.set_ylim([np.min(YH), np.max(YH)])
        # ax.set_xlabel('x direction [m]')
        # ax.set_ylabel('y direction [m]')
        # ax.set_zlabel('Directional deformation [$\mu$m]')
        # plt.show()

        # PLOT FIGURES ==============================================================
        # STARTS HERE ==============================================================

                # Define vectors and matrix
        yh = np.zeros((NR+2))
        auxtransf = np.zeros((NTETA+2))
        XH = np.zeros((NR+2,NTETA+2))
        YH = np.zeros((NR+2,NTETA+2))

        yh[0] = r2
        yh[-1] = r1
        yh[1:NR+1] = np.arange((r2 - 0.5 * dR * r1), r1, -(dR * r1))
   
        auxtransf[0] = np.pi/2 + teta0/2
        auxtransf[-1] = np.pi/2 - teta0/2
        auxtransf[1:NTETA+1] = np.arange(np.pi/2 + teta0/2 - (0.5 * dTETA * teta0), np.pi/2 - teta0/2, -dTETA * teta0)

        for ii in range(0, NR+2):
            for jj in range(0, NTETA+2):
                XH[ii, jj] = yh[ii] * np.cos(auxtransf[jj])
                YH[ii, jj] = yh[ii] * np.sin(auxtransf[jj])

        np.savetxt('XH.txt', np.c_[XH])
        np.savetxt('YH.txt', np.c_[YH])
        np.savetxt('PPdim.txt', np.c_[PPdim])
        np.savetxt('TT.txt', np.c_[TT])

        fig4 = plt.figure(4)
        ax = plt.axes(projection='3d')
        ax.plot_surface(XH, YH, 1e-6*PPdim, rstride=1, cstride=1, cmap='jet', edgecolor='none')
        # plt.savefig("pressao.png", bbox_inches="tight", dpi=600)
        # np.save(XH,"jeff_bearings/XH")
        plt.grid()
        ax.set_title('Pressure field')
        ax.set_xlim([np.min(XH), np.max(XH)])
        ax.set_ylim([np.min(YH), np.max(YH)])
        ax.set_xlabel('x direction [m]')
        ax.set_ylabel('y direction [m]')
        ax.set_zlabel('Pressure [MPa]')
        plt.show()

        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(XH, YH, TT, cmap='jet')
        plt.grid()
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Temperature field [°C]')
        ax.set_xlabel('x direction [m]')
        ax.set_ylabel('y direction [m]')
        plt.show()

        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(XH, YH, 1e-6*PPdim, cmap='jet')
        plt.grid()
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Pressure field [MPa]')
        ax.set_xlabel('x direction [m]')
        ax.set_ylabel('y direction [m]')
        plt.show()

        # PLOT FIGURES ==============================================================
        # ENDS HERE ==============================================================

        # --------------------------------------------------------------------------
        # Output values - Pmax [Pa]- hmax[m] - hmin[m] - h0[m]
        Pmax = np.max(PPdim)
        hmax = np.max(h0 * H0)
        hmin = np.min(h0 * H0)
        Tmax = np.max(TT)
        h0
        print(f'Pmax: ', Pmax)
        print(f'hmax: ', hmax)
        print(f'hmin: ', hmin)
        print(f'Tmax: ', Tmax)
        print(f'h0: ', h0)
        print(f'K:', K)
        print(f'C:', C)

#def ArAsh0Equilibrium(
#    r1,
#    rp,
#    teta0,
#    tetap,
#    mi0,
#    fz,
#    Npad,
#    NTETA,
#    NR,
#    war,
#    R1,
#    R2,
#    TETA1,
#    TETA2,
#    Rp,
#    dR,
#    dTETA,
#    k1,
#    k2,
#    k3,
#    TETAp,
#    Ti,
#    x0,
#):
def ArAsh0Equilibrium(x,*args):
    r1,rp,teta0,tetap,mi0,fz,Npad,NTETA,NR,war,R1,R2,TETA1,TETA2,Rp,dR,dTETA,k1,k2,k3,TETAp,Ti,x0=args

    """Calculates the equilibrium position of the bearing

    Parameters
    ----------
    a_r = x[0]  : pitch angle axis r [rad]
    a_s = x[1]  : pitch angle axis s [rad]
    h0 = x[2]   : oil film thickness at pivot [m]

    """

    # Variable startup
    MI = np.zeros((NR, NTETA))
    P = np.zeros((NR, NTETA))
    Mxr = np.zeros((NR, NTETA))
    Myr = np.zeros((NR, NTETA))
    Frer = np.zeros((NR, NTETA))

    # loop counters for ease of understanding
    vec_R = np.zeros(NR)
    vec_R[0] = R1 + 0.5 * dR

    vec_TETA = np.zeros(NTETA)
    vec_TETA[0] = TETA1 + 0.5 * dTETA


    for ii in range(1, NR):
        vec_R[ii] = vec_R[ii-1] + dR

    for jj in range(1, NTETA):
        vec_TETA[jj] = vec_TETA[jj-1] + dTETA

    # Pitch angles alpha_r and alpha_p and oil filme thickness at pivot h0
    a_r = x[0]  # [rad]
    a_s = x[1]  # [rad]
    h0 = x[2]  # [m]

    for ii in range(0, NR):
        for jj in range(0, NTETA):
            MI[ii, jj] = (
                1 / mi0 *  k1 * np.exp(k2 / (Ti[ii, jj] + k3))
            )  # dimensionless

    # Dimensioneless Parameters
    Ar = a_r * r1 / h0
    As = a_s * r1 / h0
    H0 = h0 / h0

    # PRESSURE FIELD - Solution of Reynolds equation
    kR = 0
    kTETA = 0

    # pressure vectorization index
    k = -1

    # number of volumes
    nk = (NR) * (NTETA)  # number of volumes

    # Coefficients Matrix
    Mat_coef = np.zeros((nk, nk))
    b = np.zeros((nk, 1))
    cont = -1

    for R in vec_R:
        for TETA in vec_TETA:

            cont = cont + 1
            TETAe = TETA + 0.5 * dTETA
            TETAw = TETA - 0.5 * dTETA
            Rn = R + 0.5 * dR
            Rs = R - 0.5 * dR

            Hne = (
                H0
                + As * (Rp - Rn * np.cos(teta0 * (TETAe - TETAp)))
                + Ar * Rn * np.sin(teta0 * (TETAe - TETAp))
            )
            Hnw = (
                H0
                + As * (Rp - Rn * np.cos(teta0 * (TETAw - TETAp)))
                + Ar * Rn * np.sin(teta0 * (TETAw - TETAp))
            )
            Hse = (
                H0
                + As * (Rp - Rs * np.cos(teta0 * (TETAe - TETAp)))
                + Ar * Rs * np.sin(teta0 * (TETAe - TETAp))
            )
            Hsw = (
                H0
                + As * (Rp - Rs * np.cos(teta0 * (TETAw - TETAp)))
                + Ar * Rs * np.sin(teta0 * (TETAw - TETAp))
            )

            if kTETA == 0 and kR == 0:
                MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                MI_w = MI[kR, kTETA]
                MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                MI_s = MI[kR, kTETA]

            if kTETA == 0 and kR > 0 and kR < NR-1:
                MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                MI_w = MI[kR, kTETA]
                MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

            if kTETA == 0 and kR == NR-1:
                MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                MI_w = MI[kR, kTETA]
                MI_n = MI[kR, kTETA]
                MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

            if kR == 0 and kTETA > 0 and kTETA < NTETA - 1:
                MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                MI_s = MI[kR, kTETA]

            if kTETA > 0 and kTETA < NTETA - 1 and kR > 0 and kR < NR-1:
                MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

            if kR == NR-1 and kTETA > 0 and kTETA < NTETA - 1:
                MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                MI_n = MI[kR, kTETA]
                MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

            if kR == 0 and kTETA == NTETA - 1:
                MI_e = MI[kR, kTETA]
                MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                MI_s = MI[kR, kTETA]

            if kTETA == NTETA - 1 and kR > 0 and kR < NR-1:
                MI_e = MI[kR, kTETA]
                MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

            if kTETA == NTETA - 1 and kR == NR-1:
                MI_e = MI[kR, kTETA]
                MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                MI_n = MI[kR, kTETA]
                MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

            # Coefficients for solving the Reynolds equation
            CE = (
                1
                / (24 * teta0 ** 2 * MI_e)
                * (dR / dTETA)
                * (Hne ** 3 / Rn + Hse ** 3 / Rs)
            )
            CW = (
                1
                / (24 * teta0 ** 2 * MI_w)
                * (dR / dTETA)
                * (Hnw ** 3 / Rn + Hsw ** 3 / Rs)
            )
            CN = Rn / (24 * MI_n) * (dTETA / dR) * (Hne ** 3 + Hnw ** 3)
            CS = Rs / (24 * MI_s) * (dTETA / dR) * (Hse ** 3 + Hsw ** 3)
            CP = -(CE + CW + CN + CS)

            # vectorization index
            k = k + 1

            b[k, 0] = dR / (4 * teta0) * (Rn * Hne + Rs * Hse - Rn * Hnw - Rs * Hsw)

            if kTETA == 0 and kR == 0:
                Mat_coef[k, k] = CP - CS - CW
                Mat_coef[k, k + 1] = CE
                Mat_coef[k, k + NTETA ] = CN

            if kTETA == 0 and kR > 0 and kR < NR-1:
                Mat_coef[k, k] = CP - CW
                Mat_coef[k, k + 1] = CE
                Mat_coef[k, k + NTETA ] = CN
                Mat_coef[k, k - NTETA ] = CS

            if kTETA == 0 and kR == NR-1:
                Mat_coef[k, k] = CP - CW - CN
                Mat_coef[k, k + 1] = CE
                Mat_coef[k, k - NTETA ] = CS

            if kR == 0 and kTETA > 0 and kTETA < NTETA - 1:
                Mat_coef[k, k] = CP - CS
                Mat_coef[k, k + 1] = CE
                Mat_coef[k, k - 1] = CW
                Mat_coef[k, k + NTETA ] = CN

            if kTETA > 0 and kTETA < NTETA - 1 and kR > 0 and kR < NR-1:
                Mat_coef[k, k] = CP
                Mat_coef[k, k - 1] = CW
                Mat_coef[k, k + NTETA ] = CN
                Mat_coef[k, k - NTETA ] = CS
                Mat_coef[k, k + 1] = CE

            if kR == NR-1 and kTETA > 0 and kTETA < NTETA - 1:
                Mat_coef[k, k] = CP - CN
                Mat_coef[k, k - 1] = CW
                Mat_coef[k, k + 1] = CE
                Mat_coef[k, k - NTETA ] = CS

            if kR == 0 and kTETA == NTETA - 1:
                Mat_coef[k, k] = CP - CE - CS
                Mat_coef[k, k - 1] = CW
                Mat_coef[k, k + NTETA ] = CN

            if kTETA == NTETA - 1 and kR > 0 and kR < NR-1:
                Mat_coef[k, k] = CP - CE
                Mat_coef[k, k - 1] = CW
                Mat_coef[k, k - NTETA ] = CS
                Mat_coef[k, k + NTETA ] = CN

            if kTETA == NTETA - 1 and kR == NR-1:
                Mat_coef[k, k] = CP - CE - CN
                Mat_coef[k, k - 1] = CW
                Mat_coef[k, k - NTETA ] = CS

            kTETA = kTETA + 1

        kR = kR + 1
        kTETA = 0


    p = np.linalg.solve(Mat_coef, b)
    cont = -1

    # pressure matrix
    for ii in range(0, NR):
        for jj in range(0, NTETA):
            cont = cont + 1
            P[ii, jj] = p[cont]

    # boundary conditions of pressure
    for ii in range(0, NR):
        for jj in range(0, NTETA):
            if P[ii, jj] < 0:
                P[ii, jj] = 0

    # dimensional pressure
    Pdim = P * (r1 ** 2) * war * mi0 / (h0 ** 2)

    # RESULTING FORCE AND MOMENTUM: Equilibrium position
    XR = r1 * vec_R
    XTETA = teta0 * vec_TETA
    Xrp = rp * (np.ones((np.size(XR))))

    for ii in range(0, NTETA):
        Mxr[:, ii] = (Pdim[:, ii] * (np.transpose(XR) ** 2)) * np.sin(
            XTETA[ii] - tetap)
        Myr[:, ii] = (
            -Pdim[:, ii]
            * np.transpose(XR)
            * np.transpose(XR * np.cos(XTETA[ii] - tetap) - Xrp)
        )
        Frer[:, ii] = Pdim[:, ii] * np.transpose(XR)

    ######################################################################
    mxr = np.trapz( Mxr, XTETA)
    myr = np.trapz( Myr, XTETA)
    frer = np.trapz( Frer, XTETA)

    mx = np.trapz(mxr, XR)
    my = np.trapz( myr, XR)
    fre = -np.trapz( frer, XR) + fz / Npad 
    ######################################################################

    score = np.linalg.norm([mx, my, fre])
    # print(x0)
    # print(f'Score: ', score)
    # print('============================================')
    # print(f'mx: ', mx)
    # print('============================================')
    # print(f'my: ', my)
    # print('============================================')
    # print(f'fre: ', fre)
    # print('')
    return score
""""
def hydroplots (r1, r2, dR, dTETA, teta0, NR, NTETA):
    
    # Define vectors and matrix
    yh = np.zeros((1, NR+2))
    auxtransf = np.zeros((1, NTETA+2))
    XH = np.zeros((NR+2,NTETA+2))
    YH = np.zeros((NR+2,NTETA+2))

    yh[0] = r2
    yh[NR+1] = r1
    yh[1:NR] = ((r2 - 0.5 * dR * r1), -(dR * r1),(r1 + 0.5 * dR * r1))
   
    auxtransf[0] = np.pi/2 + teta0/2
    auxtransf[NTETA+1] = np.pi/2 - teta0/2
    auxtransf[1:NTETA +1] = np.pi/2 + teta0/2 - (0.5 * dTETA * teta0), -dTETA * teta0, np.pi/2 - teta0/2 + (0.5 * dTETA * teta0)

    for ii in range(0, NR+1):
        for jj in range(0, NTETA+1):
            XH[ii, jj] = yh[ii] * np.cos(auxtransf[jj])
            YH[ii, jj] = yh[ii] * np.sin(auxtransf[jj])

    yraio1 = r1 * np.sin((np.pi/2 - teta0/2), dTETA * teta0, (np.pi/2 + teta0/2))
    xraio1 = r1 * np.cos(np.pi/2 - teta0/2, dTETA * teta0, np.pi/2 + teta0/2)
    yraio2 = r2 * np.sin((np.pi/2 - teta0/2), dTETA*teta0, (np.pi/2 + teta0/2))
    xraio2 = r2 * np.cos(np.pi/2 - teta0/2, dTETA*teta0, np.pi/2 +teta0/2)

    dx = (xraio1[0] - xraio2[0]/(NTETA - 2))
    xreta1 = xraio2[0], dx, xraio1[0]
    yreta1 = yraio2[0], dx * np.tan(np.pi/2 -teta0/2), yraio1[0]
    xreta2 = xraio2[-1], -dx, xraio1[-1]
    yreta2 = yraio2[0], dx * np.tan(np.pi/2 - teta0/2), yraio1[0]
"""
def thrust_bearing_example():
    """Create an example of a thrust bearing with Thermo-Hydro-Dynamic effects.
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

    r1 = 0.5 * 2300e-3  # pad inner radius [m]
    r2 = 0.5 * 3450e-3  # pad outer radius [m]
    rp = 0.5 * 2885e-3   # pad pivot radius [m]
    teta0 = 26 * np.pi / 180  # pad complete angle [rad]
    tetap = 15 * np.pi / 180  # pad pivot angle [rad]
    Tcub = 40 + 273.15 #oil bath temperature [K]
    TC = Tcub  # Collar temperature [K]
    Tin = Tcub  # Cold oil temperature [K]
    T0 = 0.5 * (TC + Tin)  # Reference temperature [K]
    rho = 867  # Oil density [kg/m³]
    cp = 1950  # Oil thermal capacity [J/kg/K]
    kt = 0.13  # Oil thermal conductivity [W/m/K]
    k1 = 1.068e-7  # Coefficient for ISO VG 32 turbine oil - Vogel's equation
    k2 = 4368  # Coefficient for ISO VG 32 turbine oil - Vogel's equation
    k3 = 0.1187  # Coefficient for ISO VG 32 turbine oil - Vogel's equation
    mi0 = k1*np.exp(k2/(T0+k3)) # Oil VG 22
    fz = 13500e3  # Loading in Y direction [N]
    Npad = 12  # Number of PADs
    # NTETA = 40  # TETA direction N volumes
    # NR = 40  # R direction N volumes
    NTETA = 10  # TETA direction N volumes
    NR = 10  # R direction N volumes
    wa = 90  # Shaft rotation speed [rads]
    war = (wa * np.pi) / 30  # Shaft rotation speed [RPM]
    R1 = 1  # Inner pad FEM radius
    R2 = r2 / r1  # Outer pad FEM radius
    TETA1 = 0  # Initial angular coordinate
    TETA2 = 1  # Final angular coordinate
    Rp = rp / r1  # Radial pivot position
    TETAp = tetap / teta0  # Angular pivot position
    dR = (R2 - R1) / (NR)  # R direction volumes length
    dTETA = (TETA2 - TETA1) / (NTETA)  # TETA direction volumes length
    Ti = T0 * (np.ones((NR, NTETA)))  # Initial temperature field [°C]
    # E_pad = 2e11 # Young Modulus of the pad [N/m^2]
    # tpad = 180e-3 # Pad thickness [m]
    # v_pad = 0.3 # Poisson's ratio
    # alpha_pad = 1.2e-5 # Thermal expansion [K^-1]
    x0 = np.array(
        (-0.000274792355106384, -1.69258824831925e-05, 0.000191418606538599)		
    )  # Initial equilibrium position
    # mi =
    # P0 =
    # MI =
    # H0 =
    # H0ne =
    # H0se =
    # H0nw =
    # H0sw =

    bearing = Thrust(
        r1=r1,
        r2=r2,
        rp=rp,
        teta0=teta0,
        tetap=tetap,
        Tcub=Tcub,
        TC=TC,
        Tin=Tin,
        T0=T0,
        rho=rho,
        cp=cp,
        kt=kt,
        k1=k1,
        k2=k2,
        k3=k3,
        mi0=mi0,
        fz=fz,
        Npad=Npad,
        NTETA=NTETA,
        NR=NR,
        wa=wa,
        war=war,
        R1=R1,
        R2=R2,
        TETA1=TETA1,
        TETA2=TETA2,
        Rp=Rp,
        TETAp=TETAp,
        dR=dR,
        dTETA=dTETA,
        Ti=Ti,
        x0=x0,
        # E_pad=E_pad,
        # tpad=tpad,
        # v_pad=v_pad,
        # alpha_pad=alpha_pad,
        # mi=mi,
        # P0=P0,
        # MI=MI,
        # H0=H0,
        # H0ne=H0ne,
        # H0se=H0se,
        # H0nw=H0nw,
        # H0sw=H0sw,
        #plot
    )

    return bearing

if __name__ == "__main__":
    thrust_bearing_example()