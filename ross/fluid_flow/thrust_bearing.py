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

        [P0, H0, H0ne, H0nw, H0se, H0sw] = PRESSURE(a_r, a_s, h0, mi)

        # --------------------------------------------------------------------------
        # Stiffness and Damping Coefficients
        wp = war  # perturbation frequency [rad/s]
        WP = wp / war
        [kk_zz, kk_arz, kk_asz] = HYDROCOEFF_z(P0, H0ne, H0nw, H0se, H0sw, mi, WP, h0)

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
