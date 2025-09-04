import numpy as np
import scipy


# import tensorflow as tf
import mpmath as fp

from numpy.linalg import pinv
from scipy.linalg import solve
from scipy.optimize import fmin
from scipy.interpolate import griddata
from decimal import Decimal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import plotly.graph_objects as go

import plotly.io as pio
pio.templates.default = "plotly_white"

# pio.kaleido.scope.mathjax = None


class THDThrust:
    """ This class calculates the pressure and temperature fields, equilibrium
    position of a tilting-pad thrust bearing. It is also possible to obtain the
    stiffness and damping coefficients.
    
    Parameters
    ----------
    Bearing Geometry
    ^^^^^^^^^^^^^^^^
    Describes the geometric characteristics.
    pad_inner_radius : float
        Inner pad radius. Default unit is meter.
    pad_outer_radius : float
        Outer pad radius. Default unit is meter.
    pad_pivot_radius : float
        Pivot pad radius. Default unit is meter.
    pad_arc_length : float
          Arc length of each pad. The unit is degree.
    angular_pivot_position : float
          Angular pivot position. The unit is degree.
    n_pad : integer
         Number of pads
    
    Operating conditions
    ^^^^^^^^^^^^^^^^^^^^
    Describes the operating conditions of the bearing
    frequency : float
        Rotor rotating frequency. Default unit is rad/s
    fz : Float
        Axial load. The unit is Newton.
    oil_supply_temperature : Float
        Oil bath temperature. The unit is °C
    x0  : array
        Initial Equilibrium Position
    
    choice_CAIMP : string
        Choose the operating condition that bearing is operating.
        - "calc_h0"
        - "impos_h0"
    
    if calc_h0
        x0[2] is h0 -> optimization of a_r and a_s
    
    if impos_h0
        x0 is overall initial values -> optimization of h0, a_r and a_s
    
    Fluid properties
    ^^^^^^^^^^^^^^^^
    Describes the fluid characteristics.
    
    lubricant : str
        Lubricant type. Can be:
        - 'ISOVG32'
        - 'ISOVG46'
        - 'ISOVG68'
        With it, we get:
        rho :  Fluid specific mass. Default unit is kg/m^3.
        kt  :  Fluid thermal conductivity. The unit is J/(s*m*°C).
        cp  :  Fluid specific heat. The unit is J/(kg*°C).
        
        
    k1, k2, and k3 : float
        Oil coefficients for the viscosity interpolation.

    Mesh discretization
    ^^^^^^^^^^^^^^^^^^^
    Describes the discretization of the fluid film
    n_radial : int
        Number of volumes along the R direction.
    n_theta : int
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
    dPdR : array 

    References
    ----------
    .. [1] BARBOSA, J.S. Analise de Modelos Termohidrodinamicos para Mancais de unidades geradoras Francis. 2016. Dissertacao de Mestrado. Universidade Federal de Uberlandia, Uberlandia. ..
    .. [2] HEINRICHSON, N.; SANTOS, I. F.; FUERST, A., The Influence of Injection Pockets on the Performance of Tilting Pad Thrust Bearings Part I Theory. Journal of Tribology, 2007. .. 
    .. [3] NICOLETTI, R., Efeitos Termicos em Mancais Segmentados Hibridos Teoria e Experimento. 1999. Dissertacao de Mestrado. Universidade Estadual de Campinas, Campinas. ..
    .. [4] LUND, J. W.; THOMSEN, K. K. A calculation method and data for the dynamic coefficients of oil lubricated journal bearings. Topics in fluid film bearing and rotor bearing system design and optimization, n. 1000118, 1978. ..
    Attributes
    ----------
    """

    def __init__(
        self,
        pad_inner_radius,
        pad_outer_radius,
        pad_pivot_radius,
        pad_arc_length,
        angular_pivot_position,
        oil_supply_temperature,
        lubricant,
        n_pad,
        n_theta,
        n_radial,
        frequency,
        # Coefs_D,
        # choice_CAIMP,
        equilibrium_position_mode,
        fzs_load,
        radial_inclination_angle,
        circumferential_inclination_angle,
        initial_film_thickness,
        print_result=False,
        print_progress=False,
        print_time=False,
    ):
        self.print_result = print_result
        self.print_progress = print_progress
        self.print_time = print_time

        self.pad_inner_radius = pad_inner_radius
        self.pad_outer_radius = pad_outer_radius
        self.pad_pivot_radius = pad_pivot_radius
        self.frequency = frequency * (np.pi / 30)
        self.pad_arc_length = pad_arc_length * np.pi / 180
        self.angular_pivot_position = angular_pivot_position * np.pi / 180
        self.oil_supply_temperature = oil_supply_temperature
        self.reference_temperature = oil_supply_temperature
        self.lubricant = lubricant
        self.n_pad = n_pad
        self.n_theta = n_theta
        self.n_radial = n_radial
        R1 = 1
        self.R1 = R1
        self.R2 = pad_outer_radius / pad_inner_radius
        TETA1 = 0
        TETA2 = 1
        self.TETA1 = TETA1
        self.TETA2 = TETA2
        self.Rp = pad_pivot_radius / pad_inner_radius
        self.TETAp = angular_pivot_position / pad_arc_length 
        self.dR = (self.R2 - self.R1) / (self.n_radial)
        self.dTETA = (TETA2 - TETA1) / (n_theta)
        # self.Ti = self.reference_temperature * (np.ones((self.n_radial, self.n_theta)))
        self.Ti = np.full((self.n_radial, self.n_theta), self.reference_temperature)

        # self.choice_CAIMP = choice_CAIMP
        # self.op_key = [*choice_CAIMP][0]
        # self.initial_position = choice_CAIMP[self.op_key]['init_guess']
        # self.Coefs_D = Coefs_D

        self.equilibrium_position_mode = equilibrium_position_mode
        self.fzs_load = fzs_load
        self.radial_inclination_angle = radial_inclination_angle
        self.circumferential_inclination_angle = circumferential_inclination_angle
        self.initial_film_thickness = initial_film_thickness
        self.initial_position = np.array([radial_inclination_angle, circumferential_inclination_angle, initial_film_thickness])

        # --------------------------------------------------------------------------
        
        # Interpolation coefficients
        lubricant_properties = self.lub_selector()
        T_muI = lubricant_properties["temp1"] 
        T_muF = lubricant_properties["temp2"] 
        mu_I = lubricant_properties["viscosity1"]
        mu_F = lubricant_properties["viscosity2"]
        self.rho = lubricant_properties["lube_density"]
        self.cp = lubricant_properties["lube_cp"]
        self.kt = lubricant_properties["lube_conduct"]

        self.b_b = np.log(mu_I/mu_F)*1/(T_muI-T_muF)
        self.a_a = mu_I/(np.exp(T_muI*self.b_b))

        self.reference_viscosity = self.a_a*np.exp(self.reference_temperature*self.b_b) #reference viscosity
        
        # Pre-processing loop counters for ease of understanding
        vec_R = np.zeros(n_radial)
        vec_R[0] = R1 + 0.5 * self.dR

        vec_TETA = np.zeros(n_theta)
        vec_TETA[0] = TETA1 + 0.5 * self.dTETA


        for ii in range(1, n_radial):
            vec_R[ii] = vec_R[ii-1] + self.dR
        self.vec_R = vec_R

        for jj in range(1, n_theta):
            vec_TETA[jj] = vec_TETA[jj-1] + self.dTETA
        self.vec_TETA = vec_TETA

        fp.mp.dps = 800  # numerical solver precision setting

        # Call the run method
        self.run()
        

    def run(self):
        
        """This method runs the calculation of the pressure and temperature field
        in oil film of a tilting pad thrust bearing. It is also possible to obtain
        the stiffness and damping coefficients. The inputs are the operation
        conditions (see documentation)
        """
        
        # self.frequency = frequency * (np.pi / 30)
        # self.choice_CAIMP = choice_CAIMP

        # self.op_key = [*choice_CAIMP][0]
        # self.initial_position = choice_CAIMP[self.op_key]['init_guess']

        H0, H0ne, H0nw, H0se, H0sw, h0, P0, mi = self.PandT_solution()

        # if "calc_h0" in self.op_key:
        # Print results based on equilibrium position mode
        if self.print_result:
            print(f"Pmax: ", self.PPdim.max())
            print(f"hmax: ", self.hmax)
            print(f"hmin: ", self.hmin)
            print(f"Tmax: ", self.TT.max())
            
            if self.equilibrium_position_mode == "calculate":
                print(f"h0: ", h0)
            elif self.equilibrium_position_mode == "imposed":
                print(f"fz: ", self.fzs_load.sum())
        
        # Calculate dynamic coefficients
        self.Coef_din(H0ne, H0nw, H0se, H0sw, h0, P0, mi)
        
        # Print dynamic coefficients if requested
        if self.print_result:
            print(f"K:", self.K)
            print(f"C:", self.C)
            # self.plot_results()                    

    def PandT_solution(self):
        # --------------------------------------------------------------------------
        # WHILE LOOP INITIALIZATION
        ResFM = 10
        tolFM = 1
        # tolFM = 1e-8

        iteration = 0
        while ResFM >= tolFM:
            iteration += 1
            if self.print_progress:
                print(f"Iteration {iteration} - Residual F&M: {ResFM:.6f}")
            # --------------------------------------------------------------------------
            # Equilibrium position optimization [ar,ap,h0]
            
            # if "impos_h0" in self.op_key:
            if self.equilibrium_position_mode == "imposed":
                # self.h0i = self.choice_CAIMP["impos_h0"]['h0']
                self.h0i = self.initial_position[2]
                x = scipy.optimize.fmin(
                    self.ArAsh0Equilibrium,                  
                    self.initial_position,
                    xtol=tolFM,
                    ftol=tolFM,
                    maxiter=100000,
                    maxfun=100000,
                    full_output=0,
                    disp=self.print_progress,
                    retall=0,
                    callback=None,
                    initial_simplex=None,
                         )
                
                a_r = x[0]  # [rad]
                a_s = x[1]  # [rad]
                h0 = self.h0i

            else:
                # self.fzs_load = self.choice_CAIMP["calc_h0"]["load"]
                # self.fzs_load = self.fzs_load

                # x = scipy.optimize.fmin(
                #     self.ArAsh0Equilibrium,                  
                #     self.initial_position,
                #     xtol=tolFM,
                #     ftol=tolFM,
                #     maxiter=100000,
                #     maxfun=100000,
                #     full_output=0,
                #     disp=self.print_progress,
                #     retall=0,
                #     callback=None,
                #     initial_simplex=None,
                #          )
                x = scipy.optimize.fmin(
                    self.ArAsh0Equilibrium,                  
                    self.initial_position,
                    xtol = 0.1,
                    ftol = 0.1,
                    maxiter = 100,
                    disp = False,
                         )
                
                a_r = x[0]  # [rad]
                a_s = x[1]  # [rad]
                h0 = x[2]

            # --------------------------------------------------------------------------
            #  Temperature field
            self.h0 = h0 
            # tolMI = 1e-6
            tolMI = 1e-5

            # TEMPERATURE ==============================================================
            # STARTS HERE ==============================================================

            dHdT = 0
            mi_i = np.zeros((self.n_radial, self.n_theta))

            # initial temperature field
            T_i = self.Ti

            for ii in range(0, self.n_radial):
                for jj in range(0, self.n_theta):
                    mi_i[ii, jj] = (
                        self.a_a * np.exp(self.b_b * T_i[ii, jj])
                    )  # [Pa.s]

            MI_new = (1 / self.reference_viscosity) * mi_i
            MI = 0.2 * MI_new

            # TEMPERATURE FIELD - Solution of ENERGY equation
            for ii in range(0, self.n_radial):
                for jj in range(0, self.n_theta):
                    varMI = np.abs((MI_new[ii, jj] - MI[ii, jj]) / MI[ii, jj])
            aux1=1
            
            while aux1 >= tolMI:

                MI = np.array(MI_new)

                # PRESSURE_THD =============================================================
                # STARTS HERE ==============================================================

                Ar = a_r * self.pad_inner_radius / h0
                As = a_s * self.pad_inner_radius / h0

                # volumes number
                nk = (self.n_radial) * (self.n_theta)

                # Variable initialization
                Mat_coef = np.zeros((nk, nk))
                b = np.zeros((nk, 1))
                H0 = np.zeros((self.n_radial, self.n_theta))
                H0ne = np.zeros((self.n_radial, self.n_theta))
                H0nw = np.zeros((self.n_radial, self.n_theta))
                H0se = np.zeros((self.n_radial, self.n_theta))
                H0sw = np.zeros((self.n_radial, self.n_theta))
                dP0dR = np.zeros((self.n_radial, self.n_theta))
                dP0dTETA = np.zeros((self.n_radial, self.n_theta))
                T_new = np.zeros((self.n_radial, self.n_theta))
                Mxr = np.zeros((self.n_radial, self.n_theta))
                Myr = np.zeros((self.n_radial, self.n_theta))
                Frer = np.zeros((self.n_radial, self.n_theta))
                P0 = np.ones((self.n_radial, self.n_theta))
                P = np.zeros((self.n_radial, self.n_theta))
                mi = np.zeros((self.n_radial, self.n_theta))

                PPdim = np.zeros((self.n_radial + 2, self.n_theta + 2))

                P0 = self.P_solution(H0, H0ne, H0nw, H0se, H0sw, h0, As, Ar, MI,  Mat_coef, b, P0)
                PPdim[1:-1, 1:-1] = (self.pad_inner_radius ** 2 * self.frequency * self.reference_viscosity / h0 ** 2) * np.flipud(P0)

                # PRESSURE_THD =============================================================
                # ENDS HERE ================================================================

                kR = 0
                kTETA = 0

                # pressure vectorization index
                k = -1

                # volumes number
                nk = (self.n_radial) * (self.n_theta)

                # Coefficients Matrix
                Mat_coef = np.zeros((nk, nk))
                b = np.zeros((nk, 1))

                for R in self.vec_R:
                    for TETA in self.vec_TETA:
                        
                        # Pressure derivatives on the faces: dPdR dPdTETA dP2dR2 dP2dTETA2
                        if kTETA == 0 and kR == 0:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / self.dTETA
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / self.dR

                        if kTETA == 0 and kR > 0 and kR < self.n_radial-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / self.dTETA
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                self.dR
                            )

                        if kTETA == 0 and kR == self.n_radial-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / self.dTETA
                            dP0dR[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * self.dR)

                        if kR == 0 and kTETA > 0 and kTETA < self.n_theta-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / (self.dTETA)
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                self.dR
                            )

                        if kTETA > 0 and kTETA < self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / (self.dTETA)
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                self.dR
                            )

                        if kR == self.n_radial-1 and kTETA > 0 and kTETA < self.n_theta-1:
                            dP0dTETA[kR, kTETA] = (
                                P0[kR, kTETA + 1] - P0[kR, kTETA]
                            ) / (self.dTETA)
                            dP0dR[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * self.dR)

                        if kR == 0 and kTETA == self.n_theta-1:
                            dP0dTETA[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * self.dTETA)
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                self.dR
                            )

                        if kTETA == self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                            dP0dTETA[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * self.dTETA)
                            dP0dR[kR, kTETA] = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / (
                                self.dR
                            )

                        if kTETA == self.n_theta-1 and kR == self.n_radial-1:
                            dP0dTETA[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * self.dTETA)
                            dP0dR[kR, kTETA] = (0 - P0[kR, kTETA]) / (0.5 * self.dR)

                        TETAe = TETA + 0.5 * self.dTETA
                        TETAw = TETA - 0.5 * self.dTETA
                        Rn = R + 0.5 * self.dR
                        Rs = R - 0.5 * self.dR

                        # Coefficients for solving the energy equation
                        aux_n = (
                            self.dTETA
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
                            self.dR
                            / (12 * self.pad_arc_length ** 2)
                            * (
                                H0[kR, kTETA] ** 3
                                / (R * MI[kR, kTETA])
                                * dP0dTETA[kR, kTETA]
                            )
                            - self.dR / (2 * self.pad_arc_length) * H0[kR, kTETA] * R
                        )
                        CE_1 = 0 * aux_e
                        CW_1 = -1 * aux_e
                        CP_2 = -(CE_1 + CW_1)

                        # difusive terms - central differences
                        CN_2 = (
                            self.kt
                            / (self.rho * self.cp * self.frequency * self.pad_inner_radius ** 2)
                            * (self.dTETA * Rn)
                            / (self.dR)
                            * H0[kR, kTETA]
                        )
                        CS_2 = (
                            self.kt
                            / (self.rho * self.cp * self.frequency * self.pad_inner_radius ** 2)
                            * (self.dTETA * Rs)
                            / (self.dR)
                            * H0[kR, kTETA]
                        )
                        CP_3 = -(CN_2 + CS_2)
                        CE_2 = (
                            self.kt
                            / (self.rho * self.cp * self.frequency * self.pad_inner_radius ** 2)
                            * self.dR
                            / (self.pad_arc_length ** 2 * self.dTETA)
                            * H0[kR, kTETA]
                            / R
                        )
                        CW_2 = (
                            self.kt
                            / (self.rho * self.cp * self.frequency * self.pad_inner_radius ** 2)
                            * self.dR
                            / (self.pad_arc_length ** 2 * self.dTETA)
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
                            self.dR
                            * self.dTETA
                            / (12 * self.pad_arc_length ** 2)
                            * (
                                H0[kR, kTETA] ** 3
                                / (MI[kR, kTETA] * R)
                                * dP0dTETA[kR, kTETA] ** 2
                            )
                        )
                        B_I = MI[kR, kTETA] * R ** 3 / (H0[kR, kTETA]) * self.dR * self.dTETA
                        B_J = (
                            self.dR
                            * self.dTETA
                            / 12
                            * (R * H0[kR, kTETA] ** 3 / MI[kR, kTETA])
                            * dP0dR[kR, kTETA] ** 2
                        )
                        B_K = (
                            self.dR
                            * self.dTETA
                            / (12 * self.pad_arc_length)
                            * (H0[kR, kTETA] ** 3 / R)
                            * dP0dTETA[kR, kTETA]
                        )
                        B_L = (
                            self.dR
                            * self.dTETA
                            / 60
                            * (H0[kR, kTETA] ** 5 / (MI[kR, kTETA] * R))
                            * dP0dR[kR, kTETA] ** 2
                        )
                        B_M = (
                            2
                            * self.dR
                            * self.dTETA
                            * (R * MI[kR, kTETA] / H0[kR, kTETA])
                            * (dHdT) ** 2
                        )
                        B_N = self.dR * self.dTETA / 3 * R * MI[kR, kTETA] * H0[kR, kTETA]
                        B_O = (
                            self.dR
                            * self.dTETA
                            / (120 * self.pad_arc_length ** 2)
                            * (H0[kR, kTETA] ** 5 / (MI[kR, kTETA] * R ** 3))
                            * dP0dTETA[kR, kTETA] ** 2
                        )

                        # vectorization index
                        k = k + 1

                        b[k, 0] = (
                            -B_F
                            + (self.frequency * self.reference_viscosity * self.pad_inner_radius ** 2 / (self.rho * self.cp * h0 ** 2 * self.reference_temperature))
                            * (B_G - B_H - B_I - B_J)
                            + (self.reference_viscosity * self.frequency / (self.rho * self.cp * self.reference_temperature))
                            * (B_K - B_L - B_M - B_N - B_O)
                        )

                        if kTETA == 0 and kR == 0:
                            Mat_coef[k, k] = CP + CS
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + self.n_theta] = CN
                            b[k, 0] = b[k, 0] - 1 * CW

                        if kTETA == 0 and kR > 0 and kR < self.n_radial-1:
                            Mat_coef[k, k] = CP
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + self.n_theta] = CN
                            Mat_coef[k, k - self.n_theta] = CS
                            b[k, 0] = b[k, 0] - 1 * CW

                        if kTETA == 0 and kR == self.n_radial-1:
                            Mat_coef[k, k] = CP + CN
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - self.n_theta] = CS
                            b[k, 0] = b[k, 0] - 1 * CW

                        if kR == 0 and kTETA > 0 and kTETA < self.n_theta-1:
                            Mat_coef[k, k] = CP + CS
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + self.n_theta] = CN

                        if kTETA > 0 and kTETA < self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                            Mat_coef[k, k] = CP
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + self.n_theta] = CN
                            Mat_coef[k, k - self.n_theta] = CS
                            Mat_coef[k, k + 1] = CE

                        if kR == self.n_radial-1 and kTETA > 0 and kTETA < self.n_theta-1:
                            Mat_coef[k, k] = CP + CN
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - self.n_theta] = CS

                        if kR == 0 and kTETA == self.n_theta-1:
                            Mat_coef[k, k] = CP + CE + CS
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + self.n_theta] = CN

                        if kTETA == self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                            Mat_coef[k, k] = CP + CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.n_theta] = CS
                            Mat_coef[k, k + self.n_theta] = CN

                        if kTETA == self.n_theta-1 and kR == self.n_radial-1:
                            Mat_coef[k, k] = CP + CN + CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.n_theta] = CS

                        kTETA = kTETA + 1

                    kR = kR + 1
                    kTETA = 0

                # Temperature field solution
                t = np.linalg.solve(Mat_coef, b)
                cont = -1

                # Temperature matrix
                for ii in range(0, self.n_radial):
                    for jj in range(0, self.n_theta):
                        cont = cont + 1
                        T_new[ii, jj] = t[cont, 0]

                # viscosity field
                varMI=np.zeros((self.n_radial, self.n_theta))
                for ii in range(0, self.n_radial):
                    for jj in range(0, self.n_theta):
                        MI_new[ii, jj] = (
                            (1 / self.reference_viscosity)
                            * self.a_a
                            * np.exp(self.b_b * (self.reference_temperature * T_new[ii, jj]))
                        )
                        varMI[ii, jj] = abs((MI_new[ii, jj] - MI[ii, jj]) / MI[ii, jj])

                T = T_new
                aux1=np.max(varMI)

            # TEMPERATURE ==============================================================
            # ENDS HERE ================================================================
            
            self.Ti = T * self.reference_temperature

            # dimensional pressure
            Pdim = P0 * (self.pad_inner_radius ** 2) * self.frequency * self.reference_viscosity / (h0 ** 2)

            # RESULTING FORCE AND MOMENTUM: Equilibrium position
            XR = self.pad_inner_radius * self.vec_R
            XTETA = self.pad_arc_length * self.vec_TETA
            Xrp = self.pad_pivot_radius * (np.ones((np.size(XR))))

            for ii in range(0, self.n_theta):
                Mxr[:, ii] = (Pdim[:, ii] * (np.transpose(XR) ** 2)) * np.sin(
                    XTETA[ii] - self.angular_pivot_position
                )
                Myr[:, ii] = (
                    -Pdim[:, ii]
                    * np.transpose(XR)
                    * np.transpose(XR * np.cos(XTETA[ii] - self.angular_pivot_position) - Xrp)
                )
                Frer[:, ii] = Pdim[:, ii] * np.transpose(XR)

            frer = np.trapezoid( Frer, XTETA)

            ######################################################################
            mxr = np.trapezoid( Mxr, XTETA)
            myr = np.trapezoid( Myr, XTETA)

            mx = np.trapezoid(mxr, XR)
            my = np.trapezoid( myr, XR)

            resMx = mx
            resMy = my


            ######################################################################

            # if self.op_key == "impos_h0":
            if self.equilibrium_position_mode == "imposed":
                ResFM = np.linalg.norm(np.array([resMx, resMy]))
                self.fzs_load = frer
            
            else:
                fre = -np.trapezoid( frer, XR) + self.fzs_load / self.n_pad

                resFre = fre
                ResFM = np.linalg.norm(np.array([resMx, resMy, resFre]))

            self.initial_position = np.array([x[0],x[1],h0])
            self.score = ResFM

        # --------------------------------------------------------------------------
        # Full temperature field
        TT = np.ones((self.n_radial + 2, self.n_theta + 2))
        TT[1:self.n_radial+1, 1:self.n_theta+1] = np.flipud(self.Ti)
        TT[:, 0] = self.reference_temperature
        TT[0, :] = TT[1, :]
        TT[self.n_radial + 1, :] = TT[self.n_radial, :]
        TT[:, self.n_theta + 1] = TT[:, self.n_theta]

        self.TT = TT

        # --------------------------------------------------------------------------
        # Viscosity field
        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                mi[ii, jj] =  self.a_a * np.exp(self.b_b * (self.Ti[ii, jj]))  # [Pa.s]

        # PRESSURE =================================================================
        # STARTS HERE ==============================================================

        Ar = a_r * self.pad_inner_radius / h0
        As = a_s * self.pad_inner_radius / h0
        MI = 1 / self.reference_viscosity * mi

        # number of volumes
        nk = (self.n_radial) * (self.n_theta)

        # Coefficients Matrix
        Mat_coef = np.zeros((nk, nk))
        b = np.zeros((nk, 1))

        P0 =  self.P_solution(H0, H0ne, H0nw, H0se, H0sw, h0, As, Ar, MI,  Mat_coef, b, P0)

        PPdim = np.zeros((self.n_radial + 2, self.n_theta + 2))
        PPdim[1:-1, 1:-1] = (self.pad_inner_radius ** 2 * self.frequency * self.reference_viscosity / h0 ** 2) * np.flipud(P0)
        self.PPdim = PPdim

        self.hmax = np.max(h0 * H0)
        self.hmin = np.min(h0 * H0)
        # PRESSURE =================================================================
        # ENDS HERE ================================================================
        return H0, H0ne, H0nw, H0se, H0sw, h0, P0, mi
    

    def ArAsh0Equilibrium(self, x):

        """Calculates the equilibrium position of the bearing

        Parameters
        ----------
        a_r = x[0]  : pitch angle axis r [rad]
        a_s = x[1]  : pitch angle axis s [rad]
        h0 = x[2]   : oil film thickness at pivot [m]

        """

        # Variable startup
        MI = np.zeros((self.n_radial, self.n_theta))
        P = np.zeros((self.n_radial, self.n_theta))
        Mxr = np.zeros((self.n_radial, self.n_theta))
        Myr = np.zeros((self.n_radial, self.n_theta))
        Frer = np.zeros((self.n_radial, self.n_theta))

        # Pitch angles alpha_r and alpha_p and oil filme thickness at pivot h0
        a_r = x[0]  # [rad]
        a_s = x[1]  # [rad]

        # Determine h0 based on equilibrium position mode
        if self.equilibrium_position_mode == "imposed":
            h0 = self.initial_position[2]  # Use imposed h0 value
        else:  # "calculate" mode
            h0 = x[2]  # h0 is optimized

        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                MI[ii, jj] = (
                    1 / self.reference_viscosity *  self.a_a * np.exp(self.b_b * (self.Ti[ii, jj]))
                )  # dimensionless

        # Dimensioneless Parameters
        Ar = a_r * self.pad_inner_radius / h0
        As = a_s * self.pad_inner_radius / h0
        H0 = h0 / h0

        # PRESSURE FIELD - Solution of Reynolds equation
        kR = 0
        kTETA = 0

        # pressure vectorization index
        k = -1

        # number of volumes
        nk = (self.n_radial) * (self.n_theta)  # number of volumes

        # Coefficients Matrix
        Mat_coef = np.zeros((nk, nk))
        b = np.zeros((nk, 1))

        for R in self.vec_R:
            for TETA in self.vec_TETA:

                TETAe = TETA + 0.5 * self.dTETA
                TETAw = TETA - 0.5 * self.dTETA
                Rn = R + 0.5 * self.dR
                Rs = R - 0.5 * self.dR

                Hne = (
                    H0
                    + As * (self.Rp - Rn * np.cos(self.pad_arc_length * (TETAe - self.TETAp)))
                    + Ar * Rn * np.sin(self.pad_arc_length * (TETAe - self.TETAp))
                )
                Hnw = (
                    H0
                    + As * (self.Rp - Rn * np.cos(self.pad_arc_length * (TETAw - self.TETAp)))
                    + Ar * Rn * np.sin(self.pad_arc_length * (TETAw - self.TETAp))
                )
                Hse = (
                    H0
                    + As * (self.Rp - Rs * np.cos(self.pad_arc_length * (TETAe - self.TETAp)))
                    + Ar * Rs * np.sin(self.pad_arc_length * (TETAe - self.TETAp))
                )
                Hsw = (
                    H0
                    + As * (self.Rp - Rs * np.cos(self.pad_arc_length * (TETAw - self.TETAp)))
                    + Ar * Rs * np.sin(self.pad_arc_length * (TETAw - self.TETAp))
                )

                if kTETA == 0 and kR == 0:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA == 0 and kR > 0 and kR < self.n_radial-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kTETA == 0 and kR == self.n_radial-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == 0 and kTETA > 0 and kTETA < self.n_theta - 1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA > 0 and kTETA < self.n_theta - 1 and kR > 0 and kR < self.n_radial-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == self.n_radial-1 and kTETA > 0 and kTETA < self.n_theta - 1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == 0 and kTETA == self.n_theta - 1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA == self.n_theta - 1 and kR > 0 and kR < self.n_radial-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kTETA == self.n_theta - 1 and kR == self.n_radial-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                # Coefficients for solving the Reynolds equation
                CE = (
                    1
                    / (24 * self.pad_arc_length ** 2 * MI_e)
                    * (self.dR / self.dTETA)
                    * (Hne ** 3 / Rn + Hse ** 3 / Rs)
                )
                CW = (
                    1
                    / (24 * self.pad_arc_length ** 2 * MI_w)
                    * (self.dR / self.dTETA)
                    * (Hnw ** 3 / Rn + Hsw ** 3 / Rs)
                )
                CN = Rn / (24 * MI_n) * (self.dTETA / self.dR) * (Hne ** 3 + Hnw ** 3)
                CS = Rs / (24 * MI_s) * (self.dTETA / self.dR) * (Hse ** 3 + Hsw ** 3)
                CP = -(CE + CW + CN + CS)

                # vectorization index
                k = k + 1

                b[k, 0] = self.dR / (4 * self.pad_arc_length) * (Rn * Hne + Rs * Hse - Rn * Hnw - Rs * Hsw)

                if kTETA == 0 and kR == 0:
                    Mat_coef[k, k] = CP - CS - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + self.n_theta ] = CN

                if kTETA == 0 and kR > 0 and kR < self.n_radial-1:
                    Mat_coef[k, k] = CP - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + self.n_theta ] = CN
                    Mat_coef[k, k - self.n_theta ] = CS

                if kTETA == 0 and kR == self.n_radial-1:
                    Mat_coef[k, k] = CP - CW - CN
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - self.n_theta ] = CS

                if kR == 0 and kTETA > 0 and kTETA < self.n_theta - 1:
                    Mat_coef[k, k] = CP - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + self.n_theta ] = CN

                if kTETA > 0 and kTETA < self.n_theta - 1 and kR > 0 and kR < self.n_radial-1:
                    Mat_coef[k, k] = CP
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + self.n_theta ] = CN
                    Mat_coef[k, k - self.n_theta ] = CS
                    Mat_coef[k, k + 1] = CE

                if kR == self.n_radial-1 and kTETA > 0 and kTETA < self.n_theta - 1:
                    Mat_coef[k, k] = CP - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - self.n_theta ] = CS

                if kR == 0 and kTETA == self.n_theta - 1:
                    Mat_coef[k, k] = CP - CE - CS
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + self.n_theta ] = CN

                if kTETA == self.n_theta - 1 and kR > 0 and kR < self.n_radial-1:
                    Mat_coef[k, k] = CP - CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - self.n_theta ] = CS
                    Mat_coef[k, k + self.n_theta ] = CN

                if kTETA == self.n_theta - 1 and kR == self.n_radial-1:
                    Mat_coef[k, k] = CP - CE - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - self.n_theta ] = CS

                kTETA = kTETA + 1

            kR = kR + 1
            kTETA = 0


        p = np.linalg.solve(Mat_coef, b)
        cont = -1

        # pressure matrix
        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                cont = cont + 1
                P[ii, jj] = p[cont, 0]

                # boundary conditions of pressure
                if P[ii, jj] < 0:
                    P[ii, jj] = 0

        # dimensional pressure
        Pdim = P * (self.pad_inner_radius ** 2) * self.frequency *self. reference_viscosity / (h0 ** 2)

        # RESULTING FORCE AND MOMENTUM: Equilibrium position
        XR = self.pad_inner_radius * self.vec_R
        XTETA = self.pad_arc_length * self.vec_TETA
        Xrp = self.pad_pivot_radius * (np.ones((np.size(XR))))

        for ii in range(0, self.n_theta):
            Mxr[:, ii] = (Pdim[:, ii] * (np.transpose(XR) ** 2)) * np.sin(
                XTETA[ii] - self.angular_pivot_position)
            Myr[:, ii] = (
                -Pdim[:, ii]
                * np.transpose(XR)
                * np.transpose(XR * np.cos(XTETA[ii] - self.angular_pivot_position) - Xrp)
            )
            Frer[:, ii] = Pdim[:, ii] * np.transpose(XR)
        
        frer = np.trapezoid( Frer, XTETA)
        ######################################################################
        mxr = np.trapezoid( Mxr, XTETA)
        myr = np.trapezoid( Myr, XTETA)

        mx = np.trapezoid(mxr, XR)
        my = np.trapezoid( myr, XR)
        ######################################################################
        
        # if self.op_key == "impos_h0":
        if self.equilibrium_position_mode == "imposed":
            score = np.linalg.norm([mx, my])
        
        else:  # "calculate" operation mode
            fre = -np.trapezoid(frer, XR) + self.fzs_load / self.n_pad
            score = np.linalg.norm([mx, my, fre])

        return score


    def P_solution(self, H0, H0ne, H0nw, H0se, H0sw, h0, As, Ar, MI,  Mat_coef, b, P0):
        
        # PRESSURE FIELD - Solution of Reynolds equation
        kR = 0
        kTETA = 0

        # pressure vectorization index
        k = -1

        for R in self.vec_R:
            for TETA in self.vec_TETA:

                TETAe = TETA + 0.5 * self.dTETA
                TETAw = TETA - 0.5 * self.dTETA
                Rn = R + 0.5 * self.dR
                Rs = R - 0.5 * self.dR

                H0[kR, kTETA] = (
                    h0 / h0
                    + As * (self.Rp - R * np.cos(self.pad_arc_length * (TETA - self.TETAp)))
                    + Ar * R * np.sin(self.pad_arc_length * (TETA - self.TETAp))
                )
                H0ne[kR, kTETA] = (
                    h0 / h0
                    + As * (self.Rp - Rn * np.cos(self.pad_arc_length * (TETAe - self.TETAp)))
                    + Ar * Rn * np.sin(self.pad_arc_length * (TETAe - self.TETAp))
                )
                H0nw[kR, kTETA] = (
                    h0 / h0
                    + As * (self.Rp - Rn * np.cos(self.pad_arc_length * (TETAw - self.TETAp)))
                    + Ar * Rn * np.sin(self.pad_arc_length * (TETAw - self.TETAp))
                )
                H0se[kR, kTETA] = (
                    h0 / h0
                    + As * (self.Rp - Rs * np.cos(self.pad_arc_length * (TETAe - self.TETAp)))
                    + Ar * Rs * np.sin(self.pad_arc_length * (TETAe - self.TETAp))
                )
                H0sw[kR, kTETA] = (
                    h0 / h0
                    + As * (self.Rp - Rs * np.cos(self.pad_arc_length * (TETAw - self.TETAp)))
                    + Ar * Rs * np.sin(self.pad_arc_length * (TETAw - self.TETAp))
                )

                if kTETA == 0 and kR == 0:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA == 0 and kR > 0 and kR < self.n_radial-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kTETA == 0 and kR == self.n_radial-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == 0 and kTETA > 0 and kTETA < self.n_theta-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA > 0 and kTETA < self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == self.n_radial-1 and kTETA > 0 and kTETA < self.n_theta-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == 0 and kTETA == self.n_theta-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA == self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kTETA == self.n_theta-1 and kR == self.n_radial-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                # Coefficients for solving the Reynolds equation
                CE = (
                    1
                    / (24 * self.pad_arc_length ** 2 * MI_e)
                    * (self.dR / self.dTETA)
                    * (H0ne[kR, kTETA] ** 3 / Rn + H0se[kR, kTETA] ** 3 / Rs)
                )
                CW = (
                    1
                    / (24 * self.pad_arc_length ** 2 * MI_w)
                    * (self.dR / self.dTETA)
                    * (H0nw[kR, kTETA] ** 3 / Rn + H0sw[kR, kTETA] ** 3 / Rs)
                )
                CN = (
                    Rn
                    / (24 * MI_n)
                    * (self.dTETA / self.dR)
                    * (H0ne[kR, kTETA] ** 3 + H0nw[kR, kTETA] ** 3)
                )
                CS = (
                    Rs
                    / (24 * MI_s)
                    * (self.dTETA / self.dR)
                    * (H0se[kR, kTETA] ** 3 + H0sw[kR, kTETA] ** 3)
                )
                CP = -(CE + CW + CN + CS)

                # vectorization index
                k = k + 1

                b[k, 0] = (
                    self.dR
                    / (4 * self.pad_arc_length)
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
                    Mat_coef[k, k + self.n_theta] = CN

                if kTETA == 0 and kR > 0 and kR < self.n_radial-1:
                    Mat_coef[k, k] = CP - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + self.n_theta] = CN
                    Mat_coef[k, k - self.n_theta] = CS

                if kTETA == 0 and kR == self.n_radial-1:
                    Mat_coef[k, k] = CP - CW - CN
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - self.n_theta] = CS

                if kR == 0 and kTETA > 0 and kTETA < self.n_theta-1:
                    Mat_coef[k, k] = CP - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + self.n_theta] = CN

                if kTETA > 0 and kTETA < self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                    Mat_coef[k, k] = CP
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + self.n_theta] = CN
                    Mat_coef[k, k - self.n_theta] = CS
                    Mat_coef[k, k + 1] = CE

                if kR == self.n_radial-1 and kTETA > 0 and kTETA < self.n_theta-1:
                    Mat_coef[k, k] = CP - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - self.n_theta] = CS

                if kR == 0 and kTETA == self.n_theta-1:
                    Mat_coef[k, k] = CP - CE - CS
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + self.n_theta] = CN

                if kTETA == self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                    Mat_coef[k, k] = CP - CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - self.n_theta] = CS
                    Mat_coef[k, k + self.n_theta] = CN

                if kTETA == self.n_theta-1 and kR == self.n_radial-1:
                    Mat_coef[k, k] = CP - CE - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - self.n_theta] = CS

                kTETA = kTETA + 1

            kR = kR + 1
            kTETA = 0

        # Pressure field solution
        p = np.linalg.solve(Mat_coef, b)
        cont = -1

        # pressure matrix
        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                cont = cont + 1
                P0[ii, jj] = p[cont, 0]

                # pressure boundary conditions
                if P0[ii, jj] < 0:
                    P0[ii, jj] = 0

        return P0


    def lub_selector(self):
        lubricant_dict = {
            "ISOVG32": {
                "viscosity1": 0.027968,                               # Pa.s
                "temp1": 40.0,                                    # degC
                "viscosity2": 0.004667,                               # Pa. s
                "temp2": 100.0,                                   # degC
                "lube_density": 873.99629,                            # kg/m³
                "lube_cp": 1948.7995685758851,                        # J/(kg*degC)
                "lube_conduct": 0.13126,                              # W/(m*degC)
            },
            "ISOVG46": {
                "viscosity1": 0.039693,                               # Pa.s
                "temp1": 40.0,                                    # degC
                "viscosity2": 0.006075,                               # Pa.s
                "temp2": 100.00000,                                   # degC
                "lube_density": 862.9,                                # kg/m³ 
                "lube_cp": 1950,                                      # J/(kg*degC)
                "lube_conduct": 0.15,                                 # W/(m*degC)
            },
            "ISOVG68": {
                "viscosity1": 0.060248,                               # Pa.s = N*s/m²
                "temp1": 40.00000,                                    # degC
                "viscosity2": 0.0076196,                              # Pa.s = N*s/m²
                "temp2": 100.00000,                                   # degC
                "lube_density": 886.00,                               # kg/m³ 
                "lube_cp": 1890.00,                                   # J/(kg*degC)
                "lube_conduct": 0.1316,                               # W/(m*degC)
            },
            "TEST": {
                "viscosity1": 0.02,                                   # Pa.s = N*s/m²
                "temp1": 50.00,                                       # degC
                "viscosity2": 0.01,                                   # Pa.s = N*s/m²
                "temp2": 80.00,                                       # degC
                "lube_density": 880.60,                               # kg/m³ 
                "lube_cp": 1800,                                      # J/(kg*degC)
                "lube_conduct": 0.1304                                # J/s*m*degC  --W/(m*degC)--
            },
        }
        return lubricant_dict[self.lubricant]


    def Coef_din(self, H0ne, H0nw, H0se, H0sw, h0, P0, mi):
        # --------------------------------------------------------------------------
        # Stiffness and Damping Coefficients
        wp = self.frequency  # perturbation frequency [rad/s]
        WP = wp / self.frequency

        # HYDROCOEFF_z =============================================================
        # STARTS HERE ==============================================================

        MI = (1 / self.reference_viscosity) * mi

        kR = 0
        kTETA = 0
        k = -1  # pressure vectorization index
        nk = (self.n_radial) * (self.n_theta)  # volumes number

        # coefficients matrix
        Mat_coef = np.zeros((nk, nk))
        b_coef = np.zeros((nk, 1),dtype=complex)
        p_coef = np.zeros((nk, 1),dtype=complex)
        P_coef = np.zeros((self.n_radial, self.n_theta),dtype=complex)
        P_dim_coef = np.zeros((self.n_radial, self.n_theta),dtype=complex)
        Mxr_coef = np.zeros((self.n_radial, self.n_theta),dtype=complex)
        Myr_coef = np.zeros((self.n_radial, self.n_theta),dtype=complex)
        Frer_coef = np.zeros((self.n_radial, self.n_theta),dtype=complex)

        for R in self.vec_R:
            for TETA in self.vec_TETA:

                TETAe = TETA + 0.5 * self.dTETA
                TETAw = TETA - 0.5 * self.dTETA
                Rn = R + 0.5 * self.dR
                Rs = R - 0.5 * self.dR

                if kTETA == 0 and kR == 0:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / self.dTETA
                    dPdTETAw = P0[kR, kTETA] / (0.5 * self.dTETA)
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / self.dR
                    dPdRs = P0[kR, kTETA] / (0.5 * self.dR)

                if kTETA == 0 and kR > 0 and kR < self.n_radial - 1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / self.dTETA
                    dPdTETAw = P0[kR, kTETA] / (0.5 * self.dTETA)
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / self.dR
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / self.dR

                if kTETA == 0 and kR == self.n_radial - 1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / self.dTETA
                    dPdTETAw = P0[kR, kTETA] / (0.5 * self.dTETA)
                    dPdRn = -P0[kR, kTETA] / (0.5 * self.dR)
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / self.dR

                if kR == 0 and kTETA > 0 and kTETA < self.n_theta-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / self.dTETA
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / self.dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / self.dR
                    dPdRs = P0[kR, kTETA] / (0.5 * self.dR)

                if kTETA > 0 and kTETA < self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / self.dTETA
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / self.dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / self.dR
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / self.dR

                if kR == self.n_radial-1 and kTETA > 0 and kTETA < self.n_theta -1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = (P0[kR, kTETA + 1] - P0[kR, kTETA]) / self.dTETA
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / self.dTETA
                    dPdRn = -P0[kR, kTETA] / (0.5 * self.dR)
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / self.dR

                if kR == 0 and kTETA == self.n_theta-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]
                    dPdTETAe = -P0[kR, kTETA] / (0.5 * self.dTETA)
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / self.dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / self.dR
                    dPdRs = P0[kR, kTETA] / (0.5 * self.dR)

                if kTETA == self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = -P0[kR, kTETA] / (0.5 * self.dTETA)
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / self.dTETA
                    dPdRn = (P0[kR + 1, kTETA] - P0[kR, kTETA]) / self.dR
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / self.dR

                if kTETA == self.n_theta-1 and kR == self.n_radial-1:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])
                    dPdTETAe = -P0[kR, kTETA] / (0.5 * self.dTETA)
                    dPdTETAw = (P0[kR, kTETA] - P0[kR, kTETA - 1]) / self.dTETA
                    dPdRn = -P0[kR, kTETA] / (0.5 * self.dR)
                    dPdRs = (P0[kR, kTETA] - P0[kR - 1, kTETA]) / self.dR

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
                    / (24 * self.pad_arc_length ** 2 * MI_e)
                    * (self.dR / self.dTETA)
                    * (
                        As_ne * H0ne[kR, kTETA] ** 3 / Rn
                        + As_se * H0se[kR, kTETA] ** 3 / Rs
                    )
                )
                CE_2 = (
                    self.dR
                    / (48 * self.pad_arc_length ** 2 * MI_e)
                    * (
                        G2_ne * H0ne[kR, kTETA] ** 3 / Rn
                        + G2_se * H0se[kR, kTETA] ** 3 / Rs
                    )
                )
                CE = CE_1 + CE_2

                CW_1 = (
                    1
                    / (24 * self.pad_arc_length ** 2 * MI_w)
                    * (self.dR / self.dTETA)
                    * (
                        As_nw * H0nw[kR, kTETA] ** 3 / Rn
                        + As_sw * H0sw[kR, kTETA] ** 3 / Rs
                    )
                )
                CW_2 = (
                    -self.dR
                    / (48 * self.pad_arc_length ** 2 * MI_w)
                    * (
                        G2_nw * H0nw[kR, kTETA] ** 3 / Rn
                        + G2_sw * H0sw[kR, kTETA] ** 3 / Rs
                    )
                )
                CW = CW_1 + CW_2

                CN_1 = (
                    Rn
                    / (24 * MI_n)
                    * (self.dTETA / self.dR)
                    * (As_ne * H0ne[kR, kTETA] ** 3 + As_nw * H0nw[kR, kTETA] ** 3)
                )
                CN_2 = (
                    Rn
                    / (48 * MI_n)
                    * (self.dTETA)
                    * (G1_ne * H0ne[kR, kTETA] ** 3 + G1_nw * H0nw[kR, kTETA] ** 3)
                )
                CN = CN_1 + CN_2

                CS_1 = (
                    Rs
                    / (24 * MI_s)
                    * (self.dTETA / self.dR)
                    * (As_se * H0se[kR, kTETA] ** 3 + As_sw * H0sw[kR, kTETA] ** 3)
                )
                CS_2 = (
                    -Rs
                    / (48 * MI_s)
                    * (self.dTETA)
                    * (G1_se * H0se[kR, kTETA] ** 3 + G1_sw * H0sw[kR, kTETA] ** 3)
                )
                CS = CS_1 + CS_2

                CP = -(CE_1 + CW_1 + CN_1 + CS_1) + (CE_2 + CW_2 + CN_2 + CS_2)

                B_1 = (Rn * self.dTETA / (8 * MI_n)) * dPdRn * (
                    As_ne * H0ne[kR, kTETA] ** 2 + As_nw * H0nw[kR, kTETA] ** 2
                ) - (Rs * self.dTETA / (8 * MI_s)) * dPdRs * (
                    As_se * H0se[kR, kTETA] ** 2 + As_sw * H0sw[kR, kTETA] ** 2
                )
                B_2 = (self.dR / (8 * self.pad_arc_length ** 2 * MI_e)) * dPdTETAe * (
                    As_ne * H0ne[kR, kTETA] ** 2 / Rn
                    + As_se * H0se[kR, kTETA] ** 2 / Rs
                ) - (self.dR / (8 * self.pad_arc_length ** 2 * MI_w)) * dPdTETAw * (
                    As_nw * H0nw[kR, kTETA] ** 2 / Rn
                    + As_sw * H0sw[kR, kTETA] ** 2 / Rs
                )
                B_3 =self. dR / (4 * self.pad_arc_length) * (As_ne * Rn + As_se * Rs) - self.dR / (
                    4 * self.pad_arc_length
                ) * (As_nw * Rn + As_sw * Rs)
                B_4 = (
                    complex(0, 1)
                    * WP
                    * self.dR
                    * self.dTETA
                    / 4
                    * (Rn * As_ne + Rn * As_nw + Rs * As_se + Rs * As_sw)
                )

                # vectorization index
                k = k + 1

                b_coef[k, 0] = -(B_1 + B_2) + B_3 + B_4

                if kTETA == 0 and kR == 0:
                    Mat_coef[k, k] = CP - CW - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + self.n_theta] = CN

                if kTETA == 0 and kR > 0 and kR < self.n_radial-1:
                    Mat_coef[k, k] = CP - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + self.n_theta] = CN
                    Mat_coef[k, k - self.n_theta] = CS

                if kTETA == 0 and kR == self.n_radial-1:
                    Mat_coef[k, k] = CP - CW - CN
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - self.n_theta] = CS

                if kR == 0 and kTETA > 0 and kTETA < self.n_theta-1:
                    Mat_coef[k, k] = CP - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + self.n_theta] = CN

                if kTETA > 0 and kTETA < self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                    Mat_coef[k, k] = CP
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + self.n_theta] = CN
                    Mat_coef[k, k - self.n_theta] = CS
                    Mat_coef[k, k + 1] = CE

                if kR == self.n_radial-1 and kTETA > 0 and kTETA < self.n_theta-1:
                    Mat_coef[k, k] = CP - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - self.n_theta] = CS

                if kR == 0 and kTETA == self.n_theta-1:
                    Mat_coef[k, k] = CP - CE - CS
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + self.n_theta] = CN

                if kTETA == self.n_theta-1 and kR > 0 and kR < self.n_radial-1:
                    Mat_coef[k, k] = CP - CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - self.n_theta] = CS
                    Mat_coef[k, k + self.n_theta] = CN

                if kTETA == self.n_theta-1 and kR == self.n_radial-1:
                    Mat_coef[k, k] = CP - CE - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k - self.n_theta] = CS

                kTETA = kTETA + 1

            kR = kR + 1
            kTETA = 0

        # vectorized pressure field solution
        p_coef = np.linalg.solve(Mat_coef, b_coef)
        cont = -1

        # pressure matrix
        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                cont = cont + 1
                P_coef[ii, jj] = p_coef[cont, 0]

        # dimensional pressure
        Pdim_coef = P_coef * (self.pad_inner_radius ** 2) * self.frequency * self.reference_viscosity / (h0 ** 3)

        # RESULTING FORCE AND MOMENTUM: Equilibrium position
        XR = self.pad_inner_radius * self.vec_R
        XTETA = self.pad_arc_length * self.vec_TETA
        Xrp = self.pad_pivot_radius * (np.ones((np.size(XR))))

        for ii in range(0, self.n_theta):
            Mxr_coef[:, ii] = (Pdim_coef[:, ii] * (np.transpose(XR) ** 2)) * np.sin(
                XTETA[ii] - self.angular_pivot_position
            )
            Myr_coef[:, ii] = (
                -Pdim_coef[:, ii]
                * np.transpose(XR)
                * np.transpose(XR * np.cos(XTETA[ii] - self.angular_pivot_position) - Xrp)
            )
            Frer_coef[:, ii] = Pdim_coef[:, ii] * np.transpose(XR)

        ######################################################################
        mxr_coef = np.trapezoid( Mxr_coef, XTETA)
        myr_coef = np.trapezoid( Myr_coef, XTETA)
        frer_coef = np.trapezoid( Frer_coef, XTETA)

        mx_coef = np.trapezoid(mxr_coef, XR)
        my_coef = np.trapezoid( myr_coef, XR)
        fre_coef = -np.trapezoid( frer_coef, XR) 
        

        ######################################################################

        # HYDROCOEFF_z =============================================================
        # ENDS HERE ================================================================

        self.K = self.n_pad * np.real(fre_coef)  # Stiffness Coefficient
        self.C = self.n_pad * 1 / wp * np.imag(fre_coef)  # Damping Coefficient


    def plot_results(self):
        # PLOT FIGURES ==============================================================
        # STARTS HERE ==============================================================
        # PLOT FIGURES ==============================================================
        # STARTS HERE ==============================================================

        # Define vectors and matrix
        yh = np.zeros((self.n_radial+2))
        auxtransf = np.zeros((self.n_theta+2))
        XH = np.zeros((self.n_radial+2,self.n_theta+2))
        YH = np.zeros((self.n_radial+2,self.n_theta+2))

        yh[0] = self.pad_outer_radius
        yh[-1] = self.pad_inner_radius
        yh[1:self.n_radial+1] = np.arange((self.pad_outer_radius - 0.5 * self.dR * self.pad_inner_radius), self.pad_inner_radius, -(self.dR * self.pad_inner_radius))
   
        auxtransf[0] = np.pi/2 + self.pad_arc_length/2
        auxtransf[-1] = np.pi/2 - self.pad_arc_length/2
        auxtransf[1:self.n_theta+1] = np.arange(np.pi/2 + self.pad_arc_length/2 - (0.5 * self.dTETA * self.pad_arc_length), np.pi/2 - self.pad_arc_length/2, -self.dTETA * self.pad_arc_length)

        for ii in range(0, self.n_radial+2):
            for jj in range(0, self.n_theta+2):
                XH[ii, jj] = yh[ii] * np.cos(auxtransf[jj])
                YH[ii, jj] = yh[ii] * np.sin(auxtransf[jj])

        ang = []
        ang_pad = 360/self.n_pad

        d_ang = 20/10
        ang1 = np.arange((ang_pad)+(ang_pad - 20)-40, ang_pad+ang_pad+d_ang-40 , d_ang)

        fig = go.Figure()
        fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=0, b=0))
        fig.add_trace(go.Surface(
                x=XH,
                y=YH,
                z=self.PPdim
            ))
        fig.update_layout(xaxis_range=[np.min(ang1), np.max(ang1)], yaxis_range=[np.min(YH), np.max(YH)])
        fig.update_layout(title="Pressure field")
        fig.update_layout(plot_bgcolor="white")
        fig.update_scenes(
            xaxis_title=dict(
                text="Angular length [rad]", font=dict(family="Times New Roman", size=22)
            ),
            xaxis_tickfont=dict(family="Times New Roman", size=14),
        )
        fig.update_scenes(
            yaxis_title=dict(
                text="Radial length [m]", font=dict(family="Times New Roman", size=22)
            ),
            yaxis_tickfont=dict(family="Times New Roman", size=14),
        )
        fig.update_scenes(
            # zaxis=dict(range=[0.53, 1]),
            zaxis_title=dict(text="Pressure[Pa]", font=dict(family="Times New Roman", size=22)),
            zaxis_tickfont=dict(family="Times New Roman", size=14),
        )
        camera = dict(
            eye=dict(x=-1.5, y=-4, z=1.5),
            center=dict(x=0, y=0, z=0),
            # up=dict(x=0, y=0, z=0)
        )
        fig.update_layout(scene_camera=camera)
        fig.update_scenes(aspectratio=dict(x=1.8,y=1.8,z=1.8))
        fig.show()
        # fig.write_image("pressure_field3D_Th.pdf", width=900, height=500, engine="kaleido")



        fig = go.Figure()
        fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=0, b=0))
        fig.add_trace(go.Surface(
                x=XH,
                y=YH,
                z=self.TT
            ))
        fig.update_layout(xaxis_range=[np.min(ang1), np.max(ang1)], yaxis_range=[np.min(YH), np.max(YH)])
        fig.update_layout(title="Temperature field")
        fig.update_layout(plot_bgcolor="white")
        fig.update_scenes(
            xaxis_title=dict(
                text="Angular length [rad]", font=dict(family="Times New Roman", size=22)
            ),
            xaxis_tickfont=dict(family="Times New Roman", size=14),
        )
        fig.update_scenes(
            yaxis_title=dict(
                text="Radial length [m]", font=dict(family="Times New Roman", size=22)
            ),
            yaxis_tickfont=dict(family="Times New Roman", size=14),
        )
        fig.update_scenes(
            # zaxis=dict(range=[0.53, 1]),
            zaxis_title=dict(text="Temperature[ºC]", font=dict(family="Times New Roman", size=22)),
            zaxis_tickfont=dict(family="Times New Roman", size=14),
        )
        camera = dict(
            eye=dict(x=-1.5, y=-4, z=1.5),
            center=dict(x=0, y=0, z=0),
            # up=dict(x=0, y=0, z=0)
        )
        fig.update_layout(scene_camera=camera)
        fig.update_scenes(aspectratio=dict(x=1.8,y=1.8,z=1.8))
        fig.show()
        # fig.write_image("temperature_field3D_Th.pdf", width=900, height=500, engine="kaleido")

        max_val = self.TT.max()


        xm, xM = XH.min(), XH.max()
        ym, yM = YH.min(), YH.max()

        xr = np.linspace(xm, xM, 800)
        yr= np.linspace(ym, yM, 800)
        xr, yr = np.meshgrid(xr, yr)

        Z = griddata((XH.flatten(), YH.flatten()), self.TT.flatten(), (xr, yr), method="cubic")

        fig = go.Figure(
            go.Contour(
                x=xr[0],
                y=yr[:, 0],
                z=Z,
                ncontours=15,
                contours=dict(start=np.nanmin(Z), end=np.nanmax(Z)),
                colorbar=dict(
                    title=dict(
                        text="Temperature (°C)", 
                        side="right", 
                        font=dict(size=30, family="Times New Roman")
                    ),
                    tickfont=dict(size=20)
                )
            )
        )
        fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_layout(title="Temperature field")
        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )
        fig.update_layout(xaxis_range=[np.min(XH), np.max(XH)])
        fig.update_xaxes(
            tickfont=dict(size=20),
            title=dict(text="X Direction (m)", font=dict(family="Times New Roman", size=20)),
            range=[np.min(XH),np.max(XH)]
        )
        fig.update_yaxes(
            tickfont=dict(size=20),
            title=dict(
                text="Y Direction(m)", font=dict(family="Times New Roman", size=20)
            ),
        )
        fig.update_layout(plot_bgcolor="white")
        # fig.write_image("temperature_field_Th.pdf", width=900, height=500, engine="kaleido")
        fig.show()

        Z = griddata((XH.flatten(), YH.flatten()), self.PPdim.flatten(), (xr, yr) , method="cubic")

        fig = go.Figure(
            go.Contour(
                x=xr[0], 
                y=yr[:, 0], 
                z=Z, 
                ncontours=15,
                contours=dict(
                    start=np.nanmin(Z), 
                    end=np.nanmax(Z)
                ),
                colorbar=dict(
                    title=dict(
                        text="Pressure (Pa)", 
                        side="right",  # ← equivalente ao antigo titleside
                        font=dict(size=30, family="Times New Roman")
                    ),
                    tickfont=dict(size=20),
                ),
            )
        )
        fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_layout(title="Pressure field")
        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )
        fig.update_layout(xaxis_range=[np.min(XH), np.max(XH)])
        fig.update_xaxes(
            tickfont=dict(size=20),
            title=dict(text="X Direction (m)", font=dict(family="Times New Roman", size=20)),
            range=[np.min(XH),np.max(XH)]
        )
        fig.update_yaxes(
            tickfont=dict(size=20),
            title=dict(
                text="Y Direction(m)", font=dict(family="Times New Roman", size=20)
            ),
        )
        fig.update_layout(plot_bgcolor="white")
        # fig.write_image("temperature_field_Th.pdf", width=900, height=500, engine="kaleido")
        fig.show()



        # max_val = self.PPdim.max()
        # fig = go.Figure()
        # fig.add_trace(go.Contour(
        #     z=self.PPdim,
        #     x=ang1, # horizontal axis
        #     y=YH[:,0], # vertical axis
        #     zmin=0,
        #     zmax=max_val,
        #     ncontours=15,
        #     colorbar=dict(
        #         title="Pressure (Pa)",  # title here
        #         titleside="right",
        #         titlefont=dict(size=30, family="Times New Roman"),
        #         tickfont=dict(size=22),
        #         ),
        #     )
        # )
        # fig.update_traces(
        #     contours_coloring="fill",
        #     contours_showlabels=True,
        #     contours_labelfont=dict(size=20),
        # )
        # fig.update_layout(xaxis_range=[0, 360])
        # fig.update_xaxes(
        #     tickfont=dict(size=22),
        #     title=dict(text="Angle (º)", font=dict(family="Times New Roman", size=30)),
        #     range=[0,22]
        # )
        # fig.update_yaxes(
        #     tickfont=dict(size=22),
        #     title=dict(
        #         text="Pad Length (m)", font=dict(family="Times New Roman", size=30)
        #     ),
        # )
        # fig.update_layout(plot_bgcolor="white")
        # fig.show()
        # fig.write_image("pressure_field_Th.pdf", width=900, height=500, engine="kaleido")

""""
def hydroplots (pad_inner_radius, pad_outer_radius, dR, dTETA, pad_arc_length, self.n_radial, self.n_theta):
    
    # Define vectors and matrix
    yh = np.zeros((1, n_radial+2))
    auxtransf = np.zeros((1, n_theta+2))
    XH = np.zeros((n_radial+2,n_theta+2))
    YH = np.zeros((n_radial+2,n_theta+2))

    yh[0] = pad_outer_radius
    yh[n_radial+1] = pad_inner_radius
    yh[1:n_radial] = ((pad_outer_radius - 0.5 * dR * pad_inner_radius), -(dR * pad_inner_radius),(pad_inner_radius + 0.5 * dR * pad_inner_radius))
   
    auxtransf[0] = np.pi/2 + pad_arc_length/2
    auxtransf[n_theta+1] = np.pi/2 - pad_arc_length/2
    auxtransf[1:n_theta +1] = np.pi/2 + pad_arc_length/2 - (0.5 * dTETA * pad_arc_length), -dTETA * pad_arc_length, np.pi/2 - pad_arc_length/2 + (0.5 * dTETA * pad_arc_length)

    for ii in range(0, n_radial+1):
        for jj in range(0, n_theta+1):
            XH[ii, jj] = yh[ii] * np.cos(auxtransf[jj])
            YH[ii, jj] = yh[ii] * np.sin(auxtransf[jj])

    yraio1 = pad_inner_radius * np.sin((np.pi/2 - pad_arc_length/2), dTETA * pad_arc_length, (np.pi/2 + pad_arc_length/2))
    xraio1 = pad_inner_radius * np.cos(np.pi/2 - pad_arc_length/2, dTETA * pad_arc_length, np.pi/2 + pad_arc_length/2)
    yraio2 = pad_outer_radius * np.sin((np.pi/2 - pad_arc_length/2), dTETA*pad_arc_length, (np.pi/2 + pad_arc_length/2))
    xraio2 = pad_outer_radius * np.cos(np.pi/2 - pad_arc_length/2, dTETA*pad_arc_length, np.pi/2 +pad_arc_length/2)

    dx = (xraio1[0] - xraio2[0]/(n_theta - 2))
    xreta1 = xraio2[0], dx, xraio1[0]
    yreta1 = yraio2[0], dx * np.tan(np.pi/2 -pad_arc_length/2), yraio1[0]
    xreta2 = xraio2[-1], -dx, xraio1[-1]
    yreta2 = yraio2[0], dx * np.tan(np.pi/2 - pad_arc_length/2), yraio1[0]
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

    bearing = THDThrust(        
        pad_inner_radius = 0.5 * 2300e-3,
        pad_outer_radius = 0.5 * 3450e-3,
        pad_pivot_radius = 0.5 * 2885e-3,
        pad_arc_length = 26,
        angular_pivot_position = 15,
        oil_supply_temperature = 40,
        lubricant = "ISOVG68",
        n_pad = 12,
        n_theta = 10,
        n_radial = 10,
        frequency = 90,
        # choice_CAIMP = {"calc_h0":{"init_guess":[-0.000274792355106384, -1.69258824831925e-05, 0.000191418606538599], "load":13320e3, "print":["result","progress"]}},
        equilibrium_position_mode = "calculate",
        fzs_load = 13320e3,
        radial_inclination_angle = -0.000274792355106384, # a_r
        circumferential_inclination_angle = -1.69258824831925e-05, # a_s
        initial_film_thickness = 0.000191418606538599, # h0
        print_result=True,
        print_progress=True,
        print_time=False,
    )
    
    # bearing.run(frequency=90, choice_CAIMP={"calc_h0":{"init_guess":[-0.000274792355106384, -1.69258824831925e-05, 0.000191418606538599], "load":13320e3, "print":["result","progress"]}}, Coefs_D={"show_coef":True})
    # bearing.run(frequency=90, choice_CAIMP={"impos_h0":{"init_guess":[-0.000274792355106384, -1.69258824831925e-05], "h0":0.0001864, "print":["result","progress"]}}, Coefs_D={"show_coef":True})
    #return bearing

    # bearing.run()
    # bearing.plot_results()

if __name__ == "__main__":
    thrust_bearing_example()
