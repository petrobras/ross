from cmath import sin
# import mpmath as fp
import numpy as np
import scipy 
from numpy.linalg import pinv
from scipy.linalg import solve
from decimal import Decimal
from scipy.optimize import fmin, Bounds

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm

class Tilting:

    """ This class calculates the pressure and temperature fields, equilibrium
    position of a tilting-pad thrust bearing. It is also possible to obtain the
    stiffness and damping coefficients.
    
    Parameters
    ----------
    Bearing Geometry
    ^^^^^^^^^^^^^^^^
    Describes the geometric characteristics.
    Rs : float
    Rotor radius. Default unit is meter
    Rp : float
    Individual Pad radius. Default unit is meter.
    npad : integer
    Number of pads.
    tpad : float  
    Pad thickness. Default unit is meter.
    betha_p : float
    Individual Pad angle. Default unit is degrees.
    rp_pad : float
    Pivot offset.
    L : float
    Bearing length. Default unit is meter.
    Cr : float
    Radial clearance. Default unit is meter.
    
    Operating conditions
    ^^^^^^^^^^^^^^^^^^^^
    Describes the operating conditions of the bearing
    speed : float
    Rotor rotating speed. Default unit is rad/s
    load : Float
    Axial load. The unit is Newton.
    Tcub : Float
    Oil tank temperature. The unit is °C
    x0  : array
    Initial Equilibrium Position

    
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
    nX : integer
    Number of volumes along the X direction.
    nZ : integer
    Number of volumes along the Z direction.
    Z1 : float
    Initial dimensionless X.
    Z2 : float
    Initial dimensionless Z.
    
    
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
    ecc : float
    Eccentricity.

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
        Rs,    # Rotor radius
        npad,  # Number of pads
        Rp,    # Pad radius
        tpad,  # Pad thickness
        betha_p, # Pad angle
        rp_pad, # Pivot offset
        L,     # Bearing length
        lubricant,
        Tcub,  # Oil tank temperature
        nX, # n° volumes x
        nZ, # n° volumes z
        Cr, # Radial Clearance [m]
        sigma,  # Array
        speed,
        choice_CAIMP, # Method (calculating or imposing equilibrium position)
        Coefs_D=None,
    ):

        self.Rs = Rs 
        self.npad = npad
        self.Rp = Rp
        self.tpad = tpad
        self.betha_p = betha_p * (np.pi / 180)   # Pad angle [rad]
        self.rp_pad = rp_pad 
        self.Cr = Cr 
        self.L = L
        self.Tcub = Tcub
        T0 = Tcub
        self.T0 = self.Tcub # Reference temperature [Celsius]
        self.lubricant = lubricant

        self.nX = nX    # direção circunferêncial
        self.nZ = nZ    # direção axial
        self.Z1 = -0.5
        self.Z2 = 0.5
        self.sigma = np.array(sigma) * (np.pi/180) #rad
        TETA1= - ( self.rp_pad ) * self.betha_p # initial coordinate in the TETA diretion
        TETA2= ( 1-self.rp_pad ) * self.betha_p # final coordinate in the TETA diretion
        self.TETA1 = TETA1
        self.TETA2 = TETA2
        self.dTETA = ( self.TETA2 - self.TETA1 ) / self.nX
        self.dZ = ( self.Z2 - self.Z1 ) / self.nZ
        self.dX = self.dTETA / self.betha_p
        self.XZ = np.zeros(self.nZ)
        self.XZ[0] = self.Z1 + 0.5 * self.dZ
        self.XTETA = np.zeros(self.nX)
        self.XTETA[0] = self.TETA1 + 0.5 * self.dTETA

        self.speed = speed * np.pi/30
        self.choice_CAIMP = choice_CAIMP
        self.op_key = [*choice_CAIMP][0]

        self.Coefs_D = Coefs_D

        ##### Center shaft speed
        xpt = 0  
        ypt = 0
        self.xpt = self.ypt = xpt
        # --------------------------------------------------------------------------
        
        # Interpolation coefficients
        lubricant_properties = self.lub_selector()
        T_1 = lubricant_properties["temp1"]
        T_2 = lubricant_properties["temp2"]
        mi_1 = lubricant_properties["viscosity1"]
        mi_2 = lubricant_properties["viscosity2"]
        self.rho = lubricant_properties["lube_density"]
        self.Cp = lubricant_properties["lube_cp"]
        self.kt = lubricant_properties["lube_conduct"]

        self.b_b = np.log(mi_1/mi_2)*1/(T_1-T_2)
        self.a_a = mi_1/(np.exp(T_1*self.b_b))

        self.mi0 = self.a_a*np.exp(self.T0*self.b_b) #reference viscosity
        # self.T0 = self.T0 + 273.15

        # pre_processing

        for ii in range(1, self.nZ):
            self.XZ[ii] = self.XZ[ii-1] + self.dZ

        for jj in range(1, self.nX):
            self.XTETA[jj] = self.XTETA[jj-1] +self.dTETA
        pass

        self.dimForca = 1/ ( self.Cr ** 2 / ( self.Rp ** 3 * self.mi0 * self.speed * self.L))
    

    def run(self):

        ############### Define parameters ##############

        self.PPdim = np.zeros((self.nZ,self.nX,self.npad))
        self.H0 = np.zeros((self.nZ,self.nX,self.npad))
        self.H = np.zeros((self.nZ,self.nX))
        self.P = np.zeros((self.nZ,self.nX))
        self.PP = np.zeros((self.nZ, self.nX, self.npad))
        self.h_pivot = np.zeros((self.npad))
        self.TT_i = self.T0 * np.ones((self.nZ,self.nX,self.npad)) #Inicial 3D - temperature field
        self.dPdX = np.zeros((self.nZ,self.nX))
        self.dPdZ = np.zeros((self.nZ,self.nX))
        self.Reyn = np.zeros((self.nZ,self.nX,self.npad))
        self.mi_turb = 1.3*np.ones((self.nZ,self.nX,self.npad)) #Turbulence
        self.Fx = np.zeros((self.npad))
        self.Fy = np.zeros((self.npad))
        self.Mj = np.zeros((self.npad))
        self.Mj_new = np.zeros((self.npad))
        self.F1 = np.zeros((self.npad))
        self.F1_new = np.zeros((self.npad))
        self.F2 = np.zeros((self.npad))
        self.F2_new = np.zeros((self.npad))
        self.P_bef = np.zeros((self.nZ,self.nX))


        if "print" in [*self.choice_CAIMP[self.op_key]] and "progress" in self.choice_CAIMP[self.op_key]["print"]:
            
            self.progress = True
                
        else:

            self.progress = False

        if "calc_EQ" in self.op_key:

            self.x0 = self.choice_CAIMP["calc_EQ"]["init_guess"]
            self.Wx = self.choice_CAIMP["calc_EQ"]["load"][0]
            self.Wy = self.choice_CAIMP["calc_EQ"]["load"][1]
            self.WX = self.Wx * ( self.Cr ** 2 / ( self.Rp ** 3 * self.mi0 * self.speed * self.L)) # Loading - X direction [dimensionless]
            self.WY = self.Wy * ( self.Cr ** 2 / ( self.Rp ** 3 * self.mi0 * self.speed * self.L)) # Loading - Y direction [dimensionless]

        elif "impos_EQ" in self.op_key:
            
            self.x0 = (self.choice_CAIMP["impos_EQ"]["ent_angle"])
            self.WX = self.WY = 0
    
        maxP, medP, maxT, medT, h_pivot0, ecc = self.PandTsolution()
        
        if self.Coefs_D is not None:
        
            self.coeffs_din()

            if "show_coef" in self.Coefs_D and self.Coefs_D["show_coef"]:
                print(f"\n>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<")
                print(f'      Kxx: ', self.Kxx)
                print(f'      Kyy: ', self.Kyy)
                print(f'      Cxx: ', self.Cxx)
                print(f'      Cyy: ', self.Cyy)
                print(f'      Kxy: ', self.Kxy)
                print(f'      Cxy: ', self.Cxy)
                print(f'      Kyx: ', self.Kyx)
                print(f'      Cyx: ', self.Cyx)

                print(f">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<")


        description = [
                    f"\n>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<\n",
                    f"      Pmax: {maxP}\n",
                    f"      Tmax: {maxT}\n",
                    f"      Eccentricity: {ecc}\n",
                    f"      h pivot: {h_pivot0}\n",
                    f">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<\n",
                ]   
                        
        for line in description:
            print(line[:-1])
        
    def PandTsolution(self):

        if "calc_EQ" in self.op_key:

            # Equilibrium position optimization

            x = scipy.optimize.fmin(
                self.HDEequilibrium,                  
                self.x0,
                xtol=1e-4,
                ftol=1e-2,
                maxiter=100000,
                maxfun=100000,
                full_output=0,
                disp=self.progress,
                retall=0,
                callback=None,
                initial_simplex=None,
            )

            ##### Dimensionless center shaft coordinates
            xx = x[0] 
            yy = x[1]
            self.xdin = x

            if "result" in self.choice_CAIMP["calc_EQ"]["print"]:
                print(x)

        elif "impos_EQ" in self.op_key:

            # Equilibrium position optimization
            x = scipy.optimize.fmin(
                self.HDEequilibrium,                  
                self.x0,
                xtol=1e-4,
                ftol=1e-2,
                maxiter=100000,
                maxfun=100000,
                full_output=0,
                disp=self.progress,
                retall=0,
                callback=None,
                initial_simplex=None,
            )
            
            # eq_0 = self.choice_CAIMP["impos_EQ"]["pos_EQ"][0]
            # eq_1 = self.choice_CAIMP["impos_EQ"]["pos_EQ"][1]

            # self.xx = (
            #     eq_0
            #     * self.Cr
            #     * np.cos(eq_1)
            # )
            # self.yy = (
            #     eq_0
            #     * self.Cr
            #     * np.sin(eq_1)
            # )
            
            # self.xryr = np.dot(
            #     [
            #         [np.cos(self.sigma[0]), np.sin(self.sigma[0])],
            #         [-np.sin(self.sigma[0]), np.cos(self.sigma[0])],
            #     ],
            #     [[self.xx], [self.yy]],
            # )
            
            # alpha_max = self.alpha()        
            # x = scipy.optimize.fminbound(
            #     self.HDEequilibrium,                  
            #     1e-8,
            #     alpha_max,
            #     xtol=1e-4,
            #     maxfun=100000,
            #     full_output=0,
            #     disp=self.progress,
            # )

            self.xdin = np.zeros((self.npad+2))
            self.xdin = self.choice_CAIMP["impos_EQ"]["pos_EQ"] + list(x)

            if "result" in self.choice_CAIMP["impos_EQ"]["print"]:
                print(x)
            
        psi_pad = np.zeros((self.npad))

        for kp in range(0, self.npad):
            psi_pad[kp] = self.xdin[kp+2] # Tilting angles of each pad

        nk = (self.nX) * (self.nZ)

        tol_T = 0.1 #Celsius degree

        for n_p in range(0, self.npad):

            T_new = self.TT_i[:,:,n_p]

            T_i = 1.1 * T_new

            cont_Temp = 0

            while abs((T_new - T_i).max()) >= tol_T:

                cont_Temp = cont_Temp + 1

                T_i = np.array(T_new)

                mi_i = (
                    self.a_a * np.exp(self.b_b*T_i )
                    )  # [Pa.s] 

                MI = mi_i * 1/self.mi0 #Dimensionless viscosity field

                k = 0 #vectorization index

                Mat_coef = np.zeros((nk, nk))

                b = np.zeros(nk)


                # transformation of coordinates - inertial to pivot referential
            
                xryr, xryrpt, xr, yr, xrpt, yrpt = self.xr_fun(n_p, self.xdin[0], self.xdin[1])

                alpha = psi_pad[n_p]
                alphapt = 0

                for ii in range(0, self.nZ):

                    for jj in range(0, self.nX):

                        TETAe = self.XTETA[jj] + 0.5 * self.dTETA
                        TETAw = self.XTETA[jj] - 0.5 * self.dTETA

                        hP = ( self.Rp - self.Rs - ( np.sin( self.XTETA[jj]) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( self.XTETA[jj] ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

                        he = ( self.Rp - self.Rs - ( np.sin( TETAe ) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( TETAe ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

                        hw = ( self.Rp - self.Rs - ( np.sin( TETAw ) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( TETAw ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

                        hn = hP

                        hs = hP

                        hpt = -(1 / (self.Cr * self.speed)) * (np.cos(self.XTETA[jj]) * xrpt + np.sin(self.XTETA[jj]) * yrpt + np.sin(self.XTETA[jj]) * (self.Rp + self.tpad) * alphapt)

                        self.H[ii,jj] = hP

                        if jj == 0 and ii == 0:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = MI[ii, jj]
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = MI[ii, jj]

                        if jj == 0 and ii > 0 and ii < self.nZ-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = MI[ii, jj]
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if jj == 0 and ii == self.nZ-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = MI[ii, jj]
                            MI_n = MI[ii, jj]
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if ii == 0 and jj > 0 and jj < self.nX-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = MI[ii, jj]

                        if jj > 0 and jj < self.nX-1 and ii > 0 and ii < self.nZ-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if ii == self.nZ-1 and jj > 0 and jj < self.nX-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = MI[ii, jj]
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if ii == 0 and jj == self.nX-1:
                            MI_e = MI[ii, jj]
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = MI[ii, jj]

                        if jj == self.nX-1 and ii > 0 and ii < self.nZ-1:
                            MI_e = MI[ii, jj]
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if jj == self.nX-1 and ii == self.nZ-1:
                            MI_e = MI[ii, jj]
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = MI[ii, jj]
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        CE = 1 / (self.betha_p ** 2) * (he ** 3 / ( 12 * MI_e ) ) * self.dZ / self.dX
                        CW = 1 / (self.betha_p ** 2) * (hw ** 3 / ( 12 * MI_w ) ) * self.dZ / self.dX
                        CN = ( self.Rp / self.L ) ** 2 * ( self.dX / self.dZ ) * ( hn ** 3 / ( 12 * MI_n ) ) 
                        CS = ( self.Rp / self.L ) ** 2 * ( self.dX / self.dZ ) * ( hs ** 3 / ( 12 * MI_s ) )
                        CP = - ( CE + CW + CN + CS )
                        B = ( self.Rs / ( 2 * self.Rp * self.betha_p ) ) * self.dZ * ( he - hw ) + hpt * self.dX * self.dZ
                        b[k] = B

                        # Mat_coef determination depending on its mesh localization
                        if ii == 0 and jj == 0:
                            Mat_coef[k, k] = CP - CS - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + self.nX] = CN

                        if ii == 0 and jj > 0 and jj < self.nX - 1:
                            Mat_coef[k, k] = CP - CS
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + self.nX] = CN

                        if ii == 0 and jj == self.nX - 1:
                            Mat_coef[k, k] = CP - CE - CS
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + self.nX] = CN

                        if jj == 0 and ii > 0 and ii < self.nZ - 1:
                            Mat_coef[k, k] = CP - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - self.nX] = CS
                            Mat_coef[k, k + self.nX] = CN
                        
                        if ii > 0 and ii < self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                            Mat_coef[k, k] = CP
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.nX] = CS
                            Mat_coef[k, k + self.nX] = CN
                            Mat_coef[k, k + 1] = CE

                        if jj == self.nX - 1 and ii > 0 and ii < self.nZ - 1:
                            Mat_coef[k, k] = CP - CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.nX] = CS
                            Mat_coef[k, k + self.nX] = CN

                        if jj == 0 and ii == self.nZ - 1:
                            Mat_coef[k, k] = CP - CN - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - self.nX] = CS

                        if ii == self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                            Mat_coef[k, k] = CP - CN
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.nX] = CS

                        if ii == self.nZ - 1 and jj == self.nX - 1:
                            Mat_coef[k, k] = CP - CE - CN
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.nX] = CS

                        k = k + 1

                # Pressure field solution ==============================================================
                p = np.linalg.solve(Mat_coef, b)

                cont = 0


                for i in np.arange(self.nZ):
                    for j in np.arange(self.nX):

                        self.P[i, j] = p[cont]
                        cont = cont + 1

                        if self.P[i, j] < 0:
                            self.P[i, j] = 0

            if np.max(self.P) > np.max(self.P_bef):
                self.pad_in = n_p 
                self.P_bef = self.P

            self.PP[:,:,n_p] = self.P
            self.H0[:,:,n_p] = self.H
            self.TT_i[:,:,n_p] = T_new
            self.PPdim = self.PP * ( self.mi0 * self.speed * self.Rp ** 2) / (self.Cr ** 2)
            self.h_pivot[n_p] = self.Cr * ( self.Rp - self.Rs - ( np.sin( 0 ) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( 0 ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

        ##### Outputs
        maxP = self.PP[:,:,self.pad_in].max() * ( self.mi0 * self.speed * self.Rp ** 2) / (self.Cr ** 2)
        medP = self.PP[:,:,self.pad_in].mean() * ( self.mi0 * self.speed * self.Rp ** 2) / (self.Cr ** 2)
        maxT = self.TT_i[:,:,self.pad_in].max()
        medT = self.TT_i[:,:,self.pad_in].mean()
        h_pivot0 = self.h_pivot[self.pad_in]
        ecc = np.sqrt( self.xdin[0] ** 2 + self.xdin[1] ** 2) / self.Cr
        
        return maxP, medP, maxT, medT, h_pivot0, ecc
    
    def coeffs_din(self):
        
        delFx = np.zeros(self.npad)
        delFy = np.zeros(self.npad)
        delMj = np.zeros(self.npad)

        kxx = np.zeros(self.npad)
        ktt = np.zeros(self.npad)
        kyy = np.zeros(self.npad)
        kxt = np.zeros(self.npad)
        ktx = np.zeros(self.npad)
        kyx = np.zeros(self.npad)
        kxy = np.zeros(self.npad)
        kyt = np.zeros(self.npad)
        kty = np.zeros(self.npad)
        self.K = np.zeros((self.npad,3,3))

        cxx = np.zeros(self.npad)
        ctt = np.zeros(self.npad)
        cyy = np.zeros(self.npad)
        cxt = np.zeros(self.npad)
        ctx = np.zeros(self.npad)
        cyx = np.zeros(self.npad)
        cxy = np.zeros(self.npad)
        cyt = np.zeros(self.npad)
        cty = np.zeros(self.npad)
        self.C = np.zeros((self.npad,3,3))
        self.Sjpt = np.zeros((self.npad,3,3), dtype = 'complex_')
        self.Aj = np.zeros((self.npad,2,2), dtype = 'complex_')
        self.Hj = np.zeros((self.npad,2,1), dtype = 'complex_')
        self.Bj = np.zeros((self.npad,1,1), dtype = 'complex_')
        self.Vj = np.zeros((self.npad,1,2), dtype = 'complex_')
        self.Sj = np.zeros((self.npad,2,2), dtype = 'complex_')
        self.Tj = np.zeros((self.npad,2,2))
        self.Ptj = np.zeros((self.npad,3,3))

        self.Sw = np.zeros((2,2), dtype = 'complex_')
        self.Kkj = np.zeros((3,3), dtype = 'complex_')
        self.Ccj = np.zeros((3,3), dtype = 'complex_')


        psi_pad = np.zeros((self.npad))
        nk = (self.nX) * (self.nZ)

        dE = 0.005 * self.Cr   # Space Perturbation
        dEv = self.speed * dE  # Speed Perturbation
        dEalpha = 0.0006 * 0.01 # Angular Perturbation
        dEalphav = self.speed * dEalpha # Angular Speed Perturbation

        tol_T = 0.1 # Celsius degree

        for a_p in range(0,4):
            for n_p in range(0, self.npad):

                xxcoef = self.xdin[:2]

                for kp in range(0, self.npad):
                    psi_pad[kp] = self.xdin[kp+2] # Tilting angles of each pad


                T_new = self.TT_i[:,:,n_p]

                T_i = 1.1 * T_new

                cont_Temp = 0

                while abs((T_new - T_i).max()) >= tol_T:

                    cont_Temp = cont_Temp + 1

                    T_i = np.array(T_new)

                    mi_i = (
                        self.a_a * np.exp(self.b_b*T_i )
                        )  # [Pa.s] 

                    MI = mi_i * 1/self.mi0 #Dimensionless viscosity field

                    k = 0 #vectorization index

                    Mat_coef = np.zeros((nk, nk))

                    b = np.zeros(nk)

                    # transformation of coordinates - inertial to pivot referential
                    xryr, xryrpt, xr, yr, xrpt, yrpt = self.xr_fun(n_p, xxcoef[0], xxcoef[1])


                    alphapt = 0

                    if a_p == 0:
                        xr = xr + dE

                    elif a_p == 1:
                        psi_pad[n_p] = psi_pad[n_p] + dEalpha
                    
                    if a_p == 2:
                        xrpt = xrpt + dEv
                    
                    if a_p == 3:
                        alphapt = alphapt + dEalphav

                    alpha = psi_pad[n_p]

                    for ii in range(0, self.nZ):

                        for jj in range(0, self.nX):

                            TETAe = self.XTETA[jj] + 0.5 * self.dTETA
                            TETAw = self.XTETA[jj] - 0.5 * self.dTETA

                            hP = ( self.Rp - self.Rs - ( np.sin( self.XTETA[jj]) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( self.XTETA[jj] ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

                            he = ( self.Rp - self.Rs - ( np.sin( TETAe ) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( TETAe ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

                            hw = ( self.Rp - self.Rs - ( np.sin( TETAw ) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( TETAw ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

                            hn = hP

                            hs = hP

                            hpt = -(1 / (self.Cr * self.speed)) * (np.cos(self.XTETA[jj]) * xrpt + np.sin(self.XTETA[jj]) * yrpt + np.sin(self.XTETA[jj]) * (self.Rp + self.tpad) * alphapt)

                            self.H[ii,jj] = hP

                            if jj == 0 and ii == 0:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = MI[ii, jj]
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = MI[ii, jj]

                            if jj == 0 and ii > 0 and ii < self.nZ-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = MI[ii, jj]
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if jj == 0 and ii == self.nZ-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = MI[ii, jj]
                                MI_n = MI[ii, jj]
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if ii == 0 and jj > 0 and jj < self.nX-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = MI[ii, jj]

                            if jj > 0 and jj < self.nX-1 and ii > 0 and ii < self.nZ-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if ii == self.nZ-1 and jj > 0 and jj < self.nX-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = MI[ii, jj]
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if ii == 0 and jj == self.nX-1:
                                MI_e = MI[ii, jj]
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = MI[ii, jj]

                            if jj == self.nX-1 and ii > 0 and ii < self.nZ-1:
                                MI_e = MI[ii, jj]
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if jj == self.nX-1 and ii == self.nZ-1:
                                MI_e = MI[ii, jj]
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = MI[ii, jj]
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            CE = 1 / (self.betha_p ** 2) * (he ** 3 / ( 12 * MI_e ) ) * self.dZ / self.dX
                            CW = 1 / (self.betha_p ** 2) * (hw ** 3 / ( 12 * MI_w ) ) * self.dZ / self.dX
                            CN = ( self.Rp / self.L ) ** 2 * ( self.dX / self.dZ ) * ( hn ** 3 / ( 12 * MI_n ) ) 
                            CS = ( self.Rp / self.L ) ** 2 * ( self.dX / self.dZ ) * ( hs ** 3 / ( 12 * MI_s ) )
                            CP = - ( CE + CW + CN + CS )
                            B = ( self.Rs / ( 2 * self.Rp * self.betha_p ) ) * self.dZ * ( he - hw ) + hpt * self.dX * self.dZ
                            b[k] = B

                            # Mat_coef determination depending on its mesh localization
                            if ii == 0 and jj == 0:
                                Mat_coef[k, k] = CP - CS - CW
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k + self.nX] = CN

                            if ii == 0 and jj > 0 and jj < self.nX - 1:
                                Mat_coef[k, k] = CP - CS
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k + self.nX] = CN

                            if ii == 0 and jj == self.nX - 1:
                                Mat_coef[k, k] = CP - CE - CS
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k + self.nX] = CN

                            if jj == 0 and ii > 0 and ii < self.nZ - 1:
                                Mat_coef[k, k] = CP - CW
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - self.nX] = CS
                                Mat_coef[k, k + self.nX] = CN
                            
                            if ii > 0 and ii < self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                                Mat_coef[k, k] = CP
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - self.nX] = CS
                                Mat_coef[k, k + self.nX] = CN
                                Mat_coef[k, k + 1] = CE

                            if jj == self.nX - 1 and ii > 0 and ii < self.nZ - 1:
                                Mat_coef[k, k] = CP - CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - self.nX] = CS
                                Mat_coef[k, k + self.nX] = CN

                            if jj == 0 and ii == self.nZ - 1:
                                Mat_coef[k, k] = CP - CN - CW
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - self.nX] = CS

                            if ii == self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                                Mat_coef[k, k] = CP - CN
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - self.nX] = CS

                            if ii == self.nZ - 1 and jj == self.nX - 1:
                                Mat_coef[k, k] = CP - CE - CN
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - self.nX] = CS

                            k = k + 1

                    # Pressure field solution ==============================================================
                    p = np.linalg.solve(Mat_coef, b)

                    cont = 0


                    for i in np.arange(self.nZ):
                        for j in np.arange(self.nX):

                            self.P[i, j] = p[cont]
                            cont = cont + 1

                            if self.P[i, j] < 0:
                                self.P[i, j] = 0

                    ##################################### Energy equation ################################
                    ################################### Pressure Gradients ###############################

                    nk = (self.nX) * (self.nZ)
                    Mat_coef_t = np.zeros((nk, nk))
                    b_t = np.zeros(nk)
                    test_diag = np.zeros(nk)

                    k = 0 #vectorization temperature index

                    for ii in range(0, self.nZ):
                        for jj in range(0, self.nX):

                            if jj == 0 and ii == 0:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj] ) / (self.dZ)

                            if jj == 0 and ii > 0 and ii < self.nZ-1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                            if jj == 0 and ii == self.nZ-1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                                self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / (self.dZ)

                            if ii == 0 and jj > 0 and jj < self.nX-1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                            if jj > 0 and jj < self.nX-1 and ii > 0 and ii < self.nZ-1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                            if ii == self.nZ-1 and jj > 0 and jj < self.nX-1:
                                self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                                self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / (self.dZ)

                            if ii == 0 and jj == self.nX-1:
                                self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / (self.dX)
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                            if jj == self.nX-1 and ii > 0 and ii < self.nZ-1:
                                self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / (self.dX)
                                self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                            if jj == self.nX-1 and ii == self.nZ-1:
                                self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / (self.dX)
                                self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / (self.dZ)

                ######################### Turbulence (Eddy viscosity) #########################
                            HP = self.H[ii, jj]

                            mi_p = MI[ii, jj]
                            
                            self.Reyn[ii, jj, n_p] = self.rho * self.speed * self.Rs * ( HP / self.L ) * self.Cr / ( self.mi0 * mi_p )

                            if self.Reyn[ii, jj, n_p] <= 500:
                                        
                                delta_turb = 0
                                        
                            elif self.Reyn[ii, jj, n_p] > 400 and self.Reyn[ii, jj, n_p] <= 1000:
                                            
                                delta_turb = 1 - ( ( 1000 - self.Reyn[ii, jj, n_p] ) / 500 ) ** ( 1 / 8 )
                                            
                            elif self.Reyn[ii, jj, n_p] > 1000:
                                        
                                delta_turb = 1

                            dudx = ( ( ( HP / self.mi_turb[ii, jj, n_p]) * self.dPdX[ii, jj] ) - ( self.speed / HP ) )
                                        
                            dwdx = ( ( HP / self.mi_turb[ii, jj, n_p] ) * self.dPdZ[ii, jj] )
                                        
                            tal = self.mi_turb[ii, jj, n_p] * np.sqrt( ( dudx ** 2 ) + ( dwdx ** 2 ) )
                                        
                            ywall = ( ( HP * self.Cr * 2 ) / ( self.mi0 * self.mi_turb[ii, jj, n_p] / self.rho ) ) * ( ( abs( tal ) / self.rho ) ** 0.5 )
                                        
                            emv = 0.4 * ( ywall - ( 10.7 * np.tanh( ywall / 10.7 ) ) )
                                        
                            self.mi_turb[ii, jj, n_p] = mi_p * ( 1 + ( delta_turb * emv ) )
                                        
                            mi_t = self.mi_turb[ii, jj, n_p]

                            #### Coefficients for the energy equation
                            aux_up = 1
                            if self.XZ[ii] < 0 :
                                aux_up = 0

                            AE =   - ( self.kt * HP * self.dZ ) / ( self.rho * self.Cp * self.speed * ( ( self.betha_p * self.Rp ) ** 2 ) * self.dX ) 

                            AW =  ( ( HP ** 3 ) * self.dPdX[ii, jj] * self.dZ ) / (12 * mi_t * (self.betha_p ** 2 ) )  - ( (self.Rs * HP * self.dZ ) / ( 2 * self.Rp * self.betha_p ) ) -  ( self.kt * HP * self.dZ ) / ( self.rho * self.Cp * self.speed * ( ( self.betha_p * self.Rp ) ** 2 ) * self.dX ) 

                            AN_1 = (aux_up - 1) * ( ( ( self.Rp ** 2 ) * ( HP ** 3 ) * self.dPdZ[ii, jj] * self.dX ) / ( 12 * ( self.L ** 2 ) * mi_t ) )

                            AS_1 = (aux_up) *  ( ( ( self.Rp ** 2 ) * ( HP ** 3 ) * self.dPdZ[ii, jj] * self.dX ) / ( 12 * (self.L ** 2 ) * mi_t ) )

                            AN_2 = - ( self.kt * HP * self.dX) / ( self.rho * self.Cp * self.speed * ( self.L ** 2 ) * self.dZ )

                            AS_2 = - ( self.kt * HP * self.dX ) / ( self.rho * self.Cp * self.speed * ( self.L ** 2 ) * self.dZ ) 

                            AN = AN_1 + AN_2

                            AS = AS_1 + AS_2

                            AP = - ( AE + AW + AN + AS )

                            auxB_t = ( self.speed * self.mi0 ) / ( self.rho * self.Cp * self.Tcub * self.Cr )

                            B_tG = ( self.mi0 * self.speed * ( self.Rs ** 2 ) * self.dX * self.dZ * self.P[ii, jj] * hpt ) / ( self.rho * self.Cp * self.T0 * ( self.Cr ** 2 ) )

                            B_tH = ( ( self.speed * self.mi0 * ( hpt ** 2 ) * 4 * mi_t * self.dX * self.dZ ) / ( self.rho * self.Cp * self.T0  * 3 * HP ) )

                            B_tI = auxB_t * ( 1 * mi_t * ( self.Rs ** 2 ) *self. dX * self.dZ ) / ( HP * self.Cr )

                            B_tJ = auxB_t * ( ( self.Rp ** 2 ) * ( HP ** 3 ) * ( self.dPdX[ii, jj] ** 2 ) * self.dX * self.dZ ) / ( 12 * self.Cr * ( self.betha_p ** 2 ) * mi_t )

                            B_tK = auxB_t * ( ( self.Rp ** 4 ) * ( HP ** 3 ) * ( self.dPdZ[ii, jj] ** 2 ) * self.dX * self.dZ ) / ( 12 * self.Cr * ( self.L ** 2 ) * mi_t)

                            B_t = B_tG + B_tH + B_tI + B_tJ + B_tK

                            b_t[k] = B_t 

                            if ii == 0 and jj == 0:
                                Mat_coef_t[k, k] = AP + AS - AW
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k + self.nX] = AN
                                b_t[k] = b_t[k] - 2 * AW * ( self.Tcub / self.T0 )                           
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k + self.nX])           
                            
                            if ii == 0 and jj > 0 and jj < self.nX - 1: 
                                Mat_coef_t[k, k] = AP + AS
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k + self.nX] = AN
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - 1] + Mat_coef_t[k, k + self.nX])   
                            
                            if ii == 0 and jj == self.nX - 1:
                                Mat_coef_t[k, k] = AP + AE + AS
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k + self.nX] = AN            
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k - 1] + Mat_coef_t[k, k + self.nX])    
                            
                            if jj == 0 and ii > 0 and ii < self.nZ - 1:
                                Mat_coef_t[k, k] = AP - AW
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k - self.nX] = AS
                                Mat_coef_t[k, k + self.nX] = AN
                                b_t[k] = b_t[k] - 2 * AW * ( self.Tcub / self.T0 )
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - self.nX] + Mat_coef_t[k, k + self.nX])           
                            
                            if ii > 0 and ii < self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                                Mat_coef_t[k, k] = AP
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k - self.nX] = AS
                                Mat_coef_t[k, k + self.nX] = AN
                                Mat_coef_t[k, k + 1] = AE
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - self.nX] + Mat_coef_t[k, k + self.nX])   
                            
                            if jj == self.nX - 1 and ii > 0 and ii < self.nZ - 1:
                                Mat_coef_t[k, k] = AP + AE
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k - self.nX] = AS
                                Mat_coef_t[k, k + self.nX] = AN
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - self.nX] + Mat_coef_t[k, k + self.nX])           
                            
                            if jj == 0 and ii == self.nZ - 1:
                                Mat_coef_t[k, k] = AP + AN - AW
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k - self.nX] = AS                        
                                b_t[k] = b_t[k] - 2 * AW * ( self.Tcub / self.T0 )
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - self.nX])
                            
                            if ii == self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                                Mat_coef_t[k, k] = AP + AN
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k - self.nX] = AS      
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - self.nX])            
                            
                            if ii == self.nZ - 1 and jj == self.nX - 1:
                                Mat_coef_t[k, k] = AP + AE + AN
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k - self.nX] = AS
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - self.nX])
                            k = k + 1

                        ################## Solution of temperature field ####################

                    t = np.linalg.solve(Mat_coef_t, b_t)

                    cont = 0

                    for i in np.arange(self.nZ):
                        for j in np.arange(self.nX):

                            T_new[i, j]= self.T0 * t[cont]

                            cont=cont+1

                # Hydrodynamic forces =================================================================
                P_dimen =  self.P * ( self.mi0 * self.speed * self.Rp ** 2) / (self.Cr ** 2)
                auxF1 = np.zeros((self.nZ,self.nX))
                auxF2 = np.zeros((self.nZ,self.nX))

                for ni in np.arange(self.nZ):
                    auxF1[ni,:] = np.cos(self.XTETA)
                    auxF2[ni,:] = np.sin(self.XTETA) 
                
                YtetaF1 = self.P * auxF1
                F1teta = np.trapz(YtetaF1,self.XTETA)
                self.F1_new[n_p] = -np.trapz(F1teta,self.XZ)

                YtetaF2 = self.P * auxF2
                F2teta = np.trapz(YtetaF2,self.XTETA)
                self.F2_new[n_p] = -np.trapz(F2teta,self.XZ)

                self.Mj_new[n_p] = self.F2_new[n_p] * (self.Rp + self.tpad)

         ## Dynamic Coefficients Calculation
                    
                delFx[n_p] =  self.F1_new[n_p] - self.F1[n_p]
                delMj[n_p] =  self.Mj_new[n_p] - self.Mj[n_p] 
                delFy[n_p] =  self.F2_new[n_p] - self.F2[n_p] 

        # X-axis perturbation
                if a_p == 0:
  
                    kxx[n_p] = delFx[n_p]/(dE) * self.dimForca
                    ktx[n_p] = delMj[n_p]/(dE) * self.dimForca 
                    self.K[n_p,0,0] = kxx[n_p]
                    self.K[n_p,2,0] = ktx[n_p]

        # Angular (pad) perturbation
                elif a_p == 1:

                    ktt[n_p] = delMj[n_p]/(dEalpha) * self.dimForca
                    kxt[n_p] = delFx[n_p]/(dEalpha) * self.dimForca

                    # Y-coefficients

                    kyy[n_p] =  ktt[n_p]/((self.Rp + self.tpad)**2)
                    kyx[n_p] = - ktx[n_p]/((self.Rp + self.tpad))
                    kxy[n_p] = - kxt[n_p]/((self.Rp + self.tpad))
                    kyt[n_p] = - ktt[n_p]/((self.Rp + self.tpad))
                    kty[n_p] = - ktt[n_p]/((self.Rp + self.tpad)) 
                    self.K[n_p,2,2] = ktt[n_p]
                    self.K[n_p,0,2] = kxt[n_p]
                    self.K[n_p,1,1] = kyy[n_p]
                    self.K[n_p,1,0] = kyx[n_p]
                    self.K[n_p,0,1] = kxy[n_p]
                    self.K[n_p,1,2] = kyt[n_p]
                    self. K[n_p,2,1] = kty[n_p]

        # x-axis speed perturbation
                elif a_p == 2:

                    cxx[n_p] = delFx[n_p]/(dEv) * self.dimForca 
                    ctx[n_p] = delMj[n_p]/(dEv) * self.dimForca
                    self.C[n_p,0,0] =  cxx[n_p]
                    self.C[n_p,2,0] =  ctx[n_p]
             
        # Angular speed perturbation
                elif a_p == 3:

                    ctt[n_p] = delMj[n_p]/(dEalphav) * self.dimForca 
                    cxt[n_p] = delFx[n_p]/(dEalphav) * self.dimForca 
                    self.C[n_p,2,2] =  ctt[n_p]
                    self.C[n_p,0,2] =  cxt[n_p]

                    # Y-coefficients

                    cyy[n_p] =  ctt[n_p]/((self.Rp + self.tpad)**2)
                    cyx[n_p] = - ctx[n_p]/((self.Rp + self.tpad))
                    cxy[n_p] = - cxt[n_p]/((self.Rp + self.tpad))
                    cyt[n_p] = - ctt[n_p]/((self.Rp + self.tpad))
                    cty[n_p] = - ctt[n_p]/((self.Rp + self.tpad))
                    self.C[n_p,2,2] = ctt[n_p]
                    self.C[n_p,0,2] = cxt[n_p]
                    self.C[n_p,1,1] = cyy[n_p]
                    self.C[n_p,1,0] = cyx[n_p]
                    self.C[n_p,0,1] = cxy[n_p]
                    self.C[n_p,1,2] = cyt[n_p]
                    self.C[n_p,2,1] = cty[n_p]

                    # Coefficient matrix reduction

                    self.Sjpt[n_p] = self.K[n_p] + self.C[n_p] * self.speed * 1j

                    self.Aj[n_p] = [[self.Sjpt[n_p,0,0], self.Sjpt[n_p,0,1]],
                                    [self.Sjpt[n_p,1,0], self.Sjpt[n_p,1,1]]]
                    self.Hj[n_p] = [[self.Sjpt[n_p,0,2]],
                                    [self.Sjpt[n_p,1,2]]]      
                    self.Bj[n_p] = [[self.Sjpt[n_p,2,2]]]
                    self.Vj[n_p] = [[self.Sjpt[n_p,2,0], self.Sjpt[n_p,2,1]]]

                    self.Sj[n_p] = self.Aj[n_p] - self.Hj[n_p] * np.linalg.inv(self.Bj[n_p]) * self. Vj[n_p]

                    self.Tj[n_p] = [[np.cos(psi_pad[n_p] + self.sigma[n_p]), np.sin(psi_pad[n_p] + self.sigma[n_p])],
                                   [-np.sin(psi_pad[n_p] + self.sigma[n_p]), np.cos(psi_pad[n_p] + self.sigma[n_p])]]

                    self.Sw = self.Sw + ((np.transpose(self.Tj[n_p])) * self.Sj[n_p]) * self.Tj[n_p]
        
        Kr = np.real(self.Sw)
        Cr = np.imag(self.Sw) / self.speed
        self.Kxx, self.Kyy, self.Kxy, self.Kyx = Kr[0,0], Kr[1,1], Kr[0,1], Kr[1,0]
        self.Cxx, self.Cyy, self.Cxy, self.Cyx = Cr[0,0], Cr[1,1], Cr[0,1], Cr[1,0] 
                    
        return self.Kxx, self.Kyy, self.Kxy, self.Kyx, self.Cxx, self.Cyy, self.Cxy, self.Cyx

    def HDEequilibrium(self,x):
        global Pressure, Temperature
        # x = [x,]
        if "calc_EQ" in self.op_key:

            ##### Dimensionless center shaft coordinates
            eq_0 = x[0] 
            eq_1 = x[1]
            #####

            psi_pad = np.zeros((self.npad))

            for kp in range(0, self.npad):
                psi_pad[kp] = x[kp+2] # Tilting angles of each pad
        
        elif "impos_EQ" in self.op_key:

            eq_0 = self.choice_CAIMP[self.op_key]["pos_EQ"][0]
            eq_1 = self.choice_CAIMP[self.op_key]["pos_EQ"][1]
            
            psi_pad = np.zeros((self.npad))

            for kp in range(0, self.npad):
                psi_pad[kp] = x[kp] # Tilting angles of each pad

        nk = (self.nX) * (self.nZ)


        tol_T = 0.1 #Celsius degree

        for n_p in range(0, self.npad):

            T_new = self.TT_i[:,:,n_p]

            T_i = 1.1 * T_new

            cont_Temp = 0

            while abs((T_new - T_i).max()) >= tol_T:

                cont_Temp = cont_Temp + 1

                T_i = np.array(T_new)

                mi_i = (
                    self.a_a * np.exp(self.b_b*T_i )
                    )  # [Pa.s] 

                MI = mi_i * 1/self.mi0 #Dimensionless viscosity field

                k = 0 #vectorization index

                Mat_coef = np.zeros((nk, nk))

                b = np.zeros(nk)

                # transformation of coordinates - inertial to pivot referential
                xryr, xryrpt, xr, yr, xrpt, yrpt = self.xr_fun(n_p, eq_0, eq_1)
                
                alpha = psi_pad[n_p]
                alphapt = 0

                for ii in range(0, self.nZ):

                    for jj in range(0, self.nX):

                        TETAe = self.XTETA[jj] + 0.5 * self.dTETA
                        TETAw = self.XTETA[jj] - 0.5 * self.dTETA

                        hP = ( self.Rp - self.Rs - ( np.sin( self.XTETA[jj]) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( self.XTETA[jj] ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

                        he = ( self.Rp - self.Rs - ( np.sin( TETAe ) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( TETAe ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

                        hw = ( self.Rp - self.Rs - ( np.sin( TETAw ) * ( yr + alpha * ( self.Rp + self.tpad ) ) + np.cos( TETAw ) * ( xr + self.Rp - self.Rs - self.Cr ) ) ) / self.Cr

                        hn = hP

                        hs = hP

                        hpt = -(1 / (self.Cr * self.speed)) * (np.cos(self.XTETA[jj]) * xrpt + np.sin(self.XTETA[jj]) * yrpt + np.sin(self.XTETA[jj]) * (self.Rp + self.tpad) * alphapt)

                        self.H[ii,jj] = hP

                        if jj == 0 and ii == 0:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = MI[ii, jj]
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = MI[ii, jj]

                        if jj == 0 and ii > 0 and ii < self.nZ-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = MI[ii, jj]
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if jj == 0 and ii == self.nZ-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = MI[ii, jj]
                            MI_n = MI[ii, jj]
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if ii == 0 and jj > 0 and jj < self.nX-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = MI[ii, jj]

                        if jj > 0 and jj < self.nX-1 and ii > 0 and ii < self.nZ-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if ii == self.nZ-1 and jj > 0 and jj < self.nX-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = MI[ii, jj]
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if ii == 0 and jj == self.nX-1:
                            MI_e = MI[ii, jj]
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = MI[ii, jj]

                        if jj == self.nX-1 and ii > 0 and ii < self.nZ-1:
                            MI_e = MI[ii, jj]
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if jj == self.nX-1 and ii == self.nZ-1:
                            MI_e = MI[ii, jj]
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = MI[ii, jj]
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        CE = 1 / (self.betha_p ** 2) * (he ** 3 / ( 12 * MI_e ) ) * self.dZ / self.dX
                        CW = 1 / (self.betha_p ** 2) * (hw ** 3 / ( 12 * MI_w ) ) * self.dZ / self.dX
                        CN = ( self.Rp / self.L ) ** 2 * ( self.dX / self.dZ ) * ( hn ** 3 / ( 12 * MI_n ) ) 
                        CS = ( self.Rp / self.L ) ** 2 * ( self.dX / self.dZ ) * ( hs ** 3 / ( 12 * MI_s ) )
                        CP = - ( CE + CW + CN + CS )
                        B = ( self.Rs / ( 2 * self.Rp * self.betha_p ) ) * self.dZ * ( he - hw ) + hpt * self.dX * self.dZ
                        b[k] = B

                        # Mat_coef determination depending on its mesh localization
                        if ii == 0 and jj == 0:
                            Mat_coef[k, k] = CP - CS - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + self.nX] = CN

                        if ii == 0 and jj > 0 and jj < self.nX - 1:
                            Mat_coef[k, k] = CP - CS
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + self.nX] = CN

                        if ii == 0 and jj == self.nX - 1:
                            Mat_coef[k, k] = CP - CE - CS
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + self.nX] = CN

                        if jj == 0 and ii > 0 and ii < self.nZ - 1:
                            Mat_coef[k, k] = CP - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - self.nX] = CS
                            Mat_coef[k, k + self.nX] = CN
                        
                        if ii > 0 and ii < self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                            Mat_coef[k, k] = CP
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.nX] = CS
                            Mat_coef[k, k + self.nX] = CN
                            Mat_coef[k, k + 1] = CE

                        if jj == self.nX - 1 and ii > 0 and ii < self.nZ - 1:
                            Mat_coef[k, k] = CP - CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.nX] = CS
                            Mat_coef[k, k + self.nX] = CN

                        if jj == 0 and ii == self.nZ - 1:
                            Mat_coef[k, k] = CP - CN - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - self.nX] = CS

                        if ii == self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                            Mat_coef[k, k] = CP - CN
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.nX] = CS

                        if ii == self.nZ - 1 and jj == self.nX - 1:
                            Mat_coef[k, k] = CP - CE - CN
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - self.nX] = CS

                        k = k + 1

                # Pressure field solution ==============================================================
                p = np.linalg.solve(Mat_coef, b)

                cont = 0


                for i in np.arange(self.nZ):
                    for j in np.arange(self.nX):

                        self.P[i, j] = p[cont]
                        cont = cont + 1

                        if self.P[i, j] < 0:
                            self.P[i, j] = 0


                ##################################### Energy equation ################################
                ################################### Pressure Gradients ###############################

                nk = (self.nX) * (self.nZ)
                Mat_coef_t = np.zeros((nk, nk))
                b_t = np.zeros(nk)
                test_diag = np.zeros(nk)

                k = 0 #vectorization temperature index

                for ii in range(0, self.nZ):
                    for jj in range(0, self.nX):

                        if jj == 0 and ii == 0:
                            self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                            self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj] ) / (self.dZ)

                        if jj == 0 and ii > 0 and ii < self.nZ-1:
                            self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                            self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                        if jj == 0 and ii == self.nZ-1:
                            self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                            self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / (self.dZ)

                        if ii == 0 and jj > 0 and jj < self.nX-1:
                            self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                            self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                        if jj > 0 and jj < self.nX-1 and ii > 0 and ii < self.nZ-1:
                            self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                            self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                        if ii == self.nZ-1 and jj > 0 and jj < self.nX-1:
                            self.dPdX[ii, jj] = (self.P[ii, jj + 1] - self.P[ii, jj]) / (self.dX)
                            self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / (self.dZ)

                        if ii == 0 and jj == self.nX-1:
                            self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / (self.dX)
                            self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                        if jj == self.nX-1 and ii > 0 and ii < self.nZ-1:
                            self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / (self.dX)
                            self.dPdZ[ii, jj] = (self.P[ii + 1, jj] - self.P[ii, jj]) / (self.dZ)

                        if jj == self.nX-1 and ii == self.nZ-1:
                            self.dPdX[ii, jj] = (0 - self.P[ii, jj]) / (self.dX)
                            self.dPdZ[ii, jj] = (0 - self.P[ii, jj]) / (self.dZ)

            ######################### Turbulence (Eddy viscosity) #########################
                        HP = self.H[ii, jj]

                        mi_p = MI[ii, jj]
                        
                        self.Reyn[ii, jj, n_p] = self.rho * self.speed * self.Rs * ( HP / self.L ) * self.Cr / ( self.mi0 * mi_p )

                        if self.Reyn[ii, jj, n_p] <= 500:
                                    
                            delta_turb = 0
                                    
                        elif self.Reyn[ii, jj, n_p] > 400 and self.Reyn[ii, jj, n_p] <= 1000:
                                        
                            delta_turb = 1 - ( ( 1000 - self.Reyn[ii, jj, n_p] ) / 500 ) ** ( 1 / 8 )
                                        
                        elif self.Reyn[ii, jj, n_p] > 1000:
                                    
                            delta_turb = 1

                        dudx = ( ( ( HP / self.mi_turb[ii, jj, n_p]) * self.dPdX[ii, jj] ) - ( self.speed / HP ) )
                                    
                        dwdx = ( ( HP / self.mi_turb[ii, jj, n_p] ) * self.dPdZ[ii, jj] )
                                    
                        tal = self.mi_turb[ii, jj, n_p] * np.sqrt( ( dudx ** 2 ) + ( dwdx ** 2 ) )
                                    
                        ywall = ( ( HP * self.Cr * 2 ) / ( self.mi0 * self.mi_turb[ii, jj, n_p] / self.rho ) ) * ( ( abs( tal ) / self.rho ) ** 0.5 )
                                    
                        emv = 0.4 * ( ywall - ( 10.7 * np.tanh( ywall / 10.7 ) ) )
                                    
                        self.mi_turb[ii, jj, n_p] = mi_p * ( 1 + ( delta_turb * emv ) )
                                    
                        mi_t = self.mi_turb[ii, jj, n_p]

                        #### Coefficients for the energy equation
                        aux_up = 1
                        if self.XZ[ii] < 0 :
                            aux_up = 0

                        AE =   - ( self.kt * HP * self.dZ ) / ( self.rho * self.Cp * self.speed * ( ( self.betha_p * self.Rp ) ** 2 ) * self.dX ) 

                        AW =  ( ( HP ** 3 ) * self.dPdX[ii, jj] * self.dZ ) / (12 * mi_t * (self.betha_p ** 2 ) )  - ( (self.Rs * HP * self.dZ ) / ( 2 * self.Rp * self.betha_p ) ) -  ( self.kt * HP * self.dZ ) / ( self.rho * self.Cp * self.speed * ( ( self.betha_p * self.Rp ) ** 2 ) * self.dX ) 

                        AN_1 = (aux_up - 1) * ( ( ( self.Rp ** 2 ) * ( HP ** 3 ) * self.dPdZ[ii, jj] * self.dX ) / ( 12 * ( self.L ** 2 ) * mi_t ) )

                        AS_1 = (aux_up) *  ( ( ( self.Rp ** 2 ) * ( HP ** 3 ) * self.dPdZ[ii, jj] * self.dX ) / ( 12 * (self.L ** 2 ) * mi_t ) )

                        AN_2 = - ( self.kt * HP * self.dX) / ( self.rho * self.Cp * self.speed * ( self.L ** 2 ) * self.dZ )

                        AS_2 = - ( self.kt * HP * self.dX ) / ( self.rho * self.Cp * self.speed * ( self.L ** 2 ) * self.dZ ) 

                        AN = AN_1 + AN_2

                        AS = AS_1 + AS_2

                        AP = - ( AE + AW + AN + AS )

                        auxB_t = ( self.speed * self.mi0 ) / ( self.rho * self.Cp * self.Tcub * self.Cr )

                        B_tG = ( self.mi0 * self.speed * ( self.Rs ** 2 ) * self.dX * self.dZ * self.P[ii, jj] * hpt ) / ( self.rho * self.Cp * self.T0 * ( self.Cr ** 2 ) )

                        B_tH = ( ( self.speed * self.mi0 * ( hpt ** 2 ) * 4 * mi_t * self.dX * self.dZ ) / ( self.rho * self.Cp * self.T0  * 3 * HP ) )

                        B_tI = auxB_t * ( 1 * mi_t * ( self.Rs ** 2 ) *self. dX * self.dZ ) / ( HP * self.Cr )

                        B_tJ = auxB_t * ( ( self.Rp ** 2 ) * ( HP ** 3 ) * ( self.dPdX[ii, jj] ** 2 ) * self.dX * self.dZ ) / ( 12 * self.Cr * ( self.betha_p ** 2 ) * mi_t )

                        B_tK = auxB_t * ( ( self.Rp ** 4 ) * ( HP ** 3 ) * ( self.dPdZ[ii, jj] ** 2 ) * self.dX * self.dZ ) / ( 12 * self.Cr * ( self.L ** 2 ) * mi_t)

                        B_t = B_tG + B_tH + B_tI + B_tJ + B_tK

                        b_t[k] = B_t 

                        if ii == 0 and jj == 0:
                            Mat_coef_t[k, k] = AP + AS - AW
                            Mat_coef_t[k, k + 1] = AE
                            Mat_coef_t[k, k + self.nX] = AN
                            b_t[k] = b_t[k] - 2 * AW * ( self.Tcub / self.T0 )                           
                            test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k + self.nX])           
                        
                        if ii == 0 and jj > 0 and jj < self.nX - 1: 
                            Mat_coef_t[k, k] = AP + AS
                            Mat_coef_t[k, k + 1] = AE
                            Mat_coef_t[k, k - 1] = AW
                            Mat_coef_t[k, k + self.nX] = AN
                            test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - 1] + Mat_coef_t[k, k + self.nX])   
                        
                        if ii == 0 and jj == self.nX - 1:
                            Mat_coef_t[k, k] = AP + AE + AS
                            Mat_coef_t[k, k - 1] = AW
                            Mat_coef_t[k, k + self.nX] = AN            
                            test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k - 1] + Mat_coef_t[k, k + self.nX])    
                        
                        if jj == 0 and ii > 0 and ii < self.nZ - 1:
                            Mat_coef_t[k, k] = AP - AW
                            Mat_coef_t[k, k + 1] = AE
                            Mat_coef_t[k, k - self.nX] = AS
                            Mat_coef_t[k, k + self.nX] = AN
                            b_t[k] = b_t[k] - 2 * AW * ( self.Tcub / self.T0 )
                            test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - self.nX] + Mat_coef_t[k, k + self.nX])           
                        
                        if ii > 0 and ii < self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                            Mat_coef_t[k, k] = AP
                            Mat_coef_t[k, k - 1] = AW
                            Mat_coef_t[k, k - self.nX] = AS
                            Mat_coef_t[k, k + self.nX] = AN
                            Mat_coef_t[k, k + 1] = AE
                            test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - self.nX] + Mat_coef_t[k, k + self.nX])   
                        
                        if jj == self.nX - 1 and ii > 0 and ii < self.nZ - 1:
                            Mat_coef_t[k, k] = AP + AE
                            Mat_coef_t[k, k - 1] = AW
                            Mat_coef_t[k, k - self.nX] = AS
                            Mat_coef_t[k, k + self.nX] = AN
                            test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - self.nX] + Mat_coef_t[k, k + self.nX])           
                        
                        if jj == 0 and ii == self.nZ - 1:
                            Mat_coef_t[k, k] = AP + AN - AW
                            Mat_coef_t[k, k + 1] = AE
                            Mat_coef_t[k, k - self.nX] = AS                        
                            b_t[k] = b_t[k] - 2 * AW * ( self.Tcub / self.T0 )
                            test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - self.nX])
                        
                        if ii == self.nZ - 1 and jj > 0 and jj < self.nX - 1:
                            Mat_coef_t[k, k] = AP + AN
                            Mat_coef_t[k, k + 1] = AE
                            Mat_coef_t[k, k - 1] = AW
                            Mat_coef_t[k, k - self.nX] = AS      
                            test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - self.nX])            
                        
                        if ii == self.nZ - 1 and jj == self.nX - 1:
                            Mat_coef_t[k, k] = AP + AE + AN
                            Mat_coef_t[k, k - 1] = AW
                            Mat_coef_t[k, k - self.nX] = AS
                            test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - self.nX])
                        k = k + 1

                    ################## Solution of temperature field ####################

                t = np.linalg.solve(Mat_coef_t, b_t)

                cont = 0

                for i in np.arange(self.nZ):
                    for j in np.arange(self.nX):

                        T_new[i, j]= self.T0 * t[cont]

                        cont=cont+1

            # Hydrodynamic forces =================================================================
            self.P_dimen =  self.P * ( self.mi0 * self.speed * self.Rp ** 2) / (self.Cr ** 2)
            
            auxF1 = np.zeros((self.nZ,self.nX))
            
            auxF2 = np.zeros((self.nZ,self.nX))

            for ni in np.arange(self.nZ):
                auxF1[ni,:] = np.cos(self.XTETA)
                auxF2[ni,:] = np.sin(self.XTETA) 
            
            YtetaF1 = self.P  * auxF1
            F1teta = np.trapz(YtetaF1,self.XTETA)
            self.F1[n_p] = -np.trapz(F1teta,self.XZ)

            YtetaF2 = self.P * auxF2
            F2teta = np.trapz(YtetaF2,self.XTETA)
            self.F2[n_p] = -np.trapz(F2teta,self.XZ)

        # Resulting forces - Inertial frame
        if "calc_EQ" in self.op_key:
            for k_i in range(0, self.npad):
                self.Fx[k_i] = self.F1[k_i] * np.cos(x[k_i+2] + self.sigma[k_i]) #angulo + posição angular (pivô)
                self.Fy[k_i] = self.F1[k_i] * np.sin(x[k_i+2] + self.sigma[k_i])
                self.Mj[k_i] = self.F2[k_i] * (self.Rp + self.tpad)

        elif "impos_EQ" in self.op_key:
            for k_i in range(0, self.npad):
                self.Fx[k_i] = self.F1[k_i] * np.cos(x[k_i] + self.sigma[k_i]) #angulo + posição angular (pivô)
                self.Fy[k_i] = self.F1[k_i] * np.sin(x[k_i] + self.sigma[k_i])
                self.Mj[k_i] = self.F2[k_i] * (self.Rp + self.tpad)
            
        Fhx = np.sum(self.Fx)

        Fhy = np.sum(self.Fy)


        self.Fhx = Fhx
        self.Fhy = Fhy

        self.FX_dim = Fhx / ( self.Cr ** 2 / ( self.Rp ** 3 * self.mi0 * self.speed * self.L))
        self.FY_dim = Fhy / ( self.Cr ** 2 / ( self.Rp ** 3 * self.mi0 * self.speed * self.L))

        if "calc_EQ" in self.op_key:

            FM = np.zeros((self.npad + 2))

            FM[0] = Fhx + self.WX
            FM[1] = Fhy + self.WY
            FM[2:len(FM)] = self.Mj[0:len(self.Mj)]

        elif "impos_EQ" in self.op_key:

            FM = self.Mj[0:len(self.Mj)]       

        score = np.linalg.norm(FM)
        print(x)
        print(f'Score: ', score)
        print(f'Score_Dim: ', self.Mj*self.dimForca)

        
        Pressure = self.P_dimen
        Temperature = T_new

        return score
    

    def xr_fun(self, n_p, eq_0, eq_1):
        
        xx = (
            eq_0
            * self.Cr
            * np.cos(eq_1)
        )
        yy = (
            eq_0
            * self.Cr
            * np.sin(eq_1)
        )
    
        xryr = np.dot(
            [
                [np.cos(self.sigma[n_p]), np.sin(self.sigma[n_p])],
                [-np.sin(self.sigma[n_p]), np.cos(self.sigma[n_p])],
            ],
            [[xx], [yy]],
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
        
        return xryr, xryrpt, xr, yr, xrpt, yrpt

    def plot_results(self):

        XH, YH = np.meshgrid(self.XTETA, self.XZ)
        ax = plt.axes(projection='3d')
        ax.plot_surface(XH, YH, 1e-6*self.PPdim[:,:,self.pad_in], rstride=1, cstride=1, cmap='jet', edgecolor='none')
        plt.grid()
        ax.set_title('Pressure field')
        ax.set_xlim([np.min(self.XTETA), np.max(self.XTETA)])
        ax.set_ylim([np.min(self.XZ), np.max(self.XZ)])
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        ax.set_zlabel('Pressure [MPa]')
        plt.show()

        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(XH, YH, self.TT_i[:, :, self.pad_in], cmap='jet')
        plt.grid()
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Temperature field [°C]')
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        plt.show()
        
    def plot_results2(self):

        d_axial = self.L/self.nZ
        axial = np.arange(0, self.L+d_axial , d_axial) 
        axial = axial[1:]-np.diff(axial)/2

        ang = []

        for k in range(self.npad):
            ang1 = (self.XTETA + self.sigma[k]) * 180 / np.pi 
            # for kk in range(len(ang1)):
            #     if ang1[kk] < 0:
            #         ang1[kk] = ang1[kk] + 360
            ang.append(ang1)

        fig_SP = self.plot_scatter(x_data=ang, y_data=self.PPdim, pos=15, y_title="Pressure (Pa)")
        fig_SP.write_image("pressure_T.pdf", width=900, height=500, engine="kaleido")

        fig_ST = self.plot_scatter(x_data=ang, y_data=self.TT_i, pos=15, y_title="Temperature (ºC)")
        fig_ST.write_image("temperature_T.pdf", width=900, height=500, engine="kaleido")

        fig_CP = self.plot_contourP(x_data=ang, y_data=axial, z_data=self.PPdim, z_title="Pressure (Pa)")
        fig_CP.write_image("pressure_field_T.pdf", width=900, height=500, engine="kaleido")

        fig_CP = self.plot_contourT(x_data=ang, y_data=axial, z_data=self.TT_i, z_title="Temperature (ºC)")
        fig_CP.write_image("temperature_field_T.pdf", width=900, height=500, engine="kaleido")

    def plot_scatter(self, x_data, y_data, pos, y_title):
        """This method plot a scatter(x,y) graph.

        Parameters
        ----------
        x_data : float
            X axis data.
        y_data : float
            Y axis data.
        pos : float
            Probe position.
        y_title : str
            Name of the Y axis

        Returns
        -------
        fig : object
            Scatter figure.
        """

        fig = go.Figure()
        for i in range(self.npad):
            fig.add_trace(
                go.Scatter(
                    x=x_data[i],  # horizontal axis
                    y=y_data[pos][:, i],  # vertical axis
                    name=f"Pad{i}",
                )
            )
        fig.update_layout(xaxis_range=[np.array(x_data).min()*1.1, 360-abs(np.array(x_data).min())])
        fig.update_layout(plot_bgcolor="white")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angle (º)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(text=y_title, font=dict(family="Times New Roman", size=30)),
        )
        fig.update_layout(legend = dict(font = dict(family = "Times New Roman", size = 22, color = "black")))
        fig.show()
        return fig

    def plot_contourP(self, x_data, y_data, z_data, z_title):
        """This method plot a contour(x,y,z) graph.

        Parameters
        ----------
        x_data : float
            X axis data.
        y_data : float
            Y axis data.
        z_data : float
            Z axis data.
        z_title : str
            Name of the z axis

        Returns
        -------
        fig : object
            Contour figure.
        """

        fig = go.Figure()
        max_val = z_data.max()
        for l in range(self.npad):
            fig.add_trace(
                go.Contour(
                    z=z_data[:, :, l],
                    x=x_data[l],  # horizontal axis
                    y=y_data,  # vertical axis
                    zmin=0,
                    zmax=max_val,
                    ncontours=15,
                    colorbar=dict(
                        title=z_title,  # title here
                        titleside="right",
                        titlefont=dict(size=30, family="Times New Roman"),
                        tickfont=dict(size=22),
                    ),
                )
            )
        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )
        fig.update_layout(xaxis_range=[np.array(x_data).min()*1.1, 360-abs(np.array(x_data).min())])
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angle (º)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(
                text="Pad Length (m)", font=dict(family="Times New Roman", size=30)
            ),
        )
        fig.update_layout(plot_bgcolor="white")
        fig.show()
        return fig

    def plot_contourT(self, x_data, y_data, z_data, z_title):
        """This method plot a contour(x,y,z) graph.

        Parameters
        ----------
        x_data : float
            X axis data.
        y_data : float
            Y axis data.
        z_data : float
            Z axis data.
        z_title : str
            Name of the z axis

        Returns
        -------
        fig : object
            Contour figure.
        """

        fig = go.Figure()
        max_val = z_data.max()
        for l in range(self.npad):
            fig.add_trace(
                go.Contour(
                    z=z_data[:, :, l],
                    x=x_data[l],  # horizontal axis
                    y=y_data,  # vertical axis
                    zmin=40,
                    zmax=max_val,
                    ncontours=25,
                    colorbar=dict(
                        title=z_title,  # title here
                        titleside="right",
                        titlefont=dict(size=30, family="Times New Roman"),
                        tickfont=dict(size=22),
                    ),
                )
            )
        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )
        fig.update_layout(xaxis_range=[np.array(x_data).min()*1.1, 360-abs(np.array(x_data).min())])
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angle (º)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(
                text="Pad Length (m)", font=dict(family="Times New Roman", size=30)
            ),
        )
        fig.update_layout(plot_bgcolor="white")
        fig.show()
        return fig

    def alpha(self):
        alphamax = (self.Rp-self.Rs-np.cos(self.TETA2)*(self.xryr[0, 0]+self.Rp-self.Rs-self.Cr))/(sin(self.TETA2)*(self.Rp+self.tpad)) - (self.xryr[1, 0])/(self.Rp+self.tpad)
        return abs(alphamax)
    
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



    def plot_results(self):
        ##### Plots
        XH, YH = np.meshgrid(self.XTETA, self.XZ)
        ax = plt.axes(projection='3d')
        ax.plot_surface(XH, YH, 1e-6*self.PPdim[:,:,self.pad_in], rstride=1, cstride=1, cmap='jet', edgecolor='none')
        plt.grid()
        ax.set_title('Pressure field')
        ax.set_xlim([np.min(self.XTETA), np.max(self.XTETA)])
        ax.set_ylim([np.min(self.XZ), np.max(self.XZ)])
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        ax.set_zlabel('Pressure [MPa]')
        plt.show()

        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(XH, YH, self.TT_i[:, :, self.pad_in], cmap='jet')
        plt.grid()
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Temperature field [°C]')
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        plt.show()


def tilting_pad_example01():
    """Create an example of a tilting pad bearing with Thermo-Hydro-Dynamic effects.
    This function returns pressure field and dynamic coefficient. The
    purpose is to make available a simple model so that a doctest can be
    written using it.

    """

    bearing = Tilting(
        Rs = 0.5 * 2000e-3,
        npad = 12,
        Rp = 0.5 * 2005e-3,
        tpad = 120e-3,
        betha_p = 20,
        rp_pad = 0.6,
        L = 350e-3,
        lubricant = "ISOVG68",
        Tcub = 45,
        nX = 30,
        nZ = 30,
        Cr = 250e-6,
        sigma = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
        speed = 90,
        choice_CAIMP = {"calc_EQ":{"init_guess":[9.98805447808967e-10, -0.000159030915145932, 0.000636496725369308,
            0.000913538939923720, 0.000690387727463155, 0.000854621895169753,
            0.000679444727719279, 0.000594279954895354, 0.000489138614623373,
            0.000377122288510832, 0.000319527936428236, 0.000340586533490991,
            0.000464320475944696, 0.000657880058055782],
            "load":[0, -757e3],
            "print":["result","progress"]}},
        Coefs_D={"b_loc":4, "show_coef":True}
    )

    # bearing = Tilting(
    #     Rs = 0.5 * 2000e-3,
    #     npad = 1,
    #     Rp = 0.5 * 2005e-3,
    #     tpad = 120e-3,
    #     betha_p = 20,
    #     rp_pad = 0.6,
    #     L = 350e-3,
    #     lubricant = "ISOVG68",
    #     Tcub = 45,
    #     nX = 30,
    #     nZ = 30,
    #     Cr = 250e-6,
    #     sigma = [0,],
    #     speed = 90,
    #     choice_CAIMP = {"calc_EQ":{"init_guess":[9.98805447808967e-10, -0.000159030915145932, 0.000636496725369308,],
    #         "load":[0, -757e3],
    #         "print":["result","progress"]}},
    #     Coefs_D={"b_loc":4, "show_coef":True}
    # )

    bearing.run()

    # bearing.plot_results()
    bearing.plot_results2()

def tilting_pad_example02():
    """Create an example of a tilting pad bearing with Thermo-Hydro-Dynamic effects.
    This function returns pressure field and dynamic coefficient. The
    purpose is to make available a simple model so that a doctest can be
    written using it.

    """

    # bearing = Tilting(
    #     Rs = 0.5 * 2000e-3,
    #     npad = 12,
    #     Rp = 0.5 * 2005e-3,
    #     tpad = 120e-3,
    #     betha_p = 20,
    #     rp_pad = 0.6,
    #     L = 350e-3,
    #     lubricant = "ISOVG68",
    #     Tcub = 45,
    #     nX = 30,
    #     nZ = 30,
    #     Cr = 250e-6,
    #     sigma = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], #graus
    #     speed = 90, #rpm
    #     choice_CAIMP = {"impos_EQ":{"pos_EQ":[1.0178313406154993e-09, -0.00015656354137611597],
    #         "ent_angle":[0.000636496725369308,
    #         0.000913538939923720, 0.000690387727463155, 0.000854621895169753,
    #         0.000679444727719279, 0.000594279954895354, 0.000489138614623373,
    #         0.000377122288510832, 0.000319527936428236, 0.000340586533490991,
    #         0.000464320475944696, 0.000657880058055782],
    #         "print":["result","progress"]}}, #pos eq = [exc, ang] ent_ang = [angulos das pads]
    #     # Coefs_D={"b_loc":4, "show_coef":True}
    # )    

    bearing = Tilting(
        Rs = 0.5 * 2000e-3,
        npad = 1,
        Rp = 0.5 * 2005e-3,
        tpad = 120e-3,
        betha_p = 20,
        rp_pad = 0.6,
        L = 350e-3,
        lubricant = "ISOVG68",
        Tcub = 45,
        nX = 30,
        nZ = 30,
        Cr = 250e-6,
        sigma = [0,],
        speed = 90,
        choice_CAIMP = {"impos_EQ":{"pos_EQ":[0.300000000000001, 0.000001],
            "ent_angle":[0.000636496725369308,],
            "print":["result","progress"]}},
        Coefs_D={"b_loc":4, "show_coef":True}
    )

    bearing.run()

    # bearing.plot_results()
    bearing.plot_results2()

if __name__ == "__main__":
    tilting_pad_example02()