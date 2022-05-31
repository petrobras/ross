from cmath import sin
from re import M
import numpy as np
import scipy 
from numpy.linalg import pinv
from scipy.linalg import solve
from decimal import Decimal
from scipy.optimize import fmin

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm

class Tilting:
    """ This class calculates the pressure and temperature fields,
        equilibrium position of a tilting-pad journal bearing,
        and the hydrodynamic forces.

    Parameters
    ----------
    Bearing Geometry
    ^^^^^^^^^^^^^^^^
    Describes the geometric characteristics.
    Rs : float
        Shaft radius. Default unit is meter.
    npad : float
         integer
         Number of pads
    Rp : float
        Pad radius. Default unit is meter.
    tpad : float
         Pad thickness. Default unit is meter.         
    betha_p : float
            Arc length of each pad. The unit is degree.
    sigma : float
          Pivot angular position. The unit is degree.
    rp_pad : float
            Pivot offset. Dimensionless.
    L : Bearing length. Default unit is meter.
    Cr : float
       Radial Clearance. Default unit is meter.

    Operating conditions
    ^^^^^^^^^^^^^^^^^^^^
    Describes the operating conditions of the bearing
    wa : float
       Rotating speed. The unit is RPM.
    Tcub : float
       Oil bath temperature. The unit is Celsius.
    Wx : float
        Load X - direction. The unit is Newton.
    Wy : float 
        Load Y - direction. The unit is Newton.

    Fluid properties
    ^^^^^^^^^^^^^^^^
    Describes the fluid characteristics
    mi0 : float
        Reference fluid viscosity. The unit is Pa*s
    rho : float
        Fluid specific mass. Default unit is kg/m^3.
    kt : float
        Fluid thermal conductivity. The unit is J/(s*m*°C).
    Cp : float
        Fluid specific heat. The unit is J/(kg*°C).
    T0 : float
        Reference fluid temperature. The unit is °C.
    k1, k2, and k3 : float
        Oil coefficients for the viscosity interpolation.

    Mesh discretization
    ^^^^^^^^^^^^^^^^^^^
    Describes the discretization of the fluid film
    nX : integer
        Number of volumes along the x (teta) direction
    nZ : integer
        Number of volumes along the z direction

    Returns
    -------
    maxP : float
         Maximum pressure. The unit is Pa.
    maxT : float
          Maximum temperature. The unit is Celsius.
    h_pivot0 : float
             Oil film thickness at the pivot point. The unit is meter.
    PPdim : array
          Pressure field. The unit is Pa.
    TT_i : array
        Temperature field. The unit is Celsius.
    pad_in : integer
           Most loaded pad. Dimensionless.

    References
    ----------
    .. [1] BARBOSA, J.S. Analise de Modelos Termohidrodinamicos para Mancais de unidades geradoras Francis. 2016. Dissertacao de Mestrado. Universidade Federal de Uberlandia, Uberlandia. ..
    .. [2] NICOLETTI, R., Efeitos Termicos em Mancais Segmentados Hibridos Teoria e Experimento. 1999. Dissertacao de Mestrado. Universidade Estadual de Campinas, Campinas. ..
    .. [3] DANIEL, G. B., Desenvolvimento de um Modelo Termohidrodinamico para Analise em Mancais Segmentados. 2012. Tese de Doutorado. Universidade Estadual de Campinas, Campinas. ..
   
    Atributes
    ---------
    XZ : array
    XTETA : array
    PPdim : array
    H0 : array
    H : array
    P : array
    Fx : array
    Fy : array
    Mj : array
    F1 : array
    F2 : array
    PP : array
    P_bef : array
    h_pivot : array
    dPdX : array
    dPdZ : array
    Reyn : array
    mi_turb : array
    """
    def __init__(
        self,
        Rs,
        npad,
        Rp,
        tpad,
        betha_p,
        sigma,
        rp_pad,
        L,
        Cr,
        wa,
        Tcub,
        Wx,
        Wy,
        rho,
        kt,
        Cp,
        k1,
        k2,
        k3,
        nX,
        nZ,
        x0,
    ):
        self.Rs = Rs
        self.npad = npad
        self.Rp = Rp
        self.tpad = tpad
        self.betha_p = betha_p
        self.sigma = sigma
        self.rp_pad = rp_pad
        self.L = L
        self.Cr = Cr
        self.wa = wa 
        self.Tcub = Tcub
        self.Wx = Wx
        self.Wy = Wy
        self.rho = rho
        self.kt = kt
        self.Cp = Cp
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.nX = nX
        self.nZ = nZ
        self.x0 = x0

        betha_p = betha_p * (np.pi/180) # Pad angle [rad]

        war = (wa*np.pi)/30 # Rotating speed [rad/s]

        sigma = sigma * (np.pi/180) # Pivot angular position [rad]

        T0 = Tcub # Reference temperature [Celsius]

        mi0 = k1 * np.exp( k2 / ( T0 + 273.15 + k3 )) # Reference Oil Viscosity [N.s/m^2]

        WX = Wx * ( Cr ** 2 / ( Rp ** 3 * mi0 * war * L)) # Loading - X direction [dimensionless]

        WY = Wy * ( Cr ** 2 / ( Rp ** 3 * mi0 * war * L)) # Loading - Y direction [dimensionless]

        TT_i = T0 * np.ones((nZ,nX,npad)) #Inicial 3D - temperature field

        TETA1= - ( rp_pad ) * betha_p # initial coordinate in the TETA diretion
        TETA2= ( 1-rp_pad ) * betha_p # final coordinate in the TETA diretion

        dTETA = ( TETA2 - TETA1 ) / nX

        Z1 = -0.5 # dimensionless initial coordinate in the z diretion
        Z2 = 0.5 # dimensionless final coordinate in the z diretion

        dZ = ( Z2 - Z1 ) / nZ
        dX = dTETA / betha_p

        XZ = np.zeros(nZ)
        XZ[0] = Z1 + 0.5 * dZ

        XTETA = np.zeros(nX)
        XTETA[0] = TETA1 + 0.5 * dTETA

        for ii in range(1, nZ):
            XZ[ii] = XZ[ii-1] + dZ

        for jj in range(1, nX):
            XTETA[jj] = XTETA[jj-1] + dTETA

        ##### Center shaft speed
        xpt = 0  
        ypt = 0
        #####

        ############### Define parameters ##############
        PPdim = np.zeros((nZ,nX,npad))
        H0 = np.zeros((nZ,nX,npad))
        H = np.zeros((nZ,nX))
        P = np.zeros((nZ,nX))
        T_new = np.zeros((nZ,nX))
        Fx = np.zeros((npad))
        Fy = np.zeros((npad))
        Mj = np.zeros((npad))
        F1 = np.zeros((npad))
        F2 = np.zeros((npad))
        PP = np.zeros((nZ, nX, npad))
        P_bef = np.zeros((nZ,nX))
        h_pivot = np.zeros((npad))
        dPdX = np.zeros((nZ,nX))
        dPdZ = np.zeros((nZ,nX))
        Reyn = np.zeros((nZ,nX,npad))
        mi_turb = 1.3*np.ones((nZ,nX,npad)) #Turbulence

        def HDEequilibrium(x,*args):
            L,Rs,Rp,Cr,tpad,mi0,war,WX,WY,XTETA,XZ,dX,dZ,xpt,ypt,nZ,nX,npad,betha_p,sigma,dTETA,TT_i,x0=args

            ##### Dimensionless center shaft coordinates
            xx = x[0] 
            yy = x[1]
            #####

            psi_pad = np.zeros((npad))

            for kp in range(0, npad):
                psi_pad[kp] = x[kp+2] # Tilting angles of each pad

            nk = (nX) * (nZ)

            tol_T = 0.1 #Celsius degree

            for n_p in range(0, npad):

                T_new = TT_i[:,:,n_p]

                T_i = 1.1 * T_new

                cont_Temp = 0

                while abs((T_new - T_i).max()) >= tol_T:

                    cont_Temp = cont_Temp + 1

                    T_i = np.array(T_new)

                    mi_i = (
                        k1 * np.exp(k2 / (T_i + 273.15 + k3))
                        )  # [Pa.s] 

                    MI = mi_i * 1/mi0 #Dimensionless viscosity field

                    k = 0 #vectorization index

                    Mat_coef = np.zeros((nk, nk))

                    b = np.zeros(nk)

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

                    alpha = psi_pad[n_p]
                    alphapt = 0

                    for ii in range(0, nZ):

                        for jj in range(0, nX):

                            TETAe = XTETA[jj] + 0.5 * dTETA
                            TETAw = XTETA[jj] - 0.5 * dTETA

                            hP = ( Rp - Rs - ( np.sin( XTETA[jj]) * ( yr + alpha * ( Rp + tpad ) ) + np.cos( XTETA[jj] ) * ( xr + Rp - Rs - Cr ) ) ) / Cr

                            he = ( Rp - Rs - ( np.sin( TETAe ) * ( yr + alpha * ( Rp + tpad ) ) + np.cos( TETAe ) * ( xr + Rp - Rs - Cr ) ) ) / Cr

                            hw = ( Rp - Rs - ( np.sin( TETAw ) * ( yr + alpha * ( Rp + tpad ) ) + np.cos( TETAw ) * ( xr + Rp - Rs - Cr ) ) ) / Cr

                            hn = hP

                            hs = hP

                            hpt = -(1 / (Cr * war)) * (np.cos(XTETA[jj]) * xrpt + np.sin(XTETA[jj]) * yrpt + np.sin(XTETA[jj]) * (Rp + tpad) * alphapt)

                            H[ii,jj] = hP

                            if jj == 0 and ii == 0:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = MI[ii, jj]
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = MI[ii, jj]

                            if jj == 0 and ii > 0 and ii < nZ-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = MI[ii, jj]
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if jj == 0 and ii == nZ-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = MI[ii, jj]
                                MI_n = MI[ii, jj]
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if ii == 0 and jj > 0 and jj < nX-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = MI[ii, jj]

                            if jj > 0 and jj < nX-1 and ii > 0 and ii < nZ-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if ii == nZ-1 and jj > 0 and jj < nX-1:
                                MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = MI[ii, jj]
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if ii == 0 and jj == nX-1:
                                MI_e = MI[ii, jj]
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = MI[ii, jj]

                            if jj == nX-1 and ii > 0 and ii < nZ-1:
                                MI_e = MI[ii, jj]
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            if jj == nX-1 and ii == nZ-1:
                                MI_e = MI[ii, jj]
                                MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                                MI_n = MI[ii, jj]
                                MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                            CE = 1 / (betha_p ** 2) * (he ** 3 / ( 12 * MI_e ) ) * dZ / dX
                            CW = 1 / (betha_p ** 2) * (hw ** 3 / ( 12 * MI_w ) ) * dZ / dX
                            CN = ( Rp / L ) ** 2 * ( dX / dZ ) * ( hn ** 3 / ( 12 * MI_n ) ) 
                            CS = ( Rp / L ) ** 2 * ( dX / dZ ) * ( hs ** 3 / ( 12 * MI_s ) )
                            CP = - ( CE + CW + CN + CS )
                            B = ( Rs / ( 2 * Rp * betha_p ) ) * dZ * ( he - hw ) + hpt * dX * dZ
                            b[k] = B

                            # Mat_coef determination depending on its mesh localization
                            if ii == 0 and jj == 0:
                                Mat_coef[k, k] = CP - CS - CW
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k + nX] = CN

                            if ii == 0 and jj > 0 and jj < nX - 1:
                                Mat_coef[k, k] = CP - CS
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k + nX] = CN

                            if ii == 0 and jj == nX - 1:
                                Mat_coef[k, k] = CP - CE - CS
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k + nX] = CN

                            if jj == 0 and ii > 0 and ii < nZ - 1:
                                Mat_coef[k, k] = CP - CW
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - nX] = CS
                                Mat_coef[k, k + nX] = CN
                            
                            if ii > 0 and ii < nZ - 1 and jj > 0 and jj < nX - 1:
                                Mat_coef[k, k] = CP
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - nX] = CS
                                Mat_coef[k, k + nX] = CN
                                Mat_coef[k, k + 1] = CE

                            if jj == nX - 1 and ii > 0 and ii < nZ - 1:
                                Mat_coef[k, k] = CP - CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - nX] = CS
                                Mat_coef[k, k + nX] = CN

                            if jj == 0 and ii == nZ - 1:
                                Mat_coef[k, k] = CP - CN - CW
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - nX] = CS

                            if ii == nZ - 1 and jj > 0 and jj < nX - 1:
                                Mat_coef[k, k] = CP - CN
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - nX] = CS

                            if ii == nZ - 1 and jj == nX - 1:
                                Mat_coef[k, k] = CP - CE - CN
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - nX] = CS

                            k = k + 1

                    # Pressure field solution ==============================================================
                    p = np.linalg.solve(Mat_coef, b)

                    cont = 0

                    for i in np.arange(nZ):
                        for j in np.arange(nX):

                            P[i, j] = p[cont]
                            cont = cont + 1

                            if P[i, j] < 0:
                                P[i, j] = 0

                    ##################################### Energy equation ################################
                    ################################### Pressure Gradients ###############################

                    nk = (nX) * (nZ)
                    Mat_coef_t = np.zeros((nk, nk))
                    b_t = np.zeros(nk)
                    test_diag = np.zeros(nk)

                    k = 0 #vectorization temperature index

                    for ii in range(0, nZ):
                        for jj in range(0, nX):

                            if jj == 0 and ii == 0:
                                dPdX[ii, jj] = (P[ii, jj + 1] - P[ii, jj]) / (dX)
                                dPdZ[ii, jj] = (P[ii + 1, jj] - P[ii, jj] ) / (dZ)

                            if jj == 0 and ii > 0 and ii < nZ-1:
                                dPdX[ii, jj] = (P[ii, jj + 1] - P[ii, jj]) / (dX)
                                dPdZ[ii, jj] = (P[ii + 1, jj] - P[ii, jj]) / (dZ)

                            if jj == 0 and ii == nZ-1:
                                dPdX[ii, jj] = (P[ii, jj + 1] - P[ii, jj]) / (dX)
                                dPdZ[ii, jj] = (0 - P[ii, jj]) / (dZ)

                            if ii == 0 and jj > 0 and jj < nX-1:
                                dPdX[ii, jj] = (P[ii, jj + 1] - P[ii, jj]) / (dX)
                                dPdZ[ii, jj] = (P[ii + 1, jj] - P[ii, jj]) / (dZ)

                            if jj > 0 and jj < nX-1 and ii > 0 and ii < nZ-1:
                                dPdX[ii, jj] = (P[ii, jj + 1] - P[ii, jj]) / (dX)
                                dPdZ[ii, jj] = (P[ii + 1, jj] - P[ii, jj]) / (dZ)

                            if ii == nZ-1 and jj > 0 and jj < nX-1:
                                dPdX[ii, jj] = (P[ii, jj + 1] - P[ii, jj]) / (dX)
                                dPdZ[ii, jj] = (0 - P[ii, jj]) / (dZ)

                            if ii == 0 and jj == nX-1:
                                dPdX[ii, jj] = (0 - P[ii, jj]) / (dX)
                                dPdZ[ii, jj] = (P[ii + 1, jj] - P[ii, jj]) / (dZ)

                            if jj == nX-1 and ii > 0 and ii < nZ-1:
                                dPdX[ii, jj] = (0 - P[ii, jj]) / (dX)
                                dPdZ[ii, jj] = (P[ii + 1, jj] - P[ii, jj]) / (dZ)

                            if jj == nX-1 and ii == nZ-1:
                                dPdX[ii, jj] = (0 - P[ii, jj]) / (dX)
                                dPdZ[ii, jj] = (0 - P[ii, jj]) / (dZ)

                ######################### Turbulence (Eddy viscosity) #########################
                            HP = H[ii, jj]

                            mi_p = MI[ii, jj]
                            
                            Reyn[ii, jj, n_p] = rho * war * Rs * ( HP / L ) * Cr / ( mi0 * mi_p )

                            if Reyn[ii, jj, n_p] <= 500:
                                        
                                delta_turb = 0
                                        
                            elif Reyn[ii, jj, n_p] > 400 and Reyn[ii, jj, n_p] <= 1000:
                                            
                                delta_turb = 1 - ( ( 1000 - Reyn[ii, jj, n_p] ) / 500 ) ** ( 1 / 8 )
                                            
                            elif Reyn[ii, jj, n_p] > 1000:
                                        
                                delta_turb = 1

                            dudx = ( ( ( HP / mi_turb[ii, jj, n_p]) * dPdX[ii, jj] ) - ( war / HP ) )
                                        
                            dwdx = ( ( HP / mi_turb[ii, jj, n_p] ) * dPdZ[ii, jj] )
                                        
                            tal = mi_turb[ii, jj, n_p] * np.sqrt( ( dudx ** 2 ) + ( dwdx ** 2 ) )
                                        
                            ywall = ( ( HP * Cr * 2 ) / ( mi0 * mi_turb[ii, jj, n_p] / rho ) ) * ( ( abs( tal ) / rho ) ** 0.5 )
                                        
                            emv = 0.4 * ( ywall - ( 10.7 * np.tanh( ywall / 10.7 ) ) )
                                        
                            mi_turb[ii, jj, n_p] = mi_p * ( 1 + ( delta_turb * emv ) )
                                        
                            mi_t = mi_turb[ii, jj, n_p]

                            #### Coefficients for the energy equation
                            aux_up = 1
                            if XZ[ii] < 0 :
                                aux_up = 0

                            AE =   - ( kt * HP * dZ ) / ( rho * Cp * war * ( ( betha_p * Rp ) ** 2 ) * dX ) 

                            AW =  ( ( HP ** 3 ) * dPdX[ii, jj] * dZ ) / (12 * mi_t * (betha_p ** 2 ) )  - ( (Rs * HP * dZ ) / ( 2 * Rp * betha_p ) ) -  ( kt * HP * dZ ) / ( rho * Cp * war * ( ( betha_p * Rp ) ** 2 ) * dX ) 

                            AN_1 = (aux_up - 1) * ( ( ( Rp ** 2 ) * ( HP ** 3 ) *dPdZ[ii, jj] * dX ) / ( 12 * ( L ** 2 ) * mi_t ) )

                            AS_1 = (aux_up) *  ( ( ( Rp ** 2 ) * ( HP ** 3 ) * dPdZ[ii, jj] * dX ) / ( 12 * (L ** 2 ) * mi_t ) )

                            AN_2 = - ( kt * HP * dX) / ( rho * Cp * war * ( L ** 2 ) * dZ )

                            AS_2 = - ( kt * HP * dX ) / ( rho * Cp * war * ( L ** 2 ) * dZ ) 

                            AN = AN_1 + AN_2

                            AS = AS_1 + AS_2

                            AP = - ( AE + AW + AN + AS )

                            auxB_t = ( war * mi0 ) / ( rho * Cp * Tcub * Cr )

                            B_tG = ( mi0 * war * ( Rs ** 2 ) * dX * dZ * P[ii, jj] * hpt ) / ( rho * Cp * T0 * ( Cr ** 2 ) )

                            B_tH = ( ( war * mi0 * ( hpt ** 2 ) * 4 * mi_t * dX * dZ ) / ( rho * Cp * T0  * 3 * HP ) )

                            B_tI = auxB_t * ( 1 * mi_t * ( Rs ** 2 ) * dX * dZ ) / ( HP * Cr )

                            B_tJ = auxB_t * ( ( Rp ** 2 ) * ( HP ** 3 ) * ( dPdX[ii, jj] ** 2 ) * dX * dZ ) / ( 12 * Cr * ( betha_p ** 2 ) * mi_t )

                            B_tK = auxB_t * ( ( Rp ** 4 ) * ( HP ** 3 ) * ( dPdZ[ii, jj] ** 2 ) * dX * dZ ) / ( 12 * Cr * ( L ** 2 ) * mi_t)

                            B_t = B_tG + B_tH + B_tI + B_tJ + B_tK

                            b_t[k] = B_t 

                            if ii == 0 and jj == 0:
                                Mat_coef_t[k, k] = AP + AS - AW
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k + nX] = AN
                                b_t[k] = b_t[k] - 2 * AW * ( Tcub / T0 )                           
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k + nX])           
                            
                            if ii == 0 and jj > 0 and jj < nX - 1: 
                                Mat_coef_t[k, k] = AP + AS
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k + nX] = AN
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - 1] + Mat_coef_t[k, k + nX])   
                            
                            if ii == 0 and jj == nX - 1:
                                Mat_coef_t[k, k] = AP + AE + AS
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k + nX] = AN            
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k - 1] + Mat_coef_t[k, k + nX])    
                            
                            if jj == 0 and ii > 0 and ii < nZ - 1:
                                Mat_coef_t[k, k] = AP - AW
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k - nX] = AS
                                Mat_coef_t[k, k + nX] = AN
                                b_t[k] = b_t[k] - 2 * AW * ( Tcub / T0 )
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - nX] + Mat_coef_t[k, k + nX])           
                            
                            if ii > 0 and ii < nZ - 1 and jj > 0 and jj < nX - 1:
                                Mat_coef_t[k, k] = AP
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k - nX] = AS
                                Mat_coef_t[k, k + nX] = AN
                                Mat_coef_t[k, k + 1] = AE
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - nX] + Mat_coef_t[k, k + nX])   
                            
                            if jj == nX - 1 and ii > 0 and ii < nZ - 1:
                                Mat_coef_t[k, k] = AP + AE
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k - nX] = AS
                                Mat_coef_t[k, k + nX] = AN
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - nX] + Mat_coef_t[k, k + nX])           
                            
                            if jj == 0 and ii == nZ - 1:
                                Mat_coef_t[k, k] = AP + AN - AW
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k - nX] = AS                        
                                b_t[k] = b_t[k] - 2 * AW * ( Tcub / T0 )
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - nX])
                            
                            if ii == nZ - 1 and jj > 0 and jj < nX - 1:
                                Mat_coef_t[k, k] = AP + AN
                                Mat_coef_t[k, k + 1] = AE
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k - nX] = AS      
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k + 1] + Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - nX])            
                            
                            if ii == nZ - 1 and jj == nX - 1:
                                Mat_coef_t[k, k] = AP + AE + AN
                                Mat_coef_t[k, k - 1] = AW
                                Mat_coef_t[k, k - nX] = AS
                                test_diag[k] = Mat_coef_t[k, k] - (Mat_coef_t[k, k - 1] + Mat_coef_t[k, k - nX])
                            k = k + 1

                        ################## Solution of temperature field ####################

                    t = np.linalg.solve(Mat_coef_t, b_t)

                    cont = 0

                    for i in np.arange(nZ):
                        for j in np.arange(nX):

                            T_new[i, j]= T0 * t[cont]

                            cont=cont+1

                # Hydrodynamic forces =================================================================
                TT_i[:,:,n_p] = np.array(T_i)
                
                auxF1 = np.zeros((nZ,nX))
                auxF2 = np.zeros((nZ,nX))

                for ni in np.arange(nZ):
                    auxF1[ni,:] = np.cos(XTETA)
                    auxF2[ni,:] = np.sin(XTETA) 
                
                YtetaF1 = P * auxF1
                F1teta = np.trapz(YtetaF1,XTETA)
                F1[n_p] = -np.trapz(F1teta,XZ)

                YtetaF2 = P * auxF2
                F2teta = np.trapz(YtetaF2,XTETA)
                F2[n_p] = -np.trapz(F2teta,XZ)

                # Resulting forces - Inertial frame
            for k_i in range(0, npad):
                Fx[k_i] = F1[k_i] * np.cos(x[k_i+2] + sigma[k_i])
                Fy[k_i] = F1[k_i] * np.sin(x[k_i+2] + sigma[k_i])
                Mj[k_i] = F2[k_i] * (Rp + tpad)

            Fhx = np.sum(Fx)

            Fhy = np.sum(Fy)

            FM = np.zeros((npad + 2,1))

            FM[0,0] = Fhx + WX
            FM[1,0] = Fhy + WY
            FM[2:len(FM),0] = Mj[0:len(Mj)]

            score = np.linalg.norm(FM)
            print(x)
            print(f'Score: ', score)

            return score

        entrada= tuple((L,Rs,Rp,Cr,tpad,mi0,war,WX,WY,XTETA,XZ,dX,dZ,xpt,ypt,nZ,nX,npad,betha_p,sigma,dTETA,TT_i,x0))
        x = scipy.optimize.fmin(
        HDEequilibrium,                  
        x0,
        args=entrada,
        xtol=1e-4,
        ftol=1e-2,
        maxiter=100000,
        maxfun=100000,
        full_output=0,
        disp=1,
        retall=0,
        callback=None,
        initial_simplex=None,
        )

        def PRESSURE(x1,TT_i1):
            L,Rs,Rp,Cr,tpad,mi0,war,dX,dZ,xpt,ypt,nZ,nX,npad,betha_p,sigma,dTETA
            #### Dimensionless center shaft coordinates
            xx = x1[0] 
            yy = x1[1]
            #####

            psi_pad = np.zeros((npad))

            for kp in range(0, npad):
                psi_pad[kp] = x1[kp+2] # Tilting angles of each pad

            nk = (nX) * (nZ)

            for n_p in range(0, npad):

                T_i = TT_i1[:,:,n_p]

                mi_i = (
                        k1 * np.exp(k2 / (T_i + 273.15 + k3))
                    )  # [Pa.s]

                MI = mi_i / mi0

                k = 0 #vectorization index

                Mat_coef = np.zeros((nk, nk))

                b = np.zeros(nk)

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

                alpha = psi_pad[n_p]
                alphapt = 0

                for ii in range(0, nZ):

                    for jj in range(0, nX):

                        TETAe = XTETA[jj] + 0.5 * dTETA
                        TETAw = XTETA[jj] - 0.5 * dTETA

                        hP = ( Rp - Rs - ( np.sin( XTETA[jj]) * ( yr + alpha * ( Rp + tpad ) ) + np.cos( XTETA[jj] ) * ( xr + Rp - Rs - Cr ) ) ) / Cr

                        he = ( Rp - Rs - ( np.sin( TETAe ) * ( yr + alpha * ( Rp + tpad ) ) + np.cos( TETAe ) * ( xr + Rp - Rs - Cr ) ) ) / Cr

                        hw = ( Rp - Rs - ( np.sin( TETAw ) * ( yr + alpha * ( Rp + tpad ) ) + np.cos( TETAw ) * ( xr + Rp - Rs - Cr ) ) ) / Cr

                        hn = hP

                        hs = hP

                        hpt = -(1 / (Cr * war)) * (np.cos(XTETA[jj]) * xrpt + np.sin(XTETA[jj]) * yrpt + np.sin(XTETA[jj]) * (Rp + tpad) * alphapt)

                        H[ii,jj] = hP

                        if jj == 0 and ii == 0:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = MI[ii, jj]
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = MI[ii, jj]

                        if jj == 0 and ii > 0 and ii < nZ-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = MI[ii, jj]
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if jj == 0 and ii == nZ-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = MI[ii, jj]
                            MI_n = MI[ii, jj]
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if ii == 0 and jj > 0 and jj < nX-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = MI[ii, jj]

                        if jj > 0 and jj < nX-1 and ii > 0 and ii < nZ-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if ii == nZ-1 and jj > 0 and jj < nX-1:
                            MI_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = MI[ii, jj]
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if ii == 0 and jj == nX-1:
                            MI_e = MI[ii, jj]
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = MI[ii, jj]

                        if jj == nX-1 and ii > 0 and ii < nZ-1:
                            MI_e = MI[ii, jj]
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        if jj == nX-1 and ii == nZ-1:
                            MI_e = MI[ii, jj]
                            MI_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            MI_n = MI[ii, jj]
                            MI_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        CE = 1 / (betha_p ** 2) * (he ** 3 / ( 12 * MI_e ) ) * dZ / dX
                        CW = 1 / (betha_p ** 2) * (hw ** 3 / ( 12 * MI_w ) ) * dZ / dX
                        CN = ( Rp / L ) ** 2 * ( dX / dZ ) * ( hn ** 3 / ( 12 * MI_n ) ) 
                        CS = ( Rp / L ) ** 2 * ( dX / dZ ) * ( hs ** 3 / ( 12 * MI_s ) )
                        CP = - ( CE + CW + CN + CS )
                        B = ( Rs / ( 2 * Rp * betha_p ) ) * dZ * ( he - hw ) + hpt * dX * dZ
                        b[k] = B

                        # Mat_coef determination depending on its mesh localization
                        if ii == 0 and jj == 0:
                            Mat_coef[k, k] = CP - CS - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k + nX] = CN

                        if ii == 0 and jj > 0 and jj < nX - 1:
                            Mat_coef[k, k] = CP - CS
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + nX] = CN

                        if ii == 0 and jj == nX - 1:
                            Mat_coef[k, k] = CP - CE - CS
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k + nX] = CN

                        if jj == 0 and ii > 0 and ii < nZ - 1:
                            Mat_coef[k, k] = CP - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - nX] = CS
                            Mat_coef[k, k + nX] = CN
                            
                        if ii > 0 and ii < nZ - 1 and jj > 0 and jj < nX - 1:
                            Mat_coef[k, k] = CP
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - nX] = CS
                            Mat_coef[k, k + nX] = CN
                            Mat_coef[k, k + 1] = CE

                        if jj == nX - 1 and ii > 0 and ii < nZ - 1:
                            Mat_coef[k, k] = CP - CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - nX] = CS
                            Mat_coef[k, k + nX] = CN

                        if jj == 0 and ii == nZ - 1:
                            Mat_coef[k, k] = CP - CN - CW
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - nX] = CS

                        if ii == nZ - 1 and jj > 0 and jj < nX - 1:
                            Mat_coef[k, k] = CP - CN
                            Mat_coef[k, k + 1] = CE
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - nX] = CS

                        if ii == nZ - 1 and jj == nX - 1:
                            Mat_coef[k, k] = CP - CE - CN
                            Mat_coef[k, k - 1] = CW
                            Mat_coef[k, k - nX] = CS

                        k = k + 1

                # Pressure field solution ==============================================================
                p = np.linalg.solve(Mat_coef, b)

                cont = 0

                for i in np.arange(nZ):
                    for j in np.arange(nX):

                        P[i, j] = p[cont]
                        cont = cont + 1

                        if P[i, j] < 0:
                            P[i, j] = 0

                PP[:,:,n_p] = P
                H0[:,:,n_p] = H
                h_pivot[n_p] = Cr * ( Rp - Rs - ( np.sin( 0 ) * ( yr + alpha * ( Rp + tpad ) ) + np.cos( 0 ) * ( xr + Rp - Rs - Cr ) ) ) / Cr
            
            return H0, h_pivot, PP

        PRESSURE (x,TT_i)

        ##### Plots
        pad_in = 0
        for n_p in range(0, npad):

            PPdim = PP[:,:,n_p]*( mi0 * war * Rp ** 2) / (Cr ** 2)

            if np.max(PPdim) > np.max(P_bef):
                pad_in = n_p 
                P_bef = PPdim

            XH, YH = np.meshgrid(XTETA, XZ)
            ax = plt.axes(projection='3d')
            ax.plot_surface(XH, YH, 1e-6*PPdim, rstride=1, cstride=1, cmap='jet', edgecolor='none')
            plt.grid()
            ax.set_title('Pressure field')
            ax.set_xlim([np.min(XTETA), np.max(XTETA)])
            ax.set_ylim([np.min(XZ), np.max(XZ)])
            ax.set_xlabel('X direction [rad]')
            ax.set_ylabel('Z direction [-]')
            ax.set_zlabel('Pressure [MPa]')
            plt.show()

            fig,ax=plt.subplots(1,1)
            cp = ax.contourf(XH, YH, TT_i[:, :, n_p], cmap='jet')
            plt.grid()
            fig.colorbar(cp) # Add a colorbar to a plot
            ax.set_title('Temperature field [°C]')
            ax.set_xlabel('X direction [rad]')
            ax.set_ylabel('Z direction [-]')
            plt.show()

        ##### Outputs
        maxP = PP[:,:,pad_in].max() * ( mi0 * war * Rp ** 2) / (Cr ** 2)
        medP = PP[:,:,pad_in].mean() * ( mi0 * war * Rp ** 2) / (Cr ** 2)
        maxT = TT_i[:,:,pad_in].max()
        h_pivot0 = h_pivot[pad_in]

        print(f'Maximum pressure: ', maxP)
        print(f'Maximum temperature: ', maxT)
        print(f'Average pressure: ', medP)
        print(f'Oil film thickness at the pivot: ', h_pivot0)
        print(f'Most loaded pad: ', pad_in)
        #####

        ##### Plots of the most loaded pad
        XH, YH = np.meshgrid(XTETA, XZ)
        ax = plt.axes(projection='3d')
        ax.plot_surface(XH, YH, 1e-6*PP[:,:,pad_in] * ( mi0 * war * Rp ** 2) / (Cr ** 2), rstride=1, cstride=1, cmap='jet', edgecolor='none')
        plt.grid()
        ax.set_title('Pressure field')
        ax.set_xlim([np.min(XTETA), np.max(XTETA)])
        ax.set_ylim([np.min(XZ), np.max(XZ)])
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        ax.set_zlabel('Pressure [MPa]')
        plt.show()

        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(XH, YH, TT_i[:, :, pad_in], cmap='jet')
        plt.grid()
        fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Temperature field [°C]')
        ax.set_xlabel('X direction [rad]')
        ax.set_ylabel('Z direction [-]')
        plt.show()

def tilting_pad_example():
    """Create an example of a tilting_pad journal bearing with Thermo effects.
    This function returns pressure and temperatures fields, hydrodynamic forces,
    and the equilibrium position of the bearing for a given operational condition.
    The purpose is to make available a simple model so that a doctest can be
    written using it.
    >>> bearing = tilting_pad_example()
    >>> bearing.L
    """
    Rs = 0.5 * 2000e-3 # Rotor radius [m]
    
    npad = 12 # Number of pads

    Rp = 0.5 * 2005e-3 # Pad radius [m]

    tpad = 120e-3 # Pad thickness [m]

    betha_p = 20 # Pad angle [degree]

    sigma = np.array(
        [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    ) # Pivot angular position [degree]

    rp_pad = 0.6 # Pivot offset

    wa = 90 # Rotating speed [rpm]

    L = 350e-3 # Bearing length [m]

    Tcub = 45 # Oil Tank temperature [Celsius]

    nX = 30 # Number of volumes along the x (teta) direction

    nZ = nX # Number of volumes along the z direction

    ##### Oil properties
    # ISO VG 68

    kt=float(0.1316)  #Thermal conductivity [J/s.m.°C]
    
    Cp=float(1890)    #Specific heat [J/kg°C]
        
    rho=float(886)    #Specific mass [kg/m³]

    ##### Vogels equation coefficients
    k1 = 5.506e-9
    k2 = 5012
    k3 = 0.1248
    #####

    Cr = 250e-6 # Radial Clearance [m]

    Wx = 0 # Loading - X direction [N]

    Wy = -757e3 # Loading - Y direction [N]

    ##### Center shaft speed
    xpt = 0  
    ypt = 0
    #####

    x0 = np.array(
        (9.98805447808967e-10, -0.000159030915145932, 0.000636496725369308,
        0.000913538939923720, 0.000690387727463155, 0.000854621895169753,
        0.000679444727719279, 0.000594279954895354, 0.000489138614623373,
        0.000377122288510832, 0.000319527936428236, 0.000340586533490991,
        0.000464320475944696, 0.000657880058055782)		
        )  # Initial equilibrium position

    bearing = Tilting(
        Rs = Rs,
        npad = npad,
        Rp = Rp,
        tpad = tpad,
        betha_p = betha_p,
        sigma = sigma,
        rp_pad = rp_pad,
        wa = wa,
        L = L,
        Tcub = Tcub,
        kt = kt,
        Cp = Cp,
        rho = rho,
        k1 = k1,
        k2 = k2,
        k3 = k3,
        Cr = Cr,
        Wx = Wx,
        Wy = Wy,
        nX = nX,
        nZ = nZ,
        x0 = x0,
    )

    return bearing

if __name__ == "__main__":
    tilting_pad_example()