import numpy as np
from numpy.linalg import pinv
from scipy.linalg import solve
from decimal import Decimal


class Thrust:
    def __init__(
        r1,
        rp,
        teta0,
        tetap,
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
    ):
        self.r1 = r1
        self.rp = rp
        self.teta0 = teta0
        self.tetap = tetap
        self.mi0 = mi0
        self.fz = fz
        self.Npad = Npad
        self.NTETA = NTETA
        self.NR = NR
        # self.war = war
        self.war = wa * (np.pi / 30)
        self.R1 = R1
        self.R2 = R2
        self.TETA1 = TETA1
        self.TETA2 = TETA2
        self.Rp = Rp
        self.TETAp = TETAp
        self.dR = dR
        self.dTETA = dTETA

        # -------------------------------------------------------------------
        # PRE PROCESSING - Preparing auxilary variables and declarations
        aux_dR = np.zeros([nK + 2])
        aux_dR[0 : nK + 1] = np.arange(
            R1 + 0.5 * dR, R2 - 0.5 * dR, self.dR
        )  # vector aux_dR dimensionless

        aux_dTETA = np.zeros([nK + 2])
        aux_dTETA[0 : nK + 1] = np.arange(
            TETA1 + 0.5 * dTETA, TETA2 - 0.5 * dTETA, self.dTETA
        )  # vector aux_dTETA dimensionless

        # ENTRY VARIABLES FROM ANOTHER CODE, STILL TO BE INTEGRATED HERE
        # Pitch angles alpha_r and alpha_p and oil filme thickness at pivot h0
        a_r = x[0]
        # [rad]
        a_s = x[1]
        # [rad]
        h0 = x[2]
        # [m]

        for ii in range(0, self.NR):
            for jj in range(0, self.NTETA):
                MI[jj, ii] = mi0 / mi0  # dimensionless

        # Dimensioneless Parameters
        Ar = a_r * r1 / h0
        As = a_s * r1 / h0
        H0 = h0 / h0

        # -------------------------------------------------------------------
        # PRESSURE FIELD - Solution of Reynolds equation
        kR = 1
        kTETA = 1
        k = 0  # index using for pressure vectorization
        nk = (NR) * (NTETA)
        # number of volumes

        Mat_coef = np.zeros(nk, nk)  # Coefficients Matrix
        b = np.zeros(nk, 1)
        cont = 0

        # for R in range(0, aux_dR)
        for R in np.arange((R1 + 0.5 * dR), (R2 - 0.5 * dR), dR):
            # for TETA in range(0, aux_dTETA):
            for TETA in np.arange((TETA1 + 0.5 * dTETA), (TETA2 - 0.5 * dTETA), dTETA):

                cont = cont + 1
                TETAe = TETA + 0.5 * dTETA
                TETAw = TETA - 0.5 * dTETA
                Rn = R + 0.5 * dR
                Rs = R - 0.5 * dR

                Hne = (
                    H0
                    + As * [Rp - Rn * np.cos(teta0 * [TETAe - TETAp])]
                    + Ar * Rn * np.sin(teta0 * [TETAe - TETAp])
                )
                Hnw = (
                    H0
                    + As * [Rp - Rn * np.cos(teta0 * [TETAw - TETAp])]
                    + Ar * Rn * np.sin(teta0 * [TETAw - TETAp])
                )
                Hse = (
                    H0
                    + As * [Rp - Rs * np.cos(teta0 * [TETAe - TETAp])]
                    + Ar * Rs * np.sin(teta0 * [TETAe - TETAp])
                )
                Hsw = (
                    H0
                    + As * [Rp - Rs * np.cos(teta0 * [TETAw - TETAp])]
                    + Ar * Rs * np.sin(teta0 * [TETAw - TETAp])
                )

                if kTETA == 1 and kR == 1:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA == 1 and kR > 1 and kR < NR:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kTETA == 1 and kR == NR:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = MI[kR, kTETA]
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == 1 and kTETA > 1 and kTETA < NTETA:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA > 1 and kTETA < NTETA and kR > 1 and kR < NR:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == NR and kTETA > 1 and kTETA < NTETA:
                    MI_e = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA + 1])
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = MI[kR, kTETA]
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kR == 1 and kTETA == NTETA:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = MI[kR, kTETA]

                if kTETA == NTETA and kR > 1 and kR < NR:
                    MI_e = MI[kR, kTETA]
                    MI_w = 0.5 * (MI[kR, kTETA] + MI[kR, kTETA - 1])
                    MI_n = 0.5 * (MI[kR, kTETA] + MI[kR + 1, kTETA])
                    MI_s = 0.5 * (MI[kR, kTETA] + MI[kR - 1, kTETA])

                if kTETA == NTETA and kR == NR:
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

                k = k + 1  # vectorization index

                b[k, 0] = dR / (4 * teta0) * (Rn * Hne + Rs * Hse - Rn * Hnw - Rs * Hsw)

                if kTETA == 1 and kR == 1:
                    Mat_coef[k, k] = CP - CS - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == 1 and kR > 1 and kR < NR:
                    Mat_coef[k, k] = CP - CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k + NTETA] = CN
                    Mat_coef[k, k - NTETA] = CS

                if kTETA == 1 and kR == NR:
                    Mat_coef[k, k] = CP - CW - CN
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - NTETA] = CS

                if kR == 1 and kTETA > 1 and kTETA < NTETA:
                    Mat_coef[k, k] = CP - CS
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN

                if kTETA > 1 and kTETA < NTETA and kR > 1 and kR < NR:
                    Mat_coef[k, k] = CP
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN
                    Mat_coef[k, k - NTETA] = CS
                    Mat_coef[k, k + 1] = CE

                if kR == NR and kTETA > 1 and kTETA < NTETA:
                    Mat_coef[k, k] = CP - CN
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + 1] = CE
                    Mat_coef[k, k - NTETA] = CS

                if kR == 1 and kTETA == NTETA:
                    Mat_coef[k, k] = CP - CE - CS
                    Mat_coef[k, k - 1] = CW
                    Mat_coef[k, k + NTETA] = CN

                if kTETA == NTETA and kR > 1 and kR < NR:
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
        kTETA = 1

        # -------------------------------------------------------------------
        # PRESSURE FIELD SOLUTION
        p = np.linalg.solve(Mat_coef, b)
        cont = 0

        for ii in np.arange(self.nZ):
            for jj in np.arange(self.ntheta):
                P[ii, jj] = p[cont]
                cont = cont + 1
                if P[ii, jj] < 0:
                    P[ii, jj] = 0

        for ii in np.range(1, NR):
            for jj in np.range(1, NTETA):
                cont = cont + 1
                P[ii, jj] = p[cont]  # matrix of pressure

        # boundary conditions of pressure
        for ii in np.range(1, NR):
            for jj in np.range(1, NTETA):
                if P[ii, jj] < 0:
                    P[ii, jj] = 0

        Pdim = P * (r1 ** 2) * war * mi0 / (h0 ** 2)  # dimensional pressure

        #            RESULTING FORCE AND MOMENTUM: Equilibrium position
        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
        XR = r1 * (np.arange((R1 + 0.5 * dR), (R2 - 0.5 * dR), dR))
        Xrp = rp * (1 + (np.zeros(ZR.shape)))
        XTETA = teta0 * (np.arange((TETA1 + 0.5 * dTETA), (TETA2 - 0.5 * dTETA), dTETA))

        for ii in range(1, NTETA):
            Mxr[:, ii] = (Pdim[:, ii] * (np.linalg.inv(XR) ** 2)) * np.sin(
                XTETA[ii] - tetap
            )
            Myr[:, ii] = (
                -Pdim[:, ii]
                * np.linalg.inv(XR)
                * np.linalg.inv(XR * np.cos(XTETA[ii] - tetap) - Xrp)
            )
            Frer[:, ii] = Pdim[:, ii] * np.linalg.inv(XR)

        mxr = np.trapz(XR, Mxr)
        myr = np.trapz(XR, Myr)
        frer = np.trapz(XR, Frer)

        mx = np.trapz(XTETA, mxr)
        my = np.trapz(XTETA, myr)
        fre = -np.trapz(XTETA, frer) + fz / Npad

        score = np.norm(
            [mx, my, fre], 2
        )  # the "2" option denotes the equivalent method to the matlab defaut one


        # Equilibrium position
        ar=x(1) #[rad]
        a_s=x(2) #[rad]
        h0=x(3) #[m]
            
        # Viscosity field
        for ii in range(1,NR):
            for jj in range(1,NTETA):
                mi[ii,jj]=mi0 #[Pa.s]
                    
        Ar=ar*r1/h0
        As=a_s*r1/h0

        MI=1/mi0*mi

        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------
        #             PRESSURE FIELD - Solution of Reynolds equation
        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------

        kR=1
        kTETA=1
        k=0  # index using for pressure vectorization
        nk=(NR)*(NTETA) # number of volumes
        
        Mat_coef=np.zeros(nk,nk) #Coefficients Matrix
        b=np.zeros(nk,1)
        cont=0


        # for R in range(0, aux_dR)
        for R in np.arange((R1 + 0.5 * dR), (R2 - 0.5 * dR), dR):
            # for TETA in range(0, aux_dTETA):
            for TETA in np.arange((TETA1 + 0.5 * dTETA), (TETA2 - 0.5 * dTETA), dTETA):

                cont=cont+1
                TETAe=TETA+0.5*dTETA
                TETAw=TETA-0.5*dTETA
                Rn=R+0.5*dR
                Rs=R-0.5*dR
                
                H0[kR,kTETA]=h0/h0+As*(Rp-R*np.cos(teta0*(TETA-TETAp)))+Ar*R*np.sin(teta0*(TETA-TETAp));
                # oil film thickness - faces
                H0ne[kR,kTETA]=h0/h0+As*(Rp-Rn*np.cos(teta0*(TETAe-TETAp)))+Ar*Rn*np.sin(teta0*(TETAe-TETAp))
                H0nw[kR,kTETA]=h0/h0+As*(Rp-Rn*np.cos(teta0*(TETAw-TETAp)))+Ar*Rn*np.sin(teta0*(TETAw-TETAp))
                H0se[kR,kTETA]=h0/h0+As*(Rp-Rs*np.cos(teta0*(TETAe-TETAp)))+Ar*Rs*np.sin(teta0*(TETAe-TETAp))
                H0sw[kR,kTETA]=h0/h0+As*(Rp-Rs*np.cos(teta0*(TETAw-TETAp)))+Ar*Rs*np.sin(teta0*(TETAw-TETAp))
                
                
                if kTETA==1 and kR==1:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= MI[kR,kTETA]
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= MI[kR,kTETA]
                
                if kTETA==1 and kR>1 and kR<NR:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= MI[kR,kTETA]
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                
                if kTETA==1 and kR==NR:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= MI[kR,kTETA]
                    MI_n= MI[kR,kTETA]
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                
                if kR==1 and kTETA>1 and kTETA<NTETA:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= MI[kR,kTETA]
                
                if kTETA>1 and kTETA<NTETA and kR>1 and kR<NR:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                
                if kR==NR and kTETA>1 and kTETA<NTETA:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= MI[kR,kTETA]
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                
                if kR==1 and kTETA==NTETA:
                    MI_e= MI[kR,kTETA]
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= MI[kR,kTETA]
                
                if kTETA==NTETA and kR>1 and kR<NR:
                    MI_e= MI[kR,kTETA]
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                
                if kTETA==NTETA and kR==NR:
                    MI_e= MI[kR,kTETA]
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= MI[kR,kTETA]
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                
                # Coefficients for solving the Reynolds equation
                CE=1/(24*teta0**2*MI_e)*(dR/dTETA)*(H0ne[kR,kTETA]**3/Rn+H0se[kR,kTETA]**3/Rs)
                CW=1/(24*teta0**2*MI_w)*(dR/dTETA)*(H0nw[kR,kTETA]**3/Rn+H0sw[kR,kTETA]**3/Rs)
                CN=Rn/(24*MI_n)*(dTETA/dR)*(H0ne[kR,kTETA]**3+H0nw[kR,kTETA]**3)
                CS=Rs/(24*MI_s)*(dTETA/dR)*(H0se[kR,kTETA]**3+H0sw[kR,kTETA]**3)
                CP=-(CE+CW+CN+CS)
                
                k=k+1; #vectorization index
                
                b[k,1]=dR/(4*teta0)*(Rn*H0ne[kR,kTETA]+Rs*H0se[kR,kTETA]-Rn*H0nw[kR,kTETA]-Rs*H0sw[kR,kTETA])
                
                if kTETA==1 and kR==1:
                    Mat_coef[k,k]=CP-CS-CW
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k+(NTETA)]=CN
                
                if kTETA==1 and kR>1 and kR<NR:
                    Mat_coef[k,k]=CP-CW
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k+(NTETA)]=CN
                    Mat_coef[k,k-(NTETA)]=CS
                
                if kTETA==1 and kR==NR:
                    Mat_coef[k,k]=CP-CW-CN
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-(NTETA)]=CS
                
                if kR==1 and kTETA>1 and kTETA<NTETA:
                    Mat_coef[k,k]=CP-CS
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+(NTETA)]=CN
                
                if kTETA>1 and kTETA<NTETA and kR>1 and kR<NR:
                    Mat_coef[k,k]=CP
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+(NTETA)]=CN
                    Mat_coef[k,k-(NTETA)]=CS
                    Mat_coef[k,k+1]=CE
                
                if kR==NR and kTETA>1 and kTETA<NTETA:
                    Mat_coef[k,k]=CP-CN
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-(NTETA)]=CS
                
                if kR==1 and kTETA==NTETA:
                    Mat_coef[k,k]=CP-CE-CS
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+(NTETA)]=CN
                
                if kTETA==NTETA and kR>1 and kR<NR:
                    Mat_coef[k,k]=CP-CE
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k-(NTETA)]=CS
                    Mat_coef[k,k+(NTETA)]=CN
                
                if kTETA==NTETA and kR==NR:
                    Mat_coef[k,k]=CP-CE-CN
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k-(NTETA)]=CS
                
                kTETA=kTETA+1
                
            kR=kR+1
            kTETA=1

        # %%%%%%%%%%%%%%%% Vectorized pressure field solution %%%%%%%%%%%%%%
        p = np.linalg.solve(Mat_coef, b) 
        cont=0

        for ii in range(1,NR):
            for jj in range(1,NTETA):
                cont=cont+1
                P0[ii,jj]=p(cont) #matrix of pressure

        # boundary conditions of pressure
        for ii in range(1,NR):
            for jj in range(1,NTETA):
                if P0[ii,jj]<0:
                    P0[ii,jj]=0








        #--------------------------------------------------------------------------
        #----------------- Stiffness and Damping Coefficients ---------------------
        #--------------------------------------------------------------------------
        #perturbation frequency [rad/s]
        wp=war
        WP=wp/war

        MI=(1/mi0)*mi

        kR=1
        kTETA=1
        k=0 #index using for pressure vectorization
        nk=(NR)*(NTETA) #number of volumes
        
        Mat_coef=np.zeros(nk,nk) #Coefficients Matrix
        b=np.zeros(nk,1)
        cont=0

        # for R in range(0, aux_dR)
        for R in np.arange((R1 + 0.5 * dR), (R2 - 0.5 * dR), dR):
            # for TETA in range(0, aux_dTETA):
            for TETA in np.arange((TETA1 + 0.5 * dTETA), (TETA2 - 0.5 * dTETA), dTETA):

                cont=cont+1
                TETAe=TETA+0.5*dTETA
                TETAw=TETA-0.5*dTETA
                Rn=R+0.5*dR
                Rs=R-0.5*dR
                
                if kTETA==1 and kR==1:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= MI[kR,kTETA]
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= MI[kR,kTETA]
                    dPdTETAe=(P0[kR,kTETA+1]-P0[kR,kTETA])/dTETA
                    dPdTETAw=P0[kR,kTETA]/(0.5*dTETA)
                    dPdRn=(P0[kR+1,kTETA]-P0[kR,kTETA])/dR
                    dPdRs=P0[kR,kTETA]/(0.5*dR)
                
                if kTETA==1 and kR>1 and kR<NR:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= MI[kR,kTETA]
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                    dPdTETAe=(P0[kR,kTETA+1]-P0[kR,kTETA])/dTETA
                    dPdTETAw=P0[kR,kTETA]/(0.5*dTETA)
                    dPdRn=(P0[kR+1,kTETA]-P0[kR,kTETA])/dR
                    dPdRs=(P0[kR,kTETA]-P0[kR-1,kTETA])/dR
                
                if kTETA==1 and kR==NR:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= MI[kR,kTETA]
                    MI_n= MI[kR,kTETA]
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                    dPdTETAe=(P0[kR,kTETA+1]-P0[kR,kTETA])/dTETA
                    dPdTETAw=P0[kR,kTETA]/(0.5*dTETA)
                    dPdRn=-P0[kR,kTETA]/(0.5*dR)
                    dPdRs=(P0[kR,kTETA]-P0[kR-1,kTETA])/dR
                    
                if kR==1 and kTETA>1 and kTETA<NTETA:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= MI[kR,kTETA]
                    dPdTETAe=(P0[kR,kTETA+1]-P0[kR,kTETA])/dTETA
                    dPdTETAw=(P0[kR,kTETA]-P0[kR,kTETA-1])/dTETA
                    dPdRn=(P0[kR+1,kTETA]-P0[kR,kTETA])/dR
                    dPdRs=P0[kR,kTETA]/(0.5*dR)
                
                if kTETA>1 and kTETA<NTETA and kR>1 and kR<NR:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                    dPdTETAe=(P0[kR,kTETA+1]-P0[kR,kTETA])/dTETA
                    dPdTETAw=(P0[kR,kTETA]-P0[kR,kTETA-1])/dTETA
                    dPdRn=(P0[kR+1,kTETA]-P0[kR,kTETA])/dR
                    dPdRs=(P0[kR,kTETA]-P0[kR-1,kTETA])/dR
                
                if kR==NR and kTETA>1 and kTETA<NTETA:
                    MI_e= 0.5*(MI[kR,kTETA]+MI[kR,kTETA+1])
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= MI[kR,kTETA]
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                    dPdTETAe=(P0[kR,kTETA+1]-P0[kR,kTETA])/dTETA
                    dPdTETAw=(P0[kR,kTETA]-P0[kR,kTETA-1])/dTETA
                    dPdRn=-P0[kR,kTETA]/(0.5*dR)
                    dPdRs=(P0[kR,kTETA]-P0[kR-1,kTETA])/dR
                
                if kR==1 and kTETA==NTETA:
                    MI_e= MI[kR,kTETA]
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= MI[kR,kTETA]
                    dPdTETAe=-P0[kR,kTETA]/dTETA
                    dPdTETAw=(P0[kR,kTETA]-P0[kR,kTETA-1])/dTETA
                    dPdRn=(P0[kR+1,kTETA]-P0[kR,kTETA])/dR
                    dPdRs=P0[kR,kTETA]/dR
                
                if kTETA==NTETA and kR>1 and kR<NR:
                    MI_e= MI[kR,kTETA]
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= 0.5*(MI[kR,kTETA]+MI[kR+1,kTETA])
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                    dPdTETAe=-P0[kR,kTETA]/(0.5*dTETA)
                    dPdTETAw=(P0[kR,kTETA]-P0[kR,kTETA-1])/dTETA
                    dPdRn=(P0[kR+1,kTETA]-P0[kR,kTETA])/dR
                    dPdRs=(P0[kR,kTETA]-P0[kR-1,kTETA])/dR
                
                if kTETA==NTETA and kR==NR:
                    MI_e= MI[kR,kTETA]
                    MI_w= 0.5*(MI[kR,kTETA]+MI[kR,kTETA-1])
                    MI_n= MI[kR,kTETA]
                    MI_s= 0.5*(MI[kR,kTETA]+MI[kR-1,kTETA])
                    dPdTETAe=-P0[kR,kTETA]/(0.5*dTETA)
                    dPdTETAw=(P0[kR,kTETA]-P0[kR,kTETA-1])/dTETA
                    dPdRn=-P0[kR,kTETA]/(0.5*dR)
                    dPdRs=(P0[kR,kTETA]-P0[kR-1,kTETA])/dR
                
                As_ne=1
                As_nw=1
                As_se=1
                As_sw=1
                
                # G1=dhpivotdR=0
                G1_ne=0
                G1_nw=0
                G1_se=0
                G1_sw=0
                
                # Gs=dhpivotdTETA=0
                G2_ne=0
                G2_nw=0
                G2_se=0
                G2_sw=0
                
                # Coefficients for solving the Reynolds equation
                CE_1=1/(24*teta0**2*MI_e)*(dR/dTETA)*(As_ne*H0ne[kR,kTETA]**3/Rn+As_se*H0se[kR,kTETA]**3/Rs)
                CE_2=dR/(48*teta0**2*MI_e)*(G2_ne*H0ne[kR,kTETA]**3/Rn+G2_se*H0se[kR,kTETA]**3/Rs)
                CE=CE_1+CE_2
                
                CW_1=1/(24*teta0**2*MI_w)*(dR/dTETA)*(As_nw*H0nw[kR,kTETA]**3/Rn+As_sw*H0sw[kR,kTETA]**3/Rs)
                CW_2=-dR/(48*teta0**2*MI_w)*(G2_nw*H0nw[kR,kTETA]**3/Rn+G2_sw*H0sw[kR,kTETA]**3/Rs)
                CW=CW_1+CW_2
                
                CN_1=Rn/(24*MI_n)*(dTETA/dR)*(As_ne*H0ne[kR,kTETA]**3+As_nw*H0nw[kR,kTETA]**3)
                CN_2=Rn/(48*MI_n)*(dTETA)*(G1_ne*H0ne[kR,kTETA]**3+G1_nw*H0nw[kR,kTETA]**3)
                CN=CN_1+CN_2
                
                CS_1=Rs/(24*MI_s)*(dTETA/dR)*(As_se*H0se[kR,kTETA]**3+As_sw*H0sw[kR,kTETA]**3)
                CS_2=-Rs/(48*MI_s)*(dTETA)*(G1_se*H0se[kR,kTETA]**3+G1_sw*H0sw[kR,kTETA]**3)
                CS=CS_1+CS_2
                
                CP=-(CE_1+CW_1+CN_1+CS_1)+(CE_2+CW_2+CN_2+CS_2)
                
                B_1=(Rn*dTETA/(8*MI_n))*dPdRn*(As_ne*H0ne[kR,kTETA]**2+As_nw*H0nw[kR,kTETA]**2)-(Rs*dTETA/(8*MI_s))*dPdRs*(As_se*H0se[kR,kTETA]**2+As_sw*H0sw[kR,kTETA]**2)
                B_2=(dR/(8*teta0**2*MI_e))*dPdTETAe*(As_ne*H0ne[kR,kTETA]**2/Rn+As_se*H0se[kR,kTETA]**2/Rs)-(dR/(8*teta0**2*MI_w))*dPdTETAw*(As_nw*H0nw[kR,kTETA]**2/Rn+As_nw*H0sw[kR,kTETA]**2/Rs)
                B_3=dR/(4*teta0)*(As_ne*Rn+As_se*Rs)-dR/(4*teta0)*(As_nw*Rn+As_sw*Rs)
                B_4=i*WP*dR*dTETA/4*(Rn*As_ne+Rn*As_nw+Rs*As_se+Rs*As_sw)
                

                k=k+1 #vectorization index
                
                b[k,1]=-(B_1+B_2)+B_3+B_4
                
                if kTETA==1 and kR==1:
                    Mat_coef[k,k]=CP-CW-CS
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k+(NTETA)]=CN
                
                if kTETA==1 and kR>1 and kR<NR:
                    Mat_coef[k,k]=CP-CW
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k+(NTETA)]=CN
                    Mat_coef[k,k-(NTETA)]=CS
                
                if kTETA==1 and kR==NR:
                    Mat_coef[k,k]=CP-CW-CN
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-(NTETA)]=CS
                
                if kR==1 and kTETA>1 and kTETA<NTETA:
                    Mat_coef[k,k]=CP-CS
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+(NTETA)]=CN
                
                if kTETA>1 and kTETA<NTETA and kR>1 and kR<NR:
                    Mat_coef[k,k]=CP
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+(NTETA)]=CN
                    Mat_coef[k,k-(NTETA)]=CS
                    Mat_coef[k,k+1]=CE
                
                if kR==NR and kTETA>1 and kTETA<NTETA:
                    Mat_coef[k,k]=CP-CN
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+1]=CE
                    Mat_coef[k,k-(NTETA)]=CS
                
                if kR==1 and kTETA==NTETA:
                    Mat_coef[k,k]=CP-CE-CS
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k+(NTETA)]=CN
                
                if kTETA==NTETA and kR>1 and kR<NR:
                    Mat_coef[k,k]=CP-CE
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k-(NTETA)]=CS
                    Mat_coef[k,k+(NTETA)]=CN
                
                if kTETA==NTETA and kR==NR:
                    Mat_coef[k,k]=CP-CE-CN
                    Mat_coef[k,k-1]=CW
                    Mat_coef[k,k-(NTETA)]=CS
                
                kTETA=kTETA+1
                
            kR=kR+1
            kTETA=1
            

        %%%%%%%%%%%%%%%%%%%%%% Pressure field solution %%%%%%%%%%%%%%%%%%%%

        p=Mat_coef\b; %solve pressure vectorized

        cont=0;

        for ii=1:NR
            for jj=1:NTETA
                cont=cont+1;
                P(ii,jj)=p(cont); %matrix of pressure
            end
        end

        Pdim=P*(r1**2)*war*mi0/(h0**3); %dimensional pressure

        % -------------------------------------------------------------------------
        % -------------------------------------------------------------------------
        %            RESULTING FORCE AND MOMENTUM: Equilibrium position
        % -------------------------------------------------------------------------
        % -------------------------------------------------------------------------

        XR=r1*(R1+0.5*dR:dR:R2-0.5*dR);

        Xrp=rp*ones(size(XR));

        XTETA=teta0*(TETA1+0.5*dTETA:dTETA:TETA2-0.5*dTETA);

        for ii=1:NTETA
            Mxr(:,ii)=(Pdim(:,ii).*(XR'.**2)).*sin(XTETA(ii)-tetap);
            Myr(:,ii)=-Pdim(:,ii).*XR'.*(XR.*cos(XTETA(ii)-tetap)-Xrp)';
            Frer(:,ii)=Pdim(:,ii).*XR';
        end

        mxr=trapz(XR,Mxr);
        myr=trapz(XR,Myr);
        frer=trapz(XR,Frer);

        mx=-trapz(XTETA,mxr);
        my=-trapz(XTETA,myr);
        fre=-trapz(XTETA,frer);

        return







        K=Npad*real(kk_zz); %Stiffness Coefficient
        C=Npad*1/wp*imag(kk_zz); %Damping Coefficient

        %----- Output values----%
        %results - Pmax [Pa]- hmax[m] - hmin[m] - h0[m] 
        Pmax=max(max(PPdim)) 
        hmax=max(max(h0*H0))
        hmin=min(min(h0*H0))
        h0


