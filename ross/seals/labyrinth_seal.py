import numpy as np
import ccp
import math
import sys
from scipy.linalg import lu_factor, lu_solve
from numpy.linalg import cond
import multiprocessing 
from ross import SealElement
from ross.units import check_units
import multiprocessing


class LabyrinthSeal(SealElement):
    """Calculate labyrinth seal with model based on Laby3.

    Parameters
    ----------
    n : int
        Node in which the seal will be located.
    frequency : float, pint.Quantity
        Shaft rotational speed (rad/s).
    inlet_pressure : float
        Inlet pressure (Pa).
    outlet_pressure : float
        Outlet pressure (Pa).
    inlet_temperature : float
        Inlet temperature (deg K).
    pre_swirl_ratio : float
        Inlet swirl velocity ratio.
        Positive values for swirl with shaft rotation and negative values for
        swirl against shaft rotations.
    frequency : list, pint.Quantity
        List with shaft speeds (rad/s).
    n_teeth : int
        Number of teeth (throttlings).
        Needs to be <= 30.
    shaft_radius : float, pint.Quantity
        Radius of shaft (m).
    radial_clearance : float, pint.Quantity
        Nominal radial clearance (m)
    pitch : float, pint.Quantity
        Seal pitch (length of land) or axial cavity length (m).
    tooth_height : float, pint.Quantity
        Height of seal strip (m).
    tooth_width : float, pint.Quantity
        Thickness of throttle (tip-width) (m), used in mass flow calculation.
    seal_type : str
        Indicates where labyrinth teeth are located.
        Specify 'rotor' if teeth are on rotor only.
        Specify 'stator' if teeth are on stator only.
        Specify 'inter' for interlocking type labyrinths.
    gas_composition : dict
        Gas composition as a dictionary {component: molar_fraction}.
    if gas composition not 'AIR':
        r: float
            gas constant
        gamma: float
            ratio of specific heats
        tz: list float
            tz[0]: temperature at state 1
            tz[1]: temperature at state 2
        muz: list float
            muz[0]: dynamic viscoosity at state 1
            muz[1]: dynamic viscoosity at state 2
    analz: string
        analz indicates what will be analysed.
        Specify "FULL" for rotordynamic calculation and leakage analysis
        Specify "LEAKAGE" for leakage analysis only
    nprt: integer
        Number of parameters to be printed in the output.
        1 maximum
        5 minimum
    iopt1: integer
        Use or no use of tangential momentum parameters introduced by Jenny and Kanki
        Specify value 0 to not use parameters
        Specify value 1 to use parameters

    
    Examples
    --------
    >>> from ross.seals.labyrinth_seal import LabyrinthSeal
    >>> seal = LabyrinthSeal(
    ...     n=0,
    ...     inlet_pressure=308000,
    ...     outlet_pressure=94300,
    ...     inlet_temperature=283.15,
    ...     pre_swirl_ratio=0.98,
    ...     frequency=Q_([5000, 8000, 11000], "RPM"),
    ...     n_teeth=16,
    ...     shaft_radius=Q_(72.5,"mm"),
    ...     radial_clearance=Q_(0.3,"mm"),
    ...     pitch=Q_(3.175,"mm"),
    ...     tooth_height=Q_(3.175,"mm"),
    ...     tooth_width=Q_(0.1524,"mm"),
    ...     seal_type="inter",
    ... )
    """

    @check_units
    def __init__(
        self,
        n = None,
        inlet_pressure = None,
        outlet_pressure = None,
        inlet_temperature = None,
        pre_swirl_ratio = None,
        frequency = None,
        n_teeth= None,
        shaft_radius = None,
        radial_clearance = None,
        pitch = None,
        tooth_height = None,
        tooth_width = None,
        seal_type = None,
        gas_composition = {"AIR":1.0},
        r = 287,
        gamma = 1.4,
        tz = None,
        muz = None,
        analz = "FULL",
        nprt = 1,
        iopt1 = 0,
        **kwargs,
    ):
    
        self.gas_composition = gas_composition
        self.tz = [0] * 2
        self.muz = [0] * 2
        
        state_in = ccp.State.define(
            p=inlet_pressure, T=inlet_temperature, fluid=self.gas_composition
        )
        state_out = ccp.State.define(
            p=outlet_pressure, h=state_in.h(), fluid=self.gas_composition
        )

        # Calculate
        # gas constant : float -> Gas constant (J / kg degK
        self.r = (state_in.gas_constant() / state_in.molar_mass()).m
        print(f"R {self.r}")
        # ratio_specific_heats : float -> Ratio of specific heats.
        self.gamma = (state_in.cp() / state_in.cv()).m
        print(f"gamma {self.gamma}")
        # tz1 : float -> Temperature at state 1 (deg K).
        self.tz[0] = state_in.T().m
        # muz1 : float -> Dynamic viscosity at state 1 (kg/(m s))
        self.muz[0] = state_in.viscosity().m
        # tz2 : float -> Temperature at state 2 (deg K).
        self.tz[1] = state_out.T().m
        # muz2 : float -> Dynamic viscosity at state 2 (kg/(m s))
        self.muz[1] = state_out.viscosity().m

        self.n = n
        self.inlet_pressure = inlet_pressure
        self.outlet_pressure = outlet_pressure
        self.inlet_temperature = inlet_temperature
        self.pre_swirl_ratio = pre_swirl_ratio
        self.n_teeth = n_teeth
        self.shaft_radius = shaft_radius
        self.radial_clearance = radial_clearance
        self.pitch = pitch
        self.tooth_height = tooth_height
        self.tooth_width = tooth_width
        self.seal_type = seal_type
       
        self.analz = analz
        self.nprt = nprt
        self.iopt1 = iopt1

        self.m_x = 61
        self.pitch = [pitch] * self.m_x
        self.radial_clearance = [radial_clearance] * self.m_x
        self.tooth_height = [tooth_height] * self.m_x
        self.pr = [0] * self.m_x
        self.tooth_width = [tooth_width] * self.m_x

        self.p = [0] * self.m_x
        self.v = [0] * self.m_x
        self.w = [0] * self.m_x
        self.p1 = [0] * self.m_x
        self.v1 = [0] * self.m_x
        self.t = [0] * self.m_x
        self.rho = [0] * self.m_x
        self.taus = [0] * self.m_x
        self.taur = [0] * self.m_x
        self.gm = [[0 for _ in range(500)] for _ in range(1000)]
        self.rhs = [[0 for _ in range(2)] for _ in range(1000)]
        self.cg = [[0 for _ in range(self.m_x)] for _ in range(9)]
        self.cx = [[0 for _ in range(self.m_x)] for _ in range(8)]
        self.vin = [0] * self.m_x
        self.vout = [0] * self.m_x
        self.kout = [0] * self.m_x
        self.vers = "1.1"
        self.vdate1 = '08/08/24'
        self.vtime1 = '22:21:00'

        coefficients_dict = {}
        if kwargs.get("kxx") is None: 
            pool = multiprocessing.Pool()
            coefficients_dict_list = pool.map(self.run, frequency)
            coefficients_dict = {k: [] for k in coefficients_dict_list[0].keys()}
            for d in coefficients_dict_list:
                for k in coefficients_dict:
                    coefficients_dict[k].append(d[k])

        super().__init__(
           self.n,
           frequency= frequency,
           **coefficients_dict,
           **kwargs,
        )

    def derv(self):
        c = 0
        error = 10000
        tol = 1 * 10**-7
        guess_low = 0.001
        guess = 0.8
        guess_high = 0.99
        n = 0
        while n < self.n_teeth:
            r = guess
            deriv_num = -2 * (n + 1) + 2 * r * math.log(r) + 1 / r - r
            deriv_den = ((1 - r**2)**0.5) * ((n - math.log(r))**1.5)
            deriv = deriv_num / deriv_den
            error = -deriv
            while abs(error) > tol:
                if error < 0:
                    guess_low = guess
                    guess = (guess + guess_high) / 2
                if error > 0:
                    guess_high = guess
                    guess = (guess + guess_low) / 2
                r = guess
                deriv_num = -2 * (n + 1) + 2 * r * math.log(r) + 1 / r - r
                deriv_den = ((1 - r**2)**0.5) * ((n - math.log(r))**1.5)
                deriv = deriv_num / deriv_den
                error = -deriv
            self.r_choke[n] = guess
            self.tooth_height_choke[n] = ((1 - self.r_choke[n]**2) / ((n + 1) - math.log(self.r_choke[n])))**0.5
            n += 1
            error = 10000
            guess_low = 0.001
            guess = 0.8
            guess_high = 0.99
        if self.pg < self.r_choke[self.n_teeth]:
            self.tooth_heighteta_nt = self.tooth_height_choke[self.n_teeth]
        else:
            self.tooth_heighteta_nt = ((1 - self.pg**2) / (self.n_teeth - math.log(self.pg)))**0.5
        return

    def setup(self):
        self.epslon = 0.6
        self.awrl = self.epslon * self.radial_clearance[0]
        self.tooth_heightwrl = self.epslon * self.radial_clearance[0]
        self.nc = self.n_teeth - 1
        self.np = self.n_teeth + 1

        for i in range(1, self.n_teeth):
            self.radial_clearance[i] = self.radial_clearance[0]
            self.tooth_height[i] = self.tooth_height[0]
            self.tooth_width[i] = self.tooth_width[0]

        for i in range(1, self.nc):
            self.pitch[i] = self.pitch[0]

        self.ndof = 8 * self.nc
        self.nbw = 33
        self.nbc = 17

        for i in range(0, self.np):
            self.w[i] = 0
            self.pr[i] = 0
            self.p[i] = 0
            self.v[i] = 0
            self.p1[i] = 0
            self.v1[i] = 0
            self.rho[i] = 0
            self.t[i] = self.inlet_temperature

        self.pi = math.pi
        self.pg = self.outlet_pressure / self.inlet_pressure
        self.omega = self.frequency 

        return

    def Vermes(self):
        sg = self.tooth_width[0] / self.radial_clearance[0]
        self.alphav = 0.67675 - (0.08519 * sg) + (0.0878 * (sg**2)) - (0.01819 * (sg**3)) + (0.00111 * (sg**4))
        self.vnu = 8.52 / (((self.pitch[0] - self.tooth_width[0]) / self.radial_clearance[0]) + 7.23)
        if self.seal_type == 'inter':
            self.vnu = 0
        if self.vnu >= 1:
            with open("output.txt", "a") as arquivo:
                arquivo.write("ERROR, alpha maior que 1 \n")
            return
        vg = 1 / (1 - self.vnu)**0.5
        self.r_choke = [0] * self.m_x
        self.tooth_height_choke = [0] * self.m_x
        self.derv()
        self.gve = 1.014 * self.alphav * vg * self.tooth_heighteta_nt

        if self.seal_type == 'inter':
            self.gve = self.gve / 1.014
        self.mdotv = self.gve * self.inlet_pressure * self.radial_clearance[0] / (self.r * self.inlet_temperature)**0.5
        leakv = self.mdotv * 2 * self.pi * (self.shaft_radius + 0.5 * self.radial_clearance[0])

        with open("output.txt", "a") as arquivo:
            arquivo.write("Computed Leakage Coefficients & Parameters: \n \n \n \n")
            arquivo.write("Vermes/Modified Egli Method: \n \n")
            arquivo.write(f"{'   Contraction Coefficient':<40} {self.alphav:>15.8f} kg/s \n")
            arquivo.write(f"{'   Mass Flow Number':<40} {self.gve:>15.8f} kg/s \n")
            arquivo.write(f"{'   Leakage Per Cir. Length':<40} {self.mdotv:>15.8f} kg/(m s)\n")
            arquivo.write(f"{'   Leakage':<40} {leakv:>15.8f} kg/s \n \n")
        self.mdot = self.mdotv

    def ZPRES(self):
        prgs = [0]*3
        fpr  = [0]*3
        gam1 = 1 / self.gamma
        gam2 = (self.gamma - 1) / self.gamma
        gam3 = 2 / gam2
        gam4 = self.r * gam3
        gam5 = 1 / gam2
        gam6 = 1 / (self.gamma - 1)
        gam7 = 2 / (self.gamma + 1)
        gam8 = self.gamma * 2 / (self.gamma + 1)

        tol1 = 1*10**(-8)
        itmx1 = 100
        ndex1 = 0

        tol_outlet_pressure = 0.00001
        tol_choked = 0.005

        tol_p = 1*10**(-4)
        a2998 = True
        c = 0


        while True:
            asaida = True
            #2998
            if a2998 == True:
                mdot_high = self.mdot*5
                mdot_low = 0
                a2998 = False
            # 4000 
            if ndex1 < 1:
                self.w[0] = 0
                self.p[0] = self.inlet_pressure
                self.rho[0] = self.p[0]/(self.r * self.t[0])
                prold = 1*10**(10)
                chok1 = (gam7 + (self.vnu*self.w[self.n_teeth-1]*self.w[self.n_teeth-1]/(gam4*self.t[self.n_teeth-1])))
                chok2 = chok1**gam5
            #check
            # do 2000
            for i in range(1, self.n_teeth+1):
                if i == self.n_teeth:
                    prgs[0] = chok2
                else:
                    prgs[0] = chok2
            
                prgs[1] = 0.9999999
                # do 150
                for j in range(0, 2):
                    fpr[j] = self.alphav * self.radial_clearance[i-1] * self.rho[i-1] * (prgs[j]**gam1) * (((self.vnu*self.w[i-1])*self.w[i-1]) + (gam4*self.t[i-1]*(1-(prgs[j]**gam2))))**0.5
                    fpr[j] = self.mdot - fpr[j]
                # 150
                if fpr[0] > 0:
                    fpr[0] = 0
                # do 1000
                for itn in range(0, itmx1):
                    prgs[2] = (prgs[0]*fpr[1] - prgs[1]*fpr[0])/(fpr[1]-fpr[0])
                    fpr[2] = self.alphav * self.radial_clearance[i-1] * self.rho[i-1] * (prgs[2]**gam1) * math.sqrt((self.vnu * self.w[i-1] * self.w[i-1]) + (gam4 * self.t[i-1] * (1. - (prgs[2]**gam2))))
                    #check
                    a2001 = True
                    if prgs[2] <= chok2:
                        a2001 = False
                        error_outlet_pressure = 0
                    # goto 2001
                        break
                    fpr[2] = self.mdot - fpr[2] 
                    
                    if fpr[2]*fpr[0] < 0:
                    # 200
                        prgs[1] = prgs[2]
                        fpr[1] = fpr[2]
                        # goto 500
                    # 300
                    elif fpr[2]*fpr[0] == 0:
                        if fpr[0] == 0:
                            prgs[2] = prgs[0]
                            fpr[2] = fpr[0]
                            prgs[1] = prgs[0]
                            fpr[1] = fpr[0]
                        else:
                            prgs[1] = prgs[2]
                            fpr[1] = fpr[2]
                            prgs[0] = prgs[2]
                            fpr[0] = fpr[2]
                        # goto 1500
                            break
                    # 400
                    elif fpr[2]*fpr[0] > 0:
                        prgs[0] = prgs[2]
                        fpr[0] = fpr[2]
                    # 500
                    if abs((prgs[2]-prold)/prgs[2]) <= tol1:
                        # go to 1500
                        break
                    prold = prgs[2]
                
                if a2001 == False:
                    break
                # 1500
                if (abs(fpr[2]) > tol_p):
                    print("Pressuere Convergence Error at Station", i)
                self.pr[i-1] = prgs [2]
                self.p[i] = self.pr[i-1] * self.p[i-1]
                self.w[i] = (self.mdot * self.r * self.t[i-1])/(self.alphav * self.p[i-1]*(self.pr[i-1]**gam1)*self.radial_clearance[i-1])
                self.rho[i] = self.rho[i-1] * (self.pr[i-1] ** gam1)
                self.t[i] = self.t[i-1] * (self.pr[i-1] ** gam2)
            

            if a2001 == True:
                i = self.np-1
                chock1 = (gam7 + (self.vnu*self.w[i-1]*self.w[i-1]/(gam4*self.t[i-1])))
                chock2 = chock1 ** gam5
                error_outlet_pressure = (self.p[self.np-1]-self.outlet_pressure)/self.outlet_pressure
                if ndex1 == 1:
                    break
            if (abs(error_outlet_pressure) >= tol_outlet_pressure and abs(self.pr[self.np-2] -chock2)/chock2 > tol_choked) or a2001 == False:
                if error_outlet_pressure < 0 or a2001 == False:
                    # 2001
                    #print("1")
                    mdot_tmp = self.mdot
                    self.mdot = (mdot_low + self.mdot)/2
                    mdot_high = mdot_tmp
                    if (self.mdot-mdot_tmp)/self.mdot == 0:
                        print("reset iteration")
                        ndex1 = 2
                        a2998 = True
                elif error_outlet_pressure>=0:
                    #print("2")
                    mdot_tmp = self.mdot
                    self.mdot = (mdot_high + self.mdot)/2
                    mdot_low = mdot_tmp
                    if(self.mdot-mdot_tmp)/self.mdot == 0:
                        print("reset iteration")
                        ndex1=2
                        a2998 = True
                asaida = False
            if asaida == True:
                break

        i = self.np -1
        chock1 = (gam7 + (self.vnu*self.w[i-1]*self.w[i-1]/(gam4*self.t[i-1])))
        chok2 = chok1 ** gam5

        if ndex1 != 1:
            leak = self.mdot * 2 * self.pi * (self.shaft_radius + 0.5*self.radial_clearance[0])

        if ndex1 != 1 and abs(self.pr[self.np-2]-chok2)/chok2 <= tol_choked:
            with open("output.txt", "a") as arquivo:        
                arquivo.write("Flow Chocked in Last Thottle")
                ndex1 = 1
        if (self.pr[self.n_teeth-1]) > 1:
            with open("output.txt", "a") as arquivo:        
                arquivo.write("ERROR IN LEAKAGE CALCULATION")
        if self.nprt > 4:
            return

        with open("output.txt", "a") as arquivo:        
                arquivo.write("Benevenuti Method: \n \n")
                arquivo.write(f"{'   Leakage Per Cir. Length':<40} {self.mdot:>15.8f} kg/(m s)\n")
                arquivo.write(f"{'   Leakage':<40} {leak:>15.8f} kg/s \n")
                arquivo.write("Benvenuti Result Used Throughout Program \n \n \n \n")
                header_1 = "#      Pr Ratio   Pressure       Norm Pr    Dens     Temp     Ax Vel  Ax Mach\n"
                header_2 = "                  N/(m**2)                kg/(m**3)  deg K    m/s\n"
                arquivo.write(header_1)
                arquivo.write(header_2)


        for i in range(0, self.np):
            mach = self.w[i]/(self.gamma * self.r *self.t[i])**0.5
            pnorm = self.p[i]/self.inlet_pressure
            linha_formatada = (
                f"{i+1:<5}  {self.pr[i]:<10.4f} {self.p[i]:<14.6E} {pnorm:<10.4f} "
                f"{self.rho[i]:<8.3f} {self.t[i]:<8.1f} {self.w[i]:<8.2f} {mach:<8.2f}\n"
            )   
            with open("output.txt", "a") as arquivo:        
                arquivo.write(linha_formatada)
        with open("output.txt", "a") as arquivo:        
            arquivo.write("\n \n")
        return

    def zvel_JEN(self):
        
        vgs = [0]*3
        fv  = [0]*3
        rov = [0]*3
        tr  = [0]*3
        ts  = [0]*3

        if self.omega == 0 and self.inlet_swirl_velocity == 0:
            return
        jc = 0
        if self.seal_type == 'stator':
            jc = 0.15
        if self.seal_type == 'rotor':
            jc = 0.35
        if self.seal_type == 'inter':
            jc = 0.90
        if jc == 0:
            print("Improper selection of labyrinth type")
            sys.exit()
        bmr = -0.25
        bms = -0.25
        bnr = 0.079
        bns = 0.079

        if self.seal_type == 'inter':
            ar = (1*self.tooth_height[0] + self.pitch [0])/self.pitch[0]
            as_py = (1*self.tooth_height[0] + self.pitch [0])/self.pitch[0]
        elif self.seal_type == 'stator':
            as_py = (2*self.tooth_height[0]+self.pitch[0])/self.pitch[0]
            ar = 1
        else:
            ar = (2*self.tooth_height[0]+self.pitch[0])/self.pitch[0]
            as_py = 1
        
        dh = 2*(self.radial_clearance[0]+self.tooth_height[0])*self.pitch[0]/(self.radial_clearance[0]+self.tooth_height[0]+self.pitch[0])
        area = (self.tooth_height[0]+self.radial_clearance[0])*self.pitch[0]

        self.v[0] = self.inlet_swirl_velocity
        self.vin[0] = self.inlet_swirl_velocity
        self.vout[0] = self.inlet_swirl_velocity
        self.taur[0] = 0
        self.taus[0] = 0
        itmx2 = 40
        tol2 = 1*10**(-8)
        vold = 1*10**(10)

        if self.gas_composition == 'AIR':
            sb = 1.426086*10**(-6)
            ss = 100
        else:
            phi1 = (self.tz[0] ** 1.5)/ self.muz[0]
            phi2 = (self.tz[1] ** 1.5)/ self.muz[1]
            sb = (self.tz[1] - self.tz[0])/ (phi2 - phi1)
            ss = (sb*phi1) - self.tz[0]
        for i in range(1, self.n_teeth):
            self.vin[i] = self.vout[i-1]
            mu = sb * (self.t[i])**0.5/(1 + (ss/self.t[i]))
            self.nu = mu/ self.rho[i]
            vgs[1] =(self.gamma*self.r*self.t[i])**0.5
            vgs[0]=-vgs[1]

            rov[0] = (self.shaft_radius*self.omega) - vgs[0]
            rov[1] = (self.shaft_radius*self.omega) - vgs[1]

            for j in range(0,2):
                tr[j] = 0.5 * self.rho[i]*rov[j]*rov[j]*bnr*((abs(rov[j])*dh/self.nu)**bmr)*math.copysign(1, rov[j])
                ts[j] = 0.5 * self.rho[i]*vgs[j]*vgs[j]*bns*((abs(vgs[j])*dh/self.nu)**bms)*math.copysign(1, vgs[j])
                fv[j] = (self.mdot*jc*(vgs[j]-self.vin[i]))-(self.pitch[0]*(tr[j]*ar-ts[j]*as_py))

            for itn2 in range(0, itmx2):
                vgs[2] = (vgs[0]*fv[1] - vgs[1]*fv[0])/(fv[1] - fv[0])
                rov[2] = (self.shaft_radius*self.omega) - vgs[2]
                tr[2] = 0.5 * self.rho[i]*rov[2]*rov[2]*bnr*((abs(rov[2])*dh/self.nu)**bmr)*math.copysign(1, rov[2])
                ts[2] = 0.5 * self.rho[i]*vgs[2]*vgs[2]*bns*((abs(vgs[2])*dh/self.nu)**bms)*math.copysign(1, vgs[2])
                fv[2] = (self.mdot*(vgs[2]-self.vin[i]))-(self.pitch[0]*(tr[2]*ar-ts[2]*as_py))

                if fv[2]*fv[0]<0:
                    vgs[1] = vgs[2]
                    fv[1] = fv[2]
                    rov[1] = rov[2]
                    tr[1] = tr[2]
                    ts[1] = ts[2]

                    if abs((vgs[2]-vold)/vgs[2]) > tol2:
                        vold = vgs[2]
                    else:
                        break
                        
                elif fv[2]*fv[0] == 0:
                    if fv[1]== 0:
                        vgs[1] = vgs[0]
                        fv[1] = fv[0]
                        vgs[2] = vgs[0]
                        fv[2] = fv[0]
                        rov[1] = rov[0]
                        tr[1] = tr[0]
                        ts[1] = ts[0]
                        rov[2] = rov[0]
                        tr[2] = tr[0]
                        ts[2] = ts[0]

                    else:
                        vgs[1] = vgs[2]
                        fv[1] = fv[2]
                        vgs[0] = vgs[2]
                        fv[0] = fv[2]
                        rov[1] = rov[2]
                        tr[1] = tr[2]
                        ts[1] = ts[2]
                        rov[0] = rov[2]
                        tr[0] = tr[2]
                        ts[0] = ts[2]
                    break
                else:
                    vgs[0] = vgs[2]
                    fv[0] = fv[2]
                    rov[0] = rov[2]
                    tr[0] = tr[2]
                    ts[0] = ts[2]

                    if abs((vgs[2]-vold)/vgs[2]) > tol2:
                        vold = vgs[2]
                    else:
                        break
            if abs(fv[2] > 0.001):
                print('Velocity Convergence Error at station', i) 
            self.v[i] = vgs[2]
            self.vout[i] = self.vin[i]*(1-jc) + self.v[i]*jc
            self.kout[i] = self.vout[i]/self.v[i]
            self.taur[i] = tr[2]
            self.taus[i] = ts[2]

            self.cg[0][i] = area / (self.r*self.t[i])
            self.cg[1][i] = (self.v[i]/self.shaft_radius)*self.cg[0][i]
            self.cg[2][i] = (self.p[i]/self.shaft_radius)*self.cg[0][i]
            self.cg[3][i] = self.mdot*self.p[i]*(1/(self.p[i]**2 - self.p[i+1]**2) + 1/(self.p[i-1]**2 - self.p[i]**2))
            self.cg[4][i] = -self.mdot * self.p[i+1]/(self.p[i]**2-self.p[i+1]**2)
            self.cg[5][i] = -self.rho[i]*self.pitch[1]
            self.cg[6][i] = (self.v[i]/self.shaft_radius)*self.cg[5][i]
            self.cg[7][i] = -self.mdot*self.p[i-1]/(self.p[i-1]**2 - self.p[i]**2)
            self.cg[8][i] = -self.cg[7][i]*jc*(self.v[i]-self.vin[i])

            self.cx[0][i] = area/self.shaft_radius
            self.cx[1][i] = self.rho[i]*area
            self.cx[2][i] = (self.v[i]/self.shaft_radius)*self.cx[1][i]
            cxx1 = ((2+bms)*self.taus[i]*as_py*self.pitch[0])/self.v[i]
            cxx2 = ((2+bmr)*self.taur[i]*ar*self.pitch[0])/rov[2]
            self.cx[3][i] = self.mdot * self.kout[i] + cxx1 +cxx2
            self.cx[4][i] = -self.mdot * self.kout[i-1]
            self.cx[5][i] = (self.mdot/self.p[i+1])*self.v[i]
            self.cx[5][i] = 0
            self.cx[6][i] = -self.mdot*jc*(self.v[i]-self.vin[i])*self.p[i]/(self.p[i-1]**2 - self.p[i]**2) + ((self.taus[i]*as_py-self.taur[i]*ar)*(self.pitch[1]/self.p[i]))
            cxx3 = (-bms*self.taus[i]*as_py + bmr*self.taur[i]*ar)*(self.pitch[0]*dh/(2*(self.radial_clearance[0]+self.tooth_height[0])**2))
            self.cx[7][i] = (self.mdot/self.radial_clearance[0])*jc*(self.vin[i]-self.v[i]) + cxx3

        if self.nprt > 3:
            return
        with open("output.txt", "a") as arquivo:
            arquivo.write("Cavity  Rotor Stress     Stator Stress    Circum Vel     Cir Mach  \n")
            arquivo.write("         N/(m**2)         N/(m**2)           m/s\n")
            for i in range(0, self.nc):
                mach = self.v[i+1]/(self.gamma*self.r*self.t[i+1])**0.5
                linha_formatada = (
                    f"{i+1:<5}  {self.taur[i+1]:<16.6E} {self.taus[i+1]:<16.6E} {self.v[i+1]:<16.6E} {mach:<5.2f} \n"
                )       
                arquivo.write(linha_formatada)
            arquivo.write("\n                   Pertubation Continuity Coeficients \n")
            arquivo.write("Cavity     1        2        3        4         5         6         7 \n")
            for i in range(1, self.n_teeth):
                linha_formatada1 = (
                    f"{i+1:<5}  {self.cg[0][i]:<8.2E} {self.cg[1][i]:<8.2E} {self.cg[2][i]:<8.2E} {self.cg[3][i]:<8.2E}"
                    f" {self.cg[4][i]:<8.2E} {self.cg[5][i]:<8.2E} {self.cg[6][i]:<8.2E}\n"
                )
                arquivo.write(linha_formatada1)
            arquivo.write("\n                        Pertubation Momentun Coeficients: \n")
            arquivo.write("Cavity     1        2        3        4         5        6        7        8 \n")
            for i in range(1, self.n_teeth):
                linha_formatada2 = (
                    f"{i+1:<5}  {self.cx[0][i]:<8.2E} {self.cx[1][i]:<8.2E} {self.cx[2][i]:<8.2E} {self.cx[3][i]:<8.2E}"
                    f" {self.cx[4][i]:<8.2E} {self.cx[5][i]:<8.2E} {self.cx[6][i]:<8.2E} {self.cx[7][i]:<8.2E}\n"
                )
                arquivo.write(linha_formatada2)
        return

    def zvel(self):
        
        vgs = [0]*3
        fv  = [0]*3
        rov = [0]*3
        tr  = [0]*3
        ts  = [0]*3

        if self.omega == 0 and self.inlet_swirl_velocity == 0:
            return

        bmr = -0.25
        bms = -0.25
        bnr = 0.079
        bns = 0.079

        if self.seal_type == 'inter':
            ar = (self.tooth_height[0] + self.pitch[0])/self.pitch[0]
            as_py = (self.tooth_height[0] + self.pitch[0])/self.pitch[0]
        elif self.seal_type == 'rotor':
            ar = (2*self.tooth_height[0]+self.pitch[0])/self.pitch[0]
            as_py = 1
        else:
            as_py = (2*self.tooth_height[0]+self.pitch[0])/self.pitch[0]
            ar = 1


        #check
        dh = 2*(self.radial_clearance[0]+self.tooth_height[0])*self.pitch[0]/(self.radial_clearance[0]+self.tooth_height[0]+self.pitch[0])
        area = (self.tooth_height[0]+self.radial_clearance[0])*self.pitch[0]

        self.v[0] = self.inlet_swirl_velocity
        self.taur[0] = 0
        self.taus[0] = 0
        itmx2 = 40
        tol2 = 1*10**(-8)
        vold = 1*10**(10)

        if self.gas_composition == 'AIR':
            sb = 1.426086*10**(-6)
            ss = 100
        else:
            phi1 = (self.tz[0] ** 1.5)/ self.muz[0]
            phi2 = (self.tz[1] ** 1.5)/ self.muz[1]
            sb = (self.tz[1] - self.tz[0])/ (phi2 - phi1)
            ss = (sb*phi1) - self.tz[0]

        for i in range(1, self.n_teeth):
            mu = sb * (self.t[i])**0.5/(1 + (ss/self.t[i]))
            self.nu = mu/ self.rho[i]
            vgs[1] = (self.gamma*self.r*self.t[i])**0.5
            vgs[0] = -vgs[1]

            rov[0] = (self.shaft_radius*self.omega) - vgs[0]
            rov[1] = (self.shaft_radius*self.omega) - vgs[1]
            #check
            for j in range(0,2):
                tr[j] = 0.5 * self.rho[i]*rov[j]*rov[j]*bnr*((abs(rov[j])*dh/self.nu)**bmr)*math.copysign(1, rov[j])
                ts[j] = 0.5 * self.rho[i]*vgs[j]*vgs[j]*bns*((abs(vgs[j])*dh/self.nu)**bms)*math.copysign(1, vgs[j])
                fv[j] = (self.mdot*(vgs[j]-self.v[i-1]))-(self.pitch[0]*(tr[j]*ar-ts[j]*as_py))
            #check
            for itn2 in range(0, itmx2):
                vgs[2] = (vgs[0]*fv[1] - vgs[1]*fv[0])/(fv[1] - fv[0])
                rov[2] = (self.shaft_radius*self.omega) - vgs[2]
                tr[2] = 0.5 * self.rho[i]*rov[2]*rov[2]*bnr*((abs(rov[2])*dh/self.nu)**bmr)*math.copysign(1, rov[2])
                ts[2] = 0.5 * self.rho[i]*vgs[2]*vgs[2]*bns*((abs(vgs[2])*dh/self.nu)**bms)*math.copysign(1, vgs[2])
                fv[2] = (self.mdot*(vgs[2]-self.v[i-1]))-(self.pitch[0]*(tr[2]*ar-ts[2]*as_py))

                if fv[2]*fv[0]<0:
                    vgs[1] = vgs[2]
                    fv[1] = fv[2]
                    rov[1] = rov[2]
                    tr[1] = tr[2]
                    ts[1] = ts[2]
                    if abs((vgs[2]-vold)/vgs[2]) > tol2:
                        vold = vgs[2]
                    else:
                        break
                        
                elif fv[2]*fv[0] == 0:
                    if fv[1]== 0:
                        vgs[1] = vgs[0]
                        fv[1] = fv[0]
                        vgs[2] = vgs[0]
                        fv[2] = fv[0]
                        rov[1] = rov[0]
                        tr[1] = tr[0]
                        ts[1] = ts[0]
                        rov[2] = rov[0]
                        tr[2] = tr[0]
                        ts[2] = ts[0]
                    else:
                        vgs[1] = vgs[2]
                        fv[1] = fv[2]
                        vgs[0] = vgs[2]
                        fv[0] = fv[2]
                        rov[1] = rov[2]
                        tr[1] = tr[2]
                        ts[1] = ts[2]
                        rov[0] = rov[2]
                        tr[0] = tr[2]
                        ts[0] = ts[2]
                    break
                else:
                    vgs[0] = vgs[2]
                    fv[0] = fv[2]
                    rov[0] = rov[2]
                    tr[0] = tr[2]
                    ts[0] = ts[2]
                    if abs((vgs[2]-vold)/vgs[2]) > tol2:
                        vold = vgs[2]
                    else:
                        break
            if abs(fv[2] > 0.001):
                print('Velocity Convergence Error at station', i) 
            self.v[i] = vgs[2]
            self.taur[i] = tr[2]
            self.taus[i] = ts[2]

            self.cg[0][i] = area / (self.r*self.t[i])
            self.cg[1][i] = (self.v[i]/self.shaft_radius)*self.cg[0][i]
            self.cg[2][i] = (self.p[i]/self.shaft_radius)*self.cg[0][i]
            self.cg[3][i] = self.mdot*self.p[i]*(1/(self.p[i]**2 - self.p[i+1]**2) + 1/(self.p[i-1]**2 - self.p[i]**2))
            self.cg[4][i] = -self.mdot * self.p[i+1]/(self.p[i]**2-self.p[i+1]**2)
            self.cg[5][i] = -self.rho[i]*self.pitch[1]
            self.cg[6][i] = (self.v[i]/self.shaft_radius)*self.cg[5][i]
            self.cg[7][i] = -self.mdot*self.p[i-1]/(self.p[i-1]**2 - self.p[i]**2)
            self.cg[8][i] = -self.cg[7][i] * (self.v[i]-self.v[i-1])
            self.cx[0][i] = area/self.shaft_radius
            self.cx[1][i] = self.rho[i]*area
            self.cx[2][i] = (self.v[i]/self.shaft_radius)*self.cx[1][i]
            cxx1 = ((2+bms)*self.taus[i]*as_py*self.pitch[0])/self.v[i]
            cxx2 = ((2+bmr)*self.taur[i]*ar*self.pitch[0])/rov[2]
            self.cx[3][i] = self.mdot + cxx1 +cxx2
            self.cx[4][i] = -self.mdot
            self.cx[5][i] = 0
            self.cx[6][i] = -self.mdot*(self.v[i]-self.v[i-1])*self.p[i]/(self.p[i-1]**2 - self.p[i]**2) + ((self.taus[i]*as_py-self.taur[i]*ar)*(self.pitch[1]/self.p[i]))
            cxx3 = (-bms*self.taus[i]*as_py + bmr*self.taur[i]*ar)*(self.pitch[0]*dh/(2*(self.radial_clearance[0]+self.tooth_height[0])**2))
            self.cx[7][i] = (self.mdot/self.radial_clearance[0])*(self.v[i-1]-self.v[i]) + cxx3
        if self.nprt > 3:
            return
        with open("output.txt", "a") as arquivo:
            arquivo.write("Cavit  Rotor Stress     Stator Stress    Circum Vel     Cir Mach  \n")
            arquivo.write("         N/(m**2)         N/(m**2)           m/s\n")
            
            for i in range(0, self.nc):
                mach = self.v[i+1]/(self.gamma*self.r*self.t[i+1])**0.5
                linha_formatada = (
                    f"{i+1:<5}  {self.taur[i+1]:<16.6E} {self.taus[i+1]:<16.6E} {self.v[i+1]:<16.6E} {mach:<5.2f} \n"
                )       
                arquivo.write(linha_formatada)
            arquivo.write("\n                   Pertubation Continuity Coeficients \n")
            arquivo.write("Cavity     1        2        3        4         5         6         7 \n")
            for i in range(1, self.n_teeth):
                linha_formatada1 = (
                    f"{i+1:<5}  {self.cg[0][i]:<8.2E} {self.cg[1][i]:<8.2E} {self.cg[2][i]:<8.2E} {self.cg[3][i]:<8.2E}"
                    f" {self.cg[4][i]:<8.2E} {self.cg[5][i]:<8.2E} {self.cg[6][i]:<8.2E}\n"

                )
                arquivo.write(linha_formatada1)
            arquivo.write("\n                        Pertubation Momentun Coeficients: \n")
            arquivo.write("Cavity     1        2        3        4         5        6        7        8 \n")
            for i in range(1, self.n_teeth):
                linha_formatada2 = (
                    f"{i+1:<5}  {self.cx[0][i]:<8.2E} {self.cx[1][i]:<8.2E} {self.cx[2][i]:<8.2E} {self.cx[3][i]:<8.2E}"
                    f" {self.cx[4][i]:<8.2E} {self.cx[5][i]:<8.2E} {self.cx[6][i]:<8.2E} {self.cx[7][i]:<8.2E}\n"
                )
                arquivo.write(linha_formatada2)

            return

    def pert(self):
        gmfull = [[0 for j in range(1000)] for i in range(1000)]
        ipvt = [0]*1000
        z = [0]*1000
        rhs1 = [0]*self.nc*8
        rhs2 = [0]*self.nc*8
        val = [0]*28
        val2 = [0]*4
        val3 = [0]*4

        ir1 = [5,6,7,8]
        ic1 = [6,5,8,7]
        ir2 = [1,2,3,4]
        ic2 = [2,1,4,3]
        ir3 = [5,6,7,8]
        ic3 = [2,1,4,3]
        ir4 = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8]
        ic4 = [1,2,5,1,2,6,3,4,7,3,4,8,1,2,5,6,1,2,5,6,3,4,7,8,3,4,7,8]
        ir5 = [2,4,5,7]
        ic6 = [2,1,4,3]
        ir7 = [1,2,3,4]
        ic7 = [2,1,4,3]

        for i in range(0, self.nc):
            for ict in range(0,4):
                if i != 0:
                    irow = (i)*8 + ir1[ict] - 1
                    icol = (i-1)*8 + ic1[ict] - 1
                    jcol = icol - irow + self.nbc -1
                    self.gm[irow][jcol] = self.cx[4][i+1]

                    icol = (i-1)*8 + ic6[ict] -1
                    jcol = icol - irow + self.nbc -1
                    self.gm[irow][jcol] = self.cg[8][i+1]

                    irow = (i)*8 + ir7[ict] -1
                    icol = (i-1)*8 + ic7[ict] -1
                    jcol = icol - irow + self.nbc -1
                    self.gm[irow][jcol] = self.cg[7][i+1]
                if i != (self.nc-1):
                    irow = (i)*8 + ir2[ict] -1
                    icol = (i+1)*8 + ic2[ict] -1
                    jcol = icol - irow + self.nbc -1
                    self.gm[irow][jcol] = self.cg[4][i+1]

                    irow = (i)*8 + ir3[ict] -1
                    icol = (i+1)*8 + ic3[ict] -1
                    jcol = icol - irow + self.nbc -1
                    self.gm[irow][jcol] = self.cx[5][i+1]
            cf1 = self.omega * self.cg[0][i+1] + self.cg[1][i+1]
            cf2 = self.cg[3][i+1]
            cf3 = self.cg[2][i+1]
            cf4 = -self.omega * self.cg[0][i+1] + self.cg[1][i+1]
            cf5 = self.cx[0][i+1]
            cf6 = self.cx[6][i+1]
            cf7 = self.omega * self.cx[1][i+1] + self.cx[2][i+1]
            cf8 = self.cx[3][i+1]
            cf9 = -self.omega*self.cx[1][i+1] + self.cx[2][i+1]

            
            val[0] =  cf1
            val[1] =  cf2
            val[2] =  cf3
            val[3] =  cf2
            val[4] = -cf1
            val[5] = -cf3
            val[6] =  cf4
            val[7] =  cf2
            val[8] =  cf3
            val[9] =  cf2
            val[10] = -cf4
            val[11] = -cf3
            val[12] =  cf5
            val[13] =  cf6
            val[14] =  cf7
            val[15] =  cf8
            val[16] =  cf6
            val[17] = -cf5
            val[18] =  cf8
            val[19] = -cf7
            val[20] =  cf5
            val[21] =  cf6
            val[22] =  cf9
            val[23] =  cf8
            val[24] =  cf6
            val[25] = -cf5
            val[26] =  cf8
            val[27] = -cf9

            for ict in range(0,28):
                irow = i*8 + ir4[ict] -1
                icol = i*8 + ic4[ict] -1
                jcol = icol - irow + self.nbc - 1
                self.gm[irow][jcol] = val[ict]
            val2[0] = 0.5 * (self.omega * self.cg[5][i+1] + self.cg[6][i+1])
            val2[1] = 0.5 * (-self.omega * self.cg[5][i+1] + self.cg[6][i+1])
            val2[2] = -0.5 * self.cx[7][i+1]
            val2[3] = val2[2]
            val3[0] = -val2[0]
            val3[1] = val2[1]
            val3[2] = -val2[3]
            val3[3] = val2[2]

            for ict in range(0,4):
                irow = i*8 + ir5[ict] -1
                self.rhs[irow][0] = self.awrl / self.epslon * val2[ict]
                self.rhs[irow][1] = self.tooth_heightwrl / self.epslon * val3[ict]
        for i in range(0,8*self.nc):
            for j in range(0,33):
                if i+j - 16 > (self.nc*8)-1 or i+j-16 < -0.1:
                    cont = 1
                else:
                    gmfull[i][i+j-16] = self.gm[i][j]
        maux = [[0 for j in range(8*self.nc)] for i in range(8*self.nc)]
        for i in range(0, self.nc*8):
            for j in range(0, self.nc*8):
                maux[i][j] = gmfull[i][j]
        A = np.array(maux)
        lu, piv = lu_factor(A)
        for i in range (0,8*self.nc):
            rhs1[i] = self.rhs[i][0]
            rhs2[i] = self.rhs[i][1]
        cnd = cond(A)
        rcond = 1/cnd

        if rcond<= 1/3.0e8:
            print("Matriz quase singular \n Sem previsão para os coeficientes dinâmicos")
            quit()
        if rcond<= 1/1.0e6:
            print("Número de condição da Matriz é alto \n número de condição de matriz e:", cnd)
        sol1 = lu_solve((lu,piv), rhs1)
        sol2 = lu_solve((lu,piv), rhs2)
        for i in range (0,8*self.nc):
            self.rhs[i][0] = sol1[i]
            self.rhs[i][1] = sol2[i]

        self.kxx = 0
        self.kxy = 0
        self.cxx = 0
        self.cxy = 0

        for i in range(0,self.nc):
            icnt = (i)*8-1
            self.kxx = self.kxx + self.rhs[icnt+2][0] + self.rhs[icnt+4][0]
            self.kxy = self.kxy + self.rhs[icnt+1][1] - self.rhs[icnt+3][1]
            self.cxx = self.cxx + self.rhs[icnt+1][0] - self.rhs[icnt+3][0]
            self.cxy = self.cxy + self.rhs[icnt+2][1] + self.rhs[icnt+4][1]
        
        self.kxx = self.pi * self.shaft_radius * self.pitch[1] * (self.epslon/self.awrl) * self.kxx
        self.kxy = self.pi * self.shaft_radius * self.pitch[1] * (self.epslon/self.tooth_heightwrl) * self.kxy
        self.kyx = -self.kxy
        if self.omega != 0:
            self.cxx = -self.pi * self.shaft_radius * self.pitch[1] * (self.epslon/self.awrl) / self.omega * self.cxx
            self.cxy = self.pi * self.shaft_radius * self.pitch[1] * (self.epslon/self.tooth_heightwrl) / self.omega * self.cxy
            self.cyx = -self.cxy
        else:
            self.cxx = 0
            self.cxy = 0
            self.cyx = 0

        with open("output.txt", "a") as arquivo:
            arquivo.write("\n Computed results \n \n")
            arquivo.write("   Stiffness Coefficients \n \n")
            
            arquivo.write(f"{'kxx':<10} {self.kxx:>15.4E} N/m \n")
            arquivo.write(f"{'kyx':<10} {self.kyx:>15.4E} N/m \n")
            arquivo.write(f"{'kxy':<10} {self.kxy:>15.4E} N/m \n")
            arquivo.write(f"{'kxx':<10} {self.kxx:>15.4E} N/m \n")

            arquivo.write("\n Damping Coefficients \n \n")
            arquivo.write(f"{'cxx':<10} {self.cxx:>15.4E} N/m \n")
            arquivo.write(f"{'cyx':<10} {self.cyx:>15.4E} N/m \n")
            arquivo.write(f"{'cxy':<10} {self.cxy:>15.4E} N/m \n")
            arquivo.write(f"{'cyy':<10} {self.cxx:>15.4E} N/m \n")

    def run(self, frequency):
        self.frequency = frequency
        self.inlet_swirl_velocity=self.pre_swirl_ratio*self.frequency*self.shaft_radius
        self.setup()
        self.Vermes()
        self.ZPRES()
        if self.iopt1 == 0:
            self.zvel()
        elif self.iopt1 == 1:
            self.zvel_JEN()
        if self.analz != "LEAKAGE":
            self.pert()
        keys = ["kxx","kyy","kxy","kyx","cxx","cyy","cxy","cyx","leakage"] 
        coefficients_dict = {}              
        coefficients_dict["kxx"] = self.kxx
        coefficients_dict["kyy"] = self.kxx
        coefficients_dict["kxy"] = self.kxy
        coefficients_dict["kyx"] = self.kyx
        coefficients_dict["cxx"] = self.cxx
        coefficients_dict["cyy"] = self.cxx
        coefficients_dict["cxy"] = self.cxy
        coefficients_dict["cyx"] = self.cyx               
        coefficients_dict["leakage"] = self.mdot
        
        return coefficients_dict