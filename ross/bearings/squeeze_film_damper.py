import numpy as np
import math
from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units

class SqueezeFilmDamper(BearingElement):
    """
    Squeeze Film Damper (SFD) element in ROSS standard format.
    Computes damping (CO), stiffness (KO), maximum pressure (P_max)
    and pressure angle (ThetaM) based on classical short-bearing theory.

    Parameters
    ----------
    Bearing Geometry
    ^^^^^^^^^^^^^^^^
    Describes the geometric characteristics.
    n : int
        Node in which the bearing will be located.
    frequency : list, pint.Quantity
        Array with the frequencies (rad/s).
    axial_length : float, pint.Quantity
        Bearing length. Default unit is meter.
    journal_radius : float
        Rotor radius. The unit is meter.
    radial_clearance : float
        Radial clearence between rotor and bearing. The unit is meter.
    eccentricity_ratio : float
        Normal cases, utilize the eccentricity radio < 0.4, because this 
        gap has better propieties. It's a dimensioneless parametre.
    Mu : float
        Lubricants viscosity. The unit is N*s/m².
    Groove : boolean
        It can be true or false, depends 
    End_Seals : boolean
        It can be true or false, depending on configuration of the squeeze film damper.
    Cav : boolean
        It can be true or false, depending on configuration of the flow type.
    
    Returns
    -------

    CO = list
        Damping coefficients. The unit is N*s/m.
    KO = list
        Stiffness coefficients. The unit is N/m. 
    P_max = list
        Maximum pressure. The unit is MPa.
    Theta = list
        Pressure angle. The unit is radian.
    
    """

    @check_units
    def __init__(
            self,
            n,
            frequency, 
            axial_length,
            journal_radius, 
            radial_clearance,
            eccentricity_ratio,
            MU,
            Groove = True, 
            End_Seals = True,
            Cav= True, 
            tag=None, 
            scale_factor=1.0
            ):  
        
        self.axial_length = axial_length
        self.journal_radius = journal_radius
        self.radial_clearance = radial_clearance
        self.eccentricity_ratio = eccentricity_ratio
        self.frequency = frequency
        self.MU = MU
        self.Groove = Groove
        self.End_Seals = End_Seals
        self.Cav = Cav

        if (not Groove) and End_Seals:
            CO, KO, Theta, P_max = self.calculate_coeficients_with_End_Seals()
        elif Groove and (not End_Seals):
            CO, KO, Theta, P_max = self.calculate_coeficients_with_Groove()
        elif Groove and End_Seals:
            CO, KO, Theta, P_max = self.calculate_coeficientes_with_Groove_and_endseals()

         
    
    
        super().__init__(
            n=n,
            frequency=[frequency],
            kxx=[KO],  
            cxx=[CO],
            tag=tag,
            scale_factor=scale_factor,
        )
    
    def calculate_coeficients_with_End_Seals(self):
        CO = 12.0 * np.pi * self.axial_length * (self.journal_radius / self.radial_clearance)**3 * self.MU
        CO /= ((2.0 + self.eccentricity_ratio**2) * np.sqrt(1.0 - self.eccentricity_ratio**2))

        KO = 24.0 * self.MU * self.axial_length * (self.journal_radius / self.radial_clearance)**3 * self.eccentricity_ratio * self.frequency
        KO /= ((2.0 + self.eccentricity_ratio**2) * (1.0 - self.eccentricity_ratio**2))

        ThetaM = -80.45 * self.eccentricity_ratio + 268.98
        Theta = math.radians(ThetaM)

        P_max_NUM = 2.0 * self.eccentricity_ratio * (2.0 + self.eccentricity_ratio * np.cos(Theta)) * np.sin(Theta)
        P_max_DEN = (2.0 + self.eccentricity_ratio**2) * (1.0 + self.eccentricity_ratio * np.cos(Theta))**2
        P_max = -P_max_NUM / P_max_DEN * 6.0 * self.MU * self.frequency * (self.journal_radius / self.radial_clearance)**2

        if self.Cav:
            KO = 0.0
        else:
            CO = 2.0 * CO
            KO = 0.0

        return CO, KO, Theta, P_max



    def calculate_coeficients_with_Groove(self):
        if self.Cav:
            CO = self.MU * (self.axial_length**3) * self.journal_radius / (2.0 * self.radial_clearance**3)
            CO *= np.pi / ((1.0 - self.eccentricity_ratio**2)**(3.0/2.0))
            CO /= 4

            KO = 2.0 * self.MU * self.frequency * self.journal_radius * (self.axial_length / self.radial_clearance)**3 * self.eccentricity_ratio
            KO /= (1.0 - self.eccentricity_ratio**2)**2
            KO /=4

        ThetaM = (270.443
                  - 191.831*self.eccentricity_ratio
                  + 218.223*self.eccentricity_ratio**2
                  - 114.803*self.eccentricity_ratio**3)
        Theta = math.radians(ThetaM)

        P_max = -1.5 * (self.axial_length / self.radial_clearance)**2 * self.MU * self.frequency * self.eccentricity_ratio * np.sin(Theta)
        P_max /= (1.0 + self.eccentricity_ratio*np.cos(Theta))**3
        P_max /= 2

        if not self.Cav:
            KO = 0.0
            CO = self.MU * (self.axial_length / self.radial_clearance)**3 * self.journal_radius * np.pi
            CO /= ((1.0 - self.eccentricity_ratio**2)**(3.0/2.0))
    
        return CO, KO, Theta, P_max

        

    def calculate_coeficientes_with_Groove_and_endseals(self):
        if self.Cav:
            CO = self.MU * (self.axial_length**3) * self.journal_radius / (2.0 * self.radial_clearance**3)
            CO *= np.pi / ((1.0 - self.eccentricity_ratio**2)**(3.0/2.0))

            KO = 2.0 * self.MU * self.frequency * self.journal_radius * (self.axial_length / self.radial_clearance)**3 * self.eccentricity_ratio
            KO /= (1.0 - self.eccentricity_ratio**2)**2

        ThetaM = (270.443
                  - 191.831*self.eccentricity_ratio
                  + 218.223*self.eccentricity_ratio**2
                  - 114.803*self.eccentricity_ratio**3)
        Theta = math.radians(ThetaM)

        P_max = -1.5 * (self.axial_length / self.radial_clearance)**2 * self.MU * self.frequency * self.eccentricity_ratio * np.sin(Theta)
        P_max /= (1.0 + self.eccentricity_ratio*np.cos(Theta))**3

        if not self.Cav:
            KO = 0.0
            CO = self.MU * (self.axial_length / self.radial_clearance)**3 * self.journal_radius * np.pi
            CO /= ((1.0 - self.eccentricity_ratio**2)**(3.0/2.0))
    
        return CO, KO, Theta, P_max
