import numpy as np
import math
from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units

class SqueezeFilmDamper(BearingElement):
    """
    Squeeze Film Damper (SFD) element in ROSS standard format.
    Computes damping (CO), stiffness (KO), maximum pressure (PMAX)
    and pressure angle (THETAM) based on classical short-bearing theory.
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
            GROOVE = True, 
            END_SEALS = True,
            CAV= True, 
            tag=None, 
            scale_factor=1.0
            ):  
        
        self.axial_length = axial_length
        self.journal_radius = journal_radius
        self.radial_clearance = radial_clearance
        self.eccentricity_ratio = eccentricity_ratio
        self.frequency = frequency
        self.MU = MU
        self.GROOVE = GROOVE
        self.END_SEALS = END_SEALS
        self.CAV = CAV

        if (not GROOVE) and END_SEALS:
            CO, KO, THETA, PMAX = self.calculate_coeficients_with_end_seals()
        elif GROOVE and (not END_SEALS):
            CO, KO, THETA, PMAX = self.calculate_coeficients_with_groove()
        elif GROOVE and END_SEALS:
            CO, KO, THETA, PMAX = self.calculate_coeficientes_with_groove_and_endseals()

         
    
    
        super().__init__(
            n=n,
            frequency=[frequency],
            kxx=[KO],  
            cxx=[CO],
            tag=tag,
            scale_factor=scale_factor,
        )
    
    def calculate_coeficients_with_end_seals(self):
        CO = 12.0 * np.pi * self.axial_length * (self.journal_radius / self.radial_clearance)**3 * self.MU
        CO /= ((2.0 + self.eccentricity_ratio**2) * np.sqrt(1.0 - self.eccentricity_ratio**2))

        KO = 24.0 * self.MU * self.axial_length * (self.journal_radius / self.radial_clearance)**3 * self.eccentricity_ratio * self.frequency
        KO /= ((2.0 + self.eccentricity_ratio**2) * (1.0 - self.eccentricity_ratio**2))

        THETAM = -80.45 * self.eccentricity_ratio + 268.98
        THETA = math.radians(THETAM)

        PMAX_NUM = 2.0 * self.eccentricity_ratio * (2.0 + self.eccentricity_ratio * np.cos(THETA)) * np.sin(THETA)
        PMAX_DEN = (2.0 + self.eccentricity_ratio**2) * (1.0 + self.eccentricity_ratio * np.cos(THETA))**2
        PMAX = -PMAX_NUM / PMAX_DEN * 6.0 * self.MU * self.frequency * (self.journal_radius / self.radial_clearance)**2

        if self.CAV:
            KO = 0.0
        else:
            CO = 2.0 * CO
            KO = 0.0

        return CO, KO, THETA, PMAX



    def calculate_coeficients_with_groove(self):
        if self.CAV:
            CO = self.MU * (self.axial_length**3) * self.journal_radius / (2.0 * self.radial_clearance**3)
            CO *= np.pi / ((1.0 - self.eccentricity_ratio**2)**(3.0/2.0))
            CO /= 4

            KO = 2.0 * self.MU * self.frequency * self.journal_radius * (self.axial_length / self.radial_clearance)**3 * self.eccentricity_ratio
            KO /= (1.0 - self.eccentricity_ratio**2)**2
            KO /=4

        THETAM = (270.443
                  - 191.831*self.eccentricity_ratio
                  + 218.223*self.eccentricity_ratio**2
                  - 114.803*self.eccentricity_ratio**3)
        THETA = math.radians(THETAM)

        PMAX = -1.5 * (self.axial_length / self.radial_clearance)**2 * self.MU * self.frequency * self.eccentricity_ratio * np.sin(THETA)
        PMAX /= (1.0 + self.eccentricity_ratio*np.cos(THETA))**3
        PMAX /= 2

        if not self.CAV:
            KO = 0.0
            CO = self.MU * (self.axial_length / self.radial_clearance)**3 * self.journal_radius * np.pi
            CO /= ((1.0 - self.eccentricity_ratio**2)**(3.0/2.0))
    
        return CO, KO, THETA, PMAX

        

    def calculate_coeficientes_with_groove_and_endseals(self):
        if self.CAV:
            CO = self.MU * (self.axial_length**3) * self.journal_radius / (2.0 * self.radial_clearance**3)
            CO *= np.pi / ((1.0 - self.eccentricity_ratio**2)**(3.0/2.0))

            KO = 2.0 * self.MU * self.frequency * self.journal_radius * (self.axial_length / self.radial_clearance)**3 * self.eccentricity_ratio
            KO /= (1.0 - self.eccentricity_ratio**2)**2

        THETAM = (270.443
                  - 191.831*self.eccentricity_ratio
                  + 218.223*self.eccentricity_ratio**2
                  - 114.803*self.eccentricity_ratio**3)
        THETA = math.radians(THETAM)

        PMAX = -1.5 * (self.axial_length / self.radial_clearance)**2 * self.MU * self.frequency * self.eccentricity_ratio * np.sin(THETA)
        PMAX /= (1.0 + self.eccentricity_ratio*np.cos(THETA))**3

        if not self.CAV:
            KO = 0.0
            CO = self.MU * (self.axial_length / self.radial_clearance)**3 * self.journal_radius * np.pi
            CO /= ((1.0 - self.eccentricity_ratio**2)**(3.0/2.0))
    
        return CO, KO, THETA, PMAX
