import numpy as np
import math
from ross.bearing_seal_element import BearingElement

class SqueezeFilmDamperElement(BearingElement):
    """
    Squeeze Film Damper (SFD) element in ROSS standard format.
    Computes damping (CO), stiffness (KO), maximum pressure (PMAX)
    and pressure angle (THETAM) based on classical short-bearing theory.
    """

def __init__(
            self,
            frequency, 
            axial_lenght,
            journal_radius, 
            radial_clearance,
            eccentricity_ratio,
            MU,
            GROOVE = True, 
            END_SEALS = True,
            CAV= True, 
            tag=None, 
            scale_factor=1.0):  
    





    super().__init__(
            frequency=frequency,
            kxx=None,  
            cxx=None,
            tag=tag,
            scale_factor=scale_factor,
        )
    
    self.axial_lenght = axial_lenght
    self.journal_radius = journal_radius
    self.radial_clearance = radial_clearance
    self.eccentricity_ratio = eccentricity_ratio
    self.frequency = frequency
    self.MU = MU
    self.GROOVE = GROOVE
    self.END_SEALS = END_SEALS
    self.CAV = CAV

