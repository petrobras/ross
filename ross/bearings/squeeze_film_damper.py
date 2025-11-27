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
            n, 
            L,
            R, 
            C,
            MU,
            TYP=0, 
            CAV=1, 
            tag=None, 
            scale_factor=1.0):
    super().__init__(
            n=n,
            kxx=None,  
            cxx=None,
            tag=tag,
            scale_factor=scale_factor,
        )
    self.L = L
    self.R = R
    self.C = C
    self.MU = MU
    self.TYP = TYP
    self.CAV = CAV