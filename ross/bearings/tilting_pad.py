"""Tilting-pad hydrodynamic bearing properties calculation module

In this module, the tilting-pad bearing properties can be estimated and
simulated for a given set of properties, until the model reaches a steady-
-state of numerical convergence. These properties can then be exported to
external files, and used in the bearing_seal_element.py routine inside the
ROSS framework. This routine can also be used for bearing properties esti-
mation for other purposes.
"""

import os
import warnings

import numpy as np
import toml
from plotly import graph_objects as go
from scipy import interpolate as interpolate

from ross.element import Element
from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow.fluid_flow_coefficients import (
    calculate_short_damping_matrix, calculate_short_stiffness_matrix)
from ross.units import Q_, check_units
from ross.utils import read_table_file

# __all__ = [
# ]

# FOR THE MOMENT, THIS CODE IS BEING TRANSLATED FROM MATLAB TO PYTHON, AND
# THEREFORE THIS PROCESS WILL BE MADE IN LINEAR PROGRAMMING STANDARDS. ONCE
# THIS INITIAL TRANSLATION IS DONE, AND THE CODE VALIDATED, THE MODULARIZATION 
# AND CONVERSION TO OBJECT ORIENTED STANDARDS WILL BE MADE.







