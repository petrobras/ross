"""Solver constants: type-flag vocabularies and convergence tolerances.

Every type flag is a string (``bearing_type == "fixed_geometry"``), and this
module is the source of truth for the legal values of each one. The wrapper
validates user input against them and the solver dispatches on them directly.
"""

import numpy as np

PI = np.pi
VIS_REF = 2.0e-6

# Tolerance, in m, of the tests that decide whether an energy-mesh node lies
# on the film/pad interface. The first film node sits h / total_ey_film above
# the pad surface, which is below 1 um for thin films on fine cross-film
# meshes, so the tolerance must stay far below that height (a tolerance of
# 1e-6 m took the first film layer for solid) and above the round-off of the
# node coordinates.
INTERFACE_TOL = 1.0e-8

# --------------------------------------------------------------------------- #
# Type-flag vocabularies                                                      #
# --------------------------------------------------------------------------- #
# The legal string values for each flag. Numba kernels compare against these
# strings directly; the user-facing bearing classes validate their inputs
# against these tuples before anything reaches the solver.

BEARING_TYPES = (
    "fixed_geometry",
    "conventional_tilting_pad",
    "inlet_groove_tilting_pad",
    "spray_bar_tilting_pad",
    "pressure_dam",
)
TILTING_PAD_TYPES = (
    "conventional_tilting_pad",
    "inlet_groove_tilting_pad",
    "spray_bar_tilting_pad",
)

OPERATING_TYPES = (
    "regular_flooded",
    "axial_flow",
    "high_ambient_pressure",
    "starved_condition_even",
    "starved_condition_uneven",
    "oil_ring_lubricated",
)

# ``None`` is the "off" sentinel: isoviscous for thermal_type, rigid for
# deform_type.
THERMAL_TYPES = (None, "adiabatic", "full")

TEMP_J_TYPES = (
    "averaged_film_temperature",
    "no_heat_flux_into_journal",
    "insulated_shaft_surface",
)

DEFORM_TYPES = (
    None,
    "pad_mechanical",
    "pad_mechanical_thermal",
    "pad_mechanical_thermal_shaft_shell_thermal",
    "pad_mechanical_thermal_shaft_shell_thermal_pivot_mechanical",
    "pad_pivot_mechanical",
)
# Deformation modes that include pivot flexibility: these need ``k_pivot``
# from ``deform_pivots`` and use the ``dynamic_reduction_pivot`` reduction.
PIVOT_FLEX_DEFORM_TYPES = (
    "pad_mechanical_thermal_shaft_shell_thermal_pivot_mechanical",
    "pad_pivot_mechanical",
)

EQUILIBRIUM_TYPES = ("match_load", "match_eccentricity")

SUMP_TYPES = ("supply_temperature", "sump_temperature")

PIVOT_TYPES = (
    "ball_in_socket",
    "button",
    "rocker_back",
    "user_specified_stiffness",
)


# Convergence criteria for the iterative solution: shear stress in Pa,
# temperatures in K.
MAX_ITERATION = 100
SHEAR_ERROR = 70.0
TEMP_ERROR = 5.0e-3
JTEMP_ERROR = 5.0e-2
TEMP_INLET_ERROR = 0.25
SUMP_TEMP_ERROR = 0.5
DEFORM_ERROR = 1.0e-3
