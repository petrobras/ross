import os
import pickle
from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearing_seal_element import (
    BallBearingElement,
    BearingElement,
    BearingElement6DoF,
    BearingFluidFlow,
    CylindricalBearing,
    MagneticBearingElement,
    RollerBearingElement,
)

from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow.fluid_flow import fluid_flow_example2
from ross.fluid_flow.fluid_flow_coefficients import (
    calculate_oil_film_force,
    calculate_short_damping_matrix,
    calculate_short_stiffness_matrix,
    calculate_stiffness_and_damping_coefficients,
    find_equilibrium_position,
)
from ross.units import Q_

# def fluid_flow_short_numerical():
#     nz = 8
#     ntheta = 32
#     length = 0.01
#     omega = 100.0 * 2 * np.pi / 60
#     p_in = 0.0
#     p_out = 0.0
#     radius_rotor = 0.08
#     radius_stator = 0.1
#     visc = 0.015
#     rho = 860.0
#     eccentricity = 0.001
#     return flow.FluidFlow(
#         nz,
#         ntheta,
#         length,
#         omega,
#         p_in,
#         p_out,
#         radius_rotor,
#         radius_stator,
#         visc,
#         rho,
#         eccentricity=eccentricity,
#         immediately_calculate_pressure_matrix_numerically=False,
#     )

# # def test_oil_film_force_short():
# bearing = fluid_flow_short_numerical()
# bearing.calculate_pressure_matrix_numerical()
# n, t, force_x, force_y = calculate_oil_film_force(bearing)
# (
#     n_numerical,
#     t_numerical,
#     force_x_numerical,
#     force_y_numerical,
# ) = calculate_oil_film_force(bearing, force_type="numerical")
# assert_allclose(n, n_numerical, rtol=0.5)
# assert_allclose(t, t_numerical, rtol=0.25)
# assert_allclose(force_x_numerical, 0, atol=1e-07)
# assert_allclose(force_y_numerical, bearing.load, atol=1e-05)

# test for BallBearingElement
nz = 30
ntheta = 20
length = 0.03
omega = [157.1]
p_in = 0.0
p_out = 0.0
radius_rotor = 0.0499
radius_stator = 0.05
load = 525
visc = 0.1
rho = 860.0
bearing = BearingFluidFlow(
    0,
    nz,
    ntheta,
    length,
    omega,
    p_in,
    p_out,
    radius_rotor,
    radius_stator,
    visc,
    rho,
    load=load,
)

# fmt: off
K = np.array([[14547442.70620538, 15571505.36655864],
                [-25596382.88167763, 12526684.40342712]])

C = np.array([[ 263025.76330117, -128749.90335233],
                [ -41535.76386708,  309417.62615761]])
# fmt: on

assert_allclose(bearing.K(0), K, rtol=1e-1)
assert_allclose(bearing.C(0), C, rtol=1e-3)