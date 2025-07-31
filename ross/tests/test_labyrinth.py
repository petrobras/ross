import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_almost_equal

from ross.seals.labyrinth_seal import LabyrinthSeal
from ross.units import Q_


@pytest.fixture
def labyrinth():

    seal = LabyrinthSeal(
        n=0,
        inlet_pressure=308000,
        outlet_pressure=94300,
        inlet_temperature=283.15,
        pre_swirl_ratio=0.98,
        frequency=Q_([8000], "RPM"),
        n_teeth=16,
        shaft_radius=Q_(72.5,"mm"),
        radial_clearance=Q_(0.3,"mm"),
        pitch=Q_(3.175,"mm"),
        tooth_height=Q_(3.175,"mm"),
        tooth_width=Q_(0.1524,"mm"),
        seal_type="inter",
        gas_composition = {"AIR": 1.00},

    )

    return seal

def test_labyrinth_coefficients(labyrinth):

    assert_allclose(labyrinth.kxx, -50448.19591, rtol=1e-4)
    assert_allclose(labyrinth.kxy, 35536.953463, rtol=1e-4)
    assert_allclose(labyrinth.kyx, -35536.953463, rtol=1e-4)
    assert_allclose(labyrinth.kyy, -50448.19591, rtol=1e-4)
    assert_allclose(labyrinth.cxx, 23.821095, rtol=1e-4)
    assert_allclose(labyrinth.cxy, 56.246427, rtol=1e-4)
    assert_allclose(labyrinth.cyx, -56.246427, rtol=1e-4)
    assert_allclose(labyrinth.cyy, 23.821095, rtol=1e-4)
