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
    )

    return seal

def test_labyrinth_coefficients(labyrinth):
    assert_allclose(labyrinth.kxx, -50242.35216407, rtol=1e-4)
    assert_allclose(labyrinth.kxy, 35764.19464202, rtol=1e-4)
    assert_allclose(labyrinth.kyx, -35764.19464202, rtol=1e-4)
    assert_allclose(labyrinth.kyy, -50242.35216407, rtol=1e-4)
    assert_allclose(labyrinth.cxx, 24.03501182, rtol=1e-4)
    assert_allclose(labyrinth.cxy, 55.95971508, rtol=1e-4)
    assert_allclose(labyrinth.cyx, -55.95971508, rtol=1e-4)
    assert_allclose(labyrinth.cyy, 24.03501182, rtol=1e-4)