import numpy as np
import pytest
from copy import deepcopy
from numpy.testing import assert_almost_equal

from ross.units import Q_
from ross.materials import Material, steel
from ross.gear_element import GearElement


@pytest.fixture
def gear():
    return GearElement(
        n=4,
        m=726.4,
        Id=56.95,
        Ip=113.9,
        width=0.101,
        n_teeth=328,
        pitch_diameter=1.1,
        pressure_angle=Q_(22.5, "deg"),
    )


def test_gear_params(gear):
    assert gear.pressure_angle == 0.39269908169872414
    assert gear.base_radius == 0.5081337428812077
    assert gear.helix_angle == 0.0
    assert gear.material == steel


def test_mass_matrix_gear(gear):
    # fmt: off
    Mg = np.array([[726.4,      0.,      0.,      0.,      0.,      0.],
                   [     0., 726.4,      0.,      0.,      0.,      0.],
                   [     0.,      0., 726.4,      0.,      0.,      0.],
                   [     0.,      0.,      0., 56.95,      0.,      0.],
                   [     0.,      0.,      0.,      0., 56.95,      0.],
                   [     0.,      0.,      0.,      0.,      0., 113.9]])
    # fmt: on

    assert_almost_equal(gear.M(), Mg, decimal=5)


def test_gyroscopic_matrix_gear(gear):
    # fmt: off
    Gg = np.array([[0., 0., 0.,     0.,    0., 0.],
                   [0., 0., 0.,     0.,    0., 0.],
                   [0., 0., 0.,     0.,    0., 0.],
                   [0., 0., 0.,     0., 113.9, 0.],
                   [0., 0., 0., -113.9,    0., 0.],
                   [0., 0., 0.,     0.,    0., 0.]])
    # fmt: on

    assert_almost_equal(gear.G(), Gg, decimal=5)

def test_gyroscopic_matrix_gear(gear):
    # fmt: off
    Gg = np.array([[0., 0., 0.,     0.,    0., 0.],
                   [0., 0., 0.,     0.,    0., 0.],
                   [0., 0., 0.,     0.,    0., 0.],
                   [0., 0., 0.,     0., 113.9, 0.],
                   [0., 0., 0., -113.9,    0., 0.],
                   [0., 0., 0.,     0.,    0., 0.]])
    # fmt: on

    assert_almost_equal(gear.G(), Gg, decimal=5)


