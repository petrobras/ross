import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ross.disk_element import DiskElement
from ross.materials import steel


@pytest.fixture
def disk():
    return DiskElement(0, 0.07, 0.05, 0.32956)


def test_mass_matrix_disk(disk):
    # fmt: off
    Md1 = np.array([[0.07, 0., 0., 0.],
                    [0., 0.07, 0., 0.],
                    [0., 0., 0.05, 0.],
                    [0., 0., 0., 0.05]])
    # fmt: on

    assert_almost_equal(disk.M(), Md1, decimal=5)


def test_gyroscopic_matrix_disk(disk):
    # fmt: off
    Gd1 = np.array([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.32956],
                    [0., 0., -0.32956, 0.]])
    # fmt: on

    assert_almost_equal(disk.G(), Gd1, decimal=5)


@pytest.fixture
def disk_from_geometry():
    return DiskElement.from_geometry(0, steel, 0.07, 0.05, 0.28)


def test_mass_matrix_disk1(disk_from_geometry):
    Md1 = np.array(
        [
            [32.58973, 0.0, 0.0, 0.0],
            [0.0, 32.58973, 0.0, 0.0],
            [0.0, 0.0, 0.17809, 0.0],
            [0.0, 0.0, 0.0, 0.17809],
        ]
    )
    assert_almost_equal(disk_from_geometry.M(), Md1, decimal=5)


def test_gyroscopic_matrix_disk1(disk_from_geometry):
    Gd1 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.32956],
            [0.0, 0.0, -0.32956, 0.0],
        ]
    )
    assert_almost_equal(disk_from_geometry.G(), Gd1, decimal=5)
