import os
from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from ross.disk_element import DiskElement, DiskElement6DoF
from ross.materials import steel


@pytest.fixture
def disk():
    return DiskElement(0, 0.07, 0.05, 0.32956)


def test_index(disk):
    assert disk.dof_local_index().x_0 == 0
    assert disk.dof_local_index().y_0 == 1
    assert disk.dof_local_index().alpha_0 == 2
    assert disk.dof_local_index().beta_0 == 3


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


def test_save_load(disk, disk_from_geometry):
    file = Path(tempdir) / "disk.toml"
    disk.save(file)
    disk_loaded = DiskElement.load(file)
    assert disk == disk_loaded


def test_mass_matrix_disk1(disk_from_geometry):
    # fmt: off
    Md1 = np.array([[ 32.58973,   0.     ,   0.     ,   0.     ],
                    [  0.     ,  32.58973,   0.     ,   0.     ],
                    [  0.     ,   0.     ,   0.17809,   0.     ],
                    [  0.     ,   0.     ,   0.     ,   0.17809]])
    # fmt: on
    assert_almost_equal(disk_from_geometry.M(), Md1, decimal=5)


def test_gyroscopic_matrix_disk1(disk_from_geometry):
    # fmt: off
    Gd1 = np.array([[ 0.     ,  0.     ,  0.     ,  0.     ],
                    [ 0.     ,  0.     ,  0.     ,  0.     ],
                    [ 0.     ,  0.     ,  0.     ,  0.32956],
                    [ 0.     ,  0.     , -0.32956,  0.     ]])
    # fmt: on
    assert_almost_equal(disk_from_geometry.G(), Gd1, decimal=5)


def test_from_table():
    for file_name in ["/data/shaft_us.xls", "/data/shaft_si.xls"]:
        file_name = os.path.dirname(os.path.realpath(__file__)) + file_name
        disks = DiskElement.from_table(file_name, sheet_name="More")

        assert_allclose(disks[1].m, 6.90999178227835)
        assert_allclose(disks[1].Ip, 0.0469996988106328, atol=1.6e-07)
        assert_allclose(disks[1].Id, 0.0249998397928898, atol=1.6e-07)


@pytest.fixture
def disk_6dof():
    return DiskElement6DoF(0, 32.58973, 0.17809, 0.32956)


def test_index_6dof(disk_6dof):
    assert disk_6dof.dof_local_index().x_0 == 0
    assert disk_6dof.dof_local_index().y_0 == 1
    assert disk_6dof.dof_local_index().z_0 == 2
    assert disk_6dof.dof_local_index().alpha_0 == 3
    assert disk_6dof.dof_local_index().beta_0 == 4
    assert disk_6dof.dof_local_index().theta_0 == 5


def test_mass_matrix_disk_6dof(disk_6dof):
    # fmt: off
    Md1 = np.array([[32.58973, 0.,       0.,       0.,      0.,      0.   ],
                    [0.,       32.58973, 0.,       0.,      0.,      0.   ],
                    [0.,       0.,       32.58973, 0.,      0.,      0.   ],
                    [0.,       0.,       0.,       0.17809, 0.,      0.   ],
                    [0.,       0.,       0.,       0.,      0.17809, 0.   ],
                    [0.,       0.,       0.,       0,       0.,      0.32956]])
    # fmt: on
    assert_almost_equal(disk_6dof.M(), Md1, decimal=5)


def test_gyroscopic_matrix_disk_6dof(disk_6dof):
    # fmt: off
    Gd1 = np.array([
            [ 0, 0, 0,  0,         0,       0],
            [ 0, 0, 0,  0,         0,       0],
            [ 0, 0, 0,  0,         0,       0],
            [ 0, 0, 0,  0,         0.32956, 0],
            [ 0, 0, 0, -0.32956,   0,       0],
            [ 0, 0, 0,  0,         0,       0]])
    # fmt: on

    assert_almost_equal(disk_6dof.G(), Gd1, decimal=5)


@pytest.fixture
def disk_from_geometry_6dof():
    return DiskElement6DoF.from_geometry(0, steel, 0.07, 0.05, 0.28)


def test_mass_matrix_disk1_6dof(disk_from_geometry_6dof):
    # fmt: off
    Md1 = np.array([[32.58973, 0.,       0.,       0.,      0.,      0.   ],
                    [0.,       32.58973, 0.,       0.,      0.,      0.   ],
                    [0.,       0.,       32.58973, 0.,      0.,      0.   ],
                    [0.,       0.,       0.,       0.17809, 0.,      0.   ],
                    [0.,       0.,       0.,       0.,      0.17809, 0.   ],
                    [0.,       0.,       0.,       0,       0.,      0.32956]])
    # fmt: on

    M_class = disk_from_geometry_6dof.M()
    assert_almost_equal(M_class, Md1, decimal=5)


def test_gyroscopic_matrix_disk1_6dof(disk_from_geometry_6dof):
    # fmt: off
    Gd1 = np.array([
            [ 0, 0, 0,  0,         0,       0],
            [ 0, 0, 0,  0,         0,       0],
            [ 0, 0, 0,  0,         0,       0],
            [ 0, 0, 0,  0,         0.32956, 0],
            [ 0, 0, 0, -0.32956,   0,       0],
            [ 0, 0, 0,  0,         0,       0]])
    # fmt: on
    assert_almost_equal(disk_from_geometry_6dof.G(), Gd1, decimal=5)


def test_from_table_6dof():
    for file_name in ["/data/shaft_us.xls", "/data/shaft_si.xls"]:
        file_name = os.path.dirname(os.path.realpath(__file__)) + file_name
        disks = DiskElement6DoF.from_table(file_name, sheet_name="More")

        assert_allclose(disks[1].m, 6.90999178227835)
        assert_allclose(disks[1].Ip, 0.0469996988106328, atol=1.6e-07)
        assert_allclose(disks[1].Id, 0.0249998397928898, atol=1.6e-07)
