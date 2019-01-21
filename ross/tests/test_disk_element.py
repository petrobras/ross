import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from ross.disk_element import DiskElement
from ross.materials import steel


@pytest.fixture
def disk():
    return DiskElement.from_geometry(0, steel, 0.07, 0.05, 0.28)


def test_mass_matrix_disk(disk):
    Md1 = np.array([[ 32.58973,   0.     ,   0.     ,   0.     ],
                    [  0.     ,  32.58973,   0.     ,   0.     ],
                    [  0.     ,   0.     ,   0.17809,   0.     ],
                    [  0.     ,   0.     ,   0.     ,   0.17809]])
    assert_almost_equal(disk.M(), Md1, decimal=5)


def test_gyroscopic_matrix_disk(disk):
    Gd1 = np.array([[ 0.     ,  0.     ,  0.     ,  0.     ],
                    [ 0.     ,  0.     ,  0.     ,  0.     ],
                    [ 0.     ,  0.     ,  0.     ,  0.32956],
                    [ 0.     ,  0.     , -0.32956,  0.     ]])
    assert_almost_equal(disk.G(), Gd1, decimal=5)


def test_errors():
    with pytest.raises(TypeError) as ex:
        DiskElement(1.0, steel, 0.07, 0.05, 0.28)
    assert 'n should be int, not float' == str(ex.value)
