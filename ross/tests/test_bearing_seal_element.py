import numpy as np
import pytest
import os
from numpy.testing import assert_allclose
import math

from ross.bearing_seal_element import BearingElement


@pytest.fixture
def bearing0():
    Kxx_bearing = np.array(
        [8.5e07, 1.1e08, 1.3e08, 1.6e08, 1.8e08, 2.0e08, 2.3e08, 2.5e08, 2.6e08]
    )
    Kyy_bearing = np.array(
        [9.2e07, 1.1e08, 1.4e08, 1.6e08, 1.9e08, 2.1e08, 2.3e08, 2.5e08, 2.6e08]
    )
    Cxx_bearing = np.array(
        [226837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    Cyy_bearing = np.array(
        [235837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    wb = np.array([314.2, 418.9, 523.6, 628.3, 733.0, 837.8, 942.5, 1047.2, 1151.9])
    bearing0 = BearingElement(
        4, kxx=Kxx_bearing, kyy=Kyy_bearing, cxx=Cxx_bearing, cyy=Cyy_bearing, w=wb
    )
    return bearing0


def test_bearing_interpol_kxx(bearing0):
    assert_allclose(bearing0.kxx.interpolated(314.2), 8.5e7)
    assert_allclose(bearing0.kxx.interpolated(1151.9), 2.6e8)


def test_bearing_interpol_kyy(bearing0):
    assert_allclose(bearing0.kyy.interpolated(314.2), 9.2e7)
    assert_allclose(bearing0.kyy.interpolated(1151.9), 2.6e8)


def test_bearing_interpol_cxx(bearing0):
    assert_allclose(bearing0.cxx.interpolated(314.2), 226837, rtol=1e5)
    assert_allclose(bearing0.cxx.interpolated(1151.9), 137973, rtol=1e5)


def test_bearing_interpol_cyy(bearing0):
    assert_allclose(bearing0.kxx.interpolated(314.2), 235837, rtol=1e5)
    assert_allclose(bearing0.kxx.interpolated(1151.9), 2.6e8, rtol=1e5)


@pytest.fixture
def bearing1():
    # using lists
    Kxx_bearing = [
        8.5e07,
        1.1e08,
        1.3e08,
        1.6e08,
        1.8e08,
        2.0e08,
        2.3e08,
        2.5e08,
        2.6e08,
    ]
    Kyy_bearing = np.array(
        [9.2e07, 1.1e08, 1.4e08, 1.6e08, 1.9e08, 2.1e08, 2.3e08, 2.5e08, 2.6e08]
    )
    Cxx_bearing = np.array(
        [226837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    Cyy_bearing = np.array(
        [235837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    wb = [314.2, 418.9, 523.6, 628.3, 733.0, 837.8, 942.5, 1047.2, 1151.9]
    bearing1 = BearingElement(
        4, kxx=Kxx_bearing, kyy=Kyy_bearing, cxx=Cxx_bearing, cyy=Cyy_bearing, w=wb
    )
    return bearing1


def test_index(bearing1):
    assert bearing1.dof_local_index()[0] == 0
    assert bearing1.dof_local_index().x0 == 0
    assert bearing1.dof_local_index()[1] == 1
    assert bearing1.dof_local_index().y0 == 1
    assert bearing1.dof_global_index().x0 == 16
    assert bearing1.dof_global_index().y0 == 17
    assert bearing1.dof_global_index()[-1] == 17


def test_bearing1_interpol_kxx(bearing1):
    assert_allclose(bearing1.kxx.interpolated(314.2), 8.5e7)
    assert_allclose(bearing1.kxx.interpolated(1151.9), 2.6e8)


def test_bearing1_interpol_kyy(bearing1):
    assert_allclose(bearing1.kyy.interpolated(314.2), 9.2e7)
    assert_allclose(bearing1.kyy.interpolated(1151.9), 2.6e8)


def test_bearing1_interpol_cxx(bearing1):
    assert_allclose(bearing1.cxx.interpolated(314.2), 226837, rtol=1e5)
    assert_allclose(bearing1.cxx.interpolated(1151.9), 137973, rtol=1e5)


def test_bearing1_interpol_cyy(bearing1):
    assert_allclose(bearing1.kxx.interpolated(314.2), 235837, rtol=1e5)
    assert_allclose(bearing1.kxx.interpolated(1151.9), 2.6e8, rtol=1e5)


def test_bearing1_matrices(bearing1):
    # fmt: off
    K = np.array([[85000000.043218,        0.      ],
                  [       0.      , 91999999.891728]])
    C = np.array([[226836.917649,      0.          ],
                  [       0.      , 235836.850213  ]])
    # fmt: on
    assert_allclose(bearing1.K(314.2), K)
    assert_allclose(bearing1.C(314.2), C)


def test_bearing_error_speed_not_given():
    speed = np.linspace(0, 10000, 5)
    kx = 1e8 * speed
    cx = 1e8 * speed
    with pytest.raises(Exception) as excinfo:
        BearingElement(-1, kxx=kx, cxx=cx)
    assert "Arguments (coefficients and w)" " must have the same dimension" in str(
        excinfo.value
    )


def test_bearing_error2():
    with pytest.raises(ValueError) as excinfo:
        BearingElement(
            4, kxx=[7e8, 8e8, 9e8], cxx=[0, 0, 0, 0], w=[10, 100, 1000, 10000]
        )
    assert "Arguments (coefficients and w) " "must have the same dimension" in str(
        excinfo.value
    )

    with pytest.raises(ValueError) as excinfo:
        BearingElement(4, kxx=[6e8, 7e8, 8e8, 9e8], cxx=[0, 0, 0, 0, 0])
    assert "Arguments (coefficients and w) " "must have the same dimension" in str(
        excinfo.value
    )


@pytest.fixture
def bearing_constant():
    bearing = BearingElement(n=4, kxx=8e7, cxx=0)
    return bearing


def test_bearing_constant(bearing_constant):
    assert_allclose(bearing_constant.kxx.interpolated(314.2), 8e7, rtol=1e5)
    assert_allclose(bearing_constant.cxx.interpolated(300.9), 0, rtol=1e5)


def test_bearing_len_2():
    bearing = BearingElement(
        n=0,
        kxx=[481, 4810],
        cxx=[3.13, 10.81],
        kyy=[481, 4810],
        kxy=[194, 2078],
        kyx=[-194, -2078],
        cyy=[3.13, 10.81],
        cxy=[0.276, 0.69],
        cyx=[-0.276, -0.69],
        w=[115.19, 345.575],
    )
    assert_allclose(bearing.kxx.interpolated(115.19), 481, rtol=1e5)


def test_bearing_len_3():
    bearing = BearingElement(
        n=0,
        kxx=[481, 4810, 18810],
        cxx=[3.13, 10.81, 22.99],
        kyy=[481, 4810, 18810],
        kxy=[194, 2078, 8776],
        kyx=[-194, -2078, -8776],
        cyy=[3.13, 10.81, 22.99],
        cxy=[0.276, 0.69, 1.19],
        cyx=[-0.276, -0.69, -1.19],
        w=[115.19, 345.575, 691.15],
    )
    assert_allclose(bearing.kxx.interpolated(115.19), 481, rtol=1e5)


def test_equality(bearing0, bearing1, bearing_constant):
    assert bearing0 == bearing0
    assert bearing0 == bearing1
    assert not bearing0 == bearing_constant
    assert not bearing0 == 1


def test_from_table():
    bearing_file = (
        os.path.dirname(os.path.realpath(__file__)) + "/data/bearing_seal.xls"
    )
    bearing = BearingElement.from_table(0, bearing_file)
    assert bearing.n == 0
    assert math.isclose(bearing.w[2], 523.6, abs_tol=0.1)


def test_bearing_link():
    b0 = BearingElement(n=0, n_link=3, kxx=1, cxx=1)
    # fmt: off
    M = np.array(
        [[1, 0, -1, 0],
         [0, 1, 0, -1],
         [-1, 0, 1, 0],
         [0, -1, 0, 1]]
    )
    # fmt: on

    assert_allclose(b0.K(0), M)
    assert_allclose(b0.C(0), M)
