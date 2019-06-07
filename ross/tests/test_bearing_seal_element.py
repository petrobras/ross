import numpy as np
import pytest
from numpy.testing import assert_allclose

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


def test_bearing_error1():
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


def test_equality(bearing0, bearing1, bearing_constant):
    assert bearing0 == bearing0
    assert bearing0 == bearing1
    assert not bearing0 == bearing_constant
    assert not bearing0 == 1
