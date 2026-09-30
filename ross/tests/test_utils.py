import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ross.rotor_assembly import rotor_example
from ross.units import Q_
from ross.utils import speed_range_from_endpoints


def test_speed_range_from_endpoints_default_num_points():
    spans = np.array([10.0, 100.0, 1000.0, 10000.0])
    num_points = np.array([len(speed_range_from_endpoints(0, span)) for span in spans])
    assert_equal(num_points, [64, 200, 633, 2000])
    assert np.all(np.diff(num_points) > 0)
    assert np.all(np.diff(num_points / spans) < 0)

    assert len(speed_range_from_endpoints(0, 0.5)) == 20
    assert len(speed_range_from_endpoints(0, 1e6)) == 2000
    assert len(speed_range_from_endpoints(0, 400, points_factor=4)) == 80


def test_speed_range_from_endpoints_units():
    speed_range = speed_range_from_endpoints(
        Q_(0, "Hz"), Q_(3000, "RPM"), num_points=31
    )
    assert_allclose(speed_range, np.linspace(0, 100 * np.pi, 31))

    speed_range = speed_range_from_endpoints(Q_(10, "Hz"), 100 * np.pi, num_points=5)
    assert_allclose(speed_range[[0, -1]], [20 * np.pi, 100 * np.pi])


def test_speed_range_from_endpoints_errors():
    with pytest.raises(ValueError, match="greater than"):
        speed_range_from_endpoints(100, 100)
    with pytest.raises(ValueError, match="at least 2"):
        speed_range_from_endpoints(0, 100, num_points=1)


def test_speed_range_from_endpoints_in_analyses():
    rotor = rotor_example()
    speed_range = speed_range_from_endpoints(Q_(0, "RPM"), Q_(50, "Hz"), num_points=11)
    expected = np.linspace(0, 100 * np.pi, 11)

    assert_allclose(rotor.run_campbell(speed_range).speed_range, expected)
    assert_allclose(
        rotor.run_unbalance_response(3, 1e-4, 0, speed_range=speed_range).speed_range,
        expected,
    )
