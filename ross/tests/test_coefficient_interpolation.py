import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearing_seal_element import (
    MIN_RECOMMENDED_AXIS_POINTS,
    BearingCoefficient,
    BearingElement,
    SealElement,
)
from ross.disk_element import DiskElement
from ross.materials import steel
from ross.rotor_assembly import Rotor
from ross.shaft_element import ShaftElement


@pytest.fixture
def speed_axis():
    return np.array([100.0, 200.0, 300.0, 400.0, 500.0])


def test_pchip_passes_through_data_without_overshoot(speed_axis):
    damping = np.array([30.0, 12.0, 7.0, 5.5, 5.0])
    coefficient = BearingCoefficient(damping, speed=speed_axis)

    assert coefficient.interpolation == "pchip"
    assert_allclose(coefficient(speed_axis), damping, rtol=1e-14)

    dense = np.linspace(speed_axis[0], speed_axis[-1], 401)
    values = coefficient(dense)
    assert np.all(np.diff(values) <= 1e-12)
    assert values.min() >= damping.min() - 1e-12
    assert values.max() <= damping.max() + 1e-12


def test_small_coefficients_are_not_smoothed():
    speed = np.array([314.2, 418.9, 523.6, 628.3, 733.0, 837.8, 942.5, 1047.2, 1151.9])
    damping = np.array([3.13, 10.81, 22.99, 30.1, 33.2, 34.0, 33.1, 31.0, 28.4])
    bearing = BearingElement(0, kxx=1e6, cxx=damping, speed=speed)
    assert_allclose(bearing.cxx_interpolated(speed), damping, rtol=1e-14)


def test_two_points_interpolate_linearly():
    coefficient = BearingCoefficient([1e6, 3e6], speed=[100.0, 300.0])
    assert_allclose(float(coefficient(200.0)), 2e6)
    assert_allclose(float(coefficient(400.0)), 4e6)


def test_linear_extrapolation_from_end_slope(speed_axis):
    stiffness = speed_axis**2 * 10.0
    coefficient = BearingCoefficient(stiffness, speed=speed_axis)
    above = coefficient(np.array([600.0, 700.0, 800.0]))
    assert_allclose(np.diff(above, 2), 0.0, atol=1e-6)
    below = coefficient(np.array([0.0, 50.0, 100.0]))
    assert_allclose(np.diff(below, 2), 0.0, atol=1e-6)
    assert above[0] > stiffness[-1]


def test_linear_interpolation_option(speed_axis):
    stiffness = speed_axis**2 * 10.0
    linear = BearingCoefficient(stiffness, speed=speed_axis, interpolation="linear")
    assert_allclose(float(linear(250.0)), 0.5 * (stiffness[1] + stiffness[2]))
    bearing = BearingElement(
        0, kxx=stiffness, cxx=1.0, speed=speed_axis, interpolation="linear"
    )
    assert bearing.interpolation == "linear"
    assert_allclose(bearing.K(250.0)[0, 0], 0.5 * (stiffness[1] + stiffness[2]))
    with pytest.raises(ValueError, match="interpolation must be one of"):
        BearingCoefficient(stiffness, speed=speed_axis, interpolation="cubic")


def test_grid_with_one_column_matches_one_dimensional_table(speed_axis):
    stiffness = np.array([1.0, 1.8, 2.9, 4.5, 6.0]) * 1e7
    table_1d = BearingCoefficient(stiffness, speed=speed_axis)
    table_2d = BearingCoefficient(
        stiffness[:, None], speed=speed_axis, frequency=[50.0]
    )
    queries = np.array([120.0, 250.0, 333.0, 480.0, 650.0])
    assert_allclose(table_2d(50.0, queries), table_1d(queries), rtol=1e-14)


def test_grid_interpolation_is_exact_on_points_and_linear_option_is_bilinear(
    speed_axis,
):
    frequency = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    table = np.outer(speed_axis, frequency)
    pchip = BearingCoefficient(table, speed=speed_axis, frequency=frequency)
    linear = BearingCoefficient(
        table, speed=speed_axis, frequency=frequency, interpolation="linear"
    )
    sq, fq = np.meshgrid(speed_axis, frequency, indexing="ij")
    assert_allclose(pchip(fq, sq), table, rtol=1e-14)
    assert_allclose(float(linear(25.0, 250.0)), 250.0 * 25.0)
    assert_allclose(float(pchip(25.0, 250.0)), 250.0 * 25.0, rtol=1e-12)


def test_interpolation_round_trips_through_save_load(speed_axis, tmp_path):
    bearing = BearingElement(
        0, kxx=speed_axis * 1e4, cxx=1.0, speed=speed_axis, interpolation="linear"
    )
    file = tmp_path / "bearing.toml"
    bearing.save(file)
    loaded = BearingElement.load(file)
    assert loaded.interpolation == "linear"
    assert loaded == bearing


def _rotor_with_bearing_table(speed, frequency=None, kxy=None):
    shaft = [ShaftElement(0.25, 0, 0.05, material=steel) for _ in range(6)]
    disks = [
        DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28),
        DiskElement.from_geometry(n=4, material=steel, width=0.07, i_d=0.05, o_d=0.35),
    ]
    stiffness = 1e6 + 1e3 * np.asarray(speed)
    bearings = [
        BearingElement(0, kxx=stiffness, cxx=100.0, speed=speed, tag="brg0"),
        BearingElement(6, kxx=stiffness, cxx=100.0, speed=speed, tag="brg1"),
    ]
    if frequency is not None:
        bearings.append(
            SealElement(
                3,
                kxx=1e4,
                cxx=10.0,
                kxy=kxy,
                kyx=-kxy,
                speed=speed,
                frequency=frequency,
                tag="seal",
            )
        )
    return Rotor(shaft, disks, bearings)


def test_sparse_table_warning_in_speed_sweep():
    rotor = _rotor_with_bearing_table(np.array([100.0, 300.0, 500.0]))
    with pytest.warns(
        UserWarning, match="brg0 are interpolated from only 3 speed points"
    ):
        rotor.run_campbell(np.array([150.0, 250.0, 350.0]), frequencies=4)


def test_no_sparse_warning_with_enough_points_or_on_grid():
    speed = np.linspace(100.0, 500.0, MIN_RECOMMENDED_AXIS_POINTS)
    rotor = _rotor_with_bearing_table(speed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rotor.run_campbell(np.array([150.0, 250.0, 350.0]), frequencies=4)

    sparse = _rotor_with_bearing_table(np.array([100.0, 300.0, 500.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sparse.run_campbell(np.array([100.0, 300.0, 500.0]), frequencies=4)


def test_extrapolation_warning_names_element_and_axis():
    rotor = _rotor_with_bearing_table(np.linspace(100.0, 500.0, 5))
    with pytest.warns(UserWarning, match="brg0 outside its speed axis"):
        rotor.run_campbell(np.array([200.0, 600.0]), frequencies=4)


def test_whirl_frequency_axis_warnings():
    speed = np.linspace(100.0, 500.0, 5)
    # the axis starts above the first modes (about 90 rad/s), so a matched
    # whirl analysis must leave it
    frequency = np.array([150.0, 225.0, 300.0, 375.0, 450.0])
    kxy = np.outer(speed, np.ones(5)) * 1e2 + np.outer(np.ones(5), frequency) * 1e3
    rotor = _rotor_with_bearing_table(speed, frequency=frequency, kxy=kxy)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rotor.run_modal(300.0, num_modes=8, frequency=200.0)

    with pytest.warns(UserWarning, match="seal outside its frequency axis"):
        rotor.run_modal(300.0, num_modes=8, frequency=1000.0)

    with pytest.warns(UserWarning, match="seal outside its frequency axis"):
        rotor.run_modal(300.0, num_modes=8, matched_whirl=True)

    with pytest.warns(UserWarning, match="seal outside its frequency axis"):
        rotor.run_freq_response(speed_range=np.array([100.0, 1000.0]), speed=300.0)
