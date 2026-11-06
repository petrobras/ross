"""Tests for the induction motor parameter estimation methods.

The estimated circuits are checked with an independent solver of the per-phase
T equivalent circuit (``circuit_at_slip``). The rated data are the Chapman
25 hp, 460 V, 60 Hz, 4 pole example (Chapman, Electric Machinery Fundamentals,
Example 6.3) and a 4 kV, 3.3 MW machine.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.motors import parameter_estimation as pe
from ross.motors.motor_element import MotorElement
from ross.units import Q_

FREQUENCY = 2 * np.pi * 60


def circuit_at_slip(result, slip, voltage, rotor=None, series_resistance=0.0):
    """Solve the T equivalent circuit of an estimation result at a slip."""
    R2, X2 = (
        (result.rotor_resistance, result.rotor_reactance) if rotor is None else rotor
    )
    z1 = result.stator_resistance + 1j * result.stator_reactance
    if np.isinf(result.Rc):
        z_mag = 1j * result.mutual_reactance
    else:
        z_mag = 1 / (1 / result.Rc + 1 / (1j * result.mutual_reactance))
    z2 = R2 / slip + series_resistance + 1j * X2
    i1 = voltage / (z1 + z_mag * z2 / (z_mag + z2))
    i2 = (voltage - i1 * z1) / z2
    s_in = 3 * voltage * np.conj(i1)
    air_gap_power = 3 * abs(i2) ** 2 * R2 / slip
    return {
        "current": abs(i1),
        "power_factor": s_in.real / abs(s_in),
        "input_power": s_in.real,
        "reactive_power": s_in.imag,
        "air_gap_power": air_gap_power,
        "converted_power": (1 - slip) * air_gap_power,
    }


@pytest.fixture
def chapman():
    return pe._Rating(
        power_rated=Q_(25, "hp").to("W").m,
        voltage_rated=460 / np.sqrt(3),
        speed_rated=Q_(1724.598, "RPM").to("rad/s").m,
        frequency_rated=FREQUENCY,
        n_poles=4,
        efficiency=0.829204,
        power_factor=0.903327,
        current_rated=31.2503,
        starting_current_ratio=4.624842,
    )


@pytest.fixture
def large_motor():
    return pe._Rating(
        power_rated=Q_(3317.69, "hp").to("W").m,
        voltage_rated=4000 / np.sqrt(3),
        speed_rated=Q_(1791, "RPM").to("rad/s").m,
        frequency_rated=FREQUENCY,
        n_poles=4,
        efficiency=0.97,
        power_factor=0.9,
    )


def rated_point(rating, result, series_resistance=0.0):
    return circuit_at_slip(
        result, rating.slip, rating.voltage_rated, series_resistance=series_resistance
    )


@pytest.mark.parametrize("rating_name", ["chapman", "large_motor"])
def test_asdp_reproduces_rated_point(rating_name, request):
    rating = request.getfixturevalue(rating_name)
    result = pe.asdp(rating)
    point = rated_point(rating, result)

    assert result.converged
    # ASDP stops when every rated quantity is within a 2 % band; the stopping
    # tolerance is the only source of error of this check.
    band = 0.02
    assert_allclose(point["current"], rating.current_rated, rtol=band)
    assert_allclose(point["power_factor"], rating.power_factor, rtol=band)
    assert_allclose(point["converted_power"], rating.power_rated, rtol=band)
    assert_allclose(
        point["converted_power"] / point["input_power"], rating.efficiency, rtol=band
    )


@pytest.mark.parametrize("rating_name", ["chapman", "large_motor"])
def test_lee_reproduces_rated_point(rating_name, request):
    rating = request.getfixturevalue(rating_name)
    result = pe.lee(rating)
    point = rated_point(rating, result)

    input_current = rating.input_power / (
        3 * rating.voltage_rated * rating.power_factor
    )
    # Lee stops when the parameters change less than 1e-3 pu per iteration.
    assert_allclose(point["current"], input_current, rtol=1e-3)
    assert_allclose(point["power_factor"], rating.power_factor, rtol=1e-3)
    mechanical_and_stray = 0.14 * (1 / rating.efficiency - 1) + pe._stray_loss_fraction(
        rating.power_rated
    )
    assert_allclose(
        point["converted_power"] / rating.power_rated,
        1 + mechanical_and_stray,
        rtol=1e-3,
    )


def test_kueck_known_stator_resistance_reproduces_rated_point(chapman):
    result = pe.kueck(chapman, known_stator_resistance=0.641)
    point = rated_point(chapman, result, result.extra["Rsll"])

    assert result.stator_resistance == 0.641
    # The relaxed iteration ends with a physical but not converged solution, so
    # the rated point is reproduced within 2 %, not to the stopping tolerance.
    assert_allclose(point["current"], chapman.current_rated, rtol=2e-2)
    assert_allclose(point["power_factor"], chapman.power_factor, rtol=2e-2)
    assert_allclose(
        point["converted_power"],
        chapman.power_rated + result.extra["Pfw"],
        rtol=3e-2,
    )


def test_kueck_estimated_r1_formula(chapman):
    result = pe.kueck(chapman)
    expected = 0.00011 * 4**0.52 * 25 ** (-1.26) * 460**2
    assert_allclose(result.stator_resistance, expected, rtol=1e-3)


@pytest.mark.parametrize(
    "rating_name, efficiency_50, power_factor_50",
    [("chapman", 0.80, 0.80), ("large_motor", 0.965, 0.82)],
)
def test_haque1_reproduces_rated_point(
    rating_name, efficiency_50, power_factor_50, request
):
    rating = request.getfixturevalue(rating_name)
    result = pe.haque1(
        rating,
        starting_torque_ratio=1.5,
        efficiency_50=efficiency_50,
        power_factor_50=power_factor_50,
    )
    point = rated_point(rating, result)

    assert result.residual < 1e-8
    assert_allclose(point["input_power"], rating.input_power, rtol=1e-6)
    assert_allclose(point["power_factor"], rating.power_factor, rtol=1e-6)
    assert_allclose(
        point["converted_power"],
        rating.power_rated + result.extra["Pfw"],
        rtol=1e-6,
    )


def test_haque1_leakage_split_follows_category(chapman):
    categories = {"N": 0.4, "H": 0.3, "D": 0.5}
    for category, fraction in categories.items():
        rating = pe._Rating(
            chapman.power_rated,
            chapman.voltage_rated,
            chapman.speed_rated,
            FREQUENCY,
            4,
            chapman.efficiency,
            chapman.power_factor,
            chapman.current_rated,
            chapman.starting_current_ratio,
            category,
        )
        result = pe.haque1(rating, 1.5, 0.80, 0.80)
        assert_allclose(
            result.stator_reactance
            / (result.stator_reactance + result.rotor_reactance),
            fraction,
            rtol=1e-12,
        )


def test_natarajan_misra_loss_balance(chapman):
    data = dict(
        efficiency_75=0.83,
        efficiency_50=0.80,
        power_factor_75=0.88,
        power_factor_50=0.82,
        starting_torque_ratio=1.5,
        maximum_torque_ratio=2.5,
    )
    result = pe.natarajan_misra(chapman, **data)

    resistance = result.stator_resistance + result.rotor_resistance
    constant_loss = result.extra["K"]
    for load, efficiency, power_factor in [
        (0.75, data["efficiency_75"], data["power_factor_75"]),
        (0.50, data["efficiency_50"], data["power_factor_50"]),
    ]:
        current = (
            load
            * chapman.power_rated
            / (3 * chapman.voltage_rated * efficiency * power_factor)
        )
        loss = load * chapman.power_rated * (1 / efficiency - 1)
        # At 50 % load the point is not used in the fit, so the loss model
        # reproduces it only approximately; the fitted points are exact.
        rtol = 1e-9 if load == 0.75 else 5e-2
        assert_allclose(3 * current**2 * resistance + constant_loss, loss, rtol=rtol)

    assert result.residual < 5e-2
    assert_allclose(
        result.stator_reactance / result.rotor_reactance,
        chapman.x1_x2_ratio,
        rtol=1e-12,
    )


def test_natarajan_misra_rejects_inconsistent_torque_ratios(chapman):
    with pytest.raises(RuntimeError):
        pe.natarajan_misra(chapman, 0.83, 0.80, 0.88, 0.82, 3.0, 2.5)


@pytest.mark.parametrize("starting, maximum", [(1.5, 2.0), (2.0, 2.5), (2.5, 3.0)])
def test_haque2_reproduces_catalog_data(chapman, starting, maximum):
    result = pe.haque2(chapman, starting, maximum)

    assert result.converged
    voltage = chapman.voltage_rated
    rated = circuit_at_slip(result, chapman.slip, voltage)
    reactive_power = chapman.input_power * np.tan(np.arccos(chapman.power_factor))
    assert_allclose(rated["input_power"], chapman.input_power, rtol=1e-8)
    assert_allclose(rated["reactive_power"], reactive_power, rtol=1e-8)
    assert_allclose(
        rated["converted_power"] - result.extra["Pfw"], chapman.power_rated, rtol=1e-8
    )

    sm = result.extra["smax"]
    rotor_at_start = (
        (0.5 + 0.5 * np.sqrt(1 / sm)) * result.rotor_resistance,
        (0.4 + 0.6 * np.sqrt(sm / 1)) * result.rotor_reactance,
    )
    start = circuit_at_slip(result, 1.0, voltage, rotor=rotor_at_start)
    rated_torque = (chapman.power_rated + result.extra["Pfw"]) / chapman.speed_rated
    omega_sync = 2 * FREQUENCY / 4
    assert_allclose(
        start["air_gap_power"] / omega_sync / rated_torque, starting, rtol=1e-8
    )


def test_haque2_warns_when_data_is_inconsistent(chapman):
    result = pe.haque2(chapman, 1.0, 3.0)
    assert not result.converged
    assert result.warnings


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(efficiency=1.2),
        dict(power_factor=0.0),
        dict(starting_current_ratio=1.0),
        dict(motor_category="X"),
        dict(speed_rated=2 * FREQUENCY / 4 * 1.01),
    ],
)
def test_rating_validation(chapman, kwargs):
    data = dict(
        power_rated=chapman.power_rated,
        voltage_rated=chapman.voltage_rated,
        speed_rated=chapman.speed_rated,
        frequency_rated=FREQUENCY,
        n_poles=4,
        efficiency=chapman.efficiency,
        power_factor=chapman.power_factor,
    )
    data.update(kwargs)
    with pytest.raises(ValueError):
        pe._Rating(**data)


RATED = dict(
    power_rated=Q_(25, "hp").to("W").m,
    voltage_rated=460 / np.sqrt(3),
    speed_rated=Q_(1724.598, "RPM").to("rad/s").m,
    frequency_rated=FREQUENCY,
    n_poles=4,
    efficiency=0.829204,
    power_factor=0.903327,
    current_rated=31.2503,
    starting_current_ratio=4.624842,
)


def test_estimate_from_catalog_requires_method_data():
    with pytest.raises(ValueError, match="efficiency_75"):
        pe.estimate_from_catalog(
            "natarajan_misra", dict(starting_torque_ratio=1.5), **RATED
        )
    with pytest.raises(ValueError, match="method"):
        pe.estimate_from_catalog("unknown", {}, **RATED)
    with pytest.raises(ValueError, match="method"):
        pe.estimate_from_nameplate("unknown", **RATED)


NAMEPLATE = dict(
    n=0,
    power_rated=Q_(25, "hp"),
    voltage_rated=460 / np.sqrt(3),
    speed_rated=Q_(1724.598, "RPM"),
    efficiency=0.829204,
    power_factor=0.903327,
    Ip_motor=0.1,
    current_rated=31.2503,
    starting_current_ratio=4.624842,
)


@pytest.mark.parametrize("method", ["asdp", "lee"])
def test_from_nameplate_matches_estimation(method):
    motor = MotorElement.from_nameplate(**NAMEPLATE, method=method, tag="M")
    result = pe.estimate_from_nameplate(method, **RATED)

    assert isinstance(motor, MotorElement)
    assert motor.tag == "M"
    assert_allclose(
        [
            motor.stator_resistance,
            motor.rotor_resistance,
            motor.stator_reactance,
            motor.rotor_reactance,
            motor.mutual_reactance,
        ],
        [
            result.stator_resistance,
            result.rotor_resistance,
            result.stator_reactance,
            result.rotor_reactance,
            result.mutual_reactance,
        ],
    )
    assert_allclose(motor.power_rated, Q_(25, "hp").to("W").m)
    assert_allclose(motor.frequency_rated, FREQUENCY)


@pytest.mark.filterwarnings("ignore:The Kueck estimation did not converge")
def test_from_nameplate_method_options():
    motor = MotorElement.from_nameplate(
        **NAMEPLATE, method="kueck", method_options=dict(known_stator_resistance=0.641)
    )
    assert motor.stator_resistance == 0.641


def test_from_nameplate_warns_without_convergence():
    with pytest.warns(UserWarning, match="did not converge"):
        MotorElement.from_nameplate(**NAMEPLATE, method="kueck")


def test_from_nameplate_unknown_method():
    with pytest.raises(ValueError, match="method"):
        MotorElement.from_nameplate(**NAMEPLATE, method="unknown")


def test_from_catalog_requires_data():
    with pytest.raises(ValueError, match="maximum_torque_ratio"):
        MotorElement.from_catalog(**NAMEPLATE, starting_torque_ratio=2.0)


@pytest.mark.parametrize("method", ["haque1", "natarajan_misra", "haque2"])
def test_from_catalog_builds_motor(method):
    motor = MotorElement.from_catalog(
        **NAMEPLATE,
        starting_torque_ratio=2.0,
        maximum_torque_ratio=2.5,
        efficiency_75=0.83,
        power_factor_75=0.88,
        efficiency_50=0.80,
        power_factor_50=0.82,
        method=method,
        Ip_load=0.2,
    )
    assert motor.Ip_load == 0.2
    assert (
        min(
            motor.stator_resistance,
            motor.rotor_resistance,
            motor.stator_reactance,
            motor.rotor_reactance,
            motor.mutual_reactance,
        )
        > 0
    )


def test_estimated_motor_runs_at_rated_speed():
    motor = MotorElement.from_catalog(
        **NAMEPLATE, starting_torque_ratio=2.0, maximum_torque_ratio=2.5
    )
    t = np.arange(0, 3.0, 1e-4)
    results = motor.run_direct_on_line(
        t, load_torque_entrance_time=0.5, load_torque_ratio=1.0
    )
    speed = np.mean(results.speed[int(0.8 * len(t)) :])
    # The model has no core loss branch and Haque 2 includes friction losses in
    # the electromagnetic torque, so the speed is only close to the rated one.
    assert_allclose(speed, Q_(1724.598, "RPM").to("rad/s").m, rtol=1e-2)
