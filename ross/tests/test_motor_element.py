"""Tests for the MotorElement class.

Tests are based on the operating scenarios described in
"Testes dos modelos do motor e conversor - ROSS.docx".

All tests use the parameters from ``motor_example()`` and the default
simulation parameters defined in the module.

Motor under test
----------------
- Nominal power  : 1.5 cv  (≈ 1103.25 W)
- Nominal voltage: 127 V (phase)
- Nominal speed  : 1710 RPM
- Nominal frequency: 60 Hz
- Poles          : 4
- Rs = 2.5 Ω, Rr = 1.8 Ω, Xs = Xr = 1.3 Ω, Xm = 43.08 Ω
- Ip_motor = 0.0372 kg·m²
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.motor.motor_element import MotorElement, motor_example
from ross.units import Q_


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def motor():
    """Return the reference motor from motor_example()."""
    return motor_example()


@pytest.fixture(scope="module")
def motor_high_inertia(motor):
    """Return the motor with Ip_motor × 10 000 to simulate a locked rotor.

    With such a large inertia the mechanical speed barely changes during the
    simulation window, so the electrical quantities converge to the
    blocked-rotor (starting) operating point.
    """
    return MotorElement(
        n=motor.n,
        tag=motor.tag,
        power_nom=motor.power_nom,
        voltage_nom=motor.voltage_nom,
        speed_nom=motor.speed_nom,
        frequency_nom=motor.frequency_nom,
        n_poles=motor.n_poles,
        stator_resistance=motor.stator_resistance,
        rotor_resistance=motor.rotor_resistance,
        stator_reactance=motor.stator_reactance,
        rotor_reactance=motor.rotor_reactance,
        mutual_reactance=motor.mutual_reactance,
        Ip_motor=motor.Ip_motor * 10_000,
        viscosity_coeff=motor.viscosity_coeff,
        Ip_load=motor.Ip_load,
        voltage_net=motor.voltage_net,
        frequency_net=motor.frequency_net,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def rms(signal):
    """Return the RMS value of *signal*."""
    return np.sqrt(np.mean(signal**2))


# ---------------------------------------------------------------------------
# Parameter tests (no simulation required)
# ---------------------------------------------------------------------------


def test_motor_example_parameters(motor):
    """Verify that motor_example() returns the expected nominal parameters."""
    assert_allclose(motor.power_nom, 1103.248125, rtol=1e-6)
    assert_allclose(motor.voltage_nom, 127.0, rtol=1e-6)
    assert_allclose(
        motor.speed_nom,
        Q_(1710, "RPM").to("rad/s").m,
        rtol=1e-5,
    )
    assert_allclose(
        motor.frequency_nom,
        Q_(60.0, "Hz").to("rad/s").m,
        rtol=1e-6,
    )
    assert motor.n_poles == 4
    assert_allclose(motor.stator_resistance, 2.5, rtol=1e-9)
    assert_allclose(motor.rotor_resistance, 1.8, rtol=1e-9)
    assert_allclose(motor.stator_reactance, 1.3, rtol=1e-9)
    assert_allclose(motor.rotor_reactance, 1.3, rtol=1e-9)
    assert_allclose(motor.mutual_reactance, 43.08, rtol=1e-9)
    assert_allclose(motor.Ip_motor, 0.0372, rtol=1e-9)


def test_motor_example_equality():
    """Two calls to motor_example() must return equal objects."""
    m1 = motor_example()
    m2 = motor_example()
    assert m1 == m2


# ---------------------------------------------------------------------------
# Test 1 – No-load (vazio) operation at nominal voltage
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results_no_load(motor):
    """Simulate the motor at no load for 3 s (nominal voltage)."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_with_AC_source(
        t,
        load_torque_entrance_time=tf + 1.0,  # load applied after simulation ends
        load_torque_ratio=0.0,
    )


def _steady_state_slice(results, fraction=0.85):
    """Return the index at which steady state is considered to begin."""
    return int(fraction * len(results.t))


def test_no_load_stator_current_rms(results_no_load):
    """No-load stator current (RMS) must be approximately 2.8 A."""
    ss = _steady_state_slice(results_no_load)
    ia_rms = rms(results_no_load.currents["a"][ss:])
    # Document: ~2.8 A rms; tolerance set to ±10 % of expected value
    assert_allclose(ia_rms, 2.85, rtol=0.10, atol=0.1,
                    err_msg="No-load RMS current outside expected range (~2.8 A)")


def test_no_load_stator_current_peak(results_no_load):
    """No-load stator current (peak) must be approximately 4 A."""
    ss = _steady_state_slice(results_no_load)
    ia_peak = np.max(np.abs(results_no_load.currents["a"][ss:]))
    # Document: ~4 A peak; tolerance ±10 %
    assert_allclose(ia_peak, 4.04, rtol=0.10, atol=0.1,
                    err_msg="No-load peak current outside expected range (~4 A)")


def test_no_load_speed(results_no_load):
    """No-load rotor speed must be approximately 1799 RPM."""
    ss = _steady_state_slice(results_no_load)
    speed_rpm = np.mean(results_no_load.speed[ss:]) * 60.0 / (2.0 * np.pi)
    # Document: 1799 RPM; tolerance ±5 RPM
    assert_allclose(speed_rpm, 1800.0, atol=5.0,
                    err_msg="No-load speed outside expected range (~1799 RPM)")


# ---------------------------------------------------------------------------
# Test 2 – Nominal load operation at nominal voltage
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results_nominal_load(motor):
    """Simulate the motor at nominal load for 3 s (nominal voltage)."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_with_AC_source(
        t,
        load_torque_entrance_time=0.5,
        load_torque_ratio=1.0,
    )


def test_nominal_load_stator_current_rms(results_nominal_load):
    """Nominal-load stator current (RMS) must be approximately 4.25 A."""
    ss = _steady_state_slice(results_nominal_load)
    ia_rms = rms(results_nominal_load.currents["a"][ss:])
    # Document: ~4.25 A rms; tolerance ±10 %
    assert_allclose(ia_rms, 4.38, rtol=0.10, atol=0.15,
                    err_msg="Nominal-load RMS current outside expected range (~4.25 A)")


def test_nominal_load_stator_current_peak(results_nominal_load):
    """Nominal-load stator current (peak) must be approximately 6.0 A."""
    ss = _steady_state_slice(results_nominal_load)
    ia_peak = np.max(np.abs(results_nominal_load.currents["a"][ss:]))
    # Document: ~6.0 A peak; tolerance ±10 %
    assert_allclose(ia_peak, 6.19, rtol=0.10, atol=0.2,
                    err_msg="Nominal-load peak current outside expected range (~6.0 A)")


def test_nominal_load_speed(results_nominal_load):
    """Nominal-load rotor speed must be approximately 1710 RPM."""
    ss = _steady_state_slice(results_nominal_load)
    speed_rpm = np.mean(results_nominal_load.speed[ss:]) * 60.0 / (2.0 * np.pi)
    # Document: 1710 RPM; tolerance ±15 RPM
    assert_allclose(speed_rpm, 1710.0, atol=15.0,
                    err_msg="Nominal-load speed outside expected range (~1710 RPM)")


# ---------------------------------------------------------------------------
# Test 3 – Locked-rotor / starting current and torque
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results_locked_rotor(motor_high_inertia):
    """Simulate starting current with very high inertia (rotor approximately locked).

    With Ip × 10 000, the speed barely changes; after the initial transient
    settles the electrical signals represent the blocked-rotor condition.
    """
    dt = 1e-4
    tf = 0.5
    t = np.arange(0, tf + dt, dt)
    return motor_high_inertia.run_with_AC_source(
        t,
        load_torque_entrance_time=0.0,
        load_torque_ratio=1.0,
    )


def test_starting_current_rms(results_locked_rotor):
    """Starting (locked-rotor) current RMS must be approximately 25 A."""
    ss = int(0.5 * len(results_locked_rotor.t))
    ia_rms = rms(results_locked_rotor.currents["a"][ss:])
    # Document: ~25 A rms; tolerance ±10 %
    assert_allclose(ia_rms, 25.64, rtol=0.10, atol=0.5,
                    err_msg="Starting RMS current outside expected range (~25 A)")


def test_starting_current_peak(results_locked_rotor):
    """Starting (locked-rotor) current peak must be approximately 35.25 A."""
    ss = int(0.5 * len(results_locked_rotor.t))
    ia_peak = np.max(np.abs(results_locked_rotor.currents["a"][ss:]))
    # Document: ~35.25 A peak; tolerance ±10 %
    assert_allclose(ia_peak, 36.26, rtol=0.10, atol=0.5,
                    err_msg="Starting peak current outside expected range (~35.25 A)")


def test_starting_electromagnetic_torque(results_locked_rotor):
    """Starting electromagnetic torque must be approximately 17.7 N·m."""
    ss = int(0.5 * len(results_locked_rotor.t))
    te_mean = np.mean(np.abs(results_locked_rotor.electric_torque[ss:]))
    # Document: ~17.7 N.m; tolerance ±10 %
    assert_allclose(te_mean, 17.70, rtol=0.10, atol=0.3,
                    err_msg="Starting electromagnetic torque outside expected range (~17.7 N·m)")


def test_speed_nearly_zero_during_locked_rotor(results_locked_rotor):
    """With Ip × 10 000 the rotor speed must remain near zero throughout."""
    speed_rpm = results_locked_rotor.speed * 60.0 / (2.0 * np.pi)
    # Speed should not exceed 5 RPM during the 0.5 s window
    assert np.max(np.abs(speed_rpm)) < 5.0, (
        f"Speed too high during locked-rotor test: {np.max(np.abs(speed_rpm)):.2f} RPM"
    )
