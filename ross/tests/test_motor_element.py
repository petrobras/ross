"""Tests for the MotorElement class.

Tests are based on the operating scenarios described in
"Testes dos modelos do motor e conversor - ROSS.docx", covering the three
electrical sources supported by ``MotorElement``:

- ``SourceAC`` (ideal AC source), driven through ``.run_direct_on_line()``;
- ``InverterVF`` (open-loop scalar V/f control), driven through
  ``.run_open_loop_vf_adjustment()``;
- ``InverterFOC`` (closed-loop indirect Field-Oriented Control), driven
  through ``.run_with_inverter_foc()``.

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
from copy import deepcopy

from ross.motors.motor_element import MotorElement
from ross.units import Q_


@pytest.fixture
def motor():
    """Return the motor element."""
    return MotorElement(
        n=0,
        tag="motor",
        power_nom=Q_(1.5, "cv"),
        voltage_nom=127,
        speed_nom=Q_(1710, "RPM"),
        frequency_nom=Q_(60.0, "Hz"),
        n_poles=4,
        stator_resistance=2.5,
        rotor_resistance=1.8,
        stator_reactance=1.3,
        rotor_reactance=1.3,
        mutual_reactance=43.08,
        Ip_motor=0.0372,
        viscosity_coeff=0.0,
        Ip_load=0.0,
        voltage_net=127,
        frequency_net=Q_(60.0, "Hz"),
    )


@pytest.fixture
def motor_high_inertia():
    """Return the motor with Ip_motor × 10 000 to simulate a locked rotor.

    With such a large inertia the mechanical speed barely changes during the
    simulation window, so the electrical quantities converge to the
    blocked-rotor (starting) operating point.
    """
    return MotorElement(
        n=0,
        tag="motor_high_inertia",
        power_nom=Q_(1.5, "cv"),
        voltage_nom=127,
        speed_nom=Q_(1710, "RPM"),
        frequency_nom=Q_(60.0, "Hz"),
        n_poles=4,
        stator_resistance=2.5,
        rotor_resistance=1.8,
        stator_reactance=1.3,
        rotor_reactance=1.3,
        mutual_reactance=43.08,
        Ip_motor=0.0372 * 10000,
        viscosity_coeff=0.0,
        Ip_load=0.0,
        voltage_net=127,
        frequency_net=Q_(60.0, "Hz"),
    )


def rms(signal):
    """Return the RMS value of *signal*."""
    return np.sqrt(np.mean(signal**2))


def _steady_state_slice(results, fraction=0.85):
    """Return the index at which steady state is considered to begin."""
    return int(fraction * len(results.t))


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


def test_motor_example_equality(motor):
    """Two calls to motor_example() must return equal objects."""
    m1 = motor
    m2 = deepcopy(motor)
    m2.tag = "motor_2"
    assert m1 == m2


@pytest.fixture
def results_no_load(motor):
    """Simulate the motor at no load for 3 s (nominal voltage)."""
    dt = 1e-4
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_direct_on_line(
        t,
        load_torque_entrance_time=tf + 1.0,  # load applied after simulation ends
        load_torque_ratio=0.0,
    )


def test_no_load_stator_current_rms(results_no_load):
    """No-load stator current (RMS) must be approximately 2.8 A."""
    ss = _steady_state_slice(results_no_load)
    ia_rms = rms(results_no_load.currents["a"][ss:])
    # Document: ~2.8 A rms; tolerance set to ±10 % of expected value
    assert_allclose(
        ia_rms,
        2.85,
        rtol=0.10,
        atol=0.1,
        err_msg="No-load RMS current outside expected range (~2.8 A)",
    )


def test_no_load_stator_current_peak(results_no_load):
    """No-load stator current (peak) must be approximately 4 A."""
    ss = _steady_state_slice(results_no_load)
    ia_peak = np.max(np.abs(results_no_load.currents["a"][ss:]))
    # Document: ~4 A peak; tolerance ±10 %
    assert_allclose(
        ia_peak,
        4.04,
        rtol=0.10,
        atol=0.1,
        err_msg="No-load peak current outside expected range (~4 A)",
    )


def test_no_load_speed(results_no_load):
    """No-load rotor speed must be approximately 1799 RPM."""
    ss = _steady_state_slice(results_no_load)
    speed_rpm = np.mean(results_no_load.speed[ss:]) * 60.0 / (2.0 * np.pi)
    # Document: 1799 RPM; tolerance ±5 RPM
    assert_allclose(
        speed_rpm,
        1800.0,
        atol=5.0,
        err_msg="No-load speed outside expected range (~1799 RPM)",
    )


@pytest.fixture
def results_nominal_load(motor):
    """Simulate the motor at nominal load for 3 s (nominal voltage)."""
    dt = 1e-4
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_direct_on_line(
        t,
        load_torque_entrance_time=0.5,
        load_torque_ratio=1.0,
    )


def test_nominal_load_stator_current_rms(results_nominal_load):
    """Nominal-load stator current (RMS) must be approximately 4.25 A."""
    ss = _steady_state_slice(results_nominal_load)
    ia_rms = rms(results_nominal_load.currents["a"][ss:])
    # Document: ~4.25 A rms; tolerance ±10 %
    assert_allclose(
        ia_rms,
        4.38,
        rtol=0.10,
        atol=0.15,
        err_msg="Nominal-load RMS current outside expected range (~4.25 A)",
    )


def test_nominal_load_stator_current_peak(results_nominal_load):
    """Nominal-load stator current (peak) must be approximately 6.0 A."""
    ss = _steady_state_slice(results_nominal_load)
    ia_peak = np.max(np.abs(results_nominal_load.currents["a"][ss:]))
    # Document: ~6.0 A peak; tolerance ±10 %
    assert_allclose(
        ia_peak,
        6.19,
        rtol=0.10,
        atol=0.2,
        err_msg="Nominal-load peak current outside expected range (~6.0 A)",
    )


def test_nominal_load_speed(results_nominal_load):
    """Nominal-load rotor speed must be approximately 1710 RPM."""
    ss = _steady_state_slice(results_nominal_load)
    speed_rpm = np.mean(results_nominal_load.speed[ss:]) * 60.0 / (2.0 * np.pi)
    # Document: 1710 RPM; tolerance ±15 RPM
    assert_allclose(
        speed_rpm,
        1710.0,
        atol=15.0,
        err_msg="Nominal-load speed outside expected range (~1710 RPM)",
    )


@pytest.fixture
def results_locked_rotor(motor_high_inertia):
    """Simulate starting current with very high inertia (rotor approximately locked).

    With Ip × 10 000, the speed barely changes; after the initial transient
    settles the electrical signals represent the blocked-rotor condition.
    """
    dt = 1e-4
    tf = 0.5
    t = np.arange(0, tf + dt, dt)
    return motor_high_inertia.run_direct_on_line(
        t,
        load_torque_entrance_time=0.0,
        load_torque_ratio=1.0,
    )


def test_starting_current_rms(results_locked_rotor):
    """Starting (locked-rotor) current RMS must be approximately 25 A."""
    ss = int(0.5 * len(results_locked_rotor.t))
    ia_rms = rms(results_locked_rotor.currents["a"][ss:])
    # Document: ~25 A rms; tolerance ±10 %
    assert_allclose(
        ia_rms,
        25.64,
        rtol=0.10,
        atol=0.5,
        err_msg="Starting RMS current outside expected range (~25 A)",
    )


def test_starting_current_peak(results_locked_rotor):
    """Starting (locked-rotor) current peak must be approximately 35.25 A."""
    ss = int(0.5 * len(results_locked_rotor.t))
    ia_peak = np.max(np.abs(results_locked_rotor.currents["a"][ss:]))
    # Document: ~35.25 A peak; tolerance ±10 %
    assert_allclose(
        ia_peak,
        36.26,
        rtol=0.10,
        atol=0.5,
        err_msg="Starting peak current outside expected range (~35.25 A)",
    )


def test_starting_electromagnetic_torque(results_locked_rotor):
    """Starting electromagnetic torque must be approximately 17.7 N·m."""
    ss = int(0.5 * len(results_locked_rotor.t))
    te_mean = np.mean(np.abs(results_locked_rotor.electric_torque[ss:]))
    # Document: ~17.7 N.m; tolerance ±10 %
    assert_allclose(
        te_mean,
        17.70,
        rtol=0.10,
        atol=0.3,
        err_msg="Starting electromagnetic torque outside expected range (~17.7 N·m)",
    )


def test_speed_nearly_zero_during_locked_rotor(results_locked_rotor):
    """With Ip × 10 000 the rotor speed must remain near zero throughout."""
    speed_rpm = results_locked_rotor.speed * 60.0 / (2.0 * np.pi)
    # Speed should not exceed 5 RPM during the 0.5 s window
    assert np.max(np.abs(speed_rpm)) < 5.0, (
        f"Speed too high during locked-rotor test: {np.max(np.abs(speed_rpm)):.2f} RPM"
    )


# ---------------------------------------------------------------------------
# Test 4 - Variable frequency drive (InverterVF / V-f scalar control)
# ---------------------------------------------------------------------------
#
# Unlike ``SourceAC``, the mechanical operating point reached with
# ``InverterVF`` is not compared against pre-recorded magic numbers (the
# SVPWM ripple shifts RMS/peak current readings a little with respect to an
# ideal sinusoidal source). Instead, these tests rely on the underlying
# physics of the induction machine, which hold regardless of the harmonic
# content injected by the modulator:
#
# - With no mechanical load and no viscous friction (``motor_example()`` has
#   ``viscosity_coeff=0``), the steady-state electric torque must converge to
#   (approximately) zero, so the rotor settles at the synchronous mechanical
#   speed set by the applied electrical frequency: ``w_sync = frequency / (n_poles/2)``.
# - Since ``InverterVF`` applies scalar V/f control, this holds at any
#   reference frequency within the linear modulation region, not only at the
#   nominal frequency.


def _synchronous_speed_rpm(motor, frequency_hz):
    """Synchronous mechanical speed [RPM] for a given electrical frequency [Hz]."""
    frequency_rad_s = Q_(frequency_hz, "Hz").to("rad/s").m
    w_sync = frequency_rad_s / (motor.n_poles / 2)
    return w_sync * 60.0 / (2.0 * np.pi)


@pytest.fixture(scope="module")
def results_inverter_vf_nominal_no_load(motor):
    """Simulate the motor with InverterVF at nominal frequency, no load."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_open_loop_vf_adjustment(
        t,
        time_step=1e-5,
        load_torque_entrance_time=tf + 1.0,  # load applied after simulation ends
        load_torque_ratio=0.0,
        time_ramp=0.5,
        frequency_ref=Q_(60.0, "Hz"),
    )


def test_inverter_vf_nominal_no_load_speed_near_synchronous(
    results_inverter_vf_nominal_no_load, motor
):
    """At nominal frequency and no load, speed must approach the 1800 RPM
    synchronous speed (60 Hz, 4 poles)."""
    ss = _steady_state_slice(results_inverter_vf_nominal_no_load)
    speed_rpm = np.mean(results_inverter_vf_nominal_no_load.speed[ss:]) * 60.0 / (
        2.0 * np.pi
    )
    w_sync_rpm = _synchronous_speed_rpm(motor, 60.0)
    assert_allclose(
        speed_rpm,
        w_sync_rpm,
        atol=15.0,
        err_msg="No-load InverterVF speed should approach the synchronous speed",
    )


@pytest.fixture(scope="module")
def results_inverter_vf_half_freq_no_load(motor):
    """Simulate the motor with InverterVF at half the nominal frequency, no load."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_open_loop_vf_adjustment(
        t,
        time_step=1e-5,
        load_torque_entrance_time=tf + 1.0,
        load_torque_ratio=0.0,
        time_ramp=0.5,
        frequency_ref=Q_(30.0, "Hz"),
    )


def test_inverter_vf_half_freq_no_load_speed_near_synchronous(
    results_inverter_vf_half_freq_no_load, motor
):
    """At half the nominal frequency (30 Hz) and no load, speed must approach
    half the synchronous speed (900 RPM), illustrating the V/f scaling law."""
    ss = _steady_state_slice(results_inverter_vf_half_freq_no_load)
    speed_rpm = np.mean(
        results_inverter_vf_half_freq_no_load.speed[ss:]
    ) * 60.0 / (2.0 * np.pi)
    w_sync_rpm = _synchronous_speed_rpm(motor, 30.0)
    assert_allclose(
        speed_rpm,
        w_sync_rpm,
        atol=15.0,
        err_msg="No-load InverterVF speed at half frequency should approach half the synchronous speed",
    )


@pytest.fixture(scope="module")
def results_inverter_vf_nominal_load(motor):
    """Simulate the motor with InverterVF at nominal frequency and nominal load."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_open_loop_vf_adjustment(
        t,
        time_step=1e-5,
        load_torque_entrance_time=0.5,
        load_torque_ratio=1.0,
        time_ramp=0.5,
        frequency_ref=Q_(60.0, "Hz"),
    )


def test_inverter_vf_nominal_load_causes_speed_droop(
    results_inverter_vf_nominal_load, results_inverter_vf_nominal_no_load
):
    """Under open-loop V/f control, applying the nominal load must reduce the
    steady-state speed with respect to the no-load operating point (slip)."""
    ss_load = _steady_state_slice(results_inverter_vf_nominal_load)
    ss_noload = _steady_state_slice(results_inverter_vf_nominal_no_load)

    speed_load_rpm = np.mean(results_inverter_vf_nominal_load.speed[ss_load:]) * 60.0 / (
        2.0 * np.pi
    )
    speed_noload_rpm = np.mean(
        results_inverter_vf_nominal_no_load.speed[ss_noload:]
    ) * 60.0 / (2.0 * np.pi)

    assert speed_load_rpm < speed_noload_rpm, (
        "Nominal-load speed should be lower than no-load speed due to slip "
        f"(load={speed_load_rpm:.1f} RPM, no-load={speed_noload_rpm:.1f} RPM)"
    )


# ---------------------------------------------------------------------------
# Test 5 - Indirect Field-Oriented Control (InverterFOC), closed loop
# ---------------------------------------------------------------------------
#
# Unlike InverterVF, InverterFOC regulates the shaft speed in closed loop
# through nested speed/current PI controllers. Given enough time to settle,
# the integral action of the speed loop is expected to drive the steady-state
# speed error to (approximately) zero, for any speed reference within the
# machine's capability and regardless of load torque - which is the key
# functional difference with respect to the open-loop V/f drive tested above.


@pytest.fixture(scope="module")
def results_foc_nominal_speed_with_load(motor):
    """Simulate the motor with InverterFOC tracking nominal speed, under
    nominal load."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_with_inverter_foc(
        t,
        time_step=1e-5,
        load_torque_entrance_time=1.5,
        load_torque_ratio=1.0,
        frequency_s=Q_(5000, "Hz"),
        time_ramp=0.5,
        wref=motor.speed_nom,
    )


def test_foc_speed_tracks_nominal_reference_despite_load(
    results_foc_nominal_speed_with_load, motor
):
    """Steady-state speed must track wref (nominal speed) closely, even after
    the nominal load torque is applied - contrasting the V/f speed droop."""
    ss = _steady_state_slice(results_foc_nominal_speed_with_load)
    speed_rpm = np.mean(
        results_foc_nominal_speed_with_load.speed[ss:]
    ) * 60.0 / (2.0 * np.pi)
    wref_rpm = motor.speed_nom * 60.0 / (2.0 * np.pi)

    assert_allclose(
        speed_rpm,
        wref_rpm,
        rtol=0.03,
        atol=15.0,
        err_msg="Closed-loop FOC speed should track the nominal speed reference under load",
    )


@pytest.fixture(scope="module")
def results_foc_half_speed_no_load(motor):
    """Simulate the motor with InverterFOC tracking half the nominal speed,
    with no load."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    wref_half = motor.speed_nom / 2.0
    return motor.run_with_inverter_foc(
        t,
        time_step=1e-5,
        load_torque_entrance_time=tf + 1.0,  # load applied after simulation ends
        load_torque_ratio=0.0,
        frequency_s=Q_(5000, "Hz"),
        time_ramp=0.5,
        wref=wref_half,
    )


def test_foc_speed_tracks_reduced_reference_no_load(
    results_foc_half_speed_no_load, motor
):
    """Steady-state speed must track an arbitrary (non-nominal) speed
    reference, unlike open-loop V/f control which only tracks the
    synchronous speed derived from the applied electrical frequency."""
    ss = _steady_state_slice(results_foc_half_speed_no_load)
    speed_rpm = np.mean(results_foc_half_speed_no_load.speed[ss:]) * 60.0 / (
        2.0 * np.pi
    )
    wref_rpm = (motor.speed_nom / 2.0) * 60.0 / (2.0 * np.pi)

    assert_allclose(
        speed_rpm,
        wref_rpm,
        rtol=0.03,
        atol=15.0,
        err_msg="Closed-loop FOC speed should track a reduced speed reference",
    )


def test_foc_speed_ramps_up_gradually(results_foc_half_speed_no_load):
    """The mechanical speed reference is ramped, so the shaft speed early in
    the simulation must be substantially lower than the final steady-state
    speed (no instantaneous jump to the reference)."""
    results = results_foc_half_speed_no_load
    idx_early = np.searchsorted(results.t, 0.1)
    speed_early_rpm = results.speed[idx_early] * 60.0 / (2.0 * np.pi)

    ss = _steady_state_slice(results)
    speed_final_rpm = np.mean(results.speed[ss:]) * 60.0 / (2.0 * np.pi)

    assert speed_early_rpm < 0.5 * speed_final_rpm, (
        "Speed should still be ramping up at t=0.1 s, well below the final "
        f"steady-state value (early={speed_early_rpm:.1f} RPM, "
        f"final={speed_final_rpm:.1f} RPM)"
    )


def test_foc_closed_loop_tracks_better_than_open_loop_under_load(
    results_foc_nominal_speed_with_load, results_inverter_vf_nominal_load, motor
):
    """Under nominal load, the closed-loop FOC speed error with respect to
    its reference must be smaller than the open-loop V/f slip-induced error,
    highlighting the benefit of closed-loop control."""
    ss = _steady_state_slice(results_foc_nominal_speed_with_load)
    foc_speed_rpm = np.mean(
        results_foc_nominal_speed_with_load.speed[ss:]
    ) * 60.0 / (2.0 * np.pi)
    wref_rpm = motor.speed_nom * 60.0 / (2.0 * np.pi)
    foc_error = abs(foc_speed_rpm - wref_rpm)

    ss_vf = _steady_state_slice(results_inverter_vf_nominal_load)
    vf_speed_rpm = np.mean(
        results_inverter_vf_nominal_load.speed[ss_vf:]
    ) * 60.0 / (2.0 * np.pi)
    vf_sync_rpm = _synchronous_speed_rpm(motor, 60.0)
    vf_error = abs(vf_speed_rpm - vf_sync_rpm)

    assert foc_error < vf_error, (
        "Closed-loop FOC speed error should be smaller than the open-loop "
        f"V/f slip (foc_error={foc_error:.2f} RPM, vf_error={vf_error:.2f} RPM)"
    )
