"""Tests for the MotorElement class.

Tests are based on the operating scenarios described in
"Testes dos modelos do motor e conversor - ROSS.docx", covering the three
electrical sources supported by ``MotorElement``:

- ``SourceAC`` (ideal AC source), driven through ``.run_direct_on_line()``;
- ``InverterVF`` (open-loop scalar V/f control), driven through
  ``.run_with_inverter_vf()``;
- ``InverterFOC`` (closed-loop indirect Field-Oriented Control), driven
  through ``.run_with_inverter_foc()``.

All tests use the parameters from ``motor_example()`` and the default
simulation parameters defined in the module.

Motor under test
----------------
- Rated power  : 1.5 cv  (≈ 1103.25 W)
- Rated voltage: 127 V (phase)
- Rated speed  : 1710 RPM
- Rated frequency: 60 Hz
- Poles          : 4
- Rs = 2.5 Ω, Rr = 1.8 Ω, Xs = Xr = 1.3 Ω, Xm = 43.08 Ω
- Ip_motor = 0.0372 kg·m²
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from copy import deepcopy
from scipy.integrate import quad
from scipy.signal import lsim

from ross.bearing_seal_element import BearingElement
from ross.coupling_element import CouplingElement
from ross.materials import steel
from ross.motors.motor_element import MotorElement, motor_example
from ross.motors.results import MotorResponseResults
from ross.multi_rotor.multi_rotor import two_shaft_rotor_example
from ross.rotor_assembly import Rotor, rotor_example
from ross.shaft_element import ShaftElement
from ross.units import Q_


@pytest.fixture(scope="module")
def motor():
    """Return the motor from motor_example()."""
    return motor_example()


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
        power_rated=Q_(1.5, "cv"),
        voltage_rated=127,
        speed_rated=Q_(1710, "RPM"),
        frequency_rated=Q_(60.0, "Hz"),
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


def test_motor_example_parameters():
    """Verify that motor_example() returns the expected rated parameters."""
    motor = motor_example()
    assert_allclose(motor.power_rated, 1103.248125, rtol=1e-6)
    assert_allclose(motor.voltage_rated, 127.0, rtol=1e-6)
    assert_allclose(
        motor.speed_rated,
        Q_(1710, "RPM").to("rad/s").m,
        rtol=1e-5,
    )
    assert_allclose(
        motor.frequency_rated,
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
    """Two calls to motor_example() must return equal objects, even if the
    tags differ."""
    m1 = motor_example()
    m2 = motor_example()
    m2.tag = "motor_2"
    assert m1 == m2


@pytest.fixture
def results_no_load(motor):
    """Simulate the motor at no load for 3 s (rated voltage)."""
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
def results_rated_load(motor):
    """Simulate the motor at rated load for 3 s (rated voltage)."""
    dt = 1e-4
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_direct_on_line(
        t,
        load_torque_entrance_time=0.5,
        load_torque_ratio=1.0,
    )


def test_rated_load_stator_current_rms(results_rated_load):
    """Rated-load stator current (RMS) must be approximately 4.25 A."""
    ss = _steady_state_slice(results_rated_load)
    ia_rms = rms(results_rated_load.currents["a"][ss:])
    # Document: ~4.25 A rms; tolerance ±10 %
    assert_allclose(
        ia_rms,
        4.38,
        rtol=0.10,
        atol=0.15,
        err_msg="Rated-load RMS current outside expected range (~4.25 A)",
    )


def test_rated_load_speed(results_rated_load):
    """Rated-load rotor speed must be approximately 1710 RPM."""
    ss = _steady_state_slice(results_rated_load)
    speed_rpm = np.mean(results_rated_load.speed[ss:]) * 60.0 / (2.0 * np.pi)
    # Document: 1710 RPM; tolerance ±15 RPM
    assert_allclose(
        speed_rpm,
        1710.0,
        atol=15.0,
        err_msg="Rated-load speed outside expected range (~1710 RPM)",
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
#   rated frequency.


def _synchronous_speed_rpm(motor, frequency_hz):
    """Synchronous mechanical speed [RPM] for a given electrical frequency [Hz]."""
    frequency_rad_s = Q_(frequency_hz, "Hz").to("rad/s").m
    w_sync = frequency_rad_s / (motor.n_poles / 2)
    return w_sync * 60.0 / (2.0 * np.pi)


@pytest.fixture(scope="module")
def results_inverter_vf_rated_no_load(motor):
    """Simulate the motor with InverterVF at rated frequency, no load."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_with_inverter_vf(
        t,
        frequency_s=Q_(5000, "Hz"),
        time_step=1e-5,
        load_torque_entrance_time=tf + 1.0,  # load applied after simulation ends
        load_torque_ratio=0.0,
        time_ramp=0.5,
        frequency_ref=Q_(60.0, "Hz"),
    )


def test_inverter_vf_rated_no_load_speed_near_synchronous(
    results_inverter_vf_rated_no_load, motor
):
    """At rated frequency and no load, speed must approach the 1800 RPM
    synchronous speed (60 Hz, 4 poles)."""
    ss = _steady_state_slice(results_inverter_vf_rated_no_load)
    speed_rpm = (
        np.mean(results_inverter_vf_rated_no_load.speed[ss:]) * 60.0 / (2.0 * np.pi)
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
    """Simulate the motor with InverterVF at half the rated frequency, no load."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_with_inverter_vf(
        t,
        frequency_s=Q_(5000, "Hz"),
        time_step=1e-5,
        load_torque_entrance_time=tf + 1.0,
        load_torque_ratio=0.0,
        time_ramp=0.5,
        frequency_ref=Q_(30.0, "Hz"),
    )


def test_inverter_vf_half_freq_no_load_speed_near_synchronous(
    results_inverter_vf_half_freq_no_load, motor
):
    """At half the rated frequency (30 Hz) and no load, speed must approach
    half the synchronous speed (900 RPM), illustrating the V/f scaling law."""
    ss = _steady_state_slice(results_inverter_vf_half_freq_no_load)
    speed_rpm = (
        np.mean(results_inverter_vf_half_freq_no_load.speed[ss:]) * 60.0 / (2.0 * np.pi)
    )
    w_sync_rpm = _synchronous_speed_rpm(motor, 30.0)
    assert_allclose(
        speed_rpm,
        w_sync_rpm,
        atol=15.0,
        err_msg="No-load InverterVF speed at half frequency should approach half the synchronous speed",
    )


@pytest.fixture(scope="module")
def results_inverter_vf_rated_load(motor):
    """Simulate the motor with InverterVF at rated frequency and rated load."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_with_inverter_vf(
        t,
        frequency_s=Q_(5000, "Hz"),
        time_step=1e-5,
        load_torque_entrance_time=0.5,
        load_torque_ratio=1.0,
        time_ramp=0.5,
        frequency_ref=Q_(60.0, "Hz"),
    )


def test_inverter_vf_rated_load_causes_speed_droop(
    results_inverter_vf_rated_load, results_inverter_vf_rated_no_load
):
    """Under open-loop V/f control, applying the rated load must reduce the
    steady-state speed with respect to the no-load operating point (slip)."""
    ss_load = _steady_state_slice(results_inverter_vf_rated_load)
    ss_noload = _steady_state_slice(results_inverter_vf_rated_no_load)

    speed_load_rpm = (
        np.mean(results_inverter_vf_rated_load.speed[ss_load:]) * 60.0 / (2.0 * np.pi)
    )
    speed_noload_rpm = (
        np.mean(results_inverter_vf_rated_no_load.speed[ss_noload:])
        * 60.0
        / (2.0 * np.pi)
    )

    assert speed_load_rpm < speed_noload_rpm, (
        "Rated-load speed should be lower than no-load speed due to slip "
        f"(load={speed_load_rpm:.1f} RPM, no-load={speed_noload_rpm:.1f} RPM)"
    )


# ---------------------------------------------------------------------------
# Test 5 - Indirect Field-Oriented Control (InverterFOC), closed loop
# ---------------------------------------------------------------------------
#
# InverterFOC takes a synchronous electrical `frequency_ref` argument, just
# like InverterVF (e.g. `Q_(60, "Hz")`), but - unlike InverterVF - it is a
# closed loop: the speed loop converts `frequency_ref` internally to the
# equivalent synchronous mechanical speed and actively corrects for slip, so
# the steady-state speed error is expected to be (approximately) zero for any
# reference within the machine's capability and regardless of load torque.
# This is the key functional difference with respect to the open-loop V/f
# drive tested above, whose steady-state speed merely approaches the
# synchronous speed at no load and droops under load.
#
# A reference at or above the rated frequency is clamped internally to the
# motor's rated mechanical speed (`InverterFOC.wn`), so
# `frequency_ref=Q_(60, "Hz")` targets the same ~1710 RPM rated operating
# point used in the SourceAC/InverterVF rated-load tests above, while a
# reduced reference (e.g. 30 Hz) targets the corresponding synchronous speed
# directly (900 RPM), exactly as `_synchronous_speed_rpm` computes for
# InverterVF.


@pytest.fixture(scope="module")
def results_foc_rated_speed_with_load(motor):
    """Simulate the motor with InverterFOC tracking the rated frequency,
    under rated load."""
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
        frequency_ref=Q_(60.0, "Hz"),
    )


def test_foc_speed_tracks_rated_reference_despite_load(
    results_foc_rated_speed_with_load, motor
):
    """Steady-state speed must track the rated speed closely, even
    after the rated load torque is applied - contrasting the V/f speed
    droop."""
    ss = _steady_state_slice(results_foc_rated_speed_with_load)
    speed_rpm = (
        np.mean(results_foc_rated_speed_with_load.speed[ss:]) * 60.0 / (2.0 * np.pi)
    )
    wref_rpm = motor.speed_rated * 60.0 / (2.0 * np.pi)

    assert_allclose(
        speed_rpm,
        wref_rpm,
        rtol=0.03,
        atol=15.0,
        err_msg="Closed-loop FOC speed should track the rated speed reference under load",
    )


@pytest.fixture(scope="module")
def results_foc_half_freq_no_load(motor):
    """Simulate the motor with InverterFOC tracking half the rated
    frequency, with no load."""
    dt = 1e-3
    tf = 3.0
    t = np.arange(0, tf + dt, dt)
    return motor.run_with_inverter_foc(
        t,
        time_step=1e-5,
        load_torque_entrance_time=tf + 1.0,  # load applied after simulation ends
        load_torque_ratio=0.0,
        frequency_s=Q_(5000, "Hz"),
        time_ramp=0.5,
        frequency_ref=Q_(30.0, "Hz"),
    )


def test_foc_speed_tracks_reduced_reference_no_load(
    results_foc_half_freq_no_load, motor
):
    """Steady-state speed must track the synchronous speed of a reduced
    (non-rated) frequency reference, correcting for slip - unlike open-loop
    V/f control, which only approaches that synchronous speed at no load."""
    ss = _steady_state_slice(results_foc_half_freq_no_load)
    speed_rpm = np.mean(results_foc_half_freq_no_load.speed[ss:]) * 60.0 / (2.0 * np.pi)
    w_sync_rpm = _synchronous_speed_rpm(motor, 30.0)

    assert_allclose(
        speed_rpm,
        w_sync_rpm,
        rtol=0.03,
        atol=15.0,
        err_msg="Closed-loop FOC speed should track a reduced frequency reference",
    )


# ---------------------------------------------------------------------------
# Test 6 - FFT frequency range for inverter-driven figures
# ---------------------------------------------------------------------------
#
# Restricting the displayed band via `frequency_range` is checked on the two
# independent implementations: `MotorResponseResults._plot_dfft` (`plot_torque`)
# and `PhaseResults.plot_dfft` (`plot_phase_currents`).

_FS_HZ = 5000.0
_FFT_FREQUENCY_RANGE = Q_((0.5, 2.1 * _FS_HZ), "Hz")


def test_inverter_vf_fft_frequency_range_narrows_torque_spectrum(
    results_inverter_vf_rated_no_load,
):
    """Restricting `plot_torque(domain="frequency")` to [0.5 Hz, 2.1 x Fs]
    must narrow the displayed frequency span with respect to the
    unrestricted spectrum (which extends all the way to the Nyquist
    frequency) and must not extend far beyond the requested upper bound."""
    results = results_inverter_vf_rated_no_load

    fig_full = results.plot_torque(domain="frequency")
    fig_restricted = results.plot_torque(
        domain="frequency", frequency_range=_FFT_FREQUENCY_RANGE
    )

    x_full = np.asarray(fig_full.data[0].x)
    x_restricted = np.asarray(fig_restricted.data[0].x)

    assert x_restricted.max() < x_full.max(), (
        "Restricting frequency_range should narrow the displayed frequency span "
        f"(restricted max={x_restricted.max():.1f} Hz, full max={x_full.max():.1f} Hz)"
    )
    assert x_restricted.max() <= 2.1 * _FS_HZ * 1.02, (
        "Restricted torque FFT should not extend much beyond 2.1 x Fs "
        f"(got max={x_restricted.max():.1f} Hz, limit={2.1 * _FS_HZ:.1f} Hz)"
    )


def test_inverter_vf_fft_frequency_range_narrows_current_spectrum(
    results_inverter_vf_rated_no_load,
):
    """Same as above, for `plot_phase_currents(domain="frequency")`, which
    goes through the separate `PhaseResults.plot_dfft` implementation."""
    results = results_inverter_vf_rated_no_load

    fig_full = results.plot_phase_currents(domain="frequency")
    fig_restricted = results.plot_phase_currents(
        domain="frequency", frequency_range=_FFT_FREQUENCY_RANGE
    )

    x_full = np.asarray(fig_full.data[0].x)
    x_restricted = np.asarray(fig_restricted.data[0].x)

    assert x_restricted.max() < x_full.max(), (
        "Restricting frequency_range should narrow the displayed frequency span "
        f"(restricted max={x_restricted.max():.1f} Hz, full max={x_full.max():.1f} Hz)"
    )
    assert x_restricted.max() <= 2.1 * _FS_HZ * 1.02, (
        "Restricted current FFT should not extend much beyond 2.1 x Fs "
        f"(got max={x_restricted.max():.1f} Hz, limit={2.1 * _FS_HZ:.1f} Hz)"
    )


# ---------------------------------------------------------------------------
# Test 7 - Motor coupled to a rotor (Rotor.run_with_motor)
# ---------------------------------------------------------------------------
#
# The rotor uses the shaft and disks of ``rotor_example()`` with damped
# bearings: with the undamped bearings of the example, the free vibration
# excited at the start of the steady-state window never decays and dominates
# the spectrum instead of the unbalance response. All three ``drive_mode``
# values are run at 60 Hz under rated load, so the shaft settles near the
# rated speed and the unbalance response is a 1X well above the first modes.
# The rotor adds 0.67 kg.m² of polar inertia to the 0.037 kg.m² of the motor,
# so the start-up takes about 8 s. The rated load is applied at 7.5 s, before
# the FOC drive reaches its speed reference, and the simulation runs to 12 s,
# so the steady-state window (10.5 s to 12 s) is past the load transient.

_ROTOR_LOAD_TIME = 7.5
_ROTOR_END_TIME = 12.0
_ROTOR_UNBALANCE_NODE = 2
_DRIVE_MODES = ("DOL", "VFD_VF", "VFD_FOC")


@pytest.fixture(scope="module")
def rotor_with_motor(motor):
    """Return a damped rotor with the motor at node 0."""
    base_rotor = rotor_example()
    bearings = [
        BearingElement(n=bearing.n, kxx=1e6, cxx=2e3)
        for bearing in base_rotor.bearing_elements
    ]
    return Rotor(
        base_rotor.shaft_elements,
        base_rotor.disk_elements,
        bearings,
        motor_element=deepcopy(motor),
    )


@pytest.fixture(scope="module", params=_DRIVE_MODES)
def results_run_with_motor(rotor_with_motor, request):
    """Simulate the rotor with each supported drive_mode under rated load."""
    drive_mode = request.param
    t = np.arange(0, _ROTOR_END_TIME + 1e-3, 1e-3)
    kwargs = {}
    if drive_mode != "DOL":
        kwargs["time_step"] = 1e-5
        kwargs["time_ramp"] = 0.5
        kwargs["frequency_ref"] = Q_(60.0, "Hz")

    return rotor_with_motor.run_with_motor(
        t=t,
        node=[_ROTOR_UNBALANCE_NODE],
        unbalance_magnitude=[5e-4],
        unbalance_phase=[0.0],
        drive_mode=drive_mode,
        load_torque_entrance_time=_ROTOR_LOAD_TIME,
        load_torque_ratio=1.0,
        **kwargs,
    )


def test_run_with_motor(results_run_with_motor, rotor_with_motor):
    """Each drive_mode must attach motor results, settle near the rated speed
    under rated load, and produce a synchronous (1X) unbalance response."""
    results = results_run_with_motor
    assert isinstance(results.motor_results, MotorResponseResults)

    speed = results.motor_results.sample_at("speed", results.t)
    speed_rpm = np.mean(speed) * 60.0 / (2.0 * np.pi)
    assert_allclose(
        speed_rpm,
        1710.0,
        atol=15.0,
        err_msg="Shaft speed under rated load should be close to the rated speed",
    )

    dof_x = rotor_with_motor.number_dof * _ROTOR_UNBALANCE_NODE
    x = results.yout[:, dof_x]
    spectrum = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x))))
    frequencies = np.fft.rfftfreq(len(x), results.t[1] - results.t[0])
    peak_frequency = frequencies[np.argmax(spectrum)]
    synchronous_frequency = np.mean(speed) / (2.0 * np.pi)

    assert np.max(np.abs(x)) > 0.0
    assert_allclose(
        peak_frequency,
        synchronous_frequency,
        atol=frequencies[1],
        err_msg="Rotor response should be dominated by the 1X component",
    )


def test_run_with_motor_invalid_drive_mode(rotor_with_motor):
    """An unknown drive_mode must raise a ValueError."""
    t = np.arange(0, 0.1, 1e-3)
    with pytest.raises(ValueError, match="drive_mode"):
        rotor_with_motor.run_with_motor(
            t=t,
            node=[_ROTOR_UNBALANCE_NODE],
            unbalance_magnitude=[5e-4],
            unbalance_phase=[0.0],
            drive_mode="invalid",
            load_torque_entrance_time=0.05,
        )


def test_run_with_motor_without_motor_element():
    """A rotor without a motor element must raise a ValueError."""
    t = np.arange(0, 0.1, 1e-3)
    with pytest.raises(ValueError, match="No motor elements"):
        rotor_example().run_with_motor(
            t=t,
            node=[_ROTOR_UNBALANCE_NODE],
            unbalance_magnitude=[5e-4],
            unbalance_phase=[0.0],
            drive_mode="DOL",
            load_torque_entrance_time=0.05,
        )


def _rotor_example_polar_inertia():
    """Return the polar moment of inertia of ``rotor_example()`` [kg.m²].

    Every part is a steel hollow cylinder, Ip = rho * pi / 32 * (D⁴ - d⁴) * L:
    six 0.25 m shaft elements with D = 0.05 m, and two disks of width 0.07 m
    with d = 0.05 m and D = 0.28 m.
    """
    rho = 7810.0
    shaft = rho * np.pi / 32 * 0.05**4 * 6 * 0.25
    disks = 2 * rho * np.pi / 32 * (0.28**4 - 0.05**4) * 0.07
    return shaft + disks


def _equivalent_circuit_torque(motor, speed):
    """Return the steady-state torque of the per-phase equivalent circuit [N.m].

    Z = Rs + jXs + jXm || (Rr / s + jXr) and Te = 3 |Ir|² (Rr / s) / ws, with
    the network resistance added to the stator resistance, as in
    ``MotorElement.run``.
    """
    synchronous_speed = motor.frequency_rated / (motor.n_poles / 2)
    slip = 1 - speed / synchronous_speed
    Rs = motor.stator_resistance + motor.short_circuit_resistance
    Zm = 1j * motor.mutual_reactance
    Zr = motor.rotor_resistance / slip + 1j * motor.rotor_reactance
    stator_current = motor.voltage_rated / (
        Rs + 1j * motor.stator_reactance + Zm * Zr / (Zm + Zr)
    )
    rotor_current = stator_current * Zm / (Zm + Zr)
    return (
        3 * np.abs(rotor_current) ** 2 * motor.rotor_resistance / slip
    ) / synchronous_speed


def test_run_with_motor_speed_includes_rotor_inertia(rotor_with_motor):
    """The motor must accelerate the rotor inertia, not only its own."""
    motor = rotor_with_motor.motor_element
    rotor_inertia = _rotor_example_polar_inertia()
    t = np.arange(0, 4.0 + 5e-4, 1e-3)
    results = rotor_with_motor.run_with_motor(
        t=t,
        node=[_ROTOR_UNBALANCE_NODE],
        unbalance_magnitude=[5e-4],
        unbalance_phase=[0.0],
        drive_mode="DOL",
        load_torque_entrance_time=3.7,
        load_torque_ratio=0.0,
    )
    speed = results.motor_results.speed

    # Reference: the motor alone, with the rotor inertia from geometry as
    # Ip_load. Same model and time grid, so only round-off separates them.
    reference_motor = deepcopy(motor)
    reference_motor.Ip_load += rotor_inertia
    reference = reference_motor.run_direct_on_line(
        t, load_torque_entrance_time=3.7, load_torque_ratio=0.0
    )
    assert_allclose(speed, reference.speed, rtol=1e-9, atol=1e-9)

    # Reference: quasi-static start-up, t = J ∫ dw / Te(w), with Te from the
    # steady-state equivalent circuit. The electrical transients last about
    # 0.1 s against a start-up of about 3.4 s, hence the 1 % tolerance.
    half_speed = 0.5 * motor.frequency_rated / (motor.n_poles / 2)
    inertia = motor.Ip_motor + motor.Ip_load + rotor_inertia
    expected_time = (
        inertia
        * quad(lambda w: 1 / _equivalent_circuit_torque(motor, w), 0, half_speed)[0]
    )
    time_to_half_speed = results.motor_results.t[np.argmax(speed >= half_speed)]
    assert_allclose(time_to_half_speed, expected_time, rtol=0.01)


@pytest.fixture(scope="module")
def results_motor_start_up(rotor_with_motor):
    """Simulate the first 0.2 s of a direct-on-line start, rated load at 0.1 s."""
    t = np.arange(0, 0.2 + 5e-5, 1e-4)
    return rotor_with_motor.run_with_motor(
        t=t,
        node=[_ROTOR_UNBALANCE_NODE],
        unbalance_magnitude=[5e-4],
        unbalance_phase=[0.0],
        drive_mode="DOL",
        load_torque_entrance_time=0.1,
        load_torque_ratio=1.0,
        steady_state=False,
    )


def _torsional_reference(rotor, t, torque):
    """Return the twist of a free torsional model driven at the motor node.

    The model uses absolute angles: the torsional rows of ``rotor.M()`` and
    ``rotor.K()``, with the motor and load inertias added at the motor node,
    and the torque applied there alone. It is written in angles relative to
    the motor node so that the rigid-body rotation, which grows to hundreds
    of radians, does not swamp the twist, and it is integrated exactly for a
    piecewise-linear torque with ``scipy.signal.lsim``. The returned angles
    have the mass-weighted mean rotation removed; the weights are returned
    with them.
    """
    motor = rotor.motor_element
    dofs = [rotor.number_dof * n + 5 for n in rotor.nodes]
    i = rotor.nodes.index(motor.n)
    M = rotor.M()[np.ix_(dofs, dofs)]
    M[i, i] += motor.Ip_motor + motor.Ip_load
    K = rotor.K(0)[np.ix_(dofs, dofs)]

    size = len(dofs)
    P = np.eye(size)
    P[:, i] = 1.0
    Mq = P.T @ M @ P
    Kq = P.T @ K @ P
    A = np.block(
        [
            [np.zeros((size, size)), np.eye(size)],
            [-np.linalg.solve(Mq, Kq), np.zeros((size, size))],
        ]
    )
    B = np.concatenate([np.zeros(size), np.linalg.solve(Mq, P.T[:, i])])[:, None]
    C = np.eye(2 * size)[:size]
    C[i, i] = 0.0
    _, relative, _ = lsim((A, B, C, np.zeros((size, 1))), torque, t)

    weights = M @ np.ones(size) / M.sum()
    return relative - (relative @ weights)[:, None], weights


def test_run_with_motor_torsional_twist(results_motor_start_up, rotor_with_motor):
    """Torsional DOFs must carry the twist only, matching a free-free model."""
    results = results_motor_start_up
    motor = rotor_with_motor.motor_element
    motor_results = results.motor_results
    speed = motor_results.sample_at("speed", results.t)
    torque = (
        motor_results.sample_at("electric_torque", results.t)
        - motor.viscosity_coeff * speed
        - motor_results.sample_at("load_torque", results.t)
    )
    expected, weights = _torsional_reference(rotor_with_motor, results.t, torque)
    dofs = [rotor_with_motor.number_dof * n + 5 for n in rotor_with_motor.nodes]
    theta = results.yout[:, dofs]

    # The torque on the free rotor sums to zero, so the mass-weighted mean
    # rotation stays at its initial value.
    assert_allclose(theta @ weights, 0.0, atol=1e-12)

    # Reference: absolute-angle torsional model above. Before the load step at
    # 0.1 s the response is dominated by the 119 Hz and 274 Hz modes, for
    # which the Newmark step of 1e-4 s gives a period error below
    # (w dt)² / 12 = 0.25 %, so the twist must match within 1 % of its peak.
    before = results.t < 0.1
    twist = theta[:, 0] - theta[:, -1]
    expected_twist = expected[:, 0] - expected[:, -1]
    assert_allclose(
        twist[before],
        expected_twist[before],
        atol=0.01 * np.max(np.abs(expected_twist)),
    )
    assert_allclose(
        theta[before], expected[before], atol=0.01 * np.max(np.abs(expected))
    )

    # The load step excites the kHz torsional modes, which the step does not
    # resolve; their contribution averages out, so after the step the mean
    # twist is compared.
    after = ~before
    assert_allclose(twist[after].mean(), expected_twist[after].mean(), rtol=0.005)


def test_run_with_motor_multi_rotor_rigid_body_mode():
    """In a geared train the torsional DOFs must also carry the twist only."""
    multi_rotor = two_shaft_rotor_example().add_elements([motor_example()])
    dofs = [multi_rotor.number_dof * n + 5 for n in multi_rotor.nodes]

    # Reference: a rigid rotation of the gear pair keeps the contact point
    # fixed, so the driven shaft turns by the ratio of the pitch diameters
    # (with the sign convention of the torsional DOFs of the mesh) and strains
    # neither the shafts nor the mesh.
    mesh = multi_rotor.mesh
    pitch_ratio = mesh.driving_gear.pitch_diameter / mesh.driven_gear.pitch_diameter
    expected_mode = np.zeros(multi_rotor.ndof)
    expected_mode[dofs] = [
        pitch_ratio if n in multi_rotor.driven_nodes else 1.0 for n in multi_rotor.nodes
    ]

    t = np.arange(0, 0.5 + 5e-4, 1e-3)
    results = multi_rotor.run_with_motor(
        t=t,
        node=[2],
        unbalance_magnitude=[0.0],
        unbalance_phase=[0.0],
        drive_mode="DOL",
        load_torque_entrance_time=0.25,
        load_torque_ratio=0.5,
        steady_state=False,
    )
    M = multi_rotor.M()
    M[dofs[0], dofs[0]] += multi_rotor.motor_element.Ip_motor
    rigid_rotation = (
        results.yout @ M @ expected_mode / (expected_mode @ M @ expected_mode)
    )

    # The torque on the free train sums to zero along the rigid-body mode, so
    # its rotation stays at round-off level.
    assert_allclose(rigid_rotation, 0.0, atol=1e-12)
    assert_allclose(multi_rotor._torsional_rigid_body_mode(0), expected_mode, atol=1e-9)


def test_run_with_motor_keeps_force_array(rotor_with_motor):
    """The force array passed by the user must not be modified."""
    t = np.arange(0, 0.05 + 5e-4, 1e-3)
    F = np.zeros((len(t), rotor_with_motor.ndof))
    rotor_with_motor.run_with_motor(
        t=t,
        node=[_ROTOR_UNBALANCE_NODE],
        unbalance_magnitude=[5e-4],
        unbalance_phase=[0.0],
        drive_mode="DOL",
        load_torque_entrance_time=0.025,
        steady_state=False,
        F=F,
    )
    assert not np.any(F)


def test_run_with_motor_disconnected_torsional_train(motor):
    """A coupling without torsional stiffness splits the train in two, so the
    motor speed does not define the rotation of the whole rotor."""
    shafts = [ShaftElement(L=0.25, idl=0, odl=0.05, material=steel) for _ in range(6)]
    coupling = CouplingElement(
        m_l=1.0,
        m_r=1.0,
        Ip_l=0.01,
        Ip_r=0.01,
        kt_x=1e7,
        kt_y=1e7,
        kr_x=1e5,
        kr_y=1e5,
    )
    rotor = Rotor(
        shafts[:3] + [coupling] + shafts[3:],
        bearing_elements=[
            BearingElement(n=0, kxx=1e6, cxx=2e3),
            BearingElement(n=7, kxx=1e6, cxx=2e3),
        ],
        motor_element=deepcopy(motor),
    )
    t = np.arange(0, 0.05 + 5e-4, 1e-3)
    with pytest.raises(ValueError, match="single train"):
        rotor.run_with_motor(
            t=t,
            node=[1],
            unbalance_magnitude=[5e-4],
            unbalance_phase=[0.0],
            drive_mode="DOL",
            load_torque_entrance_time=0.025,
            steady_state=False,
        )
