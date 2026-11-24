"""Tests for the ElectricMotor class.

Tests are based on the operating scenarios described in
"Testes dos modelos do motor e conversor - ROSS.docx", covering the three
electrical sources supported by ``ElectricMotor``:

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
- Ip = 0.0372 kg·m²
"""

import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from copy import deepcopy

from ross.bearing_seal_element import BearingElement
from ross.disk_element import DiskElement
from ross.motors.electric_motor import ElectricMotor, motor_example
from ross.motors.results import MotorResponseResults
from ross.materials import steel
from ross.rotor_assembly import Rotor, rotor_example
from ross.shaft_element import ShaftElement
from ross.units import Q_


@pytest.fixture(scope="module")
def motor():
    """Return the motor from motor_example()."""
    return motor_example()


@pytest.fixture
def motor_high_inertia():
    """Return the motor with Ip × 10 000 to simulate a locked rotor.

    With such a large inertia the mechanical speed barely changes during the
    simulation window, so the electrical quantities converge to the
    blocked-rotor (starting) operating point.
    """
    return ElectricMotor(
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
        m=10.0,
        Ip=0.0372 * 10000,
        viscosity_coeff=0.0,
        voltage_net=127,
    )


def rms(signal):
    """Return the RMS value of *signal*."""
    return np.sqrt(np.mean(signal**2))


def _steady_state_slice(results, fraction=0.85):
    """Return the index at which steady state is considered to begin."""
    return int(fraction * len(results.t))


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
# Test 7 - Motor coupled to a rotor (Rotor.run_with_motor)
# ---------------------------------------------------------------------------
#
# The rotor uses the shaft of ``rotor_example()`` with damped bearings and
# smaller disks, which give it a polar inertia of 0.07 kg.m^2, about twice the
# one of the motor. The motor simulation includes the inertia of the whole
# rotor: with the disks of the example (0.67 kg.m^2) the start would not end
# in a few seconds. The load enters at 1 s and the simulation lasts 3 s, so
# that the open-loop V/f speed and the ringing of the FOC speed loop have
# settled in the steady-state window. With the undamped bearings of the example,
# the free vibration excited at the start of the steady-state window never
# decays and dominates the spectrum instead of the unbalance response. All
# three ``drive_mode`` values are run at 60 Hz under rated load, so the shaft
# settles near the rated speed and the unbalance response is a 1X well above
# the first modes.

_ROTOR_LOAD_TIME = 1.0
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
    disks = [
        DiskElement.from_geometry(
            n=disk.n, material=steel, width=0.04, i_d=0.05, o_d=0.18
        )
        for disk in base_rotor.disk_elements
    ]
    rotor = Rotor(base_rotor.shaft_elements, disks, bearings)
    return rotor.add_motor(deepcopy(motor))


@pytest.fixture(scope="module", params=_DRIVE_MODES)
def results_run_with_motor(rotor_with_motor, request):
    """Simulate the rotor with each supported drive_mode under rated load."""
    drive_mode = request.param
    t = np.arange(0, 3.0 + 1e-3, 1e-3)
    kwargs = {}
    if drive_mode != "DOL":
        kwargs["time_step"] = 1e-5
        kwargs["time_ramp"] = 0.5
        kwargs["frequency_ref"] = Q_(60.0, "Hz")

    return rotor_with_motor.run_with_motor(
        t=t,
        unbalance_node=[_ROTOR_UNBALANCE_NODE],
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
            unbalance_node=[_ROTOR_UNBALANCE_NODE],
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
            unbalance_node=[_ROTOR_UNBALANCE_NODE],
            unbalance_magnitude=[5e-4],
            unbalance_phase=[0.0],
            drive_mode="DOL",
            load_torque_entrance_time=0.05,
        )


@pytest.mark.parametrize("drive_mode", _DRIVE_MODES)
def test_ip_load_argument_adds_to_the_motor_inertia(drive_mode):
    # Reference: a motor built with Ip equal to Ip + Ip_load, simulated without
    # load inertia.
    Ip_load = 0.05
    circuit = dict(
        power_rated=1118.5,
        voltage_rated=127,
        speed_rated=179.07,
        stator_resistance=2.5,
        rotor_resistance=1.8,
        stator_reactance=1.3,
        rotor_reactance=1.3,
        mutual_reactance=43.08,
    )
    motor = ElectricMotor(n=0, m=10.0, Ip=0.0372, **circuit)
    heavy_motor = ElectricMotor(n=0, m=10.0, Ip=0.0372 + Ip_load, **circuit)

    t = np.arange(0, 0.3 + 1e-3, 1e-3)
    methods = {
        "DOL": ("run_direct_on_line", {}),
        "VFD_VF": ("run_with_inverter_vf", {"frequency_s": 5e3 * 2 * np.pi}),
        "VFD_FOC": ("run_with_inverter_foc", {"frequency_s": 5e3 * 2 * np.pi}),
    }
    name, kwargs = methods[drive_mode]
    if drive_mode != "DOL":
        kwargs["time_step"] = 1e-5
        kwargs["frequency_ref"] = Q_(60.0, "Hz")
        kwargs["time_ramp"] = 0.2

    speed = getattr(motor, name)(t, Ip_load=Ip_load, **kwargs).sample_at("speed", t)
    reference = getattr(heavy_motor, name)(t, **kwargs).sample_at("speed", t)
    unloaded = getattr(motor, name)(t, **kwargs).sample_at("speed", t)

    if drive_mode != "VFD_FOC":
        # The FOC speed controller is tuned with the motor inertia only, so the
        # heavy motor, tuned with the total inertia, is not a reference for it.
        assert_allclose(speed, reference, rtol=1e-8)
    assert np.max(np.abs(unloaded - speed)) > 0.1 * np.max(speed)


_CIRCUIT = dict(
    power_rated=1118.5,
    voltage_rated=127,
    speed_rated=179.07,
    stator_resistance=2.5,
    rotor_resistance=1.8,
    stator_reactance=1.3,
    rotor_reactance=1.3,
    mutual_reactance=43.08,
)


@pytest.fixture
def motor_shaft_elements():
    return [ShaftElement(L=0.1, idl=0, odl=0.05, material=steel) for _ in range(3)]


def test_motor_with_shaft_elements_takes_inertia_from_the_elements(
    motor_shaft_elements,
):
    # Reference: sums over the elements and the disk, written by hand.
    disk = DiskElement(n=2, m=3.0, Id=0.01, Ip=0.02)
    motor = ElectricMotor(
        n=3,
        **_CIRCUIT,
        shaft_elements=motor_shaft_elements,
        disk_elements=[disk],
    )

    assert isinstance(motor.mechanical_model, Rotor)
    assert_allclose(motor.m, sum(sh.m for sh in motor_shaft_elements) + disk.m)
    assert_allclose(motor.Ip, sum(sh.Im for sh in motor_shaft_elements) + disk.Ip)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"m": 10.0},
        {"Ip": 0.03},
        {"disk_elements": []},
        {"bearing_elements": []},
    ],
)
def test_motor_lumped_arguments_are_validated(arguments):
    with pytest.raises(ValueError):
        ElectricMotor(n=0, **_CIRCUIT, **arguments)


@pytest.mark.parametrize("arguments", [{"disk_elements": []}, {"bearing_elements": []}])
def test_motor_lumped_rejects_fem_arguments(arguments):
    with pytest.raises(ValueError):
        ElectricMotor(n=0, **_CIRCUIT, m=10.0, Ip=0.03, **arguments)


def test_motor_with_shaft_elements_forwards_rotor_methods(motor_shaft_elements):
    motor = ElectricMotor(n=3, **_CIRCUIT, shaft_elements=motor_shaft_elements)
    rotor = motor.mechanical_model

    assert motor.nodes == rotor.nodes
    assert "run_modal" in dir(motor)
    # Reference: the same analysis called on the rotor, a different code path.
    assert_allclose(
        motor.run_modal(speed=0).wn, rotor.run_modal(speed=0).wn, rtol=1e-12
    )


def test_motor_with_shaft_elements_keeps_its_own_attributes(motor_shaft_elements):
    motor = ElectricMotor(
        n=3, **_CIRCUIT, shaft_elements=motor_shaft_elements, tag="motor"
    )

    assert motor.Ip == motor.mechanical_model.Ip
