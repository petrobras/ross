"""Sign convention of the AMB negative position stiffness ``ks``.

API 617 E.4.8.2.2 c) requires the rotordynamic analysis to include any
negative stiffness effects. ISO 14839-1 defines ``Ks`` as negative. ROSS
stores ``ks < 0`` and treats it as a stiffness in every analysis path:

* frequency domain: ``k_eq = ks + ki * k_amp * k_sense * Re{C(jw)}`` enters
  the stiffness matrix, so the force is ``-k_eq * x``;
* Newmark path: ``compute_amb_controller`` returns
  ``F = ki * i - ks * x_actuator``;
* lsim path: ``AmbTimeResponse`` uses ``-K_x`` in the dynamics and in the
  force outputs.

These tests tie the three paths together, so that a sign error in any one of
them makes the rotor behave differently between analyses.
"""

import control as ct
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross import (
    DiskElement,
    MagneticBearingElement,
    Material,
    Rotor,
    ShaftElement,
)
from ross.bearings.magnetic.amb_models import (
    rotor_example_amb_complex_controllers,
    rotor_example_amb_general_controllers,
)
from ross.bearings.magnetic.amb_time_response import AmbTimeResponse

G0 = 1e-3
I0 = 1.0
AMB_PARAMS = {"g0": G0, "i0": I0, "ag": 1e-4, "nw": 200}

# Kp at which ki * Kp = |ks| (k_amp = k_sense = 1). Below it the AMB cannot
# levitate the rotor.
KP_THRESHOLD = I0 / G0

# Representative sensor and amplifier gains for a rig whose controller does
# not cancel them, used to check that both paths apply the full loop gain
# ki * k_amp * k_sense * C.
RIG_GAINS = {"k_amp": 1.75, "k_sense": 60.0}

# Eigenvalues smaller than this (rad/s) belong to the free axial/torsional
# rigid-body modes or to pole-zero cancellations of the PID realization.
SMALL_EIGENVALUE = 1.0


def _amb(n, kp, **kwargs):
    return MagneticBearingElement(n=n, kp_pid=kp, **{**AMB_PARAMS, **kwargs})


def _p_only_rotor(kp, **amb_kwargs):
    """Small damped rotor on two proportional-only AMBs."""
    steel = Material(name="steel_amb_ks", rho=7850, E=2.1e11, Poisson=0.3)
    shaft = [
        ShaftElement(L=0.1, idl=0.0, odl=0.05, material=steel, alpha=10, beta=1e-5)
        for _ in range(6)
    ]
    disk = DiskElement(n=3, m=5.0, Id=0.01, Ip=0.02)
    bearings = [
        _amb(1, kp, tag="AMB 0", **amb_kwargs),
        _amb(5, kp, tag="AMB 1", **amb_kwargs),
    ]
    return Rotor(shaft, [disk], bearings)


def _closed_loop(rotor):
    """Build AmbTimeResponse without running the simulation."""
    t = np.linspace(0.0, 1e-3, 3)
    sim = AmbTimeResponse(rotor, t=t, speed=0, weight=False)
    sim.process_rotor()
    sim.setup_modal_domain()
    sim.build_matrices()
    return sim


def _max_real_part(eigenvalues):
    eigenvalues = np.asarray(eigenvalues)
    relevant = eigenvalues[np.abs(eigenvalues) > SMALL_EIGENVALUE]
    return np.max(relevant.real)


def test_ks_is_negative_and_consistent_with_ki():
    amb = _amb(0, kp=1.0)

    assert amb.ks < 0
    # Linearized differential pair: ks = -ki * i0 / g0.
    assert_allclose(amb.ks, -amb.ki * I0 / G0, rtol=1e-12)


@pytest.mark.parametrize("kp", [0.5 * KP_THRESHOLD, 3.0 * KP_THRESHOLD])
def test_newmark_force_matches_frequency_domain_stiffness(kp):
    """compute_amb_controller must return F = -k_eq(0) * x for a P controller."""
    amb = _amb(0, kp=kp)
    amb.build_controller(dt=1e-4)

    x = 2e-5
    force, _ = amb.compute_amb_controller(
        current_offset=0.0, setpoint=0.0, disp=x, dof_index=0
    )

    k_eq = amb.ks + amb.ki * kp
    assert_allclose(amb.k_eq[0], k_eq, rtol=1e-12)
    assert_allclose(force, -k_eq * x, rtol=1e-9)


def test_newmark_negative_stiffness_force_is_destabilizing():
    """With a weak controller the net force points along the displacement."""
    amb = _amb(0, kp=0.1 * KP_THRESHOLD)
    amb.build_controller(dt=1e-4)

    for x in (1e-5, -1e-5):
        amb.x_c = [np.zeros_like(state) for state in amb.x_c]
        force, _ = amb.compute_amb_controller(
            current_offset=0.0, setpoint=0.0, disp=x, dof_index=0
        )
        assert np.sign(force) == np.sign(x)


def test_negative_stiffness_force_uses_actuator_displacement():
    amb = _amb(0, kp=KP_THRESHOLD)
    amb.build_controller(dt=1e-4)

    x_sensor, x_actuator = 1e-5, 3e-5
    force, current = amb.compute_amb_controller(
        current_offset=0.0,
        setpoint=0.0,
        disp=x_sensor,
        dof_index=0,
        disp_actuator=x_actuator,
    )

    assert_allclose(current, -KP_THRESHOLD * x_sensor, rtol=1e-9)
    assert_allclose(force, amb.ki * current - amb.ks * x_actuator, rtol=1e-12)


@pytest.mark.parametrize("kp", [0.25 * KP_THRESHOLD, 3.0 * KP_THRESHOLD])
def test_lsim_closed_loop_matches_frequency_domain(kp):
    """AmbTimeResponse.A_cl must contain the eigenvalues of Rotor.A(0).

    With a proportional controller the AMB is a constant stiffness, so both
    paths describe the same linear system. A_cl may have extra eigenvalues
    from the non-minimal PID realization, hence the subset comparison.
    """
    rotor = _p_only_rotor(kp)

    reference = np.linalg.eigvals(rotor.A(speed=0))
    reference = reference[np.abs(reference) > SMALL_EIGENVALUE]
    closed_loop = np.linalg.eigvals(_closed_loop(rotor).A_cl)

    scale = np.max(np.abs(reference))
    for eigenvalue in reference:
        distance = np.min(np.abs(closed_loop - eigenvalue))
        assert distance <= 1e-6 * abs(eigenvalue) + 1e-9 * scale, eigenvalue


@pytest.mark.parametrize(
    "kp, levitates",
    [
        (0.25 * KP_THRESHOLD, False),
        (0.9 * KP_THRESHOLD, False),
        (1.5 * KP_THRESHOLD, True),
        (3.0 * KP_THRESHOLD, True),
    ],
)
def test_levitation_threshold(kp, levitates):
    """Both paths must agree that ki * Kp > |ks| is needed to levitate."""
    rotor = _p_only_rotor(kp)

    frequency_domain = _max_real_part(np.linalg.eigvals(rotor.A(speed=0)))
    time_domain = _max_real_part(np.linalg.eigvals(_closed_loop(rotor).A_cl))

    for max_real in (frequency_domain, time_domain):
        if levitates:
            assert max_real < SMALL_EIGENVALUE
        else:
            assert max_real > SMALL_EIGENVALUE


def test_lsim_force_outputs_use_negative_stiffness():
    """F_x and F_v outputs of AmbTimeResponse must equal -k_eq * displacement."""
    kp = 3.0 * KP_THRESHOLD
    rotor = _p_only_rotor(kp)
    sim = _closed_loop(rotor)
    k_eq = rotor.bearing_elements[0].k_eq[0]

    n_c = 2 * sim.n_amb
    N = sim.n_dof

    # Physical output matrix (undo the balancing similarity transform).
    C_phys = sim.C_cl / sim.state_scaling[None, :]

    # Static state: modal displacements of a nodal displacement field, zero
    # velocities and (empty or irrelevant) controller states set to zero.
    rng = np.random.default_rng(0)
    x_nodes = rng.normal(scale=1e-5, size=N)
    q = sim.Phi.T @ sim.rotor.M(0) @ x_nodes  # Phi is M-orthonormal
    state = np.concatenate([q, np.zeros(sim.num_modes), np.zeros(sim.n_x_c)])
    y = C_phys @ state

    x_amb = y[N : N + n_c]
    v_amb = y[N + n_c : N + 2 * n_c]
    F_x = y[N + 2 * n_c : N + 3 * n_c]
    F_v = y[N + 3 * n_c : N + 4 * n_c]

    assert_allclose(F_x, -k_eq * x_amb, rtol=1e-8)
    assert_allclose(F_v, -k_eq * v_amb, rtol=1e-8)


def test_example_general_controllers_levitates():
    sim = _closed_loop(rotor_example_amb_general_controllers())
    assert _max_real_part(np.linalg.eigvals(sim.A_cl)) < SMALL_EIGENVALUE


def test_example_complex_controllers_levitates():
    sim = _closed_loop(rotor_example_amb_complex_controllers())
    assert _max_real_part(np.linalg.eigvals(sim.A_cl)) < SMALL_EIGENVALUE


@pytest.mark.parametrize("kp", [0.5, 3.0])
def test_newmark_force_includes_sensor_and_amplifier_gains(kp):
    """With k_amp and k_sense, F = -(ks + ki * k_amp * k_sense * Kp) * x."""
    gain = RIG_GAINS["k_amp"] * RIG_GAINS["k_sense"]
    kp = kp * KP_THRESHOLD / gain
    amb = _amb(0, kp=kp, **RIG_GAINS)
    amb.build_controller(dt=1e-4)

    x = 2e-5
    force, current = amb.compute_amb_controller(
        current_offset=0.0, setpoint=0.0, disp=x, dof_index=0
    )

    k_eq = amb.ks + amb.ki * gain * kp
    assert_allclose(amb.k_eq[0], k_eq, rtol=1e-12)
    assert_allclose(current, -gain * kp * x, rtol=1e-9)
    assert_allclose(force, -k_eq * x, rtol=1e-9)


@pytest.mark.parametrize("kp", [0.25, 3.0])
def test_lsim_closed_loop_matches_frequency_domain_with_gains(kp):
    gain = RIG_GAINS["k_amp"] * RIG_GAINS["k_sense"]
    rotor = _p_only_rotor(kp * KP_THRESHOLD / gain, **RIG_GAINS)

    reference = np.linalg.eigvals(rotor.A(speed=0))
    reference = reference[np.abs(reference) > SMALL_EIGENVALUE]
    closed_loop = np.linalg.eigvals(_closed_loop(rotor).A_cl)

    scale = np.max(np.abs(reference))
    for eigenvalue in reference:
        distance = np.min(np.abs(closed_loop - eigenvalue))
        assert distance <= 1e-6 * abs(eigenvalue) + 1e-9 * scale, eigenvalue


@pytest.mark.parametrize("kp, levitates", [(0.9, False), (1.5, True)])
def test_levitation_threshold_with_gains(kp, levitates):
    """The threshold becomes ki * k_amp * k_sense * Kp > |ks| in both paths."""
    gain = RIG_GAINS["k_amp"] * RIG_GAINS["k_sense"]
    rotor = _p_only_rotor(kp * KP_THRESHOLD / gain, **RIG_GAINS)

    frequency_domain = _max_real_part(np.linalg.eigvals(rotor.A(speed=0)))
    time_domain = _max_real_part(np.linalg.eigvals(_closed_loop(rotor).A_cl))

    for max_real in (frequency_domain, time_domain):
        assert bool(max_real < SMALL_EIGENVALUE) == levitates


def test_lsim_current_output_includes_gains():
    """The current output of AmbTimeResponse is k_amp * C(k_sense * error)."""
    gain = RIG_GAINS["k_amp"] * RIG_GAINS["k_sense"]
    kp = 3.0 * KP_THRESHOLD / gain
    sim = _closed_loop(_p_only_rotor(kp, **RIG_GAINS))

    n_c = 2 * sim.n_amb
    N = sim.n_dof
    C_phys = sim.C_cl / sim.state_scaling[None, :]

    rng = np.random.default_rng(1)
    x_nodes = rng.normal(scale=1e-5, size=N)
    q = sim.Phi.T @ sim.rotor.M(0) @ x_nodes
    state = np.concatenate([q, np.zeros(sim.num_modes), np.zeros(sim.n_x_c)])
    y = C_phys @ state

    v_amb = y[N + n_c : N + 2 * n_c]
    current = y[N + 4 * n_c : N + 5 * n_c]

    assert_allclose(current, -gain * kp * v_amb, rtol=1e-8)


# Linearized differential actuator, Schweitzer and Maslen (2009), p. 343:
# k = mu0 * nw**2 * ag * cos(alpha) / 4, ki = 4 k i0 / g0**2, ks = -4 k i0**2 / g0**3.
MU0 = 4e-7 * np.pi
ALPHA = 0.39269908
K_MAGNET = MU0 * AMB_PARAMS["nw"] ** 2 * AMB_PARAMS["ag"] * np.cos(ALPHA) / 4


def _rotation(angle):
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def _axis_k_eq(i0, k_amp, kp):
    """Static equivalent stiffness of one axis with a P controller (N/m)."""
    ki = 4 * K_MAGNET * i0 / G0**2
    ks = -4 * K_MAGNET * i0**2 / G0**3
    return ks + ki * k_amp * kp


def test_per_axis_values_equal_to_scalars_give_the_same_element():
    scalar = _amb(0, kp=3.0 * KP_THRESHOLD, k_amp=1.75, k_sense=60.0)
    per_axis = _amb(
        0, kp=3.0 * KP_THRESHOLD, i0=[I0, I0], k_amp=[1.75, 1.75], k_sense=[60.0, 60.0]
    )

    assert per_axis.k_eq.shape == scalar.k_eq.shape
    for frequency in (0.0, 10.0, 1000.0):
        assert_allclose(per_axis.K(frequency), scalar.K(frequency), rtol=1e-12)
        assert_allclose(per_axis.C(frequency), scalar.C(frequency), rtol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 4])
def test_per_axis_stiffness_matches_linearized_actuator(angle):
    """Different bias and amplifier gain per axis give a diagonal K in the sensor axes."""
    i0_axes, k_amp_axes, kp = [1.0, 0.8], [1.0, 2.0], 3.0 * KP_THRESHOLD
    amb = _amb(0, kp=kp, i0=i0_axes, k_amp=k_amp_axes, sensors_axis_rotation=angle)

    expected = np.diag(
        [
            _axis_k_eq(i0, k_amp, kp)
            for i0, k_amp in zip(i0_axes, k_amp_axes, strict=True)
        ]
    )
    rotation = _rotation(angle)
    K_sensor_axes = rotation @ amb.K(0)[:2, :2] @ np.linalg.inv(rotation)

    scale = np.max(np.abs(expected))
    assert_allclose(K_sensor_axes, expected, rtol=1e-9, atol=1e-9 * scale)
    assert amb.k_eq.shape == (2, len(amb.frequency))


def test_newmark_force_uses_per_axis_coefficients():
    i0_axes, k_amp_axes, kp = [1.0, 0.8], [1.0, 2.0], 3.0 * KP_THRESHOLD
    amb = _amb(0, kp=kp, i0=i0_axes, k_amp=k_amp_axes)
    amb.build_controller(dt=1e-4)

    x = 2e-5
    for axis, (i0, k_amp) in enumerate(zip(i0_axes, k_amp_axes, strict=True)):
        force, _ = amb.compute_amb_controller(
            current_offset=0.0, setpoint=0.0, disp=x, dof_index=axis
        )
        assert_allclose(force, -_axis_k_eq(i0, k_amp, kp) * x, rtol=1e-9)


@pytest.mark.parametrize("kp", [0.9, 3.0])
def test_lsim_closed_loop_matches_frequency_domain_per_axis(kp):
    rotor = _p_only_rotor(kp * KP_THRESHOLD, i0=[1.0, 0.8], k_amp=[1.0, 2.0])

    reference = np.linalg.eigvals(rotor.A(speed=0))
    reference = reference[np.abs(reference) > SMALL_EIGENVALUE]
    closed_loop = np.linalg.eigvals(_closed_loop(rotor).A_cl)

    scale = np.max(np.abs(reference))
    for eigenvalue in reference:
        distance = np.min(np.abs(closed_loop - eigenvalue))
        assert distance <= 1e-6 * abs(eigenvalue) + 1e-9 * scale, eigenvalue


@pytest.mark.parametrize(
    "i0_axes, levitates",
    [([1.0, 0.8], True), ([1.2, 0.8], False), ([0.8, 1.2], False)],
)
def test_levitation_threshold_is_checked_per_axis(i0_axes, levitates):
    """Kp = 1100 A/m levitates an axis only when i0 / g0 < Kp, i.e. i0 < 1.1 A."""
    rotor = _p_only_rotor(1.1 * KP_THRESHOLD, i0=i0_axes)

    frequency_domain = _max_real_part(np.linalg.eigvals(rotor.A(speed=0)))
    time_domain = _max_real_part(np.linalg.eigvals(_closed_loop(rotor).A_cl))

    for max_real in (frequency_domain, time_domain):
        assert bool(max_real < SMALL_EIGENVALUE) == levitates


# Operating point under a static load. Reference: the nonlinear force of the
# differential pair with unipolar amplifiers (Schweitzer and Maslen, 2009),
#     F(x, ic) = k [(i0 + ic)+^2 / (g0 - x)^2 - (i0 - ic)+^2 / (g0 + x)^2],
# differentiated by central differences.
F_SWITCH = 4 * K_MAGNET * I0**2 / G0**2  # load at which one coil stops conducting


def _pair_force(x, ic, i0=I0):
    top, bottom = max(i0 + ic, 0.0), max(i0 - ic, 0.0)
    return K_MAGNET * (top**2 / (G0 - x) ** 2 - bottom**2 / (G0 + x) ** 2)


def _numerical_ki_ks(x, ic):
    di, dx = 1e-7, 1e-10
    ki = (_pair_force(x, ic + di) - _pair_force(x, ic - di)) / (2 * di)
    ks = -(_pair_force(x + dx, ic) - _pair_force(x - dx, ic)) / (2 * dx)
    return ki, ks


def _loaded_amb(static_load, integral=True, **kwargs):
    controller = {"ki_pid": 100.0} if integral else {}
    return _amb(
        0,
        kp=3.0 * KP_THRESHOLD,
        static_load=static_load,
        sensors_axis_rotation=0.0,
        **controller,
        **kwargs,
    )


def test_zero_static_load_gives_the_nominal_element():
    nominal = _amb(0, kp=3.0 * KP_THRESHOLD, ki_pid=100.0)
    loaded = _amb(0, kp=3.0 * KP_THRESHOLD, ki_pid=100.0, static_load=(0.0, 0.0))

    assert_allclose(loaded.ks, nominal.ks, rtol=1e-12)
    assert_allclose(loaded.ki, nominal.ki, rtol=1e-12)
    assert_allclose(loaded.K(0), nominal.K(0), rtol=1e-12)
    assert_allclose(loaded.static_current, 0.0)


@pytest.mark.parametrize(
    "load_factors",
    [(0.3, -0.9), (0.99, 1.01), (1.5, -3.0)],
    ids=["both coils", "around the switch", "one coil"],
)
def test_static_load_linearization_with_integral_action(load_factors):
    """Centered rotor: ic holds the load, ki and ks are the slopes of F there."""
    loads = F_SWITCH * np.array(load_factors)
    amb = _loaded_amb(loads)
    ki, ks = np.broadcast_to(amb.ki, 2), np.broadcast_to(amb.ks, 2)

    assert_allclose(amb.static_displacement, 0.0)
    for axis, load in enumerate(loads):
        ic = amb.static_current[axis]
        assert_allclose(_pair_force(0.0, ic), load, rtol=1e-12)
        ki_ref, ks_ref = _numerical_ki_ks(0.0, ic)
        assert_allclose(ki[axis], ki_ref, rtol=1e-6)
        assert_allclose(ks[axis], ks_ref, rtol=1e-6)


def test_static_load_closed_form_with_both_coils_conducting():
    """Below the switch, ki is unchanged and ks = ks_nominal * (1 + (ic / i0)^2)."""
    nominal = _amb(0, kp=3.0 * KP_THRESHOLD)
    amb = _loaded_amb((0.5 * F_SWITCH, 0.0))
    ic = amb.static_current[0]

    assert_allclose(amb.ki[0], nominal.ki, rtol=1e-12)
    assert_allclose(amb.ks[0], nominal.ks * (1 + (ic / I0) ** 2), rtol=1e-12)


def test_static_load_coefficients_are_continuous_at_the_switch():
    below = _loaded_amb((F_SWITCH * (1 - 1e-9), 0.0))
    above = _loaded_amb((F_SWITCH * (1 + 1e-9), 0.0))

    assert_allclose(above.ki[0], below.ki[0], rtol=1e-6)
    assert_allclose(above.ks[0], below.ks[0], rtol=1e-6)


@pytest.mark.parametrize("load_factor", [0.2, 0.5, -0.5])
def test_static_load_without_integral_action_moves_the_rotor(load_factor):
    """The rotor moves until the static controller force holds the load."""
    load = load_factor * F_SWITCH
    amb = _loaded_amb((load, 0.0), integral=False)
    x, ic = amb.static_displacement[0], amb.static_current[0]

    assert_allclose(ic, -3.0 * KP_THRESHOLD * x, rtol=1e-12)
    assert_allclose(_pair_force(x, ic), load, rtol=1e-9)
    ki_ref, ks_ref = _numerical_ki_ks(x, ic)
    assert_allclose(amb.ki[0], ki_ref, rtol=1e-6)
    assert_allclose(amb.ks[0], ks_ref, rtol=1e-6)


def test_static_load_beyond_the_controller_raises():
    with pytest.raises(ValueError, match="cannot hold a static load"):
        _amb(0, kp=0.9 * KP_THRESHOLD, static_load=(0.5 * F_SWITCH, 0.0))


def test_static_load_is_projected_on_the_control_axes():
    """At 45 deg, a vertical load splits equally; a diagonal one loads one axis.

    The angle is passed as pi / 4: the default sensors_axis_rotation
    (0.78539816) is rounded, so a vertical load splits with a relative
    difference of about 7e-9 between the axes.
    """
    params = {
        "kp": 3.0 * KP_THRESHOLD,
        "ki_pid": 100.0,
        "sensors_axis_rotation": np.pi / 4,
    }
    vertical = _amb(0, static_load=(0.0, 1.0), **params)
    diagonal = _amb(0, static_load=(1.0, 1.0), **params)

    assert_allclose(vertical.static_current[0], vertical.static_current[1], rtol=1e-12)
    assert np.ndim(vertical.ks) == 0
    assert_allclose(vertical.K(0)[0, 1], 0.0, atol=1e-9 * abs(vertical.K(0)[0, 0]))

    ic_v, ic_w = diagonal.static_current
    assert_allclose(_pair_force(0.0, ic_v), np.sqrt(2.0), rtol=1e-12)
    assert_allclose(ic_w, 0.0, atol=1e-12)
    assert abs(diagonal.K(0)[0, 1]) > 1e-6 * abs(diagonal.K(0)[0, 0])


def test_static_load_survives_save_and_load(tmp_path):
    amb = _loaded_amb((0.5 * F_SWITCH, -1.5 * F_SWITCH))
    file = tmp_path / "amb.toml"
    amb.save(file)
    loaded = MagneticBearingElement.load(file)

    assert_allclose(loaded.static_load, amb.static_load)
    assert_allclose(loaded.ks, amb.ks, rtol=1e-12)
    assert_allclose(loaded.ki, amb.ki, rtol=1e-12)
    assert_allclose(loaded.K(0), amb.K(0), rtol=1e-12)


def test_lsim_closed_loop_matches_frequency_domain_under_static_load():
    """Both paths use the operating-point ki and ks of each axis."""
    rotor = _p_only_rotor(
        3.0 * KP_THRESHOLD, static_load=(0.4 * F_SWITCH, 0.1 * F_SWITCH)
    )

    reference = np.linalg.eigvals(rotor.A(speed=0))
    reference = reference[np.abs(reference) > SMALL_EIGENVALUE]
    closed_loop = np.linalg.eigvals(_closed_loop(rotor).A_cl)

    scale = np.max(np.abs(reference))
    for eigenvalue in reference:
        distance = np.min(np.abs(closed_loop - eigenvalue))
        assert distance <= 1e-6 * abs(eigenvalue) + 1e-9 * scale, eigenvalue


def test_static_load_needs_two_values():
    with pytest.raises(ValueError, match="two values"):
        _amb(0, kp=3.0 * KP_THRESHOLD, static_load=(1.0, 2.0, 3.0))


def test_static_load_without_equilibrium_raises():
    """Without integral action the pair holds at most about F_SWITCH in the gap."""
    with pytest.raises(ValueError, match="cannot hold a static load"):
        _loaded_amb((2.0 * F_SWITCH, 0.0), integral=False)


def test_controller_static_gain_cancels_common_zeros():
    """C(s) = Kp s / s has no integral action: its static gain is Kp."""
    kp = 3.0 * KP_THRESHOLD
    amb = _amb(0, kp=kp)
    amb.get_analog_controller = lambda: ct.tf([kp, 0.0], [1.0, 0.0])

    assert_allclose(amb._controller_static_gain(), kp)


def test_hover_info_shows_static_load():
    amb = _loaded_amb((0.5 * F_SWITCH, -1.5 * F_SWITCH))
    _, hovertemplate = amb._hover_info()

    assert "Static Load (x, y)" in hovertemplate
