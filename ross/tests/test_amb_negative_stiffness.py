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

# Eigenvalues smaller than this (rad/s) belong to the free axial/torsional
# rigid-body modes or to pole-zero cancellations of the PID realization.
SMALL_EIGENVALUE = 1.0


def _amb(n, kp, **kwargs):
    return MagneticBearingElement(n=n, kp_pid=kp, **AMB_PARAMS, **kwargs)


def _p_only_rotor(kp):
    """Small damped rotor on two proportional-only AMBs."""
    steel = Material(name="steel_amb_ks", rho=7850, E=2.1e11, Poisson=0.3)
    shaft = [
        ShaftElement(L=0.1, idl=0.0, odl=0.05, material=steel, alpha=10, beta=1e-5)
        for _ in range(6)
    ]
    disk = DiskElement(n=3, m=5.0, Id=0.01, Ip=0.02)
    bearings = [_amb(1, kp, tag="AMB 0"), _amb(5, kp, tag="AMB 1")]
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "With k_amp = k_sense = 1 the controller gain is far below |ks| / ki "
        "(about 2315 A/m for this actuator); the gains or the sensor and "
        "amplifier gains of this example must be revised."
    ),
)
def test_example_complex_controllers_levitates():
    sim = _closed_loop(rotor_example_amb_complex_controllers())
    assert _max_real_part(np.linalg.eigvals(sim.A_cl)) < SMALL_EIGENVALUE
