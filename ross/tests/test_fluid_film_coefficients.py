"""Stiffness at the converged state, the cavitation floor and the film force.

References, all independent of the code under test:

- the stiffness of a case held at a prescribed journal position
  (``equilibrium_type="match_eccentricity"``), which evaluates the Jacobian
  without a Newton search, at the position a ``match_load`` case reports;
- central finite differences of the film force and the pad moments returned by
  :func:`ross.bearings.fluid_film.sweep.film_force_at` (the nonlinear film,
  not its linearisation);
- the static condensation of the engine's unreduced blocks by
  :func:`ross.bearings.film.condense.condense_static`;
- the applied load (force balance at the equilibrium) and the cavitation
  pressure (floor of the film pressure).

The perturbation (linearised) Reynolds problems and the nonlinear film are two
discretisations of the same equations, so their stiffness agrees to the
discretisation error, which shrinks as the mesh is refined (measured on 20 x 10 to
160 x 80 meshes). The cavitated region of the nonlinear film comes
from the clamp in the banded back substitution, which is not the Reynolds
condition the perturbation assumes; with cavitation part of the gap stays as
the mesh is refined, and the tests bound it instead.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.film.condense import condense_static
from ross.bearings.fluid_film import coefficients as coeff
from ross.bearings.fluid_film.driver import run_case
from ross.bearings.fluid_film.sweep import film_force_at

DATA_DIR = Path(__file__).parent / "data" / "fluid_film"
COEFFICIENT_KEYS = ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy")


def _inputs(case, **changes):
    inputs = json.loads((DATA_DIR / f"{case}.json").read_text())["inputs"]
    inputs.update(changes)
    return inputs


def _stiffness(out):
    return np.array([[out["kxx"][0], out["kxy"][0]], [out["kyx"][0], out["kyy"][0]]])


def _fd_journal(inputs, x0, y0, step, **kwargs):
    """Central differences of the film force and pad loads in x and y.

    Returns ``(-dF/du, -dF_pad/du, -dM_pad/du)`` with shapes ``(2, 2)``,
    ``(n_pads, 2, 2)`` and ``(n_pads, 2)``: the sign of a stiffness.
    """
    speed = inputs["frequency"][0]
    n_pads = len(inputs["pivot_angle"])
    k = np.zeros((2, 2))
    k_pad = np.zeros((n_pads, 2, 2))
    k_moment = np.zeros((n_pads, 2))
    for j, (dx, dy) in enumerate(((step, 0.0), (0.0, step))):
        plus = film_force_at(inputs, speed, x0 + dx, y0 + dy, **kwargs)
        minus = film_force_at(inputs, speed, x0 - dx, y0 - dy, **kwargs)
        k[:, j] = -np.array([plus["fx"] - minus["fx"], plus["fy"] - minus["fy"]]) / (
            2 * step
        )
        k_pad[:, 0, j] = -(plus["fx_pad"] - minus["fx_pad"]) / (2 * step)
        k_pad[:, 1, j] = -(plus["fy_pad"] - minus["fy_pad"]) / (2 * step)
        k_moment[:, j] = -(plus["moment_pad"] - minus["moment_pad"]) / (2 * step)
    return k, k_pad, k_moment


@pytest.fixture(scope="module")
def tilting_pad_equilibrium():
    """``tilt_5pad_isoviscous`` at its equilibrium, with the unreduced blocks."""
    inputs = _inputs("tilt_5pad_isoviscous")
    out = run_case(**inputs, diagnostics=True, _coefficient_blocks=True)
    assert out["convergence"][0]["converged"]
    cb = inputs["radial_clearance"]
    return inputs, out, out["xj_cb"][0] * cb, out["yj_cb"][0] * cb


@pytest.mark.parametrize(
    "case",
    [
        "fixed_isoviscous",
        "fixed_pdam_bt5",
        "fixed_isoviscous_starved",
        "tilt_5pad_isoviscous",
    ],
)
def test_stiffness_is_evaluated_at_the_reported_position(case):
    """The coefficients belong to the position the case reports.

    Reference: the same bearing held at the reported journal position
    (``match_eccentricity``: no Newton search). Before the fix the reported
    stiffness was the Jacobian of the Newton iterate before the converged
    one; it differed from this reference by 1.2e-3 (2-axial-groove fixture),
    2.8e-3 (pressure dam), 2.2e-3 (starved) and 3.6e-3 (tilting pad) of the
    largest stiffness. Laminar isoviscous cases only: a thermal or turbulent
    case held at a position converges its outer loops along another path.
    """
    inputs = _inputs(case)
    out = run_case(**inputs, diagnostics=True)
    assert out["convergence"][0]["converged"]
    held = run_case(
        **dict(
            inputs,
            equilibrium_type="match_eccentricity",
            xj=out["xj_cb"][0],
            yj=out["yj_cb"][0],
        )
    )
    for key in COEFFICIENT_KEYS:
        assert_allclose(out[key], held[key], rtol=1e-10, atol=0.0, err_msg=key)


def _held_gap(case, position, mesh, cavitation_pressure=-1e12):
    """Largest ``|K - K_FD|`` over ``max |K_FD|`` at a held journal position."""
    inputs = _inputs(
        case,
        cavitation_pressure=cavitation_pressure,
        equilibrium_type="match_eccentricity",
        xj=position[0],
        yj=position[1],
        total_e_x_film=mesh[0],
        total_e_z_film=mesh[1],
    )
    cb = inputs["radial_clearance"]
    out = run_case(**inputs)
    k_fd, _, _ = _fd_journal(inputs, position[0] * cb, position[1] * cb, 1e-4 * cb)
    return np.max(np.abs(_stiffness(out) - k_fd)) / np.max(np.abs(k_fd))


@pytest.mark.parametrize("position", [(0.2, -0.5), (-0.3, -0.65)])
def test_fixed_geometry_stiffness_converges_to_the_film_force_derivative(position):
    """Without cavitation, K and the FD of the film force converge together.

    Reference: central differences (step ``1e-4 C_b``) of the film force of
    :func:`film_force_at` on the 2-land axial-flow fixture (no divergent-zone
    rule), with the cavitation pressure far below the film pressures so that
    no node is clamped. The linearised and the nonlinear discrete problems
    then differ only by their discretisation errors, and the gap falls at
    second order: 2.1 / 0.45 / 0.10 % and 3.5 / 0.92 / 0.25 % of the largest
    stiffness on the 20 x 10, 40 x 20 and 80 x 40 meshes at the two
    positions.
    """
    gaps = [
        _held_gap("fixed_isoviscous_starved", position, mesh)
        for mesh in ((20, 10), (40, 20), (80, 40))
    ]
    assert gaps[0] < 0.04
    assert gaps[1] < 0.3 * gaps[0]
    assert gaps[2] < 0.3 * gaps[1]
    assert gaps[2] < 3e-3


@pytest.mark.slow
def test_fixed_geometry_stiffness_equals_the_film_force_derivative_on_a_fine_mesh():
    """On a 160 x 80 mesh K equals the FD of the film force within 0.1 %.

    Reference: as in
    ``test_fixed_geometry_stiffness_converges_to_the_film_force_derivative``
    (measured gap 0.025 % at ``(0.2, -0.5) C_b``).
    """
    assert _held_gap("fixed_isoviscous_starved", (0.2, -0.5), (160, 80)) < 1e-3


def test_pressure_dam_stiffness_converges_slowly_to_the_film_force_derivative():
    """Pressure dam: K approaches the FD of the film force at first order.

    Reference: central differences of the film force of :func:`film_force_at`
    (no clamped node, as above). The FD is mesh-converged within 1 % on the
    20 x 10 mesh, but the perturbation stiffness of the stepped film is not:
    the gap is 15 / 7.4 / 3.7 % of the largest stiffness on the 20 x 10,
    40 x 20 and 80 x 40 meshes (``kxx`` 2.6 times the FD on the coarsest
    mesh). The gap halving with each refinement shows that both describe the
    same film; the slow convergence is a known limit of the pressure-dam
    coefficients.
    """
    gaps = [
        _held_gap("fixed_pdam_bt5", (0.25, -0.45), mesh)
        for mesh in ((20, 10), (40, 20), (80, 40))
    ]
    assert gaps[1] < 0.6 * gaps[0]
    assert gaps[2] < 0.6 * gaps[1]
    assert gaps[2] < 0.05


@pytest.mark.parametrize(
    "case, bound",
    [
        ("fixed_isoviscous", 0.03),
        ("fixed_isoviscous_starved", 0.03),
        ("fixed_pdam_bt5", 0.1),
    ],
)
def test_stiffness_with_cavitation_stays_near_the_film_force_derivative(case, bound):
    """At the fixtures' equilibria (cavitated films) K stays near the FD.

    Reference: central differences (step ``1e-4 C_b``) of the film force of
    :func:`film_force_at` about the converged position. Measured gaps: 2.1 %
    (2-axial-groove), 2.4 % (axial flow) and 8.2 % (pressure dam, see
    ``test_pressure_dam_stiffness_converges_slowly_to_the_film_force_derivative``)
    of the largest stiffness. With cavitation the gap does not vanish with the
    mesh (0.3-0.8 % left on the 160 x 80 mesh of the 2-axial-groove bearing):
    the clamp of the banded back substitution is not the Reynolds condition
    the perturbation problems assume. The bound catches a sign,
    factor or wrong-state error; the shift of the evaluation point is covered by
    ``test_stiffness_is_evaluated_at_the_reported_position``.
    """
    inputs = _inputs(case)
    out = run_case(**inputs)
    cb = inputs["radial_clearance"]
    k_fd, _, _ = _fd_journal(
        inputs, out["xj_cb"][0] * cb, out["yj_cb"][0] * cb, 1e-4 * cb
    )
    assert np.max(np.abs(_stiffness(out) - k_fd)) < bound * np.max(np.abs(k_fd))


def _pad_matrix(blocks, k):
    """Stiffness of pad ``k`` on ``[x, y, tilt_k]``."""
    m = np.zeros((3, 3))
    m[:2, :2] = [[blocks.xxi[k], blocks.xyi[k]], [blocks.yxi[k], blocks.yyi[k]]]
    m[:2, 2] = [blocks.xdelta[k], blocks.ydelta[k]]
    m[2, :2] = [blocks.deltax[k], blocks.deltay[k]]
    m[2, 2] = blocks.deltadelta[k]
    return m


def _loaded_pads(blocks):
    return [k for k in range(len(blocks.deltadelta)) if blocks.deltadelta[k] != 0.0]


def test_tilting_pad_unreduced_blocks_match_the_film_force_derivative(
    tilting_pad_equilibrium,
):
    """K_uu, K_ue, K_eu and K_ee of each loaded pad against central differences.

    Reference: with the tilts frozen at their equilibrium
    values, the FD of each pad's film force and pivot moment in the journal
    position gives that pad's ``K_uu`` and ``K_eu``; with the journal frozen
    and one tilt perturbed, the FD gives ``K_ue`` and ``K_ee``. The three
    loaded pads have no cavitated node, so the gap is the discretisation
    error. Each block (``uu``, ``ue``, ``eu``, ``ee``) is compared with its
    own largest FD entry, since the blocks carry different units; the gap is
    at most 1.3 % per block on the 20 x 10 mesh. The two upper
    pads of this load-on-pad bearing are fully cavitated: their film force is
    zero at the equilibrium tilt and grows on one side only, so they have no
    derivative to compare and their engine blocks are zero.
    """
    inputs, out, x0, y0 = tilting_pad_equilibrium
    blocks = out["_coefficient_blocks"][0]["stiffness"]
    loaded = _loaded_pads(blocks)
    assert loaded == [2, 3, 4]
    cb = inputs["radial_clearance"]
    tilts = np.asarray(out["tilt_angle"][0])
    _, k_pad, k_moment = _fd_journal(inputs, x0, y0, 1e-4 * cb, tilts=tilts)
    speed = inputs["frequency"][0]
    step = 1e-4 * cb / (0.5 * inputs["journal_diameter"])
    for k in loaded:
        plus, minus = tilts.copy(), tilts.copy()
        plus[k] += step
        minus[k] -= step
        fp = film_force_at(inputs, speed, x0, y0, tilts=plus)
        fm = film_force_at(inputs, speed, x0, y0, tilts=minus)
        fd = np.zeros((3, 3))
        fd[:2, :2] = k_pad[k]
        fd[2, :2] = k_moment[k]
        fd[:2, 2] = -np.array([fp["fx"] - fm["fx"], fp["fy"] - fm["fy"]]) / (2 * step)
        fd[2, 2] = -(fp["moment_pad"][k] - fm["moment_pad"][k]) / (2 * step)
        engine = _pad_matrix(blocks, k)
        # The blocks have different units (N/m, N/rad, N*m/rad), so each one
        # is compared at its own scale.
        for name, rows, cols in (
            ("uu", slice(0, 2), slice(0, 2)),
            ("ue", slice(0, 2), slice(2, 3)),
            ("eu", slice(2, 3), slice(0, 2)),
            ("ee", slice(2, 3), slice(2, 3)),
        ):
            gap = np.max(np.abs(engine[rows, cols] - fd[rows, cols]))
            assert gap < 0.015 * np.max(np.abs(fd[rows, cols])), (k, name)


def test_tilting_pad_resolved_tilt_derivative_is_the_static_condensation(
    tilting_pad_equilibrium,
):
    """Re-solving the tilts at each perturbed position gives K(w = 0).

    Reference: central differences of the journal force with
    the pad moment balance re-solved at each position (tilt tolerance
    1e-13 rad) equal :func:`condense_static` of the engine's per-pad blocks,
    ``K_uu - K_ue K_ee^-1 K_eu``, not the synchronous coefficients the case
    reports (which include the pad inertia and damping at 1X). The gap is
    the discretisation error of the blocks (1.7 % of the largest term on the
    20 x 10 mesh).
    """
    inputs, out, x0, y0 = tilting_pad_equilibrium
    blocks = out["_coefficient_blocks"][0]["stiffness"]
    loaded = _loaded_pads(blocks)
    k_static, _ = condense_static(
        [_pad_matrix(blocks, k) for k in loaded], [np.zeros((3, 3))] * len(loaded), 2
    )
    cb = inputs["radial_clearance"]
    k_fd, _, _ = _fd_journal(inputs, x0, y0, 1e-4 * cb, tolerance_scale=1e-5)
    scale = np.max(np.abs(k_static))
    assert np.max(np.abs(k_static - k_fd)) < 0.02 * scale
    # the synchronous coefficients differ from the static ones by more than
    # the gap, so the comparison discriminates
    assert np.max(np.abs(_stiffness(out) - k_fd)) > 0.05 * scale


@pytest.mark.slow
def test_tilting_pad_block_gap_shrinks_with_the_mesh(tilting_pad_equilibrium):
    """The block gap of the tilting pad is a discretisation error.

    Reference: central differences of :func:`film_force_at` as in
    ``test_tilting_pad_unreduced_blocks_match_the_film_force_derivative``,
    at the converged position of the 20 x 10 fixture held on 20 x 10 and
    40 x 20 meshes (tilts solved at that position). The gap of the loaded
    pads' journal blocks drops by about three (1.2 % to 0.37 %).
    """
    inputs0, out0, _, _ = tilting_pad_equilibrium
    gaps = []
    for ex, ez in ((20, 10), (40, 20)):
        inputs = dict(
            inputs0,
            equilibrium_type="match_eccentricity",
            xj=out0["xj_cb"][0],
            yj=out0["yj_cb"][0],
            total_e_x_film=ex,
            total_e_z_film=ez,
        )
        out = run_case(**inputs, _coefficient_blocks=True)
        blocks = out["_coefficient_blocks"][0]["stiffness"]
        cb = inputs["radial_clearance"]
        tilts = np.asarray(out["tilt_angle"][0])
        _, k_pad, _ = _fd_journal(
            inputs,
            out["xj_cb"][0] * cb,
            out["yj_cb"][0] * cb,
            1e-4 * cb,
            tilts=tilts,
        )
        gaps.append(
            max(
                np.max(np.abs(_pad_matrix(blocks, k)[:2, :2] - k_pad[k]))
                / np.max(np.abs(_pad_matrix(blocks, k)))
                for k in _loaded_pads(blocks)
            )
        )
    assert gaps[1] < 0.45 * gaps[0]
    assert gaps[1] < 6e-3


def test_film_force_at_the_equilibrium_balances_the_load(tilting_pad_equilibrium):
    """The film force at the converged position carries the applied load.

    Reference: the applied load (``weight`` downwards, no horizontal load)
    and the Newton tolerance ``|F_net| / |F_ext| < 2e-3``. With the tilts
    re-solved at that position the tilts and the force equal those of the
    equilibrium solve, and each pad's pivot moment is balanced.
    """
    inputs, out, x0, y0 = tilting_pad_equilibrium
    speed = inputs["frequency"][0]
    weight = inputs["weight"]
    solved = film_force_at(inputs, speed, x0, y0)
    assert solved["converged"]
    assert np.hypot(solved["fx"], solved["fy"] - weight) < 2e-3 * weight
    assert_allclose(solved["tilt_angle"], out["tilt_angle"][0], rtol=0, atol=1e-8)
    assert_allclose(
        [solved["fx"], solved["fy"]], [out["fx_hydro"][0], out["fy_hydro"][0]]
    )
    loaded = solved["moment_pad"][2:]
    lever = 0.5 * inputs["journal_diameter"] + inputs["pad_thickness"]
    pad_force = np.hypot(solved["fx_pad"][2:], solved["fy_pad"][2:])
    # bisection to 1e-8 rad: the residual moment is far below force x lever
    assert np.all(np.abs(loaded) < 1e-3 * pad_force * lever)
    frozen = film_force_at(inputs, speed, x0, y0, tilts="frozen")
    assert_allclose(frozen["tilt_angle"], out["tilt_angle"][0], rtol=0, atol=0)
    assert_allclose([frozen["fx"], frozen["fy"]], [solved["fx"], solved["fy"]])


def test_film_force_at_rejects_bad_tilts():
    inputs = _inputs("tilt_5pad_isoviscous")
    speed = inputs["frequency"][0]
    with pytest.raises(ValueError, match="tilts must"):
        film_force_at(inputs, speed, 0.0, -1e-5, tilts="fixed")
    with pytest.raises(ValueError, match="tilts must hold 5"):
        film_force_at(inputs, speed, 0.0, -1e-5, tilts=[0.0, 0.0])


def test_film_force_at_is_the_force_of_a_held_case():
    """Fixed geometry: the film force equals that of a held ``run_case``.

    Reference: ``run_case`` with ``equilibrium_type="match_eccentricity"`` at
    the same position (its ``fx_hydro``, ``fy_hydro``), at three positions;
    ``tilts`` is ignored for fixed geometry.
    """
    inputs = _inputs("fixed_isoviscous")
    cb = inputs["radial_clearance"]
    speed = inputs["frequency"][0]
    for x, y in ((0.1, -0.3), (0.3, -0.5), (-0.2, -0.7)):
        held = run_case(
            **dict(inputs, equilibrium_type="match_eccentricity", xj=x, yj=y)
        )
        force = film_force_at(inputs, speed, x * cb, y * cb, tilts=[1.0, 2.0])
        assert_allclose(
            [force["fx"], force["fy"]], [held["fx_hydro"][0], held["fy_hydro"][0]]
        )
        assert_allclose(np.sum(force["fy_pad"]), force["fy"], rtol=1e-6)


@pytest.mark.parametrize(
    "case", ["fixed_isoviscous", "tilt_5pad_isoviscous", "fixed_isoviscous_starved"]
)
def test_cavitated_film_is_held_at_the_cavitation_pressure(case):
    """Every cavitated node is at the cavitation pressure, none below.

    Reference: the cavitation pressure ``p_cav = -50 kPa``. Before the fix the
    divergent-zone and fully-cavitated rules wrote 0 Pa: on the 2-axial-groove
    fixture 63 % of the nodes were at 0, none at ``p_cav``, the minimum was
    -32.5 kPa.
    """
    p_cav = -50e3
    inputs = _inputs(case, cavitation_pressure=p_cav)
    out = run_case(**inputs, diagnostics=True)
    assert out["convergence"][0]["converged"]
    cb = inputs["radial_clearance"]
    force = film_force_at(
        inputs,
        inputs["frequency"][0],
        out["xj_cb"][0] * cb,
        out["yj_cb"][0] * cb,
        tilts=np.asarray(out["tilt_angle"][0]),
    )
    pressure, cavitated = force["nodal_pressure"], force["cavitated"]
    assert np.count_nonzero(cavitated) > 0
    assert np.all(pressure[cavitated] == p_cav)
    assert np.min(pressure) >= p_cav
    assert np.all(np.isfinite(_stiffness(out)))


def test_cavitation_rules_set_the_cavitation_pressure():
    """The divergent-zone and fully-cavitated rules write ``p_cav``.

    Reference: the rules themselves. On the tilting pad the fully-cavitated
    rule marks every node of the two unloaded pads. On the 2-axial-groove
    bearing over a third of the film at ``eps = 0.54`` is cavitated; most of
    those nodes are set by the clamp of the back substitution, and the
    divergent-zone rule only adds nodes the clamp already holds on these
    fixtures. Every cavitated node holds ``p_cav`` exactly; before the fix
    the rules wrote 0 Pa.
    """
    p_cav = -50e3
    inputs = _inputs("tilt_5pad_isoviscous", cavitation_pressure=p_cav)
    cb = inputs["radial_clearance"]
    force = film_force_at(inputs, inputs["frequency"][0], 0.0, -0.6 * cb)
    pressure, cavitated = force["nodal_pressure"], force["cavitated"]
    full = [k for k in range(5) if np.all(cavitated[k])]
    assert full == [0, 1]
    assert np.all(pressure[full] == p_cav)

    inputs = _inputs("fixed_isoviscous", cavitation_pressure=p_cav)
    force = film_force_at(inputs, inputs["frequency"][0], 0.2 * cb, -0.5 * cb)
    pressure, cavitated = force["nodal_pressure"], force["cavitated"]
    assert np.count_nonzero(cavitated) > 0.3 * cavitated.size
    assert np.all(pressure[cavitated] == p_cav)


@pytest.mark.parametrize("p_cav", [0.0, -50e3])
def test_perturbation_holds_the_cavitated_nodes_of_the_film(monkeypatch, p_cav):
    """The perturbation problems fix exactly the cavitated nodes.

    Reference: the cavitated mask returned with the zeroth-order pressure
    and the film-edge nodes (the boundary set built with an empty mask). The
    old rule found cavitated nodes by value (``|p - p_cav| < 1e-6``) and,
    with ``p_cav != 0``, missed every node the divergent-zone rule had set to
    0 Pa.
    """
    original = coeff.press_bc_pert
    calls = []

    def spy(mesh, pad_index, film_onset, pads, cavitated):
        total, index, values = original(mesh, pad_index, film_onset, pads, cavitated)
        edges, edge_index, _ = original(
            mesh, pad_index, film_onset, pads, np.zeros_like(cavitated)
        )
        calls.append(
            (
                pad_index,
                set(index[:total].tolist()),
                set(edge_index[:edges].tolist()),
                set(np.flatnonzero(cavitated[pad_index]).tolist()),
            )
        )
        return total, index, values

    monkeypatch.setattr(coeff, "press_bc_pert", spy)
    inputs = _inputs(
        "fixed_isoviscous",
        cavitation_pressure=p_cav,
        equilibrium_type="match_eccentricity",
        xj=0.2,
        yj=-0.5,
    )
    run_case(**inputs)
    monkeypatch.setattr(coeff, "press_bc_pert", original)
    cb = inputs["radial_clearance"]
    force = film_force_at(inputs, inputs["frequency"][0], 0.2 * cb, -0.5 * cb)
    assert calls
    for pad, fixed, edges, mask in calls:
        assert mask == set(np.flatnonzero(force["cavitated"][pad]).tolist())
        assert fixed == edges | mask
        assert len(mask - edges) > 0


@pytest.mark.parametrize("load", [2000.0, 8000.0])
def test_film_force_at_takes_the_engine_inputs_of_an_element(load):
    """The engine inputs of a bearing element drive ``film_force_at``.

    Reference: the applied load and the Newton tolerance: at the position
    the element reports, the film carries ``load`` within ``2e-3``.
    """
    from ross.bearings.plain_journal import PlainJournal
    from ross.units import Q_

    bearing = PlainJournal(
        n=0,
        pad_axial_length=0.05,
        journal_diameter=0.1,
        radial_clearance=100e-6,
        n_pads=2,
        pad_arc=Q_(160, "deg"),
        oil_supply_temperature=Q_(40, "degC"),
        speed=Q_([3000], "RPM"),
        lubricant="ISOVG32",
        oil_flow_v=Q_(10, "l/min"),
        thermal_type=None,
        weight=load,
        total_ex_film=20,
        total_ez_film=10,
        total_ey_film=10,
        total_ey_pad=10,
    )
    speed = bearing.speed_range[0]
    op = bearing.operating_results
    assert op.converged[0]
    force = film_force_at(bearing._engine_inputs(speed), speed, op.xj[0], op.yj[0])
    assert np.hypot(force["fx"], force["fy"] - load) < 2e-3 * load
