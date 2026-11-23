"""Pad-tilt root finding, factorisation reuse and compiled glue.

References, all independent of the code under test:

- the pad-tilt bisection of the earlier engine, kept here as
  :func:`_bisection` (run to 1e-12 rad it is the reference of the tilt
  solution; run to its old 1e-8 rad it is the old engine);
- the perturbation solve of the earlier engine, which assembled and factored the
  matrix again for every right-hand side, kept here as
  :func:`_solve_pert_pressure_refactor`;
- the Python versions of the boundary conditions and the post-processing of
  the Reynolds solve and of the adiabatic element loop, kept here as
  ``_press_bc_python``, ``_postprocess_python`` and
  ``_adiabatic_assembly_python``, run on random inputs;
- :func:`ross.bearings.fluid_film.pressure.press` itself for the pressure of a
  fully cavitated pad, which the engine now writes without a solve;
- moment functions with known roots for the search itself.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import ross.bearings.fluid_film.coefficients as coeff
import ross.bearings.fluid_film.hydrodynamics as hyd
import ross.bearings.fluid_film.pressure as prs
from ross.bearings.fluid_film import banded, sweep
from ross.bearings.fluid_film._numba_kernels import adiab_assemble_all_jit
from ross.bearings.fluid_film.constants import TILT_ERROR, TILT_MOMENT_ERROR
from ross.bearings.fluid_film.driver import run_case
from ross.bearings.fluid_film.hydrodynamics import PadTiltSearch
from ross.bearings.fluid_film.sweep import BearingSetup, film_force_at

DATA_DIR = Path(__file__).parent / "data" / "fluid_film"
SPEED = 94.24777960769379  # every fixture runs at 900 rpm
K_KEYS = ("kxx", "kxy", "kyx", "kyy")
C_KEYS = ("cxx", "cxy", "cyx", "cyy")


def _inputs(case, **changes):
    inputs = json.loads((DATA_DIR / f"{case}.json").read_text())["inputs"]
    inputs.update(changes)
    return inputs


def _bisection(tolerance, zero=None):
    """The old tilt bisection, to ``tolerance`` rad, in the search protocol.

    Halves the admissible range on the sign of the moment (``M >= 0``: the
    root is at a larger tilt) and ends at the lower bound of a bracket
    narrower than ``tolerance``; a tilt that closes the film moves the upper
    bound if positive, the lower bound otherwise. With ``zero``, a moment
    below ``zero`` times the moment scale counts as zero: the moment of a
    uniform pressure about a central pivot is round-off of either sign when
    that pressure is not zero, and zero when it is.
    """

    class Bisection:
        def __init__(
            self,
            lower,
            upper,
            tilt_tolerance,
            moment_tolerance,
            start=None,
            slope=None,
        ):
            self.lower, self.upper = float(lower), float(upper)
            self.iterations = 0
            self.converged = False
            self.residual = np.inf
            self.moment_ratio = np.inf
            self.width = abs(upper - lower)
            self.slope = None

        def propose(self):
            if abs(self.upper - self.lower) < tolerance:
                return self.lower
            return 0.5 * (self.lower + self.upper)

        def reject(self, delta):
            if abs(self.upper - self.lower) < tolerance:
                return False
            if delta > 0.0:
                self.upper = delta
            else:
                self.lower = delta
            return True

        def skip(self, delta, flat):
            return False

        def update(self, delta, moment, scale, flat=False):
            self.iterations += 1
            self.width = abs(self.upper - self.lower)
            if self.width < tolerance:
                self.converged = True
                self.residual = 0.0
                return True
            if zero is not None and abs(moment) <= zero * scale:
                moment = 0.0
            if moment >= 0.0:
                self.lower = delta
            else:
                self.upper = delta
            return False

    return Bisection


@pytest.fixture
def use_bisection(monkeypatch):
    def install(tolerance, zero=None):
        monkeypatch.setattr(hyd, "PadTiltSearch", _bisection(tolerance, zero))

    return install


def _counted(monkeypatch):
    counts = {"press": 0}
    original = prs.press

    def press(*args, **kwargs):
        counts["press"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(prs, "press", press)
    return counts


def _matrix_gaps(actual, expected):
    """Largest difference of K and C over each matrix's scale, direct terms."""
    gaps = []
    for keys in (K_KEYS, C_KEYS):
        a = np.array([actual[k][0] for k in keys], dtype=float)
        b = np.array([expected[k][0] for k in keys], dtype=float)
        gaps.append(np.max(np.abs(a - b)) / np.max(np.abs(b)))
        gaps.extend(abs(a[i] - b[i]) / abs(b[i]) for i in (0, 3))
    return max(gaps)


def _drive(search, moment, flat=None, admissible=None):
    """Run the search like ``fixed_brg``; return the tilt and the solve count.

    ``moment(delta) -> (M, scale)``; ``flat(delta)`` marks a fully cavitated
    film (no solve needed); ``admissible(delta)`` is False where the film
    closes. Returns the final tilt, the number of moment evaluations at
    non-flat tilts (Reynolds solves) and every evaluated tilt.
    """
    flat = flat or (lambda d: False)
    admissible = admissible or (lambda d: True)
    solves, evaluated = 0, []
    for _ in range(100):
        while True:
            delta = search.propose()
            if not admissible(delta) and search.reject(delta):
                continue
            if search.skip(delta, flat(delta)):
                continue
            break
        evaluated.append(delta)
        solves += not flat(delta)
        m, scale = moment(delta)
        if search.update(delta, m, scale, flat=flat(delta)):
            break
    return delta, solves, evaluated


SCALE = 1.0e3  # N*m
SLOPE = -2.0e6  # N*m/rad
LIMITS = (-1.0e-3, 1.5e-3)  # rad


def _smooth(root):
    def moment(delta):
        x = delta - root
        return SLOPE * x * (1.0 + 300.0 * x + 4.0e5 * x * x), SCALE

    return moment


def _new_search(start=None, slope=None, scale=1.0):
    return PadTiltSearch(
        *LIMITS,
        TILT_ERROR * scale,
        TILT_MOMENT_ERROR * scale,
        start=start,
        slope=slope,
    )


@pytest.mark.parametrize("root", [-7.3e-4, -1.0e-5, 2.3e-4, 1.2e-3])
@pytest.mark.parametrize(
    "start, slope, max_solves",
    [(None, None, 12), (2.0e-4, None, 14), (2.0e-4, SLOPE, 12)],
    ids=["cold", "warm", "warm_slope"],
)
def test_search_finds_the_root_of_a_smooth_moment(root, start, slope, max_solves):
    """Reference: the exact root of a monotone cubic moment."""
    moment = _smooth(root)
    search = _new_search(start, slope)
    delta, solves, _ = _drive(search, moment)
    assert search.converged
    # moment criterion 1e-9 of the scale: |delta - root| <= 1e-9 * 1e3 / 2e6
    assert abs(delta - root) <= max(TILT_ERROR, 1e-9 * SCALE / abs(SLOPE)) * 1.01
    assert abs(moment(delta)[0]) <= TILT_MOMENT_ERROR * SCALE or (
        search.width <= TILT_ERROR
    )
    assert solves <= max_solves
    # the old bisection: 19 solves to its 1e-8 rad over this range
    _, bisection_solves, _ = _drive(_bisection(1e-8)(*LIMITS, 0, 0), moment)
    assert bisection_solves == 19


@pytest.mark.parametrize("offset", [1e-7, 1e-6, -3e-6, 2e-5])
def test_warm_search_near_the_root_takes_a_few_solves(offset):
    """A start near the root with the slope: five solves or fewer.

    Two solves bracket the root (the start and four secant distances
    beyond); Brent's steps then meet the moment tolerance, or the tilt
    tolerance followed by one solve at the bracket end with the smaller
    moment.
    """
    root = 3.1e-4
    search = _new_search(start=root + offset, slope=SLOPE)
    delta, solves, _ = _drive(search, _smooth(root))
    assert abs(delta - root) <= TILT_ERROR
    assert solves <= 5


@pytest.mark.parametrize("start", [None, 1.0e-4, 6.0e-4])
def test_search_settles_at_the_edge_of_a_fully_cavitated_range(start):
    """Unloaded pad: zero moment up to the edge, a negative jump beyond.

    Reference: the bisection keeps a zero moment on the side of the larger
    tilt, so it ends at the largest flat tilt, the edge ``e``; the new
    search ends there too, within the tilt tolerance and on the flat side,
    with the edge located from the film alone (few solves).
    """
    edge = 2.0225e-4

    def flat(delta):
        return delta <= edge

    def moment(delta):
        return (0.0, 0.0) if flat(delta) else (-6.0 + SLOPE * (delta - edge), 1e2)

    search = _new_search(start=start, slope=SLOPE if start else None)
    delta, solves, _ = _drive(search, moment, flat=flat)
    assert edge - TILT_ERROR <= delta <= edge
    assert search.converged
    assert solves <= 3
    reference, _, _ = _drive(_bisection(1e-12)(*LIMITS, 0, 0), moment, flat=flat)
    assert abs(delta - reference) < TILT_ERROR


@pytest.mark.parametrize("start", [None, -6.0e-4, 1.0e-4])
@pytest.mark.parametrize("roundoff", [9.0e-15, -9.0e-15])
@pytest.mark.parametrize("flagged", [True, False])
def test_search_settles_at_the_edge_of_a_uniform_pressure_range(
    start, roundoff, flagged
):
    """Unloaded pad at a nonzero uniform (cavitation or ambient) pressure.

    Up to the edge the film carries a uniform pressure: its moment about the
    central pivot is round-off (``roundoff``, of either sign) against a
    moment scale of ``(R + t) |p| A`` = 900 N*m, flagged as a flat film or
    not (the unloaded pad of the high-ambient-pressure operation is not).
    Reference: the edge ``e``, where the moment jumps to a negative value,
    as with a zero uniform pressure; a start inside the range must not end
    the search there. The tight bisection with the round-off counted as
    zero ends there too.
    """
    edge = 2.0225e-4

    def flat(delta):
        return flagged and delta <= edge

    def moment(delta):
        if delta <= edge:
            return roundoff, 900.0
        return -6.0 + SLOPE * (delta - edge), 900.0

    search = _new_search(start=start, slope=SLOPE if start else None)
    delta, _, _ = _drive(search, moment, flat=flat)
    assert edge - TILT_ERROR <= delta <= edge
    assert search.converged
    reference, _, _ = _drive(
        _bisection(1e-12, zero=1e-12)(*LIMITS, 0, 0), moment, flat=flat
    )
    assert abs(delta - reference) < TILT_ERROR


def test_search_finds_a_root_beyond_the_edge_of_a_flat_range():
    """Zero moment up to the edge, positive just beyond, then a root."""
    edge = -1.5e-4
    root = edge + 3.0 / -SLOPE

    def flat(delta):
        return delta <= edge

    def moment(delta):
        return (0.0, 0.0) if flat(delta) else (SLOPE * (delta - root), 1e2)

    delta, _, _ = _drive(_new_search(), moment, flat=flat)
    assert abs(delta - root) <= 1e-9 * 1e2 / -SLOPE * 1.01


def test_search_finds_a_root_inside_a_flat_range():
    """A pivot spring moment ``-k delta`` changes sign inside the flat range."""
    edge = 4.0e-4
    k = 5.0e3

    def flat(delta):
        return delta <= edge

    def moment(delta):
        spring = -k * delta
        if flat(delta):
            return spring, abs(spring)
        return spring - 20.0, abs(spring) + 20.0

    delta, solves, _ = _drive(_new_search(), moment, flat=flat)
    assert abs(delta) <= TILT_ERROR
    assert solves <= 2


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_search_without_a_root_ends_at_the_range_end(sign):
    """Reference: the bisection converges to the end the moment points to."""

    def moment(delta):
        return sign * (5.0 + 1e3 * delta), 10.0

    delta, _, _ = _drive(_new_search(start=1e-4, slope=SLOPE), moment)
    reference, _, _ = _drive(_bisection(1e-12)(*LIMITS, 0, 0), moment)
    assert delta == (LIMITS[1] if sign > 0 else LIMITS[0])
    assert abs(delta - reference) < 1e-11


def test_search_never_solves_a_closed_film():
    """The film closes above 5e-4 rad; the root 2.3e-4 rad is found."""
    root = 2.3e-4
    search = _new_search()
    delta, _, evaluated = _drive(
        search, _smooth(root), admissible=lambda d: d <= 5.0e-4
    )
    assert abs(delta - root) < 1e-11
    assert max(evaluated) <= 5.0e-4


TIGHT_CASES = [
    "tilt_5pad_isoviscous",
    "tilt_5pad_pivot",
    pytest.param("tilt_5pad_pivot_ball", marks=pytest.mark.slow),
    pytest.param("tilt_5pad_pivot_button", marks=pytest.mark.slow),
    pytest.param("tilt_5pad_pivot_hertz", marks=pytest.mark.slow),
    pytest.param("tilt_5pad_deform5", marks=pytest.mark.slow),
    pytest.param("tilt_5pad_full", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("case", TIGHT_CASES)
def test_tilts_agree_with_a_tight_bisection(case, use_bisection):
    """Both solvers at 1e-3 of the default tolerances.

    Reference: the old bisection run to 1e-12 rad. Tilts agree to 1e-10 rad;
    the coefficients agree to 1e-6 of each matrix's scale and of each direct
    term (measured: tilts 5e-13 to 1.3e-12 rad, coefficients 2e-10 to 5e-9;
    a cross-coupled entry of about 1e-3 of the matrix differs by up to
    1.2e-6 of itself).
    """
    inputs = _inputs(case)
    new = run_case(**inputs, _tolerance_scale=1e-3, diagnostics=True)
    use_bisection(1e-12)
    reference = run_case(**inputs, _tolerance_scale=1e-3, diagnostics=True)
    assert new["convergence"][0]["converged"]
    assert reference["convergence"][0]["converged"]
    assert_allclose(
        new["tilt_angle"][0], reference["tilt_angle"][0], rtol=0, atol=1e-10
    )
    assert _matrix_gaps(new, reference) < 1e-6


CAVITATION_CASES = [
    ("tilt_5pad_isoviscous", -5.0e4),
    ("tilt_5pad_isoviscous", -1.0e3),
    ("tilt_5pad_isoviscous", 1.0e3),
    pytest.param("tilt_5pad_full", -5.0e4, marks=pytest.mark.slow),
]
AMBIENT_CASES = [
    {
        "operating_type": "high_ambient_pressure",
        "ambient_pressure_1": 2.0e5,
        "ambient_pressure_2": 1.0e5,
    },
    {
        "operating_type": "axial_flow",
        "ambient_pressure_1": 3.0e5,
        "ambient_pressure_2": 1.0e5,
    },
]


@pytest.mark.parametrize("case, cavitation_pressure", CAVITATION_CASES)
def test_unloaded_pads_with_a_nonzero_cavitation_pressure(
    case, cavitation_pressure, use_bisection
):
    """A flat film at a nonzero cavitation pressure: the edge of its range.

    Over the tilts at which an unloaded pad's film is fully cavitated, the
    pad carries the uniform cavitation pressure, whose moment about a
    central pivot is zero (round-off when that pressure is not zero). The
    pad settles at the edge of that range, as with a zero cavitation
    pressure, whatever the start of the search. Reference: the old
    bisection run to 1e-12 rad with that round-off counted as zero (below
    1e-12 of the moment scale, a margin of about 1e5 over the round-off
    measured; the search uses 1.4e-14). Both solvers at 1e-3 of the default
    tolerances; tilts agree to 1e-10 rad and the coefficients to 1e-6. On the
    isoviscous pad, whose film does not depend on a temperature field, the
    equilibrium tilts are found again at the default tolerances from a cold
    start at the same journal position (``film_force_at``), within 2e-8 rad.
    """
    inputs = _inputs(case, cavitation_pressure=cavitation_pressure)
    new = run_case(**inputs, _tolerance_scale=1e-3, diagnostics=True)
    if case == "tilt_5pad_isoviscous":
        default = run_case(**inputs)
        setup = BearingSetup.from_inputs(**inputs)
        cb = inputs["radial_clearance"]
        cold = film_force_at(
            setup, SPEED, default["xj_cb"][0] * cb, default["yj_cb"][0] * cb
        )
        assert_allclose(cold["tilt_angle"], default["tilt_angle"][0], rtol=0, atol=2e-8)
    use_bisection(1e-12, zero=1e-12)
    reference = run_case(**inputs, _tolerance_scale=1e-3, diagnostics=True)
    assert new["convergence"][0]["converged"]
    assert reference["convergence"][0]["converged"]
    assert_allclose(
        new["tilt_angle"][0], reference["tilt_angle"][0], rtol=0, atol=1e-10
    )
    assert _matrix_gaps(new, reference) < 1e-6


@pytest.mark.parametrize(
    "changes", AMBIENT_CASES, ids=[c["operating_type"] for c in AMBIENT_CASES]
)
def test_unloaded_pads_at_a_high_ambient_pressure(changes, use_bisection):
    """Unloaded pads of the high-ambient-pressure and axial-flow operations.

    Their film carries the uniform ambient pressure over a range of tilts
    (a round-off moment that is not flagged as a flat film). Reference: the
    old bisection run to 1e-12 rad with that round-off counted as zero, both
    solvers at 1e-3 of the default tolerances: tilts agree to 1e-10 rad,
    coefficients to 1e-6.
    """
    inputs = _inputs("tilt_5pad_isoviscous", **changes)
    new = run_case(**inputs, _tolerance_scale=1e-3, diagnostics=True)
    use_bisection(1e-12, zero=1e-12)
    reference = run_case(**inputs, _tolerance_scale=1e-3, diagnostics=True)
    assert new["convergence"][0]["converged"]
    assert reference["convergence"][0]["converged"]
    assert_allclose(
        new["tilt_angle"][0], reference["tilt_angle"][0], rtol=0, atol=1e-10
    )
    assert _matrix_gaps(new, reference) < 1e-6


@pytest.mark.parametrize("case", ["tilt_5pad_isoviscous", "tilt_5pad_pivot"])
def test_default_tilts_agree_with_the_old_bisection(case, use_bisection, monkeypatch):
    """At the default tolerances, against the earlier engine.

    Reference: the old bisection to its 1e-8 rad. The tilts solved at the
    same journal positions (the equilibrium, and positions moved by 5 % and
    20 % of the clearance) agree within 2e-8 rad; the full equilibrium solve
    too, with at most half the Reynolds solves.
    """
    inputs = _inputs(case)
    cb = inputs["radial_clearance"]
    setup = BearingSetup.from_inputs(**inputs)
    counts = _counted(monkeypatch)
    new = run_case(**inputs)
    new_solves = counts["press"]
    positions = [
        (new["xj_cb"][0] * cb + dx * cb, new["yj_cb"][0] * cb + dy * cb)
        for dx, dy in ((0.0, 0.0), (0.05, 0.0), (-0.1, 0.2))
    ]
    new_tilts = [film_force_at(setup, SPEED, x, y)["tilt_angle"] for x, y in positions]

    use_bisection(1e-8)
    counts["press"] = 0
    old = run_case(**inputs)
    old_solves = counts["press"]
    old_tilts = [film_force_at(setup, SPEED, x, y)["tilt_angle"] for x, y in positions]

    assert_allclose(new["tilt_angle"][0], old["tilt_angle"][0], rtol=0, atol=2e-8)
    for new_tilt, old_tilt in zip(new_tilts, old_tilts, strict=True):
        assert_allclose(new_tilt, old_tilt, rtol=0, atol=2e-8)
    assert new_solves <= 0.5 * old_solves


@pytest.mark.parametrize("case", ["tilt_5pad_isoviscous", "tilt_5pad_spray"])
def test_pad_moment_is_balanced_at_the_solution(case):
    """Each pad's tilt meets the moment or the tilt tolerance.

    References: (a) for pressures at or above zero over a 60 deg pad, the
    pressure integral is at most ``|F_pad| / cos(30 deg)``, so the moment
    tolerance ``1e-9 (R + t) integral |p| dA`` is below
    ``1.16e-9 (R + t) |F_pad|``; (b) a tilt within 1e-10 rad of the root has
    the moment change sign between the tilts 1e-10 rad below and above it
    (the moment decreases with the tilt), evaluated with the tilts held;
    (c) a pad that carries no pressure has no moment. Checked at the
    equilibrium and at a position 10 % of the clearance away.
    """
    inputs = _inputs(case)
    cb = inputs["radial_clearance"]
    out = run_case(**inputs, diagnostics=True)
    assert out["convergence"][0]["loops"]["tilt"]["residual"] <= 1.0
    setup = BearingSetup.from_inputs(**inputs)
    lever = 0.5 * inputs["journal_diameter"] + inputs["pad_thickness"]
    for dx in (0.0, 0.1):
        x, y = (out["xj_cb"][0] + dx) * cb, out["yj_cb"][0] * cb
        solved = film_force_at(setup, SPEED, x, y)
        assert solved["converged"]
        assert np.all(solved["nodal_pressure"] >= 0.0)
        tilts = solved["tilt_angle"]
        below = film_force_at(setup, SPEED, x, y, tilts=tilts - TILT_ERROR)
        above = film_force_at(setup, SPEED, x, y, tilts=tilts + TILT_ERROR)
        pad_force = np.hypot(solved["fx_pad"], solved["fy_pad"])
        moment = solved["moment_pad"]
        within_moment = np.abs(moment) <= 1.16e-9 * lever * pad_force
        bracketed = (below["moment_pad"] >= 0.0) & (above["moment_pad"] <= 0.0)
        assert np.all(within_moment | bracketed)
        unloaded = pad_force == 0.0
        assert np.any(unloaded)  # an unloaded pad, at the edge of its flat range
        assert np.all(moment[unloaded] == 0.0)
        assert np.all(above["moment_pad"][unloaded] < 0.0)  # it is at the edge


def test_warm_start_slope_is_carried_to_the_next_speed():
    """A warm-started speed starts every pad search from the previous slope."""
    setup = BearingSetup.from_inputs(**_inputs("tilt_5pad_isoviscous"))
    _, warm = sweep.solve_case(setup, SPEED, return_warm=True)
    assert warm.tilt_slope is not None
    loaded = np.isfinite(warm.tilt_slope)
    assert np.all(warm.tilt_slope[loaded] < 0.0)
    g = {}
    warm.apply(g)
    assert np.all(g["_tilt_memory"]["known"])
    assert_allclose(g["_tilt_memory"]["slope"], warm.tilt_slope)
    g["_tilt_memory"]["slope"][:] = 0.0  # a copy, not the WarmStart's array
    assert np.all(warm.tilt_slope[loaded] < 0.0)


def _solve_pert_pressure_refactor(
    mesh,
    pad_index,
    film_onset,
    pads,
    vis_effect_3d,
    h_n,
    cavitated,
    element_source_all,
    helpers,
):
    """The perturbation solve of the earlier engine: one LU per source."""
    p = pad_index
    oil_seal = abs(pads.arc_length_rad[0] - 2.0 * np.pi) < 1.0e-6
    total_bc, press_bc_index, prescribed = coeff.press_bc_pert(
        mesh, pad_index, film_onset, pads, cavitated
    )
    gamma, g = coeff.gamma_g_pert(mesh, pad_index, pads, h_n, vis_effect_3d)
    matrix, column = helpers.zero_pressure_system(
        mesh.dim_xz, mesh.total_nodes, 2 * mesh.bandwidth - 1
    )
    e_index = np.ascontiguousarray(mesh.e_index, dtype=np.int64)
    nodes = [
        np.ascontiguousarray(getattr(mesh, f"node_{c}"), dtype=np.int64) for c in "ijkl"
    ]
    elem = e_index[: int(mesh.total_elements)]
    ni, nj, nk, nl = (n[elem] for n in nodes)
    g_e = (g[ni] + g[nj] + g[nk] + g[nl]) / 4.0
    theta_e = (
        mesh.x_rad[p, ni] + mesh.x_rad[p, nj] + mesh.x_rad[p, nk] + mesh.x_rad[p, nl]
    ) / 4.0
    q = np.ascontiguousarray(
        element_source_all(elem, ni, nj, nk, nl, gamma, g_e, theta_e, oil_seal),
        dtype=np.float64,
    )
    coeff.pert_press_assemble_all_jit(
        p,
        int(mesh.total_elements),
        e_index,
        *nodes,
        np.ascontiguousarray(gamma, dtype=np.float64),
        np.ascontiguousarray(h_n, dtype=np.float64),
        np.ascontiguousarray(mesh.e_length, dtype=np.float64),
        np.ascontiguousarray(mesh.e_width, dtype=np.float64),
        q,
        bool(oil_seal),
        int(mesh.bandwidth),
        matrix,
        column,
    )
    matrix, column = helpers.include_press(
        matrix,
        column,
        mesh.bandwidth,
        total_bc,
        press_bc_index,
        prescribed,
        mesh.total_nodes,
    )
    a, a_lower, index1, _ = banded.lu_factor(matrix, mesh.total_nodes, mesh.bandwidth)
    column = banded.lu_solve(
        a, mesh.total_nodes, mesh.bandwidth, a_lower, index1, column
    )
    out = np.zeros(mesh.dim_xz)
    for i in range(mesh.total_nodes):
        node = int(mesh.n_index[i])
        out[node] = column[node]
    return out


def _random_source(seed):
    rng = np.random.default_rng(seed)

    def source(elem, ni, nj, nk, nl, gamma, g_e, theta_e, oil_seal):
        return rng.standard_normal(elem.size) * (1.0 + gamma[ni])

    return source


@pytest.mark.parametrize(
    "case",
    [
        "tilt_5pad_isoviscous",
        "fixed_isoviscous",
        "fixed_isoviscous_starved",
        "fixed_pdam_bt5",
        "fixed_adiabatic_highpamb",
    ],
)
def test_perturbation_pressures_equal_the_refactor_per_rhs_path(case, monkeypatch):
    """Every perturbation pressure of a case equals one LU per source, to 1e-12.

    Each call of a case (stiffness and damping, every pad, every Newton
    step) is repeated when it is made with one assembly and factorisation
    per source, plus four random sources on the same matrix.
    """
    original = coeff._solve_pert_pressures
    sizes = []

    def check(pressure, reference):
        assert_allclose(
            pressure, reference, rtol=1e-12, atol=1e-12 * np.max(np.abs(reference))
        )

    def spy(*args):
        result = original(*args)
        *state, sources, helpers = args
        sizes.append(len(sources))
        for source, pressure in zip(sources, result, strict=True):
            check(pressure, _solve_pert_pressure_refactor(*state, source, helpers))
        combined = original(*state, [_random_source(s) for s in range(4)], helpers)
        for seed, pressure in enumerate(combined):
            reference = _solve_pert_pressure_refactor(
                *state, _random_source(seed), helpers
            )
            check(pressure, reference)
        return result

    monkeypatch.setattr(coeff, "_solve_pert_pressures", spy)
    run_case(**_inputs(case))
    assert set(sizes) == ({3} if case.startswith("tilt_") else {2})


@pytest.mark.parametrize(
    "case",
    [
        "fixed_isoviscous",
        "fixed_pdam_bt5",
        "tilt_5pad_isoviscous",
        "fixed_starved_m1",
        "fixed_adiabatic_highpamb",
    ],
)
def test_full_cavitation_pressure_equals_the_reynolds_solve(case):
    """``press_full_cavitation`` equals ``press`` with the pad flagged, exactly."""
    setup = BearingSetup.from_inputs(**_inputs(case))
    _, g, _ = sweep._solve(setup, SPEED)
    rng = np.random.default_rng(7)
    for pad in range(g["total_pads"]):
        full = np.zeros(g["total_pads"], dtype=bool)
        full[pad] = True
        start = rng.standard_normal(g["nodal_pressure"].shape) * 1e5
        cavitated = rng.random(g["nodal_pressure"].shape) < 0.3
        common = dict(
            mesh=g["mesh"],
            operating=g["operating"],
            pad_index=pad,
            pads=g["pads"],
            h_n=g["h_n"],
            h_min=g["h_min"],
            x_hmin=g["x_hmin"],
            nodal_pressure=start.copy(),
            dpdx_n=g["dpdx_n"].copy(),
            dpdz_n=g["dpdz_n"].copy(),
            cavitated=cavitated,
        )
        expected = prs.press(
            vis_effect_3d=g["vis_effect_3d"],
            film_onset=g["film_onset"],
            full_cavitate=full,
            **dict(common, nodal_pressure=start.copy()),
        )
        actual = prs.press_full_cavitation(**common)
        for a, e in zip(actual, expected, strict=True):
            assert_allclose(a, e, rtol=0, atol=0)


def test_fully_cavitated_pads_need_no_solve(monkeypatch):
    """The plain journal's upper land is fully cavitated: one solve per film."""
    inputs = _inputs("fixed_isoviscous")
    counts = _counted(monkeypatch)
    out = run_case(**inputs, diagnostics=True)
    loops = out["convergence"][0]["loops"]
    film_solves = loops["newton"]["iterations"]
    # two lands, one of them fully cavitated at every Newton step
    assert counts["press"] == film_solves
    assert out["convergence"][0]["reynolds_solves"] == film_solves


def _press_bc_python(mesh, operating, pad_index, pads, film_onset):
    """``press_bc`` of the earlier engine (vectorised numpy, ``np.select``)."""
    press_bc_index = np.zeros(mesh.dim_xz, dtype=np.int64)
    prescribed_press = np.zeros(mesh.dim_xz, dtype=float)
    p = pad_index
    is_360 = abs(pads.arc_length_rad[0] - 2.0 * np.pi) < 1.0e-6
    idx = np.asarray(mesh.n_index[: mesh.total_nodes], dtype=np.int64)
    z = mesh.z[p, idx]
    x = mesh.x[p, idx]
    on_edge1 = np.abs(z) < 1.0e-6
    on_edge2 = ~on_edge1 & (np.abs(z - pads.axial_length[p]) < 1.0e-6)
    before_onset = (
        ~on_edge1 & ~on_edge2 & (idx < (film_onset[p] + 1) * (mesh.total_e_z_film + 1))
    )
    on_te = (
        ~on_edge1
        & ~on_edge2
        & ~before_onset
        & (np.abs(x - pads.pad_length[p]) < 1.0e-6)
    )
    le_te = (before_onset | on_te) & (not is_360)
    selected = on_edge1 | on_edge2 | le_te
    if operating.operating_type in ("axial_flow", "high_ambient_pressure"):
        edge1_press = operating.ambient_press1
        edge2_press = operating.ambient_press2
        le_te_press = (
            z * (operating.ambient_press2 - operating.ambient_press1)
        ) / pads.axial_length[p] + operating.ambient_press1
    else:
        flooded = operating.operating_type == "regular_flooded"
        edge1_press = 0.0 if flooded else operating.ambient_press1
        edge2_press = 0.0 if flooded else operating.ambient_press2
        le_te_press = np.zeros_like(z)
    values = np.select(
        [on_edge1, on_edge2, le_te],
        [np.full_like(z, edge1_press), np.full_like(z, edge2_press), le_te_press],
    )
    total = int(np.count_nonzero(selected))
    press_bc_index[:total] = idx[selected]
    prescribed_press[:total] = values[selected]
    return total, press_bc_index, prescribed_press


def _random_film_mesh(rng, total_pads=3):
    """A structured pad mesh with shuffled node numbers and node scan."""
    nx, nz = rng.integers(4, 12), rng.integers(3, 9)
    dim_xz = (nx + 1) * (nz + 1)
    pad_length = rng.uniform(0.02, 0.2, total_pads)
    axial_length = rng.uniform(0.02, 0.2, total_pads)
    x = np.zeros((total_pads, dim_xz))
    z = np.zeros((total_pads, dim_xz))
    for p in range(total_pads):
        xs = np.linspace(0.0, pad_length[p], nx + 1)
        zs = np.linspace(0.0, axial_length[p], nz + 1)
        x[p] = np.repeat(xs, nz + 1)
        z[p] = np.tile(zs, nx + 1)
    n_index = rng.permutation(dim_xz)
    mesh = SimpleNamespace(
        x=x,
        z=z,
        n_index=n_index,
        total_nodes=dim_xz,
        dim_xz=dim_xz,
        total_e_z_film=int(nz),
        total_e_x_film=int(nx),
    )
    return mesh, pad_length, axial_length


@pytest.mark.parametrize("seed", range(6))
def test_press_bc_kernel_equals_the_python_version(seed):
    rng = np.random.default_rng(seed)
    mesh, pad_length, axial_length = _random_film_mesh(rng)
    for arc in (np.radians(60.0), 2.0 * np.pi):
        pads = SimpleNamespace(
            arc_length_rad=np.full(3, arc),
            axial_length=axial_length,
            pad_length=pad_length,
        )
        for operating_type in (
            "regular_flooded",
            "starved_condition_even",
            "axial_flow",
            "high_ambient_pressure",
        ):
            operating = SimpleNamespace(
                operating_type=operating_type,
                ambient_press1=rng.uniform(-1e5, 5e5),
                ambient_press2=rng.uniform(-1e5, 5e5),
            )
            film_onset = rng.integers(0, mesh.total_e_x_film, 3)
            for pad in range(3):
                expected = _press_bc_python(mesh, operating, pad, pads, film_onset)
                actual = prs.press_bc(mesh, operating, pad, pads, film_onset)
                assert actual[0] == expected[0]
                assert np.array_equal(actual[1], expected[1])
                assert_allclose(actual[2], expected[2], rtol=1e-14, atol=0)


def _postprocess_python(
    mesh, operating, p, column, clamped, full, h_n, h_min, x_hmin, x, pressure, cav
):
    """The end of ``press`` of the earlier engine (per-node Python loops)."""
    cav = np.array(cav, dtype=bool, copy=True)
    cav[p, :] = False
    for i in range(mesh.total_nodes):
        node = int(mesh.n_index[i])
        pressure[p, node] = column[node]
        cav[p, node] = clamped[node]
    if full:
        for i in range(mesh.total_nodes):
            node = int(mesh.n_index[i])
            pressure[p, node] = operating.press_cavitate
            cav[p, node] = True
    if operating.operating_type == "regular_flooded":
        for i in range(mesh.total_nodes - mesh.total_e_z_film - 1):
            node = int(mesh.n_index[i])
            if (
                h_n[p, node] > h_n[p, node + mesh.total_e_z_film + 1]
                and h_n[p, node] > h_min[p]
                and x[p, node] > x_hmin[p]
                and operating.bearing_type != "pressure_dam"
            ):
                pressure[p, node] = operating.press_cavitate
                cav[p, node] = True
    return pressure, cav


@pytest.mark.parametrize("seed", range(6))
def test_press_postprocess_kernel_equals_the_python_version(seed):
    rng = np.random.default_rng(100 + seed)
    mesh, _, _ = _random_film_mesh(rng)
    # the divergent-clearance rule reads the next node of the circumferential
    # line, so it needs the scan order of the mesh generator
    mesh.n_index = np.arange(mesh.dim_xz)
    shape = (3, mesh.dim_xz)
    for operating_type, bearing_type in (
        ("regular_flooded", "fixed_geometry"),
        ("regular_flooded", "pressure_dam"),
        ("starved_condition_even", "conventional_tilting_pad"),
        ("high_ambient_pressure", "conventional_tilting_pad"),
    ):
        operating = SimpleNamespace(
            operating_type=operating_type,
            bearing_type=bearing_type,
            press_cavitate=rng.uniform(-5e4, 0.0),
        )
        for full in (False, True):
            column = rng.standard_normal(mesh.dim_xz) * 1e6
            clamped = rng.random(mesh.dim_xz) < 0.4
            h_n = rng.uniform(1e-5, 1e-4, shape)
            h_min = rng.uniform(1e-5, 6e-5, 3)
            x_hmin = rng.uniform(0.0, 0.1, 3)
            pressure = rng.standard_normal(shape)
            cav = rng.random(shape) < 0.5
            pad = int(rng.integers(0, 3))
            expected = _postprocess_python(
                mesh,
                operating,
                pad,
                column,
                clamped,
                full,
                h_n,
                h_min,
                x_hmin,
                mesh.x,
                pressure.copy(),
                cav,
            )
            actual = prs._postprocess(
                mesh,
                operating,
                pad,
                column,
                clamped,
                full,
                h_n,
                h_min,
                x_hmin,
                mesh.x,
                pressure.copy(),
                cav,
            )
            assert_allclose(actual[0], expected[0], rtol=1e-14, atol=0)
            assert np.array_equal(actual[1], expected[1])


def _element_adiab_python(kx_e, kz_e, mx_e, mz_e, q_e, l_e, w_e):
    """``thermal.element_adiab`` (the element of the adiabatic equation)."""
    e = np.zeros((4, 4))
    kxw, kzl, mxw, mzl = kx_e * w_e, kz_e * l_e, mx_e * w_e, mz_e * l_e
    # fmt: off
    e[0, 0] = kxw / (3.0 * l_e) + kzl / (3.0 * w_e) - (-mxw / 6.0) - (-mzl / 6.0)
    e[0, 1] = -kxw / (3.0 * l_e) + kzl / (6.0 * w_e) - (mxw / 6.0) - (-mzl / 12.0)
    e[0, 2] = -kxw / (6.0 * l_e) - kzl / (6.0 * w_e) - (mxw / 12.0) - (mzl / 12.0)
    e[0, 3] = kxw / (6.0 * l_e) - kzl / (3.0 * w_e) - (-mxw / 12.0) - (mzl / 6.0)
    e[1, 0] = -kxw / (3.0 * l_e) + kzl / (6.0 * w_e) - (-mxw / 6.0) - (-mzl / 12.0)
    e[1, 1] = kxw / (3.0 * l_e) + kzl / (3.0 * w_e) - (mxw / 6.0) - (-mzl / 6.0)
    e[1, 2] = kxw / (6.0 * l_e) - kzl / (3.0 * w_e) - (mxw / 12.0) - (mzl / 6.0)
    e[1, 3] = -kxw / (6.0 * l_e) - kzl / (6.0 * w_e) - (-mxw / 12.0) - (mzl / 12.0)
    e[2, 0] = -kxw / (6.0 * l_e) - kzl / (6.0 * w_e) - (-mxw / 12.0) - (-mzl / 12.0)
    e[2, 1] = kxw / (6.0 * l_e) - kzl / (3.0 * w_e) - (mxw / 12.0) - (-mzl / 6.0)
    e[2, 2] = kxw / (3.0 * l_e) + kzl / (3.0 * w_e) - (mxw / 6.0) - (mzl / 6.0)
    e[2, 3] = -kxw / (3.0 * l_e) + kzl / (6.0 * w_e) - (-mxw / 6.0) - (mzl / 12.0)
    e[3, 0] = kxw / (6.0 * l_e) - kzl / (3.0 * w_e) - (-mxw / 12.0) - (-mzl / 6.0)
    e[3, 1] = -kxw / (6.0 * l_e) - kzl / (6.0 * w_e) - (mxw / 12.0) - (-mzl / 12.0)
    e[3, 2] = -kxw / (3.0 * l_e) + kzl / (6.0 * w_e) - (mxw / 6.0) - (mzl / 12.0)
    e[3, 3] = kxw / (3.0 * l_e) + kzl / (3.0 * w_e) - (-mxw / 6.0) - (mzl / 6.0)
    # fmt: on
    return e, np.full(4, q_e * l_e * w_e / 4.0)


def _adiabatic_assembly_python(pad, mesh, coeffs, pressure_dam, matrix, column):
    """The element loop of ``temp_adiabatic`` of the earlier engine."""
    kx_n, kz_n, mx_n, mz_n, q_n = coeffs
    bw = mesh.bandwidth
    for i in range(mesh.total_elements):
        current = mesh.e_index[i]
        local = [
            mesh.node_i[current],
            mesh.node_j[current],
            mesh.node_k[current],
            mesh.node_l[current],
        ]
        ni, nj, nk, nl = local
        kx_e = (kx_n[ni] + kx_n[nj] + kx_n[nk] + kx_n[nl]) / 4.0
        kz_e = (kz_n[ni] + kz_n[nj] + kz_n[nk] + kz_n[nl]) / 4.0
        mx_e = (mx_n[ni] + mx_n[nj] + mx_n[nk] + mx_n[nl]) / 4.0
        mz_e = (mz_n[ni] + mz_n[nj] + mz_n[nk] + mz_n[nl]) / 4.0
        q_e = (q_n[ni] + q_n[nj] + q_n[nk] + q_n[nl]) / 4.0
        l_e = mesh.e_length[pad, current]
        w_e = mesh.e_width[pad, current]
        if pressure_dam:
            mz_e = 0.0
        e, e_column = _element_adiab_python(kx_e, kz_e, mx_e, mz_e, q_e, l_e, w_e)
        for a in range(4):
            for b in range(4):
                matrix[local[a], local[b] - local[a] + bw - 1] += e[a, b]
            column[local[a]] += e_column[a]
    return matrix, column


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("pressure_dam", [False, True])
def test_adiabatic_assembly_kernel_equals_the_python_version(seed, pressure_dam):
    rng = np.random.default_rng(200 + seed)
    nx, nz = int(rng.integers(3, 10)), int(rng.integers(2, 7))
    dim = (nx + 1) * (nz + 1)
    perm = rng.permutation(dim)  # node numbers
    grid = perm.reshape(nx + 1, nz + 1)
    node_i = grid[:-1, :-1].ravel()
    node_j = grid[1:, :-1].ravel()
    node_k = grid[1:, 1:].ravel()
    node_l = grid[:-1, 1:].ravel()
    total_elements = nx * nz
    e_index = rng.permutation(total_elements)
    corners = np.stack([node_i, node_j, node_k, node_l])
    bandwidth = int(np.max(corners.max(axis=0) - corners.min(axis=0)) + 1)
    pads = 2
    mesh = SimpleNamespace(
        e_index=e_index,
        node_i=node_i,
        node_j=node_j,
        node_k=node_k,
        node_l=node_l,
        e_length=rng.uniform(1e-3, 1e-2, (pads, total_elements)),
        e_width=rng.uniform(1e-3, 1e-2, (pads, total_elements)),
        total_elements=total_elements,
        bandwidth=bandwidth,
    )
    coeffs = [rng.standard_normal(dim) * s for s in (0.13, 0.13, 1e3, 1e2, 1e6)]
    pad = int(rng.integers(0, pads))
    shape = (dim, 2 * bandwidth - 1)
    expected = _adiabatic_assembly_python(
        pad, mesh, coeffs, pressure_dam, np.zeros(shape), np.zeros(dim)
    )
    matrix, column = np.zeros(shape), np.zeros(dim)
    adiab_assemble_all_jit(
        pad,
        total_elements,
        e_index.astype(np.int64),
        node_i.astype(np.int64),
        node_j.astype(np.int64),
        node_k.astype(np.int64),
        node_l.astype(np.int64),
        *coeffs,
        mesh.e_length,
        mesh.e_width,
        pressure_dam,
        bandwidth,
        matrix,
        column,
    )
    assert_allclose(matrix, expected[0], rtol=1e-14, atol=0)
    assert_allclose(column, expected[1], rtol=1e-14, atol=0)


def _trapezoid_python(t, f, start, stop):
    if stop - start < 2:
        return 0.0
    total = 0.0
    for i in range(start + 1, stop):
        total += 0.5 * (t[i] - t[i - 1]) * (f[i] + f[i - 1])
    return total


def _in_track_python(pads, x, z, pad):
    in_pocket = (
        z > pads.axial_length_dam[pad]
        and z < pads.axial_length_dam[pad] + pads.axial_length_track[pad]
        and x < pads.length_track[pad]
    )
    in_dam = (
        x > pads.length_track[pad]
        or z < pads.axial_length_dam[pad]
        or z > pads.axial_length_dam[pad] + pads.axial_length_track[pad]
    )
    if in_pocket:
        return True
    if in_dam:
        return False
    on_circumferential_edge = (
        abs(x - pads.pad_length[pad]) < 1.0e-6
        and z > pads.axial_length_dam[pad]
        and z < pads.axial_length_track[pad] + pads.axial_length_dam[pad]
    )
    on_axial_end = (
        abs(z) < 1.0e-6 or abs(z - pads.axial_length[pad]) < 1.0e-6
    ) and x < pads.length_track[pad]
    return on_circumferential_edge or on_axial_end


def _pde_coeff_adiab_python(pad, mesh, pads, vx, vz, dudy, dwdy, h_n, vis, lube):
    """``thermal.pde_coeff_adiab`` of the earlier engine (per-node Python loop)."""
    kx_n, kz_n, mx_n, mz_n, q_n = (np.zeros(mesh.dim_xz) for _ in range(5))
    t, f1, f2, f3 = (np.zeros(mesh.dim_yf) for _ in range(4))
    limit1 = mesh.total_e_y_trackbl[pad] + mesh.total_e_y_trackcore[pad] + 1
    limit2 = mesh.total_e_y_film + 1
    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        kx_n[node] = -lube.conduct
        kz_n[node] = -lube.conduct
        track = _in_track_python(pads, mesh.x[pad, node], mesh.z[pad, node], pad)
        jlo, depth = (1, 0.0) if track else (limit1, pads.depth_track[pad])
        for j in range(jlo, limit2 + 1):
            m = mesh.match_nodes_xz[node, j - 1]
            t[j - 1] = mesh.y_3d[pad, m] - pads.pad_thickness - depth
            f1[j - 1] = vx[pad, m]
            f2[j - 1] = vz[pad, m]
            f3[j - 1] = vis[pad, m] * (dudy[pad, m] ** 2 + dwdy[pad, m] ** 2) / 1.0
        h = h_n[pad, node]
        mx_n[node] = (
            lube.density * lube.cp * _trapezoid_python(t, f1, jlo - 1, limit2) / h
        )
        mz_n[node] = (
            lube.density * lube.cp * _trapezoid_python(t, f2, jlo - 1, limit2) / h
        )
        q_n[node] = -_trapezoid_python(t, f3, jlo - 1, limit2) / h
    return kx_n, kz_n, mx_n, mz_n, q_n


def _update_vis_adiab_python(
    pad, mesh, pads, lube, speed, temp_3d, vis_3d, vis_avg, h_n
):
    """The node loop of ``thermal.update_vis_adiab`` of the earlier engine."""
    t, f = np.zeros(mesh.dim_yf), np.zeros(mesh.dim_yf)
    limit1 = mesh.total_e_y_trackbl[pad] + mesh.total_e_y_trackcore[pad] + 1
    limit2 = mesh.total_e_y_film + 1
    re_max = {True: 0.0, False: 0.0}
    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        for j in range(1, limit2 + 1):
            m = mesh.match_nodes_xz[node, j - 1]
            vis_3d[pad, m] = lube.viscosity_at(temp_3d[pad, m])
        track = _in_track_python(pads, mesh.x[pad, node], mesh.z[pad, node], pad)
        jlo, depth = (1, 0.0) if track else (limit1, pads.depth_track[pad])
        for j in range(jlo, limit2 + 1):
            m = mesh.match_nodes_xz[node, j - 1]
            t[j - 1] = mesh.y_3d[pad, m] - pads.pad_thickness - depth
            f[j - 1] = vis_3d[pad, m]
        h = h_n[pad, node]
        vis_avg[pad, node] = _trapezoid_python(t, f, jlo - 1, limit2) / h
        re = lube.density * speed * h / vis_avg[pad, node]
        re_max[track] = max(re_max[track], re)
    return vis_3d, vis_avg, re_max[True], re_max[False]


def _random_energy_case(rng, total_pads=2):
    """A pad film with a pocket whose edges fall on mesh lines, random fields."""
    nx, nz, ey = 8, 8, int(rng.integers(4, 9))
    dim_xz = (nx + 1) * (nz + 1)
    dim_yf = ey + 1
    x = np.tile(np.repeat(np.linspace(0.0, 0.5, nx + 1), nz + 1), (total_pads, 1))
    z = np.tile(np.tile(np.linspace(0.0, 1.0, nz + 1), nx + 1), (total_pads, 1))
    match = np.arange(dim_xz * dim_yf).reshape(dim_xz, dim_yf)
    dim_3d = dim_xz * dim_yf
    thickness = 0.02
    depth = rng.uniform(1e-5, 1e-4, total_pads)
    y_3d = np.zeros((total_pads, dim_3d))
    for p in range(total_pads):
        steps = rng.uniform(1e-6, 1e-5, (dim_xz, dim_yf))
        steps[:, 0] = 0.0
        y_3d[p] = (thickness + np.cumsum(steps, axis=1)).ravel()
    mesh = SimpleNamespace(
        x=x,
        z=z,
        y_3d=y_3d,
        n_index=np.arange(dim_xz),
        match_nodes_xz=match,
        total_nodes=dim_xz,
        dim_xz=dim_xz,
        dim_yf=dim_yf,
        total_e_y_film=ey,
        total_e_y_trackbl=np.full(total_pads, 1),
        total_e_y_trackcore=np.full(total_pads, int(rng.integers(0, ey - 2))),
    )
    pads = SimpleNamespace(
        pad_thickness=thickness,
        depth_track=depth,
        pad_length=np.full(total_pads, 0.5),
        axial_length=np.full(total_pads, 1.0),
        axial_length_dam=np.full(total_pads, 0.25),
        axial_length_track=np.full(total_pads, 0.5),
        length_track=np.full(total_pads, rng.choice([0.375, 0.5])),
    )
    fields = {
        name: rng.standard_normal((total_pads, dim_3d)) * scale
        for name, scale in (("vx", 30.0), ("vz", 3.0), ("dudy", 1e6), ("dwdy", 1e5))
    }
    fields["vis"] = rng.uniform(1e-3, 3e-2, (total_pads, dim_3d))
    fields["temp"] = rng.uniform(310.0, 390.0, (total_pads, dim_3d))
    fields["h_n"] = rng.uniform(2e-5, 2e-4, (total_pads, dim_xz))
    return mesh, pads, fields


@pytest.mark.parametrize("seed", range(5))
def test_adiabatic_coefficient_kernel_equals_the_python_version(seed):
    from ross.bearings.fluid_film import thermal
    from ross.bearings.fluid_film.state import Lubricant

    rng = np.random.default_rng(300 + seed)
    mesh, pads, fd = _random_energy_case(rng)
    lube = Lubricant(0.0277, 0.0047, 313.15, 373.15, 870.0, 1950.0, 0.13)
    # the edge classification is exercised: pocket, dam and edge nodes
    kinds = {
        (_in_track_python(pads, mesh.x[0, n], mesh.z[0, n], 0), 0.25 < mesh.z[0, n])
        for n in range(mesh.dim_xz)
    }
    assert len(kinds) >= 3
    for pad in range(2):
        args = (fd["vx"], fd["vz"], fd["dudy"], fd["dwdy"], fd["h_n"], fd["vis"])
        expected = _pde_coeff_adiab_python(pad, mesh, pads, *args, lube)
        actual = thermal.pde_coeff_adiab(
            pad,
            mesh,
            pads,
            fd["vx"],
            fd["vz"],
            fd["dudy"],
            fd["dwdy"],
            fd["h_n"],
            fd["vis"],
            lube,
        )
        for a, e in zip(actual, expected, strict=True):
            assert_allclose(a, e, rtol=1e-14, atol=0)


@pytest.mark.parametrize("seed", range(5))
def test_adiabatic_viscosity_update_equals_the_python_version(seed):
    from ross.bearings.fluid_film import thermal
    from ross.bearings.fluid_film.state import Lubricant, Turbulence

    # a transition range that the random Reynolds numbers (50-5,000) cross
    turbulence = Turbulence(200.0, 2000.0, 10.7, 0.4, 1.0)

    rng = np.random.default_rng(400 + seed)
    mesh, pads, fd = _random_energy_case(rng)
    lube = Lubricant(0.0277, 0.0047, 313.15, 373.15, 870.0, 1950.0, 0.13)
    speed = rng.uniform(5.0, 80.0)
    for pad in range(2):
        vis_3d, vis_avg, re_track, re_dam = _update_vis_adiab_python(
            pad,
            mesh,
            pads,
            lube,
            speed,
            fd["temp"],
            fd["vis"].copy(),
            np.zeros((2, mesh.dim_xz)),
            fd["h_n"],
        )
        regimes = [np.zeros(2, dtype=np.int64) for _ in range(2)]
        scales = [np.zeros(2) for _ in range(2)]
        out = thermal.update_vis_adiab(
            pad,
            mesh,
            regimes[0],
            regimes[1],
            pads,
            lube,
            speed,
            fd["temp"],
            fd["vis"].copy(),
            np.zeros((2, mesh.dim_xz)),
            fd["h_n"],
            scales[0],
            scales[1],
            turbulence,
        )
        assert_allclose(out[0], vis_3d, rtol=1e-14, atol=0)
        assert_allclose(out[1], vis_avg, rtol=1e-14, atol=0)
        # the regimes follow the largest Reynolds numbers of track and dam
        expected_regimes = [np.zeros(2, dtype=np.int64) for _ in range(2)]
        expected_scales = [np.zeros(2) for _ in range(2)]
        for k, re in enumerate((re_track, re_dam)):
            thermal._update_flow_regime(
                expected_regimes[k], expected_scales[k], pad, re, turbulence
            )
        assert np.array_equal(out[2], expected_regimes[0])
        assert np.array_equal(out[3], expected_regimes[1])
        assert_allclose(out[4], expected_scales[0], rtol=1e-14, atol=0)
        assert_allclose(out[5], expected_scales[1], rtol=1e-14, atol=0)


def test_adiabatic_temperature_expansion_equals_the_python_version():
    from ross.bearings.fluid_film import thermal

    rng = np.random.default_rng(500)
    mesh, _, fd = _random_energy_case(rng)
    temp_adiab = rng.uniform(310.0, 390.0, (2, mesh.dim_xz))
    expected = fd["temp"].copy()
    for i in range(mesh.total_nodes):
        node = mesh.n_index[i]
        for j in range(mesh.total_e_y_film + 1):
            expected[1, mesh.match_nodes_xz[node, j]] = temp_adiab[1, node]
    actual = thermal.temp_3d_adiab(1, mesh, temp_adiab, fd["temp"].copy())
    assert np.array_equal(actual, expected)
