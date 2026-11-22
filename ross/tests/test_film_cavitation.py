"""Tests of the mass-conserving cavitation solver.

The references are written in this file and do not use the solver:

- a one-dimensional slider with the film ``h = h0 (1 + a cos(2 pi x / Lx))``
  fed at both ends at ``p_s``, solved with the Jakobsson-Floberg-Olsson (JFO)
  conditions by quadrature and root finding: rupture where ``p = p_cav`` and
  ``dp/dx = 0``, cavitated film fraction ``theta = h_r / h``, reformation with
  the flux balance ``A dp/dx = U h (1 - theta^-) / 2``;
- the same slider with a starved inlet at ``p_cav`` (film fraction
  ``theta_in``), which reforms, ruptures and stays cavitated to the exit;
- the full-film analytic pressure of the same film and an independent
  full-film finite-volume solve assembled here.

Mesh study: 321 nodes give the
fronts within 0.5 % of ``Lx`` and the pressure within 0.25 % of its range for
``a`` 0.3 to 0.7, fed and starved; 161 nodes do not (1.04 % for one starved
case).
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
from scipy.optimize import brentq

from ross.bearings.film import ConvergenceWarning
from ross.bearings.film.cavitation import CavitationGrid, solve_elrod_adams

LX, H0, MU, U = 0.1, 50e-6, 0.03, 10.0
GRID = 321
JFO_CASES = [(0.3, 0.5e6), (0.5, 2.0e6), (0.7, 4.0e6)]
GAUSS_X, GAUSS_W = leggauss(16)


def film(x, a):
    """Film thickness of the test slider, m."""
    return H0 * (1.0 + a * np.cos(2.0 * np.pi * x / LX))


def pressure_rise(x1, x2, h_flux, a):
    """Return ``int_x1^x2 6 mu U (h - h_flux) / h^3 dx`` (full-film pressure rise).

    Composite 16-point Gauss-Legendre on panels of at most ``Lx / 64``; the
    integrand is analytic on the real axis, so the rule is exact to rounding.
    """
    panels = max(1, int(np.ceil(abs(x2 - x1) / (LX / 64))))
    edges = np.linspace(x1, x2, panels + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])[:, None]
    half = 0.5 * (edges[1:] - edges[:-1])[:, None]
    x = mid + half * GAUSS_X
    h = film(x, a)
    return float(np.sum(half * GAUSS_W * 6.0 * MU * U * (h - h_flux) / h**3))


def cumulative_rise(x, x0, h_flux, a):
    """Return ``pressure_rise(x0, x_i)`` for increasing nodes ``x_i >= x0``."""
    steps = [
        pressure_rise(lo, hi, h_flux, a)
        for lo, hi in zip(np.r_[x0, x[:-1]], x, strict=True)
    ]
    return np.cumsum(steps)


def jfo_fronts(a, p_s, p_cav):
    """Rupture, reformation, film thickness at rupture and peak of the fed slider."""
    x_r = brentq(
        lambda x: p_s - p_cav + pressure_rise(0.0, x, film(x, a), a),
        0.5 * LX,
        LX,
        xtol=1e-15,
        rtol=1e-15,
    )
    h_r = film(x_r, a)
    x_f = brentq(
        lambda x: pressure_rise(x, LX, h_r, a) - (p_s - p_cav),
        x_r,
        LX,
        xtol=1e-15,
        rtol=1e-15,
    )
    peak = p_s + pressure_rise(0.0, LX - x_r, h_r, a)
    return x_r, x_f, h_r, peak


def jfo_solution(x, a, p_s, p_cav):
    """JFO pressure and film fraction of the fed slider at the nodes ``x``."""
    x_r, x_f, h_r, _ = jfo_fronts(a, p_s, p_cav)
    p = np.full(x.shape, p_cav)
    theta = np.minimum(h_r / film(x, a), 1.0)
    first = x <= x_r
    p[first] = p_s + cumulative_rise(x[first], 0.0, h_r, a)
    theta[first] = 1.0
    last = x >= x_f
    p[last] = p_cav + cumulative_rise(x[last], x_f, h_r, a)
    theta[last] = 1.0
    return p, theta


def starved_solution(x, a, theta_in, p_cav):
    """Starved slider: inlet at ``p_cav`` with film fraction ``theta_in``.

    The inflow ``U theta_in h(0) / 2`` is carried by a cavitated film until
    the film reforms at ``x_s``; the full film ruptures at ``x_r`` where
    ``h = theta_in h(0)`` with ``p = p_cav`` and ``dp/dx = 0``, which fixes
    ``x_s`` through ``int_xs^xr (h - h_q) / h^3 dx = 0``.
    """
    h_q = theta_in * film(0.0, a)
    x_peak = LX / (2.0 * np.pi) * np.arccos((h_q / H0 - 1.0) / a)
    x_r = LX - x_peak
    x_s = brentq(
        lambda s: pressure_rise(s, x_r, h_q, a), 0.0, x_peak, xtol=1e-15, rtol=1e-15
    )
    p = np.full(x.shape, p_cav)
    theta = np.minimum(h_q / film(x, a), 1.0)
    full = (x >= x_s) & (x <= x_r)
    p[full] = p_cav + cumulative_rise(x[full], x_s, h_q, a)
    theta[full] = 1.0
    peak = p_cav + pressure_rise(x_s, x_peak, h_q, a)
    return p, theta, x_s, x_r, h_q, peak


def starved_fraction(a, f):
    """Inlet film fraction a fraction ``f`` of the way across the starved range.

    The inlet is starved (the film reforms downstream of it) when
    ``h_min < theta_in h(0) < h_r``, with ``h_r`` the rupture film of the
    slider fed at ``p_cav``; above ``h_r`` the inlet floods and spills back.
    """
    x_r = brentq(lambda x: pressure_rise(0.0, x, film(x, a), a), 0.5 * LX, LX)
    h_min = film(0.5 * LX, a)
    return (h_min + f * (film(x_r, a) - h_min)) / film(0.0, a)


def slider(n, a, p_feed, p_exit=None, theta_in=1.0):
    """1-D slider grid with Dirichlet nodes at both ends."""
    x = np.linspace(0.0, LX, n)
    h = film(x, a)
    ends = np.isin(np.arange(n), [0, n - 1])
    p_bc = np.full(n, p_feed if p_exit is None else p_exit)
    p_bc[0] = p_feed
    fraction = np.ones(n)
    fraction[0] = theta_in
    grid = CavitationGrid(x, [0.0], h, h**3 / (12 * MU), U / 2, ends, p_bc, fraction)
    return x, grid


def full_film_fv(x, a, p_s):
    """Full-film finite-volume solve assembled independently (dense)."""
    n = x.size
    h = film(x, a)
    diff = h**3 / (12 * MU)
    a_f = 0.5 * (diff[1:] + diff[:-1]) / np.diff(x)
    c_f = 0.5 * (U / 2) * (h[1:] + h[:-1])
    matrix = np.zeros((n, n))
    rhs = np.zeros(n)
    for i in range(1, n - 1):
        matrix[i, i] = a_f[i] + a_f[i - 1]
        matrix[i, i + 1] = -a_f[i]
        matrix[i, i - 1] = -a_f[i - 1]
        rhs[i] = c_f[i - 1] - c_f[i]
    for i in (0, n - 1):
        matrix[i, i] = 1.0
        rhs[i] = p_s
    return np.linalg.solve(matrix, rhs)


def full_film_supply(a):
    """Supply pressure 1.5 times the full-film suction of the slider, Pa."""
    i2 = pressure_rise(0.0, LX, 0.0, a)
    i3 = pressure_rise(0.0, LX, 1.0, a) - i2
    h_m = -i2 / i3
    x = np.linspace(0.0, LX, 401)
    dip = np.min(cumulative_rise(x[1:], 0.0, h_m, a))
    return 1.5 * abs(dip), h_m


def assert_converged(sol, max_iterations=50):
    """FBNS converged within the cap and the record is consistent."""
    rec = sol.record
    assert rec.converged and rec.reason == "converged"
    assert 1 <= rec.iterations <= max_iterations or rec.residual_history[0] <= 1e-10
    assert len(rec.residual_history) == rec.iterations + 1
    assert rec.final_residual <= 1e-10


def assert_cavitated_mask(sol, grid, p_cav, scale):
    """The mask marks the free nodes with a film fraction below 1.

    Cavitated nodes sit at ``p_cav``; full-film nodes have ``theta = 1``. On
    the test grids the smallest ``1 - theta`` of a cavitated node is 2e-4 and
    the smallest ``p - p_cav`` of a full-film node 4e-5 of the pressure scale.
    """
    free = ~grid.dirichlet
    partial = sol.theta < 1.0 - 1e-6
    assert np.array_equal(sol.cavitated[free], partial[free])
    assert not sol.cavitated[grid.dirichlet].any()
    assert np.all(sol.p[sol.cavitated] - p_cav < 1e-9 * scale)


@pytest.mark.parametrize("a, p_s", JFO_CASES)
def test_jfo_slider(a, p_s):
    p_cav = 0.0
    x, grid = slider(GRID, a, p_s)
    sol = solve_elrod_adams(grid, p_cav)
    assert_converged(sol)
    p_ref, theta_ref = jfo_solution(x, a, p_s, p_cav)
    x_r, x_f, h_r, peak = jfo_fronts(a, p_s, p_cav)
    p = sol.p[:, 0]
    cav = np.flatnonzero(sol.cavitated[:, 0])
    assert np.all(np.diff(cav) == 1)
    assert_cavitated_mask(sol, grid, p_cav, peak)
    assert abs(x[cav[0]] - x_r) < 0.01 * LX
    assert abs(x[cav[-1]] - x_f) < 0.01 * LX
    assert abs(p.max() - peak) < 0.01 * peak
    assert np.max(np.abs(p - p_ref)) < 0.01 * peak
    inner = (x > x_r + 0.05 * LX) & (x < x_f - 0.05 * LX)
    assert_allclose(sol.theta[inner, 0], theta_ref[inner], rtol=0.01)
    assert_allclose(sol.flow_in, U * h_r / 2, rtol=0.01)
    assert sol.mass_balance < 1e-10
    assert_allclose(grid.face_flux(sol.p, sol.theta), sol.flow_in, rtol=1e-10)
    outflow = grid.residual(sol.p, sol.theta)
    assert np.max(np.abs(outflow[1:-1])) < 1e-10 * sol.flow_in
    assert_allclose(outflow[[0, -1]], sol.boundary_flow[[0, -1]], rtol=1e-12)
    assert p.min() > p_cav - 1e-9 * peak and sol.theta.max() < 1.0 + 1e-10


@pytest.mark.parametrize("a, p_s", JFO_CASES)
def test_jfo_convergence_order(a, p_s):
    """Film fraction first order; pressure and flow second order.

    The cavitated zone carries the discrete flux exactly (the face fluxes
    telescope), so the reformed film sees the flow of the first full-film
    region and the pressure keeps the second order of the central full-film
    flux; the film fraction in the cavitated zone is ``Q / (U h_f / 2)`` at the
    downstream face, an O(dx) shift.
    """
    sizes = [161, 321, 641, 1281, 2561]
    x_r, x_f, h_r, peak = jfo_fronts(a, p_s, 0.0)
    e_p, e_theta, e_q = [], [], []
    for n in sizes:
        x, grid = slider(n, a, p_s)
        sol = solve_elrod_adams(grid, 0.0)
        assert_converged(sol)
        assert sol.mass_balance < 1e-10
        p_ref, theta_ref = jfo_solution(x, a, p_s, 0.0)
        inner = (x > x_r + 0.05 * LX) & (x < x_f - 0.05 * LX)
        e_p.append(np.max(np.abs(sol.p[:, 0] - p_ref)) / peak)
        e_theta.append(np.max(np.abs(sol.theta[inner, 0] - theta_ref[inner])))
        e_q.append(abs(sol.flow_in / (U * h_r / 2) - 1.0))
    e_p, e_theta, e_q = map(np.array, (e_p, e_theta, e_q))
    assert_allclose(e_theta[:-1] / e_theta[1:], 2.0, atol=0.4)
    assert_allclose(e_p[:-1] / e_p[1:], 4.0, atol=0.8)
    assert_allclose(e_q[:-1] / e_q[1:], 4.0, atol=0.8)
    order = -np.polyfit(np.log(sizes), np.log(e_p), 1)[0]
    assert_allclose(order, 2.0, atol=0.1)


@pytest.mark.parametrize("a, f", [(0.3, 0.6), (0.5, 0.3), (0.5, 0.9), (0.7, 0.6)])
def test_starved_slider(a, f):
    p_cav = -20e3
    theta_in = starved_fraction(a, f)
    x, grid = slider(GRID, a, p_cav, theta_in=theta_in)
    sol = solve_elrod_adams(grid, p_cav)
    assert_converged(sol)
    p_ref, theta_ref, x_s, x_r, h_q, peak = starved_solution(x, a, theta_in, p_cav)
    p = sol.p[:, 0]
    assert_cavitated_mask(sol, grid, p_cav, peak - p_cav)
    full = np.flatnonzero(~sol.cavitated[1:-1, 0]) + 1
    assert np.all(np.diff(full) == 1)
    assert abs(x[full[0]] - x_s) < 0.01 * LX
    assert abs(x[full[-1]] - x_r) < 0.01 * LX
    assert abs(p.max() - peak) < 0.01 * (peak - p_cav)
    assert np.max(np.abs(p - p_ref)) < 0.01 * (peak - p_cav)
    assert_allclose(
        sol.theta[x < x_s - 0.02 * LX, 0], theta_ref[x < x_s - 0.02 * LX], rtol=0.01
    )
    assert_allclose(sol.flow_in, U * h_q / 2, rtol=1e-3)
    assert sol.mass_balance < 1e-10


@pytest.mark.parametrize("a", [0.3, 0.5, 0.7])
def test_full_film_matches_linear_solve(a):
    p_s, _ = full_film_supply(a)
    for n in (41, 321):
        x, grid = slider(n, a, p_s)
        reference = full_film_fv(x, a, p_s)
        for theta_init, p_init in ((None, None), (0.5, 0.0), (1.0, 0.0)):
            sol = solve_elrod_adams(grid, 0.0, theta_init=theta_init, p_init=p_init)
            assert_converged(sol)
            assert not sol.cavitated.any()
            assert_allclose(sol.theta, 1.0, rtol=0, atol=1e-10)
            assert_allclose(sol.p[:, 0], reference, rtol=0, atol=1e-10 * p_s)
            assert sol.mass_balance < 1e-10


@pytest.mark.parametrize("a", [0.3, 0.5, 0.7])
def test_full_film_second_order(a):
    p_s, h_m = full_film_supply(a)
    errors = []
    for n in (41, 81, 161, 321, 641):
        x, grid = slider(n, a, p_s)
        sol = solve_elrod_adams(grid, 0.0)
        assert_converged(sol)
        analytic = p_s + np.r_[0.0, cumulative_rise(x[1:], 0.0, h_m, a)]
        errors.append(np.max(np.abs(sol.p[:, 0] - analytic)) / np.max(analytic))
        assert_allclose(sol.flow_in, U * h_m / 2, rtol=1e-2)
    errors = np.array(errors)
    assert_allclose(errors[:-1] / errors[1:], 4.0, atol=0.5)


def test_axial_neumann_edges_reproduce_1d():
    a, p_s = 0.5, 2.0e6
    x, grid_1d = slider(GRID, a, p_s)
    sol_1d = solve_elrod_adams(grid_1d, 0.0)
    z = np.array([0.0, 0.01, 0.03, 0.04])
    h = film(x, a)[:, None] * np.ones(z.size)
    ends = np.zeros(h.shape, bool)
    ends[[0, -1], :] = True
    grid = CavitationGrid(x, z, h, h**3 / (12 * MU), U / 2, ends, p_s)
    sol = solve_elrod_adams(grid, 0.0)
    assert_converged(sol)
    for j in range(z.size):
        assert_allclose(sol.p[:, j], sol_1d.p[:, 0], rtol=0, atol=1e-9 * p_s)
        assert_allclose(sol.theta[:, j], sol_1d.theta[:, 0], rtol=0, atol=1e-10)
    assert_allclose(sol.flow_in, 0.04 * sol_1d.flow_in, rtol=1e-10)
    assert sol.mass_balance < 1e-10


@pytest.mark.parametrize("width", [0.05, 0.1, 0.2])
def test_finite_pad_side_leakage(width):
    """Finite pad with ambient axial edges: symmetric, below the 1-D pressure."""
    a, p_s = 0.5, 2.0e6
    x, grid_1d = slider(161, a, p_s)
    sol_1d = solve_elrod_adams(grid_1d, 0.0)
    z = np.linspace(0.0, width, 21)
    h = film(x, a)[:, None] * np.ones(z.size)
    edge = np.zeros(h.shape, bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    p_bc = np.zeros(h.shape)
    p_bc[[0, -1], :] = p_s
    p_bc[:, [0, -1]] = 0.0
    grid = CavitationGrid(x, z, h, h**3 / (12 * MU), U / 2, edge, p_bc)
    sol = solve_elrod_adams(grid, 0.0)
    assert_converged(sol)
    assert sol.mass_balance < 1e-10
    assert sol.cavitated.any()
    assert_allclose(sol.p, sol.p[:, ::-1], rtol=0, atol=1e-10 * p_s)
    assert_allclose(sol.theta, sol.theta[:, ::-1], rtol=0, atol=1e-10)
    assert sol.p.min() > -1e-9 * p_s and sol.theta.max() < 1.0 + 1e-10
    assert np.all(sol.p[:, 1:-1] <= sol_1d.p + 1e-9 * p_s)
    assert sol.p.max() < sol_1d.p.max()
    complementarity = sol.p[~grid.dirichlet] * (1.0 - sol.theta[~grid.dirichlet])
    assert np.max(np.abs(complementarity)) < 1e-6 * p_s


def test_periodic_equals_open_with_repeated_feed():
    a, p_s, n = 0.5, 2.0e6, 160
    x_open, grid_open = slider(n + 1, a, p_s)
    sol_open = solve_elrod_adams(grid_open, 0.0)
    x = x_open[:-1]
    h = film(x, a)
    feed = np.arange(n) == 0
    grid = CavitationGrid(x, [0.0], h, h**3 / (12 * MU), U / 2, feed, p_s, period=LX)
    sol = solve_elrod_adams(grid, 0.0)
    assert_converged(sol)
    assert_allclose(sol.p[:, 0], sol_open.p[:-1, 0], rtol=0, atol=1e-9 * p_s)
    assert_allclose(sol.theta[1:, 0], sol_open.theta[1:-1, 0], rtol=0, atol=1e-10)
    assert_allclose(sol.boundary_flow[0, 0], np.sum(sol_open.boundary_flow), atol=1e-12)
    assert sol.flow_out == 0.0
    assert_allclose(sol.flow_in, 0.0, atol=1e-10 * sol_open.flow_in)


def test_periodic_2d_independent_of_the_first_node():
    """The period closes every node line of a 2-D grid, control widths too.

    A non-uniform ``x`` grid of a journal fed through a hole is solved twice,
    numbered from two different nodes: the wrap gap and the nodes whose
    control volumes straddle it change, the solution does not.
    """
    n_x, n_z, shift = 90, 9, 37
    radius, length = 0.05, 0.04
    period = 2.0 * np.pi * radius
    x = np.linspace(0.0, period, n_x, endpoint=False)
    x += 0.3 * (period / n_x) * np.sin(5.0 * np.arange(n_x))
    z = np.linspace(0.0, length, n_z)

    def solve(x_nodes):
        h = 100e-6 * (1.0 + 0.6 * np.cos(x_nodes / radius - 1.0))
        h = h[:, None] * (1.0 + 0.1 * z / length)
        dirichlet = np.zeros(h.shape, bool)
        dirichlet[:, [0, -1]] = True
        hole = np.isclose(x_nodes, x[0]) | np.isclose(x_nodes, x[0] + period)
        dirichlet[hole, n_z // 2] = True
        p_bc = np.where(dirichlet & hole[:, None], 1e5, 0.0)
        grid = CavitationGrid(
            x_nodes, z, h, h**3 / (12 * MU), 10.0, dirichlet, p_bc, period=period
        )
        return grid, solve_elrod_adams(grid, 0.0)

    grid, sol = solve(x)
    x_rolled = np.r_[x[shift:], x[:shift] + period]
    rolled_grid, rolled = solve(x_rolled)
    assert_converged(sol)
    assert_converged(rolled)
    assert sol.cavitated.any()
    assert_allclose(
        np.roll(rolled_grid.control_area, shift, axis=0), grid.control_area, rtol=1e-12
    )
    peak = sol.p.max()
    assert_allclose(np.roll(rolled.p, shift, axis=0), sol.p, rtol=0, atol=1e-9 * peak)
    assert_allclose(np.roll(rolled.theta, shift, axis=0), sol.theta, rtol=0, atol=1e-9)
    assert sol.mass_balance < 1e-10 and rolled.mass_balance < 1e-10


def journal_grid(n_x, n_z, eps, feed_width=None):
    """360-degree journal fed at ``x = 0`` with ambient axial edges at ``p_cav``.

    ``feed_width`` None is a single feed node (a hole) at mid-width; otherwise a
    groove over that fraction of the width.
    """
    radius, length, clearance, speed = 0.05, 0.04, 100e-6, 20.0
    period = 2.0 * np.pi * radius
    x = np.linspace(0.0, period, n_x, endpoint=False)
    z = np.linspace(0.0, length, n_z)
    h = clearance * (1.0 + eps * np.cos(x / radius))[:, None] * np.ones(n_z)
    dirichlet = np.zeros(h.shape, bool)
    dirichlet[:, [0, -1]] = True
    if feed_width is None:
        feed = np.arange(n_z) == n_z // 2
    else:
        feed = np.abs(z / length - 0.5) <= feed_width / 2 + 1e-12
        feed[[0, -1]] = False
    dirichlet[0, feed] = True
    p_bc = np.zeros(h.shape)
    p_bc[0, feed] = 1e5
    return CavitationGrid(
        x, z, h, h**3 / (12 * MU), speed / 2, dirichlet, p_bc, period=period
    )


@pytest.mark.parametrize("eps", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("n_x, n_z", [(60, 11), (120, 21), (240, 41)])
@pytest.mark.parametrize("feed_width", [None, 0.2, 0.5])
def test_journal_fed_through_hole_or_short_groove(eps, n_x, n_z, feed_width):
    """Regression: these films made the Fischer-Burmeister iteration circle.

    The ambient edges leave ``p - p_cav`` and ``1 - theta`` both near zero;
    the iteration switches to active-set steps. The film is
    symmetric about mid-width, the feed supplies what leaves at the edges and
    the complementarity holds.
    """
    grid = journal_grid(n_x, n_z, eps, feed_width)
    sol = solve_elrod_adams(grid, 0.0)
    assert_converged(sol)
    assert sol.mass_balance < 1e-10
    peak = sol.p.max()
    assert 0.0 < peak and sol.cavitated.any()
    assert_allclose(sol.p, sol.p[:, ::-1], rtol=0, atol=1e-9 * peak)
    assert_allclose(sol.theta, sol.theta[:, ::-1], rtol=0, atol=1e-9)
    assert sol.p.min() > -1e-9 * peak and sol.theta.max() < 1.0 + 1e-10
    free = ~grid.dirichlet
    assert np.max(np.abs(sol.p[free] * (1.0 - sol.theta[free]))) < 1e-9 * peak
    feed = grid.dirichlet & (grid.pressure_bc > 0)
    assert_allclose(sol.boundary_flow[feed].sum(), sol.flow_in, rtol=1e-10)


def test_recirculating_starved_film():
    """A periodic film that never fills carries the same flow round.

    A groove at ``x = 0`` holds ``p_cav`` and ``theta_in = 0.2``, the axial
    edges are ambient at ``p_cav``. With ``h(0) = 1.5 h0`` and
    ``h_min = 0.5 h0`` the film fraction ``theta_in h(0) / h`` stays below 1,
    so the whole film is cavitated, every ``x`` face carries the Couette flow
    ``U (h_i + h_i+1) theta_i / 4`` of the groove face and nothing is
    supplied. The boundary flows are rounding noise of either sign, so the
    mass balance is measured against the flow scale.
    """
    n, a, theta_in = 120, 0.5, 0.2
    x = np.linspace(0.0, LX, n, endpoint=False)
    z = np.linspace(0.0, 0.02, 6)
    h = film(x, a)[:, None] * np.ones(z.size)
    dirichlet = np.zeros(h.shape, bool)
    dirichlet[:, [0, -1]] = True
    dirichlet[0, :] = True
    fraction = np.ones(h.shape)
    fraction[0, 1:-1] = theta_in
    grid = CavitationGrid(
        x, z, h, h**3 / (12 * MU), U / 2, dirichlet, 0.0, fraction, period=LX
    )
    sol = solve_elrod_adams(grid, 0.0)
    assert_converged(sol)
    assert sol.cavitated[1:, 1:-1].all()
    assert np.max(np.abs(sol.p)) < 1e-9 * 6 * MU * U * LX / H0**2
    h_face = 0.5 * (h[:, 0] + np.roll(h[:, 0], -1))
    expected = theta_in * h_face[0] / h_face
    for j in range(1, z.size - 1):
        assert_allclose(sol.theta[1:, j], expected[1:], rtol=1e-9)
    assert_allclose(sol.theta[1:, 1], theta_in * h[0, 0] / h[1:, 0], rtol=0.03)
    assert np.max(np.abs(sol.boundary_flow)) < 1e-9 * U * H0 * theta_in * 0.02
    assert sol.mass_balance < 1e-10


def test_active_set_branch(monkeypatch):
    """A watchdog of zero steps runs the active-set steps from the start."""
    from ross.bearings.film import cavitation

    _, grid = slider(GRID, 0.5, 2.0e6)
    sol = solve_elrod_adams(grid, 0.0)
    calls = []

    def spy(a, b):
        calls.append(a.size)
        return min_function(a, b)

    min_function = cavitation._min_function
    monkeypatch.setattr(cavitation, "_WATCHDOG", 0)
    monkeypatch.setattr(cavitation, "_min_function", spy)
    forced = solve_elrod_adams(grid, 0.0)
    assert_converged(forced)
    assert len(calls) == forced.record.iterations
    assert_allclose(forced.p, sol.p, rtol=0, atol=1e-9 * sol.p.max())
    assert_allclose(forced.theta, sol.theta, rtol=0, atol=1e-10)


def test_hole_feed_switches_to_active_set_steps(monkeypatch):
    """The point-feed journal stalls in Fischer-Burmeister steps and switches."""
    from ross.bearings.film import cavitation

    calls = []
    min_function = cavitation._min_function

    def spy(a, b):
        calls.append(a.size)
        return min_function(a, b)

    monkeypatch.setattr(cavitation, "_min_function", spy)
    sol = solve_elrod_adams(journal_grid(240, 41, 0.6), 0.0)
    assert_converged(sol)
    assert 0 < len(calls) < sol.record.iterations


def test_cycle_stops_as_stalled(monkeypatch):
    """An active-set iteration that revisits an iterate stops early."""
    from ross.bearings.film import cavitation

    _, grid = slider(41, 0.5, 2.0e6)
    sign = [1.0]

    def flip(state, kp_ffs, kt_ffs, free, n, p_ref):
        sign[0] = -sign[0]
        dtheta = np.zeros(n)
        dtheta[free] = 0.25 * sign[0]
        return np.zeros(n), dtheta

    monkeypatch.setattr(cavitation, "_WATCHDOG", 0)
    monkeypatch.setattr(cavitation, "_newton_step", flip)
    with pytest.warns(ConvergenceWarning, match="stalled"):
        sol = solve_elrod_adams(grid, 0.0)
    assert sol.record.reason == "stalled"
    assert sol.record.iterations == 3


def test_iteration_cap_warns():
    _, grid = slider(GRID, 0.5, 2.0e6)
    with pytest.warns(ConvergenceWarning):
        sol = solve_elrod_adams(grid, 0.0, max_iterations=2)
    assert not sol.converged
    assert sol.record.reason == "max_iterations"
    assert sol.record.iterations == 2


def test_cold_start_converges_to_the_same_solution():
    for a, p_s in JFO_CASES:
        _, grid = slider(641, a, p_s)
        sol = solve_elrod_adams(grid, 0.0)
        for theta_init, p_init in ((0.5, 0.0), (1.0, 0.0), (1.0, p_s)):
            cold = solve_elrod_adams(grid, 0.0, theta_init=theta_init, p_init=p_init)
            assert_converged(cold)
            assert_allclose(cold.p, sol.p, rtol=0, atol=1e-9 * sol.p.max())


def test_invalid_inputs():
    x = np.linspace(0.0, 1.0, 5)
    ones = np.ones(5)
    ends = np.isin(np.arange(5), [0, 4])
    good = dict(
        x=x,
        z=[0.0],
        h=ones,
        diffusion=ones,
        couette=ones,
        dirichlet=ends,
        pressure_bc=ones,
    )
    bad = [
        dict(x=x[::-1]),
        dict(x=[0.0]),
        dict(z=[0.0, 0.0]),
        dict(h=-ones),
        dict(diffusion=0 * ones),
        dict(couette=np.ones(4)),
        dict(dirichlet=np.zeros(5, bool)),
        dict(film_fraction_in=1.5 * ones),
        dict(period=0.5),
        dict(pressure_bc=np.r_[np.nan, ones[1:]]),
    ]
    for change in bad:
        with pytest.raises(ValueError):
            CavitationGrid(**{**good, **change})
    grid = CavitationGrid(**good)
    for kwargs in (
        dict(p_cav=2.0),
        dict(p_cav=0.0, tol=0.0),
        dict(p_cav=0.0, max_iterations=0),
        dict(p_cav=0.0, p_ref=-1.0),
    ):
        with pytest.raises(ValueError):
            solve_elrod_adams(grid, **kwargs)


def _engine_pad_state(**changes):
    """Solved state of the two-axial-groove journal fixture (isoviscous)."""
    import json
    from pathlib import Path

    from ross.bearings.fluid_film.sweep import BearingSetup, _solve

    path = Path(__file__).parent / "data" / "fluid_film" / "fixed_isoviscous.json"
    inputs = json.loads(path.read_text())["inputs"]
    inputs.update(changes)
    setup = BearingSetup.from_inputs(**inputs)
    _, state, _ = _solve(setup, inputs["frequency"][0])
    return state


@pytest.mark.parametrize("film_fraction_in", [1.0, 0.4])
def test_journal_pad_grid_on_the_reynolds_nodes(film_fraction_in):
    """The finite-volume grid of a journal pad.

    References (closed forms of a laminar isoviscous film): the grid nodes
    are the Reynolds nodes of the pad; the Couette coefficient is ``U / 2``
    (``G = 1/2``, exact for the linear cross-film integrand); the diffusion
    coefficient is ``h^3 / (12 mu)`` (the engine integrates the quadratic
    cross-film integrand of ``Gamma`` with the trapezoid rule on its
    clustered nodes: within 2 %, and the same at every node); the Dirichlet
    nodes are the four pad edges (flooded); the inlet capacity is the
    Couette flow ``U / 2 h_f`` through the faces from the leading edge into
    the free nodes and into the half-width strips along the axial edges (the
    full axial width), and the inlet film fraction is set on the leading
    edge only.
    """
    from ross.bearings.fluid_film.pressure import cavitation_grid

    g = _engine_pad_state()
    mesh, pads, operating = g["mesh"], g["pads"], g["operating"]
    nx, nz = mesh.total_e_x_film + 1, mesh.total_e_z_film + 1
    pad = 1
    grid, capacity = cavitation_grid(
        mesh,
        operating,
        pad,
        pads,
        g["vis_effect_3d"],
        g["film_onset"],
        g["h_n"],
        film_fraction_in,
    )
    assert grid.shape == (nx, nz)
    assert_allclose(grid.x, mesh.x[pad, : nx * nz].reshape(nx, nz)[:, 0])
    assert_allclose(grid.z, mesh.z[pad, : nx * nz].reshape(nx, nz)[0])
    h = g["h_n"][pad, : nx * nz].reshape(nx, nz)
    assert_allclose(grid.h, h)
    assert_allclose(grid.couette, 0.5 * operating.speed_surface, rtol=1e-12)
    mu = g["vis_effect_3d"][pad, 0]
    assert np.ptp(g["vis_effect_3d"][pad]) == 0.0
    ratio = grid.diffusion * 12.0 * mu / h**3
    assert_allclose(ratio, ratio[0, 0], rtol=1e-9)
    assert abs(ratio[0, 0] - 1.0) < 0.02
    edges = np.zeros((nx, nz), dtype=bool)
    edges[[0, -1], :] = True
    edges[:, [0, -1]] = True
    assert np.array_equal(grid.dirichlet, edges)
    assert np.all(grid.pressure_bc[edges] == 0.0)
    assert np.all(grid.film_fraction_in[0] == film_fraction_in)
    assert np.all(grid.film_fraction_in[1:] == 1.0)
    width = np.diff(grid.z)
    control = np.r_[0.5 * width[0], 0.5 * (width[:-1] + width[1:]), 0.5 * width[-1]]
    h_face = 0.5 * (h[0] + h[1])
    expected = np.sum(0.5 * operating.speed_surface * h_face * control)
    assert_allclose(capacity, expected, rtol=1e-12)
