"""Mass-conserving (Elrod-Adams) cavitation on a structured node grid.

The film is described by the pressure ``p`` and the film fraction ``theta``
(the fraction of the gap filled with liquid). The mass flux per unit width is

    q = -A grad(p) + c h theta e_x,

with the diffusion coefficient ``A`` (``h^3 / (12 mu)`` for a laminar
isoviscous film), the Couette coefficient ``c`` (``U / 2`` for a laminar film
on a surface moving at ``U`` in ``x``) and the film thickness ``h``. The steady
problem is ``div(q) = 0`` with the complementarity conditions of Giacopini et
al. (2010)::

    p >= p_cav,   theta <= 1,   (p - p_cav) (1 - theta) = 0,

so the film is full (``theta = 1``) where the pressure is above the cavitation
pressure and the cavitated zone (``p = p_cav``) carries the Couette flow of the
striated film. Rupture and reformation (the Jakobsson-Floberg-Olsson
conditions) follow from the flux balance without being imposed.

Discretisation: node-centred finite volumes on the grid nodes (control volumes
bounded by the mid-points between nodes, half volumes on the edges). The
diffusive face flux is central with the face average of ``A``; the Couette
face flux is ``(c h)_f theta_up``, the face average of ``c h`` times the film
fraction of the upstream node. Where ``theta = 1`` this is the central
full-film flux, second order in the mesh size; in the cavitated zone it is the
first-order upwind flux of Ausas et al. (2007): the film fraction in a
cavitated zone converges at first order. The face fluxes telescope, so the
cavitated zone carries the discrete flow exactly and the pressure of a
one-dimensional cavitated film still converges at second order.

Solver: a Fischer-Burmeister Newton method with a Schur reduction, following
the approach of Woloszynski et al. (2015). The complementarity is written as ``phi(a, b) = a + b - sqrt(a^2 + b^2) = 0`` with
``a = (p - p_cav) / p_ref`` and ``b = 1 - theta``. The discrete fluxes are
linear in ``(p, theta)``, so the Newton system is the constant flux Jacobian
plus the diagonal Fischer-Burmeister block. The diagonal block is eliminated
node by node (the Schur complement): each node keeps one unknown ``t`` that
moves ``(a, b)`` along the level line of the linearised ``phi``, which is a
pressure update in the full film and a film-fraction update in the cavitated
zone, and one sparse LU of the ``n_free`` by ``n_free`` Schur matrix gives the
step.

Globalisation. Full Newton steps are taken: a backtracking line search on the
merit ``||F||^2`` stalls on the kinks of ``phi`` and needs a number of
iterations that grows with the grid. Fischer-Burmeister steps can circle
without converging on two-dimensional films fed through a hole or a short
groove, where the ambient edges leave both arguments of ``phi`` near zero.
So, when the merit has not improved on its best value for ``_WATCHDOG``
steps (a watchdog, Chamberlain et al. 1982), the iteration returns to the
best iterate and continues with the semismooth Newton steps of the
min-function ``min(a, b)``: the primal-dual active set step of Hintermueller
et al. (2002), which holds each node at ``p = p_cav`` or at ``theta = 1`` and
does not depend on the size of ``a`` against ``b``. The same Schur
elimination gives these steps. An active-set iteration that revisits an
iterate stops with ``reason="stalled"``.

Iterations (from a cold start, on the grids of the tests): one-dimensional
fed, starved and periodic films converge in at most 14 iterations on 41 to
2561 nodes; two-dimensional journal pads fed through a hole, a partial or a
full groove, flooded or starved, in 8 to 26 on 60 by 11 to 240 by 41 nodes. A cavitated zone that starts too large (a reformation
front that has to move upstream) shrinks by about one node per iteration, so
a strongly starved inlet (a short full-film zone) needs a number of
iterations that grows with the grid; on very fine one-dimensional grids it
exceeds the default ``max_iterations=50``. Mesh sequencing is the remedy: a
solution on a coarser grid, interpolated and passed as ``p_init`` and
``theta_init``, removes the growth (5 or 6 iterations per grid doubling).

The module imports nothing from ``ross.bearings.fluid_film``.

References
----------
Elrod, H. G. (1981). A cavitation algorithm. J. Lubr. Technol. 103(3),
350-354.

Ausas, R., Ragot, P., Leiva, J., Jai, M., Bayada, G., Buscaglia, G. C.
(2007). The impact of the cavitation model in the analysis of
microtextured lubricated journal bearings. J. Tribol. 129(4), 868-875.

Giacopini, M., Fowell, M. T., Dini, D., Strozzi, A. (2010). A mass-conserving
complementarity formulation to study lubricant films in the presence of
cavitation. J. Tribol. 132(4), 041702.

Woloszynski, T., Podsiadlo, P., Stachowiak, G. W. (2015). Efficient solution
to the cavitation problem in hydrodynamic lubrication. Tribol. Lett. 58, 18.

Chamberlain, R. M., Powell, M. J. D., Lemarechal, C., Pedersen, H. C. (1982).
The watchdog technique for forcing convergence in algorithms for constrained
optimization. Math. Program. Study 16, 1-17.

Hintermueller, M., Ito, K., Kunisch, K. (2002). The primal-dual active set
strategy as a semismooth Newton method. SIAM J. Optim. 13(3), 865-888.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ross.bearings.film.loops import ConvergenceRecord

__all__ = ["CavitationGrid", "CavitationSolution", "solve_elrod_adams"]

_WATCHDOG = 8
_MEMORY = 64


def _control_widths(nodes, period=None):
    """Return the control-volume width of each node of a 1-D node line.

    Parameters
    ----------
    nodes : numpy.ndarray
        Increasing node coordinates.
    period : float, optional
        Period of a periodic line (the last node connects to the first one at
        ``nodes[0] + period``). None for an open line with half volumes on
        the ends.

    Returns
    -------
    widths : numpy.ndarray
        Width of the control volume of each node. A single node has width 1
        (unit width of a one-dimensional problem).
    """
    n = nodes.size
    if n == 1:
        return np.ones(1)
    gaps = np.diff(nodes)
    widths = np.zeros(n)
    widths[:-1] += 0.5 * gaps
    widths[1:] += 0.5 * gaps
    if period is not None:
        wrap = period - (nodes[-1] - nodes[0])
        widths[0] += 0.5 * wrap
        widths[-1] += 0.5 * wrap
    return widths


@dataclass
class CavitationGrid:
    """Structured node grid of one film region.

    Nodes are the tensor product of ``x`` (direction of the surface motion,
    the only direction with Couette flow) and ``z``; nodal arrays have shape
    ``(x.size, z.size)`` (a one-dimensional array is accepted for a grid with
    one ``z`` node). Edges that are not Dirichlet nodes are closed (no flux).
    A grid with one ``z`` node is one-dimensional with unit width.

    Parameters
    ----------
    x : array_like
        Increasing node coordinates in the direction of motion, m.
    z : array_like
        Increasing node coordinates across the motion, m.
    h : array_like
        Film thickness at the nodes, m.
    diffusion : array_like
        Pressure-flow coefficient ``A`` at the nodes (positive;
        ``h^3 / (12 mu)`` for a laminar film), m^3/(Pa s).
    couette : array_like
        Couette coefficient ``c`` at the nodes (``U / 2`` for a laminar film,
        the surface speed times the engine's ``G`` in general), m/s. The
        Couette flux per unit width is ``c h theta``.
    dirichlet : array_like of bool
        Nodes with imposed pressure and film fraction (feed, ambient edges).
    pressure_bc : array_like
        Pressure at the Dirichlet nodes (ignored elsewhere), Pa.
    film_fraction_in : array_like, optional
        Film fraction at the Dirichlet nodes (ignored elsewhere). It matters
        where the Couette flow leaves a Dirichlet node into the film: 1 is a
        flooded inlet, less than 1 a starved inlet. Default is 1.
    period : float, optional
        Period in ``x`` of a closed film (360 degree journal): the last node
        line connects to the first one across ``period - (x[-1] - x[0])``.
        Default is None (open in ``x``).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0.0, 1.0, 5)
    >>> grid = CavitationGrid(x, [0.0], np.ones(5), np.ones(5), np.ones(5),
    ...                       np.isin(np.arange(5), [0, 4]), np.zeros(5))
    >>> grid.shape, grid.n_free
    ((5, 1), 3)
    """

    x: np.ndarray
    z: np.ndarray
    h: np.ndarray
    diffusion: np.ndarray
    couette: np.ndarray
    dirichlet: np.ndarray
    pressure_bc: np.ndarray
    film_fraction_in: np.ndarray = None
    period: float = None

    def __post_init__(self):
        self.x = np.atleast_1d(np.asarray(self.x, dtype=float))
        self.z = np.atleast_1d(np.asarray(self.z, dtype=float))
        if self.x.ndim != 1 or self.z.ndim != 1:
            raise ValueError("x and z must be one-dimensional node coordinates.")
        if self.x.size < 2:
            raise ValueError("x needs at least two nodes.")
        for name, axis in (("x", self.x), ("z", self.z)):
            if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                raise ValueError(f"{name} must be finite and strictly increasing.")
        shape = self.shape
        self.h = self._nodal(self.h, "h")
        self.diffusion = self._nodal(self.diffusion, "diffusion")
        self.couette = self._nodal(self.couette, "couette")
        self.pressure_bc = self._nodal(self.pressure_bc, "pressure_bc")
        if self.film_fraction_in is None:
            self.film_fraction_in = np.ones(shape)
        self.film_fraction_in = self._nodal(self.film_fraction_in, "film_fraction_in")
        self.dirichlet = self._nodal(self.dirichlet, "dirichlet") != 0.0
        if np.any(self.h <= 0):
            raise ValueError("h must be positive.")
        if np.any(self.diffusion <= 0):
            raise ValueError("diffusion must be positive.")
        fraction = self.film_fraction_in[self.dirichlet]
        if np.any((fraction < 0) | (fraction > 1)):
            raise ValueError("film_fraction_in must lie in [0, 1].")
        if self.period is not None:
            self.period = float(self.period)
            if not self.period > self.x[-1] - self.x[0]:
                raise ValueError("period must exceed x[-1] - x[0].")
        if not np.any(self.dirichlet):
            raise ValueError("At least one Dirichlet node is needed.")
        self._build_operators()

    def _nodal(self, values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1 and self.z.size == 1 and values.size == self.x.size:
            values = values[:, None]
        try:
            values = np.broadcast_to(values, self.shape).copy()
        except ValueError:
            raise ValueError(
                f"{name} must broadcast to the node grid {self.shape}."
            ) from None
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite.")
        return values

    @property
    def shape(self):
        """Node grid shape ``(x.size, z.size)``."""
        return (self.x.size, self.z.size)

    @property
    def n_free(self):
        """Number of nodes that are not Dirichlet nodes."""
        return int(np.count_nonzero(~self.dirichlet))

    def _build_operators(self):
        nx, nz = self.shape
        index = np.arange(nx * nz).reshape(nx, nz)
        width_x = _control_widths(self.x, self.period)
        width_z = _control_widths(self.z)
        a = self.diffusion
        ch = self.couette * self.h

        left = [index[:-1, :].ravel()]
        right = [index[1:, :].ravel()]
        gap = np.diff(self.x)[:, None] * np.ones((1, nz))
        if self.period is not None:
            left.append(index[-1, :])
            right.append(index[0, :])
            wrap = self.period - (self.x[-1] - self.x[0])
            gap = np.vstack([gap, np.full((1, nz), wrap)])
        left_x = np.concatenate(left)
        right_x = np.concatenate(right)
        a_flat, ch_flat = a.ravel(), ch.ravel()
        face_width = np.tile(width_z, gap.shape[0])
        cond_x = face_width * 0.5 * (a_flat[left_x] + a_flat[right_x]) / gap.ravel()
        couette_x = face_width * 0.5 * (ch_flat[left_x] + ch_flat[right_x])

        if nz > 1:
            left_z = index[:, :-1].ravel()
            right_z = index[:, 1:].ravel()
            gap_z = np.tile(np.diff(self.z), nx)
            width = np.repeat(width_x, nz - 1)
            cond_z = width * 0.5 * (a_flat[left_z] + a_flat[right_z]) / gap_z
        else:
            left_z = right_z = np.zeros(0, dtype=int)
            cond_z = np.zeros(0)

        self.face_left = np.concatenate([left_x, left_z])
        self.face_right = np.concatenate([right_x, right_z])
        self.face_conductance = np.concatenate([cond_x, cond_z])
        self.face_couette = np.concatenate([couette_x, np.zeros(cond_z.size)])
        self.face_upwind = np.where(
            self.face_couette >= 0.0, self.face_left, self.face_right
        )
        self.control_area = np.outer(width_x, width_z)

        n = nx * nz
        fl, fr, fu = self.face_left, self.face_right, self.face_upwind
        k = self.face_conductance
        rows = np.concatenate([fl, fl, fr, fr])
        cols = np.concatenate([fl, fr, fl, fr])
        vals = np.concatenate([k, -k, -k, k])
        self.pressure_operator = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        b = self.face_couette
        self.fraction_operator = sp.csr_matrix(
            (
                np.concatenate([b, -b]),
                (np.concatenate([fl, fr]), np.concatenate([fu, fu])),
            ),
            shape=(n, n),
        )

    def face_flux(self, p, theta):
        """Return the mass flux through every face, positive from left to right.

        Parameters
        ----------
        p : array_like
            Nodal pressure, Pa.
        theta : array_like
            Nodal film fraction.

        Returns
        -------
        flux : numpy.ndarray
            Flux through each face (``x`` faces first, then ``z`` faces), m^3/s
            (m^2/s for a one-dimensional grid).
        """
        p = np.ravel(p)
        theta = np.ravel(theta)
        return (
            self.face_conductance * (p[self.face_left] - p[self.face_right])
            + self.face_couette * theta[self.face_upwind]
        )

    def residual(self, p, theta):
        """Return the net outflow of every control volume.

        Parameters
        ----------
        p : array_like
            Nodal pressure, Pa.
        theta : array_like
            Nodal film fraction.

        Returns
        -------
        outflow : numpy.ndarray
            Net outflow of each control volume, shape ``grid.shape``. At a
            Dirichlet node it is the flow the boundary supplies to the film.
        """
        return (
            self.pressure_operator @ np.ravel(p)
            + self.fraction_operator @ np.ravel(theta)
        ).reshape(self.shape)


@dataclass
class CavitationSolution:
    """Solution of :func:`solve_elrod_adams`.

    Parameters
    ----------
    p : numpy.ndarray
        Nodal pressure, shape ``grid.shape``, Pa.
    theta : numpy.ndarray
        Nodal film fraction, shape ``grid.shape``.
    cavitated : numpy.ndarray
        Boolean mask of the free nodes in the cavitated zone
        (``1 - theta > (p - p_cav) / p_ref``).
    boundary_flow : numpy.ndarray
        Flow supplied by each Dirichlet node to the film (positive in,
        negative out; zero on free nodes), m^3/s. Faces between two Dirichlet
        nodes (along an ambient edge, inside a groove) are not film flow and
        are left out.
    flow_in : float
        Sum of the positive boundary flows, m^3/s.
    flow_out : float
        Minus the sum of the negative boundary flows, m^3/s.
    mass_balance : float
        ``|flow_in - flow_out| / flow_in``; relative to ``1e-3 q_ref`` (the
        flow scale of the tolerance) when less flows in, as in a periodic film
        that only recirculates.
    record : ConvergenceRecord
        Newton record; the residual history is the scaled max-norm of
        ``F = (net outflow / q_ref, phi)`` per iteration, starting with the
        initial guess.
    p_ref : float
        Pressure scale of the complementarity, Pa.
    """

    p: np.ndarray
    theta: np.ndarray
    cavitated: np.ndarray
    boundary_flow: np.ndarray
    flow_in: float
    flow_out: float
    mass_balance: float
    record: ConvergenceRecord
    p_ref: float

    @property
    def converged(self):
        """Whether the Newton iteration met its tolerance."""
        return self.record.converged


def _min_function(a, b):
    """Return ``min(a, b)`` and its generalised derivatives ``(da, db)``."""
    held = a < b
    return np.minimum(a, b), held.astype(float), (~held).astype(float)


def _fischer_burmeister(a, b):
    """Return ``phi(a, b)`` and its generalised derivatives ``(da, db)``."""
    r = np.hypot(a, b)
    phi = a + b - r
    corner = r == 0.0
    safe = np.where(corner, 1.0, r)
    da = np.where(corner, 1.0 - np.sqrt(0.5), 1.0 - a / safe)
    db = np.where(corner, 1.0 - np.sqrt(0.5), 1.0 - b / safe)
    return phi, da, db


def _newton_step(state, kp_ffs, kt_ffs, free, n, p_ref):
    """Return the Fischer-Burmeister-Newton step ``(dp, dtheta)`` or None.

    Per free node, the linearised ``phi + da da_ + db db_ = 0`` is solved by
    ``(delta a, delta b) = -phi (da, db) / (da^2 + db^2) + t (db, -da)``, and
    the scaled mass equations give the Schur system for ``t``.
    """
    _, _, mass, phi, da, db, _ = state
    norm2 = da * da + db * db
    delta_a0 = -phi * da / norm2
    delta_b0 = -phi * db / norm2
    schur = (kp_ffs @ sp.diags(db) + kt_ffs @ sp.diags(da)).tocsc()
    rhs = -mass - kp_ffs @ delta_a0 + kt_ffs @ delta_b0
    try:
        t = splu(schur).solve(rhs)
    except RuntimeError:
        return None
    if not np.all(np.isfinite(t)):
        return None
    dp = np.zeros(n)
    dtheta = np.zeros(n)
    dp[free] = p_ref * (delta_a0 + db * t)
    dtheta[free] = -(delta_b0 - da * t)
    return dp, dtheta


def _pressure_scale(grid, p_cav):
    """Return the pressure scale ``max(c h) L / max(A)`` or the supply span, Pa."""
    span = np.ptp(grid.x) if grid.period is None else grid.period
    hydrodynamic = np.max(np.abs(grid.couette * grid.h)) * span / np.max(grid.diffusion)
    supply = np.max(np.abs(grid.pressure_bc[grid.dirichlet] - p_cav))
    scale = max(hydrodynamic, supply)
    return scale if scale > 0 else 1.0


def solve_elrod_adams(
    grid,
    p_cav,
    theta_init=None,
    tol=1e-10,
    max_iterations=50,
    p_init=None,
    p_ref=None,
):
    """Solve the steady Elrod-Adams (p-theta) film problem on a grid.

    Parameters
    ----------
    grid : CavitationGrid
        Film grid, coefficients and boundary data.
    p_cav : float
        Cavitation pressure (gauge), Pa. The Dirichlet pressures must not be
        below it.
    theta_init : array_like, optional
        Initial film fraction on the grid. Default is 1 (full film): a
        cavitated zone that starts too small shrinks to its size in a few
        iterations, while one that starts too large (a reformation front
        upstream of its guess) moves by about one node per iteration.
    tol : float, optional
        Tolerance on the max-norm of the scaled residual
        ``(net outflow / q_ref, phi)``, with the flow scale
        ``q_ref = W (max(c h) + max(A) p_ref / L)`` (``W`` the film width,
        ``L`` its length in ``x``).
        Default is 1e-10.
    max_iterations : int, optional
        Newton iteration cap. Default is 50.
    p_init : array_like, optional
        Initial pressure. Default is the full-film pressure (``theta = 1``)
        clipped at ``p_cav``.
    p_ref : float, optional
        Pressure scale of the complementarity variable ``(p - p_cav) / p_ref``,
        Pa. Default is the larger of the hydrodynamic scale
        ``max(c h) L / max(A)`` and the largest ``|p_bc - p_cav|``.

    Returns
    -------
    CavitationSolution
        Pressure, film fraction, cavitated mask, boundary flows, mass balance
        and the Newton record. A run that stops without converging (iteration
        cap, ``reason="max_iterations"``; an active-set iteration that
        revisits an iterate, ``reason="stalled"``) returns its last iterate
        and emits a :class:`~ross.bearings.film.loops.ConvergenceWarning`.

    Examples
    --------
    A slider whose gap opens downstream cavitates; the flow that enters at
    ``x = 0`` leaves at ``x = 1``:

    >>> import numpy as np
    >>> x = np.linspace(0.0, 1.0, 41)
    >>> h = 1.0 + 0.5 * np.cos(2 * np.pi * x)
    >>> ends = np.isin(np.arange(41), [0, 40])
    >>> grid = CavitationGrid(x, [0.0], h, h**3 / 12, 0.5, ends, 0.01)
    >>> sol = solve_elrod_adams(grid, 0.0)
    >>> sol.converged, bool(sol.cavitated.any()), sol.mass_balance < 1e-10
    (True, True, True)
    """
    p_cav = float(p_cav)
    bc = grid.pressure_bc[grid.dirichlet]
    if np.any(bc < p_cav):
        raise ValueError("Dirichlet pressures must not be below p_cav.")
    if not (isinstance(max_iterations, (int, np.integer)) and max_iterations >= 1):
        raise ValueError("max_iterations must be a positive integer.")
    if not tol > 0:
        raise ValueError("tol must be positive.")
    if p_ref is None:
        p_ref = _pressure_scale(grid, p_cav)
    p_ref = float(p_ref)
    if not p_ref > 0:
        raise ValueError("p_ref must be positive.")

    dirichlet = grid.dirichlet.ravel()
    free = ~dirichlet
    n = dirichlet.size
    kp = grid.pressure_operator
    kt = grid.fraction_operator
    kp_ff = kp[free][:, free].tocsc()
    kp_f = kp[free]
    kt_f = kt[free]

    p = np.where(dirichlet, grid.pressure_bc.ravel(), 0.0)
    theta = np.where(dirichlet, grid.film_fraction_in.ravel(), 1.0)
    if p_init is None:
        rhs = -(kp_f @ p + kt_f @ theta)
        p[free] = np.maximum(splu(kp_ff).solve(rhs), p_cav)
    else:
        p[free] = grid._nodal(p_init, "p_init").ravel()[free]
    if theta_init is not None:
        theta[free] = grid._nodal(theta_init, "theta_init").ravel()[free]

    span = np.ptp(grid.x) if grid.period is None else grid.period
    width = grid.control_area.sum() / span
    q_ref = width * (
        np.max(np.abs(grid.couette * grid.h)) + np.max(grid.diffusion) * p_ref / span
    )

    def residual(p, theta):
        mass = (kp_f @ p + kt_f @ theta) / q_ref
        phi, da, db = _fischer_burmeister((p[free] - p_cav) / p_ref, 1.0 - theta[free])
        return mass, phi, da, db

    mass, phi, da, db = residual(p, theta)
    merit = 0.5 * (mass @ mass + phi @ phi)
    record = ConvergenceRecord("elrod_adams")
    record.residual_history.append(max(np.max(np.abs(mass)), np.max(np.abs(phi))))
    kp_ffs = (kp_ff * (p_ref / q_ref)).tocsc()
    kt_ffs = (kt_f[:, free] / q_ref).tocsc()
    best = (p, theta, mass, phi, da, db, merit)
    stalled = 0
    active_set = False
    visited = []
    for it in range(1, max_iterations + 1):
        if record.residual_history[-1] <= tol:
            record.converged, record.reason = True, "converged"
            break
        if not np.isfinite(record.residual_history[-1]):
            record.reason = "nan"
            break
        record.iterations = it
        if not active_set and stalled >= _WATCHDOG:
            active_set = True
            p, theta, mass, phi, da, db, merit = best
        if active_set:
            a_min, da_min, db_min = _min_function(
                (p[free] - p_cav) / p_ref, 1.0 - theta[free]
            )
            state = (p, theta, mass, a_min, da_min, db_min, merit)
        else:
            state = (p, theta, mass, phi, da, db, merit)
        step_state = _newton_step(state, kp_ffs, kt_ffs, free, n, p_ref)
        if step_state is None:
            record.reason = "singular"
            break
        dp, dtheta = step_state
        step = 1.0
        while True:
            p_new = p + step * dp
            theta_new = theta + step * dtheta
            mass_n, phi_n, da_n, db_n = residual(p_new, theta_new)
            merit_new = 0.5 * (mass_n @ mass_n + phi_n @ phi_n)
            finite = np.isfinite(merit_new)
            if finite or step < 1e-10:
                break
            step *= 0.5
        if not finite:
            record.residual_history.append(np.nan)
            record.reason = "nan"
            break
        p, theta = p_new, theta_new
        mass, phi, da, db, merit = mass_n, phi_n, da_n, db_n, merit_new
        record.residual_history.append(max(np.max(np.abs(mass)), np.max(np.abs(phi))))
        if merit < (1.0 - 1e-4) * best[-1]:
            best = (p, theta, mass, phi, da, db, merit)
            stalled = 0
        else:
            stalled += 1
        if active_set and record.residual_history[-1] > tol:
            if any(abs(merit - old) <= 1e-10 * merit for old in visited):
                record.reason = "stalled"
                break
            visited = visited[1 - _MEMORY :] + [merit]
    else:
        if record.residual_history[-1] <= tol:
            record.converged, record.reason = True, "converged"
        else:
            record.reason = "max_iterations"
    record.warn(stacklevel=2)

    flux = grid.face_flux(p, theta)
    film = ~(dirichlet[grid.face_left] & dirichlet[grid.face_right])
    outflow = np.bincount(grid.face_left[film], flux[film], minlength=n) - np.bincount(
        grid.face_right[film], flux[film], minlength=n
    )
    boundary_flow = np.where(dirichlet, outflow, 0.0)
    flow_in = float(np.sum(np.clip(boundary_flow, 0.0, None)))
    flow_out = float(-np.sum(np.clip(boundary_flow, None, 0.0)))
    mass_balance = abs(flow_in - flow_out) / max(flow_in, 1e-3 * q_ref)
    cavitated = np.zeros(n, dtype=bool)
    cavitated[free] = (1.0 - theta[free]) > (p[free] - p_cav) / p_ref
    shape = grid.shape
    return CavitationSolution(
        p=p.reshape(shape),
        theta=theta.reshape(shape),
        cavitated=cavitated.reshape(shape),
        boundary_flow=boundary_flow.reshape(shape),
        flow_in=flow_in,
        flow_out=flow_out,
        mass_balance=mass_balance,
        record=record,
        p_ref=p_ref,
    )
