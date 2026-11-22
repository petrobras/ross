"""Speed sweeps of a journal bearing: mesh once, warm-started chains, processes.

:class:`BearingSetup` holds everything of a bearing that does not depend on
the speed: the converted inputs, the Reynolds, energy, 3-D and pad meshes, the
pad geometry and the lubricant. Its arrays are read-only. :func:`solve_case`
solves one speed of a setup; each case works on its own copies of the two
arrays the film-thickness rebuild writes (the radial node coordinates of the
3-D and energy meshes), so a setup can be shared by any number of cases and
processes. :func:`~ross.bearings.fluid_film.driver.run_case` is
``solve_case(BearingSetup.from_inputs(**kw), ...)``.

:func:`solve_sweep` solves a list of speeds. The speeds are sorted and split
into chains of at most ``chunk_size`` consecutive speeds, which depend only
on the speeds (:func:`sweep_chains`). The first speed of a chain starts cold;
each later speed starts from the converged state of the previous one
(:class:`WarmStart`). A warm-started case that does not converge is solved
again cold. ``n_jobs`` sets how many chains run at once in worker processes;
every chain is solved with single-threaded BLAS in the serial and the parallel
path, so the results do not depend on ``n_jobs``.

:func:`film_force_at` evaluates the hydrodynamic film with the journal held at
a given position: no equilibrium (Newton) search and no dynamic coefficients.
For tilting pads the pad tilts are either solved from the pad moment balance,
held at the tilts of the case's equilibrium, or given. The thermal and
deformation loops of the case run as usual, with the journal held. It is the
finite-difference reference of the stiffness coefficients and the building
block of a force map on a grid of journal positions (one setup, no
re-meshing).
"""

import dataclasses
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CASE_KEYWORDS",
    "BearingSetup",
    "WarmStart",
    "film_force_at",
    "solve_case",
    "solve_sweep",
    "sweep_chains",
]

TILTING_PAD_TYPES = (
    "conventional_tilting_pad",
    "inlet_groove_tilting_pad",
    "spray_bar_tilting_pad",
)

#: ``run_case`` keywords that belong to one solve, not to the bearing setup.
CASE_KEYWORDS = (
    "frequency",
    "excit_ratios",
    "whirl_ratios",
    "field_outputs",
    "diagnostics",
    "pad_matrices",
    "_tolerance_scale",
    "_held_tilts",
    "_film_force_only",
)

#: Temperature and viscosity fields a warm start copies (those present).
WARM_TEMPERATURE_KEYS = (
    "temp_3d",
    "temp_adiab",
    "temp_full",
    "temp_inlet",
    "temp_j",
    "temp_sump",
    "vis_n_3d",
    "vis_n_average",
)

#: Deformation fields a warm start copies (those present).
WARM_DEFORMATION_KEYS = ("dh_n", "deform_pivot", "k_pivot")


def _copy(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, (list, tuple)):
        return type(value)(_copy(v) for v in value)
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    return value


def _lock(value):
    """Make every array reachable from ``value`` read-only."""
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _lock(v)
    elif isinstance(value, dict):
        for v in value.values():
            _lock(v)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            _lock(getattr(value, f.name))


@dataclass(frozen=True, eq=False)
class BearingSetup:
    """Everything of a journal bearing that does not depend on the speed.

    Build it with :meth:`from_inputs`. The arrays of the setup (inputs,
    meshes, pad geometry) are read-only, and no solve writes them, so one
    setup serves every speed of a sweep, in this process or in worker
    processes.

    Attributes
    ----------
    inputs : dict
        The ``run_case`` keywords of the bearing (SI), without the per-case
        ones of :data:`CASE_KEYWORDS`.
    state : dict
        The solver state shared by every speed: converted geometry, meshes
        and lubricant (see :func:`ross.bearings.fluid_film.driver._setup_state`).

    Examples
    --------
    >>> import json
    >>> from pathlib import Path
    >>> import ross
    >>> path = Path(ross.__file__).parent / "tests/data/fluid_film/fixed_isoviscous.json"
    >>> setup = BearingSetup.from_inputs(**json.loads(path.read_text())["inputs"])
    >>> setup.mesh.y_3d.flags.writeable
    False
    >>> out = solve_case(setup, 94.25)
    >>> out["convergence"][0]["converged"]
    True
    """

    inputs: dict
    state: dict = field(repr=False)

    @classmethod
    def from_inputs(cls, **kw):
        """Build the setup from ``run_case`` keywords.

        Parameters
        ----------
        **kw : dict
            Keyword arguments of
            :func:`ross.bearings.fluid_film.driver.run_case` (SI). The
            per-case keywords (:data:`CASE_KEYWORDS`) are ignored.

        Returns
        -------
        BearingSetup
        """
        from ross.bearings.fluid_film.driver import _setup_state

        inputs = {k: _copy(v) for k, v in kw.items() if k not in CASE_KEYWORDS}
        state = _setup_state(dict(inputs))
        setup = cls(inputs=inputs, state=state)
        setup._lock()
        return setup

    def _lock(self):
        _lock(self.inputs)
        _lock(self.state)

    def __setstate__(self, state):
        # unpickled arrays (worker processes) come back writeable
        self.__dict__.update(state)
        self._lock()

    @property
    def mesh(self):
        """ReynoldsMesh: the ``x``-``z`` film mesh."""
        return self.state["mesh"]

    @property
    def energy_mesh(self):
        """EnergyMesh: the ``x``-``y`` film and pad mesh."""
        return self.state["energy_mesh"]

    @property
    def pads(self):
        """PadGeometry: the per-pad geometry."""
        return self.state["pads"]

    @property
    def lube(self):
        """Lubricant: the lubricant of the journal engine."""
        return self.state["lube"]

    @property
    def total_pads(self):
        """int: number of pads (lands)."""
        return self.state["total_pads"]


@dataclass(frozen=True, eq=False)
class WarmStart:
    """Converged state of one speed, the starting point of another.

    Parameters
    ----------
    xj, yj : float
        Journal position, m. Used only by ``match_load`` cases (a
        ``match_eccentricity`` case keeps its prescribed position).
    tilt_angle : numpy.ndarray
        Pad tilts, rad.
    film_onset : numpy.ndarray of int
        Film-onset element of each pad (starved operation).
    temperature : dict
        Temperature and viscosity fields (:data:`WARM_TEMPERATURE_KEYS`).
    deformation : dict
        Film deformation and pivot fields (:data:`WARM_DEFORMATION_KEYS`).
    newton_relax : float, optional
        Relaxation of the equilibrium Newton position update of the
        warm-started case, in place of ``relax_p``. Default is 1.0 (full
        Newton steps: the start is the equilibrium of a nearby speed; the
        relaxed update converges only linearly, about one iteration per
        halving of the force residual).
    tilt_slope : numpy.ndarray, optional
        Slope of each pad's pivot moment over its tilt, N*m/rad (NaN where
        unknown). With the tilts it starts each pad's tilt search close to
        its root (:class:`~ross.bearings.fluid_film.hydrodynamics.PadTiltSearch`).
        Default is None (the search steps by a fixed fraction of the
        admissible range from the copied tilt).
    """

    xj: float
    yj: float
    tilt_angle: np.ndarray
    film_onset: np.ndarray
    temperature: dict
    deformation: dict
    newton_relax: float = 1.0
    tilt_slope: np.ndarray = None

    @classmethod
    def from_state(cls, g):
        """Copy the warm-start fields of a solved case's state.

        Parameters
        ----------
        g : dict
            Final solver state of a case.

        Returns
        -------
        WarmStart
        """
        total_pads = g["total_pads"]
        return cls(
            xj=float(g["xj"]),
            yj=float(g["yj"]),
            tilt_angle=np.array(g["tilt_angle"][:total_pads], dtype=float),
            film_onset=np.array(g["film_onset"], copy=True),
            temperature={k: _copy(g[k]) for k in WARM_TEMPERATURE_KEYS if k in g},
            deformation={k: _copy(g[k]) for k in WARM_DEFORMATION_KEYS if k in g},
            tilt_slope=(
                np.array(g["_tilt_memory"]["slope"], dtype=float)
                if "_tilt_memory" in g
                else None
            ),
        )

    def apply(self, g):
        """Write copies of the warm fields into a case's state.

        The journal position is set by the case before its initialisation;
        this sets the tilts, film onset, temperatures, viscosities and
        deformations.

        Parameters
        ----------
        g : dict
            Solver state of the case being started (updated in place).
        """
        g["tilt_angle"] = np.array(self.tilt_angle, copy=True)
        total_pads = len(self.tilt_angle)
        g["_tilt_memory"] = {
            "known": np.ones(total_pads, dtype=bool),
            "slope": (
                np.array(self.tilt_slope, dtype=float, copy=True)
                if self.tilt_slope is not None
                else np.full(total_pads, np.nan)
            ),
        }
        g["film_onset"] = np.array(self.film_onset, copy=True)
        for name, value in {**self.temperature, **self.deformation}.items():
            g[name] = _copy(value)


def _solve(
    setup,
    speed,
    excitation_ratio=1.0,
    warm=None,
    diagnostics=True,
    budget=None,
    tolerance_scale=1.0,
    **options,
):
    """Solve one case; return ``(outputs, g, monitor)``."""
    from ross.bearings.fluid_film import driver

    if budget is None:
        budget = driver._case_budget({**setup.inputs, "frequency": float(speed)})
    return driver._solve_state(
        setup.state,
        float(speed),
        float(excitation_ratio),
        budget=budget,
        warm=warm,
        diagnostics=diagnostics,
        tolerance_scale=float(tolerance_scale),
        **options,
    )


def solve_case(
    setup,
    speed,
    excitation_ratio=1.0,
    warm=None,
    diagnostics=True,
    budget=None,
    _tolerance_scale=1.0,
    *,
    whirl_ratios=None,
    field_outputs=False,
    pad_matrices=False,
    return_warm=False,
    _held_tilts=None,
    _film_force_only=False,
    _position=None,
):
    """Solve one speed of a bearing setup.

    Parameters
    ----------
    setup : BearingSetup
        The bearing.
    speed : float
        Rotor speed, rad/s.
    excitation_ratio : float, optional
        Whirl-to-speed ratio at which the coefficients are condensed.
        Default is 1.0 (synchronous).
    warm : WarmStart, optional
        Converged state of another speed to start from. Default is None
        (cold start from the setup's initial position).
    diagnostics : bool, optional
        Add the ``"convergence"`` summary and the ``"non_convergence"``
        messages (see :func:`~ross.bearings.fluid_film.driver.run_case`).
        Default is True.
    budget : CaseBudget, optional
        Work budget of the case (a new one per case). Default is None (the
        budget of the setup's ``max_reynolds_solves`` and ``time_limit``).
    _tolerance_scale : float, optional
        Factor on every loop tolerance, for tests that compare two solution
        paths. Default is 1.0.
    whirl_ratios : array_like of float, optional
        Whirl-to-speed ratios at which the solved state is condensed, one
        output entry each. Default is None (``excitation_ratio`` only).
    field_outputs, pad_matrices : bool, optional
        Add the ``"fields"`` and ``"pad_matrices"`` keys. Default is False.
    return_warm : bool, optional
        Also return the :class:`WarmStart` of the solved case. Default is
        False.
    _held_tilts, _film_force_only, _position : optional
        Used by :func:`film_force_at`: pad tilts held instead of solved
        (rad), return the film force only, and the journal position held
        instead of solved (``(xj, yj)``, m).

    Returns
    -------
    outputs : dict
        The outputs of :func:`~ross.bearings.fluid_film.driver.run_case`.
    warm : WarmStart
        Only with ``return_warm=True``.

    Examples
    --------
    >>> import json
    >>> from pathlib import Path
    >>> import ross
    >>> path = Path(ross.__file__).parent / "tests/data/fluid_film/fixed_isoviscous.json"
    >>> setup = BearingSetup.from_inputs(**json.loads(path.read_text())["inputs"])
    >>> cold, warm = solve_case(setup, 94.25, return_warm=True)
    >>> again = solve_case(setup, 100.0, warm=warm)
    >>> again["convergence"][0]["converged"]
    True
    """
    outputs, g, _ = _solve(
        setup,
        speed,
        excitation_ratio,
        warm=warm,
        diagnostics=diagnostics,
        budget=budget,
        tolerance_scale=_tolerance_scale,
        whirl_ratios=whirl_ratios,
        field_outputs=field_outputs,
        pad_matrices=pad_matrices,
        held_tilts=_held_tilts,
        film_force_only=_film_force_only,
        position=_position,
    )
    if return_warm:
        return outputs, WarmStart.from_state(g)
    return outputs


def sweep_chains(speeds, chunk_size):
    """Split a list of speeds into warm-start chains.

    The speeds are sorted ascending (a stable sort) and cut into consecutive
    chains of ``chunk_size`` speeds (the last one may be shorter). The chains
    depend only on the speeds and ``chunk_size``.

    Parameters
    ----------
    speeds : array_like of float
        Rotor speeds, rad/s, in any order.
    chunk_size : int
        Speeds per chain, >= 1.

    Returns
    -------
    list of numpy.ndarray of int
        Each chain as indices into ``speeds``, in ascending speed.

    Examples
    --------
    >>> [c.tolist() for c in sweep_chains([300.0, 100.0, 200.0, 400.0], 3)]
    [[1, 2, 0], [3]]
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError(f"chunk_size must be an integer, got {chunk_size!r}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    order = np.argsort(np.asarray(speeds, dtype=float).ravel(), kind="stable")
    return [order[i : i + chunk_size] for i in range(0, order.size, chunk_size)]


def _n_workers(n_jobs, n_chains):
    """Worker processes for ``n_chains`` chains (``-1``: one per CPU)."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
        raise TypeError(f"n_jobs must be an integer, got {n_jobs!r}")
    n_jobs = int(n_jobs)
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    elif n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer or -1, got {n_jobs}")
    return max(1, min(n_jobs, int(n_chains)))


def _record_start(outputs, start, failure="", failed_solves=0):
    """Add the warm-start record to the convergence summaries of a case."""
    record = {"start": start, "failure": failure, "failed_solves": failed_solves}
    for summary in outputs.get("convergence", []):
        summary["warm_start"] = dict(record)


def _solve_chain(setup, speeds, warm_start, options):
    """Solve one chain of speeds (ascending) in order; return the outputs."""
    from ross.bearings.film.loops import blas_single_thread

    frequency = options.pop("frequency", None)
    results = []
    with blas_single_thread():
        warm = None
        for speed in speeds:
            case = dict(options)
            if frequency is not None:
                case["whirl_ratios"] = [f / speed for f in frequency]
            start = "warm" if warm_start and warm is not None else "cold"
            outputs, g, monitor = _solve(setup, speed, warm=warm, **case)
            failure, failed_solves = "", 0
            if start == "warm" and not monitor.converged:
                failure = monitor.message()
                failed_solves = _solves(outputs)
                start = "cold_retry"
                outputs, g, monitor = _solve(setup, speed, **case)
            _record_start(outputs, start, failure, failed_solves)
            if monitor.converged:
                warm = WarmStart.from_state(g)
            results.append(outputs)
    return results


def _solves(outputs):
    summaries = outputs.get("convergence") or [{}]
    return int(summaries[0].get("reynolds_solves") or 0)


def solve_sweep(
    setup,
    speeds,
    warm_start=True,
    chunk_size=5,
    n_jobs=1,
    excitation_ratio=1.0,
    frequency=None,
    diagnostics=True,
    field_outputs=False,
    pad_matrices=False,
    _tolerance_scale=1.0,
):
    """Solve a bearing setup at several speeds.

    The speeds are split into chains (:func:`sweep_chains`); within a chain
    each speed starts from the converged state of the previous one
    (:class:`WarmStart`): journal position (``match_load``), pad tilts,
    temperature and viscosity fields, film deformation and film onset. A
    warm-started case that does not converge is solved again cold, and the
    convergence summary of every case records how it started
    (``"warm_start"``: ``start`` is ``"cold"``, ``"warm"`` or
    ``"cold_retry"``; ``failure`` the message of the failed warm attempt and
    ``failed_solves`` its Reynolds solves).

    With ``n_jobs > 1`` the chains run in ``spawn`` worker processes. Every
    chain, serial or not, is solved with single-threaded BLAS, so the results
    are the same for every ``n_jobs``. Starting the workers costs about the
    import time of ``ross`` (a few seconds). A script that uses
    ``n_jobs > 1`` must guard its entry point with
    ``if __name__ == "__main__":``. A Jupyter notebook needs no guard.

    Parameters
    ----------
    setup : BearingSetup
        The bearing.
    speeds : array_like of float
        Rotor speeds, rad/s, in any order.
    warm_start : bool, optional
        Start each speed of a chain from the previous one. Default is True.
    chunk_size : int, optional
        Speeds per chain. Default is 5.
    n_jobs : int, optional
        Chains solved at once in worker processes; -1 uses one per CPU.
        Default is 1 (serial, no process is started).
    excitation_ratio : float, optional
        Whirl-to-speed ratio of the coefficients. Default is 1.0.
    frequency : array_like of float, optional
        Whirl frequencies, rad/s, at which every speed is condensed (one
        output entry each, overrides ``excitation_ratio``). Default is None.
    diagnostics, field_outputs, pad_matrices : bool, optional
        As in :func:`solve_case`. Defaults True, False, False.
    _tolerance_scale : float, optional
        Factor on every loop tolerance. Default is 1.0.

    Returns
    -------
    list of dict
        The outputs of each speed, in the order of ``speeds``.

    Raises
    ------
    RuntimeError
        When a worker process stops without returning its results (an
        unguarded script, or a worker killed by the system). No result is
        returned; the error chains the worker failure.

    Examples
    --------
    >>> import json
    >>> from pathlib import Path
    >>> import ross
    >>> path = Path(ross.__file__).parent / "tests/data/fluid_film/fixed_isoviscous.json"
    >>> setup = BearingSetup.from_inputs(**json.loads(path.read_text())["inputs"])
    >>> outs = solve_sweep(setup, [110.0, 80.0, 95.0])
    >>> [o["convergence"][0]["warm_start"]["start"] for o in outs]
    ['warm', 'cold', 'warm']
    """
    speeds = np.atleast_1d(np.asarray(speeds, dtype=float)).ravel()
    if speeds.size == 0:
        return []
    chains = sweep_chains(speeds, chunk_size if warm_start else 1)
    options = {
        "excitation_ratio": float(excitation_ratio),
        "diagnostics": bool(diagnostics),
        "field_outputs": bool(field_outputs),
        "pad_matrices": bool(pad_matrices),
        "tolerance_scale": float(_tolerance_scale),
    }
    if frequency is not None:
        options["frequency"] = list(np.atleast_1d(np.asarray(frequency, dtype=float)))
    jobs = [([float(speeds[i]) for i in chain], chain) for chain in chains]
    workers = _n_workers(n_jobs, len(jobs))
    if workers == 1:
        solved = [
            _solve_chain(setup, chain_speeds, warm_start, dict(options))
            for chain_speeds, _ in jobs
        ]
    else:
        solved = _solve_chains_in_processes(setup, jobs, warm_start, options, workers)
    outputs = [None] * speeds.size
    for (_, chain), results in zip(jobs, solved, strict=True):
        for i, out in zip(chain, results, strict=True):
            outputs[int(i)] = out
    return outputs


def _solve_chains_in_processes(setup, jobs, warm_start, options, workers):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [
                pool.submit(
                    _solve_chain, setup, chain_speeds, warm_start, dict(options)
                )
                for chain_speeds, _ in jobs
            ]
            return [future.result() for future in futures]
    except BrokenProcessPool as error:
        raise RuntimeError(
            f"the worker processes of the n_jobs={workers} sweep stopped before "
            "returning their results, so nothing was solved. The usual cause is "
            "a script that solves in worker processes without guarding its entry "
            'point: put the call under if __name__ == "__main__":, or use '
            "n_jobs=1. The worker error is chained below."
        ) from error


def film_force_at(setup, speed, xj, yj, tilts="solve", tolerance_scale=1.0, warm=None):
    """Evaluate the film force with the journal held at ``(xj, yj)``.

    Parameters
    ----------
    setup : BearingSetup or dict
        The bearing, or the keyword arguments of
        :func:`ross.bearings.fluid_film.driver.run_case` (SI) to build it
        from (``frequency``, ``equilibrium_type``, ``xj`` and ``yj`` are not
        used). Pass a setup to evaluate many positions without re-meshing.
    speed : float
        Rotor speed, rad/s.
    xj, yj : float
        Journal centre position, m.
    tilts : {"solve", "frozen"} or array_like of float, optional
        Pad tilt angles of a tilting-pad bearing. ``"solve"`` (default)
        solves each pad's moment balance at the given position; ``"frozen"``
        holds the tilts of the equilibrium of the setup at ``speed`` (one
        extra equilibrium solve); an array holds the given tilts, rad, one
        per pad. Ignored for fixed-geometry bearings.
    tolerance_scale : float, optional
        Factor on every loop tolerance of the case (the private
        ``_tolerance_scale`` of ``run_case``). A finite difference of the
        force with re-solved tilts needs a tilt tolerance well below the tilt
        change of the step. Default is 1.0.
    warm : WarmStart, optional
        Temperatures, viscosities and deformations to start the thermal and
        deformation loops from (its position is not used). Default is None.

    Returns
    -------
    dict
        ``fx``, ``fy`` : float
            Force of the film (and of the groove pressure) on the journal, N.
        ``fx_pad``, ``fy_pad`` : numpy.ndarray
            Film force of each pad on the journal, N.
        ``moment_pad`` : numpy.ndarray
            Moment of each pad's film pressure about its pivot, N*m
            (see :func:`ross.bearings.fluid_film.hydrodynamics.film_moment`).
        ``tilt_angle`` : numpy.ndarray
            Pad tilts used, rad.
        ``h_min`` : numpy.ndarray
            Minimum film thickness of each pad, m.
        ``nodal_pressure`` : numpy.ndarray
            Film pressure, Pa, shape ``(total_pads, dim_xz)``.
        ``cavitated`` : numpy.ndarray of bool
            Nodes held at the cavitation pressure, same shape.
        ``converged`` : bool
            Every loop that ran converged (the Newton loop is not run).
        ``convergence`` : dict
            Loop records, as ``run_case(..., diagnostics=True)``.

    Examples
    --------
    >>> import json
    >>> from pathlib import Path
    >>> import ross
    >>> path = Path(ross.__file__).parent / "tests/data/fluid_film/fixed_isoviscous.json"
    >>> setup = BearingSetup.from_inputs(**json.loads(path.read_text())["inputs"])
    >>> cb = setup.inputs["radial_clearance"]
    >>> out = film_force_at(setup, 94.25, 0.3 * cb, -0.4 * cb)
    >>> out["fy"] > 0.0, out["converged"]
    (True, True)
    """
    if not isinstance(setup, BearingSetup):
        setup = BearingSetup.from_inputs(**setup)
    total_pads = setup.total_pads

    held = None
    if setup.state["bearing_type"] in TILTING_PAD_TYPES:
        if isinstance(tilts, str):
            if tilts == "frozen":
                equilibrium = solve_case(
                    setup,
                    speed,
                    warm=warm,
                    diagnostics=False,
                    _tolerance_scale=tolerance_scale,
                )
                held = np.asarray(equilibrium["tilt_angle"][0], dtype=float)
            elif tilts != "solve":
                raise ValueError(
                    f"tilts must be 'solve', 'frozen' or an array, got {tilts!r}"
                )
        else:
            held = np.asarray(tilts, dtype=float).ravel()
            if held.shape != (total_pads,) or not np.all(np.isfinite(held)):
                raise ValueError(
                    f"tilts must hold {total_pads} finite angles (rad), got {tilts!r}"
                )

    return solve_case(
        setup,
        speed,
        warm=warm,
        _tolerance_scale=tolerance_scale,
        _held_tilts=held,
        _film_force_only=True,
        _position=(float(xj), float(yj)),
    )
