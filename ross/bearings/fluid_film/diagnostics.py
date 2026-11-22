"""Convergence records and the work budget of one journal-bearing case.

Every iterative loop of :func:`~ross.bearings.fluid_film.driver.run_case`
reports its last pass to a :class:`LoopMonitor`; every zeroth-order Reynolds
solve (one call of :func:`~ross.bearings.fluid_film.pressure.press` on the
equilibrium, tilt, turbulence and thermal paths) is charged to a
:class:`CaseBudget`. The perturbation solves of the dynamic coefficients are
not charged: neither those of the final coefficients nor those of the
Jacobian that each Newton iteration evaluates, so ``reynolds_solves``
understates the work of a long Newton search.

A case is converged when every loop that ran ended below its tolerance (the
residual test of the loop itself, not its divergence counters) and the case
was not stopped by the budget. The loop names are fixed:

==========================  ======================================  ==========
Loop                        Residual                                Tolerance
==========================  ======================================  ==========
``film_onset``              pads whose film onset moved (starved)   0 moves
``newton``                  ``|F_net| / |F_ext|``                   ``2e-3``
``tilt``                    worst pad: the smaller of               1
                            ``|M| / (1e-9 M_scale)`` (pivot moment)
                            and ``width / 1e-10`` (root bracket, rad)
``turbulence``              RMS shear-stress change, Pa             ``SHEAR_ERROR``
``viscosity``               RMS film/pad temperature change, K      ``TEMP_ERROR``
``journal_temperature``     journal temperature change, K           ``JTEMP_ERROR``
``mixing``                  RMS pad inlet temperature change, K     ``TEMP_INLET_ERROR``
``elasto``                  film deformation change / ``C_b``       ``DEFORM_ERROR``
``pivot``                   RMS pivot deflection change / ``C_b``   ``0.01``
``sump``                    sump temperature change, K              ``SUMP_TEMP_ERROR``
==========================  ======================================  ==========

The ``tilt`` record also gives the worst pad's ``moment`` (``|M| / M_scale``,
see :func:`~ross.bearings.fluid_film.hydrodynamics.moment_scale`) and
``width`` (rad) separately; its iterations are the moment evaluations (one
Reynolds solve each) of that pad's last search
(:class:`~ross.bearings.fluid_film.hydrodynamics.PadTiltSearch`). Its
residual is already relative to the two scaled tolerances, so its tolerance
stays 1.

All tolerances are multiplied by the private ``_tolerance_scale`` input of
``run_case`` (default 1.0, which leaves every number unchanged). A loop that
the model does not use (``pivot`` without pivot flexibility, ``sump`` at the
supply temperature, ``turbulence`` in laminar flow, ``film_onset`` when
flooded) records one pass with a zero residual; ``newton`` under
``equilibrium_type="match_eccentricity"`` records ``reason="not_applicable"``.

Reasons of a failed loop or case: ``"max_iterations"``, ``"diverged"`` (the
loop gave up after its residual grew for several iterations), ``"budget"``,
``"time_limit"``, ``"singular_jacobian"``, ``"pivot_unloaded"``,
``"non_finite"`` and ``"contact"`` (the final journal position gives a film
thickness <= 0 on a pad).
"""

import math
import time
from dataclasses import dataclass, field

from ross.bearings.film.loops import ConvergenceRecord
from ross.bearings.fluid_film import constants

__all__ = [
    "LOOP_NAMES",
    "REASONS",
    "SolverBudgetExceeded",
    "FluidFilmConvergenceError",
    "CaseBudget",
    "LoopMonitor",
    "default_budget",
]

LOOP_NAMES = (
    "film_onset",
    "newton",
    "tilt",
    "turbulence",
    "viscosity",
    "journal_temperature",
    "mixing",
    "elasto",
    "pivot",
    "sump",
)

REASONS = (
    "max_iterations",
    "diverged",
    "budget",
    "time_limit",
    "singular_jacobian",
    "pivot_unloaded",
    "non_finite",
    "contact",
)


class SolverBudgetExceeded(RuntimeError):
    """A case used up its Reynolds-solve budget or its time limit.

    Raised by :meth:`CaseBudget.charge` and caught by
    :func:`~ross.bearings.fluid_film.driver.run_case`, which stops the case
    and reports it as failed.

    Parameters
    ----------
    reason : str
        ``"budget"`` or ``"time_limit"``.
    message : str
        Description of the limit that was hit.
    """

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class FluidFilmConvergenceError(RuntimeError):
    """A fluid-film bearing case did not converge.

    Raised by :class:`~ross.bearings.fluid_film_bearing.FluidFilmBearing`
    with ``on_failure="raise"``.
    """


def default_budget(total_pads):
    """Default Reynolds-solve limits of a case with ``total_pads`` pads.

    Parameters
    ----------
    total_pads : int
        Number of pads (lands) of the bearing.

    Returns
    -------
    max_reynolds_solves : int
        Solves allowed for the case:
        :data:`~ross.bearings.fluid_film.constants.DEFAULT_MAX_REYNOLDS_SOLVES_PER_PAD`
        per pad.
    max_solves_per_call : int
        Solves allowed within one equilibrium solve:
        :data:`~ross.bearings.fluid_film.constants.MAX_REYNOLDS_SOLVES_PER_CALL_PER_PAD`
        per pad.

    Notes
    -----
    The pad count is taken as at least
    :data:`~ross.bearings.fluid_film.constants.BUDGET_MIN_PADS`.

    Examples
    --------
    >>> default_budget(2)
    (2000000, 20000)
    >>> default_budget(8)
    (3200000, 32000)
    """
    pads = max(int(total_pads), constants.BUDGET_MIN_PADS)
    return (
        constants.DEFAULT_MAX_REYNOLDS_SOLVES_PER_PAD * pads,
        constants.MAX_REYNOLDS_SOLVES_PER_CALL_PER_PAD * pads,
    )


@dataclass
class CaseBudget:
    """Reynolds-solve and wall-clock budget of one case.

    Parameters
    ----------
    max_reynolds_solves : int, optional
        Zeroth-order Reynolds solves allowed for the case. Default is the
        five-pad value of :func:`default_budget`.
    max_solves_per_call : int, optional
        Solves allowed within one equilibrium solve (one ``hydrodynamics``
        call). Default is the five-pad value of :func:`default_budget`.
    time_limit : float or None, optional
        Wall-clock limit of the case, s. Default is None (no limit).

    Examples
    --------
    >>> budget = CaseBudget(max_reynolds_solves=2)
    >>> budget.charge(); budget.charge()
    >>> try:
    ...     budget.charge()
    ... except SolverBudgetExceeded as error:
    ...     error.reason
    'budget'
    """

    max_reynolds_solves: int = field(
        default_factory=lambda: default_budget(constants.BUDGET_MIN_PADS)[0]
    )
    max_solves_per_call: int = field(
        default_factory=lambda: default_budget(constants.BUDGET_MIN_PADS)[1]
    )
    time_limit: float = None
    solves: int = 0
    call_solves: int = 0
    max_call_solves: int = 0
    calls: int = 0
    enabled: bool = True
    start: float = field(default_factory=time.perf_counter)

    def start_call(self):
        """Start counting the solves of a new equilibrium solve."""
        self.calls += 1
        self.call_solves = 0

    def charge(self, n=1):
        """Count ``n`` Reynolds solves, or stop the case if they exceed a limit.

        Parameters
        ----------
        n : int, optional
            Number of solves. Default is 1.

        Raises
        ------
        SolverBudgetExceeded
            When the case or the current equilibrium solve has used more
            solves than allowed, or the case ran longer than ``time_limit``.
        """
        if not self.enabled:
            return
        if self.solves + n > self.max_reynolds_solves:
            raise SolverBudgetExceeded(
                "budget",
                f"{self.max_reynolds_solves} Reynolds solves used, the case budget",
            )
        if self.call_solves + n > self.max_solves_per_call:
            raise SolverBudgetExceeded(
                "budget",
                f"{self.max_solves_per_call} Reynolds solves used in one "
                "equilibrium solve, the cap per equilibrium",
            )
        if self.time_limit is not None and self.elapsed > self.time_limit:
            raise SolverBudgetExceeded(
                "time_limit", f"case ran longer than {self.time_limit} s"
            )
        self.solves += n
        self.call_solves += n
        self.max_call_solves = max(self.max_call_solves, self.call_solves)

    @property
    def elapsed(self):
        """Wall time since the budget was created, s."""
        return time.perf_counter() - self.start


class LoopMonitor:
    """Last-pass records of the named loops of one case.

    Each :meth:`record` call replaces the record of that loop, so after the
    case the monitor holds the pass that produced the returned state. The
    iterations of every pass are also summed (``total_iterations``).

    Examples
    --------
    >>> monitor = LoopMonitor()
    >>> monitor.record("newton", 4, 1.0e-3, 2.0e-3, True)
    >>> monitor.record("viscosity", 100, 0.2, 0.0056, False, "max_iterations")
    >>> monitor.converged
    False
    >>> monitor.message()
    'viscosity: max_iterations (residual 0.2)'
    """

    def __init__(self):
        self.records = {}
        self.tolerances = {}
        self.extras = {}
        self.total_iterations = {}
        self.passes = {}
        self.stop_reason = ""
        self.stop_message = ""

    def record(
        self, name, iterations, residual, tolerance, converged, reason="", **extra
    ):
        """Store the last pass of loop ``name``.

        Parameters
        ----------
        name : str
            One of :data:`LOOP_NAMES`.
        iterations : int
            Iterations of this pass.
        residual : float
            Final residual of this pass, in the units of ``tolerance``.
        tolerance : float
            Tolerance the residual is tested against.
        converged : bool
            Whether the pass met the tolerance.
        reason : str, optional
            Why a pass that did not converge stopped (one of :data:`REASONS`),
            or ``"not_applicable"``. Default is ``""`` (``"converged"`` or
            ``"max_iterations"`` is filled in).
        **extra : dict
            Additional per-loop values for the summary (for example the pad
            of the ``tilt`` record).
        """
        residual = float(residual)
        if converged and not math.isfinite(residual) and reason != "not_applicable":
            converged = False
            reason = "non_finite"
        if not reason:
            reason = "converged" if converged else "max_iterations"
        self.records[name] = ConvergenceRecord(
            name, int(iterations), [residual], bool(converged), reason
        )
        self.tolerances[name] = float(tolerance)
        self.extras[name] = extra
        self.total_iterations[name] = self.total_iterations.get(name, 0) + int(
            iterations
        )
        self.passes[name] = self.passes.get(name, 0) + 1

    def stop(self, reason, message=""):
        """Mark the case as stopped (budget, time limit or non-finite output).

        Parameters
        ----------
        reason : str
            One of :data:`REASONS`.
        message : str, optional
            Description of the stop.
        """
        if not self.stop_reason:
            self.stop_reason = reason
            self.stop_message = message

    def failures(self):
        """Records of the loops whose last pass did not converge.

        Returns
        -------
        list of ConvergenceRecord
            In the order of :data:`LOOP_NAMES`.
        """
        return [
            self.records[name]
            for name in LOOP_NAMES
            if name in self.records and not self.records[name].converged
        ]

    @property
    def converged(self):
        """True when every recorded loop converged and the case was not stopped."""
        return not self.stop_reason and not self.failures()

    @property
    def reason(self):
        """Why the case failed: the stop reason, else that of the first failed loop."""
        if self.stop_reason:
            return self.stop_reason
        failures = self.failures()
        return failures[0].reason if failures else ""

    def message(self):
        """Describe the failure as ``"<loop>: <reason> (residual r)"``.

        Returns
        -------
        str
            ``""`` for a converged case; failures are joined with ``"; "``,
            the stop (``"case: budget (...)"``) first.
        """
        parts = []
        if self.stop_reason:
            parts.append(f"case: {self.stop_reason} ({self.stop_message})")
        parts.extend(
            f"{rec.name}: {rec.reason} (residual {rec.final_residual:.3g})"
            for rec in self.failures()
        )
        return "; ".join(parts)

    def summary(self, budget=None, wall_time=None):
        """Return the convergence summary of the case as a plain dict.

        Parameters
        ----------
        budget : CaseBudget, optional
            Budget of the case, for the solve counts.
        wall_time : float, optional
            Wall time of the case, s.

        Returns
        -------
        dict
            ``converged``, ``reason``, ``message``, ``reynolds_solves``,
            ``max_call_solves``, ``hydrodynamics_calls``, ``wall_time`` and
            ``loops``: per loop name, ``iterations``, ``residual``,
            ``tolerance``, ``converged``, ``reason``, ``total_iterations`` and
            ``passes`` (plus ``pad``, ``moment`` and ``width`` for ``tilt``).
        """
        loops = {}
        for name in LOOP_NAMES:
            if name not in self.records:
                continue
            rec = self.records[name]
            loops[name] = {
                "iterations": rec.iterations,
                "residual": rec.final_residual,
                "tolerance": self.tolerances[name],
                "converged": rec.converged,
                "reason": rec.reason,
                "total_iterations": self.total_iterations[name],
                "passes": self.passes[name],
                **self.extras[name],
            }
        return {
            "converged": self.converged,
            "reason": self.reason,
            "message": self.message(),
            "reynolds_solves": None if budget is None else budget.solves,
            "max_call_solves": None if budget is None else budget.max_call_solves,
            "hydrodynamics_calls": None if budget is None else budget.calls,
            "wall_time": wall_time,
            "loops": loops,
        }
