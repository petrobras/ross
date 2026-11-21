"""Iteration records and warnings of the fluid-film solvers.

An iterative loop reports its outcome as a :class:`ConvergenceRecord`; a loop
that stops without converging emits a :class:`ConvergenceWarning`.
:func:`blas_single_thread` limits the BLAS pools inside a solve.
"""

import contextlib
import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ConvergenceRecord",
    "ConvergenceWarning",
    "blas_single_thread",
]


class ConvergenceWarning(RuntimeWarning):
    """An iterative film solve stopped without meeting its tolerance."""


@dataclass
class ConvergenceRecord:
    """Outcome of one iterative loop.

    Parameters
    ----------
    name : str
        Loop name, for example ``"thermal"``.
    iterations : int
        Iterations performed.
    residual_history : list of float
        Scaled residual (or other monitored quantity) per iteration.
    converged : bool
        Whether the tolerance was met.
    reason : str
        ``"converged"``, ``"max_iterations"``, ``"nan"`` or a loop-specific
        reason.

    Examples
    --------
    >>> rec = ConvergenceRecord("thermal", 2, [3.0, 0.0], True, "converged")
    >>> rec.final_residual
    0.0
    """

    name: str
    iterations: int = 0
    residual_history: list = field(default_factory=list)
    converged: bool = False
    reason: str = ""

    @property
    def final_residual(self):
        """Last entry of ``residual_history`` (NaN when empty)."""
        return self.residual_history[-1] if self.residual_history else float("nan")

    def warn(self, stacklevel=3):
        """Emit a :class:`ConvergenceWarning` when the loop did not converge.

        Parameters
        ----------
        stacklevel : int, optional
            Passed to ``warnings.warn``. Default is 3.
        """
        if not self.converged:
            warnings.warn(
                f"{self.name} stopped after {self.iterations} iterations without "
                f"converging (reason: {self.reason}, final residual "
                f"{self.final_residual:.3g})",
                ConvergenceWarning,
                stacklevel=stacklevel,
            )


#: Cached ``threadpoolctl.ThreadpoolController`` (built at the first call of
#: :func:`blas_single_thread`; scanning the loaded libraries costs about 3 ms).
_CONTROLLER = None


def _single_thread_limits(threadpoolctl):
    """``threadpool_limits(limits=1)`` on the cached controller (no library rescan)."""
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = threadpoolctl.ThreadpoolController()
    controller = _CONTROLLER

    class _CachedLimits(threadpoolctl.threadpool_limits):
        def __init__(self):
            # the limiter of threadpool_limits, on the cached controller
            super(threadpoolctl.threadpool_limits, self).__init__(
                controller, limits=1, user_api=None
            )

    try:
        return _CachedLimits()
    except (TypeError, AttributeError):  # another threadpoolctl API: rescan
        return threadpoolctl.threadpool_limits(limits=1)


def blas_single_thread():
    """Context manager that limits BLAS/OpenMP pools to one thread.

    Uses ``threadpoolctl.threadpool_limits(1)`` when ``threadpoolctl`` is
    importable and does nothing otherwise (it is not a ROSS dependency). The
    library scan of ``threadpoolctl`` (about 3 ms) is made once per process and
    cached, so entering costs a few microseconds; the BLAS and OpenMP
    libraries loaded after the first call (none of the film core's) are not
    limited. As with ``threadpool_limits``, the limit applies from the call and
    is restored on exit.

    Returns
    -------
    contextlib.AbstractContextManager
        The limiting context, or a null context.

    Examples
    --------
    >>> with blas_single_thread():
    ...     pass
    """
    try:
        import threadpoolctl
    except ImportError:
        return contextlib.nullcontext()
    return _single_thread_limits(threadpoolctl)
