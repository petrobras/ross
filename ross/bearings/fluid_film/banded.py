"""Banded LU factorisation and substitution.

The Reynolds (pressure), energy (temperature) and pad-deformation systems are
all banded, and all three are solved by the same partial-pivoting LU after
Numerical Recipes ``bandec`` / ``banbks``. This module is that solver; the
three subsystems differ only in how large their band is.

Band storage
------------
A system of ``total_n`` equations with half-bandwidth ``bandwidth`` is stored
as ``(rows, 2 * bandwidth - 1)``: row ``k`` of the matrix occupies row ``k`` of
the array, and the diagonal sits at column ``bandwidth - 1``. Only the stored
band is touched, so the array may be longer than ``total_n`` -- the extra rows
are simply ignored.

The pressure solve needs one extra step: the Reynolds cavitation condition
clamps the solution to the cavitation pressure during back substitution. That
is :func:`lu_solve_cavitating`; everything else uses :func:`lu_solve`.
"""

import numpy as np

from ross.bearings.fluid_film._numba_kernels import (
    lu_factor_band_jit,
    lu_solve_band_cavitating_jit,
    lu_solve_band_jit,
)

__all__ = ["lu_factor", "lu_solve", "lu_solve_cavitating"]


def lu_factor(a, total_n, bandwidth):
    """LU-decompose a banded matrix with partial pivoting.

    Parameters
    ----------
    a : array_like, shape (rows, 2 * bandwidth - 1)
        Banded matrix, diagonal at column ``bandwidth - 1``. It is copied
        before being repacked, so the caller's array is left alone -- take the
        decomposition from the return value.
    total_n : int
        Number of equations.
    bandwidth : int
        Half-bandwidth.

    Returns
    -------
    a : numpy.ndarray
        The decomposed upper band.
    a_lower : numpy.ndarray, shape (rows, bandwidth - 1)
        Lower-triangular multipliers.
    index1 : numpy.ndarray of int
        Pivot row permutation, as 1-based row numbers (the ``bandec``
        convention the substitution kernels expect).
    d : float
        Row-swap sign, ``+1`` or ``-1``.
    """
    # Copy: the kernel repacks and overwrites in place, and
    # ``ascontiguousarray`` would hand it the caller's own array whenever that
    # is already contiguous float64.
    a = np.array(a, dtype=np.float64, copy=True)
    return lu_factor_band_jit(a, total_n, bandwidth)


def lu_solve(a, total_n, bandwidth, a_lower, index1, b):
    """Solve ``A x = b`` from the factors of :func:`lu_factor`.

    Parameters
    ----------
    a, a_lower, index1 : array_like
        The factors returned by :func:`lu_factor`.
    total_n : int
        Number of equations.
    bandwidth : int
        Half-bandwidth.
    b : array_like
        Right-hand side. It is copied, so the caller's array is left alone --
        take the solution from the return value.

    Returns
    -------
    numpy.ndarray
        The solution.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    a_lower = np.ascontiguousarray(a_lower, dtype=np.float64)
    index1 = np.ascontiguousarray(index1, dtype=np.int64)
    b = np.array(b, dtype=np.float64, copy=True)
    return lu_solve_band_jit(a, total_n, bandwidth, a_lower, index1, b)


def lu_solve_cavitating(a, total_n, bandwidth, a_lower, index1, b, press_cavitate):
    """Solve ``A x = b`` and apply the Reynolds cavitation clamp.

    Identical to :func:`lu_solve` except that the back substitution floors the
    solution at ``press_cavitate``: the film cannot sustain a pressure below
    the cavitation pressure, and clamping during -- rather than after -- the
    substitution is what makes the pressure solve converge.

    Parameters
    ----------
    a, a_lower, index1 : array_like
        The factors returned by :func:`lu_factor`.
    total_n : int
        Number of equations.
    bandwidth : int
        Half-bandwidth.
    b : array_like
        Right-hand side. Copied, as in :func:`lu_solve`.
    press_cavitate : float
        Cavitation pressure, Pa.

    Returns
    -------
    numpy.ndarray
        The clamped solution.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    a_lower = np.ascontiguousarray(a_lower, dtype=np.float64)
    index1 = np.ascontiguousarray(index1, dtype=np.int64)
    b = np.array(b, dtype=np.float64, copy=True)
    return lu_solve_band_cavitating_jit(
        a, total_n, bandwidth, a_lower, index1, b, press_cavitate
    )
