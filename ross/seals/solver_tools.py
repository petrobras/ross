"""Shared helpers for the seal flow solvers."""

import multiprocessing

__all__ = ["solve_frequencies"]


def solve_frequencies(solve, frequencies, parallel_threshold):
    """Run a per-frequency solve, in parallel when it pays off.

    Uses a multiprocessing pool when the number of frequencies exceeds
    ``parallel_threshold``; below that, sequential execution avoids the
    process spawn overhead.

    Parameters
    ----------
    solve : callable
        Function mapping one frequency (rad/s) to its results. Must be
        picklable so it can be dispatched to worker processes.
    frequencies : iterable of float
        Frequencies to solve.
    parallel_threshold : int
        Number of frequencies above which a process pool is used.

    Returns
    -------
    results : list
        One result per frequency, in input order.
    """
    if len(frequencies) > parallel_threshold:
        with multiprocessing.Pool() as pool:
            return pool.map(solve, frequencies)
    return [solve(frequency) for frequency in frequencies]
