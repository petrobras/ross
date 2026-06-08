import numpy as np
from numba import njit


@njit
def phase_to_line(v):
    return v * np.sqrt(3)


@njit
def clarke_transform(a, b, c):
    alpha = 2 / 3 * (a - b / 2 - c / 2)
    beta = 2 / 3 * (b - c) * np.sqrt(3) / 2
    return alpha, beta


@njit
def park_transform(alpha, beta, theta):
    d = alpha * np.cos(theta) + beta * np.sin(theta)
    q = -alpha * np.sin(theta) + beta * np.cos(theta)
    return d, q


def windowed_dfft(signal, dt, idx_eval=None):
    """Compute the windowed Discrete Fast Fourier Transform (DFFT) of a signal.

    Parameters
    ----------
    signal : array_like
        Time domain signal.
    dt : float
        Sampling interval [s].
    idx_eval : array_like of int, optional
        Indices defining the portion of the signal to be analyzed. If None,
        the entire signal is used.

    Returns
    -------
    freq : ndarray
        Frequency range [Hz].
    mag : ndarray
        Magnitude of response in frequency domain.
    """

    signal = np.asarray(signal)

    N = len(signal)
    window = np.hanning(N)

    signal_windowed = signal * window
    fft_values = np.fft.fft(signal_windowed)

    freq = np.fft.fftfreq(N, d=dt)

    coherent_gain = np.mean(window)
    mag = np.abs(fft_values) / (N * coherent_gain)

    if idx_eval is not None:
        freq = freq[idx_eval]
        mag = mag[idx_eval]

    idx = freq >= 0
    freq = freq[idx]
    mag = mag[idx]

    # Internal components duplicated
    if len(mag) > 2:
        mag[1:-1] *= 2

    return freq, mag


def get_corresponding_indices(t, t_eval, dt_ref=1e-4):
    """Return the indices of `t` corresponding to the values in `t_eval`.

    Each value in `t_eval` must be present in `t`. The returned indices
    satisfy ``t[idx_eval] == t_eval``.

    Parameters
    ----------
    t : ndarray
        Monotonically increasing reference array.
    t_eval : ndarray
        Values whose positions in `t` are to be located.
    dt_ref : float, optional
        Reference spacing used to generate the default evaluation grid
        when `t_eval` is None. Default is 1e-4.

    Returns
    -------
    idx_eval : ndarray
        Indices of the elements in `t` corresponding to `t_eval`.
    """
    if t_eval is None:
        dt = t[1] - t[0]
        nt = int(dt / dt_ref)
        if nt > 1:
            t_eval = np.linspace(t[0], t[-1], np.float32(dt * nt))
        else:
            t_eval = t

    idx_eval = np.unique(
        np.concatenate(([0], np.searchsorted(t, t_eval), [len(t) - 1]))
    )
    idx_eval = idx_eval[idx_eval < len(t)]

    return idx_eval


def rk4_step(func, dt, y0, args):
    """Perform a single Runge-Kutta 4 (RK4) integration step.

    Parameters
    ----------
    func : callable
        Function that computes derivatives. Must accept the state variables
        followed by additional arguments, i.e. ``func(*y, *args)``.
    dt : float
        Time step size.
    y0 : array-like
        Initial state vector at the beginning of the step.
    args : tuple
        Additional arguments passed to `func`.

    Returns
    -------
    tuple
        Updated state vector after one RK4 step.
    """
    y = np.array(y0)

    k1 = func(*y, *args)

    k2 = func(*(y + 0.5 * dt * k1), *args)

    k3 = func(*(y + 0.5 * dt * k2), *args)

    k4 = func(*(y + dt * k3), *args)

    y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return tuple(y)
