"""Utility functions for electric motor simulation and signal processing.
"""

import numpy as np
from numba import njit


@njit
def phase_to_line(v_phase):
    """Convert phase voltage to line-to-line voltage.

    For a balanced three-phase system, the line-to-line voltage magnitude is
    related to the phase voltage by a factor of :math:`\\sqrt{3}`.

    Parameters
    ----------
    v_phase : float
        Phase voltage [V].

    Returns
    -------
    float
        Line-to-line voltage [V].

    Examples
    --------
    >>> phase_to_line(127.0)  # doctest: +ELLIPSIS
    219.9...
    """
    return v_phase * np.sqrt(3)


@njit
def clarke_transform(a, b, c):
    """Apply the Clarke (alpha-beta) transform to three-phase quantities.

    Transforms instantaneous phase values ``(a, b, c)`` into the stationary
    two-axis reference frame ``(alpha, beta)``. The transform uses the
    amplitude-invariant convention with a 2/3 scaling factor.

    Parameters
    ----------
    a : float
        Instantaneous value of phase a.
    b : float
        Instantaneous value of phase b.
    c : float
        Instantaneous value of phase c.

    Returns
    -------
    alpha : float
        Alpha-axis component.
    beta : float
        Beta-axis component.

    Examples
    --------
    >>> alpha, beta = clarke_transform(1.0, -0.5, -0.5)
    >>> alpha
    1.0
    >>> beta
    0.0
    """
    alpha = 2 / 3 * (a - b / 2 - c / 2)
    beta = 2 / 3 * (b - c) * np.sqrt(3) / 2
    return alpha, beta


@njit
def park_transform(alpha, beta, theta):
    """Apply the Park (d-q) transform to alpha-beta quantities.

    Rotates the stationary ``(alpha, beta)`` frame into the synchronous
    ``(d, q)`` reference frame using the electrical angle ``theta``.

    Parameters
    ----------
    alpha : float
        Alpha-axis component.
    beta : float
        Beta-axis component.
    theta : float
        Electrical angle of the synchronous reference frame [rad].

    Returns
    -------
    d : float
        Direct-axis component.
    q : float
        Quadrature-axis component.

    Examples
    --------
    >>> d, q = park_transform(1.0, 0.0, 0.0)
    >>> d
    1.0
    >>> q
    0.0
    """
    d = alpha * np.cos(theta) + beta * np.sin(theta)
    q = -alpha * np.sin(theta) + beta * np.cos(theta)
    return d, q


def windowed_dfft(signal, dt, idx_eval=None):
    """Compute the windowed Discrete Fast Fourier Transform (DFFT) of a signal.

    The input signal is multiplied by a Hanning window before computing the FFT.
    Magnitudes are normalized by the window coherent gain, and one-sided
    spectrum scaling is applied to interior frequency bins.

    Parameters
    ----------
    signal : array_like
        Time domain signal.
    dt : float
        Sampling interval [s].
    idx_eval : array_like of int, optional
        Indices selecting frequency bins to return. If None, all non-negative
        frequency bins are returned.

    Returns
    -------
    freq : ndarray
        Frequency range [Hz].
    mag : ndarray
        Magnitude of response in frequency domain.

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000, endpoint=False)
    >>> signal = np.sin(2 * np.pi * 60 * t)
    >>> freq, mag = windowed_dfft(signal, dt=t[1] - t[0])
    >>> np.argmax(mag)  # doctest: +ELLIPSIS
    60
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


def rk4_step(func, dt, y0, args):
    """Perform a single Runge-Kutta 4 (RK4) integration step.

    Parameters
    ----------
    func : callable
        Function that computes derivatives. Must accept the state variables
        followed by additional arguments, i.e. ``func(*y, *args)``.
    dt : float
        Time step size [s].
    y0 : array-like
        Initial state vector at the beginning of the step.
    args : tuple
        Additional arguments passed to ``func``.

    Returns
    -------
    tuple
        Updated state vector after one RK4 step.

    Examples
    --------
    >>> def dydt(y):
    ...     return np.array([y])
    >>> y1 = rk4_step(dydt, dt=0.1, y0=(1.0,), args=())
    >>> y1[0]  # doctest: +ELLIPSIS
    1.105...
    """
    y = np.array(y0)

    k1 = func(*y, *args)

    k2 = func(*(y + 0.5 * dt * k1), *args)

    k3 = func(*(y + 0.5 * dt * k2), *args)

    k4 = func(*(y + dt * k3), *args)

    y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return tuple(y)
