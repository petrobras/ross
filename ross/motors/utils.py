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


def windowed_dfft(signal, dt):
    """Compute the windowed Discrete Fast Fourier Transform (DFFT) of a signal.

    Parameters
    ----------
    signal : array_like
        Time domain signal.
    dt : float
        Sampling interval [s].

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

    idx = freq >= 0
    freq = freq[idx]
    mag = mag[idx]

    # Internal components duplicated
    if len(mag) > 2:
        mag[1:-1] *= 2

    return freq, mag
