import numpy as np


def involute(angle):
    """Involute function

    Calculates the involute function for a given angle. This function is
    used to describe the contact region of the gear profile.
    """
    return np.tan(angle) - float(angle)


def mod(val, max_val):
    """Calculates the remainder of a division, but replaces 0 with max_val.

    Parameters
    ----------
    val : float or array-like
        The value(s) to be divided.
    max_val : float
        The divisor.

    Returns
    -------
    mod : float or array-like
        The remainder of the division, or max_val if the remainder is 0 and val is not 0.
    """
    mod = np.mod(val, max_val)
    return np.where((np.isclose(mod, 0)) & (val != 0), max_val, mod)


def compute_dfft(y, dt, window=True):
    """Compute dFFT - Discrete Fourier Transform.

    Parameters
    ----------
    y : np.array
        Magnitude of the response in time domain (m).
    dt : int
        Time step (s).
    window : bool, optional
        If True, a Hann window is applied to the signal.
        Default is True.

    Returns
    -------
    freq : np.array
        Frequency range (Hz).
    y_amp : np.array
        Amplitude of the response in frequency domain (m).
    y_phase : np.array
        Phase of the response in frequency domain (rad).
    """
    N = len(y)

    y_centered = y - np.mean(y)

    correction = 1.0
    if window:
        w = np.hanning(N)
        y_centered = y_centered * w
        correction = 1.0 / np.mean(w)

    y_full = np.fft.rfft(y_centered)
    y_amp = (2.0 / N) * np.abs(y_full) * correction
    freq = np.fft.rfftfreq(N, d=dt)

    return freq, y_amp
