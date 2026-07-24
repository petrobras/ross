import numpy as np
from numba import njit


@njit(fastmath=True)
def involute(angle):
    """Involute function

    Calculates the involute function for a given angle. This function is
    used to describe the contact region of the gear profile.
    """
    return np.tan(angle) - float(angle)


@njit(fastmath=True)
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


@njit(fastmath=True)
def interpolate2d(x, y, x_array, y_array, z_table):
    """Interpolate a 2D table.

    Parameters
    ----------
    x : float
        The x value to interpolate.
    y : float
        The y value to interpolate.
    x_array : np.array
        The x values of the table.
    y_array : np.array
        The y values of the table.
    z_table : np.array
        The 2D array of values with shape `(len(x_array), len(y_array))`.

    Returns
    -------
    z : float
        The interpolated value.
    """
    i = np.searchsorted(x_array, x) - 1
    j = np.searchsorted(y_array, y) - 1

    i = max(0, min(i, len(x_array) - 2))
    j = max(0, min(j, len(y_array) - 2))

    x1, x2 = x_array[i], x_array[i + 1]
    y1, y2 = y_array[j], y_array[j + 1]

    wx = (x - x1) / (x2 - x1) if x2 != x1 else 0.0
    wy = (y - y1) / (y2 - y1) if y2 != y1 else 0.0

    z0, z1 = z_table[i, j : j + 2] * (1 - wx) + z_table[i + 1, j : j + 2] * wx

    return z0 * (1 - wy) + z1 * wy


@njit(fastmath=True)
def compute_contact_ratio(
    center_distance, pr_angle_op, pr_angle_nom, Ra1, Ra2, Rb1, Rb2, module
):
    """Calculate the contact ratio.

    Parameters
    ----------
    center_distance : float
        The center distance between the gears.
    pr_angle_op : float
        The operating pressure angle.
    pr_angle_nom : float
        The nominal pressure angle.
    Ra1 : float
        The addendum radius of the first gear.
    Ra2 : float
        The addendum radius of the second gear.
    Rb1 : float
        The base radius of the first gear.
    Rb2 : float
        The base radius of the second gear.
    module : float
        The module of the gears.

    Returns
    -------
    contact_ratio : float
        The contact ratio.
    """
    contact_length = (
        np.sqrt(Ra1**2 - Rb1**2)
        + np.sqrt(Ra2**2 - Rb2**2)
        - center_distance * np.sin(pr_angle_op)
    )

    base_pitch = np.pi * module * np.cos(pr_angle_nom)

    return contact_length / base_pitch


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
