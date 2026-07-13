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
