import numpy as np


def calculate_attitude_angle(eccentricity_ratio):
    """Calculates the attitude angle based on the eccentricity ratio.
    Parameters
    ----------
    eccentricity_ratio: float
        The ratio between the journal displacement, called just eccentricity, and
        the radial clearance.
    Returns
    -------
    float
        Attitude angle
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> calculate_attitude_angle(my_fluid_flow.eccentricity_ratio) # doctest: +ELLIPSIS
    1.5...
    """
    return np.arctan(np.pi * (1 - eccentricity_ratio ** 2) /
                     (4 * eccentricity_ratio))


def internal_radius_function(gama, beta, radius_rotor, eccentricity):
    """This function calculates the x and y of the internal radius of the rotor,
    as well as its distance from the origin, given the distance in the theta-axis,
    the attitude angle, the radius of the rotor and the eccentricity.
    Parameters
    ----------
    gama: float
        Gama is the distance in the theta-axis. It should range from 0 to 2*np.pi.
    beta: float
        Attitude angle. Angle between the origin and the eccentricity (rad).
    radius_rotor: float
        The radius of the journal.
    eccentricity: float
        The journal displacement from the center of the stator.
    Returns
    -------
    radius_internal: float
        The size of the internal radius at that point.
    xri: float
        The position x of the returned internal radius.
    yri: float
        The position y of the returned internal radius.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> beta = my_fluid_flow.beta
    >>> radius_rotor = my_fluid_flow.radius_rotor
    >>> eccentricity = my_fluid_flow.eccentricity
    >>> radius_internal, xri, yri = internal_radius_function(0, beta, radius_rotor, eccentricity)
    >>> radius_internal
    0.079
    """
    if (np.pi - beta) < gama < (2 * np.pi - beta):
        alpha = np.absolute(2 * np.pi - gama - beta)
    else:
        alpha = beta + gama
    radius_internal = np.sqrt(radius_rotor ** 2 - (eccentricity * np.sin(alpha)) ** 2) + eccentricity * np.cos(alpha)
    xri = radius_internal * np.cos(gama)
    yri = radius_internal * np.sin(gama)

    return radius_internal, xri, yri


def external_radius_function(gama, radius_stator):
    """This function returns the x and y of the radius of the stator, as well as its distance from the
    origin, given the distance in the theta-axis and the radius of the bearing.
    Parameters
    ----------
    gama: float
        Gama is the distance in the theta-axis. It should range from 0 to 2*np.pi.
    radius_stator : float
        The external radius of the bearing.
    Returns
    -------
    radius_external: float
        The size of the external radius at that point.
    xre: float
        The position x of the returned external radius.
    yre: float
        The position y of the returned external radius.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> radius_stator = my_fluid_flow.radius_stator
    >>> radius_external, xre, yre = external_radius_function(0, radius_stator)
    >>> radius_external
    0.1
    """
    radius_external = radius_stator
    xre = radius_external * np.cos(gama)
    yre = radius_external * np.sin(gama)

    return radius_external, xre, yre
