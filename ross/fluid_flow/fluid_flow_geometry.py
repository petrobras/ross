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
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_attitude_angle(my_fluid_flow.eccentricity_ratio) # doctest: +ELLIPSIS
    0.93...
    """
    return np.arctan(
        (np.pi * (1 - eccentricity_ratio ** 2) ** (1 / 2)) / (4 * eccentricity_ratio)
    )


def internal_radius_function(gama, attitude_angle, radius_rotor, eccentricity):
    """This function calculates the x and y of the internal radius of the rotor,
    as well as its distance from the origin, given the distance in the theta-axis,
    the attitude angle, the radius of the rotor and the eccentricity.
    Parameters
    ----------
    gama: float
        Gama is the distance in the theta-axis. It should range from 0 to 2*np.pi.
    attitude_angle: float
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
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> attitude_angle = my_fluid_flow.attitude_angle
    >>> radius_rotor = my_fluid_flow.radius_rotor
    >>> eccentricity = my_fluid_flow.eccentricity
    >>> radius_internal, xri, yri = internal_radius_function(0, attitude_angle, radius_rotor, eccentricity)
    >>> radius_internal # doctest: +ELLIPSIS
    0.2...
    """
    if (np.pi / 2 + attitude_angle) < gama < (3 * np.pi / 2 + attitude_angle):
        alpha = np.absolute(3 * np.pi / 2 - gama + attitude_angle)
    else:
        alpha = gama + np.pi / 2 - attitude_angle
    radius_internal = np.sqrt(
        radius_rotor ** 2 - (eccentricity * np.sin(alpha)) ** 2
    ) + eccentricity * np.cos(alpha)
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
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> radius_stator = my_fluid_flow.radius_stator
    >>> radius_external, xre, yre = external_radius_function(0, radius_stator)
    >>> radius_external
    0.2002
    """
    radius_external = radius_stator
    xre = radius_external * np.cos(gama)
    yre = radius_external * np.sin(gama)

    return radius_external, xre, yre


def modified_sommerfeld_number(
    radius_stator, omega, viscosity, length, load, radial_clearance
):
    """Returns the modified sommerfeld number.
    Parameters
    ----------
    radius_stator : float
        The external radius of the bearing.
    omega: float
        Rotation of the rotor (rad/s).
    viscosity: float
        Viscosity (Pa.s).
    length: float
        Length in the Z direction (m).
    load: float
        Load applied to the rotor (N).
    radial_clearance: float
        Difference between both stator and rotor radius, regardless of eccentricity.

    Returns
    -------
    float
        The modified sommerfeld number.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> radius_stator = my_fluid_flow.radius_stator
    >>> omega = my_fluid_flow.omega
    >>> viscosity = my_fluid_flow.viscosity
    >>> length = my_fluid_flow.length
    >>> load = my_fluid_flow.load
    >>> radial_clearance = my_fluid_flow.radial_clearance
    >>> modified_sommerfeld_number(radius_stator, omega, viscosity,
    ...                            length, load, radial_clearance) # doctest: +ELLIPSIS
    0.33...
    """
    return (radius_stator * 2 * omega * viscosity * (length ** 3)) / (
        8 * load * (radial_clearance ** 2)
    )


def sommerfeld_number(modified_s, radius_stator, length):
    """Returns the sommerfeld number, based on the modified sommerfeld number.
    Parameters
    ----------
    modified_s: float
        The modified sommerfeld number.
    radius_stator : float
        The external radius of the bearing.
    length: float
        Length in the Z direction (m).
    Returns
    -------
    float
        The sommerfeld number.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> radius_stator = my_fluid_flow.radius_stator
    >>> omega = my_fluid_flow.omega
    >>> viscosity = my_fluid_flow.viscosity
    >>> length = my_fluid_flow.length
    >>> load = my_fluid_flow.load
    >>> radial_clearance = my_fluid_flow.radial_clearance
    >>> modified_s = modified_sommerfeld_number(radius_stator, omega, viscosity,
    ...                            length, load, radial_clearance) # doctest: +ELLIPSIS
    >>> sommerfeld_number(modified_s, radius_stator, length) # doctest: +ELLIPSIS
    10.62...
    """
    return (modified_s / np.pi) * (radius_stator * 2 / length) ** 2


def calculate_eccentricity_ratio(modified_s):
    """Calculate the eccentricity ratio for a given load using the sommerfeld number.
    Suitable only for short bearings.
    Parameters
    ----------
    modified_s: float
        The modified sommerfeld number.
    Returns
    -------
    float
        The eccentricity ratio.
    Examples
    --------
    >>> modified_s = 6.329494061103412
    >>> calculate_eccentricity_ratio(modified_s) # doctest: +ELLIPSIS
    0.04999...
    """
    coefficients = [
        1,
        -4,
        (6 - (modified_s ** 2) * (16 - np.pi ** 2)),
        -(4 + (np.pi ** 2) * (modified_s ** 2)),
        1,
    ]
    roots = np.roots(coefficients)
    for i in range(0, len(roots)):
        if 0 <= roots[i] <= 1:
            return np.sqrt(roots[i].real)
    raise ValueError("Eccentricity ratio could not be calculated.")


def calculate_rotor_load(
    radius_stator, omega, viscosity, length, radial_clearance, eccentricity_ratio
):
    """Returns the load applied to the rotor, based on the eccentricity ratio.
    Suitable only for short bearings.
    Parameters
    ----------
    radius_stator : float
        The external radius of the bearing.
    omega: float
        Rotation of the rotor (rad/s).
    viscosity: float
        Viscosity (Pa.s).
    length: float
        Length in the Z direction (m).
    radial_clearance: float
        Difference between both stator and rotor radius, regardless of eccentricity.
    eccentricity_ratio: float
        The ratio between the journal displacement, called just eccentricity, and
        the radial clearance.
    Returns
    -------
    float
        Load applied to the rotor.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> radius_stator = my_fluid_flow.radius_stator
    >>> omega = my_fluid_flow.omega
    >>> viscosity = my_fluid_flow.viscosity
    >>> length = my_fluid_flow.length
    >>> radial_clearance = my_fluid_flow.radial_clearance
    >>> eccentricity_ratio = my_fluid_flow.eccentricity_ratio
    >>> calculate_rotor_load(radius_stator, omega, viscosity,
    ...                      length, radial_clearance, eccentricity_ratio) # doctest: +ELLIPSIS
    37.75...
    """
    return (
        (
            np.pi
            * radius_stator
            * 2
            * omega
            * viscosity
            * (length ** 3)
            * eccentricity_ratio
        )
        / (8 * (radial_clearance ** 2) * ((1 - eccentricity_ratio ** 2) ** 2))
    ) * (np.sqrt((16 / (np.pi ** 2) - 1) * eccentricity_ratio ** 2 + 1))


def move_rotor_center(fluid_flow_object, dx, dy):
    """For a given step on x or y axis,
    moves the rotor center and calculates new eccentricity, attitude angle,
    and rotor center.

    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    dx: float
        The step on x axis.
    dy: float
        The step on y axis.
    Returns
    -------
    None
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example2
    >>> my_fluid_flow = fluid_flow_example2()
    >>> move_rotor_center(my_fluid_flow, 0, 0)
    >>> my_fluid_flow.eccentricity # doctest: +ELLIPSIS
    2.6627...
    """
    fluid_flow_object.xi = fluid_flow_object.xi + dx
    fluid_flow_object.yi = fluid_flow_object.yi + dy
    fluid_flow_object.eccentricity = np.sqrt(
        fluid_flow_object.xi ** 2 + fluid_flow_object.yi ** 2
    )
    fluid_flow_object.eccentricity_ratio = (
        fluid_flow_object.eccentricity / fluid_flow_object.difference_between_radius
    )
    fluid_flow_object.attitude_angle = np.arccos(
        abs(fluid_flow_object.yi / fluid_flow_object.eccentricity)
    )
