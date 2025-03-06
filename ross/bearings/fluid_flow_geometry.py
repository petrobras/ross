import numpy as np
from scipy.optimize import least_squares, root


def calculate_attitude_angle(eccentricity_ratio):
    """Calculates the attitude angle based on the eccentricity ratio.
    Suitable only for short bearings.
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
    >>> from ross.bearings.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_attitude_angle(my_fluid_flow.eccentricity_ratio) # doctest: +ELLIPSIS
    0.93...
    """
    return np.arctan(
        (np.pi * (1 - eccentricity_ratio**2) ** (1 / 2)) / (4 * eccentricity_ratio)
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
    >>> from ross.bearings.fluid_flow import fluid_flow_example
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
        radius_rotor**2 - (eccentricity * np.sin(alpha)) ** 2
    ) + eccentricity * np.cos(alpha)
    xri = radius_internal * np.cos(gama)
    yri = radius_internal * np.sin(gama)

    return radius_internal, xri, yri


def external_radius_function(
    gama,
    radius_stator,
    radius_rotor=None,
    shape="cylindrical",
    preload=None,
    displacement=None,
    max_depth=None,
):
    """This function returns the x and y of the radius of the stator, as well as its distance from the
    origin, given the distance in the theta-axis and the radius of the bearing.
    Parameters
    ----------
    gama: float
        Gama is the distance in the theta-axis. It should range from 0 to 2*np.pi.
    radius_stator : float
        The external radius of the bearing.
    radius_rotor : float
        The internal radius of the bearing.
    shape : str
        Determines the type of bearing geometry.
        'cylindrical': cylindrical bearing;
        'eliptical': eliptical bearing;
        'wear': journal bearing wear.
        The default is 'cylindrical'.
    preload : float
        The ellipticity ratio of the bearing if the shape is eliptical. Varies between 0 and 1.
        The default is 0.05.
    displacement : float
        Angular displacement of the bearing wear in relation to the vertical axis.
    max_depth: float
        The maximum wear depth.
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
    >>> from ross.bearings.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> radius_stator = my_fluid_flow.radius_stator
    >>> radius_external, xre, yre = external_radius_function(0, radius_stator)
    >>> radius_external
    0.2002
    """
    if shape == "eliptical":
        cr = radius_stator - radius_rotor
        elip = preload * cr
        if 0 <= gama <= np.pi / 2:
            alpha = np.pi / 2 + gama
        elif np.pi / 2 < gama <= np.pi:
            alpha = 3 * np.pi / 2 - gama
        elif np.pi < gama <= 3 * np.pi / 2:
            alpha = gama - np.pi / 2
        else:
            alpha = 5 * np.pi / 2 - gama

        radius_external = elip * np.cos(alpha) + np.sqrt(
            ((radius_stator) ** 2) - (elip * np.sin(alpha)) ** 2
        )
        xre = radius_external * np.cos(gama)
        yre = radius_external * np.sin(gama)

    elif shape == "wear":
        if max_depth == 0:
            d_theta = 0
        else:
            cr = radius_stator - radius_rotor
            theta_s = np.pi / 2 + np.arccos(max_depth / cr - 1) + displacement
            theta_f_0 = np.pi / 2 - np.arccos(max_depth / cr - 1) + displacement
            theta_f = 2 * np.pi + theta_f_0

            if theta_f <= 2 * np.pi:
                if theta_s <= gama <= theta_f:
                    d_theta = max_depth - cr * (
                        1 + np.cos(gama - np.pi / 2 - displacement)
                    )
                else:
                    d_theta = 0
            else:
                if theta_s <= gama <= 2 * np.pi:
                    d_theta = max_depth - cr * (
                        1 + np.cos(gama - np.pi / 2 - displacement)
                    )
                elif 0 <= gama <= theta_f_0:
                    gama2 = gama + 2 * np.pi
                    d_theta = max_depth - cr * (
                        1 + np.cos(gama2 - np.pi / 2 - displacement)
                    )
                else:
                    d_theta = 0

        radius_external = radius_stator + d_theta
        xre = radius_external * np.cos(gama)
        yre = radius_external * np.sin(gama)

    else:
        radius_external = radius_stator
        xre = radius_external * np.cos(gama)
        yre = radius_external * np.sin(gama)

    return radius_external, xre, yre


def reynolds_number(density, characteristic_speed, radial_clearance, viscosity):
    """Returns the reynolds number based on the characteristic speed.
    This number denotes the ratio between fluid inertia (advection) forces and viscous-shear forces.
        Parameters
        ----------
        density: float
            Fluid density(Kg/m^3).
        characteristic_speed: float
            Characteristic fluid speeds.
        radial_clearance: float
            Difference between both stator and rotor radius, regardless of eccentricity.
        viscosity: float
            Viscosity (Pa.s).
        Returns
        -------
        float
            The reynolds number.
        Examples
        --------
        >>> from ross.bearings.fluid_flow import fluid_flow_example
        >>> my_fluid_flow = fluid_flow_example()
        >>> density = my_fluid_flow.density
        >>> characteristic_speed = my_fluid_flow.characteristic_speed
        >>> radial_clearance = my_fluid_flow.radial_clearance
        >>> viscosity = my_fluid_flow.viscosity
        >>> reynolds_number(density, characteristic_speed, radial_clearance, viscosity) # doctest: +ELLIPSIS
        24.01...
    """
    return (density * characteristic_speed * radial_clearance) / viscosity


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
    >>> from ross.bearings.fluid_flow import fluid_flow_example
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
    return (radius_stator * 2 * omega * viscosity * (length**3)) / (
        8 * load * (radial_clearance**2)
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
    >>> from ross.bearings.fluid_flow import fluid_flow_example
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
        (6 - (modified_s**2) * (16 - np.pi**2)),
        -(4 + (np.pi**2) * (modified_s**2)),
        1,
    ]
    roots = np.roots(coefficients)
    roots = np.sort(roots[np.isclose(roots.imag, 1e-16)].real)

    def f(x):
        """Fourth degree polynomial whose root is the square of the eccentricity ratio.
        Parameters
        ----------
        x: float
            Fourth degree polynomial coefficients.
        Returns
        -------
        float
            Polynomial value f at x."""
        c = coefficients
        return c[0] * x**4 + c[1] * x**3 + c[2] * x**2 + c[3] * x**1 + c[4] * x**0

    for i in roots:
        if 0 <= i <= 1:
            roots = root(f, i, tol=1e-10).x[0]
            return np.sqrt(roots)
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
    >>> from ross.bearings.fluid_flow import fluid_flow_example
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
            * (length**3)
            * eccentricity_ratio
        )
        / (8 * (radial_clearance**2) * ((1 - eccentricity_ratio**2) ** 2))
    ) * (np.sqrt((16 / (np.pi**2) - 1) * eccentricity_ratio**2 + 1))


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
    >>> from ross.bearings.fluid_flow import fluid_flow_example2
    >>> my_fluid_flow = fluid_flow_example2()
    >>> move_rotor_center(my_fluid_flow, 0, 0)
    >>> my_fluid_flow.eccentricity # doctest: +ELLIPSIS
    2.5...
    """
    fluid_flow_object.xi = fluid_flow_object.xi + dx
    fluid_flow_object.yi = fluid_flow_object.yi + dy
    fluid_flow_object.eccentricity = np.sqrt(
        fluid_flow_object.xi**2 + fluid_flow_object.yi**2
    )
    fluid_flow_object.eccentricity_ratio = (
        fluid_flow_object.eccentricity / fluid_flow_object.radial_clearance
    )
    fluid_flow_object.attitude_angle = np.arccos(
        abs(fluid_flow_object.yi / fluid_flow_object.eccentricity)
    )


def move_rotor_center_abs(fluid_flow_object, x, y):
    """Moves the rotor center to the coordinates (x, y) and calculates new eccentricity,
    attitude angle, and rotor center.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    x: float
        Coordinate along x axis.
    y: float
        Coordinate along y axis.
    Returns
    -------
    None
    --------
    >>> from ross.bearings.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> move_rotor_center_abs(my_fluid_flow, 0, -1e-3*my_fluid_flow.radial_clearance)
    >>> my_fluid_flow.eccentricity # doctest: +ELLIPSIS
    1.99...
    """
    fluid_flow_object.xi = x
    fluid_flow_object.yi = y
    fluid_flow_object.eccentricity = np.sqrt(
        fluid_flow_object.xi**2 + fluid_flow_object.yi**2
    )
    fluid_flow_object.eccentricity_ratio = (
        fluid_flow_object.eccentricity / fluid_flow_object.radial_clearance
    )
    fluid_flow_object.attitude_angle = np.arccos(
        abs(fluid_flow_object.yi / fluid_flow_object.eccentricity)
    )
