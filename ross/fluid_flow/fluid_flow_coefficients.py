import numpy as np
from scipy import integrate


def calculate_analytical_stiffness_matrix(load, eccentricity_ratio, radial_clearance):
    """Returns the stiffness matrix calculated analytically.
    Suitable only for short bearings.
    Parameters
    -------
    load: float
        Load applied to the rotor (N).
    eccentricity_ratio: float
        The ratio between the journal displacement, called just eccentricity, and
        the radial clearance.
    radial_clearance: float
        Difference between both stator and rotor radius, regardless of eccentricity.
    Returns
    -------
    list of floats
        A list of length four including stiffness floats in this order: kxx, kxy, kyx, kyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> load = my_fluid_flow.load
    >>> eccentricity_ratio = my_fluid_flow.eccentricity_ratio
    >>> radial_clearance = my_fluid_flow.radial_clearance
    >>> calculate_analytical_stiffness_matrix(load, eccentricity_ratio, radial_clearance) # doctest: +ELLIPSIS
    [...
    """
    # fmt: off
    h0 = 1.0 / (((np.pi ** 2) * (1 - eccentricity_ratio ** 2) + 16 * eccentricity_ratio ** 2) ** 1.5)
    a = load / radial_clearance
    kxx = a * h0 * 4 * ((np.pi ** 2) * (2 - eccentricity_ratio ** 2) + 16 * eccentricity_ratio ** 2)
    kxy = (a * h0 * np.pi * ((np.pi ** 2) * (1 - eccentricity_ratio ** 2) ** 2 -
                             16 * eccentricity_ratio ** 4) /
           (eccentricity_ratio * np.sqrt(1 - eccentricity_ratio ** 2)))
    kyx = (-a * h0 * np.pi * ((np.pi ** 2) * (1 - eccentricity_ratio ** 2) *
                              (1 + 2 * eccentricity_ratio ** 2) +
                              (32 * eccentricity_ratio ** 2) * (1 + eccentricity_ratio ** 2)) /
           (eccentricity_ratio * np.sqrt(1 - eccentricity_ratio ** 2)))
    kyy = (a * h0 * 4 * ((np.pi ** 2) * (1 + 2 * eccentricity_ratio ** 2) +
                         ((32 * eccentricity_ratio ** 2) *
                          (1 + eccentricity_ratio ** 2)) / (1 - eccentricity_ratio ** 2)))
    # fmt: on
    return [kxx, kxy, kyx, kyy]


def calculate_analytical_damping_matrix(load, eccentricity_ratio, radial_clearance, omega):
    """Returns the damping matrix calculated analytically.
    Suitable only for short bearings.
    Parameters
    -------
    load: float
        Load applied to the rotor (N).
    eccentricity_ratio: float
        The ratio between the journal displacement, called just eccentricity, and
        the radial clearance.
    radial_clearance: float
        Difference between both stator and rotor radius, regardless of eccentricity.
    omega: float
        Rotation of the rotor (rad/s).
    Returns
    -------
    list of floats
        A list of length four including damping floats in this order: cxx, cxy, cyx, cyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> load = my_fluid_flow.load
    >>> eccentricity_ratio = my_fluid_flow.eccentricity_ratio
    >>> radial_clearance = my_fluid_flow.radial_clearance
    >>> omega = my_fluid_flow.omega
    >>> calculate_analytical_damping_matrix(load, eccentricity_ratio,
    ...                                       radial_clearance, omega) # doctest: +ELLIPSIS
    [...
    """
    # fmt: off
    h0 = 1.0 / (((np.pi ** 2) * (1 - eccentricity_ratio ** 2) + 16 * eccentricity_ratio ** 2) ** 1.5)
    a = load / (radial_clearance * omega)
    cxx = (a * h0 * 2 * np.pi * np.sqrt(1 - eccentricity_ratio ** 2) *
           ((np.pi ** 2) * (
                   1 + 2 * eccentricity_ratio ** 2) - 16 * eccentricity_ratio ** 2) / eccentricity_ratio)
    cxy = (-a * h0 * 8 * (
            (np.pi ** 2) * (1 + 2 * eccentricity_ratio ** 2) - 16 * eccentricity_ratio ** 2))
    cyx = cxy
    cyy = (a * h0 * (2 * np.pi * (
            (np.pi ** 2) * (1 - eccentricity_ratio ** 2) ** 2 + 48 * eccentricity_ratio ** 2)) /
           (eccentricity_ratio * np.sqrt(1 - eccentricity_ratio ** 2)))
    # fmt: on
    return [cxx, cxy, cyx, cyy]


def calculate_oil_film_force(fluid_flow_object, force_type=None):
    """This function calculates the forces of the oil film in the N and T directions, ie in the
    opposite direction to the eccentricity and in the tangential direction.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    force_type: str
        If set, calculates the oil film force matrix analytically considering the chosen type: 'short' or 'long'.
        If set to 'numerical', calculates the oil film force numerically.
    Returns
    -------
    radial_force: float
        Force of the oil film in the opposite direction to the eccentricity direction.
    tangential_force: float
        Force of the oil film in the tangential direction
    f_x: float
        Components of forces in the x direction
    f_y: float
        Components of forces in the y direction
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_oil_film_force(my_fluid_flow) # doctest: +ELLIPSIS
    (...
    """
    if force_type != 'numerical' and (force_type == 'short' or fluid_flow_object.bearing_type == 'short_bearing'):
        radial_force = 0.5 * \
           fluid_flow_object.viscosity * \
           (fluid_flow_object.radius_rotor / fluid_flow_object.difference_between_radius) ** 2 * \
           (fluid_flow_object.length ** 3 / fluid_flow_object.radius_rotor) * \
           ((2 * fluid_flow_object.eccentricity_ratio ** 2 * fluid_flow_object.omega) /
            (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 2)

        tangential_force = 0.5 * fluid_flow_object.viscosity * \
            (fluid_flow_object.radius_rotor / fluid_flow_object.difference_between_radius) ** 2 * \
            (fluid_flow_object.length ** 3 / fluid_flow_object.radius_rotor) * (
             (np.pi * fluid_flow_object.eccentricity_ratio * fluid_flow_object.omega) /
             (2 * (1 - fluid_flow_object.eccentricity_ratio ** 2) ** (3. / 2)))
    elif force_type != 'numerical' and (force_type == 'long' or fluid_flow_object.bearing_type == 'long_bearing'):
        radial_force = 6 * fluid_flow_object.viscosity * \
            (fluid_flow_object.radius_rotor / fluid_flow_object.difference_between_radius) ** 2 * \
            fluid_flow_object.radius_rotor * fluid_flow_object.length * \
            ((2 * fluid_flow_object.eccentricity_ratio ** 2 * fluid_flow_object.omega) /
             ((2 + fluid_flow_object.eccentricity_ratio ** 2) *
              (1 - fluid_flow_object.eccentricity_ratio ** 2)))
        tangential_force = 6 * fluid_flow_object.viscosity * (fluid_flow_object.radius_rotor / fluid_flow_object.difference_between_radius) ** 2 * \
            fluid_flow_object.radius_rotor * fluid_flow_object.length * \
            ((np.pi * fluid_flow_object.eccentricity_ratio * fluid_flow_object.omega) /
             ((2 + fluid_flow_object.eccentricity_ratio ** 2) *
             (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 0.5))
    else:
        p_mat = fluid_flow_object.p_mat_numerical
        a = np.zeros([fluid_flow_object.nz, fluid_flow_object.ntheta])
        b = np.zeros([fluid_flow_object.nz, fluid_flow_object.ntheta])
        g1 = np.zeros(fluid_flow_object.nz)
        g2 = np.zeros(fluid_flow_object.nz)
        for i in range(fluid_flow_object.nz):
            for j in range(fluid_flow_object.ntheta):
                a[i][j] = p_mat[i][j] * np.cos(j*fluid_flow_object.dtheta)
                b[i][j] = p_mat[i][j] * np.sin(j*fluid_flow_object.dtheta)

        for i in range(fluid_flow_object.nz):
            g1[i] = integrate.simps(a[i][:], fluid_flow_object.gama[0])
            g2[i] = integrate.simps(b[i][:], fluid_flow_object.gama[0])

        integral1 = integrate.simps(g1, fluid_flow_object.z_list)
        integral2 = integrate.simps(g2, fluid_flow_object.z_list)

        radial_force = - fluid_flow_object.radius_rotor * integral1
        tangential_force = fluid_flow_object.radius_rotor * integral2
    force_x = - radial_force * np.cos(np.pi / 2 - fluid_flow_object.attitude_angle) \
              + tangential_force * np.sin(np.pi / 2 - fluid_flow_object.attitude_angle)
    force_y = - radial_force * np.sin(np.pi / 2 - fluid_flow_object.attitude_angle) \
              - tangential_force * np.cos(np.pi / 2 - fluid_flow_object.attitude_angle)
    return radial_force, tangential_force, force_x, force_y


def calculate_stiffness_matrix(fluid_flow_object, oil_film_force=None):
    """This function calculates the bearing stiffness matrix numerically.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    oil_film_force: str
        If set, calculates the oil film force analytically considering the chosen type: 'short' or 'long'.
    Returns
    -------
    list of floats
        A list of length four including stiffness floats in this order: kxx, kxy, kyx, kyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_stiffness_matrix(my_fluid_flow)
    """
    [radial_force, tangential_force, force_x, force_y] = calculate_oil_film_force(fluid_flow_object, force_type='numerical')
    temp = fluid_flow_object.eccentricity
    temp_angle = fluid_flow_object.attitude_angle
    e = fluid_flow_object.eccentricity
    beta = fluid_flow_object.attitude_angle
    delta_x = fluid_flow_object.difference_between_radius / 100
    eccentricity_x = (e ** 2 + delta_x ** 2 - 2 * e * delta_x *
                      np.cos(np.pi / 2 + fluid_flow_object.attitude_angle))**(1/2)
    beta_x = np.arccos((e**2 + eccentricity_x**2 - delta_x**2) / 2 * e * eccentricity_x)
    fluid_flow_object.eccentricity = eccentricity_x
    fluid_flow_object.attitude_angle = beta + beta_x
    fluid_flow_object.calculate_coefficients()
    fluid_flow_object.calculate_pressure_matrix_numerical()
    [radial_force_x, tangential_force_x, force_x_x, force_y_x] = calculate_oil_film_force(fluid_flow_object, force_type='numerical')
    delta_y = fluid_flow_object.difference_between_radius / 100
    eccentricity_y = (e ** 2 + delta_y ** 2 - 2 * e * delta_y * np.cos(fluid_flow_object.attitude_angle))**(1/2)
    beta_y = np.arccos((e ** 2 + eccentricity_y ** 2 - delta_y ** 2) / 2 * e * eccentricity_y)
    fluid_flow_object.eccentricity = eccentricity_y
    fluid_flow_object.attitude_angle = beta + beta_y
    fluid_flow_object.calculate_coefficients()
    fluid_flow_object.calculate_pressure_matrix_numerical()
    [radial_force_y, tangential_force_y, force_x_y, force_y_y] = calculate_oil_film_force(fluid_flow_object, force_type='numerical')
    k_xx = (radial_force_x - radial_force) / delta_x
    k_yx = (tangential_force_x - tangential_force) / delta_x
    k_xy = (radial_force_y - radial_force) / delta_y
    k_yy = (tangential_force_y - tangential_force) / delta_y
    fluid_flow_object.eccentricity = temp
    fluid_flow_object.attitude_angle = temp_angle
    fluid_flow_object.calculate_coefficients()
    fluid_flow_object.calculate_pressure_matrix_numerical()
    return [k_xx, k_yx, k_xy, k_yy]
