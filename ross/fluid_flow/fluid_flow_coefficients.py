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
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
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
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
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


def calculate_oil_film_force(pressure_matrix_object, force_type=None):
    """This function calculates the forces of the oil film in the N and T directions, ie in the
    opposite direction to the eccentricity and in the tangential direction.
    Parameters
    ----------
    pressure_matrix_object: A PressureMatrix object.
    force_type: str
        If set, calculates the oil film force matrix analytically considering the chosen type: 'short' or 'long'.
        If set to 'numerical', calculates the oil film force numerically.
    Returns
    -------
    normal_force: float
        Force of the oil film in the opposite direction to the eccentricity direction.
    tangential_force: float
        Force of the oil film in the tangential direction
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import pressure_matrix_example
    >>> my_fluid_flow = pressure_matrix_example()
    >>> calculate_oil_film_force(my_fluid_flow) # doctest: +ELLIPSIS
    (...
    """
    if force_type != 'numerical' and (force_type == 'short' or pressure_matrix_object.bearing_type == 'short_bearing'):
        normal_force = 0.5 * \
           pressure_matrix_object.viscosity * \
           (pressure_matrix_object.radius_rotor / pressure_matrix_object.difference_between_radius) ** 2 * \
           (pressure_matrix_object.length ** 3 / pressure_matrix_object.radius_rotor) * \
           ((2 * pressure_matrix_object.eccentricity_ratio ** 2 * pressure_matrix_object.omega) /
            (1 - pressure_matrix_object.eccentricity_ratio ** 2) ** 2)

        tangential_force = 0.5 * pressure_matrix_object.viscosity * \
            (pressure_matrix_object.radius_rotor / pressure_matrix_object.difference_between_radius) ** 2 * \
            (pressure_matrix_object.length ** 3 / pressure_matrix_object.radius_rotor) * (
             (np.pi * pressure_matrix_object.eccentricity_ratio * pressure_matrix_object.omega) /
             (2 * (1 - pressure_matrix_object.eccentricity_ratio ** 2) ** (3. / 2)))
    elif force_type != 'numerical' and (force_type == 'long' or pressure_matrix_object.bearing_type == 'long_bearing'):
        normal_force = 6 * pressure_matrix_object.viscosity * \
            (pressure_matrix_object.radius_rotor / pressure_matrix_object.difference_between_radius) ** 2 * \
            pressure_matrix_object.radius_rotor * pressure_matrix_object.length * \
            ((2 * pressure_matrix_object.eccentricity_ratio ** 2 * pressure_matrix_object.omega) /
             ((2 + pressure_matrix_object.eccentricity_ratio ** 2) *
              (1 - pressure_matrix_object.eccentricity_ratio ** 2)))
        tangential_force = 6 * pressure_matrix_object.viscosity * (pressure_matrix_object.radius_rotor / pressure_matrix_object.difference_between_radius) ** 2 * \
            pressure_matrix_object.radius_rotor * pressure_matrix_object.length * \
            ((np.pi * pressure_matrix_object.eccentricity_ratio * pressure_matrix_object.omega) /
             ((2 + pressure_matrix_object.eccentricity_ratio ** 2) *
             (1 - pressure_matrix_object.eccentricity_ratio ** 2) ** 0.5))
    else:
        p_mat = pressure_matrix_object.p_mat_numerical
        a = np.zeros([pressure_matrix_object.nz, pressure_matrix_object.ntheta])
        b = np.zeros([pressure_matrix_object.nz, pressure_matrix_object.ntheta])
        g1 = np.zeros(pressure_matrix_object.nz)
        g2 = np.zeros(pressure_matrix_object.nz)
        for i in range(pressure_matrix_object.nz):
            for j in range(pressure_matrix_object.ntheta):
                a[i][j] = p_mat[i][j] * np.cos(pressure_matrix_object.gama[i][j])
                b[i][j] = p_mat[i][j] * np.sin(pressure_matrix_object.gama[i][j])

        for i in range(pressure_matrix_object.nz):
            g1[i] = integrate.simps(a[i][:], pressure_matrix_object.gama[0])
            g2[i] = integrate.simps(b[i][:], pressure_matrix_object.gama[0])

        integral1 = integrate.simps(g1, pressure_matrix_object.z_list)
        integral2 = integrate.simps(g2, pressure_matrix_object.z_list)

        normal_force = - pressure_matrix_object.radius_rotor * integral1
        tangential_force = pressure_matrix_object.radius_rotor * integral2
    return normal_force, tangential_force










