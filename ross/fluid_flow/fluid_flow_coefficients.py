import numpy as np
from scipy import integrate
from math import isnan
from ross.fluid_flow.fluid_flow_geometry import calculate_attitude_angle, move_rotor_center
from copy import deepcopy


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
        base_vector = np.array([fluid_flow_object.xre[0][0] - fluid_flow_object.xi,
                                fluid_flow_object.yre[0][0] - fluid_flow_object.yi])
        for i in range(fluid_flow_object.nz):
            for j in range(int(fluid_flow_object.ntheta/2)):
                vector_from_rotor = np.array([fluid_flow_object.xre[i][j] - fluid_flow_object.xi,
                                              fluid_flow_object.yre[i][j] - fluid_flow_object.yi])
                angle_between_vectors = np.arccos(np.dot(base_vector, vector_from_rotor) /
                                                  (np.linalg.norm(base_vector) * np.linalg.norm(vector_from_rotor)))
                if isnan(angle_between_vectors):
                    angle_between_vectors = 0
                if angle_between_vectors != 0 and j*fluid_flow_object.dtheta > np.pi:
                    angle_between_vectors += np.pi
                a[i][j] = p_mat[i][j] * np.cos(angle_between_vectors)
                b[i][j] = p_mat[i][j] * np.sin(angle_between_vectors)

        for i in range(fluid_flow_object.nz):
            g1[i] = integrate.simps(a[i][:], fluid_flow_object.gama[0])
            g2[i] = integrate.simps(b[i][:], fluid_flow_object.gama[0])

        integral1 = integrate.simps(g1, fluid_flow_object.z_list)
        integral2 = integrate.simps(g2, fluid_flow_object.z_list)

        radial_force = - fluid_flow_object.radius_rotor * integral1
        tangential_force = fluid_flow_object.radius_rotor * integral2
    force_x = - radial_force * np.sin(fluid_flow_object.attitude_angle) \
        + tangential_force * np.cos(fluid_flow_object.attitude_angle)
    force_y = radial_force * np.cos(fluid_flow_object.attitude_angle) \
        + tangential_force * np.sin(fluid_flow_object.attitude_angle)
    return radial_force, tangential_force, force_x, force_y


def calculate_stiffness_matrix(fluid_flow_object, oil_film_force='numerical'):
    """This function calculates the bearing stiffness matrix numerically.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    oil_film_force: str
        If set, calculates the oil film force analytically considering the chosen type: 'short' or 'long'.
        If set to 'numerical', calculates the oil film force numerically.
    Returns
    -------
    list of floats
        A list of length four including stiffness floats in this order: kxx, kxy, kyx, kyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_stiffness_matrix(my_fluid_flow)  # doctest: +ELLIPSIS
    [0.003...
    """
    [radial_force, tangential_force, force_x, force_y] = \
        calculate_oil_film_force(fluid_flow_object, force_type=oil_film_force)
    delta = fluid_flow_object.difference_between_radius / 100

    fluid_flow_x = deepcopy(fluid_flow_object)
    move_rotor_center(fluid_flow_x, delta, 0)
    fluid_flow_x.calculate_coefficients()
    fluid_flow_x.calculate_pressure_matrix_numerical()
    [radial_force_x, tangential_force_x, force_x_x, force_y_x] = \
        calculate_oil_film_force(fluid_flow_x, force_type=oil_film_force)

    fluid_flow_y = deepcopy(fluid_flow_object)
    move_rotor_center(fluid_flow_y, 0, delta)
    fluid_flow_y.calculate_coefficients()
    fluid_flow_y.calculate_pressure_matrix_numerical()
    [radial_force_y, tangential_force_y, force_x_y, force_y_y] = \
        calculate_oil_film_force(fluid_flow_y, force_type=oil_film_force)

    k_xx = (force_x - force_x_x) / delta
    k_yx = (force_y - force_y_x) / delta
    k_xy = (force_x - force_x_y) / delta
    k_yy = (force_y - force_y_y) / delta

    return [k_xx, k_xy, k_yx, k_yy]


def find_equilibrium_position(fluid_flow_object, print_along=True, tolerance=1e-05,
                              increment_factor=1e-03):
    """This function returns an eccentricity value with calculated forces matching the load applied,
    meaning an equilibrium position of the rotor.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    print_along: bool, optional
        If True, prints the iteration process.
    tolerance: float, optional
    increment_factor: float, optional
        This number will multiply the first eccentricity found to reach an increment number.
    Returns
    -------
    float
        Eccentricity of the equilibrium position.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example2
    >>> my_fluid_flow = fluid_flow_example2()
    >>> eccentricity = find_equilibrium_position(my_fluid_flow, print_along=False, tolerance=0.1)
    """
    r_force, t_force, force_x, force_y = calculate_oil_film_force(fluid_flow_object, force_type='numerical')
    increment = increment_factor * fluid_flow_object.eccentricity
    resultant_force = np.sqrt(r_force**2 + t_force**2)
    error = abs(resultant_force - fluid_flow_object.load)
    k = 0
    while error > tolerance:
        k += 1
        fluid_flow_object.eccentricity = fluid_flow_object.eccentricity + increment
        fluid_flow_object.eccentricity_ratio = fluid_flow_object.eccentricity / \
            fluid_flow_object.difference_between_radius
        fluid_flow_object.attitude_angle = calculate_attitude_angle(fluid_flow_object.eccentricity_ratio)
        fluid_flow_object.calculate_coefficients()
        fluid_flow_object.calculate_pressure_matrix_numerical()
        r_force, t_force, new_force_x, new_force_y = calculate_oil_film_force(fluid_flow_object, force_type='numerical')
        new_resultant_force = np.sqrt(r_force ** 2 + t_force ** 2)
        new_error = abs(new_resultant_force - fluid_flow_object.load)
        if (new_resultant_force - fluid_flow_object.load) * (resultant_force - fluid_flow_object.load) < 0:
            increment = -increment/10
        resultant_force = new_resultant_force
        error = new_error
        if print_along:
            print("Iteration " + str(k))
            print("Eccentricity: " + str(fluid_flow_object.eccentricity))
            print("Resultant force minus load: " + str(resultant_force - fluid_flow_object.load))
    return fluid_flow_object.eccentricity

















