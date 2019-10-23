import numpy as np
from scipy import integrate
from math import isnan
from ross.fluid_flow.fluid_flow_geometry import calculate_rotor_load


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


def find_equilibrium_position(fluid_flow_object, print_along=True, relative_tolerance=1e-07,
                              numerical_fit=False):
    """This function returns an eccentricity value with calculated forces matching the load applied,
    meaning an equilibrium position of the rotor.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    print_along: bool, optional
        If True, prints the iteration process.
    relative_tolerance: float, optional
    numerical_fit: bool, optional
        If True, it makes sure the result matches numerically, changing attributes of the
        FluidFlow object.
    Returns
    -------
    float
        Eccentricity of the equilibrium position.
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example2
    >>> my_fluid_flow = fluid_flow_example2()
    >>> find_equilibrium_position(my_fluid_flow, print_along=False) # doctest: +ELLIPSIS
    0.00010000...
    """
    fluid_flow_object.calculate_pressure_matrix_numerical()
    load = calculate_rotor_load(fluid_flow_object.radius_stator, fluid_flow_object.omega,
                                fluid_flow_object.viscosity, fluid_flow_object.length,
                                fluid_flow_object.radial_clearance, fluid_flow_object.eccentricity_ratio)
    error = (load - fluid_flow_object.load)/fluid_flow_object.load
    increment = 0.1
    k = 0
    eccentricity = fluid_flow_object.eccentricity
    while np.abs(error) > relative_tolerance:
        eccentricity = eccentricity + (eccentricity * increment)
        eccentricity_ratio = eccentricity / fluid_flow_object.difference_between_radius
        load = calculate_rotor_load(fluid_flow_object.radius_stator, fluid_flow_object.omega,
                                    fluid_flow_object.viscosity, fluid_flow_object.length,
                                    fluid_flow_object.radial_clearance, eccentricity_ratio)
        new_error = (load - fluid_flow_object.load) / fluid_flow_object.load
        if print_along:
            print("Iteration " + str(k))
            print("Eccentricity: " + str(eccentricity))
            print("Load: " + str(load))
            print("Error: " + str(new_error))
        if error * new_error < 0:
            if print_along:
                print("Error changed sign. Changing sign of increment and reducing it.")
            increment = -increment/10
        elif abs(new_error) > abs(error):
            if print_along:
                print("Error was greater than previous one. Changing sign of increment and slightly "
                      "reducing it.")
            increment = -increment/5
        error = new_error
        k += 1
        if print_along:
            print()
    return eccentricity

















