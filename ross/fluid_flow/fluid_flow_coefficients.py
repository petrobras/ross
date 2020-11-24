import sys
from math import isnan

import numpy as np
from scipy import integrate
from scipy.optimize import least_squares

# fmt: off
from ross.fluid_flow.fluid_flow_geometry import (move_rotor_center,
                                                 move_rotor_center_abs)

# fmt: on


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
    if force_type != "numerical" and (
        force_type == "short" or fluid_flow_object.bearing_type == "short_bearing"
    ):
        radial_force = (
            0.5
            * fluid_flow_object.viscosity
            * (fluid_flow_object.radius_rotor / fluid_flow_object.radial_clearance) ** 2
            * (fluid_flow_object.length ** 3 / fluid_flow_object.radius_rotor)
            * (
                (
                    2
                    * fluid_flow_object.eccentricity_ratio ** 2
                    * fluid_flow_object.omega
                )
                / (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 2
            )
        )

        tangential_force = (
            0.5
            * fluid_flow_object.viscosity
            * (fluid_flow_object.radius_rotor / fluid_flow_object.radial_clearance) ** 2
            * (fluid_flow_object.length ** 3 / fluid_flow_object.radius_rotor)
            * (
                (np.pi * fluid_flow_object.eccentricity_ratio * fluid_flow_object.omega)
                / (2 * (1 - fluid_flow_object.eccentricity_ratio ** 2) ** (3.0 / 2))
            )
        )
    elif force_type != "numerical" and (
        force_type == "long" or fluid_flow_object.bearing_type == "long_bearing"
    ):
        radial_force = (
            6
            * fluid_flow_object.viscosity
            * (fluid_flow_object.radius_rotor / fluid_flow_object.radial_clearance) ** 2
            * fluid_flow_object.radius_rotor
            * fluid_flow_object.length
            * (
                (
                    2
                    * fluid_flow_object.eccentricity_ratio ** 2
                    * fluid_flow_object.omega
                )
                / (
                    (2 + fluid_flow_object.eccentricity_ratio ** 2)
                    * (1 - fluid_flow_object.eccentricity_ratio ** 2)
                )
            )
        )
        tangential_force = (
            6
            * fluid_flow_object.viscosity
            * (fluid_flow_object.radius_rotor / fluid_flow_object.radial_clearance) ** 2
            * fluid_flow_object.radius_rotor
            * fluid_flow_object.length
            * (
                (np.pi * fluid_flow_object.eccentricity_ratio * fluid_flow_object.omega)
                / (
                    (2 + fluid_flow_object.eccentricity_ratio ** 2)
                    * (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 0.5
                )
            )
        )
    else:
        p_mat = fluid_flow_object.p_mat_numerical
        a = np.zeros([fluid_flow_object.nz, fluid_flow_object.ntheta])
        b = np.zeros([fluid_flow_object.nz, fluid_flow_object.ntheta])
        g1 = np.zeros(fluid_flow_object.nz)
        g2 = np.zeros(fluid_flow_object.nz)
        base_vector = np.array(
            [
                fluid_flow_object.xre[0][0] - fluid_flow_object.xi,
                fluid_flow_object.yre[0][0] - fluid_flow_object.yi,
            ]
        )
        for i in range(fluid_flow_object.nz):
            for j in range(int(fluid_flow_object.ntheta)):
                vector_from_rotor = np.array(
                    [
                        fluid_flow_object.xre[i][j] - fluid_flow_object.xi,
                        fluid_flow_object.yre[i][j] - fluid_flow_object.yi,
                    ]
                )
                angle_between_vectors = np.arctan2(
                    vector_from_rotor[1], vector_from_rotor[0]
                ) - np.arctan2(base_vector[1], base_vector[0])
                if angle_between_vectors < 0:
                    angle_between_vectors += 2 * np.pi
                a[i][j] = p_mat[i][j] * np.cos(angle_between_vectors)
                b[i][j] = p_mat[i][j] * np.sin(angle_between_vectors)

        for i in range(fluid_flow_object.nz):
            g1[i] = integrate.simps(a[i][:], fluid_flow_object.gama[0])
            g2[i] = integrate.simps(b[i][:], fluid_flow_object.gama[0])

        integral1 = integrate.simps(g1, fluid_flow_object.z_list)
        integral2 = integrate.simps(g2, fluid_flow_object.z_list)

        angle_corr = (
            np.pi / 2
            - np.arctan2(base_vector[1], base_vector[0])
            + fluid_flow_object.attitude_angle
        )
        radial_force_aux = fluid_flow_object.radius_rotor * integral1
        tangential_force_aux = fluid_flow_object.radius_rotor * integral2
        radial_force = radial_force_aux * np.cos(
            angle_corr + np.pi
        ) + tangential_force_aux * np.cos(angle_corr + np.pi / 2)
        tangential_force = radial_force_aux * np.cos(
            angle_corr + np.pi / 2
        ) + tangential_force_aux * np.cos(angle_corr)

    force_x = -radial_force * np.sin(
        fluid_flow_object.attitude_angle
    ) + tangential_force * np.cos(fluid_flow_object.attitude_angle)
    force_y = radial_force * np.cos(
        fluid_flow_object.attitude_angle
    ) + tangential_force * np.sin(fluid_flow_object.attitude_angle)
    return radial_force, tangential_force, force_x, force_y


def calculate_stiffness_and_damping_coefficients(fluid_flow_object):
    """This function calculates the bearing stiffness and damping matrices numerically.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    Returns
    -------
    Two lists of floats
        A list of length four including stiffness floats in this order: kxx, kxy, kyx, kyy.
        And another list of length four including damping floats in this order: cxx, cxy, cyx, cyy.
        And
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_stiffness_and_damping_coefficients(my_fluid_flow)  # doctest: +ELLIPSIS
    ([429...
    """
    N = 6
    t = np.linspace(0, 2 * np.pi / fluid_flow_object.omegap, N)
    fluid_flow_object.xp = fluid_flow_object.radial_clearance * 0.0001
    fluid_flow_object.yp = fluid_flow_object.radial_clearance * 0.0001
    dx = np.zeros(N)
    dy = np.zeros(N)
    xdot = np.zeros(N)
    ydot = np.zeros(N)
    radial_force = np.zeros(N)
    tangential_force = np.zeros(N)
    force_xx = np.zeros(N)
    force_yx = np.zeros(N)
    force_xy = np.zeros(N)
    force_yy = np.zeros(N)
    X1 = np.zeros([N, 3])
    X2 = np.zeros([N, 3])
    F1 = np.zeros(N)
    F2 = np.zeros(N)
    F3 = np.zeros(N)
    F4 = np.zeros(N)

    for i in range(N):
        fluid_flow_object.t = t[i]

        delta_x = fluid_flow_object.xp * np.sin(
            fluid_flow_object.omegap * fluid_flow_object.t
        )
        move_rotor_center(fluid_flow_object, delta_x, 0)
        dx[i] = delta_x
        xdot[i] = (
            fluid_flow_object.omegap
            * fluid_flow_object.xp
            * np.cos(fluid_flow_object.omegap * fluid_flow_object.t)
        )
        fluid_flow_object.geometry_description()
        fluid_flow_object.calculate_pressure_matrix_numerical(direction="x")
        [
            radial_force[i],
            tangential_force[i],
            force_xx[i],
            force_yx[i],
        ] = calculate_oil_film_force(fluid_flow_object, force_type="numerical")

        delta_y = fluid_flow_object.yp * np.sin(
            fluid_flow_object.omegap * fluid_flow_object.t
        )
        move_rotor_center(fluid_flow_object, -delta_x, 0)
        move_rotor_center(fluid_flow_object, 0, delta_y)
        dy[i] = delta_y
        ydot[i] = (
            fluid_flow_object.omegap
            * fluid_flow_object.yp
            * np.cos(fluid_flow_object.omegap * fluid_flow_object.t)
        )
        fluid_flow_object.geometry_description()
        fluid_flow_object.calculate_pressure_matrix_numerical(direction="y")
        [
            radial_force[i],
            tangential_force[i],
            force_xy[i],
            force_yy[i],
        ] = calculate_oil_film_force(fluid_flow_object, force_type="numerical")
        move_rotor_center(fluid_flow_object, 0, -delta_y)
        fluid_flow_object.geometry_description()
        fluid_flow_object.calculate_pressure_matrix_numerical()

        X1[i] = [1, dx[i], xdot[i]]
        X2[i] = [1, dy[i], ydot[i]]
        F1[i] = -force_xx[i]
        F2[i] = -force_xy[i]
        F3[i] = -force_yx[i]
        F4[i] = -force_yy[i]

    P1 = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X1), X1)), np.transpose(X1)), F1
    )
    P2 = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X2), X2)), np.transpose(X2)), F2
    )
    P3 = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X1), X1)), np.transpose(X1)), F3
    )
    P4 = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X2), X2)), np.transpose(X2)), F4
    )

    K = [P1[1], P2[1], P3[1], P4[1]]
    C = [P1[2], P2[2], P3[2], P4[2]]

    return K, C


def calculate_short_stiffness_matrix(fluid_flow_object):
    """This function calculates the stiffness matrix for the short bearing.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    Returns
    -------
    list of floats
        A list of length four including stiffness floats in this order: kxx, kxy, kyx, kyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_short_stiffness_matrix(my_fluid_flow)  # doctest: +ELLIPSIS
    [417...
    """
    h0 = 1.0 / (
        (
            (np.pi ** 2) * (1 - fluid_flow_object.eccentricity_ratio ** 2)
            + 16 * fluid_flow_object.eccentricity_ratio ** 2
        )
        ** 1.5
    )
    a = fluid_flow_object.load / fluid_flow_object.radial_clearance
    kxx = (
        a
        * h0
        * 4
        * (
            (np.pi ** 2) * (2 - fluid_flow_object.eccentricity_ratio ** 2)
            + 16 * fluid_flow_object.eccentricity_ratio ** 2
        )
    )
    kxy = (
        a
        * h0
        * np.pi
        * (
            (np.pi ** 2) * (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 2
            - 16 * fluid_flow_object.eccentricity_ratio ** 4
        )
        / (
            fluid_flow_object.eccentricity_ratio
            * np.sqrt(1 - fluid_flow_object.eccentricity_ratio ** 2)
        )
    )
    kyx = (
        -a
        * h0
        * np.pi
        * (
            (np.pi ** 2)
            * (1 - fluid_flow_object.eccentricity_ratio ** 2)
            * (1 + 2 * fluid_flow_object.eccentricity_ratio ** 2)
            + (32 * fluid_flow_object.eccentricity_ratio ** 2)
            * (1 + fluid_flow_object.eccentricity_ratio ** 2)
        )
        / (
            fluid_flow_object.eccentricity_ratio
            * np.sqrt(1 - fluid_flow_object.eccentricity_ratio ** 2)
        )
    )
    kyy = (
        a
        * h0
        * 4
        * (
            (np.pi ** 2) * (1 + 2 * fluid_flow_object.eccentricity_ratio ** 2)
            + (
                (32 * fluid_flow_object.eccentricity_ratio ** 2)
                * (1 + fluid_flow_object.eccentricity_ratio ** 2)
            )
            / (1 - fluid_flow_object.eccentricity_ratio ** 2)
        )
    )
    return [kxx, kxy, kyx, kyy]


def calculate_short_damping_matrix(fluid_flow_object):
    """This function calculates the damping matrix for the short bearing.
    Parameters
    -------
    fluid_flow_object: A FluidFlow object.
    Returns
    -------
    list of floats
        A list of length four including damping floats in this order: cxx, cxy, cyx, cyy
    Examples
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example
    >>> my_fluid_flow = fluid_flow_example()
    >>> calculate_short_damping_matrix(my_fluid_flow) # doctest: +ELLIPSIS
    [...
    """
    # fmt: off
    h0 = 1.0 / (((np.pi ** 2) * (1 - fluid_flow_object.eccentricity_ratio ** 2)
                 + 16 * fluid_flow_object.eccentricity_ratio ** 2) ** 1.5)
    a = fluid_flow_object.load / (fluid_flow_object.radial_clearance * fluid_flow_object.omega)
    cxx = (a * h0 * 2 * np.pi * np.sqrt(1 - fluid_flow_object.eccentricity_ratio ** 2) *
           ((np.pi ** 2) * (1 + 2 * fluid_flow_object.eccentricity_ratio ** 2)
            - 16 * fluid_flow_object.eccentricity_ratio ** 2) / fluid_flow_object.eccentricity_ratio)
    cxy = (-a * h0 * 8 * ((np.pi ** 2) * (1 + 2 * fluid_flow_object.eccentricity_ratio ** 2)
                          - 16 * fluid_flow_object.eccentricity_ratio ** 2))
    cyx = cxy
    cyy = (a * h0 * (2 * np.pi * (
            (np.pi ** 2) * (1 - fluid_flow_object.eccentricity_ratio ** 2) ** 2
            + 48 * fluid_flow_object.eccentricity_ratio ** 2)) /
           (fluid_flow_object.eccentricity_ratio * np.sqrt(1 - fluid_flow_object.eccentricity_ratio ** 2)))
    # fmt: on
    return [cxx, cxy, cyx, cyy]


def find_equilibrium_position(fluid_flow_object, print_equilibrium_position=False):
    """This function finds the equilibrium position of the rotor such that the fluid flow
    forces match the applied load.
    Parameters
    ----------
    fluid_flow_object: A FluidFlow object.
    print_equilibrium_position: bool, optional
        If True, prints the equilibrium position.
    Returns
    -------
    None
    --------
    >>> from ross.fluid_flow.fluid_flow import fluid_flow_example2
    >>> my_fluid_flow = fluid_flow_example2()
    >>> find_equilibrium_position(my_fluid_flow)
    >>> (my_fluid_flow.xi, my_fluid_flow.yi) # doctest: +ELLIPSIS
    (2.24...
    """

    def residuals(x, *args):
        """Calculates x component of the forces of the oil film and the
        difference between the y component and the load.
        Parameters
        ----------
        x: array
            Rotor center coordinates
        *args : dict
            Dictionary instantiating the ross.FluidFlow class.
        Returns
        -------
        array
            Array with the x component of the forces of the oil film and the difference
            between the y component and the load.
        """
        bearing = args[0]
        move_rotor_center_abs(
            bearing,
            x[0] * fluid_flow_object.radial_clearance,
            x[1] * fluid_flow_object.radial_clearance,
        )
        bearing.geometry_description()
        bearing.calculate_pressure_matrix_numerical()
        (_, _, fx, fy) = calculate_oil_film_force(bearing, force_type="numerical")
        return np.array([fx, (fy - bearing.load)])

    if fluid_flow_object.load is None:
        sys.exit("Load must be given to calculate the equilibrium position.")
    x0 = np.array(
        [
            0 * fluid_flow_object.radial_clearance,
            -1e-3 * fluid_flow_object.radial_clearance,
        ]
    )
    move_rotor_center_abs(fluid_flow_object, x0[0], x0[1])
    fluid_flow_object.geometry_description()
    fluid_flow_object.calculate_pressure_matrix_numerical()
    (_, _, fx, fy) = calculate_oil_film_force(fluid_flow_object, force_type="numerical")
    result = least_squares(
        residuals, x0, args=[fluid_flow_object], jac="3-point", bounds=([0, -1], [1, 0])
    )
    move_rotor_center_abs(
        fluid_flow_object,
        result.x[0] * fluid_flow_object.radial_clearance,
        result.x[1] * fluid_flow_object.radial_clearance,
    )
    fluid_flow_object.geometry_description()
    if print_equilibrium_position is True:
        print(
            "The equilibrium position (x0, y0) is: (",
            result.x[0] * fluid_flow_object.radial_clearance,
            ",",
            result.x[1] * fluid_flow_object.radial_clearance,
            ")",
        )
