# fmt: off

import sys

import numpy as np

from ross.fluid_flow.fluid_flow_geometry import (calculate_attitude_angle,
                                                 calculate_eccentricity_ratio,
                                                 calculate_rotor_load,
                                                 external_radius_function,
                                                 internal_radius_function,
                                                 modified_sommerfeld_number)

# fmt: on


class FluidFlow:
    r"""This class intends to calculate the pressure matrix and the stiffness and damping matrices
    of a fluid flow with the given parameters.

    It is supposed to be an attribute of a bearing element,
    but can work on its on to provide graphics for the user.

    Parameters
    ----------
    Grid related
    ^^^^^^^^^^^^
    Describes the discretization of the problem
    nz: int
        Number of points along the Z direction (direction of flow).
    ntheta: int
        Number of points along the direction theta. NOTE: ntheta must be odd.
    nradius: int
        Number of points along the direction r.
    length: float
        Length in the Z direction (m).

    Operation conditions
    ^^^^^^^^^^^^^^^^^^^^
    Describes the operation conditions.
    omega: float
        Rotation of the rotor (rad/s).
    p_in: float
        Input Pressure (Pa).
    p_out: float
        Output Pressure (Pa).
    load: float
        Load applied to the rotor (N).

    Geometric data of the problem
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Describes the geometric data of the problem.
    radius_rotor: float
        Rotor radius (m).
    radius_stator: float
        Stator Radius (m).
    eccentricity: float, optional
        Eccentricity (m) is the euclidean distance between rotor and stator centers.
        The center of the stator is in position (0,0).
    attitude_angle: float, optional
        Attitude angle. Angle between the load line and the eccentricity (rad).

    Fluid characteristics
    ^^^^^^^^^^^^^^^^^^^^^
    Describes the fluid characteristics.
    viscosity: float
        Viscosity (Pa.s).
    density: float
        Fluid density(Kg/m^3).

    User commands
    ^^^^^^^^^^^^^
    Commands that can be passed as arguments.
    immediately_calculate_pressure_matrix_numerically: bool, optional
        If set True, calculates the pressure matrix numerically immediately.

    Returns
    -------
    An object containing the fluid flow and its data.

    Attributes
    ----------
    ltheta: float
        Length in the theta direction (rad).
    dz: float
        Range size in the Z direction.
    dtheta: float
        Range size in the theta direction.
    ntotal: int
        Number of nodes in the grid., ntheta, nradius, n_interv_z, n_interv_theta,
    n_interv_z: int
        Number of intervals on Z.
    n_interv_theta: int
      Number of intervals on theta.
    n_interv_radius: int
        Number of intervals on r.
    p_mat_analytical : array of shape (nz, ntheta)
        The analytical pressure matrix.
    p_mat_numerical : array of shape (nz, ntheta)
        The numerical pressure matrix.
    xi: float
        Eccentricity (m) (distance between rotor and stator centers) on the x-axis.
        It is the position of the center of the rotor.
        The center of the stator is in position (0,0).
    yi: float
        Eccentricity (m) (distance between rotor and stator centers) on the y-axis.
        It is the position of the center of the rotor.
        The center of the stator is in position (0,0).
    re : array of shape (nz, ntheta)
        The external radius in each position of the grid.
    ri : array of shape (nz, ntheta)
        The internal radius in each position of the grid.
    z_list : array of shape (1, nz)
        z along the object. It goes from 0 to lb.
    xre : array of shape (nz, ntheta)
        x value of the external radius.
    xri : array of shape (nz, ntheta)
        x value of the internal radius.
    yre : array of shape (nz, ntheta)
        y value of the external radius.
    yri : array of shape (nz, ntheta)
        y value of the internal radius.
    eccentricity : float
        distance between the center of the rotor and the stator.
    difference_between_radius: float
        distance between the radius of the stator and the radius of the rotor.
    eccentricity_ratio: float
        eccentricity/difference_between_radius
    bearing_type: str
        type of structure. 'short_bearing': short; 'long_bearing': long;
        'medium_size': in between short and long.
        if length/diameter <= 1/4 it is short.
        if length/diameter > 8 it is long.
    radial_clearance: float
        Difference between both stator and rotor radius, regardless of eccentricity.
    analytical_pressure_matrix_available: bool
        True if analytically calculated pressure matrix is available.
    numerical_pressure_matrix_available: bool
        True if numerically calculated pressure matrix is available.

    Examples
    --------
    >>> from ross.fluid_flow import fluid_flow as flow
    >>> from ross.fluid_flow.fluid_flow_graphics import matplot_pressure_theta
    >>> import numpy as np
    >>> from bokeh.plotting import show
    >>> nz = 8
    >>> ntheta = 64
    >>> nradius = 8
    >>> length = 0.01
    >>> omega = 100.*2*np.pi/60
    >>> p_in = 0.
    >>> p_out = 0.
    >>> radius_rotor = 0.08
    >>> radius_stator = 0.1
    >>> viscosity = 0.015
    >>> density = 860.
    >>> eccentricity = 0.001
    >>> attitude_angle = np.pi
    >>> my_fluid_flow = flow.FluidFlow(nz, ntheta, nradius, length,
    ...                                omega, p_in, p_out, radius_rotor,
    ...                                radius_stator, viscosity, density,
    ...                                attitude_angle=attitude_angle, eccentricity=eccentricity,
    ...                                immediately_calculate_pressure_matrix_numerically=False)
    >>> my_fluid_flow.calculate_pressure_matrix_analytical() # doctest: +ELLIPSIS
    array([[...
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> # to show the plots you can use:
    >>> # show(my_fluid_flow.plot_eccentricity())
    >>> # show(my_fluid_flow.plot_pressure_theta(z=int(nz/2)))
    >>> matplot_pressure_theta(my_fluid_flow, z=int(nz/2)) # doctest: +ELLIPSIS
    <matplotlib.axes...
    """

    def __init__(
        self,
        nz,
        ntheta,
        nradius,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        viscosity,
        density,
        attitude_angle=None,
        eccentricity=None,
        load=None,
        immediately_calculate_pressure_matrix_numerically=True,
    ):
        if load is None and eccentricity is None:
            sys.exit("Either load or eccentricity must be given.")
        self.nz = nz
        self.ntheta = ntheta
        self.nradius = nradius
        self.n_interv_z = nz - 1
        self.n_interv_theta = ntheta - 1
        self.n_interv_radius = nradius - 1
        self.length = length
        self.ltheta = 2.0 * np.pi
        self.dz = length / self.n_interv_z
        self.dtheta = self.ltheta / self.n_interv_theta
        self.ntotal = self.nz * self.ntheta
        self.omega = omega
        self.p_in = p_in
        self.p_out = p_out
        self.radius_rotor = radius_rotor
        self.radius_stator = radius_stator
        self.viscosity = viscosity
        self.density = density
        self.radial_clearance = self.radius_stator - self.radius_rotor
        self.difference_between_radius = radius_stator - radius_rotor
        self.bearing_type = ""
        if self.length / (2 * self.radius_stator) <= 1 / 4:
            self.bearing_type = "short_bearing"
        elif self.length / (2 * self.radius_stator) >= 8:
            self.bearing_type = "long_bearing"
        else:
            self.bearing_type = "medium_size"
        self.eccentricity = eccentricity
        self.eccentricity_ratio = None
        self.load = load
        if self.eccentricity is None:
            modified_s = modified_sommerfeld_number(
                self.radius_stator,
                self.omega,
                self.viscosity,
                self.length,
                self.load,
                self.radial_clearance,
            )
            self.eccentricity = (
                calculate_eccentricity_ratio(modified_s)
                * self.difference_between_radius
            )
        self.eccentricity_ratio = self.eccentricity / self.difference_between_radius
        if self.load is None:
            self.load = calculate_rotor_load(
                self.radius_stator,
                self.omega,
                self.viscosity,
                self.length,
                self.radial_clearance,
                self.eccentricity_ratio,
            )
        if attitude_angle is None:
            self.attitude_angle = calculate_attitude_angle(self.eccentricity_ratio)
        else:
            self.attitude_angle = attitude_angle
        self.xi = self.eccentricity * np.cos(3 * np.pi / 2 + self.attitude_angle)
        self.yi = self.eccentricity * np.sin(3 * np.pi / 2 + self.attitude_angle)
        self.re = np.zeros([self.nz, self.ntheta])
        self.ri = np.zeros([self.nz, self.ntheta])
        self.z_list = np.zeros(self.nz)
        self.xre = np.zeros([self.nz, self.ntheta])
        self.xri = np.zeros([self.nz, self.ntheta])
        self.yre = np.zeros([self.nz, self.ntheta])
        self.yri = np.zeros([self.nz, self.ntheta])
        self.p_mat_analytical = np.zeros([self.nz, self.ntheta])
        self.c1 = np.zeros([self.nz, self.ntheta])
        self.c2 = np.zeros([self.nz, self.ntheta])
        self.c0w = np.zeros([self.nz, self.ntheta])
        self.M = np.zeros([self.ntotal, self.ntotal])
        self.f = np.zeros([self.ntotal, 1])
        self.P = np.zeros([self.ntotal, 1])
        self.p_mat_numerical = np.zeros([self.nz, self.ntheta])
        self.gama = np.zeros([self.nz, self.ntheta])
        self.calculate_coefficients()
        self.analytical_pressure_matrix_available = False
        self.numerical_pressure_matrix_available = False
        self.calculate_pressure_matrix_numerical()
        if immediately_calculate_pressure_matrix_numerically:
            self.calculate_pressure_matrix_numerical()

    def calculate_pressure_matrix_analytical(self, method=0, force_type=None):
        """This function calculates the pressure matrix analytically.
        Parameters
        ----------
        method: int
            Determines the analytical method to be used, when more than one is available.
            In case of a short bearing:
                0: based on the book Tribology Series vol. 33, by Frene et al., chapter 5.
                1: based on the chapter Linear and Nonlinear Rotordynamics, by Ishida and
                Yamamoto, from the book Flow-Induced Vibrations.
            In case of a long bearing:
                0: based on the Fundamentals of Fluid Flow Lubrification, by Hamrock, chapter 10.
        force_type: str
            If set, calculates the pressure matrix analytically considering the chosen type: 'short' or 'long'.
        Returns
        -------
        p_mat_analytical: matrix of float
            Pressure matrix of size (nz x ntheta)
        Examples
        --------
        >>> my_fluid_flow = fluid_flow_example()
        >>> my_fluid_flow.calculate_pressure_matrix_analytical() # doctest: +ELLIPSIS
        array([[...
        """
        if self.bearing_type == "short_bearing" or force_type == "short":
            if method == 0:
                for i in range(0, self.nz):
                    for j in range(0, self.ntheta):
                        # fmt: off
                        self.p_mat_analytical[i][j] = (
                                ((-3 * self.viscosity * self.omega) / self.difference_between_radius ** 2) *
                                ((i * self.dz - (self.length / 2)) ** 2 - (self.length ** 2) / 4) *
                                (self.eccentricity_ratio * np.sin(j * self.dtheta)) /
                                (1 + self.eccentricity_ratio * np.cos(j * self.dtheta)) ** 3)
                        # fmt: on
                        if self.p_mat_analytical[i][j] < 0:
                            self.p_mat_analytical[i][j] = 0
            elif method == 1:
                for i in range(0, self.nz):
                    for j in range(0, self.ntheta):
                        # fmt: off
                        self.p_mat_analytical[i][j] = (3 * self.viscosity / ((self.difference_between_radius ** 2) *
                                                                             (1. + self.eccentricity_ratio * np.cos(
                                                                                 j * self.dtheta)) ** 3)) * \
                                                      (-self.eccentricity_ratio * self.omega * np.sin(
                                                          j * self.dtheta)) * \
                                                      (((i * self.dz - (self.length / 2)) ** 2) - (
                                                              self.length ** 2) / 4)
                        # fmt: on
                        if self.p_mat_analytical[i][j] < 0:
                            self.p_mat_analytical[i][j] = 0
        elif self.bearing_type == "long_bearing" or force_type == "long":
            if method == 0:
                for i in range(0, self.nz):
                    for j in range(0, self.ntheta):
                        self.p_mat_analytical[i][j] = (
                            (
                                6
                                * self.viscosity
                                * self.omega
                                * (self.ri[i][j] / self.difference_between_radius) ** 2
                                * self.eccentricity_ratio
                                * np.sin(self.dtheta * j)
                                * (
                                    2
                                    + self.eccentricity_ratio * np.cos(self.dtheta * j)
                                )
                            )
                            / (
                                (2 + self.eccentricity_ratio ** 2)
                                * (
                                    1
                                    + self.eccentricity_ratio * np.cos(self.dtheta * j)
                                )
                                ** 2
                            )
                            + self.p_in
                        )
                        if self.p_mat_analytical[i][j] < 0:
                            self.p_mat_analytical[i][j] = 0
        elif self.bearing_type == "medium_size":
            raise ValueError(
                "The pressure matrix for a bearing that is neither short or long can only be calculated "
                "numerically. Try calling calculate_pressure_matrix_numerical or setting force_type "
                "to either 'short' or 'long' in calculate_pressure_matrix_analytical."
            )
        self.analytical_pressure_matrix_available = True
        return self.p_mat_analytical

    def calculate_coefficients(self):
        """This function calculates the constants that form the Poisson equation
        of the discrete pressure (central differences in the second
        derivatives). It is executed when the class is instantiated.
        Examples
        --------
        >>> my_fluid_flow = fluid_flow_example()
        >>> my_fluid_flow.calculate_coefficients()
        >>> my_fluid_flow.c0w # doctest: +ELLIPSIS
        array([[...
        """
        for i in range(0, self.nz):
            zno = i * self.dz
            self.z_list[i] = zno
            eccentricity_error = False
            for j in range(0, self.ntheta):
                # fmt: off
                self.gama[i][j] = j * self.dtheta + np.pi / 2 + self.attitude_angle
                [radius_external, self.xre[i][j], self.yre[i][j]] = \
                    external_radius_function(self.gama[i][j], self.radius_stator)
                [radius_internal, self.xri[i][j], self.yri[i][j]] = \
                    internal_radius_function(self.gama[i][j], self.attitude_angle, self.radius_rotor,
                                             self.eccentricity)
                self.re[i][j] = radius_external
                self.ri[i][j] = radius_internal

                w = self.omega * self.radius_rotor

                k = (self.re[i][j] ** 2 * (np.log(self.re[i][j]) - 1 / 2) - self.ri[i][j] ** 2 *
                     (np.log(self.ri[i][j]) - 1 / 2)) / (self.ri[i][j] ** 2 - self.re[i][j] ** 2)

                self.c1[i][j] = (1 / (4 * self.viscosity)) * ((self.re[i][j] ** 2 * np.log(self.re[i][j]) -
                                                               self.ri[i][j] ** 2 * np.log(self.ri[i][j]) +
                                                               (self.re[i][j] ** 2 - self.ri[i][j] ** 2) *
                                                               (k - 1)) - 2 * self.re[i][j] ** 2 * (
                                                                      (np.log(self.re[i][j]) + k - 1 / 2) * np.log(
                                                                       self.re[i][j] / self.ri[i][j])))

                self.c2[i][j] = (- self.ri[i][j] ** 2) / (8 * self.viscosity) * \
                                ((self.re[i][j] ** 2 - self.ri[i][j] ** 2 -
                                  (self.re[i][j] ** 4 - self.ri[i][j] ** 4) /
                                  (2 * self.ri[i][j] ** 2)) +
                                 ((self.re[i][j] ** 2 - self.ri[i][j] ** 2) /
                                  (self.ri[i][j] ** 2 *
                                   np.log(self.re[i][j] / self.ri[i][j]))) *
                                 (self.re[i][j] ** 2 * np.log(self.re[i][j] / self.ri[i][j]) -
                                  (self.re[i][j] ** 2 - self.ri[i][j] ** 2) / 2))

                self.c0w[i][j] = (- w * self.ri[i][j] *
                                  (np.log(self.re[i][j] / self.ri[i][j]) *
                                   (1 + (self.ri[i][j] ** 2) / (self.re[i][j] ** 2 - self.ri[i][j] ** 2)) - 1 / 2))
                # fmt: on
                if not eccentricity_error:
                    if abs(self.xri[i][j]) > abs(self.xre[i][j]) or abs(
                        self.yri[i][j]
                    ) > abs(self.yre[i][j]):
                        eccentricity_error = True
            if eccentricity_error:
                raise ValueError(
                    "Error: The given parameters create a rotor that is not inside the stator. "
                    "Check parameters and fix accordingly."
                )

    def mounting_matrix(self):
        """This function assembles the matrix M and the independent vector f.
        Examples
        --------
        >>> my_fluid_flow = fluid_flow_example()
        >>> my_fluid_flow.mounting_matrix()
        >>> my_fluid_flow.M # doctest: +ELLIPSIS
        array([[...
        """
        # fmt: off
        count = 0
        for x in range(self.ntheta):
            self.M[count][count] = 1
            self.f[count][0] = self.p_in
            count = count + self.nz - 1
            self.M[count][count] = 1
            self.f[count][0] = self.p_out
            count = count + 1
        count = 0
        for x in range(self.nz - 2):
            self.M[self.ntotal - self.nz + 1 + count][1 + count] = 1
            self.M[self.ntotal - self.nz + 1 + count][self.ntotal - self.nz + 1 + count] = -1
            count = count + 1
        count = 1
        j = 0
        for i in range(1, self.nz - 1):
            a = (1 / self.dtheta ** 2) * (self.c1[i][self.ntheta - 1])
            self.M[count][self.ntotal - 2 * self.nz + count] = a
            b = (1 / self.dz ** 2) * (self.c2[i - 1, j])
            self.M[count][count - 1] = b
            c = -((1 / self.dtheta ** 2) * ((self.c1[i][j]) + self.c1[i][self.ntheta - 1])
                  + (1 / self.dz ** 2) * (self.c2[i][j] + self.c2[i - 1][j]))
            self.M[count, count] = c
            d = (1 / self.dz ** 2) * (self.c2[i][j])
            self.M[count][count + 1] = d
            e = (1 / self.dtheta ** 2) * (self.c1[i][j])
            self.M[count][count + self.nz] = e
            count = count + 1
        count = self.nz + 1
        for j in range(1, self.ntheta - 1):
            for i in range(1, self.nz - 1):
                a = (1 / self.dtheta ** 2) * (self.c1[i, j - 1])
                self.M[count][count - self.nz] = a
                b = (1 / self.dz ** 2) * (self.c2[i - 1][j])
                self.M[count][count - 1] = b
                c = -((1 / self.dtheta ** 2) * ((self.c1[i][j]) + self.c1[i][j - 1])
                      + (1 / self.dz ** 2) * (self.c2[i][j] + self.c2[i - 1][j]))
                self.M[count, count] = c
                d = (1 / self.dz ** 2) * (self.c2[i][j])
                self.M[count][count + 1] = d
                e = (1 / self.dtheta ** 2) * (self.c1[i][j])
                self.M[count][count + self.nz] = e
                count = count + 1
            count = count + 2
        count = 1
        for j in range(self.ntheta - 1):
            for i in range(1, self.nz - 1):
                if j == 0:
                    self.f[count][0] = (self.c0w[i][j] - self.c0w[i][self.ntheta - 1]) / self.dtheta
                else:
                    self.f[count][0] = (self.c0w[i, j] - self.c0w[i, j - 1]) / self.dtheta
                count = count + 1
            count = count + 2
        # fmt: on

    def resolves_matrix(self):
        """This function resolves the linear system [M]{P}={f}.
        Examples
        --------
        >>> my_fluid_flow = fluid_flow_example()
        >>> my_fluid_flow.mounting_matrix()
        >>> my_fluid_flow.resolves_matrix()
        >>> my_fluid_flow.P # doctest: +ELLIPSIS
        array([[...
        """
        self.P = np.linalg.solve(self.M, self.f)

    def calculate_pressure_matrix_numerical(self):
        """This function calculates the pressure matrix numerically.
        Returns
        -------
        p_mat_numerical: matrix of float
            Pressure matrix of size (nz x ntheta)
        Examples
        --------
        >>> my_fluid_flow = fluid_flow_example()
        >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
        array([[...
        """
        self.mounting_matrix()
        self.resolves_matrix()
        for i in range(self.nz):
            for j in range(self.ntheta):
                k = j * self.nz + i
                if self.P[k] < 0:
                    self.p_mat_numerical[i][j] = 0
                else:
                    self.p_mat_numerical[i][j] = self.P[k]
        self.numerical_pressure_matrix_available = True
        return self.p_mat_numerical


def fluid_flow_example():
    """This function returns an instance of a simple fluid flow.
    The purpose is to make available a simple model
    so that doctest can be written using it.

    Parameters
    ----------

    Returns
    -------
    An instance of a fluid flow object.

    Examples
    --------
    >>> my_fluid_flow = fluid_flow_example()
    >>> my_fluid_flow.eccentricity
    0.0001
    """
    my_pressure_matrix = FluidFlow(
        nz=8,
        ntheta=32,
        nradius=8,
        length=0.04,
        omega=100.0 * 2 * np.pi / 60,
        p_in=0.0,
        p_out=0.0,
        radius_rotor=0.2,
        radius_stator=0.2002,
        viscosity=0.015,
        density=860.0,
        eccentricity=0.0001,
        attitude_angle=np.pi / 4,
        immediately_calculate_pressure_matrix_numerically=False,
    )
    return my_pressure_matrix


def fluid_flow_example2():
    """This function returns a different instance of a simple fluid flow.
    The purpose is to make available a simple model
    so that doctest can be written using it.

    Parameters
    ----------

    Returns
    -------
    An instance of a fluid flow object.

    Examples
    --------
    >>> my_fluid_flow = fluid_flow_example2()
    >>> my_fluid_flow.load
    525
    """
    nz = 8
    ntheta = 16
    nradius = 8
    length = 0.03
    omega = 157.1
    p_in = 0.0
    p_out = 0.0
    radius_rotor = 0.0499
    radius_stator = 0.05
    load = 525
    visc = 0.1
    rho = 860.0
    return FluidFlow(
        nz,
        ntheta,
        nradius,
        length,
        omega,
        p_in,
        p_out,
        radius_rotor,
        radius_stator,
        visc,
        rho,
        load=load,
        immediately_calculate_pressure_matrix_numerically=False,
    )
