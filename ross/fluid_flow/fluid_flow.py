import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show
from scipy import integrate


class PressureMatrix:
    r"""This class calculates the pressure matrix of the object with the given parameters.

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
    beta: float, optional
        Attitude angle. Angle between the origin and the eccentricity (rad).

    Fluid characteristics
    ^^^^^^^^^^^^^^^^^^^^^
    Describes the fluid characteristics.
    viscosity: float
        Viscosity (Pa.s).
    density: float
        Fluid density(Kg/m^3).

    Returns
    -------
    An object containing the pressure matrix and its data.

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
    z : array of shape (1, nz)
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
        if length/radius_stator <= 1/4 it is short.
        if length/radius_stator > 8 it is long.
    radial_clearance: float
        Difference between both stator and rotor radius, regardless of eccentricity.
    analytical_pressure_matrix_available: bool
        True if analytically calculated pressure matrix is available.
    numerical_pressure_matrix_available: bool
        True if numerically calculated pressure matrix is available.

    Examples
    --------
    >>> from ross.fluid_flow import fluid_flow as flow
    >>> import numpy as np
    >>> from bokeh.plotting import show
    >>> nz = 8
    >>> ntheta = 64
    >>> nradius = 11
    >>> length = 0.01
    >>> omega = 100.*2*np.pi/60
    >>> p_in = 0.
    >>> p_out = 0.
    >>> radius_rotor = 0.08
    >>> radius_stator = 0.1
    >>> viscosity = 0.015
    >>> density = 860.
    >>> eccentricity = 0.001
    >>> beta = np.pi
    >>> my_fluid_flow = flow.PressureMatrix(nz, ntheta, nradius, length,
    ...                                          omega, p_in, p_out, radius_rotor,
    ...                                          radius_stator, viscosity, density,
    ...                                          beta=beta, eccentricity=eccentricity)
    >>> my_fluid_flow.calculate_pressure_matrix_analytical() # doctest: +ELLIPSIS
    array([[-0.00000...
    >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
    array([[...
    >>> # to show the plots you can use:
    >>> # show(my_fluid_flow.plot_eccentricity())
    >>> # show(my_fluid_flow.plot_pressure_theta(z=int(nz/2)))
    >>> my_fluid_flow.matplot_pressure_theta(z=int(nz/2)) # doctest: +ELLIPSIS
    <matplotlib.axes._subplots.AxesSubplot object at...
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
            beta=None,
            eccentricity=None,
            load=None,
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
        if self.length / self.radius_stator <= 1 / 4:
            self.bearing_type = "short_bearing"
        elif self.length / self.radius_stator >= 8:
            self.bearing_type = "long_bearing"
        else:
            self.bearing_type = "medium_size"
        self.eccentricity = eccentricity
        self.eccentricity_ratio = None
        self.load = load
        if self.eccentricity is None:
            self.eccentricity = (
                    self.calculate_eccentricity_ratio() * self.difference_between_radius
            )
        self.eccentricity_ratio = self.eccentricity / self.difference_between_radius
        if self.load is None:
            self.load = self.get_rotor_load()
        if beta is None:
            self.beta = self.calculate_attitude_angle()
        else:
            self.beta = beta
        self.xi = self.eccentricity * np.cos(2 * np.pi - self.beta)
        self.yi = self.eccentricity * np.sin(2 * np.pi - self.beta)
        self.re = np.zeros([self.nz, self.ntheta])
        self.ri = np.zeros([self.nz, self.ntheta])
        self.z = np.zeros([1, self.nz])
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

    def calculate_attitude_angle(self):
        """Calculates the attitude angle based on the eccentricity.
        Returns
        -------
        float
            Attitude angle
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_attitude_angle() # doctest: +ELLIPSIS
        1.5...
        """
        return np.arctan(np.pi * (1 - self.eccentricity_ratio ** 2) / (4 * self.eccentricity_ratio))

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
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_pressure_matrix_analytical() # doctest: +ELLIPSIS
        array([[...
        """
        if self.bearing_type == "short_bearing" or force_type == 'short':
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
        elif self.bearing_type == "long_bearing" or force_type == 'long':
            if method == 0:
                for i in range(0, self.nz):
                    for j in range(0, self.ntheta):
                        self.p_mat_analytical[i][j] = (6 * self.viscosity * self.omega *
                                                       (self.ri[i][j] / self.difference_between_radius) ** 2 *
                                                       self.eccentricity_ratio * np.sin(self.gama[i][j]) *
                                                       (2 + self.eccentricity_ratio * np.cos(self.gama[i][j]))) / (
                                                              (2 + self.eccentricity_ratio ** 2) *
                                                              (1 + self.eccentricity_ratio * np.cos(
                                                                  self.gama[i][j])) ** 2) + \
                                                      self.p_in
                        if self.p_mat_analytical[i][j] < 0:
                            self.p_mat_analytical[i][j] = 0
        elif self.bearing_type == "medium_size":
            raise ValueError("The pressure matrix for a bearing that is neither short or long can only be calculated "
                             "numerically. Try calling calculate_pressure_matrix_numerical or setting force_type "
                             "to either 'short' or 'long' in calculate_pressure_matrix_analytical.")
        self.analytical_pressure_matrix_available = True
        return self.p_mat_analytical

    def calculate_coefficients(self):
        """This function calculates the constants that form the Poisson equation
        of the discrete pressure (central differences in the second
        derivatives). It is executed when the class is instantiated.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_coefficients()
        >>> my_fluid_flow.c0w # doctest: +ELLIPSIS
        array([[...
        """
        for i in range(0, self.nz):
            zno = i * self.dz
            self.z[0][i] = zno
            plot_eccentricity_error = False
            position = -1
            for j in range(0, self.ntheta):
                # fmt: off
                self.gama[i][j] = j * self.dtheta + (np.pi - self.beta)
                [radius_external, self.xre[i][j], self.yre[i][j]] = \
                    self.external_radius_function(self.gama[i][j])
                [radius_internal, self.xri[i][j], self.yri[i][j]] = \
                    self.internal_radius_function(zno, self.gama[i][j])
                self.re[i][j] = radius_external
                self.ri[i][j] = radius_internal

                w = self.omega * self.ri[i][j]

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
                if not plot_eccentricity_error:
                    if abs(self.xri[i][j]) > abs(self.xre[i][j]) or abs(
                            self.yri[i][j]
                    ) > abs(self.yre[i][j]):
                        plot_eccentricity_error = True
                        position = i
            if plot_eccentricity_error:
                self.plot_eccentricity(position)
                sys.exit(
                    "Error: The given parameters create a rotor that is not inside the stator. "
                    "Check the plotted figure and fix accordingly."
                )

    def mounting_matrix(self):
        """This function assembles the matrix M and the independent vector f.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
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
        >>> my_fluid_flow = pressure_matrix_example()
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
        >>> my_fluid_flow = pressure_matrix_example()
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

    def internal_radius_function(self, z, gama):
        """This function calculates the radius of the rotor given the
        radius and the position z.
        Parameters
        ----------
        z: float
            Distance along the z-axis.
        gama: float
            Gama is the distance in the theta-axis. It should range from 0 to 2*np.pi.
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
        >>> my_fluid_flow = pressure_matrix_example()
        >>> radius_internal, xri, yri = my_fluid_flow.internal_radius_function(z=0, gama=0)
        >>> radius_internal
        0.079
        """
        if (np.pi - self.beta) < gama < (2 * np.pi - self.beta):
            alpha = np.absolute(2 * np.pi - gama - self.beta)
            radius_internal = np.sqrt(self.radius_rotor ** 2
                                      - (self.eccentricity * np.sin(alpha)) ** 2) + self.eccentricity * np.cos(alpha)
            xri = radius_internal * np.cos(gama)
            yri = radius_internal * np.sin(gama)

        else:
            alpha = self.beta + gama
            radius_internal = np.sqrt(self.radius_rotor ** 2
                                      - (self.eccentricity * np.sin(alpha)) ** 2) + self.eccentricity * np.cos(alpha)
            xri = radius_internal * np.cos(gama)
            yri = radius_internal * np.sin(gama)

        return radius_internal, xri, yri

    def external_radius_function(self, gama):
        """This function calculates the radius of the stator.
        Parameters
        ----------
        gama: float
            Gama is the distance in the theta-axis. It should range from 0 to 2*np.pi.
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
        >>> my_fluid_flow = pressure_matrix_example()
        >>> radius_external, xre, yre = my_fluid_flow.external_radius_function(gama=0)
        >>> radius_external
        0.1
        """
        radius_external = self.radius_stator
        xre = radius_external * np.cos(gama)
        yre = radius_external * np.sin(gama)

        return radius_external, xre, yre

    def get_rotor_load(self):
        """Returns the load applied to the rotor, based on the eccentricity ratio.
        Suitable only for short bearings.
        Returns
        -------
        float
            Load applied to the rotor.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.get_rotor_load() # doctest: +ELLIPSIS
        1.5...
        """
        if not self.bearing_type == "short_bearing":
            warnings.warn(
                "Function get_rotor_load suitable only for short bearings. "
                "The ratio between the bearing length and its radius should be less or "
                "equal to 0.25. Currently we have "
                + str(self.length / self.radius_stator)
                + "."
            )
        return (
                       (
                               np.pi
                               * self.radius_stator
                               * 2
                               * self.omega
                               * self.viscosity
                               * (self.length ** 3)
                               * self.eccentricity_ratio
                       )
                       / (
                               8
                               * (self.radial_clearance ** 2)
                               * ((1 - self.eccentricity_ratio ** 2) ** 2)
                       )
               ) * (np.sqrt((16 / (np.pi ** 2) - 1) * self.eccentricity_ratio ** 2 + 1))

    def modified_sommerfeld_number(self):
        """Returns the modified sommerfeld number.
        Returns
        -------
        float
            The modified sommerfeld number.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.modified_sommerfeld_number() # doctest: +ELLIPSIS
        6...
        """
        return (
                       self.radius_stator * 2 * self.omega * self.viscosity * (self.length ** 3)
               ) / (8 * self.load * (self.radial_clearance ** 2))

    def sommerfeld_number(self):
        """Returns the sommerfeld number.
        Returns
        -------
        float
            The sommerfeld number.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.sommerfeld_number() # doctest: +ELLIPSIS
        805...
        """
        modified_s = self.modified_sommerfeld_number()
        return (modified_s / np.pi) * (self.radius_stator * 2 / self.length) ** 2

    def calculate_eccentricity_ratio(self):
        """Calculate the eccentricity ratio for a given load using the sommerfeld number.
        Suitable only for short bearings.
        Returns
        -------
        float
            The eccentricity ratio.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_eccentricity_ratio() # doctest: +ELLIPSIS
        0.0...
        """
        if not self.bearing_type == "short_bearing":
            warnings.warn(
                "Function calculate_eccentricity_ratio suitable only for short bearings. "
                "The ratio between the bearing length and its radius should be less or "
                "equal to 0.25. Currently we have "
                + str(self.length / self.radius_stator)
                + "."
            )
        s = self.modified_sommerfeld_number()
        coefficients = [
            1,
            -4,
            (6 - (s ** 2) * (16 - np.pi ** 2)),
            -(4 + (np.pi ** 2) * (s ** 2)),
            1,
        ]
        roots = np.roots(coefficients)
        for i in range(0, len(roots)):
            if 0 <= roots[i] <= 1:
                return np.sqrt(roots[i].real)
        sys.exit("Eccentricity ratio could not be calculated.")

    def get_analytical_stiffness_matrix(self):
        """Returns the stiffness matrix calculated analytically.
        Suitable only for short bearings.
        Returns
        -------
        list of floats
            A list of length four including stiffness floats in this order: kxx, kxy, kyx, kyy
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.get_analytical_stiffness_matrix() # doctest: +ELLIPSIS
        [...
        """
        if not self.bearing_type == "short_bearing":
            warnings.warn(
                "Function get_analytical_stiffness_matrix suitable only for short bearings. "
                "The ratio between the bearing length and its radius should be less or "
                "equal to 0.25. Currently we have "
                + str(self.length / self.radius_stator)
                + "."
            )
        # fmt: off
        f = self.get_rotor_load()
        h0 = 1.0 / (((np.pi ** 2) * (1 - self.eccentricity_ratio ** 2) + 16 * self.eccentricity_ratio ** 2) ** 1.5)
        a = f / self.radial_clearance
        kxx = a * h0 * 4 * ((np.pi ** 2) * (2 - self.eccentricity_ratio ** 2) + 16 * self.eccentricity_ratio ** 2)
        kxy = (a * h0 * np.pi * ((np.pi ** 2) * (1 - self.eccentricity_ratio ** 2) ** 2 -
                                 16 * self.eccentricity_ratio ** 4) /
               (self.eccentricity_ratio * np.sqrt(1 - self.eccentricity_ratio ** 2)))
        kyx = (-a * h0 * np.pi * ((np.pi ** 2) * (1 - self.eccentricity_ratio ** 2) *
                                  (1 + 2 * self.eccentricity_ratio ** 2) +
                                  (32 * self.eccentricity_ratio ** 2) * (1 + self.eccentricity_ratio ** 2)) /
               (self.eccentricity_ratio * np.sqrt(1 - self.eccentricity_ratio ** 2)))
        kyy = (a * h0 * 4 * ((np.pi ** 2) * (1 + 2 * self.eccentricity_ratio ** 2) +
                             ((32 * self.eccentricity_ratio ** 2) *
                              (1 + self.eccentricity_ratio ** 2)) / (1 - self.eccentricity_ratio ** 2)))
        # fmt: on
        return [kxx, kxy, kyx, kyy]

    def get_analytical_damping_matrix(self):
        """Returns the damping matrix calculated analytically.
        Suitable only for short bearings.
        Returns
        -------
        list of floats
            A list of length four including stiffness floats in this order: cxx, cxy, cyx, cyy
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.get_analytical_damping_matrix() # doctest: +ELLIPSIS
        [...
        """
        if not self.bearing_type == "short_bearing":
            warnings.warn(
                "Function get_analytical_damping_matrix suitable only for short bearings. "
                "The ratio between the bearing length and its radius should be less or "
                "equal to 0.25. Currently we have "
                + str(self.length / self.radius_stator)
                + "."
            )
        # fmt: off
        f = self.get_rotor_load()
        h0 = 1.0 / (((np.pi ** 2) * (1 - self.eccentricity_ratio ** 2) + 16 * self.eccentricity_ratio ** 2) ** 1.5)
        a = f / (self.radial_clearance * self.omega)
        cxx = (a * h0 * 2 * np.pi * np.sqrt(1 - self.eccentricity_ratio ** 2) *
               ((np.pi ** 2) * (
                           1 + 2 * self.eccentricity_ratio ** 2) - 16 * self.eccentricity_ratio ** 2) / self.eccentricity_ratio)
        cxy = (-a * h0 * 8 * (
                    (np.pi ** 2) * (1 + 2 * self.eccentricity_ratio ** 2) - 16 * self.eccentricity_ratio ** 2))
        cyx = cxy
        cyy = (a * h0 * (2 * np.pi * (
                    (np.pi ** 2) * (1 - self.eccentricity_ratio ** 2) ** 2 + 48 * self.eccentricity_ratio ** 2)) /
               (self.eccentricity_ratio * np.sqrt(1 - self.eccentricity_ratio ** 2)))
        # fmt: on
        return [cxx, cxy, cyx, cyy]

    def calculate_oil_film_force(self, force_type=None):
        """This function calculates the forces of the oil film in the N and T directions, ie in the
        opposite direction to the eccentricity and in the tangential direction.
        Parameters
        ----------
        force_type: str
            If set, calculates the pressure matrix analytically considering the chosen type: 'short' or 'long'.
        Returns
        -------
        normal_force: float
            Force of the oil film in the opposite direction to the eccentricity direction.
        tangential_force: float
            Force of the oil film in the tangential direction
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_oil_film_force() # doctest: +ELLIPSIS
        (...
        """
        if force_type != 'numerical' and (force_type == 'short' or self.bearing_type == 'short_bearing'):
            normal_force = 0.5 * self.viscosity * (self.radius_rotor / self.difference_between_radius) ** 2 * \
                           (self.length ** 3 / self.radius_rotor) * ((2 * self.eccentricity_ratio ** 2 * self.omega) /
                                                                     (1 - self.eccentricity_ratio ** 2) ** 2)
            tangential_force = 0.5 * self.viscosity * (self.radius_rotor / self.difference_between_radius) ** 2 * \
                               (self.length ** 3 / self.radius_rotor) * (
                                           (np.pi * self.eccentricity_ratio * self.omega) /
                                           (2 * (1 - self.eccentricity_ratio ** 2) ** (3. / 2)))
        elif force_type != 'numerical' and (force_type == 'long' or self.bearing_type == 'long_bearing'):
            normal_force = 6 * self.viscosity * (self.radius_rotor / self.difference_between_radius) ** 2 * \
                           self.radius_rotor * self.length * ((2 * self.eccentricity_ratio ** 2 * self.omega) /
                                                              ((2 + self.eccentricity_ratio ** 2) *
                                                               (1 - self.eccentricity_ratio ** 2)))
            tangential_force = 6 * self.viscosity * (self.radius_rotor / self.difference_between_radius) ** 2 * \
                               self.radius_rotor * self.length * ((np.pi * self.eccentricity_ratio * self.omega) /
                                                                  ((2 + self.eccentricity_ratio ** 2) *
                                                                   (1 - self.eccentricity_ratio ** 2) ** 0.5))
        else:
            p_mat = self.p_mat_numerical
            a = np.zeros([self.nz, self.ntheta])
            b = np.zeros([self.nz, self.ntheta])
            g1 = np.zeros(self.nz)
            g2 = np.zeros(self.nz)
            theta_list = np.zeros(self.ntheta)
            z_list = np.zeros(self.nz)
            for i in range(self.nz):
                z_list[i] = i * self.dz
                for j in range(self.ntheta):
                    theta_list[j] = j * self.dtheta + (np.pi - self.beta)
                    gama = j * self.dtheta + (np.pi - self.beta)

                    a[i][j] = p_mat[i][j] * np.cos(gama)
                    b[i][j] = p_mat[i][j] * np.sin(gama)

            for i in range(self.nz):
                g1[i] = integrate.simps(a[i][:], theta_list)
                g2[i] = integrate.simps(b[i][:], theta_list)

            integral1 = integrate.simps(g1, z_list)
            integral2 = integrate.simps(g2, z_list)

            normal_force = - self.radius_rotor * integral1
            tangential_force = self.radius_rotor * integral2
        return normal_force, tangential_force

    def plot_eccentricity(self, z=0):
        """This function assembles pressure graphic along the z-axis.
        The first few plots are of a different color to indicate where theta begins.
        Parameters
        ----------
        z: int, optional
            The distance in z where to cut and plot.
        Returns
        -------
        Figure
            An object containing the plot.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> fig = my_fluid_flow.plot_eccentricity(z=int(my_fluid_flow.nz/2))
        >>> # to show the plots you can use:
        >>> # show(fig)
        """
        p = figure(
            title="Cut in plane Z=" + str(z),
            x_axis_label="X axis",
            y_axis_label="Y axis",
        )
        for j in range(0, self.ntheta):
            p.circle(self.xre[z][j], self.yre[z][j], color="red")
            p.circle(self.xri[z][j], self.yri[z][j], color="blue")
            p.circle(0, 0, color="blue")
            p.circle(self.xi, self.yi, color="red")
        p.circle(0, 0, color="black")
        return p

    def plot_pressure_z(self, theta=0):
        """This function assembles pressure graphic along the z-axis for one or both the
        numerically (blue) and analytically (red) calculated pressure matrices, depending on if
        one or both were calculated.
        Parameters
        ----------
        theta: int, optional
            The theta to be considered.
        Returns
        -------
        Figure
            An object containing the plot.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
        array([[...
        >>> fig = my_fluid_flow.plot_pressure_z(theta=int(my_fluid_flow.ntheta/2))
        >>> # to show the plots you can use:
        >>> # show(fig)
        """
        if (
                not self.numerical_pressure_matrix_available
                and not self.analytical_pressure_matrix_available
        ):
            sys.exit(
                "Must calculate the pressure matrix. "
                "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
            )
        x = []
        y_n = []
        y_a = []
        for i in range(0, self.nz):
            x.append(i * self.dz)
            y_n.append(self.p_mat_numerical[i][theta])
            y_a.append(self.p_mat_analytical[i][theta])
        p = figure(
            title="Pressure along the Z direction (direction of flow); Theta="
                  + str(theta),
            x_axis_label="Points along Z",
        )
        if self.numerical_pressure_matrix_available:
            p.line(x, y_n, legend="Numerical pressure", color="blue", line_width=2)
        if self.analytical_pressure_matrix_available:
            p.line(x, y_a, legend="Analytical pressure", color="red", line_width=2)
        return p

    def plot_shape(self, theta=0):
        """This function assembles a graphic representing the geometry of the rotor.
        Parameters
        ----------
        theta: int, optional
            The theta to be considered.
        Returns
        -------
        Figure
            An object containing the plot.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> fig = my_fluid_flow.plot_shape(theta=int(my_fluid_flow.ntheta/2))
        >>> # to show the plots you can use:
        >>> # show(fig)
        """
        x = np.zeros(self.nz)
        y_re = np.zeros(self.nz)
        y_ri = np.zeros(self.nz)
        for i in range(0, self.nz):
            x[i] = i * self.dz
            y_re[i] = self.re[i][theta]
            y_ri[i] = self.ri[i][theta]
        p = figure(
            title="Shapes of stator and rotor along Z; Theta=" + str(theta),
            x_axis_label="Points along Z",
            y_axis_label="Radial direction",
        )
        p.line(x, y_re, line_width=2, color="red")
        p.line(x, y_ri, line_width=2, color="blue")
        return p

    def plot_pressure_theta(self, z=0):
        """This function assembles pressure graphic in the theta direction for a given z
        for one or both the numerically (blue) and analytically (red) calculated pressure matrices,
        depending on if one or both were calculated.
        Parameters
        ----------
        z: int, optional
            The distance along z-axis to be considered.
        Returns
        -------
        Figure
            An object containing the plot.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
        array([[...
        >>> fig = my_fluid_flow.plot_pressure_theta(z=int(my_fluid_flow.nz/2))
        >>> # to show the plots you can use:
        >>> # show(fig)
        """
        if (
                not self.numerical_pressure_matrix_available
                and not self.analytical_pressure_matrix_available
        ):
            sys.exit(
                "Must calculate the pressure matrix. "
                "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
            )
        theta_list = []
        for theta in range(0, self.ntheta):
            theta_list.append(theta * self.dtheta)
        p = figure(
            title="Pressure along Theta; Z=" + str(z),
            x_axis_label="Points along Theta",
            y_axis_label="Pressure",
        )
        if self.numerical_pressure_matrix_available:
            p.line(
                theta_list,
                self.p_mat_numerical[z],
                legend="Numerical pressure",
                line_width=2,
                color="blue",
            )
        elif self.analytical_pressure_matrix_available:
            p.line(
                theta_list,
                self.p_mat_analytical[z],
                legend="Analytical pressure",
                line_width=2,
                color="red",
            )
        return p

    def matplot_eccentricity(self, z=0, ax=None):
        """This function assembles pressure graphic along the z-axis using matplotlib.
        The first few plots are of a different color to indicate where theta begins.
        Parameters
        ----------
        z: int, optional
            The distance in z where to cut and plot.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> ax = my_fluid_flow.matplot_eccentricity(z=int(my_fluid_flow.nz/2))
        >>> # to show the plots you can use:
        >>> # plt.show()
        """
        if ax is None:
            ax = plt.gca()
        for j in range(0, self.ntheta):
            ax.plot(self.xre[z][j], self.yre[z][j], "r.")
            ax.plot(self.xri[z][j], self.yri[z][j], "b.")
            ax.plot(0, 0, "r*")
            ax.plot(self.xi, self.yi, "b*")
        ax.set_title("Cut in plane Z=" + str(z))
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        plt.axis("equal")
        return ax

    def matplot_pressure_z(self, theta=0, ax=None):
        """This function assembles pressure graphic along the z-axis using matplotlib
        for one or both the numerically (blue) and analytically (red) calculated pressure matrices,
        depending on if one or both were calculated.
        Parameters
        ----------
        theta: int, optional
            The distance in theta where to cut and plot.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
        array([[...
        >>> ax = my_fluid_flow.matplot_pressure_z(theta=int(my_fluid_flow.ntheta/2))
        >>> # to show the plots you can use:
        >>> # plt.show()
        """
        if (
                not self.numerical_pressure_matrix_available
                and not self.analytical_pressure_matrix_available
        ):
            sys.exit(
                "Must calculate the pressure matrix. "
                "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
            )
        if ax is None:
            ax = plt.gca()
        x = np.zeros(self.nz)
        y_n = np.zeros(self.nz)
        y_a = np.zeros(self.nz)
        for i in range(0, self.nz):
            x[i] = i * self.dz
            y_n[i] = self.p_mat_numerical[i][theta]
            y_a[i] = self.p_mat_analytical[i][theta]
        if self.numerical_pressure_matrix_available:
            ax.plot(x, y_n, "b", label="Numerical pressure")
        if self.analytical_pressure_matrix_available:
            ax.plot(x, y_a, "r", label="Analytical pressure")
        ax.set_title(
            "Pressure along the Z direction (direction of flow); Theta=" + str(theta)
        )
        ax.set_xlabel("Points along Z")
        ax.set_ylabel("Pressure")
        return ax

    def matplot_shape(self, theta=0, ax=None):
        """This function assembles a graphic representing the geometry of the rotor using matplotlib.
        Parameters
        ----------
        theta: int, optional
            The theta to be considered.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> ax = my_fluid_flow.matplot_shape(theta=int(my_fluid_flow.ntheta/2))
        >>> # to show the plots you can use:
        >>> # plt.show()
        """
        if ax is None:
            ax = plt.gca()
        x = np.zeros(self.nz)
        y_ext = np.zeros(self.nz)
        y_int = np.zeros(self.nz)
        for i in range(0, self.nz):
            x[i] = i * self.dz
            y_ext[i] = self.re[i][theta]
            y_int[i] = self.ri[i][theta]
        ax.plot(x, y_ext, "r")
        ax.plot(x, y_int, "b")
        ax.set_title("Shapes of stator and rotor along Z; Theta=" + str(theta))
        ax.set_xlabel("Points along Z")
        ax.set_ylabel("Radial direction")
        return ax

    def matplot_pressure_theta_cylindrical(self, z=0, from_numerical=True, ax=None):
        """This function assembles cylindrical pressure graphic in the theta direction for a given z,
        using matplotlib.
        Parameters
        ----------
        z: int, optional
            The distance along z-axis to be considered.
        from_numerical: bool, optional
            If True, takes the numerically calculated pressure matrix as entry.
            If False, takes the analytically calculated one instead.
            If condition cannot be satisfied (matrix not calculated), it will take the one that is available
            and raise a warning.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
        array([[...
        >>> ax = my_fluid_flow.matplot_pressure_theta_cylindrical(z=int(my_fluid_flow.nz/2))
        >>> # to show the plots you can use:
        >>> # plt.show()
        """
        if (
                not self.numerical_pressure_matrix_available
                and not self.analytical_pressure_matrix_available
        ):
            sys.exit(
                "Must calculate the pressure matrix. "
                "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
            )
        if from_numerical:
            if self.numerical_pressure_matrix_available:
                p_mat = self.p_mat_numerical
            else:
                p_mat = self.p_mat_analytical
                warnings.warn(
                    "Plotting from analytically calculated pressure matrix, as numerically calculated "
                    "one is not available."
                )
        else:
            if self.analytical_pressure_matrix_available:
                p_mat = self.p_mat_analytical
            else:
                p_mat = self.p_mat_numerical
                warnings.warn(
                    "Plotting from numerically calculated pressure matrix, as analytically calculated "
                    "one is not available."
                )
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        r = np.arange(
            0,
            self.radius_stator + 0.0001,
            (self.radius_stator - self.radius_rotor) / self.nradius,
        )
        theta = np.arange(-np.pi * 0.25, 1.75 * np.pi + self.dtheta / 2, self.dtheta)

        pressure_along_theta = np.zeros(self.ntheta)
        for i in range(0, self.ntheta):
            pressure_along_theta[i] = p_mat[0][i]

        min_pressure = np.amin(pressure_along_theta)

        r_matrix, theta_matrix = np.meshgrid(r, theta)
        z_matrix = np.zeros((theta.size, r.size))
        inner_radius_list = np.zeros(self.ntheta)
        pressure_list = np.zeros((theta.size, r.size))
        for i in range(0, theta.size):
            inner_radius = np.sqrt(
                self.xri[z][i] * self.xri[z][i] + self.yri[z][i] * self.yri[z][i]
            )
            inner_radius_list[i] = inner_radius
            for j in range(0, r.size):
                if r_matrix[i][j] < inner_radius:
                    continue
                pressure_list[i][j] = pressure_along_theta[i]
                z_matrix[i][j] = pressure_along_theta[i] - min_pressure + 0.01
        ax.contourf(theta_matrix, r_matrix, z_matrix, cmap="coolwarm")
        ax.set_title("Pressure along Theta; Z=" + str(z))
        return ax

    def matplot_pressure_theta(self, z=0, ax=None):
        """This function assembles pressure graphic in the theta direction for a given z,
        using matplotlib.
        Parameters
        ----------
        z: int, optional
            The distance along z-axis to be considered.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        Examples
        --------
        >>> my_fluid_flow = pressure_matrix_example()
        >>> my_fluid_flow.calculate_pressure_matrix_numerical() # doctest: +ELLIPSIS
        array([[...
        >>> ax = my_fluid_flow.matplot_pressure_theta(z=int(my_fluid_flow.nz/2))
        >>> # to show the plots you can use:
        >>> # plt.show()
        """
        if (
                not self.numerical_pressure_matrix_available
                and not self.analytical_pressure_matrix_available
        ):
            sys.exit(
                "Must calculate the pressure matrix. "
                "Try calling calculate_pressure_matrix_numerical() or calculate_pressure_matrix_analytical() first."
            )
        if ax is None:
            ax = plt.gca()
        list_of_thetas = []
        for t in range(0, self.ntheta):
            list_of_thetas.append(t * self.dtheta)
        if self.numerical_pressure_matrix_available:
            ax.plot(
                list_of_thetas, self.p_mat_numerical[z], "b", label="Numerical pressure"
            )
        if self.analytical_pressure_matrix_available:
            ax.plot(
                list_of_thetas,
                self.p_mat_analytical[z],
                "r",
                label="Analytical pressure",
            )
        ax.set_title("Pressure along Theta; Z=" + str(z))
        ax.set_xlabel("Points along Theta")
        ax.set_ylabel("Pressure")
        return ax


def pressure_matrix_example():
    """This function returns an instance of a simple pressure matrix.
    The purpose is to make available a simple model
    so that doctest can be written using it.

    Parameters
    ----------

    Returns
    -------
    An instance of a pressure matrix object.

    Examples
    --------
    >>> my_fluid_flow = pressure_matrix_example()
    >>> my_fluid_flow.eccentricity
    0.001
    """
    my_pressure_matrix = PressureMatrix(nz=8, ntheta=64, nradius=11, length=0.01, omega=100. * 2 * np.pi / 60,
                                        p_in=0., p_out=0., radius_rotor=0.08, radius_stator=0.1,
                                        viscosity=0.015, density=860., eccentricity=0.001, beta=np.pi)
    return my_pressure_matrix
