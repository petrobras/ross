import numpy as np
import matplotlib.pyplot as plt
import sys


class PressureMatrix:
    r"""This class calculates the pressure matrix of the object with the given parameters.

    It is supposed to be an attribute of a bearing or seal element,
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

    Geometric data of the problem
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Describes the geometric data of the problem.
    radius_rotor: float
        Rotor radius (m).
    radius_stator: float
        Stator Radius (m).
    xi: float
        Eccentricity (m) (distance between rotor and stator centers) on the x-axis.
        It is the position of the center of the rotor.
        The center of the stator is in position (0,0).
    yi: float
        Eccentricity (m) (distance between rotor and stator centers) on the y-axis.
        It is the position of the center of the rotor.
        The center of the stator is in position (0,0).

    Fluid characteristics
    ^^^^^^^^^^^^^^^^^^^^^
    Describes the fluid characteristics.
    visc: float
        Viscosity (Pa.s).
    rho: float
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
    p_mat : array of shape (nz, ntheta)
        The pressure matrix.
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
    distance_between_centers : float
        distance between the center of the rotor and the stator.
    difference_between_radius: float
        distance between the radius of the stator and the radius of the rotor.
    eccentricity_ratio: float
        distance_between_centers/difference_between_radius
    bearing_seal_type: int
        type of structure. 0: short; 1: long; 2: in between short and long.
        if length/radius_stator <= 1/8 it is short.
        if length/radius_stator > 4 it is long.

    Examples
    --------
    >>> from ross.fluid_flow import fluid_flow as flow
    >>> import matplotlib.pyplot as plt
    >>> nz = 150
    >>> ntheta = 37
    >>> nradius = 11
    >>> lb = 1.
    >>> omega = -100.*2*np.pi/60
    >>> p_in = 392266.
    >>> p_out = 100000.
    >>> radius_valley = 0.034
    >>> radius_crest = 0.039
    >>> radius_stator = 0.04
    >>> lwave = 0.18
    >>> xe = 0.
    >>> ye = 0.
    >>> visc = 0.001
    >>> rho = 1000.
    >>> my_pressure_matrix = flow.PressureMatrix(nz, ntheta, nradius, lb,
    ...                                          omega, p_in, p_out, radius_valley,
    ...                                          radius_crest, radius_stator,
    ...                                          lwave, xe, ye,  visc, rho, plot_eccentricity=True)
    >>> P = my_pressure_matrix.calculate_pressure_matrix()
    >>> my_pressure_matrix.plot_pressure_z(show_immediately=False)
    >>> my_pressure_matrix.plot_shape(show_immediately=False)
    >>> my_pressure_matrix.plot_pressure_theta(z=int(nz/2), show_immediately=False)
    >>> plt.show()

    """
    def __init__(self, nz, ntheta, nradius, length, omega, p_in,
                 p_out, radius_rotor, radius_stator, xi,
                 yi, visc, rho, plot_eccentricity=False):
        self.nz = nz
        self.ntheta = ntheta
        self.nradius = nradius
        self.n_interv_z = nz - 1
        self.n_interv_theta = ntheta - 1
        self.n_interv_radius = nradius - 1
        self.length = length
        self.ltheta = 2.*np.pi
        self.dz = length / self.n_interv_z
        self.dtheta = self.ltheta / self.n_interv_theta
        self.ntotal = self.nz * self.ntheta
        self.omega = omega
        self.p_in = p_in
        self.p_out = p_out
        self.radius_rotor = radius_rotor
        self.radius_stator = radius_stator
        self.xi = xi
        self.yi = yi
        self.visc = visc
        self.rho = rho
        self.re = np.zeros([self.nz, self.ntheta])
        self.ri = np.zeros([self.nz, self.ntheta])
        self.z = np.zeros([1, self.nz])
        self.xre = np.zeros([self.nz, self.ntheta])
        self.xri = np.zeros([self.nz, self.ntheta])
        self.yre = np.zeros([self.nz, self.ntheta])
        self.yri = np.zeros([self.nz, self.ntheta])
        self.p_mat = np.zeros([self.nz, self.ntheta])
        self.bearing_seal_type = -1
        self.calculate_coefficients(plot_eccentricity)
        self.pressure_matrix_available = False
        self.distance_between_centers = np.sqrt((xi**2 + yi**2))
        self.difference_between_radius = radius_stator - radius_rotor
        self.eccentricity_ratio = self.distance_between_centers/self.difference_between_radius

    def calculate_pressure_matrix(self):
        """This function calculates the pressure matrix
        """
        if self.bearing_seal_type == 0:
            for i in range(self.nz):
                for j in range(self.ntheta):
                    self.p_mat[i][j] = ((-3*self.visc*self.omega)/self.difference_between_radius**2)*\
                                       ((i*self.dz)**2 - (self.length**2)/4)*\
                                       (self.eccentricity_ratio*np.sin(j*self.dtheta))/\
                                       (1 + self.eccentricity_ratio*np.sin(j*self.dtheta))**3
            self.pressure_matrix_available = True
        return self.p_mat

    def calculate_coefficients(self, plot_cut=False):
        """This function calculates the constants that form the Poisson equation
        of the discrete pressure (central differences in the second
        derivatives). It is executed when the class is instantiated.
        Parameters
        ----------
        plot_cut: bool
            If True, plots a cut at z=0 and show, so the user can check the
            internal and external radius as well as eccentricity.
        """
        if self.length/self.radius_stator <= 1/8:
            self.bearing_seal_type = 0
        elif self.length/self.radius_stator > 4:
            self.bearing_seal_type = 1
        else:
            self.bearing_seal_type = 2
        if plot_cut:
            plt.figure(0)
        for i in range(self.nz):
            zno = i * self.dz
            self.z[0][i] = zno
            for j in range(self.ntheta):
                gama = j * self.dtheta
                [radius_external, self.xre[i][j], self.yre[i][j]] =\
                    self.external_radius_function(gama)
                [radius_internal, self.xri[i][j], self.yri[i][j]] =\
                    self.internal_radius_function(zno, gama)

                if plot_cut:
                    # Plot a cut at z=0:
                    if i == 0:
                        plt.plot(self.xre[i][j], self.yre[i][j], 'r.')
                        plt.plot(self.xri[i][j], self.yri[i][j], 'b.')
                        plt.plot(0, 0, '*')
                        plt.title('Cut in plane z=0')
                        plt.xlabel('X axis')
                        plt.ylabel('Y axis')
                        plt.axis('equal')

                self.re[i][j] = radius_external

                self.ri[i][j] = radius_internal
        if plot_cut:
            plt.show()

    def internal_radius_function(self, z, gama):
        """This function calculates the radius of the rotor given the
        radius and the position z.
        Parameters
        ----------
        z: float
            Distance along the z-axis.
        gama: float
            Gama is the distance in the theta-axis. It should range from 0 to 2*np.pi.
        """

        e = np.sqrt(self.xi ** 2 + self.yi ** 2)
        if self.xi > 0:
            beta = np.arctan(self.yi / self.xi)
        elif self.xi < 0:
            beta = -np.pi + np.arctan(self.yi / self.xi)
        else:
            if self.yi > 0.:
                beta = np.pi / 2.
            else:
                beta = -np.pi / 2.
        alpha = gama - beta
        radius_external = e * np.cos(alpha) + np.sqrt(
            self.radius_rotor ** 2 - (e * np.sin(alpha)) ** 2
            )
        xre = radius_external * np.cos(gama)
        yre = radius_external * np.sin(gama)
        return radius_external, xre, yre

    def external_radius_function(self, gama):
        """This function calculates the radius of the stator.
        Parameters
        ----------
        gama: float
            Gama is the distance in the theta-axis. It should range from 0 to 2*np.pi.
        """

        radius_internal = self.radius_stator
        xri = radius_internal * np.cos(gama)
        yri = radius_internal * np.sin(gama)

        return radius_internal, xri, yri

    def mounting_matrix(self):
        """Mounting matrix
        This function assembles the matrix M and the independent vector f.
        """
        m = np.zeros([self.ntotal, self.ntotal])
        f = np.zeros([self.ntotal, 1])

        """ Applying the boundary conditions in Z=0 e Z=L:"""
        counter = 0
        for x in range(self.ntheta):
            m[counter][counter] = 1
            f[counter][0] = self.p_in
            counter = counter + self.nz - 1
            m[counter][counter] = 1
            f[counter][0] = self.p_out
            counter = counter + 1

        """Applying the boundary conditions p(theta=0)=P(theta=pi):):"""
        counter = 0
        for x in range(self.nz - 2):
            m[self.ntotal - self.nz + 1 + counter][1 + counter] = 1
            m[self.ntotal - self.nz + 1 + counter][self.ntotal - self.nz + 1 + counter] = -1
            counter = counter + 1

        """Border nodes with periodic boundary condition:"""
        counter = 1
        j = 0
        for i in range(1, self.nz - 1):
            a = (1 / self.dtheta ** 2) * (self.c1[i][self.ntheta - 1])
            m[counter][self.ntotal - 2 * self.nz + counter] = a
            b = (1 / self.dz ** 2) * (self.c2[i - 1, j])
            m[counter][counter - 1] = b
            c = -((1 / self.dtheta ** 2) * (
                (self.c1[i][j]) + self.c1[i][self.ntheta - 1]
                ) + (1 / self.dz ** 2) * (self.c2[i][j] + self.c2[i - 1][j]))
            m[counter, counter] = c
            d = (1 / self.dz ** 2) * (self.c2[i][j])
            m[counter][counter + 1] = d
            e = (1 / self.dtheta ** 2) * (self.c1[i][j])
            m[counter][counter + self.nz] = e
            counter = counter + 1

        # Internal nodes
        counter = self.nz + 1
        for j in range(1, self.ntheta - 1):
            for i in range(1, self.nz - 1):
                a = (1 / self.dtheta ** 2) * (self.c1[i, j - 1])
                m[counter][counter - self.nz] = a
                b = (1 / self.dz ** 2) * (self.c2[i - 1][j])
                m[counter][counter - 1] = b
                c = -((1 / self.dtheta ** 2) * (
                    (self.c1[i][j]) + self.c1[i][j - 1]) + (1 / self.dz ** 2) *
                    (self.c2[i][j] + self.c2[i - 1][j])
                    )
                m[counter, counter] = c
                d = (1 / self.dz ** 2) * (self.c2[i][j])
                m[counter][counter + 1] = d
                e = (1 / self.dtheta ** 2) * (self.c1[i][j])
                m[counter][counter + self.nz] = e
                counter = counter + 1
            counter = counter + 2

        # Assembling the vector f:
        counter = 1
        for j in range(self.ntheta - 1):
            for i in range(1, self.nz - 1):
                if j == 0:
                    k = (
                        (self.c0w[i][j] - self.c0w[i][self.ntheta - 1]) /
                        self.dtheta
                        )
                    f[counter][0] = k
                else:
                    k = ((self.c0w[i, j] - self.c0w[i, j - 1]) / self.dtheta)
                    f[counter][0] = k
                counter = counter + 1
            counter = counter + 2
        return m, f

    """Graphics
    Plots the graphs of interest.
    """

    def plot_pressure_z(self, show_immediately=True):
        """This function assembles pressure graphic along the z-axis.
        Parameters
        ----------
        show_immediately: bool
            If True, immediately plots the graphic. Otherwise, the user should call plt.show()
            at some point. It is useful in case the user wants to see one graphic alongside another.
        """
        if not self.pressure_matrix_available:
            sys.exit('Must calculate the pressure matrix.'
                     'Try calling calculate_pressure_matrix first.')
        plt.figure(1)
        for i in range(0, self.nz):
            plt.plot(i*self.dz, self.p_mat[i][0], 'bo')
        plt.title('Pressure along the Z direction (direction of flow)')
        plt.xlabel('Points along the Z direction')
        plt.ylabel('Pressure')
        plt.show(block=show_immediately)

    def plot_shape(self, theta=0, show_immediately=True):
        """This function assembles a graphic representing the geometry of the rotor.
        Parameters
        ----------
        theta: int
            The theta to be considered.
        show_immediately: bool
            If True, immediately plots the graphic. Otherwise, the user should call plt.show()
            at some point. It is useful in case the user wants to see one graphic alongside another.
        """
        if not self.pressure_matrix_available:
            sys.exit('Must calculate the pressure matrix.'
                     'Try calling calculate_pressure_matrix first.')
        plt.figure(2)
        x = np.zeros(self.nz)
        y_re = np.zeros(self.nz)
        y_ri = np.zeros(self.nz)
        for i in range(0, self.nz):
            x[i] = i * self.dz
            y_re[i] = self.re[i][theta]
            y_ri[i] = self.ri[i][theta]
        plt.plot(x, y_re, 'r')
        plt.plot(x, y_ri, 'b')
        plt.show(block=show_immediately)

    def plot_pressure_theta(self, z=0, show_immediately=True):
        """This function assembles pressure graphic in the theta direction for a given z.
        Parameters
        ----------
        z: int
            The distance along z-axis to be considered.
        show_immediately: bool
            If True, immediately plots the graphic. Otherwise, the user should call plt.show()
            at some point. It is useful in case the user wants to see one graphic alongside another.
        """
        if not self.pressure_matrix_available:
            sys.exit('Must calculate the pressure matrix.'
                     'Try calling calculate_pressure_matrix first.')
        r = np.arange(0, self.radius_stator + 0.0001, (
            self.radius_stator - self.radius_rotor)/self.nradius
            )
        theta = np.arange(0, 2*np.pi + self.dtheta/2, self.dtheta)

        pressure_along_theta = np.zeros(self.ntheta)
        for i in range(0, self.ntheta):
            pressure_along_theta[i] = self.p_mat[0][i]

        min_pressure = np.amin(pressure_along_theta)

        r_matrix, theta_matrix = np.meshgrid(r, theta)
        z_matrix = np.zeros((theta.size, r.size))
        inner_radius_list = np.zeros(self.ntheta)
        pressure_list = np.zeros((theta.size, r.size))
        for i in range(0, theta.size):
            inner_radius = np.sqrt(self.xri[z][i] * self.xri[z][i] + self.yri[z][i] * self.yri[z][i])
            inner_radius_list[i] = inner_radius
            for j in range(0, r.size):
                if r_matrix[i][j] < inner_radius:
                    continue
                pressure_list[i][j] = pressure_along_theta[i]
                z_matrix[i][j] = pressure_along_theta[i] - min_pressure + 0.01
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        ax.contourf(theta_matrix, r_matrix, z_matrix, cmap='coolwarm')
        plt.show(block=show_immediately)

    def plot_pressure_graph(self, show_immediately=True):
        """ Given an instantiated PressureMatrix, plot its pressure along theta in z = length/2 and compare with
            Tribology Series's results.
        """
        plt.figure(3)
        if not self.pressure_matrix_available:
            sys.exit('Must calculate the pressure matrix.'
                     'Try calling calculate_pressure_matrix first.')
        theta_list = []
        for theta in range(0, self.ntheta):
            theta_list.append(theta * self.dtheta)
        plt.plot(theta_list, self.p_mat[int(self.nz / 2)], 'b')
        plt.show(block=show_immediately)

