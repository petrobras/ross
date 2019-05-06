import numpy as np
import matplotlib.pyplot as plt
import sys


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

    Geometric data of the problem
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Describes the geometric data of the problem.
    radius_rotor: float
        Rotor radius (m).
    radius_stator: float
        Stator Radius (m).
    eccentricity: float
        Eccentricity (m) is the euclidean distance between rotor and stator centers.
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
        if length/radius_stator <= 1/8 it is short.
        if length/radius_stator > 4 it is long.
    plot_counter: int
        for each plot, it increments this counter, in order to plot
        figures with unique IDs.
    radial_clearance: float
        Difference between both stator and rotor radius, regardless of eccentricity.

    Examples
    --------
    >>> from ross.fluid_flow import fluid_flow as flow
    >>> import matplotlib.pyplot as plt
    >>> nz = 20
    >>> ntheta = 100
    >>> nradius = 11
    >>> length = 0.01
    >>> omega = -100.*2*np.pi/60
    >>> p_in = 1.
    >>> p_out = 1.
    >>> radius_rotor = 0.08
    >>> radius_stator = 0.1
    >>> eccentricity = 0.001
    >>> visc = 0.015
    >>> rho = 860.
    >>> my_pressure_matrix = flow.PressureMatrix(nz, ntheta, nradius, length,
    ...                                          omega, p_in, p_out, radius_rotor,
    ...                                          radius_stator, eccentricity, visc, rho,
    ...                                          plot_eccentricity=True)
    >>> P = my_pressure_matrix.calculate_pressure_matrix()
    >>> my_pressure_matrix.plot_pressure_z(show_immediately=False)
    >>> my_pressure_matrix.plot_shape(show_immediately=False)
    >>> my_pressure_matrix.plot_pressure_theta_cylindrical(z=int(nz/2), show_immediately=False)
    >>> my_pressure_matrix.plot_pressure_theta(z=int(nz/2), show_immediately=False)
    >>> plt.show()
    >>> plt.close('all')

    """
    def __init__(self, nz, ntheta, nradius, length, omega, p_in,
                 p_out, radius_rotor, radius_stator, eccentricity,
                 visc, rho, plot_eccentricity=False):
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
        self.eccentricity = eccentricity
        self.xi = np.sqrt(2)*eccentricity/2
        self.yi = -self.xi
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
        self.bearing_type = ''
        self.plot_counter = 0
        self.calculate_coefficients(plot_eccentricity)
        self.pressure_matrix_available = False
        self.difference_between_radius = radius_stator - radius_rotor
        self.eccentricity_ratio = self.eccentricity/self.difference_between_radius
        self.radial_clearance = self.radius_stator - self.radius_rotor

    def calculate_pressure_matrix(self):
        """This function calculates the pressure matrix
        """
        if self.bearing_type == 'short_bearing':
            for i in range(self.nz):
                for j in range(self.ntheta):
                    self.p_mat[i][j] = ((-3*self.visc*self.omega)/self.difference_between_radius**2) * \
                                       ((i*self.dz)**2 - (self.length**2)/4) * \
                                       (self.eccentricity_ratio*np.sin(j*self.dtheta)) / \
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
            self.bearing_type = 'short_bearing'
        elif self.length/self.radius_stator > 4:
            self.bearing_type = 'long_bearing'
        else:
            self.bearing_type = 'medium_size'
        for i in range(self.nz):
            zno = i * self.dz
            self.z[0][i] = zno
            for j in range(self.ntheta):
                gama = j * self.dtheta
                [radius_external, self.xre[i][j], self.yre[i][j]] =\
                    self.external_radius_function(gama)
                [radius_internal, self.xri[i][j], self.yri[i][j]] =\
                    self.internal_radius_function(zno, gama)
                self.re[i][j] = radius_external
                self.ri[i][j] = radius_internal
        if plot_cut:
            self.plot_eccentricity()

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
        """
        radius_internal = self.radius_rotor
        xri = radius_internal * np.cos(gama - np.pi/4)
        yri = radius_internal * np.sin(gama - np.pi/4)
        return radius_internal, xri + self.xi, yri + self.yi

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
        """
        radius_external = self.radius_stator
        xre = radius_external * np.cos(gama - np.pi/4)
        yre = radius_external * np.sin(gama - np.pi/4)

        return radius_external, xre, yre

    def get_rotor_load(self):
        """Returns the load applied to the rotor.
        Returns
        -------
        float
            Load applied to the rotor.
        """
        return -((np.pi*self.radius_stator*2*self.omega*self.visc*(self.length**3)*self.eccentricity_ratio)
                 / (8*(self.radial_clearance**2)*((1 - self.eccentricity_ratio**2)**2))) \
            * np.sqrt((16/np.pi - 1)*self.eccentricity_ratio + 1)

    def sommerfeld_number(self, f):
        """Return the sommerfeld number.
        Parameters
        ----------
        f: float
            Load applied to the rotor.
        Returns
        -------
        float
            The sommerfeld number.
        """
        modified_s = (self.radius_stator*2*self.omega*self.visc*(self.length**3)) / \
                     (8*f*(self.radial_clearance**2))
        return (modified_s/np.pi)*(self.radius_stator*2/self.length)**2

    def calculate_eccentricity_ratio(self, f):
        """Calculate the eccentricity ratio using the sommerfeld number.
        Parameters
        ----------
        f: float
            Load applied to the rotor.
        Returns
        -------
        float
            The eccentricity ratio.
        """
        s = self.sommerfeld_number(f)
        coefficients = [1, -4, (6 - (s**2)*(16 - np.pi**2)), -(4 + (np.pi**2)*(s**2)), 1]
        roots = np.roots(coefficients)
        for i in range(0, len(roots)):
            if 0 <= roots[i] <= 1:
                return np.sqrt(roots[i])
        sys.exit("Eccentricity ratio could not be calculated.")

    def plot_eccentricity(self, z=0, show_immediately=True):
        """This function assembles pressure graphic along the z-axis.
        The first few plots are of a different color to indicate where theta begins.
        Parameters
        ----------
        z: int
            The distance in z where to cut and plot.
        show_immediately: bool
            If True, immediately plots the graphic. Otherwise, the user should call plt.show()
            at some point. It is useful in case the user wants to see one graphic alongside another.
        """
        plt.figure(self.plot_counter)
        self.plot_counter += 1
        for j in range(0, self.ntheta):
            if j < 5:
                plt.plot(self.xre[z][j], self.yre[z][j], 'k.')
                plt.plot(self.xri[z][j], self.yri[z][j], 'k.')
            else:
                plt.plot(self.xre[z][j], self.yre[z][j], 'r.')
                plt.plot(self.xri[z][j], self.yri[z][j], 'b.')
            plt.plot(0, 0, '*')
            plt.title('Cut in plane Z=' + str(z))
            plt.xlabel('X axis')
            plt.ylabel('Y axis')
            plt.axis('equal')
        plt.show(block=show_immediately)
        if show_immediately:
            plt.close('all')

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
        plt.figure(self.plot_counter)
        self.plot_counter += 1
        for i in range(0, self.nz):
            plt.plot(i*self.dz, self.p_mat[i][0], 'bo')
        plt.title('Pressure along the Z direction (direction of flow); Theta=0')
        plt.xlabel('Points along Z')
        plt.ylabel('Pressure')
        plt.show(block=show_immediately)
        if show_immediately:
            plt.close('all')

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
        plt.figure(self.plot_counter)
        self.plot_counter += 1
        x = np.zeros(self.nz)
        y_re = np.zeros(self.nz)
        y_ri = np.zeros(self.nz)
        for i in range(0, self.nz):
            x[i] = i * self.dz
            y_re[i] = self.re[i][theta]
            y_ri[i] = self.ri[i][theta]
        plt.plot(x, y_re, 'r')
        plt.plot(x, y_ri, 'b')
        plt.title('Shapes of stator and rotor along Z; Theta='+str(theta))
        plt.xlabel('Points along Z')
        plt.ylabel('Radial direction')
        plt.show(block=show_immediately)
        if show_immediately:
            plt.close('all')

    def plot_pressure_theta_cylindrical(self, z=0, show_immediately=True):
        """This function assembles cylindrical pressure graphic in the theta direction for a given z.
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
        theta = np.arange(-np.pi*0.25, 1.75*np.pi + self.dtheta/2, self.dtheta)

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
        self.plot_counter += 1
        ax.contourf(theta_matrix, r_matrix, z_matrix, cmap='coolwarm')
        plt.title('Pressure along Theta; Z='+str(z))
        plt.show(block=show_immediately)
        if show_immediately:
            plt.close('all')

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
        plt.figure(self.plot_counter)
        self.plot_counter += 1
        if not self.pressure_matrix_available:
            sys.exit('Must calculate the pressure matrix.'
                     'Try calling calculate_pressure_matrix first.')
        theta_list = []
        for theta in range(0, self.ntheta):
            theta_list.append(theta * self.dtheta)
        plt.plot(theta_list, self.p_mat[z], 'b')
        plt.title('Pressure along Theta; Z='+str(z))
        plt.xlabel('Points along Theta')
        plt.ylabel('Pressure')
        plt.show(block=show_immediately)
        if show_immediately:
            plt.close('all')

