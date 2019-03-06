import numpy as np
import matplotlib.pyplot as plt


class PressureMatrix:
    """Pressure Matrix
    Creates the pressure matrix
    """
    def __init__(
            self, nz, ntheta, nradius, n_interv_z, n_interv_theta,
            n_interv_radius, lb, ltheta, dz, dtheta, ntotal, omega, p_in,
            p_out, radius_valley, radius_crest, radius_stator, lwave, xe,
            ye, visc, rho
            ):
        self.nz = nz
        self.ntheta = ntheta
        self.nradius = nradius
        self.n_interv_z = n_interv_z
        self.n_interv_theta = n_interv_theta
        self.n_interv_radius = n_interv_radius
        self.lb = lb
        self.ltheta = ltheta
        self.dz = dz
        self.dtheta = dtheta
        self.ntotal = ntotal
        self.omega = omega
        self.p_in = p_in
        self.p_out = p_out
        self.radius_valley = radius_valley
        self.radius_crest = radius_crest
        self.radius_stator = radius_stator
        self.lwave = lwave
        self.xe = xe
        self.ye = ye
        self.visc = visc
        self.rho = rho
        self.c1 = np.zeros([self.nz, self.ntheta])
        self.c2 = np.zeros([self.nz, self.ntheta])
        self.c0w = np.zeros([self.nz, self.ntheta])
        self.re = np.zeros([self.nz, self.ntheta])
        self.ri = np.zeros([self.nz, self.ntheta])
        self.z = np.zeros([1, self.nz])
        self.xre = np.zeros([self.nz, self.ntheta])
        self.xri = np.zeros([self.nz, self.ntheta])
        self.yre = np.zeros([self.nz, self.ntheta])
        self.yri = np.zeros([self.nz, self.ntheta])
        self.p_mat = np.zeros([self.nz, self.ntheta])
        self.calculate_coefficients()
        self.pressure_matrix_available = False
        self.P = None

    def calculate_pressure_matrix(self):
        """Calculate pressure matrix
        This function calculates the pressure matrix
        """
        M, f = self.mounting_matrix()
        self.P = self.resolves_matrix(M, f)
        self.p_matrix(self.P)
        return self.p_mat

    def calculate_coefficients(self):
        """Calculate coeddicients
        This function calculates the constants that form the Poisson equation
        of the discrete pressure (central differences in the second
        derivatives)
        """
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

                # Plot a cut at z=0:
                if i == 0:
                    plt.plot(self.xre[i][j], self.yre[i][j], 'r.')
                    plt.plot(self.xri[i][j], self.yri[i][j], 'b.')
                    plt.plot(0, 0, '*')
                    plt.title('Cut in plane z=0')
                    plt.xlabel('X axis')
                    plt.ylabel('Y axis')
                    plt.axis('equal')

                w = self.omega * radius_internal

                k = (1. / (radius_internal ** 2 - radius_external ** 2)) *\
                    (
                            (radius_external**2) *
                            (-(1./2) + np.log(radius_external)) -
                            (radius_internal**2) *
                            (-(1./2) + np.log(radius_internal))
                    )

                self.c1[i][j] = (1./(4 * self.visc)) *\
                (
                        (radius_external**2) * (np.log(radius_external)) -
                        (radius_internal**2) * (np.log(radius_internal)) +
                        (radius_external**2 - radius_internal**2) * (k - 1)
                ) -\
                (
                        (radius_external**2) / (2 * self.visc)
                ) *\
                (
                        (np.log(radius_external) + k - 1/2) *
                        np.log(radius_external / radius_internal)
                )

                self.c2[i][j] = - ((radius_internal**2) / (8. * self.visc)) *\
                (
                    (
                            radius_external**2 - radius_internal**2 -
                            (radius_external**4 - radius_internal**4) /
                            (2 * (radius_internal ** 2))
                    ) +
                    (
                            (radius_external ** 2 - radius_internal ** 2) /
                            (
                                    (radius_internal ** 2) *
                                    (np.log(radius_external / radius_internal))
                            )
                    ) *
                    (
                            (radius_external ** 2) * np.log
                        (
                        radius_external / radius_internal
                        ) - (
                            radius_external**2 - radius_internal**2
                            ) / 2.
                    )
                )

                self.c0w[i][j] = - (w * radius_internal) * (
                    (
                        (np.log(radius_external / radius_internal)) *
                        (
                            1 + (radius_internal ** 2) /
                            (radius_external**2 - radius_internal ** 2)
                        )
                    ) - (1. / 2)
                    )

                self.re[i][j] = radius_external

                self.ri[i][j] = radius_internal

    def external_radius_function(self, gama):
        """External radius function
        This function calculates the radius of the stator in the center of
        coordinates given the theta angle, the value of the eccentricity
        and its radius.
        Parameters
        ----------
        betha: float
            Betha is the angle that indicates the location of the eccentricity.
        alpha: float
            Alpha is the angle between THETA and eccentricity (betha).
        """
        e = np.sqrt(self.xe ** 2 + self.ye ** 2)
        if (self.xe > 0. and self.ye >= 0.) or (self.xe > 0 and self.ye < 0):
            betha = np.arctan(self.ye / self.xe)
        if (self.xe < 0. and self.ye <= 0.) or (self.xe < 0 and self.ye > 0):
            betha = -np.pi + np.arctan(self.ye / self.xe)
        if self.xe == 0.:
            if self.ye > 0.:
                betha = np.pi / 2.
            else:
                betha = -np.pi / 2.
        alpha = gama - betha
        radius_external = e * np.cos(alpha) + np.sqrt(
            self.radius_stator ** 2 - (e * np.sin(alpha)) ** 2
            )
        xre = radius_external * np.cos(gama)
        yre = radius_external * np.sin(gama)
        return radius_external, xre, yre

    def internal_radius_function(self, z, gama):
        """Internal radius function
        This function calculates the radius of the rotor given the crest
        radius, the valley radius and the position z.
        Examples
        --------
        >>>CYLINDRICAL CASE = (radius_internal = self.radius_valley)
        >>>CONICAL CASE = (radius_internal = self.radius_valley +
        ((self.radius_crest - self.radius_valley)/self.lb) * z)
        >>>SINUSOIDAL CASE = (radius_internal = (self.radius_valley +
        self.radius_crest)/2. + ((self.radius_crest - self.radius_valley)/2.)
        * np.sin((2*np.pi/self.lwave)*z + np.pi/2.))
        """

        radius_internal = self.radius_valley + (
            (self.radius_crest - self.radius_valley)/self.lb
            )*z
        xri = radius_internal * np.cos(gama)
        yri = radius_internal * np.sin(gama)

        return radius_internal, xri, yri

    def mounting_matrix(self):
        """Mounting matrix
        This function assembles the matrix M and the independent vector f.
        """
        M = np.zeros([self.ntotal, self.ntotal])
        f = np.zeros([self.ntotal, 1])

        """ Applying the boundary conditions in Z=0 e Z=L:"""
        counter = 0
        for x in range(self.ntheta):
            M[counter][counter] = 1
            f[counter][0] = self.p_in
            counter = counter + self.nz - 1
            M[counter][counter] = 1
            f[counter][0] = self.p_out
            counter = counter + 1

        """Applying the boundary conditions p(theta=0)=P(theta=pi):):"""
        counter = 0
        for x in range(self.nz - 2):
            M[self.ntotal - self.nz + 1 + counter][1 + counter] = 1
            M[self.ntotal - self.nz + 1 + counter][self.ntotal - self.nz + 1 + counter] = -1
            counter = counter + 1

        """Border nodes with periodic boundary condition:"""
        counter = 1
        j = 0
        for i in range(1, self.nz - 1):
            a = (1 / self.dtheta ** 2) * (self.c1[i][self.ntheta - 1])
            M[counter][self.ntotal - 2 * self.nz + counter] = a
            b = (1 / self.dz ** 2) * (self.c2[i - 1, j])
            M[counter][counter - 1] = b
            c = -((1 / self.dtheta ** 2) * (
                (self.c1[i][j]) + self.c1[i][self.ntheta - 1]
                ) + (1 / self.dz ** 2) * (self.c2[i][j] + self.c2[i - 1][j]))
            M[counter, counter] = c
            d = (1 / self.dz ** 2) * (self.c2[i][j])
            M[counter][counter + 1] = d
            e = (1 / self.dtheta ** 2) * (self.c1[i][j])
            M[counter][counter + self.nz] = e
            counter = counter + 1

        # Internal nodes
        counter = self.nz + 1
        for j in range(1, self.ntheta - 1):
            for i in range(1, self.nz - 1):
                a = (1 / self.dtheta ** 2) * (self.c1[i, j - 1])
                M[counter][counter - self.nz] = a
                b = (1 / self.dz ** 2) * (self.c2[i - 1][j])
                M[counter][counter - 1] = b
                c = -((1 / self.dtheta ** 2) * (
                    (self.c1[i][j]) + self.c1[i][j - 1]) + (1 / self.dz ** 2) *
                    (self.c2[i][j] + self.c2[i - 1][j])
                    )
                M[counter, counter] = c
                d = (1 / self.dz ** 2) * (self.c2[i][j])
                M[counter][counter + 1] = d
                e = (1 / self.dtheta ** 2) * (self.c1[i][j])
                M[counter][counter + self.nz] = e
                counter = counter + 1
            counter = counter + 2

        # Assembling the vector f:
        counter = 1
        for j in range(self.ntheta - 1):
            for i in range(1, self.nz - 1):
                if j == 0:
                    l = (
                        (self.c0w[i][j] - self.c0w[i][self.ntheta - 1]) /
                        (self.dtheta)
                        )
                    f[counter][0] = l
                else:
                    l = ((self.c0w[i, j] - self.c0w[i, j - 1]) / (self.dtheta))
                    f[counter][0] = l
                counter = counter + 1
            counter = counter + 2
        return M, f

    def resolves_matrix(self, M, f):
        """Resolves matrix
        This function resolves the linear system [M]{P}={f}.
        """
        P = np.linalg.solve(M, f)
        return P

    def p_matrix(self, P):
        """P matrix
        This function creates the pressure matrix.
        """
        for i in range(self.nz):
            for j in range(self.ntheta):
                k = j * self.nz + i
                self.p_mat[i][j] = P[k]

    """Graphics
    Plots the graphs of interest.
    """

    def plot_pressure_z(self):
        """Plot pressure z
        Assemble pressure graph along the z-axis.
        """
        plt.figure(1)
        for i in range(0, self.nz):
            plt.plot(i*self.dz, self.P[i], 'bo')
        plt.title('Pressure along the Z direction (direction of flow)')
        plt.xlabel('Points along the Z direction')
        plt.ylabel('Pressure')
        plt.show(block=False)

    def plot_shape(self, theta=0):
        """Plot shape
        Assemble a graph representing the geometry of the rotor.
        """
        plt.figure(2)
        x = np.zeros(self.nz)
        y_re = np.zeros(self.nz)
        y_ri = np.zeros(self.nz)
        for i in range(0, self.nz):
            x[i] = i * self.dz
            y_re[i] = self.re[i, theta]
            y_ri[i] = self.ri[i, theta]
        plt.plot(x, y_re, 'r')
        plt.plot(x, y_ri, 'b')
        plt.show(block=False)

    def plot_pressure_theta(self):
        """Plot pressure theta
        Assemble pressure graph in the theta direction.
        """
        middle_section = int(self.nz/2)

        r = np.arange(0, self.radius_stator + 0.0001, (
            self.radius_stator - self.radius_valley)/self.nradius
            )
        theta = np.arange(0, 2*np.pi + self.dtheta/2, self.dtheta)

        pressure_along_theta = np.zeros(self.ntheta)
        for i in range(0, self.ntheta):
            pressure_along_theta[i] = self.P[middle_section + i*self.nz]

        min_pressure = np.amin(pressure_along_theta)

        r_matrix, theta_matrix = np.meshgrid(r, theta)
        z_matrix = np.zeros((theta.size, r.size))
        inner_radius_list = np.zeros(self.ntheta)
        pressure_list = np.zeros((theta.size, r.size))
        for i in range(0, theta.size):
            new_x = self.xri[middle_section][i] - self.xe
            new_y = self.yri[middle_section][i] - self.ye
            inner_radius = np.sqrt(new_x * new_x + new_y * new_y)
            inner_radius_list[i] = inner_radius
            for j in range(0, r.size):
                if r_matrix[i][j] < inner_radius:
                    continue
                pressure_list[i][j] = pressure_along_theta[i]
                z_matrix[i][j] = pressure_along_theta[i] - min_pressure + 0.01
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        ax.contourf(theta_matrix, r_matrix, z_matrix, cmap='coolwarm')
        plt.show(block=False)
