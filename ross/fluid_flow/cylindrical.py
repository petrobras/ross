import time

import numpy as np
from numpy.linalg import pinv
from scipy.optimize import curve_fit, minimize

from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units


class THDCylindrical(BearingElement):
    """This class calculates the pressure and temperature field in oil film of
    a cylindrical bearing. It is also possible to obtain the stiffness and
    damping coefficients.
    The basic references for the code is found in :cite:t:`barbosa2018`, :cite:t:`daniel2012` and :cite:t:`nicoletti1999`.

    Parameters
    ----------
    Bearing Geometry
    ^^^^^^^^^^^^^^^^
    Describes the geometric characteristics.
    axial_length : float, pint.Quantity
        Bearing length. Default unit is meter.
    journal_radius : float
        Rotor radius. The unit is meter.
    radial_clearance : float
        Radial clearence between rotor and bearing. The unit is meter.
    n_pad : integer
        Number of pads that compound the bearing surface.
    pad_arc_length : float
        Arc length of each pad. The unit is degree.
    initial_guess : array
        Array with eccentricity ratio and attitude angle
    method : string
        Choose the method to calculate the dynamics coefficients. Options are:
        - 'lund'
        - 'perturbation'
    print_progress : bool
        Set it True to print the score and forces on each iteration.
        False by default.
    print_result : bool
        Set it True to print result at the end.
        False by default.
    print_time : bool
        Set it True to print the time at the end.
        False by default.

    Operation conditions
    ^^^^^^^^^^^^^^^^^^^^
    Describes the operation conditions of the bearing.
    speed : float, pint.Quantity
        Rotor rotational speed. Default unit is rad/s.
    load_x_direction : Float
        Load in X direction. The unit is newton.
    load_y_direction : Float
        Load in Y direction. The unit is newton.

    Fluid propierties
    ^^^^^^^^^^^^^^^^^
    Describes the fluid characteristics.

    lubricant : str
        Lubricant type. Can be:
        - 'ISOVG46'
    reference_temperature : float
        Oil reference temperature. The unit is celsius.
    reference_viscosity : float
        Oil viscosity at reference temperature. Unit is Pa.s.
    groove_factor : list, numpy array, tuple or float
        Ratio of oil in reservoir temperature that mixes with the circulating oil.
        Is required one factor per segment.


    Turbulence Model
    ^^^^^^^^^^^^^^^^
    Turbulence model to improve analysis in higher speed.The model represents
    the turbulence by eddy diffusivities. The basic reference is found in :cite:t:`suganami1979`

    Reyn : Array
        The Reynolds number is a dimensionless number used to calculate the
        fluid flow regime inside the bearing.
    delta_turb : float
        Eddy viscosity scaling factor. Coefficient to assign weight to laminar,
        transitional and turbulent flows to calculate viscosity.

    Mesh discretization
    ^^^^^^^^^^^^^^^^^^^
    Describes the discretization of the bearing.
    elements_circumferential : int
        Number of volumes along the direction theta (direction of flow).
    elements_axial : int
        Number of volumes along the Z direction (axial direction).



    Returns
    -------
    A THDCylindrical object.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Attributes
    ----------
    Pdim : array
        Dimensional pressure field. The unit is pascal.
    dPdz : array
        Differential pressure field in z direction.
    dPdy : array
        Differential pressure field in theta direction.
    Tdim : array
        Dimensional temperature field. The unit is celsius.
    Fhx : float
        Force in X direction. The unit is newton.
    Fhy : float
        Force in Y direction. The unit is newton.
    equilibrium_pos : array
        Array with excentricity ratio and attitude angle information.
        Its shape is: array([excentricity, angle])

    Examples
    --------
    >>> from ross.fluid_flow.cylindrical import cylindrical_bearing_example
    >>> bearing = cylindrical_bearing_example()
    >>> bearing.equilibrium_pos
    array([ 0.60678516, -0.73288691])
    """

    @check_units
    def __init__(
        self,
        axial_length,
        journal_radius,
        radial_clearance,
        elements_circumferential,
        elements_axial,
        n_y,
        n_pad,
        pad_arc_length,
        reference_temperature,
        reference_viscosity,
        speed,
        load_x_direction,
        load_y_direction,
        groove_factor,
        lubricant,
        node,
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        show_coef=False,
        print_result=False,
        print_progress=False,
        print_time=False,
    ):

        self.axial_length = axial_length
        self.journal_radius = journal_radius
        self.radial_clearance = radial_clearance
        self.elements_circumferential = elements_circumferential
        self.elements_axial = elements_axial
        self.n_y = n_y
        self.n_pad = n_pad
        self.reference_temperature = reference_temperature
        self.reference_viscosity = reference_viscosity
        self.load_x_direction = load_x_direction
        self.load_y_direction = load_y_direction
        self.lubricant = lubricant
        self.fat_mixt = np.array(groove_factor)
        self.equilibrium_pos = None
        self.sommerfeld_type = sommerfeld_type
        self.initial_guess = initial_guess
        self.method = method
        self.show_coef = show_coef
        self.print_result = print_result
        self.print_progress = print_progress
        self.print_time = print_time

        if self.n_y == None:
            self.n_y = self.elements_circumferential

        self.betha_s_dg = pad_arc_length
        self.betha_s = pad_arc_length * np.pi / 180

        self.thetaI = 0
        self.thetaF = self.betha_s
        self.dtheta = (self.thetaF - self.thetaI) / (self.elements_circumferential)

        ##
        # Dimensionless discretization variables

        self.dY = 1 / self.n_y
        self.dZ = 1 / self.elements_axial

        # Z-axis direction

        self.Z_I = 0
        self.Z_F = 1
        Z = np.zeros((self.elements_axial + 2))

        Z[0] = self.Z_I
        Z[self.elements_axial + 1] = self.Z_F
        Z[1 : self.elements_axial + 1] = np.arange(
            self.Z_I + 0.5 * self.dZ, self.Z_F, self.dZ
        )
        self.Z = Z

        # Dimensionalization

        self.dz = self.dZ * self.axial_length
        self.dy = self.dY * self.betha_s * self.journal_radius

        self.Zdim = self.Z * self.axial_length

        self.lubricant_dict = {
            "ISOVG32": {
                "viscosity1": Q_(4.05640e-06, "reyn").to_base_units().m,
                "temp1": Q_(40.00000, "degC").to_base_units().m,
                "viscosity2": Q_(6.76911e-07, "reyn").to_base_units().m,
                "temp2": Q_(100.00000, "degC").to_base_units().m,
                "lube_density": Q_(873.99629, "kg/m³").to_base_units().m,
                "lube_cp": Q_(1948.7995685758851, "J/(kg*degK)").to_base_units().m,
                "lube_conduct": Q_(0.13126, "W/(m*degC)").to_base_units().m,
            },
            "ISOVG46": {
                "viscosity1": Q_(5.757040938820288e-06, "reyn").to_base_units().m,
                "temp1": Q_(40, "degC").to_base_units().m,
                "viscosity2": Q_(8.810775697672788e-07, "reyn").to_base_units().m,
                "temp2": Q_(100, "degC").to_base_units().m,
                "lube_density": Q_(862.9, "kg/m³").to_base_units().m,
                "lube_cp": Q_(1950, "J/(kg*degK)").to_base_units().m,
                "lube_conduct": Q_(0.15, "W/(m*degC)").to_base_units().m,
            },
            "TEST": {
                "viscosity1": Q_(2.758e-6, "reyn").to_base_units().m,
                "temp1": Q_(121.7, "degF").to_base_units().m,
                "viscosity2": Q_(1.119e-6, "reyn").to_base_units().m,
                "temp2": Q_(175.7, "degF").to_base_units().m,
                "lube_density": Q_(8.0e-5, "lbf*s²/in⁴").to_base_units().m,
                "lube_cp": Q_(1.79959e2, "BTU*in/(lbf*s²*degF)").to_base_units().m,
                "lube_conduct": Q_(2.00621e-6, "BTU/(in*s*degF)").to_base_units().m,
            },
        }

        lubricant_properties = self.lubricant_dict[self.lubricant]
        T_muI = Q_(lubricant_properties["temp1"], "degK").m_as("degC")
        T_muF = Q_(lubricant_properties["temp2"], "degK").m_as("degC")
        mu_I = lubricant_properties["viscosity1"]
        mu_F = lubricant_properties["viscosity2"]
        self.rho = lubricant_properties["lube_density"]
        self.Cp = lubricant_properties["lube_cp"]
        self.k_t = lubricant_properties["lube_conduct"]

        # Interpolation coefficients
        self.a, self.b = self._interpol(T_muI, T_muF, mu_I, mu_F)

        number_of_freq = np.shape(speed)[0]

        kxx = np.zeros(number_of_freq)
        kxy = np.zeros(number_of_freq)
        kyx = np.zeros(number_of_freq)
        kyy = np.zeros(number_of_freq)

        cxx = np.zeros(number_of_freq)
        cxy = np.zeros(number_of_freq)
        cyx = np.zeros(number_of_freq)
        cyy = np.zeros(number_of_freq)

        for ii in range(number_of_freq):

            self.speed = speed[ii]

            self.run()

            coefs = self.coefficients()
            stiff = True
            for coef in coefs:
                if stiff:
                    kxx[ii] = coef[0]
                    kxy[ii] = coef[1]
                    kyx[ii] = coef[2]
                    kyy[ii] = coef[3]

                    stiff = False
                else:
                    cxx[ii] = coef[0]
                    cxy[ii] = coef[1]
                    cyx[ii] = coef[2]
                    cyy[ii] = coef[3]

        super().__init__(node, kxx, cxx, kyy, kxy, kyx, cyy, cxy, cyx, speed)

    def _forces(self, initial_guess, y0, xpt0, ypt0):
        """Calculates the forces in Y and X direction.

        Parameters
        ----------
        initial_guess : array, float
            If the other parameters are None, initial_guess is an array with eccentricity
            ratio and attitude angle. Else, initial_guess is the position of the center of
            the rotor in the x-axis.
        y0 : float
            The position of the center of the rotor in the y-axis.
        xpt0 : float
            The speed of the center of the rotor in the x-axis.
        ypt0 : float
            The speed of the center of the rotor in the y-axis.


        Returns
        -------
        Fhx : float
            Force in X direction. The unit is newton.
        Fhy : float
            Force in Y direction. The unit is newton.
        """
        if y0 is None and xpt0 is None and ypt0 is None:
            self.initial_guess = initial_guess

            xr = (
                self.initial_guess[0]
                * self.radial_clearance
                * np.cos(self.initial_guess[1])
            )
            yr = (
                self.initial_guess[0]
                * self.radial_clearance
                * np.sin(self.initial_guess[1])
            )
            self.Y = yr / self.radial_clearance
            self.X = xr / self.radial_clearance

            self.Xpt = 0
            self.Ypt = 0
        else:
            self.X = initial_guess / self.radial_clearance
            self.Y = y0 / self.radial_clearance

            self.Xpt = xpt0 / (self.radial_clearance * self.speed)
            self.Ypt = ypt0 / (self.radial_clearance * self.speed)

        T_conv = 0.8 * self.reference_temperature

        T_mist = self.reference_temperature * np.ones(self.n_pad)

        Reyn = np.zeros(
            (self.elements_axial, self.elements_circumferential, self.n_pad)
        )

        pad_ct = [ang for ang in range(0, 360, int(360 / self.n_pad))]

        self.thetaI = np.radians(
            [pad + (180 / self.n_pad) - (self.betha_s_dg / 2) for pad in pad_ct]
        )

        self.thetaF = np.radians(
            [pad + (180 / self.n_pad) + (self.betha_s_dg / 2) for pad in pad_ct]
        )

        Ytheta = [
            np.linspace(t1, t2, self.elements_circumferential)
            for t1, t2 in zip(self.thetaI, self.thetaF)
        ]

        self.pad_ct = [ang for ang in range(0, 360, int(360 / self.n_pad))]

        self.thetaI = np.radians(
            [pad + (180 / self.n_pad) - (self.betha_s_dg / 2) for pad in self.pad_ct]
        )

        self.thetaF = np.radians(
            [pad + (180 / self.n_pad) + (self.betha_s_dg / 2) for pad in self.pad_ct]
        )

        Ytheta = [
            np.linspace(t1, t2, self.elements_circumferential)
            for t1, t2 in zip(self.thetaI, self.thetaF)
        ]

        while (T_mist[0] - T_conv) >= 1e-2:

            self.P = np.zeros(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )
            dPdy = np.zeros(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )
            dPdz = np.zeros(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )
            T = np.ones(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )
            T_new = (
                np.ones(
                    (self.elements_axial, self.elements_circumferential, self.n_pad)
                )
                * 1.2
            )

            T_conv = T_mist[0]

            mu_new = 1.1 * np.ones(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )
            mu_turb = 1.3 * np.ones(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )

            PP = np.zeros(((self.elements_axial), (2 * self.elements_circumferential)))

            nk = (self.elements_axial) * (self.elements_circumferential)

            Mat_coef = np.zeros((nk, nk))
            Mat_coef_T = np.zeros((nk, nk))
            b = np.zeros((nk, 1))
            b_T = np.zeros((nk, 1))

            for n_p in np.arange(self.n_pad):

                T_ref = T_mist[n_p - 1]

                # Temperature convergence while

                while (
                    np.linalg.norm(T_new[:, :, n_p] - T[:, :, n_p])
                    / np.linalg.norm(T[:, :, n_p])
                    >= 1e-3
                ):

                    T_ref = T_mist[n_p - 1]

                    mu = mu_new
                    self.mu_l = mu_new

                    T[:, :, n_p] = T_new[:, :, n_p]

                    ki = 0
                    kj = 0
                    k = 0

                    # Solution of pressure field initialization

                    for ii in np.arange((self.Z_I + 0.5 * self.dZ), self.Z_F, self.dZ):
                        for jj in np.arange(
                            self.thetaI[n_p] + (self.dtheta / 2),
                            self.thetaF[n_p],
                            self.dtheta,
                        ):

                            hP = 1 - self.X * np.cos(jj) - self.Y * np.sin(jj)
                            he = (
                                1
                                - self.X * np.cos(jj + 0.5 * self.dtheta)
                                - self.Y * np.sin(jj + 0.5 * self.dtheta)
                            )
                            hw = (
                                1
                                - self.X * np.cos(jj - 0.5 * self.dtheta)
                                - self.Y * np.sin(jj - 0.5 * self.dtheta)
                            )
                            hn = hP
                            hs = hn

                            if kj == 0 and ki == 0:
                                MU_e = 0.5 * (mu[ki, kj] + mu[ki, kj + 1])
                                MU_w = mu[ki, kj]
                                MU_s = mu[ki, kj]
                                MU_n = 0.5 * (mu[ki, kj] + mu[ki + 1, kj])

                            if kj == 0 and ki > 0 and ki < self.elements_axial - 1:
                                MU_e = 0.5 * (mu[ki, kj] + mu[ki, kj + 1])
                                MU_w = mu[ki, kj]
                                MU_s = 0.5 * (mu[ki, kj] + mu[ki - 1, kj])
                                MU_n = 0.5 * (mu[ki, kj] + mu[ki + 1, kj])

                            if kj == 0 and ki == self.elements_axial - 1:
                                MU_e = 0.5 * (mu[ki, kj] + mu[ki, kj + 1])
                                MU_w = mu[ki, kj]
                                MU_s = 0.5 * (mu[ki, kj] + mu[ki - 1, kj])
                                MU_n = mu[ki, kj]

                            if (
                                ki == 0
                                and kj > 0
                                and kj < self.elements_circumferential - 1
                            ):
                                MU_e = 0.5 * (mu[ki, kj] + mu[ki, kj + 1])
                                MU_w = 0.5 * (mu[ki, kj] + mu[ki, kj - 1])
                                MU_s = mu[ki, kj]
                                MU_n = 0.5 * (mu[ki, kj] + mu[ki + 1, kj])

                            if (
                                kj > 0
                                and kj < self.elements_circumferential - 1
                                and ki > 0
                                and ki < self.elements_axial - 1
                            ):
                                MU_e = 0.5 * (mu[ki, kj] + mu[ki, kj + 1])
                                MU_w = 0.5 * (mu[ki, kj] + mu[ki, kj - 1])
                                MU_s = 0.5 * (mu[ki, kj] + mu[ki - 1, kj])
                                MU_n = 0.5 * (mu[ki, kj] + mu[ki + 1, kj])

                            if (
                                ki == self.elements_axial - 1
                                and kj > 0
                                and kj < self.elements_circumferential - 1
                            ):
                                MU_e = 0.5 * (mu[ki, kj] + mu[ki, kj + 1])
                                MU_w = 0.5 * (mu[ki, kj] + mu[ki, kj - 1])
                                MU_s = 0.5 * (mu[ki, kj] + mu[ki - 1, kj])
                                MU_n = mu[ki, kj]

                            if ki == 0 and kj == self.elements_circumferential - 1:
                                MU_e = mu[ki, kj]
                                MU_w = 0.5 * (mu[ki, kj] + mu[ki, kj - 1])
                                MU_s = mu[ki, kj]
                                MU_n = 0.5 * (mu[ki, kj] + mu[ki + 1, kj])

                            if (
                                kj == self.elements_circumferential - 1
                                and ki > 0
                                and ki < self.elements_axial - 1
                            ):
                                MU_e = mu[ki, kj]
                                MU_w = 0.5 * (mu[ki, kj] + mu[ki, kj - 1])
                                MU_s = 0.5 * (mu[ki, kj] + mu[ki - 1, kj])
                                MU_n = 0.5 * (mu[ki, kj] + mu[ki + 1, kj])

                            if (
                                kj == self.elements_circumferential - 1
                                and ki == self.elements_axial - 1
                            ):
                                MU_e = mu[ki, kj]
                                MU_w = 0.5 * (mu[ki, kj] + mu[ki, kj - 1])
                                MU_s = 0.5 * (mu[ki, kj] + mu[ki - 1, kj])
                                MU_n = mu[ki, kj]

                            CE = (self.dZ * he**3) / (
                                12 * MU_e[n_p] * self.dY * self.betha_s**2
                            )
                            CW = (self.dZ * hw**3) / (
                                12 * MU_w[n_p] * self.dY * self.betha_s**2
                            )
                            CN = (self.dY * (self.journal_radius**2) * hn**3) / (
                                12 * MU_n[n_p] * self.dZ * self.axial_length**2
                            )
                            CS = (self.dY * (self.journal_radius**2) * hs**3) / (
                                12 * MU_s[n_p] * self.dZ * self.axial_length**2
                            )
                            CP = -(CE + CW + CN + CS)

                            B = (self.dZ / (2 * self.betha_s)) * (he - hw) - (
                                (self.Ypt * np.cos(jj) + self.Xpt * np.sin(jj))
                                * self.dy
                                * self.dZ
                            )

                            k = k + 1
                            b[k - 1, 0] = B

                            if ki == 0 and kj == 0:
                                Mat_coef[k - 1, k - 1] = CP - CS - CW
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = CN

                            elif kj == 0 and ki > 0 and ki < self.elements_axial - 1:
                                Mat_coef[k - 1, k - 1] = CP - CW
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = CS
                                Mat_coef[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = CN

                            elif kj == 0 and ki == self.elements_axial - 1:
                                Mat_coef[k - 1, k - 1] = CP - CN - CW
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = CS

                            elif ki == 0 and kj > 0 and kj < self.n_y - 1:
                                Mat_coef[k - 1, k - 1] = CP - CS
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = CN

                            elif (
                                ki > 0
                                and ki < self.elements_axial - 1
                                and kj > 0
                                and kj < self.n_y - 1
                            ):
                                Mat_coef[k - 1, k - 1] = CP
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = CS
                                Mat_coef[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = CN
                                Mat_coef[k - 1, k] = CE

                            elif (
                                ki == self.elements_axial - 1
                                and kj > 0
                                and kj < self.n_y - 1
                            ):
                                Mat_coef[k - 1, k - 1] = CP - CN
                                Mat_coef[k - 1, k] = CE
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = CS

                            elif ki == 0 and kj == self.n_y - 1:
                                Mat_coef[k - 1, k - 1] = CP - CE - CS
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = CN

                            elif (
                                kj == self.n_y - 1
                                and ki > 0
                                and ki < self.elements_axial - 1
                            ):
                                Mat_coef[k - 1, k - 1] = CP - CE
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = CS
                                Mat_coef[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = CN

                            elif ki == self.elements_axial - 1 and kj == self.n_y - 1:
                                Mat_coef[k - 1, k - 1] = CP - CE - CN
                                Mat_coef[k - 1, k - 2] = CW
                                Mat_coef[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = CS

                            kj = kj + 1

                        kj = 0
                        ki = ki + 1

                    # Solution of pressure field end

                    p = np.linalg.solve(Mat_coef, b)
                    cont = 0

                    for i in np.arange(self.elements_axial):
                        for j in np.arange(self.elements_circumferential):

                            self.P[i, j, n_p] = p[cont]
                            cont = cont + 1

                            if self.P[i, j, n_p] < 0:
                                self.P[i, j, n_p] = 0

                    # Dimensional pressure fied

                    self.Pdim = (
                        self.P
                        * self.reference_viscosity
                        * self.speed
                        * (self.journal_radius**2)
                    ) / (self.radial_clearance**2)

                    ki = 0
                    kj = 0
                    k = 0

                    # Solution of temperature field initialization

                    for ii in np.arange(
                        (self.Z_I + 0.5 * self.dZ), (self.Z_F), self.dZ
                    ):
                        for jj in np.arange(
                            self.thetaI[n_p] + (self.dtheta / 2),
                            self.thetaF[n_p],
                            self.dtheta,
                        ):

                            # Pressure gradients

                            if kj == 0 and ki == 0:
                                dPdy[ki, kj, n_p] = (self.P[ki, kj + 1, n_p] - 0) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (self.P[ki + 1, kj, n_p] - 0) / (
                                    2 * self.dZ
                                )

                            if kj == 0 and ki > 0 and ki < self.elements_axial - 1:
                                dPdy[ki, kj, n_p] = (self.P[ki, kj + 1, n_p] - 0) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (
                                    self.P[ki + 1, kj, n_p] - self.P[ki - 1, kj, n_p]
                                ) / (2 * self.dZ)

                            if kj == 0 and ki == self.elements_axial - 1:
                                dPdy[ki, kj, n_p] = (self.P[ki, kj + 1, n_p] - 0) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (0 - self.P[ki - 1, kj, n_p]) / (
                                    2 * self.dZ
                                )

                            if (
                                ki == 0
                                and kj > 0
                                and kj < self.elements_circumferential - 1
                            ):
                                dPdy[ki, kj, n_p] = (
                                    self.P[ki, kj + 1, n_p] - self.P[ki, kj - 1, n_p]
                                ) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (self.P[ki + 1, kj, n_p] - 0) / (
                                    2 * self.dZ
                                )

                            if (
                                kj > 0
                                and kj < self.elements_circumferential - 1
                                and ki > 0
                                and ki < self.elements_axial - 1
                            ):
                                dPdy[ki, kj, n_p] = (
                                    self.P[ki, kj + 1, n_p] - self.P[ki, kj - 1, n_p]
                                ) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (
                                    self.P[ki + 1, kj, n_p] - self.P[ki - 1, kj, n_p]
                                ) / (2 * self.dZ)

                            if (
                                ki == self.elements_axial - 1
                                and kj > 0
                                and kj < self.elements_circumferential - 1
                            ):
                                dPdy[ki, kj, n_p] = (
                                    self.P[ki, kj + 1, n_p] - self.P[ki, kj - 1, n_p]
                                ) / (2 * self.dY)
                                dPdz[ki, kj, n_p] = (0 - self.P[ki - 1, kj, n_p]) / (
                                    2 * self.dZ
                                )

                            if ki == 0 and kj == self.elements_circumferential - 1:
                                dPdy[ki, kj, n_p] = (0 - self.P[ki, kj - 1, n_p]) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (self.P[ki + 1, kj, n_p] - 0) / (
                                    2 * self.dZ
                                )

                            if (
                                kj == self.elements_circumferential - 1
                                and ki > 0
                                and ki < self.elements_axial - 1
                            ):
                                dPdy[ki, kj, n_p] = (0 - self.P[ki, kj - 1, n_p]) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (
                                    self.P[ki + 1, kj, n_p] - self.P[ki - 1, kj, n_p]
                                ) / (2 * self.dZ)

                            if (
                                kj == self.elements_circumferential - 1
                                and ki == self.elements_axial - 1
                            ):
                                dPdy[ki, kj, n_p] = (0 - self.P[ki, kj - 1, n_p]) / (
                                    2 * self.dY
                                )
                                dPdz[ki, kj, n_p] = (0 - self.P[ki - 1, kj, n_p]) / (
                                    2 * self.dZ
                                )

                            HP = 1 - self.X * np.cos(jj) - self.Y * np.sin(jj)
                            hpt = -self.Ypt * np.cos(jj) + self.Xpt * np.sin(jj)

                            mu_p = mu[ki, kj, n_p]

                            Reyn[ki, kj, n_p] = (
                                self.rho
                                * self.speed
                                * self.journal_radius
                                * (HP / self.axial_length)
                                * self.radial_clearance
                                / (self.reference_viscosity)
                            )

                            if Reyn[ki, kj, n_p] <= 500:

                                self.delta_turb = 0

                            elif Reyn[ki, kj, n_p] > 400 and Reyn[ki, kj, n_p] <= 1000:

                                self.delta_turb = 1 - (
                                    (1000 - Reyn[ki, kj, n_p]) / 500
                                ) ** (1 / 8)

                            elif Reyn[ki, kj, n_p] > 1000:

                                self.delta_turb = 1

                            dudy = ((HP / mu_turb[ki, kj, n_p]) * dPdy[ki, kj, n_p]) - (
                                self.speed / HP
                            )

                            dwdy = (HP / mu_turb[ki, kj, n_p]) * dPdz[ki, kj, n_p]

                            tal = mu_turb[ki, kj, n_p] * np.sqrt(
                                (dudy**2) + (dwdy**2)
                            )

                            x_wall = (
                                (HP * self.radial_clearance * 2)
                                / (
                                    self.reference_viscosity
                                    * mu_turb[ki, kj, n_p]
                                    / self.rho
                                )
                            ) * ((abs(tal) / self.rho) ** 0.5)

                            emv = 0.4 * (x_wall - (10.7 * np.tanh(x_wall / 10.7)))

                            mu_turb[ki, kj, n_p] = mu_p * (1 + (self.delta_turb * emv))

                            mi_t = mu_turb[ki, kj, n_p]

                            AE = -(self.k_t * HP * self.dZ) / (
                                self.rho
                                * self.Cp
                                * self.speed
                                * ((self.betha_s * self.journal_radius) ** 2)
                                * self.dY
                            )
                            AW = (
                                (
                                    ((HP**3) * dPdy[ki, kj, n_p] * self.dZ)
                                    / (12 * mi_t * (self.betha_s**2))
                                )
                                - ((HP) * self.dZ / (2 * self.betha_s))
                                - (
                                    (self.k_t * HP * self.dZ)
                                    / (
                                        self.rho
                                        * self.Cp
                                        * self.speed
                                        * ((self.betha_s * self.journal_radius) ** 2)
                                        * self.dY
                                    )
                                )
                            )

                            AN = -(
                                (
                                    (self.journal_radius**2)
                                    * (HP**3)
                                    * (dPdz[ki, kj, n_p] * self.dY)
                                )
                                / (2 * 12 * (self.axial_length**2) * mi_t)
                            ) - (
                                (self.k_t * HP * self.dY)
                                / (
                                    self.rho
                                    * self.Cp
                                    * self.speed
                                    * (self.axial_length**2)
                                    * self.dZ
                                )
                            )

                            AS = (
                                (
                                    (self.journal_radius**2)
                                    * (HP**3)
                                    * (dPdz[ki, kj, n_p] * self.dY)
                                )
                                / (2 * 12 * (self.axial_length**2) * mi_t)
                            ) - (
                                (self.k_t * HP * self.dY)
                                / (
                                    self.rho
                                    * self.Cp
                                    * self.speed
                                    * (self.axial_length**2)
                                    * self.dZ
                                )
                            )

                            AP = -(AE + AW + AN + AS)

                            auxb_T = (self.speed * self.reference_viscosity) / (
                                self.rho
                                * self.Cp
                                * self.reference_temperature
                                * self.radial_clearance
                            )
                            b_TG = (
                                self.reference_viscosity
                                * self.speed
                                * (self.journal_radius**2)
                                * self.dY
                                * self.dZ
                                * self.P[ki, kj, n_p]
                                * hpt
                            ) / (
                                self.rho
                                * self.Cp
                                * self.reference_temperature
                                * (self.radial_clearance**2)
                            )
                            b_TH = (
                                self.speed
                                * self.reference_viscosity
                                * (hpt**2)
                                * 4
                                * mi_t
                                * self.dY
                                * self.dZ
                            ) / (
                                self.rho * self.Cp * self.reference_temperature * 3 * HP
                            )
                            b_TI = (
                                auxb_T
                                * (
                                    mi_t
                                    * (self.journal_radius**2)
                                    * self.dY
                                    * self.dZ
                                )
                                / (HP * self.radial_clearance)
                            )
                            b_TJ = (
                                auxb_T
                                * (
                                    (self.journal_radius**2)
                                    * (HP**3)
                                    * (dPdy[ki, kj, n_p] ** 2)
                                    * self.dY
                                    * self.dZ
                                )
                                / (
                                    12
                                    * self.radial_clearance
                                    * (self.betha_s**2)
                                    * mi_t
                                )
                            )
                            b_TK = (
                                auxb_T
                                * (
                                    (self.journal_radius**4)
                                    * (HP**3)
                                    * (dPdz[ki, kj, n_p] ** 2)
                                    * self.dY
                                    * self.dZ
                                )
                                / (
                                    12
                                    * self.radial_clearance
                                    * (self.axial_length**2)
                                    * mi_t
                                )
                            )

                            B_T = b_TG + b_TH + b_TI + b_TJ + b_TK

                            k = k + 1

                            b_T[k - 1, 0] = B_T

                            if ki == 0 and kj == 0:
                                Mat_coef_T[k - 1, k - 1] = AP + AS - AW
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = AN
                                b_T[k - 1, 0] = b_T[k - 1, 0] - 2 * AW * (
                                    T_ref / self.reference_temperature
                                )

                            elif kj == 0 and ki > 0 and ki < self.elements_axial - 1:
                                Mat_coef_T[k - 1, k - 1] = AP - AW
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = AS
                                Mat_coef_T[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = AN
                                b_T[k - 1, 0] = b_T[k - 1, 0] - 2 * AW * (
                                    T_ref / self.reference_temperature
                                )

                            elif kj == 0 and ki == self.elements_axial - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AN - AW
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = AS
                                b_T[k - 1, 0] = b_T[k - 1, 0] - 2 * AW * (
                                    T_ref / self.reference_temperature
                                )

                            elif ki == 0 and kj > 0 and kj < self.n_y - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AS
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = AN

                            elif (
                                ki > 0
                                and ki < self.elements_axial - 1
                                and kj > 0
                                and kj < self.n_y - 1
                            ):
                                Mat_coef_T[k - 1, k - 1] = AP
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = AS
                                Mat_coef_T[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = AN
                                Mat_coef_T[k - 1, k] = AE

                            elif (
                                ki == self.elements_axial - 1
                                and kj > 0
                                and kj < self.n_y - 1
                            ):
                                Mat_coef_T[k - 1, k - 1] = AP + AN
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = AS

                            elif ki == 0 and kj == self.n_y - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AE + AS
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = AN

                            elif (
                                kj == self.n_y - 1
                                and ki > 0
                                and ki < self.elements_axial - 1
                            ):
                                Mat_coef_T[k - 1, k - 1] = AP + AE
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = AS
                                Mat_coef_T[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = AN

                            elif ki == self.elements_axial - 1 and kj == self.n_y - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AE + AN
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = AS

                            kj = kj + 1

                        kj = 0
                        ki = ki + 1

                    # Solution of temperature field end

                    t = np.linalg.solve(Mat_coef_T, b_T)
                    cont = 0

                    for i in np.arange(self.elements_axial):
                        for j in np.arange(self.elements_circumferential):

                            T_new[i, j, n_p] = t[cont]
                            cont = cont + 1

                    Tdim = T_new * self.reference_temperature

                    T_end = np.sum(Tdim[:, -1, n_p]) / self.elements_axial

                    T_mist[n_p] = (
                        self.fat_mixt[n_p] * self.reference_temperature
                        + (1 - self.fat_mixt[n_p]) * T_end
                    )

                    for i in np.arange(self.elements_axial):
                        for j in np.arange(self.elements_circumferential):

                            mu_new[i, j, n_p] = (
                                self.a * (Tdim[i, j, n_p]) ** self.b
                            ) / self.reference_viscosity

        PP = np.zeros(
            ((self.elements_axial), (self.n_pad * self.elements_circumferential))
        )

        i = 0
        for i in range(self.elements_axial):

            PP[i] = self.Pdim[i, :, :].ravel("F")

        Ytheta = np.array(Ytheta)
        Ytheta = Ytheta.flatten()

        auxF = np.zeros((2, len(Ytheta)))

        auxF[0, :] = np.cos(Ytheta)
        auxF[1, :] = np.sin(Ytheta)

        dA = self.dy * self.dz

        auxP = PP * dA

        vector_auxF_x = auxF[0, :]
        vector_auxF_y = auxF[1, :]

        auxFx = auxP * vector_auxF_x
        auxFy = auxP * vector_auxF_y

        fxj = -np.sum(auxFx)
        fyj = -np.sum(auxFy)

        Fhx = fxj
        Fhy = fyj
        self.Fhx = Fhx
        self.Fhy = Fhy
        return Fhx, Fhy

    def run(self):
        """This method runs the optimization to find the equilibrium position of
        the rotor's center.


        """
        args = self.print_progress
        t1 = time.time()
        res = minimize(
            self._score,
            self.initial_guess,
            args,
            method="Nelder-Mead",
            tol=10e-3,
            options={"maxiter": 1000},
        )
        self.equilibrium_pos = res.x
        t2 = time.time()

        if self.print_result:
            print(res)

        if self.print_time:
            print(f"Time Spent: {t2-t1} seconds")

    def _interpol(self, T_muI, T_muF, mu_I, mu_F):
        """
        This method is used to create a relationship between viscosity and
        temperature.

        Parameters
        ----------
        T_muI: float
            Reference temperature 1.
        T_muF: float
            Reference temperature 2.
        mu_I: float
            Viscosity at temperature 1.
        mu_F: float
            Viscosity at temperature 1.

        Returns
        -------
        a,b: Float
            Coeficients of the curve viscosity vs temperature.
        """

        def viscosity(x, a, b):
            return a * (x**b)

        xdata = [T_muI, T_muF]  # changed boundary conditions to avoid division by ]
        ydata = [mu_I, mu_F]

        popt, pcov = curve_fit(viscosity, xdata, ydata, p0=(6.0, -1.0))
        a, b = popt

        return a, b

    def coefficients(self):
        """Calculates the dynamic coefficients of stiffness "k" and damping "c".
        Basic reference is found at :cite:t:`lund1978`
        Parameters
        ----------

        Returns
        -------
        coefs : tuple
            Bearing stiffness and damping coefficients.
            Its shape is: ((kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy))

        References
        ----------
        .. bibliography::
            :filter: docname in docnames
        """

        if self.equilibrium_pos is None:
            self.run()
            self.coefficients()
        else:
            if self.method == "lund":
                k, c = self._lund_method()
            elif self.method == "perturbation":
                k, c = self._pertubation_method()

            if self.show_coef:
                print(f"kxx = {k[0]}")
                print(f"kxy = {k[1]}")
                print(f"kyx = {k[2]}")
                print(f"kyy = {k[3]}")

                print(f"cxx = {c[0]}")
                print(f"cxy = {c[1]}")
                print(f"cyx = {c[2]}")
                print(f"cyy = {c[3]}")

            coefs = (k, c)

            return coefs

    def _pertubation_method(self):
        """In this method the formulation is based in application of virtual
        displacements and speeds on the rotor from its equilibrium position to
        determine the bearing stiffness and damping coefficients.

        """

        xeq = (
            self.equilibrium_pos[0]
            * self.radial_clearance
            * np.cos(self.equilibrium_pos[1])
        )
        yeq = (
            self.equilibrium_pos[0]
            * self.radial_clearance
            * np.sin(self.equilibrium_pos[1])
        )

        dE = 0.001
        epix = np.abs(dE * self.radial_clearance * np.cos(self.equilibrium_pos[1]))
        epiy = np.abs(dE * self.radial_clearance * np.sin(self.equilibrium_pos[1]))

        Va = self.speed * (self.journal_radius)
        epixpt = 0.000001 * np.abs(Va * np.sin(self.equilibrium_pos[1]))
        epiypt = 0.000001 * np.abs(Va * np.cos(self.equilibrium_pos[1]))

        Auinitial_guess1 = self._forces(xeq + epix, yeq, 0, 0)
        Auinitial_guess2 = self._forces(xeq - epix, yeq, 0, 0)
        Auinitial_guess3 = self._forces(xeq, yeq + epiy, 0, 0)
        Auinitial_guess4 = self._forces(xeq, yeq - epiy, 0, 0)

        Auinitial_guess5 = self._forces(xeq, yeq, epixpt, 0)
        Auinitial_guess6 = self._forces(xeq, yeq, -epixpt, 0)
        Auinitial_guess7 = self._forces(xeq, yeq, 0, epiypt)
        Auinitial_guess8 = self._forces(xeq, yeq, 0, -epiypt)

        Kxx = -self.sommerfeld(Auinitial_guess1[0], Auinitial_guess2[1]) * (
            (Auinitial_guess1[0] - Auinitial_guess2[0]) / (epix / self.radial_clearance)
        )
        Kxy = -self.sommerfeld(Auinitial_guess3[0], Auinitial_guess4[1]) * (
            (Auinitial_guess3[0] - Auinitial_guess4[0]) / (epiy / self.radial_clearance)
        )
        Kyx = -self.sommerfeld(Auinitial_guess1[1], Auinitial_guess2[1]) * (
            (Auinitial_guess1[1] - Auinitial_guess2[1]) / (epix / self.radial_clearance)
        )
        Kyy = -self.sommerfeld(Auinitial_guess3[1], Auinitial_guess4[1]) * (
            (Auinitial_guess3[1] - Auinitial_guess4[1]) / (epiy / self.radial_clearance)
        )

        Cxx = -self.sommerfeld(Auinitial_guess5[0], Auinitial_guess6[0]) * (
            (Auinitial_guess6[0] - Auinitial_guess5[0])
            / (epixpt / self.radial_clearance / self.speed)
        )
        Cxy = -self.sommerfeld(Auinitial_guess7[0], Auinitial_guess8[0]) * (
            (Auinitial_guess8[0] - Auinitial_guess7[0])
            / (epiypt / self.radial_clearance / self.speed)
        )
        Cyx = -self.sommerfeld(Auinitial_guess5[1], Auinitial_guess6[1]) * (
            (Auinitial_guess6[1] - Auinitial_guess5[1])
            / (epixpt / self.radial_clearance / self.speed)
        )
        Cyy = -self.sommerfeld(Auinitial_guess7[1], Auinitial_guess8[1]) * (
            (Auinitial_guess8[1] - Auinitial_guess7[1])
            / (epiypt / self.radial_clearance / self.speed)
        )

        kxx = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / self.radial_clearance
        ) * Kxx
        kxy = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / self.radial_clearance
        ) * Kxy
        kyx = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / self.radial_clearance
        ) * Kyx
        kyy = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / self.radial_clearance
        ) * Kyy

        cxx = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / (self.radial_clearance * self.speed)
        ) * Cxx
        cxy = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / (self.radial_clearance * self.speed)
        ) * Cxy
        cyx = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / (self.radial_clearance * self.speed)
        ) * Cyx
        cyy = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / (self.radial_clearance * self.speed)
        ) * Cyy

        return (kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy)

    def _lund_method(self):
        """In this method a small amplitude whirl of the journal center (a first
        order perturbation solution) is aplied. The four stiffness coefficients,
        and the four damping coefficients is obtained by integration of the pressure
        field.


        """

        p = self.P

        initial_guess = self.equilibrium_pos

        dZ = 1 / self.elements_axial

        Z1 = 0
        Z2 = 1
        Z = np.arange(Z1 + 0.5 * dZ, Z2, dZ)
        Zdim = Z * self.axial_length

        Ytheta = np.zeros((self.n_pad, self.elements_circumferential))

        # Dimensionless
        xr = initial_guess[0] * self.radial_clearance * np.cos(initial_guess[1])
        yr = initial_guess[0] * self.radial_clearance * np.sin(initial_guess[1])
        Y = yr / self.radial_clearance
        X = xr / self.radial_clearance

        nk = (self.elements_axial) * (self.elements_circumferential)

        gamma = 0.001

        wp = gamma * self.speed

        Mat_coef = np.zeros((nk, nk))

        bX = np.zeros((nk, 1)).astype(complex)

        bY = np.zeros((nk, 1)).astype(complex)

        hX = np.zeros((self.n_pad, self.elements_circumferential))

        hY = np.zeros((self.n_pad, self.elements_circumferential))

        PX = np.zeros(
            (self.elements_axial, self.elements_circumferential, self.n_pad)
        ).astype(complex)

        PY = np.zeros(
            (self.elements_axial, self.elements_circumferential, self.n_pad)
        ).astype(complex)

        H = np.zeros((2, 2)).astype(complex)

        n_p = 0

        for n_p in np.arange(self.n_pad):

            Ytheta[n_p, :] = np.arange(
                self.thetaI[n_p] + (self.dtheta / 2), self.thetaF[n_p], self.dtheta
            )

            ki = 0
            kj = 0

            k = 0

            for ii in np.arange((self.Z_I + 0.5 * self.dZ), self.Z_F, self.dZ):
                for jj in np.arange(
                    self.thetaI[n_p] + (self.dtheta / 2), self.thetaF[n_p], self.dtheta
                ):

                    hP = 1 - X * np.cos(jj) - Y * np.sin(jj)
                    he = (
                        1
                        - X * np.cos(jj + 0.5 * self.dtheta)
                        - self.Y * np.sin(jj + 0.5 * self.dtheta)
                    )
                    hw = (
                        1
                        - X * np.cos(jj - 0.5 * self.dtheta)
                        - self.Y * np.sin(jj - 0.5 * self.dtheta)
                    )
                    hn = hP
                    hs = hn

                    hXP = -np.cos(jj)
                    hXe = -np.cos(jj + 0.5 * self.dtheta)
                    hXw = -np.cos(jj - 0.5 * self.dtheta)
                    hXn = hXP
                    hXs = hXn

                    hYP = -np.sin(jj)
                    hYe = -np.sin(jj + 0.5 * self.dtheta)
                    hYw = -np.sin(jj - 0.5 * self.dtheta)
                    hYn = hYP
                    hYs = hYn

                    if ki == 0:
                        hX[n_p, kj] = hXP
                        hY[n_p, kj] = hYP

                    if kj == 0 and ki == 0:
                        MU_e = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj + 1, n_p]
                        )
                        MU_w = self.mu_l[ki, kj, n_p]
                        MU_s = self.mu_l[ki, kj, n_p]
                        MU_n = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki + 1, kj, n_p]
                        )

                        pE = p[ki, kj + 1, n_p]
                        pW = -p[ki, kj, n_p]
                        pN = p[ki + 1, kj, n_p]
                        pS = -p[ki, kj, n_p]

                    if kj == 0 and ki > 0 and ki < self.elements_axial - 1:
                        MU_e = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj + 1, n_p]
                        )
                        MU_w = self.mu_l[ki, kj, n_p]
                        MU_s = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki - 1, kj, n_p]
                        )
                        MU_n = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki + 1, kj, n_p]
                        )

                        pE = p[ki, kj + 1, n_p]
                        pW = -p[ki, kj, n_p]
                        pN = p[ki + 1, kj, n_p]
                        pS = p[ki - 1, kj, n_p]

                    if kj == 0 and ki == self.elements_axial - 1:
                        MU_e = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj + 1, n_p]
                        )
                        MU_w = self.mu_l[ki, kj, n_p]
                        MU_s = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki - 1, kj, n_p]
                        )
                        MU_n = self.mu_l[ki, kj, n_p]

                        pE = p[ki, kj + 1, n_p]
                        pW = -p[ki, kj, n_p]
                        pN = -p[ki, kj, n_p]
                        pS = p[ki - 1, kj, n_p]

                    if ki == 0 and kj > 0 and kj < self.elements_circumferential - 1:
                        MU_e = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj + 1, n_p]
                        )
                        MU_w = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj - 1, n_p]
                        )
                        MU_s = self.mu_l[ki, kj, n_p]
                        MU_n = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki + 1, kj, n_p]
                        )

                        pE = p[ki, kj + 1, n_p]
                        pW = p[ki, kj - 1, n_p]
                        pN = p[ki + 1, kj, n_p]
                        pS = -p[ki, kj, n_p]

                    if (
                        kj > 0
                        and kj < self.elements_circumferential - 1
                        and ki > 0
                        and ki < self.elements_axial - 1
                    ):
                        MU_e = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj + 1, n_p]
                        )
                        MU_w = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj - 1, n_p]
                        )
                        MU_s = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki - 1, kj, n_p]
                        )
                        MU_n = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki + 1, kj, n_p]
                        )

                        pE = p[ki, kj + 1, n_p]
                        pW = p[ki, kj - 1, n_p]
                        pN = p[ki + 1, kj, n_p]
                        pS = p[ki - 1, kj, n_p]

                    if (
                        ki == self.elements_axial - 1
                        and kj > 0
                        and kj < self.elements_circumferential - 1
                    ):
                        MU_e = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj + 1, n_p]
                        )
                        MU_w = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj - 1, n_p]
                        )
                        MU_s = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki - 1, kj, n_p]
                        )
                        MU_n = self.mu_l[ki, kj, n_p]

                        pE = p[ki, kj + 1, n_p]
                        pW = p[ki, kj - 1, n_p]
                        pN = -p[ki, kj, n_p]
                        pS = p[ki - 1, kj, n_p]

                    if ki == 0 and kj == self.elements_circumferential - 1:
                        MU_e = self.mu_l[ki, kj, n_p]
                        MU_w = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj - 1, n_p]
                        )
                        MU_s = self.mu_l[ki, kj, n_p]
                        MU_n = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki + 1, kj, n_p]
                        )

                        pE = -p[ki, kj, n_p]
                        pW = p[ki, kj - 1, n_p]
                        pN = p[ki + 1, kj, n_p]
                        pS = -p[ki, kj, n_p]

                    if (
                        kj == self.elements_circumferential - 1
                        and ki > 0
                        and ki < self.elements_axial - 1
                    ):
                        MU_e = self.mu_l[ki, kj, n_p]
                        MU_w = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj - 1, n_p]
                        )
                        MU_s = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki - 1, kj, n_p]
                        )
                        MU_n = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki + 1, kj, n_p]
                        )

                        pE = -p[ki, kj, n_p]
                        pW = p[ki, kj - 1, n_p]
                        pN = p[ki + 1, kj, n_p]
                        pS = p[ki - 1, kj, n_p]

                    if (
                        kj == self.elements_circumferential - 1
                        and ki == self.elements_axial - 1
                    ):
                        MU_e = self.mu_l[ki, kj, n_p]
                        MU_w = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj - 1, n_p]
                        )
                        MU_s = 0.5 * (
                            self.mu_l[ki, kj, n_p] + self.mu_l[ki - 1, kj, n_p]
                        )
                        MU_n = self.mu_l[ki, kj, n_p]

                        pE = -p[ki, kj, n_p]
                        pW = p[ki, kj - 1, n_p]
                        pN = -p[ki, kj, n_p]
                        pS = p[ki - 1, kj, n_p]

                    pP = p[ki, kj, n_p]

                    CE = (self.dZ * he**3) / (12 * MU_e * self.dY * self.betha_s**2)
                    CW = (self.dZ * hw**3) / (12 * MU_w * self.dY * self.betha_s**2)
                    CN = (self.dY * (self.journal_radius**2) * hn**3) / (
                        12 * MU_n * self.dZ * self.axial_length**2
                    )
                    CS = (self.dY * (self.journal_radius**2) * hs**3) / (
                        12 * MU_s * self.dZ * self.axial_length**2
                    )

                    CP = -(CE + CW + CN + CS)

                    BXE = -(self.dZ / (self.dY * self.betha_s**2)) * (
                        (3 * he**2 * hXe) / (12 * MU_e)
                    )

                    BYE = -(self.dZ / (self.dY * self.betha_s**2)) * (
                        (3 * he**2 * hYe) / (12 * MU_e)
                    )

                    BXW = -(self.dZ / (self.dY * self.betha_s**2)) * (
                        (3 * hw**2 * hXw) / (12 * MU_w)
                    )

                    BYW = -(self.dZ / (self.dY * self.betha_s**2)) * (
                        (3 * hw**2 * hYw) / (12 * MU_w)
                    )

                    BXN = -(
                        (self.journal_radius**2)
                        * self.dY
                        / (self.dZ * self.axial_length**2)
                    ) * ((3 * hn**2 * hXn) / (12 * MU_n))

                    BYN = -(
                        (self.journal_radius**2)
                        * self.dY
                        / (self.dZ * self.axial_length**2)
                    ) * ((3 * hn**2 * hYn) / (12 * MU_n))

                    BXS = -(
                        (self.journal_radius**2)
                        * self.dY
                        / (self.dZ * self.axial_length**2)
                    ) * ((3 * hs**2 * hXs) / (12 * MU_s))

                    BYS = -(
                        (self.journal_radius**2)
                        * self.dY
                        / (self.dZ * self.axial_length**2)
                    ) * ((3 * hs**2 * hYs) / (12 * MU_s))

                    BXP = -(BXE + BXW + BXN + BXS)

                    BYP = -(BYE + BYW + BYN + BYS)

                    BX = (
                        (self.dZ / (2 * self.betha_s)) * (hXe - hXw)
                        + (self.dY * self.dZ * 1j * gamma * hXP)
                        + BXE * pE
                        + BXW * pW
                        + BXN * pN
                        + BXS * pS
                        + BXP * pP
                    )

                    BY = (
                        (self.dZ / (2 * self.betha_s)) * (hYe - hYw)
                        + (self.dY * self.dZ * 1j * gamma * hYP)
                        + BYE * pE
                        + BYW * pW
                        + BYN * pN
                        + BYS * pS
                        + BYP * pP
                    )

                    k = k + 1
                    bX[k - 1, 0] = BX
                    bY[k - 1, 0] = BY

                    if ki == 0 and kj == 0:
                        Mat_coef[k - 1, k - 1] = CP - CS - CW
                        Mat_coef[k - 1, k] = CE
                        Mat_coef[k - 1, k + self.elements_circumferential - 1] = CN

                    elif kj == 0 and ki > 0 and ki < self.elements_axial - 1:
                        Mat_coef[k - 1, k - 1] = CP - CW
                        Mat_coef[k - 1, k] = CE
                        Mat_coef[k - 1, k - self.elements_circumferential - 1] = CS
                        Mat_coef[k - 1, k + self.elements_circumferential - 1] = CN

                    elif kj == 0 and ki == self.elements_axial - 1:
                        Mat_coef[k - 1, k - 1] = CP - CN - CW
                        Mat_coef[k - 1, k] = CE
                        Mat_coef[k - 1, k - self.elements_circumferential - 1] = CS

                    elif ki == 0 and kj > 0 and kj < self.elements_circumferential - 1:
                        Mat_coef[k - 1, k - 1] = CP - CS
                        Mat_coef[k - 1, k] = CE
                        Mat_coef[k - 1, k - 2] = CW
                        Mat_coef[k - 1, k + self.elements_circumferential - 1] = CN

                    if (
                        ki > 0
                        and ki < self.elements_axial - 1
                        and kj > 0
                        and kj < self.elements_circumferential - 1
                    ):
                        Mat_coef[k - 1, k - 1] = CP
                        Mat_coef[k - 1, k - 2] = CW
                        Mat_coef[k - 1, k - self.elements_circumferential - 1] = CS
                        Mat_coef[k - 1, k + self.elements_circumferential - 1] = CN
                        Mat_coef[k - 1, k] = CE

                    elif (
                        ki == self.elements_axial - 1
                        and kj > 0
                        and kj < self.elements_circumferential - 1
                    ):
                        Mat_coef[k - 1, k - 1] = CP - CN
                        Mat_coef[k - 1, k] = CE
                        Mat_coef[k - 1, k - 2] = CW
                        Mat_coef[k - 1, k - self.elements_circumferential - 1] = CS

                    elif ki == 0 and kj == self.elements_circumferential - 1:
                        Mat_coef[k - 1, k - 1] = CP - CE - CS
                        Mat_coef[k - 1, k - 2] = CW
                        Mat_coef[k - 1, k + self.elements_circumferential - 1] = CN

                    elif (
                        kj == self.elements_circumferential - 1
                        and ki > 0
                        and ki < self.elements_axial - 1
                    ):
                        Mat_coef[k - 1, k - 1] = CP - CE
                        Mat_coef[k - 1, k - 2] = CW
                        Mat_coef[k - 1, k - self.elements_circumferential - 1] = CS
                        Mat_coef[k - 1, k + self.elements_circumferential - 1] = CN

                    elif (
                        ki == self.elements_axial - 1
                        and kj == self.elements_circumferential - 1
                    ):
                        Mat_coef[k - 1, k - 1] = CP - CE - CN
                        Mat_coef[k - 1, k - 2] = CW
                        Mat_coef[k - 1, k - self.elements_circumferential - 1] = CS

                    kj = kj + 1

                kj = 0
                ki = ki + 1

                #    ###################### Solution of pressure field #######################

            pX = np.linalg.solve(Mat_coef, bX)

            pY = np.linalg.solve(Mat_coef, bY)

            cont = 0

            for i in np.arange(self.elements_axial):
                for j in np.arange(self.elements_circumferential):

                    PX[i, j, n_p] = pX[cont]
                    PY[i, j, n_p] = pY[cont]
                    cont = cont + 1

        PPlotX = np.zeros(
            (self.elements_axial, self.elements_circumferential * self.n_pad)
        ).astype(complex)
        PPlotY = np.zeros(
            (self.elements_axial, self.elements_circumferential * self.n_pad)
        ).astype(complex)

        i = 0
        for i in range(self.elements_axial):

            PPlotX[i] = PX[i, :, :].ravel("F")
            PPlotY[i] = PY[i, :, :].ravel("F")

        Ytheta = Ytheta.flatten()

        PPlotXdim = (
            PPlotX
            * (self.reference_viscosity * self.speed * (self.journal_radius**2))
            / (self.radial_clearance**3)
        )

        PPlotYdim = (
            PPlotY
            * (self.reference_viscosity * self.speed * (self.journal_radius**2))
            / (self.radial_clearance**3)
        )

        hX = hX.flatten()
        hY = hY.flatten()

        aux_intXX = PPlotXdim * hX.T

        aux_intXY = PPlotXdim * hY.T

        aux_intYX = PPlotYdim * hX.T

        aux_intYY = PPlotYdim * hY.T

        H[0, 0] = -np.trapz(np.trapz(aux_intXX, Ytheta * self.journal_radius), Zdim)

        H[0, 1] = -np.trapz(np.trapz(aux_intXY, Ytheta * self.journal_radius), Zdim)

        H[1, 0] = -np.trapz(np.trapz(aux_intYX, Ytheta * self.journal_radius), Zdim)

        H[1, 1] = -np.trapz(np.trapz(aux_intYY, Ytheta * self.journal_radius), Zdim)

        K = np.real(H)
        C = np.imag(H) / wp

        kxx = K[0, 0]
        kxy = K[0, 1]
        kyx = K[1, 0]
        kyy = K[1, 1]

        cxx = C[0, 0]
        cxy = C[0, 1]
        cyx = C[1, 0]
        cyy = C[1, 1]

        return (kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy)

    def _score(self, x, print_progress=False):
        """This method used to set the objective function of minimize optimization.

        Parameters
        ----------
        x: array
           Balanced Force expression between the load aplied in bearing and the
           resultant force provide by oil film.

        Returns
        -------
        Score coefficient.

        """
        Fhx, Fhy = self._forces(x, None, None, None)
        score = np.sqrt(
            ((self.load_x_direction + Fhx) ** 2) + ((self.load_y_direction + Fhy) ** 2)
        )
        if print_progress:
            print(f"Score: ", score)
            print("============================================")
            print(f"Force x direction: ", Fhx)
            print("============================================")
            print(f"Force y direction: ", Fhy)
            print("")

        return score

    def sommerfeld(self, force_x, force_y):
        """Calculate the sommerfeld number. This dimensionless number is used to
        calculate the dynamic coeficients.

        Parameters
        ----------
        force_x : float
            Force in x direction. The unit is newton.
        force_y : float
            Force in y direction. The unit is newton.

        Returns
        -------
        Ss : float
            Sommerfeld number.
        """
        if self.sommerfeld_type == 1:
            S = (
                self.reference_viscosity
                * ((self.journal_radius) ** 3)
                * self.axial_length
                * self.speed
            ) / (
                np.pi
                * (self.radial_clearance**2)
                * np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            )

        elif self.sommerfeld_type == 2:
            S = 1 / (
                2
                * ((self.axial_length / (2 * self.journal_radius)) ** 2)
                * (np.sqrt((force_x**2) + (force_y**2)))
            )

        Ss = S * ((self.axial_length / (2 * self.journal_radius)) ** 2)

        return Ss


def cylindrical_bearing_example():
    """Create an example of a cylindrical bearing with termo hydrodynamic effects.
    This function returns pressure and temperature field and dynamic coefficient.
    The purpose is to make available a simple model so that a doctest can be written
    using it.
    Returns
    -------
    THDCylindrical : ross.THDCylindrical Object
        An instance of a termo-hydrodynamic cylendrical bearing model object.
    Examples
    --------
    >>> bearing = cylindrical_bearing_example()
    >>> bearing.axial_length
    0.263144
    """

    bearing = THDCylindrical(
        axial_length=0.263144,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_circumferential=11,
        elements_axial=3,
        n_y=None,
        n_pad=2,
        pad_arc_length=176,
        reference_temperature=50,
        reference_viscosity=0.02,
        speed=Q_([900], "RPM"),
        load_x_direction=0,
        load_y_direction=-112814.91,
        groove_factor=[0.52, 0.48],
        lubricant="ISOVG32",
        node=3,
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        show_coef=False,
        print_result=False,
        print_progress=False,
        print_time=False,
    )

    return bearing
