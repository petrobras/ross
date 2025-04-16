import time

import numpy as np
from numpy.linalg import pinv
from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from ross.plotly_theme import tableau_colors
from scipy.optimize import curve_fit, minimize
from ross.bearings.lubricants import lubricants_dict

from plotly import graph_objects as go
from plotly import figure_factory as ff


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
    n : int
        Node in which the bearing will be located.
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
    preload: float
        Preload of the pad. The preload is defined as m=1-Cb/Cp where Cb is the radail clearance and Cp is
        the pad ground-in clearance.Preload is dimensionless.
    geometry: string
        Refers to bearing geometry. The options are: 'circular', 'lobe' or 'elliptical'.
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
    frequency : float, pint.Quantity
        Rotor rotational speed. Default unit is rad/s.
    load_x_direction : Float
        Load in X direction. The unit is newton.
    load_y_direction : Float
        Load in Y direction. The unit is newton.
    operating_type : string
        Choose the operating condition that bearing is operating.
        - 'flooded'
        - 'starvation'

    Fluid properties
    ^^^^^^^^^^^^^^^^^
    Describes the fluid characteristics.

    lubricant : str, dict
        Lubricant type. Can be:
        - 'ISOVG46' (lubricants in ross.bearings.lubricants)
    reference_temperature : float
        Oil reference temperature. The unit is celsius.
    groove_factor : list, numpy array, tuple or float
        Ratio of oil in reservoir temperature that mixes with the circulating oil.
        Is required one factor per segment.
    oil_flow: float
        Suply oil flow to bearing. Only used when operating type 'starvation' is
        selected. Unit is Litre per minute (l/min)
    injection_pressure: float
        Suply oil pressure that bearing receives at groove regions. Only used
        when operating type 'starvation' is selected. Unit is Pascal (Pa).

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
    >>> from ross.bearings.cylindrical import cylindrical_bearing_example
    >>> bearing = cylindrical_bearing_example()
    >>> bearing.equilibrium_pos
    array([ 0.68733194, -0.79394211])
    """

    @check_units
    def __init__(
        self,
        n,
        axial_length,
        journal_radius,
        radial_clearance,
        elements_circumferential,
        elements_axial,
        n_pad,
        pad_arc_length,
        preload,
        geometry,
        reference_temperature,
        frequency,
        load_x_direction,
        load_y_direction,
        groove_factor,
        lubricant,
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        operating_type="flooded",
        injection_pressure=None,
        oil_flow=None,
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
        self.n_pad = n_pad
        self.preload = preload
        self.geometry = geometry
        self.reference_temperature = reference_temperature
        self.load_x_direction = load_x_direction
        self.load_y_direction = load_y_direction
        self.lubricant = lubricant
        self.fat_mixt = np.array(groove_factor)
        self.equilibrium_pos = None
        self.sommerfeld_type = sommerfeld_type
        self.initial_guess = initial_guess
        self.method = method
        self.operating_type = operating_type
        self.injection_pressure = injection_pressure
        self.oil_flow = oil_flow
        self.show_coef = show_coef
        self.print_result = print_result
        self.print_progress = print_progress
        self.print_time = print_time

        self.betha_s_dg = pad_arc_length
        self.betha_s = pad_arc_length * np.pi / 180

        self.thetaI = 0
        self.thetaF = self.betha_s
        self.dtheta = (self.thetaF - self.thetaI) / (self.elements_circumferential)

        ##
        # Dimensionless discretization variables

        self.dY = 1 / self.elements_circumferential
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

        self.oil_flow = self.oil_flow / 60000

        # lubricant_properties = lubricants_dict[self.lubricant]
        lubricant_properties = (
            self.lubricant
            if isinstance(self.lubricant, dict)
            else lubricants_dict[self.lubricant]
        )

        T_muI = Q_(lubricant_properties["temperature1"], "degK").m_as("degC")
        T_muF = Q_(lubricant_properties["temperature2"], "degK").m_as("degC")
        mu_I = lubricant_properties["liquid_viscosity1"]
        mu_F = lubricant_properties["liquid_viscosity2"]
        self.rho = lubricant_properties["liquid_density"]
        self.Cp = lubricant_properties["liquid_specific_heat"]
        self.k_t = lubricant_properties["liquid_thermal_conductivity"]

        # Interpolation coefficients
        self.a, self.b = self._interpol(T_muI, T_muF, mu_I, mu_F)

        self.reference_viscosity = self.a * (self.reference_temperature**self.b)

        if self.geometry == "lobe":
            self.theta_pivot = np.array([90, 270]) * np.pi / 180

        number_of_freq = np.shape(frequency)[0]

        kxx = np.zeros(number_of_freq)
        kxy = np.zeros(number_of_freq)
        kyx = np.zeros(number_of_freq)
        kyy = np.zeros(number_of_freq)

        cxx = np.zeros(number_of_freq)
        cxy = np.zeros(number_of_freq)
        cyx = np.zeros(number_of_freq)
        cyy = np.zeros(number_of_freq)

        for ii in range(number_of_freq):
            self.frequency = frequency[ii]

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

        super().__init__(
            n, 
            kxx, cxx, kyy, kxy, kyx, cyy, cxy, cyx, 
            frequency,
        )

    def _flooded(self, n_p, Mat_coef, b_P, mu, initial_guess, y0):
        """Provides an analysis in which the bearing always receive sufficient oil feed to operate.

        Parameters
        ----------
        n_p : integer,
           current pad in analysis.
        Mat_coef : np.array
            Coeficient matrix.
        b_P: np.array
            Coefficients to pressure independent terms.
        mu : np.array
            Viscosity matrix.


        Returns
        -------
        self.P : np.array
            Pressure distribution in current pad vector.
        """

        ki = 0
        kj = 0
        k = 0

        for ii in np.arange((self.Z_I + 0.5 * self.dZ), self.Z_F, self.dZ):
            for jj in np.arange(
                self.thetaI[n_p] + (self.dtheta / 2),
                self.thetaF[n_p],
                self.dtheta,
            ):
                if self.geometry == "circular":
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

                else:
                    if self.geometry == "lobe":
                        hP = (
                            1 / (1 - self.preload)
                            - self.X * np.cos(jj)
                            - self.Y * np.sin(jj)
                            - self.preload
                            / (1 - self.preload)
                            * np.cos(jj - self.theta_pivot[n_p])
                        )
                        he = (
                            1 / (1 - self.preload)
                            - self.X * np.cos(jj + 0.5 * self.dtheta)
                            - self.Y * np.sin(jj + 0.5 * self.dtheta)
                            - self.preload
                            / (1 - self.preload)
                            * np.cos(jj + 0.5 * self.dtheta - self.theta_pivot[n_p])
                        )
                        hw = (
                            1 / (1 - self.preload)
                            - self.X * np.cos(jj - 0.5 * self.dtheta)
                            - self.Y * np.sin(jj - 0.5 * self.dtheta)
                            - self.preload
                            / (1 - self.preload)
                            * np.cos(jj - 0.5 * self.dtheta - self.theta_pivot[n_p])
                        )

                    if self.geometry == "elliptical":
                        hP = (
                            1
                            - self.X * np.cos(jj)
                            - self.Y * np.sin(jj)
                            + self.preload / (1 - self.preload) * (np.cos(jj)) ** 2
                        )

                        he = (
                            1
                            - self.X * np.cos(jj + 0.5 * self.dtheta)
                            - self.Y * np.sin(jj + 0.5 * self.dtheta)
                            + self.preload
                            / (1 - self.preload)
                            * (np.cos(jj + 0.5 * self.dtheta)) ** 2
                        )

                        hw = (
                            1
                            - self.X * np.cos(jj - 0.5 * self.dtheta)
                            - self.Y * np.sin(jj - 0.5 * self.dtheta)
                            + self.preload
                            / (1 - self.preload)
                            * (np.cos(jj - 0.5 * self.dtheta)) ** 2
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

                if ki == 0 and kj > 0 and kj < self.elements_circumferential - 1:
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

                CE = (self.dZ * he**3) / (12 * MU_e[n_p] * self.dY * self.betha_s**2)
                CW = (self.dZ * hw**3) / (12 * MU_w[n_p] * self.dY * self.betha_s**2)
                CN = (self.dY * (self.journal_radius**2) * hn**3) / (
                    12 * MU_n[n_p] * self.dZ * self.axial_length**2
                )
                CS = (self.dY * (self.journal_radius**2) * hs**3) / (
                    12 * MU_s[n_p] * self.dZ * self.axial_length**2
                )
                CP = -(CE + CW + CN + CS)

                B = (self.dZ / (2 * self.betha_s)) * (he - hw) - (
                    (self.Xpt * np.cos(jj) + self.Ypt * np.sin(jj)) * self.dY * self.dZ
                )

                k = k + 1
                b_P[k - 1, 0] = B

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

                elif (
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

        # Solution of pressure field end

        p = np.linalg.solve(Mat_coef, b_P)
        cont = 0

        for i in np.arange(self.elements_axial):
            for j in np.arange(self.elements_circumferential):
                self.P[i, j, n_p] = p[cont, 0]
                cont = cont + 1

                if self.P[i, j, n_p] < 0:
                    self.P[i, j, n_p] = 0

        # Dimensional pressure fied

        self.Pdim = (
            self.P * self.reference_viscosity * self.frequency * (self.journal_radius**2)
        ) / (self.radial_clearance**2)

        return self.P

    def _starvation(
        self,
        n_p,
        Mat_coef_st,
        mu,
        p_old,
        p,
        B,
        B_theta,
        nk,
        initial_guess,
        y0,
        xpt0,
        ypt0,
    ):
        """Provides an analysis in which the bearing may receive insufficient oil feed.

        Parameters
        ----------
        n_p : integer,
           current pad in analysis.
        Mat_coef_st : np.array
            Coeficient matrix.
        mu : np.array
            Viscosity matrix.
        p_old : np.array
            Past pressure matrix.
        p : np.array
            Current pressure matrix.
        B: np.array
            Coefficients to independent terms.
        B: np.array
            Coefficients to volumetric fraction independent terms.
        nk: integer
            counter.


        Returns
        -------
        self.P : np.array
            Pressure distribution in current pad vector.
        """

        while self.erro >= 0.01:
            p_old = np.array(p)

            theta_vol_old = np.array(self.theta_vol)

            k = 0
            ki = 0
            kj = 0

            for ii in np.arange((self.Z_I + 0.5 * self.dZ), self.Z_F, self.dZ):
                for jj in np.arange(
                    self.thetaI[n_p] + (self.dtheta / 2),
                    self.thetaF[n_p],
                    self.dtheta,
                ):
                    if self.geometry == "circular":
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

                    else:
                        if self.geometry == "lobe":
                            hP = (
                                1 / (1 - self.preload)
                                - self.X * np.cos(jj)
                                - self.Y * np.sin(jj)
                                - self.preload
                                / (1 - self.preload)
                                * np.cos(jj - self.theta_pivot[n_p])
                            )
                            he = (
                                1 / (1 - self.preload)
                                - self.X * np.cos(jj + 0.5 * self.dtheta)
                                - self.Y * np.sin(jj + 0.5 * self.dtheta)
                                - self.preload
                                / (1 - self.preload)
                                * np.cos(jj + 0.5 * self.dtheta - self.theta_pivot[n_p])
                            )
                            hw = (
                                1 / (1 - self.preload)
                                - self.X * np.cos(jj - 0.5 * self.dtheta)
                                - self.Y * np.sin(jj - 0.5 * self.dtheta)
                                - self.preload
                                / (1 - self.preload)
                                * np.cos(jj - 0.5 * self.dtheta - self.theta_pivot[n_p])
                            )

                        if self.geometry == "elliptical":
                            hP = (
                                1
                                - self.X * np.cos(jj)
                                - self.Y * np.sin(jj)
                                + self.preload / (1 - self.preload) * (np.cos(jj)) ** 2
                            )

                            he = (
                                1
                                - self.X * np.cos(jj + 0.5 * self.dtheta)
                                - self.Y * np.sin(jj + 0.5 * self.dtheta)
                                + self.preload
                                / (1 - self.preload)
                                * (np.cos(jj + 0.5 * self.dtheta)) ** 2
                            )

                            hw = (
                                1
                                - self.X * np.cos(jj - 0.5 * self.dtheta)
                                - self.Y * np.sin(jj - 0.5 * self.dtheta)
                                + self.preload
                                / (1 - self.preload)
                                * (np.cos(jj - 0.5 * self.dtheta)) ** 2
                            )

                    hn = hP
                    hs = hn

                    hpt = -self.Xpt * np.cos(jj) - self.Ypt * np.sin(jj)

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

                    if ki == 0 and kj > 0 and kj < self.elements_circumferential - 1:
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

                    # Termo Fonte
                    KP1 = -(self.dZ / (2 * self.betha_s)) * he

                    KP2 = -hpt * self.dY * self.dZ

                    KP = KP1 + KP2

                    KW = (self.dZ / (2 * self.betha_s)) * hw

                    if (
                        kj > 0
                        and kj < self.elements_circumferential - 1
                        and ki > 0
                        and ki < self.elements_axial - 1
                    ):  # Center region
                        Mat_coef_st[k, k] = CP
                        Mat_coef_st[k, k + 1] = CE
                        Mat_coef_st[k, k - 1] = CW
                        Mat_coef_st[k, k + self.elements_circumferential] = CN
                        Mat_coef_st[k, k - self.elements_circumferential] = CS

                        if p[k] > 0:
                            self.theta_vol[k] = 1
                            B[k] = -KP * self.theta_vol[k] - KW * self.theta_vol[k - 1]
                            pp = np.zeros((nk - 1, 1))
                            pp = np.delete(p, k)
                            C = np.zeros((1, nk))
                            C = Mat_coef_st[k, :]
                            C = np.delete(C, k)
                            p[k] = (B[k] - np.matmul(C, pp)) / Mat_coef_st[k, k]

                        else:
                            p[k] = 0
                            B_theta[k] = -np.matmul(Mat_coef_st[k, :], p)
                            self.theta_vol[k] = (
                                B_theta[k] - KW * self.theta_vol[k - 1]
                            ) / KP

                    elif (
                        kj > 0 and kj < self.elements_circumferential - 1 and ki == 0
                    ):  # Inferior edge without corners
                        Mat_coef_st[k, k] = CP - CS
                        Mat_coef_st[k, k + 1] = CE
                        Mat_coef_st[k, k - 1] = CW
                        Mat_coef_st[k, k + self.elements_circumferential] = CN

                        if p[k] > 0:
                            self.theta_vol[k] = 1
                            B[k] = -KP * self.theta_vol[k] - KW * self.theta_vol[k - 1]
                            pp = np.zeros((nk - 1, 1))
                            pp = np.delete(p, k)
                            C = np.zeros((1, nk))
                            C = Mat_coef_st[k, :]
                            C = np.delete(C, k)
                            p[k] = (B[k] - np.matmul(C, pp)) / Mat_coef_st[k, k]

                        else:
                            p[k] = 0
                            B_theta[k] = -np.matmul(Mat_coef_st[k, :], p)
                            self.theta_vol[k] = (
                                B_theta[k] - KW * self.theta_vol[k - 1]
                            ) / KP

                    elif (
                        kj > 0
                        and kj < self.elements_circumferential - 1
                        and ki == self.elements_axial - 1
                    ):  # Superior edge without corners
                        Mat_coef_st[k, k] = CP - CN
                        Mat_coef_st[k, k + 1] = CE
                        Mat_coef_st[k, k - 1] = CW
                        Mat_coef_st[k, k - self.elements_circumferential] = CS

                        if p[k] > 0:
                            self.theta_vol[k] = 1
                            B[k] = -KP * self.theta_vol[k] - KW * self.theta_vol[k - 1]
                            pp = np.zeros((nk - 1, 1))
                            pp = np.delete(p, k)
                            C = np.zeros((1, nk))
                            C = Mat_coef_st[k, :]
                            C = np.delete(C, k)
                            p[k] = (B[k] - np.matmul(C, pp)) / Mat_coef_st[k, k]

                        else:
                            p[k] = 0
                            B_theta[k] = -np.matmul(Mat_coef_st[k, :], p)
                            self.theta_vol[k] = (
                                B_theta[k] - KW * self.theta_vol[k - 1]
                            ) / KP

                    elif (
                        kj == 0 and ki > 0 and ki < self.elements_axial - 1
                    ):  # Left edge without corners
                        Mat_coef_st[k, k] = CP - CW
                        Mat_coef_st[k, k + 1] = CE
                        Mat_coef_st[k, k - self.elements_circumferential] = CS
                        Mat_coef_st[k, k + self.elements_circumferential] = CN

                        if p[k] > 0:
                            self.theta_vol[k] = 1
                            B[k] = (
                                -KP * self.theta_vol[k]
                                - self.theta_vol_groove[n_p] * KW
                                - 2 * CW * self.injection_pressure
                            )
                            pp = np.zeros((nk - 1, 1))
                            pp = np.delete(p, k)
                            C = np.zeros((1, nk))
                            C = Mat_coef_st[k, :]
                            C = np.delete(C, k)
                            p[k] = (B[k] - np.matmul(C, pp)) / Mat_coef_st[k, k]

                        else:
                            p[k] = 0
                            B_theta[k] = -np.matmul(Mat_coef_st[k, :], p)
                            self.theta_vol[k] = (
                                B_theta[k] - self.theta_vol_groove[n_p] * KW * 1
                            ) / KP

                    elif (
                        kj == self.elements_circumferential - 1
                        and ki > 0
                        and ki < self.elements_axial - 1
                    ):  # Right edge without corners
                        Mat_coef_st[k, k] = CP - CE

                        Mat_coef_st[k, k - 1] = CW
                        Mat_coef_st[k, k + self.elements_circumferential] = CN
                        Mat_coef_st[k, k - self.elements_circumferential] = CS

                        if p[k] > 0:
                            self.theta_vol[k] = 1
                            B[k] = -KP * self.theta_vol[k] - KW * self.theta_vol[k - 1]
                            pp = np.zeros((nk - 1, 1))
                            pp = np.delete(p, k)
                            C = np.zeros((1, nk))
                            C = Mat_coef_st[k, :]
                            C = np.delete(C, k)
                            p[k] = (B[k] - np.matmul(C, pp)) / Mat_coef_st[k, k]

                        else:
                            p[k] = 0
                            B_theta[k] = -np.matmul(Mat_coef_st[k, :], p)
                            self.theta_vol[k] = (
                                B_theta[k] - KW * self.theta_vol[k - 1]
                            ) / KP

                    elif kj == 0 and ki == 0:  # Corner inferior left
                        Mat_coef_st[k, k] = CP - CS - CW
                        Mat_coef_st[k, k + 1] = CE
                        Mat_coef_st[k, k + self.elements_circumferential] = CN

                        if p[k] > 0:
                            self.theta_vol[k] = 1
                            B[k] = (
                                -KP * self.theta_vol[k]
                                - self.theta_vol_groove[n_p] * KW
                                - 2 * CW * self.injection_pressure
                            )
                            pp = np.zeros((nk - 1, 1))
                            pp = np.delete(p, k)
                            C = np.zeros((1, nk))
                            C = Mat_coef_st[k, :]
                            C = np.delete(C, k)
                            p[k] = (B[k] - np.matmul(C, pp)) / Mat_coef_st[k, k]

                        else:
                            p[k] = 0
                            B_theta[k] = -np.matmul(Mat_coef_st[k, :], p)
                            self.theta_vol[k] = (
                                B_theta[k] - self.theta_vol_groove[n_p] * KW * 1
                            ) / KP

                    elif (
                        kj == self.elements_circumferential - 1 and ki == 0
                    ):  # Corner inferior right
                        Mat_coef_st[k, k] = CP - CS - CE
                        Mat_coef_st[k, k - 1] = CW
                        Mat_coef_st[k, k + self.elements_circumferential] = CN

                        if p[k] > 0:
                            self.theta_vol[k] = 1
                            B[k] = -KP * self.theta_vol[k] - KW * self.theta_vol[k - 1]
                            pp = np.zeros((nk - 1, 1))
                            pp = np.delete(p, k)
                            C = np.zeros((1, nk))
                            C = Mat_coef_st[k, :]
                            C = np.delete(C, k)
                            p[k] = (B[k] - np.matmul(C, pp)) / Mat_coef_st[k, k]

                        else:
                            p[k] = 0
                            B_theta[k] = -np.matmul(Mat_coef_st[k, :], p)
                            self.theta_vol[k] = (
                                B_theta[k] - KW * self.theta_vol[k - 1]
                            ) / KP

                    elif (
                        kj == 0 and ki == self.elements_axial - 1
                    ):  # Corner superior left
                        Mat_coef_st[k, k] = CP - CN - CW
                        Mat_coef_st[k, k + 1] = CE
                        Mat_coef_st[k, k - self.elements_circumferential] = CS

                        if p[k] > 0:
                            self.theta_vol[k] = 1
                            B[k] = (
                                -KP * self.theta_vol[k]
                                - self.theta_vol_groove[n_p] * KW
                                - 2 * CW * self.injection_pressure
                            )
                            pp = np.zeros((nk - 1, 1))
                            pp = np.delete(p, k)
                            C = np.zeros((1, nk))
                            C = Mat_coef_st[k, :]
                            C = np.delete(C, k)
                            p[k] = (B[k] - np.matmul(C, pp)) / Mat_coef_st[k, k]

                        else:
                            p[k] = 0
                            B_theta[k] = -np.matmul(Mat_coef_st[k, :], p)
                            self.theta_vol[k] = (
                                B_theta[k] - self.theta_vol_groove[n_p] * KW * 1
                            ) / KP

                    elif (
                        kj == self.elements_circumferential - 1
                        and ki == self.elements_axial - 1
                    ):  # Corner superior right
                        Mat_coef_st[k, k] = CP - CN - CE
                        Mat_coef_st[k, k - 1] = CW
                        Mat_coef_st[k, k - self.elements_circumferential] = CS

                        if p[k] > 0:
                            self.theta_vol[k] = 1
                            B[k] = -KP * self.theta_vol[k] - KW * self.theta_vol[k - 1]
                            pp = np.zeros((nk - 1, 1))
                            pp = np.delete(p, k)
                            C = np.zeros((1, nk))
                            C = Mat_coef_st[k, :]
                            C = np.delete(C, k)
                            p[k] = (B[k] - np.matmul(C, pp)) / Mat_coef_st[k, k]

                        else:
                            p[k] = 0
                            B_theta[k] = -np.matmul(Mat_coef_st[k, :], p)
                            self.theta_vol[k] = (
                                B_theta[k] - KW * self.theta_vol[k - 1]
                            ) / KP

                    k = k + 1
                    kj = kj + 1

                kj = 0
                ki = ki + 1

            self.erro = np.linalg.norm(p - p_old) + np.linalg.norm(
                self.theta_vol - theta_vol_old
            )

        cont = 0

        for i in np.arange(self.elements_axial):
            for j in np.arange(self.elements_circumferential):
                self.P[i, j, n_p] = p[cont]

                self.Theta_vol[i, j, n_p] = self.theta_vol[cont]

                cont = cont + 1

                if self.P[i, j, n_p] < 0:
                    self.P[i, j, n_p] = 0

        # Dimensional pressure fied

        self.Pdim = (
            self.P * self.reference_viscosity * self.frequency * (self.journal_radius**2)
        ) / (self.radial_clearance**2)

        return self.P

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

            self.Xpt = xpt0 / (self.radial_clearance * self.frequency)
            self.Ypt = ypt0 / (self.radial_clearance * self.frequency)

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
            np.linspace(
                t1 + self.dtheta / 2,
                t2 - self.dtheta / 2,
                self.elements_circumferential,
            )
            for t1, t2 in zip(self.thetaI, self.thetaF)
        ]

        self.theta_vol_groove = 0.8 * np.ones(self.n_pad)

        T_end = np.ones(self.n_pad)

        while (T_mist[0] - T_conv) >= 0.5:
            H_PLOT = np.zeros((self.elements_circumferential, self.n_pad))
            nk = (self.elements_axial) * (self.elements_circumferential)
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

            self.Theta_vol = np.zeros(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )

            mu_new = 1.1 * np.ones(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )
            mu_turb = 1.3 * np.ones(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )

            T_conv = T_mist[0]

            self.H = np.ones((self.elements_circumferential, self.n_pad))

            U = 0.5 * np.ones(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )

            self.V = np.zeros(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )

            self.Qedim = np.ones(self.n_pad)

            self.Qsdim = np.ones(self.n_pad)

            self.Qldim = np.ones(self.n_pad)

            b_T = np.zeros((nk, 1))

            b_P = np.zeros((nk, 1))

            Mat_coef = np.zeros((nk, nk))  # Coeficients matrix

            B = np.zeros((nk, 1))  # Termo fonte for pressure

            for n_p in np.arange(self.n_pad):
                T_ref = T_mist[n_p]

                while (
                    np.linalg.norm(T_new[:, :, n_p] - T[:, :, n_p])
                    / np.linalg.norm(T[:, :, n_p])
                    >= 0.01
                ):
                    T_ref = T_mist[n_p]

                    mu = mu_new

                    self.mu_l = mu_new

                    T[:, :, n_p] = T_new[:, :, n_p]

                    self.erro = 1

                    p_old = np.zeros((nk, 1))

                    self.theta_vol = np.zeros((nk, 1))  # Theta volumetric vector

                    Mat_coef_st = np.zeros((nk, nk))  # Coeficients matrix

                    Mat_coef_T = np.zeros((nk, nk))

                    p = np.ones((nk, 1))  # Pressure vector

                    B_theta = np.zeros((nk, 1))

                    if self.operating_type == "flooded":
                        self._flooded(n_p, Mat_coef, b_P, mu, initial_guess, y0)

                    elif self.operating_type == "starvation":
                        self._starvation(
                            n_p,
                            Mat_coef_st,
                            mu,
                            p_old,
                            p,
                            B,
                            B_theta,
                            nk,
                            initial_guess,
                            y0,
                            xpt0,
                            ypt0,
                        )

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

                            if self.geometry == "circular":
                                HP = 1 - self.X * np.cos(jj) - self.Y * np.sin(jj)

                            else:
                                if self.geometry == "lobe":
                                    HP = (
                                        1 / (1 - self.preload)
                                        - self.X * np.cos(jj)
                                        - self.Y * np.sin(jj)
                                        - self.preload
                                        / (1 - self.preload)
                                        * np.cos(jj - self.theta_pivot[n_p])
                                    )

                                if self.geometry == "elliptical":
                                    HP = (
                                        1
                                        - self.X * np.cos(jj)
                                        - self.Y * np.sin(jj)
                                        + self.preload
                                        / (1 - self.preload)
                                        * (np.cos(jj)) ** 2
                                    )

                            if ki == 0:
                                H_PLOT[kj, n_p] = HP

                            hpt = -self.Xpt * np.cos(jj) - self.Ypt * np.sin(jj)

                            self.H[kj, n_p] = HP

                            mu_p = mu[ki, kj, n_p]

                            if self.operating_type == "starvation":
                                Reyn[ki, kj, n_p] = (
                                    self.Theta_vol[ki, kj, n_p]
                                    * self.rho
                                    * self.frequency
                                    * self.journal_radius
                                    * (HP / self.axial_length)
                                    * self.radial_clearance
                                    / (self.reference_viscosity * mu_p)
                                )

                            else:
                                Reyn[ki, kj, n_p] = (
                                    self.rho
                                    * self.frequency
                                    * self.journal_radius
                                    * (HP / self.axial_length)
                                    * self.radial_clearance
                                    / (self.reference_viscosity * mu_p)
                                )

                            if Reyn[ki, kj, n_p] <= 500:
                                self.delta_turb = 0

                            elif Reyn[ki, kj, n_p] > 500 and Reyn[ki, kj, n_p] <= 1000:
                                self.delta_turb = 1 - (
                                    (1000 - Reyn[ki, kj, n_p]) / 500
                                ) ** (1 / 8)

                            elif Reyn[ki, kj, n_p] > 1000:
                                self.delta_turb = 1

                            dudy = ((HP / mu_turb[ki, kj, n_p]) * dPdy[ki, kj, n_p]) - (
                                self.frequency / HP
                            )

                            dwdy = (HP / mu_turb[ki, kj, n_p]) * dPdz[ki, kj, n_p]

                            tal = mu_turb[ki, kj, n_p] * np.sqrt((dudy**2) + (dwdy**2))

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

                            U[ki, kj, n_p] = (
                                -(HP**2)
                                / (12 * mi_t * self.betha_s)
                                * dPdy[ki, kj, n_p]
                                + 1 / 2
                            )

                            AE = -(self.k_t * HP * self.dZ) / (
                                self.rho
                                * self.Cp
                                * self.frequency
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
                                        * self.frequency
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
                                    * self.frequency
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
                                    * self.frequency
                                    * (self.axial_length**2)
                                    * self.dZ
                                )
                            )

                            AP = -(AE + AW + AN + AS)

                            auxb_T = (self.frequency * self.reference_viscosity) / (
                                self.rho
                                * self.Cp
                                * self.reference_temperature
                                * self.radial_clearance
                            )
                            b_TG = (
                                self.reference_viscosity
                                * self.frequency
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
                                self.frequency
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
                                * (mi_t * (self.journal_radius**2) * self.dY * self.dZ)
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

                            elif (
                                ki == 0
                                and kj > 0
                                and kj < self.elements_circumferential - 1
                            ):
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
                                and kj < self.elements_circumferential - 1
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
                                and kj < self.elements_circumferential - 1
                            ):
                                Mat_coef_T[k - 1, k - 1] = AP + AN
                                Mat_coef_T[k - 1, k] = AE
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[
                                    k - 1, k - self.elements_circumferential - 1
                                ] = AS

                            elif ki == 0 and kj == self.elements_circumferential - 1:
                                Mat_coef_T[k - 1, k - 1] = AP + AE + AS
                                Mat_coef_T[k - 1, k - 2] = AW
                                Mat_coef_T[
                                    k - 1, k + self.elements_circumferential - 1
                                ] = AN

                            elif (
                                kj == self.elements_circumferential - 1
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

                            elif (
                                ki == self.elements_axial - 1
                                and kj == self.elements_circumferential - 1
                            ):
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
                            T_new[i, j, n_p] = t[cont, 0]
                            cont = cont + 1

                    Tdim = T_new * self.reference_temperature

                    T_end[n_p] = np.sum(Tdim[:, -1, n_p]) / self.elements_axial

                    if self.operating_type == "flooded":
                        T_mist[n_p - 1] = (
                            self.fat_mixt[n_p] * self.reference_temperature
                            + (1 - self.fat_mixt[n_p]) * T_end[n_p]
                        )

                    mu_new[:, :, n_p] = (
                        self.a * (Tdim[:, :, n_p]) ** self.b
                    ) / self.reference_viscosity

            if self.operating_type == "starvation":
                for n_p in np.arange(self.n_pad):
                    self.Qedim[n_p] = (
                        self.radial_clearance
                        * self.H[0, n_p]
                        * self.frequency
                        * self.journal_radius
                        * self.axial_length
                        * self.Theta_vol[0, 0, n_p]
                        * (np.mean(U[:, 0, n_p]))
                    )

                    self.Qsdim[n_p] = (
                        self.radial_clearance
                        * self.H[-1, n_p]
                        * self.frequency
                        * self.journal_radius
                        * self.axial_length
                        * self.Theta_vol[0, -1, n_p]
                        * (np.mean(U[:, -1, n_p]))
                    )

                geometry_factor = np.ones(self.n_pad)

                for n_p in np.arange(self.n_pad):
                    geometry_factor[n_p] = (self.Qedim[n_p] + self.Qsdim[n_p - 1]) / (
                        np.sum(self.Qedim) + np.sum(self.Qsdim)
                    )

                for n_p in np.arange(self.n_pad):
                    T_mist[n_p] = (
                        (self.Qsdim[n_p - 1] * T_end[n_p - 1])
                        + (
                            self.reference_temperature
                            * geometry_factor[n_p]
                            * self.oil_flow
                        )
                    ) / (geometry_factor[n_p] * self.oil_flow + self.Qsdim[n_p - 1])

                    self.theta_vol_groove[n_p] = (
                        0.8
                        * (geometry_factor[n_p] * self.oil_flow + self.Qsdim[n_p - 1])
                        / self.Qedim[n_p]
                    )

                    if self.theta_vol_groove[n_p] > 1:
                        self.theta_vol_groove[n_p] = 1

        PPlot = np.zeros(
            (self.elements_axial, self.elements_circumferential * self.n_pad)
        )

        TPlot = np.zeros(
            (self.elements_axial, self.elements_circumferential * self.n_pad)
        )

        for i in range(self.elements_axial):
            PPlot[i] = self.Pdim[i, :, :].ravel("F")
            TPlot[i] = Tdim[i, :, :].ravel("F")

        Ytheta = np.array(Ytheta)
        Ytheta = Ytheta.flatten()

        auxF = np.zeros((2, len(Ytheta)))

        auxF[0, :] = np.cos(Ytheta)
        auxF[1, :] = np.sin(Ytheta)

        fx1 = np.trapz(PPlot * auxF[0, :], self.journal_radius * Ytheta)
        Fhx = -np.trapz(fx1, self.axial_length * self.Z[1 : self.elements_axial + 1])

        fy1 = np.trapz(PPlot * auxF[1, :], self.journal_radius * Ytheta)
        Fhy = -np.trapz(fy1, self.axial_length * self.Z[1 : self.elements_axial + 1])
        F1 = Fhx
        F2 = Fhy

        Fhx = F1
        Fhy = F2

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
            bounds=[(0, 1), (-2 * np.pi, 2 * np.pi)],
            tol=0.8,
            options={"maxiter": 1e10},
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

        popt, pcov = curve_fit(viscosity, xdata, ydata, p0=(6.0, 1.0))
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

        Va = self.frequency * (self.journal_radius)
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

        Cxx = -self.sommerfeld(Auinitial_guess5[0], Auinitial_guess6[1]) * (
            (Auinitial_guess5[0] - Auinitial_guess6[0])
            / (epixpt / self.radial_clearance / self.frequency)
        )
        Cxy = -self.sommerfeld(Auinitial_guess7[0], Auinitial_guess8[1]) * (
            (Auinitial_guess7[0] - Auinitial_guess8[0])
            / (epiypt / self.radial_clearance / self.frequency)
        )
        Cyx = -self.sommerfeld(Auinitial_guess5[0], Auinitial_guess6[1]) * (
            (Auinitial_guess5[1] - Auinitial_guess6[1])
            / (epixpt / self.radial_clearance / self.frequency)
        )
        Cyy = -self.sommerfeld(Auinitial_guess7[0], Auinitial_guess8[1]) * (
            (Auinitial_guess7[1] - Auinitial_guess8[1])
            / (epiypt / self.radial_clearance / self.frequency)
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
            / (self.radial_clearance * self.frequency)
        ) * Cxx
        cxy = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / (self.radial_clearance * self.frequency)
        ) * Cxy
        cyx = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / (self.radial_clearance * self.frequency)
        ) * Cyx
        cyy = (
            np.sqrt((self.load_x_direction**2) + (self.load_y_direction**2))
            / (self.radial_clearance * self.frequency)
        ) * Cyy

        return (kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy)

    def _lund_method(self):
        """In this method a small amplitude whirl of the journal center (a first
        order perturbation solution) is aplied. The four stiffness coefficients,
        and the four damping coefficients is obtained by integration of the pressure
        field.
        """

        Ytheta = [
            np.linspace(
                t1 + self.dtheta / 2,
                t2 - self.dtheta / 2,
                self.elements_circumferential,
            )
            for t1, t2 in zip(self.thetaI, self.thetaF)
        ]

        nk = (self.elements_axial) * (self.elements_circumferential)

        gamma = 0.001

        HX = -np.cos(Ytheta)

        HY = -np.sin(Ytheta)

        PX = np.zeros(
            (self.elements_axial, self.elements_circumferential, self.n_pad)
        ).astype(complex)

        PY = np.zeros(
            (self.elements_axial, self.elements_circumferential, self.n_pad)
        ).astype(complex)

        if self.operating_type == "flooded":
            self.theta_vol_groove = np.ones(self.n_pad)
            self.Theta_vol = np.ones(
                (self.elements_axial, self.elements_circumferential, self.n_pad)
            )

        for n_p in np.arange(self.n_pad):
            erro = 1

            while erro > 1e-6:
                PX_old = np.array(PX)
                PY_old = np.array(PY)

                Mat_coef = np.zeros((nk, nk))
                Mat_coefX = np.zeros((nk, nk))
                Mat_coefY = np.zeros((nk, nk))

                ki = 0
                kj = 0

                k = 0

                for ii in np.arange((self.Z_I + 0.5 * self.dZ), self.Z_F, self.dZ):
                    for jj in np.arange(
                        self.thetaI[n_p] + (self.dtheta / 2),
                        self.thetaF[n_p],
                        self.dtheta,
                    ):
                        if self.P[ki, kj, n_p] > 0:
                            if self.geometry == "circular":
                                HP = 1 - self.X * np.cos(jj) - self.Y * np.sin(jj)
                                He = (
                                    1
                                    - self.X * np.cos(jj + 0.5 * self.dtheta)
                                    - self.Y * np.sin(jj + 0.5 * self.dtheta)
                                )
                                Hw = (
                                    1
                                    - self.X * np.cos(jj - 0.5 * self.dtheta)
                                    - self.Y * np.sin(jj - 0.5 * self.dtheta)
                                )

                            else:
                                if self.geometry == "lobe":
                                    HP = (
                                        1 / (1 - self.preload)
                                        - self.X * np.cos(jj)
                                        - self.Y * np.sin(jj)
                                        - self.preload
                                        / (1 - self.preload)
                                        * np.cos(jj - self.theta_pivot[n_p])
                                    )
                                    He = (
                                        1 / (1 - self.preload)
                                        - self.X * np.cos(jj + 0.5 * self.dtheta)
                                        - self.Y * np.sin(jj + 0.5 * self.dtheta)
                                        - self.preload
                                        / (1 - self.preload)
                                        * np.cos(
                                            jj
                                            + 0.5 * self.dtheta
                                            - self.theta_pivot[n_p]
                                        )
                                    )
                                    Hw = (
                                        1 / (1 - self.preload)
                                        - self.X * np.cos(jj - 0.5 * self.dtheta)
                                        - self.Y * np.sin(jj - 0.5 * self.dtheta)
                                        - self.preload
                                        / (1 - self.preload)
                                        * np.cos(
                                            jj
                                            - 0.5 * self.dtheta
                                            - self.theta_pivot[n_p]
                                        )
                                    )

                                if self.geometry == "elliptical":
                                    HP = (
                                        1
                                        - self.X * np.cos(jj)
                                        - self.Y * np.sin(jj)
                                        + self.preload
                                        / (1 - self.preload)
                                        * (np.cos(jj)) ** 2
                                    )

                                    He = (
                                        1
                                        - self.X * np.cos(jj + 0.5 * self.dtheta)
                                        - self.Y * np.sin(jj + 0.5 * self.dtheta)
                                        + self.preload
                                        / (1 - self.preload)
                                        * (np.cos(jj + 0.5 * self.dtheta)) ** 2
                                    )

                                    Hw = (
                                        1
                                        - self.X * np.cos(jj - 0.5 * self.dtheta)
                                        - self.Y * np.sin(jj - 0.5 * self.dtheta)
                                        + self.preload
                                        / (1 - self.preload)
                                        * (np.cos(jj - 0.5 * self.dtheta)) ** 2
                                    )

                            HXP = -np.cos(jj)
                            HXe = -np.cos(jj + 0.5 * self.dtheta)
                            HXw = -np.cos(jj - 0.5 * self.dtheta)

                            HYP = -np.sin(jj)
                            HYe = -np.sin(jj + 0.5 * self.dtheta)
                            HYw = -np.sin(jj - 0.5 * self.dtheta)

                            HXptP = 0
                            HYptP = 0

                            if kj == 0 and ki == 0:
                                MU_e = 0.5 * (
                                    self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj + 1, n_p]
                                )
                                MU_w = self.mu_l[ki, kj, n_p]
                                MU_s = self.mu_l[ki, kj, n_p]
                                MU_n = 0.5 * (
                                    self.mu_l[ki, kj, n_p] + self.mu_l[ki + 1, kj, n_p]
                                )

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

                            if kj == 0 and ki == self.elements_axial - 1:
                                MU_e = 0.5 * (
                                    self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj + 1, n_p]
                                )
                                MU_w = self.mu_l[ki, kj, n_p]
                                MU_s = 0.5 * (
                                    self.mu_l[ki, kj, n_p] + self.mu_l[ki - 1, kj, n_p]
                                )
                                MU_n = self.mu_l[ki, kj, n_p]

                            if (
                                ki == 0
                                and kj > 0
                                and kj < self.elements_circumferential - 1
                            ):
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

                            if ki == 0 and kj == self.elements_circumferential - 1:
                                MU_e = self.mu_l[ki, kj, n_p]
                                MU_w = 0.5 * (
                                    self.mu_l[ki, kj, n_p] + self.mu_l[ki, kj - 1, n_p]
                                )
                                MU_s = self.mu_l[ki, kj, n_p]
                                MU_n = 0.5 * (
                                    self.mu_l[ki, kj, n_p] + self.mu_l[ki + 1, kj, n_p]
                                )

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

                            CE = (
                                1
                                / self.betha_s**2
                                * He**3
                                / (12 * MU_e)
                                * self.dZ
                                / self.dY
                            )
                            CW = (
                                1
                                / self.betha_s**2
                                * Hw**3
                                / (12 * MU_w)
                                * self.dZ
                                / self.dY
                            )
                            CN = (
                                (self.journal_radius / self.axial_length) ** 2
                                * HP**3
                                / (12 * MU_n)
                                * self.dY
                                / self.dZ
                            )
                            CS = (
                                (self.journal_radius / self.axial_length) ** 2
                                * HP**3
                                / (12 * MU_s)
                                * self.dY
                                / self.dZ
                            )
                            CP = -(CE + CW + CN + CS)

                            CXE = (
                                -1
                                / self.betha_s**2
                                * He**2
                                * HXe
                                / (4 * MU_e)
                                * self.dZ
                                / self.dY
                            )
                            CXW = (
                                -1
                                / self.betha_s**2
                                * Hw**2
                                * HXw
                                / (4 * MU_w)
                                * self.dZ
                                / self.dY
                            )
                            CXN = (
                                -((self.journal_radius / self.axial_length) ** 2)
                                * HP**2
                                * HXP
                                / (4 * MU_n)
                                * self.dY
                                / self.dZ
                            )
                            CXS = (
                                -((self.journal_radius / self.axial_length) ** 2)
                                * HP**2
                                * HXP
                                / (4 * MU_s)
                                * self.dY
                                / self.dZ
                            )
                            CXP = -(CXE + CXW + CXN + CXS)

                            CYE = (
                                -1
                                / self.betha_s**2
                                * He**2
                                * HYe
                                / (4 * MU_e)
                                * self.dZ
                                / self.dY
                            )
                            CYW = (
                                -1
                                / self.betha_s**2
                                * Hw**2
                                * HYw
                                / (4 * MU_w)
                                * self.dZ
                                / self.dY
                            )
                            CYN = (
                                -((self.journal_radius / self.axial_length) ** 2)
                                * HP**2
                                * HYP
                                / (4 * MU_n)
                                * self.dY
                                / self.dZ
                            )
                            CYS = (
                                -((self.journal_radius / self.axial_length) ** 2)
                                * HP**2
                                * HYP
                                / (4 * MU_s)
                                * self.dY
                                / self.dZ
                            )
                            CYP = -(CYE + CYW + CYN + CYS)

                            KXW = -1 / (2 * self.betha_s) * HXw * self.dZ
                            KXP = (
                                1 / (2 * self.betha_s) * HXe * self.dZ
                                + (HXptP + 1j * gamma * HXP) * self.dY * self.dZ
                            )

                            KYW = -1 / (2 * self.betha_s) * HYw * self.dZ
                            KYP = (
                                1 / (2 * self.betha_s) * HYe * self.dZ
                                + (HYptP + 1j * gamma * HYP) * self.dY * self.dZ
                            )

                            PP = self.P[:, :, n_p].flatten()

                            PPX = PX[:, :, n_p].flatten()
                            PPX = np.delete(PPX, k)

                            PPY = PY[:, :, n_p].flatten()
                            PPY = np.delete(PPY, k)

                            if kj == 0 and ki == 0:
                                Mat_coef[k, k] = CP - CW - CS
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k + self.elements_circumferential] = CN

                                Mat_coefX[k, k] = CXP - CXW - CXS
                                Mat_coefX[k, k + 1] = CXE
                                Mat_coefX[k, k + self.elements_circumferential] = CXN

                                Mat_coefY[k, k] = CYP - CYW - CYS
                                Mat_coefY[k, k + 1] = CYE
                                Mat_coefY[k, k + self.elements_circumferential] = CYN

                                C = Mat_coef[k, :].T
                                C = np.delete(C, k)

                                CX = Mat_coefX[k, :].T
                                CY = Mat_coefY[k, :].T

                                BX = (
                                    np.matmul(CX, PP)
                                    + KXW * self.theta_vol_groove[n_p]
                                    + KXP * self.Theta_vol[ki, kj, n_p]
                                    + 2 * CXW * self.injection_pressure
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.theta_vol_groove[n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                    + 2 * CYW * self.injection_pressure
                                )

                                PX[ki, kj, n_p] = (BX - np.matmul(C, PPX)) / Mat_coef[
                                    k, k
                                ]
                                PY[ki, kj, n_p] = (BY - np.matmul(C, PPY)) / Mat_coef[
                                    k, k
                                ]

                            if kj == 0 and ki > 0 and ki < self.elements_axial - 1:
                                Mat_coef[k, k] = CP - CW
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k + self.elements_circumferential] = CN
                                Mat_coef[k, k - self.elements_circumferential] = CS

                                Mat_coefX[k, k] = CXP - CXW
                                Mat_coefX[k, k + 1] = CXE
                                Mat_coefX[k, k + self.elements_circumferential] = CXN
                                Mat_coefX[k, k - self.elements_circumferential] = CXS

                                Mat_coefY[k, k] = CYP - CYW
                                Mat_coefY[k, k + 1] = CYE
                                Mat_coefY[k, k + self.elements_circumferential] = CYN
                                Mat_coefY[k, k - self.elements_circumferential] = CYS

                                C = Mat_coef[k, :].T
                                C = np.delete(C, k)

                                CX = Mat_coefX[k, :].T
                                CY = Mat_coefY[k, :].T

                                BX = (
                                    np.matmul(CX, PP)
                                    + KXW * self.theta_vol_groove[n_p]
                                    + KXP * self.Theta_vol[ki, kj, n_p]
                                    + 2 * CXW * self.injection_pressure
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.theta_vol_groove[n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                    + 2 * CYW * self.injection_pressure
                                )

                                PX[ki, kj, n_p] = (BX - np.matmul(C, PPX)) / Mat_coef[
                                    k, k
                                ]
                                PY[ki, kj, n_p] = (BY - np.matmul(C, PPY)) / Mat_coef[
                                    k, k
                                ]

                            if kj == 0 and ki == self.elements_axial - 1:
                                Mat_coef[k, k] = CP - CW - CN
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - self.elements_circumferential] = CS

                                Mat_coefX[k, k] = CXP - CXW - CXN
                                Mat_coefX[k, k + 1] = CXE
                                Mat_coefX[k, k - self.elements_circumferential] = CXS

                                Mat_coefY[k, k] = CYP - CYW - CYN
                                Mat_coefY[k, k + 1] = CYE
                                Mat_coefY[k, k - self.elements_circumferential] = CYS

                                C = Mat_coef[k, :].T
                                C = np.delete(C, k)

                                CX = Mat_coefX[k, :].T
                                CY = Mat_coefY[k, :].T

                                BX = (
                                    np.matmul(CX, PP)
                                    + KXW * self.theta_vol_groove[n_p]
                                    + KXP * self.Theta_vol[ki, kj, n_p]
                                    + 2 * CXW * self.injection_pressure
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.theta_vol_groove[n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                    + 2 * CYW * self.injection_pressure
                                )

                                PX[ki, kj, n_p] = (BX - np.matmul(C, PPX)) / Mat_coef[
                                    k, k
                                ]
                                PY[ki, kj, n_p] = (BY - np.matmul(C, PPY)) / Mat_coef[
                                    k, k
                                ]

                            if (
                                ki == 0
                                and kj > 0
                                and kj < self.elements_circumferential - 1
                            ):
                                Mat_coef[k, k] = CP - CS
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k + self.elements_circumferential] = CN

                                Mat_coefX[k, k] = CXP - CXS
                                Mat_coefX[k, k + 1] = CXE
                                Mat_coefX[k, k - 1] = CXW
                                Mat_coefX[k, k + self.elements_circumferential] = CXN

                                Mat_coefY[k, k] = CYP - CYS
                                Mat_coefY[k, k + 1] = CYE
                                Mat_coefY[k, k - 1] = CYW
                                Mat_coefY[k, k + self.elements_circumferential] = CYN

                                C = Mat_coef[k, :].T
                                C = np.delete(C, k)

                                CX = Mat_coefX[k, :].T
                                CY = Mat_coefY[k, :].T

                                BX = (
                                    np.matmul(CX, PP)
                                    + KXW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KXP * self.Theta_vol[ki, kj, n_p]
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                )

                                PX[ki, kj, n_p] = (BX - np.matmul(C, PPX)) / Mat_coef[
                                    k, k
                                ]
                                PY[ki, kj, n_p] = (BY - np.matmul(C, PPY)) / Mat_coef[
                                    k, k
                                ]

                            if (
                                kj > 0
                                and kj < self.elements_circumferential - 1
                                and ki > 0
                                and ki < self.elements_axial - 1
                            ):
                                Mat_coef[k, k] = CP
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k + self.elements_circumferential] = CN
                                Mat_coef[k, k - self.elements_circumferential] = CS

                                Mat_coefX[k, k] = CXP
                                Mat_coefX[k, k + 1] = CXE
                                Mat_coefX[k, k - 1] = CXW
                                Mat_coefX[k, k + self.elements_circumferential] = CXN
                                Mat_coefX[k, k - self.elements_circumferential] = CXS

                                Mat_coefY[k, k] = CYP
                                Mat_coefY[k, k + 1] = CYE
                                Mat_coefY[k, k - 1] = CYW
                                Mat_coefY[k, k + self.elements_circumferential] = CYN
                                Mat_coefY[k, k - self.elements_circumferential] = CYS

                                C = Mat_coef[k, :].T
                                C = np.delete(C, k)

                                CX = Mat_coefX[k, :].T
                                CY = Mat_coefY[k, :].T

                                BX = (
                                    np.matmul(CX, PP)
                                    + KXW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KXP * self.Theta_vol[ki, kj, n_p]
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                )

                                PX[ki, kj, n_p] = (BX - np.matmul(C, PPX)) / Mat_coef[
                                    k, k
                                ]
                                PY[ki, kj, n_p] = (BY - np.matmul(C, PPY)) / Mat_coef[
                                    k, k
                                ]

                            if (
                                ki == self.elements_axial - 1
                                and kj > 0
                                and kj < self.elements_circumferential - 1
                            ):
                                Mat_coef[k, k] = CP - CN
                                Mat_coef[k, k + 1] = CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - self.elements_circumferential] = CS

                                Mat_coefX[k, k] = CXP - CXN
                                Mat_coefX[k, k + 1] = CXE
                                Mat_coefX[k, k - 1] = CXW
                                Mat_coefX[k, k - self.elements_circumferential] = CXS

                                Mat_coefY[k, k] = CYP - CYN
                                Mat_coefY[k, k + 1] = CYE
                                Mat_coefY[k, k - 1] = CYW
                                Mat_coefY[k, k - self.elements_circumferential] = CYS

                                C = Mat_coef[k, :].T
                                C = np.delete(C, k)

                                CX = Mat_coefX[k, :].T
                                CY = Mat_coefY[k, :].T

                                BX = (
                                    np.matmul(CX, PP)
                                    + KXW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KXP * self.Theta_vol[ki, kj, n_p]
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                )

                                PX[ki, kj, n_p] = (BX - np.matmul(C, PPX)) / Mat_coef[
                                    k, k
                                ]
                                PY[ki, kj, n_p] = (BY - np.matmul(C, PPY)) / Mat_coef[
                                    k, k
                                ]

                            if ki == 0 and kj == self.elements_circumferential - 1:
                                Mat_coef[k, k] = CP - CE - CS
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k + self.elements_circumferential] = CN

                                Mat_coefX[k, k] = CXP - CXE - CXS
                                Mat_coefX[k, k - 1] = CXW
                                Mat_coefX[k, k + self.elements_circumferential] = CXN

                                Mat_coefY[k, k] = CYP - CYE - CYS
                                Mat_coefY[k, k - 1] = CYW
                                Mat_coefY[k, k + self.elements_circumferential] = CYN

                                C = Mat_coef[k, :].T
                                C = np.delete(C, k)

                                CX = Mat_coefX[k, :].T
                                CY = Mat_coefY[k, :].T

                                BX = (
                                    np.matmul(CX, PP)
                                    + KXW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KXP * self.Theta_vol[ki, kj, n_p]
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                )

                                PX[ki, kj, n_p] = (BX - np.matmul(C, PPX)) / Mat_coef[
                                    k, k
                                ]
                                PY[ki, kj, n_p] = (BY - np.matmul(C, PPY)) / Mat_coef[
                                    k, k
                                ]

                            if (
                                kj == self.elements_circumferential - 1
                                and ki > 0
                                and ki < self.elements_axial - 1
                            ):
                                Mat_coef[k, k] = CP - CE
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k + self.elements_circumferential] = CN
                                Mat_coef[k, k - self.elements_circumferential] = CS

                                Mat_coefX[k, k] = CXP - CXE
                                Mat_coefX[k, k - 1] = CXW
                                Mat_coefX[k, k + self.elements_circumferential] = CXN
                                Mat_coefX[k, k - self.elements_circumferential] = CXS

                                Mat_coefY[k, k] = CYP - CYE
                                Mat_coefY[k, k - 1] = CYW
                                Mat_coefY[k, k + self.elements_circumferential] = CYN
                                Mat_coefY[k, k - self.elements_circumferential] = CYS

                                C = Mat_coef[k, :].T
                                C = np.delete(C, k)

                                CX = Mat_coefX[k, :].T
                                CY = Mat_coefY[k, :].T

                                BX = (
                                    np.matmul(CX, PP)
                                    + KXW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KXP * self.Theta_vol[ki, kj, n_p]
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                )

                                PX[ki, kj, n_p] = (BX - np.matmul(C, PPX)) / Mat_coef[
                                    k, k
                                ]
                                PY[ki, kj, n_p] = (BY - np.matmul(C, PPY)) / Mat_coef[
                                    k, k
                                ]

                            if (
                                kj == self.elements_circumferential - 1
                                and ki == self.elements_axial - 1
                            ):
                                Mat_coef[k, k] = CP - CE - CN
                                Mat_coef[k, k - 1] = CW
                                Mat_coef[k, k - self.elements_circumferential] = CS

                                Mat_coefX[k, k] = CXP - CXE - CXN
                                Mat_coefX[k, k - 1] = CXW
                                Mat_coefX[k, k - self.elements_circumferential] = CXS

                                Mat_coefY[k, k] = CYP - CYE - CYN
                                Mat_coefY[k, k - 1] = CYW
                                Mat_coefY[k, k - self.elements_circumferential] = CYS

                                C = Mat_coef[k, :].T
                                C = np.delete(C, k)

                                CX = Mat_coefX[k, :].T
                                CY = Mat_coefY[k, :].T

                                BX = (
                                    np.matmul(CX, PP)
                                    + KXW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KXP * self.Theta_vol[ki, kj, n_p]
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.Theta_vol[ki, kj - 1, n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                )

                                PX[ki, kj, n_p] = (BX - np.matmul(C, PPX)) / Mat_coef[
                                    k, k
                                ]
                                PY[ki, kj, n_p] = (BY - np.matmul(C, PPY)) / Mat_coef[
                                    k, k
                                ]

                        k = k + 1
                        kj = kj + 1

                    kj = 0
                    ki = ki + 1

                erro = np.linalg.norm(PX - PX_old) + np.linalg.norm(PY - PY_old)

        PXdim = (
            PX
            * (self.reference_viscosity * self.frequency * (self.journal_radius**2))
            / (self.radial_clearance**3)
        )

        PYdim = (
            PY
            * (self.reference_viscosity * self.frequency * (self.journal_radius**2))
            / (self.radial_clearance**3)
        )

        PXPlot = np.zeros(
            (self.elements_axial, self.elements_circumferential * self.n_pad)
        ).astype(complex)
        PYPlot = np.zeros(
            (self.elements_axial, self.elements_circumferential * self.n_pad)
        ).astype(complex)

        for i in range(self.elements_axial):
            PXPlot[i] = PXdim[i, :, :].ravel("F")
            PYPlot[i] = PYdim[i, :, :].ravel("F")

        Ytheta = np.array(Ytheta).flatten()

        HX = HX.flatten()
        HY = HY.flatten()

        kxx = -np.real(
            np.trapz(
                np.trapz(PXPlot * HX, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        )

        kxy = -np.real(
            np.trapz(
                np.trapz(PYPlot * HX, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        )

        kyx = -np.real(
            np.trapz(
                np.trapz(PXPlot * HY, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        )

        kyy = -np.real(
            np.trapz(
                np.trapz(PYPlot * HY, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        )

        cxx = -np.imag(
            np.trapz(
                np.trapz(PXPlot * HX, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        ) / (gamma * self.frequency)

        cxy = -np.imag(
            np.trapz(
                np.trapz(PYPlot * HX, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        ) / (gamma * self.frequency)

        cyx = -np.imag(
            np.trapz(
                np.trapz(PXPlot * HY, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        ) / (gamma * self.frequency)

        cyy = -np.imag(
            np.trapz(
                np.trapz(PYPlot * HY, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        ) / (gamma * self.frequency)

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
            print(x)
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
                * self.frequency
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
        Ss = S

        return Ss

    def plot_bearing_representation(self, fig=None, rotation=90, **kwargs):
        """Plot the bearing representation.

        Parameters
        ----------
        rotation: float
            The default it is 90 degrees.
        subplots : Plotly graph_objects.make_subplots()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. plot_bgcolor="white", ...).
            *See Plotly Python make_subplot Reference for more information.

        Returns
        -------
            The figure object with the plot.
        """
        if fig is None:
            fig = go.Figure()

        groove = (360 / self.n_pad) - self.betha_s_dg
        hG = groove / 2

        pads = [hG, self.betha_s_dg, hG] * self.n_pad
        colors = ["#F5F5DC", "#929591", "#F5F5DC"] * self.n_pad

        fig = go.Figure(data=[go.Pie(values=pads, hole=0.85)])
        fig.update_traces(
            sort=False,
            hoverinfo="label",
            textinfo="none",
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=20)),
            rotation=rotation,
        )

        fig.add_annotation(
            x=self.load_x_direction,
            y=self.load_y_direction,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,  # style arrow
            arrowsize=2.5,
            arrowwidth=3,
            arrowcolor="green",
        )

        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            **kwargs,
        )
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)
        return fig

    def plot_pressure_distribution(self, axial_element_index=None, fig=None, **kwargs):
        """Plot pressure distribution.

        Parameters
        ----------
        axial_element_index : int, optional
            Show pressure distribution on bearing for the respective axial element.
            Default is the element closest to the middle of the bearing.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """

        if fig is None:
            fig = go.Figure()

        if axial_element_index is None:
            axial_element_index = self.elements_axial // 2

        n_elements = self.elements_circumferential

        # Plot bearing
        total_points = 1000  # Number of points to create a circle
        num_points = int(self.dtheta * n_elements * total_points / (2 * np.pi))

        thetaI = 0
        self.theta_p = []
        bearing_plot = []

        for n_p in range(self.n_pad):
            thetaF = self.thetaF[n_p]
            theta_ref = np.sort(np.arange(thetaF, thetaI, -self.dtheta))
            self.theta_p.append((theta_ref[0], theta_ref[-1]))
            thetaI = thetaF

            # Plot pad
            theta = np.linspace(self.theta_p[n_p][0], self.theta_p[n_p][1], num_points)
            x = np.cos(theta)
            y = np.sin(theta)

            bearing_plot.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=tableau_colors["gray"], width=6),
                    hoverinfo="text",
                    text=f"Pad {n_p}",
                    name=f"Pad {n_p} plot",
                )
            )

        P_distribution = self.P[axial_element_index, :, :]
        points = {"x": [], "y": []}
        pressure_plot = []

        for n_p in range(self.n_pad):
            # Plot the normals scaled by pressure
            scale = P_distribution[:, n_p] / np.max(np.abs(P_distribution)) * 0.5

            theta = np.arange(
                self.theta_p[n_p][0] + self.dtheta / 2,
                self.theta_p[n_p][1],
                self.dtheta,
            )
            x = np.cos(theta)
            y = np.sin(theta)

            for i in range(n_elements):
                x_i = x[i]
                y_i = y[i]
                x_f = x_i + scale[i] * np.cos(theta[i])
                y_f = y_i + scale[i] * np.sin(theta[i])

                angle = theta[i] * 180 / np.pi
                pressure = P_distribution[i, n_p]
                data_info = f"Pad {n_p}<br>Angle: {angle:.0f} deg<br>Pressure: {pressure:.3e} Pa"
                name = f"Pad {n_p} distribution"

                if abs(np.sqrt(x_f**2 + y_f**2) - 1) > 1e-2:
                    pressure_plot.append(
                        go.Scatter(
                            x=[x_i, x_f],
                            y=[y_i, y_f],
                            mode="lines+markers",
                            line=dict(width=3, color=tableau_colors["orange"]),
                            marker=dict(size=9, symbol="arrow", angleref="previous"),
                            hoverinfo="text",
                            text=data_info,
                            name=name,
                        )
                    )

                points["x"].append(x_f)
                points["y"].append(y_f)

        points["x"].append(points["x"][0])
        points["y"].append(points["y"][0])

        fig.add_traces(data=[*pressure_plot, *bearing_plot])

        # Plot distribution curve
        fig.add_trace(
            go.Scatter(
                x=points["x"],
                y=points["y"],
                mode="lines",
                line_shape="spline",
                line=dict(color="black", width=1.5, dash="dash"),
                hoverinfo="none",
                name="Distribution curve",
            )
        )

        P_min = np.min(P_distribution)
        P_max = np.max(P_distribution)
        fig.add_annotation(
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            align="right",
            showarrow=False,
            font=dict(size=16, color="black"),
            text=f"<b>Pressure Distribution</b><br>Min: {P_min:.3e} Pa<br>Max: {P_max:.3e} Pa",
        )

        fig.update_layout(
            title="Cylindrical Bearing",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
            ),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=800,
            showlegend=False,
            **kwargs,
        )

        return fig


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
        n=3,
        axial_length=0.263144,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_circumferential=11,
        elements_axial=3,
        n_pad=2,
        pad_arc_length=176,
        preload=0,
        geometry="circular",
        reference_temperature=50,
        frequency=Q_([900], "RPM"),
        load_x_direction=0,
        load_y_direction=-112814.91,
        groove_factor=[0.52, 0.48],
        lubricant="ISOVG32",
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        operating_type="flooded",
        injection_pressure=0,
        oil_flow=37.86,
        show_coef=False,
        print_result=False,
        print_progress=False,
        print_time=False,
    )

    return bearing
