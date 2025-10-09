import time
import numpy as np
from numpy.linalg import norm
from scipy.optimize import curve_fit, minimize
from plotly import graph_objects as go
from numba import njit
from scipy import sparse

from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from ross.plotly_theme import tableau_colors
from ross.bearings.lubricants import lubricants_dict


class PlainJournal(BearingElement):
    """Plain journal bearing - Advanced thermo-hydro-dynamic model.

    This class provides a **comprehensive numerical bearing model** for detailed design
    and optimization. Solves full Reynolds equation with thermal effects on a discretized
    grid to calculate pressure, temperature fields, and bearing coefficients.

    **When to use this class:**
    - Detailed bearing design and optimization
    - Thermo-hydro-dynamic (THD) analysis
    - Complex bearing geometries (circular, lobe, elliptical)
    - Multi-pad configurations with preload
    - High-speed applications requiring turbulence models
    - Oil starvation or flooded conditions
    - When accuracy is more important than computational speed

    **For quick calculations and preliminary design, consider using CylindricalBearing
    instead, which provides:**
    - Fast analytical solutions
    - Simple geometry assumptions
    - Closed-form coefficients

    The basic references for the code are found in :cite:t:`barbosa2018`,
    :cite:t:`daniel2012` and :cite:t:`nicoletti1999`.

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
    model_type : str, optional
        Type of model to be used. Options:
        - 'thermo_hydro_dynamic': Thermo-Hydro-Dynamic model
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
    frequency : list, pint.Quantity
        Array with the frequencies (rad/s).
    fxs_load : float, pint.Quantity
        Load in X direction. The unit is newton.
    fys_load : float, pint.Quantity
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
    oil_flow_v: float, pint.Quantity
        Suply oil flow to bearing. Only used when operating type 'starvation' is
        selected. Default unit is meter**3/second
    oil_supply_pressure: float, Pint.Quantity
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
    A PlainJournal object.

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

    Example
    --------
    >>> from ross.bearings.plain_journal import PlainJournal
    >>> bearing = PlainJournal(
    ...    n=3,
    ...    axial_length=0.263144,
    ...    journal_radius=0.2,
    ...    radial_clearance=1.95e-4,
    ...    elements_circumferential=11,
    ...    elements_axial=3,
    ...    n_pad=2,
    ...    pad_arc_length=176,
    ...    preload=0,
    ...    geometry="circular",
    ...    reference_temperature=50,
    ...    frequency=Q_([900], "RPM"),
    ...    fxs_load=0,
    ...    fys_load=-112814.91,
    ...    groove_factor=[0.52, 0.48],
    ...    lubricant="ISOVG32",
    ...    sommerfeld_type=2,
    ...    initial_guess=[0.1, -0.1],
    ...    method="perturbation",
    ...    operating_type="flooded",
    ...    oil_supply_pressure=0,
    ...    oil_flow_v=Q_(37.86, "l/min"),
    ...    show_coeffs=False,
    ...    print_result=False,
    ...    print_progress=False,
    ...    print_time=False,
    ... )
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
        fxs_load,
        fys_load,
        groove_factor,
        lubricant,
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        model_type="thermo_hydro_dynamic",
        operating_type="flooded",
        oil_supply_pressure=None,
        oil_flow_v=None,
        show_coeffs=False,
        print_result=False,
        print_progress=False,
        print_time=False,
        **kwargs,
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
        self.fxs_load = fxs_load
        self.fys_load = fys_load
        self.lubricant = lubricant
        self.fat_mixt = np.array(groove_factor)
        self.equilibrium_pos = None
        self.sommerfeld_type = sommerfeld_type
        self.initial_guess = initial_guess
        self.method = method
        self.operating_type = operating_type
        self.model_type = model_type
        self.oil_supply_pressure = oil_supply_pressure
        self.oil_flow_v = oil_flow_v
        self.show_coeffs = show_coeffs
        self.print_result = print_result
        self.print_progress = print_progress
        self.print_time = print_time

        self.betha_s_dg = pad_arc_length
        self.betha_s = pad_arc_length * np.pi / 180

        self.thetaI = 0
        self.thetaF = self.betha_s
        self.dtheta = (self.thetaF - self.thetaI) / (self.elements_circumferential)

        pad_ct = np.arange(0, 360, int(360 / self.n_pad))
        self.thetaI = np.radians(pad_ct + 180 / self.n_pad - self.betha_s_dg / 2)
        self.thetaF = np.radians(pad_ct + 180 / self.n_pad + self.betha_s_dg / 2)
        self.theta_range = [
            np.arange(start_rad + (self.dtheta / 2), end_rad, self.dtheta)
            for start_rad, end_rad in zip(self.thetaI, self.thetaF)
        ]

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

        # Interpolation for viscosity
        a, b = self._get_interp_coeffs(
            T_muI, T_muF, mu_I, mu_F
        )  # Interpolation coefficients
        self.interpolate = lambda reference: a * (
            reference**b
        )  # Interpolation function
        self.reference_viscosity = self.interpolate(self.reference_temperature)

        # Pivot angle for lobe geometry
        self.theta_pivot = np.array([90, 270]) * np.pi / 180

        n_freq = np.shape(frequency)[0]

        kxx = np.zeros(n_freq)
        kxy = np.zeros(n_freq)
        kyx = np.zeros(n_freq)
        kyy = np.zeros(n_freq)

        cxx = np.zeros(n_freq)
        cxy = np.zeros(n_freq)
        cyx = np.zeros(n_freq)
        cyy = np.zeros(n_freq)

        for i in range(n_freq):
            speed = frequency[i]

            if self.model_type == "thermo_hydro_dynamic":
                self.run_thermo_hydro_dynamic(speed)

            coeffs = self.coefficients(speed)
            kxx[i], kxy[i], kyx[i], kyy[i] = coeffs[0]
            cxx[i], cxy[i], cyx[i], cyy[i] = coeffs[1]

        super().__init__(
            n, kxx, cxx, kyy, kxy, kyx, cyy, cxy, cyx, frequency=frequency, **kwargs
        )

    def _forces(self, initial_guess, speed, y0=None, xpt0=None, ypt0=None):
        """Calculates the forces in Y and X direction.

        Parameters
        ----------
        initial_guess : array, float
            If the other parameters are None, initial_guess is an array with eccentricity
            ratio and attitude angle. Else, initial_guess is the position of the center of
            the rotor in the x-axis.
        speed : float
            Rotor speed. The unit is rad/s.
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

            self.Xpt = xpt0 / (self.radial_clearance * speed)
            self.Ypt = ypt0 / (self.radial_clearance * speed)

        shape_2d = (self.elements_axial, self.elements_circumferential)
        shape_3d = (self.elements_axial, self.elements_circumferential, self.n_pad)
        nk = self.elements_axial * self.elements_circumferential

        self.theta_vol_groove = 0.8 * np.ones(self.n_pad)
        T_end = np.ones(self.n_pad)
        T_conv = 0.8 * self.reference_temperature
        T_mist = self.reference_temperature * np.ones(self.n_pad)

        Qedim = np.ones(self.n_pad)
        Qsdim = np.ones(self.n_pad)

        Pdim = np.zeros(shape_3d)
        P = np.zeros(shape_3d)
        T = np.ones(shape_3d)
        Tdim = np.ones(shape_3d)
        T_new = np.ones(shape_3d) * 1.2
        Theta_vol = np.zeros(shape_3d)

        U = 0.5 * np.ones(shape_3d)
        mu_new = 1.1 * np.ones(shape_3d)
        mu_turb = 1.3 * np.ones(shape_3d)

        H = np.ones((self.elements_circumferential, self.n_pad))

        p = np.ones((nk, 1))
        theta_vol = np.zeros((nk, 1))

        # Coefficient matrices
        Mat_coef_st = np.zeros((nk, nk))
        Mat_coef_T = np.zeros((nk, nk))
        Mat_coef = np.zeros((nk, nk))

        # Source terms arrays
        b_T = np.zeros((nk, 1))
        b_P = np.zeros((nk, 1))
        B = np.zeros((nk, 1))
        B_theta = np.zeros((nk, 1))

        while (T_mist[0] - T_conv) >= 0.5:
            T_conv = T_mist[0]

            P[:, :, :] = 0.0
            T[:, :, :] = 1.0
            Tdim[:, :, :] = 1.0
            T_new[:, :, :] = 1.2
            Theta_vol[:, :, :] = 0.0

            U[:, :, :] = 0.5
            mu_new[:, :, :] = 1.1
            mu_turb[:, :, :] = 1.3

            H[:, :] = 1.0

            Qedim[:] = 1.0
            Qsdim[:] = 1.0

            Mat_coef[:, :] = 0.0
            b_T[:] = 0.0
            b_P[:] = 0.0
            B[:] = 0.0

            for n_p in np.arange(self.n_pad):
                T_ref = T_mist[n_p]

                while (
                    norm(T_new[:, :, n_p] - T[:, :, n_p]) / norm(T[:, :, n_p]) >= 0.01
                ):
                    T_ref = T_mist[n_p]
                    T[:, :, n_p] = T_new[:, :, n_p]
                    self.mu_l = mu_new

                    if self.operating_type == "flooded":
                        Mat_coef, b_P = _flooded(
                            Mat_coef,
                            b_P,
                            H[:, n_p],
                            self.mu_l[:, :, n_p],
                            self.theta_range[n_p],
                            self.dtheta,
                            self.elements_axial,
                            self.elements_circumferential,
                            self.X,
                            self.Y,
                            self.dY,
                            self.dZ,
                            self.Xpt,
                            self.Ypt,
                            self.betha_s,
                            self.axial_length,
                            self.journal_radius,
                            self.geometry,
                            self.preload,
                            self.theta_pivot[n_p],
                        )

                        P_sol = _solve(Mat_coef, b_P)
                        P_sol = np.where(P_sol < 0, 0, P_sol).reshape(shape_2d)

                    elif self.operating_type == "starvation":
                        p[:] = 1.0
                        theta_vol[:] = 0.0
                        B_theta[:] = 0.0
                        Mat_coef_st[:, :] = 0.0
                        Theta_vol[:, :, n_p] = 0.0

                        P_sol, Theta_vol_sol = _starvation(
                            p,
                            theta_vol,
                            Mat_coef_st,
                            B,
                            B_theta,
                            H[:, n_p],
                            self.mu_l[:, :, n_p],
                            self.theta_vol_groove[n_p],
                            self.theta_range[n_p],
                            self.dtheta,
                            self.elements_axial,
                            self.elements_circumferential,
                            self.X,
                            self.Y,
                            self.dY,
                            self.dZ,
                            self.Xpt,
                            self.Ypt,
                            self.betha_s,
                            self.oil_supply_pressure,
                            self.axial_length,
                            self.journal_radius,
                            self.geometry,
                            self.preload,
                            self.theta_pivot[n_p],
                        )

                        Theta_vol[:, :, n_p] = Theta_vol_sol

                    P[:, :, n_p] = P_sol

                    Pdim[:, :, n_p] = (
                        P_sol
                        * self.reference_viscosity
                        * speed
                        * (self.journal_radius**2)
                    ) / (self.radial_clearance**2)

                    Mat_coef_T[:, :] = 0.0

                    Mat_coef_T, b_T = _temperature(
                        Mat_coef_T,
                        b_T,
                        T_ref,
                        P[:, :, n_p],
                        U[:, :, n_p],
                        H[:, n_p],
                        Theta_vol[:, :, n_p],
                        self.theta_range[n_p],
                        self.mu_l[:, :, n_p],
                        mu_turb[:, :, n_p],
                        speed,
                        self.reference_temperature,
                        self.reference_viscosity,
                        self.rho,
                        self.Cp,
                        self.k_t,
                        self.elements_axial,
                        self.elements_circumferential,
                        self.dY,
                        self.dZ,
                        self.Xpt,
                        self.Ypt,
                        self.betha_s,
                        self.axial_length,
                        self.journal_radius,
                        self.radial_clearance,
                        self.operating_type,
                    )

                    T_sol = _solve(Mat_coef_T, b_T).reshape(shape_2d)
                    T_new[:, :, n_p] = T_sol
                    Tdim[:, :, n_p] = T_sol * self.reference_temperature

                    mu_new[:, :, n_p] = (
                        self.interpolate(Tdim[:, :, n_p]) / self.reference_viscosity
                    )

                T_end[n_p] = np.sum(Tdim[:, -1, n_p]) / self.elements_axial

                if self.operating_type == "flooded":
                    T_mist[n_p - 1] = (
                        self.fat_mixt[n_p] * self.reference_temperature
                        + (1 - self.fat_mixt[n_p]) * T_end[n_p]
                    )

                if self.operating_type == "starvation":
                    Qedim[n_p] = (
                        self.radial_clearance
                        * H[0, n_p]
                        * speed
                        * self.journal_radius
                        * self.axial_length
                        * Theta_vol[0, 0, n_p]
                        * (np.mean(U[:, 0, n_p]))
                    )

                    Qsdim[n_p] = (
                        self.radial_clearance
                        * H[-1, n_p]
                        * speed
                        * self.journal_radius
                        * self.axial_length
                        * Theta_vol[0, -1, n_p]
                        * (np.mean(U[:, -1, n_p]))
                    )

            if self.operating_type == "starvation":
                for n_p in np.arange(self.n_pad):
                    geometry_factor = (Qedim[n_p] + Qsdim[n_p - 1]) / (
                        np.sum(Qedim) + np.sum(Qsdim)
                    )

                    T_mist[n_p] = (
                        (Qsdim[n_p - 1] * T_end[n_p - 1])
                        + (
                            self.reference_temperature
                            * geometry_factor
                            * self.oil_flow_v
                        )
                    ) / (geometry_factor * self.oil_flow_v + Qsdim[n_p - 1])

                    self.theta_vol_groove[n_p] = (
                        0.8
                        * (geometry_factor * self.oil_flow_v + Qsdim[n_p - 1])
                        / Qedim[n_p]
                    )

                    if self.theta_vol_groove[n_p] > 1:
                        self.theta_vol_groove[n_p] = 1

        # self.P must receive the adimensional pressure field
        self.P = (
            Pdim
            * (self.radial_clearance**2)
            / (self.reference_viscosity * speed * (self.journal_radius**2))
        )
        self.Theta_vol = Theta_vol

        # Reshape dimensional pressure field from (axial, circumferential, pads) to (axial, all_other_dims)
        PPlot = Pdim.reshape(self.elements_axial, -1, order="F")

        Ytheta = np.sort(
            np.linspace(
                self.thetaI + self.dtheta / 2,
                self.thetaF - self.dtheta / 2,
                self.elements_circumferential,
            ).ravel()
        )

        fx1 = np.trapezoid(PPlot * np.cos(Ytheta), self.journal_radius * Ytheta)
        fy1 = np.trapezoid(PPlot * np.sin(Ytheta), self.journal_radius * Ytheta)

        z_vals = self.axial_length * self.Z[1:-1]
        Fhx = -np.trapezoid(fx1, z_vals)
        Fhy = -np.trapezoid(fy1, z_vals)

        return Fhx, Fhy

    def run_thermo_hydro_dynamic(self, speed):
        """This method runs the optimization to find the equilibrium position of
        the rotor's center.
        """

        args = (speed, self.print_progress)
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
            print(f"Time Spent: {t2 - t1} seconds")

    def _get_interp_coeffs(self, T_muI, T_muF, mu_I, mu_F):
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
        a, b: Float
            Coefficients of the curve viscosity vs temperature.
        """

        def viscosity(x, a, b):
            return a * (x**b)

        xdata = [T_muI, T_muF]  # changed boundary conditions to avoid division by zero
        ydata = [mu_I, mu_F]

        popt, pcov = curve_fit(viscosity, xdata, ydata, p0=(6.0, 1.0))
        a, b = popt

        return a, b

    @check_units
    def coefficients(self, speed):
        """Calculates the dynamic coefficients of stiffness "k" and damping "c".
        Basic reference is found at :cite:t:`lund1978`

        Parameters
        ----------
        speed : float, pint.Quantity
            Rotational speed to evaluate coefficients. The unit is rad/s.

        Returns
        -------
        coeffs : tuple
            Bearing stiffness and damping coefficients.
            Its shape is: ((kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy))

        References
        ----------
        .. bibliography::
            :filter: docname in docnames
        """

        if self.equilibrium_pos is None:
            self.run_thermo_hydro_dynamic(speed)
            self.coefficients(speed)
        else:
            if self.method == "lund":
                k, c = self._lund_method(speed)
            elif self.method == "perturbation":
                k, c = self._perturbation_method(speed)

            if self.show_coeffs:
                print(f"kxx = {k[0]}")
                print(f"kxy = {k[1]}")
                print(f"kyx = {k[2]}")
                print(f"kyy = {k[3]}")

                print(f"cxx = {c[0]}")
                print(f"cxy = {c[1]}")
                print(f"cyx = {c[2]}")
                print(f"cyy = {c[3]}")

            coeffs = (k, c)

            return coeffs

    def _perturbation_method(self, speed):
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

        Va = speed * (self.journal_radius)
        epixpt = 0.000001 * np.abs(Va * np.sin(self.equilibrium_pos[1]))
        epiypt = 0.000001 * np.abs(Va * np.cos(self.equilibrium_pos[1]))

        Auinitial_guess1 = self._forces(xeq + epix, speed, yeq, 0, 0)
        Auinitial_guess2 = self._forces(xeq - epix, speed, yeq, 0, 0)
        Auinitial_guess3 = self._forces(xeq, speed, yeq + epiy, 0, 0)
        Auinitial_guess4 = self._forces(xeq, speed, yeq - epiy, 0, 0)

        Auinitial_guess5 = self._forces(xeq, speed, yeq, epixpt, 0)
        Auinitial_guess6 = self._forces(xeq, speed, yeq, -epixpt, 0)
        Auinitial_guess7 = self._forces(xeq, speed, yeq, 0, epiypt)
        Auinitial_guess8 = self._forces(xeq, speed, yeq, 0, -epiypt)

        Kxx = -self.sommerfeld(speed, Auinitial_guess1[0], Auinitial_guess2[1]) * (
            (Auinitial_guess1[0] - Auinitial_guess2[0]) / (epix / self.radial_clearance)
        )
        Kxy = -self.sommerfeld(speed, Auinitial_guess3[0], Auinitial_guess4[1]) * (
            (Auinitial_guess3[0] - Auinitial_guess4[0]) / (epiy / self.radial_clearance)
        )
        Kyx = -self.sommerfeld(speed, Auinitial_guess1[1], Auinitial_guess2[1]) * (
            (Auinitial_guess1[1] - Auinitial_guess2[1]) / (epix / self.radial_clearance)
        )
        Kyy = -self.sommerfeld(speed, Auinitial_guess3[1], Auinitial_guess4[1]) * (
            (Auinitial_guess3[1] - Auinitial_guess4[1]) / (epiy / self.radial_clearance)
        )

        Cxx = -self.sommerfeld(speed, Auinitial_guess5[0], Auinitial_guess6[1]) * (
            (Auinitial_guess5[0] - Auinitial_guess6[0])
            / (epixpt / self.radial_clearance / speed)
        )
        Cxy = -self.sommerfeld(speed, Auinitial_guess7[0], Auinitial_guess8[1]) * (
            (Auinitial_guess7[0] - Auinitial_guess8[0])
            / (epiypt / self.radial_clearance / speed)
        )
        Cyx = -self.sommerfeld(speed, Auinitial_guess5[0], Auinitial_guess6[1]) * (
            (Auinitial_guess5[1] - Auinitial_guess6[1])
            / (epixpt / self.radial_clearance / speed)
        )
        Cyy = -self.sommerfeld(speed, Auinitial_guess7[0], Auinitial_guess8[1]) * (
            (Auinitial_guess7[1] - Auinitial_guess8[1])
            / (epiypt / self.radial_clearance / speed)
        )

        ratio = np.sqrt((self.fxs_load**2) + (self.fys_load**2)) / self.radial_clearance

        kxx = ratio * Kxx
        kxy = ratio * Kxy
        kyx = ratio * Kyx
        kyy = ratio * Kyy

        ratio *= 1 / speed

        cxx = ratio * Cxx
        cxy = ratio * Cxy
        cyx = ratio * Cyx
        cyy = ratio * Cyy

        return (kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy)

    def _lund_method(self, speed):
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
                                    + 2 * CXW * self.oil_supply_pressure
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.theta_vol_groove[n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                    + 2 * CYW * self.oil_supply_pressure
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
                                    + 2 * CXW * self.oil_supply_pressure
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.theta_vol_groove[n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                    + 2 * CYW * self.oil_supply_pressure
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
                                    + 2 * CXW * self.oil_supply_pressure
                                )
                                BY = (
                                    np.matmul(CY, PP)
                                    + KYW * self.theta_vol_groove[n_p]
                                    + KYP * self.Theta_vol[ki, kj, n_p]
                                    + 2 * CYW * self.oil_supply_pressure
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

                erro = norm(PX - PX_old) + norm(PY - PY_old)

        PXdim = (
            PX
            * (self.reference_viscosity * speed * (self.journal_radius**2))
            / (self.radial_clearance**3)
        )

        PYdim = (
            PY
            * (self.reference_viscosity * speed * (self.journal_radius**2))
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
            np.trapezoid(
                np.trapezoid(PXPlot * HX, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        )

        kxy = -np.real(
            np.trapezoid(
                np.trapezoid(PYPlot * HX, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        )

        kyx = -np.real(
            np.trapezoid(
                np.trapezoid(PXPlot * HY, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        )

        kyy = -np.real(
            np.trapezoid(
                np.trapezoid(PYPlot * HY, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        )

        cxx = -np.imag(
            np.trapezoid(
                np.trapezoid(PXPlot * HX, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        ) / (gamma * speed)

        cxy = -np.imag(
            np.trapezoid(
                np.trapezoid(PYPlot * HX, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        ) / (gamma * speed)

        cyx = -np.imag(
            np.trapezoid(
                np.trapezoid(PXPlot * HY, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        ) / (gamma * speed)

        cyy = -np.imag(
            np.trapezoid(
                np.trapezoid(PYPlot * HY, Ytheta * self.journal_radius),
                self.Zdim[1 : self.elements_axial + 1],
            )
        ) / (gamma * speed)

        return (kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy)

    def _score(self, x, speed, print_progress=False):
        """This method used to set the objective function of minimize optimization.

        Parameters
        ----------
        x: array
           Balanced Force expression between the load aplied in bearing and the
           resultant force provide by oil film.
        speed: float
            Rotational speed to evaluate bearing coefficients. The unit is rad/s.

        Returns
        -------
        Score coefficient.
        """

        Fhx, Fhy = self._forces(x, speed)
        score = np.sqrt(((self.fxs_load + Fhx) ** 2) + ((self.fys_load + Fhy) ** 2))
        if print_progress:
            print(x)
            print(f"Score: ", score)
            print("============================================")
            print(f"Force x direction: ", Fhx)
            print("============================================")
            print(f"Force y direction: ", Fhy)
            print("")

        return score

    def sommerfeld(self, speed, force_x, force_y):
        """Calculate the sommerfeld number. This dimensionless number is used to
        calculate the dynamic coeficients.

        Parameters
        ----------
        speed : float
            Rotor speed. The unit is rad/s.
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
                * speed
            ) / (
                np.pi
                * (self.radial_clearance**2)
                * np.sqrt((self.fxs_load**2) + (self.fys_load**2))
            )

        elif self.sommerfeld_type == 2:
            S = 1 / (
                2
                * ((self.axial_length / (2 * self.journal_radius)) ** 2)
                * (np.sqrt((force_x**2) + (force_y**2)))
            )

        # Ss = S * ((self.axial_length / (2 * self.journal_radius)) ** 2)
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
            x=self.fxs_load,
            y=self.fys_load,
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
            title="Plain Journal Bearing",
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


@njit
def _evaluate_bearing_clearance(X, Y, theta, dtheta, geometry, preload, theta_pivot):
    if geometry == "circular":
        hp = 1 - X * np.cos(theta) - Y * np.sin(theta)
        he = 1 - X * np.cos(theta + 0.5 * dtheta) - Y * np.sin(theta + 0.5 * dtheta)
        hw = 1 - X * np.cos(theta - 0.5 * dtheta) - Y * np.sin(theta - 0.5 * dtheta)

    elif geometry == "lobe":
        hp = (
            1 / (1 - preload)
            - preload / (1 - preload) * np.cos(theta - theta_pivot)
            - X * np.cos(theta)
            - Y * np.sin(theta)
        )
        he = (
            1 / (1 - preload)
            - preload / (1 - preload) * np.cos(theta + 0.5 * dtheta - theta_pivot)
            - X * np.cos(theta + 0.5 * dtheta)
            - Y * np.sin(theta + 0.5 * dtheta)
        )
        hw = (
            1 / (1 - preload)
            - preload / (1 - preload) * np.cos(theta - 0.5 * dtheta - theta_pivot)
            - X * np.cos(theta - 0.5 * dtheta)
            - Y * np.sin(theta - 0.5 * dtheta)
        )

    elif geometry == "elliptical":
        hp = (
            1
            + preload / (1 - preload) * (np.cos(theta)) ** 2
            - X * np.cos(theta)
            - Y * np.sin(theta)
        )
        he = (
            1
            + preload / (1 - preload) * (np.cos(theta + 0.5 * dtheta)) ** 2
            - X * np.cos(theta + 0.5 * dtheta)
            - Y * np.sin(theta + 0.5 * dtheta)
        )
        hw = (
            1
            + preload / (1 - preload) * (np.cos(theta - 0.5 * dtheta)) ** 2
            - X * np.cos(theta - 0.5 * dtheta)
            - Y * np.sin(theta - 0.5 * dtheta)
        )

    hn = hp
    hs = hn

    return he, hw, hn, hs, hp


@njit
def _calculate_discretization_coeffs(
    kj,
    ki,
    elm_cir,
    elm_axi,
    dY,
    dZ,
    journal_radius,
    axial_length,
    mu,
    beta_s,
    he,
    hw,
    hn,
    hs,
):
    mu_p = mu[ki, kj]

    if kj == 0:
        MU_w = mu_p
        MU_e = 0.5 * (mu_p + mu[ki, kj + 1])
    elif kj == elm_cir - 1:
        MU_w = 0.5 * (mu_p + mu[ki, kj - 1])
        MU_e = mu_p
    else:
        MU_w = 0.5 * (mu_p + mu[ki, kj - 1])
        MU_e = 0.5 * (mu_p + mu[ki, kj + 1])

    if ki == 0:
        MU_s = mu_p
        MU_n = 0.5 * (mu_p + mu[ki + 1, kj])
    elif ki == elm_axi - 1:
        MU_s = 0.5 * (mu_p + mu[ki - 1, kj])
        MU_n = mu_p
    else:
        MU_s = 0.5 * (mu_p + mu[ki - 1, kj])
        MU_n = 0.5 * (mu_p + mu[ki + 1, kj])

    aux_0 = dZ / (12 * dY * beta_s**2)
    aux_1 = dY * journal_radius**2 / (12 * dZ * axial_length**2)

    CE = aux_0 * he**3 / MU_e
    CW = aux_0 * hw**3 / MU_w
    CN = aux_1 * hn**3 / MU_n
    CS = aux_1 * hs**3 / MU_s
    CP = -(CE + CW + CN + CS)

    return CE, CW, CN, CS, CP


@njit
def _flooded(
    A_P,
    b_P,
    film_thickness,
    mu,
    theta_range,
    dtheta,
    elm_axi,
    elm_cir,
    X,
    Y,
    dY,
    dZ,
    Xpt,
    Ypt,
    beta_s,
    axial_length,
    journal_radius,
    geometry,
    preload,
    theta_pivot,
):
    kj = 0
    k = 0

    for ki in range(elm_axi):
        for theta in theta_range:
            he, hw, hn, hs, hp = _evaluate_bearing_clearance(
                X, Y, theta, dtheta, geometry, preload, theta_pivot
            )

            film_thickness[kj] = hp

            b_P[k, 0] = (dZ / (2 * beta_s)) * (he - hw) - (
                (Xpt * np.cos(theta) + Ypt * np.sin(theta)) * dY * dZ
            )

            CE, CW, CN, CS, CP = _calculate_discretization_coeffs(
                kj,
                ki,
                elm_cir,
                elm_axi,
                dY,
                dZ,
                journal_radius,
                axial_length,
                mu,
                beta_s,
                he,
                hw,
                hn,
                hs,
            )

            A_P[k, k] = CP

            if ki == 0:
                A_P[k, k] -= CS
            else:
                A_P[k, k - elm_cir] = CS

            if ki == elm_axi - 1:
                A_P[k, k] -= CN
            else:
                A_P[k, k + elm_cir] = CN

            if kj == 0:
                A_P[k, k] -= CW
            else:
                A_P[k, k - 1] = CW

            if kj == elm_cir - 1:
                A_P[k, k] -= CE
            else:
                A_P[k, k + 1] = CE

            k += 1
            kj += 1

        kj = 0

    return A_P, b_P


@njit
def _starvation(
    P,
    theta_vol,
    A_P,
    b_P,
    b_theta,
    film_thickness,
    mu,
    theta_vol_groove,
    theta_range,
    dtheta,
    elm_axi,
    elm_cir,
    X,
    Y,
    dY,
    dZ,
    Xpt,
    Ypt,
    beta_s,
    injection_pressure,
    axial_length,
    journal_radius,
    geometry,
    preload,
    theta_pivot,
):
    k_range = np.arange(elm_axi * elm_cir)

    CW = np.zeros((elm_axi * elm_cir))
    KP = np.zeros((elm_axi * elm_cir))
    KW = np.zeros((elm_axi * elm_cir))

    k = 0
    kj = 0

    for ki in range(elm_axi):
        for theta in theta_range:
            he, hw, hn, hs, hp = _evaluate_bearing_clearance(
                X, Y, theta, dtheta, geometry, preload, theta_pivot
            )

            film_thickness[kj] = hp

            CE, CW[k], CN, CS, CP = _calculate_discretization_coeffs(
                kj,
                ki,
                elm_cir,
                elm_axi,
                dY,
                dZ,
                journal_radius,
                axial_length,
                mu,
                beta_s,
                he,
                hw,
                hn,
                hs,
            )

            # Source term
            KP1 = -(dZ / (2 * beta_s)) * he
            hpt = -Xpt * np.cos(theta) - Ypt * np.sin(theta)
            KP2 = -hpt * dY * dZ

            KP[k] = KP1 + KP2
            KW[k] = (dZ / (2 * beta_s)) * hw

            A_P[k, k] = CP

            if ki == 0:
                A_P[k, k] -= CS
            else:
                A_P[k, k - elm_cir] = CS

            if ki == elm_axi - 1:
                A_P[k, k] -= CN
            else:
                A_P[k, k + elm_cir] = CN

            if kj == 0:
                A_P[k, k] -= CW[k]
            else:
                A_P[k, k - 1] = CW[k]

            if kj == elm_cir - 1:
                A_P[k, k] -= CE
            else:
                A_P[k, k + 1] = CE

            k += 1
            kj += 1

        kj = 0

    tol = 1e-2
    res = 1.0

    while res >= tol:
        P_old = P.copy()
        theta_vol_old = theta_vol.copy()

        k = 0

        for ki in range(elm_axi):
            # kj == 0:
            if P[k] > 0:
                theta_vol[k] = 1
                b_P[k] = (
                    -KP[k] * theta_vol[k]
                    - KW[k] * theta_vol_groove
                    - 2 * CW[k] * injection_pressure
                )
                rmvk = k_range != k
                P[k] = (b_P[k] - A_P[k, rmvk] @ P[rmvk]) / A_P[k, k]

            else:
                P[k] = 0
                b_theta[k] = -A_P[k, :] @ P
                theta_vol[k] = (b_theta[k] - KW[k] * theta_vol_groove) / KP[k]
            k += 1

            # kj != 0:
            for theta in theta_range[1:]:
                if P[k] > 0:
                    theta_vol[k] = 1
                    b_P[k] = -KP[k] * theta_vol[k] - KW[k] * theta_vol[k - 1]
                    rmvk = k_range != k
                    P[k] = (b_P[k] - A_P[k, rmvk] @ P[rmvk]) / A_P[k, k]

                else:
                    P[k] = 0
                    b_theta[k] = -A_P[k, :] @ P
                    theta_vol[k] = (b_theta[k] - KW[k] * theta_vol[k - 1]) / KP[k]
                k += 1

        res = norm(P - P_old) + norm(theta_vol - theta_vol_old)

    P = np.where(P < 0, 0, P).reshape((elm_axi, elm_cir))
    theta_vol = theta_vol.reshape((elm_axi, elm_cir))

    return P, theta_vol


@njit
def _compute_turbulence_props(
    film_thickness,
    speed,
    dPdy,
    dPdz,
    rho,
    mu,
    mu_t,
    reference_viscosity,
    beta_s,
    axial_length,
    journal_radius,
    radial_clearance,
    operating_type,
    theta_vol,
):
    Reyn = (
        rho
        * speed
        * journal_radius
        * (film_thickness / axial_length)
        * radial_clearance
        / (reference_viscosity * mu)
    )

    if operating_type == "starvation":
        Reyn *= theta_vol

    delta_turb = 0
    if Reyn > 500 and Reyn <= 1000:
        delta_turb = 1 - ((1000 - Reyn) / 500) ** (1 / 8)
    elif Reyn > 1000:
        delta_turb = 1

    dudy = ((film_thickness / mu_t) * dPdy) - (speed / film_thickness)
    dwdy = (film_thickness / mu_t) * dPdz

    tau_w = mu_t * np.sqrt((dudy**2) + (dwdy**2))
    u_s = (abs(tau_w) / rho) ** 0.5
    nu = reference_viscosity * mu_t / rho
    y_w = (2 * film_thickness * radial_clearance) / nu * u_s

    emv = 0.4 * (y_w - (10.7 * np.tanh(y_w / 10.7)))
    mu_t = mu * (1 + (delta_turb * emv))

    U = 0.5 - (film_thickness**2) / (12 * mu_t * beta_s) * dPdy

    return U, mu_t


@njit
def _temperature(
    A_T,
    b_T,
    T_ref,
    P,
    U,
    film_thickness,
    Theta_vol,
    theta_range,
    mu,
    mu_turb,
    speed,
    reference_temperature,
    reference_viscosity,
    rho,
    Cp,
    k_t,
    elm_axi,
    elm_cir,
    dY,
    dZ,
    Xpt,
    Ypt,
    beta_s,
    axial_length,
    journal_radius,
    radial_clearance,
    operating_type,
):
    kj = 0
    k = 0

    for ki in range(elm_axi):
        for theta in theta_range:
            h = film_thickness[kj]

            # Pressure gradient
            p_west = 0.0
            p_east = 0.0
            p_south = 0.0
            p_north = 0.0

            if kj > 0:
                p_west = P[ki, kj - 1]
            if kj < elm_cir - 1:
                p_east = P[ki, kj + 1]

            if ki > 0:
                p_south = P[ki - 1, kj]
            if ki < elm_axi - 1:
                p_north = P[ki + 1, kj]

            dPdy = (p_east - p_west) / (2 * dY)
            dPdz = (p_north - p_south) / (2 * dZ)

            U[ki, kj], mu_t = _compute_turbulence_props(
                h,
                speed,
                dPdy,
                dPdz,
                rho,
                mu[ki, kj],
                mu_turb[ki, kj],
                reference_viscosity,
                beta_s,
                axial_length,
                journal_radius,
                radial_clearance,
                operating_type,
                Theta_vol[ki, kj],
            )

            mu_turb[ki, kj] = mu_t

            aux_0 = h**3 / (12 * mu_t)
            aux_1 = (speed * reference_viscosity) / (rho * Cp * reference_temperature)
            aux_2 = aux_1 / radial_clearance**2
            aux_3 = aux_0 * (journal_radius**2 * dPdz * dY) / (2 * axial_length**2)
            aux_4 = k_t * h / (rho * Cp * speed)

            hpt = -Xpt * np.cos(theta) - Ypt * np.sin(theta)

            b_TG = aux_2 * (journal_radius**2 * dY * dZ * P[ki, kj] * hpt)
            b_TH = aux_1 * ((4 * mu_t * hpt**2 * dY * dZ) / (3 * h))
            b_TI = aux_2 * (mu_t * journal_radius**2 * dY * dZ / h)
            b_TJ = aux_0 * aux_2 * (journal_radius**2 * dPdy**2 * dY * dZ / beta_s**2)
            b_TK = aux_2 * aux_3 * (2 * journal_radius**2 * dPdz * dZ)

            AE = -aux_4 / ((beta_s * journal_radius) ** 2 * dY)
            AW = aux_0 * dPdy * dZ / beta_s**2 - h * dZ / (2 * beta_s) + AE
            AN = -aux_3 - aux_4 * dY / (axial_length**2 * dZ)
            AS = aux_3 - aux_4 * dY / (axial_length**2 * dZ)
            AP = -(AE + AW + AN + AS)

            AP_mod = AP
            b_mod = b_TG + b_TH + b_TI + b_TJ + b_TK

            if ki == 0:
                AP_mod += AS
            else:
                A_T[k, k - elm_cir] = AS

            if ki == elm_axi - 1:
                AP_mod += AN
            else:
                A_T[k, k + elm_cir] = AN

            if kj == 0:
                AP_mod -= AW
                b_mod -= 2 * AW * (T_ref / reference_temperature)
            else:
                A_T[k, k - 1] = AW

            if kj == elm_cir - 1:
                AP_mod += AE
            else:
                A_T[k, k + 1] = AE

            A_T[k, k] = AP_mod
            b_T[k, 0] = b_mod

            k += 1
            kj += 1

        kj = 0

    return A_T, b_T


def _solve(A, b):
    """Solve the linear system Ax = b using sparse matrix solver."""
    return sparse.linalg.spsolve(sparse.csr_matrix(A), b)
