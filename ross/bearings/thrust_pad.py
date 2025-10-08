import time
import numpy as np
from scipy.optimize import fmin
from scipy.interpolate import griddata
import plotly.graph_objects as go
from prettytable import PrettyTable

from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from ross.plotly_theme import tableau_colors
from ross.bearings.lubricants import lubricants_dict


class ThrustPad(BearingElement):
    """Thermo-Hydro-Dynamic (THD) Tilting Pad Thrust Bearing.

    This class implements a comprehensive thermo-hydro-dynamic analysis for
    tilting pad thrust bearings, calculating pressure and temperature fields,
    equilibrium position, and dynamic coefficients (stiffness and damping).

    The analysis solves the Reynolds equation for pressure distribution and
    the energy equation for temperature field, considering viscosity variations
    with temperature and turbulent effects.

    Parameters
    ----------
    n : int
        Node number for the bearing element.
    pad_inner_radius : float or Quantity
        Inner radius of the pad. Default unit is meter.
    pad_outer_radius : float or Quantity
        Outer radius of the pad. Default unit is meter.
    pad_pivot_radius : float or Quantity
        Radius of the pivot point. Default unit is meter.
    pad_arc_length : float or Quantity
        Arc length of each pad. Default unit is degrees.
    angular_pivot_position : float or Quantity
        Angular position of the pivot point. Default unit is degrees.
    oil_supply_temperature : float or Quantity
        Oil supply temperature. Default unit is degrees Celsius.
    lubricant : str or dict
        Lubricant specification. Can be:
        - String: 'ISOVG32', 'ISOVG46', 'ISOVG68'
        - Dictionary: Custom lubricant properties
    n_pad : int
        Number of pads in the bearing.
    n_theta : int
        Number of mesh elements in circumferential direction.
    n_radial : int
        Number of mesh elements in radial direction.
    frequency : array_like or Quantity
        Rotor rotating frequency(ies). Default unit is rad/s.
    equilibrium_position_mode : str
        Equilibrium position calculation mode:
        - 'calculate': Calculate film thickness and inclination angles
        - 'imposed': Use imposed film thickness, calculate inclination angles
    model_type : str, optional
        Type of model to be used. Options:
        - 'thermo_hydro_dynamic': Thermo-Hydro-Dynamic model
    radial_inclination_angle : float or Quantity
        Initial radial inclination angle. Default unit is radians.
    circumferential_inclination_angle : float or Quantity
        Initial circumferential inclination angle. Default unit is radians.
    initial_film_thickness : float or Quantity
        Initial film thickness at pivot point. Default unit is meters.
    fzs_load : float, optional
        Axial load applied to the bearing. Default is None.
    print_result : bool, optional
        Whether to print calculation results. Default is False.
    print_progress : bool, optional
        Whether to print convergence progress. Default is False.
    print_time : bool, optional
        Whether to print calculation time. Default is False.
    compare_coefficients : bool, optional
        Whether to compare dynamic coefficients by each frequency in a table. Default is False.
    **kwargs
        Additional keyword arguments passed to BearingElement.

    Attributes
    ----------
    pressure_field_dimensional : ndarray
        Dimensional pressure field [Pa]. Shape: (n_radial+2, n_theta+2)
    temperature_field : ndarray
        Temperature field [°C]. Shape: (n_radial+2, n_theta+2)
    pivot_film_thickness : float
        Oil film thickness at the pivot point [m]
    max_thickness : float
        Maximum oil film thickness [m]
    min_thickness : float
        Minimum oil film thickness [m]
    kzz : ndarray
        Axial stiffness coefficient [N/m]. Shape: (n_frequencies,)
    czz : ndarray
        Axial damping coefficient [N*s/m]. Shape: (n_frequencies,)
    viscosity_field : ndarray
        Viscosity field [Pa*s]. Shape: (n_radial, n_theta)
    film_thickness_center_array : ndarray
        Film thickness at cell centers. Shape: (n_radial, n_theta)

    Notes
    -----
    The class implements a finite volume method to solve the Reynolds equation
    and energy equation simultaneously. The solution includes:

    1. Pressure Field: Solved using finite volume discretization of the
       Reynolds equation with appropriate boundary conditions.

    2. Temperature Field: Solved using the energy equation considering
       viscous heating, convection, and conduction effects.

    3. Viscosity Variation: Temperature-dependent viscosity using
       exponential interpolation: μ = a * exp(b * T)

    4. Equilibrium Position: Found by minimizing residual forces and
       moments using scipy.optimize.fmin.

    5. Dynamic Coefficients: Calculated using perturbation method
       for stiffness and damping coefficients.

    The mesh discretization uses a structured grid with n_radial by n_theta
    control volumes. Boundary conditions include atmospheric pressure at
    pad edges and oil supply temperature at boundaries.

    Examples
    --------
    >>> from ross.bearings.thrust_pad import ThrustPad
    >>> from ross.units import Q_
    >>> bearing = ThrustPad(
    ...     n=1,
    ...     pad_inner_radius=Q_(1150, "mm"),
    ...     pad_outer_radius=Q_(1725, "mm"),
    ...     pad_pivot_radius=Q_(1442.5, "mm"),
    ...     pad_arc_length=Q_(26, "deg"),
    ...     angular_pivot_position=Q_(15, "deg"),
    ...     oil_supply_temperature=Q_(40, "degC"),
    ...     lubricant="ISOVG68",
    ...     n_pad=12,
    ...     n_theta=10,
    ...     n_radial=10,
    ...     frequency=Q_([90], "RPM"),
    ...     equilibrium_position_mode="calculate",
    ...     fzs_load=13.320e6,
    ...     radial_inclination_angle=Q_(-2.75e-04, "rad"),
    ...     circumferential_inclination_angle=Q_(-1.70e-05, "rad"),
    ...     initial_film_thickness=Q_(0.2, "mm")
    ... )

    References
    ----------
    .. [1] BARBOSA, J.S. Analise de Modelos Termohidrodinamicos para Mancais de unidades geradoras Francis. 2016. Dissertacao de Mestrado. Universidade Federal de Uberlandia, Uberlandia.
    .. [2] HEINRICHSON, N.; SANTOS, I. F.; FUERST, A., The Influence of Injection Pockets on the Performance of Tilting Pad Thrust Bearings Part I Theory. Journal of Tribology, 2007.
    .. [3] NICOLETTI, R., Efeitos Termicos em Mancais Segmentados Hibridos Teoria e Experimento. 1999. Dissertacao de Mestrado. Universidade Estadual de Campinas, Campinas.
    .. [4] LUND, J. W.; THOMSEN, K. K. A calculation method and data for the dynamic coefficients of oil lubricated journal bearings. Topics in fluid film bearing and rotor bearing system design and optimization, n. 1000118, 1978.
    """

    @check_units
    def __init__(
        self,
        n,
        pad_inner_radius,
        pad_outer_radius,
        pad_pivot_radius,
        pad_arc_length,
        angular_pivot_position,
        oil_supply_temperature,
        lubricant,
        n_pad,
        n_theta,
        n_radial,
        frequency,
        equilibrium_position_mode,
        radial_inclination_angle,
        circumferential_inclination_angle,
        initial_film_thickness,
        model_type="thermo_hydro_dynamic",
        fzs_load=None,
        print_result=False,
        print_progress=False,
        print_time=False,
        compare_coefficients=False,
        **kwargs,
    ):
        self.print_result = print_result
        self.print_progress = print_progress
        self.print_time = print_time
        self.compare_coefficients = compare_coefficients

        self.model_type = model_type
        self.pad_inner_radius = pad_inner_radius
        self.pad_outer_radius = pad_outer_radius
        self.pad_pivot_radius = pad_pivot_radius
        self.frequency = frequency
        self.pad_arc_length = pad_arc_length
        self.angular_pivot_position = angular_pivot_position.m_as("rad")
        self.oil_supply_temperature = Q_(oil_supply_temperature, "degK").m_as("degC")
        self.reference_temperature = self.oil_supply_temperature
        self.lubricant = lubricant
        self.n_pad = n_pad
        self.n_theta = n_theta
        self.n_radial = n_radial
        self.rp = self.pad_pivot_radius / self.pad_inner_radius
        self.theta_pad = self.angular_pivot_position / self.pad_arc_length
        self.d_radius = (self.pad_outer_radius / self.pad_inner_radius - 1) / (
            self.n_radial
        )
        self.d_theta = 1 / (self.n_theta)
        self.initial_temperature = np.full(
            (self.n_radial, self.n_theta), self.reference_temperature
        )

        self.equilibrium_position_mode = equilibrium_position_mode
        self.fzs_load = fzs_load
        self.radial_inclination_angle = radial_inclination_angle
        self.circumferential_inclination_angle = circumferential_inclination_angle
        self.initial_film_thickness = initial_film_thickness
        self.initial_position = np.array(
            [
                radial_inclination_angle,
                circumferential_inclination_angle,
                initial_film_thickness,
            ]
        )

        if isinstance(lubricant, str):
            if lubricant not in lubricants_dict:
                raise KeyError(f"Lubricant {lubricant} not found in the database.")
            self.lubricant_properties = lubricants_dict[lubricant]
        elif isinstance(lubricant, dict):
            self.lubricant_properties = lubricant
        else:
            raise TypeError("Lubricant must be either a string or a dictionary.")

        self.rho = self.lubricant_properties["liquid_density"]
        self.kt = self.lubricant_properties["liquid_thermal_conductivity"]
        self.cp = self.lubricant_properties["liquid_specific_heat"]

        T_1 = Q_(self.lubricant_properties["temperature1"], "degK").m_as("degC")
        mu_1 = self.lubricant_properties["liquid_viscosity1"]
        T_2 = Q_(self.lubricant_properties["temperature2"], "degK").m_as("degC")
        mu_2 = self.lubricant_properties["liquid_viscosity2"]

        # Viscosity interpolation coefficients
        self.a_a, self.b_b = self._get_interp_coeffs(T_1, T_2, mu_1, mu_2)

        self.reference_viscosity = self.a_a * np.exp(
            self.reference_temperature * self.b_b
        )

        # Discretization of the mesh
        self.radius_array = np.linspace(
            1 + 0.5 * self.d_radius,
            1 + (self.n_radial - 0.5) * self.d_radius,
            self.n_radial,
        )
        self.theta_array = np.linspace(
            0.5 * self.d_theta, (self.n_theta - 0.5) * self.d_theta, self.n_theta
        )

        n_freq = np.shape(frequency)[0]

        kzz = np.zeros(n_freq)
        czz = np.zeros(n_freq)

        for i in range(n_freq):
            self.speed = self.frequency[i]

            if self.model_type == "thermo_hydro_dynamic":
                self.run_thermo_hydro_dynamic()

            kzz[i], czz[i] = self.kzz, self.czz

        super().__init__(
            n, kxx=0, cxx=0, kzz=kzz, czz=czz, frequency=frequency, **kwargs
        )

        if self.compare_coefficients:
            print("\n" + "=" * 60)
            print("           DYNAMIC COEFFICIENTS COMPARISON TABLE")
            print("=" * 60)

            comparison_table = self.format_table(
                frequency=self.frequency,
                coefficients=["kzz", "czz"],
                frequency_units="RPM",
                stiffness_units="N/m",
                damping_units="N*s/m",
            )

            print(comparison_table)
            print("=" * 60)

    def run_thermo_hydro_dynamic(self):
        """
        Execute the complete thermo-hydrodynamic analysis for the tilting pad thrust bearing.

        This method performs the main computational sequence for analyzing a tilting pad
        thrust bearing, including pressure and temperature field calculations, equilibrium
        position determination, and dynamic coefficient computation for each operating
        frequency.

        The analysis includes:
        - Initialization of field arrays (pressure, temperature, film thickness)
        - Iterative solution of Reynolds and energy equations
        - Calculation of hydrodynamic forces and moments
        - Computation of stiffness and damping coefficients
        - Optional display of results

        Parameters
        ----------
        None
            This method uses the bearing parameters defined during initialization.

        Returns
        -------
        None
            Results are stored as instance attributes and optionally displayed.

        Attributes
        ----------
        pressure_field_dimensional : ndarray
            Dimensional pressure field [Pa]. Shape: (n_radial+2, n_theta+2).
        temperature_field : ndarray
            Temperature field [°C]. Shape: (n_radial+2, n_theta+2).
        pivot_film_thickness : float
            Oil film thickness at pivot point [m].
        max_thickness : float
            Maximum oil film thickness [m].
        min_thickness : float
            Minimum oil film thickness [m].
        kzz : ndarray
            Axial stiffness coefficient [N/m]. Shape: (n_frequencies,).
        czz : ndarray
            Axial damping coefficient [N*s/m]. Shape: (n_frequencies,).
        viscosity_field : ndarray
            Viscosity field [Pa*s]. Shape: (n_radial, n_theta).

        Notes
        -----
        The method processes each frequency in the frequency array sequentially.
        For each frequency, it:
        1. Solves the pressure field using Reynolds equation with finite volume method
        2. Solves the temperature field using energy equation considering viscous heating
        3. Calculates dynamic coefficients using perturbation method
        4. Stores results in instance attributes

        The solution includes viscosity variation with temperature using exponential
        interpolation: μ = a * exp(b * T), where a and b are temperature-dependent
        coefficients calculated from lubricant properties.

        Examples
        --------
        >>> from ross.bearings.thrust_pad import thrust_pad_example
        >>> bearing = thrust_pad_example()
        """
        if self.print_time:
            t1 = time.time()

        # Solve the fields (pressure and temperature fields)
        self.solve_fields()

        # Calculate the dynamic coefficients
        self.coefficients()

        # Display the results
        if self.print_result:
            self.show_results()
            self.plot_results()

        if self.print_time:
            t2 = time.time()
            print("Calculation time spent: {0:.2f} seconds".format(t2 - t1))

    def show_results(self):
        """Display thrust bearing calculation results in a formatted table.

        This method prints the main results from the thrust bearing analysis
        using PrettyTable, including operating conditions, field results,
        load information, and dynamic coefficients.

        Parameters
        ----------
        None
            This method uses the bearing parameters and results stored as
            instance attributes.

        Returns
        -------
        None
            Results are printed to the console in a formatted table.

        Notes
        -----
        The displayed results include:
        - Operating speed in RPM
        - Equilibrium position mode
        - Maximum and minimum pressure values
        - Maximum and minimum temperature values
        - Film thickness values (maximum, minimum, and pivot)
        - Axial load information
        - Dynamic coefficients (stiffness kzz and damping czz)

        Examples
        --------
        >>> from ross.bearings.thrust_pad import thrust_pad_example
        >>> bearing = thrust_pad_example()
        >>> bearing.show_results()
        <BLANKLINE>
        ================================================
                    THRUST BEARING RESULTS
        ================================================
        +------------------------+-------------+-------+
        |       Parameter        |    Value    |  Unit |
        +------------------------+-------------+-------+
        |    Operating Speed     |     90.0    |  RPM  |
        |    Equilibrium Mode    |  calculate  |   -   |
        |    Maximum Pressure    |  6957021.42 |   Pa  |
        |  Maximum Temperature   |     70.4    |   °C  |
        | Maximum Film Thickness |   0.000207  |   m   |
        | Minimum Film Thickness |   0.000082  |   m   |
        |  Pivot Film Thickness  |   0.000131  |   m   |
        |       Axial Load       | 13320000.00 |   N   |
        |    kzz (Stiffness)     |  3.1763e+11 |  N/m  |
        |     czz (Damping)      |  1.0806e+10 | N*s/m |
        +------------------------+-------------+-------+
        ================================================
        """

        print("\n" + "=" * 48)
        print("            THRUST BEARING RESULTS")
        print("=" * 48)

        table = PrettyTable()
        table.field_names = ["Parameter", "Value", "Unit"]

        # Operating conditions
        table.add_row(["Operating Speed", f"{self.speed * 30 / np.pi:.1f}", "RPM"])
        table.add_row(["Equilibrium Mode", self.equilibrium_position_mode, "-"])

        # Field results
        table.add_row(
            ["Maximum Pressure", f"{self.pressure_field_dimensional.max():.2f}", "Pa"]
        )
        table.add_row(
            ["Maximum Temperature", f"{self.temperature_field.max():.1f}", "°C"]
        )
        table.add_row(["Maximum Film Thickness", f"{self.max_thickness:.6f}", "m"])
        table.add_row(["Minimum Film Thickness", f"{self.min_thickness:.6f}", "m"])
        table.add_row(["Pivot Film Thickness", f"{self.pivot_film_thickness:.6f}", "m"])

        # Load results
        if self.equilibrium_position_mode == "imposed":
            table.add_row(["Axial Load", f"{self.fzs_load.sum():.2f}", "N"])
        elif self.equilibrium_position_mode == "calculate":
            table.add_row(["Axial Load", f"{self.fzs_load:.2f}", "N"])

        # Dynamic coefficients
        table.add_row(["kzz (Stiffness)", f"{self.kzz.item():.4e}", "N/m"])
        table.add_row(["czz (Damping)", f"{self.czz.item():.4e}", "N*s/m"])

        print(table)
        print("=" * 48)

    def solve_fields(self):
        residual_force_moment = 10
        tolerance_force_moment = 1

        # Residual force and moment convergence loop
        iteration = 0
        while residual_force_moment >= tolerance_force_moment:
            iteration += 1
            if self.print_progress:
                print(
                    f"Iteration {iteration} - Residual Force & Moment: {residual_force_moment:.6f}"
                )

            if self.equilibrium_position_mode == "imposed":
                self.h0i = self.initial_position[2]
                x = fmin(
                    self._equilibrium_objective,
                    self.initial_position,
                    xtol=tolerance_force_moment,
                    ftol=tolerance_force_moment,
                    maxiter=100000,
                    maxfun=100000,
                    full_output=0,
                    disp=self.print_progress,
                    retall=0,
                    callback=None,
                    initial_simplex=None,
                )

                radial_inclination_angle = x[0]
                circumferential_inclination_angle = x[1]
                self.pivot_film_thickness = self.h0i

            else:
                x = fmin(
                    self._equilibrium_objective,
                    self.initial_position,
                    xtol=0.1,
                    ftol=0.1,
                    maxiter=100,
                    disp=False,
                )

                radial_inclination_angle = x[0]
                circumferential_inclination_angle = x[1]
                self.pivot_film_thickness = x[2]

            viscosity_convergence_tolerance = 1e-5

            dh_dT = 0
            mi_i = np.zeros((self.n_radial, self.n_theta))

            # Initial temperature field
            T_i = self.initial_temperature

            for ii in range(0, self.n_radial):
                for jj in range(0, self.n_theta):
                    mi_i[ii, jj] = self.a_a * np.exp(self.b_b * T_i[ii, jj])  # [Pa.s]

            mu_update = (1 / self.reference_viscosity) * mi_i
            self.mu = 0.2 * mu_update

            # Viscosity convergence loop
            for ii in range(0, self.n_radial):
                for jj in range(0, self.n_theta):
                    viscosity_variation = np.abs(
                        (mu_update[ii, jj] - self.mu[ii, jj]) / self.mu[ii, jj]
                    )
            max_viscosity_variation = 1

            while max_viscosity_variation >= viscosity_convergence_tolerance:
                self.mu = np.array(mu_update)

                radial_param = (
                    radial_inclination_angle
                    * self.pad_inner_radius
                    / self.pivot_film_thickness
                )
                circum_param = (
                    circumferential_inclination_angle
                    * self.pad_inner_radius
                    / self.pivot_film_thickness
                )

                # Volumes number
                volumes_number = (self.n_radial) * (self.n_theta)

                # Variable initialization
                mat_coeff = np.zeros((volumes_number, volumes_number))
                source_term_vector = np.zeros((volumes_number, 1))
                self.film_thickness_center_array = np.zeros(
                    (self.n_radial, self.n_theta)
                )
                self.film_thickness_ne = np.zeros((self.n_radial, self.n_theta))
                self.film_thickness_nw = np.zeros((self.n_radial, self.n_theta))
                self.film_thickness_se = np.zeros((self.n_radial, self.n_theta))
                self.film_thickness_sw = np.zeros((self.n_radial, self.n_theta))
                dp0_dr = np.zeros((self.n_radial, self.n_theta))
                dp0_dtheta = np.zeros((self.n_radial, self.n_theta))
                temperature_updt = np.zeros((self.n_radial, self.n_theta))
                moment_x = np.zeros((self.n_radial, self.n_theta))
                moment_y = np.zeros((self.n_radial, self.n_theta))
                force_axial = np.zeros((self.n_radial, self.n_theta))
                self.pressure_field = np.ones((self.n_radial, self.n_theta))
                self.viscosity_field = np.zeros((self.n_radial, self.n_theta))

                pressure_field_dimensional = np.zeros(
                    (self.n_radial + 2, self.n_theta + 2)
                )

                try:
                    self._solve_pressure_field(
                        circum_param, radial_param, mat_coeff, source_term_vector
                    )
                    if (
                        self.pressure_field is not None
                        and self.pressure_field.ndim >= 1
                    ):
                        pressure_field_dimensional[1:-1, 1:-1] = (
                            self.pad_inner_radius**2
                            * self.speed
                            * self.reference_viscosity
                            / self.pivot_film_thickness**2
                        ) * np.flipud(self.pressure_field)
                    else:
                        raise ValueError("pressure_field is not a valid array")
                except Exception as e:
                    print(f"Error in pressure calculation: {e}")
                    pressure_field_dimensional[1:-1, 1:-1] = np.zeros(
                        (self.n_radial, self.n_theta)
                    )

                radial_idx = 0
                angular_idx = 0

                # Pressure vectorization index
                vectorization_idx = -1

                # Volumes number
                volumes_number = (self.n_radial) * (self.n_theta)

                # Coefficients Matrix
                mat_coeff = np.zeros((volumes_number, volumes_number))
                source_term_vector = np.zeros((volumes_number, 1))

                for radius in self.radius_array:
                    for theta in self.theta_array:
                        if angular_idx == 0 and radial_idx == 0:
                            dp0_dtheta[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx, angular_idx + 1]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / self.d_theta
                            dp0_dr[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx + 1, angular_idx]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / self.d_radius

                        if (
                            angular_idx == 0
                            and radial_idx > 0
                            and radial_idx < self.n_radial - 1
                        ):
                            dp0_dtheta[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx, angular_idx + 1]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / self.d_theta
                            dp0_dr[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx + 1, angular_idx]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / (self.d_radius)

                        if angular_idx == 0 and radial_idx == self.n_radial - 1:
                            dp0_dtheta[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx, angular_idx + 1]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / self.d_theta
                            dp0_dr[radial_idx, angular_idx] = (
                                0 - self.pressure_field[radial_idx, angular_idx]
                            ) / (0.5 * self.d_radius)

                        if (
                            radial_idx == 0
                            and angular_idx > 0
                            and angular_idx < self.n_theta - 1
                        ):
                            dp0_dtheta[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx, angular_idx + 1]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / (self.d_theta)
                            dp0_dr[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx + 1, angular_idx]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / (self.d_radius)

                        if (
                            angular_idx > 0
                            and angular_idx < self.n_theta - 1
                            and radial_idx > 0
                            and radial_idx < self.n_radial - 1
                        ):
                            dp0_dtheta[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx, angular_idx + 1]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / (self.d_theta)
                            dp0_dr[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx + 1, angular_idx]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / (self.d_radius)

                        if (
                            radial_idx == self.n_radial - 1
                            and angular_idx > 0
                            and angular_idx < self.n_theta - 1
                        ):
                            dp0_dtheta[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx, angular_idx + 1]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / (self.d_theta)
                            dp0_dr[radial_idx, angular_idx] = (
                                0 - self.pressure_field[radial_idx, angular_idx]
                            ) / (0.5 * self.d_radius)

                        if radial_idx == 0 and angular_idx == self.n_theta - 1:
                            dp0_dtheta[radial_idx, angular_idx] = (
                                0 - self.pressure_field[radial_idx, angular_idx]
                            ) / (0.5 * self.d_theta)
                            dp0_dr[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx + 1, angular_idx]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / (self.d_radius)

                        if (
                            angular_idx == self.n_theta - 1
                            and radial_idx > 0
                            and radial_idx < self.n_radial - 1
                        ):
                            dp0_dtheta[radial_idx, angular_idx] = (
                                0 - self.pressure_field[radial_idx, angular_idx]
                            ) / (0.5 * self.d_theta)
                            dp0_dr[radial_idx, angular_idx] = (
                                self.pressure_field[radial_idx + 1, angular_idx]
                                - self.pressure_field[radial_idx, angular_idx]
                            ) / (self.d_radius)

                        if (
                            angular_idx == self.n_theta - 1
                            and radial_idx == self.n_radial - 1
                        ):
                            dp0_dtheta[radial_idx, angular_idx] = (
                                0 - self.pressure_field[radial_idx, angular_idx]
                            ) / (0.5 * self.d_theta)
                            dp0_dr[radial_idx, angular_idx] = (
                                0 - self.pressure_field[radial_idx, angular_idx]
                            ) / (0.5 * self.d_radius)

                        theta_e = theta + 0.5 * self.d_theta
                        theta_w = theta - 0.5 * self.d_theta
                        radius_n = radius + 0.5 * self.d_radius
                        radius_s = radius - 0.5 * self.d_radius

                        north_coeff = (
                            self.d_theta
                            / 12
                            * (
                                radius
                                * self.film_thickness_center_array[
                                    radial_idx, angular_idx
                                ]
                                ** 3
                                / self.mu[radial_idx, angular_idx]
                                * dp0_dr[radial_idx, angular_idx]
                            )
                        )
                        energy_n = 0.5 * north_coeff
                        energy_s = -0.5 * north_coeff
                        energy_c = -(energy_s + energy_n)
                        east_coeff = (
                            self.d_radius
                            / (12 * self.pad_arc_length**2)
                            * (
                                self.film_thickness_center_array[
                                    radial_idx, angular_idx
                                ]
                                ** 3
                                / (radius * self.mu[radial_idx, angular_idx])
                                * dp0_dtheta[radial_idx, angular_idx]
                            )
                            - self.d_radius
                            / (2 * self.pad_arc_length)
                            * self.film_thickness_center_array[radial_idx, angular_idx]
                            * radius
                        )
                        energy_e = 0 * east_coeff
                        energy_w = -1 * east_coeff
                        energy_c2 = -(energy_e + energy_w)

                        # Difusive terms - central differences
                        diffusion_n = (
                            self.kt
                            / (
                                self.rho
                                * self.cp
                                * self.speed
                                * self.pad_inner_radius**2
                            )
                            * (self.d_theta * radius_n)
                            / (self.d_radius)
                            * self.film_thickness_center_array[radial_idx, angular_idx]
                        )
                        diffusion_s = (
                            self.kt
                            / (
                                self.rho
                                * self.cp
                                * self.speed
                                * self.pad_inner_radius**2
                            )
                            * (self.d_theta * radius_s)
                            / (self.d_radius)
                            * self.film_thickness_center_array[radial_idx, angular_idx]
                        )
                        diffusion_c1 = -(diffusion_n + diffusion_s)
                        diffusion_e = (
                            self.kt
                            / (
                                self.rho
                                * self.cp
                                * self.speed
                                * self.pad_inner_radius**2
                            )
                            * self.d_radius
                            / (self.pad_arc_length**2 * self.d_theta)
                            * self.film_thickness_center_array[radial_idx, angular_idx]
                            / radius
                        )
                        diffusion_w = (
                            self.kt
                            / (
                                self.rho
                                * self.cp
                                * self.speed
                                * self.pad_inner_radius**2
                            )
                            * self.d_radius
                            / (self.pad_arc_length**2 * self.d_theta)
                            * self.film_thickness_center_array[radial_idx, angular_idx]
                            / radius
                        )
                        diffusion_c2 = -(diffusion_e + diffusion_w)

                        coeff_w = energy_w + diffusion_w
                        coeff_s = energy_s + diffusion_s
                        coeff_n = energy_n + diffusion_n
                        coeff_e = energy_e + diffusion_e
                        coeff_c = energy_c + energy_c2 + diffusion_c1 + diffusion_c2

                        source_f = 0
                        source_g = 0
                        source_h = (
                            self.d_radius
                            * self.d_theta
                            / (12 * self.pad_arc_length**2)
                            * (
                                self.film_thickness_center_array[
                                    radial_idx, angular_idx
                                ]
                                ** 3
                                / (self.mu[radial_idx, angular_idx] * radius)
                                * dp0_dtheta[radial_idx, angular_idx] ** 2
                            )
                        )
                        source_i = (
                            self.mu[radial_idx, angular_idx]
                            * radius**3
                            / (
                                self.film_thickness_center_array[
                                    radial_idx, angular_idx
                                ]
                            )
                            * self.d_radius
                            * self.d_theta
                        )
                        source_j = (
                            self.d_radius
                            * self.d_theta
                            / 12
                            * (
                                radius
                                * self.film_thickness_center_array[
                                    radial_idx, angular_idx
                                ]
                                ** 3
                                / self.mu[radial_idx, angular_idx]
                            )
                            * dp0_dr[radial_idx, angular_idx] ** 2
                        )
                        source_k = (
                            self.d_radius
                            * self.d_theta
                            / (12 * self.pad_arc_length)
                            * (
                                self.film_thickness_center_array[
                                    radial_idx, angular_idx
                                ]
                                ** 3
                                / radius
                            )
                            * dp0_dtheta[radial_idx, angular_idx]
                        )
                        source_l = (
                            self.d_radius
                            * self.d_theta
                            / 60
                            * (
                                self.film_thickness_center_array[
                                    radial_idx, angular_idx
                                ]
                                ** 5
                                / (self.mu[radial_idx, angular_idx] * radius)
                            )
                            * dp0_dr[radial_idx, angular_idx] ** 2
                        )
                        source_m = (
                            2
                            * self.d_radius
                            * self.d_theta
                            * (
                                radius
                                * self.mu[radial_idx, angular_idx]
                                / self.film_thickness_center_array[
                                    radial_idx, angular_idx
                                ]
                            )
                            * (dh_dT) ** 2
                        )
                        source_n = (
                            self.d_radius
                            * self.d_theta
                            / 3
                            * radius
                            * self.mu[radial_idx, angular_idx]
                            * self.film_thickness_center_array[radial_idx, angular_idx]
                        )
                        source_o = (
                            self.d_radius
                            * self.d_theta
                            / (120 * self.pad_arc_length**2)
                            * (
                                self.film_thickness_center_array[
                                    radial_idx, angular_idx
                                ]
                                ** 5
                                / (self.mu[radial_idx, angular_idx] * radius**3)
                            )
                            * dp0_dtheta[radial_idx, angular_idx] ** 2
                        )

                        # Vectorization index
                        vectorization_idx = vectorization_idx + 1

                        source_term_vector[vectorization_idx, 0] = (
                            -source_f
                            + (
                                self.speed
                                * self.reference_viscosity
                                * self.pad_inner_radius**2
                                / (
                                    self.rho
                                    * self.cp
                                    * self.pivot_film_thickness**2
                                    * self.reference_temperature
                                )
                            )
                            * (source_g - source_h - source_i - source_j)
                            + (
                                self.reference_viscosity
                                * self.speed
                                / (self.rho * self.cp * self.reference_temperature)
                            )
                            * (source_k - source_l - source_m - source_n - source_o)
                        )

                        if angular_idx == 0 and radial_idx == 0:
                            mat_coeff[vectorization_idx, vectorization_idx] = (
                                coeff_c + coeff_s
                            )
                            mat_coeff[vectorization_idx, vectorization_idx + 1] = (
                                coeff_e
                            )
                            mat_coeff[
                                vectorization_idx, vectorization_idx + self.n_theta
                            ] = coeff_n
                            source_term_vector[vectorization_idx, 0] = (
                                source_term_vector[vectorization_idx, 0] - 1 * coeff_w
                            )

                        if (
                            angular_idx == 0
                            and radial_idx > 0
                            and radial_idx < self.n_radial - 1
                        ):
                            mat_coeff[vectorization_idx, vectorization_idx] = coeff_c
                            mat_coeff[vectorization_idx, vectorization_idx + 1] = (
                                coeff_e
                            )
                            mat_coeff[
                                vectorization_idx, vectorization_idx + self.n_theta
                            ] = coeff_n
                            mat_coeff[
                                vectorization_idx, vectorization_idx - self.n_theta
                            ] = coeff_s
                            source_term_vector[vectorization_idx, 0] = (
                                source_term_vector[vectorization_idx, 0] - 1 * coeff_w
                            )

                        if angular_idx == 0 and radial_idx == self.n_radial - 1:
                            mat_coeff[vectorization_idx, vectorization_idx] = (
                                coeff_c + coeff_n
                            )
                            mat_coeff[vectorization_idx, vectorization_idx + 1] = (
                                coeff_e
                            )
                            mat_coeff[
                                vectorization_idx, vectorization_idx - self.n_theta
                            ] = coeff_s
                            source_term_vector[vectorization_idx, 0] = (
                                source_term_vector[vectorization_idx, 0] - 1 * coeff_w
                            )

                        if (
                            radial_idx == 0
                            and angular_idx > 0
                            and angular_idx < self.n_theta - 1
                        ):
                            mat_coeff[vectorization_idx, vectorization_idx] = (
                                coeff_c + coeff_s
                            )
                            mat_coeff[vectorization_idx, vectorization_idx + 1] = (
                                coeff_e
                            )
                            mat_coeff[vectorization_idx, vectorization_idx - 1] = (
                                coeff_w
                            )
                            mat_coeff[
                                vectorization_idx, vectorization_idx + self.n_theta
                            ] = coeff_n

                        if (
                            angular_idx > 0
                            and angular_idx < self.n_theta - 1
                            and radial_idx > 0
                            and radial_idx < self.n_radial - 1
                        ):
                            mat_coeff[vectorization_idx, vectorization_idx] = coeff_c
                            mat_coeff[vectorization_idx, vectorization_idx - 1] = (
                                coeff_w
                            )
                            mat_coeff[
                                vectorization_idx, vectorization_idx + self.n_theta
                            ] = coeff_n
                            mat_coeff[
                                vectorization_idx, vectorization_idx - self.n_theta
                            ] = coeff_s
                            mat_coeff[vectorization_idx, vectorization_idx + 1] = (
                                coeff_e
                            )

                        if (
                            radial_idx == self.n_radial - 1
                            and angular_idx > 0
                            and angular_idx < self.n_theta - 1
                        ):
                            mat_coeff[vectorization_idx, vectorization_idx] = (
                                coeff_c + coeff_n
                            )
                            mat_coeff[vectorization_idx, vectorization_idx - 1] = (
                                coeff_w
                            )
                            mat_coeff[vectorization_idx, vectorization_idx + 1] = (
                                coeff_e
                            )
                            mat_coeff[
                                vectorization_idx, vectorization_idx - self.n_theta
                            ] = coeff_s

                        if radial_idx == 0 and angular_idx == self.n_theta - 1:
                            mat_coeff[vectorization_idx, vectorization_idx] = (
                                coeff_c + coeff_e + coeff_s
                            )
                            mat_coeff[vectorization_idx, vectorization_idx - 1] = (
                                coeff_w
                            )
                            mat_coeff[
                                vectorization_idx, vectorization_idx + self.n_theta
                            ] = coeff_n

                        if (
                            angular_idx == self.n_theta - 1
                            and radial_idx > 0
                            and radial_idx < self.n_radial - 1
                        ):
                            mat_coeff[vectorization_idx, vectorization_idx] = (
                                coeff_c + coeff_e
                            )
                            mat_coeff[vectorization_idx, vectorization_idx - 1] = (
                                coeff_w
                            )
                            mat_coeff[
                                vectorization_idx, vectorization_idx - self.n_theta
                            ] = coeff_s
                            mat_coeff[
                                vectorization_idx, vectorization_idx + self.n_theta
                            ] = coeff_n

                        if (
                            angular_idx == self.n_theta - 1
                            and radial_idx == self.n_radial - 1
                        ):
                            mat_coeff[vectorization_idx, vectorization_idx] = (
                                coeff_c + coeff_n + coeff_e
                            )
                            mat_coeff[vectorization_idx, vectorization_idx - 1] = (
                                coeff_w
                            )
                            mat_coeff[
                                vectorization_idx, vectorization_idx - self.n_theta
                            ] = coeff_s

                        angular_idx = angular_idx + 1

                    radial_idx = radial_idx + 1
                    angular_idx = 0

                # Temperature field solution
                temperature_solution = np.linalg.solve(mat_coeff, source_term_vector)
                temp_idx = -1

                # Temperature matrix
                for ii in range(0, self.n_radial):
                    for jj in range(0, self.n_theta):
                        temp_idx = temp_idx + 1
                        temperature_updt[ii, jj] = temperature_solution[temp_idx, 0]

                # Viscosity field
                viscosity_variation = np.zeros((self.n_radial, self.n_theta))
                for ii in range(0, self.n_radial):
                    for jj in range(0, self.n_theta):
                        mu_update[ii, jj] = (
                            (1 / self.reference_viscosity)
                            * self.a_a
                            * np.exp(
                                self.b_b
                                * (
                                    self.reference_temperature
                                    * temperature_updt[ii, jj]
                                )
                            )
                        )
                        viscosity_variation[ii, jj] = abs(
                            (mu_update[ii, jj] - self.mu[ii, jj]) / self.mu[ii, jj]
                        )

                temperature = temperature_updt
                max_viscosity_variation = np.max(viscosity_variation)

            self.initial_temperature = temperature * self.reference_temperature

            # Dimensional pressure
            pressure_dim = (
                self.pressure_field
                * (self.pad_inner_radius**2)
                * self.speed
                * self.reference_viscosity
                / (self.pivot_film_thickness**2)
            )

            # Resulting force and momentum: Equilibrium position
            radius_coords = self.pad_inner_radius * self.radius_array
            theta_coords = self.pad_arc_length * self.theta_array
            pivot_coords = self.pad_pivot_radius * (np.ones((np.size(radius_coords))))

            for ii in range(0, self.n_theta):
                moment_x[:, ii] = (
                    pressure_dim[:, ii] * (np.transpose(radius_coords) ** 2)
                ) * np.sin(theta_coords[ii] - self.angular_pivot_position)
                moment_y[:, ii] = (
                    -pressure_dim[:, ii]
                    * np.transpose(radius_coords)
                    * np.transpose(
                        radius_coords
                        * np.cos(theta_coords[ii] - self.angular_pivot_position)
                        - pivot_coords
                    )
                )
                force_axial[:, ii] = pressure_dim[:, ii] * np.transpose(radius_coords)

            force_radial = np.trapezoid(force_axial, theta_coords)

            mom_x_radial = np.trapezoid(moment_x, theta_coords)
            mom_y_radial = np.trapezoid(moment_y, theta_coords)

            mom_x_total = np.trapezoid(mom_x_radial, radius_coords)
            mom_y_total = np.trapezoid(mom_y_radial, radius_coords)

            residual_moment_x = mom_x_total
            residual_moment_y = mom_y_total

            if self.equilibrium_position_mode == "imposed":
                residual_force_moment = np.linalg.norm(
                    np.array([residual_moment_x, residual_moment_y])
                )
                self.fzs_load = force_radial

            else:
                axial_force_residual = (
                    -np.trapezoid(force_radial, radius_coords)
                    + self.fzs_load / self.n_pad
                )
                residual_force_moment = np.linalg.norm(
                    np.array(
                        [residual_moment_x, residual_moment_y, axial_force_residual]
                    )
                )

            self.initial_position = np.array([x[0], x[1], self.pivot_film_thickness])
            self.score = residual_force_moment

        temperature_field_full = np.ones((self.n_radial + 2, self.n_theta + 2))

        temperature_field_full[1 : self.n_radial + 1, 1 : self.n_theta + 1] = np.flipud(
            self.initial_temperature
        )

        # Boundary conditions
        temperature_field_full[:, 0] = self.reference_temperature  # Left boundary
        temperature_field_full[0, :] = temperature_field_full[1, :]  # Top boundary
        temperature_field_full[self.n_radial + 1, :] = temperature_field_full[
            self.n_radial, :
        ]  # Bottom boundary
        temperature_field_full[:, self.n_theta + 1] = temperature_field_full[
            :, self.n_theta
        ]  # Right boundary

        # Store as instance attribute
        self.temperature_field = temperature_field_full

        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                self.viscosity_field[ii, jj] = self.a_a * np.exp(
                    self.b_b * (self.initial_temperature[ii, jj])
                )  # [Pa.s]

        radial_param = (
            radial_inclination_angle * self.pad_inner_radius / self.pivot_film_thickness
        )
        circum_param = (
            circumferential_inclination_angle
            * self.pad_inner_radius
            / self.pivot_film_thickness
        )
        self.mu = 1 / self.reference_viscosity * self.viscosity_field

        # Number of volumes
        volumes_number = (self.n_radial) * (self.n_theta)

        # Coefficients matrix
        mat_coeff = np.zeros((volumes_number, volumes_number))
        source_term_vector = np.zeros((volumes_number, 1))

        self._solve_pressure_field(
            circum_param, radial_param, mat_coeff, source_term_vector
        )

        pressure_field_dimensional = np.zeros((self.n_radial + 2, self.n_theta + 2))
        pressure_field_dimensional[1:-1, 1:-1] = (
            self.pad_inner_radius**2
            * self.speed
            * self.reference_viscosity
            / self.pivot_film_thickness**2
        ) * np.flipud(self.pressure_field)
        self.pressure_field_dimensional = pressure_field_dimensional

        self.max_thickness = np.max(
            self.pivot_film_thickness * self.film_thickness_center_array
        )
        self.min_thickness = np.min(
            self.pivot_film_thickness * self.film_thickness_center_array
        )

    def _equilibrium_objective(self, x):
        """Calculates the equilibrium position of the bearing

        Parameters
        ----------
        radial_inclination_angle = x[0]  : pitch angle axis r [rad]
        circumferential_inclination_angle = x[1]  : pitch angle axis s [rad]
        self.pivot_film_thickness = x[2]   : oil film thickness at pivot [m]

        """

        self.mu = np.zeros((self.n_radial, self.n_theta))
        pressure = np.zeros((self.n_radial, self.n_theta))
        moment_x = np.zeros((self.n_radial, self.n_theta))
        moment_y = np.zeros((self.n_radial, self.n_theta))
        force_axial = np.zeros((self.n_radial, self.n_theta))

        radial_inclination_angle = x[0]  # [rad]
        circumferential_inclination_angle = x[1]  # [rad]

        # Determine self.pivot_film_thickness based on equilibrium position mode
        if self.equilibrium_position_mode == "imposed":
            self.pivot_film_thickness = self.initial_position[2]
        else:  # "calculate" mode
            self.pivot_film_thickness = x[2]

        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                self.mu[ii, jj] = (
                    1
                    / self.reference_viscosity
                    * self.a_a
                    * np.exp(self.b_b * (self.initial_temperature[ii, jj]))
                )  # dimensionless

        # Dimensioneless Parameters
        radial_param = (
            radial_inclination_angle * self.pad_inner_radius / self.pivot_film_thickness
        )
        circum_param = (
            circumferential_inclination_angle
            * self.pad_inner_radius
            / self.pivot_film_thickness
        )
        film_thickness_center = self.pivot_film_thickness / self.pivot_film_thickness

        radial_idx = 0
        angular_idx = 0

        # Pressure vectorization index
        vectorization_idx = -1

        # Number of volumes
        volumes_number = (self.n_radial) * (self.n_theta)

        # Coefficients matrix
        mat_coeff = np.zeros((volumes_number, volumes_number))
        source_term_vector = np.zeros((volumes_number, 1))

        for radius in self.radius_array:
            for theta in self.theta_array:
                theta_e = theta + 0.5 * self.d_theta
                theta_w = theta - 0.5 * self.d_theta
                radius_n = radius + 0.5 * self.d_radius
                radius_s = radius - 0.5 * self.d_radius

                h_ne = (
                    film_thickness_center
                    + circum_param
                    * (
                        self.rp
                        - radius_n
                        * np.cos(self.pad_arc_length * (theta_e - self.theta_pad))
                    )
                    + radial_param
                    * radius_n
                    * np.sin(self.pad_arc_length * (theta_e - self.theta_pad))
                )
                h_nw = (
                    film_thickness_center
                    + circum_param
                    * (
                        self.rp
                        - radius_n
                        * np.cos(self.pad_arc_length * (theta_w - self.theta_pad))
                    )
                    + radial_param
                    * radius_n
                    * np.sin(self.pad_arc_length * (theta_w - self.theta_pad))
                )
                h_se = (
                    film_thickness_center
                    + circum_param
                    * (
                        self.rp
                        - radius_s
                        * np.cos(self.pad_arc_length * (theta_e - self.theta_pad))
                    )
                    + radial_param
                    * radius_s
                    * np.sin(self.pad_arc_length * (theta_e - self.theta_pad))
                )
                h_sw = (
                    film_thickness_center
                    + circum_param
                    * (
                        self.rp
                        - radius_s
                        * np.cos(self.pad_arc_length * (theta_w - self.theta_pad))
                    )
                    + radial_param
                    * radius_s
                    * np.sin(self.pad_arc_length * (theta_w - self.theta_pad))
                )

                if angular_idx == 0 and radial_idx == 0:
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = self.mu[radial_idx, angular_idx]
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = self.mu[radial_idx, angular_idx]

                if (
                    angular_idx == 0
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = self.mu[radial_idx, angular_idx]
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if angular_idx == 0 and radial_idx == self.n_radial - 1:
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = self.mu[radial_idx, angular_idx]
                    mu_n = self.mu[radial_idx, angular_idx]
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if (
                    radial_idx == 0
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = self.mu[radial_idx, angular_idx]

                if (
                    angular_idx > 0
                    and angular_idx < self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if (
                    radial_idx == self.n_radial - 1
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = self.mu[radial_idx, angular_idx]
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if radial_idx == 0 and angular_idx == self.n_theta - 1:
                    mu_e = self.mu[radial_idx, angular_idx]
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = self.mu[radial_idx, angular_idx]

                if (
                    angular_idx == self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mu_e = self.mu[radial_idx, angular_idx]
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if angular_idx == self.n_theta - 1 and radial_idx == self.n_radial - 1:
                    mu_e = self.mu[radial_idx, angular_idx]
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = self.mu[radial_idx, angular_idx]
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                # Reynolds equation coefficients
                coeff_e = (
                    1
                    / (24 * self.pad_arc_length**2 * mu_e)
                    * (self.d_radius / self.d_theta)
                    * (h_ne**3 / radius_n + h_se**3 / radius_s)
                )
                coeff_w = (
                    1
                    / (24 * self.pad_arc_length**2 * mu_w)
                    * (self.d_radius / self.d_theta)
                    * (h_nw**3 / radius_n + h_sw**3 / radius_s)
                )
                coeff_n = (
                    radius_n
                    / (24 * mu_n)
                    * (self.d_theta / self.d_radius)
                    * (h_ne**3 + h_nw**3)
                )
                coeff_s = (
                    radius_s
                    / (24 * mu_s)
                    * (self.d_theta / self.d_radius)
                    * (h_se**3 + h_sw**3)
                )
                coeff_c = -(coeff_e + coeff_w + coeff_n + coeff_s)

                # Vectorization index
                vectorization_idx = vectorization_idx + 1

                source_term_vector[vectorization_idx, 0] = (
                    self.d_radius
                    / (4 * self.pad_arc_length)
                    * (
                        radius_n * h_ne
                        + radius_s * h_se
                        - radius_n * h_nw
                        - radius_s * h_sw
                    )
                )

                if angular_idx == 0 and radial_idx == 0:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_s - coeff_w
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if (
                    angular_idx == 0
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                if angular_idx == 0 and radial_idx == self.n_radial - 1:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_w - coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                if (
                    radial_idx == 0
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_s
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if (
                    angular_idx > 0
                    and angular_idx < self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e

                if (
                    radial_idx == self.n_radial - 1
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_n
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                if radial_idx == 0 and angular_idx == self.n_theta - 1:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_e - coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if (
                    angular_idx == self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if angular_idx == self.n_theta - 1 and radial_idx == self.n_radial - 1:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_e - coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                angular_idx = angular_idx + 1

            radial_idx = radial_idx + 1
            angular_idx = 0

        pressure_solution = np.linalg.solve(mat_coeff, source_term_vector)
        pressure_idx = -1

        # Pressure matrix
        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                pressure_idx = pressure_idx + 1
                pressure[ii, jj] = pressure_solution[pressure_idx, 0]

                # Pressure boundary conditions
                if pressure[ii, jj] < 0:
                    pressure[ii, jj] = 0

        # Dimensional pressure
        pressure_dim = (
            pressure
            * (self.pad_inner_radius**2)
            * self.speed
            * self.reference_viscosity
            / (self.pivot_film_thickness**2)
        )

        radius_coords = self.pad_inner_radius * self.radius_array
        theta_coords = self.pad_arc_length * self.theta_array
        pivot_coords = self.pad_pivot_radius * (np.ones((np.size(radius_coords))))

        for ii in range(0, self.n_theta):
            moment_x[:, ii] = (
                pressure_dim[:, ii] * (np.transpose(radius_coords) ** 2)
            ) * np.sin(theta_coords[ii] - self.angular_pivot_position)
            moment_y[:, ii] = (
                -pressure_dim[:, ii]
                * np.transpose(radius_coords)
                * np.transpose(
                    radius_coords
                    * np.cos(theta_coords[ii] - self.angular_pivot_position)
                    - pivot_coords
                )
            )
            force_axial[:, ii] = pressure_dim[:, ii] * np.transpose(radius_coords)

        force_radial = np.trapezoid(force_axial, theta_coords)
        mom_x_radial = np.trapezoid(moment_x, theta_coords)
        mom_y_radial = np.trapezoid(moment_y, theta_coords)

        mom_x_total = np.trapezoid(mom_x_radial, radius_coords)
        mom_y_total = np.trapezoid(mom_y_radial, radius_coords)

        if self.equilibrium_position_mode == "imposed":
            score = np.linalg.norm([mom_x_total, mom_y_total])

        else:  # "calculate" operation mode
            axial_force_residual = (
                -np.trapezoid(force_radial, radius_coords) + self.fzs_load / self.n_pad
            )
            score = np.linalg.norm([mom_x_total, mom_y_total, axial_force_residual])

        return score

    def _solve_pressure_field(
        self, circum_param, radial_param, mat_coeff, source_term_vector
    ):
        radial_idx = 0
        angular_idx = 0

        # Pressure vectorization index
        vectorization_idx = -1

        for radius in self.radius_array:
            for theta in self.theta_array:
                theta_e = theta + 0.5 * self.d_theta
                theta_w = theta - 0.5 * self.d_theta
                radius_n = radius + 0.5 * self.d_radius
                radius_s = radius - 0.5 * self.d_radius

                self.film_thickness_center_array[radial_idx, angular_idx] = (
                    self.pivot_film_thickness / self.pivot_film_thickness
                    + circum_param
                    * (
                        self.rp
                        - radius
                        * np.cos(self.pad_arc_length * (theta - self.theta_pad))
                    )
                    + radial_param
                    * radius
                    * np.sin(self.pad_arc_length * (theta - self.theta_pad))
                )
                self.film_thickness_ne[radial_idx, angular_idx] = (
                    self.pivot_film_thickness / self.pivot_film_thickness
                    + circum_param
                    * (
                        self.rp
                        - radius_n
                        * np.cos(self.pad_arc_length * (theta_e - self.theta_pad))
                    )
                    + radial_param
                    * radius_n
                    * np.sin(self.pad_arc_length * (theta_e - self.theta_pad))
                )
                self.film_thickness_nw[radial_idx, angular_idx] = (
                    self.pivot_film_thickness / self.pivot_film_thickness
                    + circum_param
                    * (
                        self.rp
                        - radius_n
                        * np.cos(self.pad_arc_length * (theta_w - self.theta_pad))
                    )
                    + radial_param
                    * radius_n
                    * np.sin(self.pad_arc_length * (theta_w - self.theta_pad))
                )
                self.film_thickness_se[radial_idx, angular_idx] = (
                    self.pivot_film_thickness / self.pivot_film_thickness
                    + circum_param
                    * (
                        self.rp
                        - radius_s
                        * np.cos(self.pad_arc_length * (theta_e - self.theta_pad))
                    )
                    + radial_param
                    * radius_s
                    * np.sin(self.pad_arc_length * (theta_e - self.theta_pad))
                )
                self.film_thickness_sw[radial_idx, angular_idx] = (
                    self.pivot_film_thickness / self.pivot_film_thickness
                    + circum_param
                    * (
                        self.rp
                        - radius_s
                        * np.cos(self.pad_arc_length * (theta_w - self.theta_pad))
                    )
                    + radial_param
                    * radius_s
                    * np.sin(self.pad_arc_length * (theta_w - self.theta_pad))
                )

                if angular_idx == 0 and radial_idx == 0:
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = self.mu[radial_idx, angular_idx]
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = self.mu[radial_idx, angular_idx]

                if (
                    angular_idx == 0
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = self.mu[radial_idx, angular_idx]
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if angular_idx == 0 and radial_idx == self.n_radial - 1:
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = self.mu[radial_idx, angular_idx]
                    mu_n = self.mu[radial_idx, angular_idx]
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if (
                    radial_idx == 0
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = self.mu[radial_idx, angular_idx]

                if (
                    angular_idx > 0
                    and angular_idx < self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if (
                    radial_idx == self.n_radial - 1
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = self.mu[radial_idx, angular_idx]
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if radial_idx == 0 and angular_idx == self.n_theta - 1:
                    mu_e = self.mu[radial_idx, angular_idx]
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = self.mu[radial_idx, angular_idx]

                if (
                    angular_idx == self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mu_e = self.mu[radial_idx, angular_idx]
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                if angular_idx == self.n_theta - 1 and radial_idx == self.n_radial - 1:
                    mu_e = self.mu[radial_idx, angular_idx]
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = self.mu[radial_idx, angular_idx]
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )

                # Reynolds equation coefficients
                coeff_e = (
                    1
                    / (24 * self.pad_arc_length**2 * mu_e)
                    * (self.d_radius / self.d_theta)
                    * (
                        self.film_thickness_ne[radial_idx, angular_idx] ** 3 / radius_n
                        + self.film_thickness_se[radial_idx, angular_idx] ** 3
                        / radius_s
                    )
                )
                coeff_w = (
                    1
                    / (24 * self.pad_arc_length**2 * mu_w)
                    * (self.d_radius / self.d_theta)
                    * (
                        self.film_thickness_nw[radial_idx, angular_idx] ** 3 / radius_n
                        + self.film_thickness_sw[radial_idx, angular_idx] ** 3
                        / radius_s
                    )
                )
                coeff_n = (
                    radius_n
                    / (24 * mu_n)
                    * (self.d_theta / self.d_radius)
                    * (
                        self.film_thickness_ne[radial_idx, angular_idx] ** 3
                        + self.film_thickness_nw[radial_idx, angular_idx] ** 3
                    )
                )
                coeff_s = (
                    radius_s
                    / (24 * mu_s)
                    * (self.d_theta / self.d_radius)
                    * (
                        self.film_thickness_se[radial_idx, angular_idx] ** 3
                        + self.film_thickness_sw[radial_idx, angular_idx] ** 3
                    )
                )
                coeff_c = -(coeff_e + coeff_w + coeff_n + coeff_s)

                # Vectorization index
                vectorization_idx = vectorization_idx + 1

                source_term_vector[vectorization_idx, 0] = (
                    self.d_radius
                    / (4 * self.pad_arc_length)
                    * (
                        radius_n * self.film_thickness_ne[radial_idx, angular_idx]
                        + radius_s * self.film_thickness_se[radial_idx, angular_idx]
                        - radius_n * self.film_thickness_nw[radial_idx, angular_idx]
                        - radius_s * self.film_thickness_sw[radial_idx, angular_idx]
                    )
                )

                if angular_idx == 0 and radial_idx == 0:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_s - coeff_w
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if (
                    angular_idx == 0
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                if angular_idx == 0 and radial_idx == self.n_radial - 1:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_w - coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                if (
                    radial_idx == 0
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_s
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if (
                    angular_idx > 0
                    and angular_idx < self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e

                if (
                    radial_idx == self.n_radial - 1
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_n
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                if radial_idx == 0 and angular_idx == self.n_theta - 1:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_e - coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if (
                    angular_idx == self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if angular_idx == self.n_theta - 1 and radial_idx == self.n_radial - 1:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_e - coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                angular_idx = angular_idx + 1

            radial_idx = radial_idx + 1
            angular_idx = 0

        # Pressure field solution
        pressure_solution = np.linalg.solve(mat_coeff, source_term_vector)
        pressure_idx = -1

        # Pressure matrix
        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                pressure_idx = pressure_idx + 1
                self.pressure_field[ii, jj] = pressure_solution[pressure_idx, 0]

                # Pressure boundary conditions
                if self.pressure_field[ii, jj] < 0:
                    self.pressure_field[ii, jj] = 0

    def _get_interp_coeffs(self, T_muI, T_muF, mu_I, mu_F):
        """
        Calculate viscosity interpolation coefficients.

        Parameters
        ----------
        T_muI : float
            First temperature point [°C]
        T_muF : float
            Second temperature point [°C]
        mu_I : float
            Viscosity at first temperature [Pa·s]
        mu_F : float
            Viscosity at second temperature [Pa·s]

        Returns
        -------
        tuple
            Coefficients (a, b) for viscosity equation: μ = a * exp(b * T)
        """
        b = np.log(mu_F / mu_I) / (T_muF - T_muI)
        a = mu_I / np.exp(b * T_muI)
        return a, b

    def coefficients(self):
        perturbation_frequency = self.speed
        normalized_frequency = perturbation_frequency / self.speed

        self.mu = (1 / self.reference_viscosity) * self.viscosity_field

        radial_idx = 0
        angular_idx = 0
        vectorization_idx = -1
        volumes_number = (self.n_radial) * (self.n_theta)

        mat_coeff = np.zeros((volumes_number, volumes_number))
        source_vector = np.zeros((volumes_number, 1), dtype=complex)
        pressure_vector = np.zeros((volumes_number, 1), dtype=complex)
        pressure_field_coeff = np.zeros((self.n_radial, self.n_theta), dtype=complex)
        moment_x_radial_coeff = np.zeros((self.n_radial, self.n_theta), dtype=complex)
        moment_y_radial_coeff = np.zeros((self.n_radial, self.n_theta), dtype=complex)
        force_radial_coeff = np.zeros((self.n_radial, self.n_theta), dtype=complex)

        for radius in self.radius_array:
            for theta in self.theta_array:
                theta_e = theta + 0.5 * self.d_theta
                theta_w = theta - 0.5 * self.d_theta
                radius_n = radius + 0.5 * self.d_radius
                radius_s = radius - 0.5 * self.d_radius

                if angular_idx == 0 and radial_idx == 0:
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = self.mu[radial_idx, angular_idx]
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = self.mu[radial_idx, angular_idx]
                    dp_dtheta_e = (
                        self.pressure_field[radial_idx, angular_idx + 1]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_theta
                    dp_dtheta_w = self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_theta
                    )
                    dp_dr_n = (
                        self.pressure_field[radial_idx + 1, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_radius
                    dp_dr_s = self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_radius
                    )

                if (
                    angular_idx == 0
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = self.mu[radial_idx, angular_idx]
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )
                    dp_dtheta_e = (
                        self.pressure_field[radial_idx, angular_idx + 1]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_theta
                    dp_dtheta_w = self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_theta
                    )
                    dp_dr_n = (
                        self.pressure_field[radial_idx + 1, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_radius
                    dp_dr_s = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx - 1, angular_idx]
                    ) / self.d_radius

                if angular_idx == 0 and radial_idx == self.n_radial - 1:
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = self.mu[radial_idx, angular_idx]
                    mu_n = self.mu[radial_idx, angular_idx]
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )
                    dp_dtheta_e = (
                        self.pressure_field[radial_idx, angular_idx + 1]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_theta
                    dp_dtheta_w = self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_theta
                    )
                    dp_dr_n = -self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_radius
                    )
                    dp_dr_s = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx - 1, angular_idx]
                    ) / self.d_radius

                if (
                    radial_idx == 0
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = self.mu[radial_idx, angular_idx]
                    dp_dtheta_e = (
                        self.pressure_field[radial_idx, angular_idx + 1]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_theta
                    dp_dtheta_w = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx - 1]
                    ) / self.d_theta
                    dp_dr_n = (
                        self.pressure_field[radial_idx + 1, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_radius
                    dp_dr_s = self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_radius
                    )

                if (
                    angular_idx > 0
                    and angular_idx < self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )
                    dp_dtheta_e = (
                        self.pressure_field[radial_idx, angular_idx + 1]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_theta
                    dp_dtheta_w = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx - 1]
                    ) / self.d_theta
                    dp_dr_n = (
                        self.pressure_field[radial_idx + 1, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_radius
                    dp_dr_s = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx - 1, angular_idx]
                    ) / self.d_radius

                if (
                    radial_idx == self.n_radial - 1
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mu_e = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx + 1]
                    )
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = self.mu[radial_idx, angular_idx]
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )
                    dp_dtheta_e = (
                        self.pressure_field[radial_idx, angular_idx + 1]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_theta
                    dp_dtheta_w = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx - 1]
                    ) / self.d_theta
                    dp_dr_n = -self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_radius
                    )
                    dp_dr_s = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx - 1, angular_idx]
                    ) / self.d_radius

                if radial_idx == 0 and angular_idx == self.n_theta - 1:
                    mu_e = self.mu[radial_idx, angular_idx]
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = self.mu[radial_idx, angular_idx]
                    dp_dtheta_e = -self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_theta
                    )
                    dp_dtheta_w = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx - 1]
                    ) / self.d_theta
                    dp_dr_n = (
                        self.pressure_field[radial_idx + 1, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_radius
                    dp_dr_s = self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_radius
                    )

                if (
                    angular_idx == self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mu_e = self.mu[radial_idx, angular_idx]
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx + 1, angular_idx]
                    )
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )
                    dp_dtheta_e = -self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_theta
                    )
                    dp_dtheta_w = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx - 1]
                    ) / self.d_theta
                    dp_dr_n = (
                        self.pressure_field[radial_idx + 1, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx]
                    ) / self.d_radius
                    dp_dr_s = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx - 1, angular_idx]
                    ) / self.d_radius

                if angular_idx == self.n_theta - 1 and radial_idx == self.n_radial - 1:
                    mu_e = self.mu[radial_idx, angular_idx]
                    mu_w = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx, angular_idx - 1]
                    )
                    mu_n = self.mu[radial_idx, angular_idx]
                    mu_s = 0.5 * (
                        self.mu[radial_idx, angular_idx]
                        + self.mu[radial_idx - 1, angular_idx]
                    )
                    dp_dtheta_e = -self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_theta
                    )
                    dp_dtheta_w = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx, angular_idx - 1]
                    ) / self.d_theta
                    dp_dr_n = -self.pressure_field[radial_idx, angular_idx] / (
                        0.5 * self.d_radius
                    )
                    dp_dr_s = (
                        self.pressure_field[radial_idx, angular_idx]
                        - self.pressure_field[radial_idx - 1, angular_idx]
                    ) / self.d_radius

                pert_coeff_ne, pert_coeff_nw, pert_coeff_se, pert_coeff_sw = np.ones(4)
                rad_pert_ne, rad_pert_nw, rad_pert_se, rad_pert_sw = np.zeros(4)
                circ_pert_ne, circ_pert_nw, circ_pert_se, circ_pert_sw = np.zeros(4)

                # Dynamic coefficients calculation terms
                energy_e = (
                    1
                    / (24 * self.pad_arc_length**2 * mu_e)
                    * (self.d_radius / self.d_theta)
                    * (
                        pert_coeff_ne
                        * self.film_thickness_ne[radial_idx, angular_idx] ** 3
                        / radius_n
                        + pert_coeff_se
                        * self.film_thickness_se[radial_idx, angular_idx] ** 3
                        / radius_s
                    )
                )
                diffusion_e = (
                    self.d_radius
                    / (48 * self.pad_arc_length**2 * mu_e)
                    * (
                        circ_pert_ne
                        * self.film_thickness_ne[radial_idx, angular_idx] ** 3
                        / radius_n
                        + circ_pert_se
                        * self.film_thickness_se[radial_idx, angular_idx] ** 3
                        / radius_s
                    )
                )
                coeff_e = energy_e + diffusion_e

                energy_w = (
                    1
                    / (24 * self.pad_arc_length**2 * mu_w)
                    * (self.d_radius / self.d_theta)
                    * (
                        pert_coeff_nw
                        * self.film_thickness_nw[radial_idx, angular_idx] ** 3
                        / radius_n
                        + pert_coeff_sw
                        * self.film_thickness_sw[radial_idx, angular_idx] ** 3
                        / radius_s
                    )
                )
                diffusion_w = (
                    -self.d_radius
                    / (48 * self.pad_arc_length**2 * mu_w)
                    * (
                        circ_pert_nw
                        * self.film_thickness_nw[radial_idx, angular_idx] ** 3
                        / radius_n
                        + circ_pert_sw
                        * self.film_thickness_sw[radial_idx, angular_idx] ** 3
                        / radius_s
                    )
                )
                coeff_w = energy_w + diffusion_w

                energy_n = (
                    radius_n
                    / (24 * mu_n)
                    * (self.d_theta / self.d_radius)
                    * (
                        pert_coeff_ne
                        * self.film_thickness_ne[radial_idx, angular_idx] ** 3
                        + pert_coeff_nw
                        * self.film_thickness_nw[radial_idx, angular_idx] ** 3
                    )
                )
                diffusion_n = (
                    radius_n
                    / (48 * mu_n)
                    * (self.d_theta)
                    * (
                        rad_pert_ne
                        * self.film_thickness_ne[radial_idx, angular_idx] ** 3
                        + rad_pert_nw
                        * self.film_thickness_nw[radial_idx, angular_idx] ** 3
                    )
                )
                coeff_n = energy_n + diffusion_n

                energy_s = (
                    radius_s
                    / (24 * mu_s)
                    * (self.d_theta / self.d_radius)
                    * (
                        pert_coeff_se
                        * self.film_thickness_se[radial_idx, angular_idx] ** 3
                        + pert_coeff_sw
                        * self.film_thickness_sw[radial_idx, angular_idx] ** 3
                    )
                )
                diffusion_s = (
                    -radius_s
                    / (48 * mu_s)
                    * (self.d_theta)
                    * (
                        rad_pert_se
                        * self.film_thickness_se[radial_idx, angular_idx] ** 3
                        + rad_pert_sw
                        * self.film_thickness_sw[radial_idx, angular_idx] ** 3
                    )
                )
                coeff_s = energy_s + diffusion_s

                coeff_c = -(energy_e + energy_w + energy_n + energy_s) + (
                    diffusion_e + diffusion_w + diffusion_n + diffusion_s
                )

                radial_term = (radius_n * self.d_theta / (8 * mu_n)) * dp_dr_n * (
                    pert_coeff_ne * self.film_thickness_ne[radial_idx, angular_idx] ** 2
                    + pert_coeff_nw
                    * self.film_thickness_nw[radial_idx, angular_idx] ** 2
                ) - (radius_s * self.d_theta / (8 * mu_s)) * dp_dr_s * (
                    pert_coeff_se * self.film_thickness_se[radial_idx, angular_idx] ** 2
                    + pert_coeff_sw
                    * self.film_thickness_sw[radial_idx, angular_idx] ** 2
                )
                circ_term = (
                    self.d_radius / (8 * self.pad_arc_length**2 * mu_e)
                ) * dp_dtheta_e * (
                    pert_coeff_ne
                    * self.film_thickness_ne[radial_idx, angular_idx] ** 2
                    / radius_n
                    + pert_coeff_se
                    * self.film_thickness_se[radial_idx, angular_idx] ** 2
                    / radius_s
                ) - (
                    self.d_radius / (8 * self.pad_arc_length**2 * mu_w)
                ) * dp_dtheta_w * (
                    pert_coeff_nw
                    * self.film_thickness_nw[radial_idx, angular_idx] ** 2
                    / radius_n
                    + pert_coeff_sw
                    * self.film_thickness_sw[radial_idx, angular_idx] ** 2
                    / radius_s
                )
                conv_term = self.d_radius / (4 * self.pad_arc_length) * (
                    pert_coeff_ne * radius_n + pert_coeff_se * radius_s
                ) - self.d_radius / (4 * self.pad_arc_length) * (
                    pert_coeff_nw * radius_n + pert_coeff_sw * radius_s
                )
                inertial_term = (
                    complex(0, 1)
                    * normalized_frequency
                    * self.d_radius
                    * self.d_theta
                    / 4
                    * (
                        radius_n * pert_coeff_ne
                        + radius_n * pert_coeff_nw
                        + radius_s * pert_coeff_se
                        + radius_s * pert_coeff_sw
                    )
                )

                # Vectorization index
                vectorization_idx = vectorization_idx + 1

                source_vector[vectorization_idx, 0] = (
                    -(radial_term + circ_term) + conv_term + inertial_term
                )

                if angular_idx == 0 and radial_idx == 0:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_w - coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if (
                    angular_idx == 0
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                if angular_idx == 0 and radial_idx == self.n_radial - 1:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_w - coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                if (
                    radial_idx == 0
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_s
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if (
                    angular_idx > 0
                    and angular_idx < self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e

                if (
                    radial_idx == self.n_radial - 1
                    and angular_idx > 0
                    and angular_idx < self.n_theta - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_n
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + 1] = coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                if radial_idx == 0 and angular_idx == self.n_theta - 1:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_e - coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if (
                    angular_idx == self.n_theta - 1
                    and radial_idx > 0
                    and radial_idx < self.n_radial - 1
                ):
                    mat_coeff[vectorization_idx, vectorization_idx] = coeff_c - coeff_e
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )
                    mat_coeff[vectorization_idx, vectorization_idx + self.n_theta] = (
                        coeff_n
                    )

                if angular_idx == self.n_theta - 1 and radial_idx == self.n_radial - 1:
                    mat_coeff[vectorization_idx, vectorization_idx] = (
                        coeff_c - coeff_e - coeff_n
                    )
                    mat_coeff[vectorization_idx, vectorization_idx - 1] = coeff_w
                    mat_coeff[vectorization_idx, vectorization_idx - self.n_theta] = (
                        coeff_s
                    )

                angular_idx = angular_idx + 1

            radial_idx = radial_idx + 1
            angular_idx = 0

        # Vectorized pressure field solution
        pressure_vector = np.linalg.solve(mat_coeff, source_vector)
        pressure_idx = -1

        # Pressure matrix
        for ii in range(0, self.n_radial):
            for jj in range(0, self.n_theta):
                pressure_idx = pressure_idx + 1
                pressure_field_coeff[ii, jj] = pressure_vector[pressure_idx, 0]

        # Dimensional pressure
        pressure_coeff_dim = (
            pressure_field_coeff
            * (self.pad_inner_radius**2)
            * self.speed
            * self.reference_viscosity
            / (self.pivot_film_thickness**3)
        )

        # Resulting force and momentum: Equilibrium position
        radius_coords = self.pad_inner_radius * self.radius_array
        theta_coords = self.pad_arc_length * self.theta_array
        pivot_coords = self.pad_pivot_radius * (np.ones((np.size(radius_coords))))

        for ii in range(0, self.n_theta):
            moment_x_radial_coeff[:, ii] = (
                pressure_coeff_dim[:, ii] * (np.transpose(radius_coords) ** 2)
            ) * np.sin(theta_coords[ii] - self.angular_pivot_position)
            moment_y_radial_coeff[:, ii] = (
                -pressure_coeff_dim[:, ii]
                * np.transpose(radius_coords)
                * np.transpose(
                    radius_coords
                    * np.cos(theta_coords[ii] - self.angular_pivot_position)
                    - pivot_coords
                )
            )
            force_radial_coeff[:, ii] = pressure_coeff_dim[:, ii] * np.transpose(
                radius_coords
            )

        force_radial = np.trapezoid(force_radial_coeff, theta_coords)

        force_axial = -np.trapezoid(force_radial, radius_coords)

        self.kzz = self.n_pad * np.real(force_axial)
        self.czz = self.n_pad * 1 / perturbation_frequency * np.imag(force_axial)

    def plot_results(self):
        """Plot pressure and temperature field results.

        Creates 3D surface plots and 2D contour plots for both pressure
        and temperature fields using Plotly.
        """
        # Define coordinate vectors and matrices
        radial_coords = np.zeros(self.n_radial + 2)
        angular_coords = np.zeros(self.n_theta + 2)
        x_coords = np.zeros((self.n_radial + 2, self.n_theta + 2))
        y_coords = np.zeros((self.n_radial + 2, self.n_theta + 2))

        # Set boundary values
        radial_coords[0] = self.pad_outer_radius
        radial_coords[-1] = self.pad_inner_radius
        radial_coords[1 : self.n_radial + 1] = np.arange(
            self.pad_outer_radius - 0.5 * self.d_radius * self.pad_inner_radius,
            self.pad_inner_radius,
            -(self.d_radius * self.pad_inner_radius),
        )

        angular_coords[0] = np.pi / 2 + self.pad_arc_length / 2
        angular_coords[-1] = np.pi / 2 - self.pad_arc_length / 2
        angular_coords[1 : self.n_theta + 1] = np.arange(
            np.pi / 2
            + self.pad_arc_length / 2
            - (0.5 * self.d_theta * self.pad_arc_length),
            np.pi / 2 - self.pad_arc_length / 2,
            -self.d_theta * self.pad_arc_length,
        )

        # Create coordinate mesh
        for i in range(self.n_radial + 2):
            for j in range(self.n_theta + 2):
                x_coords[i, j] = radial_coords[i] * np.cos(angular_coords[j])
                y_coords[i, j] = radial_coords[i] * np.sin(angular_coords[j])

        # Calculate angle range for plotting
        angle_per_pad = 360 / self.n_pad
        angle_step = 20 / 10
        angle_range = np.arange(
            angle_per_pad + (angle_per_pad - 20) - 40,
            angle_per_pad + angle_per_pad + angle_step - 40,
            angle_step,
        )

        # Plot 3D pressure field
        self._plot_3d_surface(
            x_coords,
            y_coords,
            self.pressure_field_dimensional,
            "Pressure field",
            "Pressure [Pa]",
            angle_range,
        )

        # Plot 3D temperature field
        self._plot_3d_surface(
            x_coords,
            y_coords,
            self.temperature_field,
            "Temperature field",
            "Temperature [°C]",
            angle_range,
        )

        # Create interpolation grid for contour plots
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_interp = np.linspace(x_min, x_max, 800)
        y_interp = np.linspace(y_min, y_max, 800)
        x_grid, y_grid = np.meshgrid(x_interp, y_interp)

        # Plot 2D temperature contour
        temp_interpolated = griddata(
            (x_coords.flatten(), y_coords.flatten()),
            self.temperature_field.flatten(),
            (x_grid, y_grid),
            method="cubic",
        )
        self._plot_2d_contour(
            x_grid, y_grid, temp_interpolated, "Temperature field", "Temperature (°C)"
        )

        # Plot 2D pressure contour
        pressure_interpolated = griddata(
            (x_coords.flatten(), y_coords.flatten()),
            self.pressure_field_dimensional.flatten(),
            (x_grid, y_grid),
            method="cubic",
        )
        self._plot_2d_contour(
            x_grid, y_grid, pressure_interpolated, "Pressure field", "Pressure (Pa)"
        )

    def _plot_3d_surface(self, x_coords, y_coords, z_data, title, z_label, angle_range):
        """Create 3D surface plot using Plotly with tableau colors.

        Parameters
        ----------
        x_coords : ndarray
            X coordinates mesh
        y_coords : ndarray
            Y coordinates mesh
        z_data : ndarray
            Z data for surface plot
        title : str
            Plot title
        z_label : str
            Z-axis label
        angle_range : ndarray
            Angle range for axis limits
        """
        fig = go.Figure()
        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0),
            title=title,
            plot_bgcolor="white",
        )

        colorscale = [
            [0, tableau_colors["blue"]],
            [0.25, tableau_colors["cyan"]],
            [0.5, tableau_colors["green"]],
            [0.75, tableau_colors["orange"]],
            [1, tableau_colors["red"]],
        ]

        fig.add_trace(
            go.Surface(
                x=x_coords,
                y=y_coords,
                z=z_data,
                colorscale=colorscale,
                colorbar=dict(
                    title=z_label,
                    tickfont=dict(size=22),
                ),
            )
        )

        fig.update_layout(
            xaxis_range=[np.min(angle_range), np.max(angle_range)],
            yaxis_range=[np.min(y_coords), np.max(y_coords)],
        )

        fig.update_scenes(
            xaxis_title=dict(
                text="Angular length [rad]",
                font=dict(family="Times New Roman", size=22),
            ),
            xaxis_tickfont=dict(family="Times New Roman", size=14),
            yaxis_title=dict(
                text="Radial length [m]", font=dict(family="Times New Roman", size=22)
            ),
            yaxis_tickfont=dict(family="Times New Roman", size=14),
            zaxis_title=dict(
                text=z_label, font=dict(family="Times New Roman", size=22)
            ),
            zaxis_tickfont=dict(family="Times New Roman", size=14),
            aspectratio=dict(x=1.8, y=1.8, z=1.8),
        )

        # Camera position
        camera = dict(eye=dict(x=-1.5, y=-4, z=1.5), center=dict(x=0, y=0, z=0))
        fig.update_layout(scene_camera=camera)

        fig.show()

    def _plot_2d_contour(self, x_grid, y_grid, z_data, title, colorbar_title):
        """Create 2D contour plot using Plotly with tableau colors.

        Parameters
        ----------
        x_grid : ndarray
            X coordinates grid
        y_grid : ndarray
            Y coordinates grid
        z_data : ndarray
            Z data for contour plot
        title : str
            Plot title
        colorbar_title : str
            Colorbar title
        """
        fig = go.Figure()

        colorscale = [
            [0, tableau_colors["blue"]],
            [0.25, tableau_colors["cyan"]],
            [0.5, tableau_colors["green"]],
            [0.75, tableau_colors["orange"]],
            [1, tableau_colors["red"]],
        ]

        fig.add_trace(
            go.Contour(
                x=x_grid[0],
                y=y_grid[:, 0],
                z=z_data,
                ncontours=15,
                contours=dict(start=np.nanmin(z_data), end=np.nanmax(z_data)),
                colorscale=colorscale,
                colorbar=dict(
                    title=colorbar_title,
                    tickfont=dict(size=22),
                ),
            )
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0),
            title=title,
            plot_bgcolor="white",
        )

        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )

        fig.update_layout(xaxis_range=[np.min(x_grid), np.max(x_grid)])
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(
                text="X Direction (m)", font=dict(family="Times New Roman", size=30)
            ),
            range=[np.min(x_grid), np.max(x_grid)],
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(
                text="Y Direction (m)", font=dict(family="Times New Roman", size=30)
            ),
        )

        fig.show()


def thrust_pad_example():
    """Create an example of a thrust bearing with Thermo-Hydro-Dynamic effects.

    This function creates and returns a ThrustPad bearing instance with predefined
    parameters for demonstration purposes. The bearing is configured with 12 pads
    and operates at 90 RPM.

    Returns
    -------
    bearing : ThrustPad
        A configured thrust bearing instance ready for analysis.

    Examples
    --------
    >>> from ross.bearings.thrust_pad import thrust_pad_example
    >>> bearing = thrust_pad_example()

    Notes
    -----
    This example uses the following configuration:
    - 12 tilting pads with 26° arc length each
    - Pad inner radius: 1150 mm
    - Pad outer radius: 1725 mm
    - Pad pivot radius: 1442.5 mm
    - Angular pivot position: 15°
    - ISOVG68 lubricant at 40°C
    - Operating frequency: 90 RPM
    - Equilibrium position mode: "calculate"
    - Axial load: 13.32 MN
    - Initial film thickness: 0.2 mm
    - Radial inclination angle: -2.75e-04 rad
    - Circumferential inclination angle: -1.70e-05 rad
    """

    bearing = ThrustPad(
        n=1,
        pad_inner_radius=Q_(1150, "mm"),
        pad_outer_radius=Q_(1725, "mm"),
        pad_pivot_radius=Q_(1442.5, "mm"),
        pad_arc_length=Q_(26, "deg"),
        angular_pivot_position=Q_(15, "deg"),
        oil_supply_temperature=Q_(40, "degC"),
        lubricant="ISOVG68",
        n_pad=12,
        n_theta=10,
        n_radial=10,
        frequency=Q_([90], "RPM"),
        equilibrium_position_mode="calculate",
        model_type="thermo_hydro_dynamic",
        fzs_load=13.320e6,
        radial_inclination_angle=Q_(-2.75e-04, "rad"),
        circumferential_inclination_angle=Q_(-1.70e-05, "rad"),
        initial_film_thickness=Q_(0.2, "mm"),
        print_result=False,
        print_progress=False,
        print_time=False,
        compare_coefficients=False,
    )

    return bearing
