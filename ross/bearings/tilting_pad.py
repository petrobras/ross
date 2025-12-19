import time
import numpy as np
from scipy.optimize import fmin, minimize
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from prettytable import PrettyTable

from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from ross.plotly_theme import tableau_colors
from ross.bearings.lubricants import lubricants_dict


class TiltingPad(BearingElement):
    """Tilting-pad journal bearing - Thermo-Hydro-Dynamic (THD) model.

    This class provides a **comprehensive numerical model** for tilting-pad journal
    bearings using thermo-hydro-dynamic (THD) analysis. Each pad is treated independently
    with its own pressure and temperature fields, pivot mechanics, and load distribution.

    **Theoretical Approach:**

    The model solves the **complete THD problem** for each pad using:

    1. **Reynolds Equation** (for pressure field):
       - 2D finite difference method on a structured grid (nx × nz)
       - Accounts for pad rotation and journal motion
       - Enforces zero pressure at pad edges (cavitation boundary)
       - Viscosity varies spatially due to temperature field

    2. **Energy Equation** (for temperature field):
       - 2D finite difference with upwind scheme
       - Includes viscous dissipation and heat conduction
       - Models turbulent effects using Reynolds number-dependent viscosity
       - Oil supply temperature as boundary condition

    3. **Equilibrium Calculation**:
       - Iterative optimization to find journal equilibrium position
       - Two modes: match specified eccentricity or determine complete equilibrium
       - Minimizes pad moment imbalance using Nelder-Mead optimization (fmin)
       - Each pad rotates independently about its pivot point

    4. **Dynamic Coefficients** (stiffness and damping):
       - Uses virtual perturbations of displacements and speeds to determine the coefficients
       - Applies small perturbations to journal position (0.5% of clearance)
       - Applies small perturbations to journal velocity (2.5% of operating speed)
       - Solves complete THD problem for each perturbation state
       - Extracts coefficients from force differences

    For reference check :cite:`barbosa2018`, :cite:`heinrichson2007influence` and :cite:`nicoletti1999`.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    journal_diameter : float
        Journal diameter. Default unit is meter.
    pre_load : array_like
        Pre-load factor for each pad. Dimensionless.
    pad_thickness : float
        Pad thickness. Default unit is meter.
    pad_arc : array_like
        Individual pad arc angle for each pad. Default unit is degrees.
    offset : array_like
        Pivot offset for each pad. Dimensionless (0.5 = centered).
    pad_axial_length : array_like
        Pad axial length for each pad. Default unit is meter.
    lubricant : str or dict
        Lubricant type. Can be:
        - 'ISOVG32'
        - 'ISOVG46'
        - 'ISOVG68'
        Or a dictionary with lubricant properties.
    oil_supply_temperature : float
        Oil supply temperature. Default unit is °C.
    radial_clearance : float
        Radial clearance. Default unit is meter.
    pivot_angle : array_like
        Pivot angle for each pad. Default unit is degrees.
    frequency : array_like
        Operating frequencies. Default unit is RPM.
    nx : int, optional
        Number of volumes along the circumferential direction. Default is 30.
    nz : int, optional
        Number of volumes along the axial direction. Default is 30.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
    xj : float, optional
        Journal position in X direction. Default unit is meter.
    yj : float, optional
        Journal position in Y direction. Default unit is meter.
    equilibrium_type: str, optional
        Type of equilibrium calculation. Options:
        - 'match_eccentricity': Uses the provided eccentricity and
        attitude_angle and optimizes only the pad rotation angles.
        - 'determine_eccentricity': Optimizes eccentricity, attitude_angle,
        and pad rotation angles to balance the applied loads; requires
        both fxs_load and fys_load to be provided.
    model_type : str, optional
        Type of model to be used. Options:
        - 'thermo_hydro_dynamic': Thermo-Hydro-Dynamic model
    eccentricity : float, optional
        Eccentricity ratio. Dimensionless.
    attitude_angle : float, optional
        Attitude angle. Default unit is degrees.
    fxs_load : float, optional
        External load in X direction. Default unit is Newton.
    fys_load : float, optional
        External load in Y direction. Default unit is Newton.
    initial_pads_angles : array_like, optional
        Initial pad angles. Default unit is radians.
    **kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    None
        The class instance contains all calculated results as attributes.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Attributes
    ----------
    kxx, kyy, kxy, kyx : float
        Stiffness coefficients in N/m.
    cxx, cyy, cxy, cyx : float
        Damping coefficients in N·s/m.
    pressure_dim : array
        Dimensional pressure field in Pa.
    temperature_init : array
        Temperature field in °C.
    force_x_dim, force_y_dim : array
        Dimensional forces in N.
    moment_j_dim : array
        Dimensional moments in N·m.
    maxP : float
        Maximum pressure in Pa.
    maxT : float
        Maximum temperature in °C.
    h_pivot : float
        Oil film thickness at pivot point in m.
    ecc : float
        Eccentricity ratio.

    Examples
    --------
    >>> from ross.bearings.tilting_pad import TiltingPad
    >>> from ross.units import Q_
    >>> bearing = TiltingPad(
    ...     n=1,
    ...     frequency=Q_([3000], "RPM"),
    ...     equilibrium_type="match_eccentricity",
    ...     journal_diameter=101.6e-3,
    ...     radial_clearance=74.9e-6,
    ...     pad_thickness=12.7e-3,
    ...     pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
    ...     pad_arc=Q_([60]*5, "deg"),
    ...     pad_axial_length=Q_([50.8e-3]*5, "m"),
    ...     pre_load=[0.5]*5,
    ...     offset=[0.5]*5,
    ...     lubricant="ISOVG32",
    ...     oil_supply_temperature=Q_(40, "degC"),
    ...     eccentricity=0.483,
    ...     attitude_angle=Q_(267.5, "deg")
    ... )
    """

    @check_units
    def __init__(
        self,
        n,
        journal_diameter,
        pre_load,
        pad_thickness,
        pad_arc,
        offset,
        pad_axial_length,
        lubricant,
        oil_supply_temperature,
        radial_clearance,
        pivot_angle,
        frequency,
        nx=30,
        nz=30,
        n_link=None,
        xj=None,
        yj=None,
        equilibrium_type=None,
        eccentricity=0.3,
        attitude_angle=np.pi * 3/2,  # 270°
        fxs_load=None,
        fys_load=None,
        initial_pads_angles=None,
        model_type="thermo_hydro_dynamic",
        **kwargs,
    ):
        self.n = n
        self.n_link = n_link
        self.model_type = model_type

        # Bearing Geometry
        self.journal_radius = journal_diameter / 2
        self.n_pad = len(pre_load)
        self.radial_clearance = radial_clearance
        self.pad_radius = self.journal_radius + self.radial_clearance / (
            1 - pre_load[0]
        )
        self.pad_thickness = pad_thickness
        self.pivot_angle = pivot_angle
        self.eccentricity = eccentricity
        self.attitude_angle = attitude_angle
        # Get the first value of the arrays
        self.pad_arc = pad_arc[0]
        self.offset = offset[0]
        self.pad_axial_length = pad_axial_length[0]

        # Initial position of the journal
        self.xj = xj
        self.yj = yj

        # Initial position of the pivot
        self.x_pt = 0
        self.y_pt = 0

        # Operating conditions
        self.equilibrium_type = equilibrium_type

        if self.equilibrium_type == "determine_eccentricity":
            if fxs_load is None or fys_load is None:
                raise ValueError(
                    "fxs_load and fys_load must be provided when"
                    " equilibrium_type is 'determine_eccentricity'."
                )

        self.oil_supply_temperature = Q_(oil_supply_temperature, "degK").m_as("degC")
        self.reference_temperature = self.oil_supply_temperature
        self.lubricant = lubricant
        self.frequency = frequency
        self.speed = None
        self.fxs_load = fxs_load
        self.fys_load = fys_load
        self.initial_pads_angles = initial_pads_angles

        # Mesh discretization and equation's terms setup
        self.nx = nx
        self.nz = nz
        self.z1 = -0.5
        self.z2 = 0.5
        self.theta_1 = -self.offset * self.pad_arc
        self.theta_2 = (1 - self.offset) * self.pad_arc
        self.dtheta = (self.theta_2 - self.theta_1) / self.nx
        self.dz = (self.z2 - self.z1) / self.nz
        self.dx = self.dtheta / self.pad_arc
        self.xz = np.zeros(self.nz)
        self.xz[0] = self.z1 + 0.5 * self.dz
        self.xtheta = np.zeros(self.nx)
        self.xtheta[0] = self.theta_1 + 0.5 * self.dtheta

        # Lubricant properties setup
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
        self.mu_0 = self.a_a * np.exp(self.b_b * self.oil_supply_temperature)

        # Discretization of the mesh
        for ii in range(1, self.nz):
            self.xz[ii] = self.xz[ii - 1] + self.dz

        for jj in range(1, self.nx):
            self.xtheta[jj] = self.xtheta[jj - 1] + self.dtheta

        self._initialize_arrays()

        # Storage lists for results at each frequency
        self.maxP_list = []
        self.maxT_list = []
        self.minH_list = []
        self.ecc_list = []
        self.attitude_angle_list = []
        self.psi_pad_list = []
        self.force_x_total_list = []
        self.force_y_total_list = []
        self.momen_rot_list = []

        # Storage lists for field data at each frequency
        self.pressure_fields = []
        self.temperature_fields = []

        # Optimization convergence tracking
        self.optimization_history = {}

        n_freq = np.shape(self.frequency)[0]

        kxx = np.zeros(n_freq)
        kxy = np.zeros(n_freq)
        kyx = np.zeros(n_freq)
        kyy = np.zeros(n_freq)

        cxx = np.zeros(n_freq)
        cxy = np.zeros(n_freq)
        cyx = np.zeros(n_freq)
        cyy = np.zeros(n_freq)

        for i in range(n_freq):
            self.speed = self.frequency[i]
            self._current_freq_index = i
            self.optimization_history[i] = []

            if self.model_type == "thermo_hydro_dynamic":
                self.run_thermo_hydro_dynamic()

            kxx[i], kxy[i], kyx[i], kyy[i] = self.kxx, self.kxy, self.kyx, self.kyy
            cxx[i], cxy[i], cyx[i], cyy[i] = self.cxx, self.cxy, self.cyx, self.cyy

        super().__init__(
            n,
            kxx=kxx,
            cxx=cxx,
            kyy=kyy,
            kxy=kxy,
            kyx=kyx,
            cyy=cyy,
            cxy=cxy,
            cyx=cyx,
            frequency=frequency,
            **kwargs,
        )

    def run_thermo_hydro_dynamic(self):
        """
        Execute the complete thermo-hydrodynamic analysis for the tilting pad bearing.

        This method performs the main computational sequence for analyzing a tilting pad
        journal bearing, including pressure and temperature field calculations, equilibrium
        position determination, and dynamic coefficient computation for each operating
        frequency.

        The analysis includes:
        - Initialization of field arrays (pressure, temperature, film thickness)
        - Iterative solution of Reynolds and energy equations for each frequency
        - Calculation of hydrodynamic forces and moments
        - Computation of stiffness and damping coefficients
        - Generation of result plots and output

        Parameters
        ----------
        None
            This method uses the bearing parameters defined during initialization.

        Returns
        -------
        None
            Results are stored as instance attributes and plots are generated.

        Attributes
        ----------
        pressure_dim : ndarray
            Dimensional pressure field for all pads [Pa]. Shape: (nz, nx, n_pad).
        temperature_init : ndarray
            Temperature field for all pads [°C]. Shape: (nz, nx, n_pad).
        h_pivot : ndarray
            Oil film thickness at pivot point for each pad [m]. Shape: (n_pad,).
        kxx, kyy, kxy, kyx : float
            Stiffness coefficients [N/m].
        cxx, cyy, cxy, cyx : float
            Damping coefficients [N·s/m].
        force_x_dim, force_y_dim : ndarray
            Dimensional forces in X and Y directions [N]. Shape: (n_pad,).
        moment_j_dim : ndarray
            Dimensional moments [N·m]. Shape: (n_pad,).

        Notes
        -----
        The method processes each frequency in the frequency array sequentially.
        For each frequency, it:
        1. Initializes field arrays and dimensionless parameters
        2. Solves the thermo-hydrodynamic equations to find equilibrium
        3. Calculates dynamic coefficients using perturbation methods
        4. Updates the parent BearingElement with computed coefficients
        5. Generates visualization plots

        The analysis assumes steady-state operation and uses finite difference
        methods for solving the governing equations.

        Examples
        --------
        >>> from ross.bearings.tilting_pad import tilting_pad_example
        >>> bearing = tilting_pad_example()
        """

        self.initial_time = time.time()

        self.dimensionless_force = np.full(
            self.n_pad,
            1
            / (
                self.radial_clearance**2
                / (self.pad_radius**3 * self.mu_0 * self.speed * self.pad_axial_length)
            ),
        )

        self._reset_force_arrays()

        maxP, medP, maxT, medT, h_pivot, ecc = self.solve_fields()

        self.coefficients()

        self.final_time = time.time()

    def coefficients(self):
        """
        Calculate dynamic stiffness and damping coefficients for the tilting pad bearing.

        This method computes the dynamic coefficients (stiffness and damping) of the
        tilting pad bearing using a perturbation approach. The coefficients are
        determined by applying small perturbations to the journal position and velocity
        and calculating the resulting force changes.

        The method performs the following steps:
        1. Applies four types of perturbations (x-displacement, y-displacement,
        x-velocity, y-velocity) to each pad
        2. Solves the Reynolds and energy equations for each perturbation
        3. Calculates force differences due to perturbations
        4. Computes per-pad stiffness and damping coefficients
        5. Transforms coefficients from pad coordinate system to inertial system
        6. Reduces the multi-pad system to equivalent 2x2 coefficient matrices

        Parameters
        ----------
        None
            This method uses the bearing parameters and equilibrium position from
            the solve_fields() method.

        Returns
        -------
        None
            Results are stored as instance attributes.

        Attributes
        ----------
        kxx, kyy, kxy, kyx : float
            Stiffness coefficients in inertial coordinate system [N/m].
        cxx, cyy, cxy, cyx : float
            Damping coefficients in inertial coordinate system [N·s/m].
        K : ndarray
            Per-pad stiffness matrix. Shape: (n_pad, 3, 3).
        C : ndarray
            Per-pad damping matrix. Shape: (n_pad, 3, 3).
        Sjpt : ndarray
            Complex dynamic matrix for each pad. Shape: (n_pad, 3, 3).
        Sjipt : ndarray
            Transformed complex dynamic matrix for each pad. Shape: (n_pad, 3, 3).
        Sw : ndarray
            Final reduced complex dynamic matrix. Shape: (2, 2).

        Notes
        -----
        The perturbation method uses:
        - Space perturbation: 0.5% of radial clearance
        - Speed perturbation: 2.5% of operating speed × space perturbation

        The method applies four perturbation types:
        - a_p = 0: X-displacement perturbation
        - a_p = 1: Y-displacement perturbation
        - a_p = 2: X-velocity perturbation
        - a_p = 3: Y-velocity perturbation

        For each perturbation, the method:
        1. Solves Reynolds equation for pressure field
        2. Solves energy equation for temperature field
        3. Calculates hydrodynamic forces and moments
        4. Computes force differences and coefficients

        The final coefficients are obtained by matrix reduction from the
        multi-pad system to an equivalent 2x2 system representing the
        overall bearing behavior.

        The method assumes the equilibrium position has been previously
        calculated by the solve_fields() method.
        """

        # Loads initialization
        del_force_x = np.zeros(self.n_pad)
        del_force_y = np.zeros(self.n_pad)
        del_moment_j = np.zeros(self.n_pad)

        # Stiffness coefficients initialization
        k_xx = np.zeros(self.n_pad)
        k_tt = np.zeros(self.n_pad)
        k_yy = np.zeros(self.n_pad)
        k_xt = np.zeros(self.n_pad)
        k_tx = np.zeros(self.n_pad)
        k_yx = np.zeros(self.n_pad)
        k_xy = np.zeros(self.n_pad)
        k_yt = np.zeros(self.n_pad)
        k_ty = np.zeros(self.n_pad)
        self.K = np.zeros((self.n_pad, 3, 3))

        # Damping coefficients initialization
        c_xx = np.zeros(self.n_pad)
        c_tt = np.zeros(self.n_pad)
        c_yy = np.zeros(self.n_pad)
        c_xt = np.zeros(self.n_pad)
        c_tx = np.zeros(self.n_pad)
        c_yx = np.zeros(self.n_pad)
        c_xy = np.zeros(self.n_pad)
        c_yt = np.zeros(self.n_pad)
        c_ty = np.zeros(self.n_pad)
        self.C = np.zeros((self.n_pad, 3, 3))

        self.Sjpt = np.zeros((self.n_pad, 3, 3), dtype="complex")
        self.Sjipt = np.zeros((self.n_pad, 3, 3), dtype="complex")
        self.Aj = np.zeros((2, 2), dtype="complex")
        self.Hj = np.zeros((2, self.n_pad), dtype="complex")
        self.Vj = np.zeros((self.n_pad, 2), dtype="complex")
        self.Bj = np.zeros((self.n_pad, self.n_pad), dtype="complex")
        self.Sj = np.zeros((self.n_pad, 2, 2), dtype="complex")
        self.Tj = np.zeros((self.n_pad, 3, 3))
        self.Sw = np.zeros((2, 2), dtype="complex")

        # Perturbation setup
        self.space_perturbation = 0.005 * self.radial_clearance
        self.speed_perturbation = 0.025 * self.speed * self.space_perturbation

        psi_pad = np.zeros(self.n_pad)
        n_k = self.nx * self.nz

        temperature_tolerance = 0.1

        # DIAGNOSTIC: Check equilibrium forces BEFORE perturbations
        print(f"\n[DIAG EQ] ========== EQUILIBRIUM STATE CHECK ==========")
        print(f"  xdin (equilibrium state) = {self.xdin}")
        print(f"  eccentricity = {self.eccentricity:.6f}")
        print(f"  attitude_angle = {np.degrees(self.attitude_angle):.2f}°")
        print(f"  psi_pad (deg) = {[np.degrees(p) for p in self.xdin[2:]]}")
        print(f"  force_1 (all pads) = {self.force_1}")
        print(f"  force_2 (all pads) = {self.force_2}")
        print(f"  force_x_dim (all pads) = {self.force_x_dim}")
        print(f"  force_y_dim (all pads) = {self.force_y_dim}")
        print(f"  Total Fx = {np.sum(self.force_x_dim):.2f} N (expected: {self.fxs_load if self.fxs_load is not None else 'N/A'})")
        print(f"  Total Fy = {np.sum(self.force_y_dim):.2f} N (expected: {self.fys_load if self.fys_load is not None else 'N/A'})")
        print(f"[DIAG EQ] ============================================\n")

        for a_p in range(4):
            for n_p in range(self.n_pad):
                xx_coef = self.xdin[:2]

                for k_pad in range(self.n_pad):
                    psi_pad[k_pad] = self.xdin[k_pad + 2]

                temperature_referance = self.temperature_init[:, :, n_p]
                temperature_iteration = 1.1 * temperature_referance
                cont_temp = 0

                while (
                    abs((temperature_referance - temperature_iteration).max())
                    >= temperature_tolerance
                ):
                    cont_temp += 1
                    temperature_iteration = np.array(temperature_referance)

                    mi_i = self.a_a * np.exp(self.b_b * temperature_iteration)
                    mi = mi_i / self.mu_0

                    k_idx = 0
                    mat_coef = np.zeros((n_k, n_k))
                    b_vec = np.zeros(n_k)

                    xryr, xryrpt, xr, yr, xrpt, yrpt = self._transform_coordinates(n_p)

                    alphapt = 0

                    if a_p == 0:
                        xr += self.space_perturbation
                    elif a_p == 1:
                        yr += self.space_perturbation
                    if a_p == 2:
                        xrpt += self.speed_perturbation
                    if a_p == 3:
                        yrpt += self.speed_perturbation

                    alpha = psi_pad[n_p]

                    for ii in range(self.nz):
                        for jj in range(self.nx):
                            teta_e = self.xtheta[jj] + 0.5 * self.dtheta
                            teta_w = self.xtheta[jj] - 0.5 * self.dtheta

                            h_p = (
                                self.pad_radius
                                - self.journal_radius
                                - (
                                    np.sin(self.xtheta[jj])
                                    * (
                                        yr
                                        + alpha * (self.pad_radius + self.pad_thickness)
                                    )
                                    + np.cos(self.xtheta[jj])
                                    * (
                                        xr
                                        + self.pad_radius
                                        - self.journal_radius
                                        - self.radial_clearance
                                    )
                                )
                            ) / self.radial_clearance

                            h_e = (
                                self.pad_radius
                                - self.journal_radius
                                - (
                                    np.sin(teta_e)
                                    * (
                                        yr
                                        + alpha * (self.pad_radius + self.pad_thickness)
                                    )
                                    + np.cos(teta_e)
                                    * (
                                        xr
                                        + self.pad_radius
                                        - self.journal_radius
                                        - self.radial_clearance
                                    )
                                )
                            ) / self.radial_clearance

                            h_w = (
                                self.pad_radius
                                - self.journal_radius
                                - (
                                    np.sin(teta_w)
                                    * (
                                        yr
                                        + alpha * (self.pad_radius + self.pad_thickness)
                                    )
                                    + np.cos(teta_w)
                                    * (
                                        xr
                                        + self.pad_radius
                                        - self.journal_radius
                                        - self.radial_clearance
                                    )
                                )
                            ) / self.radial_clearance

                            h_n = h_p
                            h_s = h_p

                            h_pt = -(1 / (self.radial_clearance * self.speed)) * (
                                np.cos(self.xtheta[jj]) * xrpt
                                + np.sin(self.xtheta[jj]) * yrpt
                                + np.sin(self.xtheta[jj])
                                * (self.pad_radius + self.pad_thickness)
                                * alphapt
                            )

                            self.h[ii, jj] = h_p

                            # viscosity at faces
                            if jj == 0 and ii == 0:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = mi[ii, jj]
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = mi[ii, jj]

                            if jj == 0 and 0 < ii < self.nz - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = mi[ii, jj]
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if jj == 0 and ii == self.nz - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = mi[ii, jj]
                                mi_n = mi[ii, jj]
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if ii == 0 and 0 < jj < self.nx - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = mi[ii, jj]

                            if 0 < jj < self.nx - 1 and 0 < ii < self.nz - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if ii == self.nz - 1 and 0 < jj < self.nx - 1:
                                mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = mi[ii, jj]
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if ii == 0 and jj == self.nx - 1:
                                mi_e = mi[ii, jj]
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = mi[ii, jj]

                            if jj == self.nx - 1 and 0 < ii < self.nz - 1:
                                mi_e = mi[ii, jj]
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            if jj == self.nx - 1 and ii == self.nz - 1:
                                mi_e = mi[ii, jj]
                                mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                                mi_n = mi[ii, jj]
                                mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                            c_e = (
                                (1 / (self.pad_arc**2))
                                * (h_e**3 / (12 * mi_e))
                                * (self.dz / self.dx)
                            )
                            c_w = (
                                (1 / (self.pad_arc**2))
                                * (h_w**3 / (12 * mi_w))
                                * (self.dz / self.dx)
                            )
                            c_n = (
                                (self.pad_radius / self.pad_axial_length) ** 2
                                * (self.dx / self.dz)
                                * (h_n**3 / (12 * mi_n))
                            )
                            c_s = (
                                (self.pad_radius / self.pad_axial_length) ** 2
                                * (self.dx / self.dz)
                                * (h_s**3 / (12 * mi_s))
                            )
                            c_p = -(c_e + c_w + c_n + c_s)
                            b_val = (
                                self.journal_radius
                                / (2 * self.pad_radius * self.pad_arc)
                            ) * self.dz * (h_e - h_w) + h_pt * self.dx * self.dz
                            b_vec[k_idx] = b_val

                            # fill mat_coef according to mesh location
                            if ii == 0 and jj == 0:
                                mat_coef[k_idx, k_idx] = c_p - c_s - c_w
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx + self.nx] = c_n

                            if ii == 0 and 0 < jj < self.nx - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_s
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx + self.nx] = c_n

                            if ii == 0 and jj == self.nx - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_e - c_s
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx + self.nx] = c_n

                            if jj == 0 and 0 < ii < self.nz - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_w
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx - self.nx] = c_s
                                mat_coef[k_idx, k_idx + self.nx] = c_n

                            if 0 < ii < self.nz - 1 and 0 < jj < self.nx - 1:
                                mat_coef[k_idx, k_idx] = c_p
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx - self.nx] = c_s
                                mat_coef[k_idx, k_idx + self.nx] = c_n
                                mat_coef[k_idx, k_idx + 1] = c_e

                            if jj == self.nx - 1 and 0 < ii < self.nz - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_e
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx - self.nx] = c_s
                                mat_coef[k_idx, k_idx + self.nx] = c_n

                            if jj == 0 and ii == self.nz - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_n - c_w
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx - self.nx] = c_s

                            if ii == self.nz - 1 and 0 < jj < self.nx - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_n
                                mat_coef[k_idx, k_idx + 1] = c_e
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx - self.nx] = c_s

                            if ii == self.nz - 1 and jj == self.nx - 1:
                                mat_coef[k_idx, k_idx] = c_p - c_e - c_n
                                mat_coef[k_idx, k_idx - 1] = c_w
                                mat_coef[k_idx, k_idx - self.nx] = c_s

                            k_idx += 1

                    # Pressure field solution
                    mat_coef = self._check_diagonal(mat_coef)
                    p_vec = np.linalg.solve(mat_coef, b_vec)

                    cont = 0
                    for i_lin in np.arange(self.nz):
                        for j_col in np.arange(self.nx):
                            self.pressure[i_lin, j_col] = p_vec[cont]
                            cont += 1
                            if self.pressure[i_lin, j_col] < 0:
                                self.pressure[i_lin, j_col] = 0

                    # Energy equation & pressure gradients
                    n_k = self.nx * self.nz
                    mat_coef_t = np.zeros((n_k, n_k))
                    b_t = np.zeros(n_k)
                    test_diag = np.zeros(n_k)

                    k_t_idx = 0
                    for ii in range(self.nz):
                        for jj in range(self.nx):
                            if jj == 0 and ii == 0:
                                self.dp_dx[ii, jj] = (
                                    self.pressure[ii, jj + 1] - self.pressure[ii, jj]
                                ) / self.dx
                                self.dp_dz[ii, jj] = (
                                    self.pressure[ii + 1, jj] - self.pressure[ii, jj]
                                ) / self.dz

                            if jj == 0 and 0 < ii < self.nz - 1:
                                self.dp_dx[ii, jj] = (
                                    self.pressure[ii, jj + 1] - self.pressure[ii, jj]
                                ) / self.dx
                                self.dp_dz[ii, jj] = (
                                    self.pressure[ii + 1, jj] - self.pressure[ii, jj]
                                ) / self.dz

                            if jj == 0 and ii == self.nz - 1:
                                self.dp_dx[ii, jj] = (
                                    self.pressure[ii, jj + 1] - self.pressure[ii, jj]
                                ) / self.dx
                                self.dp_dz[ii, jj] = (
                                    0 - self.pressure[ii, jj]
                                ) / self.dz

                            if ii == 0 and 0 < jj < self.nx - 1:
                                self.dp_dx[ii, jj] = (
                                    self.pressure[ii, jj + 1] - self.pressure[ii, jj]
                                ) / self.dx
                                self.dp_dz[ii, jj] = (
                                    self.pressure[ii + 1, jj] - self.pressure[ii, jj]
                                ) / self.dz

                            if 0 < jj < self.nx - 1 and 0 < ii < self.nz - 1:
                                self.dp_dx[ii, jj] = (
                                    self.pressure[ii, jj + 1] - self.pressure[ii, jj]
                                ) / self.dx
                                self.dp_dz[ii, jj] = (
                                    self.pressure[ii + 1, jj] - self.pressure[ii, jj]
                                ) / self.dz

                            if ii == self.nz - 1 and 0 < jj < self.nx - 1:
                                self.dp_dx[ii, jj] = (
                                    self.pressure[ii, jj + 1] - self.pressure[ii, jj]
                                ) / self.dx
                                self.dp_dz[ii, jj] = (
                                    0 - self.pressure[ii, jj]
                                ) / self.dz

                            if ii == 0 and jj == self.nx - 1:
                                self.dp_dx[ii, jj] = (
                                    0 - self.pressure[ii, jj]
                                ) / self.dx
                                self.dp_dz[ii, jj] = (
                                    self.pressure[ii + 1, jj] - self.pressure[ii, jj]
                                ) / self.dz

                            if jj == self.nx - 1 and 0 < ii < self.nz - 1:
                                self.dp_dx[ii, jj] = (
                                    0 - self.pressure[ii, jj]
                                ) / self.dx
                                self.dp_dz[ii, jj] = (
                                    self.pressure[ii + 1, jj] - self.pressure[ii, jj]
                                ) / self.dz

                            if jj == self.nx - 1 and ii == self.nz - 1:
                                self.dp_dx[ii, jj] = (
                                    0 - self.pressure[ii, jj]
                                ) / self.dx
                                self.dp_dz[ii, jj] = (
                                    0 - self.pressure[ii, jj]
                                ) / self.dz

                            # Turbulence modeling (Eddy viscosity)
                            h_p_loc = self.h[ii, jj]
                            mi_p = mi[ii, jj]

                            self.reynolds_field[ii, jj, n_p] = (
                                self.rho
                                * self.speed
                                * self.journal_radius
                                * (h_p_loc / self.pad_axial_length)
                                * self.radial_clearance
                                / (self.mu_0 * mi_p)
                            )

                            if self.reynolds_field[ii, jj, n_p] <= 500:
                                delta_turb = 0
                            elif 400 < self.reynolds_field[ii, jj, n_p] <= 1000:
                                delta_turb = 1 - (
                                    (1000 - self.reynolds_field[ii, jj, n_p]) / 500
                                ) ** (1 / 8)
                            else:
                                delta_turb = 1

                            du_dx = (
                                (h_p_loc / self.mu_turb[ii, jj, n_p])
                                * self.dp_dx[ii, jj]
                            ) - (self.speed / h_p_loc)
                            dw_dx = (h_p_loc / self.mu_turb[ii, jj, n_p]) * self.dp_dz[
                                ii, jj
                            ]

                            tau = self.mu_turb[ii, jj, n_p] * np.sqrt(
                                du_dx**2 + dw_dx**2
                            )
                            y_wall = (
                                (h_p_loc * self.radial_clearance * 2)
                                / (self.mu_0 * self.mu_turb[ii, jj, n_p] / self.rho)
                            ) * ((abs(tau) / self.rho) ** 0.5)

                            emv = 0.4 * (y_wall - (10.7 * np.tanh(y_wall / 10.7)))
                            self.mu_turb[ii, jj, n_p] = mi_p * (1 + (delta_turb * emv))
                            mi_t = self.mu_turb[ii, jj, n_p]

                            # Coefficients for the energy equation
                            aux_up = 1
                            if self.xz[ii] < 0:
                                aux_up = 0

                            a_e = -(self.kt * h_p_loc * self.dz) / (
                                self.rho
                                * self.cp
                                * self.speed
                                * ((self.pad_arc * self.pad_radius) ** 2)
                                * self.dx
                            )

                            a_w = (
                                ((h_p_loc**3) * self.dp_dx[ii, jj] * self.dz)
                                / (12 * mi_t * (self.pad_arc**2))
                                - (
                                    (self.journal_radius * h_p_loc * self.dz)
                                    / (2 * self.pad_radius * self.pad_arc)
                                )
                                - (self.kt * h_p_loc * self.dz)
                                / (
                                    self.rho
                                    * self.cp
                                    * self.speed
                                    * ((self.pad_arc * self.pad_radius) ** 2)
                                    * self.dx
                                )
                            )

                            a_n_1 = (aux_up - 1) * (
                                (
                                    (self.pad_radius**2)
                                    * (h_p_loc**3)
                                    * self.dp_dz[ii, jj]
                                    * self.dx
                                )
                                / (12 * (self.pad_axial_length**2) * mi_t)
                            )
                            a_s_1 = (aux_up) * (
                                (
                                    (self.pad_radius**2)
                                    * (h_p_loc**3)
                                    * self.dp_dz[ii, jj]
                                    * self.dx
                                )
                                / (12 * (self.pad_axial_length**2) * mi_t)
                            )

                            a_n_2 = -(self.kt * h_p_loc * self.dx) / (
                                self.rho
                                * self.cp
                                * self.speed
                                * (self.pad_axial_length**2)
                                * self.dz
                            )

                            a_s_2 = -(self.kt * h_p_loc * self.dx) / (
                                self.rho
                                * self.cp
                                * self.speed
                                * (self.pad_axial_length**2)
                                * self.dz
                            )

                            a_n = a_n_1 + a_n_2
                            a_s = a_s_1 + a_s_2
                            a_p_coef = -(a_e + a_w + a_n + a_s)

                            aux_b_t = (self.speed * self.mu_0) / (
                                self.rho
                                * self.cp
                                * self.oil_supply_temperature
                                * self.radial_clearance
                            )

                            b_t_g = (
                                self.mu_0
                                * self.speed
                                * (self.journal_radius**2)
                                * self.dx
                                * self.dz
                                * self.pressure[ii, jj]
                                * h_pt
                            ) / (
                                self.rho
                                * self.cp
                                * self.reference_temperature
                                * (self.radial_clearance**2)
                            )

                            b_t_h = (
                                self.speed
                                * self.mu_0
                                * (h_pt**2)
                                * 4
                                * mi_t
                                * self.dx
                                * self.dz
                            ) / (
                                self.rho
                                * self.cp
                                * self.reference_temperature
                                * 3
                                * h_p_loc
                            )

                            b_t_i = (
                                aux_b_t
                                * (
                                    1
                                    * mi_t
                                    * (self.journal_radius**2)
                                    * self.dx
                                    * self.dz
                                )
                                / (h_p_loc * self.radial_clearance)
                            )

                            b_t_j = (
                                aux_b_t
                                * (
                                    (self.pad_radius**2)
                                    * (h_p_loc**3)
                                    * (self.dp_dx[ii, jj] ** 2)
                                    * self.dx
                                    * self.dz
                                )
                                / (
                                    12
                                    * self.radial_clearance
                                    * (self.pad_arc**2)
                                    * mi_t
                                )
                            )

                            b_t_k = (
                                aux_b_t
                                * (
                                    (self.pad_radius**4)
                                    * (h_p_loc**3)
                                    * (self.dp_dz[ii, jj] ** 2)
                                    * self.dx
                                    * self.dz
                                )
                                / (
                                    12
                                    * self.radial_clearance
                                    * (self.pad_axial_length**2)
                                    * mi_t
                                )
                            )

                            b_t_val = b_t_g + b_t_h + b_t_i + b_t_j + b_t_k
                            b_t[k_t_idx] = b_t_val

                            # fill mat_coef_t according to mesh location
                            if ii == 0 and jj == 0:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_s - a_w
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n
                                b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (
                                    self.oil_supply_temperature
                                    / self.reference_temperature
                                )
                                test_diag[k_t_idx] = mat_coef_t[k_t_idx, k_t_idx] - (
                                    mat_coef_t[k_t_idx, k_t_idx + 1]
                                    + mat_coef_t[k_t_idx, k_t_idx + self.nx]
                                )

                            if ii == 0 and 0 < jj < self.nx - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_s
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n
                                test_diag[k_t_idx] = mat_coef_t[k_t_idx, k_t_idx] - (
                                    mat_coef_t[k_t_idx, k_t_idx + 1]
                                    + mat_coef_t[k_t_idx, k_t_idx - 1]
                                    + mat_coef_t[k_t_idx, k_t_idx + self.nx]
                                )

                            if ii == 0 and jj == self.nx - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e + a_s
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n
                                test_diag[k_t_idx] = mat_coef_t[k_t_idx, k_t_idx] - (
                                    mat_coef_t[k_t_idx, k_t_idx - 1]
                                    + mat_coef_t[k_t_idx, k_t_idx + self.nx]
                                )

                            if jj == 0 and 0 < ii < self.nz - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef - a_w
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
                                mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n
                                b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (
                                    self.oil_supply_temperature
                                    / self.reference_temperature
                                )
                                test_diag[k_t_idx] = mat_coef_t[k_t_idx, k_t_idx] - (
                                    mat_coef_t[k_t_idx, k_t_idx + 1]
                                    + mat_coef_t[k_t_idx, k_t_idx - 1]
                                    + mat_coef_t[k_t_idx, k_t_idx + self.nx]
                                )

                            if 0 < ii < self.nz - 1 and 0 < jj < self.nx - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
                                mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                test_diag[k_t_idx] = mat_coef_t[k_t_idx, k_t_idx] - (
                                    mat_coef_t[k_t_idx, k_t_idx + 1]
                                    + mat_coef_t[k_t_idx, k_t_idx - 1]
                                    + mat_coef_t[k_t_idx, k_t_idx - self.nx]
                                    + mat_coef_t[k_t_idx, k_t_idx + self.nx]
                                )

                            if jj == self.nx - 1 and 0 < ii < self.nz - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
                                mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n
                                test_diag[k_t_idx] = mat_coef_t[k_t_idx, k_t_idx] - (
                                    mat_coef_t[k_t_idx, k_t_idx - 1]
                                    + mat_coef_t[k_t_idx, k_t_idx - self.nx]
                                    + mat_coef_t[k_t_idx, k_t_idx + self.nx]
                                )

                            if jj == 0 and ii == self.nz - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_n - a_w
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
                                b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (
                                    self.oil_supply_temperature
                                    / self.reference_temperature
                                )
                                test_diag[k_t_idx] = mat_coef_t[k_t_idx, k_t_idx] - (
                                    mat_coef_t[k_t_idx, k_t_idx + 1]
                                    + mat_coef_t[k_t_idx, k_t_idx - self.nx]
                                )

                            if ii == self.nz - 1 and 0 < jj < self.nx - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_n
                                mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
                                test_diag[k_t_idx] = mat_coef_t[k_t_idx, k_t_idx] - (
                                    mat_coef_t[k_t_idx, k_t_idx + 1]
                                    + mat_coef_t[k_t_idx, k_t_idx - 1]
                                    + mat_coef_t[k_t_idx, k_t_idx - self.nx]
                                )

                            if ii == self.nz - 1 and jj == self.nx - 1:
                                mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e + a_n
                                mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
                                mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
                                test_diag[k_t_idx] = mat_coef_t[k_t_idx, k_t_idx] - (
                                    mat_coef_t[k_t_idx, k_t_idx - 1]
                                    + mat_coef_t[k_t_idx, k_t_idx - self.nx]
                                )

                            k_t_idx += 1

                    # Solution of temperature field
                    mat_coef_t = self._check_diagonal(mat_coef_t)
                    t_vec = np.linalg.solve(mat_coef_t, b_t)
                    cont = 0
                    for i_lin in np.arange(self.nz):
                        for j_col in np.arange(self.nx):
                            temperature_referance[i_lin, j_col] = (
                                self.reference_temperature * t_vec[cont]
                            )
                            cont += 1

                    # Hydrodynamic forces
                    p_dimen = (
                        self.pressure
                        * (self.mu_0 * self.speed * self.pad_radius**2)
                        / (self.radial_clearance**2)
                    )

                    aux_f1 = np.zeros((self.nz, self.nx))
                    aux_f2 = np.zeros((self.nz, self.nx))
                    for ni in np.arange(self.nz):
                        aux_f1[ni, :] = np.cos(self.xtheta)
                        aux_f2[ni, :] = np.sin(self.xtheta)

                    y_teta_f1 = self.pressure * aux_f1
                    f1_teta = np.trapezoid(y_teta_f1, self.xtheta)
                    self.force_new[n_p] = -np.trapezoid(f1_teta, self.xz)

                    y_teta_f2 = self.pressure * aux_f2
                    f2_teta = np.trapezoid(y_teta_f2, self.xtheta)
                    self.force_2_new[n_p] = -np.trapezoid(f2_teta, self.xz)

                    self.moment_j_new[n_p] = self.force_2_new[n_p] * (
                        self.pad_radius + self.pad_thickness
                    )

                    # Dynamic loads
                    del_force_x[n_p] = -(self.force_new[n_p] - self.force_1[n_p]) # here
                    del_force_y[n_p] = -(self.force_2_new[n_p] - self.force_2[n_p])
                    del_moment_j[n_p] = -(self.moment_j_new[n_p] - self.moment_j[n_p])

                    # X-axis perturbation
                    if a_p == 0:
                        k_xx[n_p] = (
                            del_force_x[n_p]
                            / self.space_perturbation
                            * self.dimensionless_force[n_p]
                        )
                        k_yx[n_p] = (
                            del_force_y[n_p]
                            / self.space_perturbation
                            * self.dimensionless_force[n_p]
                        )

                        if n_p == 0:
                            print(f"[DIAG] force_1[{n_p}]={self.force_1[n_p]:.2e}")
                            print(f"[DIAG] force_new[{n_p}]={self.force_new[n_p]:.2e}")
                            print(f"[DIAG] del_force_x[{n_p}]={del_force_x[n_p]:.2e}")
                            print(f"[DIAG] k_xx[{n_p}]={k_xx[n_p]:.2e}")

                        self.K[n_p, 0, 0] = k_xx[n_p]
                        self.K[n_p, 1, 0] = k_yx[n_p]

                    # Angular (pad) perturbation
                    elif a_p == 1:
                        k_yy[n_p] = (
                            del_force_y[n_p]
                            / self.space_perturbation
                            * self.dimensionless_force[n_p]
                        )
                        k_xy[n_p] = (
                            del_force_x[n_p]
                            / self.space_perturbation
                            * self.dimensionless_force[n_p]
                        )

                        if n_p == 0:
                            print(f"[DIAG Y] force_2[{n_p}]={self.force_2[n_p]:.2e}")
                            print(f"[DIAG Y] force_2_new[{n_p}]={self.force_2_new[n_p]:.2e}")
                            print(f"[DIAG Y] del_force_y[{n_p}]={del_force_y[n_p]:.2e}")
                            print(f"[DIAG Y] k_yy[{n_p}]={k_yy[n_p]:.2e}")

                        self.K[n_p, 1, 1] = k_yy[n_p]
                        self.K[n_p, 0, 1] = k_xy[n_p]

                        k_tt[n_p] = k_yy[n_p] * (
                            (self.pad_radius + self.pad_thickness) ** 2
                        )
                        k_tx[n_p] = -k_yx[n_p] * (self.pad_radius + self.pad_thickness)
                        k_xt[n_p] = -k_xy[n_p] * (self.pad_radius + self.pad_thickness)
                        k_ty[n_p] = -k_yy[n_p] * (self.pad_radius + self.pad_thickness)
                        k_yt[n_p] = -k_yy[n_p] * (self.pad_radius + self.pad_thickness)
                        self.K[n_p, 2, 2] = k_tt[n_p]
                        self.K[n_p, 0, 2] = k_xt[n_p]
                        self.K[n_p, 2, 0] = k_tx[n_p]
                        self.K[n_p, 1, 2] = k_yt[n_p]
                        self.K[n_p, 2, 1] = k_ty[n_p]

                    # X-axis speed perturbation
                    elif a_p == 2:
                        c_xx[n_p] = (
                            del_force_x[n_p]
                            / self.speed_perturbation
                            * self.dimensionless_force[n_p]
                        )
                        c_yx[n_p] = (
                            del_force_y[n_p]
                            / self.speed_perturbation
                            * self.dimensionless_force[n_p]
                        )
                        self.C[n_p, 0, 0] = c_xx[n_p]
                        self.C[n_p, 1, 0] = c_yx[n_p]

                    # Angular speed perturbation
                    elif a_p == 3:
                        c_yy[n_p] = (
                            del_force_y[n_p]
                            / self.speed_perturbation
                            * self.dimensionless_force[n_p]
                        )
                        c_xy[n_p] = (
                            del_force_x[n_p]
                            / self.speed_perturbation
                            * self.dimensionless_force[n_p]
                        )
                        self.C[n_p, 1, 1] = c_yy[n_p]
                        self.C[n_p, 0, 1] = c_xy[n_p]

                        c_tt[n_p] = c_yy[n_p] * (
                            (self.pad_radius + self.pad_thickness) ** 2
                        )
                        c_tx[n_p] = -c_yx[n_p] * (self.pad_radius + self.pad_thickness)
                        c_xt[n_p] = -c_xy[n_p] * (self.pad_radius + self.pad_thickness)
                        c_ty[n_p] = -c_yy[n_p] * (self.pad_radius + self.pad_thickness)
                        c_yt[n_p] = -c_yy[n_p] * (self.pad_radius + self.pad_thickness)
                        self.C[n_p, 2, 2] = c_tt[n_p]
                        self.C[n_p, 0, 2] = c_xt[n_p]
                        self.C[n_p, 2, 0] = c_tx[n_p]
                        self.C[n_p, 1, 2] = c_yt[n_p]
                        self.C[n_p, 2, 1] = c_ty[n_p]

                        # Coefficient matrix reduction (per-pad contribution)
                        self.Sjpt[n_p] = self.K[n_p] + self.C[n_p] * self.speed * 1j
                        self.Tj[n_p] = np.array(
                            [
                                [
                                    np.cos(psi_pad[n_p] + self.pivot_angle[n_p]),
                                    np.sin(psi_pad[n_p] + self.pivot_angle[n_p]),
                                    0,
                                ],
                                [
                                    -np.sin(psi_pad[n_p] + self.pivot_angle[n_p]),
                                    np.cos(psi_pad[n_p] + self.pivot_angle[n_p]),
                                    0,
                                ],
                                [0, 0, 1],
                            ]
                        )
                        self.Sjipt[n_p] = self.Tj[n_p].T @ self.Sjpt[n_p] @ self.Tj[n_p]

                        # DIAGNOSTIC: Check transformation for first pad
                        if n_p == 0 and a_p == 3:  # Last perturbation, first pad
                            print(f"\n[DIAG TRANS] ========== COORDINATE TRANSFORMATION (Pad {n_p}) ==========")
                            print(f"  Pivot angle = {np.degrees(self.pivot_angle[n_p]):.2f}°")
                            print(f"  Pad angle (psi) = {np.degrees(psi_pad[n_p]):.6f}°")
                            print(f"  Total angle (psi + pivot) = {np.degrees(psi_pad[n_p] + self.pivot_angle[n_p]):.2f}°")
                            print(f"  K[pad] (before transform):")
                            print(f"    k_xx={self.K[n_p, 0, 0]:.2e}, k_xy={self.K[n_p, 0, 1]:.2e}")
                            print(f"    k_yx={self.K[n_p, 1, 0]:.2e}, k_yy={self.K[n_p, 1, 1]:.2e}")
                            print(f"  Sjipt[pad] (after transform):")
                            print(f"    Sjipt[0,0]={self.Sjipt[n_p, 0, 0]:.2e}, Sjipt[0,1]={self.Sjipt[n_p, 0, 1]:.2e}")
                            print(f"    Sjipt[1,0]={self.Sjipt[n_p, 1, 0]:.2e}, Sjipt[1,1]={self.Sjipt[n_p, 1, 1]:.2e}")
                            print(f"[DIAG TRANS] ===================================================\n")

                        # Add 2x2 block in Aj and gyro term in Bj
                        self.Aj += np.array(
                            [
                                [self.Sjipt[n_p, 0, 0], self.Sjipt[n_p, 0, 1]],
                                [self.Sjipt[n_p, 1, 0], self.Sjipt[n_p, 1, 1]],
                            ]
                        )
                        self.Bj[n_p, n_p] = self.Sjipt[n_p, 2, 2]

        self.Hj = np.array(
            [
                [self.Sjipt[m, 0, 2] for m in range(self.n_pad)],
                [self.Sjipt[m, 1, 2] for m in range(self.n_pad)],
            ],
            dtype="complex",
        )
        self.Vj = np.array(
            [[self.Sjipt[m, 2, 0], self.Sjipt[m, 2, 1]] for m in range(self.n_pad)],
            dtype="complex",
        )

        # Final reduction: Sw = Aj - Hj * Bj^{-1} * Vj
        Bj_checked = self._check_diagonal(self.Bj)
        self.Sw = self.Aj - (self.Hj @ np.linalg.inv(Bj_checked) @ self.Vj)

        k_r = np.real(self.Sw)
        c_r = np.imag(self.Sw) / self.speed
        self.kxx, self.kyy, self.kxy, self.kyx = (
            k_r[0, 0],
            k_r[1, 1],
            2 * k_r[0, 1],
            2 * k_r[1, 0],
        )
        self.cxx, self.cyy, self.cxy, self.cyx = (
            c_r[0, 0],
            c_r[1, 1],
            c_r[0, 1],
            c_r[1, 0],
        )

    def solve_fields(self):
        """Solve the thermo-hydrodynamic equations to determine equilibrium position and field distributions.

        This method performs the complete thermo-hydrodynamic analysis for tilting pad bearings,
        including equilibrium position calculation and field solution for pressure and temperature.

        The method supports two equilibrium calculation types:
        - 'match_eccentricity': Imposes eccentricity and optimizes only pad angles
        - 'determine_eccentricity': Determines complete equilibrium position including eccentricity

        For each pad, the method solves:
        1. Reynolds equation for pressure field using finite difference method
        2. Energy equation for temperature field with turbulent viscosity modeling
        3. Iterative convergence between pressure and temperature fields

        Parameters
        ----------
        None
            This method uses the bearing parameters defined during initialization.

        Returns
        -------
        tuple
            A tuple containing the following results:
            - max_p : float
                Maximum pressure in Pa
            - med_p : float
                Mean pressure in Pa
            - max_t : float
                Maximum temperature in °C
            - med_t : float
                Mean temperature in °C
            - h_pivot : float
                Oil film thickness at pivot point in m
            - ecc : float
                Eccentricity ratio (dimensionless)

        Notes
        -----
        The method performs the following steps:

        1. Equilibrium Calculation:
        - For 'match_eccentricity': Optimizes pad rotation angles using fmin
        - For 'determine_eccentricity': Optimizes complete system (eccentricity, attitude angle, pad angles)

        2. Field Solution:
        - Iterative solution of Reynolds and energy equations for each pad
        - Temperature convergence tolerance: 0.1°C
        - Pressure field with non-negative constraint

        3. Results Processing:
        - Stores pressure and temperature fields for all pads
        - Calculates dimensional quantities
        - Determines pad with maximum pressure

        The method assumes steady-state operation and uses finite difference
        methods with upwind scheme for energy equation.

        See Also
        --------
        run : Execute complete analysis including dynamic coefficients
        coefficients : Calculate stiffness and damping coefficients
        """

        # Choose equilibrium calculation method
        if self.equilibrium_type == "match_eccentricity":
            ang_rot = np.zeros(self.n_pad)
            momen_rot = np.zeros(self.n_pad)
            self.alpha_max_chut = np.zeros(self.n_pad)
            self.alpha_min_chut = np.zeros(self.n_pad)

            for n_pad in range(self.n_pad):
                if self.xj is None or self.yj is None:
                    xx_alpha = (
                        self.eccentricity
                        * self.radial_clearance
                        * np.cos(self.attitude_angle)
                    )
                    yy_alpha = (
                        self.eccentricity
                        * self.radial_clearance
                        * np.sin(self.attitude_angle)
                    )
                else:
                    xx_alpha = self.xj
                    yy_alpha = self.yj
                    self.eccentricity = (
                        np.sqrt(self.xj**2 + self.yj**2) / self.radial_clearance
                    )
                    self.attitude_angle = np.arctan2(self.yj, self.xj)

                xryr_alpha = np.dot(
                    [
                        [
                            np.cos(self.pivot_angle[n_pad]),
                            np.sin(self.pivot_angle[n_pad]),
                        ],
                        [
                            -np.sin(self.pivot_angle[n_pad]),
                            np.cos(self.pivot_angle[n_pad]),
                        ],
                    ],
                    [[xx_alpha], [yy_alpha]],
                )

                self.alpha_max_chut[n_pad] = (
                    self.pad_radius
                    - self.journal_radius
                    - np.cos(self.theta_2)
                    * (
                        xryr_alpha[0, 0]
                        + self.pad_radius
                        - self.journal_radius
                        - self.radial_clearance
                    )
                ) / (np.sin(self.theta_2) * (self.pad_radius + self.pad_thickness)) - (
                    xryr_alpha[1, 0]
                ) / (self.pad_radius + self.pad_thickness)

                self.alpha_min_chut[n_pad] = (
                    self.pad_radius
                    - self.journal_radius
                    - np.cos(self.theta_1)
                    * (
                        xryr_alpha[0, 0]
                        + self.pad_radius
                        - self.journal_radius
                        - self.radial_clearance
                    )
                ) / (np.sin(self.theta_1) * (self.pad_radius + self.pad_thickness)) - (
                    xryr_alpha[1, 0]
                ) / (self.pad_radius + self.pad_thickness)

            self.x_0 = 0.4 * self.alpha_max_chut

            for con_np in range(self.n_pad):
                self.con_np = con_np
                idx = con_np
                self.score_dim = 100000

                x_opt = fmin(
                    self.get_equilibrium_position,
                    self.x_0[con_np],
                    xtol=1e-6,
                    ftol=1e-6,
                    maxiter=1000,
                    disp=False,
                )

                ang_rot[idx] = x_opt.item() if hasattr(x_opt, "item") else x_opt
                momen_rot[idx] = self.score_dim

            self.psi_pad = ang_rot
            self.force_x_dim = np.sum(self.force_x_dim)
            self.force_y_dim = np.sum(self.force_y_dim)

            self.ecc_list.append(self.eccentricity)
            self.attitude_angle_list.append(self.attitude_angle)
            self.psi_pad_list.append(ang_rot.copy())
            self.force_x_total_list.append(self.force_x_dim)
            self.force_y_total_list.append(self.force_y_dim)
            self.momen_rot_list.append(momen_rot.copy())

            self.xdin = np.zeros(self.n_pad + 2)
            self.xdin = [self.eccentricity, self.attitude_angle] + list(ang_rot)

        elif self.equilibrium_type == "determine_eccentricity":
            # Initial guess: [eccentricity, attitude_angle, pad_angles...]
            if self.initial_pads_angles is not None:
                x0 = [self.eccentricity, self.attitude_angle] + list(
                    self.initial_pads_angles
                )
            else:
                x0 = [0.3, np.deg2rad(270)] + [1e-4] * self.n_pad

            def _nm_callback(xk):
                print(f"[determine_ecc] callback x={xk}")
            # Optimize complete system
            result = fmin(
                self._equilibrium_objective,
                x0,
                xtol=1e-3,
                ftol=1e-3,
                maxiter=1000,
                disp=False,
                full_output=True,
                callback=_nm_callback,
            )
            # def _optimization_callback(xk):
            #     """Monitor optimization progress"""
            #     if not hasattr(self, "iteration_count"):
            #         self.iteration_count = 0
            #     self.iteration_count += 1
                
            #     # Calculate current score
            #     score = self._equilibrium_objective(xk)
                
            #     # Extract state variables
            #     ecc, att = xk[0], xk[1]
            #     pads = xk[2:]
                
            #     # Format output
            #     pads_str = "[" + ", ".join(f"{np.degrees(p):.2f}" for p in pads) + "]°"
                
            #     # Print every iteration
            #     print(f"[Powell] iter={self.iteration_count:4d} "
            #         f"ecc={ecc:.4f} att={np.degrees(att):6.2f}° "
            #         f"pads={pads_str} score={score:.4e}")
                
            #     # Record in history
            #     self.record_optimization_residual(score)

            # Optimize complete system
            # result = minimize(
            #     self._equilibrium_objective,
            #     x0,
            #     # method="Powell",
            #     callback=_optimization_callback,
            #     options={
            #         "xtol": 1e-6,
            #         "ftol": 1e-6,
            #         "maxiter": 5000,
            #         "disp": True,
            #     }
            # )
            # result = minimize(
            #     self._equilibrium_objective,
            #     x0,
            #     method='L-BFGS-B',  # Método com gradiente numérico + bounds
            #     bounds=[
            #         (1e-3, 1),        # eccentricity: longe de 0 e 1
            #         (0, 2*np.pi),      # attitude_angle
            #         (-1e-5, 1e-5),     # pad angles (pequenos)
            #         (-1e-5, 1e-5),
            #         (-1e-5, 1e-5),
            #         (-1e-5, 1e-5),
            #         (-1e-5, 1e-5),
            #     ],
            #     callback=_optimization_callback,
            #     options={
            #         'ftol': 1e-8,
            #         'maxiter': 1000,
            #         'disp': True,
            #     }
            # )

            # Extract only x_opt from tuple
            if isinstance(result, tuple):
                x_opt = result[0]
            else:
                x_opt = result

            # Update state variables with optimized values
            self.eccentricity, self.attitude_angle, self.psi_pad = (
                x_opt[0],
                x_opt[1],
                x_opt[2:],
            )

            # Update state vector
            self.xdin = x_opt.copy()

            # Update state variables with optimized values
            self.eccentricity, self.attitude_angle, self.psi_pad = (
                x_opt[0],
                x_opt[1],
                x_opt[2:],
            )

            # Update state vector
            self.xdin = x_opt.copy()

            self.ecc_list.append(self.eccentricity)
            self.attitude_angle_list.append(self.attitude_angle)
            self.psi_pad_list.append(self.psi_pad.copy())
            self.force_x_total_list.append(np.sum(self.force_x_dim))
            self.force_y_total_list.append(np.sum(self.force_y_dim))
            self.momen_rot_list.append(None)

            print(
                "[determine_ecc] optimization done: "
                f"ecc={self.eccentricity:.4f}, "
                f"att={np.degrees(self.attitude_angle):.2f}deg, "
                f"final_score={self.optimization_history.get(self._current_freq_index, [])[-1]:.3e}"
            )

        # Thermo-hydrodynamic field solution
        psi_pad = np.zeros(self.n_pad)
        for k_pad in range(self.n_pad):
            psi_pad[k_pad] = self.xdin[k_pad + 2]

        n_k = self.nx * self.nz
        temperature_tolerance = 0.1

        for n_p in range(self.n_pad):
            temperature_referance = self.temperature_init[:, :, n_p]
            temperature_iteration = 1.1 * temperature_referance
            cont_temp = 0

            while (
                abs((temperature_referance - temperature_iteration).max())
                >= temperature_tolerance
            ):
                cont_temp += 1
                temperature_iteration = np.array(temperature_referance)

                mi_i = self.a_a * np.exp(self.b_b * temperature_iteration)
                mi = mi_i / self.mu_0

                k_idx = 0
                mat_coef = np.zeros((n_k, n_k))
                b_vec = np.zeros(n_k)

                xryr, xryrpt, xr, yr, xrpt, yrpt = self._transform_coordinates(n_p)

                alpha = psi_pad[n_p]
                alphapt = 0

                for ii in range(self.nz):
                    for jj in range(self.nx):
                        teta_e = self.xtheta[jj] + 0.5 * self.dtheta
                        teta_w = self.xtheta[jj] - 0.5 * self.dtheta

                        h_p = (
                            self.pad_radius
                            - self.journal_radius
                            - (
                                np.sin(self.xtheta[jj])
                                * (yr + alpha * (self.pad_radius + self.pad_thickness))
                                + np.cos(self.xtheta[jj])
                                * (
                                    xr
                                    + self.pad_radius
                                    - self.journal_radius
                                    - self.radial_clearance
                                )
                            )
                        ) / self.radial_clearance

                        h_e = (
                            self.pad_radius
                            - self.journal_radius
                            - (
                                np.sin(teta_e)
                                * (yr + alpha * (self.pad_radius + self.pad_thickness))
                                + np.cos(teta_e)
                                * (
                                    xr
                                    + self.pad_radius
                                    - self.journal_radius
                                    - self.radial_clearance
                                )
                            )
                        ) / self.radial_clearance

                        h_w = (
                            self.pad_radius
                            - self.journal_radius
                            - (
                                np.sin(teta_w)
                                * (yr + alpha * (self.pad_radius + self.pad_thickness))
                                + np.cos(teta_w)
                                * (
                                    xr
                                    + self.pad_radius
                                    - self.journal_radius
                                    - self.radial_clearance
                                )
                            )
                        ) / self.radial_clearance

                        h_n = h_p
                        h_s = h_p

                        h_pt = -(1 / (self.radial_clearance * self.speed)) * (
                            np.cos(self.xtheta[jj]) * xrpt
                            + np.sin(self.xtheta[jj]) * yrpt
                            + np.sin(self.xtheta[jj])
                            * (self.pad_radius + self.pad_thickness)
                            * alphapt
                        )

                        self.h[ii, jj] = h_p

                        if jj == 0 and ii == 0:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = mi[ii, jj]
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = mi[ii, jj]

                        if jj == 0 and 0 < ii < self.nz - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = mi[ii, jj]
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if jj == 0 and ii == self.nz - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = mi[ii, jj]
                            mi_n = mi[ii, jj]
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if ii == 0 and 0 < jj < self.nx - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = mi[ii, jj]

                        if 0 < jj < self.nx - 1 and 0 < ii < self.nz - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if ii == self.nz - 1 and 0 < jj < self.nx - 1:
                            mi_e = 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = mi[ii, jj]
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if ii == 0 and jj == self.nx - 1:
                            mi_e = mi[ii, jj]
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = mi[ii, jj]

                        if jj == self.nx - 1 and 0 < ii < self.nz - 1:
                            mi_e = mi[ii, jj]
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        if jj == self.nx - 1 and ii == self.nz - 1:
                            mi_e = mi[ii, jj]
                            mi_w = 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                            mi_n = mi[ii, jj]
                            mi_s = 0.5 * (mi[ii, jj] + mi[ii - 1, jj])

                        c_e = (
                            (1 / (self.pad_arc**2))
                            * (h_e**3 / (12 * mi_e))
                            * (self.dz / self.dx)
                        )
                        c_w = (
                            (1 / (self.pad_arc**2))
                            * (h_w**3 / (12 * mi_w))
                            * (self.dz / self.dx)
                        )
                        c_n = (
                            (self.pad_radius / self.pad_axial_length) ** 2
                            * (self.dx / self.dz)
                            * (h_n**3 / (12 * mi_n))
                        )
                        c_s = (
                            (self.pad_radius / self.pad_axial_length) ** 2
                            * (self.dx / self.dz)
                            * (h_s**3 / (12 * mi_s))
                        )
                        c_p = -(c_e + c_w + c_n + c_s)
                        b_val = (
                            self.journal_radius / (2 * self.pad_radius * self.pad_arc)
                        ) * self.dz * (h_e - h_w) + h_pt * self.dx * self.dz
                        b_vec[k_idx] = b_val

                        if ii == 0 and jj == 0:
                            mat_coef[k_idx, k_idx] = c_p - c_s - c_w
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx + self.nx] = c_n

                        if ii == 0 and 0 < jj < self.nx - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_s
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx + self.nx] = c_n

                        if ii == 0 and jj == self.nx - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_e - c_s
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx + self.nx] = c_n

                        if jj == 0 and 0 < ii < self.nz - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_w
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx - self.nx] = c_s
                            mat_coef[k_idx, k_idx + self.nx] = c_n

                        if 0 < ii < self.nz - 1 and 0 < jj < self.nx - 1:
                            mat_coef[k_idx, k_idx] = c_p
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx - self.nx] = c_s
                            mat_coef[k_idx, k_idx + self.nx] = c_n
                            mat_coef[k_idx, k_idx + 1] = c_e

                        if jj == self.nx - 1 and 0 < ii < self.nz - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_e
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx - self.nx] = c_s
                            mat_coef[k_idx, k_idx + self.nx] = c_n

                        if jj == 0 and ii == self.nz - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_n - c_w
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx - self.nx] = c_s

                        if ii == self.nz - 1 and 0 < jj < self.nx - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_n
                            mat_coef[k_idx, k_idx + 1] = c_e
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx - self.nx] = c_s

                        if ii == self.nz - 1 and jj == self.nx - 1:
                            mat_coef[k_idx, k_idx] = c_p - c_e - c_n
                            mat_coef[k_idx, k_idx - 1] = c_w
                            mat_coef[k_idx, k_idx - self.nx] = c_s

                        k_idx += 1

                mat_coef = self._check_diagonal(mat_coef)
                p_vec = np.linalg.solve(mat_coef, b_vec)

                cont_lin = 0
                for i_lin in np.arange(self.nz):
                    for j_col in np.arange(self.nx):
                        self.pressure[i_lin, j_col] = p_vec[cont_lin]
                        cont_lin += 1
                        if self.pressure[i_lin, j_col] < 0:
                            self.pressure[i_lin, j_col] = 0

            if np.max(self.pressure) > np.max(self.pressure_prev):
                self.pad_in = n_p
                self.pressure_prev = self.pressure.copy()

            self.pressure_nd[:, :, n_p] = self.pressure
            self.h_0[:, :, n_p] = self.h

            self.pressure_dim[:, :, n_p] = (
                self.pressure_nd[:, :, n_p]
                * (self.mu_0 * self.speed * self.pad_radius**2)
                / (self.radial_clearance**2)
            )
            self.h_dim[:, :, n_p] = self.h * self.radial_clearance
            self.h_pivot[n_p] = (
                self.radial_clearance
                * (
                    self.pad_radius
                    - self.journal_radius
                    - (
                        np.sin(0)
                        * (yr + alpha * (self.pad_radius + self.pad_thickness))
                        + np.cos(0)
                        * (
                            xr
                            + self.pad_radius
                            - self.journal_radius
                            - self.radial_clearance
                        )
                    )
                )
                / self.radial_clearance
            )

        max_p = (
            self.pressure_nd[:, :, self.pad_in].max()
            * (self.mu_0 * self.speed * self.pad_radius**2)
            / (self.radial_clearance**2)
        )
        med_p = (
            self.pressure_nd[:, :, self.pad_in].mean()
            * (self.mu_0 * self.speed * self.pad_radius**2)
            / (self.radial_clearance**2)
        )
        max_t = self.temperature_init[:, :, self.pad_in].max()
        med_t = self.temperature_init[:, :, self.pad_in].mean()
        h_pivot = self.h_pivot[self.pad_in]
        ecc = np.sqrt(xr**2 + yr**2) / self.radial_clearance

        self.maxP_list.append(max_p)
        self.maxT_list.append(max_t)
        self.minH_list.append(h_pivot)

        # Store complete fields for plotting
        self.pressure_fields.append(self.pressure_dim.copy())
        self.temperature_fields.append(self.temperature_init.copy())

        return max_p, med_p, max_t, med_t, h_pivot, ecc

    def _equilibrium_objective(self, x):
        """Objective function for complete equilibrium optimization.

        This method serves as the objective function for optimization algorithms
        to find the complete equilibrium position of a tilting pad bearing,
        including eccentricity, attitude angle, and pad rotation angles.

        Parameters
        ----------
        x : array_like
            State vector containing [eccentricity, attitude_angle, pad_angles...].
            The first two elements are the journal position parameters, followed
            by the rotation angles for each pad.

        Returns
        -------
        score : float
            The equilibrium residual (norm of force and moment residuals).
            Lower values indicate better equilibrium convergence.

        Notes
        -----
        The method performs the following steps:

        1. State Extraction: Extracts eccentricity, attitude angle, and pad angles
        2. Field Solution: For each pad, solves Reynolds and energy equations
        3. Force Calculation: Computes hydrodynamic forces and moments
        4. Residual Evaluation: Calculates force and moment equilibrium residuals

        The objective function evaluates the equilibrium condition by minimizing
        the sum of squared residuals for:
        - Force equilibrium in X and Y directions
        - Moment equilibrium for each pad

        Temperature convergence tolerance is set to 0.1°C for field calculations.

        See Also
        --------
        solve_fields : Main method for equilibrium calculation
        get_equilibrium_position : Single pad equilibrium optimization
        """

        t0 = time.time()

        # Increment iteration counter
        if not hasattr(self, "iteration_count"):
            self.iteration_count = 0
        self.iteration_count += 1

        # Extract state variables
        eccentricity = x[0]
        attitude_angle = x[1]

        psi_pad = np.zeros(self.n_pad)
        for kp in range(self.n_pad):
            psi_pad[kp] = x[kp + 2]  # Tilting angles of each pad

        n_k = self.nx * self.nz
        tol_T = 0.1  # Celsius degree

        # Reset force arrays
        self._reset_force_arrays()

        for n_p in range(self.n_pad):
            T_new = self.temperature_init[:, :, n_p]
            T_i = 1.1 * T_new
            cont_Temp = 0

            while abs((T_new - T_i).max()) >= tol_T:
                cont_Temp += 1
                T_i = np.array(T_new)

                # Calculate viscosity
                mi_i = self.a_a * np.exp(self.b_b * T_i)  # [Pa.s]
                MI = mi_i / self.mu_0  # Dimensionless viscosity field

                k = 0  # vectorization index
                Mat_coef = np.zeros((n_k, n_k))
                b = np.zeros(n_k)

                # Transformation of coordinates - inertial to pivot referential
                xryr, xryrpt, xr, yr, xrpt, yrpt = self._transform_coordinates(
                    n_p, eccentricity, attitude_angle
                )

                alpha = psi_pad[n_p]
                alphapt = 0

                for ii in range(self.nz):
                    for jj in range(self.nx):
                        teta_e = self.xtheta[jj] + 0.5 * self.dtheta
                        teta_w = self.xtheta[jj] - 0.5 * self.dtheta

                        h_p = (
                            self.pad_radius
                            - self.journal_radius
                            - (
                                np.sin(self.xtheta[jj])
                                * (yr + alpha * (self.pad_radius + self.pad_thickness))
                                + np.cos(self.xtheta[jj])
                                * (
                                    xr
                                    + self.pad_radius
                                    - self.journal_radius
                                    - self.radial_clearance
                                )
                            )
                        ) / self.radial_clearance

                        h_e = (
                            self.pad_radius
                            - self.journal_radius
                            - (
                                np.sin(teta_e)
                                * (yr + alpha * (self.pad_radius + self.pad_thickness))
                                + np.cos(teta_e)
                                * (
                                    xr
                                    + self.pad_radius
                                    - self.journal_radius
                                    - self.radial_clearance
                                )
                            )
                        ) / self.radial_clearance

                        h_w = (
                            self.pad_radius
                            - self.journal_radius
                            - (
                                np.sin(teta_w)
                                * (yr + alpha * (self.pad_radius + self.pad_thickness))
                                + np.cos(teta_w)
                                * (
                                    xr
                                    + self.pad_radius
                                    - self.journal_radius
                                    - self.radial_clearance
                                )
                            )
                        ) / self.radial_clearance

                        h_n = h_p
                        h_s = h_p

                        h_pt = -(1 / (self.radial_clearance * self.speed)) * (
                            np.cos(self.xtheta[jj]) * xrpt
                            + np.sin(self.xtheta[jj]) * yrpt
                            + np.sin(self.xtheta[jj])
                            * (self.pad_radius + self.pad_thickness)
                            * alphapt
                        )

                        self.h[ii, jj] = h_p

                        # Viscosity at faces
                        if jj == 0 and ii == 0:
                            mi_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            mi_w = MI[ii, jj]
                            mi_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            mi_s = MI[ii, jj]
                        elif jj == 0 and 0 < ii < self.nz - 1:
                            mi_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            mi_w = MI[ii, jj]
                            mi_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            mi_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])
                        elif jj == 0 and ii == self.nz - 1:
                            mi_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            mi_w = MI[ii, jj]
                            mi_n = MI[ii, jj]
                            mi_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])
                        elif ii == 0 and 0 < jj < self.nx - 1:
                            mi_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            mi_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            mi_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            mi_s = MI[ii, jj]
                        elif 0 < jj < self.nx - 1 and 0 < ii < self.nz - 1:
                            mi_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            mi_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            mi_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            mi_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])
                        elif ii == self.nz - 1 and 0 < jj < self.nx - 1:
                            mi_e = 0.5 * (MI[ii, jj] + MI[ii, jj + 1])
                            mi_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            mi_n = MI[ii, jj]
                            mi_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])
                        elif ii == 0 and jj == self.nx - 1:
                            mi_e = MI[ii, jj]
                            mi_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            mi_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            mi_s = MI[ii, jj]
                        elif jj == self.nx - 1 and 0 < ii < self.nz - 1:
                            mi_e = MI[ii, jj]
                            mi_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            mi_n = 0.5 * (MI[ii, jj] + MI[ii + 1, jj])
                            mi_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])
                        else:  # jj == self.nx - 1 and ii == self.nz - 1
                            mi_e = MI[ii, jj]
                            mi_w = 0.5 * (MI[ii, jj] + MI[ii, jj - 1])
                            mi_n = MI[ii, jj]
                            mi_s = 0.5 * (MI[ii, jj] + MI[ii - 1, jj])

                        c_e = (
                            (1 / (self.pad_arc**2))
                            * (h_e**3 / (12 * mi_e))
                            * (self.dz / self.dx)
                        )
                        c_w = (
                            (1 / (self.pad_arc**2))
                            * (h_w**3 / (12 * mi_w))
                            * (self.dz / self.dx)
                        )
                        c_n = (
                            (self.pad_radius / self.pad_axial_length) ** 2
                            * (self.dx / self.dz)
                            * (h_n**3 / (12 * mi_n))
                        )
                        c_s = (
                            (self.pad_radius / self.pad_axial_length) ** 2
                            * (self.dx / self.dz)
                            * (h_s**3 / (12 * mi_s))
                        )
                        c_p = -(c_e + c_w + c_n + c_s)
                        b_val = (
                            self.journal_radius / (2 * self.pad_radius * self.pad_arc)
                        ) * self.dz * (h_e - h_w) + h_pt * self.dx * self.dz
                        b[k] = b_val

                        # Fill matrix coefficients
                        if ii == 0 and jj == 0:
                            Mat_coef[k, k] = c_p - c_s - c_w
                            Mat_coef[k, k + 1] = c_e
                            Mat_coef[k, k + self.nx] = c_n
                        elif ii == 0 and 0 < jj < self.nx - 1:
                            Mat_coef[k, k] = c_p - c_s
                            Mat_coef[k, k + 1] = c_e
                            Mat_coef[k, k - 1] = c_w
                            Mat_coef[k, k + self.nx] = c_n
                        elif ii == 0 and jj == self.nx - 1:
                            Mat_coef[k, k] = c_p - c_e - c_s
                            Mat_coef[k, k - 1] = c_w
                            Mat_coef[k, k + self.nx] = c_n
                        elif jj == 0 and 0 < ii < self.nz - 1:
                            Mat_coef[k, k] = c_p - c_w
                            Mat_coef[k, k + 1] = c_e
                            Mat_coef[k, k - self.nx] = c_s
                            Mat_coef[k, k + self.nx] = c_n
                        elif 0 < ii < self.nz - 1 and 0 < jj < self.nx - 1:
                            Mat_coef[k, k] = c_p
                            Mat_coef[k, k - 1] = c_w
                            Mat_coef[k, k - self.nx] = c_s
                            Mat_coef[k, k + self.nx] = c_n
                            Mat_coef[k, k + 1] = c_e
                        elif jj == self.nx - 1 and 0 < ii < self.nz - 1:
                            Mat_coef[k, k] = c_p - c_e
                            Mat_coef[k, k - 1] = c_w
                            Mat_coef[k, k - self.nx] = c_s
                            Mat_coef[k, k + self.nx] = c_n
                        elif jj == 0 and ii == self.nz - 1:
                            Mat_coef[k, k] = c_p - c_n - c_w
                            Mat_coef[k, k + 1] = c_e
                            Mat_coef[k, k - self.nx] = c_s
                        elif ii == self.nz - 1 and 0 < jj < self.nx - 1:
                            Mat_coef[k, k] = c_p - c_n
                            Mat_coef[k, k + 1] = c_e
                            Mat_coef[k, k - 1] = c_w
                            Mat_coef[k, k - self.nx] = c_s
                        else:  # ii == self.nz - 1 and jj == self.nx - 1
                            Mat_coef[k, k] = c_p - c_e - c_n
                            Mat_coef[k, k - 1] = c_w
                            Mat_coef[k, k - self.nx] = c_s

                        k += 1

                # Solve pressure field
                Mat_coef = self._check_diagonal(Mat_coef)
                p_vec = np.linalg.solve(Mat_coef, b)

                cont = 0
                for i_lin in range(self.nz):
                    for j_col in np.arange(self.nx):
                        self.pressure[i_lin, j_col] = p_vec[cont]
                        cont += 1
                        if self.pressure[i_lin, j_col] < 0:
                            self.pressure[i_lin, j_col] = 0

                # Calculate temperature field using energy equation
                T_new = self._solve_energy_equation(MI, n_p, is_equilibrium=True)

            # Calculate hydrodynamic forces and moments
            self._calculate_hydrodynamic_forces(n_p, psi_pad, is_equilibrium=True)

        # Calculate equilibrium residuals
        # FM = np.zeros(self.n_pad + 2)

        # # Force equilibrium in X and Y directions
        # Fhx = np.sum(self.force_x_dim)
        # Fhy = np.sum(self.force_y_dim)

        # FM[0] = Fhx + (self.fxs_load if self.fxs_load is not None else 0)
        # FM[1] = Fhy + (self.fys_load if self.fys_load is not None else 0)
        # FM[2:] = self.moment_j_dim

        # score = np.linalg.norm(FM)

        # Calculate equilibrium residuals
        # FM = np.zeros(self.n_pad + 2)

        # # Force equilibrium in X and Y directions
        # Fhx = np.sum(self.force_x_dim)
        # Fhy = np.sum(self.force_y_dim)

        # # Raw residuals
        # force_x_res = Fhx + (self.fxs_load if self.fxs_load is not None else 0)
        # force_y_res = Fhy + (self.fys_load if self.fys_load is not None else 0)
        # moment_res = self.moment_j_dim

        # # Characteristic scales for normalization
        # F_scale = max(abs(self.fxs_load) if self.fxs_load is not None else 0,
        #             abs(self.fys_load) if self.fys_load is not None else 0,
        #             100.0)  # Minimum 100 N to avoid division by very small numbers

        # M_scale = F_scale * (self.pad_radius)  # N·m (force × lever arm)

        # # Normalized residuals (dimensionless)
        # FM[0] = force_x_res / F_scale
        # FM[1] = force_y_res / F_scale
        # FM[2:] = moment_res / M_scale

        # # Single scalar objective (dimensionless)
        # score = np.linalg.norm(FM)
        # Calculate equilibrium residuals
        Fhx = np.sum(self.force_x_dim)
        Fhy = np.sum(self.force_y_dim)

        # Raw residuals
        force_x_res = Fhx + (self.fxs_load if self.fxs_load is not None else 0)
        force_y_res = Fhy + (self.fys_load if self.fys_load is not None else 0)
        moment_res = self.moment_j_dim

        # INDEPENDENT NORMALIZATION - each residual by its own scale
        Fx_scale = max(abs(self.fxs_load) if self.fxs_load is not None else 0, 100.0)
        Fy_scale = max(abs(self.fys_load) if self.fys_load is not None else 0, 100.0)
        M_scale = max(Fx_scale, Fy_scale) * self.pad_radius  # Use max for moment scale

        # Build residual vector
        FM = np.zeros(self.n_pad + 2)
        FM[0] = force_x_res / Fx_scale  # Normalize X by its own load
        FM[1] = force_y_res / Fy_scale  # Normalize Y by its own load
        FM[2:] = moment_res / M_scale   # Normalize moments

        # Single scalar objective (all components have similar magnitude now)
        score = np.linalg.norm(FM)

        # Record optimization residual
        self.record_optimization_residual(score)

        psi_deg = np.degrees(psi_pad)
        psi_str = "[" + ", ".join(f"{v:.2f}" for v in psi_deg) + "]deg"
        print(
            f"[determine_ecc] iter={getattr(self, 'iteration_count', -1)} "
            f"ecc={eccentricity:.4f} att={np.degrees(attitude_angle):.2f}deg "
            f"{psi_str} "
            f"Fx={Fhx:.3e} Fy={Fhy:.3e} "
            f"Mpads={[f'{m:.3e}' for m in self.moment_j_dim]} "
            f"score={score:.3e}"
        )

        print(f"[determine_ecc] obj_time={time.time()-t0:.2f}s score={score:.3e}")
        if self.iteration_count % 10 == 0:
            ecc, att, pads = x[0], x[1], x[2:]
            print(
                f"[determine_ecc] iter={self.iteration_count} "
                f"ecc={ecc:.4f} att_deg={np.degrees(att):.2f} "
                f"pads_deg={[f'{v:.2f}' for v in np.degrees(pads)]} "
                f"score={score:.3e}"
            )

        return score

    def get_equilibrium_position(self, x):
        """
        Calculate the equilibrium position for a single pad.

        This method serves as the objective function for optimization algorithms
        to find the equilibrium position of a tilting pad. It performs a complete
        thermo-hydrodynamic analysis and returns the absolute moment value.

        Parameters
        ----------
        x : float
            Pad rotation angle [rad] to be evaluated.

        Returns
        -------
        float
            Absolute value of the dimensional moment [N·m] acting on the pad.
        """
        n_p = self.con_np
        temperature_referance = self.reference_temperature * np.ones((self.nz, self.nx))

        # Validation and adjustment of x limits
        x = self._validate_and_adjust_x(x, n_p)
        psi_pad = np.zeros(self.n_pad)
        psi_pad[n_p] = x.item() if hasattr(x, "item") else x

        # Temperature convergence loop (main loop)
        temperature_referance = self._temperature_convergence_loop(
            temperature_referance, n_p, psi_pad
        )

        # Save updated temperature
        self.temperature_init[:, :, n_p] = temperature_referance

        # Calculation of hydrodynamic forces
        self._calculate_hydrodynamic_forces(n_p, psi_pad)

        residual = abs(self.score_dim)

        # Record optimization residual
        self.record_optimization_residual(residual)

        return residual

    def _validate_and_adjust_x(self, x, n_p):
        """
        Validate and adjust pad rotation angle within physical limits.

        Parameters
        ----------
        x : float
            Pad rotation angle [rad] to be validated.
        n_p : int
            Pad index for accessing limits.

        Returns
        -------
        float
            Adjusted pad rotation angle [rad] within physical limits.
        """
        if x > 0.9 * self.alpha_max_chut[n_p]:
            x = 0.8 * self.alpha_max_chut[n_p]
        if x <= 0.8 * self.alpha_min_chut[n_p]:
            x = 0.8 * self.alpha_min_chut[n_p]
        return x

    def _temperature_convergence_loop(self, temperature_referance, n_p, psi_pad):
        """
        Perform temperature convergence loop for thermo-hydrodynamic analysis.

        Parameters
        ----------
        temperature_referance : ndarray
            Initial temperature field [°C]. Shape: (nz, nx).
        n_p : int
            Pad index for the current analysis.
        psi_pad : ndarray
            Pad rotation angles [rad]. Shape: (n_pad,).

        Returns
        -------
        ndarray
            Converged temperature field [°C]. Shape: (nz, nx).
        """
        temperature_tolerance = 0.1  # Celsius degrees (tolerance)
        temperature_iteration = 1.1 * temperature_referance
        cont_temp = 0

        while (
            abs((temperature_referance - temperature_iteration).max())
            >= temperature_tolerance
        ):
            cont_temp += 1
            temperature_iteration = np.array(temperature_referance)

            # Calculate viscosity
            mi = self._calculate_viscosity(temperature_iteration)

            # Assembly and solution of Reynolds equation
            self._solve_reynolds_equation(mi, n_p, psi_pad)

            # Calculate pressure gradients
            self._calculate_pressure_gradients()

            # Solve energy equation
            temperature_referance = self._solve_energy_equation(mi, n_p)

        return temperature_referance

    def _calculate_viscosity(self, temperature_iteration):
        """
        Calculate dimensionless viscosity based on temperature.

        Parameters
        ----------
        temperature_iteration : ndarray
            Temperature field [°C]. Shape: (nz, nx).

        Returns
        -------
        ndarray
            Dimensionless viscosity. Shape: (nz, nx).
        """
        mi_i = self.a_a * np.exp(self.b_b * temperature_iteration)
        # Dimensionless viscosity
        mi = mi_i / self.mu_0
        return mi

    def _solve_reynolds_equation(self, mi, n_p, psi_pad):
        """
        Solve Reynolds equation for pressure field using finite difference method.

        Parameters
        ----------
        mi : ndarray
            Dimensionless viscosity field. Shape: (nz, nx).
        n_p : int
            Pad index for the current analysis.
        psi_pad : ndarray
            Pad rotation angles [rad]. Shape: (n_pad,).

        Returns
        -------
        None
            Results are stored in self.pressure field.
        """
        n_k = self.nx * self.nz
        mat_coef = np.zeros((n_k, n_k))
        b_vec = np.zeros(n_k)

        # Transformation of coordinates (inertial -> pivot)
        xryr, xryrpt, xr, yr, xrpt, yrpt = self._transform_coordinates(n_p)
        alpha = psi_pad[n_p]
        alphapt = 0

        # Vectorization index
        k_idx = 0

        for ii in range(self.nz):
            for jj in range(self.nx):
                # Film thicknesses
                h_p, h_e, h_w, h_n, h_s = self._calculate_film_thicknesses(
                    ii, jj, xr, yr, alpha, n_p
                )

                # Temporal derivative of thickness
                h_pt = self._calculate_h_pt(ii, jj, xrpt, yrpt, alphapt, n_p)

                self.h[ii, jj] = h_p

                # Viscosities at faces (boundary conditions)
                mi_e, mi_w, mi_n, mi_s = self._calculate_face_viscosities(mi, ii, jj)

                # Reynolds equation coefficients (main terms)
                c_e, c_w, c_n, c_s, c_p = self._calculate_reynolds_coefficients(
                    h_e, h_w, h_n, h_s, mi_e, mi_w, mi_n, mi_s
                )

                # Source term (right hand side)
                b_val = self._calculate_reynolds_source_term(h_e, h_w, h_pt, ii, jj)
                b_vec[k_idx] = b_val

                # Fill coefficients matrix (left hand side)
                self._fill_reynolds_matrix(
                    mat_coef, k_idx, ii, jj, c_e, c_w, c_n, c_s, c_p
                )

                k_idx += 1

        # Solution of pressure field
        mat_coef = self._check_diagonal(mat_coef)
        p_vec = np.linalg.solve(mat_coef, b_vec)
        self._update_pressure_field(p_vec)

    def _calculate_film_thicknesses(self, ii, jj, xr, yr, alpha, n_p):
        """
        Calculate film thicknesses at control volume faces.

        Parameters
        ----------
        ii : int
            Axial mesh index.
        jj : int
            Circumferential mesh index.
        xr : float
            Journal position in pad coordinate system [m].
        yr : float
            Journal position in pad coordinate system [m].
        alpha : float
            Pad rotation angle [rad].
        n_p : int
            Pad index.

        Returns
        -------
        tuple
            Film thicknesses (h_p, h_e, h_w, h_n, h_s) where:
            - h_p: thickness at center point
            - h_e: thickness at east face
            - h_w: thickness at west face
            - h_n: thickness at north face
            - h_s: thickness at south face
        """

        def _thickness_at_angle(theta):
            """Calculate thickness at a specific angle."""
            return (
                self.pad_radius
                - self.journal_radius
                - (
                    np.sin(theta)
                    * (yr + alpha * (self.pad_radius + self.pad_thickness))
                    + np.cos(theta)
                    * (
                        xr
                        + self.pad_radius
                        - self.journal_radius
                        - self.radial_clearance
                    )
                )
            ) / self.radial_clearance

        theta_center = self.xtheta[jj]
        theta_e = theta_center + 0.5 * self.dtheta
        theta_w = theta_center - 0.5 * self.dtheta

        h_p = _thickness_at_angle(theta_center)
        h_e = _thickness_at_angle(theta_e)
        h_w = _thickness_at_angle(theta_w)
        h_n = h_p  # Simplified for north face
        h_s = h_p  # Simplified for south face

        return h_p, h_e, h_w, h_n, h_s

    def _calculate_h_pt(self, ii, jj, xrpt, yrpt, alphapt, n_p):
        """
        Calculate temporal derivative of film thickness.

        Parameters
        ----------
        ii : int
            Axial mesh index.
        jj : int
            Circumferential mesh index.
        xrpt : float
            Journal velocity in pad coordinate system [m/s].
        yrpt : float
            Journal velocity in pad coordinate system [m/s].
        alphapt : float
            Pad angular velocity [rad/s].
        n_p : int
            Pad index.

        Returns
        -------
        float
            Temporal derivative of film thickness [1/s].
        """
        h_pt = -(1 / (self.radial_clearance * self.speed)) * (
            np.cos(self.xtheta[jj]) * xrpt
            + np.sin(self.xtheta[jj]) * yrpt
            + np.sin(self.xtheta[jj]) * (self.pad_radius + self.pad_thickness) * alphapt
        )
        return h_pt

    def _calculate_face_viscosities(self, mi, ii, jj):
        """
        Calculate viscosities at control volume faces using boundary conditions.

        Parameters
        ----------
        mi : ndarray
            Dimensionless viscosity field. Shape: (nz, nx).
        ii : int
            Axial mesh index.
        jj : int
            Circumferential mesh index.

        Returns
        -------
        tuple
            Face viscosities (mi_e, mi_w, mi_n, mi_s) where:
            - mi_e: viscosity at east face
            - mi_w: viscosity at west face
            - mi_n: viscosity at north face
            - mi_s: viscosity at south face
        """

        def _get_face_viscosity(direction):
            """Helper function to calculate viscosity at a specific face."""
            if direction == "east":
                if jj < self.nx - 1:
                    return 0.5 * (mi[ii, jj] + mi[ii, jj + 1])
                else:
                    return mi[ii, jj]
            elif direction == "west":
                if jj > 0:
                    return 0.5 * (mi[ii, jj] + mi[ii, jj - 1])
                else:
                    return mi[ii, jj]
            elif direction == "north":
                if ii < self.nz - 1:
                    return 0.5 * (mi[ii, jj] + mi[ii + 1, jj])
                else:
                    return mi[ii, jj]
            elif direction == "south":
                if ii > 0:
                    return 0.5 * (mi[ii, jj] + mi[ii - 1, jj])
                else:
                    return mi[ii, jj]

        mi_e = _get_face_viscosity("east")
        mi_w = _get_face_viscosity("west")
        mi_n = _get_face_viscosity("north")
        mi_s = _get_face_viscosity("south")

        return mi_e, mi_w, mi_n, mi_s

    def _calculate_reynolds_coefficients(
        self, h_e, h_w, h_n, h_s, mi_e, mi_w, mi_n, mi_s
    ):
        """
        Calculate finite difference coefficients for Reynolds equation.

        Parameters
        ----------
        h_e, h_w, h_n, h_s : float
            Film thicknesses at east, west, north, and south faces.
        mi_e, mi_w, mi_n, mi_s : float
            Viscosities at east, west, north, and south faces.

        Returns
        -------
        tuple
            Reynolds equation coefficients (c_e, c_w, c_n, c_s, c_p) where:
            - c_e: east face coefficient
            - c_w: west face coefficient
            - c_n: north face coefficient
            - c_s: south face coefficient
            - c_p: center point coefficient
        """
        c_e = (1 / (self.pad_arc**2)) * (h_e**3 / (12 * mi_e)) * (self.dz / self.dx)
        c_w = (1 / (self.pad_arc**2)) * (h_w**3 / (12 * mi_w)) * (self.dz / self.dx)
        c_n = (
            (self.pad_radius / self.pad_axial_length) ** 2
            * (self.dx / self.dz)
            * (h_n**3 / (12 * mi_n))
        )
        c_s = (
            (self.pad_radius / self.pad_axial_length) ** 2
            * (self.dx / self.dz)
            * (h_s**3 / (12 * mi_s))
        )
        c_p = -(c_e + c_w + c_n + c_s)

        return c_e, c_w, c_n, c_s, c_p

    def _calculate_reynolds_source_term(self, h_e, h_w, h_pt, ii, jj):
        """
        Calculate source term for Reynolds equation right-hand side.

        Parameters
        ----------
        h_e, h_w : float
            Film thicknesses at east and west faces.
        h_pt : float
            Temporal derivative of film thickness [1/s].
        ii, jj : int
            Axial and circumferential mesh indices.

        Returns
        -------
        float
            Source term value for Reynolds equation.
        """
        b_val = (
            self.journal_radius / (2 * self.pad_radius * self.pad_arc)
        ) * self.dz * (h_e - h_w) + h_pt * self.dx * self.dz
        return b_val

    def _fill_reynolds_matrix(self, mat_coef, k_idx, ii, jj, c_e, c_w, c_n, c_s, c_p):
        """
        Fill coefficients matrix for Reynolds equation finite difference system.

        Parameters
        ----------
        mat_coef : ndarray
            Coefficient matrix to be filled. Shape: (n_k, n_k).
        k_idx : int
            Linear index for current mesh point.
        ii, jj : int
            Axial and circumferential mesh indices.
        c_e, c_w, c_n, c_s, c_p : float
            Finite difference coefficients for east, west, north, south, and center.

        Returns
        -------
        None
            Matrix is modified in place.
        """
        if ii == 0 and jj == 0:
            mat_coef[k_idx, k_idx] = c_p - c_s - c_w
            mat_coef[k_idx, k_idx + 1] = c_e
            mat_coef[k_idx, k_idx + self.nx] = c_n
        elif ii == 0 and 0 < jj < self.nx - 1:
            mat_coef[k_idx, k_idx] = c_p - c_s
            mat_coef[k_idx, k_idx + 1] = c_e
            mat_coef[k_idx, k_idx - 1] = c_w
            mat_coef[k_idx, k_idx + self.nx] = c_n
        elif ii == 0 and jj == self.nx - 1:
            mat_coef[k_idx, k_idx] = c_p - c_e - c_s
            mat_coef[k_idx, k_idx - 1] = c_w
            mat_coef[k_idx, k_idx + self.nx] = c_n
        elif jj == 0 and 0 < ii < self.nz - 1:
            mat_coef[k_idx, k_idx] = c_p - c_w
            mat_coef[k_idx, k_idx + 1] = c_e
            mat_coef[k_idx, k_idx - self.nx] = c_s
            mat_coef[k_idx, k_idx + self.nx] = c_n
        elif 0 < ii < self.nz - 1 and 0 < jj < self.nx - 1:
            mat_coef[k_idx, k_idx] = c_p
            mat_coef[k_idx, k_idx - 1] = c_w
            mat_coef[k_idx, k_idx - self.nx] = c_s
            mat_coef[k_idx, k_idx + self.nx] = c_n
            mat_coef[k_idx, k_idx + 1] = c_e
        elif jj == self.nx - 1 and 0 < ii < self.nz - 1:
            mat_coef[k_idx, k_idx] = c_p - c_e
            mat_coef[k_idx, k_idx - 1] = c_w
            mat_coef[k_idx, k_idx - self.nx] = c_s
            mat_coef[k_idx, k_idx + self.nx] = c_n
        elif jj == 0 and ii == self.nz - 1:
            mat_coef[k_idx, k_idx] = c_p - c_n - c_w
            mat_coef[k_idx, k_idx + 1] = c_e
            mat_coef[k_idx, k_idx - self.nx] = c_s
        elif ii == self.nz - 1 and 0 < jj < self.nx - 1:
            mat_coef[k_idx, k_idx] = c_p - c_n
            mat_coef[k_idx, k_idx + 1] = c_e
            mat_coef[k_idx, k_idx - 1] = c_w
            mat_coef[k_idx, k_idx - self.nx] = c_s
        else:  # ii == self.nz - 1 and jj == self.nx - 1
            mat_coef[k_idx, k_idx] = c_p - c_e - c_n
            mat_coef[k_idx, k_idx - 1] = c_w
            mat_coef[k_idx, k_idx - self.nx] = c_s

    def _update_pressure_field(self, p_vec):
        """
        Update pressure field from solution vector with non-negative constraint.

        Parameters
        ----------
        p_vec : ndarray
            Solution vector from linear system. Shape: (n_k,).

        Returns
        -------
        None
            Pressure field is updated in self.pressure.
        """
        cont = 0
        for i_lin in range(self.nz):
            for j_col in range(self.nx):
                self.pressure[i_lin, j_col] = p_vec[cont]
                cont += 1
                if self.pressure[i_lin, j_col] < 0:
                    self.pressure[i_lin, j_col] = 0

    def _calculate_pressure_gradients(self):
        """
        Calculate pressure gradients using finite differences.

        Returns
        -------
        None
            Pressure gradients are stored in self.dp_dx and self.dp_dz.
        """
        for ii in range(self.nz):
            for jj in range(self.nx):
                # Gradient in X direction
                if jj == 0:
                    self.dp_dx[ii, jj] = (
                        self.pressure[ii, jj + 1] - self.pressure[ii, jj]
                    ) / self.dx
                elif jj == self.nx - 1:
                    self.dp_dx[ii, jj] = (0 - self.pressure[ii, jj]) / self.dx
                else:
                    self.dp_dx[ii, jj] = (
                        self.pressure[ii, jj + 1] - self.pressure[ii, jj]
                    ) / self.dx

                # Gradient in Z direction
                if ii == 0:
                    self.dp_dz[ii, jj] = (
                        self.pressure[ii + 1, jj] - self.pressure[ii, jj]
                    ) / self.dz
                elif ii == self.nz - 1:
                    self.dp_dz[ii, jj] = (0 - self.pressure[ii, jj]) / self.dz
                else:
                    self.dp_dz[ii, jj] = (
                        self.pressure[ii + 1, jj] - self.pressure[ii, jj]
                    ) / self.dz

    def _solve_energy_equation(self, mi, n_p, is_equilibrium=False):
        """
        Solve energy equation for temperature field using finite difference method.

        Parameters
        ----------
        mi : ndarray
            Dimensionless viscosity field. Shape: (nz, nx).
        n_p : int
            Pad index for the current analysis.
        is_equilibrium : bool, optional
            Whether this is called during equilibrium optimization. Default is False.

        Returns
        -------
        ndarray
            Temperature field [°C]. Shape: (nz, nx).
        """
        n_k = self.nx * self.nz
        mat_coef_t = np.zeros((n_k, n_k))
        b_t = np.zeros(n_k)

        k_t_idx = 0
        for ii in range(self.nz):
            for jj in range(self.nx):
                # Turbulence and turbulent viscosity
                mi_t = self._calculate_turbulent_viscosity(
                    mi, ii, jj, n_p, is_equilibrium
                )

                # Energy equation coefficients
                a_e, a_w, a_n, a_s, a_p_coef = self._calculate_energy_coefficients(
                    ii, jj, mi_t
                )

                # Energy equation source term (right hand side)
                b_t_val = self._calculate_energy_source_term(ii, jj, mi_t, n_p)
                b_t[k_t_idx] = b_t_val

                # Fill coefficients matrix of energy equation (left hand side)
                self._fill_energy_matrix(
                    mat_coef_t, b_t, k_t_idx, ii, jj, a_e, a_w, a_n, a_s, a_p_coef
                )

                k_t_idx += 1

        # Solution of temperature field
        mat_coef_t = self._check_diagonal(mat_coef_t)
        t0 = time.time()
        t_vec = np.linalg.solve(mat_coef_t, b_t)
        temperature_referance = self._update_temperature_field(t_vec)

        print(f"[solve_energy_equation] solve_time={time.time()-t0:.2f}s")

        return temperature_referance

    def _calculate_turbulent_viscosity(self, mi, ii, jj, n_p, is_equilibrium=False):
        """
        Calculate turbulent viscosity using van Driest turbulence model.

        Parameters
        ----------
        mi : ndarray
            Dimensionless viscosity field. Shape: (nz, nx).
        ii, jj : int
            Axial and circumferential mesh indices.
        n_p : int
            Pad index for the current analysis.
        is_equilibrium : bool, optional
            Whether this is called during equilibrium optimization. Default is False.

        Returns
        -------
        float
            Turbulent viscosity at the specified mesh point.
        """
        h_p_loc = self.h[ii, jj]
        mi_p = mi[ii, jj]

        # Local Reynolds number
        reynolds_local = (
            self.rho
            * self.speed
            * self.journal_radius
            * (h_p_loc / self.pad_axial_length)
            * self.radial_clearance
            / (self.mu_0 * mi_p)
        )

        if not is_equilibrium:
            # Store in instance attribute for normal calculation
            self.reynolds_field[ii, jj, n_p] = reynolds_local

        # Transition factor for turbulence
        if reynolds_local <= 500:
            delta_turb = 0
        elif 400 < reynolds_local <= 1000:
            delta_turb = 1 - ((1000 - reynolds_local) / 500) ** (1 / 8)
        else:
            delta_turb = 1

        # Calculate velocity derivatives
        if is_equilibrium:
            # Use mi_p directly for equilibrium
            du_dx = ((h_p_loc / mi_p) * self.dp_dx[ii, jj]) - (self.speed / h_p_loc)
            dw_dx = (h_p_loc / mi_p) * self.dp_dz[ii, jj]
            tau = mi_p * np.sqrt(du_dx**2 + dw_dx**2)
        else:
            # Use turbulent viscosity for normal calculation
            du_dx = ((h_p_loc / self.mu_turb[ii, jj, n_p]) * self.dp_dx[ii, jj]) - (
                self.speed / h_p_loc
            )
            dw_dx = (h_p_loc / self.mu_turb[ii, jj, n_p]) * self.dp_dz[ii, jj]
            tau = self.mu_turb[ii, jj, n_p] * np.sqrt(du_dx**2 + dw_dx**2)

        # Dimensionless distance from wall
        if is_equilibrium:
            y_wall = (
                (h_p_loc * self.radial_clearance * 2) / (self.mu_0 * mi_p / self.rho)
            ) * ((abs(tau) / self.rho) ** 0.5)
        else:
            y_wall = (
                (h_p_loc * self.radial_clearance * 2)
                / (self.mu_0 * self.mu_turb[ii, jj, n_p] / self.rho)
            ) * ((abs(tau) / self.rho) ** 0.5)

        # Turbulent viscosity (van Driest model)
        emv = 0.4 * (y_wall - (10.7 * np.tanh(y_wall / 10.7)))

        if is_equilibrium:
            mi_t = mi_p * (1 + (delta_turb * emv))
        else:
            self.mu_turb[ii, jj, n_p] = mi_p * (1 + (delta_turb * emv))
            mi_t = self.mu_turb[ii, jj, n_p]

        return mi_t

    def _calculate_energy_coefficients(self, ii, jj, mi_t):
        """
        Calculate finite difference coefficients for energy equation.

        Parameters
        ----------
        ii, jj : int
            Axial and circumferential mesh indices.
        mi_t : float
            Turbulent viscosity at the mesh point.

        Returns
        -------
        tuple
            Energy equation coefficients (a_e, a_w, a_n, a_s, a_p_coef) where:
            - a_e: east face coefficient
            - a_w: west face coefficient
            - a_n: north face coefficient
            - a_s: south face coefficient
            - a_p_coef: center point coefficient
        """
        h_p_loc = self.h[ii, jj]

        # Auxiliary factor for flow direction
        aux_up = 1 if self.xz[ii] >= 0 else 0

        # Convection and conduction coefficients
        a_e = -(self.kt * h_p_loc * self.dz) / (
            self.rho
            * self.cp
            * self.speed
            * ((self.pad_arc * self.pad_radius) ** 2)
            * self.dx
        )

        a_w = (
            ((h_p_loc**3) * self.dp_dx[ii, jj] * self.dz)
            / (12 * mi_t * (self.pad_arc**2))
            - (
                (self.journal_radius * h_p_loc * self.dz)
                / (2 * self.pad_radius * self.pad_arc)
            )
            - (self.kt * h_p_loc * self.dz)
            / (
                self.rho
                * self.cp
                * self.speed
                * ((self.pad_arc * self.pad_radius) ** 2)
                * self.dx
            )
        )

        # North and south coefficients (with upwind scheme)
        a_n_1 = (aux_up - 1) * (
            ((self.pad_radius**2) * (h_p_loc**3) * self.dp_dz[ii, jj] * self.dx)
            / (12 * (self.pad_axial_length**2) * mi_t)
        )
        a_s_1 = (aux_up) * (
            ((self.pad_radius**2) * (h_p_loc**3) * self.dp_dz[ii, jj] * self.dx)
            / (12 * (self.pad_axial_length**2) * mi_t)
        )

        a_n_2 = -(self.kt * h_p_loc * self.dx) / (
            self.rho * self.cp * self.speed * (self.pad_axial_length**2) * self.dz
        )
        a_s_2 = -(self.kt * h_p_loc * self.dx) / (
            self.rho * self.cp * self.speed * (self.pad_axial_length**2) * self.dz
        )

        a_n = a_n_1 + a_n_2
        a_s = a_s_1 + a_s_2
        a_p_coef = -(a_e + a_w + a_n + a_s)

        return a_e, a_w, a_n, a_s, a_p_coef

    def _calculate_energy_source_term(self, ii, jj, mi_t, n_p):
        """
        Calculate source term for energy equation right-hand side.

        Parameters
        ----------
        ii, jj : int
            Axial and circumferential mesh indices.
        mi_t : float
            Turbulent viscosity at the mesh point.
        n_p : int
            Pad index for the current analysis.

        Returns
        -------
        float
            Source term value for energy equation.
        """
        h_p_loc = self.h[ii, jj]
        h_pt = -(1 / (self.radial_clearance * self.speed)) * (
            np.cos(self.xtheta[jj]) * 0  # xrpt
            + np.sin(self.xtheta[jj]) * 0  # yrpt
            + np.sin(self.xtheta[jj])
            * (self.pad_radius + self.pad_thickness)
            * 0  # alphapt
        )

        aux_b_t = (self.speed * self.mu_0) / (
            self.rho * self.cp * self.oil_supply_temperature * self.radial_clearance
        )

        # Heat generation terms by viscous dissipation (right hand side)
        b_t_g = (
            self.mu_0
            * self.speed
            * (self.journal_radius**2)
            * self.dx
            * self.dz
            * self.pressure[ii, jj]
            * h_pt
        ) / (
            self.rho * self.cp * self.reference_temperature * (self.radial_clearance**2)
        )

        b_t_h = (self.speed * self.mu_0 * (h_pt**2) * 4 * mi_t * self.dx * self.dz) / (
            self.rho * self.cp * self.reference_temperature * 3 * h_p_loc
        )

        b_t_i = (
            aux_b_t
            * (mi_t * (self.journal_radius**2) * self.dx * self.dz)
            / (h_p_loc * self.radial_clearance)
        )

        b_t_j = (
            aux_b_t
            * (
                (self.pad_radius**2)
                * (h_p_loc**3)
                * (self.dp_dx[ii, jj] ** 2)
                * self.dx
                * self.dz
            )
            / (12 * self.radial_clearance * (self.pad_arc**2) * mi_t)
        )

        b_t_k = (
            aux_b_t
            * (
                (self.pad_radius**4)
                * (h_p_loc**3)
                * (self.dp_dz[ii, jj] ** 2)
                * self.dx
                * self.dz
            )
            / (12 * self.radial_clearance * (self.pad_axial_length**2) * mi_t)
        )

        b_t_val = b_t_g + b_t_h + b_t_i + b_t_j + b_t_k
        return b_t_val

    def _fill_energy_matrix(
        self, mat_coef_t, b_t, k_t_idx, ii, jj, a_e, a_w, a_n, a_s, a_p_coef
    ):
        """
        Fill coefficients matrix for energy equation finite difference system.

        Parameters
        ----------
        mat_coef_t : ndarray
            Coefficient matrix to be filled. Shape: (n_k, n_k).
        b_t : ndarray
            Right-hand side vector to be modified.
        k_t_idx : int
            Linear index for current mesh point.
        ii, jj : int
            Axial and circumferential mesh indices.
        a_e, a_w, a_n, a_s, a_p_coef : float
            Energy equation coefficients for east, west, north, south, and center.

        Returns
        -------
        None
            Matrix and vector are modified in place.
        """
        # Filling based on position in the mesh (left hand side)
        if ii == 0 and jj == 0:
            mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_s - a_w
            mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
            mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n
            b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (
                self.oil_supply_temperature / self.reference_temperature
            )

        elif ii == 0 and 0 < jj < self.nx - 1:
            mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_s
            mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
            mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
            mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n

        elif ii == 0 and jj == self.nx - 1:
            mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e + a_s
            mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
            mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n

        elif jj == 0 and 0 < ii < self.nz - 1:
            mat_coef_t[k_t_idx, k_t_idx] = a_p_coef - a_w
            mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
            mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
            mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n
            b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (
                self.oil_supply_temperature / self.reference_temperature
            )

        elif 0 < ii < self.nz - 1 and 0 < jj < self.nx - 1:
            mat_coef_t[k_t_idx, k_t_idx] = a_p_coef
            mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
            mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
            mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n
            mat_coef_t[k_t_idx, k_t_idx + 1] = a_e

        elif jj == self.nx - 1 and 0 < ii < self.nz - 1:
            mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e
            mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
            mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
            mat_coef_t[k_t_idx, k_t_idx + self.nx] = a_n

        elif jj == 0 and ii == self.nz - 1:
            mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_n - a_w
            mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
            mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s
            b_t[k_t_idx] = b_t[k_t_idx] - 2 * a_w * (
                self.oil_supply_temperature / self.reference_temperature
            )

        elif ii == self.nz - 1 and 0 < jj < self.nx - 1:
            mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_n
            mat_coef_t[k_t_idx, k_t_idx + 1] = a_e
            mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
            mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s

        else:  # ii == self.nz - 1 and jj == self.nx - 1
            mat_coef_t[k_t_idx, k_t_idx] = a_p_coef + a_e + a_n
            mat_coef_t[k_t_idx, k_t_idx - 1] = a_w
            mat_coef_t[k_t_idx, k_t_idx - self.nx] = a_s

    def _update_temperature_field(self, t_vec):
        """
        Update temperature field from solution vector.

        Parameters
        ----------
        t_vec : ndarray
            Solution vector from linear system. Shape: (n_k,).

        Returns
        -------
        ndarray
            Temperature field [°C]. Shape: (nz, nx).
        """
        temperature_referance = self.reference_temperature * np.ones((self.nz, self.nx))
        cont = 0
        for i_lin in range(self.nz):
            for j_col in range(self.nx):
                temperature_referance[i_lin, j_col] = (
                    self.reference_temperature * t_vec[cont]
                )
                cont += 1
        return temperature_referance

    def _calculate_hydrodynamic_forces(self, n_p, psi_pad, is_equilibrium=False):
        """
        Calculate hydrodynamic forces and moments from pressure field.

        Parameters
        ----------
        n_p : int
            Pad index for the current analysis.
        psi_pad : ndarray
            Pad rotation angles [rad]. Shape: (n_pad,).
        is_equilibrium : bool, optional
            Whether this is called during equilibrium optimization. Default is False.

        Returns
        -------
        None
            Forces and moments are stored as instance attributes.
        """
        # Dimensional pressure
        if is_equilibrium:
            p_dimen = (
                self.pressure
                * (self.mu_0 * self.speed * self.pad_radius**2)
                / (self.radial_clearance**2)
            )
        else:
            self.P_dimen = (
                self.pressure
                * (self.mu_0 * self.speed * self.pad_radius**2)
                / (self.radial_clearance**2)
            )

        # Auxiliary matrices for integration
        aux_f1 = np.zeros((self.nz, self.nx))
        aux_f2 = np.zeros((self.nz, self.nx))
        for ni in range(self.nz):
            aux_f1[ni, :] = np.cos(self.xtheta)
            aux_f2[ni, :] = np.sin(self.xtheta)

        # Forces in pad coordinate system
        y_teta_f1 = self.pressure * aux_f1
        f1_teta = np.trapezoid(y_teta_f1, self.xtheta)
        self.force_1[n_p] = -np.trapezoid(f1_teta, self.xz)

        y_teta_f2 = self.pressure * aux_f2
        f2_teta = np.trapezoid(y_teta_f2, self.xtheta)
        self.force_2[n_p] = -np.trapezoid(f2_teta, self.xz)

        # Resultant forces in inertial coordinate system
        self.force_x[n_p] = self.force_1[n_p] * np.cos(
            psi_pad[n_p] + self.pivot_angle[n_p]
        )
        self.force_y[n_p] = self.force_1[n_p] * np.sin(
            psi_pad[n_p] + self.pivot_angle[n_p]
        )
        self.moment_j[n_p] = self.force_2[n_p] * (self.pad_radius + self.pad_thickness)

        # Dimensional forces
        self.force_x_dim[n_p] = self.force_x[n_p] * self.dimensionless_force[n_p]
        self.force_y_dim[n_p] = self.force_y[n_p] * self.dimensionless_force[n_p]
        self.moment_j_dim[n_p] = self.moment_j[n_p] * self.dimensionless_force[n_p]
        self.force_j_dim[n_p] = self.force_1[n_p] * self.dimensionless_force[n_p]

        # Dimensional score for return (only for normal calculation)
        if not is_equilibrium:
            self.score_dim = self.moment_j[n_p] * self.dimensionless_force[n_p]

    def _transform_coordinates(self, n_p, eccentricity=None, attitude_angle=None):
        """
        Transform coordinates from inertial to pad coordinate system.

        Parameters
        ----------
        n_p : int
            Pad index for the current analysis.
        eccentricity : float, optional
            Eccentricity ratio for equilibrium calculation.
        attitude_angle : float, optional
            Attitude angle [rad] for equilibrium calculation.

        Returns
        -------
        tuple
            Coordinate transformation results (xryr, xryrpt, xr, yr, xrpt, yrpt) where:
            - xryr: position vector in pad coordinates
            - xryrpt: velocity vector in pad coordinates
            - xr, yr: journal position in pad coordinates [m]
            - xrpt, yrpt: journal velocity in pad coordinates [m/s]
        """
        if eccentricity is not None and attitude_angle is not None:
            # Use provided parameters for equilibrium calculation
            xx = eccentricity * self.radial_clearance * np.cos(attitude_angle)
            yy = eccentricity * self.radial_clearance * np.sin(attitude_angle)
        else:
            # Use instance attributes for normal calculation
            xx = self.eccentricity * self.radial_clearance * np.cos(self.attitude_angle)
            yy = self.eccentricity * self.radial_clearance * np.sin(self.attitude_angle)

        xryr = np.dot(
            [
                [np.cos(self.pivot_angle[n_p]), np.sin(self.pivot_angle[n_p])],
                [-np.sin(self.pivot_angle[n_p]), np.cos(self.pivot_angle[n_p])],
            ],
            [[xx], [yy]],
        )

        xryrpt = np.dot(
            [
                [np.cos(self.pivot_angle[n_p]), np.sin(self.pivot_angle[n_p])],
                [-np.sin(self.pivot_angle[n_p]), np.cos(self.pivot_angle[n_p])],
            ],
            [[self.x_pt], [self.y_pt]],
        )

        xr = xryr[0, 0]
        yr = xryr[1, 0]

        xrpt = xryrpt[0, 0]
        yrpt = xryrpt[1, 0]

        return xryr, xryrpt, xr, yr, xrpt, yrpt

    def _initialize_arrays(self):
        """Initialize all arrays used in the thermo-hydrodynamic analysis.

        This method creates and initializes all the arrays needed for the analysis,
        including pressure, temperature, film thickness, forces, and other field variables.

        Returns
        -------
        None
            Arrays are stored as instance attributes.
        """
        shape_2d = (self.nz, self.nx)
        shape_3d = (self.nz, self.nx, self.n_pad)

        # Pressure and film thickness arrays
        self.pressure_dim = np.zeros(shape_3d)
        self.h_0 = np.zeros(shape_3d)
        self.h_dim = np.zeros(shape_3d)
        self.h = np.zeros(shape_2d)
        self.pressure = np.zeros(shape_2d)
        self.pressure_nd = np.zeros(shape_3d)
        self.pressure_prev = np.zeros(shape_2d)

        # Temperature and thermal arrays
        self.temperature_init = self.reference_temperature * np.ones(shape_3d)

        # Pressure gradients
        self.dp_dx = np.zeros(shape_2d)
        self.dp_dz = np.zeros(shape_2d)

        # Turbulence and Reynolds arrays
        self.reynolds_field = np.zeros(shape_3d)
        self.mu_turb = 1.3 * np.ones(shape_3d)

        # Force and moment arrays
        self.force_x = np.zeros(self.n_pad)
        self.force_y = np.zeros(self.n_pad)
        self.moment_j = np.zeros(self.n_pad)
        self.force_x_dim = np.zeros(self.n_pad)
        self.force_y_dim = np.zeros(self.n_pad)
        self.moment_j_dim = np.zeros(self.n_pad)
        self.moment_j_new = np.zeros(self.n_pad)
        self.force_1 = np.zeros(self.n_pad)
        self.force_new = np.zeros(self.n_pad)
        self.force_j_dim = np.zeros(self.n_pad)
        self.force_2 = np.zeros(self.n_pad)
        self.force_2_new = np.zeros(self.n_pad)

        # Pivot film thickness
        self.h_pivot = np.zeros(self.n_pad)

    def _reset_force_arrays(self):
        """
        Reset force arrays for each frequency iteration.

        Returns
        -------
        None
            Force arrays are reset to zero in place.
        """
        self.force_x_dim = np.zeros(self.n_pad)
        self.force_y_dim = np.zeros(self.n_pad)
        self.moment_j_dim = np.zeros(self.n_pad)
        self.force_x = np.zeros(self.n_pad)
        self.force_y = np.zeros(self.n_pad)
        self.moment_j = np.zeros(self.n_pad)

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

    def _check_diagonal(self, matrix, residual=1e-10):
        """Check matrix diagonal for zero elements and replace them with a small residual value.

        Parameters
        ----------
        matrix : np.ndarray
            Matrix to be checked
        residual : float, optional
            Small value to replace zeros in diagonal. Default is 1e-10.

        Returns
        -------
        np.ndarray
            Matrix with adjusted diagonal
        """
        matrix = matrix.copy()
        diag_indices = np.diag_indices_from(matrix)
        zero_diag = np.abs(matrix[diag_indices]) < residual
        if np.any(zero_diag):
            matrix[diag_indices[0][zero_diag], diag_indices[1][zero_diag]] = residual
        return matrix

    def plot_pressure_distribution(self, fig=None, **kwargs):
        """Plot pressure distribution for the tilting pad bearing.

        Parameters
        ----------
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

        XH, YH = np.meshgrid(self.xtheta, self.xz)

        # Surface plot for pressure
        fig.add_trace(
            go.Surface(
                x=XH,
                y=YH,
                z=1e-6 * self.pressure_dim[:, :, self.pad_in],
                colorscale="Viridis",
                name="Pressure field",
                showscale=True,
            )
        )

        fig.update_layout(
            title=dict(text="Pressure Distribution", font=dict(size=24)),
            scene=dict(
                xaxis_title=dict(text="X direction [rad]", font=dict(size=14)),
                yaxis_title=dict(text="Z direction [-]", font=dict(size=14)),
                zaxis_title=dict(text="Pressure [MPa]", font=dict(size=14)),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            **kwargs,
        )

        return fig

    def plot_temperature_distribution(self, fig=None, **kwargs):
        """Plot temperature distribution for the tilting pad bearing.

        Parameters
        ----------
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

        XH, YH = np.meshgrid(self.xtheta, self.xz)

        # Contour plot for temperature
        fig.add_trace(
            go.Contour(
                x=XH[0],
                y=YH[:, 0],
                z=self.temperature_init[:, :, self.pad_in],
                colorscale="Viridis",
                name="Temperature field",
                showscale=True,
                colorbar=dict(title="Temperature [°C]", titleside="right"),
            )
        )

        fig.update_layout(
            title=dict(text="Temperature Distribution", font=dict(size=24)),
            xaxis=dict(
                title=dict(text="X direction [rad]", font=dict(size=14)),
                showgrid=True,
                gridwidth=1,
                gridcolor=tableau_colors["gray"],
            ),
            yaxis=dict(
                title=dict(text="Z direction [-]", font=dict(size=14)),
                showgrid=True,
                gridwidth=1,
                gridcolor=tableau_colors["gray"],
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            **kwargs,
        )

        return fig

    def plot_pad_results(self, fig=None, **kwargs):
        """Plot pad results including forces and moments.

        Parameters
        ----------
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

        for i in range(self.n_pad):
            fig.add_trace(
                go.Scatter(
                    x=self.xtheta,
                    y=self.pressure_dim[:, i],
                    name=f"Pad {i + 1}",
                    line=dict(
                        color=tableau_colors[
                            list(tableau_colors.keys())[i % len(tableau_colors)]
                        ]
                    ),
                )
            )

        fig.update_layout(
            title=dict(text="Pad Results", font=dict(size=24)),
            xaxis=dict(
                title=dict(text="Angle [rad]", font=dict(size=14)),
                showgrid=True,
                gridwidth=1,
                gridcolor=tableau_colors["gray"],
            ),
            yaxis=dict(
                title=dict(text="Pressure [Pa]", font=dict(size=14)),
                showgrid=True,
                gridwidth=1,
                gridcolor=tableau_colors["gray"],
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            **kwargs,
        )

        return fig

    def plot_results(self, show_plots=False, freq_index=0):
        """
        Plot pressure and temperature field results for tilting pad bearing analysis.

        This method generates comprehensive visualization plots for the calculated
        pressure and temperature fields at a specific frequency. It creates scatter
        plots and contour plots for pressure and temperature distributions across
        all pads.

        Parameters
        ----------
        show_plots : bool, optional
            Whether to automatically display the plots. If True, attempts to show
            all plots using the default display method. If False, returns figure
            objects for manual display. Default is False.
        freq_index : int, optional
            Index of the frequency to plot results for. Must be within the range
            of calculated frequencies. Default is 0 (first frequency).

        Returns
        -------
        dict
            Dictionary containing four Plotly figure objects:
            - 'pressure_scatter': Scatter plot of pressure distribution
            - 'temperature_scatter': Scatter plot of temperature distribution
            - 'pressure_2D': 2D contour plot of pressure field
            - 'temperature_2D': 2D contour plot of temperature field

        Notes
        -----
        The method creates visualizations for all pads showing the pressure and
        temperature distributions in both scatter and contour formats for the
        selected frequency.
        """
        # Get fields for the selected frequency
        pressure_field = self.pressure_fields[freq_index]
        temperature_field = self.temperature_fields[freq_index]

        d_axial = self.pad_axial_length / self.nz
        axial = np.arange(0, self.pad_axial_length + d_axial, d_axial)
        axial = axial[1:] - np.diff(axial) / 2

        ang = []

        for k in range(self.n_pad):
            ang1 = (self.xtheta + self.pivot_angle[k]) * 180 / np.pi
            ang.append(ang1)

        fig_SP = self.plot_scatter(
            x_data=ang, y_data=pressure_field, pos=15, y_title="Pressure (Pa)"
        )

        fig_ST = self.plot_scatter(
            x_data=ang, y_data=temperature_field, pos=15, y_title="Temperature (ºC)"
        )

        fig_CP = self.plot_contourP(
            x_data=ang, y_data=axial, z_data=pressure_field, z_title="Pressure (Pa)"
        )

        fig_CT = self.plot_contourT(
            x_data=ang,
            y_data=axial,
            z_data=temperature_field,
            z_title="Temperature (ºC)",
        )

        figures = {
            "pressure_scatter": fig_SP,
            "temperature_scatter": fig_ST,
            "pressure_2D": fig_CP,
            "temperature_2D": fig_CT,
        }

        if show_plots:
            try:
                for fig in figures.values():
                    fig.show()
            except Exception as e:
                print(f"Warning: Could not display plots automatically. Error: {e}")
                print("The figure objects are still available for manual display.")

        return figures

    def plot_scatter(self, x_data, y_data, pos, y_title):
        """This method plot a scatter(x,y) graph.

        Parameters
        ----------
        x_data : float
            X axis data.
        y_data : float
            Y axis data.
        pos : float
            Probe position.
        y_title : str
            Name of the Y axis

        Returns
        -------
        fig : object
            Scatter figure.
        """

        fig = go.Figure()
        for i in range(self.n_pad):
            fig.add_trace(
                go.Scatter(
                    x=x_data[i],
                    y=y_data[pos][:, i],
                    name=f"Pad{i}",
                )
            )
        fig.update_layout(
            xaxis_range=[
                np.array(x_data).min() * 1.1,
                360 - abs(np.array(x_data).min()),
            ]
        )
        fig.update_layout(plot_bgcolor="white")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angle (º)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(text=y_title, font=dict(family="Times New Roman", size=30)),
        )
        fig.update_layout(
            legend=dict(font=dict(family="Times New Roman", size=22, color="black"))
        )
        return fig

    def plot_contourP(self, x_data, y_data, z_data, z_title):
        """This method plot a contour(x,y,z) graph.

        Parameters
        ----------
        x_data : float
            X axis data.
        y_data : float
            Y axis data.
        z_data : float
            Z axis data.
        z_title : str
            Name of the z axis

        Returns
        -------
        fig : object
            Contour figure.
        """

        fig = go.Figure()
        max_val = z_data.max()
        for l in range(self.n_pad):
            fig.add_trace(
                go.Contour(
                    z=z_data[:, :, l],
                    x=x_data[l],
                    y=y_data,
                    zmin=0,
                    zmax=max_val,
                    ncontours=15,
                    colorscale="Viridis",
                    colorbar=dict(
                        title=z_title,
                        tickfont=dict(size=22),
                    ),
                )
            )
        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )
        fig.update_layout(
            xaxis_range=[
                np.array(x_data).min() * 1.1,
                360 - abs(np.array(x_data).min()),
            ]
        )
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angle (º)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(
                text="Pad Length (m)", font=dict(family="Times New Roman", size=30)
            ),
        )
        fig.update_layout(plot_bgcolor="white")
        return fig

    def plot_contourT(self, x_data, y_data, z_data, z_title):
        """This method plot a contour(x,y,z) graph.

        Parameters
        ----------
        x_data : float
            X axis data.
        y_data : float
            Y axis data.
        z_data : float
            Z axis data.
        z_title : str
            Name of the z axis

        Returns
        -------
        fig : object
            Contour figure.
        """

        fig = go.Figure()
        max_val = z_data.max()
        for l in range(self.n_pad):
            fig.add_trace(
                go.Contour(
                    z=z_data[:, :, l],
                    x=x_data[l],
                    y=y_data,
                    zmin=40,
                    zmax=max_val,
                    ncontours=25,
                    colorscale="Viridis",
                    colorbar=dict(
                        title=z_title,
                        tickfont=dict(size=22),
                    ),
                )
            )
        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )
        fig.update_layout(
            xaxis_range=[
                np.array(x_data).min() * 1.1,
                360 - abs(np.array(x_data).min()),
            ]
        )
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angle (º)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(
                text="Pad Length (m)", font=dict(family="Times New Roman", size=30)
            ),
        )
        fig.update_layout(plot_bgcolor="white")
        return fig

    def show_results(self):
        """Display tilting pad bearing calculation results in a formatted table.

        This method prints the main results from the tilting pad bearing analysis
        using PrettyTable, including operating conditions, field results,
        load information, and dynamic coefficients for each frequency.

        Parameters
        ----------
        None
            This method uses the bearing parameters and results stored as
            instance attributes.

        Returns
        -------
        None
            Results are printed to the console in a formatted table.
        """

        if self.frequency.size == 1:
            self._print_single_frequency_results(0)
        else:
            for i in range(self.frequency.size):
                self._print_single_frequency_results(i)

    def _print_single_frequency_results(self, freq_index):
        """Print results for a single frequency."""

        freq = self.frequency[freq_index]

        # Define a fixed width for all columns
        column_width = 20

        table = PrettyTable()
        table.field_names = ["Parameter", "Value", "Unit"]

        for field in table.field_names:
            table.max_width[field] = column_width
            table.min_width[field] = column_width

        # Set column alignment
        table.align["Parameter"] = "l"
        table.align["Value"] = "r"
        table.align["Unit"] = "c"

        # Operating conditions
        table.add_row(["Operating Speed", f"{freq * 30 / np.pi:12.1f}", "RPM"])
        table.add_row(["Equilibrium Type", f"{self.equilibrium_type:>12}", "-"])
        table.add_row(["Number of Pads", f"{self.n_pad:12d}", "-"])

        # Field results
        table.add_row(["Maximum Pressure", f"{self.maxP_list[freq_index]:12.4e}", "Pa"])
        table.add_row(
            [
                "Maximum Temperature",
                f"{self.maxT_list[freq_index]:12.2f}",
                "°C",
            ]
        )
        table.add_row(
            ["Minimum Film Thickness", f"{self.minH_list[freq_index]:12.4e}", "m"]
        )

        # Equilibrium results
        table.add_row(["Eccentricity", f"{self.ecc_list[freq_index]:12.4f}", "-"])
        table.add_row(
            [
                "Attitude Angle",
                f"{np.degrees(self.attitude_angle_list[freq_index]):12.2f}",
                "°",
            ]
        )

        # Load information
        table.add_row(
            ["Total Force X", f"{self.force_x_total_list[freq_index]:12.4e}", "N"]
        )
        table.add_row(
            ["Total Force Y", f"{self.force_y_total_list[freq_index]:12.4e}", "N"]
        )

        # Stiffness coefficients
        table.add_row(["kxx (Stiffness)", f"{self.kxx[freq_index]:12.4e}", "N/m"])
        table.add_row(["kxy (Stiffness)", f"{self.kxy[freq_index]:12.4e}", "N/m"])
        table.add_row(["kyx (Stiffness)", f"{self.kyx[freq_index]:12.4e}", "N/m"])
        table.add_row(["kyy (Stiffness)", f"{self.kyy[freq_index]:12.4e}", "N/m"])

        # Damping coefficients
        table.add_row(["cxx (Damping)", f"{self.cxx[freq_index]:12.4e}", "N*s/m"])
        table.add_row(["cxy (Damping)", f"{self.cxy[freq_index]:12.4e}", "N*s/m"])
        table.add_row(["cyx (Damping)", f"{self.cyx[freq_index]:12.4e}", "N*s/m"])
        table.add_row(["cyy (Damping)", f"{self.cyy[freq_index]:12.4e}", "N*s/m"])

        pad_table = PrettyTable()
        pad_table.align["Pad #"] = "c"

        # Check if moment data is available (match_eccentricity mode)
        if (
            self.momen_rot_list[freq_index] is not None
            and self.equilibrium_type == "match_eccentricity"
        ):
            pad_table.field_names = [
                "Pad #",
                "Moment [N·m]",
                "Angle [rad]",
                "Angle [°]",
            ]
            pad_table.align["Moment [N·m]"] = "r"
            pad_table.align["Angle [rad]"] = "r"
            pad_table.align["Angle [°]"] = "r"

            for i in range(self.n_pad):
                pad_table.add_row(
                    [
                        i + 1,
                        f"{self.momen_rot_list[freq_index][i]:12.4e}",
                        f"{self.psi_pad_list[freq_index][i]:12.4e}",
                        f"{np.degrees(self.psi_pad_list[freq_index][i]):12.4e}",
                    ]
                )
        else:
            # determine_eccentricity mode
            pad_table.field_names = ["Pad #", "Angle [rad]", "Angle [°]"]
            pad_table.align["Angle [rad]"] = "r"
            pad_table.align["Angle [°]"] = "r"

            for i in range(self.n_pad):
                pad_table.add_row(
                    [
                        i + 1,
                        f"{self.psi_pad_list[freq_index][i]:12.4e}",
                        f"{np.degrees(self.psi_pad_list[freq_index][i]):12.4e}",
                    ]
                )

        column_width = 14
        for field in pad_table.field_names:
            pad_table.max_width[field] = column_width
            pad_table.min_width[field] = column_width

        table_str = table.get_string()
        final_width = len(table_str.split("\n")[0])

        print("\n" + "=" * final_width)
        print(
            f"TILTING PAD BEARING RESULTS - {freq * 30 / np.pi:.1f} RPM".center(
                final_width
            )
        )
        print("=" * final_width)
        print(table)

        # Print pad rotation angles
        print("\n" + "-" * final_width)
        print("PAD ROTATION ANGLES".center(final_width))
        print("-" * final_width)
        print(pad_table)
        print("=" * final_width)

    def show_execution_time(self):
        """Display the simulation execution time.

        This method calculates and displays the total time spent during the
        complete bearing analysis execution, including all frequency calculations.

        Parameters
        ----------
        None
            This method uses the initial_time and final_time attributes
            stored during the simulation execution.

        Returns
        -------
        float
            Total simulation time in seconds. Returns None if simulation
            hasn't been executed yet.
        """
        if hasattr(self, "initial_time") and hasattr(self, "final_time"):
            total_time = self.final_time - self.initial_time
            print(f"Execution time: {total_time:.2f} seconds")
        else:
            print("Simulation hasn't been executed yet.")

    def show_coefficients_comparison(self):
        """Display dynamic coefficients comparison table.

        This method creates and displays a formatted table comparing dynamic
        coefficients (stiffness and damping) across different frequencies.

        Parameters
        ----------
        None
            This method uses the frequency array and coefficients stored as
            instance attributes.

        Returns
        -------
        None
            Results are printed to the console in a formatted table.
        """

        freq_rpm = np.atleast_1d(self.frequency).astype(float) * 30.0 / np.pi

        table = PrettyTable()
        headers = [
            "Frequency [RPM]",
            "kxx [N/m]",
            "kxy [N/m]",
            "kyx [N/m]",
            "kyy [N/m]",
            "cxx [N*s/m]",
            "cxy [N*s/m]",
            "cyx [N*s/m]",
            "cyy [N*s/m]",
        ]
        table.field_names = headers

        for i in range(len(freq_rpm)):
            row = [
                f"{freq_rpm[i]:.1f}",
                f"{self.kxx[i]:.4e}",
                f"{self.kxy[i]:.4e}",
                f"{self.kyx[i]:.4e}",
                f"{self.kyy[i]:.4e}",
                f"{self.cxx[i]:.4e}",
                f"{self.cxy[i]:.4e}",
                f"{self.cyx[i]:.4e}",
                f"{self.cyy[i]:.4e}",
            ]
            table.add_row(row)

        # Table width
        desired_width = 25

        table.max_width = desired_width
        table.min_width = desired_width

        table_str = table.get_string()
        table_lines = table_str.split("\n")
        actual_width = len(table_lines[0])

        print("\n" + "=" * actual_width)
        print("DYNAMIC COEFFICIENTS COMPARISON TABLE".center(actual_width))
        print("=" * actual_width)
        print(table)
        print("=" * actual_width)

    def record_optimization_residual(
        self, residual_value: float, iteration: int | None = None
    ) -> None:
        """
        Store the residual value for the current frequency.

        - If 'iteration' is provided, the value is placed at that index.
        - If 'iteration' is None, the value is appended.

        Notes
        -----
        Requires 'self._current_freq_index' to be set (done in the frequency loop).
        """
        idx = getattr(self, "_current_freq_index", None)
        if idx is None:
            return

        if idx not in self.optimization_history:
            self.optimization_history[idx] = []

        if iteration is None:
            self.optimization_history[idx].append(residual_value)
        else:
            if len(self.optimization_history[idx]) <= iteration:
                self.optimization_history[idx] += [None] * (
                    iteration + 1 - len(self.optimization_history[idx])
                )
            self.optimization_history[idx][iteration] = residual_value
        
        if self.equilibrium_type == "determine_eccentricity":
            print(f"[determine_ecc] freq_idx={idx} iter={iteration or len(self.optimization_history[idx])} residual={residual_value:.3e}")

    def show_optimization_convergence(
        self, by: str = "index", show_plots: bool = False
    ) -> None:
        """
        Display the optimization residuals per iteration for each processed frequency.

        Parameters
        ----------
        by : str
            'index' -> show frequencies by their index (default)
            'value' -> show frequencies by their value (as stored in self.frequency)
        show_plots : bool
            Whether to show the convergence plot. Default is False.

        Notes
        -----
        Requires 'self.optimization_history' to be populated during the solve.
        """
        if not hasattr(self, "optimization_history") or not self.optimization_history:
            print("No residual history available. Run the analysis first.")
            return

        for i, res_list in self.optimization_history.items():
            if not res_list:
                continue

            freq = self.frequency[i]
            rpm = freq * 30 / np.pi

            # Check equilibrium type
            if self.equilibrium_type == "match_eccentricity":
                # Separate residuals by pad
                n_pads = self.n_pad

                # Split residuals by pad (estimate iterations per pad)
                total_iters = len(res_list)
                approx_iters_per_pad = total_iters // n_pads

                # Create table
                table = PrettyTable()
                table.field_names = ["Pad", "Iterations", "Final Residual [N]"]

                pad_residuals = []
                for pad_idx in range(n_pads):
                    start_idx = pad_idx * approx_iters_per_pad
                    end_idx = (
                        (pad_idx + 1) * approx_iters_per_pad
                        if pad_idx < n_pads - 1
                        else total_iters
                    )
                    pad_res = [r for r in res_list[start_idx:end_idx] if r is not None]

                    if pad_res:
                        final_res = pad_res[-1]
                        table.add_row([pad_idx + 1, len(pad_res), f"{final_res:.6f}"])
                        pad_residuals.append((pad_idx + 1, pad_res))

                desired_width = 25
                table.max_width = desired_width
                table.min_width = desired_width

                table_str = table.get_string()
                table_lines = table_str.split("\n")
                actual_width = len(table_lines[0])

                print("\n" + "=" * actual_width)
                print(f"OPTIMIZATION CONVERGENCE - {rpm:.1f} RPM".center(actual_width))
                print("=" * actual_width)
                print(table)
                print("=" * actual_width)

                # Display plot if requested - one subplot per pad
                if show_plots:

                    n_rows = (n_pads + 1) // 2  # 2 columns
                    fig = make_subplots(
                        rows=n_rows,
                        cols=2,
                        subplot_titles=[f"Pad {p[0]}" for p in pad_residuals],
                    )

                    for idx, (pad_num, pad_res) in enumerate(pad_residuals):
                        row = (idx // 2) + 1
                        col = (idx % 2) + 1

                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(pad_res))),
                                y=pad_res,
                                mode="lines+markers",
                                name=f"Pad {pad_num}",
                                line=dict(width=2),
                                marker=dict(size=4),
                            ),
                            row=row,
                            col=col,
                        )

                        fig.update_xaxes(title_text="Iteration", row=row, col=col)
                        fig.update_yaxes(title_text="Residual [N]", row=row, col=col)

                    fig.update_layout(
                        title=f"Optimization Convergence by Pad - {rpm:.1f} RPM",
                        template="ross",
                        showlegend=False,
                        height=300 * n_rows,
                    )
                    fig.show()

            else:  # determine_eccentricity - single global optimization
                table = PrettyTable()
                table.field_names = ["Iteration", "Residual [N]"]

                for it, res in enumerate(res_list):
                    if res is not None:
                        table.add_row([it, f"{res:.6f}"])

                desired_width = 25
                table.max_width = desired_width
                table.min_width = desired_width

                table_str = table.get_string()
                table_lines = table_str.split("\n")
                actual_width = len(table_lines[0])

                print("\n" + "=" * actual_width)
                print(f"OPTIMIZATION CONVERGENCE - {rpm:.1f} RPM".center(actual_width))
                print("=" * actual_width)
                print(table)
                print("=" * actual_width)

                # Display plot if requested
                if show_plots:
                    iterations = list(range(len(res_list)))
                    residuals = [res if res is not None else 0 for res in res_list]

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=iterations,
                            y=residuals,
                            mode="lines+markers",
                            name=f"Convergence - {rpm:.1f} RPM",
                            line=dict(width=2),
                            marker=dict(size=6),
                        )
                    )
                    fig.update_layout(
                        title=f"Global Optimization Convergence - {rpm:.1f} RPM",
                        xaxis_title="Iteration",
                        yaxis_title="Residual [N]",
                        template="ross",
                    )
                    fig.show()


def tilting_pad_example():
    """Create an example of a tilting pad bearing with Thermo-Hydro-Dynamic effects.

    This function creates and returns a TiltingPad bearing instance with predefined
    parameters for demonstration purposes. The bearing is configured with 5 pads
    and operates at two different frequencies.

    Returns
    -------
    bearing : TiltingPad
        A configured tilting pad bearing instance ready for analysis.

    Examples
    --------
    >>> from ross.bearings.tilting_pad import tilting_pad_example
    >>> bearing = tilting_pad_example()

    Notes
    -----
    This example uses the following configuration:
    - 5 tilting pads arranged at 18°, 90°, 162°, 234°, and 306°
    - Journal diameter: 101.6 mm
    - Radial clearance: 74.9 μm
    - Pad thickness: 12.7 mm
    - Pad arc: 60° per pad
    - Pad axial length: 50.8 mm
    - ISOVG32 lubricant at 40°C
    - Operating frequencies: 3000 RPM
    - Pre-load factor: 0.5 for all pads
    - Pivot offset: 0.5 (centered) for all pads

    The bearing is configured for "match_eccentricity" equilibrium type,
    which automatically calculates the equilibrium position based on the
    specified eccentricity and attitude angle.
    """

    bearing = TiltingPad(
        n=1,
        frequency=Q_([3000], "RPM"),
        equilibrium_type="match_eccentricity",
        model_type="thermo_hydro_dynamic",
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
        pad_arc=Q_([60, 60, 60, 60, 60], "deg"),
        pad_axial_length=Q_([50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3], "m"),
        pre_load=[0.5, 0.5, 0.5, 0.5, 0.5],
        offset=[0.5, 0.5, 0.5, 0.5, 0.5],
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        nx=30,
        nz=30,
        eccentricity=0.483,
        attitude_angle=Q_(267.5, "deg"),
    )

    return bearing
