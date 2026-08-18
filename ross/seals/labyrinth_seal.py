import numpy as np
from scipy.linalg import lu_factor, lu_solve
from numpy.linalg import cond
from warnings import warn
import multiprocessing
from ross import SealElement
from ross.units import check_units, Q_
from ross.seals.gas_model import extract_gas_properties, IdealGas, RealGas
import plotly.graph_objects as go
import ccp

__all__ = ["LabyrinthSeal"]


class LabyrinthSeal(SealElement):
    """Labyrinth seal - Compressible flow model with rotordynamic coefficients.

    This class provides a **comprehensive analytical model** for labyrinth seals
    based on compressible gas flow through multiple throttling stages (teeth). The
    model calculates leakage rates and dynamic coefficients for rotordynamic analysis.

    **Theoretical Approach:**

    The model solves the **1D compressible flow problem** through a series of teeth using:

    1. **Mass Flow Calculation**:
       - Iterative solution for mass flow rate through multiple throttling stages
       - Accounts for choked flow conditions at each tooth
       - Uses discharge coefficients based on tooth geometry
       - Isentropic relations for pressure drops across teeth
       - Carry-over factor (ν) for flow momentum between cavities

    2. **Pressure Distribution**:
       - Solves for static pressure at each cavity using regula falsi method
       - Handles both choked and unchoked flow conditions
       - Critical pressure ratio check at each throttle
       - Pressure balance ensures outlet pressure match

    3. **Velocity Field (Swirl)**:
       - Tangential velocity calculated at each cavity
       - Accounts for inlet pre-swirl conditions
       - Rotor and stator shear stress effects (friction factors)
       - Reynolds number-dependent shear coefficients
       - Jenny and Kanki parameters for improved tangential momentum (optional)

    4. **Dynamic Coefficients** (stiffness and damping):
       - Perturbation method applied to continuity and momentum equations
       - Small perturbations in radial displacement and clearance
       - Linearized system of equations solved using LU decomposition
       - Cross-coupled stiffness terms capture destabilizing forces
       - Frequency-dependent coefficients for each operating speed

    Parameters
    ----------
    n : int
        Node in which the seal will be located.
    shaft_diameter : float, pint.Quantity
        Diameter of the shaft (m).
    radial_clearance : float, pint.Quantity
        Nominal radial clearance (m).
    n_teeth : int
        Number of teeth (throttlings). Needs to be <= 30.
    pitch : float, pint.Quantity
        Seal pitch (length of land) or axial cavity length (m).
    tooth_height : float, pint.Quantity
        Height of seal strip (m).
    tooth_width : float, pint.Quantity
        Thickness of throttle (tip-width) (m), used in mass flow calculation.
    seal_type : str
        Indicates where labyrinth teeth are located.
        Specify 'rotor' if teeth are on rotor only.
        Specify 'stator' if teeth are on stator only.
        Specify 'inter' for interlocking type labyrinths.
    inlet_pressure : float
        Inlet pressure (Pa).
    outlet_pressure : float
        Outlet pressure (Pa).
    inlet_temperature : float
        Inlet temperature (deg K).
    frequency : float, pint.Quantity
        Shaft rotational speed (rad/s).
    preswirl : float
        Inlet swirl velocity ratio. Positive values for swirl with shaft rotation
        and negative values for swirl against shaft rotations.
    gas_composition : dict, optional
        Gas composition as a dictionary {component: molar_fraction}.
        If gas_composition is None, provide molar_mass, gamma, reference_temperatures, and reference_viscosities parameters.
        Default is None.
    gas_model : str, optional
        Thermodynamic model used by the internal flow solver.
        Specify "ideal" for the perfect-gas model (Z = 1, constant gamma); results
        are identical to previous versions.
        Specify "real" for the equation-of-state (real-gas) model, which evaluates
        density, temperature, sound speed, enthalpy and choking along the inlet
        isentrope from a table built once at construction. Requires gas_composition.
        Default is "ideal".
    molar_mass : float, pint.Quantity, optional
        Molecular mass (kg/kgmol). For Air: molar_mass=28.97 kg/kgmol.
        Required if gas_composition is None. Default is None.
    gamma : float, optional
        Ratio of specific heats. Required if gas_composition is None.
        Default is None.
    reference_temperatures : list of float, optional
        Temperature at states: [T_state1, T_state2] (deg K).
        Required if gas_composition is None.
        Default is None.
    reference_viscosities : list of float, optional
        Dynamic viscosity at states: [mu_state1, mu_state2] (kg/(m·s)).
        Required if gas_composition is None.
        Default is None.
    use_jenny_kanki : bool, optional
        If True, use the tangential momentum parameters introduced by Jenny
        and Kanki in the swirl velocity calculation.
        Default is False.
    print_results : bool, optional
        If True, print results to console.
        Default is False.
    tag : str, optional
        A tag to name the element.
        Default is None.
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is "#77ACA2".

    Examples
    --------
    >>> from ross.seals.labyrinth_seal import LabyrinthSeal
    >>> from ross.units import Q_
    >>> seal = LabyrinthSeal(
    ...     n=0,
    ...     shaft_diameter=Q_(145, "mm"),
    ...     radial_clearance=Q_(0.3, "mm"),
    ...     n_teeth=16,
    ...     pitch=Q_(3.175, "mm"),
    ...     tooth_height=Q_(3.175, "mm"),
    ...     tooth_width=Q_(0.1524, "mm"),
    ...     seal_type="inter",
    ...     inlet_pressure=308000,
    ...     outlet_pressure=94300,
    ...     inlet_temperature=283.15,
    ...     frequency=Q_([5000, 8000, 11000], "RPM"),
    ...     preswirl=0.98,
    ...     gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},
    ... )
    """

    @check_units
    def __init__(
        self,
        n,
        shaft_diameter,
        radial_clearance,
        n_teeth,
        pitch,
        tooth_height,
        tooth_width,
        seal_type,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        frequency,
        preswirl,
        gas_composition=None,
        gas_model="ideal",
        molar_mass=None,
        gamma=None,
        reference_temperatures=None,
        reference_viscosities=None,
        use_jenny_kanki=False,
        print_results=False,
        **kwargs,
    ):
        self.print_results = print_results
        self.gas_composition = gas_composition
        self.gas_model = gas_model

        if self.gas_composition is not None:
            state_in, molar_mass, gamma, R = extract_gas_properties(
                self.gas_composition, inlet_pressure, inlet_temperature
            )
            state_out = ccp.State(
                p=outlet_pressure, h=state_in.h(), fluid=self.gas_composition
            )
        else:
            R = (
                8314.0 / molar_mass
            )  # Universal gas constant (J/(kmol·K)) over molar_mass mass.

        self.R = R
        self.molar_mass = molar_mass
        self.gamma = gamma

        if self.gas_model == "real":
            if self.gas_composition is None:
                raise ValueError(
                    "gas_model='real' requires gas_composition to query the "
                    "equation of state."
                )
            self.gas = RealGas(
                self.R,
                self.gamma,
                self.gas_composition,
                inlet_pressure,
                outlet_pressure=outlet_pressure,
                inlet_temperature=inlet_temperature,
            )
            self._real_gas = True
        elif self.gas_model == "ideal":
            self.gas = IdealGas(self.R, self.gamma)
            self._real_gas = False
        else:
            raise ValueError(
                f"Invalid gas_model {self.gas_model!r}; expected 'ideal' or 'real'."
            )

        if reference_temperatures is None:
            # reference_temperatures: Temperature at state 1 e 2 (deg K)
            reference_temperatures = [state_in.T().m, state_out.T().m]
        if reference_viscosities is None:
            # reference_viscosities: Dynamic viscosity at state 1 e 2 (kg/(m s))
            reference_viscosities = [state_in.viscosity().m, state_out.viscosity().m]

        self.reference_temperatures = reference_temperatures
        self.reference_viscosities = reference_viscosities

        self.n = n
        self.inlet_pressure = inlet_pressure
        self.outlet_pressure = outlet_pressure
        self.inlet_temperature = inlet_temperature
        self.preswirl = preswirl
        self.n_teeth = n_teeth
        self.shaft_diameter = shaft_diameter
        self._shaft_radius = shaft_diameter / 2
        self.radial_clearance = radial_clearance
        self.pitch = pitch
        self.tooth_height = tooth_height
        self.tooth_width = tooth_width
        self.seal_type = seal_type

        self.use_jenny_kanki = use_jenny_kanki

        self._max_stations = 61
        self.pitch = np.full(self._max_stations, pitch)
        self.radial_clearance = np.full(self._max_stations, radial_clearance)
        self.tooth_height = np.full(self._max_stations, tooth_height)
        self.pr = np.zeros(self._max_stations)
        self.tooth_width = np.full(self._max_stations, tooth_width)

        self.z = np.zeros(self._max_stations)
        for i in range(0, self.n_teeth + 1):
            self.z[i] = i * self.pitch[i]

        self.p = np.zeros(self._max_stations)
        self.v = np.zeros(self._max_stations)
        self.w = np.zeros(self._max_stations)
        self.p1 = np.zeros(self._max_stations)
        self.v1 = np.zeros(self._max_stations)
        self.t = np.zeros(self._max_stations)
        self.rho = np.zeros(self._max_stations)
        self.taus = np.zeros(self._max_stations)
        self.taur = np.zeros(self._max_stations)
        self.gm = np.zeros((1000, 500))
        self.rhs = np.zeros((1000, 2))
        self.cg = np.zeros((9, self._max_stations))
        self.cx = np.zeros((8, self._max_stations))
        self.vin = np.zeros(self._max_stations)
        self.vout = np.zeros(self._max_stations)
        self.kout = np.zeros(self._max_stations)

        coefficients_dict = {}
        if kwargs.get("kxx") is None:
            # Use multiprocessing only when beneficial (>4 frequencies)
            # For small workloads, sequential execution avoids process spawn overhead
            if len(frequency) > 4:
                with multiprocessing.Pool() as pool:
                    results = pool.map(self.run, frequency)
            else:
                results = [self.run(freq) for freq in frequency]

            self.p = [r["pressure"] for r in results]

            coefficients_dict = {
                c: [k[c] for k in results]
                for c in results[0].keys()
                if c not in ["pressure", "pert_rcond", "pert_condition_number"]
            }
            self.pert_rcond = [r["pert_rcond"] for r in results]
            self.pert_condition_number = [r["pert_condition_number"] for r in results]

        super().__init__(
            self.n,
            frequency=frequency,
            **coefficients_dict,
            **kwargs,
        )

    def _solve_choked_flow_functions(self):
        error = 10000
        tol = 1 * 10**-7
        guess_low = 0.001
        guess = 0.8
        guess_high = 0.99
        n = 0
        while n < self.n_teeth:
            r = guess
            deriv_num = -2 * (n + 1) + 2 * r * np.log(r) + 1 / r - r
            deriv_den = ((1 - r**2) ** 0.5) * ((n - np.log(r)) ** 1.5)
            deriv = deriv_num / deriv_den
            error = -deriv
            while abs(error) > tol:
                if error < 0:
                    guess_low = guess
                    guess = (guess + guess_high) / 2
                if error > 0:
                    guess_high = guess
                    guess = (guess + guess_low) / 2
                r = guess
                deriv_num = -2 * (n + 1) + 2 * r * np.log(r) + 1 / r - r
                deriv_den = ((1 - r**2) ** 0.5) * ((n - np.log(r)) ** 1.5)
                deriv = deriv_num / deriv_den
                error = -deriv
            self.r_choke[n] = guess
            self.flow_function_choke[n] = (
                (1 - self.r_choke[n] ** 2) / ((n + 1) - np.log(self.r_choke[n]))
            ) ** 0.5
            n += 1
            error = 10000
            guess_low = 0.001
            guess = 0.8
            guess_high = 0.99
        if self.overall_pressure_ratio < self.r_choke[self.n_teeth - 1]:
            self.flow_function_last_tooth = self.flow_function_choke[self.n_teeth - 1]
        else:
            self.flow_function_last_tooth = (
                (1 - self.overall_pressure_ratio**2)
                / (self.n_teeth - np.log(self.overall_pressure_ratio))
            ) ** 0.5

    def _reset_perturbation_arrays(self):
        """Reset perturbation and velocity-coupling workspace arrays.
        Zeros ``gm``, ``rhs``, ``cg``, ``cx``, ``taur``, and ``taus`` before
        each base-flow and perturbation solve in ``_reset_state()``.
        """
        self.gm.fill(0)
        self.rhs.fill(0)
        self.cg.fill(0)
        self.cx.fill(0)
        self.taur.fill(0)
        self.taus.fill(0)

    def _reset_state(self):
        self.perturbation_eccentricity = 0.6
        self.pert_amplitude_direct = (
            self.perturbation_eccentricity * self.radial_clearance[0]
        )
        self.pert_amplitude_cross = (
            self.perturbation_eccentricity * self.radial_clearance[0]
        )
        self.n_cavities = self.n_teeth - 1
        self.n_stations = self.n_teeth + 1

        self._reset_perturbation_arrays()

        self.ndof = 8 * self.n_cavities
        self._band_width = 33
        self._band_center = 17

        for i in range(0, self.n_stations):
            self.w[i] = 0
            self.pr[i] = 0
            self.p[i] = 0
            self.v[i] = 0
            self.p1[i] = 0
            self.v1[i] = 0
            self.rho[i] = 0
            self.t[i] = self.inlet_temperature

        self.overall_pressure_ratio = self.outlet_pressure / self.inlet_pressure
        self.omega = self.frequency

    def _vermes_leakage(self):
        width_to_clearance_ratio = self.tooth_width[0] / self.radial_clearance[0]
        self.discharge_coefficient = (
            0.67675
            - (0.08519 * width_to_clearance_ratio)
            + (0.0878 * (width_to_clearance_ratio**2))
            - (0.01819 * (width_to_clearance_ratio**3))
            + (0.00111 * (width_to_clearance_ratio**4))
        )
        self.carryover_factor = 8.52 / (
            ((self.pitch[0] - self.tooth_width[0]) / self.radial_clearance[0]) + 7.23
        )
        if self.seal_type == "inter":
            self.carryover_factor = 0
        if self.carryover_factor >= 1:
            raise ValueError(
                f"The Vermes carry-over factor is {self.carryover_factor:.4f}; "
                "values >= 1 are not physical. Check the seal geometry "
                "(pitch, tooth_width, radial_clearance)."
            )

        carryover_velocity_ratio = 1 / (1 - self.carryover_factor) ** 0.5
        self.r_choke = [0] * self._max_stations
        self.flow_function_choke = [0] * self._max_stations
        self._solve_choked_flow_functions()
        self.vermes_flow_factor = (
            1.014
            * self.discharge_coefficient
            * carryover_velocity_ratio
            * self.flow_function_last_tooth
        )

        if self.seal_type == "inter":
            self.vermes_flow_factor = self.vermes_flow_factor / 1.014
        self.mdot_vermes = (
            self.vermes_flow_factor
            * self.inlet_pressure
            * self.radial_clearance[0]
            / (self.R * self.inlet_temperature) ** 0.5
        )
        leakage_vermes = (
            self.mdot_vermes
            * 2
            * np.pi
            * (self._shaft_radius + 0.5 * self.radial_clearance[0])
        )
        if self.print_results:
            print(f"{'   Leakage':<40} {leakage_vermes:>15.8f} kg/s \n \n")
        self.mdot = self.mdot_vermes

    def _solve_pressure_distribution(self):
        prgs = [0] * 3
        fpr = [0] * 3
        gam1 = 1 / self.gamma
        gam2 = (self.gamma - 1) / self.gamma
        gam3 = 2 / gam2
        gam4 = self.R * gam3
        gam5 = 1 / gam2
        gam6 = 1 / (self.gamma - 1)
        gam7 = 2 / (self.gamma + 1)
        gam8 = self.gamma * 2 / (self.gamma + 1)

        tol1 = 1 * 10 ** (-8)
        itmx1 = 100
        ndex1 = 0

        tol_outlet_pressure = 0.00001
        tol_choked = 0.005

        tol_p = 1 * 10 ** (-4)
        a2998 = True

        while True:
            asaida = True
            if a2998:
                mdot_high = self.mdot * 5
                mdot_low = 0
                a2998 = False
            if ndex1 < 1:
                self.w[0] = 0
                self.p[0] = self.inlet_pressure
                self.rho[0] = self.gas.inlet_density(self.p[0], self.t[0])
                prold = 1 * 10 ** (10)
                if self._real_gas:
                    chok2 = self.gas.critical_pr(
                        self.p[self.n_teeth - 1],
                        self.w[self.n_teeth - 1],
                        self.carryover_factor,
                        self.t[self.n_teeth - 1],
                    )
                else:
                    chok1 = gam7 + (
                        self.carryover_factor
                        * self.w[self.n_teeth - 1]
                        * self.w[self.n_teeth - 1]
                        / (gam4 * self.t[self.n_teeth - 1])
                    )
                    chok2 = chok1**gam5
            for i in range(1, self.n_teeth + 1):
                prgs[0] = chok2
                prgs[1] = 0.9999999
                for j in range(0, 2):
                    fpr[j] = self.gas.throttle_mass_flux(
                        self.discharge_coefficient,
                        self.radial_clearance[i - 1],
                        self.p[i - 1],
                        prgs[j],
                        self.rho[i - 1],
                        self.t[i - 1],
                        self.w[i - 1],
                        self.carryover_factor,
                    )
                    fpr[j] = self.mdot - fpr[j]
                if fpr[0] > 0:
                    fpr[0] = 0
                for itn in range(0, itmx1):
                    prgs[2] = (prgs[0] * fpr[1] - prgs[1] * fpr[0]) / (fpr[1] - fpr[0])
                    fpr[2] = self.gas.throttle_mass_flux(
                        self.discharge_coefficient,
                        self.radial_clearance[i - 1],
                        self.p[i - 1],
                        prgs[2],
                        self.rho[i - 1],
                        self.t[i - 1],
                        self.w[i - 1],
                        self.carryover_factor,
                    )
                    a2001 = True
                    if prgs[2] <= chok2:
                        a2001 = False
                        error_outlet_pressure = 0
                        break
                    fpr[2] = self.mdot - fpr[2]

                    if fpr[2] * fpr[0] < 0:
                        prgs[1] = prgs[2]
                        fpr[1] = fpr[2]
                    elif fpr[2] * fpr[0] == 0:
                        if fpr[0] == 0:
                            prgs[2] = prgs[0]
                            fpr[2] = fpr[0]
                            prgs[1] = prgs[0]
                            fpr[1] = fpr[0]
                        else:
                            prgs[1] = prgs[2]
                            fpr[1] = fpr[2]
                            prgs[0] = prgs[2]
                            fpr[0] = fpr[2]
                            break
                    elif fpr[2] * fpr[0] > 0:
                        prgs[0] = prgs[2]
                        fpr[0] = fpr[2]
                    if abs((prgs[2] - prold) / prgs[2]) <= tol1:
                        break
                    prold = prgs[2]

                if not a2001:
                    break
                if abs(fpr[2]) > tol_p:
                    warn(f"Pressure Convergence Error at Station {i}")

                self.pr[i - 1] = prgs[2]
                self.p[i] = self.pr[i - 1] * self.p[i - 1]
                self.w[i] = self.gas.throat_velocity(
                    self.mdot,
                    self.discharge_coefficient,
                    self.radial_clearance[i - 1],
                    self.p[i - 1],
                    self.pr[i - 1],
                    self.t[i - 1],
                )
                self.rho[i] = self.gas.density_isentropic(
                    self.p[i - 1], self.pr[i - 1], self.rho[i - 1]
                )
                self.t[i] = self.gas.temperature_isentropic(
                    self.p[i - 1], self.pr[i - 1], self.t[i - 1]
                )

            if a2001:
                i = self.n_stations - 1
                if self._real_gas:
                    chock2 = self.gas.critical_pr(
                        self.p[i - 1],
                        self.w[i - 1],
                        self.carryover_factor,
                        self.t[i - 1],
                    )
                else:
                    chock1 = gam7 + (
                        self.carryover_factor
                        * self.w[i - 1]
                        * self.w[i - 1]
                        / (gam4 * self.t[i - 1])
                    )
                    chock2 = chock1**gam5
                error_outlet_pressure = (
                    self.p[self.n_stations - 1] - self.outlet_pressure
                ) / self.outlet_pressure
                if ndex1 == 1:
                    break
            if (
                abs(error_outlet_pressure) >= tol_outlet_pressure
                and abs(self.pr[self.n_stations - 2] - chock2) / chock2 > tol_choked
            ) or not a2001:
                if error_outlet_pressure < 0 or not a2001:
                    mdot_tmp = self.mdot
                    self.mdot = (mdot_low + self.mdot) / 2
                    mdot_high = mdot_tmp
                    if (self.mdot - mdot_tmp) / self.mdot == 0:
                        if self.print_results:
                            print("Reset iteration")
                        ndex1 = 2
                        a2998 = True
                elif error_outlet_pressure >= 0:
                    mdot_tmp = self.mdot
                    self.mdot = (mdot_high + self.mdot) / 2
                    mdot_low = mdot_tmp
                    if (self.mdot - mdot_tmp) / self.mdot == 0:
                        if self.print_results:
                            print("Reset iteration")
                        ndex1 = 2
                        a2998 = True
                asaida = False
            if asaida:
                break

        i = self.n_stations - 1
        if self._real_gas:
            chok2 = self.gas.critical_pr(
                self.p[i - 1], self.w[i - 1], self.carryover_factor, self.t[i - 1]
            )
        else:
            chok1 = gam7 + (
                self.carryover_factor
                * self.w[i - 1]
                * self.w[i - 1]
                / (gam4 * self.t[i - 1])
            )
            chok2 = chok1**gam5

        if ndex1 != 1:
            leakage = (
                self.mdot
                * 2
                * np.pi
                * (self._shaft_radius + 0.5 * self.radial_clearance[0])
            )

        if (
            ndex1 != 1
            and abs(self.pr[self.n_stations - 2] - chok2) / chok2 <= tol_choked
        ):
            warn("Flow Chocked in Last Thottle")
            ndex1 = 1
        if (self.pr[self.n_teeth - 1]) > 1:
            raise ValueError("Error in Leakage Calculation")

        if self.print_results:
            print(f"{'   Leakage':<40} {leakage:>15.8f} kg/s \n")

    def _solve_swirl_velocities_jenny_kanki(self):
        vgs = np.zeros(3)
        fv = np.zeros(3)
        rov = np.zeros(3)
        tr = np.zeros(3)
        ts = np.zeros(3)

        if self.omega == 0 and self.inlet_swirl_velocity == 0:
            return

        jenny_kanki_factor = 0
        if self.seal_type == "stator":
            jenny_kanki_factor = 0.15
        elif self.seal_type == "rotor":
            jenny_kanki_factor = 0.35
        elif self.seal_type == "inter":
            jenny_kanki_factor = 0.90
        else:
            raise ValueError("Improper selection of labyrinth type.")

        friction_exponent_rotor = -0.25
        friction_exponent_stator = -0.25
        friction_coefficient_rotor = 0.079
        friction_coefficient_stator = 0.079

        if self.seal_type == "inter":
            area_ratio_rotor = (1 * self.tooth_height[0] + self.pitch[0]) / self.pitch[
                0
            ]
            area_ratio_stator = (1 * self.tooth_height[0] + self.pitch[0]) / self.pitch[
                0
            ]
        elif self.seal_type == "stator":
            area_ratio_stator = (2 * self.tooth_height[0] + self.pitch[0]) / self.pitch[
                0
            ]
            area_ratio_rotor = 1
        else:
            area_ratio_rotor = (2 * self.tooth_height[0] + self.pitch[0]) / self.pitch[
                0
            ]
            area_ratio_stator = 1

        hydraulic_diameter = (
            2
            * (self.radial_clearance[0] + self.tooth_height[0])
            * self.pitch[0]
            / (self.radial_clearance[0] + self.tooth_height[0] + self.pitch[0])
        )
        area = (self.tooth_height[0] + self.radial_clearance[0]) * self.pitch[0]

        self.v[0] = self.inlet_swirl_velocity
        self.vin[0] = self.inlet_swirl_velocity
        self.vout[0] = self.inlet_swirl_velocity
        self.taur[0] = 0
        self.taus[0] = 0
        itmx2 = 40
        tol2 = 1 * 10 ** (-8)
        vold = 1 * 10 ** (10)

        phi1 = (self.reference_temperatures[0] ** 1.5) / self.reference_viscosities[0]
        phi2 = (self.reference_temperatures[1] ** 1.5) / self.reference_viscosities[1]
        sutherland_b = (
            self.reference_temperatures[1] - self.reference_temperatures[0]
        ) / (phi2 - phi1)
        sutherland_s = (sutherland_b * phi1) - self.reference_temperatures[0]
        for i in range(1, self.n_teeth):
            self.vin[i] = self.vout[i - 1]
            mu = sutherland_b * (self.t[i]) ** 0.5 / (1 + (sutherland_s / self.t[i]))
            self.nu = mu / self.rho[i]
            vgs[1] = self.gas.sound_speed(self.p[i], self.t[i])
            vgs[0] = -vgs[1]

            rov[0] = (self._shaft_radius * self.omega) - vgs[0]
            rov[1] = (self._shaft_radius * self.omega) - vgs[1]

            for j in range(0, 2):
                tr[j] = (
                    0.5
                    * self.rho[i]
                    * rov[j]
                    * rov[j]
                    * friction_coefficient_rotor
                    * (
                        (abs(rov[j]) * hydraulic_diameter / self.nu)
                        ** friction_exponent_rotor
                    )
                    * np.copysign(1, rov[j])
                )
                ts[j] = (
                    0.5
                    * self.rho[i]
                    * vgs[j]
                    * vgs[j]
                    * friction_coefficient_stator
                    * (
                        (abs(vgs[j]) * hydraulic_diameter / self.nu)
                        ** friction_exponent_stator
                    )
                    * np.copysign(1, vgs[j])
                )
                fv[j] = (self.mdot * jenny_kanki_factor * (vgs[j] - self.vin[i])) - (
                    self.pitch[0]
                    * (tr[j] * area_ratio_rotor - ts[j] * area_ratio_stator)
                )

            for itn2 in range(0, itmx2):
                vgs[2] = (vgs[0] * fv[1] - vgs[1] * fv[0]) / (fv[1] - fv[0])
                rov[2] = (self._shaft_radius * self.omega) - vgs[2]
                tr[2] = (
                    0.5
                    * self.rho[i]
                    * rov[2]
                    * rov[2]
                    * friction_coefficient_rotor
                    * (
                        (abs(rov[2]) * hydraulic_diameter / self.nu)
                        ** friction_exponent_rotor
                    )
                    * np.copysign(1, rov[2])
                )
                ts[2] = (
                    0.5
                    * self.rho[i]
                    * vgs[2]
                    * vgs[2]
                    * friction_coefficient_stator
                    * (
                        (abs(vgs[2]) * hydraulic_diameter / self.nu)
                        ** friction_exponent_stator
                    )
                    * np.copysign(1, vgs[2])
                )
                fv[2] = (self.mdot * (vgs[2] - self.vin[i])) - (
                    self.pitch[0]
                    * (tr[2] * area_ratio_rotor - ts[2] * area_ratio_stator)
                )

                if fv[2] * fv[0] < 0:
                    vgs[1] = vgs[2]
                    fv[1] = fv[2]
                    rov[1] = rov[2]
                    tr[1] = tr[2]
                    ts[1] = ts[2]

                    if abs((vgs[2] - vold) / vgs[2]) > tol2:
                        vold = vgs[2]
                    else:
                        break

                elif fv[2] * fv[0] == 0:
                    if fv[1] == 0:
                        vgs[1] = vgs[0]
                        fv[1] = fv[0]
                        vgs[2] = vgs[0]
                        fv[2] = fv[0]
                        rov[1] = rov[0]
                        tr[1] = tr[0]
                        ts[1] = ts[0]
                        rov[2] = rov[0]
                        tr[2] = tr[0]
                        ts[2] = ts[0]

                    else:
                        vgs[1] = vgs[2]
                        fv[1] = fv[2]
                        vgs[0] = vgs[2]
                        fv[0] = fv[2]
                        rov[1] = rov[2]
                        tr[1] = tr[2]
                        ts[1] = ts[2]
                        rov[0] = rov[2]
                        tr[0] = tr[2]
                        ts[0] = ts[2]
                    break
                else:
                    vgs[0] = vgs[2]
                    fv[0] = fv[2]
                    rov[0] = rov[2]
                    tr[0] = tr[2]
                    ts[0] = ts[2]

                    if abs((vgs[2] - vold) / vgs[2]) > tol2:
                        vold = vgs[2]
                    else:
                        break
            if abs(fv[2]) > 0.001:
                warn(f"Velocity Convergence Error at station {i}")

            self.v[i] = vgs[2]
            self.vout[i] = (
                self.vin[i] * (1 - jenny_kanki_factor) + self.v[i] * jenny_kanki_factor
            )
            self.kout[i] = self.vout[i] / self.v[i]
            self.taur[i] = tr[2]
            self.taus[i] = ts[2]

            self.cg[0][i] = self.gas.cg0(area, self.p[i], self.t[i])
            self.cg[1][i] = (self.v[i] / self._shaft_radius) * self.cg[0][i]
            self.cg[2][i] = (self.p[i] / self._shaft_radius) * self.cg[0][i]
            self.cg[3][i] = (
                self.mdot
                * self.p[i]
                * (
                    1 / (self.p[i] ** 2 - self.p[i + 1] ** 2)
                    + 1 / (self.p[i - 1] ** 2 - self.p[i] ** 2)
                )
            )
            self.cg[4][i] = (
                -self.mdot * self.p[i + 1] / (self.p[i] ** 2 - self.p[i + 1] ** 2)
            )
            self.cg[5][i] = -self.rho[i] * self.pitch[1]
            self.cg[6][i] = (self.v[i] / self._shaft_radius) * self.cg[5][i]
            self.cg[7][i] = (
                -self.mdot * self.p[i - 1] / (self.p[i - 1] ** 2 - self.p[i] ** 2)
            )
            self.cg[8][i] = (
                -self.cg[7][i] * jenny_kanki_factor * (self.v[i] - self.vin[i])
            )

            self.cx[0][i] = area / self._shaft_radius
            self.cx[1][i] = self.rho[i] * area
            self.cx[2][i] = (self.v[i] / self._shaft_radius) * self.cx[1][i]
            cxx1 = (
                (2 + friction_exponent_stator)
                * self.taus[i]
                * area_ratio_stator
                * self.pitch[0]
            ) / self.v[i]
            cxx2 = (
                (2 + friction_exponent_rotor)
                * self.taur[i]
                * area_ratio_rotor
                * self.pitch[0]
            ) / rov[2]
            self.cx[3][i] = self.mdot * self.kout[i] + cxx1 + cxx2
            self.cx[4][i] = -self.mdot * self.kout[i - 1]
            self.cx[5][i] = 0
            self.cx[6][i] = -self.mdot * jenny_kanki_factor * (
                self.v[i] - self.vin[i]
            ) * self.p[i] / (self.p[i - 1] ** 2 - self.p[i] ** 2) + (
                (self.taus[i] * area_ratio_stator - self.taur[i] * area_ratio_rotor)
                * (self.pitch[1] / self.p[i])
            )
            cxx3 = (
                -friction_exponent_stator * self.taus[i] * area_ratio_stator
                + friction_exponent_rotor * self.taur[i] * area_ratio_rotor
            ) * (
                self.pitch[0]
                * hydraulic_diameter
                / (2 * (self.radial_clearance[0] + self.tooth_height[0]) ** 2)
            )
            self.cx[7][i] = (
                self.mdot / self.radial_clearance[0]
            ) * jenny_kanki_factor * (self.vin[i] - self.v[i]) + cxx3

    def _solve_swirl_velocities(self):
        vgs = np.zeros(3)
        fv = np.zeros(3)
        rov = np.zeros(3)
        tr = np.zeros(3)
        ts = np.zeros(3)

        if self.omega == 0 and self.inlet_swirl_velocity == 0:
            return

        friction_exponent_rotor = -0.25
        friction_exponent_stator = -0.25
        friction_coefficient_rotor = 0.079
        friction_coefficient_stator = 0.079

        if self.seal_type == "inter":
            area_ratio_rotor = (self.tooth_height[0] + self.pitch[0]) / self.pitch[0]
            area_ratio_stator = (self.tooth_height[0] + self.pitch[0]) / self.pitch[0]
        elif self.seal_type == "rotor":
            area_ratio_rotor = (2 * self.tooth_height[0] + self.pitch[0]) / self.pitch[
                0
            ]
            area_ratio_stator = 1
        else:
            area_ratio_stator = (2 * self.tooth_height[0] + self.pitch[0]) / self.pitch[
                0
            ]
            area_ratio_rotor = 1

        hydraulic_diameter = (
            2
            * (self.radial_clearance[0] + self.tooth_height[0])
            * self.pitch[0]
            / (self.radial_clearance[0] + self.tooth_height[0] + self.pitch[0])
        )
        area = (self.tooth_height[0] + self.radial_clearance[0]) * self.pitch[0]

        self.v[0] = self.inlet_swirl_velocity
        self.taur[0] = 0
        self.taus[0] = 0
        itmx2 = 40
        tol2 = 1 * 10 ** (-8)
        vold = 1 * 10 ** (10)

        phi1 = (self.reference_temperatures[0] ** 1.5) / self.reference_viscosities[0]
        phi2 = (self.reference_temperatures[1] ** 1.5) / self.reference_viscosities[1]
        sutherland_b = (
            self.reference_temperatures[1] - self.reference_temperatures[0]
        ) / (phi2 - phi1)
        sutherland_s = (sutherland_b * phi1) - self.reference_temperatures[0]

        for i in range(1, self.n_teeth):
            mu = sutherland_b * (self.t[i]) ** 0.5 / (1 + (sutherland_s / self.t[i]))
            self.nu = mu / self.rho[i]
            vgs[1] = self.gas.sound_speed(self.p[i], self.t[i])
            vgs[0] = -vgs[1]

            rov[0] = (self._shaft_radius * self.omega) - vgs[0]
            rov[1] = (self._shaft_radius * self.omega) - vgs[1]
            for j in range(0, 2):
                tr[j] = (
                    0.5
                    * self.rho[i]
                    * rov[j]
                    * rov[j]
                    * friction_coefficient_rotor
                    * (
                        (abs(rov[j]) * hydraulic_diameter / self.nu)
                        ** friction_exponent_rotor
                    )
                    * np.copysign(1, rov[j])
                )
                ts[j] = (
                    0.5
                    * self.rho[i]
                    * vgs[j]
                    * vgs[j]
                    * friction_coefficient_stator
                    * (
                        (abs(vgs[j]) * hydraulic_diameter / self.nu)
                        ** friction_exponent_stator
                    )
                    * np.copysign(1, vgs[j])
                )
                fv[j] = (self.mdot * (vgs[j] - self.v[i - 1])) - (
                    self.pitch[0]
                    * (tr[j] * area_ratio_rotor - ts[j] * area_ratio_stator)
                )
            for itn2 in range(0, itmx2):
                vgs[2] = (vgs[0] * fv[1] - vgs[1] * fv[0]) / (fv[1] - fv[0])
                rov[2] = (self._shaft_radius * self.omega) - vgs[2]
                tr[2] = (
                    0.5
                    * self.rho[i]
                    * rov[2]
                    * rov[2]
                    * friction_coefficient_rotor
                    * (
                        (abs(rov[2]) * hydraulic_diameter / self.nu)
                        ** friction_exponent_rotor
                    )
                    * np.copysign(1, rov[2])
                )
                ts[2] = (
                    0.5
                    * self.rho[i]
                    * vgs[2]
                    * vgs[2]
                    * friction_coefficient_stator
                    * (
                        (abs(vgs[2]) * hydraulic_diameter / self.nu)
                        ** friction_exponent_stator
                    )
                    * np.copysign(1, vgs[2])
                )
                fv[2] = (self.mdot * (vgs[2] - self.v[i - 1])) - (
                    self.pitch[0]
                    * (tr[2] * area_ratio_rotor - ts[2] * area_ratio_stator)
                )

                if fv[2] * fv[0] < 0:
                    vgs[1] = vgs[2]
                    fv[1] = fv[2]
                    rov[1] = rov[2]
                    tr[1] = tr[2]
                    ts[1] = ts[2]
                    if abs((vgs[2] - vold) / vgs[2]) > tol2:
                        vold = vgs[2]
                    else:
                        break

                elif fv[2] * fv[0] == 0:
                    if fv[1] == 0:
                        vgs[1] = vgs[0]
                        fv[1] = fv[0]
                        vgs[2] = vgs[0]
                        fv[2] = fv[0]
                        rov[1] = rov[0]
                        tr[1] = tr[0]
                        ts[1] = ts[0]
                        rov[2] = rov[0]
                        tr[2] = tr[0]
                        ts[2] = ts[0]
                    else:
                        vgs[1] = vgs[2]
                        fv[1] = fv[2]
                        vgs[0] = vgs[2]
                        fv[0] = fv[2]
                        rov[1] = rov[2]
                        tr[1] = tr[2]
                        ts[1] = ts[2]
                        rov[0] = rov[2]
                        tr[0] = tr[2]
                        ts[0] = ts[2]
                    break
                else:
                    vgs[0] = vgs[2]
                    fv[0] = fv[2]
                    rov[0] = rov[2]
                    tr[0] = tr[2]
                    ts[0] = ts[2]
                    if abs((vgs[2] - vold) / vgs[2]) > tol2:
                        vold = vgs[2]
                    else:
                        break
            if abs(fv[2]) > 0.001:
                warn(f"Velocity Convergence Error at station {i}")

            self.v[i] = vgs[2]
            self.taur[i] = tr[2]
            self.taus[i] = ts[2]

            self.cg[0][i] = self.gas.cg0(area, self.p[i], self.t[i])
            self.cg[1][i] = (self.v[i] / self._shaft_radius) * self.cg[0][i]
            self.cg[2][i] = (self.p[i] / self._shaft_radius) * self.cg[0][i]
            self.cg[3][i] = (
                self.mdot
                * self.p[i]
                * (
                    1 / (self.p[i] ** 2 - self.p[i + 1] ** 2)
                    + 1 / (self.p[i - 1] ** 2 - self.p[i] ** 2)
                )
            )
            self.cg[4][i] = (
                -self.mdot * self.p[i + 1] / (self.p[i] ** 2 - self.p[i + 1] ** 2)
            )
            self.cg[5][i] = -self.rho[i] * self.pitch[1]
            self.cg[6][i] = (self.v[i] / self._shaft_radius) * self.cg[5][i]
            self.cg[7][i] = (
                -self.mdot * self.p[i - 1] / (self.p[i - 1] ** 2 - self.p[i] ** 2)
            )
            self.cg[8][i] = -self.cg[7][i] * (self.v[i] - self.v[i - 1])
            self.cx[0][i] = area / self._shaft_radius
            self.cx[1][i] = self.rho[i] * area
            self.cx[2][i] = (self.v[i] / self._shaft_radius) * self.cx[1][i]
            cxx1 = (
                (2 + friction_exponent_stator)
                * self.taus[i]
                * area_ratio_stator
                * self.pitch[0]
            ) / self.v[i]
            cxx2 = (
                (2 + friction_exponent_rotor)
                * self.taur[i]
                * area_ratio_rotor
                * self.pitch[0]
            ) / rov[2]
            self.cx[3][i] = self.mdot + cxx1 + cxx2
            self.cx[4][i] = -self.mdot
            self.cx[5][i] = 0
            self.cx[6][i] = -self.mdot * (self.v[i] - self.v[i - 1]) * self.p[i] / (
                self.p[i - 1] ** 2 - self.p[i] ** 2
            ) + (
                (self.taus[i] * area_ratio_stator - self.taur[i] * area_ratio_rotor)
                * (self.pitch[1] / self.p[i])
            )
            cxx3 = (
                -friction_exponent_stator * self.taus[i] * area_ratio_stator
                + friction_exponent_rotor * self.taur[i] * area_ratio_rotor
            ) * (
                self.pitch[0]
                * hydraulic_diameter
                / (2 * (self.radial_clearance[0] + self.tooth_height[0]) ** 2)
            )
            self.cx[7][i] = (self.mdot / self.radial_clearance[0]) * (
                self.v[i - 1] - self.v[i]
            ) + cxx3

    def _solve_perturbation_system(self):
        gmfull = np.zeros((1000, 1000))
        rhs1 = np.zeros(self.n_cavities * 8)
        rhs2 = np.zeros(self.n_cavities * 8)
        val = np.zeros(28)
        val2 = np.zeros(4)
        val3 = np.zeros(4)

        ir1 = [5, 6, 7, 8]
        ic1 = [6, 5, 8, 7]
        ir2 = [1, 2, 3, 4]
        ic2 = [2, 1, 4, 3]
        ir3 = [5, 6, 7, 8]
        ic3 = [2, 1, 4, 3]
        ir4 = [
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            3,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
        ]
        ic4 = [
            1,
            2,
            5,
            1,
            2,
            6,
            3,
            4,
            7,
            3,
            4,
            8,
            1,
            2,
            5,
            6,
            1,
            2,
            5,
            6,
            3,
            4,
            7,
            8,
            3,
            4,
            7,
            8,
        ]
        ir5 = [2, 4, 5, 7]
        ic6 = [2, 1, 4, 3]
        ir7 = [1, 2, 3, 4]
        ic7 = [2, 1, 4, 3]

        for i in range(0, self.n_cavities):
            for ict in range(0, 4):
                if i != 0:
                    irow = (i) * 8 + ir1[ict] - 1
                    icol = (i - 1) * 8 + ic1[ict] - 1
                    jcol = icol - irow + self._band_center - 1
                    self.gm[irow][jcol] = self.cx[4][i + 1]

                    icol = (i - 1) * 8 + ic6[ict] - 1
                    jcol = icol - irow + self._band_center - 1
                    self.gm[irow][jcol] = self.cg[8][i + 1]

                    irow = (i) * 8 + ir7[ict] - 1
                    icol = (i - 1) * 8 + ic7[ict] - 1
                    jcol = icol - irow + self._band_center - 1
                    self.gm[irow][jcol] = self.cg[7][i + 1]
                if i != (self.n_cavities - 1):
                    irow = (i) * 8 + ir2[ict] - 1
                    icol = (i + 1) * 8 + ic2[ict] - 1
                    jcol = icol - irow + self._band_center - 1
                    self.gm[irow][jcol] = self.cg[4][i + 1]

                    irow = (i) * 8 + ir3[ict] - 1
                    icol = (i + 1) * 8 + ic3[ict] - 1
                    jcol = icol - irow + self._band_center - 1
                    self.gm[irow][jcol] = self.cx[5][i + 1]
            cf1 = self.omega * self.cg[0][i + 1] + self.cg[1][i + 1]
            cf2 = self.cg[3][i + 1]
            cf3 = self.cg[2][i + 1]
            cf4 = -self.omega * self.cg[0][i + 1] + self.cg[1][i + 1]
            cf5 = self.cx[0][i + 1]
            cf6 = self.cx[6][i + 1]
            cf7 = self.omega * self.cx[1][i + 1] + self.cx[2][i + 1]
            cf8 = self.cx[3][i + 1]
            cf9 = -self.omega * self.cx[1][i + 1] + self.cx[2][i + 1]

            val[0] = cf1
            val[1] = cf2
            val[2] = cf3
            val[3] = cf2
            val[4] = -cf1
            val[5] = -cf3
            val[6] = cf4
            val[7] = cf2
            val[8] = cf3
            val[9] = cf2
            val[10] = -cf4
            val[11] = -cf3
            val[12] = cf5
            val[13] = cf6
            val[14] = cf7
            val[15] = cf8
            val[16] = cf6
            val[17] = -cf5
            val[18] = cf8
            val[19] = -cf7
            val[20] = cf5
            val[21] = cf6
            val[22] = cf9
            val[23] = cf8
            val[24] = cf6
            val[25] = -cf5
            val[26] = cf8
            val[27] = -cf9

            for ict in range(0, 28):
                irow = i * 8 + ir4[ict] - 1
                icol = i * 8 + ic4[ict] - 1
                jcol = icol - irow + self._band_center - 1
                self.gm[irow][jcol] = val[ict]
            val2[0] = 0.5 * (self.omega * self.cg[5][i + 1] + self.cg[6][i + 1])
            val2[1] = 0.5 * (-self.omega * self.cg[5][i + 1] + self.cg[6][i + 1])
            val2[2] = -0.5 * self.cx[7][i + 1]
            val2[3] = val2[2]
            val3[0] = -val2[0]
            val3[1] = val2[1]
            val3[2] = -val2[3]
            val3[3] = val2[2]

            for ict in range(0, 4):
                irow = i * 8 + ir5[ict] - 1
                self.rhs[irow][0] = (
                    self.pert_amplitude_direct
                    / self.perturbation_eccentricity
                    * val2[ict]
                )
                self.rhs[irow][1] = (
                    self.pert_amplitude_cross
                    / self.perturbation_eccentricity
                    * val3[ict]
                )
        for i in range(0, 8 * self.n_cavities):
            for j in range(0, 33):
                if 0 <= i + j - 16 <= (self.n_cavities * 8) - 1:
                    gmfull[i][i + j - 16] = self.gm[i][j]
        A = gmfull[: 8 * self.n_cavities, : 8 * self.n_cavities].copy()
        lu, piv = lu_factor(A)
        for i in range(0, 8 * self.n_cavities):
            rhs1[i] = self.rhs[i][0]
            rhs2[i] = self.rhs[i][1]
        cnd = cond(A)
        rcond = 1 / cnd
        self.pert_condition_number = cnd
        self.pert_rcond = rcond

        if rcond <= 1 / 3.0e8:
            raise ValueError(
                "The perturbation system is almost singular "
                f"(condition number {cnd:.3e}); no prediction is possible for "
                "the dynamic coefficients at this operating condition."
            )

        if rcond <= 1 / 1.0e6:
            warn(f"Array condition number is high \n array condition number e:{cnd}")

        sol1 = lu_solve((lu, piv), rhs1)
        sol2 = lu_solve((lu, piv), rhs2)
        for i in range(0, 8 * self.n_cavities):
            self.rhs[i][0] = sol1[i]
            self.rhs[i][1] = sol2[i]

        self.kxx = 0
        self.kxy = 0
        self.cxx = 0
        self.cxy = 0

        for i in range(0, self.n_cavities):
            icnt = (i) * 8 - 1
            self.kxx = self.kxx + self.rhs[icnt + 2][0] + self.rhs[icnt + 4][0]
            self.kxy = self.kxy + self.rhs[icnt + 1][1] - self.rhs[icnt + 3][1]
            self.cxx = self.cxx + self.rhs[icnt + 1][0] - self.rhs[icnt + 3][0]
            self.cxy = self.cxy + self.rhs[icnt + 2][1] + self.rhs[icnt + 4][1]

        self.kxx = (
            np.pi
            * self._shaft_radius
            * self.pitch[1]
            * (self.perturbation_eccentricity / self.pert_amplitude_direct)
            * self.kxx
        )
        self.kxy = (
            np.pi
            * self._shaft_radius
            * self.pitch[1]
            * (self.perturbation_eccentricity / self.pert_amplitude_cross)
            * self.kxy
        )
        self.kyx = -self.kxy
        if self.omega != 0:
            self.cxx = (
                -np.pi
                * self._shaft_radius
                * self.pitch[1]
                * (self.perturbation_eccentricity / self.pert_amplitude_direct)
                / self.omega
                * self.cxx
            )
            self.cxy = (
                np.pi
                * self._shaft_radius
                * self.pitch[1]
                * (self.perturbation_eccentricity / self.pert_amplitude_cross)
                / self.omega
                * self.cxy
            )
            self.cyx = -self.cxy
        else:
            self.cxx = 0
            self.cxy = 0
            self.cyx = 0

    def run(self, frequency):
        self.frequency = frequency
        self.inlet_swirl_velocity = self.preswirl * self.frequency * self._shaft_radius
        self._reset_state()
        self._vermes_leakage()
        self._solve_pressure_distribution()
        if self.use_jenny_kanki:
            self._solve_swirl_velocities_jenny_kanki()
        else:
            self._solve_swirl_velocities()
        self._solve_perturbation_system()

        attribute_coef = {
            "kxx": "kxx",
            "kyy": "kxx",
            "kxy": "kxy",
            "kyx": "kyx",
            "cxx": "cxx",
            "cyy": "cxx",
            "cxy": "cxy",
            "cyx": "cyx",
        }
        coefficients_dict = {k: getattr(self, v) for k, v in attribute_coef.items()}
        coefficients_dict["pressure"] = self.p.copy()
        coefficients_dict["seal_leakage"] = (
            self.mdot
            * 2
            * np.pi
            * (self._shaft_radius + 0.5 * self.radial_clearance[0])
        )
        coefficients_dict["pert_rcond"] = self.pert_rcond
        coefficients_dict["pert_condition_number"] = self.pert_condition_number

        return coefficients_dict

    def plot_pressure_distribution(
        self, pressure_units="MPa", length_units="m", fig=None, **kwargs
    ):
        """Plot pressure distribution for the labyrinth seal.

        Parameters
        ----------
        pressure_units : str, optional
            Pressure units for plotting.
            Default is "MPa".
        length_units : str, optional
            Length units for axial position.
            Default is "m".
        fig : Plotly graph_objects.Figure(), optional
            The figure object with the plot. If None, creates a new figure.
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

        n_cavities = self.n_teeth + 1

        fig.add_trace(
            go.Scatter(
                x=Q_(self.z[:n_cavities], "m").to(length_units).m,
                y=Q_(self.p[0][:n_cavities], "Pa").to(pressure_units).m,
                mode="lines+markers",
                name="Labyrinth Seal",
                line=dict(width=2),
                hovertemplate="<b>Position:</b> %{x:.3f} "
                + length_units
                + "<br>"
                + f"<b>Pressure:</b> %{{y:.3f}} {pressure_units}<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text="Pressure Distribution - Labyrinth Seal",
            ),
            xaxis_title=f"Axial Position ({length_units})",
            yaxis_title=f"Pressure ({pressure_units})",
            showlegend=False,
            **kwargs,
        )

        return fig
