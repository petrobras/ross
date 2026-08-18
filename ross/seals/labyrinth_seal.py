import multiprocessing
from warnings import warn

import ccp
import numpy as np
import plotly.graph_objects as go
from numpy.linalg import cond
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq, root_scalar

from ross import SealElement
from ross.seals.gas_model import IdealGas, RealGas, extract_gas_properties
from ross.units import Q_, check_units

__all__ = ["LabyrinthSeal"]

BLASIUS_FRICTION_COEFFICIENT = 0.079
BLASIUS_FRICTION_EXPONENT = -0.25
JENNY_KANKI_MOMENTUM_FACTOR = {"stator": 0.15, "rotor": 0.35, "inter": 0.90}

PERTURBATION_DOFS = 8
CONTINUITY_ROWS = (0, 1, 2, 3)
MOMENTUM_ROWS = (4, 5, 6, 7)
SWAPPED_PRESSURE_COLS = (1, 0, 3, 2)
SWAPPED_VELOCITY_COLS = (5, 4, 7, 6)
RHS_ROWS = (1, 3, 4, 6)


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
       - Solves for static pressure at each cavity using bracketed root-finding
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
        Number of teeth (throttlings). Must be at least 2.
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
        If gas_composition is None, provide molar_mass, gamma,
        reference_temperatures, and reference_viscosities parameters.
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
        if seal_type not in ("rotor", "stator", "inter"):
            raise ValueError(
                f"Invalid seal_type {seal_type!r}; expected 'rotor', 'stator' "
                "or 'inter'."
            )
        if n_teeth < 2:
            raise ValueError("The labyrinth model requires at least 2 teeth.")

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
            R = 8314.0 / molar_mass  # Universal gas constant over molar mass.

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
        elif self.gas_model == "ideal":
            self.gas = IdealGas(self.R, self.gamma)
        else:
            raise ValueError(
                f"Invalid gas_model {self.gas_model!r}; expected 'ideal' or 'real'."
            )
        self._real_gas = self.gas_model == "real"

        if reference_temperatures is None:
            reference_temperatures = [state_in.T().m, state_out.T().m]
        if reference_viscosities is None:
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

        self.n_stations = n_teeth + 1
        self.n_cavities = n_teeth - 1
        self.ndof = PERTURBATION_DOFS * self.n_cavities
        self.z = np.arange(self.n_stations) * pitch

        self.perturbation_eccentricity = 0.6
        self.pert_amplitude_direct = self.perturbation_eccentricity * radial_clearance
        self.pert_amplitude_cross = self.perturbation_eccentricity * radial_clearance

        coefficients_dict = {}
        if kwargs.get("kxx") is None:
            # Use multiprocessing only when beneficial (>4 frequencies);
            # sequential execution avoids process spawn overhead otherwise.
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

    def _reset_state(self):
        """Reset the per-run flow state before solving a new frequency."""
        self.pr = np.zeros(self.n_teeth)
        self.p = np.zeros(self.n_stations)
        self.w = np.zeros(self.n_stations)
        self.v = np.zeros(self.n_stations)
        self.rho = np.zeros(self.n_stations)
        self.t = np.full(self.n_stations, float(self.inlet_temperature))
        self.taur = np.zeros(self.n_stations)
        self.taus = np.zeros(self.n_stations)
        self.vin = np.zeros(self.n_stations)
        self.vout = np.zeros(self.n_stations)
        self.kout = np.zeros(self.n_stations)
        self.cg = np.zeros((9, self.n_stations))
        self.cx = np.zeros((8, self.n_stations))

        self.overall_pressure_ratio = self.outlet_pressure / self.inlet_pressure
        self.omega = self.frequency

    def _circumferential_leakage(self, mdot):
        """Convert mass flux per unit circumference into total leakage (kg/s)."""
        return mdot * 2 * np.pi * (self._shaft_radius + 0.5 * self.radial_clearance)

    def _solve_choked_flow_function(self):
        """Find the choked pressure ratio and the seal flow function.

        The choked pressure ratio maximizes the Vermes flow function
        ``eta(r) = sqrt((1 - r**2) / (n_teeth - ln(r)))``; it is located by
        finding the root of ``d(eta)/d(r)``. If the overall pressure ratio is
        below the choked ratio the seal is choked and the flow function is
        evaluated at the choked ratio instead.
        """
        n = self.n_teeth - 1

        def flow_function_derivative(r):
            num = -2 * (n + 1) + 2 * r * np.log(r) + 1 / r - r
            den = ((1 - r**2) ** 0.5) * ((n - np.log(r)) ** 1.5)
            return num / den

        self.choked_pressure_ratio = brentq(
            flow_function_derivative, 1e-3, 0.99, xtol=1e-12
        )
        self.flow_function_choke = (
            (1 - self.choked_pressure_ratio**2)
            / (self.n_teeth - np.log(self.choked_pressure_ratio))
        ) ** 0.5

        if self.overall_pressure_ratio < self.choked_pressure_ratio:
            self.flow_function_last_tooth = self.flow_function_choke
        else:
            self.flow_function_last_tooth = (
                (1 - self.overall_pressure_ratio**2)
                / (self.n_teeth - np.log(self.overall_pressure_ratio))
            ) ** 0.5

    def _vermes_leakage(self):
        """Estimate the initial leakage with the Vermes model.

        Provides the starting mass flux for the pressure distribution solver.
        """
        width_to_clearance_ratio = self.tooth_width / self.radial_clearance
        self.discharge_coefficient = (
            0.67675
            - (0.08519 * width_to_clearance_ratio)
            + (0.0878 * (width_to_clearance_ratio**2))
            - (0.01819 * (width_to_clearance_ratio**3))
            + (0.00111 * (width_to_clearance_ratio**4))
        )
        self.carryover_factor = 8.52 / (
            ((self.pitch - self.tooth_width) / self.radial_clearance) + 7.23
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
        self._solve_choked_flow_function()
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
            * self.radial_clearance
            / (self.R * self.inlet_temperature) ** 0.5
        )
        if self.print_results:
            leakage_vermes = self._circumferential_leakage(self.mdot_vermes)
            print(f"{'   Leakage':<40} {leakage_vermes:>15.8f} kg/s \n \n")
        self.mdot = self.mdot_vermes

    def _throttle_pressure_ratio(self, station, critical_pr):
        """Solve the pressure ratio across one tooth for the current mass flux.

        The pressure ratio is the root of the throttle mass-flux balance,
        bracketed between the critical (choked) ratio and 1. Returns None when
        the required mass flux exceeds the choked flux, meaning the current
        ``mdot`` cannot pass through this tooth.
        """
        i = station
        pr_high = 0.9999999

        def residual(pressure_ratio):
            flux = self.gas.throttle_mass_flux(
                self.discharge_coefficient,
                self.radial_clearance,
                self.p[i - 1],
                pressure_ratio,
                self.rho[i - 1],
                self.t[i - 1],
                self.w[i - 1],
                self.carryover_factor,
            )
            return self.mdot - flux

        if residual(critical_pr) >= 0:
            return None
        if residual(pr_high) <= 0:
            warn(f"Pressure Convergence Error at Station {i}")
            return pr_high
        pressure_ratio = brentq(residual, critical_pr, pr_high, xtol=1e-14)
        if abs(residual(pressure_ratio)) > 1e-4:
            warn(f"Pressure Convergence Error at Station {i}")
        return pressure_ratio

    def _outlet_critical_pr(self):
        """Return the critical pressure ratio at the last throttle."""
        i = self.n_teeth - 1
        return self.gas.critical_pr(
            self.p[i], self.w[i], self.carryover_factor, self.t[i]
        )

    def _solve_pressure_distribution(self):
        """Solve the cavity pressures and the leakage mass flux.

        Outer loop: bisection on the mass flux ``mdot`` until the pressure at
        the outlet station matches the outlet pressure, or the last throttle
        chokes. Inner loop: march through the teeth solving the pressure ratio
        that passes ``mdot`` across each one, updating the isentropic state
        (pressure, throat velocity, density, temperature) cavity by cavity.
        """
        tol_outlet_pressure = 1e-5
        tol_choked = 0.005

        mdot_low, mdot_high = 0.0, self.mdot * 5
        refresh_bracket = False
        frozen_inlet_state = False
        error_outlet_pressure = 0.0

        while True:
            if refresh_bracket:
                mdot_low, mdot_high = 0.0, self.mdot * 5
                refresh_bracket = False
            if not frozen_inlet_state:
                self.w[0] = 0.0
                self.p[0] = self.inlet_pressure
                self.rho[0] = self.gas.inlet_density(self.p[0], self.t[0])
                critical_pr = self._outlet_critical_pr()

            choked_mid_seal = False
            for i in range(1, self.n_teeth + 1):
                pressure_ratio = self._throttle_pressure_ratio(i, critical_pr)
                if pressure_ratio is None:
                    choked_mid_seal = True
                    break
                self.pr[i - 1] = pressure_ratio
                self.p[i] = pressure_ratio * self.p[i - 1]
                self.w[i] = self.gas.throat_velocity(
                    self.mdot,
                    self.discharge_coefficient,
                    self.radial_clearance,
                    self.p[i - 1],
                    pressure_ratio,
                    self.t[i - 1],
                )
                self.rho[i] = self.gas.density_isentropic(
                    self.p[i - 1], pressure_ratio, self.rho[i - 1]
                )
                self.t[i] = self.gas.temperature_isentropic(
                    self.p[i - 1], pressure_ratio, self.t[i - 1]
                )

            if not choked_mid_seal:
                outlet_critical_pr = self._outlet_critical_pr()
                error_outlet_pressure = (
                    self.p[self.n_teeth] - self.outlet_pressure
                ) / self.outlet_pressure
                near_choked = (
                    abs(self.pr[self.n_teeth - 1] - outlet_critical_pr)
                    / outlet_critical_pr
                    <= tol_choked
                )
                if abs(error_outlet_pressure) < tol_outlet_pressure or near_choked:
                    break

            mdot_old = self.mdot
            if choked_mid_seal or error_outlet_pressure < 0:
                self.mdot = (mdot_low + self.mdot) / 2
                mdot_high = mdot_old
            else:
                self.mdot = (mdot_high + self.mdot) / 2
                mdot_low = mdot_old
            if self.mdot == mdot_old:
                # The bisection stagnated; restart it around the current flux
                # and keep the current inlet state and critical ratio.
                if self.print_results:
                    print("Reset iteration")
                frozen_inlet_state = True
                refresh_bracket = True

        if (
            abs(self.pr[self.n_teeth - 1] - self._outlet_critical_pr())
            / self._outlet_critical_pr()
            <= tol_choked
        ):
            warn("Flow choked in the last throttle.")
        if self.pr[self.n_teeth - 1] > 1:
            raise ValueError("Error in Leakage Calculation")

        if self.print_results:
            leakage = self._circumferential_leakage(self.mdot)
            print(f"{'   Leakage':<40} {leakage:>15.8f} kg/s \n")

    def _solve_swirl_velocities(self):
        """Solve the cavity swirl velocities and the base-flow gradients.

        For each cavity, the swirl velocity balances the through-flow momentum
        against the rotor and stator wall shear (Blasius-type friction). With
        ``use_jenny_kanki=True``, the Jenny and Kanki momentum parameters
        reduce the fraction of the through-flow momentum exchanged in each
        cavity; the classic model corresponds to a momentum factor of 1.

        The method also assembles the base-flow gradient tables ``cg``
        (continuity) and ``cx`` (momentum) used by the perturbation solver.
        """
        if self.omega == 0 and self.inlet_swirl_velocity == 0:
            return

        if self.use_jenny_kanki:
            momentum_factor = JENNY_KANKI_MOMENTUM_FACTOR[self.seal_type]
        else:
            momentum_factor = 1.0

        if self.seal_type == "inter":
            area_ratio_rotor = (self.tooth_height + self.pitch) / self.pitch
            area_ratio_stator = area_ratio_rotor
        elif self.seal_type == "rotor":
            area_ratio_rotor = (2 * self.tooth_height + self.pitch) / self.pitch
            area_ratio_stator = 1.0
        else:
            area_ratio_stator = (2 * self.tooth_height + self.pitch) / self.pitch
            area_ratio_rotor = 1.0

        cavity_height = self.radial_clearance + self.tooth_height
        hydraulic_diameter = (
            2 * cavity_height * self.pitch / (cavity_height + self.pitch)
        )
        area = cavity_height * self.pitch

        surface_velocity = self._shaft_radius * self.omega

        self.v[0] = self.inlet_swirl_velocity
        self.vin[0] = self.inlet_swirl_velocity
        self.vout[0] = self.inlet_swirl_velocity
        # In the Jenny-Kanki model the first cavity has no upstream momentum
        # recovery, so its upstream factor is zero; the classic model uses 1.
        self.kout[0] = 0.0 if self.use_jenny_kanki else 1.0

        phi1 = self.reference_temperatures[0] ** 1.5 / self.reference_viscosities[0]
        phi2 = self.reference_temperatures[1] ** 1.5 / self.reference_viscosities[1]
        sutherland_b = (
            self.reference_temperatures[1] - self.reference_temperatures[0]
        ) / (phi2 - phi1)
        sutherland_s = (sutherland_b * phi1) - self.reference_temperatures[0]

        for i in range(1, self.n_teeth):
            self.vin[i] = self.vout[i - 1]
            mu = sutherland_b * self.t[i] ** 0.5 / (1 + sutherland_s / self.t[i])
            nu = mu / self.rho[i]

            def wall_shear(velocity):
                return (
                    0.5
                    * self.rho[i]
                    * velocity
                    * velocity
                    * BLASIUS_FRICTION_COEFFICIENT
                    * (abs(velocity) * hydraulic_diameter / nu)
                    ** BLASIUS_FRICTION_EXPONENT
                    * np.copysign(1.0, velocity)
                )

            def momentum_residual(v_guess):
                return (self.mdot * (v_guess - self.vin[i])) - self.pitch * (
                    wall_shear(surface_velocity - v_guess) * area_ratio_rotor
                    - wall_shear(v_guess) * area_ratio_stator
                )

            sound_speed = self.gas.sound_speed(self.p[i], self.t[i])
            try:
                v_swirl = brentq(
                    momentum_residual, -sound_speed, sound_speed, xtol=1e-12
                )
            except ValueError:
                v_swirl = root_scalar(
                    momentum_residual,
                    x0=-sound_speed,
                    x1=sound_speed,
                    method="secant",
                ).root
            if abs(momentum_residual(v_swirl)) > 0.001:
                warn(f"Velocity Convergence Error at station {i}")

            v_relative = surface_velocity - v_swirl
            self.v[i] = v_swirl
            self.vout[i] = (
                self.vin[i] * (1 - momentum_factor) + v_swirl * momentum_factor
            )
            self.kout[i] = self.vout[i] / self.v[i] if self.use_jenny_kanki else 1.0
            self.taur[i] = wall_shear(v_relative)
            self.taus[i] = wall_shear(v_swirl)

            self.cg[0, i] = self.gas.cg0(area, self.p[i], self.t[i])
            self.cg[1, i] = (self.v[i] / self._shaft_radius) * self.cg[0, i]
            self.cg[2, i] = (self.p[i] / self._shaft_radius) * self.cg[0, i]
            self.cg[3, i] = (
                self.mdot
                * self.p[i]
                * (
                    1 / (self.p[i] ** 2 - self.p[i + 1] ** 2)
                    + 1 / (self.p[i - 1] ** 2 - self.p[i] ** 2)
                )
            )
            self.cg[4, i] = (
                -self.mdot * self.p[i + 1] / (self.p[i] ** 2 - self.p[i + 1] ** 2)
            )
            self.cg[5, i] = -self.rho[i] * self.pitch
            self.cg[6, i] = (self.v[i] / self._shaft_radius) * self.cg[5, i]
            self.cg[7, i] = (
                -self.mdot * self.p[i - 1] / (self.p[i - 1] ** 2 - self.p[i] ** 2)
            )
            self.cg[8, i] = -self.cg[7, i] * momentum_factor * (self.v[i] - self.vin[i])

            self.cx[0, i] = area / self._shaft_radius
            self.cx[1, i] = self.rho[i] * area
            self.cx[2, i] = (self.v[i] / self._shaft_radius) * self.cx[1, i]
            shear_gradient_stator = (
                (2 + BLASIUS_FRICTION_EXPONENT)
                * self.taus[i]
                * area_ratio_stator
                * self.pitch
            ) / self.v[i]
            shear_gradient_rotor = (
                (2 + BLASIUS_FRICTION_EXPONENT)
                * self.taur[i]
                * area_ratio_rotor
                * self.pitch
            ) / v_relative
            self.cx[3, i] = (
                self.mdot * self.kout[i] + shear_gradient_stator + shear_gradient_rotor
            )
            self.cx[4, i] = -self.mdot * self.kout[i - 1]
            self.cx[5, i] = 0
            self.cx[6, i] = -self.mdot * momentum_factor * (
                self.v[i] - self.vin[i]
            ) * self.p[i] / (self.p[i - 1] ** 2 - self.p[i] ** 2) + (
                (self.taus[i] * area_ratio_stator - self.taur[i] * area_ratio_rotor)
                * (self.pitch / self.p[i])
            )
            shear_clearance_gradient = (
                -BLASIUS_FRICTION_EXPONENT * self.taus[i] * area_ratio_stator
                + BLASIUS_FRICTION_EXPONENT * self.taur[i] * area_ratio_rotor
            ) * (self.pitch * hydraulic_diameter / (2 * cavity_height**2))
            self.cx[7, i] = (self.mdot / self.radial_clearance) * momentum_factor * (
                self.vin[i] - self.v[i]
            ) + shear_clearance_gradient

    def _assemble_perturbation_system(self):
        """Assemble the linearized perturbation system.

        Each interior cavity contributes 8 degrees of freedom: the cosine and
        sine components of the pressure (0-3) and swirl velocity (4-7)
        perturbations for the two whirl directions. Rows 0-3 are the
        linearized continuity equations and rows 4-7 the tangential momentum
        equations. Cavities couple to their upstream and downstream neighbors
        through the base-flow gradient tables ``cg`` and ``cx``.

        Returns the system matrix ``A`` and the right-hand sides for the
        direct and cross whirl perturbations as a ``(ndof, 2)`` array.
        """
        A = np.zeros((self.ndof, self.ndof))
        rhs = np.zeros((self.ndof, 2))

        for i in range(self.n_cavities):
            row0 = PERTURBATION_DOFS * i
            station = i + 1

            if i > 0:
                upstream0 = PERTURBATION_DOFS * (i - 1)
                for r, c in zip(MOMENTUM_ROWS, SWAPPED_VELOCITY_COLS):
                    A[row0 + r, upstream0 + c] = self.cx[4, station]
                for r, c in zip(MOMENTUM_ROWS, SWAPPED_PRESSURE_COLS):
                    A[row0 + r, upstream0 + c] = self.cg[8, station]
                for r, c in zip(CONTINUITY_ROWS, SWAPPED_PRESSURE_COLS):
                    A[row0 + r, upstream0 + c] = self.cg[7, station]
            if i < self.n_cavities - 1:
                downstream0 = PERTURBATION_DOFS * (i + 1)
                for r, c in zip(CONTINUITY_ROWS, SWAPPED_PRESSURE_COLS):
                    A[row0 + r, downstream0 + c] = self.cg[4, station]
                for r, c in zip(MOMENTUM_ROWS, SWAPPED_PRESSURE_COLS):
                    A[row0 + r, downstream0 + c] = self.cx[5, station]

            cf1 = self.omega * self.cg[0, station] + self.cg[1, station]
            cf2 = self.cg[3, station]
            cf3 = self.cg[2, station]
            cf4 = -self.omega * self.cg[0, station] + self.cg[1, station]
            cf5 = self.cx[0, station]
            cf6 = self.cx[6, station]
            cf7 = self.omega * self.cx[1, station] + self.cx[2, station]
            cf8 = self.cx[3, station]
            cf9 = -self.omega * self.cx[1, station] + self.cx[2, station]

            diagonal_entries = (
                (0, 0, cf1),
                (0, 1, cf2),
                (0, 4, cf3),
                (1, 0, cf2),
                (1, 1, -cf1),
                (1, 5, -cf3),
                (2, 2, cf4),
                (2, 3, cf2),
                (2, 6, cf3),
                (3, 2, cf2),
                (3, 3, -cf4),
                (3, 7, -cf3),
                (4, 0, cf5),
                (4, 1, cf6),
                (4, 4, cf7),
                (4, 5, cf8),
                (5, 0, cf6),
                (5, 1, -cf5),
                (5, 4, cf8),
                (5, 5, -cf7),
                (6, 2, cf5),
                (6, 3, cf6),
                (6, 6, cf9),
                (6, 7, cf8),
                (7, 2, cf6),
                (7, 3, -cf5),
                (7, 6, cf8),
                (7, 7, -cf9),
            )
            for r, c, value in diagonal_entries:
                A[row0 + r, row0 + c] = value

            forcing_direct = (
                0.5 * (self.omega * self.cg[5, station] + self.cg[6, station]),
                0.5 * (-self.omega * self.cg[5, station] + self.cg[6, station]),
                -0.5 * self.cx[7, station],
                -0.5 * self.cx[7, station],
            )
            forcing_cross = (
                -forcing_direct[0],
                forcing_direct[1],
                -forcing_direct[3],
                forcing_direct[2],
            )
            for r, direct, cross in zip(RHS_ROWS, forcing_direct, forcing_cross):
                rhs[row0 + r, 0] = (
                    self.pert_amplitude_direct / self.perturbation_eccentricity * direct
                )
                rhs[row0 + r, 1] = (
                    self.pert_amplitude_cross / self.perturbation_eccentricity * cross
                )

        return A, rhs

    def _solve_perturbation_system(self):
        """Solve the perturbation system and extract the dynamic coefficients.

        The perturbation pressures are integrated around the circumference and
        along the seal to produce the direct and cross-coupled stiffness and
        damping coefficients.
        """
        A, rhs = self._assemble_perturbation_system()

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

        lu, piv = lu_factor(A)
        solution = lu_solve((lu, piv), rhs)
        solution_direct = solution[:, 0].reshape(self.n_cavities, PERTURBATION_DOFS)
        solution_cross = solution[:, 1].reshape(self.n_cavities, PERTURBATION_DOFS)

        kxx = np.sum(solution_direct[:, 1] + solution_direct[:, 3])
        kxy = np.sum(solution_cross[:, 0] - solution_cross[:, 2])
        cxx = np.sum(solution_direct[:, 0] - solution_direct[:, 2])
        cxy = np.sum(solution_cross[:, 1] + solution_cross[:, 3])

        scale_direct = (
            np.pi
            * self._shaft_radius
            * self.pitch
            * (self.perturbation_eccentricity / self.pert_amplitude_direct)
        )
        scale_cross = (
            np.pi
            * self._shaft_radius
            * self.pitch
            * (self.perturbation_eccentricity / self.pert_amplitude_cross)
        )

        self.kxx = scale_direct * kxx
        self.kxy = scale_cross * kxy
        self.kyx = -self.kxy
        if self.omega != 0:
            self.cxx = -scale_direct / self.omega * cxx
            self.cxy = scale_cross / self.omega * cxy
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
        self._solve_swirl_velocities()
        self._solve_perturbation_system()

        coefficients_dict = {
            "kxx": self.kxx,
            "kyy": self.kxx,
            "kxy": self.kxy,
            "kyx": self.kyx,
            "cxx": self.cxx,
            "cyy": self.cxx,
            "cxy": self.cxy,
            "cyx": self.cyx,
            "pressure": self.p,
            "seal_leakage": self._circumferential_leakage(self.mdot),
            "pert_rcond": self.pert_rcond,
            "pert_condition_number": self.pert_condition_number,
        }
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
