"""Electric Motor Element module.

This module defines the MotorElement class which represents a 3-phase Induction Electric Motor
simulated using a 4th-order Runge-Kutta method, considering magnetic fluxes and currents.
"""

import numpy as np
import plotly.graph_objects as go

from ross.element import Element
from ross.units import Q_, check_units

from .motor_sourceAC import SourceAC
from .motor_results import MotorTimeResponseResults

__all__ = ["MotorElement", "motor_example", "run_motor_example"]


class MotorElement(Element):
    """A 3-phase Induction Motor element.

    This class creates a electric Three-Phase Induction Motor (TPIM) assuming
    rotor flux as sinchronous reference

    Parameters
    ----------
    n: int
        Node in which the motor will be coupled.
    power_nom : float, pint.Quantity
        Nominal power [W].
    voltage_nom : float, pint.Quantity
        Nominal voltage [V].
    speed_nom : float, pint.Quantity
        Nominal machine rotation [rad/s].
    stator_resistance : float, pint.Quantity
        Stator resistance [Ohm].
    rotor_resistance : float, pint.Quantity
        Rotor resistance [Ohm].
    stator_reactance : float
        Stator self-reactance at frequency [Ohm].
    rotor_reactance : float
        Rotor self-reactance at frequency [Ohm].
    mutual_reactance : float
        Mutual reactance at frequency [Ohm].
    Ip_motor : float, pint.Quantity
        Polar Moment of inertia related to motor axis [kg*m²].
    frequency : float, optional, pint.Quantity
        Nominal frequency [rad/s].
        Default is 60.0 Hz.
    n_poles: int, optional
        Number of machine's poles.
        Default is 4.
    viscosity_coeff : float, optional, pint.Quantity
        Viscosity coefficient [Pa*s].
        Default is 0.0.
    Ip_load : float, optional, pint.Quantity
        Polar Moment of inertia related to load [kg*m²].
        Default is 0.0.
    voltage_net : float, optional, pint.Quantity
        Electrical tension of Power Supply [V].
        Default is None, which adopts the motor's nominal voltage (`voltage`).
    frequency_net : float, optional, pint.Quantity
        Electrical frequency of Power Supply [rad/s].
        Default is None, which adopts the motor's nominal frequency (`frequency`).
    initial_angle_net : float, optional, pint.Quantity
        Initial angular phase frequency of Power Supply [rad].
        Default is 20.0 degrees.
    short_circuit_ratio_net : float, optional
        Short-Circuit Ratio in Common Coupling Point with Power Supply.
        Default is 50.0.
    XR_ratio_net : float, optional
        Reactance (X) / Resistance (R) Ratio in Coupling Point with Power Supply.
        Default is 80.0.
    tag : str, optional
        A tag to name the element.
        Default is None.
    """

    @check_units
    def __init__(
        self,
        n,
        power_nom,
        voltage_nom,
        speed_nom,
        stator_resistance,
        rotor_resistance,
        stator_reactance,
        rotor_reactance,
        mutual_reactance,
        Ip_motor,
        frequency=60.0,
        n_poles=4,
        viscosity_coeff=0.0,
        Ip_load=0.0,
        voltage_net=None,
        frequency_net=None,
        initial_angle_net=None,
        short_circuit_ratio_net=50.0,
        XR_ratio_net=80.0,
        tag=None,
    ):

        # Numerical Validation of NOMP entries
        self.power_nom = float(power_nom)
        self.voltage_nom = float(voltage_nom)
        self.speed_nom = float(speed_nom)
        self.n_poles = int(n_poles)
        self.stator_resistance = float(stator_resistance)
        self.rotor_resistance = float(rotor_resistance)
        self.stator_reactance = float(stator_reactance)
        self.rotor_reactance = float(rotor_reactance)
        self.mutual_reactance = float(mutual_reactance)
        self.Ip_motor = float(Ip_motor)
        self.viscosity_coeff = float(viscosity_coeff)
        self.Ip_load = float(Ip_load)

        if frequency is None:
            self.frequency = Q_(60.0, "Hz").to("rad/s").m
        else:
            self.frequency = float(frequency)

        # Numerical Validation of SCIP entries
        self.voltage_net = self.voltage if voltage_net is None else float(voltage_net)
        self.frequency_net = (
            self.frequency if frequency_net is None else float(frequency_net)
        )
        self.short_circuit_ratio_net = float(short_circuit_ratio_net)
        self.XR_ratio_net = float(XR_ratio_net)

        if initial_angle_net is None:
            self.initial_angle_net = Q_(20.0, "deg").to("rad").m
        else:
            self.initial_angle_net = float(initial_angle_net)

        # Internal model inductances parameters derived from CEMP
        Lls = self.stator_reactance / self.frequency
        Llr = self.rotor_reactance / self.frequency
        self.Lm = self.mutual_reactance / self.frequency
        self.Lss = Lls + self.Lm
        self.Lrr = Llr + self.Lm

        # Internal Electric Motor constants derived from NOMP and CEMP
        snom = (1 - self.speed_nom * self.n_poles / (2 * self.frequency)) * 100
        wnom = (self.frequency * (1 - snom / 100)) / (self.n_poles / 2)
        sigma = 1 - self.Lm**2 / (self.Lss * self.Lrr)
        self.Tnom = self.power_nom / wnom
        self.a = 1 / (sigma * self.Lss)
        self.b = 1 / (sigma * self.Lrr)
        self.c = self.Lm / (sigma * self.Lss * self.Lrr)

        # Short-Circuit Power and Impedances parameters derived from SCIP
        SCC_net = self.short_circuit_ratio_net * self.power_nom
        Zsc = self.voltage_net**2 / SCC_net
        Xsc = Zsc * self.XR_ratio_net / np.sqrt(1 + self.XR_ratio_net**2)
        self.short_circuit_resistance = Xsc / self.XR_ratio_net

        # Motor AC Source instance
        self.sourceAC = SourceAC(
            voltage_net=self.voltage_nom,
            frequency_net=Q_(self.frequency, "rad/s").to("Hz").m,
        )

        self.n = n
        self.tag = tag

    def __str__(self):
        """Convert object into string.

        Returns
        -------
        The object's parameters translated to strings.
        """
        return (
            f"Tag:                                {self.tag}"
            f"\nNode:                               {self.n}"
            f"\n--- Nominal Parameters (NOMP) ---"
            f"\nNominal Power (W):                  {self.power_nom}"
            f"\nNominal Voltage (V):                {self.voltage_nom}"
            f"\nNominal Rotation (rad/s):           {self.speed_nom}"
            f"\nNominal Frequency (Hz):             {Q_(self.frequency, 'rad/s').to('Hz').m}"
            f"\nNumber of Poles:                    {self.n_poles}"
            f"\n--- Circuit Parameters (CEMP) ---"
            f"\nStator Resistance (Ohm):            {self.stator_resistance}"
            f"\nRotor Resistance (Ohm):             {self.rotor_resistance}"
            f"\nStator Reactance (Ohm):             {self.stator_reactance}"
            f"\nRotor Reactance (Ohm):              {self.rotor_reactance}"
            f"\nMutual Reactance (Ohm):             {self.mutual_reactance}"
            f"\nMotor Inertia (kg*m2):              {self.Ip_motor}"
            f"\nViscosity Coefficient (Pa*s):       {self.viscosity_coeff}"
            f"\nLoad Inertia (kg*m2):               {self.Ip_load}"
            f"\n--- Power Supply Parameters (SCIP) ---"
            f"\nSupply Voltage (V):                 {self.voltage_net}"
            f"\nSupply Frequency (Hz):              {self.frequency_net}"
            f"\nInitial Phase Angle (deg):          {self.initial_angle_net}"
            f"\nShort-Circuit Ratio (ad):           {self.short_circuit_ratio_net}"
            f"\nX/R Ratio (ad):                     {self.XR_ratio_net}"
        )

    def __repr__(self):
        pass

    def __eq__(self, other):
        pass

    def __hash__(self):
        return hash(self.tag)

    def dof_mapping(self):
        pass

    def M(self):
        pass

    def K(self):
        pass

    def C(self):
        pass

    def G(self):
        pass

    def _calculate_dLds_dt(self, Lds, Ldr, Lqs, vds):
        """Calculate the derivative of the d-axis stator inductance based on the
        motor's state and input voltages.

        This method computes the rate of change of the d-axis stator inductance using
        the motor's current state variables and the input d-axis voltage.

        Parameters
        ----------
        Lds : float
            d-axis inductance for stator [H].
        Ldr : float
            d-axis inductance for rotor [H].
        Lqs : float
            q-axis inductance for stator [H].
        vds : float
            d-axis voltage for stator [V].

        Returns
        -------
        dLds_dt : float
            Derivative of the d-axis stator inductance with respect to time.
        """
        R = self.stator_resistance + self.short_circuit_resistance
        w = self.frequency

        return vds - R * self.a * Lds + R * self.c * Ldr + w * Lqs

    def _calculate_dLqs_dt(self, Lqs, Lds, Lqr, vqs):
        """Calculate the derivative of the q-axis stator inductance based on the
        motor's state and input voltages.

        This method computes the rate of change of the q-axis stator inductance using
        the motor's current state variables and the input q-axis voltage.

        Parameters
        ----------
        Lqs : float
            q-axis inductance for stator [H].
        Lds : float
            d-axis inductance for stator [H].
        Lqr : float
            q-axis inductance for rotor [H].
        vqs : float
            q-axis voltage for stator [V].

        Returns
        -------
        dLqs_dt : float
            Derivative of the q-axis stator inductance with respect to time.
        """
        R = self.stator_resistance + self.short_circuit_resistance
        w = self.frequency

        return vqs - R * self.a * Lqs + R * self.c * Lqr - w * Lds

    def _calculate_dLdr_dt(self, Ldr, Lds, Lqr, vdr, wr):
        """Calculate the derivative of the d-axis rotor inductance based on the
        motor's state and input voltages.

        This method computes the rate of change of the d-axis rotor inductance using
        the motor's current state variables and the input d-axis voltage.

        Parameters
        ----------
        Ldr : float
            d-axis inductance for rotor [H].
        Lds : float
            d-axis inductance for stator [H].
        Lqr : float
            q-axis inductance for rotor [H].
        vdr : float
            d-axis voltage for rotor [V].
        wr : float
            Rotor angular speed [rad/s].

        Returns
        -------
        dLdr_dt : float
            Derivative of the d-axis rotor inductance with respect to time.
        """
        R = self.rotor_resistance
        w = self.frequency - wr * self.n_poles / 2

        return vdr - R * self.b * Ldr + R * self.c * Lds + w * Lqr

    def _calculate_dLqr_dt(self, Lqr, Lqs, Ldr, vqr, wr):
        """Calculate the derivative of the q-axis rotor inductance based on the
        motor's state and input voltages.

        This method computes the rate of change of the q-axis rotor inductance using
        the motor's current state variables and the input q-axis voltage.

        Parameters
        ----------
        Lqr : float
            q-axis inductance for rotor [H].
        Lqs : float
            q-axis inductance for stator [H].
        Ldr : float
            d-axis inductance for rotor [H].
        vqr : float
            q-axis voltage for rotor [V].
        wr : float
            Rotor angular speed [rad/s].

        Returns
        -------
        dLqr_dt : float
            Derivative of the q-axis rotor inductance with respect to time.
        """
        R = self.rotor_resistance
        w = self.frequency - wr * self.n_poles / 2

        return vqr - R * self.b * Lqr + R * self.c * Lqs - w * Ldr

    def _calculate_dwr_dt(self, Tl, Te, wr):
        """Calculate the derivative of the rotor angular speed

        This method computes the rate of change of the rotor angular speed using
        the motor's current state variables, the input load torque, and the electric
        torque.

        Parameters
        ----------
        Tl : float
            Load torque applied to the shaft [N.m].
        Te : float
            Electric torque generated by the motor [N.m].
        wr : float
            Rotor angular speed [rad/s].

        Returns
        -------
        dwr_dt : float
            Derivative of the rotor angular speed with respect to time.
        """
        J = self.Ip_motor + self.Ip_load

        return (Te - self.viscosity_coeff * wr - Tl) / J

    def _calculate_electrical_torque(self, Lds, Ldr, Lqs, Lqr):
        """Calculate the electric torque generated by the motor.

        This method computes the electric torque based on the current inductances
        of the stator and rotor.

        Parameters
        ----------
        Lds : float
            d-axis inductance for stator [H].
        Ldr : float
            d-axis inductance for rotor [H].
        Lqs : float
            q-axis inductance for stator [H].
        Lqr : float
            q-axis inductance for rotor [H].

        Returns
        -------
        Te : float
            Electric torque generated by the motor [N.m].
        """
        return 1.5 * self.c * (Lqs * Ldr - Lds * Lqr) * self.n_poles / 2

    def _solve_rk4_step(
        self, dt, Tl, Te0, wr0, Lds0, Lqs0, Ldr0, Lqr0, vds, vqs, vdr, vqr
    ):
        """Solve a single 4th-order Runge-Kutta (RK4) integration step
        for the motor dynamic model.

        This method computes the motor state evolution over one simulation
        time step using the RK4 numerical integration method, based on the
        input voltages, load torque, and initial state variables.

        Parameters
        ----------
        dt : float
            Simulation time step (step size) for the Runge-Kutta integration.
        Tl : float
            Load torque applied to the shaft [N.m].
        Te0 : float
            Initial estimate of electric torque [N.m].
        wr0 : float
            Initial estimate of rotor angular speed [rad/s].
        Lds0 : float
            Initial estimate of d-axis inductance for stator [H].
        Lqs0 : float
            Initial estimate of q-axis inductance for stator [H].
        Ldr0 : float
            Initial estimate of d-axis inductance for rotor [H].
        Lqr0 : float
            Initial estimate of q-axis inductance for rotor [H].
        vds : float
            d-axis voltage for stator [V].
        vqs : float
            q-axis voltage for stator [V].
        vdr : float
            d-axis voltage for rotor [V].
        vqr : float
            q-axis voltage for rotor [V].

        Returns
        -------
        Te : float
            Current electric torque [N.m].
        wr : float
            Current rotor angular speed [rad/s].
        Lds : float
            Current d-axis inductance for stator [H].
        Lqs : float
            Current q-axis inductance for stator [H].
        Ldr : float
            Current d-axis inductance for rotor [H].
        Lqr : float
            Current q-axis inductance for rotor [H].
        """
        # Determine step size h based on current time or use fixed internal h
        # Note: The original logic relies on a fixed h for the RK coefficients.
        # We assume the user calls this sequentially or we rely on the internal h.

        # Runge-Kutta 4th Order Step
        estimate_rk4_stage = lambda h, y, k: y + k * h / 2
        integrate_rk4_state = lambda h, y, k1, k2, k3, k4: (
            y + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
        )

        k = np.zeros((4, 5))

        Lds = Lds0
        Lqs = Lqs0
        Ldr = Ldr0
        Lqr = Lqr0
        wr = wr0
        Te = Te0

        # Calculating k1, k2, k3, k4 for each state variable
        for i in range(4):
            k[i, 0] = self._calculate_dLds_dt(Lds, Ldr, Lqs, vds)
            k[i, 1] = self._calculate_dLqs_dt(Lqs, Lds, Lqr, vqs)
            k[i, 2] = self._calculate_dLdr_dt(Ldr, Lds, Lqr, vdr, wr0)
            k[i, 3] = self._calculate_dLqr_dt(Lqr, Lqs, Ldr, vqr, wr0)
            k[i, 4] = self._calculate_dwr_dt(Tl, Te, wr)

            Lds = estimate_rk4_stage(dt, Lds0, k[i, 0])
            Lqs = estimate_rk4_stage(dt, Lqs0, k[i, 1])
            Ldr = estimate_rk4_stage(dt, Ldr0, k[i, 2])
            Lqr = estimate_rk4_stage(dt, Lqr0, k[i, 3])
            wr = estimate_rk4_stage(dt, wr0, k[i, 4])
            Te = self._calculate_electrical_torque(Lds, Ldr, Lqs, Lqr)

        # Calculating final state values using the RK4 formula
        Lds = integrate_rk4_state(dt, Lds0, k[0, 0], k[1, 0], k[2, 0], k[3, 0])
        Lqs = integrate_rk4_state(dt, Lqs0, k[0, 1], k[1, 1], k[2, 1], k[3, 1])
        Ldr = integrate_rk4_state(dt, Ldr0, k[0, 2], k[1, 2], k[2, 2], k[3, 2])
        Lqr = integrate_rk4_state(dt, Lqr0, k[0, 3], k[1, 3], k[2, 3], k[3, 3])
        wr = integrate_rk4_state(dt, wr0, k[0, 4], k[1, 4], k[2, 4], k[3, 4])
        Te = self._calculate_electrical_torque(Lds, Ldr, Lqs, Lqr)

        return Te, wr, Lds, Lqs, Ldr, Lqr

    def run(self, t, load_torque_entrance_time=None, load_torque_ratio=1.0):
        """Run the motor simulation over a specified time vector.

        Parameters
        ----------
        t : array-like
            An array of time points at which to perform the simulation steps.
        load_torque_entrance_time : float, optional
            Time at which the load torque is applied to the motor shaft.
            Default is half the simulation time.
        load_torque_ratio : float, optional
            Load torque ratio applied at the entrance time. This is a multiplier
            for the nominal load torque, e.g., a value of 1.0 applies 100% of the
            nominal torque at entrance time.
            Default is 1.0.

        Returns
        -------
        results : dict
            A dictionary containing lists of results for the entire simulation:
            - tempo, Ias, Ibs, Ics, Ialfas, Ibetas, Ids, Iqs, TE, TC.
        """

        # Initial values of Electrical Torque, Rotor speed and Flux angle
        Te = 0.0  # Electrical Torque in N*m
        wr = 0.0  # Rotor's angular speed in rad*s
        thetar = 0.0  # Rotor's angle in rad
        flux_angle = self.initial_angle_net - np.pi / 2  # Flux's initial angle in rad

        # Initial alpha-beta and dq currents (based in nulled instantaneous phase currents)
        ias, ibs, ics = 0, 0, 0
        i_alpha = 2 / 3 * (ias - ibs / 2 - ics / 2)
        i_beta = 2 / 3 * (ibs - ics) * np.sqrt(3) / 2
        ids = i_alpha * np.cos(flux_angle) + i_beta * np.sin(flux_angle)
        iqs = -i_alpha * np.sin(flux_angle) + i_beta * np.cos(flux_angle)

        # Initial rotor and stator's inductances
        Lds = self.Lss * ids + self.Lm * 0
        Lqs = self.Lss * iqs + self.Lm * 0
        Ldr = self.Lrr * 0 + self.Lm * ids
        Lqr = self.Lrr * 0 + self.Lm * iqs

        # Initial simulation parameters scheme
        t = np.array(t)

        if load_torque_entrance_time is None:
            load_torque_entrance_time = (t[-1] - t[0]) / 2

        # Catching the near index to time do load_torque entrance
        idx = np.abs(t - load_torque_entrance_time).argmin()
        load_torque = np.ones_like(t) * self.Tnom * load_torque_ratio
        load_torque[0:idx] = 0.0

        nt = len(t)

        speed = np.zeros(nt)
        electric_torque = np.zeros(nt)
        currents = {
            "a": np.zeros(nt),
            "b": np.zeros(nt),
            "c": np.zeros(nt),
            "alpha": np.zeros(nt),
            "beta": np.zeros(nt),
            "d": np.zeros(nt),
            "q": np.zeros(nt),
        }
        voltages = {"a": np.zeros(nt), "b": np.zeros(nt), "c": np.zeros(nt)}

        for i in range(1, nt):
            dt = t[i] - t[i - 1]

            # Updating angles
            flux_angle += self.frequency * dt
            thetar += (wr * self.n_poles / 2) * dt

            # Electrical 3-phase tensions
            vas, vbs, vcs = self.sourceAC(t[i])

            # Clarke & Park Transforms for Voltages
            v_alpha = 2 / 3 * (vas - vbs / 2 - vcs / 2)
            v_beta = 2 / 3 * (vbs - vcs) * np.sqrt(3) / 2
            vds = v_alpha * np.cos(flux_angle) + v_beta * np.sin(flux_angle)
            vqs = -v_alpha * np.sin(flux_angle) + v_beta * np.cos(flux_angle)
            vdr, vqr = 0, 0

            Tl = load_torque[i]

            # Run single step estimation
            Te, wr, Lds, Lqs, Ldr, Lqr = self._solve_rk4_step(
                dt, Tl, Te, wr, Lds, Lqs, Ldr, Lqr, vds, vqs, vdr, vqr
            )

            # Calculate outputs
            ids = self.a * Lds - self.c * Ldr
            iqs = self.a * Lqs - self.c * Lqr
            i_alpha = ids * np.cos(flux_angle) - iqs * np.sin(flux_angle)
            i_beta = ids * np.sin(flux_angle) + iqs * np.cos(flux_angle)
            ias = i_alpha
            ibs = -i_alpha / 2 + np.sqrt(3) * i_beta / 2
            ics = -i_alpha / 2 - np.sqrt(3) * i_beta / 2

            speed[i] = wr
            electric_torque[i] = Te
            currents["a"][i] = ias
            currents["b"][i] = ibs
            currents["c"][i] = ics
            currents["alpha"][i] = i_alpha
            currents["beta"][i] = i_beta
            currents["d"][i] = ids
            currents["q"][i] = iqs
            voltages["a"][i] = vas
            voltages["b"][i] = vbs
            voltages["c"][i] = vcs

        results = MotorTimeResponseResults(
            t, electric_torque, load_torque, speed, currents, voltages
        )

        return results


def motor_example():
    """Create an example of notor element.

    This function returns an instance of a simple electric motor. The purpose is
    to make available a simple model so that doctest can be written using it.

    Returns
    -------
    motor : ross.MotorElement
        An instance of a motor object.
    """

    return MotorElement(
        n=0,
        tag=None,
        power_nom=Q_(1.5, "cv"),
        voltage_nom=127,
        speed_nom=Q_(1725, "RPM"),
        frequency=Q_(60.0, "Hz"),
        n_poles=4,
        stator_resistance=2.5,
        rotor_resistance=1.8,
        stator_reactance=1.3,
        rotor_reactance=1.3,
        mutual_reactance=43.08,
        Ip_motor=0.0372,
        viscosity_coeff=0.0,
        Ip_load=0.0,
        voltage_net=127,
        frequency_net=Q_(60.0, "Hz"),
    )


def run_motor_example():
    """Run the motor example and plot the results."""
    motor = motor_example()

    # Adjusting simulation parameters
    motor.sourceAC.harmonics(fHO=[5, 7], aHO=[5, 5])
    motor.sourceAC.voltage_net = 90.0
    motor.sourceAC.harmonics("enable")
    motor.sourceAC.unbalances("disable")

    tI = 0.0
    tF = 5.0
    step = 1e-4
    t = np.arange(tI, tF, step)

    results = motor.run(t, load_torque_entrance_time=3.0, load_torque_ratio=1.5)

    fig_torque = results.plot_torque()
    fig_speed = results.plot_speed()
    fig_currents = results.plot_phase_currents(reference_frame="a-b-c")
    fig_voltages = results.plot_phase_tensions()

    fig_torque.show()
    fig_speed.show()
    fig_currents.show()
    fig_voltages.show()
