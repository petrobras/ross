"""Electric Motor Element module.

This module defines the MotorElement class which represents a 3-phase Induction Electric Motor
simulated using a 4th-order Runge-Kutta method, considering magnetic fluxes and currents.
"""

import numpy as np
import scipy as sp
import plotly.graph_objects as go

from ross.element import Element
from ross.units import Q_, check_units

from .motor_sourceAC import SourceAC
from .motor_results import MotorResponseResults

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
    
    def _calculate_load_torque(self, t, load_torque_entrance_time, load_torque_ratio):
        """Calculate the load torque.

        This method computes the load torque based on time.

        Parameters
        ----------
        t : float
            Specific time [s].
        load_torque_entrance_time : float
            Time at which the load torque is applied to the motor shaft [s].
        load_torque_ratio : float
            Load torque ratio applied at the entrance time.

        Returns
        -------
        Tl : float
            Load torque generated in time [N.m].
        """
        if t < load_torque_entrance_time:
            Tl = 0.0
        else:
            Tl = self.Tnom * load_torque_ratio

        return Tl
    
    def _calculate_electromagnetic_flux_angle(self, t, flux_angle_0):
        """Calculate instantaneous electromagnetic flux angle
        
        This method computes the electromagnetic flux angle based on time.

        Parameters
        ----------
        t : float
            Specific time [s].
        flux_angle_0 : float
            Initial electromagnetic flux angle [rad].

        Returns
        -------
        flux_angle_0 : float
            Magnetic flux angle [rad].
        """
        return flux_angle_0 + self.frequency * t
    
    def _ode_system(self, t, y, flux_angle_0, load_torque_entrance_time, load_torque_ratio):
        """
        Defines the Ordinary Differential Equation (ODE) system for the motor simulation.

        Calculates the time derivatives of the electromagnetic fluxes and rotor angular velocity 
        based on the input voltages, electrical torque, and mechanical load torque. 
        This method is designed to be passed directly to `scipy.integrate.solve_ivp`.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        y : array_like
            Current state vector containing the dynamic variables:
            [Lds, Lqs, Ldr, Lqr, wr], where:
            - Lds, Lqs: Stator flux linkages in the d and q axes.
            - Ldr, Lqr: Rotor flux linkages in the d and q axes.
            - wr: Rotor angular velocity.
        flux_angle_0 : float
            Initial electromagnetic flux angle [rad].
        load_torque_entrance_time : float
            Time at which the load torque is applied to the motor shaft [s].
        load_torque_ratio : float
            Load torque ratio applied at the entrance time.

        Returns
        -------
        dLds_dt, dLqs_dt, dLdr_dt, dLqr_dt, dwr_dt : tuple of float
            The time derivatives of the state variables in the corresponding order.
        """

        Lds, Lqs, Ldr, Lqr, wr = y

        Tl = self._calculate_load_torque(t, load_torque_entrance_time, load_torque_ratio)
        Te = self._calculate_electrical_torque(Lds, Ldr, Lqs, Lqr)

        # Electrical 3-phase tensions
        vas, vbs, vcs = self.sourceAC.get_phase_voltages(t)

        # Clarke & Park Transforms for Voltages
        v_alpha = 2 / 3 * (vas - vbs / 2 - vcs / 2)
        v_beta = 2 / 3 * (vbs - vcs) * np.sqrt(3) / 2

        flux_angle = self._calculate_electromagnetic_flux_angle(t, flux_angle_0)

        vds = v_alpha * np.cos(flux_angle) + v_beta * np.sin(flux_angle)
        vqs = -v_alpha * np.sin(flux_angle) + v_beta * np.cos(flux_angle)
        vdr, vqr = 0, 0

        dLds_dt = self._calculate_dLds_dt(Lds, Ldr, Lqs, vds)
        dLqs_dt = self._calculate_dLqs_dt(Lqs, Lds, Lqr, vqs)
        dLdr_dt = self._calculate_dLdr_dt(Ldr, Lds, Lqr, vdr, wr)
        dLqr_dt = self._calculate_dLqr_dt(Lqr, Lqs, Ldr, vqr, wr)
        dwr_dt = self._calculate_dwr_dt(Tl, Te, wr)

        return dLds_dt, dLqs_dt, dLdr_dt, dLqr_dt, dwr_dt

    @check_units
    def run_with_AC_source(self, 
        t,
        load_torque_entrance_time=None,
        load_torque_ratio=1.0,
        voltage_net=None,
        frequency_net=None,
        initial_phase_angle=0.0,
        harmonics=None,
        unbalances=None,
    ):
        """Run the motor simulation over a specified time vector.

        This method considers driving motor with AC source.

        Parameters
        ----------
        t : array-like
            Times at which to store the computed solution.
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
        results : ross.MotorResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.MotorResponseResults`
        """

        voltage_net = voltage_net or self.voltage_nom
        frequency_net = frequency_net or self.frequency

        self.sourceAC = SourceAC(
            voltage_net=voltage_net,
            frequency_net=Q_(frequency_net, "rad/s").to("Hz").m,
            initial_phase_angle=initial_phase_angle,
            harmonics=harmonics,
            unbalances=unbalances
        )

        # Initial values
        wr0 = 0.0  # Rotor's angular speed
        thetar0 = 0.0  # Rotor's angular position
        flux_angle_0 = self.initial_angle_net - np.pi / 2  # Flux initial angle

        # Initial alpha-beta and dq currents (based in nulled instantaneous phase currents)
        ias, ibs, ics = 0, 0, 0
        i_alpha = 2 / 3 * (ias - ibs / 2 - ics / 2)
        i_beta = 2 / 3 * (ibs - ics) * np.sqrt(3) / 2
        ids = i_alpha * np.cos(flux_angle_0) + i_beta * np.sin(flux_angle_0)
        iqs = -i_alpha * np.sin(flux_angle_0) + i_beta * np.cos(flux_angle_0)

        # Initial rotor and stator's inductances
        Lds0 = self.Lss * ids + self.Lm * 0
        Lqs0 = self.Lss * iqs + self.Lm * 0
        Ldr0 = self.Lrr * 0 + self.Lm * ids
        Lqr0 = self.Lrr * 0 + self.Lm * iqs

        # Get solution in time
        t = np.array(t)

        solution = sp.integrate.solve_ivp(
            fun=self._ode_system,
            t_span=(t[0], t[-1]),
            y0=(Lds0, Lqs0, Ldr0, Lqr0, wr0),
            args=(flux_angle_0, load_torque_entrance_time, load_torque_ratio),
            t_eval=t
        )

        Lds, Lqs, Ldr, Lqr, speed = solution.y

        # Compute outputs
        angular_position = thetar0 + (speed * self.n_poles / 2) * t
        
        load_torque = np.vectorize(self._calculate_load_torque)(t, load_torque_entrance_time, load_torque_ratio)
        electric_torque = np.vectorize(self._calculate_electrical_torque)(Lds, Ldr, Lqs, Lqr)

        flux_angle = np.vectorize(self._calculate_electromagnetic_flux_angle)(t, flux_angle_0)

        ids = self.a * Lds - self.c * Ldr
        iqs = self.a * Lqs - self.c * Lqr
        i_alpha = ids * np.cos(flux_angle) - iqs * np.sin(flux_angle)
        i_beta = ids * np.sin(flux_angle) + iqs * np.cos(flux_angle)
        ias = i_alpha
        ibs = -i_alpha / 2 + np.sqrt(3) * i_beta / 2
        ics = -i_alpha / 2 - np.sqrt(3) * i_beta / 2

        vas, vbs, vcs = np.vectorize(self.sourceAC.get_phase_voltages)(t)

        # Save currents and voltages in dictionary
        currents = dict()
        currents["a"] = ias
        currents["b"] = ibs
        currents["c"] = ics
        currents["alpha"] = i_alpha
        currents["beta"] = i_beta
        currents["d"] = ids
        currents["q"] = iqs

        voltages = dict()
        voltages["a"] = vas
        voltages["b"] = vbs
        voltages["c"] = vcs

        results = MotorResponseResults(
            t, electric_torque, load_torque, speed, currents, voltages
        )

        return results


def motor_example():
    """Create an example of motor element.

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

    tI = 0.0
    tF = 5.0
    dt = 1e-4
    t = np.arange(tI, tF, dt)

    results = motor.run_with_AC_source(
        t,
        load_torque_entrance_time=3.0,
        load_torque_ratio=1.5,
        voltage_net=90.0,
        harmonics={
            "enable": True,
            "orders": [5, 7],
            "amplitudes": [5, 5],
        },
        unbalances={
            "enable": False
        }
    )

    fig_torque = results.plot_torque()
    fig_speed = results.plot_speed()
    fig_currents = results.plot_phase_currents(reference_frame="a-b-c")
    fig_voltages = results.plot_phase_tensions()

    fig_torque.show()
    fig_speed.show()
    fig_currents.show()
    fig_voltages.show()
