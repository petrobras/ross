"""Electric Motor Element module.

This module defines the MotorElement class which represents a 3-phase Induction Electric Motor
simulated using a 4th-order Runge-Kutta method, considering magnetic fluxes and currents.
"""

import numpy as np
import scipy as sp

from ross.element import Element
from ross.units import Q_, check_units

from .sources import SourceAC
from .inverters import InverterVF
from .results import MotorResponseResults
from .utils import phase_to_line, clarke_transform, park_transform

__all__ = ["MotorElement", "motor_example"]


class MotorElement(Element):
    """Create a 3-phase induction motor element for rotordynamic analysis.

    This class represents a Three-Phase Induction Motor (TPIM) using a synchronous
    reference frame for rotor flux orientation.

    Parameters
    ----------
    n : int
        Node at which the motor is coupled.
    power_nom : float, pint.Quantity
        Nominal power [W].
    voltage_nom : float
        Nominal voltage [V].
    speed_nom : float, pint.Quantity
        Nominal machine speed [rad/s].
    stator_resistance : float
        Stator resistance [Ohm].
    rotor_resistance : float
        Rotor resistance [Ohm].
    stator_reactance : float
        Stator self-reactance at frequency [Ohm].
    rotor_reactance : float
        Rotor self-reactance at frequency [Ohm].
    mutual_reactance : float
        Mutual reactance at frequency [Ohm].
    Ip_motor : float, pint.Quantity
        Polar Moment of inertia related to motor axis [kg.m²].
    frequency_nom : float, optional, pint.Quantity
        Nominal frequency [rad/s].
        Default is 60 Hz.
    n_poles: int, optional
        Number of machine's poles.
        Default is 4.
    viscosity_coeff : float, optional, pint.Quantity
        Viscosity coefficient [Pa.s].
        Default is 0.
    Ip_load : float, optional, pint.Quantity
        Polar Moment of inertia related to load [kg.m²].
        Default is 0.
    voltage_net : float, optional
        Electrical tension of Power Supply [V].
        Default is None, which adopts the motor's nominal voltage (`voltage`).
    frequency_net : float, optional, pint.Quantity
        Electrical frequency of Power Supply [rad/s].
        Default is None, which adopts the motor's nominal frequency (`frequency`).
    initial_angle_net : float, optional, pint.Quantity
        Initial angular phase frequency of Power Supply [rad].
        Default is 20 degrees.
    short_circuit_ratio_net : float, optional
        Short-Circuit Ratio at common coupling point with power supply.
        Default is 50.
    XR_ratio_net : float, optional
        Reactance (X) / Resistance (R) Ratio at coupling point with power supply.
        Default is 80.
    tag : str, optional
        Tag to name the element.
        Default is None.

    Examples
    --------
    >>> from ross import MotorElement
    >>> from ross.units import Q_
    >>> motor = MotorElement(
    ...     n=0,
    ...     tag=None,
    ...     power_nom=Q_(1.5, "cv"),
    ...     voltage_nom=127,
    ...     speed_nom=Q_(1725, "RPM"),
    ...     frequency_nom=Q_(60.0, "Hz"),
    ...     n_poles=4,
    ...     stator_resistance=2.5,
    ...     rotor_resistance=1.8,
    ...     stator_reactance=1.3,
    ...     rotor_reactance=1.3,
    ...     mutual_reactance=43.08,
    ...     Ip_motor=0.0372,
    ...     viscosity_coeff=0.0,
    ...     Ip_load=0.0,
    ...     voltage_net=127,
    ...     frequency_net=Q_(60.0, "Hz"),
    ... )
    >>> motor.power_nom
    1103.248125
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
        frequency_nom=None,
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

        if frequency_nom is None:
            self.frequency_nom = Q_(60.0, "Hz").to("rad/s").m
        else:
            self.frequency_nom = float(frequency_nom)

        self.frequency_ref = self.frequency_nom

        # Numerical Validation of SCIP entries
        self.voltage_net = self.voltage if voltage_net is None else float(voltage_net)
        self.frequency_net = (
            self.frequency_nom if frequency_net is None else float(frequency_net)
        )
        self.short_circuit_ratio_net = float(short_circuit_ratio_net)
        self.XR_ratio_net = float(XR_ratio_net)

        if initial_angle_net is None:
            self.initial_angle_net = Q_(20.0, "deg").to("rad").m
        else:
            self.initial_angle_net = float(initial_angle_net)

        # Internal model inductances parameters derived from CEMP
        Lls = self.stator_reactance / self.frequency_nom
        Llr = self.rotor_reactance / self.frequency_nom
        self.Lm = self.mutual_reactance / self.frequency_nom
        self.Lss = Lls + self.Lm
        self.Lrr = Llr + self.Lm

        # Internal Electric Motor constants derived from NOMP and CEMP
        snom = (1 - self.speed_nom * self.n_poles / (2 * self.frequency_nom)) * 100
        wnom = (self.frequency_nom * (1 - snom / 100)) / (self.n_poles / 2)
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

        self.n = n
        self.tag = tag

    def __str__(self):
        """Convert object into string.

        Returns
        -------
        str
            The object's parameters translated to strings.
        """
        return (
            f"Tag:                                {self.tag}"
            f"\nNode:                               {self.n}"
            f"\n--- Nominal Parameters (NOMP) ---"
            f"\nNominal Power (W):                  {self.power_nom}"
            f"\nNominal Voltage (V):                {self.voltage_nom}"
            f"\nNominal Rotation (rad/s):           {self.speed_nom}"
            f"\nNominal Frequency (Hz):             {Q_(self.frequency_nom, 'rad/s').to('Hz').m}"
            f"\nNumber of Poles:                    {self.n_poles}"
            f"\n--- Circuit Parameters (CEMP) ---"
            f"\nStator Resistance (Ohm):            {self.stator_resistance}"
            f"\nRotor Resistance (Ohm):             {self.rotor_resistance}"
            f"\nStator Reactance (Ohm):             {self.stator_reactance}"
            f"\nRotor Reactance (Ohm):              {self.rotor_reactance}"
            f"\nMutual Reactance (Ohm):             {self.mutual_reactance}"
            f"\nMotor Inertia (kg.m²):              {self.Ip_motor}"
            f"\nViscosity Coefficient (Pa.s):       {self.viscosity_coeff}"
            f"\nLoad Inertia (kg.m²):               {self.Ip_load}"
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

    def _calculate_dLds_dt(self, Lds, Ldr, Lqs, vds, w_shaft):
        """Calculate time derivative of the d-axis stator inductance.

        This method computes the rate of change of the d-axis stator inductance using
        the motor's current state variables and the input d-axis voltage.

        Parameters
        ----------
        Lds : float
            d-axis stator inductance [H].
        Ldr : float
            d-axis rotor inductance [H].
        Lqs : float
            q-axis stator inductance [H].
        vds : float
            d-axis stator voltage [V].
        w_shaft : float
            Shaft angular speed [rad/s].

        Returns
        -------
        float
            Time derivative of the d-axis stator inductance [H/s].
        """
        R = self.stator_resistance + self.short_circuit_resistance
        w = w_shaft

        return vds - R * self.a * Lds + R * self.c * Ldr + w * Lqs

    def _calculate_dLqs_dt(self, Lqs, Lds, Lqr, vqs, w_shaft):
        """Calculate time derivative of the q-axis stator inductance.

        This method computes the rate of change of the q-axis stator inductance using
        the motor's current state variables and the input q-axis voltage.

        Parameters
        ----------
        Lqs : float
            q-axis stator inductance [H].
        Lds : float
            d-axis stator inductance [H].
        Lqr : float
            q-axis rotor inductance [H].
        vqs : float
            q-axis stator voltage [V].
        w_shaft : float
            Shaft angular speed [rad/s].

        Returns
        -------
        float
            Time derivative of the q-axis stator inductance [H/s].
        """
        R = self.stator_resistance + self.short_circuit_resistance
        w = w_shaft

        return vqs - R * self.a * Lqs + R * self.c * Lqr - w * Lds

    def _calculate_dLdr_dt(self, Ldr, Lds, Lqr, vdr, wr, w_shaft):
        """Calculate time derivative of the d-axis rotor inductance.

        This method computes the rate of change of the d-axis rotor inductance using
        the motor's current state variables and the input d-axis voltage.

        Parameters
        ----------
        Ldr : float
            d-axis rotor inductance [H].
        Lds : float
            d-axis stator inductance [H].
        Lqr : float
            q-axis rotor inductance [H].
        vdr : float
            d-axis rotor voltage [V].
        wr : float
            Rotor angular speed [rad/s].
        w_shaft : float
            Shaft angular speed [rad/s].

        Returns
        -------
        float
            Time derivative of the d-axis rotor inductance [H/s].
        """
        R = self.rotor_resistance
        w = w_shaft - wr * self.n_poles / 2

        return vdr - R * self.b * Ldr + R * self.c * Lds + w * Lqr

    def _calculate_dLqr_dt(self, Lqr, Lqs, Ldr, vqr, wr, w_shaft):
        """Calculate time derivative of the q-axis rotor inductance.

        This method computes the rate of change of the q-axis rotor inductance using
        the motor's current state variables and the input q-axis voltage.

        Parameters
        ----------
        Lqr : float
            q-axis rotor inductance [H].
        Lqs : float
            q-axis stator inductance [H].
        Ldr : float
            d-axis rotor inductance [H].
        vqr : float
            q-axis rotor voltage [V].
        wr : float
            Rotor angular speed [rad/s].
        w_shaft : float
            Shaft angular speed [rad/s].

        Returns
        -------
        float
            Time derivative of the q-axis rotor inductance [H/s].
        """
        R = self.rotor_resistance
        w = w_shaft - wr * self.n_poles / 2

        return vqr - R * self.b * Lqr + R * self.c * Lqs - w * Ldr

    def _calculate_dwr_dt(self, Tl, Te, wr):
        """Calculate the time derivative of rotor angular speed.

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
        float
            Time derivative of rotor angular speed [rad/s²].
        """
        J = self.Ip_motor + self.Ip_load

        return (Te - self.viscosity_coeff * wr - Tl) / J

    def _calculate_electrical_torque(self, Lds, Ldr, Lqs, Lqr):
        """Calculate the electric torque generated by the motor.

        Parameters
        ----------
        Lds : float
            d-axis stator inductance [H].
        Ldr : float
            d-axis rotor inductance [H].
        Lqs : float
            q-axis stator inductance [H].
        Lqr : float
            q-axis rotor inductance [H].

        Returns
        -------
        float
            Electrical torque [N.m].
        """
        return 1.5 * self.c * (Lqs * Ldr - Lds * Lqr) * self.n_poles / 2

    def _calculate_load_torque(self, t, load_torque_entrance_time, load_torque_ratio):
        """Calculate load torque as a function of time.

        Parameters
        ----------
        t : float
            Time point [s].
        load_torque_entrance_time : float
            Time at which load torque is applied to the motor shaft [s].
        load_torque_ratio : float
            Load torque ratio applied at the entrance time.

        Returns
        -------
        Tl : float
            Load torque at time `t` [N.m].
        """
        if t < load_torque_entrance_time:
            Tl = 0.0
        else:
            Tl = self.Tnom * load_torque_ratio

        return Tl

    def _calculate_electromagnetic_flux_angle(self, t, flux_angle_0, w_shaft):
        """Calculate electromagnetic flux angle at time point.

        Parameters
        ----------
        t : float
            Time point [s].
        flux_angle_0 : float
            Initial electromagnetic flux angle [rad].
        w_shaft : float
            Shaft angular speed [rad/s].

        Returns
        -------
        float
            Magnetic flux angle at time `t` [rad].
        """
        return flux_angle_0 + w_shaft * t

    def _ode_system(
        self, t, y, flux_angle_0, load_torque_entrance_time, load_torque_ratio, element
    ):
        """Define the ODE system for motor simulation.

        Computes time derivatives of electromagnetic flux linkages and rotor angular
        velocity based on input voltages, electrical torque, and load torque.
        Designed for use with `scipy.integrate.solve_ivp`.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        y : array_like
            State vector [Lds, Lqs, Ldr, Lqr, wr] where:

            - Lds, Lqs : d- and q-axis stator flux linkages [H]
            - Ldr, Lqr : d- and q-axis rotor flux linkages [H]
            - wr : Rotor angular velocity [rad/s]

        flux_angle_0 : float
            Initial electromagnetic flux angle [rad].
        load_torque_entrance_time : float
            Time at which the load torque is applied to the motor shaft [s].
        load_torque_ratio : float
            Load torque ratio applied at the entrance time.

        Returns
        -------
        tuple of float
            Time derivatives `(dLds_dt, dLqs_dt, dLdr_dt, dLqr_dt, dwr_dt)`.
        """

        Lds, Lqs, Ldr, Lqr, wr = y

        Tl = self._calculate_load_torque(
            t, load_torque_entrance_time, load_torque_ratio
        )
        Te = self._calculate_electrical_torque(Lds, Ldr, Lqs, Lqr)

        w_shaft = element.get_frequency(t)

        # Electrical 3-phase voltages
        vas, vbs, vcs = element.get_phase_voltages(t, w_shaft)

        # Clarke & Park Transforms for Voltages
        v_alpha, v_beta = clarke_transform(vas, vbs, vcs)

        flux_angle = self._calculate_electromagnetic_flux_angle(
            t, flux_angle_0, w_shaft
        )
        vds, vqs = park_transform(v_alpha, v_beta, flux_angle)
        vdr, vqr = 0, 0

        dLds_dt = self._calculate_dLds_dt(Lds, Ldr, Lqs, vds, w_shaft)
        dLqs_dt = self._calculate_dLqs_dt(Lqs, Lds, Lqr, vqs, w_shaft)
        dLdr_dt = self._calculate_dLdr_dt(Ldr, Lds, Lqr, vdr, wr, w_shaft)
        dLqr_dt = self._calculate_dLqr_dt(Lqr, Lqs, Ldr, vqr, wr, w_shaft)
        dwr_dt = self._calculate_dwr_dt(Tl, Te, wr)

        return dLds_dt, dLqs_dt, dLdr_dt, dLqr_dt, dwr_dt

    def _run(
        self,
        t,
        load_torque_entrance_time=None,
        load_torque_ratio=1.0,
        element=None,
    ):
        """Run motor generical simulation for specified time points.

        Parameters
        ----------
        t : array-like
            Time points to store the computed solution [s].
        load_torque_entrance_time : float, optional
            Time when load torque is applied to the motor shaft [s].
            Default is half the simulation time.
        load_torque_ratio : float, optional
            Load torque ratio applied at the entrance time. This is a multiplier
            for the nominal load torque, e.g., a value of 1.0 applies 100% of the
            nominal torque at entrance time. Default is 1.0.

        Returns
        -------
        results : ross.MotorResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.MotorResponseResults`
        """

        if element is None:
            raise ValueError(
                "An element providing the 'get_phase_voltages(t)' method must be provided for simulation."
            )

        # Initial values
        wr0 = 0.0  # Rotor's angular speed
        thetar0 = 0.0  # Rotor's angular position
        flux_angle_0 = self.initial_angle_net - np.pi / 2  # Flux initial angle

        # Initial alpha-beta and dq currents (based in nulled instantaneous phase currents)
        ias, ibs, ics = 0, 0, 0
        i_alpha, i_beta = clarke_transform(ias, ibs, ics)
        ids, iqs = park_transform(i_alpha, i_beta, flux_angle_0)

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
            args=(flux_angle_0, load_torque_entrance_time, load_torque_ratio, element),
            t_eval=t,
        )

        Lds, Lqs, Ldr, Lqr, speed = solution.y

        # Compute outputs
        angular_position = thetar0 + (speed * self.n_poles / 2) * t

        load_torque = np.vectorize(self._calculate_load_torque)(
            t, load_torque_entrance_time, load_torque_ratio
        )

        electric_torque = np.vectorize(self._calculate_electrical_torque)(
            Lds, Ldr, Lqs, Lqr
        )

        w_shaft = np.vectorize(element.get_frequency)(t)

        flux_angle = np.vectorize(self._calculate_electromagnetic_flux_angle)(
            t, flux_angle_0, w_shaft
        )

        vas, vbs, vcs = np.vectorize(element.get_phase_voltages)(t, w_shaft)

        voltages = dict()
        voltages["a"] = vas
        voltages["b"] = vbs
        voltages["c"] = vcs

        ids = self.a * Lds - self.c * Ldr
        iqs = self.a * Lqs - self.c * Lqr
        i_alpha = ids * np.cos(flux_angle) - iqs * np.sin(flux_angle)
        i_beta = ids * np.sin(flux_angle) + iqs * np.cos(flux_angle)
        ias = i_alpha
        ibs = -i_alpha / 2 + np.sqrt(3) * i_beta / 2
        ics = -i_alpha / 2 - np.sqrt(3) * i_beta / 2

        currents = dict()
        currents["a"] = ias
        currents["b"] = ibs
        currents["c"] = ics
        currents["alpha"] = i_alpha
        currents["beta"] = i_beta
        currents["d"] = ids
        currents["q"] = iqs

        results = MotorResponseResults(
            t, electric_torque, load_torque, speed, currents, voltages
        )

        return results

    @check_units
    def run_with_AC_source(
        self,
        t,
        load_torque_entrance_time=None,
        load_torque_ratio=1.0,
        voltage_net=None,
        frequency_net=None,
        initial_phase_angle=0.0,
        harmonics=None,
        unbalances=None,
    ):
        """Run motor simulation with AC source for specified time points.

        Parameters
        ----------
        t : array-like
            Time points to store the computed solution [s].
        load_torque_entrance_time : float, optional
            Time when load torque is applied to the motor shaft [s].
            Default is half the simulation time.
        load_torque_ratio : float, optional
            Load torque ratio applied at the entrance time. This is a multiplier
            for the nominal load torque, e.g., a value of 1.0 applies 100% of the
            nominal torque at entrance time. Default is 1.0.
        voltage_net : float, pint.Quantity, optional
            Power supply voltage [V]. If None, uses motor nominal voltage.
        frequency_net : float, pint.Quantity, optional
            Power supply frequency [rad/s]. If None, uses motor nominal frequency.
        initial_phase_angle : float, pint.Quantity, optional
            Initial power supply phase angle [rad]. Default is 0.
        harmonics : dict, optional
            Harmonic configuration with keys:

            - "orders" : list of int
                Harmonic orders (e.g., [5, 7])
            - "amplitudes" : list of float
                Harmonic amplitudes as % of nominal voltage
            - "enable" : bool
                Enable harmonics

        unbalances : dict, optional
            Unbalance configuration with keys:

            - "voltage_percent" : list of float
                Voltage magnitude unbalance per phase [%]
            - "angle_deviation" : list of float, pint.Quantity
                Angle deviation per phase [rad]
            - "enable" : bool
                Enable unbalances

        Returns
        -------
        results : ross.MotorResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.MotorResponseResults`

        Examples
        --------
        >>> motor = motor_example()
        >>> t = np.arange(0, 1, 1e-3)
        >>> results = motor.run_with_AC_source(
        ...     t,
        ...     load_torque_entrance_time=3.0,
        ...     load_torque_ratio=1.5,
        ...     initial_phase_angle=0.0,
        ...     harmonics={
        ...         "enable": True,
        ...         "orders": [5, 7],
        ...         "amplitudes": [5, 5],
        ...     },
        ...     unbalances={"enable": False},
        ... )
        >>> fig_torque = results.plot_torque()
        >>> fig_speed = results.plot_speed()
        >>> fig_currents = results.plot_phase_currents(reference_frame="a-b-c")
        >>> fig_voltages = results.plot_phase_voltages()
        """
        # Creating AC source instance
        if harmonics:
            harmonic_orders = harmonics.get("orders")
            harmonic_amplitudes = harmonics.get("amplitudes")
        else:
            harmonic_orders, harmonic_amplitudes = None, None

        if unbalances:
            unbalance_voltage_percent = unbalances.get("voltage_percent")
            unbalance_angle_deviation = unbalances.get("angle_deviation")
        else:
            unbalance_voltage_percent, unbalance_angle_deviation = None, None

        source = SourceAC(
            voltage_net=voltage_net or self.voltage_nom,
            frequency_net=frequency_net or self.frequency_nom,
            initial_phase_angle=initial_phase_angle,
            harmonic_orders=harmonic_orders,
            harmonic_amplitudes=harmonic_amplitudes,
            harmonic_enable=harmonics.get("enable") if harmonics else False,
            unbalance_voltage_percent=unbalance_voltage_percent,
            unbalance_angle_deviation=unbalance_angle_deviation,
            unbalance_enable=unbalances.get("enable") if unbalances else False,
        )

        results = self._run(
            t,
            load_torque_entrance_time=load_torque_entrance_time,
            load_torque_ratio=load_torque_ratio,
            element=source,
        )

        return results

    @check_units
    def run_with_inverter(
        self,
        t,
        load_torque_entrance_time=None,
        load_torque_ratio=1.0,
        frequency_s=None,
        time_ramp=1.0,
        frequency_ref=None,
    ):
        """Simulate motor with variable frequency inverter control.

        Runs motor simulation with a three-phase voltage source inverter employing
        Space Vector PWM modulation and scalar V/f speed control. The inverter
        generates three-phase voltages based on the reference frequency.

        Parameters
        ----------
        t : array-like
            Time points to store the computed solution [s].
        load_torque_entrance_time : float, optional
            Time when load torque is applied to the motor shaft [s].
            Default is None (no load applied).
        load_torque_ratio : float, optional
            Load torque ratio applied at the entrance time. This is a multiplier
            for the nominal load torque, e.g., a value of 1 applies 100% of the
            nominal torque. Default is 1.
        frequency_s : float or pint.Quantity
            IGBT switching frequency [rad/s].
        time_ramp : float, optional
            Acceleration ramp time [s] for frequency ramping. Default is 1.
        frequency_ref : float or pint.Quantity, optional
            Reference frequency for V/f control [rad/s]. If None, uses half the
            motor nominal frequency. Default is None.

        Returns
        -------
        results : ross.MotorResponseResults
            Motor simulation results containing time-domain response data.
            For more information on attributes and methods available see:
            :py:class:`ross.MotorResponseResults`

        Examples
        --------
        >>> motor = motor_example()
        >>> t = np.arange(0, 1, 1e-3)
        >>> results = motor.run_with_inverter(
        ...     t,
        ...     load_torque_entrance_time=2.5,
        ...     load_torque_ratio=1.0,
        ...     frequency_s=Q_(5000, "Hz"),
        ...     time_ramp=0.6667,
        ...     frequency_ref=Q_(30.0, "Hz")
        ... )
        >>> results.speed.shape
        (1000,)
        """

        frequency_ref = float(frequency_ref) or self.frequency_nom / 2

        inverter = InverterVF(
            voltage_dc=1.35 * phase_to_line(self.voltage_nom),  # 1.35?
            frequency_s=frequency_s,
            voltage_nom=phase_to_line(self.voltage_nom),
            frequency_nom=self.frequency_nom,
            time_ramp=time_ramp,
            frequency_ref=frequency_ref,
        )

        results = self._run(
            t,
            load_torque_entrance_time=load_torque_entrance_time,
            load_torque_ratio=load_torque_ratio,
            element=inverter,
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

    Examples
    --------
    >>> motor = motor_example()
    >>> motor.power_nom
    1103.248125
    """

    return MotorElement(
        n=0,
        tag=None,
        power_nom=Q_(1.5, "cv"),
        voltage_nom=127,
        speed_nom=Q_(1710, "RPM"),
        frequency_nom=Q_(60.0, "Hz"),
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
