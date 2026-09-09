"""Simulate three-phase voltage source inverters.

This module provides implementations for modeling and simulating three-phase VSIs
using Space Vector PWM (SVPWM) modulation, either with scalar V/f speed control
(:class:`InverterVF`) or with indirect Field-Oriented Control (:class:`InverterFOC`).

References
----------
Wu, B. & Narimani, M. (2016). High-Power Converters and AC Drives. Wiley.
Novotny, D. & Lipo, T. (1996). Vector Control and Dynamics of AC Drives.
"""

import numpy as np

from ross.units import check_units

from .utils import clarke_transform, inverse_clarke_transform, park_transform


class Inverter:
    """Base class for three-phase voltage source inverters using SVPWM.

    Provides the shared Space Vector PWM modulator used by
    :class:`InverterVF` and :class:`InverterFOC`. Subclasses implement
    speed control and the triangular carrier via `_get_carrier`.

    Parameters
    ----------
    voltage_dc : float
        DC link voltage [V].
    frequency_s : float or pint.Quantity
        IGBT switching frequency [rad/s].
    voltage_nom : float
        Nominal line voltage [V].
    frequency_nom : float or pint.Quantity
        Nominal operating frequency [rad/s].
    time_ramp : float, optional
        Acceleration ramp time [s]. Default is 0.6667.
    frequency_ref : float or pint.Quantity, optional
        Reference electrical frequency [rad/s]. Default is half the
        nominal frequency.
    """

    def __init__(
        self,
        voltage_dc,
        frequency_s,
        voltage_nom,
        frequency_nom,
        time_ramp=0.6667,
        frequency_ref=None,
    ):

        self.voltage_dc = voltage_dc
        self.frequency_s = frequency_s
        self.Ts = 2 * np.pi / frequency_s

        self.voltage_nom = float(voltage_nom)
        self.frequency_nom = float(frequency_nom)
        self.time_ramp = float(time_ramp)
        self.frequency_ref = float(frequency_ref or frequency_nom / 2)

        # Switching SVPWM table, shared by every inverter implemented in this module.
        # Each column represents the states of the upper switches for the space
        # vectors V0, V1, V3, V2, V6, V4, V5 and V7
        self.sw_table = np.array(
            [
                [0, 1, 1, 0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ]
        )

        # Active vectors according to the sector, shared by every inverter implemented
        # in this module.
        self.actv_vet = np.array([[2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 2]])

        self.V0 = 1
        self.V7 = 8

    def _svpwm(self, t, va_ref, vb_ref, vc_ref):
        """Synthesize phase voltages via Space Vector PWM modulation.

        Parameters
        ----------
        t : float
            Time [s].
        va_ref, vb_ref, vc_ref : float
            Reference three-phase voltages [V].

        Returns
        -------
        van, vbn, vcn : float
            Instantaneous phase voltages [V].
        """
        v_alpha, v_beta = clarke_transform(va_ref, vb_ref, vc_ref)

        # Space vector and SVPWM hexagon sector
        vr = np.sqrt(v_alpha**2 + v_beta**2)
        theta = np.arctan2(v_beta, v_alpha)
        if theta < 0:
            theta += 2 * np.pi

        S = int(np.floor(theta / (np.pi / 3))) + 1
        S = max(1, min(S, 6))

        # Angle within the sector
        theta_k = theta - (S - 1) * np.pi / 3

        # Modulation index and dwell times
        M = (np.sqrt(3) * vr) / self.voltage_dc

        T1 = self.Ts * M * np.sin(np.pi / 3 - theta_k)
        T2 = self.Ts * M * np.sin(theta_k)
        T0 = self.Ts - T1 - T2

        eps = np.finfo(float).eps

        # Correction for overmodulation conditions
        if M > 0.907 and M <= 1:
            # Region I (0.907 < M ≤ 1)
            T0 = max(T0, 0)
            factor = self.Ts / (T0 + T1 + T2 + eps)
            T1 *= factor
            T2 *= factor
            T0 = self.Ts - T1 - T2
        elif M > 1 and M <= 1.1547:
            # Region II (1 < M ≤ 1.1547)
            T0 = 0
            factor = self.Ts / (T1 + T2 + eps)
            T1 *= factor
            T2 *= factor
        elif M > 1.1547:
            # Region III (Six-Step)
            T0 = 0
            if theta_k <= (np.pi / 3) / 2:
                T1, T2 = self.Ts, 0
            else:
                T1, T2 = 0, self.Ts

        # Normalization - ensures all values are non-negative
        T1 = max(T1, 0)
        T2 = max(T2, 0)
        T0 = max(T0, 0)

        sumT = T1 + T2 + T0

        if abs(sumT - self.Ts) > 1e-12:
            if sumT > 0:
                T1 *= self.Ts / sumT
                T2 *= self.Ts / sumT
                T0 = self.Ts - T1 - T2
            else:
                # Extreme case: zero vector only
                T1, T2, T0 = 0, 0, self.Ts

        # Sequence and timing of vector application
        vetor_seq = (
            np.array(
                [self.V0, self.actv_vet[S - 1, 0], self.actv_vet[S - 1, 1], self.V7]
            )
            - 1
        )

        # Switching states
        S_bits = self.sw_table[:, vetor_seq]

        # Duty cycles - clamping values within the [0, 1] interval
        t_seq = np.array([T0 / 2, T1, T2, T0 / 2])
        D = np.clip((S_bits @ t_seq) / self.Ts, 0, 1)

        # Carrier synchronized with the switching period
        carrier = self._get_carrier(t)

        # Thresholds derived from duty cycles
        Ref = 2 * D - 1

        # Comparisons to determine switching states
        Ss = (carrier <= Ref).astype(float)

        # Pole voltages
        vao, vbo, vco = (2 * Ss - 1) * (self.voltage_dc / 2)

        # Phase voltages
        van = (2 / 3) * vao - (1 / 3) * (vbo + vco)
        vbn = (2 / 3) * vbo - (1 / 3) * (vao + vco)
        vcn = (2 / 3) * vco - (1 / 3) * (vbo + vao)

        return van, vbn, vcn

    def _get_carrier(self, t):
        """Return the continuous triangular PWM carrier at time `t`.

        Parameters
        ----------
        t : float
            Current simulation time [s].

        Returns
        -------
        float
            Carrier value in the [-1, 1] interval.
        """
        if self.Ts == 0:
            u = 0
        else:
            u = (t % self.Ts) / self.Ts

        return 1 - 4 * np.abs(u - 0.5)


class InverterVF(Inverter):
    """Simulate a three-phase voltage source inverter with V/f adjustment technique.

    This class implements a three-phase VSI using Space Vector PWM (SVPWM)
    modulation with V/f adjustment technique. The inverter generates three-phase
    output voltages based on a reference frequency and DC link voltage.

    Parameters
    ----------
    voltage_dc : float
        DC link voltage [V].
    frequency_s : float or pint.Quantity
        IGBT switching frequency [rad/s].
    voltage_nom : float
        Nominal line voltage [V].
    frequency_nom : float or pint.Quantity
        Nominal operating frequency [rad/s].
    time_ramp : float, optional
        Acceleration ramp time [s] for frequency ramping. Default is 0.6667.
    frequency_ref : float or pint.Quantity, optional
        Reference frequency for V/f adjustment [rad/s]. Default is half
        the nominal frequency.

    References
    ----------
    Wu, B. & Narimani, M. (2016). High-Power Converters and AC Drives. Wiley.

    Examples
    --------
    >>> from ross.units import Q_

    >>> inverter = InverterVF(
    ...     voltage_dc=300, frequency_s=Q_(5000, "Hz"),
    ...     voltage_nom=220, frequency_nom=Q_(60, "Hz"),
    ...     time_ramp=1, frequency_ref=Q_(90, "Hz"),
    ... )

    >>> freq = Q_(100, "Hz").to("rad/s").m
    >>> Vp = inverter.speed_control(frequency=freq)
    >>> float(np.round(Vp, 2))
    179.63

    >>> van, vbn, vcn = inverter.get_phase_voltages(t=0.001, frequency=freq)
    >>> np.round([van, vbn, vcn], 2)
    array([ 100., -200.,  100.])

    >>> f = inverter.get_frequency(t=0.05, frequency_ref=freq)
    >>> float(np.round(f, 2))
    18.85

    >>> f, van, vbn, vcn = inverter.get_current_state(t=0.5, frequency_ref=freq)
    >>> np.round([f, van, vbn, vcn], 2)
    array([188.5,   0. ,   0. ,   0. ])
    """

    @check_units
    def __init__(
        self,
        voltage_dc,
        frequency_s,
        voltage_nom,
        frequency_nom,
        time_ramp=0.6667,
        frequency_ref=None,
    ):

        super().__init__(
            voltage_dc,
            frequency_s,
            voltage_nom,
            frequency_nom,
            time_ramp,
            frequency_ref,
        )

        # Nominal phase voltage peak value
        self.voltage_phase_peak_nom = (voltage_nom / np.sqrt(3)) * np.sqrt(2)

        self.f_0 = 0.0

    def speed_control(self, frequency):
        """Calculate the phase voltage peak.

        Computes the peak voltage for the phase voltages based on the V/f ratio,
        ensuring proportional control between voltage and frequency.

        Parameters
        ----------
        frequency : float
            Operating frequency [rad/s].

        Returns
        -------
        Vp : float
            Peak phase voltage [V], saturated at nominal value.
        """
        # Peak value of the phase voltage proportional to the V/f ratio
        Vp = self.voltage_phase_peak_nom * (frequency / self.frequency_nom)

        # Saturation at the nominal value
        Vp = min(Vp, self.voltage_phase_peak_nom)
        return Vp

    def get_phase_voltages(self, t, frequency, theta_0):
        """Generate three-phase voltages using SVPWM modulation.

        Computes the instantaneous phase voltages (A, B, C) using Space Vector
        PWM modulation based on the reference frequency and switching configuration.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        frequency : float
            Operating frequency [rad/s].
        theta_0 : float
            Initial flux angle [rad].

        Returns
        -------
        van : float
            Phase A voltage with respect to neutral [V].
        vbn : float
            Phase B voltage with respect to neutral [V].
        vcn : float
            Phase C voltage with respect to neutral [V].
        """
        # Reference peak voltage
        Vp = self.speed_control(frequency)

        # Reference voltages
        theta = theta_0 + frequency * t
        va_ref = Vp * np.sin(theta)
        vb_ref = Vp * np.sin(theta - 2 * np.pi / 3)
        vc_ref = Vp * np.sin(theta + 2 * np.pi / 3)

        van, vbn, vcn = self._svpwm(t, va_ref, vb_ref, vc_ref)

        return van, vbn, vcn

    def get_frequency(self, t, frequency_ref=None):
        """Calculate the current operating frequency with acceleration ramp.

        Computes the instantaneous frequency accounting for the acceleration ramp,
        which linearly increases frequency from zero to the reference value over the
        ramp time.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        frequency_ref : float, optional
            Reference frequency [rad/s]. If None, uses `self.frequency_ref`.

        Returns
        -------
        f_curr : float
            Current operating frequency [rad/s].
        """
        if frequency_ref is None:
            frequency_ref = self.frequency_ref

        fref = min(max(frequency_ref, 0), self.frequency_nom)

        f_curr = self.f_0 + fref / self.time_ramp * t
        f_curr = min(f_curr, fref)

        return f_curr

    def get_current_state(self, t, theta_0, frequency_ref=None):
        """Get the fundamental frequency and phase voltages at time `t`.

        Parameters
        ----------
        t : float
            Time [s].
        theta_0 : float
            Initial flux angle [rad].
        frequency_ref : float, optional
            Reference frequency [rad/s]. If None, uses `self.frequency_ref`.

        Returns
        -------
        frequency : float
            Fundamental frequency [rad/s].
        van : float
            Phase A voltage with respect to neutral [V].
        vbn : float
            Phase B voltage with respect to neutral [V].
        vcn : float
            Phase C voltage with respect to neutral [V].
        """
        freq = self.get_frequency(t, frequency_ref)
        van, vbn, vcn = self.get_phase_voltages(t, freq, theta_0)

        return freq, van, vbn, vcn


class InverterFOC(Inverter):
    """Simulate a three-phase voltage source inverter with indirect Field-Oriented Control.

    This class implements a three-phase VSI operating under indirect
    Field-Oriented Control (iFOC), synthesizing the output voltages through
    Space Vector PWM (SVPWM), analogous to :class:`InverterVF` but replacing
    scalar V/f control with closed-loop rotor-flux-oriented current control.

    Unlike :class:`InverterVF` and :class:`SourceAC`,
    this inverter is a closed-loop element: its output voltages depend on
    the instantaneous rotor speed and stator currents fed back from the
    motor at every time step. Therefore it cannot be pre-computed
    independently of the motor's state; it must be driven from within the
    motor's time-stepping loop through ``get_current_state``, one step at a
    time, with rotor speed and stator current feedback.

    Parameters
    ----------
    voltage_dc : float
        DC link voltage [V].
    frequency_s : float or pint.Quantity
        IGBT switching frequency [rad/s].
    voltage_nom : float
        Nominal line voltage [V].
    frequency_nom : float or pint.Quantity
        Nominal (synchronous) electrical frequency [rad/s].
    n_poles : int
        Number of machine poles.
    speed_nom : float or pint.Quantity
        Nominal mechanical speed [rad/s].
    torque_nom : float
        Nominal load torque [N.m].
    stator_resistance : float
        Stator resistance [Ohm].
    rotor_resistance : float
        Rotor resistance [Ohm].
    stator_reactance : float
        Stator leakage reactance at nominal frequency [Ohm].
    rotor_reactance : float
        Rotor leakage reactance at nominal frequency [Ohm].
    mutual_reactance : float
        Magnetizing (mutual) reactance at nominal frequency [Ohm].
    Ip_motor : float
        Rotor polar moment of inertia [kg.m²].
    time_ramp : float, optional
        Acceleration ramp time [s] for the mechanical speed reference.
        Default is 1.
    frequency_ref : float or pint.Quantity, optional
        Synchronous electrical frequency reference [rad/s]. Default is
        half the nominal frequency.

    Notes
    -----
    Gains for the speed and current PI controllers are computed with the
    bandwidth method, mirroring the design used for scalar control.

    References
    ----------
    Wu, B. & Narimani, M. (2016). High-Power Converters and AC Drives. Wiley.
    Novotny, D. & Lipo, T. (1996). Vector Control and Dynamics of AC Drives.
    """

    @check_units
    def __init__(
        self,
        voltage_dc,
        frequency_s,
        voltage_nom,
        frequency_nom,
        n_poles,
        speed_nom,
        torque_nom,
        stator_resistance,
        rotor_resistance,
        stator_reactance,
        rotor_reactance,
        mutual_reactance,
        Ip_motor,
        time_ramp=1.0,
        frequency_ref=None,
    ):

        super().__init__(
            voltage_dc,
            frequency_s,
            voltage_nom,
            frequency_nom,
            time_ramp,
            frequency_ref,
        )

        self.np_pairs = n_poles / 2
        self.speed_nom = float(speed_nom)

        self.Rs = float(stator_resistance)
        self.Rr = float(rotor_resistance)

        Lls = float(stator_reactance) / self.frequency_nom
        Llr = float(rotor_reactance) / self.frequency_nom
        self.Lm = float(mutual_reactance) / self.frequency_nom
        self.Lss = Lls + self.Lm
        self.Lrr = Llr + self.Lm

        self.taur = self.Lrr / self.Rr

        # Modified equivalent circuit rotor resistance
        nominal_slip = 1 - self.np_pairs * self.speed_nom / self.frequency_nom
        Rr_eq = (self.Rr * (self.Lm / self.Lrr) ** 2) / nominal_slip

        # Auxiliary variables for calculating Zeq
        Ls1 = self.Lss - (self.Lm**2 / self.Lrr)
        Lm1 = (self.Lm**2) / self.Lrr

        # Equivalent impedance
        Zeq = (self.Rs + 1j * self.frequency_nom * Ls1) + (
            1j * self.frequency_nom * Lm1 * Rr_eq
        ) / (1j * self.frequency_nom * Lm1 + Rr_eq)

        Vpeak = self.voltage_nom * (np.sqrt(2) / np.sqrt(3))
        Is = Vpeak / Zeq

        # Modified equivalent circuit voltage Er
        Er = Vpeak - (self.Rs + 1j * self.frequency_nom * Ls1) * Is
        self.ids_ref = abs(Er) / (self.frequency_nom * Lm1)

        # Nominal stator current magnitude
        self.Is_nom = abs(Is)

        # PI controller gains (bandwidth method)
        BWp_iqs = frequency_s / 8
        BWp_ids = frequency_s / 8
        BWi_ids = frequency_s / 8
        BWp_w = BWi_ids / 8
        BWi_w = BWp_w / 8

        KL = float(torque_nom) / (self.np_pairs * self.speed_nom)  # Load constant
        J = float(Ip_motor)

        # Proportional and integral controller gains
        self.kp_w = J * 2 * np.pi * (BWp_w + BWi_w) - KL
        self.ki_w = J * 4 * (np.pi**2) * BWp_w * BWi_w
        self.kp_ids = self.Lss * 2 * np.pi * (BWp_ids + BWi_ids)
        self.ki_ids = self.Lss * 4 * (np.pi**2) * BWp_ids * BWi_ids

        self.Lsline = (
            self.Lss - (self.Lm**2) / self.Lrr
        )  # Equivalent leakage inductance
        self.kp_iqs = 2 * np.pi * self.Lsline * BWp_iqs

    def initialize_control_state(self, dt, theta_0=0.0):
        """Initialize integrator states and the discrete PWM carrier.

        Stores the closed-loop runtime state in `self.control_state`.
        Must be called before the first time step of a closed-loop
        simulation.

        Parameters
        ----------
        dt : float
            Simulation time step [s], used to set the number of samples per
            switching period.
        theta_0 : float, optional
            Initial flux angle [rad]. Default is 0.
        """
        self.control_state = {
            "int_err_w": 0.0,
            "int_err_ids": 0.0,
            "theta": theta_0,
            "carrier_index": 1,
            "n_samples": max(1, int(round(self.Ts / dt))),
        }

    def _get_carrier(self, t):
        """Return the discrete triangular PWM carrier.

        The carrier is sampled once per simulation step and advanced by
        the internal counter stored in `control_state["carrier_index"]`. The time
        argument is unused and kept for interface compatibility with `_svpwm`.

        Parameters
        ----------
        t : float
            Current simulation time [s] (unused).

        Returns
        -------
        float
            Carrier value in the [-1, 1] interval.
        """
        state = self.control_state
        n_in_period = (state["carrier_index"] - 1) % state["n_samples"]
        u = 0 if state["n_samples"] == 1 else n_in_period / (state["n_samples"] - 1)

        state["carrier_index"] += 1

        return 1 - 4 * np.abs(u - 0.5)

    def speed_control(self, t, frequency_ref):
        """Compute the mechanical speed reference with acceleration ramp.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        frequency_ref : float, optional
            Desired synchronous electrical frequency reference [rad/s].
            If None, uses `self.frequency_ref`.

        Returns
        -------
        w_ramp : float
            Ramped mechanical speed reference [rad/s].
        """
        if frequency_ref is None:
            frequency_ref = self.frequency_ref

        wref_mech = frequency_ref / self.np_pairs
        wref = min(max(wref_mech, 0), self.speed_nom)

        w_ramp = wref / self.time_ramp * t
        w_ramp = min(w_ramp, wref)

        return w_ramp

    def get_current_state(self, t, dt, rotor_speed, ia, ib, ic, frequency_ref=None):
        """Run the iFOC control loop and synthesize phase voltages via SVPWM.

        Computes the instantaneous phase voltages (A, B, C) from the rotor
        speed and stator current feedback, using indirect Field-Oriented
        Control followed by Space Vector PWM modulation.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        dt : float
            Time step [s].
        rotor_speed : float
            Measured (feedback) rotor mechanical speed [rad/s].
        ia, ib, ic : float
            Measured (feedback) stator phase currents [A].
        frequency_ref : float, optional
            Desired synchronous electrical frequency reference [rad/s].
            If None, uses `self.frequency_ref`.

        Returns
        -------
        w_sync : float
            Synchronous electrical speed [rad/s].
        van : float
            Phase A voltage with respect to neutral [V].
        vbn : float
            Phase B voltage with respect to neutral [V].
        vcn : float
            Phase C voltage with respect to neutral [V].
        """
        state = self.control_state

        wref = self.speed_control(t, frequency_ref)
        err_w = wref - rotor_speed

        u_prop = self.kp_w * err_w
        u_int = self.ki_w * state["int_err_w"]
        iqs_ref_unsat = u_prop + u_int

        # Torque-producing (q-axis) current limit
        iqs_max = 3.0 * self.Is_nom
        iqs_ref = np.clip(iqs_ref_unsat, -iqs_max, iqs_max)

        # Slip frequency
        wsl = (1 / self.taur) * (iqs_ref / self.ids_ref)

        # Synchronous electrical speed
        w_sync = wsl + self.np_pairs * rotor_speed

        state["theta"] += w_sync * dt

        # q aligned with cosine (torque), d aligned with sine (flux), matching the iFOC
        # convention: d_std = iqs, q_std = -ids
        i_alpha, i_beta = clarke_transform(ia, ib, ic)
        d_std, q_std = park_transform(i_alpha, i_beta, state["theta"])

        iqs, ids = d_std, -q_std

        err_iqs = iqs_ref - iqs
        vqs_ref = self.kp_iqs * err_iqs + self.Rs * iqs + self.Lss * w_sync * ids

        err_ids = self.ids_ref - ids
        state["int_err_ids"] += err_ids * dt

        vds_ref = (
            self.kp_ids * err_ids
            + self.ki_ids * state["int_err_ids"]
            + self.Rs * ids
            - self.Lsline * w_sync * iqs
        )

        # Saturation & anti-windup
        Vmax = self.voltage_dc / np.sqrt(3)
        Vref = np.sqrt(vqs_ref**2 + vds_ref**2)

        saturated = Vref > Vmax

        if saturated:
            scale = Vmax / Vref
            vqs_ref *= scale
            vds_ref *= scale

        # Anti-windup: freeze the speed-loop integrator whenever either the
        # current limit or the voltage limit is actively clipping the output
        # in the same direction as the error (otherwise the integral term
        # would keep growing without any corresponding effect on the actual
        # torque/voltage applied to the motor).
        current_saturated = abs(iqs_ref_unsat) > iqs_max

        any_saturated = saturated or current_saturated
        if (not any_saturated) or (np.sign(err_w) != np.sign(iqs_ref_unsat)):
            state["int_err_w"] += err_w * dt

        iqs_ref = self.kp_w * err_w + self.ki_w * state["int_err_w"]

        v_alpha, v_beta = park_transform(vqs_ref, -vds_ref, -state["theta"])
        va_ref, vb_ref, vc_ref = inverse_clarke_transform(v_alpha, v_beta)

        van, vbn, vcn = self._svpwm(t, va_ref, vb_ref, vc_ref)

        return w_sync, van, vbn, vcn
