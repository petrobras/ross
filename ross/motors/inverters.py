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

# Switching SVPWM table, shared by every inverter implemented in this module.
# Each column represents the states of the upper switches for the space
# vectors V0, V1, V3, V2, V6, V4, V5 and V7
_SW_TABLE = np.array(
    [
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ]
)

# Active vectors according to the sector, shared by every inverter implemented
# in this module.
_ACTIVE_VECTORS = np.array([[2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 2]])

# Null (zero) vectors
_V0 = 1
_V7 = 8


class InverterVF:
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
        Reference frequency for V/f adjustment [rad/s]. Default is 0.

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
    >>> np.round(Vp, 2)
    179.63

    >>> van, vbn, vcn = inverter.get_phase_voltages(t=0.001, frequency=freq)
    >>> np.round([van, vbn, vcn], 2)
    array([ 100., -200.,  100.])

    >>> f = inverter.get_frequency(t=0.05, frequency_ref=freq)
    >>> np.round(f, 2)
    18.85

    >>> f, van, vbn, vcn = inverter.get_operating_state(t=0.5, frequency_ref=freq)
    >>> np.round([f, van, vbn, vcn], 2)
    array([188.5,   0. ,   0. ,   0. ])

    References
    ----------
    Wu, B. & Narimani, M. (2016). High-Power Converters and AC Drives. Wiley.
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

        self.voltage_dc = voltage_dc
        self.frequency_s = frequency_s
        self.Ts = 2 * np.pi / frequency_s

        self.voltage_nom = float(voltage_nom)
        self.frequency_nom = float(frequency_nom)
        self.time_ramp = float(time_ramp)
        self.frequency_ref = float(frequency_ref or frequency_nom / 2)

        # Nominal phase voltage peak value
        self.voltage_phase_peak_nom = (voltage_nom / np.sqrt(3)) * np.sqrt(2)

        self.f_0 = 0.0

        # Switching SVPWM table and active vectors (shared across inverters)
        self.sw_table = _SW_TABLE
        self.actv_vet = _ACTIVE_VECTORS

        self.V0 = _V0
        self.V7 = _V7

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

    def get_phase_voltages(self, t, frequency):
        """Generate three-phase voltages using SVPWM modulation.

        Computes the instantaneous phase voltages (A, B, C) using Space Vector
        PWM modulation based on the reference frequency and switching configuration.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        frequency : float
            Operating frequency [rad/s].

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
        theta = frequency * t
        va_ref = Vp * np.sin(theta)
        vb_ref = Vp * np.sin(theta - 2 * np.pi / 3)
        vc_ref = Vp * np.sin(theta + 2 * np.pi / 3)

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

        # Dwell times for linear operation
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
            # Rescales the switching intervals to fit within Ts
            if sumT > 0:
                T1 *= self.Ts / sumT
                T2 *= self.Ts / sumT
                T0 = self.Ts - T1 - T2

            else:
                # Extreme case: zero vector only
                T1, T2, T0 = 0, 0, self.Ts

        # Sequence and timing of vector application
        # Symmetrical switching sequence
        vetor_seq = (
            np.array(
                [self.V0, self.actv_vet[S - 1, 0], self.actv_vet[S - 1, 1], self.V7]
            )
            - 1
        )

        # Switching states
        S_bits = self.sw_table[:, vetor_seq]

        # Duty cycles - campling values within the [0, 1] interval
        t_seq = np.array([T0 / 2, T1, T2, T0 / 2])
        D = np.clip((S_bits @ t_seq) / self.Ts, 0, 1)

        # Triangular carrier
        if self.Ts == 0:
            u = 0
        else:
            u = (t % self.Ts) / self.Ts
        carrier = 1 - 4 * np.abs(u - 0.5)

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

    def get_operating_state(self, t, frequency_ref=None):
        """Get the fundamental frequency and phase voltages of
        the inverter at time `t`.

        Parameters
        ----------
        t : float
            Time [s].
        frequency_ref : float, optional
            Reference frequency [rad/s]. If None, uses `self.frequency_ref`.

        Returns
        -------
        frequency : float
            Fundamental frequency [rad/s].
        vas, vbs, vcs : tuple of float
            Instantaneous phase voltages [V].
        """
        freq = self.get_frequency(t, frequency_ref)
        van, vbn, vcn = self.get_phase_voltages(t, freq)

        return freq, van, vbn, vcn


class InverterFOC:
    """Simulate a three-phase voltage source inverter with indirect Field-Oriented Control.

    This class implements a three-phase VSI operating under indirect
    Field-Oriented Control (iFOC), synthesizing the output voltages through
    Space Vector PWM (SVPWM), analogous to :class:`InverterVF` but replacing
    scalar V/f control with closed-loop rotor-flux-oriented current control.

    Unlike class:`InverterVF` and class:`SourceAC`, this inverter is a
    closed-loop element: its output voltages depend on the instantaneous
    rotor speed and stator currents fed back from the motor at every time
    step. Therefore it cannot be pre-computed independently of the motor's
    state through ``get_operating_state(t)`` alone; it must be driven from
    within the motor's time-stepping loop, one step at a time.

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
    time_step : float
        Simulation time step [s], used to synchronize the discrete carrier
        with the switching period.
    time_ramp : float, optional
        Acceleration ramp time [s] for the mechanical speed reference.
        Default is 1.

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
        time_step,
        time_ramp=1.0,
    ):

        self.voltage_dc = voltage_dc
        self.frequency_s = frequency_s
        self.Ts = 2 * np.pi / frequency_s
        self.deltat = float(time_step)  # Simulation time step [s]

        self.voltage_nom = float(voltage_nom)
        self.frequency_nom = float(frequency_nom)
        self.time_ramp = float(time_ramp)
        self.wref_ramp = 0.0  # Current mechanical speed reference (ramp state)

        # Nominal phase voltage peak value
        self.voltage_phase_peak_nom = (voltage_nom / np.sqrt(3)) * np.sqrt(2)

        # Pole pairs
        self.p = n_poles / 2

        # Nominal mechanical angular speed [rad/s]
        self.wn = float(speed_nom)

        # Electrical parameters
        self.Rs = float(stator_resistance)
        self.Rr = float(rotor_resistance)
        self.Tn = float(torque_nom)
        self.J = float(Ip_motor)

        # Inductances derived from reactances (same convention as MotorElement)
        Lls = float(stator_reactance) / self.frequency_nom
        Llr = float(rotor_reactance) / self.frequency_nom
        self.Lm = float(mutual_reactance) / self.frequency_nom
        self.Lss = Lls + self.Lm
        self.Lrr = Llr + self.Lm

        # Rotor time constant
        self.taur = self.Lrr / self.Rr

        # =========== Direct-axis Reference Current Calculation ===============

        # Nominal slip
        self.sn = (self.frequency_nom - self.p * self.wn) / self.frequency_nom

        # Modified equivalent circuit rotor resistance
        Rr_eq = (self.Rr * (self.Lm / self.Lrr) ** 2) / self.sn

        # Auxiliary variables for calculating Zeq
        Ls1 = self.Lss - (self.Lm**2 / self.Lrr)
        Lm1 = (self.Lm**2) / self.Lrr

        # Equivalent impedance
        Zeq = (self.Rs + 1j * self.frequency_nom * Ls1) + (
            1j * self.frequency_nom * Lm1 * Rr_eq
        ) / (1j * self.frequency_nom * Lm1 + Rr_eq)

        # Phase voltage peak
        Vmax = self.voltage_nom * (np.sqrt(2) / np.sqrt(3))

        # Total stator current
        Is = Vmax / Zeq

        # Modified equivalent circuit voltage Er
        Er = Vmax - (self.Rs + 1j * self.frequency_nom * Ls1) * Is

        # Direct-axis reference current
        self.ids_ref = abs(Er) / (self.frequency_nom * Lm1)

        # Nominal stator current magnitude, used to bound the torque-producing
        # (q-axis) current reference commanded by the speed loop (see
        # `get_phase_voltages`). Without this limit, a speed-loop proportional
        # gain acting on a large transient speed error (e.g. a step from
        # standstill to a fast-ramped reference) can command an unbounded
        # `iqs_ref`, which directly drives the estimated slip frequency
        # (`wsl = iqs_ref / (taur * ids_ref)`) and makes the synchronous
        # frequency estimate diverge well before the voltage saturation
        # further down has a chance to act.
        self.Is_nom = abs(Is)

        # =============== PI controller gains (bandwidth method) ===============
        BWp_iqs = frequency_s / 8  # Bandwidth of the proportional q-axis current controller
        BWp_ids = frequency_s / 8  # Bandwidth of the proportional d-axis current controller
        BWi_ids = frequency_s / 8  # Bandwidth of the integral d-axis current controller
        BWp_w = BWi_ids / 8  # Bandwidth of the proportional speed controller
        BWi_w = BWp_w / 8  # Bandwidth of the integral speed controller

        # Load constant
        KL = self.Tn / (self.p * self.wn)

        # Proportional speed controller gain
        self.kp_w = self.J * 2 * np.pi * (BWp_w + BWi_w) - KL

        # Integral speed controller gain
        self.ki_w = self.J * 4 * (np.pi**2) * BWp_w * BWi_w

        # Proportional d-axis current controller gain
        self.kp_ids = self.Lss * 2 * np.pi * (BWp_ids + BWi_ids)

        # Integral d-axis current controller gain
        self.ki_ids = self.Lss * 4 * (np.pi**2) * BWp_ids * BWi_ids

        # Proportional q-axis current controller gain
        self.Lsline = self.Lss - (self.Lm**2) / self.Lrr  # Equivalent leakage inductance
        self.kp_iqs = 2 * np.pi * self.Lsline * BWp_iqs

        # ========================== Internal states ==========================
        self.int_errow = 0.0
        self.int_erroids = 0.0
        self.teta = 0.0

        # Switching SVPWM table and active vectors (shared across inverters)
        self.sw_table = _SW_TABLE
        self.actv_vet = _ACTIVE_VECTORS

        self.V0 = _V0
        self.V7 = _V7

        # Discrete carrier state
        self.k = 1
        # Number of simulation samples per switching period
        self.Ns = max(1, int(round(self.Ts / self.deltat)))

    def speed_control(self, wref):
        """Compute the mechanical speed reference with acceleration ramp.

        Parameters
        ----------
        wref : float
            Desired mechanical speed reference [rad/s].

        Returns
        -------
        float
            Ramped mechanical speed reference [rad/s].
        """
        # Reference saturation
        if wref < 0:
            wref = 0
        if wref > self.wn:
            wref = self.wn

        # Maximum speed variation per integration step, so that the ramp
        # reaches the reference speed exactly in `time_ramp` seconds
        dw_max = self.wn * self.deltat / self.time_ramp

        if self.wref_ramp < wref:
            self.wref_ramp += dw_max
            if self.wref_ramp > wref:
                self.wref_ramp = wref
        elif self.wref_ramp > wref:
            self.wref_ramp -= dw_max
            if self.wref_ramp < wref:
                self.wref_ramp = wref

        return self.wref_ramp

    def get_phase_voltages(self, wref, wr, ia, ib, ic, dt):
        """Run the iFOC control loop and synthesize phase voltages via SVPWM.

        Computes the instantaneous phase voltages (A, B, C) from the rotor
        speed and stator current feedback, using indirect Field-Oriented
        Control followed by Space Vector PWM modulation.

        Parameters
        ----------
        wref : float
            Desired mechanical speed reference [rad/s].
        wr : float
            Measured (feedback) rotor mechanical speed [rad/s].
        ia, ib, ic : float
            Measured (feedback) stator phase currents [A].
        dt : float
            Integration time step [s].

        Returns
        -------
        van : float
            Phase A voltage with respect to neutral [V].
        vbn : float
            Phase B voltage with respect to neutral [V].
        vcn : float
            Phase C voltage with respect to neutral [V].
        teta : float
            Synchronous electrical angle [rad].
        wshaft : float
            Synchronous electrical speed [rad/s].
        """
        # ======================== Speed control loop =========================
        wref = self.speed_control(wref)
        err_w = wref - wr

        u_prop = self.kp_w * err_w
        u_int = self.ki_w * self.int_errow

        iqs_ref_unsat = u_prop + u_int

        # Torque-producing (q-axis) current limit. The estimated slip
        # frequency below is directly proportional to `iqs_ref`, so an
        # unbounded speed-loop output (as happens transiently for a large
        # speed error, e.g. a step from standstill) drives the synchronous
        # frequency estimate to unphysical values before the voltage
        # saturation further down ever gets a chance to act. Limiting
        # `iqs_ref` to a multiple of the nominal stator current keeps the
        # torque command within a realistic transient rating (consistent
        # with how real drives limit acceleration current).
        iqs_max = 3.0 * self.Is_nom
        iqs_ref = np.clip(iqs_ref_unsat, -iqs_max, iqs_max)
        current_saturated = abs(iqs_ref_unsat) > iqs_max

        # Slip frequency calculation
        wsl = (1 / self.taur) * (iqs_ref / self.ids_ref)

        # Synchronous electrical speed
        wshaft = wsl + self.p * wr

        self.teta += wshaft * dt  # Angle integration

        # abc -> dq current transformation (Clarke + Park), q aligned with
        # cosine (torque), d aligned with sine (flux), matching the iFOC
        # convention: d_std = iqs, q_std = -ids
        i_alpha, i_beta = clarke_transform(ia, ib, ic)
        d_std, q_std = park_transform(i_alpha, i_beta, self.teta)
        iqs, ids = d_std, -q_std

        # ======================== iqs current loop ===========================
        err_iqs = iqs_ref - iqs

        vqs_ref = self.kp_iqs * err_iqs + self.Rs * iqs + self.Lss * wshaft * ids

        # ======================== ids current loop ===========================
        err_ids = self.ids_ref - ids

        # Integrator
        self.int_erroids += err_ids * dt

        vds_ref = (
            self.kp_ids * err_ids
            + self.ki_ids * self.int_erroids
            + self.Rs * ids
            - self.Lsline * wshaft * iqs
        )

        # =================== Saturation & anti-windup ========================
        Vmax = self.voltage_dc / np.sqrt(3)  # Maximum phase voltage
        Vref = np.sqrt(vqs_ref**2 + vds_ref**2)  # Voltage reference magnitude

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
        any_saturated = saturated or current_saturated
        if (not any_saturated) or (np.sign(err_w) != np.sign(iqs_ref_unsat)):
            self.int_errow += err_w * dt

        iqs_ref = self.kp_w * err_w + self.ki_w * self.int_errow

        # dq -> abc voltage transformation (inverse Park via the -theta
        # identity, followed by inverse Clarke)
        v_alpha, v_beta = park_transform(vqs_ref, -vds_ref, -self.teta)
        va_ref, vb_ref, vc_ref = inverse_clarke_transform(v_alpha, v_beta)

        # =============== Phase voltage synthesis through SVPWM ===============
        van, vbn, vcn = self.svpwm(va_ref, vb_ref, vc_ref)

        return van, vbn, vcn, self.teta, wshaft

    def svpwm(self, va_ref, vb_ref, vc_ref):
        """Synthesize phase voltages via Space Vector PWM modulation.

        Parameters
        ----------
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

        # Discrete triangular carrier, synchronized with the switching period
        n_in_period = (self.k - 1) % self.Ns
        u = 0 if self.Ns == 1 else n_in_period / (self.Ns - 1)
        carrier = 1 - 4 * np.abs(u - 0.5)

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

        self.k += 1

        return van, vbn, vcn

    def get_operating_state(self, t, wref, wr=0.0, ia=0.0, ib=0.0, ic=0.0, dt=None):
        """Get the synchronous frequency and phase voltages of the inverter.

        Parameters
        ----------
        t : float
            Time [s] (kept for interface compatibility with InverterVF).
        wref : float
            Desired mechanical speed reference [rad/s].
        wr : float, optional
            Measured (feedback) rotor mechanical speed [rad/s]. Default is 0.
        ia, ib, ic : float, optional
            Measured (feedback) stator phase currents [A]. Default is 0.
        dt : float, optional
            Integration time step [s]. If None, uses `self.deltat`.

        Returns
        -------
        frequency : float
            Synchronous electrical frequency [rad/s].
        van, vbn, vcn : tuple of float
            Instantaneous phase voltages [V].
        """
        if dt is None:
            dt = self.deltat

        van, vbn, vcn, teta, wshaft = self.get_phase_voltages(wref, wr, ia, ib, ic, dt)

        return wshaft, van, vbn, vcn
