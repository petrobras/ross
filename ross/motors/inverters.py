"""Inverter module

This module provides implementations for modeling and simulating three-phase VSIs
using Space Vector PWM (SVPWM) modulation with V/f adjustment technique.
"""

import numpy as np

from ross.units import check_units

from .utils import clarke_transform


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

        # Switching SVPWM table
        # Each column represents the states of the upper switches
        # for the space vectors V0, V1, V3, V2, V6, V4, V5 and V7
        self.sw_table = np.array(
            [
                [0, 1, 1, 0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ]
        )

        # Active vectors according to the sector
        self.actv_vet = np.array([[2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 2]])

        self.V0 = 1
        self.V7 = 8

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
