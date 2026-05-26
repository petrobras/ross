"""Inverters module

Includes an inverter class for simulating a three-phase VSI with scalar V/f control.
"""

import numpy as np

from ross.units import Q_, check_units

from .utils import clarke_transform

class InverterVF:
    """A three-phase VSI element with scalar V/f control.

    This class defines the inverter_vf class, which represents a three-phase VSI
    simulated using the Space Vector PWM (SVPWM) modulation technique together
    with scalar V/f speed control.

    SVPWM based on Wu, B. & Narimani, M. High-Power Converters and AC Drives, 2016.
    This class creates a three-phase Voltage Source Inverter employing
    the Space Vector PWM (SVPWM) modulation technique for phase voltage
    synthesis together with scalar V/f speed control.

    Parameters:
    -----------
    voltage_dc : float
        DC Link Voltage (line)
    frequency_s : int, pint.Quantity
        IGBT switching frequency [rad/s]
    deltat : float
        Simulation integration time step. Must be smaller than the switching period.
    voltage_nom : float
        Nominal line voltage [V]
    frequency_nom : float, pint.Quantity
        Nominal frequency [rad/s]
    time_ramp : float
        Acceleration ramp time [s].
        Default is 1.
    """

    # @check_units
    def __init__(
        self, voltage_dc, frequency_s, deltat, voltage_nom, frequency_nom, time_ramp=1.0
    ):

        self.voltage_dc = voltage_dc
        self.frequency_s = frequency_s
        self.Ts = 1 / frequency_s
        self.deltat = deltat

        self.voltage_nom = voltage_nom
        self.frequency_nom = frequency_nom
        self.time_ramp = time_ramp

        # Nominal phase voltage peak value
        self.voltage_phase_peak_nom = (voltage_nom / np.sqrt(3)) * np.sqrt(2)

        # Current applied frequency
        self.f_curr = 0.0

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

        self.V0 = 1  # Corresponding to the V0 vector
        self.V7 = 8  # Corresponding to the V7 vector

        # Internal carrier state
        self.k = 1

        # Number of integration steps within one switching period
        self.Ns = max(1, int(round(self.Ts / self.deltat)))

    def speed_control(self, fref):
        # ===================== V/f Adjustment with Acceleration Ramp =============

        # Reference saturation
        fref = min(max(fref, 0), self.frequency_nom)

        # ------------------------ Acceleration Ramp --------------------------
        # Maximum frequency variation per integration step
        # to reach the reference frequency exactly in time_ramp seconds
        df_max = (fref / self.time_ramp) * self.deltat

        if self.f_curr < fref:
            self.f_curr = min(self.f_curr + df_max, fref)

        elif self.f_curr > fref:
            self.f_curr = max(self.f_curr - df_max, fref)

        # --------------------- Scalar V/f Speed Control ----------------------
        # Peak value of the phase voltage proportional
        # to the V/f ratio
        Vp = self.voltage_phase_peak_nom * (self.f_curr / self.frequency_nom)

        # Saturation at the nominal value
        Vp = min(Vp, self.voltage_phase_peak_nom)

        # Desired peak voltage and frequency
        return Vp, self.f_curr
    
    def get_wref(self):
        return 2 * np.pi * self.f_curr

    def get_phase_voltages(self, t, fref):
        # ===================== SVPWM Computation =================================

        # Reference peak voltage and frequency
        Vp, freq = self.speed_control(fref)

        # Reference voltages
        w = 2 * np.pi * freq
        va_ref = Vp * np.sin(w * t)
        vb_ref = Vp * np.sin(w * t - 2 * np.pi / 3)
        vc_ref = Vp * np.sin(w * t + 2 * np.pi / 3)

        # Clarke transformation
        v_alpha, v_beta = clarke_transform(va_ref, vb_ref, vc_ref)

        # Space vector and SVPWM hexagon sector
        vr = np.sqrt(v_alpha**2 + v_beta**2)
        theta = np.arctan2(v_beta, v_alpha)

        if theta < 0:
            theta += 2 * np.pi

        S = int(np.floor(theta * np.pi / 3)) + 1
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

        # Normalization
        # Ensures all values are non-negative
        T1 = max(T1, 0)
        T2 = max(T2, 0)
        T0 = max(T0, 0)

        # Total sum of the switching intervals
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

        t_seq = np.array([T0 / 2, T1, T2, T0 / 2])

        # Switching
        Sa_bits = self.sw_table[0, vetor_seq]
        Sb_bits = self.sw_table[1, vetor_seq]
        Sc_bits = self.sw_table[2, vetor_seq]

        # Duty cycles
        Da = np.dot(t_seq, Sa_bits) / self.Ts
        Db = np.dot(t_seq, Sb_bits) / self.Ts
        Dc = np.dot(t_seq, Sc_bits) / self.Ts

        # Clamps values within the [0, 1] interval
        Da = np.clip(Da, 0, 1)
        Db = np.clip(Db, 0, 1)
        Dc = np.clip(Dc, 0, 1)

        # Triangular carrier
        n_in_period = (self.k - 1) % self.Ns

        if self.Ns == 1:
            u = 0
        else:
            u = n_in_period / (self.Ns - 1)

        # Triangle centered at zero with a peak of +1
        carrier = 1 - 4 * np.abs(u - 0.5)

        # Thresholds derived from duty cycles
        RefA, RefB, RefC = 2 * np.array([Da, Db, Dc]) - 1

        # Comparisons to determine switching states
        Ss = np.array([float(carrier <= RefA), float(carrier <= RefB), float(carrier <= RefC)])

        # Pole voltages
        vao, vbo, vco = (2 * Ss - 1) * (self.voltage_dc / 2)

        # Phase voltages
        van = (2 / 3) * vao - (1 / 3) * (vbo + vco)
        vbn = (2 / 3) * vbo - (1 / 3) * (vao + vco)
        vcn = (2 / 3) * vco - (1 / 3) * (vbo + vao)

        self.k += 1

        return van, vbn, vcn
