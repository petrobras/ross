"""SourceAC module

Ideal 3-phase AC source with harmonic distortion and voltage unbalance support.
"""

import plotly.graph_objects as go

import numpy as np

from .motor_results import VoltageTimeResults

class SourceAC:
    """
    Ideal 3-phase AC source with support for:
      - Arbitrary harmonic orders and amplitudes (harmonic_orders, harmonic_amplitudes)
      - Per-phase voltage magnitude unbalance (voltage_unb_percent) [%]
      - Per-phase voltage angle deviation (angle_deviation) [degrees]

    Parameters
    ----------
    voltage_net  : float  – RMS phase voltage [V]
    frequency_net  : float  – Fundamental frequency [Hz]
    initial_phase_angle: float  – Initial phase angle [rad] (default 0)
    harmonic_orders   : list/array of int   – Harmonic orders  (e.g. [5, 7, 11])
    harmonic_amplitudes   : list/array of float – Harmonic amplitudes as % of voltage_net (e.g. [10, 5, 2])
    voltage_unb_percent  : list/array of float – Voltage magnitude unbalance per phase [%]
                                  Order: [phase_A, phase_B, phase_C]
                                  Positive → higher voltage; Negative → lower voltage.
    angle_deviation  : list/array of float – Angle deviation per phase [degrees]
                                  Order: [phase_A, phase_B, phase_C]
    Behaviour
    ----------
    Configuring Harmonics and Unbalances: set harmonic_orders,harmonic_amplitudes, voltage_unb_percent,angle_deviation --> _pending
    Enabling disturbances: call .harmonics("enable") / .unbalances("enable") -->_active
    Disabling disturbances: call .harmonics("disable") / .unbalances("disable"), keeping configuration

    """

    # Phase angle offsets for a balanced 3-phase system [rad]
    # [phase_A, phase_B, phase_C]
    _PHASE_OFFSET = [0.0, -2 * np.pi / 3, +2 * np.pi / 3]
    _PHASE_LABEL = ["A", "B", "C"]

    def __init__(
        self,
        voltage_net,
        frequency_net,
        initial_phase_angle=0.0,
        harmonics=None,
        unbalances=None,
    ):

        self.voltage_net = voltage_net
        self.frequency_net = frequency_net
        self.initial_phase_angle = initial_phase_angle

        # --- Harmonics ---
        if harmonics:
            self.set_harmonics(harmonics.get("orders"), harmonics.get("amplitudes"), harmonics.get("enable"))
        else:
            self.set_harmonics(None, None, False)
        
        # --- Unbalances ---
        if unbalances:
            self.set_unbalances(unbalances.get("voltage_percent"), unbalances.get("angle_deviation"), unbalances.get("enable"))
        else:
            self.set_unbalances(None, None, False)

    def set_harmonics(self, harmonic_orders, harmonic_amplitudes, enable=True):
        """Validate and align harmonic_orders / harmonic_amplitudes vectors."""
        if harmonic_orders is not None:
            harmonic_orders = list(harmonic_orders)
            n = len(harmonic_orders)
            # Adjust harmonic_amplitudes vector to the harmonic_orders length
            if harmonic_amplitudes is None:
                harmonic_amplitudes = [0.0] * n
            else:
                harmonic_amplitudes = list(harmonic_amplitudes)
                if len(harmonic_amplitudes) < n:
                    harmonic_amplitudes += [0.0] * (n - len(harmonic_amplitudes))  # filling up to the harmonic_orders length
                else:
                    harmonic_amplitudes = harmonic_amplitudes[:n]  # truncate to harmonic_orders length

            # Clamp amplitudes to [0, 100]
            harmonic_amplitudes = [float(np.clip(a, 0.0, 100.0)) for a in harmonic_amplitudes]
            # Truncate harmonic_orders to integer numbers
            harmonic_orders = [int(h) for h in harmonic_orders]

        self.harmonics = {
            "enable": enable or False,
            "orders": harmonic_orders,
            "amplitudes": harmonic_amplitudes
        }

    @staticmethod
    def _parse_3vec(v, default):
        """Return a 3-element list. Fills missing entries with `default`."""
        """Addresses both unbalances and deviations."""
        if v is None:
            return [default, default, default]
        v = list(v)
        if len(v) < 3:
            v += [default] * (3 - len(v))
        elif len(v) > 3:
            v = v[:3]
        return [float(x) for x in v]
    
    def set_unbalances(self, voltage_percent, angle_deviation, enable=True):
        
        voltage_percent = self._parse_3vec(voltage_percent, 0.0)
        angle_deviation = self._parse_3vec(angle_deviation, 0.0)
    
        self.unbalances = {
            "enable": enable or False,
            "voltage_percent": voltage_percent,
            "angle_deviation": angle_deviation,
        }


    def __repr__(self):
        return (
            f"SourceAC(voltage_net={self.voltage_net} V, frequency_net={self.frequency_net} Hz, "
            f"initial_phase_angle={self.initial_phase_angle:.4f} rad, "
            f"harmonics={'ON' if self.harmonics['enable'] else 'OFF'}, "
            f"unbalances={'ON' if self.unbalances['enable'] else 'OFF'})"
        )
    
    def _build_phase_voltage(self, t, phase_id):
        """
        Compute instantaneous voltage for one phase at time `t`, applying
        harmonics frequencies, unbalanced amplitudes and angle deviations
        phase_id : 0 → A, 1 → B, 2 → C
        """
        phi = self._PHASE_OFFSET[phase_id]  # nominal phase shift
        w = 2 * np.pi * self.frequency_net

        if self.unbalances["enable"]:
            vd_pct = self.unbalances["voltage_percent"][phase_id]  # % magnitude deviation
            ad_deg = self.unbalances["angle_deviation"][phase_id]  # angle deviation [°]
        else:
            vd_pct = 0.0
            ad_deg = 0.0

        mag_factor = (100.0 + vd_pct) / 100.0  # amplitude scale
        angle_offset = ad_deg * np.pi / 180.0  # extra phase [rad]

        # Phase A gets +angle_deviation, Phase B neutral, Phase C gets –angle_deviation
        # (mirrors the sign convention used in the original snippet)
        sign = [+1.0, 0.0, -1.0][phase_id]
        angle_offset *= sign

        # --- Harmonics ---
        a_harm = 0.0
        v = 0.0

        if self.harmonics["enable"]:
            for h, a in zip(self.harmonics["orders"], self.harmonics["amplitudes"]):
                a_harm += a

                v += (
                    np.sqrt(2)
                    * self.voltage_net
                    * mag_factor
                    * (a / 100.0)
                    * np.cos(h * w * t + self.initial_phase_angle + phi + angle_offset)
                )

        v += (
            np.sqrt(2)
            * self.voltage_net
            * mag_factor
            * (100 - a_harm) / 100
            * np.cos(w * t + self.initial_phase_angle + phi + angle_offset)
        )

        return v

    def get_phase_voltages(self, t):
        """Return instantaneous phase voltages (vas, vbs, vcs) at time *t*."""
        vas = self._build_phase_voltage(t, 0)
        vbs = self._build_phase_voltage(t, 1)
        vcs = self._build_phase_voltage(t, 2)
        return vas, vbs, vcs

    def run(self, t):
        """Run the simulation for a series of time steps.

        Parameters
        ----------
        t : array_like
            Array of time steps.

        Returns
        -------
        results : dict
            A dictionary containing lists of results for the entire simulation:
            - time, Vas, Vbs, Vcs.
        """
        vas, vbs, vcs = np.vectorize(self.get_phase_voltages)(t)

        voltages = dict()
        voltages["a"] = vas
        voltages["b"] = vbs
        voltages["c"] = vcs

        results = VoltageTimeResults(t, voltages)

        return results

def sourceAC_example():
    src = SourceAC(voltage_net=220.0, frequency_net=60.0)
    print(src)
    print()

    # --- Configure harmonics ---
    src.harmonics(
        harmonic_orders=[5, 7, 11, 13], harmonic_amplitudes=[10.0, 5.0, 3.0]
    )  # harmonic_amplitudes shorter → 13th = 0%
    src.harmonics("enable")
    src.harmonics()  # print report

    print()

    # --- Configure unbalances ---
    src.unbalances(voltage_unb_percent=[5.0, 0.0, -3.0], angle_deviation=[2.0, 0.0, -2.0])
    src.unbalances("enable")
    src.unbalances()  # print report

    print()

    # --- Compute a few voltages ---
    t_samples = np.linspace(0, 1 / 60, 5)
    resp = src.run(t_samples)
    fig_V = resp.plot()

    print()

    # --- Disable harmonics ---
    src.harmonics("disable")
    src.harmonics()

    fig_V.write_html("plot.html", auto_open=True)

    return src, fig_V
