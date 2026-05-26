"""Sources module

Includes an ideal 3-phase AC source with harmonic distortion and voltage unbalance support.
"""

import numpy as np

from ross.units import Q_, check_units
from .results import VoltageTimeResults


class SourceAC:
    """Create an ideal 3-phase AC source with harmonics and voltage unbalance support.

    Supports arbitrary harmonic orders and amplitudes, per-phase voltage magnitude
    unbalance, and per-phase voltage angle deviation.

    Parameters
    ----------
    voltage_net : float, pint.Quantity
        RMS phase voltage [V].
    frequency_net : float, pint.Quantity
        Fundamental frequency [rad/s].
    initial_phase_angle : float, pint.Quantity, optional
        Initial phase angle [rad]. Default is 0.
    harmonic_orders : list of int
        Harmonic orders (e.g., [5, 7, 11]).
    harmonic_amplitudes : list of float
        Harmonic amplitudes as % of voltage_net (e.g., [10, 5, 2]).
    harmonic_enable : bool
        Whether to enable harmonics. Default is False.
    unbalance_voltage_percent : list of float
        Voltage magnitude unbalance per phase [%] in order [phase_A, phase_B, phase_C].
        Positive → higher voltage; Negative → lower voltage.
    unbalance_angle_deviation : list of float, pint.Quantity
        Angle deviation per phase [rad] in order [phase_A, phase_B, phase_C].
    unbalance_enable : bool
        Whether to enable unbalances. Default is False.

    Notes
    -----
    Harmonics and unbalances can be configured via `.set_harmonics()` and `.set_unbalances()`.

    Examples
    --------
    >>> src = SourceAC(voltage_net=220.0, frequency_net=Q_(60.0, "Hz"), initial_phase_angle=0.0)

    Configure harmonics
    >>> src.set_harmonics(
    ...    harmonic_orders=[5, 7, 11, 13],
    ...    harmonic_amplitudes=[10.0, 5.0, 3.0],
    ...    enable=True,
    ... )  # harmonic_amplitudes shorter → 13th = 0%

    Configure unbalances
    >>> src.set_unbalances(
    ...    voltage_percent=[5.0, 0.0, -3.0],
    ...    angle_deviation=Q_([2.0, 0.0, -2.0], "deg"),
    ...    enable=True,
    ... )

    Compute a few voltages
    >>> t_samples = np.linspace(0, 1 / 60, 5)
    >>> resp = src.run(t_samples)
    >>> fig = resp.plot()

    Disable harmonics
    >>> src.harmonics["enable"] = False
    """

    # Phase angle offsets for a balanced 3-phase system [rad]
    # [phase_A, phase_B, phase_C]
    _PHASE_OFFSET = [0.0, -2 * np.pi / 3, +2 * np.pi / 3]
    _PHASE_LABEL = ["A", "B", "C"]

    @check_units
    def __init__(
        self,
        voltage_net,
        frequency_net,
        initial_phase_angle=0.0,
        harmonic_orders=None,
        harmonic_amplitudes=None,
        harmonic_enable=False,
        unbalance_voltage_percent=None,
        unbalance_angle_deviation=None,
        unbalance_enable=False,
    ):
        self.voltage_net = voltage_net
        self.frequency_net = frequency_net
        self.initial_phase_angle = initial_phase_angle

        if harmonic_orders:
            self.set_harmonics(
                harmonic_orders,
                harmonic_amplitudes,
                harmonic_enable,
            )
        else:
            self.set_harmonics(None, None, False)

        self.set_unbalances(
            unbalance_voltage_percent,
            unbalance_angle_deviation,
            unbalance_enable,
        )

    def set_harmonics(self, harmonic_orders, harmonic_amplitudes=None, enable=True):
        """Set harmonic orders and amplitudes for the AC source.

        Parameters
        ----------
        harmonic_orders : list of int
            Harmonic orders (e.g., [5, 7, 11]).
        harmonic_amplitudes : list of float
            Harmonic amplitudes as % of voltage_net. If shorter than harmonic_orders,
            remaining amplitudes are set to 0.
        enable : bool, optional
            Enable harmonics. Default is True.
        """
        if harmonic_orders is not None:
            harmonic_orders = list(harmonic_orders)
            n = len(harmonic_orders)

            # Adjust harmonic_amplitudes vector to the harmonic_orders length
            if harmonic_amplitudes is None:
                harmonic_amplitudes = [0.0] * n
            else:
                harmonic_amplitudes = list(harmonic_amplitudes)
                if len(harmonic_amplitudes) < n:
                    harmonic_amplitudes += [0.0] * (
                        n - len(harmonic_amplitudes)
                    )  # filling up to the harmonic_orders length
                else:
                    harmonic_amplitudes = harmonic_amplitudes[
                        :n
                    ]  # truncate to harmonic_orders length

            # Clamp amplitudes to [0, 100]
            harmonic_amplitudes = [
                float(np.clip(a, 0.0, 100.0)) for a in harmonic_amplitudes
            ]

            # Truncate harmonic_orders to integer numbers
            harmonic_orders = [int(h) for h in harmonic_orders]

        self.harmonics = {
            "enable": enable or False,
            "orders": harmonic_orders,
            "amplitudes": harmonic_amplitudes,
        }

    @staticmethod
    def _parse_3vec(v, default):
        """Parse input into a 3-element list, filling missing entries with default.

        Parameters
        ----------
        v : list or None
            Input vector (may have fewer than 3 elements).
        default : float
            Default value for missing entries.

        Returns
        -------
        list
            3-element list with float entries.
        """
        if v is None:
            return [default, default, default]

        v = list(v)

        if len(v) < 3:
            v += [default] * (3 - len(v))
        elif len(v) > 3:
            v = v[:3]

        return [float(x) for x in v]

    @check_units
    def set_unbalances(self, voltage_percent, angle_deviation, enable=True):
        """Set voltage magnitude unbalance and angle deviation for the AC source.

        Parameters
        ----------
        voltage_percent : list of float
            Voltage magnitude unbalance per phase [%] in order [phase_A, phase_B, phase_C].
            Positive → higher voltage; Negative → lower voltage.
        angle_deviation : list of float, pint.Quantity
            Angle deviation per phase [rad] in order [phase_A, phase_B, phase_C].
        enable : bool, optional
            Enable unbalances. Default is True.
        """
        voltage_percent = self._parse_3vec(voltage_percent, 0.0)
        angle_deviation = self._parse_3vec(angle_deviation, 0.0)

        self.unbalances = {
            "enable": enable or False,
            "voltage_percent": voltage_percent,
            "angle_deviation": angle_deviation,
        }

    def __repr__(self):
        return (
            f"SourceAC(voltage_net={self.voltage_net} V, frequency_net={Q_(self.frequency_net, 'rad/s').to('Hz').m} Hz, "
            f"initial_phase_angle={self.initial_phase_angle:.4f} rad, "
            f"harmonics={'ON' if self.harmonics['enable'] else 'OFF'}, "
            f"unbalances={'ON' if self.unbalances['enable'] else 'OFF'})"
        )

    def _build_phase_voltage(self, t, phase_id):
        """Compute instantaneous voltage for one phase at time t.

        Parameters
        ----------
        t : float
            Time [s].
        phase_id : int
            Phase identifier: 0 → A, 1 → B, 2 → C.

        Returns
        -------
        float
            Instantaneous voltage [V] with harmonics, unbalance, and angle deviations applied.
        """
        phi = self._PHASE_OFFSET[phase_id]  # nominal phase shift
        w = self.frequency_net

        if self.unbalances["enable"]:
            vd_pct = self.unbalances["voltage_percent"][phase_id]
            angle_offset = self.unbalances["angle_deviation"][phase_id]
        else:
            vd_pct = 0.0
            angle_offset = 0.0

        mag_factor = (100.0 + vd_pct) / 100.0  # amplitude scale

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
            * (100 - a_harm)
            / 100
            * np.cos(w * t + self.initial_phase_angle + phi + angle_offset)
        )

        return v

    def get_phase_voltages(self, t):
        """Get instantaneous phase voltages at time `t`.

        Parameters
        ----------
        t : float
            Time [s].

        Returns
        -------
        tuple of float
            Instantaneous phase voltages `(vas, vbs, vcs)` [V].
        """
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
        results : ross.VoltageTimeResults
            For more information on attributes and methods available see:
            :py:class:`ross.VoltageTimeResults`
        """
        vas, vbs, vcs = np.vectorize(self.get_phase_voltages)(t)

        voltages = dict()
        voltages["a"] = vas
        voltages["b"] = vbs
        voltages["c"] = vcs

        results = VoltageTimeResults(t, voltages)

        return results
