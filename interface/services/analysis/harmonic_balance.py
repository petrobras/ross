# -*- coding: utf-8 -*-
"""Harmonic balance response."""

import numpy as np

from .base import Runner, register
from domain.conversion import get_converted_param


@register
class HarmonicBalanceRunner(Runner):
    name = "harmonic_balance"

    def spec(self, params, rotor):
        return {
            "speed": get_converted_param(params, "speed", 200, "rad/s"),
            "t_initial": self.number(params, "t_initial", 0.0),
            "t_final": self.number(params, "t_final", 0.5),
            "t_steps": self.integer(params, "t_steps", 1001),
            "hb_node": self.integer(params, "hb_node", 0),
            "hb_magnitudes": self.literal(params, "hb_magnitudes") or [2000.0],
            "hb_phases": self.literal(params, "hb_phases") or [0.0],
            "hb_harmonics": self.literal(params, "hb_harmonics") or [1],
            "gravity": self.flag(params, "gravity", False),
            "n_harmonics": self.integer(params, "n_harmonics", 1),
        }

    def compute(self, rotor, spec):
        time_grid = np.linspace(spec["t_initial"], spec["t_final"], spec["t_steps"])
        forces = [
            {
                "node": spec["hb_node"],
                "magnitudes": spec["hb_magnitudes"],
                "phases": spec["hb_phases"],
                "harmonics": spec["hb_harmonics"],
            }
        ]
        return rotor.run_harmonic_balance_response(
            spec["speed"],
            time_grid,
            forces,
            gravity=spec["gravity"],
            n_harmonics=spec["n_harmonics"],
        )

    def plot(self, result, params, rotor):
        return result.plot(
            probe=self.probes(params),
            **self.units(params, ["amplitude_units", "frequency_units"]),
        )
