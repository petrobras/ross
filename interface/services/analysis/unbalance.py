# -*- coding: utf-8 -*-
"""Unbalance response."""

import numpy as np

from .base import Runner, register

METHODS = {
    "Default": "plot",
    "Magnitude": "plot_magnitude",
    "Phase": "plot_phase",
    "Bode": "plot_bode",
    "Polar Bode": "plot_polar_bode",
}


@register
class UnbalanceRunner(Runner):
    name = "unbalance"

    def spec(self, params, rotor):
        minimum, maximum = self.speed_bounds(params)
        nodes, magnitudes, phases = self.unbalances(
            params, {"node": 0, "mag": 0.01, "phase": 0.0}, 0.01, 0.0, clamp_rotor=rotor
        )
        return {
            "speed_min": minimum,
            "speed_max": maximum,
            "node": nodes,
            "unbalance_magnitude": magnitudes,
            "unbalance_phase": phases,
            "modes": self.literal(params, "modes"),
        }

    def compute(self, rotor, spec):
        speeds = np.linspace(spec["speed_min"], spec["speed_max"], 50)
        kwargs = {}
        if spec["modes"]:
            kwargs["modes"] = spec["modes"]
        return rotor.run_unbalance_response(
            node=spec["node"],
            unbalance_magnitude=spec["unbalance_magnitude"],
            unbalance_phase=spec["unbalance_phase"],
            frequency=speeds,
            **kwargs,
        )

    def plot(self, result, params, rotor):
        kind = params.get("plot_type", "Default")
        method = METHODS.get(kind, "plot")

        kwargs = self.units(
            params, ["probe_units", "frequency_units", "amplitude_units"]
        )
        if kind in ("Default", "Phase", "Bode", "Polar Bode"):
            kwargs.update(self.units(params, ["phase_units"]))
        if kind == "Magnitude":
            kwargs.update(self.units(params, ["line_shape"]))

        return getattr(result, method)(probe=self.probes(params), **kwargs)
