# -*- coding: utf-8 -*-
"""Close-clearance check of the unbalance response after API 617.

petrobras/ross#1377 rewrote `run_clearance_analysis`: it no longer takes one
speed and a frequency list but a speed range from zero to trip, the minimum
allowable and maximum continuous speeds, and the machine's vibration probes
(Amax is read off them, so they are part of the computation, not the drawing).
The unbalance became optional -- left out, ROSS places the API 617 unbalance
from the mode shape; given, the table overrides that placement.
"""

import numpy as np

from .base import Runner, register

METHODS = {
    "Default": "plot",
    "Response": "plot_response",
    "Probe Response": "plot_probe_response",
}


@register
class ClearanceRunner(Runner):
    name = "clearance"

    def spec(self, params, rotor):
        minimum, maximum = self.speed_bounds(params)
        # Nma and Nmc are physical parameters: blank is a refusal that names
        # the field, not a value ROSS never received from the user.
        speeds = {}
        for key in ("minimum_allowable_speed", "maximum_continuous_speed"):
            speeds[key] = self.optional_quantity(params, key, "rad/s")
            if speeds[key] is None:
                raise ValueError(
                    "Field '%s' is required by the clearance analysis and is empty."
                    % key
                )
        # The table is read only when it has rows. `unbalances()` would stand
        # in a row nobody typed, and here an empty table means something else:
        # the API 617 placement, computed by ROSS from the mode shape. The
        # three keys stay in the spec either way (empty lists) so the cache
        # key has one shape.
        if params.get("unbalances"):
            nodes, magnitudes, phases = self.unbalances(
                params,
                {"node": 0, "mag": 0.05, "phase": 0.0},
                0.05,
                0.0,
                clamp_rotor=rotor,
            )
        else:
            nodes, magnitudes, phases = [], [], []
        cap = self.text(params, "scale_factor_cap")
        probe_rows = params.get("probes") or [{"node": 0, "angle": 0.0}]
        return {
            "speed_min": minimum,
            "speed_max": maximum,
            "steps": self.integer(params, "speed_steps", 101),
            "minimum_allowable_speed": speeds["minimum_allowable_speed"],
            "maximum_continuous_speed": speeds["maximum_continuous_speed"],
            "probes": [
                {"node": int(row["node"]), "angle": float(row.get("angle", 0.0))}
                for row in probe_rows
            ],
            "mode": self.integer(params, "mode", 0),
            "node": nodes,
            "unbalance_magnitude": magnitudes,
            "unbalance_phase": phases,
            "scale_factor_cap": None if cap is None else float(cap),
            "num_modes": self.integer(params, "num_modes", 12),
        }

    def compute(self, rotor, spec):
        import ross as rs

        speeds = np.linspace(spec["speed_min"], spec["speed_max"], spec["steps"])
        probes = [rs.Probe(row["node"], row["angle"]) for row in spec["probes"]]
        kwargs = {"num_modes": spec["num_modes"]}
        if spec["scale_factor_cap"] is not None:
            kwargs["scale_factor_cap"] = spec["scale_factor_cap"]
        if spec["node"]:
            kwargs["node"] = spec["node"]
            kwargs["unbalance_magnitude"] = spec["unbalance_magnitude"]
            kwargs["unbalance_phase"] = spec["unbalance_phase"]
        else:
            kwargs["mode"] = spec["mode"]
        return rotor.run_clearance_analysis(
            speeds,
            spec["minimum_allowable_speed"],
            spec["maximum_continuous_speed"],
            probes,
            **kwargs,
        )

    def plot(self, result, params, rotor):
        kind = params.get("plot_type", "Default")
        method = METHODS.get(kind, "plot")
        kwargs = self.units(params, ["length_units"])
        if method != "plot":
            kwargs.update(self.units(params, ["speed_units", "line_shape"]))
        return getattr(result, method)(**kwargs)
