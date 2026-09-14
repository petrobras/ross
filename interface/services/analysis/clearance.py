# -*- coding: utf-8 -*-
"""Bearing clearance analysis."""

from .base import Runner, register
from domain.conversion import get_converted_param


@register
class ClearanceRunner(Runner):
    name = "clearance"

    def spec(self, params, rotor):
        # The three unbalance arguments have to agree in shape, and that is not
        # a matter of style: `run_unbalance_response` pairs them with a `zip`,
        # so a scalar node beside a list magnitude either loses values in
        # silence or -- since numpy 2.5 -- raises. Reading the three out of one
        # table makes disagreement impossible: they come from the same rows and
        # cannot differ in length.
        nodes, magnitudes, phases = self.unbalances(
            params,
            {"node": 0, "mag": 0.05, "phase": 0.0},
            0.05,
            0.0,
            clamp_rotor=rotor,
            analysis="clearance",
        )
        return {
            "speed": get_converted_param(params, "speed", 600, "rad/s"),
            "node": nodes,
            "unbalance_magnitude": magnitudes,
            "unbalance_phase": phases,
            "frequency": self.literal(params, "frequency"),
            "modes": self.literal(params, "modes"),
        }

    def compute(self, rotor, spec):
        kwargs = {}
        if spec["frequency"]:
            kwargs["frequency"] = spec["frequency"]
        if spec["modes"]:
            kwargs["modes"] = spec["modes"]
        return rotor.run_clearance_analysis(
            spec["speed"],
            spec["node"],
            spec["unbalance_magnitude"],
            spec["unbalance_phase"],
            **kwargs,
        )

    def plot(self, result, params, rotor):
        return result.plot()
