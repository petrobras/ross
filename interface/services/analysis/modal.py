# -*- coding: utf-8 -*-
"""Modos de vibrar."""

from .base import Runner, register
from domain.conversion import get_converted_param


@register
class ModalRunner(Runner):
    name = "modes"

    def spec(self, params, rotor):
        return {
            "speed": get_converted_param(params, "speed", 0.0, "rad/s"),
            "num_modes": self.integer(params, "num_modes", 12),
            "sparse": self.flag(params, "sparse", True),
            "synchronous": self.flag(params, "synchronous"),
        }

    def compute(self, rotor, spec):
        return rotor.run_modal(
            speed=spec["speed"],
            num_modes=spec["num_modes"],
            sparse=spec["sparse"],
            synchronous=spec["synchronous"],
        )

    def plot(self, result, params, rotor):
        position = self.integer(params, "plot_idx", 0)
        kind = params.get("plot_type", "2D")

        if kind == "3D":
            kwargs = self.units(
                params,
                [
                    "frequency_type",
                    "length_units",
                    "phase_units",
                    "frequency_units",
                    "damping_parameter",
                ],
            )
            kwargs["animation"] = self.flag(params, "animation")
            return result.plot_mode_3d(position, **kwargs)

        if kind == "Orbit":
            kwargs = {}
            nodes = self.literal(params, "nodes")
            if nodes:
                kwargs["nodes"] = nodes
            return result.plot_orbit(position, **kwargs)

        kwargs = self.units(
            params,
            ["orientation", "frequency_type", "frequency_units", "damping_parameter"],
        )
        return result.plot_mode_2d(position, **kwargs)
