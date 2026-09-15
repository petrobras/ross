# -*- coding: utf-8 -*-
"""Modos de vibrar."""

from .base import Runner, register
from ross.interface.domain.conversion import get_converted_param

# ROSS refuses the pair too, but its message names the arguments and not the
# fields; this one says what to clear.
ONE_WHIRL_CHOICE = (
    "Fill in 'Whirl Frequency (fixed)' or turn 'Matched Whirl' on, not both: the "
    "coefficients are evaluated either at the one frequency typed or at each "
    "mode's own whirl frequency."
)


@register
class ModalRunner(Runner):
    name = "modes"

    def spec(self, params, rotor):
        spec = {
            "speed": get_converted_param(params, "speed", 0.0, "rad/s"),
            "num_modes": self.integer(params, "num_modes", 12),
            "sparse": self.flag(params, "sparse", True),
            "synchronous": self.flag(params, "synchronous"),
            "frequency": self.optional_quantity(params, "frequency", "rad/s"),
            "matched_whirl": self.flag(params, "matched_whirl"),
        }
        if spec["frequency"] is not None and spec["matched_whirl"]:
            raise ValueError(ONE_WHIRL_CHOICE)
        return spec

    def compute(self, rotor, spec):
        # The fixed whirl frequency goes only when typed: an explicit None is
        # ROSS's default today and nothing more.
        kwargs = {}
        if spec["frequency"] is not None:
            kwargs["frequency"] = spec["frequency"]
        return rotor.run_modal(
            speed=spec["speed"],
            num_modes=spec["num_modes"],
            sparse=spec["sparse"],
            synchronous=spec["synchronous"],
            matched_whirl=spec["matched_whirl"],
            **kwargs,
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
