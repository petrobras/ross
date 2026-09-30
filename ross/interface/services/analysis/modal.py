# -*- coding: utf-8 -*-
"""Modos de vibrar."""

from .base import Runner, register
from ross.interface.domain.conversion import get_converted_param

# ROSS refuses the pair too, but its message names the arguments and not the
# fields; this one says what to clear.
# Two refusals ROSS does not make. Both end in a chart with axes, a title and
# no curve, which is also what a torsional mode legitimately looks like -- so
# without these the user cannot tell "I typed a node that does not exist" from
# "this mode has no orbit".
NODES_UNREADABLE = (
    "'%s' is not a list of nodes. Type one node (3) or several ([3, 4]), or "
    "leave the field empty to draw every node of the rotor."
)

NO_SUCH_NODE = "The rotor has no node %s. Its nodes are %s."

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

    def orbit_nodes(self, params, rotor):
        """Which nodes the orbit is drawn for, refusing what ROSS would swallow.

        Measured on ROSS 3 (`results.py`): `plot_orbit` keeps
        `[o for o in self.orbits if o.node in nodes]`, so a node that does not
        exist gives a chart with no curve in it, in silence. `nodes=None` used
        to do the same (it became `[None]`); ROSS now reads it as every node.

        The field was declared optional in the catalogue and is, in practice,
        required. Rather than make the person type something, an empty field
        now means every node -- which is the useful answer, because the point of
        the plot is how the orbit changes along the shaft line.
        """
        every = [int(node) for node in rotor.nodes]

        raw = str(params.get("nodes", "")).strip()
        if raw == "":
            return every

        # `literal` answers None both for an empty field and for text it cannot
        # read, and here those are different things: the first is a choice, the
        # second is a typo that would otherwise become "every node".
        chosen = self.literal(params, "nodes")
        if chosen is None:
            raise ValueError(NODES_UNREADABLE % raw)
        if not isinstance(chosen, (list, tuple, set)):
            chosen = [chosen]

        wanted = []
        for value in chosen:
            try:
                wanted.append(int(value))
            except (TypeError, ValueError):
                raise ValueError(NODES_UNREADABLE % raw) from None

        missing = [node for node in wanted if node not in every]
        if missing:
            raise ValueError(
                NO_SUCH_NODE
                % (
                    ", ".join(str(node) for node in missing),
                    ", ".join(str(node) for node in every),
                )
            )
        return wanted

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
            return result.plot_orbit(position, nodes=self.orbit_nodes(params, rotor))

        kwargs = self.units(
            params,
            ["orientation", "frequency_type", "frequency_units", "damping_parameter"],
        )
        return result.plot_mode_2d(position, **kwargs)
