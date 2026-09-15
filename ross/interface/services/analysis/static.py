# -*- coding: utf-8 -*-
"""Static analysis: deflection, shear, moment and free-body diagram."""

from .base import Runner, register


@register
class StaticRunner(Runner):
    name = "static"

    def spec(self, params, rotor):
        # run_static takes no parameters at all: an empty spec already identifies
        # the computation, because the rotor enters the key from outside.
        return {}

    def compute(self, rotor, spec):
        return rotor.run_static()

    def plot(self, result, params, rotor):
        kind = params.get("plot_type", "Free Body Diagram")
        if kind == "Deformation":
            return result.plot_deformation(
                **self.units(params, ["deformation_units", "rotor_length_units"])
            )
        if kind == "Shearing Force":
            return result.plot_shearing_force(
                **self.units(params, ["force_units", "rotor_length_units"])
            )
        if kind == "Bending Moment":
            return result.plot_bending_moment(
                **self.units(params, ["moment_units", "rotor_length_units"])
            )
        return result.plot_free_body_diagram(
            **self.units(params, ["force_units", "rotor_length_units"])
        )
