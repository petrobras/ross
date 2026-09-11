# -*- coding: utf-8 -*-
"""Time-domain analyses: forced response and ROSS's three faults.

All four share the same plotting tail (1D, 2D, 3D and DFFT over probes), which
lives in `TransientRunner`. What differs between them is only the computation.
"""

import numpy as np

from .base import Runner, register
from domain.conversion import get_converted_param
from services.expressions import safe_expression_eval

# The row used when the unbalance table is empty. The cells themselves fall
# back to zero -- that is what the previous version did.
DEFAULT_LINE = {"node": 0, "mag": 5e-4, "phase": -1.57}


class TransientRunner(Runner):
    """Base of the four: same time and unbalance reading, same plot."""

    def time_and_unbalance(self, params, speed_default):
        nodes, magnitudes, phases = self.unbalances(params, DEFAULT_LINE, 0.0, 0.0)
        return {
            "speed": get_converted_param(params, "speed", speed_default, "rad/s"),
            "t_initial": self.number(params, "t_initial", 0.0),
            "t_final": self.number(params, "t_final", 0.5),
            "t_steps": self.integer(params, "t_steps", 5000),
            "node": nodes,
            "unbalance_magnitude": magnitudes,
            "unbalance_phase": phases,
        }

    @staticmethod
    def time_vector(spec):
        return np.linspace(spec["t_initial"], spec["t_final"], spec["t_steps"])

    def plot(self, result, params, rotor):
        kind = params.get("plot_type", "1D")
        probes = self.probes(params)

        if kind == "Frequency (DFFT)":
            return result.plot_dfft(
                probe=probes,
                **self.units(
                    params, ["probe_units", "displacement_units", "frequency_units"]
                ),
            )
        if kind == "2D":
            lines = params.get("probes") or [{"node": 0}]
            return result.plot_2d(
                node=lines[0]["node"], **self.units(params, ["displacement_units"])
            )
        if kind == "3D":
            return result.plot_3d(
                **self.units(params, ["displacement_units", "rotor_length_units"])
            )
        return result.plot_1d(
            probe=probes,
            **self.units(params, ["probe_units", "displacement_units", "time_units"]),
        )


@register
class TimeResponseRunner(TransientRunner):
    name = "time_response"

    def spec(self, params, rotor):
        forces = params.get("forces") or []
        if not forces:
            # With no force the response is zero at every node -- a flat chart that
            # looks like a result. The same reasoning as BE-05, just below: a null
            # excitation must not pass for a computation.
            raise ValueError(
                "Add at least one applied force F(t) to run the time response."
            )

        return {
            "speed": get_converted_param(params, "speed", 100.0, "rad/s"),
            "t_max": self.number(params, "t_max", 1.0),
            "steps": self.integer(params, "steps", 1000),
            "forces": forces,
            "method": params.get("method", "default"),
            # The number of degrees of freedom decides where each force enters the
            # global vector, so it is part of what gets computed.
            "ndof": int(rotor.ndof),
            "dofs_per_node": int(rotor.number_dof),
            "n_nodes": len(rotor.nodes),
        }

    def compute(self, rotor, spec):
        time_grid = np.linspace(0, spec["t_max"], spec["steps"])
        forces = np.zeros((len(time_grid), spec["ndof"]))
        variables = {"t": time_grid, "speed": spec["speed"]}

        for force in spec["forces"]:
            node_index = min(int(force.get("node", 0)), spec["n_nodes"] - 1)
            dof = int(force.get("dof", 0))
            global_dof = node_index * spec["dofs_per_node"] + dof
            expression = str(force.get("func", "0")).strip() or "0"
            try:
                forces[:, global_dof] += safe_expression_eval(expression, variables)
            except Exception as exc:
                # BE-05: this used to be `except: pass` and the analysis ran with a null
                # excitation -- a wrong result, with no warning.
                raise ValueError(
                    f"Invalid force at node {node_index}, DoF {dof}: '{expression}' -- {exc}"
                )

        return rotor.run_time_response(
            spec["speed"], forces, time_grid, method=spec["method"]
        )


@register
class MisalignmentRunner(TransientRunner):
    """Coupling misalignment.

    ROSS reads this one's parameters with `kwargs.get(...)`, without defaults:
    whatever is missing arrives as None and the arithmetic blows up inside with
    `unsupported operand type(s) for ** or pow(): 'NoneType' and 'int'` -- a
    message that does not say which field was left blank. That is why the
    requirement lives here, naming the field, following the slice's own rule: a
    physical parameter is mandatory.

    Which ones are mandatory depends on the misalignment type, because ROSS
    picks the force function from it (MisalignmentFlex.__init__): parallel uses
    the two distances and the radial stiffness, angular uses the angle and the
    bending stiffness, and combined uses all five.
    """

    name = "misalignment"

    # Genuinely optional fields: ROSS has a default for both (zero).
    OPTIONAL = ("input_torque", "load_torque")

    FLEX_FIELDS = (
        "mis_distance_x",
        "mis_distance_y",
        "mis_angle",
        "radial_stiffness",
        "bending_stiffness",
    )

    REQUIRED_BY_TYPE = {
        "parallel": ("mis_distance_x", "mis_distance_y", "radial_stiffness"),
        "angular": ("mis_angle", "bending_stiffness"),
        "combined": (
            "mis_distance_x",
            "mis_distance_y",
            "mis_angle",
            "radial_stiffness",
            "bending_stiffness",
        ),
    }

    def spec(self, params, rotor):
        base = self.time_and_unbalance(params, 125.66)
        coupling = params.get("coupling", "flex")
        extras = {"coupling": coupling}

        for probe_name in self.OPTIONAL:
            if self.text(params, probe_name) is not None:
                extras[probe_name] = self.number(params, probe_name, 0.0)

        if coupling == "flex":
            extras["n"] = self.required_integer(params, "n", "misalignment")
            kind = self.text(params, "mis_type")
            if kind not in self.REQUIRED_BY_TYPE:
                # ROSS's message is "Check the misalignment type!", which does not
                # say which ones exist.
                raise ValueError(
                    "Invalid misalignment type: %r. Choose one of %s."
                    % (kind, ", ".join(sorted(self.REQUIRED_BY_TYPE)))
                )
            extras["mis_type"] = kind

            required_here = self.REQUIRED_BY_TYPE[kind]
            for probe_name in self.FLEX_FIELDS:
                if probe_name in required_here:
                    extras[probe_name] = self.required_number(
                        params, probe_name, "misalignment"
                    )
                elif self.text(params, probe_name) is not None:
                    # Passed along even when this type does not require it. ROSS keeps all
                    # five on the fault object and decides internally which to read; which
                    # one it uses is not the interface's decision, and a future release may
                    # change it.
                    extras[probe_name] = self.number(params, probe_name, 0.0)
        else:
            extras["n"] = self.required_integer(params, "n", "misalignment")
            extras["mis_distance"] = self.required_number(
                params, "mis_distance", "misalignment"
            )

        base["extras"] = extras
        return base

    def compute(self, rotor, spec):
        return rotor.run_misalignment(
            node=spec["node"],
            unbalance_magnitude=spec["unbalance_magnitude"],
            unbalance_phase=spec["unbalance_phase"],
            speed=spec["speed"],
            t=self.time_vector(spec),
            **spec["extras"],
        )


@register
class RubbingRunner(TransientRunner):
    name = "rubbing"

    def spec(self, params, rotor):
        base = self.time_and_unbalance(params, 125.66)
        base.update(
            {
                "n": self.required_integer(params, "n", "rubbing"),
                "distance": self.required_number(params, "distance", "rubbing"),
                "contact_stiffness": self.required_number(
                    params, "contact_stiffness", "rubbing"
                ),
                "contact_damping": self.required_number(
                    params, "contact_damping", "rubbing"
                ),
                "friction_coeff": self.required_number(
                    params, "friction_coeff", "rubbing"
                ),
                "torque": self.flag(params, "torque", False),
            }
        )
        return base

    def compute(self, rotor, spec):
        return rotor.run_rubbing(
            n=spec["n"],
            distance=spec["distance"],
            contact_stiffness=spec["contact_stiffness"],
            contact_damping=spec["contact_damping"],
            friction_coeff=spec["friction_coeff"],
            node=spec["node"],
            unbalance_magnitude=spec["unbalance_magnitude"],
            unbalance_phase=spec["unbalance_phase"],
            speed=spec["speed"],
            t=self.time_vector(spec),
            torque=spec["torque"],
        )


@register
class CrackRunner(TransientRunner):
    name = "crack"

    def spec(self, params, rotor):
        base = self.time_and_unbalance(params, 125.66)
        base.update(
            {
                "n": self.required_integer(params, "n", "crack"),
                # Zero depth is an intact rotor: with no value, the crack analysis would
                # return a result with no crack at all.
                "depth_ratio": self.required_number(params, "depth_ratio", "crack"),
                "crack_model": params.get("crack_model", "Mayes"),
                "cross_divisions": (
                    self.integer(params, "cross_divisions", 0)
                    if self.text(params, "cross_divisions") is not None
                    else None
                ),
            }
        )
        return base

    def compute(self, rotor, spec):
        kwargs = {"crack_model": spec["crack_model"]}
        if spec["cross_divisions"] is not None:
            kwargs["cross_divisions"] = spec["cross_divisions"]
        return rotor.run_crack(
            n=spec["n"],
            depth_ratio=spec["depth_ratio"],
            node=spec["node"],
            unbalance_magnitude=spec["unbalance_magnitude"],
            unbalance_phase=spec["unbalance_phase"],
            speed=spec["speed"],
            t=self.time_vector(spec),
            **kwargs,
        )
