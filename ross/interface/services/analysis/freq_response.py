# -*- coding: utf-8 -*-
"""Frequency response, with one curve per input/output pair."""

import numpy as np

from .base import Runner, register

COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

NO_LATERAL = (
    "Probe '%s' is radial, and this rotor model has no lateral degree of freedom "
    "to read it in."
)
NO_AXIAL = (
    "Probe '%s' is axial, and this rotor model has no axial degree of freedom: "
    "run the analysis on the full model."
)


def probe_output(result, node, angle, dofs_per_node):
    """The frequency response a radial probe reads, as results ROSS can plot.

    ROSS's `FrequencyResponseResults` plots one degree of freedom as the
    output; a probe at `angle` reads x cos(angle) + y sin(angle) of its node,
    which is the response's projection in its direction -- the same reading
    the unbalance response makes of a probe. The system is linear, so the
    projection of the response is the response of the projection, and output
    0 of the results returned here is exactly that, for every input. They
    keep the input axis whole and ROSS's own plotting and units.

    `domain/python_export.py` writes this same function into the script
    (`PROBE_OUTPUT`); `tests/test_freq_response_probes.py` holds the two to
    the same numbers.
    """
    from ross.results import FrequencyResponseResults

    x = node * dofs_per_node
    c, s = np.cos(angle), np.sin(angle)

    def along(response):
        return (c * response[:, x, :] + s * response[:, x + 1, :])[:, None, :]

    return FrequencyResponseResults(
        along(result.freq_resp),
        along(result.velc_resp),
        along(result.accl_resp),
        result.speed_range,
        result.number_dof,
    )


METHODS = {
    "Default": "plot",
    "Magnitude": "plot_magnitude",
    "Phase": "plot_phase",
    "Polar Bode": "plot_polar_bode",
}


@register
class FreqResponseRunner(Runner):
    name = "freq_response"

    def spec(self, params, rotor):
        minimum, maximum = self.speed_bounds(params)
        return {
            "speed_min": minimum,
            "speed_max": maximum,
            "steps": self.steps(params, "speed_steps", 50),
            "free_free": self.flag(params, "free_free"),
            "modes": self.literal(params, "modes"),
            "speed": self.optional_quantity(params, "speed", "rad/s"),
        }

    def compute(self, rotor, spec):
        speeds = np.linspace(spec["speed_min"], spec["speed_max"], spec["steps"])
        kwargs = {"free_free": spec["free_free"]}
        if spec["modes"]:
            kwargs["modes"] = spec["modes"]
        # Absent, not None: with the rotor speed given, `speeds` becomes the
        # excitation sweep; without it ROSS keeps the synchronous sweep.
        if spec["speed"] is not None:
            kwargs["speed"] = spec["speed"]
        return rotor.run_freq_response(speeds, **kwargs)

    def plot(self, result, params, rotor):
        kind = params.get("plot_type", "Default")
        method = METHODS.get(kind, "plot")

        kwargs = self.units(params, ["frequency_units", "amplitude_units"])
        if kind in ("Default", "Phase", "Polar Bode"):
            kwargs.update(self.units(params, ["phase_units"]))
        if kind == "Magnitude":
            kwargs.update(self.units(params, ["line_shape"]))

        # The input/output pairs are walked in parallel; the shorter list
        # repeats its last item to keep up with the longer one.
        #
        # `or`, not `get(key, default)`: the screen sends `[]` when the user
        # deletes every row of the table, and the key exists -- so `get`'s default
        # never applied. The result was a loop that did not run, `figure` staying
        # None, and the route blowing up with "'NoneType' object has no attribute
        # 'update_layout'" -- an error pointing at Plotly when the problem is an
        # empty table. The helpers in `base.py` (`probes`, `unbalances`) always used
        # `or`; this was the only one off the pattern.
        entries = params.get("inps") or [{"node": 0, "dof": 0}]
        outputs = params.get("outs") or [{"node": 0, "dof": 0}]
        count = max(len(entries), len(outputs))
        entries = (
            entries + [entries[-1]] * (count - len(entries))
            if entries
            else [{"node": 0, "dof": 0}] * count
        )
        outputs = (
            outputs + [outputs[-1]] * (count - len(outputs))
            if outputs
            else [{"node": 0, "dof": 0}] * count
        )

        degrees_per_node = rotor.number_dof
        figure = None
        for i in range(count):
            entry, output = entries[i], outputs[i]
            g_inp = entry["node"] * degrees_per_node + entry["dof"]
            source, g_out, read_at = self.output(result, output, degrees_per_node)

            partial = getattr(source, method)(inp=g_inp, out=g_out, **kwargs)
            colour = COLOURS[i % len(COLOURS)]
            for j, trace in enumerate(partial.data):
                trace.name = f"In(N{entry['node']} D{entry['dof']}) | Out({read_at})"
                trace.legendgroup = f"group_{i}"
                trace.showlegend = j == 0
                if hasattr(trace, "line") and trace.line is not None:
                    trace.line.color = colour

            if figure is None:
                figure = partial
            else:
                figure.add_traces(partial.data)
        return figure

    @staticmethod
    def output(result, row, degrees_per_node):
        """What an output row is read from: `(results, out, label)`.

        A typed row is a degree of freedom of a node, as it always was. A row
        that named a probe of the model has been filled in from it
        (domain/probe_refs.py): an axial probe reads z of its node; a radial
        one reads its direction, from `probe_output`.
        """
        if not row.get("probe"):
            g_out = row["node"] * degrees_per_node + row["dof"]
            return result, g_out, f"N{row['node']} D{row['dof']}"
        name = row.get("tag") or "probe at node %d" % row["node"]
        if row.get("direction") == "axial":
            if degrees_per_node < 6:
                raise ValueError(NO_AXIAL % name)
            return result, row["node"] * degrees_per_node + 2, name
        if degrees_per_node < 2:
            raise ValueError(NO_LATERAL % name)
        view = probe_output(result, row["node"], row["angle"] or 0.0, degrees_per_node)
        return view, 0, name
