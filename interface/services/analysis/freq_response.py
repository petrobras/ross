# -*- coding: utf-8 -*-
"""Frequency response, with one curve per input/output pair."""

import numpy as np

from .base import Runner, register

COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

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
            "free_free": self.flag(params, "free_free"),
            "modes": self.literal(params, "modes"),
        }

    def compute(self, rotor, spec):
        speeds = np.linspace(spec["speed_min"], spec["speed_max"], 50)
        kwargs = {"free_free": spec["free_free"]}
        if spec["modes"]:
            kwargs["modes"] = spec["modes"]
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
            g_out = output["node"] * degrees_per_node + output["dof"]

            partial = getattr(result, method)(inp=g_inp, out=g_out, **kwargs)
            colour = COLOURS[i % len(COLOURS)]
            for j, trace in enumerate(partial.data):
                trace.name = (
                    f"In(N{entry['node']} D{entry['dof']}) | "
                    f"Out(N{output['node']} D{output['dof']})"
                )
                trace.legendgroup = f"group_{i}"
                trace.showlegend = j == 0
                if hasattr(trace, "line") and trace.line is not None:
                    trace.line.color = colour

            if figure is None:
                figure = partial
            else:
                figure.add_traces(partial.data)
        return figure
