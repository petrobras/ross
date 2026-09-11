# -*- coding: utf-8 -*-
"""The shape of an analysis runner.

Until Phase 2 the twelve analyses lived in a 460-line `if/elif` inside the
route, repeating the same sequence twelve times: read the parameters, build the
kwargs, check the cache, run, build the plotting kwargs, plot. Beyond the size,
that blocked Phase 4 -- a job queue needs one callable per analysis, not a
branch of an `if`.

## The separation that matters

Each runner has three halves:

* **`spec(params, rotor)`** -> a JSON-serialisable dict with everything the
  computation needs, and nothing beyond it;
* **`compute(rotor, spec)`** -> the ROSS result object;
* **`plot(result, params, rotor)`** -> the Plotly figure.

The point is that **`compute` receives only the `spec`, never the `params`**.
That is not style: the cache key is the fingerprint of the `spec` itself, so a
computation cannot depend on anything left out of the key. The key used to be
"the params minus a hand-kept list of plotting keys", and the list was wrong:
Campbell read `plot_type` during the computation (to clip the speeds to 15 in
mode-shape mode) while `plot_type` sat outside the key. Opening Mode Shape and
going back to Default returned a 15-point diagram with the form still saying
50 -- and no error on screen.

`plot`, by contrast, receives the whole `params`: choosing an axis unit or a
chart type must never recompute anything.
"""

import ast

from domain.analysis_catalog import plot_parameters
from domain.conversion import get_converted_param


class Runner:
    """Base of an analysis. Subclasses declare `name` and implement the three."""

    name = ""

    # Parameters that ONLY the drawing reads, in this analysis. Not a global
    # list: `frequency_type` is computation in Campbell and drawing in Modal.
    # A test spies on `spec`'s reads and requires none of these keys to show
    # up there -- if one did, picking another chart option would return the
    # result of a different configuration, silently.
    #
    # Until slice 4 this was a tuple written by hand in each runner, next to a
    # form declaring the same names in another language. The two lists drifted
    # apart: `probes`, `inps` and `outs` are read in `plot` and were in no
    # tuple at all, so the guard above did not cover them. Now it comes from
    # the catalog, which is where the field is declared.
    @property
    def PLOT_PARAMS(self):
        return plot_parameters(self.name)

    def spec(self, params, rotor):
        raise NotImplementedError

    def compute(self, rotor, spec):
        raise NotImplementedError

    def plot(self, result, params, rotor):
        raise NotImplementedError

    # --- shared helpers -----------------------------------------------------

    @staticmethod
    def flag(params, key, default=False):
        """Read a boolean <select>, which arrives as the text 'True' or 'False'."""
        return str(params.get(key, default)).lower() == "true"

    @staticmethod
    def literal(params, key):
        """Read a field carrying a list or tuple as text. None when empty."""
        value = params.get(key, "")
        if not value:
            return None
        try:
            return ast.literal_eval(value)
        except Exception:
            return None

    @staticmethod
    def number(params, key, default):
        value = params.get(key, default)
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)

    @staticmethod
    def integer(params, key, default):
        return int(Runner.number(params, key, default))

    # A blank field either falls back to the default OR raises, and the
    # difference matters.
    #
    # The previous version always raised -- `float('')` -- with the message
    # "could not convert string to float: ''", which does not say which
    # field. Replacing that with a default everywhere would be worse: a blank
    # `depth_ratio` would become a zero-depth crack, and the crack analysis
    # would return an intact rotor without saying anything. That is the
    # antipattern this whole refactoring chases -- a wrong result in place of
    # an error.
    #
    # The rule: a discretisation parameter (how many steps, how many modes)
    # has a default; a physical parameter (a distance, a node, a depth) is
    # required and the error names the field.

    @staticmethod
    def required_number(params, key, analysis):
        value = params.get(key)
        if value is None or str(value).strip() == "":
            raise ValueError(
                "Field '%s' is required by the %s analysis and is empty."
                % (key, analysis)
            )
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Field '%s' of the %s analysis needs a number; got %r."
                % (key, analysis, value)
            )

    @staticmethod
    def required_integer(params, key, analysis):
        return int(Runner.required_number(params, key, analysis))

    @staticmethod
    def text(params, key, default=None):
        """Return the field's text, or the default when empty."""
        value = params.get(key)
        if value is None or str(value).strip() == "":
            return default
        return value

    @staticmethod
    def units(params, keys):
        """Gather the unit parameters that were filled in, to pass to plot."""
        return {
            cache_key: params[cache_key]
            for cache_key in keys
            if cache_key in params and str(params[cache_key]).strip() != ""
        }

    @staticmethod
    def probes(params):
        """Turn the screen's probe list into rs.Probe objects."""
        import ross as rs

        probe_rows = params.get("probes") or [{"node": 0, "angle": 0.0}]
        return [
            rs.Probe(int(s["node"]), float(s.get("angle", 0.0))) for s in probe_rows
        ]

    @staticmethod
    def unbalances(
        params,
        fallback,
        mag_default,
        phase_default,
        clamp_rotor=None,
        analysis=None,
    ):
        """Read the unbalance table into three parallel lists.

        `fallback` is the row used when the table is empty; `mag_default` and
        `phase_default` are each cell's default. The two are distinct on
        purpose -- the unbalance response and the fault analyses use different
        values, and the previous version of this code already did so.

        `analysis` turns the fallback off: named, an empty table raises with the
        analysis in the message instead of quietly standing in a row nobody
        typed. Clearance asks for that because the unbalance IS the excitation
        being measured against the bearing gap -- inventing one would answer a
        question the user did not ask, and the answer would look valid.
        """
        lines = params.get("unbalances")
        if not lines and analysis is not None:
            raise ValueError(
                "%s needs at least one unbalance: add a row to the table" % analysis
            )
        lines = lines or [fallback]
        nodes = [int(line.get("node", 0)) for line in lines]
        if clamp_rotor is not None:
            last = len(clamp_rotor.nodes) - 1
            nodes = [min(node, last) for node in nodes]
        return (
            nodes,
            [float(line.get("mag", mag_default)) for line in lines],
            [float(line.get("phase", phase_default)) for line in lines],
        )

    @staticmethod
    def speed_bounds(params):
        """The sweep's speed range, already in rad/s.

        What goes into the spec is the two ends and the number of steps, not
        the vector: the cache key is the spec's fingerprint, and a vector of 50
        floats would only make it bigger without saying anything more.
        """
        return (
            get_converted_param(params, "speed_min", 0.0, "rad/s"),
            get_converted_param(params, "speed_max", 400.0, "rad/s"),
        )


REGISTRY = {}


def register(runner_cls):
    """Register a runner under the name the interface uses."""
    runner = runner_cls()
    assert runner.name, "runner with no name: %r" % runner_cls
    assert runner.name not in REGISTRY, "duplicate runner: %s" % runner.name
    REGISTRY[runner.name] = runner
    return runner_cls


def get_runner(analysis_type):
    runner = REGISTRY.get(analysis_type)
    if runner is None:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError("Unknown analysis: %r. Available: %s" % (analysis_type, known))
    return runner
