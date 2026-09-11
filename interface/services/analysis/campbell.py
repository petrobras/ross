# -*- coding: utf-8 -*-
"""Campbell diagram, with the mode shape inside the interface itself."""

import numpy as np

from .base import Runner, register

# ROSS's defaults in _plot_with_mode_shape. Repeated here because the click on
# the diagram arrives in a request of its own, without the object that held
# them.
DEFAULT_UNITS = {
    "speed_units": "RPM",
    "frequency_units": "RPM",
    "damping_parameter": "log_dec",
}


@register
class CampbellRunner(Runner):
    """Campbell, and the 3D mode shape of whichever point the user clicks.

    ## What used to be here

    The mode shape came out of ROSS's `plot_with_mode_shape`, which starts a
    **Dash** server on a random port. To find the port, the interface replaced
    `sys.stdout` with an object that spied on what Dash printed, started a
    thread and polled for up to 10 seconds; the URL became the `src` of an
    iframe inside the card (BE-04).

    Three problems, beyond the ugliness. In a console-less PyInstaller
    executable there is no `sys.stdout` to spy on. If `dash` is not installed,
    ROSS raises ImportError inside the thread, the error becomes a `print`, and
    the user waits the full 10 seconds to be told "it took too long to respond"
    -- the real cause never reaches the screen. And the replaced `sys.stdout`
    was never restored.

    ## What is here now

    None of that is necessary. `plot_with_mode_shape` is a thin shell over
    `_plot_with_mode_shape`, which in turn does only two things: `self.plot(...)`
    -- the same diagram Default mode already draws -- and a callback that, given
    the clicked point, calls `self._update_plot_mode_3d(...)`.

    And that callback reads `self.modal_results`, which `run_campbell` **has
    already filled** with one modal result per speed in the range. In other
    words: the click recomputes nothing; it picks among results already in
    memory.

    So the interface draws the Campbell normally, and a click on a point asks
    for the 3D figure in a request of its own. No Dash, no thread, no stdout, no
    waiting.

    ## And the 15-speed clipping went with it

    The old code reduced the range to 15 speeds in mode-shape mode, because
    `_plot_with_mode_shape` ran an extra `run_modal` for every critical speed,
    up front. Here there is no up-front work at all: the mode shape costs the
    same as Default. The clipping lost its reason to exist, and with it goes the
    reason `plot_type` took part in the computation -- see the history in
    tests/test_campbell.py.
    """

    name = "campbell"

    def spec(self, params, rotor):
        minimum, maximum = self.speed_bounds(params)
        return {
            "speed_min": minimum,
            "speed_max": maximum,
            "steps": self.integer(params, "speed_steps", 50),
            "frequencies": self.integer(params, "frequencies", 6),
            "frequency_type": params.get("frequency_type", "wd"),
            "torsional_analysis": self.flag(params, "torsional_analysis"),
        }

    def compute(self, rotor, spec):
        speeds = np.linspace(spec["speed_min"], spec["speed_max"], spec["steps"])
        return rotor.run_campbell(
            speeds,
            frequencies=spec["frequencies"],
            frequency_type=spec["frequency_type"],
            torsional_analysis=spec["torsional_analysis"],
        )

    def plot(self, result, params, rotor):
        """The diagram, identical in both modes. What changes is what the screen does with it."""
        kwargs = self.units(
            params, ["frequency_units", "speed_units", "damping_parameter"]
        )
        harmonics = self.literal(params, "harmonics")
        if harmonics:
            kwargs["harmonics"] = harmonics
        return result.plot(**kwargs)

    def mode_shape_figure(self, result, params, point):
        """The 3D figure of the mode matching the point clicked on the diagram.

        `x` and `y` arrive in the units the diagram was drawn in, which are the
        same ones ROSS expects here. `curve_name` comes from the clicked trace
        itself: that is how ROSS tells the torsional-analysis curve apart, and
        the browser is what has the name -- there is no need to rebuild the
        figure on the backend just to look it up.
        """
        target = result
        if point.get("curve_name") == "Torsional Analysis":
            torsional = getattr(result, "campbell_torsional", None)
            if torsional is None:
                raise ValueError(
                    "This point belongs to the torsional-analysis curve, which "
                    "was not computed. Turn 'torsional_analysis' on and run again."
                )
            target = torsional

        for axis_name in ("x", "y"):
            if point.get(axis_name) is None:
                raise ValueError(
                    "The clicked point did not carry the %r coordinate." % axis_name
                )

        def unit(key):
            value = params.get(key)
            return value if value not in (None, "") else DEFAULT_UNITS[key]

        # modal_results_crit is left empty on purpose: it is _update_plot_mode_3d's
        # fallback path, used only when self.modal_results lacks the speed -- and
        # run_campbell fills self.modal_results with one entry per speed in the
        # range, so the main path always resolves.
        return target._update_plot_mode_3d(
            point["x"],
            point["y"],
            {},
            unit("speed_units"),
            unit("frequency_units"),
            unit("damping_parameter"),
            self.flag(params, "animation"),
        )
