# -*- coding: utf-8 -*-
"""3.1 Campbell diagram, with the 3D mode of the clicked point.

The diagram is `CampbellResults.plot`, and the click calls
`_update_plot_mode_3d` on the result kept in the session: `run_campbell`
already holds one modal result per speed, so the click computes nothing new.
This is how the ROSS interface does it (services/analysis/campbell.py).
"""

import numpy as np
from ross.units import Q_

from .base import figure, integer, integer_list, number

UNITS = {"speed_units": "RPM", "frequency_units": "RPM", "damping_parameter": "log_dec"}


def run(current, params):
    rotor = current.rotor
    start = number(params, "speed_min", "the initial speed", minimum=0)
    end = number(params, "speed_max", "the final speed", minimum=0)
    if end <= start:
        raise ValueError("The final speed must be greater than the initial one.")
    points = integer(params, "points", "the number of points", 2, 200)
    harmonics = integer_list(params, "harmonics", "Harmonics", 1, 15) or [1]

    speeds = Q_(np.linspace(start, end, points), "RPM").to("rad/s").m
    result = rotor.run_campbell(speeds)
    current.campbell = result

    fig = result.plot(harmonics=harmonics, frequency_units=UNITS["frequency_units"])
    return {"layout": "campbell", "figures": [figure(fig, height=560)]}


def mode_shape(current, point):
    """The 3D mode of a point clicked on the diagram, in the diagram's units."""
    result = getattr(current, "campbell", None)
    if result is None:
        raise ValueError("Run the Campbell diagram before picking a point.")
    if not isinstance(point, dict):
        raise ValueError("Invalid point.")
    try:
        x = float(point.get("x"))
        y = float(point.get("y"))
    except (TypeError, ValueError):
        raise ValueError("The clicked point has no numeric coordinates.")
    fig = result._update_plot_mode_3d(
        x,
        y,
        {},
        UNITS["speed_units"],
        UNITS["frequency_units"],
        UNITS["damping_parameter"],
        False,
    )
    return {"figure": figure(fig, height=560)}
