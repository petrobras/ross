# -*- coding: utf-8 -*-
"""3.1 Campbell diagram, with the 3D mode of the clicked point.

The Streamlit page used `_plot_with_mode_shape` and kept its callback in
`st.session_state`. Here the diagram is `CampbellResults.plot`, and the click
calls `_update_plot_mode_3d` on the result kept in the session: `run_campbell`
already holds one modal result per speed, so the click computes nothing new.
This is how the ROSS interface does it (services/analysis/campbell.py).
"""

import numpy as np
from ross.units import Q_

from .base import figure, integer, integer_list, number

UNITS = {"speed_units": "RPM", "frequency_units": "RPM", "damping_parameter": "log_dec"}


def run(current, params):
    rotor = current.rotor
    start = number(params, "speed_min", "a velocidade inicial", minimum=0)
    end = number(params, "speed_max", "a velocidade final", minimum=0)
    if end <= start:
        raise ValueError("A velocidade final deve ser maior que a inicial.")
    points = integer(params, "points", "o número de pontos", 2, 200)
    harmonics = integer_list(params, "harmonics", "Harmônicos", 1, 15) or [1]

    speeds = Q_(np.linspace(start, end, points), "RPM").to("rad/s").m
    result = rotor.run_campbell(speeds)
    current.campbell = result

    fig = result.plot(harmonics=harmonics, frequency_units=UNITS["frequency_units"])
    return {"layout": "campbell", "figures": [figure(fig, height=560)]}


def mode_shape(current, point):
    """The 3D mode of a point clicked on the diagram, in the diagram's units."""
    result = getattr(current, "campbell", None)
    if result is None:
        raise ValueError("Execute o diagrama de Campbell antes de escolher um ponto.")
    if not isinstance(point, dict):
        raise ValueError("Ponto inválido.")
    try:
        x = float(point.get("x"))
        y = float(point.get("y"))
    except (TypeError, ValueError):
        raise ValueError("O ponto clicado não tem coordenadas numéricas.")
    fig = result._update_plot_mode_3d(
        x, y, {}, UNITS["speed_units"], UNITS["frequency_units"], UNITS["damping_parameter"], False
    )
    return {"figure": figure(fig, height=560)}
