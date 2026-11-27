# -*- coding: utf-8 -*-
"""3.3 Clearance analysis: the vibration at the bearings against their clearance.

Same procedure as the Digital Twin app: the API residual unbalance at the speed
is placed at the node of largest amplitude of the first forward mode; the
response at the bearings is scaled so that its largest value equals the API
amplitude limit A1 = 25.4 sqrt(12000 / N) [µm pk-pk]; and each bearing's
amplitude is compared with its radial clearance and with 75% of it.
"""

import math

import numpy as np
import plotly.graph_objects as go
import ross as rs
from ross.units import Q_

from .base import figure, number, residual_unbalance, table


def run(current, params):
    rotor = current.rotor
    speed = number(params, "speed", "the speed", minimum=1)
    speed_rad = Q_(speed, "RPM").to("rad/s").m
    bearings = list(rotor.bearing_elements)
    if not bearings:
        raise ValueError("The rotor has no bearings.")

    modal = rotor.run_modal(speed=speed_rad)
    forward = [
        index
        for index, whirl in enumerate(modal.whirl_direction())
        if whirl == "Forward"
    ]
    if not forward:
        raise ValueError("No forward whirl mode at this speed.")
    shape = modal.shapes[forward[0]]
    per_element = max(1, len(shape.major_axis) // len(shape.shaft_elements_length))
    worst_node = min(
        int(np.argmax(shape.major_axis) // per_element), len(rotor.nodes) - 1
    )

    unbalance = residual_unbalance(rotor.m, speed)
    response = rotor.run_unbalance_response(
        worst_node, Q_(unbalance, "g*mm"), 0, np.array([speed_rad])
    )
    probes = [rs.Probe(int(b.n), Q_(0, "deg"), tag="Node %d" % b.n) for b in bearings]
    data = response.data_magnitude(probe=probes, amplitude_units="um pkpk")
    raw = np.array([float(data.loc[0, probe.tag]) for probe in probes])
    if raw.max() <= 0:
        raise ValueError("The response at the bearings is zero at this speed.")
    limit = 25.4 * math.sqrt(12000.0 / speed)
    amplitudes = raw * limit / raw.max()

    clearances = np.array([_clearance_um(b) for b in bearings])
    used = amplitudes / clearances * 100.0
    used_75 = amplitudes / (0.75 * clearances) * 100.0

    labels = [str(int(b.n)) for b in bearings]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=clearances,
        name="Radial clearance (100%)",
        marker_color="red",
        width=0.2,
        hovertemplate="Clearance: %{y:.1f} µm<extra></extra>",
    )
    fig.add_bar(
        x=labels,
        y=0.75 * clearances,
        name="Alert level (75%)",
        marker_color="blue",
        width=0.2,
        hovertemplate="75%%: %{y:.1f} µm<extra></extra>",
    )
    fig.add_scatter(
        x=labels,
        y=amplitudes,
        mode="lines+markers+text",
        name="Response (%g RPM)" % speed,
        text=[
            "%s / %s" % (_percent(a), _percent(b))
            for a, b in zip(used_75, used, strict=True)
        ],
        textposition="top left",
        line={"shape": "spline", "color": "purple", "width": 3},
        hovertemplate="Amplitude: %{y:.2f} µm pk-pk<extra></extra>",
    )
    fig.update_layout(
        title="Vibration response x bearing clearance limits",
        xaxis_title="Station (node)",
        yaxis_title="Clearance and peak amplitude [µm]",
        barmode="overlay",
        hovermode="x unified",
        plot_bgcolor="white",
        yaxis={"showgrid": True, "gridcolor": "lightgray"},
        legend={"orientation": "h", "y": 1.08},
        xaxis={"type": "category"},
    )
    return {
        "layout": "single",
        "figures": [figure(fig, height=520)],
        "tables": [
            table(
                "Bearings (Umax = %.2f g·mm at node %d, A1 = %.1f µm pk-pk)"
                % (unbalance, worst_node, limit),
                [
                    "Node",
                    "Amplitude [µm pk-pk]",
                    "Radial clearance [µm]",
                    "% of clearance",
                    "% of 75% of clearance",
                ],
                [
                    [int(b.n), a, c, u, u75]
                    for b, a, c, u, u75 in zip(
                        bearings, amplitudes, clearances, used, used_75, strict=True
                    )
                ],
            )
        ],
    }


def _clearance_um(bearing):
    value = getattr(bearing, "radial_clearance", None)
    if value is None:
        return float("nan")
    microns = np.atleast_1d(Q_(value, "m").to("micron").m).astype(float)
    return float(np.max(microns))


def _percent(value):
    return "—" if not math.isfinite(value) else "%.1f%%" % value
