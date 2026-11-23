# -*- coding: utf-8 -*-
"""3.4 Stability: whirl, damped frequency and log decrement at one speed.

Two 4-DoF models, as in the Digital Twin app: rotor and bearings only, and rotor
with bearings and seals. The difference between the two is what the seals do
to the stability.
"""

import ross as rs
from ross.units import Q_
from ross.utils import convert_6dof_to_4dof

from .base import number, table

COLUMNS = ["Mode", "Whirl", "Frequency [Hz]", "Logarithmic decrement"]


def run(current, params):
    rotor = current.rotor
    speed = number(params, "speed", "the speed", minimum=0)
    speed_rad = Q_(speed, "RPM").to("rad/s").m

    bearings_only = [
        b for b in rotor.bearing_elements if not isinstance(b, rs.SealElement)
    ]
    without_seals = rs.Rotor(
        rotor.shaft_elements,
        rotor.disk_elements,
        bearings_only,
        point_mass_elements=rotor.point_mass_elements,
    )
    return {
        "layout": "tables",
        "tables": [
            _modes("Rotor + Bearings", convert_6dof_to_4dof(without_seals), speed_rad),
            _modes("Rotor + Bearings + Seals", convert_6dof_to_4dof(rotor), speed_rad),
        ],
    }


def _modes(title, rotor, speed_rad):
    modal = rotor.run_modal(speed_rad)
    frequencies = Q_(modal.wd, "rad/s").to("Hz").m
    rows = [
        [index + 1, str(whirl), frequency, log_dec]
        for index, (whirl, frequency, log_dec) in enumerate(
            zip(modal.whirl_direction(), frequencies, modal.log_dec, strict=True)
        )
    ]
    return table(title, COLUMNS, rows)
