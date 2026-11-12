# -*- coding: utf-8 -*-
"""3.2 Unbalance response: Bode at two probes and the deflected shape at the MCS.

The residual unbalance of API 684, Umax = 6350 W / N [g.mm], is spread over the
disks in proportion to their mass, at random phases, as a random residual
unbalance is. The user may add unbalances of their own at any node.
"""

import random

import numpy as np
import ross as rs
from ross.units import Q_

from .base import figure, integer, node, number, residual_unbalance, table


def run(current, params):
    rotor = current.rotor
    start = number(params, "freq_min", "a frequência inicial", minimum=0)
    end = number(params, "freq_max", "a frequência final", minimum=0)
    if end <= start:
        raise ValueError("A frequência final deve ser maior que a inicial.")
    points = integer(params, "points", "o número de pontos", 2, 200)
    mcs = number(params, "mcs", "a velocidade máxima contínua", minimum=1)
    if not start <= mcs <= end:
        raise ValueError("A velocidade máxima contínua deve estar dentro do intervalo de frequência.")
    percent = number(params, "percent", "o % do desbalanceamento permissível", minimum=0, maximum=1000, default=100)
    probes = [
        (node(params, "probe1_node", "o nó do probe 1", rotor), number(params, "probe1_angle", "a fase do probe 1", default=0)),
        (node(params, "probe2_node", "o nó do probe 2", rotor), number(params, "probe2_angle", "a fase do probe 2", default=0)),
    ]
    extra = _extra_unbalances(params.get("unbalances"), rotor)

    total = residual_unbalance(rotor.m, mcs) * percent / 100.0
    disks = rotor.disk_elements
    disk_mass = sum(disk.m for disk in disks)
    rows = []
    for disk in disks:
        share = total * disk.m / disk_mass if disk_mass > 0 else 0.0
        rows.append((int(disk.n), share, random.uniform(0, 360), "Residual (disco)"))
    rows.extend((n, magnitude, phase, "Adicionado") for n, magnitude, phase in extra)
    if not rows:
        raise ValueError("O rotor não tem discos. Adicione um desbalanceamento em algum nó.")

    speeds = np.unique(np.append(np.linspace(start, end, points), mcs))
    speeds_rad = Q_(speeds, "RPM").to("rad/s").m
    response = rotor.run_unbalance_response(
        [row[0] for row in rows],
        Q_(np.array([row[1] for row in rows]), "g*mm"),
        Q_(np.array([row[2] for row in rows]), "deg"),
        speeds_rad,
    )

    bode = response.plot(
        probe=[rs.Probe(n, Q_(angle, "deg"), tag="Probe %d (nó %d)" % (index + 1, n)) for index, (n, angle) in enumerate(probes)],
        frequency_units="RPM",
        amplitude_units="um pkpk",
        phase_units="deg",
    )
    deflected = response.plot_deflected_shape_3d(
        speed=Q_(mcs, "RPM").to("rad/s").m, amplitude_units="um pkpk"
    )
    return {
        "layout": "stack",
        "figures": [figure(bode, height=560), figure(deflected, height=560)],
        "tables": [
            table(
                "Desbalanceamentos aplicados (Umax = %.2f g·mm a %g RPM)" % (total, mcs),
                ["Nó", "Desbalanceamento [g·mm]", "Fase [°]", "Origem"],
                rows,
            )
        ],
    }


def _extra_unbalances(raw, rotor):
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("Lista de desbalanceamentos inválida.")
    extra = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("Desbalanceamento %d inválido." % (index + 1))
        label = "Desbalanceamento %d" % (index + 1)
        extra.append(
            (
                node(item, "node", label + ": o nó", rotor),
                number(item, "magnitude", label + ": a magnitude", minimum=0),
                number(item, "phase", label + ": a fase", default=0),
            )
        )
    return extra
