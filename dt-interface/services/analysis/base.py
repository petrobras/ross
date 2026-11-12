# -*- coding: utf-8 -*-
"""What the Digital Twin analyses share: reading the form and the API 684 numbers."""

import json
import math

import numpy as np

# API 684 / API 617 residual unbalance, Umax = 6350 W / N  [g.mm], with W in kg
# and N the maximum continuous speed in RPM. The standard caps N at 25000 RPM.
API_UNBALANCE_FACTOR = 6350.0
API_SPEED_CAP = 25000.0


def number(params, key, label, minimum=None, maximum=None, default=None):
    """Read a number from the form. A blank field takes `default`, or is refused."""
    raw = params.get(key, default)
    if raw is None or str(raw).strip() == "":
        if default is None:
            raise ValueError("Informe %s." % label)
        raw = default
    try:
        value = float(str(raw).replace(",", "."))
    except (TypeError, ValueError):
        raise ValueError("%s: informe um número." % label)
    if not math.isfinite(value):
        raise ValueError("%s: informe um número finito." % label)
    if minimum is not None and value < minimum:
        raise ValueError("%s deve ser no mínimo %g." % (label, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("%s deve ser no máximo %g." % (label, maximum))
    return value


def integer(params, key, label, minimum, maximum, default=None):
    value = number(params, key, label, minimum, maximum, default)
    if value != int(value):
        raise ValueError("%s precisa ser um inteiro." % label)
    return int(value)


def node(params, key, label, rotor):
    return integer(params, key, label, 0, len(rotor.nodes) - 1)


def integer_list(params, key, label, minimum, maximum):
    """Read '1, 2, 3' or a list into integers."""
    raw = params.get(key)
    if isinstance(raw, list):
        items = raw
    else:
        items = [item for item in str(raw or "").replace(";", ",").split(",") if item.strip()]
    values = []
    for item in items:
        try:
            value = int(str(item).strip())
        except ValueError:
            raise ValueError("%s: '%s' não é um inteiro." % (label, item))
        if value < minimum or value > maximum:
            raise ValueError("%s: use valores entre %d e %d." % (label, minimum, maximum))
        values.append(value)
    return sorted(set(values))


def residual_unbalance(rotor_mass, speed_rpm):
    """API residual unbalance [g.mm] for the whole rotor."""
    return API_UNBALANCE_FACTOR * rotor_mass / min(speed_rpm, API_SPEED_CAP)


def figure(fig, **layout):
    """A Plotly figure as JSON, sized by the card rather than by ROSS."""
    fig.update_layout(autosize=True, width=None, **layout)
    return json.loads(fig.to_json())


def table(title, columns, rows):
    return {"title": title, "columns": columns, "rows": [[_cell(value) for value in row] for row in rows]}


def _cell(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, np.integer):
        return int(value)
    return value
