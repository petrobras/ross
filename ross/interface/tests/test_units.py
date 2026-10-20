# -*- coding: utf-8 -*-
"""The unit of each parameter, and the alternatives the screen offers.

Before Phase 1 this dictionary existed twice, word for word: in `app.py` and in
the JS. BE-13 of the audit was one of the copies growing stale.

The guard that would have caught BE-13 on its own, on the day of the refactor,
is the first one below: every mapped unit has to point at a parameter the ROSS
class really accepts."""

import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import ross as rs

from ross.interface.domain import units
from ross.interface.domain.element_registry import resolve_constructor


def test_every_mapped_unit_is_a_real_ross_parameter():
    """This test would have caught BE-13 on its own, on the day of the bearing refactor."""
    problems = []
    for ross_class, mapping in units.UNITS_MAPPING.items():
        # "Class.method" is an alternative constructor (DiskElement.from_geometry):
        # its own signature is the one that has to take the parameter.
        target = resolve_constructor(ross_class)
        if target is None:
            problems.append(f"rs.{ross_class} does not exist")
            continue
        method = target if "." in ross_class else target.__init__
        signature = set(inspect.signature(method).parameters)
        for parameter in mapping:
            if parameter not in signature:
                problems.append(f"{ross_class}.{parameter}")
    assert not problems, "parameters that no longer exist in ROSS: " + ", ".join(
        problems
    )


def test_every_default_unit_has_alternatives():
    for mapping in units.UNITS_MAPPING.values():
        for unit in mapping.values():
            assert units.alternatives_for(unit), f"no alternatives for {unit}"
            assert units.alternatives_for(unit)[0] == unit


# --- the element list's filter --------------------------------------------------
#
# The filter compares a field's values in one unit: "mass > 10 kg" has to find a
# disk typed as 25 lb. The screen does not convert units -- pint does, here --
# and what cannot be read as one number is left out of the comparison rather
# than guessed at.

from ross.interface.domain.conversion import to_unit  # noqa: E402


def test_the_filter_reads_a_typed_value_in_the_unit_it_asks_for():
    assert abs(to_unit("250", "mm", "m") - 0.25) < 1e-12
    assert abs(to_unit(" 25 ", "lb", "kg") - 11.33980925) < 1e-9
    assert abs(to_unit("2*pi", None, None) - 6.283185307179586) < 1e-12
    assert to_unit("3", "", "") == 3.0


def test_what_is_not_one_number_is_not_compared():
    for typed in ["", None, "[1e6, 2e6]", "{'a': 1}", "(0.1, -0.1)", "oil"]:
        assert to_unit(typed, "mm", "m") is None, typed
    assert to_unit("10", "kg", "mm") is None, "units that do not convert"


def test_the_route_answers_in_the_order_asked():
    from ross.interface.api.security import SESSION_TOKEN
    from ross.interface.app import app

    app.config["TESTING"] = True
    items = [
        {"value": "1", "unit": "m", "to": "mm"},
        {"value": "[1, 2]", "unit": "N/m", "to": "N/m"},
        {"value": "22.0462", "unit": "lb", "to": "kg"},
    ]
    with app.test_client() as client:
        answer = client.post(
            "/api/units/convert",
            json={"items": items},
            headers={"X-ROSS-Token": SESSION_TOKEN},
        )
        refused = client.post(
            "/api/units/convert",
            json={"items": ["250"]},
            headers={"X-ROSS-Token": SESSION_TOKEN},
        )
    values = answer.get_json()["values"]
    assert answer.status_code == 200
    assert values[0] == 1000.0 and values[1] is None and abs(values[2] - 10.0) < 1e-4
    assert (
        refused.status_code == 400
        and "value, unit and to" in refused.get_json()["message"]
    )
