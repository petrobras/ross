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

from domain import units


def test_every_mapped_unit_is_a_real_ross_parameter():
    """This test would have caught BE-13 on its own, on the day of the bearing refactor."""
    problems = []
    for ross_class, mapping in units.UNITS_MAPPING.items():
        cls = getattr(rs, ross_class, None)  # noqa: F405
        if cls is None:
            problems.append(f"class rs.{ross_class} does not exist")
            continue
        signature = set(inspect.signature(cls.__init__).parameters)
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
