# -*- coding: utf-8 -*-
"""The hybrid seal's two stages, from the fields of its form to ROSS's dicts.

`HybridSeal` takes each stage as a dictionary, `hole_pattern_parameters` and
`labyrinth_parameters`. The form used to ask for them as typed dictionaries,
which the users found hard to read and to get right, so it now asks for one
field per key, `hp_<key>` and `lab_<key>` (domain/field_catalog.py), each with
its unit. `folded` puts the dictionaries back together, in SI, for the builder
and for the exported script alike, so the two cannot disagree.

A dictionary typed the old way is still read, and a field given wins over the
same key there: a project saved before keeps working, and can be moved to the
fields one value at a time.
"""

import ast

from ross.units import Q_

from .field_catalog import HOLE_PATTERN, LABYRINTH
from .units import INT_PARAMETERS

STAGES = (("hp_", HOLE_PATTERN), ("lab_", LABYRINTH))

STAGE_UNREADABLE = "The %s of the hybrid seal is not a dictionary: %s"


def _stage_fields(row, prefix):
    return {
        key: value
        for key, value in row.items()
        if key.startswith(prefix)
        and not key.endswith("_unit")
        and str(value).strip() != ""
    }


def _typed_dictionary(row, parameter):
    raw = row.get(parameter)
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        value = ast.literal_eval(str(raw).strip())
    except (ValueError, SyntaxError) as error:
        raise ValueError(STAGE_UNREADABLE % (parameter, raw)) from error
    if not isinstance(value, dict):
        raise ValueError(STAGE_UNREADABLE % (parameter, raw))
    return value


def folded(row):
    """`row` with its `hp_`/`lab_` fields folded into the two dictionaries.

    The dictionaries come back as Python literals (text), the way the form
    stores a typed dictionary, so the builder and the exported script read
    them as they always have. A row with no stage field is returned as it was.
    """
    if not any(_stage_fields(row, prefix) for prefix, _ in STAGES):
        return row
    # Here and not at the top: rotor_builder imports this module.
    from .rotor_builder import extract_kwargs

    out = {
        key: value
        for key, value in row.items()
        if not key.startswith(tuple(prefix for prefix, _ in STAGES))
    }
    for prefix, parameter in STAGES:
        fields = _stage_fields(row, prefix)
        stage = _typed_dictionary(row, parameter)
        if fields:
            typed = dict(fields)
            typed.update(
                {
                    k: v
                    for k, v in row.items()
                    if k.startswith(prefix) and k.endswith("_unit")
                }
            )
            for name, value in extract_kwargs(typed, {}, "HybridSeal", []).items():
                key = name[len(prefix) :]
                if isinstance(value, Q_):
                    value = value.to_base_units().magnitude
                if hasattr(value, "tolist"):
                    value = value.tolist()
                if key in INT_PARAMETERS and isinstance(value, float):
                    value = int(value)
                stage[key] = value
        if stage:
            out[parameter] = repr(stage)
    return out
