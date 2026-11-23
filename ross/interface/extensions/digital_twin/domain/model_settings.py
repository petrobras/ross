# -*- coding: utf-8 -*-
"""The additional settings of a rotor model, read from its source.

A model declares its adjustable quantities as module-level constants,
`RADIAL_BEARING_CLEARANCE = Q_(0.16 / 2, "mm")`, and `create_model` reads the
user's values from `configured_data={"RADIAL_BEARING_CLEARANCE": [value,
unit]}`. These are the model's additional settings (step 2.1).

The defaults are read with `ast`, without importing the model: only
`Q_(<arithmetic on numbers>, "<unit>")` is evaluated, never arbitrary code.
"""

import ast
import operator

from ross.units import Q_

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _number(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_number(node.left), _number(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_number(node.operand))
    raise ValueError("not a number")


def _quantity(node):
    """Return `(value, unit)` for `Q_(number, "unit")`, or None."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Q_"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return None
    try:
        return _number(node.args[0]), node.args[1].value
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def defaults(source):
    """Return the model's settings as `[{"name", "value", "unit"}, ...]`."""
    settings = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        found = _quantity(node.value)
        if found is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                settings.append(
                    {"name": target.id, "value": found[0], "unit": found[1]}
                )
    return settings


def configured_data(declared, chosen):
    """Return the `configured_data` argument of `create_model`.

    Parameters
    ----------
    declared : list of dict
        The settings the model declares, from `defaults`.
    chosen : dict
        `{name: {"value": number, "unit": str}}` sent by the screen. A setting
        left out keeps its default.

    Returns
    -------
    dict
        `{name: [value, unit]}`, the format the models read.
    """
    if not isinstance(chosen, dict):
        raise ValueError("The model settings must be an object.")
    names = {item["name"] for item in declared}
    unknown = sorted(set(chosen) - names)
    if unknown:
        raise ValueError("Unknown model settings: %s" % ", ".join(unknown))

    result = {}
    for item in declared:
        entry = chosen.get(item["name"]) or {}
        if not isinstance(entry, dict):
            raise ValueError("Setting %s must be an object." % item["name"])
        value = entry.get("value", item["value"])
        unit = entry.get("unit", item["unit"])
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("%s must be a number." % item["name"]) from None
        if not isinstance(unit, str):
            raise ValueError("The unit of %s must be text." % item["name"])
        try:
            Q_(value, unit).to(item["unit"])
        except Exception:
            raise ValueError(
                "%s: %r is not a unit of the same kind as %r."
                % (item["name"], unit, item["unit"])
            ) from None
        result[item["name"]] = [value, unit]
    return result
