# -*- coding: utf-8 -*-
"""Read a numeric form field already in the unit ROSS expects.

The interface sends everything as text, with the unit picked from a selector
beside the field. Here the text becomes a number (an expression is accepted
too, through services.expressions) and pint converts it to the target unit.
"""

from ross.units import Q_

from ross.interface.services.expressions import safe_math_eval


def get_converted_param(params, key, default_val, target_unit):
    val = params.get(key)
    if val is None or str(val).strip() == "":
        val_num = default_val
    else:
        try:
            val_num = float(val)
        except ValueError:
            val_num = safe_math_eval(str(val))

    unit = params.get(f"{key}_unit")
    if unit and target_unit:
        return float(Q_(val_num, unit).to(target_unit).m)
    return float(val_num)


def to_unit(value, unit, target):
    """What the element list's filter compares: one typed value as a number.

    `value` is what was typed -- `250`, `0.25`, `2*pi` -- and `unit` the one
    picked beside it; the answer is in `target`. None when there is no single
    number to compare: an empty field, a list (a bearing's coefficients per
    speed), a word, or units that do not convert into each other. The filter
    then leaves that element out, instead of guessing.
    """
    text = "" if value is None else str(value).strip()
    if not text or text[0] in "[{(":
        return None
    try:
        number = float(text)
    except ValueError:
        try:
            number = float(safe_math_eval(text))
        except Exception:
            return None
    if unit and target:
        try:
            return float(Q_(number, unit).to(target).m)
        except Exception:
            return None
    return number
