# -*- coding: utf-8 -*-
"""Read a numeric form field already in the unit ROSS expects.

The interface sends everything as text, with the unit picked from a selector
beside the field. Here the text becomes a number (an expression is accepted
too, through services.expressions) and pint converts it to the target unit.
"""

from ross.units import Q_

from services.expressions import safe_math_eval


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
