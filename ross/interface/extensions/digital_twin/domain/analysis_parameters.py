# -*- coding: utf-8 -*-
"""The parameters of the rossml balancing analysis, as fields of a form.

The screen shows one field per value, with the rossml default beside it, and
an empty field leaves rossml that default. `FIELDS` lists the values the
estimate uses and where each one goes in the arguments of `rossml.api.API.run`
(`number_points` and `w_f0` on their own, the filter factors inside `direct`
and `indirect`); nothing else reaches rossml.

Only the values rossml reads are here. `signal_parameters` and `kalman_flow`
are taken by `API.run` and then dropped by its `_clearmemory` without being
read, so a field of theirs would change nothing; they belong here again when
rossml reads them again.

The parameters of a machine are not read here. They belong to the machine,
not to the library, so they stay in the Digital Twin app with the rest of its
data; this screen starts from the rossml defaults for every rotor.

A factor of the filter is a mantissa times a power of ten. Its `direct` value
is used as given; its `indirect` bounds, a range for the mantissa and a range
for the exponent, are where the optimizer searches when there is no direct
value.
"""

import math

FACTORS = ("P", "Q", "S")

# key, group, path inside the arguments of `API.run`, whole number or not.
FIELDS = [
    ("number_points", "signal", ("number_points",), True),
    ("w_f0", "kalman", ("w_f0",), False),
]
FIELDS += [
    ("direct_%s" % factor, "direct", ("direct", "%s_factor" % factor), False)
    for factor in FACTORS + ("R",)
]
FIELDS += [
    (
        "indirect_%s_%s_%s" % (factor, part, end),
        "indirect",
        ("indirect", "%s_factor" % factor, row, column),
        True,
    )
    for factor in FACTORS
    for row, part in enumerate(("mantissa", "exponent"))
    for column, end in enumerate(("min", "max"))
]
KEYS = [field[0] for field in FIELDS]
WHOLE = {key: whole for key, _group, _path, whole in FIELDS}
WHOLE_PATH = {path: whole for _key, _group, path, whole in FIELDS}

# What rossml uses when the field is empty, so the screen can show it. The
# source is `rossml/api.py`: the defaults of `API.run` for `number_points` and
# `w_f0`, and `_default_indirect()` for the search bounds. `direct` has no
# default: without one the optimizer searches inside the indirect bounds.
# `test_the_defaults_shown_are_the_ones_rossml_uses` compares them.
DEFAULTS = {
    "number_points": 4096,
    "w_f0": 0.65,
    "direct_P": None,
    "direct_Q": None,
    "direct_S": None,
    "direct_R": None,
    "indirect_P_mantissa_min": 1,
    "indirect_P_mantissa_max": 9,
    "indirect_P_exponent_min": -9,
    "indirect_P_exponent_max": 9,
    "indirect_Q_mantissa_min": 1,
    "indirect_Q_mantissa_max": 9,
    "indirect_Q_exponent_min": -20,
    "indirect_Q_exponent_max": -8,
    "indirect_S_mantissa_min": 1,
    "indirect_S_mantissa_max": 9,
    "indirect_S_exponent_min": 2,
    "indirect_S_exponent_max": 9,
}

# Values that only make sense together: all of them or none.
PAIRS = [tuple("direct_%s" % factor for factor in FACTORS + ("R",))] + [
    tuple(key for key in KEYS if key.startswith("indirect_%s_" % factor))
    for factor in FACTORS
]


def defaults():
    """Return every field empty, which is the rossml default for each one.

    Returns
    -------
    dict
        `{key: None}` for every key of `KEYS`.
    """
    return dict.fromkeys(KEYS)


def chosen(values=None):
    """Return the fields with the values the user filled in.

    Parameters
    ----------
    values : dict, optional
        `{key: number, numeric text or None}` for the fields filled on the
        screen. An empty field is `None` and leaves rossml its default.

    Returns
    -------
    dict
        `{key: number or None}` for every key of `KEYS`.
    """
    fields = defaults()
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("The analysis parameters must be an object.")
    for key, value in values.items():
        if key not in fields:
            raise ValueError("Unknown analysis parameter: %s." % key)
        fields[key] = _number(key, value)
    for keys in PAIRS:
        given = [fields[key] is not None for key in keys]
        if any(given) and not all(given):
            raise ValueError(
                "Fill all of %s, or leave them all empty." % ", ".join(keys)
            )
    for factor in FACTORS:
        for part in ("mantissa", "exponent"):
            low = fields["indirect_%s_%s_min" % (factor, part)]
            high = fields["indirect_%s_%s_max" % (factor, part)]
            if low is not None and low > high:
                raise ValueError(
                    "The %s range of the %s factor must not start above its end."
                    % (part, factor)
                )
    return fields


def arguments(fields):
    """Turn the fields into the arguments rossml's `API.run` takes."""
    nested = {}
    for key, _group, path, _whole in FIELDS:
        if fields.get(key) is not None:
            _put(nested, path, fields[key])
    if "indirect" in nested:
        nested["indirect"]["R_factor"] = None
    return nested


def describe():
    """Return the fields as the screen draws them: key, group, default, whole."""
    return [
        {"key": key, "group": group, "default": DEFAULTS[key], "whole": whole}
        for key, group, _path, whole in FIELDS
    ]


def _number(key, value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    whole = WHOLE[key]
    refusal = ValueError("%s must be a number." % key)
    if isinstance(value, bool):
        raise refusal
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise refusal from None
    if not math.isfinite(number):
        raise refusal
    if whole and not number.is_integer():
        raise ValueError("%s must be a whole number." % key)
    return number


def _put(data, path, value):
    for step, following in zip(path, path[1:], strict=False):
        empty = [None, None] if isinstance(following, int) else {}
        if isinstance(step, int):
            if data[step] is None:
                data[step] = empty
        else:
            data.setdefault(step, empty)
        data = data[step]
    data[path[-1]] = int(value) if WHOLE_PATH[path] else value
