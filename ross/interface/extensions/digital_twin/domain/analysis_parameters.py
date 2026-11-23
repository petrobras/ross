# -*- coding: utf-8 -*-
"""The parameters of the rossml balancing analysis, as fields of a form.

rossml takes them as nested dictionaries (`signal_parameters`, `kalman_flow`,
`resultant_force_limits`, `direct`, `indirect`). The screen shows each value
as a field, empty until the user fills it, and an empty field leaves rossml
its own default. `FIELDS` lists the values the estimate uses; nothing else
reaches rossml.

The parameters of a machine are not read here. They belong to the machine,
not to the library, so they stay in the Digital Twin app with the rest of its
data; this screen starts from the rossml defaults for every rotor.

A factor of the filter is a mantissa times a power of ten. Its `direct` value
is used as given; its `indirect` bounds, a range for the mantissa and a range
for the exponent, are where the optimizer searches when there is no direct
value.
"""

import copy
import math

FACTORS = ("P", "Q", "S")

# key, group, path inside rossml's dictionaries, whole number or not.
FIELDS = [
    ("number_points", "signal", ("signal_parameters", "number_points"), True),
    ("time_end", "signal", ("signal_parameters", "time_end"), False),
    ("noise_rate", "signal", ("signal_parameters", "noise_rate"), False),
    ("periods_number", "signal", ("signal_parameters", "periods_number"), True),
    (
        "decomposition_threshold",
        "signal",
        ("signal_parameters", "decomposition_threshold"),
        False,
    ),
    ("df", "signal", ("signal_parameters", "df"), False),
    ("w_f0", "kalman", ("kalman_flow", "w_f0"), False),
    ("df_multiple", "kalman", ("kalman_flow", "df_multiple"), True),
    ("mu_max_iter", "kalman", ("kalman_flow", "mu_max_iter"), True),
    ("mu_pop_size", "kalman", ("kalman_flow", "mu_pop_size"), True),
    ("mu_d_min", "limits", ("resultant_force_limits", "mu_d", 0), False),
    ("mu_d_max", "limits", ("resultant_force_limits", "mu_d", 1), False),
    ("phi_min", "limits", ("resultant_force_limits", "phi", 0), False),
    ("phi_max", "limits", ("resultant_force_limits", "phi", 1), False),
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

# Values that only make sense together: all of them or none.
PAIRS = [
    ("mu_d_min", "mu_d_max"),
    ("phi_min", "phi_max"),
    tuple("direct_%s" % factor for factor in FACTORS + ("R",)),
] + [
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
    for low, high in (("mu_d_min", "mu_d_max"), ("phi_min", "phi_max")):
        if fields[low] is not None and fields[low] > fields[high]:
            raise ValueError("%s must not be larger than %s." % (low, high))
    return fields


def arguments(fields):
    """Turn the fields into the dictionaries rossml's `API.run` takes."""
    nested = {}
    for key, _group, path, _whole in FIELDS:
        if fields.get(key) is not None:
            _put(nested, path, fields[key])
    if "resultant_force_limits" in nested:
        limits = copy.deepcopy(nested["resultant_force_limits"])
        nested.setdefault("kalman_flow", {})["resultant_force_limits"] = limits
    if "indirect" in nested:
        nested["indirect"]["R_factor"] = None
    return nested


def describe():
    """Return the fields as the screen draws them: key, group, whole number."""
    return [
        {"key": key, "group": group, "whole": whole}
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
