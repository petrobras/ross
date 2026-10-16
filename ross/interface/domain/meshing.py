"""Cutting the shaft line into elements short enough for the model to be right.

THE RULE. API RP 684 (Tutorial on the API Standard Paragraphs Covering Rotor
Dynamics and Balancing, 1st ed., 1996), section 1.5.2.1, "Division of Rotor
into Discrete Sections": the rotor is first divided at every step of its
outside or inside diameter, and then

    a. the length to diameter ratio of any section should not exceed 1.0
       (0.5 is preferred) -- enough resolution for the first three critical
       speeds;
    b. the length to diameter ratio of any section should not be less than
       0.10 -- very different lengths side by side can make the numbers go
       wrong.

The first division is the user's: each shaft row is a section they drew. This
module does the second: every element longer than `max_ld` diameters is cut
into as few equal parts as keep each of them at or under it -- so no node the
model has is lost, and every disk, bearing and seal stays where it was. An
element shorter than `min_ld` diameters cannot be fixed by cutting, and is
reported.

ROSS does the same to a built rotor, `Rotor.refine_mesh`, written with this
module, and `tests/test_meshing.py` holds the two to the same answer. This one
does not call it, for the reasons `domain/splitting.py` gives for not calling
`add_nodes`: the project carries what a `Rotor` does not (the material by
name, the model, the tags, every field of the form), and meshing is modelling
-- done on a model that may not build yet.

The diameter is the element's largest outer one, so a taper is cut by its
thick end. Elements on the same pair of nodes -- a sleeve over a shaft -- are
cut together, by the largest of them. A span holding a coupling is left whole:
a coupling is one element in ROSS, and cannot be cut.
"""

import copy
import math
import warnings

from ross.units import Q_

from .conversion import to_unit
from .node_resolver import effective_nodes, listed_nodes
from .splitting import _number, _renumber, _unique_tag
from .units import UNITS_MAPPING

SHAFT_UNITS = UNITS_MAPPING["ShaftElement"]
DIAMETERS = ("idl", "odl", "idr", "odr")

BAD_RATIO = "The largest L/D has to be a positive number, not '%s'."
BAD_MINIMUM = "The smallest L/D has to be between 0 and the largest (%s), not '%s'."
BAD_PARTS = "The number of parts has to be a whole number of 1 or more, not '%s'."
BAD_MODES = "The number of frequencies has to be a whole number of 1 or more, not '%s'."
BAD_TOLERANCE = "The tolerance has to be a positive percentage, not '%s'."
BAD_SPEED = "The speed has to be a number of rpm, not '%s'."
BAD_METHOD = "There is no way of discretizing called '%s'."
BAD_INDEX = "Shaft rows are asked for by their position in the list, not by %r."

# An element of exactly `max_ld` diameters stays whole, whatever the last bit
# of the floating point division says (0.07 / (0.7 * 0.1) is 1.0000000000000002).
TOLERANCE = 1e-9


def _unit(row, key):
    return str(row.get(key + "_unit", "") or "").strip() or SHAFT_UNITS[key]


def _metres(row, key, fallback=None):
    """A field of the row in metres, whatever unit it was typed in."""
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return fallback
    value = to_unit(raw, _unit(row, key), "m")
    return fallback if value is None else value


def _written(row, key, metres):
    """`metres` in the unit the row's field is typed in, as the form stores it."""
    return _number(Q_(metres, "m").to(_unit(row, key)).m)


def _geometry(row):
    """L and the four diameters of a row in metres, blanks read as ROSS reads
    them: `idl` 0, `idr` as `idl`, `odr` as `odl`. None when L or `odl` is not
    a number."""
    length = _metres(row, "L")
    odl = _metres(row, "odl")
    if length is None or odl is None or length <= 0 or odl <= 0:
        return None
    idl = _metres(row, "idl", 0.0)
    return {
        "L": length,
        "idl": idl,
        "odl": odl,
        "idr": _metres(row, "idr", idl),
        "odr": _metres(row, "odr", odl),
    }


def _ratio(raw, message, *values):
    try:
        value = float(str(raw).strip())
    except ValueError:
        raise ValueError(message % (values + (raw,))) from None
    if not math.isfinite(value):
        raise ValueError(message % (values + (raw,)))
    return value


def _coupling_spans(project):
    """The node each coupling starts at, by the builder's rule for them
    (`node_resolver.listed_nodes`: its `n`, or its place in the list)."""
    return set(listed_nodes(project.get("couplings", []) or []))


def _count(raw, message):
    """A whole number of one or more, from what was typed."""
    text = str(raw).strip()
    try:
        value = float(text)
    except ValueError:
        raise ValueError(message % raw) from None
    if not math.isfinite(value) or value < 1 or value != int(value):
        raise ValueError(message % raw)
    return int(value)


def plan(project, max_ld=0.5, min_ld=0.1, indexes=None, parts=None):
    """How the shafts would be cut, without cutting them.

    By the ratio (`max_ld`), or -- when `parts` is given -- every span asked
    for in that many equal parts, whatever its ratio (ROSS's
    `refine_mesh(parts=...)`, the `nel_r` of `Rotor.from_section`).

    One entry per span (a pair of nodes) of the shaft line, in node order:
    `node`, the `rows` on it (positions in the shaft list), `ratio` (L/D),
    `parts`, and `why` when it is left as it is -- "coupling", "unreadable",
    "layers" (elements on the same nodes with different lengths) or "outside"
    (not among `indexes`). `indexes`, when given, are the shaft rows asked
    for; a span is cut when any of its rows is.
    """
    fixed = None if parts is None else _count(parts, BAD_PARTS)
    largest = math.inf if fixed else _ratio(max_ld, BAD_RATIO)
    if largest <= 0:
        raise ValueError(BAD_RATIO % max_ld)
    smallest = _ratio(min_ld, BAD_MINIMUM, _number(largest))
    if smallest < 0 or smallest > largest:
        raise ValueError(BAD_MINIMUM % (_number(largest), min_ld))

    shafts = list(project.get("shafts", []) or [])
    nodes = effective_nodes(shafts)
    couplings = _coupling_spans(project)
    asked = None
    if indexes is not None:
        for index in indexes:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(BAD_INDEX % (index,))
        asked = set(indexes)

    spans = {}
    for index, node in enumerate(nodes):
        spans.setdefault(node, []).append(index)

    steps = []
    for node in sorted(spans):
        rows = spans[node]
        step = {"node": node, "rows": rows, "ratio": None, "parts": 1, "why": ""}
        steps.append(step)
        shapes = [_geometry(shafts[i]) for i in rows]
        if any(shape is None for shape in shapes):
            step["why"] = "unreadable"
            continue
        length = shapes[0]["L"]
        diameter = max(max(shape["odl"], shape["odr"]) for shape in shapes)
        step["ratio"] = length / diameter
        if any(abs(shape["L"] - length) > TOLERANCE * length for shape in shapes):
            step["why"] = "layers"
            continue
        if node in couplings:
            step["why"] = "coupling"
            continue
        if asked is not None and not asked.intersection(rows):
            step["why"] = "outside"
            continue
        step["parts"] = fixed or max(
            1, math.ceil(length / (largest * diameter) - TOLERANCE)
        )
    return steps, smallest


def _pieces(row, parts, node, taken):
    """`row` cut in `parts` equal pieces, the diameters taken linearly along it
    -- the rule `add_nodes` follows (see `domain/splitting.py`)."""
    shape = _geometry(row)
    pieces = []
    tag = str(row.get("tag", "")).strip()
    pinned = str(row.get("n", "")).strip() != ""
    for k in range(parts):
        a, b = k / parts, (k + 1) / parts
        piece = copy.deepcopy(row)
        piece["L"] = _written(row, "L", shape["L"] / parts)
        for key, (start, end), where in (
            ("idl", ("idl", "idr"), a),
            ("odl", ("odl", "odr"), a),
            ("idr", ("idl", "idr"), b),
            ("odr", ("odl", "odr"), b),
        ):
            value = shape[start] + where * (shape[end] - shape[start])
            piece[key] = _written(row, key, value)
        # Every piece is that element: the first keeps its name, the others
        # are told apart by a number, as a split names its right half.
        if tag and k > 0:
            piece["tag"] = _unique_tag(tag, taken)
            taken.add(piece["tag"])
        if pinned:
            piece["n"] = str(node + k)
        pieces.append(piece)
    return pieces


def mesh_shafts(project, max_ld=0.5, min_ld=0.1, indexes=None, parts=None):
    """Return `project` with its shafts cut by `plan`, and the plan.

    The spans are cut from the last to the first: cutting one moves only the
    nodes above it, so the ones still to be cut keep their numbers.
    """
    steps, smallest = plan(project, max_ld, min_ld, indexes, parts)
    built = copy.deepcopy(project)
    originals = list(built.get("shafts", []) or [])
    taken = {str(s.get("tag", "")).strip() for s in originals}
    for step in reversed(steps):
        parts = step["parts"]
        if parts < 2:
            continue
        node = step["node"]
        _renumber(built, node, parts - 1)
        # By identity and not by position: a span cut before this one may
        # have been further up the list, and moved these rows down it.
        targets = [originals[i] for i in step["rows"]]
        rebuilt = []
        for row in built.get("shafts", []) or []:
            if any(row is target for target in targets):
                rebuilt.extend(_pieces(row, parts, node, taken))
            else:
                rebuilt.append(row)
        built["shafts"] = rebuilt

    report = {
        "cut": [
            {"rows": s["rows"], "parts": s["parts"], "ratio": s["ratio"]}
            for s in steps
            if s["parts"] > 1
        ],
        "short": [
            {"rows": s["rows"], "ratio": s["ratio"]}
            for s in steps
            if s["ratio"] is not None and s["ratio"] < smallest
        ],
        "kept": [
            {"rows": s["rows"], "why": s["why"]}
            for s in steps
            if s["why"] in ("coupling", "unreadable", "layers")
        ],
        "before": len(project.get("shafts", []) or []),
        "after": len(built.get("shafts", []) or []),
    }
    return built, report


# --- by the convergence of the natural frequencies --------------------------------
#
# `Rotor.refine_mesh_by_convergence` does the work, on the rotor built from the
# project: the rotor as it is and then refined to L/D 2, 1, 0.5, 0.25 and
# 0.125, each compared with the next, until refining changes none of the
# lowest `n_modes` natural frequencies by more than `rtol`. What comes back is
# the ratio of the mesh chosen, and the project is cut to it here, by
# `mesh_shafts` -- which `tests/test_meshing.py` holds to `refine_mesh`, so the
# project is the rotor ROSS chose.
#
# It is the whole rotor or nothing: a convergence of the frequencies is a
# property of the whole model, and the rows ticked in the list do not enter.


def mesh_by_convergence(project, n_modes="6", rtol="0.1", speed="0", min_ld=0.1):
    """Return `project` cut to the coarsest mesh whose frequencies converged.

    `rtol` is a percentage and `speed` is in rpm, as the dialog asks for them.
    The report is `mesh_shafts`'s, with `convergence`: one row per mesh
    analysed (`max_ld`, None for the rotor as it is; `elements`, the shaft
    elements; `change`, the largest relative change of a frequency when that
    mesh is refined, in %; `wn`, the frequencies in Hz) and `chosen`.
    """
    # Here and not at the top: the rest of this module needs no ROSS rotor,
    # and the builder brings all of ROSS with it.
    from .rotor_builder import build_rotor_from_ui

    modes = _count(n_modes, BAD_MODES)
    tolerance = _ratio(rtol, BAD_TOLERANCE)
    if tolerance <= 0:
        raise ValueError(BAD_TOLERANCE % rtol)
    rpm = _ratio(speed, BAD_SPEED)

    rotor = build_rotor_from_ui(project)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, results = rotor.refine_mesh_by_convergence(
            n_modes=modes,
            rtol=tolerance / 100,
            speed=Q_(rpm, "rpm").to("rad/s").m,
        )

    chosen = results.chosen
    picked = results.max_ld[-1 if chosen is None else chosen]
    if picked is None:
        # Nothing cut, and still what is too short to fix.
        built, report = mesh_shafts(project, min_ld=min_ld, parts=1)
    else:
        built, report = mesh_shafts(project, max_ld=picked, min_ld=min_ld)

    couplings = len(project.get("couplings", []) or [])
    report["convergence"] = {
        "rows": [
            {
                "max_ld": ld,
                "elements": int(count) - couplings,
                "change": None if math.isnan(change) else float(change),
                "wn": [float(w) / (2 * math.pi) for w in wn],
            }
            for ld, count, change, wn in zip(
                results.max_ld,
                results.el_num,
                results.error_arr,
                results.wn,
                strict=True,
            )
        ],
        "chosen": chosen,
        "n_modes": modes,
        "rtol": tolerance,
    }
    return built, report
