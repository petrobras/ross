"""Splitting one shaft element in two, which is what ROSS calls `add_nodes`.

WHY THIS DOES NOT CALL `Rotor.add_nodes`, having been written from it.

`add_nodes` takes a rotor and gives back a rotor. The interface does not hold a
rotor -- it holds the project the screen edits, and that project carries things
a `Rotor` does not: `element_type`, the *name* of the material rather than the
object, and every advanced field of the form. Rebuilding the shaft list out of
a returned rotor would drop all of it, so the dictionaries have to be edited
whatever else happens. Once they are being edited, calling `add_nodes` buys
nothing and costs three things:

  * it wipes the tag of **every** shaft, not only the two it makes -- measured,
    a rotor whose four elements were tagged A, B, C came back as
    `Shaft Element 0..3`, so its names would have to be put back from the
    project we already hold;
  * it needs a rotor, and `build_rotor_from_ui` refuses a model with no bearing
    or with a gap in the numbering. Splitting an element is a *modelling*
    action, done while the model is still half built; refusing it then is
    refusing it exactly when it is wanted;
  * it leaves `n_l`/`n_r` stale on disks (measured: a disk moved to `n=3` still
    reported `n_l=n_r=2`), which is the same upstream defect `concatenate` has.

What ROSS does own here are two rules, and the probe measured both to be total,
with no hidden freedom:

  1. the four diameters are interpolated **linearly** along the element -- a
     tapered hollow element 0.01/0.10 -> 0.02/0.20 split at a quarter of its
     length gave exactly 0.0125/0.125;
  2. every node above the split moves up by one, link nodes included -- a disk
     at 2 became 3, a bearing at 3 became 4 with its `n_link` 4 -> 5, and the
     point mass at 4 became 5.

So the rules are restated here, and `tests/test_splitting.py` compares the
result of this module, element by element and diameter by diameter, against
what `Rotor.add_nodes` produces from the same project. The oracle stays ROSS;
only the runtime dependency is gone.
"""

import copy

from ross.units import Q_

from .node_resolver import NUMBER_SYNTAX, effective_nodes, listed_nodes
from .units import UNITS_MAPPING

# Categories whose `n` is resolved without the shaft list, each by the rule the
# builder uses for it: `listed_nodes` for couplings, `effective_nodes` for the
# rest. Splitting a shaft does not move these lists, so an element with a
# blank `n` above the split has to be moved -- and pinned -- by hand. Shafts are
# not here: their numbering follows their own list, so inserting the right
# half renumbers them by itself (see `_shaft_rows`).
NUMBERED_ALONE = {
    "couplings": listed_nodes,
    "disks": effective_nodes,
    "gears": effective_nodes,
    "bearings": effective_nodes,
    "seals": effective_nodes,
    "pointmasses": effective_nodes,
}

NO_SUCH_SHAFT = "Shaft #%d does not exist: the model has %d."
LENGTH_UNREADABLE = "Shaft #%d has no readable length, so it cannot be split."
OFFSET_UNREADABLE = "'%s' is not a distance."
OFFSET_OUTSIDE = (
    "The split has to fall inside shaft #%d, which is %s %s long: %s %s would "
    "land on node %d, which already exists."
)
UNIT_UNREADABLE = "Shaft #%d gives '%s' as the unit of %s, which is not a length."

# What a length field means when the row names no unit: the form's default,
# which is also what `rotor_builder.extract_kwargs` falls back to.
SHAFT_UNITS = UNITS_MAPPING["ShaftElement"]


def _number(value):
    """A float back to the string form the screen stores.

    `%.10g` and not `repr`: the project is JSON typed by hand, and
    `0.30000000000000004` in a length field is noise the user did not write.
    Ten significant digits is past any dimension a rotor has and short of the
    place where binary floating point starts showing.
    """
    return "%.10g" % float(value)


def _read(row, key, fallback=None):
    """A form field as a float, or `fallback` when it is blank or unreadable.

    The fallback is how ROSS itself reads a shaft: `idr` defaults to `idl` and
    `odr` to `odl`, so an element typed with two diameters is a cylinder and
    not an element with a zero right face.
    """
    raw = str(row.get(key, "")).strip()
    # `NUMBER_SYNTAX` and not a bare `float()`: `float('1_0')` is ten and
    # `float('inf')` is a length no rotor has. The rule is the one the node
    # fields already use, imported rather than written again.
    if not NUMBER_SYNTAX.match(raw):
        return fallback
    return float(raw)


def _offset(raw):
    value = _read({"x": raw}, "x")
    if value is None:
        raise ValueError(OFFSET_UNREADABLE % raw)
    return value


def _unique_tag(wanted, taken):
    """`wanted`, or `wanted (2)`, `(3)` ... until it is nobody else's."""
    if wanted not in taken:
        return wanted
    number = 2
    while "%s (%d)" % (wanted, number) in taken:
        number += 1
    return "%s (%d)" % (wanted, number)


def _unit(row, key, fallback=None):
    """The unit a length field is typed in: its own `*_unit`, else `fallback`.

    `fallback` is how a blank `idr` borrows `idl`: the value it stands for is
    `idl`'s, so its unit has to be `idl`'s too. A falsy `fallback` is the
    form's default for `key`.
    """
    unit = str(row.get(key + "_unit", "") or "").strip()
    if unit:
        return unit
    return fallback or SHAFT_UNITS[key]


def _in_metres(value, unit, index, key):
    try:
        return float(Q_(value, unit).to("m").m)
    except Exception as error:
        raise ValueError(UNIT_UNREADABLE % (index + 1, unit, key)) from error


def _shaft_rows(shafts, index, offset, length, split_node):
    """The shaft list with element `index` replaced by its two halves.

    Called **after** `_renumber`, and the order is not a detail: the two halves
    are copies of one row, so a renumbering that ran afterwards would either
    move both or neither. Here the left half keeps the node the original had
    and the right half is given the node the split created, once.

    Every field carries its own unit (`odl_unit`, `L_unit`, ...), so the
    interpolation runs in metres and each result is written back in the unit
    of the field it lands in. Interpolating the raw numbers would mix them:
    `odl = 0.05 m` and `odr = 100 mm` would give a 50 m diameter.
    """
    original = shafts[index]
    fraction = offset / length

    idl_value = _read(original, "idl", 0.0)
    odl_value = _read(original, "odl", 0.0)
    idr_value = _read(original, "idr")
    odr_value = _read(original, "odr")

    units = {"idl": _unit(original, "idl"), "odl": _unit(original, "odl")}
    units["idr"] = _unit(original, "idr", idr_value is None and units["idl"])
    units["odr"] = _unit(original, "odr", odr_value is None and units["odl"])

    metres = {
        "idl": _in_metres(idl_value, units["idl"], index, "idl"),
        "odl": _in_metres(odl_value, units["odl"], index, "odl"),
    }
    # A blank right diameter is the left one, so it is taken in metres
    # already rather than read again with a unit it was never typed in.
    metres["idr"] = (
        metres["idl"]
        if idr_value is None
        else _in_metres(idr_value, units["idr"], index, "idr")
    )
    metres["odr"] = (
        metres["odl"]
        if odr_value is None
        else _in_metres(odr_value, units["odr"], index, "odr")
    )
    mid_id = metres["idl"] + fraction * (metres["idr"] - metres["idl"])
    mid_od = metres["odl"] + fraction * (metres["odr"] - metres["odl"])

    def written(row, key, value_in_metres):
        row[key] = _number(Q_(value_in_metres, "m").to(units[key]).m)
        # Pinned only when it is not the default, so a row typed in the
        # form's own unit comes back as it went in.
        if units[key] != SHAFT_UNITS[key] or key + "_unit" in row:
            row[key + "_unit"] = units[key]

    left = copy.deepcopy(original)
    right = copy.deepcopy(original)

    # All four are written on both halves even where the original left them
    # blank. A blank `odr` means "the same as `odl`", which stops being true the
    # moment the element is cut anywhere along a taper -- and writing the value
    # the user would have had to work out is the point of the button.
    # `L` and `offset` are both in the unit of `L`, which is left as it was.
    for row, (a_id, a_od, b_id, b_od, span) in (
        (left, (metres["idl"], metres["odl"], mid_id, mid_od, offset)),
        (right, (mid_id, mid_od, metres["idr"], metres["odr"], length - offset)),
    ):
        row["L"] = _number(span)
        written(row, "idl", a_id)
        written(row, "odl", a_od)
        written(row, "idr", b_id)
        written(row, "odr", b_od)

    # The name goes on both halves, because both halves are that element --
    # the second one is told apart by a suffix rather than renamed, so the
    # model still reads as the thing the user drew.
    tag = str(original.get("tag", "")).strip()
    if tag:
        taken = {str(s.get("tag", "")).strip() for s in shafts}
        right["tag"] = _unique_tag(tag, taken)

    # Only when the original pinned its node. A shaft with a blank `n` owes it
    # to its place in the list, and the insertion is what moves the list.
    if str(original.get("n", "")).strip() != "":
        right["n"] = str(split_node + 1)

    return shafts[:index] + [left, right] + shafts[index + 1 :]


def split_shaft(project, index, offset):
    """Return `project` with shaft `index` cut `offset` from its left face.

    `offset` is in the unit the shaft's length is typed in (`L_unit`, else the
    form's `mm` from `domain/units.py`), because that is the unit the prompt
    names and the one its default -- half of `L` -- is written in.
    """
    shafts = list(project.get("shafts", []) or [])
    if not isinstance(index, int) or index < 0 or index >= len(shafts):
        raise ValueError(NO_SUCH_SHAFT % (index + 1, len(shafts)))

    length = _read(shafts[index], "L")
    if length is None or length <= 0:
        raise ValueError(LENGTH_UNREADABLE % (index + 1))

    distance = _offset(offset)
    # The one refusal the per-element button cannot make impossible. A split at
    # 0 or at L is a node that already exists, and ROSS answers that request by
    # doing nothing at all -- while still wiping every tag. Refusing by name is
    # the difference between "that node is already there" and a button that
    # sometimes does nothing.
    split_node = effective_nodes(shafts)[index]
    if distance <= 0 or distance >= length:
        landed = split_node if distance <= 0 else split_node + 1
        raise ValueError(
            OFFSET_OUTSIDE
            % (
                index + 1,
                _number(length),
                _unit(shafts[index], "L"),
                _number(distance),
                _unit(shafts[index], "L"),
                landed,
            )
        )

    built = copy.deepcopy(project)
    _renumber(built, split_node)
    built["shafts"] = _shaft_rows(
        list(built.get("shafts", []) or []), index, distance, length, split_node
    )
    return built


def _moved(node, split_node):
    return node + 1 if node > split_node else node


def _renumber(project, split_node):
    """Move every node above the split up by one, in place."""
    for row in project.get("shafts", []) or []:
        _remap_explicit(row, "n", split_node)

    for category, numbering in NUMBERED_ALONE.items():
        rows = project.get(category, []) or []
        # Resolved before anything is written: an element with a blank `n` owes
        # its node to the ones around it, so the whole category has to be read
        # before the category starts changing.
        for row, before in zip(rows, numbering(rows), strict=True):
            after = _moved(before, split_node)
            if after != before:
                # It was implicit and it has to move: from here on it is
                # pinned. Leaving it blank would keep the *number* the
                # interface picked and lose the place the user meant.
                row["n"] = str(after)
            _remap_explicit(row, "n_link", split_node)


def _remap_explicit(row, key, split_node):
    raw = str(row.get(key, "")).strip()
    if raw == "":
        return
    node = _read(row, key)
    if node is None:
        return
    row[key] = str(_moved(int(node), split_node))
