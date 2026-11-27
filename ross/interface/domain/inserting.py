"""Adding a shaft element on a face of another one: the sketch's extrusion
(frontend/features/sketch3d.js), drawn on the start or the end face of a shaft.

The new element takes the face's node and pushes the line past it along by
one node: every node after the face moves up by one, the way `splitting.py`
moves the nodes above a split (`_renumber`), link nodes included.

What decides which parts move is the face that was sketched on: what sits on
that face stays on it. Sketched on the **end** face of shaft `i` (node `e`),
the new element goes after `i`; a disk or a bearing at `e` stays at `e`, on the
end of `i`, and the next element of the line -- a shaft or a coupling starting
at `e` -- moves to `e + 1`, past the new one. Sketched on the **start** face of
shaft `i` (node `s`), the new element goes before `i`, and everything at `s`
moves to `s + 1` with `i`, whose face it is. In ROSS's terms, with node 0 at
z = 0: the elements past the face move along by the new element's length.

`tests/test_inserting.py` holds the result to the rotor ROSS builds from it.
"""

import copy

from .node_resolver import effective_nodes
from .splitting import NUMBERED_ALONE, _moved, _remap_explicit

SIDES = ("start", "end")

NO_SUCH_SHAFT = "Shaft #%d does not exist: the model has %d."
NO_SUCH_SIDE = "A shaft has a 'start' face and an 'end' face, not '%s'."
NOT_A_ROW = "The new shaft has to be a row of the shaft list."


def insert_shaft(project, index, side, row):
    """Return `project` with `row` added on the `side` face of shaft `index`.

    `row` is a row of the shaft list as the form writes it. It is given the
    face's node only when shaft `index` pins its own (`n`): a shaft with a
    blank `n` owes its node to its place in the list, and the insertion is
    what moves the list.
    """
    shafts = list(project.get("shafts", []) or [])
    if not isinstance(index, int) or index < 0 or index >= len(shafts):
        raise ValueError(NO_SUCH_SHAFT % (index + 1, len(shafts)))
    if side not in SIDES:
        raise ValueError(NO_SUCH_SIDE % side)
    if not isinstance(row, dict):
        raise ValueError(NOT_A_ROW)

    start = effective_nodes(shafts)[index]
    if side == "end":
        face = start + 1
        # The line past the face moves (a shaft or coupling starting on it);
        # what sits on the face stays.
        line_above, part_above = face - 1, face
        position, node = index + 1, face
    else:
        face = start
        line_above = part_above = face - 1
        position, node = index, face

    built = copy.deepcopy(project)
    _shift(built, line_above, part_above)
    new = copy.deepcopy(row)
    new.pop("n", None)
    if str(shafts[index].get("n", "")).strip() != "":
        new["n"] = str(node)
    rows = list(built.get("shafts", []) or [])
    rows.insert(position, new)
    built["shafts"] = rows
    return built


def _shift(project, line_above, part_above):
    """Move the nodes past the face up by one, in place: those above
    `line_above` for the elements of the shaft line (shafts, couplings), those
    above `part_above` for what sits on a node, and every link node."""
    for row in project.get("shafts", []) or []:
        _remap_explicit(row, "n", line_above)

    for category, numbering in NUMBERED_ALONE.items():
        above = line_above if category == "couplings" else part_above
        rows = project.get(category, []) or []
        # Resolved before anything is written, as `_renumber` does it: a blank
        # `n` owes its node to the rows around it.
        for row, before in zip(rows, numbering(rows), strict=True):
            after = _moved(before, above)
            if after != before:
                row["n"] = str(after)
            _remap_explicit(row, "n_link", part_above)
