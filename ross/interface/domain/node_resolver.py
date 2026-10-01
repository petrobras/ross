"""Single source of node numbering.

This logic existed twice, in two languages: get_eff_nodes in app.py and
getEffectiveNodes in frontend/app.js. The frontend one decided the nodes of the
exported Python script and the backend one decided the nodes of the chart -- any
drift between them would produce a script that does not reproduce the screen.
"""

import re

# Attributes that may carry a node index on a ROSS element. n_link belongs on
# the list: a bearing linked to node 7 makes node 7 exist.
NODE_ATTRIBUTES = ("n_l", "n_r", "n", "n_link")

_MAX_LISTED_MISSING = 10


# The syntax accepted in a numeric field. The regex is explicit on purpose:
# Python's float() accepts '1_0', 'inf' and 'nan', and JavaScript's Number()
# accepts '0x10' -- each side turned different garbage into a node, and the two
# have to agree (the frontend labels the list by the same rule, in
# getEffectiveNodes).
#
# Public, and named for numbers rather than for nodes, because it is not a rule
# about nodes: it is the rule for what a number typed into this interface may
# look like. `domain/splitting.py` reads lengths and diameters with it, and a
# second copy there would be a second answer to 'is `1_0` a distance'.
NUMBER_SYNTAX = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _explicit_node(element):
    """Return the node the user pinned on this element, or None."""
    raw = str(element.get("n", "")).strip()
    if not NUMBER_SYNTAX.match(raw):
        return None
    parsed = float(raw)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return int(parsed)


def effective_nodes(elements):
    """Resolve the node of each element in a category.

    Elements with an explicit `n` keep it. The others take the lowest node not
    already pinned by an explicit one, in list order.
    """
    elements = list(elements)
    pinned = {n for n in (_explicit_node(e) for e in elements) if n is not None}

    resolved = []
    next_auto = 0
    for element in elements:
        explicit = _explicit_node(element)
        if explicit is not None:
            resolved.append(explicit)
            continue
        while next_auto in pinned:
            next_auto += 1
        resolved.append(next_auto)
        pinned.add(next_auto)
        next_auto += 1
    return resolved


def listed_nodes(elements):
    """Resolve the node of each coupling: its explicit `n`, else its position.

    The builder numbers couplings this way, not by `effective_nodes`, and
    whatever reads a coupling's node back from the project has to use the
    same rule -- two rules disagree as soon as a blank `n` follows a pinned
    one: `[{"n": "2"}, {}]` is `[2, 1]` here and `[2, 0]` there.
    """
    resolved = []
    for position, element in enumerate(elements):
        explicit = _explicit_node(element)
        resolved.append(position if explicit is None else explicit)
    return resolved


def collect_nodes(ross_elements):
    """Return every node index occupied by the built ROSS elements."""
    nodes = set()
    for element in ross_elements:
        for attribute in NODE_ATTRIBUTES:
            value = getattr(element, attribute, None)
            if value is None:
                continue
            try:
                nodes.add(int(value))
            except (TypeError, ValueError):
                continue
    return nodes


def validate_node_topology(ross_elements):
    """Raise ValueError when the node numbering has gaps.

    ROSS requires nodes numbered contiguously from 0. The previous version of
    this check tested `hasattr(elm, 'nodes')` first, but no ROSS element class
    exposes `nodes` -- that branch never ran, and the branch that did run never
    looked at `n_link`, so a bearing linked to an empty node left an invisible
    gap. Returns the node set on success.
    """
    nodes = collect_nodes(ross_elements)
    if not nodes:
        return nodes

    largest = max(nodes)
    missing = sorted(set(range(largest + 1)) - nodes)
    if missing:
        listed = ", ".join(str(n) for n in missing[:_MAX_LISTED_MISSING])
        if len(missing) > _MAX_LISTED_MISSING:
            listed += ", ..."
        raise ValueError(
            f"Invalid node topology. The largest assigned node is {largest}, "
            f"but node(s) {listed} do not exist. ROSS requires uninterrupted "
            f"numbering starting at 0 (ex: 0, 1, 2, 3)."
        )
    return nodes
