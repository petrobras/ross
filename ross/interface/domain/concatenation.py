# -*- coding: utf-8 -*-
"""Join two rotors end to end, keeping the project editable.

ROSS has `Rotor.concatenate`, and it does exactly what the screen needs: the
elements of the second rotor move to the right, the node where the first ends
becomes the node where the second begins, and the link nodes of **both** are
renumbered to sit after every structural node.

## Why this does not round-trip through a file

The obvious shape was to concatenate and read the result back through
`domain/ross_import.project_from_ross_file`: derive instead of copy, reusing a
reader that already exists. It is lossy where it matters most. ROSS's own
docstring says so, in `bearing_seal_element.save_coefficient_table`:

    Solver-based bearings (``FluidFilmBearing`` and its configuration classes,
    ``ThrustPad``, ``SqueezeFilmDamper``) use this as their ``save()``: the file
    holds the solved dynamic-coefficient table (...) Re-create the object from
    its constructor to change the bearing model.

A PlainJournal the user built from clearance, viscosity and geometry would come
back as a plain `BearingElement` carrying a table of kxx. The rotor would run
and the form would have nothing left to show. Faithful in the dynamics,
destructive in the model.

## Why it does not reimplement the offsets either

The second shape was to apply ROSS's rule to the project ourselves: offset the
nodes of the second rotor by `max(nodes)` of the first, and renumber the link
nodes. That is a copy of someone else's rule -- the pattern this whole project
exists to avoid -- and measuring the rule showed the copy would be worse than
usual. Link-node order is `list(df["n_link"].dropna().unique())`: the order in
which `n_link` values first appear in ROSS's internal dataframe. It is an
implementation detail with no docstring, and half a machine's numbering would
depend on it.

## What it does instead

It **asks**. The two rotors are built, `Rotor.concatenate` joins them, and the
result is used for one thing only: to read off where each element ended up.
Those numbers are written into a copy of the project; every other field the user
typed is carried across untouched, because it never left the project.

The correspondence between a project element and its element in the joined rotor
is positional -- `concatenate` extends the lists in order, and the builder feeds
them in a known order -- and positional correspondence is the kind of assumption
that breaks quietly. So it is checked rather than trusted: ROSS tags each
element it copies with `(R0)` or `(R1)`, and `_rotor_index` reads that tag back.
A mismatch raises here instead of producing a rotor whose bearings moved.
"""

import copy
import re

import ross as rs

from .material_names import material_key, ross_material_name
from .node_resolver import effective_nodes, listed_nodes
from .rotor_builder import build_rotor_from_ui

# The project categories, in the order the screen lists them. Unlike the ROSS
# element lists, this order is ours and does not change under us.
CATEGORIES = (
    "shafts",
    "couplings",
    "disks",
    "gears",
    "bearings",
    "seals",
    "pointmasses",
)

# How the builder resolves a blank `n`, by category: couplings by their place in
# the list (`rotor_builder.py`), everything else by `effective_nodes`.
NUMBERING = {"couplings": listed_nodes}

# Which ROSS list each kind of element ends up in. Used only to sweep every
# element of a built rotor, never to line them up with project elements.
ROSS_LISTS = (
    "shaft_elements",
    "disk_elements",
    "bearing_elements",
    "point_mass_elements",
)

# The suffix ROSS appends to every tag it copies: `"{tag} (R{i})"`.
ROTOR_SUFFIX = re.compile(r"\s\(R(\d+)\)$")

A_MULTIROTOR = (
    "A MultiRotor cannot be concatenated. It is two shafts coupled through a "
    "gear mesh, each turning at its own speed; concatenating makes one shaft "
    "line turning at one speed. Concatenate the rotors it was built from."
)

MIXED_MODELS = (
    "These two rotors were analysed under different rotor models: %s and %s. "
    "Concatenating them would put both halves under a single model. Recompute "
    "one of them under the other's model, or clear its analyses first."
)

MIXED_ON_ONE = (
    "The analyses of %s were computed under more than one rotor model (%s). "
    "There is no single model to carry into the concatenation. Recompute them "
    "under one model, or clear them first."
)

UNMAPPED_NODE = (
    "Node %s of %s has no counterpart in the joined rotor. The map is learned "
    "from the tags ROSS stamps on the elements it copies, and no tagged element "
    "of that rotor sits on this node."
)

CONTRADICTORY_NODE = (
    "Reading the joined rotor back, node %s of %s is said to become both %s and "
    "%s. The new numbering is learned from the tags ROSS stamps on the elements "
    "it copies, and two elements of that rotor sharing a tag on different nodes "
    "give no single answer. Rename one of them in the Tag field."
)

MODEL_NAMES = {"": "6 DoF", "4dof": "4 DoF", "torsional": "Torsional"}


def model_of(conversions):
    """The rotor model its analyses agree on.

    Three answers, and the difference between the first two is the whole point:

    * `None`  -- the rotor has no analyses, so it has no model yet and
      concatenates with anything;
    * `""`, `"4dof"`, `"torsional"` -- every analysis used this one;
    * `"mixed"` -- they disagree.

    The frontend's `unanimousConversion` answers the same question for the
    export, and folds the first two together (`[]` and `[""]` both give `""`).
    That is harmless when the question is "can these analyses share one script";
    it is not harmless here, because a rotor nobody has analysed yet would stop
    being concatenable with a 4 DoF one, and nothing on the screen would say
    why."""
    if not conversions:
        return None
    distinct = {c or "" for c in conversions}
    if len(distinct) > 1:
        return "mixed"
    return distinct.pop()


def refuse_mismatched_models(first_conversions, second_conversions, names=("A", "B")):
    """Raise unless the two rotors agree on a model, or have none."""
    for conversions, name in (
        (first_conversions, names[0]),
        (second_conversions, names[1]),
    ):
        if model_of(conversions) == "mixed":
            used = ", ".join(
                sorted({MODEL_NAMES.get(c or "", c or "") for c in conversions})
            )
            raise ValueError(MIXED_ON_ONE % (name, used))

    first = model_of(first_conversions)
    second = model_of(second_conversions)
    if first is None or second is None:
        return
    if first != second:
        raise ValueError(
            MIXED_MODELS
            % (MODEL_NAMES.get(first, first), MODEL_NAMES.get(second, second))
        )


def _rotor_index(element):
    """Which rotor ROSS says this element came from, read from its tag."""
    found = ROTOR_SUFFIX.search(str(getattr(element, "tag", "") or ""))
    return int(found.group(1)) if found else None


def _elements_of(project, category):
    return list(project.get(category) or [])


# The attributes that name a node on any element, and the two more that only a
# shaft element may be read for.
#
# `n_l` and `n_r` are the left and right nodes of a shaft, and they matter here
# because a shaft's `n` is only its left one: the last node of a rotor is the
# right node of its last shaft and the `n` of nothing at all. Without them the
# map comes out one node short.
#
# But they are read for shafts **only**, and that is not tidiness. Measured on a
# joined rotor, a disk that moved from node 1 to node 3 comes out as
#
#     tag='disk_0 (R1)'   n=3   n_l=1   n_r=1
#
# `Rotor.concatenate` updates `n` and `n_link` and nothing else, so on anything
# that is not a shaft those two are left over from the rotor the element used to
# belong to. Reading them would have taught the map that node 1 goes to 3 and to
# 1 at the same time.
NODE_ATTRIBUTES = ("n", "n_link")
SHAFT_ONLY_ATTRIBUTES = ("n_l", "n_r")


def _all_elements(rotor):
    """Every element of a built rotor, with the name of the list it came from."""
    for name in ROSS_LISTS:
        for element in getattr(rotor, name, []) or []:
            yield name, element


def _nodes_of(element, ross_list):
    """The node each readable attribute names, `None` where there is none."""
    attributes = NODE_ATTRIBUTES
    if ross_list == "shaft_elements":
        attributes = attributes + SHAFT_ONLY_ATTRIBUTES
    read = []
    for attribute in attributes:
        value = getattr(element, attribute, None)
        read.append(None if value is None else int(value))
    return tuple(read)


def _node_map(joined, source, rotor_index, name):
    """`old node -> new node`, learned from the tags ROSS stamped itself.

    The first version of this lined the project's element lists up with the
    joined rotor's, position by position. That is wrong, and `Rotor.__init__`
    says why on line 379:

        self.shaft_elements   = sorted(shaft_elements,   key=lambda el: el.n)
        self.bearing_elements = sorted(bearing_elements, key=lambda el: el.n)

    A built rotor holds its elements in **node order**, not in the order they
    were passed. After a concatenation the bearings of the two halves interleave,
    so the second element of the joined list is the first bearing of the *second*
    rotor. The positional version produced a rotor whose bearings had moved, and
    the tag check written beside it is what caught that on the first run.

    So nothing is matched by position. `concatenate` renames every element it
    copies to `"{tag} (R{i})"`, which is ROSS's own record of where the element
    came from; matching a ROSS element to a ROSS element by that tag is reading
    what ROSS wrote, not guessing. What comes out is a map between node numbers,
    and node numbers are all the project needs.
    """
    after = {}
    for ross_list, element in _all_elements(joined):
        if _rotor_index(element) != rotor_index:
            continue
        key = ROTOR_SUFFIX.sub("", str(element.tag or ""))
        after[key] = _nodes_of(element, ross_list)

    mapping = {}
    for ross_list, element in _all_elements(source):
        tag = str(getattr(element, "tag", "") or "")
        if tag not in after:
            continue
        pairs = zip(_nodes_of(element, ross_list), after[tag], strict=True)
        for before_node, after_node in pairs:
            if before_node is None or after_node is None:
                continue
            if mapping.get(before_node, after_node) != after_node:
                raise ValueError(
                    CONTRADICTORY_NODE
                    % (name, before_node, mapping[before_node], after_node)
                )
            mapping[before_node] = after_node

    # Every node the rotor occupies has to be in the map. A node left out would
    # otherwise reach the merged project unchanged -- an element that quietly
    # stayed behind while the rest of its rotor moved.
    for node in list(source.nodes) + list(source.link_nodes):
        if int(node) not in mapping:
            raise ValueError(UNMAPPED_NODE % (int(node), name))
    return mapping


def _merged_materials(first, second):
    """One material list, and the map from the second rotor's names to it.

    Same name and same properties is one material. Same name and different
    properties are two, and the second is renamed with ROSS's own suffix -- the
    alternative is silence, because `rotor_builder` answers an unknown material
    name with `list(mat_dict.values())[0]`, the first of the list. A collision
    left alone does not raise: it builds half a machine out of the wrong metal.
    """
    merged = [copy.deepcopy(m) for m in _elements_of(first, "materials")]
    # Matched the way the builder matches them (domain/material_names.py):
    # "Stainless Steel" and "Stainless_Steel" are one name to ROSS.
    by_name = {material_key(m.get("name", "")): m for m in merged}
    renamed = {}

    for material in _elements_of(second, "materials"):
        material = copy.deepcopy(material)
        name = ross_material_name(material.get("name", ""))
        key = material_key(name)
        existing = by_name.get(key)
        if existing is not None and _same_material(existing, material):
            continue
        if existing is not None:
            # `_R1`, and not the ` (R1)` ROSS puts on tags: a material name has
            # a stricter grammar than a tag. `materials.py:75` is explicit --
            # `if " " in name: raise ValueError("Spaces are not allowed in
            # Material name")` -- so the convention that reads best on a tag is
            # simply illegal here, and borrowing it would raise on the build.
            new_name = "%s_R1" % name
            suffix = 1
            while material_key(new_name) in by_name:
                suffix += 1
                new_name = "%s_R1_%d" % (name, suffix)
            material["name"] = new_name
            renamed[key] = new_name
        merged.append(material)
        by_name[material_key(material.get("name", ""))] = material
    return merged, renamed


def _same_material(one, other):
    """Equal on every field either of them declares, `name` aside."""
    keys = (set(one) | set(other)) - {"name"}
    return all(
        str(one.get(k, "")).strip() == str(other.get(k, "")).strip() for k in keys
    )


def concatenated_project(first, second, first_conversions=(), second_conversions=()):
    """The project of the two rotors joined end to end.

    `first_conversions` and `second_conversions` are the rotor models the saved
    analyses of each were computed under -- empty when the rotor has none.
    """
    if first.get("isMultiRotor") or second.get("isMultiRotor"):
        raise ValueError(A_MULTIROTOR)

    refuse_mismatched_models(first_conversions, second_conversions)

    built = {
        0: build_rotor_from_ui(first),
        1: build_rotor_from_ui(second),
    }
    joined = rs.Rotor.concatenate(built[0], built[1])
    maps = {
        index: _node_map(joined, rotor, index, name)
        for index, (rotor, name) in enumerate(
            ((built[0], "the first rotor"), (built[1], "the second rotor"))
        )
    }
    materials, renamed = _merged_materials(first, second)

    merged = {"materials": materials}
    for category in CATEGORIES:
        rows = []
        for rotor_index, project in ((0, first), (1, second)):
            elements = _elements_of(project, category)
            # The project numbers its nodes the way the builder does, and
            # `node_resolver` is the one place those rules live. Reading them
            # here rather than trusting the `n` field is what makes an element
            # with a blank `n` move with the rest of its rotor.
            here = NUMBERING.get(category, effective_nodes)(elements)
            for element, before in zip(elements, here, strict=True):
                row = copy.deepcopy(element)
                row["n"] = str(maps[rotor_index][before])
                link = str(row.get("n_link", "")).strip()
                if link:
                    row["n_link"] = str(maps[rotor_index][int(float(link))])
                tag = str(row.get("tag", "")).strip()
                if tag:
                    row["tag"] = "%s (R%d)" % (tag, rotor_index)
                if rotor_index == 1:
                    material = material_key(row.get("material", ""))
                    if material in renamed:
                        row["material"] = renamed[material]
                rows.append(row)
        merged[category] = rows
    return merged


def structural_nodes(project):
    """The structural nodes the project occupies, without building the rotor.

    Used by the screen to say where the joint will fall, before the user
    commits to anything. It is not what decides the concatenation -- that is
    read back from ROSS -- so a disagreement between the two is a bug in this
    function and not a wrong rotor.
    """
    lefts = effective_nodes(_elements_of(project, "shafts")) + listed_nodes(
        _elements_of(project, "couplings")
    )
    if not lefts:
        return 0
    return max(lefts) + 1
