# -*- coding: utf-8 -*-
"""The name a material has to have for ROSS to accept it.

`rs.Material` refuses any space in the name (`materials.py`: ``if " " in name:
raise ValueError("Spaces are not allowed in Material name")``). The form lets a
person type "Stainless Steel", and before this module the rotor then failed to
build, far from the field that caused it.

The screen now writes the name the way ROSS wants it when the material is saved
(`frontend/core/material_names.js`). This is the same rule on the server's side,
for the projects saved before that: a rotor stored in the browser or in a file
with "Stainless Steel" in it must still build, and its shafts, which name the
material the same way, must still find it. So the rule is applied on both ends
of the reference -- the material's name and the element's `material` -- and the
two meet.

Runs of whitespace become one underscore, and the ends are trimmed:
"Stainless  Steel " is "Stainless_Steel". The cases both sides are held to live
in `tests/golden/material_names.json`.
"""


def ross_material_name(name):
    """The name as ROSS accepts it."""
    return "_".join(str(name).split())


def material_key(name):
    """How a reference to a material is matched: ROSS's name, without case."""
    return ross_material_name(name).lower()


# --- a reference that finds nothing ------------------------------------------
#
# A shaft names its material by text. Until this guard, a name no material
# carried -- a material deleted, a rotor loaded from a file written by hand, a
# project edited outside the screen -- reached `extract_kwargs` and was answered
# with `list(mat_dict.values())[0]`: the **first material of the list**, or
# ROSS's own steel when the list was empty. Measured: a shaft asking for
# "Inox" was built out of Steel, the rotor came back whole, and nothing on
# screen said the analysis had been run on another machine than the one
# described.
#
# So the reference is checked before anything is built, and the refusal names
# the element and the name it asks for.

# What each category is called in the refusal. It is a sentence a person reads,
# so "point mass" and not "pointmasses"; `tests/test_material_names.py` checks
# that every category of the element registry is named here.
ELEMENT_WORDS = {
    "shafts": "shaft",
    "couplings": "coupling",
    "disks": "disk",
    "gears": "gear",
    "bearings": "bearing",
    "seals": "seal",
    "pointmasses": "point mass",
}

# The screen's own way of saying "no material of mine: use ROSS's steel".
DEFAULT_MATERIAL = "default (steel)"

# How many offenders the message lists before it starts counting them.
_SHOWN = 5


def _named_materials(project):
    return {
        material_key(material.get("name", ""))
        for material in (project.get("materials") or [])
    }


def missing_materials(project):
    """Every element whose material this project does not have.

    Each entry is `(category, position, tag, reference)`, with `position`
    counted the way the screen numbers the list, from 1.
    """
    available = _named_materials(project)
    found = []
    for category, word in ELEMENT_WORDS.items():
        for position, element in enumerate(project.get(category) or [], 1):
            if not isinstance(element, dict):
                continue
            reference = str(element.get("material", "")).strip()
            if not reference or reference.lower() == DEFAULT_MATERIAL:
                continue
            if material_key(reference) in available:
                continue
            found.append(
                (word, position, str(element.get("tag", "")).strip(), reference)
            )
    return found


def validate_materials(project):
    """Raise when an element names a material the project does not have."""
    found = missing_materials(project)
    if not found:
        return
    listed = ", ".join(
        "%s #%d%s uses '%s'"
        % (word, position, (" (%s)" % tag) if tag else "", reference)
        for word, position, tag, reference in found[:_SHOWN]
    )
    if len(found) > _SHOWN:
        listed += ", and %d more" % (len(found) - _SHOWN)
    names = [
        str(material.get("name", "")) for material in (project.get("materials") or [])
    ]
    available = ", ".join(name for name in names if name) or "none"
    raise ValueError(
        "This rotor has no material by that name: %s. Materials of this rotor: %s."
        % (listed, available)
    )
