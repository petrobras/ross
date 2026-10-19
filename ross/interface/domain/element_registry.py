"""Single source of the UI-type -> ROSS-class mapping.

Before Phase 1 this map existed in three different, incompatible shapes: a
`type_map` inside build_bearing, another `type_map` inside build_seal, a
`class_map` in /load_ross_file (inverted) and a `ClassMap` in app.js (keyed
like 'BASIC_bearings'). Adding one type meant editing all four.

Each entry names what ROSS is called with to build the element: a class
(`"GearElement"`) or one of its alternative constructors
(`"DiskElement.from_geometry"`, a classmethod that takes the disk's dimensions
and works out the mass and inertias; `"GearElement.from_geometry"`, the same for
a gear). The element either way is an instance
of the class; only the form and the call differ.
"""

import ross as rs

# Fields the screen keeps on an element for its own drawing, which ROSS never
# sees: the shape the 3D view draws the element with, from the geometry bank
# (frontend/core/shapes3d.js). Only a picture, by Leonardo's decision -- ROSS
# computes with the form's own values -- so the builder leaves these out of
# the constructor's arguments, the element cache leaves them out of its key,
# and the exported script leaves them out of the code.
VIEW_FIELDS = frozenset({"shape3d"})

# UI category -> { element type -> ROSS constructor: "Class" or "Class.method" }
ELEMENTS = {
    "materials": {"BASIC": "Material"},
    "shafts": {"BASIC": "ShaftElement"},
    "disks": {
        "BASIC": "DiskElement",
        "Geometry": "DiskElement.from_geometry",
    },
    "gears": {
        "BASIC": "GearElement",
        "Geometry": "GearElement.from_geometry",
        "TVMS": "GearElementTVMS",
    },
    "couplings": {"BASIC": "CouplingElement"},
    "bearings": {
        "BASIC": "BearingElement",
        "BallBearing": "BallBearingElement",
        "RollerBearing": "RollerBearingElement",
        "MagneticBearing": "MagneticBearingElement",
        "Cylindrical": "CylindricalBearing",
        "PlainJournal": "PlainJournal",
        "SqueezeFilm": "SqueezeFilmDamper",
        "ThrustPad": "ThrustPad",
        "TiltingPad": "TiltingPad",
    },
    "seals": {
        "BASIC": "SealElement",
        "HolePattern": "HolePatternSeal",
        "Labyrinth": "LabyrinthSeal",
        "Hybrid": "HybridSeal",
    },
    "pointmasses": {"BASIC": "PointMass"},
    # A probe is not part of the rotor: ROSS's `Rotor` takes no probes, and
    # its response plots take them as an argument (`probe=[rs.Probe(...)]`).
    # It is a category of the screen so a probe is placed on the model -- and
    # renumbered with it -- and drawn in the 3D view; the builder checks it
    # with `rs.Probe` and hands it to nobody (domain/rotor_builder.py).
    "probes": {"BASIC": "Probe"},
}

# The class used when the given type does not exist in the category.
_FALLBACK = {category: types["BASIC"] for category, types in ELEMENTS.items()}

# The reverse direction, for importing native ROSS files. A file names the
# class of each element, never the call that built it -- a disk made from its
# dimensions is saved as a plain DiskElement with m, Ip and Id -- so only the
# classes themselves come back.
_BY_ROSS_CLASS = {
    ross_class: (category, ui_type)
    for category, types in ELEMENTS.items()
    for ui_type, ross_class in types.items()
    if "." not in ross_class
}


def categories():
    """Return the UI categories, in the order the sidebar shows them."""
    return list(ELEMENTS)


def ross_constructor(category, element_type=None):
    """Return what ROSS is called with: "Class" or "Class.method".

    Falls back to the category's BASIC class when the type is unknown, which
    preserves the previous behaviour of the inline type_map dictionaries.
    """
    types = ELEMENTS.get(category)
    if types is None:
        raise KeyError(f"unknown category: {category!r}")
    return types.get(element_type or "BASIC", _FALLBACK[category])


def ross_class_name(category, element_type=None):
    """Return the ROSS class of the element a UI type builds."""
    return ross_constructor(category, element_type).split(".")[0]


def resolve_constructor(constructor):
    """The callable a constructor name stands for in the installed ROSS.

    None when ROSS has no such class or method, which is what the registry
    tests look for after a ROSS update.
    """
    target = rs
    for part in constructor.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target


def takes_units(constructor):
    """Whether ROSS converts pint quantities for this call.

    ROSS's `@check_units` turns a `Q_(70, 'mm')` into 0.07 before the call
    runs. Most constructors have it; `GearElement.from_geometry` does not -- it
    computes rho * w * d^2 with whatever it is given, and a quantity comes out
    in m^3 where the gear expects kg. A call without it has to get plain
    numbers in SI. The decorator is recognised by what `functools.wraps` leaves
    behind, `__wrapped__`.
    """
    target = resolve_constructor(constructor)
    if isinstance(target, type):
        target = target.__init__
    return hasattr(target, "__wrapped__")


def ui_type_for_ross_class(ross_class):
    """Return (category, element_type) for a ROSS class name, or None."""
    return _BY_ROSS_CLASS.get(ross_class)
