"""Single source of the UI-type -> ROSS-class mapping.

Before Phase 1 this map existed in three different, incompatible shapes: a
`type_map` inside build_bearing, another `type_map` inside build_seal, a
`class_map` in /load_ross_file (inverted) and a `ClassMap` in app.js (keyed
like 'BASIC_bearings'). Adding one type meant editing all four.
"""

# UI category -> { element type -> ROSS class name }
ELEMENTS = {
    "materials": {"BASIC": "Material"},
    "shafts": {"BASIC": "ShaftElement"},
    "disks": {"BASIC": "DiskElement"},
    "gears": {
        "BASIC": "GearElement",
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
}

# The class used when the given type does not exist in the category.
_FALLBACK = {category: types["BASIC"] for category, types in ELEMENTS.items()}

# The reverse direction, for importing native ROSS files.
_BY_ROSS_CLASS = {
    ross_class: (category, ui_type)
    for category, types in ELEMENTS.items()
    for ui_type, ross_class in types.items()
}


def categories():
    """Return the UI categories, in the order the sidebar shows them."""
    return list(ELEMENTS)


def ross_class_name(category, element_type=None):
    """Return the ROSS class name for a UI category and element type.

    Falls back to the category's BASIC class when the type is unknown, which
    preserves the previous behaviour of the inline type_map dictionaries.
    """
    types = ELEMENTS.get(category)
    if types is None:
        raise KeyError(f"unknown category: {category!r}")
    return types.get(element_type or "BASIC", _FALLBACK[category])


def ui_type_for_ross_class(ross_class):
    """Return (category, element_type) for a ROSS class name, or None."""
    return _BY_ROSS_CLASS.get(ross_class)
