# -*- coding: utf-8 -*-
"""Read element dictionaries written before ROSS 3 renamed the table axis.

ROSS 3.0 (petrobras/ross#1371) made every coefficient table declare the physical
axis it is tabulated on: ``speed=`` for the rotor speed and ``frequency=`` for
the excitation (whirl) frequency. Fluid-film bearings and seals produce speed
tables, so their ``frequency`` keyword became ``speed`` -- and the forms here
followed the signature.

A project saved by the interface before that (the browser's stored state, a
rotor exported from the Hub as JSON, a native ROSS file written by version 2)
still carries ``frequency`` on those elements. ROSS itself would refuse it
(``PlainJournal``) or read it as the *other* axis (the seals), so the key is
translated here, in one place, before it reaches a constructor, the exported
script or the screen.

Which elements are translated is not a list written by hand. A class is
translated when its form offers ``speed`` and a bare ``frequency`` cannot be
what the current form means -- either because the form has no ``frequency``
field at all (the fluid-film bearings) or because ROSS requires ``speed``, so
``frequency`` alone was necessarily the old spelling (the seals, whose form
offers ``frequency`` as the optional second axis of a 2-D table). A class whose
form keeps both and needs neither (``BearingElement``, ``SealElement``)
genuinely accepts both axes, and a ``frequency`` there is left alone.
"""

import inspect

from .element_registry import ELEMENTS, ross_class_name
from .field_catalog import FIELDS
from .schema import _class_signature

LEGACY_AXIS = "frequency"
CURRENT_AXIS = "speed"


def _offers(fields, name):
    return any(field["name"] == name for field in fields)


def _requires_speed(ross_class):
    parameters = _class_signature(ross_class)[1]
    parameter = parameters.get(CURRENT_AXIS)
    return parameter is not None and parameter.default is inspect.Parameter.empty


def _forms():
    for category, subtypes in FIELDS.items():
        if category not in ELEMENTS:
            continue
        for subtype, fields in subtypes.items():
            yield ross_class_name(category, subtype), fields


# ROSS classes whose form tabulates on `speed` only: a `frequency` there is
# always the old spelling, and a stale one beside a filled `speed` is dropped.
SPEED_ONLY_CLASSES = frozenset(
    ross_class
    for ross_class, fields in _forms()
    if _offers(fields, CURRENT_AXIS) and not _offers(fields, LEGACY_AXIS)
)

# ROSS classes where a `frequency` with no `speed` is read as the speed table.
LEGACY_SPEED_CLASSES = SPEED_ONLY_CLASSES | frozenset(
    ross_class
    for ross_class, fields in _forms()
    if _offers(fields, CURRENT_AXIS) and _requires_speed(ross_class)
)


def migrate_element(element, ross_class):
    """Return the element with a version-2 ``frequency`` table read as ``speed``.

    The dictionary comes back unchanged (the same object) when there is nothing
    to translate. When ``frequency`` is present and ``speed`` is not, the key
    is renamed in place -- keeping its position, so that whatever is generated
    from the dictionary keeps its order -- together with its ``_unit`` suffix.

    When both are present, the form decides. On a class whose form has no
    ``frequency`` field the stale key is dropped: ROSS would read it as a
    second, excitation-frequency axis the user never asked for. On a class
    whose form offers ``frequency`` the pair is a 2-D table and is kept.
    """
    if ross_class not in LEGACY_SPEED_CLASSES or LEGACY_AXIS not in element:
        return element

    has_current = str(element.get(CURRENT_AXIS) or "").strip() != ""
    if has_current and ross_class not in SPEED_ONLY_CLASSES:
        return element

    migrated = {}
    for key, value in element.items():
        if key == LEGACY_AXIS or key == LEGACY_AXIS + "_unit":
            if has_current:
                continue
            key = key.replace(LEGACY_AXIS, CURRENT_AXIS, 1)
        migrated[key] = value
    return migrated
