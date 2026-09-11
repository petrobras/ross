# -*- coding: utf-8 -*-
"""Build the form schema from the installed ROSS.

Three sources combine here, each owning what it knows:

  ROSS introspection -> which parameters exist, defaults, help text
  field_catalog.py   -> order, PT/EN label, basic or advanced, section
  units.py           -> input unit and alternatives

The frontend builds the forms from this. A new parameter on a ROSS class shows
up on the pending list by itself; a renamed parameter disappears from the schema
and the tests say so -- that is how BE-13 went unnoticed for ten days before
Phase 0.
"""

import inspect
import re

import ross as rs

from .element_registry import ross_class_name
from .field_catalog import FIELDS, SECTIONS
from .units import UNIT_ALTERNATIVES, UNITS_MAPPING, alternatives_for

LANGUAGES = ("en", "pt")

# The UI uses lowercase 'poisson'; ROSS expects 'Poisson'.
PARAMETER_ALIASES = {"poisson": "Poisson"}

# Parameters the interface handles itself; they must not become pending.
_NOT_FORM_FIELDS = {"self", "n"}

_DOC_SECTIONS = {
    "Returns",
    "Attributes",
    "Examples",
    "References",
    "Raises",
    "Notes",
    "See Also",
    "Yields",
    "Other Parameters",
}

_schema_cache = {}
_unit_map_cache = {}


def _documented_parameters(doc):
    """Parse the Parameters section of a numpydoc docstring."""
    lines = (doc or "").splitlines()
    try:
        start = next(
            i
            for i, line in enumerate(lines)
            if line.strip() == "Parameters"
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) == {"-"}
        )
    except StopIteration:
        return {}

    documented, current, buffer = {}, None, []
    for line in lines[start + 2 :]:
        stripped = line.strip()
        if stripped in _DOC_SECTIONS:
            break
        header = re.match(r"^(\w+) : (.*)$", line)
        if header and not line.startswith(" "):
            if current:
                documented[current] = " ".join(buffer).strip()
            current, buffer = header.group(1), []
        elif current and stripped:
            buffer.append(stripped)
    if current:
        documented[current] = " ".join(buffer).strip()
    return documented


def _first_sentence(text):
    if not text:
        return None
    sentence = re.split(r"(?<=[.])\s", text)[0].strip()
    if not sentence or sentence.lower().startswith("default is"):
        return None
    return sentence


def _jsonable(value):
    """ROSS defaults may be any object; the schema carries only what serialises."""
    if value is inspect.Parameter.empty or value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _class_signature(ross_class):
    """Return (class, parameters, docs), following **kwargs up the MRO.

    Classes like PlainJournal declare only their own parameters and pass the
    rest along with **kwargs -- tag, color, scale_factor and n_link live in
    BearingElement's signature. Without walking the MRO, those fields would
    show up as "does not exist in ROSS" when they do.
    """
    cls = getattr(rs, ross_class, None)
    if cls is None:
        return None, {}, {}

    parameters, documentation, declared_by_class = {}, {}, set()
    for klass in cls.__mro__:
        if klass is object:
            break

        initialiser = klass.__dict__.get("__init__")
        if initialiser is None:
            continue
        try:
            signature = inspect.signature(initialiser)
        except (TypeError, ValueError):
            continue

        forwards_kwargs = False
        declared_here = set()
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                forwards_kwargs = True
                continue
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            parameters.setdefault(name, parameter)  # the subclass wins
            declared_here.add(name)

        # 'declared by the class itself' is what matters for pending fields:
        # kxx, cxx and friends, inherited from BearingElement, do not belong on
        # a specialised bearing, which computes them instead of taking them.
        if not declared_by_class and declared_here:
            declared_by_class = declared_here

        # getdoc, not __doc__: the parser depends on the dedented docstring,
        # and __doc__ returns the raw text, with the class's indentation.
        for name, text in _documented_parameters(inspect.getdoc(klass)).items():
            documentation.setdefault(name, text)

        if not forwards_kwargs:
            break  # complete signature: nothing left to look for in the base

    return cls, parameters, documentation, declared_by_class


def _unit_for(cls, ross_class, parameter):
    """Unit for a parameter, following the same inheritance the parameter does.

    BallBearingElement declares no units of its own, but inherits cxx and cyy
    from BearingElement -- the unit has to come along, or the field loses its
    selector.
    """
    names = [k.__name__ for k in cls.__mro__] if cls is not None else [ross_class]
    for field_name in names:
        unit_name = UNITS_MAPPING.get(field_name, {}).get(parameter)
        if unit_name:
            return unit_name
    return None


def _build_field(field, cls, ross_class, parameters, documentation, language):
    target = PARAMETER_ALIASES.get(field["name"], field["name"])
    parameter = parameters.get(target)
    default = _jsonable(parameter.default) if parameter is not None else None

    control = field["control"]
    options = field["options"]
    if isinstance(default, bool):
        control, options = "boolean", ["false", "true"]

    unit = _unit_for(cls, ross_class, target)

    return {
        "name": field["name"],
        "label": field["label"][language],
        "group": field["group"],
        "section": (SECTIONS[field["section"]][language] if field["section"] else None),
        "control": control,
        "options": options,
        "optional": field["optional"],
        "placeholder": field["placeholder"],
        "is_dict": field["is_dict"],
        "unit": unit,
        "unit_options": alternatives_for(unit) if unit else None,
        "help": _first_sentence(documentation.get(target)),
        "ross_default": default,
        # False means the field no longer exists on the class: the interface
        # still shows it, but ROSS would refuse it. The tests fail on this.
        "known_to_ross": parameter is not None,
    }


def build_schema(language="en"):
    """Return the form schema for every element category, in one language."""
    if language not in LANGUAGES:
        language = "en"
    if language in _schema_cache:
        return _schema_cache[language]

    # unit_alternatives comes along because the analysis dashboards also
    # offer unit switching, outside the element schema.
    schema = {
        "language": language,
        "unit_alternatives": UNIT_ALTERNATIVES,
        "categories": {},
    }
    for category, subtypes in FIELDS.items():
        schema["categories"][category] = {}
        for subtype, fields in subtypes.items():
            ross_class = ross_class_name(category, subtype)
            cls, parameters, documentation, declared = _class_signature(ross_class)

            built = [
                _build_field(
                    field, cls, ross_class, parameters, documentation, language
                )
                for field in fields
            ]
            covered = {PARAMETER_ALIASES.get(f["name"], f["name"]) for f in fields}

            schema["categories"][category][subtype] = {
                "ross_class": ross_class,
                "exists": cls is not None,
                "fields": built,
                # Parameters ROSS accepts that the form does not offer yet.
                "not_in_form": sorted(
                    name
                    for name in declared
                    if name not in covered and name not in _NOT_FORM_FIELDS
                ),
            }

    _schema_cache[language] = schema
    return schema


def unit_map_by_class():
    """Return {ROSS class: {parameter: unit}} from the schema.

    This is the same derivation the frontend's unitMapFor does over the
    payload, and deliberately so: the exported script and the screen have to
    read the same unit. It comes from the schema, not from raw UNITS_MAPPING,
    because inherited units are already resolved here -- a BallBearingElement's
    cxx is declared on BearingElement.
    """
    if _unit_map_cache:
        return _unit_map_cache
    for subtypes in build_schema("en")["categories"].values():
        for definition in subtypes.values():
            mapping = _unit_map_cache.setdefault(definition["ross_class"], {})
            for spec in definition["fields"]:
                if spec["unit"]:
                    mapping[spec["name"]] = spec["unit"]
    return _unit_map_cache


def schema_problems(language="en"):
    """Return the fields the installed ROSS no longer accepts. Empty is good."""
    problems = []
    for category, subtypes in build_schema(language)["categories"].items():
        for subtype, definition in subtypes.items():
            if not definition["exists"]:
                problems.append(
                    f"{category}/{subtype}: rs.{definition['ross_class']} does not exist"
                )
                continue
            for field in definition["fields"]:
                if not field["known_to_ross"]:
                    problems.append(
                        f"{category}/{subtype}.{field['name']} "
                        f"does not exist in rs.{definition['ross_class']}"
                    )
    return problems
