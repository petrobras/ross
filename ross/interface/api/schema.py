# -*- coding: utf-8 -*-
"""The form schema, derived from the installed ROSS."""

from flask import Blueprint, jsonify, request

from ross.interface.domain.analysis_catalog import catalog, titles
from ross.interface.domain.compatibility import table
from ross.interface.domain.conversion import to_unit
from ross.interface.domain.requests import UNITS_REQUEST
from ross.interface.domain.schema import build_schema

schema = Blueprint("schema", __name__)


@schema.route("/api/schema/elements")
def elements():
    """Describe every element form from the installed ROSS.

    The frontend builds the forms from this, instead of keeping a copy of the
    domain tables. `?lang=pt|en` picks the language of the labels.
    """
    return jsonify(build_schema(request.args.get("lang", "en")))


@schema.route("/api/schema/analyses")
def analyses():
    """Describe the fields of every analysis form.

    Unlike the element schema, this is not derived from ROSS, and that is by
    measurement: of the 147 fields, only 43% appear in a library signature. The
    rest is composition (`speed_min`/`max`/`steps` become a `linspace`), list
    editors, or presentation. What the tests cross-check is in
    `tests/test_analysis_catalog.py`.
    """
    language = request.args.get("lang", "en")
    # The compatibility table comes along, from the same place the `/run_analysis`
    # route refuses from: the screen warns before computing by reading exactly what
    # the server will enforce. Two copies would drift apart, and the one drifting
    # silently would be the screen's -- the user would see "allowed" and get
    # "not allowed".
    return jsonify(
        {
            "fields": catalog(language),
            "titles": titles(language),
            "unsupported": table(language),
        }
    )


# How many values one request may ask for: a list of a few hundred elements,
# with room to spare, and not an unbounded loop on the server.
MOST_VALUES = 5000


@schema.route("/api/units/convert", methods=["POST"])
def convert_units():
    """Read typed values as numbers in a unit, for the element list's filter.

    Next to the schema because it answers with the same knowledge the forms
    are built from -- the units -- and needs no rotor. Each answer is a number
    or None (`domain/conversion.to_unit` says when), in the order asked.
    """
    payload = UNITS_REQUEST.read(request.get_json(silent=True))
    items = payload["items"]
    if len(items) > MOST_VALUES:
        raise ValueError(
            "Too many values to convert at once: %d (at most %d)."
            % (len(items), MOST_VALUES)
        )
    values = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(
                "Each value to convert is an object with value, unit and to."
            )
        values.append(to_unit(item.get("value"), item.get("unit"), item.get("to")))
    return jsonify({"status": "success", "values": values})
