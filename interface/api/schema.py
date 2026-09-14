# -*- coding: utf-8 -*-
"""The form schema, derived from the installed ROSS."""

from flask import Blueprint, jsonify, request

from domain.analysis_catalog import catalog, titles
from domain.compatibility import table
from domain.schema import build_schema

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
