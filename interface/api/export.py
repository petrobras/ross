# -*- coding: utf-8 -*-
"""Export of the Python script equivalent to the project on screen."""

import ast

from flask import Blueprint, jsonify, request

from domain.python_export import build_script
from domain.requests import EXPORT_REQUEST

export = Blueprint("export", __name__)


@export.route("/api/export/python", methods=["POST"])
def python_script():
    """Return the Python script equivalent to the project on screen.

    The script used to be assembled in the browser, from the DOM, with its own
    copies of node numbering, the unit map and the ROSS class names. Here it is
    born from the same functions that build the real rotor, so the exported file
    and the on-screen chart cannot disagree. `ast.parse` is the safety net: an
    invalid script comes back as an error instead of becoming a broken download.
    """
    payload = EXPORT_REQUEST.read(request.get_json(silent=True))
    script = build_script(
        payload["project"], payload["analyses"], payload["conversion_type"]
    )

    try:
        ast.parse(script)
    except SyntaxError as error:
        raise RuntimeError(
            "The generated script is not valid Python (line %s): %s"
            % (error.lineno, error.msg)
        )

    return jsonify({"status": "success", "script": script})
