# -*- coding: utf-8 -*-
"""Export of the Python script equivalent to the project on screen."""

import ast

from flask import Blueprint, jsonify, request

from ross.interface.domain.probe_refs import model_probes, references, resolved
from ross.interface.domain.python_export import build_script
from ross.interface.domain.rotor_builder import assemble_rotor
from ross.interface.domain.requests import EXPORT_REQUEST

export = Blueprint("export", __name__)


def with_model_probes(project, analyses):
    """The analyses with each probe that names one of the model's filled in.

    The script reads at what the chart read at: the model is built only when
    an analysis names one of its probes, as the run does
    (services/analysis/pipeline.py).
    """
    analyses = list(analyses or [])
    if not any(references(a.get("params")) for a in analyses if isinstance(a, dict)):
        return analyses
    probes = model_probes(assemble_rotor(project), project)
    return [
        dict(a, params=resolved(a.get("params") or {}, probes))
        if isinstance(a, dict) and references(a.get("params"))
        else a
        for a in analyses
    ]


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
        payload["project"],
        with_model_probes(payload["project"], payload["analyses"]),
        payload["conversion_type"],
    )

    try:
        ast.parse(script)
    except SyntaxError as error:
        raise RuntimeError(
            "The generated script is not valid Python (line %s): %s"
            % (error.lineno, error.msg)
        )

    return jsonify({"status": "success", "script": script})
