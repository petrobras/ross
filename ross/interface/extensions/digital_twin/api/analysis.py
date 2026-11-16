# -*- coding: utf-8 -*-
"""Digital Twin analysis routes. The work lives in services.analysis."""

from flask import Blueprint, jsonify, request

from ross.interface.extensions.digital_twin.services.analysis import (
    campbell_mode_shape,
    run,
)

analysis = Blueprint("dt_analysis", __name__)


def _body():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Send the request as JSON.")
    return payload


@analysis.route("/api/dt/analysis", methods=["POST"])
def run_analysis():
    """Run one analysis card and return its figures and tables."""
    body = _body()
    return jsonify(run(body.get("session"), body.get("type"), body.get("params") or {}))


@analysis.route("/api/dt/campbell/mode-shape", methods=["POST"])
def mode_shape():
    """Return the 3D mode of the point clicked on the Campbell diagram."""
    body = _body()
    return jsonify(campbell_mode_shape(body.get("session"), body.get("point")))
