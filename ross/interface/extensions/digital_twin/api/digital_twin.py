# -*- coding: utf-8 -*-
"""Digital Twin routes. The work lives in services.digital_twin."""

from flask import Blueprint, jsonify, request

from ross.interface.extensions.digital_twin.domain import equipment, session
from ross.interface.extensions.digital_twin.services.digital_twin import (
    create_model,
    load_process_data,
)

digital_twin = Blueprint("dt_digital_twin", __name__)


def _body():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Send the request as JSON.")
    return payload


@digital_twin.route("/api/dt/equipment")
def equipment_list():
    """List platforms and tags, saying which models run here."""
    return jsonify({"platforms": equipment.catalog()})


@digital_twin.route("/api/dt/session", methods=["POST"])
def new_session():
    """Open a session directory."""
    current = session.create()
    return jsonify({"id": current.id, "name": current.name})


@digital_twin.route("/api/dt/process-data", methods=["POST"])
def process_data():
    """Load the process data for the chosen tag and time."""
    body = _body()
    return jsonify(
        load_process_data(
            body.get("session"),
            body.get("equipment"),
            body.get("pi_time"),
            body.get("rotation"),
        )
    )


@digital_twin.route("/api/dt/model", methods=["POST"])
def model():
    """Create the rotor model and return its figure."""
    return jsonify(create_model(_body().get("session")))
