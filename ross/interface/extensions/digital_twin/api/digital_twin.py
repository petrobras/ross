# -*- coding: utf-8 -*-
"""Digital Twin routes. The work lives in services.digital_twin."""

from flask import Blueprint, jsonify, request

from ross.interface.extensions.digital_twin.domain import (
    equipment,
    rossml_balancing,
    session,
)
from ross.interface.extensions.digital_twin.services.digital_twin import (
    create_model,
    create_user_model,
    load_process_data,
    model_settings_for,
    rotor_project,
)
from ross.interface.extensions.digital_twin.services.balancing_analysis import (
    figure_for,
    run_balancing,
    set_parameters,
    settings_for,
)
from ross.interface.extensions.digital_twin.services.measurement import (
    apply_sensor,
    set_planes,
)

digital_twin = Blueprint("dt_digital_twin", __name__)


def _body():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Send the request as JSON.")
    return payload


@digital_twin.route("/api/dt/features")
def features():
    """Tell which steps can run here: the balancing needs rossml installed."""
    return jsonify({"balancing": rossml_balancing.available()})


@digital_twin.route("/api/dt/equipment")
def equipment_list():
    """List platforms and tags, saying which models run here."""
    return jsonify({"platforms": equipment.catalog()})


@digital_twin.route("/api/dt/model-settings")
def model_settings():
    """Return the additional settings of the chosen equipment's model."""
    return jsonify(model_settings_for(request.args.get("equipment")))


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
            body.get("source", "pi"),
        )
    )


@digital_twin.route("/api/dt/model", methods=["POST"])
def model():
    """Create the rotor model and return its figure."""
    body = _body()
    return jsonify(create_model(body.get("session"), body.get("settings")))


@digital_twin.route("/api/dt/user-model", methods=["POST"])
def user_model():
    """Use a rotor of the user's Rotor Hub as the session's model."""
    body = _body()
    return jsonify(
        create_user_model(
            body.get("session"),
            body.get("project"),
            body.get("name"),
            body.get("rotation", "clockwise"),
        )
    )


@digital_twin.route("/api/dt/project", methods=["POST"])
def project():
    """Return the session's rotor as a Rotor Hub project."""
    return jsonify(rotor_project(_body().get("session")))


@digital_twin.route("/api/dt/planes", methods=["POST"])
def planes():
    """Replace the measuring and balancing planes of the session."""
    body = _body()
    return jsonify(
        set_planes(
            body.get("session"), body.get("measuring", []), body.get("balancing", [])
        )
    )


@digital_twin.route("/api/dt/balancing-settings")
def balancing_settings():
    """Tell whether the balancing analysis can run, and return its last result."""
    return jsonify(settings_for(request.args.get("session")))


@digital_twin.route("/api/dt/balancing", methods=["POST"])
def balancing():
    """Run the rossml balancing estimate on the session."""
    body = _body()
    return jsonify(
        run_balancing(
            body.get("session"),
            body.get("iterations", 0),
            body.get("population", 1),
            body.get("optimize", False),
            values=body.get("parameters"),
        )
    )


@digital_twin.route("/api/dt/balancing-parameters", methods=["POST"])
def balancing_parameters():
    """Choose the analysis parameters: the fields filled on the screen."""
    body = _body()
    return jsonify(set_parameters(body.get("session"), body.get("parameters")))


@digital_twin.route("/api/dt/balancing-figure", methods=["POST"])
def balancing_figure():
    """Draw one chart of the last balancing run: polar, forces or virtual sensing."""
    body = _body()
    return jsonify(
        figure_for(
            body.get("session"),
            body.get("chart"),
            node=body.get("node"),
            sensor=body.get("sensor", 0),
            positions=body.get("positions"),
            radii=body.get("radii"),
        )
    )


@digital_twin.route("/api/dt/sensor", methods=["POST"])
def sensor():
    """Apply one sensor: its angle and, when sent, its waveform CSV."""
    body = _body()
    return jsonify(
        apply_sensor(
            body.get("session"),
            body.get("node"),
            body.get("plane"),
            body.get("angle"),
            body.get("filename"),
            body.get("content"),
        )
    )
