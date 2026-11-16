# -*- coding: utf-8 -*-
"""BRBalance routes. The arithmetic lives in services.balance."""

import io

from flask import Blueprint, jsonify, request, send_file

from ross.interface.extensions.digital_twin.services.balance import (
    load_brb,
    report_bytes,
    run,
    save_brb,
)

balance = Blueprint("dt_balance", __name__)

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _body():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Send the form as JSON.")
    return payload


@balance.route("/api/balance/calculate", methods=["POST"])
def calculate():
    """Return the correction, the prediction and the polar figure."""
    return jsonify(run(_body()))


@balance.route("/api/balance/report", methods=["POST"])
def report():
    """Return the Word report for the same form."""
    payload, name = report_bytes(_body())
    return send_file(
        io.BytesIO(payload),
        mimetype=DOCX,
        as_attachment=True,
        download_name=name,
    )


@balance.route("/api/balance/save", methods=["POST"])
def save():
    """Return the form as a .brb file."""
    payload, name = save_brb(_body())
    return send_file(
        io.BytesIO(payload),
        mimetype="application/zip",
        as_attachment=True,
        download_name=name,
    )


@balance.route("/api/balance/open", methods=["POST"])
def open_file():
    """Read an uploaded .brb and return the form fields."""
    upload = request.files.get("file")
    if upload is None:
        raise ValueError("Send a .brb file.")
    return jsonify(load_brb(upload.read()))
