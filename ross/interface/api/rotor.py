# -*- coding: utf-8 -*-
"""Rotor assembly and drawing, and import of a native ROSS file."""

import json

from flask import Blueprint, jsonify, request

from ross.interface.domain.concatenation import concatenated_project
from ross.interface.domain.requests import (
    CONCATENATE_REQUEST,
    ROSS_FILE_REQUEST,
    ROTOR_REQUEST,
    SPLIT_REQUEST,
)
from ross.interface.domain.rotor_builder import build_rotor_from_ui
from ross.interface.domain.ross_import import project_from_ross_file
from ross.interface.domain.splitting import split_shaft

rotor_api = Blueprint("rotor", __name__)


@rotor_api.route("/build_rotor", methods=["POST"])
def build_rotor():
    """Draw the rotor described in the body {"project": {...}}.

    The project arrives inside an envelope, not loose at the root, so the
    interface can send only what describes the rotor. It used to send the whole
    projectData -- savedAnalyses included, with the already-rendered Plotly
    figures -- on every keystroke.

    BE-12, resolved in Phase 3 slice 5: margin, background, size, legend
    position and menu position left here and went to the screen, which is what
    knows the size of the panel the figure has to fit.

    What stayed is not presentation: `fix_shapes` repairs the `xref` of the
    lines ROSS draws anchored to the paper, and for that it needs the rotor's
    length. That is domain geometry, and the domain is here.
    """
    payload = ROTOR_REQUEST.read(request.get_json(silent=True))
    rotor = build_rotor_from_ui(payload["project"])
    fig = rotor.plot_rotor()

    fig_json_str = fig.to_json()
    fig_dict = json.loads(fig_json_str)

    layout = fig_dict.get("layout", {})

    def fix_shapes(shapes):
        """Make the lines ROSS anchors to the paper follow the x axis instead.

        It needs the rotor's length -- which is why it stays here, and not with
        the rest of the layout, which went to the screen.
        """
        if not shapes:
            return []
        for shape in shapes:
            if (
                shape.get("type") == "line"
                and shape.get("xref") == "paper"
                and shape.get("yref") == "y"
            ):
                shape["xref"] = "x"
                shape["x0"] = 0
                shape["x1"] = float(rotor.L)
        return shapes

    layout["shapes"] = fix_shapes(layout.get("shapes", []))

    # The menu buttons carry shapes too, inside their arguments.
    for menu in layout.get("updatemenus", []):
        for button in menu.get("buttons", []):
            args = button.get("args", [])
            if not args:
                continue
            method = button.get("method", "relayout")
            arguments = args[1] if method == "update" and len(args) > 1 else args[0]
            if isinstance(arguments, dict) and "shapes" in arguments:
                arguments["shapes"] = fix_shapes(arguments["shapes"])

    # No safety net here, deliberately. `Rotor.__init__` and
    # `MultiRotor.__init__` always assign `m` and `Ip` (rotor_assembly.py:495
    # and :511), so the old net -- `except Exception: mass = 0.0` -- covered a
    # case that does not exist and, if it ever did, would show zero mass on
    # screen as though it were measured. That is BE-05's shape: a wrong number
    # in place of an error.
    return jsonify(
        {
            "status": "success",
            "plot_json": json.dumps(fig_dict),
            "mass": float(rotor.m),
            "ip": float(rotor.Ip),
        }
    )


@rotor_api.route("/load_ross_file", methods=["POST"])
def load_ross_file():
    """Translate a native ROSS file into the interface's project."""
    payload = ROSS_FILE_REQUEST.read(request.get_json(silent=True))
    return jsonify(
        {
            "status": "success",
            "projectData": project_from_ross_file(payload["content"]),
        }
    )


@rotor_api.route("/api/rotor/concatenate", methods=["POST"])
def concatenate_rotors():
    """Join two rotors end to end and answer the project of the result.

    The answer is a **project**, in the same shape `/load_ross_file` returns,
    and not a chart: what the hub does with it is add a rotor to the library,
    editable like any other. That is the whole reason the domain reads the new
    numbering back from ROSS instead of keeping the joined `Rotor` -- a rotor
    the screen cannot edit would be a dead end with a picture on it.

    Nothing is caught here. A refusal from the domain is a `ValueError`, and
    `api/errors.py` turns that into a 400 carrying the message: the user reads
    "these two rotors were analysed under different rotor models" instead of a
    500 about an internal error. That channel is the one every domain refusal in
    this project already uses.
    """
    payload = CONCATENATE_REQUEST.read(request.get_json(silent=True))
    merged = concatenated_project(
        payload["first"],
        payload["second"],
        payload["first_conversions"],
        payload["second_conversions"],
    )
    return jsonify({"status": "success", "projectData": merged})


@rotor_api.route("/api/rotor/split_shaft", methods=["POST"])
def split_shaft_route():
    """Cut one shaft element in two and answer the project of the result.

    A project in and a project out, like `/concatenate`, and for the same
    reason: what comes back is what the modelling screen goes on editing.

    The route is here rather than in the browser even though nothing it calls
    needs ROSS. Two rules decide the result -- the linear interpolation of the
    four diameters and the shift of every node above the split -- and both
    belong to ROSS. Keeping them in Python is what lets `tests/test_splitting.py`
    check them against `Rotor.add_nodes` itself; the same rules written in
    JavaScript could only be checked against somebody's memory of it.

    Nothing is caught: a refusal from the domain is a `ValueError` and
    `api/errors.py` turns it into a 400 carrying the sentence, so the user reads
    "would land on node 2, which already exists" instead of a generic failure.
    """
    payload = SPLIT_REQUEST.read(request.get_json(silent=True))
    return jsonify(
        {
            "status": "success",
            "projectData": split_shaft(
                payload["project"], payload["index"], payload["offset"]
            ),
        }
    )
