# -*- coding: utf-8 -*-
"""Steps 1 to 3 of the Digital Twin: equipment, process data, rotor model."""

import importlib.util
import json
import math
import os
import re

from ross.interface.domain.ross_import import project_from_ross_file
from ross.interface.domain.rotor_builder import assemble_rotor
from ross.interface.extensions.digital_twin.domain import (
    equipment,
    model_settings,
    session,
)
from ross.interface.extensions.digital_twin.services import measurement
from ross.interface.extensions.digital_twin.domain.rotor_compat import (
    OVERRIDES,
    accept_any_signature,
)

ROTATIONS = {"clockwise", "counterclockwise"}
# The name a rotor of the user's Rotor Hub is saved under in the session.
USER_MODEL = "user_rotor"
SOURCES = {"pi", "sample"}
PI_TIME = re.compile(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}")
PI_CREDENTIALS = ("USER_SWAGGER", "SWAGGER_PASSWORD")


def load_process_data(identifier, tag, pi_time, rotation, source="pi"):
    """Read the process data for a tag and keep it in the session.

    Parameters
    ----------
    identifier : str
        Session id.
    tag : str
        Equipment tag, a key of the Digital Twin app's settings.json.
    pi_time : str
        Start of the 15-minute window, `YYYY/MM/DD HH:MM:SS`.
    rotation : str
        `clockwise` or `counterclockwise`.
    source : str
        `pi` reads the PI Data server through `pandaspi`; `sample` reads the
        parquet file kept next to the model.
    """
    current = session.get(identifier)
    if rotation not in ROTATIONS:
        raise ValueError("Invalid rotation direction.")
    if source not in SOURCES:
        raise ValueError("Invalid process data source: %s" % source)
    if not isinstance(pi_time, str) or not PI_TIME.fullmatch(pi_time):
        raise ValueError("Enter date and time as YYYY/MM/DD HH:MM:SS.")
    model = equipment.model_for(tag)
    module = equipment.import_model(model)
    if source == "pi" and _reads_pi(module):
        _check_pi_access()
    frame = module.get_pi_data(tag, pi_time, source == "sample")
    if frame is None or len(frame) == 0:
        raise ValueError("The model returned no process data.")

    current.equipment = tag
    current.model = model
    current.rotation = rotation
    current.pi_time = pi_time
    current.source = source
    current.process_data = frame
    current.rotor = None
    current.components = None
    current.campbell = None
    return {
        "session": _summary(current),
        "table": _table(frame),
    }


def model_settings_for(tag):
    """Return the model of a tag and the additional settings it declares."""
    model = equipment.model_for(tag)
    return {"model": model, "settings": equipment.settings(model)}


def create_model(identifier, settings=None):
    """Build the ROSS rotor from the session's process data and draw it.

    Parameters
    ----------
    identifier : str
        Session id.
    settings : dict, optional
        `{name: {"value": number, "unit": str}}` for the model's additional
        settings (`model_settings.configured_data`). Missing ones keep the
        model's default.
    """
    current = session.get(identifier)
    if current.process_data is None:
        raise ValueError("Load the process data before creating the model.")
    configured = model_settings.configured_data(
        equipment.settings(current.model), settings or {}
    )
    module = equipment.import_model(current.model)
    rotor, bearings, labyrinths = module.create_model(
        df=current.process_data, fluid=None, configured_data=configured
    )
    accept_any_signature(rotor)
    rotor.save(os.path.join(current.session_files, current.model + ".toml"))

    current.rotor = rotor
    current.settings = configured
    measurement.prune(current)
    current.campbell = None
    current.components = {
        "bearings": _describe(bearings),
        "labyrinths": _describe(labyrinths),
    }
    return _model_answer(current)


def create_user_model(identifier, project, name, rotation):
    """Use a rotor of the user's Rotor Hub as the session's model.

    Parameters
    ----------
    identifier : str
        Session id.
    project : dict
        The rotor's project, as the Rotor Hub keeps it; it is built with the
        same code as the modelling screen (`domain/rotor_builder.py`).
    name : str
        The rotor's name in the Hub, shown as the equipment.
    rotation : str
        `clockwise` or `counterclockwise`.
    """
    current = session.get(identifier)
    if rotation not in ROTATIONS:
        raise ValueError("Invalid rotation direction.")
    if not isinstance(project, dict):
        raise ValueError("Choose a rotor of the Rotor Hub.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("The rotor needs a name.")
    rotor = assemble_rotor(project).rotor

    current.equipment = name.strip()
    current.model = USER_MODEL
    current.rotation = rotation
    current.pi_time = None
    current.source = None
    current.process_data = None
    current.settings = None
    current.rotor = rotor
    current.campbell = None
    current.components = {
        "bearings": _describe(list(rotor.bearing_elements)),
        "labyrinths": "",
    }
    measurement.prune(current)
    rotor.save(os.path.join(current.session_files, USER_MODEL + ".toml"))
    return _model_answer(current)


def _model_answer(current):
    """What the screen draws after a model is built."""
    rotor = current.rotor
    nan_warning = (
        "nan"
        in (current.components["bearings"] + current.components["labyrinths"]).lower()
    )
    return {
        "session": _summary(current),
        "rotor": {
            "nodes": len(rotor.nodes),
            "mass": float(rotor.m),
            "bearings": len(rotor.bearing_elements),
            "disks": len(rotor.disk_elements),
            "bearing_nodes": sorted({int(b.n) for b in rotor.bearing_elements}),
        },
        "components": current.components,
        "nan_warning": nan_warning,
        "settings": current.settings,
        "measurement": measurement.describe(current),
        "figure": _rotor_figure(rotor),
    }


def rotor_project(identifier):
    """Return the session's rotor as a project of the interface's Rotor Hub.

    The rotor goes through the same TOML file `create_model` saved and the
    same reader as "Load Rotor" (`domain/ross_import.py`), so the Hub receives
    what it would receive if the user loaded that file by hand.
    """
    current = session.get(identifier)
    if current.rotor is None:
        raise ValueError("Create the model before opening it in the Rotor Hub.")
    path = os.path.join(current.session_files, current.model + ".toml")
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    project = project_from_ross_file(content)
    project["name"] = "%s (%s)" % (current.equipment, current.pi_time)
    warnings = []
    if any(stored in vars(current.rotor) for stored in OVERRIDES.values()):
        warnings.append(
            "The model adjusts the assembled matrices after building the "
            "rotor. The Hub rebuilds the rotor from its elements, without "
            "that adjustment."
        )
    return {"projectData": project, "warnings": warnings}


def _reads_pi(module):
    path = getattr(module, "__file__", None)
    if not path:
        return False
    with open(path, encoding="utf-8", errors="replace") as handle:
        return "pandaspi" in handle.read()


def _check_pi_access():
    """Refuse a PI request that cannot work, saying what is missing."""
    if importlib.util.find_spec("pandaspi") is None:
        raise ValueError(
            "The pandaspi package is not installed: the process data cannot "
            "be read from PI Data. Install it, or choose the sample data."
        )
    missing = [name for name in PI_CREDENTIALS if not os.environ.get(name)]
    if missing:
        raise ValueError(
            "PI Data credentials not set: %s. Set them in the environment "
            "before starting ross-interface." % " and ".join(missing)
        )


def _rotor_figure(rotor):
    figure = rotor.plot_rotor()
    figure.update_xaxes(title_text="Axial Length [m]", showgrid=False, mirror=True)
    figure.update_yaxes(title_text="Radial Length [m]", showgrid=False, mirror=True)
    figure.update_layout(
        title={"text": "Model", "x": 0.5, "xanchor": "center"},
        height=500,
        margin={"r": 20},
    )
    return json.loads(figure.to_json())


def _table(frame):
    columns = [str(column) for column in frame.columns]
    rows = []
    for index, values in zip(frame.index, frame.itertuples(index=False), strict=True):
        rows.append([str(index)] + [_cell(value) for value in values])
    return {"columns": ["Time"] + columns, "rows": rows}


def _cell(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _describe(value):
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return "" if value is None else str(value)


def _summary(current):
    return {
        "id": current.id,
        "name": current.name,
        "equipment": current.equipment,
        "model": current.model,
        "rotation": current.rotation,
        "pi_time": current.pi_time,
        "source": current.source,
        "has_rotor": current.rotor is not None,
    }
