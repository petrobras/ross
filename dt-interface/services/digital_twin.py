# -*- coding: utf-8 -*-
"""Steps 1 to 3 of the Digital Twin: equipment, process data, rotor model."""

import json
import math
import os
import re

from domain import equipment, session
from domain.rotor_compat import accept_any_signature

ROTATIONS = {"clockwise", "counterclockwise"}
PI_TIME = re.compile(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}")


def load_process_data(identifier, tag, pi_time, rotation):
    """Read the process data for a tag and keep it in the session."""
    current = session.get(identifier)
    if rotation not in ROTATIONS:
        raise ValueError("Sentido de rotação inválido.")
    if not isinstance(pi_time, str) or not PI_TIME.fullmatch(pi_time):
        raise ValueError("Informe data e hora no formato AAAA/MM/DD HH:MM:SS.")
    model = equipment.model_for(tag)
    module = equipment.import_model(model)
    frame = module.get_pi_data(tag, pi_time, True)
    if frame is None or len(frame) == 0:
        raise ValueError("O modelo não devolveu dados de processo.")

    current.equipment = tag
    current.model = model
    current.rotation = rotation
    current.pi_time = pi_time
    current.process_data = frame
    current.rotor = None
    current.components = None
    current.campbell = None
    return {"session": _summary(current), "table": _table(frame)}


def create_model(identifier):
    """Build the ROSS rotor from the session's process data and draw it."""
    current = session.get(identifier)
    if current.process_data is None:
        raise ValueError("Carregue os dados de processo antes de criar o modelo.")
    module = equipment.import_model(current.model)
    rotor, bearings, labyrinths = module.create_model(df=current.process_data, fluid=None)
    accept_any_signature(rotor)
    rotor.save(os.path.join(current.session_files, current.model + ".toml"))

    current.rotor = rotor
    current.campbell = None
    current.components = {
        "bearings": _describe(bearings),
        "labyrinths": _describe(labyrinths),
    }
    nan_warning = "nan" in (current.components["bearings"] + current.components["labyrinths"]).lower()
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
        "figure": _rotor_figure(rotor),
    }


def _rotor_figure(rotor):
    figure = rotor.plot_rotor()
    figure.update_xaxes(title_text="Axial Length [m]", showgrid=False, mirror=True)
    figure.update_yaxes(title_text="Radial Length [m]", showgrid=False, mirror=True)
    figure.update_layout(
        title={"text": "Modelo", "x": 0.5, "xanchor": "center"},
        height=500,
        margin={"r": 20},
    )
    return json.loads(figure.to_json())


def _table(frame):
    columns = [str(column) for column in frame.columns]
    rows = []
    for index, values in zip(frame.index, frame.itertuples(index=False)):
        rows.append([str(index)] + [_cell(value) for value in values])
    return {"columns": ["Horário"] + columns, "rows": rows}


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
        "has_rotor": current.rotor is not None,
    }
