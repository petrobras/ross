# -*- coding: utf-8 -*-
"""Step 4 of the Digital Twin: measuring planes, balancing planes and sensors.

The same inputs as section 4 of the Streamlit app, the balancing analysis,
kept in the files `rossml` reads:

* the measuring and balancing planes are rotor nodes;
* each measuring node has two sensors, planes X and Y. The Y sensor sits 90
  degrees after the X sensor, as in the Streamlit app, so only the X angle is
  chosen;
* each sensor's waveform is a CSV file separated by `;`, written to
  `database/data_<node>_<x|y>.csv` in the session.

The analysis needs at least two measuring planes, which is when the Streamlit
app shows the sensor cards.
"""

import io
import os

import pandas as pd

from ross.interface.extensions.digital_twin.domain import session

PLANES = ("X", "Y")
Y_OFFSET = 90.0
SENSITIVITY = 1.0
MIN_MEASURING = 2


def set_planes(identifier, measuring, balancing):
    """Replace the measuring and balancing nodes of the session.

    Parameters
    ----------
    identifier : str
        Session id.
    measuring, balancing : list of int
        Rotor nodes, each at most once.

    Returns
    -------
    dict
        The step's state (`describe`). Sensors of nodes no longer measured
        are dropped, with their files.
    """
    current = _with_rotor(identifier)
    current.measuring_nodes = _nodes(current, measuring, "measuring")
    current.balancing_nodes = _nodes(current, balancing, "balancing")
    for node in list(current.sensors_plans):
        if int(node) not in current.measuring_nodes:
            _drop_sensor_files(current, node)
            del current.sensors_plans[node]
    return describe(current)


def apply_sensor(identifier, node, plane, angle, filename=None, content=None):
    """Keep one sensor's angle and, when sent, its waveform file.

    Parameters
    ----------
    identifier : str
        Session id.
    node : int
        A measuring node.
    plane : str
        `X` or `Y`.
    angle : float
        Angle of the X sensor, in degrees. The Y sensor is placed 90 degrees
        after it; the angle sent with a Y sensor is ignored.
    filename : str, optional
        Name of the CSV file.
    content : str, optional
        Text of the CSV file, separated by `;`. Without it, the file applied
        before is kept; a sensor's first application needs one.
    """
    current = _with_rotor(identifier)
    if node not in current.measuring_nodes or isinstance(node, bool):
        raise ValueError("Node %s is not a measuring plane." % node)
    if plane not in PLANES:
        raise ValueError("The sensor plane must be X or Y.")
    try:
        angle = float(angle)
    except (TypeError, ValueError):
        raise ValueError("The sensor position must be a number of degrees.") from None

    sensors = current.sensors_plans.setdefault(str(node), {})
    previous = sensors.get(plane)
    if content is None:
        if previous is None:
            raise ValueError("Choose the signal file (.csv) of this sensor.")
        filename, rows = previous["filename"], previous["rows"]
    else:
        rows = _write_waveform(current, node, plane, filename, content)

    sensors[plane] = {
        "probes_angle": angle,
        "probes_sensitivity": SENSITIVITY,
        "filename": filename,
        "rows": rows,
    }
    if plane == "X" and "Y" in sensors:
        sensors["Y"]["probes_angle"] = angle + Y_OFFSET
    if plane == "Y":
        x_sensor = sensors.get("X")
        sensors["Y"]["probes_angle"] = (
            x_sensor["probes_angle"] + Y_OFFSET if x_sensor else angle
        )
    return describe(current)


def describe(current):
    """Return the step's state for the screen."""
    pending = [
        "%d %s" % (node, plane)
        for node in current.measuring_nodes
        for plane in PLANES
        if plane not in current.sensors_plans.get(str(node), {})
    ]
    return {
        "nodes": len(current.rotor.nodes) if current.rotor is not None else 0,
        "measuring": list(current.measuring_nodes),
        "balancing": list(current.balancing_nodes),
        "sensors": current.sensors_plans,
        "pending": pending,
        "complete": len(current.measuring_nodes) >= MIN_MEASURING and not pending,
    }


def prune(current):
    """Drop the planes a rebuilt rotor no longer has."""
    count = len(current.rotor.nodes)
    current.measuring_nodes = [n for n in current.measuring_nodes if n < count]
    current.balancing_nodes = [n for n in current.balancing_nodes if n < count]
    for node in list(current.sensors_plans):
        if int(node) >= count:
            _drop_sensor_files(current, node)
            del current.sensors_plans[node]


def waveform_path(current, node, plane):
    """Return the session file of one sensor's waveform."""
    return os.path.join(current.database, "data_%d_%s.csv" % (int(node), plane.lower()))


def _with_rotor(identifier):
    current = session.get(identifier)
    if current.rotor is None:
        raise ValueError("Create the model (step 3) before choosing the planes.")
    return current


def _nodes(current, values, kind):
    if not isinstance(values, list):
        raise ValueError("The %s planes must be a list of nodes." % kind)
    count = len(current.rotor.nodes)
    nodes = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Invalid %s node: %r" % (kind, value))
        if not 0 <= value < count:
            raise ValueError(
                "Node %d does not exist: the model has nodes 0 to %d."
                % (value, count - 1)
            )
        if value in nodes:
            raise ValueError("Node %d is already a %s plane." % (value, kind))
        nodes.append(value)
    return nodes


def _write_waveform(current, node, plane, filename, content):
    if not isinstance(filename, str) or not filename.lower().endswith(".csv"):
        raise ValueError("The signal file must be a .csv file.")
    if not isinstance(content, str):
        raise ValueError("The signal file could not be read.")
    try:
        data = pd.read_csv(io.StringIO(content), delimiter=";")
    except (ValueError, pd.errors.ParserError) as error:
        raise ValueError("Invalid signal file %s: %s" % (filename, error)) from None
    if data.empty:
        raise ValueError("The signal file %s has no rows." % filename)
    os.makedirs(current.database, exist_ok=True)
    data.to_csv(waveform_path(current, node, plane), sep=";", index=False)
    return len(data)


def _drop_sensor_files(current, node):
    for plane in PLANES:
        path = waveform_path(current, node, plane)
        if os.path.isfile(path):
            os.remove(path)
