# -*- coding: utf-8 -*-
"""Step 4 of the Digital Twin: measuring planes, balancing planes and sensors.

The same inputs as section 4 of the Digital Twin app, the balancing analysis,
kept in the files `rossml` reads:

* the measuring planes are rotor nodes, kept in ascending order, because
  `rossml` matches the signal files to the sensors in the order their names
  sort (`domain/rossml_balancing.py`);
* a balancing plane is a node and the radius, in millimetres, where a
  correction mass can be placed there: the estimate is a mass times a radius,
  and the radius turns it into a mass;
* each measuring node has two sensors, planes X and Y. The Y sensor sits 90
  degrees after the X sensor, so only the X angle is
  chosen;
* each sensor's waveform is a CSV file separated by `;`, written to
  `database/data_<node>_<x|y>.csv` in the session. It keeps the header the
  measuring system writes, which is where `rossml` reads the speed and the
  units of the signal from.

The analysis needs at least two measuring planes.
"""

import io
import os

import pandas as pd

from ross.interface.extensions.digital_twin.domain import session

PLANES = ("X", "Y")
Y_OFFSET = 90.0
SENSITIVITY = 1.0
MIN_MEASURING = 2

# The measuring system writes eleven lines before the samples: one title, nine
# of `name;value`, and the line that names the two columns. `rossml` reads the
# speed and the units out of them by name, and the samples with `skiprows=11`
# and a comma for the decimal point, so a file without them cannot be read.
HEADER_ROWS = 10
HEADER_KEYS = ("Point Name", "Sample Speed", "X-Axis Unit", "Y-Axis Unit")
SAMPLES_SKIP = 11

FORMAT = (
    "The signal file %s is not in the format of the measuring system: the "
    "samples must come after a header that gives %s. Export the waveform "
    "again instead of editing it."
)


def set_planes(identifier, measuring, balancing):
    """Replace the measuring and balancing planes of the session.

    Parameters
    ----------
    identifier : str
        Session id.
    measuring : list of int
        Rotor nodes, each at most once. They are kept sorted.
    balancing : list of dict
        One `{"node": int, "radius": float}` per plane, the radius in
        millimetres.

    Returns
    -------
    dict
        The step's state (`describe`). Sensors of nodes no longer measured
        are dropped, with their files.
    """
    current = _with_rotor(identifier)
    current.measuring_nodes = sorted(_nodes(current, measuring, "measuring"))
    current.balancing_planes = _planes(current, balancing)
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
        "balancing": [dict(plane) for plane in current.balancing_planes],
        "sensors": current.sensors_plans,
        "pending": pending,
        "complete": len(current.measuring_nodes) >= MIN_MEASURING and not pending,
    }


def sensor_angles(current):
    """Return the X angle of each measuring node, in ascending node order."""
    return {
        node: current.sensors_plans[str(node)]["X"]["probes_angle"]
        for node in sorted(current.measuring_nodes)
    }


def waveform_speed(current):
    """Return the speed the signals were measured at, as `[value, unit]`.

    It is read from the header of the first sensor's file, which is where
    `rossml` reads it too, so the analysis is never given a speed the signals
    disagree with.
    """
    node = sorted(current.measuring_nodes)[0]
    with open(waveform_path(current, node, "X"), encoding="utf-8") as handle:
        header = _header(handle.read())
    value = str(header.loc["Sample Speed"].values[0]).strip()
    number, _, unit = value.partition(" ")
    return [float(number), unit.strip() or "rpm"]


def prune(current):
    """Drop the planes a rebuilt rotor no longer has."""
    count = len(current.rotor.nodes)
    current.measuring_nodes = [n for n in current.measuring_nodes if n < count]
    current.balancing_planes = [
        plane for plane in current.balancing_planes if plane["node"] < count
    ]
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
        raise ValueError("Create the model before choosing the planes.")
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


def _planes(current, values):
    if not isinstance(values, list):
        raise ValueError("The balancing planes must be a list.")
    nodes = _nodes(current, [_plane_node(item) for item in values], "balancing")
    planes = []
    for node, item in zip(nodes, values, strict=True):
        planes.append({"node": node, "radius": _radius(node, item)})
    return planes


def _plane_node(item):
    if not isinstance(item, dict):
        raise ValueError("Invalid balancing plane: %r" % (item,))
    return item.get("node")


def _radius(node, item):
    radius = item.get("radius")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise ValueError(
            "Give the radius, in millimetres, where the correction mass goes "
            "on node %d." % node
        )
    if not radius > 0:
        raise ValueError("The radius of node %d must be greater than zero." % node)
    return float(radius)


def _header(content):
    """Return the header of a waveform file, indexed by the name of each line."""
    return pd.read_csv(
        io.StringIO(content), sep=";", nrows=HEADER_ROWS, index_col=0, dtype=str
    )


def _write_waveform(current, node, plane, filename, content):
    if not isinstance(filename, str) or not filename.lower().endswith(".csv"):
        raise ValueError("The signal file must be a .csv file.")
    if not isinstance(content, str):
        raise ValueError("The signal file could not be read.")
    try:
        header = _header(content)
        missing = [key for key in HEADER_KEYS if key not in header.index]
        samples = pd.read_csv(
            io.StringIO(content),
            sep=";",
            decimal=",",
            skiprows=SAMPLES_SKIP,
            names=["time", "amplitude"],
        ).astype(float)
    except (ValueError, IndexError, pd.errors.ParserError) as error:
        raise ValueError("Invalid signal file %s: %s" % (filename, error)) from None
    if missing:
        raise ValueError(FORMAT % (filename, ", ".join(missing)))
    if samples.empty:
        raise ValueError("The signal file %s has no samples." % filename)
    os.makedirs(current.database, exist_ok=True)
    with open(waveform_path(current, node, plane), "w", encoding="utf-8") as handle:
        handle.write(content)
    return len(samples)


def _drop_sensor_files(current, node):
    for plane in PLANES:
        path = waveform_path(current, node, plane)
        if os.path.isfile(path):
            os.remove(path)
