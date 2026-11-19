# -*- coding: utf-8 -*-
"""The balancing analysis of step 4: what the EXECUTE button of the app does.

Not `services/balance.py`, which is the BRBalance calculator: that one takes
the influence coefficients of trial masses, while this one estimates the
unbalance from the measured signals with `rossml`.

The Streamlit app saves its session and hands it to `RossmlApi.run()`, which
unpickles it and calls `rossml`. Here the session is already in memory, so
`domain/rossml_balancing.py` is called with it and the answer is kept on the
session.

What comes back is an unbalance per balancing plane: a mass times a radius,
in g·mm, and its angle. The correction goes on the other side of the shaft,
so its angle is the estimate's plus 180 degrees, and its mass is the estimate
divided by the radius of that plane.
"""

import json

from ross.interface.extensions.digital_twin.domain import (
    equipment,
    rossml_balancing,
    session,
)
from ross.interface.extensions.digital_twin.services import measurement
from ross.interface.extensions.digital_twin.services.digital_twin import USER_MODEL
from ross.interface.extensions.digital_twin.services.analysis.base import figure

HALF_TURN = 180.0

# What the Streamlit app always passes to rossml: the hyperparameters of the
# machine, from `models_files/<model>/input_parameters.json`. They are not in
# ROSS, and the library defaults of rossml are not the same numbers, so an
# estimate computed without them does not reproduce the app's.
NO_PARAMETERS = (
    "Without the analysis parameters of the machine (input_parameters.json), "
    "the estimate runs on the library defaults of rossml and will not match "
    "the Digital Twin app."
)
MAX_ITERATIONS = 20
MAX_POPULATION = 50


def settings_for(identifier):
    """Return what the screen needs to offer the analysis."""
    current = session.get(identifier)
    chosen = _hyperparameters(current)
    return {
        "available": rossml_balancing.available(),
        "message": "" if rossml_balancing.available() else rossml_balancing.MISSING,
        "ready": _refusal(current) is None,
        "refusal": _refusal(current) or "",
        "balancing": _answer(current),
        "parameters": chosen,
        "parameters_warning": "" if chosen else NO_PARAMETERS,
    }


def set_parameters(identifier, filename, content):
    """Keep the analysis parameters of the machine for this session.

    Parameters
    ----------
    identifier : str
        Session id.
    filename : str
        Name of the JSON file, as the user chose it.
    content : str
        Its text, an `input_parameters.json` of the Digital Twin app. `None`
        drops the file and goes back to the model's own, when it has one.
    """
    current = session.get(identifier)
    if content is None:
        current.balancing_parameters = None
        return settings_for(identifier)
    if not isinstance(filename, str) or not filename.lower().endswith(".json"):
        raise ValueError("The analysis parameters must be a .json file.")
    try:
        values = json.loads(content)
    except ValueError as error:
        raise ValueError("Invalid parameters file %s: %s" % (filename, error)) from None
    if not isinstance(values, dict) or not values:
        raise ValueError(
            "The parameters file must hold a JSON object, as "
            "input_parameters.json does."
        )
    current.balancing_parameters = {"name": filename, "values": values}
    return settings_for(identifier)


def run_balancing(identifier, iterations=0, population=1, optimize=False):
    """Estimate the unbalance of the session's rotor and the correction masses.

    Parameters
    ----------
    identifier : str
        Session id.
    iterations : int, optional
        Extra iterations of the optimizer, 0 to 20.
    population : int, optional
        Initial population of the optimizer, 1 to 50.
    optimize : bool, optional
        Let the analysis choose which balancing planes to use. It triples the
        running time.

    Returns
    -------
    dict
        The state of the analysis, as `settings_for`, with the result in
        `balancing`.
    """
    current = session.get(identifier)
    refusal = _refusal(current)
    if refusal is not None:
        raise ValueError(refusal)

    planes = [plane["node"] for plane in current.balancing_planes]
    estimate = rossml_balancing.run(
        current.rotor,
        measurement.sensor_angles(current),
        planes,
        measurement.waveform_speed(current),
        current.database,
        current.results,
        iterations=_whole("extra iterations", iterations, 0, MAX_ITERATIONS),
        population=_whole("population", population, 1, MAX_POPULATION),
        optimize=bool(optimize),
        rotation=current.rotation,
        hyperparameters=(_hyperparameters(current) or {}).get("values"),
    )
    current.balancing = _corrections(current, estimate)
    current.balancing_polar = _polar(estimate)
    return settings_for(identifier)


def polar_for(identifier, view, sensor):
    """Return the polar chart of each balancing plane, for one view and sensor.

    Parameters
    ----------
    identifier : str
        Session id.
    view : str
        `A`, seen from the driven machine towards the driver, or `B`.
    sensor : str
        The sensor the chart is turned to, as `balancing.sensors` names it.

    Returns
    -------
    dict
        `charts`, one `{node, figure}` per balancing plane.
    """
    current = session.get(identifier)
    charts = (current.balancing_polar or {}).get((view, sensor))
    if charts is None:
        raise ValueError("Run the analysis before asking for its charts.")
    return {"view": view, "sensor": sensor, "charts": charts}


def _polar(estimate):
    """Group the charts by view and sensor, the two the screen chooses from."""
    charts = {}
    for item in estimate.get("polar", []):
        key = (item["view"], item["sensor"])
        charts.setdefault(key, []).append(
            {"node": item["node"], "figure": figure(item["figure"])}
        )
    return charts


def _corrections(current, estimate):
    """Turn the estimate into one correction mass per balancing plane."""
    radii = {plane["node"]: plane["radius"] for plane in current.balancing_planes}
    planes = []
    for index, node in enumerate(estimate["planes"]):
        unbalance = estimate["mu_d"][index]
        phase = estimate["phase"][index]
        radius = radii.get(node)
        planes.append(
            {
                "node": node,
                "radius": radius,
                "unbalance": unbalance,
                "unbalance_phase": phase,
                "mass": None if unbalance is None or not radius else unbalance / radius,
                "phase": None if phase is None else (phase + HALF_TURN) % 360.0,
            }
        )
    return {
        "planes": planes,
        "loss": estimate["loss"],
        "speed": measurement.waveform_speed(current),
        "measuring": sorted(current.measuring_nodes),
        # The angle of the unbalance is read against one sensor, so the screen
        # offers the choice the Streamlit app does. The mass does not depend
        # on it.
        "phases": estimate.get("phases", {}),
        "sensors": _sensors(current, estimate),
        "views": list(rossml_balancing.VIEWS),
    }


def _sensors(current, estimate):
    """Name each sensor of `phases` by the node and plane it sits on.

    rossml calls them `Sensor #1`, `#2`, ... in the order the measuring nodes
    were given, X before Y, which is the order `sensor_angles` keeps.
    """
    names = []
    for node in sorted(current.measuring_nodes):
        for plane in measurement.PLANES:
            names.append({"node": node, "plane": plane})
    keys = list(estimate.get("phases", {}))
    for index, key in enumerate(keys):
        if index < len(names):
            names[index]["key"] = key
    return [name for name in names if "key" in name]


def _answer(current):
    return current.balancing


def _hyperparameters(current):
    """Return the parameters to use: the uploaded file, or the model's own."""
    if current.balancing_parameters is not None:
        return {
            "name": current.balancing_parameters["name"],
            "source": "file",
            "keys": sorted(current.balancing_parameters["values"]),
            "values": current.balancing_parameters["values"],
        }
    values = _model_parameters(current)
    if not values:
        return None
    return {
        "name": "input_parameters.json",
        "source": "model",
        "keys": sorted(values),
        "values": values,
    }


def _model_parameters(current):
    """The parameters of a pre-established model, which lives outside ROSS."""
    if not current.model or current.model == USER_MODEL:
        return None
    try:
        return equipment.analysis_parameters(current.model)
    except (ValueError, OSError):
        return None


def _refusal(current):
    """Say why the analysis cannot run yet, or `None` when it can."""
    if not rossml_balancing.available():
        return rossml_balancing.MISSING
    if current.rotor is None:
        return "Create the model (step 3) before the analysis."
    step = measurement.describe(current)
    if not step["complete"]:
        return (
            "Choose at least two measuring planes and apply the signal of "
            "every sensor before the analysis."
        )
    if not current.balancing_planes:
        return "Choose at least one balancing plane."
    return None


def _whole(name, value, least, most):
    whole = "The %s must be a whole number." % name
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(whole)
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(whole) from None
    if not least <= number <= most:
        raise ValueError("The %s must be between %d and %d." % (name, least, most))
    return number
