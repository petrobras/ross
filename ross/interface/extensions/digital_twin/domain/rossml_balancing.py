# -*- coding: utf-8 -*-
"""Run the `rossml` balancing estimate on a rotor and its measured signals.

`rossml` is the Petrobras package that estimates the unbalance of a running
machine with an augmented Kalman filter. It is not a dependency of ROSS: the
Digital Twin asks `available()` first and says what is missing.

The estimate goes through `rossml.api.API`, the entry point rossml offers to
other programs. `API.run` reads the waveform CSVs of the
session, runs the filter and keeps the results, and its `plot_*` methods draw
them: the virtual sensing of the displacements, the estimated forces, the
polar chart of each balancing plane and the decomposition of a correction
into masses at chosen positions. The `API` object is kept with the session,
so the charts are drawn when the screen asks for them.

The signals are read from the folder in the order `natsort` puts the file
names, and matched one by one to the sensors given here: the measuring nodes
must be in ascending order, X before Y, or the signals land on the wrong
degrees of freedom. `services.measurement` keeps them that way.
"""

import importlib.util
import inspect

PACKAGE = "rossml"

MISSING = (
    "The rossml package is not installed: the balancing analysis cannot run "
    "here. Install rossml and restart the interface."
)

PLANES = ("x", "y")
REFERENCE_SENSOR = "Sensor #1"

# What `run` decides itself, so a parameters file cannot replace it: the
# signals, the planes and the settings chosen on the screen.
OWNED = {
    "self",
    "rotor_model",
    "sensors_plans",
    "probes_angle",
    "probes_sensitivity",
    "balance_plans",
    "unbalance_plans",
    "speed",
    "rotation_sense",
    "verbose",
    "kalman_max_iter",
    "kalman_pop_size",
    "database_directory",
    "figs_directory",
    "config_balance_plans",
    "kwargs",
}

# The charts the screen offers. `node` names the plane a chart is drawn for;
# the polar chart has one per balancing plane and the 3D view takes none.
CHARTS = {
    "polar": "plot_balance_polar",
    "decomposed": "plot_balance_polar_decomposed",
    "forces_1d": "plot_balance_1d",
    "forces_2d": "plot_balance_2d",
    "vs_1d": "plot_vs_1d",
    "vs_2d": "plot_vs_2d",
    "vs_3d": "plot_vs_3d",
}


def available():
    """Tell whether rossml can be imported."""
    return importlib.util.find_spec(PACKAGE) is not None


def run(
    rotor,
    sensors,
    balancing,
    speed,
    database,
    figures,
    iterations=0,
    population=1,
    optimize=False,
    rotation="clockwise",
    hyperparameters=None,
):
    """Estimate the unbalance of the rotor from the measured signals.

    Parameters
    ----------
    rotor : ross.Rotor
        The rotor model of the session.
    sensors : dict
        Angle of the X sensor, in degrees, of each measuring node, in
        ascending node order.
    balancing : list of int
        Nodes where a correction mass can be placed.
    speed : list
        Rotation speed as `[value, unit]`, the speed the signal files were
        measured at.
    database : str
        Folder with the waveform files, `data_<node>_<x|y>.csv`.
    figures : str
        Folder where rossml writes its figures.
    iterations : int, optional
        Extra iterations of the optimizer. Its own iterations are
        `24 * (1 + iterations)`.
    population : int, optional
        Initial population of the optimizer.
    optimize : bool, optional
        Search which subset of the balancing planes explains the signals
        best, instead of using all of them. It triples the running time.
    rotation : str, optional
        `clockwise` or `counterclockwise`, seen from the driver.
    hyperparameters : dict, optional
        The analysis parameters, as `analysis_parameters.arguments` builds
        them, which replace the library defaults of rossml.

    Returns
    -------
    api : rossml.api.API
        The run, kept to draw its charts with `chart`.
    answer : dict
        `planes`, the nodes the estimate ended up using; `mu_d`, one
        unbalance in g·mm per plane; `phase`, its angle in degrees, seen from
        the first sensor; `phases`, the same angles for every sensor; and
        `loss`, the last value of the objective function, when rossml reports
        one.
    """
    if not available():
        raise ValueError(MISSING)

    from rossml.api import API

    nodes = [int(node) for node in sensors]
    if nodes != sorted(nodes):
        raise ValueError("The measuring nodes must be in ascending order.")

    api = API()
    api.run(
        rotor_model=rotor,
        sensors_plans=nodes,
        probes_angle=[float(sensors[node]) for node in sensors],
        balance_plans=[int(node) for node in balancing],
        speed=list(speed),
        rotation_sense=rotation,
        kalman_max_iter=int(iterations),
        kalman_pop_size=int(population),
        database_directory=_folder(database),
        figs_directory=_folder(figures),
        config_balance_plans=plane_groups(balancing) if optimize else None,
        verbose=False,
        **settings(hyperparameters),
    )
    answer = _answer(rotor, balancing, api)
    answer["phases"] = _phases(api.phi, answer["planes"])
    return api, answer


def settings(hyperparameters):
    """Turn the analysis parameters into arguments of `API.run`.

    They win over the library defaults wherever `API.run` takes the key, and
    `number_points` comes out of `signal_parameters`. What `run` decides (the
    signals, the planes, the speed the signals were measured at and the
    settings chosen on the screen) is not taken from them.
    """
    if not hyperparameters:
        return {}
    if not isinstance(hyperparameters, dict):
        raise ValueError("The analysis parameters must be an object.")
    from rossml.api import API

    accepted = set(inspect.signature(API.run).parameters) - OWNED
    chosen = {key: value for key, value in hyperparameters.items() if key in accepted}
    signal = hyperparameters.get("signal_parameters")
    if isinstance(signal, dict) and signal.get("number_points"):
        chosen["number_points"] = signal["number_points"]
    return chosen


def chart(api, kind, node=None, sensor=0, positions=None, radii=None):
    """Draw one of the charts of a run.

    Parameters
    ----------
    api : rossml.api.API
        A run returned by `run`.
    kind : str
        One of `CHARTS`.
    node : int, optional
        The plane the chart is for. The polar chart and the 3D view take
        none.
    sensor : int, optional
        Index of the reference sensor of the angles, X then Y of each
        measuring plane: the polar chart and the decomposition use it.
    positions, radii : list of float, optional
        For the decomposition: the angles, in degrees, and radii, in mm, of
        the positions the correction is split into.

    Returns
    -------
    list of plotly.graph_objects.Figure
    """
    if kind not in CHARTS:
        raise ValueError("Unknown chart: %s." % kind)
    method = getattr(api, CHARTS[kind])
    if kind == "polar":
        drawn = method(sensor_ref=int(sensor))
    elif kind == "decomposed":
        drawn = method(
            node=int(node),
            positions_list=[float(value) for value in positions],
            radius_list=[float(value) for value in radii],
            sensor_ref=int(sensor),
        )
    elif kind == "vs_3d":
        drawn = method()
    else:
        drawn = method(node=int(node))
    return list(drawn) if isinstance(drawn, (list, tuple)) else [drawn]


def plane_groups(balancing):
    """Return the subsets of the balancing planes the optimizer may choose.

    Every leading and every trailing run of the planes, as `RossmlApi.run()`
    builds them, each as `(node, ["x", "y"])` pairs.
    """
    nodes = [int(node) for node in balancing]
    groups = [nodes[:end] for end in range(1, len(nodes) + 1)]
    groups += [nodes[start:] for start in range(1, len(nodes))]
    return [[(node, list(PLANES)) for node in group] for group in groups]


def _folder(path):
    return path if path.endswith("/") else path + "/"


def _answer(rotor, balancing, api):
    planes = _planes_used(rotor, balancing, api.results_balance)
    phi = api.phi
    phases = phi.get(REFERENCE_SENSOR) if isinstance(phi, dict) else phi
    answer = {
        "planes": planes,
        "mu_d": [],
        "phase": [],
        "loss": _last(api.loss_function),
    }
    for index in range(len(planes)):
        answer["mu_d"].append(_number(_item(api.mu_d, index)))
        answer["phase"].append(_number(_item(phases, index)))
    return answer


def _phases(phi, planes):
    """Return the unbalance angle of every plane, for each sensor."""
    if not isinstance(phi, dict):
        return {}
    return {
        sensor: [_number(_item(values, index)) for index in range(len(planes))]
        for sensor, values in phi.items()
    }


def _planes_used(rotor, balancing, results):
    """Return the nodes the estimate used, which `optimize` may cut down to a few."""
    forces = getattr(results, "dof_forces", None)
    if forces is None:
        return [int(node) for node in balancing]
    number_dof = int(getattr(results, "number_dof", rotor.number_dof))
    used = []
    for dof in forces:
        node = int(dof) // number_dof
        if node not in used:
            used.append(node)
    return used


def _last(values):
    """The last value of the objective function's history, or the value itself."""
    if values is None:
        return None
    try:
        values = list(values)
    except TypeError:
        return _number(values)
    return _number(values[-1]) if values else None


def _item(values, index):
    if values is None:
        return None
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return None


def _number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
