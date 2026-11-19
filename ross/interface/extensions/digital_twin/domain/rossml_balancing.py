# -*- coding: utf-8 -*-
"""Run the `rossml` balancing estimate on a rotor and its measured signals.

`rossml` is the Petrobras package that estimates the unbalance of a running
machine with an augmented Kalman filter. It is not a dependency of ROSS: the
Digital Twin asks `available()` first and says what is missing.

The Streamlit app reaches it through `RossmlApi.run()`, which reads the
Streamlit session pickled to disk. `rossml.api.API.run()` is the same two
calls without the pickles, but it reads the signals from a `database` folder
inside the rossml checkout, so the interface makes those two calls itself and
points `database_directory` at the session:

* `experimental_unbalance_fault_calling` reads the waveform CSVs and cuts and
  filters them against the reference sensor;
* `apply_api_calling` runs the filter and returns the estimated unbalance
  `mu_d`, in g·mm, and its phase, in degrees, for each balancing plane.

Every other hyperparameter is rossml's own default, read from `rossml.api`,
so no tuned value of theirs is copied into ROSS.

The signals are read from the folder, in the order `natsorted` puts the file
names, and matched one by one to the sensors given here: the measuring nodes
must be in ascending order, X before Y, or the signals land on the wrong
degrees of freedom. `services.measurement` keeps them that way.
"""

import importlib.util

PACKAGE = "rossml"

MISSING = (
    "The rossml package is not installed: the balancing analysis cannot run "
    "here. Install rossml (and rossxl, which it needs) and restart the "
    "interface."
)

PLANES = ("x", "y")
REFERENCE_SENSOR = "Sensor #1"

# The polar figures come in the order `BalancingResults.polar` builds them:
# the plan view, then the balancing plane, then the sensor whose angle the
# chart is drawn against. A view is where the machine is seen from, as the
# Streamlit app names it: `A` from the driven machine towards the driver, `B`
# the other way round.
VIEWS = ("A", "B")


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
        The contents of the machine's `input_parameters.json`, which replace
        the library defaults of rossml. Without them the estimate is not the
        one the Streamlit app computes, because that app always passes the
        file of the chosen model.

    Returns
    -------
    dict
        `planes`, the nodes the estimate ended up using; `mu_d`, one
        unbalance in g·mm per plane; `phase`, its angle in degrees, seen from
        the first sensor; `phases`, the same angles for every sensor;
        `polar`, one Plotly figure per view, plane and sensor; and `loss`, the
        value of the objective function, when rossml reports one.
    """
    if not available():
        raise ValueError(MISSING)

    from ross.interface.extensions.digital_twin.domain import headless_streamlit

    headless_streamlit.install()

    from rossml.api import (
        _default_indirect,
        _default_kalman_flow,
        _default_plot_figs,
        _default_reduction_params,
        _default_signal_parameters,
    )
    from rossml.kalman_filter import KalmanFilterApp

    nodes = [int(node) for node in sensors]
    if nodes != sorted(nodes):
        raise ValueError("The measuring nodes must be in ascending order.")

    kalman_flow = dict(_default_kalman_flow())
    kalman_flow["max_iter"] = int(iterations)
    kalman_flow["pop_size"] = int(population)

    parameters = {
        "speed": list(speed),
        "parquet_speed": list(speed),
        "xvc_list": [1],
        "number_points": 4096,
        "rotation_sense": rotation,
        "plots_lists_forces": ["1d", "polar", "2d"],
        "plots_lists_vs": ["1d", "2d", "3d", "deformed"],
        "indirect": _default_indirect(),
        "numeric_test": False,
        "numeric_settings": None,
        "reduction_params": _default_reduction_params(),
        "domain": "modal",
        "type_execAPI": "interface",
        "w_f0": 0.65,
        "verbose": False,
        "view_plans": "all",
        "dx": None,
        "N_multiple": None,
        "noise_rate": None,
        "mu_max_iter": None,
        "mu_pop_size": None,
        "df_multiple": None,
        "points_number": None,
        "transient_len": None,
        "resultant_force_limits": None,
        "signal_parameters": _default_signal_parameters(),
        "plot_figs": _default_plot_figs(),
        "kalman_flow": kalman_flow,
        "maxiter": kalman_flow["max_iter"],
        "popsize": kalman_flow["pop_size"],
        "analysis_type": "indirect",
        "plot_options_vs": ["1d", "2d", "3d", "deformed"],
        "plot_options_balancing": ["1d", "2d", "3d", "polar"],
        "config_balance_plans": plane_groups(balancing) if optimize else None,
        "local": True,
        "database_directory": database if database.endswith("/") else database + "/",
        "figs_directory": figures if figures.endswith("/") else figures + "/",
        "balance_plans": [(int(node), list(PLANES)) for node in balancing],
        "unbalance_plans": [int(node) for node in balancing],
        "sensors_plans": {int(node): float(angle) for node, angle in sensors.items()},
        "balancing_plans_nodes_list": [{"Nodes": int(node)} for node in balancing],
        "measuring_plans_nodes_list": list(nodes),
        "probes_angle_dict": {
            str(node): [float(angle), float(angle) + 90.0]
            for node, angle in sensors.items()
        },
    }

    filtered, built = KalmanFilterApp().experimental_unbalance_fault_calling(
        rotor_model=rotor, parameters_data=_with(parameters, hyperparameters)
    )
    _, results, loss, mu_d, phi = KalmanFilterApp().apply_api_calling(
        kalman_filter=filtered, parameters_data=built
    )
    answer = _answer(rotor, balancing, results, loss, mu_d, phi)
    answer["phases"] = _phases(phi, answer["planes"])
    answer["polar"] = _polar(results, built, mu_d, phi, rotation, answer["planes"])
    return answer


def _polar(results, parameters, mu_d, phi, rotation, planes):
    """Return the polar charts, one per view, balancing plane and sensor.

    `plot_results_balance_calling` draws them this way in the Streamlit app:
    the angles of the reference sensor are what the chart is turned to, so
    there is one chart per sensor, and the app lets the user choose which.
    """
    parameters["mu_d"], parameters["phi"] = mu_d, phi
    angles = parameters["probes_angle_dict"]
    reference = angles[list(angles)[0]]
    figures = results.polar(
        mu_d=mu_d,
        ang_sensor_ref=reference,
        phi=phi,
        rotation_sense=rotation,
        parameters_data=parameters,
    )
    sensors = list(phi) if isinstance(phi, dict) else [REFERENCE_SENSOR]
    found = []
    for view in VIEWS:
        for node in planes:
            for sensor in sensors:
                found.append({"view": view, "node": node, "sensor": sensor})
    if len(found) != len(figures):
        raise ValueError(
            "rossml answered %d polar charts, not the %d of %d views, %d "
            "planes and %d sensors."
            % (len(figures), len(found), len(VIEWS), len(planes), len(sensors))
        )
    for item, figure in zip(found, figures, strict=True):
        item["figure"] = figure
    return found


def _phases(phi, planes):
    """Return the unbalance angle of every plane, for each sensor."""
    if not isinstance(phi, dict):
        return {}
    return {
        sensor: [_number(_item(values, index)) for index in range(len(planes))]
        for sensor, values in phi.items()
    }


def _with(parameters, hyperparameters):
    """Lay the machine's hyperparameters over the defaults, as rossml does.

    `RossmlApi.run` builds its dictionary and then spreads the JSON file over
    it, so the file wins wherever both have a key, and takes `number_points`
    out of `signal_parameters`. The speed of the signals is kept: it is read
    from the signal files, while the file carries the speed of the machine the
    parameters were tuned at.
    """
    if not hyperparameters:
        return parameters
    if not isinstance(hyperparameters, dict):
        raise ValueError("The analysis parameters must be a JSON object.")
    merged = {**parameters, **hyperparameters}
    signal = merged.get("signal_parameters")
    if isinstance(signal, dict) and signal.get("number_points"):
        merged["number_points"] = signal["number_points"]
    merged["speed"] = parameters["speed"]
    merged["parquet_speed"] = parameters["parquet_speed"]
    merged["database_directory"] = parameters["database_directory"]
    merged["figs_directory"] = parameters["figs_directory"]
    merged["kalman_flow"] = {
        **merged.get("kalman_flow", {}),
        "max_iter": parameters["kalman_flow"]["max_iter"],
        "pop_size": parameters["kalman_flow"]["pop_size"],
    }
    merged["maxiter"] = parameters["maxiter"]
    merged["popsize"] = parameters["popsize"]
    merged["config_balance_plans"] = parameters["config_balance_plans"]
    return merged


def plane_groups(balancing):
    """Return the subsets of the balancing planes the optimizer may choose.

    Every leading and every trailing run of the planes, as `RossmlApi.run()`
    builds them, each as `(node, ["x", "y"])` pairs.
    """
    nodes = [int(node) for node in balancing]
    groups = [nodes[:end] for end in range(1, len(nodes) + 1)]
    groups += [nodes[start:] for start in range(1, len(nodes))]
    return [[(node, list(PLANES)) for node in group] for group in groups]


def _answer(rotor, balancing, results, loss, mu_d, phi):
    planes = _planes_used(rotor, balancing, results)
    phases = phi.get(REFERENCE_SENSOR) if isinstance(phi, dict) else phi
    answer = {"planes": planes, "mu_d": [], "phase": [], "loss": _number(loss)}
    for index in range(len(planes)):
        answer["mu_d"].append(_number(_item(mu_d, index)))
        answer["phase"].append(_number(_item(phases, index)))
    return answer


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
