from ross import MagneticBearingElement


def get_ambs(rotor) -> list:
    """
    Get magnetic bearing elements from a rotor, ordered by node number.

    Parameters
    ----------
    rotor : ross.Rotor
        The rotor model.

    Returns
    -------
    ambs : list
        A list of MagneticBearingElement objects found in the rotor,
        sorted by the node (n) in which they are located.

    Examples
    --------
    >>> import ross as rs
    >>> rotor_amb = rs.rotor_example_amb_complex_controllers()
    >>> ambs = get_ambs(rotor_amb)
    >>> len(ambs)
    2
    """
    ambs = [
        brg for brg in rotor.bearing_elements if isinstance(brg, MagneticBearingElement)
    ]
    return sorted(ambs, key=lambda brg: brg.n)


def has_ambs(rotor):
    """
    Check if the rotor has magnetic bearing elements.

    Parameters
    ----------
    rotor : ross.Rotor
        The rotor model.

    Returns
    -------
    has_ambs : bool
        True if the rotor has magnetic bearing elements, False otherwise.

    Examples
    --------
    >>> import ross as rs
    >>> rotor_amb = rs.rotor_example_amb_complex_controllers()
    >>> has_ambs(rotor_amb)
    True
    """
    ambs = get_ambs(rotor)
    return len(ambs) > 0


def apply_sensitivity_disturbance(
    step, x_dof, v_disp, w_disp, sens_dof, sens_dist, sens_results
):
    """
    Apply a disturbance to the sensor signal for sensitivity analysis.

    Parameters
    ----------
    step : int
        Current simulation step.
    x_dof : int
        DOF index for x-direction.
    v_disp : float
        Displacement in v-direction.
    w_disp : float
        Displacement in w-direction.
    sens_dof : int
        DOF index being disturbed.
    sens_dist : array-like
        Array of disturbances.
    sens_results : dict
        Dictionary to store sensitivity results.

    Returns
    -------
    v_disp, w_disp : float
        Updated displacements.

    Examples
    --------
    >>> sens_results = {"excitation_signal": [], "disturbed_signal": [], "sensor_signal": []}
    >>> v_disp, w_disp = apply_sensitivity_disturbance(
    ...     0, 0, 0.1, 0.2, 0, [0.01], sens_results
    ... )
    >>> v_disp
    0.11
    """
    is_x_dof = sens_dof == x_dof
    sensor_signal = v_disp if is_x_dof else w_disp
    excitation = sens_dist[step]

    if is_x_dof:
        v_disp += excitation
    else:
        w_disp += excitation

    disturbed_signal = v_disp if is_x_dof else w_disp

    sens_results["excitation_signal"].append(excitation)
    sens_results["disturbed_signal"].append(disturbed_signal)
    sens_results["sensor_signal"].append(sensor_signal)

    return v_disp, w_disp


def log_amb_data(
    amb_data,
    step,
    i,
    x_disp,
    y_disp,
    force_v,
    force_w,
    force_x,
    force_y,
    current_v,
    current_w,
):
    """
    Log magnetic bearing data for a given simulation step.

    Parameters
    ----------
    amb_data : dict
        Dictionary to store AMB data.
    step : int
        Current simulation step.
    i : int
        Index of the magnetic bearing.
    x_disp, y_disp : float
        Displacements in x and y directions.
    force_v, force_w : float
        Forces in v and w directions.
    force_x, force_y : float
        Forces in x and y directions.
    current_v, current_w : float
        Currents in v and w directions.

    Returns
    -------
    None

    Examples
    --------
    >>> import numpy as np
    >>> amb_data = {"x_amb": np.zeros((1, 2)), "v_amb": np.zeros((1, 2)),
    ...             "F_x": np.zeros((1, 2)), "F_v": np.zeros((1, 2)),
    ...             "I": np.zeros((1, 2))}
    >>> log_amb_data(amb_data, 1, 0, 0.1, 0.2, 10, 20, 30, 40, 1, 2)
    >>> amb_data["x_amb"][0, 0:2]
    array([0.1, 0.2])
    """
    idx = 2 * i
    step_idx = step - 1

    amb_data["x_amb"][step_idx, idx : idx + 2] = [x_disp, y_disp]
    amb_data["v_amb"][step_idx, idx : idx + 2] = [x_disp, y_disp]
    amb_data["F_x"][step_idx, idx : idx + 2] = [force_x, force_y]
    amb_data["F_v"][step_idx, idx : idx + 2] = [force_v, force_w]
    amb_data["I"][step_idx, idx : idx + 2] = [current_v, current_w]


def print_progress(step, time_step, progress_interval, force_x, force_y, tag):
    """
    Print the simulation progress for magnetic bearing forces.

    Parameters
    ----------
    step : int
        Current simulation step.
    time_step : float
        Time step size.
    progress_interval : float
        Interval for printing progress.
    force_x, force_y : float
        Forces in x and y directions.
    tag : str
        Tag for the bearing or rotor.

    Returns
    -------
    None

    Examples
    --------
    >>> print_progress(10, 0.1, 1.0, 100, 200, "AMB_1")
    Force x / y (N): 100.000000 / 200.000000 (AMB_1)
    """
    time_progress_ratio = (step * time_step) / progress_interval
    if abs(time_progress_ratio - round(time_progress_ratio)) < 1e-9:
        print(f"Force x / y (N): {force_x:.6f} / {force_y:.6f} ({tag})")
