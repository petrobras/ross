"""Modal sensor-actuator non-collocation analysis for magnetic bearings."""

import numpy as np

from ross.bearing_seal_element import MagneticBearingElement
from ross.bearings.magnetic.amb_utils import get_ambs
from ross.results import AmbNonCollocationResults


_NUMERICAL_TOLERANCE = 1e-12


def _validate_real_scalar(value, name):
    """Return a finite real scalar as float."""
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar.")

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar.") from exc

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def _as_unique_integer_array(values, name):
    """Return a non-empty array of unique integers."""
    if isinstance(values, (int, np.integer)) and not isinstance(
        values, (bool, np.bool_)
    ):
        values = [int(values)]
    else:
        try:
            values = list(values)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be an integer or an array-like of integers."
            ) from exc

    if not values:
        raise ValueError(f"{name} must contain at least one value.")

    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise TypeError(f"Every value in {name} must be an integer.")

    values = np.asarray(values, dtype=int)

    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} must not contain repeated values.")

    return values


def _resolve_direction(direction):
    """Return the lateral projection angle in radians."""
    if isinstance(direction, str):
        direction = direction.strip().lower()

        if direction == "x":
            return 0.0
        if direction == "y":
            return np.pi / 2.0

        raise ValueError("direction must be 'x', 'y' or a numeric angle in radians.")

    direction_angle = _validate_real_scalar(direction, "direction")
    return direction_angle


def _select_lateral_modes(rotor, speed, requested_mode_indices):
    """Run the modal analysis and retain the requested lateral modes."""
    if requested_mode_indices is None:
        num_modes = 24
    else:
        num_modes = max(
            12,
            2 * (int(np.max(requested_mode_indices)) + 1),
        )

    modal = rotor.run_modal(
        speed=speed,
        num_modes=num_modes,
        sparse=num_modes < rotor.ndof - 1,
    )

    available_modes = min(
        len(modal.wd),
        modal.evectors.shape[1],
        len(modal.shapes),
    )

    if requested_mode_indices is None:
        lateral_modes = [
            index
            for index in range(available_modes)
            if str(modal.shapes[index].mode_type).lower() == "lateral"
        ]

        if len(lateral_modes) < 4:
            raise ValueError(
                "The modal analysis did not return at least four lateral modes."
            )

        requested_mode_indices = np.arange(
            lateral_modes[3] + 1,
            dtype=int,
        )

    elif np.max(requested_mode_indices) >= available_modes:
        raise ValueError(
            "A requested modal index is not available. "
            f"The modal analysis returned {available_modes} modes."
        )

    requested_mode_types = np.asarray(
        [str(modal.shapes[int(index)].mode_type) for index in requested_mode_indices],
        dtype=object,
    )

    lateral_mask = np.asarray(
        [mode_type.lower() == "lateral" for mode_type in requested_mode_types],
        dtype=bool,
    )

    mode_indices = requested_mode_indices[lateral_mask]
    excluded_mode_indices = requested_mode_indices[~lateral_mask]
    excluded_mode_types = requested_mode_types[~lateral_mask]

    if mode_indices.size == 0:
        excluded = ", ".join(
            f"{int(index)} ({mode_type})"
            for index, mode_type in zip(
                excluded_mode_indices,
                excluded_mode_types,
            )
        )
        raise ValueError(
            f"No lateral mode remains after filtering the requested modes: {excluded}."
        )

    frequencies = np.asarray(modal.wd, dtype=float) / (2.0 * np.pi)

    return {
        "modal": modal,
        "requested_mode_indices": requested_mode_indices,
        "mode_indices": mode_indices,
        "natural_frequencies": frequencies[mode_indices],
        "excluded_mode_indices": excluded_mode_indices,
        "excluded_mode_types": excluded_mode_types,
        "excluded_natural_frequencies": frequencies[excluded_mode_indices],
    }


def _project_mode_shapes(
    rotor,
    modal,
    mode_indices,
    rotor_nodes,
    actuator_index,
    direction_angle,
):
    """Project, phase-align, normalize and orient lateral mode shapes."""
    x_dofs = rotor.number_dof * rotor_nodes
    y_dofs = x_dofs + 1

    if np.max(y_dofs) >= rotor.ndof:
        raise ValueError(
            "Could not determine the lateral global DOFs for the rotor nodes."
        )

    direction_cosine = np.cos(direction_angle)
    direction_sine = np.sin(direction_angle)
    mode_shapes = np.zeros(
        (len(mode_indices), len(rotor_nodes)),
        dtype=float,
    )

    for row, mode_index in enumerate(mode_indices):
        eigenvector = np.asarray(
            modal.evectors[: rotor.ndof, mode_index],
            dtype=complex,
        )
        projected_shape = (
            direction_cosine * eigenvector[x_dofs]
            + direction_sine * eigenvector[y_dofs]
        )

        maximum = np.max(np.abs(projected_shape))
        if maximum <= _NUMERICAL_TOLERANCE:
            continue

        phase_reference = projected_shape[actuator_index]
        if np.abs(phase_reference) <= _NUMERICAL_TOLERANCE * max(1.0, maximum):
            phase_reference = projected_shape[int(np.argmax(np.abs(projected_shape)))]

        projected_shape *= np.exp(-1j * np.angle(phase_reference))

        real_shape = np.real(projected_shape)
        imaginary_shape = np.imag(projected_shape)
        if (
            np.max(np.abs(real_shape)) <= _NUMERICAL_TOLERANCE
            and np.max(np.abs(imaginary_shape)) > _NUMERICAL_TOLERANCE
        ):
            projected_shape = imaginary_shape
        else:
            projected_shape = real_shape

        projected_shape = np.asarray(projected_shape, dtype=float)
        maximum = np.max(np.abs(projected_shape))
        if maximum <= _NUMERICAL_TOLERANCE:
            continue

        projected_shape /= maximum

        if projected_shape[actuator_index] < -_NUMERICAL_TOLERANCE:
            projected_shape = -projected_shape

        mode_shapes[row] = projected_shape

    return mode_shapes


def _compute_modal_residues(
    mode_shapes,
    actuator_index,
    sensor_indices,
    residue_tolerance,
):
    """Calculate, normalize and classify modal residues."""
    modal_residues = (
        mode_shapes[:, actuator_index, np.newaxis] * mode_shapes[:, sensor_indices]
    )

    row_maxima = np.max(np.abs(modal_residues), axis=1)
    normalized_residues = np.divide(
        modal_residues,
        row_maxima[:, np.newaxis],
        out=np.zeros_like(modal_residues, dtype=float),
        where=row_maxima[:, np.newaxis] > _NUMERICAL_TOLERANCE,
    )

    classifications = np.zeros(normalized_residues.shape, dtype=int)
    classifications[normalized_residues > residue_tolerance] = 1
    classifications[normalized_residues < -residue_tolerance] = -1

    return modal_residues, normalized_residues, classifications


def run_amb_non_collocation(
    rotor,
    magnetic_bearing,
    speed=0.0,
    modes=None,
    sensor_nodes=None,
    direction="x",
    residue_tolerance=0.05,
):
    """Run a modal sensor-actuator non-collocation analysis.

    For each retained lateral mode, the modal residue is calculated as

    ``R_r(x_s) = phi_r(x_a) * phi_r(x_s)``

    where ``x_a`` is the actuator node, ``x_s`` is a candidate sensor node
    and ``phi_r`` is the real lateral mode shape projected in the requested
    direction.

    Parameters
    ----------
    rotor : Rotor
        Rotor model used in the modal analysis.
    magnetic_bearing : MagneticBearingElement
        Magnetic bearing selected for the analysis.
    speed : float, optional
        Rotor speed in rad/s. Only ``speed=0`` is currently supported.
    modes : int or array_like of int, optional
        Original zero-based modal indices requested for the analysis. Only
        lateral modes are retained. If omitted, the first four lateral modes
        are selected.
    sensor_nodes : int or array_like of int, optional
        Candidate sensor nodes. If omitted, all rotor nodes are used.
    direction : {"x", "y"} or float, optional
        Lateral projection direction. Numeric values are angles in radians
        measured from the global x direction.
    residue_tolerance : float, optional
        Threshold used to classify normalized residues near zero.

    Returns
    -------
    AmbNonCollocationResults
        Modal non-collocation analysis results.
    """
    if not isinstance(magnetic_bearing, MagneticBearingElement):
        raise TypeError("magnetic_bearing must be a MagneticBearingElement.")

    magnetic_bearings = get_ambs(rotor)
    if not any(bearing is magnetic_bearing for bearing in magnetic_bearings):
        raise ValueError("magnetic_bearing must belong to the analyzed rotor.")

    speed = _validate_real_scalar(speed, "speed")
    if not np.isclose(
        speed,
        0.0,
        atol=_NUMERICAL_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("run_amb_non_collocation currently supports only speed=0.")

    residue_tolerance = _validate_real_scalar(
        residue_tolerance,
        "residue_tolerance",
    )
    if not 0.0 <= residue_tolerance < 1.0:
        raise ValueError("residue_tolerance must satisfy 0 <= residue_tolerance < 1.")

    direction_angle = _resolve_direction(direction)

    requested_mode_indices = (
        None if modes is None else _as_unique_integer_array(modes, "modes")
    )
    if requested_mode_indices is not None and np.any(requested_mode_indices < 0):
        raise ValueError("Modal indices must be non-negative.")

    rotor_nodes = np.asarray(rotor.nodes, dtype=int)
    rotor_positions = np.asarray(rotor.nodes_pos, dtype=float)
    node_to_index = {int(node): index for index, node in enumerate(rotor_nodes)}

    actuator_node = int(magnetic_bearing.n)
    selected_sensor_node = int(
        magnetic_bearing.n
        if magnetic_bearing.sensor_node is None
        else magnetic_bearing.sensor_node
    )

    if actuator_node not in node_to_index:
        raise ValueError(f"Actuator node {actuator_node} is not present in the rotor.")
    if selected_sensor_node not in node_to_index:
        raise ValueError(
            f"Sensor node {selected_sensor_node} is not present in the rotor."
        )

    if sensor_nodes is None:
        candidate_sensor_nodes = rotor_nodes.copy()
    else:
        candidate_sensor_nodes = _as_unique_integer_array(
            sensor_nodes,
            "sensor_nodes",
        )
        invalid_nodes = [
            int(node)
            for node in candidate_sensor_nodes
            if int(node) not in node_to_index
        ]
        if invalid_nodes:
            raise ValueError(
                f"Sensor node {invalid_nodes[0]} is not present in the rotor."
            )

    if selected_sensor_node not in candidate_sensor_nodes:
        raise ValueError(
            "The magnetic bearing sensor_node must be included in sensor_nodes."
        )

    candidate_sensor_indices = np.asarray(
        [node_to_index[int(node)] for node in candidate_sensor_nodes],
        dtype=int,
    )
    sensor_positions = rotor_positions[candidate_sensor_indices]

    modal_data = _select_lateral_modes(
        rotor,
        speed,
        requested_mode_indices,
    )

    actuator_index = node_to_index[actuator_node]
    mode_shapes = _project_mode_shapes(
        rotor,
        modal_data["modal"],
        modal_data["mode_indices"],
        rotor_nodes,
        actuator_index,
        direction_angle,
    )

    (
        modal_residues,
        normalized_residues,
        classifications,
    ) = _compute_modal_residues(
        mode_shapes,
        actuator_index,
        candidate_sensor_indices,
        residue_tolerance,
    )

    all_actuator_nodes = np.asarray(
        [int(bearing.n) for bearing in magnetic_bearings],
        dtype=int,
    )
    all_sensor_nodes = np.asarray(
        [
            int(bearing.n if bearing.sensor_node is None else bearing.sensor_node)
            for bearing in magnetic_bearings
        ],
        dtype=int,
    )
    all_amb_tags = np.asarray(
        [
            (
                str(bearing.tag)
                if bearing.tag is not None
                else f"Magnetic Bearing {index}"
            )
            for index, bearing in enumerate(magnetic_bearings)
        ],
        dtype=object,
    )

    for actuator, sensor in zip(all_actuator_nodes, all_sensor_nodes):
        if int(actuator) not in node_to_index:
            raise ValueError(
                f"Actuator node {int(actuator)} is not present in the rotor."
            )
        if int(sensor) not in node_to_index:
            raise ValueError(f"Sensor node {int(sensor)} is not present in the rotor.")

    return AmbNonCollocationResults(
        speed=speed,
        actuator_node=actuator_node,
        sensor_node=selected_sensor_node,
        sensor_nodes=candidate_sensor_nodes,
        sensor_positions=sensor_positions,
        rotor_nodes=rotor_nodes,
        rotor_positions=rotor_positions,
        mode_indices=modal_data["mode_indices"],
        natural_frequencies=modal_data["natural_frequencies"],
        mode_shapes=mode_shapes,
        modal_residues=modal_residues,
        normalized_residues=normalized_residues,
        classifications=classifications,
        direction_angle=direction_angle,
        residue_tolerance=residue_tolerance,
        all_actuator_nodes=all_actuator_nodes,
        all_sensor_nodes=all_sensor_nodes,
        all_amb_tags=all_amb_tags,
        requested_mode_indices=modal_data["requested_mode_indices"],
        excluded_mode_indices=modal_data["excluded_mode_indices"],
        excluded_mode_types=modal_data["excluded_mode_types"],
        excluded_natural_frequencies=modal_data["excluded_natural_frequencies"],
    )
