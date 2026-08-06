import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import copy
from types import SimpleNamespace

from ross import SensitivityResults, MagneticBearingElement, AmbNonCollocationResults
from ross.bearings.magnetic.amb_models import (
    rotor_example_amb_simple,
    rotor_example_amb_general_controllers,
)
from ross.probe import Probe
from ross.units import Q_


def test_run_time_response_amb_values():
    rotor = rotor_example_amb_simple()
    t = np.arange(0, 2, 1e-6)
    F = np.zeros((len(t), rotor.ndof))
    speed = 0

    response = rotor.run_time_response(speed=speed, F=F, t=t, weight=True)

    expected_yout = np.array(
        [
            0.00000000e00,
            -1.35870226e-08,
            -5.43468162e-08,
            -1.22276734e-07,
            -2.17373160e-07,
            -3.39631510e-07,
            -4.89046215e-07,
            -6.65610691e-07,
            -8.69317307e-07,
            -1.10015734e-06,
        ]
    )

    expected_x_amb_0 = np.array(
        [
            0.00000000e00,
            2.52056841e-28,
            1.53697973e-27,
            4.36834784e-27,
            1.00165154e-26,
            2.31173210e-26,
            4.06316599e-26,
            5.63156133e-26,
            9.05077127e-26,
            1.14879669e-25,
        ]
    )

    expected_v_amb_0 = np.array(
        [
            0.00000000e00,
            -3.46710962e-12,
            -1.38683907e-11,
            -3.12037695e-11,
            -5.54731702e-11,
            -8.66765149e-11,
            -1.24813724e-10,
            -1.69884714e-10,
            -2.21889403e-10,
            -2.80827704e-10,
        ]
    )

    expected_F_y_amb_0 = np.array(
        [
            0.00000000e00,
            2.29979284e-06,
            9.19913968e-06,
            2.06979917e-05,
            3.67962986e-05,
            5.74940088e-05,
            8.27910692e-05,
            1.12687425e-04,
            1.47183022e-04,
            1.86277802e-04,
        ]
    )

    expected_F_v_amb_0 = np.array(
        [
            0.00000000e00,
            1.62619911e-06,
            6.50477405e-06,
            1.46356903e-05,
            2.60189123e-05,
            4.06544035e-05,
            5.85421265e-05,
            7.96820427e-05,
            1.04074113e-04,
            1.31718297e-04,
        ]
    )

    expected_I_v_amb_0 = np.array(
        [
            0.00000000e00,
            3.46710617e-07,
            1.38683769e-06,
            3.12037386e-06,
            5.54731155e-06,
            8.66764297e-06,
            1.24813601e-05,
            1.69884548e-05,
            2.21889187e-05,
            2.80827432e-05,
        ]
    )

    x_amb_0 = response.xout[0][:10, 0]
    v_amb_0 = response.xout[1][:10, 0]
    F_y_amb_0 = response.xout[2][:10, 1]
    F_v_amb_0 = response.xout[3][:10, 0]
    I_v_amb_0 = response.xout[4][:10, 0]
    yout = response.yout[:10, 5 * 6 + 1]

    assert_allclose(yout, expected_yout, atol=1e-5)
    assert_allclose(x_amb_0, expected_x_amb_0, atol=1e-5)
    assert_allclose(v_amb_0, expected_v_amb_0, atol=1e-5)
    assert_allclose(F_y_amb_0, expected_F_y_amb_0, atol=1e-5)
    assert_allclose(F_v_amb_0, expected_F_v_amb_0, atol=1e-5)
    assert_allclose(I_v_amb_0, expected_I_v_amb_0, atol=1e-5)

    fig1 = response.plot_amb_disps(axes=0)
    assert fig1 is not None

    fig2 = response.plot_amb_disps(axes=1)
    assert fig2 is not None

    fig3 = response.plot_amb_currents()
    assert fig3 is not None

    fig4 = response.plot_amb_forces(axes=0)
    assert fig4 is not None

    fig5 = response.plot_amb_forces(axes=1)
    assert fig5 is not None

    probe_disk_y = Probe(node=5, angle=Q_(90, "deg"), tag="Node 5 - Y")
    fig6 = response.plot_1d(probe=[probe_disk_y])
    assert fig6 is not None


def test_amb_controller():
    # Test for the magnetic_bearing_controller method.

    rot_speed = 1200
    dt = 0.001
    t = np.arange(0.0, 500 * dt, dt)
    unbalance_node = 27
    probe_node = 12

    rotor = rotor_example_amb_general_controllers()
    n = len(t)
    F = np.zeros((n, rotor.ndof))
    m_u = 0.010  # kg
    ex = 0.002  # m
    F0 = m_u * ex * rot_speed**2
    F[:, rotor.number_dof * unbalance_node + 0] = F0 * np.sin(rot_speed * t)
    F[:, rotor.number_dof * unbalance_node + 1] = F0 * np.cos(rot_speed * t)

    response = rotor.run_time_response(rot_speed, F, t, method="newmark")

    response_x = response.yout[:, rotor.number_dof * probe_node + 0]
    response_y = response.yout[:, rotor.number_dof * probe_node + 1]

    mse_x = 1 / n * np.sum(response_x**2)
    mse_y = 1 / n * np.sum(response_y**2)

    assert_allclose(mse_x, np.array(9.228097168398774e-10), rtol=1e-6, atol=1e-6)
    assert_allclose(mse_y, np.array(2.2135792430227363e-10), rtol=1e-6, atol=1e-6)


def test_amb_generic_controller():

    kp = 100.0
    ki = 0
    kd = 10.0
    n_f = 10_000

    s = MagneticBearingElement.s
    pid_controller = kp + ki / s + kd * s * (1 / (1 + (1 / n_f) * s))

    k_lead = 1
    T_lead = 0.5
    alpha_lead = 0.1
    lead_controller = k_lead * (T_lead * s + 1) / (alpha_lead * T_lead * s + 1)

    controller_transfer_function = pid_controller * lead_controller

    rot_speed = 1200
    dt = 0.001
    t = np.arange(0.0, 500 * dt, dt)
    unbalance_node = 27
    probe_node = 12

    rotor = rotor_example_amb_general_controllers(controller_transfer_function)
    n = len(t)
    F = np.zeros((n, rotor.ndof))
    m_u = 0.010  # kg
    ex = 0.002  # m
    F0 = m_u * ex * rot_speed**2
    F[:, rotor.number_dof * unbalance_node + 0] = F0 * np.sin(rot_speed * t)
    F[:, rotor.number_dof * unbalance_node + 1] = F0 * np.cos(rot_speed * t)

    response = rotor.run_time_response(rot_speed, F, t, method="newmark")

    response_x = response.yout[:, rotor.number_dof * probe_node + 0]
    response_y = response.yout[:, rotor.number_dof * probe_node + 1]

    mse_x = 1 / n * np.sum(response_x**2)
    mse_y = 1 / n * np.sum(response_y**2)

    assert_allclose(mse_x, np.array(7.934767106972457e-11), rtol=1e-6, atol=1e-6)
    assert_allclose(mse_y, np.array(3.6781959914042914e-11), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "number_of_ambs",
    [1, 3, 4],
)


def test_magnetic_bearing_controller_routes_sensor_to_actuator(
    monkeypatch,
    number_of_ambs,
    ):
    """Controller should measure and apply forces at each AMB node pair."""
    rotor = rotor_example_amb_general_controllers()

    template_amb = next(
        bearing
        for bearing in rotor.bearing_elements
        if isinstance(
            bearing,
            MagneticBearingElement,
        )
    )

    rotor_nodes = np.asarray(
        rotor.nodes,
        dtype=int,
    )

    actuator_nodes = rotor_nodes[
        1 : number_of_ambs + 1
    ]
    sensor_nodes = rotor_nodes[
        -number_of_ambs:
    ]

    assert len(actuator_nodes) == number_of_ambs
    assert len(sensor_nodes) == number_of_ambs
    assert set(actuator_nodes).isdisjoint(
        sensor_nodes
    )

    magnetic_bearings = []
    disp_resp = np.zeros(rotor.ndof)
    expected_force = np.zeros(rotor.ndof)

    for index, (
        actuator_node,
        sensor_node,
    ) in enumerate(
        zip(
            actuator_nodes,
            sensor_nodes,
        )
    ):
        amb = copy.deepcopy(template_amb)
        amb.n = int(actuator_node)
        amb.sensor_node = int(sensor_node)

        magnetic_bearings.append(amb)

        sensor_x_dof = (
            rotor.number_dof
            * amb.sensor_node
        )
        sensor_y_dof = sensor_x_dof + 1

        actuator_x_dof = (
            rotor.number_dof
            * amb.n
        )
        actuator_y_dof = actuator_x_dof + 1

        sensor_x_disp = float(index + 1)
        sensor_y_disp = -float(index + 2)

        disp_resp[sensor_x_dof] = sensor_x_disp
        disp_resp[sensor_y_dof] = sensor_y_disp

        # These values must not be used as measurements.
        disp_resp[actuator_x_dof] = 100.0 + index
        disp_resp[actuator_y_dof] = -100.0 - index

        expected_force[
            actuator_x_dof
        ] = sensor_x_disp
        expected_force[
            actuator_y_dof
        ] = sensor_y_disp

    def fake_compute_amb_controller(
        self,
        *,
        current_offset,
        setpoint,
        disp,
        dof_index,
    ):
        return disp, 0.0

    monkeypatch.setattr(
        MagneticBearingElement,
        "compute_amb_controller",
        fake_compute_amb_controller,
    )

    magnetic_force = (
        rotor.magnetic_bearing_controller(
            step=0,
            magnetic_bearings=magnetic_bearings,
            time_step=1e-3,
            disp_resp=disp_resp,
            sensor_angle=0.0,
        )
    )

    assert_allclose(
        magnetic_force,
        expected_force,
    )


def test_run_amb_sensitivity():
    """
    Tests the run_amb_sensitivity method for correctness of outputs and handling of various scenarios.
    """
    EXPECTED_SENSITIVITY_RESULTS = {
        "max_abs": {
            "Magnetic Bearing 0": {"x": 0.9915881235, "y": 0.9915881235},
            "Magnetic Bearing 1": {"x": 0.9880851953, "y": 0.9880851953},
        },
        "abs_slice": {
            "Magnetic Bearing 0": {
                "x": np.array(
                    [0.99158812, 0.99156866, 0.99153061, 0.99147841, 0.99142154]
                ),
                "y": np.array(
                    [0.99158812, 0.99156866, 0.99153061, 0.99147841, 0.99142154]
                ),
            },
            "Magnetic Bearing 1": {
                "x": np.array(
                    [0.9880852, 0.98805746, 0.98800146, 0.98792434, 0.98784035]
                ),
                "y": np.array(
                    [0.9880852, 0.98805746, 0.98800146, 0.98792434, 0.98784035]
                ),
            },
        },
        "phase_slice": {
            "Magnetic Bearing 0": {
                "x": np.array(
                    [
                        0.00000000e00,
                        8.77852477e-05,
                        1.59040274e-04,
                        2.11855244e-04,
                        2.44736262e-04,
                    ]
                ),
                "y": np.array(
                    [
                        0.00000000e00,
                        8.77852477e-05,
                        1.59040274e-04,
                        2.11855244e-04,
                        2.44736262e-04,
                    ]
                ),
            },
            "Magnetic Bearing 1": {
                "x": np.array(
                    [
                        0.00000000e00,
                        1.29420004e-04,
                        2.35610181e-04,
                        3.14075980e-04,
                        3.62979207e-04,
                    ]
                ),
                "y": np.array(
                    [
                        0.00000000e00,
                        1.29420004e-04,
                        2.35610181e-04,
                        3.14075980e-04,
                        3.62979207e-04,
                    ]
                ),
            },
        },
        "dofs": {
            "Magnetic Bearing 0": {"x": 72, "y": 73},
            "Magnetic Bearing 1": {"x": 258, "y": 259},
        },
        "time_results_slice": {
            "t": np.array([0.0, 0.0001, 0.0002, 0.0003, 0.0004]),
            "excitation": np.array(
                [
                    0.00000000e00,
                    6.67703996e-12,
                    1.42083014e-11,
                    2.27030685e-11,
                    3.22846065e-11,
                ]
            ),
            "disturbed": np.array(
                [
                    0.00000000e00,
                    6.67703996e-12,
                    1.42060807e-11,
                    2.26922938e-11,
                    3.22559336e-11,
                ]
            ),
            "sensor": np.array(
                [
                    0.00000000e00,
                    0.00000000e00,
                    -2.22067882e-15,
                    -1.07746729e-14,
                    -2.86728919e-14,
                ]
            ),
        },
        "frequencies_slice": np.array([0.0, 100.0, 200.0, 300.0, 400.0]),
    }

    r_tol = 0
    a_tol = 1e-8

    # Setup - run the analysis
    rotor = rotor_example_amb_general_controllers()
    results = rotor.run_amb_sensitivity(
        speed=0,
        t_max=1e-2,
        dt=1e-4,
        disturbance_amplitude=10e-6,
        disturbance_min_frequency=0.001,
        disturbance_max_frequency=150,
    )

    # Scenario 1: Default run verification
    # ------------------------------------
    assert isinstance(results, SensitivityResults)

    # Check types and shapes
    assert isinstance(results.sensitivities_frequencies, np.ndarray)
    assert isinstance(results.sensitivities_abs, dict)
    assert len(results.sensitivities_frequencies) == len(
        results.sensitivities_abs["Magnetic Bearing 0"]["x"]
    )

    # Check numerical values against golden values
    assert_allclose(
        results.sensitivities_frequencies[:5],
        EXPECTED_SENSITIVITY_RESULTS["frequencies_slice"],
        atol=a_tol,
        rtol=r_tol,
    )
    for amb_tag in results.max_abs_sensitivities:
        for axis in ["x", "y"]:
            assert_allclose(
                results.max_abs_sensitivities[amb_tag][axis],
                EXPECTED_SENSITIVITY_RESULTS["max_abs"][amb_tag][axis],
                atol=a_tol,
                rtol=r_tol,
            )
            assert_allclose(
                results.sensitivities_abs[amb_tag][axis][:5],
                EXPECTED_SENSITIVITY_RESULTS["abs_slice"][amb_tag][axis],
                atol=a_tol,
                rtol=r_tol,
            )
            assert_allclose(
                results.sensitivities_phase[amb_tag][axis][:5],
                EXPECTED_SENSITIVITY_RESULTS["phase_slice"][amb_tag][axis],
                atol=a_tol,
                rtol=r_tol,
            )

    assert_equal(results.sensitivity_compute_dofs, EXPECTED_SENSITIVITY_RESULTS["dofs"])

    time_results_amb_0_x = results.sensitivity_run_time_results["Magnetic Bearing 0"][
        "x"
    ]
    assert_allclose(
        results.sensitivity_run_time_results["t"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["t"],
        atol=a_tol,
        rtol=r_tol,
    )
    assert_allclose(
        time_results_amb_0_x["excitation_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["excitation"],
        atol=a_tol,
        rtol=r_tol,
    )
    assert_allclose(
        time_results_amb_0_x["disturbed_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["disturbed"],
        atol=a_tol,
        rtol=r_tol,
    )
    assert_allclose(
        time_results_amb_0_x["sensor_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["sensor"],
        atol=a_tol,
        rtol=r_tol,
    )

    # Scenario 2: Test with `amb_tags` argument
    # -----------------------------------------
    results_tagged = rotor.run_amb_sensitivity(
        speed=1200, t_max=1e-2, dt=1e-4, amb_tags=["Magnetic Bearing 1"]
    )
    assert "Magnetic Bearing 1" in results_tagged.sensitivities
    assert "Magnetic Bearing 0" not in results_tagged.sensitivities
    assert len(results_tagged.sensitivities) == 1

    # Test for non-existent tag
    with pytest.raises(RuntimeError) as excinfo:
        rotor.run_amb_sensitivity(
            speed=1200, t_max=1e-2, dt=1e-4, amb_tags=["NonExistentAMB"]
        )
    assert "No Magnetic Bearing with the given tag was found" in str(excinfo.value)

    # Test for incorrect type for amb_tags
    with pytest.raises(ValueError) as excinfo:
        rotor.run_amb_sensitivity(
            speed=1200, t_max=1e-2, dt=1e-4, amb_tags="Magnetic Bearing 0"
        )
    assert "`amb_tags` must be a list of strings" in str(excinfo.value)

    # Scenario 3: Test with custom disturbance parameters
    # ----------------------------------------------------
    results_custom_freq = rotor.run_amb_sensitivity(
        speed=1200,
        t_max=1e-2,
        dt=1e-4,
        disturbance_min_frequency=10,
        disturbance_max_frequency=200,
    )
    # Check if max sensitivity differs, indicating parameters were used
    assert not np.allclose(
        results.max_abs_sensitivities["Magnetic Bearing 0"]["x"],
        results_custom_freq.max_abs_sensitivities["Magnetic Bearing 0"]["x"],
    )


def test_run_amb_non_collocation():
    """Return modal results and metadata for the selected AMB."""
    rotor = rotor_example_amb_general_controllers()

    ambs = [
        bearing
        for bearing in rotor.bearing_elements
        if isinstance(
            bearing,
            MagneticBearingElement,
        )
    ]

    analyzed_amb = ambs[0]
    analyzed_amb.sensor_node = int(analyzed_amb.n) - 1

    results = rotor.run_amb_non_collocation(
        magnetic_bearing=analyzed_amb,
        modes=range(8),
    )

    assert isinstance(
        results,
        AmbNonCollocationResults,
    )
    assert results.actuator_node == analyzed_amb.n
    assert results.sensor_node == analyzed_amb.sensor_node

    np.testing.assert_array_equal(
        results.requested_mode_indices,
        np.arange(8),
    )
    np.testing.assert_array_equal(
        results.mode_indices,
        [0, 1, 2, 3, 6, 7],
    )
    np.testing.assert_array_equal(
        results.excluded_mode_indices,
        [4, 5],
    )
    np.testing.assert_array_equal(
        results.excluded_mode_types,
        ["Torsional", "Torsional"],
    )

    np.testing.assert_array_equal(
        results.sensor_nodes,
        rotor.nodes,
    )
    np.testing.assert_array_equal(
        results.all_actuator_nodes,
        [amb.n for amb in ambs],
    )
    np.testing.assert_array_equal(
        results.all_sensor_nodes,
        [amb.sensor_node for amb in ambs],
    )
    np.testing.assert_array_equal(
        results.all_amb_tags,
        [amb.tag for amb in ambs],
    )

    expected_shape = (
        len(results.mode_indices),
        len(rotor.nodes),
    )

    assert all(
        array.shape == expected_shape
        for array in (
            results.mode_shapes,
            results.modal_residues,
            results.normalized_residues,
            results.classifications,
        )
    )


def test_run_amb_non_collocation_with_multiple_ambs(
    monkeypatch,
    ):
    """All AMBs should be stored and individually selectable."""
    rotor = rotor_example_amb_general_controllers()

    modal = rotor.run_modal(
        speed=0.0,
        num_modes=12,
    )

    ambs = [
        bearing
        for bearing in rotor.bearing_elements
        if isinstance(
            bearing,
            MagneticBearingElement,
        )
    ]

    actuator_nodes = {
        int(amb.n)
        for amb in ambs
    }

    third_amb = copy.deepcopy(ambs[0])
    third_amb.n = next(
        int(node)
        for node in rotor.nodes
        if int(node) not in actuator_nodes
    )
    third_amb.sensor_node = third_amb.n
    third_amb.tag = "Magnetic Bearing 2"

    rotor.bearing_elements.append(third_amb)
    ambs.append(third_amb)

    monkeypatch.setattr(
        rotor,
        "run_modal",
        lambda **kwargs: modal,
    )

    expected_actuator_nodes = [
        amb.n
        for amb in ambs
    ]
    expected_sensor_nodes = [
        amb.sensor_node
        for amb in ambs
    ]
    expected_tags = [
        amb.tag
        for amb in ambs
    ]

    for analyzed_amb in ambs:
        results = rotor.run_amb_non_collocation(
            magnetic_bearing=analyzed_amb,
            modes=[0, 1],
        )

        assert results.actuator_node == analyzed_amb.n
        assert results.sensor_node == analyzed_amb.sensor_node

        np.testing.assert_array_equal(
            results.all_actuator_nodes,
            expected_actuator_nodes,
        )
        np.testing.assert_array_equal(
            results.all_sensor_nodes,
            expected_sensor_nodes,
        )
        np.testing.assert_array_equal(
            results.all_amb_tags,
            expected_tags,
        )


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (
            {
                "speed": 100.0,
            },
            ValueError,
            "supports only speed=0",
        ),
        (
            {
                "modes": [],
            },
            ValueError,
            "must contain at least one",
        ),
        (
            {
                "modes": [-1],
            },
            ValueError,
            "must be non-negative",
        ),
        (
            {
                "modes": [0, 0],
            },
            ValueError,
            "must not be repeated",
        ),
        (
            {
                "modes": [0.5],
            },
            TypeError,
            "Every modal index must be an integer",
        ),
        (
            {
                "direction": "z",
            },
            ValueError,
            "direction must be",
        ),
        (
            {
                "residue_tolerance": 1.0,
            },
            ValueError,
            "0 <= residue_tolerance < 1",
        ),
    ],
    ids=[
        "nonzero_speed",
        "empty_modes",
        "negative_mode",
        "repeated_mode",
        "noninteger_mode",
        "invalid_direction",
        "invalid_residue_tolerance",
    ],
)


def test_run_amb_non_collocation_invalid_arguments(
    kwargs,
    error,
    message,
    ):
    """Invalid analysis arguments should be rejected."""
    rotor = (
        rotor_example_amb_general_controllers()
    )

    amb = next(
        bearing
        for bearing in rotor.bearing_elements
        if isinstance(
            bearing,
            MagneticBearingElement,
        )
    )

    call_arguments = {
        "magnetic_bearing": amb,
        "modes": [0, 1],
    }
    call_arguments.update(kwargs)

    with pytest.raises(
        error,
        match=message,
    ):
        rotor.run_amb_non_collocation(
            **call_arguments
        )


def test_run_amb_non_collocation_low_actuator_participation(
    monkeypatch,
    ):
    """Low actuator participation should not discard the modal residue."""
    rotor = (
        rotor_example_amb_general_controllers()
    )

    amb = next(
        bearing
        for bearing in rotor.bearing_elements
        if isinstance(
            bearing,
            MagneticBearingElement,
        )
    )

    rotor_nodes = np.asarray(
        rotor.nodes,
        dtype=int,
    )
    x_dofs = (
        rotor.number_dof
        * rotor_nodes
    )

    eigenvectors = np.zeros(
        (
            rotor.ndof,
            1,
        ),
        dtype=complex,
    )
    eigenvectors[x_dofs, 0] = 1.0

    actuator_index = int(
        np.flatnonzero(
            rotor_nodes == amb.n
        )[0]
    )

    eigenvectors[
        x_dofs[actuator_index],
        0,
    ] = 0.01

    modal = SimpleNamespace(
        wd=np.array(
            [
                100.0,
            ]
        ),
        evectors=eigenvectors,
        shapes=[
            SimpleNamespace(
                mode_type="Lateral",
            ),
        ],
    )

    monkeypatch.setattr(
        rotor,
        "run_modal",
        lambda **kwargs: modal,
    )

    results = rotor.run_amb_non_collocation(
        magnetic_bearing=amb,
        modes=[0],
        direction="x",
        residue_tolerance=0.05,
    )

    expected_mode_shape = np.ones(
        len(rotor_nodes),
    )
    expected_mode_shape[
        actuator_index
    ] = 0.01

    np.testing.assert_allclose(
        results.mode_shapes[0],
        expected_mode_shape,
    )

    expected_residues = (
        expected_mode_shape[
            actuator_index
        ]
        * expected_mode_shape
    )

    np.testing.assert_allclose(
        results.modal_residues[0],
        expected_residues,
    )

    np.testing.assert_allclose(
        results.normalized_residues[0],
        expected_mode_shape,
    )

    expected_classifications = np.ones(
        len(rotor_nodes),
        dtype=int,
    )
    expected_classifications[
        actuator_index
    ] = 0

    np.testing.assert_array_equal(
        results.classifications[0],
        expected_classifications,
    )
