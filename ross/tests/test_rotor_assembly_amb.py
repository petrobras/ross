import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.linalg import eigh
import copy
from types import SimpleNamespace

from ross import SensitivityResults, MagneticBearingElement, AmbNonCollocationResults
from ross.bearings.magnetic.amb_models import (
    rotor_example_amb_simple,
    rotor_example_amb_complex_controllers,
    rotor_example_amb_general_controllers,
)
from ross.bearings.magnetic.amb_time_response import AmbTimeResponse
from ross.probe import Probe
from ross.units import Q_


def test_amb_test_rig_rotor_matches_r6_free_free_frequencies():
    """The test rig rotor reproduces the free-free bending modes of model R6.

    Reference: first three free-free bending frequencies of the consolidated
    model R6 of the LMEst/UFU test rig (5 mm mesh, disk with its center of mass
    1.03 mm away from its node). The example places the disk on its node,
    which shifts these frequencies by -0.05 %, +0.25 % and -0.35 %.
    """
    rotor = rotor_example_amb_complex_controllers(ambs=False)

    assert_allclose(rotor.m, 5.89, rtol=1e-12)
    assert_allclose(rotor.nodes_pos[-1] - rotor.nodes_pos[0], 0.645, rtol=1e-12)

    # Bending in the x plane: x and beta of every node (6 dof per node).
    nodes = np.arange(len(rotor.nodes))
    dofs = np.ravel(np.column_stack([6 * nodes, 6 * nodes + 4]))
    M = rotor.M(0)[np.ix_(dofs, dofs)]
    K = rotor.K(0)[np.ix_(dofs, dofs)]
    eigenvalues = eigh(K, M, eigvals_only=True)

    # Two rigid-body modes (translation and tilt), then the bending modes.
    assert np.all(np.abs(eigenvalues[:2]) < (2 * np.pi * 0.1) ** 2)
    frequencies = np.sqrt(eigenvalues[2:5]) / (2 * np.pi)
    assert_allclose(frequencies, [106.913218, 400.902276, 769.279991], rtol=4e-3)


@pytest.mark.parametrize(
    "rotor_example",
    [rotor_example_amb_complex_controllers, rotor_example_amb_general_controllers],
)
def test_run_time_response_amb_controllers_is_finite(rotor_example):
    rotor = rotor_example()
    t = np.linspace(0, 0.1, 101)
    F = np.zeros((len(t), rotor.ndof))

    response = rotor.run_time_response(speed=0, F=F, t=t, weight=True)

    assert np.isfinite(response.yout).all()
    assert np.abs(response.yout).max() > 0


def test_amb_time_response_initial_condition_is_unscaled():
    rotor = rotor_example_amb_complex_controllers()
    t = np.linspace(0, 0.01, 11)
    x_0 = 1e-5
    sim = AmbTimeResponse(
        rotor, t=t, speed=0, disturbance=np.zeros((len(t), 4)), x_0=x_0, weight=False
    )
    sim.run()

    assert np.isfinite(sim.y).all()
    assert_allclose(sim.y[0, 1 : rotor.ndof : 6], x_0, rtol=1e-6)
    assert np.all(sim.state_scaling > 0)


def test_run_time_response_amb_values():
    rotor = rotor_example_amb_simple()
    t = np.arange(0, 2, 1e-6)
    F = np.zeros((len(t), rotor.ndof))
    speed = 0

    response = rotor.run_time_response(speed=speed, F=F, t=t, weight=True)

    expected_yout = np.array(
        [
            0.00000000e00,
            -4.90324857e-12,
            -1.96129877e-11,
            -4.41292062e-11,
            -7.84518919e-11,
            -1.22581032e-10,
            -1.76516612e-10,
            -2.40258617e-10,
            -3.13807034e-10,
            -3.97161847e-10,
        ]
    )

    expected_x_amb_0 = np.array(
        [
            0.00000000e00,
            1.71407425e-28,
            6.95526899e-28,
            1.31669341e-27,
            2.22296773e-27,
            4.30253731e-27,
            5.64867466e-27,
            8.87205587e-27,
            1.31782629e-26,
            2.09278634e-26,
        ]
    )

    expected_v_amb_0 = np.array(
        [
            0.00000000e00,
            -3.46710962e-12,
            -1.38683907e-11,
            -3.12037697e-11,
            -5.54731708e-11,
            -8.66765164e-11,
            -1.24813726e-10,
            -1.69884719e-10,
            -2.21889412e-10,
            -2.80827718e-10,
        ]
    )

    expected_F_y_amb_0 = np.array(
        [
            0.00000000e00,
            2.25425235e-06,
            9.01697835e-06,
            2.02881302e-05,
            3.60676587e-05,
            5.63555134e-05,
            8.11516424e-05,
            1.10455993e-04,
            1.44268510e-04,
            1.82589139e-04,
        ]
    )

    expected_F_v_amb_0 = np.array(
        [
            0.00000000e00,
            1.59399712e-06,
            6.37596653e-06,
            1.43458744e-05,
            2.55036861e-05,
            3.98493657e-05,
            5.73828766e-05,
            7.81041814e-05,
            1.02013242e-04,
            1.29110018e-04,
        ]
    )

    expected_I_v_amb_0 = np.array(
        [
            0.00000000e00,
            3.46710617e-07,
            1.38683769e-06,
            3.12037388e-06,
            5.54731161e-06,
            8.66764311e-06,
            1.24813604e-05,
            1.69884553e-05,
            2.21889196e-05,
            2.80827446e-05,
        ]
    )

    x_amb_0 = response.xout[0][:10, 0]
    v_amb_0 = response.xout[1][:10, 0]
    F_y_amb_0 = response.xout[2][:10, 1]
    F_v_amb_0 = response.xout[3][:10, 0]
    I_v_amb_0 = response.xout[4][:10, 0]
    yout = response.yout[:10, 5 * 6 + 1]

    # The signals are O(1e-8) to O(1e-4), so an absolute tolerance of 1e-5
    # accepted almost any result (including the wrong sign of ks). Compare
    # relative to each signal instead. x_amb_0 is the horizontal direction,
    # which only carries round-off, so it keeps an absolute check.
    assert_allclose(yout, expected_yout, rtol=1e-6, atol=1e-15)
    assert_allclose(x_amb_0, expected_x_amb_0, rtol=0, atol=1e-18)
    assert_allclose(v_amb_0, expected_v_amb_0, rtol=1e-6, atol=1e-18)
    assert_allclose(F_y_amb_0, expected_F_y_amb_0, rtol=1e-6, atol=1e-12)
    assert_allclose(F_v_amb_0, expected_F_v_amb_0, rtol=1e-6, atol=1e-12)
    assert_allclose(I_v_amb_0, expected_I_v_amb_0, rtol=1e-6, atol=1e-14)

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


def test_amb_time_response_plots_are_downsampled():
    """The AMB time-response plots keep at most n_points original samples."""
    rotor = rotor_example_amb_simple()
    t = np.linspace(0, 0.1, 5001)
    F = np.zeros((len(t), rotor.ndof))
    response = rotor.run_time_response(speed=0, F=F, t=t, weight=True)

    for plot in (
        response.plot_amb_disps,
        response.plot_amb_currents,
        response.plot_amb_forces,
    ):
        full = plot()  # default: no reduction
        reduced = plot(n_points=500)
        assert len(reduced.data) == len(full.data) == 4
        for trace_full, trace in zip(full.data, reduced.data, strict=True):
            assert len(trace_full.x) == len(t)
            assert len(trace.x) <= 500
            assert np.isin(trace.x, trace_full.x).all()
            assert trace.x[0] == trace_full.x[0]
            assert trace.x[-1] == trace_full.x[-1]


def test_sensitivity_time_plot_is_downsampled():
    """Long sensor signals are reduced; the excitation keeps its visible window."""
    rotor = rotor_example_amb_simple()
    results = rotor.run_amb_sensitivity(
        speed=0,
        t_max=1e-2,
        dt=1e-5,
        disturbance_amplitude=1e-5,
        disturbance_min_frequency=0.001,
        disturbance_max_frequency=150,
        amb_tags=["AMB_1"],
        verbose=0,
    )
    t = results.sensitivity_run_time_results["t"]
    assert len(t) > 100

    full = results.plot_time_results()  # default: no reduction
    reduced = results.plot_time_results(n_points=100)

    for trace_full, trace in zip(full.data, reduced.data, strict=True):
        if trace.yaxis == "y2":  # excitation signal: first 30 time steps
            assert_equal(np.asarray(trace.x), t[:31])
            assert_equal(np.asarray(trace_full.x), t[:31])
        else:  # sensor signal
            assert len(trace_full.x) == len(t)
            assert len(trace.x) <= 100
            assert np.isin(trace.x, trace_full.x).all()


def test_amb_controller():
    # Test for the magnetic_bearing_controller method.

    rot_speed = 1200
    dt = 0.001
    t = np.arange(0.0, 500 * dt, dt)
    rotor = rotor_example_amb_general_controllers()
    unbalance_node = rotor.disk_elements[0].n
    probe_node = rotor.bearing_elements[0].n
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

    # atol=1e-6 made these checks vacuous (the MSE is O(1e-10)); compare
    # relative values only.
    # Regression values of the Newmark path. With dt = 1 ms the response is
    # not converged in dt (the lightly damped bending modes of the test rig
    # ring for the whole 0.5 s): the MSE changes by about 30 % at dt = 0.05 ms.
    assert_allclose(mse_x, np.array(9.384022345331516e-10), rtol=1e-6, atol=0)
    assert_allclose(mse_y, np.array(2.2015824426032983e-10), rtol=1e-6, atol=0)


def test_amb_generic_controller():

    # kp must exceed |ks| / ki = i0 / g0 = 1000 for the rotor to levitate
    # (the lead stage has unit DC gain). kp = 100 gave a negative static
    # stiffness, which only looked stable with the old sign of ks.
    kp = 3000.0
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
    rotor = rotor_example_amb_general_controllers(controller_transfer_function)
    unbalance_node = rotor.disk_elements[0].n
    probe_node = rotor.bearing_elements[0].n
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

    # Regression values of the Newmark path. With dt = 1 ms the response is
    # not converged in dt (the lightly damped bending modes of the test rig
    # ring for the whole 0.5 s): the MSE changes by about 30 % at dt = 0.05 ms.
    assert_allclose(mse_x, np.array(5.8702652415637476e-12), rtol=1e-6, atol=0)
    assert_allclose(mse_y, np.array(3.5803835185052845e-12), rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "number_of_ambs",
    [1, 3, 4],
)
def test_magnetic_bearing_controller_routes_sensor_to_actuator(
    monkeypatch,
    number_of_ambs,
):
    """Controller should read disp from the sensor node and disp_actuator
    from the actuator node, then apply the resulting force at the actuator.

    The stub below returns ``disp + 1000 * disp_actuator``, so the two
    contributions never overlap numerically. If disp_actuator ever picked
    up the sensor's value instead of the actuator's (or vice versa), the
    resulting force would be off by roughly a factor of 1000 relative to
    what is asserted below -- not a subtle rounding difference.
    """
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

    actuator_nodes = rotor_nodes[1 : number_of_ambs + 1]
    sensor_nodes = rotor_nodes[-number_of_ambs:]

    assert len(actuator_nodes) == number_of_ambs
    assert len(sensor_nodes) == number_of_ambs
    assert set(actuator_nodes).isdisjoint(sensor_nodes)

    magnetic_bearings = []
    disp_resp = np.zeros(rotor.ndof)
    expected_force = np.zeros(rotor.ndof)

    DISP_ACTUATOR_WEIGHT = 1000.0

    for index, (
        actuator_node,
        sensor_node,
    ) in enumerate(
        zip(
            actuator_nodes,
            sensor_nodes,
            strict=True,
        )
    ):
        amb = copy.deepcopy(template_amb)
        amb.n = int(actuator_node)
        amb.sensor_node = int(sensor_node)

        magnetic_bearings.append(amb)

        sensor_x_dof = rotor.number_dof * amb.sensor_node
        sensor_y_dof = sensor_x_dof + 1

        actuator_x_dof = rotor.number_dof * amb.n
        actuator_y_dof = actuator_x_dof + 1

        sensor_x_disp = float(index + 1)
        sensor_y_disp = -float(index + 2)

        # Deliberately distinct from the sensor values above, so that
        # routing either displacement to the wrong argument produces a
        # force that clearly does not match expected_force below.
        actuator_x_disp = 100.0 + index
        actuator_y_disp = -100.0 - index

        disp_resp[sensor_x_dof] = sensor_x_disp
        disp_resp[sensor_y_dof] = sensor_y_disp

        disp_resp[actuator_x_dof] = actuator_x_disp
        disp_resp[actuator_y_dof] = actuator_y_disp

        expected_force[actuator_x_dof] = (
            sensor_x_disp + DISP_ACTUATOR_WEIGHT * actuator_x_disp
        )
        expected_force[actuator_y_dof] = (
            sensor_y_disp + DISP_ACTUATOR_WEIGHT * actuator_y_disp
        )

    def passthrough_compute_amb_controller(
        self,
        *,
        current_offset,
        setpoint,
        disp,
        dof_index,
        disp_actuator=None,
    ):
        # disp must come from the sensor node, disp_actuator from the
        # actuator node -- see DISP_ACTUATOR_WEIGHT above for why a
        # routing mistake cannot pass silently here.
        if disp_actuator is None:
            disp_actuator = disp
        return disp + DISP_ACTUATOR_WEIGHT * disp_actuator, 0.0

    monkeypatch.setattr(
        MagneticBearingElement,
        "compute_amb_controller",
        passthrough_compute_amb_controller,
    )

    magnetic_force = rotor.magnetic_bearing_controller(
        step=0,
        magnetic_bearings=magnetic_bearings,
        time_step=1e-3,
        disp_resp=disp_resp,
        sensor_angle=0.0,
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
            "Magnetic Bearing 0": {"x": 0.9903847953, "y": 0.9903847953},
            "Magnetic Bearing 1": {"x": 0.9853701742, "y": 0.9853701742},
        },
        "abs_slice": {
            "Magnetic Bearing 0": {
                "x": np.array(
                    [0.9903847953, 0.99036109, 0.99031601, 0.99025433, 0.99018717]
                ),
                "y": np.array(
                    [0.9903847953, 0.99036109, 0.99031601, 0.99025433, 0.99018717]
                ),
            },
            "Magnetic Bearing 1": {
                "x": np.array(
                    [0.9853701742, 0.98533447, 0.98526399, 0.98516711, 0.98506162]
                ),
                "y": np.array(
                    [0.9853701742, 0.98533447, 0.98526399, 0.98516711, 0.98506162]
                ),
            },
        },
        "phase_slice": {
            "Magnetic Bearing 0": {
                "x": np.array(
                    [
                        0.00000000e00,
                        1.0432e-04,
                        1.8834e-04,
                        2.5073e-04,
                        2.8961e-04,
                    ]
                ),
                "y": np.array(
                    [
                        0.00000000e00,
                        1.0432e-04,
                        1.8834e-04,
                        2.5073e-04,
                        2.8961e-04,
                    ]
                ),
            },
            "Magnetic Bearing 1": {
                "x": np.array(
                    [
                        0.00000000e00,
                        1.6359e-04,
                        2.9704e-04,
                        3.9578e-04,
                        4.5735e-04,
                    ]
                ),
                "y": np.array(
                    [
                        0.00000000e00,
                        1.6359e-04,
                        2.9704e-04,
                        3.9578e-04,
                        4.5735e-04,
                    ]
                ),
            },
        },
        "dofs": {
            "Magnetic Bearing 0": {"x": 66, "y": 67},
            "Magnetic Bearing 1": {"x": 234, "y": 235},
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
                    1.42063148e-11,
                    2.26930764e-11,
                    3.22573880e-11,
                ]
            ),
            "sensor": np.array(
                [
                    0.00000000e00,
                    0.00000000e00,
                    -1.98665610e-15,
                    -9.99212194e-15,
                    -2.72185261e-14,
                ]
            ),
        },
        "frequencies_slice": np.array([0.0, 100.0, 200.0, 300.0, 400.0]),
    }

    r_tol = 0
    a_tol = 1e-8
    # The time signals are 1e-15 to 1e-11 m, so a_tol would accept any value.
    signal_rtol = 1e-6
    signal_atol = 1e-21

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
        atol=signal_atol,
        rtol=signal_rtol,
    )
    assert_allclose(
        time_results_amb_0_x["disturbed_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["disturbed"],
        atol=signal_atol,
        rtol=signal_rtol,
    )
    assert_allclose(
        time_results_amb_0_x["sensor_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["sensor"],
        atol=signal_atol,
        rtol=signal_rtol,
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
    """Return modal results for the selected AMB.

    Modes 0-11 of the example hold ten lateral modes and the first two
    torsional modes (indices 8 and 9), so the non-lateral filter is
    exercised.
    """
    rotor = rotor_example_amb_general_controllers()

    ambs = [
        bearing
        for bearing in rotor.bearing_elements
        if isinstance(bearing, MagneticBearingElement)
    ]
    analyzed_amb = ambs[0]
    analyzed_amb.sensor_node = int(analyzed_amb.n) - 1

    results = rotor.run_amb_non_collocation(
        magnetic_bearing=analyzed_amb,
        modes=range(12),
    )

    assert isinstance(results, AmbNonCollocationResults)
    assert results.actuator_node == analyzed_amb.n
    assert results.sensor_node == analyzed_amb.sensor_node

    np.testing.assert_array_equal(
        results.requested_mode_indices,
        np.arange(12),
    )
    np.testing.assert_array_equal(
        np.sort(np.concatenate([results.mode_indices, results.excluded_mode_indices])),
        np.arange(12),
    )
    assert len(results.excluded_mode_indices) == 2
    np.testing.assert_array_equal(
        results.excluded_mode_types,
        ["Torsional", "Torsional"],
    )

    np.testing.assert_array_equal(
        results.sensor_nodes,
        rotor.nodes,
    )

    n_modes = len(results.mode_indices)

    assert results.mode_shapes.shape == (
        n_modes,
        len(rotor.nodes),
    )

    assert all(
        array.shape
        == (
            n_modes,
            len(results.sensor_nodes),
        )
        for array in (
            results.modal_residues,
            results.normalized_residues,
            results.classifications,
        )
    )


def test_run_amb_non_collocation_with_multiple_ambs():
    """Store all AMBs while analyzing the selected one."""
    rotor = rotor_example_amb_general_controllers()

    ambs = [
        bearing
        for bearing in rotor.bearing_elements
        if isinstance(bearing, MagneticBearingElement)
    ]
    analyzed_amb = ambs[-1]

    results = rotor.run_amb_non_collocation(
        magnetic_bearing=analyzed_amb,
        modes=[0, 1],
    )

    assert results.actuator_node == analyzed_amb.n
    assert results.sensor_node == analyzed_amb.sensor_node

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


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (
            {"speed": 100.0},
            ValueError,
            "supports only speed=0",
        ),
        (
            {"modes": []},
            ValueError,
            "must contain at least one",
        ),
        (
            {"modes": [-1]},
            ValueError,
            "must be non-negative",
        ),
        (
            {"modes": [0, 0]},
            ValueError,
            "must not contain repeated values",
        ),
        (
            {"modes": [0.5]},
            TypeError,
            "Every value in modes must be an integer",
        ),
        (
            {"direction": "z"},
            ValueError,
            "direction must be",
        ),
        (
            {"residue_tolerance": 1.0},
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
    """Reject invalid public-analysis arguments."""
    rotor = rotor_example_amb_general_controllers()

    amb = next(
        bearing
        for bearing in rotor.bearing_elements
        if isinstance(bearing, MagneticBearingElement)
    )

    call_arguments = {
        "magnetic_bearing": amb,
        "modes": [0, 1],
        **kwargs,
    }

    with pytest.raises(
        error,
        match=message,
    ):
        rotor.run_amb_non_collocation(
            **call_arguments,
        )


def test_run_amb_non_collocation_low_actuator_participation(
    monkeypatch,
):
    """Keep modal residues when actuator participation is low."""
    rotor = rotor_example_amb_general_controllers()

    amb = next(
        bearing
        for bearing in rotor.bearing_elements
        if isinstance(bearing, MagneticBearingElement)
    )

    rotor_nodes = np.asarray(rotor.nodes, dtype=int)
    x_dofs = rotor.number_dof * rotor_nodes
    actuator_index = int(np.flatnonzero(rotor_nodes == amb.n)[0])

    eigenvectors = np.zeros(
        (rotor.ndof, 1),
        dtype=complex,
    )
    eigenvectors[x_dofs, 0] = 1.0
    eigenvectors[x_dofs[actuator_index], 0] = 0.01

    modal = SimpleNamespace(
        wd=np.array([100.0]),
        evectors=eigenvectors,
        shapes=[
            SimpleNamespace(
                mode_type="Lateral",
            )
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

    expected_mode_shape = np.ones(len(rotor_nodes))
    expected_mode_shape[actuator_index] = 0.01

    assert_allclose(
        results.mode_shapes[0],
        expected_mode_shape,
    )
    assert_allclose(
        results.modal_residues[0],
        0.01 * expected_mode_shape,
    )
    assert_allclose(
        results.normalized_residues[0],
        expected_mode_shape,
    )

    expected_classifications = np.ones(
        len(rotor_nodes),
        dtype=int,
    )
    expected_classifications[actuator_index] = 0

    assert_equal(
        results.classifications[0],
        expected_classifications,
    )
