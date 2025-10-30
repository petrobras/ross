import pytest
from numpy.testing import assert_allclose
import numpy as np
from scipy import integrate

from ross.bearing_seal_element import BearingElement
from ross.disk_element import DiskElement
from ross.materials import steel
from ross.rotor_assembly import Rotor
from ross.shaft_element import ShaftElement


@pytest.fixture
def rotor1():
    #  Rotor with damping
    #  Rotor with 6 shaft elements, 2 disks and 2 bearings
    i_d = 0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l,
            i_d,
            o_d,
            material=steel,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
            alpha=10,
            beta=1e-4,
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.35)

    stfx = 1e6
    stfy = 1e6
    c = 1e3
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=c, cyy=c)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=c, cyy=c)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


@pytest.fixture
def rotor2():
    #  Rotor with damping
    #  Rotor with 6 shaft elements, 2 disks and 2 bearings with frequency dependent coefficients
    i_d = 0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l,
            i_d,
            o_d,
            material=steel,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
            alpha=10,
            beta=1e-4,
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.35)

    stfx = [1e6, 1.5e6]
    stfy = [1e6, 1.5e6]
    c = [1e3, 1.5e3]
    frequency = [50, 5000]
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=c, cyy=c, frequency=frequency)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=c, cyy=c, frequency=frequency)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def simulation_parameters():
    dt = 5e-4
    t0 = 0
    tf = 3.5
    t = np.arange(t0, tf + dt, dt)

    step1 = np.where(t == 3.0)[0][0]
    step2 = len(t) - 1

    probe_node = 3
    number_dof = 6
    dofx = probe_node * number_dof
    dofy = dofx + 1

    probe_params = {"time": [step1, step2], "dofs": [dofx, dofy]}

    return t, probe_params


def unbalance_force(rotor, speed, t):
    nodes = [2, 4]
    mag_unb = [0.01, 0.01]
    phase_unb = [np.pi / 2, 0]

    num_dof = rotor.number_dof
    F_size = rotor.ndof

    F = np.zeros((len(t), F_size))

    speed_is_array = isinstance(speed, (list, tuple, np.ndarray))

    if speed_is_array:
        theta = integrate.cumulative_trapezoid(speed, t, initial=0)
    else:
        theta = speed * t

    for i, node in enumerate(nodes):
        phi = phase_unb[i] + theta

        Fx = mag_unb[i] * (speed**2) * np.cos(phi)
        Fy = mag_unb[i] * (speed**2) * np.sin(phi)

        F[:, node * num_dof + 0] += Fx
        F[:, node * num_dof + 1] += Fy

    return F


def test_for_cte_speed(rotor1, rotor2):
    # Test for constant speed
    t, probe_params = simulation_parameters()
    speed = 50.0

    F = unbalance_force(rotor1, speed, t)

    response1 = rotor1.run_time_response(speed, F, t, method="newmark")
    response2 = rotor2.run_time_response(speed, F, t, method="newmark")

    s0 = probe_params["time"][0]
    s1 = probe_params["time"][1]
    dofx = probe_params["dofs"][0]
    dofy = probe_params["dofs"][1]

    assert_allclose(response1.yout[s0:s1, dofx], response2.yout[s0:s1, dofx])
    assert_allclose(response1.yout[s0:s1, dofy], response2.yout[s0:s1, dofy])

    freq = 7.96813
    abs_max = 7.51498e-5

    change_sign = np.where(np.diff(np.sign(response1.yout[s0:s1, dofx])))[0]
    freq1 = 1 / (t[change_sign[2]] - t[change_sign[0]])
    abs_max1 = np.max(np.abs(response1.yout[s0:s1, dofx]))

    assert_allclose(freq, freq1, rtol=1e-3)
    assert_allclose(abs_max, abs_max1, rtol=1e-3)


def test_for_var_speed_1(rotor1):
    # Test for variable speed and bearings with fixed coefficients
    t, probe_params = simulation_parameters()
    speed = np.linspace(50, 500, len(t))

    F = unbalance_force(rotor1, speed, t)

    # Running with direct method
    resp_common = rotor1.run_time_response(speed, F, t)

    # Running with pseudo-modal method
    resp_pseudo_modal = rotor1.run_time_response(
        speed, F, t, model_reduction={"num_modes": 24}
    )

    # Running with pseudo-modal method and add_to_RHS callable function
    def unb_force(step, **curr_state):
        return F[step, :]

    resp_add_to_RHS = rotor1.run_time_response(
        speed,
        np.zeros((len(t), rotor1.ndof)),
        t,
        add_to_RHS=unb_force,
        model_reduction={"num_modes": 24},
        progress_interval=5,
    )

    s0 = probe_params["time"][0]
    s1 = probe_params["time"][1]
    dofx = probe_params["dofs"][0]
    dofy = probe_params["dofs"][1]

    assert_allclose(
        resp_common.yout[s0:s1, dofx], resp_pseudo_modal.yout[s0:s1, dofx], rtol=0.1
    )
    assert_allclose(
        resp_common.yout[s0:s1, dofy], resp_pseudo_modal.yout[s0:s1, dofy], rtol=0.1
    )

    assert_allclose(
        resp_pseudo_modal.yout[s0:s1, dofx], resp_add_to_RHS.yout[s0:s1, dofx]
    )
    assert_allclose(
        resp_pseudo_modal.yout[s0:s1, dofy], resp_add_to_RHS.yout[s0:s1, dofy]
    )

    freq = 68.96552
    abs_max = 1.52977e-4

    change_sign = np.where(np.diff(np.sign(resp_common.yout[s0:s1, dofx])))[0]
    freq1 = 1 / (t[change_sign[2]] - t[change_sign[0]])
    abs_max1 = np.max(np.abs(resp_common.yout[s0:s1, dofx]))

    assert_allclose(freq, freq1, rtol=1e-3)
    assert_allclose(abs_max, abs_max1, rtol=1e-3)


def test_for_var_speed_2(rotor2):
    # Test for variable speed and bearings with frequency dependent coefficients
    t, probe_params = simulation_parameters()
    speed = np.linspace(50, 500, len(t))

    F = unbalance_force(rotor2, speed, t)

    # Running with direct method
    resp_common = rotor2.run_time_response(speed, F, t)

    # Running with pseudo-modal method
    resp_pseudo_modal = rotor2.run_time_response(
        speed, F, t, model_reduction={"num_modes": 24}
    )

    # Running with pseudo-modal method and add_to_RHS callable function
    def unb_force(step, **curr_state):
        return F[step, :]

    resp_add_to_RHS = rotor2.run_time_response(
        speed,
        np.zeros((len(t), rotor2.ndof)),
        t,
        add_to_RHS=unb_force,
        model_reduction={"num_modes": 24},
        progress_interval=5,
    )

    s0 = probe_params["time"][0]
    s1 = probe_params["time"][1]
    dofx = probe_params["dofs"][0]
    dofy = probe_params["dofs"][1]

    assert_allclose(
        resp_common.yout[s0:s1, dofx], resp_pseudo_modal.yout[s0:s1, dofx], rtol=0.1
    )
    assert_allclose(
        resp_common.yout[s0:s1, dofy], resp_pseudo_modal.yout[s0:s1, dofy], rtol=0.1
    )

    assert_allclose(
        resp_pseudo_modal.yout[s0:s1, dofx], resp_add_to_RHS.yout[s0:s1, dofx]
    )
    assert_allclose(
        resp_pseudo_modal.yout[s0:s1, dofy], resp_add_to_RHS.yout[s0:s1, dofy]
    )

    freq = 68.96552
    abs_max = 1.53523e-4

    change_sign = np.where(np.diff(np.sign(resp_common.yout[s0:s1, dofx])))[0]
    freq1 = 1 / (t[change_sign[2]] - t[change_sign[0]])
    abs_max1 = np.max(np.abs(resp_common.yout[s0:s1, dofx]))

    assert_allclose(freq, freq1, rtol=1e-3)
    assert_allclose(abs_max, abs_max1, rtol=1e-3)
