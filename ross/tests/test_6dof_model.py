import pytest
from numpy.testing import assert_allclose
import numpy as np
import ross as rs


@pytest.fixture
def rotor_4dof():
    return rs.rotor_example()


@pytest.fixture
def rotor_6dof(rotor_4dof):
    shaft_elem = [
        rs.ShaftElement6DoF(
            material=rotor_4dof.shaft_elements[l].material,
            L=rotor_4dof.shaft_elements[l].L,
            n=rotor_4dof.shaft_elements[l].n,
            idl=rotor_4dof.shaft_elements[l].idl,
            odl=rotor_4dof.shaft_elements[l].odl,
            idr=rotor_4dof.shaft_elements[l].idr,
            odr=rotor_4dof.shaft_elements[l].odr,
        )
        for l, p in enumerate(rotor_4dof.shaft_elements)
    ]

    disk_elem = [
        rs.DiskElement6DoF(
            n=rotor_4dof.disk_elements[l].n,
            m=rotor_4dof.disk_elements[l].m,
            Id=rotor_4dof.disk_elements[l].Id,
            Ip=rotor_4dof.disk_elements[l].Ip,
        )
        for l in range(len(rotor_4dof.disk_elements))
    ]

    bearing_elem = [
        rs.BearingElement6DoF(
            n=rotor_4dof.bearing_elements[l].n,
            kxx=rotor_4dof.bearing_elements[l].kxx,
            kyy=rotor_4dof.bearing_elements[l].kyy,
            cxx=rotor_4dof.bearing_elements[l].cxx,
            cyy=rotor_4dof.bearing_elements[l].cyy,
            kxy=rotor_4dof.bearing_elements[l].kxy,
            kyx=rotor_4dof.bearing_elements[l].kyx,
            cxy=rotor_4dof.bearing_elements[l].cxy,
            cyx=rotor_4dof.bearing_elements[l].cyx,
            frequency=rotor_4dof.bearing_elements[l].frequency,
        )
        for l in range(len(rotor_4dof.bearing_elements))
    ]

    return rs.Rotor(shaft_elem, disk_elem, bearing_elem)


def test_run_static(rotor_6dof):
    static = rotor_6dof.run_static()

    bearing_forces = {"node_0": 432.3774305443053, "node_6": 432.3774305443162}
    disk_forces = {"node_2": 319.59116422953997, "node_4": 319.59116422953997}

    # fmt:off
    deformation = np.array([
        -4.32377431e-18, -3.73948262e-04, -6.48759479e-04, -7.45972747e-04,
        -6.48759479e-04, -3.73948262e-04, -4.32377431e-18
    ])
    # fmt:on

    for n in bearing_forces:
        assert_allclose(static.bearing_forces[n], bearing_forces[n], rtol=1e-6)

    for n in disk_forces:
        assert_allclose(static.disk_forces[n], disk_forces[n], rtol=1e-6)

    assert_allclose(static.deformation, deformation, rtol=1e-6)


def test_static_results_equality(rotor_6dof, rotor_4dof):
    static1 = rotor_6dof.run_static()
    static2 = rotor_4dof.run_static()

    for n in static1.bearing_forces:
        assert_allclose(static1.bearing_forces[n], static2.bearing_forces[n], rtol=1e-6)

    for n in static1.disk_forces:
        assert_allclose(static1.disk_forces[n], static2.disk_forces[n], rtol=1e-6)

    assert_allclose(static1.deformation, static2.deformation, rtol=1e-6)


def test_run_modal(rotor_6dof):
    speed = 100.0
    modal = rotor_6dof.run_modal(speed, num_modes=14)

    # fmt:off
    wn = np.array([
         91.78677 ,  96.29604 , 274.059892, 296.994125, 717.35166 ,
        770.203624, 774.349678
    ])
    wd = np.array([
         91.78677 ,  96.29604 , 274.059892, 296.994125, 717.35166 ,
        770.203624, 774.349678
    ])
    # fmt:on

    assert_allclose(modal.wn, wn, rtol=5e-2, atol=1)
    assert_allclose(modal.wd, wd, rtol=5e-2, atol=1)


def test_modal_results_equality(rotor_6dof, rotor_4dof):
    speed = 100.0
    modal_6dof = rotor_6dof.run_modal(speed, num_modes=10)
    modal_4dof = rotor_4dof.run_modal(speed, num_modes=10)

    assert_allclose(modal_6dof.wn, modal_4dof.wn, rtol=5e-2, atol=1)
    assert_allclose(modal_6dof.wd, modal_4dof.wd, rtol=5e-2, atol=1)


def test_campbell(rotor_6dof):
    speed_range = np.linspace(315, 1150, 31)
    campbell = rotor_6dof.run_campbell(speed_range)

    # fmt:off
    wd = np.array([
        91.70122235, 91.68341248, 91.66467645, 91.64459554, 91.62319309,
        91.60049282, 91.57652003, 91.55129967, 91.52485766, 91.49721995,
        91.46841277, 91.43846178, 91.40739326, 91.37523348, 91.34200733,
        91.30774144, 91.27245973, 91.23618811, 91.19894932, 91.1607676 ,
        91.12166632, 91.08166769, 91.04079376, 90.99906592, 90.95650489,
        90.91313073, 90.86896298, 90.82402031, 90.77832143, 90.73188463,
        90.68472497
    ])
    # fmt:on

    assert_allclose(campbell.wd[:, 0], wd, rtol=1e-3)


def test_campbell_equality(rotor_6dof, rotor_4dof):
    speed_range = np.linspace(315, 1150, 31)
    campbell1 = rotor_6dof.run_campbell(speed_range)
    campbell2 = rotor_4dof.run_campbell(speed_range)

    assert_allclose(campbell1.wd[:, 0], campbell2.wd[:, 0], rtol=1e-3)


def test_run_freq(rotor_6dof):
    speed_range = np.linspace(315, 1150, 31)
    response = rotor_6dof.run_freq_response(speed_range=speed_range)

    # fmt:off
    abs_resp = np.array([
        [2.48335649e-06, 8.84174113e-07],
        [8.84174113e-07, 1.07819686e-06]
    ])
    ang_resp = np.array([
        [3.14159265, 1.57079633],
        [1.57079633, 3.14159265]
    ])
    # fmt:on

    assert_allclose(abs(response.freq_resp[:2, :2, 0]), abs_resp, atol=1e-7)
    assert_allclose(abs(np.angle(response.freq_resp[:2, :2, 0])), ang_resp, atol=1e-7)


def test_freq_resp_equality(rotor_6dof, rotor_4dof):
    speed_range = np.linspace(315, 1150, 31)
    response1 = rotor_6dof.run_freq_response(speed_range=speed_range)
    response2 = rotor_4dof.run_freq_response(speed_range=speed_range)

    assert_allclose(
        abs(response1.freq_resp[:2, :2, :]),
        abs(response2.freq_resp[:2, :2, :]),
        atol=1e-7,
    )


def test_run_unb(rotor_6dof):
    speed = np.linspace(0, 100, 31)
    response = rotor_6dof.run_unbalance_response(
        node=3, unbalance_magnitude=10.0, unbalance_phase=0.0, frequency=speed
    )

    abs_resp = np.array([0.01763464, 0.02290302])
    ang_resp = np.array([1.18722483e-12, -1.57079633e00])

    assert_allclose(abs(response.forced_resp[:2, 15]), abs_resp, atol=1e-7)
    assert_allclose(np.angle(response.forced_resp[:2, 15]), ang_resp, atol=1e-7)


def test_unb_resp_equality(rotor_6dof, rotor_4dof):
    speed = np.linspace(0, 100, 31)
    response1 = rotor_6dof.run_unbalance_response(
        node=3, unbalance_magnitude=10.0, unbalance_phase=0.0, frequency=speed
    )
    response2 = rotor_4dof.run_unbalance_response(
        node=3, unbalance_magnitude=10.0, unbalance_phase=0.0, frequency=speed
    )

    assert_allclose(
        abs(response1.forced_resp[:2, :]), abs(response2.forced_resp[:2, :]), atol=1e-7
    )


def input_run_time(rotor):
    speed = 500.0
    size = 1000
    node = 3
    t = np.linspace(0, 10, size)
    F = np.zeros((size, rotor.ndof))
    F[:, rotor.number_dof * node + 0] = 10 * np.cos(2 * t)
    F[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)

    return speed, F, t


def test_run_time(rotor_6dof):
    input_6dof = input_run_time(rotor_6dof)

    response1 = rotor_6dof.run_time_response(*input_6dof)
    response2 = rotor_6dof.run_time_response(*input_6dof, integrator="newmark")
    # fmt:off
    response3 = np.array([
        4.23256631e-06,  7.70160885e-06,  0.00000000e+00, -2.57207762e-05,
        2.17306960e-05,  0.00000000e+00,  9.46767650e-06,  1.38792218e-05,
        0.00000000e+00, -2.25172698e-05,  1.91808098e-05,  0.00000000e+00,
        1.34302952e-05,  1.84580413e-05,  0.00000000e+00, -1.30644256e-05,
        1.17692246e-05,  0.00000000e+00,  1.49418357e-05,  2.01980866e-05,
        0.00000000e+00,  1.33928884e-17, -6.43108349e-17,  0.00000000e+00,
        1.34302952e-05,  1.84580413e-05,  0.00000000e+00,  1.30644256e-05,
       -1.17692246e-05,  0.00000000e+00,  9.46767650e-06,  1.38792218e-05,
        0.00000000e+00,  2.25172698e-05, -1.91808098e-05,  0.00000000e+00,
        4.23256631e-06,  7.70160885e-06,  0.00000000e+00,  2.57207762e-05,
       -2.17306960e-05,  0.00000000e+00
    ])
    # fmt:on

    dof = 3 * 6 + 1
    assert_allclose(
        np.mean(response1.yout[:, dof]), np.mean(response2.yout[:, dof]), atol=1e-7
    )
    assert_allclose(response1.yout[50, :], response3, atol=1e-7)


def test_time_resp_equality(rotor_6dof, rotor_4dof):
    input_6dof = input_run_time(rotor_6dof)
    input_4dof = input_run_time(rotor_4dof)

    response1 = rotor_6dof.run_time_response(*input_6dof)
    response2 = rotor_4dof.run_time_response(*input_4dof)

    dof1 = 3 * 6 + 1
    dof2 = 3 * 4 + 1
    assert_allclose(response1.yout[:, dof1], response2.yout[:, dof2], atol=1e-7)
