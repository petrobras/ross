import numpy as np
import pytest
from numpy.testing import assert_allclose

import ross as rs
from ross import Q_, Probe, MisalignmentFlex, MisalignmentRigid

from numpy.testing import assert_allclose


@pytest.fixture
def rotor():
    steel2 = rs.Material(name="Steel", rho=7850, E=2.17e11, Poisson=0.2992610837438423)

    #  Rotor with 6 DoFs, with internal damping, with 10 shaft elements, 2 disks and 2 bearings.
    i_d = 0
    o_d = 0.019

    # fmt: off
    L = np.array([
        0  ,  25,  64, 104, 124, 143, 175, 207, 239, 271,
        303, 335, 345, 355, 380, 408, 436, 466, 496, 526,
        556, 586, 614, 647, 657, 667, 702, 737, 772, 807,
        842, 862, 881, 914
    ])/ 1000
    # fmt: on

    L = [L[i] - L[i - 1] for i in range(1, len(L))]

    shaft_elem = [
        rs.ShaftElement(
            l,
            i_d,
            o_d,
            material=steel2,
            alpha=8.0501,
            beta=1.0e-5,
        )
        for l in L
    ]

    Id = 0.003844540885417
    Ip = 0.007513248437500

    disk0 = rs.DiskElement(n=12, m=2.6375, Id=Id, Ip=Ip)
    disk1 = rs.DiskElement(n=24, m=2.6375, Id=Id, Ip=Ip)

    kxx1 = 4.40e5
    kyy1 = 4.6114e5
    kzz = 0
    cxx1 = 27.4
    cyy1 = 2.505
    czz = 0
    kxx2 = 9.50e5
    kyy2 = 1.09e8
    cxx2 = 50.4
    cyy2 = 100.4553

    bearing0 = rs.BearingElement(
        n=4, kxx=kxx1, kyy=kyy1, kzz=kzz, cxx=cxx1, cyy=cyy1, czz=czz
    )
    bearing1 = rs.BearingElement(
        n=31, kxx=kxx2, kyy=kyy2, kzz=kzz, cxx=cxx2, cyy=cyy2, czz=czz
    )

    return rs.Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


@pytest.fixture
def flex_params():
    return dict(
        n=0,
        radial_stiffness=40e3,
        bending_stiffness=38e3,
        mis_distance_x=2e-4,
        mis_distance_y=2e-4,
        mis_angle=5 * np.pi / 180,
        input_torque=0,
        load_torque=0,
    )


def test_mis_flex(rotor, flex_params):
    misalignment = MisalignmentFlex(rotor, mis_type="combined", **flex_params)

    assert misalignment.delta_x == 2e-4
    assert misalignment.delta_y == 2e-4
    assert misalignment.radial_stiffness == 40e3
    assert misalignment.bending_stiffness == 38e3
    assert round(misalignment.mis_angle, 3) == 0.087
    assert misalignment.shaft_elem.n == 0
    assert round(misalignment.shaft_elem.L, 3) == 0.025


@pytest.fixture
def run_mis_combined(rotor, flex_params):
    results = rotor.run_misalignment(
        coupling="flex",
        mis_type="combined",
        node=[12, 24],
        unbalance_magnitude=[5e-4, 0],
        unbalance_phase=Q_([-90.0, 0.0], "degrees"),
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
        num_modes=12,
        **flex_params,
    )

    return results


def test_mis_comb_resp(run_mis_combined):
    probe1 = Probe(12, Q_(45, "deg"))
    probe2 = Probe(20, Q_(90, "deg"))

    resp_prob1 = np.array(
        [
            0.00000000e00,
            -4.00480050e-09,
            -1.99806748e-08,
            -5.17934582e-08,
            -9.92341611e-08,
        ]
    )
    resp_prob2 = np.array(
        [
            0.00000000e00,
            -1.21720976e-09,
            -6.06026932e-09,
            -1.57142071e-08,
            -3.02555141e-08,
        ]
    )

    data = run_mis_combined.data_time_response(probe=[probe1, probe2])
    print(data["probe_resp[0]"].to_numpy()[:5])
    print(data["probe_resp[1]"].to_numpy()[:5])
    assert_allclose(data["probe_resp[0]"].to_numpy()[:5], resp_prob1)
    assert_allclose(data["probe_resp[1]"].to_numpy()[:5], resp_prob2)


@pytest.fixture
def run_mis_parallel(rotor, flex_params):
    results = rotor.run_misalignment(
        coupling="flex",
        mis_type="parallel",
        node=[12, 24],
        unbalance_magnitude=[5e-4, 0],
        unbalance_phase=Q_([-90.0, 0.0], "degrees"),
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
        num_modes=12,
        **flex_params,
    )

    return results


def test_mis_parallel_resp(run_mis_parallel):
    probe1 = Probe(12, Q_(45, "deg"))
    probe2 = Probe(20, Q_(90, "deg"))

    resp_prob1 = np.array(
        [
            0.00000000e00,
            -3.79800046e-09,
            -1.89450583e-08,
            -4.90968354e-08,
            -9.40439176e-08,
        ]
    )
    resp_prob2 = np.array(
        [
            0.00000000e00,
            -1.23987299e-09,
            -6.19980733e-09,
            -1.61619609e-08,
            -3.12891931e-08,
        ]
    )

    data = run_mis_parallel.data_time_response(probe=[probe1, probe2])
    print(data["probe_resp[0]"].to_numpy()[:5])
    print(data["probe_resp[1]"].to_numpy()[:5])
    assert_allclose(data["probe_resp[0]"].to_numpy()[:5], resp_prob1)
    assert_allclose(data["probe_resp[1]"].to_numpy()[:5], resp_prob2)


@pytest.fixture
def run_mis_angular(rotor, flex_params):
    results = rotor.run_misalignment(
        coupling="flex",
        mis_type="angular",
        node=[12, 24],
        unbalance_magnitude=[5e-4, 0],
        unbalance_phase=Q_([-90.0, 0.0], "degrees"),
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
        num_modes=12,
        **flex_params,
    )

    return results


def test_mis_angular_resp(run_mis_angular):
    probe1 = Probe(12, Q_(45, "deg"))
    probe2 = Probe(20, Q_(90, "deg"))

    resp_prob1 = np.array(
        [
            0.00000000e00,
            -4.46279262e-09,
            -2.22490352e-08,
            -5.76155767e-08,
            -1.10258226e-07,
        ]
    )
    resp_prob2 = np.array(
        [
            0.00000000e00,
            -1.58998969e-09,
            -7.94707043e-09,
            -2.06773631e-08,
            -3.98682773e-08,
        ]
    )

    data = run_mis_angular.data_time_response(probe=[probe1, probe2])
    print(data["probe_resp[0]"].to_numpy()[:5])
    print(data["probe_resp[1]"].to_numpy()[:5])
    assert_allclose(data["probe_resp[0]"].to_numpy()[:5], resp_prob1)
    assert_allclose(data["probe_resp[1]"].to_numpy()[:5], resp_prob2)


@pytest.fixture
def mis_rigid(rotor):
    results = rotor.run_misalignment(
        coupling="rigid",
        n=0,
        mis_distance=2e-4,
        input_torque=0,
        load_torque=0,
        node=[12, 24],
        unbalance_magnitude=[5e-4, 0],
        unbalance_phase=Q_([-90.0, 0.0], "degrees"),
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
        num_modes=12,
    )

    return results


def test_mis_rigid(rotor):
    misalignment = MisalignmentRigid(rotor, n=0, mis_distance=2e-4)

    assert misalignment.delta == 2e-4
    assert misalignment.shaft_elem.n == 0
    assert round(misalignment.shaft_elem.L, 3) == 0.025
    assert round(misalignment.phi, 3) == -0.017
    assert int(misalignment.kl1) == 469638975
    assert int(misalignment.kt1) == 42737
    assert int(misalignment.kl2) == 654160580
    assert int(misalignment.kt2) == 70133


def test_mis_rigid_resp(mis_rigid):
    probe1 = Probe(12, Q_(45, "deg"))
    probe2 = Probe(20, Q_(90, "deg"))

    resp_prob1 = np.array(
        [0.00000000e00, 6.15126513e-08, 3.71773834e-07, 1.18252510e-06, 2.74026919e-06]
    )
    resp_prob2 = np.array(
        [
            0.00000000e00,
            -1.99130062e-07,
            -1.18066869e-06,
            -3.67565279e-06,
            -8.32209309e-06,
        ]
    )

    data = mis_rigid.data_time_response(probe=[probe1, probe2])
    print(data["probe_resp[0]"].to_numpy()[:5])
    print(data["probe_resp[1]"].to_numpy()[:5])
    assert_allclose(data["probe_resp[0]"].to_numpy()[:5], resp_prob1)
    assert_allclose(data["probe_resp[1]"].to_numpy()[:5], resp_prob2)
