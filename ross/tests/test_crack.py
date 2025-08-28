import numpy as np
import pytest

import ross as rs
from ross import Q_, Probe, Crack

from numpy.testing import assert_allclose


@pytest.fixture
def rotor():
    steel2 = rs.Material(name="Steel", rho=7850, E=2.17e11, Poisson=0.2992610837438423)

    #  Rotor with internal damping, with 10 shaft elements, 2 disks and 2 bearings.
    i_d = 0
    o_d = 0.019
    n = 33

    # fmt: off
    L = np.array(
            [0  ,  25,  64, 104, 124, 143, 175, 207, 239, 271,
            303, 335, 345, 355, 380, 408, 436, 466, 496, 526,
            556, 586, 614, 647, 657, 667, 702, 737, 772, 807,
            842, 862, 881, 914]
            )/ 1000
    # fmt: on

    L = [L[i] - L[i - 1] for i in range(1, len(L))]

    shaft_elem = [
        rs.ShaftElement(
            material=steel2,
            L=l,
            idl=i_d,
            odl=o_d,
            idr=i_d,
            odr=o_d,
            alpha=8.0501,
            beta=1.0e-5,
            rotary_inertia=True,
            shear_effects=True,
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
        n=4, kxx=kxx1, kyy=kyy1, cxx=cxx1, cyy=cyy1, kzz=kzz, czz=czz
    )
    bearing1 = rs.BearingElement(
        n=31, kxx=kxx2, kyy=kyy2, cxx=cxx2, cyy=cyy2, kzz=kzz, czz=czz
    )

    return rs.Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def test_crack_mayes(rotor):
    crack = Crack(rotor, n=18, depth_ratio=0.2, crack_model="Mayes")

    assert crack.depth_ratio == 0.2
    assert crack.shaft_elem.n == 18
    assert round(crack.shaft_elem.L, 2) == 0.03

    Ko = np.array(
        [
            [3.27824895e08, 0.00000000e00, 0.00000000e00, -4.91737342e06],
            [0.00000000e00, 3.27824895e08, 4.91737342e06, 0.00000000e00],
            [0.00000000e00, 4.91737342e06, 1.20033082e05, 0.00000000e00],
            [-4.91737342e06, 0.00000000e00, 0.00000000e00, 1.20033082e05],
        ]
    )

    Kc = np.array(
        [
            [3.23937606e08, 1.83859799e07, 4.48801757e05, -4.82248470e06],
            [1.83859799e07, 2.69376932e08, 3.49065869e06, -4.48801757e05],
            [4.48801757e05, 3.49065869e06, 8.52069763e04, -1.09552506e04],
            [-4.82248470e06, -4.48801757e05, -1.09552506e04, 1.17716848e05],
        ]
    )

    assert_allclose(crack.Ko, Ko)
    assert_allclose(crack.Kc, Kc)


@pytest.fixture
def run_crack_mayes(rotor):
    n1 = rotor.disk_elements[0].n
    n2 = rotor.disk_elements[1].n

    unb_node = [n1, n2]
    unb_mag = [5e-4, 0]
    unb_phase = [-np.pi / 2, 0]

    results = rotor.run_crack(
        n=18,
        depth_ratio=0.2,
        node=unb_node,
        unbalance_magnitude=unb_mag,
        unbalance_phase=unb_phase,
        crack_model="Mayes",
        speed=125.66370614359172,
        t=np.arange(0, 0.5, 0.0001),
    )

    return results


@pytest.fixture
def run_crack_mayes_units(rotor):
    unb_node = [12, 24]
    unb_mag = Q_([0.043398083107259365, 0], "lb*in")
    unb_phase = Q_([-90.0, 0.0], "degrees")

    results = rotor.run_crack(
        n=18,
        node=unb_node,
        depth_ratio=0.2,
        unbalance_magnitude=unb_mag,
        speed=Q_(1200, "RPM"),
        unbalance_phase=unb_phase,
        crack_model="Mayes",
        t=np.arange(0, 0.5, 0.0001),
    )

    return results


@pytest.fixture
def run_crack_gasch(rotor):
    n1 = rotor.disk_elements[0].n
    n2 = rotor.disk_elements[1].n

    unb_node = [n1, n2]
    unb_mag = [5e-4, 0]
    unb_phase = [-np.pi / 2, 0]

    results = rotor.run_crack(
        depth_ratio=0.2,
        n=18,
        speed=125.66370614359172,
        t=np.arange(0, 0.5, 1e-4),
        node=unb_node,
        unbalance_magnitude=unb_mag,
        unbalance_phase=unb_phase,
        crack_model="Gasch",
    )

    return results


def test_crack_mayes_resp(run_crack_mayes):
    probe1 = Probe(12, Q_(45, "deg"))
    probe2 = Probe(20, Q_(90, "deg"))

    resp_prob1 = np.array(
        [0.000000e00, -2.214779e-08, -1.103084e-07, -2.855495e-07, -5.467894e-07]
    )
    resp_prob2 = np.array(
        [0.000000e00, -2.435100e-08, -1.211254e-07, -3.136489e-07, -6.050479e-07]
    )

    data = run_crack_mayes.data_time_response(probe=[probe1, probe2])
    assert_allclose(data["probe_resp[0]"].to_numpy()[:5], resp_prob1, rtol=1e-6)
    assert_allclose(data["probe_resp[1]"].to_numpy()[:5], resp_prob2, rtol=1e-6)


def test_crack_mayes_equality(run_crack_mayes, run_crack_mayes_units):
    probe1 = Probe(14, Q_(45, "deg"))
    probe2 = Probe(22, Q_(90, "deg"))

    data1 = run_crack_mayes.data_time_response(probe=[probe1, probe2])
    data2 = run_crack_mayes_units.data_time_response(probe=[probe1, probe2])

    assert_allclose(
        data1["probe_resp[0]"].to_numpy(), data2["probe_resp[0]"].to_numpy()
    )
    assert_allclose(
        data1["probe_resp[1]"].to_numpy(), data2["probe_resp[1]"].to_numpy()
    )


def test_crack_gasch_resp(run_crack_gasch):
    probe1 = Probe(3, Q_(45, "deg"))
    probe2 = Probe(24, Q_(90, "deg"))

    resp_prob1 = np.array(
        [0.000000e00, -1.719230e-08, -8.529809e-08, -2.191474e-07, -4.161626e-07]
    )
    resp_prob2 = np.array(
        [0.000000e00, -2.454247e-08, -1.228631e-07, -3.200085e-07, -6.162724e-07]
    )

    data = run_crack_gasch.data_time_response(probe=[probe1, probe2])
    assert_allclose(data["probe_resp[0]"].to_numpy()[:5], resp_prob1, rtol=1e-6)
    assert_allclose(data["probe_resp[1]"].to_numpy()[:5], resp_prob2, rtol=1e-6)
