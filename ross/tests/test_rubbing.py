import numpy as np
import pytest

import ross as rs
from ross import Q_, Probe, Rubbing

from numpy.testing import assert_allclose


@pytest.fixture
def rotor():
    steel2 = rs.Material(name="Steel", rho=7850, E=2.17e11, Poisson=0.2992610837438423)

    #  Rotor with internal damping, with 10 shaft elements, 2 disks and 2 bearings.
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
def run_rubbing(rotor):
    n1 = rotor.disk_elements[0].n
    n2 = rotor.disk_elements[1].n

    unb_node = [n1, n2]
    unb_mag = [5e-4, 0]
    unb_phase = Q_([-90.0, 0.0], "degrees")

    results = rotor.run_rubbing(
        unbalance_magnitude=unb_mag,
        unbalance_phase=unb_phase,
        node=unb_node,
        n=12,
        distance=7.95e-5,
        contact_stiffness=1.1e6,
        contact_damping=40,
        friction_coeff=0.3,
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
        num_modes=12,
    )

    return results


def test_rubbing(rotor):
    rubbing = Rubbing(
        rotor,
        n=12,
        distance=7.95e-5,
        contact_stiffness=1.1e6,
        contact_damping=40,
        friction_coeff=0.3,
    )

    assert rubbing.delta == 7.95e-5
    assert rubbing.shaft_elem.n == 12
    assert round(rubbing.shaft_elem.L, 2) == 0.01


def test_rubbing_resp(run_rubbing):
    probe1 = Probe(12, Q_(45, "deg"))
    probe2 = Probe(20, Q_(90, "deg"))

    resp_prob1 = np.array(
        [
            0.00000000e00,
            -4.25599257e-09,
            -2.12134187e-08,
            -5.49189539e-08,
            -1.05067982e-07,
        ]
    )
    resp_prob2 = np.array(
        [
            0.00000000e00,
            -1.61265293e-09,
            -8.08660844e-09,
            -2.11251170e-08,
            -4.09019563e-08,
        ]
    )

    data = run_rubbing.data_time_response(probe=[probe1, probe2])
    assert_allclose(data["probe_resp[0]"].to_numpy()[:5], resp_prob1)
    assert_allclose(data["probe_resp[1]"].to_numpy()[:5], resp_prob2)
