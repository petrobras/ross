import os
from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

import ross as rs
from ross.faults.crack import Crack
from ross.units import Q_

steel2 = rs.Material(name="Steel", rho=7850, E=2.17e11, Poisson=0.2992610837438423)

#  Rotor with 6 DoFs, with internal damping, with 10 shaft elements, 2 disks and 2 bearings.
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
    rs.ShaftElement6DoF(
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

disk0 = rs.DiskElement6DoF(n=12, m=2.6375, Id=Id, Ip=Ip)
disk1 = rs.DiskElement6DoF(n=24, m=2.6375, Id=Id, Ip=Ip)

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

bearing0 = rs.BearingElement6DoF(
    n=4, kxx=kxx1, kyy=kyy1, cxx=cxx1, cyy=cyy1, kzz=kzz, czz=czz
)
bearing1 = rs.BearingElement6DoF(
    n=31, kxx=kxx2, kyy=kyy2, cxx=cxx2, cyy=cyy2, kzz=kzz, czz=czz
)

rotor = rs.Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


@pytest.fixture
def crack_mayes():
    unbalance_magnitudet = np.array([5e-4, 0])
    unbalance_phaset = np.array([-np.pi / 2, 0])

    crack = rotor.run_crack(
        dt=0.01,
        tI=0,
        tF=0.5,
        depth_ratio=0.2,
        n_crack=18,
        speed=125.66370614359172,
        unbalance_magnitude=unbalance_magnitudet,
        unbalance_phase=unbalance_phaset,
        crack_type="Mayes",
        print_progress=False,
    )

    return crack


@pytest.fixture
def crack_mayes_units():
    unbalance_magnitudet = Q_(np.array([0.043398083107259365, 0]), "lb*in")
    unbalance_phaset = Q_(np.array([-90.0, 0.0]), "degrees")

    crack = rotor.run_crack(
        dt=0.01,
        tI=0,
        tF=0.5,
        depth_ratio=0.2,
        n_crack=18,
        speed=Q_(1200, "RPM"),
        unbalance_magnitude=unbalance_magnitudet,
        unbalance_phase=unbalance_phaset,
        crack_type="Mayes",
        print_progress=False,
    )

    return crack


@pytest.fixture
def crack_gasch():
    unbalance_magnitudet = np.array([5e-4, 0])
    unbalance_phaset = np.array([-np.pi / 2, 0])

    crack = rotor.run_crack(
        dt=0.01,
        tI=0,
        tF=0.5,
        depth_ratio=0.2,
        n_crack=18,
        speed=125.66370614359172,
        unbalance_magnitude=unbalance_magnitudet,
        unbalance_phase=unbalance_phaset,
        crack_type="Gasch",
        print_progress=False,
    )

    return crack


def test_crack_mayes_parameters(crack_mayes):
    assert crack_mayes.dt == 0.01
    assert crack_mayes.tI == 0
    assert crack_mayes.tF == 0.5
    assert crack_mayes.depth_ratio == 0.2
    assert crack_mayes.n_crack == 18
    assert crack_mayes.speed == 125.66370614359172


def test_crack_mayes_parameters_units(crack_mayes_units):
    assert crack_mayes_units.dt == 0.01
    assert crack_mayes_units.tI == 0
    assert crack_mayes_units.tF == 0.5
    assert crack_mayes_units.depth_ratio == 0.2
    assert crack_mayes_units.n_crack == 18
    assert crack_mayes_units.speed == 125.66370614359172


def test_crack_mayes_parameters(crack_gasch):
    assert crack_gasch.dt == 0.01
    assert crack_gasch.tI == 0
    assert crack_gasch.tF == 0.5
    assert crack_gasch.depth_ratio == 0.2
    assert crack_gasch.n_crack == 18
    assert crack_gasch.speed == 125.66370614359172


def test_crack_mayes_kc(crack_mayes):
    assert crack_mayes.kc == pytest.approx(
        # fmt: off
        np.array([[ 3.30362822e+08,  1.90566722e+07, -4.61655368e+05,
         4.91833707e+06],
       [ 1.90566722e+07,  2.73811859e+08, -3.54836778e+06,
         4.61655368e+05],
       [-4.61655368e+05, -3.54836778e+06,  8.59606031e+04,
        -1.11837826e+04],
       [ 4.91833707e+06,  4.61655368e+05, -1.11837826e+04,
         1.19148647e+05]])
        # fmt: on
    )


def test_crack_mayes_kc_units(crack_mayes_units):
    assert crack_mayes_units.kc == pytest.approx(
        # fmt: off
        np.array([[ 3.30362822e+08,  1.90566722e+07, -4.61655368e+05,
         4.91833707e+06],
       [ 1.90566722e+07,  2.73811859e+08, -3.54836778e+06,
         4.61655368e+05],
       [-4.61655368e+05, -3.54836778e+06,  8.59606031e+04,
        -1.11837826e+04],
       [ 4.91833707e+06,  4.61655368e+05, -1.11837826e+04,
         1.19148647e+05]])
        # fmt: on
    )
