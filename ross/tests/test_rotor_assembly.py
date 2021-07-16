from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from ross.bearing_seal_element import *
from ross.disk_element import *
from ross.materials import steel
from ross.point_mass import *
from ross.rotor_assembly import *
from ross.shaft_element import *


@pytest.fixture
def rotor1():
    #  Rotor without damping with 2 shaft elements - no disks and no bearings
    le_ = 0.25
    i_d_ = 0
    o_d_ = 0.05

    tim0 = ShaftElement(
        le_,
        i_d_,
        o_d_,
        material=steel,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    )
    tim1 = ShaftElement(
        le_,
        i_d_,
        o_d_,
        material=steel,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    )

    shaft_elm = [tim0, tim1]
    return Rotor(shaft_elm, [], [])


def test_index_eigenvalues_rotor1(rotor1):
    evalues = np.array(
        [
            -3.8 + 68.6j,
            -3.8 - 68.6j,
            -1.8 + 30.0j,
            -1.8 - 30.0j,
            -0.7 + 14.4j,
            -0.7 - 14.4j,
        ]
    )
    evalues2 = np.array(
        [0.0 + 68.7j, 0.0 - 68.7j, 0.0 + 30.1j, 0.0 - 30.1j, -0.0 + 14.4j, -0.0 - 14.4j]
    )
    assert_almost_equal([4, 2, 0, 1, 3, 5], rotor1._index(evalues))
    assert_almost_equal([4, 2, 0, 1, 3, 5], rotor1._index(evalues2))


def test_mass_matrix_rotor1(rotor1):
    # fmt: off
    Mr1 = np.array([[ 1.421,  0.   ,  0.   ,  0.049,  0.496,  0.   ,  0.   , -0.031,  0.   ,  0.   ,  0.   ,  0.   ],
                    [ 0.   ,  1.421, -0.049,  0.   ,  0.   ,  0.496,  0.031,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
                    [ 0.   , -0.049,  0.002,  0.   ,  0.   , -0.031, -0.002,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
                    [ 0.049,  0.   ,  0.   ,  0.002,  0.031,  0.   ,  0.   , -0.002,  0.   ,  0.   ,  0.   ,  0.   ],
                    [ 0.496,  0.   ,  0.   ,  0.031,  2.841,  0.   ,  0.   ,  0.   ,  0.496,  0.   ,  0.   , -0.031],
                    [ 0.   ,  0.496, -0.031,  0.   ,  0.   ,  2.841,  0.   ,  0.   ,  0.   ,  0.496,  0.031,  0.   ],
                    [ 0.   ,  0.031, -0.002,  0.   ,  0.   ,  0.   ,  0.005,  0.   ,  0.   , -0.031, -0.002,  0.   ],
                    [-0.031,  0.   ,  0.   , -0.002,  0.   ,  0.   ,  0.   ,  0.005,  0.031,  0.   ,  0.   , -0.002],
                    [ 0.   ,  0.   ,  0.   ,  0.   ,  0.496,  0.   ,  0.   ,  0.031,  1.421,  0.   ,  0.   , -0.049],
                    [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.496, -0.031,  0.   ,  0.   ,  1.421,  0.049,  0.   ],
                    [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.031, -0.002,  0.   ,  0.   ,  0.049,  0.002,  0.   ],
                    [ 0.   ,  0.   ,  0.   ,  0.   , -0.031,  0.   ,  0.   , -0.002, -0.049,  0.   ,  0.   ,  0.002]])
    # fmt: on
    assert_almost_equal(rotor1.M(), Mr1, decimal=3)


def test_raise_if_element_outside_shaft():
    le_ = 0.25
    i_d_ = 0
    o_d_ = 0.05

    tim0 = ShaftElement(
        le_,
        i_d_,
        o_d_,
        material=steel,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    )
    tim1 = ShaftElement(
        le_,
        i_d_,
        o_d_,
        material=steel,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    )

    shaft_elm = [tim0, tim1]
    disk0 = DiskElement.from_geometry(3, steel, 0.07, 0.05, 0.28)
    stf = 1e6
    bearing0 = BearingElement(0, kxx=stf, cxx=0)
    bearing1 = BearingElement(3, kxx=stf, cxx=0)
    bearings = [bearing0, bearing1]

    with pytest.raises(ValueError) as excinfo:
        Rotor(shaft_elm, [disk0])
    assert "Trying to set disk or bearing outside shaft" == str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        Rotor(shaft_elm, bearing_elements=bearings)
    assert "Trying to set disk or bearing outside shaft" == str(excinfo.value)


@pytest.fixture
def rotor2():
    #  Rotor without damping with 2 shaft elements 1 disk and 2 bearings
    le_ = 0.25
    i_d_ = 0
    o_d_ = 0.05

    tim0 = ShaftElement(
        le_,
        i_d_,
        o_d_,
        material=steel,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    )
    tim1 = ShaftElement(
        le_,
        i_d_,
        o_d_,
        material=steel,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    )

    shaft_elm = [tim0, tim1]
    disk0 = DiskElement.from_geometry(1, steel, 0.07, 0.05, 0.28)
    stf = 1e6
    bearing0 = BearingElement(0, kxx=stf, cxx=0)
    bearing1 = BearingElement(2, kxx=stf, cxx=0)

    return Rotor(shaft_elm, [disk0], [bearing0, bearing1])


def test_mass_matrix_rotor2(rotor2):
    # fmt: off
    Mr2 = np.array([[  1.421,   0.   ,   0.   ,   0.049,   0.496,   0.   ,   0.   ,  -0.031,   0.   ,   0.   ,   0.   ,   0.   ],
                    [  0.   ,   1.421,  -0.049,   0.   ,   0.   ,   0.496,   0.031,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ],
                    [  0.   ,  -0.049,   0.002,   0.   ,   0.   ,  -0.031,  -0.002,   0.   ,   0.   ,   0.   ,   0.   ,   0.   ],
                    [  0.049,   0.   ,   0.   ,   0.002,   0.031,   0.   ,   0.   ,  -0.002,   0.   ,   0.   ,   0.   ,   0.   ],
                    [  0.496,   0.   ,   0.   ,   0.031,  35.431,   0.   ,   0.   ,   0.   ,   0.496,   0.   ,   0.   ,  -0.031],
                    [  0.   ,   0.496,  -0.031,   0.   ,   0.   ,  35.431,   0.   ,   0.   ,   0.   ,   0.496,   0.031,   0.   ],
                    [  0.   ,   0.031,  -0.002,   0.   ,   0.   ,   0.   ,   0.183,   0.   ,   0.   ,  -0.031,  -0.002,   0.   ],
                    [ -0.031,   0.   ,   0.   ,  -0.002,   0.   ,   0.   ,   0.   ,   0.183,   0.031,   0.   ,   0.   ,  -0.002],
                    [  0.   ,   0.   ,   0.   ,   0.   ,   0.496,   0.   ,   0.   ,   0.031,   1.421,   0.   ,   0.   ,  -0.049],
                    [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.496,  -0.031,   0.   ,   0.   ,   1.421,   0.049,   0.   ],
                    [  0.   ,   0.   ,   0.   ,   0.   ,   0.   ,   0.031,  -0.002,   0.   ,   0.   ,   0.049,   0.002,   0.   ],
                    [  0.   ,   0.   ,   0.   ,   0.   ,  -0.031,   0.   ,   0.   ,  -0.002,  -0.049,   0.   ,   0.   ,   0.002]])
    # fmt: on
    assert_almost_equal(rotor2.M(), Mr2, decimal=3)


def test_a0_0_matrix_rotor2(rotor2):
    # fmt: off
    A0_0 = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    # fmt: on
    assert_almost_equal(rotor2.A()[:12, :12], A0_0, decimal=3)


def test_a0_1_matrix_rotor2(rotor2):
    # fmt: off
    A0_1 = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    # fmt: on
    assert_almost_equal(rotor2.A()[:12, 12:24], A0_1, decimal=3)


def test_a1_0_matrix_rotor2(rotor2):
    # fmt: off
    A1_0 = np.array([[  20.63 ,   -0.   ,    0.   ,    4.114,  -20.958,    0.   ,    0.   ,    1.11 ,    0.056,   -0.   ,   -0.   ,   -0.014],
                     [   0.   ,   20.63 ,   -4.114,    0.   ,   -0.   ,  -20.958,   -1.11 ,    0.   ,   -0.   ,    0.056,    0.014,    0.   ],
                     [   0.   ,  697.351, -131.328,    0.   ,   -0.   , -705.253,  -44.535,    0.   ,   -0.   ,    2.079,    0.596,    0.   ],
                     [-697.351,    0.   ,   -0.   , -131.328,  705.253,   -0.   ,   -0.   ,  -44.535,   -2.079,    0.   ,    0.   ,    0.596],
                     [   0.442,    0.   ,   -0.   ,    0.072,   -0.887,   -0.   ,   -0.   ,   -0.   ,    0.442,    0.   ,    0.   ,   -0.072],
                     [   0.   ,    0.442,   -0.072,    0.   ,   -0.   ,   -0.887,    0.   ,    0.   ,    0.   ,    0.442,    0.072,   -0.   ],
                     [   0.   ,    6.457,   -0.837,    0.   ,   -0.   ,    0.   ,   -1.561,    0.   ,   -0.   ,   -6.457,   -0.837,   -0.   ],
                     [  -6.457,   -0.   ,    0.   ,   -0.837,    0.   ,    0.   ,    0.   ,   -1.561,    6.457,    0.   ,    0.   ,   -0.837],
                     [   0.056,   -0.   ,    0.   ,    0.014,  -20.958,    0.   ,    0.   ,   -1.11 ,   20.63 ,    0.   ,    0.   ,   -4.114],
                     [   0.   ,    0.056,   -0.014,    0.   ,   -0.   ,  -20.958,    1.11 ,    0.   ,    0.   ,   20.63 ,    4.114,   -0.   ],
                     [  -0.   ,   -2.079,    0.596,   -0.   ,    0.   ,  705.253,  -44.535,   -0.   ,   -0.   , -697.351, -131.328,    0.   ],
                     [   2.079,    0.   ,   -0.   ,    0.596, -705.253,   -0.   ,    0.   ,  -44.535,  697.351,    0.   ,    0.   , -131.328]])
    # fmt: on
    assert_almost_equal(rotor2.A()[12:24, :12] / 1e7, A1_0, decimal=3)


def test_a1_1_matrix_rotor2(rotor2):
    # fmt: off
    A1_1 = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    # fmt: on
    assert_almost_equal(rotor2.A()[12:24, 12:24] / 1e7, A1_1, decimal=3)


def test_evals_sorted_rotor2(rotor2):
    evals_sorted = np.array(
        [
            1.4667459679e-12 + 215.3707255735j,
            3.9623200168e-12 + 215.3707255733j,
            7.4569772223e-11 + 598.0247411492j,
            1.1024641658e-11 + 598.0247411456j,
            4.3188161105e-09 + 3956.2249777612j,
            2.5852376472e-11 + 3956.2249797838j,
            4.3188161105e-09 - 3956.2249777612j,
            2.5852376472e-11 - 3956.2249797838j,
            7.4569772223e-11 - 598.0247411492j,
            1.1024641658e-11 - 598.0247411456j,
            1.4667459679e-12 - 215.3707255735j,
            3.9623200168e-12 - 215.3707255733j,
        ]
    )

    evals_sorted_w_10000 = np.array(
        [
            -4.838034e-14 + 34.822138j,
            -5.045245e-01 + 215.369011j,
            5.045245e-01 + 215.369011j,
            8.482603e-08 + 3470.897616j,
            4.878990e-07 + 3850.212629j,
            4.176291e01 + 3990.22903j,
            4.176291e01 - 3990.22903j,
            4.878990e-07 - 3850.212629j,
            8.482603e-08 - 3470.897616j,
            5.045245e-01 - 215.369011j,
            -5.045245e-01 - 215.369011j,
            -4.838034e-14 - 34.822138j,
        ]
    )
    modal2_0 = rotor2.run_modal(speed=0)
    rotor2_evals, rotor2_evects = rotor2._eigen(speed=0)
    assert_allclose(rotor2_evals, evals_sorted, rtol=1e-3)
    assert_allclose(modal2_0.evalues, evals_sorted, rtol=1e-3)
    modal2_10000 = rotor2.run_modal(speed=10000)
    assert_allclose(modal2_10000.evalues, evals_sorted_w_10000, rtol=1e-1)


@pytest.fixture
def rotor3():
    #  Rotor without damping with 6 shaft elements 2 disks and 2 bearings
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
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.35)

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


@pytest.fixture
def rotor3_odd():
    #  Rotor without damping with odd number of shaft elements (7)
    #  2 disks and 2 bearings
    i_d = 0
    o_d = 0.05
    n = 7
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
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.35)

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def test_rotor_attributes(rotor1, rotor3, rotor3_odd):
    assert len(rotor1.nodes) == 3
    assert len(rotor1.nodes_i_d) == 3
    assert len(rotor1.nodes_o_d) == 3
    assert rotor1.L == 0.5
    assert rotor1.m_disks == 0
    assert rotor1.m_shaft == 7.6674495701675891
    assert rotor1.m == 7.6674495701675891
    assert rotor1.nodes_pos[0] == 0
    assert rotor1.nodes_pos[1] == 0.25
    assert rotor1.nodes_pos[-1] == 0.5
    assert len(rotor3.shaft_elements_length) == 6
    assert len(rotor3_odd.shaft_elements_length) == 7


def test_kappa_rotor3(rotor3):
    # TODO: Move this to test_results.py
    modal3_0 = rotor3.run_modal(speed=0)
    assert_allclose(modal3_0.kappa(0, 0)["Frequency"], 82.653037, rtol=1e-3)
    assert_allclose(modal3_0.kappa(0, 0)["Major axes"], 0.001454062985920231, rtol=1e-3)
    assert_allclose(
        modal3_0.kappa(0, 0)["Minor axes"], 2.0579515874459978e-11, rtol=1e-3, atol=1e-6
    )
    assert_allclose(
        modal3_0.kappa(0, 0)["kappa"], -1.415311171090584e-08, rtol=1e-3, atol=1e-6
    )

    modal3_2000 = rotor3.run_modal(speed=2000)
    assert_allclose(modal3_2000.kappa(0, 0)["Frequency"], 77.37957042, rtol=1e-3)
    assert_allclose(
        modal3_2000.kappa(0, 0)["Major axes"], 0.0011885396330204021, rtol=1e-3
    )
    assert_allclose(
        modal3_2000.kappa(0, 0)["Minor axes"], 0.0007308144427338161, rtol=1e-3
    )
    assert_allclose(modal3_2000.kappa(0, 0)["kappa"], -0.6148843693807821, rtol=1e-3)

    assert_allclose(modal3_2000.kappa(0, 1)["Frequency"], 88.98733511566752, rtol=1e-3)
    assert_allclose(
        modal3_2000.kappa(0, 1)["Major axes"], 0.0009947502339267566, rtol=1e-3
    )
    assert_allclose(
        modal3_2000.kappa(0, 1)["Minor axes"], 0.0008412470069506472, rtol=1e-3
    )
    assert_allclose(modal3_2000.kappa(0, 1)["kappa"], 0.8456866641084784, rtol=1e-3)

    assert_allclose(modal3_2000.kappa(1, 1)["Frequency"], 88.98733511566752, rtol=1e-3)
    assert_allclose(
        modal3_2000.kappa(1, 1)["Major axes"], 0.0018877975750108973, rtol=1e-3
    )
    assert_allclose(
        modal3_2000.kappa(1, 1)["Minor axes"], 0.0014343257484060105, rtol=1e-3
    )
    assert_allclose(modal3_2000.kappa(1, 1)["kappa"], 0.7597878964314968, rtol=1e-3)


def test_kappa_mode_rotor3(rotor3):
    modal3_2000 = rotor3.run_modal(2000)
    assert_allclose(
        modal3_2000.kappa_mode(0),
        [-0.614884, -0.696056, -0.723983, -0.729245, -0.708471, -0.656976, -0.513044],
        rtol=1e-3,
    )

    assert_allclose(
        modal3_2000.kappa_mode(1),
        [0.845687, 0.759788, 0.734308, 0.737393, 0.778295, 0.860137, 0.948157],
        rtol=1e-3,
    )


@pytest.fixture
def rotor4():
    #  Rotor without damping with 6 shaft elements 2 disks and 2 bearings
    #  Same as rotor3, but constructed with sections.
    i_d = 0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    n0 = len(L) // 2
    n1 = len(L) // 2
    L0 = sum(L[:n0])
    L1 = sum(L[n1:])
    sec0 = ShaftElement.section(L0, n0, i_d, o_d, material=steel)
    sec1 = ShaftElement.section(L1, n1, i_d, o_d, material=steel)

    shaft_elem = [sec0, sec1]

    disk0 = DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.35)

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def test_evals_rotor3_rotor4(rotor3, rotor4):
    rotor3_evals, rotor3_evects = rotor3._eigen(speed=0)
    rotor4_evals, rotor4_evects = rotor4._eigen(speed=0)

    assert_allclose(rotor3_evals, rotor4_evals, rtol=1e-3)


def test_campbell(rotor4):
    speed = np.linspace(0, 300, 3)
    camp = rotor4.run_campbell(speed)

    camp_calculated = camp.wd
    # fmt: off
    camp_desired = np.array([[82.65303734,  86.65811435, 254.52047828, 274.31285391, 679.48903239, 716.78631221],
                             [82.60929602,  86.68625235, 251.70037114, 276.87787937, 652.85679897, 742.60864608],
                             [82.48132723,  86.76734307, 245.49092844, 282.33294699, 614.05536277, 779.07778334]])
    # fmt: on
    assert_allclose(camp_calculated, camp_desired)


@pytest.mark.skip(reason="Needs investigation. It fails depending on system.")
def test_freq_response(rotor4):
    magdb_exp = np.array(
        [
            [
                [-120.0, -120.86944548, -115.66348242, -125.09053613],
                [-363.3527912, -151.34622928, -119.93523136, -131.80470016],
                [-354.61148814, -160.12580074, -126.60092157, -129.32321566],
                [-123.52182518, -115.10369362, -117.98804019, -114.71703185],
            ],
            [
                [-372.78089953, -151.34622928, -119.93523136, -131.80470016],
                [-118.06179974, -118.64323754, -120.32380413, -123.40124075],
                [-121.58362492, -113.542778, -118.87061678, -113.43742325],
                [-359.49336181, -167.93504252, -134.49524237, -130.94575121],
            ],
            [
                [-373.93814241, -160.12580074, -126.60092157, -129.32321566],
                [-121.58362492, -113.542778, -118.87061678, -113.43742325],
                [-101.07120376, -105.55913457, -104.14094712, -102.44191459],
                [-370.75325104, -173.53567801, -139.05415269, -127.7170584],
            ],
            [
                [-123.52182518, -115.10369362, -117.98804019, -114.71703185],
                [-362.65206982, -167.93504252, -134.49524237, -130.94575121],
                [-350.39254778, -173.53567801, -139.05415269, -127.7170584],
                [-101.29234967, -106.9521567, -104.66576262, -103.46014727],
            ],
        ]
    )

    magdb_exp_modes_4 = np.array(
        [
            [
                [-186.09498071, -141.31217447, -156.3727046, -164.2331948],
                [-343.18648319, -177.48024148, -185.20860324, -186.64998732],
                [-334.7831122, -177.53606335, -187.59501345, -184.89095401],
                [-153.70571976, -128.91233707, -141.534854, -146.49160424],
            ],
            [
                [-359.4389246, -177.48024148, -185.20860324, -186.64998732],
                [-122.88901214, -139.43588496, -154.12804564, -161.85419832],
                [-124.04894039, -128.97278476, -141.32571597, -146.31133247],
                [-347.60421616, -175.0690129, -185.41011193, -182.54955925],
            ],
            [
                [-350.13764012, -177.53606335, -187.59501345, -184.89095401],
                [-124.04894039, -128.97278476, -141.32571597, -146.31133247],
                [-111.60564526, -122.9491126, -123.76248808, -122.27201722],
                [-337.19738844, -162.11699607, -159.52366304, -159.38118889],
            ],
            [
                [-153.70571976, -128.91233707, -141.534854, -146.49160424],
                [-333.15975187, -175.0690129, -185.41011193, -182.54955925],
                [-323.43173195, -162.11699607, -159.52366304, -159.38118889],
                [-121.31645881, -120.44617713, -124.36604496, -122.47735964],
            ],
        ]
    )

    omega = np.linspace(0.0, 450.0, 4)
    freq_resp = rotor4.run_freq_response(speed_range=omega)
    magdb = 20.0 * np.log10(freq_resp.magnitude)
    assert_allclose(magdb[:4, :4, :4], magdb_exp)

    freq_resp = rotor4.run_freq_response(speed_range=omega, modes=list(range(4)))
    magdb = 20.0 * np.log10(freq_resp.magnitude)
    assert_allclose(magdb[:4, :4, :4], magdb_exp_modes_4)


def test_freq_response_w_force(rotor4):
    # modal4 = rotor4.run_modal(0)
    F0 = np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 22.5 + 0.0j, 90.0 + 0.0j, 202.5 + 0.0j],
            [0.0 + 0.0j, 0.0 - 22.5j, 0.0 - 90.0j, 0.0 - 202.5j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )
    mag_exp = np.array(
        [
            [0.00000000e00, 1.14259057e-06, 1.88932819e-04, 4.50376020e-05],
            [0.00000000e00, 3.02252319e-06, 1.50551126e-04, 4.98323245e-05],
            [0.00000000e00, 1.97842812e-05, 5.19405022e-05, 2.80824236e-05],
            [0.00000000e00, 2.02593969e-05, 1.64498124e-05, 1.06100461e-05],
        ]
    )
    mag_exp_2_unb = np.array(
        [
            [0.00000000e00, 4.80337594e-06, 2.31170438e-04, 6.90062268e-05],
            [0.00000000e00, 3.15307288e-06, 1.87793923e-04, 8.08531462e-05],
            [0.00000000e00, 3.79692673e-05, 5.97050225e-05, 5.48105215e-05],
            [0.00000000e00, 4.16812885e-05, 1.38592416e-05, 2.20209089e-05],
        ]
    )

    omega = np.linspace(0.0, 450.0, 4)
    freq_resp = rotor4.run_forced_response(force=F0, speed_range=omega)
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[:4, :4], mag_exp)

    freq_resp = rotor4.run_unbalance_response(2, 0.001, 0, frequency=omega)
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[:4, :4], mag_exp)

    freq_resp = rotor4.run_unbalance_response(2, 0.001, 0, frequency=omega)
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[:4, :4], mag_exp)

    freq_resp = rotor4.run_unbalance_response(
        [2, 3], [0.001, 0.001], [0.0, 0], frequency=omega
    )
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[:4, :4], mag_exp_2_unb)


def test_mesh_convergence(rotor3):
    rotor3.convergence(n_eigval=0, err_max=1e-08)
    modal3 = rotor3.run_modal(speed=0)

    assert_allclose(len(rotor3.shaft_elements), 96, atol=0)
    assert_allclose(modal3.wn[0], 82.653037335, atol=1e-02)
    assert_allclose(rotor3.shaft_elements[0].L, 0.015625, atol=1e-06)
    assert_allclose(rotor3.disk_elements[0].n, 32, atol=0)
    assert_allclose(rotor3.disk_elements[1].n, 64, atol=0)
    assert_allclose(rotor3.bearing_elements[0].n, 0, atol=0)
    assert_allclose(rotor3.bearing_elements[1].n, 96, atol=0)
    assert rotor3.error_arr[-1] <= 1e-08 * 100


def test_static_analysis_rotor3(rotor3):
    static = rotor3.run_static()

    assert_almost_equal(
        static.deformation[0],
        np.array(
            [
                -4.94274533e-12,
                -4.51249085e-04,
                -7.88420867e-04,
                -9.18114192e-04,
                -8.08560219e-04,
                -4.68788888e-04,
                -5.56171636e-12,
            ]
        ),
        decimal=6,
    )
    assert_almost_equal(
        static.Vx[0],
        np.array(
            [
                -494.2745,
                -456.6791,
                -456.6791,
                -419.0837,
                -99.4925,
                -61.8971,
                -61.8971,
                -24.3017,
                480.9808,
                518.5762,
                518.5762,
                556.1716,
            ]
        ),
        decimal=3,
    )
    assert_almost_equal(
        static.Bm[0],
        np.array(
            [
                0.0,
                -118.8692,
                -118.8692,
                -228.3396,
                -228.3396,
                -248.5133,
                -248.5133,
                -259.2881,
                -259.2881,
                -134.3435,
                -134.3435,
                0.0,
            ]
        ),
        decimal=3,
    )


@pytest.fixture
def rotor5():
    #  Rotor without damping with 10 shaft elements 2 disks and 2 bearings
    i_d = 0
    o_d = 0.05
    n = 10
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
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement.from_geometry(6, steel, 0.07, 0.05, 0.35)

    stfx = 1e6
    stfy = 1e6
    bearing0 = BearingElement(2, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(8, kxx=stfx, kyy=stfy, cxx=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def test_static_analysis_rotor5(rotor5):
    static = rotor5.run_static()

    assert_almost_equal(
        static.deformation[0],
        np.array(
            [
                8.12651626e-04,
                4.08939282e-04,
                -5.69465378e-12,
                -4.05876595e-04,
                -7.15824882e-04,
                -8.36443708e-04,
                -7.35964234e-04,
                -4.23416398e-04,
                -6.31362481e-12,
                4.28859620e-04,
                8.52492302e-04,
            ]
        ),
        decimal=6,
    )
    assert_almost_equal(
        static.Vx[0],
        np.array(
            [
                0.0,
                37.5954,
                37.5954,
                75.1908,
                -494.2745,
                -456.6791,
                -456.6791,
                -419.0837,
                -99.4925,
                -61.8971,
                -61.8971,
                -24.3017,
                480.9808,
                518.5762,
                518.5762,
                556.1716,
                -75.1908,
                -37.5954,
                -37.5954,
                -0.0,
            ]
        ),
        decimal=3,
    )
    assert_almost_equal(
        static.Bm[0],
        np.array(
            [
                0.0,
                4.6994,
                4.6994,
                18.7977,
                18.7977,
                -100.0715,
                -100.0715,
                -209.5418,
                -209.5418,
                -229.7155,
                -229.7155,
                -240.4904,
                -240.4904,
                -115.5458,
                -115.5458,
                18.7977,
                18.7977,
                4.6994,
                4.6994,
                0.0,
            ]
        ),
        decimal=3,
    )


@pytest.fixture
def rotor6():
    #  Overhung rotor without damping with 10 shaft elements
    #  2 disks and 2 bearings
    i_d = 0
    o_d = 0.05
    n = 10
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
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(5, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement.from_geometry(10, steel, 0.07, 0.05, 0.35)

    stfx = 1e6
    stfy = 1e6
    bearing0 = BearingElement(2, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(8, kxx=stfx, kyy=stfy, cxx=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def test_static_analysis_rotor6(rotor6):
    static = rotor6.run_static()

    assert_almost_equal(
        static.deformation[0],
        np.array(
            [
                -1.03951876e-04,
                -4.93624668e-05,
                -1.79345202e-12,
                3.74213098e-05,
                7.66066703e-05,
                1.29084322e-04,
                1.85016673e-04,
                1.72933811e-04,
                -1.02148266e-11,
                -3.96409257e-04,
                -9.20006704e-04,
            ]
        ),
        decimal=6,
    )
    assert_almost_equal(
        static.Vx[0],
        np.array(
            [
                -0.0,
                37.5954,
                37.5954,
                75.1908,
                -104.1544,
                -66.5589,
                -66.5589,
                -28.9635,
                -28.9635,
                8.6319,
                328.2231,
                365.8185,
                365.8185,
                403.4139,
                403.4139,
                441.0093,
                -580.4733,
                -542.8779,
                -542.8779,
                -505.2825,
            ]
        ),
        decimal=3,
    )
    assert_almost_equal(
        static.Bm[0],
        np.array(
            [
                0.0,
                4.6994,
                4.6994,
                18.7977,
                18.7977,
                -2.5415,
                -2.5415,
                -14.4818,
                -14.4818,
                -17.0232,
                -17.0232,
                69.732,
                69.732,
                165.886,
                165.886,
                271.439,
                271.439,
                131.0201,
                131.02,
                0.0,
            ]
        ),
        decimal=3,
    )


def test_run_critical_speed(rotor5, rotor6):
    results5 = rotor5.run_critical_speed(num_modes=12, rtol=0.005)
    results6 = rotor6.run_critical_speed(num_modes=12, rtol=0.005)

    wn5 = np.array(
        [
            86.10505193,
            86.60492546,
            198.93259257,
            207.97165539,
            244.95609413,
            250.53522782,
        ]
    )
    wd5 = np.array(
        [
            86.1050519,
            86.60492544,
            198.93259256,
            207.97165539,
            244.95609413,
            250.53522782,
        ]
    )
    log_dec5 = np.zeros_like(wd5)
    damping_ratio5 = np.zeros_like(wd5)

    wd6 = np.array(
        [
            61.52110644,
            63.72862198,
            117.49491374,
            118.55829416,
            233.83724523,
            236.40346235,
        ]
    )
    wn6 = np.array(
        [
            61.52110644,
            63.72862198,
            117.49491375,
            118.55829421,
            233.83724523,
            236.40346458,
        ]
    )
    log_dec6 = np.zeros_like(wd6)
    damping_ratio6 = np.zeros_like(wd6)

    assert_almost_equal(results5._wd, wd5, decimal=4)
    assert_almost_equal(results5._wn, wn5, decimal=4)
    assert_almost_equal(results5.log_dec, log_dec5, decimal=4)
    assert_almost_equal(results5.damping_ratio, damping_ratio5, decimal=4)

    assert_almost_equal(results6._wd, wd6, decimal=4)
    assert_almost_equal(results6._wn, wn6, decimal=4)
    assert_almost_equal(results6.log_dec, log_dec6, decimal=4)
    assert_almost_equal(results6.damping_ratio, damping_ratio6, decimal=4)


@pytest.fixture
def coaxrotor():
    #  Co-axial rotor system with 2 shafts, 4 disks and
    #  4 bearings (3 to ground and 1 to body)
    i_d = 0
    o_d = 0.05
    n = 10
    L = [0.25 for _ in range(n)]

    axial_shaft = [ShaftElement(l, i_d, o_d, material=steel) for l in L]

    i_d = 0.25
    o_d = 0.30
    n = 6
    L = [0.25 for _ in range(n)]

    coaxial_shaft = [ShaftElement(l, i_d, o_d, material=steel) for l in L]

    disk0 = DiskElement.from_geometry(
        n=1, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=9, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk2 = DiskElement.from_geometry(
        n=13, material=steel, width=0.07, i_d=0.20, o_d=0.48
    )
    disk3 = DiskElement.from_geometry(
        n=15, material=steel, width=0.07, i_d=0.20, o_d=0.48
    )

    shaft = [axial_shaft, coaxial_shaft]
    disks = [disk0, disk1, disk2, disk3]

    stfx = 1e6
    stfy = 1e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(10, kxx=stfx, kyy=stfy, cxx=0)
    bearing2 = BearingElement(11, kxx=stfx, kyy=stfy, cxx=0)
    bearing3 = BearingElement(8, n_link=17, kxx=stfx, kyy=stfy, cxx=0)
    bearings = [bearing0, bearing1, bearing2, bearing3]

    return CoAxialRotor(shaft, disks, bearings)


def test_coaxial_rotor_assembly(coaxrotor):
    # fmt: off
    assert list(coaxrotor.df["shaft_number"]) == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    ]
    assert coaxrotor.nodes_pos == [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
        0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
    ]
    assert list(coaxrotor.df_shaft["nodes_pos_l"]) == [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
        0.5, 0.75, 1.0, 1.25, 1.5, 1.75
    ]
    assert list(coaxrotor.df_shaft["nodes_pos_r"]) == [
        0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
        0.75, 1.0, 1.25, 1.5, 1.75, 2.0
    ]
    assert list(coaxrotor.df["y_pos"].dropna()) == [
        0.025, 0.05, 0.025, 0.05, 0.025, 0.15, 0.3, 0.3
    ]
    assert list(np.round(coaxrotor.df["y_pos_sup"].dropna(), 3)) == [
        0.319, 0.125, 0.319, 0.444
    ]
    # fmt: on


def test_from_section():
    #  Rotor built from classmethod from_section
    #  2 disks and 2 bearings
    leng_data = [0.5, 1.0, 2.0, 1.0, 0.5]
    leng_data_error = [0.5, 1.0, 2.0, 1.0]

    odl_data = [0.1, 0.2, 0.3, 0.2, 0.1]
    odr_data_error = [0.1, 0.2, 0.3, 0.2]

    idl_data = [0, 0, 0, 0, 0]
    material = steel
    material_error = [steel, steel]
    disk_data = [
        DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28),
        DiskElement.from_geometry(n=3, material=steel, width=0.07, i_d=0.05, o_d=0.35),
    ]
    brg_seal_data = [
        BearingElement(n=0, kxx=1e6, cxx=0, kyy=1e6, cyy=0),
        BearingElement(n=5, kxx=1e6, cxx=0, kyy=1e6, cyy=0),
    ]

    rotor1 = Rotor.from_section(
        leng_data=leng_data,
        idl_data=idl_data,
        odl_data=odl_data,
        material_data=material,
        disk_data=disk_data,
        brg_seal_data=brg_seal_data,
        nel_r=4,
    )

    assert_allclose(len(rotor1.shaft_elements), 20, atol=0)
    assert_allclose(rotor1.shaft_elements[0].L, 0.125, atol=0)
    assert_allclose(rotor1.shaft_elements[4].L, 0.25, atol=0)
    assert_allclose(rotor1.shaft_elements[8].L, 0.5, atol=0)
    assert_allclose(rotor1.shaft_elements[12].L, 0.25, atol=0)
    assert_allclose(rotor1.shaft_elements[16].L, 0.125, atol=0)
    assert_allclose(rotor1.disk_elements[0].n, 8, atol=0)
    assert_allclose(rotor1.disk_elements[1].n, 12, atol=0)
    assert_allclose(rotor1.bearing_elements[0].n, 0, atol=0)
    assert_allclose(rotor1.bearing_elements[1].n, 20, atol=0)

    with pytest.raises(ValueError) as excinfo:
        Rotor.from_section(
            leng_data=leng_data_error,
            idl_data=idl_data,
            odl_data=odl_data,
            material_data=material,
            disk_data=disk_data,
            brg_seal_data=brg_seal_data,
            nel_r=4,
        )
    assert "The lists size do not match (leng_data, odl_data and idl_data)." == str(
        excinfo.value
    )

    with pytest.raises(ValueError) as excinfo:
        Rotor.from_section(
            leng_data=leng_data,
            idl_data=idl_data,
            odl_data=odl_data,
            odr_data=odr_data_error,
            material_data=material,
            disk_data=disk_data,
            brg_seal_data=brg_seal_data,
            nel_r=4,
        )
    assert "The lists size do not match (leng_data, odr_data and idr_data)." == str(
        excinfo.value
    )

    with pytest.raises(AttributeError) as excinfo:
        Rotor.from_section(
            leng_data=leng_data,
            idl_data=idl_data,
            odl_data=odl_data,
            material_data=None,
            disk_data=disk_data,
            brg_seal_data=brg_seal_data,
            nel_r=4,
        )
    assert "Please define a material or a list of materials" == str(excinfo.value)

    with pytest.raises(IndexError) as excinfo:
        Rotor.from_section(
            leng_data=leng_data,
            idl_data=idl_data,
            odl_data=odl_data,
            material_data=material_error,
            disk_data=disk_data,
            brg_seal_data=brg_seal_data,
            nel_r=4,
        )
    assert "material_data size does not match size of other lists" == str(excinfo.value)


@pytest.fixture
def rotor7():
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
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.35)

    stfx = 1e6
    stfy = 1e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=1e3, cyy=1e3)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=1e3, cyy=1e3)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def test_whirl_values(rotor7):
    speed_range = np.linspace(50, 500, 10)
    for speed in speed_range:
        modal7 = rotor7.run_modal(speed)
        assert_allclose(modal7.whirl_values(), [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], atol=0)
        assert_equal(
            modal7.whirl_direction(),
            np.array(
                ["Backward", "Forward", "Backward", "Forward", "Backward", "Forward"],
                dtype="<U8",
            ),
        )


def test_kappa_mode(rotor7):
    modal7 = rotor7.run_modal(100.0)
    assert_allclose(
        modal7.kappa_mode(0),
        [
            -0.999999999989335,
            -0.9999999999893868,
            -0.9999999999894262,
            -0.9999999999894176,
            -0.9999999999893875,
            -0.9999999999893159,
            -0.9999999999891641,
        ],
        rtol=1e-7,
    )
    assert_allclose(
        modal7.kappa_mode(1),
        [
            0.9999999999926857,
            0.9999999999925702,
            0.9999999999925301,
            0.9999999999924822,
            0.9999999999924919,
            0.9999999999925321,
            0.9999999999926336,
        ],
        rtol=1e-7,
    )
    assert_allclose(
        modal7.kappa_mode(2),
        [
            -0.9999999999586078,
            -0.9999999999590045,
            -0.9999999999595016,
            -0.9999999999682825,
            -0.9999999999577597,
            -0.999999999961294,
            -0.9999999999628185,
        ],
        rtol=1e-7,
    )

    modal7 = rotor7.run_modal(speed=250.0)
    assert_allclose(
        modal7.kappa_mode(0),
        [
            -0.9999999999996795,
            -0.9999999999997023,
            -0.9999999999997117,
            -0.9999999999997297,
            -0.9999999999997392,
            -0.9999999999997269,
            -0.9999999999997203,
        ],
        rtol=1e-7,
    )
    assert_allclose(
        modal7.kappa_mode(1),
        [
            0.9999999999992075,
            0.999999999999222,
            0.9999999999992263,
            0.9999999999992275,
            0.9999999999992394,
            0.9999999999992564,
            0.9999999999992875,
        ],
        rtol=1e-7,
    )
    assert_allclose(
        modal7.kappa_mode(2),
        [
            -0.9999999999955613,
            -0.999999999995006,
            -0.9999999999949597,
            -0.9999999999897796,
            -0.999999999996037,
            -0.9999999999966488,
            -0.9999999999969151,
        ],
        rtol=1e-7,
    )

    modal7 = rotor7.run_modal(500.0)
    assert_allclose(
        modal7.kappa_mode(0),
        [
            -0.9999999999986061,
            -0.999999999998796,
            -0.9999999999988834,
            -0.9999999999989619,
            -0.999999999998994,
            -0.9999999999989716,
            -0.9999999999989015,
        ],
        rtol=1e-7,
    )
    assert_allclose(
        modal7.kappa_mode(1),
        [
            0.9999999999995656,
            0.9999999999993939,
            0.9999999999993113,
            0.9999999999992302,
            0.999999999999194,
            0.9999999999992081,
            0.9999999999992395,
        ],
        rtol=1e-7,
    )
    assert_allclose(
        modal7.kappa_mode(2),
        [
            -0.999999999997584,
            -0.9999999999976369,
            -0.9999999999979048,
            -0.9999999999986678,
            -0.9999999999977003,
            -0.9999999999983235,
            -0.9999999999986461,
        ],
        rtol=1e-7,
    )


def test_kappa_axes_values(rotor7):
    modal7 = rotor7.run_modal(50)
    assert_allclose(modal7.kappa(3, 0)["Minor axes"], 0.0024460977827471028, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Minor axes"], 0.0024415401094917922, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Minor axes"], 7.753006465896838e-05, atol=1e-8)
    assert_allclose(modal7.kappa(3, 0)["Major axes"], 0.0024460977827550083, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Major axes"], 0.0024415401094980776, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Major axes"], 7.753006466024783e-05, atol=1e-8)

    modal7 = rotor7.run_modal(200)
    assert_allclose(modal7.kappa(3, 0)["Minor axes"], 0.002453197790184042, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Minor axes"], 0.0024349531472631354, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Minor axes"], 8.081580235887124e-05, atol=1e-8)
    assert_allclose(modal7.kappa(3, 0)["Major axes"], 0.002453197790191339, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Major axes"], 0.0024349531472711047, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Major axes"], 8.081580235956821e-05, atol=1e-8)

    modal7 = rotor7.run_modal(400)
    assert_allclose(modal7.kappa(3, 0)["Minor axes"], 0.002463187671800876, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Minor axes"], 0.0024266089747119572, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Minor axes"], 8.480305842194371e-05, atol=1e-8)
    assert_allclose(modal7.kappa(3, 0)["Major axes"], 0.002463187671801488, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Major axes"], 0.0024266089747121845, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Major axes"], 8.480305842205874e-05, atol=1e-8)


@pytest.mark.skip(reason="Fails for very small values")
def test_H_kappa(rotor7):
    rotor7.w = 400
    assert_allclose(
        rotor7.H_kappa(3, 0),
        [[6.06729351e-06, -6.33478357e-19], [-6.33478357e-19, 6.06729351e-06]],
        rtol=1e-2,
    )
    assert_allclose(
        rotor7.H_kappa(3, 0),
        [[5.88843112e-06, 2.88604638e-20], [2.88604638e-20, 5.88843112e-06]],
        rtol=1e-2,
    )
    assert_allclose(
        rotor7.H_kappa(3, 0),
        [[7.19155872e-09, 9.75123448e-21], [9.75123448e-21, 7.19155872e-09]],
        rtol=1e-2,
    )

    rotor7.w = 200
    assert_allclose(
        rotor7.H_kappa(3, 0),
        [[6.0181794e-06, 1.9785678e-18], [1.9785678e-18, 6.0181794e-06]],
        rtol=1e-2,
    )
    assert_allclose(
        rotor7.H_kappa(3, 0),
        [[5.92899683e-06, -1.24262274e-17], [-1.24262274e-17, 5.92899683e-06]],
        rtol=1e-2,
    )
    assert_allclose(
        rotor7.H_kappa(3, 0),
        [[6.53119391e-09, 4.73407722e-20], [4.73407722e-20, 6.53119391e-09]],
        rtol=1e-2,
    )


def test_save_load():
    a = rotor_example()
    a.save("teste00000000000000001")
    b = Rotor.load("teste00000000000000001.rsm")
    (Path.cwd() / "teste00000000000000001.rsm").unlink()

    assert a == b


def test_global_index():
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
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, n_link=7, kxx=stfx, kyy=stfy, cxx=0)
    support0 = BearingElement(7, kxx=stfx, kyy=stfy, cxx=0, tag="Support0")
    bearing1 = BearingElement(6, n_link=8, kxx=stfx, kyy=stfy, cxx=0)
    support1 = BearingElement(8, kxx=stfx, kyy=stfy, cxx=0, tag="Support1")

    point_mass0 = PointMass(7, m=1.0)
    point_mass1 = PointMass(8, m=1.0)

    rotor = Rotor(
        shaft_elem,
        [disk0, disk1],
        [bearing0, bearing1, support0, support1],
        [point_mass0, point_mass1],
    )

    shaft = rotor.shaft_elements
    disks = rotor.disk_elements
    bearings = rotor.bearing_elements
    pointmass = rotor.point_mass_elements

    assert shaft[0].dof_global_index.x_0 == 0
    assert shaft[0].dof_global_index.y_0 == 1
    assert shaft[0].dof_global_index.alpha_0 == 2
    assert shaft[0].dof_global_index.beta_0 == 3
    assert shaft[0].dof_global_index.x_1 == 4
    assert shaft[0].dof_global_index.y_1 == 5
    assert shaft[0].dof_global_index.alpha_1 == 6
    assert shaft[0].dof_global_index.beta_1 == 7

    assert disks[0].dof_global_index.x_2 == 8
    assert disks[0].dof_global_index.y_2 == 9
    assert disks[0].dof_global_index.alpha_2 == 10
    assert disks[0].dof_global_index.beta_2 == 11

    assert bearings[0].dof_global_index.x_0 == 0
    assert bearings[0].dof_global_index.y_0 == 1
    assert bearings[0].dof_global_index.x_7 == 28
    assert bearings[0].dof_global_index.y_7 == 29
    assert bearings[1].dof_global_index.x_6 == 24
    assert bearings[1].dof_global_index.y_6 == 25
    assert bearings[1].dof_global_index.x_8 == 30
    assert bearings[1].dof_global_index.y_8 == 31
    assert bearings[2].dof_global_index.x_7 == 28
    assert bearings[2].dof_global_index.y_7 == 29
    assert bearings[3].dof_global_index.x_8 == 30
    assert bearings[3].dof_global_index.y_8 == 31

    assert pointmass[0].dof_global_index.x_7 == 28
    assert pointmass[0].dof_global_index.y_7 == 29
    assert pointmass[1].dof_global_index.x_8 == 30
    assert pointmass[1].dof_global_index.y_8 == 31


def test_distincts_dof_elements_error():
    with pytest.raises(Exception):
        i_d = 0
        o_d = 0.05
        n = 6
        L = [0.25 for _ in range(n)]

        shaft_elem = [
            ShaftElement6DoF(
                material=steel,
                L=0.25,
                idl=0,
                odl=0.05,
                idr=0,
                odr=0.05,
                alpha=0,
                beta=0,
                rotary_inertia=False,
                shear_effects=False,
            )
            for l in L
        ]

        # purposeful error here!
        disk0 = DiskElement.from_geometry(
            n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
        )
        disk1 = DiskElement6DoF.from_geometry(
            n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
        )

        kxx = 1e6
        kyy = 0.8e6
        kzz = 1e5
        cxx = 0
        cyy = 0
        czz = 0
        bearing0 = BearingElement6DoF(
            n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, kzz=kzz, czz=czz
        )
        bearing1 = BearingElement6DoF(
            n=6, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, kzz=kzz, czz=czz
        )
        Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1], n_eigen=36)


@pytest.fixture
def rotor_6dof():
    i_d = 0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement6DoF(
            material=steel,
            L=0.25,
            idl=0,
            odl=0.05,
            idr=0,
            odr=0.05,
            alpha=0,
            beta=0,
            rotary_inertia=False,
            shear_effects=False,
        )
        for l in L
    ]

    disk0 = DiskElement6DoF.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement6DoF.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    kxx = 1e6
    kyy = 0.8e6
    kzz = 1e5
    cxx = 0
    cyy = 0
    czz = 0
    bearing0 = BearingElement6DoF(
        n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, kzz=kzz, czz=czz
    )
    bearing1 = BearingElement6DoF(
        n=6, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, kzz=kzz, czz=czz
    )

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


# @pytest.mark.skip(
#   reason="Needs investigation. It fails depending on system. Most likely due to eig solution precision"
# )
def test_modal_6dof(rotor_6dof):
    modal = rotor_6dof.run_modal(speed=0, sparse=False)
    wn = np.array(
        [
            9.91427121e-05,
            4.76215566e01,
            9.17987032e01,
            9.62914807e01,
            2.74579882e02,
            2.96518344e02,
        ]
    )
    wd = np.array(
        [
            9.91427121e-05,
            4.76215566e01,
            9.17987032e01,
            9.62914807e01,
            2.74579882e02,
            2.96518344e02,
        ]
    )

    assert_almost_equal(modal.wn[:6], wn, decimal=2)
    assert_almost_equal(modal.wd[:6], wd, decimal=2)


@pytest.fixture
def rotor8():
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
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.35)

    stfx = [1e7, 1.5e7]
    stfy = [1e7, 1.5e7]
    c = [1e3, 1.5e3]
    frequency = [50, 5000]
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=c, cyy=c, frequency=frequency)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=c, cyy=c, frequency=frequency)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def test_ucs_calc(rotor8):
    exp_stiffness_range = np.array([1000000.0, 1832980.710832, 3359818.286284])
    exp_rotor_wn = np.array([86.658114, 95.660573, 101.868254])
    exp_intersection_points_x = np.array(
        [10058123.652648, 10058123.652648, 10363082.398797]
    )
    exp_intersection_points_y = np.array([107.542416, 107.542416, 409.451575])
    ucs_results = rotor8.run_ucs()
    assert_allclose(ucs_results.stiffness_log[:3], exp_stiffness_range)
    assert_allclose(ucs_results.wn[0, :3], exp_rotor_wn)
    assert_allclose(
        ucs_results.intersection_points["x"][:3], exp_intersection_points_x, rtol=1e-3
    )
    assert_allclose(
        ucs_results.intersection_points["y"][:3], exp_intersection_points_y, rtol=1e-3
    )


def test_save_load(rotor8):
    file = Path(tempdir) / "rotor8.toml"
    rotor8.save(file)
    rotor8_loaded = Rotor.load(file)

    rotor8 == rotor8_loaded


def test_plot_rotor(rotor8):
    fig = rotor8.plot_rotor()

    for d in fig.data:
        if d["name"] == "Disk 0":
            actual_x = d["x"]
            actual_y = d["y"]
    expected_x = [
        0.5,
        0.5083333333333333,
        0.49166666666666664,
        0.5,
        None,
        0.5,
        0.5083333333333333,
        0.49166666666666664,
        0.5,
    ]
    expected_y = [0.025, 0.125, 0.125, 0.025, None, -0.025, -0.125, -0.125, -0.025]
    assert_allclose(actual_x[:4], expected_x[:4])
    assert_allclose(actual_y[:4], expected_y[:4])

    # mass scale factor
    for disk in rotor8.disk_elements:
        disk.scale_factor = "mass"

    fig = rotor8.plot_rotor()
    for d in fig.data:
        if d["name"] == "Disk 0":
            actual_x = d["x"]
            actual_y = d["y"]
    expected_x = [
        0.5,
        0.5068020833333333,
        0.4931979166666667,
        0.5,
        None,
        0.5,
        0.5068020833333333,
        0.4931979166666667,
        0.5,
    ]
    expected_y = [
        0.025,
        0.106625,
        0.106625,
        0.025,
        None,
        -0.025,
        -0.106625,
        -0.106625,
        -0.025,
    ]
    assert_allclose(actual_x[:4], expected_x[:4])
    assert_allclose(actual_y[:4], expected_y[:4])
