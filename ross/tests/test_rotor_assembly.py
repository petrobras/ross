import pickle
from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from ross.bearing_seal_element import *
from ross.disk_element import *
from ross.materials import Material, steel
from ross.point_mass import *
from ross.probe import Probe
from ross.rotor_assembly import *
from ross.shaft_element import *
from ross.units import Q_


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
    return Rotor(shaft_elm)


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
    assert_almost_equal(rotor1.M(0), Mr1, decimal=3)


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


def test_rotor_equality(rotor1, rotor2):
    assert rotor1 != rotor2


@pytest.fixture
def rotor2_bearing_mass():
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
    m = 20e-3
    bearing0 = BearingElement(0, kxx=stf, cxx=0, mxx=m)
    bearing1 = BearingElement(2, kxx=stf, cxx=0, mxx=m)

    return Rotor(shaft_elm, [disk0], [bearing0, bearing1])


def test_mass_matrix_rotor2_with_bearing_mass(rotor2_bearing_mass):
    # fmt: off
    Mr2 = np.array([[ 1.441,  0.   ,  0.   ,  0.049,  0.496,  0.   ,  0.   , -0.031, 0.   ,  0.   ,  0.   ,  0.   ],
                    [ 0.   ,  1.441, -0.049,  0.   ,  0.   ,  0.496,  0.031,  0.   , 0.   ,  0.   ,  0.   ,  0.   ],
                    [ 0.   , -0.049,  0.002,  0.   ,  0.   , -0.031, -0.002,  0.   , 0.   ,  0.   ,  0.   ,  0.   ],
                    [ 0.049,  0.   ,  0.   ,  0.002,  0.031,  0.   ,  0.   , -0.002, 0.   ,  0.   ,  0.   ,  0.   ],
                    [ 0.496,  0.   ,  0.   ,  0.031, 35.431,  0.   ,  0.   ,  0.   , 0.496,  0.   ,  0.   , -0.031],
                    [ 0.   ,  0.496, -0.031,  0.   ,  0.   , 35.431,  0.   ,  0.   , 0.   ,  0.496,  0.031,  0.   ],
                    [ 0.   ,  0.031, -0.002,  0.   ,  0.   ,  0.   ,  0.183,  0.   , 0.   , -0.031, -0.002,  0.   ],
                    [-0.031,  0.   ,  0.   , -0.002,  0.   ,  0.   ,  0.   ,  0.183, 0.031,  0.   ,  0.   , -0.002],
                    [ 0.   ,  0.   ,  0.   ,  0.   ,  0.496,  0.   ,  0.   ,  0.031, 1.441,  0.   ,  0.   , -0.049],
                    [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.496, -0.031,  0.   , 0.   ,  1.441,  0.049,  0.   ],
                    [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.031, -0.002,  0.   , 0.   ,  0.049,  0.002,  0.   ],
                    [ 0.   ,  0.   ,  0.   ,  0.   , -0.031,  0.   ,  0.   , -0.002, -0.049,  0.   ,  0.   ,  0.002]])
    # fmt: on
    assert_almost_equal(rotor2_bearing_mass.M(0), Mr2, decimal=3)


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
    assert_almost_equal(rotor2.M(0), Mr2, decimal=3)


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
        ]
    )
    modal2_0 = rotor2.run_modal(speed=0)
    rotor2_evals, rotor2_evects = rotor2._eigen(speed=0)
    assert_allclose(rotor2_evals[:6], evals_sorted, rtol=1e-3)
    assert_allclose(modal2_0.evalues[:6], evals_sorted, rtol=1e-3)
    modal2_10000 = rotor2.run_modal(speed=10000)
    assert_allclose(modal2_10000.evalues[:6], evals_sorted_w_10000, rtol=1e-1)

    # test run_modal with Q_
    modal2_10000 = rotor2.run_modal(speed=Q_(95492.96585513721, "RPM"))
    assert_allclose(modal2_10000.evalues[:6], evals_sorted_w_10000, rtol=1e-1)


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


def test_add_nodes_simple(rotor3):
    new_nodes_pos = [0.42, 1.036]
    modified_rotor = rotor3.add_nodes(new_nodes_pos)

    assert_equal(modified_rotor.nodes[-1], rotor3.nodes[-1] + len(new_nodes_pos))
    assert_equal(
        len(modified_rotor.shaft_elements),
        len(rotor3.shaft_elements) + len(new_nodes_pos),
    )
    assert_almost_equal(modified_rotor.m, rotor3.m)
    assert_almost_equal(modified_rotor.CG, rotor3.CG)

    assert_equal(modified_rotor.nodes_pos[2], 0.42)
    assert_equal(
        modified_rotor.shaft_elements_length[1]
        + modified_rotor.shaft_elements_length[2],
        rotor3.shaft_elements_length[1],
    )
    assert_equal(
        modified_rotor.shaft_elements[1].L + modified_rotor.shaft_elements[2].L,
        rotor3.shaft_elements[1].L,
    )

    assert_equal(modified_rotor.nodes_pos[6], 1.036)
    assert_equal(
        modified_rotor.shaft_elements_length[5]
        + modified_rotor.shaft_elements_length[6],
        rotor3.shaft_elements_length[5],
    )
    assert_equal(
        modified_rotor.shaft_elements[5].L + modified_rotor.shaft_elements[6].L,
        rotor3.shaft_elements[5].L,
    )

    assert_equal(modified_rotor.disk_elements[0].n, rotor3.disk_elements[0].n + 1)
    assert_equal(modified_rotor.disk_elements[1].n, rotor3.disk_elements[1].n + 1)

    assert_equal(modified_rotor.bearing_elements[0].n, rotor3.bearing_elements[0].n)
    assert_equal(
        modified_rotor.bearing_elements[-1].n, rotor3.bearing_elements[-1].n + 2
    )


def test_modal_fig_orientation(rotor3):
    modal1 = rotor3.run_modal(Q_(900, "RPM"))
    fig1 = modal1.plot_mode_2d(1, orientation="major")
    data_major = fig1.data[0].y

    # fmt: off
    expected_data_major = np.array([
        0.3330699 , 0.41684076, 0.49947039, 0.5796177 , 0.65594162,
        0.65594162, 0.72732014, 0.79268256, 0.85076468, 0.90030229,
        0.90030229, 0.9402937 , 0.97041024, 0.99039723, 1.        ,
        1.        , 0.99901483, 0.98731591, 0.9647654 , 0.93122548,
        0.93122548, 0.88677476, 0.83255026, 0.77000169, 0.70057879,
        0.70057879, 0.62550815, 0.54607111, 0.46379946, 0.38022502
    ])
    # fmt: on

    modal2 = rotor3.run_modal(Q_(900, "RPM"))
    fig2 = modal2.plot_mode_2d(1, orientation="x")
    data_x = fig2.data[0].y

    modal3 = rotor3.run_modal(Q_(900, "RPM"))
    fig3 = modal3.plot_mode_2d(1, orientation="y")
    data_y = fig3.data[0].y

    # fmt: off
    expected_data_y = np.array([
        1.63888742e-13, 1.97035201e-13, 2.29738935e-13, 2.61467959e-13,
        2.91690288e-13, 2.91690288e-13, 3.19972642e-13, 3.45901475e-13,
        3.68974412e-13, 3.88689077e-13, 3.88689077e-13, 4.04657656e-13,
        4.16754177e-13, 4.24869024e-13, 4.28892585e-13, 4.28892585e-13,
        4.28743563e-13, 4.24376114e-13, 4.15733802e-13, 4.02760190e-13,
        4.02760190e-13, 3.85469076e-13, 3.64306492e-13, 3.39864356e-13,
        3.12734588e-13, 3.12734588e-13, 2.83402610e-13, 2.52356655e-13,
        2.20192854e-13, 1.87507335e-13
    ])
    # fmt: on

    assert_almost_equal(data_major, expected_data_major)
    assert_almost_equal(data_x, expected_data_major)
    assert_almost_equal(data_y, expected_data_y)


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
    assert_allclose(modal3_0.kappa(0, 0)["Major axis"], 0.3826857177947612, rtol=1e-3)
    assert_allclose(modal3_0.kappa(0, 0)["Minor axis"], 0.0, rtol=1e-3, atol=1e-6)
    assert_allclose(modal3_0.kappa(0, 0)["kappa"], -0.0, rtol=1e-3, atol=1e-6)

    modal3_2000 = rotor3.run_modal(speed=2000)
    assert_allclose(modal3_2000.kappa(0, 0)["Frequency"], 77.37957042, rtol=1e-3)
    assert_allclose(modal3_2000.kappa(0, 0)["Major axis"], 0.384089, rtol=1e-3)
    assert_allclose(modal3_2000.kappa(0, 0)["Minor axis"], 0.23617, rtol=1e-3)
    assert_allclose(modal3_2000.kappa(0, 0)["kappa"], -0.6148843693807821, rtol=1e-3)

    assert_allclose(modal3_2000.kappa(0, 1)["Frequency"], 88.98733511566752, rtol=1e-3)
    assert_allclose(modal3_2000.kappa(0, 1)["Major axis"], 0.353984, rtol=1e-3)
    assert_allclose(modal3_2000.kappa(0, 1)["Minor axis"], 0.299359, rtol=1e-3)
    assert_allclose(modal3_2000.kappa(0, 1)["kappa"], 0.8456866641084784, rtol=1e-3)

    assert_allclose(modal3_2000.kappa(1, 1)["Frequency"], 88.98733511566752, rtol=1e-3)
    assert_allclose(modal3_2000.kappa(1, 1)["Major axis"], 0.671776, rtol=1e-3)
    assert_allclose(modal3_2000.kappa(1, 1)["Minor axis"], 0.510407, rtol=1e-3)
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
            [0.000000e00, 1.130932e-06, 1.891347e-04, 4.522926e-05],
            [0.000000e00, 3.010645e-06, 1.507711e-04, 5.001870e-05],
            [0.000000e00, 1.967641e-05, 5.365586e-05, 2.973699e-05],
            [0.000000e00, 2.015411e-05, 1.783092e-05, 1.228659e-05],
        ]
    )
    mag_exp_2_unb = np.array(
        [
            [0.000000e00, 4.787628e-06, 2.312706e-04, 6.889610e-05],
            [0.000000e00, 3.137313e-06, 1.879121e-04, 8.067994e-05],
            [0.000000e00, 3.808153e-05, 6.077380e-05, 5.456006e-05],
            [0.000000e00, 4.179332e-05, 1.456000e-05, 2.197655e-05],
        ]
    )

    omega = np.linspace(0.0, 450.0, 4)
    freq_resp = rotor4.run_forced_response(force=F0, speed_range=omega)
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[:4, :4], mag_exp, rtol=1e-6)

    freq_resp = rotor4.run_unbalance_response(2, 0.001, 0, frequency=omega)
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[:4, :4], mag_exp, rtol=1e-6)

    freq_resp = rotor4.run_unbalance_response(2, 0.001, 0, frequency=omega)
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[:4, :4], mag_exp, rtol=1e-6)

    freq_resp = rotor4.run_unbalance_response(
        [2, 3], [0.001, 0.001], [0.0, 0], frequency=omega
    )
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[:4, :4], mag_exp_2_unb, rtol=1e-6)


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

    fig = static.plot_free_body_diagram()
    assert list(fig.select_annotations())[7]["text"] == "Shaft weight = 225.57N"

    expected_deformation = np.array(
        [
            -4.942745e-18,
            -4.512491e-04,
            -7.884209e-04,
            -9.181142e-04,
            -8.085602e-04,
            -4.687889e-04,
            -5.561716e-18,
        ]
    )

    assert_allclose(
        static.deformation,
        expected_deformation,
    )
    fig = static.plot_deformation()
    assert_allclose(fig.data[1]["y"], expected_deformation)

    expected_vx = np.array(
        [
            -494.274533,
            -456.679111,
            -456.679111,
            -419.083689,
            -99.492525,
            -61.897103,
            -61.897103,
            -24.301681,
            480.980792,
            518.576214,
            518.576214,
            556.171636,
        ]
    )
    assert_allclose(static.Vx, expected_vx)
    fig = static.plot_shearing_force()
    assert_allclose((fig.data[1]["y"]), expected_vx)

    expected_moment = np.array(
        [
            0.0,
            -118.8692056,
            -118.8692056,
            -228.3395557,
            -228.3395557,
            -248.5132592,
            -248.5132592,
            -259.2881072,
            -259.2881072,
            -134.3434813,
            -134.3434813,
            0.0,
        ]
    )
    assert_almost_equal(static.Bm, expected_moment)
    fig = static.plot_bending_moment()
    assert_allclose((fig.data[1]["y"]), expected_moment)


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
        static.deformation,
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
        static.Vx,
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
        static.Bm,
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


@pytest.fixture
def rotor9():
    # More complex rotor based on a centrifugal compressor
    rotor9 = Rotor.load(Path(__file__).parent / "data/rotor.toml")
    return rotor9


def test_add_nodes_complex(rotor9):
    new_nodes_pos = [1.244, 0.052, 1.637]
    modified_rotor = rotor9.add_nodes(new_nodes_pos)

    assert_equal(modified_rotor.nodes[-1], rotor9.nodes[-1] + len(new_nodes_pos))
    assert_equal(
        len(modified_rotor.shaft_elements),
        len(rotor9.shaft_elements) + len(new_nodes_pos) + 2,
    )
    assert_almost_equal(modified_rotor.m, rotor9.m)
    assert_almost_equal(modified_rotor.CG, rotor9.CG)

    assert_equal(
        modified_rotor.shaft_elements[1].L + modified_rotor.shaft_elements[2].L,
        rotor9.shaft_elements[1].L,
    )
    assert_equal(
        modified_rotor.shaft_elements_length[1]
        + modified_rotor.shaft_elements_length[2],
        rotor9.shaft_elements_length[1],
    )
    assert_equal(modified_rotor.nodes_pos[2], 0.052)

    assert_equal(
        modified_rotor.shaft_elements[67].L, modified_rotor.shaft_elements[68].L
    )
    assert_equal(
        modified_rotor.shaft_elements[67].L + modified_rotor.shaft_elements[69].L,
        rotor9.shaft_elements[67].L,
    )
    assert_equal(
        modified_rotor.shaft_elements_length[41]
        + modified_rotor.shaft_elements_length[42],
        rotor9.shaft_elements_length[40],
    )
    assert_equal(modified_rotor.nodes_pos[42], 1.244)

    for i, elm in enumerate(rotor9.disk_elements):
        assert_equal(modified_rotor.disk_elements[i].n, elm.n + 1)

    assert_equal(modified_rotor.bearing_elements[0].n, rotor9.bearing_elements[0].n + 1)
    assert_equal(
        modified_rotor.bearing_elements[-1].n, rotor9.bearing_elements[-1].n + 2
    )


def test_static_analysis_rotor6(rotor6):
    static = rotor6.run_static()

    assert_almost_equal(
        static.deformation,
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
        static.Vx,
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
        static.Bm,
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


def test_static_analysis_rotor9(rotor9):
    static = rotor9.run_static()
    assert_allclose(sum(static.bearing_forces.values()), rotor9.m * 9.8065)
    assert_almost_equal(static.bearing_forces["node_7"], 1216.2827567768297)


def test_static_analysis_high_stiffness(rotor2):
    static = rotor2.run_static()
    assert_allclose(sum(static.bearing_forces.values()), rotor2.m * 9.8065)
    assert_almost_equal(static.bearing_forces["node_0"], 197.39100421969422)
    stf = 1e14
    bearing0 = BearingElement(0, kxx=stf, cxx=0)
    bearing1 = BearingElement(2, kxx=stf, cxx=0)
    rotor2 = Rotor(
        shaft_elements=rotor2.shaft_elements,
        disk_elements=rotor2.disk_elements,
        bearing_elements=[bearing0, bearing1],
    )
    static = rotor2.run_static()
    assert_allclose(sum(static.bearing_forces.values()), rotor2.m * 9.8065)
    assert_almost_equal(static.bearing_forces["node_0"], 197.39100421969422)


def test_static_bearing_with_disks(rotor3):
    # this test is related to #845, where a bearing is added at the same node as a disk
    disk0 = DiskElement(n=0, m=1, Id=0, Ip=0)
    disks = rotor3.disk_elements + [disk0]
    rotor = Rotor(
        shaft_elements=rotor3.shaft_elements,
        bearing_elements=rotor3.bearing_elements,
        disk_elements=disks,
    )

    static = rotor.run_static()

    assert_allclose(sum(static.bearing_forces.values()), rotor.m * 9.8065)
    assert_almost_equal(static.bearing_forces["node_0"], 504.08103349786404)
    expected_deformation = np.array(
        [
            -5.04081033e-18,
            -4.51249080e-04,
            -7.88420862e-04,
            -9.18114186e-04,
            -8.08560214e-04,
            -4.68788883e-04,
            -5.56171636e-18,
        ]
    )

    assert_allclose(static.deformation, expected_deformation)

    # test plots
    fig = static.plot_deformation()
    assert_allclose(fig.data[1]["y"], expected_deformation)


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
    assert_allclose(modal7.kappa(3, 0)["Minor axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Minor axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Minor axis"], 0.128085, atol=1e-6)
    assert_allclose(modal7.kappa(3, 0)["Major axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Major axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Major axis"], 0.128085, atol=1e-6)

    modal7 = rotor7.run_modal(200)
    assert_allclose(modal7.kappa(3, 0)["Minor axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Minor axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Minor axis"], 0.132574, atol=1e-6)
    assert_allclose(modal7.kappa(3, 0)["Major axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Major axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Major axis"], 0.132574, atol=1e-6)

    modal7 = rotor7.run_modal(400)
    assert_allclose(modal7.kappa(3, 0)["Minor axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Minor axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Minor axis"], 0.131419, atol=1e-6)
    assert_allclose(modal7.kappa(3, 0)["Major axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 1)["Major axis"], 1.0, atol=1e-6)
    assert_allclose(modal7.kappa(3, 2)["Major axis"], 0.131419, atol=1e-6)


def test_plot_mode(rotor7):
    # run this test with sparse=False, since small differences in the
    # eigenvector can cause the assertion to fail
    modal7 = rotor7.run_modal(50, sparse=False)

    fig = modal7.plot_orbit(1, 3)
    expected_radius = 1

    assert fig.data[0]["line"]["color"] == "#1f77b4"  # blue
    assert_allclose(
        np.sqrt(fig.data[0].x ** 2 + fig.data[0].y ** 2)[0], expected_radius
    )

    fig = modal7.plot_mode_2d(1)

    mode_shape = fig.data[0].y
    mode_x = fig.data[0].x

    poly_coefs = np.polyfit(mode_x, mode_shape, 3)

    expected_coefs = np.array([-0.05672087, -1.04116649, 1.719815])

    assert fig.data[0]["line"]["color"] == "#1f77b4"  # blue
    assert_allclose(poly_coefs[:-1], expected_coefs, rtol=1e-5)


def test_unbalance(rotor3):
    unb = rotor3.run_unbalance_response(
        node=0, unbalance_magnitude=1, unbalance_phase=0, frequency=[50, 100]
    )
    amplitude_expected = np.array([0.003026, 0.004018])
    data = unb.data_magnitude(probe=[(0, 45)], probe_units="deg")
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)
    data = unb.data_magnitude(probe=[Probe(0, Q_(45, "deg"), tag="Probe 1 - Node 0")])
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)

    phase_expected = np.array([0.785398, 0.785398])
    data = unb.data_phase(probe=[(0, 45)], probe_units="deg")
    assert_allclose(data["Probe 1 - Node 0"], phase_expected, rtol=1e-4)
    data = unb.data_phase(probe=[Probe(0, Q_(45, "deg"), tag="Probe 1 - Node 0")])
    assert_allclose(data["Probe 1 - Node 0"], phase_expected, rtol=1e-4)

    amplitude_expected = np.array([0.003487, 0.005363])
    data = unb.data_magnitude(probe=[(0, "major")])
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)
    data = unb.data_magnitude(probe=[Probe(0, "major", tag="Probe 1 - Node 0")])
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)

    phase_expected = np.array([1.5742, 1.573571])
    data = unb.data_phase(probe=[(0, "major")], probe_units="deg")
    assert_allclose(data["Probe 1 - Node 0"], phase_expected, rtol=1e-4)
    data = unb.data_phase(probe=[Probe(0, "major", tag="Probe 1 - Node 0")])
    assert_allclose(data["Probe 1 - Node 0"], phase_expected, rtol=1e-4)


def test_deflected_shape(rotor7):
    # change to asymmetric stiffness to it is easier to get the major axis at the same place
    bearing0 = BearingElement(0, kxx=1e6, kyy=2e6, cxx=1e3, cyy=1e3)
    bearing1 = BearingElement(6, kxx=1e6, kyy=2e6, cxx=1e3, cyy=1e3)
    rotor7 = Rotor(
        shaft_elements=rotor7.shaft_elements,
        disk_elements=rotor7.disk_elements,
        bearing_elements=[bearing0, bearing1],
    )

    forced = rotor7.run_unbalance_response(
        node=0, unbalance_magnitude=1, unbalance_phase=0, frequency=[50]
    )
    fig = forced.plot_deflected_shape_3d(speed=50)
    # check major axis
    expected_x = np.array([0.0, 0.0625, 0.125, 0.1875, 0.25, 0.25, 0.3125, 0.375])
    expected_y = np.array(
        [
            0.002703,
            0.002678,
            0.002646,
            0.002605,
            0.002552,
            0.002552,
            0.002487,
            0.002414,
        ]
    )
    expected_z = np.array(
        [
            5.653374e-05,
            5.639119e-05,
            5.596652e-05,
            5.516375e-05,
            5.388689e-05,
            5.388689e-05,
            5.215109e-05,
            5.016332e-05,
        ]
    )
    assert_allclose(fig.data[-3]["x"][:8], expected_x, rtol=1e-3)
    assert_allclose(fig.data[-3]["y"][:8], expected_y, rtol=1e-3)
    assert_allclose(fig.data[-3]["z"][:8], expected_z, rtol=1e-3)


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

    assert shaft[0].dof_global_index["x_0"] == 0
    assert shaft[0].dof_global_index["y_0"] == 1
    assert shaft[0].dof_global_index["alpha_0"] == 2
    assert shaft[0].dof_global_index["beta_0"] == 3
    assert shaft[0].dof_global_index["x_1"] == 4
    assert shaft[0].dof_global_index["y_1"] == 5
    assert shaft[0].dof_global_index["alpha_1"] == 6
    assert shaft[0].dof_global_index["beta_1"] == 7

    assert disks[0].dof_global_index["x_2"] == 8
    assert disks[0].dof_global_index["y_2"] == 9
    assert disks[0].dof_global_index["alpha_2"] == 10
    assert disks[0].dof_global_index["beta_2"] == 11

    assert bearings[0].dof_global_index["x_0"] == 0
    assert bearings[0].dof_global_index["y_0"] == 1
    assert bearings[0].dof_global_index["x_7"] == 28
    assert bearings[0].dof_global_index["y_7"] == 29
    assert bearings[1].dof_global_index["x_6"] == 24
    assert bearings[1].dof_global_index["y_6"] == 25
    assert bearings[1].dof_global_index["x_8"] == 30
    assert bearings[1].dof_global_index["y_8"] == 31
    assert bearings[2].dof_global_index["x_7"] == 28
    assert bearings[2].dof_global_index["y_7"] == 29
    assert bearings[3].dof_global_index["x_8"] == 30
    assert bearings[3].dof_global_index["y_8"] == 31

    assert pointmass[0].dof_global_index["x_7"] == 28
    assert pointmass[0].dof_global_index["y_7"] == 29
    assert pointmass[1].dof_global_index["x_8"] == 30
    assert pointmass[1].dof_global_index["y_8"] == 31


def test_distinct_dof_elements_error():
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
            l,
            i_d,
            o_d,
            material=steel,
            alpha=1,
            beta=1e-5,
        )
        for l in L
    ]

    disk0 = DiskElement6DoF.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement6DoF.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    bearing0 = BearingElement6DoF(n=0, kxx=1e6, kyy=8e5, kzz=1e5, cxx=0, cyy=0, czz=0)
    bearing1 = BearingElement6DoF(n=6, kxx=1e6, kyy=8e5, kzz=1e5, cxx=0, cyy=0, czz=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def test_modal_6dof(rotor_6dof):
    modal = rotor_6dof.run_modal(speed=0, sparse=False)
    wn = np.array([0.0, 47.62138, 91.79647, 96.28891, 274.56591, 296.5005])
    wd = np.array([0.01079, 47.62156, 91.79656, 96.289, 274.56607, 296.50068])

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


def test_ucs_calc_rotor2(rotor2):
    exp_rotor_wn = np.array(
        [
            [
                215.37072557,
                283.80994851,
                366.86912206,
                460.53914551,
                555.6339197,
                640.32650515,
                706.0780713,
                751.34310872,
                779.83176945,
                796.72990074,
                806.39721162,
                811.81289519,
                814.81102149,
                816.45987555,
                817.36339085,
                817.85749941,
                818.1274194,
                818.27478247,
                818.35520922,
                818.39909624,
            ],
            [
                598.02474114,
                799.46636352,
                1057.70436175,
                1373.74248509,
                1729.64320906,
                2079.77186424,
                2367.99373458,
                2567.69744007,
                2690.52418862,
                2761.4174174,
                2801.13738888,
                2823.0913233,
                2835.14811303,
                2841.74862093,
                2845.35623074,
                2847.32634838,
                2848.40174108,
                2848.98860223,
                2849.30882097,
                2849.4835338,
            ],
            [
                3956.224973,
                4062.07552707,
                4249.58400228,
                4573.6000451,
                5112.58991935,
                5963.9057796,
                7227.66815371,
                8977.98586645,
                11209.15347504,
                13742.89203466,
                16175.83274873,
                18056.34394912,
                19282.21914334,
                20006.24626009,
                20414.15343628,
                20639.66981718,
                20763.42435719,
                20831.12561174,
                20868.11080392,
                20888.30264105,
            ],
            [
                4965.28982295,
                5025.38045612,
                5136.67510967,
                5343.17672239,
                5723.17881707,
                6398.10091437,
                7508.44360294,
                9150.62680727,
                11311.73143116,
                13795.63516793,
                16186.47408516,
                18081.26060018,
                19332.61770536,
                20072.40777584,
                20489.35934417,
                20719.90875241,
                20846.4293277,
                20915.646125,
                20953.45866009,
                20974.10006581,
            ],
        ]
    )
    ucs_results = rotor2.run_ucs()
    assert_allclose(ucs_results.wn, exp_rotor_wn, rtol=1e-6)

    exp_rotor_wn = np.array(
        [
            [
                2.15371329e02,
                2.83812357e02,
                3.66877867e02,
                4.60566615e02,
                5.55704705e02,
                6.40471324e02,
                7.06315117e02,
                7.51667171e02,
                7.80222437e02,
                7.97164913e02,
                8.06859321e02,
                8.12290756e02,
                8.15297782e02,
                8.16951585e02,
                8.17857830e02,
                8.18353436e02,
                8.18624175e02,
                8.18771986e02,
                8.18852657e02,
                8.18896678e02,
            ],
            [
                1.88802037e03,
                2.28476088e03,
                2.78000154e03,
                3.41117879e03,
                4.23129187e03,
                5.31089177e03,
                6.73193403e03,
                8.56265868e03,
                1.08004453e04,
                1.32860389e04,
                1.56703469e04,
                1.75831951e04,
                1.88890430e04,
                1.96891387e04,
                2.01514788e04,
                2.04110573e04,
                2.05547675e04,
                2.06337750e04,
                2.06770552e04,
                2.07007192e04,
            ],
            [
                4.00701928e03,
                4.11399080e03,
                4.30342165e03,
                4.63058499e03,
                5.17436636e03,
                6.03219769e03,
                7.30332537e03,
                9.05967180e03,
                1.12948065e04,
                1.38430517e04,
                1.63370844e04,
                1.83684523e04,
                1.97674949e04,
                2.06278515e04,
                2.11255558e04,
                2.14050391e04,
                2.15597608e04,
                2.16448157e04,
                2.16914058e04,
                2.17168786e04,
            ],
            [
                3.75678825e04,
                3.75949522e04,
                3.76446065e04,
                3.77357411e04,
                3.79031768e04,
                3.82113096e04,
                3.87797759e04,
                3.98314192e04,
                4.17770276e04,
                4.53362190e04,
                5.16143414e04,
                6.19874527e04,
                7.79184129e04,
                1.01059654e05,
                1.33577562e05,
                1.78470328e05,
                2.39880642e05,
                3.23483351e05,
                4.37008883e05,
                5.90956828e05,
            ],
        ]
    )
    ucs_results = rotor2.run_ucs(synchronous=True)
    assert_allclose(ucs_results.wn, exp_rotor_wn)


def test_ucs_calc(rotor8):
    exp_rotor_wn = np.array(
        [
            [
                86.65811435,
                95.66057326,
                101.86825429,
                105.77854216,
                108.09888143,
                109.42658356,
                110.17043717,
                110.58225348,
                110.80874181,
                110.93285112,
                111.00072364,
                111.03780095,
                111.05804338,
                111.06909117,
                111.07511968,
                111.07840898,
                111.0802036,
                111.08118271,
                111.08171688,
                111.0820083,
            ],
            [
                274.31285391,
                325.11814102,
                366.19875009,
                394.79384259,
                412.67726755,
                423.17290554,
                429.12464766,
                432.439225,
                434.26762238,
                435.27109643,
                435.82032762,
                436.12049426,
                436.28441024,
                436.37388292,
                436.42270952,
                436.44935147,
                436.46388748,
                436.47181809,
                436.47614484,
                436.47850536,
            ],
            [
                716.78631221,
                838.8349461,
                975.91941882,
                1094.44878855,
                1172.5681763,
                1216.43412434,
                1239.90023448,
                1252.41864548,
                1259.14192099,
                1262.77498349,
                1264.74615415,
                1265.81822821,
                1266.4021086,
                1266.72035103,
                1266.89388148,
                1266.98852606,
                1267.04015246,
                1267.06831513,
                1267.08367898,
                1267.09206063,
            ],
            [
                1066.1562004,
                1194.71380733,
                1383.21963246,
                1611.2607587,
                1811.78963328,
                1932.88449588,
                1992.93768733,
                2022.35515478,
                2037.28135654,
                2045.0830944,
                2049.23769603,
                2051.47407412,
                2052.68517943,
                2053.34324267,
                2053.70146233,
                2053.89665559,
                2054.00307642,
                2054.06111426,
                2054.09277044,
                2054.11003892,
            ],
        ]
    )
    ucs_results = rotor8.run_ucs()
    assert_allclose(ucs_results.wn, exp_rotor_wn)

    exp_rotor_wn = np.array(
        [
            [
                86.90424993,
                96.07439757,
                102.44368355,
                106.4793955,
                108.88395104,
                110.26337332,
                111.03737524,
                111.46625295,
                111.70223863,
                111.83158678,
                111.9023347,
                111.94098589,
                111.96208851,
                111.97360604,
                111.97989096,
                111.98332019,
                111.98519116,
                111.98621193,
                111.98676882,
                111.98707265,
            ],
            [
                288.61681387,
                340.2268304,
                381.1079096,
                409.06350367,
                426.3367896,
                436.40207642,
                442.08713715,
                445.24635051,
                446.98699943,
                447.94170533,
                448.46406291,
                448.7494886,
                448.90533871,
                448.99040391,
                449.03682385,
                449.0621522,
                449.07597137,
                449.08351086,
                449.0876242,
                449.08986829,
            ],
            [
                925.78185127,
                1119.90966576,
                1392.50590974,
                1750.68004093,
                2182.0883474,
                2636.00694936,
                3028.14253219,
                3302.1410502,
                3467.15137752,
                3559.91633395,
                3610.85969749,
                3638.66057981,
                3653.81442225,
                3662.07522781,
                3666.57962521,
                3669.03626945,
                3670.37627387,
                3671.10725239,
                3671.50602245,
                3671.72356858,
            ],
            [
                1115.3680757,
                1270.72738636,
                1513.13125359,
                1856.25476513,
                2290.15538281,
                2763.19618667,
                3184.21829187,
                3484.87824241,
                3668.02539974,
                3771.46132472,
                3828.3605264,
                3859.43124229,
                3876.37175328,
                3885.60756844,
                3890.64388364,
                3893.39070372,
                3894.8890089,
                3895.7063473,
                3896.15223098,
                3896.39548013,
            ],
        ]
    )
    ucs_results = rotor8.run_ucs(synchronous=True)
    assert_allclose(ucs_results.wn, exp_rotor_wn)


def test_ucs_rotor9(rotor9):
    exp_rotor_wn = np.array(
        [
            [
                89.61923784473375,
                120.89655743664413,
                162.59246554800114,
                217.42553100241312,
                287.6459465590863,
                372.935347725518,
                466.2581426686537,
                551.4076269641628,
                613.1237259191993,
                650.4689786203988,
                671.2479800816767,
                682.5353774789179,
                688.6485929814294,
                691.965477524608,
                693.7688181845082,
                694.7506707995042,
                695.2857209934174,
                695.577438207025,
                695.7365318523782,
                695.8233102639163,
            ],
            [
                124.8108608009496,
                168.92618197956605,
                228.5753022242499,
                309.1399173060739,
                417.7337051968829,
                563.5536938601297,
                757.9439326311913,
                1013.3462956039914,
                1338.3333635278868,
                1716.48711727428,
                2004.8471556137,
                2078.8761601177744,
                2097.725398329982,
                2104.9250739620065,
                2108.224809182831,
                2109.8706285775806,
                2110.7269388638374,
                2111.182371668304,
                2111.427443127732,
                2111.5601496202053,
            ],
            [
                976.610014941276,
                979.8071596437271,
                985.7435440438873,
                996.8720841293019,
                1018.0435385669914,
                1059.040021915284,
                1138.8336540231767,
                1287.1396017447778,
                1530.728131538211,
                1866.6004173519655,
                2224.740955271474,
                2566.1521460703593,
                2812.5568536324695,
                2954.956819348177,
                3030.8828351975917,
                3070.974195701119,
                3092.3284826481,
                3103.806643182663,
                3110.0148616594533,
                3113.3854057718704,
            ],
            [
                2159.640747974065,
                2159.756470270719,
                2159.969971035258,
                2160.366035282546,
                2161.1082776995636,
                2162.5260227884364,
                2165.334410630516,
                2171.3131180341056,
                2186.1348008252444,
                2238.1871377763114,
                2490.9613952181917,
                2978.354255657329,
                3456.7805801535656,
                3814.959990456543,
                4040.908191796628,
                4171.396315088025,
                4244.09186723666,
                4284.062958149884,
                4305.937322108033,
                4317.887004191494,
            ],
        ]
    )
    ucs_results = rotor9.run_ucs()
    assert_allclose(ucs_results.wn, exp_rotor_wn)

    exp_rotor_wn = np.array(
        [
            [
                89.61947064,
                120.89741889,
                162.5960395,
                217.44095694,
                287.71229812,
                373.20676063,
                467.22436151,
                554.01174191,
                618.0228653,
                657.38198264,
                679.5050572,
                691.58822264,
                698.15087843,
                701.71684971,
                703.65712249,
                704.71396966,
                705.29001687,
                705.60412456,
                705.77544067,
                705.86888929,
            ],
            [
                126.08902147,
                170.65739193,
                230.92107312,
                312.32101392,
                422.05546021,
                569.45142419,
                766.09122048,
                1025.02536231,
                1357.18701994,
                1759.29639866,
                2139.22396755,
                2225.72234144,
                2241.24852987,
                2246.71661805,
                2249.14967666,
                2250.34715925,
                2250.96610863,
                2251.29417729,
                2251.47039346,
                2251.56572169,
            ],
            [
                1006.73169597,
                1010.04096437,
                1016.17902002,
                1027.66388689,
                1049.44232444,
                1091.39953214,
                1172.55460038,
                1322.87033841,
                1570.73575288,
                1916.3041552,
                2265.71692139,
                2655.54276614,
                2969.42618151,
                3163.00092959,
                3269.25575144,
                3325.84538219,
                3356.03761948,
                3372.26520948,
                3381.03936768,
                3385.80171304,
            ],
            [
                2282.84557629,
                2282.91523796,
                2283.04371506,
                2283.28189556,
                2283.72771552,
                2284.57736115,
                2286.25338975,
                2289.79373785,
                2298.45136454,
                2328.81587039,
                2527.00408181,
                3011.54457196,
                3511.90570401,
                3911.96812784,
                4184.16765863,
                4351.24520897,
                4448.09334563,
                4502.60333569,
                4532.82984364,
                4549.46314886,
            ],
        ]
    )
    ucs_results = rotor9.run_ucs(synchronous=True)
    assert_allclose(ucs_results.wn, exp_rotor_wn, rtol=1e-6)


def test_pickle(rotor8):
    rotor8_pickled = pickle.loads(pickle.dumps(rotor8))
    assert rotor8 == rotor8_pickled


def test_pickle(rotor_6dof):
    rotor_6dof_pickled = pickle.loads(pickle.dumps(rotor_6dof))
    assert rotor_6dof == rotor_6dof_pickled


def test_save_load(rotor8):
    file = Path(tempdir) / "rotor8.toml"
    rotor8.save(file)
    rotor8_loaded = Rotor.load(file)

    assert rotor8 == rotor8_loaded


def test_plot_rotor(rotor8):
    fig = rotor8.plot_rotor()

    for d in fig.data:
        if d["name"] == "Disk 0":
            actual_x = d["x"]
            actual_y = d["y"]
    expected_x = [0.5, 0.502, 0.498, 0.5]
    expected_y = [0.025, 0.125, 0.125, 0.025]

    assert_allclose(actual_x[:4], expected_x)
    assert_allclose(actual_y[:4], expected_y)

    # mass scale factor
    for disk in rotor8.disk_elements:
        disk.scale_factor = "mass"

    fig = rotor8.plot_rotor()
    for d in fig.data:
        if d["name"] == "Disk 0":
            actual_x = d["x"]
            actual_y = d["y"]
    expected_x = [0.5, 0.5016325, 0.4983675, 0.5]
    expected_y = [0.025, 0.106625, 0.106625, 0.025]
    assert_allclose(actual_x[:4], expected_x)
    assert_allclose(actual_y[:4], expected_y)


def test_plot_rotor_without_disk(rotor1):
    fig = rotor1.plot_rotor()
    expected_element_y = np.array(
        [0.0, 0.025, 0.025, 0.0, 0.0, -0.0, -0.025, -0.025, -0.0, -0.0]
    )
    assert_allclose(fig.data[-1]["y"], expected_element_y)


def test_axial_force():
    steel = Material("steel", E=211e9, G_s=81.2e9, rho=7810)
    L = 0.25
    N = 6
    idl = 0
    odl = 0.05

    bearings = [
        BearingElement(n=0, kxx=1e6, cxx=0, scale_factor=2),
        BearingElement(n=N, kxx=1e6, cxx=0, scale_factor=2),
    ]
    disks = [
        DiskElement.from_geometry(
            n=2, material=steel, width=0.07, i_d=odl, o_d=0.28, scale_factor="mass"
        ),
        DiskElement.from_geometry(
            n=4, material=steel, width=0.07, i_d=odl, o_d=0.35, scale_factor="mass"
        ),
    ]

    shaft = [
        ShaftElement(L=L, idl=idl, odl=odl, material=steel, axial_force=Q_(100, "kN"))
        for i in range(N)
    ]
    rotor = Rotor(shaft_elements=shaft, disk_elements=disks, bearing_elements=bearings)
    modal = rotor.run_modal(Q_(4000, "RPM"))
    expected_wd = np.array(
        [93.416071, 95.2335, 267.475281, 309.918575, 634.40757, 873.763214]
    )
    assert_allclose(modal.wd, expected_wd)


def test_torque():
    steel = Material("steel", E=211e9, G_s=81.2e9, rho=7810)
    L = 0.25
    N = 6
    idl = 0
    odl = 0.05

    bearings = [
        BearingElement(n=0, kxx=1e6, cxx=0, scale_factor=2),
        BearingElement(n=N, kxx=1e6, cxx=0, scale_factor=2),
    ]
    disks = [
        DiskElement.from_geometry(
            n=2, material=steel, width=0.07, i_d=odl, o_d=0.28, scale_factor="mass"
        ),
        DiskElement.from_geometry(
            n=4, material=steel, width=0.07, i_d=odl, o_d=0.35, scale_factor="mass"
        ),
    ]

    shaft = [
        ShaftElement(L=L, idl=idl, odl=odl, material=steel, torque=Q_(100, "kN*m"))
        for i in range(N)
    ]
    rotor = Rotor(shaft_elements=shaft, disk_elements=disks, bearing_elements=bearings)
    modal = rotor.run_modal(Q_(4000, "RPM"))
    expected_wd = np.array(
        [81.324905, 84.769077, 242.822862, 286.158147, 591.519983, 827.003048]
    )
    assert_allclose(modal.wd, expected_wd)


@pytest.fixture
def rotor_conical():
    L_total = 0.5
    odl = 0.1
    odr = 0.01
    shaft_elements = []

    N = 4
    delta_d = (odl - odr) / N

    odl_el = odl
    odr_el = odl - delta_d

    for i in range(N):
        L = L_total / N

        sh_el = ShaftElement(
            n=i,
            idl=0.0,
            odl=odl_el,
            idr=0.0,
            odr=odr_el,
            material=steel,
            L=L,
        )
        shaft_elements.append(sh_el)

        odl_el -= delta_d
        odr_el -= delta_d

    bearing0 = BearingElement(n=0, kxx=1e20, kyy=1e20, cxx=0)
    bearing1 = BearingElement(n=N, kxx=1e20, kyy=1e20, cxx=0)

    rotor_conical = Rotor(
        shaft_elements=shaft_elements, bearing_elements=[bearing0, bearing1]
    )

    return rotor_conical


def test_rotor_conical_frequencies(rotor_conical):
    modal = rotor_conical.run_modal(speed=0)
    expected_wn = np.array(
        [
            1630.7509182,
            1630.75091824,
            9899.54379138,
            9899.54396231,
            21074.45440319,
            21074.45440372,
        ]
    )
    assert_allclose(modal.wn, expected_wn, rtol=1e-5)
