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

    # test run_modal with Q_
    modal2_10000 = rotor2.run_modal(speed=Q_(95492.96585513721, "RPM"))
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
    amplitude_expected = np.array([0.003065, 0.004169])
    data = unb.data_magnitude(probe=[(0, 45)], probe_units="deg")
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)
    data = unb.data_magnitude(probe=[Probe(0, Q_(45, "deg"), tag="Probe 1 - Node 0")])
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)

    phase_expected = np.array([0.785398, 0.785398])
    data = unb.data_phase(probe=[(0, 45)], probe_units="deg")
    assert_allclose(data["Probe 1 - Node 0"], phase_expected, rtol=1e-4)
    data = unb.data_phase(probe=[Probe(0, Q_(45, "deg"), tag="Probe 1 - Node 0")])
    assert_allclose(data["Probe 1 - Node 0"], phase_expected, rtol=1e-4)

    amplitude_expected = np.array([0.003526, 0.005518])
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
            0.00274183,
            0.00269077,
            0.00263888,
            0.0025852,
            0.00252877,
            0.00252877,
            0.0024688,
            0.00240457,
        ]
    )
    expected_z = np.array(
        [
            5.72720829e-05,
            5.59724347e-05,
            5.46564072e-05,
            5.33052546e-05,
            5.19002311e-05,
            5.19002311e-05,
            5.04258251e-05,
            4.88678837e-05,
        ]
    )
    assert_allclose(fig.data[-3]["x"][:8], expected_x, rtol=1e-4)
    assert_allclose(fig.data[-3]["y"][:8], expected_y, rtol=1e-4)
    assert_allclose(fig.data[-3]["z"][:8], expected_z, rtol=1e-4)


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


def test_ucs_calc_rotor2(rotor2):
    ucs_results = rotor2.run_ucs()
    expected_intersection_points_y = [
        215.37072557303264,
        215.37072557303264,
        598.024741157381,
        598.024741157381,
        3956.224979518562,
        3956.224979518562,
        4965.289823255782,
        4965.289823255782,
    ]
    assert_allclose(
        ucs_results.intersection_points["y"], expected_intersection_points_y
    )
    fig = ucs_results.plot()
    assert_allclose(fig.data[4]["y"], expected_intersection_points_y)


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


def test_ucs_rotor9(rotor9):
    ucs_results = rotor9.run_ucs(num_modes=8)
    fig = ucs_results.plot()
    expected_x = np.array(
        [
            1.00000000e06,
            1.83298071e06,
            3.35981829e06,
            6.15848211e06,
            1.12883789e07,
            2.06913808e07,
            3.79269019e07,
            6.95192796e07,
            1.27427499e08,
            2.33572147e08,
            4.28133240e08,
            7.84759970e08,
            1.43844989e09,
            2.63665090e09,
            4.83293024e09,
            8.85866790e09,
            1.62377674e10,
            2.97635144e10,
            5.45559478e10,
            1.00000000e11,
        ]
    )
    assert_allclose(fig.data[0]["x"], expected_x)


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
