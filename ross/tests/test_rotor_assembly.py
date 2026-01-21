import pickle
from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from scipy.signal import find_peaks
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from ross import SensitivityResults
from ross.bearing_seal_element import *
from ross.disk_element import *
from ross.materials import Material, steel
from ross.point_mass import *
from ross.probe import Probe
from ross.rotor_assembly import *
from ross.shaft_element import *
from ross.units import Q_


def get_dofs(ndof):
    dofs_6 = np.arange(2, ndof, 3)
    dofs_4 = np.setdiff1d(np.arange(0, ndof), dofs_6)
    return dofs_4, dofs_6


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
    M4r1 = np.array([[ 1.421,  0.   ,  0.   ,  0.049,  0.496,  0.   ,  0.   , -0.031,  0.   ,  0.   ,  0.   ,  0.   ],
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
    M6r1 = np.array([[ 1.278, 0.   , 0.639, 0.   , 0.   , 0.],
                     [ 0.   , 0.   , 0.   , 0.   , 0.   , 0.],
                     [ 0.639, 0.   , 2.556, 0.   , 0.639, 0.],
                     [ 0.   , 0.   , 0.   , 0.001, 0.   , 0.],
                     [ 0.   , 0.   , 0.639, 0.   , 1.278, 0.],
                     [ 0.   , 0.   , 0.   , 0.   , 0.   , 0.]])
    # fmt: on
    dofs_4, dofs_6 = get_dofs(rotor1.ndof)
    assert_almost_equal(rotor1.M(0)[np.ix_(dofs_4, dofs_4)], M4r1, decimal=3)
    assert_almost_equal(rotor1.M(0)[np.ix_(dofs_6, dofs_6)], M6r1, decimal=3)


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
    M4r2 = np.array([[ 1.441,  0.   ,  0.   ,  0.049,  0.496,  0.   ,  0.   , -0.031, 0.   ,  0.   ,  0.   ,  0.   ],
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
    M6r2 = np.array([[ 1.278,  0.   ,  0.639,  0.   ,  0.   ,  0.],
                     [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.],
                     [ 0.639,  0.   , 35.146,  0.   ,  0.639,  0.],
                     [ 0.   ,  0.   ,  0.   ,  0.33 ,  0.   ,  0.],
                     [ 0.   ,  0.   ,  0.639,  0.   ,  1.278,  0.],
                     [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.]])
    # fmt: on
    dofs_4, dofs_6 = get_dofs(rotor2_bearing_mass.ndof)
    assert_almost_equal(
        rotor2_bearing_mass.M(0)[np.ix_(dofs_4, dofs_4)], M4r2, decimal=3
    )
    assert_almost_equal(
        rotor2_bearing_mass.M(0)[np.ix_(dofs_6, dofs_6)], M6r2, decimal=3
    )


def test_mass_matrix_rotor2(rotor2):
    # fmt: off
    M4r2 = np.array([[  1.421,   0.   ,   0.   ,   0.049,   0.496,   0.   ,   0.   ,  -0.031,   0.   ,   0.   ,   0.   ,   0.   ],
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
    M6r2 = np.array([[ 1.278,  0.   ,  0.639,  0.   ,  0.   ,  0.],
                     [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.],
                     [ 0.639,  0.   , 35.146,  0.   ,  0.639,  0.],
                     [ 0.   ,  0.   ,  0.   ,  0.33 ,  0.   ,  0.],
                     [ 0.   ,  0.   ,  0.639,  0.   ,  1.278,  0.],
                     [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.]])
    # fmt: on
    dofs_4, dofs_6 = get_dofs(rotor2.ndof)
    assert_almost_equal(rotor2.M(0)[np.ix_(dofs_4, dofs_4)], M4r2, decimal=3)
    assert_almost_equal(rotor2.M(0)[np.ix_(dofs_6, dofs_6)], M6r2, decimal=3)


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
    A0_1 = np.diag(np.ones(rotor2.ndof))
    dof = 3 * rotor2.number_dof
    assert_almost_equal(rotor2.A()[:dof, dof : 2 * dof], A0_1, decimal=3)


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
    dofs_4, dofs_6 = get_dofs(2 * rotor2.ndof)
    dof = 3 * rotor2.number_dof
    row = np.intersect1d(np.arange(dof, 2 * dof), dofs_4)
    col = np.intersect1d(np.arange(0, dof), dofs_4)
    assert_almost_equal(rotor2.A()[np.ix_(row, col)] / 1e7, A1_0, decimal=3)

    # fmt: off
    A1_0 = np.array([[-133.282,   -0.   ,  136.884,   -0.   ,   -3.602,   -0.   ],
                     [  -0.   ,  -49.951,   -0.   ,   49.996,   -0.   ,   -0.045],
                     [   7.204,   -0.   ,  -14.408,   -0.   ,    7.204,   -0.   ],
                     [  -0.   ,    0.091,   -0.   ,   -0.181,   -0.   ,    0.091],
                     [  -3.602,   -0.   ,  136.884,   -0.   , -133.282,   -0.   ],
                     [  -0.   ,   -0.045,   -0.   ,   49.996,   -0.   ,  -49.951]])
    # fmt: on
    row = np.intersect1d(np.arange(dof, 2 * dof), dofs_6)
    col = np.intersect1d(np.arange(0, dof), dofs_6)
    assert_almost_equal(rotor2.A()[np.ix_(row, col)] / 1e7, A1_0, decimal=3)


def test_a1_1_matrix_rotor2(rotor2):
    A1_1 = np.zeros((rotor2.ndof, rotor2.ndof))
    dof = 3 * rotor2.number_dof
    assert_almost_equal(rotor2.A()[dof : 2 * dof, dof : 2 * dof] / 1e7, A1_1, decimal=3)


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
    rotor2_evals, rotor2_evects = rotor2._eigen(speed=0, sparse=False)
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
    camp_desired = np.array([
        [82.65303734,  86.65811435, 254.52047828, 274.31285391, 650.40786626, 679.48903239],
        [82.60929602,  86.68625235, 251.70037114, 276.87787937, 650.40786627, 652.85679897],
        [82.48132723,  86.76734307, 245.49092844, 282.33294699, 650.40786626, 614.05536277]
    ])
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

    F0 = np.zeros((rotor4.ndof, 4), dtype=complex)
    F0[2 * rotor4.number_dof, 1] = 22.5 + 0.0j
    F0[2 * rotor4.number_dof, 2] = 90.0 + 0.0j
    F0[2 * rotor4.number_dof, 3] = 202.5 + 0.0j
    F0[2 * rotor4.number_dof + 1, 1] = 0.0 - 22.5j
    F0[2 * rotor4.number_dof + 1, 2] = 0.0 - 90j
    F0[2 * rotor4.number_dof + 1, 3] = 0.0 - 202.5j

    freq_resp = rotor4.run_forced_response(force=F0, speed_range=omega)
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[0:2, :], mag_exp[0:2, :])
    assert_allclose(mag[3:5, :], mag_exp[2:4, :])

    freq_resp = rotor4.run_unbalance_response(2, 0.001, 0, frequency=omega)
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[0:2, :], mag_exp[0:2, :])
    assert_allclose(mag[3:5, :], mag_exp[2:4, :])

    freq_resp = rotor4.run_unbalance_response(2, 0.001, 0, frequency=omega)
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[0:2, :], mag_exp[0:2, :])
    assert_allclose(mag[3:5, :], mag_exp[2:4, :])

    freq_resp = rotor4.run_unbalance_response(
        [2, 3], [0.001, 0.001], [0.0, 0], frequency=omega
    )
    mag = abs(freq_resp.forced_resp)
    assert_allclose(mag[0:2, :], mag_exp_2_unb[0:2, :])
    assert_allclose(mag[3:5, :], mag_exp_2_unb[2:4, :])


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
        modal7 = rotor7.run_modal(speed, num_modes=14)
        whirl_val = modal7.whirl_values()
        whirl_dir = modal7.whirl_direction()
        assert_allclose(
            whirl_val[~np.isnan(whirl_val)], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], atol=0
        )
        assert_equal(
            whirl_dir[whirl_dir != "None"],
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
    amplitude_expected = np.array([0.003158927232913641, 0.004620055491206476])
    data = unb.data_magnitude(probe=[(0, 45)], probe_units="deg")
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)
    data = unb.data_magnitude(probe=[Probe(0, Q_(45, "deg"), tag="Probe 1 - Node 0")])
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)
    phase_expected = np.array([0.9096239298802793, 1.0057118170915373])
    data = unb.data_phase(probe=[(0, 45)], probe_units="deg")
    assert_allclose(data["Probe 1 - Node 0"], phase_expected, rtol=1e-4)
    data = unb.data_phase(probe=[Probe(0, Q_(45, "deg"), tag="Probe 1 - Node 0")])
    assert_allclose(data["Probe 1 - Node 0"], phase_expected, rtol=1e-4)
    amplitude_expected = np.array([0.0035259958428500164, 0.005518031001232163])
    data = unb.data_magnitude(probe=[(0, "major")])
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)
    data = unb.data_magnitude(probe=[Probe(0, "major", tag="Probe 1 - Node 0")])
    assert_allclose(data["Probe 1 - Node 0"], amplitude_expected, rtol=1e-4)
    phase_expected = np.array([1.5742963267948966, 1.57357163267948966])
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
    assert_allclose(fig.data[-4]["x"][:8], expected_x, rtol=1e-4)
    assert_allclose(fig.data[-4]["y"][:8], expected_y, rtol=1e-4)
    assert_allclose(fig.data[-4]["z"][:8], expected_z, rtol=1e-4)


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
    assert shaft[0].dof_global_index["z_0"] == 2
    assert shaft[0].dof_global_index["alpha_0"] == 3
    assert shaft[0].dof_global_index["beta_0"] == 4
    assert shaft[0].dof_global_index["theta_0"] == 5
    assert shaft[0].dof_global_index["x_1"] == 6
    assert shaft[0].dof_global_index["y_1"] == 7
    assert shaft[0].dof_global_index["z_1"] == 8
    assert shaft[0].dof_global_index["alpha_1"] == 9
    assert shaft[0].dof_global_index["beta_1"] == 10
    assert shaft[0].dof_global_index["theta_1"] == 11

    assert disks[0].dof_global_index["x_2"] == 12
    assert disks[0].dof_global_index["y_2"] == 13
    assert disks[0].dof_global_index["z_2"] == 14
    assert disks[0].dof_global_index["alpha_2"] == 15
    assert disks[0].dof_global_index["beta_2"] == 16
    assert disks[0].dof_global_index["theta_2"] == 17

    assert bearings[0].dof_global_index["x_0"] == 0
    assert bearings[0].dof_global_index["y_0"] == 1
    assert bearings[0].dof_global_index["z_0"] == 2
    assert bearings[0].dof_global_index["x_7"] == 42
    assert bearings[0].dof_global_index["y_7"] == 43
    assert bearings[0].dof_global_index["z_7"] == 44
    assert bearings[1].dof_global_index["x_6"] == 36
    assert bearings[1].dof_global_index["y_6"] == 37
    assert bearings[1].dof_global_index["z_6"] == 38
    assert bearings[1].dof_global_index["x_8"] == 45
    assert bearings[1].dof_global_index["y_8"] == 46
    assert bearings[1].dof_global_index["z_8"] == 47
    assert bearings[2].dof_global_index["x_7"] == 42
    assert bearings[2].dof_global_index["y_7"] == 43
    assert bearings[2].dof_global_index["z_7"] == 44
    assert bearings[3].dof_global_index["x_8"] == 45
    assert bearings[3].dof_global_index["y_8"] == 46
    assert bearings[3].dof_global_index["z_8"] == 47

    assert pointmass[0].dof_global_index["x_7"] == 42
    assert pointmass[0].dof_global_index["y_7"] == 43
    assert pointmass[0].dof_global_index["z_7"] == 44
    assert pointmass[1].dof_global_index["x_8"] == 45
    assert pointmass[1].dof_global_index["y_8"] == 46
    assert pointmass[1].dof_global_index["z_8"] == 47


def test_distinct_dof_elements_error():
    with pytest.raises(Exception):
        i_d = 0
        o_d = 0.05
        n = 6
        L = [0.25 for _ in range(n)]

        shaft_elem = [
            ShaftElement(
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
        disk1 = DiskElement.from_geometry(
            n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
        )

        kxx = 1e6
        kyy = 0.8e6
        kzz = 1e5
        cxx = 0
        cyy = 0
        czz = 0
        bearing0 = BearingElement(
            n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, kzz=kzz, czz=czz
        )
        bearing1 = BearingElement(
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
        ShaftElement(
            l,
            i_d,
            o_d,
            material=steel,
            alpha=1,
            beta=1e-5,
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    bearing0 = BearingElement(n=0, kxx=1e6, kyy=8e5, kzz=1e5, cxx=0, cyy=0, czz=0)
    bearing1 = BearingElement(n=6, kxx=1e6, kyy=8e5, kzz=1e5, cxx=0, cyy=0, czz=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def test_modal_6dof(rotor_6dof):
    modal = rotor_6dof.run_modal(speed=0, sparse=False)
    wn = np.array([47.62138, 91.79647, 96.28891, 274.56591, 296.5005])
    wd = np.array([47.62156, 91.79656, 96.289, 274.56607, 296.50068])

    assert_almost_equal(modal.wn[:5], wn, decimal=2)
    assert_almost_equal(modal.wd[:5], wd, decimal=2)


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
    # Simplified test: use num=5 instead of num=20 to reduce test time (~75% faster)
    exp_rotor_wn = np.array(
        [
            [
                89.61923786,
                350.42266504,
                662.41928664,
                694.07268245,
                695.82331031,
            ],
            [
                124.81086083,
                523.02183296,
                1891.18155397,
                2108.7448203,
                2111.56014972,
            ],
            [
                976.61001489,
                1046.09648924,
                2047.58519431,
                3043.40464892,
                3113.38540495,
            ],
            [
                2159.64074905,
                2162.07931839,
                2321.68023491,
                4080.81909788,
                4317.88700623,
            ],
        ]
    )
    ucs_results = rotor9.run_ucs(num=5)
    assert_allclose(ucs_results.wn, exp_rotor_wn)

    exp_rotor_wn = np.array(
        [
            [
                89.61947056,
                350.61514002,
                670.08583401,
                703.98416314,
                705.86888929,
            ],
            [
                126.08902134,
                528.47472934,
                1971.00830161,
                2249.52916259,
                2251.5657217,
            ],
            [
                1006.73169597,
                1078.17877977,
                2100.24701415,
                3286.9082896,
                3385.80171304,
            ],
            [
                2282.84557629,
                2284.30991972,
                2383.54778909,
                4234.40738673,
                4549.46314886,
            ],
        ]
    )
    ucs_results = rotor9.run_ucs(synchronous=True, num=5)
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
        [93.416071, 95.2335, 267.475281, 309.918575, 634.40757, 650.407866]
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
        [81.324905, 84.769077, 242.822862, 286.158147, 591.519983, 650.407866]
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


def test_harmonic_response(rotor9):
    speed = 200.0
    t = np.arange(0, 10, 1e-4)

    A1, A2, A3 = 1.0, 10.0, 5.0
    p1, p2, p3 = 0.0, 0.0, 0.0
    m, e = 0.2, 0.01

    probe = Probe(15, Q_(45, "deg"))

    hb_results = rotor9.run_harmonic_balance_response(
        speed=speed,
        t=t,
        harmonic_forces=[
            {
                "node": 29,
                "magnitudes": [A1, A2, A3],
                "phases": [p1, p2, p3],
                "harmonics": [1, 2, 3],
            },
            {
                "node": 33,
                "magnitudes": [m * e * speed**2],
                "phases": [0],
                "harmonics": [1],
            },
        ],
        n_harmonics=3,
    )

    hb_fig = hb_results.plot([probe], frequency_units="Hz")

    hb_time_resp = hb_results.get_time_response()
    hb_fig = hb_time_resp.plot_dfft(
        probe=[probe],
        frequency_units="Hz",
        frequency_range=(0, 100),
        yaxis_type="log",
        fig=hb_fig,
    )

    x_hb = np.nan_to_num(np.array(hb_fig.data[0].x, dtype=float), nan=0)
    y_hb = np.nan_to_num(np.array(hb_fig.data[0].y, dtype=float), nan=0)
    idx_hb = find_peaks(y_hb)[0]
    x_hb = x_hb[idx_hb]
    y_hb = y_hb[idx_hb]

    x_dfft = np.array(hb_fig.data[1].x)
    y_dfft = np.array(hb_fig.data[1].y)
    idx_dfft = find_peaks(y_dfft)[0]
    x_dfft = x_dfft[idx_dfft]
    y_dfft = y_dfft[idx_dfft]

    x = np.array([31.83098861837907, 63.66197723675814, 95.49296585513721])
    y = np.array(
        [2.1140161650605384e-07, 3.479347424205027e-08, 2.1467969724866964e-08]
    )

    assert_allclose(x_hb, x_dfft, rtol=1e-3, atol=1e-6)
    assert_allclose(x_hb, x, rtol=1e-3, atol=1e-6)
    assert_allclose(y_hb, y_dfft, rtol=1e-3, atol=1e-6)
    assert_allclose(y_hb, y, rtol=1e-3, atol=1e-6)


def test_amb_controller():
    # Test for the magnetic_bearing_controller method.
    from ross.rotor_assembly import rotor_amb_example

    rot_speed = 1200
    dt = 0.001
    t = np.arange(0.0, 500 * dt, dt)
    unbalance_node = 27
    probe_node = 12

    rotor = rotor_amb_example()
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

    assert_allclose(mse_x, np.array(9.228097168398774e-10), rtol=1e-6, atol=1e-6)
    assert_allclose(mse_y, np.array(2.2135792430227363e-10), rtol=1e-6, atol=1e-6)


def test_amb_generic_controller():
    from ross.rotor_assembly import rotor_amb_example

    kp = 100.0
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
    unbalance_node = 27
    probe_node = 12

    rotor = rotor_amb_example(controller_transfer_function)
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

    assert_allclose(mse_x, np.array(7.934767106972457e-11), rtol=1e-6, atol=1e-6)
    assert_allclose(mse_y, np.array(3.6781959914042914e-11), rtol=1e-6, atol=1e-6)


def test_run_amb_sensitivity():
    """
    Tests the run_amb_sensitivity method for correctness of outputs and handling of various scenarios.
    """
    EXPECTED_SENSITIVITY_RESULTS = {
        "max_abs": {
            "Magnetic Bearing 0": {"x": 0.9915881235, "y": 0.9915881235},
            "Magnetic Bearing 1": {"x": 0.9880851953, "y": 0.9880851953},
        },
        "abs_slice": {
            "Magnetic Bearing 0": {
                "x": np.array(
                    [0.99158812, 0.99156866, 0.99153061, 0.99147841, 0.99142154]
                ),
                "y": np.array(
                    [0.99158812, 0.99156866, 0.99153061, 0.99147841, 0.99142154]
                ),
            },
            "Magnetic Bearing 1": {
                "x": np.array(
                    [0.9880852, 0.98805746, 0.98800146, 0.98792434, 0.98784035]
                ),
                "y": np.array(
                    [0.9880852, 0.98805746, 0.98800146, 0.98792434, 0.98784035]
                ),
            },
        },
        "phase_slice": {
            "Magnetic Bearing 0": {
                "x": np.array(
                    [
                        0.00000000e00,
                        8.77852477e-05,
                        1.59040274e-04,
                        2.11855244e-04,
                        2.44736262e-04,
                    ]
                ),
                "y": np.array(
                    [
                        0.00000000e00,
                        8.77852477e-05,
                        1.59040274e-04,
                        2.11855244e-04,
                        2.44736262e-04,
                    ]
                ),
            },
            "Magnetic Bearing 1": {
                "x": np.array(
                    [
                        0.00000000e00,
                        1.29420004e-04,
                        2.35610181e-04,
                        3.14075980e-04,
                        3.62979207e-04,
                    ]
                ),
                "y": np.array(
                    [
                        0.00000000e00,
                        1.29420004e-04,
                        2.35610181e-04,
                        3.14075980e-04,
                        3.62979207e-04,
                    ]
                ),
            },
        },
        "dofs": {
            "Magnetic Bearing 0": {"x": 72, "y": 73},
            "Magnetic Bearing 1": {"x": 258, "y": 259},
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
                    1.42060807e-11,
                    2.26922938e-11,
                    3.22559336e-11,
                ]
            ),
            "sensor": np.array(
                [
                    0.00000000e00,
                    0.00000000e00,
                    -2.22067882e-15,
                    -1.07746729e-14,
                    -2.86728919e-14,
                ]
            ),
        },
        "frequencies_slice": np.array([0.0, 100.0, 200.0, 300.0, 400.0]),
    }

    r_tol = 0
    a_tol = 1e-8

    # Setup - run the analysis
    rotor = rotor_amb_example()
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
        atol=a_tol,
        rtol=r_tol,
    )
    assert_allclose(
        time_results_amb_0_x["disturbed_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["disturbed"],
        atol=a_tol,
        rtol=r_tol,
    )
    assert_allclose(
        time_results_amb_0_x["sensor_signal"][:5],
        EXPECTED_SENSITIVITY_RESULTS["time_results_slice"]["sensor"],
        atol=a_tol,
        rtol=r_tol,
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
