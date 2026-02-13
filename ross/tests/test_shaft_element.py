import os
import pickle
from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from ross.materials import steel
from ross.shaft_element import ShaftElement
from ross.coupling_element import CouplingElement


@pytest.fixture
def eb():
    #  Euler-Bernoulli element
    le_ = 0.25
    i_d_l = 0
    o_d_l = 0.05
    i_d_r = 0
    o_d_r = 0.05
    return ShaftElement(
        le_,
        i_d_l,
        o_d_l,
        i_d_r,
        o_d_r,
        steel,
        shear_effects=False,
        rotary_inertia=False,
        n=3,
    )


def test_index(eb):
    assert eb.dof_local_index().x_0 == 0
    assert eb.dof_local_index().y_0 == 1
    assert eb.dof_local_index().z_0 == 2
    assert eb.dof_local_index().alpha_0 == 3
    assert eb.dof_local_index().beta_0 == 4
    assert eb.dof_local_index().theta_0 == 5


def test_parameters_eb(eb):
    assert eb.phi == 0
    assert eb.L == 0.25
    assert eb.idl == 0
    assert eb.odl == 0.05
    assert eb.idr == 0
    assert eb.odr == 0.05
    assert eb.material.E == 211e9
    assert eb.material.G_s == 81.2e9
    assert eb.material.rho == 7810
    assert_almost_equal(eb.material.Poisson, 0.29926108)
    assert_almost_equal(eb.A, 0.00196349)
    assert_almost_equal(eb.Ie * 1e7, 3.06796157)


def test_mass_matrix_eb(eb):
    # fmt: off
    M0e_eb_4 = np.array([
        [ 1.42395,  0.     ,  0.     ,  0.0502 ,  0.49291,  0.     ,  0.     , -0.02967],
        [ 0.     ,  1.42395, -0.0502 ,  0.     ,  0.     ,  0.49291,  0.02967,  0.     ],
        [ 0.     , -0.0502 ,  0.00228,  0.     ,  0.     , -0.02967, -0.00171,  0.     ],
        [ 0.0502 ,  0.     ,  0.     ,  0.00228,  0.02967,  0.     ,  0.     , -0.00171],
        [ 0.49291,  0.     ,  0.     ,  0.02967,  1.42395,  0.     ,  0.     , -0.0502 ],
        [ 0.     ,  0.49291, -0.02967,  0.     ,  0.     ,  1.42395,  0.0502 ,  0.     ],
        [ 0.     ,  0.02967, -0.00171,  0.     ,  0.     ,  0.0502 ,  0.00228,  0.     ],
        [-0.02967,  0.     ,  0.     , -0.00171, -0.0502 ,  0.     ,  0.     ,  0.00228]
    ])
    M0e_eb_6 = np.array([
        [1.27791, 0.     , 0.63895, 0.     ],
        [0.     , 0.0004 , 0.     , 0.0002 ],
        [0.63895, 0.     , 1.27791, 0.     ],
        [0.     , 0.0002 , 0.     , 0.0004 ]
    ])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_almost_equal(eb.M()[np.ix_(dofs_4, dofs_4)], M0e_eb_4, decimal=5)
    assert_almost_equal(eb.M()[np.ix_(dofs_6, dofs_6)], M0e_eb_6, decimal=5)


def test_stiffness_matrix_eb(eb):
    # fmt: off
    K0e_eb_4 = np.array([
        [ 4.97157,  0.     ,  0.     ,  0.62145, -4.97157,  0.     ,  0.     ,  0.62145],
        [ 0.     ,  4.97157, -0.62145,  0.     ,  0.     , -4.97157, -0.62145,  0.     ],
        [ 0.     , -0.62145,  0.10357,  0.     ,  0.     ,  0.62145,  0.05179,  0.     ],
        [ 0.62145,  0.     ,  0.     ,  0.10357, -0.62145,  0.     ,  0.     ,  0.05179],
        [-4.97157,  0.     ,  0.     , -0.62145,  4.97157,  0.     ,  0.     , -0.62145],
        [ 0.     , -4.97157,  0.62145,  0.     ,  0.     ,  4.97157,  0.62145,  0.     ],
        [ 0.     , -0.62145,  0.05179,  0.     ,  0.     ,  0.62145,  0.10357,  0.     ],
        [ 0.62145,  0.     ,  0.     ,  0.05179, -0.62145,  0.     ,  0.     ,  0.10357]
    ])
    K0e_eb_6 = np.array([
        [ 165.71901,    0.     , -165.71901,    0.     ],
        [   0.     ,    0.01993,    0.     ,   -0.01993],
        [-165.71901,    0.     ,  165.71901,    0.     ],
        [   0.     ,   -0.01993,    0.     ,    0.01993]
    ])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_almost_equal(eb.K()[np.ix_(dofs_4, dofs_4)] / 1e7, K0e_eb_4, decimal=5)
    assert_almost_equal(eb.K()[np.ix_(dofs_6, dofs_6)] / 1e7, K0e_eb_6, decimal=5)


@pytest.fixture
def tim():
    #  Timoshenko element
    z_ = 0
    le_ = 0.25
    i_d_l = 0
    o_d_l = 0.05
    i_d_r = 0
    o_d_r = 0.05
    return ShaftElement(
        le_, i_d_l, o_d_l, i_d_r, o_d_r, steel, rotary_inertia=True, shear_effects=True
    )


def test_shaft_element_equality(tim, eb):
    assert tim != eb


def test_parameters_tim(tim):
    assert_almost_equal(tim.phi, 0.08795566)
    assert_almost_equal(tim.material.Poisson, 0.29926108)
    assert_almost_equal(tim.A, 0.00196349)
    assert_almost_equal(tim.Ie * 1e7, 3.06796157)


def test_mass_matrix_tim(tim):
    # fmt: off
    M0e_tim_4 = np.array([
        [ 1.42051,  0.     ,  0.     ,  0.04932,  0.49635,  0.     ,  0.     , -0.03055],
        [ 0.     ,  1.42051, -0.04932,  0.     ,  0.     ,  0.49635,  0.03055,  0.     ],
        [ 0.     , -0.04932,  0.00231,  0.     ,  0.     , -0.03055, -0.00178,  0.     ],
        [ 0.04932,  0.     ,  0.     ,  0.00231,  0.03055,  0.     ,  0.     , -0.00178],
        [ 0.49635,  0.     ,  0.     ,  0.03055,  1.42051,  0.     ,  0.     , -0.04932],
        [ 0.     ,  0.49635, -0.03055,  0.     ,  0.     ,  1.42051,  0.04932,  0.     ],
        [ 0.     ,  0.03055, -0.00178,  0.     ,  0.     ,  0.04932,  0.00231,  0.     ],
        [-0.03055,  0.     ,  0.     , -0.00178, -0.04932,  0.     ,  0.     ,  0.00231]
    ])
    M0e_tim_6 = np.array([
        [1.27790826, 0.        , 0.63895413, 0.        ],
        [0.        , 0.00039935, 0.        , 0.00019967],
        [0.63895413, 0.        , 1.27790826, 0.        ],
        [0.        , 0.00019967, 0.        , 0.00039935]
    ])

    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_almost_equal(tim.M()[np.ix_(dofs_4, dofs_4)], M0e_tim_4, decimal=5)
    assert_almost_equal(tim.M()[np.ix_(dofs_6, dofs_6)], M0e_tim_6, decimal=5)


def test_stiffness_matrix_tim(tim):
    # fmt: off
    K0e_tim_4 = np.array([
        [ 4.56964,  0.     ,  0.     ,  0.57121, -4.56964,  0.     ,  0.     ,  0.57121],
        [ 0.     ,  4.56964, -0.57121,  0.     ,  0.     , -4.56964, -0.57121,  0.     ],
        [ 0.     , -0.57121,  0.09729,  0.     ,  0.     ,  0.57121,  0.04551,  0.     ],
        [ 0.57121,  0.     ,  0.     ,  0.09729, -0.57121,  0.     ,  0.     ,  0.04551],
        [-4.56964,  0.     ,  0.     , -0.57121,  4.56964,  0.     ,  0.     , -0.57121],
        [ 0.     , -4.56964,  0.57121,  0.     ,  0.     ,  4.56964,  0.57121,  0.     ],
        [ 0.     , -0.57121,  0.04551,  0.     ,  0.     ,  0.57121,  0.09729,  0.     ],
        [ 0.57121,  0.     ,  0.     ,  0.04551, -0.57121,  0.     ,  0.     ,  0.09729]
    ])
    K0e_tim_6 = np.array([
        [ 165.71901,    0.     , -165.71901,    0.     ],
        [   0.     ,    0.01993,    0.     ,   -0.01993],
        [-165.71901,    0.     ,  165.71901,    0.     ],
        [   0.     ,   -0.01993,    0.     ,    0.01993]
    ])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_almost_equal(tim.K()[np.ix_(dofs_4, dofs_4)] / 1e7, K0e_tim_4, decimal=5)
    assert_almost_equal(tim.K()[np.ix_(dofs_6, dofs_6)] / 1e7, K0e_tim_6, decimal=5)


def test_gyroscopic_matrix_tim(tim):
    # fmt: off
    G0e_tim_4 = np.array([
        [ -0.     ,  19.43344,  -0.22681,  -0.     ,  -0.     , -19.43344,  -0.22681,  -0.     ],
        [-19.43344,  -0.     ,  -0.     ,  -0.22681,  19.43344,  -0.     ,  -0.     ,  -0.22681],
        [  0.22681,  -0.     ,  -0.     ,   0.1524 ,  -0.22681,  -0.     ,  -0.     ,  -0.04727],
        [ -0.     ,   0.22681,  -0.1524 ,  -0.     ,  -0.     ,  -0.22681,   0.04727,  -0.     ],
        [ -0.     , -19.43344,   0.22681,  -0.     ,  -0.     ,  19.43344,   0.22681,  -0.     ],
        [ 19.43344,  -0.     ,  -0.     ,   0.22681, -19.43344,  -0.     ,  -0.     ,   0.22681],
        [  0.22681,  -0.     ,  -0.     ,  -0.04727,  -0.22681,  -0.     ,  -0.     ,   0.1524 ],
        [ -0.     ,   0.22681,   0.04727,  -0.     ,  -0.     ,  -0.22681,  -0.1524 ,  -0.     ]
    ])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_almost_equal(tim.G()[np.ix_(dofs_4, dofs_4)] * 1e3, G0e_tim_4, decimal=5)
    assert_almost_equal(
        tim.G()[np.ix_(dofs_6, dofs_6)] * 1e3, np.zeros((4, 4)), decimal=5
    )


def test_from_table():
    for shaft_file in [
        os.path.dirname(os.path.realpath(__file__)) + "/data/shaft_us.xls",
        os.path.dirname(os.path.realpath(__file__)) + "/data/shaft_si.xls",
    ]:
        shaft = ShaftElement.from_table(
            shaft_file, sheet_type="Model", sheet_name="Model"
        )
        el0 = shaft[0]
        assert el0.n == 0
        assert_allclose(el0.i_d, 0.1409954)
        assert_allclose(el0.o_d, 0.151003)

        mat0 = el0.material
        assert_allclose(mat0.rho, 7833.41, atol=0.003)
        assert_allclose(mat0.E, 206842718795.05, rtol=3e-06)
        assert_allclose(mat0.G_s, 82737087518.02, rtol=2e-06)

        # test if node is the same for elements in different layers
        assert shaft[8].n == 8
        assert shaft[9].n == 8
        assert_allclose(shaft[8].material.E, 206842718795.05, rtol=3e-06)
        assert_allclose(shaft[9].material.E, 6894.75, atol=0.008)


# Shaft Tapered Element tests
@pytest.fixture
def tap_tim():
    #  Timoshenko element
    L = 0.4
    i_d_l = 0.0
    i_d_r = 0.0
    o_d_l = 0.25
    o_d_r = 0.10

    return ShaftElement(
        L,
        i_d_l,
        o_d_l,
        i_d_r,
        o_d_r,
        steel,
        shear_effects=True,
        rotary_inertia=True,
        n=3,
    )


def test_tapered_index(tap_tim):
    assert tap_tim.dof_local_index().x_0 == 0
    assert tap_tim.dof_local_index().y_0 == 1
    assert tap_tim.dof_local_index().z_0 == 2
    assert tap_tim.dof_local_index().alpha_0 == 3
    assert tap_tim.dof_local_index().beta_0 == 4
    assert tap_tim.dof_local_index().theta_0 == 5
    assert tap_tim.dof_local_index().x_1 == 6
    assert tap_tim.dof_local_index().y_1 == 7
    assert tap_tim.dof_local_index().z_1 == 8
    assert tap_tim.dof_local_index().alpha_1 == 9
    assert tap_tim.dof_local_index().beta_1 == 10
    assert tap_tim.dof_local_index().theta_1 == 11


def test_parameters_tap_tim(tap_tim):
    assert tap_tim.L == 0.4
    assert tap_tim.i_d == 0.0
    assert tap_tim.idl == 0.0
    assert tap_tim.idr == 0.0
    assert tap_tim.o_d == 0.175
    assert tap_tim.odl == 0.25
    assert tap_tim.odr == 0.1
    assert tap_tim.material.E == 211e9
    assert tap_tim.material.G_s == 81.2e9
    assert tap_tim.material.rho == 7810
    assert_almost_equal(tap_tim.material.Poisson, 0.29926108)
    assert_almost_equal(tap_tim.A, 0.024052818754)
    assert_almost_equal(tap_tim.Ie * 1e5, 4.60385984)
    assert_almost_equal(tap_tim.phi, 0.4208816002)


def test_mass_matrix_tap_tim(tap_tim):
    # fmt: off
    M0e_tim_4 = np.array([
        [41.84146162,  0.        ,  0.        ,  1.85735573, 10.27465893,  0.        ,  0.        , -1.16147981],
        [ 0.        , 41.84146162, -1.85735573,  0.        ,  0.        , 10.27465893,  1.16147981,  0.        ],
        [ 0.        , -1.85735573,  0.18686024,  0.        ,  0.        , -1.04154454, -0.10311193,  0.        ],
        [ 1.85735573,  0.        ,  0.        ,  0.18686024,  1.04154454,  0.        ,  0.        , -0.10311193],
        [10.27465893,  0.        ,  0.        ,  1.04154454, 17.35069604,  0.        ,  0.        , -1.1330391 ],
        [ 0.        , 10.27465893, -1.04154454,  0.        ,  0.        , 17.35069604,  1.1330391 ,  0.        ],
        [ 0.        ,  1.16147981, -0.10311193,  0.        ,  0.        ,  1.1330391 ,  0.09851808,  0.        ],
        [-1.16147981,  0.        ,  0.        , -0.10311193, -1.1330391 ,  0.        ,  0.        ,  0.09851808]
    ])
    M0e_tim_6 = np.array([
        [29.64747167,  0.        , 14.82373584,  0.        ],
        [ 0.        ,  0.2047848 ,  0.        ,  0.1023924 ],
        [14.82373584,  0.        , 29.64747167,  0.        ],
        [ 0.        ,  0.1023924 ,  0.        ,  0.2047848 ]
    ])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_almost_equal(tap_tim.M()[np.ix_(dofs_4, dofs_4)], M0e_tim_4, decimal=5)
    assert_almost_equal(tap_tim.M()[np.ix_(dofs_6, dofs_6)], M0e_tim_6, decimal=5)


def test_stiffness_matrix_tap_tim(tap_tim):
    # fmt: off
    K0e_tim_4 = np.array([
        [1.91471213e+09, 0.00000000e+00, 0.00000000e+00, 5.45588167e+08, -1.91471213e+09, 0.00000000e+00, 0.00000000e+00, 2.20296684e+08],
        [0.00000000e+00, 1.91471213e+09, -5.45588167e+08, 0.00000000e+00, 0.00000000e+00, -1.91471213e+09, -2.20296684e+08, 0.00000000e+00],
        [0.00000000e+00, -5.45588167e+08, 1.75017153e+08, 0.00000000e+00, 0.00000000e+00, 5.45588167e+08, 4.32181136e+07, 0.00000000e+00],
        [5.45588167e+08, 0.00000000e+00, 0.00000000e+00, 1.75017153e+08, -5.45588167e+08, 0.00000000e+00, 0.00000000e+00, 4.32181136e+07],
        [-1.91471213e+09, 0.00000000e+00, 0.00000000e+00, -5.45588167e+08, 1.91471213e+09, 0.00000000e+00, 0.00000000e+00, -2.20296684e+08],
        [0.00000000e+00, -1.91471213e+09, 5.45588167e+08, 0.00000000e+00, 0.00000000e+00, 1.91471213e+09, 2.20296684e+08, 0.00000000e+00],
        [0.00000000e+00, -2.20296684e+08, 4.32181136e+07, 0.00000000e+00, 0.00000000e+00, 2.20296684e+08, 4.49005599e+07, 0.00000000e+00],
        [2.20296684e+08, 0.00000000e+00, 0.00000000e+00, 4.32181136e+07, -2.20296684e+08, 0.00000000e+00, 0.00000000e+00, 4.49005599e+07]
    ])
    K0e_tim_6 = np.array([
        [ 1501.82855,     0.     , -1501.82855,     0.     ],
        [    0.     ,     3.99212,     0.     ,    -3.99212],
        [-1501.82855,     0.     ,  1501.82855,     0.     ],
        [    0.     ,    -3.99212,     0.     ,     3.99212]
    ])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_allclose(tap_tim.K()[np.ix_(dofs_4, dofs_4)], K0e_tim_4)
    assert_almost_equal(tap_tim.K()[np.ix_(dofs_6, dofs_6)] / 1e7, K0e_tim_6, decimal=5)


def test_gyroscopic_matrix_tap_tim(tap_tim):
    # fmt: off
    G0e_tim_4 = np.array([
            [ 0.        ,  1.23853234,  0.15544731,  0.        ,  0.        , -1.23853234, -0.03173553,  0.        ],
            [-1.23853234,  0.        ,  0.        ,  0.15544731,  1.23853234,  0.        ,  0.        , -0.03173553],
            [-0.15544731,  0.        ,  0.        ,  0.11850408,  0.15544731,  0.        ,  0.        , -0.01563679],
            [ 0.        , -0.15544731, -0.11850408,  0.        ,  0.        ,  0.15544731,  0.01563679,  0.        ],
            [ 0.        , -1.23853234, -0.15544731,  0.        ,  0.        ,  1.23853234,  0.03173553,  0.        ],
            [ 1.23853234,  0.        ,  0.        , -0.15544731, -1.23853234,  0.        ,  0.        ,  0.03173553],
            [ 0.03173553,  0.        ,  0.        , -0.01563679, -0.03173553,  0.        ,  0.        ,  0.01089192],
            [ 0.        ,  0.03173553,  0.01563679,  0.        ,  0.        , -0.03173553, -0.01089192,  0.        ]])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_almost_equal(tap_tim.G()[np.ix_(dofs_4, dofs_4)], G0e_tim_4, decimal=5)
    assert_almost_equal(
        tap_tim.G()[np.ix_(dofs_6, dofs_6)], np.zeros((4, 4)), decimal=5
    )


@pytest.fixture
def tap_tim_hollow():
    #  Timoshenko element
    L = 0.4
    i_d_l = 0.05
    i_d_r = 0.05
    o_d_l = 0.25
    o_d_r = 0.10

    return ShaftElement(
        L,
        i_d_l,
        o_d_l,
        i_d_r,
        o_d_r,
        steel,
        shear_effects=True,
        rotary_inertia=True,
        n=3,
    )


def test_mass_matrix_tap_tim_hollow(tap_tim_hollow):
    # fmt: off
    M0e_tim_4 = np.array([
        [39.26343318, 0., 0., 1.701278, 9.66844022, 0., 0., -1.11365499],
        [0., 39.26343318, -1.701278, 0., 0., 9.66844022, 1.11365499, 0.],
        [0., -1.701278, 0.17815913, 0., 0., -0.96970348, -0.09540863, 0.],
        [1.701278, 0., 0., 0.17815913, 0.96970348, 0., 0., -0.09540863],
        [9.66844022, 0., 0., 0.96970348, 15.00720225, 0., 0., -0.99985206],
        [0., 9.66844022, -0.96970348, 0., 0., 15.00720225, 0.99985206, 0.],
        [0., 1.11365499, -0.09540863, 0., 0., 0.99985206, 0.08959186, 0.],
        [-1.11365499, 0., 0., -0.09540863, -0.99985206, 0., 0., 0.08959186]
    ])
    M0e_tim_6 = np.array([
        [27.60282,  0.     , 13.80141,  0.     ],
        [ 0.     ,  0.20415,  0.     ,  0.10207],
        [13.80141,  0.     , 27.60282,  0.     ],
        [ 0.     ,  0.10207,  0.     ,  0.20415]
    ])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_almost_equal(
        tap_tim_hollow.M()[np.ix_(dofs_4, dofs_4)], M0e_tim_4, decimal=5
    )
    assert_almost_equal(
        tap_tim_hollow.M()[np.ix_(dofs_6, dofs_6)], M0e_tim_6, decimal=5
    )


def test_stiffness_matrix_tap_tim_hollow(tap_tim_hollow):
    # fmt: off
    K0e_tim_4 = np.array([
        [1.72061778e+09, 0.00000000e+00, 0.00000000e+00, 4.94146120e+08, -1.72061778e+09, 0.00000000e+00, 0.00000000e+00, 1.94100990e+08],
        [0.00000000e+00, 1.72061778e+09, -4.94146120e+08, 0.00000000e+00, 0.00000000e+00, -1.72061778e+09, -1.94100990e+08, 0.00000000e+00],
        [0.00000000e+00, -4.94146120e+08, 1.62042273e+08, 0.00000000e+00, 0.00000000e+00, 4.94146120e+08, 3.56161745e+07, 0.00000000e+00],
        [4.94146120e+08, 0.00000000e+00, 0.00000000e+00, 1.62042273e+08, -4.94146120e+08, 0.00000000e+00, 0.00000000e+00, 3.56161745e+07],
        [-1.72061778e+09, 0.00000000e+00, 0.00000000e+00, -4.94146120e+08, 1.72061778e+09, 0.00000000e+00, 0.00000000e+00, -1.94100990e+08],
        [0.00000000e+00, -1.72061778e+09, 4.94146120e+08, 0.00000000e+00, 0.00000000e+00, 1.72061778e+09, 1.94100990e+08, 0.00000000e+00],
        [0.00000000e+00, -1.94100990e+08, 3.56161745e+07, 0.00000000e+00, 0.00000000e+00, 1.94100990e+08, 4.20242215e+07, 0.00000000e+00],
        [1.94100990e+08, 0.00000000e+00, 0.00000000e+00,3.56161745e+07, -1.94100990e+08, 0.00000000e+00, 0.00000000e+00, 4.20242215e+07]
    ])
    K0e_tim_6 = np.array([
        [ 1398.25417,     0.     , -1398.25417,     0.     ],
        [    0.     ,     3.97967,     0.     ,    -3.97967],
        [-1398.25417,     0.     ,  1398.25417,     0.     ],
        [    0.     ,    -3.97967,     0.     ,     3.97967]
    ])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_allclose(tap_tim_hollow.K()[np.ix_(dofs_4, dofs_4)], K0e_tim_4)
    assert_almost_equal(
        tap_tim_hollow.K()[np.ix_(dofs_6, dofs_6)] / 1e7, K0e_tim_6, decimal=5
    )


def test_gyroscopic_matrix_tap_tim_hollow(tap_tim_hollow):
    # fmt: off
    G0e_tim_4 = np.array([
        [0., 1.04768554, 0.16077186, 0., 0., -1.04768554, -0.01188345, 0.],
        [-1.04768554, 0., 0., 0.16077186, 1.04768554, 0., 0., -0.01188345],
        [-0.16077186, 0., 0., 0.12336918, 0.16077186, 0., 0., -0.01335772],
        [0., -0.16077186, -0.12336918, 0., 0., 0.16077186, 0.01335772, 0.],
        [0., -1.04768554, -0.16077186, 0., 0., 1.04768554, 0.01188345, 0.],
        [1.04768554, 0., 0., -0.16077186, -1.04768554, 0., 0., 0.01188345],
        [0.01188345, 0., 0., -0.01335772, -0.01188345, 0., 0., 0.00994601],
        [0., 0.01188345, 0.01335772, 0., 0., -0.01188345, -0.00994601, 0.]
    ])
    # fmt: on
    dofs_4 = [0, 1, 3, 4, 6, 7, 9, 10]
    dofs_6 = [2, 5, 8, 11]
    assert_almost_equal(
        tap_tim_hollow.G()[np.ix_(dofs_4, dofs_4)], G0e_tim_4, decimal=5
    )
    assert_almost_equal(
        tap_tim_hollow.G()[np.ix_(dofs_6, dofs_6)], np.zeros((4, 4)), decimal=5
    )


@pytest.fixture
def tap2():
    #  Timoshenko element
    L = 0.4
    i_d_l = 0.0
    i_d_r = 0.0
    o_d_l = 0.25
    o_d_r = 0.25

    return ShaftElement(
        L, i_d_l, o_d_l, i_d_r, o_d_r, steel, shear_effects=True, rotary_inertia=True
    )


@pytest.fixture
def tim2():
    #  Timoshenko element
    le_ = 0.4
    i_d_ = 0
    o_d_ = 0.25
    return ShaftElement(
        le_, i_d_, o_d_, material=steel, rotary_inertia=True, shear_effects=True
    )


def test_match_mass_matrix(tap2, tim2):
    M_tap = tap2.M()
    M_tim = tim2.M()
    assert_almost_equal(M_tap, M_tim, decimal=5)


def test_match_stiffness_matrix(tap2, tim2):
    K_tap = tap2.K()
    K_tim = tim2.K()
    assert_almost_equal(K_tap, K_tim, decimal=5)


def test_match_gyroscopic_matrix(tap2, tim2):
    G_tap = tap2.G()
    G_tim = tim2.G()
    assert_almost_equal(G_tap, G_tim, decimal=5)


def test_pickle(tim):
    tim_pickled = pickle.loads(pickle.dumps(tim))
    assert tim == tim_pickled


def test_save_load(tim, tap_tim):
    file = Path(tempdir) / "tim.toml"
    tim.save(file)
    tim_loaded = ShaftElement.load(file)
    assert tim == tim_loaded

    file = Path(tempdir) / "tap_tim.toml"
    tap_tim.save(file)
    tap_tim_loaded = ShaftElement.load(file)
    assert tap_tim == tap_tim_loaded


def test_save_load_json(tim, tap_tim):
    file = Path(tempdir) / "tim.json"
    tim.save(file)
    tim_loaded = ShaftElement.load(file)
    assert tim == tim_loaded

    file = Path(tempdir) / "tap_tim.json"
    tap_tim.save(file)
    tap_tim_loaded = ShaftElement.load(file)
    assert tap_tim == tap_tim_loaded


@pytest.fixture
def tim_6dof():
    #  Timoshenko element
    L = 0.25
    i_d = 0
    o_d = 0.05
    return ShaftElement(L, i_d, o_d, material=steel, alpha=1, beta=1e-5)


def test_parameters_tim_6dof(tim_6dof):
    assert tim_6dof.L == 0.25
    assert tim_6dof.idl == 0
    assert tim_6dof.odl == 0.05
    assert tim_6dof.idr == 0
    assert tim_6dof.odr == 0.05
    assert tim_6dof.material.E == 211e9
    assert tim_6dof.material.G_s == 81.2e9
    assert tim_6dof.material.rho == 7810
    assert_almost_equal(tim_6dof.phi, 0.08795566)
    assert_almost_equal(tim_6dof.material.Poisson, 0.29926108)
    assert_almost_equal(tim_6dof.A, 0.00196349)
    assert_almost_equal(tim_6dof.Ie * 1e7, 3.06796157)


def test_mass_matrix_tim_6dof(tim_6dof):
    # fmt: off
    M0e_tim = np.array([
        [ 1.42051,  0.00000,  0.00000,  0.00000,  0.04932,  0.0000,  0.49635,  0.00000,  0.00000,  0.00000, -0.03055,  0.0000],
        [ 0.00000,  1.42051,  0.00000, -0.04932,  0.00000,  0.0000,  0.00000,  0.49635,  0.00000,  0.03055,  0.00000,  0.0000],
        [ 0.00000,  0.00000,  1.27791,  0.00000,  0.00000,  0.0000,  0.00000,  0.00000,  0.63895,  0.00000,  0.00000,  0.0000],
        [ 0.00000, -0.04932,  0.00000,  0.00231,  0.00000,  0.0000,  0.00000, -0.03055,  0.00000, -0.00178,  0.00000,  0.0000],
        [ 0.04932,  0.00000,  0.00000,  0.00000,  0.00231,  0.0000,  0.03055,  0.00000,  0.00000,  0.00000, -0.00178,  0.0000],
        [ 0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.0004,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.0002],
        [ 0.49635,  0.00000,  0.00000,  0.00000,  0.03055,  0.0000,  1.42051,  0.00000,  0.00000,  0.00000, -0.04932,  0.0000],
        [ 0.00000,  0.49635,  0.00000, -0.03055,  0.00000,  0.0000,  0.00000,  1.42051,  0.00000,  0.04932,  0.00000,  0.0000],
        [ 0.00000,  0.00000,  0.63895,  0.00000,  0.00000,  0.0000,  0.00000,  0.00000,  1.27791,  0.00000,  0.00000,  0.0000],
        [ 0.00000,  0.03055,  0.00000, -0.00178,  0.00000,  0.0000,  0.00000,  0.04932,  0.00000,  0.00231,  0.00000,  0.0000],
        [-0.03055,  0.00000,  0.00000,  0.00000, -0.00178,  0.0000, -0.04932,  0.00000,  0.00000,  0.00000,  0.00231,  0.0000],
        [ 0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.0002,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  0.0004],
])
    # fmt: on
    assert_almost_equal(tim_6dof.M(), M0e_tim, decimal=5)


def test_stiffness_matrix_tim_6dof(tim_6dof):
    # fmt: off
    K0e_tim = np.array([
        [ 0.45696,  0.00000,   0.0000,  0.00000,  0.05712,  0.00000, -0.45696,  0.00000,   0.0000,  0.00000,  0.05712,  0.00000],
        [ 0.00000,  0.45696,   0.0000, -0.05712,  0.00000,  0.00000,  0.00000, -0.45696,   0.0000, -0.05712,  0.00000,  0.00000],
        [ 0.00000,  0.00000,  16.5719,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -16.5719,  0.00000,  0.00000,  0.00000],
        [ 0.00000, -0.05712,   0.0000,  0.00973,  0.00000,  0.00000,  0.00000,  0.05712,   0.0000,  0.00455,  0.00000,  0.00000],
        [ 0.05712,  0.00000,   0.0000,  0.00000,  0.00973,  0.00000, -0.05712,  0.00000,   0.0000,  0.00000,  0.00455,  0.00000],
        [ 0.00000,  0.00000,   0.0000,  0.00000,  0.00000,  0.00199,  0.00000,  0.00000,   0.0000,  0.00000,  0.00000, -0.00199],
        [-0.45696,  0.00000,   0.0000,  0.00000, -0.05712,  0.00000,  0.45696,  0.00000,   0.0000,  0.00000, -0.05712,  0.00000],
        [ 0.00000, -0.45696,   0.0000,  0.05712,  0.00000,  0.00000,  0.00000,  0.45696,   0.0000,  0.05712,  0.00000,  0.00000],
        [ 0.00000,  0.00000, -16.5719,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  16.5719,  0.00000,  0.00000,  0.00000],
        [ 0.00000, -0.05712,   0.0000,  0.00455,  0.00000,  0.00000,  0.00000,  0.05712,   0.0000,  0.00973,  0.00000,  0.00000],
        [ 0.05712,  0.00000,   0.0000,  0.00000,  0.00455,  0.00000, -0.05712,  0.00000,   0.0000,  0.00000,  0.00973,  0.00000],
        [ 0.00000,  0.00000,   0.0000,  0.00000,  0.00000, -0.00199,  0.00000,  0.00000,   0.0000,  0.00000,  0.00000,  0.00199],
])
    # fmt: on
    assert_almost_equal(tim_6dof.K() / 1e8, K0e_tim, decimal=5)


def test_dynamic_stiffness_matrix_tim_6dof(tim_6dof):
    # fmt: off
    Kst0e_tim = np.array([
        [0.0, -23.00235,  0.0,  0.47922,  0.0,  0.0,  0.0,  23.00235,  0.0,  0.47922,  0.0,  0.0],
        [0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0,  0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0],
        [0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0,  0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0],
        [0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0,  0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0],
        [0.0 , -0.47922,  0.0,  0.15974,  0.0,  0.0,  0.0,   0.47922,  0.0, -0.03993,  0.0,  0.0],
        [0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0,  0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0],
        [0.0,  23.00235,  0.0, -0.47922,  0.0,  0.0,  0.0, -23.00235,  0.0, -0.47922,  0.0,  0.0],
        [0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0,  0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0],
        [0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0,  0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0],
        [0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0,  0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0],
        [0.0 , -0.47922,  0.0, -0.03993,  0.0,  0.0,  0.0,   0.47922,  0.0,  0.15974,  0.0,  0.0],
        [0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0,  0.0,   0.00000,  0.0,  0.00000,  0.0,  0.0],
])
    # fmt: on
    assert_almost_equal(tim_6dof.Kst() * 1e3, Kst0e_tim, decimal=5)


def test_damping_matrix_tim_6dof(tim_6dof):
    # fmt: off
    C0e_tim = np.array([
        [ 0.45838,  0.00000,   0.00000,  0.00000,  0.05717,  0.00000, -0.45647,  0.00000,   0.00000,  0.00000,  0.05709,  0.00000],
        [ 0.00000,  0.45838,   0.00000, -0.05717,  0.00000,  0.00000,  0.00000, -0.45647,   0.00000, -0.05709,  0.00000,  0.00000],
        [ 0.00000,  0.00000,  16.57318,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000, -16.57126,  0.00000,  0.00000,  0.00000],
        [ 0.00000, -0.05717,   0.00000,  0.00973,  0.00000,  0.00000,  0.00000,  0.05709,   0.00000,  0.00455,  0.00000,  0.00000],
        [ 0.05717,  0.00000,   0.00000,  0.00000,  0.00973,  0.00000, -0.05709,  0.00000,   0.00000,  0.00000,  0.00455,  0.00000],
        [ 0.00000,  0.00000,   0.00000,  0.00000,  0.00000,  0.00199,  0.00000,  0.00000,   0.00000,  0.00000,  0.00000, -0.00199],
        [-0.45647,  0.00000,   0.00000,  0.00000, -0.05709,  0.00000,  0.45838,  0.00000,   0.00000,  0.00000, -0.05717,  0.00000],
        [ 0.00000, -0.45647,   0.00000,  0.05709,  0.00000,  0.00000,  0.00000,  0.45838,   0.00000,  0.05717,  0.00000,  0.00000],
        [ 0.00000,  0.00000, -16.57126,  0.00000,  0.00000,  0.00000,  0.00000,  0.00000,  16.57318,  0.00000,  0.00000,  0.00000],
        [ 0.00000, -0.05709,   0.00000,  0.00455,  0.00000,  0.00000,  0.00000,  0.05717,   0.00000,  0.00973,  0.00000,  0.00000],
        [ 0.05709,  0.00000,   0.00000,  0.00000,  0.00455,  0.00000, -0.05717,  0.00000,   0.00000,  0.00000,  0.00973,  0.00000],
        [ 0.00000,  0.00000,   0.00000,  0.00000,  0.00000, -0.00199,  0.00000,  0.00000,   0.00000,  0.00000,  0.00000,  0.00199],
])
    # fmt: on
    assert_almost_equal(tim_6dof.C() / 1e3, C0e_tim, decimal=5)


def test_gyroscopic_matrix_tim_6dof(tim_6dof):
    # fmt: off
    G0e_tim = np.array([
        [  0.00000,  19.43344,  0.0, -0.22681,  0.00000,  0.0,   0.00000, -19.43344,  0.0, -0.22681,  0.00000,  0.0],
        [-19.43344,   0.00000,  0.0,  0.00000, -0.22681,  0.0,  19.43344,   0.00000,  0.0,  0.00000, -0.22681,  0.0],
        [  0.00000,   0.00000,  0.0,  0.00000,  0.00000,  0.0,   0.00000,   0.00000,  0.0,  0.00000,  0.00000,  0.0],
        [  0.22681,   0.00000,  0.0,  0.00000,  0.15240,  0.0 , -0.22681,   0.00000,  0.0,  0.00000, -0.04727,  0.0],
        [  0.00000,   0.22681,  0.0, -0.15240,  0.00000,  0.0,   0.00000 , -0.22681,  0.0,  0.04727,  0.00000,  0.0],
        [  0.00000,   0.00000,  0.0,  0.00000,  0.00000,  0.0,   0.00000,   0.00000,  0.0,  0.00000,  0.00000,  0.0],
        [  0.00000, -19.43344,  0.0,  0.22681,  0.00000,  0.0,   0.00000,  19.43344,  0.0,  0.22681,  0.00000,  0.0],
        [ 19.43344,   0.00000,  0.0,  0.00000,  0.22681,  0.0, -19.43344,   0.00000,  0.0,  0.00000,  0.22681,  0.0],
        [  0.00000,   0.00000,  0.0,  0.00000,  0.00000,  0.0,   0.00000,   0.00000,  0.0,  0.00000,  0.00000,  0.0],
        [  0.22681,   0.00000,  0.0,  0.00000, -0.04727,  0.0 , -0.22681,   0.00000,  0.0,  0.00000,  0.15240,  0.0],
        [  0.00000,   0.22681,  0.0,  0.04727,  0.00000,  0.0,   0.00000 , -0.22681,  0.0, -0.15240,  0.00000,  0.0],
        [  0.00000,   0.00000,  0.0,  0.00000,  0.00000,  0.0,   0.00000,   0.00000,  0.0,  0.00000,  0.00000,  0.0],
])
    # fmt: on
    assert_almost_equal(tim_6dof.G() * 1e3, G0e_tim, decimal=5)


@pytest.fixture
def coupling():
    mass_station = 151 / 2  # Assuming equal distribution of mass (in kg)
    Ip_station = 1.74  # Polar moment of inertia, assuming equal distribution (in kg·m²)

    return CouplingElement(
        m_l=mass_station,
        m_r=mass_station,
        Ip_l=Ip_station,
        Ip_r=Ip_station,
        kt_x=1e6,
        kt_y=2e6,
        kt_z=3e6,  # Axial stiffness (in N/m)
        kr_x=4e6,
        kr_y=5e6,
        kr_z=6e6,  # Torsional stiffness (in N·m/rad)
    )


def test_parameters_coupling(coupling):
    assert coupling.m == 151
    assert coupling.Ip == 3.48
    assert coupling.Id_l == 0.87
    assert coupling.Id_r == 0.87
    assert coupling.kt_x == 1e6
    assert coupling.kt_y == 2e6
    assert coupling.kt_z == 3e6
    assert coupling.kr_x == 4e6
    assert coupling.kr_y == 5e6
    assert coupling.kr_z == 6e6
    assert coupling.ct_x == 0.0
    assert coupling.ct_y == 0.0
    assert coupling.ct_z == 0.0
    assert coupling.cr_x == 0.0
    assert coupling.cr_y == 0.0
    assert coupling.cr_z == 0.0


def test_mass_matrix_coupling(coupling):
    # fmt: off
    M0 = np.array([
        [75.5 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  , 75.5 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  , 75.5 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.87,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.87,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.74,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , 75.5 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , 75.5 ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , 75.5 ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.87,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.87,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.74]
    ])
    # fmt: on
    assert_almost_equal(coupling.M(), M0, decimal=5)


def test_stiffness_matrix_coupling(coupling):
    # fmt: off
    K0 = np.array([
        [ 1.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  2.,  0.,  0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  3.,  0.,  0.,  0.,  0.,  0., -3.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.,  0., -4.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  5.,  0.,  0.,  0.,  0.,  0., -5.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  6.,  0.,  0.,  0.,  0.,  0., -6.],
        [-1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -2.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.],
        [ 0.,  0., -3.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,  0.],
        [ 0.,  0.,  0., -4.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,  0.],
        [ 0.,  0.,  0.,  0., -5.,  0.,  0.,  0.,  0.,  0.,  5.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -6.,  0.,  0.,  0.,  0.,  0.,  6.]
    ])
    # fmt: on
    assert_almost_equal(coupling.K() / 1e6, K0, decimal=5)


def test_dynamic_stiffness_matrix_coupling(coupling):
    Kst0 = np.zeros((12, 12))
    assert_almost_equal(coupling.Kst(), Kst0, decimal=5)


def test_damping_matrix_coupling(coupling):
    C0 = np.zeros((12, 12))
    assert_almost_equal(coupling.C(), C0, decimal=5)


def test_gyroscopic_matrix_coupling(coupling):
    # fmt: off
    G0 = np.array([
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  1.74,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  , -1.74,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.74,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -1.74,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]
    ])
    # fmt: on
    assert_almost_equal(coupling.G(), G0, decimal=5)


def test_pickle_coupling(coupling):
    coupling_pickled = pickle.loads(pickle.dumps(coupling))
    assert coupling == coupling_pickled


def test_save_load_coupling(coupling):
    file = Path(tempdir) / "coupling.toml"
    coupling.save(file)
    coupling_loaded = CouplingElement.load(file)
    assert coupling == coupling_loaded


def test_save_load_coupling_json(coupling):
    file = Path(tempdir) / "coupling.json"
    coupling.save(file)
    coupling_loaded = CouplingElement.load(file)
    assert coupling == coupling_loaded
