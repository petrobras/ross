import os

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from ross.materials import steel
from ross.shaft_element import ShaftElement, ShaftTaperedElement

test_dir = os.path.dirname(__file__)


@pytest.fixture
def eb():
    #  Euler-Bernoulli element
    le_ = 0.25
    i_d_ = 0
    o_d_ = 0.05
    return ShaftElement(
        le_, i_d_, o_d_, steel, shear_effects=False, rotary_inertia=False, n=3
    )


def test_index(eb):
    assert eb.dof_local_index().x0 == 0
    assert eb.dof_local_index().y0 == 1
    assert eb.dof_local_index().alpha0 == 2
    assert eb.dof_local_index().beta0 == 3
    assert eb.dof_global_index().x0 == 12
    assert eb.dof_global_index().y0 == 13
    assert eb.dof_global_index().alpha0 == 14
    assert eb.dof_global_index().beta0 == 15


def test_parameters_eb(eb):
    assert eb.phi == 0
    assert eb.L == 0.25
    assert eb.i_d == 0
    assert eb.o_d == 0.05
    assert eb.material.E == 211e9
    assert eb.material.G_s == 81.2e9
    assert eb.material.rho == 7810
    assert_almost_equal(eb.material.Poisson, 0.29926108)
    assert_almost_equal(eb.A, 0.00196349)
    assert_almost_equal(eb.Ie * 1e7, 3.06796157)


def test_mass_matrix_eb(eb):
    # fmt: off
    M0e_eb = np.array([[ 1.42395,  0.     ,  0.     ,  0.0502 ,  0.49291,  0.     ,  0.     , -0.02967],
                       [ 0.     ,  1.42395, -0.0502 ,  0.     ,  0.     ,  0.49291,  0.02967,  0.     ],
                       [ 0.     , -0.0502 ,  0.00228,  0.     ,  0.     , -0.02967, -0.00171,  0.     ],
                       [ 0.0502 ,  0.     ,  0.     ,  0.00228,  0.02967,  0.     ,  0.     , -0.00171],
                       [ 0.49291,  0.     ,  0.     ,  0.02967,  1.42395,  0.     ,  0.     , -0.0502 ],
                       [ 0.     ,  0.49291, -0.02967,  0.     ,  0.     ,  1.42395,  0.0502 ,  0.     ],
                       [ 0.     ,  0.02967, -0.00171,  0.     ,  0.     ,  0.0502 ,  0.00228,  0.     ],
                       [-0.02967,  0.     ,  0.     , -0.00171, -0.0502 ,  0.     ,  0.     ,  0.00228]])
    # fmt: on
    assert_allclose(eb.M(), M0e_eb, rtol=1e-3)


def test_stiffness_matrix_eb(eb):
    # fmt: off
    K0e_eb = np.array([[ 4.97157,  0.     ,  0.     ,  0.62145, -4.97157,  0.     ,  0.     ,  0.62145],
                       [ 0.     ,  4.97157, -0.62145,  0.     ,  0.     , -4.97157, -0.62145,  0.     ],
                       [ 0.     , -0.62145,  0.10357,  0.     ,  0.     ,  0.62145,  0.05179,  0.     ],
                       [ 0.62145,  0.     ,  0.     ,  0.10357, -0.62145,  0.     ,  0.     ,  0.05179],
                       [-4.97157,  0.     ,  0.     , -0.62145,  4.97157,  0.     ,  0.     , -0.62145],
                       [ 0.     , -4.97157,  0.62145,  0.     ,  0.     ,  4.97157,  0.62145,  0.     ],
                       [ 0.     , -0.62145,  0.05179,  0.     ,  0.     ,  0.62145,  0.10357,  0.     ],
                       [ 0.62145,  0.     ,  0.     ,  0.05179, -0.62145,  0.     ,  0.     ,  0.10357]])
    # fmt: on
    assert_almost_equal(eb.K() / 1e7, K0e_eb, decimal=5)


@pytest.fixture
def tim():
    #  Timoshenko element
    z_ = 0
    le_ = 0.25
    i_d_ = 0
    o_d_ = 0.05
    return ShaftElement(le_, i_d_, o_d_, steel, rotary_inertia=True, shear_effects=True)


def test_parameters_tim(tim):
    assert_almost_equal(tim.phi, 0.08795566)
    assert_almost_equal(tim.material.Poisson, 0.29926108)
    assert_almost_equal(tim.A, 0.00196349)
    assert_almost_equal(tim.Ie * 1e7, 3.06796157)


def test_mass_matrix_tim(tim):
    # fmt: off
    M0e_tim = np.array([[ 1.42051,  0.     ,  0.     ,  0.04932,  0.49635,  0.     ,  0.     , -0.03055],
                        [ 0.     ,  1.42051, -0.04932,  0.     ,  0.     ,  0.49635,  0.03055,  0.     ],
                        [ 0.     , -0.04932,  0.00231,  0.     ,  0.     , -0.03055, -0.00178,  0.     ],
                        [ 0.04932,  0.     ,  0.     ,  0.00231,  0.03055,  0.     ,  0.     , -0.00178],
                        [ 0.49635,  0.     ,  0.     ,  0.03055,  1.42051,  0.     ,  0.     , -0.04932],
                        [ 0.     ,  0.49635, -0.03055,  0.     ,  0.     ,  1.42051,  0.04932,  0.     ],
                        [ 0.     ,  0.03055, -0.00178,  0.     ,  0.     ,  0.04932,  0.00231,  0.     ],
                        [-0.03055,  0.     ,  0.     , -0.00178, -0.04932,  0.     ,  0.     ,  0.00231]])
    # fmt: on
    assert_almost_equal(tim.M(), M0e_tim, decimal=5)


def test_stiffness_matrix_tim(tim):
    # fmt: off
    K0e_tim = np.array([[ 4.56964,  0.     ,  0.     ,  0.57121, -4.56964,  0.     ,  0.     ,  0.57121],
                        [ 0.     ,  4.56964, -0.57121,  0.     ,  0.     , -4.56964, -0.57121,  0.     ],
                        [ 0.     , -0.57121,  0.09729,  0.     ,  0.     ,  0.57121,  0.04551,  0.     ],
                        [ 0.57121,  0.     ,  0.     ,  0.09729, -0.57121,  0.     ,  0.     ,  0.04551],
                        [-4.56964,  0.     ,  0.     , -0.57121,  4.56964,  0.     ,  0.     , -0.57121],
                        [ 0.     , -4.56964,  0.57121,  0.     ,  0.     ,  4.56964,  0.57121,  0.     ],
                        [ 0.     , -0.57121,  0.04551,  0.     ,  0.     ,  0.57121,  0.09729,  0.     ],
                        [ 0.57121,  0.     ,  0.     ,  0.04551, -0.57121,  0.     ,  0.     ,  0.09729]])
    # fmt: on
    assert_almost_equal(tim.K() / 1e7, K0e_tim, decimal=5)


def test_gyroscopic_matrix_tim(tim):
    # fmt: off
    G0e_tim = np.array([[ -0.     ,  19.43344,  -0.22681,  -0.     ,  -0.     , -19.43344,  -0.22681,  -0.     ],
                        [-19.43344,  -0.     ,  -0.     ,  -0.22681,  19.43344,  -0.     ,  -0.     ,  -0.22681],
                        [  0.22681,  -0.     ,  -0.     ,   0.1524 ,  -0.22681,  -0.     ,  -0.     ,  -0.04727],
                        [ -0.     ,   0.22681,  -0.1524 ,  -0.     ,  -0.     ,  -0.22681,   0.04727,  -0.     ],
                        [ -0.     , -19.43344,   0.22681,  -0.     ,  -0.     ,  19.43344,   0.22681,  -0.     ],
                        [ 19.43344,  -0.     ,  -0.     ,   0.22681, -19.43344,  -0.     ,  -0.     ,   0.22681],
                        [  0.22681,  -0.     ,  -0.     ,  -0.04727,  -0.22681,  -0.     ,  -0.     ,   0.1524 ],
                        [ -0.     ,   0.22681,   0.04727,  -0.     ,  -0.     ,  -0.22681,  -0.1524 ,  -0.     ]])
    # fmt: on
    assert_almost_equal(tim.G() * 1e3, G0e_tim, decimal=5)


def test_from_table():
    for shaft_file in [os.path.dirname(os.path.realpath(__file__)) + "/data/shaft_us.xls",
                       os.path.dirname(os.path.realpath(__file__)) + "/data/shaft_si.xls"]:
        shaft = ShaftElement.from_table(shaft_file, sheet="Model")
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
def tap_eb():
    #  Euler-Bernoulli element
    L = 0.4
    i_d_l = 0.
    i_d_r = 0.
    o_d_l = 0.25
    o_d_r = 0.10

    return ShaftTaperedElement(
        L, i_d_l, i_d_r, o_d_l, o_d_r, steel, shear_effects=False, rotary_inertia=False, n=3
    )


def test_tapered_index(tap_eb):
    assert tap_eb.dof_local_index().x0 == 0
    assert tap_eb.dof_local_index().y0 == 1
    assert tap_eb.dof_local_index().alpha0 == 2
    assert tap_eb.dof_local_index().beta0 == 3
    assert tap_eb.dof_global_index().x0 == 12
    assert tap_eb.dof_global_index().y0 == 13
    assert tap_eb.dof_global_index().alpha0 == 14
    assert tap_eb.dof_global_index().beta0 == 15


def test_parameters_tap(tap_eb):
    assert tap_eb.phi == 0
    assert tap_eb.L == 0.4
    assert tap_eb.i_d == 0.0
    assert tap_eb.i_d_l == 0.0
    assert tap_eb.i_d_r == 0.0
    assert tap_eb.o_d == 0.175
    assert tap_eb.o_d_l == 0.25
    assert tap_eb.o_d_r == 0.1
    assert tap_eb.material.E == 211e9
    assert tap_eb.material.G_s == 81.2e9
    assert tap_eb.material.rho == 7810
    assert_almost_equal(tap_eb.material.Poisson, 0.29926108)
    assert_almost_equal(tap_eb.A, 0.024052818754)
    assert_almost_equal(tap_eb.Ie * 1e5, 4.44854429)


def test_mass_matrix_tap_eb(tap_eb):
    # fmt: off
    M0e_tap = np.array([[42.8500896 ,  0.        ,  0.        ,  2.13520215,  9.90196344,         0.        ,  0.        , -1.01356381],
                        [ 0.        , 42.8500896 , -2.13520215,  0.        ,  0.        ,         9.90196344,  1.01356381,  0.        ],
                        [ 0.        , -2.13520215,  0.14254154,  0.        ,  0.        ,        -0.89088462, -0.087628  ,  0.        ],
                        [ 2.13520215,  0.        ,  0.        ,  0.14254154,  0.89088462,         0.        ,  0.        , -0.087628  ],
                        [ 9.90196344,  0.        ,  0.        ,  0.89088462, 17.08745904,         0.        ,  0.        , -1.1537686 ],
                        [ 0.        ,  9.90196344, -0.89088462,  0.        ,  0.        ,        17.08745904,  1.1537686 ,  0.        ],
                        [ 0.        ,  1.01356381, -0.087628  ,  0.        ,  0.        ,         1.1537686 ,  0.09346986,  0.        ],
                        [-1.01356381,  0.        ,  0.        , -0.087628  , -1.1537686 ,         0.        ,  0.        ,  0.09346986]])
    # fmt: on
    assert_allclose(tap_eb.M(), M0e_tap, rtol=1e-3)


def test_stiffness_matrix_tap_eb(tap_eb):
    # fmt: off
    K0e_tap = np.array([[-484.29115593,    0.        ,    0.        ,   80.59381675,         484.29115593,    0.        ,    0.        ,   29.13029549],
                        [   0.        , -484.29115593,  -80.59381675,    0.        ,           0.        ,  484.29115593,  -29.13029549,    0.        ],
                        [   0.        ,  -80.59381675,   24.33997997,    0.        ,           0.        ,   80.59381675,    7.89754672,    0.        ],
                        [  80.59381675,    0.        ,    0.        ,   24.33997997,         -80.59381675,    0.        ,    0.        ,    7.89754672],
                        [ 484.29115593,    0.        ,    0.        ,  -80.59381675,        -484.29115593,    0.        ,    0.        ,  -29.13029549],
                        [   0.        ,  484.29115593,   80.59381675,    0.        ,           0.        , -484.29115593,   29.13029549,    0.        ],
                        [   0.        ,  -29.13029549,    7.89754672,    0.        ,           0.        ,   29.13029549,    3.75457147,    0.        ],
                        [  29.13029549,    0.        ,    0.        ,    7.89754672,         -29.13029549,    0.        ,    0.        ,    3.75457147]])
    # fmt: on
    assert_almost_equal(tap_eb.K() / 1e7, K0e_tap, decimal=5)


@pytest.fixture
def tap_tim():
    #  Timoshenko element
    L = 0.4
    i_d_l = 0.
    i_d_r = 0.
    o_d_l = 0.25
    o_d_r = 0.10

    return ShaftTaperedElement(
        L, i_d_l, i_d_r, o_d_l, o_d_r, steel, shear_effects=True, rotary_inertia=True
    )


def test_parameters_tap_tim(tap_tim):
    assert_almost_equal(tap_tim.phi, 0.40668276)
    assert_almost_equal(tap_tim.material.Poisson, 0.29926108)
    assert_almost_equal(tap_tim.A, 0.02405282)
    assert_almost_equal(tap_tim.Ie * 1e5, 4.44854429057887)


def test_mass_matrix_tap_tim(tap_tim):
    # fmt: off
    M0e_tim = np.array([[41.85725794,  0.        ,  0.        ,  1.8603991 , 10.27411395,         0.        ,  0.        , -1.14850813],
                        [ 0.        , 41.85725794, -1.8603991 ,  0.        ,  0.        ,        10.27411395,  1.14850813,  0.        ],
                        [ 0.        , -1.8603991 ,  0.18670638,  0.        ,  0.        ,        -1.04155143, -0.10238917,  0.        ],
                        [ 1.8603991 ,  0.        ,  0.        ,  0.18670638,  1.04155143,         0.        ,  0.        , -0.10238917],
                        [10.27411395,  0.        ,  0.        ,  1.04155143, 16.54087906,         0.        ,  0.        , -1.14296052],
                        [ 0.        , 10.27411395, -1.04155143,  0.        ,  0.        ,        16.54087906,  1.14296052,  0.        ],
                        [ 0.        ,  1.14850813, -0.10238917,  0.        ,  0.        ,         1.14296052,  0.09198448,  0.        ],
                        [-1.14850813,  0.        ,  0.        , -0.10238917, -1.14296052,         0.        ,  0.        ,  0.09198448]])
    # fmt: on
    assert_almost_equal(tap_tim.M(), M0e_tim, decimal=5)


def test_stiffness_matrix_tap_tim(tap_tim):
    # fmt: off
    K0e_tim = np.array([[-19.5879458 ,   0.        ,   0.        ,   5.57911856,         19.5879458 ,   0.        ,   0.        ,   1.92061626],
                        [  0.        , -19.5879458 ,  -5.57911856,   0.        ,          0.        ,  19.5879458 ,  -1.92061626,   0.        ],
                        [  0.        ,  -5.57911856,   1.78916039,   0.        ,          0.        ,   5.57911856,   0.44248703,   0.        ],
                        [  5.57911856,   0.        ,   0.        ,   1.78916039,         -5.57911856,   0.        ,   0.        ,   0.44248703],
                        [ 19.5879458 ,   0.        ,   0.        ,  -5.57911856,        -19.5879458 ,   0.        ,   0.        ,   0.03401395],
                        [  0.        ,  19.5879458 ,   5.57911856,   0.        ,          0.        , -19.5879458 ,  -0.03401395,   0.        ],
                        [  0.        ,  -1.92061626,   0.44248703,   0.        ,          0.        ,  -0.03401395,   0.32575947,   0.        ],
                        [  1.92061626,   0.        ,   0.        ,   0.44248703,          0.03401395,   0.        ,   0.        ,   0.32575947]])
    # fmt: on
    assert_almost_equal(tap_tim.K() / 1e8, K0e_tim, decimal=5)


def test_gyroscopic_matrix_tap_tim(tap_tim):
    # fmt: off
    G0e_tim = np.array([[ 0.        ,  1.19360236, -0.15864756,  0.        ,  0.        ,        -1.19360236,  0.05013491,  0.        ],
                        [ 1.19360236,  0.        ,  0.        ,  0.15864756, -1.19360236,         0.        ,  0.        , -0.05013491],
                        [ 0.15864756,  0.        ,  0.        ,  0.11754246, -0.15864756,         0.        ,  0.        , -0.01449657],
                        [ 0.        , -0.15864756,  0.11754246,  0.        ,  0.        ,         0.15864756, -0.01449657,  0.        ],
                        [ 0.        , -1.19360236,  0.15864756,  0.        ,  0.        ,        -0.39661891, -0.05013491,  0.        ],
                        [-1.19360236,  0.        ,  0.        , -0.15864756, -0.39661891,         0.        ,  0.        , -0.05013491],
                        [-0.05013491,  0.        ,  0.        , -0.01449657,  0.05013491,         0.        ,  0.        , -0.00213197],
                        [ 0.        ,  0.05013491, -0.01449657,  0.        ,  0.        ,        -0.05013491, -0.00213197,  0.        ]])
    # fmt: on
    assert_almost_equal(tap_tim.G(), G0e_tim, decimal=5)
