import os

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from ross.materials import steel
from ross.shaft_element import ShaftElement

test_dir = os.path.dirname(__file__)


@pytest.fixture
def eb():
    #  Euler-Bernoulli element
    le_ = 0.25
    i_d_l = 0
    o_d_l = 0.05
    i_d_r = 0
    o_d_r = 0.05
    return ShaftElement(
        le_, i_d_l, o_d_l, i_d_r, o_d_r, steel, shear_effects=False, rotary_inertia=False, n=3
    )


def test_index(eb):
    assert eb.dof_local_index().x_0 == 0
    assert eb.dof_local_index().y_0 == 1
    assert eb.dof_local_index().alpha_0 == 2
    assert eb.dof_local_index().beta_0 == 3


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
    i_d_l = 0
    o_d_l = 0.05
    i_d_r = 0
    o_d_r = 0.05
    return ShaftElement(le_, i_d_l, o_d_l, i_d_r, o_d_r, steel, rotary_inertia=True, shear_effects=True)


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
    assert tap_tim.dof_local_index().alpha_0 == 2
    assert tap_tim.dof_local_index().beta_0 == 3
    assert tap_tim.dof_local_index().x_1 == 4
    assert tap_tim.dof_local_index().y_1 == 5
    assert tap_tim.dof_local_index().alpha_1 == 6
    assert tap_tim.dof_local_index().beta_1 == 7


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
    M0e_tim = np.array([
            [41.84146162,  0.        ,  0.        ,  1.85735573, 10.27465893,  0.        ,  0.        , -1.16147981],
            [ 0.        , 41.84146162, -1.85735573,  0.        ,  0.        , 10.27465893,  1.16147981,  0.        ],
            [ 0.        , -1.85735573,  0.18686024,  0.        ,  0.        , -1.04154454, -0.10311193,  0.        ],
            [ 1.85735573,  0.        ,  0.        ,  0.18686024,  1.04154454,  0.        ,  0.        , -0.10311193],
            [10.27465893,  0.        ,  0.        ,  1.04154454, 17.35069604,  0.        ,  0.        , -1.1330391 ],
            [ 0.        , 10.27465893, -1.04154454,  0.        ,  0.        , 17.35069604,  1.1330391 ,  0.        ],
            [ 0.        ,  1.16147981, -0.10311193,  0.        ,  0.        ,  1.1330391 ,  0.09851808,  0.        ],
            [-1.16147981,  0.        ,  0.        , -0.10311193, -1.1330391 ,  0.        ,  0.        ,  0.09851808]])
    # fmt: on
    assert_almost_equal(tap_tim.M(), M0e_tim, decimal=5)


def test_stiffness_matrix_tap_tim(tap_tim):
    # fmt: off
    K0e_tim = np.array([
            [ 23.3411494 ,   0.        ,   0.        ,   6.2946873 , -23.3411494 ,   0.        ,   0.        ,   3.04177246],
            [  0.        ,  23.3411494 ,  -6.2946873 ,   0.        ,   0.        , -23.3411494 ,  -3.04177246,   0.        ],
            [  0.        ,  -6.2946873 ,   1.91793266,   0.        ,   0.        ,   6.2946873 ,   0.59994226,   0.        ],
            [  6.2946873 ,   0.        ,   0.        ,   1.91793266,  -6.2946873 ,   0.        ,   0.        ,   0.59994226],
            [-23.3411494 ,   0.        ,   0.        ,  -6.2946873 ,  23.3411494 ,   0.        ,   0.        ,  -3.04177246],
            [  0.        , -23.3411494 ,   6.2946873 ,   0.        ,   0.        ,  23.3411494 ,   3.04177246,   0.        ],
            [  0.        ,  -3.04177246,   0.59994226,   0.        ,   0.        ,   3.04177246,   0.61676672,   0.        ],
            [  3.04177246,   0.        ,   0.        ,   0.59994226,  -3.04177246,   0.        ,   0.        ,   0.61676672]])
    # fmt: on
    assert_almost_equal(tap_tim.K() / 1e8, K0e_tim, decimal=5)


def test_gyroscopic_matrix_tap_tim(tap_tim):
    # fmt: off
    G0e_tim = np.array([
            [ 0.        ,  1.23853234,  0.15544731,  0.        ,  0.        , -1.23853234, -0.03173553,  0.        ],
            [-1.23853234,  0.        ,  0.        ,  0.15544731,  1.23853234,  0.        ,  0.        , -0.03173553],
            [-0.15544731,  0.        ,  0.        ,  0.11850408,  0.15544731,  0.        ,  0.        , -0.01563679],
            [ 0.        , -0.15544731, -0.11850408,  0.        ,  0.        ,  0.15544731,  0.01563679,  0.        ],
            [ 0.        , -1.23853234, -0.15544731,  0.        ,  0.        ,  1.23853234,  0.03173553,  0.        ],
            [ 1.23853234,  0.        ,  0.        , -0.15544731, -1.23853234,  0.        ,  0.        ,  0.03173553],
            [ 0.03173553,  0.        ,  0.        , -0.01563679, -0.03173553,  0.        ,  0.        ,  0.01089192],
            [ 0.        ,  0.03173553,  0.01563679,  0.        ,  0.        , -0.03173553, -0.01089192,  0.        ]])
    # fmt: on
    assert_almost_equal(tap_tim.G(), G0e_tim, decimal=5)


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
    return ShaftElement(le_, i_d_, o_d_, material=steel, rotary_inertia=True, shear_effects=True)


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
