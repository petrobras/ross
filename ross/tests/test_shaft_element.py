import os

import numpy as np
import pytest
from pathlib import Path
from numpy.testing import assert_almost_equal, assert_allclose

from ross.materials import steel
from ross.shaft_element import ShaftElement

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


@pytest.skip("Wait for .from_table() modification.")
def test_from_table():
    for shaft_file in ["/data/shaft_us.xls", "/data/shaft_si.xls"]:
        shaft = ShaftElement.from_table(str(Path.cwd()) + shaft_file, sheet="Model")
        el0 = shaft[0]
        assert el0.n == 0
        assert_allclose(el0.i_d, 0.1409954)
        assert_allclose(el0.o_d, 0.151003)

        mat0 = el0.material
        assert_allclose(mat0.rho, 7833.41)
        assert_allclose(mat0.E, 206842718795.05)
        assert_allclose(mat0.G_s, 82737087518.02)

        # test if node is the same for elements in different layers
        assert shaft[8].n == 8
        assert shaft[9].n == 8
        assert_allclose(shaft[8].material.E, 206842718795.05)
        assert_allclose(shaft[9].material.E, 6894.75)
