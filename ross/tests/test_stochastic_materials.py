"""Tests file.
Tests for:
    st_materials.py
"""

import pytest
from numpy.testing import assert_allclose

from ross.stochastic.st_materials import ST_Material


###############################################################################
# tests for a single random variables
###############################################################################
def test_rho():
    mat = ST_Material(name="test", rho=[7810, 7850], E=209e9, G_s=81e9)
    material = list(mat.__iter__())
    assert [m.rho for m in material] == [7810, 7850]


def test_E_single_random_var():
    mat = ST_Material(name="test", rho=7850, E=[209e9, 210e9], G_s=81e9)
    material = list(mat.__iter__())
    assert [m.Poisson for m in material] == [0.2901234567901234, 0.2962962962962963]
    assert [m.E for m in material] == [209000000000.0, 210000000000.0]
    assert [m.G_s for m in material] == [81000000000.0, 81000000000.0]


def test_G_s_single_random_var():
    mat = ST_Material(name="test", rho=7850, E=211e9, G_s=[80e9, 82e9])
    material = list(mat.__iter__())
    assert [m.Poisson for m in material] == [0.3187500000000001, 0.28658536585365857]
    assert [m.G_s for m in material] == [80000000000.0, 82000000000.0]
    assert [m.E for m in material] == [211000000000.0, 211000000000.0]


def test_Poisson_single_random_var():
    mat = ST_Material(name="test", rho=7850, E=211e9, Poisson=[0.27, 0.28])
    material = list(mat.__iter__())
    assert [m.G_s for m in material] == [83070866141.73228, 82421875000.0]
    assert [m.E for m in material] == [211000000000.0, 211000000000.0]
    assert [m.Poisson for m in material] == [0.27, 0.28]


###############################################################################
# tests for multiple random variables
###############################################################################
def test_E_multiple_random_var():
    mat = ST_Material(name="test", rho=7850, G_s=[80e9, 82e9], Poisson=[0.27, 0.28])
    material = list(mat.__iter__())
    assert [m.E for m in material] == [203200000000.0, 209920000000.0]
    assert [m.Poisson for m in material] == [0.27, 0.28]
    assert [m.G_s for m in material] == [80000000000.0, 82000000000.0]


def test_G_s_multiple_random_var():
    mat = ST_Material(name="test", rho=7850, E=[209e9, 210e9], Poisson=[0.27, 0.28])
    material = list(mat.__iter__())
    assert [m.G_s for m in material] == [82283464566.92914, 82031250000.0]
    assert [m.E for m in material] == [209000000000.0, 210000000000.0]
    assert [m.Poisson for m in material] == [0.27, 0.28]


def test_Poisson_multiple_random_var():
    mat = ST_Material(name="test", rho=7850, E=[209e9, 210e9], G_s=[80e9, 82e9])
    material = list(mat.__iter__())
    assert_allclose(
        [m.Poisson for m in material], [0.3062499999999999, 0.2804878048780488]
    )
    assert [m.E for m in material] == [209000000000.0, 210000000000.0]
    assert [m.G_s for m in material] == [80000000000.0, 82000000000.0]


###############################################################################
# testing error message
###############################################################################
def test_error_E_G_s_Poisson():
    with pytest.raises(ValueError) as ex:
        ST_Material(
            name="steel",
            rho=7810,
            E=[209e9, 211e9],
            G_s=[80e9, 82e9],
            Poisson=[0.29, 0.30],
        )
    assert "Exactly 2 arguments from E, G_s and Poisson should be provided" in str(
        ex.value
    )


def test_error_material_name_with_space():
    with pytest.raises(ValueError) as ex:
        ST_Material(name="with space", rho=1.0, E=[10.0, 20.0], G_s=[1.0, 2.0])
    assert "Spaces are not allowed in Material name" in str(ex.value)
