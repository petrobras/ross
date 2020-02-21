import pytest
from numpy.testing import assert_allclose

from ross.materials import *


@pytest.fixture
def AISI4140():
    return Material(
        name="AISI4140", rho=7850, E=203200000000.0, G_s=80000000000.0, color="#525252",
    )


def test_raise_name_material():
    with pytest.raises(ValueError) as excinfo:
        Material("with space", rho=7850, G_s=80e9, Poisson=0.27)
    assert "Spaces are not allowed" in str(excinfo.value)


def test_E():
    mat = Material(name="test", rho=7850, G_s=80e9, Poisson=0.27)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_G_s():
    mat = Material(name="test", rho=7850, E=203.2e9, Poisson=0.27)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_Poisson():
    mat = Material(name="test", rho=7850, E=203.2e9, G_s=80e9)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_E_G_s_Poisson():
    mat = Material(name="test", rho=7850, E=203.2e9, G_s=80e9)
    assert_allclose(mat.E, 203.2e9)
    assert_allclose(mat.G_s, 80e9)
    assert_allclose(mat.Poisson, 0.27)


def test_specific_material(AISI4140):
    assert_allclose(AISI4140.rho, 7850)
    assert_allclose(AISI4140.E, 203.2e9)
    assert_allclose(AISI4140.G_s, 80e9)
    assert_allclose(AISI4140.Poisson, 0.27)


def test_error_rho():
    with pytest.raises(TypeError) as ex:
        Material(name="test", E=203.2e9, G_s=80e9)

    assert "__init__() missing 1 required positional argument: 'rho'" == str(ex.value)


def test_error_E_G_s_Poisson():
    with pytest.raises(ValueError) as ex:
        Material(name="test", rho=785, E=203.2e9)
    assert "Exactly 2 arguments from E" in str(ex.value)
    with pytest.raises(ValueError) as ex:
        Material(name="test", rho=785, E=203.2e9, G_s=80e9, Poisson=0.27)
    assert "Exactly 2 arguments from E" in str(ex.value)


# Serialization tests.


def test_available_materials():
    Material.available_materials()


def test_serialization():
    available = Material.available_materials()
    obj1 = Material(name="obj1", rho=92e1, E=281.21, G_s=20e9)
    obj1.save_material()
    obj2 = Material.use_material("obj1")
    assert obj1.__dict__ == obj2.__dict__

    obj1.remove_material("obj1")
    available_after = Material.available_materials()
    assert available == available_after


def test_repr():
    mat0 = Material(name="obj1", rho=92e1, E=281.21, G_s=20e9)
    mat1 = eval(repr(mat0))
    assert mat0 == mat1
