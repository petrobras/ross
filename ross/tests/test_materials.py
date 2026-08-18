import sys

import pytest
from numpy.testing import assert_allclose

from ross.materials import *


@pytest.fixture
def AISI4140():
    return Material(
        name="AISI4140", rho=7850, E=203200000000.0, G_s=80000000000.0, color="#525252"
    )


@pytest.fixture
def A216WCB():
    return Material(
        name="A216WCB",
        rho=7820.0,
        E=210000000000.0,
        G_s=81395348837.2093,
        color="#525252",
    )


def test_material_equality(A216WCB, AISI4140):
    assert A216WCB != AISI4140


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
    assert "__init__() missing 1 required positional argument: 'rho'" in str(ex.value)


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
    obj2 = Material.load_material("obj1")
    assert obj1.__dict__ == obj2.__dict__

    obj1.remove_material("obj1")
    available_after = Material.available_materials()
    assert available == available_after


def test_repr():
    mat0 = Material(name="obj1", rho=92e1, E=281.21, G_s=20e9)
    mat1 = eval(repr(mat0))
    assert mat0 == mat1


def test_save_without_thermal_props_keeps_other_materials(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ross.materials.AVAILABLE_MATERIALS_PATH", tmp_path / "materials.toml"
    )
    with_thermal = Material(
        name="mat_thermal",
        rho=7810,
        E=211e9,
        G_s=81.2e9,
        specific_heat=434.0,
        thermal_conductivity=60.5,
    )
    with_thermal.save_material()

    without_thermal = Material(name="mat_plain", rho=7850, E=203.2e9, G_s=80e9)
    without_thermal.save_material()

    reloaded = Material.load_material("mat_thermal")
    assert_allclose(reloaded.specific_heat, 434.0)
    assert_allclose(reloaded.thermal_conductivity, 60.5)

    plain_entry = Material.get_data()["Materials"]["mat_plain"]
    assert "specific_heat" not in plain_entry
    assert "thermal_conductivity" not in plain_entry


def test_resave_without_thermal_props_writes_no_zeros(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ross.materials.AVAILABLE_MATERIALS_PATH", tmp_path / "materials.toml"
    )
    Material(
        name="mat0",
        rho=7810,
        E=211e9,
        G_s=81.2e9,
        specific_heat=434.0,
        thermal_conductivity=60.5,
    ).save_material()

    Material(name="mat0", rho=7810, E=211e9, G_s=81.2e9).save_material()

    entry = Material.get_data()["Materials"]["mat0"]
    assert "specific_heat" not in entry
    assert "thermal_conductivity" not in entry

    reloaded = Material.load_material("mat0")
    assert reloaded.specific_heat is None
    assert reloaded.thermal_conductivity is None


def test_round_trip_with_thermal_props(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ross.materials.AVAILABLE_MATERIALS_PATH", tmp_path / "materials.toml"
    )
    mat0 = Material(
        name="mat0",
        rho=7810,
        E=211e9,
        G_s=81.2e9,
        specific_heat=434.0,
        thermal_conductivity=60.5,
    )
    mat0.save_material()

    mat1 = Material.load_material("mat0")
    assert mat0.__dict__ == mat1.__dict__
    assert_allclose(mat1.specific_heat, 434.0)
    assert_allclose(mat1.thermal_conductivity, 60.5)


def test_round_trip_without_thermal_props(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "ross.materials.AVAILABLE_MATERIALS_PATH", tmp_path / "materials.toml"
    )
    mat0 = Material(name="mat0", rho=7810, E=211e9, G_s=81.2e9)
    mat0.save_material()

    mat1 = Material.load_material("mat0")
    assert mat0.__dict__ == mat1.__dict__
    assert mat1.specific_heat is None
    assert mat1.thermal_conductivity is None
