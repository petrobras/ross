import numpy as np
import pytest
from copy import deepcopy
from numpy.testing import assert_allclose

from ross.units import Q_
from ross.materials import Material, steel
from ross.gear_element import GearElement, GearElementTVMS, Mesh


@pytest.fixture
def gear():
    return GearElement(
        n=4,
        m=726.4,
        Id=56.95,
        Ip=113.9,
        n_teeth=328,
        pitch_diameter=1.1,
        pr_angle=Q_(22.5, "deg"),
    )


def test_gear_params(gear):
    assert gear.pr_angle == 0.39269908169872414
    assert gear.base_radius == 0.5081337428812077
    assert gear.helix_angle == 0.0
    assert gear.material == steel


def test_mass_matrix_gear(gear):
    # fmt: off
    Mg = np.array([[726.4,      0.,      0.,      0.,      0.,      0.],
                   [     0., 726.4,      0.,      0.,      0.,      0.],
                   [     0.,      0., 726.4,      0.,      0.,      0.],
                   [     0.,      0.,      0., 56.95,      0.,      0.],
                   [     0.,      0.,      0.,      0., 56.95,      0.],
                   [     0.,      0.,      0.,      0.,      0., 113.9]])
    # fmt: on

    assert_allclose(gear.M(), Mg, rtol=1e-6, atol=1e-5)


def test_gyroscopic_matrix_gear(gear):
    # fmt: off
    Gg = np.array([[0., 0., 0.,     0.,    0., 0.],
                   [0., 0., 0.,     0.,    0., 0.],
                   [0., 0., 0.,     0.,    0., 0.],
                   [0., 0., 0.,     0., 113.9, 0.],
                   [0., 0., 0., -113.9,    0., 0.],
                   [0., 0., 0.,     0.,    0., 0.]])
    # fmt: on

    assert_allclose(gear.G(), Gg, rtol=1e-6, atol=1e-5)


@pytest.fixture
def gear_tvms():
    gear_material = Material(
        "Gear_steel", rho=Q_(7.81, "g/cm**3"), E=Q_(206, "GPa"), Poisson=0.3
    )

    return GearElementTVMS(
        n=0,
        material=gear_material,
        module=Q_(2, "mm"),
        width=Q_(20, "mm"),
        n_teeth=62,
        bore_diameter=Q_(2 * 17.5, "mm"),
        pr_angle=Q_(20, "deg"),
    )


def test_gear_tvms_params(gear_tvms):
    assert gear_tvms.pr_angle == 0.3490658503988659
    assert gear_tvms.base_radius == 0.05826094248872632
    assert gear_tvms.m == 1.7360332618790646
    assert gear_tvms.Ip == 0.0036024860225567935
    assert gear_tvms.Id == 0.0018591107866743656

    radii_dict = {
        "base": 0.05826094248872632,
        "pitch": 0.062,
        "addendum": 0.064,
        "root": 0.0595,
        "cutter_tip": 0.0007599016822903686,
    }

    tooth_dict = {
        "root_angle": 0.0456584787349235,
        "base_angle": 0.04023980849306058,
        "a": 0.0017400983177096314,
        "b": 0.0030128107986983586,
    }

    pr_angles_dict = {
        "pitch": 0.3490658503988659,
        "addendum": 0.42672233774374957,
        "start_point": 0.2577380345253824,
    }

    assert gear_tvms.radii_dict == radii_dict
    assert gear_tvms.tooth_dict == tooth_dict
    assert gear_tvms.pr_angles_dict == pr_angles_dict


def test_mass_matrix_gear_tvms(gear_tvms):
    # fmt: off
    Mgt = np.array([[1.73603,      0.,      0.,      0.,      0.,      0.],
                    [     0., 1.73603,      0.,      0.,      0.,      0.],
                    [     0.,      0., 1.73603,      0.,      0.,      0.],
                    [     0.,      0.,      0., 0.00185,      0.,      0.],
                    [     0.,      0.,      0.,      0., 0.00185,      0.],
                    [     0.,      0.,      0.,      0.,      0., 0.00360]])
    # fmt: on

    assert_allclose(gear_tvms.M(), Mgt, rtol=1e-6, atol=1e-5)


def test_gyroscopic_matrix_gear_tvms(gear_tvms):
    # fmt: off
    Ggt = np.array([[0., 0., 0.,       0.,      0., 0.],
                    [0., 0., 0.,       0.,      0., 0.],
                    [0., 0., 0.,       0.,      0., 0.],
                    [0., 0., 0.,       0., 0.00360, 0.],
                    [0., 0., 0., -0.00360,      0., 0.],
                    [0., 0., 0.,       0.,      0., 0.]])
    # fmt: on

    assert_allclose(gear_tvms.G(), Ggt, rtol=1e-6, atol=1e-5)


@pytest.fixture
def mesh(gear_tvms):
    return Mesh(gear_tvms, deepcopy(gear_tvms))


def test_mesh_params(mesh):
    assert mesh.pressure_angle == 0.3490658503988659
    assert mesh.gear_ratio == 1.0
    assert mesh.contact_ratio == 1.7897798668330458
    assert mesh.hertzian_stiffness == 3555868607.9093266
    assert_allclose(mesh.stiffness, 419569438.98401976, rtol=1e-6, atol=1e-5)
    assert_allclose(
        mesh.stiffness_range[:5],
        [
            396622921.3581085,
            396738680.0375346,
            396854147.2919756,
            396969323.136738,
            397084207.5865779,
        ],
        rtol=1e-6,
        atol=1e-5,
    )
