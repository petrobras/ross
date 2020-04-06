"""Tests file.
Tests for:
    st_shaft_element.py
    st_disk_element.py
    st_bearing_seal_element.py
    st_point_mass.py
"""
import numpy as np
import pytest

from ross.stochastic.st_bearing_seal_element import ST_BearingElement
from ross.stochastic.st_disk_element import ST_DiskElement
from ross.stochastic.st_materials import ST_Material
from ross.stochastic.st_point_mass import ST_PointMass
from ross.stochastic.st_shaft_element import ST_ShaftElement


@pytest.fixture
def rand_shaft():
    E = [209e9, 211e9]
    st_steel = ST_Material(name="Steel", rho=7810, E=E, G_s=81.2e9)
    elm = ST_ShaftElement(
        L=[1.0, 1.1],
        idl=[0.01, 0.02],
        odl=[0.1, 0.2],
        material=st_steel,
        is_random=["L", "idl", "odl", "material"],
    )
    return elm


@pytest.fixture
def rand_disk_from_inertia():
    elm = ST_DiskElement(
        n=1, m=[30, 40], Id=[0.2, 0.3], Ip=[0.5, 0.7], is_random=["m", "Id", "Ip"],
    )
    return elm


@pytest.fixture
def rand_disk_from_geometry():
    E = [209e9, 211e9]
    st_steel = ST_Material(name="Steel", rho=7810, E=E, G_s=81.2e9)
    elm = ST_DiskElement.from_geometry(
        n=1,
        material=st_steel,
        width=[0.07, 0.08],
        i_d=[0.05, 0.06],
        o_d=[0.30, 0.40],
        is_random=["material", "width", "i_d", "o_d"],
    )
    return elm


@pytest.fixture
def rand_bearing_constant_coefs():
    kxx = [1e6, 2e6]
    cxx = [1e3, 2e3]
    elm = ST_BearingElement(n=1, kxx=kxx, cxx=cxx, is_random=["kxx", "cxx"])

    return elm


@pytest.fixture
def rand_bearing_varying_coefs():
    kxx = [[1e6, 1.1e6], [2e6, 2.1e6]]
    kxy = [[1.5e6, 1.6e6], [2.5e6, 2.6e6]]
    kyx = [[1.5e6, 1.6e6], [2.5e6, 2.6e6]]
    kyy = [[3e6, 3.1e6], [4e6, 4.1e6]]
    cxx = [[1e3, 1.1e3], [2e3, 2.1e3]]
    cxy = [[1.5e3, 1.6e3], [2.5e3, 2.6e3]]
    cyx = [[1.5e3, 1.6e3], [2.5e3, 2.6e3]]
    cyy = [[3e3, 3.1e3], [4e3, 4.1e3]]
    frequency = np.array([500, 800])
    elm = ST_BearingElement(
        n=1,
        kxx=kxx,
        kxy=kxy,
        kyx=kyx,
        kyy=kyy,
        cxx=cxx,
        cxy=cxy,
        cyx=cyx,
        cyy=cyy,
        frequency=frequency,
        is_random=["kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"],
    )

    return elm


@pytest.fixture
def rand_point_mass():
    mx = [2.0, 2.5]
    my = [3.0, 3.5]
    elm = ST_PointMass(n=1, mx=mx, my=my, is_random=["mx", "my"])
    return elm


def test_st_shaft_element(rand_shaft):
    elm = list(rand_shaft.__iter__())
    assert [sh.L for sh in elm] == [1.0, 1.1]
    assert [sh.idl for sh in elm] == [0.01, 0.02]
    assert [sh.odl for sh in elm] == [0.1, 0.2]
    assert [sh.idr for sh in elm] == [0.01, 0.02]
    assert [sh.odr for sh in elm] == [0.1, 0.2]
    assert [sh.material.E for sh in elm] == [209000000000.0, 211000000000.0]


def test_st_disk_element_from_inertia(rand_disk_from_inertia):
    elm = list(rand_disk_from_inertia.__iter__())
    assert [dk.n for dk in elm] == [1, 1]
    assert [dk.m for dk in elm] == [30, 40]
    assert [dk.Id for dk in elm] == [0.2, 0.3]
    assert [dk.Ip for dk in elm] == [0.5, 0.7]


def test_st_disk_element_from_geometry(rand_disk_from_geometry):
    elm = list(rand_disk_from_geometry.__iter__())
    assert [dk.n for dk in elm] == [1, 1]
    assert [dk.m for dk in elm] == [37.570502893821185, 76.74810321754951]
    assert [dk.Id for dk in elm] == [0.2325457585365474, 0.8256816771154702]
    assert [dk.Ip for dk in elm] == [0.43440893970980754, 1.5694987107988876]


def test_st_bearing_element_constant_coef(rand_bearing_constant_coefs):
    elm = list(rand_bearing_constant_coefs.__iter__())
    assert [brg.n for brg in elm] == [1, 1]
    assert [brg.kxx.coefficient for brg in elm] == [[1000000.0], [2000000.0]]
    assert [brg.kyy.coefficient for brg in elm] == [[1000000.0], [2000000.0]]
    assert [brg.kxy.coefficient for brg in elm] == [[0], [0]]
    assert [brg.kyx.coefficient for brg in elm] == [[0], [0]]
    assert [brg.cxx.coefficient for brg in elm] == [[1000.0], [2000.0]]
    assert [brg.cyy.coefficient for brg in elm] == [[1000.0], [2000.0]]
    assert [brg.cxy.coefficient for brg in elm] == [[0], [0]]
    assert [brg.cyx.coefficient for brg in elm] == [[0], [0]]


def test_st_bearing_element_varying_coef(rand_bearing_varying_coefs):
    elm = list(rand_bearing_varying_coefs.__iter__())
    assert [brg.n for brg in elm] == [1, 1]
    assert [brg.kxx.coefficient for brg in elm] == [
        [1000000.0, 2000000.0],
        [1100000.0, 2100000.0],
    ]
    assert [brg.kyy.coefficient for brg in elm] == [
        [3000000.0, 4000000.0],
        [3100000.0, 4100000.0],
    ]
    assert [brg.kxy.coefficient for brg in elm] == [
        [1500000.0, 2500000.0],
        [1600000.0, 2600000.0],
    ]
    assert [brg.kyx.coefficient for brg in elm] == [
        [1500000.0, 2500000.0],
        [1600000.0, 2600000.0],
    ]
    assert [brg.cxx.coefficient for brg in elm] == [[1000.0, 2000.0], [1100.0, 2100.0]]
    assert [brg.cyy.coefficient for brg in elm] == [[3000.0, 4000.0], [3100.0, 4100.0]]
    assert [brg.cxy.coefficient for brg in elm] == [[1500.0, 2500.0], [1600.0, 2600.0]]
    assert [brg.cyx.coefficient for brg in elm] == [[1500.0, 2500.0], [1600.0, 2600.0]]


def test_st_point_mass(rand_point_mass):
    elm = list(rand_point_mass.__iter__())
    assert [pm.n for pm in elm] == [1, 1]
    assert [pm.mx for pm in elm] == [2.0, 2.5]
    assert [pm.my for pm in elm] == [3.0, 3.5]


###############################################################################
# testing error messages
###############################################################################


def test_st_bearing_error_messages():
    kxx = [1e6, 2e6]
    cxx = [1e3, 2e3]
    freq = [500, 1000]
    with pytest.raises(ValueError) as ex:
        ST_BearingElement(
            n=1, kxx=kxx, cxx=cxx, frequency=freq, is_random=["kxx", "cxx", "frequency"]
        )
    assert "frequency can not be a random variable" in str(ex.value)
