"""Tests file.
Tests for st_rotor_assembly.py
"""
# fmt: off
import pytest
from ross.bearing_seal_element import BearingElement
from ross.disk_element import DiskElement
from ross.materials import steel
from ross.shaft_element import ShaftElement
from ross.stochastic.st_bearing_seal_element import ST_BearingElement
from ross.stochastic.st_disk_element import ST_DiskElement
from ross.stochastic.st_point_mass import ST_PointMass
from ross.stochastic.st_rotor_assembly import ST_Rotor
from ross.stochastic.st_shaft_element import ST_ShaftElement

# fmt: on


###############################################################################
# testing error messages
###############################################################################
def test_st_shaft_elements_odd_length():
    tim0 = ST_ShaftElement(
        L=[1, 1.1], idl=0, odl=[0.1, 0.2], material=steel, is_random=["L", "odl"],
    )
    tim1 = ST_ShaftElement(
        L=[1, 1.1, 1.2],
        idl=0,
        odl=[0.1, 0.2, 0.3],
        material=steel,
        is_random=["L", "odl"],
    )
    shaft_elm = [tim0, tim1]

    with pytest.raises(ValueError) as ex:
        ST_Rotor(shaft_elm)
    assert "not all random shaft elements lists have same length." in str(ex.value)


def test_st_disk_elements_odd_length():
    tim0 = ShaftElement(L=0.25, idl=0, odl=0.05, material=steel)
    tim1 = ShaftElement(L=0.25, idl=0, odl=0.05, material=steel)
    shaft_elm = [tim0, tim1]

    disk0 = ST_DiskElement(n=0, m=[20, 30], Id=1, Ip=1, is_random=["m"])
    disk1 = ST_DiskElement(n=2, m=[20, 30, 40], Id=1, Ip=1, is_random=["m"])

    with pytest.raises(ValueError) as ex:
        ST_Rotor(shaft_elm, [disk0, disk1])
    assert "not all random disk elements lists have same length." in str(ex.value)


def test_st_bearing_elements_odd_length():
    tim0 = ShaftElement(L=0.25, idl=0, odl=0.05, material=steel,)
    tim1 = ShaftElement(L=0.25, idl=0, odl=0.05, material=steel,)
    shaft_elm = [tim0, tim1]

    disk0 = DiskElement(n=1, m=20, Id=1, Ip=1)

    brg0 = ST_BearingElement(
        n=0, kxx=[1e6, 2e6], cxx=[1e3, 2e3], is_random=["kxx", "cxx"],
    )
    brg1 = ST_BearingElement(
        n=2, kxx=[1e6, 2e6, 3e6], cxx=[1e3, 2e3, 3e3], is_random=["kxx", "cxx"],
    )

    with pytest.raises(ValueError) as ex:
        ST_Rotor(shaft_elm, [disk0], [brg0, brg1])
    assert "not all random bearing elements lists have same length." in str(ex.value)


def test_st_point_mass_elements_odd_length():
    tim0 = ShaftElement(L=0.25, idl=0, odl=0.05, material=steel)
    tim1 = ShaftElement(L=0.25, idl=0, odl=0.05, material=steel)
    shaft_elm = [tim0, tim1]

    disk0 = DiskElement(n=1, m=20, Id=1, Ip=1)

    brg0 = BearingElement(n=0, kxx=1e6, cxx=1e3, n_link=3)
    brg1 = BearingElement(n=2, kxx=1e6, cxx=1e3, n_link=4)
    sup0 = BearingElement(n=3, kxx=1e6, cxx=1e3)
    sup1 = BearingElement(n=4, kxx=1e6, cxx=1e3)

    pm0 = ST_PointMass(n=3, m=[1, 2], is_random=["m"])
    pm1 = ST_PointMass(n=4, m=[1, 2, 3], is_random=["m"])

    with pytest.raises(ValueError) as ex:
        ST_Rotor(shaft_elm, [disk0], [brg0, brg1, sup0, sup1], [pm0, pm1])
    assert "not all random point mass lists have same length." in str(ex.value)
