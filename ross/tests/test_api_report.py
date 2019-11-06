import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from ross.api_report import Report
from ross.bearing_seal_element import BearingElement
from ross.disk_element import DiskElement
from ross.materials import steel
from ross.rotor_assembly import Rotor
from ross.shaft_element import ShaftElement


@pytest.fixture
def report0():
    # rotor type: between bearings
    i_d = 0.0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l, i_d, o_d, steel, shear_effects=True, rotary_inertia=True, gyroscopic=True
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.35
    )

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.5)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.5)

    rotor = Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])

    minspeed = 400.0
    maxspeed = 1000.0
    machine_type = "compressor"
    units = "rad/s"

    return Report(rotor, minspeed, maxspeed, machine_type, units)


@pytest.fixture
def report1():
    # rotor type: single overhung
    i_d = 0.0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l, i_d, o_d, steel, shear_effects=True, rotary_inertia=True, gyroscopic=True
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=0, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(2, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.5)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.5)

    rotor = Rotor(shaft_elem, [disk0], [bearing0, bearing1])

    minspeed = 400.0
    maxspeed = 1000.0
    machine_type = "turbine"
    units = "rad/s"

    return Report(rotor, minspeed, maxspeed, machine_type, units)


@pytest.fixture
def report2():
    # rotor type: single double overhung
    i_d = 0.0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l, i_d, o_d, steel, shear_effects=True, rotary_inertia=True, gyroscopic=True
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=0, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=6, material=steel, width=0.07, i_d=0.05, o_d=0.35
    )

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(2, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.5)
    bearing1 = BearingElement(4, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.5)

    rotor = Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])

    minspeed = 3820.0
    maxspeed = 9550.0
    machine_type = "pump"
    units = "rpm"

    return Report(rotor, minspeed, maxspeed, machine_type, units)


def test_initial_attributes(report0, report1, report2):
    assert report0.rotor_type == 'between_bearings'
    assert report0.disk_nodes == [2, 4]
    assert report0.machine_type == 'compressor'
    assert report0.tag == 'Rotor 0'
    assert_allclose(report0.maxspeed, 1000.0, atol=1e-2)
    assert_allclose(report0.minspeed, 400.0, atol=1e-2)
    assert report1.rotor_type == 'single_overhung_l'
    assert report1.disk_nodes == [0]
    assert report1.machine_type == 'turbine'
    assert report1.tag == 'Rotor 0'
    assert_allclose(report1.maxspeed, 1000.0, atol=1e-2)
    assert_allclose(report1.minspeed, 400.0, atol=1e-2)
    assert report2.rotor_type == 'double_overhung'
    assert report2.disk_nodes == [0, 6]
    assert report2.machine_type == 'compressor'
    assert report2.tag == 'Rotor 0'
    assert_allclose(report2.maxspeed, 1000.0736613927509, atol=1e-8)
    assert_allclose(report2.minspeed, 400.0294645571003, atol=1e-8)


def test_report_static_forces(report0, report1, report2):
    F_0 = report0.static_forces()
    F_1 = report1.static_forces()
    F_2 = report2.static_forces()
    assert_allclose(F_0[0], 50.40534339, atol=1e-6)
    assert_allclose(F_0[1], 56.7174833, atol=1e-6)
    assert_allclose(F_1[0], 66.13980523, atol=1e-6)
    assert_allclose(F_1[1], -10.54402692, atol=1e-6)
    assert_allclose(F_2[0], 25.15678377, atol=1e-6)
    assert_allclose(F_2[1], 81.96604293, atol=1e-6)


def test_unbalance_forces(report0, report1, report2):
    Uforce_00 = report0.unbalance_forces(mode=0)
    assert_allclose(Uforce_00, [71.23351373], atol=1e-6)

    Uforce_02 = report0.unbalance_forces(mode=2)
    assert_allclose(Uforce_02, [33.51806362, 37.71545011], atol=1e-6)

    Uforce_10 = report1.unbalance_forces(mode=0)
    assert_allclose(Uforce_10, [26.769833052956542], atol=1e-6)

    Uforce_12 = report1.unbalance_forces(mode=2)
    assert_allclose(Uforce_12, [26.769833052956542], atol=1e-6)

    Uforce_20 = report2.unbalance_forces(mode=0)
    assert_allclose(Uforce_20, np.array([26.7678613, 39.35850396]), atol=1e-6)

    Uforce_22 = report2.unbalance_forces(mode=2)
    assert_allclose(Uforce_22, np.array([26.7678613, 39.35850396]), atol=1e-6)


def test_report_mode_shape(report0, report1, report2):
    n1, n2 = report0.mode_shape(mode=0)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [3]

    n1, n2 = report1.mode_shape(mode=0)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [0]

    n1, n2 = report1.mode_shape(mode=3)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [0]

    n1, n2 = report2.mode_shape(mode=0)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [0, 6]

    n1, n2 = report2.mode_shape(mode=3)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [0, 6]
