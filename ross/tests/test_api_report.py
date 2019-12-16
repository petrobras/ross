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
    n = 50
    L = [0.03 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l, i_d, o_d, material=steel, shear_effects=True, rotary_inertia=True, gyroscopic=True
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=15, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=35, material=steel, width=0.07, i_d=0.05, o_d=0.35
    )

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=1000)
    bearing1 = BearingElement(50, kxx=stfx, kyy=stfy, cxx=1000)

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
    n = 50
    L = [0.03 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l, i_d, o_d, material=steel, shear_effects=True, rotary_inertia=True, gyroscopic=True
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=0, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(15, kxx=stfx, kyy=stfy, cxx=1000)
    bearing1 = BearingElement(50, kxx=stfx, kyy=stfy, cxx=1000)

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
    n = 50
    L = [0.03 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l, i_d, o_d, material=steel, shear_effects=True, rotary_inertia=True, gyroscopic=True
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=0, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=50, material=steel, width=0.07, i_d=0.05, o_d=0.35
    )

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(15, kxx=stfx, kyy=stfy, cxx=1000)
    bearing1 = BearingElement(35, kxx=stfx, kyy=stfy, cxx=1000)

    rotor = Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])

    minspeed = 3820.0
    maxspeed = 9550.0
    machine_type = "pump"
    units = "rpm"

    return Report(rotor, minspeed, maxspeed, machine_type, units)


def test_initial_attributes(report0, report1, report2):
    assert report0.rotor_type == 'between_bearings'
    assert report0.disk_nodes == [15, 35]
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
    assert report2.disk_nodes == [0, 50]
    assert report2.machine_type == 'compressor'
    assert report2.tag == 'Rotor 0'
    assert_allclose(report2.maxspeed, 1000.0736613927509, atol=1e-8)
    assert_allclose(report2.minspeed, 400.0294645571003, atol=1e-8)


def test_report_static_forces(report0, report1, report2):
    F_0 = report0.static_forces()
    F_1 = report1.static_forces()
    F_2 = report2.static_forces()
    assert_allclose(F_0[0], 49.77310967, atol=1e-6)
    assert_allclose(F_0[1], 57.34971702, atol=1e-6)
    assert_allclose(F_1[0], 62.98883394, atol=1e-6)
    assert_allclose(F_1[1], -7.39305563, atol=1e-6)
    assert_allclose(F_2[0], 29.88833937, atol=1e-6)
    assert_allclose(F_2[1], 77.22429001, atol=1e-6)


def test_unbalance_forces(report0, report1, report2):
    Uforce_00 = report0.unbalance_forces(mode=0)
    assert_allclose(Uforce_00, [71.23351373], atol=1e-6)

    Uforce_02 = report0.unbalance_forces(mode=2)
    assert_allclose(Uforce_02, [33.09764688, 38.13586685], atol=1e-6)

    Uforce_10 = report1.unbalance_forces(mode=0)
    assert_allclose(Uforce_10, [26.25997031], atol=1e-6)

    Uforce_12 = report1.unbalance_forces(mode=2)
    assert_allclose(Uforce_12, [26.25997031], atol=1e-6)

    Uforce_20 = report2.unbalance_forces(mode=0)
    assert_allclose(Uforce_20, np.array([26.25803611, 38.84867878]), atol=1e-6)

    Uforce_22 = report2.unbalance_forces(mode=2)
    assert_allclose(Uforce_22, np.array([26.25803611, 38.84867878]), atol=1e-6)


def test_report_mode_shape(report0, report1, report2):
    n1, n2 = report0.mode_shape(mode=0)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [26]

    n1, n2 = report1.mode_shape(mode=0)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [0]

    n1, n2 = report1.mode_shape(mode=3)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [0]

    n1, n2 = report2.mode_shape(mode=0)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [0, 50]

    n1, n2 = report2.mode_shape(mode=3)
    nodes = [int(node) for sub_nodes in [n1, n2] for node in sub_nodes]
    assert nodes == [0, 50]


def test_stability_level1(report0, report1, report2):
    D = [0.28, 0.35]
    H = [0.07, 0.07]
    HP = [6000, 8000]
    RHO_ratio = [1.11, 1.14]
    RHOd = 30.45
    RHOs = 37.65
    oper_speed = 1000.0

    report0.stability_level_1(D, H, HP, oper_speed, RHO_ratio, RHOs, RHOd)

    assert_allclose(report0.Q0, 81599.87755102041, atol=1e-4)
    assert_allclose(report0.Qa, 20399.969387755104, atol=1e-4)
    assert_allclose(report0.log_dec_a, 0.30527030743121375, atol=1e-4)
    assert_allclose(report0.CSR, 12.096802976513656, atol=1e-4)
    assert_allclose(report0.Qratio, 4.0, atol=1e-4)
    assert_allclose(report0.crit_speed, 82.66646997074625, atol=1e-4)
    assert_allclose(report0.MCS, 1000.0, atol=1e-4)
    assert_allclose(report0.RHO_gas, 34.05, atol=1e-4)
    assert report0.condition == 'required'

    report1.stability_level_1(D, H, HP, oper_speed, RHO_ratio, RHOs, RHOd)

    assert_allclose(report1.Q0, 79730.98330241187, atol=1e-4)
    assert_allclose(report1.Qa, 4385.204081632653, atol=1e-4)
    assert_allclose(report1.log_dec_a, 0.8370720236581191, atol=1e-4)
    assert_allclose(report1.CSR, 14.043968256888528, atol=1e-4)
    assert_allclose(report1.Qratio, 18.181818181818183, atol=1e-4)
    assert_allclose(report1.crit_speed, 71.20494590334201, atol=1e-4)
    assert_allclose(report1.MCS, 1000.0, atol=1e-4)
    assert_allclose(report1.RHO_gas, 34.05, atol=1e-4)
    assert report1.condition == 'not required'

    report2.stability_level_1(D, H, HP, oper_speed, RHO_ratio, RHOs, RHOd)

    assert_allclose(report2.Q0, 61199.90816326531, atol=1e-4)
    assert_allclose(report2.Qa, 20399.969387755104, atol=1e-4)
    assert_allclose(report2.log_dec_a, 0.6761496778932745, atol=1e-4)
    assert_allclose(report2.CSR, 26.13634690107343, atol=1e-4)
    assert_allclose(report2.Qratio, 3.0, atol=1e-4)
    assert_allclose(report2.crit_speed, 38.263712414670984, atol=1e-4)
    assert_allclose(report2.MCS, 1000.0736613927509, atol=1e-4)
    assert_allclose(report2.RHO_gas, 34.05, atol=1e-4)
    assert report2.condition == 'required'


def test_stability_level2(report0, report1, report2):
    df0 = report0.stability_level_2()
    df1 = report1.stability_level_2()

    assert_allclose(
        df0["log_dec"].tolist(),
        [
            0.17669652215696618,
            0.15801304519741688,
            0.14761936907792816,
            0.3153250139555406,
            0.1476193691000094,
        ],
        atol=1e-6,
    )
    assert_allclose(
        df1["log_dec"].tolist(),
        [
            0.14898201611278591,
            0.14898201641839076,
            0.8368485552898145,
            0.14898201644744202,
        ],
        atol=1e-6,
    )
