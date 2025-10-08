import numpy as np
import pytest
from copy import deepcopy
from numpy.testing import assert_almost_equal

import ross as rs

from ross.units import Q_
from ross.materials import Material, steel
from ross.gear_element import GearElement
from ross.multi_rotor import MultiRotor


@pytest.fixture
def multi_rotor():
    # A spur geared two-shaft rotor system.
    material = rs.Material(name="mat_steel", rho=7800, E=207e9, G_s=79.5e9)

    # Rotor 1
    L1 = [0.1, 4.24, 1.16, 0.3]
    d1 = [0.3, 0.3, 0.22, 0.22]
    shaft1 = [
        rs.ShaftElement(
            L=L1[i],
            idl=0.0,
            odl=d1[i],
            material=material,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(len(L1))
    ]

    generator = rs.DiskElement(
        n=1,
        m=525.7,
        Id=16.1,
        Ip=32.2,
    )
    disk = rs.DiskElement(
        n=2,
        m=116.04,
        Id=3.115,
        Ip=6.23,
    )

    pressure_angle = rs.Q_(22.5, "deg")
    base_radius = 0.5086
    pitch_diameter = 2 * base_radius / np.cos(pressure_angle)

    N1 = 328  # Number of teeth of gear 1
    m = 726.4
    Id = 56.95
    Ip = 113.9
    width = (4 * m) / (material.rho * np.pi * (pitch_diameter**2 - d1[-1] ** 2))
    gear1 = rs.GearElement(
        n=4,
        m=m,
        Id=Id,
        Ip=Ip,
        width=width,
        n_teeth=N1,
        pitch_diameter=pitch_diameter,
        pressure_angle=pressure_angle,
        material=material,
        helix_angle=0,
    )

    bearing1 = rs.BearingElement(n=0, kxx=183.9e6, kyy=200.4e6, cxx=3e3)
    bearing2 = rs.BearingElement(n=3, kxx=183.9e6, kyy=200.4e6, cxx=3e3)

    rotor1 = rs.Rotor(
        shaft1,
        [generator, disk, gear1],
        [bearing1, bearing2],
    )

    # Rotor 2
    L2 = [0.3, 5, 0.1]
    d2 = [0.15, 0.15, 0.15]
    shaft2 = [
        rs.ShaftElement(
            L=L2[i],
            idl=0.0,
            odl=d2[i],
            material=material,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(len(L2))
    ]

    base_radius = 0.03567
    pitch_diameter = 2 * base_radius / np.cos(pressure_angle)

    N2 = 23  # Number of teeth of gear 2
    m = 5
    Id = 0.002
    Ip = 0.004
    gear2 = rs.GearElement(
        n=0,
        m=m,
        Id=Id,
        Ip=Ip,
        width=width,
        n_teeth=N2,
        pitch_diameter=pitch_diameter,
        pressure_angle=pressure_angle,
        material=material,
        helix_angle=0,
    )

    turbine = rs.DiskElement(n=2, m=7.45, Id=0.0745, Ip=0.149)

    bearing3 = rs.BearingElement(n=1, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    bearing4 = rs.BearingElement(n=3, kxx=10.1e6, kyy=41.6e6, cxx=3e3)

    rotor2 = rs.Rotor(
        shaft2,
        [gear2, turbine],
        [bearing3, bearing4],
    )

    return MultiRotor(
        rotor1,
        rotor2,
        coupled_nodes=(4, 0),
        orientation_angle=0.0,
        position="below",
    )


def test_multi_rotor(multi_rotor):
    assert multi_rotor.contact_ratio == 1.6377334309511224
    assert multi_rotor.gear_mesh_stiffness == 1918871975.9364796


def test_coupling_matrix_gear(multi_rotor):
    coupling_matrix = np.array(
        [
            [
                1.46446609e-01,
                3.53553391e-01,
                -1.79345371e-17,
                -0.00000000e00,
                9.87304651e-18,
                1.94632794e-01,
                -1.46446609e-01,
                -3.53553391e-01,
                1.79345371e-17,
                0.00000000e00,
                -6.92433285e-19,
                -1.36503180e-02,
            ],
            [
                3.53553391e-01,
                8.53553391e-01,
                -4.32978028e-17,
                -0.00000000e00,
                2.38356428e-17,
                4.69885130e-01,
                -3.53553391e-01,
                -8.53553391e-01,
                4.32978028e-17,
                0.00000000e00,
                -1.67168183e-18,
                -3.29547829e-02,
            ],
            [
                -1.79345371e-17,
                -4.32978028e-17,
                2.19634735e-33,
                0.00000000e00,
                -1.20909948e-33,
                -2.38356428e-17,
                1.79345371e-17,
                4.32978028e-17,
                -2.19634735e-33,
                -0.00000000e00,
                8.47986207e-35,
                1.67168183e-18,
            ],
            [
                -0.00000000e00,
                -0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                9.87304651e-18,
                2.38356428e-17,
                -1.20909948e-33,
                -0.00000000e00,
                6.65614914e-34,
                1.31216327e-17,
                -9.87304651e-18,
                -2.38356428e-17,
                1.20909948e-33,
                0.00000000e00,
                -4.66820369e-35,
                -9.20268659e-19,
            ],
            [
                1.94632794e-01,
                4.69885130e-01,
                -2.38356428e-17,
                -0.00000000e00,
                1.31216327e-17,
                2.58673960e-01,
                -1.94632794e-01,
                -4.69885130e-01,
                2.38356428e-17,
                0.00000000e00,
                -9.20268659e-19,
                -1.81417620e-02,
            ],
            [
                -1.46446609e-01,
                -3.53553391e-01,
                1.79345371e-17,
                0.00000000e00,
                -9.87304651e-18,
                -1.94632794e-01,
                1.46446609e-01,
                3.53553391e-01,
                -1.79345371e-17,
                -0.00000000e00,
                6.92433285e-19,
                1.36503180e-02,
            ],
            [
                -3.53553391e-01,
                -8.53553391e-01,
                4.32978028e-17,
                0.00000000e00,
                -2.38356428e-17,
                -4.69885130e-01,
                3.53553391e-01,
                8.53553391e-01,
                -4.32978028e-17,
                -0.00000000e00,
                1.67168183e-18,
                3.29547829e-02,
            ],
            [
                1.79345371e-17,
                4.32978028e-17,
                -2.19634735e-33,
                -0.00000000e00,
                1.20909948e-33,
                2.38356428e-17,
                -1.79345371e-17,
                -4.32978028e-17,
                2.19634735e-33,
                0.00000000e00,
                -8.47986207e-35,
                -1.67168183e-18,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -0.00000000e00,
                -0.00000000e00,
            ],
            [
                -6.92433285e-19,
                -1.67168183e-18,
                8.47986207e-35,
                0.00000000e00,
                -4.66820369e-35,
                -9.20268659e-19,
                6.92433285e-19,
                1.67168183e-18,
                -8.47986207e-35,
                -0.00000000e00,
                3.27398399e-36,
                6.45418463e-20,
            ],
            [
                -1.36503180e-02,
                -3.29547829e-02,
                1.67168183e-18,
                0.00000000e00,
                -9.20268659e-19,
                -1.81417620e-02,
                1.36503180e-02,
                3.29547829e-02,
                -1.67168183e-18,
                -0.00000000e00,
                6.45418463e-20,
                1.27234890e-03,
            ],
        ]
    )

    assert_almost_equal(multi_rotor.coupling_matrix(), coupling_matrix, decimal=5)
