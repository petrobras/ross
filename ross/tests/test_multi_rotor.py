import numpy as np
import pytest
from copy import deepcopy
from numpy.testing import assert_allclose

import ross as rs


@pytest.fixture
def multi_rotor():
    """A spur geared two-shaft rotor system."""

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

    pressure_angle = rs.Q_(22.5, "deg").to_base_units().m
    base_radius = 0.5086
    pitch_diameter = 2 * base_radius / np.cos(pressure_angle)

    N1 = 328
    m = 726.4
    Id = 56.95
    Ip = 113.9

    gear1 = rs.GearElement(
        n=4,
        m=m,
        Id=Id,
        Ip=Ip,
        n_teeth=N1,
        pitch_diameter=pitch_diameter,
        pr_angle=pressure_angle,
        bore_diameter=d1[-1],
        material=material,
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

    N2 = 23
    m = 5
    Id = 0.002
    Ip = 0.004

    gear2 = rs.GearElement(
        n=0,
        m=m,
        Id=Id,
        Ip=Ip,
        n_teeth=N2,
        pitch_diameter=pitch_diameter,
        pr_angle=pressure_angle,
    )

    turbine = rs.DiskElement(n=2, m=7.45, Id=0.0745, Ip=0.149)

    bearing3 = rs.BearingElement(n=1, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    bearing4 = rs.BearingElement(n=3, kxx=10.1e6, kyy=41.6e6, cxx=3e3)

    rotor2 = rs.Rotor(
        shaft2,
        [gear2, turbine],
        [bearing3, bearing4],
    )

    return rs.MultiRotor(
        rotor1,
        rotor2,
        coupled_nodes=(4, 0),
        orientation_angle=0.0,
        position="below",
    )


def test_add_elements(multi_rotor):
    n_disks = len(multi_rotor.disk_elements)
    n_bearings = len(multi_rotor.bearing_elements)

    disk_driving = rs.DiskElement(n=2, m=10.0, Id=0.1, Ip=0.2)
    disk_driven = rs.DiskElement(n=7, m=1.0, Id=0.01, Ip=0.02)
    seal = rs.SealElement(n=3, kxx=1e6, kyy=0.8e6, cxx=2e2, cyy=1.5e2)

    new_rotor = multi_rotor.add_elements([disk_driving, disk_driven, seal])

    assert isinstance(new_rotor, rs.MultiRotor)
    assert len(new_rotor.disk_elements) == n_disks + 2
    assert len(new_rotor.bearing_elements) == n_bearings + 1

    driving_masses = [d.m for d in new_rotor.rotors["driving"].disk_elements]
    driven_masses = [d.m for d in new_rotor.rotors["driven"].disk_elements]
    assert 10.0 in driving_masses
    assert 1.0 in driven_masses

    driven_disk = next(d for d in new_rotor.rotors["driven"].disk_elements if d.m == 1.0)
    assert driven_disk.n == 2

    assembled_disk = next(d for d in new_rotor.disk_elements if d.m == 1.0)
    assert assembled_disk.n == 7

    with pytest.raises(ValueError, match="does not belong"):
        multi_rotor.add_elements([rs.DiskElement(n=99, m=1.0, Id=0.0, Ip=0.0)])


def test_mesh(multi_rotor):
    assert_allclose(
        multi_rotor.mesh.contact_ratio, 1.6377334309511222, rtol=1e-6, atol=1e-5
    )
    assert_allclose(multi_rotor.mesh.stiffness, 1937234387.18946, rtol=1e-6, atol=1e-5)


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

    assert_allclose(multi_rotor.K_coupling, coupling_matrix, rtol=1e-6, atol=1e-5)


@pytest.fixture
def multi_rotor_with_backlash():

    steel = rs.Material(name="Steel", rho=7850, E=2e11, Poisson=0.3)
    steel_stiff = rs.Material(name="Steel_Stiff", rho=0.01, E=1e15, Poisson=0.3)
    shaft = rs.ShaftElement(n=0, L=0.0001, idl=0.0, odl=0.0001, material=steel_stiff)

    kxx = kyy = 1.0e8
    cxx = cyy = 512.64
    bearing = rs.BearingElement(n=0, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy)

    n_teeth = 20
    module = 0.01
    pitch_diam = module * n_teeth
    width = 0.030
    m = 6.57

    gear = rs.GearElementTVMS(
        n=0,
        material=steel,
        width=width,
        bore_diameter=np.sqrt(pitch_diam**2 - (4 * m) / (np.pi * width * steel.rho)),
        module=module,
        n_teeth=n_teeth,
        pr_angle=rs.Q_(20.0, "deg"),
        helix_angle=0,
        addendum_coeff=1,
        tip_clearance_coeff=0.25,
    )

    rotor1 = rs.Rotor(
        shaft_elements=[shaft], disk_elements=[gear], bearing_elements=[bearing]
    )

    rotor2 = deepcopy(rotor1)

    return rs.MultiRotor(
        driving_rotor=rotor1,
        driven_rotor=rotor2,
        coupled_nodes=(0, 0),
        square_varying_stiffness={"enable": True, "amplitude_ratio": 0.275},
        backlash={
            "enable": True,
            "initial_value": 5e-5,
            "error_amp": 2e-5,
            "smooth_operator": False,
            "sigma": 1e5,
        },
        orientation_angle=0.0,
        position="above",
    )


def test_mesh_with_backlash(multi_rotor_with_backlash):
    T10, T1a = 300.0, 100.0
    T20, T2a = 300.0, 100.0

    speed = rs.Q_(1000, "RPM").to("rad/s").m
    Tm = 2 * np.pi / speed

    tf = 0.25
    n_cycles = int(np.ceil(tf / Tm))
    n_points = 6000

    t = np.linspace(0, n_cycles * Tm, n_cycles * n_points)

    nodes = [
        int(e.n)
        for e in multi_rotor_with_backlash.disk_elements
        if isinstance(e, rs.GearElement)
    ]

    w1 = speed
    w2 = multi_rotor_with_backlash.mesh.gear_ratio * w1
    num_dof = multi_rotor_with_backlash.number_dof

    F = np.zeros((len(t), multi_rotor_with_backlash.ndof))
    F[:, nodes[0] * num_dof + 5] = T10 + T1a * np.sin(w1 * t)
    F[:, nodes[1] * num_dof + 5] = T20 + T2a * np.sin(w2 * t)

    results = multi_rotor_with_backlash.run_time_response(
        speed=speed, t=t, F=F, method="newmark", newmark_type="robust"
    )

    dte = 7.152756140961932e-05
    bt = 6.497875817066328e-05
    Fm = 3191.99620748778
    km = 517925405.2396409
    d = 0.20004375431223567
    alpha = 0.34966621486324245
    cr = 1.5525088765723407

    mesh_results = results.mesh_dynamics

    assert_allclose(np.mean(mesh_results["transmission_error"]), dte, rtol=1e-4)
    assert_allclose(np.mean(mesh_results["backlash"]), bt, rtol=1e-2)
    assert_allclose(np.mean(mesh_results["mesh_force"]), Fm, rtol=1e-2)
    assert_allclose(np.mean(mesh_results["mesh_stiffness"]), km, rtol=1e-2)
    assert_allclose(np.mean(mesh_results["center_distance"]), d, rtol=1e-2)
    assert_allclose(np.mean(mesh_results["pressure_angle"]), alpha, rtol=1e-2)
    assert_allclose(np.mean(mesh_results["contact_ratio"]), cr, rtol=1e-2)
