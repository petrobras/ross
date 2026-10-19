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

    driven_disk = next(
        d for d in new_rotor.rotors["driven"].disk_elements if d.m == 1.0
    )
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


@pytest.mark.parametrize("suffix", [".toml", ".json"])
def test_save_load_multi_rotor(multi_rotor, tmp_path, suffix):
    file = tmp_path / f"multi_rotor{suffix}"
    configured = rs.MultiRotor(
        driving_rotor=multi_rotor.rotors["driving"],
        driven_rotor=multi_rotor.rotors["driven"],
        coupled_nodes=multi_rotor.coupled_nodes,
        gear_mesh_stiffness=multi_rotor.mesh.stiffness,
        orientation_angle=rs.Q_(12, "deg"),
        position="above",
    )

    configured.save(file)
    loaded = rs.Rotor.load(file)

    assert isinstance(loaded, rs.MultiRotor)
    assert loaded.coupled_nodes == configured.coupled_nodes
    assert loaded.mesh.orientation_angle == configured.mesh.orientation_angle
    assert loaded.mesh.stiffness == configured.mesh.stiffness
    assert_allclose(loaded.K(0), configured.K(0))


@pytest.mark.parametrize("suffix", [".toml", ".json"])
def test_save_load_multi_rotor_with_backlash(
    multi_rotor_with_backlash, tmp_path, suffix
):
    file = tmp_path / f"multi_rotor_backlash{suffix}"

    multi_rotor_with_backlash.save(file)
    loaded = rs.MultiRotor.load(file)

    assert loaded.mesh.backlash is not None
    assert loaded.backlash == multi_rotor_with_backlash.backlash
    assert loaded.square_varying_stiffness == (
        multi_rotor_with_backlash.square_varying_stiffness
    )
    assert_allclose(loaded.mesh.stiffness, multi_rotor_with_backlash.mesh.stiffness)


def test_inherited_analyses_preserve_multi_rotor(multi_rotor):
    ucs = multi_rotor.run_ucs(num=2, num_modes=8)
    level1 = multi_rotor.run_level1(n=0, stiffness_range=(1e6, 1e7), num=2)
    convergence = multi_rotor.convergence(err_max=1e3)
    static = multi_rotor.run_static()

    assert isinstance(ucs, rs.UCSResults)
    assert isinstance(level1, rs.Level1Results)
    assert isinstance(convergence, rs.ConvergenceResults)
    assert isinstance(static, rs.StaticResults)
    assert isinstance(multi_rotor, rs.MultiRotor)
    assert multi_rotor.mesh is not None


@pytest.mark.parametrize("node, whirl", [(2, -1j), (7, 1j)])
def test_unbalance_force_whirls_with_shaft(multi_rotor, node, whirl):
    dof = node * multi_rotor.number_dof
    F0 = multi_rotor._unbalance_force(node, 1e-3, 0.3, np.array([50.0, 100.0]))
    ratio = abs(multi_rotor._node_gear_ratio(node))

    assert_allclose(F0[dof + 1], whirl * F0[dof])
    assert_allclose(np.abs(F0[dof]), 1e-3 * (ratio * np.array([50.0, 100.0])) ** 2)


@pytest.mark.parametrize("node", [2, 7])
def test_unbalance_force_matches_time_domain(multi_rotor, node):
    speed = 80.0
    phase = 0.7
    t = np.linspace(0, 0.1, 51)
    frequency = abs(multi_rotor._node_gear_ratio(node)) * speed

    F0 = multi_rotor._unbalance_force(node, 1e-3, phase, [speed])[:, 0]
    F_time = multi_rotor.unbalance_force_over_time([node], [1e-3], [phase], speed, t)

    assert_allclose(
        np.real(np.outer(F0, np.exp(1j * frequency * t))), F_time, atol=1e-9
    )


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


@pytest.mark.parametrize(
    "kwargs, missing, unknown",
    [
        ({"backlash": {"enable": True}}, "initial_value", None),
        (
            {
                "backlash": {
                    "enable": True,
                    "initial_value": 5e-5,
                    "error_amp": 2e-5,
                    "sigma": 1e5,
                }
            },
            "smooth_operator",
            None,
        ),
        (
            {
                "backlash": {
                    "enable": True,
                    "initial_val": 5e-5,
                    "error_amp": 2e-5,
                    "smooth_operator": False,
                    "sigma": 1e5,
                }
            },
            "initial_value",
            "initial_val",
        ),
        ({"square_varying_stiffness": {"enable": True}}, "amplitude_ratio", None),
        (
            {
                "square_varying_stiffness": {
                    "enable": True,
                    "amplitude_ratio": 0.275,
                    "ratio": 0.1,
                }
            },
            None,
            "ratio",
        ),
    ],
)
def test_incomplete_mesh_dicts_raise(
    multi_rotor_with_backlash, kwargs, missing, unknown
):
    rotors = multi_rotor_with_backlash.rotors

    with pytest.raises(ValueError) as error:
        rs.MultiRotor(rotors["driving"], rotors["driven"], (0, 0), **kwargs)

    message = str(error.value)
    assert next(iter(kwargs)) in message
    for key in [missing, unknown]:
        if key is not None:
            assert f"'{key}'" in message


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


@pytest.fixture
def three_shaft_rotors():
    """Three shafts A -> B -> C geared 60/30 and 40/20, so B = -2A and C = 4A."""
    steel = rs.Material(name="steel_three_shaft", rho=7810, E=211e9, G_s=81.2e9)
    pressure_angle = rs.Q_(20, "deg")

    def shaft(n_elements):
        return [
            rs.ShaftElement(L=0.2, idl=0, odl=0.05, material=steel, gyroscopic=True)
            for _ in range(n_elements)
        ]

    def gear(n, n_teeth):
        return rs.GearElement(
            n=n,
            m=5,
            Id=0.02,
            Ip=0.04,
            n_teeth=n_teeth,
            base_diameter=n_teeth * 0.004 * np.cos(pressure_angle.to("rad").m),
            pr_angle=pressure_angle,
        )

    def bearings(n_left, n_right):
        return [
            rs.BearingElement(n=n_left, kxx=1e7, cxx=1e3),
            rs.BearingElement(n=n_right, kxx=1e7, cxx=1e3),
        ]

    rotor_a = rs.Rotor(
        shaft(4),
        [rs.DiskElement(n=1, m=10, Id=0.1, Ip=0.2), gear(4, 60)],
        bearings(0, 3),
    )
    rotor_b = rs.Rotor(shaft(4), [gear(0, 30), gear(4, 40)], bearings(1, 3))
    rotor_c = rs.Rotor(
        shaft(3),
        [gear(0, 20), rs.DiskElement(n=2, m=5, Id=0.05, Ip=0.1)],
        bearings(1, 3),
    )

    return rotor_a, rotor_b, rotor_c


def nest_driving(rotor_a, rotor_b, rotor_c):
    """Build the three-shaft system as MultiRotor(MultiRotor(A, B), C)."""
    rotor_ab = rs.MultiRotor(
        rotor_a, rotor_b, coupled_nodes=(4, 0), gear_mesh_stiffness=1e8
    )
    return rs.MultiRotor(
        rotor_ab, rotor_c, coupled_nodes=(9, 0), gear_mesh_stiffness=1e8
    )


def nest_driven(rotor_a, rotor_b, rotor_c):
    """Build the three-shaft system as MultiRotor(A, MultiRotor(B, C))."""
    rotor_bc = rs.MultiRotor(
        rotor_b, rotor_c, coupled_nodes=(4, 0), gear_mesh_stiffness=1e8
    )
    return rs.MultiRotor(
        rotor_a, rotor_bc, coupled_nodes=(4, 0), gear_mesh_stiffness=1e8
    )


@pytest.mark.parametrize("nest", [nest_driving, nest_driven])
def test_three_shaft_speed_ratios(three_shaft_rotors, nest):
    multi_rotor = nest(*three_shaft_rotors)

    assert [multi_rotor._node_gear_ratio(n) for n in (2, 6, 11)] == [1.0, -2.0, 4.0]


@pytest.mark.parametrize("nest", [nest_driving, nest_driven])
def test_three_shaft_matrices(three_shaft_rotors, nest):
    from scipy.linalg import block_diag

    rotor_a, rotor_b, rotor_c = three_shaft_rotors
    multi_rotor = nest(*three_shaft_rotors)

    assert_allclose(
        multi_rotor.G(), block_diag(rotor_a.G(), -2 * rotor_b.G(), 4 * rotor_c.G())
    )
    assert_allclose(
        multi_rotor.Ksdt(),
        block_diag(rotor_a.Ksdt(), -2 * rotor_b.Ksdt(), 4 * rotor_c.Ksdt()),
    )


def test_three_shaft_nestings_match(three_shaft_rotors):
    by_driving = nest_driving(*three_shaft_rotors)
    by_driven = nest_driven(*three_shaft_rotors)

    assert_allclose(by_driving.K(0, 100), by_driven.K(0, 100))
    assert_allclose(by_driving.C(0, 100), by_driven.C(0, 100))
    assert_allclose(by_driving.M(0, 100), by_driven.M(0, 100))
    assert_allclose(
        by_driving.run_modal(100).wn[:6], by_driven.run_modal(100).wn[:6], rtol=1e-6
    )


@pytest.mark.parametrize("nest", [nest_driving, nest_driven])
@pytest.mark.parametrize("node, whirl", [(2, -1j), (6, 1j), (11, -1j)])
def test_three_shaft_unbalance_force(three_shaft_rotors, nest, node, whirl):
    multi_rotor = nest(*three_shaft_rotors)
    dof = node * multi_rotor.number_dof
    ratio = abs(multi_rotor._node_gear_ratio(node))

    F0 = multi_rotor._unbalance_force(node, 1e-3, 0.0, [10.0])[:, 0]

    assert_allclose(F0[dof + 1], whirl * F0[dof])
    assert_allclose(abs(F0[dof]), 1e-3 * (ratio * 10.0) ** 2)


def test_summary_per_shaft(three_shaft_rotors):
    multi_rotor = nest_driving(*three_shaft_rotors)
    summary = multi_rotor.summary()
    masses = [rotor.m for rotor in three_shaft_rotors]

    assert summary.nodes == [(0, 4), (5, 9), (10, 13)]
    assert_allclose(summary.Ip, [rotor.Ip for rotor in three_shaft_rotors])
    assert_allclose(summary.It, [rotor.It for rotor in three_shaft_rotors])
    assert_allclose(np.dot(masses, summary.CG) / np.sum(masses), multi_rotor.CG)

    rotor_table = summary.plot().data[0]
    headers = list(rotor_table.header["values"])
    table_masses = rotor_table.cells["values"][headers.index("Rotor Mass (kg)")]
    assert_allclose([float(m) for m in table_masses], masses, atol=5e-4)
