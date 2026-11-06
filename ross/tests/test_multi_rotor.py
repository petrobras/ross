import numpy as np
import pytest
from copy import deepcopy
from numpy.testing import assert_allclose

import ross as rs
from ross.motors.motor_element import motor_example
from ross.multi_rotor.mesh import Mesh
from ross.utils import convert_6dof_to_4dof


def build_multi_rotor(**kwargs):
    """Build a spur geared two-shaft rotor system."""

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
        **kwargs,
    )


@pytest.fixture
def multi_rotor():
    """A spur geared two-shaft rotor system."""
    return build_multi_rotor()


def test_add_elements():
    multi_rotor = build_multi_rotor(mesh_damping_ratio=0.05)
    n_disks = len(multi_rotor.disk_elements)
    n_bearings = len(multi_rotor.bearing_elements)

    disk_driving = rs.DiskElement(n=2, m=10.0, Id=0.1, Ip=0.2)
    disk_driven = rs.DiskElement(n=7, m=1.0, Id=0.01, Ip=0.02)
    seal = rs.SealElement(n=3, kxx=1e6, kyy=0.8e6, cxx=2e2, cyy=1.5e2)

    new_rotor = multi_rotor.add_elements([disk_driving, disk_driven, seal])

    assert isinstance(new_rotor, rs.MultiRotor)
    assert len(new_rotor.disk_elements) == n_disks + 2
    assert len(new_rotor.bearing_elements) == n_bearings + 1
    assert new_rotor.mesh.damping_ratio == 0.05

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


@pytest.mark.parametrize(
    "invalid",
    [
        "junk",
        None,
        rs.Material(name="add_elements_mat", rho=7800, E=2e11, G_s=8e10),
        [rs.DiskElement(n=2, m=1.0, Id=0.01, Ip=0.02)],
    ],
    ids=["str", "None", "Material", "nested-list"],
)
def test_add_elements_rejects_invalid(multi_rotor, invalid):
    # Same message as Rotor.add_elements, which raised ValueError in 3.0.0.
    with pytest.raises(ValueError, match="is not a valid element"):
        multi_rotor.add_elements([invalid])


def test_add_elements_motor(multi_rotor):
    motor = motor_example()

    new_rotor = multi_rotor.add_elements([motor])

    assert multi_rotor.motor_element is None
    assert new_rotor.motor_element == motor
    assert new_rotor.rotors["driving"].motor_element == motor


def test_new_parameters_keep_3_0_positional_calls():
    rotor1, rotor2 = gear_pair_rotors()
    square = {"enable": True, "amplitude_ratio": 0.275}
    backlash = {
        "enable": True,
        "initial_value": 5e-5,
        "error_amp": 2e-5,
        "smooth_operator": False,
        "sigma": 1e5,
    }

    multi_rotor = rs.MultiRotor(
        rotor1, rotor2, (0, 0), None, False, square, backlash, 0.05, 0.0, "above"
    )
    assert multi_rotor.mesh.backlash is not None
    assert multi_rotor.mesh.damping_ratio == 0.05
    assert multi_rotor.mesh.orientation_angle == 0.0
    assert multi_rotor.dy_pos >= 0

    gear_1 = multi_rotor.mesh.driving_gear
    gear_2 = multi_rotor.mesh.driven_gear
    mesh = Mesh(gear_1, gear_2, None, square, backlash, 0.07)
    assert mesh.backlash is not None
    assert mesh.damping_ratio == 0.07


@pytest.mark.parametrize(
    "mesh_damping_ratio",
    [0.07, rs.Q_(0.07, "dimensionless"), rs.Q_(7, "percent")],
)
def test_mesh_damping_ratio_units(mesh_damping_ratio):
    multi_rotor = build_multi_rotor(mesh_damping_ratio=mesh_damping_ratio)

    assert multi_rotor.mesh.damping_ratio == pytest.approx(0.07)


def test_mesh(multi_rotor):
    assert_allclose(
        multi_rotor.mesh.contact_ratio, 1.6377334309511222, rtol=1e-6, atol=1e-5
    )
    assert_allclose(multi_rotor.mesh.stiffness, 1937234387.18946, rtol=1e-6, atol=1e-5)
    assert multi_rotor.mesh.backlash is None


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
    static = multi_rotor.run_static()

    assert isinstance(ucs, rs.UCSResults)
    assert isinstance(level1, rs.Level1Results)
    assert isinstance(static, rs.StaticResults)
    assert isinstance(multi_rotor, rs.MultiRotor)

    with pytest.raises(NotImplementedError, match="refine is not available"):
        multi_rotor.run_mesh_convergence()
    assert multi_rotor.mesh is not None


@pytest.mark.parametrize("log_k", [6, 8, 10])
def test_ucs_keeps_every_multi_rotor_mode(multi_rotor, log_k):
    num_modes = 12
    ucs = multi_rotor.run_ucs(
        stiffness_range=(log_k, log_k + 1), num=2, num_modes=num_modes
    )

    # Reference: the multi-rotor assembled again with the UCS bearings. The
    # gear mesh couples the lateral planes, so no mode is a repeated pair.
    rotors = [
        rs.Rotor(
            rotor.shaft_elements,
            rotor.disk_elements,
            [
                rs.BearingElement(n=brg.n, kxx=10.0**log_k, cxx=0)
                for brg in rotor.bearing_elements
            ],
        )
        for rotor in multi_rotor.rotors.values()
    ]
    reference = rs.MultiRotor(
        *rotors,
        coupled_nodes=multi_rotor.coupled_nodes,
        orientation_angle=0.0,
        position="below",
    )
    wn = convert_6dof_to_4dof(reference).run_modal(0, num_modes=num_modes).wn

    assert_allclose(ucs.wn[:, 0], wn[: num_modes // 2], rtol=1e-6)


@pytest.mark.parametrize("shaft, node", [(0, 2), (1, 7)])
def test_critical_speed_at_shaft_speed(multi_rotor, shaft, node):
    ratio = abs(multi_rotor._node_gear_ratio(node))
    rtol = 0.005
    results = multi_rotor.run_critical_speed(num_modes=8, rtol=rtol, shaft=shaft)

    # At a critical speed the 1X excitation of the node's shaft equals the
    # natural frequency of the mode; rtol is the Newton stopping tolerance.
    for i, speed in enumerate(results.wn()):
        modal = multi_rotor.run_modal(speed, num_modes=8, frequency=ratio * speed)
        assert_allclose(ratio * speed, modal.wn[i], rtol=rtol)


@pytest.mark.parametrize("node, whirl", [(2, -1j), (7, 1j)])
def test_unbalance_force_whirls_with_shaft(multi_rotor, node, whirl):
    dof = node * multi_rotor.number_dof
    F0 = multi_rotor._unbalance_force(node, 1e-3, 0.3, np.array([50.0, 100.0]))
    ratio = abs(multi_rotor._node_gear_ratio(node))

    assert_allclose(F0[dof + 1], whirl * F0[dof])
    assert_allclose(np.abs(F0[dof]), 1e-3 * (ratio * np.array([50.0, 100.0])) ** 2)


@pytest.mark.parametrize("node", [2, 7])
def test_deflected_shape_plots_run_for_each_shaft(multi_rotor, node):
    speed_range = np.linspace(50, 150, 5)
    response = multi_rotor.run_unbalance_response(
        node=node,
        unbalance_magnitude=1e-3,
        unbalance_phase=0,
        speed_range=speed_range,
    )

    fig_3d = response.plot_deflected_shape_3d(speed=speed_range[2])
    fig_2d = response.plot_deflected_shape_2d(speed=speed_range[2])

    unbalance_marker = [
        trace for trace in fig_3d.data if trace.legendgroup == "Unbalance"
    ][0]
    assert_allclose(unbalance_marker.y, [multi_rotor.nodes_pos[node]] * 2)
    assert len(fig_2d.data) > 0


def test_deflected_shape_3d_separates_center_lines(multi_rotor):
    speed_range = np.linspace(50, 150, 5)
    response = multi_rotor.run_unbalance_response(
        node=7,
        unbalance_magnitude=1e-3,
        unbalance_phase=0,
        speed_range=speed_range,
    )

    fig = response.plot_deflected_shape_3d(speed=speed_range[2])

    center_lines = [
        trace
        for trace in fig.data
        if trace.mode == "lines"
        and trace.line.dash == "dashdot"
        and trace.hoverinfo == "none"
        and trace.legendgroup is None
    ]
    levels = sorted({trace.z[0] for trace in center_lines})
    assert len(levels) == 2
    driving, driven = levels[1], levels[0]
    assert driving == 0
    assert driven < 0

    node_z = {
        node: trace.z[0]
        for node in (2, 7)
        for trace in fig.data
        if trace.name == f"node {node}" and trace.mode == "markers"
    }
    assert abs(node_z[2] - driving) < abs(node_z[2] - driven)
    assert abs(node_z[7] - driven) < abs(node_z[7] - driving)


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

    assert_allclose(multi_rotor.coupling_matrix, coupling_matrix, rtol=1e-6, atol=1e-5)
    assert_allclose(multi_rotor.coupling_matrix, multi_rotor.coupling_matrix.T)


def test_mesh_damping_matrix(multi_rotor):
    frequency = 0.0
    base_C = multi_rotor._join_matrices(
        multi_rotor.rotors["driving"].C(frequency, frequency),
        multi_rotor.rotors["driven"].C(
            frequency, frequency * multi_rotor.mesh.gear_ratio
        ),
    )

    assert_allclose(multi_rotor.C(frequency), base_C)

    damped = build_multi_rotor(mesh_damping_ratio=0.07)
    actual = damped.C(frequency)
    dofs_1 = damped.mesh.driving_gear.dof_global_index.values()
    dofs_2 = damped.mesh.driven_gear.dof_global_index.values()
    dofs = [*dofs_1, *dofs_2]
    mesh_block = (actual - base_C)[np.ix_(dofs, dofs)]
    nonzero = np.abs(damped.coupling_matrix) > 1e-12
    c_m = mesh_block[nonzero] / damped.coupling_matrix[nonzero]

    assert c_m[0] > 0
    assert_allclose(c_m, c_m[0])
    assert_allclose(mesh_block[~nonzero], 0.0, atol=1e-12)
    first_ndof = damped.rotors["driving"].ndof
    assert np.any(actual[:first_ndof, first_ndof:] != 0.0)
    assert_allclose(actual, actual.T)


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.filterwarnings("ignore:k >= N - 1:RuntimeWarning")
@pytest.mark.parametrize("zeta", [0.0, 0.01, 0.07, 0.2, 0.5])
def test_mesh_damping_ratio_of_the_torsional_mesh_mode(zeta):
    # Rigid gears on laterally rigid, undamped supports: the only elastic mode
    # is the torsional mesh mode, a single-DOF oscillator on the line of action
    # with mass M_eq, stiffness k and damping c_m. Its natural frequency is
    # sqrt(k / M_eq) and its modal damping ratio is the input zeta.
    k = 1e8
    rotor1, rotor2 = gear_pair_rotors(bearing_k=1e14, bearing_c=0.0)
    multi_rotor = rs.MultiRotor(
        rotor1,
        rotor2,
        coupled_nodes=(0, 0),
        gear_mesh_stiffness=k,
        mesh_damping_ratio=zeta,
    )
    gear_1 = multi_rotor.mesh.driving_gear
    gear_2 = multi_rotor.mesh.driven_gear
    M_eq = 1 / (gear_1.base_radius**2 / gear_1.Ip + gear_2.base_radius**2 / gear_2.Ip)
    wn_expected = np.sqrt(k / M_eq)

    evalues = multi_rotor.run_modal(speed=0, num_modes=2 * multi_rotor.ndof).evalues
    wn = np.abs(evalues)
    mesh_mode = np.argmin(np.abs(wn - wn_expected))

    assert_allclose(wn[mesh_mode], wn_expected, rtol=1e-5)
    assert_allclose(-evalues[mesh_mode].real / wn[mesh_mode], zeta, atol=1e-5)


def test_mesh_damping_updates_with_stiffness():
    multi_rotor = build_multi_rotor(mesh_damping_ratio=0.07)
    stiffness_1 = multi_rotor.mesh.stiffness
    stiffness_2 = 1.5 * stiffness_1

    C_mesh_1 = multi_rotor.C_mesh(np.zeros((multi_rotor.ndof, multi_rotor.ndof)))
    multi_rotor._mesh_stiffness = stiffness_2
    C_mesh_2 = multi_rotor.C_mesh(np.zeros((multi_rotor.ndof, multi_rotor.ndof)))

    dofs_1 = multi_rotor.mesh.driving_gear.dof_global_index.values()
    dofs_2 = multi_rotor.mesh.driven_gear.dof_global_index.values()
    dofs = [*dofs_1, *dofs_2]
    nonzero = np.abs(multi_rotor.coupling_matrix) > 1e-12
    ratio = (
        C_mesh_2[np.ix_(dofs, dofs)][nonzero] / C_mesh_1[np.ix_(dofs, dofs)][nonzero]
    )

    assert_allclose(ratio, np.sqrt(stiffness_2 / stiffness_1))


def gear_pair_rotors(bearing_k=1.0e8, bearing_c=512.64):
    """Build two identical single-gear rotors for a gear pair."""
    steel = rs.Material(name="Steel", rho=7850, E=2e11, Poisson=0.3)
    steel_stiff = rs.Material(name="Steel_Stiff", rho=0.01, E=1e15, Poisson=0.3)
    shaft = rs.ShaftElement(n=0, L=0.0001, idl=0.0, odl=0.0001, material=steel_stiff)

    bearing = rs.BearingElement(
        n=0, kxx=bearing_k, kyy=bearing_k, cxx=bearing_c, cyy=bearing_c
    )

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

    return rotor1, deepcopy(rotor1)


@pytest.fixture
def multi_rotor_with_backlash():
    rotor1, rotor2 = gear_pair_rotors()

    # ROSS 3.0 used 0.07 for backlash models; the reference values below use it.
    return rs.MultiRotor(
        driving_rotor=rotor1,
        driven_rotor=rotor2,
        coupled_nodes=(0, 0),
        square_varying_stiffness={"enable": True, "amplitude_ratio": 0.275},
        mesh_damping_ratio=0.07,
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
    assert_allclose(
        multi_rotor_with_backlash.mesh.backlash.M_eq,
        multi_rotor_with_backlash.mesh.M_eq,
    )

    frequency = 0.0
    expected_C = multi_rotor_with_backlash._join_matrices(
        multi_rotor_with_backlash.rotors["driving"].C(frequency, frequency),
        multi_rotor_with_backlash.rotors["driven"].C(
            frequency, frequency * multi_rotor_with_backlash.mesh.gear_ratio
        ),
    )
    expected_K = multi_rotor_with_backlash._join_matrices(
        multi_rotor_with_backlash.rotors["driving"].K(frequency, frequency),
        multi_rotor_with_backlash.rotors["driven"].K(
            frequency, frequency * multi_rotor_with_backlash.mesh.gear_ratio
        ),
    )
    assert_allclose(multi_rotor_with_backlash.C(frequency), expected_C)
    assert_allclose(multi_rotor_with_backlash.K(frequency), expected_K)

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
def test_three_shaft_critical_speed(three_shaft_rotors, nest):
    multi_rotor = nest(*three_shaft_rotors)
    rtol = 0.005
    results = multi_rotor.run_critical_speed(num_modes=4, rtol=rtol, shaft=2)

    # Shaft C turns 4 times faster than shaft A (gears 60/30 and 40/20).
    for i, speed in enumerate(results.wn()):
        modal = multi_rotor.run_modal(speed, num_modes=4, frequency=4 * speed)
        assert_allclose(4 * speed, modal.wn[i], rtol=rtol)

    with pytest.raises(ValueError, match="shaft must be between 0 and 2"):
        multi_rotor.run_critical_speed(shaft=3)


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


@pytest.fixture
def fault_shaft_ratios(multi_rotor):
    """Shaft element numbers on each shaft and the expected speed ratio."""
    driving_gear = multi_rotor.mesh.driving_gear
    driven_gear = multi_rotor.mesh.driven_gear
    driving_elm = multi_rotor.shaft_elements[0].n
    driven_elm = next(
        elm.n for elm in multi_rotor.shaft_elements if elm.n in multi_rotor.driven_nodes
    )
    return [
        (driving_elm, 1.0),
        (driven_elm, -driving_gear.n_teeth / driven_gear.n_teeth),
    ]


@pytest.mark.parametrize("shaft", [0, 1])
def test_rubbing_uses_shaft_speed(multi_rotor, fault_shaft_ratios, monkeypatch, shaft):
    n, ratio = fault_shaft_ratios[shaft]
    speed = 100.0
    speeds = []
    compute = rs.faults.rubbing.Rubbing.compute_rubbing_force

    def spy(self, y, ydot, ang_speed):
        speeds.append(ang_speed)
        return compute(self, y, ydot, ang_speed)

    monkeypatch.setattr(rs.faults.rubbing.Rubbing, "compute_rubbing_force", spy)

    multi_rotor.run_rubbing(
        n=n,
        distance=1e-3,
        contact_stiffness=1e6,
        contact_damping=40,
        friction_coeff=0.3,
        node=[2],
        unbalance_magnitude=[1e-4],
        unbalance_phase=[0.0],
        speed=speed,
        t=np.linspace(0, 0.01, 11),
    )

    assert_allclose(speeds, ratio * speed)


@pytest.mark.parametrize("shaft", [0, 1])
def test_crack_uses_shaft_angle(multi_rotor, fault_shaft_ratios, monkeypatch, shaft):
    n, ratio = fault_shaft_ratios[shaft]
    speed = 100.0
    t = np.linspace(0, 0.01, 11)
    angles = []
    mayes = rs.faults.crack.Crack.mayes

    def spy(self, ap):
        angles.append(ap)
        return mayes(self, ap)

    monkeypatch.setattr(rs.faults.crack.Crack, "mayes", spy)

    multi_rotor.run_crack(
        n=n,
        depth_ratio=0.2,
        node=[2],
        unbalance_magnitude=[1e-4],
        unbalance_phase=[0.0],
        speed=speed,
        t=t,
    )

    assert_allclose(angles, ratio * speed * t[1 : len(angles) + 1])


def test_refine_mesh_refines_each_rotor_of_a_multi_rotor():
    from ross.multi_rotor.multi_rotor import two_shaft_rotor_example

    multi_rotor = two_shaft_rotor_example()
    refined = multi_rotor.refine_mesh(max_ld=0.5)
    for name in ("driving", "driven"):
        before = multi_rotor.rotors[name]
        after = refined.rotors[name]
        assert after.m == pytest.approx(before.m)
        ratios = [elm.L / max(elm.odl, elm.odr) for elm in after.shaft_elements]
        assert max(ratios) <= 0.5 * (1 + 1e-9)
        assert len(after.shaft_elements) >= len(before.shaft_elements)
    assert refined.m == pytest.approx(multi_rotor.m)


def test_rebuild_preserves_mesh_configuration(multi_rotor_with_backlash):
    multi_rotor = multi_rotor_with_backlash
    rebuilt_rotors = [
        multi_rotor.add_nodes([5e-5]),
        multi_rotor.add_elements([rs.DiskElement(n=0, m=0.1, Id=0.01, Ip=0.02)]),
    ]

    for rebuilt in rebuilt_rotors:
        assert rebuilt.mesh.backlash is not None
        assert rebuilt.mesh.stiffness_type == multi_rotor.mesh.stiffness_type
        assert rebuilt.mesh.damping_ratio == multi_rotor.mesh.damping_ratio
        assert (
            rebuilt.mesh.backlash.damping_ratio
            == multi_rotor.mesh.backlash.damping_ratio
        )


@pytest.mark.parametrize("parts", [2, 3])
def test_refine_mesh_parts_refines_each_rotor_of_a_multi_rotor(parts):
    from ross.multi_rotor.multi_rotor import two_shaft_rotor_example

    multi_rotor = two_shaft_rotor_example()
    refined = multi_rotor.refine_mesh(parts=parts)
    for name in ("driving", "driven"):
        expected = multi_rotor.rotors[name].refine(parts)
        after = refined.rotors[name]
        assert len(after.shaft_elements) == parts * len(
            multi_rotor.rotors[name].shaft_elements
        )
        assert_allclose(after.nodes_pos, expected.nodes_pos, atol=1e-12)
        assert [elm.n for elm in after.disk_elements] == [
            elm.n for elm in expected.disk_elements
        ]
        assert [elm.n for elm in after.bearing_elements] == [
            elm.n for elm in expected.bearing_elements
        ]

    expected = multi_rotor._rebuild(
        multi_rotor.rotors["driving"].refine(parts),
        multi_rotor.rotors["driven"].refine(parts),
    )
    # The dense eigensolver: the sparse one moves single frequencies by 1e-9
    # between matrices that differ by round-off only.
    assert_allclose(
        refined.run_modal(speed=0, num_modes=12, sparse=False).wn[:12],
        expected.run_modal(speed=0, num_modes=12, sparse=False).wn[:12],
        rtol=1e-9,
    )


@pytest.mark.parametrize("shaft", [0, 1])
def test_misalignment_flex_uses_shaft_angle(
    multi_rotor, fault_shaft_ratios, monkeypatch, shaft
):
    n, ratio = fault_shaft_ratios[shaft]
    speed = 100.0
    t = np.linspace(0, 0.01, 11)
    angles = []
    parallel = rs.faults.misalignment.MisalignmentFlex._parallel

    def spy(self, ang_pos):
        angles.append(ang_pos)
        return parallel(self, ang_pos)

    monkeypatch.setattr(rs.faults.misalignment.MisalignmentFlex, "_parallel", spy)

    multi_rotor.run_misalignment(
        coupling="flex",
        n=n,
        mis_type="parallel",
        mis_distance_x=2e-4,
        mis_distance_y=2e-4,
        mis_angle=0,
        radial_stiffness=40e3,
        bending_stiffness=38e3,
        node=[2],
        unbalance_magnitude=[1e-4],
        unbalance_phase=[0.0],
        speed=speed,
        t=t,
    )

    assert_allclose(angles[0], ratio * speed * t)


@pytest.mark.parametrize("shaft", [0, 1])
def test_misalignment_rigid_uses_shaft_angle(
    multi_rotor, fault_shaft_ratios, monkeypatch, shaft
):
    n, ratio = fault_shaft_ratios[shaft]
    speed = 100.0
    t = np.linspace(0, 0.01, 11)
    angles = []
    compute = rs.faults.misalignment.MisalignmentRigid.compute_reaction_force

    def spy(self, y, ap):
        angles.append(ap)
        return compute(self, y, ap)

    monkeypatch.setattr(
        rs.faults.misalignment.MisalignmentRigid, "compute_reaction_force", spy
    )

    multi_rotor.run_misalignment(
        coupling="rigid",
        n=n,
        mis_distance=2e-4,
        node=[2],
        unbalance_magnitude=[1e-4],
        unbalance_phase=[0.0],
        speed=speed,
        t=t,
    )

    assert_allclose(angles, ratio * speed * t[1 : len(angles) + 1])


@pytest.mark.parametrize("nest", [nest_driving, nest_driven])
@pytest.mark.parametrize("node", [2, 6, 11])
def test_three_shaft_harmonic_balance(three_shaft_rotors, nest, node):
    multi_rotor = nest(*three_shaft_rotors)
    ratio = abs(multi_rotor._node_gear_ratio(node))
    harmonic = int(ratio)
    speed = 50.0
    magnitude = 1e-3
    phase = 0.3

    results = multi_rotor.run_harmonic_balance_response(
        speed=speed,
        t=np.linspace(0, 2 * np.pi / speed, 201),
        harmonic_forces=[
            {
                "node": node,
                "magnitudes": [magnitude * (ratio * speed) ** 2],
                "phases": [phase],
                "harmonics": [1],
            }
        ],
        n_harmonics=harmonic,
    )

    # Reference: the unbalance response, which excites each shaft at its own
    # speed and whirl direction through a separate code path.
    reference = multi_rotor.run_unbalance_response(
        node, magnitude, phase, speed_range=[speed]
    ).forced_resp[:, 0]

    assert_allclose(results.dQ[:, harmonic - 1], reference, rtol=1e-6, atol=1e-15)
    assert_allclose(np.delete(results.dQ, harmonic - 1, axis=1), 0, atol=1e-15)
