import numpy as np
import pytest
from copy import deepcopy
from numpy.testing import assert_allclose

import ross as rs
from ross.motors.motor_element import motor_example
from ross.multi_rotor.mesh import Mesh


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
        rotor1, rotor2, (0, 0), None, False, square, backlash, 0.0, "above"
    )
    assert multi_rotor.mesh.backlash is not None
    assert multi_rotor.mesh.damping_ratio == 0.0
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
    multi_rotor.mesh.stiffness = stiffness_2
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
