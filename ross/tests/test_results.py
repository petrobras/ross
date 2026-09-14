from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

import ross as rs
from ross import Q_, Probe
from ross.results import *
from ross.rotor_assembly import *
from ross.bearings.magnetic.amb_models import rotor_example_amb_complex_controllers
from ross.utils import equal_dicts
from ross.bearings.magnetic.amb_time_response import AmbTimeResponse


@pytest.fixture
def rotor1():
    return rotor_example()


@pytest.fixture
def rotor_amb():
    return rotor_example_amb_complex_controllers()


def test_save_load_campbell(rotor1):
    speed = np.linspace(0, 1000, 51)
    response = rotor1.run_campbell(speed)

    file = Path(tempdir) / "campbell.toml"
    response.save(file)
    response2 = CampbellResults.load(file)

    assert response2.speed_range.all() == response.speed_range.all()
    assert response2.wd.all() == response.wd.all()
    assert response2.log_dec.all() == response.log_dec.all()
    assert response2.whirl_values.all() == response.whirl_values.all()


def test_save_load_criticalspeed(rotor1):
    response = rotor1.run_critical_speed()

    file = Path(tempdir) / "critical_speed.toml"
    response.save(file)
    response2 = CriticalSpeedResults.load(file)

    assert response2._wn.all() == response._wn.all()
    assert response2._wd.all() == response._wd.all()
    assert response2.log_dec.all() == response.log_dec.all()
    assert response2.damping_ratio.all() == response.damping_ratio.all()


def test_save_load_modal(rotor1):
    response = rotor1.run_modal(0)

    file = Path(tempdir) / "modal.toml"
    response.save(file)
    response2 = ModalResults.load(file)

    assert response2.speed == response.speed
    assert response2.evalues.all() == response.evalues.all()
    assert response2.evectors.all() == response.evectors.all()
    assert response2.wn.all() == response.wn.all()
    assert response2.wd.all() == response.wd.all()
    assert response2.damping_ratio.all() == response.damping_ratio.all()
    assert response2.log_dec.all() == response.log_dec.all()
    assert response2.ndof == response.ndof
    assert np.array(response2.nodes).all() == np.array(response.nodes).all()
    assert np.array(response2.nodes_pos).all() == np.array(response.nodes_pos).all()
    assert (
        np.array(response2.shaft_elements_length).all()
        == np.array(response.shaft_elements_length).all()
    )


def test_save_load_freqresponse(rotor1):
    speed = np.linspace(0, 1000, 11)
    response = rotor1.run_freq_response(speed_range=speed)

    file = Path(tempdir) / "frf.toml"
    response.save(file)
    response2 = FrequencyResponseResults.load(file)

    assert response2.freq_resp.all() == response.freq_resp.all()
    assert response2.speed_range.all() == response.speed_range.all()
    assert response2.velc_resp.all() == response.velc_resp.all()
    assert response2.accl_resp.all() == response.accl_resp.all()


def test_save_load_unbalance_response(rotor1):
    speed = np.linspace(0, 1000, 51)
    response = rotor1.run_unbalance_response(3, 0.01, 0.0, speed)

    file = Path(tempdir) / "unbalance.toml"
    response.save(file)
    response2 = ForcedResponseResults.load(file)

    assert response2.rotor == response.rotor
    assert response2.forced_resp.all() == response.forced_resp.all()
    assert response2.speed_range.all() == response.speed_range.all()
    assert response2.velc_resp.all() == response.velc_resp.all()
    assert response2.accl_resp.all() == response.accl_resp.all()
    assert response2.unbalance.all() == response.unbalance.all()


def test_save_load_static(rotor1):
    response = rotor1.run_static()

    file = Path(tempdir) / "static.toml"
    response.save(file)
    response2 = StaticResults.load(file)

    assert np.array(response2.deformation).all() == np.array(response.deformation).all()
    assert np.array(response2.Vx).all() == np.array(response.Vx).all()
    assert np.array(response2.Bm).all() == np.array(response.Bm).all()
    assert np.array(response2.w_shaft).all() == np.array(response.w_shaft).all()
    assert response2.disk_forces == response.disk_forces
    assert response2.bearing_forces == response.bearing_forces
    assert np.array(response2.nodes).all() == np.array(response.nodes).all()
    assert np.array(response2.nodes_pos).all() == np.array(response.nodes_pos).all()
    assert np.array(response2.Vx_axis).all() == np.array(response.Vx_axis).all()


def test_save_load_convergence(rotor1):
    response = rotor1.convergence()

    file = Path(tempdir) / "convergence.toml"
    response.save(file)
    response2 = ConvergenceResults.load(file)

    assert response2.el_num.all() == response.el_num.all()
    assert response2.eigv_arr.all() == response.eigv_arr.all()
    assert response2.error_arr.all() == response.error_arr.all()


def test_save_load_timeresponse(rotor1):
    speed = 500.0
    size = 1000
    node = 3
    t = np.linspace(0, 10, size)
    F = np.zeros((size, rotor1.ndof))
    F[:, rotor1.number_dof * node] = 10 * np.cos(2 * t)
    F[:, rotor1.number_dof * node + 1] = 10 * np.sin(2 * t)
    response = rotor1.run_time_response(speed, F, t)

    file = Path(tempdir) / "time.toml"
    response.save(file)
    response2 = TimeResponseResults.load(file)

    assert response2.t.all() == response.t.all()
    assert response2.yout.all() == response.yout.all()
    assert response2.xout.all() == response.xout.all()
    assert response2.rotor == response.rotor


def test_save_load_sensitivity(rotor_amb):
    result = rotor_amb.run_amb_sensitivity(
        speed=0,
        t_max=5e-4,
        dt=1e-4,
        disturbance_amplitude=10e-6,
        disturbance_min_frequency=0.001,
        disturbance_max_frequency=150,
        amb_tags=["Magnetic Bearing 0"],
    )

    file_amb = Path(tempdir) / "amb_sensitivities.toml"

    result.save(file_amb)
    result_load = SensitivityResults.load(file_amb)
    compare_results = equal_dicts(vars(result), vars(result_load))

    # Show what is different between results
    if not compare_results[0]:
        print(f"The results are different: {compare_results[1]}")

    assert compare_results[0]


def test_save_load_campbell_json(rotor1):
    speed = np.linspace(0, 1000, 51)
    response = rotor1.run_campbell(speed)

    file = Path(tempdir) / "campbell.json"
    response.save(file)
    response2 = CampbellResults.load(file)

    assert response2.speed_range.all() == response.speed_range.all()
    assert response2.wd.all() == response.wd.all()
    assert response2.log_dec.all() == response.log_dec.all()
    assert response2.whirl_values.all() == response.whirl_values.all()


def test_save_load_unbalance_response_json(rotor1):
    speed = np.linspace(0, 1000, 51)
    response = rotor1.run_unbalance_response(3, 0.01, 0.0, speed)

    file = Path(tempdir) / "unbalance.json"
    response.save(file)
    response2 = ForcedResponseResults.load(file)

    assert response2.rotor == response.rotor
    assert response2.forced_resp.all() == response.forced_resp.all()
    assert response2.speed_range.all() == response.speed_range.all()
    assert response2.velc_resp.all() == response.velc_resp.all()
    assert response2.accl_resp.all() == response.accl_resp.all()
    assert response2.unbalance.all() == response.unbalance.all()


def test_campbell_plot(rotor1):
    speed = np.linspace(0, 400, 101)
    camp = rotor1.run_campbell(speed)
    fig = camp.plot(
        harmonics=[1, 2],
        damping_parameter="damping_ratio",
        frequency_range=Q_((2000, 10000), "RPM"),
        damping_range=(-0.1, 100),
        frequency_units="RPM",
    )
    crit_array_x = np.array(
        [
            2590.2641754,
            1306.51513941,
            2868.14592367,
            1420.76907353,
            3264.81334336,
        ]
    )
    crit_array_y = np.array(
        [
            2590.2641754,
            2613.03027882,
            2868.14592367,
            2841.53814705,
            6529.62668672,
        ]
    )
    assert_allclose(fig.data[0]["x"], crit_array_x)
    assert_allclose(fig.data[0]["y"], crit_array_y)


def test_orbit():
    orb = Orbit(node=0, node_pos=0, ru_e=(1 + 1j), rv_e=(1 - 1j))
    assert_allclose(orb.minor_axis, np.sqrt(2))
    assert_allclose(orb.major_axis, np.sqrt(2))
    assert_allclose(orb.kappa, 1)
    assert orb.whirl == "Forward"

    orb = Orbit(node=0, node_pos=0, ru_e=(1 - 1j), rv_e=(1 + 1j))
    assert_allclose(orb.minor_axis, np.sqrt(2))
    assert_allclose(orb.major_axis, np.sqrt(2))
    assert_allclose(orb.kappa, -1)
    assert orb.whirl == "Backward"


def test_orbit_calculate_amplitude():
    # create orbit with major axis at 45deg
    orb = Orbit(node=0, node_pos=0, ru_e=(2 + 1j), rv_e=(2 - 1j))

    assert_allclose(orb.calculate_amplitude(Q_(0, "deg"))[0], 2.23606797749979)
    assert_allclose(orb.calculate_amplitude(Q_(45, "deg"))[0], 2.8284271247461903)
    assert_allclose(
        orb.calculate_amplitude(Q_(135, "deg"))[0], 1.4142135623730947, rtol=1e-3
    )
    assert_allclose(orb.calculate_amplitude("minor")[0], 1.4142135623730947)
    assert_allclose(orb.calculate_amplitude("major")[0], 2.8284271247461903)


def test_plot_orbit_lateral_mode(rotor1):
    modal = rotor1.run_modal(speed=Q_(4000, "RPM"), num_modes=14)
    lateral_mode = next(
        i for i, shape in enumerate(modal.shapes) if shape.mode_type == "Lateral"
    )

    fig = modal.plot_orbit(lateral_mode, nodes=[2, 4])

    assert len(fig.data) == 4
    orbit_node_2 = next(
        orbit for orbit in modal.shapes[lateral_mode].orbits if orbit.node == 2
    )
    assert_allclose(fig.data[0]["x"], orbit_node_2.x_circle[:-10])
    assert_allclose(fig.data[0]["y"], orbit_node_2.y_circle[:-10])


def test_plot_orbit_non_lateral_mode(rotor1):
    modal = rotor1.run_modal(speed=Q_(4000, "RPM"), num_modes=14)
    non_lateral_mode = next(
        i for i, shape in enumerate(modal.shapes) if shape.mode_type != "Lateral"
    )
    shape = modal.shapes[non_lateral_mode]

    assert shape.mode_type == "Torsional"
    assert shape.orbits is None

    with pytest.warns(UserWarning, match="has no orbit"):
        fig = modal.plot_orbit(non_lateral_mode, nodes=[2, 4])

    assert len(fig.data) == 0
    assert "Torsional mode has no orbit." in [
        annotation["text"] for annotation in fig.layout.annotations
    ]


def test_probe_response(rotor1):
    speed = 500.0
    size = 50
    node = 3
    t = np.linspace(0, 10, size)
    F = np.zeros((size, rotor1.ndof))
    F[:, rotor1.number_dof * node] = 10 * np.cos(2 * t)
    F[:, rotor1.number_dof * node + 1] = 10 * np.sin(2 * t)
    response = rotor1.run_time_response(speed, F, t)

    probe1 = Probe(3, Q_(0, "deg"))  # node 3, orientation 0° (X dir.)
    probe2 = Probe(3, Q_(90, "deg"))  # node 3, orientation 90°(Y dir.)
    resp_prob1 = np.array(
        [0.00000000e00, 4.07504756e-06, 1.19778973e-05, 1.68562228e-05, 1.34097882e-05]
    )
    resp_prob2 = np.array(
        [0.00000000e00, 4.13295078e-06, 8.25529257e-06, 1.28932310e-05, 1.59791798e-05]
    )
    data = response.data_time_response(probe=[probe1, probe2])
    assert_allclose(data["probe_resp[0]"].to_numpy()[:5], resp_prob1)
    assert_allclose(data["probe_resp[1]"].to_numpy()[:5], resp_prob2)


def test_summary_results(rotor1):
    s = rotor1.summary()

    def column(fig, header):
        for table in fig.data:
            headers = list(table.header["values"])
            if header in headers:
                values = table.cells["values"][headers.index(header)]
                return np.array([float(v) for v in values])
        raise KeyError(header)

    fig_default = s.plot()
    fig_mm = s.plot(length_units="mm", mass_units="g", force_units="mN")

    assert "Length (m)" in list(fig_default.data[1].header["values"])
    assert "Length (mm)" in list(fig_mm.data[1].header["values"])

    assert_allclose(
        column(fig_mm, "Length (mm)"),
        column(fig_default, "Length (m)") * 1000,
        rtol=1e-2,
    )
    assert_allclose(
        column(fig_mm, "Mass (g)"),
        column(fig_default, "Mass (kg)") * 1000,
        rtol=1e-2,
    )

    fig_imperial = s.plot(length_units="in", mass_units="lb", force_units="lbf")
    assert fig_imperial is not None


@pytest.fixture
def rotor_with_clearances():
    rotor = rotor_example()
    bearings = [
        rs.BearingElement(
            n=0, kxx=1e6, cxx=1e3, radial_clearance=Q_(100, "um"), tag="DE"
        ),
        rs.BearingElement(
            n=6, kxx=1e6, cxx=1e3, radial_clearance=Q_(120, "um"), tag="NDE"
        ),
        rs.SealElement(n=3, kxx=0, cxx=0, radial_clearance=Q_(250, "um"), tag="eye"),
        rs.SealElement(n=2, kxx=0, cxx=0, tag="no_clearance"),
    ]
    return rs.Rotor(rotor.shaft_elements, rotor.disk_elements, bearings)


@pytest.fixture
def clearance_probes():
    return [
        Probe(0, Q_(45, "deg"), tag="DE-45"),
        Probe(6, Q_(-45, "deg"), tag="NDE-45"),
        Probe(3, direction="axial", tag="axial"),
    ]


def api617_unbalance_magnitude(load, speed_rpm):
    return Q_(2 * 6350 * load / speed_rpm, "g*mm").to("kg*m").m


def test_api617_unbalance_first_mode(rotor_with_clearances):
    rotor = rotor_with_clearances
    unbalance = rotor.api617_unbalance(mode=0, maximum_continuous_speed=Q_(9000, "RPM"))

    assert unbalance["node"] == [3]
    assert unbalance["unbalance_phase"] == [0.0]
    assert_allclose(unbalance["static_load"], [rotor.m], rtol=1e-6)
    assert_allclose(
        unbalance["unbalance_magnitude"],
        [api617_unbalance_magnitude(rotor.m, 9000)],
        rtol=1e-6,
    )
    modal = rotor.run_modal(speed=Q_(9000, "RPM"))
    assert modal.whirl_direction()[unbalance["mode_index"]] == "Forward"
    assert_allclose(unbalance["mode_frequency"], modal.wd[unbalance["mode_index"]])


def test_api617_unbalance_conical_mode(rotor_with_clearances):
    rotor = rotor_with_clearances
    unbalance = rotor.api617_unbalance(mode=1, maximum_continuous_speed=Q_(9000, "RPM"))

    assert len(unbalance["node"]) == 2
    assert unbalance["node"][0] < 3 < unbalance["node"][1]
    assert_allclose(abs(np.diff(unbalance["unbalance_phase"])), [np.pi])
    assert_allclose(unbalance["static_load"], [rotor.m / 2, rotor.m / 2], rtol=1e-6)
    assert_allclose(
        unbalance["unbalance_magnitude"],
        [api617_unbalance_magnitude(rotor.m / 2, 9000)] * 2,
        rtol=1e-6,
    )


def test_api617_unbalance_overhung_mode():
    shaft = [
        rs.ShaftElement(L=0.25, idl=0, odl=0.05, material=rs.steel) for _ in range(6)
    ]
    disk = rs.DiskElement.from_geometry(
        n=6, material=rs.steel, width=0.07, i_d=0.05, o_d=0.28
    )
    bearings = [
        rs.BearingElement(n=0, kxx=1e7, cxx=1e3, radial_clearance=Q_(100, "um")),
        rs.BearingElement(n=2, kxx=1e7, cxx=1e3, radial_clearance=Q_(100, "um")),
    ]
    rotor = rs.Rotor(shaft, [disk], bearings)
    unbalance = rotor.api617_unbalance(mode=0, maximum_continuous_speed=Q_(6000, "RPM"))

    overhung_mass = disk.m + sum(sh.m for sh in shaft[2:])
    assert unbalance["node"] == [6]
    assert_allclose(unbalance["static_load"], [overhung_mass], rtol=1e-6)
    assert_allclose(
        unbalance["unbalance_magnitude"],
        [api617_unbalance_magnitude(overhung_mass, 6000)],
        rtol=1e-6,
    )


def test_api617_unbalance_high_speed_residual(rotor_with_clearances):
    unbalance = rotor_with_clearances.api617_unbalance(
        mode=0, maximum_continuous_speed=Q_(30000, "RPM")
    )
    expected = Q_(2 * rotor_with_clearances.m / 3.937, "g*mm").to("kg*m").m
    assert_allclose(unbalance["unbalance_magnitude"], [expected], rtol=1e-6)


def test_api617_unbalance_mode_not_available(rotor_with_clearances):
    with pytest.raises(ValueError, match="forward modes"):
        rotor_with_clearances.api617_unbalance(
            mode=20, maximum_continuous_speed=Q_(9000, "RPM")
        )


def test_run_clearance_analysis(rotor_with_clearances, clearance_probes):
    rotor = rotor_with_clearances
    speed_range = Q_(np.linspace(0, 10000, 101), "RPM")
    results = rotor.run_clearance_analysis(
        speed_range=speed_range,
        minimum_allowable_speed=Q_(7000, "RPM"),
        maximum_continuous_speed=Q_(9000, "RPM"),
        probes=clearance_probes,
    )

    assert results.clearance_tags == ["DE", "eye", "NDE"]
    assert results.clearance_nodes == [0, 3, 6]
    assert_allclose(results.diametral_clearance, [200e-6, 500e-6, 240e-6])
    assert_allclose(results.clearance_limit, [150e-6, 375e-6, 180e-6])
    assert_allclose(results.clearance_positions, [0.0, 0.75, 1.5])

    assert results.probe_tags == ["DE-45", "NDE-45"]
    assert results.probe_response.shape == (2, len(results.speed_range))
    assert results.clearance_response.shape == (3, len(results.speed_range))

    assert_allclose(results.vibration_limit, 25.4e-6)
    in_range = (results.speed_range >= Q_(7000, "RPM").to("rad/s").m) & (
        results.speed_range <= Q_(9000, "RPM").to("rad/s").m
    )
    assert_allclose(
        results.max_probe_amplitude, results.probe_response[:, in_range].max()
    )
    assert_allclose(
        results.scale_factor, results.vibration_limit / results.max_probe_amplitude
    )

    unbalance = rotor.api617_unbalance(mode=0, maximum_continuous_speed=Q_(9000, "RPM"))
    assert results.unbalance_node == unbalance["node"]
    assert_allclose(results.unbalance_magnitude, unbalance["unbalance_magnitude"])
    assert results.mode == 0
    assert results.mode_index == unbalance["mode_index"]

    response = rotor.run_unbalance_response(
        unbalance["node"],
        unbalance["unbalance_magnitude"],
        unbalance["unbalance_phase"],
        results.speed_range,
    )
    major_axis = response._calculate_major_axis_per_node(node=6, angle="major")[3]
    assert_allclose(
        results.clearance_response[2], 2 * results.scale_factor * major_axis.real
    )
    probe_pkpk = response.data_magnitude(
        probe=clearance_probes, amplitude_units="m pkpk"
    )
    assert_allclose(results.probe_response[0], probe_pkpk["DE-45"].to_numpy())

    df = results.data()
    assert list(df["tag"]) == ["DE", "eye", "NDE"]
    assert_allclose(df["max amplitude pp (um)"], 1e6 * results.max_clearance_response)
    assert_allclose(
        df["% of limit"], 100 * results.max_clearance_response / results.clearance_limit
    )
    assert all(df["status"] == np.where(results.passed, "OK", "EXCEEDED"))


def test_run_clearance_analysis_vibration_limit_high_speed(
    rotor_with_clearances, clearance_probes
):
    results = rotor_with_clearances.run_clearance_analysis(
        speed_range=Q_(np.linspace(1000, 25000, 25), "RPM"),
        minimum_allowable_speed=Q_(15000, "RPM"),
        maximum_continuous_speed=Q_(20000, "RPM"),
        probes=clearance_probes,
    )
    assert_allclose(results.vibration_limit, 25.4e-6 * np.sqrt(12000 / 20000))


def test_run_clearance_analysis_scale_factor_cap(
    rotor_with_clearances, clearance_probes
):
    kwargs = dict(
        speed_range=Q_(np.linspace(0, 10000, 51), "RPM"),
        minimum_allowable_speed=Q_(7000, "RPM"),
        maximum_continuous_speed=Q_(9000, "RPM"),
        probes=clearance_probes,
    )
    uncapped = rotor_with_clearances.run_clearance_analysis(**kwargs)
    capped = rotor_with_clearances.run_clearance_analysis(
        scale_factor_cap=0.5 * uncapped.scale_factor, **kwargs
    )

    assert uncapped.scale_factor_cap is None
    assert_allclose(capped.scale_factor, 0.5 * uncapped.scale_factor)
    assert_allclose(capped.clearance_response, 0.5 * uncapped.clearance_response)
    assert_allclose(capped.probe_response, uncapped.probe_response)


def test_run_clearance_analysis_scaled_response_independent_of_unbalance(
    rotor_with_clearances, clearance_probes
):
    kwargs = dict(
        speed_range=Q_(np.linspace(0, 10000, 51), "RPM"),
        minimum_allowable_speed=Q_(7000, "RPM"),
        maximum_continuous_speed=Q_(9000, "RPM"),
        probes=clearance_probes,
        node=3,
        unbalance_phase=0.0,
    )
    small = rotor_with_clearances.run_clearance_analysis(
        unbalance_magnitude=Q_(20, "g*mm"), **kwargs
    )
    large = rotor_with_clearances.run_clearance_analysis(
        unbalance_magnitude=Q_(80, "g*mm"), **kwargs
    )

    assert small.mode is None
    assert small.unbalance_node == [3]
    assert_allclose(large.max_probe_amplitude, 4 * small.max_probe_amplitude)
    assert_allclose(large.scale_factor, small.scale_factor / 4)
    assert_allclose(large.clearance_response, small.clearance_response)


def test_run_clearance_analysis_adds_operating_speeds(
    rotor_with_clearances, clearance_probes
):
    results = rotor_with_clearances.run_clearance_analysis(
        speed_range=Q_(np.linspace(0, 10000, 11), "RPM"),
        minimum_allowable_speed=Q_(6500, "RPM"),
        maximum_continuous_speed=Q_(8500, "RPM"),
        probes=clearance_probes,
    )
    assert len(results.speed_range) == 13
    assert Q_(6500, "RPM").to("rad/s").m in results.speed_range
    assert Q_(8500, "RPM").to("rad/s").m in results.speed_range


def test_run_clearance_analysis_errors(rotor_with_clearances, clearance_probes):
    kwargs = dict(
        speed_range=Q_(np.linspace(0, 10000, 11), "RPM"),
        minimum_allowable_speed=Q_(7000, "RPM"),
        maximum_continuous_speed=Q_(9000, "RPM"),
    )
    with pytest.raises(ValueError, match="radial probe"):
        rotor_with_clearances.run_clearance_analysis(
            probes=[clearance_probes[2]], **kwargs
        )
    with pytest.raises(ValueError, match="unbalance_magnitude"):
        rotor_with_clearances.run_clearance_analysis(
            probes=clearance_probes, node=3, **kwargs
        )
    with pytest.raises(ValueError, match="minimum_allowable_speed"):
        rotor_with_clearances.run_clearance_analysis(
            speed_range=kwargs["speed_range"],
            minimum_allowable_speed=Q_(9500, "RPM"),
            maximum_continuous_speed=Q_(9000, "RPM"),
            probes=clearance_probes,
        )
    with pytest.raises(ValueError, match="close-clearance"):
        rotor_example().run_clearance_analysis(probes=clearance_probes, **kwargs)


def test_clearance_plots_line_shape(rotor_with_clearances, clearance_probes):
    results = rotor_with_clearances.run_clearance_analysis(
        speed_range=Q_(np.linspace(0, 10000, 21), "RPM"),
        minimum_allowable_speed=Q_(7000, "RPM"),
        maximum_continuous_speed=Q_(9000, "RPM"),
        probes=clearance_probes,
    )

    fig = results.plot_response()
    assert fig.data[0].line.shape == "linear"
    fig = results.plot_response(line_shape="spline")
    assert fig.data[0].line.shape == "spline"
    assert fig.data[1].line.dash == "dash"

    fig = results.plot_probe_response()
    assert fig.data[0].line.shape == "linear"
    fig = results.plot_probe_response(line_shape="spline")
    assert fig.data[0].line.shape == "spline"
    assert fig.data[-1].name == "Avl"

    assert len(results.plot().data) == 3


def test_save_load_clearance(rotor_with_clearances, clearance_probes):
    results = rotor_with_clearances.run_clearance_analysis(
        speed_range=Q_(np.linspace(0, 10000, 21), "RPM"),
        minimum_allowable_speed=Q_(7000, "RPM"),
        maximum_continuous_speed=Q_(9000, "RPM"),
        probes=clearance_probes,
        scale_factor_cap=6,
    )

    for suffix in (".toml", ".json"):
        file = Path(tempdir) / f"clearance{suffix}"
        results.save(file)
        loaded = ClearanceResults.load(file)

        assert_allclose(loaded.speed_range, results.speed_range)
        assert_allclose(loaded.probe_response, results.probe_response)
        assert_allclose(loaded.clearance_response, results.clearance_response)
        assert_allclose(loaded.diametral_clearance, results.diametral_clearance)
        assert loaded.clearance_tags == results.clearance_tags
        assert loaded.probe_tags == results.probe_tags
        assert loaded.unbalance_node == results.unbalance_node
        assert loaded.scale_factor == results.scale_factor
        assert loaded.scale_factor_cap == 6
        assert loaded.mode == 0
        assert loaded.mode_index == results.mode_index
        assert_allclose(loaded.mode_frequency, results.mode_frequency)


def test_save_load_amb_time_response(rotor_amb):
    t = np.linspace(0, 0.01, 100)
    # The rotor already has 2 magnetic bearings
    n_amb = 2
    d_v = np.zeros((len(t), n_amb * 2))
    sim = AmbTimeResponse(rotor_amb, t=t, speed=0, disturbance=d_v)
    sim.run()

    results = AmbTimeResponseResults(
        rotor_amb, sim.t, sim.y, [sim.x_disp, sim.y_disp, [], [], []]
    )

    file = Path(tempdir) / "amb_time.toml"
    results.save(file)
    results2 = AmbTimeResponseResults.load(file)

    assert_allclose(results2.t, results.t, atol=1e-10)
    assert_allclose(results2.x_amb, results.x_amb, atol=1e-10)
    assert_allclose(results2.v_amb, results.v_amb, atol=1e-10)
    assert_allclose(results2.F_x, results.F_x, atol=1e-10)
    assert_allclose(results2.F_v, results.F_v, atol=1e-10)
    assert_allclose(results2.I, results.I, atol=1e-10)

    assert (
        results2.rotor.bearing_elements[0].tag == results.rotor.bearing_elements[0].tag
    )


def test_plot_mode_3d_frame_is_right_handed(rotor1):
    modal = rotor1.run_modal(speed=Q_(4000, "RPM"))
    fig = modal.plot_mode_3d(0)

    # the scene is a rotation of the rotor frame: rotor x on the reversed
    # scene x axis, the length on the scene y axis, rotor y up. No camera is
    # set, so the modebar reset returns to the same default view
    scene = fig.layout.scene
    assert scene.xaxis.range[0] > scene.xaxis.range[1]
    assert scene.yaxis.autorange != "reversed"
    assert scene.yaxis.title.text.startswith("Rotor Length")
    assert scene.camera.eye.x is None

    axes = [trace for trace in fig.data if trace.name == "Axes"]
    assert [trace.mode for trace in axes] == ["lines", "text"]
    assert list(axes[1].text) == ["x", "y", "z", "ω"]
    # the triad sits on the rotor axis at z = 0 and toggles from the legend
    assert [trace.showlegend for trace in axes] == [True, False]
    lines = axes[0]
    assert (lines.x[0], lines.y[0], lines.z[0]) == (0.0, 0.0, 0.0)


def test_campbell_mode_shape_keeps_the_plot_mode_3d_view(rotor1):
    speed_range = np.linspace(0, 400, 5)
    campbell = rotor1.run_campbell(speed_range)
    camp_fig, update_mode_3d = campbell._plot_with_mode_shape()

    # the Dash page used to force its own camera on the mode shape, so the
    # view differed from plot_mode_3d and the modebar reset jumped elsewhere
    reference = campbell.modal_results[speed_range[0]].plot_mode_3d(0)
    for fig in (
        update_mode_3d(),
        update_mode_3d(
            {"x": camp_fig.data[1].x[-1], "y": camp_fig.data[1].y[-1], "curveNumber": 1}
        ),
    ):
        assert fig.layout.scene.camera.eye.x is None
        assert fig.layout.scene.aspectratio == reference.layout.scene.aspectratio
        assert fig.layout.scene.xaxis.range[0] > fig.layout.scene.xaxis.range[1]


def test_plot_mode_2d_has_no_axes_indicator(rotor1):
    modal = rotor1.run_modal(speed=Q_(4000, "RPM"))
    fig = modal.plot_mode_2d(0)

    assert len(fig.layout.shapes) == 0


def test_plot_orbit_axis_names(rotor1):
    modal = rotor1.run_modal(speed=Q_(4000, "RPM"), num_modes=14)
    lateral_mode = next(
        i for i, shape in enumerate(modal.shapes) if shape.mode_type == "Lateral"
    )
    fig = modal.plot_orbit(lateral_mode, nodes=[2])

    assert len(fig.layout.shapes) == 0
    assert fig.layout.xaxis.title.text == "<i>x</i>"
    assert fig.layout.yaxis.title.text == "<i>y</i>"


def test_plot_deflected_shape_3d_frame_is_right_handed(rotor1):
    speed = Q_(4000, "RPM").to("rad/s").m
    response = rotor1.run_unbalance_response(
        node=3, unbalance_magnitude=0.001, unbalance_phase=0, speed_range=[speed]
    )
    fig = response.plot_deflected_shape_3d(speed=speed)

    scene = fig.layout.scene
    assert scene.xaxis.range[0] > scene.xaxis.range[1]
    assert scene.yaxis.title.text.startswith("Rotor Length")
    assert [trace.name for trace in fig.data].count("Axes") == 2
