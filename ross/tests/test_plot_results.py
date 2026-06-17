import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose

from ross import Q_, Probe
from ross.bearing_seal_element import BearingElement
from ross.bearings.bearing_results import (
    PlainJournalResults,
    ThrustPadResults,
    TiltingPadResults,
)
from ross.bearings.fluid_flow import fluid_flow_example
from ross.bearings.fluid_flow_graphics import (
    plot_eccentricity,
    plot_pressure_surface,
    plot_pressure_theta,
    plot_pressure_z,
)
from ross.bearings.squeeze_film_damper import SqueezeFilmDamper
from ross.disk_element import DiskElement
from ross.materials import steel
from ross.rotor_assembly import Rotor, rotor_example
from ross.shaft_element import ShaftElement


def assert_plotly_figure(fig, min_traces=1):
    """Assert that *fig* is a Plotly figure with at least *min_traces* traces."""
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= min_traces


def find_traces_by_name(fig, name):
    """Return trace indices whose ``name`` matches *name*."""
    return [i for i, trace in enumerate(fig.data) if trace.name == name]


def assert_trace_allclose(
    trace,
    x=None,
    y=None,
    z=None,
    r=None,
    theta=None,
    rtol=1e-5,
    atol=0,
):
    """Compare trace arrays to expected values using ``assert_allclose``."""
    if x is not None:
        assert_allclose(np.asarray(trace.x), np.asarray(x), rtol=rtol, atol=atol)
    if y is not None:
        assert_allclose(np.asarray(trace.y), np.asarray(y), rtol=rtol, atol=atol)
    if z is not None:
        assert_allclose(np.asarray(trace.z), np.asarray(z), rtol=rtol, atol=atol)
    if r is not None:
        assert_allclose(np.asarray(trace.r), np.asarray(r), rtol=rtol, atol=atol)
    if theta is not None:
        assert_allclose(
            np.asarray(trace.theta), np.asarray(theta), rtol=rtol, atol=atol
        )


def assert_surface_z_matches(trace, expected_z, rtol=1e-12, atol=0):
    """Verify a Plotly surface trace ``z`` values match the expected field."""
    assert_allclose(np.asarray(trace.z), np.asarray(expected_z), rtol=rtol, atol=atol)


def assert_contour_z_matches(trace, expected_z, rtol=1e-12, atol=0):
    """Verify a Plotly contour trace ``z`` values match the expected field."""
    assert_allclose(np.asarray(trace.z), np.asarray(expected_z), rtol=rtol, atol=atol)


def dataframe_trace_columns(df, x_col="frequency"):
    """Return DataFrame columns that correspond to plot traces."""
    return [col for col in df.columns if col != x_col]


def assert_plot_traces_match_dataframe(fig, df, x_col, y_cols, trace_offset=0):
    """Verify plot trace data matches columns from a results DataFrame."""
    expected_x = df[x_col].to_numpy()
    for i, col in enumerate(y_cols):
        trace = fig.data[trace_offset + i]
        assert_allclose(np.asarray(trace.x), expected_x, rtol=1e-12, atol=0)
        assert_allclose(np.asarray(trace.y), df[col].to_numpy(), rtol=1e-12, atol=0)


def assert_plot_traces_match_polar_dataframe(
    fig, df_mag, df_phase, y_cols, trace_offset=0
):
    """Verify polar plot traces match magnitude and phase DataFrame columns."""
    for i, col in enumerate(y_cols):
        trace = fig.data[trace_offset + i]
        assert_allclose(np.asarray(trace.r), df_mag[col].to_numpy(), rtol=1e-12, atol=0)
        assert_allclose(
            np.asarray(trace.theta), df_phase[col].to_numpy(), rtol=1e-12, atol=0
        )


@pytest.fixture
def rotor():
    return rotor_example()


@pytest.fixture
def freq_response(rotor):
    speed = np.linspace(100, 500, 11)
    return rotor.run_freq_response(speed_range=speed)


@pytest.fixture
def forced_response(rotor):
    speed = np.linspace(100, 500, 11)
    return rotor.run_unbalance_response(3, 0.01, 0.0, speed)


@pytest.fixture
def time_response(rotor):
    speed = 500.0
    size = 50
    node = 3
    t = np.linspace(0, 10, size)
    force = np.zeros((size, rotor.ndof))
    force[:, rotor.number_dof * node] = 10 * np.cos(2 * t)
    force[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)
    return rotor.run_time_response(speed, force, t)


@pytest.fixture
def modal_response(rotor):
    return rotor.run_modal(0)


@pytest.fixture
def static_response(rotor):
    return rotor.run_static()


@pytest.fixture
def probe_node3():
    return Probe(3, Q_(0, "deg"))


@pytest.fixture
def ucs_response(rotor):
    return rotor.run_ucs(num=5)


@pytest.fixture
def level1_response():
    i_d = 0
    o_d = 0.05
    n = 6
    length = [0.25 for _ in range(n)]
    shaft_elem = [
        ShaftElement(
            element_length,
            i_d,
            o_d,
            material=steel,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for element_length in length
    ]
    disk0 = DiskElement.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    bearing0 = BearingElement(0, kxx=1e6, kyy=0.8e6, cxx=0)
    bearing1 = BearingElement(6, kxx=1e6, kyy=0.8e6, cxx=0)
    rotor = Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1], rated_w=0)
    return rotor.run_level1(n=0, stiffness_range=(1e6, 1e11), num=5)


@pytest.fixture
def bearing_element():
    kxx = np.array(
        [8.5e07, 1.1e08, 1.3e08, 1.6e08, 1.8e08, 2.0e08, 2.3e08, 2.5e08, 2.6e08]
    )
    kyy = np.array(
        [9.2e07, 1.1e08, 1.4e08, 1.6e08, 1.9e08, 2.1e08, 2.3e08, 2.5e08, 2.6e08]
    )
    cxx = np.array(
        [226837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    cyy = np.array(
        [235837, 211247, 197996, 185523, 174610, 163697, 153563, 144209, 137973]
    )
    frequency = np.array(
        [314.2, 418.9, 523.6, 628.3, 733.0, 837.8, 942.5, 1047.2, 1151.9]
    )
    return BearingElement(4, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, frequency=frequency)


@pytest.fixture
def tilting_pad_results():
    nz, nx, n_pad = 4, 5, 2
    pressure = np.linspace(0, 1e5, nz * nx * n_pad).reshape(nz, nx, n_pad)
    temperature = np.linspace(40, 60, nz * nx * n_pad).reshape(nz, nx, n_pad)

    return TiltingPadResults(
        frequency=[100.0],
        pressure_fields=[pressure],
        temperature_fields=[temperature],
        maxP_list=[pressure.max()],
        maxT_list=[temperature.max()],
        minH_list=[1e-5],
        ecc_list=[0.3],
        attitude_angle_list=[1.0],
        psi_pad_list=[np.zeros(n_pad)],
        force_x_total_list=[100.0],
        force_y_total_list=[-200.0],
        momen_rot_list=[np.zeros(n_pad)],
        kxx=np.array([1e8]),
        kxy=np.array([0.0]),
        kyx=np.array([0.0]),
        kyy=np.array([1e8]),
        cxx=np.array([1e5]),
        cxy=np.array([0.0]),
        cyx=np.array([0.0]),
        cyy=np.array([1e5]),
        equilibrium_type="match_eccentricity",
        n_pad=n_pad,
        xtheta=np.linspace(-0.5, 0.5, nx),
        xz=np.linspace(-0.5, 0.5, nz),
        pivot_angle=np.array([0.3, 1.8]),
        pad_axial_length=0.05,
        nz=nz,
        nx=nx,
        optimization_history={},
    )


@pytest.fixture
def thrust_pad_results():
    n_radial, n_theta = 3, 5
    d_radius, d_theta = 0.2, 0.2
    pad_arc_length = 0.4538
    pressure = (
        np.arange((n_radial + 2) * (n_theta + 2), dtype=float).reshape(
            n_radial + 2, n_theta + 2
        )
        * 1000
    )
    temperature = pressure / 100

    return ThrustPadResults(
        frequency=[10.0],
        pressure_fields=[pressure],
        temperature_fields=[temperature],
        max_thicknesses=[0.001],
        min_thicknesses=[5e-5],
        pivot_film_thicknesses=[8e-5],
        equilibrium_position_mode="calculate",
        axial_load=1e6,
        kzz=np.array([1e11]),
        czz=np.array([1e9]),
        n_radial=n_radial,
        n_theta=n_theta,
        pad_outer_radius=1.725,
        pad_inner_radius=1.15,
        d_radius=d_radius,
        d_theta=d_theta,
        pad_arc_length=pad_arc_length,
        optimization_history={},
    )


@pytest.fixture
def plain_journal_results():
    elements_axial = 3
    elements_circumferential = 4
    n_pad = 2
    circumferential_total = elements_circumferential * n_pad
    pressure = np.linspace(1e4, 2e5, elements_axial * circumferential_total).reshape(
        elements_axial, circumferential_total
    )
    temperature = np.linspace(40, 70, elements_axial * circumferential_total).reshape(
        elements_axial, circumferential_total
    )
    theta = np.linspace(0, 2 * np.pi, circumferential_total)
    z = np.linspace(0, 0.1, elements_axial)
    theta_grid, z_grid = np.meshgrid(theta, z)
    nondimensional_pressure = np.ones((elements_axial, elements_circumferential, n_pad))

    return PlainJournalResults(
        frequency=[100.0],
        pressure_fields=[pressure],
        temperature_fields=[temperature],
        theta_grids=[theta_grid],
        z_grids=[z_grid],
        P_nondim_fields=[nondimensional_pressure],
        kxx=np.array([1e9]),
        kxy=np.array([0.0]),
        kyx=np.array([0.0]),
        kyy=np.array([1e9]),
        cxx=np.array([1e5]),
        cxy=np.array([0.0]),
        cyx=np.array([0.0]),
        cyy=np.array([1e5]),
        fxs_load=0.0,
        fys_load=-1000.0,
        n_pad=n_pad,
        betha_s_dg=176.0,
        dtheta=0.1,
        thetaF=np.array([np.pi, 2 * np.pi]),
        elements_axial=elements_axial,
        elements_circumferential=elements_circumferential,
        equilibrium_pos_by_speed={100.0: [0.5, 0.1]},
        opt_results={},
        exec_times={},
        optimization_history={},
    )


@pytest.fixture
def fluid_flow():
    bearing = fluid_flow_example()
    bearing.calculate_pressure_matrix_analytical()
    return bearing


def _expected_frf_magnitude(freq_response, inp, out, amplitude_units="m/N"):
    dummy_var = Q_(1, amplitude_units)
    if dummy_var.check("[length]/[force]"):
        magnitude = np.abs(freq_response.freq_resp[inp, out, :])
        magnitude = Q_(magnitude, "m/N").to(amplitude_units).m
    elif dummy_var.check("[speed]/[force]"):
        magnitude = np.abs(freq_response.velc_resp[inp, out, :])
        magnitude = Q_(magnitude, "m/s/N").to(amplitude_units).m
    else:
        magnitude = np.abs(freq_response.accl_resp[inp, out, :])
        magnitude = Q_(magnitude, "m/s**2/N").to(amplitude_units).m
    frequency = Q_(freq_response.speed_range, "rad/s").to("rad/s").m
    return frequency, magnitude


def _expected_frf_phase(freq_response, inp, out, phase_units="rad"):
    phase = np.angle(freq_response.freq_resp[inp, out, :])
    phase = Q_(phase, "rad").to(phase_units).m
    if phase_units in ["rad", "radian", "radians"]:
        phase = np.array([value + 2 * np.pi if value < 0 else value for value in phase])
    else:
        phase = np.array([value + 360 if value < 0 else value for value in phase])
    frequency = Q_(freq_response.speed_range, "rad/s").to("rad/s").m
    return frequency, phase


def _expected_bearing_coefficient_plot(bearing, coefficient, frequency_units="rad/s"):
    frequency_range = np.linspace(min(bearing.frequency), max(bearing.frequency), 30)
    coefficient_values = getattr(bearing, f"{coefficient}_interpolated")(
        frequency_range
    )
    if coefficient.startswith("k"):
        values = Q_(coefficient_values, "N/m").to("N/m").m
    else:
        values = Q_(coefficient_values, "N*s/m").to("N*s/m").m
    frequencies = Q_(frequency_range, "rad/s").to(frequency_units).m
    return frequencies, values


def test_frequency_response_plot_magnitude(freq_response):
    inp, out = 8, 8
    fig = freq_response.plot_magnitude(inp=inp, out=out)
    assert_plotly_figure(fig)

    expected_x, expected_y = _expected_frf_magnitude(freq_response, inp, out)
    assert_trace_allclose(fig.data[0], x=expected_x, y=expected_y)

    expected_y_slice = np.array(
        [1.13318797e-06, 5.77749106e-07, 3.49173341e-07, 2.33468561e-07, 1.66921083e-07]
    )
    assert_allclose(fig.data[0].y[:5], expected_y_slice)


def test_frequency_response_plot_phase(freq_response):
    inp, out = 8, 8
    fig = freq_response.plot_phase(inp=inp, out=out)
    assert_plotly_figure(fig)

    expected_x, expected_y = _expected_frf_phase(freq_response, inp, out)
    assert_trace_allclose(fig.data[0], x=expected_x, y=expected_y)


def test_frequency_response_plot_polar_bode(freq_response):
    inp, out = 8, 8
    fig = freq_response.plot_polar_bode(inp=inp, out=out)
    assert_plotly_figure(fig)

    expected_x, expected_r = _expected_frf_magnitude(freq_response, inp, out)
    _, expected_theta = _expected_frf_phase(freq_response, inp, out)
    expected_theta = np.array(
        [value + 2 * np.pi if value < 0 else value for value in expected_theta]
    )

    assert_trace_allclose(fig.data[0], r=expected_r, theta=expected_theta)
    assert_allclose(fig.data[0].r[:3], expected_r[:3])


def test_forced_response_plot_magnitude(forced_response, probe_node3):
    probe = [probe_node3]
    fig = forced_response.plot_magnitude(probe=probe)
    assert_plotly_figure(fig)

    df = forced_response.data_magnitude(probe=probe)
    assert_plot_traces_match_dataframe(
        fig, df, "frequency", dataframe_trace_columns(df)
    )

    expected_y = np.array([0.00202625, 0.00026848, 0.00019152, 0.00016191, 0.00014354])
    assert_allclose(fig.data[0].y[:5], expected_y, rtol=1e-4)


def test_forced_response_plot_phase(forced_response, probe_node3):
    probe = [probe_node3]
    fig = forced_response.plot_phase(probe=probe)
    assert_plotly_figure(fig)

    df = forced_response.data_phase(probe=probe)
    assert_plot_traces_match_dataframe(
        fig, df, "frequency", dataframe_trace_columns(df)
    )


def test_forced_response_plot_bode(forced_response, probe_node3):
    probe = [probe_node3]
    fig = forced_response.plot_bode(probe=probe)
    assert_plotly_figure(fig, min_traces=2)

    df_mag = forced_response.data_magnitude(probe=probe)
    df_phase = forced_response.data_phase(probe=probe)
    y_cols = dataframe_trace_columns(df_mag)
    assert_plot_traces_match_dataframe(fig, df_mag, "frequency", y_cols, trace_offset=0)
    assert_plot_traces_match_dataframe(
        fig, df_phase, "frequency", y_cols, trace_offset=1
    )


def test_forced_response_plot_polar_bode(forced_response, probe_node3):
    probe = [probe_node3]
    fig = forced_response.plot_polar_bode(probe=probe)
    assert_plotly_figure(fig)

    df_mag = forced_response.data_magnitude(probe=probe)
    df_phase = forced_response.data_phase(probe=probe)
    assert_plot_traces_match_polar_dataframe(
        fig, df_mag, df_phase, dataframe_trace_columns(df_mag)
    )


def test_forced_response_plot_deflected_shape_2d(forced_response):
    fig = forced_response.plot_deflected_shape_2d(speed=300)
    assert_plotly_figure(fig)

    expected_y = np.array([0.00013749, 0.00014132, 0.00014479, 0.00014756, 0.00014927])
    shape_trace = next(
        trace for trace in fig.data if trace.y is not None and len(trace.y) > 5
    )
    assert_allclose(np.asarray(shape_trace.y)[:5], expected_y, rtol=1e-4)


def test_time_response_plot_1d(time_response, probe_node3):
    probe = [probe_node3]
    fig = time_response.plot_1d(probe=probe)
    assert_plotly_figure(fig)

    df = time_response.data_time_response(probe=probe)
    expected_x = df["time"].values
    expected_y = df["probe_resp[0]"].values
    assert_trace_allclose(fig.data[0], x=expected_x, y=expected_y)

    expected_y_slice = np.array(
        [0.0, 4.07504755e-06, 1.19778973e-05, 1.68562228e-05, 1.34097882e-05]
    )
    assert_allclose(fig.data[0].y[:5], expected_y_slice)


def test_time_response_plot_2d(time_response):
    node = 3
    fig = time_response.plot_2d(node=node)
    assert_plotly_figure(fig)

    ndof = time_response.rotor.number_dof
    expected_x = time_response.yout[:, ndof * node]
    expected_y = time_response.yout[:, ndof * node + 1]
    assert_trace_allclose(fig.data[0], x=expected_x, y=expected_y)


def test_time_response_plot_dfft(time_response, probe_node3):
    probe = [probe_node3]
    fig = time_response.plot_dfft(probe=probe)
    assert_plotly_figure(fig)

    expected_y = np.array(
        [3.67360008e-06, 1.78864925e-05, 1.43350914e-05, 4.07024586e-06, 1.39274833e-06]
    )
    assert_allclose(fig.data[0].y[:5], expected_y, rtol=1e-4)


def test_modal_plot_mode_2d(modal_response):
    fig = modal_response.plot_mode_2d(0)
    assert_plotly_figure(fig)

    expected_y = np.array([0.41220064, 0.48871721, 0.56410565, 0.63704623, 0.70621921])
    assert_allclose(fig.data[0].y[:5], expected_y, rtol=1e-4)
    assert np.max(np.abs(fig.data[0].y)) == pytest.approx(1.0, rel=1e-6)


def test_static_plot_deformation(static_response):
    fig = static_response.plot_deformation()
    assert_plotly_figure(fig, min_traces=2)
    assert_allclose(fig.data[1].y, static_response.deformation)


def test_static_plot_shearing_force(static_response):
    fig = static_response.plot_shearing_force()
    assert_plotly_figure(fig, min_traces=2)
    assert_allclose(fig.data[1].y, static_response.Vx)


def test_static_plot_bending_moment(static_response):
    fig = static_response.plot_bending_moment()
    assert_plotly_figure(fig, min_traces=2)
    assert_allclose(fig.data[1].y, static_response.Bm)


def test_ucs_plot(ucs_response):
    fig = ucs_response.plot()
    assert_plotly_figure(fig, min_traces=4)

    stiffness = Q_(ucs_response.stiffness_log, "N/m").to("N/m").m
    natural_frequencies = Q_(ucs_response.wn, "rad/s").to("rad/s").m

    for mode_idx in range(natural_frequencies.shape[0]):
        assert_allclose(fig.data[mode_idx].x, stiffness)
        assert_allclose(fig.data[mode_idx].y, natural_frequencies[mode_idx])


def test_level1_plot(level1_response):
    fig = level1_response.plot()
    assert_plotly_figure(fig)

    assert_trace_allclose(
        fig.data[0],
        x=level1_response.stiffness_range,
        y=level1_response.log_dec,
    )


def test_bearing_element_plot_stiffness(bearing_element):
    fig = bearing_element.plot(coefficients="kxx")
    assert_plotly_figure(fig)

    expected_x, expected_y = _expected_bearing_coefficient_plot(bearing_element, "kxx")
    assert_trace_allclose(fig.data[0], x=expected_x, y=expected_y)

    expected_y_slice = np.array(
        [8.50000000e07, 9.39094443e07, 1.00985975e08, 1.06782950e08, 1.11853726e08]
    )
    assert_allclose(fig.data[0].y[:5], expected_y_slice)


def test_bearing_element_plot_damping(bearing_element):
    fig = bearing_element.plot(coefficients="cxx")
    assert_plotly_figure(fig)

    expected_x, expected_y = _expected_bearing_coefficient_plot(bearing_element, "cxx")
    assert_trace_allclose(fig.data[0], x=expected_x, y=expected_y)


def test_tilting_pad_plot_pressure_3d(tilting_pad_results):
    fig = tilting_pad_results.plot_pressure_3d()
    assert_plotly_figure(fig)

    pressure = tilting_pad_results.pressure_fields[0]
    pad_index = int(np.argmax(pressure.max(axis=(0, 1))))
    expected_z = 1e-6 * pressure[:, :, pad_index]
    assert_surface_z_matches(fig.data[0], expected_z)


def test_tilting_pad_plot_pressure_2d(tilting_pad_results):
    fig = tilting_pad_results.plot_pressure_2d()
    assert_plotly_figure(fig, min_traces=tilting_pad_results.n_pad)

    pressure = tilting_pad_results.pressure_fields[0]
    for pad_index in range(tilting_pad_results.n_pad):
        assert_contour_z_matches(fig.data[pad_index], pressure[:, :, pad_index])


def test_tilting_pad_plot_temperature_3d(tilting_pad_results):
    fig = tilting_pad_results.plot_temperature_3d()
    assert_plotly_figure(fig)

    temperature = tilting_pad_results.temperature_fields[0]
    pad_index = int(np.argmax(temperature.max(axis=(0, 1))))
    assert_surface_z_matches(fig.data[0], temperature[:, :, pad_index])


def test_tilting_pad_plot_results(tilting_pad_results):
    figures = tilting_pad_results.plot_results()
    assert {
        "pressure_2d",
        "pressure_3d",
        "temperature_2d",
        "temperature_3d",
        "pressure_scatter",
        "temperature_scatter",
    }.issubset(figures.keys())
    for fig in figures.values():
        assert_plotly_figure(fig)


def test_thrust_pad_plot_pressure_3d(thrust_pad_results):
    fig = thrust_pad_results.plot_pressure_3d()
    assert_plotly_figure(fig)
    assert_surface_z_matches(fig.data[0], thrust_pad_results.pressure_fields[0])


def test_thrust_pad_plot_pressure_2d(thrust_pad_results):
    fig = thrust_pad_results.plot_pressure_2d()
    assert_plotly_figure(fig)
    assert np.nanmax(fig.data[0].z) == pytest.approx(
        thrust_pad_results.pressure_fields[0].max(), rel=0.02
    )


def test_thrust_pad_plot_temperature_3d(thrust_pad_results):
    fig = thrust_pad_results.plot_temperature_3d()
    assert_plotly_figure(fig)
    assert_surface_z_matches(fig.data[0], thrust_pad_results.temperature_fields[0])


def test_plain_journal_plot_pressure_2d(plain_journal_results):
    fig = plain_journal_results.plot_pressure_2d()
    assert_plotly_figure(fig)
    assert_contour_z_matches(fig.data[0], plain_journal_results.pressure_fields[0])


def test_plain_journal_plot_pressure_3d(plain_journal_results):
    fig = plain_journal_results.plot_pressure_3d()
    assert_plotly_figure(fig)
    assert_surface_z_matches(fig.data[0], plain_journal_results.pressure_fields[0])


def test_plain_journal_plot_bearing_representation(plain_journal_results):
    fig = plain_journal_results.plot_bearing_representation()
    assert_plotly_figure(fig)
    groove = (360 / plain_journal_results.n_pad) - plain_journal_results.betha_s_dg
    expected_values = [groove / 2, plain_journal_results.betha_s_dg, groove / 2] * (
        plain_journal_results.n_pad
    )
    assert_allclose(fig.data[0].values, expected_values)


def test_squeeze_film_damper_field_plots_not_available():
    bearing = SqueezeFilmDamper(
        n=0,
        frequency=Q_([18600], "rpm"),
        axial_length=Q_(0.9, "inches"),
        journal_radius=Q_(2.55, "inches"),
        radial_clearance=Q_(0.003, "inches"),
        eccentricity_ratio=0.5,
        lubricant="ISOVG32",
        geometry="groove-end_seals",
        cavitation=True,
    )

    with pytest.raises(NotImplementedError):
        bearing.plot_pressure_2d()

    figures = bearing.plot_results()
    assert figures == {}


def test_fluid_flow_plot_pressure_theta(fluid_flow):
    z = int(fluid_flow.nz / 2)
    fig = plot_pressure_theta(fluid_flow, z=z)
    assert_plotly_figure(fig)
    assert_trace_allclose(
        fig.data[0],
        x=fluid_flow.gama[z],
        y=fluid_flow.p_mat_analytical[z],
    )


def test_fluid_flow_plot_pressure_z(fluid_flow):
    theta = int(fluid_flow.ntheta / 2)
    fig = plot_pressure_z(fluid_flow, theta=theta)
    assert_plotly_figure(fig)
    assert_trace_allclose(
        fig.data[0],
        x=fluid_flow.z_list,
        y=fluid_flow.p_mat_analytical[:, theta],
    )


def test_fluid_flow_plot_pressure_surface():
    bearing = fluid_flow_example()
    bearing.calculate_pressure_matrix_numerical()
    fig = plot_pressure_surface(bearing)
    assert_plotly_figure(fig)
    assert_surface_z_matches(fig.data[0], bearing.p_mat_numerical.T)


def test_fluid_flow_plot_eccentricity(fluid_flow):
    z = int(fluid_flow.nz / 2)
    fig = plot_eccentricity(fluid_flow, z=z)
    assert_plotly_figure(fig, min_traces=2)
    assert len(fig.data[0].x) == fluid_flow.ntheta
    assert len(fig.data[1].x) == fluid_flow.ntheta
