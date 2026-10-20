import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose

from ross import Q_, Probe
from ross.bearing_seal_element import BearingElement
from ross.bearings.bearing_results import ThrustPadResults
from ross.bearings.squeeze_film_damper import SqueezeFilmDamper
from ross.disk_element import DiskElement
from ross.materials import steel
from ross.results import TimeResponseResults
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
    return BearingElement(4, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy, speed=frequency)


@pytest.fixture
def thrust_pad_results():
    n_radial, n_theta = 3, 5
    d_radius, d_theta = 0.2, 0.2
    pad_arc = 0.4538
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
        pad_arc=pad_arc,
        optimization_history={},
    )


@pytest.fixture
def cross_coupled_rotor():
    shaft = [ShaftElement(L=0.25, idl=0, odl=0.05, material=steel) for _ in range(6)]
    disks = [
        DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28),
        DiskElement.from_geometry(n=4, material=steel, width=0.07, i_d=0.05, o_d=0.35),
    ]
    bearings = [
        BearingElement(n=0, kxx=1e6, kyy=1.5e6, kxy=4e5, kyx=-6e5, cxx=200, cyy=300),
        BearingElement(n=6, kxx=1.2e6, kyy=0.8e6, kxy=3e5, kyx=-3e5, cxx=150, cyy=100),
    ]
    return Rotor(shaft, disks, bearings)


def _expected_frf_magnitude(freq_response, inp, out, amplitude_units="m/N"):
    dummy_var = Q_(1, amplitude_units)
    if dummy_var.check("[length]/[force]"):
        magnitude = np.abs(freq_response.freq_resp[out, inp, :])
        magnitude = Q_(magnitude, "m/N").to(amplitude_units).m
    elif dummy_var.check("[speed]/[force]"):
        magnitude = np.abs(freq_response.velc_resp[out, inp, :])
        magnitude = Q_(magnitude, "m/s/N").to(amplitude_units).m
    else:
        magnitude = np.abs(freq_response.accl_resp[out, inp, :])
        magnitude = Q_(magnitude, "m/s**2/N").to(amplitude_units).m
    frequency = Q_(freq_response.speed_range, "rad/s").to("rad/s").m
    return frequency, magnitude


def _expected_frf_phase(freq_response, inp, out, phase_units="rad"):
    phase = np.angle(freq_response.freq_resp[out, inp, :])
    phase = Q_(phase, "rad").to(phase_units).m
    if phase_units in ["rad", "radian", "radians"]:
        phase = np.array([value + 2 * np.pi if value < 0 else value for value in phase])
    else:
        phase = np.array([value + 360 if value < 0 else value for value in phase])
    frequency = Q_(freq_response.speed_range, "rad/s").to("rad/s").m
    return frequency, phase


def _expected_bearing_coefficient_plot(bearing, coefficient, frequency_units="rad/s"):
    frequency_range = np.linspace(min(bearing.speed), max(bearing.speed), 30)
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


@pytest.mark.parametrize("speed", [None, 0.0, 300.0, 900.0])
def test_frequency_response_plots_read_response_at_out_to_force_at_inp(
    cross_coupled_rotor, speed
):
    """Each FRF plot shows the response at `out` to a unit force at `inp`.

    The reference solves the dynamic stiffness Z(w) x = e_inp directly at each
    frequency, without the transfer matrix. The bearings are anisotropic and
    cross-coupled, with kxy != -kyx at node 0, and the disks are gyroscopic,
    so H is not symmetric and a plot that read H[inp, out] instead of
    H[out, inp] would differ in magnitude and phase.
    """
    rotor = cross_coupled_rotor
    frequencies = np.array([60.0, 250.0, 520.0, 900.0])
    if speed is None:
        frf = rotor.run_freq_response(speed_range=frequencies)
        speeds = frequencies
    else:
        frf = rotor.run_freq_response(speed_range=frequencies, speed=speed)
        speeds = np.full_like(frequencies, speed)

    d = rotor.number_dof
    pairs = [(4 * d, 2 * d + 1), (2 * d + 1, 4 * d), (4 * d, 2 * d), (4 * d + 1, d + 1)]
    for inp, out in pairs:
        expected = []
        for w, s in zip(frequencies, speeds, strict=True):
            Z = (
                -(w**2) * rotor.M(w, s)
                + 1j * w * (rotor.C(w, s) + s * rotor.G())
                + rotor.K(w, s)
            )
            force = np.zeros(rotor.ndof, dtype=complex)
            force[inp] = 1.0
            expected.append(np.linalg.solve(Z, force)[out])
        expected = np.array(expected)
        transposed = frf.freq_resp[inp, out, :]
        assert not np.allclose(transposed, expected, rtol=1e-3, atol=0)

        magnitude = frf.plot_magnitude(inp=inp, out=out).data[0]
        assert_allclose(magnitude.y, np.abs(expected), rtol=1e-8)

        phase = frf.plot_phase(inp=inp, out=out).data[0]
        assert_allclose(np.exp(1j * np.array(phase.y)), expected / np.abs(expected))

        polar = frf.plot_polar_bode(inp=inp, out=out).data[0]
        assert_allclose(polar.r, np.abs(expected), rtol=1e-8)
        assert_allclose(np.exp(1j * np.array(polar.theta)), expected / np.abs(expected))

        combined = frf.plot(inp=inp, out=out)
        assert_allclose(combined.data[0].y, np.abs(expected), rtol=1e-8)
        assert_allclose(
            np.exp(1j * np.array(combined.data[1].y)), expected / np.abs(expected)
        )
        assert_allclose(combined.data[2].r, np.abs(expected), rtol=1e-8)

        velocity = 1j * frequencies * expected
        acceleration = -(frequencies**2) * expected
        for units, reference in (("m/s/N", velocity), ("m/s**2/N", acceleration)):
            magnitude = frf.plot_magnitude(inp, out, amplitude_units=units).data[0]
            assert_allclose(magnitude.y, np.abs(reference), rtol=1e-8)
            phase = frf.plot_phase(inp, out, amplitude_units=units).data[0]
            assert_allclose(
                np.exp(1j * np.array(phase.y)), reference / np.abs(reference)
            )
            polar = frf.plot_polar_bode(inp, out, amplitude_units=units).data[0]
            assert_allclose(polar.r, np.abs(reference), rtol=1e-8)
            assert_allclose(
                np.exp(1j * np.array(polar.theta)), reference / np.abs(reference)
            )


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


@pytest.mark.parametrize("displacement_units", ["mm", "um"])
def test_time_response_plot_1d_displacement_units(
    time_response, probe_node3, displacement_units
):
    probe = [probe_node3]
    fig_m = time_response.plot_1d(probe=probe)
    fig = time_response.plot_1d(probe=probe, displacement_units=displacement_units)

    df = time_response.data_time_response(
        probe=probe, displacement_units=displacement_units
    )
    assert_allclose(fig.data[0].y, df["probe_resp[0]"].values)
    assert_allclose(
        fig.data[0].y,
        Q_(np.asarray(fig_m.data[0].y), "m").to(displacement_units).m,
    )


def test_time_response_results_store_run_speed(rotor):
    speed = 2 * np.pi * 4
    time = np.arange(0.0, 1.0, 1e-3)
    force = np.zeros((len(time), rotor.ndof))

    result = rotor.run_time_response(speed, force, time)

    assert result.speed == pytest.approx(speed)
    t_initial, t_final = result._get_revolution(result.t)
    assert t_final - t_initial == pytest.approx(1 / 4)


def test_time_response_plot_2d(time_response):
    node = 3
    fig = time_response.plot_2d(node=node)
    assert_plotly_figure(fig)

    ndof = time_response.rotor.number_dof
    expected_x = time_response.yout[:, ndof * node]
    expected_y = time_response.yout[:, ndof * node + 1]
    assert_trace_allclose(fig.data[0], x=expected_x, y=expected_y)
    assert len(fig.data) == 1

    fig_3d = time_response.plot_3d()
    assert len(fig_3d.data) == len(time_response.rotor.nodes) + 1


def test_time_response_plots_use_time_window(time_response, probe_node3):
    node = 3
    initial_index = 10
    final_index = 20
    t_initial = Q_(time_response.t[initial_index], "s")
    t_final = Q_(time_response.t[final_index], "s")
    original_t = time_response.t.copy()
    original_yout = time_response.yout.copy()

    fig_1d = time_response.plot_1d(
        probe=[probe_node3],
        t_initial=t_initial,
        t_final=t_final,
    )
    fig_2d = time_response.plot_2d(
        node=node,
        t_initial=t_initial,
        t_final=t_final,
    )
    fig_3d = time_response.plot_3d(
        t_initial=t_initial,
        t_final=t_final,
    )

    expected_t = time_response.t[initial_index : final_index + 1]
    expected_yout = time_response.yout[initial_index : final_index + 1]
    expected_df = time_response.data_time_response(
        probe=[probe_node3],
        t=expected_t,
        yout=expected_yout,
    )
    ndof = time_response.rotor.number_dof

    assert_trace_allclose(
        fig_1d.data[0],
        x=expected_df["time"].values,
        y=expected_df["probe_resp[0]"].values,
    )
    assert len(fig_1d.data) == 1
    assert_allclose(
        np.asarray(fig_1d.layout.xaxis.range),
        [expected_df["time"].values[0], expected_df["time"].values[-1]],
    )

    assert_trace_allclose(
        fig_2d.data[0],
        x=expected_yout[:, ndof * node],
        y=expected_yout[:, ndof * node + 1],
    )
    assert fig_2d.data[1].marker.symbol == "circle"
    assert fig_2d.data[2].marker.symbol == "x"

    first_node = time_response.rotor.nodes[0]
    expected_x = np.full(len(expected_t), time_response.rotor.nodes_pos[first_node])
    assert_trace_allclose(
        fig_3d.data[0],
        x=expected_x,
        y=expected_yout[:, ndof * first_node],
        z=expected_yout[:, ndof * first_node + 1],
    )
    assert fig_3d.data[1].marker.symbol == "circle"
    assert fig_3d.data[2].marker.symbol == "x"
    assert fig_3d.data[1].marker.size == 2
    assert fig_3d.data[2].marker.size == 1

    assert_allclose(time_response.t, original_t)
    assert_allclose(time_response.yout, original_yout)


@pytest.mark.parametrize(
    "t_initial, t_final, time_units",
    [
        (Q_(500, "ms"), Q_(1500, "ms"), "s"),
        (500, 1500, "ms"),
        (0.5, 1.5, "s"),
    ],
)
def test_time_response_window_converts_time_units(
    rotor, t_initial, t_final, time_units
):
    time = np.arange(0.0, 3.0, 1e-3)
    response = np.zeros((len(time), rotor.ndof))
    result = TimeResponseResults(rotor, time, response, [])

    window_time, _, windowed = result._get_plot_data(
        {"t_initial": t_initial, "t_final": t_final}, time_units
    )
    fig = result.plot_1d(
        probe=[Probe(3, Q_(0, "rad"))],
        t_initial=t_initial,
        t_final=t_final,
        time_units=time_units,
    )

    assert windowed
    assert window_time[0] == pytest.approx(0.5)
    assert window_time[-1] == pytest.approx(1.5)
    assert_allclose(
        np.asarray(fig.layout.xaxis.range),
        Q_([0.5, 1.5], "s").to(time_units).m,
    )


def test_time_response_plots_use_one_revolution(rotor):
    time = np.arange(0.0, 6.0, 1e-3)
    node = 3
    revolutions_per_second = 2.0
    response = np.zeros((len(time), rotor.ndof))
    dof_x = node * rotor.number_dof
    response[:, dof_x] = np.sin(2 * np.pi * revolutions_per_second * time)
    response[:, dof_x + 1] = np.cos(2 * np.pi * revolutions_per_second * time)

    result = TimeResponseResults(
        rotor, time, response, [], speed=2 * np.pi * revolutions_per_second
    )
    window_time, window_response = result._get_window(one_revolution=True)
    initial_index = np.searchsorted(time, window_time[0])

    assert_allclose(
        [window_time[0], window_time[-1]], [time[-1] - 0.5, time[-1]], atol=1e-3
    )
    assert_allclose(window_response, response[initial_index:])

    probe = Probe(node, Q_(0, "rad"))
    fig_1d = result.plot_1d(probe=[probe], one_revolution=True)
    fig_2d = result.plot_2d(node=node, one_revolution=True)
    fig_3d = result.plot_3d(one_revolution=True)

    assert len(fig_1d.data[0].x) == len(window_time)
    assert_allclose(
        np.asarray(fig_1d.layout.xaxis.range),
        [window_time[0], window_time[-1]],
    )
    assert len(fig_2d.data[0].x) == len(window_time)
    assert len(fig_3d.data[0].x) == len(window_time)

    initial_window, _ = result._get_window(t_initial=1.0, one_revolution=True)
    final_window, _ = result._get_window(t_final=3.0, one_revolution=True)
    assert initial_window[0] == pytest.approx(1.0, abs=1e-3)
    assert initial_window[-1] == pytest.approx(1.5, abs=1e-3)
    assert final_window[0] == pytest.approx(2.5, abs=1e-3)
    assert final_window[-1] == pytest.approx(3.0, abs=1e-3)

    fig_ms = result.plot_1d(
        probe=[probe], one_revolution=True, t_final=3000, time_units="ms"
    )
    assert_allclose(np.asarray(fig_ms.layout.xaxis.range), [2500, 3000])

    with pytest.raises(ValueError, match="one_revolution=True cannot be used"):
        result._get_window(t_initial=1.0, t_final=2.0, one_revolution=True)


def test_one_revolution_integrates_a_varying_speed(rotor):
    time = np.linspace(0.0, 2.0, 2001)
    response = np.zeros((len(time), rotor.ndof))
    acceleration = 4 * np.pi
    speed = acceleration * time
    result = TimeResponseResults(rotor, time, response, [], speed=speed)

    t_initial, t_final = result._get_revolution(time)

    # The shaft angle is acceleration * t**2 / 2.
    assert t_final == pytest.approx(2.0)
    assert t_initial == pytest.approx(np.sqrt(2.0**2 - 4 * np.pi / acceleration))


def test_one_revolution_needs_a_speed(rotor):
    time = np.arange(0.0, 1.0, 1e-3)
    response = np.zeros((len(time), rotor.ndof))

    with pytest.raises(ValueError, match="needs the rotor speed"):
        TimeResponseResults(rotor, time, response, [])._get_window(one_revolution=True)
    with pytest.raises(ValueError, match="does not complete one revolution"):
        TimeResponseResults(rotor, time, response, [], speed=1.0)._get_window(
            one_revolution=True
        )


def test_time_response_plot_dfft(time_response, probe_node3):
    probe = [probe_node3]
    fig = time_response.plot_dfft(probe=probe)
    assert_plotly_figure(fig)

    expected_y = np.array(
        [3.67360008e-06, 1.78864925e-05, 1.43350914e-05, 4.07024586e-06, 1.39274833e-06]
    )
    assert_allclose(fig.data[0].y[:5], expected_y, rtol=1e-4)


@pytest.mark.parametrize("displacement_units", ["mm", "um"])
def test_time_response_plot_dfft_displacement_units(
    time_response, probe_node3, displacement_units
):
    probe = [probe_node3]
    fig_m = time_response.plot_dfft(probe=probe)
    fig = time_response.plot_dfft(probe=probe, displacement_units=displacement_units)

    assert_allclose(
        fig.data[0].y,
        Q_(np.asarray(fig_m.data[0].y), "m").to(displacement_units).m,
    )


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
        [8.50000000e07, 9.24121279e07, 9.94509523e07, 1.06081472e08, 1.12218094e08]
    )
    assert_allclose(fig.data[0].y[:5], expected_y_slice)


def test_bearing_element_plot_damping(bearing_element):
    fig = bearing_element.plot(coefficients="cxx")
    assert_plotly_figure(fig)

    expected_x, expected_y = _expected_bearing_coefficient_plot(bearing_element, "cxx")
    assert_trace_allclose(fig.data[0], x=expected_x, y=expected_y)


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


def test_thrust_pad_plot_film_temperature_3d(thrust_pad_results):
    fig = thrust_pad_results.plot_film_temperature_3d()
    assert_plotly_figure(fig)
    assert_surface_z_matches(fig.data[0], thrust_pad_results.temperature_fields[0])


def test_plot_pad_temperature_3d_not_available(thrust_pad_results):
    with pytest.raises(NotImplementedError, match="solid pad temperature"):
        thrust_pad_results.plot_pad_temperature_3d()


def test_squeeze_film_damper_field_plots_not_available():
    bearing = SqueezeFilmDamper(
        n=0,
        frequency=Q_([18600], "rpm"),
        axial_length=Q_(0.9, "inches"),
        journal_diameter=Q_(5.1, "inches"),
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


@pytest.fixture
def clearance_results():
    rotor = rotor_example()
    bearings = [
        BearingElement(n=0, kxx=1e6, cxx=1e3, radial_clearance=Q_(100, "um"), tag="DE"),
        BearingElement(
            n=6, kxx=1e6, cxx=1e3, radial_clearance=Q_(120, "um"), tag="NDE"
        ),
    ]
    rotor = Rotor(rotor.shaft_elements, rotor.disk_elements, bearings)
    probes = [
        Probe(0, Q_(45, "deg"), tag="DE-45"),
        Probe(6, Q_(45, "deg"), tag="NDE-45"),
    ]
    return rotor.run_clearance_analysis(
        speed_range=Q_(np.linspace(0, 10000, 51), "RPM"),
        minimum_allowable_speed=Q_(7000, "RPM"),
        maximum_continuous_speed=Q_(9000, "RPM"),
        probes=probes,
    )


def test_clearance_plot(clearance_results):
    fig = clearance_results.plot(length_units="mm")
    assert_plotly_figure(fig, min_traces=3)
    assert_trace_allclose(
        fig.data[find_traces_by_name(fig, "Min. diametral clearance")[0]],
        y=1e3 * clearance_results.diametral_clearance,
    )
    assert_trace_allclose(
        fig.data[find_traces_by_name(fig, "75% of clearance")[0]],
        y=1e3 * clearance_results.clearance_limit,
    )
    assert_trace_allclose(
        fig.data[find_traces_by_name(fig, "Scaled amplitude (pk-pk)")[0]],
        y=1e3 * clearance_results.max_clearance_response,
    )


def test_clearance_plot_response(clearance_results):
    fig = clearance_results.plot_response(speed_units="rad/s")
    assert_plotly_figure(fig, min_traces=4)
    assert_trace_allclose(
        fig.data[find_traces_by_name(fig, "DE")[0]],
        x=clearance_results.speed_range,
        y=1e6 * clearance_results.clearance_response[0],
    )
    assert_trace_allclose(
        fig.data[find_traces_by_name(fig, "NDE limit")[0]],
        y=[1e6 * clearance_results.clearance_limit[1]] * 2,
    )


def test_clearance_plot_probe_response(clearance_results):
    fig = clearance_results.plot_probe_response(speed_units="rad/s")
    assert_plotly_figure(fig, min_traces=3)
    assert_trace_allclose(
        fig.data[find_traces_by_name(fig, "NDE-45")[0]],
        x=clearance_results.speed_range,
        y=1e6 * clearance_results.probe_response[1],
    )
    assert_trace_allclose(fig.data[find_traces_by_name(fig, "Avl")[0]], y=[25.4, 25.4])


@pytest.mark.parametrize(
    "speed, dt, steps_per_revolution",
    [
        (4 * np.pi, 1e-3, 500),
        (100.0, 1e-3, 2 * np.pi / 100.0 / 1e-3),
        (600.0, 2 * np.pi / 600.0 / 50, 50),
        (500.0, 2 * np.pi / 500.0 / 9, 9),
    ],
)
def test_one_revolution_with_enough_samples(rotor, speed, dt, steps_per_revolution):
    time = np.arange(0.0, 200 * dt + 2 * np.pi / speed, dt)
    response = np.zeros((len(time), rotor.ndof))
    result = TimeResponseResults(rotor, time, response, [], speed=speed)

    # A revolution of 2*pi/speed seconds sampled every dt seconds holds
    # floor(2*pi/speed/dt) + 1 samples.
    expected = np.floor(steps_per_revolution + 1e-9) + 1
    window_time, _ = result._get_window(one_revolution=True)
    anchored_time, _ = result._get_window(t_initial=time[3], one_revolution=True)

    assert len(window_time) == expected
    assert len(anchored_time) == expected
    fig = result.plot_2d(node=3, one_revolution=True)
    assert len(fig.data[0].x) == expected


@pytest.mark.parametrize(
    "speed, dt, steps_per_revolution",
    [
        (500.0, 10 / 999, 2 * np.pi / 500.0 / (10 / 999)),
        (600.0, 16 / 1000, 2 * np.pi / 600.0 / (16 / 1000)),
        (20 * np.pi, 0.02, 5),
        (500.0, 2 * np.pi / 500.0 / 8, 8),
    ],
)
def test_one_revolution_with_too_few_samples(rotor, speed, dt, steps_per_revolution):
    time = np.arange(0.0, 200 * dt + 2 * np.pi / speed, dt)
    response = np.zeros((len(time), rotor.ndof))
    result = TimeResponseResults(rotor, time, response, [], speed=speed)
    samples = int(np.floor(steps_per_revolution + 1e-9)) + 1
    noun = "sample" if samples == 1 else "samples"
    message = (
        f"lasts {2 * np.pi / speed:.4g} s and the time step is {dt:.4g} s, so it "
        f"holds {samples} {noun}; at least 10 are needed"
    )

    with pytest.raises(ValueError, match=message):
        result._get_window(one_revolution=True)
    with pytest.raises(ValueError, match=message):
        result.plot_1d(probe=[Probe(3, Q_(0, "rad"))], one_revolution=True)
    with pytest.raises(ValueError, match=message):
        result.plot_2d(node=3, one_revolution=True)
    with pytest.raises(ValueError, match=message):
        result.plot_3d(one_revolution=True, t_final=time[-5])


@pytest.mark.parametrize(
    "dt, t_initial, t_final, samples",
    [
        (1e-3, 1.0, 1.05, 51),
        (5e-3, 1.0, 1.045, 10),
        (1e-2, 1.0, 1.05, 6),
        (1e-2, 1.0, 1.08, 9),
        (1e-2, 1.001, 1.009, 0),
    ],
)
def test_time_window_minimum_samples(rotor, dt, t_initial, t_final, samples):
    time = np.round(np.arange(0.0, 2.0, dt), 12)
    response = np.zeros((len(time), rotor.ndof))
    result = TimeResponseResults(rotor, time, response, [])

    if samples >= 10:
        window_time, _ = result._get_window(t_initial, t_final)
        assert len(window_time) == samples
        fig = result.plot_3d(t_initial=t_initial, t_final=t_final)
        assert len(fig.data[0].x) == samples
    else:
        message = (
            f"lasts {t_final - t_initial:.4g} s and the time step is {dt:.4g} s, "
            f"so it holds {samples} samples; at least 10 are needed"
        )
        with pytest.raises(ValueError, match=message):
            result._get_window(t_initial, t_final)
        with pytest.raises(ValueError, match=message):
            result.plot_1d(
                probe=[Probe(3, Q_(0, "rad"))],
                t_initial=Q_(t_initial * 1e3, "ms"),
                t_final=Q_(t_final * 1e3, "ms"),
            )


@pytest.mark.parametrize("t_initial, t_final", [(1.0, 1.0), (1.2, 1.1)])
def test_time_window_needs_t_initial_before_t_final(rotor, t_initial, t_final):
    time = np.linspace(0.0, 2.0, 2001)
    response = np.zeros((len(time), rotor.ndof))
    result = TimeResponseResults(rotor, time, response, [])

    with pytest.raises(ValueError, match="t_initial must be smaller than t_final"):
        result._get_window(t_initial, t_final)
    with pytest.raises(ValueError, match="t_initial must be smaller than t_final"):
        result.plot_2d(node=3, t_initial=t_initial, t_final=t_final)
