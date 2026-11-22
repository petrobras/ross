import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from plotly import graph_objects as go

from ross.bearings.fixed_geometry import FixedGeometryBearing
from ross.bearings.fluid_film_bearing import (
    FluidFilmBearing,
    fluid_film_bearing_example,
)
from ross.units import Q_

DATA_DIR = Path(__file__).parent / "data" / "fluid_film"


def bearing_kwargs_from_fixture(case_name, **overrides):
    """Translate a solver fixture's inputs into FluidFilmBearing kwargs."""
    doc = json.loads((DATA_DIR / f"{case_name}.json").read_text())
    inp = doc["inputs"]
    lubricant = {
        "liquid_viscosity1": inp["viscosity1"],
        "temperature1": inp["temp1"],
        "liquid_viscosity2": inp["viscosity2"],
        "temperature2": inp["temp2"],
        "liquid_density": inp["lube_density"],
        "liquid_specific_heat": inp["lube_cp"],
        "liquid_thermal_conductivity": inp["lube_conduct"],
    }
    probes = list(
        zip(inp["probe_pad_number"], inp["probe_theta"], inp["r_location"], strict=True)
    )
    kwargs = dict(
        n=0,
        speed=inp["frequency"],
        journal_diameter=inp["journal_diameter"],
        radial_clearance=inp["radial_clearance"],
        pad_thickness=inp["pad_thickness"],
        pivot_angle=inp["pivot_angle"],
        pad_arc=inp["pad_arc"],
        pad_axial_length=inp["pad_axial_length"],
        preload=inp["preload"],
        offset=inp["offset"],
        lubricant=lubricant,
        oil_supply_temperature=inp["oil_supply_temperature"],
        oil_flow_v=inp["oil_flow_v"],
        weight=inp["weight"],
        fxs_load=inp["fxs_load"],
        fys_load=inp["fys_load"],
        bearing_type=inp["bearing_type"],
        operating_type=inp["operating_type"],
        thermal_type=inp["thermal_type"],
        temp_j_type=inp["temp_j_type"],
        deform_type=inp["deform_type"],
        equilibrium_type=inp["equilibrium_type"],
        sump_type=inp["sump_type"],
        pivot_type=inp["pivot_type"],
        total_ex_film=inp["total_e_x_film"],
        total_ey_film=inp["total_e_y_film"],
        total_ez_film=inp["total_e_z_film"],
        total_ey_pad=inp["total_e_y_pad"],
        track_arc=inp["track_arc"],
        track_axial_length=inp["track_axial_length"],
        track_depth=inp["track_depth"],
        taper_depth_le=inp["taper_depth_le"],
        taper_arc_le=inp["taper_arc_le"],
        taper_depth_te=inp["taper_depth_te"],
        taper_arc_te=inp["taper_arc_te"],
        pocket_arc=inp["pocket_arc"],
        pocket_axial_length=inp["pocket_axial_length"],
        pad_E=inp["pad_young"],
        pad_poisson=inp["pad_poisson"],
        pad_conductivity=inp["pad_conductivity"],
        pad_expansion=inp["pad_expansion"],
        pad_density=inp["pad_density"],
        journal_expansion=inp["journal_expansion"],
        shell_expansion=inp["shell_expansion"],
        pad_convection=inp["pad_convection"],
        edges_convection=inp["edges_convection"],
        environment_temperature=inp["environment_temperature"],
        environment_convection=inp["environment_convection"],
        sump_convect_area=inp["sump_convect_area"],
        house_diameter=inp["house_diameter"],
        pivot_diameter=inp["pivot_diameter"],
        pivot_stiffness=inp["pivot_stiffness"],
        crush_fit=inp["crush_fit"],
        shell_id=inp["shell_id"],
        shell_od=inp["shell_od"],
        ambient_pressure_1=inp["ambient_pressure_1"],
        ambient_pressure_2=inp["ambient_pressure_2"],
        cavitation_pressure=inp["cavitation_pressure"],
        oil_supply_pressure=inp["oil_supply_pressure"],
        reference_temperature=inp["reference_temperature"],
        journal_temperature=inp["journal_temperature"],
        probes=probes,
        excitation_ratio=inp["excit_ratios"],
        initial_position=(inp["xj"], inp["yj"]),
        starvation_number=inp["starve_number"],
        hot_oil_lambda=inp["hot_oil_lambda"],
        relax_pressure=inp["relax_p"],
        relax_temperature=inp["relax_t"],
        relax_deformation=inp["relax_d"],
        relax_pivot=inp["relax_pivot"],
        re_laminar=inp["re_lower"],
        re_turbulent=inp["re_upper"],
    )
    kwargs.update(overrides)
    return kwargs, doc["outputs"]


@pytest.fixture(scope="module")
def fixture_bearing():
    kwargs, outputs = bearing_kwargs_from_fixture("fixed_isoviscous")
    return FluidFilmBearing(**kwargs), outputs


def test_matches_solver_fixture(fixture_bearing):
    """The wrapper's input assembly reproduces the raw solver fixture."""
    bearing, outputs = fixture_bearing
    for name in ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"):
        assert_allclose(
            np.asarray(getattr(bearing, name), dtype=float),
            np.asarray(outputs[name], dtype=float),
            rtol=1e-8,
            err_msg=f"{name} differs from the solver fixture",
        )


def test_results_summary_outputs(fixture_bearing):
    bearing, outputs = fixture_bearing
    result_out = bearing._results.outputs[0]
    assert_allclose(result_out["eccentricity"], outputs["eccentricity"], rtol=1e-8)
    assert_allclose(result_out["power_loss"], outputs["power_loss"], rtol=1e-8)


def test_coefficients_entry_point(fixture_bearing):
    bearing, outputs = fixture_bearing
    speed = bearing.speed[0]
    stiffness, damping = bearing.coefficients(speed)
    assert_allclose(stiffness[0], outputs["kxx"][0], rtol=1e-8)
    assert_allclose(damping[3], outputs["cyy"][0], rtol=1e-8)


def test_plots(fixture_bearing):
    bearing, _ = fixture_bearing
    for method in (
        "plot_pressure_2d",
        "plot_pressure_3d",
        "plot_temperature_2d",
        "plot_film_temperature_3d",
        "plot_film_thickness_2d",
    ):
        fig = getattr(bearing, method)()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == bearing.n_pads
    figures = bearing.plot_results()
    assert set(figures) == {
        "pressure_2d",
        "pressure_3d",
        "temperature_2d",
        "film_temperature_3d",
    }


@pytest.fixture(scope="module")
def full_thermal_bearing():
    kwargs, outputs = bearing_kwargs_from_fixture("fixed_full_coarse")
    return FluidFilmBearing(**kwargs), outputs


def test_pad_temperature_fields(full_thermal_bearing):
    bearing, _ = full_thermal_bearing
    results = bearing._results
    field = results.pad_temperature_fields[0]
    radii = results.pad_radial_positions
    n_pads, _, n_radial = field.shape
    assert n_pads == bearing.n_pads
    assert radii.shape == (n_radial,)
    assert_allclose(radii[-1] - radii[0], float(bearing.pad_thickness), rtol=1e-12)
    out = results.outputs[0]
    assert_allclose(field[:, :, 0], np.asarray(out["tc"][0]), rtol=1e-12)
    assert_allclose(field[:, :, -1], np.asarray(out["tb"][0]), rtol=1e-12)


def test_plot_pad_temperature_3d(full_thermal_bearing):
    bearing, _ = full_thermal_bearing
    field = bearing._results.pad_temperature_fields[0]
    radii = bearing._results.pad_radial_positions
    n_pads, n_theta, n_radial = field.shape

    fig = bearing.plot_pad_temperature_3d()
    mesh = fig.data[0]
    assert [trace.type for trace in fig.data] == ["mesh3d"] + ["scatter3d"] * 3
    assert [trace.name for trace in fig.data[2:]] == ["Axes", "Axes"]
    assert len(mesh.x) == n_pads * n_theta * n_radial * 2
    assert_allclose(
        [np.min(mesh.intensity), np.max(mesh.intensity)],
        [field.min() - 273.15, field.max() - 273.15],
        rtol=1e-12,
    )
    # the bearing axis runs along the scene y axis, like the rotor shape plots
    radius = np.hypot(np.asarray(mesh.x), np.asarray(mesh.z))
    assert_allclose([radius.min(), radius.max()], [radii[0], radii[-1]], rtol=1e-12)

    fig = bearing.plot_pad_temperature_3d(show_interface=False)
    assert [trace.type for trace in fig.data] == ["mesh3d"] + ["scatter3d"] * 2


def test_plot_pad_temperature_3d_requires_full_thermal(fixture_bearing):
    bearing, _ = fixture_bearing
    with pytest.raises(ValueError, match="full thermal model"):
        bearing.plot_pad_temperature_3d()


def test_multi_speed_and_parallel():
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    kwargs["speed"] = [80.0, 110.0]
    serial = FluidFilmBearing(**kwargs)
    assert np.asarray(serial.kxx).shape == (2,)
    assert len(serial._results.pressure_fields) == 2
    assert float(serial.kxx[1]) != float(serial.kxx[0])

    parallel = FluidFilmBearing(**kwargs, num_processes=2)
    for name in ("kxx", "kyy", "cxx", "cyy"):
        assert_allclose(
            np.asarray(getattr(parallel, name), dtype=float),
            np.asarray(getattr(serial, name), dtype=float),
            rtol=0.0,
            atol=0.0,
        )


def test_lubricant_resolution():
    bearing = fluid_film_bearing_example()
    assert bearing.lubricant_properties["lube_density"] == pytest.approx(873.99629)

    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    kwargs["lubricant"] = "NOT_A_LUBRICANT"
    with pytest.raises(ValueError, match="lubricant must be one of"):
        FluidFilmBearing(**kwargs)

    kwargs["lubricant"] = {"liquid_viscosity1": 0.02}
    with pytest.raises(ValueError, match="missing properties"):
        FluidFilmBearing(**kwargs)


def test_input_validation():
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")

    bad = dict(kwargs, total_ex_film=21)
    with pytest.raises(ValueError, match="must be an even number"):
        FluidFilmBearing(**bad)

    bad = dict(kwargs, thermal_type="not_a_model")
    with pytest.raises(ValueError, match="thermal_type must be one of"):
        FluidFilmBearing(**bad)

    bad = dict(kwargs, oil_flow_v=None)
    with pytest.raises(ValueError, match="oil_flow_v not informed"):
        FluidFilmBearing(**bad)

    bad = dict(kwargs, preload=[0.0])
    with pytest.raises(ValueError, match="inconsistent with number of pads"):
        FluidFilmBearing(**bad)


def test_save_downgrades_to_coefficient_table(fixture_bearing, tmp_path):
    from ross.utils import load_data

    bearing, _ = fixture_bearing
    file = tmp_path / "bearing.toml"
    bearing.save(file)
    data = load_data(file)
    (key,) = data.keys()
    assert key.startswith("BearingElement_")
    assert_allclose(data[key]["kxx"], np.asarray(bearing.kxx, dtype=float), rtol=1e-12)

    from ross.bearing_seal_element import BearingElement

    loaded = BearingElement.load(file)
    assert_allclose(
        np.asarray(loaded.kxx, dtype=float),
        np.asarray(bearing.kxx, dtype=float),
        rtol=1e-12,
    )
    assert_allclose(loaded.speed, bearing.speed)


def test_example_with_pint_units():
    bearing = fluid_film_bearing_example()
    assert bearing.n_pads == 2
    assert bearing.journal_diameter == pytest.approx(Q_(15.748, "in").to("m").m)
    assert float(bearing.kxx[0]) > 1e8


def test_2d_frequency_table():
    """2-D (speed, frequency) table: one results entry per pair."""
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    speed = np.atleast_1d(np.asarray(kwargs.pop("speed"), dtype=float))
    kwargs.pop("excitation_ratio", None)
    frequencies = np.array([0.5, 1.0]) * speed[0]

    bearing = FluidFilmBearing(speed=speed, frequency=frequencies, **kwargs)

    kxx = np.asarray(bearing.kxx, dtype=float)
    assert kxx.shape == (speed.size, frequencies.size)
    assert bearing.kxx_interpolated.kind == "grid"

    # a rigid fixed-geometry film has no internal dofs to condense, so its
    # reduced coefficients do not depend on the whirl ratio
    reference = FluidFilmBearing(speed=speed, excitation_ratio=0.5, **kwargs)
    assert_allclose(kxx[0], reference.kxx[0], rtol=1e-10)

    # the element lookup follows both axes
    assert_allclose(
        float(bearing.kxx_interpolated(frequencies[0], speed[0])),
        kxx[0, 0],
        rtol=1e-12,
    )
    stiffness, _ = bearing.coefficients(speed[0], frequencies[0])
    assert_allclose(stiffness[0], kxx[0, 0], rtol=1e-12)

    # the results object lists one case per (speed, frequency) pair
    assert bearing._results.frequency.shape == (speed.size * frequencies.size,)


def test_2d_frequency_table_requires_nonzero_speed():
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    kwargs.pop("speed")
    with pytest.raises(ValueError, match="nonzero speeds"):
        FluidFilmBearing(speed=[0.0], frequency=[10.0], **kwargs)


def test_2d_frequency_table_save_load(tmp_path):
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    speed = np.atleast_1d(np.asarray(kwargs.pop("speed"), dtype=float))
    kwargs.pop("excitation_ratio", None)
    bearing = FluidFilmBearing(
        speed=speed, frequency=np.array([0.5, 1.0]) * speed[0], **kwargs
    )
    file = tmp_path / "bearing_2d.toml"
    bearing.save(file)

    from ross.bearing_seal_element import BearingElement

    loaded = BearingElement.load(file)
    assert_allclose(loaded.speed, bearing.speed)
    assert_allclose(loaded.frequency, bearing.frequency)
    assert_allclose(np.array(loaded.kxx), np.array(bearing.kxx), rtol=1e-12)
    assert_allclose(
        loaded.K(frequency=60.0, speed=speed[0]),
        bearing.K(frequency=60.0, speed=speed[0]),
    )


def fitzgerald_neal_bearing(rpm, ey_film):
    """Build the two-axial-groove bearing of Fitzgerald and Neal (1992), full thermal."""
    inch = 0.0254
    lbf = 4.4482216152605
    reyn = lbf / inch**2
    return FixedGeometryBearing(
        n=0,
        speed=Q_([rpm], "RPM"),
        bearing_type="fixed_geometry",
        thermal_type="full",
        journal_diameter=3.0 * inch,
        radial_clearance=0.003 * inch,
        pad_thickness=0.374 * inch,
        pivot_angle=Q_([90.0, 270.0], "deg"),
        pad_arc=Q_([151.3, 151.3], "deg"),
        pad_axial_length=[1.5 * inch, 1.5 * inch],
        preload=[0.0, 0.0],
        offset=[0.5, 0.5],
        lubricant={
            "liquid_viscosity1": 4.13e-6 * reyn,
            "temperature1": Q_(100.0, "degF").to("K").m,
            "liquid_viscosity2": 6.8e-7 * reyn,
            "temperature2": Q_(210.0, "degF").to("K").m,
            "liquid_density": 855.0,
            "liquid_specific_heat": 1884.0,
            "liquid_thermal_conductivity": 0.13,
        },
        oil_supply_temperature=Q_(122.0, "degF"),
        oil_flow_v=Q_(10.0, "gallon/min"),
        weight=1220.0 * lbf,
        fxs_load=0.0,
        fys_load=0.0,
        hot_oil_lambda=0.8,
        total_ex_film=20,
        total_ey_film=ey_film,
        total_ez_film=10,
        total_ey_pad=10,
    )


@pytest.mark.parametrize(
    "rpm, ey_coarse, ey_fine",
    [(2000.0, 12, 14), (4000.0, 16, 18), (6000.0, 18, 22), (8000.0, 20, 24)],
)
def test_full_thermal_first_film_layer_keeps_dissipation(rpm, ey_coarse, ey_fine):
    """Refining the cross-film mesh past h_min / total_ey_film = 1 um is smooth.

    At each speed the coarse mesh puts the first film node more than 1 um
    above the pad at the minimum film thickness, and the fine mesh less than
    1 um. When the film/pad interface test used a tolerance of 1e-6 m, the
    first film layer of the fine mesh was taken for solid and lost its viscous
    dissipation: between the two meshes the peak pad temperature dropped by
    0.7 to 1.0 K and the power loss rose by 0.4 to 0.9 %. With the interface
    found correctly the same refinement moves them by less than 0.02 K and
    0.1 %. The bounds sit between the two.
    """
    coarse = fitzgerald_neal_bearing(rpm, ey_coarse)._results.outputs[0]
    fine = fitzgerald_neal_bearing(rpm, ey_fine)._results.outputs[0]
    assert min(fine["h_min"][0]) / ey_fine < 1.0e-6
    assert min(coarse["h_min"][0]) / ey_coarse > 1.0e-6
    assert_allclose(fine["tpad_max"][0], coarse["tpad_max"][0], atol=0.05)
    assert_allclose(fine["power_loss"][0], coarse["power_loss"][0], rtol=2.0e-3)


class _SolverCalled(Exception):
    """Raised by the patched solver: a test reached the first solve."""


@pytest.fixture
def no_solve(monkeypatch):
    """Make any solver call fail, to show that a check runs before it."""
    import ross.bearings.fluid_film_bearing as module

    def solve(inputs):
        raise _SolverCalled

    monkeypatch.setattr(module, "_solve_case", solve)


# Inputs for the range checks: pivot data, a zero speed in a 1-D table,
# preload, pad layout and the other arguments. The valid range
# of each argument follows from the solver's formulas, not from a run: the pad
# clearance radial_clearance / (1 - preload) is infinite at preload 1, the
# Hertz pivot formulas divide by house_diameter - pivot_diameter, the
# user-specified pivot deflection is force / pivot_stiffness, and the leading
# edge of a pad is pivot_angle - offset * pad_arc.
def _pads(pivot_angle_deg, pad_arc_deg):
    """Per-pad inputs of a layout with ``len(pivot_angle_deg)`` plain lands."""
    n_pads = len(pivot_angle_deg)
    per_pad = dict(
        pivot_angle=np.radians(pivot_angle_deg),
        pad_arc=np.radians(pad_arc_deg),
        pad_axial_length=[0.2] * n_pads,
        preload=[0.0] * n_pads,
        offset=[0.5] * n_pads,
        pad_convection=735.9,
    )
    for name in (
        "track_arc",
        "track_axial_length",
        "track_depth",
        "taper_depth_le",
        "taper_arc_le",
        "taper_depth_te",
        "taper_arc_te",
    ):
        per_pad[name] = None
    return per_pad


INVALID_INPUTS = [
    ("speed", dict(speed=[0.0, 300.0]), "nonzero speeds"),
    ("speed", dict(speed=[300.0, -10.0]), "nonzero speeds"),
    ("speed", dict(speed=[np.nan]), "nonzero speeds"),
    ("speed", dict(speed=[np.inf]), "nonzero speeds"),
    ("frequency", dict(speed=[300.0], frequency=[-1.0, 100.0]), "frequency"),
    ("frequency", dict(speed=[300.0], frequency=[np.nan]), "frequency"),
    ("excitation_ratio", dict(excitation_ratio=-0.5), "excitation_ratio"),
    ("excitation_ratio", dict(excitation_ratio=np.nan), "excitation_ratio"),
    ("journal_diameter", dict(journal_diameter=0.0), "journal_diameter"),
    ("radial_clearance", dict(radial_clearance=-1e-4), "radial_clearance"),
    ("radial_clearance", dict(radial_clearance=np.nan), "radial_clearance"),
    ("pad_thickness", dict(pad_thickness=None), "pad_thickness"),
    ("pad_thickness", dict(pad_thickness=0.0), "pad_thickness"),
    ("oil_flow_v", dict(oil_flow_v=0.0), "oil_flow_v"),
    ("pad_density", dict(pad_density=0.0), "pad_density"),
    ("pad_axial_length", dict(pad_axial_length=[0.2, 0.0]), "pad_axial_length"),
    ("pad_arc", dict(pad_arc=[np.radians(176), 0.0]), "pad_arc"),
    ("preload", dict(preload=[1.0, 1.0]), "preload"),
    ("preload", dict(preload=[0.0, -0.1]), "preload"),
    ("offset", dict(offset=[0.5, 1.2]), "offset"),
    ("offset", dict(offset=[-0.1, 0.5]), "offset"),
    # three 170 deg pads and two 200 deg pads (more than 360 deg in total)
    ("pad_arc", _pads([60, 180, 300], [170] * 3), "more than 360 deg"),
    ("pad_arc", dict(pad_arc=np.radians([200, 200])), "more than 360 deg"),
    # two 100 deg pads centred 90 deg apart overlap by 10 deg (total 200 deg)
    (
        "pad_arc",
        dict(pivot_angle=np.radians([90, 180]), pad_arc=np.radians([100, 100])),
        "overlaps",
    ),
    # the offset moves the leading edge: centred pads 180 deg apart with 170 deg
    # arcs fit, but offset 0 on pad 1 moves its arc 85 deg forwards into pad 2
    (
        "pad_arc",
        dict(pad_arc=np.radians([170, 170]), offset=[0.0, 0.5]),
        "overlaps",
    ),
    # the leading edge is pivot - offset * arc (not +): pads at 90 and 200 deg,
    # 100 deg arcs, offsets 0.5 and 1 span [40, 140] and [100, 200] deg; with
    # "+" they would span [140, 240] and [300, 400] and fit
    (
        "pad_arc",
        dict(
            pivot_angle=np.radians([90, 200]),
            pad_arc=np.radians([100, 100]),
            offset=[0.5, 1.0],
        ),
        "overlaps",
    ),
    # an overlap of 1e-6 rad with the total arc under 360 deg: half-circle land
    # from 2e-6 rad to pi + 2e-6 rad, and a land from pi + 1e-6 rad to 2 pi
    (
        "pad_arc",
        dict(
            pivot_angle=[np.pi / 2 + 2e-6, 3 * np.pi / 2],
            pad_arc=[np.pi, np.pi - 2e-6],
        ),
        "overlaps",
    ),
    # pivot models without the data they need
    (
        "pivot_stiffness",
        dict(
            deform_type="pad_pivot_mechanical",
            pivot_type="user_specified_stiffness",
            pivot_stiffness=0.0,
        ),
        "pivot_stiffness",
    ),
    (
        "pivot_stiffness",
        dict(
            deform_type="pad_pivot_mechanical",
            pivot_type="user_specified_stiffness",
            pivot_stiffness=-1e8,
        ),
        "pivot_stiffness",
    ),
    (
        "pivot_diameter",
        dict(deform_type="pad_pivot_mechanical", pivot_type="ball_in_socket"),
        "pivot_diameter",
    ),
    (
        "pivot_diameter",
        dict(
            deform_type="pad_mechanical_thermal_shaft_shell_thermal_pivot_mechanical",
            pivot_type="button",
            pivot_diameter=0.4,
            house_diameter=0.4,
        ),
        "house_diameter",
    ),
    (
        "pivot_diameter",
        dict(
            deform_type="pad_pivot_mechanical",
            pivot_type="ball_in_socket",
            pivot_diameter=0.42,
            house_diameter=0.4,
        ),
        "house_diameter",
    ),
]


@pytest.mark.parametrize(
    "argument, overrides, match",
    INVALID_INPUTS,
    ids=[f"{case[0]}-{i}" for i, case in enumerate(INVALID_INPUTS)],
)
def test_invalid_inputs_raise_before_solving(no_solve, argument, overrides, match):
    """Each input outside the solver's range raises before the first solve."""
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous", **overrides)
    with pytest.raises(ValueError, match=match) as error:
        FluidFilmBearing(**kwargs)
    assert argument in str(error.value)


@pytest.mark.parametrize(
    "overrides",
    [
        # pivot data that the formulas accept: no pivot flexibility selected
        dict(pivot_type="ball_in_socket"),
        dict(
            deform_type="pad_mechanical",
            pivot_type="user_specified_stiffness",
            pivot_stiffness=0.0,
        ),
        # a single 360 deg land, and two lands that meet exactly
        _pads([180], [360]),
        _pads([60, 180, 300], [119.99] * 3),
        dict(pad_arc=np.radians([180, 180])),
        # a pivot angle beyond 360 deg is the same layout
        dict(pivot_angle=np.radians([90 + 360, 270])),
        # offset 1 puts the arc behind the pivot, [-10, 90] deg, clear of the
        # second pad [220, 320] deg (with "+" it would be [190, 290] deg)
        dict(
            pivot_angle=np.radians([90, 270]),
            pad_arc=np.radians([100, 100]),
            offset=[1.0, 0.5],
        ),
        # a gap of 1e-6 rad: [2e-6, pi + 2e-6] and [pi + 3e-6, 2 pi] rad
        dict(
            pivot_angle=[np.pi / 2 + 2e-6, 3 * np.pi / 2 + 1e-6],
            pad_arc=[np.pi, np.pi - 4e-6],
        ),
        dict(preload=[0.99, 0.0], offset=[0.0, 1.0], pad_arc=np.radians([90, 90])),
        dict(speed=[300.0], frequency=[0.0, 100.0]),
        dict(excitation_ratio=0.0),
    ],
)
def test_valid_inputs_reach_the_solver(no_solve, overrides):
    """Inputs at the edges of the valid ranges pass the checks."""
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous", **overrides)
    with pytest.raises(_SolverCalled):
        FluidFilmBearing(**kwargs)


@pytest.mark.parametrize(
    "load",
    [dict(fys_load=-500.0), dict(fxs_load=200.0), dict(weight=1e3)],
)
def test_match_eccentricity_with_load_warns(no_solve, load):
    """The held-eccentricity equilibrium does not use the load."""
    loads = {"weight": 0.0, "fxs_load": 0.0, "fys_load": 0.0} | load
    kwargs, _ = bearing_kwargs_from_fixture(
        "fixed_isoviscous",
        equilibrium_type="match_eccentricity",
        ambient_pressure_1=0.0,
        ambient_pressure_2=0.0,
        **loads,
    )
    with pytest.warns(UserWarning, match="match_load"):
        with pytest.raises(_SolverCalled):
            FluidFilmBearing(**kwargs)

    kwargs["equilibrium_type"] = "match_load"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(_SolverCalled):
            FluidFilmBearing(**kwargs)


@pytest.mark.parametrize(
    "operating_type, pressures, warns",
    [
        ("regular_flooded", (100001.56, 100001.56), True),
        ("regular_flooded", (0.0, 2e5), True),
        ("regular_flooded", (0.0, 0.0), False),
        ("high_ambient_pressure", (2e5, 2e5), False),
        ("axial_flow", (1e5, 0.0), False),
    ],
)
def test_flooded_ambient_pressure_warns(no_solve, operating_type, pressures, warns):
    """The flooded mode puts the axial edges at gauge 0 (pressure.py)."""
    kwargs, _ = bearing_kwargs_from_fixture(
        "fixed_isoviscous",
        operating_type=operating_type,
        ambient_pressure_1=pressures[0],
        ambient_pressure_2=pressures[1],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_SolverCalled):
            FluidFilmBearing(**kwargs)
    messages = [str(w.message) for w in caught if "ambient_pressure" in str(w.message)]
    assert bool(messages) is warns
    # the groove pressure is still applied in flooded mode (see the test below)
    assert all("groove" in m and "not used" not in m for m in messages)


def test_flooded_groove_pressure_acts_as_a_constant_load():
    """In flooded mode the ambient pressures still load the grooves.

    Two lands of 160 and 100 deg (pivots 270 and 60 deg, offset 0.5) leave
    grooves of 20 and 80 deg. The groove pressure p = 2e5 Pa pushes on the
    journal with F = -p R L * integral of (cos, sin) over the groove arcs,
    integrated here by quadrature. The build with p must therefore match the
    build at p = 0 with F added to the applied load. The 160 deg land sits
    under the load: with it at 90 deg, away from the load, none of the three
    equilibria converges.
    """
    from scipy.integrate import quad

    p = 2e5
    layout = dict(
        pivot_angle=np.radians([270.0, 60.0]),
        pad_arc=np.radians([160.0, 100.0]),
        offset=[0.5, 0.5],
        operating_type="regular_flooded",
    )
    kwargs, _ = bearing_kwargs_from_fixture(
        "fixed_isoviscous", ambient_pressure_1=p, ambient_pressure_2=p, **layout
    )
    radius = kwargs["journal_diameter"] / 2
    length = kwargs["pad_axial_length"][0]
    grooves = np.radians([(350.0, 370.0), (110.0, 190.0)])
    fx, fy = (
        -sum(p * radius * length * quad(f, a, b)[0] for a, b in grooves)
        for f in (np.cos, np.sin)
    )
    assert abs(np.hypot(fx, fy)) > 1e3  # the asymmetric grooves do not cancel

    with pytest.warns(UserWarning, match="groove"):
        loaded = FluidFilmBearing(**kwargs)
    kwargs.update(
        ambient_pressure_1=0.0,
        ambient_pressure_2=0.0,
        fxs_load=float(kwargs["fxs_load"]) + fx,
        fys_load=float(kwargs["fys_load"]) + fy,
    )
    shifted = FluidFilmBearing(**kwargs)
    assert loaded.results.converged[0] and shifted.results.converged[0]
    out_loaded = loaded._results.outputs[0]
    out_shifted = shifted._results.outputs[0]
    for key in ("eccentricity", "attitude"):
        assert_allclose(out_loaded[key][0], out_shifted[key][0], rtol=1e-9)
    for coef in ("kxx", "kxy", "kyx", "kyy", "cxx", "cyy"):
        assert_allclose(
            getattr(loaded, coef), getattr(shifted, coef), rtol=1e-9, err_msg=coef
        )
    # and differs from the p = 0 build without the extra load
    kwargs.update(
        fxs_load=float(kwargs["fxs_load"]) - fx, fys_load=float(kwargs["fys_load"]) - fy
    )
    unloaded = FluidFilmBearing(**kwargs)
    assert unloaded.results.converged[0]
    assert abs(unloaded.kyy[0] / loaded.kyy[0] - 1) > 5e-3


def test_groove_force_does_not_depend_on_the_pad_order():
    """The same two lands listed in either order give the same bearing.

    With the pads listed as (270, 60) deg the groove between them crosses
    0 rad; its arc was computed by adding 360 (degrees) to an angle in radians,
    which turned a 20 deg groove into a 354 rad one. Listed as
    (60, 270) deg the wrap is handled by the first pad's branch, which was
    right. Reference: the second ordering, and the groove-force quadrature of
    ``test_flooded_groove_pressure_acts_as_a_constant_load``.
    """
    common = dict(
        pad_arc=np.radians([160.0, 100.0]),
        offset=[0.5, 0.5],
        ambient_pressure_1=2e5,
        ambient_pressure_2=2e5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        listed = []
        for pivots, arcs in (
            ((270.0, 60.0), (160.0, 100.0)),
            ((60.0, 270.0), (100.0, 160.0)),
        ):
            kwargs, _ = bearing_kwargs_from_fixture(
                "fixed_isoviscous",
                **dict(
                    common, pivot_angle=np.radians(pivots), pad_arc=np.radians(arcs)
                ),
            )
            listed.append(FluidFilmBearing(**kwargs))
    first, second = listed
    assert first.results.converged[0] and second.results.converged[0]
    for key in ("eccentricity", "attitude", "power_loss"):
        assert_allclose(
            first.results.outputs[0][key][0],
            second.results.outputs[0][key][0],
            rtol=1e-9,
            err_msg=key,
        )
    for coef in ("kxx", "kxy", "kyx", "kyy", "cxx", "cyy"):
        assert_allclose(
            getattr(first, coef), getattr(second, coef), rtol=1e-9, err_msg=coef
        )


def _blend(re_max, re_lower, re_upper, exponent):
    """Transition-regime turbulence scale, written out from its definition."""
    return 1.0 - ((re_upper - re_max) / (re_upper - re_lower)) ** exponent


TURBULENCE_LIMITS = [(1000.0, 2000.0, 0.5), (500.0, 1000.0, 0.125), (3.0, 7.0, 1.0)]


@pytest.mark.parametrize("re_lower, re_upper, exponent", TURBULENCE_LIMITS)
def test_flow_regime_at_the_laminar_limit(re_lower, re_upper, exponent):
    """Re = re_lower is the laminar end of the blend, (1, 0.0), in both paths.

    Reference: the blend 1 - ((re_upper - Re) / (re_upper - re_lower)) ** n is 0
    at re_lower, so the regime is continuous there; before the fix the value
    re_lower itself was sent to the turbulent regime (2, 1.0).
    """
    from types import SimpleNamespace

    from ross.bearings.fluid_film.driver import _flow_regime
    from ross.bearings.fluid_film.thermal import _update_flow_regime

    turbulence = SimpleNamespace(
        re_lower=re_lower, re_upper=re_upper, scale_factor_exponent=exponent
    )

    def thermal_path(re_max):
        regime = np.full(2, -1, dtype=np.int64)
        scale = np.full(2, -1.0)
        _update_flow_regime(regime, scale, 1, re_max, turbulence)
        return int(regime[1]), float(scale[1])

    below = np.nextafter(re_lower, -np.inf)
    above = re_lower * (1 + 1e-9)
    inside = 0.5 * (re_lower + re_upper)

    def driver_path(re_max):
        return _flow_regime(re_max, turbulence)

    for path in (driver_path, thermal_path):
        regime, scale = path(re_lower)
        assert (regime, scale) == (1, 0.0)
        assert path(below) == (0, 0.0)
        regime, scale = path(above)
        assert regime == 1
        assert_allclose(scale, _blend(above, re_lower, re_upper, exponent), rtol=1e-12)
        assert scale < 1e-6
        regime, scale = path(inside)
        assert regime == 1
        assert_allclose(scale, _blend(inside, re_lower, re_upper, exponent), rtol=1e-12)
        assert path(re_upper) == (2, 1.0)


def _one_node_flow_regime(region, re_lower, re_upper, exponent):
    """Call the numba flow-regime kernel on a one-node, one-pad film.

    The node sits in the pocket (track) when ``region == "track"`` and in the
    dam otherwise; the film is 1e-4 m thick with a uniform viscosity.
    """
    from ross.bearings.fluid_film._numba_kernels import flow_regime_jit

    pad_thickness = 0.01
    h = 1.0e-4
    length_track = 1.0 if region == "track" else 0.0
    return flow_regime_jit(
        0,
        3,
        0,
        0,
        2,
        1,
        np.array([0], dtype=np.int64),
        np.array([[0, 1, 2]], dtype=np.int64),
        np.array([2.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([length_track]),
        np.array([0.0]),
        pad_thickness,
        np.array([[0.5]]),
        np.array([[0.5]]),
        850.0,
        40.0,
        re_lower,
        re_upper,
        np.full((1, 3), 0.02),
        np.array([[pad_thickness, pad_thickness + h / 2, pad_thickness + h]]),
        np.array([[h]]),
        np.zeros((1, 1)),
        exponent,
    )


@pytest.mark.parametrize("region", ["track", "dam"])
@pytest.mark.parametrize("exponent", [0.5, 0.125])
def test_numba_flow_regime_at_the_laminar_limit(region, exponent):
    """Re = re_lower in the numba kernel, for the track and the dam regions."""
    re_node = _one_node_flow_regime(region, 1e9, 2e9, exponent)[4]
    # rho U h / mu with the inputs above
    assert_allclose(re_node, 850.0 * 40.0 * 1.0e-4 / 0.02, rtol=1e-12)

    index = (0, 1) if region == "track" else (2, 3)

    def regime(re_lower):
        out = _one_node_flow_regime(region, re_lower, 2 * re_node, exponent)
        return int(out[index[0]]), float(out[index[1]])

    assert regime(re_node) == (1, 0.0)
    assert regime(np.nextafter(re_node, np.inf)) == (0, 0.0)
    regime_below, scale_below = regime(re_node * (1 - 1e-9))
    assert regime_below == 1
    assert_allclose(
        scale_below,
        _blend(re_node, re_node * (1 - 1e-9), 2 * re_node, exponent),
        rtol=1e-6,
    )
    assert scale_below < 1e-6


def _attitude_quadrants(xj, yj):
    """Attitude angle as computed before this check was added (0 on both axes)."""
    if xj == 0.0 or yj == 0.0:
        return 0.0
    a = np.arctan(abs(xj) / abs(yj))
    if xj > 0.0 and yj > 0.0:
        return np.pi - a
    if xj < 0.0 and yj > 0.0:
        return -np.pi + a
    if xj < 0.0 and yj < 0.0:
        return -a
    return a


def _same_angle(a, b, atol=1e-9):
    return abs(np.angle(np.exp(1j * (a - b)))) < atol


@pytest.mark.parametrize(
    "position, expected, off_axis",
    [
        ((0.0, -0.3), 0.0, (1e-12, -0.3)),
        ((0.0, 0.3), np.pi, (1e-12, 0.3)),
        ((-0.0, 0.3), np.pi, (1e-12, 0.3)),
        ((0.3, 0.0), np.pi / 2, (0.3, -1e-12)),
        ((0.3, -0.0), np.pi / 2, (0.3, -1e-12)),
        ((-0.3, 0.0), -np.pi / 2, (-0.3, -1e-12)),
    ],
)
def test_attitude_on_the_axes(position, expected, off_axis):
    """The attitude is arctan2(x, -y), continuous onto the axes.

    Reference: the quadrant formula used before the fix, evaluated just off
    the axis (it returned 0 on the axes themselves). The +y axis takes the
    limit from x -> 0+ (+pi).
    """
    from ross.bearings.fluid_film.driver import _attitude_rad

    attitude = _attitude_rad(*position)
    assert attitude == pytest.approx(expected, abs=1e-15)
    assert _same_angle(attitude, _attitude_quadrants(*off_axis), atol=1e-11)


def test_attitude_off_the_axes():
    from ross.bearings.fluid_film.driver import _attitude_rad

    assert _attitude_rad(0.0, 0.0) == 0.0
    assert _attitude_rad(-0.0, -0.0) == 0.0
    rng = np.random.default_rng(1705)
    radius = rng.uniform(1e-6, 1.5, 200)
    angle = rng.uniform(-np.pi, np.pi, 200)
    for xj, yj in zip(radius * np.cos(angle), radius * np.sin(angle), strict=True):
        new = _attitude_rad(xj, yj)
        assert -np.pi < new <= np.pi
        assert _same_angle(new, _attitude_quadrants(xj, yj), atol=1e-12)


@pytest.mark.parametrize(
    "stiffness",
    [
        (0.0, 0.0, 0.0, 0.0),
        (1e8, 0.0, 0.0, 0.0),
        (0.0, 1e3, -1e3, 0.0),
        (0.0, 0.0, 0.0, 1e-7),
        (np.nan, 0.0, 0.0, 1e8),
        (1e8, np.inf, 0.0, 1e8),
    ],
)
def test_newton_step_treats_zero_or_nonfinite_stiffness_as_singular(stiffness):
    """A film with no force on any pad has zero stiffness: no Newton step.

    Before the fix ``kxx / kyy`` raised ZeroDivisionError (soft pivot) and
    a non-finite stiffness produced a NaN journal position.
    """
    from ross.bearings.fluid_film.hydrodynamics import _newton_step

    assert _newton_step(*stiffness, 0.0, -2800.0, 75e-6) == (None, None)


@pytest.mark.parametrize("kxy, kyx", [(0.0, 0.0), (1e3, -1e3)])
def test_newton_step_zero_kxx_keeps_the_decoupled_step(kxy, kyx):
    """``kxx = 0`` with a finite ``kyy`` is not singular: y-step ``fy / kyy``.

    This is the step the solver took before the soft-pivot fix; only
    ``kyy = 0`` divided by zero.
    """
    from ross.bearings.fluid_film.hydrodynamics import _newton_step

    step = _newton_step(0.0, kxy, kyx, 1e8, 30.0, -200.0, 75e-6)
    assert step == (0.0, -200.0 / 1e8)


@pytest.mark.parametrize(
    "stiffness, force",
    [
        ((1.2e8, 3e7, -2e7, 1.5e8), (10.0, -200.0)),
        ((5e8, -1e8, 2e8, 4e8), (-50.0, 40.0)),
        ((2e8, 1e-7, 1e-7, 3e8), (100.0, -100.0)),
    ],
)
def test_newton_step_regular_system(stiffness, force):
    """Within the step limit the step solves K d = F (numpy as reference)."""
    from ross.bearings.fluid_film.hydrodynamics import _newton_step

    kxx, kxy, kyx, kyy = stiffness
    step = _newton_step(kxx, kxy, kyx, kyy, *force, 75e-6)
    reference = np.linalg.solve([[kxx, kxy], [kyx, kyy]], force)
    assert_allclose(step, reference, rtol=1e-12)


COARSE_MESH = dict(
    total_ex_film=20, total_ez_film=10, total_ey_film=10, total_ey_pad=10
)


def _adiabatic_tilting_pad(**overrides):
    """Five-pad load-matched adiabatic tilting pad, coarse mesh."""
    from ross.bearings.tilting_pad import TiltingPad

    kwargs = dict(
        n=1,
        speed=Q_([3000], "RPM"),
        equilibrium_type="match_load",
        thermal_type="adiabatic",
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
        pad_arc=Q_([60] * 5, "deg"),
        pad_axial_length=[50.8e-3] * 5,
        preload=[0.5] * 5,
        offset=[0.5] * 5,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(10, "l/min"),
        fys_load=-2.8e3,
        **COARSE_MESH,
    )
    kwargs.update(overrides)
    return TiltingPad(**kwargs)


def _same_solution(first, second, rtol):
    for name in ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"):
        assert_allclose(
            np.asarray(getattr(first, name), dtype=float),
            np.asarray(getattr(second, name), dtype=float),
            rtol=rtol,
            atol=0.0,
            err_msg=name,
        )
    for key in ("eccentricity", "attitude", "power_loss", "tpad_max"):
        assert_allclose(
            first.results.outputs[0][key][0],
            second.results.outputs[0][key][0],
            rtol=rtol,
            atol=0.0,
            err_msg=key,
        )
    assert_allclose(
        first.results.temperature_fields[0],
        second.results.temperature_fields[0],
        rtol=rtol,
        atol=0.0,
    )


@pytest.fixture(scope="module")
def viscosity_law_runs():
    from ross.bearings.film.lubricant import Lubricant

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        runs = {
            "name": _adiabatic_tilting_pad(),
            "exponential_instance": _adiabatic_tilting_pad(
                lubricant=Lubricant.from_name("ISOVG32", "exponential")
            ),
            "walther": _adiabatic_tilting_pad(viscosity_model="walther"),
        }
    return runs


def test_name_and_exponential_instance_give_the_same_bearing(viscosity_law_runs):
    """A name resolves to the exponential law, as an exponential instance does."""
    name = viscosity_law_runs["name"]
    instance = viscosity_law_runs["exponential_instance"]
    assert name.viscosity_model == instance.viscosity_model == "exponential"
    assert name.lubricant_properties == instance.lubricant_properties
    _same_solution(name, instance, rtol=1e-12)


def test_an_instance_keeps_its_own_viscosity_law(viscosity_law_runs):
    """An explicit viscosity_model that differs from the instance's law warns."""
    from ross.bearings.film.lubricant import Lubricant

    oil = Lubricant.from_name("ISOVG32", "walther")
    with pytest.warns(UserWarning, match="its own viscosity law: 'walther' is used"):
        bearing = _adiabatic_tilting_pad(lubricant=oil, viscosity_model="exponential")
    assert bearing.viscosity_model == "walther"
    _same_solution(bearing, viscosity_law_runs["walther"], rtol=1e-12)
    # the same instance with the same (or no) model does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        from ross.bearings.fluid_film_bearing import _resolve_lubricant

        for model in (None, "walther"):
            _, law, used = _resolve_lubricant(oil, model, 313.15)
            assert used == "walther" and law["lube_model"] == 1
    with pytest.raises(ValueError, match="viscosity_model must be one of"):
        _resolve_lubricant("ISOVG32", "roelands", 313.15)


def test_walther_lowers_the_film_temperature_and_power_loss(viscosity_law_runs):
    """Sign of the law difference on an adiabatic tilting pad.

    Through the same table points Walther is below the exponential law
    between them (40 to 100 C for ISOVG32), so the film shears less, runs
    cooler and loses less power, while its viscosity at the hottest film
    temperature is still the lower one.
    """
    from ross.bearings.film.lubricant import Lubricant

    exponential = viscosity_law_runs["name"].operating_results.iloc[0]
    walther = viscosity_law_runs["walther"].operating_results.iloc[0]
    assert exponential["converged"] and walther["converged"]
    assert walther["T_max_film"] < exponential["T_max_film"]
    assert walther["power_loss"] < exponential["power_loss"]
    mu_walther = Lubricant.from_name("ISOVG32", "walther").viscosity(
        walther["T_max_film"]
    )
    mu_exponential = Lubricant.from_name("ISOVG32", "exponential").viscosity(
        exponential["T_max_film"]
    )
    assert mu_walther < mu_exponential


def test_walther_on_an_isoviscous_lubricant_equals_the_exponential_run():
    """Limiting case: both table viscosities equal, so both laws are constant."""
    lubricant = {
        "liquid_viscosity1": 0.02,
        "temperature1": 313.15,
        "liquid_viscosity2": 0.02,
        "temperature2": 373.15,
        "liquid_density": 870.0,
        "liquid_specific_heat": 1950.0,
        "liquid_thermal_conductivity": 0.13,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        exponential = _adiabatic_tilting_pad(lubricant=lubricant)
        walther = _adiabatic_tilting_pad(lubricant=lubricant, viscosity_model="walther")
    assert walther.viscosity_model == "walther"
    _same_solution(walther, exponential, rtol=1e-12)


def test_datasheet_lubricant_runs_with_supply_temperature_properties():
    """rho, c_p and k of a temperature-dependent lubricant at the supply."""
    from ross.bearings.film.lubricant import Lubricant
    from ross.bearings.fluid_film_bearing import _resolve_lubricant
    from ross.bearings.lubricants import lubricants_dict

    oil = Lubricant.from_datasheet(Q_(32.0, "cSt"), Q_(5.4, "cSt"), 870.0)
    supply = 323.15
    properties, law, model = _resolve_lubricant(oil, None, supply)
    assert model == "walther" and law["lube_model"] == 1
    assert properties["lube_density"] == oil.density_at(supply)
    assert properties["lube_cp"] == oil.specific_heat_at(supply)
    assert properties["lube_conduct"] == oil.thermal_conductivity
    assert law["walther_density"] == oil.density
    assert law["walther_expansion"] == oil.volumetric_expansion
    # the same oil as a dictionary (to_dict) keeps its law and properties
    assert _resolve_lubricant(oil.to_dict(), None, supply) == (properties, law, model)
    # a name with constant properties: the seven table values, unchanged
    from ross.bearings.fluid_film_bearing import _lubricant_properties

    properties, law, model = _resolve_lubricant("ISOVG46", "walther", supply)
    assert properties == _lubricant_properties("ISOVG46")
    assert model == "walther" and law["lube_model"] == 1
    # a plain dictionary with the exponential law keeps its table values,
    # temperature coefficients included
    data = dict(lubricants_dict["ISOVG46"], volumetric_expansion=7e-4)
    data["specific_heat_slope"] = 3.6
    assert _resolve_lubricant(data, None, supply) == (
        _lubricant_properties("ISOVG46"),
        {"lube_model": 0},
        "exponential",
    )


def _large_fast_journal(**overrides):
    """300 mm, 12,000 rpm plain journal, Re on the clearance about 1470."""
    from ross.bearings.plain_journal import PlainJournal

    kwargs = dict(
        n=0,
        pad_axial_length=0.2,
        journal_diameter=0.3,
        radial_clearance=250e-6,
        n_pads=2,
        pad_arc=Q_(160, "deg"),
        oil_supply_temperature=Q_(40, "degC"),
        speed=Q_([12000], "RPM"),
        fys_load=-20000.0,
        lubricant="ISOVG32",
        oil_flow_v=Q_(10, "l/min"),
        thermal_type=None,
        **COARSE_MESH,
    )
    kwargs.update(overrides)
    return PlainJournal(**kwargs)


def test_taylor_transition_limits():
    """Re_c = 41.2 sqrt(R / C_b) (Taylor 1923); fully turbulent at 2 Re_c."""
    bearing = _large_fast_journal(re_laminar="taylor")
    re_c = 41.2 * np.sqrt(0.15 / 250e-6)  # R / C_b = 600, Re_c = 1009
    assert_allclose(bearing.re_turbulent, 2.0 * re_c, rtol=1e-14)
    inputs = bearing._engine_inputs(float(bearing.speed_range[0]))
    assert_allclose(inputs["re_lower"], re_c, rtol=1e-14)
    assert_allclose(inputs["re_upper"], 2.0 * re_c, rtol=1e-14)
    op = bearing.operating_results.iloc[0]
    assert op["converged"]
    # the local re_max (about 1.5e3) is in the blend between the limits
    assert re_c < op["re_max"] < 2.0 * re_c
    assert_allclose(op["re_taylor"], re_c, rtol=1e-14)
    # re_cb = rho U C_b / mu_supply, by hand from the ISOVG32 table
    from ross.bearings.lubricants import lubricants_dict

    table = lubricants_dict["ISOVG32"]
    surface_speed = 12000 * np.pi / 30 * 0.15
    assert_allclose(
        op["re_cb"],
        table["liquid_density"]
        * surface_speed
        * 250e-6
        / table["liquid_viscosity1"],  # supply at the 40 C table point
        rtol=1e-12,
    )
    # Turbulence adds power loss: the Taylor run lies above the forced-laminar
    # run, and above the run with a higher turbulent limit (smaller blend
    # weight). The default limits are not a reference here: that run stops
    # before the equilibrium converges.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        laminar = _large_fast_journal(re_laminar=1e8, re_turbulent=2e8)
        wider = _large_fast_journal(re_laminar="taylor", re_turbulent=3000.0)
    for other in (laminar, wider):
        assert other.operating_results["converged"][0]
    assert wider.re_turbulent == 3000.0
    assert (
        laminar.operating_results["power_loss"][0]
        < wider.operating_results["power_loss"][0]
        < op["power_loss"]
    )


def test_turbulence_limits_resolution():
    from ross.bearings.fluid_film_bearing import _turbulence_limits

    re_c = 41.2 * np.sqrt(0.15 / 250e-6)
    assert _turbulence_limits(500, None, 0.15, 250e-6) == (500.0, 1000.0)
    assert _turbulence_limits(500, 1000, 0.15, 250e-6) == (500.0, 1000.0)
    assert _turbulence_limits(500, "taylor", 0.15, 250e-6) == (500.0, 2.0 * re_c)
    assert _turbulence_limits("taylor", None, 0.15, 250e-6) == (re_c, 2.0 * re_c)
    assert _turbulence_limits("taylor", 3000.0, 0.15, 250e-6) == (re_c, 3000.0)
    # an explicit 1000 is used as given, also when it equals the default of
    # the numerical case (R / C_b = 400: Re_c = 824 < 1000 < 2 Re_c)
    re_c = 41.2 * np.sqrt(0.1 / 250e-6)
    assert _turbulence_limits("taylor", 1000, 0.1, 250e-6) == (re_c, 1000.0)
    assert _turbulence_limits("taylor", None, 0.1, 250e-6) == (re_c, 2.0 * re_c)


@pytest.fixture
def constructed(monkeypatch):
    """Stop each element before its solve and keep it, with its inputs."""
    import ross.bearings.fluid_film_bearing as module

    elements = []

    def solve_all(self):
        elements.append((self, self._engine_inputs(float(self.speed_range[0]))))
        raise _SolverCalled

    monkeypatch.setattr(module.FluidFilmBearing, "_solve_all", solve_all)
    return elements


def _build_each_class(**overrides):
    from ross.bearings.fixed_geometry import EllipticalBearing

    kwargs, _ = bearing_kwargs_from_fixture(
        "fixed_isoviscous", ambient_pressure_1=0.0, ambient_pressure_2=0.0
    )
    kwargs.pop("re_turbulent")  # the fixture states the former default
    kwargs.update(overrides)
    builders = [
        lambda: FluidFilmBearing(**kwargs),
        lambda: _adiabatic_tilting_pad(**overrides),
        lambda: _large_fast_journal(**overrides),
        lambda: EllipticalBearing(
            n=0,
            speed=Q_([3000], "RPM"),
            journal_diameter=0.2,
            radial_clearance=150e-6,
            pad_thickness=0.05,
            pad_arc=Q_(150, "deg"),
            preload=0.5,
            pad_axial_length=[0.16, 0.16],
            lubricant="ISOVG32",
            oil_supply_temperature=Q_(40, "degC"),
            oil_flow_v=Q_(30, "l/min"),
            weight=45e3,
            thermal_type=None,
            **COARSE_MESH,
            **overrides,
        ),
    ]
    for build in builders:
        with pytest.raises(_SolverCalled):
            build()


def test_default_turbulent_limit_is_unchanged_in_every_class(constructed):
    """re_turbulent=None resolves to the former default 1000."""
    _build_each_class()
    _build_each_class(re_turbulent=1000)
    assert len(constructed) == 8
    for element, inputs in constructed:
        assert type(element.re_turbulent) is int and element.re_turbulent == 1000
        assert (inputs["re_lower"], inputs["re_upper"]) == (500.0, 1000.0)
        assert element.viscosity_model == "exponential"
        assert inputs.get("lube_model", 0) == 0


def test_taylor_limits_in_every_class(constructed):
    _build_each_class(re_laminar="taylor")
    assert len(constructed) == 4
    for element, inputs in constructed:
        radius = 0.5 * element.journal_diameter
        re_c = 41.2 * np.sqrt(radius / element.radial_clearance)
        assert element.re_laminar == "taylor"
        assert_allclose(element.re_turbulent, 2.0 * re_c, rtol=1e-14)
        assert_allclose(
            [inputs["re_lower"], inputs["re_upper"]], [re_c, 2.0 * re_c], rtol=1e-14
        )


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(re_laminar="laminar"), "re_laminar must be a number or 'taylor'"),
        (dict(re_laminar="taylor", re_turbulent=900.0), "re_laminar < re_turbulent"),
        (dict(viscosity_model="roelands"), "viscosity_model must be one of"),
    ],
)
def test_lubricant_and_turbulence_options_are_checked(no_solve, overrides, match):
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous", **overrides)
    with pytest.raises(ValueError, match=match):
        FluidFilmBearing(**kwargs)
