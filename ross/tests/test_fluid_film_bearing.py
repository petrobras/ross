import json
import os
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


# Mass-conserving (Elrod-Adams) film in the journal engine.
#
# References, all independent of the mass-conserving film:
# - the Reynolds-model film of the same engine (finite elements, its own
#   perturbation coefficients) where nothing cavitates: the two are two
#   discretisations of the same full-film equation, so they converge
#   together at second order with the mesh, and the damping (same perturbation
#   operator, same empty cavitated set) is the same;
# - the Reynolds model on the flooded two-axial-groove journal, whose film
#   ruptures and reforms in the grooves, where both conditions agree
#   (literature: small differences for flooded journals; measured at most
#   0.3 % on the 40 x 20 mesh);
# - central differences of the film force of ``film_force_at`` (a whole
#   case solved from scratch at each position) for the stiffness;
# - closed forms: the Couette inflow ``U / 2 h`` of a starved inlet.

MC = "mass_conserving"
MC_MESH = dict(total_e_x_film=40, total_e_z_film=20)


def _engine_inputs(case, **changes):
    inputs = json.loads((DATA_DIR / f"{case}.json").read_text())["inputs"]
    inputs.update(changes)
    return inputs


def _assert_film(convergence, tolerance=1e-10):
    """The film solves converged and conserve the mass on every pad."""
    film = convergence["loops"]["cavitation"]
    assert film["converged"], film
    assert max(film["mass_balance"]) < tolerance, film["mass_balance"]


def _block_gap(reference, other):
    """Largest gap of each stiffness block (uu, ue, eu, ee) over its scale."""
    gaps = []
    for rows, cols in (
        (slice(0, 2), slice(0, 2)),
        (slice(0, 2), slice(2, 3)),
        (slice(2, 3), slice(0, 2)),
        (slice(2, 3), slice(2, 3)),
    ):
        ref = np.asarray(reference)[rows, cols]
        gaps.append(
            np.max(np.abs(np.asarray(other)[rows, cols] - ref)) / np.max(np.abs(ref))
        )
    return max(gaps)


def _no_cavitation_gaps(position, mesh):
    """Force, stiffness and damping gaps of the two models, uncavitated pads."""
    from ross.bearings.fluid_film.driver import run_case
    from ross.bearings.fluid_film.sweep import film_force_at

    inputs = _engine_inputs(
        "tilt_5pad_isoviscous",
        equilibrium_type="match_eccentricity",
        xj=position[0],
        yj=position[1],
        total_e_x_film=mesh[0],
        total_e_z_film=mesh[1],
    )
    speed, cb = inputs["frequency"][0], inputs["radial_clearance"]
    reynolds = run_case(**inputs, diagnostics=True, pad_matrices=True)
    tilts = np.asarray(reynolds["tilt_angle"][0])
    held = dict(inputs, _held_tilts=tilts)
    reynolds = run_case(**held, diagnostics=True, pad_matrices=True)
    mass = run_case(**held, cavitation_model=MC, diagnostics=True, pad_matrices=True)
    _assert_film(mass["convergence"][0])
    x, y = position[0] * cb, position[1] * cb
    force_r = film_force_at(inputs, speed, x, y, tilts=tilts)
    force_m = film_force_at(dict(inputs, cavitation_model=MC), speed, x, y, tilts=tilts)
    # no node cavitates under either model
    assert not force_r["cavitated"].any()
    assert not force_m["cavitated"].any()
    assert_allclose(force_m["film_fraction"], 1.0, rtol=0, atol=1e-12)
    load = np.hypot(force_r["fx"], force_r["fy"])
    force_gap = np.hypot(force_m["fx"] - force_r["fx"], force_m["fy"] - force_r["fy"])
    pads_r, pads_m = reynolds["pad_matrices"][0], mass["pad_matrices"][0]
    stiffness_gap = max(
        _block_gap(pads_r.K[k], pads_m.K[k]) for k in range(len(pads_r.K))
    )
    damping_gap = max(
        np.max(np.abs(np.asarray(pads_m.C[k]) - np.asarray(pads_r.C[k])))
        / np.max(np.abs(np.asarray(pads_r.C[k])))
        for k in range(len(pads_r.C))
    )
    return force_gap / load, stiffness_gap, damping_gap


@pytest.mark.parametrize("position", [(0.0, -0.1), (0.05, -0.2), (0.0, -0.3)])
def test_mass_conserving_film_without_cavitation_equals_the_reynolds_film(position):
    """A lightly loaded tilting pad: no node cavitates under either model.

    Reference: the Reynolds-model film and its perturbation coefficients, at
    the same journal position and pad tilts (those of the Reynolds model) on
    the 40 x 20 mesh. The force and the finite-difference stiffness of the
    mass-conserving film are within 1 % (measured 0.31-0.34 %); the damping,
    which both models take from the same perturbation operator with the same
    (empty) cavitated set, is the same to 1e-10.
    """
    force, stiffness, damping = _no_cavitation_gaps(position, (40, 20))
    assert force < 0.01
    assert stiffness < 0.01
    assert damping < 1e-10


def test_mass_conserving_and_reynolds_films_converge_together():
    """The no-cavitation gap is a discretisation error (second order).

    Reference: as above, on the 20 x 10, 40 x 20 and 80 x 40 meshes: the
    force and stiffness gaps fall by about four per refinement (measured
    3.96-4.08).
    """
    gaps = np.array(
        [
            _no_cavitation_gaps((0.05, -0.2), mesh)
            for mesh in ((20, 10), (40, 20), (80, 40))
        ]
    )
    for column in (0, 1):
        ratios = gaps[:-1, column] / gaps[1:, column]
        assert np.all((ratios > 3.5) & (ratios < 4.5)), ratios
    assert np.all(gaps[:, 2] < 1e-10)


@pytest.mark.parametrize("eccentricity", [0.3, 0.5, 0.7, 0.8])
@pytest.mark.parametrize("angle", [0.0, 0.5])
def test_flooded_two_axial_groove_load_capacity(eccentricity, angle):
    """Flooded journal: the mass-conserving load capacity is the Reynolds one.

    Reference: the Reynolds-model force of the same two-axial-groove journal
    at the same position (40 x 20 mesh). The oil enters each land full at
    its leading groove and the film ruptures in the divergent clearance; it
    reforms only in the next groove, where both cavitation conditions give
    the same film, so the load capacity agrees closely (literature: small
    differences for flooded journals). Asserted within 5 %; measured within
    0.3 %. The film fraction is checked against the complementarity:
    ``theta <= 1``, ``p >= p_cav``, and the cavitated nodes are at ``p_cav``
    with ``theta < 1``.
    """
    from ross.bearings.fluid_film.sweep import film_force_at

    inputs = _engine_inputs("fixed_isoviscous", **MC_MESH)
    speed, cb = inputs["frequency"][0], inputs["radial_clearance"]
    x = eccentricity * np.sin(angle) * cb
    y = -eccentricity * np.cos(angle) * cb
    reynolds = film_force_at(inputs, speed, x, y)
    mass = film_force_at(dict(inputs, cavitation_model=MC), speed, x, y)
    assert mass["converged"]
    _assert_film(mass["convergence"])
    load_r = np.hypot(reynolds["fx"], reynolds["fy"])
    load_m = np.hypot(mass["fx"], mass["fy"])
    assert abs(load_m / load_r - 1.0) < 0.05
    theta = mass["film_fraction"]
    pressure = mass["nodal_pressure"] / mass["nodal_pressure"].max()
    cavitated = mass["cavitated"]
    assert cavitated.any() and (theta < 0.99).any()
    assert np.all(theta <= 1.0 + 1e-12)
    assert np.all(pressure >= -1e-9)
    assert_allclose(np.minimum(pressure, 1.0 - theta), 0.0, rtol=0, atol=1e-9)
    assert np.all(pressure[cavitated] <= 1e-9)


def _starved(oil_flow_v, eccentricity=0.8, hot_oil_lambda=0.0):
    from ross.bearings.fluid_film.sweep import film_force_at

    inputs = _engine_inputs(
        "fixed_isoviscous",
        **MC_MESH,
        cavitation_model=MC,
        operating_type="starved_condition_even",
        ambient_pressure_1=0.0,
        ambient_pressure_2=0.0,
        hot_oil_lambda=hot_oil_lambda,
        oil_flow_v=oil_flow_v,
    )
    speed, cb = inputs["frequency"][0], inputs["radial_clearance"]
    return inputs, film_force_at(inputs, speed, 0.0, -eccentricity * cb)


def test_starved_inlet_film_fraction_is_the_supply_over_the_couette_inflow():
    """The inlet film fraction of a starved land from the Couette inflow.

    Reference (closed form, a regression check of the inlet discretisation
    rather than an independent physics reference; the supply balance is
    tested on its own below): without hot-oil carry-over each land gets half
    the supply ``q``. At this supply the film of both lands cavitates at the
    leading-edge pressure, so no pressure flow returns to the supply and the
    inlet draws only the Couette flow ``U / 2 h theta_in``.
    Through the first control-volume faces of the land, with
    ``h = C - y_j sin(phi)`` (journal at ``(0, y_j)``) averaged between the
    leading edge and the next node and the full axial width ``L`` (the free
    nodes and the half-width strips along the axial edges),
    ``theta_in = (q / 2) / (U / 2 h_f L)``.
    """
    q = 3e-4
    inputs, out = _starved(q)
    assert out["converged"]
    _assert_film(out["convergence"])
    theta_in = np.asarray(out["convergence"]["loops"]["cavitation"]["film_fraction_in"])
    cb, radius = inputs["radial_clearance"], 0.5 * inputs["journal_diameter"]
    speed_surface = inputs["frequency"][0] * radius
    length = inputs["pad_axial_length"][0]
    width = length
    yj = -0.8 * cb
    expected = []
    for pivot, arc in zip(inputs["pivot_angle"], inputs["pad_arc"], strict=True):
        leading = pivot - 0.5 * arc
        step = arc / inputs["total_e_x_film"]
        h_face = 0.5 * sum(cb - yj * np.sin(leading + i * step) for i in (0, 1))
        expected.append(0.5 * q / (0.5 * speed_surface * h_face * width))
    assert_allclose(theta_in, expected, rtol=1e-9)


def test_starved_inlet_draws_exactly_the_available_oil():
    """The supply conserves the oil: a starved land draws all of its share.

    Reference (mass conservation at the supply): without hot-oil carry-over
    each land has ``q / 2`` available. A land whose inlet film fraction is
    below 1 draws exactly that net flow through its leading edge (Couette
    inflow less the pressure backflow into the supply); a land that floods
    draws at most ``q / 2`` and spills the rest. At ``q = 5e-4`` the loaded
    land's pressurised zone reaches its leading edge, so its film fraction
    is above the Couette-only value ``(q / 2) / C``.
    """
    for q in (3e-4, 5e-4, 6e-4, 8e-4):
        _, out = _starved(q)
        assert out["converged"]
        _assert_film(out["convergence"])
        record = out["convergence"]["loops"]["cavitation"]
        fraction = np.asarray(record["film_fraction_in"])
        draw = np.asarray(record["inlet_flow"])
        starved = fraction < 1.0
        assert_allclose(draw[starved], 0.5 * q, rtol=1e-5)
        assert np.all(draw[~starved] <= 0.5 * q * (1 + 1e-5))
        if q == 5e-4:
            assert np.all(starved)
    # The loaded land (journal at -0.8 C_b, the lower land) floods at 6e-4.
    _, out = _starved(6e-4)
    assert out["convergence"]["loops"]["cavitation"]["film_fraction_in"][1] == 1.0


def test_starved_inlet_load_capacity_falls_with_the_inlet_film_fraction():
    """Starvation reduces the load capacity.

    Reference (physics trend): at a fixed eccentricity (0.8) the load grows
    with the inlet film fraction of the loaded land until its inlet floods
    (here between ``q = 5e-4`` and ``6e-4``); from then on the load is the
    flooded load, whatever the supply (the unloaded land carries no
    pressure, so its starvation does not change the force). A supply larger
    than both inlets can draw gives the flooded film (same boundary
    pressures, inlet film fraction 1) to 1e-10. Without the mass-conserving
    film there is no inlet film fraction: the Reynolds model starves the
    film by moving its onset instead.
    """
    from ross.bearings.fluid_film.sweep import film_force_at

    supplies = (2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 8e-4)
    loads, fractions = [], []
    for q in supplies:
        _, out = _starved(q)
        assert out["converged"]
        _assert_film(out["convergence"])
        fractions.append(out["convergence"]["loops"]["cavitation"]["film_fraction_in"])
        loads.append(np.hypot(out["fx"], out["fy"]))
    fractions, loads = np.array(fractions), np.array(loads)
    assert np.all(np.diff(fractions, axis=0) >= 0)
    assert np.all(np.diff(fractions[:, 0]) > 0)
    starved = fractions[:, 1] < 1.0
    assert list(starved) == [True, True, True, True, False, False]
    assert np.all(np.diff(loads[starved]) > 0)

    inputs, full = _starved(2e-3)
    assert full["convergence"]["loops"]["cavitation"]["film_fraction_in"] == [1.0, 1.0]
    flooded_inputs = dict(inputs, operating_type="regular_flooded")
    cb = inputs["radial_clearance"]
    flooded = film_force_at(flooded_inputs, inputs["frequency"][0], 0.0, -0.8 * cb)
    assert_allclose(
        [full["fx"], full["fy"]], [flooded["fx"], flooded["fy"]], rtol=1e-10, atol=0
    )
    flooded_load = np.hypot(flooded["fx"], flooded["fy"])
    assert np.all(loads[starved] < flooded_load)
    assert_allclose(loads[~starved], flooded_load, rtol=1e-6)


def test_starved_inlet_with_hot_oil_carry_over():
    """The carry-over makes the inlet film fraction a fixed point; it converges.

    Reference: the available oil of each land, ``q / 2`` plus ``lambda``
    times the exit flow of the land before it, is at least the supply share,
    so the inlet film fraction is at least that of the case without
    carry-over and the load at least as large.
    """
    _, cold = _starved(3e-4)
    _, hot = _starved(3e-4, hot_oil_lambda=0.8)
    assert hot["converged"]
    _assert_film(hot["convergence"])
    record = hot["convergence"]["loops"]["film_onset"]
    assert record["converged"] and record["residual"] < 1e-6
    fraction_cold = cold["convergence"]["loops"]["cavitation"]["film_fraction_in"]
    fraction_hot = hot["convergence"]["loops"]["cavitation"]["film_fraction_in"]
    assert np.all(np.asarray(fraction_hot) > np.asarray(fraction_cold))
    assert np.hypot(hot["fx"], hot["fy"]) > np.hypot(cold["fx"], cold["fy"])


def _starved_equilibrium(
    oil_flow_l_min,
    mesh=(20, 10),
    operating_type="starved_condition_even",
    lam=0.8,
    model=MC,
    thermal_type=None,
):
    """Equilibrium of the starved two-axial-groove journal.

    The isoviscous journal of ``fluid_film_bearing_example`` on a
    ``mesh`` (circumferential x axial elements) mesh, supplied with
    ``oil_flow_l_min`` and the default carry-over ``hot_oil_lambda=0.8``.
    """
    kwargs = dict(
        n=0,
        speed=Q_([900], "RPM"),
        journal_diameter=Q_(15.748, "in"),
        radial_clearance=Q_(0.00766, "in"),
        pad_thickness=Q_(5.89034, "in"),
        pivot_angle=Q_([90, 270], "deg"),
        pad_arc=Q_([176, 176], "deg"),
        pad_axial_length=Q_([10.36, 10.36], "in"),
        preload=[0, 0],
        offset=[0.5, 0.5],
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(50, "degC"),
        oil_flow_v=Q_(oil_flow_l_min, "l/min"),
        weight=Q_(25361.8, "lbf"),
        thermal_type=thermal_type,
        total_ex_film=mesh[0],
        total_ez_film=mesh[1],
        operating_type=operating_type,
        cavitation_model=model,
        hot_oil_lambda=lam,
    )
    return FluidFilmBearing(**kwargs)


def _assert_starved_balance(bearing):
    """The case converged, with the oil and load balances of a starved film."""
    weight = Q_(25361.8, "lbf").to("N").m
    summary = bearing.results.convergence[0]
    assert summary["converged"], summary["message"]
    _assert_film(summary)
    loops = summary["loops"]
    assert loops["film_onset"]["residual"] < 1e-6
    assert loops["newton"]["residual"] < 2e-6
    fraction = np.asarray(loops["cavitation"]["film_fraction_in"])
    assert np.all((fraction > 0.0) & (fraction < 1.0))
    operating = bearing.operating_results
    assert_allclose(
        np.sum(loops["cavitation"]["inlet_flow"]),
        operating["q_supply"][0] + operating["q_carryover"][0],
        rtol=1e-5,
    )
    assert_allclose(np.sum(operating["pad_load"][0]), weight, rtol=1e-5)
    _assert_oil_balance(bearing)


def _assert_oil_balance(bearing, starved=True):
    """The reported flows of a mass-conserving case balance the oil.

    Each pad draws ``q_in`` at its inlet, which leaves through the side edges
    (``q_sides``) or the trailing edge (``q_out``), of which ``hot_oil_lambda``
    is carried to the next pad and the rest drains. When every land is
    starved it draws all the oil available to it, so the supply equals the
    side leakage plus the trailing-edge flow that is not carried over.
    Tolerances: the mass balance of the film solves (1e-10) per pad, the
    inlet film-fraction tolerance (1e-6) for the supply.
    """
    out = bearing.results.outputs[0]
    q_in, q_out, q_sides = (
        np.asarray(out[key], dtype=float).ravel()
        for key in ("q_in", "q_out", "q_sides")
    )
    scale = np.max(q_in)
    assert_allclose(q_in - q_out - q_sides, 0.0, rtol=0, atol=1e-10 * scale)
    assert np.all(q_sides >= -1e-12 * scale)
    film = bearing.results.convergence[0]["loops"]["cavitation"]
    assert_allclose(q_in, film["inlet_flow"], rtol=1e-12)
    operating = bearing.operating_results
    lam = bearing.hot_oil_lambda
    assert_allclose(operating["q_side"][0], np.sum(q_sides), rtol=1e-12)
    assert_allclose(operating["q_carryover"][0], lam * np.sum(q_out), rtol=1e-12)
    drained = operating["q_side"][0] + (1.0 - lam) * np.sum(q_out)
    if starved:
        assert_allclose(drained, operating["q_supply"][0], rtol=1e-5)
    else:
        # A flooded pad draws less than it is offered and spills the rest.
        assert drained < operating["q_supply"][0]


def test_mass_conserving_flows_converge_to_the_full_film_flows():
    """Where nothing cavitates the two models report the same flows.

    Reference: the Reynolds-model flows of the same full film (inlet flow
    and side leakage from the velocity profiles integrated across the film;
    they move by 1e-3 between the 20 x 10 and 160 x 80 meshes). The
    mass-conserving flows come from the finite-volume fluxes, including the
    half-width strips of film along the axial edges, and converge to them at
    second order: measured gaps 6.6e-4 and 2.1e-4 of the inlet flow, 8.6e-3
    and 2.5e-3 of the side leakage, on the 20 x 10 and 40 x 20 meshes.
    """
    from ross.bearings.fluid_film.driver import run_case

    gaps = []
    for mesh in ((20, 10), (40, 20)):
        inputs = _engine_inputs(
            "tilt_5pad_isoviscous",
            equilibrium_type="match_eccentricity",
            xj=0.0,
            yj=-0.1,
            total_e_x_film=mesh[0],
            total_e_z_film=mesh[1],
        )
        tilts = np.asarray(run_case(**inputs)["tilt_angle"][0])
        held = dict(inputs, _held_tilts=tilts)
        reynolds = run_case(**held, diagnostics=True)
        mass = run_case(**held, cavitation_model=MC, diagnostics=True)
        row = []
        for key in ("q_in", "q_sides"):
            ref = np.asarray(reynolds[key][0], dtype=float)
            row.append(np.max(np.abs(np.asarray(mass[key][0]) - ref)) / np.max(ref))
        gaps.append(row)
    gaps = np.array(gaps)
    assert np.all(gaps[1] < [1e-3, 1e-2]), gaps
    assert np.all(gaps[0] / gaps[1] > 2.5), gaps


@pytest.mark.parametrize("hot_oil_lambda", [0.8, 0.0])
def test_flooded_mass_conserving_oil_balance(hot_oil_lambda):
    """Flooded two-axial-groove journal: the flows close the oil balance.

    References: the oil balance of :func:`_assert_oil_balance` (each pad's
    draw leaves through its sides and trailing edge, and a flooded pad
    spills part of what it is offered), and the side leakage of the
    Reynolds model on the same journal, whose film ruptures and reforms in
    the grooves only (measured 0.1 % apart on the 40 x 20 mesh, 1.7 % on the
    20 x 10 mesh).
    """
    reynolds = _starved_equilibrium(
        100.0,
        mesh=(40, 20),
        operating_type="regular_flooded",
        lam=hot_oil_lambda,
        model="reynolds",
    )
    mass = _starved_equilibrium(
        100.0, mesh=(40, 20), operating_type="regular_flooded", lam=hot_oil_lambda
    )
    assert mass.results.convergence[0]["converged"]
    _assert_oil_balance(mass, starved=False)
    assert_allclose(
        mass.operating_results["q_side"][0],
        reynolds.operating_results["q_side"][0],
        rtol=1e-2,
    )


def test_strongly_starved_mass_conserving_equilibrium():
    """Strongly starved mass-conserving journals reach their equilibrium.

    At 4 l/min the starved film carries no load at the first trial positions,
    where the Jacobian is singular: the journal has to step along the load.
    The inlet-fraction loop needs the force balance solved well inside the
    fraction tolerance.

    References (independent of the solver's output): the oil balance of the
    supply, each land drawing all the oil available to it, so the draws of
    the two lands add up to the supply plus the carry-over; the load balance,
    the film carrying the weight; and the physics of starvation, the
    eccentricity falling as the supply grows and staying above that of the
    flooded journal.
    """
    eccentricities = []
    for oil_flow in (4.0, 6.0, 8.0):
        bearing = _starved_equilibrium(oil_flow)
        _assert_starved_balance(bearing)
        eccentricities.append(bearing.operating_results["eccentricity"][0])
    assert np.all(np.diff(eccentricities) < 0.0)
    flooded = _starved_equilibrium(200.0)
    assert flooded.results.convergence[0]["converged"]
    assert eccentricities[-1] > flooded.operating_results["eccentricity"][0]


@pytest.mark.parametrize("oil_flow", [4.0, 6.0])
def test_strongly_starved_mass_conserving_equilibrium_fine_mesh(oil_flow):
    """The strongly starved cases converge on the 40 x 20 mesh.

    On this mesh 4 l/min also needs the step along the load, and 6 l/min
    needs the tight force balance inside the inlet-fraction loop, without
    which that loop cycles. References as in
    :func:`test_strongly_starved_mass_conserving_equilibrium`. The 4 l/min
    case takes many inlet-fraction passes; it runs with
    ``ROSS_FLUID_FILM_SLOW`` set.
    """
    if oil_flow == 4.0 and not os.environ.get("ROSS_FLUID_FILM_SLOW"):
        pytest.skip("slow case; set ROSS_FLUID_FILM_SLOW=1 to run")
    _assert_starved_balance(_starved_equilibrium(oil_flow, mesh=(40, 20)))


def test_mass_conserving_film_through_a_pad_is_not_solved():
    """A journal position through a pad surface does not raise.

    The Newton search of a starved case can step the journal through a pad
    surface; the finite-volume grid then raised ``ValueError("h must be
    positive")``, an uncaught exception instead of a failed case. Reference:
    there is no film on the crossed pad, so it carries no pressure and no
    force, while the other pad is solved as usual.
    """
    from ross.bearings.fluid_film.sweep import film_force_at

    inputs = _engine_inputs("fixed_isoviscous", **MC_MESH, cavitation_model=MC)
    speed, cb = inputs["frequency"][0], inputs["radial_clearance"]
    out = film_force_at(inputs, speed, 0.0, -1.02 * cb)
    crossed = int(np.argmin(out["h_min"]))
    assert out["h_min"][crossed] < 0.0
    assert np.all(out["nodal_pressure"][crossed] == 0.0)
    assert out["fx_pad"][crossed] == 0.0 and out["fy_pad"][crossed] == 0.0
    other = 1 - crossed
    assert out["h_min"][other] > 0.0
    assert np.all(np.isfinite(out["nodal_pressure"][other]))


def _fd_pad_blocks(inputs, x0, y0, tilts, step):
    """Per-pad ``-dF/du`` of ``film_force_at`` (tilts held), on ``[x, y, tilt]``."""
    from ross.bearings.fluid_film.sweep import film_force_at

    speed = inputs["frequency"][0]
    n_pads = len(inputs["pivot_angle"])
    blocks = np.zeros((n_pads, 3, 3))

    def loads(x, y, held):
        out = film_force_at(inputs, speed, x, y, tilts=held)
        return np.array([out["fx_pad"], out["fy_pad"], out["moment_pad"]])

    for column, (dx, dy) in enumerate(((step, 0.0), (0.0, step))):
        slope = -(loads(x0 + dx, y0 + dy, tilts) - loads(x0 - dx, y0 - dy, tilts)) / (
            2 * step
        )
        blocks[:, :, column] = slope.T
    if tilts is not None:
        tilt_step = step / (0.5 * inputs["journal_diameter"])
        for k in range(n_pads):
            plus, minus = tilts.copy(), tilts.copy()
            plus[k] += tilt_step
            minus[k] -= tilt_step
            slope = -(loads(x0, y0, plus) - loads(x0, y0, minus)) / (2 * tilt_step)
            blocks[k, :, 2] = slope[:, k]
    return blocks


@pytest.mark.parametrize(
    "case, position",
    [
        ("fixed_isoviscous", (0.3, -0.6)),
        ("fixed_isoviscous", (-0.2, -0.75)),
        ("tilt_5pad_isoviscous", (0.0, -0.5)),
    ],
)
def test_mass_conserving_stiffness_is_the_film_force_derivative(case, position):
    """The stiffness blocks are central differences of the cavitated film force.

    Reference: central differences (step ``1e-4 C_b``, tilt step
    ``1e-4 C_b / R``) of the per-pad film force and pivot moment of
    ``film_force_at`` with the mass-conserving film, each a case solved from
    scratch at a held journal position and held tilts; measured agreement
    2e-12 of the largest entry. The films cavitate; the Reynolds-model
    perturbation stiffness is not this derivative there (it differs by 1 to
    40 % of the largest entry on these films). Pads cavitated everywhere
    have no derivative and zero blocks.
    """
    from ross.bearings.fluid_film.driver import run_case

    inputs = _engine_inputs(
        case,
        cavitation_model=MC,
        equilibrium_type="match_eccentricity",
        xj=position[0],
        yj=position[1],
    )
    out = run_case(**inputs, diagnostics=True, pad_matrices=True)
    assert out["convergence"][0]["converged"]
    _assert_film(out["convergence"][0])
    tilting = case.startswith("tilt_")
    tilts = np.asarray(out["tilt_angle"][0]) if tilting else None
    cb = inputs["radial_clearance"]
    fd = _fd_pad_blocks(inputs, position[0] * cb, position[1] * cb, tilts, 1e-4 * cb)
    pads = out["pad_matrices"][0]
    loaded = 0
    for k in range(len(inputs["pivot_angle"])):
        engine = np.asarray(pads.K[k])
        size = 3 if tilting else 2
        if not np.any(engine):
            continue
        loaded += 1
        scale = np.max(np.abs(engine))
        assert_allclose(
            engine[:size, :size], fd[k][:size, :size], rtol=0, atol=1e-9 * scale
        )
    assert loaded >= 1
    if not tilting:
        k_engine = np.array(
            [[out["kxx"][0], out["kxy"][0]], [out["kyx"][0], out["kyy"][0]]]
        )
        scale = np.max(np.abs(k_engine))
        assert_allclose(k_engine, fd.sum(axis=0)[:2, :2], rtol=0, atol=1e-9 * scale)


def test_every_film_solve_is_charged_to_the_budget(monkeypatch):
    """Each mass-conserving film solve counts as one solve of the budget.

    Reference: a count of the calls of the film solver (a spy, the solver
    itself runs unchanged) on a load-matched tilting pad, including the
    finite-difference solves of the stiffness; and a budget smaller than the
    equilibrium needs stops the case with reason ``"budget"``.
    """
    import ross.bearings.film.cavitation as cavitation
    from ross.bearings.fluid_film.driver import run_case

    calls = []
    original = cavitation.solve_elrod_adams

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cavitation, "solve_elrod_adams", spy)
    inputs = _engine_inputs("tilt_5pad_isoviscous", cavitation_model=MC)
    out = run_case(**inputs, diagnostics=True)
    summary = out["convergence"][0]
    assert summary["converged"]
    assert summary["reynolds_solves"] == len(calls) > 0
    assert summary["loops"]["cavitation"]["solves"] == len(calls)

    calls.clear()
    stopped = run_case(**inputs, diagnostics=True, max_reynolds_solves=20)
    summary = stopped["convergence"][0]
    assert not summary["converged"] and summary["reason"] == "budget"
    assert summary["reynolds_solves"] == len(calls)


def test_mass_conserving_thermal_tilting_pad_equilibrium():
    """An adiabatic tilting pad at its load: converged and mass-conserving.

    Reference: the applied load (force balance of the converged equilibrium,
    ``|F_net| / |F_ext| < 2e-3``) and the Reynolds model of the same case,
    whose eccentricity it matches within 2 % (the loaded pads hardly
    cavitate).
    """
    from ross.bearings.fluid_film.driver import run_case

    inputs = _engine_inputs("tilt_5pad_isoviscous", thermal_type="adiabatic")
    reynolds = run_case(**inputs)
    mass = run_case(**inputs, cavitation_model=MC, diagnostics=True)
    assert mass["convergence"][0]["converged"]
    _assert_film(mass["convergence"][0])
    load = inputs["weight"]
    assert_allclose(np.hypot(mass["fx_hydro"][0], mass["fy_hydro"][0]), load, rtol=2e-3)
    assert_allclose(mass["eccentricity"][0], reynolds["eccentricity"][0], rtol=0.02)


@pytest.mark.parametrize(
    "overrides, error, match",
    [
        (dict(thermal_type="full"), NotImplementedError, "thermal_type"),
        (dict(deform_type="pad_mechanical"), NotImplementedError, "deform_type"),
        (dict(operating_type="axial_flow"), NotImplementedError, "operating_type"),
        (
            dict(operating_type="high_ambient_pressure"),
            NotImplementedError,
            "operating_type",
        ),
        (
            dict(operating_type="starved_condition_uneven"),
            NotImplementedError,
            "operating_type",
        ),
        (
            dict(operating_type="oil_ring_lubricated"),
            NotImplementedError,
            "operating_type",
        ),
        (dict(bearing_type="pressure_dam"), NotImplementedError, "pressure dam"),
        (dict(cavitation_pressure=1e3), ValueError, "cavitation_pressure"),
        (
            dict(operating_type="starved_condition_even", ambient_pressure_1=-1e3),
            ValueError,
            "cavitation_pressure",
        ),
        (dict(cavitation_model="elrod"), ValueError, "cavitation_model must be one of"),
    ],
)
def test_mass_conserving_limits_raise_before_solving(no_solve, overrides, error, match):
    """Combinations outside the limits of the mass-conserving film raise.

    Both the element (before its first solve) and the engine entry point
    check them.
    """
    from ross.bearings.fluid_film.driver import run_case

    overrides = dict({"cavitation_model": MC, "thermal_type": None}, **overrides)
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous", **overrides)
    with pytest.raises(error, match=match):
        FluidFilmBearing(**kwargs)
    with pytest.raises(error, match=match):
        run_case(**_engine_inputs("fixed_isoviscous", **overrides))


def test_mass_conserving_film_does_not_model_the_360_deg_pad():
    """The short-bearing oil seal (a first pad of 2 pi) is not supported."""
    from ross.bearings.fluid_film.pressure import check_cavitation_model

    with pytest.raises(NotImplementedError, match="360 deg"):
        check_cavitation_model(
            MC, None, None, "regular_flooded", "fixed_geometry", [2 * np.pi], 0.0
        )
    check_cavitation_model(
        MC, None, None, "regular_flooded", "fixed_geometry", [np.radians(359.9)], 0.0
    )


def test_mass_conserving_bearing_element():
    """The element with the mass-conserving film, over three speeds.

    Reference: the Reynolds-model element of the same flooded two-axial-groove
    journal (the coefficients of the two cavitation conditions differ by a
    few percent on this cavitated film: the perturbation stiffness of the
    Reynolds model is not the derivative of its force, and the
    cavitated sets differ), and the convergence records of each speed (the
    second and third start from the previous speed).
    """
    speeds = [94.24777960769379, 150.0, 200.0]
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous", speed=speeds)
    reynolds = FluidFilmBearing(**kwargs)
    mass = FluidFilmBearing(**kwargs, cavitation_model=MC)
    assert mass.cavitation_model == MC
    for summary in mass.results.convergence:
        assert summary["converged"]
        _assert_film(summary)
    for name in ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"):
        expected = np.asarray(getattr(reynolds, name), dtype=float)
        actual = np.asarray(getattr(mass, name), dtype=float)
        assert np.all(np.isfinite(actual)), name
        assert np.max(np.abs(actual - expected)) < 0.2 * np.max(np.abs(expected)), name


@pytest.mark.filterwarnings("ignore:operating_type='regular_flooded':UserWarning")
def test_film_fraction_field_and_plot():
    """The element keeps the film fraction of the mass-conserving film.

    References (independent of the solver's output): the complementarity of
    the Elrod-Adams film (full film, ``theta = 1``, wherever the pressure is
    above the cavitation pressure 0; the pressure at 0 wherever
    ``theta < 1``), and the mass balance of the striated film in a cavitated
    zone, where the pressure is uniform and the oil is carried by the
    Couette flow only. On the unloaded upper pad of this flooded
    two-axial-groove journal every interior node is cavitated, so along each
    axial station the upwind face flux ``(h_i + h_(i+1)) / 2 theta_i`` equals
    the flux that enters the leading edge full, ``(h_0 + h_1) / 2``: the film
    fraction follows from the film thickness alone. The Reynolds model has
    no film fraction.
    """
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    reynolds = FluidFilmBearing(**kwargs)
    assert reynolds.results.film_fraction_fields is None
    with pytest.raises(ValueError, match="mass-conserving"):
        reynolds.plot_film_fraction_2d()

    bearing = FluidFilmBearing(**kwargs, cavitation_model=MC)
    results = bearing.results
    pressure = results.pressure_fields[0]
    fraction = results.film_fraction_fields[0]
    thickness = results.film_thickness_fields[0]
    assert fraction.shape == pressure.shape
    assert np.all((fraction > 0.0) & (fraction <= 1.0))

    cavitated = fraction < 1.0 - 1e-9
    assert np.count_nonzero(cavitated) > 0
    pressure_scale = np.max(pressure)
    assert np.max(np.abs(pressure[cavitated])) < 1e-9 * pressure_scale
    assert np.all(fraction[pressure > 1e-9 * pressure_scale] == 1.0)

    upper = 0
    assert np.all(cavitated[upper, 1:-1, 1:-1])
    h = thickness[upper]
    face = 0.5 * (h[1:, 1:-1] + h[:-1, 1:-1])
    inflow = np.broadcast_to(face[:1], face.shape)
    assert_allclose(face * fraction[upper, :-1, 1:-1], inflow, rtol=1e-9)

    fig = bearing.plot_film_fraction_2d()
    middle = fraction.shape[2] // 2
    assert len(fig.data) == fraction.shape[0]
    for pad, trace in enumerate(fig.data):
        assert_allclose(trace.y, fraction[pad, :, middle], rtol=0, atol=0)


def test_rigid_rotor_stability_with_singular_damping():
    """A singular damping matrix gives no threshold instead of raising.

    Reference: Lund's single-mass criterion divides by the determinant of
    the damping matrix; with ``cxx cyy = cxy cyx`` there is no finite
    threshold, which the function reports as zero.
    """
    from ross.bearings.fluid_film.driver import _rigid_rotor_stability

    g = dict(f_total=1e4, cb=1e-4, weight_case=1e4, speed_rpm=3000.0)
    reduced = (1e8, 2e7, -2e7, 1e8, 1e6, 1e6, 1e6, 1e6)
    threshold, whirl, *_ = _rigid_rotor_stability(g, reduced)
    assert threshold == 0.0 and whirl == 0.0
