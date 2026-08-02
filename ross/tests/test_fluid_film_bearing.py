import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from plotly import graph_objects as go

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
    probes = list(zip(inp["probe_pad_number"], inp["probe_theta"], inp["r_location"]))
    kwargs = dict(
        n=0,
        frequency=inp["frequency"],
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
    frequency = bearing.frequency[0]
    stiffness, damping = bearing.coefficients(frequency)
    assert_allclose(stiffness[0], outputs["kxx"][0], rtol=1e-8)
    assert_allclose(damping[3], outputs["cyy"][0], rtol=1e-8)


def test_plots(fixture_bearing):
    bearing, _ = fixture_bearing
    for method in (
        "plot_pressure_2d",
        "plot_pressure_3d",
        "plot_temperature_2d",
        "plot_temperature_3d",
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
        "temperature_3d",
    }


def test_multi_speed_and_parallel():
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    kwargs["frequency"] = [80.0, 110.0]
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
    assert_allclose(loaded.frequency, bearing.frequency)


def test_example_with_pint_units():
    bearing = fluid_film_bearing_example()
    assert bearing.n_pads == 2
    assert bearing.journal_diameter == pytest.approx(Q_(15.748, "in").to("m").m)
    assert float(bearing.kxx[0]) > 1e8
