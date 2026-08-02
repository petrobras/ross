import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.tilting_pad import (
    TiltingPad,
    tilting_pad_adiabatic_example,
)
from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture
from ross.units import Q_


@pytest.fixture(scope="module")
def tilting_pad():
    return tilting_pad_adiabatic_example()


def test_geometry_translation(tilting_pad):
    assert type(tilting_pad) is TiltingPad
    assert tilting_pad.n_pads == 5
    assert_allclose(tilting_pad.pivot_angle, np.radians([18, 90, 162, 234, 306]))
    assert_allclose(tilting_pad.pad_arc, np.radians([60] * 5))
    assert_allclose(tilting_pad.preload, [0.5] * 5)
    assert tilting_pad.bearing_type == "conventional_tilting_pad"
    assert tilting_pad.thermal_type == "adiabatic"
    assert_allclose(tilting_pad.hot_oil_lambda, 0.8)
    assert_allclose(tilting_pad.pad_conductivity, 116.0)
    assert_allclose(tilting_pad.edges_convection, 1500.0)
    assert tilting_pad.total_ex_film == 20
    assert tilting_pad.total_ez_film == 10
    assert tilting_pad.total_ey_pad == 10


def test_coefficients_solved(tilting_pad):
    for name in ("kxx", "kyy", "cxx", "cyy"):
        assert float(getattr(tilting_pad, name)[0]) > 0
    out = tilting_pad._results.outputs[0]
    assert len(out["tilt_angle"][0]) == 5


def test_matches_solver_fixture():
    """TiltingPad reproduces the raw 5-pad tilting-pad solver fixture."""
    kwargs, outputs = bearing_kwargs_from_fixture("tilt_5pad_isoviscous")
    bearing = TiltingPad(
        n=kwargs["n"],
        frequency=kwargs["frequency"],
        journal_diameter=kwargs["journal_diameter"],
        radial_clearance=kwargs["radial_clearance"],
        pad_thickness=kwargs["pad_thickness"],
        pivot_angle=kwargs["pivot_angle"],
        pad_arc=kwargs["pad_arc"],
        pad_axial_length=kwargs["pad_axial_length"],
        pre_load=kwargs["preload"],
        offset=kwargs["offset"],
        lubricant=kwargs["lubricant"],
        oil_supply_temperature=kwargs["oil_supply_temperature"],
        oil_flow_v=kwargs["oil_flow_v"],
        thermal_type=kwargs["thermal_type"],
        equilibrium_type=kwargs["equilibrium_type"],
        xj=kwargs["initial_position"][0],
        yj=kwargs["initial_position"][1],
        load=[kwargs["fxs_load"], kwargs["fys_load"]],
        hot_oil_carry_over=kwargs["hot_oil_lambda"],
        journal_temperature=kwargs["journal_temperature"],
        nx=kwargs["total_ex_film"],
        nz=kwargs["total_ez_film"],
        nr_pad=kwargs["total_ey_pad"],
        total_ey_film=kwargs["total_ey_film"],
        weight=kwargs["weight"],
        probes=kwargs["probes"],
        pad_E=kwargs["pad_E"],
        pad_poisson=kwargs["pad_poisson"],
        k_pad=kwargs["pad_conductivity"],
        pad_expansion=kwargs["pad_expansion"],
        pad_density=kwargs["pad_density"],
        journal_expansion=kwargs["journal_expansion"],
        shell_expansion=kwargs["shell_expansion"],
        pad_convection=kwargs["pad_convection"],
        h_edge=kwargs["edges_convection"],
        environment_temperature=kwargs["environment_temperature"],
        environment_convection=kwargs["environment_convection"],
        reference_temperature=kwargs["reference_temperature"],
        excitation_ratio=kwargs["excitation_ratio"],
        starvation_number=kwargs["starvation_number"],
        relax_pressure=kwargs["relax_pressure"],
        relax_t=kwargs["relax_temperature"],
        relax_deformation=kwargs["relax_deformation"],
        re_laminar=kwargs["re_laminar"],
        re_turbulent=kwargs["re_turbulent"],
    )
    for name in ("kxx", "kyy", "cxx", "cyy"):
        assert_allclose(
            np.asarray(getattr(bearing, name), dtype=float),
            np.asarray(outputs[name], dtype=float),
            rtol=1e-8,
            err_msg=f"{name} differs from the solver fixture",
        )


def test_eccentricity_attitude_initial_position():
    bearing = TiltingPad(
        n=1,
        frequency=Q_([3000], "RPM"),
        equilibrium_type="match_eccentricity",
        thermal_type=None,
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
        pad_arc=Q_([60] * 5, "deg"),
        pad_axial_length=[50.8e-3] * 5,
        pre_load=[0.5] * 5,
        offset=[0.5] * 5,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(10, "l/min"),
        eccentricity=0.35,
        attitude_angle=Q_(287.5, "deg"),
        nx=20,
        nz=10,
        nr_pad=10,
        total_ey_film=10,
    )
    expected = (
        0.35 * np.cos(np.radians(287.5)),
        0.35 * np.sin(np.radians(287.5)),
    )
    assert_allclose(bearing.initial_position, expected)
    out = bearing._results.outputs[0]
    assert_allclose(out["eccentricity"][0], 0.35, rtol=1e-6)


def test_deprecations_and_restrictions():
    kwargs = dict(
        n=1,
        frequency=Q_([3000], "RPM"),
        equilibrium_type="match_load",
        thermal_type=None,
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
        pad_arc=Q_([60] * 5, "deg"),
        pad_axial_length=[50.8e-3] * 5,
        pre_load=[0.5] * 5,
        offset=[0.5] * 5,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(10, "l/min"),
        load=[8.8405e02, -2.6704e03],
        nx=20,
        nz=10,
        nr_pad=10,
        total_ey_film=10,
    )

    with pytest.warns(DeprecationWarning, match="solver_options"):
        TiltingPad(solver_options={"xtol": 1e-2}, **kwargs)
    with pytest.warns(DeprecationWarning, match="initial_pads_angles"):
        TiltingPad(initial_pads_angles=[1e-3] * 5, **kwargs)
    with pytest.warns(UserWarning, match="interpreted"):
        TiltingPad(journal_temperature=25.0, **kwargs)

    with pytest.raises(ValueError, match="FixedGeometryBearing"):
        TiltingPad(bearing_type="fixed_geometry", **kwargs)

    no_flow = dict(kwargs)
    no_flow.pop("oil_flow_v")
    with pytest.warns(UserWarning, match="ample flooded supply"):
        TiltingPad(**no_flow)


def test_pivot_flexibility_runs():
    bearing = TiltingPad(
        n=1,
        frequency=Q_([3000], "RPM"),
        equilibrium_type="match_load",
        thermal_type=None,
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
        pad_arc=Q_([60] * 5, "deg"),
        pad_axial_length=[50.8e-3] * 5,
        pre_load=[0.5] * 5,
        offset=[0.5] * 5,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(10, "l/min"),
        load=[8.8405e02, -2.6704e03],
        deform_type="pad_pivot_mechanical",
        pivot_type="user_specified_stiffness",
        pivot_stiffness=5e8,
        nx=20,
        nz=10,
        nr_pad=10,
        total_ey_film=10,
    )
    out = bearing._results.outputs[0]
    assert_allclose(out["k_pivot"][0], [5e8] * 5)
    assert max(out["deform_pivot"][0]) > 0.0
    assert float(bearing.kxx[0]) > 0
