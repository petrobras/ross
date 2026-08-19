import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.fixed_geometry import FixedGeometryBearing
from ross.bearings.plain_journal import PlainJournal
from ross.units import Q_


@pytest.fixture(scope="module")
def plain_journal():
    return PlainJournal(
        n=3,
        pad_axial_length=Q_(10.3600055944, "in"),
        journal_diameter=0.4,
        radial_clearance=1.95e-4,
        total_ex_film=20,
        total_ez_film=10,
        n_pads=2,
        pad_arc=Q_(176, "deg"),
        preload=0,
        oil_supply_temperature=Q_(50, "degC"),
        frequency=Q_([900], "RPM"),
        fxs_load=0,
        fys_load=-112814.91,
        lubricant="ISOVG32",
        oil_flow_v=Q_(37.86, "l/min"),
        thermal_type=None,
        total_ey_film=10,
        total_ey_pad=10,
    )


def test_geometry_translation(plain_journal):
    assert type(plain_journal) is PlainJournal
    assert plain_journal.n_pads == 2
    assert_allclose(plain_journal.pivot_angle, np.radians([90, 270]))
    assert_allclose(plain_journal.pad_arc, np.radians([176, 176]))
    assert_allclose(plain_journal.pad_axial_length, [0.263144] * 2, rtol=1e-6)
    assert_allclose(plain_journal.journal_diameter, 0.4)
    assert_allclose(plain_journal.offset, [0.5, 0.5])
    assert plain_journal.operating_type == "regular_flooded"
    assert plain_journal.thermal_type is None
    assert_allclose(plain_journal.oil_supply_temperature, 323.15)
    assert plain_journal.total_ex_film == 20
    assert plain_journal.total_ez_film == 10


def test_matches_explicit_fixed_geometry(plain_journal):
    """PlainJournal is pure translation over FixedGeometryBearing."""
    explicit = FixedGeometryBearing(
        n=3,
        frequency=Q_([900], "RPM"),
        journal_diameter=0.4,
        radial_clearance=1.95e-4,
        pad_thickness=0.1,
        pivot_angle=Q_([90, 270], "deg"),
        pad_arc=Q_([176, 176], "deg"),
        pad_axial_length=Q_([10.3600055944, 10.3600055944], "in"),
        preload=[0, 0],
        offset=[0.5, 0.5],
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(50, "degC"),
        oil_flow_v=Q_(37.86, "l/min"),
        fys_load=-112814.91,
        initial_position=(0.1, -0.1),
        thermal_type=None,
        total_ex_film=20,
        total_ey_film=10,
        total_ez_film=10,
        total_ey_pad=10,
    )
    for name in ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"):
        assert_allclose(
            np.asarray(getattr(plain_journal, name), dtype=float),
            np.asarray(getattr(explicit, name), dtype=float),
            rtol=1e-6,
            err_msg=f"{name} differs from the explicit construction",
        )


def test_equilibrium_pos(plain_journal):
    eccentricity, attitude = plain_journal.equilibrium_pos
    assert 0.0 < eccentricity < 1.0
    assert abs(attitude) < np.pi


def test_plain_radians_pad_arc(plain_journal):
    plain_arc = PlainJournal(
        n=3,
        pad_axial_length=Q_(10.3600055944, "in"),
        journal_diameter=0.4,
        radial_clearance=1.95e-4,
        total_ex_film=20,
        total_ez_film=10,
        n_pads=2,
        pad_arc=np.radians(176),
        oil_supply_temperature=Q_(50, "degC"),
        frequency=Q_([900], "RPM"),
        fys_load=-112814.91,
        lubricant="ISOVG32",
        oil_flow_v=Q_(37.86, "l/min"),
        thermal_type=None,
        total_ey_film=10,
        total_ey_pad=10,
    )
    assert_allclose(plain_arc.pad_arc, plain_journal.pad_arc)
    assert_allclose(
        np.asarray(plain_arc.kxx, dtype=float),
        np.asarray(plain_journal.kxx, dtype=float),
    )


def test_oil_flow_v_default_warns():
    with pytest.warns(UserWarning, match="ample flooded supply"):
        bearing = PlainJournal(
            n=3,
            pad_axial_length=0.263144,
            journal_diameter=0.4,
            radial_clearance=1.95e-4,
            total_ex_film=20,
            total_ez_film=10,
            n_pads=2,
            pad_arc=Q_(176, "deg"),
            oil_supply_temperature=Q_(50, "degC"),
            frequency=Q_([900], "RPM"),
            fys_load=-112814.91,
            lubricant="ISOVG32",
            thermal_type=None,
            total_ey_film=10,
            total_ey_pad=10,
        )
    assert_allclose(bearing.oil_flow_v, 1.0e-2)


def test_odd_element_counts_raise():
    with pytest.raises(ValueError, match="must be an even number"):
        PlainJournal(
            n=3,
            pad_axial_length=0.263144,
            journal_diameter=0.4,
            radial_clearance=1.95e-4,
            total_ex_film=11,
            total_ez_film=4,
            n_pads=2,
            pad_arc=Q_(176, "deg"),
            oil_supply_temperature=Q_(50, "degC"),
            frequency=Q_([900], "RPM"),
            fys_load=-112814.91,
            lubricant="ISOVG32",
            oil_flow_v=Q_(37.86, "l/min"),
            thermal_type=None,
            total_ey_film=10,
            total_ey_pad=10,
        )


def test_thermal_model_runs():
    bearing = PlainJournal(
        n=3,
        pad_axial_length=0.263144,
        journal_diameter=0.4,
        radial_clearance=1.95e-4,
        total_ex_film=20,
        total_ez_film=10,
        n_pads=2,
        pad_arc=Q_(176, "deg"),
        oil_supply_temperature=Q_(50, "degC"),
        frequency=Q_([900], "RPM"),
        fys_load=-112814.91,
        lubricant="ISOVG32",
        oil_flow_v=Q_(37.86, "l/min"),
        total_ey_film=10,
        total_ey_pad=10,
    )
    assert bearing.thermal_type == "adiabatic"
    out = bearing._results.outputs[0]
    assert out["tpad_max"][0] > 323.15
    assert float(bearing.kxx[0]) > 0
