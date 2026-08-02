import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.fixed_geometry import (
    EllipticalBearing,
    FixedGeometryBearing,
    MultiLobeBearing,
    OffsetHalvesBearing,
    PartialArcBearing,
    PressureDamBearing,
    elliptical_bearing_example,
    pressure_dam_bearing_example,
)
from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture
from ross.units import Q_

COMMON = dict(
    n=0,
    frequency=Q_([3000], "RPM"),
    journal_diameter=0.2,
    radial_clearance=150e-6,
    pad_thickness=0.05,
    lubricant="ISOVG32",
    oil_supply_temperature=Q_(40, "degC"),
    oil_flow_v=Q_(30, "l/min"),
    weight=45e3,
    thermal_type=None,
    total_ex_film=20,
    total_ey_film=10,
    total_ez_film=10,
    total_ey_pad=10,
)


def test_partial_arc_translation():
    bearing = PartialArcBearing(
        pad_arc=Q_(150, "deg"),
        pad_axial_length=[0.16],
        **COMMON,
    )
    assert bearing.n_pads == 1
    assert_allclose(bearing.pivot_angle, [3 * np.pi / 2])
    assert_allclose(bearing.pad_arc, [np.radians(150)])
    assert_allclose(bearing.preload, [0.0])
    assert_allclose(bearing.offset, [0.5])
    assert float(bearing.kxx[0]) > 0


def test_elliptical_translation():
    bearing = elliptical_bearing_example()
    assert bearing.n_pads == 2
    assert_allclose(bearing.pivot_angle, np.radians([90, 270]))
    assert_allclose(bearing.preload, [0.5, 0.5])
    assert_allclose(bearing.offset, [0.5, 0.5])
    assert float(bearing.kxx[0]) > 0


def test_offset_halves_translation():
    bearing = OffsetHalvesBearing(
        pad_arc=Q_(150, "deg"),
        preload=0.4,
        offset=0.7,
        pad_axial_length=[0.16, 0.16],
        **COMMON,
    )
    assert_allclose(bearing.offset, [0.7, 0.7])
    assert_allclose(bearing.preload, [0.4, 0.4])
    assert float(bearing.kxx[0]) > 0


def test_multi_lobe_translation():
    bearing = MultiLobeBearing(
        n_lobes=3,
        pad_arc=Q_(100, "deg"),
        preload=0.4,
        pad_axial_length=[0.16] * 3,
        **COMMON,
    )
    assert bearing.n_pads == 3
    assert_allclose(bearing.pivot_angle, np.radians([60, 180, 300]))
    assert_allclose(bearing.preload, [0.4] * 3)
    assert float(bearing.kxx[0]) > 0

    shifted = MultiLobeBearing(
        n_lobes=4,
        pad_arc=Q_(70, "deg"),
        preload=[0.3, 0.4, 0.3, 0.4],
        first_lobe_angle=Q_(0, "deg"),
        pad_axial_length=[0.16] * 4,
        **COMMON,
    )
    assert_allclose(shifted.pivot_angle, np.radians([0, 90, 180, 270]))
    assert_allclose(shifted.preload, [0.3, 0.4, 0.3, 0.4])


def test_pressure_dam_translation():
    bearing = pressure_dam_bearing_example()
    assert bearing.bearing_type == "pressure_dam"
    assert_allclose(bearing.track_arc, [np.radians(90), 0.0])
    assert_allclose(bearing.track_axial_length, [0.1, 0.0])
    assert_allclose(bearing.track_depth, [250e-6, 0.0])


def test_pressure_dam_matches_solver_fixture():
    """PressureDamBearing reproduces the raw pressure-dam solver fixture."""
    kwargs, outputs = bearing_kwargs_from_fixture("fixed_pdam_bt5")
    inp_track_arc = kwargs.pop("track_arc")
    inp_track_axial = kwargs.pop("track_axial_length")
    inp_track_depth = kwargs.pop("track_depth")
    for name in (
        "bearing_type",
        "pivot_angle",
        "pad_arc",
        "preload",
        "offset",
    ):
        kwargs.pop(name)
    bearing = PressureDamBearing(
        pad_arc=Q_(176, "deg"),
        dam_arc=inp_track_arc[0],
        dam_axial_length=inp_track_axial[0],
        dam_depth=inp_track_depth[0],
        dam_pads=(0, 1),
        **kwargs,
    )
    for name in ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"):
        assert_allclose(
            np.asarray(getattr(bearing, name), dtype=float),
            np.asarray(outputs[name], dtype=float),
            rtol=1e-8,
            err_msg=f"{name} differs from the solver fixture",
        )


def test_fixed_geometry_restrictions():
    kwargs = dict(
        COMMON,
        pivot_angle=Q_([90, 270], "deg"),
        pad_arc=Q_([150, 150], "deg"),
        pad_axial_length=[0.16, 0.16],
        preload=[0, 0],
        offset=[0.5, 0.5],
    )
    with pytest.raises(ValueError, match="use\nTiltingPad|use TiltingPad"):
        FixedGeometryBearing(bearing_type="conventional_tilting_pad", **kwargs)
    with pytest.raises(
        ValueError, match="tilting-pad\nbearings only|tilting-pad bearings only"
    ):
        FixedGeometryBearing(deform_type="pad_pivot_mechanical", **kwargs)


def test_required_geometry_arguments():
    with pytest.raises(ValueError, match="pad_arc must be informed"):
        PartialArcBearing(pad_axial_length=[0.16], **COMMON)
    with pytest.raises(ValueError, match="preload must be informed"):
        EllipticalBearing(
            pad_arc=Q_(150, "deg"), pad_axial_length=[0.16, 0.16], **COMMON
        )
    with pytest.raises(ValueError, match="must be informed"):
        PressureDamBearing(
            pad_arc=Q_(150, "deg"), pad_axial_length=[0.16, 0.16], **COMMON
        )
    with pytest.raises(ValueError, match="n_lobes"):
        MultiLobeBearing(
            pad_arc=Q_(100, "deg"), preload=0.4, pad_axial_length=[0.16], **COMMON
        )


def test_subclass_save_downgrades(tmp_path):
    from ross.bearing_seal_element import BearingElement
    from ross.utils import load_data

    bearing = elliptical_bearing_example()
    file = tmp_path / "elliptical.toml"
    bearing.save(file)
    data = load_data(file)
    (key,) = data.keys()
    assert key.startswith("BearingElement_")
    loaded = BearingElement.load(file)
    assert_allclose(
        np.asarray(loaded.kxx, dtype=float),
        np.asarray(bearing.kxx, dtype=float),
        rtol=1e-12,
    )
