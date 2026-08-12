"""Fixed-geometry journal bearing configurations.

:class:`FixedGeometryBearing` is the generic per-pad-array surface of the
fluid-film engine for non-tilting bearings; the classes below translate the
classic fixed-geometry configurations -- partial arc, elliptical (lemon),
offset halves, multi-lobe and pressure dam -- into those arrays. Every
class accepts the full operating-condition / model-flag surface of
:class:`ross.bearings.fluid_film_bearing.FluidFilmBearing` as keyword
arguments.

References
----------
.. [1] Someya, T. (Ed.). (1989). Journal-Bearing Databook. Springer.
.. [2] Allaire, P. E., & Flack, R. D. (1981). Design of journal bearings
       for rotating machinery. Proceedings of the 10th Turbomachinery
       Symposium, 25-45.
.. [3] Nicholas, J. C., & Allaire, P. E. (1980). Analysis of step journal
       bearings -- finite length, stability. ASLE Transactions, 23(2),
       197-207.
"""

import numpy as np

from ross.bearings.fluid_film.constants import PIVOT_FLEX_DEFORM_TYPES
from ross.bearings.fluid_film_bearing import FluidFilmBearing
from ross.units import Q_, check_units

__all__ = [
    "FixedGeometryBearing",
    "PartialArcBearing",
    "EllipticalBearing",
    "OffsetHalvesBearing",
    "MultiLobeBearing",
    "PressureDamBearing",
    "elliptical_bearing_example",
    "pressure_dam_bearing_example",
]


class FixedGeometryBearing(FluidFilmBearing):
    """Fixed-geometry journal bearing described by per-pad arrays.

    The power-user surface: each land (pad) of the bore is given directly
    by its ``pivot_angle`` (arc center), ``pad_arc``, ``preload`` and
    ``offset``, plus the optional pocket (``track_*``) and taper
    (``taper_*``) fields -- any fixed-profile journal bearing the engine
    supports can be expressed this way. The configuration classes below
    build these arrays from friendlier parameters.

    Accepts every parameter of
    :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`;
    ``bearing_type`` is restricted to ``"fixed_geometry"`` (default) or
    ``"pressure_dam"``, and the pivot-flexibility deformation models are
    rejected (they need tilting pads).

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    **kwargs : dict
        Parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`.

    Returns
    -------
    A FixedGeometryBearing object.

    Examples
    --------
    >>> import ross as rs
    >>> from ross.units import Q_
    >>> bearing = rs.FixedGeometryBearing(
    ...     n=0,
    ...     frequency=Q_([900], "RPM"),
    ...     journal_diameter=0.4,
    ...     radial_clearance=195e-6,
    ...     pad_thickness=0.15,
    ...     pivot_angle=Q_([90, 270], "deg"),
    ...     pad_arc=Q_([176, 176], "deg"),
    ...     pad_axial_length=[0.263, 0.263],
    ...     preload=[0, 0],
    ...     offset=[0.5, 0.5],
    ...     lubricant="ISOVG32",
    ...     oil_supply_temperature=Q_(40, "degC"),
    ...     oil_flow_v=Q_(30, "l/min"),
    ...     weight=112.8e3,
    ...     thermal_type=None,
    ...     total_ex_film=20,
    ...     total_ey_film=10,
    ...     total_ez_film=10,
    ...     total_ey_pad=10,
    ... )
    >>> bearing.n_pads
    2
    """

    def __init__(self, n, **kwargs):
        bearing_type = kwargs.setdefault("bearing_type", "fixed_geometry")
        if bearing_type not in ("fixed_geometry", "pressure_dam"):
            raise ValueError(
                "bearing_type must be 'fixed_geometry' or 'pressure_dam' "
                f"for a fixed-geometry bearing, not {bearing_type!r}; use "
                "TiltingPad for the tilting-pad types"
            )
        if kwargs.get("deform_type") in PIVOT_FLEX_DEFORM_TYPES:
            raise ValueError(
                "pivot-flexibility deformation models apply to tilting-pad "
                f"bearings only, not {kwargs['deform_type']!r}"
            )
        super().__init__(n, **kwargs)


class PartialArcBearing(FixedGeometryBearing):
    """Partial-arc journal bearing: a single fixed pad.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    pad_arc : float, pint.Quantity
        Arc length of the pad, rad.
    arc_center : float, pint.Quantity, optional
        Angular position of the pad center, rad. Default is 3*pi/2 (the
        pad centered under a vertical gravity load).
    preload : float, optional
        Preload factor. Default is 0.
    offset : float, optional
        Offset fraction. Default is 0.5 (centered).
    **kwargs : dict
        Parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`.

    Returns
    -------
    A PartialArcBearing object.

    References
    ----------
    Someya, T. (Ed.). (1989). Journal-Bearing Databook. Springer.
    """

    @check_units
    def __init__(
        self,
        n,
        pad_arc=None,
        arc_center=4.71238898038469,
        preload=0.0,
        offset=0.5,
        **kwargs,
    ):
        if pad_arc is None:
            raise ValueError("pad_arc must be informed")
        super().__init__(
            n,
            pivot_angle=[float(arc_center)],
            pad_arc=[float(pad_arc)],
            preload=[float(preload)],
            offset=[float(offset)],
            **kwargs,
        )


class EllipticalBearing(FixedGeometryBearing):
    """Elliptical (lemon-bore) journal bearing: two preloaded lobes.

    The bore is two arcs whose centers are displaced along the vertical
    axis, expressed as two pads at 90 and 270 degrees with equal preload
    and centered offset. The machined (pad) clearance follows from the set
    clearance as ``radial_clearance / (1 - preload)``.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    pad_arc : float, pint.Quantity
        Arc length of each lobe, rad.
    preload : float
        Lobe preload factor; typical lemon-bore values are 0.5-0.75.
    **kwargs : dict
        Parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`.

    Returns
    -------
    An EllipticalBearing object.

    References
    ----------
    Someya, T. (Ed.). (1989). Journal-Bearing Databook. Springer.

    Examples
    --------
    >>> from ross.bearings.fixed_geometry import elliptical_bearing_example
    >>> bearing = elliptical_bearing_example()
    >>> bearing.n_pads
    2
    """

    @check_units
    def __init__(self, n, pad_arc=None, preload=None, **kwargs):
        if pad_arc is None:
            raise ValueError("pad_arc must be informed")
        if preload is None:
            raise ValueError("preload must be informed (lemon-bore factor)")
        super().__init__(
            n,
            pivot_angle=Q_([90.0, 270.0], "deg"),
            pad_arc=[float(pad_arc)] * 2,
            preload=[float(preload)] * 2,
            offset=[0.5, 0.5],
            **kwargs,
        )


class OffsetHalvesBearing(FixedGeometryBearing):
    """Offset-halves journal bearing: two halves displaced at the split.

    Two pads at 90 and 270 degrees whose pivot offset differs from the
    centered 0.5, producing the converging wedges of the offset split
    line.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    pad_arc : float, pint.Quantity
        Arc length of each half, rad.
    preload : float
        Preload factor of each half.
    offset : float
        Pivot offset fraction of each half (0.5 = centered halves).
    **kwargs : dict
        Parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`.

    Returns
    -------
    An OffsetHalvesBearing object.

    References
    ----------
    Allaire, P. E., & Flack, R. D. (1981). Design of journal bearings for
    rotating machinery. Proceedings of the 10th Turbomachinery Symposium.
    """

    @check_units
    def __init__(self, n, pad_arc=None, preload=None, offset=None, **kwargs):
        if pad_arc is None:
            raise ValueError("pad_arc must be informed")
        if preload is None or offset is None:
            raise ValueError("preload and offset must be informed")
        super().__init__(
            n,
            pivot_angle=Q_([90.0, 270.0], "deg"),
            pad_arc=[float(pad_arc)] * 2,
            preload=[float(preload)] * 2,
            offset=[float(offset)] * 2,
            **kwargs,
        )


class MultiLobeBearing(FixedGeometryBearing):
    """Multi-lobe journal bearing: ``n_lobes`` preloaded arcs.

    Lobes are evenly spaced around the bore; the first lobe center sits at
    ``180 / n_lobes`` degrees (so a two-lobe bearing has its lobes at 90
    and 270 degrees, matching the elliptical layout) unless
    ``first_lobe_angle`` says otherwise.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    n_lobes : int
        Number of lobes (3 and 4 are the common configurations).
    pad_arc : float, pint.Quantity
        Arc length of each lobe, rad.
    preload : float or array_like
        Lobe preload factor, one value or one per lobe.
    offset : float or array_like, optional
        Lobe offset fraction, one value or one per lobe. Default is 0.5.
    first_lobe_angle : float, pint.Quantity, optional
        Angular position of the first lobe center, rad. Default is
        ``pi / n_lobes``.
    **kwargs : dict
        Parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`.

    Returns
    -------
    A MultiLobeBearing object.

    References
    ----------
    Someya, T. (Ed.). (1989). Journal-Bearing Databook. Springer.
    """

    @check_units
    def __init__(
        self,
        n,
        n_lobes=None,
        pad_arc=None,
        preload=None,
        offset=0.5,
        first_lobe_angle=None,
        **kwargs,
    ):
        if n_lobes is None or n_lobes < 2:
            raise ValueError("n_lobes must be an integer >= 2")
        if pad_arc is None:
            raise ValueError("pad_arc must be informed")
        if preload is None:
            raise ValueError("preload must be informed")
        n_lobes = int(n_lobes)
        if first_lobe_angle is None:
            first_lobe_angle = np.pi / n_lobes
        pivot_angle = float(first_lobe_angle) + np.arange(n_lobes) * (
            2.0 * np.pi / n_lobes
        )
        super().__init__(
            n,
            pivot_angle=pivot_angle,
            pad_arc=[float(pad_arc)] * n_lobes,
            preload=np.broadcast_to(
                np.atleast_1d(np.asarray(preload, dtype=float)), (n_lobes,)
            ).copy(),
            offset=np.broadcast_to(
                np.atleast_1d(np.asarray(offset, dtype=float)), (n_lobes,)
            ).copy(),
            **kwargs,
        )


class PressureDamBearing(FixedGeometryBearing):
    """Pressure-dam journal bearing: a stepped pocket in selected pads.

    Two pads at 90 and 270 degrees; the pads listed in ``dam_pads`` carry
    a machined pocket (``dam_arc`` x ``dam_axial_length``, ``dam_depth``
    deep) whose step raises the film pressure and loads the journal. By
    default the dam sits in the top (unloaded) pad only.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    pad_arc : float, pint.Quantity
        Arc length of each pad, rad.
    dam_arc : float, pint.Quantity
        Pocket arc length from the pad leading edge, rad.
    dam_axial_length : float, pint.Quantity
        Pocket axial length, m.
    dam_depth : float, pint.Quantity
        Pocket depth, m.
    dam_pads : tuple of int, optional
        0-based pads carrying the dam; pad 0 is at 90 degrees (top) and
        pad 1 at 270 degrees (bottom). Default is ``(0,)``.
    preload : float, optional
        Pad preload factor. Default is 0.
    **kwargs : dict
        Parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`.

    Returns
    -------
    A PressureDamBearing object.

    References
    ----------
    Nicholas, J. C., & Allaire, P. E. (1980). Analysis of step journal
    bearings -- finite length, stability. ASLE Transactions, 23(2).

    Examples
    --------
    >>> from ross.bearings.fixed_geometry import pressure_dam_bearing_example
    >>> bearing = pressure_dam_bearing_example()
    >>> bearing.bearing_type
    'pressure_dam'
    """

    @check_units
    def __init__(
        self,
        n,
        pad_arc=None,
        dam_arc=None,
        dam_axial_length=None,
        dam_depth=None,
        dam_pads=(0,),
        preload=0.0,
        **kwargs,
    ):
        for name, value in (
            ("pad_arc", pad_arc),
            ("dam_arc", dam_arc),
            ("dam_axial_length", dam_axial_length),
            ("dam_depth", dam_depth),
        ):
            if value is None:
                raise ValueError(f"{name} must be informed")
        track_arc = np.zeros(2)
        track_axial_length = np.zeros(2)
        track_depth = np.zeros(2)
        for pad in dam_pads:
            track_arc[pad] = float(dam_arc)
            track_axial_length[pad] = float(dam_axial_length)
            track_depth[pad] = float(dam_depth)
        super().__init__(
            n,
            bearing_type="pressure_dam",
            pivot_angle=Q_([90.0, 270.0], "deg"),
            pad_arc=[float(pad_arc)] * 2,
            preload=[float(preload)] * 2,
            offset=[0.5, 0.5],
            track_arc=track_arc,
            track_axial_length=track_axial_length,
            track_depth=track_depth,
            **kwargs,
        )


def elliptical_bearing_example():
    """Create an example elliptical (lemon-bore) bearing.

    Coarse mesh and isoviscous model, so it runs fast enough for
    documentation examples.

    Returns
    -------
    An EllipticalBearing object.

    Examples
    --------
    >>> from ross.bearings.fixed_geometry import elliptical_bearing_example
    >>> bearing = elliptical_bearing_example()
    >>> [round(p, 2) for p in bearing.preload]
    [0.5, 0.5]
    """
    return EllipticalBearing(
        n=0,
        frequency=Q_([3000], "RPM"),
        pad_arc=Q_(150, "deg"),
        preload=0.5,
        journal_diameter=0.2,
        radial_clearance=150e-6,
        pad_thickness=0.05,
        pad_axial_length=[0.16, 0.16],
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


def pressure_dam_bearing_example():
    """Create an example pressure-dam bearing.

    Coarse mesh and isoviscous model, so it runs fast enough for
    documentation examples.

    Returns
    -------
    A PressureDamBearing object.

    Examples
    --------
    >>> from ross.bearings.fixed_geometry import pressure_dam_bearing_example
    >>> bearing = pressure_dam_bearing_example()
    >>> bearing.n_pads
    2
    """
    return PressureDamBearing(
        n=0,
        frequency=Q_([3000], "RPM"),
        pad_arc=Q_(150, "deg"),
        dam_arc=Q_(90, "deg"),
        dam_axial_length=0.1,
        dam_depth=250e-6,
        journal_diameter=0.2,
        radial_clearance=150e-6,
        pad_thickness=0.05,
        pad_axial_length=[0.16, 0.16],
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
