"""Plain journal bearing solved by the fluid-film TEHD engine.

:class:`PlainJournal` models the classic multi-pad cylindrical journal
bearing (a full bore split into lands by axial grooves) on the solver in
:mod:`ross.bearings.fluid_film`. The constructor uses the canonical
:class:`ross.bearings.fluid_film_bearing.FluidFilmBearing` parameter
vocabulary.
"""

import warnings

import numpy as np

from ross.bearings.fixed_geometry import FixedGeometryBearing
from ross.units import check_units

__all__ = ["PlainJournal"]


class PlainJournal(FixedGeometryBearing):
    """Plain (axial-groove) cylindrical journal bearing.

    The bore is ``n_pads`` fixed lands separated by axial oil grooves, each
    land spanning ``pad_arc``. For every entry of ``frequency`` the
    fluid-film engine solves the journal equilibrium and the film
    pressure/temperature, and reduces the solution to the synchronous
    stiffness and damping coefficients.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    pad_axial_length : float, pint.Quantity
        Bearing (land) axial length, m.
    journal_diameter : float, pint.Quantity
        Journal diameter, m.
    radial_clearance : float, pint.Quantity
        Radial clearance, m.
    n_pads : int
        Number of lands (pads) between the axial grooves.
    pad_arc : float, pint.Quantity
        Arc of each land, rad.
    preload : float, optional
        Land preload factor. Default is 0 (cylindrical bore).
    oil_supply_temperature : float, pint.Quantity
        Lubricant supply temperature, K.
    frequency : array_like, pint.Quantity
        Operating frequencies, rad/s.
    fxs_load : float, pint.Quantity, optional
        Static load in x, N. Default is 0.
    fys_load : float, pint.Quantity, optional
        Static load in y, N (negative = downward). Default is 0.
    lubricant : str or dict
        Key of :data:`ross.bearings.lubricants.lubricants_dict` or a dict
        with the same field names (SI).
    initial_position : tuple of float, optional
        Initial journal position guess as fractions of the radial
        clearance. Default is (0.1, -0.1).
    oil_supply_pressure : float, pint.Quantity, optional
        Oil supply pressure, Pa. Default is 0.
    oil_flow_v : float, pint.Quantity, optional
        Supplied oil flow, m**3/s. When omitted, an ample flooded supply
        is assumed (with a warning).
    pad_thickness : float, pint.Quantity, optional
        Radial pad (bush wall) thickness, m. Default is a quarter of the
        journal diameter.
    **kwargs : dict
        Further parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`
        (``thermal_type``, ``operating_type``, ``deform_type``, mesh
        overrides such as ``total_ex_film`` / ``total_ez_film``, ...).

    Returns
    -------
    A PlainJournal object.

    Attributes
    ----------
    equilibrium_pos : ndarray
        ``[eccentricity_ratio, attitude_angle]`` (rad) of the first
        frequency.

    Examples
    --------
    >>> from ross.bearings.plain_journal import PlainJournal
    >>> from ross.units import Q_
    >>> bearing = PlainJournal(
    ...     n=3,
    ...     pad_axial_length=0.263144,
    ...     journal_diameter=0.4,
    ...     radial_clearance=1.95e-4,
    ...     n_pads=2,
    ...     pad_arc=Q_(176, "deg"),
    ...     preload=0,
    ...     oil_supply_temperature=Q_(50, "degC"),
    ...     frequency=Q_([900], "RPM"),
    ...     fxs_load=0,
    ...     fys_load=-112814.91,
    ...     lubricant="ISOVG32",
    ...     oil_flow_v=Q_(37.86, "l/min"),
    ...     thermal_type=None,
    ...     total_ex_film=20,
    ...     total_ez_film=10,
    ...     total_ey_film=10,
    ...     total_ey_pad=10,
    ... )
    >>> bearing.n_pads
    2
    """

    @check_units
    def __init__(
        self,
        n,
        pad_axial_length=None,
        journal_diameter=None,
        radial_clearance=None,
        n_pads=None,
        pad_arc=None,
        preload=0.0,
        oil_supply_temperature=None,
        frequency=None,
        fxs_load=0,
        fys_load=0,
        lubricant=None,
        initial_position=(0.1, -0.1),
        oil_supply_pressure=0,
        oil_flow_v=None,
        pad_thickness=None,
        **kwargs,
    ):
        if oil_supply_temperature is None:
            raise ValueError("oil_supply_temperature must be informed")

        if oil_flow_v is None:
            warnings.warn(
                "oil_flow_v not informed; assuming an ample flooded supply "
                "(1e-2 m^3/s).",
                UserWarning,
                stacklevel=2,
            )
            oil_flow_v = 1.0e-2

        n_pads = int(n_pads)
        pad_centers_deg = np.arange(0.0, 360.0, 360.0 / n_pads) + 180.0 / n_pads
        pad_arc = float(pad_arc)

        kwargs.setdefault("thermal_type", "adiabatic")
        kwargs.setdefault("total_ey_film", 10)
        kwargs.setdefault("total_ey_pad", 10)
        kwargs.setdefault("reference_temperature", float(oil_supply_temperature))
        if pad_thickness is None:
            pad_thickness = float(journal_diameter) / 4.0

        super().__init__(
            n,
            frequency=frequency,
            journal_diameter=journal_diameter,
            radial_clearance=radial_clearance,
            pad_thickness=pad_thickness,
            pivot_angle=np.radians(pad_centers_deg),
            pad_arc=[pad_arc] * n_pads,
            pad_axial_length=[float(pad_axial_length)] * n_pads,
            preload=[float(preload)] * n_pads,
            offset=[0.5] * n_pads,
            lubricant=lubricant,
            oil_supply_temperature=oil_supply_temperature,
            oil_flow_v=oil_flow_v,
            oil_supply_pressure=oil_supply_pressure,
            fxs_load=fxs_load,
            fys_load=fys_load,
            initial_position=tuple(initial_position),
            **kwargs,
        )

        if getattr(self, "_results", None) is not None:
            first = self._results.outputs[0]
            self.equilibrium_pos = np.array(
                [first["eccentricity"][0], first["attitude"][0]]
            )
