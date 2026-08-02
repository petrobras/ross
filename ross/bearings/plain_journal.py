"""Plain journal bearing solved by the fluid-film TEHD engine.

:class:`PlainJournal` models the classic multi-pad cylindrical journal
bearing (a full bore split into lands by axial grooves) on the solver in
:mod:`ross.bearings.fluid_film`. The constructor keeps the historical
parameter surface where meaningful and warns on the parameters the new
solver retires; results differ from previous ROSS versions (different
turbulence model, cavitation treatment and energy equation).
"""

import warnings

import numpy as np

from ross.bearings.fixed_geometry import FixedGeometryBearing
from ross.units import Q_, check_units

__all__ = ["PlainJournal"]

# Old operating_type vocabulary -> the solver's.
_OPERATING_TYPE_MAP = {
    "flooded": "regular_flooded",
    "starvation": "starved_condition_even",
}


class PlainJournal(FixedGeometryBearing):
    """Plain (axial-groove) cylindrical journal bearing.

    The bore is ``n_pad`` fixed lands separated by axial oil grooves, each
    land spanning ``pad_arc_length``. For every entry of ``frequency`` the
    fluid-film engine solves the journal equilibrium and the film
    pressure/temperature, and reduces the solution to the synchronous
    stiffness and damping coefficients.

    This class replaces the previous ROSS implementation: one perturbation
    solve now produces the coefficients (the ``method`` split is gone), the
    groove mixing is modeled through the solver's hot-oil carryover factor
    instead of ``groove_factor``, and results differ from previous
    versions -- the solver uses a different turbulence model, cavitation
    treatment and energy equation.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    axial_length : float, pint.Quantity
        Bearing axial length, m.
    journal_radius : float, pint.Quantity
        Journal radius, m.
    radial_clearance : float, pint.Quantity
        Radial clearance, m.
    elements_circumferential : int
        Circumferential film elements per pad (rounded up to even).
    elements_axial : int
        Axial film elements (rounded up to even).
    n_pad : int
        Number of lands (pads) between the axial grooves.
    pad_arc_length : float, pint.Quantity
        Arc length of each land. A plain number is taken in **degrees**
        (historical convention); pint quantities carry their own unit.
    preload : float, optional
        Land preload factor. Default is 0 (cylindrical bore).
    geometry : str, optional
        ``"circular"`` (default). The values ``"lobe"`` and
        ``"elliptical"`` are deprecated: they still run (through the
        ``preload`` factor) but
        :class:`ross.bearings.fixed_geometry.MultiLobeBearing` and
        :class:`ross.bearings.fixed_geometry.EllipticalBearing` are their
        dedicated classes.
    reference_temperature : float, pint.Quantity
        Lubricant supply temperature. Pint quantities are converted; a
        plain number below 200 is interpreted as degC (the historical
        convention) with a warning, otherwise as K.
    frequency : array_like, pint.Quantity
        Operating frequencies, rad/s.
    fxs_load : float, pint.Quantity, optional
        Static load in x, N. Default is 0.
    fys_load : float, pint.Quantity, optional
        Static load in y, N (negative = downward). Default is 0.
    lubricant : str or dict
        Key of :data:`ross.bearings.lubricants.lubricants_dict` or a dict
        with the same field names (SI).
    sommerfeld_type : int, optional
        Deprecated and ignored (the Sommerfeld number is reported by the
        solver directly).
    initial_guess : tuple of float, optional
        Initial journal position guess as fractions of the radial
        clearance. Default is (0.1, -0.1).
    method : str, optional
        Deprecated and ignored: the solver has a single perturbation
        route (the previous "lund"/"perturbation" split is gone).
    model_type : str, optional
        ``"thermo_hydro_dynamic"`` (default) selects the film energy
        equation (the solver's ``"adiabatic"`` thermal model); pass
        ``thermal_type`` explicitly for the isoviscous or full
        (film + pad conduction) models.
    operating_type : str, optional
        ``"flooded"`` (default) or ``"starvation"`` (historical names),
        or any of the solver's operating types.
    groove_factor : list, optional
        Deprecated and ignored: groove mixing is modeled through the
        solver's hot-oil carryover factor (``hot_oil_lambda``).
    oil_supply_pressure : float, pint.Quantity, optional
        Oil supply pressure, Pa. Default is 0.
    oil_flow_v : float, pint.Quantity, optional
        Supplied oil flow, m**3/s. When omitted, an ample flooded supply
        is assumed (with a warning).
    pad_thickness : float, pint.Quantity, optional
        Radial pad (bush wall) thickness, m. Default is half the journal
        radius.
    **kwargs : dict
        Further parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`
        (``thermal_type``, ``deform_type``, mesh overrides, ...).

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
    ...     axial_length=0.263144,
    ...     journal_radius=0.2,
    ...     radial_clearance=1.95e-4,
    ...     elements_circumferential=20,
    ...     elements_axial=10,
    ...     n_pad=2,
    ...     pad_arc_length=176,
    ...     preload=0,
    ...     geometry="circular",
    ...     reference_temperature=Q_(50, "degC"),
    ...     frequency=Q_([900], "RPM"),
    ...     fxs_load=0,
    ...     fys_load=-112814.91,
    ...     lubricant="ISOVG32",
    ...     oil_flow_v=Q_(37.86, "l/min"),
    ...     thermal_type=None,
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
        axial_length=None,
        journal_radius=None,
        radial_clearance=None,
        elements_circumferential=None,
        elements_axial=None,
        n_pad=None,
        pad_arc_length=None,
        preload=0.0,
        geometry="circular",
        reference_temperature=None,
        frequency=None,
        fxs_load=0,
        fys_load=0,
        lubricant=None,
        sommerfeld_type=None,
        initial_guess=(0.1, -0.1),
        method=None,
        model_type="thermo_hydro_dynamic",
        operating_type="flooded",
        groove_factor=None,
        oil_supply_pressure=0,
        oil_flow_v=None,
        pad_thickness=None,
        **kwargs,
    ):
        if sommerfeld_type is not None:
            warnings.warn(
                "sommerfeld_type is deprecated and ignored: the Sommerfeld "
                "number is reported by the solver directly.",
                DeprecationWarning,
                stacklevel=2,
            )
        if method is not None:
            warnings.warn(
                "method is deprecated and ignored: the solver has a single "
                "perturbation route for the dynamic coefficients.",
                DeprecationWarning,
                stacklevel=2,
            )
        if groove_factor is not None:
            warnings.warn(
                "groove_factor is deprecated and ignored: groove mixing is "
                "modeled through the solver's hot-oil carryover factor "
                "(hot_oil_lambda).",
                DeprecationWarning,
                stacklevel=2,
            )
        if geometry not in ("circular", "lobe", "elliptical"):
            raise ValueError(
                f"geometry must be 'circular', 'lobe' or 'elliptical', not {geometry!r}"
            )
        if geometry != "circular":
            warnings.warn(
                f"geometry={geometry!r} is deprecated: use "
                "MultiLobeBearing / EllipticalBearing instead. The preload "
                "factor is honored either way.",
                DeprecationWarning,
                stacklevel=2,
            )
        if model_type != "thermo_hydro_dynamic":
            raise ValueError(
                "model_type must be 'thermo_hydro_dynamic'; select other "
                "models with the thermal_type parameter"
            )

        n_pad = int(n_pad)
        pad_centers_deg = np.arange(0.0, 360.0, 360.0 / n_pad) + 180.0 / n_pad

        # Historical contract: a plain number is the arc in degrees. A pint
        # quantity reaches this point already converted to radians, and no
        # real land is shorter than 2*pi degrees, so the magnitude decides.
        pad_arc_length = float(pad_arc_length)
        if pad_arc_length > 2.0 * np.pi:
            pad_arc_rad = np.radians(pad_arc_length)
        else:
            pad_arc_rad = pad_arc_length

        if reference_temperature is None:
            raise ValueError("reference_temperature must be informed")
        reference_temperature = float(reference_temperature)
        if reference_temperature < 200.0:
            warnings.warn(
                f"reference_temperature={reference_temperature} interpreted "
                "as degC (historical convention); pass a pint quantity or "
                "kelvin to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
            reference_temperature += 273.15

        if oil_flow_v is None:
            warnings.warn(
                "oil_flow_v not informed; assuming an ample flooded supply "
                "(1e-2 m^3/s).",
                UserWarning,
                stacklevel=2,
            )
            oil_flow_v = 1.0e-2

        operating_type = _OPERATING_TYPE_MAP.get(operating_type, operating_type)

        # Historical attribute surface kept for convenience. The base class
        # stores its own args (including ``reference_temperature``, which is
        # passed through below) after this block.
        self.axial_length = float(axial_length)
        self.journal_radius = float(journal_radius)
        self.n_pad = n_pad
        self.pad_arc_length = float(np.degrees(pad_arc_rad))
        self.geometry = geometry

        def even(value):
            value = int(value)
            return value if value % 2 == 0 else value + 1

        kwargs.setdefault("thermal_type", "adiabatic")
        kwargs.setdefault("total_ey_film", 10)
        kwargs.setdefault("total_ey_pad", 10)
        if pad_thickness is None:
            pad_thickness = float(journal_radius) / 2.0

        super().__init__(
            n,
            frequency=frequency,
            journal_diameter=2.0 * float(journal_radius),
            radial_clearance=radial_clearance,
            pad_thickness=pad_thickness,
            pivot_angle=np.radians(pad_centers_deg),
            pad_arc=[pad_arc_rad] * n_pad,
            pad_axial_length=[float(axial_length)] * n_pad,
            preload=[float(preload)] * n_pad,
            offset=[0.5] * n_pad,
            lubricant=lubricant,
            oil_supply_temperature=reference_temperature,
            reference_temperature=reference_temperature,
            oil_flow_v=oil_flow_v,
            oil_supply_pressure=oil_supply_pressure,
            fxs_load=fxs_load,
            fys_load=fys_load,
            operating_type=operating_type,
            total_ex_film=even(elements_circumferential),
            total_ez_film=even(elements_axial),
            initial_position=tuple(initial_guess),
            **kwargs,
        )

        if getattr(self, "_results", None) is not None:
            first = self._results.outputs[0]
            self.equilibrium_pos = np.array(
                [first["eccentricity"][0], first["attitude"][0]]
            )
