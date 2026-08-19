"""Tilting-pad journal bearing solved by the fluid-film TEHD engine.

:class:`TiltingPad` models tilting-pad journal bearings on the solver in
:mod:`ross.bearings.fluid_film`, using the canonical
:class:`ross.bearings.fluid_film_bearing.FluidFilmBearing` parameter
vocabulary. The solver's tilting-pad capabilities are available here:
pivot flexibility (Hertzian contact or user stiffness),
leading-edge-groove and spray-bar lubrication, pad and pivot deformation,
and the load-matched equilibrium.
"""

import warnings

import numpy as np

from ross.bearings.fluid_film.constants import TILTING_PAD_TYPES
from ross.bearings.fluid_film_bearing import FluidFilmBearing
from ross.units import Q_, check_units

__all__ = [
    "TiltingPad",
    "tilting_pad_adiabatic_example",
    "tilting_pad_full_thermal_example",
]


class TiltingPad(FluidFilmBearing):
    """Tilting-pad journal bearing.

    Each pad pivots freely; for every entry of ``frequency`` the fluid-film
    engine solves the pad tilt and journal equilibrium together with the
    film pressure/temperature (and, when selected, pad and pivot
    deformation), then condenses the pad degrees of freedom into the
    synchronous stiffness and damping coefficients.

    ``pivot_type`` selects the pivot flexibility model (with
    ``deform_type`` including pivot deformation), ``bearing_type`` selects
    conventional, leading-edge-groove (``"inlet_groove_tilting_pad"``) or
    spray-bar (``"spray_bar_tilting_pad"``) lubrication, and
    ``equilibrium_type`` selects the load-matched or held-eccentricity
    equilibrium.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    journal_diameter : float, pint.Quantity
        Journal diameter, m.
    preload : array_like
        Per-pad preload factor.
    pad_thickness : float, pint.Quantity
        Radial pad thickness, m.
    pad_arc : array_like, pint.Quantity
        Per-pad arc length, rad.
    offset : array_like
        Per-pad pivot offset fraction (0.5 = centered).
    pad_axial_length : array_like, pint.Quantity
        Per-pad axial length, m.
    lubricant : str or dict
        Key of :data:`ross.bearings.lubricants.lubricants_dict` or a dict
        with the same field names (SI).
    oil_supply_temperature : float, pint.Quantity
        Lubricant supply temperature, K.
    radial_clearance : float, pint.Quantity
        Radial (bearing-set) clearance, m.
    pivot_angle : array_like, pint.Quantity
        Per-pad pivot angular position, rad.
    frequency : array_like, pint.Quantity
        Operating frequencies, rad/s.
    total_ex_film : int, optional
        Circumferential film elements per pad (must be even).
        Default is 30.
    total_ez_film : int, optional
        Axial film elements (must be even). Default is 30.
    total_ey_pad : int, optional
        Radial pad elements (must be even). Default is 16.
    xj, yj : float, optional
        Journal position as fractions of the radial clearance; overrides
        ``eccentricity`` / ``attitude_angle`` when given.
    equilibrium_type : str, optional
        ``"match_eccentricity"`` (default: the journal is held at the
        given position) or ``"match_load"`` (the position is solved for
        the applied load).
    eccentricity : float, optional
        Eccentricity ratio of the (initial or held) journal position.
        Default is 0.3.
    attitude_angle : float, pint.Quantity, optional
        Attitude angle of the (initial or held) journal position, rad,
        measured from the +x axis. Default is 3*pi/2 (bottom).
    fxs_load : float, pint.Quantity, optional
        Applied static load in x, N. Default is 0.
    fys_load : float, pint.Quantity, optional
        Applied static load in y, N (negative = downward). Default is 0.
    thermal_type : str or None, optional
        ``"full"`` (default), ``"adiabatic"`` or ``None`` (isoviscous).
    pad_conductivity : float, optional
        Pad thermal conductivity, W/(m*K). Default is 116.
    edges_convection : float, optional
        Pad edge convection coefficient, W/(m**2*K). Default is 1500.
    relax_temperature : float, optional
        Temperature under-relaxation factor. Default is 0.5.
    journal_temperature : float, pint.Quantity, optional
        Initial journal temperature estimate, K. Default is the supply
        temperature.
    oil_flow_v : float, pint.Quantity, optional
        Supplied oil flow, m**3/s. When omitted, an ample flooded supply
        is assumed (with a warning).
    **kwargs : dict
        Further parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`
        (``bearing_type``, ``pivot_type``, ``deform_type``,
        ``pivot_stiffness``, ``house_diameter``, ``pivot_diameter``,
        ``hot_oil_lambda``, ...).

    Returns
    -------
    A TiltingPad object.

    Examples
    --------
    >>> from ross.bearings.tilting_pad import tilting_pad_adiabatic_example
    >>> bearing = tilting_pad_adiabatic_example()
    >>> bearing.n_pads
    5
    """

    @check_units
    def __init__(
        self,
        n,
        journal_diameter=None,
        preload=None,
        pad_thickness=None,
        pad_arc=None,
        offset=None,
        pad_axial_length=None,
        lubricant=None,
        oil_supply_temperature=None,
        radial_clearance=None,
        pivot_angle=None,
        frequency=None,
        total_ex_film=30,
        total_ez_film=30,
        total_ey_pad=16,
        xj=None,
        yj=None,
        equilibrium_type="match_eccentricity",
        eccentricity=0.3,
        attitude_angle=np.pi * 3 / 2,
        fxs_load=0,
        fys_load=0,
        thermal_type="full",
        pad_conductivity=116.0,
        edges_convection=1500.0,
        relax_temperature=0.5,
        journal_temperature=None,
        oil_flow_v=None,
        **kwargs,
    ):
        bearing_type = kwargs.setdefault("bearing_type", "conventional_tilting_pad")
        if bearing_type not in TILTING_PAD_TYPES:
            raise ValueError(
                f"bearing_type must be one of {sorted(TILTING_PAD_TYPES)} "
                f"for a tilting-pad bearing, not {bearing_type!r}; use "
                "FixedGeometryBearing for the fixed types"
            )

        if oil_flow_v is None:
            warnings.warn(
                "oil_flow_v not informed; assuming an ample flooded supply "
                "(1e-2 m^3/s).",
                UserWarning,
                stacklevel=2,
            )
            oil_flow_v = 1.0e-2

        if xj is not None and yj is not None:
            initial_position = (float(xj), float(yj))
        else:
            initial_position = (
                float(eccentricity) * np.cos(float(attitude_angle)),
                float(eccentricity) * np.sin(float(attitude_angle)),
            )

        super().__init__(
            n,
            frequency=frequency,
            journal_diameter=journal_diameter,
            radial_clearance=radial_clearance,
            pad_thickness=pad_thickness,
            pivot_angle=pivot_angle,
            pad_arc=pad_arc,
            pad_axial_length=pad_axial_length,
            preload=preload,
            offset=offset,
            lubricant=lubricant,
            oil_supply_temperature=oil_supply_temperature,
            oil_flow_v=oil_flow_v,
            fxs_load=fxs_load,
            fys_load=fys_load,
            equilibrium_type=equilibrium_type,
            thermal_type=thermal_type,
            initial_position=initial_position,
            pad_conductivity=pad_conductivity,
            edges_convection=edges_convection,
            relax_temperature=relax_temperature,
            journal_temperature=journal_temperature,
            total_ex_film=total_ex_film,
            total_ez_film=total_ez_film,
            total_ey_pad=total_ey_pad,
            **kwargs,
        )


def tilting_pad_adiabatic_example():
    """Create an example tilting-pad bearing with the adiabatic model.

    Five pads on a 101.6 mm journal; the adiabatic thermal model solves
    the film energy equation only. Coarse mesh, so it runs fast enough
    for documentation examples.

    Returns
    -------
    A TiltingPad object.

    Examples
    --------
    >>> from ross.bearings.tilting_pad import tilting_pad_adiabatic_example
    >>> bearing = tilting_pad_adiabatic_example()
    >>> bearing.n_pads
    5
    """
    return TiltingPad(
        n=1,
        frequency=Q_([3000], "RPM"),
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
        fxs_load=8.8405e02,
        fys_load=-2.6704e03,
        total_ex_film=20,
        total_ez_film=10,
        total_ey_pad=10,
        total_ey_film=10,
    )


def tilting_pad_full_thermal_example():
    """Create an example tilting-pad bearing with the full thermal model.

    Like :func:`tilting_pad_adiabatic_example` but with the coupled
    film + pad conduction energy equation and steel pads. The full model
    iterates the thermo-elastic solution, so this example takes several
    seconds to run.

    Returns
    -------
    A TiltingPad object.
    """
    return TiltingPad(
        n=1,
        frequency=Q_([3000], "RPM"),
        equilibrium_type="match_load",
        thermal_type="full",
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
        fxs_load=8.8405e02,
        fys_load=-2.6704e03,
        pad_conductivity=45.0,
        edges_convection=2000.0,
        journal_temperature=Q_(25, "degC"),
        total_ex_film=20,
        total_ez_film=10,
        total_ey_pad=10,
        total_ey_film=10,
    )
