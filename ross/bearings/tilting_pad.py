"""Tilting-pad journal bearing solved by the fluid-film TEHD engine.

:class:`TiltingPad` models tilting-pad journal bearings on the solver in
:mod:`ross.bearings.fluid_film`. The constructor keeps the historical
parameter surface where meaningful and warns on the parameters the new
solver retires; results differ from previous ROSS versions (different
turbulence model, cavitation treatment and energy equation). The rewrite
also brings the solver's additional capabilities to this class: pivot
flexibility (Hertzian contact or user stiffness), leading-edge-groove and
spray-bar lubrication, pad and pivot deformation, and the load-matched
equilibrium.
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

# Solver-iteration knobs of the previous implementation with no counterpart
# in the new solver (its convergence tolerances are fixed, calibrated
# values).
_RETIRED_SOLVER_KNOBS = (
    "solver_options",
    "initial_pads_angles",
    "inlet_temperature_tolerance",
    "max_inlet_iterations",
    "max_jtemp_iter",
    "jtemp_error",
    "max_relax_change",
    "h_sump",
)


class TiltingPad(FluidFilmBearing):
    """Tilting-pad journal bearing.

    Each pad pivots freely; for every entry of ``frequency`` the fluid-film
    engine solves the pad tilt and journal equilibrium together with the
    film pressure/temperature (and, when selected, pad and pivot
    deformation), then condenses the pad degrees of freedom into the
    synchronous stiffness and damping coefficients.

    This class replaces the previous ROSS implementation. Results differ
    from previous versions (different turbulence model, cavitation
    treatment and energy equation), and the solver's tilting-pad
    capabilities are now available here: ``pivot_type`` selects the pivot
    flexibility model (with ``deform_type`` including pivot deformation),
    ``bearing_type`` selects conventional, leading-edge-groove
    (``"inlet_groove_tilting_pad"``) or spray-bar
    (``"spray_bar_tilting_pad"``) lubrication, and ``equilibrium_type``
    selects the load-matched or held-eccentricity equilibrium.

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    journal_diameter : float, pint.Quantity
        Journal diameter, m.
    pre_load : array_like
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
    nx : int, optional
        Circumferential film elements per pad (rounded up to even).
        Default is 30.
    nz : int, optional
        Axial film elements (rounded up to even). Default is 30.
    nr_pad : int, optional
        Radial pad elements (rounded up to even). Default is 16.
    xj, yj : float, optional
        Journal position as fractions of the radial clearance; overrides
        ``eccentricity`` / ``attitude_angle`` when given.
    equilibrium_type : str, optional
        ``"match_eccentricity"`` (default, historical behavior: the
        journal is held at the given position) or ``"match_load"`` (the
        position is solved for the applied load).
    eccentricity : float, optional
        Eccentricity ratio of the (initial or held) journal position.
        Default is 0.3.
    attitude_angle : float, pint.Quantity, optional
        Attitude angle of the (initial or held) journal position, rad,
        measured from the +x axis. Default is 3*pi/2 (bottom).
    load : list of float, optional
        Applied static load ``[fx, fy]``, N.
    thermal_type : str or None, optional
        ``"full"`` (default), ``"adiabatic"`` or ``None`` (isoviscous).
    hot_oil_carry_over : float, optional
        Hot-oil carryover factor of the groove mixing model.
        Default is 0.8.
    k_pad : float, optional
        Pad thermal conductivity, W/(m*K). Default is 116.
    h_edge : float, optional
        Pad edge convection coefficient, W/(m**2*K). Default is 1500.
    relax_t : float, optional
        Temperature under-relaxation factor. Default is 0.5.
    journal_temperature : float, pint.Quantity, optional
        Initial journal temperature estimate. Pint quantities are
        converted; a plain number below 200 is interpreted as degC (the
        historical convention) with a warning. Default is the supply
        temperature.
    oil_flow_v : float, pint.Quantity, optional
        Supplied oil flow, m**3/s. When omitted, an ample flooded supply
        is assumed (with a warning).
    solver_options, initial_pads_angles, inlet_temperature_tolerance, \
max_inlet_iterations, max_jtemp_iter, jtemp_error, max_relax_change, \
h_sump :
        Deprecated and ignored: the solver owns its iteration strategy and
        convergence tolerances.
    **kwargs : dict
        Further parameters of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`
        (``bearing_type``, ``pivot_type``, ``deform_type``,
        ``pivot_stiffness``, ``house_diameter``, ``pivot_diameter``, ...).

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
        pre_load=None,
        pad_thickness=None,
        pad_arc=None,
        offset=None,
        pad_axial_length=None,
        lubricant=None,
        oil_supply_temperature=None,
        radial_clearance=None,
        pivot_angle=None,
        frequency=None,
        nx=30,
        nz=30,
        nr_pad=16,
        xj=None,
        yj=None,
        equilibrium_type="match_eccentricity",
        eccentricity=0.3,
        attitude_angle=np.pi * 3 / 2,
        load=None,
        thermal_type="full",
        hot_oil_carry_over=0.8,
        k_pad=116.0,
        h_edge=1500.0,
        relax_t=0.5,
        journal_temperature=None,
        oil_flow_v=None,
        **kwargs,
    ):
        retired = [
            name for name in _RETIRED_SOLVER_KNOBS if kwargs.pop(name, None) is not None
        ]
        if retired:
            warnings.warn(
                f"{', '.join(retired)} deprecated and ignored: the solver "
                "owns its iteration strategy and convergence tolerances.",
                DeprecationWarning,
                stacklevel=2,
            )

        bearing_type = kwargs.setdefault("bearing_type", "conventional_tilting_pad")
        if bearing_type not in TILTING_PAD_TYPES:
            raise ValueError(
                f"bearing_type must be one of {sorted(TILTING_PAD_TYPES)} "
                f"for a tilting-pad bearing, not {bearing_type!r}; use "
                "FixedGeometryBearing for the fixed types"
            )

        if journal_temperature is not None:
            journal_temperature = float(journal_temperature)
            if journal_temperature < 200.0:
                warnings.warn(
                    f"journal_temperature={journal_temperature} interpreted "
                    "as degC (historical convention); pass a pint quantity "
                    "or kelvin to silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
                journal_temperature += 273.15

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

        fxs_load, fys_load = (0.0, 0.0) if load is None else load

        def even(value):
            value = int(value)
            return value if value % 2 == 0 else value + 1

        super().__init__(
            n,
            frequency=frequency,
            journal_diameter=journal_diameter,
            radial_clearance=radial_clearance,
            pad_thickness=pad_thickness,
            pivot_angle=pivot_angle,
            pad_arc=pad_arc,
            pad_axial_length=pad_axial_length,
            preload=pre_load,
            offset=offset,
            lubricant=lubricant,
            oil_supply_temperature=oil_supply_temperature,
            oil_flow_v=oil_flow_v,
            fxs_load=fxs_load,
            fys_load=fys_load,
            equilibrium_type=equilibrium_type,
            thermal_type=thermal_type,
            initial_position=initial_position,
            hot_oil_lambda=hot_oil_carry_over,
            pad_conductivity=k_pad,
            edges_convection=h_edge,
            relax_temperature=relax_t,
            journal_temperature=journal_temperature,
            total_ex_film=even(nx),
            total_ez_film=even(nz),
            total_ey_pad=even(nr_pad),
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
        pre_load=[0.5] * 5,
        offset=[0.5] * 5,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(40, "degC"),
        oil_flow_v=Q_(10, "l/min"),
        load=[8.8405e02, -2.6704e03],
        k_pad=45.0,
        h_edge=2000.0,
        journal_temperature=Q_(25, "degC"),
        nx=20,
        nz=10,
        nr_pad=10,
        total_ey_film=10,
    )
