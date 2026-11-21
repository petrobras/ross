"""Shared base class for journal bearings solved by the fluid-film engine.

:class:`FluidFilmBearing` connects the internal thermo-elasto-hydrodynamic
solver (:mod:`ross.bearings.fluid_film`) to the ROSS element interface: it
resolves the lubricant, assembles the solver inputs, runs one solver case
per rotor speed (serially by default, in parallel on request),
builds the dynamic-coefficient table consumed by
:class:`ross.BearingElement`, and exposes the solved fields through a
results object with the standard ``plot_*`` / ``show_*`` methods.

Configuration classes (plain journal, elliptical, multi-lobe, pressure dam,
tilting pad, ...) subclass this and only translate their user-facing
geometry into the per-pad arrays this class consumes.
"""

import multiprocessing
import os
import time
import warnings

import numpy as np

import ross.units
from ross.bearing_seal_element import BearingElement
from ross.bearings.bearing_results import FluidFilmBearingResults
from ross.bearings.fluid_film.constants import (
    BEARING_TYPES,
    DEFORM_TYPES,
    EQUILIBRIUM_TYPES,
    OPERATING_TYPES,
    PIVOT_TYPES,
    SUMP_TYPES,
    TEMP_J_TYPES,
    THERMAL_TYPES,
)
from ross.units import Q_, _to_base_unit, check_units

__all__ = ["FluidFilmBearing", "fluid_film_bearing_example"]

ON_FAILURE = ("warn", "raise", "keep")
_NO_OPERATING_RESULTS = (
    "operating_results needs a solved bearing (this element was built from a "
    "coefficient table)"
)

# Engine lubricant field -> the ross.bearings.lubricants.lubricants_dict key
# it comes from. These seven properties fully define the lubricant for the
# solver (two-point exponential viscosity law plus density, specific heat
# and thermal conductivity).
_LUBRICANT_FIELDS = {
    "viscosity1": "liquid_viscosity1",
    "temp1": "temperature1",
    "viscosity2": "liquid_viscosity2",
    "temp2": "temperature2",
    "lube_density": "liquid_density",
    "lube_cp": "liquid_specific_heat",
    "lube_conduct": "liquid_thermal_conductivity",
}

# Frames skipped when a warning is attributed to the caller: the bearing
# classes and the unit-conversion decorator that wraps their constructors.
_INTERNAL_FILES = (os.path.dirname(__file__) + os.sep, ross.units.__file__)

_TYPE_FLAGS = {
    "bearing_type": BEARING_TYPES,
    "operating_type": OPERATING_TYPES,
    "thermal_type": THERMAL_TYPES,
    "temp_j_type": TEMP_J_TYPES,
    "deform_type": DEFORM_TYPES,
    "equilibrium_type": EQUILIBRIUM_TYPES,
    "sump_type": SUMP_TYPES,
    "pivot_type": PIVOT_TYPES,
}


def _lubricant_properties(lubricant):
    """Resolve a lubricant into the seven properties the solver uses.

    The name or dictionary is resolved by
    :func:`ross.bearings.film.lubricant.lubricant_data`.

    Parameters
    ----------
    lubricant : str or dict
        A key of :data:`ross.bearings.lubricants.lubricants_dict`, or a dict
        holding (at least) the same field names in SI units.

    Returns
    -------
    dict
        The solver's lubricant inputs: ``viscosity1`` / ``viscosity2``
        (Pa*s), ``temp1`` / ``temp2`` (K), ``lube_density`` (kg/m**3),
        ``lube_cp`` (J/(kg*K)) and ``lube_conduct`` (W/(m*K)).
    """
    # imported here: ``import ross`` does not load the film core
    from ross.bearings.film.lubricant import lubricant_data

    properties = lubricant_data(lubricant, required=tuple(_LUBRICANT_FIELDS.values()))
    return {
        engine_name: float(properties[public_name])
        for engine_name, public_name in _LUBRICANT_FIELDS.items()
    }


def _solve_case(inputs):
    """Run one solver case; module-level so it can cross process boundaries.

    Parameters
    ----------
    inputs : dict
        Keyword arguments for
        :func:`ross.bearings.fluid_film.driver.run_case`, for a single
        rotor speed.

    Returns
    -------
    dict
        The solver output dict (with field arrays).
    """
    from ross.bearings.fluid_film.driver import run_case

    return run_case(**inputs)


class FluidFilmBearing(BearingElement):
    """Journal bearing solved by the fluid-film TEHD engine.

    Shared base for the hydrodynamic journal bearing classes. The bearing
    is described by per-pad arrays -- pivot position, arc, preload, offset
    and the optional pocket / taper fields -- plus operating conditions and
    model-selection flags; any fixed-geometry or tilting-pad journal
    bearing the engine supports can be expressed directly through this
    class. The configuration subclasses only translate friendlier
    constructor surfaces into these arrays.

    For every entry of ``speed`` the engine solves the journal
    equilibrium (film pressure, temperature and deformation as selected by
    the model flags) and reduces the result to the synchronous 2x2
    stiffness and damping matrices that make up the element's coefficient
    table. Solved pressure / temperature / film-thickness fields are kept
    on a results object; every ``plot_*`` / ``show_*`` call on the bearing
    is delegated to it.

    ``save()`` deliberately downgrades the element to a plain
    :class:`ross.BearingElement` holding the solved coefficient table:
    loading the file restores the rotordynamic behavior without re-running
    the solver (the solver inputs are not round-tripped).

    Parameters
    ----------
    n : int
        Node in which the bearing will be located.
    speed : array_like, pint.Quantity
        Rotor speeds, rad/s, finite and positive. One solver case runs per
        entry.
    frequency : array_like, pint.Quantity, optional
        Excitation (whirl) frequencies, rad/s. When given, one solver case
        runs per (speed, frequency) pair with the whirl ratio
        ``frequency / speed``, and the element carries a 2-D coefficient
        table of shape ``(len(speed), len(frequency))`` interpolated on both
        axes (``excitation_ratio`` is ignored). Finite and >= 0.
        Default is None (1-D table over the speed axis, whirl ratio
        ``excitation_ratio``).
    journal_diameter : float, pint.Quantity
        Journal diameter, m.
    radial_clearance : float, pint.Quantity
        Radial (bearing-set) clearance, m.
    pad_thickness : float, pint.Quantity
        Radial pad thickness, m.
    pivot_angle : array_like, pint.Quantity
        Per-pad pivot (or arc-center) angular position, rad.
    pad_arc : array_like, pint.Quantity
        Per-pad arc length, rad. A first pad of exactly 2 pi (within
        1e-6 rad) selects the short-bearing film model of an oil seal (see
        Notes).
    pad_axial_length : array_like, pint.Quantity
        Per-pad axial length, m.
    preload : array_like
        Per-pad preload factor, ``0 <= preload < 1`` (0 for a cylindrical
        land).
    offset : array_like
        Per-pad pivot offset fraction, ``0 <= offset <= 1`` (0.5 = centered).
        The pad leading edge is at ``pivot_angle - offset * pad_arc``; the
        pads must not overlap and their arcs must add up to 360 deg or less.
    lubricant : str or dict
        Key of :data:`ross.bearings.lubricants.lubricants_dict` or a dict
        with the same field names (SI).
    oil_supply_temperature : float, pint.Quantity
        Lubricant supply temperature, K.
    oil_flow_v : float, pint.Quantity
        Supplied lubricant volumetric flow, m**3/s.
    weight : float, pint.Quantity, optional
        Gravity load on the bearing (applied in -y), N. Default is 0.
    fxs_load, fys_load : float, pint.Quantity, optional
        Additional static load in x and y, N. Default is 0. The total
        static load must be nonzero for the load-matched equilibrium.
    bearing_type : str, optional
        One of ``"fixed_geometry"`` (default), ``"conventional_tilting_pad"``,
        ``"inlet_groove_tilting_pad"``, ``"spray_bar_tilting_pad"``,
        ``"pressure_dam"``.
    operating_type : str, optional
        One of ``"regular_flooded"`` (default), ``"axial_flow"``,
        ``"high_ambient_pressure"``, ``"starved_condition_even"``,
        ``"starved_condition_uneven"``, ``"oil_ring_lubricated"``.
    thermal_type : str or None, optional
        ``"full"`` (default): energy equation over film and pad with
        conduction; ``"adiabatic"``: 2-D film energy equation; ``None``:
        isoviscous.
    temp_j_type : str, optional
        Journal surface temperature treatment:
        ``"averaged_film_temperature"`` (default),
        ``"no_heat_flux_into_journal"`` or ``"insulated_shaft_surface"``.
    deform_type : str or None, optional
        Pad/pivot deformation model: ``None`` (default, rigid),
        ``"pad_mechanical"``, ``"pad_mechanical_thermal"``,
        ``"pad_mechanical_thermal_shaft_shell_thermal"``,
        ``"pad_mechanical_thermal_shaft_shell_thermal_pivot_mechanical"``
        or ``"pad_pivot_mechanical"``.
    equilibrium_type : str, optional
        ``"match_load"`` (default) solves the journal position for the
        applied load; ``"match_eccentricity"`` holds ``initial_position``
        and does not use the load (a warning is issued when one is given).
    sump_type : str, optional
        Groove-mixing supply temperature source: ``"supply_temperature"``
        (default) or ``"sump_temperature"``.
    pivot_type : str, optional
        Pivot flexibility model (used by the pivot deformation types):
        ``"ball_in_socket"``, ``"button"``, ``"rocker_back"`` or
        ``"user_specified_stiffness"`` (default).
    total_ex_film, total_ey_film, total_ez_film, total_ey_pad : int, optional
        Element counts of the film (circumferential / radial / axial) and
        pad (radial) meshes; all must be even. Default is 40 / 30 / 20 / 20.
    track_arc, taper_arc_le, taper_arc_te : array_like, pint.Quantity, optional
        Per-pad pocket (or dam track) arc and leading/trailing taper arcs,
        rad. Default is 0.
    track_axial_length : array_like, pint.Quantity, optional
        Per-pad pocket/track axial length, m. Default is 0.
    track_depth, taper_depth_le, taper_depth_te : array_like, pint.Quantity, optional
        Per-pad pocket/track depth and taper depths, m. Default is 0.
    pocket_arc : float, pint.Quantity, optional
        Leading-edge-groove pocket arc, rad. Default is 0.
    pocket_axial_length : float, pint.Quantity, optional
        Leading-edge-groove pocket axial length, m. Default is 0.
    pad_E : float, pint.Quantity, optional
        Pad Young's modulus, Pa. Default is 206.8e9 (steel).
    pad_poisson : float, optional
        Pad Poisson ratio. Default is 0.3.
    pad_conductivity : float, pint.Quantity, optional
        Pad thermal conductivity, W/(m*K). Default is 50.
    pad_expansion, journal_expansion, shell_expansion : float, optional
        Thermal expansion coefficients, 1/K. Default is 1.17e-5 (steel).
    pad_density : float, optional
        Pad material density, kg/m**3. Default is 7830.
    pad_convection : float or array_like, optional
        Pad back-face convection coefficient, W/(m**2*K). A scalar is
        applied to every pad. Default is 735.903.
    edges_convection : float, optional
        Pad edge convection coefficient, W/(m**2*K). Default is 73.59.
    environment_temperature : float, pint.Quantity, optional
        Environment temperature, K. Default is 294.261.
    environment_convection : float, optional
        Sump-to-environment convection coefficient, W/(m**2*K).
        Default is 735.903.
    sump_convect_area : float, optional
        Sump convection area, m**2 (oil-ring-lubricated operation).
        Default is 0.
    house_diameter, pivot_diameter : float, pint.Quantity, optional
        Housing / pivot contact diameters for the Hertzian pivot models
        (``"ball_in_socket"``, ``"button"``), m; with a pivot deformation
        type they must satisfy ``0 < pivot_diameter < house_diameter``.
        Default is 0.
    pivot_stiffness : float, pint.Quantity, optional
        Pivot stiffness for ``"user_specified_stiffness"``, N/m; positive
        when ``deform_type`` includes pivot flexibility. Default is 0.
    crush_fit : float, pint.Quantity, optional
        Shell crush (shrink) fit, m. Default is 0.
    shell_id, shell_od : float, pint.Quantity, optional
        Shell inner/outer diameter for the shell thermal growth model, m.
        Default is 0.
    ambient_pressure_1, ambient_pressure_2 : float, pint.Quantity, optional
        Ambient pressures at the two axial edges, Pa, gauge (relative to the
        drain pressure). ``"axial_flow"``, ``"high_ambient_pressure"`` and
        the starved and oil-ring modes apply them to the film boundaries. In
        every mode the inter-pad grooves carry their mean,
        ``0.5 * (ambient_pressure_1 + ambient_pressure_2)``, which adds a net
        force on the journal unless the pad layout is symmetric. In
        ``"regular_flooded"`` operation the axial film edges stay at 0 and a
        nonzero value issues a warning. Default is 0.
    cavitation_pressure : float, pint.Quantity, optional
        Cavitation pressure, Pa. Default is 0.
    oil_supply_pressure : float, pint.Quantity, optional
        Lubricant supply pressure, Pa. Default is 0.
    reference_temperature : float, pint.Quantity, optional
        Reference (assembly) temperature for thermal growth, K.
        Default is 297.039.
    journal_temperature : float, pint.Quantity, optional
        Initial journal temperature estimate, K. Default is the supply
        temperature.
    probes : list of tuple, optional
        Temperature probes as ``(pad_number, theta_location, r_location)``:
        1-based pad number, circumferential position as % of the pad arc
        from the leading edge, and radial distance from the pad surface
        (m, accepts pint). Default is no probes.
    excitation_ratio : float, optional
        Whirl-to-rotation frequency ratio for the dynamic reduction, >= 0.
        Default is 1 (synchronous).
    initial_position : tuple of float, optional
        Initial journal position guess ``(x, y)`` as fractions of the
        radial clearance (held fixed for ``"match_eccentricity"``).
        Default is (0.15, -0.2).
    starvation_number : int, optional
        Starvation model parameter. Default is 1.
    hot_oil_lambda : float, optional
        Hot-oil carryover factor for the groove mixing model.
        Default is 0.8.
    relax_pressure, relax_temperature, relax_deformation, relax_pivot : float, optional
        Under-relaxation factors of the solver iterations.
        Default is 0.5 / 1 / 1 / 1.
    re_laminar, re_turbulent : float, optional
        Reynolds numbers bounding the laminar-turbulent transition.
        Default is 500 / 1000.
    num_processes : int, optional
        Solve the speed cases in ``num_processes`` worker processes
        instead of serially. Default is None (serial).
    on_failure : str, optional
        What to do with a case that does not converge (see Notes):
        ``"warn"`` leaves its speed out of the coefficient table with a
        :class:`~ross.bearings.film.loops.ConvergenceWarning`, ``"raise"``
        raises :class:`~ross.bearings.fluid_film.diagnostics.FluidFilmConvergenceError`,
        ``"keep"`` keeps every row and warns. Default is ``"warn"``.
    max_reynolds_solves : int, optional
        Work budget of each case, in Reynolds solves of the film equation.
        Default is None (400,000 per pad, counting at least 5 pads; about 4
        times the largest converging case measured, an 8-pad full-TEHD
        tilting pad). Each equilibrium solve within a case is also capped at
        4,000 solves per pad (at least 20,000).
    time_limit : float, pint.Quantity, optional
        Wall-clock limit of each case, s. Default is None (no limit).
    tag : str, optional
        A tag to name the element.

    Attributes
    ----------
    results : FluidFilmBearingResults
        Every solved case, converged or not: fields, solver outputs,
        coefficients, the per-case ``converged`` flags and ``convergence``
        summaries, and :attr:`operating_results` (also ``_results``; the
        ``plot_*`` and ``show_*`` methods are reachable on the element).

    Returns
    -------
    A FluidFilmBearing object.

    Raises
    ------
    ValueError
        Before any solve, when an input is outside its valid range (the
        message names the argument): non-positive or non-finite speeds,
        negative frequencies or ``excitation_ratio``, non-positive
        geometry, density or flow, preload or offset out of range,
        overlapping pads, or a pivot model without the data it needs.
    FluidFilmConvergenceError
        With ``on_failure="raise"``, when a case does not converge.

    Warns
    -----
    ConvergenceWarning
        With ``on_failure="warn"`` or ``"keep"``, when a case does not
        converge.

    Notes
    -----
    Defaults. :class:`ross.bearings.tilting_pad.TiltingPad` uses different
    defaults from this class: ``equilibrium_type="match_eccentricity"``
    (here ``"match_load"``), ``pad_conductivity=116`` (here 50),
    ``edges_convection=1500`` (here 73.59), ``relax_temperature=0.5``
    (here 1) and the mesh ``total_ex_film=30``, ``total_ez_film=30``,
    ``total_ey_pad=16`` (here 40, 20, 20). Results built on the defaults of
    one class are not comparable with the other's unless these arguments
    are passed explicitly.

    Pressures (ambient, cavitation, supply) are gauge pressures relative to
    the drain pressure.

    Convergence. A case is converged when every iteration of the solver
    (equilibrium Newton, pad tilt, turbulence, film and pad temperature,
    journal temperature, inlet mixing, deformation, pivot, sump and the
    starved film onset) ended below its tolerance and the case stayed
    within its work budget; the outputs must also be finite. The
    equilibrium test is the force balance ``|F_net| / |F_ext| < 2e-3``.
    ``results.convergence`` holds the per-loop iterations, residuals and
    tolerances of each case. A case that reaches ``max_reynolds_solves``
    (or a fixed cap per equilibrium solve, or ``time_limit``) stops with
    reason ``"budget"`` (``"time_limit"``) and its outputs are evaluated at
    the last Newton iterate of the journal position, with the temperature
    fields of the last completed thermal pass.

    Full 360 deg pad. When the first pad spans exactly 2 pi, the engine
    drops the circumferential pressure flow and the leading/trailing-edge
    pressure conditions, as for an annular oil seal: the film follows the
    short-bearing (Ocvirk) approximation, whose load at ``L/D = 1`` is about
    twice that of the finite bearing. Use an arc slightly below 2 pi (for
    example 359.9 deg) for a full journal solved with the finite-length
    Reynolds equation.

    Failed cases. With ``on_failure="warn"`` the speeds with a failed case
    are left out of the coefficient table (for a 2-D table, the whole speed
    row), so the element interpolates between the converged speeds; when
    no case converged every row is kept and the warning says the element
    must not be used. The failed cases stay in ``results`` and
    :attr:`operating_results` with ``converged=False``; :meth:`save` writes
    the coefficient table, so only the kept rows.

    Examples
    --------
    >>> from ross.bearings.fluid_film_bearing import fluid_film_bearing_example
    >>> bearing = fluid_film_bearing_example()
    >>> bearing.n_pads
    2
    >>> float(bearing.kxx[0]) > 1e8
    True
    """

    @check_units
    def __init__(
        self,
        n,
        speed=None,
        frequency=None,
        journal_diameter=None,
        radial_clearance=None,
        pad_thickness=None,
        pivot_angle=None,
        pad_arc=None,
        pad_axial_length=None,
        preload=None,
        offset=None,
        lubricant=None,
        oil_supply_temperature=None,
        oil_flow_v=None,
        weight=0,
        fxs_load=0,
        fys_load=0,
        bearing_type="fixed_geometry",
        operating_type="regular_flooded",
        thermal_type="full",
        temp_j_type="averaged_film_temperature",
        deform_type=None,
        equilibrium_type="match_load",
        sump_type="supply_temperature",
        pivot_type="user_specified_stiffness",
        total_ex_film=40,
        total_ey_film=30,
        total_ez_film=20,
        total_ey_pad=20,
        track_arc=None,
        track_axial_length=None,
        track_depth=None,
        taper_depth_le=None,
        taper_arc_le=None,
        taper_depth_te=None,
        taper_arc_te=None,
        pocket_arc=0,
        pocket_axial_length=0,
        pad_E=206.8e9,
        pad_poisson=0.3,
        pad_conductivity=50.0,
        pad_expansion=1.17e-5,
        pad_density=7830.0,
        journal_expansion=1.17e-5,
        shell_expansion=1.17e-5,
        pad_convection=735.9030318060636,
        edges_convection=73.59030318060636,
        environment_temperature=294.261,
        environment_convection=735.9030318060636,
        sump_convect_area=0,
        house_diameter=0,
        pivot_diameter=0,
        pivot_stiffness=0,
        crush_fit=0,
        shell_id=0,
        shell_od=0,
        ambient_pressure_1=0,
        ambient_pressure_2=0,
        cavitation_pressure=0,
        oil_supply_pressure=0,
        reference_temperature=297.03888888888889,
        journal_temperature=None,
        probes=None,
        excitation_ratio=1,
        initial_position=(0.15, -0.2),
        starvation_number=1,
        hot_oil_lambda=0.8,
        relax_pressure=0.5,
        relax_temperature=1,
        relax_deformation=1,
        relax_pivot=1,
        re_laminar=500,
        re_turbulent=1000,
        num_processes=None,
        on_failure="warn",
        max_reynolds_solves=None,
        time_limit=None,
        **kwargs,
    ):
        # ``locals()`` of a method that uses zero-argument ``super()`` also
        # contains the implicit ``__class__`` cell -- storing that would
        # rebind the instance's class, silently downcasting subclasses.
        for name, value in locals().items():
            if name not in ("self", "kwargs", "num_processes", "__class__"):
                setattr(self, name, value)

        self.time_limit = None if time_limit is None else _to_base_unit(time_limit, "s")
        self._validate_solver_controls()

        for name, valid in _TYPE_FLAGS.items():
            value = getattr(self, name)
            if value not in valid:
                raise ValueError(
                    f"{name} must be one of "
                    f"{sorted(str(v) for v in valid)}, not {value!r}"
                )

        for name in ("total_ex_film", "total_ey_film", "total_ez_film", "total_ey_pad"):
            if getattr(self, name) % 2 != 0:
                raise ValueError(f"{name} must be an even number")

        if speed is None or np.asarray(speed).size == 0:
            raise ValueError("speed must be informed")
        if oil_flow_v is None:
            raise ValueError("oil_flow_v not informed")
        if oil_supply_temperature is None:
            raise ValueError("oil_supply_temperature must be informed")
        if crush_fit and not (shell_id and shell_od):
            raise ValueError("a nonzero crush_fit requires shell_id and shell_od")

        self.pivot_angle = np.atleast_1d(np.asarray(pivot_angle, dtype=float))
        self.n_pads = self.pivot_angle.size
        for name in ("pad_arc", "pad_axial_length", "preload", "offset"):
            value = np.atleast_1d(np.asarray(getattr(self, name), dtype=float))
            if value.size != self.n_pads:
                raise ValueError(
                    f"{name} argument is inconsistent with number of pads."
                )
            setattr(self, name, value)
        for name in (
            "track_arc",
            "track_axial_length",
            "track_depth",
            "taper_depth_le",
            "taper_arc_le",
            "taper_depth_te",
            "taper_arc_te",
        ):
            value = getattr(self, name)
            if value is None:
                value = np.zeros(self.n_pads)
            else:
                value = np.atleast_1d(np.asarray(value, dtype=float))
                if value.size != self.n_pads:
                    raise ValueError(
                        f"{name} argument is inconsistent with number of pads."
                    )
            setattr(self, name, value)
        self.pad_convection = np.broadcast_to(
            np.atleast_1d(np.asarray(pad_convection, dtype=float)),
            (self.n_pads,),
        ).copy()

        self.lubricant_properties = _lubricant_properties(lubricant)

        if journal_temperature is None:
            self.journal_temperature = float(oil_supply_temperature)

        probes = [] if probes is None else probes
        self.probes = [
            (
                float(pad_number),
                float(theta_location),
                r.to("m").m if hasattr(r, "to") else float(r),
            )
            for pad_number, theta_location, r in probes
        ]

        self.speed_range = np.atleast_1d(np.asarray(speed, dtype=float))
        self.frequency_range = (
            None
            if frequency is None
            else np.atleast_1d(np.asarray(frequency, dtype=float))
        )

        coefficient_table = {}
        case_outputs = None
        table_speed = self.speed_range
        initial_time = final_time = None
        if kwargs.get("kxx") is None:
            self._validate_inputs()
            initial_time = time.time()
            case_outputs = self._solve_all(num_processes, self._engine_cases())
            final_time = time.time()
            convergence = [out.pop("convergence")[0] for out in case_outputs]
            converged = np.array([c["converged"] for c in convergence], dtype=bool)
            rows = self._table_rows(converged, convergence)
            table_speed = self.speed_range[rows]

            coefficients = ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy")
            coefficient_table = {
                name: np.array([out[name][0] for out in case_outputs], dtype=float)
                for name in coefficients
            }
            if self.frequency_range is not None:
                shape = (self.speed_range.size, self.frequency_range.size)
                coefficient_table = {
                    name: table.reshape(shape)[rows]
                    for name, table in coefficient_table.items()
                }
            else:
                coefficient_table = {
                    name: table[rows] for name, table in coefficient_table.items()
                }

        super().__init__(
            n=n,
            speed=table_speed,
            frequency=self.frequency_range,
            **coefficient_table,
            **kwargs,
        )

        if case_outputs is not None:
            case_speeds = self.speed_range
            case_frequencies = self.excitation_ratio * self.speed_range
            if self.frequency_range is not None:
                case_speeds = np.repeat(self.speed_range, self.frequency_range.size)
                case_frequencies = np.tile(self.frequency_range, self.speed_range.size)
            fields = [out.pop("fields")[0] for out in case_outputs]

            def case_values(name):
                return np.array([out[name][0] for out in case_outputs], dtype=float)

            self.results = self._results = FluidFilmBearingResults(
                frequency=case_speeds,
                pressure_fields=[f["pressure"] for f in fields],
                temperature_fields=[f["film_temperature"] for f in fields],
                film_thickness_fields=[f["film_thickness"] for f in fields],
                theta_grids=[f["theta"] for f in fields],
                z_grids=[f["axial_position"] for f in fields],
                leading_edge_angles=fields[0]["leading_edge_angle"],
                pad_temperature_fields=(
                    [f["pad_temperature"] for f in fields]
                    if "pad_temperature" in fields[0]
                    else None
                ),
                pad_radial_positions=fields[0].get("pad_radial_position"),
                outputs=case_outputs,
                kxx=case_values("kxx"),
                kxy=case_values("kxy"),
                kyx=case_values("kyx"),
                kyy=case_values("kyy"),
                cxx=case_values("cxx"),
                cxy=case_values("cxy"),
                cyx=case_values("cyx"),
                cyy=case_values("cyy"),
                initial_time=initial_time,
                final_time=final_time,
                converged=converged,
                convergence=convergence,
                whirl_frequency=case_frequencies,
                pad_forces=[f["pad_force"] for f in fields],
                bearing_data=self._bearing_data(),
            )

    @property
    def operating_results(self):
        """Operating quantities of every solved case, converged or not.

        One row per case; see
        :attr:`ross.bearings.bearing_results.FluidFilmBearingResults.operating_results`
        for the columns.

        Returns
        -------
        pandas.DataFrame
        """
        results = self.__dict__.get("_results")
        if results is None:
            raise AttributeError(_NO_OPERATING_RESULTS)
        return results.operating_results

    def _bearing_data(self):
        """Inputs the operating table needs besides the solver outputs."""
        from ross.bearings.film.lubricant import exponential_viscosity

        props = self.lubricant_properties
        return {
            "journal_diameter": float(self.journal_diameter),
            "radial_clearance": float(self.radial_clearance),
            "axial_length": float(self.pad_axial_length[0]),
            "viscosity_supply": float(
                exponential_viscosity(
                    props["viscosity1"],
                    props["viscosity2"],
                    props["temp1"],
                    props["temp2"],
                    float(self.oil_supply_temperature),
                )
            ),
            "density": props["lube_density"],
            "thermal_type": self.thermal_type,
            "load": (float(self.fxs_load), float(self.fys_load) - float(self.weight)),
        }

    def _validate_solver_controls(self):
        """Check ``on_failure``, ``max_reynolds_solves`` and ``time_limit``.

        Raises
        ------
        ValueError
            If ``on_failure`` is not one of ``"warn"``, ``"raise"``,
            ``"keep"``, or the budget or time limit is not positive.
        """
        if self.on_failure not in ON_FAILURE:
            raise ValueError(
                f"on_failure must be one of {list(ON_FAILURE)}, not {self.on_failure!r}"
            )
        budget = self.max_reynolds_solves
        if budget is not None and not (
            np.isfinite(float(budget)) and float(budget) == int(budget) >= 1
        ):
            raise ValueError(
                f"max_reynolds_solves must be a positive integer, got {budget!r}"
            )
        limit = self.time_limit
        if limit is not None and not (np.isfinite(limit) and limit > 0):
            raise ValueError(f"time_limit must be positive, got {limit!r} s")

    def _table_rows(self, converged, convergence):
        """Speed rows of the coefficient table, per ``on_failure``.

        Parameters
        ----------
        converged : numpy.ndarray of bool
            One flag per solved case, in case order.
        convergence : list of dict
            The convergence summary of each case.

        Returns
        -------
        numpy.ndarray of bool
            One flag per speed: True for a row of the coefficient table.

        Raises
        ------
        FluidFilmConvergenceError
            With ``on_failure="raise"``, when a case did not converge.
        """
        from ross.bearings.film.loops import ConvergenceWarning
        from ross.bearings.fluid_film.diagnostics import FluidFilmConvergenceError

        n_speeds = self.speed_range.size
        per_speed = converged.reshape(n_speeds, -1).all(axis=1)
        if converged.all():
            return per_speed
        failed = []
        for index in np.flatnonzero(~converged):
            speed = self.speed_range[index // (converged.size // n_speeds)]
            case = f"speed {speed:.6g} rad/s"
            if self.frequency_range is not None:
                frequency = self.frequency_range[index % self.frequency_range.size]
                case += f", frequency {frequency:.6g} rad/s"
            failed.append(f"{case} ({convergence[index]['message']})")
        failed = "; ".join(failed)
        if self.on_failure == "raise":
            raise FluidFilmConvergenceError(f"cases did not converge: {failed}")
        if self.on_failure == "keep" or not per_speed.any():
            note = (
                "every row is kept (on_failure='keep')"
                if self.on_failure == "keep"
                else "no speed converged, so every row is kept: the table holds the "
                "last iterates of failed solves, which are not bearing "
                "coefficients (do not use this element in a rotor analysis)"
            )
            warnings.warn(
                f"cases did not converge: {failed}; {note}",
                ConvergenceWarning,
                skip_file_prefixes=_INTERNAL_FILES,
            )
            return np.ones(n_speeds, dtype=bool)
        warnings.warn(
            f"cases did not converge: {failed}; speeds "
            f"{np.round(self.speed_range[~per_speed], 6).tolist()} rad/s are left out "
            "of the element coefficient table (it interpolates between the "
            "converged speeds; the failed cases stay in results and "
            "operating_results)",
            ConvergenceWarning,
            skip_file_prefixes=_INTERNAL_FILES,
        )
        return per_speed

    def _validate_inputs(self):
        """Check the solver inputs before any case is solved.

        Raises
        ------
        ValueError
            If an input is outside the range the solver can handle. The
            message names the argument and the offending value.

        Warns
        -----
        UserWarning
            If a load is given with ``equilibrium_type="match_eccentricity"``
            (the load is not used), or ambient pressures are given in
            ``"regular_flooded"`` operation (the axial film edges stay at 0
            while the grooves carry the mean ambient pressure).
        """
        speed = self.speed_range
        if not np.all(np.isfinite(speed) & (speed > 0)):
            raise ValueError(
                "speed must hold finite, positive values: the solver requires "
                f"nonzero speeds, got {speed[~(np.isfinite(speed) & (speed > 0))]} "
                "rad/s"
            )
        if self.frequency_range is not None:
            frequency = self.frequency_range
            bad = ~(np.isfinite(frequency) & (frequency >= 0))
            if np.any(bad):
                raise ValueError(
                    f"frequency must hold finite values >= 0, got {frequency[bad]} "
                    "rad/s"
                )
        if not (np.isfinite(self.excitation_ratio) and self.excitation_ratio >= 0):
            raise ValueError(
                "excitation_ratio must be finite and >= 0, got "
                f"{self.excitation_ratio!r}"
            )

        for name in (
            "journal_diameter",
            "radial_clearance",
            "pad_thickness",
            "oil_flow_v",
            "pad_density",
            "pad_arc",
            "pad_axial_length",
        ):
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"{name} must be informed")
            value = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(value) & (value > 0)):
                raise ValueError(
                    f"{name} must be finite and positive, got {value.tolist()}"
                )

        if not np.all(np.isfinite(self.pivot_angle)):
            raise ValueError(
                f"pivot_angle must be informed and finite, got "
                f"{self.pivot_angle.tolist()}"
            )
        if not np.all((self.preload >= 0) & (self.preload < 1)):
            raise ValueError(
                f"preload must satisfy 0 <= preload < 1 for every pad, got "
                f"{self.preload.tolist()}"
            )
        if not np.all((self.offset >= 0) & (self.offset <= 1)):
            raise ValueError(
                f"offset must satisfy 0 <= offset <= 1 for every pad, got "
                f"{self.offset.tolist()}"
            )
        self._validate_pad_layout()
        self._validate_pivots()

        load = {
            "weight": float(self.weight),
            "fxs_load": float(self.fxs_load),
            "fys_load": float(self.fys_load),
        }
        if self.equilibrium_type == "match_eccentricity" and any(load.values()):
            given = ", ".join(f"{k}={v:g} N" for k, v in load.items() if v)
            warnings.warn(
                "equilibrium_type='match_eccentricity' holds the journal at "
                f"initial_position and does not use the applied load ({given}); "
                "pass equilibrium_type='match_load' to solve the journal position "
                "for the load.",
                UserWarning,
                skip_file_prefixes=_INTERNAL_FILES,
            )
        if self.operating_type == "regular_flooded" and (
            self.ambient_pressure_1 or self.ambient_pressure_2
        ):
            warnings.warn(
                "operating_type='regular_flooded' keeps the axial film edges at "
                "gauge 0 (the drain pressure); ambient_pressure_1 and "
                "ambient_pressure_2 only set the inter-pad groove pressure, "
                "0.5 * (ambient_pressure_1 + ambient_pressure_2), whose net force "
                "on the journal cancels only for a symmetric pad layout. Use "
                "'axial_flow' or 'high_ambient_pressure' to apply them to the film "
                "edges, or pass gauge pressures (0 for an open drain).",
                UserWarning,
                skip_file_prefixes=_INTERNAL_FILES,
            )

    def _validate_pad_layout(self):
        """Check that the pads do not overlap and fit in one revolution.

        The leading edge of a pad is ``pivot_angle - offset * pad_arc`` (as in
        the solver) and its trailing edge is ``pad_arc`` further on. A single
        360-degree pad is allowed.

        Raises
        ------
        ValueError
            If the total arc exceeds 360 degrees or two pads overlap
            (tolerance 1e-9 rad).
        """
        tolerance = 1e-9
        total_arc = float(np.sum(self.pad_arc))
        if total_arc > 2 * np.pi + tolerance:
            raise ValueError(
                f"pad_arc: the pads cover {np.degrees(total_arc):.6g} deg in total, "
                "more than 360 deg"
            )
        leading = np.mod(self.pivot_angle - self.offset * self.pad_arc, 2 * np.pi)
        order = np.argsort(leading, kind="stable")
        for position, pad in enumerate(order):
            following = order[(position + 1) % order.size]
            next_leading = leading[following]
            if position == order.size - 1:
                next_leading += 2 * np.pi
            trailing = leading[pad] + self.pad_arc[pad]
            if trailing > next_leading + tolerance:
                raise ValueError(
                    f"pad_arc / pivot_angle: pad {pad + 1} (leading edge "
                    f"{np.degrees(leading[pad]):.6g} deg, arc "
                    f"{np.degrees(self.pad_arc[pad]):.6g} deg) overlaps pad "
                    f"{following + 1} (leading edge "
                    f"{np.degrees(leading[following]):.6g} deg); the leading edge "
                    "is pivot_angle - offset * pad_arc"
                )

    def _validate_pivots(self):
        """Check that the pivot flexibility model has the data it needs.

        Raises
        ------
        ValueError
            If ``deform_type`` includes pivot flexibility and
            ``"user_specified_stiffness"`` has ``pivot_stiffness <= 0``, or
            the Hertz pivots (``"ball_in_socket"``, ``"button"``) do not have
            ``0 < pivot_diameter < house_diameter``.
        """
        from ross.bearings.fluid_film.constants import PIVOT_FLEX_DEFORM_TYPES

        if self.deform_type not in PIVOT_FLEX_DEFORM_TYPES:
            return
        if self.pivot_type == "user_specified_stiffness":
            stiffness = float(self.pivot_stiffness)
            if not (np.isfinite(stiffness) and stiffness > 0):
                raise ValueError(
                    "pivot_stiffness must be finite and positive for "
                    "pivot_type='user_specified_stiffness' with deform_type="
                    f"{self.deform_type!r}, got {stiffness!r} N/m"
                )
        elif self.pivot_type in ("ball_in_socket", "button"):
            pivot_diameter = float(self.pivot_diameter)
            house_diameter = float(self.house_diameter)
            if not (0 < pivot_diameter < house_diameter < np.inf):
                raise ValueError(
                    f"pivot_diameter and house_diameter must satisfy 0 < "
                    f"pivot_diameter < house_diameter for pivot_type="
                    f"{self.pivot_type!r} with deform_type={self.deform_type!r} "
                    f"(Hertz contact), got pivot_diameter={pivot_diameter!r} m, "
                    f"house_diameter={house_diameter!r} m"
                )

    def __getattr__(self, name):
        if name.startswith("plot_") or name.startswith("show_"):
            return getattr(self._results, name)
        if name == "operating_results":
            # Python retries a property that raised AttributeError here.
            raise AttributeError(_NO_OPERATING_RESULTS)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def save(self, file):
        """Save the element as a plain coefficient-table BearingElement.

        The downgrade is deliberate (and applies to every subclass): the
        saved file holds the solved dynamic-coefficient table, so loading
        it restores the rotordynamic behavior instantly instead of
        re-running the solver. Re-create the object from its constructor
        to change the bearing model.

        Parameters
        ----------
        file : str or pathlib.Path
            File to write (created or updated).
        """
        self.save_coefficient_table(file)

    def _engine_cases(self):
        """List the solver cases of the coefficient table, in table order.

        Returns
        -------
        list of dict
            One :meth:`_engine_inputs` dict per rotor speed (1-D table) or per
            (speed, frequency) pair in row-major order (2-D table).
        """
        if self.frequency_range is None:
            return [self._engine_inputs(w) for w in self.speed_range]
        return [
            self._engine_inputs(w, excitation_ratio=f / w)
            for w in self.speed_range
            for f in self.frequency_range
        ]

    def _engine_inputs(self, speed, excitation_ratio=None):
        """Assemble the solver keyword arguments for one rotor speed.

        Parameters
        ----------
        speed : float
            Rotor speed, rad/s.
        excitation_ratio : float, optional
            Whirl-to-rotation ratio of the dynamic reduction. Default is the
            element's ``excitation_ratio``.

        Returns
        -------
        dict
            Keyword arguments for
            :func:`ross.bearings.fluid_film.driver.run_case`.
        """
        # imported here: ``import ross`` does not load the film core
        from ross.bearings.film.turbulence import REICHARDT_DELTA, REICHARDT_KAPPA

        if excitation_ratio is None:
            excitation_ratio = self.excitation_ratio
        xj, yj = self.initial_position
        inputs = {
            "frequency": float(speed),
            "field_outputs": True,
            "diagnostics": True,
            "max_reynolds_solves": self.max_reynolds_solves,
            "time_limit": self.time_limit,
            # meshes
            "total_e_x_film": int(self.total_ex_film),
            "total_e_y_film": int(self.total_ey_film),
            "total_e_z_film": int(self.total_ez_film),
            "total_e_y_pad": int(self.total_ey_pad),
            # model flags
            "bearing_type": self.bearing_type,
            "operating_type": self.operating_type,
            "thermal_type": self.thermal_type,
            "temp_j_type": self.temp_j_type,
            "deform_type": self.deform_type,
            "equilibrium_type": self.equilibrium_type,
            "sump_type": self.sump_type,
            "pivot_type": self.pivot_type,
            "ta_type": 0,
            # geometry
            "journal_diameter": float(self.journal_diameter),
            "radial_clearance": float(self.radial_clearance),
            "pad_thickness": float(self.pad_thickness),
            "pivot_angle": self.pivot_angle,
            "pad_arc": self.pad_arc,
            "pad_axial_length": self.pad_axial_length,
            "preload": self.preload,
            "offset": self.offset,
            "track_arc": self.track_arc,
            "track_axial_length": self.track_axial_length,
            "track_depth": self.track_depth,
            "taper_depth_le": self.taper_depth_le,
            "taper_arc_le": self.taper_arc_le,
            "taper_depth_te": self.taper_depth_te,
            "taper_arc_te": self.taper_arc_te,
            "pocket_arc": float(self.pocket_arc),
            "pocket_axial_length": float(self.pocket_axial_length),
            "k_rotate": np.zeros(self.n_pads),
            # lubricant
            **self.lubricant_properties,
            # pad material and thermal boundary
            "pad_young": float(self.pad_E),
            "pad_poisson": float(self.pad_poisson),
            "pad_conductivity": float(self.pad_conductivity),
            "pad_expansion": float(self.pad_expansion),
            "pad_density": float(self.pad_density),
            "journal_expansion": float(self.journal_expansion),
            "shell_expansion": float(self.shell_expansion),
            "pad_convection": self.pad_convection,
            "edges_convection": float(self.edges_convection),
            "environment_temperature": float(self.environment_temperature),
            "environment_convection": float(self.environment_convection),
            "sump_convect_area": float(self.sump_convect_area),
            # pivot / shell
            "house_diameter": float(self.house_diameter),
            "pivot_diameter": float(self.pivot_diameter),
            "pivot_stiffness": float(self.pivot_stiffness),
            "crush_fit": float(self.crush_fit),
            "shell_id": float(self.shell_id),
            "shell_od": float(self.shell_od),
            # operating conditions
            "weight": float(self.weight),
            "fxs_load": float(self.fxs_load),
            "fys_load": float(self.fys_load),
            "oil_flow_v": float(self.oil_flow_v),
            "ambient_pressure_1": float(self.ambient_pressure_1),
            "ambient_pressure_2": float(self.ambient_pressure_2),
            "cavitation_pressure": float(self.cavitation_pressure),
            "oil_supply_pressure": float(self.oil_supply_pressure),
            "oil_supply_temperature": float(self.oil_supply_temperature),
            "reference_temperature": float(self.reference_temperature),
            "journal_temperature": float(self.journal_temperature),
            # probes
            "probe_pad_number": np.array([p[0] for p in self.probes], dtype=int),
            "probe_theta": np.array([p[1] for p in self.probes], dtype=float),
            "r_location": np.array([p[2] for p in self.probes], dtype=float),
            # solution controls
            "excit_ratios": float(excitation_ratio),
            "xj": float(xj),
            "yj": float(yj),
            "starve_number": int(self.starvation_number),
            "hot_oil_lambda": float(self.hot_oil_lambda),
            "relax_p": float(self.relax_pressure),
            "relax_t": float(self.relax_temperature),
            "relax_d": float(self.relax_deformation),
            "relax_pivot": float(self.relax_pivot),
            "re_lower": float(self.re_laminar),
            "re_upper": float(self.re_turbulent),
            "weight_e": 0,
            "weight_h": 0,
            "reichardt_delta": REICHARDT_DELTA,
            "reichardt_kappa": REICHARDT_KAPPA,
            "turb_scal_fac_exp": 0.125,
        }
        return inputs

    def _solve_all(self, num_processes, per_case):
        """Solve the given engine cases; serial by default.

        Parameters
        ----------
        num_processes : int or None
            Number of worker processes; None or 1 solves serially.
        per_case : list of dict
            Engine keyword arguments, one dict per case (see
            :meth:`_engine_inputs`).

        Returns
        -------
        list of dict
            One solver output dict per case.
        """
        if num_processes is not None and num_processes > 1:
            with multiprocessing.Pool(num_processes) as pool:
                return pool.map(_solve_case, per_case)
        return [_solve_case(inputs) for inputs in per_case]

    def coefficients(self, speed, frequency=None):
        """Return the stiffness and damping matrices at a rotor speed.

        Coefficients are interpolated on the element's table; for a 2-D
        table the excitation ``frequency`` selects the whirl axis.

        Parameters
        ----------
        speed : float, pint.Quantity
            Rotor speed, rad/s.
        frequency : float, pint.Quantity, optional
            Excitation (whirl) frequency, rad/s. Default is the rotor speed
            (synchronous evaluation).

        Returns
        -------
        stiffness : tuple of float
            ``(kxx, kxy, kyx, kyy)``, N/m.
        damping : tuple of float
            ``(cxx, cxy, cyx, cyy)``, N*s/m.
        """
        stiffness = tuple(
            float(getattr(self, f"{name}_interpolated")(frequency, speed))
            for name in ("kxx", "kxy", "kyx", "kyy")
        )
        damping = tuple(
            float(getattr(self, f"{name}_interpolated")(frequency, speed))
            for name in ("cxx", "cxy", "cyx", "cyy")
        )
        return stiffness, damping


def fluid_film_bearing_example():
    """Create an example fluid-film journal bearing.

    A two-pad (two-axial-groove) fixed-geometry bearing on a coarse mesh
    with the isoviscous model, so it runs fast enough for documentation
    examples.

    Returns
    -------
    A FluidFilmBearing object.

    Examples
    --------
    >>> from ross.bearings.fluid_film_bearing import fluid_film_bearing_example
    >>> bearing = fluid_film_bearing_example()
    >>> bearing.n_pads
    2
    """
    return FluidFilmBearing(
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
        oil_flow_v=Q_(26.4172, "gallon/min"),
        weight=Q_(25361.8, "lbf"),
        thermal_type=None,
        total_ex_film=20,
        total_ey_film=10,
        total_ez_film=10,
        total_ey_pad=10,
    )
