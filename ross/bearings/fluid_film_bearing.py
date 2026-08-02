"""Shared base class for journal bearings solved by the fluid-film engine.

:class:`FluidFilmBearing` connects the internal thermo-elasto-hydrodynamic
solver (:mod:`ross.bearings.fluid_film`) to the ROSS element interface: it
resolves the lubricant, assembles the solver inputs, runs one solver case
per operating frequency (serially by default, in parallel on request),
builds the dynamic-coefficient table consumed by
:class:`ross.BearingElement`, and exposes the solved fields through a
results object with the standard ``plot_*`` / ``show_*`` methods.

Configuration classes (plain journal, elliptical, multi-lobe, pressure dam,
tilting pad, ...) subclass this and only translate their user-facing
geometry into the per-pad arrays this class consumes.
"""

import multiprocessing
import time

import numpy as np

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
from ross.units import Q_, check_units

__all__ = ["FluidFilmBearing", "fluid_film_bearing_example"]

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
    from ross.bearings.lubricants import lubricants_dict

    if isinstance(lubricant, str):
        try:
            properties = lubricants_dict[lubricant]
        except KeyError:
            raise ValueError(
                f"lubricant must be one of {sorted(lubricants_dict)} or a "
                f"dict of properties, not {lubricant!r}"
            )
    elif isinstance(lubricant, dict):
        properties = lubricant
    else:
        raise TypeError(
            "lubricant must be a lubricant name (str) or a dict of "
            f"properties, not {type(lubricant).__name__}"
        )

    missing = [name for name in _LUBRICANT_FIELDS.values() if name not in properties]
    if missing:
        raise ValueError(f"lubricant is missing properties: {missing}")
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
        frequency.

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

    For every entry of ``frequency`` the engine solves the journal
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
    frequency : array_like, pint.Quantity
        Operating frequencies, rad/s. One solver case runs per entry.
    journal_diameter : float, pint.Quantity
        Journal diameter, m.
    radial_clearance : float, pint.Quantity
        Radial (bearing-set) clearance, m.
    pad_thickness : float, pint.Quantity
        Radial pad thickness, m.
    pivot_angle : array_like, pint.Quantity
        Per-pad pivot (or arc-center) angular position, rad.
    pad_arc : array_like, pint.Quantity
        Per-pad arc length, rad.
    pad_axial_length : array_like, pint.Quantity
        Per-pad axial length, m.
    preload : array_like
        Per-pad preload factor (0 for a cylindrical land).
    offset : array_like
        Per-pad pivot offset fraction (0.5 = centered).
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
        applied load; ``"match_eccentricity"`` holds ``initial_position``.
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
        Housing / pivot contact diameters for the Hertzian pivot models, m.
        Default is 0.
    pivot_stiffness : float, pint.Quantity, optional
        Pivot stiffness for ``"user_specified_stiffness"``, N/m.
        Default is 0.
    crush_fit : float, pint.Quantity, optional
        Shell crush (shrink) fit, m. Default is 0.
    shell_id, shell_od : float, pint.Quantity, optional
        Shell inner/outer diameter for the shell thermal growth model, m.
        Default is 0.
    ambient_pressure_1, ambient_pressure_2 : float, pint.Quantity, optional
        Ambient (edge) pressures, Pa. Default is 0.
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
        Whirl-to-rotation frequency ratio for the dynamic reduction.
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
        Solve the frequency cases in ``num_processes`` worker processes
        instead of serially. Default is None (serial).
    tag : str, optional
        A tag to name the element.

    Returns
    -------
    A FluidFilmBearing object.

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
        **kwargs,
    ):
        for name, value in locals().items():
            if name not in ("self", "kwargs", "num_processes"):
                setattr(self, name, value)

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

        if frequency is None or np.asarray(frequency).size == 0:
            raise ValueError("frequency must be informed")
        if oil_flow_v is None:
            raise ValueError("oil_flow_v not informed")
        if oil_supply_temperature is None:
            raise ValueError("oil_supply_temperature must be informed")

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

        self.frequency_range = np.atleast_1d(np.asarray(frequency, dtype=float))

        coefficient_table = {}
        case_outputs = None
        initial_time = final_time = None
        if kwargs.get("kxx") is None:
            initial_time = time.time()
            case_outputs = self._solve_all(num_processes)
            final_time = time.time()

            coefficients = ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy")
            coefficient_table = {
                name: np.array([out[name][0] for out in case_outputs], dtype=float)
                for name in coefficients
            }

        super().__init__(
            n=n,
            frequency=self.frequency_range,
            **coefficient_table,
            **kwargs,
        )

        if case_outputs is not None:
            fields = [out.pop("fields")[0] for out in case_outputs]
            self._results = FluidFilmBearingResults(
                frequency=self.frequency_range,
                pressure_fields=[f["pressure"] for f in fields],
                temperature_fields=[f["film_temperature"] for f in fields],
                film_thickness_fields=[f["film_thickness"] for f in fields],
                theta_grids=[f["theta"] for f in fields],
                z_grids=[f["axial_position"] for f in fields],
                leading_edge_angles=fields[0]["leading_edge_angle"],
                outputs=case_outputs,
                kxx=self.kxx,
                kxy=self.kxy,
                kyx=self.kyx,
                kyy=self.kyy,
                cxx=self.cxx,
                cxy=self.cxy,
                cyx=self.cyx,
                cyy=self.cyy,
                initial_time=initial_time,
                final_time=final_time,
            )

    def __getattr__(self, name):
        if name.startswith("plot_") or name.startswith("show_"):
            return getattr(self._results, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def _engine_inputs(self, frequency):
        """Assemble the solver keyword arguments for one frequency.

        Parameters
        ----------
        frequency : float
            Shaft speed, rad/s.

        Returns
        -------
        dict
            Keyword arguments for
            :func:`ross.bearings.fluid_film.driver.run_case`.
        """
        xj, yj = self.initial_position
        inputs = {
            "frequency": float(frequency),
            "field_outputs": True,
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
            "excit_ratios": float(self.excitation_ratio),
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
            "reichardt_delta": 8.8,
            "reichardt_kappa": 0.4,
            "turb_scal_fac_exp": 0.125,
        }
        return inputs

    def _solve_all(self, num_processes):
        """Solve every frequency case; serial by default.

        Parameters
        ----------
        num_processes : int or None
            Number of worker processes; None or 1 solves serially.

        Returns
        -------
        list of dict
            One solver output dict per frequency.
        """
        per_case = [self._engine_inputs(f) for f in self.frequency_range]
        if num_processes is not None and num_processes > 1:
            with multiprocessing.Pool(num_processes) as pool:
                return pool.map(_solve_case, per_case)
        return [_solve_case(inputs) for inputs in per_case]

    def coefficients(self, frequency):
        """Return the stiffness and damping matrices at a frequency.

        Coefficients are interpolated on the element's frequency table.

        Parameters
        ----------
        frequency : float, pint.Quantity
            Frequency, rad/s.

        Returns
        -------
        stiffness : tuple of float
            ``(kxx, kxy, kyx, kyy)``, N/m.
        damping : tuple of float
            ``(cxx, cxy, cyx, cyy)``, N*s/m.
        """
        stiffness = tuple(
            float(getattr(self, f"{name}_interpolated")(frequency))
            for name in ("kxx", "kxy", "kyx", "kyy")
        )
        damping = tuple(
            float(getattr(self, f"{name}_interpolated")(frequency))
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
        frequency=Q_([900], "RPM"),
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
