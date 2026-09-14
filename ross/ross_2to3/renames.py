"""Single source of truth for the ROSS 2 to ROSS 3 parameter renames.

Every table in this module is consumed by the ``ross_2to3`` converters
(:mod:`ross.ross_2to3.models` for saved rotor files and
:mod:`ross.ross_2to3.scripts` for Python sources) and rendered by
:func:`migration_table_rst` into the migration guide of the release notes,
so a rename registered here is documented and converted at once.
"""

from collections import namedtuple

Change = namedtuple("Change", "new value values note", defaults=(None, None, None, ""))
Change.__doc__ = """Describe what happens to one constructor parameter.

Attributes
----------
new : str or None
    Name of the parameter in ROSS 3, or None if it was removed.
value : str or None
    Key of the value conversion to apply (see ``VALUE_CONVERSIONS``).
values : dict or None
    Mapping of old to new string values for enumerated parameters.
note : str
    Free text shown in the report and in the migration table.
"""


def renamed(new, value=None, values=None, note=""):
    """Build the Change of a parameter that keeps existing under a new name."""
    return Change(new, value, values, note)


def removed(note):
    """Build the Change of a parameter that has no replacement."""
    return Change(None, None, None, note)


def revalued(values, note=""):
    """Build the Change of a parameter whose name stays but whose values map."""
    return Change(None, None, values, note)


VALUE_CONVERSIONS = {
    "double": "value ×2, radius to diameter",
    "deg_to_rad": "plain numbers were degrees, now radians",
    "degc_to_kelvin": "plain numbers were degrees Celsius, now kelvin",
    "int_to_bool": "0/1 becomes False/True",
}

DOUBLE = "double"
DEG_TO_RAD = "deg_to_rad"
DEGC_TO_KELVIN = "degc_to_kelvin"
INT_TO_BOOL = "int_to_bool"

SPEED = renamed("speed")

_SOLVER_KNOB = "removed: the solver owns its iteration strategy and tolerances"

FLUID_FILM_RENAMES = {
    "frequency": SPEED,
}

PLAIN_JOURNAL_RENAMES = {
    "frequency": SPEED,
    "axial_length": renamed("pad_axial_length"),
    "journal_radius": renamed("journal_diameter", value=DOUBLE),
    "n_pad": renamed("n_pads"),
    "pad_arc_length": renamed("pad_arc", value=DEG_TO_RAD),
    "reference_temperature": renamed("oil_supply_temperature", value=DEGC_TO_KELVIN),
    "initial_guess": renamed(
        "initial_position",
        note="now the (x, y) journal position as fractions of the radial clearance",
    ),
    "elements_circumferential": renamed("total_ex_film"),
    "elements_axial": renamed("total_ez_film"),
    "operating_type": revalued(
        {"flooded": "regular_flooded", "starvation": "starved_condition_even"},
        note="values follow the fluid-film engine vocabulary",
    ),
    "geometry": removed(
        "removed: use MultiLobeBearing or EllipticalBearing for non-circular bores"
    ),
    "model_type": removed("removed: the engine is always thermo-hydro-dynamic"),
    "sommerfeld_type": removed("removed: the Sommerfeld number is reported directly"),
    "method": removed(
        "removed: the coefficients come from a single perturbation route"
    ),
    "groove_factor": removed("removed: groove mixing is set by hot_oil_lambda"),
}

TILTING_PAD_RENAMES = {
    "frequency": SPEED,
    "pre_load": renamed("preload"),
    "nx": renamed("total_ex_film", note="must be even"),
    "nz": renamed("total_ez_film"),
    "nr_pad": renamed("total_ey_pad"),
    "load": renamed(
        "fxs_load, fys_load", note="the [fx, fy] pair is split into two arguments"
    ),
    "hot_oil_carry_over": renamed("hot_oil_lambda"),
    "k_pad": renamed("pad_conductivity"),
    "h_edge": renamed("edges_convection"),
    "relax_t": renamed("relax_temperature"),
    "journal_temperature": renamed("journal_temperature", value=DEGC_TO_KELVIN),
    "equilibrium_type": revalued({"determine_eccentricity": "match_load"}),
    "initial_pads_angles": removed(_SOLVER_KNOB),
    "solver_options": removed(_SOLVER_KNOB),
    "inlet_temperature_tolerance": removed(_SOLVER_KNOB),
    "max_inlet_iterations": removed(_SOLVER_KNOB),
    "h_sump": removed(_SOLVER_KNOB),
    "max_jtemp_iter": removed(_SOLVER_KNOB),
    "jtemp_error": removed(_SOLVER_KNOB),
    "max_relax_change": removed(_SOLVER_KNOB),
}

THRUST_PAD_RENAMES = {
    "frequency": SPEED,
    "n_pad": renamed("n_pads"),
    "pad_arc_length": renamed("pad_arc"),
    "angular_pivot_position": renamed("pivot_angle"),
}

SQUEEZE_FILM_DAMPER_RENAMES = {
    "journal_radius": renamed("journal_diameter", value=DOUBLE),
}

_GAS_RENAMES = {
    "frequency": SPEED,
    "shaft_radius": renamed("shaft_diameter", value=DOUBLE),
    "molar": renamed("molar_mass"),
}

LABYRINTH_SEAL_RENAMES = {
    **_GAS_RENAMES,
    "tz": renamed("reference_temperatures"),
    "muz": renamed("reference_viscosities"),
    "iopt1": renamed("use_jenny_kanki", value=INT_TO_BOOL),
    "nprt": removed("removed: printing is controlled by print_results"),
    "analz": removed("removed: leakage and dynamic coefficients are always computed"),
}

HOLE_PATTERN_SEAL_RENAMES = {
    **_GAS_RENAMES,
    "length": renamed("axial_length"),
    "roughness": renamed("relative_roughness"),
    "whirl_ratio": renamed("excitation_ratio"),
    "entr_coef": renamed("entrance_loss_coefficient"),
    "exit_coef": renamed("exit_loss_coefficient"),
    "rlx_factor": renamed("relaxation_factor"),
    "b_suther": renamed("sutherland_b"),
    "s_suther": renamed("sutherland_s"),
}

HYBRID_SEAL_RENAMES = dict(_GAS_RENAMES)

HYBRID_SEAL_NESTED = {
    "hole_pattern_parameters": "HolePatternSeal",
    "labyrinth_parameters": "LabyrinthSeal",
}

CLASS_RENAMES = {
    "BearingElement": {"frequency": SPEED},
    "SealElement": {"frequency": SPEED},
    "ST_BearingElement": {"frequency": SPEED},
    "FluidFilmBearing": FLUID_FILM_RENAMES,
    "FixedGeometryBearing": FLUID_FILM_RENAMES,
    "PartialArcBearing": FLUID_FILM_RENAMES,
    "EllipticalBearing": FLUID_FILM_RENAMES,
    "OffsetHalvesBearing": FLUID_FILM_RENAMES,
    "MultiLobeBearing": FLUID_FILM_RENAMES,
    "PressureDamBearing": FLUID_FILM_RENAMES,
    "PlainJournal": PLAIN_JOURNAL_RENAMES,
    "TiltingPad": TILTING_PAD_RENAMES,
    "ThrustPad": THRUST_PAD_RENAMES,
    "SqueezeFilmDamper": SQUEEZE_FILM_DAMPER_RENAMES,
    "LabyrinthSeal": LABYRINTH_SEAL_RENAMES,
    "HolePatternSeal": HOLE_PATTERN_SEAL_RENAMES,
    "HybridSeal": HYBRID_SEAL_RENAMES,
    "MultiRotor": {
        "square_stiffness_amplitude_ratio": removed(
            "removed: the mesh stiffness is described by the Mesh class"
        ),
    },
    "Mesh": {
        "square_stiffness_amplitude_ratio": removed(
            "removed: the mesh stiffness is described by the Mesh class"
        ),
    },
}

METHOD_RENAMES = {
    "run_unbalance_response": {"frequency": renamed("speed_range")},
    "run_ucs": {"bearing_frequency_range": renamed("bearing_speed_range")},
}

MOVED_MODULES = {
    "ross.gear_element": "ross.multi_rotor.gear_element",
    "ross.multi_rotor": "ross.multi_rotor.multi_rotor",
    "ross.bearings.magnetic.controllers": "ross.bearings.magnetic.amb_controllers",
}

REMOVED_MODULES = {
    "ross.bearings.fluid_flow": "use PlainJournal (TEHD) or CylindricalBearing (analytical)",
}

REMOVED_NAMES = {
    "BearingFluidFlow": "use PlainJournal or CylindricalBearing",
    "FluidFlow": "use PlainJournal or CylindricalBearing",
    "PlainJournalResults": "FluidFilmBearingResults is created by the bearing",
    "TiltingPadResults": "FluidFilmBearingResults is created by the bearing",
    "from_fluid_flow": "build the coefficient arrays with PlainJournal instead",
}

SOLVER_BEARING_CLASSES = {
    "FluidFilmBearing",
    "FixedGeometryBearing",
    "PartialArcBearing",
    "EllipticalBearing",
    "OffsetHalvesBearing",
    "MultiLobeBearing",
    "PressureDamBearing",
    "PlainJournal",
    "TiltingPad",
    "ThrustPad",
}

FREQUENCY_TABLE_CLASSES = {"SqueezeFilmDamper", "MagneticBearingElement"}

SEAL_CLASSES = {"LabyrinthSeal", "HolePatternSeal", "HybridSeal"}

REORDERED_CLASSES = SOLVER_BEARING_CLASSES | SEAL_CLASSES | {"SqueezeFilmDamper"}

COEFFICIENT_KEYS = {
    f"{kind}{axes}"
    for kind in ("k", "c", "m")
    for axes in ("xx", "xy", "yx", "yy", "zz")
}

TABLE_KEYS = COEFFICIENT_KEYS | {
    "n",
    "speed",
    "frequency",
    "tag",
    "n_link",
    "scale_factor",
    "color",
    "interpolation",
    "seal_leakage",
}


def describe(change):
    """Return the human readable target of a Change for reports and docs."""
    if change.new is None and change.values is None:
        return change.note or "removed"
    if change.new is None:
        pairs = ", ".join(
            f"``{old}`` becomes ``{new}``" for old, new in change.values.items()
        )
        return f"same name; {pairs}"
    text = ", ".join(f"``{name.strip()}``" for name in change.new.split(","))
    details = [
        detail
        for detail in (VALUE_CONVERSIONS.get(change.value), change.note)
        if detail
    ]
    if details:
        text += f" ({'; '.join(details)})"
    return text


def migration_table_rst():
    """Render the class rename tables as a reStructuredText list-table.

    Returns
    -------
    str
        The table used in the version 3.0.0 release notes.
    """
    rows = []
    for cls, table in CLASS_RENAMES.items():
        if table is FLUID_FILM_RENAMES and cls != "FluidFilmBearing":
            continue
        for old, change in table.items():
            rows.append((cls, old, describe(change)))
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 22 28 50",
        "",
        "   * - Class",
        "     - ROSS 2",
        "     - ROSS 3",
    ]
    for cls, old, new in rows:
        lines += [f"   * - ``{cls}``", f"     - ``{old}``", f"     - {new}"]
    return "\n".join(lines) + "\n"
