"""Single source of the domain's units.

Before Phase 1 this dictionary existed twice, word for word: in app.py and in
frontend/app.js. Any drift between the copies produced an exported Python
script that did not reproduce what the screen was showing.
"""

# The unit each field is typed in, by ROSS class. ROSS works in SI
# internally; the conversion happens on the way in.
UNITS_MAPPING = {
    "Material": {"rho": "kg/m**3", "E": "N/m**2", "G_s": "N/m**2"},
    "ShaftElement": {"L": "mm", "idl": "mm", "odl": "mm", "idr": "mm", "odr": "mm"},
    "DiskElement": {"m": "kg", "Id": "kg*m**2", "Ip": "kg*m**2"},
    "GearElement": {
        "m": "kg",
        "Id": "kg*m**2",
        "Ip": "kg*m**2",
        "base_diameter": "mm",
        "pitch_diameter": "mm",
        "pr_angle": "deg",
        "helix_angle": "deg",
        "bore_diameter": "mm",
    },
    "GearElementTVMS": {
        "width": "mm",
        "bore_diameter": "mm",
        "module": "mm",
        "pr_angle": "deg",
        "helix_angle": "deg",
    },
    "CouplingElement": {
        "m_l": "kg",
        "m_r": "kg",
        "Ip_l": "kg*m**2",
        "Ip_r": "kg*m**2",
        "Id_l": "kg*m**2",
        "Id_r": "kg*m**2",
        "o_d": "mm",
        "L": "mm",
    },
    "BearingElement": {
        "kxx": "N/m",
        "kxy": "N/m",
        "kyx": "N/m",
        "kyy": "N/m",
        "kzz": "N/m",
        "cxx": "N*s/m",
        "cxy": "N*s/m",
        "cyx": "N*s/m",
        "cyy": "N*s/m",
        "czz": "N*s/m",
        "mxx": "kg",
        "mxy": "kg",
        "myx": "kg",
        "myy": "kg",
        "mzz": "kg",
        "frequency": "RPM",
    },
    "BallBearingElement": {},
    "RollerBearingElement": {},
    "MagneticBearingElement": {},
    "CylindricalBearing": {
        "speed": "RPM",
        "weight": "N",
        "bearing_length": "mm",
        "journal_diameter": "mm",
        "radial_clearance": "mm",
        "oil_viscosity": "Pa*s",
    },
    "PlainJournal": {
        "pad_axial_length": "mm",
        "journal_diameter": "mm",
        "radial_clearance": "mm",
        "pad_arc": "deg",
        "oil_supply_temperature": "degC",
        "frequency": "RPM",
        "fxs_load": "N",
        "fys_load": "N",
        "oil_supply_pressure": "Pa",
        "pad_thickness": "mm",
    },
    "SqueezeFilmDamper": {
        "frequency": "RPM",
        "axial_length": "mm",
        "journal_diameter": "mm",
        "radial_clearance": "mm",
    },
    "ThrustPad": {
        "pad_inner_radius": "mm",
        "pad_outer_radius": "mm",
        "pad_pivot_radius": "mm",
        "pad_arc": "deg",
        "pivot_angle": "deg",
        "oil_supply_temperature": "degC",
        "frequency": "RPM",
        "radial_inclination_angle": "rad",
        "circumferential_inclination_angle": "rad",
        "initial_film_thickness": "mm",
        "axial_load": "N",
    },
    "TiltingPad": {
        "journal_diameter": "mm",
        "pad_axial_length": "mm",
        "pad_thickness": "mm",
        "pad_arc": "deg",
        "radial_clearance": "mm",
        "pivot_angle": "deg",
        "oil_supply_temperature": "degC",
        "journal_temperature": "degC",
        "frequency": "RPM",
        "xj": "mm",
        "yj": "mm",
        "attitude_angle": "rad",
        "fxs_load": "N",
        "fys_load": "N",
    },
    "SealElement": {
        "kxx": "N/m",
        "kxy": "N/m",
        "kyx": "N/m",
        "kyy": "N/m",
        "kzz": "N/m",
        "cxx": "N*s/m",
        "cxy": "N*s/m",
        "cyx": "N*s/m",
        "cyy": "N*s/m",
        "czz": "N*s/m",
        "mxx": "kg",
        "mxy": "kg",
        "myx": "kg",
        "myy": "kg",
        "mzz": "kg",
        "frequency": "RPM",
    },
    "HolePatternSeal": {
        "shaft_diameter": "mm",
        "radial_clearance": "mm",
        "axial_length": "mm",
        "cell_length": "mm",
        "cell_width": "mm",
        "cell_depth": "mm",
        "inlet_pressure": "Pa",
        "outlet_pressure": "Pa",
        "inlet_temperature": "degC",
        "frequency": "RPM",
    },
    "LabyrinthSeal": {
        "shaft_diameter": "mm",
        "radial_clearance": "mm",
        "pitch": "mm",
        "tooth_height": "mm",
        "tooth_width": "mm",
        "inlet_pressure": "Pa",
        "outlet_pressure": "Pa",
        "inlet_temperature": "degC",
        "frequency": "RPM",
    },
    "HybridSeal": {
        "shaft_diameter": "mm",
        "inlet_pressure": "Pa",
        "outlet_pressure": "Pa",
        "inlet_temperature": "degC",
        "frequency": "RPM",
    },
    "PointMass": {"m": "kg", "mx": "kg", "my": "kg", "mz": "kg"},
}

# Units offered in the selector beside each field. The first one is default.
UNIT_ALTERNATIVES = {
    "kg/m**3": ["kg/m**3", "g/cm**3", "lb/in**3"],
    "N/m**2": ["N/m**2", "Pa", "MPa", "GPa", "psi", "bar"],
    "Pa": ["Pa", "MPa", "bar", "psi"],
    "m": ["m", "mm", "cm", "in"],
    "mm": ["mm", "m", "cm", "in"],
    "kg": ["kg", "g", "lb"],
    "kg*m**2": ["kg*m**2", "lb*in**2", "g*cm**2"],
    "deg": ["deg", "rad"],
    "rad": ["rad", "deg"],
    "N/m": ["N/m", "N/mm", "lbf/in"],
    "N*s/m": ["N*s/m", "lbf*s/in"],
    "RPM": ["RPM", "rad/s", "Hz"],
    "N": ["N", "lbf", "kN"],
    "Pa*s": ["Pa*s", "cP"],
    "degC": ["degC", "kelvin", "degF"],
    "l/min": ["l/min", "m**3/s"],
    "rad/s": ["rad/s", "RPM", "Hz"],
}

# Parameters ROSS expects as integers, whatever the user typed (the interface
# sends everything as text).
INT_PARAMETERS = frozenset(
    {
        "n_pad",
        "n_pads",
        "n_theta",
        "n_radial",
        "n_teeth",
        "n_rollers",
        "n_balls",
        "nx",
        "nz",
        "nr_pad",
        "max_inlet_iterations",
        "max_jtemp_iter",
        "max_iterations",
        "elements_circumferential",
        "elements_axial",
        "n_link",
        "n_l",
        "n_r",
        "total_ex_film",
        "total_ez_film",
        "total_ey_pad",
    }
)

# Historical note: PLOT_ONLY_KEYS used to live here -- the list of parameters
# that only affected the drawing. It was a global approximation of a property
# that is per-analysis: `frequency_type` is a computation parameter in Campbell
# (run_campbell takes it) and a plotting one in Modal (plot_mode_2d takes it),
# and a single list cannot be right about both. Since Phase 2 each runner
# declares its own in PLOT_PARAMS, and a test checks that none of them reaches
# the computation. See services/analysis/base.py.


def unit_for(ross_class, parameter):
    """Return the input unit for a parameter, or None if it is dimensionless."""
    return UNITS_MAPPING.get(ross_class, {}).get(parameter)


def alternatives_for(unit):
    """Return the unit options offered for a given default unit."""
    return list(UNIT_ALTERNATIVES.get(unit, []))
