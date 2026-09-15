import ast
import inspect
import json
from pathlib import Path

import pytest
import toml
from numpy.testing import assert_allclose

import ross as rs
from ross.ross_2to3 import (
    Report,
    convert_model_data,
    convert_model_text,
    convert_notebook,
    convert_source,
    migration_table_rst,
)
from ross.ross_2to3.cli import main
from ross.ross_2to3.models import check_model_file, is_model
from ross.ross_2to3.renames import CLASS_RENAMES
from ross.ross_2to3.report import CHANGED, CHECK, ERROR, MANUAL, SKIPPED, VERIFIED

V2_DATA = Path(__file__).parent / "data" / "v2"

V2_FILES = [
    "bearing.toml",
    "seal.toml",
    "sfd.toml",
    "laby.toml",
    "holep.toml",
    "hybrid.toml",
    "thrust.toml",
    "plain.toml",
    "tilting.toml",
    "rotor_seals.toml",
    "rotor_example.json",
]

PLAIN_JOURNAL_V2 = """
from ross.bearings.plain_journal import PlainJournal
from ross.units import Q_

bearing = PlainJournal(
    n=3,
    axial_length=0.263144,
    journal_radius=0.2,
    radial_clearance=1.95e-4,
    elements_circumferential=11,
    elements_axial=3,
    n_pad=2,
    pad_arc_length=176,
    preload=0,
    geometry="circular",
    reference_temperature=50,
    frequency=Q_([900], "RPM"),
    fxs_load=0,
    fys_load=-112814.91,
    groove_factor=[0.52, 0.48],
    lubricant="ISOVG32",
    sommerfeld_type=2,
    initial_guess=[0.1, -0.1],
    method="perturbation",
    operating_type="flooded",
    oil_supply_pressure=0,
    oil_flow_v=Q_(37.86, "l/min"),
)
"""

TILTING_PAD_V2 = """
import ross as rs
from ross.units import Q_

bearing = rs.TiltingPad(
    n=1,
    frequency=Q_([3000], "RPM"),
    equilibrium_type="determine_eccentricity",
    journal_diameter=101.6e-3,
    radial_clearance=74.9e-6,
    pad_thickness=12.7e-3,
    pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
    pad_arc=Q_([60] * 5, "deg"),
    pad_axial_length=Q_([50.8e-3] * 5, "m"),
    pre_load=[0.5] * 5,
    offset=[0.5] * 5,
    lubricant="ISOVG32",
    oil_supply_temperature=Q_(40, "degC"),
    eccentricity=0.35,
    attitude_angle=Q_(287.5, "deg"),
    load=[8.8405e02, -2.6704e03],
    nx=12,
    nz=8,
    nr_pad=15,
    hot_oil_carry_over=0.8,
    k_pad=116.0,
    h_edge=1500.0,
    relax_t=0.5,
    solver_options={"xtol": 1e-3},
    journal_temperature=25.0,
)
"""

THRUST_PAD_V2 = """
import ross as rs
from ross.units import Q_

bearing = rs.ThrustPad(
    n=1,
    pad_inner_radius=Q_(1150, "mm"),
    pad_outer_radius=Q_(1725, "mm"),
    pad_pivot_radius=Q_(1442.5, "mm"),
    pad_arc_length=Q_(26, "deg"),
    angular_pivot_position=Q_(15, "deg"),
    oil_supply_temperature=Q_(40, "degC"),
    lubricant="ISOVG68",
    n_pad=12,
    n_theta=10,
    n_radial=10,
    frequency=Q_([90], "RPM"),
    equilibrium_position_mode="calculate",
    axial_load=13.320e6,
)
"""

SQUEEZE_FILM_DAMPER_V2 = """
import ross as rs
Q_ = rs.units.Q_
SFD = rs.SqueezeFilmDamper(
    n=0,
    frequency=Q_([18600, 20000, 22000], "rpm"),
    axial_length=Q_(0.9, "inches"),
    journal_radius=Q_(2.55, "inches"),
    radial_clearance=Q_(0.003, "inches"),
    eccentricity_ratio=0.5,
    lubricant="ISOVG32",
    geometry="groove",
    cavitation=True,
)
"""

LABYRINTH_SEAL_V2 = """
from ross.seals.labyrinth_seal import LabyrinthSeal
from ross.units import Q_
seal = LabyrinthSeal(
    n=0,
    shaft_radius=Q_(72.5, "mm"),
    radial_clearance=Q_(0.3, "mm"),
    n_teeth=16,
    pitch=Q_(3.175, "mm"),
    tooth_height=Q_(3.175, "mm"),
    tooth_width=Q_(0.1524, "mm"),
    seal_type="inter",
    inlet_pressure=308000,
    outlet_pressure=94300,
    inlet_temperature=283.15,
    frequency=Q_([5000, 8000, 11000], "RPM"),
    preswirl=0.98,
    molar=28.97,
    gamma=1.4,
    tz=[300.0, 299.5],
    muz=[1.85e-05, 1.84e-05],
    analz="FULL",
    nprt=1,
    iopt1=1,
)
"""

HOLE_PATTERN_SEAL_V2 = """
from ross.seals.holepattern_seal import HolePatternSeal
from ross.units import Q_
holepattern = HolePatternSeal(
    n=0,
    shaft_radius=0.0725,
    radial_clearance=0.0003,
    length=0.04699,
    roughness=0.0001,
    cell_length=0.003175,
    cell_width=0.003175,
    cell_depth=0.0025,
    inlet_pressure=689000.0,
    outlet_pressure=94300.0,
    inlet_temperature=322.0,
    frequency=Q_([8000], "RPM"),
    gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},
    preswirl=0.8,
    entr_coef=0.5,
    exit_coef=1.0,
    whirl_ratio=1.0,
    rlx_factor=0.1,
    b_suther=1.4e-6,
    s_suther=112.0,
    nz=18
)
"""

HYBRID_SEAL_V2 = """
from ross.seals.hybrid_seal import HybridSeal
from ross.units import Q_
holep_params = {
  "radial_clearance": 0.0003,
  "length": 0.04,
  "roughness": 0.0001,
  "cell_length": 0.003,
  "cell_width": 0.003,
  "cell_depth": 0.002,
  "preswirl": 0.8,
  "entr_coef": 0.5,
  "exit_coef": 1.0,
}
laby_params = {
  "radial_clearance": Q_(0.25, "mm"),
  "n_teeth": 10,
  "pitch": Q_(3, "mm"),
  "tooth_height": Q_(3, "mm"),
  "tooth_width": Q_(0.15, "mm"),
  "seal_type": "inter",
  "preswirl": 0.9,
  "tz": [300.0, 299.5],
  "muz": [1.85e-05, 1.84e-05],
  "nprt": 1,
}
hybrid = HybridSeal(
  n=0,
  shaft_radius=Q_(25, "mm"),
  inlet_pressure=500000,
  outlet_pressure=100000,
  inlet_temperature=300.0,
  frequency=Q_([2000, 3000, 5000], "RPM"),
  gas_composition={"Nitrogen": 0.7812, "Oxygen": 0.2096, "Argon": 0.0092},
  hole_pattern_parameters=holep_params,
  labyrinth_parameters=laby_params,
)
"""

CONSTRUCTOR_EXAMPLES = [
    (PLAIN_JOURNAL_V2, "PlainJournal"),
    (TILTING_PAD_V2, "TiltingPad"),
    (THRUST_PAD_V2, "ThrustPad"),
    (SQUEEZE_FILM_DAMPER_V2, "SqueezeFilmDamper"),
    (LABYRINTH_SEAL_V2, "LabyrinthSeal"),
    (HOLE_PATTERN_SEAL_V2, "HolePatternSeal"),
    (HYBRID_SEAL_V2, "HybridSeal"),
]


@pytest.fixture
def report():
    return Report()


def convert_file(name, tmp_path, report):
    source = V2_DATA / name
    text = convert_model_text(
        source.read_text(), source.suffix, report, path=source, version=rs.__version__
    )
    target = tmp_path / name
    target.write_text(text)
    return target


def converted_data(name, report):
    return convert_model_data(toml.load(V2_DATA / name), report)


def accepted_keywords(class_name):
    cls = getattr(rs, class_name)
    names = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        parameters = inspect.signature(init).parameters
        names |= {
            name
            for name, p in parameters.items()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
        if not any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
            break
    names.discard("self")
    return names


def call_keywords(source, class_name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            )
            if name == class_name:
                return {keyword.arg for keyword in node.keywords}
    raise AssertionError(f"no call to {class_name}")


def dict_keys(source, variable):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and node.targets[0].id == variable:
            return {key.value for key in node.value.keys}
    raise AssertionError(f"no dict {variable}")


@pytest.mark.parametrize("name", ["laby.toml", "holep.toml", "sfd.toml", "thrust.toml"])
def test_v2_solver_files_do_not_load_unconverted(name):
    data = toml.load(V2_DATA / name)
    class_name = next(iter(data)).split("_")[0]
    with pytest.raises(TypeError):
        getattr(rs, class_name).load(V2_DATA / name)


@pytest.mark.parametrize("name", V2_FILES)
def test_v2_files_convert_and_load(name, tmp_path, report):
    target = convert_file(name, tmp_path, report)
    assert check_model_file(target, report)
    assert report.count(ERROR) == 0
    assert report.count(VERIFIED) == 1


def test_ross_version_is_updated(report):
    data = json.loads((V2_DATA / "rotor_example.json").read_text())
    assert data["ross_version"] == "2.3.0"
    converted = convert_model_data(data, report, version="3.0.0")
    assert converted["ross_version"] == "3.0.0"
    assert converted["_note"] == data["_note"]


def test_plain_journal_section_becomes_coefficient_table(report):
    original = toml.load(V2_DATA / "plain.toml")["BearingElement_plain"]
    section = converted_data("plain.toml", report)["BearingElement_plain"]
    assert "speed" in section and "frequency" not in section
    assert section["speed"] == original["frequency"]
    assert section["kxx"] == original["kxx"]
    assert not {"journal_radius", "axial_length", "geometry"} & set(section)


def test_thrust_pad_and_damper_are_downgraded_to_tables(report):
    thrust = converted_data("thrust.toml", report)
    assert list(thrust) == ["BearingElement_thrust"]
    assert "speed" in thrust["BearingElement_thrust"]
    assert thrust["BearingElement_thrust"]["kzz"] == pytest.approx([317634346752.7728])
    damper = converted_data("sfd.toml", report)
    assert list(damper) == ["BearingElement_sfd"]
    assert "frequency" in damper["BearingElement_sfd"]
    assert "speed" not in damper["BearingElement_sfd"]


def test_labyrinth_seal_section_renames(report):
    original = toml.load(V2_DATA / "laby.toml")["LabyrinthSeal_laby"]
    section = converted_data("laby.toml", report)["LabyrinthSeal_laby"]
    assert section["shaft_diameter"] == pytest.approx(0.145)
    assert section["use_jenny_kanki"] is False
    assert section["molar_mass"] == original["molar"]
    assert section["reference_temperatures"] == original["tz"]
    assert section["reference_viscosities"] == original["muz"]
    assert section["speed"] == original["frequency"]
    assert isinstance(section["pitch"], float)
    assert not {"analz", "nprt", "iopt1", "molar", "tz", "muz", "shaft_radius"} & set(
        section
    )
    seal = rs.LabyrinthSeal.read_toml_data(section)
    assert_allclose(seal.kxx, original["kxx"])


def test_hole_pattern_seal_section_renames(report):
    section = converted_data("holep.toml", report)["HolePatternSeal_holep"]
    expected = {
        "shaft_diameter",
        "axial_length",
        "relative_roughness",
        "entrance_loss_coefficient",
        "exit_loss_coefficient",
        "excitation_ratio",
        "relaxation_factor",
        "sutherland_b",
        "sutherland_s",
        "molar_mass",
        "speed",
    }
    assert expected <= set(section)
    assert "kwargs" not in section
    assert section["tag"] == "holep"


def test_rotor_file_round_trip(tmp_path, report):
    original = toml.load(V2_DATA / "rotor_seals.toml")
    rotor = rs.Rotor.load(convert_file("rotor_seals.toml", tmp_path, report))
    assert len(rotor.shaft_elements) == 6
    seals = {type(el).__name__: el for el in rotor.bearing_elements}
    assert {"BearingElement", "LabyrinthSeal", "HolePatternSeal"} <= set(seals)
    assert_allclose(seals["LabyrinthSeal"].kxx, original["LabyrinthSeal_laby"]["kxx"])
    assert_allclose(
        seals["LabyrinthSeal"].speed, original["LabyrinthSeal_laby"]["frequency"]
    )


def test_json_rotor_round_trip(tmp_path, report):
    target = convert_file("rotor_example.json", tmp_path, report)
    data = json.loads(target.read_text())
    assert "speed" in data["BearingElement_Bearing 0"]
    assert "frequency" not in data["BearingElement_Bearing 0"]
    rotor = rs.Rotor.load(target)
    assert len(rotor.bearing_elements) == 2


def test_non_model_files_are_left_alone(report):
    assert not is_model({"a": 1})
    assert convert_model_text('{"a": 1}', ".json", report) is None


@pytest.mark.parametrize("source, class_name", CONSTRUCTOR_EXAMPLES)
def test_constructor_keywords_match_ross_3(source, class_name, report):
    converted = convert_source(source, "script.py", report)
    used = call_keywords(converted, class_name)
    unknown = used - accepted_keywords(class_name)
    assert not unknown, unknown
    assert report.count(MANUAL) == 0
    assert report.count(ERROR) == 0


def test_plain_journal_values_are_converted(report):
    converted = convert_source(PLAIN_JOURNAL_V2, "script.py", report)
    assert "journal_diameter=0.4," in converted
    assert 'pad_arc=Q_(176, "deg")' in converted
    assert 'oil_supply_temperature=Q_(50, "degC")' in converted
    assert 'operating_type="regular_flooded"' in converted
    assert 'speed=Q_([900], "RPM")' in converted
    assert "initial_position=[0.1, -0.1]" in converted
    for removed in ("geometry", "sommerfeld_type", "method", "groove_factor"):
        assert removed not in converted
    assert len(converted.splitlines()) == len(PLAIN_JOURNAL_V2.splitlines()) - 4


def test_tilting_pad_conversion(report):
    converted = convert_source(TILTING_PAD_V2, "script.py", report)
    assert "fxs_load=8.8405e02, fys_load=-2.6704e03" in converted
    assert 'equilibrium_type="match_load"' in converted
    assert 'journal_temperature=Q_(25.0, "degC")' in converted
    assert "preload=[0.5] * 5" in converted
    assert "solver_options" not in converted
    for new in ("total_ex_film=12", "total_ez_film=8", "total_ey_pad=15"):
        assert new in converted
    for new in (
        "hot_oil_lambda",
        "pad_conductivity",
        "edges_convection",
        "relax_temperature",
    ):
        assert new in converted


def test_load_expression_is_indexed(report):
    converted = convert_source("rs.TiltingPad(n=0, load=loads)", "s.py", report)
    assert converted == "rs.TiltingPad(n=0, fxs_load=loads[0], fys_load=loads[1])"
    assert report.count(CHECK) == 1


def test_quantities_are_doubled_inside(report):
    converted = convert_source(SQUEEZE_FILM_DAMPER_V2, "script.py", report)
    assert 'journal_diameter=Q_(5.1, "inches")' in converted
    assert "frequency=" in converted


def test_hybrid_seal_nested_dicts(report):
    converted = convert_source(HYBRID_SEAL_V2, "script.py", report)
    assert 'shaft_diameter=Q_(50, "mm")' in converted
    hole = dict_keys(converted, "holep_params")
    assert {"axial_length", "relative_roughness", "entrance_loss_coefficient"} <= hole
    assert not {"length", "roughness", "entr_coef", "exit_coef"} & hole
    laby = dict_keys(converted, "laby_params")
    assert {"reference_temperatures", "reference_viscosities"} <= laby
    assert not {"tz", "muz", "nprt"} & laby


def test_inline_nested_dict_and_non_literal_are_handled(report):
    source = (
        "rs.HybridSeal(n=0, labyrinth_parameters={'tz': [1, 2], 'iopt1': 1}, "
        "hole_pattern_parameters=params)"
    )
    converted = convert_source(source, "s.py", report)
    assert "'reference_temperatures': [1, 2], 'use_jenny_kanki': True" in converted
    assert report.count(MANUAL) == 1


def test_variable_radius_is_doubled_with_a_check(report):
    converted = convert_source("rs.LabyrinthSeal(n=0, shaft_radius=r)", "s.py", report)
    assert converted == "rs.LabyrinthSeal(n=0, shaft_diameter=2 * r)"
    assert report.count(CHECK) == 1
    converted = convert_source(
        "PlainJournal(n=0, journal_radius=r + 1)", "s.py", report
    )
    assert "journal_diameter=2 * (r + 1)" in converted


def test_q_import_is_added_when_missing(report):
    source = "from ross.bearings.plain_journal import PlainJournal\n\nb = PlainJournal(n=0, pad_arc_length=176)\n"
    converted = convert_source(source, "s.py", report)
    assert converted.splitlines()[:2] == [
        "from ross.bearings.plain_journal import PlainJournal",
        "from ross.units import Q_",
    ]
    assert 'pad_arc=Q_(176, "deg")' in converted


def test_module_alias_gives_the_quantity_accessor(report):
    source = "import ross as rs\nb = rs.PlainJournal(n=0, reference_temperature=50)\n"
    converted = convert_source(source, "s.py", report)
    assert 'oil_supply_temperature=rs.Q_(50, "degC")' in converted
    assert "from ross.units" not in converted


def test_non_literal_unit_values_are_reported(report):
    convert_source("PlainJournal(n=0, pad_arc_length=arc)", "s.py", report)
    assert report.count(MANUAL) == 1
    assert "degrees" in report.findings[-1].message


def test_last_arguments_removed_without_dangling_comma(report):
    source = 'LabyrinthSeal(n=0, iopt1=0, nprt=1, analz="FULL")'
    assert (
        convert_source(source, "s.py", report)
        == "LabyrinthSeal(n=0, use_jenny_kanki=False)"
    )


def test_unpacked_kwargs_positional_and_removed_names_are_reported(report):
    source = (
        "import ross as rs\n"
        "brg = rs.SealElement(0, 1e6, 1e3, **extra)\n"
        "pj = rs.PlainJournal(3, 0.26, 0.2)\n"
        "old = rs.bearings.fluid_flow.BearingFluidFlow(1)\n"
        "from ross.bearings.fluid_flow import fluid_flow\n"
    )
    converted = convert_source(source, "s.py", report)
    assert converted == source
    messages = [f.message for f in report.findings]
    assert any("**extra" in m for m in messages)
    assert any("positional" in m for m in messages)
    assert any("BearingFluidFlow was removed" in m for m in messages)
    assert any("ross.bearings.fluid_flow was removed" in m for m in messages)
    assert report.count(MANUAL) == 3 and report.count(CHECK) == 1


def test_method_keywords_and_moved_modules(report):
    source = (
        "import ross.gear_element as ge\n"
        "from ross.gear_element import GearElement\n"
        "from ross.multi_rotor import MultiRotor\n"
        "import numpy as np\n"
        "r = rotor.run_unbalance_response(3, 0.001, 0.0, frequency=np.linspace(0, 1000, 11))\n"
        "u = rotor.run_ucs(bearing_frequency_range=(1e6, 1e9))\n"
    )
    converted = convert_source(source, "s.py", report)
    assert converted.splitlines()[:3] == [
        "import ross.multi_rotor.gear_element as ge",
        "from ross.multi_rotor.gear_element import GearElement",
        "from ross.multi_rotor.multi_rotor import MultiRotor",
    ]
    assert "speed_range=np.linspace(0, 1000, 11)" in converted
    assert "bearing_speed_range=(1e6, 1e9)" in converted
    assert "frequency" not in converted


def test_bearing_element_frequency_becomes_speed(report):
    source = "rs.BearingElement(n=0, kxx=[1e6, 1.1e6], cxx=[1e3, 1.2e3], frequency=[100.0, 200.0])"
    assert "speed=[100.0, 200.0]" in convert_source(source, "s.py", report)
    source = "rs.MagneticBearingElement(n=0, frequency=[100.0])"
    assert convert_source(source, "s.py", report) == source


def test_probe_tuples_become_probe_objects(report):
    source = (
        "import ross as rs\n"
        'res.plot_1d(probe=[(3, 0), (5, 45, "DE")], probe_units="deg")\n'
        'res.data_magnitude([(0, "major"), (2, np.pi / 2, "tag")])\n'
    )
    converted = convert_source(source, "s.py", report)
    assert converted.splitlines()[1:] == [
        'res.plot_1d(probe=[rs.Probe(3, rs.Q_(0, "deg")), '
        'rs.Probe(5, rs.Q_(45, "deg"), tag="DE")])',
        'res.data_magnitude([rs.Probe(0, "major"), rs.Probe(2, np.pi / 2, tag="tag")])',
    ]
    assert report.count(CHANGED) == 2
    assert "probe tuples -> Probe objects" in report.findings[0].message


def test_probe_list_variable_and_missing_imports(report):
    source = (
        "probes = [(3, 0), (3, angle)]\n"
        "res.plot_1d(probe=probes, probe_units=units)\n"
        "res.plot_dfft(probe=probes)\n"
        "fig = unb.plot(\n"
        "    probe=[\n"
        "        (0, 45),\n"
        "    ],\n"
        '    probe_units="deg",\n'
        ")\n"
    )
    converted = convert_source(source, "s.py", report)
    assert converted == (
        "from ross.units import Q_\n"
        "from ross import Probe\n"
        "probes = [Probe(3, Q_(0, units)), Probe(3, Q_(angle, units))]\n"
        "res.plot_1d(probe=probes)\n"
        "res.plot_dfft(probe=probes)\n"
        "fig = unb.plot(\n"
        "    probe=[\n"
        '        Probe(0, Q_(45, "deg")),\n'
        "    ],\n"
        ")\n"
    )


def test_probe_objects_and_foreign_calls_are_left_alone(report):
    source = (
        "from ross import Probe\n"
        'res.plot_1d(probe=[Probe(3, 0)], probe_units="rad")\n'
        "ax.plot([(1, 2)])\n"
        "res.plot(probe=[(3, 0)], probe_units='rad')\n"
    )
    converted = convert_source(source, "s.py", report)
    assert converted.splitlines()[1:] == [
        'res.plot_1d(probe=[Probe(3, 0)], probe_units="rad")',
        "ax.plot([(1, 2)])",
        "res.plot(probe=[Probe(3, 0)])",
    ]


def test_syntax_errors_are_skipped(report):
    source = "x = (\n"
    assert convert_source(source, "s.py", report) == source
    assert report.count(SKIPPED) == 1


def make_notebook(*sources):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["# ROSS 2\n"]}]
    for source in sources:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        )
    return (
        json.dumps(
            {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5},
            indent=1,
        )
        + "\n"
    )


def test_notebook_conversion(report):
    text = make_notebook(
        "%matplotlib inline\nimport ross as rs\nQ_ = rs.Q_\n",
        "!ls\nb = rs.PlainJournal(n=0, journal_radius=0.1, pad_arc_length=176, frequency=[100.0])\n",
    )
    converted = json.loads(convert_notebook(text, "nb.ipynb", report))
    cell = "".join(converted["cells"][2]["source"])
    assert cell == (
        "!ls\nb = rs.PlainJournal(n=0, journal_diameter=0.2, "
        'pad_arc=Q_(176, "deg"), speed=[100.0])\n'
    )
    assert "".join(converted["cells"][1]["source"]).startswith("%matplotlib inline\n")
    assert all(f.location.startswith("cell 3, line 2") for f in report.findings)


def test_notebook_probe_accessor_is_shared_between_cells(report):
    text = make_notebook(
        "from ross import Probe\n",
        'fig = res.plot_1d(probe=[(3, 0, "DE")])\n',
    )
    converted = json.loads(convert_notebook(text, "nb.ipynb", report))
    assert "".join(converted["cells"][2]["source"]) == (
        'fig = res.plot_1d(probe=[Probe(3, 0, tag="DE")])\n'
    )


def test_unchanged_notebook_text_is_returned_verbatim(report):
    text = make_notebook("import ross as rs\nrotor = rs.rotor_example()\n")
    assert convert_notebook(text, "nb.ipynb", report) is text


def test_cli_preview_does_not_write(tmp_path, capsys):
    target = tmp_path / "laby.toml"
    target.write_text((V2_DATA / "laby.toml").read_text())
    assert main([str(target)]) == 0
    assert target.read_text() == (V2_DATA / "laby.toml").read_text()
    out = capsys.readouterr().out
    assert "+shaft_diameter = 0.145" in out
    assert "Summary:" in out and "verified" in out


def test_cli_write_keeps_a_backup(tmp_path, capsys):
    target = tmp_path / "holep.toml"
    original = (V2_DATA / "holep.toml").read_text()
    target.write_text(original)
    assert main(["-w", str(target)]) == 0
    assert (tmp_path / "holep.toml.bak").read_text() == original
    rs.HolePatternSeal.load(target)
    assert main(["-w", "-n", str(target)]) == 0
    assert not (tmp_path / "holep.toml.toml.bak").exists()


def test_cli_output_dir_and_report(tmp_path, capsys):
    source = tmp_path / "src"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "rotor.toml").write_text(
        (V2_DATA / "rotor_seals.toml").read_text()
    )
    (source / "script.py").write_text(PLAIN_JOURNAL_V2)
    (source / "settings.json").write_text('{"theme": "dark"}\n')
    (source / ".hidden").mkdir()
    (source / ".hidden" / "old.toml").write_text((V2_DATA / "laby.toml").read_text())
    out_dir = tmp_path / "out"
    report_file = tmp_path / "report.txt"
    assert main(["-o", str(out_dir), "--report", str(report_file), str(source)]) == 0
    assert (out_dir / "sub" / "rotor.toml").exists()
    assert 'pad_arc=Q_(176, "deg")' in (out_dir / "script.py").read_text()
    assert not (out_dir / "settings.json").exists()
    assert not (out_dir / ".hidden").exists()
    text = report_file.read_text()
    assert "not a ROSS rotor or element file" in text
    assert "converted file loads with ROSS" in text
    rs.Rotor.load(out_dir / "sub" / "rotor.toml")


def test_cli_reports_missing_files(tmp_path, capsys):
    assert main([str(tmp_path / "missing.toml")]) == 1
    assert "file not found" in capsys.readouterr().out


def test_release_notes_document_every_rename():
    notes = Path(__file__).parents[2] / "docs" / "release_notes" / "version-3.0.0.rst"
    if not notes.exists():
        pytest.skip("release notes are not part of the installed package")
    text = notes.read_text(encoding="utf-8")
    assert "ross_2to3" in text
    for table in CLASS_RENAMES.values():
        for old in table:
            assert f"``{old}``" in text, old
    assert migration_table_rst().strip() in text


def test_report_rendering(report):
    report.add("a.py", "line 10", CHANGED, "x -> y")
    report.add("a.py", "line 2", MANUAL, "fix me")
    rendered = report.render()
    assert rendered.index("fix me") < rendered.index("x -> y")
    assert rendered.endswith(
        "Summary: 1 converted automatically, 1 needs manual conversion"
    )
    with pytest.raises(ValueError):
        report.add("a.py", "", "bogus", "message")
