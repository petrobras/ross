# -*- coding: utf-8 -*-
"""The hybrid seal's stages, one field per key (domain/seal_stages.py).

The users found the two typed dictionaries hard to read and to get right, so
the form asks for one field per key, each with its unit. What has to hold:
the fields fold into the dictionaries ROSS takes, in SI, and equal to the
ones typed the old way; an old project still builds; a field wins over the
same key of a typed dictionary; and the builder and the exported script read
the same folded row.
"""

import ast

import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

from ross.interface.domain import rotor_builder  # noqa: E402
from ross.interface.domain.python_export import build_script  # noqa: E402
from ross.interface.domain.schema import build_schema  # noqa: E402
from ross.interface.domain.seal_stages import STAGE_UNREADABLE, folded  # noqa: E402

HOLE_PATTERN = {
    "radial_clearance": 0.0003,
    "axial_length": 0.04,
    "relative_roughness": 0.0001,
    "cell_length": 0.003,
    "cell_width": 0.003,
    "cell_depth": 0.002,
    "preswirl": 0.8,
    "entrance_loss_coefficient": 0.5,
    "exit_loss_coefficient": 1.0,
}
LABYRINTH = {
    "radial_clearance": 0.00025,
    "n_teeth": 10,
    "pitch": 0.003,
    "tooth_height": 0.003,
    "tooth_width": 0.00015,
    "seal_type": "inter",
    "preswirl": 0.9,
    "reference_temperatures": [300.0, 299.5],
    "reference_viscosities": [1.85e-05, 1.84e-05],
}
COMMON = {
    "element_type": "Hybrid",
    "shaft_diameter": "50",
    "inlet_pressure": "500000",
    "outlet_pressure": "100000",
    "inlet_temperature": "26.85",
    "speed": "2000",
}
FIELDS = {
    "hp_radial_clearance": "0.3",
    "hp_axial_length": "4",
    "hp_axial_length_unit": "cm",
    "hp_relative_roughness": "0.0001",
    "hp_cell_length": "3",
    "hp_cell_width": "3",
    "hp_cell_depth": "2",
    "hp_preswirl": "0.8",
    "hp_entrance_loss_coefficient": "0.5",
    "hp_exit_loss_coefficient": "1.0",
    "lab_radial_clearance": "0.25",
    "lab_n_teeth": "10",
    "lab_pitch": "3",
    "lab_tooth_height": "3",
    "lab_tooth_width": "0.15",
    "lab_seal_type": "inter",
    "lab_preswirl": "0.9",
    "lab_reference_temperatures": "300, 299.5",
    "lab_reference_viscosities": "[1.85e-05, 1.84e-05]",
}


def stages(row):
    out = folded(row)
    return (
        ast.literal_eval(out["hole_pattern_parameters"]),
        ast.literal_eval(out["labyrinth_parameters"]),
    )


def same(one, other):
    assert set(one) == set(other)
    for key in one:
        if isinstance(one[key], (int, float)) and not isinstance(one[key], bool):
            assert one[key] == pytest.approx(other[key], rel=1e-12), key
        else:
            assert one[key] == other[key], key


def test_the_fields_fold_into_the_dictionaries_typed_the_old_way():
    hole, laby = stages(dict(COMMON, **FIELDS))
    same(hole, HOLE_PATTERN)
    same(laby, LABYRINTH)
    assert type(laby["n_teeth"]) is int
    assert not any(
        key.startswith(("hp_", "lab_")) for key in folded(dict(COMMON, **FIELDS))
    )


def test_an_old_project_is_read_as_it_was():
    old = dict(
        COMMON,
        hole_pattern_parameters=repr(HOLE_PATTERN),
        labyrinth_parameters=repr(LABYRINTH),
    )
    assert folded(old) is old


def test_a_field_wins_over_the_same_key_of_a_typed_dictionary():
    mixed = dict(COMMON, hole_pattern_parameters=repr(HOLE_PATTERN), hp_cell_depth="5")
    hole, _ = stages(dict(mixed, labyrinth_parameters=repr(LABYRINTH)))
    assert hole["cell_depth"] == pytest.approx(0.005)
    assert hole["cell_length"] == pytest.approx(0.003)


def test_a_typed_dictionary_that_is_not_one_is_refused_by_name():
    with pytest.raises(ValueError) as refused:
        folded(dict(COMMON, hp_cell_depth="5", hole_pattern_parameters="{oops"))
    assert str(refused.value) == STAGE_UNREADABLE % ("hole_pattern_parameters", "{oops")


def test_every_stage_field_is_a_key_ross_documents():
    form = build_schema("en")["categories"]["seals"]["Hybrid"]
    stage_fields = [f for f in form["fields"] if f["name"].startswith(("hp_", "lab_"))]
    assert len(stage_fields) == 27
    assert all(f["known_to_ross"] and f["help"] for f in stage_fields)
    assert form["not_in_form"] == []


def test_the_builder_hands_ross_the_folded_dictionaries(monkeypatch):
    """Recorded at the call: building a real hybrid seal takes about 20 s."""
    seen = {}

    class Recorder(ross.SealElement):
        def __init__(self, n, **kwargs):
            seen.update(kwargs)
            super().__init__(n=n, kxx=1e6, cxx=0, tag=kwargs.get("tag"))

    monkeypatch.setattr(ross, "HybridSeal", Recorder)
    project = {
        "materials": [],
        "shafts": [
            {"L": "250", "odl": "50", "idl": "0", "material": "Default (Steel)"}
            for _ in range(2)
        ],
        "bearings": [
            {"n": "0", "kxx": "1e6", "cxx": "0"},
            {"n": "2", "kxx": "1e6", "cxx": "0"},
        ],
        "seals": [dict(COMMON, n="1", tag="recorded-hybrid", **FIELDS)],
    }
    rotor_builder.build_rotor_from_ui(project)
    same(seen["hole_pattern_parameters"], HOLE_PATTERN)
    same(seen["labyrinth_parameters"], LABYRINTH)


def test_the_exported_script_carries_the_same_dictionaries():
    project = {
        "materials": [],
        "shafts": [{"L": "250", "odl": "50", "idl": "0"}],
        "seals": [dict(COMMON, n="0", **FIELDS)],
    }
    script = build_script(project, [], "")
    ast.parse(script)
    line = next(text for text in script.splitlines() if "rs.HybridSeal" in text)
    assert "hp_" not in line and "lab_" not in line
    call = ast.parse(line.strip().rstrip(","), mode="eval").body
    arguments = {k.arg: k.value for k in call.elts[1].keywords}
    same(ast.literal_eval(arguments["hole_pattern_parameters"]), HOLE_PATTERN)
    same(ast.literal_eval(arguments["labyrinth_parameters"]), LABYRINTH)
