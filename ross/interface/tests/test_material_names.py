# -*- coding: utf-8 -*-
"""A material named with spaces, from the form to ROSS and to the script.

`rs.Material` refuses a space in the name, and the form let a person type
"Stainless Steel": the rotor then failed to build. The screen now writes the
name the way ROSS wants it when the material is saved; the server applies the
same rule (domain/material_names.py) to projects saved before that, on both ends
of the reference -- the material and the shafts that name it.
"""

import copy
import json
import os
import sys

import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ross.interface.domain.concatenation import concatenated_project  # noqa: E402
from ross.interface.domain.material_names import ross_material_name  # noqa: E402
from ross.interface.domain.python_export import build_script  # noqa: E402
from ross.interface.domain.rotor_builder import build_rotor_from_ui  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "golden", "material_names.json")

# A shaft whose material name is not found gets the *first* material of the
# list, silently (`rotor_builder.extract_kwargs`). So plain steel goes first,
# and the density tells a shaft that found its stainless apart from one that
# fell back.
STEEL = {"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}
STAINLESS = {"name": "Stainless Steel", "rho": "8000", "E": "193e9", "G_s": "77e9"}


def project(material_name="Stainless Steel", shaft_names="Stainless Steel"):
    """A rotor in the screen's format: two shafts of one material, a bearing."""
    return {
        "materials": [dict(STEEL), dict(STAINLESS, name=material_name)],
        "shafts": [
            {
                "element_type": "BASIC",
                "L": "250",
                "idl": "0",
                "odl": "50",
                "material": shaft_names,
            }
            for _ in range(2)
        ],
        "disks": [],
        "gears": [],
        "couplings": [],
        "seals": [],
        "bearings": [
            {"element_type": "BASIC", "n": "0", "kxx": "1e6", "cxx": "0"},
            {"element_type": "BASIC", "n": "2", "kxx": "1e6", "cxx": "0"},
        ],
        "pointmasses": [],
    }


def test_the_rule_matches_the_cases_the_screen_is_held_to():
    """The same file drives tests/js/test_material_names.js."""
    with open(GOLDEN, encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    assert len(cases) > 5
    for typed, expected in cases:
        assert ross_material_name(typed) == expected, typed


def test_ross_still_refuses_the_space():
    """What justifies the rule. If ROSS starts accepting spaces, this says so."""
    with pytest.raises(ValueError, match="Spaces are not allowed"):
        ross.Material(name="Stainless Steel", rho=8000, E=193e9, G_s=77e9)


def test_a_project_saved_with_a_space_builds_and_its_shafts_find_the_material():
    rotor = build_rotor_from_ui(project())
    for shaft in rotor.shaft_elements:
        assert shaft.material.name == "Stainless_Steel"
        assert shaft.material.rho == pytest.approx(8000)


@pytest.mark.parametrize(
    "material_name, shaft_names",
    [("Stainless_Steel", "Stainless Steel"), ("Stainless Steel", "stainless_steel")],
)
def test_either_spelling_on_either_end_meets_the_other(material_name, shaft_names):
    """A material saved again from the form is underscored; a shaft saved before
    still says it with a space. The two must still meet."""
    rotor = build_rotor_from_ui(project(material_name, shaft_names))
    assert [s.material.rho for s in rotor.shaft_elements] == pytest.approx([8000, 8000])


def test_the_exported_script_uses_the_name_ross_accepts():
    script = build_script(project())
    assert "name='Stainless_Steel'" in script
    assert "'Stainless Steel'" not in script
    # ... and the shafts look it up under the same key it is stored under.
    assert script.count("'stainless_steel'") >= 3


# --- a reference that finds no material --------------------------------------
#
# Measured before the guard: a shaft asking for "Inox" in a rotor whose
# materials are Steel and Stainless was built out of **Steel** -- the first of
# the list -- and the rotor came back whole. The analysis then ran on a machine
# nobody described.


def test_ross_is_not_the_one_that_would_refuse_it():
    """The guard is ours because nothing below it raises: `extract_kwargs`
    answers an unknown name with the first material of the list, and with no
    materials at all with ROSS's own steel. Either way something gets built."""
    from ross.interface.domain.rotor_builder import extract_kwargs

    steel = ross.materials.steel
    chosen = extract_kwargs({"material": "Inox"}, {"steel": steel}, "ShaftElement")
    assert chosen["material"] is steel


def test_an_element_naming_a_material_the_rotor_does_not_have_is_refused():
    broken = project()
    broken["shafts"][1]["material"] = "Inox"
    broken["shafts"][1]["tag"] = "LP_1"
    with pytest.raises(ValueError) as refused:
        build_rotor_from_ui(broken)
    message = str(refused.value)
    assert "shaft #2" in message, message
    assert "LP_1" in message, message
    assert "Inox" in message, message
    # ... and it says what the rotor does have, which is what the person has to
    # pick from.
    assert "Stainless Steel" in message


def test_the_exported_script_is_refused_the_same_way():
    """A script is worse than a screen: it leaves and runs somewhere else."""
    broken = project()
    broken["gears"] = [{"element_type": "BASIC", "n": "1", "material": "Inox"}]
    with pytest.raises(ValueError, match="gear #1"):
        build_script(broken)


def test_a_multirotor_is_checked_half_by_half():
    broken = project()
    broken["shafts"][0]["material"] = "Inox"
    multi = {
        "isMultiRotor": True,
        "driving_rotor": project(),
        "driven_rotor": broken,
        "multi_params": {"coupled_nodes": "0, 0"},
    }
    with pytest.raises(ValueError, match="Inox"):
        build_rotor_from_ui(multi)
    with pytest.raises(ValueError, match="Inox"):
        build_script(multi)


@pytest.mark.parametrize("reference", ["Default (Steel)", "default (steel)", "", "   "])
def test_the_screens_own_default_is_not_a_missing_material(reference):
    """ "Default (Steel)" is the screen saying "none of mine": it is ROSS's steel,
    not a name the rotor is supposed to carry. A blank says the same."""
    fine = project()
    for shaft in fine["shafts"]:
        shaft["material"] = reference
    rotor = build_rotor_from_ui(fine)
    assert rotor.shaft_elements[0].material.name == "Steel"


def test_every_category_of_element_is_named_in_the_refusal():
    """The message is read by a person. A category added to the registry and
    not named here would be checked and reported as nothing."""
    from ross.interface.domain.element_registry import ELEMENTS
    from ross.interface.domain.material_names import ELEMENT_WORDS

    assert set(ELEMENT_WORDS) == set(ELEMENTS) - {"materials"}
    assert all(word.strip() for word in ELEMENT_WORDS.values())


def test_the_refusal_stops_counting_after_a_few_and_says_how_many():
    """A rotor built by a script can have hundreds of elements naming a material
    that went away; the message stays readable."""
    broken = project()
    broken["shafts"] = [dict(broken["shafts"][0], material="Inox") for _ in range(9)]
    with pytest.raises(ValueError) as refused:
        build_rotor_from_ui(broken)
    message = str(refused.value)
    assert "and 4 more" in message, message
    # ... and the listing really stops: five named, the rest counted.
    assert message.count("shaft #") == 5, message
    assert "shaft #9" not in message, message


def test_concatenation_sees_the_two_spellings_as_one_material():
    """Same properties, one spelled with a space: one material, not a `_R1`."""
    first = project("Stainless Steel", "Stainless Steel")
    second = project("Stainless_Steel", "Stainless_Steel")
    merged = concatenated_project(first, copy.deepcopy(second))
    assert len(merged["materials"]) == 2, [m["name"] for m in merged["materials"]]
    build_rotor_from_ui(merged)
