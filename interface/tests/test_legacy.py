# -*- coding: utf-8 -*-
"""Projects saved before ROSS 3 renamed the table axis still open.

ROSS 3.0 (petrobras/ross#1371) made `frequency=` mean the excitation frequency
and moved the fluid-film bearings and seals to `speed=`. The forms followed,
but a rotor saved by the interface before that -- the browser's stored state, a
JSON exported from the Hub, a version-2 ROSS file -- still says `frequency`.
`domain/legacy.py` translates it at the three doors: the rotor builder, the
script exporter and the ROSS-file import."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain.legacy import LEGACY_SPEED_CLASSES, SPEED_ONLY_CLASSES, migrate_element
from domain.python_export import build_script

try:
    import ross  # noqa: F401

    HAS_ROSS = True
except Exception:  # pragma: no cover
    HAS_ROSS = False

needs_ross = pytest.mark.skipif(not HAS_ROSS, reason="requires ROSS installed")


def test_the_translated_classes_are_derived_from_the_forms_and_the_signatures():
    """Derived, not listed -- and this is the answer it has to give today.

    The fluid-film bearings have no `frequency` field, so a bare one is the old
    spelling and a stale one is dropped. The seals offer `frequency` as the
    second axis of a 2-D table, but ROSS requires their `speed`, so a bare
    `frequency` there is still the old spelling."""
    assert SPEED_ONLY_CLASSES == {
        "CylindricalBearing",
        "PlainJournal",
        "ThrustPad",
        "TiltingPad",
    }
    assert LEGACY_SPEED_CLASSES == SPEED_ONLY_CLASSES | {
        "HolePatternSeal",
        "LabyrinthSeal",
        "HybridSeal",
    }


def test_frequency_becomes_speed_in_place_with_its_unit():
    element = {
        "element_type": "PlainJournal",
        "frequency": "[900]",
        "frequency_unit": "Hz",
        "fys_load": "1000",
    }
    migrated = migrate_element(element, "PlainJournal")
    assert list(migrated) == ["element_type", "speed", "speed_unit", "fys_load"]
    assert migrated["speed"] == "[900]"
    assert migrated["speed_unit"] == "Hz"
    assert "frequency" in element and "speed" not in element, (
        "the original was not touched"
    )


def test_a_stale_frequency_is_dropped_when_speed_is_already_filled():
    """On a form with no `frequency` field ROSS would read the pair as a 2-D
    table the user never asked for."""
    migrated = migrate_element(
        {"element_type": "TiltingPad", "speed": "[900]", "frequency": "[100]"},
        "TiltingPad",
    )
    assert migrated == {"element_type": "TiltingPad", "speed": "[900]"}


def test_a_seal_with_both_axes_is_a_2d_table_and_is_kept():
    """The seal forms offer `frequency` as the excitation axis: the pair is real."""
    element = {
        "element_type": "Labyrinth",
        "speed": "[8000]",
        "frequency": "[4000, 8000]",
    }
    assert migrate_element(element, "LabyrinthSeal") is element


def test_a_seal_with_frequency_alone_is_still_the_old_spelling():
    """ROSS requires the seal's `speed`, so `frequency` alone cannot be the new axis."""
    migrated = migrate_element(
        {"element_type": "HolePattern", "frequency": "[900]"}, "HolePatternSeal"
    )
    assert migrated == {"element_type": "HolePattern", "speed": "[900]"}


def test_an_empty_speed_does_not_count_as_filled():
    migrated = migrate_element({"speed": "", "frequency": "[100]"}, "TiltingPad")
    assert migrated == {"speed": "[100]"}


def test_classes_that_accept_both_axes_are_left_alone():
    """BearingElement and SealElement take `frequency` as the excitation table."""
    element = {"kxx": "1e6", "cxx": "0", "frequency": "[100, 200]"}
    assert migrate_element(element, "BearingElement") is element
    assert migrate_element(element, "SealElement") is element
    assert migrate_element(element, "MagneticBearingElement") is element
    assert migrate_element(element, "SqueezeFilmDamper") is element


def test_an_element_without_the_old_key_is_the_same_object():
    element = {"speed": "[900]"}
    assert migrate_element(element, "PlainJournal") is element


@needs_ross
def test_the_exported_script_speaks_the_current_vocabulary():
    """The script leaves the program: it has to run against the ROSS it targets."""
    project = {
        "shafts": [{"L": "500", "odl": "100", "idl": "0", "n": "0"}],
        "bearings": [
            {"element_type": "PlainJournal", "frequency": "[900]", "n": "0"},
            {"element_type": "SqueezeFilm", "frequency": "[900]", "n": "1"},
            {"element_type": "BASIC", "kxx": "1e6", "frequency": "[100]", "n": "1"},
        ],
        "seals": [{"element_type": "Labyrinth", "frequency": "[8000]", "n": "0"}],
    }
    script = build_script(project, [], "")
    assert "rs.PlainJournal, dict(speed=Q_(np.array([900]), 'RPM')" in script
    assert "rs.LabyrinthSeal, dict(speed=Q_(np.array([8000]), 'RPM')" in script
    assert "rs.SqueezeFilmDamper, dict(frequency=Q_(np.array([900]), 'RPM')" in script
    assert (
        "rs.BearingElement, dict(kxx=Q_(1e6, 'N/m'), frequency=Q_(np.array([100]), 'RPM')"
        in script
    )


@needs_ross
def test_a_version_2_ross_file_lands_on_the_speed_field_in_the_form_unit():
    """The key is translated before the unit conversion, or 900 rad/s would show
    up as 900 RPM."""
    from domain.ross_import import project_from_ross_file

    content = (
        "[PlainJournal_0]\n"
        "n = 0\n"
        "journal_diameter = 0.1\n"
        "frequency = [ 94.24777960769379,]\n"
    )
    project = project_from_ross_file(content)
    (bearing,) = project["bearings"]
    assert bearing["element_type"] == "PlainJournal"
    assert "frequency" not in bearing
    assert bearing["speed"] == "[900.0]"
    assert bearing["journal_diameter"] == "100.0"
