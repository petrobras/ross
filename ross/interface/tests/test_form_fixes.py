# -*- coding: utf-8 -*-
"""The element forms, after the first round of user testing.

What the testers found, and what holds it in place:

* TiltingPad: `xj`/`yj` had no description and a unit they do not have --
  they are fractions of the radial clearance, and were read as millimetres;
  `equilibrium_type` was typed instead of chosen;
* the shaft's `shear_method_calc` and every `lubricant` were typed instead of
  chosen among what ROSS accepts;
* ball, roller and magnetic bearings had no units;
* a single speed or frequency had to be typed between brackets;
* a point mass's `mx`, `my`, `mz` read as something other than masses.

And one found while fixing them: a quantity ROSS does not convert itself --
a ball bearing's `fs`, any argument of a constructor without `@check_units` --
reached ROSS as a pint Quantity. Which ones ROSS converts is restated in
`element_registry.ross_converts`, held here to ROSS's own `check_units`.
"""

import warnings

import numpy as np
import pytest

ross = pytest.importorskip("ross", reason="requires ROSS installed")

from ross.bearings.lubricants import lubricants_dict  # noqa: E402
from ross.interface.domain.element_registry import (  # noqa: E402
    ross_constructor,
    ross_converts,
)
from ross.interface.domain.field_catalog import FIELDS  # noqa: E402
from ross.interface.domain.rotor_builder import (  # noqa: E402
    build_rotor_from_ui,
    extract_kwargs,
)
from ross.interface.domain.schema import build_schema  # noqa: E402
from ross.units import Q_, check_units  # noqa: E402


def field(category, subtype, name):
    form = build_schema("en")["categories"][category][subtype]
    return next(f for f in form["fields"] if f["name"] == name)


# --- the forms ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["xj", "yj"])
def test_the_journal_position_says_what_it_is_and_has_no_unit(name):
    spec = field("bearings", "TiltingPad", name)
    assert spec["help"].startswith("Journal position as fractions of the radial")
    assert spec["unit"] is None


def test_a_journal_position_saved_with_a_unit_is_read_as_typed():
    """A row saved while xj was mapped to mm still carries `xj_unit`."""
    kwargs = extract_kwargs(
        {"xj": "0.3", "xj_unit": "mm", "yj": "-0.2", "yj_unit": "mm"},
        {},
        "TiltingPad",
        [],
    )
    assert (kwargs["xj"], kwargs["yj"]) == (0.3, -0.2)


@pytest.mark.parametrize(
    "category, subtype, name, options",
    [
        (
            "bearings",
            "TiltingPad",
            "equilibrium_type",
            ["match_eccentricity", "match_load"],
        ),
        ("shafts", "BASIC", "shear_method_calc", ["cowper", "hutchinson"]),
    ],
)
def test_a_choice_among_fixed_words_is_a_choice(category, subtype, name, options):
    spec = field(category, subtype, name)
    assert (spec["control"], spec["options"]) == ("select", options)


def test_the_shear_methods_offered_are_the_ones_ross_computes():
    """Each one builds a shaft; a word ROSS does not know raises."""
    steel = ross.materials.steel
    for method in field("shafts", "BASIC", "shear_method_calc")["options"]:
        ross.ShaftElement(0.25, 0, 0.05, material=steel, shear_method_calc=method)
    with pytest.raises(Warning):
        ross.ShaftElement(0.25, 0, 0.05, material=steel, shear_method_calc="other")


def test_every_lubricant_is_chosen_among_the_ones_ross_knows():
    offered = [
        (category, subtype)
        for category, subtypes in FIELDS.items()
        for subtype, fields in subtypes.items()
        if any(f["name"] == "lubricant" for f in fields)
    ]
    assert len(offered) == 4
    names = [name for name in lubricants_dict if name != "TEST"]
    for category, subtype in offered:
        spec = field(category, subtype, "lubricant")
        assert (spec["control"], spec["options"]) == ("select", names)


@pytest.mark.parametrize(
    "subtype, units",
    [
        ("BallBearing", {"d_balls": "m", "fs": "N", "alpha": "rad"}),
        ("RollerBearing", {"l_rollers": "m", "fs": "N", "alpha": "rad"}),
        (
            "MagneticBearing",
            {
                "g0": "m",
                "i0": "A",
                "ag": "m**2",
                "alpha": "rad",
                "k_amp": "V/A",
                "k_sense": "V/m",
            },
        ),
    ],
)
def test_the_bearings_that_had_no_units_have_them(subtype, units):
    for name, unit in units.items():
        spec = field("bearings", subtype, name)
        assert spec["unit"] == unit and spec["unit_options"][0] == unit


def test_the_masses_of_a_point_mass_say_they_are_masses():
    for axis in "xyz":
        spec = field("pointmasses", "BASIC", "m" + axis)
        assert spec["label"] == "Mass in %s (m%s)" % (axis, axis)
        assert spec["unit"] == "kg"


# --- what ROSS converts ---------------------------------------------------------------


def _converted_by_ross(name, unit):
    """Whether ROSS's `check_units` turns a quantity for `name` into a number."""
    namespace = {}
    exec("def target(%s=None):\n    return %s" % (name, name), namespace)
    value = check_units(namespace["target"])(**{name: Q_(1.0, unit)})
    return not isinstance(value, Q_)


def test_ross_converts_says_what_check_units_does():
    """Every field with a unit, of every form whose constructor checks units."""
    seen = 0
    for category, subtypes in build_schema("en")["categories"].items():
        for subtype, form in subtypes.items():
            constructor = ross_constructor(category, subtype)
            for spec in form["fields"]:
                if not spec["unit"]:
                    continue
                expected = ross_converts(constructor, spec["name"])
                if expected:
                    assert _converted_by_ross(spec["name"], spec["unit"]), (
                        constructor,
                        spec["name"],
                    )
                    seen += 1
    assert seen > 50


def test_a_constructor_without_check_units_converts_nothing():
    assert not ross_converts("BallBearingElement", "d_balls")
    assert not ross_converts("MagneticBearingElement", "g0")
    assert ross_converts("CylindricalBearing", "oil_viscosity")
    assert not ross_converts("PointMass", "mz")


def _rotor(bearing):
    return {
        "materials": [],
        "shafts": [
            {"L": "250", "idl": "0", "odl": "50", "material": "Default (Steel)"}
            for _ in range(2)
        ],
        "bearings": [dict(bearing, n="0"), {"n": "2", "kxx": "1e6", "cxx": "0"}],
    }


def _bearing(project):
    return build_rotor_from_ui(project).bearing_elements[0]


@pytest.mark.parametrize(
    "si, typed",
    [
        (
            {
                "element_type": "BallBearing",
                "n_balls": "8",
                "d_balls": "0.03",
                "fs": "500",
                "alpha": "0.5235987755982988",
            },
            {
                "element_type": "BallBearing",
                "n_balls": "8",
                "d_balls": "30",
                "d_balls_unit": "mm",
                "fs": "0.5",
                "fs_unit": "kN",
                "alpha": "30",
                "alpha_unit": "deg",
            },
        ),
        (
            {
                "element_type": "MagneticBearing",
                "g0": "1e-3",
                "i0": "1",
                "ag": "1e-4",
                "nw": "200",
                "kp_pid": "1",
                "kd_pid": "1",
            },
            {
                "element_type": "MagneticBearing",
                "g0": "1",
                "g0_unit": "mm",
                "i0": "1000",
                "i0_unit": "mA",
                "ag": "100",
                "ag_unit": "mm**2",
                "nw": "200",
                "kp_pid": "1",
                "kd_pid": "1",
            },
        ),
    ],
)
def test_a_bearing_typed_in_other_units_is_the_same_bearing(si, typed):
    one, other = _bearing(_rotor(si)), _bearing(_rotor(typed))
    assert np.allclose(one.K(0), other.K(0), rtol=1e-6)
    assert np.allclose(one.C(0), other.C(0), rtol=1e-6)


def test_a_point_mass_with_its_axial_mass_builds():
    """`mz` is not in ROSS's unit table, so `@check_units` passes a Quantity
    through, and PointMass then fails with "Cannot convert from 'kilogram' to
    'dimensionless'". It reaches ROSS as its number in kg."""
    project = _rotor({"n": "2", "kxx": "1e6", "cxx": "0"})
    project["pointmasses"] = [
        {"n": "1", "m": "2", "mx": "2", "my": "2", "mz": "2000", "mz_unit": "g"}
    ]
    point_mass = build_rotor_from_ui(project).point_mass_elements[0]
    assert (point_mass.mx, point_mass.my, point_mass.mz) == (2.0, 2.0, 2.0)


# --- lists without brackets -----------------------------------------------------------

CYLINDRICAL = {
    "element_type": "Cylindrical",
    "weight": "525",
    "bearing_length": "30",
    "journal_diameter": "10",
    "radial_clearance": "0.1",
    "oil_viscosity": "0.1",
}
SQUEEZE_FILM = {
    "element_type": "SqueezeFilm",
    "axial_length": "22.86",
    "journal_diameter": "129.54",
    "radial_clearance": "7.62e-2",
    "eccentricity_ratio": "0.5",
    "lubricant": "ISOVG32",
    "geometry": "groove",
    "cavitation": "True",
}


@pytest.mark.parametrize(
    "bearing, key, alone, listed",
    [
        (CYLINDRICAL, "speed", "1500", "[1500]"),
        (SQUEEZE_FILM, "frequency", "18600", "[18600]"),
    ],
)
def test_a_single_value_needs_no_brackets(bearing, key, alone, listed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = _bearing(_rotor(dict(bearing, **{key: alone})))
        other = _bearing(_rotor(dict(bearing, **{key: listed})))
    assert np.allclose(one.K(0), other.K(0))


def test_a_list_typed_without_brackets_is_the_list():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = _bearing(_rotor(dict(CYLINDRICAL, speed="1000, 1500")))
        bracketed = _bearing(_rotor(dict(CYLINDRICAL, speed="[1000, 1500]")))
    assert len(bare.speed) == 2
    assert np.allclose(bare.speed, bracketed.speed)
    assert np.allclose(bare.K(150), bracketed.K(150))


def test_a_name_with_commas_stays_a_name():
    assert (
        extract_kwargs({"tag": "1, 2"}, {}, "CylindricalBearing", [])["tag"] == "1, 2"
    )
