"""The schema derived from ROSS by introspection."""

import inspect
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Since slice 4 app.py is only the entry point. Each name below comes from its
# own layer -- and this explicit import is the documentation of that.
import ross as rs
from ross.interface.app import app
from ross.interface.api.security import SESSION_TOKEN

from ross.interface.domain import element_registry, units
from ross.interface.domain.field_catalog import FIELDS, SECTIONS
from ross.interface.domain.schema import LANGUAGES, build_schema, schema_problems

AUTH = {"X-ROSS-Token": SESSION_TOKEN}  # noqa: F405


@pytest.fixture
def client():
    app.config["TESTING"] = True  # noqa: F405
    with app.test_client() as client:  # noqa: F405
        yield client


def _every_field(language="en"):
    for category, subtypes in build_schema(language)["categories"].items():
        for subtype, definition in subtypes.items():
            for field in definition["fields"]:
                yield category, subtype, definition, field


# --- alignment with the installed ROSS --------------------------------------


def test_schema_has_no_unknown_fields():
    """The central test: no form field that ROSS would refuse.

    It is the automatic version of the manual check that revealed BE-13. If a ROSS
    refactor renames a parameter, this fails the same day."""
    assert schema_problems() == []


def test_every_form_field_is_accepted_by_its_class():
    """Belt and braces: checks straight against the signature, without the schema."""
    from ross.interface.domain.schema import _class_signature, _parameter_name

    problems = []
    for category, subtypes in FIELDS.items():
        for subtype, fields in subtypes.items():
            klass = element_registry.ross_constructor(category, subtype)
            _, parameters, _, _ = _class_signature(klass)
            for field in fields:
                target = _parameter_name(klass, field["name"])
                if target not in parameters:
                    problems.append(
                        f"{category}/{subtype}.{field['name']} -> rs.{klass}"
                    )
    assert problems == []


def test_coupling_has_no_link_node_field():
    """BE-16: CouplingElement does not accept n_link, nor inherit from anyone who does."""
    assert "n_link" not in inspect.signature(rs.CouplingElement.__init__).parameters  # noqa: F405
    assert not any(c["name"] == "n_link" for c in FIELDS["couplings"]["BASIC"])


def test_squeeze_film_has_no_colour_field():
    """BE-17: SqueezeFilmDamper does not accept color and has no **kwargs."""
    assert "color" not in inspect.signature(rs.SqueezeFilmDamper.__init__).parameters  # noqa: F405
    assert not any(c["name"] == "color" for c in FIELDS["bearings"]["SqueezeFilm"])


# --- shape of the schema ----------------------------------------------------


def test_schema_covers_every_registered_form():
    schema_payload = build_schema()["categories"]
    for category, subtypes in element_registry.ELEMENTS.items():
        assert category in schema_payload
        for subtype in subtypes:
            assert subtype in schema_payload[category], (
                f"{category}/{subtype} fora do schema"
            )
            assert schema_payload[category][subtype]["fields"], (
                f"{category}/{subtype} with no fields"
            )


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_field_has_a_label(language):
    for category, subtype, _, field in _every_field(language):
        assert field["label"], (
            f"{category}/{subtype}.{field['name']} with no label in {language}"
        )


def test_labels_differ_between_languages_where_expected():
    """Symbols like kxx are the same in both languages; running text should not be."""
    english = {(c, s, f["name"]): f["label"] for c, s, _, f in _every_field("en")}
    portuguese = {(c, s, f["name"]): f["label"] for c, s, _, f in _every_field("pt")}
    assert english.keys() == portuguese.keys()
    translated = sum(1 for k in english if english[k] != portuguese[k])
    assert translated > 200, (
        f"only {translated} labels differ -- incomplete translation?"
    )


def test_units_come_from_the_domain_module():
    """The unit comes from units.py -- from the class itself or from a base of
    it, and for an alternative constructor ("Class.method") from its own entry
    first."""
    for category, subtype, definition, field in _every_field():
        constructor = definition["ross_class"]
        klass = getattr(rs, constructor.split(".")[0])  # noqa: F405
        names = [base.__name__ for base in klass.__mro__]
        if "." in constructor:
            names.insert(0, constructor)
        candidates = [
            units.UNITS_MAPPING.get(name, {}).get(field["name"]) for name in names
        ]
        expected = next((u for u in candidates if u), None)
        assert field["unit"] == expected, f"{category}/{subtype}.{field['name']}"
        if field["unit"]:
            assert field["unit_options"], f"{field['name']} with no unit alternatives"
            assert field["unit_options"][0] == field["unit"]


def test_units_are_inherited_like_the_parameters_are():
    """BallBearingElement declares no units, but cxx inherits N*s/m from the base.

    The old injectUnits looked the key up in every class and got it right by
    accident; searching by class alone would lose the unit selector here."""
    fields = {
        c["name"]: c
        for c in build_schema()["categories"]["bearings"]["BallBearing"]["fields"]
    }
    assert units.UNITS_MAPPING["BallBearingElement"] == {}
    assert fields["cxx"]["unit"] == "N*s/m"
    assert fields["cxx"]["unit_options"] == ["N*s/m", "lbf*s/in"]


def test_booleans_are_inferred_from_ross_defaults():
    """cavitation, shear_effects e afins viram select True/False sozinhos."""
    booleans = {f["name"] for _, _, _, f in _every_field() if f["control"] == "boolean"}
    assert {"shear_effects", "rotary_inertia", "gyroscopic", "cavitation"} <= booleans
    for _, _, _, field in _every_field():
        if field["control"] == "boolean":
            assert field["options"] == ["false", "true"]


def test_help_text_comes_from_ross_docstrings():
    fields = list(_every_field())
    with_help = [f for _, _, _, f in fields if f["help"]]
    assert len(with_help) > 300, (
        f"only {len(with_help)} of {len(fields)} fields with help"
    )


def test_a_parameter_written_without_the_space_before_the_colon_keeps_its_help():
    """numpydoc writes "name : type"; some of ROSS's docstrings write
    "name: type" (DiskElement's scale_factor, from_geometry's material). The
    parser used to miss those lines, so the field had no help and the one
    above it swallowed its text."""
    disk = build_schema("en")["categories"]["disks"]
    basic = {f["name"]: f["help"] for f in disk["BASIC"]["fields"]}
    geometry = {f["name"]: f["help"] for f in disk["Geometry"]["fields"]}
    assert (
        basic["scale_factor"] == "The scale factor is used to scale the disk drawing."
    )
    assert "scale_factor" not in basic["tag"]
    assert geometry["material"] == "Disk material."


def test_the_geometry_disk_form_is_from_geometry_in_millimetres():
    """The second disk form calls DiskElement.from_geometry: its fields are
    that method's parameters -- not DiskElement's -- and the three dimensions
    are typed in millimetres, like a shaft's."""
    definition = build_schema("en")["categories"]["disks"]["Geometry"]
    assert definition["ross_class"] == "DiskElement.from_geometry"
    assert definition["exists"] and definition["not_in_form"] == []
    fields = {f["name"]: f for f in definition["fields"]}
    for name in ("width", "o_d", "i_d"):
        assert fields[name]["known_to_ross"]
        assert fields[name]["unit"] == "mm"
        assert fields[name]["group"] == "main"
    assert fields["material"]["control"] == "material_ref"
    assert "m" not in fields and "Ip" not in fields


def test_the_geometry_gear_form_names_rosss_diameters_for_what_they_are():
    """GearElement.from_geometry calls its o_d the pitch diameter and its i_d
    the bore; the form says so, and the angles keep their degrees."""
    definition = build_schema("en")["categories"]["gears"]["Geometry"]
    assert definition["ross_class"] == "GearElement.from_geometry"
    assert definition["exists"] and definition["not_in_form"] == []
    fields = {f["name"]: f for f in definition["fields"]}
    assert fields["o_d"]["label"] == "Pitch Diameter"
    assert fields["i_d"]["label"] == "Bore Diameter"
    for name in ("width", "o_d", "i_d"):
        assert (fields[name]["unit"], fields[name]["group"]) == ("mm", "main")
    for name in ("pr_angle", "helix_angle"):
        assert (fields[name]["unit"], fields[name]["group"]) == ("deg", "advanced")
    assert fields["n_teeth"]["known_to_ross"]
    assert "m" not in fields and "pitch_diameter" not in fields


def test_material_fields_use_the_dynamic_control():
    for _, _, _, field in _every_field():
        if field["name"] == "material":
            assert field["control"] == "material_ref"


def test_sections_are_translated():
    for key, labels in SECTIONS.items():
        for language in LANGUAGES:
            assert labels[language], f"section {key} with no label in {language}"


# --- endpoint -------------------------------------------------------------


def test_schema_endpoint_requires_token(client):
    assert client.get("/api/schema/elements").status_code == 403


def test_schema_endpoint_serves_both_languages(client):
    for language in LANGUAGES:
        response = client.get(f"/api/schema/elements?lang={language}", headers=AUTH)
        assert response.status_code == 200
        body = response.get_json()
        assert body["language"] == language
        assert set(body["categories"]) == set(element_registry.ELEMENTS)


def test_schema_endpoint_falls_back_to_english(client):
    response = client.get("/api/schema/elements?lang=klingon", headers=AUTH)
    assert response.get_json()["language"] == "en"


# --- the frontend may no longer carry copies of the tables ------------------


REMOVED_FROM_THE_FRONTEND = (
    "FormTemplates",
    "UNITS_MAPPING",
    "UNIT_ALTERNATIVES",
    "injectUnits",
    "ClassMap",
)


def test_frontend_no_longer_declares_the_domain_tables():
    """A guard against a dangling reference: `node --check` does not catch it.

    `UNIT_ALTERNATIVES[x]` is syntactically valid even without the variable
    existing; it only throws as a ReferenceError when someone opens a dashboard.
    It happened."""
    from ross.interface.tests.frontend_source import code_lines

    occurrences = []
    for module, number, line in code_lines():
        for name in REMOVED_FROM_THE_FRONTEND:
            if name in line:
                occurrences.append(f"{module}:{number}: {name}")
    assert occurrences == [], "referencias a tabelas removidas: " + "; ".join(
        occurrences
    )


def test_schema_payload_carries_unit_alternatives():
    """The analysis dashboards switch units outside the element schema."""
    alternatives = build_schema()["unit_alternatives"]
    assert alternatives["RPM"] == ["RPM", "rad/s", "Hz"]
    assert alternatives == units.UNIT_ALTERNATIVES
