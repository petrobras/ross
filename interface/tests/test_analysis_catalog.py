# -*- coding: utf-8 -*-
"""The analysis catalogue leaves the frontend and starts being cross-checked.

Until this point the 147 fields of the analysis forms lived in
`AnalysisDashboards`, 198 lines in the frontend -- and the **same names** were
already written on the other side: each runner's `spec` reads its parameters by
name, and each runner declared a `PLOT_PARAMS`. Two hand-kept lists, in two
languages, with nothing checking that they agreed.

## Why it was not derived, the way the elements are

The slice started with a measurement, not with a design. Of the 147 fields, 63
appear in some ROSS signature and only **28** in a `run_*` parameter. The rest
are not missing by oversight -- they cannot come from a signature:

* composition: `speed_min`/`speed_max`/`speed_steps` are three fields that
  become one `np.linspace` in `speed_range`. There is no `speed_min` in ROSS;
* list editors: `probes`, `forces`, `unbalances`, `inps`;
* presentation: `plot_type`, default units, conditional visibility, sliders.

A form half derived and half declared would be harder to understand than either
pure one. So the copy does not go away -- and the way out is `getEffectiveNodes`'
one: the rule is written on one side only, and the test cross-checks.

## What the slice found

* three fields of `freq_response` were declared as compute and are not: they are
  read in `plot` and appeared in no tuple;
* `test_declared_plot_parameters_are_actually_used` was **vacuous**: it searched
  for the parameter name in the runner source, and the tuple itself was in that
  source;
* Campbell's `plot_type` is read by no runner -- what consumes it is the screen.
  It gained the role `interface`, and is enforced from the frontend side."""

import inspect
import io
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain.analysis_catalog import ANALYSES, ROSS_METHOD, fields, catalog
from services.analysis import REGISTRY
from test_runners import PARAMS as PARAMS_BASE, SpyParams, FakeRotor

NAMES = sorted(ANALYSES)
ROTOR_REQUEST = FakeRotor()

try:
    import ross as rs

    HAS_ROSS = True
except Exception:  # pragma: no cover
    HAS_ROSS = False

needs_ross = pytest.mark.skipif(not HAS_ROSS, reason="requires ROSS installed")


@pytest.fixture
def client_with_token():
    """An application client with the session token already in the header."""
    from api import create_app
    from api.security import SESSION_TOKEN

    application = create_app()
    application.config["TESTING"] = True
    with application.test_client() as client:
        original = client.post

        def with_token(*args, **kwargs):
            header = dict(kwargs.pop("headers", {}))
            header["X-ROSS-Token"] = SESSION_TOKEN
            return original(*args, headers=header, **kwargs)

        client.post = with_token
        yield client


def _runner_source(name):
    """The runner source plus the helpers', without comments.

    The helpers (`probes`, `unbalances`, `units`) read the parameters inside
    `base.py`; searching only in the runner's module would give a false negative."""
    modules = [
        sys.modules[type(REGISTRY[name]).__module__],
        sys.modules["services.analysis.base"],
    ]
    js = "\n".join(inspect.getsource(m) for m in modules)
    return "\n".join(
        line for line in js.split("\n") if not line.strip().startswith("#")
    )


def _read_by_the_spec(name):
    """Everything the `spec` consults, walking the options of each <select>.

    A single battery of parameters is not enough: `misalignment` reads different
    fields depending on `mis_type`, and an unvisited branch would show up as a
    dead field."""
    runner = REGISTRY[name]
    variants = [{}]
    for field in ANALYSES[name]:
        for option in field.get("options", ()) or ():
            variants.append({field["id"]: option})

    seen = set()
    for extra in variants:
        values = {c["id"]: c.get("val") for c in ANALYSES[name]}
        values.update(PARAMS_BASE.get(name, {}))
        values.update(extra)
        spy = SpyParams(values)
        try:
            runner.spec(spy, ROTOR_REQUEST)
        except Exception:
            pass  # branches needing the real ROSS; what matters is what was read
        seen |= spy.read_names
    return seen


# --- o catalogo e o runner concordam ----------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_calculation_field_is_read_by_the_spec(name):
    """A field the computation does not read is a dead field on the screen.

    The user fills it in, presses Update, and the number changes nothing -- with no
    error."""
    read_names = _read_by_the_spec(name)
    dead = [
        c["id"]
        for c in ANALYSES[name]
        if c["role"] == "compute"
        and c["id"] not in read_names
        and (c["id"] + "_unit") not in read_names
    ]
    assert dead == [], "%s declares %s as compute and the spec does not read it" % (
        name,
        dead,
    )


@pytest.mark.parametrize("name", NAMES)
def test_every_drawing_field_is_read_by_the_runner(name):
    """A plot field has to be read by `plot` -- or there is no reason for it."""
    js = _runner_source(name)
    dead = [
        c["id"]
        for c in ANALYSES[name]
        if c["role"] == "plot"
        and '"%s"' % c["id"] not in js
        and "'%s'" % c["id"] not in js
    ]
    assert dead == [], "%s declares %s as plot and does not read it" % (name, dead)


@pytest.mark.parametrize("name", NAMES)
def test_an_interface_field_is_read_by_nobody_in_the_backend(name):
    """`interface` is the role of what only the screen consumes; if the runner reads
    it, it lied."""
    js = _runner_source(name)
    intruders = [
        c["id"]
        for c in ANALYSES[name]
        if c["role"] == "interface"
        and ('"%s"' % c["id"] in js or "'%s'" % c["id"] in js)
    ]
    assert intruders == [], "%s: %s has role interface and the backend reads it" % (
        name,
        intruders,
    )


def test_the_interface_fields_are_used_by_the_screen():
    """...and, from the other side, the screen really has to use it."""
    frontend = ""
    for handle in ("analysis.js", "campbell.js"):
        with io.open(
            os.path.join(ROOT, "frontend", "features", handle),
            encoding="utf-8",
            newline="",
        ) as js:
            frontend += js.read()
    for name, items in ANALYSES.items():
        for field in items:
            if field["role"] != "interface":
                continue
            assert (
                "'%s'" % field["id"] in frontend or '"%s"' % field["id"] in frontend
            ), "%s/%s has role interface and the screen does not use it" % (
                name,
                field["id"],
            )


# Parameters the runner consults on purpose with no field. The list is short and
# each entry has its reason written down -- without that it would become a dump.
# Emptied when ROSS fixed `bearing_frequency_range` at commit 2a253e6 and the
# field came back to the form. Kept as an empty dictionary and not deleted: the
# mechanism -- an exception that has to carry its reason, and a control that
# fails when the reason expires -- is what let this entry leave, and the next
# parameter with no field will need it.
NO_FIELD = {}


@pytest.mark.parametrize("name", NAMES)
def test_every_parameter_the_runner_reads_has_a_field(name):
    """A parameter with no field is functionality the user cannot reach.

    That is how `controller_transfer_function` disappeared from the MagneticBearing
    form: the backend accepted it, the screen did not offer it."""
    declared = {c["id"] for c in ANALYSES[name]}
    read_names = {c for c in _read_by_the_spec(name) if not c.endswith("_unit")}
    missing = sorted(p for p in read_names - declared if (name, p) not in NO_FIELD)
    assert missing == [], "%s reads %s and there is no field" % (name, missing)


def test_the_documented_exceptions_are_still_needed():
    """Control: an exception that stopped being necessary drops off the list.

    Without this, `NO_FIELD` only grows -- each entry suppresses forever a check
    that may no longer need suppressing.

    It is empty today, and an empty loop asserts nothing -- which is worth
    saying out loud rather than leaving as a silent pass. The only entry there
    ever was left because ROSS fixed the defect behind it, which is this test
    working exactly as written."""
    assert NO_FIELD == {} or all(NO_FIELD.values()), "an exception with no reason"
    for (name, parameter), reason in NO_FIELD.items():
        read_names = {c for c in _read_by_the_spec(name) if not c.endswith("_unit")}
        assert parameter in read_names, (
            "%s no longer reads %r -- take the exception off the list (%s)"
            % (name, parameter, reason)
        )


# --- o catalogo e o ROSS -----------------------------------------------------


@needs_ross
@pytest.mark.parametrize("name", NAMES)
def test_every_mapped_field_is_a_real_ross_parameter(name):
    """The 28 fields pointing at a `run_*` parameter still exist.

    It is BE-13 again, on the analysis side: when ROSS renamed 14 seal fields, the
    interface went on offering the old names and the error only showed up for
    whoever filled in the right field."""
    method = getattr(rs.Rotor, ROSS_METHOD[name], None)
    if method is None:
        pytest.skip("%s does not exist in this version of ROSS" % ROSS_METHOD[name])
    signature = set(inspect.signature(method).parameters)
    pointed_at = [
        (c["id"], c["ross_param"]) for c in ANALYSES[name] if "ross_param" in c
    ]
    missing_ones = [target for _, target in pointed_at if target not in signature]
    assert missing_ones == [], "%s: %s no longer exists in %s" % (
        name,
        missing_ones,
        ROSS_METHOD[name],
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_declared_ross_method_is_the_one_the_runner_calls(name):
    """The method name in the catalogue has to be the one the runner calls.

    Writing that map by hand has cost before: I had put `run_harmonic_balance` and
    `run_clearance` when ROSS calls `run_harmonic_balance_response` and
    `run_clearance_analysis`. `getattr` gave back None and the rename test
    **skipped in silence** -- a guard that guards nothing is worse than none,
    because it looks like coverage."""
    js = inspect.getsource(sys.modules[type(REGISTRY[name]).__module__])
    called = set(re.findall(r"rotor\.(run_\w+)\(", js))
    assert called, "%s calls no run_* -- the sweep stopped working" % name
    assert ROSS_METHOD[name] in called, (
        "%s: the catalogue says %s, the runner calls %s"
        % (name, ROSS_METHOD[name], sorted(called))
    )


def test_every_field_that_points_at_ross_is_accounted_for():
    """Control: if `ross_param` left the catalogue, the test above would pass empty.

    The number is exact and not a floor. A floor only ever moves down: when the
    count fell from 37 to 34 the honest reading was "three fields deliberately
    stopped pointing at ROSS", and a `>= 35` would have been *lowered to 34* to
    make the suite green -- which is how a control quietly stops controlling.
    Written exactly, the same event forces this docstring to say what happened.

    37 -> 34: clearance's `node`, `unbalance_magnitude` and `unbalance_phase`
    became one `unbalances` table, which points at no single ROSS parameter
    because it feeds three of them at once."""
    pointed_at = sum(
        1 for items in ANALYSES.values() for c in items if "ross_param" in c
    )
    assert pointed_at == 34, "%d fields point at ROSS, not 34" % pointed_at


# --- the catalogue and the screen ---------------------------------------------


def test_the_screen_offers_exactly_what_the_catalogue_has():
    """The twelve options left the HTML: the screen offers what the catalogue says.

    Until the i18n slice the titles were written in two places -- the `<option>` of
    index.html and a `typeNames` in the JS. Now they come from the catalogue,
    already translated, and the guard becomes what always mattered: the catalogue
    and the runner registry cover the same analyses."""
    with io.open(
        os.path.join(ROOT, "frontend", "index.html"), encoding="utf-8", newline=""
    ) as handle:
        html = handle.read()
    block = html[html.index('id="analysis-type"') :]
    block = block[: block.index("</select>")]
    fixed = {v for v in re.findall(r'<option value="([^"]*)"', block) if v}
    assert fixed == set(), "analysis option hard-coded in the HTML: %s" % sorted(fixed)

    from frontend_source import source

    assert "fillAnalysisTypes" in source()
    assert "typeNames" not in source()

    assert set(REGISTRY) == set(ANALYSES)


def test_the_frontend_no_longer_declares_analysis_fields():
    """The 198 lines must not come back."""
    from frontend_source import source

    assert "AnalysisDashboards" not in source()
    assert "analysisFieldsFor(" in source()


@pytest.mark.parametrize("name", NAMES)
def test_every_field_has_a_label_in_both_languages(name):
    for field in ANALYSES[name]:
        for language in ("en", "pt"):
            assert field["label"].get(language), "%s/%s with no label in %s" % (
                name,
                field["id"],
                language,
            )


@pytest.mark.parametrize("language", ["en", "pt"])
def test_the_catalogue_serialises_with_the_chosen_language(language):
    output = catalog(language)
    assert set(output) == set(ANALYSES)
    for name, items in output.items():
        for field, original in zip(items, ANALYSES[name], strict=True):
            assert isinstance(field["label"], str)
            assert field["label"] == original["label"][language]


def test_reading_the_catalogue_does_not_mutate_it():
    """`fields()` gives back a copy: the screen writes `val` to fill the card in."""
    before = json.dumps(ANALYSES["ucs"], sort_keys=True, ensure_ascii=False)
    copy_of = fields("ucs", "en")
    copy_of[0]["val"] = "alterado"
    copy_of[0]["label"] = "alterado"
    assert json.dumps(ANALYSES["ucs"], sort_keys=True, ensure_ascii=False) == before


def test_the_javascript_fixture_matches_the_catalogue():
    """The JS stand-ins answer with the real catalogue, not an invented one.

    A catalogue written by hand in the tests would make both ends agree with
    themselves. The golden file is generated from here and this test keeps it equal."""
    from domain.analysis_catalog import titles
    from domain.compatibility import table

    path = os.path.join(ROOT, "tests", "golden", "analysis_schema.json")
    with io.open(path, encoding="utf-8") as handle:
        golden = json.load(handle)
    expected = {
        "fields": catalog("en"),
        "titles": titles("en"),
        "unsupported": table("en"),
    }
    assert golden == expected, (
        "regenere tests/golden/analysis_schema.json a partir da rota"
    )


def test_the_frozen_dashboards_still_match_the_catalogue():
    """The golden file of the old form is what the port has to reproduce.

    `tests/golden/analysis_dashboards.json` was taken from the real
    `AnalysisDashboards` by evaluating the module -- not with a parser of mine,
    which would fail on the first unforeseen syntax exception.
    `tests/js/test_analysis_form.js` builds the twelve panels from it and from the
    catalogue and requires identical HTML; here the comparison is of the data,
    field by field.

    Three new fields stay out of the comparison because they did not exist before:
    `role` and `ross_param`, from the catalogue (slice 4), and `deps_de`, which
    says which field each conditional visibility depends on (FE-07, slice 5).
    Their presence is enforced elsewhere -- in this same file for `deps_de` -- so
    that taking them out of the comparison does not become taking them out of the
    checking.

    `CHANGED_ON_PURPOSE` is the other kind of exception, and it costs more. This
    golden's whole claim is "the port did not change the form", and once a form
    is deliberately changed that claim is simply no longer true of it. The
    tempting move -- regenerate the file -- would make the claim untrue of every
    analysis at once, silently, because a golden regenerated from the code it
    guards agrees with anything. So the file stays frozen, the one analysis we
    changed is named here with its reason, and the control below fails if a
    named analysis stops differing: an exception that outlives its cause is a
    hole with a comment on it."""
    path = os.path.join(ROOT, "tests", "golden", "analysis_dashboards.json")
    with io.open(path, encoding="utf-8") as handle:
        old_one = json.load(handle)

    new = catalog("en")
    assert set(old_one) == set(new)
    for name in old_one:
        if name in CHANGED_ON_PURPOSE:
            continue
        assert len(old_one[name]) == len(new[name]), "%s changed size" % name
        for old, current in zip(old_one[name], new[name], strict=True):
            cleaned_up = {
                k: v
                for k, v in current.items()
                if k not in ("role", "ross_param", "deps_de")
            }
            assert cleaned_up == old, "%s/%s: %r != %r" % (
                name,
                old["id"],
                cleaned_up,
                old,
            )


# The analyses whose form we deliberately changed after the port, and why. Read
# by the test above and by its control just below.
CHANGED_ON_PURPOSE = {
    "clearance": (
        "the single `node` box beside list-valued magnitude and phase could "
        "never take more than one value, and the one value it did take reached "
        "ROSS as a one-element array where a number was expected -- a "
        "DeprecationWarning until numpy 2.5, an error after it. The three "
        "fields became one unbalance table."
    ),
    "ucs": (
        "the bearing frequency range came back. It was taken out of the form "
        "because ROSS raised on any value for it (@check_units turned the "
        "sequence into an array and `if bearing_frequency_range:` refused it); "
        "commit 2a253e6 fixed that line, and the field returned as the pair "
        "`bearing_freq_min`/`bearing_freq_max`, which is how the campbell "
        "already asks for a range."
    ),
}


def test_every_deliberate_form_change_still_differs_from_the_frozen_one():
    """Control: an exception that no longer excepts anything has to be removed.

    If `clearance` were ever brought back to the old three fields, the entry
    above would go on silently skipping the comparison for it. The rule is the
    one `DELIBERATE` follows in the Portuguese sweep: every exception carries a
    test that it is still needed."""
    path = os.path.join(ROOT, "tests", "golden", "analysis_dashboards.json")
    with io.open(path, encoding="utf-8") as handle:
        old_one = json.load(handle)
    new = catalog("en")
    for name, reason in CHANGED_ON_PURPOSE.items():
        assert name in old_one, "%s is not in the frozen file" % name
        assert reason.strip(), "%s: an exception with no reason" % name
        cleaned_up = [
            {
                k: v
                for k, v in field.items()
                if k not in ("role", "ross_param", "deps_de")
            }
            for field in new[name]
        ]
        assert cleaned_up != old_one[name], (
            "%s matches the frozen form again: take it off CHANGED_ON_PURPOSE "
            "so the comparison covers it once more" % name
        )


@pytest.mark.parametrize("name", NAMES)
def test_every_conditional_field_names_the_field_it_depends_on(name):
    """`deps` with no `deps_de` goes back to comparing against every select."""
    missing = [
        c["id"] for c in ANALYSES[name] if c.get("deps") and not c.get("deps_de")
    ]
    assert missing == [], "%s: %s has deps with no deps_de" % (name, missing)


@pytest.mark.parametrize("name", NAMES)
def test_the_field_a_dependency_points_at_exists_and_offers_the_values(name):
    """The field pointed at has to exist, be a select, and offer the values.

    A `deps_de` for a field that does not exist would hide the dependent field
    forever -- and in silence, which is how this class of defect always shows up in
    this project."""
    fields = {c["id"]: c for c in ANALYSES[name]}
    for field in ANALYSES[name]:
        if not field.get("deps"):
            continue
        owner = fields.get(field["deps_de"])
        assert owner is not None, "%s/%s depends on %s, which does not exist" % (
            name,
            field["id"],
            field["deps_de"],
        )
        assert owner["type"] == "select", (
            "%s/%s depends on a field that is not a select" % (name, field["id"])
        )
        options = owner.get("options") or []
        for value in field["deps"]:
            assert value in options, "%s/%s expects %r, which %s does not offer" % (
                name,
                field["id"],
                value,
                owner["id"],
            )


@pytest.mark.parametrize("name", NAMES)
def test_a_dependency_value_belongs_to_exactly_one_select(name):
    """The coincidence that made FE-07 work, now written down as a rule.

    As long as **a value used in `deps`** appears in the options of only one
    select, reading "every select on the card" gives the same result as reading the
    right one. It was true, and it was luck.

    The first version of this test forbade *any* repeated option between two
    selects -- and failed on five analyses, with `True`/`False`, `RPM`/`rad/s` and
    `m`/`mm`. Repetition like that is legitimate and not ambiguous: none of them is
    a dependency value. The right invariant is narrower, and it is this one."""
    owners = {}
    for field in ANALYSES[name]:
        if field["type"] != "select":
            continue
        for value in field.get("options") or []:
            owners.setdefault(value, []).append(field["id"])

    used = {v for field in ANALYSES[name] for v in (field.get("deps") or ())}
    ambiguous = {v: owners.get(v, []) for v in used if len(owners.get(v, [])) != 1}
    assert ambiguous == {}, "%s: deps value in zero or two selects -- %s" % (
        name,
        ambiguous,
    )


def test_the_screen_reads_the_field_the_catalogue_names():
    """`checkDeps` reads a specific select, not every one on the card."""
    from frontend_source import source

    js = source()
    assert "data-deps-de=" in js
    assert "item.dataset.depsDe" in js
    assert 'querySelectorAll(`select[id$="-${uniqueId}"]`)' not in js
