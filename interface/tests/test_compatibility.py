# -*- coding: utf-8 -*-
"""Which analysis-and-conversion combinations ROSS does not support, and how to warn.

The user converted the rotor to torsional, ran an analysis and got the raw
error from the library. Worse than that: in some combinations **there is no
error** -- the conversion swaps the rotor's matrix methods, and an analysis that
builds its own gives back a chart carrying the reduced model's badge and the
numbers of the full model.

The table in `domain/compatibility.py` was MEASURED, combination by
combination, against a concrete version of ROSS (see
`tools/conversion_probe.py`). These tests hold its three sides: the route
refuses by name, the screen reads the same table the route enforces, and the
exported script warns instead of keeping quiet."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain.analysis_catalog import ANALYSES
from waiting import answer_for
from domain.compatibility import (
    RAISES,
    IGNORED,
    MODEL_NAMES,
    UNSUPPORTED,
    NOT_VALIDATED,
    reason,
    supported,
    table,
)

try:
    import ross  # noqa: F401

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


def _answer(client, response):
    """`answer_for`, carrying the token the fixture above only adds to posts.

    The fixture wraps `post` and not `get`, and asking about a job is a `get`.
    Without this the polling would be refused by the session guard and every
    test here would read a 403 as the analysis's answer.
    """
    from api.security import SESSION_TOKEN

    return answer_for(client, response, {"X-ROSS-Token": SESSION_TOKEN})


def _script(analyses, conversion):
    from domain.python_export import build_script

    project = {
        "materials": [{"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [{"L": "100", "odl": "50", "idl": "0", "material": "Steel"}],
        "bearings": [{"n": "0", "kxx": "1e6", "cxx": "1e3"}],
        "disks": [],
        "gears": [],
        "seals": [],
        "couplings": [],
        "pointmasses": [],
    }
    return build_script(project, analyses, conversion)


# The key travels as text: a pair in `parametrize` is unpacked by some runners
# as though it were two arguments.
@pytest.mark.parametrize("key", sorted("%s+%s" % pair for pair in UNSUPPORTED))
def test_every_unsupported_entry_names_something_real(key):
    """An entry with a wrong name never fires, and nobody notices."""
    analysis, conversion = key.split("+")
    assert analysis in ANALYSES, "%s is not an analysis" % analysis
    assert conversion in MODEL_NAMES, "%s is not a rotor model" % conversion
    assert conversion != "", (
        "6 DoF is the original model; nothing can be incompatible with it"
    )

    entry = UNSUPPORTED[(analysis, conversion)]
    assert entry["failure"] in (RAISES, IGNORED, NOT_VALIDATED)
    assert entry["origin"] in ("documented", "measured", "domain")
    for language in ("en", "pt"):
        assert len(entry["reason"][language]) > 40, "reason too short in %s" % language


def test_the_route_refuses_the_combination_by_name(client_with_token):
    """The user gets the reason, not the ROSS traceback.

    It is BE-10 again: an error that answers a different question from the one
    asked costs more than the defect."""
    for (analysis, conversion), entry in UNSUPPORTED.items():
        response = _answer(
            client_with_token,
            client_with_token.post(
                "/run_analysis",
                json={
                    "analysis_type": analysis,
                    "conversion_type": conversion,
                    "params": {},
                    "project": {},
                },
            ),
        )
        assert response.status_code == 400, "%s/%s passou" % (analysis, conversion)
        message = response.get_json()["message"]
        assert message == entry["reason"]["en"]


def test_a_supported_combination_is_not_refused_by_the_table(client_with_token):
    """Control: if the route refused everything, the test above would pass just the same."""
    response = _answer(
        client_with_token,
        client_with_token.post(
            "/run_analysis",
            json={
                "analysis_type": "campbell",
                "conversion_type": "torsional",
                "params": {},
                "project": {},
            },
        ),
    )
    message = (response.get_json() or {}).get("message", "")
    assert message, (
        "the control stopped reading a message at all: it would pass on any "
        "answer, which is exactly what it exists not to do"
    )
    reasons = {e["reason"]["en"] for e in UNSUPPORTED.values()}
    assert message not in reasons, "the table refused a combination that is not in it"


def test_the_screen_reads_the_same_table_the_route_enforces():
    """A second table in the frontend would diverge -- and in silence."""
    from frontend_source import source

    js = source()
    assert "analysisUnsupported(" in js
    assert "unsupported" in js
    # The reason must not be written in the JS: it comes from the server.
    for entry in UNSUPPORTED.values():
        for language in ("en", "pt"):
            excerpt = entry["reason"][language][:40]
            assert excerpt not in js, "the reason was copied into the frontend"


@pytest.mark.parametrize("language", ["en", "pt"])
def test_the_table_serialises_in_both_languages(language):
    output = table(language)
    for analysis, byConversion in output.items():
        for conversion, entry in byConversion.items():
            assert isinstance(entry["reason"], str)
            assert (
                entry["reason"]
                == UNSUPPORTED[(analysis, conversion)]["reason"][language]
            )


def test_supported_answers_the_inverse_of_the_table():
    for analysis in ANALYSES:
        for conversion in MODEL_NAMES:
            expected = (analysis, conversion) not in UNSUPPORTED
            assert supported(analysis, conversion) is expected
            assert (reason(analysis, conversion) is None) is expected


def test_the_table_the_screen_gets_is_the_one_the_route_enforces():
    """A second table on the screen would diverge -- and in silence."""
    served = table("en")
    flattened = {(a, cv) for a, byConversion in served.items() for cv in byConversion}
    assert flattened == set(UNSUPPORTED)
    for (analysis, conversion), entry in UNSUPPORTED.items():
        assert served[analysis][conversion]["failure"] == entry["failure"]
        assert not supported(analysis, conversion)


def test_the_domain_decisions_say_so(client_with_token):
    """The entries of `domain` origin run without error -- and that is why they are a
    decision.

    Crack, misalignment and rubbing in 4 DoF **work** on ross 2.3.0: they do not
    throw and do not give back the 6 DoF chart. They are in the table because the
    library's fault models were developed for the 6 DoF rotor, and running is not
    the same as being validated. No probe measures that.

    The test requires the reason to be written down: a domain entry with no
    justification becomes superstition on the next reading."""
    domain_origin = [
        (a, c) for (a, c), e in UNSUPPORTED.items() if e["origin"] == "domain"
    ]
    assert domain_origin, "no domain entry -- the origin became a dead letter"
    for analysis, conversion in domain_origin:
        entry = UNSUPPORTED[(analysis, conversion)]
        assert entry["failure"] == NOT_VALIDATED
        for language in ("en", "pt"):
            text = entry["reason"][language].lower()
            assert "6 dof" in text, (
                "%s/%s does not say which model it was built for"
                % (analysis, conversion)
            )


def test_the_exported_script_warns_about_a_refused_combination():
    """The interface prevents it; the script is the user's code and cannot keep quiet.

    Refusing the export would be deciding for them. What is right is that they
    know, in the file, what the interface knew -- otherwise the script fails in
    their hands with the raw ROSS error, or worse, runs and gives back the numbers
    of the full model under the reduced model's name."""
    script = _script([{"type": "ucs", "params": {"num_modes": "4"}}], "torsional")
    assert script.startswith("# ")
    assert "WARNING" in script.split("import ross")[0]
    assert "ucs" in script.split("import ross")[0]
    assert UNSUPPORTED[("ucs", "torsional")]["reason"]["en"][:40] in " ".join(
        script.split("import ross")[0].split()
    )


def test_a_supported_combination_exports_without_a_header():
    """A constant header would change every script and say nothing in most of them."""
    script = _script(
        [{"type": "campbell", "params": {"speed_max": "4000"}}], "torsional"
    )
    assert script.startswith("import ross as rs")
    script = _script([{"type": "ucs", "params": {"num_modes": "4"}}], "")
    assert script.startswith("import ross as rs")


def test_the_warned_script_is_still_valid_python():
    """The warning is a comment; a forgotten `#` would be a SyntaxError in the download."""
    import ast as _ast

    script = _script(
        [
            {"type": "ucs", "params": {"num_modes": "4"}},
            {"type": "static", "params": {"plot_type": "Deformation"}},
        ],
        "torsional",
    )
    _ast.parse(script)
    header = script.split("import ross")[0].strip().split("\n")
    assert all(line.startswith("#") for line in header if line), header


def test_the_warning_names_every_refused_analysis_in_the_script():
    """Two refused analyses give two entries -- and a repeated one gives just one."""
    script = _script(
        [
            {"type": "ucs", "params": {"num_modes": "4"}},
            {"type": "ucs", "params": {"num_modes": "8"}},
            {"type": "static", "params": {"plot_type": "Deformation"}},
        ],
        "torsional",
    )
    header = script.split("import ross")[0]
    assert header.count("\n#   ucs") == 1
    assert header.count("\n#   static") == 1
