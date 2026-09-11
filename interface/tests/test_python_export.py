# -*- coding: utf-8 -*-
"""The Python script generator, moved out of the browser and into the backend.

While it lived in app.js, the frontend had to keep its own copy of the node
numbering, the unit map and the names of the ROSS classes. The copies diverged
in silence -- the chart came out of one path and the exported file out of
another, with no error on screen at all. These tests pin the three ends:

* the files in tests/golden/ hold the output of the OLD generator, case by
  case, for a battery of 44 projects. They were produced by running the
  pre-slice-4 app.js and the new generator side by side: 43 came out identical
  byte for byte, and the only different one is the material-name escaping fix,
  covered below. A future change to the generator has to come through here.
* the generated script has to be valid Python -- the old generator never checked.
* the frontend node numbering and the backend one have to match, case by case."""

import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain.node_resolver import effective_nodes
from domain.python_export import build_script, _js_number, _js_is_nan, _format_kwargs
from domain.schema import unit_map_by_class

GOLDEN = os.path.join(ROOT, "tests", "golden")
JS_LIST = os.path.join(ROOT, "frontend", "components", "list.js")


def _load(name):
    with io.open(os.path.join(GOLDEN, name), encoding="utf-8") as handle:
        return json.load(handle)


CASES = _load("export_cases.json")
EXPECTED = _load("export_scripts.json")


@pytest.mark.parametrize("name", sorted(CASES))
def test_generated_script_matches_the_reference(name):
    case = CASES[name]
    generated = build_script(case["project"], case["analyses"], case["conversion_type"])
    assert generated == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_generated_script_is_valid_python(name):
    """The old generator produced files that would not even open, and nobody checked."""
    case = CASES[name]
    script = build_script(case["project"], case["analyses"], case["conversion_type"])
    ast.parse(script)


UNBALANCE_CALLS = re.compile(
    r"rotor\.(run_unbalance_response|run_clearance_analysis)\((.*?)\)\n"
)
UNBALANCE_TRIO = ("node", "unbalance_magnitude", "unbalance_phase")


def test_the_exported_call_sends_the_three_unbalance_columns_as_lists():
    """`ast.parse` says the file opens; it does not say the file runs.

    The clearance block used to write `node=1` beside
    `unbalance_magnitude=[0.05]`. That parses, so the guard above passed it for
    four slices -- and this is the copy that leaves the program, so whoever ran
    the exported script got the failure on their own machine with no interface
    to blame. ROSS pairs the three with a `zip`; mixed shapes either lose rows
    in silence or, from numpy 2.5, raise.

    The sweep reads the generated scripts as text on purpose. Comparing them to
    the reference files would only prove the generator still agrees with itself;
    what has to hold is a property of the text -- three bracketed columns --
    which stays true no matter how the references are regenerated."""
    problems, calls = [], 0
    for name in sorted(CASES):
        case = CASES[name]
        script = build_script(
            case["project"], case["analyses"], case["conversion_type"]
        )
        for method, arguments in UNBALANCE_CALLS.findall(script):
            calls += 1
            for key in UNBALANCE_TRIO:
                written = re.search(r"\b%s=([^,)]+)" % key, arguments)
                if written is None:
                    problems.append("%s/%s: no %s" % (name, method, key))
                elif not written.group(1).strip().startswith("["):
                    problems.append(
                        "%s/%s: %s=%s is not a list"
                        % (name, method, key, written.group(1).strip())
                    )
    assert calls >= 2, "the sweep found %d unbalance calls: it stopped working" % calls
    assert problems == [], "unbalance arguments out of shape:\n  " + "\n  ".join(
        problems
    )


def test_material_name_with_a_quote_no_longer_breaks_the_script():
    """A material named O'Brien used to close the quote inside rs.Material()."""
    project = {
        "materials": [
            {
                "element_type": "BASIC",
                "name": "O'Brien \\ Steel",
                "rho": "7810",
                "E": "211e9",
                "G_s": "81.2e9",
            }
        ],
        "shafts": [
            {
                "element_type": "BASIC",
                "L": "100",
                "idl": "0",
                "odl": "50",
                "material": "O'Brien \\ Steel",
            }
        ],
    }
    script = build_script(project)
    tree = ast.parse(script)  # the old version raised SyntaxError here

    # and the name survives intact all the way to the literal
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert "O'Brien \\ Steel" in literals
    assert "o'brien \\ steel" in literals


def test_an_empty_project_still_produces_a_runnable_script():
    script = build_script({})
    ast.parse(script)
    assert "rotor = rs.Rotor(" in script
    assert "rotor.plot_rotor().show()" in script


def test_analyses_are_optional_and_never_half_written():
    """A card with no parameters stays out: it became a truncated block in the script."""
    script = build_script({}, [{"type": "campbell"}, {"params": {"speed": "1"}}])
    assert "# Analysis" not in script
    ast.parse(script)


# --- unidades ---------------------------------------------------------------


def test_units_are_inherited_from_the_ross_base_class():
    """cxx of a BallBearingElement is declared on BearingElement.

    The raw map in units.py has BallBearingElement empty; what resolves the
    inheritance is the schema. If the generator went back to reading the raw map,
    the value would come out without Q_ and ROSS would read it as SI -- silently
    wrong."""
    assert unit_map_by_class()["BallBearingElement"]["cxx"] == "N*s/m"
    args = _format_kwargs({"cxx": "1500"}, [], "BallBearingElement")
    assert args == "cxx=Q_(1500, 'N*s/m')"


def test_the_unit_typed_by_the_user_wins_over_the_default():
    assert (
        _format_kwargs({"L": "0.1", "L_unit": "m"}, [], "ShaftElement")
        == "L=Q_(0.1, 'm')"
    )
    assert _format_kwargs({"L": "100"}, [], "ShaftElement") == "L=Q_(100, 'mm')"


def test_a_list_with_a_unit_becomes_a_numpy_array():
    assert (
        _format_kwargs({"kxx": "[1e6, 2e6]"}, [], "BearingElement")
        == "kxx=Q_(np.array([1e6, 2e6]), 'N/m')"
    )


def test_a_tuple_stays_a_tuple():
    """initial_position=(0.1, -0.1) became a string once, and ROSS refused it."""
    assert (
        _format_kwargs({"initial_position": "(0.1, -0.1)"}, [], "PlainJournal")
        == "initial_position=(0.1, -0.1)"
    )


def test_text_and_booleans_do_not_become_numbers():
    assert (
        _format_kwargs({"tag": "bearing A"}, [], "BearingElement") == "tag='bearing A'"
    )
    assert (
        _format_kwargs({"rotary_inertia": "true"}, [], "ShaftElement")
        == "rotary_inertia=True"
    )
    assert (
        _format_kwargs({"rotary_inertia": "False"}, [], "ShaftElement")
        == "rotary_inertia=False"
    )


def test_empty_fields_are_dropped_so_ross_uses_its_own_default():
    assert (
        _format_kwargs({"L": "", "odl": "50"}, [], "ShaftElement") == "odl=Q_(50, 'mm')"
    )


# --- value conversion in the JavaScript style --------------------------------


@pytest.mark.parametrize(
    "text, is_nan",
    [
        ("", False),
        ("  ", False),
        ("42", False),
        (" 42 ", False),
        ("1e5", False),
        ("0x10", False),
        ("Infinity", False),
        ("-3.5", False),
        (".5", False),
        ("5.", False),
        ("abc", True),
        ("1.5.2", True),
        ("1_0", True),
        ("nan", True),
        ("inf", True),
        ("2,3", True),
        ("[1, 2]", True),
    ],
)
def test_number_conversion_follows_javascript_rules(text, is_nan):
    """Python's float() accepts '1_0' and 'nan'; JavaScript's Number() does not.

    If this test loosens, a field with '1_0' starts coming out as a number here
    and as text in what the user sees -- the same class of divergence the whole
    slice exists to close."""
    assert _js_is_nan(text) is is_nan


@pytest.mark.parametrize(
    "value, text",
    [
        (1e-7, "1e-7"),
        (1.5e-8, "1.5e-8"),
        (1e-6, "0.000001"),
        (0.5, "0.5"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
        (123.456, "123.456"),
        (-1e-9, "-1e-9"),
        (1e300, "1e+300"),
    ],
)
def test_float_formatting_matches_javascript(value, text):
    """repr() do Python escreve 1e-07; o JavaScript escreve 1e-7."""
    assert _js_number(value) == text


# --- node numbering: the two languages have to agree -------------------------

NODE_CASES = [
    [],
    [{}, {}, {}],
    [{"n": "0"}, {"n": "1"}, {"n": "2"}],
    [{"n": "2"}, {}, {"n": "0"}, {}],
    [{"n": ""}, {"n": "  "}, {"n": None}],
    [{"n": "3"}, {"n": "3"}, {}],
    [{"n": "2.9"}, {}, {"n": "-1"}],
    [{"n": "abc"}, {"n": "1.5.2"}, {"n": "2,3"}],
    [{"n": "0x10"}, {"n": "1_0"}, {"n": "0b11"}],
    [{"n": "Infinity"}, {"n": "-Infinity"}, {"n": "NaN"}, {"n": "nan"}],
    [{"n": "1e2"}, {"n": " 4 "}, {"n": "+3"}, {"n": "-0"}],
    [{"n": 3}, {"n": 0}, {"n": 2.9}],
    [{"n": ".5"}, {"n": "5."}, {"n": "1e400"}],
]

# The bridge imports the real module. Until Phase 3 slice 3 it cut app.js by a
# text marker and `eval`ed it -- a cut that already broke four guards of this
# refactor when a comment moved somewhere else.
JS_BRIDGE = """
const fs = require('fs');
const { pathToFileURL } = require('url');
(async () => {
    const { getEffectiveNodes } = await import(pathToFileURL(process.argv[2]).href);
    const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
    console.log(JSON.stringify(cases.map(getEffectiveNodes)));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_numbering_matches_the_frontend(tmp_path):
    """The rule on both sides has to be the same, including for typed garbage.

    The backend decides the nodes of the figure and of the script; the frontend
    only labels the list on screen. But it is the same number the user compares.
    Before slice 4 the JavaScript used parseInt (which accepts '0x10' and '3abc')
    and Python used float() (which accepts '1_0'): each turned a different piece
    of garbage into a node."""
    bridge = tmp_path / "bridge.js"
    bridge.write_text(JS_BRIDGE, encoding="utf-8")
    entry = tmp_path / "cases.json"
    entry.write_text(json.dumps(NODE_CASES), encoding="utf-8")

    result = subprocess.run(
        ["node", str(bridge), JS_LIST, str(entry)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    from_the_frontend = json.loads(result.stdout)
    from_the_backend = [effective_nodes(case) for case in NODE_CASES]
    for case, front, backwards in zip(
        NODE_CASES, from_the_frontend, from_the_backend, strict=True
    ):
        assert front == backwards, "diverged on %r: js=%r python=%r" % (
            case,
            front,
            backwards,
        )


# --- the frontend no longer generates Python ---------------------------------


def test_the_frontend_no_longer_builds_the_script():
    """The functions that built the script in the browser must not come back."""
    from frontend_source import raw

    frontend = raw()
    for func in (
        "function pyString",
        "function formatKwargs",
        "function withNodeArg",
        "function _buildRotorClassPython",
    ):
        assert func not in frontend, "%s came back to the frontend" % func
    assert "/api/export/python" in frontend


# --- a rota ------------------------------------------------------------------


@pytest.fixture
def client():
    from app import app as application

    application.config["TESTING"] = True
    with application.test_client() as client:
        yield client


def _auth():
    from api.security import SESSION_TOKEN

    return {"X-ROSS-Token": SESSION_TOKEN}


def test_export_route_returns_a_parseable_script(client):
    body = {
        "project": CASES["minimo"]["project"],
        "analyses": [
            {
                "type": "campbell",
                "params": {
                    "speed_min": "0",
                    "speed_max": "4000",
                    "speed_max_unit": "RPM",
                },
            }
        ],
        "conversion_type": "",
    }
    response = client.post("/api/export/python", json=body, headers=_auth())
    assert response.status_code == 200
    assert response.json["status"] == "success"
    ast.parse(response.json["script"])
    assert "run_campbell" in response.json["script"]


def test_export_route_survives_an_empty_body(client):
    response = client.post("/api/export/python", json={}, headers=_auth())
    assert response.status_code == 200
    ast.parse(response.json["script"])
