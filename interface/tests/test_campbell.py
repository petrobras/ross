# -*- coding: utf-8 -*-
"""The Campbell mode shape without Dash (BE-04).

## What there was

The mode shape came out through ROSS's `plot_with_mode_shape`, which starts a
**Dash** server on a random port. To find the port, the backend replaced
`sys.stdout` with an object that spied on what Dash printed, started a daemon
thread and polled for up to 10 seconds; the URL became the `src` of an iframe
on the screen.

Four problems: in a PyInstaller executable with no console there is no
`sys.stdout` to spy on -- and packaging is the Phase 4 decision; without `dash`
installed ROSS raises an ImportError inside the thread, the error becomes a
`print`, and the user waits the 10 seconds to get "took too long to answer",
never learning the cause; the replaced `sys.stdout` was never restored; and
Dash went into the bundle.

## What there is

`plot_with_mode_shape` is a thin shell over `_plot_with_mode_shape`, which does
two things: `self.plot(...)` -- the same diagram the Default mode already draws
-- and a callback calling `self._update_plot_mode_3d(...)` with the clicked
point. And that method reads `self.modal_results`, which `run_campbell`
**already filled in** with one modal result per speed of the range.

That is: the click recomputes nothing, it picks among results already in
memory. The interface draws the Campbell normally and asks for the 3D figure in
a request of its own.

The price is depending on two private ROSS methods. The signature tests in
`test_ross_premises.py` exist for that: if a future version changes either of
them, the suite fails the same day."""

import ast
import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import structure
from services.analysis import get_runner
from services.analysis.campbell import DEFAULT_UNITS

from frontend_source import raw as _frontend
from waiting import answer_for

RUNNER = get_runner("campbell")

try:
    import ross  # noqa: F401

    HAS_ROSS = True
except Exception:  # pragma: no cover
    HAS_ROSS = False

needs_ross = pytest.mark.skipif(not HAS_ROSS, reason="requires ROSS installed")


# --- the two private ROSS methods we depend on -------------------------------

# --- dispatching the click, with a stand-in ----------------------------------


class FakeResult:
    """Records the `_update_plot_mode_3d` call instead of computing."""

    def __init__(self, with_torsional=False):
        self.call = None
        if with_torsional:
            self.campbell_torsional = FakeResult()

    def _update_plot_mode_3d(self, *args):
        self.call = args
        return "figure"


PARAMS = {
    "speed_units": "RPM",
    "frequency_units": "Hz",
    "damping_parameter": "damping_ratio",
    "animation": "True",
}
POINT = {"x": 1500.0, "y": 42.0, "curve_name": "Mode 1"}


def test_the_click_reaches_ross_with_the_point_and_the_units():
    result = FakeResult()
    assert RUNNER.mode_shape_figure(result, PARAMS, POINT) == "figure"

    speed, frequency_value, fallback, u_vel, u_freq, damping, animation = result.call
    assert (speed, frequency_value) == (1500.0, 42.0)
    assert (u_vel, u_freq, damping) == ("RPM", "Hz", "damping_ratio")
    assert animation is True


def test_the_fallback_dictionary_goes_in_empty_on_purpose():
    """See test_run_campbell_still_fills_modal_results_for_every_speed."""
    result = FakeResult()
    RUNNER.mode_shape_figure(result, PARAMS, POINT)
    assert result.call[2] == {}


@pytest.mark.parametrize("key", sorted(DEFAULT_UNITS))
def test_a_blank_unit_falls_back_to_the_ross_default(key):
    """The card only sends the unit if the user chose one."""
    result = FakeResult()
    RUNNER.mode_shape_figure(result, dict(PARAMS, **{key: ""}), POINT)

    position = {"speed_units": 3, "frequency_units": 4, "damping_parameter": 5}[key]
    assert result.call[position] == DEFAULT_UNITS[key]


def test_a_torsional_point_is_answered_by_the_torsional_results():
    """This is how ROSS itself separates the two curves, by the trace name."""
    result = FakeResult(with_torsional=True)
    RUNNER.mode_shape_figure(
        result, PARAMS, dict(POINT, curve_name="Torsional Analysis")
    )

    assert result.call is None
    assert result.campbell_torsional.call is not None


def test_a_torsional_point_without_torsional_results_says_what_to_do():
    """Without the torsional analysis switched on, the attribute does not even exist."""
    with pytest.raises(ValueError) as error:
        RUNNER.mode_shape_figure(
            FakeResult(), PARAMS, dict(POINT, curve_name="Torsional Analysis")
        )
    assert "torsional_analysis" in str(error.value)


@pytest.mark.parametrize("missing", ["x", "y"])
def test_a_point_without_coordinates_is_refused_by_name(missing):
    point = dict(POINT)
    point[missing] = None
    with pytest.raises(ValueError) as error:
        RUNNER.mode_shape_figure(FakeResult(), PARAMS, point)
    assert missing in str(error.value)


# --- a rota -----------------------------------------------------------------


@pytest.fixture
def client():
    from app import app as application

    application.config["TESTING"] = True
    with application.test_client() as client:
        yield client


def test_the_mode_shape_route_requires_the_session_token(client):
    response = client.post("/api/campbell/mode_shape", json={"project": {}})
    assert response.status_code == 403


@needs_ross
def test_the_mode_shape_route_answers_a_click(client):
    from api.security import SESSION_TOKEN

    project = {
        "materials": [{"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}],
        "shafts": [
            {"L": "100", "odl": "50", "idl": "0", "material": "Steel"} for _ in range(3)
        ],
        "bearings": [
            {"element_type": "BASIC", "n": "0", "kxx": "1e6", "cxx": "1e3"},
            {"element_type": "BASIC", "n": "3", "kxx": "1e6", "cxx": "1e3"},
        ],
    }
    body = {
        "project": project,
        "conversion_type": "",
        "params": {
            "speed_min": "0",
            "speed_max": "300",
            "speed_steps": "4",
            "frequencies": "4",
            "plot_type": "Mode Shape",
        },
        "point": {"x": 100.0, "y": 50.0, "curve_name": "Mode 1"},
    }
    headers = {"X-ROSS-Token": SESSION_TOKEN}
    response = answer_for(
        client,
        client.post("/api/campbell/mode_shape", json=body, headers=headers),
        headers,
    )
    assert response.status_code == 200, response.json
    assert response.json["status"] == "success"
    assert response.json["plot_json"]


# --- o Dash saiu de cena ----------------------------------------------------

# `domain/python_export.py` is left out: it does not *call* ROSS, it writes a
# Python script for the user to run outside. In that context opening a Dash app
# is a reasonable choice -- who cannot depend on it is the interface.
OUTSIDE_THE_SWEEP = (
    "_backup",
    os.sep + "tests" + os.sep,
    os.path.join("domain", "python_export.py"),
)


def _backend_trees():
    for path, tree in structure.modules(ROOT, "**", "*.py", recursive=True):
        if any(chunk in path for chunk in OUTSIDE_THE_SWEEP):
            continue
        yield path, tree


# The one file allowed to move `sys.stdout`, and why. Read by the guard below
# and by its control, which fails if the reason stops being true.
HANDS_STDOUT_OVER = {
    os.path.join("services", "worker", "child.py"): (
        "the worker child speaks the protocol on stdout, and ROSS prints to it "
        "while it computes -- `run_crack` and `run_rubbing` write 'Running "
        "direct method' there. The child takes the real descriptor aside and "
        "points `sys.stdout` at stderr, which is the opposite of the hijack "
        "this guard was written for: that one made output disappear from the "
        "server, this one keeps a channel clean in a different process."
    )
}


def test_nothing_hijacks_stdout_anymore():
    """The sys.stdout hijack must not come back by oversight."""
    for path, tree in _backend_trees():
        if any(path.endswith(allowed) for allowed in HANDS_STDOUT_OVER):
            continue
        assert not structure.assigns_to(tree, "sys.stdout"), (
            "%s swaps sys.stdout" % path
        )


def test_the_only_file_allowed_to_move_stdout_still_moves_it():
    """Control: an exception that stops excepting anything has to be removed.

    Same rule the Portuguese sweep's `DELIBERATE` follows. If the worker ever
    stopped redirecting, this entry would go on silently exempting a file that
    no longer needs it -- and the next thing to hijack stdout in that file would
    pass unnoticed.

    What the redirection has to *be* is checked in `tests/test_worker.py`, which
    owns that rule; here we only keep the exemption honest.
    """
    for relative, reason in HANDS_STDOUT_OVER.items():
        assert reason.strip(), "%s is exempt with no reason written" % relative
        full = os.path.join(ROOT, relative)
        assert os.path.exists(full), "%s is exempt and does not exist" % relative
        with io.open(full, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        assert structure.assigns_to(tree, "sys.stdout"), (
            "%s no longer moves sys.stdout: take it off HANDS_STDOUT_OVER so "
            "the guard covers it again" % relative
        )


def test_the_backend_no_longer_starts_a_dash_server():
    """By the tree: the runner docstring mentions the method in order to explain it."""
    for path, tree in _backend_trees():
        assert not structure.calls_method(tree, "plot_with_mode_shape"), (
            "%s still calls the Dash path" % path
        )


def test_the_frontend_no_longer_embeds_a_dash_iframe():
    app_js = _frontend()
    assert 'status === "dash"' not in app_js
    assert "<iframe" not in app_js
    assert "/api/campbell/mode_shape" in app_js
    assert "plotly_click" in app_js


def test_the_three_drawing_paths_all_ask_the_same_question():
    """Running, restoring from memory and loading from a file draw the same card.

    If one of them forgot the panels, a restored Campbell in mode-shape mode would
    show up with no 3D panel and not react to the click -- exactly the kind of
    silent divergence this refactor is chasing."""
    app_js = _frontend()

    assert app_js.count("function isModeShape(") == 1
    assert app_js.count("isModeShape(") == 4  # the definition plus three uses
    assert app_js.count("prepareModeShapePanels(divNode") == 2
    assert app_js.count("prepareModeShapePanels(div,") == 1
    assert app_js.count("wireModeShapeClick(") == 4  # the definition plus three uses


def test_the_conversion_travels_with_a_saved_analysis():
    """Without this, a card computed in 4 DoF came back as 6 DoF when clicked.

    In Phase 3 the conversion stopped living in `div.rossConversion` and became a
    field of the analysis record. What this test guards is still the same: that it
    is recorded at creation, at both restorations, and comes out in what is saved."""
    app_js = _frontend()

    # creation and the two restorations (back from the Hub, load from a file)
    assert app_js.count("registerAnalysis(") == 4  # the definition plus three uses
    assert "conversion: record.conversion" in app_js
    assert "conversion: conversion || ''" in app_js


# --- the UCS field ROSS could not receive, and now can ------------------------
#
# Kept in the past tense rather than deleted: this is the diagnosis that became
# the report, the report became a one-line fix upstream, and the fix brought the
# field back to the form. What it records is a way of reading `@check_units`
# that will be needed again -- the same pattern is still live at
# `multi_rotor/gear_element.py:113` and `:116`.
#
# `run_ucs` is decorated with `@check_units`. The decorator splits the argument
# name into pieces (`bearing_frequency_range` -> bearing / frequency / range),
# finds "frequency" in the unit dictionary (`units.py:41`, radian/second) and
# converts the value with `Q_(v, "radian/second").m`. pint gives back the
# magnitude of a list as a **numpy array**.
#
# Two lines later, `rotor_assembly.py:4197`:
#
#     if bearing_frequency_range:
#
# -- a truth test on a two-element array, which numpy refuses. No value can
# escape it: any sequence becomes an array, and a scalar breaks on the next
# line, which indexes [0] and [1].
#
# The fix was in ROSS, and was one line: `if bearing_frequency_range is not
# None:`. **Commit 2a253e6 applied exactly that**, so the field is back on the
# form as the pair `bearing_freq_min`/`bearing_freq_max`. What holds the premise
# up now is `test_ross_accepts_a_bearing_frequency_range`, which asserts the fix
# instead of the defect.
