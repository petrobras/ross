# -*- coding: utf-8 -*-
"""Step 4 of the Digital Twin: the rossml balancing estimate.

`rossml` is not a dependency of ROSS, so the tests come in two kinds. The
contract tests put a stand-in in `sys.modules` and check the dictionary the
interface hands rossml, against what rossml reads from it. The run at the end
only happens where rossml is installed.
"""

import importlib.util
import json
import os
import shutil
import sys

import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose

import ross as rs
from ross.interface.extensions.digital_twin.domain import rossml_balancing, session
from ross.interface.extensions.digital_twin.services import (
    balancing_analysis,
    measurement,
)
from ross.interface.tests.dt_app import HEADERS, create_app
from ross.interface.tests.test_dt_models import WAVEFORM

MEASURING = (8, 30)
BALANCING = ({"node": 14, "radius": 68.0}, {"node": 25, "radius": 68.0})


def long_rotor():
    """A shaft of 38 elements with two disks and two bearings.

    Nodes 8, 30, 14 and 25 exist, which is what the signal files of rossml's
    own sample database are named after.
    """
    steel = rs.Material(name="steel", rho=7810, E=211e9, G_s=81.2e9)
    shaft = [
        rs.ShaftElement(L=0.025, idl=0, odl=0.03, material=steel) for _ in range(38)
    ]
    disks = [
        rs.DiskElement.from_geometry(
            n=14, material=steel, width=0.03, i_d=0.03, o_d=0.2
        ),
        rs.DiskElement.from_geometry(
            n=25, material=steel, width=0.03, i_d=0.03, o_d=0.2
        ),
    ]
    bearings = [
        rs.BearingElement(n=8, kxx=1e6, cxx=1e3),
        rs.BearingElement(n=30, kxx=1e6, cxx=1e3),
    ]
    return rs.Rotor(shaft, disks, bearings)


@pytest.fixture
def measured(monkeypatch):
    """A session with the rotor, both sensors of two planes, and two planes."""
    current = session.create()
    current.rotor = long_rotor()
    current.equipment = "rig"
    measurement.set_planes(current.id, list(MEASURING), [dict(p) for p in BALANCING])
    for node in MEASURING:
        for plane in ("X", "Y"):
            measurement.apply_sensor(current.id, node, plane, 0.0, "s.csv", WAVEFORM)
    return current


class FakeKalmanFilterApp:
    """Stands in for rossml, keeping the parameters it is given."""

    seen = {}

    def experimental_unbalance_fault_calling(self, rotor_model, parameters_data):
        FakeKalmanFilterApp.seen = parameters_data
        return "filter", parameters_data

    def apply_api_calling(self, kalman_filter, parameters_data):
        planes = parameters_data["balance_plans"]
        phi = {"Sensor #1": [10.0, 200.0], "Sensor #2": [100.0, 290.0]}
        results = type(
            "Results",
            (),
            {
                "dof_forces": [node * 6 for node, _ in planes],
                "number_dof": 6,
                # One chart per view, plane and sensor, as `polar` builds them.
                "polar": lambda self, **kwargs: [
                    go.Figure() for _ in range(2 * len(planes) * len(kwargs["phi"]))
                ],
            },
        )()
        return None, results, 0.5, [34.0, 68.0], phi


@pytest.fixture
def fake_rossml(monkeypatch):
    """Install a stand-in rossml, so the contract can be checked anywhere."""
    api = type(sys)("rossml.api")
    api._default_indirect = lambda: {"P_factor": [[1, 9], [-9, 9]]}
    api._default_reduction_params = lambda: {"tol": 5e-4}
    api._default_signal_parameters = lambda: {"number_points": None}
    api._default_plot_figs = lambda: {"plot_rec": None}
    api._default_kalman_flow = lambda: {"max_iter": 0, "pop_size": 1, "w_f0": None}
    kalman = type(sys)("rossml.kalman_filter")
    kalman.KalmanFilterApp = FakeKalmanFilterApp
    monkeypatch.setitem(sys.modules, "rossml", type(sys)("rossml"))
    monkeypatch.setitem(sys.modules, "rossml.api", api)
    monkeypatch.setitem(sys.modules, "rossml.kalman_filter", kalman)
    monkeypatch.setattr(rossml_balancing, "available", lambda: True)
    FakeKalmanFilterApp.seen = {}
    return FakeKalmanFilterApp


def test_the_parameters_are_what_rossml_reads(measured, fake_rossml):
    # The keys and shapes rossml reads out of this dictionary:
    # `sensors_plans` as {node: angle} and `balance_plans` as
    # [(node, ["x", "y"])] (rossml/api.py), `database_directory` with the
    # trailing slash that is joined to the file names, and `number_points`,
    # which `signal_decompfilter` uses as the length of the resampled signal
    # and which fails on None (rossml/datapipeline/methods.py).
    balancing_analysis.run_balancing(measured.id, iterations=2, population=3)
    seen = fake_rossml.seen
    assert seen["sensors_plans"] == {8: 0.0, 30: 0.0}
    assert seen["balance_plans"] == [(14, ["x", "y"]), (25, ["x", "y"])]
    assert seen["measuring_plans_nodes_list"] == [8, 30]
    assert seen["balancing_plans_nodes_list"] == [{"Nodes": 14}, {"Nodes": 25}]
    assert seen["probes_angle_dict"] == {"8": [0.0, 90.0], "30": [0.0, 90.0]}
    assert seen["database_directory"].endswith(os.sep + "database" + "/")
    assert isinstance(seen["number_points"], int)
    assert seen["kalman_flow"]["max_iter"] == 2 and seen["maxiter"] == 2
    assert seen["kalman_flow"]["pop_size"] == 3 and seen["popsize"] == 3
    assert seen["config_balance_plans"] is None
    assert seen["rotation_sense"] == "clockwise"
    # The speed is the one the signal files were measured at, not one typed in.
    assert seen["speed"] == [840.0, "rpm"] == seen["parquet_speed"]


def test_the_correction_is_the_estimate_over_the_radius_half_a_turn_away(
    measured, fake_rossml
):
    answer = balancing_analysis.run_balancing(measured.id)
    planes = answer["balancing"]["planes"]
    assert [plane["node"] for plane in planes] == [14, 25]
    # 34 g.mm over a radius of 68 mm is 0.5 g, and the correction goes on the
    # other side of the shaft: 10 degrees becomes 190.
    assert_allclose([plane["mass"] for plane in planes], [0.5, 1.0])
    assert_allclose([plane["phase"] for plane in planes], [190.0, 20.0])
    assert_allclose([plane["unbalance"] for plane in planes], [34.0, 68.0])
    assert answer["balancing"]["loss"] == 0.5
    assert answer["balancing"]["measuring"] == [8, 30]
    assert answer["ready"] and answer["refusal"] == ""
    # The angle is read against one sensor, so every sensor's angles come back
    # and the screen names each one by the node and plane it sits on.
    assert answer["balancing"]["phases"]["Sensor #2"] == [100.0, 290.0]
    assert answer["balancing"]["sensors"] == [
        {"node": 8, "plane": "X", "key": "Sensor #1"},
        {"node": 8, "plane": "Y", "key": "Sensor #2"},
    ]
    assert answer["balancing"]["views"] == ["A", "B"]


def test_a_polar_chart_per_view_plane_and_sensor(measured, fake_rossml):
    balancing_analysis.run_balancing(measured.id)
    for view in ("A", "B"):
        for sensor in ("Sensor #1", "Sensor #2"):
            answer = balancing_analysis.polar_for(measured.id, view, sensor)
            assert [chart["node"] for chart in answer["charts"]] == [14, 25]
            for chart in answer["charts"]:
                assert "data" in chart["figure"] and "layout" in chart["figure"]
    with pytest.raises(ValueError, match="charts"):
        balancing_analysis.polar_for(measured.id, "C", "Sensor #1")
    with pytest.raises(ValueError, match="Run the analysis"):
        balancing_analysis.polar_for(session.create().id, "A", "Sensor #1")


# A file of the shape the Digital Twin app keeps per machine. The values are
# not the Petrobras ones: what is checked is that the file replaces the
# library defaults of rossml and that the run keeps what it owns.
PARAMETERS = {
    "speed": [20, "Hz"],
    "indirect": {"P_factor": [[2, 8], [-5, 5]], "R_factor": None},
    "resultant_force_limits": {"mu_d": [100, 1000], "phi": [0, 360]},
    "signal_parameters": {"number_points": 2048, "decomposition_threshold": 4},
    "kalman_flow": {"df_multiple": 11, "mu_pop_size": 2, "max_iter": 9, "pop_size": 9},
}


def test_the_machine_parameters_replace_the_rossml_defaults(measured, fake_rossml):
    answer = balancing_analysis.settings_for(measured.id)
    assert answer["parameters"] is None
    assert "will not match" in answer["parameters_warning"]

    answer = balancing_analysis.set_parameters(
        measured.id, "input_parameters.json", json.dumps(PARAMETERS)
    )
    assert answer["parameters"]["name"] == "input_parameters.json"
    assert answer["parameters"]["source"] == "file"
    assert answer["parameters_warning"] == ""

    balancing_analysis.run_balancing(measured.id, iterations=2, population=3)
    seen = fake_rossml.seen
    # The file wins where it has a key.
    assert seen["indirect"] == PARAMETERS["indirect"]
    assert seen["resultant_force_limits"] == PARAMETERS["resultant_force_limits"]
    assert seen["number_points"] == 2048
    assert seen["kalman_flow"]["df_multiple"] == 11
    # What the run owns is kept: the iterations and population chosen on the
    # screen, the speed read from the signal files, and the folders.
    assert seen["kalman_flow"]["max_iter"] == 2 and seen["kalman_flow"]["pop_size"] == 3
    assert seen["maxiter"] == 2 and seen["popsize"] == 3
    assert seen["speed"] == [840.0, "rpm"] == seen["parquet_speed"]
    assert seen["database_directory"].endswith("/")

    answer = balancing_analysis.set_parameters(measured.id, None, None)
    assert answer["parameters"] is None


def test_a_parameters_file_must_be_a_json_object(measured, fake_rossml):
    for filename, content in (
        ("p.txt", "{}"),
        ("p.json", "not json"),
        ("p.json", "[1, 2]"),
        ("p.json", "{}"),
    ):
        with pytest.raises(ValueError):
            balancing_analysis.set_parameters(measured.id, filename, content)


def test_the_parameters_routes(measured, fake_rossml):
    client = create_app().test_client()
    answer = client.post(
        "/api/dt/balancing-parameters",
        json={
            "session": measured.id,
            "filename": "input_parameters.json",
            "content": json.dumps(PARAMETERS),
        },
        headers=HEADERS,
    )
    assert answer.status_code == 200
    assert answer.get_json()["parameters"]["keys"] == sorted(PARAMETERS)
    refused = client.post(
        "/api/dt/balancing-parameters",
        json={"session": measured.id, "filename": "p.json", "content": "nope"},
        headers=HEADERS,
    )
    assert refused.status_code == 400


def test_the_optimization_offers_every_leading_and_trailing_run(measured, fake_rossml):
    balancing_analysis.run_balancing(measured.id, optimize=True)
    assert fake_rossml.seen["config_balance_plans"] == [
        [(14, ["x", "y"])],
        [(14, ["x", "y"]), (25, ["x", "y"])],
        [(25, ["x", "y"])],
    ]
    assert rossml_balancing.plane_groups([1, 2, 3]) == [
        [(1, ["x", "y"])],
        [(1, ["x", "y"]), (2, ["x", "y"])],
        [(1, ["x", "y"]), (2, ["x", "y"]), (3, ["x", "y"])],
        [(2, ["x", "y"]), (3, ["x", "y"])],
        [(3, ["x", "y"])],
    ]


def test_the_planes_the_estimate_used_may_be_fewer_than_the_planes_offered(
    measured, fake_rossml, monkeypatch
):
    # With the optimization on, rossml may report forces on a subset of the
    # planes. The answer follows what it used, not what was offered.
    def apply(self, kalman_filter, parameters_data):
        results = type(
            "Results",
            (),
            {
                "dof_forces": [25 * 6],
                "number_dof": 6,
                "polar": lambda self, **kwargs: [go.Figure(), go.Figure()],
            },
        )()
        return None, results, None, [68.0], {"Sensor #1": [200.0]}

    monkeypatch.setattr(FakeKalmanFilterApp, "apply_api_calling", apply)
    planes = balancing_analysis.run_balancing(measured.id, optimize=True)["balancing"][
        "planes"
    ]
    assert [plane["node"] for plane in planes] == [25]
    assert_allclose(planes[0]["mass"], 1.0)


def test_the_analysis_says_what_is_missing(measured, fake_rossml):
    current = session.create()
    current.rotor = long_rotor()
    assert (
        "two measuring planes" in balancing_analysis.settings_for(current.id)["refusal"]
    )
    measurement.set_planes(current.id, list(MEASURING), [])
    for node in MEASURING:
        for plane in ("X", "Y"):
            measurement.apply_sensor(current.id, node, plane, 0.0, "s.csv", WAVEFORM)
    assert "balancing plane" in balancing_analysis.settings_for(current.id)["refusal"]
    with pytest.raises(ValueError, match="balancing plane"):
        balancing_analysis.run_balancing(current.id)

    empty = session.create()
    assert "Create the model" in balancing_analysis.settings_for(empty.id)["refusal"]


def test_without_rossml_the_analysis_is_refused(measured, monkeypatch):
    monkeypatch.setattr(rossml_balancing, "available", lambda: False)
    answer = balancing_analysis.settings_for(measured.id)
    assert not answer["available"] and not answer["ready"]
    assert "rossml" in answer["refusal"]
    with pytest.raises(ValueError, match="rossml"):
        balancing_analysis.run_balancing(measured.id)


def test_the_settings_are_whole_numbers_in_range(measured, fake_rossml):
    for iterations in (-1, 21, "two", 1.5, True):
        with pytest.raises(ValueError):
            balancing_analysis.run_balancing(measured.id, iterations=iterations)
    for population in (0, 51, None):
        with pytest.raises(ValueError):
            balancing_analysis.run_balancing(measured.id, population=population)


def test_the_balancing_routes(measured, fake_rossml):
    client = create_app().test_client()
    settings = client.get(
        "/api/dt/balancing-settings?session=" + measured.id, headers=HEADERS
    )
    assert settings.status_code == 200
    assert settings.get_json()["balancing"] is None
    answer = client.post(
        "/api/dt/balancing",
        json={"session": measured.id, "iterations": 0, "population": 1},
        headers=HEADERS,
    )
    assert answer.status_code == 200
    assert len(answer.get_json()["balancing"]["planes"]) == 2
    charts = client.get(
        "/api/dt/balancing-polar?session=%s&view=A&sensor=%s"
        % (measured.id, "Sensor+%231"),
        headers=HEADERS,
    )
    assert charts.status_code == 200
    assert len(charts.get_json()["charts"]) == 2
    refused = client.post(
        "/api/dt/balancing",
        json={"session": measured.id, "iterations": 99},
        headers=HEADERS,
    )
    assert refused.status_code == 400
    assert client.post("/api/dt/balancing", json={}).status_code == 403


def sample_signals():
    """The folder of waveform files rossml keeps as its sample, or `None`.

    The package is located without importing it: importing rossml needs the
    stand-in for Streamlit in place, which collecting a test does not do.
    """
    spec = importlib.util.find_spec("rossml") if rossml_balancing.available() else None
    if spec is None or not spec.origin:
        return None
    folder = os.path.join(os.path.dirname(os.path.dirname(spec.origin)), "database")
    files = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    return folder if len(files) >= 4 else None


# statsmodels 0.15 warns that `adfuller` will return a result object instead of
# a tuple in 0.16 or July 2027. rossml calls it, so the fix belongs there and
# the interface's rule that a FutureWarning fails the run is held off for this
# one test (noted 2026-11-18).
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(
    sample_signals() is None,
    reason="rossml, with its sample signals, is not installed here",
)
def test_a_real_run_answers_one_correction_per_plane(measured):
    # rossml's own sample signals, renamed to this rotor's measuring nodes.
    # They were measured on another machine, so the numbers are not a physical
    # reference: what is checked is that a run answers one finite correction
    # mass per balancing plane, at the radius given, half a turn from the
    # estimated unbalance.
    folder = sample_signals()
    files = sorted(name for name in os.listdir(folder) if name.endswith(".csv"))
    for node, (x_file, y_file) in zip(MEASURING, (files[:2], files[2:4]), strict=True):
        for plane, name in (("X", x_file), ("Y", y_file)):
            shutil.copy(
                os.path.join(folder, name),
                measurement.waveform_path(measured, node, plane),
            )

    answer = balancing_analysis.run_balancing(measured.id)
    planes = answer["balancing"]["planes"]
    assert [plane["node"] for plane in planes] == [14, 25]
    # Four sensors on two measuring planes, so eight charts per view.
    assert len(answer["balancing"]["sensors"]) == 4
    for view in ("A", "B"):
        for sensor in answer["balancing"]["sensors"]:
            charts = balancing_analysis.polar_for(measured.id, view, sensor["key"])
            assert [chart["node"] for chart in charts["charts"]] == [14, 25]
            for chart in charts["charts"]:
                assert chart["figure"]["data"]
    for plane in planes:
        assert plane["mass"] > 0
        assert_allclose(plane["mass"], plane["unbalance"] / plane["radius"])
        assert_allclose(plane["phase"], (plane["unbalance_phase"] + 180.0) % 360.0)
        assert 0 <= plane["phase"] < 360
