# -*- coding: utf-8 -*-
"""Step 4 of the Digital Twin: the rossml balancing estimate.

`rossml` is not a dependency of ROSS, so the tests come in two kinds. The
contract tests put a stand-in for `rossml.api.API` in `sys.modules` and check
the arguments the interface hands it. The tests at the end only happen where
rossml is installed: one checks those arguments against the real `API.run`,
the other runs it on rossml's sample signals and draws every chart.
"""

import importlib.util
import inspect
import os
import shutil
import sys

import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose

import ross as rs
from ross.interface.extensions.digital_twin.domain import (
    analysis_parameters,
    rossml_balancing,
    session,
)
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


class FakeAPI:
    """Stands in for `rossml.api.API`, keeping the arguments `run` is given.

    The signature of `run` lists the arguments of the real one that the
    interface uses; `test_the_arguments_are_those_of_rossml` checks them
    against the real one where rossml is installed.
    """

    seen = {}
    drawn = []

    def run(
        self,
        rotor_model=None,
        sensors_plans=None,
        probes_angle=None,
        balance_plans=None,
        speed=None,
        number_points=4096,
        w_f0=0.65,
        rotation_sense="clockwise",
        indirect=None,
        view_plans="all",
        signal_parameters=None,
        kalman_flow=None,
        verbose=True,
        kalman_max_iter=0,
        kalman_pop_size=1,
        database_directory=None,
        figs_directory=None,
        config_balance_plans=None,
        direct=None,
        **kwargs,
    ):
        arguments = dict(locals())
        del arguments["self"], arguments["kwargs"]
        FakeAPI.seen = {**arguments, **kwargs}
        self.phi = {"Sensor #1": [10.0, 200.0], "Sensor #2": [100.0, 290.0]}
        self.mu_d = [34.0, 68.0]
        self.loss_function = [0.9, 0.5]
        self.results_balance = type(
            "Results",
            (),
            {"dof_forces": [node * 6 for node in balance_plans], "number_dof": 6},
        )()

    def _figure(self, name, **arguments):
        FakeAPI.drawn.append((name, arguments))
        return go.Figure()

    def plot_balance_polar(self, sensor_ref=None):
        # One chart per balancing plane, as `polar_API` draws them.
        FakeAPI.drawn.append(("polar", {"sensor_ref": sensor_ref}))
        return [go.Figure() for _ in self.mu_d]

    def plot_balance_polar_decomposed(self, node=None, **arguments):
        return [self._figure("decomposed", node=node, **arguments)]

    def plot_balance_1d(self, node=None):
        return self._figure("forces_1d", node=node)

    def plot_balance_2d(self, node=None):
        return self._figure("forces_2d", node=node)

    def plot_vs_1d(self, node=None):
        return self._figure("vs_1d", node=node)

    def plot_vs_2d(self, node=None):
        return self._figure("vs_2d", node=node)

    def plot_vs_3d(self):
        return self._figure("vs_3d")


@pytest.fixture
def fake_rossml(monkeypatch):
    """Install a stand-in rossml, so the contract can be checked anywhere."""
    api = type(sys)("rossml.api")
    api.API = FakeAPI
    monkeypatch.setitem(sys.modules, "rossml", type(sys)("rossml"))
    monkeypatch.setitem(sys.modules, "rossml.api", api)
    monkeypatch.setattr(rossml_balancing, "available", lambda: True)
    FakeAPI.seen = {}
    FakeAPI.drawn = []
    return FakeAPI


def test_the_arguments_are_what_the_api_reads(measured, fake_rossml):
    # `sensors_plans` and `probes_angle` are matched one to one, in the order
    # of the measuring nodes, and `database_directory` carries the trailing
    # slash rossml joins the file names to (rossml/datapipeline/methods.py).
    balancing_analysis.run_balancing(measured.id, iterations=2, population=3)
    seen = fake_rossml.seen
    assert seen["sensors_plans"] == [8, 30]
    assert seen["probes_angle"] == [0.0, 0.0]
    assert seen["balance_plans"] == [14, 25]
    assert seen["database_directory"].endswith(os.sep + "database" + "/")
    assert seen["figs_directory"].endswith("/")
    assert seen["kalman_max_iter"] == 2 and seen["kalman_pop_size"] == 3
    assert seen["config_balance_plans"] is None
    assert seen["rotation_sense"] == "clockwise"
    # A server must not clear the terminal it runs in, which `verbose` does.
    assert seen["verbose"] is False
    # The speed is the one the signal files were measured at, not one typed in.
    assert seen["speed"] == [840.0, "rpm"]


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
    # rossml answers the history of the objective function; the last is kept.
    assert answer["balancing"]["loss"] == 0.5
    assert answer["balancing"]["measuring"] == [8, 30]
    assert answer["ready"] and answer["refusal"] == ""
    # The angle is read against one sensor, so every sensor's angles come back
    # and the screen names each one by the node and plane it sits on.
    assert answer["balancing"]["phases"]["Sensor #2"] == [100.0, 290.0]
    assert answer["balancing"]["sensors"] == [
        {"node": 8, "plane": "X", "key": "Sensor #1", "index": 0},
        {"node": 8, "plane": "Y", "key": "Sensor #2", "index": 1},
    ]
    # The virtual sensing is offered on the balancing planes the estimate
    # used, then on the measuring planes.
    assert answer["balancing"]["vs_nodes"] == [14, 25, 8, 30]
    assert answer["balancing"]["charts"] == list(rossml_balancing.CHARTS)


def test_every_chart_of_a_run(measured, fake_rossml):
    balancing_analysis.run_balancing(measured.id)

    answer = balancing_analysis.figure_for(measured.id, "polar", sensor=1)
    assert [chart["node"] for chart in answer["charts"]] == [14, 25]
    assert fake_rossml.drawn[-1] == ("polar", {"sensor_ref": 1})
    for chart in answer["charts"]:
        assert "data" in chart["figure"] and "layout" in chart["figure"]

    for kind, node in (
        ("forces_1d", 14),
        ("forces_2d", 25),
        ("vs_1d", 8),
        ("vs_2d", 30),
    ):
        answer = balancing_analysis.figure_for(measured.id, kind, node=node)
        assert [chart["node"] for chart in answer["charts"]] == [node]
        assert fake_rossml.drawn[-1] == (kind, {"node": node})

    answer = balancing_analysis.figure_for(measured.id, "vs_3d")
    assert [chart["node"] for chart in answer["charts"]] == [None]

    answer = balancing_analysis.figure_for(
        measured.id,
        "decomposed",
        node=25,
        sensor=1,
        positions=[30, 120],
        radii=[68, 50],
    )
    assert fake_rossml.drawn[-1] == (
        "decomposed",
        {
            "node": 25,
            "positions_list": [30.0, 120.0],
            "radius_list": [68.0, 50.0],
            "sensor_ref": 1,
        },
    )


def test_a_chart_is_refused_with_the_reason(measured, fake_rossml):
    with pytest.raises(ValueError, match="Run the analysis"):
        balancing_analysis.figure_for(measured.id, "polar")
    balancing_analysis.run_balancing(measured.id)
    refusals = (
        ({"chart": "bode"}, "Unknown chart"),
        # The forces are estimated on the balancing planes only.
        ({"chart": "forces_1d", "node": 8}, "balancing plane among 14, 25"),
        ({"chart": "vs_2d", "node": 3}, "plane among 14, 25, 8, 30"),
        ({"chart": "vs_1d", "node": 1.5}, "plane among"),
        ({"chart": "polar", "sensor": 2}, "reference sensor among 0, 1"),
        ({"chart": "decomposed", "node": 14}, "at least two positions"),
        (
            {"chart": "decomposed", "node": 14, "positions": [0], "radii": [68]},
            "at least two positions",
        ),
        (
            {"chart": "decomposed", "node": 14, "positions": [0, 90], "radii": [68, 0]},
            "positive",
        ),
    )
    for arguments, message in refusals:
        with pytest.raises(ValueError, match=message):
            balancing_analysis.figure_for(measured.id, **arguments)


# One value per field, each different from the rossml default, so a test that
# sees it in the arguments knows the field replaced the default. The values are
# not the Petrobras ones.
PARAMETERS = {
    "number_points": 2048,
    "w_f0": 0.35,
    "direct_P": 0.1,
    "direct_Q": 1e-18,
    "direct_S": 3e8,
    "direct_R": 7e-15,
    "indirect_P_mantissa_min": 2,
    "indirect_P_mantissa_max": 8,
    "indirect_P_exponent_min": -5,
    "indirect_P_exponent_max": 5,
    "indirect_Q_mantissa_min": 1,
    "indirect_Q_mantissa_max": 7,
    "indirect_Q_exponent_min": -18,
    "indirect_Q_exponent_max": -9,
    "indirect_S_mantissa_min": 3,
    "indirect_S_mantissa_max": 8,
    "indirect_S_exponent_min": 4,
    "indirect_S_exponent_max": 7,
}
# The same values as rossml's `API.run` takes them.
DIRECT = {"P_factor": 0.1, "Q_factor": 1e-18, "S_factor": 3e8, "R_factor": 7e-15}
INDIRECT = {
    "P_factor": [[2, 8], [-5, 5]],
    "Q_factor": [[1, 7], [-18, -9]],
    "S_factor": [[3, 8], [4, 7]],
    "R_factor": None,
}
# The dictionaries `API.run` takes and then drops in `_clearmemory` without
# reading them, so the interface has no field of theirs to send.
UNREAD = ("signal_parameters", "kalman_flow")


def test_the_analysis_parameters_read_no_machine_file():
    """The parameters of a machine are not read by the library.

    They belong to the machine, so `models_files/<model>/input_parameters.json`
    stays in the Digital Twin app and no screen of the interface shows it. The
    fields start from the rossml defaults for every rotor.
    """
    source = inspect.getsource(analysis_parameters)
    for name in ("models_files", "input_parameters", "APP_DIR", "preset"):
        assert name not in source, name


def test_the_fields_start_empty_at_the_rossml_defaults(measured, fake_rossml):
    # Every field of the form has a value in PARAMETERS, so the tests below
    # cover each one.
    assert set(PARAMETERS) == set(analysis_parameters.KEYS)
    assert set(analysis_parameters.defaults().values()) == {None}
    answer = balancing_analysis.settings_for(measured.id)
    fields = answer["parameters"]["fields"]
    assert {field["key"] for field in fields} == set(analysis_parameters.KEYS)
    # The screen shows the default beside each field, so the user can see what
    # an empty one falls back to.
    assert {field["key"]: field["default"] for field in fields} == (
        analysis_parameters.DEFAULTS
    )
    assert set(answer["parameters"]["values"].values()) == {None}
    balancing_analysis.run_balancing(measured.id)
    seen = fake_rossml.seen
    for key in ("indirect", "direct") + UNREAD:
        assert seen[key] is None, key
    # Nothing is passed for an empty field, so rossml keeps its own defaults.
    assert analysis_parameters.arguments(answer["parameters"]["values"]) == {}
    assert seen["number_points"] == 4096
    assert seen["w_f0"] == 0.65


@pytest.mark.skipif(not rossml_balancing.available(), reason="rossml is not installed")
def test_the_defaults_shown_are_the_ones_rossml_uses():
    # The defaults the screen shows are written down in the interface, so that
    # they are there without rossml installed. This is the check that they are
    # the ones rossml falls back to, and that no field of the form goes to an
    # argument rossml takes and never reads.
    from rossml.api import API, _default_indirect

    taken = inspect.signature(API.run).parameters
    indirect = _default_indirect()
    for key, _group, path, _whole in analysis_parameters.FIELDS:
        shown = analysis_parameters.DEFAULTS[key]
        assert path[0] not in UNREAD, key
        if len(path) == 1:
            assert shown == taken[path[0]].default, key
        elif path[0] == "indirect":
            assert shown == indirect[path[1]][path[2]][path[3]], key
        else:
            assert shown is None and taken[path[0]].default is None, key


def test_the_chosen_fields_replace_the_rossml_defaults(measured, fake_rossml):
    balancing_analysis.run_balancing(
        measured.id,
        iterations=2,
        population=3,
        values={**PARAMETERS, "w_f0": "0.4"},
    )
    seen = fake_rossml.seen
    assert seen["number_points"] == 2048
    assert seen["w_f0"] == 0.4
    assert seen["direct"] == DIRECT
    assert seen["indirect"] == INDIRECT
    # The dictionaries rossml drops unread are not sent, and neither are the
    # keys it no longer reads.
    for key in UNREAD + ("N_multiple", "plot_figs"):
        assert seen.get(key) is None, key
    # What the run owns is kept: the iterations and population chosen on the
    # screen, the speed read from the signal files, and the folders.
    assert seen["kalman_max_iter"] == 2 and seen["kalman_pop_size"] == 3
    assert seen["speed"] == [840.0, "rpm"]
    assert seen["database_directory"].endswith(os.sep + "database" + "/")
    # The session keeps the choice for the screen.
    answer = balancing_analysis.settings_for(measured.id)
    assert answer["parameters"]["values"]["w_f0"] == 0.4
    assert answer["parameters"]["values"]["number_points"] == 2048


def test_the_search_bounds_go_on_without_the_direct_factors(measured, fake_rossml):
    # Without a direct value the optimizer searches inside the indirect bounds,
    # which is what rossml does with `direct=None`.
    values = {key: value for key, value in PARAMETERS.items() if "direct_" != key[:7]}
    balancing_analysis.run_balancing(measured.id, values=values)
    assert fake_rossml.seen["direct"] is None
    assert fake_rossml.seen["indirect"] == INDIRECT


def test_the_parameters_are_refused_when_they_cannot_be_used(measured):
    for values in (
        {"unknown": 1},
        {"mu_d_min": 100},
        {"w_f0": "abc"},
        {"w_f0": True},
        {"w_f0": float("inf")},
        {"number_points": 10.5},
        {"indirect_P_exponent_min": 1.5},
        {"direct_P": 0.1},
        {"indirect_P_mantissa_min": 2},
        {**PARAMETERS, "indirect_P_mantissa_min": 9, "indirect_P_mantissa_max": 2},
        {**PARAMETERS, "indirect_S_exponent_min": 9, "indirect_S_exponent_max": 2},
        [1, 2],
    ):
        with pytest.raises(ValueError):
            balancing_analysis.set_parameters(measured.id, values)


def test_the_parameters_routes(measured, fake_rossml):
    client = create_app().test_client()
    answer = client.post(
        "/api/dt/balancing-parameters",
        json={"session": measured.id, "parameters": {"number_points": 2048}},
        headers=HEADERS,
    )
    assert answer.status_code == 200
    parameters = answer.get_json()["parameters"]
    assert parameters["values"]["number_points"] == 2048
    assert parameters["values"]["w_f0"] is None
    refused = client.post(
        "/api/dt/balancing-parameters",
        json={"session": measured.id, "parameters": {"number_points": "x"}},
        headers=HEADERS,
    )
    assert refused.status_code == 400
    run = client.post(
        "/api/dt/balancing",
        json={"session": measured.id, "parameters": PARAMETERS},
        headers=HEADERS,
    )
    assert run.status_code == 200
    assert fake_rossml.seen["number_points"] == 2048


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
    run = FakeAPI.run

    def subset(self, **arguments):
        run(self, **arguments)
        self.results_balance.dof_forces = [25 * 6]
        self.mu_d = [68.0]
        self.phi = {"Sensor #1": [200.0]}

    monkeypatch.setattr(FakeAPI, "run", subset)
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


@pytest.mark.parametrize("installed", [True, False])
def test_the_screen_is_told_whether_the_balancing_can_run(monkeypatch, installed):
    monkeypatch.setattr(rossml_balancing, "available", lambda: installed)
    client = create_app().test_client()
    assert client.get("/api/dt/features").status_code == 403
    answer = client.get("/api/dt/features", headers=HEADERS)
    assert answer.get_json() == {"balancing": installed}


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
    charts = client.post(
        "/api/dt/balancing-figure",
        json={"session": measured.id, "chart": "polar", "sensor": 0},
        headers=HEADERS,
    )
    assert charts.status_code == 200
    assert len(charts.get_json()["charts"]) == 2
    one = client.post(
        "/api/dt/balancing-figure",
        json={"session": measured.id, "chart": "vs_2d", "node": 8},
        headers=HEADERS,
    )
    assert [chart["node"] for chart in one.get_json()["charts"]] == [8]
    wrong = client.post(
        "/api/dt/balancing-figure",
        json={"session": measured.id, "chart": "vs_2d", "node": 99},
        headers=HEADERS,
    )
    assert wrong.status_code == 400
    refused = client.post(
        "/api/dt/balancing",
        json={"session": measured.id, "iterations": 99},
        headers=HEADERS,
    )
    assert refused.status_code == 400
    assert client.post("/api/dt/balancing", json={}).status_code == 403


def sample_signals():
    """The folder of waveform files rossml keeps as its sample, or `None`."""
    spec = importlib.util.find_spec("rossml") if rossml_balancing.available() else None
    if spec is None or not spec.origin:
        return None
    folder = os.path.join(os.path.dirname(os.path.dirname(spec.origin)), "database")
    files = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    return folder if len(files) >= 4 else None


@pytest.mark.skipif(not rossml_balancing.available(), reason="rossml is not installed")
def test_the_arguments_are_those_of_rossml():
    # The stand-in above takes what the interface passes; this is the check
    # that the real `API.run` takes the same, and that every key the
    # interface owns is one `API.run` would otherwise read from the file.
    from rossml.api import API

    real = inspect.signature(API.run).parameters
    stand_in = inspect.signature(FakeAPI.run).parameters
    assert set(stand_in) <= set(real)
    assert rossml_balancing.OWNED - {"self", "kwargs"} <= set(real)
    for name, kind in rossml_balancing.CHARTS.items():
        assert callable(getattr(API, kind)), name


# statsmodels 0.15 warns that `adfuller` will return a result object instead of
# a tuple in 0.16 or July 2027. rossml calls it, so the fix belongs there and
# the interface's rule that a FutureWarning fails the run is held off for this
# one test (noted 2026-11-18).
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.skipif(
    sample_signals() is None,
    reason="rossml, with its sample signals, is not installed here",
)
def test_a_real_run_answers_a_correction_per_plane_and_draws_every_chart(measured):
    # rossml's own sample signals, renamed to this rotor's measuring nodes.
    # They were measured on another machine, so the numbers are not a physical
    # reference: what is checked is that a run answers one finite correction
    # mass per balancing plane, at the radius given, half a turn from the
    # estimated unbalance, and that every chart the screen offers is drawn.
    folder = sample_signals()
    files = sorted(name for name in os.listdir(folder) if name.endswith(".csv"))
    for node, (x_file, y_file) in zip(MEASURING, (files[:2], files[2:4]), strict=True):
        for plane, name in (("X", x_file), ("Y", y_file)):
            shutil.copy(
                os.path.join(folder, name),
                measurement.waveform_path(measured, node, plane),
            )

    answer = balancing_analysis.run_balancing(measured.id)
    result = answer["balancing"]
    planes = result["planes"]
    assert [plane["node"] for plane in planes] == [14, 25]
    for plane in planes:
        assert plane["mass"] > 0
        assert_allclose(plane["mass"], plane["unbalance"] / plane["radius"])
        assert_allclose(plane["phase"], (plane["unbalance_phase"] + 180.0) % 360.0)
        assert 0 <= plane["phase"] < 360
    # Four sensors on two measuring planes.
    assert [sensor["index"] for sensor in result["sensors"]] == [0, 1, 2, 3]

    for sensor in result["sensors"]:
        charts = balancing_analysis.figure_for(
            measured.id, "polar", sensor=sensor["index"]
        )["charts"]
        assert [chart["node"] for chart in charts] == [14, 25]
        assert all(chart["figure"]["data"] for chart in charts)
    for kind, nodes in (
        ("forces_1d", [14, 25]),
        ("forces_2d", [14, 25]),
        ("vs_1d", result["vs_nodes"]),
        ("vs_2d", result["vs_nodes"]),
    ):
        for node in nodes:
            charts = balancing_analysis.figure_for(measured.id, kind, node=node)[
                "charts"
            ]
            assert charts[0]["figure"]["data"], (kind, node)
    charts = balancing_analysis.figure_for(measured.id, "vs_3d")["charts"]
    assert charts[0]["figure"]["data"]
    charts = balancing_analysis.figure_for(
        measured.id, "decomposed", node=14, positions=[0, 90], radii=[68, 68]
    )["charts"]
    assert charts[0]["figure"]["data"]
