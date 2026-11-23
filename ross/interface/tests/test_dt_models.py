# -*- coding: utf-8 -*-
"""Pre-established rotor models, behind the network check.

These tests build a small Digital Twin app in a temporary folder, with a
model written the way the `models_files` models are (`get_pi_data` and
`create_model`, which reads `configured_data`), so they run without the
private repository or the Petrobras network.
"""

import os
import sys
import tempfile
import textwrap

os.environ.setdefault("DT_SESSIONS_DIR", tempfile.mkdtemp(prefix="dt-sessions-"))

import pytest
from numpy.testing import assert_allclose

from ross.interface.extensions.digital_twin.domain import (
    dt_app,
    equipment,
    model_settings,
    network,
    session,
)
from ross.interface.extensions.digital_twin.services import digital_twin, measurement
from ross.interface.tests.dt_app import HEADERS, create_app

TAG = "P99-C-UC-0000001 A"
MISSING_TAG = "P99-C-UC-0000002 A"
PI_TIME = "2026/11/11 00:00:00"

MODEL = """
import pandas as pd
import ross as rs
from ross.units import Q_

try:
    import pandaspi as pdpi
except ImportError:
    pass

RADIAL_BEARING_CLEARANCE = Q_(0.16 / 2, "mm")
OIL_SUPPLY_TEMPERATURE = Q_(40, "degC")


def get_pi_data(compressor, initial_time, test=False):
    if not test:
        raise RuntimeError("the PI server is not reachable from the tests")
    return pd.DataFrame({"speed": [3000.0, 3010.0]}, index=["t0", "t1"])


def create_model(df, fluid=None, **kwargs):
    clearance = Q_(
        float(kwargs["configured_data"]["RADIAL_BEARING_CLEARANCE"][0]),
        kwargs["configured_data"]["RADIAL_BEARING_CLEARANCE"][1],
    )
    steel = rs.Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
    shaft = [rs.ShaftElement(L=0.25, idl=0, odl=0.05, material=steel) for _ in range(4)]
    disk = rs.DiskElement.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    stiffness = 1e6 * (0.08e-3 / clearance.to("m").m)
    bearings = [
        rs.BearingElement(n=0, kxx=stiffness, cxx=0),
        rs.BearingElement(n=4, kxx=stiffness, cxx=0),
    ]
    return rs.Rotor(shaft, [disk], bearings), bearings, ["There is no seals"]
"""


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(text))


@pytest.fixture
def fake_app(tmp_path, monkeypatch):
    """A Digital Twin app with two models, and the network check passing."""
    _write(str(tmp_path / "models_files" / "__init__.py"), "")
    _write(str(tmp_path / "models_files" / "fake_rotor" / "fake_rotor.py"), MODEL)
    _write(
        str(tmp_path / "models_files" / "needs_pkg" / "needs_pkg.py"),
        "import pandas as pd\n",
    )
    _write(
        str(tmp_path / "models_files" / "needs_pkg" / "create_labyrinth_seals.py"),
        "import package_that_is_not_installed_anywhere\n",
    )
    _write(
        str(tmp_path / "utils" / "settings.json"),
        '{"%s": {"rotor_model": "fake_rotor"},'
        ' "%s": {"rotor_model": "needs_pkg"}}' % (TAG, MISSING_TAG),
    )
    for module in (dt_app, equipment):
        monkeypatch.setattr(module, "APP_DIR", str(tmp_path))
    monkeypatch.setattr(network, "on_corporate_network", lambda: True)
    cached = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "models_files" or name.startswith("models_files.")
    }
    yield tmp_path
    for name in list(sys.modules):
        if name == "models_files" or name.startswith("models_files."):
            del sys.modules[name]
    sys.modules.update(cached)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


def _status(tag):
    for platform in equipment.catalog():
        for item in platform["equipment"]:
            if item["tag"] == tag:
                return item
    raise KeyError(tag)


def test_a_model_with_its_packages_installed_is_available(fake_app):
    assert _status(TAG)["available"]


def test_a_model_whose_folder_imports_a_missing_package_is_refused(fake_app):
    item = _status(MISSING_TAG)
    assert not item["available"]
    assert "package_that_is_not_installed_anywhere" in item["reason"]
    current = session.create()
    with pytest.raises(ValueError, match="package_that_is_not_installed_anywhere"):
        digital_twin.load_process_data(current.id, MISSING_TAG, PI_TIME, "clockwise")


def test_settings_are_read_from_the_source_without_importing_it(fake_app):
    answer = digital_twin.model_settings_for(TAG)
    assert answer["model"] == "fake_rotor"
    assert answer["settings"] == [
        {"name": "RADIAL_BEARING_CLEARANCE", "value": 0.08, "unit": "mm"},
        {"name": "OIL_SUPPLY_TEMPERATURE", "value": 40.0, "unit": "degC"},
    ]
    assert "models_files.fake_rotor.fake_rotor" not in sys.modules


def test_only_quantities_of_numbers_are_settings():
    source = textwrap.dedent(
        """
        A = Q_(2 * 3 - 1, "mm")
        B = Q_(open("x").read(), "mm")
        C = 5
        D = Q_(1, unit)
        E = Q_(-1 / 4, "N/m")
        """
    )
    assert model_settings.defaults(source) == [
        {"name": "A", "value": 5.0, "unit": "mm"},
        {"name": "E", "value": -0.25, "unit": "N/m"},
    ]


def test_chosen_settings_are_checked_against_the_declared_ones():
    declared = [{"name": "RADIAL_BEARING_CLEARANCE", "value": 0.08, "unit": "mm"}]
    assert model_settings.configured_data(declared, {}) == {
        "RADIAL_BEARING_CLEARANCE": [0.08, "mm"]
    }
    assert model_settings.configured_data(
        declared, {"RADIAL_BEARING_CLEARANCE": {"value": "0.1", "unit": "inch"}}
    ) == {"RADIAL_BEARING_CLEARANCE": [0.1, "inch"]}
    for chosen in (
        {"OTHER": {"value": 1}},
        {"RADIAL_BEARING_CLEARANCE": {"value": "abc"}},
        {"RADIAL_BEARING_CLEARANCE": {"value": 1, "unit": "degC"}},
        {"RADIAL_BEARING_CLEARANCE": {"value": 1, "unit": "not_a_unit"}},
    ):
        with pytest.raises(ValueError):
            model_settings.configured_data(declared, chosen)


def test_the_model_runs_and_receives_the_settings(fake_app):
    current = session.create()
    loaded = digital_twin.load_process_data(
        current.id, TAG, PI_TIME, "clockwise", "sample"
    )
    assert loaded["table"]["columns"] == ["Time", "speed"]
    assert loaded["session"]["source"] == "sample"

    # Two clearances: the bearing stiffness of the model is inversely
    # proportional to the clearance, so the rotor must change with it.
    built = {}
    for clearance in (0.08, 0.16):
        answer = digital_twin.create_model(
            current.id,
            {"RADIAL_BEARING_CLEARANCE": {"value": clearance, "unit": "mm"}},
        )
        assert answer["settings"]["RADIAL_BEARING_CLEARANCE"] == [clearance, "mm"]
        assert answer["settings"]["OIL_SUPPLY_TEMPERATURE"] == [40.0, "degC"]
        built[clearance] = current.rotor.bearing_elements[0].kxx_interpolated(0)
    assert_allclose(built[0.08], 1e6)
    assert_allclose(built[0.16], 0.5e6)
    assert os.path.isfile(os.path.join(current.session_files, "fake_rotor.toml"))


def test_the_rotor_goes_to_the_hub_as_a_project(fake_app):
    current = session.create()
    digital_twin.load_process_data(current.id, TAG, PI_TIME, "clockwise", "sample")
    with pytest.raises(ValueError, match="Create the model"):
        digital_twin.rotor_project(current.id)
    digital_twin.create_model(current.id)

    answer = digital_twin.rotor_project(current.id)
    project = answer["projectData"]
    assert project["name"] == "%s (%s)" % (TAG, PI_TIME)
    assert len(project["shafts"]) == 4
    assert len(project["disks"]) == 1
    assert len(project["bearings"]) == 2
    assert answer["warnings"] == []


def test_pi_data_needs_pandaspi_and_credentials(fake_app, monkeypatch):
    current = session.create()
    real_find_spec = digital_twin.importlib.util.find_spec
    monkeypatch.setattr(
        digital_twin.importlib.util,
        "find_spec",
        lambda name: None if name == "pandaspi" else real_find_spec(name),
    )
    with pytest.raises(ValueError, match="pandaspi package is not installed"):
        digital_twin.load_process_data(current.id, TAG, PI_TIME, "clockwise", "pi")

    monkeypatch.setattr(
        digital_twin.importlib.util,
        "find_spec",
        lambda name: object() if name == "pandaspi" else real_find_spec(name),
    )
    for name in digital_twin.PI_CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="USER_SWAGGER and SWAGGER_PASSWORD"):
        digital_twin.load_process_data(current.id, TAG, PI_TIME, "clockwise", "pi")

    with pytest.raises(ValueError, match="source"):
        digital_twin.load_process_data(current.id, TAG, PI_TIME, "clockwise", "csv")


def test_the_routes_walk_to_the_hub(fake_app):
    client = create_app().test_client()
    opened = client.post("/api/dt/session", headers=HEADERS).get_json()
    settings = client.get(
        "/api/dt/model-settings", query_string={"equipment": TAG}, headers=HEADERS
    )
    assert settings.get_json()["model"] == "fake_rotor"
    data = client.post(
        "/api/dt/process-data",
        json={
            "session": opened["id"],
            "equipment": TAG,
            "pi_time": PI_TIME,
            "rotation": "clockwise",
            "source": "sample",
        },
        headers=HEADERS,
    )
    assert data.status_code == 200
    model = client.post(
        "/api/dt/model",
        json={
            "session": opened["id"],
            "settings": {"RADIAL_BEARING_CLEARANCE": {"value": 0.1, "unit": "mm"}},
        },
        headers=HEADERS,
    )
    assert model.status_code == 200
    project = client.post(
        "/api/dt/project", json={"session": opened["id"]}, headers=HEADERS
    )
    assert project.status_code == 200
    assert len(project.get_json()["projectData"]["bearings"]) == 2


def test_outside_the_network_only_the_pre_established_routes_are_refused(
    monkeypatch,
):
    monkeypatch.setattr(network, "on_corporate_network", lambda: False)
    client = create_app().test_client()
    status = client.get("/api/dt/network", headers=HEADERS)
    assert status.status_code == 200
    assert status.get_json() == {"available": False, "message": network.refusal()}
    for method, path in (
        ("get", "/api/dt/equipment"),
        ("get", "/api/dt/model-settings"),
        ("post", "/api/dt/process-data"),
        ("post", "/api/dt/model"),
        ("post", "/api/dt/project"),
    ):
        answer = getattr(client, method)(path, json={}, headers=HEADERS)
        assert answer.status_code == 403, path
        assert answer.get_json()["message"] == network.refusal()
    # The user's own rotors: a 400 for the empty body, not the network refusal.
    for path in ("/api/dt/user-model", "/api/dt/planes", "/api/dt/sensor"):
        answer = client.post(path, json={}, headers=HEADERS)
        assert answer.status_code == 400, path
    assert client.post("/api/dt/session", headers=HEADERS).status_code == 200
    assert client.get("/api/dt/network").status_code == 403


def test_the_network_check_opens_a_connection_and_remembers(monkeypatch):
    calls = []

    def reachable(host, port, timeout=network.TIMEOUT):
        calls.append((host, port))
        return host == "inside.example"

    monkeypatch.delenv("DT_NETWORK_CHECK", raising=False)
    monkeypatch.setattr(network, "reachable", reachable)
    monkeypatch.setenv("DT_NETWORK_PROBE", "outside.example:443, inside.example:80")
    network.forget()
    try:
        assert network.on_corporate_network(now=0.0)
        assert calls == [("outside.example", 443), ("inside.example", 80)]
        assert network.on_corporate_network(now=10.0)
        assert len(calls) == 2
        monkeypatch.setenv("DT_NETWORK_PROBE", "outside.example:443")
        assert not network.on_corporate_network(now=10.0 + network.CACHE_SECONDS)

        monkeypatch.setenv("DT_NETWORK_CHECK", "off")
        network.forget()
        assert network.on_corporate_network(now=0.0)
        assert len(calls) == 3
    finally:
        network.forget()


def test_without_a_probe_host_the_check_fails_closed(monkeypatch):
    # No internal address is built in, so an interface that was never told
    # which host to test must answer "not on the network" and say what to set,
    # rather than offer the models to everyone.
    monkeypatch.delenv("DT_NETWORK_CHECK", raising=False)
    monkeypatch.delenv("DT_NETWORK_PROBE", raising=False)
    monkeypatch.setattr(
        network, "reachable", lambda host, port, timeout=network.TIMEOUT: True
    )
    network.forget()
    try:
        assert network.probes() == []
        assert not network.on_corporate_network(now=0.0)
        assert network.refusal() == network.UNSET
        monkeypatch.setenv("DT_NETWORK_PROBE", "inside.example:443")
        assert network.refusal() == network.OUTSIDE
    finally:
        network.forget()


def test_a_closed_port_is_not_reachable():
    assert not network.reachable("127.0.0.1", 9, timeout=0.5)


# The format the measuring system exports and `rossml` reads: a title line,
# nine lines of `name;value`, the line that names the columns, and then the
# samples with a comma for the decimal point. `rossml` looks the speed and the
# units up by name and reads the samples with `skiprows=11`
# (rossml/datapipeline/methods.py), so a file without this header cannot be
# used for the analysis.
WAVEFORM = (
    "Machine Name;RIG\n"
    "Point Name;XT-1-X\n"
    "Wf Amp;1,0\n"
    "Number of Revs;16\n"
    "X-Axis Unit;ms\n"
    "Y-Axis Unit;um\n"
    "Sample Speed;840 rpm\n"
    "Sample Status;Valid\n"
    "Timestamp;01/01/2026 00:00:00\n"
    "Variable;Disp Wf\n"
    "X-Axis Value;Y-Axis Value\n"
    "0;0,134\n"
    "0,5;0,210\n"
    "1;-0,180\n"
)

PLANE = {"node": 2, "radius": 100.0}


@pytest.fixture
def modelled(fake_app):
    """A session with the fake rotor built: 5 nodes, 0 to 4."""
    current = session.create()
    digital_twin.load_process_data(current.id, TAG, PI_TIME, "clockwise", "sample")
    digital_twin.create_model(current.id)
    return current


def test_planes_wait_for_the_model(fake_app):
    current = session.create()
    with pytest.raises(ValueError, match="Create the model"):
        measurement.set_planes(current.id, [1, 3], [PLANE])


def test_planes_are_rotor_nodes_each_at_most_once(modelled):
    step = measurement.set_planes(modelled.id, [3, 1], [PLANE])
    assert step["nodes"] == 5
    # The measuring nodes are sorted: `rossml` matches the signal files to the
    # sensors in the order the file names sort, so the order they were added
    # in cannot decide which signal lands on which node.
    assert step["measuring"] == [1, 3]
    assert step["balancing"] == [{"node": 2, "radius": 100.0}]
    assert step["pending"] == ["1 X", "1 Y", "3 X", "3 Y"]
    assert not step["complete"]
    for measuring in ([1, 1], [5], [-1], ["1"], [True], "1"):
        with pytest.raises(ValueError):
            measurement.set_planes(modelled.id, measuring, [])


def test_a_balancing_plane_needs_a_radius(modelled):
    # The analysis estimates a mass times a radius, so without the radius
    # there is no correction mass to report.
    for balancing in ([2], [{"node": 2}], [{"node": 2, "radius": 0}], [{"radius": 1}]):
        with pytest.raises(ValueError):
            measurement.set_planes(modelled.id, [1, 3], balancing)


def test_sensors_write_the_waveforms_and_place_y_after_x(modelled):
    measurement.set_planes(modelled.id, [1, 3], [PLANE])
    for node in (1, 3):
        for plane in ("X", "Y"):
            step = measurement.apply_sensor(
                modelled.id, node, plane, 45.0, "signal.csv", WAVEFORM
            )
    assert step["complete"] and step["pending"] == []
    sensors = step["sensors"]["1"]
    assert sensors["X"]["probes_angle"] == 45.0
    assert sensors["Y"]["probes_angle"] == 135.0
    assert sensors["X"]["rows"] == 3

    path = measurement.waveform_path(modelled, 3, "Y")
    assert path.endswith(os.path.join("database", "data_3_y.csv"))
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == WAVEFORM

    # A new X angle without a file keeps the file and moves the Y sensor.
    step = measurement.apply_sensor(modelled.id, 1, "X", 10.0)
    assert step["sensors"]["1"]["X"]["filename"] == "signal.csv"
    assert step["sensors"]["1"]["Y"]["probes_angle"] == 100.0


def test_a_sensor_needs_a_valid_csv_the_first_time(modelled):
    measurement.set_planes(modelled.id, [1, 3], [])
    with pytest.raises(ValueError, match="Choose the signal file"):
        measurement.apply_sensor(modelled.id, 1, "X", 0.0)
    with pytest.raises(ValueError, match=".csv"):
        measurement.apply_sensor(modelled.id, 1, "X", 0.0, "signal.txt", WAVEFORM)
    with pytest.raises(ValueError, match="format of the measuring system"):
        measurement.apply_sensor(
            modelled.id, 1, "X", 0.0, "signal.csv", "time;amplitude\n0;1\n"
        )
    with pytest.raises(ValueError, match="not a measuring plane"):
        measurement.apply_sensor(modelled.id, 2, "X", 0.0, "signal.csv", WAVEFORM)
    with pytest.raises(ValueError, match="X or Y"):
        measurement.apply_sensor(modelled.id, 1, "Z", 0.0, "signal.csv", WAVEFORM)


def test_removing_a_measuring_plane_drops_its_sensors(modelled):
    measurement.set_planes(modelled.id, [1, 3], [])
    measurement.apply_sensor(modelled.id, 3, "X", 0.0, "signal.csv", WAVEFORM)
    path = measurement.waveform_path(modelled, 3, "X")
    assert os.path.isfile(path)
    step = measurement.set_planes(modelled.id, [1], [])
    assert "3" not in step["sensors"]
    assert not os.path.isfile(path)


def test_the_planes_routes(modelled):
    client = create_app().test_client()
    planes = client.post(
        "/api/dt/planes",
        json={"session": modelled.id, "measuring": [0, 4], "balancing": [PLANE]},
        headers=HEADERS,
    )
    assert planes.status_code == 200
    sensor = client.post(
        "/api/dt/sensor",
        json={
            "session": modelled.id,
            "node": 4,
            "plane": "X",
            "angle": 30,
            "filename": "s.csv",
            "content": WAVEFORM,
        },
        headers=HEADERS,
    )
    assert sensor.status_code == 200
    assert sensor.get_json()["sensors"]["4"]["X"]["probes_angle"] == 30.0
    refused = client.post(
        "/api/dt/planes",
        json={"session": modelled.id, "measuring": [9], "balancing": []},
        headers=HEADERS,
    )
    assert refused.status_code == 400


@pytest.fixture
def hub_project(tmp_path):
    """The project the Rotor Hub keeps for `rs.rotor_example()`.

    Read through "Load Rotor" (`domain/ross_import.py`), a different path
    from the one under test.
    """
    from ross.interface.domain.ross_import import project_from_ross_file
    import ross as rs

    path = tmp_path / "example.toml"
    rs.rotor_example().save(str(path))
    project = project_from_ross_file(path.read_text(encoding="utf-8"))
    project["name"] = "My compressor"
    return project


def test_a_hub_rotor_is_the_model_and_the_planes_follow_it(hub_project):
    import ross as rs

    current = session.create()
    answer = digital_twin.create_user_model(
        current.id, hub_project, "My compressor", "counterclockwise"
    )
    reference = rs.rotor_example()
    assert answer["rotor"]["nodes"] == len(reference.nodes)
    assert answer["rotor"]["bearings"] == len(reference.bearing_elements)
    assert_allclose(answer["rotor"]["mass"], reference.m)
    assert answer["session"]["equipment"] == "My compressor"
    assert answer["session"]["rotation"] == "counterclockwise"
    assert answer["measurement"]["nodes"] == len(reference.nodes)
    assert os.path.isfile(os.path.join(current.session_files, "user_rotor.toml"))

    step = measurement.set_planes(current.id, [2, 4], [{"node": 3, "radius": 50.0}])
    assert step["measuring"] == [2, 4]


def test_a_hub_rotor_is_refused_when_it_cannot_be_built(hub_project):
    current = session.create()
    with pytest.raises(ValueError):
        digital_twin.create_user_model(current.id, hub_project, "R", "upwards")
    with pytest.raises(ValueError):
        digital_twin.create_user_model(current.id, None, "R", "clockwise")
    with pytest.raises(ValueError):
        digital_twin.create_user_model(current.id, hub_project, " ", "clockwise")


def test_the_user_model_route_answers_outside_the_network(hub_project, monkeypatch):
    monkeypatch.setattr(network, "on_corporate_network", lambda: False)
    client = create_app().test_client()
    opened = client.post("/api/dt/session", headers=HEADERS).get_json()
    answer = client.post(
        "/api/dt/user-model",
        json={"session": opened["id"], "project": hub_project, "name": "My compressor"},
        headers=HEADERS,
    )
    assert answer.status_code == 200
    assert answer.get_json()["figure"]["data"]
