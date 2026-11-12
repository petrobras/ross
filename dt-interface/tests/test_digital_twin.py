# -*- coding: utf-8 -*-
"""Steps 1 to 3 of the Digital Twin, with the LMEST bench rotor."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DT_SESSIONS_DIR", tempfile.mkdtemp(prefix="dt-sessions-"))

from api import SESSION_TOKEN, create_app
from domain import equipment, session
from services.digital_twin import create_model, load_process_data

HEADERS = {"X-ROSS-Token": SESSION_TOKEN}
LMEST = "LMEST-2-DISCOS"
PI_TIME = "2026/11/11 00:00:00"


def _all_equipment():
    return [item for platform in equipment.catalog() for item in platform["equipment"]]


def test_lmest_is_available_and_streamlit_models_are_not():
    items = {item["tag"]: item for item in _all_equipment()}
    assert items[LMEST]["available"]
    assert not items["P77-C-UC-1254001 A-LP"]["available"]
    assert "Streamlit" in items["P77-C-UC-1254001 A-LP"]["reason"]


def test_a_streamlit_model_is_refused_before_import():
    current = session.create()
    try:
        load_process_data(current.id, "P77-C-UC-1254001 A-LP", PI_TIME, "clockwise")
    except ValueError as error:
        assert "Streamlit" in str(error)
    else:
        raise AssertionError("a Streamlit model should not run here")


def test_unknown_tag_and_bad_time_are_refused():
    current = session.create()
    for tag, pi_time in (("NAO-EXISTE", PI_TIME), (LMEST, "11/11/2026")):
        try:
            load_process_data(current.id, tag, pi_time, "clockwise")
        except ValueError:
            pass
        else:
            raise AssertionError("expected a refusal for %s %s" % (tag, pi_time))


def test_lmest_model_is_created_and_saved():
    current = session.create()
    loaded = load_process_data(current.id, LMEST, PI_TIME, "counterclockwise")
    assert loaded["table"]["rows"]
    assert loaded["session"]["rotation"] == "counterclockwise"

    built = create_model(current.id)
    assert built["rotor"]["nodes"] == 39
    assert built["rotor"]["bearings"] == 2
    assert built["figure"]["data"]
    assert os.path.isfile(os.path.join(current.session_files, "lmest_2discos.toml"))


def test_model_needs_process_data_first():
    current = session.create()
    try:
        create_model(current.id)
    except ValueError as error:
        assert "dados de processo" in str(error)
    else:
        raise AssertionError("the model should wait for the process data")


def test_routes_walk_the_three_steps():
    client = create_app().test_client()
    assert client.get("/api/dt/equipment").status_code == 403

    opened = client.post("/api/dt/session", headers=HEADERS).get_json()
    data = client.post(
        "/api/dt/process-data",
        json={"session": opened["id"], "equipment": LMEST, "pi_time": PI_TIME, "rotation": "clockwise"},
        headers=HEADERS,
    )
    assert data.status_code == 200
    model = client.post("/api/dt/model", json={"session": opened["id"]}, headers=HEADERS)
    assert model.status_code == 200
    assert model.get_json()["session"]["has_rotor"]

    unknown = client.post("/api/dt/model", json={"session": "../../etc"}, headers=HEADERS)
    assert unknown.status_code == 400
