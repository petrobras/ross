# -*- coding: utf-8 -*-
"""Steps 1 to 3 of the Digital Twin, with the LMEST bench rotor."""

import os
import tempfile

import pytest

os.environ.setdefault("DT_SESSIONS_DIR", tempfile.mkdtemp(prefix="dt-sessions-"))

from ross.interface.tests.dt_app import HEADERS, create_app, requires_dt_app
from ross.interface.extensions.digital_twin.domain import equipment, network, session
from ross.interface.extensions.digital_twin.services.digital_twin import (
    create_model,
    load_process_data,
)

LMEST = "LMEST-2-DISCOS"
PI_TIME = "2026/11/11 00:00:00"

pytestmark = requires_dt_app


@pytest.fixture(autouse=True)
def inside_network(monkeypatch):
    monkeypatch.setattr(network, "on_corporate_network", lambda: True)


def _all_equipment():
    return [item for platform in equipment.catalog() for item in platform["equipment"]]


def test_streamlit_models_are_not_refused_for_importing_streamlit():
    items = {item["tag"]: item for item in _all_equipment()}
    assert items[LMEST]["available"]
    for item in items.values():
        assert "Streamlit" not in item["reason"], item


def test_unknown_tag_and_bad_time_are_refused():
    current = session.create()
    for tag, pi_time in (("DOES-NOT-EXIST", PI_TIME), (LMEST, "11/11/2026")):
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
        assert "process data" in str(error)
    else:
        raise AssertionError("the model should wait for the process data")


def test_routes_walk_the_three_steps():
    client = create_app().test_client()
    assert client.get("/api/dt/equipment").status_code == 403

    opened = client.post("/api/dt/session", headers=HEADERS).get_json()
    data = client.post(
        "/api/dt/process-data",
        json={
            "session": opened["id"],
            "equipment": LMEST,
            "pi_time": PI_TIME,
            "rotation": "clockwise",
        },
        headers=HEADERS,
    )
    assert data.status_code == 200
    model = client.post(
        "/api/dt/model", json={"session": opened["id"]}, headers=HEADERS
    )
    assert model.status_code == 200
    assert model.get_json()["session"]["has_rotor"]

    unknown = client.post(
        "/api/dt/model", json={"session": "../../etc"}, headers=HEADERS
    )
    assert unknown.status_code == 400
