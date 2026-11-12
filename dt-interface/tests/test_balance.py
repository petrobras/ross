# -*- coding: utf-8 -*-
"""BRBalance answers from the existing calculator, through the new route."""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import SESSION_TOKEN, create_app
from services.balance import load_brb, report_bytes, run, save_brb


def _one_plane():
    """One sensor, one test mass of 10 at 0°, current vibration 10 at 0°."""
    return {
        "n_sensors": 1,
        "n_tests": 1,
        "n_planes": 1,
        "n_weights": 1,
        "ref_vib_amp": [0],
        "ref_vib_phase": [0],
        "ref_run_amp": [0],
        "ref_run_phase": [0],
        "test1_vib_amp": [10],
        "test_1_vib_phase": [0],
        "test0_plane0_mass": [10],
        "test0_plane0_phase": [0],
        "curr_vib_amp": [10],
        "curr_vib_phase": [0],
        "curr_run_amp": [0],
        "curr_run_phase": [0],
        "corr_plane0_mass": [0],
        "corr_plane0_phase": [0],
    }


def test_one_plane_correction_is_opposite_the_vibration():
    answer = run(_one_plane())
    assert abs(answer["correction"][0]["magnitude"] - 10) < 1e-6
    assert abs(answer["correction"][0]["phase"] - 180) < 1e-6
    assert answer["figure"]["data"]


def test_incomplete_tests_are_refused():
    state = _one_plane()
    state["test0_plane0_mass"] = [0]
    try:
        run(state)
    except ValueError as error:
        assert "teste" in str(error).lower()
    else:
        raise AssertionError("zeros should not produce a correction")


def test_brb_roundtrip_keeps_the_form():
    state = _one_plane()
    state["session_name"] = "bancada"
    state["notes"] = "primeira rodada"
    blob, name = save_brb(state)
    assert name == "bancada.brb"
    loaded = load_brb(blob)
    assert loaded["notes"] == "primeira rodada"
    assert loaded["test1_vib_amp"] == [10]


def test_report_is_a_docx():
    payload, name = report_bytes(_one_plane())
    assert name.endswith(".docx")
    assert payload.startswith(b"PK")


def test_route_requires_the_token_and_returns_the_correction():
    client = create_app().test_client()
    refused = client.post("/api/balance/calculate", json=_one_plane())
    assert refused.status_code == 403

    accepted = client.post(
        "/api/balance/calculate",
        json=_one_plane(),
        headers={"X-ROSS-Token": SESSION_TOKEN},
    )
    assert accepted.status_code == 200
    assert accepted.get_json()["correction"][0]["phase"] == 180 or abs(
        accepted.get_json()["correction"][0]["phase"] - 180
    ) < 1e-6


def test_open_route_reads_a_brb():
    blob, _name = save_brb(_one_plane())
    client = create_app().test_client()
    response = client.post(
        "/api/balance/open",
        data={"file": (io.BytesIO(blob), "caso.brb")},
        content_type="multipart/form-data",
        headers={"X-ROSS-Token": SESSION_TOKEN},
    )
    assert response.status_code == 200
    assert response.get_json()["n_planes"] == 1


def test_a_broken_zip_is_a_client_error():
    client = create_app().test_client()
    response = client.post(
        "/api/balance/open",
        data={"file": (io.BytesIO(b"not a zip"), "caso.brb")},
        content_type="multipart/form-data",
        headers={"X-ROSS-Token": SESSION_TOKEN},
    )
    assert response.status_code == 400
    assert "inválido" in response.get_json()["message"]
