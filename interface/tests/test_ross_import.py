# -*- coding: utf-8 -*-
"""Round trip through a native ROSS file.

This is the route that slice 4 moved **and** changed the behaviour of, and the
only one with no end-to-end coverage. Before, `/load_ross_file` answered **200
with status "error"** when something went wrong: a client looking only at the
HTTP code would think it had worked. Now the central handler gives back 400.

And its body -- translating a ROSS file into the screen's project -- moved to
`domain/ross_import.py`. That function converts each quantity from the SI base
unit to the unit the form uses, and a mistake there would not show up as an
error: it would show up as a rotor with the wrong dimensions.

The test closes the circuit: it builds a rotor, has ROSS write it, reads the
file back through the route, rebuilds the rotor from what the route gave back
and compares mass and inertia with the original. If the unit conversion is
wrong at any point, the mass does not match."""

import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from api import create_app
from api.security import SESSION_TOKEN
from domain.rotor_builder import build_rotor_from_ui

AUTH = {"X-ROSS-Token": SESSION_TOKEN}

try:
    import ross as rs

    HAS_ROSS = True
except Exception:  # pragma: no cover
    HAS_ROSS = False

needs_ross = pytest.mark.skipif(not HAS_ROSS, reason="requires ROSS installed")


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _original_rotor():
    steel = rs.materials.steel
    shafts = [
        rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel, n=i)
        for i in range(3)
    ]
    return rs.Rotor(
        shaft_elements=shafts,
        disk_elements=[rs.DiskElement(n=1, m=5.0, Id=0.02, Ip=0.04, tag="disk_0")],
        bearing_elements=[
            rs.BearingElement(n=0, kxx=1e6, cxx=1e3, tag="b0"),
            rs.BearingElement(n=3, kxx=1e6, cxx=1e3, tag="b1"),
        ],
    )


@needs_ross
def test_a_rotor_survives_the_round_trip_through_a_ross_file(client):
    """Write with ROSS, read through the route, rebuild -- and the mass has to match.

    The comparison is by mass and inertia on purpose: they are numbers that depend
    on **every** converted dimension. A millimetre read as a metre in any field
    changes the mass by orders of magnitude."""
    original = _original_rotor()

    with tempfile.TemporaryDirectory() as folder:
        handle = os.path.join(folder, "rotor.toml")
        original.save(handle)
        with open(handle, encoding="utf-8") as fh:
            content = fh.read()

    response = client.post("/load_ross_file", json={"content": content}, headers=AUTH)
    assert response.status_code == 200, response.json
    assert response.json["status"] == "success"

    project = response.json["projectData"]
    assert project["shafts"], "the file had shafts and the translation lost every one"

    rebuilt = build_rotor_from_ui(project)
    assert rebuilt.m == pytest.approx(original.m, rel=1e-6)
    assert rebuilt.Ip == pytest.approx(original.Ip, rel=1e-6)


@needs_ross
def test_the_translated_project_can_be_drawn(client):
    """What the route gives back has to serve as input to /build_rotor."""
    original = _original_rotor()
    with tempfile.TemporaryDirectory() as folder:
        handle = os.path.join(folder, "rotor.toml")
        original.save(handle)
        content = open(handle, encoding="utf-8").read()

    project = client.post(
        "/load_ross_file", json={"content": content}, headers=AUTH
    ).json["projectData"]
    desenho = client.post("/build_rotor", json={"project": project}, headers=AUTH)

    assert desenho.status_code == 200, desenho.json
    assert desenho.json["mass"] == pytest.approx(original.m, rel=1e-6)


def test_a_file_that_is_neither_toml_nor_json_is_refused(client):
    """Before, the second error arrived alone, as though it were the only problem."""
    response = client.post(
        "/load_ross_file", json={"content": "this is not a rotor {{{"}, headers=AUTH
    )
    assert response.status_code == 400
    assert "TOML" in response.json["message"] or "JSON" in response.json["message"]


def test_a_failure_is_no_longer_answered_with_200(client):
    """The behaviour change of slice 4, pinned by a test.

    The old route did `return jsonify({"status": "error", ...})` with no code,
    that is, **200**. A client looking only at the HTTP code would conclude it
    had worked."""
    response = client.post("/load_ross_file", json={"content": "[["}, headers=AUTH)
    assert response.status_code != 200
    assert response.json["status"] == "error"


def test_a_body_without_content_is_refused_by_name(client):
    response = client.post("/load_ross_file", json={}, headers=AUTH)
    assert response.status_code == 400
    assert "content" in response.json["message"]


@needs_ross
def test_a_json_rotor_file_is_read_too(client):
    """As versoes antigas do ROSS gravavam em JSON; a alternativa continua viva."""
    original = _original_rotor()
    with tempfile.TemporaryDirectory() as folder:
        handle = os.path.join(folder, "rotor.json")
        original.save(handle)
        content = open(handle, encoding="utf-8").read()

    # only worth doing if ROSS really wrote JSON at this path
    try:
        json.loads(content)
    except ValueError:
        pytest.skip("this version of ROSS writes .json in TOML")

    response = client.post("/load_ross_file", json={"content": content}, headers=AUTH)
    assert response.status_code == 200, response.json
    assert response.json["projectData"]["shafts"]
