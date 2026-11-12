# -*- coding: utf-8 -*-
"""Step 3 analyses: Campbell, unbalance response, clearances and stability."""

import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DT_SESSIONS_DIR", tempfile.mkdtemp(prefix="dt-sessions-"))

import numpy as np  # noqa: E402
import ross as rs  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402
from ross.units import Q_  # noqa: E402

from api import SESSION_TOKEN, create_app  # noqa: E402
from domain import session  # noqa: E402
from services import analysis  # noqa: E402
from services.analysis.base import residual_unbalance  # noqa: E402
from services.digital_twin import create_model, load_process_data  # noqa: E402

HEADERS = {"X-ROSS-Token": SESSION_TOKEN}
LMEST = "LMEST-2-DISCOS"
PI_TIME = "2026/11/11 00:00:00"


@pytest.fixture(scope="module")
def lmest():
    current = session.create()
    load_process_data(current.id, LMEST, PI_TIME, "clockwise")
    create_model(current.id)
    return current


@pytest.fixture
def example():
    current = session.create()
    current.rotor = rs.rotor_example()
    return current


def test_lmest_matrices_accept_the_ross_3_signature(lmest):
    rotor = lmest.rotor
    assert rotor.M(None, 0.0, synchronous=True) is rotor.mat_rotor_M
    assert rotor.K(0.0, 0.0) is rotor.mat_rotor_K


def test_analyses_wait_for_the_model():
    current = session.create()
    with pytest.raises(ValueError, match="modelo"):
        analysis.run(current.id, "campbell", {})


def test_unknown_analysis_is_refused(lmest):
    with pytest.raises(ValueError, match="desconhecida"):
        analysis.run(lmest.id, "nada", {})


def test_residual_unbalance_is_api_684():
    # Umax = 6350 W / N [g.mm], N capped at 25000 RPM (API 684, 6350 W/N).
    assert_allclose(residual_unbalance(100.0, 5000.0), 127.0)
    assert_allclose(residual_unbalance(100.0, 50000.0), 6350.0 * 100.0 / 25000.0)


def test_campbell_and_the_mode_of_a_point(lmest):
    out = analysis.run(lmest.id, "campbell", {"speed_min": 0, "speed_max": 3000, "points": 6, "harmonics": "1, 2"})
    names = {trace.get("name") for trace in out["figures"][0]["data"]}
    assert {"1x speed", "2x speed"} <= names

    result = lmest.campbell
    point = {
        "x": Q_(result.speed_range[2], "rad/s").to("RPM").m,
        "y": Q_(result.wd[2][0], "rad/s").to("RPM").m,
    }
    mode = analysis.campbell_mode_shape(lmest.id, point)
    assert mode["figure"]["data"]


@pytest.mark.parametrize("bad", [{"speed_min": 100, "speed_max": 10}, {"points": 1}, {"harmonics": "0"}])
def test_campbell_refuses_bad_fields(lmest, bad):
    params = {"speed_min": 0, "speed_max": 3000, "points": 6, "harmonics": "1"}
    params.update(bad)
    with pytest.raises(ValueError):
        analysis.run(lmest.id, "campbell", params)


@pytest.mark.parametrize("mcs, percent", [(600, 100), (1150, 50)])
def test_unbalance_spreads_the_api_unbalance_over_the_disks(lmest, mcs, percent):
    out = analysis.run(
        lmest.id,
        "unbalance",
        {
            "freq_min": 315, "freq_max": 1150, "points": 8, "mcs": mcs, "percent": percent,
            "probe1_node": 5, "probe1_angle": 0, "probe2_node": 35, "probe2_angle": 90,
            "unbalances": [{"node": 14, "magnitude": 30, "phase": 45}],
        },
    )
    assert len(out["figures"]) == 2
    rows = out["tables"][0]["rows"]
    residual = [row for row in rows if row[3] == "Residual (disco)"]
    expected = 6350.0 * lmest.rotor.m / mcs * percent / 100.0
    assert_allclose(sum(row[1] for row in residual), expected, rtol=1e-12)
    masses = {int(d.n): d.m for d in lmest.rotor.disk_elements}
    assert_allclose([row[1] / expected for row in residual], [masses[row[0]] / sum(masses.values()) for row in residual])
    assert [14, 30.0, 45.0, "Adicionado"] in rows


def test_unbalance_mcs_must_be_inside_the_range(lmest):
    with pytest.raises(ValueError, match="intervalo"):
        analysis.run(lmest.id, "unbalance", {"freq_min": 315, "freq_max": 1150, "points": 8, "mcs": 2000,
                                             "probe1_node": 5, "probe2_node": 35})


@pytest.mark.parametrize("speed", [1000, 3000, 9000])
def test_clearance_scales_the_largest_amplitude_to_a1(example, speed):
    out = analysis.run(example.id, "clearance", {"speed": speed})
    amplitudes = [row[1] for row in out["tables"][0]["rows"]]
    # API 617 amplitude limit, A1 = 25.4 sqrt(12000 / N) um pk-pk.
    assert_allclose(max(amplitudes), 25.4 * math.sqrt(12000.0 / speed), rtol=1e-12)
    assert [row[0] for row in out["tables"][0]["rows"]] == [int(b.n) for b in example.rotor.bearing_elements]


def test_clearance_compares_with_the_bearing_clearance(example):
    for bearing in example.rotor.bearing_elements:
        bearing.radial_clearance = 100e-6
    row = analysis.run(example.id, "clearance", {"speed": 3000})["tables"][0]["rows"][0]
    _, amplitude, clearance, used, used_75 = row
    assert_allclose(clearance, 100.0)
    assert_allclose(used, amplitude / 100.0 * 100.0)
    assert_allclose(used_75, amplitude / 75.0 * 100.0)


@pytest.mark.parametrize("speed", [0, 1000, 5000])
def test_stability_matches_a_direct_modal_run(example, speed):
    out = analysis.run(example.id, "stability", {"speed": speed})
    with_seals = out["tables"][1]
    modal = rs.utils.convert_6dof_to_4dof(example.rotor).run_modal(Q_(speed, "RPM").to("rad/s").m)
    assert_allclose([row[2] for row in with_seals["rows"]], Q_(modal.wd, "rad/s").to("Hz").m)
    assert_allclose([row[3] for row in with_seals["rows"]], modal.log_dec)


def test_routes_run_a_card_and_refuse_without_token(lmest):
    client = create_app().test_client()
    body = {"session": lmest.id, "type": "stability", "params": {"speed": 500}}
    assert client.post("/api/dt/analysis", json=body).status_code == 403
    answer = client.post("/api/dt/analysis", json=body, headers=HEADERS)
    assert answer.status_code == 200
    assert len(answer.get_json()["tables"]) == 2
    refused = client.post("/api/dt/analysis", json=dict(body, params={"speed": "abc"}), headers=HEADERS)
    assert refused.status_code == 400
    assert np.isfinite(answer.get_json()["tables"][0]["rows"][0][2])
