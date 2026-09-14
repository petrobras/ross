# -*- coding: utf-8 -*-
"""A lean payload and an honest cache (BE-06 and BE-07).

Two defects, both invisible on screen:

* the element cache was emptied on every assembly, which made switching between
  rotors in the Hub rebuild everything -- exactly the case the cache existed to
  cover;
* the analysis cache key was the whole payload, and the payload carried
  `savedAnalyses`, with the already-rendered Plotly figures. Saving a chart
  invalidated the cache of every analysis that followed, and every slider move
  sent that bundle back up to the server."""

import json
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from domain.cache import (
    ANALYSIS_CACHE,
    BoundedCache,
    ELEMENT_CACHE,
    STRUCTURAL_KEYS,
    rotor_key,
    spec_key,
    structural_project,
)
from domain.element_registry import categories
from waiting import answer_for

# A rotor ROSS really accepts: three shafts (nodes 0 to 3) and bearings at both
# ends. Without the bearings, run_static refuses -- which is how the first
# version of this file failed.
PROJECT_REQUEST = {
    "name": "Compressor A",
    "uid": "rotor_123",
    "materials": [{"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}],
    "shafts": [
        {"L": "100", "odl": "50", "idl": "0", "material": "Steel"} for _ in range(3)
    ],
    "bearings": [
        {"element_type": "BASIC", "n": "0", "kxx": "1e6", "cxx": "1e3"},
        {"element_type": "BASIC", "n": "3", "kxx": "1e6", "cxx": "1e3"},
    ],
    "disks": [],
    "gears": [],
    "seals": [],
    "couplings": [],
    "pointmasses": [],
}


def _with_charts(project, how_many=3):
    """The same project, with saved analyses -- whole Plotly figures."""
    copy_of = json.loads(json.dumps(project))
    copy_of["savedAnalyses"] = [
        {
            "title": "Campbell %d" % i,
            "type": "campbell",
            "params": {"speed_max": "4000"},
            "data": [{"x": list(range(200)), "y": list(range(200))}],
            "layout": {"title": "x" * 500},
        }
        for i in range(how_many)
    ]
    return copy_of


# --- BE-07: what goes into the key -------------------------------------------

SPEC = {"speed_min": 0.0, "speed_max": 418.9, "steps": 50}


def test_saved_charts_do_not_change_the_cache_key():
    """Saving a chart must not cool the cache of the other analyses."""
    without_charts_key = spec_key(PROJECT_REQUEST, "", "campbell", SPEC)
    with_charts_key = spec_key(_with_charts(PROJECT_REQUEST), "", "campbell", SPEC)
    assert without_charts_key == with_charts_key


def test_renaming_the_rotor_does_not_change_the_cache_key():
    """The name does not go into the computation, so it must not go into the key."""
    renamed = dict(PROJECT_REQUEST, name="Compressor B", uid="rotor_999")
    assert rotor_key(PROJECT_REQUEST) == rotor_key(renamed)


def test_a_real_change_does_change_the_cache_key():
    """Control: without this, the two tests above would pass with a fixed key."""
    other = json.loads(json.dumps(PROJECT_REQUEST))
    other["shafts"][0]["L"] = "200"
    assert rotor_key(PROJECT_REQUEST) != rotor_key(other)
    assert spec_key(PROJECT_REQUEST, "", "campbell", SPEC) != spec_key(
        other, "", "campbell", SPEC
    )


def test_the_conversion_is_part_of_the_key():
    """4 DoF and 6 DoF are different rotors: they cannot share a result."""
    assert rotor_key(PROJECT_REQUEST, "") != rotor_key(PROJECT_REQUEST, "4dof")
    assert spec_key(PROJECT_REQUEST, "", "modes", SPEC) != spec_key(
        PROJECT_REQUEST, "torsional", "modes", SPEC
    )


def test_a_different_spec_changes_the_key():
    """Control: the key has to react to the spec, not only to the rotor."""
    assert spec_key(PROJECT_REQUEST, "", "campbell", SPEC) != spec_key(
        PROJECT_REQUEST, "", "campbell", dict(SPEC, speed_max=500.0)
    )


def test_the_analysis_type_is_part_of_the_key():
    """Two different computations with the same spec must not collide."""
    assert spec_key(PROJECT_REQUEST, "", "campbell", SPEC) != spec_key(
        PROJECT_REQUEST, "", "ucs", SPEC
    )


# Plot parameters staying out of the key stopped being a property of a filter
# and became a property of the runners: the spec is what compute receives.
# tests/test_runners.py checks that on all twelve, one by one.


# --- allow list of the structural keys ---------------------------------------


def test_every_element_category_is_structural():
    """A new category in the registry joins the cache key on its own.

    If it were left out, two rotors differing only in that category would share a
    result -- one's chart would show up for the other."""
    for category in categories():
        assert category in STRUCTURAL_KEYS


def test_multirotor_fields_are_structural():
    for key in ("isMultiRotor", "driving_rotor", "driven_rotor", "multi_params"):
        assert key in STRUCTURAL_KEYS


def test_structural_project_drops_the_screen_only_fields():
    lean = structural_project(_with_charts(PROJECT_REQUEST))
    assert "savedAnalyses" not in lean
    assert "name" not in lean
    assert "uid" not in lean
    assert lean["shafts"] == PROJECT_REQUEST["shafts"]


def test_structural_project_reaches_into_the_two_rotors_of_a_multirotor():
    multi = {
        "isMultiRotor": True,
        "name": "MR",
        "driving_rotor": _with_charts(PROJECT_REQUEST),
        "driven_rotor": _with_charts(PROJECT_REQUEST),
        "multi_params": {"coupled_nodes": "2, 0"},
    }
    lean = structural_project(multi)
    assert "savedAnalyses" not in lean["driving_rotor"]
    assert "savedAnalyses" not in lean["driven_rotor"]
    assert lean["multi_params"]["coupled_nodes"] == "2, 0"


def test_structural_project_survives_garbage():
    assert structural_project(None) == {}
    assert structural_project("not a project") == {}
    assert structural_project({}) == {}


# --- BE-06: the cache with a bound and a lock --------------------------------


def test_the_cache_discards_the_least_recently_used():
    cache = BoundedCache(2, "test")
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # 'a' becomes the most recent again
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_the_cache_never_grows_past_its_limit():
    cache = BoundedCache(5, "test")
    for i in range(500):
        cache.put(i, i)
    assert len(cache) == 5


def test_get_or_create_builds_only_once():
    cache = BoundedCache(10, "test")
    calls = []

    def factory():
        calls.append(1)
        return "value"

    assert cache.get_or_create("k", factory) == "value"
    assert cache.get_or_create("k", factory) == "value"
    assert len(calls) == 1


def test_a_cached_none_is_not_mistaken_for_a_miss():
    """Without a sentinel, a None value would make the factory run on every lookup."""
    cache = BoundedCache(10, "test")
    calls = []
    cache.get_or_create("k", lambda: calls.append(1) or None)
    cache.get_or_create("k", lambda: calls.append(1) or None)
    assert len(calls) == 1


def test_the_cache_survives_many_threads():
    """Flask serves in threads; the two old dictionaries had no lock.

    Without the lock, eviction by bound runs while another thread writes -- the
    OrderedDict corrupts or raises in the middle of a request."""
    cache = BoundedCache(20, "test")
    errors = []

    def hammer(seed):
        try:
            for i in range(2000):
                key = (seed * 7 + i) % 60
                cache.get_or_create(key, lambda c=key: c * 2)
                cache.get(key)
                if i % 100 == 0:
                    cache.stats()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(cache) <= 20


def test_the_shipped_caches_are_bounded():
    """Both caches in the process have a bound -- neither grows without end."""
    for cache in (ELEMENT_CACHE, ANALYSIS_CACHE):
        assert cache.maxsize > 0
        assert len(cache) <= cache.maxsize


# --- the two premises that make sharing ROSS objects safe --------------------
#
# The cache keeps already-built elements and hands them to different rotors.
# That is only safe because:
#   1. Rotor.__init__ copies the shaft elements before touching them;
#   2. for the others it only calls set_tag, which respects a tag already set.
# If a future ROSS version changes either of the two, the tests below fail --
# and not a wrong chart three weeks later.

try:
    import ross  # noqa: F401

    HAS_ROSS = True
except Exception:  # pragma: no cover
    HAS_ROSS = False

needs_ross = pytest.mark.skipif(not HAS_ROSS, reason="requires ROSS installed")


@needs_ross
def test_switching_rotors_no_longer_empties_the_element_cache():
    """BE-06: the old pruning deleted everything that was not from the last rotor.

    Building A, building B and going back to A rebuilt A entirely. Now the two
    live together, which is the normal use of the Hub."""
    from domain.rotor_builder import build_rotor_from_ui

    def project(length):
        return {
            "materials": [
                {"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}
            ],
            "shafts": [
                {"L": length, "odl": "50", "idl": "0", "material": "Steel"}
                for _ in range(3)
            ],
            "bearings": [
                {"n": "0", "kxx": "1e6", "cxx": "1e3"},
                {"n": "3", "kxx": "1e6", "cxx": "1e3"},
            ],
        }

    ELEMENT_CACHE.clear()
    build_rotor_from_ui(project("100"))
    build_rotor_from_ui(project("200"))
    after_b = ELEMENT_CACHE.stats()

    build_rotor_from_ui(project("100"))
    back_in_a = ELEMENT_CACHE.stats()

    # Going back to A must build nothing new: everything came from the cache.
    assert back_in_a["misses"] == after_b["misses"]
    assert back_in_a["hits"] > after_b["hits"]


# --- the routes with the new envelope ----------------------------------------


@pytest.fixture
def client():
    from app import app as application

    application.config["TESTING"] = True
    with application.test_client() as client:
        yield client


def _auth():
    from api.security import SESSION_TOKEN

    return {"X-ROSS-Token": SESSION_TOKEN}


@needs_ross
def test_saved_charts_in_the_payload_hit_the_same_cache_entry(client):
    """The end-to-end proof of BE-07, by counting cache hits."""
    ANALYSIS_CACHE.clear()
    body = {
        "analysis_type": "static",
        "params": {},
        "conversion_type": "",
        "project": PROJECT_REQUEST,
    }

    first = answer_for(
        client, client.post("/run_analysis", json=body, headers=_auth()), _auth()
    )
    assert first.status_code == 200, first.json
    hits = ANALYSIS_CACHE.stats()["hits"]

    # the same rotor, now with three charts saved in the project: before, this
    # changed the key and recomputed the whole analysis
    with_charts = dict(body, project=_with_charts(PROJECT_REQUEST))
    segunda = answer_for(
        client, client.post("/run_analysis", json=with_charts, headers=_auth()), _auth()
    )
    assert segunda.status_code == 200, segunda.json
    assert ANALYSIS_CACHE.stats()["hits"] == hits + 1
