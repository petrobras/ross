# -*- coding: utf-8 -*-
"""The session token: who may call the routes, and who needs no token.

The interface runs on `127.0.0.1`, and a page from another origin in the same
browser reaches that address. The token is drawn at each run, injected into
`index.html` by the server and required on every route that is not public.

The list that matters here is the one of **public** routes, not of protected
ones: the page and the plotting library have to go out with no token, because
that is where the token comes from. Any new route is born protected by
omission, and the guard walks Flask's route map -- not a hand-kept list, which
would grow stale in silence."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from app import app
from api import create_app
from api.security import PUBLIC_ENDPOINTS, SESSION_TOKEN

APP = create_app()


def _private_routes():
    """Every route not declared as public."""
    return sorted(
        (rule.rule, sorted(rule.methods & {"GET", "POST"}))
        for rule in APP.url_map.iter_rules()
        if rule.endpoint not in PUBLIC_ENDPOINTS
    )


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.mark.parametrize(
    "route", ["/build_rotor", "/run_analysis", "/load_ross_file", "/shutdown"]
)
def test_protected_routes_reject_missing_token(client, route):
    """With no token, no API route answers -- including /shutdown."""
    assert client.post(route, json={}).status_code == 403


def test_protected_routes_reject_wrong_token(client):
    response = client.post("/build_rotor", json={}, headers={"X-ROSS-Token": "errado"})
    assert response.status_code == 403


def test_index_injects_token():
    """The page served by Flask carries the session token."""
    with app.test_client() as c:
        html = c.get("/").get_data(as_text=True)
    assert SESSION_TOKEN in html
    assert "window.ROSS_TOKEN" in html


def test_export_route_requires_the_session_token(client):
    """The route handles project data: it joins the protected list like the others."""
    response = client.post("/api/export/python", json={"project": {}})
    assert response.status_code == 403


def test_there_is_at_least_one_private_route_to_check():
    """Control: without this, the battery below would pass empty."""
    assert len(_private_routes()) >= 5


@pytest.mark.parametrize("route, methods", _private_routes())
def test_every_private_route_refuses_a_request_without_the_token(
    route, methods, client
):
    """Walks Flask's route map, not a hand-kept list.

    Until slice 4 there was a PROTECTED_ROUTES set kept by hand. Adding a route and
    forgetting to add it there left the route **open**, and nothing warned. Turned
    around, the oversight leaves the route closed -- which fails in your face. This
    test covers any new route automatically."""
    for method in methods:
        response = getattr(client, method.lower())(route, json={})
        assert response.status_code == 403, "%s %s went through with no token" % (
            method,
            route,
        )


def test_the_public_routes_are_reachable_without_a_token(client):
    """The page and the plotting library: they are where the token comes from."""
    assert client.get("/").status_code == 200
    assert client.get("/lib/plotly.min.js").status_code == 200


def test_the_page_carries_the_token(client):
    assert SESSION_TOKEN in client.get("/").get_data(as_text=True)
