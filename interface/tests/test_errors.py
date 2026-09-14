# -*- coding: utf-8 -*-
"""Whose fault an error is, and what reaches the screen because of that.

Before, **every error became a 400**, including a defect of ours: an internal
`KeyError` reached the user as though they had typed something wrong, with the
message `'odl'` and nothing else. And the traceback went to `stderr`, which in
a packaged executable with no console does not exist.

The split is by type: a `ValueError` is the user's -- it is how ROSS refuses a
model ("Add at least one Shaft!") -- and keeps its message; any other exception
is ours, becomes a 500 and goes whole into the log file.

The handling is central. A route with a `try/except` of its own would bypass
it, and the guard below refuses that by reading the tree."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import structure
from api import create_app
from api.security import SESSION_TOKEN

APP = create_app()
AUTH = {"X-ROSS-Token": SESSION_TOKEN}


def _app_with_a_raising_route(exception_class):
    app = create_app()
    app.config["TESTING"] = True

    @app.route("/_explode")
    def _explode():
        raise exception_class

    return app


@pytest.fixture
def client():
    APP.config["TESTING"] = True
    with APP.test_client() as client:
        yield client


def test_a_value_error_is_the_users_fault_and_keeps_its_message():
    """This is how ROSS refuses a model: the message helps whoever is at the screen."""
    app = _app_with_a_raising_route(ValueError("Add at least one Shaft!"))
    with app.test_client() as client:
        response = client.get("/_explode", headers=AUTH)
    assert response.status_code == 400
    assert response.json["message"] == "Add at least one Shaft!"
    assert response.json["status"] == "error"


def test_any_other_exception_is_a_server_fault():
    """Before, everything became a 400 -- a defect of ours disguised as bad input.

    And the message was the str() of the exception: a KeyError reached the screen
    as 'odl', with no context at all."""
    app = _app_with_a_raising_route(KeyError("odl"))
    with app.test_client() as client:
        response = client.get("/_explode", headers=AUTH)
    assert response.status_code == 500
    assert "KeyError" in response.json["message"]
    assert "odl" in response.json["message"]


def test_an_unknown_route_keeps_its_own_status(client):
    """A 404 must not become a 500 through the generic handler."""
    response = client.get("/route-that-does-not-exist", headers=AUTH)
    assert response.status_code == 404
    assert response.json["status"] == "error"


def test_no_route_carries_its_own_try_except():
    """The handling is central; a generic catch in a route would bypass it.

    By the tree, and not by the text: the first version of this test accused the
    very comment that explained why the catch had been removed."""
    for path, tree in structure.modules(ROOT, "api", "*.py"):
        if path.endswith("errors.py"):
            continue
        assert not structure.catches_everything(tree), (
            "%s catches Exception on its own" % path
        )
