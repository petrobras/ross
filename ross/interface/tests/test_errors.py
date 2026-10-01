# -*- coding: utf-8 -*-
"""Whose fault an error is, and what reaches the screen because of that.

Before, **every error became a 400**, including a defect of ours: an internal
`KeyError` reached the user as though they had typed something wrong, with the
message `'odl'` and nothing else. And the traceback went to `stderr`, which in
a packaged executable with no console does not exist.

The split: a `ValueError` is the user's -- it is how ROSS refuses a model ("Add
at least one Shaft!") -- and keeps its message; so does anything ROSS raises on
purpose (its gear refusal is a `TypeError`); any other exception is ours,
becomes a 500 and goes whole into the log file.

The handling is central. A route with a `try/except` of its own would bypass
it, and the guard below refuses that by reading the tree."""

import dis
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ross.interface.tests import structure
from ross.interface.api import create_app
from ross.interface.api.security import SESSION_TOKEN

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


# --- what ROSS refuses on purpose -----------------------------------------------
#
# ROSS does not refuse only through ValueError: "Each rotor needs a GearElement
# in the coupled nodes!" is a TypeError. What separates a refusal from a crash
# is not the type but how it was raised -- a `raise` in ROSS's own code.


def _function_in(module, source):
    """A function whose frames report `module` as theirs, as if it lived there."""
    namespace = {"__name__": module}
    exec(compile(source, "<%s>" % module, "exec"), namespace)
    return namespace["run"]


def _answer(function):
    app = create_app()
    app.config["TESTING"] = True

    @app.route("/_explode")
    def _explode():
        function()

    with app.test_client() as client:
        return client.get("/_explode", headers=AUTH)


RAISES = "def run():\n    raise TypeError('Each rotor needs a GearElement!')\n"
BREAKS = "def run():\n    return None + 1\n"


def test_an_exception_ross_raises_on_purpose_is_a_refusal_with_its_own_words():
    response = _answer(_function_in("ross.multi_rotor.multi_rotor", RAISES))
    assert response.status_code == 400
    assert response.json["message"] == "Each rotor needs a GearElement!"


def test_a_crash_inside_ross_is_still_a_server_fault():
    """Same type, no `raise`: arithmetic on a None is a crash, not a refusal."""
    response = _answer(_function_in("ross.materials", BREAKS))
    assert response.status_code == 500
    assert "TypeError" in response.json["message"]


ASSERTS = "def run(value=0):\n    assert value > 0, 'value must be positive'\n"


def test_an_assert_failing_inside_ross_is_a_server_fault():
    """An `assert` is a `raise` in the bytecode, and an invariant broken."""
    response = _answer(_function_in("ross.rotor_assembly", ASSERTS))
    assert response.status_code == 500
    assert "AssertionError" in response.json["message"]


def test_the_assert_really_ends_in_a_raise():
    """Control on the test above: without the type check it would be a 400.

    If `assert` ever stops compiling to `RAISE_VARARGS`, the test above would
    pass for the wrong reason, and this one says so.
    """
    from ross.interface.api.errors import raised_by_ross

    run = _function_in("ross.rotor_assembly", ASSERTS)
    with pytest.raises(AssertionError) as failed:
        run()
    frame = failed.value.__traceback__
    while frame.tb_next is not None:
        frame = frame.tb_next
    code = frame.tb_frame.f_code.co_code
    assert code[frame.tb_lasti] == dis.opmap["RAISE_VARARGS"]
    assert not raised_by_ross(failed.value)


def test_a_raise_of_this_interface_is_not_one_of_rosss():
    """`ross.interface` sits inside the `ross` package, and it is ours."""
    response = _answer(_function_in("ross.interface.domain.fake", RAISES))
    assert response.status_code == 500


def test_a_raise_outside_ross_is_not_one_of_rosss():
    """Control: a `raise` alone decides nothing -- numpy's are not refusals."""
    response = _answer(_function_in("numpy.fake", RAISES))
    assert response.status_code == 500


def test_the_real_ross_still_refuses_that_way():
    """Measured on the ROSS installed, not only on stand-ins: the gear refusal
    is a `raise` in ross.multi_rotor, and a name that is not text is a crash in
    ross.materials. If ROSS changes either, this says so."""
    import ross as rs

    from ross.interface.api.errors import raised_by_ross

    steel = rs.materials.steel

    def rotor():
        shafts = [
            rs.ShaftElement(L=0.25, idl=0, odl=0.05, material=steel) for _ in range(2)
        ]
        bearings = [
            rs.BearingElement(n=0, kxx=1e6, cxx=0),
            rs.BearingElement(n=2, kxx=1e6, cxx=0),
        ]
        return rs.Rotor(shafts, bearing_elements=bearings)

    with pytest.raises(TypeError) as refused:
        rs.MultiRotor(rotor(), rotor(), coupled_nodes=(0, 0), gear_mesh_stiffness=1e8)
    assert raised_by_ross(refused.value)

    with pytest.raises(TypeError) as crashed:
        rs.Material(name=None, rho=7810, E=211e9, G_s=81.2e9)
    assert not raised_by_ross(crashed.value)


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
