# -*- coding: utf-8 -*-
"""How the application is assembled, and the boundary that keeps it exercisable.

`create_app()` is a function, and not a module-level `app`, so that each test
builds its own without inheriting configuration from another.

The guard that matters most is the boundary one: **the lower layers do not know
about Flask**. If `domain/` or `services/` imported `flask`, there would be no
way to exercise the rotor assembly or an analysis without starting a server --
and most of this suite would stop existing in the form it exists in."""

import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import structure
from api import BLUEPRINTS, create_app

APP = create_app()


@pytest.mark.parametrize("folder", ["domain", "services"])
def test_the_lower_layers_do_not_know_about_flask(folder):
    """If the domain imported Flask, there would be no exercising it without a server."""
    import ast

    for path, tree in structure.modules(ROOT, folder, "**", "*.py", recursive=True):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                origin = getattr(node, "module", None) or ""
                names = [a.name for a in node.names]
                assert "flask" not in (origin + " " + " ".join(names)).lower(), (
                    "%s imports Flask" % path
                )


def test_the_entry_point_declares_no_routes():
    """app.py starts the server, opens the browser and picks a mode. Nothing else.

    The size limit counts **statements**, not lines, and that is a correction.
    Counted in lines it stood at "under 60", and slice 6a took the file to 74 by
    adding a mode -- of which fifteen lines are the comment explaining why the
    mode has to be decided before the application is built. A limit that counts
    prose punishes exactly the thing this project wants more of, and the way it
    fails is by being raised: a ceiling that only ever moves up measures nothing
    after the second time somebody moves it.

    Counted in statements the file has not grown much in three slices, which is
    the property the limit was always trying to express: the entry point wires
    things together and computes nothing.
    """
    import ast

    with io.open(os.path.join(ROOT, "app.py"), encoding="utf-8") as handle:
        source = handle.read()
    assert "@app.route" not in source

    statements = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.stmt)]
    assert len(statements) < 40, (
        "app.py is doing %d things; it is supposed to start the server, open the "
        "browser and dispatch a mode" % len(statements)
    )


def test_every_blueprint_is_registered():
    """A blueprint written and not registered is a route that disappears with no warning."""
    registered = {
        rule.endpoint.split(".")[0]
        for rule in APP.url_map.iter_rules()
        if "." in rule.endpoint
    }
    assert {bp.name for bp in BLUEPRINTS} == registered


def test_create_app_builds_independent_applications():
    """Each test can build its own, without inheriting configuration from another."""
    first, segunda = create_app(), create_app()
    assert first is not segunda
    first.config["TESTING"] = True
    assert segunda.config["TESTING"] is False
