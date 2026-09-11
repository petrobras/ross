# -*- coding: utf-8 -*-
"""The transport layer: routes, and nothing beyond that.

Each blueprint reads the body through an envelope from `domain/requests.py`,
calls the domain or the services, and returns JSON. No route has a `try/except`:
error handling is central, in `api/errors.py`, and that is what decides between
400 and 500. No route computes anything: that lives in `domain/` and
`services/`.

Before slice 4 all of this was a 1190-line `app.py` with routes, rotor
assembly, the expression evaluator, the cache and ROSS file reading mixed
together.
"""

from flask import Flask

from .analysis import analysis
from .errors import register_handlers
from .export import export
from .paths import FRONTEND_DIR, HOST, PORT
from .rotor import rotor_api
from .schema import schema
from .security import SESSION_TOKEN, register_guard
from .system import system
from .waiting import answer_for
from .worker import worker_api

BLUEPRINTS = (system, schema, rotor_api, analysis, export, worker_api)


def create_app():
    """Build the application. A function, so tests can build their own."""
    app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

    register_handlers(app)
    register_guard(app, HOST, PORT)
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)

    return app


# `answer_for` is re-exported on purpose, and not only for convenience: it is
# how a caller of the job routes waits, and naming it here is what guarantees
# PyInstaller bundles it. `selftest.py` reaches it through a function-level
# import, and a module that only a function-level import mentions is exactly the
# kind of thing that works from source and is missing from the executable.
__all__ = [
    "BLUEPRINTS",
    "FRONTEND_DIR",
    "HOST",
    "PORT",
    "SESSION_TOKEN",
    "answer_for",
    "create_app",
]
