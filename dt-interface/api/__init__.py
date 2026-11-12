# -*- coding: utf-8 -*-
"""Transport: routes, the session token and error handling.

A route reads the request, calls `domain` or `services`, and returns JSON.
No route computes a rotor, a balance or a cluster job.
"""

from flask import Flask

from .analysis import analysis
from .balance import balance
from .digital_twin import digital_twin
from .errors import register_handlers
from .paths import FRONTEND_DIR, HOST, PORT
from .security import SESSION_TOKEN, register_guard
from .system import system

BLUEPRINTS = (system, balance, digital_twin, analysis)


def create_app():
    """Build the application. A function, so tests can build their own."""
    app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
    register_handlers(app)
    register_guard(app, HOST, PORT)
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)
    return app


__all__ = [
    "BLUEPRINTS",
    "FRONTEND_DIR",
    "HOST",
    "PORT",
    "SESSION_TOKEN",
    "create_app",
]
