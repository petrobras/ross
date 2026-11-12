# -*- coding: utf-8 -*-
"""Who may talk to this application.

The page receives a token generated on every run. Every route except the page
itself and the static files requires that token in the `X-ROSS-Token` header.
The public list is the short one: a new route is closed unless someone adds
it here.
"""

import secrets

from flask import jsonify, request

SESSION_TOKEN = secrets.token_urlsafe(32)

PUBLIC_ENDPOINTS = frozenset({"static", "system.index", "system.plotly_bundle"})


def register_guard(app, host, port):
    """Install the token check ahead of every route."""

    @app.before_request
    def require_session_token():
        if request.endpoint in PUBLIC_ENDPOINTS:
            return None
        if request.headers.get("X-ROSS-Token") == SESSION_TOKEN:
            return None
        return jsonify(
            {
                "status": "error",
                "message": (
                    "Invalid session. Open the interface at "
                    "http://%s:%s/ and reload the page." % (host, port)
                ),
            }
        ), 403
