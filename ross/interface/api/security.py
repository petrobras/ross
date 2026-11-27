# -*- coding: utf-8 -*-
"""Who may talk to this application.

The interface is served by Flask itself and receives, injected into
`index.html`, a token generated on every run. Every route requires that token
in the `X-ROSS-Token` header. With no open CORS, a page from another origin
cannot read it.

**The list here is of the public routes, not the protected ones.** A
hand-written set of protected routes would have to be remembered on every new
route, and forgetting to add one would leave that route open with nothing to
say so. This way round, forgetting leaves the route *closed* -- which fails
in your face instead of silently. `tests/test_security.py` walks Flask's route map
and requires every non-public route to refuse a caller with no token.
"""

import secrets

from flask import jsonify, request

# Generated on every run: closing the interface invalidates the token.
SESSION_TOKEN = secrets.token_urlsafe(32)

# What may be fetched without a token. `static` covers the frontend files
# (css, js, fonts), which Flask serves with static_url_path="".
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
