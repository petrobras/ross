# -*- coding: utf-8 -*-
"""The page and the process lifetime."""

import os
import threading
import time

from flask import Blueprint, Response, jsonify, send_file

from .paths import FRONTEND_DIR
from .security import SESSION_TOKEN

system = Blueprint("system", __name__)


@system.route("/")
def index():
    """Serve index.html with this run's token injected."""
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as handle:
        html = handle.read()
    script = '<script>window.ROSS_TOKEN = "%s";</script>' % SESSION_TOKEN
    if "</head>" in html:
        html = html.replace("</head>", "    %s\n</head>" % script, 1)
    else:
        html = script + html
    response = Response(html, mimetype="text/html")
    response.headers["Cache-Control"] = "no-store"
    return response


@system.route("/lib/plotly.min.js")
def plotly_bundle():
    """Serve the plotly.js that shipped inside the installed plotly.py."""
    import plotly

    bundle = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    if not os.path.isfile(bundle):
        return jsonify({"status": "error", "message": "plotly.min.js was not found."}), 404
    return send_file(bundle, mimetype="text/javascript")


@system.route("/api/health")
def health():
    """Answer that this process is the new interface, not the Streamlit one."""
    return jsonify({"status": "ok", "application": "ross-digital-twin"})


@system.route("/shutdown", methods=["POST"])
def shutdown():
    """Stop the server. Protected by the token, like every other route."""

    def stop():
        time.sleep(0.3)
        os._exit(0)

    threading.Thread(target=stop, daemon=True).start()
    return jsonify({"status": "success"})
