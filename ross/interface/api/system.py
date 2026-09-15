# -*- coding: utf-8 -*-
"""The page, the plotting library and shutting down."""

import os
import re
import threading
import time

from flask import Blueprint, Response, jsonify, request, send_file

from .paths import FRONTEND_DIR, PLOTLY_BUNDLE
from .security import SESSION_TOKEN

system = Blueprint("system", __name__)


@system.after_app_request
def no_cache_on_statics(response):
    """The frontend's JS and CSS are never cached by the browser.

    The index's version stamp settles the entry file, but with ES modules one
    file imports another **from inside the JS**, where there is no HTML to
    stamp. Running a new `main.js` against an old `core/state.js` would be worse
    than running everything old: half a fix. Here the interface is local, for a
    single user -- bandwidth costs nothing and doubt costs a lot.
    """
    if request.endpoint == "static":
        response.headers["Cache-Control"] = "no-store"
    return response


@system.route("/lib/plotly.min.js")
def plotly_bundle():
    """Serve the plotly.js that shipped inside the installed plotly.py."""
    return send_file(PLOTLY_BUNDLE, mimetype="text/javascript")


# A local reference in the index: `src="app.js"`, `href="style.css"`. What is
# absolute (`https:`, `//`, `/lib/...`) is left out; it does not live here.
LOCAL_REFERENCE = re.compile(r'(src|href)="(?!https?:|//|/)([^"?#]+)"')


def stamp_versions(html):
    """Put each local file's last-write time into its URL.

    The index always goes out `no-store`, so the stamp it carries is always the
    current one; a changed file gets a different URL and no cache can serve the
    previous one. Without this, updating the interface and still seeing the old
    version is a real possibility -- and a silent one, which is the worst kind.
    The `Cache-Control` of static files comes from Flask and depends on the
    installed version: on Flask 1.x the default is twelve hours, and even a
    `no-cache` leaves the decision to the browser.

    It applies to every local reference, not to a list of two files: a third
    file added to the index could not be remembered.
    """

    def stamp(found):
        attribute, path = found.group(1), found.group(2)
        handle = os.path.join(FRONTEND_DIR, *path.split("/"))
        if not os.path.isfile(handle):
            return found.group(0)
        return '%s="%s?v=%d"' % (attribute, path, os.path.getmtime(handle))

    return LOCAL_REFERENCE.sub(stamp, html)


@system.route("/")
def index():
    """Serve index.html with this session's token injected."""
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as handle:
        html = handle.read()

    html = stamp_versions(html)

    script = '<script>window.ROSS_TOKEN = "%s";</script>' % SESSION_TOKEN
    if "</head>" in html:
        html = html.replace("</head>", "    %s\n</head>" % script, 1)
    else:
        html = script + html

    response = Response(html, mimetype="text/html")
    # No caching: the token changes on every run, and a cached page would carry
    # the previous run's token -- which would only give a 403.
    response.headers["Cache-Control"] = "no-store"
    return response


@system.route("/shutdown", methods=["POST"])
def shutdown():
    """Shut the server down. Protected by the token, like every route."""

    def stop():
        time.sleep(0.3)
        os._exit(0)

    threading.Thread(target=stop, daemon=True).start()
    return jsonify({"status": "success"})
