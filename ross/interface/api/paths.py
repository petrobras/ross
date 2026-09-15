# -*- coding: utf-8 -*-
"""Where the files live, in development and inside the executable."""

import os

import plotly

HOST = "127.0.0.1"
PORT = 5001


def frontend_dir():
    """The frontend folder: package data next to this package, frozen or not.

    Installed, it is `ross/interface/frontend` in site-packages; frozen, the
    same path under the bundle, where PyInstaller collects ROSS's data.
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
    )


FRONTEND_DIR = frontend_dir()

# The plotly.js served is the one inside the installed plotly.py. That way the
# interface draws with exactly the version a fig.show() in a script would use
# -- the two ends cannot drift apart on a package upgrade.
PLOTLY_BUNDLE = os.path.join(
    os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"
)
