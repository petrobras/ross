# -*- coding: utf-8 -*-
"""What the Digital Twin adds to the ROSS interface."""

import os

from ross.interface.api.extensions import Extension

from .api import BLUEPRINTS

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

EXTENSION = Extension(
    name="dt",
    static_dir=FRONTEND_DIR,
    blueprints=BLUEPRINTS,
    script="main.js",
    styles=("style.css",),
    home="home.html",
    hub="hub.html",
    analysis="analysis.html",
    screens="screens.html",
)
