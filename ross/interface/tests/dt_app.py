# -*- coding: utf-8 -*-
"""The interface with the Digital Twin extension only, for the test_dt_* files."""

import os

import pytest

from ross.interface.api import SESSION_TOKEN
from ross.interface.api import create_app as create_ross_app
from ross.interface.extensions.digital_twin.domain.dt_app import APP_DIR
from ross.interface.extensions.digital_twin.extension import EXTENSION

HEADERS = {"X-ROSS-Token": SESSION_TOKEN}


def create_app():
    """An application with this extension, whatever else is listed or installed."""
    return create_ross_app(extensions=[EXTENSION])


# The BRBalance calculator and the equipment models are still read from the
# Digital Twin app, a separate repository (see domain/dt_app.py). Where it is
# not checked out next to ROSS, or pointed at by DT_APP_DIR, the tests that
# call them cannot run.
requires_dt_app = pytest.mark.skipif(
    not os.path.isdir(APP_DIR),
    reason="Digital Twin app not found at %s (set DT_APP_DIR)" % APP_DIR,
)
