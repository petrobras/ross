# -*- coding: utf-8 -*-
"""Digital Twin sessions: a directory on disk and what is still in memory.

The directory follows the Streamlit app's layout (`database/`,
`results/session_files/`), because that is what `rossml` reads later. The
process data frame and the ROSS rotor stay in memory for this process; the
rotor is also saved as `<model>.toml`, the file the `.rossdt` carries.
"""

import os
import re
import secrets
from datetime import datetime

# Outside the package, which may be installed read-only in site-packages.
SESSIONS_ROOT = os.environ.get("DT_SESSIONS_DIR") or os.path.join(
    os.path.expanduser("~"), ".ross", "dt_sessions"
)

SESSION_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")

_OPEN = {}


class Session:
    """One user's work on one equipment."""

    def __init__(self, identifier, name):
        self.id = identifier
        self.name = name
        self.equipment = None
        self.model = None
        self.rotation = "clockwise"
        self.pi_time = None
        self.source = None
        self.settings = None
        self.process_data = None
        self.rotor = None
        self.components = None
        self.campbell = None
        self.measuring_nodes = []
        self.balancing_nodes = []
        self.sensors_plans = {}

    @property
    def directory(self):
        return os.path.join(SESSIONS_ROOT, self.id)

    @property
    def database(self):
        return os.path.join(self.directory, "database")

    @property
    def session_files(self):
        return os.path.join(self.directory, "results", "session_files")


def create():
    """Open a new session and its directory tree."""
    identifier = secrets.token_hex(8)
    session = Session(identifier, datetime.now().strftime("%Y-%m-%d_%H-%M"))
    for sub in ("database", os.path.join("results", "session_files")):
        os.makedirs(os.path.join(session.directory, sub), exist_ok=True)
    _OPEN[identifier] = session
    return session


def get(identifier):
    """Return an open session, refusing ids that are malformed or unknown."""
    if not isinstance(identifier, str) or not SESSION_ID.fullmatch(identifier):
        raise ValueError("Invalid session.")
    session = _OPEN.get(identifier)
    if session is None:
        raise ValueError("Session not found. Reload the page.")
    return session
