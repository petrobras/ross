# -*- coding: utf-8 -*-
"""Where the files live, and which address the server binds."""

import os

# Beside the original interface (port 5001) and the Streamlit app (port 8501).
HOST = "127.0.0.1"
PORT = 5002


def frontend_dir():
    """The frontend folder next to this package."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "frontend")


FRONTEND_DIR = frontend_dir()
