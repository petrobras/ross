# -*- coding: utf-8 -*-
"""Where the Streamlit Digital Twin app lives, and how its modules are reached.

The calculators and rotor models stay in `a17404-digital-twin-app/app` until
each one is brought across. `DT_APP_DIR` points somewhere else when needed.
"""

import os
import sys

APP_DIR = os.environ.get("DT_APP_DIR") or os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "a17404-digital-twin-app",
        "app",
    )
)


def ensure_on_path():
    """Make `brbalance_calc` and `models_files` importable."""
    if not os.path.isdir(APP_DIR):
        raise RuntimeError("Digital Twin app not found at %s. Set DT_APP_DIR." % APP_DIR)
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    return APP_DIR
