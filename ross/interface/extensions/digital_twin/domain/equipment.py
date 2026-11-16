# -*- coding: utf-8 -*-
"""Equipment tags, their rotor models, and which models run without Streamlit.

The tag list is `utils/settings.json` of the Digital Twin app. A model is
available here only when its module does not import Streamlit: those models
draw widgets and read `st.session_state` inside `get_pi_data` and
`create_model`, and cannot run behind a JSON route. The check reads the
source with `ast`, so a model becomes available the day it stops importing
Streamlit.
"""

import ast
import importlib
import json
import os
import re

from .dt_app import APP_DIR, ensure_on_path

MODEL_NAME = re.compile(r"[A-Za-z0-9_]+")


def load_settings():
    """Return `{tag: model}` from settings.json."""
    path = os.path.join(APP_DIR, "utils", "settings.json")
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {tag: entry["rotor_model"] for tag, entry in raw.items()}


def catalog():
    """Group the tags by platform, the prefix before the first dash."""
    platforms = {}
    for tag, model in load_settings().items():
        available, reason = model_status(model)
        platforms.setdefault(tag.split("-")[0], []).append(
            {"tag": tag, "model": model, "available": available, "reason": reason}
        )
    return [
        {"name": name, "equipment": sorted(items, key=lambda item: item["tag"])}
        for name, items in sorted(platforms.items())
    ]


def model_status(model):
    """Tell whether a model can run here, and why not when it cannot."""
    path = _model_path(model)
    if path is None:
        return False, "Model file not found."
    with open(path, encoding="utf-8", errors="replace") as handle:
        tree = ast.parse(handle.read(), filename=path)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        if any(name.split(".")[0] == "streamlit" for name in names):
            return False, "The model still depends on Streamlit."
    return True, ""


def model_for(tag):
    """Return the model name of a tag, refusing tags outside settings.json."""
    settings = load_settings()
    if tag not in settings:
        raise ValueError("Invalid equipment: %s" % tag)
    model = settings[tag]
    available, reason = model_status(model)
    if not available:
        raise ValueError("%s: %s" % (tag, reason))
    return model


def import_model(model):
    """Import `models_files.<model>.<model>` for a model listed in settings.json."""
    if model not in set(load_settings().values()) or not MODEL_NAME.fullmatch(model):
        raise ValueError("Invalid rotor model: %s" % model)
    ensure_on_path()
    return importlib.import_module("models_files.%s.%s" % (model, model))


def _model_path(model):
    if not MODEL_NAME.fullmatch(model or ""):
        return None
    path = os.path.join(APP_DIR, "models_files", model, model + ".py")
    return path if os.path.isfile(path) else None
