# -*- coding: utf-8 -*-
"""Equipment tags, their rotor models, and which models can run here.

The tag list is `utils/settings.json` of the Digital Twin app. The models
import `streamlit`, which `headless_streamlit` stands in for, so Streamlit is
not needed. A model is unavailable only when a package it imports at the top
of its module is not installed (`rossml` and `rossxl`, for the models with
metamodels). The check reads the source with `ast`, without importing it.
"""

import ast
import importlib
import importlib.util
import json
import os
import re
import sys

from . import headless_streamlit, model_settings
from .dt_app import APP_DIR, ensure_on_path

MODEL_NAME = re.compile(r"[A-Za-z0-9_]+")

# Provided by the stand-in, or modules of the Digital Twin app itself.
PROVIDED = {"streamlit", "safe_pickle", "models_files"}


def load_settings():
    """Return `{tag: model}` from settings.json."""
    path = os.path.join(APP_DIR, "utils", "settings.json")
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {tag: entry["rotor_model"] for tag, entry in raw.items()}


def catalog():
    """Group the tags by platform, the prefix before the first dash."""
    platforms = {}
    status = {}
    for tag, model in load_settings().items():
        if model not in status:
            status[model] = model_status(model)
        available, reason = status[model]
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
    missing = missing_packages(model)
    if missing:
        return False, "Missing packages: %s." % ", ".join(missing)
    return True, ""


def missing_packages(model):
    """Return the packages the model's folder imports that are not installed.

    Every `.py` of the folder is read, because the labyrinth seals are built
    by `create_labyrinth_seals.py` in a subprocess. Only module-level imports
    count: imports inside `try` (the models import `pandaspi` that way) are
    optional, and `services.digital_twin` says what is missing when the
    process data is asked of the PI server.
    """
    folder = os.path.dirname(_model_path(model))
    local = {name[:-3] for name in os.listdir(folder) if name.endswith(".py")}
    missing = set()
    for name in sorted(local):
        with open(
            os.path.join(folder, name + ".py"), encoding="utf-8", errors="replace"
        ) as handle:
            imported = _top_level_imports(handle.read())
        for package in imported - local - PROVIDED:
            if package in sys.stdlib_module_names:
                continue
            if importlib.util.find_spec(package) is None:
                missing.add(package)
    return sorted(missing)


def settings(model):
    """Return the additional settings the model declares (`model_settings.defaults`)."""
    _check_name(model)
    return model_settings.defaults(_source(model))


def analysis_parameters(model):
    """Return the model's `input_parameters.json`, or `None` when it has none.

    These are the hyperparameters of the balancing analysis: the search
    bounds of the Kalman factors, the limits of the resultant force, the
    number of points of the signal. They are tuned per machine and belong to
    the Digital Twin app, not to ROSS, so they are read from the model folder
    and never copied here. Without them the analysis runs on the library
    defaults of rossml, which answer a different estimate.
    """
    _check_name(model)
    path = os.path.join(APP_DIR, "models_files", model, "input_parameters.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def model_for(tag):
    """Return the model name of a tag, refusing tags outside settings.json."""
    current = load_settings()
    if tag not in current:
        raise ValueError("Invalid equipment: %s" % tag)
    model = current[tag]
    available, reason = model_status(model)
    if not available:
        raise ValueError("%s: %s" % (tag, reason))
    return model


def import_model(model):
    """Import `models_files.<model>.<model>` for a model listed in settings.json."""
    _check_name(model)
    ensure_on_path()
    headless_streamlit.install()
    return importlib.import_module("models_files.%s.%s" % (model, model))


def _check_name(model):
    if not MODEL_NAME.fullmatch(model or "") or model not in set(
        load_settings().values()
    ):
        raise ValueError("Invalid rotor model: %s" % model)


def _top_level_imports(source):
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def _source(model):
    path = _model_path(model)
    if path is None:
        raise ValueError("Model file not found: %s" % model)
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _model_path(model):
    if not MODEL_NAME.fullmatch(model or ""):
        return None
    path = os.path.join(APP_DIR, "models_files", model, model + ".py")
    return path if os.path.isfile(path) else None
