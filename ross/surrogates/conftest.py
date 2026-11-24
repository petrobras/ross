"""Skip the collection of the modules that import the surrogate when the optional dependencies are missing."""

import importlib.util

_REQUIRED = ("sklearn", "json_tricks", "tqdm")

collect_ignore_glob = []
if any(importlib.util.find_spec(name) is None for name in _REQUIRED):
    collect_ignore_glob = ["surrogate_core.py", "bearings/*.py"]
