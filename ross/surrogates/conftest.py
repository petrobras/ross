"""Skip the collection of the surrogate modules when the optional dependencies are missing."""

import importlib.util

_REQUIRED = ("smt", "sklearn", "json_tricks", "tqdm")

collect_ignore_glob = []
if any(importlib.util.find_spec(name) is None for name in _REQUIRED):
    collect_ignore_glob = ["*.py", "bearings/*.py"]
