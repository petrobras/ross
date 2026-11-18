"""Surrogate models for bearings.

- ``ross.surrogates.surrogate_core``: ``CoefficientSurrogate`` and ``GaussianProcessSurrogate``.
- ``ross.surrogates.bearings``: ``TiltingPadSurrogate``, ``ThrustPadSurrogate``, ``PlainJournalSurrogate``.

The classes need the optional dependencies installed with
``pip install "ross-rotordynamics[surrogates]"``. They are imported on first access,
so ``import ross.surrogates`` works without them and the error appears only when a
class is used.
"""

import importlib

_CLASS_MODULES = {
    "CoefficientSurrogate": "ross.surrogates.surrogate_core",
    "GaussianProcessSurrogate": "ross.surrogates.surrogate_core",
    "TiltingPadSurrogate": "ross.surrogates.bearings",
    "ThrustPadSurrogate": "ross.surrogates.bearings",
    "PlainJournalSurrogate": "ross.surrogates.bearings",
}

__all__ = list(_CLASS_MODULES)


def __getattr__(name):
    if name in _CLASS_MODULES:
        return getattr(importlib.import_module(_CLASS_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
