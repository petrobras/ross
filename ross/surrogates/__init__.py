"""Surrogate models for bearings.

- ``ross.surrogates.kriging``: ``Kriging`` metamodel (numpy and scipy only) and the sampling plans.
- ``ross.surrogates.surrogate_core``: ``CoefficientSurrogate`` and ``GaussianProcessSurrogate``.
- ``ross.surrogates.bearings``: ``TiltingPadSurrogate``, ``ThrustPadSurrogate``, ``PlainJournalSurrogate``.

The classes are imported on first access, so ``import ross.surrogates`` stays cheap.
``Kriging`` needs only numpy and scipy; the other classes also use scikit-learn,
json_tricks and tqdm.
"""

import importlib

_CLASS_MODULES = {
    "Kriging": "ross.surrogates.kriging",
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
