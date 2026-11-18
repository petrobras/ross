"""Surrogate models for bearings and seals.

- ``ross.surrogates.surrogate_core``: ``CoefficientSurrogate`` and ``GaussianProcessSurrogate``.
- ``ross.surrogates.bearings``: ``TiltingPadSurrogate``, ``ThrustPadSurrogate``, ``PlainJournalSurrogate``.
"""

from .surrogate_core import (
    CoefficientSurrogate,
    GaussianProcessSurrogate,
)
from .bearings import PlainJournalSurrogate, ThrustPadSurrogate, TiltingPadSurrogate

__all__ = [
    "CoefficientSurrogate",  # deprecated alias
    "GaussianProcessSurrogate",
    "TiltingPadSurrogate",
    "ThrustPadSurrogate",
    "PlainJournalSurrogate",
]
