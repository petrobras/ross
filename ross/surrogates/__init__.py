"""Surrogate models for bearings and seals.

- ``rossml.surrogates.surrogate_core``: ``CoefficientSurrogate`` and ``GaussianProcessSurrogate``.
- ``rossml.surrogates.bearings``: ``TiltingPadSurrogate``, ``ThrustPadSurrogate``, ``PlainJournalSurrogate``.
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
