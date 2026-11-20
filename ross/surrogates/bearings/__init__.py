"""Bearing surrogate models."""

from .tiltingpad import TiltingPadSurrogate
from .thrustpad import ThrustPadSurrogate
from .plain_journal import PlainJournalSurrogate

__all__ = [
    "TiltingPadSurrogate",
    "ThrustPadSurrogate",
    "PlainJournalSurrogate",
]
