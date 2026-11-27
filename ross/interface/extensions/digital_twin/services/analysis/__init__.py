# -*- coding: utf-8 -*-
"""The Digital Twin analyses of step 3: one module per analysis.

Each module has `run(session, params)` and returns what the card draws:
`layout` (how the figures sit in the card), `figures` and `tables`.
"""

from ross.interface.extensions.digital_twin.domain import session

from . import campbell, clearance, stability, unbalance

ANALYSES = {
    "campbell": campbell.run,
    "unbalance": unbalance.run,
    "clearance": clearance.run,
    "stability": stability.run,
}


def run(identifier, kind, params):
    """Run one analysis on the session's rotor."""
    current = _with_rotor(identifier)
    if kind not in ANALYSES:
        raise ValueError("Unknown analysis: %s" % kind)
    if not isinstance(params, dict):
        raise ValueError("Invalid parameters.")
    return ANALYSES[kind](current, params)


def campbell_mode_shape(identifier, point):
    """The 3D mode of a point of the last Campbell diagram of the session."""
    return campbell.mode_shape(_with_rotor(identifier), point)


def _with_rotor(identifier):
    current = session.get(identifier)
    if current.rotor is None:
        raise ValueError("Create the rotor model (step 3) before running the analyses.")
    return current
