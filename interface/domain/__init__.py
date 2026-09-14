"""Domain rules of the ROSS interface.

This package is the single source of truth for everything that describes the
domain: units, the mapping from UI types to ROSS classes, and node numbering.
Nothing here imports Flask -- it is plain domain code, testable on its own.
"""

from .units import (
    UNITS_MAPPING,
    UNIT_ALTERNATIVES,
    INT_PARAMETERS,
    unit_for,
    alternatives_for,
)
from .element_registry import (
    ELEMENTS,
    ross_class_name,
    ui_type_for_ross_class,
    categories,
)
from .node_resolver import effective_nodes, validate_node_topology

__all__ = [
    "UNITS_MAPPING",
    "UNIT_ALTERNATIVES",
    "INT_PARAMETERS",
    "unit_for",
    "alternatives_for",
    "ELEMENTS",
    "ross_class_name",
    "ui_type_for_ross_class",
    "categories",
    "effective_nodes",
    "validate_node_topology",
]
