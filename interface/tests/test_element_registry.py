# -*- coding: utf-8 -*-
"""The map between what the screen calls an element and the ROSS class.

Before Phase 1 this map existed in three different, incompatible shapes. The
guard that matters is the regression one: every registered class has to exist
in the installed ROSS, so that a rename in a library update shows up as a
failure here, and not as an empty form on the user's screen."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import ross as rs

from domain import element_registry


def test_every_registered_class_exists_in_ross():
    """Regression guard: catches a class rename in a ROSS update."""
    for category, types in element_registry.ELEMENTS.items():
        for ui_type, ross_class in types.items():
            assert hasattr(rs, ross_class), (  # noqa: F405
                f"{category}/{ui_type} points at rs.{ross_class}, which does not exist"
            )


def test_registry_round_trip():
    for category, types in element_registry.ELEMENTS.items():
        for ui_type, ross_class in types.items():
            assert element_registry.ross_class_name(category, ui_type) == ross_class
            assert element_registry.ui_type_for_ross_class(ross_class) == (
                category,
                ui_type,
            )


def test_registry_falls_back_to_basic():
    assert element_registry.ross_class_name("bearings", "NaoExiste") == "BearingElement"
    assert element_registry.ross_class_name("seals", None) == "SealElement"
