# -*- coding: utf-8 -*-
"""Which nodes a rotor really has, and where one is missing.

The effective numbering is not `range(len(shafts))`: a bearing can point at a
*link* node, past the last node of the shaft, and a point mass lives there. The
rule is here and the same one has to hold on the screen -- the user compares the
two numbers.

And the topology validation refuses the gap: a node named without the
previous one existing lets ROSS build a singular matrix without saying anything."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from ross.interface.domain import node_resolver


# A fake element: `node_resolver` only looks at attributes (`n`, `n_link`), so
# building real rotors here would cost the installed ROSS without measuring
# anything more.
class _FakeElement:
    def __init__(self, **attributes):
        for name, value in attributes.items():
            setattr(self, name, value)


@pytest.mark.parametrize(
    "elements, expected",
    [
        ([{}, {}, {}], [0, 1, 2]),
        ([{"n": "5"}, {"n": "2"}], [5, 2]),
        ([{"n": "1"}, {}, {}, {"n": "0"}], [1, 2, 3, 0]),
        ([{"n": "  "}, {"n": "abc"}, {}], [0, 1, 2]),
        ([], []),
    ],
)
def test_effective_nodes(elements, expected):
    assert node_resolver.effective_nodes(elements) == expected


@pytest.mark.parametrize(
    "elements, expected",
    [
        ([{}, {}, {}], [0, 1, 2]),
        ([{"n": "2"}, {}], [2, 1]),
        ([{}, {"n": "0"}], [0, 0]),
        ([{"n": "abc"}, {"n": "3.0"}], [0, 3]),
        ([], []),
    ],
)
def test_listed_nodes_is_the_position_unless_pinned(elements, expected):
    """The builder's rule for couplings, which is not `effective_nodes`."""
    assert node_resolver.listed_nodes(elements) == expected


def test_validate_node_topology_accepts_link_nodes():
    """A bearing with n_link plus a PointMass on the link node is valid topology."""
    elements = [
        _FakeElement(n_l=0, n_r=1),
        _FakeElement(n_l=1, n_r=2),
        _FakeElement(n=0, n_link=3),
        _FakeElement(n=3),
        _FakeElement(n=2, n_link=4),
        _FakeElement(n=4),
    ]
    assert node_resolver.validate_node_topology(elements) == {0, 1, 2, 3, 4}


def test_validate_node_topology_detects_gap_behind_n_link():
    """n_link is collected too, so a gap behind it does not slip through."""
    elements = [_FakeElement(n_l=0, n_r=1), _FakeElement(n=0, n_link=5)]
    with pytest.raises(ValueError) as exc:
        node_resolver.validate_node_topology(elements)
    assert "2, 3, 4" in str(exc.value)


def test_validate_node_topology_detects_classic_gap():
    elements = [_FakeElement(n_l=0, n_r=1), _FakeElement(n=7)]
    with pytest.raises(ValueError):
        node_resolver.validate_node_topology(elements)


def test_validate_node_topology_accepts_empty_rotor():
    assert node_resolver.validate_node_topology([]) == set()
