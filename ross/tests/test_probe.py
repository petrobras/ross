import pytest
from numpy.testing import assert_allclose

from ross.probe import Probe
from ross.units import Q_


@pytest.fixture
def probe():
    return Probe(10, Q_(45, "degree"), "V1")


def test_parameters(probe):
    assert_allclose(probe.node, 10)
    assert_allclose(probe.angle, 0.7853981633974483)


def test_info(probe):
    node, angle = probe.info
    assert_allclose(node, 10)
    assert_allclose(angle, 0.7853981633974483)
