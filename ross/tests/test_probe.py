import pytest
from numpy.testing import assert_allclose

from ross import Probe, Q_


@pytest.fixture
def probe():
    return Probe(10, Q_(45, "degree"), tag="V1")


def test_parameters(probe):
    assert_allclose(probe.node, 10)
    assert_allclose(probe.angle, 0.7853981633974483)
    assert probe.direction == "radial"


def test_info(probe):
    node, angle, direc = probe.info
    assert_allclose(node, 10)
    assert_allclose(angle, 0.7853981633974483)
    assert direc == "radial"


@pytest.fixture
def probe2():
    return Probe(10, direction="axial", tag="AX1")


def test_parameters_axial(probe2):
    assert_allclose(probe2.node, 10)
    assert probe2.angle is None
    assert probe2.direction == "axial"


def test_info_axial(probe2):
    node, angle, direc = probe2.info
    assert_allclose(node, 10)
    assert angle is None
    assert direc == "axial"
