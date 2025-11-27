import numpy as np
import pytest

from numpy.testing import assert_allclose

from ross.seals.hybrid_seal import HybridSeal
from ross.units import Q_


# Common test parameters for hybrid seal
COMMON_PARAMS = {
    "n": 0,
    "inlet_pressure": 500000,
    "outlet_pressure": 100000,
    "inlet_temperature": 300.0,
    "frequency": Q_([5000], "RPM"),
}

# Labyrinth seal parameters
LABYRINTH_PARAMS = {
    "n_teeth": 10,
    "shaft_radius": Q_(25, "mm"),
    "radial_clearance": Q_(0.25, "mm"),
    "pitch": Q_(3, "mm"),
    "tooth_height": Q_(3, "mm"),
    "tooth_width": Q_(0.15, "mm"),
    "seal_type": "inter",
    "pre_swirl_ratio": 0.9,
}

# Hole-pattern seal parameters
HOLEPATTERN_PARAMS = {
    "length": 0.04,
    "radius": 0.025,
    "clearance": 0.0003,
    "roughness": 0.0001,
    "cell_length": 0.003,
    "cell_width": 0.003,
    "cell_depth": 0.002,
    "preswirl": 0.8,
    "entr_coef": 0.5,
    "exit_coef": 1.0,
}

# Manual thermodynamic parameters
MANUAL_PARAMS = {
    "r": 287.05,
    "gamma": 1.4,
    "tz": [300.0, 299.5],
    "muz": [1.85e-05, 1.84e-05],
    "b_suther": 1.458e-6,
    "s_suther": 110.4,
    "molar": 29.0,
}

# Gas composition (air)
GAS_COMPOSITION = {
    "Nitrogen": 0.7812,
    "Oxygen": 0.2096,
    "Argon": 0.0092,
}


@pytest.fixture
def hybrid_seal():
    """HybridSeal fixture for testing hybrid seal functionality."""
    seal = HybridSeal(
        **COMMON_PARAMS,
        **LABYRINTH_PARAMS,
        **HOLEPATTERN_PARAMS,
        **MANUAL_PARAMS,
    )
    return seal


def test_hybrid_interface_pressure(hybrid_seal):
    """Test that interface pressure is within physical bounds."""
    assert hybrid_seal.interface_pressure > COMMON_PARAMS["outlet_pressure"]
    assert hybrid_seal.interface_pressure < COMMON_PARAMS["inlet_pressure"]

    assert_allclose(hybrid_seal.interface_pressure, 463934.707642, rtol=1e-4)


def test_hybrid_leakage_matching(hybrid_seal):
    """Test that leakage is matched between labyrinth and hole-pattern stages."""
    laby_leakage = hybrid_seal.laby.seal_leakage[0]
    hole_leakage = hybrid_seal.hole.seal_leakage[0]
    hybrid_leakage = hybrid_seal.seal_leakage

    assert_allclose(laby_leakage, hole_leakage, rtol=1e-5)
    assert_allclose(hybrid_leakage, 0.032909, rtol=1e-4)


def test_hybrid_coefficients_combination(hybrid_seal):
    """Test that hybrid coefficients are proper series combination."""
    laby_kxx = hybrid_seal.laby.kxx[0]
    hole_kxx = hybrid_seal.hole.kxx[0]
    hybrid_kxx = hybrid_seal.kxx[0]
    assert_allclose(hybrid_kxx, laby_kxx + hole_kxx, rtol=1e-6)
    assert_allclose(hybrid_kxx, 178630.49443027, rtol=1e-4)

    laby_kxy = hybrid_seal.laby.kxy[0]
    hole_kxy = hybrid_seal.hole.kxy[0]
    hybrid_kxy = hybrid_seal.kxy[0]
    assert_allclose(hybrid_kxy, laby_kxy + hole_kxy, rtol=1e-6)
    assert_allclose(hybrid_kxy, 25473.36057114, rtol=1e-4)

    # Check direct damping
    laby_cxx = hybrid_seal.laby.cxx[0]
    hole_cxx = hybrid_seal.hole.cxx[0]
    hybrid_cxx = hybrid_seal.cxx[0]
    assert_allclose(hybrid_cxx, laby_cxx + hole_cxx, rtol=1e-6)
    assert_allclose(hybrid_cxx, 54.954439, rtol=1e-4)

    # Check cross-coupled damping
    laby_cxy = hybrid_seal.laby.cxy[0]
    hole_cxy = hybrid_seal.hole.cxy[0]
    hybrid_cxy = hybrid_seal.cxy[0]
    assert_allclose(hybrid_cxy, laby_cxy + hole_cxy, rtol=1e-6)
    assert_allclose(hybrid_cxy, -4.859273, rtol=1e-4)


@pytest.fixture
def hybrid_seal_gc():
    """HybridSeal fixture for testing hybrid seal with gas composition."""
    seal = HybridSeal(
        **COMMON_PARAMS,
        **LABYRINTH_PARAMS,
        **HOLEPATTERN_PARAMS,
        gas_composition=GAS_COMPOSITION,
    )
    return seal


def test_hybrid_gc_interface_pressure(hybrid_seal_gc):
    """Test that interface pressure is within physical bounds."""
    assert hybrid_seal_gc.interface_pressure > COMMON_PARAMS["outlet_pressure"]
    assert hybrid_seal_gc.interface_pressure < COMMON_PARAMS["inlet_pressure"]

    assert_allclose(hybrid_seal_gc.interface_pressure, 463932.037354, rtol=1e-4)


def test_hybrid_gc_leakage_matching(hybrid_seal_gc):
    """Test that leakage is matched between labyrinth and hole-pattern stages."""
    laby_leakage = hybrid_seal_gc.laby.seal_leakage[0]
    hole_leakage = hybrid_seal_gc.hole.seal_leakage[0]
    hybrid_leakage = hybrid_seal_gc.seal_leakage

    assert_allclose(laby_leakage, hole_leakage, rtol=1e-5)
    assert_allclose(hybrid_leakage, 0.032906, rtol=1e-4)


def test_hybrid_gc_coefficients_combination(hybrid_seal_gc):
    """Test that hybrid coefficients are proper series combination."""
    laby_kxx = hybrid_seal_gc.laby.kxx[0]
    hole_kxx = hybrid_seal_gc.hole.kxx[0]
    hybrid_kxx = hybrid_seal_gc.kxx[0]
    assert_allclose(hybrid_kxx, laby_kxx + hole_kxx, rtol=1e-6)
    assert_allclose(hybrid_kxx, 180016.44722638, rtol=1e-2)

    laby_kxy = hybrid_seal_gc.laby.kxy[0]
    hole_kxy = hybrid_seal_gc.hole.kxy[0]
    hybrid_kxy = hybrid_seal_gc.kxy[0]
    assert_allclose(hybrid_kxy, laby_kxy + hole_kxy, rtol=1e-6)
    assert_allclose(hybrid_kxy, 25513.24346651, rtol=1e-2)

    # Check direct damping
    laby_cxx = hybrid_seal_gc.laby.cxx[0]
    hole_cxx = hybrid_seal_gc.hole.cxx[0]
    hybrid_cxx = hybrid_seal_gc.cxx[0]
    assert_allclose(hybrid_cxx, laby_cxx + hole_cxx, rtol=1e-6)
    assert_allclose(hybrid_cxx, 55.035337, rtol=1e-2)

    # Check cross-coupled damping
    laby_cxy = hybrid_seal_gc.laby.cxy[0]
    hole_cxy = hybrid_seal_gc.hole.cxy[0]
    hybrid_cxy = hybrid_seal_gc.cxy[0]
    assert_allclose(hybrid_cxy, laby_cxy + hole_cxy, rtol=1e-6)
    assert_allclose(hybrid_cxy, -4.871907, rtol=1e-2)
