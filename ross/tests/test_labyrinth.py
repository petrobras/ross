import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_almost_equal

from ross.seals.labyrinth_seal import LabyrinthSeal
from ross.units import Q_

import ccp

# Detect REFPROP availability using ccp
# ccp is always available as a dependency, but REFPROP is optional
# Check if REFPROP version is available (not "n/a")
REFPROP_AVAILABLE = "REFPROP : n/a" not in ccp.__version__full


# Common test parameters
COMMON_PARAMS = {
    "n": 0,
    "inlet_pressure": 308000,
    "outlet_pressure": 94300,
    "inlet_temperature": 283.15,
    "preswirl": 0.98,
    "frequency": Q_([8000], "RPM"),
    "n_teeth": 16,
    "shaft_radius": Q_(72.5, "mm"),
    "radial_clearance": Q_(0.3, "mm"),
    "pitch": Q_(3.175, "mm"),
    "tooth_height": Q_(3.175, "mm"),
    "tooth_width": Q_(0.1524, "mm"),
    "seal_type": "inter",
}

MANUAL_PARAMS = {
    "r": 287.05,
    "tz": [283.15, 282.60903080958565],
    "muz": [1.7746561138374613e-05, 1.7687886306966975e-05],
    "gamma": 1.41,
}

GAS_COMPOSITION = {
    "Nitrogen": 0.7812,
    "Oxygen": 0.2096,
    "Argon": 0.0092,
}


@pytest.fixture
def labyrinth_manual():
    """LabyrinthSeal with manual thermodynamic parameters."""
    seal = LabyrinthSeal(**COMMON_PARAMS, **MANUAL_PARAMS)
    return seal


@pytest.fixture
def labyrinth_gas_composition():
    """LabyrinthSeal with gas_composition."""
    seal = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)
    return seal


def test_labyrinth_manual_coefficients(labyrinth_manual):
    """Test LabyrinthSeal coefficients with manual parameters.

    This test should always work regardless of REFPROP availability.
    """
    assert_allclose(labyrinth_manual.kxx, -50448.19591, rtol=1e-4)
    assert_allclose(labyrinth_manual.kxy, 35536.953463, rtol=1e-4)
    assert_allclose(labyrinth_manual.kyx, -35536.953463, rtol=1e-4)
    assert_allclose(labyrinth_manual.kyy, -50448.19591, rtol=1e-4)
    assert_allclose(labyrinth_manual.cxx, 23.821095, rtol=1e-4)
    assert_allclose(labyrinth_manual.cxy, 56.246427, rtol=1e-4)
    assert_allclose(labyrinth_manual.cyx, -56.246427, rtol=1e-4)
    assert_allclose(labyrinth_manual.cyy, 23.821095, rtol=1e-4)


def test_labyrinth_manual_attributes(labyrinth_manual):
    """Test that manual parameters are correctly stored."""
    assert labyrinth_manual.r == 287.05
    assert labyrinth_manual.gamma == 1.41
    assert_allclose(labyrinth_manual.tz[0], 283.15, rtol=1e-6)
    assert_allclose(labyrinth_manual.tz[1], 282.60903080958565, rtol=1e-6)
    assert_allclose(labyrinth_manual.muz[0], 1.7746561138374613e-05, rtol=1e-6)
    assert_allclose(labyrinth_manual.muz[1], 1.7687886306966975e-05, rtol=1e-6)


def test_labyrinth_gas_composition_creation(labyrinth_gas_composition):
    """Test that LabyrinthSeal can be created with gas_composition.

    This should work with both REFPROP and HEOS backends.
    """
    assert labyrinth_gas_composition is not None
    assert hasattr(labyrinth_gas_composition, "kxx")
    assert hasattr(labyrinth_gas_composition, "seal_leakage")


def test_labyrinth_gas_composition_derived_properties(labyrinth_gas_composition):
    """Test that thermodynamic properties are auto-derived correctly."""
    # Check that properties were derived
    assert labyrinth_gas_composition.r is not None
    assert labyrinth_gas_composition.gamma is not None
    assert labyrinth_gas_composition.tz is not None
    assert labyrinth_gas_composition.muz is not None

    assert_allclose(labyrinth_gas_composition.r, 287, rtol=0.02)
    assert_allclose(labyrinth_gas_composition.gamma, 1.4, rtol=0.03)
    assert len(labyrinth_gas_composition.tz) == 2
    assert_allclose(labyrinth_gas_composition.tz[0], 283.15, rtol=1e-4)
    assert len(labyrinth_gas_composition.muz) == 2
    assert_allclose(labyrinth_gas_composition.muz[0], 1.77e-05, rtol=0.05)


def test_labyrinth_gas_composition_coefficients_range(labyrinth_gas_composition):
    """Test that coefficients are in reasonable range.

    Results may differ slightly between REFPROP and HEOS, but should be
    within a few percent of each other.
    """
    # Expected values with some tolerance for backend differences
    assert_allclose(labyrinth_gas_composition.kxx[0], -50440, rtol=0.01)  # ±1%
    assert_allclose(abs(labyrinth_gas_composition.kxy[0]), 35515, rtol=0.01)  # ±1%
    assert_allclose(labyrinth_gas_composition.cxx[0], 23.8, rtol=0.01)  # ±1%
    assert_allclose(
        labyrinth_gas_composition.seal_leakage[0], 0.05195, rtol=0.01
    )  # ±1%


def test_labyrinth_gas_composition_vs_manual_similarity():
    """Test that gas_composition and manual parameters give similar results.

    They won't be identical because:
    1. gas_composition auto-derives properties from real gas data
    2. Manual parameters use simplified air properties

    But they should be within ~5% of each other.
    """
    labyrinth_gas = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)
    labyrinth_man = LabyrinthSeal(**COMMON_PARAMS, **MANUAL_PARAMS)

    assert_allclose(labyrinth_gas.kxx[0], labyrinth_man.kxx[0], rtol=0.05)
    assert_allclose(labyrinth_gas.cxx[0], labyrinth_man.cxx[0], rtol=0.05)
    assert_allclose(
        labyrinth_gas.seal_leakage[0], labyrinth_man.seal_leakage[0], rtol=0.05
    )


@pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP")
def test_labyrinth_refprop_backend():
    """Test LabyrinthSeal with REFPROP backend explicitly.

    This test only runs when REFPROP is available.
    """
    # Ensure we're using REFPROP
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "REFPROP"

        labyrinth = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        # REFPROP expected values (from testing)
        assert_allclose(labyrinth.kxx[0], -50435.21, rtol=1e-3)
        assert_allclose(labyrinth.cxx[0], 23.808, rtol=1e-3)
        assert_allclose(labyrinth.seal_leakage[0], 0.051951, rtol=1e-3)

        # Check derived properties
        assert_allclose(labyrinth.r, 287.12, rtol=1e-3)
        assert_allclose(labyrinth.gamma, 1.41, rtol=1e-3)

    finally:
        ccp.config.EOS = original_eos


def test_labyrinth_heos_backend():
    """Test LabyrinthSeal with HEOS backend explicitly.

    This test should always work.
    """
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "HEOS"

        labyrinth = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        # HEOS expected values (from testing)
        # Note: HEOS values differ slightly from REFPROP (<0.05%)
        assert_allclose(labyrinth.kxx[0], -50449.96, rtol=1e-3)
        assert_allclose(labyrinth.cxx[0], 23.797, rtol=1e-3)
        assert_allclose(labyrinth.seal_leakage[0], 0.051951, rtol=1e-3)

        # Check derived properties
        assert_allclose(labyrinth.r, 287.12, rtol=1e-3)
        assert_allclose(labyrinth.gamma, 1.41, rtol=1e-3)

    finally:
        ccp.config.EOS = original_eos


@pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP")
def test_labyrinth_refprop_vs_heos_consistency():
    """Compare REFPROP and HEOS results for the same configuration.

    Results should be within 1% of each other.
    """
    original_eos = ccp.config.EOS

    try:
        # Test with REFPROP
        ccp.config.EOS = "REFPROP"
        labyrinth_refprop = LabyrinthSeal(
            **COMMON_PARAMS, gas_composition=GAS_COMPOSITION
        )

        # Test with HEOS
        ccp.config.EOS = "HEOS"
        labyrinth_heos = LabyrinthSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        # Compare coefficients (should be within 1%)
        assert_allclose(
            labyrinth_refprop.kxx[0],
            labyrinth_heos.kxx[0],
            rtol=0.01,
            err_msg="kxx differs by >1% between REFPROP and HEOS",
        )
        assert_allclose(
            labyrinth_refprop.cxx[0],
            labyrinth_heos.cxx[0],
            rtol=0.01,
            err_msg="cxx differs by >1% between REFPROP and HEOS",
        )
        assert_allclose(
            labyrinth_refprop.seal_leakage[0],
            labyrinth_heos.seal_leakage[0],
            rtol=0.01,
            err_msg="seal_leakage differs by >1% between REFPROP and HEOS",
        )

        # Derived properties should match closely
        assert_allclose(labyrinth_refprop.r, labyrinth_heos.r, rtol=1e-3)
        assert_allclose(labyrinth_refprop.gamma, labyrinth_heos.gamma, rtol=1e-2)

    finally:
        ccp.config.EOS = original_eos


def test_labyrinth_invalid_gas_composition():
    """Test that invalid gas composition raises appropriate error."""
    invalid_gas = {"InvalidFluid123": 1.0}

    with pytest.raises((ValueError, KeyError)):
        LabyrinthSeal(**COMMON_PARAMS, gas_composition=invalid_gas)


@pytest.fixture
def labyrinth():
    """Original fixture for backward compatibility."""
    seal = LabyrinthSeal(
        n=0,
        inlet_pressure=308000,
        outlet_pressure=94300,
        inlet_temperature=283.15,
        preswirl=0.98,
        frequency=Q_([8000], "RPM"),
        n_teeth=16,
        shaft_radius=Q_(72.5, "mm"),
        radial_clearance=Q_(0.3, "mm"),
        pitch=Q_(3.175, "mm"),
        tooth_height=Q_(3.175, "mm"),
        tooth_width=Q_(0.1524, "mm"),
        seal_type="inter",
        r=287.05,
        tz=[283.15, 282.60903080958565],
        muz=[1.7746561138374613e-05, 1.7687886306966975e-05],
        gamma=1.41,
    )

    return seal


def test_labyrinth_coefficients(labyrinth):
    """Original test - kept for backward compatibility."""
    assert_allclose(labyrinth.kxx, -50448.19591, rtol=1e-4)
    assert_allclose(labyrinth.kxy, 35536.953463, rtol=1e-4)
    assert_allclose(labyrinth.kyx, -35536.953463, rtol=1e-4)
    assert_allclose(labyrinth.kyy, -50448.19591, rtol=1e-4)
    assert_allclose(labyrinth.cxx, 23.821095, rtol=1e-4)
    assert_allclose(labyrinth.cxy, 56.246427, rtol=1e-4)
    assert_allclose(labyrinth.cyx, -56.246427, rtol=1e-4)
    assert_allclose(labyrinth.cyy, 23.821095, rtol=1e-4)
