import os
import numpy as np
import pytest

from numpy.testing import assert_allclose

from ross.seals.holepattern_seal import HolePatternSeal
from ross.units import Q_

import ccp

# Detect REFPROP availability using ccp
# ccp is always available as a dependency, but REFPROP is optional
# Check if REFPROP version is available (not "n/a")
REFPROP_AVAILABLE = "REFPROP : n/a" not in ccp.__version__full


# Common test parameters
COMMON_PARAMS = {
    "n": 0,
    "frequency": Q_([5000], "RPM"),
    "length": 0.0254,
    "radius": 0.0751,
    "clearance": 0.0004,
    "roughness": 0.00198,
    "cell_length": 0.001,
    "cell_width": 0.001,
    "cell_depth": 0.00229,
    "inlet_pressure": 1830000.0,
    "outlet_pressure": 823500.0,
    "inlet_temperature": 300.0,
    "preswirl": 1.0,
    "entr_coef": 0.1,
    "exit_coef": 0.5,
}

MANUAL_PARAMS = {
    "b_suther": 1.458e-6,
    "s_suther": 110.4,
    "molar": 29.0,
    "gamma": 1.4,
}

GAS_COMPOSITION = {
    "Nitrogen": 0.7812,
    "Oxygen": 0.2096,
    "Argon": 0.0092,
}


@pytest.fixture
def holepattern_manual():
    """HolePatternSeal with manual thermodynamic parameters."""
    seal = HolePatternSeal(**COMMON_PARAMS, **MANUAL_PARAMS)
    return seal


@pytest.fixture
def holepattern_gas_composition():
    """HolePatternSeal with gas_composition."""
    seal = HolePatternSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)
    return seal


def test_holepattern_manual_coefficients(holepattern_manual):
    """Test HolePatternSeal coefficients with manual parameters.

    This test should always work regardless of REFPROP availability.
    """
    assert_allclose(holepattern_manual.kxx, 586228.88958017, rtol=1e-4)
    assert_allclose(holepattern_manual.kxy, 159741.17580073, rtol=1e-4)
    assert_allclose(holepattern_manual.kyx, -159741.17580073, rtol=1e-4)
    assert_allclose(holepattern_manual.kyy, 586228.88958017, rtol=1e-4)
    assert_allclose(holepattern_manual.cxx, 294.42942927, rtol=1e-4)
    assert_allclose(holepattern_manual.cxy, -27.16217997, rtol=1e-4)
    assert_allclose(holepattern_manual.cyx, 27.16217997, rtol=1e-4)
    assert_allclose(holepattern_manual.cyy, 294.42942927, rtol=1e-4)
    assert_allclose(holepattern_manual.seal_leakage, 0.6313559209954082, rtol=1e-4)


def test_holepattern_manual_attributes(holepattern_manual):
    """Test that manual parameters are correctly stored."""
    assert holepattern_manual.molar == 29.0
    assert holepattern_manual.gamma == 1.4
    assert holepattern_manual.b_suther == 1.458e-6
    assert holepattern_manual.s_suther == 110.4


def test_holepattern_gas_composition_creation(holepattern_gas_composition):
    """Test that HolePatternSeal can be created with gas_composition.

    This should work with both REFPROP and HEOS backends.
    """
    assert holepattern_gas_composition is not None
    assert hasattr(holepattern_gas_composition, "kxx")
    assert hasattr(holepattern_gas_composition, "seal_leakage")


def test_holepattern_gas_composition_derived_properties(holepattern_gas_composition):
    """Test that thermodynamic properties are auto-derived correctly."""
    # Check that properties were derived
    assert holepattern_gas_composition.molar is not None
    assert holepattern_gas_composition.gamma is not None
    assert holepattern_gas_composition.b_suther is not None
    assert holepattern_gas_composition.s_suther is not None

    # Molar mass should be close to air (28.97 g/mol)
    # Expected: ~28.96 kg/kgmol, tolerance: 2%
    assert_allclose(holepattern_gas_composition.molar, 28.97, rtol=0.02)

    # Gamma should be close to air (1.4)
    # Expected: ~1.4, tolerance: 3% (HEOS and REFPROP differ slightly)
    assert_allclose(holepattern_gas_composition.gamma, 1.4, rtol=0.03)

    # Sutherland coefficients should be in reasonable range for air
    # Expected: ~1.46e-6, tolerance: 25% (varies between HEOS and REFPROP)
    assert_allclose(holepattern_gas_composition.b_suther, 1.46e-6, rtol=0.25)
    # Expected: ~100, tolerance: 25% (varies between HEOS and REFPROP)
    assert_allclose(holepattern_gas_composition.s_suther, 100, rtol=0.25)


def test_holepattern_gas_composition_coefficients_range(holepattern_gas_composition):
    """Test that coefficients are in reasonable range.

    Results may differ slightly between REFPROP and HEOS, but should be
    within 1% of each other.
    """
    # Expected values with some tolerance for backend differences
    # REFPROP: kxx ≈ 606105, HEOS: kxx ≈ 605752
    # Use average of REFPROP and HEOS as expected value
    assert_allclose(holepattern_gas_composition.kxx[0], 606000, rtol=0.025)  # ±2.5%
    assert_allclose(abs(holepattern_gas_composition.kxy[0]), 160000, rtol=0.1)  # ±10%
    assert_allclose(holepattern_gas_composition.cxx[0], 300, rtol=0.05)  # ±5%
    assert_allclose(
        holepattern_gas_composition.seal_leakage[0], 0.634, rtol=0.02
    )  # ±2%


def test_holepattern_gas_composition_vs_manual_similarity():
    """Test that gas_composition and manual parameters give similar results.

    They won't be identical because:
    1. gas_composition auto-derives properties from real gas data
    2. Manual parameters use simplified air properties

    But they should be within ~5% of each other.
    """
    holepattern_gas = HolePatternSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)
    holepattern_man = HolePatternSeal(**COMMON_PARAMS, **MANUAL_PARAMS)

    # Compare stiffness coefficients (should be within 5%)
    assert_allclose(holepattern_gas.kxx[0], holepattern_man.kxx[0], rtol=0.05)
    assert_allclose(holepattern_gas.cxx[0], holepattern_man.cxx[0], rtol=0.05)
    assert_allclose(
        holepattern_gas.seal_leakage[0], holepattern_man.seal_leakage[0], rtol=0.01
    )


@pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP")
def test_holepattern_refprop_backend():
    """Test HolePatternSeal with REFPROP backend explicitly.

    This test only runs when REFPROP is available.
    """
    # Ensure we're using REFPROP
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "REFPROP"

        holepattern = HolePatternSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        # REFPROP expected values (from testing)
        assert_allclose(holepattern.kxx[0], 606104.82, rtol=1e-3)
        assert_allclose(holepattern.cxx[0], 299.71, rtol=1e-3)
        assert_allclose(holepattern.seal_leakage[0], 0.633606, rtol=1e-3)

        # Check derived properties
        assert_allclose(holepattern.molar, 28.9586, rtol=1e-3)
        assert_allclose(holepattern.gamma, 1.4321, rtol=1e-3)

    finally:
        ccp.config.EOS = original_eos


def test_holepattern_heos_backend():
    """Test HolePatternSeal with HEOS backend explicitly.

    This test should always work.
    """
    original_eos = ccp.config.EOS
    try:
        ccp.config.EOS = "HEOS"

        holepattern = HolePatternSeal(**COMMON_PARAMS, gas_composition=GAS_COMPOSITION)

        # HEOS expected values (from testing)
        # Note: HEOS values differ slightly from REFPROP (<0.1%)
        assert_allclose(holepattern.kxx[0], 605751.93, rtol=1e-2)
        assert_allclose(holepattern.cxx[0], 299.76, rtol=1e-2)
        assert_allclose(holepattern.seal_leakage[0], 0.633636, rtol=1e-3)

        # Check derived properties
        assert_allclose(holepattern.molar, 28.9586, rtol=1e-3)
        # HEOS gamma calculation should be similar to REFPROP
        assert_allclose(holepattern.gamma, 1.4321, rtol=1e-2)

    finally:
        ccp.config.EOS = original_eos


@pytest.mark.skipif(not REFPROP_AVAILABLE, reason="Requires REFPROP")
def test_holepattern_refprop_vs_heos_consistency():
    """Compare REFPROP and HEOS results for the same configuration.

    Results should be within 1% of each other.
    """
    original_eos = ccp.config.EOS

    try:
        # Test with REFPROP
        ccp.config.EOS = "REFPROP"
        holepattern_refprop = HolePatternSeal(
            **COMMON_PARAMS, gas_composition=GAS_COMPOSITION
        )

        # Test with HEOS
        ccp.config.EOS = "HEOS"
        holepattern_heos = HolePatternSeal(
            **COMMON_PARAMS, gas_composition=GAS_COMPOSITION
        )

        # Compare coefficients (should be within 1%)
        assert_allclose(
            holepattern_refprop.kxx[0],
            holepattern_heos.kxx[0],
            rtol=0.01,
            err_msg="kxx differs by >1% between REFPROP and HEOS",
        )
        assert_allclose(
            holepattern_refprop.cxx[0],
            holepattern_heos.cxx[0],
            rtol=0.01,
            err_msg="cxx differs by >1% between REFPROP and HEOS",
        )
        assert_allclose(
            holepattern_refprop.seal_leakage[0],
            holepattern_heos.seal_leakage[0],
            rtol=0.001,
            err_msg="seal_leakage differs by >0.1% between REFPROP and HEOS",
        )

        # Derived properties should match closely
        assert_allclose(holepattern_refprop.molar, holepattern_heos.molar, rtol=1e-4)
        assert_allclose(holepattern_refprop.gamma, holepattern_heos.gamma, rtol=1e-2)

    finally:
        ccp.config.EOS = original_eos


def test_holepattern_without_gas_params_raises():
    """Test that creating HolePatternSeal without gas params raises error."""
    with pytest.raises(TypeError):
        # Should fail because neither gas_composition nor manual params provided
        # Raises TypeError: unsupported operand type(s) for /: 'float' and 'NoneType'
        HolePatternSeal(**COMMON_PARAMS)


def test_holepattern_invalid_gas_composition():
    """Test that invalid gas composition raises appropriate error."""
    invalid_gas = {"InvalidFluid123": 1.0}

    with pytest.raises((ValueError, KeyError)):
        HolePatternSeal(**COMMON_PARAMS, gas_composition=invalid_gas)


def test_holepattern_multiple_frequencies():
    """Test HolePatternSeal with multiple frequency points."""
    params = COMMON_PARAMS.copy()
    params["frequency"] = Q_([3000, 5000, 7000], "RPM")

    holepattern = HolePatternSeal(**params, **MANUAL_PARAMS)

    # Should have 3 values for each coefficient
    assert len(holepattern.kxx) == 3
    assert len(holepattern.cxx) == 3
    assert len(holepattern.seal_leakage) == 3

    # All should be positive
    assert all(np.array(holepattern.kxx) > 0)
    assert all(np.array(holepattern.cxx) > 0)
    assert all(np.array(holepattern.seal_leakage) > 0)


@pytest.fixture
def holepattern():
    """Original fixture for backward compatibility."""
    seal = HolePatternSeal(
        n=0,
        frequency=Q_([5000], "RPM"),
        length=0.0254,
        radius=0.0751,
        clearance=0.0004,
        roughness=0.00198,
        cell_length=0.001,
        cell_width=0.001,
        cell_depth=0.00229,
        inlet_pressure=1830000.0,
        outlet_pressure=823500.0,
        inlet_temperature=300.0,
        b_suther=1.458e-6,
        s_suther=110.4,
        molar=29.0,
        gamma=1.4,
        preswirl=1.0,
        entr_coef=0.1,
        exit_coef=0.5,
    )
    return seal


def test_holepattern_coefficients(holepattern):
    """Original test - kept for backward compatibility."""
    assert_allclose(holepattern.kxx, 586228.88958017, rtol=1e-4)
    assert_allclose(holepattern.kxy, 159741.17580073, rtol=1e-4)
    assert_allclose(holepattern.kyx, -159741.17580073, rtol=1e-4)
    assert_allclose(holepattern.kyy, 586228.88958017, rtol=1e-4)
    assert_allclose(holepattern.cxx, 294.42942927, rtol=1e-4)
    assert_allclose(holepattern.cxy, -27.16217997, rtol=1e-4)
    assert_allclose(holepattern.cyx, 27.16217997, rtol=1e-4)
    assert_allclose(holepattern.cyy, 294.42942927, rtol=1e-4)
    assert_allclose(holepattern.seal_leakage, 0.6313559209954082, rtol=1e-4)
