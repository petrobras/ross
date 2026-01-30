import pytest
from numpy.testing import assert_allclose
import numpy as np

from ross.bearings.tilting_pad import TiltingPad
from ross.units import Q_


@pytest.fixture
def tilting_pad_match():
    """
    Tilting pad bearing with match_eccentricity equilibrium type.

    Parameters
    ----------
    frequency : pint.Quantity
        Rotational speed (3000 RPM)
    journal_diameter : float
        Journal diameter (101.6 mm)
    radial_clearance : float
        Radial clearance (74.9 μm)
    pivot_angle : pint.Quantity
        Pivot angles for 5 pads [18°, 90°, 162°, 234°, 306°]
    pad_arc : pint.Quantity
        Arc length for each pad (60° each)
    lubricant : str
        Lubricant type (ISOVG32)
    oil_supply_temperature : pint.Quantity
        Supply temperature (40°C)

    Returns
    -------
    bearing : TiltingPad
        Configured tilting pad bearing object

    """
    frequency = Q_([3000], "RPM")
    pivot_angle = Q_([18, 90, 162, 234, 306], "deg")
    pad_arc = Q_([60, 60, 60, 60, 60], "deg")
    pad_axial_length = Q_([50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3], "m")
    oil_supply_temperature = Q_(40, "degC")
    attitude_angle = Q_(287.5, "deg")

    bearing = TiltingPad(
        n=1,
        frequency=frequency,
        equilibrium_type="match_eccentricity",
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=pivot_angle,
        pad_arc=pad_arc,
        pad_axial_length=pad_axial_length,
        pre_load=[0.5, 0.5, 0.5, 0.5, 0.5],
        offset=[0.5, 0.5, 0.5, 0.5, 0.5],
        lubricant="ISOVG32",
        oil_supply_temperature=oil_supply_temperature,
        nx=30,
        nz=30,
        eccentricity=0.35,
        attitude_angle=attitude_angle,
        load=[8.8405e02, -2.6704e03],
    )

    return bearing


@pytest.fixture
def tilting_pad_determine():
    """
    Tilting pad bearing with determine_eccentricity equilibrium type.

    Uses thermo-hydrodynamic model to determine equilibrium position
    from initial guesses of eccentricity and attitude angle.

    Parameters
    ----------
    frequency : pint.Quantity
        Rotational speed (3000 RPM)
    journal_diameter : float
        Journal diameter (101.6 mm)
    radial_clearance : float
        Radial clearance (74.9 μm)
    model_type : str
        Analysis model type ('thermo_hydro_dynamic')
    eccentricity : float
        Initial guess for eccentricity (0.30)
    attitude_angle : pint.Quantity
        Initial guess for attitude angle (267.5°)

    Returns
    -------
    bearing : TiltingPad
        Configured tilting pad bearing object

    """
    frequency = Q_([3000], "RPM")
    pivot_angle = Q_([18, 90, 162, 234, 306], "deg")
    pad_arc = Q_([60, 60, 60, 60, 60], "deg")
    pad_axial_length = Q_([50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3, 50.8e-3], "m")
    oil_supply_temperature = Q_(40, "degC")

    bearing = TiltingPad(
        n=1,
        frequency=frequency,
        equilibrium_type="determine_eccentricity",
        model_type="thermo_hydro_dynamic",
        journal_diameter=101.6e-3,
        radial_clearance=74.9e-6,
        pad_thickness=12.7e-3,
        pivot_angle=pivot_angle,
        pad_arc=pad_arc,
        pad_axial_length=pad_axial_length,
        pre_load=[0.5, 0.5, 0.5, 0.5, 0.5],
        offset=[0.5, 0.5, 0.5, 0.5, 0.5],
        lubricant="ISOVG32",
        oil_supply_temperature=oil_supply_temperature,
        nx=30,
        nz=30,
        eccentricity=0.30,
        attitude_angle=Q_(267.5, "deg"),
        load=[8.8405e02, -2.6704e03],
    )

    return bearing


def test_tilting_pad_parameters(tilting_pad_match):
    """Test basic geometric and operational parameters."""
    assert_allclose(tilting_pad_match.journal_radius, 0.0508)
    assert_allclose(tilting_pad_match.radial_clearance, 74.9e-6)
    assert_allclose(tilting_pad_match.frequency, 314.1592653589793)
    assert_allclose(tilting_pad_match.reference_temperature, 40)
    assert_allclose(tilting_pad_match.n_pad, 5)


def test_tilting_pad_parameters_determine(tilting_pad_determine):
    """Test basic geometric and operational parameters for determine case."""
    assert_allclose(tilting_pad_determine.journal_radius, 0.0508)
    assert_allclose(tilting_pad_determine.radial_clearance, 74.9e-6)
    assert_allclose(tilting_pad_determine.frequency, 314.1592653589793)
    assert_allclose(tilting_pad_determine.reference_temperature, 40)
    assert_allclose(tilting_pad_determine.n_pad, 5)


def test_tilting_pad_equilibrium_pos(tilting_pad_match):
    """
    Test equilibrium position for match_eccentricity case.

    Validates eccentricity, attitude angle, and pad tilting angles (psi_pad)
    against expected values with 1% tolerance for position and 10% for angles.

    """
    assert_allclose(tilting_pad_match.eccentricity, 0.35, rtol=0.01)
    assert_allclose(tilting_pad_match.attitude_angle, 5.017821599483698, rtol=0.01)

    expected_angles = [0.00107421, 0.00072079, 0.00029369, 0.00034969, 0.00081604]
    for i, expected_angle in enumerate(expected_angles):
        assert_allclose(tilting_pad_match.psi_pad[i], expected_angle, rtol=0.1)


def test_tilting_pad_equilibrium_pos_determine(tilting_pad_determine):
    """
    Test equilibrium position for determine_eccentricity case.

    Validates converged eccentricity, attitude angle, and pad tilting angles.
    Expected values obtained from numerical convergence.

    """
    assert_allclose(tilting_pad_determine.eccentricity, 0.3366, rtol=0.01)
    assert_allclose(tilting_pad_determine.attitude_angle, 5.744921616394679, rtol=0.01)

    expected_angles = [
        6.43383963e-05,
        -9.61893982e-05,
        -2.99569549e-06,
        2.62173313e-04,
        5.28712308e-04,
    ]
    for i, expected_angle in enumerate(expected_angles):
        assert_allclose(tilting_pad_determine.psi_pad[i], expected_angle, rtol=0.1)


def test_tilting_pad_coefficients(tilting_pad_match):
    """
    Test dynamic coefficients for match_eccentricity case.

    Validates stiffness coefficients (kxx, kxy, kyx, kyy) in N/m and
    damping coefficients (cxx, cxy, cyx, cyy) in N·s/m.

    """
    # Stiffness coefficients
    assert_allclose(tilting_pad_match.kxx, 1.06151681e08, rtol=0.001)
    assert_allclose(tilting_pad_match.kxy, -1.59240211e07, rtol=0.001)
    assert_allclose(tilting_pad_match.kyx, -1.59240211e07, rtol=0.001)
    assert_allclose(tilting_pad_match.kyy, 1.32123081e08, rtol=0.001)

    # Damping coefficients
    assert_allclose(tilting_pad_match.cxx, 355116.7381765, rtol=0.001)
    assert_allclose(tilting_pad_match.cxy, -43069.39477648, rtol=0.001)
    assert_allclose(tilting_pad_match.cyx, -43069.39477648, rtol=0.001)
    assert_allclose(tilting_pad_match.cyy, 439616.8694803, rtol=0.001)


def test_tilting_pad_coefficients_determine(tilting_pad_determine):
    """
    Test dynamic coefficients for determine_eccentricity case.

    Validates stiffness and damping coefficients obtained from
    thermo-hydrodynamic analysis.

    """
    # Stiffness coefficients
    assert_allclose(tilting_pad_determine.kxx, 50345502.66521944, rtol=0.01)
    assert_allclose(tilting_pad_determine.kxy, -88355785.34408838, rtol=0.01)
    assert_allclose(tilting_pad_determine.kyx, -88355785.34408846, rtol=0.01)
    assert_allclose(tilting_pad_determine.kyy, 94978188.16410919, rtol=0.01)

    # Damping coefficients
    assert_allclose(tilting_pad_determine.cxx, 142936.07610921, rtol=0.01)
    assert_allclose(tilting_pad_determine.cxy, -95219.03385087, rtol=0.01)
    assert_allclose(tilting_pad_determine.cyx, -95219.03385087, rtol=0.01)
    assert_allclose(tilting_pad_determine.cyy, 301134.63820087, rtol=0.01)


def test_tilting_pad_forces(tilting_pad_match):
    """
    Test dimensional forces for match_eccentricity case.

    Validates hydrodynamic forces in x and y directions for each pad.
    Forces are in Newtons.

    """
    expected_force_x = np.array(
        [-9.22582848e02, 3.02996821e-01, 5.51054460e02, 1.04734115e03, -1.57342924e03]
    )
    expected_force_y = np.array(
        [-300.86140003, -420.36660054, -178.8695415, 1442.6020094, 2161.92734728]
    )
    assert_allclose(tilting_pad_match.force_x_dim, expected_force_x, rtol=0.01)
    assert_allclose(tilting_pad_match.force_y_dim, expected_force_y, rtol=0.01)


def test_tilting_pad_forces_determine(tilting_pad_determine):
    """
    Test dimensional forces for determine_eccentricity case.

    Validates hydrodynamic forces obtained from thermo-hydrodynamic
    equilibrium solution. Forces are in Newtons.

    """
    expected_force_x = np.array(
        [-37.08619363, -0.0, 65.94292621, 526.88848024, -1439.89082258]
    )
    expected_force_y = np.array(
        [-12.05267279, -0.0, -21.42637395, 725.59974753, 1979.6378051]
    )
    assert_allclose(tilting_pad_determine.force_x_dim, expected_force_x, rtol=0.01)
    assert_allclose(tilting_pad_determine.force_y_dim, expected_force_y, rtol=0.01)