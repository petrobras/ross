import pytest
from numpy.testing import assert_allclose

from ross.bearings.plain_journal import PlainJournal
from ross.units import Q_


@pytest.fixture
def plain_journal_perturbation():
    """Fixture for PlainJournal with perturbation method"""
    frequency = Q_([900], "RPM")
    L = Q_(10.3600055944, "in")
    oil_flow = Q_(37.86, "l/min")

    bearing = PlainJournal(
        n=3,
        axial_length=L,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_circumferential=11,
        elements_axial=3,
        n_pad=2,
        pad_arc_length=176,
        preload=0,
        geometry="circular",
        reference_temperature=50,
        frequency=frequency,
        fxs_load=0,
        fys_load=-112814.91,
        groove_factor=[0.52, 0.48],
        lubricant="ISOVG32",
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        operating_type="flooded",
        oil_supply_pressure=0,
        oil_flow_v=oil_flow,
    )

    return bearing


@pytest.fixture
def plain_journal_lund():
    """Fixture for PlainJournal with lund method"""
    frequency = Q_([900], "RPM")
    L = Q_(10.3600055944, "in")
    oil_flow = Q_(37.86, "l/min")

    bearing = PlainJournal(
        n=3,
        axial_length=L,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_circumferential=11,
        elements_axial=3,
        n_pad=2,
        pad_arc_length=176,
        preload=0,
        geometry="circular",
        reference_temperature=50,
        frequency=frequency,
        fxs_load=0,
        fys_load=-112814.91,
        groove_factor=[0.52, 0.48],
        lubricant="ISOVG32",
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="lund",
        operating_type="flooded",
        oil_supply_pressure=0,
        oil_flow_v=oil_flow,
    )

    return bearing


def test_plain_journal_parameters_perturbation(plain_journal_perturbation):
    """Test basic parameters for perturbation method"""
    assert_allclose(plain_journal_perturbation.axial_length, 0.263144, rtol=0.0001)
    assert_allclose(plain_journal_perturbation.journal_radius, 0.2)
    assert_allclose(plain_journal_perturbation.frequency, 94.24777961)
    assert_allclose(plain_journal_perturbation.rho, 873.99629)
    assert_allclose(plain_journal_perturbation.reference_temperature, 50)


def test_plain_journal_parameters_lund(plain_journal_lund):
    """Test basic parameters for lund method"""
    assert_allclose(plain_journal_lund.axial_length, 0.263144, rtol=0.0001)
    assert_allclose(plain_journal_lund.journal_radius, 0.2)
    assert_allclose(plain_journal_lund.frequency, 94.24777961)
    assert_allclose(plain_journal_lund.rho, 873.99629)
    assert_allclose(plain_journal_lund.reference_temperature, 50)


def test_plain_journal_equilibrium_pos_perturbation(plain_journal_perturbation):
    """Test equilibrium position for perturbation method"""
    assert_allclose(
        plain_journal_perturbation.equilibrium_pos[0], 0.68733194, rtol=0.01
    )
    assert_allclose(
        plain_journal_perturbation.equilibrium_pos[1], -0.79394211, rtol=0.01
    )


def test_plain_journal_equilibrium_pos_lund(plain_journal_lund):
    """Test equilibrium position for lund method"""
    assert_allclose(plain_journal_lund.equilibrium_pos[0], 0.68733194, rtol=0.01)
    assert_allclose(plain_journal_lund.equilibrium_pos[1], -0.79394211, rtol=0.01)


def test_plain_journal_coefficients_perturbation(plain_journal_perturbation):
    """Test coefficients for perturbation method"""
    frequency = Q_(900, "RPM")
    coeffs = plain_journal_perturbation.coefficients(frequency)
    kxx, kxy, kyx, kyy = coeffs[0]
    cxx, cxy, cyx, cyy = coeffs[1]

    assert_allclose(kxx, 2497674531.1749372, rtol=0.0001)
    assert_allclose(kxy, 783937669.6587772, rtol=0.0001)
    assert_allclose(kyx, -3140562821.5290236, rtol=0.0001)
    assert_allclose(kyy, 2562440911.734241, rtol=0.0001)
    assert_allclose(cxx, 36950674.61976142, rtol=0.0001)
    assert_allclose(cxy, -37265296.2322692, rtol=0.0001)
    assert_allclose(cyx, -42642543.712838694, rtol=0.0001)
    assert_allclose(cyy, 100992315.0043159, rtol=0.0001)


def test_plain_journal_coefficients_lund(plain_journal_lund):
    """Test coefficients for lund method"""
    frequency = Q_(900, "RPM")
    coeffs = plain_journal_lund.coefficients(frequency)
    kxx, kxy, kyx, kyy = coeffs[0]
    cxx, cxy, cyx, cyy = coeffs[1]

    assert_allclose(kxx, 947508775.2790189, rtol=0.0001)
    assert_allclose(kxy, 156786732.38415018, rtol=0.0001)
    assert_allclose(kyx, -2006480985.1711535, rtol=0.0001)
    assert_allclose(kyy, 2165272905.1621084, rtol=0.0001)
    assert_allclose(cxx, 11502933.224462334, rtol=0.0001)
    assert_allclose(cxy, -13040765.779177427, rtol=0.0001)
    assert_allclose(cyx, -13051009.888004456, rtol=0.0001)
    assert_allclose(cyy, 40600798.873926796, rtol=0.0001)
