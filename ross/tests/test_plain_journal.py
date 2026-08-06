import pytest
from numpy.testing import assert_allclose

from ross.bearings.plain_journal import PlainJournal
from ross.units import Q_


@pytest.fixture
def plain_journal_circular_perturbation():
    """PlainJournal fixture for a circular bearing using the perturbation method"""
    frequency = Q_([900], "RPM")
    L = Q_(10.3600055944, "in")

    bearing = PlainJournal(
        n=3,
        geometry="circular",
        axial_length=L,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_circumferential=11,
        elements_axial=3,
        n_pad=2,
        pad_arc_length=176,
        reference_temperature=50,
        frequency=frequency,
        fxs_load=0,
        fys_load=-112814.91,
        hot_oil_factor=[0.48, 0.52],
        lubricant="ISOVG32",
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        operating_type="flooded",
        oil_supply_pressure=0,
    )

    return bearing


@pytest.fixture
def plain_journal_circular_lund():
    """PlainJournal fixture for a circular bearing using the Lund method"""
    frequency = Q_([900], "RPM")
    L = Q_(10.3600055944, "in")

    bearing = PlainJournal(
        n=3,
        geometry="circular",
        axial_length=L,
        journal_radius=0.2,
        radial_clearance=1.95e-4,
        elements_circumferential=11,
        elements_axial=3,
        n_pad=2,
        pad_arc_length=176,
        reference_temperature=50,
        frequency=frequency,
        fxs_load=0,
        fys_load=-112814.91,
        hot_oil_factor=[0.48, 0.52],
        lubricant="ISOVG32",
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="lund",
        operating_type="flooded",
        oil_supply_pressure=0,
    )

    return bearing


@pytest.fixture
def plain_journal_offset_perturbation():
    """PlainJournal fixture for an offset halves bearing using the perturbation method"""
    frequency = Q_([12649], "RPM")

    bearing = PlainJournal(
        n=3,
        geometry="lobe",
        axial_length=140e-3,
        journal_radius=70e-3,
        radial_clearance=140e-6,
        elements_circumferential=11,
        elements_axial=3,
        n_pad=2,
        pad_arc_length=165,
        preload=0.349,
        offset=1,
        rotation_angle=17.5,
        reference_temperature=57,
        frequency=frequency,
        fxs_load=-21338,
        fys_load=57772,
        hot_oil_factor=[0.3, 0.3],
        lubricant="ISOVG46",
        sommerfeld_type=2,
        initial_guess=[0.5, 1.571],
        method="perturbation",
        operating_type="flooded",
        oil_supply_pressure=0,
    )

    return bearing


def test_plain_journal_circular_parameters_perturbation(
    plain_journal_circular_perturbation,
):
    """Verifies the initialization of input parameters for a circular bearing simulation based on the perturbation method"""
    assert_allclose(
        plain_journal_circular_perturbation.axial_length, 0.263144, rtol=0.0001
    )
    assert_allclose(plain_journal_circular_perturbation.journal_radius, 0.2)
    assert_allclose(plain_journal_circular_perturbation.frequency, 94.24777961)
    assert_allclose(plain_journal_circular_perturbation.rho, 873.99629)
    assert_allclose(plain_journal_circular_perturbation.reference_temperature, 50)


def test_plain_journal_circular_parameters_lund(plain_journal_circular_lund):
    """Verifies the initialization of input parameters for a circular bearing simulation based on the Lund method"""
    assert_allclose(plain_journal_circular_lund.axial_length, 0.263144, rtol=0.0001)
    assert_allclose(plain_journal_circular_lund.journal_radius, 0.2)
    assert_allclose(plain_journal_circular_lund.frequency, 94.24777961)
    assert_allclose(plain_journal_circular_lund.rho, 873.99629)
    assert_allclose(plain_journal_circular_lund.reference_temperature, 50)


def test_plain_journal_circular_equilibrium_pos_perturbation(
    plain_journal_circular_perturbation,
):
    """Verifies the equilibrium position computed by a circular bearing simulation based on the perturbation method"""
    assert_allclose(
        plain_journal_circular_perturbation.equilibrium_pos[0], 0.68362988, rtol=0.01
    )
    assert_allclose(
        plain_journal_circular_perturbation.equilibrium_pos[1], -0.79073692, rtol=0.01
    )


def test_plain_journal_circular_equilibrium_pos_lund(plain_journal_circular_lund):
    """Verifies the equilibrium position computed by a circular bearing simulation based on the Lund method"""
    assert_allclose(
        plain_journal_circular_lund.equilibrium_pos[0], 0.68362988, rtol=0.01
    )
    assert_allclose(
        plain_journal_circular_lund.equilibrium_pos[1], -0.79073692, rtol=0.01
    )


def test_plain_journal_offset_equilibrium_pos_perturbation(
    plain_journal_offset_perturbation,
):
    """Verifies the equilibrium position computed by an offset halves bearing simulation based on the perturbation method"""
    assert_allclose(
        plain_journal_offset_perturbation.equilibrium_pos[0], 0.85859691, rtol=0.01
    )
    assert_allclose(
        plain_journal_offset_perturbation.equilibrium_pos[1], 2.22633916, rtol=0.01
    )


def test_plain_journal_circular_coefficients_perturbation(
    plain_journal_circular_perturbation,
):
    """Verifies the dynamic coefficients computed by a circular bearing simulation based on the perturbation method"""
    frequency = Q_(900, "RPM")
    coeffs = plain_journal_circular_perturbation.coefficients(frequency)
    kxx, kxy, kyx, kyy = coeffs[0]
    cxx, cxy, cyx, cyy = coeffs[1]

    assert_allclose(kxx, 1075500985.171117, rtol=0.0001)
    assert_allclose(kxy, 339393154.12713265, rtol=0.0001)
    assert_allclose(kyx, -1929530769.525935, rtol=0.0001)
    assert_allclose(kyy, 1566137995.7410197, rtol=0.0001)
    assert_allclose(cxx, 15856164.24177218, rtol=0.0001)
    assert_allclose(cxy, -15890940.232336765, rtol=0.0001)
    assert_allclose(cyx, -18214986.865570847, rtol=0.0001)
    assert_allclose(cyy, 43404619.73486264, rtol=0.0001)


def test_plain_journal_circular_coefficients_lund(plain_journal_circular_lund):
    """Verifies the dynamic coefficients computed by a circular bearing simulation based on the Lund method"""
    frequency = Q_(900, "RPM")
    coeffs = plain_journal_circular_lund.coefficients(frequency)
    kxx, kxy, kyx, kyy = coeffs[0]
    cxx, cxy, cyx, cyy = coeffs[1]

    assert_allclose(kxx, 1068973932.3633764, rtol=0.0001)
    assert_allclose(kxy, 380806906.870878, rtol=0.0001)
    assert_allclose(kyx, -2149004717.7513924, rtol=0.0001)
    assert_allclose(kyy, 1867770345.440784, rtol=0.0001)
    assert_allclose(cxx, 16042019.32504611, rtol=0.0001)
    assert_allclose(cxy, -16222640.780704116, rtol=0.0001)
    assert_allclose(cyx, -18644123.85471422, rtol=0.0001)
    assert_allclose(cyy, 44512317.73285746, rtol=0.0001)


def test_plain_journal_offset_coefficients_perturbation(
    plain_journal_offset_perturbation,
):
    """Verifies the dynamic coefficients computed by an offset halves bearing simulation based on the perturbation method"""
    frequency = Q_(12649, "RPM")
    coeffs = plain_journal_offset_perturbation.coefficients(frequency)
    kxx, kxy, kyx, kyy = coeffs[0]
    cxx, cxy, cyx, cyy = coeffs[1]

    assert_allclose(kxx, 863841320.6631312, rtol=0.0001)
    assert_allclose(kxy, -38331944.50214172, rtol=0.0001)
    assert_allclose(kyx, -1049189588.14207, rtol=0.0001)
    assert_allclose(kyy, 612441747.1291603, rtol=0.0001)
    assert_allclose(cxx, 798029.9099911602, rtol=0.0001)
    assert_allclose(cxy, -701213.3056542532, rtol=0.0001)
    assert_allclose(cyx, -761789.3511112307, rtol=0.0001)
    assert_allclose(cyy, 1072583.6162213983, rtol=0.0001)
