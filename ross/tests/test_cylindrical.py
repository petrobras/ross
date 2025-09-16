import pytest
from numpy.testing import assert_allclose

from ross.bearings.cylindrical import PlainJournal
from ross.units import Q_


@pytest.fixture
def cylindrical():
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
        show_coeffs=False,
        print_result=False,
        print_progress=False,
        print_time=False,
    )

    return bearing


def test_cylindrical_parameters(cylindrical):
    assert_allclose(cylindrical.axial_length, 0.263144, rtol=0.0001)
    assert_allclose(cylindrical.journal_radius, 0.2)
    assert_allclose(cylindrical.frequency, 94.24777961)
    assert_allclose(cylindrical.rho, 873.99629)
    assert_allclose(cylindrical.reference_temperature, 50)


def test_cylindrical_equilibrium_pos(cylindrical):
    assert_allclose(cylindrical.equilibrium_pos[0], 0.68733194, rtol=0.01)
    assert_allclose(cylindrical.equilibrium_pos[1], -0.79394211, rtol=0.01)


def test_cylindrical_coefficients(cylindrical):
    frequency = Q_(900, "RPM")
    coeffs = cylindrical.coefficients(frequency)
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
