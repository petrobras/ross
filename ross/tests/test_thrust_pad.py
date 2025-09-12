import pytest
from numpy.testing import assert_allclose

from ross.bearings.thrust_pad import ThrustPad
from ross.units import Q_

@pytest.fixture
def thrust_pad():
    frequency = Q_([90], "RPM")
    pad_inner_radius = Q_(1150, "mm")
    pad_outer_radius = Q_(1725, "mm")
    pad_pivot_radius = Q_(1442.5, "mm")
    pad_arc_length = Q_(26, "deg")
    angular_pivot_position = Q_(15, "deg")
    oil_supply_temperature = Q_(40, "degC")
    radial_inclination_angle = Q_(-2.75e-04, "rad")
    circumferential_inclination_angle = Q_(-1.70e-05, "rad")
    initial_film_thickness = Q_(0.2, "mm")
    
    bearing = ThrustPad(
        n=1,
        pad_inner_radius=pad_inner_radius,
        pad_outer_radius=pad_outer_radius,
        pad_pivot_radius=pad_pivot_radius,
        pad_arc_length=pad_arc_length,
        angular_pivot_position=angular_pivot_position,
        oil_supply_temperature=oil_supply_temperature,
        lubricant="ISOVG68",
        n_pad=12,
        n_theta=10,
        n_radial=10,
        frequency=frequency,
        equilibrium_position_mode="calculate",
        model_type="thermo-hydro-dynamic",
        fzs_load=13.320e6,
        radial_inclination_angle=radial_inclination_angle,
        circumferential_inclination_angle=circumferential_inclination_angle,
        initial_film_thickness=initial_film_thickness,
        print_result=False,
        print_progress=False,
        print_time=False,
        compare_coefficients=False,
    )

    return bearing


def test_thrust_pad_parameters(thrust_pad):
    assert_allclose(thrust_pad.pad_inner_radius, 1.15, rtol=0.0001)
    assert_allclose(thrust_pad.pad_outer_radius, 1.725, rtol=0.0001)
    assert_allclose(thrust_pad.pad_pivot_radius, 1.4425, rtol=0.0001)
    assert_allclose(thrust_pad.frequency[0], 9.42477796, rtol=0.0001)
    assert_allclose(thrust_pad.rho, 867.0, rtol=0.0001)
    assert_allclose(thrust_pad.reference_temperature, 40.0, rtol=0.0001)


def test_thrust_pad_equilibrium_pos(thrust_pad):
    assert_allclose(thrust_pad.pivot_film_thickness, 0.000131, rtol=0.01)
    assert_allclose(thrust_pad.radial_inclination_angle, -2.75e-04, rtol=0.01)
    assert_allclose(thrust_pad.circumferential_inclination_angle, -1.70e-05, rtol=0.01)


def test_thrust_pad_coefficients(thrust_pad):
    assert_allclose(thrust_pad.kzz[0], 317633126111.6322, rtol=0.01)
    assert_allclose(thrust_pad.czz[0], 10805941381.16573, rtol=0.01)


def test_thrust_pad_field_results(thrust_pad):
    # Test pressure field results
    assert thrust_pad.pressure_field_dimensional is not None
    assert thrust_pad.pressure_field_dimensional.shape == (thrust_pad.n_radial + 2, thrust_pad.n_theta + 2)
    assert_allclose(thrust_pad.pressure_field_dimensional.max(), 6957021.42, rtol=0.01)
    
    # Test temperature field results
    assert thrust_pad.temperature_field is not None
    assert thrust_pad.temperature_field.shape == (thrust_pad.n_radial + 2, thrust_pad.n_theta + 2)
    assert_allclose(thrust_pad.temperature_field.max(), 70.4, rtol=0.01)
    
    # Test film thickness results
    assert_allclose(thrust_pad.max_thickness, 0.000207, rtol=0.01)
    assert_allclose(thrust_pad.min_thickness, 0.000082, rtol=0.01)
    assert_allclose(thrust_pad.pivot_film_thickness, 0.000131, rtol=0.01)

def test_thrust_pad_load(thrust_pad):
    assert_allclose(thrust_pad.fzs_load, 13320000.0, rtol=0.0001)
