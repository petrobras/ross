import pytest
from numpy.testing import assert_allclose

from ross.units import Q_, check_units, units


def test_new_units_loaded():
    speed = Q_(1, "RPM")
    assert speed.magnitude == 1


@pytest.fixture
def auxiliary_function():
    @check_units
    def func(E, G_s, rho, L, idl, idr, odl, odr, speed, frequency):
        return E, G_s, rho, L, idl, idr, odl, odr, speed, frequency

    return func


def test_units(auxiliary_function):
    results = auxiliary_function(
        E=1, G_s=1, rho=1, L=1, idl=1, idr=1, odl=1, odr=1, speed=1, frequency=1
    )
    # check if all available units are tested
    assert len(results) == len(units)

    E, G_s, rho, L, idl, idr, odl, odr, speed, frequency = results

    assert E == 1
    assert G_s == 1
    assert rho == 1
    assert L == 1
    assert idl == 1
    assert idr == 1
    assert odl == 1
    assert odr == 1
    assert speed == 1
    assert frequency == 1


def test_unit_Q_(auxiliary_function):
    results = auxiliary_function(
        E=Q_(1, "N/m**2"),
        G_s=Q_(1, "N/m**2"),
        rho=Q_(1, "kg/m**3"),
        L=Q_(1, "meter"),
        idl=Q_(1, "meter"),
        idr=Q_(1, "meter"),
        odl=Q_(1, "meter"),
        odr=Q_(1, "meter"),
        speed=Q_(1, "radian/second"),
        frequency=Q_(1, "radian/second"),
    )

    # check if all available units are tested
    assert len(results) == len(units)

    E, G_s, rho, L, idl, idr, odl, odr, speed, frequency = results

    assert E == 1
    assert G_s == 1
    assert rho == 1
    assert L == 1
    assert idl == 1
    assert idr == 1
    assert odl == 1
    assert odr == 1
    assert speed == 1
    assert frequency == 1


def test_unit_Q_conversion(auxiliary_function):
    results = auxiliary_function(
        E=Q_(1, "lbf/in**2"),
        G_s=Q_(1, "lbf/in**2"),
        rho=Q_(1, "lb/foot**3"),
        L=Q_(1, "inches"),
        idl=Q_(1, "inches"),
        idr=Q_(1, "inches"),
        odl=Q_(1, "inches"),
        odr=Q_(1, "inches"),
        speed=Q_(1, "RPM"),
        frequency=Q_(1, "RPM"),
    )

    # check if all available units are tested
    assert len(results) == len(units)

    E, G_s, rho, L, idl, idr, odl, odr, speed, frequency = results

    assert E == 6894.7572931683635
    assert G_s == 6894.7572931683635
    assert_allclose(rho, 16.01846337396014)
    assert L == 0.0254
    assert idl == 0.0254
    assert idr == 0.0254
    assert odl == 0.0254
    assert odr == 0.0254
    assert speed == 0.10471975511965977
    assert frequency == 0.10471975511965977


# NOTE ABOUT TESTS BELOW
# For now, we only return the magnitude for the converted Quantity
# If pint is fully adopted by ross in the future, and we have all Quantities
# using it, we could remove this, which would allows us to use pint in its full capability
@pytest.mark.skip("Skip since we are not fully using pint")
def test_units_pint(auxiliary_function):
    results = auxiliary_function(
        E=1, G_s=1, rho=1, L=1, idl=1, idr=1, odl=1, odr=1, speed=1, frequency=1
    )
    # check if all available units are tested
    assert len(results) == len(units)

    E, G_s, rho, L, idl, idr, odl, odr, speed, frequency = results

    assert E.magnitude == 1
    assert E.units == "newton/meter**2"

    assert G_s.magnitude == 1
    assert G_s.units == "newton/meter**2"

    assert rho.magnitude == 1
    assert rho.units == "kilogram/meter**3"

    assert L.magnitude == 1
    assert L.units == "meter"

    assert idl.magnitude == 1
    assert idl.units == "meter"

    assert idr.magnitude == 1
    assert idr.units == "meter"

    assert odl.magnitude == 1
    assert odl.units == "meter"

    assert odr.magnitude == 1
    assert odr.units == "meter"

    assert speed.magnitude == 1
    assert speed.units == "radian/second"

    assert frequency.magnitude == 1
    assert frequency.units == "radian/second"


@pytest.mark.skip("Skip since we are not fully using pint")
def test_unit_Q__pint(auxiliary_function):
    results = auxiliary_function(
        E=Q_(1, "N/m**2"),
        G_s=Q_(1, "N/m**2"),
        rho=Q_(1, "kg/m**3"),
        L=Q_(1, "meter"),
        idl=Q_(1, "meter"),
        idr=Q_(1, "meter"),
        odl=Q_(1, "meter"),
        odr=Q_(1, "meter"),
        speed=Q_(1, "radian/second"),
        frequency=Q_(1, "radian/second"),
    )

    # check if all available units are tested
    assert len(results) == len(units)

    E, G_s, rho, L, idl, idr, odl, odr, speed, frequency = results

    assert E.magnitude == 1
    assert E.units == "newton/meter**2"

    assert G_s.magnitude == 1
    assert G_s.units == "newton/meter**2"

    assert rho.magnitude == 1
    assert rho.units == "kilogram/meter**3"

    assert L.magnitude == 1
    assert L.units == "meter"

    assert idl.magnitude == 1
    assert idl.units == "meter"

    assert idr.magnitude == 1
    assert idr.units == "meter"

    assert odl.magnitude == 1
    assert odl.units == "meter"

    assert odr.magnitude == 1
    assert odr.units == "meter"

    assert speed.magnitude == 1
    assert speed.units == "radian/second"

    assert frequency.magnitude == 1
    assert frequency.units == "radian/second"


@pytest.mark.skip("Skip since we are not fully using pint")
def test_unit_Q_conversion_pint(auxiliary_function):
    results = auxiliary_function(
        E=Q_(1, "lbf/in**2"),
        G_s=Q_(1, "lbf/in**2"),
        rho=Q_(1, "lb/foot**3"),
        L=Q_(1, "inches"),
        idl=Q_(1, "inches"),
        idr=Q_(1, "inches"),
        odl=Q_(1, "inches"),
        odr=Q_(1, "inches"),
        speed=Q_(1, "RPM"),
        frequency=Q_(1, "RPM"),
    )

    # check if all available units are tested
    assert len(results) == len(units)

    E, G_s, rho, L, idl, idr, odl, odr, speed, frequency = results

    assert E.magnitude == 6894.7572931683635
    assert E.units == "newton/meter**2"

    assert G_s.magnitude == 6894.7572931683635
    assert G_s.units == "newton/meter**2"

    assert_allclose(rho.magnitude, 16.01846337396014)
    assert rho.units == "kilogram/meter**3"

    assert L.magnitude == 0.0254
    assert L.units == "meter"

    assert idl.magnitude == 0.0254
    assert idl.units == "meter"

    assert idr.magnitude == 0.0254
    assert idr.units == "meter"

    assert odl.magnitude == 0.0254
    assert odl.units == "meter"

    assert odr.magnitude == 0.0254
    assert odr.units == "meter"

    assert speed.magnitude == 0.10471975511965977
    assert speed.units == "radian/second"

    assert frequency.magnitude == 0.10471975511965977
    assert frequency.units == "radian/second"
