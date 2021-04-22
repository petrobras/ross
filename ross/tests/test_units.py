import pytest
from numpy.testing import assert_allclose

from ross.units import Q_, check_units, units


def test_new_units_loaded():
    speed = Q_(1, "RPM")
    assert speed.magnitude == 1


@pytest.fixture
def auxiliary_function():
    @check_units
    def func(
        E,
        G_s,
        rho,
        L,
        idl,
        idr,
        odl,
        odr,
        speed,
        frequency,
        frequency_range,
        m,
        mx,
        my,
        Ip,
        Id,
        width,
        depth,
        pitch,
        height,
        shaft_radius,
        radial_clearance,
        i_d,
        o_d,
        unbalance_magnitude,
        unbalance_phase,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        inlet_swirl_velocity,
        kxx,
        kxy,
        kxz,
        kyx,
        kyy,
        kyz,
        kzx,
        kzy,
        kzz,
        cxx,
        cxy,
        cxz,
        cyx,
        cyy,
        cyz,
        czx,
        czy,
        czz,
    ):
        return (
            E,
            G_s,
            rho,
            L,
            idl,
            idr,
            odl,
            odr,
            speed,
            frequency,
            frequency_range,
            m,
            mx,
            my,
            Ip,
            Id,
            width,
            depth,
            pitch,
            height,
            shaft_radius,
            radial_clearance,
            i_d,
            o_d,
            unbalance_magnitude,
            unbalance_phase,
            inlet_pressure,
            outlet_pressure,
            inlet_temperature,
            inlet_swirl_velocity,
            kxx,
            kxy,
            kxz,
            kyx,
            kyy,
            kyz,
            kzx,
            kzy,
            kzz,
            cxx,
            cxy,
            cxz,
            cyx,
            cyy,
            cyz,
            czx,
            czy,
            czz,
        )

    return func


def test_units(auxiliary_function):
    results = auxiliary_function(
        E=1,
        G_s=1,
        rho=1,
        L=1,
        idl=1,
        idr=1,
        odl=1,
        odr=1,
        speed=1,
        frequency=1,
        frequency_range=(1, 1),
        m=1,
        mx=1,
        my=1,
        Ip=1,
        Id=1,
        width=1,
        depth=1,
        pitch=1,
        height=1,
        shaft_radius=1,
        radial_clearance=1,
        i_d=1,
        o_d=1,
        unbalance_magnitude=1,
        unbalance_phase=1,
        inlet_pressure=1,
        outlet_pressure=1,
        inlet_temperature=1,
        inlet_swirl_velocity=1,
        kxx=1,
        kxy=1,
        kxz=1,
        kyx=1,
        kyy=1,
        kyz=1,
        kzx=1,
        kzy=1,
        kzz=1,
        cxx=1,
        cxy=1,
        cxz=1,
        cyx=1,
        cyy=1,
        cyz=1,
        czx=1,
        czy=1,
        czz=1,
    )
    # check if all available units are tested
    assert len(results) == len(units)

    (
        E,
        G_s,
        rho,
        L,
        idl,
        idr,
        odl,
        odr,
        speed,
        frequency,
        frequency_range,
        m,
        mx,
        my,
        Ip,
        Id,
        width,
        depth,
        pitch,
        height,
        shaft_radius,
        radial_clearance,
        i_d,
        o_d,
        unbalance_magnitude,
        unbalance_phase,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        inlet_swirl_velocity,
        kxx,
        kxy,
        kxz,
        kyx,
        kyy,
        kyz,
        kzx,
        kzy,
        kzz,
        cxx,
        cxy,
        cxz,
        cyx,
        cyy,
        cyz,
        czx,
        czy,
        czz,
    ) = results

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
    assert_allclose(frequency_range, (1, 1))
    assert m == 1
    assert mx == 1
    assert my == 1
    assert Ip == 1
    assert Id == 1
    assert width == 1
    assert depth == 1
    assert pitch == 1
    assert height == 1
    assert shaft_radius == 1
    assert radial_clearance == 1
    assert i_d == 1
    assert o_d == 1
    assert unbalance_magnitude == 1
    assert unbalance_phase == 1
    assert inlet_pressure == 1
    assert outlet_pressure == 1
    assert inlet_temperature == 1
    assert inlet_swirl_velocity == 1
    assert kxx == 1
    assert kxy == 1
    assert kxz == 1
    assert kyx == 1
    assert kyy == 1
    assert kyz == 1
    assert kzx == 1
    assert kzy == 1
    assert kzz == 1
    assert cxx == 1
    assert cxy == 1
    assert cxz == 1
    assert cyx == 1
    assert cyy == 1
    assert cyz == 1
    assert czx == 1
    assert czy == 1
    assert czz == 1


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
        frequency_range=Q_((1, 1), "radian/second"),
        m=Q_(1, "kg"),
        mx=Q_(1, "kg"),
        my=Q_(1, "kg"),
        Ip=Q_(1, "kg*m**2"),
        Id=Q_(1, "kg*m**2"),
        width=Q_(1, "meter"),
        depth=Q_(1, "meter"),
        pitch=Q_(1, "meter"),
        height=Q_(1, "meter"),
        shaft_radius=Q_(1, "meter"),
        radial_clearance=Q_(1, "meter"),
        i_d=Q_(1, "meter"),
        o_d=Q_(1, "meter"),
        unbalance_magnitude=Q_(1, "kg*m"),
        unbalance_phase=Q_(1, "rad"),
        inlet_pressure=Q_(1, "pascal"),
        outlet_pressure=Q_(1, "pascal"),
        inlet_temperature=Q_(1, "degK"),
        inlet_swirl_velocity=Q_(1, "m/s"),
        kxx=Q_(1, "N/m"),
        kxy=Q_(1, "N/m"),
        kxz=Q_(1, "N/m"),
        kyx=Q_(1, "N/m"),
        kyy=Q_(1, "N/m"),
        kyz=Q_(1, "N/m"),
        kzx=Q_(1, "N/m"),
        kzy=Q_(1, "N/m"),
        kzz=Q_(1, "N/m"),
        cxx=Q_(1, "N*s/m"),
        cxy=Q_(1, "N*s/m"),
        cxz=Q_(1, "N*s/m"),
        cyx=Q_(1, "N*s/m"),
        cyy=Q_(1, "N*s/m"),
        cyz=Q_(1, "N*s/m"),
        czx=Q_(1, "N*s/m"),
        czy=Q_(1, "N*s/m"),
        czz=Q_(1, "N*s/m"),
    )

    # check if all available units are tested
    assert len(results) == len(units)
    (
        E,
        G_s,
        rho,
        L,
        idl,
        idr,
        odl,
        odr,
        speed,
        frequency,
        frequency_range,
        m,
        mx,
        my,
        Ip,
        Id,
        width,
        depth,
        pitch,
        height,
        shaft_radius,
        radial_clearance,
        i_d,
        o_d,
        unbalance_magnitude,
        unbalance_phase,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        inlet_swirl_velocity,
        kxx,
        kxy,
        kxz,
        kyx,
        kyy,
        kyz,
        kzx,
        kzy,
        kzz,
        cxx,
        cxy,
        cxz,
        cyx,
        cyy,
        cyz,
        czx,
        czy,
        czz,
    ) = results

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
    assert_allclose(frequency_range, (1, 1))
    assert m == 1
    assert mx == 1
    assert my == 1
    assert Ip == 1
    assert Id == 1
    assert width == 1
    assert depth == 1
    assert pitch == 1
    assert height == 1
    assert shaft_radius == 1
    assert radial_clearance == 1
    assert i_d == 1
    assert o_d == 1
    assert unbalance_magnitude == 1
    assert unbalance_phase == 1
    assert inlet_pressure == 1
    assert outlet_pressure == 1
    assert inlet_temperature == 1
    assert inlet_swirl_velocity == 1
    assert kxx == 1
    assert kxy == 1
    assert kxz == 1
    assert kyx == 1
    assert kyy == 1
    assert kyz == 1
    assert kzx == 1
    assert kzy == 1
    assert kzz == 1
    assert cxx == 1
    assert cxy == 1
    assert cxz == 1
    assert cyx == 1
    assert cyy == 1
    assert cyz == 1
    assert czx == 1
    assert czy == 1
    assert czz == 1


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
        frequency_range=Q_((1, 1), "RPM"),
        m=Q_(1, "lb"),
        mx=Q_(1, "lb"),
        my=Q_(1, "lb"),
        Ip=Q_(1, "lb*in**2"),
        Id=Q_(1, "lb*in**2"),
        width=Q_(1, "inches"),
        depth=Q_(1, "inches"),
        pitch=Q_(1, "inches"),
        height=Q_(1, "inches"),
        shaft_radius=Q_(1, "inches"),
        radial_clearance=Q_(1, "inches"),
        i_d=Q_(1, "inches"),
        o_d=Q_(1, "inches"),
        unbalance_magnitude=Q_(1, "lb*in"),
        unbalance_phase=Q_(1, "deg"),
        inlet_pressure=Q_(1, "kgf/cm²"),
        outlet_pressure=Q_(1, "kgf/cm²"),
        inlet_temperature=Q_(1, "degC"),
        inlet_swirl_velocity=Q_(1, "ft/s"),
        kxx=Q_(1, "lbf/in"),
        kxy=Q_(1, "lbf/in"),
        kxz=Q_(1, "lbf/in"),
        kyx=Q_(1, "lbf/in"),
        kyy=Q_(1, "lbf/in"),
        kyz=Q_(1, "lbf/in"),
        kzx=Q_(1, "lbf/in"),
        kzy=Q_(1, "lbf/in"),
        kzz=Q_(1, "lbf/in"),
        cxx=Q_(1, "lbf*s/in"),
        cxy=Q_(1, "lbf*s/in"),
        cxz=Q_(1, "lbf*s/in"),
        cyx=Q_(1, "lbf*s/in"),
        cyy=Q_(1, "lbf*s/in"),
        cyz=Q_(1, "lbf*s/in"),
        czx=Q_(1, "lbf*s/in"),
        czy=Q_(1, "lbf*s/in"),
        czz=Q_(1, "lbf*s/in"),
    )

    # check if all available units are tested
    assert len(results) == len(units)

    (
        E,
        G_s,
        rho,
        L,
        idl,
        idr,
        odl,
        odr,
        speed,
        frequency,
        frequency_range,
        m,
        mx,
        my,
        Ip,
        Id,
        width,
        depth,
        pitch,
        height,
        shaft_radius,
        radial_clearance,
        i_d,
        o_d,
        unbalance_magnitude,
        unbalance_phase,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        inlet_swirl_velocity,
        kxx,
        kxy,
        kxz,
        kyx,
        kyy,
        kyz,
        kzx,
        kzy,
        kzz,
        cxx,
        cxy,
        cxz,
        cyx,
        cyy,
        cyz,
        czx,
        czy,
        czz,
    ) = results

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
    assert_allclose(frequency_range, (0.10471975511965977, 0.10471975511965977))
    assert m == 0.4535923700000001
    assert mx == 0.4535923700000001
    assert my == 0.4535923700000001
    assert Ip == 0.0002926396534292
    assert Id == 0.0002926396534292
    assert width == 0.0254
    assert depth == 0.0254
    assert i_d == 0.0254
    assert o_d == 0.0254
    assert kxx == 175.12683524647645
    assert kxy == 175.12683524647645
    assert kxz == 175.12683524647645
    assert kyx == 175.12683524647645
    assert kyy == 175.12683524647645
    assert kyz == 175.12683524647645
    assert kzx == 175.12683524647645
    assert kzy == 175.12683524647645
    assert kzz == 175.12683524647645
    assert cxx == 175.12683524647645
    assert cxy == 175.12683524647645
    assert cxz == 175.12683524647645
    assert cyx == 175.12683524647645
    assert cyy == 175.12683524647645
    assert cyz == 175.12683524647645
    assert czx == 175.12683524647645
    assert czy == 175.12683524647645
    assert czz == 175.12683524647645
    assert unbalance_magnitude == 0.011521246198000002
    assert unbalance_phase == 0.017453292519943295


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
