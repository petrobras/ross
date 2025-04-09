import pickle
from collections import namedtuple

import pytest
from numpy.testing import assert_allclose

from ross.units import Q_, check_units


def test_new_units_loaded():
    speed = Q_(1, "RPM")
    assert speed.m == 1
    # check if h is hour instead of planck constant
    v = Q_(3600, "m/h")
    assert v.to("m/s").m == 1


def test_units_pickle():
    speed = Q_(1, "RPM")
    speed_pickled = pickle.loads(pickle.dumps(speed))
    assert speed == speed_pickled


# each possible argument
Argument = namedtuple(
    "Argument", ["single_value", "si_unit", "other_unit", "expected_converted_value"]
)
arguments = {
    "E": Argument(1, "N/m²", "lbf/in²", 6894.7572931683635),
    "G_s": Argument(1, "N/m²", "lbf/in²", 6894.7572931683635),
    "rho": Argument(1, "kg/m³", "lb/foot**3", 16.01846337396014),
    "L": Argument(1, "meter", "inches", 0.0254),
    "idl": Argument(1, "meter", "inches", 0.0254),
    "idr": Argument(1, "meter", "inches", 0.0254),
    "odl": Argument(1, "meter", "inches", 0.0254),
    "odr": Argument(1, "meter", "inches", 0.0254),
    "speed": Argument(1, "rad/s", "RPM", 0.10471975511965977),
    "frequency": Argument(1, "rad/s", "RPM", 0.10471975511965977),
    "frequency_range": Argument(
        (1, 1), "rad/s", "RPM", (0.10471975511965977, 0.10471975511965977)
    ),
    "m": Argument(1, "kg", "lb", 0.4535923700000001),
    "mx": Argument(1, "kg", "lb", 0.4535923700000001),
    "my": Argument(1, "kg", "lb", 0.4535923700000001),
    "Ip": Argument(1, "kg*m²", "lb*in²", 0.0002926396534292),
    "Id": Argument(1, "kg*m²", "lb*in²", 0.0002926396534292),
    "width": Argument(1, "meter", "inches", 0.0254),
    "depth": Argument(1, "meter", "inches", 0.0254),
    "pitch": Argument(1, "meter", "inches", 0.0254),
    "height": Argument(1, "meter", "inches", 0.0254),
    "shaft_radius": Argument(1, "meter", "inches", 0.0254),
    "radial_clearance": Argument(1, "meter", "inches", 0.0254),
    "i_d": Argument(1, "meter", "inches", 0.0254),
    "o_d": Argument(1, "meter", "inches", 0.0254),
    "length": Argument(1, "meter", "inches", 0.0254),
    "distance": Argument(1, "meter", "inches", 0.0254),
    "area": Argument(1, "meter**2", "inches**2", 0.00064516),
    "unbalance_magnitude": Argument(1, "kg*m", "lb*in", 0.011521246198000002),
    "unbalance_phase": Argument(1, "rad", "deg", 0.017453292519943295),
    "inlet_pressure": Argument(1, "pascal", "kgf/cm²", 98066.5),
    "outlet_pressure": Argument(1, "pascal", "kgf/cm²", 98066.5),
    "inlet_temperature": Argument(1, "degK", "degC", 274.15),
    "inlet_swirl_velocity": Argument(1, "m/s", "ft/s", 0.3048),
    "pad_thickness": Argument(1, "meter", "inches", 0.0254),
    "kxx": Argument(1, "N/m", "lbf/in", 175.12683524647645),
    "kxy": Argument(1, "N/m", "lbf/in", 175.12683524647645),
    "kxz": Argument(1, "N/m", "lbf/in", 175.12683524647645),
    "kyx": Argument(1, "N/m", "lbf/in", 175.12683524647645),
    "kyy": Argument(1, "N/m", "lbf/in", 175.12683524647645),
    "kyz": Argument(1, "N/m", "lbf/in", 175.12683524647645),
    "kzx": Argument(1, "N/m", "lbf/in", 175.12683524647645),
    "kzy": Argument(1, "N/m", "lbf/in", 175.12683524647645),
    "kzz": Argument(1, "N/m", "lbf/in", 175.12683524647645),
    "cxx": Argument(1, "N*s/m", "lbf*s/in", 175.12683524647645),
    "cxy": Argument(1, "N*s/m", "lbf*s/in", 175.12683524647645),
    "cxz": Argument(1, "N*s/m", "lbf*s/in", 175.12683524647645),
    "cyx": Argument(1, "N*s/m", "lbf*s/in", 175.12683524647645),
    "cyy": Argument(1, "N*s/m", "lbf*s/in", 175.12683524647645),
    "cyz": Argument(1, "N*s/m", "lbf*s/in", 175.12683524647645),
    "czx": Argument(1, "N*s/m", "lbf*s/in", 175.12683524647645),
    "czy": Argument(1, "N*s/m", "lbf*s/in", 175.12683524647645),
    "czz": Argument(1, "N*s/m", "lbf*s/in", 175.12683524647645),
    "viscosity": Argument(1, "Pa*s", "lb/ft/s", 1.488164),
    "weight": Argument(1, "N", "lbf", 4.4482216152605005),
    "load": Argument(1, "N", "lbf", 4.4482216152605005),
    "force": Argument(1, "N", "lbf", 4.4482216152605005),
    "torque": Argument(1, "N*m", "lbf*ft", 1.3558179483314006),
}


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
        length,
        distance,
        area,
        unbalance_magnitude,
        unbalance_phase,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        inlet_swirl_velocity,
        pad_thickness,
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
        viscosity,
        weight,
        load,
        force,
        torque,
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
            length,
            distance,
            area,
            unbalance_magnitude,
            unbalance_phase,
            inlet_pressure,
            outlet_pressure,
            inlet_temperature,
            inlet_swirl_velocity,
            pad_thickness,
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
            viscosity,
            weight,
            load,
            force,
            torque,
        )

    return func


def test_units(auxiliary_function):
    results = auxiliary_function(**{k: v.single_value for k, v in arguments.items()})
    results_dict = {k: v for k, v in zip(arguments.keys(), results)}
    for arg, actual in results_dict.items():
        assert_allclose(actual, arguments[arg].single_value)


def test_unit_Q_(auxiliary_function):
    kwargs = {k: Q_(v.single_value, v.si_unit) for k, v in arguments.items()}
    results = auxiliary_function(**kwargs)
    results_dict = {k: v for k, v in zip(arguments.keys(), results)}
    for arg, actual in results_dict.items():
        assert_allclose(actual, arguments[arg].single_value)


def test_unit_Q_conversion(auxiliary_function):
    kwargs = {k: Q_(v.single_value, v.other_unit) for k, v in arguments.items()}
    results = auxiliary_function(**kwargs)
    results_dict = {k: v for k, v in zip(arguments.keys(), results)}
    for arg, actual in results_dict.items():
        assert_allclose(actual, arguments[arg].expected_converted_value)
