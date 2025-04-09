"""This module deals with units conversion in the ROSS library."""

import inspect
import warnings
from functools import wraps
from pathlib import Path

import pint

new_units_path = Path(__file__).parent / "new_units.txt"
ureg = pint.get_application_registry()
if isinstance(ureg.get(), pint.registry.LazyRegistry):
    ureg = pint.UnitRegistry()
    ureg.load_definitions(str(new_units_path))
    # set ureg to make pickle possible
    pint.set_application_registry(ureg)

Q_ = ureg.Quantity

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])

__all__ = ["Q_", "check_units"]

units = {
    "E": "N/m**2",
    "G_s": "N/m**2",
    "rho": "kg/m**3",
    "density": "kg/m**3",
    "L": "meter",
    "idl": "meter",
    "idr": "meter",
    "odl": "meter",
    "odr": "meter",
    "id": "meter",
    "od": "meter",
    "i_d": "meter",
    "o_d": "meter",
    "speed": "radian/second",
    "frequency": "radian/second",
    "m": "kg",
    "mx": "kg",
    "my": "kg",
    "Ip": "kg*m**2",
    "Id": "kg*m**2",
    "width": "meter",
    "depth": "meter",
    "thickness": "meter",
    "pitch": "meter",
    "height": "meter",
    "radius": "meter",
    "diameter": "meter",
    "clearance": "meter",
    "length": "meter",
    "distance": "meter",
    "area": "meter**2",
    "unbalance_magnitude": "kg*m",
    "unbalance_phase": "rad",
    "pressure": "pascal",
    "pressure_ratio": "dimensionless",
    "p": "pascal",
    "temperature": "degK",
    "T": "degK",
    "velocity": "m/s",
    "angle": "rad",
    "arc": "rad",
    "convection": "W/(m²*degK)",
    "conductivity": "W/(m*degK)",
    "expansion": "1/degK",
    "stiffness": "N/m",
    "damping": "N*s/m",
    "weight": "N",
    "load": "N",
    "force": "N",
    "torque": "N*m",
    "flow_v": "meter**3/second",
    "flow_m": "kilogram/second",
    "fit": "m",
    "viscosity": "pascal*s",
    "h": "joule/kilogram",
    "s": "joule/(kelvin kilogram)",
    "b": "meter",
    "D": "meter",
    "d": "meter",
    "roughness": "meter",
    "head": "joule/kilogram",
    "eff": "dimensionless",
    "power": "watt",
}
for i, unit in zip(["k", "c", "m"], ["N/m", "N*s/m", "kg"]):
    for j in ["x", "y", "z"]:
        for k in ["x", "y", "z"]:
            units["".join([i, j, k])] = unit


def check_units(func):
    """Wrapper to check and convert units to base_units.

    If we use the check_units decorator in a function the arguments are checked,
    and if they are in the dictionary, they are converted to the 'default' unit given
    in the dictionary.
    The check is carried out by splitting the argument name on '_', and checking
    if any of the names are in the dictionary. So an argument such as 'inlet_pressure',
    will be split into ['inlet', 'pressure'], and since we have the name 'pressure'
    in the dictionary mapped to 'Pa', we will automatically convert the value to
    this default unit.

    For example:
    >>> units = {
    ... "L": "meter",
    ... }

    >>> @check_units
    ... def foo(L=None):
    ...     print(L)
    ...

    If we call the function with the argument as a float:
    >>> foo(L=0.5)
    0.5

    If we call the function with a pint.Quantity object the value is automatically
    converted to the default:
    >>> foo(L=Q_(0.5, 'inches'))
    0.0127
    """

    @wraps(func)
    def inner(*args, **kwargs):
        base_unit_args = []
        args_names = inspect.getfullargspec(func)[0]

        for arg_name, arg_value in zip(args_names, args):
            names = arg_name.split("_")
            if "units" in names:
                base_unit_args.append(arg_value)
                continue

            # treat flow_v and flow_m separately
            if "flow_v" in arg_name:
                names.insert(0, "flow_v")
            if "flow_m" in arg_name:
                names.insert(0, "flow_m")

            if arg_name not in names:
                # check first for arg_name in units
                names.insert(0, arg_name)
            for name in names:
                if name in units and arg_value is not None:
                    # For now, we only return the magnitude for the converted Quantity
                    # If pint is fully adopted by ross in the future, and we have all Quantities
                    # using it, we could remove this, which would allows us to use pint in its full capability
                    try:
                        base_unit_args.append(arg_value.to(units[name]).m)
                    except AttributeError:
                        try:
                            base_unit_args.append(Q_(arg_value, units[name]).m)
                        except TypeError:
                            # Handle erros that we get with bool for example
                            base_unit_args.append(arg_value)
                    break
            else:
                base_unit_args.append(arg_value)

        base_unit_kwargs = {}
        for k, v in kwargs.items():
            names = k.split("_")
            if "units" in names:
                base_unit_kwargs[k] = v
                continue

            # treat flow_v and flow_m separately
            if "flow_v" in k:
                names.insert(0, "flow_v")
            if "flow_m" in k:
                names.insert(0, "flow_m")

            if k not in names:
                # check first for arg_name in units
                names.insert(0, k)
            for name in names:
                if name in units and v is not None:
                    try:
                        base_unit_kwargs[k] = v.to(units[name]).m
                    except AttributeError:
                        try:
                            base_unit_kwargs[k] = Q_(v, units[name]).m
                        except TypeError:
                            # Handle errors that we get with bool for example
                            base_unit_kwargs[k] = v
                    break
            else:
                base_unit_kwargs[k] = v

        return func(*base_unit_args, **base_unit_kwargs)

    return inner
