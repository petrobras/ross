"""This module deals with units conversion in the ROSS library."""
import inspect
import warnings
from functools import wraps
from pathlib import Path

from pint import Quantity, UnitRegistry

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])

new_units_path = Path(__file__).parent / "new_units.txt"
ureg = UnitRegistry()
ureg.load_definitions(str(new_units_path))
Q_ = ureg.Quantity

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
    "area": "meter**2",
    "unbalance_magnitude": "kg*m",
    "unbalance_phase": "rad",
    "pressure": "pascal",
    "temperature": "degK",
    "velocity": "m/s",
    "angle": "rad",
    "arc": "rad",
    "convection": "W/(m²*degK)",
    "conductivity": "W/(m*degK)",
    "expansion": "1/degK",
    "stiffness": "N/m",
    "weight": "N",
    "load": "N",
    "flowv": "m³/s",
    "fit": "m",
}
for i, unit in zip(["k", "c"], ["N/m", "N*s/m"]):
    for j in ["x", "y", "z"]:
        for k in ["x", "y", "z"]:
            units["".join([i, j, k])] = unit


def check_units(func):
    """Wrapper to check and convert units to base_units.

    If we use the check_units decorator in a function the arguments are checked,
    and if they are in the dictionary, they are converted to the 'default' unit given
    in the dictionary.

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
            if arg_name not in names:
                names.append(arg_name)
            for name in names:
                if name in units and arg_value is not None:
                    # For now, we only return the magnitude for the converted Quantity
                    # If pint is fully adopted by ross in the future, and we have all Quantities
                    # using it, we could remove this, which would allows us to use pint in its full capability
                    try:
                        base_unit_args.append(arg_value.to(units[name]).m)
                    except AttributeError:
                        base_unit_args.append(Q_(arg_value, units[name]).m)
                    break
            else:
                base_unit_args.append(arg_value)

        base_unit_kwargs = {}
        for k, v in kwargs.items():
            names = k.split("_")
            if k not in names:
                names.append(k)
            for name in names:
                if name in units and v is not None:
                    try:
                        base_unit_kwargs[k] = v.to(units[name]).m
                    except AttributeError:
                        base_unit_kwargs[k] = Q_(v, units[name]).m
                    break
            else:
                base_unit_kwargs[k] = v

        return func(*base_unit_args, **base_unit_kwargs)

    return inner
