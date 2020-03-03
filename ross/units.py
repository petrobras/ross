"""
This module deals with units conversion in the ROSS library. 
"""
import inspect
from pint import UnitRegistry
from functools import wraps
from pathlib import Path

new_units_path = Path(__file__).parent / "new_units.txt"
ureg = UnitRegistry()
ureg.load_definitions(str(new_units_path))
Q_ = ureg.Quantity

__all__ = ["Q_", "check_units"]

units = {
    "E": "N/m**2",
    "G_s": "N/m**2",
    "rho": "kg/m**3",
    "L": "meter",
    "idl": "meter",
    "idr": "meter",
    "odl": "meter",
    "odr": "meter",
    "speed": "radian/second",
    "frequency": "radian/second",
}


def check_units(func):
    """Wrapper to check and convert units to base_units.

    If we use the check_units decorator in a function the arguments are checked, and if they are in the units
    dictionary, they are converted to the 'default' unit given in the dictionary.

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

    If we call the function with a pint.Quantity object the value is automatically converted to the default:
    >>> foo(L=Q_(0.5, 'inches'))
    0.0127
    """

    @wraps(func)
    def inner(*args, **kwargs):
        base_unit_args = []
        args_names = inspect.getfullargspec(func)[0]

        for arg_name, arg_value in zip(args_names, args):
            if arg_name in units:
                # For now, we only return the magnitude for the converted Quantity
                # If pint is fully adopted by ross in the future, and we have all Quantities
                # using it, we could remove this, which would allows us to use pint in its full capability
                try:
                    base_unit_args.append(arg_value.to(units[arg_name]).m)
                except AttributeError:
                    base_unit_args.append(Q_(arg_value, units[arg_name]).m)
            else:
                base_unit_args.append(arg_value)

        base_unit_kwargs = {}
        for k, v in kwargs.items():
            if k in units and v is not None:
                try:
                    base_unit_kwargs[k] = v.to(units[k]).m
                except AttributeError:
                    base_unit_kwargs[k] = Q_(v, units[k]).m
            else:
                base_unit_kwargs[k] = v

        return func(*base_unit_args, **base_unit_kwargs)

    return inner
