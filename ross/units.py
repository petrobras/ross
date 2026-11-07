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

__all__ = ["Q_", "check_units", "format_unit"]

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
    "mz": "kg",
    "Ip": "kg*m**2",
    "Id": "kg*m**2",
    "Ixx": "kg*m**2",
    "Iyy": "kg*m**2",
    "Izz": "kg*m**2",
    "k_x": "N/m",
    "k_y": "N/m",
    "k_z": "N/m",
    "kr_x": "N*m/rad",
    "kr_y": "N*m/rad",
    "kr_z": "N*m/rad",
    "ct_x": "N*s/m",
    "ct_y": "N*s/m",
    "ct_z": "N*s/m",
    "cr_x": "N*m*s/rad",
    "cr_y": "N*m*s/rad",
    "cr_z": "N*m*s/rad",
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
    "molar_mass": "kg/kmol",
    "h": "joule/kilogram",
    "s": "joule/(kelvin kilogram)",
    "b": "meter",
    "D": "meter",
    "d": "meter",
    "head": "joule/kilogram",
    "eff": "dimensionless",
    "power": "watt",
    "module": "meter",
    "mesh_damping_ratio": "dimensionless",
}
for i, unit in zip(["k", "c", "m"], ["N/m", "N*s/m", "kg"], strict=True):
    for j in ["x", "y", "z"]:
        for k in ["x", "y", "z"]:
            units["".join([i, j, k])] = unit

for i, unit in zip(["kr", "cr"], ["N*m/rad", "N*m*s/rad"], strict=True):
    for suffix in ["xx", "xy", "yx", "yy", "zz"]:
        units[f"{i}_{suffix}"] = unit


def _to_base_unit(value, unit):
    """Return the magnitude of value in unit.

    Strings are options such as "major" or "wd", not magnitudes, and are returned
    unchanged, as are values pint does not accept as a magnitude, such as bool.

    Parameters
    ----------
    value : float, array_like, str, pint.Quantity
        Argument passed to a function decorated with check_units. A value that is
        not a pint.Quantity is taken to be in unit already.
    unit : str
        Unit that ROSS uses for this argument internally.

    Returns
    -------
    float, array_like or str
        Magnitude in unit, or value unchanged.
    """
    if isinstance(value, str):
        return value
    # Only the magnitude is returned, since ROSS does not keep Quantities internally.
    try:
        return value.to(unit).m
    except AttributeError:
        try:
            return Q_(value, unit).m
        except TypeError:
            return value


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

        for arg_name, arg_value in zip(args_names, args, strict=False):
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
                    base_unit_args.append(_to_base_unit(arg_value, units[name]))
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
                    base_unit_kwargs[k] = _to_base_unit(v, units[name])
                    break
            else:
                base_unit_kwargs[k] = v

        return func(*base_unit_args, **base_unit_kwargs)

    return inner


def format_unit(unit):
    """Format a ``check_units`` unit string for a plot axis label.

    Parameters
    ----------
    unit : str
        Unit string as stored in the ``units`` dictionary used by
        ``check_units``, for example ``"N/m**2"`` or ``"radian/second"``.
        Any unit string accepted by pint also works.

    Returns
    -------
    label : str
        Compact label with short unit names, unicode exponents and a
        dot operator for multiplication.

    Examples
    --------
    >>> format_unit("N/m**2")
    'N/m²'
    >>> format_unit("kg*m**2")
    'kg⋅m²'
    >>> format_unit("N*m")
    'N⋅m'
    >>> format_unit("radian/second")
    'rad/s'
    >>> format_unit("meter")
    'm'
    >>> format_unit("dimensionless")
    '--'
    """
    if unit == "dimensionless":
        return "--"

    label = ureg.formatter.format_unit(
        Q_(1, unit).units,
        "~P",
        sort_func=lambda items, _registry: items,
    )

    # Older pint pretty-prints multiplication as MIDDLE DOT (U+00B7).
    # Keep DOT OPERATOR (U+22C5), which is also the decimal mark inside a
    # superscript exponent (s¹⋅⁵) and must not be rewritten.
    label = label.replace("\u00b7", "\u22c5")

    return label
