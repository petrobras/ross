"""This module deals with materials dictionary in the ROSS library."""

from ross.units import Q_

materials_dict = {
    "steel": {
        "density": Q_(7850, "kg/m**3").to_base_units().m,
        "specific_heat": Q_(434, "J/(kg*degC)").to_base_units().m,
        "thermal_conductivity": Q_(60.5, "W/(m*degK)").to_base_units().m,
    },
    "brass": {
        "density": Q_(8600, "kg/m**3").to_base_units().m,
        "specific_heat": Q_(380, "J/(kg*degC)").to_base_units().m,
        "thermal_conductivity": Q_(109, "W/(m*degK)").to_base_units().m,
    },
}
