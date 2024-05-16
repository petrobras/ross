"""This module deals with lubricants dictionary in the ROSS library."""

from ross.units import Q_

lubricant_dict = {
    "ISOVG32": {
        "viscosity1": Q_(4.05640e-06, "reyn").to_base_units().m,
        "temp1": Q_(40.00000, "degC").to_base_units().m,
        "viscosity2": Q_(6.76911e-07, "reyn").to_base_units().m,
        "temp2": Q_(100.00000, "degC").to_base_units().m,
        "lube_density": Q_(873.99629, "kg/m³").to_base_units().m,
        "lube_cp": Q_(1948.7995685758851, "J/(kg*degK)").to_base_units().m,
        "lube_conduct": Q_(0.13126, "W/(m*degC)").to_base_units().m,
    },
    "ISOVG46": {
        "viscosity1": Q_(5.757040938820288e-06, "reyn").to_base_units().m,
        "temp1": Q_(40, "degC").to_base_units().m,
        "viscosity2": Q_(8.810775697672788e-07, "reyn").to_base_units().m,
        "temp2": Q_(100, "degC").to_base_units().m,
        "lube_density": Q_(862.9, "kg/m³").to_base_units().m,
        "lube_cp": Q_(1950, "J/(kg*degK)").to_base_units().m,
        "lube_conduct": Q_(0.15, "W/(m*degC)").to_base_units().m,
    },
    "TEST": {
        "viscosity1": Q_(0.04, "Pa*s").to_base_units().m,
        "temp1": Q_(40, "degC").to_base_units().m,
        "viscosity2": Q_(0.01, "Pa*s").to_base_units().m,
        "temp2": Q_(100, "degC").to_base_units().m,
        "lube_density": Q_(863.61302696, "kg/m³").to_base_units().m,
        "lube_cp": Q_(1951.88616, "J/(kg*degK)").to_base_units().m,
        "lube_conduct": Q_(0.15, "W/(m*degC)").to_base_units().m,
    },
    "ISOVG68": {
        "viscosity1": Q_(0.0570722, "Pa*s").to_base_units().m,
        "temp1": Q_(40, "degC").to_base_units().m,
        "viscosity2": Q_(0.00766498, "Pa*s").to_base_units().m,
        "temp2": Q_(100, "degC").to_base_units().m,
        "lube_density": Q_(867, "kg/m³").to_base_units().m,
        "lube_cp": Q_(1951, "J/(kg*degK)").to_base_units().m,
        "lube_conduct": Q_(0.15, "W/(m*degC)").to_base_units().m,
    },
}
