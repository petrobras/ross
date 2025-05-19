from ross.seals.labyrinth_seal import LabyrinthSeal
from ross.units import Q_

seal = LabyrinthSeal(
    n=0,
    inlet_pressure=308000,
    outlet_pressure=94300,
    inlet_temperature=283.15,
    pre_swirl_ratio=0.98,
    frequency=Q_([8000], "RPM"),
    n_teeth=16,
    shaft_radius=Q_(72.5,"mm"),
    radial_clearance=Q_(0.3,"mm"),
    pitch=Q_(3.175,"mm"),
    tooth_height=Q_(3.175,"mm"),
    tooth_width=Q_(0.1524,"mm"),
    seal_type="inter",
)

# seal = LabyrinthSeal(
#     n=0,
#     inlet_pressure=6500000.0,
#     outlet_pressure=1500000.0,
#     inlet_temperature=434.82,
#     pre_swirl_ratio=0.2,
#     frequency=Q_([4000], "RPM"),
#     n_teeth=14,
#     shaft_radius=Q_(271 / 2, "mm"),
#     radial_clearance=Q_(270 / 2, "microm"),
#     pitch=Q_(5,"mm"),
#     tooth_height=Q_(4.3,"mm"),
#     tooth_width=Q_(0.2,"mm"),
#     seal_type="STATOR",
#     gas_composition={
#              "METHANE": 58.976,
#              "ETHANE": 3.099,
#              "PROPANE": 0.6,
#              "N-BUTANE": 0.08,
#              "ISOBUTANE": 0.05,
#              "PENTANE": 0.01,
#              "ISOPENTANE": 0.01,
#              "N2": 0.55,
#              "H2S": 0.02,
#              "CO2": 36.605,
#          },
# )
print(seal)