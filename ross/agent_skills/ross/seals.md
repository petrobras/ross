# Seals

Source: `docs/user_guide/tutorial_seals.ipynb`, `docs/theory_and_code/labyrinth_seal.ipynb`

Three physics-based seal models are available, all `SealElement` subclasses that
drop straight into `Rotor`: `LabyrinthSeal` (multi-tooth throttling seal),
`HolePatternSeal` (bulk-flow annular seal with hole/honeycomb cells) and
`HybridSeal` (hole-pattern + labyrinth in series with mass-flow matching at the
interface). Seals carry no static load: `run_static` ignores them and
`run_level1` removes them before applying the cross-coupled stiffness.

## Labyrinth Seal

```python
import ross as rs
from ross.units import Q_

laby = rs.LabyrinthSeal(
    n=3,
    shaft_diameter=Q_(145, "mm"),
    radial_clearance=Q_(0.3, "mm"),
    n_teeth=16,
    pitch=Q_(3.175, "mm"),
    tooth_height=Q_(3.175, "mm"),
    tooth_width=Q_(0.1524, "mm"),
    seal_type="inter",  # "rotor", "stator" or "inter" (interlocking)
    inlet_pressure=308_000,  # Pa
    outlet_pressure=94_300,
    inlet_temperature=283.15,  # K
    speed=Q_([5000, 8000, 11000], "RPM"),  # rotor speeds of the table
    preswirl=0.98,
    gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},
)
```

- `speed` (array): rotor speeds of the coefficient table (rad/s). The base flow
  (leakage, cavity pressures, swirl) is solved once per speed
- Without `gas_composition`, pass `molar_mass`, `gamma`, `reference_temperatures`
  (two K values) and `reference_viscosities` (two Pa·s values); `gas_model="real"`
  uses an equation of state along the inlet isentrope (needs `gas_composition`)
- `laby.kxx`, `laby.kxy`, `laby.cxx`, ... are plain lists aligned with `speed`
  (`kyy = kxx`, `kyx = -kxy`, `cyy = cxx`, `cyx = -cxy`)
- `laby.seal_leakage`: total mass flow in kg/s, one value per speed
- `laby.format_table(frequency_units="RPM")`, `laby.plot(["kxx", "kxy"])`,
  `laby.plot_pressure_distribution(pressure_units="bar")`

## Hole-Pattern Seal

```python
holep = rs.HolePatternSeal(
    n=3,
    shaft_diameter=0.145,
    radial_clearance=0.0003,
    axial_length=0.04699,
    relative_roughness=0.0001,
    cell_length=0.003175,
    cell_width=0.003175,
    cell_depth=0.0025,
    inlet_pressure=689_000,
    outlet_pressure=94_300,
    inlet_temperature=322.0,
    speed=Q_([8000], "RPM"),
    gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},
    preswirl=0.8,
    entrance_loss_coefficient=0.5,
    exit_loss_coefficient=1.0,
    nz=18,  # axial discretization (default 80; 18 keeps examples fast)
)
```

- Also returns mass coefficients (`mxx`, `mxy`, ...) from the fluid inertia
- `excitation_ratio` (default 1): whirl-to-speed ratio used for the 1-D table,
  i.e. the coefficients are evaluated at the whirl frequency `excitation_ratio * speed`
- Without `gas_composition`, pass `molar_mass`, `gamma`, `sutherland_b`, `sutherland_s`

## Hybrid Seal

```python
hybrid = rs.HybridSeal(
    n=3,
    shaft_diameter=Q_(50, "mm"),
    inlet_pressure=500_000,
    outlet_pressure=100_000,
    inlet_temperature=300.0,
    speed=Q_([2000, 3000, 5000], "RPM"),
    gas_composition={"Nitrogen": 0.7812, "Oxygen": 0.2096, "Argon": 0.0092},
    hole_pattern_parameters={
        "radial_clearance": 0.0003,
        "axial_length": 0.04,
        "relative_roughness": 0.0001,
        "cell_length": 0.003,
        "cell_width": 0.003,
        "cell_depth": 0.002,
        "preswirl": 0.8,
        "entrance_loss_coefficient": 0.5,
        "exit_loss_coefficient": 1.0,
        "nz": 18,
    },
    labyrinth_parameters={
        "radial_clearance": Q_(0.25, "mm"),
        "n_teeth": 10,
        "pitch": Q_(3, "mm"),
        "tooth_height": Q_(3, "mm"),
        "tooth_width": Q_(0.15, "mm"),
        "seal_type": "inter",
        "preswirl": 0.9,
    },
)
hybrid.summary_results()  # iterations, interface pressure, leakage
fig = hybrid.plot_convergence()
```

The interface pressure is found by bisection so that both stages pass the same
mass flow; the combined coefficients are the sum of the two stages
(`hybrid.laby`, `hybrid.hole_pattern`).

## 2-D Tables: Rotor Speed vs Whirl Frequency

The synchronous tables above evaluate the coefficients at whirl frequency =
rotor speed. Gas-seal forces depend on both: the base flow (swirl, leakage) is
set by the **rotor speed** and the unsteady reaction by the **whirl (excitation)
frequency**. Pass `frequency=` to any of the three classes to build a 2-D table:

```python
speed = Q_([5000, 8000, 11000], "RPM").to("rad/s").m
whirl = Q_([2000, 4000, 8000, 12000], "RPM").to("rad/s").m  # strictly increasing

laby_2d = rs.LabyrinthSeal(
    n=3,
    shaft_diameter=Q_(145, "mm"),
    radial_clearance=Q_(0.3, "mm"),
    n_teeth=16,
    pitch=Q_(3.175, "mm"),
    tooth_height=Q_(3.175, "mm"),
    tooth_width=Q_(0.1524, "mm"),
    seal_type="inter",
    inlet_pressure=308_000,
    outlet_pressure=94_300,
    inlet_temperature=283.15,
    speed=speed,
    preswirl=0.98,
    frequency=whirl,  # whirl frequencies -> 2-D table
    gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},
)

import numpy as np

np.array(laby_2d.kxy).shape  # (len(speed), len(frequency))
laby_2d.kxx_interpolated.kind  # "grid"
laby_2d.kxy_interpolated(frequency=whirl[0], speed=speed[1])  # bilinear lookup
laby_2d.kxy_interpolated(speed[1])  # single value -> synchronous diagonal
laby_2d.seal_leakage  # base flow: one value per speed, independent of whirl
```

- Each speed solves the base flow once and the perturbation once per whirl
  frequency, so a grid costs about `len(speed)` base flows plus
  `len(speed) * len(frequency)` perturbation solves
- The point `frequency == speed` reproduces the synchronous (1-D) result exactly;
  for `HolePatternSeal` the point `frequency == excitation_ratio * speed`
  reproduces the 1-D table
- `HybridSeal(frequency=...)` matches the interface pressure with the synchronous
  stages and then rebuilds both stages with 2-D tables at that pressure
- Both axes must be strictly increasing; lookups outside the grid extrapolate
  linearly

## Rotor Analyses with a 2-D Seal

```python
steel = rs.Material(name="steel", rho=7810, E=211e9, G_s=81.2e9)
shaft = [rs.ShaftElement(L=0.25, idl=0, odl=0.145, material=steel) for _ in range(6)]
disks = [
    rs.DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.145, o_d=0.45),
    rs.DiskElement.from_geometry(n=4, material=steel, width=0.07, i_d=0.145, o_d=0.45),
]
bearings = [
    rs.BearingElement(n=0, kxx=5e7, kyy=6e7, cxx=2e4, cyy=2e4),
    rs.BearingElement(n=6, kxx=5e7, kyy=6e7, cxx=2e4, cyy=2e4),
]
rotor = rs.Rotor(shaft, disks, bearings + [laby_2d])

w = Q_(8000, "RPM").to("rad/s").m

# 1) synchronous coefficients (frequency = speed) -- the classic result
modal_sync = rotor.run_modal(speed=w)

# 2) coefficients at a fixed whirl frequency for every mode
modal_fixed = rotor.run_modal(speed=w, frequency=0.5 * w)

# 3) each mode at its own damped natural frequency (fixed-point iteration)
modal_matched = rotor.run_modal(speed=w, matched_whirl=True)
modal_matched.whirl_frequency  # converged whirl frequency per mode (== wd)
modal_matched.log_dec  # compare with modal_sync.log_dec

# Campbell diagram with matched-whirl coefficients at every speed
speed_range = np.linspace(0.5 * w, 1.4 * w, 10)
campbell = rotor.run_campbell(speed_range, frequencies=4, matched_whirl=True)
fig = campbell.plot(harmonics=[1])
```

- `frequency` and `matched_whirl` are mutually exclusive; `whirl_rtol` (default
  1e-3) and `whirl_max_iter` (default 15) control the iteration
- The existing `synchronous=True` flag is unrelated: it folds the gyroscopic
  matrix into the mass matrix (Rouch's formulation)
- Subsynchronous stability: destabilizing modes whirl near the first natural
  frequency, well below the running speed, so `matched_whirl=True` (or a
  matching `frequency=`) gives the relevant log decrement; the synchronous
  lookup evaluates the seal at the running speed instead
- 1-D (`speed=` only) seals are constant with respect to the whirl frequency, so
  all three calls give the same result for them

See [modal_analysis.md](modal_analysis.md) and
[campbell_diagram.md](campbell_diagram.md) for the analysis options and
[bearings_advanced.md](bearings_advanced.md) for the coefficient table kinds.
