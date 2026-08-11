# Advanced Bearings

Source: `docs/user_guide/tutorial_bearings_part_1.ipynb`, `docs/user_guide/tutorial_bearings_part_2.ipynb`

## Speed-Dependent Coefficients

Bearing stiffness and damping that vary with rotor speed:

```python
import ross as rs
import numpy as np

frequency = np.array([0, 500, 1000])    # rad/s
kxx = np.array([1e6, 1.5e6, 2e6])      # N/m
kyy = np.array([0.8e6, 1.2e6, 1.6e6])
cxx = np.array([100, 150, 200])         # N·s/m
cyy = np.array([80, 120, 160])

brg = rs.BearingElement(
    n=0,
    kxx=kxx, kyy=kyy,
    cxx=cxx, cyy=cyy,
    frequency=frequency,
)
```

ROSS interpolates coefficients at the analysis frequency automatically. When using `run_modal(speed=w)`, bearing coefficients are evaluated at `w`.

## Cross-Coupled Coefficients

```python
brg = rs.BearingElement(
    n=0,
    kxx=1e6, kyy=1e6,
    kxy=5e4, kyx=-5e4,    # cross-coupled stiffness (N/m)
    cxx=100, cyy=100,
    cxy=10, cyx=-10,      # cross-coupled damping (N·s/m)
)
```

## Seal Elements

Seals are modeled similarly to bearings but represent fluid-film forces in seals:

```python
seal = rs.SealElement(
    n=3,
    kxx=1e5, kyy=1e5,
    kxy=2e4, kyx=-2e4,
    cxx=50, cyy=50,
)
```

## Specialized Bearing Types

```python
# Ball bearing (stiffness from geometry)
ball = rs.BallBearingElement(n=0, n_balls=8, d_balls=0.01, fs=500, alpha=0.3)

# Roller bearing
roller = rs.RollerBearingElement(n=0, n_rollers=12, l_rollers=0.02, fs=500, alpha=0.0)
```

## Fluid-Film Journal Bearings

Hydrodynamic journal bearings solve a thermo-elasto-hydro-dynamic (TEHD)
model and produce speed-dependent stiffness and damping coefficients
automatically. All classes are `BearingElement` subclasses — pass them
straight to `Rotor`.

```python
import ross as rs
from ross.units import Q_

# Plain cylindrical bearing (two axial grooves)
plain = rs.PlainJournal(
    n=3,
    axial_length=0.263,
    journal_radius=0.2,
    radial_clearance=1.95e-4,
    n_pad=2,
    pad_arc_length=Q_(176, "deg"),
    reference_temperature=Q_(50, "degC"),
    frequency=Q_([900, 1200], "RPM"),
    fys_load=-112815,
    lubricant="ISOVG32",
    oil_flow_v=Q_(30, "l/min"),
)

# Tilting-pad bearing (5 pads, load between pads)
tpb = rs.TiltingPad(
    n=1,
    frequency=Q_([3000], "RPM"),
    equilibrium_type="match_load",
    load=[884.05, -2670.4],
    journal_diameter=101.6e-3,
    radial_clearance=74.9e-6,
    pad_thickness=12.7e-3,
    pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
    pad_arc=Q_([60] * 5, "deg"),
    pad_axial_length=[50.8e-3] * 5,
    pre_load=[0.5] * 5,
    offset=[0.5] * 5,
    lubricant="ISOVG32",
    oil_supply_temperature=Q_(40, "degC"),
    oil_flow_v=Q_(10, "l/min"),
)
```

The classic fixed-geometry configurations have dedicated classes:
`PartialArcBearing`, `EllipticalBearing` (lemon bore), `OffsetHalvesBearing`,
`MultiLobeBearing`, `PressureDamBearing`. Arbitrary pad layouts (per-pad
preload, offset, pockets, tapers) use `FixedGeometryBearing` directly, and
`FluidFilmBearing` is the shared base with the full model-flag surface
(thermal model, turbulence, pivot flexibility, starvation, ...).

```python
lemon = rs.EllipticalBearing(
    n=0,
    frequency=Q_([3000], "RPM"),
    pad_arc=Q_(150, "deg"),
    preload=0.5,
    journal_diameter=0.2,
    radial_clearance=150e-6,
    pad_thickness=0.05,
    pad_axial_length=[0.16, 0.16],
    lubricant="ISOVG32",
    oil_supply_temperature=Q_(40, "degC"),
    oil_flow_v=Q_(30, "l/min"),
    weight=45e3,
)
```

Useful knobs and post-processing:

- `lubricant`: a key of `rs.lubricants_dict` (`"ISOVG32"`, `"ISOVG46"`, `"ISOVG68"`, ...)
- `thermal_type`: `None` (isoviscous), `"adiabatic"` or `"full"` (pad conduction)
- `num_processes`: solve the frequency table in parallel
- `bearing.coefficients(frequency)` returns `(kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy)` interpolated at any speed
- Plots: `plot_pressure_2d()`, `plot_pressure_3d()`, `plot_temperature_2d()`, `plot_film_temperature_3d()`, `plot_film_thickness_2d()`; `show_results()` prints a per-speed summary table
- `plot_pad_temperature_3d()` draws the pads as real geometry colored by the solid pad conduction field, resolved through the pad thickness (`thermal_type="full"` only)
- `bearing.save(file)` stores the solved coefficient table (reloads as a plain `BearingElement`, no re-solve)

See `docs/user_guide/tutorial_bearings_part_2.ipynb` for the full tour.
