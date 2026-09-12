# Advanced Bearings

Source: `docs/user_guide/tutorial_bearings_part_1.ipynb`, `docs/user_guide/tutorial_bearings_part_2.ipynb`

## Coefficient Tables: Speed, Frequency, or Both

A `BearingElement` declares the physical axis of its coefficient table through
the axis keyword it receives:

- `speed=` — a table over the **rotor speed** (the base flow). This is what
  fluid-film bearings and seals produce; it is interpolated at the rotor speed
  and is constant with respect to the excitation (whirl) frequency.
- `frequency=` — a table over the **excitation (whirl) frequency**. Used by the
  elements that react to the vibration frequency (`SqueezeFilmDamper`,
  `MagneticBearingElement`); constant with respect to the rotor speed.
- `speed=` and `frequency=` together with 2-D arrays of shape
  `(len(speed), len(frequency))` — a grid interpolated on both axes (linear,
  extrapolating; both axes strictly increasing).

In synchronous analyses (the default of every `run_*` method) speed and
frequency are the same value, so the three kinds give the same results.

```python
import ross as rs
import numpy as np

speed = np.array([0, 500, 1000])  # rad/s
kxx = np.array([1e6, 1.5e6, 2e6])  # N/m
kyy = np.array([0.8e6, 1.2e6, 1.6e6])
cxx = np.array([100, 150, 200])  # N·s/m
cyy = np.array([80, 120, 160])

brg = rs.BearingElement(
    n=0,
    kxx=kxx,
    kyy=kyy,
    cxx=cxx,
    cyy=cyy,
    speed=speed,
)
```

Scalars are broadcast to the axes; arrays must match the axes lengths. A 2-D table:

```python
whirl = np.array([100, 300, 600])  # rad/s
kxx_2d = np.array([[1.0e6, 1.1e6, 1.3e6], [1.5e6, 1.6e6, 1.9e6], [2.0e6, 2.2e6, 2.6e6]])

brg_2d = rs.BearingElement(
    n=0,
    kxx=kxx_2d,
    cxx=100,  # scalar -> constant over the whole grid
    speed=speed,
    frequency=whirl,
)
```

Each coefficient is a `BearingCoefficient` reachable as `brg.<coeff>_interpolated`:

```python
brg.kxx  # the table as a plain (nested) list -- wrap in np.array for arithmetic
brg.kxx_interpolated.kind  # "constant", "speed", "frequency" or "grid"
brg.kxx_interpolated(750.0)  # single value -> synchronous diagonal (speed == frequency)
brg_2d.kxx_interpolated(frequency=200.0, speed=750.0)  # full (frequency, speed) lookup
brg_2d.K(frequency=200.0, speed=750.0)  # element matrices take the same pair
brg_2d.format_table(speed=[500], frequency=[100, 300])
brg_2d.plot("kxx")  # one curve per speed against the frequency axis
```

`run_modal(speed=w)` evaluates every table at `w` on both axes; see
[modal_analysis.md](modal_analysis.md) for evaluating the frequency axis at a
fixed or per-mode whirl frequency.

## Cross-Coupled Coefficients

```python
brg = rs.BearingElement(
    n=0,
    kxx=1e6,
    kyy=1e6,
    kxy=5e4,
    kyx=-5e4,  # cross-coupled stiffness (N/m)
    cxx=100,
    cyy=100,
    cxy=10,
    cyx=-10,  # cross-coupled damping (N·s/m)
)
```

## Seal Elements

Seals are modeled similarly to bearings but represent fluid-film forces in seals:

```python
seal = rs.SealElement(
    n=3,
    kxx=1e5,
    kyy=1e5,
    kxy=2e4,
    kyx=-2e4,
    cxx=50,
    cyy=50,
)
```

Physics-based seals (`LabyrinthSeal`, `HolePatternSeal`, `HybridSeal`) are
covered in [seals.md](seals.md).

## Specialized Bearing Types

```python
# Ball bearing (stiffness from geometry)
ball = rs.BallBearingElement(n=0, n_balls=8, d_balls=0.01, fs=500, alpha=0.3)

# Roller bearing
roller = rs.RollerBearingElement(n=0, n_rollers=12, l_rollers=0.02, fs=500, alpha=0.0)
```

## Fluid-Film Journal Bearings

Hydrodynamic journal bearings solve a thermo-hydrodynamic (THD) model by
default and produce speed-dependent stiffness and damping coefficients
automatically; pad/pivot elasticity (full TEHD) is opt-in via `deform_type`. All classes are `BearingElement` subclasses — pass them
straight to `Rotor`.

```python
import ross as rs
from ross.units import Q_

# Plain cylindrical bearing (two axial grooves)
plain = rs.PlainJournal(
    n=3,
    pad_axial_length=0.263,
    journal_diameter=0.4,
    radial_clearance=1.95e-4,
    n_pads=2,
    pad_arc=Q_(176, "deg"),
    oil_supply_temperature=Q_(50, "degC"),
    speed=Q_([900, 1200], "RPM"),
    fys_load=-112815,
    lubricant="ISOVG32",
    oil_flow_v=Q_(30, "l/min"),
)

# Tilting-pad bearing (5 pads, load between pads)
tpb = rs.TiltingPad(
    n=1,
    speed=Q_([3000], "RPM"),
    equilibrium_type="match_load",
    fxs_load=884.05,
    fys_load=-2670.4,
    journal_diameter=101.6e-3,
    radial_clearance=74.9e-6,
    pad_thickness=12.7e-3,
    pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
    pad_arc=Q_([60] * 5, "deg"),
    pad_axial_length=[50.8e-3] * 5,
    preload=[0.5] * 5,
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
    speed=Q_([3000], "RPM"),
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

### Whirl-Dependent (2-D) Tables

One engine case runs per entry of `speed`, reducing the film to the 2x2
matrices at whirl ratio `excitation_ratio` (default 1, synchronous). Passing
`frequency=` as well runs one case per `(speed, frequency)` pair with whirl
ratio `frequency / speed` (`excitation_ratio` is ignored; speeds must be
nonzero) and stores a 2-D table:

```python
tpb_2d = rs.TiltingPad(
    n=1,
    speed=Q_([2000, 3000], "RPM"),
    frequency=Q_([500, 1500, 3000], "RPM"),  # whirl frequencies
    equilibrium_type="match_load",
    fxs_load=884.05,
    fys_load=-2670.4,
    journal_diameter=101.6e-3,
    radial_clearance=74.9e-6,
    pad_thickness=12.7e-3,
    pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
    pad_arc=Q_([60] * 5, "deg"),
    pad_axial_length=[50.8e-3] * 5,
    preload=[0.5] * 5,
    offset=[0.5] * 5,
    lubricant="ISOVG32",
    oil_supply_temperature=Q_(40, "degC"),
    oil_flow_v=Q_(10, "l/min"),
    num_processes=4,  # 6 engine cases here
)
np.array(tpb_2d.kxx).shape  # (2, 3)
```

Tilting pads condense the pad degrees of freedom at the whirl frequency, so
their reduced coefficients genuinely change along the frequency axis; a rigid
fixed-geometry film gives the same coefficients at every whirl ratio.

Useful knobs and post-processing:

- `lubricant`: a key of `rs.lubricants_dict` (`"ISOVG32"`, `"ISOVG46"`, `"ISOVG68"`, ...)
- `thermal_type`: `None` (isoviscous), `"adiabatic"` or `"full"` (pad conduction)
- `deform_type`: `None` (rigid pads, default) or one of the `"pad_mechanical*"` options for pad/pivot elasticity (full TEHD)
- `num_processes`: solve the speed table (or the `(speed, frequency)` grid) in parallel
- `bearing.coefficients(speed, frequency=None)` returns `(kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy)` interpolated at a rotor speed (and whirl frequency for 2-D tables)
- Plots: `plot_pressure_2d()`, `plot_pressure_3d()`, `plot_temperature_2d()`, `plot_film_temperature_3d()`, `plot_film_thickness_2d()`; `show_results()` prints a per-case summary table
- `plot_pad_temperature_3d()` draws the pads as real geometry colored by the solid pad conduction field, resolved through the pad thickness (`thermal_type="full"` only)
- `bearing.save(file)` stores the solved coefficient table, 2-D included (reloads as a plain `BearingElement`, no re-solve)

See `docs/user_guide/tutorial_bearings_part_2.ipynb` for the full tour.
