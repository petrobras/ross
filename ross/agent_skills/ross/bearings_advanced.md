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
  `(len(speed), len(frequency))` — a grid interpolated on both axes (both axes
  strictly increasing).

Every table is interpolated with a shape-preserving cubic (PCHIP) along each
axis, which passes through the tabulated values without overshooting, and is
extrapolated linearly from the end slopes; `interpolation="linear"` on any
element selects piecewise-linear interpolation instead (useful to match codes
that interpolate linearly). Tabulate at least 5 speeds (and 5 whirl
frequencies for 2-D tables) spanning the analysis range: analyses warn when
they interpolate a sparser table or leave an axis.

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

One engine case runs per entry of `speed`, condensing the film and pad
coordinates to the 2x2 matrices at whirl ratio `excitation_ratio` (default 1,
synchronous; 0 gives the static condensation). Passing `frequency=` as well
condenses the state solved at each speed at every frequency (one solve per
speed, `excitation_ratio` is ignored; speeds must be nonzero) and stores a 2-D
table:

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
    n_jobs=2,  # warm-start chains solved in 2 worker processes
)
np.array(tpb_2d.kxx).shape  # (2, 3)
```

Tilting pads condense the pad degrees of freedom (tilt and, with pivot
flexibility, pivot displacement, with the pad mass and inertia about the pivot)
at the whirl frequency, so their reduced coefficients genuinely change along
the frequency axis; a rigid fixed-geometry film gives the same coefficients at
every whirl ratio. `bearing.results` keeps one entry per `(speed, frequency)`
pair, sharing the equilibrium of the speed, and the unreduced matrices of each
solved speed (`i` indexes `speed`):

```python
res = tpb_2d.results
K, C = res.coefficients(0, frequency=Q_(2200, "RPM").to("rad/s").m)  # 2x2, exact
K0, C0 = res.coefficients(0, frequency=0)  # static limit (pad tilts in equilibrium)
pads = res.pad_matrices(
    0
)  # PadMatrices: K, C, M per pad, labels, pad_mass, pad_inertia
D = pads.impedance(-20.0 + 300.0j)  # 2x2 complex D(s) at a damped eigenvalue s
```

`results.coefficients` condenses at any frequency without a new solve (the
element's `coefficients()` still interpolates its table, as the rotor does).
`impedance(s)` gives `D(s) = sum_k Z_uu - Z_ue Z_ee^-1 Z_eu` for complex `s`, the
bearing at a damped rotor eigenvalue (`run_modal(matched_whirl=True)` uses
`K(w) + i w C(w)` at the damped natural frequency, which is exact only for an
undamped mode).

Inputs are checked before any case is solved: a `ValueError` names the
argument for speeds that are not finite and positive, negative whirl
frequencies or `excitation_ratio`, non-positive geometry, density or oil flow,
`preload` outside `[0, 1)`, `offset` outside `[0, 1]`, pads that overlap
(leading edge at `pivot_angle - offset * pad_arc`) or cover more than 360 deg,
and pivot flexibility without its data (`pivot_stiffness > 0` for
`"user_specified_stiffness"`, `0 < pivot_diameter < house_diameter` for
`"ball_in_socket"` and `"button"`).

`TiltingPad` defaults to `equilibrium_type="match_eccentricity"`: the journal
is held at `eccentricity` / `attitude_angle` and the load is not used (a
`UserWarning` says so when a load is given). Pass
`equilibrium_type="match_load"` to solve the position for the load, as in the
examples above. `TiltingPad` also has its own thermal and mesh defaults
(`pad_conductivity=116`, `edges_convection=1500`, `relax_temperature=0.5`,
mesh 30/30/16); `FluidFilmBearing` uses 50, 73.59, 1 and 40/20/20.

Pressures are gauge, relative to the drain. `ambient_pressure_1/2` set the
film boundaries in `"axial_flow"`, `"high_ambient_pressure"` and the starved
and oil-ring modes. In every mode the inter-pad grooves carry their mean, which
adds a net force on the journal for an asymmetric pad layout. In the default
`"regular_flooded"` mode the axial film edges stay at 0 and a nonzero ambient
pressure warns.

### Convergence and operating results

Every case records whether its iterations converged (equilibrium force
balance `|F_net| / |F_ext| < 2e-3`, pad tilt, turbulence, temperatures,
deformation, pivot, sump, starved film onset) and runs under a work budget
(`max_reynolds_solves`, default 400,000 Reynolds solves per pad and case,
plus a cap of 4,000 per pad and equilibrium solve, both counting at least 5
pads; optional `time_limit` in s). A case that does
not converge emits a `ConvergenceWarning`; `on_failure` decides what happens
to its speed:

- `"warn"` (default): the speed is left out of the coefficient table (for a 2-D
  table, the whole speed row); the element interpolates between the converged
  speeds and `save()` writes only those rows. If no case converged, every row
  is kept and the warning says the element must not be used.
- `"raise"`: `FluidFilmConvergenceError`
  (`ross.bearings.fluid_film.diagnostics`).
- `"keep"`: every row is kept, with the warning.

A first pad of exactly 360 deg (`pad_arc=2 pi`) switches the film to the
short-bearing (oil seal) model: no circumferential pressure flow, so the load
at L/D 1 is about twice that of a finite bearing. Use 359.9 deg for a full
journal solved with the finite-length Reynolds equation.

Failed cases stay in `bearing.results` (`results.converged`,
`results.convergence[i]` with the per-loop iterations, residuals, tolerances
and reasons) and in `bearing.operating_results`, a `pandas.DataFrame` with one
row per case:

```python
op = plain.operating_results
op[
    [
        "speed",
        "converged",
        "reason",
        "eccentricity",
        "attitude",
        "h_min",
        "p_max",
        "power_loss",
        "sommerfeld",
        "re_max",
        "re_cb",
        "re_taylor",
    ]
]
```

Columns: `speed`, `frequency`, `converged`, `reason`, `iterations` and
`residual` (equilibrium Newton), `reynolds_solves`, `wall_time`,
`eccentricity`, `attitude` (from `-y`), `attitude_from_load`, `xj`, `yj`,
`h_min`, `h_min_pad`, `p_max` (also isoviscous), `T_max_pad`, `T_max_film`
(NaN isoviscous), `power_loss`, `q_supply`, `q_side`, `q_carryover`,
`sommerfeld` (`mu N L D / W (R/C)^2`, N in rev/s, supply viscosity),
`specific_load`, `re_max` (local film), `re_cb` (`rho U C / mu`), `re_taylor`
(`41.2 sqrt(R/C)`), `pad_load` and `tilt_angle` (per pad). Reasons:
`"max_iterations"`, `"diverged"`, `"budget"`, `"time_limit"`,
`"singular_jacobian"`, `"pivot_unloaded"`, `"non_finite"`, `"contact"`
(the final journal position crosses the bearing surface).

Each pad tilt solves the pivot moment balance to `|M| <= 1e-9` of the pad's
moment scale or a 1e-10 rad root bracket (Brent's method from the previous
tilt). The `tilt` loop record holds the worst pad's residual relative to
these tolerances (tolerance 1), plus `pad`, `moment` (`|M| / M_scale`) and
`width` (rad). An unloaded pad (no film pressure over a range of tilts)
settles at the largest tilt of that range.

Useful knobs and post-processing:

- `lubricant`: a key of `rs.lubricants_dict` (`"ISOVG32"`, `"ISOVG46"`, `"ISOVG68"`, ...), a dict in its format, or a `Lubricant` (`from ross.bearings.film.lubricant import Lubricant`)
- `viscosity_model`: `None` (default, `"exponential"`: the two-point exponential fit) or `"walther"` (ASTM D341) for a name or dict. A `Lubricant`, or a dict with a `viscosity_model` key, keeps its own law; a different `viscosity_model` is ignored with a warning. Through the ISOVG32 table points (40 and 100 C) the exponential fit is 11-18 % above Walther over 50-80 C, 33 % below at 20 C and 20 % below at 120 C
- Oil datasheet: `Lubricant.from_datasheet(Q_(32, "cSt"), Q_(5.4, "cSt"), 870.0)` (kinematic viscosities at 40 and 100 C, density at 15 C; Walther law by default) builds the density from the API MPMS 11.1 form (linearised at 40 C) and `c_p`, `k` from Cragoe; override any of them with `specific_heat=`, `thermal_conductivity=`, `volumetric_expansion=`. The journal energy equation uses `rho`, `c_p`, `k` at `oil_supply_temperature` as constants. `oil.to_dict()` gives the `lubricants_dict` format plus `viscosity_model`
- Turbulence onset: `re_laminar=500` and `re_turbulent=None` (1000) are compared with the largest local film Reynolds number `re_max`; between them the eddy viscosity is blended. `re_laminar="taylor"` uses `Re_c = 41.2 sqrt(R/C_b)` and, with `re_turbulent=None`, `2 Re_c` (the factor 2 is a choice of the implementation, not part of Taylor's criterion); an explicit `re_turbulent` number is used as given. `Re_c` is compared with the local `re_max`, based on the local film thickness, while Taylor defines it on the mean clearance: on preloaded pads (film up to about `2 C_b`) the onset comes earlier. `operating_results` reports `re_max`, `re_cb` and `re_taylor`. Example, 300 mm, 12,000 rpm plain journal (`re_cb` 1470): 0.65 MW forced laminar, 0.74 MW with `"taylor"`
- `thermal_type`: `None` (isoviscous), `"adiabatic"` or `"full"` (pad conduction)
- `deform_type`: `None` (rigid pads, default) or one of the `"pad_mechanical*"` options for pad/pivot elasticity (full TEHD)
- Speed sweeps: the meshes are built once; the speeds are cut into chains of `chunk_size` (default 5) ascending speeds, and each speed after the first of its chain starts from the converged state of the previous one (`warm_start=True`, default). A 10-speed isoviscous tilting-pad sweep takes about a tenth of the Reynolds solves of cold starts with the earlier tilt bisection. The coefficients at a speed then depend, within the loop tolerances, on which other speeds are in the sweep (up to about 0.3 % at the default tolerances); `warm_start=False` solves every speed independently, exactly as a single-speed bearing. `results.convergence[i]["warm_start"]["start"]` is `"cold"`, `"warm"` or `"cold_retry"` (a warm start that failed and was solved again cold)
- `n_jobs`: chains solved at once in `spawn` worker processes (`-1`: one per CPU); results are identical for every `n_jobs`; a script using it needs an `if __name__ == "__main__":` guard (a notebook does not), and a `RuntimeError` is raised when the workers cannot start. `num_processes` is an alias
- `bearing.coefficients(speed, frequency=None)` returns `(kxx, kxy, kyx, kyy), (cxx, cxy, cyx, cyy)` interpolated at a rotor speed (and whirl frequency for 2-D tables)
- Plots: `plot_pressure_2d()`, `plot_pressure_3d()`, `plot_temperature_2d()`, `plot_film_temperature_3d()`, `plot_film_thickness_2d()`; `show_results()` prints a per-case summary table
- `plot_pad_temperature_3d()` draws the pads as real geometry colored by the solid pad conduction field, resolved through the pad thickness (`thermal_type="full"` only)
- `bearing.save(file)` stores the solved coefficient table, 2-D included (reloads as a plain `BearingElement`, no re-solve)
- `cavitation_pressure` (Pa, gauge): the film never falls below it and the cavitated film is held at it
- Film force at a prescribed journal position (no equilibrium search, no coefficients), e.g. for a force map or a finite-difference check: `from ross.bearings.fluid_film.sweep import BearingSetup, film_force_at`; `setup = BearingSetup.from_inputs(**bearing._engine_inputs(speed))` meshes once, then `film_force_at(setup, speed, xj, yj, tilts="solve")` (positions in m; the keyword dict itself is accepted too, at the cost of a mesh per call) returns `fx`, `fy`, per-pad forces and pivot moments; `tilts="frozen"` holds the tilts of the equilibrium, an array holds given tilts

See `docs/user_guide/tutorial_bearings_part_2.ipynb` for the full tour.
