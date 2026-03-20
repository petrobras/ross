# Static Analysis

Source: `docs/user_guide/tutorial_part_2_1.ipynb`

## Run

```python
import ross as rs

rotor = rs.rotor_example()
static = rotor.run_static()
```

No parameters needed — computes deformation under gravity loads.

## Results: `StaticResults`

```python
static.deformation        # shaft deflection array
static.Vx                 # shear force array
static.Bm                 # bending moment array
static.bearing_forces     # reaction forces at bearings (dict)
```

## Plotting

```python
# Static deformation (sag curve)
fig = static.plot_deformation()

# Bending moment diagram
fig = static.plot_bending_moment()

# Shear force diagram
fig = static.plot_shearing_force()

# Free body diagram (bearing reactions and disk weights)
fig = static.plot_free_body_diagram()
```

Unit options: `deformation_units` (default `"m"`), `force_units` (default `"N"`), `moment_units` (default `"N*m"`), `rotor_length_units` (default `"m"`).

## Interpreting Results

- Maximum deflection typically occurs between bearings
- Bearing reactions should sum to total rotor weight
- Used to verify model correctness before dynamic analyses
- Overhung rotors show larger deflections at the free end
