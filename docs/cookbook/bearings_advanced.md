# Advanced Bearings

Source: `docs/user_guide/tutorial_part_1_2.ipynb`, Examples 7, 9

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

## Fluid Flow Bearings

For hydrodynamic journal bearings with computed coefficients:

```python
# Cylindrical bearing (short bearing theory)
from ross.bearings.fluid_flow import fluid_flow_example
bearing = fluid_flow_example()

# Tilting pad bearing
from ross.bearings.tilting_pad import tilting_pad_example
tpb = tilting_pad_example()
```

See `docs/user_guide/fluid_flow_*.ipynb` for fluid flow theory and examples.
