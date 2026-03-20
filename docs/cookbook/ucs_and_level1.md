# UCS and Level 1 Stability Analysis

Source: `docs/user_guide/tutorial_part_2_1.ipynb`

## Undamped Critical Speed Map (UCS)

Plots undamped natural frequencies as a function of bearing stiffness.

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
ucs = rotor.run_ucs(
    stiffness_range=np.logspace(5, 10, 50),  # N/m
    num_modes=16,
    synchronous=False,
)

fig = ucs.plot(stiffness_units="N/m", frequency_units="RPM")
```

- `stiffness_range` (array, optional): bearing stiffness values in N/m (log-spaced recommended)
- `num_modes` (int): number of modes to track (default 16)
- `synchronous` (bool): evaluate at synchronous frequency

### Results: `UCSResults`

```python
ucs.stiffness_range   # bearing stiffness array (N/m)
ucs.wn                # natural frequencies, shape (num_stiffnesses, num_modes)
```

## Level 1 Stability Analysis (API 617)

Evaluates rotor stability per API 617 guidelines.

```python
level1 = rotor.run_level1(n=5, stiffness_range=np.logspace(5, 10, 5), num=5)
fig = level1.plot()
```

- `n` (int): number of modes to compute
- `stiffness_range` (array, optional): bearing stiffness range
- `num` (int): number of points for interpolation

### Results: `Level1Results`

The plot shows the stability threshold relative to bearing support stiffness, helping identify whether the rotor is stable across the expected range of bearing conditions.

## Interpreting Results

- **UCS map**: horizontal lines = rigid-body modes, rising curves = flexural modes. Intersection with bearing stiffness gives approximate critical speeds
- **Level 1**: check if the logarithmic decrement stays positive across the operating range
- These analyses help during the design phase before detailed damped analysis
