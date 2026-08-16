# UCS and Level 1 Stability Analysis

Source: `docs/user_guide/tutorial_part_2_2.ipynb` (UCS); `ross/rotor_assembly.py` `run_level1` (no notebook source)

## Undamped Critical Speed Map (UCS)

Plots undamped natural frequencies as a function of bearing stiffness.

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
ucs = rotor.run_ucs(
    stiffness_range=(5, 10),  # log10 exponents: 1e5 to 1e10 N/m
    num=50,                   # number of stiffness points
    num_modes=16,
    synchronous=False,
)

fig = ucs.plot(stiffness_units="N/m", frequency_units="RPM")
```

- `stiffness_range` (tuple, optional): `(start, end)` exponents of a base-10 log scale — `(5, 10)` spans 1e5 to 1e10 N/m. Default `(6, 11)`, or ±3 decades around the bearing stiffness at `rated_w`
- `num` (int): number of stiffness points in the range (default 20)
- `num_modes` (int): number of eigenvalues computed (default 16); `num_modes // 4` forward-mode curves are produced
- `synchronous` (bool): evaluate at synchronous frequency

### Results: `UCSResults`

```python
ucs.stiffness_log     # bearing stiffness values used (N/m), size num
ucs.stiffness_range   # the (start, end) exponent tuple
ucs.wn                # undamped natural frequencies (rad/s), shape (num_modes // 4, num)
```

## Level 1 Stability Analysis (API 617)

Sweeps a destabilizing cross-coupled stiffness `Q` applied at one node and tracks the
logarithmic decrement. Requires a rotor with a rated speed (`rated_w`), since the modal
analysis runs at that speed.

```python
rotor = rs.rotor_example()
rotor.rated_w = rs.Q_(4000, "RPM").to("rad/s").m

level1 = rotor.run_level1(n=3, stiffness_range=(1e6, 1e11), num=5)
fig = level1.plot()
```

- `n` (int): node where the cross-coupled stiffness (`kxy=Q`, `kyx=-Q`) is applied — typically the impeller/midspan node
- `stiffness_range` (tuple): `(start, end)` of the applied cross-coupled stiffness sweep in N/m (evenly spaced; always pass it explicitly)
- `num` (int): number of Q values evaluated (default 5)

### Results: `Level1Results`

```python
level1.stiffness_range  # applied cross-coupled stiffness values Q (N/m)
level1.log_dec          # log decrement at each Q
```

The plot shows log decrement versus applied cross-coupled stiffness; the Q where the
log decrement crosses zero is the instability threshold Q0.

## Interpreting Results

- **UCS map**: horizontal lines = rigid-body modes, rising curves = flexural modes. Intersection with bearing stiffness gives approximate critical speeds
- **Level 1**: check that the logarithmic decrement stays positive across the range of applied cross-coupling expected in operation (API 617)
- These analyses help during the design phase before detailed damped analysis
