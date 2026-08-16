# Critical Speed Analysis

Source: `docs/user_guide/tutorial_part_2_1.ipynb`

## Run

```python
import ross as rs

rotor = rs.rotor_example()
cs = rotor.run_critical_speed(num_modes=12)
```

- `speed_range` (tuple, optional): `(start, end)` in rad/s — only critical speeds inside this range are returned; when given, `num_modes` is ignored
- `num_modes` (int): number of modes to compute (default 12)
- `rtol` (float): relative tolerance for critical speed convergence (default 0.005)

## Results: `CriticalSpeedResults`

`wn` and `wd` are methods taking a `frequency_units` argument; the rest are attributes:

```python
cs.wn()            # undamped critical speeds (rad/s), array
cs.wd()            # damped critical speeds (rad/s), array
cs.wd("RPM")       # converted units
cs.log_dec         # logarithmic decrements at each critical speed
cs.damping_ratio   # damping ratios at each critical speed
cs.whirl_direction # "Forward" / "Backward" / "Mixed" per critical speed
```

## Usage

```python
# Print critical speeds in RPM
for i, (wn, ld) in enumerate(zip(cs.wn("RPM"), cs.log_dec)):
    print(f"Mode {i}: {wn:.0f} RPM, log_dec = {ld:.4f}")
```

## Interpreting Results

- Critical speeds are where the synchronous excitation line crosses a natural frequency
- `log_dec > 0` at a critical speed means the rotor can safely pass through it
- Check separation margin from operating speed per API 617 requirements
- For anisotropic bearings, each mode may split into forward and backward critical speeds
