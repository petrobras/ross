# Critical Speed Analysis

Source: `docs/user_guide/tutorial_part_2_1.ipynb`

## Run

```python
import ross as rs

rotor = rs.rotor_example()
cs = rotor.run_critical_speed(num_modes=12)
```

- `speed_range` (array, optional): speed range for search (auto-selected if None)
- `num_modes` (int): number of modes to compute (default 12)
- `rtol` (float): relative tolerance for critical speed convergence (default 0.005)

## Results: `CriticalSpeedResults`

```python
cs.wn     # undamped critical speeds (rad/s), array
cs.wd     # damped critical speeds (rad/s), array
cs.log_dec   # logarithmic decrements at each critical speed
cs.damping_ratio  # damping ratios at each critical speed
```

## Usage

```python
# Print critical speeds in RPM
for i, (wn, ld) in enumerate(zip(cs.wn, cs.log_dec)):
    rpm = rs.Q_(wn, "rad/s").to("RPM").m
    print(f"Mode {i}: {rpm:.0f} RPM, log_dec = {ld:.4f}")
```

## Interpreting Results

- Critical speeds are where the synchronous excitation line crosses a natural frequency
- `log_dec > 0` at a critical speed means the rotor can safely pass through it
- Check separation margin from operating speed per API 617 requirements
- For anisotropic bearings, each mode may split into forward and backward critical speeds
