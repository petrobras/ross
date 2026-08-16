# Modal Analysis

Source: `docs/user_guide/tutorial_part_2_1.ipynb`

## Run

```python
import ross as rs

rotor = rs.rotor_example()
modal = rotor.run_modal(speed=0, num_modes=12)
```

- `speed` (float): rotor speed in rad/s (use `rs.Q_(4000, "RPM").to("rad/s").m` to convert)
- `num_modes` (int): number of modes to compute (default 12)
- `synchronous` (bool): if True, evaluates bearing coefficients at synchronous frequency

## Results: `ModalResults`

```python
modal.wn              # natural frequencies (rad/s), array of size num_modes
modal.wd              # damped natural frequencies (rad/s)
modal.damping_ratio   # damping ratios (dimensionless)
modal.log_dec         # logarithmic decrements
modal.lti             # scipy LTI system object (state-space)
```

## Plotting

```python
# 2D mode shape (specify mode index, 0-based)
fig = modal.plot_mode_2d(0)

# 3D mode shape
fig = modal.plot_mode_3d(0)

# Orbit at specific nodes
fig = modal.plot_orbit(mode=0, nodes=[2, 4])
```

Plot options:
- `frequency_type`: `"wd"` (damped, default) or `"wn"` (undamped)
- `frequency_units`: `"rad/s"` (default), `"RPM"`, `"Hz"`
- `damping_parameter`: `"log_dec"` (default) or `"damping_ratio"`

## Interpreting Results

- Modes come in pairs (forward/backward whirl) for each natural frequency
- Forward whirl: precession in same direction as rotation
- Backward whirl: precession opposite to rotation
- `log_dec > 0` indicates stable mode; `log_dec < 0` indicates unstable
- At `speed=0`, forward and backward frequencies are identical (no gyroscopic effect)
