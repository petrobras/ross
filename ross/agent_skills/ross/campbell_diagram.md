# Campbell Diagram

Source: `docs/user_guide/tutorial_part_2_1.ipynb`

## Run

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
speed_range = np.linspace(0, rs.Q_(10000, "RPM").to("rad/s").m, 50)
campbell = rotor.run_campbell(speed_range, frequencies=6)
```

- `speed_range` (array): rotor speeds in rad/s
- `frequencies` (int): number of frequencies to track (default 6)
- `frequency_type`: `"wd"` (damped, default) or `"wn"` (undamped)

## Results: `CampbellResults`

```python
campbell.speed_range  # speed array (rad/s)
campbell.wd  # tracked frequencies, shape (num_speeds, num_frequencies) — holds wn values if frequency_type="wn" was used
campbell.log_dec  # log decrements, shape (num_speeds, num_frequencies)
campbell.damping_ratio  # damping ratios, shape (num_speeds, num_frequencies)
campbell.whirl_values  # whirl direction per point: 0 (backward), 0.5 (mixed), 1 (forward)
```

## Plotting

```python
# Basic Campbell diagram with 1x synchronous line
fig = campbell.plot(harmonics=[1])

# With multiple harmonics and custom units
fig = campbell.plot(
    harmonics=[0.5, 1, 2],
    frequency_units="RPM",
    speed_units="RPM",
    damping_parameter="log_dec",
)

# Interactive Campbell + mode shapes (requires the optional `dash` package;
# launches a Dash app inline and returns None instead of a Figure)
campbell.plot_with_mode_shape(harmonics=[1])
```

## Interpreting Results

- **Crossing points** where a natural frequency curve intersects the Nx synchronous line indicate potential critical speeds
- **1x line** (harmonic=1): excitation frequency equals rotation speed (most common for unbalance)
- **0.5x, 2x lines**: sub- and super-synchronous excitations
- Diverging forward/backward frequencies with speed → gyroscopic effect
- Check `log_dec` at crossings: if negative, the mode is unstable at that speed
