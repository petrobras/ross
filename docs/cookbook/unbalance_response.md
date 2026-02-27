# Unbalance Response

Source: `docs/user_guide/tutorial_part_2_2.ipynb`

## Run

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
frequency_range = np.linspace(0, rs.Q_(10000, "RPM").to("rad/s").m, 200)
response = rotor.run_unbalance_response(
    node=2,
    unbalance_magnitude=0.001,  # kg·m (mass × eccentricity)
    unbalance_phase=0,          # rad
    frequency=frequency_range,
)
```

- `node` (int): node where unbalance is applied
- `unbalance_magnitude` (float): in kg·m (NOT kg — this is mass times eccentricity)
- `unbalance_phase` (float): phase angle in rad
- `frequency` (array): excitation frequencies in rad/s

Multiple unbalance sources: pass arrays for `node`, `unbalance_magnitude`, `unbalance_phase`.

## Results: `ForcedResponseResults`

Uses `Probe` objects to specify measurement locations:

```python
probe1 = rs.Probe(node=2, angle=0)                           # radial, 0°
probe2 = rs.Probe(node=4, angle=rs.Q_(45, "deg"), tag="DE")  # radial, 45°
```

## Plotting

```python
# Magnitude vs frequency
fig = response.plot_magnitude(probe=[probe1, probe2])

# Phase vs frequency
fig = response.plot_phase(probe=[probe1])

# Bode plot (magnitude + phase)
fig = response.plot_bode(probe=[probe1])

# Polar (Nyquist) plot
fig = response.plot_polar_bode(probe=[probe1])

# Deflected shape at a specific speed
fig = response.plot_deflected_shape(speed=rs.Q_(5000, "RPM").to("rad/s").m)

# 2D deflected shape only
fig = response.plot_deflected_shape_2d(speed=rs.Q_(5000, "RPM").to("rad/s").m)

# Bending moment at a specific speed
fig = response.plot_bending_moment(speed=rs.Q_(5000, "RPM").to("rad/s").m)
```

Plot unit options: `frequency_units`, `amplitude_units`, `phase_units` (e.g. `"RPM"`, `"m"`, `"deg"`).

## Interpreting Results

- Peaks in the magnitude plot correspond to critical speeds
- Phase changes ~180° through each resonance
- Multiple probes at different angles help identify whirl direction
- Compare results with API 617 acceptance criteria if applicable
