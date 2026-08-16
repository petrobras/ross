# Time Response

Source: `docs/user_guide/tutorial_part_2_2.ipynb`

## Run

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
speed = 500.0  # rad/s

size = 1000
t = np.linspace(0, 10, size)
F = np.zeros((size, rotor.ndof))

# Apply harmonic force at node 3, x and y directions
node = 3
F[:, rotor.number_dof * node + 0] = 10 * np.cos(2 * t)  # x-direction
F[:, rotor.number_dof * node + 1] = 10 * np.sin(2 * t)  # y-direction

response = rotor.run_time_response(speed, F, t)
```

- `speed` (float or array): rotor speed in rad/s. If array, Newmark method is used automatically (for run-up/coast-down)
- `F` (array): force array, shape `(len(t), rotor.ndof)`. Each row = one time step, each column = one DOF
- `t` (array): time array in seconds
- `method`: `"default"` (scipy integrate) or `"newmark"` (explicit Newmark-beta)

### DOF Indexing for Force Array

Column index = `rotor.number_dof * node + direction`
- direction: 0=x, 1=y, 2=θx, 3=θy (for standard 4-DOF)

## Results: `TimeResponseResults`

```python
response.t     # time array
response.yout  # displacement array, shape (len(t), ndof)
response.xout  # full state vector (displacements + velocities)
```

Access displacement at a specific DOF:
```python
x_node3 = response.yout[:, rotor.number_dof * 3 + 0]  # node 3, x
y_node3 = response.yout[:, rotor.number_dof * 3 + 1]  # node 3, y
```

## Plotting

```python
probe1 = rs.Probe(3, 0)

# 1D: displacement vs time
fig = response.plot_1d(probe=[probe1])

# 2D: orbit at a node
fig = response.plot_2d(node=3)

# 3D: full rotor orbit visualization
fig = response.plot_3d()

# FFT of response
fig = response.plot_dfft(probe=[probe1])
```

## Variable Speed (Run-up / Coast-down)

```python
speed = np.linspace(0, 1000, size)  # speed ramp from 0 to 1000 rad/s
response = rotor.run_time_response(speed, F, t)
# Newmark method is chosen automatically when speed is an array
```
