# Frequency Response (FRF) and Forced Response

Source: `docs/user_guide/tutorial_part_2_2.ipynb`

## Frequency Response Function (FRF)

FRF computes the transfer function between input and output DOFs.

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
speed_range = np.linspace(0, 1000, 200)
frf = rotor.run_freq_response(speed_range=speed_range)
```

- `speed_range` (array): frequency range in rad/s (auto-selected if None)
- `free_free` (bool): if True, ignore bearing stiffness (free-free condition)

### DOF Indexing for FRF

`inp` and `out` are DOF indices: `rotor.number_dof * node + direction`

- direction: 0=x, 1=y, 2=θx, 3=θy (for 4-DOF)

```python
# FRF: force at node 3 x-direction → displacement at node 2 x-direction
inp = rotor.number_dof * 3 + 0  # input DOF
out = rotor.number_dof * 2 + 0  # output DOF

fig = frf.plot_magnitude(inp=inp, out=out, frequency_units="rad/s")
fig = frf.plot_phase(inp=inp, out=out)
fig = frf.plot_polar_bode(inp=inp, out=out)
```

### Results: `FrequencyResponseResults`

```python
frf.freq_resp    # complex FRF matrix, shape (ndof, ndof, num_frequencies)
frf.speed_range  # frequency array (rad/s)
```

## General Forced Response

For applying arbitrary frequency-domain forces:

```python
speed_range = np.linspace(0, 1000, 200)
force = np.zeros((rotor.ndof, len(speed_range)), dtype=complex)
dof = rotor.number_dof * 3 + 0  # force at node 3, x-direction
force[dof, :] = 10.0  # constant 10 N across all frequencies

response = rotor.run_forced_response(force=force, speed_range=speed_range)
```

- `force` (array): complex force array, shape (ndof, num_frequencies)
- `speed_range` (array): frequency array in rad/s
- `unbalance` (list, optional): alternative to force — list of `(node, magnitude, phase)` tuples

### Plotting

Uses `Probe` objects, same as unbalance response:

```python
probe = rs.Probe(node=2, angle=0)
fig = response.plot_magnitude(probe=[probe])
fig = response.plot_phase(probe=[probe])
fig = response.plot_bode(probe=[probe])
```

See [unbalance_response.md](unbalance_response.md) for full plotting details.
