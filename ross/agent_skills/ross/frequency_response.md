# Frequency Response (FRF) and Forced Response

Source: `docs/user_guide/tutorial_analyses_part_2.ipynb`

## Frequency Response Function (FRF)

FRF computes the transfer function between input and output DOFs.

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
speed_range = np.linspace(0, 1000, 200)
frf = rotor.run_freq_response(speed_range=speed_range)
```

- `speed_range` (array): excitation frequencies in rad/s (auto-selected if None). By default the rotor speed follows the sweep (synchronous excitation: speed = frequency at every point)
- `speed` (float, optional): hold the rotor speed fixed at this value while `speed_range` sweeps the excitation frequency — the gyroscopic term and speed-dependent bearing/seal coefficients use `speed`, frequency-dependent coefficients follow the sweep
- `free_free` (bool): if True, evaluates the transfer matrix at zero rotating speed (no gyroscopic term; speed-dependent bearing coefficients taken at 0 rad/s) while the excitation frequency sweeps — bearings are still included (equivalent to `speed=0`)

```python
# FRF at a fixed running speed, excitation swept from 0 to 1000 rad/s
w = rs.Q_(4000, "RPM").to("rad/s").m
frf_fixed = rotor.run_freq_response(speed_range=speed_range, speed=w)
```

### DOF Indexing for FRF

`inp` and `out` are DOF indices: `rotor.number_dof * node + direction`

- direction: 0=x, 1=y, 2=z, 3=alpha (rotation about x), 4=beta (rotation about y), 5=theta (torsion)

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
frf.freq_resp  # complex FRF matrix, shape (ndof, ndof, num_frequencies)
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
- `speed_range` (array): excitation frequency array in rad/s
- `speed` (float, optional): fixed rotor speed for the whole sweep (default: synchronous, speed = frequency), e.g. `rotor.run_forced_response(force=force, speed_range=speed_range, speed=w)`
- `unbalance` (array, optional): cosmetic only — a `(3, n)` array `np.vstack((nodes, magnitudes, phases))` used to draw unbalance markers on `plot_deflected_shape`; it does not generate force (`force` is still required)

### Plotting

Uses `Probe` objects, same as unbalance response:

```python
probe = rs.Probe(node=2, angle=0)
fig = response.plot_magnitude(probe=[probe])
fig = response.plot_phase(probe=[probe])
fig = response.plot_bode(probe=[probe])
```

See [unbalance_response.md](unbalance_response.md) for full plotting details.
