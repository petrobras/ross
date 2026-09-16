# Modal Analysis

Source: `docs/user_guide/tutorial_analyses_part_1.ipynb`

## Run

```python
import ross as rs

rotor = rs.rotor_example()
modal = rotor.run_modal(speed=0, num_modes=12)
```

- `speed` (float): rotor speed in rad/s (use `rs.Q_(4000, "RPM").to("rad/s").m` to convert)
- `num_modes` (int): number of eigenvalues to compute (default 12); the results contain `num_modes // 2` mode pairs
- `sparse` (bool): use the sparse ARPACK eigensolver (True, default) or dense `scipy.linalg.eig` (False)
- `synchronous` (bool): if True, runs a synchronous analysis — the shaft gyroscopic terms are folded into the mass matrix so the whirl frequency equals the rotor speed (Rouch's formulation, default False). Unrelated to the coefficient options below
- `frequency` (float, optional): evaluate the frequency-dependent bearing/seal coefficients at this excitation (whirl) frequency while the gyroscopic effect keeps `speed`. Default None: synchronous coefficients (frequency = speed)
- `matched_whirl` (bool): iterate each mode until the coefficients are evaluated at the mode's own damped natural frequency (default False; mutually exclusive with `frequency`)
- `whirl_rtol` (float, 1e-3) and `whirl_max_iter` (int, 15): tolerance and iteration cap of the `matched_whirl` fixed point (a warning is issued if a mode does not converge)

## Results: `ModalResults`

```python
modal.wn  # undamped natural frequencies (rad/s), array of size num_modes // 2
modal.wd  # damped natural frequencies (rad/s)
modal.damping_ratio  # damping ratios (dimensionless)
modal.log_dec  # logarithmic decrements
modal.evalues  # raw complex eigenvalues (modal.evectors for eigenvectors)
modal.whirl_frequency  # whirl frequency the coefficients were evaluated at, per mode
```

## Coefficients at the Whirl Frequency

Only matters for elements whose coefficients depend on the excitation frequency:
`frequency=` tables (squeeze film dampers, magnetic bearings) and 2-D
`speed=` + `frequency=` tables (seals and fluid-film bearings built with
`frequency=`, see [seals.md](seals.md)). Speed-only tables are constant along
the whirl axis, so the three calls below coincide for them.

```python
w = rs.Q_(8000, "RPM").to("rad/s").m
modal_sync = rotor.run_modal(speed=w)  # coefficients at frequency = w
modal_fixed = rotor.run_modal(speed=w, frequency=0.5 * w)  # all modes at 0.5 w
modal_matched = rotor.run_modal(speed=w, matched_whirl=True)  # each mode at its own wd
modal_matched.whirl_frequency  # converged per-mode whirl frequencies (== modal_matched.wd)
```

`matched_whirl` solves one eigenproblem per mode per iteration (typically 2–3
iterations for `num_modes // 2` modes), so it costs a few times a plain
`run_modal`. Use it for subsynchronous stability, where the destabilizing mode
whirls well below the running speed and the synchronous lookup misestimates the
log decrement.

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
- At `speed=0` there is no gyroscopic splitting, so forward/backward pairs coincide only if the supports are isotropic (`kxx == kyy`); anisotropic bearings still split each pair at zero speed (e.g. `rs.rotor_example()` gives 91.8 / 96.3 rad/s)
