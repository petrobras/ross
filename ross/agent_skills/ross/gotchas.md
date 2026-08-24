# Common Gotchas

## Units

- **Speeds are in rad/s**, not RPM. Convert: `rs.Q_(4000, "RPM").to("rad/s").m` → ~418.88 rad/s
- **Unbalance magnitude is kg·m** (mass × eccentricity), not kg. A 1 g unbalance at 10 mm radius = `1e-3 * 10e-3 = 1e-5` kg·m
- **Stiffness is N/m**, damping is N·s/m. Common bearing stiffness: 1e5–1e9 N/m
- All `@check_units`-decorated methods accept `pint.Quantity` objects, but results are always in base SI

## Node Numbering

- Nodes are numbered 0 to N (not 1 to N)
- `n` shaft elements produce `n+1` nodes
- Disks and bearings attach to nodes, shaft elements connect consecutive nodes
- Node 0 is always the left end of the rotor

## DOF Indexing

- Every node has 6 DOF: `[x, y, z, alpha, beta, theta]` — 0=x (horizontal), 1=y (vertical), 2=z (axial), 3=alpha (rotation about x), 4=beta (rotation about y), 5=theta (torsion about z)
- DOF index = `rotor.number_dof * node + direction`
- `rotor.number_dof` gives the DOFs per node (6 in current ROSS)
- `rotor.ndof` gives the total DOFs for the entire rotor

## Force Array Shape (Time Response)

- `F` shape must be `(len(t), rotor.ndof)` — rows are time steps, columns are DOFs
- This is the transpose of the forced response `force` array which is `(ndof, num_frequencies)`

## Speed-Dependent Bearings

- The `frequency` array in `BearingElement` defines the speeds at which coefficients are known
- Coefficients are interpolated automatically — no need to manually evaluate
- Any coefficient given as an array (`kxx`, `kyy`, `cxx`, etc.) must have the same length as `frequency`; scalars are broadcast to that length automatically

## Plotting

- All `plot_*` methods return a Plotly `Figure` object — call `fig.show()` to display, or `fig.write_image("file.png")` to save. Exception: `CampbellResults.plot_with_mode_shape` needs the optional `dash` package and launches an interactive app instead of returning a figure
- Probe objects require `angle` for radial probes: `rs.Probe(node, angle=0)`. Omitting angle raises an error
- Use `rs.Q_(45, "deg")` for probe angles in degrees

## Campbell Diagram

- `frequencies` parameter is the number of frequencies to track, not the frequency values
- `speed_range` should start from 0 (or near 0) so harmonic-line crossings are meaningful — a convention, not enforced by the code
- `harmonics` in `plot()` are multipliers of the synchronous line (1 = 1x, 0.5 = 0.5x, etc.)

## Modal Analysis at Speed=0

- Gyroscopic splitting only appears at nonzero speed
- At zero speed, forward and backward modes are identical only for isotropic supports (`kxx == kyy`); anisotropic bearings still split each mode pair
- Always run modal analysis at the operating speed for realistic results
