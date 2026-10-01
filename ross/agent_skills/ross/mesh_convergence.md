# Mesh Convergence

Source: `Rotor.run_mesh_convergence` and `Rotor.refine` docstrings

## Run

```python
import ross as rs

rotor = rs.compressor_example()
results = rotor.run_mesh_convergence(rtol=1e-3, frequencies=6, strategy="cheapest")
refined_rotor = results.rotor  # use this rotor in the analyses
```

- `rtol` (float): maximum relative error of the natural frequencies against the converged reference (default `1e-3`, i.e. 0.1%)
- `frequencies` (int): number of natural frequencies that must converge, the lowest ones (default 6)
- `strategy` (str): `"cheapest"` (fewest shaft elements found among uniform and graded meshes, default) or `"uniform"` (every element at most `h` long)
- `speed` (float, `Q_`): rotor speed of the modal analyses in rad/s (default 0)
- `max_elements` (int): cap on the shaft elements of any mesh evaluated (default 1000)

The rotor is not modified. Its shaft elements are the coarsest mesh considered: they are split, never merged.

## Results: `MeshConvergenceResults`

```python
results.rotor  # new Rotor with the chosen discretization
results.subdivisions  # elements per original shaft interval
results.wn  # lowest natural frequencies of results.rotor (rad/s)
results.wn_reference  # natural frequencies of the converged reference (rad/s)
results.error  # relative error of each frequency
results.reference_elements  # shaft elements of the reference mesh
# maximum error vs number of shaft elements, with the tolerance line
fig = results.plot()
```

## Usage

```python
# Split every shaft interval in 4 equal elements
fine = rotor.refine(4)

# Reuse a converged discretization on a rotor with the same shaft intervals
# (e.g. after changing bearing coefficients)
other = rs.Rotor(rotor.shaft_elements, rotor.disk_elements, new_bearings)
other_refined = other.refine(results.subdivisions)

# Converge the modes at the maximum continuous speed
results = rotor.run_mesh_convergence(speed=rs.Q_(12000, "RPM"))
```

## Interpreting Results

- The reference halves the element length until the frequencies change less than `rtol / 10`; the errors are measured against it
- The k-th lowest frequency of a mesh is compared with the k-th lowest of the reference; when two different modes are closer than the mesh error (e.g. a bending and a torsional mode), check them on `results.rotor`
- The graded mesh gives thin sections more elements than thick ones (a fixed number of elements per local bending wavelength), which pays off on rotors with varying diameters
- Each mesh evaluated is one modal analysis: industrial models can take about a minute, dominated by the reference; lower `max_elements` to cap it
- Not available for `MultiRotor` and `CoAxialRotor`
