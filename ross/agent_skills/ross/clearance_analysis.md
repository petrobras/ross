# Clearance Analysis (API 617)

Source: `docs/user_guide/tutorial_analyses_part_2.ipynb` (Section 3)

Checks the unbalance response against the running clearances of the machine following API 617 (9th edition, 6.8.2.10 and 6.8.2.11).

## Run

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
bearings = [
    rs.BearingElement(
        n=0, kxx=1e6, cxx=1e3, radial_clearance=rs.Q_(100, "um"), tag="DE"
    ),
    rs.BearingElement(
        n=6, kxx=1e6, cxx=1e3, radial_clearance=rs.Q_(120, "um"), tag="NDE"
    ),
    rs.SealElement(n=3, kxx=0, cxx=0, radial_clearance=rs.Q_(300, "um"), tag="eye"),
]
rotor = rs.Rotor(rotor.shaft_elements, rotor.disk_elements, bearings)

probes = [
    rs.Probe(0, rs.Q_(45, "deg"), tag="DE"),
    rs.Probe(6, rs.Q_(45, "deg"), tag="NDE"),
]

results = rotor.run_clearance_analysis(
    speed_range=rs.Q_(np.linspace(0, 10000, 201), "RPM"),  # zero to trip speed
    minimum_allowable_speed=rs.Q_(7000, "RPM"),  # Nma
    maximum_continuous_speed=rs.Q_(9000, "RPM"),  # Nmc
    probes=probes,  # machine vibration probes (location and orientation)
    mode=0,  # forward mode used to place the unbalance (0 = first forward mode)
    scale_factor_cap=None,  # API 617 caps Scc at 6; None applies no cap
)
```

- `speed_range` (array, rad/s): sweep of the unbalance response; `Nma` and `Nmc` are added if absent
- `probes` (list of `Probe`): `Amax` is the largest peak-to-peak amplitude over these probes between `Nma` and `Nmc`
- `mode` (int): forward mode index; the unbalance amount and placement come from `rotor.api617_unbalance(mode, maximum_continuous_speed)`
- `node`, `unbalance_magnitude`, `unbalance_phase`: pass all three to use an explicit unbalance instead of `mode`
- Close-clearance locations are NOT passed: every bearing/seal element with a `radial_clearance` is used (fluid-film bearings, `LabyrinthSeal`, `HolePatternSeal`, or `BearingElement`/`SealElement` built with `radial_clearance=`). The diametral clearance is `2 * radial_clearance`.

## Procedure implemented

1. Unbalance response with `Ua = 2 * Ur`, `Ur = 6350 W / Nmc` g·mm (`W` in kg, `Nmc` in rpm), placed at the antinodes of the mode shape (single antinode between bearings: `W` = sum of journal loads; conical mode: one unbalance per antinode, 180° apart, `W` = nearest journal load; overhung antinode: `W` = overhung mass)
2. `Avl = min(25.4, 25.4 * sqrt(12000 / Nmc))` µm peak to peak
3. `Amax` = largest probe amplitude (pk-pk) in `[Nma, Nmc]`
4. `Scc = Avl / Amax` (capped by `scale_factor_cap` when given)
5. Scaled major-axis pk-pk amplitude at each location vs. 75 % of the minimum diametral clearance, at the worst speed of `speed_range`

The scaled response does not depend on the unbalance amount unless the cap is active, so results from reports using older API editions (4 × Ur) compare directly after scaling.

## Results: `ClearanceResults`

Amplitudes are stored in metres peak to peak; clearances are diametral, in metres.

```python
results.vibration_limit  # Avl (m pk-pk)
results.max_probe_amplitude  # Amax (m pk-pk)
results.scale_factor  # Scc
results.unbalance_node, results.unbalance_magnitude, results.unbalance_phase
results.clearance_tags, results.clearance_nodes  # close-clearance locations
results.diametral_clearance, results.clearance_limit  # clearance and 75 % limit (m)
results.clearance_response  # scaled pk-pk amplitude, shape (n_locations, n_speeds)
results.max_clearance_response, results.speed_at_max_response, results.passed
results.probe_response  # unscaled probe amplitude, shape (n_probes, n_speeds)

# one row per location with % of limit and status
results.data(length_units="um", speed_units="RPM")
```

## Plotting

```python
fig = results.plot()  # bars: clearance, 75 % limit, scaled max amplitude per location
fig = results.plot_response()  # scaled amplitude vs. speed per location with its limit
# probe response with Avl and the Nma-Nmc range (API 617 Figure 4)
fig = results.plot_probe_response()
```

Unit options: `length_units` (default `"um"`), `speed_units` (default `"RPM"`).

## API 617 unbalance only

```python
unbalance = rotor.api617_unbalance(mode=0, maximum_continuous_speed=rs.Q_(9000, "RPM"))
unbalance["node"], unbalance["unbalance_magnitude"], unbalance["unbalance_phase"]
unbalance["static_load"]  # W used for each unbalance (kg)
response = rotor.run_unbalance_response(
    unbalance["node"],
    unbalance["unbalance_magnitude"],
    unbalance["unbalance_phase"],
    speed_range,
)
```

## Interpreting Results

- `status == "EXCEEDED"` at a location means the scaled response is above 75 % of its diametral clearance (API 617 6.8.2.11.1)
- `Scc` below 1 means the probe response already exceeds the test vibration limit (6.8.2.10)
- Use `plot_response()` to see whether the worst point is at a critical speed or at the top of the range
- Running clearances differ from assembled ones (thermal growth, bearing lift, sag): set the element `radial_clearance` to the running value
