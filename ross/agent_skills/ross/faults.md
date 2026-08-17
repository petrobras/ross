# Fault Analysis

Source: `docs/user_guide/tutorial_faults.ipynb`

All fault analyses return `TimeResponseResults` — same plotting as [time_response.md](time_response.md).
The unbalance arguments (`node`, `unbalance_magnitude`, `unbalance_phase`) are lists — one entry per unbalance source.

## Rubbing

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
t = np.linspace(0, 5, 5000)

response = rotor.run_rubbing(
    n=3,  # shaft ELEMENT where rubbing occurs (between nodes 3 and 4)
    distance=5e-4,  # clearance (m)
    contact_stiffness=1e6,  # contact stiffness (N/m)
    contact_damping=1e2,  # contact damping (N·s/m)
    friction_coeff=0.3,  # friction coefficient
    node=[2],  # unbalance node(s)
    unbalance_magnitude=[0.001],  # kg·m
    unbalance_phase=[0],  # rad
    speed=500,  # rad/s
    t=t,
    torque=False,  # include friction torque effect
)

probe = rs.Probe(3, 0)
fig = response.plot_1d(probe=[probe])
fig = response.plot_dfft(probe=[probe])
```

## Crack

```python
response = rotor.run_crack(
    n=3,  # cracked shaft element index
    depth_ratio=0.2,  # crack depth / element diameter; max 0.5 (Mayes/Gasch), 0.6 (Flex models)
    node=[2],  # unbalance node(s)
    unbalance_magnitude=[0.001],
    unbalance_phase=[0],
    speed=500,
    t=t,
    crack_model="Mayes",  # "Mayes", "Gasch", "Flex Open" or "Flex Breathing"
)
```

- `cross_divisions` (int, optional): cross-section divisions, used by the "Flex Breathing" model

## Misalignment

```python
# Flexible coupling misalignment
response = rotor.run_misalignment(
    node=[2],  # unbalance node(s) — NOT the coupling location
    unbalance_magnitude=[0.001],
    unbalance_phase=[0],
    speed=500,
    t=t,
    coupling="flex",  # "flex" or "rigid"
    # required kwargs for coupling="flex":
    n=0,  # shaft ELEMENT where the misalignment occurs
    mis_type="parallel",  # "parallel", "angular" or "combined"
    mis_distance_x=2e-4,  # m
    mis_distance_y=2e-4,  # m
    mis_angle=np.deg2rad(0.5),  # rad (used by "angular"/"combined")
    radial_stiffness=40e3,  # N/m
    bending_stiffness=38e3,  # N·m/rad
    input_torque=0,
    load_torque=0,
)
# coupling="rigid" instead requires: n, mis_distance (+ optional input_torque/load_torque)
```

## Interpreting Fault Signatures

- **Rubbing**: sub-harmonics and super-harmonics in FFT (1/2x, 3/2x, 2x, etc.)
- **Crack**: breathing crack introduces 2x and higher harmonics; severity increases with `depth_ratio`
- **Misalignment**: strong 2x component in FFT, characteristic orbit shapes
- Use `plot_dfft()` to identify fault-specific frequency content
- Compare orbits at different nodes using `plot_2d(node=n)`
