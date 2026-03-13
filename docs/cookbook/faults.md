# Fault Analysis

Source: Examples 12 (rubbing), 15 (crack); `docs/user_guide/example_*.ipynb`

All fault analyses return `TimeResponseResults` — same plotting as [time_response.md](time_response.md).

## Rubbing

```python
import ross as rs
import numpy as np

rotor = rs.rotor_example()
t = np.linspace(0, 5, 5000)

response = rotor.run_rubbing(
    n=3,                    # node where rubbing occurs
    distance=5e-4,          # clearance (m)
    contact_stiffness=1e6,  # contact stiffness (N/m)
    contact_damping=1e2,    # contact damping (N·s/m)
    friction_coeff=0.3,     # friction coefficient
    node=2,                 # unbalance node
    unbalance_magnitude=0.001,  # kg·m
    unbalance_phase=0,      # rad
    speed=500,              # rad/s
    t=t,
    torque=False,           # include friction torque effect
)

probe = rs.Probe(3, 0)
fig = response.plot_1d(probe=[probe])
fig = response.plot_dfft(probe=[probe])
```

## Crack

```python
response = rotor.run_crack(
    n=3,                    # cracked element index
    depth_ratio=0.2,        # crack depth / shaft radius (0 to 1)
    node=2,                 # unbalance node
    unbalance_magnitude=0.001,
    unbalance_phase=0,
    speed=500,
    t=t,
    crack_model="Mayes",    # "Mayes" or "Gasch"
)
```

## Misalignment

```python
# Flexible coupling misalignment
response = rotor.run_misalignment(
    node=3,                 # coupling node
    unbalance_magnitude=0.001,
    unbalance_phase=0,
    speed=500,
    t=t,
    coupling="flex",        # "flex" or "rigid"
    # Additional kwargs depend on coupling type
)
```

## Interpreting Fault Signatures

- **Rubbing**: sub-harmonics and super-harmonics in FFT (1/2x, 3/2x, 2x, etc.)
- **Crack**: breathing crack introduces 2x and higher harmonics; severity increases with `depth_ratio`
- **Misalignment**: strong 2x component in FFT, characteristic orbit shapes
- Use `plot_dfft()` to identify fault-specific frequency content
- Compare orbits at different nodes using `plot_2d(node=n)`
