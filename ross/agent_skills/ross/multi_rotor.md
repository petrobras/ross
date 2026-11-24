# Multi-Rotor (Geared Shafts)

Source: `docs/user_guide/tutorial_multirotor.ipynb`

`MultiRotor` couples a driving and a driven `Rotor` through a pair of gears; the gear mesh couples the lateral and torsional motion of the two shafts. It is a `Rotor` subclass, so the analyses of the other recipes run on it. Every speed passed to an analysis is the speed of the driving rotor.

## Build

```python
import numpy as np
import ross as rs
from ross.units import Q_

material = rs.Material(name="mat_steel", rho=7800, E=207e9, G_s=79.5e9)

# Driving rotor (generator side): the gear is at node 4
shaft1 = [
    rs.ShaftElement(L=L, idl=0.0, odl=d, material=material)
    for L, d in zip([0.1, 4.24, 1.16, 0.3], [0.3, 0.3, 0.22, 0.22])
]
gear1 = rs.GearElement(
    n=4,
    m=726.4,
    Id=56.95,
    Ip=113.9,
    n_teeth=328,
    base_diameter=0.5086 * 2,
    pr_angle=Q_(22.5, "deg"),
)
rotor1 = rs.Rotor(
    shaft1,
    [
        rs.DiskElement(n=1, m=525.7, Id=16.1, Ip=32.2),
        rs.DiskElement(n=2, m=116.04, Id=3.115, Ip=6.23),
        gear1,
    ],
    [
        rs.BearingElement(n=0, kxx=183.9e6, kyy=200.4e6, cxx=3e3),
        rs.BearingElement(n=3, kxx=183.9e6, kyy=200.4e6, cxx=3e3),
    ],
)

# Driven rotor (turbine side): the gear is at node 0
shaft2 = [
    rs.ShaftElement(L=L, idl=0.0, odl=0.15, material=material) for L in [0.3, 5, 0.1]
]
pressure_angle = Q_(22.5, "deg")
gear2 = rs.GearElement(
    n=0,
    m=5,
    Id=0.002,
    Ip=0.004,
    n_teeth=23,
    pitch_diameter=2 * Q_(0.03567, "m") / np.cos(pressure_angle),
    pr_angle=pressure_angle,
)
rotor2 = rs.Rotor(
    shaft2,
    [gear2, rs.DiskElement(n=2, m=7.45, Id=0.0745, Ip=0.149)],
    [
        rs.BearingElement(n=1, kxx=10.1e6, kyy=41.6e6, cxx=3e3),
        rs.BearingElement(n=3, kxx=10.1e6, kyy=41.6e6, cxx=3e3),
    ],
)

multi_rotor = rs.MultiRotor(
    rotor1,
    rotor2,
    coupled_nodes=(4, 0),  # gear node of each rotor, in its own numbering
    gear_mesh_stiffness=1e8,  # N/m
    orientation_angle=0.0,  # line of gear centers from the x axis (rad)
    position="below",  # where the driven rotor is drawn
)
fig = multi_rotor.plot_rotor()
```

- The driven rotor nodes are numbered after those of the driving rotor (`multi_rotor.driven_nodes`); here the turbine disk, node 2 of `rotor2`, is node 7.
- `multi_rotor.mesh.gear_ratio` is the driven-to-driving speed ratio. The driven shaft turns the other way: `multi_rotor.to_shaft_rotation(speed, node)` converts a driving rotor speed, angle or acceleration to the shaft of `node`, with its sign.
- Pass the arguments after `backlash` (`mesh_damping_ratio`, `orientation_angle`, `position`, `min_w`, `max_w`, `rated_w`, `tag`) by keyword. `min_w`, `max_w` and `rated_w` are driving rotor speeds, as on a `Rotor`: `run_ucs` and `run_level1` take the bearing stiffness at `rated_w`.
- `backlash` and `square_varying_stiffness` must hold every documented key, even with `"enable": False`, or `ValueError` is raised.
- More than two shafts: nest the multi-rotors. The first coupled node is then in the numbering of the nested multi-rotor: `rs.MultiRotor(multi_rotor, rotor3, coupled_nodes=(n_gear, n_gear3), ...)`. Gear ratios multiply along the train.

## Analyses

```python
speed = Q_(1500, "RPM")  # driving rotor speed
modal = multi_rotor.run_modal(speed, num_modes=26)

speed_range = Q_(np.arange(0, 5000, 100), "RPM").to("rad/s").m
campbell = multi_rotor.run_campbell(speed_range, frequencies=13)
gear_ratio = multi_rotor.mesh.gear_ratio
fig = campbell.plot(frequency_units="Hz", harmonics=[1, round(gear_ratio, 3)])

# Critical speeds excited by the synchronous (1X) motion of one shaft,
# returned as driving rotor speeds; shaft=0 is the driving shaft and shafts are
# counted along the gear train
crit_driving = multi_rotor.run_critical_speed(num_modes=8)
crit_driven = multi_rotor.run_critical_speed(num_modes=8, shaft=1)

# Each unbalance whirls at the speed of its own shaft
response = multi_rotor.run_unbalance_response(
    node=7,
    unbalance_magnitude=0.449e-3,
    unbalance_phase=0,
    speed_range=np.linspace(0, 100, 51),
)

static = multi_rotor.run_static()
ucs = multi_rotor.run_ucs()
level1 = multi_rotor.run_level1(n=2)  # cross-coupled stiffness at node 2
summary = multi_rotor.summary()  # one row per shaft
fig = summary.plot()
```

- `run_unbalance_response` raises `ValueError` when the unbalances are on shafts that turn at different speeds: run one shaft at a time.
- `run_ucs`, `run_level1` (at `rated_w`, or at speed 0 without it), `run_static` and `summary` rebuild the multi-rotor with its gear topology.
- Faults (`run_rubbing`, `run_crack`, `run_misalignment`) and `run_harmonic_balance_response` use the speed and angle of the shaft where they act; a harmonic whose order is not an integer on its shaft warns.

### Time response with unbalance

```python
dt = 1e-3
t = np.arange(0, 2, dt)
speed1 = Q_(5000, "RPM").to("rad/s").m

# Unbalance force of both shafts, each at its own speed and direction
F = multi_rotor.unbalance_force_over_time(
    node=[2, 7], magnitude=[35.505e-3, 0.449e-3], phase=[0, 0], omega=speed1, t=t
)
time_resp = multi_rotor.run_time_response(speed1, F.T, t)  # F.T: shape (len(t), ndof)
fig = time_resp.plot_2d(node=7)
```

## Concatenate, save and load

```python
longer = multi_rotor + rs.rotor_example()  # added at the end of the driven shaft
longer = rs.rotor_example() + multi_rotor  # added at the start of the driving shaft
longer = rs.MultiRotor.concatenate(rotor1, multi_rotor, rotor2, tag="train")

multi_rotor.save("train.toml")
loaded = rs.Rotor.load("train.toml")  # returns a MultiRotor
```

Rotor files record their class, so `Rotor.load` returns the class that was saved, and results saved with a `MultiRotor` load with it.

## Gear mesh stiffness and backlash

- `GearElementTVMS` builds the gear from its tooth geometry (`material`, `width`, `bore_diameter`, `module`, `n_teeth`, `pr_angle`, `helix_angle`, ...). The mesh stiffness is then computed (`gear_mesh_stiffness` can be left out); `update_mesh_stiffness=True` makes it vary with the mesh angle in time responses, and `square_varying_stiffness={"enable": True, "amplitude_ratio": 0.225}` replaces the profile with a square wave. `multi_rotor.mesh.stiffness` and `multi_rotor.mesh.contact_ratio` give the mean stiffness and the contact ratio; `multi_rotor.mesh.plot_stiffness_profile(n_mesh_period=2)` draws the profile.
- `mesh_damping_ratio` sets the mesh damping `c_m = 2 zeta sqrt(k_m M_eq)` (default 0).
- `backlash={"enable": True, "initial_value": 5e-5, "error_amp": 2e-5, "smooth_operator": False, "sigma": 1e5}` adds the nonlinear contact of a gear pair with clearance. Run it with `run_time_response(..., method="newmark", newmark_type="robust")`. The results (`BacklashResults`, a `TimeResponseResults`) hold `mesh_dynamics` (`transmission_error`, `backlash`, `mesh_force`, `mesh_stiffness`, `center_distance`, `pressure_angle`, `contact_ratio`), with `plot_transmission_error()`, `plot_backlash()`, `plot_mesh_force()` and `plot_dashboard()`.
