# Induction Motor

Source: `docs/user_guide/tutorial_motor.ipynb`

`ElectricMotor` models a three-phase induction motor (TPIM) in a rotor-flux-oriented synchronous frame. Besides its electrical parameters, it carries the structure of the motor as a mechanical model, chosen by the arguments given to it:

| Model | Arguments | What `Rotor.add_motor` adds to the rotor | `motor.n_torque` |
|-------|-----------|------------------------------------------|------------------|
| Lumped | `m`, `Ip` and, optionally, `Id` | A disk with the mass and the inertias of the motor, at node `n` | `n` |
| Finite element | `shaft_elements` and, optionally, `disk_elements`, `bearing_elements`, `point_mass_elements` | The rotor of the motor, concatenated to the rotor of the load | Node shared by the two rotors |

`Rotor.run_with_motor` applies the motor torque to the torsional DOF (`theta`) of node `motor.n_torque`, or of `electric_torque_node` if given. Use the lumped model when the motor is stiff compared to the rest of the machine; the finite element model adds the flexibility of the motor shaft and bearings and allows a coupling between the motor and the load.

## Motor model

```python
import numpy as np
import ross as rs
from ross.units import Q_

# electrical parameters, shared by both mechanical models
electrical = dict(
    power_rated=Q_(1.5, "hp"),
    voltage_rated=127,  # V (phase)
    speed_rated=Q_(1710, "RPM"),
    frequency_rated=Q_(60, "Hz"),
    n_poles=4,
    stator_resistance=2.5,  # Ohm
    rotor_resistance=1.8,  # Ohm
    stator_reactance=1.3,  # Ohm
    rotor_reactance=1.3,  # Ohm
    mutual_reactance=43.08,  # Ohm
    viscosity_coeff=0.0,
)

# lumped model: mass and inertias at node n
motor = rs.ElectricMotor(
    n=0,
    tag="TPIM",
    m=10.0,  # kg
    Ip=0.0372,  # kg·m²
    Id=0.0,  # kg·m², optional
    **electrical,
)

motor = rs.motor_example()  # the same 1.5 hp, 127 V / 60 Hz, 4-pole motor at node 0
motor.T_rated  # rated torque (N·m), the reference for load_torque_ratio
```

### Finite element model

```python
# the motor rotor, numbered from node 0; m, Ip and Id are computed from the elements
motor_fem = rs.ElectricMotor(
    n=0,  # first node of the load rotor: motor on the left (last node: on the right)
    tag="TPIM",
    shaft_elements=[
        rs.ShaftElement(L=0.05, idl=0, odl=0.03, material=rs.steel) for _ in range(5)
    ],
    disk_elements=[  # rotor core
        rs.DiskElement.from_geometry(
            n=2, material=rs.steel, width=0.08, i_d=0.03, o_d=0.15
        )
    ],
    bearing_elements=[
        rs.BearingElement(n=0, kxx=1e6, cxx=2e3),
        rs.BearingElement(n=4, kxx=1e6, cxx=2e3),
    ],
    **electrical,
)
motor_fem.m, motor_fem.Ip  # 11.98 kg, 0.0312 kg·m²
motor_fem.mechanical_model  # the rotor of the motor

# the motor also answers to the Rotor methods it does not define
fig = motor_fem.plot_rotor()
modal = motor_fem.run_modal(speed=0)
campbell = motor_fem.run_campbell(np.linspace(0, Q_(4000, "RPM").to("rad/s").m, 21))
fig = campbell.plot()
```

- With `shaft_elements`, the attributes of the motor (`m`, `Ip`, `tag`, `save()`...) and the arguments of `ElectricMotor` are not forwarded to the rotor: use `motor_fem.mechanical_model.shaft_elements`, for example. The lumped motor has no rotor and forwards nothing.
- Either `m` and `Ip` (lumped model) or `shaft_elements` (finite element model) is required. Without `shaft_elements`, a missing `m` or `Ip`, or any `disk_elements`, `bearing_elements` or `point_mass_elements`, raises `ValueError`; with `shaft_elements`, `m`, `Ip` and `Id` raise `ValueError`. The motor-only simulations (`run_direct_on_line`, `run_with_inverter_vf`, `run_with_inverter_foc`) use `Ip` plus the `Ip_load` argument of the method, the polar inertia of the driven load in kg·m² (zero by default).
- The direct-on-line supply voltage and frequency are the `voltage_net` and `frequency_net` arguments of `run_direct_on_line`, which default to the rated values. `ElectricMotor` has no `frequency_net`; its `voltage_net` does not reach the supply and only sets the network short-circuit impedance. `run_with_motor` does not pass a supply voltage or frequency, so its `"DOL"` mode always runs at the rated values. The inverters set the voltage and frequency themselves.
- Speeds and frequencies are rad/s internally; pass `Q_(..., "RPM")` or `Q_(..., "Hz")` to convert.

## Estimating the circuit parameters

When only rated data are available, `ElectricMotor.from_nameplate` and `ElectricMotor.from_catalog` estimate the stator and rotor resistances and reactances and the magnetizing reactance, and return the `ElectricMotor`. Voltage and current are per phase of the star equivalent (phase voltage = line voltage / sqrt(3)); the core loss branch of the methods is not represented in the motor model.

```python
motor = rs.ElectricMotor.from_nameplate(
    n=0,
    power_rated=Q_(25, "hp"),
    voltage_rated=460 / np.sqrt(3),  # V (phase)
    speed_rated=Q_(1724.6, "RPM"),
    efficiency=0.83,
    power_factor=0.90,
    m=10.0,  # kg
    Ip=0.1,  # kg·m²
    current_rated=31.25,  # A (phase)
    starting_current_ratio=4.6,
    motor_category="N",  # N, H or D: splits the leakage reactance
    method="asdp",  # "asdp", "kueck" or "lee"
)

motor = rs.ElectricMotor.from_catalog(
    n=0,
    power_rated=Q_(25, "hp"),
    voltage_rated=460 / np.sqrt(3),
    speed_rated=Q_(1724.6, "RPM"),
    efficiency=0.83,
    power_factor=0.90,
    m=10.0,
    Ip=0.1,
    starting_torque_ratio=2.0,
    maximum_torque_ratio=2.5,
    method="haque2",  # "haque1", "natarajan_misra" or "haque2"
)
```

- `haque1` also needs `efficiency_50` and `power_factor_50`; `natarajan_misra` needs `efficiency_75`, `power_factor_75`, `efficiency_50`, `power_factor_50` and both torque ratios.
- Options of a method go in `method_options`, for example `{"known_stator_resistance": 2.44}` for `"kueck"`. The structure of the motor (`m` and `Ip`, or `shaft_elements`) and the other `ElectricMotor` arguments (`tag`, `n_poles`, ...) are accepted as keywords.
- A `UserWarning` is issued when the method does not converge; check the estimated values before using them.

## Motor-only analyses

Each method takes the time array `t` (s) at which results are stored and returns a `MotorResponseResults`.

| Method | Supply | Speed control |
|--------|--------|---------------|
| `motor.run_direct_on_line(t, ...)` | Ideal three-phase source (`SourceAC`), optional harmonics and unbalance | None |
| `motor.run_with_inverter_vf(t, ...)` | SVPWM inverter (`InverterVF`) | Open-loop scalar V/f |
| `motor.run_with_inverter_foc(t, ...)` | SVPWM inverter (`InverterFOC`) | Closed-loop indirect field-oriented control |
| `motor.run(t, element=...)` | Any source object with `get_current_state` | Depends on the source |

### Direct-on-line start

```python
t = np.arange(0, 3.0 + 1e-3, 1e-3)

results_dol = motor.run_direct_on_line(
    t,
    load_torque_entrance_time=1.5,  # s; default is half the simulation time
    load_torque_ratio=1.0,  # fraction of motor.T_rated
)

results_d = motor.run_direct_on_line(
    t,
    load_torque_entrance_time=1.5,
    # amplitudes in % of the supply voltage
    harmonics={"enable": True, "orders": [5, 7], "amplitudes": [3.0, 2.0]},
    unbalances={
        "enable": True,
        "voltage_percent": [-1.0, 2.0, 3.0],  # phases A, B, C (%)
        "angle_deviation": Q_([1.0, 0.0, -2.0], "deg"),  # phases A, B, C
    },
)
```

### Variable frequency drives

```python
results_vf = motor.run_with_inverter_vf(
    t,
    time_step=1e-5,  # s, integration step
    load_torque_entrance_time=1.5,
    frequency_s=Q_(5000, "Hz"),  # IGBT switching frequency
    time_ramp=0.6667,  # s, linear ramp of the reference frequency
    frequency_ref=Q_(30, "Hz"),  # default: half the rated frequency
)

results_foc = motor.run_with_inverter_foc(
    t,
    time_step=1e-5,
    load_torque_entrance_time=1.5,
    frequency_s=Q_(5000, "Hz"),
    time_ramp=0.6667,
    # synchronous frequency; speed reference = frequency_ref / (n_poles / 2)
    frequency_ref=Q_(30, "Hz"),
)
```

- `time_step` must resolve the SVPWM carrier: use at least 20 samples per switching period (`time_step <= 1e-5` s at 5 kHz), otherwise the carrier is aliased and the voltage is wrong.
- FOC runs in closed loop on the motor state, so it cannot be precomputed and is slower than V/f for the same `t` and `time_step`.
- With V/f the speed settles below synchronous because of slip; the FOC speed loop brings it back to synchronous after the load step.

### Results: `MotorResponseResults`

```python
results_dol.t  # integration time points (s); results_dol.t_eval holds the requested t
results_dol.electric_torque  # N·m
results_dol.load_torque  # N·m
results_dol.speed  # rad/s
results_dol.currents  # dict: "a", "b", "c", "alpha", "beta", "d", "q" (A)
results_dol.voltages  # dict: "a", "b", "c" (V)
results_dol.line_voltages  # dict: "ab", "bc", "ca" (V)

# cubic interpolation at t_eval (or pass t_eval=...)
speed_at_t = results_vf.sample_at("speed")
```

### Plotting

```python
fig = results_dol.plot_torque()  # electric and load torque
fig = results_dol.plot_speed(speed_units="RPM")
fig = results_dol.plot_phase_currents(reference_frame="a-b-c")  # or "alpha-beta", "d-q"
fig = results_dol.plot_phase_voltages()
fig = results_dol.plot_line_voltages(time_range=(0.1, 0.3))  # x-axis range (s)

# windowed FFT instead of the time history
fig = results_d.plot_phase_voltages(domain="frequency")
fig = results_d.plot_torque(domain="frequency")

# long inverter runs: downsample each trace (MinMaxLTTB) to n_shown_samples points
fig = results_vf.plot_torque(n_shown_samples=2000)
fig = results_vf.plot_phase_currents(
    reference_frame="d-q", n_shown_samples=2000, time_range=(2.0, 2.5)
)
fig = results_vf.plot_line_voltages(
    domain="frequency", frequency_range=Q_((0, 50e3), "Hz"), n_shown_samples=2000
)
```

Every plot method accepts `domain` (`"time"` or `"frequency"`), `frequency_units`, `frequency_range`, `time_range`, `n_shown_samples` and `fig`. Without `n_shown_samples`, signals are downsampled to an equivalent 0.1 ms step.

## Motor coupled to a rotor: `Rotor.run_with_motor`

```python
base_rotor = rs.rotor_example()
bearings = [
    rs.BearingElement(n=bearing.n, kxx=1e6, cxx=2e3)
    for bearing in base_rotor.bearing_elements
]
disks = [
    rs.DiskElement.from_geometry(
        n=disk.n, material=rs.steel, width=0.04, i_d=0.05, o_d=0.18
    )
    for disk in base_rotor.disk_elements
]
load_rotor = rs.Rotor(base_rotor.shaft_elements, disks, bearings)

# lumped model: a disk with m, Id and Ip at node n; the torque acts at n
rotor_lumped = load_rotor.add_motor(motor)  # returns a new rotor
fig = rotor_lumped.plot_rotor()  # the motor is drawn at its node

# finite element model: motor_fem of the Motor model section, with a coupling
# between the motor and the load. The coupling is the first element of the load
# rotor, so the torque (at the joint node) reaches the load through the coupling
coupling = rs.CouplingElement(
    m_l=0.5,
    m_r=0.5,
    Ip_l=2e-4,
    Ip_r=2e-4,  # kg, kg·m² per hub
    kt_x=1e7,
    kt_y=1e7,  # N/m
    kr_x=1e4,
    kr_y=1e4,  # N·m/rad
    kr_z=5e3,  # torsional, N·m/rad
    o_d=0.06,
    L=0.05,  # drawing only
)
coupled_load_rotor = rs.Rotor(
    [
        coupling,
        *[
            rs.ShaftElement(L=0.25, idl=0, odl=0.05, material=rs.steel)
            for _ in range(6)
        ],
    ],
    [
        rs.DiskElement.from_geometry(
            n=n, material=rs.steel, width=0.07, i_d=0.05, o_d=0.28
        )
        for n in (3, 5)
    ],
    [rs.BearingElement(n=n, kxx=1e6, cxx=2e3) for n in (1, 7)],
)
rotor_fem = coupled_load_rotor.add_motor(
    motor_fem
)  # concatenates the motor rotor and the load rotor
rotor_fem.motor.n_torque  # 5: the joint node (motor side of the coupling), where the torque acts
# the load rotor nodes are shifted by 5: coupling 5-6, disks at nodes 8 and 10

t = np.arange(0, 4.0 + 1e-4, 1e-4)
response = rotor_lumped.run_with_motor(
    t=t,
    unbalance_node=[2],  # unbalance nodes (use [8] for rotor_fem)
    unbalance_magnitude=[5e-4],  # kg·m
    unbalance_phase=[0],  # rad
    drive_mode="DOL",  # "DOL", "VFD_VF" or "VFD_FOC"
    load_torque_entrance_time=2.5,
    load_torque_ratio=1.0,
)
```

- `add_motor` with the finite element model accepts only `n` equal to the first or the last node of the rotor; otherwise it raises `ValueError`. With `n` = first node, the last node of the motor rotor is joined to it and the load nodes are shifted; with `n` = last node, the first node of the motor rotor is joined to it.
- A coupling between the motor and the load goes at the start (motor on the left) or the end (motor on the right) of the load rotor, not in `shaft_elements` of the motor: otherwise the torque is applied on the load side of the coupling and bypasses its torsional stiffness. Build the load shaft with new `ShaftElement` objects: the elements of an assembled rotor keep their node numbers, so `[coupling, *rotor_example().shaft_elements]` fails.
- `electric_torque_node` and `load_torque_node` (arguments of `run_with_motor`) are the nodes where the electromagnetic torque and the load torque are applied. By default both are `motor.n_torque`: the node of the lumped motor, or the node shared by the motor and the load rotors. Give `electric_torque_node` the node of the rotor core of a finite element motor, and `load_torque_node` the node of the driven machine, so that the torque twists the shafts between them. The motor speed is the shaft speed at `electric_torque_node`. `ValueError` if a node is not in the rotor.
- A rotor holds one motor; `add_motor` on a rotor that already has one raises `ValueError`. `rotor.add_elements` does not add motors.
- The finite element model adds the flexibility of the motor shaft and the motor bearings, so the natural frequencies differ from the lumped model (in the tutorial, with the coupling, the second pair moves from 40.4 Hz to 52.2 Hz).
- A rotor with a motor round-trips `save`/`load` with either model. A motor with `shaft_elements` cannot be saved on its own before it is connected to a rotor.
- The rotor model is the only source of inertia: the motor is already in it, as a disk (lumped) or as part of the rotor (finite element). The motor is simulated first with the chosen drive, with the polar inertia of the whole rotor, `rotor.Ip`: the `Ip` of the motor plus `rotor.Ip - motor.Ip` passed as `Ip_load`, which is the inertia of the rest of the rotor. The speed controller of `"VFD_FOC"` is tuned with this inertia. Then the rotor is integrated with Newmark, with the net torque (electric minus load) on the `theta` DOF of `electric_torque_node` (electric) and `load_torque_node` (load) and unbalance forces computed from the motor speed, which varies in time.
- With a `MultiRotor`, the motor speed is the speed of the shaft that holds the motor. `run_with_motor` divides it by the signed speed ratio of that shaft (negative on a driven shaft) to get the speed of the driving rotor that the analysis uses.
- `ac_source_harmonics` / `ac_source_unbalances` are the `harmonics` / `unbalances` dicts of `run_direct_on_line`, used only with `"DOL"`.
- `time_step`, `time_ramp` and `frequency_ref` are the inverter arguments, used only with `"VFD_VF"` and `"VFD_FOC"`. The switching frequency is fixed at 5 kHz.
- `steady_state=True` (default) integrates only the last third of the interval after `load_torque_entrance_time`: `response.t` starts there, not at `t[0]`. The motor start-up and load transient are skipped, but the rotor starts from rest at the start of that window, so the window begins with a new lateral transient. In the run above the amplitude at the start of the window is about 25 % above the steady value and has settled by its end. Use `steady_state=False` (or a longer `t`) when amplitudes matter.
- `F` (shape `(len(t), rotor.ndof)`) adds external forces and is not modified; extra keyword arguments go to the Newmark integrator (`gamma`, `beta`, `tol`, ...) and `Rotor.integrate_system`.
- `run_with_motor` raises `ValueError` if the rotor has no motor (`rotor.motor` is None), or if its torsional DOFs do not form a single train (for example a coupling with `kr_z=0`, which leaves the load without torque).
- The start-up time grows with the polar inertia of the whole rotor, which is larger than `Ip` of the motor. Use a `t` long enough for the shaft to reach its speed, and `load_torque_entrance_time` after it.
- Use a time step suited to the rotor integration (0.1 ms here); finer steps make the Newmark run slow.

### Results

`run_with_motor` returns a `TimeResponseResults` (same plots as `run_time_response`, see [time_response.md](time_response.md)); the motor simulation is in `response.motor_results`, a `MotorResponseResults`.

```python
probe = rs.Probe(2, 0)
fig = response.plot_1d(probe=[probe])
# last revolution, integrating the varying speed
fig = response.plot_2d(node=2, one_revolution=True)
fig = response.plot_dfft(probe=[probe])

fig = response.motor_results.plot_speed()
fig = response.motor_results.plot_torque()
```
