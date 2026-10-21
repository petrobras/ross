# Induction Motor

Source: `docs/user_guide/tutorial_motor.ipynb`

`MotorElement` models a three-phase induction motor (TPIM) in a rotor-flux-oriented synchronous frame. It is not a structural element: it adds no mass, stiffness, damping or gyroscopic terms to the rotor matrices. Its node `n` is where the motor torque enters the torsional DOF (`theta`) in `Rotor.run_with_motor`.

## Motor model

```python
import numpy as np
import ross as rs
from ross.units import Q_

motor = rs.MotorElement(
    n=0,
    tag="TPIM",
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
    Ip_motor=0.0372,  # kg·m²
    Ip_load=0.0,  # kg·m²
    viscosity_coeff=0.0,
)

motor = rs.motor_example()  # the same 1.5 hp, 127 V / 60 Hz, 4-pole motor at node 0
motor.T_rated  # rated torque (N·m), the reference for load_torque_ratio
```

- The direct-on-line supply voltage and frequency are the `voltage_net` and `frequency_net` arguments of `run_direct_on_line`, which default to the rated values. The `MotorElement` arguments of the same name do not reach the supply: `voltage_net` only sets the network short-circuit impedance and `frequency_net` is not used. `run_with_motor` does not pass a supply voltage or frequency, so its `"DOL"` mode always runs at the rated values. The inverters set the voltage and frequency themselves.
- Speeds and frequencies are rad/s internally; pass `Q_(..., "RPM")` or `Q_(..., "Hz")` to convert.

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
rotor = rs.Rotor(
    base_rotor.shaft_elements,
    base_rotor.disk_elements,
    bearings,
    motor_element=motor,  # pass by keyword
)
# or: rotor = rs.rotor_example().add_elements([rs.motor_example()])
fig = rotor.plot_rotor()  # the motor is drawn at its node

t = np.arange(0, 3.0 + 1e-4, 1e-4)
response = rotor.run_with_motor(
    t=t,
    node=[2],  # unbalance nodes
    unbalance_magnitude=[5e-4],  # kg·m
    unbalance_phase=[0],  # rad
    drive_mode="DOL",  # "DOL", "VFD_VF" or "VFD_FOC"
    load_torque_entrance_time=1.5,
    load_torque_ratio=1.0,
)
```

- The motor is simulated first with the chosen drive. Then the rotor is integrated with Newmark, with the net torque (electric minus load) on the motor node's `theta` DOF and unbalance forces computed from the motor speed, which varies in time.
- `ac_source_harmonics` / `ac_source_unbalances` are the `harmonics` / `unbalances` dicts of `run_direct_on_line`, used only with `"DOL"`.
- `time_step`, `time_ramp` and `frequency_ref` are the inverter arguments, used only with `"VFD_VF"` and `"VFD_FOC"`. The switching frequency is fixed at 5 kHz.
- `steady_state=True` (default) integrates only the last third of the interval after `load_torque_entrance_time`: `response.t` starts there, not at `t[0]`. The motor start-up and load transient are skipped, but the rotor starts from rest at the start of that window, so the window begins with a new lateral transient. In the run above the amplitudes at the end of the window are still 20-40 % above the steady value. Use `steady_state=False` (or a longer `t`) when amplitudes matter.
- `F` (shape `(len(t), rotor.ndof)`) adds external forces; extra keyword arguments go to the Newmark integrator (`gamma`, `beta`, `tol`, ...) and `Rotor.integrate_system`.
- A rotor holds one motor (`rotor.motor_element`); `run_with_motor` raises `ValueError` if there is none.
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
