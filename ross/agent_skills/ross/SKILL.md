---
name: ross
description: >-
  Rotordynamic analysis with ROSS (Rotordynamic Open Source Software, the
  ross-rotordynamics Python package). Use when building rotor models
  (shaft elements, disks, bearings, seals, materials) or running rotordynamic
  analyses: modal analysis, Campbell diagram, critical speeds, unbalance
  response, frequency response (FRF), static analysis, time response,
  undamped critical speed (UCS) map, API 617 Level 1 stability, fault
  analyses (rubbing, crack, misalignment), shaft mesh convergence, gas seals
  with speed- and whirl-frequency-dependent coefficients, and three-phase
  induction motors driving the rotor (direct-on-line, V/f and field-oriented
  control drives).
license: Apache-2.0
---

# ROSS Cookbook

Concise recipes for rotordynamics analyses with ROSS. Each file is self-contained — read only the recipe you need.

> Skill version: development (repo checkout)

| Recipe | File | Key Methods |
|--------|------|-------------|
| Building a rotor from scratch | [building_rotors.md](building_rotors.md) | `Material`, `ShaftElement`, `DiskElement`, `BearingElement`, `Rotor` |
| Modal analysis | [modal_analysis.md](modal_analysis.md) | `run_modal` (incl. `frequency=`, `matched_whirl=`) |
| Campbell diagram | [campbell_diagram.md](campbell_diagram.md) | `run_campbell` |
| Unbalance response | [unbalance_response.md](unbalance_response.md) | `run_unbalance_response` |
| Clearance analysis (API 617) | [clearance_analysis.md](clearance_analysis.md) | `run_clearance_analysis`, `api617_unbalance` |
| Static analysis | [static_analysis.md](static_analysis.md) | `run_static` |
| Critical speeds | [critical_speed.md](critical_speed.md) | `run_critical_speed` |
| Frequency response (FRF) | [frequency_response.md](frequency_response.md) | `run_freq_response`, `run_forced_response` |
| Time response | [time_response.md](time_response.md) | `run_time_response` |
| UCS and Level 1 stability | [ucs_and_level1.md](ucs_and_level1.md) | `run_ucs`, `run_level1` |
| Fault analysis | [faults.md](faults.md) | `run_rubbing`, `run_crack`, `run_misalignment` |
| Mesh convergence | [mesh_convergence.md](mesh_convergence.md) | `run_mesh_convergence`, `refine` |
| Induction motor | [motor.md](motor.md) | `MotorElement`, `run_direct_on_line`, `run_with_inverter_vf`, `run_with_inverter_foc`, `run_with_motor` |
| Harmonic balance (no recipe yet) | — | `run_harmonic_balance_response`: see `help(rs.Rotor.run_harmonic_balance_response)` |
| AMB sensitivity (no recipe yet) | — | `run_amb_sensitivity`: see `help(rs.Rotor.run_amb_sensitivity)` |
| AMB sensor-actuator non-collocation (no recipe yet) | — | `run_amb_non_collocation`: see `help(rs.Rotor.run_amb_non_collocation)` |
| Advanced bearings | [bearings_advanced.md](bearings_advanced.md) | `BearingElement` speed / frequency / 2-D tables, fluid-film bearings |
| Seals | [seals.md](seals.md) | `LabyrinthSeal`, `HolePatternSeal`, `HybridSeal`, whirl-dependent tables, `matched_whirl` |
| Common gotchas | [gotchas.md](gotchas.md) | — |

All values are SI internally: speed in rad/s, stiffness in N/m, damping in N·s/m, unbalance in kg·m. Convert with `rs.Q_(value, "unit")`, e.g. `rs.Q_(4000, "RPM").to("rad/s").m`.

If the recipes disagree with the installed ROSS (missing methods, changed signatures), the skill may be stale — compare the version above with `python -c "import ross; print(ross.__version__)"` and re-run `ross-install-skill` after upgrading.
