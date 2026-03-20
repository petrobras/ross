# ROSS Cookbook

Concise recipes for rotordynamics analyses with ROSS. Each file is self-contained — read only the recipe you need.

| Recipe | File | Key Methods |
|--------|------|-------------|
| Building a rotor from scratch | [building_rotors.md](building_rotors.md) | `Material`, `ShaftElement`, `DiskElement`, `BearingElement`, `Rotor` |
| Modal analysis | [modal_analysis.md](modal_analysis.md) | `run_modal` |
| Campbell diagram | [campbell_diagram.md](campbell_diagram.md) | `run_campbell` |
| Unbalance response | [unbalance_response.md](unbalance_response.md) | `run_unbalance_response` |
| Static analysis | [static_analysis.md](static_analysis.md) | `run_static` |
| Critical speeds | [critical_speed.md](critical_speed.md) | `run_critical_speed` |
| Frequency response (FRF) | [frequency_response.md](frequency_response.md) | `run_freq_response`, `run_forced_response` |
| Time response | [time_response.md](time_response.md) | `run_time_response` |
| UCS and Level 1 stability | [ucs_and_level1.md](ucs_and_level1.md) | `run_ucs`, `run_level1` |
| Fault analysis | [faults.md](faults.md) | `run_rubbing`, `run_crack`, `run_misalignment` |
| Advanced bearings | [bearings_advanced.md](bearings_advanced.md) | `BearingElement` with arrays, fluid flow |
| Common gotchas | [gotchas.md](gotchas.md) | — |

**Maintenance:** These recipes correspond to tutorials and examples in `docs/user_guide/`. When adding a new tutorial or `run_*` method, update the relevant recipe here.

**Sources:** `tutorial_part_1_1` (modeling), `tutorial_part_2_1` (static/modal), `tutorial_part_2_2` (time/frequency), examples 1–32.
