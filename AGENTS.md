# AGENTS.md — ROSS Quick Start

```python
import ross as rs
rotor = rs.rotor_example()  # ready-to-use rotor for testing
modal = rotor.run_modal(speed=0)
print(modal.wn[:4])  # first 4 natural frequencies (rad/s)
```

**Core pattern:** Elements → `rs.Rotor(shafts, disks, bearings)` → `rotor.run_*()` → Results with `.plot_*()` methods.

**Units:** All SI — speeds in rad/s, stiffness in N/m, damping in N·s/m, unbalance in kg·m. Use `rs.Q_(value, "unit")` to convert.

**Analysis methods:** `run_modal`, `run_campbell`, `run_critical_speed`, `run_static`, `run_unbalance_response`, `run_forced_response`, `run_freq_response`, `run_time_response`, `run_ucs`, `run_level1`, `run_rubbing`, `run_crack`, `run_misalignment`.

See [CLAUDE.md](./CLAUDE.md) for the full API reference table, and `docs/cookbook/` for complete analysis recipes. See CLAUDE.md `## Development` for build, test, lint, and code conventions.
