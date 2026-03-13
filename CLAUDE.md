# CLAUDE.md — ROSS

## Using ROSS for Analysis

### Core Workflow

```python
import ross as rs
import numpy as np

# Elements → Rotor → Analysis → Results
steel = rs.Material(name="steel", rho=7810, E=211e9, G_s=81.2e9)
shaft = [rs.ShaftElement(L=0.05, idl=0, odl=0.05, material=steel) for _ in range(6)]
disks = [
    rs.DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28),
    rs.DiskElement.from_geometry(n=4, material=steel, width=0.07, i_d=0.05, o_d=0.35),
]
bearings = [rs.BearingElement(n=0, kxx=1e6, cxx=0), rs.BearingElement(n=6, kxx=1e6, cxx=0)]
rotor = rs.Rotor(shaft, disks, bearings)
modal = rotor.run_modal(speed=0)
modal.plot_mode_2d(0)
```

### Analysis Methods

| Method | Purpose | Returns | Key Plots | Cookbook |
|--------|---------|---------|-----------|---------|
| `run_modal(speed)` | Natural frequencies and mode shapes | `ModalResults` | `plot_mode_2d()`, `plot_mode_3d()`, `plot_orbit()` | [modal_analysis](docs/cookbook/modal_analysis.md) |
| `run_campbell(speed_range)` | Campbell diagram | `CampbellResults` | `plot()`, `plot_with_mode_shape()` | [campbell_diagram](docs/cookbook/campbell_diagram.md) |
| `run_critical_speed()` | Critical speeds and damping ratios | `CriticalSpeedResults` | — | [critical_speed](docs/cookbook/critical_speed.md) |
| `run_static()` | Static deformation and forces | `StaticResults` | `plot_deformation()`, `plot_bending_moment()`, `plot_shearing_force()`, `plot_free_body_diagram()` | [static_analysis](docs/cookbook/static_analysis.md) |
| `run_unbalance_response(node, ...)` | Unbalance response | `ForcedResponseResults` | `plot_magnitude(probe)`, `plot_phase(probe)`, `plot_bode(probe)`, `plot_deflected_shape(speed)` | [unbalance_response](docs/cookbook/unbalance_response.md) |
| `run_forced_response(force, ...)` | General forced response | `ForcedResponseResults` | same as above | [frequency_response](docs/cookbook/frequency_response.md) |
| `run_freq_response()` | Frequency response function (FRF) | `FrequencyResponseResults` | `plot_magnitude(inp, out)`, `plot_phase(inp, out)`, `plot_polar_bode(inp, out)` | [frequency_response](docs/cookbook/frequency_response.md) |
| `run_time_response(speed, F, t)` | Time-domain response | `TimeResponseResults` | `plot_1d(probe)`, `plot_2d(node)`, `plot_3d()`, `plot_dfft(probe)` | [time_response](docs/cookbook/time_response.md) |
| `run_ucs()` | Undamped critical speed map | `UCSResults` | `plot()` | [ucs_and_level1](docs/cookbook/ucs_and_level1.md) |
| `run_level1()` | API 617 Level 1 stability | `Level1Results` | `plot()` | [ucs_and_level1](docs/cookbook/ucs_and_level1.md) |
| `run_misalignment(...)` | Misalignment fault analysis | `TimeResponseResults` | `plot_1d()`, `plot_2d()`, `plot_dfft()` | [faults](docs/cookbook/faults.md) |
| `run_rubbing(...)` | Rubbing fault analysis | `TimeResponseResults` | same as above | [faults](docs/cookbook/faults.md) |
| `run_crack(...)` | Crack fault analysis | `TimeResponseResults` | same as above | [faults](docs/cookbook/faults.md) |
| `run_harmonic_balance_response(...)` | Harmonic balance steady-state | `HarmonicBalanceResults` | `plot()` | — |
| `run_amb_sensitivity(...)` | AMB sensitivity analysis | `SensitivityResults` | `plot()`, `plot_time_results()` | — |

### Units

All values are SI internally:

- **Speed**: rad/s (convert with `rs.Q_(4000, "RPM").to("rad/s").m`)
- **Stiffness**: N/m
- **Damping**: N·s/m
- **Unbalance magnitude**: kg·m
- **Mass moment of inertia**: kg·m²

Use `rs.Q_(value, "unit")` for pint-based unit conversion.

### Example Rotors

| Function | Description |
|----------|-------------|
| `rs.rotor_example()` | Simple rotor: 6 shafts, 2 disks, 2 bearings |
| `rs.compressor_example()` | Industrial compressor: 91 shafts, 7 disks, 14 bearings/seals |
| `rs.rotor_example_6dof()` | 6-DOF rotor with axial stiffness |
| `rs.rotor_example_with_damping()` | Rotor with internal material damping |
| `rs.coaxrotor_example()` | Coaxial rotor (two concentric shafts) |
| `rs.rotor_amb_example()` | Rotor with active magnetic bearings |

### Save / Load

```python
rotor.save("my_rotor.toml")
rotor = rs.Rotor.load("my_rotor.toml")
```

### Cookbook

For complete analysis recipes, read the relevant file in `docs/cookbook/`. See the [index](docs/cookbook/README.md) for all available recipes.

---

## Development

### Project Overview

ROSS (Rotordynamic Open Source Software) is a Python library for rotordynamic analysis. Package name: `ross-rotordynamics`.

### Git Workflow

- **Main branch**: `main`
- **Upstream remote**: `upstream` → `petrobras/ross` (PRs target this repo)
- **Fork remote**: `origin` → the developer's personal fork
- Push development branches to `origin`, then open PRs against `upstream/main`

### Build & Install

```bash
pip install -e ".[dev]"    # development install with test/lint/docs deps
```

Requires Python >= 3.9.

### Testing

```bash
pytest ross                 # run from repo root
```

- Doctests enabled via `--doctest-modules` (configured in `pytest.ini`)
- Tests live in `ross/tests/`, one file per module (e.g. `test_shaft_element.py`, `test_rotor_assembly.py`)
- **No test classes** — all tests are plain functions (`def test_*():`)
- Shared setup goes in `@pytest.fixture` functions at the top of the file
- Use `assert_allclose` / `assert_almost_equal` from `numpy.testing` for numerical comparisons
- Docstring examples are tested by CI; use `# doctest: +ELLIPSIS` with `...` for truncated output

### Linting & Formatting

Ruff handles both linting and formatting. Pre-commit hooks are configured.

```bash
ruff check ross             # lint
ruff format ross            # format
pre-commit run --all-files  # run all hooks
```

- Double quotes for strings and docstrings
- Quote style enforced in `[tool.ruff.format]` and `[tool.ruff.lint.flake8-quotes]`

### Code Conventions

#### Naming
- **PascalCase** for classes, **snake_case** for functions/methods
- Names should be self-explanatory — do not use comments to split or label sections of code; rely on clear class and function names instead

#### Docstrings
- NumPy-style: `Parameters`, `Returns`, `Examples`, `References` sections
- First line in imperative mood, ending with a period
- Type info goes in docstrings, **not** type annotations

#### Element Pattern
All elements inherit from `Element` ABC (`ross/element.py`) and must implement:
- `M()` — mass matrix
- `K(frequency)` — stiffness matrix
- `C(frequency)` — damping matrix
- `G()` — gyroscopic matrix
- `dof_mapping()` — degree-of-freedom mapping

#### Common Patterns
- **Factory classmethods**: `from_geometry()`, `from_table()` for alternative constructors
- **Example functions**: `*_example()` returning simple instances for use in doctests
- **`@check_units` decorator**: pint unit handling on `__init__` methods
- **Serialization**: TOML-based via `save()` / `load()` methods
- **Visualization**: Plotly with custom ROSS theme; results objects have `.plot_*()` methods
- **Matrix formatting**: use `# fmt: off` / `# fmt: on` to preserve matrix layout

### Architecture

```
Elements → Rotor assembly → .run_*() analysis methods → Results objects with .plot_*() methods
```

Key modules:
- `ross/element.py` — base `Element` ABC
- `ross/shaft_element.py`, `ross/disk_element.py`, `ross/bearing_seal_element.py` — core elements
- `ross/rotor_assembly.py` — `Rotor` class assembling global matrices and running analyses
- `ross/results.py` — results containers with plotting methods

### Key Dependencies

numpy, scipy, plotly, pandas, pint, numba, toml
