# AGENTS.md — ROSS

Instructions for coding agents (Claude Code, Codex, Copilot, Cursor, …) and for the people working with them. The first half explains how to use ROSS; the second half explains how to change it.

## Using ROSS

### Core workflow

```python
import ross as rs

# Elements → Rotor → Analysis → Results
steel = rs.Material(name="steel", rho=7810, E=211e9, G_s=81.2e9)
shaft = [rs.ShaftElement(L=0.05, idl=0, odl=0.05, material=steel) for _ in range(6)]
disks = [
    rs.DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28),
    rs.DiskElement.from_geometry(n=4, material=steel, width=0.07, i_d=0.05, o_d=0.35),
]
bearings = [
    rs.BearingElement(n=0, kxx=1e6, cxx=0),
    rs.BearingElement(n=6, kxx=1e6, cxx=0),
]
rotor = rs.Rotor(shaft, disks, bearings)
modal = rotor.run_modal(speed=0)
modal.plot_mode_2d(0)
```

### Analysis methods

| Method | Purpose | Returns | Key plots | Recipe |
|--------|---------|---------|-----------|--------|
| `run_modal(speed)` | Natural frequencies and mode shapes | `ModalResults` | `plot_mode_2d()`, `plot_mode_3d()`, `plot_orbit()` | [modal_analysis](ross/agent_skills/ross/modal_analysis.md) |
| `run_campbell(speed_range)` | Campbell diagram | `CampbellResults` | `plot()`, `plot_with_mode_shape()` | [campbell_diagram](ross/agent_skills/ross/campbell_diagram.md) |
| `run_critical_speed()` | Critical speeds and damping ratios | `CriticalSpeedResults` | — | [critical_speed](ross/agent_skills/ross/critical_speed.md) |
| `run_static()` | Static deformation and forces | `StaticResults` | `plot_deformation()`, `plot_bending_moment()`, `plot_shearing_force()`, `plot_free_body_diagram()` | [static_analysis](ross/agent_skills/ross/static_analysis.md) |
| `run_unbalance_response(node, ...)` | Unbalance response | `ForcedResponseResults` | `plot_magnitude(probe)`, `plot_phase(probe)`, `plot_bode(probe)`, `plot_deflected_shape(speed)` | [unbalance_response](ross/agent_skills/ross/unbalance_response.md) |
| `run_forced_response(force, ...)` | General forced response | `ForcedResponseResults` | same as above | [frequency_response](ross/agent_skills/ross/frequency_response.md) |
| `run_freq_response()` | Frequency response function (FRF) | `FrequencyResponseResults` | `plot_magnitude(inp, out)`, `plot_phase(inp, out)`, `plot_polar_bode(inp, out)` | [frequency_response](ross/agent_skills/ross/frequency_response.md) |
| `run_time_response(speed, F, t)` | Time-domain response | `TimeResponseResults` | `plot_1d(probe)`, `plot_2d(node)`, `plot_3d()`, `plot_dfft(probe)` | [time_response](ross/agent_skills/ross/time_response.md) |
| `run_ucs()` | Undamped critical speed map | `UCSResults` | `plot()` | [ucs_and_level1](ross/agent_skills/ross/ucs_and_level1.md) |
| `run_level1()` | API 617 Level 1 stability | `Level1Results` | `plot()` | [ucs_and_level1](ross/agent_skills/ross/ucs_and_level1.md) |
| `run_misalignment(...)` | Misalignment fault | `TimeResponseResults` | `plot_1d()`, `plot_2d()`, `plot_dfft()` | [faults](ross/agent_skills/ross/faults.md) |
| `run_rubbing(...)` | Rubbing fault | `TimeResponseResults` | same as above | [faults](ross/agent_skills/ross/faults.md) |
| `run_crack(...)` | Crack fault | `TimeResponseResults` | same as above | [faults](ross/agent_skills/ross/faults.md) |
| `run_harmonic_balance_response(...)` | Harmonic balance steady state | `HarmonicBalanceResults` | `plot()` | — |
| `run_amb_sensitivity(...)` | AMB sensitivity | `SensitivityResults` | `plot()`, `plot_time_results()` | — |
| `run_amb_non_collocation(magnetic_bearing, ...)` | AMB sensor-actuator non-collocation | `AmbNonCollocationResults` | `plot()`, `plot_mode_shape()`, `plot_sensor_position_map()` | — |
| `run_clearance_analysis(speed_range, Nma, Nmc, probes, ...)` | API 617 close-clearance check | `ClearanceResults` | `plot()`, `plot_response()`, `plot_probe_response()` | [clearance_analysis](ross/agent_skills/ross/clearance_analysis.md) |
| `run_mesh_convergence(rtol, frequencies, strategy)` | Shaft discretization with converged natural frequencies | `MeshConvergenceResults` | `plot()` | [mesh_convergence](ross/agent_skills/ross/mesh_convergence.md) |
| `run_with_motor(t, node, unbalance_magnitude, unbalance_phase, drive_mode, ...)` | Time response driven by the induction-motor torque | `TimeResponseResults` (motor simulation in `.motor_results`: `MotorResponseResults`) | `plot_1d(probe)`, `plot_2d(node)`, `plot_3d()`, `plot_dfft(probe)`; `motor_results.plot_torque()`, `plot_speed()` | [motor](ross/agent_skills/ross/motor.md) |

### Units

All values are SI internally: speed in rad/s, stiffness in N/m, damping in N·s/m, unbalance in kg·m, mass moment of inertia in kg·m². Convert with pint: `rs.Q_(4000, "RPM").to("rad/s").m`.

### Example rotors

| Function | Description |
|----------|-------------|
| `rs.rotor_example()` | Simple rotor: 6 shafts, 2 disks, 2 bearings |
| `rs.compressor_example()` | Industrial compressor: 91 shafts, 7 disks, 14 bearings/seals |
| `rs.rotor_example_6dof()` | 6-DOF rotor with axial stiffness |
| `rs.rotor_example_with_damping()` | Rotor with internal material damping |
| `rs.rotor_example_amb_simple()` | Rotor with active magnetic bearings |
| `rs.rotor_example_amb_complex_controllers()` | AMB rotor with complex controllers |
| `rs.rotor_example_amb_general_controllers()` | AMB rotor with general controllers |
| `rs.motor_example()` | 1.5 hp, 127 V / 60 Hz, 4-pole induction motor (`MotorElement`) at node 0 |

### Save, load, interface, recipes

- `rotor.save("my_rotor.toml")` and `rs.Rotor.load("my_rotor.toml")`; results objects have the same pair.
- `pip install "ross-rotordynamics[interface]"`, then `ross-interface` starts the local web interface (`ross/interface/`). `ross-interface --selftest` runs every analysis without a browser.
- Complete analysis recipes live in `ross/agent_skills/ross/` (index: [SKILL.md](ross/agent_skills/ross/SKILL.md)). The folder ships in the package as an [Agent Skill](https://agentskills.io), and `ross-install-skill` copies it into the skills directory of the coding agents found on the machine.

---

## Developing ROSS

### What a contribution must meet

ROSS judges the change, not how it was produced. Code written entirely by an agent and PRs opened by an agent are welcome, and nobody is asked which tool wrote what. Every PR is held to the same bar:

1. **In scope**: it belongs in ROSS (see [Scope](#scope)).
2. **Architecturally consistent**: it follows the Element → Rotor → `run_*` → Results pattern (see [Architecture](#architecture)).
3. **Honestly tested**: existing tests are not weakened, and new physics is checked against an independent reference (see [Tests](#tests)).
4. **Readable**: docstrings, docs and the PR text say what is needed and nothing more (see [Writing](#writing)).
5. **Described**: the PR text follows `.github/PULL_REQUEST_TEMPLATE.md`, and CI is green.

### Setup and workflow

- Python >= 3.13. Development install: `pip install -e ".[dev]"`.
- Remotes: `upstream` → `petrobras/ross`, `origin` → your fork. Branch from `upstream/main`, push to `origin`, open the PR against `petrobras/ross:main`.
- Tests: `pytest ross` from the repository root, or `pytest ross -n auto` in parallel. `--doctest-modules` is on (`pytest.ini`). `ross/conftest.py` caps the BLAS thread pools to one thread per process; without it, xdist workers oversubscribe the cores.
- Lint and format: `ruff check ross` and `ruff format ross` (also run by `pre-commit run --all-files`). Double quotes for strings and docstrings.
- ROSS 2 migration: `ross_2to3 PATH...` (`ross/ross_2to3/`) converts v2 files, scripts and notebooks. `ross/ross_2to3/renames.py` is the single source of truth for renamed API; every entry must appear in `docs/release_notes/version-3.0.0.rst` (checked by `test_ross_2to3.py`).

### Scope

ROSS models rotor-bearing systems and runs rotordynamic analyses on them. When a change adds a new subpackage under `ross/`, a new runtime dependency in `requirements.txt`, or a new physical domain, say so in the PR's **Scope** section and explain why it belongs here. If you are unsure, ask the maintainers before building it: a short issue or a draft PR with the summary is enough.

- **Core**: finite-element rotor models (shafts, disks, couplings, point masses); bearing and seal models, including fluid-film and magnetic bearings; multi-rotor and coaxial assemblies; electrical machines and drives coupled to the rotor; the analyses in the table above; faults (crack, rubbing, misalignment); stochastic analysis and model reduction of rotor models; API 617/684 checks; plotting, save/load, the interface, `ross_2to3` and the agent skill.
- **Belongs elsewhere**: tools whose subject is not the rotor-bearing system, even when they feed it. They live in separate packages that ROSS can depend on or that can depend on ROSS. Example: compressor thermodynamic performance lives in `ccp`.
- **Not decided yet, ask first**

Maintainers record each new scope decision as a line in this section.

### Architecture

```
Elements → Rotor assembly → Rotor.run_*() → Results objects with .plot_*() and save/load
```

| Module | Role |
|--------|------|
| `ross/element.py` | `Element` ABC |
| `ross/shaft_element.py`, `disk_element.py`, `bearing_seal_element.py`, `coupling_element.py`, `point_mass.py` | Core elements |
| `ross/bearings/`, `ross/seals/` | Bearing and seal models |
| `ross/motors/` | `MotorElement`, electrical sources and inverters (`SourceAC`, `InverterVF`, `InverterFOC`), `MotorResponseResults`, `PhaseResults` |
| `ross/rotor_assembly.py`, `ross/multi_rotor/` | `Rotor`, `MultiRotor`: assembly and analyses |
| `ross/results.py` | `Results` ABC and the rotor results classes; the bearing, multi-rotor, motor and stochastic results classes live in their subpackages |
| `ross/units.py` | `@check_units` and `Q_` |

Rules, each with the safe way to follow it:

- **Elements** subclass `Element` and implement `M()`, `K(frequency)`, `C(frequency)`, `G()` and `dof_mapping()`. Unit handling goes through `@check_units` on `__init__`. A new element must build inside `Rotor` and `MultiRotor`, and must round-trip `save`/`load`.
- **Analyses are `run_*` methods on the rotor classes**, never on elements or free functions. Each returns an instance of a `Results` subclass. A new analysis adds a row to the table in this file, a recipe or recipe section in `ross/agent_skills/ross/`, and a row in that folder's `SKILL.md`.
- **Results** subclass `Results`. Plots are `plot_*` methods built with Plotly and the ROSS theme. A new results class must round-trip `save`/`load`.
- **Units**: values are SI inside ROSS. Conversion happens only at the boundary (`@check_units`, `Q_`).
- **Reuse before adding**: search for an existing class, helper or results type before creating a parallel one. A second way to do the same thing is a defect.
- **Public API changes**: every renamed, moved or removed public name (attributes included), changed signature and changed default must be listed in the PR's **API changes** section, with the reason. A moved import path counts as a rename. The maintainers decide whether a change needs a deprecated alias. `ross/ross_2to3/renames.py` covers only the v2 → v3 migration.
- **Dependencies**: a new runtime dependency is a scope question; justify it in the PR.
- **Conventions**: PascalCase classes and snake_case functions; alternative constructors as classmethods (`from_geometry()`, `from_table()`); `*_example()` functions return ready instances for doctests; TOML serialization through `save()`/`load()`; `# fmt: off` / `# fmt: on` around matrices to keep their layout; type information in docstrings, not type annotations; no comments that split or label sections of code; use clear names instead.

### Tests

- Tests live in `ross/tests/`, one file per module (`test_shaft_element.py`, `test_rotor_assembly.py`, …). They are plain functions, with no test classes. Shared setup goes in `@pytest.fixture` functions at the top of the file. Compare numbers with `assert_allclose` / `assert_almost_equal` from `numpy.testing`. Docstring examples run in CI; use `# doctest: +ELLIPSIS` with `...` for truncated output.
- **Never weaken a check to make it pass.** Do not loosen `rtol`/`atol`/`decimal`, remove assertions, delete tests, add `skip`/`xfail`, or edit reference data or test configuration (`conftest.py`, `pytest.ini`, coverage and CI settings) to get a green run. If an existing expectation is wrong, change it and explain the physical or numerical reason in the PR's **Changes to existing tests** section. Moving a test to another file is fine; say so.
- **A tolerance tighter than the computation can deliver is a wrong expectation**, and changing it is not weakening. Sources: platform round-off (LAPACK/BLAS builds, ARPACK start vectors, reduction order), the stopping tolerance of an iterative solver, ill-conditioning, a value near zero that needs an `atol` scaled to the problem, the confidence interval of a stochastic result, or the digits a reference publishes. Name the source and size the tolerance from it (solver tolerance, condition number times machine epsilon, about ten times the spread seen across platforms), not from the failing value. Keep it well below the effect the test checks, and show that the test still fails on the defect it targets. Use one tolerance on every platform, and give the reason in a comment next to it. When the noise is not what the test is about, prefer removing it: compare matrices or invariants, sort modes, fix the seed. A change that moves results is never a reason to widen a tolerance.
- **Reference values must be independent of the code under test.** Valid sources are literature, an analytical or limiting case, or a different ROSS code path. Never use a value produced by running the new code. Cite the source next to the test.
- **Cover a range**, not a single parameter point: several speeds, geometries or load cases where the physics changes.
- **A bug fix comes with a test that fails without the fix.**
- Do not special-case inputs that tests use, swallow exceptions, or mock the physics being tested.

### Writing

This applies to docstrings, documentation pages, notebooks and PR text.

- Docstrings are NumPy style: `Parameters`, `Returns`, `Examples`, `References`. The first line is in the imperative mood and ends with a period. A parameter description gives its meaning and unit; it does not restate the name.
- **Length follows content.** There is no word limit. A new physical model may need a long docstring or PR text, and that is fine when every sentence tells the reader something they need. Cut sentences that:
  - repeat the same information in several places;
  - restate what the code or the signature already says;
  - describe how the change was produced instead of what it does;
  - use filler or promotional words ("it is worth noting", "robust", "comprehensive", "seamlessly");
  - add emoji.
- Derivations and theory belong in the documentation pages, with references in `docs/references.bib`. A docstring points to them.
- Extend an existing tutorial or example before adding a new page.

### Pull requests

To prepare a PR, follow the `ross-pr-create` skill in `.agents/skills/ross-pr-create/SKILL.md`. Agents with skill support load it automatically; Claude Code finds it through `.claude/skills/`. It runs `python tools/pr_facts.py`, which compares the branch with `main` without executing it and lists the scope, API, architecture, test-integrity and writing signals a reviewer will look at.

- One concern per PR. Keep core-library changes, interface changes and vendored third-party code in separate PRs. Split large work into a stack of PRs that can each be reviewed on their own.
- Plan however you like, locally. The plan is not committed; the PR text carries what a reviewer needs: scope, the design decisions to confirm, API changes, evidence, changes to existing tests and documentation. Fill in every section of the template, writing "none" where nothing applies.
- Do not paste session logs, local notes or agent transcripts into the PR.
- Keep the cookbook in sync: the recipes in `ross/agent_skills/ross/` follow the tutorials in `docs/user_guide/`: `tutorial_modeling_part_1` and `tutorial_modeling_part_2` for modeling, `tutorial_analyses_part_1` for static and modal, `tutorial_analyses_part_2` for time and frequency, `tutorial_motor` for motors, and examples 1–32. When a tutorial or `run_*` method changes, update the matching recipe and the `SKILL.md` index. The skill folder must stay self-contained: recipes only link to files inside it.

### Reviewing a pull request

When asked to review a PR, follow the `ross-pr-review` skill in `.agents/skills/ross-pr-review/SKILL.md`. In short: check it against the five points in [What a contribution must meet](#what-a-contribution-must-meet), then look for correctness problems. Report only findings that affect correctness, scope, architecture, test integrity or readability. Give each one a `file:line` reference and, for correctness claims, a way to reproduce it.

**Never post to GitHub on your own.** Comments, reviews, approvals, labels and pushes to someone else's branch happen only after the person who asked for the review has read the findings and explicitly approved posting that content.
