# CLAUDE.md — ROSS Development Conventions

## Project Overview

ROSS (Rotordynamic Open Source Software) is a Python library for rotordynamic analysis. Package name: `ross-rotordynamics`.

## Git Workflow

- **Main branch**: `main`
- **Upstream remote**: `upstream` → `petrobras/ross` (PRs target this repo)
- **Fork remote**: `origin` → the developer's personal fork
- Push development branches to `origin`, then open PRs against `upstream/main`

## Build & Install

```bash
pip install -e ".[dev]"    # development install with test/lint/docs deps
pip install -e ".[mcp]"    # MCP server dependencies
```

Requires Python >= 3.9.

## Testing

```bash
pytest ross                 # run from repo root
```

- Doctests enabled via `--doctest-modules` (configured in `pytest.ini`)
- `ross/mcp/` is excluded from pytest collection (`--ignore=ross/mcp`)
- Tests live in `ross/tests/`, one file per module (e.g. `test_shaft_element.py`, `test_rotor_assembly.py`)
- **No test classes** — all tests are plain functions (`def test_*():`)
- Shared setup goes in `@pytest.fixture` functions at the top of the file
- Use `assert_allclose` / `assert_almost_equal` from `numpy.testing` for numerical comparisons
- Docstring examples are tested by CI; use `# doctest: +ELLIPSIS` with `...` for truncated output

## Linting & Formatting

Ruff handles both linting and formatting. Pre-commit hooks are configured.

```bash
ruff check ross             # lint
ruff format ross            # format
pre-commit run --all-files  # run all hooks
```

- Double quotes for strings and docstrings
- Quote style enforced in `[tool.ruff.format]` and `[tool.ruff.lint.flake8-quotes]`

## Code Conventions

### Naming
- **PascalCase** for classes, **snake_case** for functions/methods
- Names should be self-explanatory — do not use comments to split or label sections of code; rely on clear class and function names instead

### Docstrings
- NumPy-style: `Parameters`, `Returns`, `Examples`, `References` sections
- First line in imperative mood, ending with a period
- Type info goes in docstrings, **not** type annotations

### Element Pattern
All elements inherit from `Element` ABC (`ross/element.py`) and must implement:
- `M()` — mass matrix
- `K(frequency)` — stiffness matrix
- `C(frequency)` — damping matrix
- `G()` — gyroscopic matrix
- `dof_mapping()` — degree-of-freedom mapping

### Common Patterns
- **Factory classmethods**: `from_geometry()`, `from_table()` for alternative constructors
- **Example functions**: `*_example()` returning simple instances for use in doctests
- **`@check_units` decorator**: pint unit handling on `__init__` methods
- **Serialization**: TOML-based via `save()` / `load()` methods
- **Visualization**: Plotly with custom ROSS theme; results objects have `.plot_*()` methods
- **Matrix formatting**: use `# fmt: off` / `# fmt: on` to preserve matrix layout

## Architecture

```
Elements → Rotor assembly → .run_*() analysis methods → Results objects with .plot_*() methods
```

Key modules:
- `ross/element.py` — base `Element` ABC
- `ross/shaft_element.py`, `ross/disk_element.py`, `ross/bearing_seal_element.py` — core elements
- `ross/rotor_assembly.py` — `Rotor` class assembling global matrices and running analyses
- `ross/results.py` — results containers with plotting methods

## Key Dependencies

numpy, scipy, plotly, pandas, pint, numba, toml

## MCP Server

The `ross/mcp/` package provides an MCP server for AI-assisted rotordynamics analysis.

```bash
python -m ross.mcp          # run via stdio transport
ross-mcp                    # CLI entry point
```

Excluded from pytest. Configured in `.mcp.json` for Claude Code auto-discovery.
