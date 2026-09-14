# ROSS Graphical Interface

This is a web-based graphical interface for the **ROSS** library, designed to simplify the modeling and analysis of rotating machinery. The project integrates an interactive JavaScript frontend with a robust Python (Flask) backend, enabling rotor dynamics simulations through a visual and intuitive workflow.

## 🚀 Features

- **Comprehensive Modeling:** Add and edit materials, shaft elements, disks, gears, bearings, seals, couplings, and point masses.
- **Real-Time Visualization:** 3D visualization of the rotor model that updates as elements are added.
- **Advanced Analysis Dashboards:**
    - Campbell Diagram;
    - Unbalanced Critical Speed (UCS);
    - Frequency Response;
    - Time Response;
    - Vibration Modes (2D and 3D);
    - Unbalance Response;
    - Static Analysis.
- **Python Script Generation:** Automatically export your model and analysis settings into a ready-to-run Python script using native ROSS syntax.
- **Data Portability:** Save and load your rotor models and analysis configurations using JSON files.

## 🛠️ Technologies

- **Frontend:** HTML5, CSS3, JavaScript (using Plotly.js for charting and Sortable.js for list management).
- **Backend:** Python 3, Flask (Web Server), ROSS-rotordynamics (Calculation Engine).

## 📂 Project Structure

The project was refactored from four monolithic files into layers. Nothing in
the interface duplicates what ROSS already knows: the element forms are derived
from the installed library by introspection, so a renamed parameter disappears
from the form instead of becoming a `TypeError` at build time.

```
app.py         Entry point: starts the server, opens the browser, answers --selftest
selftest.py    What the executable checks about itself before anyone trusts it
check.py       Runs ruff, pytest and the node batteries; writes check_report.txt
ross-interface.spec  How PyInstaller turns all of this into a program
ruff.toml      Lint and format, copied from ROSS's own configuration
api/           Transport only: routes, request envelope, session token, error handling
services/      What the application does: one runner per analysis, the expression evaluator
domain/        What the application knows: units, nodes, ROSS classes, schema,
               rotor assembly, cache, Python export
frontend/
  index.html   UI structure for the hub, modeling and analysis screens
  style.css    Visual styling and responsive rules
  main.js      Bootstrap
  core/        Store, API client, shared state, schema loader, i18n, persistence
  components/  Form builder, element list, modals, contextual help
  features/    One module per screen: hub, modeling, analysis, multirotor, export
tests/         Python suites, plus tests/js/ behaviour batteries run with node
tools/         Measuring instruments: they assert nothing and change nothing
ci/            The CI workflow, installed at .github/workflows/ in the repository root
```

## 📋 Prerequisites

Python 3.10 or newer. A virtual environment is recommended. `node` is optional
but recommended: without it the behaviour batteries in `tests/js/` are skipped,
and they are the only tests that exercise the screen rather than the source
text.

## 🔧 Installation

```bash
pip install -r interface/requirements.txt
```

ROSS itself is pinned to a specific commit in `requirements.txt`. That is
deliberate: the analysis/degree-of-freedom compatibility table in
`domain/compatibility.py` was **measured** against that commit, and what a
newer version breaks or fixes can only be known by measuring again.

## 💻 How to Execute

```bash
python interface/app.py
```

The browser opens on `http://127.0.0.1:5001/`. Everything runs locally, on the
loopback interface, for a single user and a single session.

## 📦 Building the executable

The interface travels as **source**, not as a binary. That is not only about
size: PyInstaller does not cross-compile, so a Windows program can only be
produced on Windows, a macOS one on macOS and a Linux one on Linux. There is no
single file that could be shipped to everyone, and there is no reason for a
library repository to carry one.

```bash
pip install -r interface/requirements-dev.txt
python -m PyInstaller --noconfirm ross-interface.spec
```

The result is `dist/ross-interface/`, a folder of about 460 MB. Most of that
weight is not rotordynamics: `import ross` requires matplotlib, Pillow,
scikit-learn and Arrow at import time, through `control` and `ccp-performance`.

Before trusting it:

```bash
dist/ross-interface/ross-interface --selftest
```

On Windows, `dist\ross-interface\ross-interface.exe --selftest`.

It builds a rotor and runs all twelve analyses from inside the bundle. That
matters because PyInstaller follows what the code *imports* and not what the
code *opens*, and this interface reads two files that live inside installed
packages: plotly's `plotly.min.js` and ROSS's `new_units.txt`. **A build that
finishes is not a build that works** -- the first three builds of this folder
all finished, and none of them ran.

These are the same commands CI runs on Ubuntu, macOS and Windows
(`ci/interface.yml`), and a test keeps the two texts in step. The instructions
are not documentation that might be right: they are documentation that is
executed on three systems on every change.

## ✅ Tests

```bash
cd interface && python check.py
```

`check.py` is the one verification recipe: ruff, pytest and the node batteries,
with the result written to `check_report.txt` -- including what was **skipped**,
because a silent skip is coverage nobody knows they lost. Plain `pytest
interface` from the repository root works too, and runs less.

The `tests/js/` batteries run under `node` and are executed by the Python suite,
so a single `pytest` covers both. Suites that need ROSS itself are skipped when
it is not installed.

## 🔗 Relationship with the ROSS repository

This folder lives inside the ROSS repository but is deliberately **not** part of
the `ross` package:

- `pytest ross` — the command ROSS CI runs — never reaches it.
- A bare `pytest` at the repository root ignores it too: `conftest.py` here
  drops out of collection whenever pytest's rootdir is not this folder. Without
  that, ROSS's `--doctest-modules` would import every module here and fail on
  the missing Flask.
- `ruff check ross` does not reach it either.
- Flask is declared here, not in ROSS's `requirements.txt`: installing
  `ross-rotordynamics` to write a script should not pull in a web server.

**One line is required in the ROSS `pyproject.toml`.** Its
`[tool.setuptools.packages.find]` resolves in namespace mode, so a top-level
folder is picked up even without an `__init__.py` and would ship inside the
`ross-rotordynamics` wheel as a top-level `interface` package:

```toml
[tool.setuptools.packages.find]
exclude = ["ross.tests*", "interface*"]   # <- add "interface*"
```

`tests/test_packaging.py` proves both halves of this: that the folder is picked up
without the line, and that the line removes it.
