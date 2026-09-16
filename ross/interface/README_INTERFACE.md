# ROSS Graphical Interface

This is a web-based graphical interface for the **ROSS** library, designed to simplify the modeling and analysis of rotating machinery. The project integrates an interactive JavaScript frontend with a robust Python (Flask) backend, enabling rotor dynamics simulations through a visual and intuitive workflow. It is the `ross.interface` subpackage: `pip install "ross-rotordynamics[interface]"` installs it and `ross-interface` starts it.

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
- **Light and Dark Themes:** The button in every top bar switches the theme; with no choice made, the interface follows the system. The figures follow it too.

## 🛠️ Technologies

- **Frontend:** HTML5, CSS3, JavaScript (using Plotly.js for charting and Sortable.js for list management).
- **Backend:** Python 3, Flask (Web Server), ROSS-rotordynamics (Calculation Engine).

## 📂 Project Structure

The project was refactored from four monolithic files into layers. Nothing in
the interface duplicates what ROSS already knows: the element forms are derived
from the installed library by introspection, so a renamed parameter disappears
from the form instead of becoming a `TypeError` at build time.

```
app.py         Entry point behind `ross-interface`: starts the server, opens the
               browser, answers --selftest and --version
__main__.py    `python -m ross.interface`, the same entry point
selftest.py    What the executable checks about itself before anyone trusts it
version.py     The ROSS version, printed by --version, the selftest header and the About dialog
check.py       Runs ruff, pytest and the node batteries; writes check_report.txt
ross-interface.spec  How PyInstaller turns all of this into a program
api/           Transport only: routes, request envelope, session token, error handling
services/      What the application does: one runner per analysis, the expression evaluator
domain/        What the application knows: units, nodes, ROSS classes, schema,
               rotor assembly, cache, Python export
frontend/
  index.html   UI structure for the hub, modeling and analysis screens
  style.css    Visual styling and responsive rules, written in the design tokens
  design/      The ROSS design system: tokens and fonts, copied from docs/_static
  main.js      Bootstrap
  core/        Store, API client, shared state, schema loader, i18n, persistence, theme
  components/  Form builder, element list, modals, contextual help
  features/    One module per screen: hub, modeling, analysis, multirotor, export
tests/         Python suites, plus tests/js/ behaviour batteries run with node
tools/         Measuring instruments: they assert nothing and change nothing
ci/            The CI workflow, installed at .github/workflows/ in the repository root
```

## 📋 Prerequisites

Python 3.12 or newer. A virtual environment is recommended. `node` is optional
but recommended: without it the behaviour batteries in `tests/js/` are skipped,
and they are the only tests that exercise the screen rather than the source
text.

## 🔧 Installation

As a user, from PyPI:

```bash
pip install "ross-rotordynamics[interface]"
```

To work on it, from the root of the ROSS repository:

```bash
pip install -e ".[dev,interface]"
```

The `interface` extra of ROSS's `pyproject.toml` adds what only the interface
needs (Flask, PyInstaller); `dev` adds ruff and pytest. The interface is the
`ross.interface` subpackage, so it runs, tests and ships with the ROSS it lives
in, and CI runs this folder's suite whenever `ross/` changes. The
analysis/degree-of-freedom compatibility table in `domain/compatibility.py`
records what was measured on which ROSS revision; `tools/conversion_probe.py`
measures it again when the library changes.

## 💻 How to Execute

```bash
ross-interface
```

`python -m ross.interface` is the same thing. The browser opens on
`http://127.0.0.1:5001/`. Everything runs locally, on the loopback interface,
for a single user and a single session. The server writes `ross_interface.log`
in the directory it was started from (next to the executable, when frozen).

## 📦 Downloading the executable (Windows)

Every [GitHub release of ROSS](https://github.com/petrobras/ross/releases)
carries `ross-interface-<tag>-windows-x64.zip`, built by CI from the tagged
commit for 64-bit Windows. No Python is needed: the bundle carries its own.
Unpack the zip, keep the `ross-interface` folder together and run
`ross-interface.exe` from inside it.

The bundle is **not code-signed** yet, so SmartScreen shows "Windows protected
your PC" the first time; choose "More info", then "Run anyway". Signing may
follow in a later release.

Windows is the only system with a prebuilt bundle. Linux users have a Python
and run the interface from source (see above) or use ROSS as a library; macOS
is not a target, because an unsigned, un-notarized bundle cannot be opened
there without a detour through System Settings.

## 📦 Building the executable

Between releases, or on a system the release does not cover, build it yourself.
PyInstaller does not cross-compile, so a Windows program can only be produced
on Windows, a macOS one on macOS and a Linux one on Linux.

```bash
cd ross/interface
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

These are the same commands CI runs on Windows (`ci/interface.yml`), and a
test keeps the two texts in step. The instructions are not documentation that
might be right: they are documentation that is executed on every change. When
a release is published, the same job zips the bundle it just self-tested and
attaches it to the release; the asset above is that build, not a separate one.

## 🎨 Design system

The interface looks like the ROSS documentation because it is styled with the
same design system. `frontend/design/ross-tokens.css` is a verbatim copy of
`docs/_static/ross-tokens.css` (fonts included, under `design/fonts/`), and
`tests/test_design_system.py` keeps the copy equal to the original -- the
interface cannot load the docs folder, neither from a checkout nor from the
packaged executable, so it carries its own.

`style.css` is written only in those tokens: there is no hex colour in it, in
`index.html`, or in the HTML the JavaScript builds, and the same test refuses
one. That is what makes the dark theme a single attribute -- the tokens re-point
their semantic aliases under `html[data-theme="dark"]`, exactly as the docs do
-- instead of a second stylesheet. `core/theme.js` sets the attribute, remembers
the choice under `ross-theme`, follows the system while no choice is made, and
repaints the Plotly figures from the tokens on the page, so a chart and the
card it sits on come from the same values.

To change a colour, change the token in `docs/_static/ross-tokens.css` and copy
the file here. The guide is in the documentation, under *Design system*.

## ✅ Tests

```bash
cd ross/interface && python check.py
```

`check.py` is the one verification recipe: ruff, pytest and the node batteries,
with the result written to `check_report.txt` -- including what was **skipped**,
because a silent skip is coverage nobody knows they lost. Plain `pytest
ross/interface` from the repository root works too, and so does `pytest ross`,
which runs this suite together with ROSS's own.

The `tests/js/` batteries run under `node` and are executed by the Python suite,
so a single `pytest` covers both. Suites that need ROSS itself are skipped when
it is not installed.

## 🔗 Relationship with the ROSS package

This folder is the `ross.interface` subpackage, and the wheel ships it whole:
code, `frontend/` as package data, and this folder's tests, so `pytest --pyargs
ross.interface` checks an installation the way `pytest --pyargs ross` does.

- `pytest ross` and `ruff check ross` — the commands ROSS CI runs — reach it.
  ROSS's CI installs the `interface` extra for that.
- An installation **without** the extra still passes `pytest --pyargs ross`:
  `conftest.py` here drops the folder from collection when Flask cannot be
  imported, so ROSS's `--doctest-modules` never trips over a missing import.
- Flask and PyInstaller are the `interface` extra of ROSS's `pyproject.toml`,
  not its requirements: installing `ross-rotordynamics` to write a script does
  not pull in a web server.
- `ross-interface` is a console script of ROSS (`ross.interface.app:main`).

`tests/test_packaging.py` proves the shape: discovery finds `ross.interface`
and nothing outside `ross/`, the collection hook hides the folder exactly when
the extra is missing, and the workflow installs the checkout.
