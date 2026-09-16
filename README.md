# ROSS — Rotordynamic Open-Source Software

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/petrobras/ross/main)
![github actions](https://github.com/petrobras/ross/workflows/Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/ross/badge/?version=latest)](https://ross.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/petrobras/ross/branch/main/graph/badge.svg)](https://codecov.io/gh/petrobras/ross)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02120/status.svg)](https://doi.org/10.21105/joss.02120)

ROSS is a Python library for rotordynamic analysis, which allows the construction of rotor models and their numerical
simulation. Shaft elements are modeled with the Timoshenko beam theory, which considers shear and rotary inertia
effects, and discretized by means of the Finite Element Method. Disks are assumed to be rigid bodies, thus their strain
energy is not taken into account. Bearings and seals are included as linear stiffness and damping coefficients.

After defining the elements for the model, you can plot the rotor geometry and run simulations such as static analysis,
modal analysis, undamped critical speed, frequency response, unbalance response, time response, and more.

## Quick start

ROSS can be tried in the browser, without installation, on [Binder](https://mybinder.org/v2/gh/petrobras/ross/main).

To install it locally:
```bash
pip install ross-rotordynamics
```

ROSS also provides a graphical interface, a local web application built on top of the library. Install it with the
`interface` extra and start it with a single command:
```bash
pip install "ross-rotordynamics[interface]"
ross-interface
```
Windows users without a Python installation can download a prebuilt bundle from the
[releases page](https://github.com/petrobras/ross/releases).

If you work with an AI coding agent (Claude Code, GitHub Copilot, Cursor, Codex), install the ROSS agent skill so the
agent knows how to build rotors and run analyses:
```bash
ross-install-skill
```

The skill activates automatically whenever you ask the agent about rotordynamics with ROSS. In Claude Code it can
also be invoked explicitly:
```
/ross create a rotor with 6 shaft elements, 2 disks and 2 bearings, then plot the Campbell diagram
```

Version 3 renamed the bearing and seal parameters. Rotor files, scripts and notebooks written for ROSS 2 can be
converted with:
```bash
ross_2to3 my_rotor.toml analysis.py     # preview
ross_2to3 -w my_rotor.toml analysis.py  # rewrite in place (keeps .bak copies)
```

## Documentation

The full documentation is available at [ross.readthedocs.io](https://ross.readthedocs.io).

Key sections:
- [Installation guide](https://ross.readthedocs.io/en/latest/getting_started/installation.html)
- [User guide](https://ross.readthedocs.io/en/latest/user_guide/user_guide.html)
- [API reference](https://ross.readthedocs.io/en/latest/references/api.html)
- [Release notes](https://ross.readthedocs.io/en/latest/release_notes/release_notes.html)

## AI assistance

ROSS supports AI-assisted workflows in two ways:

- **In your coding agent.** The ROSS agent skill, which follows the [Agent Skills](https://agentskills.io) open
  standard, teaches Claude Code, GitHub Copilot, Cursor and Codex how to build rotor models and run analyses. Install
  it with `ross-install-skill` (see Quick start above); in Claude Code, invoke it explicitly with `/ross`.
- **In your browser.** [ROSS GPT](https://chatgpt.com/g/g-6838c48fbfa081918b61d77b997fdc33-ross-gpt) is the official
  chat assistant for the ROSS package. Use it to create and modify rotor models, request practical examples for modal
  analysis, Campbell diagrams, unbalance response and more, and get detailed technical explanations on elements such
  as shafts, disks, bearings and couplings.

## Support and questions

For questions, guidance or discussion of ideas, please use the
[Discussions](https://github.com/petrobras/ross/discussions) tab.

To report a bug, unexpected behavior or request a new feature, please open an
[issue](https://github.com/petrobras/ross/issues) describing the problem and how to reproduce it.

## Contributing

ROSS is a community-driven project. Contributions are welcome; please read
[CONTRIBUTING.md](https://github.com/petrobras/ross?tab=contributing-ov-file) before opening a pull request.

The code was initially developed by Petrobras in cooperation with the Federal University of Rio de Janeiro (UFRJ),
with contributions from the Federal University of Uberlândia (UFU). Currently, Petrobras has a cooperation agreement
with UFRJ (LAVI) and UFU (LMEST) to develop and maintain the code.
