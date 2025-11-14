# ROSS — Rotordynamic Open-Source Software
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/petrobras/ross/main)
![github actions](https://github.com/petrobras/ross/workflows/Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/ross/badge/?version=latest)](https://ross.readthedocs.io/en/latest/?badge=latest)
<a href="https://codecov.io/gh/petrobras/ross">
<img src="https://codecov.io/gh/petrobras/ross/branch/main/graph/badge.svg">
</a>
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02120/status.svg)](https://doi.org/10.21105/joss.02120)

ROSS is a Python library for rotordynamic analysis, which allows the construction of rotor models and their numerical
simulation. Shaft elements are modeled with the Timoshenko beam theory, which considers shear and rotary inertia
effects, and discretized by means of the Finite Element Method. Disks are assumed to be rigid bodies, thus their strain
energy is not taken into account. Bearings and seals are included as linear stiffness and damping coefficients.

After defining the elements for the model, you can plot the rotor geometry and run simulations such as static analysis,
modal analysis, undamped critical speed, frequency response, unbalance response, time response, and more.

## 🚀 Quick Start

You can try ROSS instantly in your browser:

👉 [**Launch ROSS in Binder**](https://mybinder.org/v2/gh/petrobras/ross/main)

Or install it locally:
```bash
pip install ross-rotordynamics
```

## 📚 Documentation

Access full documentation [**here**](https://ross.readthedocs.io).

Key sections:
- [Installation guide](https://ross.readthedocs.io/en/latest/installation.html)
- [User guide](https://ross.readthedocs.io/en/latest/user_guide/user_guide.html)
- [API reference](https://ross.readthedocs.io/en/latest/api.html)
- [Release notes](https://ross.readthedocs.io/en/latest/release_notes/release_notes.html)

## 🤖 ROSS GPT

Meet [**ROSS GPT**](https://bit.ly/rossgpt), the official AI assistant for the ROSS package. With ROSS GPT, you can:
- Create and modify rotor models using ROSS in Python.
- Request practical examples for modal analysis, Campbell diagrams, unbalance response, and more.
- Get detailed technical explanations on elements such as shafts, disks, bearings, and couplings.

## ❓ Support & Questions

If you have **questions**, need **guidance**, or want to **discuss ideas**, please use the [**Discussions**](https://github.com/petrobras/ross/discussions) tab.

If you encounter a **bug**, experience unexpected behavior, or want to **request a new feature**, please open an [**Issue**](https://github.com/petrobras/ross/issues) describing the problem and how to reproduce it.

## 🤝 Contributing

ROSS is a community-driven project. If you want to contribute to the project, please
check [**CONTRIBUTING.md**](https://github.com/petrobras/ross?tab=contributing-ov-file). 

The code has been initially developed by Petrobras in cooperation with the Federal University of Rio de Janeiro (UFRJ)
with contributions from the Federal University from Uberlândia (UFU).
Currently, Petrobras has a cooperation agreement with UFRJ (LAVI) and UFU (LMEST) to develop and maintain the code.