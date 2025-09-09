# Rotordynamic Open Source Software (ROSS)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ross-rotordynamics/ross/main)
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

You can try it out now by running the tutorial on [binder](https://mybinder.org/v2/gh/ross-rotordynamics/ross/main)

# Documentation 
Access the documentation [here](https://ross.readthedocs.io/en/latest/index.html).
The documentation provides the [installation procedure](https://ross.readthedocs.io/en/latest/getting_started/installation.html), 
[a tutorial](https://ross.readthedocs.io/en/latest/tutorials/tutorial_part_1.html), 
[examples](https://ross.readthedocs.io/en/latest/discussions/discussions.html) and the 
[API reference](https://ross.readthedocs.io/en/latest/references/api.html).

# Questions
If you have any questions, you can use the [Discussions](https://github.com/petrobras/ross/discussions) area in the repository.

# 💡 ROSS GPT – Your Virtual Assistant for Rotordynamics

Meet [ROSS GPT](https://bit.ly/rossgpt), a virtual assistant specialized in the ROSS (Rotordynamic Open-Source Software) package. With ROSS GPT, you can:

- Create and modify rotor models using ROSS in Python.
- Request practical examples for modal analysis, Campbell diagrams, unbalance response, and more.
- Get detailed technical explanations on elements such as shafts, disks, bearings, and couplings.

Access [ROSS GPT here](https://bit.ly/rossgpt) and enhance your modeling workflow with expert automated support!


# Contributing to ROSS
ROSS is a community-driven project. If you want to contribute to the project, please
check [CONTRIBUTING.md](https://github.com/petrobras/ross/blob/main/CONTRIBUTING.md). 

The code has been initially developed by Petrobras in cooperation with the Federal University of Rio de Janeiro (UFRJ)
with contributions from the Federal University from Uberlândia (UFU).
Currently, Petrobras has a cooperation agreement with UFRJ (LAVI) and UFU (LMEST) to develop and maintain the code.

