# Rotordynamic Open Source Software (ROSS)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ross-rotordynamics/ross/0.3?filepath=%2Fdocs%2Fexamples)
![github actions](https://github.com/ross-rotordynamics/ross/workflows/Tests/badge.svg)
![github actions](https://github.com/ross-rotordynamics/ross/workflows/Docs/badge.svg)
<a href="https://codecov.io/gh/ross-rotordynamics/ross">
<img src="https://codecov.io/gh/ross-rotordynamics/ross/branch/master/graph/badge.svg">
</a>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02120/status.svg)](https://doi.org/10.21105/joss.02120)

ROSS is a library written in Python for rotordynamic analysis. It allows the construction of rotor models and their 
numerical simulation. Shaft elements, as a default, are modeled with the Timoshenko beam theory, which considers shear 
and rotary inertia effects, and discretized by means of the Finite Element Method. Disks are assumed to be rigid bodies, 
thus their strain energy is not taken into account. And bearings/seals are included as linear stiffness/damping coefficients.

After defining the elements for the model, you can plot the rotor geometry and runs simulations such as static analysis, 
Campbell Diagram, mode shapes, frequency response, and time response.

You can try it out now by running the tutorial on [binder](https://mybinder.org/v2/gh/ross-rotordynamics/ross/0.3?filepath=%2Fdocs%2Fexamples).

# Documentation 
Access the documentation [here](https://ross-rotordynamics.github.io/ross-website/).
There you can find the [installation procedure](https://ross-rotordynamics.github.io/ross-website/installation.html), 
[a tutorial](https://ross-rotordynamics.github.io/ross-website/examples/tutorial.html), 
[examples](https://ross-rotordynamics.github.io/ross-website/examples.html) and the 
[API reference](https://ross-rotordynamics.github.io/ross-website/api.html).

# Questions
If you have any questions you can open a new issue with the tag <question>.

# Contributing to ROSS
Check [CONTRIBUTING.rst](https://github.com/ross-rotordynamics/ross/blob/master/CONTRIBUTING.rst).
