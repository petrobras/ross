# Installation

(introduction)=

## Install Python

The first step is to install Python. Since ROSS requires several packages to be installed besides Python, such as
numpy and scipy, we recommend installing [miniforge](https://conda-forge.org/download/) or [Anaconda](https://docs.anaconda.com/free/anaconda/index.html) (version 3.9 or higher) which is a
scientific Python distribution that aims to simplify package management and deployment. It contains Python and a large
number of packages that are commonly used.
Alternatively, you may refer to the [Python website](http://www.python.org/).
ROSS code is tested in Python 3.9 and higher.

## Install ROSS

Using the terminal (or the Anaconda prompt if on Windows) you can install the latest release version with:

```{code-block}
pip install ross-rotordynamics
```

Alternatively, you can install the development version from GitHub:

```{code-block}
pip install git+https://github.com/petrobras/ross.git
```

# 💡 ROSS GPT – Virtual Assistant

Need help building your rotor model or running an analysis?

Meet [**ROSS GPT**](https://bit.ly/rossgpt), a virtual assistant trained specifically for the ROSS package. You can:

- Generate rotor models in Python with just a description.
- Run and interpret modal analysis, Campbell diagrams, and more.
- Understand technical aspects of ROSS elements like ShaftElement, DiskElement, BearingElement, etc.

👉 [Click here to start using ROSS GPT](https://bit.ly/rossgpt).

## ROSS graphical interface (Windows)

A standalone Windows bundle (PyInstaller) is built on GitHub Actions with the source under `apps/`. Each successful run on `main` publishes `ROSS-Interface-Windows.zip` inside the documentation site’s static downloads folder so users can install the Flask backend and browser UI without a separate Python setup.

[Download ROSS-Interface-Windows.zip](../_static/downloads/ROSS-Interface-Windows.zip)

```{note}
This ZIP is inserted during the **Windows interface bundle and documentation** workflow run and is only present in HTML builds produced by that pipeline (for example GitHub Pages). If you are viewing docs built elsewhere (local Sphinx or Read the Docs), download the ZIP from the latest workflow artifacts on [GitHub Actions](https://github.com/petrobras/ross/actions/workflows/windows-interface-and-docs.yml).
```
