# Installation

(introduction)=

## Install Python

The first step is to install Python. Since ROSS requires several packages to be installed besides Python, such as
numpy and scipy, we recommend installing [Anaconda](https://docs.anaconda.com/free/anaconda/index.html) (version 3.7 or higher) which is a
scientific Python distribution that aims to simplify package management and deployment. It contains Python and a large
number of packages that are commonly used.
Alternatively, you may refer to the [Python website](http://www.python.org/).
ROSS code is tested in Python 3.7 and higher.

## Install ROSS

Using the terminal (or the Anaconda prompt if on Windows) you can install the latest release version with:

```{code-block}
pip install ross-rotordynamics
```

Alternatively, you can install the development version from github:

```{code-block}
pip install git+https://github.com/petrobras/ross.git
```
