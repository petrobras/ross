## Feedback and Contribution

We welcome any contribution via [ROSS issue tracker](https://github.com/petrobras/ross/issues).
These include bug reports, problems on the documentation, feedback, enhancement proposals etc.
You can use the repository [Discussions](https://github.com/petrobras/ross/discussions)
section for questions and further information.

## Code style: Ruff

To format our code we use [Ruff](https://docs.astral.sh/ruff/), which is described as *"An extremely fast Python linter and code formatter"*. You can configure your development environment to use Ruff before a commit. More information on how to set this is given at [Ruff's documentation](https://docs.astral.sh/ruff/integrations/).

We also recommend using the [pre-commit](https://docs.astral.sh/ruff/integrations/#pre-commit) tool so that Ruff is automatically run when doing a commit.

(git-configuration)=

## How to contribute to ROSS using git

Git is a version control system (VCS) for tracking changes in code during software development.
To download the ROSS source code and contribute to its development,
you need Git installed in your machine. Refer to the [Git website](https://git-scm.com/) and follow the instructions to download and install it.
Once you have Git installed, you will be able to follow the instructions in [How to contribute to ROSS using git](#how-to-contribute-to-ross-using-git), which explains how to download and contribute to ROSS.

To use git to contribute to ROSS project, follow the steps below:
*For Windows users: commands provided here can be executed using Git Bash instead of Git GUI.*

### Step 1: Make your copy (fork) of ROSS

Go to <https://github.com/petrobras/ross>
In the top-right corner of the page, click Fork, to fork it to your GitHub account.

From the command line:

```
git clone https://github.com/your-user-name/ross.git
cd ross
git remote add upstream https://github.com/petrobras/ross.git
```

### Step 2: Keep in sync with changes in ROSS

Setup your local repository, so it pulls from upstream by default:

```
git config branch.main.remote upstream
git config branch.main.merge refs/heads/main
```

This can also be done by editing the config file inside your ross/.git directory.
It should look like this:

```
[core]
        repositoryformatversion = 0
        filemode = true
        bare = false
        logallrefupdates = true
[remote "origin"]
        url = https://github.com/your-user-name/ross.git
        fetch = +refs/heads/*:refs/remotes/origin/*
[remote "upstream"]
        url = https://github.com/petrobras/ross.git
        fetch = +refs/heads/*:refs/remotes/upstream/*
        fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*
[branch "main"]
        remote = origin
        merge = refs/heads/main
```

The part {code}`fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*` will make pull requests available in your local repository after a git fetch.

For example, assuming `$ID` is the pull request number and `$BRANCHNAME` is the name of the new local branch you wish to create:

```
git fetch upstream pull/$ID/head:$BRANCHNAME
```

Switch to the newly created branch:

```
git switch $BRANCHNAME
```

(setup-environment)=

### Step 3: Set up development environment

To set up a development environment you can [create a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
    
```
conda create -n rs
conda activate rs
```

or a virtualenv:

```
python3 -m venv env
. env/bin/activate
# or "env\Scripts\activate" on Windows
```

and then install ROSS in editable mode with development dependencies:

```
pip install -e ".[dev]"
```

### Step 4: Make a new feature branch

```
git fetch upstream
git checkout -b my-new-feature upstream/main
```

### Step 5: Testing the code

We use pytest to test the code. Unit tests are placed in the `~/ross/ross/tests` folder. We also test our docstrings to
assure that the examples are working.
If you want to run all the tests you can do it with (from the `~/ross/ross` folder):

```
pytest
```

Code is only merged to main if tests pass. This is checked by GitHub Actions, so make sure tests are passing before pushing your code to GitHub.

### Step 6: Push changes to your git repository

After a complete working set of related changes are made:

```
git add modified_file
git commit
git push origin my-new-feature
```

The following blog posts have some good information on how to write commit messages:

[A Note About Git Commit Messages](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)

[On commit messages](https://who-t.blogspot.com/2009/12/on-commit-messages.html)

### Step 7: Push changes to the main repo

To create a Pull Request (PR), refer to [the github PR guide](https://help.github.com/articles/about-pull-requests/).

## Docstrings for class and methods

A new method must have a docstring presenting a summary for what the method does.
ROSS' docstrings follow the NumPy [docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
It's important to follow the NumPy's template due to the formatting that will be presented on the ROSS website.

Example of docstring:

```
def foo(arg1, arg2, arg3):
"""Title (First line should be in imperative mood and end with a period)

A brief explanation of what this method does. (Optional)

Parameters (if the method receives any arguments)
----------
arg1 : TYPE
    DESCRIPTION.
arg2 : TYPE
    DESCRIPTION.
arg3 : TYPE
    DESCRIPTION.

References (if applicable)
----------
.. bibliography:: ../../../docs/refs.bib

Raises (if there's any error message raised)
-----
SomeError
    DESCRIPTION

Returns (if the method return something)
-------
result : TYPE
    DESCRIPTION.

Examples (if applicable)
--------
>>> a = 1
>>> b = 2
>>> c = 3
>>> s = foo(a, b, c)
6
"""
result = arg1 + arg2 + arg3
return result
```

It is possible to add other sections in addition to those previously presented (e.g. `Notes`, `See Also`, `Warnings`...).
Just follow the same rules and it's good to go.

When creating examples, be aware of code lines that return any result from a method or class.
The example output must match what the method returns because GitHub Actions (the CI that runs tests for ROSS) checks the examples and raise errors,
if the example output does not match the actual output.

Sometimes, it's not possible to represent all the output (e.g. a figure, a large matrix, etc),
so it's recommended to use the comment `# doctest: +ELLIPSIS`, and then, truncate the function output with a `...`, and add this comment beside the command line.

Example:

```
from bokeh.plotting import figure

def foo():
    """Plot a bokeh figure.

    Returns
    -------
    figure : bokeh.figure
        A figure.

    Examples
    --------
    >>> figure = foo()
    >>> figure # doctest: +ELLIPSIS
    Figure...
    """
    fig = figure()
    fig.line([1, 2, 3], [1, 2, 3])

    return fig
```

## Documentation

We use [sphinx](http://www.sphinx-doc.org/en/master/) to generate the project's documentation. We keep the source
files at ~/ross/docs, and the website is hosted
[here](https://ross.readthedocs.io/en/latest/).
The website tracks the documentation for the released version with the 'Docs'
GitHub Action.

If you want to test the documentation locally:

- Install [pandoc](https://pandoc.org/installing.html), which is needed to convert the notebook files;
- Install ROSS development version so that you have all packages required to build the documentation (see {ref}`setup-environment`).

Go to the ~/ross/docs folder and run:

```
make html
```

Optionally, if you don't want run all notebooks you can use:

```
make EXECUTE_NOTEBOOKS='off' html
```

After building the docs, go to the \_build/html directory (~/ross/docs/\_build/html)
and start a python http server:

```
python -m http.server
```

After that you can access your local server (<http://0.0.0.0:8000/>) and see the generated docs.

## Making new releases

To make a new release, first we need to change the version in the __init__.py file.

Release notes can be created using the ross-bott script.

The next step is to create a tag using git and push to GitHub:

```
git tag <version number>
git push upstream --tags
```

Pushing the new tag to the GitHub repository will start a new build on GitHub actions. If all the tests succeed, GitHub will
upload the new package to PyPI (see the deploy command on .github/workflows/publish-to-pypi.yml).

It is recommended to first use release candidate versions (e.g. v1.1.2rc1). These will only be installed with:

```
pip install --pre ross-rotordynamics
```

and it is useful to test the installation process before the final release.

## ROSS structure

To explain how ROSS is structured, we will describe the following building blocks:

- Elements: represent the physical components of the rotor (e.g. shaft, disks, bearings, etc);
- Rotor: represent the rotor itself, which is composed by elements;
- Results: represent the results of the simulation (e.g. displacement, velocity, etc).

### Elements

Elements are the building blocks of the rotors. They are the physical components of the rotor (e.g. shaft, disks, bearings, etc).
Each element has its own class, which is responsible for calculating the element's stiffness and damping matrices, and its gyroscopic effect.

All the elements classes inherit from the `Element` class, which is defined in the `ross/element.py` file. 

The `Element` class is an abstract base class, which means that classes that inherit from it must implement the methods defined in the `Element` class.
Some of these abstract methods are:
- `M(self)`: returns the mass matrix of the element;
- `K(self)`: returns the stiffness matrix of the element;
- `C(self)`: returns the damping matrix of the element;
- `G(self)`: returns the gyroscopic matrix of the element.
- `dof_mapping(self)`: returns the degree of freedom mapping of the element.
- _patch(self): returns a `plotly.graph_objects.Figure` that will be used in the rotor plot.

If a new element is created, it must inherit from the `Element` class and implement the methods described above.
With that, the element will be compatible with the rest of the code.

### Rotor

The `Rotor` class is defined in the `ross/rotor_assembly.py` file. It is responsible for assembling the rotor, which means that it will
assemble the stiffness, damping and gyroscopic matrices of the rotor.

After having a `Rotor` object, it is possible to run different analysis which are available as methods with the prefix `.run_`.

### Results

The `Results` class is defined in the `ross/results.py` file. It is responsible for storing the results of each analysis executed with a `.run_` method.
A `Results` object will also have some methods to plot the results that are stored in the object.

## Coding conventions

### Save and load data

Each element or rotor has a `save` method that saves the object in a `.toml` file. 

The `load` method is a class method that loads the object from a `.toml` file.

Here is a code example for saving and loading a rotor:

```python
import ross as rs

# create a rotor
rotor = rs.rotor_example()

# save the rotor
rotor.save("rotor.toml")

# load the rotor using the class method
rotor_loaded = rs.Rotor.load("rotor.toml")
```

Additionally, if the calculation of a specific object is expensive, we implement a `.run` method that will check if the object was already calculated and saved in the `.toml` file.

This way we avoid recalculating when loading the object.
