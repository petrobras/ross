# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import re

from setuptools import find_packages, setup


def read(path, encoding="utf-8"):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(
        r"""^__version__ = ['"]([^'"]*)['"]""", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Package meta-data.
NAME = "ross-rotordynamics"
DESCRIPTION = "ROSS: Rotordynamic Open Source Software"
EMAIL = "raphaelts@petrobras.com.br"
AUTHOR = "ROSS developers"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = version("ross/__init__.py")

# What packages are required for this module to be executed?
with open("requirements.txt") as f:
    REQUIRED = f.read().splitlines()

# What packages are optional?
EXTRAS = {
    "dev": [
        "pytest>=4.6",
        "pytest-cov",
        "coverage",
        "codecov",
        "sphinx",
        "myst-nb",
        "sphinx-book-theme",
        "sphinx-panels",
        "sphinx-copybutton",
        "sphinx-rtd-theme",
        "linkify-it-py",
        "numpydoc",
        "sphinxcontrib-bibtex>=2.2",
        "ruff",
        "sphinx-design",
    ]
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    project_urls={
        "Documentation": "https://ross.readthedocs.io/en/stable/",
        "Bug Tracker": "https://github.com/petrobras/ross/issues",
        "Discussions": "https://github.com/petrobras/ross/discussions",
        "Source Code": "https://github.com/petrobras/ross",
    },
    packages=find_packages(exclude=("tests",)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    package_data={"": ["new_units.txt"]},
    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)
