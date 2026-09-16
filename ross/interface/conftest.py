# -*- coding: utf-8 -*-
"""Collect the interface's tests only where they can run.

The interface is a subpackage of ROSS, so `pytest ross` reaches it, with
`--doctest-modules` **importing** every module it finds. Its dependencies are
the `interface` extra of ROSS's `pyproject.toml` -- Flask above all -- and an
installation without that extra must still be able to run `pytest --pyargs
ross`. The hook below settles that: when the extra is missing, nothing here is
collected, and ROSS's own suite is untouched.

    pip install -e ".[dev]"             -> pytest ross skips this folder
    pip install -e ".[dev,interface]"   -> pytest ross runs it too
    cd ross/interface && pytest          -> the same, with this folder's pytest.ini
"""

import importlib.util
import logging
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# The suite must not write into the application's log.
#
# `ross_interface.log` is where somebody looks when the program misbehaves on
# their machine, and the suite runs the application in this same process: it
# builds applications, refuses bodies on purpose, kills fake workers on purpose.
# All of that used to land in that file, and it reads exactly like a program
# falling apart -- a `KeyError` traceback from a test that asserts a 500, a
# "the worker was lost" from a test that kills one deliberately, a "worker ready
# in 0.5 s" from a worker that is a dictionary with no process behind it. The
# first time the log was used as evidence, it could not answer the question.
#
# `configure_logging()` in api/errors.py returns early when the logger already
# has a handler. Claiming it here, before any test module is imported, is
# therefore enough: the application's own configuration never runs during the
# suite, no file is opened, and nothing of ours is written where the user looks.
# `tests/test_worker.py` guards both halves of that -- no handler writing to the
# real file, and the early return this depends on still being there.
logging.getLogger("ross_interface").addHandler(logging.NullHandler())


def the_interface_can_be_imported():
    """Is the `interface` extra installed for this interpreter?

    Asked of the import system, not of pip: what matters is whether `import
    flask` will succeed when collection imports our modules. Kept apart from
    the hook so it can be tested without starting all of pytest.
    """
    return importlib.util.find_spec("flask") is not None


def pytest_ignore_collect(collection_path, config):
    return not the_interface_can_be_imported()
