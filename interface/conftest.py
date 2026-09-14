# -*- coding: utf-8 -*-
"""Keep this folder out of ROSS's own `pytest`.

The interface lives in a folder of its own inside the ROSS repository, and the
acceptance condition is that it **must not get in the way** of whoever works on
ROSS. Their CI runs `pytest ross`, which already does not reach us -- but a
developer typing `pytest` at the repository root would reach us: pytest's
rootdir becomes the repository, their `pytest.ini` applies (with
`--doctest-modules`, which **imports** every module it finds), and collection
walks the whole tree. With no Flask installed -- and it is not in ROSS's
`requirements.txt`, nor should it be -- the import fails and their suite breaks
because of the interface.

The hook below settles that from our side, without editing a single file of
theirs: when pytest's rootdir is not this folder, whoever called was not calling
us, and there is nothing here to collect.

    pytest                      at the repo root -> ignores the interface entirely
    pytest interface            from the repo root -> rootdir is this folder, collects
    cd interface && pytest      -> the same

Requires `pytest >= 7.0` (the hook taking `collection_path`; `config.rootpath`
arrived in 6.1). ROSS's CI installs the current pytest.
"""

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


def this_folder_is_the_rootdir(rootdir):
    """Is pytest's rootdir for this run the interface folder?

    Kept apart from the hook so it can be tested without starting all of pytest.
    """
    return os.path.abspath(str(rootdir)) == HERE


def pytest_ignore_collect(collection_path, config):
    return not this_folder_is_the_rootdir(config.rootpath)
