# -*- coding: utf-8 -*-
"""What belongs to the repository, and what merely lives inside its folder.

Guards that walk the tree from the root find whatever else lives in the
folder: a virtual environment **inside** the project (`site-packages`, with a
folder called `ross`, hundreds of `test_*.py` of other libraries, CRLF, a
fixture in `big5`), or the `build/` and `dist/` PyInstaller writes there.

The question is not about any one of those. It is: *what, inside this folder,
is not the repository?* And that question
already has an answer written down, maintained, and in the one place everybody
updates when they add build output -- the `.gitignore`.

So the names are not listed here. They are read from there, plus a small floor
of things that are never ours whatever any file says, plus the marker that
identifies a virtual environment by what Python itself writes into one:
`pyvenv.cfg`. The name of a folder is a convention; the marker is a fact, and
the `.gitignore` is a declaration.
"""

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# True whatever any file declares: none of these is ever part of a repository.
NEVER_OURS = ("__pycache__", "node_modules", "site-packages", ".git")


def _declared_in_the_gitignore():
    """The plain names the repository already declares as not its own.

    Only the simple entries -- a bare name, with or without a trailing slash.
    Patterns with wildcards or paths are left to git, which is the only thing
    that has to understand them fully.
    """
    names = set()
    path = os.path.join(ROOT, ".gitignore")
    if not os.path.exists(path):
        return names
    with io.open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#")[0].strip()
            if not line or line.startswith("!") or "*" in line or "?" in line:
                continue
            line = line.rstrip("/")
            if "/" not in line:
                names.add(line)
    return names


NOT_THE_REPOSITORY = frozenset(NEVER_OURS) | _declared_in_the_gitignore()


def is_a_virtual_environment(folder):
    return os.path.exists(os.path.join(folder, "pyvenv.cfg"))


def ours(folder, names):
    """The subfolders of `folder` that belong to the repository.

    Meant for `folders[:] = ours(root, folders)` inside an `os.walk`: assigning
    to the slice prunes the walk, where filtering the output only hides it.
    """
    return [
        name
        for name in names
        if name not in NOT_THE_REPOSITORY
        and not is_a_virtual_environment(os.path.join(folder, name))
    ]
