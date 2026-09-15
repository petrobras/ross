# -*- coding: utf-8 -*-
"""What belongs to the repository, and what merely lives inside its folder.

Written after the first full run on Linux, where the virtual environment sits
**inside** the project instead of beside it. Six guards failed at once, all of
them walking the tree from the root and finding `site-packages`: one found a
folder called `ross`, another hundreds of `test_*.py` belonging to other
libraries, another third-party code to parse, another CRLF, and two died with
`UnicodeDecodeError` on a fixture shipped in `big5`.

Then, one slice later, `build/` and `dist/` did it again. PyInstaller writes
both inside the project, and the same guards walked straight into them --
accusing ROSS's own installed folder, and the CRLF in scikit-learn's test data.

**The second time is the interesting one.** The first fix taught this module
about virtual environments; the question was never about virtual environments.
It is: *what, inside this folder, is not the repository?* And that question
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
