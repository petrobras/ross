# -*- coding: utf-8 -*-
"""The frontend JS, as one text, for the guards that read text.

Until Phase 3 slice 3 it was one file (`app.js`). Now there are twelve modules,
and the question these guards ask -- "does this still exist in the frontend?" --
holds for the set, not for one file. The order is fixed so that a guard's
`index()` does not change result with the file system.

The comment filter stays here and stays mandatory: four guards in this suite
have already tripped over a comment of mine."""

import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")


# Third-party libraries, served as they are: they are not our code and they do
# not go through the guards.
THIRD_PARTY = ("vendor", "lib", "node_modules")


def modules():
    """All the frontend JS that is ours, in a stable order.

    It used to sweep only `main.js` and the three named folders. `frontend/schema.js`
    -- a forgotten copy from before the move to modules, which nobody imported and
    whose own `import`s did not even resolve from there -- stayed invisible to every
    guard by living somewhere the list did not mention. Now the sweep walks the
    whole tree: a new file joins on its own, and an orphan has nowhere to hide."""
    paths = []
    for root_path, folders, names in os.walk(FRONTEND):
        folders[:] = sorted(p for p in folders if p not in THIRD_PARTY)
        for name in sorted(names):
            if name.endswith(".js"):
                paths.append(os.path.join(root_path, name))
    # `main.js` first: it is the entry point, and the order has to be stable so
    # that a guard's `index()` does not change result with the file system.
    entry = os.path.join(FRONTEND, "main.js")
    rest = sorted(c for c in paths if c != entry)
    return ([entry] if entry in paths else []) + rest


def relative(path):
    return os.path.relpath(path, FRONTEND).replace(os.sep, "/")


def code_lines():
    """(module, number, line) without the whole-line comments."""
    for path in modules():
        with io.open(path, encoding="utf-8", newline="") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip().startswith("//"):
                    continue
                yield relative(path), number, line


def source():
    """All the frontend JS, without the whole-line comments."""
    return "".join(line for _, _, line in code_lines())


def raw():
    """All the frontend JS, comments included."""
    parts = []
    for path in modules():
        with io.open(path, encoding="utf-8", newline="") as handle:
            parts.append(handle.read())
    return "\n".join(parts)
