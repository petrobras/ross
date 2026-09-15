# -*- coding: utf-8 -*-
"""Questions about the shape of the code, answered by the tree and not by the text.

Twice a guard written as a text search tripped over a comment of mine:
`"iframe" not in app_js` accused the sentence that **explained** that the iframe
had gone, and `"except Exception" not in source` accused the comment explaining
why the old net had been removed. In both cases the guard was right about the
code and wrong about the file.

With `ast`, comments and docstrings do not exist -- only what executes. And the
question gets more precise as a bonus: a bare `except:` is a generic catch too,
and no text search would find it under that name."""

import ast
import glob
import io
import os


def modules(*path_parts, recursive=False):
    """Gives back (path, tree) for each .py under the given path."""
    pattern = os.path.join(*path_parts)
    for path in sorted(glob.glob(pattern, recursive=recursive)):
        with io.open(path, encoding="utf-8") as handle:
            yield path, ast.parse(handle.read(), filename=path)


def catches_everything(tree):
    """True if some `except` catches Exception, BaseException or nothing."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:  # `except:` nu
            return True
        kinds = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
        for kind in kinds:
            name = getattr(kind, "id", getattr(kind, "attr", ""))
            if name in ("Exception", "BaseException"):
                return True
    return False


def assigns_to(tree, target):
    """True if the module assigns to `target` (e.g. 'sys.stdout')."""
    an_object, _, attribute = target.partition(".")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        destinations = node.targets if isinstance(node, ast.Assign) else [node.target]
        for destination in destinations:
            if (
                isinstance(destination, ast.Attribute)
                and destination.attr == attribute
                and getattr(destination.value, "id", None) == an_object
            ):
                return True
    return False


def calls_method(tree, name):
    """True if the module calls `<something>.name(...)` anywhere."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        ):
            return True
    return False
