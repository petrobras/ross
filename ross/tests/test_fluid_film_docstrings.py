"""Docstring/signature consistency for the fluid-film solver and bearings.

Every function in ``ross/bearings/fluid_film`` (and the bearing classes
built on it) whose docstring has a NumPy-style ``Parameters`` section must
document exactly the parameters of its signature -- both directions of
drift (stale documented names, missing entries) fail here. Functions
without a ``Parameters`` section are skipped: this enforces consistency,
not coverage.
"""

import ast
from pathlib import Path

FLUID_FILM_DIR = Path(__file__).parent.parent / "bearings" / "fluid_film"
EXTRA_FILES = [
    Path(__file__).parent.parent / "bearings" / "fluid_film_bearing.py",
]

_SECTIONS = {
    "Parameters",
    "Other Parameters",
    "Attributes",
    "Returns",
    "Yields",
    "Receives",
    "Raises",
    "Warns",
    "Warnings",
    "See Also",
    "Notes",
    "References",
    "Examples",
}


def _documented_params(docstring):
    """Extract the names documented in Parameters sections, or None."""
    lines = docstring.split("\n")
    names = set()
    found_section = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        underlined = (
            i + 1 < len(lines)
            and set(lines[i + 1].strip()) == {"-"}
            and len(lines[i + 1].strip()) >= len(line) - 1 > 0
        )
        if not (underlined and line in ("Parameters", "Other Parameters")):
            i += 1
            continue

        found_section = True
        i += 2
        pending = ""
        expect_type = False
        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()
            next_underlined = (
                i + 1 < len(lines)
                and set(lines[i + 1].strip()) == {"-"}
                and stripped in _SECTIONS
            )
            if next_underlined:
                break
            if stripped and not raw.startswith(" "):
                if expect_type:
                    expect_type = False
                elif " : " in stripped or stripped.endswith(":"):
                    name_part = stripped.split(" : ")[0].rstrip(":")
                    for name in (pending + name_part).split(","):
                        name = name.strip().lstrip("*")
                        if name.isidentifier():
                            names.add(name)
                    pending = ""
                    expect_type = stripped.endswith(":")
                elif stripped.endswith(","):
                    pending += stripped
                elif stripped.isidentifier() or (
                    "," in stripped
                    and all(
                        p.strip().lstrip("*").isidentifier()
                        for p in stripped.split(",")
                        if p.strip()
                    )
                ):
                    for name in (pending + stripped).split(","):
                        name = name.strip().lstrip("*")
                        if name.isidentifier():
                            names.add(name)
                    pending = ""
            i += 1
    if not found_section:
        return None
    return names


def _signature_params(node):
    """Return all parameter names of a function definition node."""
    a = node.args
    names = [arg.arg for arg in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def _check_file(path):
    """Return (lineno, qualname, stale, missing) mismatches for one file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    stack = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                documented = _documented_params(docstring)
                if documented is not None:
                    actual = _signature_params(node)
                    if stack and isinstance(stack[-1], ast.ClassDef):
                        if actual and actual[0] in ("self", "cls"):
                            actual = actual[1:]
                    stale = documented - set(actual)
                    missing = [n for n in actual if n not in documented]
                    if stale or missing:
                        qual = ".".join([s.name for s in stack] + [node.name])
                        found.append((node.lineno, qual, sorted(stale), missing))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.append(node)
            for child in ast.iter_child_nodes(node):
                visit(child)
            stack.pop()
        else:
            for child in ast.iter_child_nodes(node):
                visit(child)

    visit(tree)
    return found


def test_fluid_film_docstring_parameters_match_signatures():
    mismatches = []
    for path in sorted(FLUID_FILM_DIR.rglob("*.py")) + EXTRA_FILES:
        for lineno, qual, stale, missing in _check_file(path):
            mismatches.append((path, lineno, qual, stale, missing))
    report = "\n".join(
        f"{f}:{line}: {qual}: stale={stale} missing={missing}"
        for f, line, qual, stale, missing in mismatches
    )
    assert not mismatches, f"docstring Parameters drifted from signatures:\n{report}"
