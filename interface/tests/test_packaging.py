# -*- coding: utf-8 -*-
"""The interface goes into the ROSS repository without getting in the way.

The acceptance condition is not "our tests pass" -- it is **the ROSS tests keep
passing with the interface in the tree**, and a ROSS developer never has to
touch anything here. That is a property of the fit, not of our code, and that is
why it has a guard of its own: without one, the first person to find out that
the interface broke the ROSS suite would be the maintainer, in someone else's
PR.

What is measured here:

* the hook that hides this folder when it is ROSS running `pytest`;
* that our folder does not become a package inside the `ross-rotordynamics` wheel;
* that nothing of ours lives where `pytest ross` looks;
* that the dependencies only we use are declared only here."""

import io
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import repository
from domain.analysis_catalog import ANALYSES

# The name the folder will have inside the ROSS repository. It is written down
# because two guards depend on it, and because changing the name means changing
# the line we ask for in their `exclude`.
FOLDER_IN_ROSS = "interface"


# --- the hook that erases us from ROSS's `pytest` -----------------------------


def test_the_hook_hides_us_when_pytest_was_not_called_on_us():
    """`pytest` at the root of the ROSS repo must not even try to import the interface.

    ROSS's `pytest.ini` turns on `--doctest-modules`, which **imports** every module
    collection finds. With no Flask installed -- and it is not, nor should it be, in
    ROSS's `requirements.txt` -- the import fails and their suite breaks because of
    the interface."""
    from conftest import this_folder_is_the_rootdir

    assert this_folder_is_the_rootdir(ROOT), (
        "called on the interface: collection has to happen"
    )
    assert not this_folder_is_the_rootdir(os.path.dirname(ROOT)), (
        "pytest rootdir at the repository: the interface has to vanish from collection"
    )
    assert not this_folder_is_the_rootdir(os.path.join(os.path.dirname(ROOT), "ross"))


def test_the_hook_is_wired_to_the_pytest_entry_point():
    """Control: the function above can be right and be wired to nothing."""
    import conftest

    assert hasattr(conftest, "pytest_ignore_collect"), "the hook is gone"

    class FakeConfig(object):
        def __init__(self, root):
            self.rootpath = root

    assert conftest.pytest_ignore_collect(ROOT, FakeConfig(ROOT)) is False
    assert (
        conftest.pytest_ignore_collect(ROOT, FakeConfig(os.path.dirname(ROOT))) is True
    )


def test_our_pytest_config_does_not_turn_on_doctests():
    """We have no doctests: turning them on would only make `pytest` import for nothing."""
    with io.open(os.path.join(ROOT, "pytest.ini"), encoding="utf-8") as handle:
        # Without stripping the comments, the guard accused the comment that EXPLAINS
        # why the option is not turned on.
        lines = [line.split("#")[0] for line in handle]
    content = "".join(lines)
    assert "[pytest]" in content, (
        "without this section pytest's rootdir does not land here"
    )
    assert "--doctest-modules" not in content


# --- not entering the ROSS wheel ----------------------------------------------

ROSS_FIND_CONFIG = dict(
    where=".", include=["*", "ross.new_units.txt*"], exclude=["ross.tests*"]
)


def _discover(base, exclude):
    """The packages setuptools would find, with ROSS's configuration."""
    from setuptools import find_namespace_packages

    previous = os.getcwd()
    os.chdir(base)
    try:
        return find_namespace_packages(
            where=".", include=ROSS_FIND_CONFIG["include"], exclude=exclude
        )
    finally:
        os.chdir(previous)


def _fake_repository(destination):
    """A tree shaped like the ROSS repo with the interface inside."""
    for path in (
        "ross/tests",
        "docs",
        "%s/api" % FOLDER_IN_ROSS,
        "%s/tests" % FOLDER_IN_ROSS,
    ):
        os.makedirs(os.path.join(destination, path))
    for module in (
        "ross/__init__.py",
        "ross/rotor_assembly.py",
        "%s/api/__init__.py" % FOLDER_IN_ROSS,
    ):
        io.open(os.path.join(destination, module), "w").close()
    return destination


def test_the_folder_would_be_shipped_inside_the_ross_wheel_without_one_line(tmp_path):
    """Why we ask for one line in their `pyproject.toml`, and exactly which.

    ROSS's `[tool.setuptools.packages.find]` resolves in *namespace* mode: a folder
    at the root enters the distribution **even without `__init__.py`**. Without the
    line, anyone running `pip install ross-rotordynamics` starts receiving an
    `interface` package at the top of their own namespace.

    This test is the evidence for the request: without the line we are packaged,
    with it we are not. If setuptools ever changes behaviour, it says so -- and the
    request to upstream stops making sense."""
    base = _fake_repository(str(tmp_path))

    without_the_line = _discover(base, ROSS_FIND_CONFIG["exclude"])
    ours = sorted(p for p in without_the_line if p.split(".")[0] == FOLDER_IN_ROSS)
    assert ours, (
        "setuptools did not pick the folder up: if this became true, the "
        "line asked of upstream is no longer needed"
    )

    with_the_line = _discover(
        base, ROSS_FIND_CONFIG["exclude"] + ["%s*" % FOLDER_IN_ROSS]
    )
    still_ours = [p for p in with_the_line if p.split(".")[0] == FOLDER_IN_ROSS]
    assert still_ours == [], "the line we ask for does not solve it: %s" % still_ours

    assert "ross" in with_the_line, "control: ROSS itself has to still be there"


def test_the_project_root_is_not_an_importable_package():
    """An `__init__.py` here would make us a package even in setuptools' old mode."""
    assert not os.path.exists(os.path.join(ROOT, "__init__.py"))


# --- not living where `pytest ross` looks -------------------------------------


def test_nothing_of_ours_lives_under_a_ross_directory():
    """`pytest ross` and `ruff check ross` sweep by path, not by package."""
    intruders = []
    for root, folders, names in os.walk(ROOT):
        folders[:] = repository.ours(root, folders)
        if os.path.basename(root) == "ross":
            intruders.append(os.path.relpath(root, ROOT))
    assert intruders == [], (
        "a folder called 'ross' inside the interface: %s" % intruders
    )


def test_our_tests_are_all_in_our_own_tests_folder():
    """A test loose outside `tests/` would escape `testpaths` and the guard above."""
    loose = []
    for root, folders, names in os.walk(ROOT):
        folders[:] = repository.ours(root, folders)
        inside = os.path.join(os.path.normcase(os.path.abspath(ROOT)), "tests")
        here = os.path.normcase(os.path.abspath(root))
        if here == inside or here.startswith(inside + os.sep):
            continue
        for name in names:
            if name.startswith("test_") and name.endswith(".py"):
                loose.append(os.path.relpath(os.path.join(root, name), ROOT))
    assert loose == [], "test outside tests/: %s" % loose


# --- the dependencies are ours, and they stay here ----------------------------


def _requirements():
    with io.open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
        return [
            line.split("#")[0].strip()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        ]


def test_the_dependencies_only_we_need_are_declared_here():
    """Flask is ours. Declaring it in ROSS would impose it on whoever only wants the library."""
    declared = " ".join(_requirements()).lower()
    assert "flask" in declared
    assert "ross-rotordynamics" in declared


def test_the_ross_pin_is_exact():
    """The compatibility table was MEASURED against one version. That is the pin.

    `domain/compatibility.py` says which analyses break under a degree-of-freedom
    conversion, and every line came from running the probe against a concrete
    version of ROSS. An open range (`>=`) would let the table speak for a version
    nobody measured -- and the failure mode of that table is not an error, it is a
    chart carrying the wrong badge.

    This test was born from a rule of mine that was wrong: I had written a guard
    forbidding the redeclaration of `numpy`/`toml`, and Leonardo's
    `requirements.txt` declares them on purpose, because ROSS does not restrict
    numpy and BE-03 of the audit was exactly a NumPy 2 break. The guard was replaced
    by the one that measures what actually matters."""
    line = [r for r in _requirements() if "ross-rotordynamics" in r.lower()]
    assert len(line) == 1, "ROSS has to appear once: %s" % line
    pin = line[0]
    assert ("@" in pin and "git+" in pin) or "==" in pin, (
        "ROSS is pinned to an open range (%s): the compatibility "
        "table was measured against a single version" % pin
    )


def test_plotly_is_declared_with_a_ceiling():
    """ROSS leaves plotly open, and plotly 7.0.0 broke it at import time.

    `ross/__init__.py` imports `ross.plotly_theme`, which registers a template
    containing a `scattermapbox` series; plotly dropped that trace type in
    7.0.0, so the registration raises and `import ross` never completes. ROSS
    declares `plotly>=5.11` with no upper bound, which means a clean
    `pip install` today cannot import the library at all.

    The two machines of slice 4 differed in exactly one library -- plotly 6.7.0
    against 7.0.0, everything else identical down to the patch number -- and
    that is the whole evidence for the ceiling. Same reasoning as the numpy
    range on the line above it, and the same reasoning as the exact ROSS pin: a
    range nobody measured speaks for versions nobody ran.
    """
    line = [r for r in _requirements() if r.lower().startswith("plotly")]
    assert len(line) == 1, "plotly has to appear once: %s" % line
    assert "<" in line[0], (
        "plotly is declared with no ceiling (%s): 7.0.0 makes `import ross` raise"
        % line[0]
    )


def test_every_third_party_import_is_declared_somewhere():
    """An import nobody declares only fails on someone else's machine.

    The sweep was a regular expression over `.strip()`ed lines, and a docstring
    sentence starting with "from BearingElement -- the unit has to come along" was
    read as an import. `ast` tells prose from syntax with no heuristic: it is the
    same correction as the f-string guard, and for the same reason."""
    import ast

    # The standard library is asked of the interpreter, not listed by hand. The
    # hand-written list held twenty-seven names and was missing `queue` -- which
    # is how a list of what somebody remembered fails: in silence, and only when
    # a new import happens to land in the gap. `sys.stdlib_module_names` exists
    # from Python 3.10, which is this project's floor.
    FROM_THE_STANDARD_LIBRARY = set(sys.stdlib_module_names)

    # And what is ours is what is in our folders, for the same reason: `selftest`
    # sat at the root and was missing too.
    OURS = {
        name[: -len(".py")]
        for folder in (ROOT, os.path.join(ROOT, "tests"))
        for name in os.listdir(folder)
        if name.endswith(".py")
    } | {"api", "domain", "services"}

    # Control for the two derived sets above: a set that answered "yes" to
    # everything would make this guard pass without measuring anything, and a
    # derived set fails that way silently where a hand-written one does not.
    assert "json" in FROM_THE_STANDARD_LIBRARY, "the interpreter's list is not a list"
    assert "flask" not in FROM_THE_STANDARD_LIBRARY, (
        "flask counted as standard library: this guard would pass for anything"
    )
    assert "selftest" in OURS and "ross" not in OURS

    # They come with ROSS or with Flask, which is why they are not repeated in our file.
    VIA_THE_CHAIN = {"plotly", "werkzeug", "ross", "pytest", "setuptools"}

    found_names = set()
    for folder in ("api", "domain", "services"):
        for folder_name, _, names in os.walk(os.path.join(ROOT, folder)):
            if "__pycache__" in folder_name:
                continue
            for name in sorted(n for n in names if n.endswith(".py")):
                with io.open(
                    os.path.join(folder_name, name), encoding="utf-8"
                ) as handle:
                    tree = ast.parse(handle.read(), filename=name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            found_names.add(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        found_names.add((node.module or "").split(".")[0])
    assert found_names, "the import sweep found nothing"

    declared = {
        r.split(">")[0].split("=")[0].split("<")[0].strip().lower()
        for r in _requirements()
    }
    missing = sorted(
        found_names - FROM_THE_STANDARD_LIBRARY - OURS - VIA_THE_CHAIN - declared - {""}
    )
    assert missing == [], "undeclared third-party import: %s" % missing


# --- what the slice cleaned up ------------------------------------------------


def test_the_frontend_sweep_walks_the_whole_tree():
    """`frontend/schema.js` sat dead for months in a place the sweep could not see.

    Nobody imported it and its own `import`s did not even resolve from there -- but
    `frontend_source.modules()` listed `main.js` and three folders by name, and it
    was in none of them. No guard could see it. Now the sweep walks the whole tree."""
    from frontend_source import FRONTEND, modules, relative

    seen = {relative(path) for path in modules()}

    on_disk = set()
    for root, folders, names in os.walk(FRONTEND):
        folders[:] = [p for p in folders if p not in ("vendor", "lib", "node_modules")]
        for name in names:
            if name.endswith(".js"):
                on_disk.add(relative(os.path.join(root, name)))

    assert on_disk, "control: is there no JS in the frontend?"
    assert seen == on_disk, "outside the sweep: %s" % sorted(on_disk - seen)
    assert "schema.js" not in seen, "a copia morta voltou"


def _names_bound_in_module(tree):
    """Everything the module defines: assignment, argument, import, def, class,
    comprehension target, `with ... as`, `except ... as`, and the builtins."""
    import ast
    import builtins

    bound_names = set(dir(builtins))
    for ast_node in ast.walk(tree):
        if isinstance(ast_node, ast.Name) and isinstance(
            ast_node.ctx, (ast.Store, ast.Del)
        ):
            bound_names.add(ast_node.id)
        elif isinstance(
            ast_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            bound_names.add(ast_node.name)
        elif isinstance(ast_node, ast.arg):
            bound_names.add(ast_node.arg)
        elif isinstance(ast_node, (ast.Import, ast.ImportFrom)):
            for alias in ast_node.names:
                bound_names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(ast_node, (ast.Global, ast.Nonlocal)):
            bound_names.update(ast_node.names)
        elif isinstance(ast_node, ast.ExceptHandler) and ast_node.name:
            bound_names.add(ast_node.name)
    return bound_names


def test_no_expression_refers_to_a_name_that_does_not_exist():
    """A dangling reference anywhere -- not only inside an f-string.

    The first version of this guard only descended into `ast.JoinedStr`, because
    that is where the first case came from (Python 3.10's `tokenize` cannot see
    inside an f-string). That made it too narrow: in a `%` format string,
    `"...%r..." % axis_name` with the loop still using `eixo` produces the same
    `NameError` and slipped through. The defect is not "f-string"; it is **a name
    that does not exist**, and `ast` knows that in any expression.

    What caught that case was a behaviour test on Leonardo's machine. A guard that
    only covers the example that created it is not a guard, it is a record of the
    example."""
    import ast

    found_items = []
    for folder, subfolders, names in os.walk(ROOT):
        subfolders[:] = repository.ours(folder, subfolders)
        for name in sorted(n for n in names if n.endswith(".py")):
            path = os.path.join(folder, name)
            with io.open(path, encoding="utf-8") as handle:
                try:
                    tree = ast.parse(handle.read(), filename=path)
                except SyntaxError:
                    continue
            for missing, line in _dangling_names(tree):
                found_items.append(
                    "%s:%d %s" % (os.path.relpath(path, ROOT), line, missing)
                )
    assert found_items == [], "name that does not exist: %s" % found_items


# --- the visible names are PER SCOPE, and not per module ---------------------
#
# The first version gathered everything the module bound, at any depth. It took
# only a neighbouring function doing `import inspect` inside itself for
# `inspect` to look available in the whole file. That is how two `NameError`
# crossed this guard in the reorganisation of the tests by module and reached
# Leonardo's `pytest`: `source()` in a file where `source` was only imported
# inside two other functions, and `inspect` in the same situation.
#
# A class body is the only scope the nested ones do NOT inherit: a method does
# not see a class attribute by its bare name, and that is the rule of Python.


def _scope_types():
    import ast

    return (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _arguments_of(args):
    names = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        names.append(args.vararg)
    if args.kwarg:
        names.append(args.kwarg)
    return {a.arg for a in names}


def _without_descending(body, when):
    """Walks `body`, stopping at the boundary of each nested scope.

    What is OUTSIDE the nested scope is visited all the same: a decorator and an
    argument default are evaluated by whoever declares, not by what is declared."""
    import ast

    scopes = _scope_types()

    def see(node):
        if isinstance(node, scopes):
            for decorator in getattr(node, "decorator_list", []):
                see(decorator)
            args = getattr(node, "args", None)
            if args is not None:
                for default in list(args.defaults) + [k for k in args.kw_defaults if k]:
                    see(default)
            for base in getattr(node, "bases", []):
                see(base)
            return
        when(node)
        for child in ast.iter_child_nodes(node):
            see(child)

    for node in body:
        see(node)


def _nested_scopes(body):
    import ast

    scopes = _scope_types()
    found = []

    def see(node):
        if isinstance(node, scopes):
            found.append(node)
            return
        for child in ast.iter_child_nodes(node):
            see(child)

    for node in body:
        see(node)
    return found


def _bound_here(body):
    import ast

    bound = set()

    def when(node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)

    _without_descending(body, when)
    for nested in _nested_scopes(body):
        if not isinstance(nested, ast.Lambda):
            bound.add(nested.name)
    return bound


def _loads_here(body):
    import ast

    found = []
    _without_descending(
        body,
        lambda node: (
            found.append((node.id, node.lineno))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            else None
        ),
    )
    return found


def _dangling_names(tree):
    """[(name, line)] used where they are not bound."""
    import ast
    import builtins

    builtin = set(dir(builtins)) | {
        "__file__",
        "__name__",
        "__doc__",
        "__all__",
        "__qualname__",
        "__module__",
        "__class__",
    }
    missing = []

    def walk(body, arguments, visible, in_a_class):
        own = visible | _bound_here(body) | arguments
        for name, line in _loads_here(body):
            if name not in own:
                missing.append((name, line))
        inherited = visible if in_a_class else own
        for nested in _nested_scopes(body):
            if isinstance(nested, ast.Lambda):
                walk([nested.body], _arguments_of(nested.args), inherited, False)
            elif isinstance(nested, ast.ClassDef):
                walk(nested.body, set(), inherited, True)
            else:
                walk(nested.body, _arguments_of(nested.args), inherited, False)

    walk(tree.body, set(), builtin, False)
    return missing


def test_the_f_string_sweep_can_see_inside_one():
    """Control: if `ast` stopped descending into an f-string, everything would pass."""
    import ast

    tree = ast.parse("x = f'{bruto} e {tabela}'")
    inside = [
        n.id
        for ast_node in ast.walk(tree)
        if isinstance(ast_node, ast.JoinedStr)
        for n in ast.walk(ast_node)
        if isinstance(n, ast.Name)
    ]
    assert sorted(inside) == ["bruto", "tabela"], inside


# --- A moved test must not lose its fixture ----------------------------------
#
# Reorganising the suite by module moves functions between files, and a fixture
# does not travel with them: it lives in the module of origin. `pytest`
# accuses that at collection -- but only whoever can run the full `pytest` sees
# it, and in this project whoever edits the files does not reach the installed
# ROSS. This guard reads the tree and says the same with no ROSS at all.

PYTEST_FIXTURES = {
    "tmp_path",
    "tmpdir",
    "capsys",
    "capfd",
    "monkeypatch",
    "request",
    "caplog",
    "recwarn",
    "pytestconfig",
}


def _parametrizados(node):
    """The names a `@pytest.mark.parametrize` supplies to this function."""
    import ast

    names = set()
    for decorator in node.decorator_list:
        if not (isinstance(decorator, ast.Call) and decorator.args):
            continue
        if "parametrize" not in ast.dump(decorator.func):
            continue
        first = decorator.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.update(part.strip() for part in first.value.split(","))
    return names


def test_every_test_argument_has_a_fixture_that_exists():
    import ast

    folder = os.path.join(ROOT, "tests")
    missing = []
    checked = 0
    for name in sorted(os.listdir(folder)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        with io.open(os.path.join(folder, name), encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=name)

        fixtures = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and any("fixture" in ast.dump(d) for d in node.decorator_list)
        }
        fixtures |= PYTEST_FIXTURES

        for node in tree.body:
            if not (
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            ):
                continue
            checked += 1
            provided = fixtures | _parametrizados(node)
            for argument in node.args.args:
                if argument.arg not in provided:
                    missing.append("%s::%s(%s)" % (name, node.name, argument.arg))

    assert checked > 200, "only %d tests swept: the sweep changed" % checked
    assert missing == [], "argument with no fixture: %s" % missing


# --- the style is checked, and checked the way the destination checks it ------


def _resolved_ruff_settings():
    """What ruff **resolved** for this folder, asked of ruff itself.

    It used to read `ruff.toml`, and that was right while the file was the only
    possible source. It stops being right the moment this folder sits inside the
    ROSS repository: the header of `ruff.toml` says the file should be deleted
    there, so that the configuration is inherited from their `pyproject.toml` --
    one source of truth instead of two that can drift.

    Both arrangements are legitimate, and a guard that reads our file passes in
    one and crashes in the other. So the subject of the guard changed from *our
    file* to **the configuration the check actually runs under**, and the only
    honest way to know that is to ask the tool that resolves it. Re-implementing
    ruff's resolution rules here would guard a model of ruff, not ruff.

    Returns None when ruff is not installed, and the caller skips -- the same
    shape as the node batteries. `check.py` is what fails loudly in that case.
    """
    probe = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True
    )
    command = [sys.executable, "-m", "ruff"] if probe.returncode == 0 else None
    if command is None:
        found = shutil.which("ruff")
        if not found:
            return None
        command = [found]

    settings = subprocess.run(
        command + ["check", "--show-settings", "."],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return settings.stdout if settings.returncode == 0 else None


def _enabled_families(settings):
    """The rule families ruff has switched on, read off the resolved list.

    `select = ["B", "C", "E", "F", "Q", "W"]` is not echoed back as those six
    letters: ruff expands it into every rule, each printed with its code --
    `unused-import (F401)`. The families are the prefixes of those codes, which
    is the same information at the far end of the resolution.
    """
    block = re.search(r"linter\.rules\.enabled = \[(.*?)\n\]", settings, re.S)
    if block is None:
        return set()
    return set(re.findall(r"\(([A-Z]+)\d+\)", block.group(1)))


def _dev_requirements():
    with io.open(
        os.path.join(ROOT, "requirements-dev.txt"), encoding="utf-8"
    ) as handle:
        return [
            line.split("#")[0].strip()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        ]


def test_the_ruff_configuration_is_the_one_the_destination_uses():
    """A configuration that is quietly loosened turns a guard into decoration.

    The cheapest way to make `ruff check` pass is to take a rule family out of
    `select`, and nothing else would notice. What is pinned is what would go
    silently: the six families ROSS enables, and the quote style, which is the
    one decision that touches every line of every file.

    Asked of ruff and not read out of a file, so that it holds in **both**
    arrangements -- our `ruff.toml` standing alone, and the file deleted with
    ROSS's `pyproject.toml` inheriting from above. See `_resolved_ruff_settings`.
    """
    settings = _resolved_ruff_settings()
    if settings is None:
        pytest.skip("ruff is not installed for this interpreter")

    families = _enabled_families(settings)
    missing = sorted({"B", "C", "E", "F", "Q", "W"} - families)
    assert missing == [], (
        "rule families %s are switched off: the check stopped meaning what "
        "ROSS's does" % missing
    )
    assert "quote_style = double" in settings, "the quote style is not ROSS's any more"


def test_the_settings_really_were_read_from_ruff():
    """Control: with nothing parsed, the guard above passes for nothing.

    An empty `linter.rules.enabled` -- a changed output format, a ruff that
    printed to stderr, a regular expression that stopped matching -- would make
    `missing` empty and the assertion vacuous. So the sweep has to find a
    plausible number of rules, not merely fail to find a missing family.
    """
    settings = _resolved_ruff_settings()
    if settings is None:
        pytest.skip("ruff is not installed for this interpreter")

    families = _enabled_families(settings)
    assert len(families) >= 6, "only %s parsed out of the resolved settings" % sorted(
        families
    )
    assert "linter.rules.enabled" in settings, "ruff stopped reporting its rule list"


def test_ruff_is_pinned_to_an_exact_version():
    """Two versions of a formatter disagree about where to break a line.

    With an open range, `ruff format --check` says whatever the most recent
    install happens to think -- and the diff lands on whoever updated last,
    for no reason they can see. Same rule as the ROSS pin, for the same reason.
    """
    line = [r for r in _dev_requirements() if r.lower().startswith("ruff")]
    assert len(line) == 1, "ruff has to appear once: %s" % line
    assert "==" in line[0], (
        "ruff is on an open range (%s): the formatter would change under the "
        "suite's feet" % line[0]
    )


def test_the_dev_requirements_declare_what_it_takes_to_run_the_suite():
    """A fresh clone has to be able to run the tests, and ruff alone does not.

    On Windows the runner came from the system Python, so nothing here noticed
    that nothing declared it. The first virtual environment ever built from
    these files -- on Linux, for slice 4 -- answered `No module named pytest`,
    and the suite could not start at all. A dependency is invisible exactly
    where it happens to be installed already.
    """
    declared = " ".join(_dev_requirements()).lower()
    assert "pytest" in declared, "nothing declares the test runner"
    assert "ruff" in declared, "nothing declares the linter"


def test_the_check_script_really_runs_ruff():
    """A configuration nobody executes is a file, not a guard."""
    with io.open(os.path.join(ROOT, "check.py"), encoding="utf-8") as handle:
        script = handle.read()
    assert '"check", "."' in script, "check.py no longer runs `ruff check`"
    assert '"format", "--check", "."' in script, (
        "check.py no longer runs `ruff format --check` -- the formatting stops "
        "being verified and starts being a matter of opinion"
    )


# --- the line endings are one, and they are LF --------------------------------


SOURCE_SUFFIXES = (
    ".py",
    ".js",
    ".html",
    ".css",
    ".json",
    ".txt",
    ".ini",
    ".toml",
    ".md",
    ".spec",
    ".yml",
)


def _has_crlf(raw):
    return b"\r\n" in raw


def _source_files():
    found = []
    for folder, folders, names in os.walk(ROOT):
        folders[:] = repository.ours(folder, folders)
        for name in sorted(names):
            if (
                name.endswith(SOURCE_SUFFIXES)
                and name not in repository.NOT_THE_REPOSITORY
            ):
                found.append(os.path.join(folder, name))
    return found


def test_the_walk_skips_a_virtual_environment(tmp_path):
    """Control: it is `pyvenv.cfg` that marks one, not the name `.venv`.

    Without this, the six guards that walk the tree would go back to reading
    `site-packages` the moment someone renames their environment folder -- and
    the symptom would again be six unrelated-looking failures.
    """
    (tmp_path / "not_called_venv").mkdir()
    (tmp_path / "not_called_venv" / "pyvenv.cfg").write_text(
        "home = /usr\n", encoding="utf-8"
    )
    (tmp_path / "api").mkdir()
    kept = repository.ours(str(tmp_path), ["not_called_venv", "api", "__pycache__"])
    assert kept == ["api"], kept


WORKFLOW = os.path.join("ci", "interface.yml")


def _workflow():
    """The workflow file, and the same file without its comments.

    The second one exists because of a mistake this project has made three
    times: a guard that reads source as text accusing the comment that explains
    the very rule it protects. The header of that file says, in prose, that it
    deliberately does not run `ruff` or `pytest` directly -- and the guard below
    forbids exactly those words. Without the stripping, documenting the rule
    would break the rule.
    """
    with io.open(os.path.join(ROOT, WORKFLOW), encoding="utf-8") as handle:
        raw = handle.read()
    steps = "\n".join(
        line for line in raw.split("\n") if not line.strip().startswith("#")
    )
    return raw, steps


def test_the_workflow_runs_our_own_check_script():
    """CI runs the recipe that already exists; it does not keep a second one.

    Spelling out `ruff check`, `pytest` and the node batteries in YAML would
    create a second verification recipe, maintained by hand, beside `check.py`.
    Two copies of one intent diverge -- and the one that rots is always the one
    nobody reads, which in CI is the one that only runs on somebody else's
    machine.
    """
    _, steps = _workflow()
    assert "python check.py" in steps, "CI no longer runs check.py"
    for other in ("ruff check", "ruff format", "-m pytest", "node test_"):
        assert other not in steps, (
            "the workflow runs %r on its own: that is a second recipe next to "
            "check.py, and copies drift" % other
        )


def test_the_workflow_builds_with_our_spec_and_runs_the_selftest():
    """A binary produced and never executed is not a deliverable, it is a hope.

    PyInstaller finishing proves the imports resolved. It says nothing about
    the files the code *opens* -- which is how the first three builds of this
    project failed. So the upload has to come after `--selftest`, not after the
    build.
    """
    _, steps = _workflow()
    assert "ross-interface.spec" in steps, "CI no longer builds with our spec"
    assert "--selftest" in steps, "CI builds the executable and never runs it"
    assert steps.index("--selftest") < steps.index(
        "upload-artifact@v4\n        with:\n          name: ross-interface"
    ), "the artifact is uploaded before the self-test proves it works"


def test_the_readme_build_commands_are_the_ones_ci_runs():
    """The step-by-step is the deliverable, so something has to run it.

    The interface goes into the ROSS repository as source: whoever wants the
    program builds it, following the README. Prose about commands rots the
    moment the commands change, and it rots in silence, because nobody executes
    a README. CI runs the same commands on three systems on every change --
    this is what keeps the two texts from being two.
    """
    with io.open(os.path.join(ROOT, "README_INTERFACE.md"), encoding="utf-8") as handle:
        readme = handle.read()
    _, steps = _workflow()
    for command in (
        "python -m PyInstaller --noconfirm ross-interface.spec",
        "--selftest",
    ):
        assert command in readme, (
            "the README stopped telling anyone to run %r" % command
        )
        assert command in steps, "CI stopped running %r" % command


def test_the_workflow_covers_the_three_systems():
    """PyInstaller does not cross-compile: three systems means three runners."""
    _, steps = _workflow()
    for system in ("ubuntu-latest", "macos-latest", "windows-latest"):
        assert steps.count(system) >= 2, (
            "%s is missing from the check or the package matrix" % system
        )


def test_the_workflow_only_wakes_for_our_folder():
    """Leonardo's third condition, written in YAML.

    The interface must never get in the way of ROSS development. A `paths`
    filter naming our folder is what makes that true mechanically: a pull
    request that does not touch it never starts this job, so a ROSS developer
    does not see it, does not wait for it, and is never blocked by it.

    The folder name is read from `FOLDER_IN_ROSS` and not written again here:
    renaming the folder has to break in one place, not two.
    """
    _, steps = _workflow()
    assert '"%s/**"' % FOLDER_IN_ROSS in steps, (
        "the workflow no longer limits itself to %s/: it would run on every "
        "pull request in the repository" % FOLDER_IN_ROSS
    )
    assert "working-directory: %s" % FOLDER_IN_ROSS in steps, (
        "the steps no longer run inside %s/" % FOLDER_IN_ROSS
    )


def test_the_installed_workflow_is_the_one_we_keep():
    """GitHub only reads workflows from the repository root, so there are two.

    This folder keeps the source of truth; installing it means copying the file
    to `.github/workflows/interface.yml` at the root of the ROSS repository.
    That is a copy, and copies drift.

    Both are read with **universal newlines, on purpose**, and the comparison is
    of text and not of bytes. The two copies live under different rules: the one
    inside this folder is pinned to LF by our `.gitattributes`, and the one at
    the repository root answers to whatever the host repository says -- on a
    Windows clone with `core.autocrlf` on, git rewrites it to CRLF. `git add`
    said exactly that, in a warning naming this file. Comparing bytes would turn
    the guard red on a machine where nobody touched anything, which is the
    failure mode this project spends most of its guards avoiding.
    """
    installed = os.path.join(
        os.path.dirname(ROOT), ".github", "workflows", "interface.yml"
    )
    if not os.path.exists(installed):
        return
    with io.open(installed, encoding="utf-8") as handle:
        there = handle.read()
    here, _ = _workflow()
    assert there == here, "%s and %s have drifted apart" % (WORKFLOW, installed)


def _list_in(path, name):
    """The plain string list assigned to `name` at the top level of a file."""
    import ast

    with io.open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return [item.value for item in node.value.elts]
    return None


def test_every_exclusion_was_measured():
    """Nothing leaves the bundle on a hunch.

    `excludes` is the one place in packaging where a wrong entry produces a
    program that builds, starts, and dies later on a path nobody exercised. So
    every name in the spec's `EXCLUDED` has to be one the probe actually put a
    question to: `tools/exclusion_probe.py` blocks it on `sys.meta_path` and
    runs the twelve analyses, which is the same absence, measured.

    The probe's list is the question; the spec's is the answer. An answer that
    was never asked is a guess wearing the clothes of a measurement.
    """
    asked = _list_in(os.path.join("tools", "exclusion_probe.py"), "CANDIDATES")
    cut = _list_in("ross-interface.spec", "EXCLUDED")
    assert asked, "the probe no longer declares CANDIDATES"
    assert cut, "the spec no longer declares EXCLUDED"
    assert len(asked) > 10, "the probe's question list shrank to %d" % len(asked)

    unmeasured = sorted(set(cut) - set(asked))
    assert unmeasured == [], (
        "excluded without the probe ever testing it: %s" % unmeasured
    )


def _scripts_that_run_them_all():
    """Every file outside `tests/` that asks the catalogue for default parameters.

    The list is derived and not written down, because a written one would not
    cover the next probe somebody adds -- and the next probe is exactly where
    the mistake this guards against gets made again. `default_params` is the
    marker: a runner names its own analysis and has every right to, while a
    script that calls `default_params` is running what the catalogue lists.

    Not quite by itself, though. The first version of this sweep took the call
    on its own as proof, and `tools/degeneracy_probe.py` falsified it: that one
    asks for the parameters of **one** analysis, deliberately, because the
    question it investigates is about that analysis and no other. Naming it
    there is the point, not a slip. So the marker gained a second half -- a
    script that sweeps the catalogue says `ANALYSES` -- and what the first half
    alone used to cover is now covered by the guard two below, which asks the
    sharper question: does the name that was written actually exist?
    """
    found = []
    for folder, folders, names in os.walk(ROOT):
        folders[:] = [f for f in repository.ours(folder, folders) if f != "tests"]
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            with io.open(path, encoding="utf-8") as handle:
                source = handle.read()
            if "def default_params" in source or "default_params" not in source:
                continue
            if "ANALYSES" not in source:
                continue
            found.append((os.path.relpath(path, ROOT), source))
    return found


def _names_handed_to_the_catalogue():
    """Every literal argument written into a `default_params(...)` call.

    Read with `ast` and not with a regular expression, because the answer has
    to tell an argument from a mention -- and the docstring above this line
    contains the word. Lesson 10 of the project notes is the record of that
    exact confusion, made three times.
    """
    import ast

    asked = []
    for folder, folders, names in os.walk(ROOT):
        folders[:] = [f for f in repository.ours(folder, folders) if f != "tests"]
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            with io.open(path, encoding="utf-8") as handle:
                source = handle.read()
            if "default_params" not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Call):
                    continue
                called = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if called != "default_params":
                    continue
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(
                        argument.value, str
                    ):
                        asked.append((os.path.relpath(path, ROOT), argument.value))
    return asked


def test_no_script_that_runs_them_all_names_an_analysis():
    """They run what the catalogue lists, and keep no list of their own.

    The first executable this project ever produced died on `KeyError: 'modal'`.
    The module is `services/analysis/modal.py`; the catalogue's key is `modes`,
    and I took the file's name for the key. It built, it started, it served the
    page, and it failed on the only check that mattered -- on a guess of mine,
    not on anything the packaging got wrong.

    The fix was not a better guess. These scripts iterate `sorted(ANALYSES)`, so
    there is no name left to get wrong; this guard is what keeps a name from
    creeping back in. It sweeps them all rather than the one that caused it,
    because a guard written around its own example is the mistake this project
    has now made four times.
    """
    offenders = []
    for relative, source in _scripts_that_run_them_all():
        named = sorted(name for name in ANALYSES if '"%s"' % name in source)
        if named:
            offenders.append("%s: %s" % (relative, named))
    assert offenders == [], "analysis names written by hand:\n  " + "\n  ".join(
        offenders
    )


def test_the_sweep_finds_the_scripts_that_run_them_all():
    """Control: with nothing found, the guard above passes for the wrong reason."""
    found = [relative for relative, _ in _scripts_that_run_them_all()]
    assert "selftest.py" in found, (
        "the self-test stopped asking the catalogue for the parameters: %s" % found
    )
    assert len(found) >= 2, "the sweep found only %s" % found


def test_every_analysis_asked_for_by_name_exists():
    """A name written by hand has to be a name the catalogue has.

    This is the guard that answers the original accident directly. The
    executable died on `KeyError: 'modal'` -- the module is `modal.py`, the
    catalogue's key is `modes`, and the two are one letter apart. The guard
    above forbids hand-written names in the scripts that sweep everything; this
    one covers the other case, a script that means one analysis and says which,
    where forbidding the name would forbid the script.

    Both are needed and neither implies the other: a sweeping script can name a
    real analysis and still be wrong to, and a targeted script can name one
    that does not exist.
    """
    wrong = [
        "%s: %r" % (relative, name)
        for relative, name in _names_handed_to_the_catalogue()
        if name not in ANALYSES
    ]
    assert wrong == [], "asked the catalogue for an analysis it does not have:\n  " + (
        "\n  ".join(wrong)
    )


def test_the_sweep_reads_the_arguments_and_not_the_word():
    """Control: an empty list makes the guard above pass for nothing.

    It also pins the thing that made this worth reading with `ast`: the sweep
    must find the argument inside `tools/degeneracy_probe.py` -- the file that
    forced the split -- and must not be counting the word where it merely
    appears in prose.
    """
    asked = _names_handed_to_the_catalogue()
    assert asked, "nothing hands a literal to default_params any more"
    probe = [
        name for relative, name in asked if relative.endswith("degeneracy_probe.py")
    ]
    assert probe == ["modes"], (
        "the probe that measures one analysis no longer names it: %s" % probe
    )


def test_the_walk_uses_what_the_gitignore_already_declares(tmp_path):
    """Control: `build/` and `dist/` are pruned because a file says so.

    They are not in any tuple in `repository.py`. They are in the `.gitignore`,
    which is where somebody adding build output already writes it down. The
    first version of that module listed names by hand, and `build/` and `dist/`
    reproduced -- one slice later -- the exact failure `.venv/` had produced.
    """
    assert {"build", "dist"} <= repository.NOT_THE_REPOSITORY, (
        "the .gitignore no longer declares the build output, or it stopped being read"
    )
    assert "dist" not in repository.NEVER_OURS, (
        "`dist` went back to being a hand-written name: the derivation is decorative"
    )
    assert len(repository.NOT_THE_REPOSITORY) > len(repository.NEVER_OURS), (
        "nothing was read from the .gitignore"
    )


def test_the_line_ending_detector_tells_the_two_apart():
    """Control: a detector that never sees CRLF would leave the guard green."""
    assert _has_crlf(b"a\r\nb")
    assert not _has_crlf(b"a\nb")
    assert not _has_crlf(b"a\rb")


def test_the_line_endings_survive_a_checkout():
    """The guard below measures the tree; this one measures what git will do.

    `core.autocrlf` is on by default in the Git for Windows installer, and with
    it a clone rewrites every text file to CRLF. The working tree would then be
    CRLF, the guard below would go red, and nobody would have touched a file --
    the change came from outside the repository. A `.gitattributes` that pins
    `eol=lf` is what makes the measurement below true for everyone who clones,
    and not only for the machine where the files were written.
    """
    with io.open(os.path.join(ROOT, ".gitattributes"), encoding="utf-8") as handle:
        rules = "\n".join(
            line
            for line in handle.read().split("\n")
            if not line.strip().startswith("#")
        )
    assert "text=auto" in rules and "eol=lf" in rules, (
        "the checkout no longer pins LF: a clone on Windows would rewrite every "
        "text file to CRLF and turn the guard below red with nobody at fault"
    )


def test_every_source_file_uses_lf_line_endings():
    """A file saved with CRLF changes what the guards that read source see.

    Some of them open the file with `newline=""`, which keeps the `\\r`; the
    others use universal newlines, which drops it. Either is defensible; what is
    not is the two disagreeing about the same file. And nothing shows it:
    `tests/test_rotor.py` reached slice 3 with 135 CRLF lines and 72 LF ones,
    written by two editors over months, and every test stayed green.

    It matters more now than it did: the destination repository is built on
    Linux, macOS and Windows, and a mixed file is the kind of difference that
    only appears on the machine nobody is looking at.
    """
    files = _source_files()
    assert len(files) > 100, "the sweep stopped finding files: %d" % len(files)

    with_crlf = []
    for path in files:
        with io.open(path, "rb") as handle:
            if _has_crlf(handle.read()):
                with_crlf.append(os.path.relpath(path, ROOT))
    assert with_crlf == [], "CRLF line endings: %s" % sorted(with_crlf)


# --- a deprecation is a scheduled failure -------------------------------------


def _pytest_ini():
    with io.open(os.path.join(ROOT, "pytest.ini"), encoding="utf-8") as handle:
        return handle.read()


def _filter_lines():
    """The filters declared in `pytest.ini`, without the comments."""
    lines, inside = [], False
    for raw in _pytest_ini().split("\n"):
        if raw.startswith("filterwarnings"):
            inside = True
            continue
        if inside:
            if not raw.startswith((" ", "\t")) or not raw.strip():
                break
            if not raw.strip().startswith("#"):
                lines.append(raw.strip())
    return lines


SCHEDULED = ("DeprecationWarning", "PendingDeprecationWarning", "FutureWarning")


def test_a_deprecation_fails_the_run_instead_of_being_printed():
    """The guard for the lesson `rotor_assembly.py:2418` cost us.

    That warning said "will error in future" in every report we produced, for
    four slices, and was read every time by someone who then did something else.
    numpy 2.5 collected: the clearance analysis stopped running on the three
    systems at once. What failed was not the code, it was the assumption that a
    printed warning gets acted on.

    A mechanism nobody can switch off by accident is worth more than the
    intention to look. If this line goes, the suite goes back to printing them.
    """
    declared = _filter_lines()
    assert declared, "pytest.ini no longer declares filterwarnings at all"
    for category in SCHEDULED:
        assert "error::%s" % category in declared, (
            "%s is no longer an error: a deprecation would go back to being a "
            "line in the report that nobody reads" % category
        )


def test_every_silenced_warning_says_why_it_is_silenced():
    """An `ignore` with no reason is how this turns back into what it replaced.

    There are none today, and that is the point of writing the rule now rather
    than the first time one is needed: the first one will be added in a hurry,
    on a red CI, by someone who wants the run green. The comment above it is
    what lets the next person tell a measured decision from a workaround."""
    text = _pytest_ini().split("\n")
    unexplained = []
    for number, raw in enumerate(text):
        if not raw.strip().startswith("ignore"):
            continue
        above = [line for line in text[:number] if line.strip()]
        explained = above and above[-1].strip().startswith("#")
        if not explained:
            unexplained.append("pytest.ini:%d %s" % (number + 1, raw.strip()))
    assert unexplained == [], (
        "silenced warnings with no reason written above them:\n  "
        + "\n  ".join(unexplained)
    )


def test_the_reason_sweep_can_tell_an_explained_line_from_a_bare_one():
    """Control: without this, the test above passes because there is nothing to find."""
    from_a_bare_file = [
        "filterwarnings =",
        "    error::DeprecationWarning",
        "    ignore:x:UserWarning",
    ]
    explained = [
        "filterwarnings =",
        "    # ROSS 2.3.0, waiting on upstream",
        "    ignore:x:UserWarning",
    ]

    def unexplained(lines):
        out = []
        for number, raw in enumerate(lines):
            if not raw.strip().startswith("ignore"):
                continue
            above = [line for line in lines[:number] if line.strip()]
            if not (above and above[-1].strip().startswith("#")):
                out.append(raw.strip())
        return out

    assert unexplained(from_a_bare_file), "the sweep does not see a bare ignore"
    assert unexplained(explained) == [], "the sweep does not see a written reason"


def test_the_shipped_code_cites_no_note_of_ours():
    """Our working notes do not travel with the folder, so a citation dangles.

    The `claude/` documents live in the project, not in `interface/`. Code that
    points at one of them reads, to a ROSS maintainer, as a reference to a file
    that does not exist -- and there is no way for them to tell whether it was
    deleted or never shipped.

    Four such citations existed when this guard was written, three of them added
    the same day by the slice that found them: the Portuguese sweep flagged the
    file **names**, which is how a problem about repositories arrived disguised
    as a problem about language. What the comments should carry is the fact that
    was measured, not the path to where it was written down.
    """
    import re

    # A **path**, not the folder's name. The first version of this line looked
    # for "claude/" and immediately accused the docstring above, which has to
    # name the folder to explain the rule -- lesson 10, for the fourth time in
    # this project: a guard that reads source has to discount the text that
    # talks about the source.
    citation = re.compile(r"claude/[\w.-]+\.md")
    citing = []
    for relative, path in _source_files_by_name(".py"):
        with io.open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if citation.search(line):
                    citing.append("%s:%d" % (relative, number))
    assert citing == [], (
        "code pointing at a note that does not ship:\n  " + "\n  ".join(citing)
    )


def _source_files_by_name(suffix):
    """Every file of ours with that suffix, as (relative, absolute)."""
    found = []
    for folder, folders, names in os.walk(ROOT):
        folders[:] = repository.ours(folder, folders)
        for name in sorted(names):
            if name.endswith(suffix):
                path = os.path.join(folder, name)
                found.append((os.path.relpath(path, ROOT), path))
    return found


def test_the_citation_sweep_reads_the_files_it_claims_to():
    """Control: an empty sweep would make the guard above pass for nothing."""
    found = _source_files_by_name(".py")
    assert len(found) > 50, "the sweep found %d python files" % len(found)
    assert any(relative == "app.py" for relative, _ in found)
