# -*- coding: utf-8 -*-
"""Bridge to the behaviour batteries written in node (tests/js/).

Some defects only show up in the browser's behaviour: the form that disappears
when the list is redrawn, the old request that overwrites the chart, the screen
that ends up half in each language. The batteries build minimal stand-ins (DOM,
`fetch`, `localStorage`) and **import the real modules** -- since Phase 3 they
no longer read the source as text.

## The silence that cost dearly

These batteries are the only guards in the project that measure the SCREEN
rather than the text of the code. Even so they went months without running, and
the failure mode is the lesson:

* the `skipif` below skips everything when `node` is not on the PATH -- and it
  is not, on the Windows Python where the suite runs day to day;
* a `pytest` that skips 14 tests is a green `pytest`: the default output does
  not even list the skips;
* in that silence, two batteries rotted. `test_formbox.js` and
  `test_requests.js` read `frontend/app.js` with `require()`, and the file
  stopped existing in the move to modules, while the `package.json` of
  `tests/js/` became `"type": "module"`. Both blew up on their first line for
  months. The `test_requests.js` one covered FE-05 of the audit -- an
  out-of-order answer overwriting the chart -- so the coverage of a critical
  defect already fixed vanished with nobody seeing.

That is why the bridge now **fails, instead of skipping, when the environment
says it should have node**: in CI, or when `ROSS_INTERFACE_REQUIRE_NODE` is set.
Skipping is still right on a machine that only wants to run the backend; what
must not happen is the absence going unnoticed where it matters."""

import glob
import io
import json
import os
import shutil
import subprocess

import pytest

JS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "js")
BATTERIES = sorted(
    os.path.basename(c) for c in glob.glob(os.path.join(JS_DIR, "test_*.js"))
)

HAS_NODE = shutil.which("node") is not None


def _declared_node_floor():
    """The floor, read from the file node itself reads it from."""
    with io.open(os.path.join(JS_DIR, "package.json"), encoding="utf-8") as handle:
        declared = json.load(handle)["engines"]["node"]
    assert declared.startswith(">="), "unexpected engine range: %r" % declared
    return tuple(int(piece) for piece in declared[2:].split("."))


def _node_version():
    output = subprocess.run(
        ["node", "--version"], capture_output=True, text=True, encoding="utf-8"
    )
    assert output.returncode == 0, output.stderr
    return tuple(int(piece) for piece in output.stdout.strip().lstrip("v").split("."))


# Where the absence of node is a defect and not a choice: ROSS's CI runs on
# runners that ship node, and there the screen coverage has to happen.
REQUIRE_NODE = bool(
    os.environ.get("CI") or os.environ.get("ROSS_INTERFACE_REQUIRE_NODE")
)


def test_the_version_floor_is_declared_where_node_looks_for_it():
    """Control: `package.json` is where node reads it, so that is where it lives.

    Most of the batteries use top-level `await`, which node only
    understands from 14.8 on. Repeating the number here would make two copies
    of it; reading it from there makes one.
    """
    assert _declared_node_floor() >= (14, 8), (
        "the batteries use top-level await: below 14.8 node cannot even parse them"
    )


@pytest.mark.skipif(not HAS_NODE, reason="node is not on the PATH")
def test_the_node_on_this_machine_is_new_enough():
    """An old node fails with an error that never mentions the version.

    Ubuntu 22.04 ships node 12.22 as `nodejs`, and on it ten batteries died at
    once with `SyntaxError: Unexpected reserved word`, pointing at the word
    `await`. Nothing in that message says "your node is too old", and the three
    that happened not to use top-level await passed -- which reads like a code
    defect in ten files rather than one fact about the machine.
    """
    floor, found = _declared_node_floor(), _node_version()
    assert found >= floor, (
        "node %s is below the declared floor %s: the batteries use top-level "
        "await and would die with `SyntaxError: Unexpected reserved word`"
        % (".".join(map(str, found)), ".".join(map(str, floor)))
    )


def test_node_is_available_where_it_is_required():
    """With no node there is no screen coverage -- and that must not pass in silence."""
    if not REQUIRE_NODE:
        pytest.skip("node is optional on this machine (set CI=1 to require it)")
    assert HAS_NODE, (
        "node is not on the PATH: the %d behaviour batteries did not run, "
        "and they are the only guards that measure the screen" % len(BATTERIES)
    )


@pytest.mark.skipif(not HAS_NODE, reason="node is not on the PATH")
@pytest.mark.parametrize("battery", BATTERIES)
def test_javascript_suites(battery):
    result = subprocess.run(
        ["node", os.path.join(JS_DIR, battery)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=JS_DIR,
    )
    assert result.returncode == 0, "\n" + result.stdout + result.stderr


def test_the_javascript_suites_are_discovered():
    """If the folder emptied by mistake, the `parametrize` above would pass empty."""
    assert len(BATTERIES) >= 12, "baterias encontradas: %s" % BATTERIES


def test_every_battery_is_a_module_and_not_the_old_commonjs():
    """What killed the two dead batteries, turned into a guard.

    `require()` in a folder declared as ESM, or a read of a file that no longer
    exists, throws on the first line -- and only whoever ran node would see it.
    Because this sweep reads the source, it accuses even on a machine with no node."""
    import io
    import json

    with io.open(os.path.join(JS_DIR, "package.json"), encoding="utf-8") as handle:
        assert json.load(handle).get("type") == "module", (
            "the `package.json` changed: this guard is measuring the wrong rule"
        )

    import re

    problems = []
    for battery in BATTERIES:
        with io.open(os.path.join(JS_DIR, battery), encoding="utf-8") as handle:
            js = handle.read()
        # Without stripping the comments, the guard accuses the header that EXPLAINS
        # the defect -- which is what happened on its first run, and is the same trap
        # as `pytest.ini` in the previous slice. Reading code as text always means
        # discounting the text that talks about the code.
        source = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", js)
        if "require(" in source:
            problems.append("%s: uses require() in an ES module" % battery)
        if "app.js" in source:
            problems.append(
                "%s: reads app.js, which has not existed since Phase 3" % battery
            )
        # On Windows, `new URL(...).pathname` gives back `/C:/Users/...` -- with the
        # leading slash. Passing that to `path.join` produces `C:\C:\Users\...` and an
        # ENOENT. It was the only defect that showed up when the suite first ran
        # outside Linux, and it comes back if someone repeats the idiom.
        if re.search(r"import\.meta\.url\s*\)\s*\.pathname", source):
            problems.append(
                "%s: builds a path with `.pathname`; use `fileURLToPath`" % battery
            )
    assert problems == [], problems
