# -*- coding: utf-8 -*-
"""Run the whole suite and write the result to `check_report.txt`.

It exists for a concrete reason: whoever writes the code of this interface (the
assistant) reaches the files, but does not reach the installed ROSS -- that
lives on a Windows machine the bridge's shell does not execute. Without a
report in a file, every verification turned into copy-and-paste from a
terminal, and what is not pasted is not seen.

Usage:

    python check.py

It decides nothing and fixes nothing: it runs `ruff`, runs `pytest`, runs the
`tests/js/` batteries with `node` and records what happened -- including what was
**skipped**, because a silent skip is exactly what hid two broken batteries for
months.
"""

import io
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(ROOT, "check_report.txt")


def run(description, command, cwd=None):
    start = time.time()
    try:
        output = subprocess.run(
            command,
            cwd=cwd or ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        code, text = output.returncode, (output.stdout or "") + (output.stderr or "")
    except OSError as error:
        code, text = -1, "could not execute: %s" % error
    return {
        "description": description,
        "command": " ".join(command),
        "code": code,
        "seconds": time.time() - start,
        "output": text,
    }


def installed_versions():
    """What the suite actually ran against.

    Read from the distribution metadata, and deliberately not by importing:
    this block exists because of a run where `import ross` **was** the failure
    (ROSS 2.3.0 against a plotly that had dropped `scattermapbox`), and a report
    that cannot name the versions in exactly that case is a report that omits
    the answer. `python --version` was in the header from the start; the
    libraries were not, and the library was where the difference lived.
    """
    from importlib import metadata

    named = []
    for name in (
        "ross-rotordynamics",
        "plotly",
        "numpy",
        # Numba is not ours and not named in `requirements.txt`: it arrives
        # through ROSS, and it is what compiles the crack and the campbell. It
        # earned this line by breaking a whole run -- `ImportError: Numba needs
        # NumPy 2.4 or less` -- while the header reported numpy and stayed
        # silent about the package that was refusing it. A header that names one
        # half of an incompatibility is a header that hides the other.
        "numba",
        "scipy",
        "flask",
        "pint",
        "toml",
        "pytest",
        "ruff",
    ):
        try:
            named.append("%s %s" % (name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            named.append("%s MISSING" % name)
    return named


def source_revision():
    """Which revision of the code produced this report.

    The header already names what the suite ran *against*: python, node and
    nine libraries. It did not name what it ran *on*. Two machines reported
    `0 of 16` on the same afternoon and one of them was three tests behind --
    a working copy that had not been synchronised. Both reports were true, both
    looked identical, and nothing in either said which tree it had measured.

    A green report from two machines is only worth something if the two
    measured the same code.
    """
    found = shutil.which("git")
    if found is None:
        return "git is not installed"
    where = subprocess.run(
        [found, "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if where.returncode != 0:
        return "not a git repository"
    changed = subprocess.run(
        [found, "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    marker = " + uncommitted changes" if (changed.stdout or "").strip() else ""
    return where.stdout.strip() + marker


def node_description():
    """The path and the version: an old node is a different failure from none."""
    found = shutil.which("node")
    if found is None:
        return "MISSING"
    probe = subprocess.run(
        [found, "--version"], capture_output=True, text=True, encoding="utf-8"
    )
    return "%s  %s" % (found, probe.stdout.strip() or "(version unknown)")


def ruff_command():
    """How to call ruff, or None if it is not installed.

    `python -m ruff` first, and not the `ruff` on the PATH, because the version
    matters: a formatter is a rewriting tool, and two versions of it disagree
    about where to break a line. This way the ruff that checks the code is the
    one installed next to the interpreter that runs the tests.
    """
    probe = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "ruff"]
    found = shutil.which("ruff")
    return [found] if found else None


MISSING_RUFF = """ruff is not installed for this interpreter.

`ruff check` and `ruff format --check` are what ROSS's CI runs, and the
interface is meant to go into that repository. Without them the style drifts
until the pull request is the place where it is discovered.

    %s -m pip install -r requirements-dev.txt
"""


def main():
    parts = []

    ruff = ruff_command()
    if ruff is None:
        for description in ("ruff check", "ruff format --check"):
            parts.append(
                {
                    "description": description,
                    "command": "python -m ruff ...",
                    "code": -1,
                    "seconds": 0.0,
                    "output": MISSING_RUFF % sys.executable,
                }
            )
    else:
        parts.append(run("ruff check", ruff + ["check", "."]))
        parts.append(run("ruff format --check", ruff + ["format", "--check", "."]))

    # `-ra` lists the skips in full: an invisible skip is coverage nobody knows
    # they have lost.
    parts.append(run("pytest", [sys.executable, "-m", "pytest", "-ra", "-q"]))

    js_folder = os.path.join(ROOT, "tests", "js")
    if shutil.which("node") is None:
        parts.append(
            {
                "description": "behaviour batteries (node)",
                "command": "node tests/js/test_*.js",
                "code": -1,
                "seconds": 0.0,
                "output": "node is not on the PATH.\n\n"
                "These are the only guards that measure the SCREEN rather\n"
                "than the text of the code. Without node they do not run --\n"
                "and it was in that silence that two of them stayed broken\n"
                "for months. Install Node.js and run again.",
            }
        )
    else:
        for name in sorted(os.listdir(js_folder)):
            if name.startswith("test_") and name.endswith(".js"):
                parts.append(run(name, ["node", name], cwd=js_folder))

    failures = [p for p in parts if p["code"] != 0]

    with io.open(REPORT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("ROSS INTERFACE CHECK\n")
        handle.write("%s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        handle.write("python %s on %s\n" % (sys.version.split()[0], sys.platform))
        handle.write("node: %s\n" % node_description())
        handle.write("installed: %s\n" % ", ".join(installed_versions()))
        handle.write("source: %s\n\n" % source_revision())

        handle.write("SUMMARY\n")
        for p in parts:
            handle.write(
                "  %-32s %-6s %5.1fs\n"
                % (p["description"], "ok" if p["code"] == 0 else "FAILED", p["seconds"])
            )
        handle.write("\n%d of %d failed\n" % (len(failures), len(parts)))

        for p in parts:
            handle.write(
                "\n"
                + "=" * 70
                + "\n%s  (%s)\nexit code: %s\n%s\n"
                % (p["description"], p["command"], p["code"], "-" * 70)
            )
            handle.write(p["output"].strip() + "\n")

    print("report in %s" % REPORT)
    print("%d of %d failed" % (len(failures), len(parts)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
