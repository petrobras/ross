# -*- mode: python ; coding: utf-8 -*-
"""How the interface becomes an executable, on the three systems.

## Three systems means three machines

PyInstaller does not cross-compile. It freezes the interpreter of the machine
it runs on, together with that platform's binary wheels -- numba, llvmlite,
scipy and numpy are all compiled code. There is no `--target-os`. So this one
file is built three times, on three runners, and produces three different
bundles. That is why the packaging lives next to a CI matrix and not next to a
button on somebody's desk.

## One folder, not one file

`--onefile` unpacks the whole bundle into a temporary folder on **every** run.
With numba, scipy, pandas and plotly inside, that is hundreds of megabytes
copied before the first pixel. `--onedir` starts in seconds, and -- the part
that matters more here -- when a file is missing the error names the file
instead of dying during an extraction nobody can see.

## What PyInstaller does not find on its own

It follows `import`. It does not follow a file *opened by path* from inside an
installed package, and this interface depends on two of those:

* `plotly/package_data/plotly.min.js`, served by the `/lib/plotly.min.js` route;
* ROSS's `new_units.txt`, which pint reads for the units the library adds --
  ROSS's own `pyproject.toml` lists it under `include`, which is how we know it
  ships as data and not as code.

And the list of *which* packages do that is not written here either. The first
build proved why: it died on `ccp/config/new_units.txt`, inside
`ccp-performance` -- a package this project never mentions, imported by ROSS
from `seals/labyrinth_seal.py`. Three names I could think of, and the fourth
was the one that mattered. So the spec derives the closure from what
`requirements.txt` declares and asks each installed package what it carries.

## What is cut, and how it was decided

The first bundle was **619 MB**, and most of it was not rotordynamics; with
the cut below it is **464 MB**, and the twelve analyses still run. Every
name in `EXCLUDED` below was **measured** by `tools/exclusion_probe.py`, which
blocks a package on `sys.meta_path` -- the same absence `excludes` produces --
and then runs the executable's own self-test over all twelve analyses. The same
sixteen came out on Windows and on Linux, so there is one list and not two.

Four candidates went back in, and their import chains are worth writing down,
because they are ROSS's and not ours:

    ross.bearing_seal_element -> control -> control.timeplot -> matplotlib.pyplot -> PIL
    ross.rotor_assembly -> ross.seals.labyrinth_seal -> ccp -> sklearn -> sklearn.utils.fixes -> pyarrow

That is 120 MB of plotting library, image library, machine learning and Arrow
that `import ross` requires **at import time**, on any machine, for any use.
Nothing here can cut them; only the libraries above can.

Still no `hiddenimports`. Adding submodules "just in case" feels safe and is a
guess; there has been no failure asking for one. A spec full of entries nobody
can explain is how packaging becomes folklore.

`console=True` for the same reason: while the executable is young, a window
that prints the traceback is worth more than a window that looks tidy.
`--selftest` also writes to that console.
"""

import io
import os
import re
from importlib import metadata

from PyInstaller.utils.hooks import collect_data_files

NAME = "ross-interface"
HERE = SPECPATH  # noqa: F821 -- injected by PyInstaller: the folder of this file


def declared_distributions():
    """The distributions `requirements.txt` names -- our declared world."""
    names = []
    with io.open(os.path.join(HERE, "requirements.txt"), encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#")[0].strip()
            if line:
                names.append(re.split(r"[<>=!\[;@\s]", line, 1)[0])
    return [name for name in names if name]


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_closure(roots):
    """Every distribution the roots require, transitively.

    The first build died on `ccp/config/new_units.txt` -- a data file inside
    `ccp-performance`, which ROSS imports from `seals/labyrinth_seal.py` on
    `import ross`. The spec was collecting data for plotly, ross and pint:
    the three I could name. The fourth existed, and I could not name it.

    So the list is not written here. It is derived from the metadata of what is
    installed, starting at what `requirements.txt` declares -- because a list of
    package names rots exactly like any other copy, and it rots in the worst
    place: a clean build on somebody else's machine, months from now.
    """
    seen, queue = set(), [canonical(root) for root in roots]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        try:
            required = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        seen.add(name)
        for line in required:
            # `foo; extra == "dev"` is a dependency of an extra nobody asked for
            if "extra ==" in line:
                continue
            queue.append(canonical(re.split(r"[<>=!\[;@\s(]", line, 1)[0]))
    return seen


def importable_names(distributions):
    """Distribution names are not import names: `ccp-performance` imports `ccp`."""
    by_distribution = {}
    for module, owners in metadata.packages_distributions().items():
        for owner in owners:
            by_distribution.setdefault(canonical(owner), set()).add(module)
    found = set()
    for name in distributions:
        found |= by_distribution.get(name, set())
    return sorted(found)


# Measured by `tools/exclusion_probe.py`, never guessed. Anything added here
# without the probe agreeing is a bet, and `test_every_exclusion_was_measured`
# is what refuses the bet.
EXCLUDED = [
    "IPython",
    "adodbapi",
    "dash",
    "google",
    "google_auth_httplib2",
    "googleapiclient",
    "httplib2",
    "isapi",
    "notebook",
    "oauth2client",
    "pytest",
    "pythonwin",
    "tkinter",
    "uritemplate",
    "win32com",
    "xlwings",
]

# The interface's own files, plus the data of every package in the declared
# dependency closure that survives the cut. Test fixtures are left out: they are
# the bulk of what numpy, scipy and pandas carry, and nothing here reads them.
datas = [(os.path.join(HERE, "frontend"), "frontend")]
collected = []
for package in importable_names(dependency_closure(declared_distributions())):
    if package in EXCLUDED:
        continue
    try:
        files = collect_data_files(
            package, excludes=["**/tests/**", "**/test/**", "**/testing/**"]
        )
    except Exception:  # a module that is not a package carries no data
        files = []
    if files:
        datas += files
        collected.append("%s(%d)" % (package, len(files)))

print("[spec] data collected from: %s" % " ".join(sorted(collected)))
print("[spec] %d data entries in total" % len(datas))

analysis = Analysis(  # noqa: F821
    [os.path.join(HERE, "app.py")],
    pathex=[HERE],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDED,
    noarchive=False,
)

archive = PYZ(analysis.pure)  # noqa: F821

executable = EXE(  # noqa: F821
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

folder = COLLECT(  # noqa: F821
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=NAME,
)
