# -*- coding: utf-8 -*-
"""Which of the fat packages the interface can do without -- measured, not guessed.

The first bundle came to **619 MB**, and most of the weight is not
rotordynamics: 95 MB of Google API discovery documents, 79 MB of pyarrow, 30 MB
of Dash JavaScript, plus scikit-learn, matplotlib and Pillow. None of it is
imported by this interface; all of it arrives because ROSS imports
`ccp-performance`, which drags that world behind it.

Cutting them means `excludes=[...]` in the spec -- and every entry there is a
bet that nothing imports the module, ever, on any path. The obvious way to
settle the bet is to cut, build and run the self-test. That costs a full build
on two machines per attempt, and the traceback names only the first mistake.

This does the same experiment in seconds. A finder on `sys.meta_path` that
refuses the candidates reproduces exactly what PyInstaller's `excludes` does at
runtime -- the module is simply not there -- and then the probe runs
`selftest.run()`, the same oracle the executable uses, so what is measured here
and what is measured there are the same twelve analyses.

Each attempt runs in a **fresh process**: once a module is imported it stays
imported, and a second attempt in the same interpreter would measure nothing.
The probe re-invokes itself, learns which module was really needed, puts that
one back, and tries again -- so one run converges on the smallest exclusion set
that still passes, instead of one bet per build.

Only what surfaces in a traceback counts as needed. A blocked import that some
`try: import ... except ImportError:` swallowed was refused and *not* needed:
putting that one back would be paying for a cut nobody uses.

Run it in the venv where ROSS is installed, from the project root:

    python tools/exclusion_probe.py

It changes nothing. It prints the list to paste into the spec's `excludes`.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# What the first build showed as heavy and foreign to rotordynamics. Being on
# this list is a question, not an answer: the probe is what decides.
CANDIDATES = [
    "googleapiclient",
    "google",
    "google_auth_httplib2",
    "httplib2",
    "uritemplate",
    "oauth2client",
    "pyarrow",
    "dash",
    "sklearn",
    "matplotlib",
    "PIL",
    "xlwings",
    "win32com",
    "pythonwin",
    "isapi",
    "adodbapi",
    "tkinter",
    "IPython",
    "notebook",
    "pytest",
]

REFUSED = []


class Blocker:
    """Makes a module unimportable, the way `excludes` does inside a bundle."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.names:
            REFUSED.append(root)
            raise ImportError("blocked by the exclusion probe: %s" % fullname)
        return None


def child(blocked):
    sys.meta_path.insert(0, Blocker(blocked))
    code = 1
    try:
        import selftest

        code = selftest.run()
    except BaseException as error:  # an import that fails outside a check
        print("        %s: %s" % (type(error).__name__, error))
    print("PROBE-ATTEMPTED %s" % ",".join(sorted(set(REFUSED))))
    return code


def main():
    blocked, needed = list(CANDIDATES), []
    for attempt in range(1, len(CANDIDATES) + 2):
        print("\n=== attempt %d, blocking %d packages ===" % (attempt, len(blocked)))
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", ",".join(blocked)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=ROOT,
        )
        print(result.stdout.strip()[-3000:])
        if result.returncode == 0:
            print("\n%d packages can go:\n" % len(blocked))
            print("    excludes=[")
            for name in sorted(blocked):
                print('        "%s",' % name)
            print("    ]")
            if needed:
                print(
                    "\nput back, because something imports them: %s" % ", ".join(needed)
                )
            return 0

        surfaced = re.findall(
            r"blocked by the exclusion probe: ([\w.]+)", result.stdout
        )
        guilty = sorted({name.split(".")[0] for name in surfaced} & set(blocked))
        if not guilty:
            print("\nit failed with no blocked import in the traceback:")
            print("this is not about the cuts -- read the failure above.")
            print(result.stderr.strip()[-2000:])
            return 1
        for name in guilty:
            blocked.remove(name)
            needed.append(name)
        print("needed after all: %s" % ", ".join(guilty))
    return 1


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.exit(child(sys.argv[sys.argv.index("--child") + 1].split(",")))
    sys.exit(main())
