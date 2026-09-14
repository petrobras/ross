# -*- coding: utf-8 -*-
"""What the executable checks about itself, before anyone trusts it.

**A build that finishes is not a build that works.** PyInstaller follows what
the code *imports*; it does not follow what the code *opens*. This interface
depends on two files that are data inside installed packages, reached by path:

* `plotly/package_data/plotly.min.js`, which the `/lib/plotly.min.js` route
  serves -- the charting library the browser loads;
* ROSS's own `new_units.txt`, which pint reads to learn the units the library
  adds (it is in the `include` of ROSS's `pyproject.toml`, which is how we know
  it ships as package data and not as code).

Neither is an `import`, so neither is followed by default. Missing, they produce
an executable that starts, shows the page, and dies on the first analysis --
the worst possible moment for the discovery, because by then somebody believed
it worked.

So the check is not "does it import". It builds a six-element rotor and runs a
modal analysis through the same route the screen uses, which exercises ROSS,
numba and scipy **inside the bundle** rather than merely their import. Run it
with:

    ross-interface --selftest

It prints one line per check and exits non-zero at the first failure, which is
what makes it usable as the gate of a build in CI: a binary that was produced
but never ran is not a deliverable, it is a hope.
"""

import io
import os
import sys

ROTOR = {
    "name": "Selftest",
    "materials": [{"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}],
    "shafts": [
        {"L": "100", "odl": "50", "idl": "0", "material": "Steel"} for _ in range(6)
    ],
    "disks": [{"n": "3", "m": "32", "Id": "0.2", "Ip": "0.3"}],
    "bearings": [
        {"element_type": "BASIC", "n": "0", "kxx": "1e6", "cxx": "1e3"},
        {"element_type": "BASIC", "n": "6", "kxx": "1e6", "cxx": "1e3"},
    ],
    "gears": [],
    "seals": [],
    "couplings": [],
    "pointmasses": [],
}


def _the_page_is_there():
    from api.paths import FRONTEND_DIR

    index = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index):
        return "the frontend is not in the bundle: %s" % index
    with io.open(index, encoding="utf-8") as handle:
        if "data-i18n" not in handle.read():
            return "%s is there but is not our page" % index
    return None


def _the_charting_library_is_there():
    from api.paths import PLOTLY_BUNDLE

    if not os.path.exists(PLOTLY_BUNDLE):
        return (
            "plotly.js did not come along: %s\n"
            "        the spec has to collect plotly's package data" % PLOTLY_BUNDLE
        )
    if os.path.getsize(PLOTLY_BUNDLE) < 100000:
        return "%s is too small to be plotly.js" % PLOTLY_BUNDLE
    return None


def _every_analysis_answers():
    """The whole stack, for all twelve, through the route the screen uses.

    One analysis proves the bundle is complete. It does not prove a cut is safe,
    and cutting is what comes next: `CoolProp` is what `ccp` reads for gas
    properties, and what needs it are the seal analyses -- which a modal run
    never touches. An oracle covering one path licenses cuts that break eleven.

    The list is not written here. It is `sorted(ANALYSES)`, which is also why
    this file no longer contains the name of any analysis: the first executable
    this project produced died on `KeyError: 'modal'`, a name I had guessed from
    a file name. What cannot be written down cannot be guessed wrong.
    """
    from api import create_app
    from api.security import SESSION_TOKEN
    from api.waiting import answer_for
    from domain.analysis_catalog import ANALYSES, default_params

    client = create_app().test_client()
    header = {"X-ROSS-Token": SESSION_TOKEN}
    broken = []

    for name in sorted(ANALYSES):
        body = {
            "analysis_type": name,
            "params": default_params(name),
            "conversion_type": "none",
            "project": ROTOR,
        }
        # Since slice 6c the route answers a **job**, not a chart: `202` with a
        # name, and the chart from `GET /api/jobs/<id>`. This file went on
        # reading the old contract, called all twelve analyses failed, and took
        # the CI package job down on three systems -- while `check.bat`, which
        # does not build, stayed green. The waiting is `api/waiting.py`, the one
        # place that knows it, and it ships precisely so that this file can
        # reach it.
        try:
            answer = answer_for(
                client, client.post("/run_analysis", json=body, headers=header), header
            )
            if answer is None:
                payload = {"message": "the job never answered"}
            else:
                payload = answer.get_json()
        except Exception as error:
            payload = {"message": "%s: %s" % (type(error).__name__, error)}
        payload = payload or {}

        if payload.get("status") == "success" and payload.get("plot_json"):
            print("            %-18s ok" % name)
        else:
            print(
                "            %-18s FAILED  %s"
                % (name, (payload.get("message") or "no chart in the answer")[:150])
            )
            broken.append(name)

    if broken:
        return "%d of %d analyses failed: %s" % (
            len(broken),
            len(ANALYSES),
            ", ".join(broken),
        )
    return None


CHECKS = (
    ("the page is in the bundle", _the_page_is_there),
    ("plotly.js is in the bundle", _the_charting_library_is_there),
    ("ROSS answers every analysis in the catalogue", _every_analysis_answers),
)


def run():
    print("ross-interface --selftest")
    print("python %s on %s" % (sys.version.split()[0], sys.platform))
    print("frozen: %s" % bool(getattr(sys, "frozen", False)))
    print("")

    for description, check in CHECKS:
        try:
            complaint = check()
        except Exception as error:  # the traceback is the answer here
            import traceback

            complaint = "%s: %s\n%s" % (
                type(error).__name__,
                error,
                traceback.format_exc(),
            )
        if complaint:
            print("  FAILED  %s" % description)
            print("        %s" % complaint)
            return 1
        print("  ok      %s" % description)

    print("")
    print("the executable works.")
    return 0
