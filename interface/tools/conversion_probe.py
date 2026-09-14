# -*- coding: utf-8 -*-
"""Measure which analyses work on a rotor converted to 4 DoF and to torsional.

Run this in the venv where ROSS is installed, from the project root:

    python tools/conversion_probe.py                 # what the interface does today
    python tools/conversion_probe.py --measure-all   # ignore the table and measure ROSS

It lives in `tools/` and not in `tests/` because it is not a test: it asserts
nothing, it does not fail, and it depends on the real ROSS. It is a measuring
instrument, and what it measures becomes `domain/compatibility.py` -- which is
guarded by tests.

**The difference matters.** Once the compatibility table was in place, the route
refuses the known combinations -- so a normal run gives back our own refusal
messages, and confirming the table with it would be circular. To measure ROSS
again (when upgrading the library, for instance), use `--measure-all`: it
empties the table for this run only and lets every combination reach the
library.

It changes nothing -- it only runs the 36 combinations (12 analyses x 3 models)
through the same route the interface uses, with the default values of each
form, and prints the result. Paste the output into the conversation.

Why measure instead of deduce: ROSS itself warns, in the docstring of
`convert_6dof_to_torsional`, that "some Rotor class methods, such as `run_ucs`,
and `run_unbalance_response`, may not work correctly" -- with a "such as" that
leaves the list open. And there is a second risk no error reveals: the
conversion swaps the rotor's matrix methods for reduced versions, so an analysis
that builds its own matrices would use the 6 DoF ones **with no error at all**.
That is why the probe also compares each conversion's figure with the 6 DoF one:
an identical figure means the conversion had no effect.
"""

import hashlib
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# These imports come after `sys.path.insert` above -- that is what makes them
# resolvable when the probe is run as a script from the project root.
from api import create_app  # noqa: E402
from api.security import SESSION_TOKEN  # noqa: E402
from api.waiting import answer_for  # noqa: E402
from domain.analysis_catalog import ANALYSES, default_params  # noqa: E402
from domain import compatibility  # noqa: E402

PROJECT_REQUEST = {
    "name": "Probe",
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

MODEL_NAMES = [("", "6 DoF"), ("4dof", "4 DoF"), ("torsional", "Torsional")]


def fingerprint(plot_json):
    """A fingerprint of the figure's data, to compare across models."""
    try:
        figure = json.loads(plot_json)
    except Exception:
        return None
    payload = json.dumps(figure.get("data", []), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def main():
    measure_all = "--measure-all" in sys.argv
    if measure_all:
        compatibility.UNSUPPORTED.clear()

    client = create_app().test_client()
    header = {"X-ROSS-Token": SESSION_TOKEN}
    lines = []

    for analysis in sorted(ANALYSES):
        result = {}
        for conversion, label in MODEL_NAMES:
            body = {
                "analysis_type": analysis,
                "params": default_params(analysis),
                "conversion_type": conversion,
                "project": PROJECT_REQUEST,
            }
            try:
                # The route answers a job since slice 6c; `api/waiting.py` is the
                # one place that knows how to wait for one.
                response = answer_for(
                    client,
                    client.post("/run_analysis", json=body, headers=header),
                    header,
                )
                payload = (response.get_json() if response is not None else None) or {}
                if payload.get("status") == "success":
                    result[conversion] = (
                        "ok",
                        fingerprint(payload.get("plot_json", "")),
                    )
                else:
                    result[conversion] = ("error", (payload.get("message") or "")[:110])
            except Exception:
                result[conversion] = (
                    "error",
                    traceback.format_exc().strip().split("\n")[-1][:110],
                )

        base = result[""][1] if result[""][0] == "ok" else None
        for conversion, label in MODEL_NAMES[1:]:
            state, detail = result[conversion]
            if state == "error":
                verdict = "ERROR"
            elif base is not None and detail == base:
                verdict = "NO EFFECT"  # same figure as the 6 DoF one
            else:
                verdict = "ok"
            lines.append((analysis, label, verdict, detail if state == "error" else ""))
        if result[""][0] == "error":
            lines.append((analysis, "6 DoF", "ERROR", result[""][1]))

    print("=" * 78)
    print("CONVERSION PROBE -- paste everything below into the conversation")
    print(
        "mode: %s"
        % (
            "measuring ROSS (table ignored)"
            if measure_all
            else "what the interface does today (table active)"
        )
    )
    print("=" * 78)
    try:
        import ross as rs

        print("ross %s" % getattr(rs, "__version__", "?"))
    except Exception as error:
        print("ross unavailable: %s" % error)
    print()
    print("%-18s %-11s %-11s %s" % ("analysis", "model", "verdict", "message"))
    print("-" * 78)
    for analysis, label, verdict, detail in lines:
        print("%-18s %-11s %-11s %s" % (analysis, label, verdict, detail))
    print("-" * 78)
    print("%d combinations with a problem" % sum(1 for row in lines if row[2] != "ok"))


if __name__ == "__main__":
    main()
