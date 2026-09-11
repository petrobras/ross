# -*- coding: utf-8 -*-
"""Can a ROSS result cross a process boundary, and what does that cost?

Real cancellation needs a separate process. A thread cannot interrupt a running
numpy/numba call, so cancelling inside one only stops the waiting while the CPU
goes on burning -- which is the defect BE-11 exists to fix, with machinery
around it. A process can be killed.

That makes the whole design depend on a question nobody has answered: **does the
result survive the trip back?** This probe answers it instead of assuming, and
measures the three things that would decide the shape of the worker:

WHAT IT MEASURES, AND WHY EACH ONE CHANGES THE DESIGN.

* `pickle` -- whether the result can be serialised at all. ROSS results carry
  numpy arrays, references to the rotor and, in some cases, plotly figures. If
  one of the twelve cannot be pickled, the worker cannot return results and has
  to return the finished figure instead, which breaks the Campbell click: that
  route reads the **object** from the cache to pick a mode shape out of it, not
  the JSON.
* the size in kilobytes -- a result that pickles into 50 MB is technically
  transferable and practically a second problem. The chart payload is at most
  139 kB; if the pickle is two orders of magnitude larger, the queue trades a
  wait for a copy.
* the figure after the round trip -- a result that pickles but comes back
  subtly different is worse than one that refuses, because nothing would say so.
  The comparison is on the figure JSON, which is what the user actually sees.
* the child process -- whether it works at all, and what a **cold** one costs.
  On Windows `multiprocessing` spawns rather than forks, so the child imports
  ROSS from scratch. If that costs seconds, the worker has to be persistent, and
  a persistent worker is also where the numba compilation would be paid once.

WHAT IT DOES NOT ANSWER. How `multiprocessing` behaves inside the PyInstaller
bundle. That can only be measured in the built executable, and it is the next
step if these numbers come out favourable.

WHY IT NAMES NO ANALYSIS. It iterates `sorted(ANALYSES)` and asks the catalogue
for each one's default parameters, like the self-test and the other probes.

    python tools/boundary_probe.py
    python tools/boundary_probe.py --elements 20
"""

import argparse
import json
import multiprocessing
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from domain.analysis_catalog import ANALYSES, default_params  # noqa: E402
from domain.rotor_builder import build_rotor_from_ui  # noqa: E402
from selftest import ROTOR  # noqa: E402
from services.analysis import get_runner  # noqa: E402
from timing_probe import _sized_like  # noqa: E402


def _compute_in_child(name, project, params, sink):
    """Run one analysis and send the result back. Must be importable by the child.

    On Windows the child is spawned, not forked: it re-imports this module, so a
    closure or a lambda could not be the target. That is also why the import of
    ROSS happens again there, which is the cost the parent measures.

    It pickles the result **itself** and sends bytes. Handing the object to the
    queue would look simpler and would measure nothing: a `Queue` serialises on a
    background feeder thread, so a failure there is raised in that thread, not at
    `put()`, and the parent would sit waiting for a result that is never coming
    until the timeout -- reporting "the child did not answer" for what is really
    "this result does not pickle". Two different findings with one appearance is
    exactly what a probe must not produce.
    """
    started = time.perf_counter()
    rotor = build_rotor_from_ui(project)
    runner = get_runner(name)
    spec = runner.spec(dict(params), rotor)
    result = runner.compute(rotor, spec)
    inside = time.perf_counter() - started
    try:
        blob = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        sink.put((False, inside, "%s: %s" % (type(error).__name__, error)))
        return
    sink.put((True, inside, blob))


def _figure_json(name, result, params, rotor):
    return get_runner(name).plot(result, dict(params), rotor).to_json()


def _in_process(name, project, params):
    """Pickle the result here, and see whether the figure survives the round trip."""
    rotor = build_rotor_from_ui(project)
    runner = get_runner(name)
    spec = runner.spec(dict(params), rotor)
    result = runner.compute(rotor, spec)
    before = _figure_json(name, result, params, rotor)

    started = time.perf_counter()
    try:
        blob = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        return before, {
            "pickles": False,
            "why": "%s: %s" % (type(error).__name__, error),
        }
    dumped = time.perf_counter() - started

    started = time.perf_counter()
    restored = pickle.loads(blob)
    loaded = time.perf_counter() - started

    try:
        after = _figure_json(name, restored, params, rotor)
        identical = after == before
        drawing = None
    except Exception as error:
        identical = False
        drawing = "%s: %s" % (type(error).__name__, error)

    return before, {
        "pickles": True,
        "kilobytes": len(blob) / 1024.0,
        "dumps": dumped,
        "loads": loaded,
        "identical": identical,
        "drawing": drawing,
    }


def _through_a_child(name, project, params, timeout, reference):
    """The same computation in a spawned process, with the result sent back.

    `reference` is the figure JSON produced in this process. The result that
    comes back is redrawn and compared against it, because "the bytes arrived"
    and "the chart is the same chart" are different claims and only the second
    one is what a worker has to deliver.
    """
    sink = multiprocessing.Queue()
    child = multiprocessing.Process(
        target=_compute_in_child, args=(name, project, params, sink)
    )
    started = time.perf_counter()
    child.start()
    try:
        arrived, inside, payload = sink.get(timeout=timeout)
    except Exception as error:
        child.terminate()
        child.join()
        return {"crossed": False, "why": "%s: %s" % (type(error).__name__, error)}
    total = time.perf_counter() - started
    child.join()

    if not arrived:
        return {"crossed": False, "why": payload}

    try:
        rotor = build_rotor_from_ui(project)
        same_chart = (
            _figure_json(name, pickle.loads(payload), params, rotor) == reference
        )
    except Exception as error:
        return {
            "crossed": False,
            "why": "came back but would not redraw: %s: %s"
            % (type(error).__name__, error),
        }
    return {
        "crossed": True,
        "total": total,
        "inside": inside,
        "overhead": total - inside,
        "same_chart": same_chart,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--elements", type=int, help="stretch the rotor to N elements")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds to wait for a child before giving up (default 300)",
    )
    parser.add_argument("--only", help="a single analysis, by its catalogue key")
    options = parser.parse_args(argv)

    project = _sized_like(ROTOR, options.elements) if options.elements else ROTOR
    names = sorted(ANALYSES)
    if options.only:
        if options.only not in ANALYSES:
            parser.error(
                "%r is not in the catalogue. Available: %s"
                % (options.only, ", ".join(names))
            )
        names = [options.only]

    print("ROSS INTERFACE BOUNDARY PROBE")
    print("python %s on %s" % (sys.version.split()[0], sys.platform))
    print("start method: %s" % multiprocessing.get_start_method())
    print(
        "rotor: %s"
        % (
            "%d elements" % options.elements
            if options.elements
            else "the self-test rotor"
        )
    )
    print("")

    header = "%-18s %8s %10s %8s %8s %10s %9s %9s %9s" % (
        "analysis",
        "pickles",
        "size kB",
        "dumps",
        "loads",
        "identical",
        "child",
        "inside",
        "overhead",
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in names:
        params = default_params(name)
        try:
            reference, here = _in_process(name, project, params)
        except Exception as error:
            print("%-18s FAILED  %s: %s" % (name, type(error).__name__, error))
            rows.append({"analysis": name, "failed": "%s" % error})
            continue

        there = _through_a_child(name, project, params, options.timeout, reference)
        row = {"analysis": name}
        row.update(here)
        row.update({"child_" + k: v for k, v in there.items()})
        rows.append(row)

        if not here["pickles"]:
            print("%-18s %8s  %s" % (name, "NO", here["why"][:70]))
            continue
        print(
            "%-18s %8s %10.1f %8.3f %8.3f %10s %9s %9s %9s"
            % (
                name,
                "yes",
                here["kilobytes"],
                here["dumps"],
                here["loads"],
                "yes" if here["identical"] else "NO",
                ("yes" if there.get("same_chart") else "differs")
                if there.get("crossed")
                else "NO",
                "%.3f" % there["inside"] if there.get("crossed") else "-",
                "%.3f" % there["overhead"] if there.get("crossed") else "-",
            )
        )
        if here.get("drawing"):
            print(
                "%-18s   redraw after the round trip: %s" % ("", here["drawing"][:70])
            )
        if not there.get("crossed"):
            print("%-18s   child: %s" % ("", there.get("why", "")[:70]))

    measured = [r for r in rows if "failed" not in r]
    print("")
    refused = [r["analysis"] for r in measured if not r.get("pickles")]
    changed = [
        r["analysis"] for r in measured if r.get("pickles") and not r.get("identical")
    ]
    blocked = [r["analysis"] for r in measured if not r.get("child_crossed")]
    differs = [
        r["analysis"]
        for r in measured
        if r.get("child_crossed") and not r.get("child_same_chart")
    ]
    print("do not pickle:            %s" % (", ".join(refused) or "none"))
    print("figure changed here:      %s" % (", ".join(changed) or "none"))
    print("did not cross:            %s" % (", ".join(blocked) or "none"))
    print("crossed but chart differs: %s" % (", ".join(differs) or "none"))
    crossed = [r for r in measured if r.get("child_crossed")]
    if crossed:
        worst = max(crossed, key=lambda r: r["child_overhead"])
        print(
            "worst child overhead: %s at %.3f s (spawn, import and transfer)"
            % (worst["analysis"], worst["child_overhead"])
        )
    print("")
    print("--- paste the block below back ---")
    print(json.dumps(rows, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
