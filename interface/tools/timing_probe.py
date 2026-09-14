# -*- coding: utf-8 -*-
"""How long each analysis really takes, and where the time is.

BE-11 proposes a job queue with progress and cancellation. Before designing one
it is worth knowing whether there is a wait to manage: `/run_analysis` is
synchronous today, and the frontend's `AbortController` only stops waiting --
the server goes on computing to the end. Whether that matters is a question
about numbers nobody has measured. Leonardo's answer when asked was "it varies a
lot by analysis", which is the reason this probe exists rather than a design.

WHY IT SPLITS THE REQUEST INTO STAGES. A queue can only remove the part it
owns. If an analysis spends four seconds in `compute`, a queue with progress and
cancellation is worth building; if it spends four seconds in `fig.to_json()`
turning a large result into a payload, the same queue changes nothing at all,
because that work happens after the answer exists.

WHY `compute` IS MEASURED TWICE. The first version of this probe reported one
compute time and a "warm" total, and the first run showed campbell at 12.4 s
cold and 1.5 s warm. That gap was not the cache -- it was the probe's own
defect, and the number it produced by accident turned out to be the important
one. **The first execution of an analysis in a process is not the same work as
the second**: numba compiles, and the compilation is paid once for the process,
not once per request. A design that reads 16 s and builds a queue would be
sizing itself against a cost that a warm-up at startup removes for free.

WHY THERE ARE TWO WARM COLUMNS AND NOT ONE. The second version of this probe
reused one `Rotor` object for the repetitions and reported `modes` at **3.7
microseconds**. Nothing compiles that away: ROSS memoises on the rotor
instance, so the repetitions were reading its memo. The route builds the rotor
again on every request (`prepared_rotor`), throws it away with the response,
and therefore never benefits from that memo -- so the number measured was one
the application cannot have.

`fresh` rebuilds the rotor for each repetition, which is what a second request
from the browser really costs. `same` keeps the object, which is what the same
request would cost if the application held the rotor between calls. The
difference between the two is not noise: it is the size of a rotor cache that
does not exist yet, measured before anyone argues about building one.

So `first` is the first execution in this process (compilation included),
`fresh` what every later request pays, `same` what it could pay, and `cached`
what a repeated identical request pays today.

WHY `--elements`. Leonardo's point that a bigger rotor costs more is obviously
true and, until now, unquantified: the self-test rotor has six elements and the
saved project measured the same, so neither says anything about the shape of the
curve. `--elements` repeats the template's shaft element N times and moves the
far bearing to the last node, which gives a family of rotors that differ in one
dimension only.

WHY `cached` IS SEPARATE, AND CHECKED. The analysis cache short-circuits
`compute` for a repeated spec; what is left -- rebuilding the rotor, drawing,
serialising -- is the floor no queue can remove. The probe **asserts that the
cache really answered**: the first version silently recomputed when the key did
not match, and reported the result as if it were a cache hit. An instrument the
test suite cannot reach (this one needs ROSS) has to check its own premises at
runtime, or it is free to be wrong in the direction nobody looks.

WHY IT ALSO REPORTS THE PAYLOAD SIZE. A wait can be CPU or it can be bytes on
the wire, and the two have opposite fixes. Nothing in the stage timings tells
them apart: a chart with 200 000 points serialises fast and still takes seconds
to reach the browser.

WHY THE ROTOR IS AN ARGUMENT. Timings on the six-element rotor of the self-test
would answer a question nobody asked. That rotor is the default only because it
is the one this repository already carries; `--project` points the probe at a
real saved project.

WHY IT NAMES NO ANALYSIS. It iterates `sorted(ANALYSES)` and asks the catalogue
for each one's default parameters, exactly as the self-test and the conversion
probe do. The first executable this project produced died on `KeyError: 'modal'`
because I had written an analysis name by hand.

    python tools/timing_probe.py
    python tools/timing_probe.py --project my_rotor.json --repeat 3
"""

import argparse
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from domain.analysis_catalog import ANALYSES, default_params  # noqa: E402
from domain.cache import ANALYSIS_CACHE, ELEMENT_CACHE, spec_key  # noqa: E402
from domain.rotor_builder import build_rotor_from_ui  # noqa: E402
from selftest import ROTOR  # noqa: E402
from services.analysis import get_runner  # noqa: E402

# What the route sends when no degree-of-freedom conversion is asked for. Spelled
# the same way here so the cache key is the one the application would form.
NO_CONVERSION = "none"

COLUMNS = ("rotor", "spec", "first", "fresh", "same", "plot", "json", "cached")


class CacheDidNotAnswer(Exception):
    """The premise of the `cached` measurement failed, so there is no number.

    Raised rather than timed, because a recomputation reported as a cache hit is
    worse than a missing column: it is a number that looks like an answer.
    """


def _timed(work):
    """Run `work`, return (seconds, value)."""
    started = time.perf_counter()
    value = work()
    return time.perf_counter() - started, value


def _measure(name, project, params, repeat):
    """Time one analysis end to end, separating the one-off from the recurring."""
    ANALYSIS_CACHE.clear()
    ELEMENT_CACHE.clear()
    runner = get_runner(name)

    seconds = {}
    seconds["rotor"], rotor = _timed(lambda: build_rotor_from_ui(project))
    seconds["spec"], spec = _timed(lambda: runner.spec(dict(params), rotor))
    seconds["first"], result = _timed(lambda: runner.compute(rotor, spec))

    # The same computation again, warm, in the two shapes that matter. The
    # fastest repetition is the estimate, not the average -- the slow ones carry
    # whatever else the machine was doing, and what is measured here is the work.
    #
    # `fresh` rebuilds the rotor exactly as the route does, so ROSS's own memo
    # starts cold, as it does in production. `same` reuses the object, which is
    # the only reason `modes` once measured 3.7 microseconds here.
    fresh, same = [], []
    for _ in range(max(1, repeat)):
        rebuilt = build_rotor_from_ui(project)
        respec = runner.spec(dict(params), rebuilt)
        elapsed, _ = _timed(lambda: runner.compute(rebuilt, respec))
        fresh.append(elapsed)

        elapsed, result = _timed(lambda: runner.compute(rotor, spec))
        same.append(elapsed)
    seconds["fresh"] = min(fresh)
    seconds["same"] = min(same)
    repetitions = {"fresh": fresh, "same": same}

    seconds["plot"], figure = _timed(lambda: runner.plot(result, dict(params), rotor))
    seconds["json"], payload = _timed(figure.to_json)

    # Now the cache path, as the route would take it -- and checked, not assumed.
    ANALYSIS_CACHE.put(spec_key(project, NO_CONVERSION, name, spec), result)
    started = time.perf_counter()
    rebuilt = build_rotor_from_ui(project)
    respec = runner.spec(dict(params), rebuilt)
    answer = ANALYSIS_CACHE.get(spec_key(project, NO_CONVERSION, name, respec))
    if answer is None:
        raise CacheDidNotAnswer(
            "the key formed from a second identical request did not match the "
            "one just stored: the spec is not stable across two builds of the "
            "same rotor, which would also mean the application never reuses a "
            "cached result for this analysis"
        )
    redrawn = runner.plot(answer, dict(params), rebuilt)
    redrawn.to_json()
    seconds["cached"] = time.perf_counter() - started

    return seconds, len(payload), repetitions


def _sized_like(template, elements):
    """The template rotor stretched to `elements` shaft elements.

    One dimension changes and nothing else: the same shaft element repeated, the
    disk at the middle node, the far bearing moved to the last one. It is not a
    realistic rotor and does not need to be -- what it measures is how the cost
    grows with the model, which is a question about the slope and not about any
    particular machine.
    """
    shafts = template.get("shafts") or []
    bearings = template.get("bearings") or []
    if not shafts or not bearings:
        raise ValueError("--elements needs a template with a shaft and a bearing")
    grown = dict(template)
    grown["shafts"] = [dict(shafts[0]) for _ in range(elements)]
    grown["disks"] = [
        dict(disk, n="%d" % (elements // 2)) for disk in template.get("disks") or []
    ]
    grown["bearings"] = [
        dict(bearings[0], n="0"),
        dict(bearings[-1], n="%d" % elements),
    ]
    return grown


def _project_from(path):
    if path is None:
        return ROTOR
    with io.open(path, encoding="utf-8") as handle:
        saved = json.load(handle)
    # A file saved by the interface may be the project itself or an envelope
    # around it. Accepting both here costs one line and saves the next person a
    # confusing KeyError.
    return saved.get("project", saved) if isinstance(saved, dict) else saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project", help="a rotor saved by the interface (JSON)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="warm repetitions of compute per analysis (default 2)",
    )
    parser.add_argument(
        "--only",
        help="measure a single analysis by its catalogue key (default: all of them)",
    )
    parser.add_argument(
        "--elements",
        help="shaft-element counts to sweep, comma separated (default: the rotor as it is)",
    )
    options = parser.parse_args(argv)

    template = _project_from(options.project)
    if options.elements:
        sizes = [int(piece) for piece in options.elements.split(",")]
        rotors = [("%d elements" % n, _sized_like(template, n)) for n in sizes]
    else:
        rotors = [("as saved", template)]
    names = sorted(ANALYSES)
    if options.only:
        if options.only not in ANALYSES:
            parser.error(
                "%r is not in the catalogue. Available: %s"
                % (options.only, ", ".join(names))
            )
        names = [options.only]

    print("ROSS INTERFACE TIMING PROBE")
    print("python %s on %s" % (sys.version.split()[0], sys.platform))
    print("rotor: %s" % (options.project or "the self-test rotor (6 elements)"))
    print("warm repetitions per analysis: %d" % options.repeat)
    print("")
    print("first  = first execution in this process (includes any compilation)")
    print("fresh  = warm, with the rotor rebuilt -- what a later request costs")
    print("same   = warm, reusing the rotor object -- what it could cost")
    print("cached = the whole request with the result already cached")
    print("")

    header = "%-18s %-12s %6s %6s %8s %8s %8s %6s %6s %7s %9s" % (
        ("analysis", "rotor size") + COLUMNS + ("payload",)
    )
    print(header)
    print("-" * len(header))

    rows = []
    for label, project in rotors:
        for name in names:
            params = default_params(name)
            try:
                seconds, payload, repetitions = _measure(
                    name, project, params, options.repeat
                )
            except Exception as error:
                print(
                    "%-18s %-12s FAILED  %s: %s"
                    % (name, label, type(error).__name__, error)
                )
                rows.append({"analysis": name, "size": label, "failed": "%s" % error})
                continue

            print(
                "%-18s %-12s %6.3f %6.3f %8.3f %8.3f %8.3f %6.3f %6.3f %7.3f %8.1fk"
                % (
                    (name, label)
                    + tuple(seconds[column] for column in COLUMNS)
                    + (payload / 1024.0,)
                )
            )
            row = {
                "analysis": name,
                "size": label,
                "payload": payload,
                "repetitions": repetitions,
            }
            row.update(seconds)
            rows.append(row)

    measured = [r for r in rows if "failed" not in r]
    print("")
    if measured:
        one_off = max(measured, key=lambda r: r["first"] - r["fresh"])
        print(
            "largest one-off cost: %s (%s), %.3f s paid on the first run only"
            % (
                one_off["analysis"],
                one_off["size"],
                one_off["first"] - one_off["fresh"],
            )
        )
        slowest = max(measured, key=lambda r: r["fresh"])
        print(
            "slowest per request: %s (%s) at %.3f s of compute (%.3f s cached)"
            % (
                slowest["analysis"],
                slowest["size"],
                slowest["fresh"],
                slowest["cached"],
            )
        )
        keeping = max(measured, key=lambda r: r["fresh"] - r["same"])
        print(
            "most to gain from keeping the rotor: %s (%s), %.3f s -> %.3f s"
            % (keeping["analysis"], keeping["size"], keeping["fresh"], keeping["same"])
        )
        heaviest = max(measured, key=lambda r: r["payload"])
        print(
            "largest payload: %s at %.1f kB"
            % (heaviest["analysis"], heaviest["payload"] / 1024.0)
        )
    print("")
    print("--- paste the block below back ---")
    print(json.dumps(rows, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
