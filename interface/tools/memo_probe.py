# -*- coding: utf-8 -*-
"""Does ROSS's memo survive the parameter changes a user actually makes?

THE QUESTION, AND WHY IT COMES BEFORE ANY CODE. The phase-4 measurement found
that keeping the `Rotor` object between requests would take the campbell from
6.64 s to 0.064 s, the freq_response to 0.28 ms and the modes to 24 us -- and do
nothing at all for the other nine analyses. It also wrote down, in the same
paragraph, the thing nobody had measured:

    the rotor cache only adds value when the **spec** changes and the rotor does
    not -- the user drags the speed slider and recomputes. Whether ROSS's
    memoisation survives that depends on which parameter moved: changing the
    campbell's speed range changes the speeds and the memo misses; changing how
    many curves to show does not.

That sentence is the whole of slice 6d. If the memo dies on the parameters people
actually move, the rotor cache is machinery that speeds up the case the analysis
cache already covers -- and the right outcome is to not build it. Finding that
out costs this probe; finding it out afterwards costs a slice.

HOW IT ASKS. Two passes, and the first decides whether the second is worth
running.

1. **Is there a memo at all?** For each analysis: compute once on a rotor
   (paying numba), again on the *same* rotor, and again on a *rebuilt* one.
   If keeping the object buys nothing, there is nothing for a parameter change
   to spoil, and the analysis is reported and skipped.
2. **Does it survive a change?** For each parameter, the user's actual
   sequence: build a rotor, run it once with the defaults -- which is what warms
   the memo -- then change **one** parameter and run again on that same rotor.
   Fast means the memo was still good; slow means the change threw it away.

WHAT IT DOES NOT DO. It does not guess which parameters matter. It nudges every
one the catalogue offers, by type, and reports the ones it could not vary
honestly instead of inventing a value for them. It writes no analysis name: the
list is `sorted(ANALYSES)`, for the reason the first executable of this project
died of -- `KeyError: 'modal'`, a name guessed from a file name.

    python tools/memo_probe.py
    python tools/memo_probe.py --only <analysis>
"""

import argparse
import copy
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from domain.analysis_catalog import ANALYSES, default_params  # noqa: E402
from domain.cache import ANALYSIS_CACHE, ELEMENT_CACHE  # noqa: E402
from domain.rotor_builder import build_rotor_from_ui  # noqa: E402
from selftest import ROTOR  # noqa: E402
from services.analysis import get_runner  # noqa: E402

# Below this, keeping the rotor bought nothing worth a parameter sweep: the
# measurement that motivated this probe separated its three winners from the
# other nine by three orders of magnitude, so a factor of two is a generous
# floor and not a fine judgement.
WORTH_KEEPING = 2.0

# And an absolute floor as well: a computation that takes a millisecond is not
# made better by a cache, however good the ratio looks.
WORTH_MEASURING = 0.01


def timed(work):
    started = time.perf_counter()
    value = work()
    return time.perf_counter() - started, value


def nudged(value):
    """A different, still valid value of the same kind -- or None.

    None is an answer: a parameter this cannot vary honestly is reported as not
    measured, which is worth more than a number produced from a guess about what
    the field means.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.1 if value else 0.1
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, bool) or not isinstance(last, (int, float)):
            return None
        return value[:-1] + [last * 1.1 if isinstance(last, float) else last + 1]
    return None


def is_there_a_memo(runner, params):
    """Pass one: what keeping the rotor buys, before anything is changed.

    Returns (first, same, fresh). `first` carries numba's compilation, which is
    why neither of the other two is compared against it.
    """
    rotor = build_rotor_from_ui(ROTOR)
    spec = runner.spec(copy.deepcopy(params), rotor)
    first, _ = timed(lambda: runner.compute(rotor, spec))

    same, _ = timed(lambda: runner.compute(rotor, spec))

    rebuilt = build_rotor_from_ui(ROTOR)
    respec = runner.spec(copy.deepcopy(params), rebuilt)
    fresh, _ = timed(lambda: runner.compute(rebuilt, respec))

    return first, same, fresh


def survives(runner, params, field, fresh):
    """Pass two: the user ran it once, then moved one control.

    The rotor is built and warmed with the defaults here rather than reused from
    the caller, so that no earlier parameter's measurement can have warmed
    anything this one is about to ask for.
    """
    changed = copy.deepcopy(params)
    replacement = nudged(changed.get(field))
    if replacement is None:
        return None, "not varied (%s)" % type(changed.get(field)).__name__

    rotor = build_rotor_from_ui(ROTOR)
    runner.compute(rotor, runner.spec(copy.deepcopy(params), rotor))

    changed[field] = replacement
    try:
        spec = runner.spec(changed, rotor)
        seconds, _ = timed(lambda: runner.compute(rotor, spec))
    except Exception as error:
        return None, "refused: %s" % ("%s: %s" % (type(error).__name__, error))[:60]

    # Where it landed between "the memo answered" (near zero) and "everything was
    # recomputed" (near `fresh`).
    return seconds, "kept" if seconds < fresh / WORTH_KEEPING else "lost"


def run(only=None):
    names = [only] if only else sorted(ANALYSES)
    if only and only not in ANALYSES:
        print("no analysis called %r. The catalogue has: %s" % (only, sorted(ANALYSES)))
        return 1

    print("ROTOR CACHE PROBE -- does ROSS's memo survive a parameter change?")
    print("python %s on %s" % (sys.version.split()[0], sys.platform))
    print("")

    for name in names:
        ANALYSIS_CACHE.clear()
        ELEMENT_CACHE.clear()
        params = default_params(name)
        runner = get_runner(name)

        try:
            first, same, fresh = is_there_a_memo(runner, params)
        except Exception:
            print("%s" % name)
            print("    could not be measured:")
            print("        %s" % traceback.format_exc().strip().split("\n")[-1][:120])
            print("")
            continue

        gain = (fresh / same) if same else float("inf")
        print(
            "%-18s first %7.3f s   rebuilt %7.3f s   kept %7.3f s   x%.0f"
            % (name, first, fresh, same, gain)
        )

        if fresh < WORTH_MEASURING or gain < WORTH_KEEPING:
            print("    keeping the rotor buys nothing here -- no parameters tried")
            print("")
            continue

        for field in sorted(params):
            seconds, verdict = survives(runner, params, field, fresh)
            if seconds is None:
                print("    %-22s --          %s" % (field, verdict))
            else:
                print("    %-22s %7.3f s   %s" % (field, seconds, verdict))
        print("")

    print("Reading it: `kept` means the memo answered even though the parameter")
    print("moved, so a rotor cache would pay for that control. `lost` means the")
    print("change threw the memo away and the cache would do nothing for it.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", help="measure a single analysis by catalogue name")
    sys.exit(run(parser.parse_args().only))
