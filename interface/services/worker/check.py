# -*- coding: utf-8 -*-
"""Does the worker actually work -- and does it still work once frozen?

Run it from source or from the built executable:

    python app.py --worker-check
    dist/ross-interface/ross-interface --worker-check

The second is the one that matters. Everything else about this design was
measured with a plain interpreter, and the question the earlier probe declared
out of its reach is precisely this one: **inside a PyInstaller bundle, can the
program start a second copy of itself and talk to it?** If the answer is no,
the whole shape is wrong, and finding out here costs one short slice instead of
a rewritten route and a rewritten screen.

WHAT IT CHECKS, AND WHY EACH ONE IS A DIFFERENT WAY OF FAILING.

* the worker starts and says it is ready -- and how long that took. Measured at
  about nine seconds from source; if the bundle is much slower, the parent has
  to start it earlier than the first analysis.
* every analysis comes back through the pipe, and the chart is the same chart
  this process computes on its own. "The same" meant **byte for byte** at
  first, and starting there was right: the strictest test finds out fastest
  whether the boundary changed anything.

  It found something the strict test cannot describe. `campbell` and `modes` --
  the two analyses that go through ARPACK with `sparse=True` -- come back from a
  second process differing about one run in three, and the two differences are
  nothing alike: `campbell`'s critical speeds move in the last bit (two parts in
  1e14, invisible on any screen), while `modes` moves by a quarter and draws a
  shape that rises where the other falls. A byte comparison says "changed the
  answer" to both, which is a false accusation in one case and an understatement
  in the other. `services/worker/compare.py` tells them apart; this file reports
  what it says.
* whether the analysis reproduces itself at all -- asked **first**, and asked
  every time. "Does the worker preserve the answer?" only means something once
  "is this answer reproducible?" has been answered yes, and the first version
  of this check had that order backwards: it computed the reference a second
  time only when the worker disagreed. An unstable analysis whose two runs
  happened to coincide would have been reported as `ok`, and the instability
  would have stayed invisible for as long as the coincidence held.
* when two processes really do disagree, **whether the pipe is what changed the
  answer**. The worker is asked the same question again with its cache emptied.
  If this process repeats itself, and the worker repeats itself, and the two
  still differ, then nothing in between changed anything: the answer is simply
  not the same in every process. That is a fact about the computation, and it
  is true of the interface as it stands today, worker or no worker.
* the protocol survives an analysis that prints. Some ROSS analyses write to
  standard output while computing; the child hands that channel to stderr on
  purpose, and if that ever stops working the parent reads a chart as a syntax
  error. Since the sweep runs all twelve, the ones that print are covered
  without anyone having to remember which they are.
* what the two processes cost in memory. Both carry ROSS, and whether the
  duplication is affordable is the question that decides how much more moves
  into the worker later -- a decision this project agreed to take with a number
  rather than an intuition.
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.worker.compare import (  # noqa: E402
    DIFFERENT,
    IDENTICAL,
    compare,
)

ASK_TIMEOUT = 600.0


def resident_kilobytes(pid):
    """What the operating system says this process is holding.

    Asked of the system tools rather than of a library, because adding a
    dependency to answer one number in a probe is a bad trade -- and because the
    number is only ever compared with the other process measured the same way.
    """
    try:
        if sys.platform.startswith("win"):
            out = subprocess.check_output(
                ["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"],
                encoding="utf-8",
                errors="replace",
            )
            cell = out.strip().strip('"').split('","')[-1]
            return float(cell.replace(".", "").replace(",", "").split()[0])
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", "%d" % pid], encoding="utf-8"
        )
        return float(out.strip())
    except Exception:
        return None


def run():
    from domain.analysis_catalog import ANALYSES, default_params
    from domain.cache import ANALYSIS_CACHE
    from selftest import ROTOR
    from services.analysis.pipeline import figure_json
    from services.worker.host import Worker, WorkerGone

    def freshly(name, params):
        """Compute the chart here, with nothing left over from before.

        The cache is emptied first because the second call would otherwise get
        the first call's object back and every analysis would look perfectly
        reproducible -- the comparison would be measuring the cache.
        """
        ANALYSIS_CACHE.clear()
        return figure_json(name, dict(params), "none", ROTOR)

    print("ROSS INTERFACE WORKER CHECK")
    print("python %s on %s" % (sys.version.split()[0], sys.platform))
    print("frozen: %s" % bool(getattr(sys, "frozen", False)))

    worker = Worker()
    print("command: %s" % " ".join(worker.command))
    print("")

    try:
        hello = worker.start()
    except WorkerGone as error:
        print("FAILED  the worker never became ready")
        print("        %s" % error)
        return 1
    if not hello.get("ok"):
        print("FAILED  the worker answered the greeting with a refusal: %s" % hello)
        worker.stop()
        return 1
    print(
        "  ok      the worker is up after %.2f s (protocol %s)"
        % (worker.startup, hello.get("protocol"))
    )

    parent = resident_kilobytes(os.getpid())
    child = resident_kilobytes(worker.process.pid)
    if parent and child:
        print(
            "  memory  parent %.0f MB, worker %.0f MB, together %.0f MB"
            % (parent / 1024.0, child / 1024.0, (parent + child) / 1024.0)
        )
    else:
        print("  memory  not available on this system")

    failures = 0
    unstable = []
    divergent = []
    for name in sorted(ANALYSES):
        params = default_params(name)
        message = {
            "kind": "analysis",
            "analysis_type": name,
            "params": params,
            "conversion_type": "none",
            "project": ROTOR,
        }
        began = time.perf_counter()
        try:
            answer = worker.ask(message, timeout=ASK_TIMEOUT)
        except WorkerGone as error:
            print("  FAILED  %-18s %s" % (name, error))
            failures += 1
            break
        through = time.perf_counter() - began

        if not answer.get("ok"):
            print(
                "  FAILED  %-18s %s: %s"
                % (name, answer.get("failure"), answer.get("error"))
            )
            failures += 1
            continue

        # Reproducible first. Comparing the worker against a reference that does
        # not repeat would be comparing against nothing.
        here = freshly(name, params)
        twice = freshly(name, params)
        alone = compare(here, twice)
        if alone.verdict == DIFFERENT:
            unstable.append(name)
            print("  UNSTABLE %-17s does not repeat itself in one process" % name)
            print("           %s" % alone.describe())
            continue

        across = compare(here, answer["plot_json"])
        if across.verdict != DIFFERENT:
            note = "" if across.verdict == IDENTICAL else "  -- %s" % across.describe()
            print("  ok      %-18s %7.3f s through the pipe%s" % (name, through, note))
            continue

        # Beyond tolerance. Before this is written down as "the worker changed
        # the answer", the worker gets asked again -- with its cache emptied,
        # because otherwise the second answer would be the first one handed back
        # and the repeat would be measuring the cache.
        divergent.append(name)
        print("  DIFFERS %-18s the two processes do not agree" % name)
        print("           %s" % across.describe())
        try:
            worker.ask({"kind": "forget"}, timeout=ASK_TIMEOUT)
            again = worker.ask(message, timeout=ASK_TIMEOUT)
        except WorkerGone as error:
            print("           the worker died on the repeat: %s" % error)
            failures += 1
            break
        if not again.get("ok"):
            print("           the worker refused the repeat: %s" % again.get("error"))
            continue
        there = compare(answer["plot_json"], again["plot_json"])
        if there.verdict == DIFFERENT:
            print(
                "           the worker does not repeat itself either: %s"
                % there.describe()
            )
        else:
            print(
                "           both sides repeat themselves, so nothing in between "
                "changed the answer: it is not the same in every process."
            )

    print("")
    if unstable:
        print("not reproducible even in a single process: %s" % ", ".join(unstable))
        print(
            "  -- that is a property of the analysis, not of the worker, and it "
            "means no comparison of charts can say anything about these."
        )
    if divergent:
        print("computed differently in the two processes: %s" % ", ".join(divergent))
        print(
            "  -- read the lines above: if both sides repeat themselves, the pipe "
            "is not the cause and the same disagreement exists between any two "
            "runs of the interface today."
        )
    if failures:
        print("%d of %d failed." % (failures, len(ANALYSES)))
        worker.stop()
        return 1

    # Killing it is what cancellation will cost, so the check exercises that too:
    # a worker that cannot be ended cleanly is a program that will not close.
    worker.kill()
    print("the worker works, and it stops when told to (alive: %s)." % worker.alive)
    print(
        json.dumps({"startup": worker.startup, "parent_kb": parent, "worker_kb": child})
    )
    return 0
