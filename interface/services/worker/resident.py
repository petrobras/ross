# -*- coding: utf-8 -*-
"""The application's one worker, and what happens when it is not there.

Slice 6a proved a second process can be started, talked to and killed --
including from inside a PyInstaller bundle. This is the piece that makes it
*useful*: the route stops computing and asks here instead.

Four decisions, and each one is a place where this could go wrong in silence.

## 1. One worker, started early, owned by the application

Importing ROSS costs about eight and a half seconds, measured on both machines
and in the bundle. Paying that at the first click would make the first analysis
mysteriously slower than every analysis after it -- the classic shape of a
performance complaint nobody can reproduce. So the worker is started **in the
background when the program starts**, while the user is still drawing a rotor.

It is started from `app.py`, not from `create_app()`, and that is deliberate:
the test suite builds applications by the dozen, and a `create_app()` that
spawned a three-hundred-megabyte process would make the suite unusable. A
resident that was never started simply has no worker, which is the case the next
decision is about.

## 2. No worker is not an error

If the worker never started -- a locked executable, an antivirus, a machine
where spawning fails -- the application must still work. It does: `handle()` is
a pure function of a message, and this process can call it exactly as the child
does. **The fallback is not a second implementation of anything**; it is the
same function, run in a different place, which is the whole reason `protocol.py`
was written without a process in it.

What a fallback must never be is invisible. A worker that quietly never runs
would leave `--worker-check` green while the product got none of the benefit, so
every answer says where it was computed (`computed_by` in the response), the
detour is counted, and the reason is kept for whoever asks.

## 3. One pipe, one conversation at a time

Flask serves on threads. Two requests writing to the same stdin and reading from
the same answer queue would swap each other's charts -- FE-05 of the audit, the
out-of-order response overwriting the right chart, reproduced on the server
side. A lock makes the conversation serial.

Serial is not a compromise here: the work is CPU-bound in one child, so two
overlapping analyses could not go faster anyway. What the lock costs is that a
long analysis makes the next request wait -- which is exactly what happens
today, in one process, and is the problem slice 6c exists to solve with a queue
and a screen that does not block.

## 4. A worker that dies takes only its own request down

`WorkerGone` means the child is gone or out of step. The request that hit it is
answered here, the corpse is dropped, and a replacement starts in the
background. The user gets their chart; the next analysis gets a worker again.

## 5. A worker that was killed on purpose must *not* take that detour

Slice 6c-2 gives the user a button that kills the worker mid-computation, which
is the only way to stop something already inside ROSS. Without this decision the
button would do the opposite of what it says: the pipe would go quiet, decision
4 would call it a death, and this process would helpfully recompute -- for the
same ten seconds -- the very analysis the user just asked to stop.

The two cases are told apart by a **generation number**, bumped by `interrupt()`
and read by `ask()` around the call. A `WorkerGone` under a generation that
changed is a kill, and it becomes `Interrupted` instead of a detour. A counter
and not a flag, because a flag has to be consumed by somebody and there is no
moment that is obviously the right one to consume it in.
"""

import builtins
import logging
import threading

from services.worker.host import Worker, WorkerGone

# The same logger the transport layer configures, reached by name rather than by
# import: this package must not know that HTTP exists. Without a line in the log,
# a worker that never starts is a program that is quietly slower than it should
# be, on somebody else's machine, with nothing to look at.
logger = logging.getLogger("ross_interface")

# Generous on purpose. There is no timeout at all today -- a slow analysis takes
# as long as it takes -- and a cap that cut off a big rotor's Campbell would be
# a regression dressed as robustness. What this number is for is the pipe
# falling silent altogether, and fifteen minutes is more than an order of
# magnitude past the slowest analysis ever measured here (thirteen seconds, the
# crack, with its numba compilation included).
ASK_TIMEOUT = 900.0

WORKER = "worker"
HERE = "this process"


class WorkerFailure(Exception):
    """A refusal from the worker whose exception class this process cannot name."""


class Interrupted(Exception):
    """The worker was killed on purpose while it was computing this.

    Not a failure of the request and not a reason to detour: somebody asked for
    exactly this to stop. Whoever catches it already knows -- `Jobs` marked the
    job cancelled before the kill -- so it carries no blame, only the fact.
    """


def raise_if_refused(answer):
    """Turn a refused answer back into the exception it was.

    The error contract must not change because the computation moved. Today a
    `ValueError` reaches the user as **400 with its message** -- that is the
    channel ROSS refuses a model through ("Add at least one Shaft!") and the
    channel this application's validators refuse a field through -- and anything
    else is a **500** naming the type. See `api/errors.py`.

    Across a pipe an exception is a name and a string, so the name is looked up
    among the built-in exceptions and raised again. Anything not found there --
    a ROSS exception, one of ours -- becomes `WorkerFailure` **carrying the
    original name in the message**, so the 500 still says what really happened
    instead of hiding it behind the transport.
    """
    if answer.get("ok"):
        return answer
    name = answer.get("failure") or "Exception"
    message = answer.get("error") or ""
    known = getattr(builtins, name, None)
    if isinstance(known, type) and issubclass(known, Exception):
        raise known(message)
    raise WorkerFailure("%s: %s" % (name, message))


class Resident(object):
    """The one worker the application talks to, and the way around it."""

    def __init__(self, build=None, here=None):
        self._build = build or Worker
        self._here = here
        self._lock = threading.Lock()
        self._worker = None
        self._wanted = False
        # Wanting a worker and being allowed to *start* one are two different
        # permissions, and conflating them cost this project a test that spawned
        # a real ROSS process on every run. See `use`.
        self._spawning = False
        # Bumped by `interrupt`, read by `ask`. See decision 5 at the top.
        self._generation = 0
        self.detours = 0
        self.trouble = None

    # --- the process ---------------------------------------------------------

    def start_in_background(self):
        """Ask for a worker, and do not wait for it.

        Returning immediately is the point: the browser opens two seconds after
        this, and the page has to be there.
        """
        self._wanted = True
        self._spawning = True
        threading.Thread(target=self._start, daemon=True).start()

    def use(self, worker):
        """Install a worker directly, without starting a process. Ever.

        This is how the suite drives every path through here -- a fake worker
        answers and the route is exercised for real, with no child anywhere.

        "Without starting a process" used to hold only until that worker died:
        the replacement rule would then start a **real** child, because wanting a
        worker and being allowed to start one were the same flag. A suite that
        installs a dying fake -- and one does, to prove the detour works -- was
        spawning a process that imported ROSS for nine seconds and died on a
        closed pipe, every run, invisibly. A resident starts a process only when
        somebody asked it to; that somebody is `start_in_background`.
        """
        with self._lock:
            self._worker = worker
            self._wanted = True

    def stop(self):
        """Let go of the worker. Called when the application closes.

        A worker left behind is three hundred megabytes the user cannot see and
        did not ask for.
        """
        with self._lock:
            worker, self._worker = self._worker, None
            self._wanted = False
            self._spawning = False
        if worker is not None:
            worker.stop()

    def interrupt(self):
        """Kill the worker now, and start another. Returns whether there was one.

        **It must not take the lock.** `ask` holds the lock for the whole
        computation, and that computation is precisely what the caller is asking
        to stop -- a polite `interrupt` would wait for it to finish and then
        announce that it had been interrupted.

        Nothing here needs the three steps to be atomic with each other. The
        generation is bumped *before* the kill, so by the time the pipe goes
        quiet `ask` can already see that this was deliberate; bumping it when
        there is no worker costs nothing and keeps the two orders identical.
        """
        worker = self._worker
        self._generation += 1
        if worker is None:
            return False
        try:
            worker.kill()
        except Exception as error:  # a corpse that will not lie down
            logger.warning("the worker did not die cleanly: %s", error)
        logger.info("the worker was killed on request; a replacement is starting")
        if self._spawning:
            threading.Thread(target=self._start, daemon=True).start()
        return True

    @property
    def working(self):
        return self._worker is not None

    def _start(self):
        try:
            worker = self._build()
            worker.start()
        except Exception as error:
            self._note("the worker did not start: %s" % error)
            return
        with self._lock:
            if self._wanted:
                self._worker = worker
                logger.info("worker ready in %.1f s", worker.startup or 0.0)
                return
        worker.stop()  # stopped while we were starting

    def _note(self, trouble):
        self.trouble = trouble
        logger.warning("%s -- the analysis was computed in this process", trouble)

    # --- the conversation ----------------------------------------------------

    def ask(self, message, timeout=ASK_TIMEOUT):
        """Answer one message, and say where the answer came from.

        Returns `(answer, where)`. Never raises `WorkerGone`: a worker that is
        gone is a detour, not a failure of the request. It does raise
        `Interrupted` when the worker was killed on purpose -- see decision 5.
        """
        with self._lock:
            worker = self._worker
            generation = self._generation
            if worker is not None:
                try:
                    return worker.ask(message, timeout=timeout), WORKER
                except WorkerGone as error:
                    # Only this corpse. A replacement cannot be installed while
                    # the lock is held here, but the guard costs one comparison
                    # and the alternative is throwing away a live worker.
                    if self._worker is worker:
                        self._worker = None
                    if self._generation != generation:
                        # `interrupt` is already starting the replacement.
                        raise Interrupted(
                            "the worker was killed while it was computing this"
                        )
                    self._note("the worker was lost: %s" % error)
                    if self._spawning:
                        threading.Thread(target=self._start, daemon=True).start()
        self.detours += 1
        return self.answer_here(message), HERE

    def answer_here(self, message):
        """The same function the child runs, run here.

        Imported at the moment of use and not at the top of the file, so that
        this module -- and everything that tests it -- stays free of ROSS.
        """
        if self._here is not None:
            return self._here(message)
        from services.worker.protocol import handle

        return handle(message)


RESIDENT = Resident()
