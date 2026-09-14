# -*- coding: utf-8 -*-
"""The line of work, and the right to leave it.

Slice 6b moved the computation into another process. It did not change **when**
the answer arrives: the route still waits, holding the pipe, and everyone behind
it waits too. This is the piece that breaks that.

## The defect this exists to remove, and it is already there

`apiFetchLatest` in the browser aborts the previous request on a card when the
user changes a parameter and runs again. Aborting a `fetch` **does not stop the
computation**: the server thread carries on to the end, and -- since 6b -- it
holds the resident's lock while it does. So today, changing a card's parameters
twice makes the second analysis wait for the first, which nobody is waiting for,
with nothing on screen to say so. The cancellation is a client-side illusion.

A queue makes it real, and the first half costs almost nothing: a job that has
not started yet can simply be taken out of the line.

## What "cancel" means here, decided with Leonardo

* **Superseded** -- a new job arrives for the same subject (the same card). If
  the old one is still waiting, it never runs. If it has already started, it
  runs to the end and its answer is thrown away: stopping a computation that is
  already inside ROSS means killing the process, and that has a price (the next
  analysis pays the eight to twelve seconds of importing ROSS again, plus up to
  sixteen of numba on the campbell and the crack, all measured).
* **Cancelled** -- slice 6c-2's button, the other half. The user asked for
  everything to stop and was told the price first. The line is emptied and the
  worker is killed, which is the only way to stop a computation that is already
  inside ROSS. The job that was running keeps the ending it was given: whatever
  the dying worker was about to say arrives too late to change it.

## Why one thread and not a pool

The work is CPU inside a single child process. Two jobs at once could not go
faster; they could only take turns worse. What the queue buys is not
parallelism, it is **order and the right to leave** -- knowing what is waiting,
what is running, and being able to drop what nobody wants.

## Nothing here knows about ROSS, or about HTTP

`Jobs` is built with the function that answers a message. The application passes
the resident; the suite passes a dictionary-returning stub and drives every
state -- queued, running, done, failed, superseded -- with no process and no
library anywhere.
"""

import itertools
import queue
import threading
import time

from services.worker.resident import RESIDENT

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SUPERSEDED = "superseded"
CANCELLED = "cancelled"

FINISHED = frozenset({DONE, FAILED, SUPERSEDED, CANCELLED})

# How many finished jobs stay readable. The browser asks once more after a job
# finishes, and that is all anyone needs; the bound is what keeps a long session
# from growing a dictionary of every chart it ever drew.
REMEMBERED = 60


class Job(object):
    """One unit of work, and everything the screen may want to know about it."""

    def __init__(self, number, message, key):
        # The number is the order it was asked in, and it is what decides who is
        # ahead of whom. Timestamps would be the obvious choice and are the wrong
        # one: two jobs submitted in the same millisecond would each be able to
        # claim it was first, and the screen would count the line differently
        # depending on which card asked.
        self.number = number
        self.id = "j%d" % number
        self.message = message
        self.key = key
        self.status = QUEUED
        self.answer = None
        self.where = None
        self.failure = None
        self.submitted = time.time()
        self.started = None
        self.finished = None

    @property
    def waiting(self):
        """Seconds since it was asked for -- what the screen counts up."""
        return (self.finished or time.time()) - self.submitted

    def report(self):
        report = {"job_id": self.id, "state": self.status, "waiting": self.waiting}
        if self.status == RUNNING and self.started is not None:
            report["running"] = (self.finished or time.time()) - self.started
        return report


class Jobs(object):
    """The line, the single worker thread, and what is worth remembering."""

    def __init__(self, ask=None, interrupt=None, remembered=REMEMBERED):
        self._ask = ask
        # How to stop what is already running. Injected for the same reason as
        # `ask`: this file must not know that a subprocess is what is on the
        # other end, and the suite must be able to drive the whole cancellation
        # with a function that only records that it was called.
        self._interrupt = interrupt
        self._waiting = queue.Queue()
        self._known = {}
        self._order = []
        self._remembered = remembered
        self._lock = threading.Lock()
        self._numbers = itertools.count(1)
        self._thread = None

    # --- what the routes call ------------------------------------------------

    def submit(self, message, key=None):
        """Put a job in the line and return it, without waiting for anything.

        A `key` is the subject the job belongs to -- one card on the screen.
        Submitting a second job for the same subject takes the first out of the
        line if it has not started. That is the honest version of what the
        browser has been pretending to do since Phase 2.
        """
        job = Job(next(self._numbers), message, key)
        with self._lock:
            if key is not None:
                for other in self._known.values():
                    if other.key == key and other.status == QUEUED:
                        other.status = SUPERSEDED
                        other.finished = time.time()
            self._known[job.id] = job
            self._order.append(job.id)
            self._forget_the_oldest()
        self._waiting.put(job.id)
        self._make_sure_something_is_working()
        return job

    def look(self, job_id):
        with self._lock:
            return self._known.get(job_id)

    def waiting_now(self):
        """How many are in the line, for whoever wants to show it."""
        with self._lock:
            return sum(1 for job in self._known.values() if job.status == QUEUED)

    def report(self, job):
        """What the job says about itself, plus what only the line can say.

        `ahead` is the measurement slice 6c-2 was built around. On screen,
        *computing* and *waiting for another card* looked identical -- both cards
        said "updating" -- and Leonardo read two analyses taking turns as two
        analyses running at once. They are not distinguishable by anything the
        job knows about itself: a job is queued or running, and how long that
        will last is a fact about everybody else.
        """
        report = job.report()
        if job.status == QUEUED:
            report["ahead"] = self._ahead_of(job)
        return report

    def _ahead_of(self, job):
        """How many have to end before this one starts.

        The one that is running counts: from the card's point of view there is no
        difference between waiting behind a queued job and waiting behind the
        computation itself.
        """
        with self._lock:
            return sum(
                1
                for other in self._known.values()
                if other.status == RUNNING
                or (other.status == QUEUED and other.number < job.number)
            )

    def stop_everything(self):
        """Empty the line and kill what is running. The button the user pays for.

        Everything, and not just one card, because that is what killing a worker
        actually does -- the computations of every card live in the same process.
        A per-card button would have been a smaller promise than the mechanism
        can keep, which is the kind of interface that teaches people not to
        believe the next button.

        The order matters. Jobs are marked **before** the worker is killed, so
        that the answer the dying worker may still get out arrives to a job that
        has already ended and is ignored -- see `_finish`. Marking afterwards
        would leave a window in which a cancelled analysis draws its chart.
        """
        cancelled = 0
        was_running = False
        with self._lock:
            for job in self._known.values():
                if job.status == RUNNING:
                    was_running = True
                elif job.status != QUEUED:
                    continue
                job.status = CANCELLED
                job.finished = time.time()
                cancelled += 1
        killed = bool(self._interrupt()) if self._interrupt is not None else False
        return {
            "cancelled": cancelled,
            "was_running": was_running,
            "worker_killed": killed,
        }

    # --- the single thread ---------------------------------------------------

    def _make_sure_something_is_working(self):
        """Start the one thread, once.

        It never ends, and that is deliberate. A thread that stopped when the
        line went quiet would need `submit` to decide whether to start another
        one -- and between `is_alive()` saying yes and the thread actually
        returning there is a window where nobody starts anything and the job
        waits forever. A blocked thread costs nothing; that window costs an
        analysis that never answers, once in a long while, unreproducibly.
        """
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._work, daemon=True)
            self._thread.start()

    def _work(self):
        while True:
            job = self.look(self._waiting.get())
            if job is None or job.status != QUEUED:
                continue  # superseded while it waited: it never runs
            self._answer(job)

    def _answer(self, job):
        # Taking it out of the line and starting it is one decision, so it is one
        # locked step: between a `status` read and a `status` write there is room
        # for a cancellation to be lost, and a lost cancellation is an analysis
        # the user stopped appearing anyway, half a minute later.
        with self._lock:
            if job.status != QUEUED:
                return
            job.status = RUNNING
            job.started = time.time()

        try:
            answer, where = self._ask(job.message)
        except Exception as error:
            # The resident does not raise for a lost worker -- that is a detour,
            # not a failure. What does reach here is `Interrupted` (deliberate,
            # and the job is already cancelled, so `_finish` will ignore this)
            # and the unforeseen, which a job that swallowed it would turn into a
            # screen waiting forever.
            self._finish(
                job,
                FAILED,
                failure={"failure": type(error).__name__, "error": "%s" % error},
            )
        else:
            if answer.get("ok"):
                self._finish(job, DONE, answer=answer, where=where)
            else:
                self._finish(
                    job,
                    FAILED,
                    answer=answer,
                    where=where,
                    failure={
                        "failure": answer.get("failure"),
                        "error": answer.get("error"),
                    },
                )
        # Pruned here as well as on submit: a job only becomes forgettable when
        # it finishes, and pruning only on the way in would leave the tail of a
        # session -- the jobs nobody submits after -- kept forever.
        with self._lock:
            self._forget_the_oldest()

    def _finish(self, job, status, answer=None, where=None, failure=None):
        """Give the job its ending -- unless it already has one.

        A cancelled job may still be being computed: by a worker that has not
        died yet, or by this process, which cannot be stopped at all. What comes
        back then is the answer to a question nobody is asking any more, and
        writing it here would put the chart on a card the user has cancelled.
        """
        with self._lock:
            if job.status != RUNNING:
                return
            job.answer, job.where, job.failure = answer, where, failure
            job.status = status
            job.finished = time.time()

    # --- housekeeping --------------------------------------------------------

    def _forget_the_oldest(self):
        """Drop finished jobs once there are more than anyone will ask about.

        Only finished ones: a job still queued or running is dropped by nobody,
        because the screen is waiting for exactly that answer.
        """
        while len(self._order) > self._remembered:
            for position, job_id in enumerate(self._order):
                if self._known[job_id].status in FINISHED:
                    del self._known[job_id]
                    self._order.pop(position)
                    break
            else:
                return  # nothing finished to drop; the line is simply long


# The application's one queue. Built with the resident's `ask` and nothing else:
# everything about where a computation happens was already decided there, and
# this file only decides when.
JOBS = Jobs(ask=RESIDENT.ask, interrupt=RESIDENT.interrupt)
