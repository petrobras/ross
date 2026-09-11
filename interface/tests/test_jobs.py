# -*- coding: utf-8 -*-
"""The line of work: order, the right to leave it, and what is remembered.

Every state a job can be in is driven here with **no process and no ROSS**:
`Jobs` is built with the function that answers a message, and the stub passed in
decides when each answer arrives. That is what makes "the second job jumped the
queue" or "a superseded job ran anyway" a failing test instead of a story about
a machine somebody else has.
"""

import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services.worker.jobs import (  # noqa: E402
    CANCELLED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    SUPERSEDED,
    Job,
    Jobs,
)

PATIENCE = 5.0


def _settles(job, states, patience=PATIENCE):
    """Wait for a job to reach one of these states. Returns whether it did.

    Polling and not an event, because the property under test is the job's own
    state and an event would be a second source of truth about it.
    """
    until = time.time() + patience
    while time.time() < until:
        if job.status in states:
            return True
        time.sleep(0.005)
    return False


class _Answers(object):
    """A stand-in for the resident: it answers when the test lets it.

    `hold()` makes the next answer wait on a latch, which is how a job is kept
    in `running` for as long as a test needs to look at what is behind it.
    """

    def __init__(self, answer=None):
        self.answer = answer or {"ok": True, "plot_json": "{}"}
        self.asked = []
        self.latch = None
        self.raises = None

    def hold(self):
        self.latch = threading.Event()
        return self.latch

    def __call__(self, message):
        self.asked.append(message)
        if self.latch is not None:
            self.latch.wait(PATIENCE)
        if self.raises is not None:
            raise self.raises
        return dict(self.answer), "worker"


def test_a_job_is_answered_and_carries_where_it_was_computed():
    answers = _Answers()
    jobs = Jobs(ask=answers)

    job = jobs.submit({"kind": "analysis"})
    assert _settles(job, {DONE}), job.status
    assert job.answer["plot_json"] == "{}"
    assert job.where == "worker"
    assert job.report()["state"] == DONE


def test_the_route_gets_a_name_back_before_anything_is_computed():
    """The whole point of the slice: the answer is a job, not a wait."""
    answers = _Answers()
    answers.hold()
    jobs = Jobs(ask=answers)

    job = jobs.submit({"kind": "analysis"})
    assert job.id, "a job with no name cannot be asked about"
    assert job.status in (QUEUED, RUNNING)
    assert jobs.look(job.id) is job


def test_a_job_that_has_not_started_leaves_the_line_when_the_card_asks_again():
    """The defect this slice exists to remove.

    Until now the browser aborted its own `fetch` and the server carried on to
    the end, so a card run twice made the second analysis wait for the first --
    which nobody wanted any more, with nothing on screen to say so.
    """
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers)

    busy = jobs.submit({"kind": "analysis", "n": 0})  # takes the thread
    assert _settles(busy, {RUNNING})

    stale = jobs.submit({"kind": "analysis", "n": 1}, key="card:7")
    wanted = jobs.submit({"kind": "analysis", "n": 2}, key="card:7")
    assert stale.status == SUPERSEDED, "the stale job is still in the line"
    assert wanted.status == QUEUED

    latch.set()
    assert _settles(wanted, {DONE}), wanted.status
    assert [message["n"] for message in answers.asked] == [0, 2], (
        "the superseded job was computed anyway: %s" % answers.asked
    )


def test_a_job_already_running_is_left_to_finish():
    """Decided with Leonardo: stopping ROSS mid-computation means killing the
    process, and that costs the next analysis its startup. So a running job runs
    to the end and its answer is simply not the one the card is waiting for.
    """
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers)

    running = jobs.submit({"kind": "analysis", "n": 0}, key="card:7")
    assert _settles(running, {RUNNING})

    replacing = jobs.submit({"kind": "analysis", "n": 1}, key="card:7")
    assert running.status == RUNNING, "a running job must not be marked superseded"

    latch.set()
    assert _settles(replacing, {DONE}), replacing.status
    assert running.status == DONE
    assert [message["n"] for message in answers.asked] == [0, 1]


def test_another_card_is_not_taken_out_of_the_line():
    """Control: the key is what separates subjects, and it has to separate them.

    Without this, a supersede that ignored the key would look correct in every
    test above and would quietly cancel the rotor figure whenever an analysis
    was re-run.
    """
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers)

    jobs.submit({"kind": "analysis", "n": 0})
    other = jobs.submit({"kind": "analysis", "n": 1}, key="card:1")
    jobs.submit({"kind": "analysis", "n": 2}, key="card:2")
    jobs.submit({"kind": "analysis", "n": 3}, key="card:2")

    assert other.status == QUEUED, "a job of another card left the line"
    latch.set()
    assert _settles(other, {DONE}), other.status


def test_a_refusal_is_kept_as_what_it_was():
    """The error contract survives the wait.

    A `ValueError` from ROSS has to reach the user as a 400 with its message
    even though nobody was on the line when it happened, so the job keeps the
    name of the exception and not just the fact that something went wrong.
    """
    answers = _Answers(
        answer={"ok": False, "failure": "ValueError", "error": "no shaft"}
    )
    jobs = Jobs(ask=answers)

    job = jobs.submit({"kind": "analysis"})
    assert _settles(job, {FAILED}), job.status
    assert job.failure == {"failure": "ValueError", "error": "no shaft"}


def test_an_unforeseen_failure_ends_the_job_instead_of_the_thread():
    """The thread is the only one there is.

    If an exception escaped, the queue would stop answering and every card on
    screen would wait forever -- no error, no chart, nothing to look at. The
    resident does not raise, which is exactly why anything that reaches here is
    worth surviving.
    """
    answers = _Answers()
    answers.raises = RuntimeError("the sky fell")
    jobs = Jobs(ask=answers)

    job = jobs.submit({"kind": "analysis"})
    assert _settles(job, {FAILED}), job.status
    assert job.failure["failure"] == "RuntimeError"
    assert "sky fell" in job.failure["error"]

    answers.raises = None
    after = jobs.submit({"kind": "analysis"})
    assert _settles(after, {DONE}), (
        "the queue stopped working after one failure: %s" % after.status
    )


def test_finished_jobs_are_forgotten_and_unfinished_ones_are_not():
    """A session that draws a thousand charts must not remember a thousand.

    The bound only ever drops *finished* jobs: one still queued or running is
    the answer a card is waiting for, and forgetting it would leave that card
    polling a name nobody knows.
    """
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers, remembered=3)

    held = jobs.submit({"kind": "analysis", "n": -1})
    assert _settles(held, {RUNNING})

    waiting = [jobs.submit({"kind": "analysis", "n": n}) for n in range(6)]
    assert jobs.look(held.id) is held, "the running job was forgotten"
    assert all(jobs.look(job.id) is job for job in waiting), (
        "a queued job was forgotten"
    )

    latch.set()
    assert _settles(waiting[-1], {DONE}), waiting[-1].status
    remembered = [job for job in waiting if jobs.look(job.id) is not None]
    assert len(remembered) <= 3, "nothing is being forgotten: %d kept" % len(remembered)


def test_the_line_is_answered_in_the_order_it_was_asked():
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers)

    jobs.submit({"kind": "analysis", "n": 0})
    later = [jobs.submit({"kind": "analysis", "n": n}) for n in (1, 2, 3)]
    latch.set()
    assert _settles(later[-1], {DONE}), later[-1].status
    assert [message["n"] for message in answers.asked] == [0, 1, 2, 3]


def test_the_report_counts_the_waiting_for_whoever_shows_it():
    """Slice 6c-2 draws this; the number has to be there before it can."""
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers)

    job = jobs.submit({"kind": "analysis"})
    assert _settles(job, {RUNNING})
    time.sleep(0.03)
    report = job.report()
    assert report["waiting"] > 0
    assert report["running"] > 0
    assert report["waiting"] >= report["running"], (
        "a job cannot have been running for longer than it has existed"
    )
    latch.set()


# --- what only the line can say (slice 6c-2) ---------------------------------


def test_a_waiting_job_is_told_how_many_are_ahead_of_it():
    """The measurement the whole slice was built on.

    On screen, *computing* and *waiting for another card* looked identical --
    both cards said "updating" -- so two analyses taking turns were read as two
    analyses running at once. A job cannot know this about itself: how long it
    will wait is a fact about everybody else.
    """
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers)

    running = jobs.submit({"kind": "analysis", "n": 0})
    assert _settles(running, {RUNNING})
    first = jobs.submit({"kind": "analysis", "n": 1})
    second = jobs.submit({"kind": "analysis", "n": 2})

    assert jobs.report(first)["ahead"] == 1, "the one being computed does not count"
    assert jobs.report(second)["ahead"] == 2
    latch.set()
    assert _settles(second, {DONE}), second.status


def test_a_job_that_is_running_is_not_told_a_position():
    """Control: `ahead` is what separates the two sentences on screen.

    A running job carrying `ahead: 0` would be indistinguishable from a queued
    one with nothing in front of it, and the card would be back to having a
    single thing to say -- which is the defect.
    """
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers)

    job = jobs.submit({"kind": "analysis"})
    assert _settles(job, {RUNNING})
    assert "ahead" not in jobs.report(job)
    latch.set()


def test_the_position_ignores_what_has_already_ended():
    """A finished job is not in front of anybody, and it is remembered for a
    while -- so counting the dictionary instead of the line would make every
    card report a queue that only grows."""
    answers = _Answers()
    jobs = Jobs(ask=answers)

    for n in range(4):
        done = jobs.submit({"kind": "analysis", "n": n})
        assert _settles(done, {DONE}), done.status

    latch = answers.hold()
    running = jobs.submit({"kind": "analysis", "n": 98})
    assert _settles(running, {RUNNING})
    waiting = jobs.submit({"kind": "analysis", "n": 99})
    assert jobs.report(waiting)["ahead"] == 1
    latch.set()


# --- stopping everything ------------------------------------------------------


class _Killer(object):
    """A stand-in for the resident's `interrupt`: it records, and says whether
    there was a worker to kill."""

    def __init__(self, had_worker=True):
        self.had_worker = had_worker
        self.times = 0

    def __call__(self):
        self.times += 1
        return self.had_worker


def test_stopping_everything_empties_the_line_and_kills_the_worker():
    answers = _Answers()
    latch = answers.hold()
    killer = _Killer()
    jobs = Jobs(ask=answers, interrupt=killer)

    running = jobs.submit({"kind": "analysis", "n": 0})
    assert _settles(running, {RUNNING})
    queued = [jobs.submit({"kind": "analysis", "n": n}) for n in (1, 2)]

    report = jobs.stop_everything()
    assert [job.status for job in queued] == [CANCELLED, CANCELLED]
    assert running.status == CANCELLED
    assert killer.times == 1, "the worker was not killed: %d" % killer.times
    assert report == {"cancelled": 3, "was_running": True, "worker_killed": True}

    latch.set()
    assert _settles(running, {CANCELLED})
    assert [message["n"] for message in answers.asked] == [0], (
        "a cancelled job was computed anyway: %s" % answers.asked
    )


def test_an_answer_that_arrives_after_the_cancellation_is_thrown_away():
    """The race the order of operations exists to close.

    Killing a process is not instantaneous, and the worker may still get its
    answer out -- or the analysis may be running in this process, where nothing
    can stop it at all. Either way the chart belongs to a question nobody is
    asking, and drawing it would put a result on a card the user just cancelled.
    """
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers, interrupt=_Killer())

    job = jobs.submit({"kind": "analysis"})
    assert _settles(job, {RUNNING})
    jobs.stop_everything()

    latch.set()  # the worker answers anyway, a moment too late
    time.sleep(0.05)
    assert job.status == CANCELLED, "the late answer overwrote the cancellation"
    assert job.answer is None


def test_with_no_worker_to_kill_the_work_is_still_cancelled():
    """Nothing will be drawn -- which is what was asked for -- but the answer
    does not claim a kill that did not happen. With no worker the analysis is
    running inside the interface, and no route can end that."""
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers, interrupt=_Killer(had_worker=False))

    job = jobs.submit({"kind": "analysis"})
    assert _settles(job, {RUNNING})
    report = jobs.stop_everything()

    assert job.status == CANCELLED
    assert report["worker_killed"] is False
    assert report["was_running"] is True
    latch.set()


def test_stopping_an_empty_line_kills_nothing_and_says_so():
    """Control: without this, a `was_running` that was always true would pass
    every test above, and the screen would announce an interrupted computation
    to somebody who interrupted nothing."""
    killer = _Killer()
    jobs = Jobs(ask=_Answers(), interrupt=killer)

    report = jobs.stop_everything()
    assert report["cancelled"] == 0
    assert report["was_running"] is False


def test_the_line_still_works_after_everything_was_stopped():
    """There is one thread. If a cancellation left it confused, the interface
    would look fine and never answer again."""
    answers = _Answers()
    latch = answers.hold()
    jobs = Jobs(ask=answers, interrupt=_Killer())

    jobs.submit({"kind": "analysis", "n": 0})
    jobs.stop_everything()
    latch.set()

    after = jobs.submit({"kind": "analysis", "n": 1})
    assert _settles(after, {DONE}), (
        "the queue stopped working after a cancellation: %s" % after.status
    )


def test_a_job_cancelled_in_the_instant_before_it_starts_does_not_start():
    """The window between taking a job out of the line and starting it.

    `_work` checks that the job is still queued before handing it over and
    `_answer` checks again, under the lock, before marking it running. Between
    those two checks there is room for a cancellation to land, and a
    cancellation lost there is an analysis the user stopped running to the end
    anyway -- rare, unreproducible, and never found afterwards.

    It is reached by calling `_answer` directly because the window is one
    instruction wide: there is no honest way to stand inside it from outside, and
    the alternative -- a hook in the production code that exists only so a test
    can pause there -- would be a seam kept open for nobody.
    """
    answers = _Answers()
    jobs = Jobs(ask=answers, interrupt=_Killer())

    job = Job(1, {"kind": "analysis"}, None)
    job.status = CANCELLED
    jobs._answer(job)

    assert answers.asked == [], "a cancelled job was handed to the worker anyway"
    assert job.status == CANCELLED
