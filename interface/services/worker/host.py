# -*- coding: utf-8 -*-
"""The parent's handle on the worker: start it, ask it, restart it.

This is the half the test suite cannot reach -- it needs a real process -- so it
is kept as small as the job allows, and everything it does that could be decided
without a process was moved into `protocol.py`.

WHY IT LAUNCHES *ITSELF*. Frozen by PyInstaller there is no `python` to call and
no script to point at: there is one executable. So the worker is a **mode** of
that same executable, selected by `--worker`, exactly as `--selftest` already
is. Running from source the command is the interpreter plus this repository's
entry point; frozen, it is the executable plus the flag. Nothing else changes.

WHY NOT `multiprocessing`. It serialises the target and its arguments, and
serialisation across this boundary is the thing that is broken (two of the
twelve ROSS results refuse it). Its spawn also re-executes the bundle with
private arguments, which is the classic way a frozen program ends up opening
copies of itself. A pipe carrying text has neither problem.

WHY STDERR IS DRAINED BY A THREAD AND NEVER IGNORED. The child writes to stderr:
ROSS prints while it computes, and warnings land there. A pipe nobody reads
fills up, and a child writing into a full pipe **blocks forever** -- which would
look exactly like an analysis that never finishes, on the machine of whoever is
waiting. So a thread reads it continuously and keeps the last lines for the
diagnosis of a worker that dies.
"""

import collections
import json
import os
import queue
import subprocess
import sys
import threading
import time

READY_TIMEOUT = 180.0
KEPT_STDERR_LINES = 50


class WorkerGone(Exception):
    """The worker did not answer, or is no longer there."""


def worker_command():
    """How to start a second copy of this program in worker mode.

    Frozen, `sys.executable` is the bundle itself. From source it is the
    interpreter, and the entry point is this repository's `app.py` -- named from
    here rather than from `sys.argv[0]`, which is whatever script the caller
    happened to run (a probe, a test runner) and not necessarily ours.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker"]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return [sys.executable, os.path.join(root, "app.py"), "--worker"]


class Worker(object):
    """One persistent child process, and the conversation with it."""

    def __init__(self, command=None, ready_timeout=READY_TIMEOUT):
        self.command = command or worker_command()
        self.ready_timeout = ready_timeout
        self.process = None
        self.startup = None
        self._answers = queue.Queue()
        self._complaints = collections.deque(maxlen=KEPT_STDERR_LINES)
        self._next_id = 0

    # --- the process ---------------------------------------------------------

    def start(self):
        """Launch the child and wait until it says it is ready.

        The wait is the point: importing ROSS costs about nine seconds, and
        paying it here -- while the user is still drawing a rotor -- is the
        difference between a slow startup and an analysis that mysteriously
        takes nine seconds longer than the one after it.
        """
        began = time.perf_counter()
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        threading.Thread(target=self._read_answers, daemon=True).start()
        threading.Thread(target=self._read_complaints, daemon=True).start()

        answer = self.ask({"kind": "ping"}, timeout=self.ready_timeout)
        self.startup = time.perf_counter() - began
        return answer

    def stop(self):
        """Close the conversation, and end the process if it does not leave."""
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
            self.process.wait()
        self.process = None

    def kill(self):
        """End the computation now. This is what cancellation costs.

        The child dies with the cache and the compiled code inside it, so the
        next analysis pays the startup again: about nine seconds to import ROSS
        and up to sixteen more of numba, both measured. That is why cancelling
        has to be something the user asks for, never something the interface
        does on its own.
        """
        if self.process is not None:
            self.process.kill()
            self.process.wait()
            self.process = None

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    # --- the conversation ----------------------------------------------------

    def ask(self, message, timeout):
        """Send one message and wait for its answer."""
        if self.process is None:
            raise WorkerGone("the worker was never started, or was stopped")

        self._next_id += 1
        message = dict(message, id=self._next_id)
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except Exception as error:
            raise WorkerGone("could not reach the worker: %s%s" % (error, self.why()))

        try:
            answer = self._answers.get(timeout=timeout)
        except queue.Empty:
            raise WorkerGone(
                "the worker did not answer in %.0f s%s" % (timeout, self.why())
            )
        if answer is None:
            raise WorkerGone("the worker ended before answering%s" % self.why())
        # The queue has no addressing: `ask` takes whatever comes out of the
        # pipe, and that is only correct while one conversation happens at a
        # time. If the identifiers stop matching, two answers have been swapped
        # -- which on a chart route means somebody looking at the answer to a
        # question they did not ask, with nothing on screen to say so. That is
        # FE-05 of the audit, on this side of the wire. A conversation out of
        # step cannot be repaired, so it ends here and the worker is replaced.
        if answer.get("id") != message["id"]:
            raise WorkerGone(
                "the worker answered %r to message %r: the conversation is out "
                "of step%s" % (answer.get("id"), message["id"], self.why())
            )
        return answer

    def why(self):
        """The child's last words, for a message that would otherwise say nothing."""
        if not self._complaints:
            return ""
        return "\n  last from the worker:\n    " + "\n    ".join(self._complaints)

    # --- the two reading threads ---------------------------------------------

    def _read_answers(self):
        process = self.process
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._answers.put(json.loads(line))
            except ValueError:
                # Not protocol. The child guards its own stdout, so this means
                # something wrote to the descriptor anyway -- keep it as a
                # complaint instead of throwing it away.
                self._complaints.append("stdout: %s" % line[:200])
        self._answers.put(None)

    def _read_complaints(self):
        for line in self.process.stderr:
            line = line.rstrip()
            if line:
                self._complaints.append(line[:200])
