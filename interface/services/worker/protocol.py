# -*- coding: utf-8 -*-
"""One message in, one message out -- and no process anywhere in sight.

The whole protocol is this function. It takes a dictionary and returns a
dictionary, which means the suite can drive every analysis through the worker's
own code path without spawning anything, and the process is reduced to what it
really is: a pipe with a loop around it.

That division is not tidiness. The parts of this design that can go wrong are
of two very different kinds -- "the answer is wrong" and "the pipe broke" -- and
mixing them into one testable surface would leave both half covered.

**Errors are answers, not exceptions.** A message that fails comes back as a
message saying so, with the exception's class name and text. A worker that dies
on a bad request would take the cache and the warm numba with it, and the next
analysis would pay nine seconds of spawn plus up to sixteen of compilation --
for a typo in a field. The only thing that should ever kill this process is the
parent deciding to kill it.
"""

from domain.cache import ANALYSIS_CACHE
from services.analysis.pipeline import figure_json, mode_shape_json

PROTOCOL_VERSION = 1


def _answer(message, **fields):
    answer = {"id": message.get("id"), "ok": True}
    answer.update(fields)
    return answer


def _refusal(message, error):
    return {
        "id": message.get("id"),
        "ok": False,
        "failure": type(error).__name__,
        "error": "%s" % error,
    }


def _ping(message):
    """Is the worker up, and is it speaking the version we expect?

    The parent waits for this before sending anything, which is how the nine
    seconds of importing ROSS become a startup cost rather than a surprise in
    the middle of the first analysis.
    """
    return _answer(message, ready=True, protocol=PROTOCOL_VERSION)


def _analysis(message):
    return _answer(
        message,
        plot_json=figure_json(
            message["analysis_type"],
            message["params"],
            message["conversion_type"],
            message["project"],
            message.get("language", "en"),
        ),
    )


def _mode_shape(message):
    return _answer(
        message,
        plot_json=mode_shape_json(
            message["params"],
            message["conversion_type"],
            message["project"],
            message["point"],
        ),
    )


def _forget(message):
    """Empty the worker's result cache.

    One caller, one reason. The check needs to ask the worker the same question
    twice and find out whether it answers the same way twice -- and a cache
    would hand back the first answer, so the repeat would be measuring the cache
    instead of the computation. That exact defect cost this project a whole
    round of misleading timings once already, in another process; naming the
    operation is cheaper than discovering it again.

    It is also the first half of eviction, which the queue will need anyway.
    """
    ANALYSIS_CACHE.clear()
    return _answer(message, forgotten=True)


KINDS = {
    "ping": _ping,
    "analysis": _analysis,
    "mode_shape": _mode_shape,
    "forget": _forget,
}


def handle(message):
    """Answer one message. Never raises: a failure is a message too."""
    if not isinstance(message, dict):
        return {
            "id": None,
            "ok": False,
            "failure": "ProtocolError",
            "error": "a message has to be an object, got %s" % type(message).__name__,
        }
    work = KINDS.get(message.get("kind"))
    if work is None:
        return {
            "id": message.get("id"),
            "ok": False,
            "failure": "ProtocolError",
            "error": "unknown kind %r; known: %s"
            % (message.get("kind"), ", ".join(sorted(KINDS))),
        }
    try:
        return work(message)
    except Exception as error:
        return _refusal(message, error)
