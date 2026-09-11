# -*- coding: utf-8 -*-
"""The one thing the user can do *to* the worker: stop it.

A blueprint of its own, and not another route in `api/analysis.py`, because it
is not about an analysis. It is about the process every analysis shares -- which
is also why the button it serves is a single global one and not one per card.

## Why stopping needs a route at all

Everything else in slice 6c is arranged so the user never has to think about the
worker. This is the exception, and it exists because of a hard fact of the
domain: **a computation already inside ROSS cannot be interrupted politely.**
There is no cancellation point in an eigenvalue solve; the only way out is to
end the process. So the interface has exactly one honest offer to make -- stop
everything, and pay the startup again -- and the screen is required to say that
price *before* spending it, never after.

## What it costs, measured and not guessed

Killing the worker throws away the imported library, the analysis cache and the
code numba compiled. The next analysis pays: eight to twelve seconds to import
ROSS, plus up to sixteen more of numba on the campbell and the crack. Those are
measurements from this project's own machines, and they are the numbers the
confirmation on screen quotes.

## What it does not promise

If the worker never started, the analysis is running **in this process** and no
route can stop it: there is no second process to end, and killing this one would
close the interface. The job is still cancelled -- nothing will be drawn, which
is what the user asked for -- and the answer says `worker_killed: false` rather
than claiming a kill that did not happen.
"""

from flask import Blueprint, jsonify

from services.worker.jobs import JOBS

worker_api = Blueprint("worker_api", __name__)


@worker_api.route("/api/worker/interrupt", methods=["POST"])
def interrupt():
    """Empty the line, kill what is running, and say what actually happened.

    `cancelled` counts the jobs that will now never produce anything -- the one
    that was running included. `was_running` separates "I stopped a computation"
    from "I emptied a queue", which are worth different words on screen.
    `worker_killed` is false when there was no worker to kill: see above.
    """
    return jsonify(dict(JOBS.stop_everything(), status="interrupted"))
