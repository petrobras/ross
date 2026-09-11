# -*- coding: utf-8 -*-
"""The analyses: dispatch to the runners, and the click on the Campbell diagram.

Since slice 6a this module is transport and nothing else. What an analysis *is*
-- refuse, build, cache, compute, draw, serialise -- lives in
`services/analysis/pipeline.py`.

Since slice 6b it does not choose **where** that happens: a message goes to the
`RESIDENT`, which sends it to the worker process when there is one and answers
it here when there is not. Either way the same `handle()` runs on the same
message.

Since slice 6c it does not choose **when**, either. The route puts the work in a
line and answers immediately with the name of the job; the browser asks about
that name until there is a chart. Three things come out of that, and only the
first was the point:

* a job that nobody wants any more can leave the queue before it runs, which
  until now was something the browser only pretended to do;
* the screen stops being held by a computation it is not showing yet;
* and a result can only land in the card that asked for it, because the answer
  carries the name of the question. That is FE-05 of the audit, fixed by the
  shape of the thing instead of by everyone remembering to abort.

What did **not** change is the error contract. A refusal still arrives as **400
with its message** -- the channel ROSS says "Add at least one Shaft!" through --
and anything else as **500 naming the type**. The refusal now waits in a job
until somebody asks, and `raise_if_refused` raises it then, so `api/errors.py`
decides exactly as it always did.
"""

from flask import Blueprint, abort, jsonify, request

from domain.requests import ANALYSIS_REQUEST, MODE_SHAPE_REQUEST
from services.worker.jobs import CANCELLED, DONE, FAILED, JOBS, SUPERSEDED
from services.worker.resident import raise_if_refused

analysis = Blueprint("analysis", __name__)


def _accepted(job):
    """202: the work exists and has a name. It does not exist *yet* as a chart."""
    return jsonify(dict(JOBS.report(job), status="accepted")), 202


@analysis.route("/run_analysis", methods=["POST"])
def run_analysis():
    """Ask for one analysis of the rotor described in the body.

    Answers with a job, not a chart. The computation lives in
    services/analysis/: one runner per analysis, each with `spec` (what the
    computation needs), `compute` (what ROSS does) and `plot` (the figure).
    Until Phase 2 this was a 460-line if/elif with the same block repeated
    twelve times.
    """
    payload = ANALYSIS_REQUEST.read(request.get_json(silent=True))
    job = JOBS.submit(
        {
            "kind": "analysis",
            "analysis_type": payload["analysis_type"],
            "params": payload["params"],
            "conversion_type": payload["conversion_type"],
            "project": payload["project"],
            "language": request.args.get("lang", "en"),
        },
        key=payload["key"] or None,
    )
    return _accepted(job)


@analysis.route("/api/campbell/mode_shape", methods=["POST"])
def campbell_mode_shape():
    """Ask for the 3D mode shape of a point clicked on the Campbell diagram.

    This one has to reach **the same** worker as the diagram it was clicked on:
    the Campbell result is taken from the cache, and the cache lives inside the
    process that computed it. One resident, one worker, so it does -- but the
    day there is more than one, this is the route that will silently start
    recomputing a fifty-speed Campbell on every click.
    """
    payload = MODE_SHAPE_REQUEST.read(request.get_json(silent=True))
    job = JOBS.submit(
        {
            "kind": "mode_shape",
            "params": payload["params"],
            "conversion_type": payload["conversion_type"],
            "project": payload["project"],
            "point": payload["point"],
        },
        key=payload["key"] or None,
    )
    return _accepted(job)


@analysis.route("/api/jobs/<job_id>", methods=["GET"])
def job_state(job_id):
    """Where is this job, and -- when it is over -- what did it produce?

    The four endings are four different things to the person waiting, and the
    route keeps them apart instead of flattening them into "no chart":

    * **done** -- the chart, and where it was computed;
    * **failed** -- raised again here, so the status code still tells the truth
      about whose fault it was;
    * **superseded** -- this job was taken out of the line because the same card
      asked for something else. Nobody is wrong, and nothing should appear on
      screen: whoever is polling this one is a request the user already replaced;
    * **cancelled** -- the user pressed the button, having been told the price.
      Unlike a supersede, this one *is* addressed to whoever is watching: they
      stopped it, and a card that quietly went back to normal would leave them
      wondering whether the button did anything.

    And while it is still working, the answer carries `ahead` -- how many have to
    end first. That number is slice 6c-2 itself: without it the screen cannot
    tell *computing* from *waiting for another card*, which is exactly how two
    analyses taking turns were read as two analyses running at once.
    """
    job = JOBS.look(job_id)
    if job is None:
        abort(
            404,
            description=(
                "No job called '%s'. It either never existed or finished long "
                "enough ago to be forgotten." % job_id
            ),
        )

    if job.status == DONE:
        return jsonify(
            dict(
                JOBS.report(job),
                status="success",
                plot_json=job.answer["plot_json"],
                computed_by=job.where,
            )
        )

    if job.status == FAILED:
        raise_if_refused(dict(job.failure or {}, ok=False))

    if job.status == SUPERSEDED:
        return jsonify(dict(JOBS.report(job), status="superseded"))

    if job.status == CANCELLED:
        return jsonify(dict(JOBS.report(job), status="cancelled"))

    return jsonify(dict(JOBS.report(job), status="working"))
