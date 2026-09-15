# -*- coding: utf-8 -*-
"""Asking about a job until it is over. Written once, and it ships.

Since slice 6c a POST to `/run_analysis` answers **a job**, not a chart: `202`
with a name, and the chart arrives later from `GET /api/jobs/<id>`. Every
caller that wants the chart therefore has to wait, and this is the only place
that knows how.

## Why it lives here and not in the test suite

It started in `tests/waiting.py`, written exactly because five tests had each
been about to grow their own polling loop. That was the right instinct and the
wrong address: **the test suite is not the only caller.** `selftest.py` -- which
is bundled into the executable and is what CI runs to decide whether a build
works -- posts to the same route, and so does `tools/conversion_probe.py`.
Neither can import from `tests/`, and neither shipped with it.

The cost of getting that wrong was measured: after slice 6c the selftest went on
checking `status == "success"` on the 202, reported **all twelve analyses as
failed**, and took down the package job on all three systems. The guards could
not have caught it, because they swept `tests/` and this file is not a test.

So the rule is one step wider than it was: a contract belongs to every caller,
and the helper that hides a contract belongs where **every** caller can reach
it -- which for something shipped inside the executable means shipped code.

## What it does not do

It does not judge. A refusal keeps its 400 and its message, an unforeseen
failure its 500, and a job that never ends comes back as `None` rather than an
exception -- the suite turns that into a failed assertion, the selftest into a
printed line, and neither behaviour belongs to the waiting.
"""

import time

POLL_SECONDS = 0.01

# Generous, because the slowest caller is the selftest running twelve real ROSS
# analyses on a cold CI runner, numba compilation included. What this number is
# for is a job that will never answer at all.
PATIENCE = 300.0


def answer_for(client, accepted, headers, patience=PATIENCE):
    """Return the answer the submitted job eventually has.

    `accepted` is what the POST returned. Anything that is not a `202` is given
    back unchanged: a body the envelope refused never became a job, and its
    message is already the answer.

    Returns `None` -- and only then -- if the job never stopped working.
    """
    if accepted.status_code != 202:
        return accepted
    job_id = (accepted.get_json() or {}).get("job_id")
    if not job_id:
        return accepted

    until = time.time() + patience
    while time.time() < until:
        state = client.get("/api/jobs/%s" % job_id, headers=headers)
        if state.status_code != 200:
            return state
        if (state.get_json() or {}).get("status") != "working":
            return state
        time.sleep(POLL_SECONDS)
    return None
