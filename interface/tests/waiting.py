# -*- coding: utf-8 -*-
"""The suite's way of waiting for a job: `api/waiting.py`, plus assertions.

Slice 6c changed what `POST /run_analysis` answers: a job, not a chart. Five
tests in four other files still expected the old contract, and they failed
together -- which is the good outcome, but they failed *after* the change went
out instead of while it was being made. The lesson is the project's sixth one,
in a new place: a contract that changes belongs to every caller, and the search
for them is a `grep`, not a memory.

**And the first version of this file got the address wrong.** The waiting was
written once -- here -- but `selftest.py`, bundled inside the executable, is a
caller too, and it cannot import from `tests/`. It went on reading the old
contract, reported all twelve analyses as failed, and took the CI package job
down on three systems. The loop now lives in `api/waiting.py`, which ships; what
is left here is only what belongs to a test, which is the assertions.
"""

# Generous: on CI a cold analysis can take the numba compilation with it, and a
# test that is flaky on a slow machine is worse than a test that is slow.
PATIENCE = 30.0


def answer_for(client, response, headers, patience=PATIENCE):
    """Take the 202 of a submitted job and return the answer it eventually has.

    Everything the routes can say is preserved: the chart with its 200, a
    refusal with its 400 and message, an unforeseen failure with its 500. What
    this removes is only the waiting.
    """
    # Imported here and not at the top of the file: `from api.waiting import ...`
    # runs `api/__init__.py`, which builds the blueprints and pulls in ROSS. At
    # module level that would make every test file importing this one need ROSS
    # to be *collected* -- and a test that cannot be collected is a test nobody
    # notices is gone. It is the same reason `services/worker/resident.py`
    # imports the protocol at the moment of use.
    from api.waiting import answer_for as poll

    assert response.status_code == 202, "the route did not accept the work: %s %s" % (
        response.status_code,
        response.json,
    )
    answer = poll(client, response, headers, patience=patience)
    assert answer is not None, "job %s never stopped working" % response.json["job_id"]
    return answer
