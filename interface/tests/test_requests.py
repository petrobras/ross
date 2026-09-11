# -*- coding: utf-8 -*-
"""The envelope of each route: what it accepts, and what it refuses by name.

Until Phase 2 each route read the body with loose `.get()` calls. A wrong field
name -- `analysisType` instead of `analysis_type` -- raised nothing: it became
`None` and the analysis carried on with the default value. The user saw a
result, only of another configuration.

The envelope declares the fields it accepts and **refuses any key that is not
declared**. The pre-Phase-2 format, with the project loose at the root, is now
recognised by the name of its own keys -- with no special case, and with a
message that talks about the contract instead of complaining about the rotor."""

import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from api import create_app
from api.security import SESSION_TOKEN
from domain.requests import (
    ANALYSIS_REQUEST,
    ROSS_FILE_REQUEST,
    EXPORT_REQUEST,
    MODE_SHAPE_REQUEST,
    ROTOR_REQUEST,
)
from waiting import answer_for

APP = create_app()
AUTH = {"X-ROSS-Token": SESSION_TOKEN}

ALL_ENVELOPES = [
    ROTOR_REQUEST,
    ANALYSIS_REQUEST,
    MODE_SHAPE_REQUEST,
    EXPORT_REQUEST,
    ROSS_FILE_REQUEST,
]


@pytest.fixture
def client():
    APP.config["TESTING"] = True
    with APP.test_client() as client:
        yield client


PROJECT_REQUEST = {
    "name": "Compressor A",
    "uid": "rotor_123",
    "materials": [{"name": "Steel", "rho": "7810", "E": "211e9", "G_s": "81.2e9"}],
    "shafts": [
        {"L": "100", "odl": "50", "idl": "0", "material": "Steel"} for _ in range(3)
    ],
    "bearings": [
        {"element_type": "BASIC", "n": "0", "kxx": "1e6", "cxx": "1e3"},
        {"element_type": "BASIC", "n": "3", "kxx": "1e6", "cxx": "1e3"},
    ],
    "disks": [],
    "gears": [],
    "seals": [],
    "couplings": [],
    "pointmasses": [],
}


def _auth():
    return dict(AUTH)


def _example(field):
    return {"analysis_type": "campbell", "content": "[]"}.get(
        field.name, {dict: {}, list: [], str: ""}[field.kind]
    )


@pytest.mark.parametrize("envelope", ALL_ENVELOPES, ids=lambda e: e.name)
def test_an_unknown_field_is_refused_by_name(envelope):
    """`analysisType` in place of `analysis_type` became None in silence."""
    body = {field.name: _example(field) for field in envelope.fields}
    body["campoQueNaoExiste"] = 1
    with pytest.raises(ValueError) as error:
        envelope.read(body)
    assert "campoQueNaoExiste" in str(error.value)


@pytest.mark.parametrize("envelope", ALL_ENVELOPES, ids=lambda e: e.name)
def test_a_well_formed_body_passes(envelope):
    """Control for the test above."""
    body = {field.name: _example(field) for field in envelope.fields}
    assert set(envelope.read(body)) == envelope.declared


@pytest.mark.parametrize("envelope", ALL_ENVELOPES, ids=lambda e: e.name)
def test_a_field_of_the_wrong_type_says_which_and_what(envelope):
    field = envelope.fields[0]
    body = {c.name: _example(c) for c in envelope.fields}
    body[field.name] = 12345 if field.kind is not int else "text"
    with pytest.raises(ValueError) as error:
        envelope.read(body)
    assert field.name in str(error.value)


def test_a_missing_required_field_is_named():
    with pytest.raises(ValueError) as error:
        ANALYSIS_REQUEST.read({"params": {}})
    assert "analysis_type" in str(error.value)


def test_the_old_flat_format_is_recognised_by_its_own_keys():
    """No special case: the structural keys are unknown to the envelope.

    Before, this was an `if` written by hand inside app.py. Now it falls out of
    the unknown-key check by itself, and the message says it is the old format."""
    with pytest.raises(ValueError) as error:
        ROTOR_REQUEST.read({"materials": [], "shafts": [], "bearings": []})
    message = str(error.value)
    assert "old format" in message
    assert "shafts" in message


def test_an_empty_body_is_not_mistaken_for_the_old_format():
    """Control: the new refusal must not itself become a misleading message."""
    assert ROTOR_REQUEST.read({}) == {"project": {}}
    assert ROTOR_REQUEST.read(None) == {"project": {}}


def test_a_body_that_is_not_an_object_is_refused():
    with pytest.raises(ValueError):
        ROTOR_REQUEST.read([1, 2, 3])


def test_the_defaults_are_not_shared_between_calls():
    """A shared mutable default leaks one call's project into the next."""
    first = ROTOR_REQUEST.read({})
    first["project"]["shafts"] = ["sujeira"]
    assert ROTOR_REQUEST.read({}) == {"project": {}}


def test_the_old_flat_payload_is_refused_by_name(client):
    """The contract changed: say that, instead of complaining about the rotor.

    Sending the project loose at the root -- the pre-Phase-2 format -- the backend
    read an empty project and answered 'Add at least one Shaft!'. The message
    talked about the rotor when the problem was the envelope, and sent whoever
    read it looking in the wrong place."""
    old_one = dict(
        PROJECT_REQUEST, analysis_type="static", params={}, conversion_type=""
    )
    response = client.post("/run_analysis", json=old_one, headers=_auth())
    assert response.status_code == 400
    assert "old format" in response.json["message"]

    response = client.post("/build_rotor", json=PROJECT_REQUEST, headers=_auth())
    assert response.status_code == 400
    assert "old format" in response.json["message"]


def test_an_empty_envelope_is_not_mistaken_for_the_old_format(client):
    """Control: a body with nothing structural must not fall into the refusal above.

    The envelope accepts this one -- `project` simply defaults to empty -- so
    the refusal is the rotor's, and since slice 6c it arrives with the job
    rather than with the request.
    """
    response = answer_for(
        client,
        client.post(
            "/run_analysis",
            json={"analysis_type": "static", "params": {}},
            headers=_auth(),
        ),
        _auth(),
    )
    assert response.status_code == 400
    assert "old format" not in response.json["message"]


def test_the_analysis_route_survives_an_empty_body(client):
    response = client.post("/run_analysis", json={}, headers=_auth())
    assert response.status_code == 400
    assert response.json["status"] == "error"


# --- the work: where it is computed, and when (slices 6b and 6c) --------------
#
# The route stopped computing in 6b and stopped waiting in 6c. These are the
# guards that say both moves were invisible from outside: the same chart, the
# same status codes, the same messages -- and an application that still works
# when the worker does not.


class _FakeWorker(object):
    """A worker with no process behind it, installed into the real resident.

    `holds` makes it answer only when the test says so, which is how a job is
    kept running long enough for another one to queue behind it -- the state the
    screen could not describe before slice 6c-2.
    """

    def __init__(self, answer=None, raises=None, holds=False):
        self.answer = answer
        self.raises = raises
        self.asked = []
        self.killed = False
        self.reached = threading.Event()
        self._go = threading.Event()
        if not holds:
            self._go.set()

    def ask(self, message, timeout=None):
        self.asked.append(message)
        self.reached.set()
        self._go.wait(10.0)
        if self.raises is not None:
            raise self.raises
        return dict(self.answer, id=message.get("id"))

    def release(self):
        self._go.set()

    def stop(self):
        self._go.set()

    def kill(self):
        self.killed = True
        self._go.set()


@pytest.fixture
def worker():
    """Install a fake worker for one test, and take it away afterwards.

    `RESIDENT` is one object for the whole application, so a test that left a
    worker behind would decide the answer of every test after it.
    """
    from services.worker.resident import RESIDENT

    def install(**kwargs):
        fake = _FakeWorker(**kwargs)
        RESIDENT.use(fake)
        return fake

    try:
        yield install
    finally:
        RESIDENT.stop()


ANALYSIS_BODY = {
    "analysis_type": "static",
    "params": {},
    "conversion_type": "none",
    "project": PROJECT_REQUEST,
}


def test_asking_for_an_analysis_answers_with_a_job_and_not_a_chart(client, worker):
    """The point of 6c: the request comes back before the work is done."""
    worker(answer={"ok": True, "plot_json": '{"data": [], "layout": {}}'})
    response = client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth())

    assert response.status_code == 202
    assert response.json["status"] == "accepted"
    assert response.json["job_id"]
    assert response.json["state"] in ("queued", "running")


def test_the_chart_comes_from_the_worker_when_there_is_one(client, worker):
    fake = worker(answer={"ok": True, "plot_json": '{"data": [], "layout": {}}'})
    answer = answer_for(
        client,
        client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth()),
        _auth(),
    )

    assert answer.status_code == 200
    assert answer.json["plot_json"] == '{"data": [], "layout": {}}'
    assert answer.json["computed_by"] == "worker"
    assert fake.asked[0]["kind"] == "analysis"
    assert fake.asked[0]["analysis_type"] == "static"


def test_a_refusal_crossing_the_pipe_is_still_a_400_carrying_its_message(
    client, worker
):
    """ROSS refuses a model through this channel, and the message is the point.

    "Add at least one Shaft!" is what tells the person at the screen what to do.
    If the transport swallowed the exception class -- or if the queue kept only
    the fact that something failed -- that sentence would arrive as a 500 about
    an internal error.
    """
    worker(
        answer={
            "ok": False,
            "failure": "ValueError",
            "error": "Add at least one Shaft!",
        }
    )
    answer = answer_for(
        client,
        client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth()),
        _auth(),
    )

    assert answer.status_code == 400
    assert answer.json["message"] == "Add at least one Shaft!"


def test_a_failure_that_is_not_a_refusal_is_a_500_that_names_it(client, worker):
    """Control: not everything from the worker is a 400.

    Without this, a transport that turned *every* answer into a 400 would pass
    the guard above -- and the status code would stop telling the truth about
    whose fault it is, which is the whole point of BE-10."""
    worker(answer={"ok": False, "failure": "KeyError", "error": "'odl'"})
    answer = answer_for(
        client,
        client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth()),
        _auth(),
    )

    assert answer.status_code == 500
    assert "KeyError" in answer.json["message"]


def test_a_dead_worker_does_not_take_the_analysis_down_with_it(client, worker):
    """The child is gone; the person at the screen still gets their chart.

    This one really computes, in this process, through the same function the
    child runs -- which is what makes the detour a detour and not a second
    implementation.
    """
    from services.worker.host import WorkerGone

    worker(raises=WorkerGone("the worker died"))
    answer = answer_for(
        client,
        client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth()),
        _auth(),
    )

    assert answer.status_code == 200
    assert answer.json["computed_by"] == "this process"
    assert '"data"' in answer.json["plot_json"], "no chart came back"


def test_with_no_worker_at_all_the_route_answers_from_here(client):
    """Control: this is the state the whole suite runs in, and it has to work."""
    answer = answer_for(
        client,
        client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth()),
        _auth(),
    )

    assert answer.status_code == 200
    assert answer.json["computed_by"] == "this process"
    assert '"data"' in answer.json["plot_json"]


def test_the_subject_of_a_request_reaches_the_queue_as_the_key(client, worker):
    """The key is how a stale job leaves the line, and it comes from the body.

    Dropped on the way in, every supersede in `tests/test_jobs.py` would still
    pass and nothing would ever leave the queue in the running program.
    """
    from services.worker.jobs import JOBS

    worker(answer={"ok": True, "plot_json": "{}"})
    response = client.post(
        "/run_analysis", json=dict(ANALYSIS_BODY, key="card:42"), headers=_auth()
    )

    assert JOBS.look(response.json["job_id"]).key == "card:42"
    answer_for(client, response, _auth())


def test_a_job_nobody_has_heard_of_is_a_404(client):
    """Asked about a name that does not exist -- a reload, a job long finished.

    404 and not 400: the request is well formed, and what is missing is the
    thing it names.
    """
    response = client.get("/api/jobs/j-nonsense", headers=_auth())
    assert response.status_code == 404
    assert "j-nonsense" in response.json["message"]


# The screen of slice 6c-2: which of the two states a card is in, and the one
# button that can end the work. The arithmetic of the line is driven with no
# process and no HTTP in `tests/test_jobs.py`; what these guards are for is that
# it reaches the browser at all, and that pressing the button does what the
# confirmation on screen promised.


def test_a_waiting_job_tells_the_browser_how_many_are_ahead_of_it(client, worker):
    """Without this number the card has one sentence for two states.

    That is not a hypothesis: with both cards saying "updating", the Campbell and
    the crack taking turns were read as running at once -- and there is one
    worker, so they cannot.
    """
    fake = worker(answer={"ok": True, "plot_json": "{}"}, holds=True)
    running = client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth())
    assert fake.reached.wait(10.0), "the first analysis never reached the worker"

    waiting = client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth())
    state = client.get("/api/jobs/%s" % waiting.json["job_id"], headers=_auth())

    assert state.json["status"] == "working"
    assert state.json["state"] == "queued"
    assert "ahead" in state.json, "the position in the line never reaches the browser"
    assert state.json["ahead"] >= 1, "the analysis being computed was not counted"

    fake.release()
    answer_for(client, running, _auth())
    answer_for(client, waiting, _auth())


def test_the_stop_button_cancels_the_work_and_kills_the_worker(client, worker):
    """The only honest offer the interface can make about work in progress.

    A computation already inside ROSS has no cancellation point; the only way out
    is to end the process. So the route empties the line and kills, and the
    screen quotes the price -- eight to twelve seconds of importing ROSS again,
    plus up to sixteen of numba -- before spending it.
    """
    fake = worker(answer={"ok": True, "plot_json": "{}"}, holds=True)
    client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth())
    assert fake.reached.wait(10.0), "the analysis never reached the worker"

    stopped = client.post("/api/worker/interrupt", headers=_auth())

    assert stopped.status_code == 200
    assert stopped.json["status"] == "interrupted"
    assert stopped.json["cancelled"] >= 1
    assert stopped.json["was_running"] is True
    assert stopped.json["worker_killed"] is True
    assert fake.killed, "the worker was not killed, so nothing actually stopped"
    fake.release()


def test_a_cancelled_job_is_an_answer_and_not_an_error(client, worker):
    """The card has to be able to tell "you stopped this" from "this broke".

    Sent through the refusal channel a cancellation would arrive as a 500 about
    an internal error, and the user would get a red box for something they chose
    -- with the real message, "Interrupted", nowhere on screen.
    """
    fake = worker(answer={"ok": True, "plot_json": "{}"}, holds=True)
    response = client.post("/run_analysis", json=ANALYSIS_BODY, headers=_auth())
    assert fake.reached.wait(10.0), "the analysis never reached the worker"
    client.post("/api/worker/interrupt", headers=_auth())

    state = client.get("/api/jobs/%s" % response.json["job_id"], headers=_auth())

    assert state.status_code == 200, (
        "a cancellation the user asked for is not a failure"
    )
    assert state.json["status"] == "cancelled"
    assert "plot_json" not in state.json, "a cancelled job must not carry a chart"
    fake.release()


def test_stopping_with_nothing_running_is_not_an_error(client):
    """Control: the button exists on a page where the user may press it twice."""
    stopped = client.post("/api/worker/interrupt", headers=_auth())

    assert stopped.status_code == 200
    assert stopped.json["status"] == "interrupted"
