# -*- coding: utf-8 -*-
"""The worker: the half that can be tested without a process.

Slice 6a moved the computation of an analysis behind a boundary. Two very
different things can go wrong at a boundary -- the answer is wrong, or the pipe
broke -- and this file covers the first, plus the structural properties of the
second that can be read from the source.

What is deliberately **not** here: starting a real child. That is
`app.py --worker-check`, and it has to run from the built executable as well as
from source, because the question it answers -- can a frozen program start a
second copy of itself -- has a different answer in each.
"""

import ast
import io
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services.worker.child import serve  # noqa: E402


def _source(*parts):
    with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


def _without_prose(text):
    """The code, with comments and docstrings taken out.

    Three guards in this project have gone red at the comment that explained
    them (lesson 10). A sweep that reads source has to discount the text that
    talks *about* the source.
    """
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
    # `ast.unparse` writes every string with single quotes; normalising them
    # here keeps the guards below from depending on which quote the author
    # happened to type.
    return ast.unparse(ast.fix_missing_locations(tree)).replace("'", '"')


# --- the loop, with no process anywhere ---------------------------------------


def _echo(message):
    return {"id": message.get("id"), "ok": True, "echo": message.get("kind")}


def _run(text):
    outgoing = io.StringIO()
    serve(io.StringIO(text), outgoing, answer=_echo)
    return [json.loads(line) for line in outgoing.getvalue().splitlines() if line]


def test_the_loop_answers_one_line_per_message():
    answers = _run('{"id": 1, "kind": "ping"}\n{"id": 2, "kind": "ping"}\n')
    assert [a["id"] for a in answers] == [1, 2]


def test_blank_lines_are_not_messages():
    """A flush, a stray newline, the end of a write -- none of them is a request."""
    assert _run('\n   \n{"id": 7, "kind": "ping"}\n\n') == [
        {"id": 7, "ok": True, "echo": "ping"}
    ]


def test_a_line_that_is_not_json_is_answered_and_not_fatal():
    """The worker must survive the parent's mistakes.

    Dying here would take the cache and the compiled code with it, and the next
    analysis would pay nine seconds of startup plus up to sixteen of numba --
    for one malformed line.
    """
    answers = _run('this is not json\n{"id": 3, "kind": "ping"}\n')
    assert answers[0]["ok"] is False
    assert answers[0]["failure"] == "ProtocolError"
    assert answers[1]["id"] == 3, "the loop stopped at the bad line"


def test_the_answer_carries_the_identifier_it_was_asked_with():
    """Without it the parent cannot tell whose answer it is reading."""
    assert _run('{"id": 41, "kind": "ping"}')[0]["id"] == 41


# --- the child guards the channel it speaks on --------------------------------


def test_the_child_hands_stdout_to_stderr_before_serving():
    """ROSS prints while it computes, and this channel carries the protocol.

    `run_crack` and `run_rubbing` write "Running direct method" to standard
    output. One line of that between two protocol lines and the parent reads a
    chart as a syntax error. The child takes the real descriptor aside and
    points `sys.stdout` at stderr, and it has to do that **before** anything
    else runs.
    """
    body = _without_prose(_source("services", "worker", "child.py"))
    lines = [line.strip() for line in body.split("\n")]
    redirect = [n for n, line in enumerate(lines) if line == "sys.stdout = sys.stderr"]
    serving = [n for n, line in enumerate(lines) if line.startswith("serve(")]
    assert redirect, "the child no longer moves sys.stdout out of the way"
    assert serving, "the sweep no longer finds the call to serve()"
    assert redirect[0] < serving[0], (
        "the redirection has to happen before the loop starts, or the first "
        "thing ROSS prints corrupts the first answer"
    )


def test_the_protocol_is_written_to_the_descriptor_taken_aside():
    """Writing to `sys.stdout` after the redirect would send answers to stderr."""
    body = _without_prose(_source("services", "worker", "child.py"))
    assert "serve(sys.stdin, protocol_out)" in body, (
        "the loop no longer writes to the stream captured before the redirect"
    )


# --- the parent's side, read from the source ----------------------------------


def test_the_parent_drains_the_child_stderr():
    """A pipe nobody reads fills up, and the child then blocks forever.

    The symptom is an analysis that never finishes, on the machine of whoever is
    waiting -- with no error anywhere. The only defence is that somebody is
    always reading.
    """
    body = _without_prose(_source("services", "worker", "host.py"))
    assert "_read_complaints" in body, "nothing reads the worker's stderr"
    assert body.count("threading.Thread") >= 2, (
        "the two streams are no longer drained by threads of their own"
    )


def test_the_worker_is_started_as_this_same_program():
    """Frozen there is no interpreter to call and no script to point at."""
    from services.worker.host import worker_command

    command = worker_command()
    assert command[-1] == "--worker"
    assert command[0] == sys.executable
    body = _without_prose(_source("services", "worker", "host.py"))
    assert 'getattr(sys, "frozen", False)' in body, (
        "the command no longer distinguishes the frozen executable from the "
        "interpreter, and one of the two forms is wrong"
    )


def test_the_worker_mode_is_decided_before_the_application_is_built():
    """`--worker` has to win before `create_app()` runs, or the child builds Flask."""
    body = _without_prose(_source("app.py"))
    lines = [line.strip() for line in body.split("\n")]
    flag = [n for n, line in enumerate(lines) if '"--worker" in sys.argv' in line]
    built = [n for n, line in enumerate(lines) if line.startswith("app = create_app()")]
    assert flag, "app.py no longer dispatches --worker"
    assert built, "the sweep no longer finds where the application is built"
    assert flag[0] < built[0]


def test_the_worker_package_does_not_reach_the_transport_layer():
    """It answers messages; it must not know that HTTP exists.

    If Flask leaked in here, the worker would carry the web framework into its
    own process for nothing -- and the boundary would be decorative, because the
    two sides would share the layer the boundary exists to separate.
    """
    intruders = []
    folder = os.path.join(ROOT, "services", "worker")
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".py"):
            continue
        body = _without_prose(_source("services", "worker", name))
        for forbidden in ("flask", "api."):
            if forbidden in body:
                intruders.append("%s: %s" % (name, forbidden))
    assert intruders == [], "the worker reaches into transport: %s" % intruders


def test_the_route_stopped_computing():
    """After 6a the route is transport, and the pipeline is the single copy.

    Two callers of the same steps -- the route and the worker -- is exactly the
    shape this refactoring exists to prevent. The guard is that the transport
    layer no longer names the pieces of a computation.
    """
    body = _without_prose(_source("api", "analysis.py"))
    for computing in (
        "import ross",
        "ANALYSIS_CACHE",
        "spec_key",
        "build_rotor_from_ui",
    ):
        assert computing not in body, (
            "api/analysis.py is computing again (%s): the worker and the route "
            "would drift apart" % computing
        )


def test_the_pipeline_is_the_only_place_that_draws_an_analysis():
    """Control: if the route kept its own copy, the guard above would still pass."""
    pipeline = _without_prose(_source("services", "analysis", "pipeline.py"))
    for step in ("build_rotor_from_ui", "ANALYSIS_CACHE", "update_layout", "to_json"):
        assert step in pipeline, "the pipeline no longer does %s" % step


# --- comparing two charts -----------------------------------------------------
#
# The numbers below are not invented. They are the two charts a real run of
# `--worker-check` printed when it accused the worker of changing the answer:
# `campbell`'s first two critical speeds and `modes`' first three magnitudes, as
# this process and a second process computed them. Keeping the measured bytes
# means the comparator is tested against the case it exists for, and that the
# day someone tightens the tolerance the suite says which real chart it breaks.

CAMPBELL_HERE = (
    '{"data": [{"type": "scatter", '
    '"x": {"dtype": "f8", "bdata": "DrLArx4Bn0CNMP9FNQGfQA=="}}]}'
)
CAMPBELL_THERE = (
    '{"data": [{"type": "scatter", '
    '"x": {"dtype": "f8", "bdata": "X7HArx4Bn0B2M/9FNQGfQA=="}}]}'
)
MODES_HERE = (
    '{"data": [{"type": "scatter3d", "customdata": '
    '{"dtype": "f8", "bdata": "EZ4GoEGEiD8Hkrvf2KaIP3tpfWIkyIg/"}}]}'
)
MODES_THERE = (
    '{"data": [{"type": "scatter3d", "customdata": '
    '{"dtype": "f8", "bdata": "34RMvTDWoj+YWBgeStKiP5hRHeeIzqI/"}}]}'
)


def test_the_measured_bytes_really_are_the_measured_numbers():
    """Control for everything below: what do those blobs actually contain?

    If the decoder were wrong -- the endianness, the width, the base64 -- the
    two verdicts further down could still come out right by luck, and the
    comparator would be reading noise. So the numbers are pinned here first.
    """
    from services.worker.compare import decoded

    speeds = decoded(json.loads(CAMPBELL_HERE)["data"][0]["x"])
    assert ["%.13f" % v for v in speeds] == ["1984.2799673184086", "1984.3020248292335"]
    shape = decoded(json.loads(MODES_HERE)["data"][0]["customdata"])
    assert ["%.6f" % v for v in shape] == ["0.011971", "0.012037", "0.012100"]


def test_a_chart_equal_to_itself_is_identical_without_being_parsed():
    from services.worker.compare import IDENTICAL, compare

    assert compare(CAMPBELL_HERE, CAMPBELL_HERE).verdict == IDENTICAL


def test_campbell_last_bit_noise_is_close_and_not_a_difference():
    """1984.2799673184086 against 1984.2799673183688: two parts in 1e14.

    No axis, no label and no exported figure can show that. Calling it "the
    worker changed the answer" is the accusation this comparator exists to stop
    making.
    """
    from services.worker.compare import CLOSE, compare

    verdict = compare(CAMPBELL_HERE, CAMPBELL_THERE)
    assert verdict.verdict == CLOSE, verdict.describe()
    assert verdict.worst < 1e-12, verdict.describe()


def test_the_orbit_angle_that_moved_between_processes_is_a_difference():
    """0.011971 rad against 0.036790 rad. Two hundred per cent apart.

    `customdata` here is ROSS's `angle_0` -- the orbit angle in the hover box --
    so the disagreement is about a degree and a half in an angle near zero. It
    is small in radians and enormous in relative terms, and relative is the
    right measure for a comparator that must not care how a quantity is scaled.
    """
    from services.worker.compare import DIFFERENT, compare

    verdict = compare(MODES_HERE, MODES_THERE)
    assert verdict.verdict == DIFFERENT, verdict.describe()
    assert "customdata" in verdict.describe(), verdict.describe()


def test_the_report_says_which_arrays_differ_and_which_agree():
    """Stopping at the first difference is what made me misread this case.

    The check reported `modes` differing at `customdata[0]` and stopped, so
    nothing in the output said whether the plotted deflection `y` had moved as
    well -- and I told Leonardo the picture had changed when what had changed
    was the hover annotation. The sweep now runs to the end of the numbers, and
    the report has to name both halves: which array is out, and how many are
    not.
    """
    from services.worker.compare import DIFFERENT, compare

    here = '{"data": [{"customdata": [0.011971], "y": [1.0, 2.0], "x": [0.0, 1.0]}]}'
    there = '{"data": [{"customdata": [0.036790], "y": [1.0, 2.0], "x": [0.0, 1.0]}]}'
    verdict = compare(here, there)
    assert verdict.verdict == DIFFERENT
    assert sorted(verdict.apart) == [".data[0].customdata"], verdict.apart
    assert verdict.describe().startswith("1 of 3 arrays differ"), verdict.describe()
    assert "2 agree" in verdict.describe(), verdict.describe()


def test_the_worst_difference_wins_even_when_it_comes_last():
    """Control: a sweep that quietly kept stopping early would report the first.

    The arrays here are **binary**, not JSON lists, and that is the whole point
    of the control. The first version of this test used lists, and a mutation
    that put the early stop back inside the base64 loop passed it untouched --
    the two representations are two different code paths, and a real chart uses
    the one the test was not exercising. Lesson 10's shape again: the guard
    covered the example and not the category.
    """
    from services.worker.compare import compare

    first = '{"dtype": "f8", "bdata": "AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/"}'
    changed = '{"dtype": "f8", "bdata": "AAAAAAAA+D8AAAAAAADwPwAAAAAAAFlA"}'
    same = '{"dtype": "f8", "bdata": "AAAAAAAA8D8AAAAAAADwPw=="}'
    verdict = compare(
        '{"a": %s, "z": %s}' % (first, same),
        '{"a": %s, "z": %s}' % (changed, same),
    )
    assert "1 of 2 arrays differ" in verdict.describe(), verdict.describe()
    assert "1 agree" in verdict.describe(), verdict.describe()
    assert ".a[2]" in verdict.describe(), (
        "the worst difference is the last value of the array and the report "
        "stopped before it: %s" % verdict.describe()
    )
    assert "2 of 3 values" in verdict.describe(), verdict.describe()


def test_a_difference_just_above_the_threshold_is_still_a_difference():
    """Control: the tolerance has an upper edge, and it is where it claims.

    Without this, a comparator that answered `close` to everything would pass
    every test above.
    """
    from services.worker.compare import CLOSE, DIFFERENT, compare

    under = '{"y": [1.0, 1.0000000001]}'
    over = '{"y": [1.0, 1.00000001]}'
    assert compare('{"y": [1.0, 1.0]}', under).verdict == CLOSE
    assert compare('{"y": [1.0, 1.0]}', over).verdict == DIFFERENT


def test_zero_needs_an_absolute_floor_because_it_has_no_relative_scale():
    from services.worker.compare import CLOSE, DIFFERENT, compare

    assert compare('{"y": [0.0]}', '{"y": [1e-15]}').verdict == CLOSE
    assert compare('{"y": [0.0]}', '{"y": [0.001]}').verdict == DIFFERENT


def test_structure_is_never_forgiven():
    """A tolerance for numbers must not become a tolerance for a changed chart.

    A renamed trace, a missing axis, one point fewer, a flag that turned into a
    number -- none of those is numerical noise, and each of them is a bug that a
    lenient comparison would hide.
    """
    from services.worker.compare import DIFFERENT, compare

    one_number = '{"dtype": "f8", "bdata": "AAAAAAAA8D8="}'
    two_numbers = '{"dtype": "f8", "bdata": "AAAAAAAA8D8AAAAAAAAAQA=="}'
    four_flat = (
        '{"dtype": "f8", "bdata": "AAAAAAAA8D8AAAAAAAAAQAAAAAAAAAhAAAAAAAAAEEA="}'
    )
    four_square = four_flat[:-1] + ', "shape": "2, 2"}'
    cases = [
        ('{"name": "Forward"}', '{"name": "Backward"}'),
        ('{"y": [1.0, 2.0]}', '{"y": [1.0, 2.0, 3.0]}'),
        ('{"y": [1.0], "type": "scatter"}', '{"y": [1.0]}'),
        ('{"visible": true}', '{"visible": 1}'),
        ('{"y": %s}' % one_number, '{"y": [1.0]}'),
        # A trace that lost a point, and a surface flattened into a line: both
        # arrive as arrays of doubles in base64, where the numbers that remain
        # agree perfectly. Only the length and the declared shape say so.
        ('{"y": %s}' % one_number, '{"y": %s}' % two_numbers),
        ('{"z": %s}' % four_square, '{"z": %s}' % four_flat),
    ]
    for one, other in cases:
        assert compare(one, other).verdict == DIFFERENT, "forgave %s vs %s" % (
            one,
            other,
        )


def test_the_same_values_written_differently_are_not_reported_as_identical():
    """`identical` has to keep meaning "the bytes agree", or it means nothing."""
    from services.worker.compare import CLOSE, compare

    verdict = compare('{"y": [1.0], "name": "a"}', '{"name": "a", "y": [1.0]}')
    assert verdict.verdict == CLOSE
    assert "formatting" in verdict.describe()


# --- the resident: which process answers, and what happens when one dies ------


class _Fake(object):
    """A worker with no process behind it.

    Everything the resident does can be driven this way, which is the point of
    keeping the process itself in one small file: the decisions -- ask there,
    answer here, drop a dead one, never let two conversations overlap -- are
    testable without spawning anything.
    """

    def __init__(self, answer=None, raises=None, delay=0.0):
        self.answer = answer or {"ok": True, "plot_json": "{}"}
        self.raises = raises
        self.delay = delay
        self.startup = 0.5
        self.asked = []
        self.stopped = False
        self.overlapped = False
        self._busy = False

    def ask(self, message, timeout=None):
        if self._busy:
            self.overlapped = True
        self._busy = True
        try:
            time.sleep(self.delay)
            self.asked.append(message)
            if self.raises is not None:
                raise self.raises
            return dict(self.answer, id=message.get("id"))
        finally:
            self._busy = False

    def start(self):
        return {"ok": True}

    def stop(self):
        self.stopped = True

    def kill(self):
        self.stopped = True


def _resident(**kwargs):
    """A resident that can never start a real process.

    The default `build` is the real `Worker`, and a resident whose worker dies
    starts a replacement -- so a test that installed a dying fake and said
    nothing else spawned an actual child, which then imported ROSS in the
    background for nine seconds and died on a closed pipe. It passed, it was
    invisible, and it left a process behind on somebody's machine.

    Supplying the fake here rather than in each test means no test in this file
    can make that mistake: spawning has to be asked for, and nothing here asks.
    """
    from services.worker.resident import Resident

    kwargs.setdefault("build", _Fake)
    return Resident(**kwargs)


def test_a_message_goes_to_the_worker_when_there_is_one():
    from services.worker.resident import WORKER

    fake = _Fake()
    resident = _resident(here=lambda message: {"ok": True, "plot_json": "here"})
    resident.use(fake)

    answer, where = resident.ask({"kind": "analysis"})
    assert where == WORKER
    assert answer["plot_json"] == "{}"
    assert len(fake.asked) == 1


def test_without_a_worker_the_same_function_answers_here():
    """A resident that was never started is not a broken application.

    This is the case the whole test suite runs in, and the case of a machine
    where spawning a child fails. The answer has to be a real answer.
    """
    from services.worker.resident import HERE

    resident = _resident(here=lambda message: {"ok": True, "plot_json": "here"})
    answer, where = resident.ask({"kind": "analysis"})
    assert where == HERE
    assert answer["plot_json"] == "here"
    assert resident.detours == 1


def test_a_worker_that_dies_costs_its_request_a_detour_and_not_a_failure():
    """The child is gone; the person at the screen still gets their chart."""
    from services.worker.host import WorkerGone
    from services.worker.resident import HERE, WORKER

    fake = _Fake(raises=WorkerGone("it died"))
    resident = _resident(
        build=lambda: _Fake(), here=lambda message: {"ok": True, "plot_json": "here"}
    )
    # Started properly, and only then handed the dying worker. A resident that
    # was merely *handed* a worker is not allowed to start another -- see
    # `test_a_resident_given_a_worker_never_starts_one_of_its_own` -- so the
    # replacement at the end of this test is a property of a resident that asked
    # for a worker, which is the one the application has.
    resident.start_in_background()
    for _ in range(100):
        if resident.working:
            break
        time.sleep(0.01)
    assert resident.working, "the first worker never arrived"
    resident.use(fake)

    answer, where = resident.ask({"kind": "analysis"})
    assert where == HERE, "a dead worker must not take the request down with it"
    assert answer["plot_json"] == "here"
    assert resident.detours == 1
    assert "it died" in (resident.trouble or ""), (
        "the reason has to survive, or a worker that never runs is invisible"
    )
    assert not resident.working, "the corpse is still installed"

    # And a replacement arrives, so the next analysis is not a detour too.
    for _ in range(100):
        if resident.working:
            break
        time.sleep(0.01)
    assert resident.working, "no replacement was started"
    assert resident.ask({"kind": "analysis"})[1] == WORKER


def test_two_threads_never_share_the_pipe():
    """One stdin, one answer queue, and no addressing between them.

    Two overlapping conversations swap each other's answers, and on a chart
    route that means someone looking at the result of a question they did not
    ask, with nothing on screen to say so -- FE-05 of the audit, on the server
    side this time.
    """
    fake = _Fake(delay=0.02)
    resident = _resident()
    resident.use(fake)

    threads = [
        threading.Thread(target=lambda: resident.ask({"kind": "analysis"}))
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(fake.asked) == 6
    assert not fake.overlapped, "two requests were inside the worker at once"


def test_the_conversation_ends_when_an_answer_belongs_to_another_question():
    """Control for the lock: if it ever fails, this is what it would look like.

    A mismatched identifier is not recoverable -- the pipe is out of step -- so
    it has to become a loud failure rather than a chart handed to the wrong
    question.
    """
    from services.worker.host import Worker, WorkerGone

    class _Pipe(object):
        def write(self, text):
            pass

        def flush(self):
            pass

    class _Process(object):
        stdin = _Pipe()

    worker = Worker(command=["never", "run"])
    worker.process = _Process()
    worker._answers.put({"id": 4321, "ok": True})

    try:
        worker.ask({"kind": "ping"}, timeout=1.0)
    except WorkerGone as error:
        assert "out of step" in "%s" % error, error
    else:
        raise AssertionError("an answer to another question was accepted")


def test_a_refusal_keeps_the_exception_class_it_had():
    """The error contract must not change because the computation moved.

    `ValueError` is how ROSS refuses a model and how this application refuses a
    field, and `api/errors.py` turns it into a 400 carrying the message. Across
    a pipe an exception is a name and a string; if the name were dropped, every
    "Add at least one Shaft!" would reach the user as a 500.
    """
    from services.worker.resident import WorkerFailure, raise_if_refused

    try:
        raise_if_refused(
            {"ok": False, "failure": "ValueError", "error": "Add at least one Shaft!"}
        )
    except ValueError as error:
        assert "%s" % error == "Add at least one Shaft!"
    else:
        raise AssertionError("a refused model no longer raises ValueError")

    assert raise_if_refused({"ok": True, "plot_json": "{}"})["plot_json"] == "{}"


def test_a_refusal_this_process_cannot_name_still_says_what_it_was():
    """Control: the lookup must not turn an unknown class into a silent success.

    A ROSS exception has no name in `builtins`, and the fallback has to carry
    the original name into the message -- otherwise the 500 blames the
    transport for something the computation did.
    """
    from services.worker.resident import WorkerFailure, raise_if_refused

    try:
        raise_if_refused(
            {
                "ok": False,
                "failure": "RotorCrashed",
                "error": "the shaft is upside down",
            }
        )
    except WorkerFailure as error:
        assert "RotorCrashed" in "%s" % error
        assert "upside down" in "%s" % error
    else:
        raise AssertionError("an unnameable refusal passed through")


def test_the_way_around_the_worker_is_the_function_the_worker_runs():
    """Not a second implementation: the same `handle`, in another place.

    A fallback that computed the chart its own way would be the copy kept by
    hand that this whole refactoring exists to remove, and it would drift in
    exactly the case nobody exercises -- the machine where the worker fails to
    start.
    """
    body = _without_prose(_source("services", "worker", "resident.py"))
    assert "from services.worker.protocol import handle" in body, (
        "the resident no longer answers with the worker's own function"
    )
    for computing in ("figure_json", "mode_shape_json", "get_runner", "import ross"):
        assert computing not in body, (
            "the resident is computing on its own (%s)" % computing
        )


def test_the_route_hands_the_work_over_instead_of_doing_it():
    """After 6c the route chooses nothing: not where the work happens, not when.

    Read per route and not as text. The first version of this guard asked
    whether `raise_if_refused` appeared **anywhere** in the file -- and the line
    `from services.worker.resident import raise_if_refused` satisfied that, so
    deleting every call left it green. That is exactly the change that turns
    each of ROSS's refusals into a 500 about an internal error.

    The routes are found by their decorator rather than counted, because the
    count is the thing slice 6c changed: two routes became three when the
    waiting moved out of the request.
    """
    tree = ast.parse(_source("api", "analysis.py"))

    routes = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and (
                getattr(decorator.func, "attr", "") == "route"
            ):
                routes[decorator.args[0].value] = node

    assert sorted(routes) == [
        "/api/campbell/mode_shape",
        "/api/jobs/<job_id>",
        "/run_analysis",
    ], "the routes of this blueprint changed: %s" % sorted(routes)

    def calls(node):
        return {
            getattr(call.func, "id", None) or getattr(call.func, "attr", None)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        }

    for path in ("/run_analysis", "/api/campbell/mode_shape"):
        made = calls(routes[path])
        assert "submit" in made, "%s no longer puts the work in the queue" % path
        assert not made & {"figure_json", "mode_shape_json"}, (
            "%s still computes directly" % path
        )

    asking = calls(routes["/api/jobs/<job_id>"])
    assert "raise_if_refused" in asking, (
        "the job route no longer turns a refusal back into its own exception, "
        "so every message from ROSS would reach the user as a 500"
    )


def test_the_application_asks_for_a_worker_and_the_factory_does_not():
    """`create_app()` is called by the dozen in the suite; `main()` runs once.

    If the resident were started where applications are built, every test that
    builds one would spawn a three-hundred-megabyte child.
    """
    entry = _without_prose(_source("app.py"))
    factory = _without_prose(_source("api", "__init__.py"))
    assert "RESIDENT.start_in_background()" in entry, "nobody starts the worker"
    assert "RESIDENT.stop()" in entry, (
        "nobody stops it, and it outlives the window it belonged to"
    )
    assert "start_in_background" not in factory, (
        "building an application now spawns a process: the suite would spawn "
        "one per test"
    )


def test_a_detour_leaves_a_line_in_the_log():
    """A fallback nobody can see is a program that is quietly slower than it should be.

    The detour is correct -- the chart still arrives -- which is exactly what
    makes it dangerous: on a machine where the worker never starts, nothing on
    screen and nothing in the output would ever say so, and the whole of slice 6
    would be doing nothing at all.
    """
    import logging

    from services.worker.host import WorkerGone

    kept = []

    class _Keep(logging.Handler):
        def emit(self, record):
            kept.append(record.getMessage())

    logger = logging.getLogger("ross_interface")
    handler = _Keep()
    logger.addHandler(handler)
    try:
        resident = _resident(here=lambda message: {"ok": True})
        resident.use(_Fake(raises=WorkerGone("it died")))
        resident.ask({"kind": "analysis"})
    finally:
        logger.removeHandler(handler)

    assert any("it died" in line for line in kept), (
        "the detour left no trace anywhere: %s" % kept
    )


def test_the_suite_never_writes_into_the_application_log():
    """`ross_interface.log` is evidence, and the suite must not forge any.

    It lives with the worker tests because the worker is what produces the most
    convincing forgeries: a test that kills a fake worker on purpose writes "the
    worker was lost", and a fake with no process behind it reports being ready
    in half a second. Read later by somebody diagnosing a real machine, those
    lines say the product is broken when nothing is.

    The mechanism is in `conftest.py`, and it leans on `configure_logging()`
    returning early when the logger already has a handler -- so both halves are
    checked here: that nothing is writing to the file now, and that the early
    return the trick depends on is still in the code.
    """
    import logging

    writing = [
        handler
        for handler in logging.getLogger("ross_interface").handlers
        if os.path.basename(getattr(handler, "baseFilename", ""))
        == "ross_interface.log"
    ]
    assert writing == [], (
        "the suite is writing into the log the user reads: %s" % writing
    )

    conftest = _without_prose(_source("conftest.py"))
    assert 'logging.getLogger("ross_interface").addHandler' in conftest, (
        "nothing claims the application logger before the tests run, so the "
        "suite's own noise goes into the user's log file"
    )

    errors = _without_prose(_source("api", "errors.py"))
    assert "if logger.handlers:" in errors, (
        "configure_logging no longer returns early when the logger is already "
        "configured, so claiming it in conftest stopped working"
    )


# --- the contract the async routes now have -----------------------------------

ASYNC_ROUTES = ("/run_analysis", "/api/campbell/mode_shape")

# Everything in the repository, and that width is the point. The first version
# of this guard swept `tests/` only -- and the caller it could not see was
# `selftest.py`, which ships inside the executable and is what CI runs to decide
# whether a build works. It went on reading the old contract, called all twelve
# analyses failed, and took the package job down on three systems. `check.bat`
# stayed green throughout, because it does not build.
SKIPPED_FOLDERS = ("__pycache__", ".git", "build", "dist", "node_modules", ".venv")


def _python_files():
    for folder, folders, names in os.walk(ROOT):
        folders[:] = [f for f in folders if f not in SKIPPED_FOLDERS]
        for name in sorted(names):
            if name.endswith(".py"):
                yield os.path.relpath(os.path.join(folder, name), ROOT)


def _is_post_to_an_async_route(node):
    """`<something>.post("/run_analysis", ...)`, at any depth of a call chain."""
    while isinstance(node, ast.Call):
        if getattr(node.func, "attr", None) == "post" and node.args:
            first = node.args[0]
            return isinstance(first, ast.Constant) and first.value in ASYNC_ROUTES
        node = getattr(node.func, "value", None)
    return False


def _reads_the_body(node, name):
    """`name.json`, `name.get_json()` -- the body of a response called `name`."""
    if isinstance(node, ast.Attribute) and node.attr == "json":
        return getattr(node.value, "id", None) == name
    if isinstance(node, ast.Call):
        inner = node.func
        if isinstance(inner, ast.Attribute) and inner.attr == "get_json":
            return getattr(inner.value, "id", None) == name
    return False


def _names_in(function):
    """What each local name holds: the POST's response, or the POST's body.

    One hop of following, which is as far as these files ever go: either the
    body is taken on the same line as the post, or the response is kept and the
    body read from it on the next.
    """
    responses, bodies = set(), set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if isinstance(value, ast.BoolOp) and value.values:
            value = value.values[0]  # `x = resp.get_json() or {}`
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not targets:
            continue
        if any(_reads_the_body(value, held) for held in responses):
            bodies.update(targets)
        elif isinstance(value, ast.Call) and _is_post_to_an_async_route(value):
            taken_here = getattr(value.func, "attr", None) in ("json", "get_json")
            (bodies if taken_here else responses).update(targets)
        elif isinstance(value, ast.Attribute) and value.attr == "json":
            if getattr(value.value, "id", None) in responses:
                bodies.update(targets)
    return responses, bodies


def _asks_for(node, name, key):
    """`name["key"]` or `name.get("key")`."""
    if (
        isinstance(node, ast.Subscript)
        and getattr(node.value, "id", None) == name
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == key
    ):
        return True
    return (
        isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "get"
        and getattr(getattr(node.func, "value", None), "id", None) == name
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == key
    )


def _expecting_a_chart_from_the_request():
    """Every place that posts to an async route and then reads the old contract."""
    guilty = []
    for relative in _python_files():
        try:
            tree = ast.parse(_source(relative))
        except SyntaxError:  # not ours to judge
            continue
        for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            responses, bodies = _names_in(function)
            if not responses and not bodies:
                continue
            where = "%s::%s" % (relative.replace(os.sep, "/"), function.name)
            for inner in ast.walk(function):
                for held in responses:
                    if (
                        isinstance(inner, ast.Compare)
                        and isinstance(inner.left, ast.Attribute)
                        and inner.left.attr == "status_code"
                        and getattr(inner.left.value, "id", None) == held
                        and any(
                            isinstance(c, ast.Constant) and c.value == 200
                            for c in inner.comparators
                        )
                    ):
                        guilty.append("%s expects 200 from the request" % where)
                    if (
                        isinstance(inner, ast.Subscript)
                        and isinstance(inner.value, ast.Attribute)
                        and inner.value.attr == "json"
                        and getattr(inner.value.value, "id", None) == held
                        and isinstance(inner.slice, ast.Constant)
                        and inner.slice.value == "plot_json"
                    ):
                        guilty.append("%s reads plot_json from the request" % where)
                for held in bodies:
                    if _asks_for(inner, held, "plot_json"):
                        guilty.append("%s reads plot_json from the request" % where)
                    if (
                        isinstance(inner, ast.Compare)
                        and _asks_for(inner.left, held, "status")
                        and any(
                            isinstance(c, ast.Constant) and c.value == "success"
                            for c in inner.comparators
                        )
                    ):
                        guilty.append("%s expects success from the request" % where)
    return sorted(set(guilty))


def test_nothing_expects_a_chart_from_the_request_that_asks_for_one():
    """Since 6c a POST to those routes answers a job. It never answers a chart.

    This guard exists twice over. First because five tests in four files went on
    expecting the old contract and I only looked in the file I was editing --
    the project's sixth lesson, in a new place. Then because the version written
    for *that* swept `tests/` only, and the caller it missed was `selftest.py`:
    bundled into the executable, run by CI to decide whether a build works, and
    reading the old contract all along. Twelve analyses reported as failed, the
    package job down on three systems, and every local check green.

    So the sweep is the repository, not a folder. `tests/`, `tools/`, the
    selftest, anything: whoever posts must wait, and `api/waiting.py` is the one
    place that knows how.

    What it still cannot catch is a test expecting a *domain* refusal to be
    synchronous. No mechanical rule separates "the envelope said no" from "the
    rotor said no" without knowing the domain.
    """
    guilty = _expecting_a_chart_from_the_request()
    assert guilty == [], (
        "a POST to an async route answers a job, not a chart:\n  " + "\n  ".join(guilty)
    )


def test_the_sweep_reaches_beyond_the_test_folder():
    """Control, and it is the whole correction.

    A sweep that silently stopped finding the shipped callers would make the
    guard above pass for nothing -- which is exactly the state that let the
    selftest break CI. So it is not enough that the sweep finds *something*: it
    has to find files that are not tests.
    """
    posting = set()
    for relative in _python_files():
        tree = ast.parse(_source(relative))
        for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and _is_post_to_an_async_route(node):
                    posting.add(relative.replace(os.sep, "/"))

    assert posting, "the sweep no longer finds any post to an async route"
    assert "tests/test_requests.py" in posting, (
        "the file that defines the route contract is not among them: %s"
        % sorted(posting)
    )
    assert "selftest.py" in posting, (
        "the caller that ships inside the executable is not among them: %s"
        % sorted(posting)
    )
    outside = {p for p in posting if not p.startswith("tests/")}
    assert len(outside) >= 2, (
        "only one caller outside the suite was found, and there are more: %s"
        % sorted(posting)
    )


# --- the check's own reasoning ------------------------------------------------


def test_the_check_asks_the_worker_to_forget_before_asking_again():
    """The repeat has to be a computation, not the cache handing back an answer.

    This is the defect the timing probe already paid for once: a "second run"
    that was really a cache read, and a conclusion drawn from it. Here it would
    be worse than a wrong number -- it would answer "does this process compute
    the same thing twice?" with "yes" every single time, and the divergence
    would be blamed on the pipe forever.
    """
    check = _without_prose(_source("services", "worker", "check.py"))
    protocol = _without_prose(_source("services", "worker", "protocol.py"))
    assert '{"kind": "forget"}' in check, (
        "the check no longer empties the worker's cache before repeating a "
        "question, so the repeat measures the cache"
    )
    assert '"forget": _forget' in protocol, "the worker cannot forget anything"
    assert "ANALYSIS_CACHE.clear()" in protocol, "forgetting no longer clears anything"


def test_the_check_compares_charts_through_the_comparator():
    """Control for the guard above: the byte comparison must be gone.

    Both could be true at once -- a check that asks for a repeat and still
    compares with `!=` would report the same false accusation it did before.
    """
    check = _without_prose(_source("services", "worker", "check.py"))
    assert "compare(here, answer[" in check, "the check is not using the comparator"
    assert 'answer["plot_json"] != here' not in check, (
        "the byte comparison is still there, and it is what accuses the worker "
        "of noise it did not create"
    )


# --- killing a worker on purpose (slice 6c-2) ---------------------------------


class _Killable(object):
    """A worker that stops when it is killed, and not before -- like a real one.

    A real child in the middle of an eigenvalue solve answers nothing and cannot
    be asked to stop; the pipe goes quiet only when the process dies. That is the
    shape being reproduced here, because it is the shape that makes the two
    decisions below necessary.
    """

    def __init__(self):
        self.startup = 0.0
        self.reached = threading.Event()
        self.killed = False
        self._over = threading.Event()

    def ask(self, message, timeout=None):
        from services.worker.host import WorkerGone

        self.reached.set()
        self._over.wait(5.0)
        raise WorkerGone("the worker ended before answering")

    def start(self):
        return {"ok": True}

    def stop(self):
        self._over.set()

    def kill(self):
        self.killed = True
        self._over.set()


def test_a_worker_killed_on_purpose_does_not_send_the_work_back_here():
    """Otherwise the stop button would do the opposite of what it says.

    A worker that dies is a detour: the request is answered in this process so
    the user still gets their chart. Apply that rule to a deliberate kill and the
    button becomes a ten-second recomputation of the very analysis somebody just
    asked to stop -- the same wait, now with no way out of it at all.

    The control for this is
    `test_a_worker_that_dies_costs_its_request_a_detour_and_not_a_failure`: a
    death nobody asked for must still detour, or this rule would have eaten it.
    """
    from services.worker.resident import Interrupted

    computed_here = []

    def here(message):
        computed_here.append(message)
        return {"ok": True, "plot_json": "here"}

    resident = _resident(here=here)
    worker = _Killable()
    resident.use(worker)

    interrupted = []

    def asking():
        try:
            resident.ask({"kind": "analysis"})
        except Interrupted as error:
            interrupted.append(error)

    thread = threading.Thread(target=asking)
    thread.start()
    assert worker.reached.wait(5.0), "the fake worker never got the message"

    assert resident.interrupt() is True
    thread.join(5.0)

    assert worker.killed, "the worker was not killed"
    assert interrupted, "the interrupted request did not say it had been"
    assert computed_here == [], (
        "this process recomputed the analysis the user cancelled: %s" % computed_here
    )


class _Blocking(object):
    """A worker that keeps the conversation open until the test lets it go.

    Unlike `_Killable` it does **not** end when killed: it is here to hold the
    resident's lock, which is the thing the next test is about.
    """

    def __init__(self):
        self.startup = 0.0
        self.reached = threading.Event()
        self.release = threading.Event()

    def ask(self, message, timeout=None):
        self.reached.set()
        self.release.wait(5.0)
        return {"ok": True, "plot_json": "{}", "id": message.get("id")}

    def start(self):
        return {"ok": True}

    def stop(self):
        self.release.set()

    def kill(self):
        pass


def test_interrupting_does_not_wait_for_the_computation_it_is_interrupting():
    """The trap this file is one line away from at all times.

    `ask` holds the lock for the whole computation -- that is decision 3, and it
    is right. An `interrupt` that took the same lock would be correct, obvious,
    and useless: it would wait for the analysis to finish and then announce that
    it had been stopped. The user would press the button and watch nothing
    happen for ten seconds.

    Run in a thread with a deadline so that the failure is a failure and not a
    suite that hangs.
    """
    resident = _resident()
    worker = _Blocking()
    resident.use(worker)

    asking = threading.Thread(target=lambda: resident.ask({"kind": "analysis"}))
    asking.start()
    assert worker.reached.wait(5.0), "the fake worker never got the message"

    returned = threading.Event()

    def interrupting():
        resident.interrupt()
        returned.set()

    threading.Thread(target=interrupting).start()
    assert returned.wait(2.0), (
        "interrupt waited for the lock `ask` holds: it can only stop a "
        "computation after that computation has ended"
    )

    worker.release.set()
    asking.join(5.0)


def test_a_resident_given_a_worker_never_starts_one_of_its_own():
    """Found while writing the route guard for the stop button, and it was
    already there.

    `use()` promises a worker "without starting a process", and it kept that
    promise exactly until the worker died. Then the replacement rule -- right for
    the application, which asked for a worker -- started a **real** child in a
    suite that never asked for one: `tests/test_requests.py` installs a fake that
    raises `WorkerGone`, and every run of that test was spawning a process that
    imported ROSS for nine seconds and died on a closed pipe. Green, invisible,
    on the machine of whoever ran the suite and in CI.

    It is the same defect as `_resident`'s default `build` (lesson 24), in the
    file I did not look at. The rule now says what it always meant: **the
    resident starts a process only when somebody asked it to start one**, and
    that somebody is `start_in_background`.
    """
    built = []

    def build():
        built.append(1)
        return _Fake()

    from services.worker.host import WorkerGone
    from services.worker.resident import HERE, Resident

    resident = Resident(build=build, here=lambda message: {"ok": True})
    resident.use(_Fake(raises=WorkerGone("the worker died")))

    answer, where = resident.ask({"kind": "analysis"})
    assert where == HERE, "the detour did not happen"
    time.sleep(0.05)  # a replacement would be started on a thread
    assert built == [], "a worker was spawned for a resident that was handed one"


def test_the_worker_the_button_kills_is_replaced():
    """Otherwise the stop button ends the worker for the rest of the session.

    Every analysis after it would be computed in this process -- slower, all of
    them, with nothing on screen to say why. The user would have pressed a button
    that said it would cost the next analysis a startup, and it would have cost
    them every analysis instead.
    """
    from services.worker.resident import Resident

    built = []

    def build():
        built.append(1)
        return _Fake()

    resident = Resident(build=build, here=lambda message: {"ok": True})
    resident.start_in_background()
    for _ in range(100):
        if resident.working:
            break
        time.sleep(0.01)
    assert resident.working, "the first worker never arrived"

    assert resident.interrupt() is True
    for _ in range(200):
        if len(built) > 1:
            break
        time.sleep(0.01)
    assert len(built) > 1, "the killed worker was never replaced"
