# -*- coding: utf-8 -*-
"""The loop inside the worker process: read a line, answer a line.

Deliberately thin. Everything that can be decided without a process was decided
in `protocol.py`; what is left here is the part the test suite cannot reach, and
the less of it there is, the less is unverified.

WHY THE FIRST THING IT DOES IS TAKE STDOUT AWAY FROM EVERYONE ELSE. ROSS prints.
`run_crack` and `run_rubbing` write "Running direct method" to standard output
while they compute, and warnings go there too depending on how the interpreter
is configured. On this channel that is not noise -- it is corruption: a line of
ROSS's prose between two protocol lines makes the parent read a chart as a
syntax error.

So the real stdout is captured once, at startup, and `sys.stdout` is pointed at
stderr. Anything the library prints from then on lands in stderr, where the
parent can log it and nothing depends on its shape. The protocol writes to the
descriptor that was taken aside, which nothing else can reach.

WHY LINES AND JSON, and not `pickle` or `multiprocessing`. Two of the twelve
ROSS results cannot be pickled -- one holds a lambda defined inside
`run_campbell`, the other a bound `lru_cache` wrapper -- so anything that
serialises Python objects across this boundary would fail for them. Text does
not have that problem, and it has a second virtue: a person can read the
traffic.
"""

import json
import sys


def serve(incoming, outgoing, answer=None):
    """Answer messages until the incoming stream ends.

    Takes the two streams as arguments so a test can hand it two `StringIO` and
    exercise the loop -- the framing, the blank lines, the broken JSON -- with
    no process and no pipe. `answer` is injectable, and imported lazily when it
    is not given, for the same reason: `protocol.handle` reaches ROSS, so a
    module-level import would make this loop unreachable to any test that runs
    without it. The loop and the work it dispatches to are separate failures and
    deserve separate coverage.
    """
    if answer is None:
        from services.worker.protocol import handle as answer
    for line in incoming:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError as error:
            reply = {
                "id": None,
                "ok": False,
                "failure": "ProtocolError",
                "error": "the line is not JSON: %s" % error,
            }
        else:
            reply = answer(message)
        outgoing.write(json.dumps(reply) + "\n")
        outgoing.flush()


def main():
    """Run as the child: `<the executable> --worker`."""
    protocol_out = sys.stdout
    sys.stdout = sys.stderr  # from here on, whatever ROSS prints is stderr's problem

    if hasattr(protocol_out, "reconfigure"):
        protocol_out.reconfigure(encoding="utf-8", newline="\n")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    serve(sys.stdin, protocol_out)
    return 0


if __name__ == "__main__":  # pragma: no cover - the child is started by the parent
    sys.exit(main())
