# -*- coding: utf-8 -*-
"""Entry point of the ROSS interface.

What used to be a 1190-line file -- routes, rotor assembly, expression
evaluator, cache, the twelve analyses and reading ROSS files, all together --
became three layers:

    api/       transport: routes, envelope, token, error handling
    services/  what the application does: the analysis runners, the evaluator
    domain/    what the application knows: units, nodes, ROSS classes, schema,
               rotor assembly, cache, script export

What is left here is what only makes sense in the executable: starting the
server, opening the browser, and answering `--selftest`.
"""

import multiprocessing
import sys
import threading
import time
import webbrowser

# `freeze_support` before anything heavy, and that is why the import below is
# not at the top. Inside a PyInstaller bundle on Windows, a child process
# re-runs this file from the beginning; without this line, anything that starts
# a process makes the executable open itself again, and again. It costs one
# line and one `noqa`; the failure it prevents looks like the program launching
# forever with no error at all.
multiprocessing.freeze_support()

# Worker mode is decided here, above everything, and for a reason of the same
# family as the line above: by the end of this module body the Flask
# application has been built and the session token minted, and a worker needs
# neither. The child is this same executable with `--worker` -- see
# `services/worker/host.py` for why it is a mode of the program and not a
# separate one.
if "--worker" in sys.argv:
    from services.worker.child import main as serve_as_worker  # noqa: E402

    sys.exit(serve_as_worker())

from api import HOST, PORT, create_app  # noqa: E402

app = create_app()


def open_browser():
    time.sleep(2.0)
    webbrowser.open("http://%s:%s/" % (HOST, PORT))


def main():
    if "--selftest" in sys.argv:
        import selftest

        return selftest.run()

    # The one question about the worker that only the built executable can
    # answer: whether a frozen program can start a second copy of itself at all.
    # It lives here, and not in `tools/`, so the same check runs from source and
    # from the bundle -- inside the bundle there is no `python` to run a script
    # with.
    if "--worker-check" in sys.argv:
        from services.worker.check import run

        return run()

    # The worker is asked for here and not inside `create_app()`, and the
    # difference is the test suite: it builds applications by the dozen, and a
    # `create_app()` that spawned a three-hundred-megabyte child would make the
    # suite unusable. This function runs once, in the real program.
    #
    # It does not wait. Importing ROSS in the child costs about eight and a half
    # seconds; the browser opens in two, and the page has to be there by then.
    # Whatever arrives before the worker is ready is computed in this process --
    # the same function in another place. See services/worker/resident.py.
    from services.worker.resident import RESIDENT

    RESIDENT.start_in_background()
    threading.Thread(target=open_browser, daemon=True).start()
    try:
        app.run(host=HOST, port=PORT, use_reloader=False)
    finally:
        # A worker outliving the window it belonged to is three hundred
        # megabytes the user cannot see and did not ask for.
        RESIDENT.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
