# -*- coding: utf-8 -*-
"""How an error reaches the user (BE-10).

Before, every route ended with the same block:

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400

Three problems. **Every error became a 400**, including a defect of ours -- the
client had no way to tell "you sent something invalid" from "it broke in here".
The message was the `str()` of some exception, so a `KeyError: 'odl'` reached
the screen as `'odl'`, with no context. And `traceback.print_exc()` writes to
`sys.stdout`, which **does not exist** in a console-less PyInstaller executable
-- precisely the distribution form chosen for Phase 4.

Now it is centralised, and the split is by type:

* `ValueError` -> **400**, with the message. This is the channel through which
  ROSS refuses a model ("Add at least one Shaft!", "Rotor has no bearings") and
  through which this application's validators refuse a field. Those messages
  help whoever is at the screen, and they still arrive whole.
* an exception **ROSS raises on purpose** -> **400** as well, with its message
  and nothing added. ROSS does not refuse only through `ValueError`: "Each
  rotor needs a GearElement in the coupled nodes!" is a `TypeError`, and it
  used to reach the screen as "Unexpected error (TypeError): ...", as though
  the program had broken. "On purpose" is read off the traceback, not guessed
  from the type: the innermost frame is in ROSS's own code (not this
  interface's) and the instruction running there is a `raise`. A `TypeError`
  from arithmetic inside ROSS -- a real crash -- is not a `raise` and stays a
  500. So is an `assert` failing in ROSS, although it compiles to a `raise`.
  The instruction is read from the bytecode, not from the source line,
  because the packaged executable ships without sources.
* any other exception -> **500**, naming the type. It still shows the detail,
  because this is a local single-user application and hiding it would protect
  nobody -- but the status code now tells the truth about whose fault it is.
"""

import dis
import logging
import os
import sys

from flask import jsonify, request
from werkzeug.exceptions import HTTPException

logger = logging.getLogger("ross_interface")


def configure_logging():
    """Send the log to a file, and to the console when there is one.

    In a packaged executable without a console there is no stderr to write to,
    and a lost traceback is the difference between an investigable defect and a
    report saying "it didn't work".
    """
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    try:
        # Frozen, next to the executable; from source, in the directory the
        # program was started from -- never inside the installed package,
        # which is site-packages and not a place a log belongs.
        base = (
            os.path.dirname(sys.executable)
            if getattr(sys, "frozen", False)
            else os.getcwd()
        )
        handle = logging.FileHandler(
            os.path.join(base, "ross_interface.log"), encoding="utf-8"
        )
        handle.setFormatter(fmt)
        logger.addHandler(handle)
    except OSError:
        pass  # read-only folder: the console is what is left, if any

    if sys.stderr is not None:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

    return logger


_RAISE = dis.opmap["RAISE_VARARGS"]


def raised_by_ross(error):
    """Whether ROSS itself raised this, with a `raise` of its own.

    A heuristic, and its limit is that a `raise` is not always a refusal. An
    `except` block in ROSS that wraps a crash in an exception of its own, or
    re-raises it, ends in a `raise` too, and reaches the screen as a 400. The
    one case told apart here is `assert`: it compiles to a `raise` of
    `AssertionError`, and a failed assertion is an invariant of ROSS broken,
    which is a crash and stays a 500.
    """
    if isinstance(error, AssertionError):
        return False
    frame = error.__traceback__
    if frame is None:
        return False
    while frame.tb_next is not None:
        frame = frame.tb_next
    module = str(frame.tb_frame.f_globals.get("__name__", ""))
    if module != "ross" and not module.startswith("ross."):
        return False
    if module == "ross.interface" or module.startswith("ross.interface."):
        return False
    code = frame.tb_frame.f_code.co_code
    return 0 <= frame.tb_lasti < len(code) and code[frame.tb_lasti] == _RAISE


def register_handlers(app):
    """Install the handlers; with them, routes need no try/except."""
    configure_logging()

    @app.errorhandler(ValueError)
    def _input_refused(error):
        logger.info("input refused at %s: %s", request.path, error)
        return jsonify({"status": "error", "message": str(error)}), 400

    @app.errorhandler(HTTPException)
    def _http_error(error):
        # 404, 405 and the like keep the status Flask chose.
        return jsonify({"status": "error", "message": error.description}), error.code

    @app.errorhandler(Exception)
    def _unexpected_error(error):
        if raised_by_ross(error):
            logger.info("ROSS refused at %s: %s", request.path, error)
            return jsonify({"status": "error", "message": str(error)}), 400
        logger.exception("unexpected error in %s", request.path)
        return jsonify(
            {
                "status": "error",
                "message": "Unexpected error (%s): %s" % (type(error).__name__, error),
            }
        ), 500
