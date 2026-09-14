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
* any other exception -> **500**, naming the type. It still shows the detail,
  because this is a local single-user application and hiding it would protect
  nobody -- but the status code now tells the truth about whose fault it is.
"""

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
        base = (
            os.path.dirname(sys.executable)
            if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        logger.exception("unexpected error in %s", request.path)
        return jsonify(
            {
                "status": "error",
                "message": "Unexpected error (%s): %s" % (type(error).__name__, error),
            }
        ), 500
