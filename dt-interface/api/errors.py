# -*- coding: utf-8 -*-
"""How an error reaches the browser.

`ValueError` is a refused input and answers 400. Any other exception answers
500 and names its type. Routes do not catch exceptions themselves.
"""

import logging

from flask import jsonify
from werkzeug.exceptions import HTTPException

logger = logging.getLogger("dt_interface")


def register_handlers(app):
    """Install the handlers."""
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    @app.errorhandler(ValueError)
    def _input_refused(error):
        logger.info("input refused at %s: %s", request_path(), error)
        return jsonify({"status": "error", "message": str(error)}), 400

    @app.errorhandler(HTTPException)
    def _http_error(error):
        return jsonify({"status": "error", "message": error.description}), error.code

    @app.errorhandler(Exception)
    def _unexpected_error(error):
        logger.exception("unexpected error")
        return jsonify(
            {
                "status": "error",
                "message": "Unexpected error (%s): %s" % (type(error).__name__, error),
            }
        ), 500


def request_path():
    """The path of the current request, or an empty string outside one."""
    from flask import request

    return request.path if request else ""
