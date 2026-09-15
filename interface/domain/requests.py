# -*- coding: utf-8 -*-
"""The shape of every request body, written once.

Until this point each route read the body with loose `.get()` calls. A wrong
field name -- `analysisType` instead of `analysis_type` -- raised nothing at
all: it became `None` and the analysis carried on with the default value. And
the refusal of the pre-Phase-2 format was an `if` written by hand inside app.py.

Here the two become one thing. The envelope declares the fields it accepts, and
**refuses any key that is not declared** -- including the old format's, which is
now recognised by name with no special case.

Pydantic was considered for this, as the decisions document had planned. Two
reasons for a hand-written validator instead: ROSS does not ship Pydantic (it
would be a new dependency inside the PyInstaller executable, for fewer than ten
fields), and the environment where these files are written cannot execute it --
the validation layer would be the only one reaching the user unverified.
"""

from .cache import STRUCTURAL_KEYS

TYPE_NAMES = {dict: "an object", list: "a list", str: "a string"}


class Field:
    """One envelope field: name, type, and what to do when it is missing."""

    def __init__(self, name, kind, required=False, default=None):
        self.name = name
        self.kind = kind
        self.required = required
        self.default = default


class Envelope:
    """Read and validate a route's request body."""

    def __init__(self, name, *fields):
        self.name = name
        self.fields = fields
        self.declared = {field.name for field in fields}

    def read(self, body):
        """Return a dict of the declared fields, or raise ValueError."""
        if body is None:
            body = {}
        if not isinstance(body, dict):
            raise ValueError("The body of %s must be a JSON object." % self.name)

        self._refuse_unknown_keys(body)

        read_fields = {}
        for field in self.fields:
            value = body.get(field.name)
            if value is None:
                if field.required:
                    raise ValueError(
                        "Field '%s' is required in %s and was not sent."
                        % (field.name, self.name)
                    )
                read_fields[field.name] = (
                    dict(field.default)
                    if isinstance(field.default, dict)
                    else list(field.default)
                    if isinstance(field.default, list)
                    else field.default
                )
                continue

            if not isinstance(value, field.kind):
                raise ValueError(
                    "Field '%s' of %s must be %s; got %s."
                    % (
                        field.name,
                        self.name,
                        TYPE_NAMES.get(field.kind, field.kind.__name__),
                        type(value).__name__,
                    )
                )
            read_fields[field.name] = value
        return read_fields

    def _refuse_unknown_keys(self, body):
        unknown_keys = [key for key in body if key not in self.declared]
        if not unknown_keys:
            return

        # A structural key at the root is the pre-Phase-2 format. It is worth
        # naming the problem: before, a body like this left the project empty
        # and the response complained about the rotor -- a message about the
        # model for a contract error, which sends you looking in the wrong
        # place.
        from_old_format = [c for c in unknown_keys if c in STRUCTURAL_KEYS]
        if from_old_format:
            raise ValueError(
                "Body in the old format: the project now goes inside "
                "{'project': {...}}, not loose at the root (found %s outside)."
                % ", ".join(sorted(from_old_format))
            )

        raise ValueError(
            "Unknown field in %s: %s. Accepted: %s."
            % (
                self.name,
                ", ".join(sorted(unknown_keys)),
                ", ".join(sorted(self.declared)),
            )
        )


ROTOR_REQUEST = Envelope(
    "/build_rotor",
    Field("project", dict, default={}),
)

# `key` names the subject a request belongs to -- one card on the screen. The
# server uses it to take an older job for the same subject out of the queue
# before it runs: until slice 6c the browser aborted its own `fetch` and the
# computation carried on regardless, so a card run twice made the second
# analysis wait for the first, which nobody wanted any more.
#
# It is optional, and a request without one is simply never superseded.
ANALYSIS_REQUEST = Envelope(
    "/run_analysis",
    Field("analysis_type", str, required=True),
    Field("params", dict, default={}),
    Field("conversion_type", str, default=""),
    Field("project", dict, default={}),
    Field("key", str, default=""),
)

MODE_SHAPE_REQUEST = Envelope(
    "/api/campbell/mode_shape",
    Field("params", dict, default={}),
    Field("conversion_type", str, default=""),
    Field("project", dict, default={}),
    Field("point", dict, required=True),
    Field("key", str, default=""),
)

EXPORT_REQUEST = Envelope(
    "/api/export/python",
    Field("project", dict, default={}),
    Field("analyses", list, default=[]),
    Field("conversion_type", str, default=""),
)

ROSS_FILE_REQUEST = Envelope(
    "/load_ross_file",
    Field("content", str, required=True),
)
