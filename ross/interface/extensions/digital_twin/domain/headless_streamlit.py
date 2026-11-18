# -*- coding: utf-8 -*-
"""Run the Digital Twin rotor models without Streamlit.

The models in `models_files/` were written for the Streamlit app: they import
`streamlit`, draw a few widgets in `get_pi_data` (the tag table), report
progress with `st.warning` / `st.info`, and read three values from
`st.session_state`:

* `local`: True runs the labyrinth-seal subprocess with `sys.executable`;
* `id_session`: the folder where the seal subprocess writes its files;
* a few more keys the model writes and never reads back here.

`module()` returns a stand-in module that answers those calls without a
browser: a text input returns its default value, containers do nothing, and
the messages are collected so the route can return them. `install()` puts it
in `sys.modules["streamlit"]`, so a model's `import streamlit as st` receives
it whether or not Streamlit is installed. The ROSS interface does not use
Streamlit itself.

A model that reports the same message `MAX_REPEATS` times is stopped with
`ModelLoop`: that is how the seal models retry a failing subprocess forever.

Only the calls the models make are provided. A model that calls anything else
fails with an `AttributeError` naming the call, instead of doing nothing
silently.
"""

import sys
import threading
import types
from contextlib import contextmanager

_STATE = threading.local()
_RUN_LOCK = threading.Lock()


class SessionState(dict):
    """`st.session_state`: a dict that also answers attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class _Message:
    """What `st.info` and `st.warning` return: the models call `.empty()` on it."""

    def empty(self):
        return None


class _Container:
    """`st.expander` and the columns of `st.columns`: a no-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text_input(self, label, value="", **kwargs):
        return value

    def write(self, *args, **kwargs):
        return None

    def empty(self):
        return None


def _messages():
    if not hasattr(_STATE, "messages"):
        _STATE.messages = []
    return _STATE.messages


class ModelLoop(RuntimeError):
    """A model reported the same message too many times in one run."""


# The seal models retry their subprocess forever when it fails, reporting
# the same restart message each time. The same message this many times in one run
# stops the model instead of letting the request hang.
MAX_REPEATS = 3


def _report(level):
    def report(body, *args, **kwargs):
        text = str(body)
        messages = _messages()
        messages.append({"level": level, "text": text})
        if sum(item["text"] == text for item in messages) >= MAX_REPEATS:
            raise ModelLoop(
                "The model reported %r %d times and was stopped. Its "
                "subprocess is probably failing: run it by hand to see why."
                % (text, MAX_REPEATS)
            )
        return _Message()

    report.__name__ = level
    return report


def _fragment(function=None, **kwargs):
    if function is None:
        return lambda wrapped: wrapped
    return function


def _columns(spec, **kwargs):
    count = spec if isinstance(spec, int) else len(spec)
    return [_Container() for _ in range(count)]


def module():
    """Return the stand-in `streamlit` module."""
    stand_in = types.ModuleType("streamlit")
    stand_in.__doc__ = "Headless stand-in installed by the ROSS interface."
    stand_in.session_state = SessionState()
    stand_in.fragment = _fragment
    stand_in.expander = lambda *args, **kwargs: _Container()
    stand_in.columns = _columns
    stand_in.text_input = _Container().text_input
    stand_in.write = _Container().write
    stand_in.table = lambda *args, **kwargs: None
    for level in ("info", "warning", "error", "success"):
        setattr(stand_in, level, _report(level))
    return stand_in


STREAMLIT = module()


def install():
    """Make `import streamlit` return the stand-in."""
    sys.modules["streamlit"] = STREAMLIT
    return STREAMLIT


@contextmanager
def model_run(session_directory):
    """Run a model call with a fresh session state and collect its messages.

    Yields the list the messages are appended to, as
    `{"level": "warning", "text": "..."}`. The session state is one object
    shared by every model module, so model runs take turns.
    """
    with _RUN_LOCK:
        install()
        STREAMLIT.session_state.clear()
        STREAMLIT.session_state.update(
            {"local": True, "id_session": session_directory, "status": False}
        )
        _STATE.messages = []
        try:
            yield _STATE.messages
        finally:
            STREAMLIT.session_state.clear()
