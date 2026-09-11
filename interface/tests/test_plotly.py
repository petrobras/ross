# -*- coding: utf-8 -*-
"""The figure goes from ROSS to the screen without hand-written serialisation.

Two things are guaranteed here. The plotting library the interface serves is
the one **of the installed plotly.py**, and not a downloaded copy: a figure
built by one version and drawn by another is an error that only shows up in one
specific chart, in one specific browser.

And the conversion to JSON is plotly's own (`fig.to_json()`). Before, there
were three hand-written functions producing non-strict JSON -- a bare `NaN`,
which `JSON.parse` refuses -- and the defect arrived as a chart that does not
appear."""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from app import app
from api import paths as app_module


def _decode_plotly_array(value):
    """Gives back the numbers of an array serialised by plotly.

    plotly.py 6 serialises numpy arrays as base64 typed arrays
    ({'dtype': 'f8', 'bdata': ...}) and plain Python lists as a JSON list.
    This helper accepts both shapes."""
    if isinstance(value, dict) and "bdata" in value:
        import base64
        import struct

        raw = base64.b64decode(value["bdata"])
        fmt = {"f8": "d", "f4": "f"}[value["dtype"]]
        size = struct.calcsize(fmt)
        return list(struct.unpack(f"<{len(raw) // size}{fmt}", raw))
    return value


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_plotly_bundle_comes_from_the_installed_package(client):
    """The interface serves plotly.js from inside the installed plotly.py."""
    import plotly

    assert app_module.PLOTLY_BUNDLE.startswith(os.path.dirname(plotly.__file__))
    assert os.path.exists(app_module.PLOTLY_BUNDLE)

    response = client.get("/lib/plotly.min.js")
    assert response.status_code == 200
    with open(app_module.PLOTLY_BUNDLE, "rb") as fh:
        assert response.get_data() == fh.read()


def test_figure_to_json_is_strict_json():
    """The basis of the move to to_json(): strict JSON out, and the values survive.

    It is what decode_bdata + remove_nans + NumpyEncoder did by hand. Plotly
    serialises the numpy array as a base64 typed array, so the NaN travels inside
    the binary -- and plotly.js draws NaN as a gap, just like null."""
    import math

    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(y=np.array([1.0, np.nan, np.inf, 3.0])))
    text = figure.to_json()

    def refuse(constant):
        raise AssertionError(f"JSON that is not strict: token {constant}")

    # parse_constant is only called for bare NaN/Infinity/-Infinity. If none shows
    # up, the browser's JSON.parse will not break either.
    payload = json.loads(text, parse_constant=refuse)

    y = _decode_plotly_array(payload["data"][0]["y"])
    assert y[0] == 1.0
    assert y[3] == 3.0
    assert y[1] is None or math.isnan(y[1])
    assert y[2] is None or math.isinf(y[2])


def test_figure_to_json_handles_plain_python_lists():
    """A Python list with NaN also has to come out as strict JSON."""
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(y=[1.0, float("nan"), float("inf"), 3.0]))

    def refuse(constant):
        raise AssertionError(f"JSON that is not strict: token {constant}")

    payload = json.loads(figure.to_json(), parse_constant=refuse)
    y = _decode_plotly_array(payload["data"][0]["y"])
    assert y[0] == 1.0 and y[3] == 3.0


def test_run_analysis_serialisation_helpers_are_gone():
    """The three hand-written functions went away with the move to to_json().

    After slice 4 they could not come back anywhere, so the search became the
    whole of the server code."""
    import glob

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for dead_one in ("NumpyEncoder", "decode_bdata", "remove_nans", "get_int"):
        for path in (
            glob.glob(os.path.join(root_path, "*.py"))
            + glob.glob(os.path.join(root_path, "api", "*.py"))
            + glob.glob(os.path.join(root_path, "domain", "*.py"))
            + glob.glob(
                os.path.join(root_path, "services", "**", "*.py"), recursive=True
            )
        ):
            with open(path, encoding="utf-8") as handle:
                assert "def %s" % dead_one not in handle.read(), "%s voltou em %s" % (
                    dead_one,
                    path,
                )
