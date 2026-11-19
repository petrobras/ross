# -*- coding: utf-8 -*-
"""The transient plots' time window, from ROSS #1416.

ROSS's `plot_1d`, `plot_2d` and `plot_3d` take `t_initial`, `t_final` and
`one_revolution` and crop the response before drawing. The interface offers
them as three plot fields of the four transient analyses -- plot fields, so
moving the window redraws without integrating again.

Two things here are not obvious and are what these tests hold:

* **the unit.** A plain number is read in `time_units` by `plot_1d` and in
  seconds by `plot_2d` and `plot_3d`. The fields say seconds, so they reach
  ROSS as `Q_(value, 's')`, which keeps its unit in all three;
* **the names.** The faults already have `t_initial`/`t_final` -- the
  integration time. The window is `window_start`/`window_end` on the form, and
  the refusals name those, not ROSS's keywords.
"""

import ast
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ross.interface.domain.analysis_catalog import ANALYSES
from ross.interface.domain.python_export import build_script
from ross.interface.services.analysis import REGISTRY
from ross.units import Q_

TRANSIENTS = ("time_response", "misalignment", "rubbing", "crack")
WINDOW = ("window_start", "window_end", "one_revolution")


@pytest.mark.parametrize("name", TRANSIENTS)
def test_the_four_transients_offer_the_window_on_the_time_plots(name):
    by_id = {field["id"]: field for field in ANALYSES[name]}
    for field_id in WINDOW:
        field = by_id[field_id]
        assert field["role"] == "plot"
        # The DFFT takes no window in ROSS, so the fields hide with it.
        assert field["deps"] == ["1D", "2D", "3D"]
        assert field["deps_de"] == "plot_type"
    assert by_id["one_revolution"]["options"] == ["False", "True"]


def window(**params):
    return REGISTRY["time_response"].time_window(params)


def test_an_empty_window_passes_nothing():
    assert window() == {}
    assert window(window_start="", window_end="  ", one_revolution="False") == {}


def test_the_times_go_as_seconds_and_zero_is_a_time():
    got = window(window_start="0", window_end="0.25")
    assert got == {"t_initial": Q_(0.0, "s"), "t_final": Q_(0.25, "s")}


def test_one_revolution_takes_one_boundary_or_none():
    assert window(one_revolution="True") == {"one_revolution": True}
    got = window(window_end="0.2", one_revolution="True")
    assert got == {"t_final": Q_(0.2, "s"), "one_revolution": True}


def test_half_a_window_is_refused_by_the_form_s_names():
    with pytest.raises(ValueError, match="Window Start.*Window End"):
        window(window_start="0.1")


def test_a_revolution_with_both_boundaries_is_refused_by_the_form_s_names():
    with pytest.raises(ValueError, match="One Revolution"):
        window(window_start="0.1", window_end="0.2", one_revolution="True")


# --- against ROSS -------------------------------------------------------------

SPEED = 200.0  # rad/s: one revolution is 2*pi/200 = 0.0314 s


@pytest.fixture(scope="module")
def response():
    import ross as rs

    rotor = rs.rotor_example()
    t = np.linspace(0, 0.1, 401)
    forces = np.zeros((len(t), rotor.ndof))
    forces[:, rotor.number_dof * 3] = 10 * np.cos(SPEED * t)
    return rotor.run_time_response(SPEED, forces, t)


def plot(response, **params):
    params.setdefault("probes", [{"node": 3, "angle": 0}])
    return REGISTRY["time_response"].plot(response, params, None)


def test_the_1d_plot_is_cropped_in_seconds_whatever_the_time_unit(response):
    """`time_units='min'` changes the axis, not where the window falls."""
    seconds = plot(response, plot_type="1D", window_start="0.02", window_end="0.06")
    x = np.asarray(seconds.data[0].x)
    assert x.min() == pytest.approx(0.02) and x.max() == pytest.approx(0.06)

    minutes = plot(
        response,
        plot_type="1D",
        time_units="min",
        window_start="0.02",
        window_end="0.06",
    )
    x = np.asarray(minutes.data[0].x)
    assert x.min() == pytest.approx(0.02 / 60) and x.max() == pytest.approx(0.06 / 60)


def test_the_orbit_draws_only_the_window(response):
    whole = plot(response, plot_type="2D")
    cropped = plot(response, plot_type="2D", window_start="0.02", window_end="0.06")
    assert len(whole.data[0].x) == 401
    # 401 samples over 0.1 s: one every 0.25 ms, so 0.02..0.06 holds 161.
    assert len(cropped.data[0].x) == 161


def test_one_revolution_draws_one_turn_of_the_shaft(response):
    figure = plot(response, plot_type="1D", one_revolution="True")
    x = np.asarray(figure.data[0].x)
    assert x.max() - x.min() == pytest.approx(2 * np.pi / SPEED, abs=0.25e-3)
    assert x.max() == pytest.approx(0.1)


def test_the_dfft_ignores_the_window(response):
    """ROSS's `plot_dfft` takes no window; the fields are hidden with it and,
    if a value is left behind from another chart, not passed on."""
    plot(
        response,
        plot_type="Frequency (DFFT)",
        window_start="0.02",
        one_revolution="True",
    )


# --- the exported script ------------------------------------------------------


def script_for(**params):
    analysis = {
        "type": "time_response",
        "params": {
            "speed": "200",
            "t_max": "0.1",
            "steps": "401",
            "forces": [{"node": 3, "dof": 0, "func": "10 * np.cos(speed * t)"}],
            "probes": [{"node": 3, "angle": 0}],
            **params,
        },
    }
    return build_script({}, [analysis])


def test_the_script_crops_the_same_window_in_seconds():
    script = script_for(plot_type="2D", window_start="0", window_end="0.06")
    ast.parse(script)
    assert "plot_2d(node=3, t_initial=Q_(0, 's'), t_final=Q_(0.06, 's'))" in script


def test_the_script_asks_for_one_revolution():
    script = script_for(plot_type="1D", one_revolution="True")
    assert "one_revolution=True" in script
    assert "t_initial" not in script.split("plot_1d")[1]


def test_the_script_leaves_the_window_out_of_the_dfft():
    script = script_for(
        plot_type="Frequency (DFFT)", window_start="0", window_end="0.06"
    )
    assert "Q_(0.06, 's')" not in script
