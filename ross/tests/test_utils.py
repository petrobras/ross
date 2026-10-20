import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from plotly import graph_objects as go

import ross
from ross.rotor_assembly import rotor_example
from ross.units import Q_
from ross.utils import downsample_figure, speed_range_from_endpoints


@pytest.fixture(params=["tsdownsample", "numpy"])
def downsampler(request, monkeypatch):
    """Run the test with tsdownsample and with the numpy fallback.

    Setting the module to None in sys.modules makes importing it raise
    ImportError, as on an installation without the optional dependency.
    """
    if request.param == "tsdownsample":
        pytest.importorskip("tsdownsample")
    else:
        monkeypatch.setitem(sys.modules, "tsdownsample", None)
    return request.param


def test_speed_range_from_endpoints_default_num_points():
    spans = np.array([10.0, 100.0, 1000.0, 10000.0])
    num_points = np.array([len(speed_range_from_endpoints(0, span)) for span in spans])
    assert_equal(num_points, [64, 200, 633, 2000])
    assert np.all(np.diff(num_points) > 0)
    assert np.all(np.diff(num_points / spans) < 0)

    assert len(speed_range_from_endpoints(0, 0.5)) == 20
    assert len(speed_range_from_endpoints(0, 1e6)) == 2000
    assert len(speed_range_from_endpoints(0, 400, points_factor=4)) == 80


def test_speed_range_from_endpoints_units():
    speed_range = speed_range_from_endpoints(
        Q_(0, "Hz"), Q_(3000, "RPM"), num_points=31
    )
    assert_allclose(speed_range, np.linspace(0, 100 * np.pi, 31))

    speed_range = speed_range_from_endpoints(Q_(10, "Hz"), 100 * np.pi, num_points=5)
    assert_allclose(speed_range[[0, -1]], [20 * np.pi, 100 * np.pi])


def test_speed_range_from_endpoints_errors():
    with pytest.raises(ValueError, match="greater than"):
        speed_range_from_endpoints(100, 100)
    with pytest.raises(ValueError, match="at least 2"):
        speed_range_from_endpoints(0, 100, num_points=1)


def test_speed_range_from_endpoints_in_analyses():
    rotor = rotor_example()
    speed_range = speed_range_from_endpoints(Q_(0, "RPM"), Q_(50, "Hz"), num_points=11)
    expected = np.linspace(0, 100 * np.pi, 11)

    assert_allclose(rotor.run_campbell(speed_range).speed_range, expected)
    assert_allclose(
        rotor.run_unbalance_response(3, 1e-4, 0, speed_range=speed_range).speed_range,
        expected,
    )


def test_downsample_figure_keeps_the_original_samples(downsampler):
    x = np.linspace(0, 1, 200_001)
    y = np.sin(2 * np.pi * 7 * x) + 0.1 * np.sin(2 * np.pi * 400 * x)
    y[123_457] = 5.0
    y[170_000] = -4.0
    x_short = np.arange(100.0)
    y_short = np.cos(x_short)
    fig = go.Figure([go.Scatter(x=x, y=y), go.Scatter(x=x_short, y=y_short)])

    fig = downsample_figure(fig, 1000)

    x_out = np.asarray(fig.data[0].x)
    y_out = np.asarray(fig.data[0].y)
    assert 0 < len(x_out) <= 1000
    assert np.all(np.diff(x_out) > 0)
    assert x_out[0] == x[0] and x_out[-1] == x[-1]
    assert_equal(y_out, y[np.searchsorted(x, x_out)])
    assert y_out.max() == 5.0
    assert y_out.min() == -4.0
    assert_equal(fig.data[1].x, x_short)
    assert_equal(fig.data[1].y, y_short)


def test_downsample_figure_without_n_out_returns_the_figure(downsampler):
    x = np.linspace(0, 1, 5000)
    fig = go.Figure(go.Scatter(x=x, y=x**2))

    assert downsample_figure(fig, None) is fig
    assert_equal(fig.data[0].x, x)


@pytest.mark.parametrize("n_out", [0, 1, 2, 3, 4])
def test_downsample_figure_keeps_four_points_at_least(downsampler, n_out):
    rng = np.random.default_rng(1)
    x = np.arange(5000.0)
    fig = go.Figure(go.Scatter(x=x, y=rng.normal(size=x.size)))

    fig = downsample_figure(fig, n_out)

    x_out = np.asarray(fig.data[0].x)
    assert 2 <= len(x_out) <= 4
    assert x_out[0] == x[0] and x_out[-1] == x[-1]


@pytest.mark.parametrize("n_out", [0, 1, 2, 3, 4])
def test_downsample_figure_without_tsdownsample_keeps_the_extremes(monkeypatch, n_out):
    monkeypatch.setitem(sys.modules, "tsdownsample", None)
    rng = np.random.default_rng(1)
    x = np.arange(5000.0)
    y = rng.normal(size=x.size)
    fig = go.Figure(go.Scatter(x=x, y=y))

    fig = downsample_figure(fig, n_out)

    y_out = np.asarray(fig.data[0].y)
    assert len(y_out) == 4
    assert y_out.max() == y.max() and y_out.min() == y.min()


@pytest.mark.parametrize("n_out", [4, 5, 7, 50])
def test_downsample_figure_returns_at_most_n_out_points(downsampler, n_out):
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 1, 997))
    y = rng.normal(size=x.size)
    fig = go.Figure(go.Scatter(x=x, y=y))

    fig = downsample_figure(fig, n_out)

    assert len(fig.data[0].x) <= n_out


def test_downsample_figure_without_tsdownsample_keeps_every_peak(monkeypatch):
    """The fallback keeps the lowest and highest sample of each bucket, so
    each period of a sine keeps its own maximum and minimum."""
    monkeypatch.setitem(sys.modules, "tsdownsample", None)
    n_periods = 50
    samples_per_period = 2000
    x = np.arange(n_periods * samples_per_period) / samples_per_period
    y = np.sin(2 * np.pi * x) * (1 + x)
    fig = go.Figure(go.Scatter(x=x, y=y))

    fig = downsample_figure(fig, 1000)

    x_out = np.asarray(fig.data[0].x)
    y_out = np.asarray(fig.data[0].y)
    period = np.floor(x).astype(int)
    period_out = np.floor(x_out).astype(int)
    for k in range(n_periods):
        assert y_out[period_out == k].max() == y[period == k].max()
        assert y_out[period_out == k].min() == y[period == k].min()


def test_import_ross_without_tsdownsample():
    code = "\n".join(
        [
            "import sys",
            "sys.modules['tsdownsample'] = None",
            "import numpy as np",
            "from plotly import graph_objects as go",
            "import ross",
            "x = np.linspace(0, 1, 10_000)",
            "fig = go.Figure(go.Scatter(x=x, y=np.sin(40 * x)))",
            "fig = ross.utils.downsample_figure(fig, 500)",
            "assert len(fig.data[0].x) <= 500",
            "print('ross_file=' + ross.__file__)",
        ]
    )
    package_root = str(Path(ross.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [package_root] + [p for p in [env.get("PYTHONPATH")] if p]
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )

    assert result.returncode == 0, result.stderr
    # Other packages imported by ross (ccp) may print to stdout.
    lines = [
        line for line in result.stdout.splitlines() if line.startswith("ross_file=")
    ]
    assert len(lines) == 1, result.stdout
    assert (
        Path(lines[0].removeprefix("ross_file=")).resolve()
        == Path(ross.__file__).resolve()
    )
