"""Convergence diagnostics, work budget and operating table of the journal engine.

References used here, all independent of the code under test:

- the engine's own convergence tests, re-evaluated from the returned outputs
  (force balance ``|F_hyd + F_ext| / |F_ext|`` against ``2e-3``);
- the convergence status of each stored solver fixture, read from its own
  outputs (force balance, loop residuals);
- closed forms: Sommerfeld number, specific load, Reynolds numbers, the film
  thickness of a 360 deg land held at eccentricity ``eps``, and Petroff's
  power loss of a concentric journal.
"""

import functools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.film.loops import ConvergenceWarning
from ross.bearings.fluid_film import constants, driver
from ross.bearings.fluid_film import hydrodynamics as hyd
from ross.bearings.fluid_film.diagnostics import (
    LOOP_NAMES,
    REASONS,
    CaseBudget,
    FluidFilmConvergenceError,
    LoopMonitor,
    default_budget,
)
from ross.bearings.fluid_film.driver import run_case
from ross.bearings.fluid_film_bearing import FluidFilmBearing
from ross.bearings.plain_journal import PlainJournal
from ross.units import Q_

DATA_DIR = Path(__file__).parent / "data" / "fluid_film"
CASE_NAMES = sorted(p.stem for p in DATA_DIR.glob("*.json"))
SLOW_CASES = {"fixed_adiabatic_highpamb", "tilt_5pad_full"}

# Convergence status of the stored fixtures, checked case by case. Every
# fixture converges except ``fixed_isoviscous_pdam``, whose pinned outputs are
# the last iterate of a Newton search that gave up: the film force is 159 kN
# for a 113 kN load.
EXPECTED_FAILURES = {"fixed_isoviscous_pdam": ("diverged", "newton")}

GRID = dict(total_ex_film=20, total_ez_film=10, total_ey_film=10, total_ey_pad=10)

# A two-pad plain journal, isoviscous, carried here from light to beyond capacity.
HEAVY_JOURNAL = dict(
    n=0,
    pad_axial_length=0.05,
    journal_diameter=0.1,
    radial_clearance=100e-6,
    n_pads=2,
    pad_arc=Q_(160, "deg"),
    oil_supply_temperature=Q_(40, "degC"),
    speed=Q_([3000], "RPM"),
    lubricant="ISOVG32",
    oil_flow_v=Q_(10, "l/min"),
    thermal_type=None,
    **GRID,
)

# A 5-pad load-between-pads tilting pad.
TILTING_PAD = dict(
    n=1,
    speed=Q_([3000], "RPM"),
    thermal_type=None,
    journal_diameter=101.6e-3,
    radial_clearance=74.9e-6,
    pad_thickness=12.7e-3,
    pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
    pad_arc=Q_([60] * 5, "deg"),
    pad_axial_length=[50.8e-3] * 5,
    preload=[0.5] * 5,
    offset=[0.5] * 5,
    lubricant="ISOVG32",
    oil_supply_temperature=Q_(40, "degC"),
    oil_flow_v=Q_(10, "l/min"),
    equilibrium_type="match_load",
    **GRID,
)


def _fixture_inputs(name):
    return json.loads((DATA_DIR / f"{name}.json").read_text())["inputs"]


def _numeric_keys(outputs):
    keys = []
    for key, value in outputs.items():
        if key in ("convergence", "fields"):
            continue
        flat = np.atleast_1d(np.asarray(value, dtype=object)).ravel()
        if not any(isinstance(v, str) for v in flat):
            keys.append(key)
    return keys


def _force_residual(inputs, outputs):
    """The engine's equilibrium test, re-evaluated from the outputs."""
    fx_ext = float(np.ravel(inputs["fxs_load"])[0])
    fy_ext = float(np.ravel(inputs["fys_load"])[0]) - float(
        np.ravel(inputs["weight"])[0]
    )
    fx_net = outputs["fx_hydro"][0] + fx_ext
    fy_net = outputs["fy_hydro"][0] + fy_ext
    return np.hypot(fx_net, fy_net) / np.hypot(fx_ext, fy_ext)


@pytest.mark.parametrize(
    "case_name",
    [
        pytest.param(name, marks=pytest.mark.slow) if name in SLOW_CASES else name
        for name in CASE_NAMES
    ],
)
def test_fixture_status_and_numbers(case_name):
    """Every fixture reports its expected status, and the flag changes no number."""
    inputs = _fixture_inputs(case_name)
    with_diagnostics = run_case(**inputs, diagnostics=True)
    plain = run_case(**inputs)

    # without the flag: the pinned key set and the empty message
    assert set(with_diagnostics) == set(plain) | {"convergence"}
    assert plain["non_convergence"] == [""]
    for key in _numeric_keys(plain):
        assert_allclose(
            np.asarray(with_diagnostics[key], dtype=float),
            np.asarray(plain[key], dtype=float),
            rtol=0,
            atol=0,
            err_msg=key,
        )

    summary = with_diagnostics["convergence"][0]
    assert set(summary["loops"]) <= set(LOOP_NAMES)
    if case_name in EXPECTED_FAILURES:
        reason, loop = EXPECTED_FAILURES[case_name]
        assert not summary["converged"]
        assert summary["reason"] == reason
        assert with_diagnostics["non_convergence"][0].startswith(f"{loop}: {reason}")
    else:
        assert summary["converged"], summary["message"]
        assert summary["reason"] == ""
        assert with_diagnostics["non_convergence"] == [""]
    assert summary["reynolds_solves"] >= 1

    newton = summary["loops"]["newton"]
    if inputs["equilibrium_type"] == "match_load":
        # the recorded residual is the engine's force test on the returned state
        assert_allclose(newton["residual"], _force_residual(inputs, plain), rtol=1e-9)
        assert newton["converged"] == (newton["residual"] < 2e-3)
    else:
        assert newton["reason"] == "not_applicable"


def test_tolerance_scale_one_is_the_default_solve():
    """``_tolerance_scale=1`` reproduces the default solve bit for bit.

    The default solve is pinned against the stored fixtures by
    ``test_fluid_film_solver.py``.
    """
    for case in ("fixed_adiabatic", "fixed_full_coarse", "tilt_5pad_isoviscous"):
        inputs = _fixture_inputs(case)
        scaled = run_case(**inputs, _tolerance_scale=1.0, diagnostics=True)
        plain = run_case(**inputs)
        for key in _numeric_keys(plain):
            assert_allclose(
                np.asarray(scaled[key], dtype=float),
                np.asarray(plain[key], dtype=float),
                rtol=0,
                atol=0,
                err_msg=f"{case}/{key}",
            )


def test_tolerance_scale_tightens_the_equilibrium():
    """A smaller scale drives the force residual below the scaled tolerance."""
    inputs = _fixture_inputs("fixed_isoviscous")
    for scale in (1.0, 1e-1, 1e-3):
        outputs = run_case(**inputs, diagnostics=True, _tolerance_scale=scale)
        newton = outputs["convergence"][0]["loops"]["newton"]
        assert newton["tolerance"] == pytest.approx(2e-3 * scale, rel=1e-15)
        assert _force_residual(inputs, outputs) < 2e-3 * scale


@pytest.mark.parametrize("load", [2e4, 1e5, 4e5])
def test_heavy_load_journal_converges_on_the_force_balance(load):
    """At 4e5 N the engine flag says diverged, the force balance says converged.

    The force residual recomputed from the outputs is below the engine's
    tolerance 2e-3, so the case is reported converged.
    """
    bearing = PlainJournal(**HEAVY_JOURNAL, fys_load=-load)
    assert bool(bearing.results.converged[0])
    op = bearing.operating_results.iloc[0]
    fy = bearing.results.outputs[0]["fy_hydro"][0]
    fx = bearing.results.outputs[0]["fx_hydro"][0]
    assert_allclose(op.residual, np.hypot(fx, fy - load) / load, rtol=1e-9)
    assert op.residual < 2e-3
    if load == 4e5:
        assert_allclose(op.residual, 1.76e-3, rtol=1e-2)


def test_overloaded_journal_stops_on_the_budget():
    """An adiabatic plain journal far beyond capacity stops on its budget.

    Left alone, the Newton search of this case pushes the journal into the
    bore after about 1,300 Reynolds solves (reason ``"contact"``); budgets
    below that stop it first.
    """
    kwargs = dict(HEAVY_JOURNAL, thermal_type="adiabatic", fys_load=-8e5)
    start = time.perf_counter()
    with pytest.warns(ConvergenceWarning, match="budget"):
        bearing = PlainJournal(**kwargs, max_reynolds_solves=500)
    assert time.perf_counter() - start < 60.0
    summary = bearing.results.convergence[0]
    assert not summary["converged"]
    assert summary["reason"] == "budget"
    assert summary["reynolds_solves"] == 500
    assert bearing.operating_results.reason[0] == "budget"

    with pytest.raises(FluidFilmConvergenceError, match="budget"):
        PlainJournal(**kwargs, max_reynolds_solves=200, on_failure="raise")


def test_hang_case_terminates_with_a_reason():
    """The 8e5 N journal that never returned stops with a recorded reason.

    Root cause of the hang: with the journal position beyond the
    clearance no tilt can open the film, and the film loop of ``fixed_brg``
    repeated the same film-thickness evaluation forever.
    """
    start = time.perf_counter()
    with pytest.warns(ConvergenceWarning):
        bearing = PlainJournal(
            **HEAVY_JOURNAL, fys_load=-8e5, max_reynolds_solves=20_000
        )
    assert time.perf_counter() - start < 60.0
    summary = bearing.results.convergence[0]
    assert not summary["converged"]
    assert summary["reason"] in REASONS
    assert summary["message"]


def test_per_call_cap_ends_a_newton_loop_that_never_converges(monkeypatch):
    """The cap per equilibrium solve stops a Newton search that makes no progress."""
    monkeypatch.setattr(hyd, "MAX_ITERATION", 10**9)
    monkeypatch.setattr(hyd, "_newton_step", lambda *args: (0.0, 0.0))
    # 60 per pad, counted as 5 pads: 300 per equilibrium solve
    monkeypatch.setattr(constants, "MAX_REYNOLDS_SOLVES_PER_CALL_PER_PAD", 60)
    outputs = run_case(**_fixture_inputs("fixed_isoviscous"), diagnostics=True)
    summary = outputs["convergence"][0]
    assert summary["reason"] == "budget"
    assert summary["max_call_solves"] == 300
    assert "one equilibrium solve" in summary["message"]
    assert np.isfinite(outputs["kxx"][0])


def test_default_budget_scales_with_the_pad_count():
    """400,000 solves per pad and case, 4,000 per pad and equilibrium solve."""
    assert default_budget(1) == default_budget(5) == (2_000_000, 20_000)
    assert default_budget(10) == (4_000_000, 40_000)
    inputs = _fixture_inputs("tilt_5pad_isoviscous")
    budget = driver._case_budget(dict(inputs, pivot_angle=np.zeros(8)))
    assert (budget.max_reynolds_solves, budget.max_solves_per_call) == (
        3_200_000,
        32_000,
    )
    budget = driver._case_budget(dict(inputs, max_reynolds_solves=1234))
    assert (budget.max_reynolds_solves, budget.max_solves_per_call) == (1234, 20_000)


def test_multi_pad_thermal_tilting_pad_converges_on_the_default_budget():
    """A 4-pad full-TEHD tilting pad needs 320,000 solves and converges.

    A budget of 300,000 solves per case would report this valid bearing as
    failed on ``"budget"``. Its equilibrium is checked
    on the force balance recomputed from the outputs.
    """
    from ross.bearings.tilting_pad import TiltingPad

    pivots = (np.arange(4) * 90.0 + 45.0) % 360.0
    bearing = TiltingPad(
        n=1,
        speed=Q_([12000], "RPM"),
        thermal_type="full",
        journal_diameter=0.2,
        radial_clearance=150e-6,
        pad_thickness=0.02,
        pivot_angle=Q_(list(pivots), "deg"),
        pad_arc=Q_([72.0] * 4, "deg"),
        pad_axial_length=[0.1] * 4,
        preload=[0.3] * 4,
        offset=[0.5] * 4,
        lubricant="ISOVG32",
        oil_supply_temperature=Q_(45, "degC"),
        oil_flow_v=Q_(100, "l/min"),
        equilibrium_type="match_load",
        fys_load=-2e4,
        total_ex_film=8,
        total_ez_film=4,
        total_ey_film=10,
        total_ey_pad=10,
    )
    summary = bearing.results.convergence[0]
    assert summary["converged"], summary["message"]
    assert summary["reynolds_solves"] > 300_000
    assert summary["max_call_solves"] < default_budget(4)[1]
    out = bearing.results.outputs[0]
    assert np.hypot(out["fx_hydro"][0], out["fy_hydro"][0] - 2e4) / 2e4 < 2e-3


def test_stopped_thermal_case_reports_the_last_newton_iterate(monkeypatch):
    """A case stopped inside its first thermal pass is evaluated where the Newton was.

    The thermal loops solve on a copy of the solver state, so before the fix
    the outputs were evaluated at the initial guess. The film
    solves are spied on: the extra solve after the stop must use the
    position of the interrupted one.
    """
    inputs = _fixture_inputs("fixed_adiabatic")
    positions = []
    original = hyd.fixed_brg

    @functools.wraps(original)
    def spy(**kwargs):
        positions.append((kwargs["xj"], kwargs["yj"]))
        return original(**kwargs)

    monkeypatch.setattr(hyd, "fixed_brg", spy)
    outputs = run_case(**inputs, diagnostics=True, max_reynolds_solves=12)
    summary = outputs["convergence"][0]
    assert summary["reason"] == "budget"
    assert summary["hydrodynamics_calls"] == 1  # stopped in the first pass

    cb = float(inputs["radial_clearance"])
    initial = np.hypot(inputs["xj"], inputs["yj"])  # relative to C_b
    last = positions[-2]  # the interrupted film solve; [-1] is the final one
    assert positions[-1] == last
    eccentricity = outputs["eccentricity"][0]
    assert_allclose(eccentricity, np.hypot(*last) / cb, rtol=1e-12)
    assert abs(eccentricity - initial) > 1e-3


def test_time_limit_stops_the_case():
    inputs = _fixture_inputs("tilt_5pad_isoviscous")
    outputs = run_case(**inputs, diagnostics=True, time_limit=1e-9)
    summary = outputs["convergence"][0]
    assert summary["reason"] == "time_limit"
    # the limit is checked before each solve, so not even the first one ran
    assert summary["reynolds_solves"] == 0
    assert "longer than 1e-09 s" in summary["message"]


@pytest.fixture
def three_speed_kwargs():
    from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture

    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    kwargs["speed"] = [80.0, 95.0, 110.0]
    return kwargs


@pytest.fixture
def fail_second_speed(monkeypatch):
    """Give the 95 rad/s case a budget of one Reynolds solve."""
    original = driver._case_budget

    def budget(kw):
        if abs(float(np.ravel(kw["frequency"])[0]) - 95.0) < 1e-9:
            return CaseBudget(max_reynolds_solves=1)
        return original(kw)

    monkeypatch.setattr(driver, "_case_budget", budget)


def test_failed_speed_is_left_out_of_the_table(
    three_speed_kwargs, fail_second_speed, tmp_path
):
    with pytest.warns(ConvergenceWarning, match="left out"):
        bearing = FluidFilmBearing(**three_speed_kwargs)
    results = bearing.results
    assert results.converged.tolist() == [True, False, True]
    assert_allclose(bearing.speed, [80.0, 110.0])
    for name in ("kxx", "kyy", "cxx", "cyy"):
        assert_allclose(
            np.asarray(getattr(bearing, name), dtype=float),
            np.asarray(getattr(results, name))[[0, 2]],
            rtol=0,
            atol=0,
        )
    # every case stays in the results and the operating table
    n_cases = 3
    assert len(results.outputs) == len(results.pressure_fields) == n_cases
    assert len(results.convergence) == np.size(results.kxx) == n_cases
    op = bearing.operating_results
    assert len(op) == n_cases
    assert op.converged.tolist() == [True, False, True]
    assert op.reason.tolist() == ["", "budget", ""]
    assert_allclose(op.speed, [80.0, 95.0, 110.0])

    from ross.bearing_seal_element import BearingElement

    file = tmp_path / "bearing.toml"
    bearing.save(file)
    loaded = BearingElement.load(file)
    assert_allclose(loaded.speed, [80.0, 110.0])
    assert_allclose(np.array(loaded.kxx), np.asarray(results.kxx)[[0, 2]], rtol=1e-12)


def test_failed_speed_keep_and_raise(three_speed_kwargs, fail_second_speed):
    with pytest.warns(ConvergenceWarning, match="every row is kept"):
        kept = FluidFilmBearing(**three_speed_kwargs, on_failure="keep")
    assert_allclose(kept.speed, [80.0, 95.0, 110.0])
    with pytest.raises(FluidFilmConvergenceError, match="speed 95 rad/s"):
        FluidFilmBearing(**three_speed_kwargs, on_failure="raise")


def test_no_converged_speed_keeps_every_row(three_speed_kwargs, monkeypatch):
    monkeypatch.setattr(driver, "_case_budget", lambda kw: CaseBudget(1))
    with pytest.warns(ConvergenceWarning, match="no speed converged"):
        bearing = FluidFilmBearing(**three_speed_kwargs)
    assert_allclose(bearing.speed, [80.0, 95.0, 110.0])
    assert not bearing.results.converged.any()


def test_two_dimensional_table_drops_the_whole_speed_row(monkeypatch):
    from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture

    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    kwargs.pop("speed")
    kwargs.pop("excitation_ratio")
    original = driver._case_budget

    def budget(kw):
        speed = float(np.ravel(kw["frequency"])[0])
        ratio = float(np.ravel(kw["excit_ratios"])[0])
        if abs(speed - 110.0) < 1e-9 and abs(ratio * speed - 50.0) < 1e-9:
            return CaseBudget(max_reynolds_solves=1)
        return original(kw)

    monkeypatch.setattr(driver, "_case_budget", budget)
    with pytest.warns(ConvergenceWarning, match="frequency 50 rad/s"):
        bearing = FluidFilmBearing(
            speed=[80.0, 110.0], frequency=[50.0, 90.0], **kwargs
        )
    assert_allclose(bearing.speed, [80.0])
    assert np.asarray(bearing.kxx).shape == (1, 2)
    assert bearing.results.converged.tolist() == [True, True, False, True]
    assert_allclose(bearing.operating_results.frequency, [50.0, 90.0, 50.0, 90.0])


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(on_failure="ignore"), "on_failure must be one of"),
        (dict(max_reynolds_solves=0), "max_reynolds_solves must be a positive"),
        (dict(max_reynolds_solves=2.5), "max_reynolds_solves must be a positive"),
        (
            dict(max_reynolds_solves=float("inf")),
            "max_reynolds_solves must be a positive",
        ),
        (dict(time_limit=-1.0), "time_limit must be positive"),
    ],
)
def test_solver_controls_are_validated(overrides, match):
    from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture

    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous", **overrides)
    with pytest.raises(ValueError, match=match):
        FluidFilmBearing(**kwargs)


def test_time_limit_accepts_pint():
    from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture

    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous", time_limit=Q_(2, "min"))
    bearing = FluidFilmBearing(**kwargs)
    assert bearing.time_limit == pytest.approx(120.0)


def _isoviscous_lubricant(mu1, mu2, t1=300.0, t2=360.0, rho=870.0):
    return dict(
        liquid_viscosity1=mu1,
        temperature1=t1,
        liquid_viscosity2=mu2,
        temperature2=t2,
        liquid_density=rho,
        liquid_specific_heat=1900.0,
        liquid_thermal_conductivity=0.13,
    )


@pytest.mark.parametrize(
    "rpm, load, supply_temperature",
    [(1500.0, 5e3, 300.0), (3000.0, 2e4, 320.0), (6000.0, 5e4, 345.0)],
)
def test_operating_table_against_hand_formulas(rpm, load, supply_temperature):
    """Sommerfeld, specific load and Reynolds numbers from their definitions."""
    diameter, clearance, length = 0.1, 100e-6, 0.06
    mu1, mu2, t1, t2, rho = 0.05, 0.01, 300.0, 360.0, 870.0
    bearing = PlainJournal(
        n=0,
        pad_axial_length=length,
        journal_diameter=diameter,
        radial_clearance=clearance,
        n_pads=2,
        pad_arc=Q_(160, "deg"),
        oil_supply_temperature=supply_temperature,
        speed=Q_([rpm], "RPM"),
        fys_load=-load,
        lubricant=_isoviscous_lubricant(mu1, mu2, t1, t2, rho),
        oil_flow_v=Q_(10, "l/min"),
        thermal_type=None,
        **GRID,
    )
    op = bearing.operating_results.iloc[0]
    # two-point exponential law, written out
    mu = mu1 * np.exp(np.log(mu2 / mu1) * (supply_temperature - t1) / (t2 - t1))
    n_rev = rpm / 60.0
    radius = diameter / 2
    assert_allclose(
        op.sommerfeld,
        mu * n_rev * length * diameter / load * (radius / clearance) ** 2,
        rtol=1e-12,
    )
    # the engine's own Sommerfeld output (another code path) agrees
    assert_allclose(op.sommerfeld, bearing.results.outputs[0]["x_som_m"][0], rtol=1e-12)
    assert_allclose(op.specific_load, load / (length * diameter), rtol=1e-12)
    surface_speed = 2 * np.pi * n_rev * radius
    assert_allclose(op.re_cb, rho * surface_speed * clearance / mu, rtol=1e-12)
    assert_allclose(op.re_taylor, 41.2 * np.sqrt(radius / clearance), rtol=1e-12)
    # local film Reynolds number: the film is thicker than C_b on part of the land
    assert op.re_max > op.re_cb
    assert op.converged


@pytest.mark.parametrize("load", [5e3, 2e4])
def test_p_max_is_reported_for_the_isoviscous_model(load):
    bearing = PlainJournal(**HEAVY_JOURNAL, fys_load=-load)
    op = bearing.operating_results.iloc[0]
    assert bearing.results.outputs[0]["max_pressure"][0] == 0.0
    assert op.p_max == np.max(bearing.results.pressure_fields[0])
    # the nodal maximum of the engine's other output path agrees
    assert_allclose(op.p_max, bearing.results.outputs[0]["y_max_p"][0], rtol=1e-12)
    # a film that carries W over L D peaks above the mean pressure
    assert op.p_max > op.specific_load
    assert np.isnan(op.T_max_pad) and np.isnan(op.T_max_film)


def _concentric_land(eps, rpm=3000.0, mu=0.03, n_ex=40):
    return FluidFilmBearing(
        n=0,
        speed=[rpm * np.pi / 30.0],
        journal_diameter=0.1,
        radial_clearance=100e-6,
        pad_thickness=0.01,
        pivot_angle=[np.pi],
        pad_arc=[2 * np.pi],
        pad_axial_length=[0.2],
        preload=[0.0],
        offset=[0.5],
        lubricant=_isoviscous_lubricant(mu, 0.5 * mu),
        oil_supply_temperature=300.0,
        oil_flow_v=1e-4,
        thermal_type=None,
        equilibrium_type="match_eccentricity",
        initial_position=(0.0, -eps),
        total_ex_film=n_ex,
        total_ez_film=10,
        total_ey_film=10,
        total_ey_pad=10,
    )


@pytest.mark.parametrize("eps", [0.01, 0.1, 0.3, 0.6])
def test_h_min_of_a_land_held_at_eccentricity(eps):
    """``h = C (1 - eps cos(theta))``: the minimum is ``C (1 - eps)``."""
    bearing = _concentric_land(eps)
    op = bearing.operating_results.iloc[0]
    assert_allclose(op.h_min, 100e-6 * (1 - eps), rtol=1e-12)
    assert_allclose(op.eccentricity, eps, rtol=1e-12)
    assert_allclose(op.yj, -eps * 100e-6, rtol=1e-12)
    assert np.isnan(op.residual) and op.converged
    # no applied load, so no load direction
    assert np.isnan(op.attitude_from_load)


@pytest.mark.parametrize("rpm", [1500.0, 3000.0])
def test_power_loss_of_a_nearly_concentric_journal_is_petroff(rpm, monkeypatch):
    """Petroff ``P = 2 pi mu w**2 R**3 L / C`` for L/D 2, laminar, eps 0.01.

    The engine scales the shear dissipation downstream of the minimum film by
    the film fraction ``q_out / q_x`` (``hydrodynamics.cavitation_scale``),
    which for this land is ``h_exit / h = 1 - O(eps)``; the test first checks
    that the scale stays within ``[1 - eps, 1]``, so the comparison with the
    full-film Petroff value is a 1 % statement at ``eps = 0.01``.
    """
    eps, mu = 0.01, 0.03
    scales = []
    original = hyd.cavitation_scale

    def capture(*args, **kwargs):
        scale = original(*args, **kwargs)
        scales.append(np.array(scale))
        return scale

    monkeypatch.setattr(hyd, "cavitation_scale", capture)
    bearing = _concentric_land(eps, rpm=rpm, mu=mu)
    film = scales[-1][0][scales[-1][0] > 0]
    assert film.min() >= 1 - eps - 1e-12 and film.max() <= 1 + 1e-12

    op = bearing.operating_results.iloc[0]
    assert op.re_max < 500  # laminar
    omega, radius, length, clearance = rpm * np.pi / 30, 0.05, 0.2, 100e-6
    petroff = 2 * np.pi * mu * omega**2 * radius**3 * length / clearance
    assert_allclose(op.power_loss, petroff, rtol=0.02)


@pytest.mark.parametrize("case", ["fixed_adiabatic", "fixed_full_coarse"])
def test_every_loop_is_recorded_in_thermal_cases(case):
    outputs = run_case(**_fixture_inputs(case), diagnostics=True)
    loops = outputs["convergence"][0]["loops"]
    assert set(loops) == set(LOOP_NAMES)
    for name, record in loops.items():
        assert record["iterations"] >= 1, name
        assert record["converged"], name
        assert record["total_iterations"] >= record["iterations"], name


@pytest.mark.slow
def test_every_loop_is_recorded_in_the_full_thermal_tilting_pad():
    outputs = run_case(**_fixture_inputs("tilt_5pad_full"), diagnostics=True)
    loops = outputs["convergence"][0]["loops"]
    assert set(loops) == set(LOOP_NAMES)
    assert all(record["iterations"] >= 1 for record in loops.values())
    # tilting pads: the tilt bisection takes more than one step
    assert loops["tilt"]["iterations"] > 1


def test_tilting_pad_operating_table():
    bearing = _tilting_pad(fys_load=-2.8e3)
    op = bearing.operating_results.iloc[0]
    assert op.converged
    pad_load = np.asarray(op.pad_load)
    tilt = np.asarray(op.tilt_angle)
    assert pad_load.shape == tilt.shape == (5,)
    # the pad forces and the groove force balance the load (residual < 2e-3)
    out = bearing.results.outputs[0]
    assert_allclose(np.hypot(out["fx_hydro"][0], out["fy_hydro"][0]), 2.8e3, rtol=2e-3)
    assert pad_load.max() <= 2.8e3 * 1.5
    # load along -y: the attitude from the load equals the attitude
    assert_allclose(op.attitude_from_load, op.attitude, atol=1e-15)


def _tilting_pad(**kwargs):
    from ross.bearings.tilting_pad import TiltingPad

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return TiltingPad(**dict(TILTING_PAD, **kwargs))


@pytest.mark.parametrize("fx_load", [2e3, -2e3])
def test_attitude_from_the_load_direction(fx_load):
    """Turning the load and the lands by 90 deg turns the journal by 90 deg.

    Two 160 deg lands centred on the load line: the layout is the same for a
    vertical and a horizontal load once the lands follow the load, so the
    attitude measured from the load (and the eccentricity) must not change.
    """

    def build(pivots_deg, **load):
        return FluidFilmBearing(
            n=0,
            speed=Q_([3000], "RPM"),
            journal_diameter=0.1,
            radial_clearance=100e-6,
            pad_thickness=0.025,
            pivot_angle=np.radians(pivots_deg),
            pad_arc=[np.radians(160)] * 2,
            pad_axial_length=[0.05] * 2,
            preload=[0.0] * 2,
            offset=[0.5] * 2,
            lubricant="ISOVG32",
            oil_supply_temperature=Q_(40, "degC"),
            oil_flow_v=Q_(10, "l/min"),
            thermal_type=None,
            initial_position=(0.2, 0.15),
            **load,
            **GRID,
        ).operating_results.iloc[0]

    down = build([90, 270], fys_load=-2e3)
    side = build([180, 360], fxs_load=fx_load)
    assert_allclose(down.attitude_from_load, down.attitude, atol=1e-15)
    # same equilibrium up to the Newton tolerance (force residual < 2e-3)
    assert_allclose(side.eccentricity, down.eccentricity, rtol=5e-3)
    assert_allclose(side.attitude_from_load, down.attitude_from_load, atol=5e-3)
    load_angle = np.arctan2(fx_load, 0.0)
    assert_allclose(
        np.angle(np.exp(1j * (side.attitude - load_angle))),
        side.attitude_from_load,
        atol=1e-12,
    )


def test_full_360_deg_pad_is_the_short_bearing_model():
    """A pad of exactly 2 pi drops the circumferential pressure flow (oil seal).

    Reference: the short-bearing (Ocvirk) load
    ``W = mu U L**3 / (4 C**2) eps / (1 - eps**2)**2
    sqrt(pi**2 (1 - eps**2) + 16 eps**2)``. A 359.9 deg land is solved with
    the finite-length equation and carries about half of it at ``L/D = 1``.
    """
    eps, mu, rpm, length, clearance, radius = 0.6, 0.03, 3000.0, 0.1, 100e-6, 0.05

    def load(arc):
        bearing = FluidFilmBearing(
            n=0,
            speed=[rpm * np.pi / 30.0],
            journal_diameter=2 * radius,
            radial_clearance=clearance,
            pad_thickness=0.01,
            pivot_angle=[np.pi],
            pad_arc=[arc],
            pad_axial_length=[length],
            preload=[0.0],
            offset=[0.5],
            lubricant=_isoviscous_lubricant(mu, 0.5 * mu),
            oil_supply_temperature=300.0,
            oil_flow_v=1e-4,
            thermal_type=None,
            equilibrium_type="match_eccentricity",
            initial_position=(0.0, -eps),
            total_ex_film=60,
            total_ez_film=20,
            total_ey_film=10,
            total_ey_pad=10,
        )
        out = bearing.results.outputs[0]
        return np.hypot(out["fx_hydro"][0], out["fy_hydro"][0])

    surface_speed = rpm * np.pi / 30.0 * radius
    ocvirk = (
        mu
        * surface_speed
        * length**3
        / (4 * clearance**2)
        * eps
        / (1 - eps**2) ** 2
        * np.sqrt(np.pi**2 * (1 - eps**2) + 16 * eps**2)
    )
    assert_allclose(load(2 * np.pi), ocvirk, rtol=0.01)
    assert load(np.radians(359.9)) < 0.6 * ocvirk


def test_operating_results_of_an_unsolved_element():
    bearing = FluidFilmBearing.__new__(FluidFilmBearing)
    with pytest.raises(AttributeError, match="needs a solved bearing"):
        _ = bearing.operating_results


def test_loop_monitor_and_budget_units():
    monitor = LoopMonitor()
    monitor.record("newton", 3, 1e-3, 2e-3, True)
    monitor.record("newton", 5, 1e-4, 2e-3, True)
    monitor.record("mixing", 2, np.nan, 0.27, True)
    summary = monitor.summary()
    assert summary["loops"]["newton"]["iterations"] == 5
    assert summary["loops"]["newton"]["total_iterations"] == 8
    assert summary["loops"]["newton"]["passes"] == 2
    assert summary["loops"]["mixing"]["reason"] == "non_finite"
    assert not monitor.converged and monitor.reason == "non_finite"
    monitor.stop("budget", "more than 10 Reynolds solves")
    assert monitor.reason == "budget"
    assert monitor.message().startswith("case: budget")

    budget = CaseBudget(max_reynolds_solves=100, max_solves_per_call=4)
    budget.start_call()
    budget.charge(4)
    budget.start_call()
    budget.charge(3)
    assert (budget.solves, budget.max_call_solves, budget.calls) == (7, 4, 2)


@pytest.mark.parametrize(
    "pivot_type, k_pivot, reason",
    [
        ("ball_in_socket", [0.0, 0.0, 2e9, 3e9], "converged"),
        ("button", [0.0, np.nan, 2e9, 3e9], "pivot_unloaded"),
        ("rocker_back", [0.0, 0.0, 0.0, 0.0], "pivot_unloaded"),
        ("user_specified_stiffness", [1e8] * 4, "converged"),
    ],
)
def test_pivot_record(pivot_type, k_pivot, reason):
    """A pad with no film force has no pivot stiffness and is valid; NaN is not.

    The Hertz stiffness ``~ F**(1/3)`` is 0 at zero pad force and NaN for a
    negative one (the film pulls the pad off its pivot).
    """
    monitor = LoopMonitor()
    driver._record_pivot(monitor, 2, 1e-4, 1e-2, pivot_type, np.array(k_pivot))
    assert monitor.records["pivot"].reason == reason
