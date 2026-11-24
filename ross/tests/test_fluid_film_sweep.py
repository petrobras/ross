"""Bearing setups, warm-started speed sweeps and worker processes.

References, all independent of the code under test:

- ``run_case`` on a fresh setup (a new mesh per call, the path every pinned
  fixture runs; the fixtures did not change when the setup was introduced);
- cold solves of each speed: a warm-started speed converges to the same
  equilibrium within the loop tolerances, so with every tolerance scaled by
  1e-3 both agree far below 0.05 %;
- the serial sweep for the parallel one (the chains do not depend on
  ``n_jobs``), bit for bit;
- the applied load for the film force at the equilibrium (Newton tolerance
  ``|F_net| / |F_ext| < 2e-3``).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.fluid_film import driver
from ross.bearings.fluid_film.diagnostics import CaseBudget
from ross.bearings.fluid_film.driver import run_case
from ross.bearings.fluid_film.sweep import (
    BearingSetup,
    WarmStart,
    film_force_at,
    solve_case,
    solve_sweep,
    sweep_chains,
)

DATA_DIR = Path(__file__).parent / "data" / "fluid_film"
CASE_NAMES = sorted(p.stem for p in DATA_DIR.glob("*.json"))
SLOW_CASES = {"fixed_adiabatic_highpamb", "tilt_5pad_full"}
COEFFICIENT_KEYS = ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy")
SPEED = 94.24777960769379  # every fixture runs at 900 rpm


def _inputs(case, **changes):
    inputs = json.loads((DATA_DIR / f"{case}.json").read_text())["inputs"]
    inputs.update(changes)
    return inputs


def _assert_same_outputs(actual, expected, rtol=1e-12):
    assert set(actual) == set(expected)
    for key, exp in expected.items():
        flat = np.atleast_1d(np.asarray(exp, dtype=object)).ravel()
        if any(isinstance(v, str) for v in flat):
            assert actual[key] == exp, key
            continue
        assert_allclose(
            np.asarray(actual[key], dtype=float),
            np.asarray(exp, dtype=float),
            rtol=rtol,
            atol=0.0,
            err_msg=key,
        )


def _arrays(value, prefix=""):
    """Every array reachable from ``value``, as ``{path: array}``."""
    import dataclasses

    found = {}
    if isinstance(value, np.ndarray):
        found[prefix] = value
    elif isinstance(value, dict):
        for k, v in value.items():
            found.update(_arrays(v, f"{prefix}/{k}"))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            found.update(_arrays(v, f"{prefix}/{i}"))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            found.update(_arrays(getattr(value, f.name), f"{prefix}.{f.name}"))
    return found


def _matrices(out, j=0):
    k = np.array([[out["kxx"][j], out["kxy"][j]], [out["kyx"][j], out["kyy"][j]]])
    c = np.array([[out["cxx"][j], out["cxy"][j]], [out["cyx"][j], out["cyy"][j]]])
    return k, c


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_cold_solve_case_on_a_shared_setup_equals_run_case(case_name):
    """A setup solved at another speed first still gives ``run_case`` exactly.

    Reference: ``run_case`` on its own fresh setup. Before the setup existed
    the film-thickness rebuild wrote the mesh's radial node coordinates in
    place, so a second speed on the same mesh started from the
    first speed's coordinates; the setup is now read-only and every case
    copies the arrays it writes.
    """
    if case_name in SLOW_CASES and not os.environ.get("ROSS_FLUID_FILM_SLOW"):
        pytest.skip("slow case; set ROSS_FLUID_FILM_SLOW=1 to run")
    inputs = _inputs(case_name)
    setup = BearingSetup.from_inputs(**inputs)
    solve_case(setup, 1.3 * SPEED, diagnostics=False)
    shared = solve_case(setup, SPEED, diagnostics=False)
    _assert_same_outputs(shared, run_case(**inputs))


def test_setup_arrays_are_read_only_and_unchanged_by_a_sweep():
    """No solve writes the setup; its arrays refuse writes.

    Reference: a copy of every array of the setup taken before the sweep.
    """
    setup = BearingSetup.from_inputs(**_inputs("tilt_5pad_isoviscous"))
    arrays = _arrays(setup.state) | _arrays(setup.inputs, "inputs")
    before = {path: a.copy() for path, a in arrays.items()}
    assert {"/y_3d", "/y_energy", "/mesh.y_3d", "/energy_mesh.y"} <= set(arrays)
    solve_sweep(setup, SPEED * np.array([0.8, 1.0, 1.2]))
    for path, a in arrays.items():
        assert not a.flags.writeable, path
        assert np.array_equal(a, before[path]), path
    with pytest.raises(ValueError, match="read-only"):
        setup.mesh.y_3d[0, 0] = 1.0


def test_setup_ignores_the_case_keywords_and_copies_the_inputs():
    inputs = _inputs("fixed_isoviscous")
    preload = np.asarray(inputs["preload"], dtype=float)
    inputs["preload"] = preload
    setup = BearingSetup.from_inputs(**inputs, whirl_ratios=[0.5], diagnostics=True)
    assert "frequency" not in setup.inputs and "whirl_ratios" not in setup.inputs
    # the caller's array is not locked by the setup
    assert preload.flags.writeable


def test_setup_survives_pickling_read_only():
    import pickle

    setup = BearingSetup.from_inputs(**_inputs("fixed_isoviscous"))
    copy = pickle.loads(pickle.dumps(setup))
    assert not copy.mesh.y_3d.flags.writeable
    assert not copy.state["y_energy"].flags.writeable
    _assert_same_outputs(
        solve_case(copy, SPEED, diagnostics=False),
        solve_case(setup, SPEED, diagnostics=False),
        rtol=0.0,
    )


def test_sweep_chains_depend_only_on_the_speeds():
    """Sorted ascending, cut every ``chunk_size``; ties keep the input order."""
    speeds = [500.0, 100.0, 300.0, 200.0, 400.0, 300.0, 50.0]
    chains = sweep_chains(speeds, 3)
    assert [c.tolist() for c in chains] == [[6, 1, 3], [2, 5, 4], [0]]
    assert [c.tolist() for c in sweep_chains(speeds, 7)] == [[6, 1, 3, 2, 5, 4, 0]]
    assert [c.tolist() for c in sweep_chains(speeds, 1)] == [[i] for i in chains[0]] + [
        [i] for i in chains[1]
    ] + [[0]]
    with pytest.raises(ValueError, match="chunk_size"):
        sweep_chains(speeds, 0)
    with pytest.raises(TypeError, match="chunk_size"):
        sweep_chains(speeds, 2.5)


def test_sweep_order_does_not_change_the_results():
    """Unsorted speeds give the outputs of the sorted sweep, bit for bit.

    The chains are built from the sorted speeds, so the input order only
    permutes the outputs.
    """
    setup = BearingSetup.from_inputs(**_inputs("fixed_isoviscous"))
    speeds = SPEED * np.array([1.2, 0.7, 1.0, 0.85, 1.35, 0.6, 1.1])
    order = np.argsort(speeds)
    unsorted = solve_sweep(setup, speeds, chunk_size=3, diagnostics=False)
    ordered = solve_sweep(setup, speeds[order], chunk_size=3, diagnostics=False)
    for i, j in enumerate(order):
        _assert_same_outputs(unsorted[j], ordered[i], rtol=0.0)


# fixture, input changes, speed factors (ascending, unsorted)
FAST_SPEEDS = (
    [2 / 3, 5 / 6, 1.0, 7 / 6, 4 / 3],
    [1.0, 2 / 3, 4 / 3, 5 / 6, 7 / 6, 1.25],
)
THERMAL_SPEEDS = ([0.8, 1.0, 1.2], [1.2, 0.8, 1.0])
WARM_CASES = {
    "tilting_pad_isoviscous": ("tilt_5pad_isoviscous", {}, FAST_SPEEDS),
    "tilting_pad_pivot_deformation": ("tilt_5pad_deform5", {}, THERMAL_SPEEDS),
    "plain_journal_isoviscous": ("fixed_isoviscous", {}, FAST_SPEEDS),
    "plain_journal_full_tehd": ("fixed_full_coarse", {}, THERMAL_SPEEDS),
}
# about 1-4 min per test at the scaled tolerance
SLOW_WARM_CASES = {
    "tilting_pad_adiabatic": (
        "tilt_5pad_isoviscous",
        {"thermal_type": "adiabatic"},
        THERMAL_SPEEDS,
    ),
    "tilting_pad_full_thermal": (
        "tilt_5pad_isoviscous",
        {"thermal_type": "full"},
        THERMAL_SPEEDS,
    ),
}


@pytest.mark.parametrize("unsorted", [False, True], ids=["ascending", "unsorted"])
@pytest.mark.parametrize(
    "case",
    sorted(WARM_CASES)
    + [pytest.param(c, marks=pytest.mark.slow) for c in sorted(SLOW_WARM_CASES)],
)
def test_warm_chain_agrees_with_cold_solves(case, unsorted):
    """Warm-started speeds converge to the cold equilibria.

    Reference: each speed solved cold. Both runs scale every tolerance by
    1e-3 so that the agreement measures the equilibria, not the
    tolerances, and the check allows 0.05 % (a small cross-coupled
    tilting-pad term can differ by more relative to itself). Each stiffness
    and damping matrix is compared at its own scale and the direct
    terms one by one. With five or more speeds and ``unsorted`` the input
    spans two chains of the default length 5.
    """
    fixture, changes, factors = {**WARM_CASES, **SLOW_WARM_CASES}[case]
    setup = BearingSetup.from_inputs(**_inputs(fixture, **changes))
    speeds = SPEED * np.array(factors[1] if unsorted else factors[0])
    warm = solve_sweep(setup, speeds, _tolerance_scale=1e-3)
    cold = solve_sweep(setup, speeds, warm_start=False, _tolerance_scale=1e-3)
    starts = [out["convergence"][0]["warm_start"]["start"] for out in warm]
    assert starts.count("warm") == len(speeds) - len(sweep_chains(speeds, 5))
    for w, c, speed in zip(warm, cold, speeds, strict=True):
        assert w["convergence"][0]["converged"] and c["convergence"][0]["converged"]
        for mw, mc in zip(_matrices(w), _matrices(c), strict=True):
            assert np.max(np.abs(mw - mc)) <= 5e-4 * np.max(np.abs(mc)), speed
            assert_allclose(np.diag(mw), np.diag(mc), rtol=5e-4)
        assert_allclose(w["eccentricity"], c["eccentricity"], rtol=5e-4)


def test_warm_start_keeps_a_prescribed_position():
    """``match_eccentricity`` cases keep their position; only the fields warm.

    Reference: the prescribed position of the inputs (``xj``, ``yj`` in units
    of the clearance).
    """
    inputs = _inputs("fixed_eccentricity", thermal_type="adiabatic")
    setup = BearingSetup.from_inputs(**inputs)
    outs = solve_sweep(setup, SPEED * np.array([0.8, 1.0, 1.2]))
    assert [o["convergence"][0]["warm_start"]["start"] for o in outs] == [
        "cold",
        "warm",
        "warm",
    ]
    for out in outs:
        assert_allclose(out["xj_cb"], inputs["xj"], rtol=0, atol=0)
        assert_allclose(out["yj_cb"], inputs["yj"], rtol=0, atol=0)


def test_warm_start_copies_and_does_not_alias():
    """A warm start reused twice gives the same case; the source is not written."""
    setup = BearingSetup.from_inputs(
        **_inputs("fixed_isoviscous", thermal_type="adiabatic")
    )
    _, warm = solve_case(setup, SPEED, return_warm=True)
    assert isinstance(warm, WarmStart)
    saved = {k: np.array(v, copy=True) for k, v in warm.temperature.items()}
    first = solve_case(setup, 1.2 * SPEED, warm=warm, diagnostics=False)
    second = solve_case(setup, 1.2 * SPEED, warm=warm, diagnostics=False)
    _assert_same_outputs(first, second, rtol=0.0)
    for k, v in saved.items():
        assert np.array_equal(warm.temperature[k], v), k


def test_failed_warm_case_is_retried_cold(monkeypatch):
    """A warm-started speed that fails is solved again cold, and recorded.

    The warm attempt of the second speed gets a budget of one Reynolds solve
    (monkeypatched); its cold retry has the normal budget. Reference: the
    cold solve of that speed on its own, which the retry equals bit for bit.
    """
    setup = BearingSetup.from_inputs(**_inputs("fixed_isoviscous"))
    speeds = SPEED * np.array([0.8, 1.0, 1.2])
    original = driver._case_budget
    calls = []

    def budget(kw):
        calls.append(float(kw["frequency"]))
        if abs(calls[-1] - speeds[1]) < 1e-9 and calls.count(calls[-1]) == 1:
            return CaseBudget(max_reynolds_solves=1)
        return original(kw)

    monkeypatch.setattr(driver, "_case_budget", budget)
    outs = solve_sweep(setup, speeds)
    record = outs[1]["convergence"][0]["warm_start"]
    assert record["start"] == "cold_retry"
    assert record["failure"].startswith("case: budget")
    assert record["failed_solves"] == 1
    assert [o["convergence"][0]["converged"] for o in outs] == [True, True, True]
    # the third speed starts from the converged retry
    assert outs[2]["convergence"][0]["warm_start"]["start"] == "warm"
    monkeypatch.setattr(driver, "_case_budget", original)
    cold = solve_case(setup, speeds[1], diagnostics=False)
    retried = {k: v for k, v in outs[1].items() if k != "convergence"}
    _assert_same_outputs(retried, cold, rtol=0.0)


def test_film_force_at_the_equilibrium_balances_the_load(monkeypatch):
    """The film force at a solved position is ``-F_ext`` within the Newton tolerance.

    Reference: the applied load and the tolerance ``|F_net| / |F_ext| < 2e-3``,
    for a fixed-geometry and a tilting-pad bearing and three speeds each, on
    one setup (no re-meshing).
    """
    from ross.bearings.fluid_film import mesh

    for case in ("fixed_isoviscous", "tilt_5pad_isoviscous"):
        inputs = _inputs(case)
        setup = BearingSetup.from_inputs(**inputs)
        f_ext = np.array([inputs["fxs_load"], inputs["fys_load"] - inputs["weight"]])
        cb = inputs["radial_clearance"]
        for speed in SPEED * np.array([0.8, 1.0, 1.25]):
            out = solve_case(setup, speed)
            assert out["convergence"][0]["converged"]
            calls = []
            original = mesh.mesh_reynolds
            monkeypatch.setattr(
                mesh,
                "mesh_reynolds",
                lambda *a, **k: calls.append(1) or original(*a, **k),
            )
            force = film_force_at(
                setup, speed, out["xj_cb"][0] * cb, out["yj_cb"][0] * cb
            )
            monkeypatch.undo()
            assert not calls
            assert force["converged"]
            net = np.array([force["fx"], force["fy"]]) + f_ext
            assert np.hypot(*net) < 2e-3 * np.hypot(*f_ext)


def test_film_force_at_on_a_setup_equals_the_inputs_path():
    inputs = _inputs("tilt_5pad_isoviscous")
    setup = BearingSetup.from_inputs(**inputs)
    cb = inputs["radial_clearance"]
    a = film_force_at(setup, SPEED, 0.1 * cb, -0.4 * cb)
    b = film_force_at(inputs, SPEED, 0.1 * cb, -0.4 * cb)
    for key in ("fx", "fy", "fx_pad", "fy_pad", "moment_pad", "tilt_angle"):
        assert_allclose(a[key], b[key], rtol=0, atol=0)


def test_num_processes_maps_to_n_jobs(monkeypatch):
    """``num_processes`` is an alias of ``n_jobs``; the sweep receives it."""
    import ross.bearings.fluid_film_bearing as module
    from ross.bearings.fluid_film_bearing import FluidFilmBearing
    from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture

    jobs = []

    class Stop(Exception):
        pass

    def capture(job):
        jobs.append(job)
        raise Stop

    monkeypatch.setattr(module, "_solve_case", capture)
    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    for extra, expected in (
        ({"num_processes": 2}, 2),
        ({"num_processes": 1}, 1),
        ({"n_jobs": 3}, 3),
        ({"n_jobs": 2, "num_processes": 2}, 2),
        ({}, 1),
    ):
        with pytest.raises(Stop):
            FluidFilmBearing(**kwargs, **extra)
        assert jobs[-1]["n_jobs"] == expected
        assert jobs[-1]["warm_start"] is True and jobs[-1]["chunk_size"] == 5
    with pytest.raises(ValueError, match="alias of n_jobs"):
        FluidFilmBearing(**kwargs, n_jobs=3, num_processes=2)
    with pytest.raises(ValueError, match="n_jobs"):
        FluidFilmBearing(**kwargs, n_jobs=0)
    with pytest.raises(ValueError, match="chunk_size"):
        FluidFilmBearing(**kwargs, chunk_size=0)


def test_bearing_without_warm_start_equals_single_speed_bearings():
    """``warm_start=False`` solves every speed as its own bearing would.

    Reference: one ``FluidFilmBearing`` per speed (a single speed always
    starts cold), bit for bit; the warm-started table agrees within the
    default tolerances.
    """
    from ross.bearings.fluid_film_bearing import FluidFilmBearing
    from ross.tests.test_fluid_film_bearing import bearing_kwargs_from_fixture

    kwargs, _ = bearing_kwargs_from_fixture("fixed_isoviscous")
    kwargs["speed"] = [80.0, 95.0, 110.0]
    cold = FluidFilmBearing(**kwargs, warm_start=False)
    warm = FluidFilmBearing(**kwargs, chunk_size=2)
    assert [c["warm_start"]["start"] for c in warm.results.convergence] == [
        "cold",
        "warm",
        "cold",
    ]
    assert [c["warm_start"]["start"] for c in cold.results.convergence] == ["cold"] * 3
    for i, speed in enumerate(kwargs["speed"]):
        single = FluidFilmBearing(**dict(kwargs, speed=[speed]))
        for name in COEFFICIENT_KEYS:
            assert getattr(cold.results, name)[i] == getattr(single.results, name)[0]
            assert_allclose(
                getattr(warm.results, name)[i],
                getattr(single.results, name)[0],
                rtol=1e-2,
            )


@pytest.mark.slow
def test_parallel_sweep_equals_serial_sweep():
    """``n_jobs=2`` gives the serial results bit for bit (7 speeds, 2 chains).

    Reference: the serial sweep. Both run the same chains (the speeds decide
    them) with single-threaded BLAS; the workers use ``spawn``.
    """
    setup = BearingSetup.from_inputs(**_inputs("tilt_5pad_isoviscous"))
    speeds = SPEED * np.array([1.3, 0.7, 1.0, 0.85, 1.15, 0.6, 1.45])
    serial = solve_sweep(setup, speeds, diagnostics=False, pad_matrices=True)
    parallel = solve_sweep(
        setup, speeds, n_jobs=2, diagnostics=False, pad_matrices=True
    )
    for s, p in zip(serial, parallel, strict=True):
        ms, mp = s.pop("pad_matrices")[0], p.pop("pad_matrices")[0]
        _assert_same_outputs(p, s, rtol=0.0)
        for a, b in zip(ms.K + ms.C + ms.M, mp.K + mp.C + mp.M, strict=True):
            assert np.array_equal(a, b)


UNGUARDED_SCRIPT = """
import json
from pathlib import Path

from ross.bearings.fluid_film.sweep import BearingSetup, solve_sweep

inputs = json.loads(Path({path!r}).read_text())["inputs"]
setup = BearingSetup.from_inputs(**inputs)
outputs = solve_sweep(setup, [80.0, 110.0], chunk_size=1, n_jobs=2)
print("SOLVED", len(outputs))
"""


@pytest.mark.slow
def test_unguarded_script_with_worker_processes_raises(tmp_path):
    """A script without the entry-point guard stops with an error, unsolved.

    The ``spawn`` workers import the script as their main module, so the
    unguarded ``solve_sweep`` call runs again in each worker, which may not
    start processes of its own while it bootstraps; the workers die before
    returning anything. The parent must raise (no serial fallback, no
    results), say what to do, and chain the worker failure.
    """
    import subprocess
    import sys

    import ross

    tree = str(Path(ross.__file__).parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [tree] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    script = tmp_path / "unguarded.py"
    script.write_text(
        UNGUARDED_SCRIPT.format(path=str(DATA_DIR / "fixed_isoviscous.json"))
    )
    run = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
        env=env,
    )
    assert run.returncode != 0
    assert "SOLVED" not in run.stdout
    assert "RuntimeError: the worker processes of the n_jobs=2 sweep" in run.stderr
    assert 'if __name__ == "__main__":' in run.stderr
    assert "n_jobs=1" in run.stderr
    assert "BrokenProcessPool" in run.stderr
    assert "direct cause of the following exception" in run.stderr
