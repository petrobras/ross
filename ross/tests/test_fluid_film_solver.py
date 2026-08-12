"""Regression tests for the internal fluid-film TEHD solver.

Each fixture in ``ross/tests/data/fluid_film/`` is a self-contained
``{"inputs", "outputs"}`` document in SI units: the inputs are the keyword
arguments of :func:`ross.bearings.fluid_film.driver.run_case` and the
outputs its full return surface. The solver is pinned at ``rtol=5e-4`` --
this is a same-code regression guard, not physical validation (validation
against published bearing data lives with the user-facing bearing classes).
The tolerance is relaxed relative to bit-identical replay because the
numba-accelerated kernels accumulate floating-point differences across
OS / Python / numpy builds; ``1e-8`` is too tight for the CI matrix.

The fixture family covers the solver's control space: fixed-geometry and
tilting-pad bearings, isoviscous / adiabatic / full thermal models, pad and
pivot deformation (all four pivot models), starved and high-ambient-pressure
operation, pressure-dam geometry, and the specified-eccentricity equilibrium.

The two slowest fixtures (deep iteration counts, ~15 s each) are skipped
unless the environment variable ``ROSS_FLUID_FILM_SLOW`` is set.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.bearings.fluid_film.driver import run_case

DATA_DIR = Path(__file__).parent / "data" / "fluid_film"

CASE_NAMES = sorted(p.stem for p in DATA_DIR.glob("*.json"))

SLOW_CASES = {"fixed_adiabatic_highpamb", "tilt_5pad_full"}


def _flatten(value):
    return np.atleast_1d(np.asarray(value, dtype=float)).ravel()


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_run_case_matches_fixture(case_name):
    if case_name in SLOW_CASES and not os.environ.get("ROSS_FLUID_FILM_SLOW"):
        pytest.skip("slow case; set ROSS_FLUID_FILM_SLOW=1 to run")
    doc = json.loads((DATA_DIR / f"{case_name}.json").read_text())
    result = run_case(**doc["inputs"])
    expected = doc["outputs"]

    assert set(result) == set(expected), "solver output key set changed"
    for key, exp in expected.items():
        act = result[key]
        exp_flat = np.atleast_1d(np.asarray(exp, dtype=object)).ravel()
        if any(isinstance(v, str) for v in exp_flat):
            assert act == exp, f"{case_name}: string output {key!r} changed"
            continue
        assert_allclose(
            _flatten(act),
            _flatten(exp),
            rtol=1e-4,
            atol=0.0,
            err_msg=f"{case_name}: {key!r} drifted from the fixture",
        )
