"""Golden-master tests for the seal solvers.

These tests pin the numerical behavior of :class:`~ross.seals.labyrinth_seal.LabyrinthSeal`
and :class:`~ross.seals.holepattern_seal.HolePatternSeal` across several geometries,
operating conditions and model options. Beyond the integrated rotordynamic
coefficients, they also pin the internal distributions (cavity pressures, swirl
velocities, temperatures, shear stresses), which catch refactor slips that the
integrated coefficients would average away.

All cases use manually specified gas properties, so the reference values do not
depend on the equation-of-state backend available on the machine.

The reference values live in ``ross/tests/data/seals_golden.json``. To
regenerate them after an intentional behavior change, run::

    python -m ross.tests.test_seals_golden

and review the diff of the JSON file before committing it.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ross.seals.holepattern_seal import HolePatternSeal
from ross.seals.labyrinth_seal import LabyrinthSeal
from ross.units import Q_

GOLDEN_FILE = Path(__file__).parent / "data" / "seals_golden.json"

LABYRINTH_COEFFICIENTS = ["kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"]
HOLEPATTERN_COEFFICIENTS = LABYRINTH_COEFFICIENTS + ["mxx", "mxy", "myx", "myy"]

LABYRINTH_BASE = {
    "n": 0,
    "inlet_pressure": 308000,
    "outlet_pressure": 94300,
    "inlet_temperature": 283.15,
    "preswirl": 0.98,
    "n_teeth": 16,
    "shaft_diameter": Q_(145, "mm"),
    "radial_clearance": Q_(0.3, "mm"),
    "pitch": Q_(3.175, "mm"),
    "tooth_height": Q_(3.175, "mm"),
    "tooth_width": Q_(0.1524, "mm"),
    "seal_type": "inter",
    "molar": 28.96807,
    "gamma": 1.41,
    "tz": [283.15, 282.60903080958565],
    "muz": [1.7746561138374613e-05, 1.7687886306966975e-05],
}

LABYRINTH_CASES = {
    "inter": {
        **LABYRINTH_BASE,
        "frequency": Q_([5000, 8000, 11000], "RPM"),
    },
    "rotor": {
        **LABYRINTH_BASE,
        "seal_type": "rotor",
        "preswirl": 0.3,
        "radial_clearance": Q_(0.25, "mm"),
        "frequency": Q_([6000, 9000], "RPM"),
    },
    "stator_jenny_kanki": {
        **LABYRINTH_BASE,
        "seal_type": "stator",
        "iopt1": 1,
        "frequency": Q_([8000], "RPM"),
    },
    "choked": {
        **LABYRINTH_BASE,
        "inlet_pressure": 1.0e6,
        "n_teeth": 3,
        "frequency": Q_([8000], "RPM"),
    },
}

HOLEPATTERN_BASE = {
    "n": 0,
    "axial_length": 0.0254,
    "shaft_diameter": 0.1502,
    "radial_clearance": 0.0004,
    "relative_roughness": 0.00198,
    "cell_length": 0.001,
    "cell_width": 0.001,
    "cell_depth": 0.00229,
    "inlet_pressure": 1830000.0,
    "outlet_pressure": 823500.0,
    "inlet_temperature": 300.0,
    "preswirl": 1.0,
    "entr_coef": 0.1,
    "exit_coef": 0.5,
    "b_suther": 1.458e-6,
    "s_suther": 110.4,
    "molar": 29.0,
    "gamma": 1.4,
}

HOLEPATTERN_CASES = {
    "base": {
        **HOLEPATTERN_BASE,
        "frequency": Q_([5000], "RPM"),
    },
    "low_swirl_subsync": {
        **HOLEPATTERN_BASE,
        "preswirl": 0.3,
        "excitation_ratio": 0.5,
        "nz": 40,
        "frequency": Q_([4000, 8000], "RPM"),
    },
}


@lru_cache(maxsize=None)
def _build_labyrinth(case, frequency_index=None):
    """Build the seal for a case, optionally restricted to one frequency.

    Single-frequency builds are used to capture the internal distributions:
    they are unambiguous regardless of how the multi-frequency loop stores
    per-run state.
    """
    params = dict(LABYRINTH_CASES[case])
    if frequency_index is not None:
        params["frequency"] = params["frequency"][frequency_index : frequency_index + 1]
    return LabyrinthSeal(**params)


@lru_cache(maxsize=None)
def _build_holepattern(case, frequency_index=None):
    params = dict(HOLEPATTERN_CASES[case])
    if frequency_index is not None:
        params["frequency"] = params["frequency"][frequency_index : frequency_index + 1]
    return HolePatternSeal(**params)


def _labyrinth_snapshot(case):
    seal = _build_labyrinth(case)
    n_cavities = seal.n_teeth + 1
    snapshot = {
        "coefficients": {
            c: np.atleast_1d(getattr(seal, c)).astype(float).tolist()
            for c in LABYRINTH_COEFFICIENTS
        },
        "seal_leakage": np.atleast_1d(seal.seal_leakage).astype(float).tolist(),
        "pert_rcond": np.atleast_1d(seal.pert_rcond).astype(float).tolist(),
        "distributions": [],
    }
    for k in range(len(LABYRINTH_CASES[case]["frequency"])):
        single = _build_labyrinth(case, k)
        snapshot["distributions"].append(
            {
                "pressure": np.asarray(single.p[0][:n_cavities], float).tolist(),
                "swirl_velocity": np.asarray(single.v[: seal.n_teeth], float).tolist(),
                "throat_velocity": np.asarray(single.w[:n_cavities], float).tolist(),
                "temperature": np.asarray(single.t[:n_cavities], float).tolist(),
                "density": np.asarray(single.rho[:n_cavities], float).tolist(),
                "shear_rotor": np.asarray(single.taur[: seal.n_teeth], float).tolist(),
                "shear_stator": np.asarray(single.taus[: seal.n_teeth], float).tolist(),
            }
        )
    return snapshot


def _holepattern_snapshot(case):
    seal = _build_holepattern(case)
    n_stations = seal.nz + 1
    snapshot = {
        "coefficients": {
            c: np.atleast_1d(getattr(seal, c)).astype(float).tolist()
            for c in HOLEPATTERN_COEFFICIENTS
        },
        "seal_leakage": np.atleast_1d(seal.seal_leakage).astype(float).tolist(),
        "distributions": [],
    }
    for k in range(len(HOLEPATTERN_CASES[case]["frequency"])):
        single = _build_holepattern(case, k)
        snapshot["distributions"].append(
            {
                "pressure": np.asarray(single.p[0], float).tolist(),
                "axial_mach_squared": np.asarray(
                    single.mz2[:n_stations], float
                ).tolist(),
                "tangential_mach": np.asarray(single.mt[:n_stations], float).tolist(),
                "temperature": np.asarray(single.t[:n_stations], float).tolist(),
            }
        )
    return snapshot


def _generate():
    golden = {
        "labyrinth": {case: _labyrinth_snapshot(case) for case in LABYRINTH_CASES},
        "holepattern": {
            case: _holepattern_snapshot(case) for case in HOLEPATTERN_CASES
        },
    }
    with open(GOLDEN_FILE, "w") as f:
        json.dump(golden, f, indent=1)
    return golden


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_FILE) as f:
        return json.load(f)


def _assert_snapshot_matches(snapshot, expected):
    for name, values in expected["coefficients"].items():
        assert_allclose(
            snapshot["coefficients"][name],
            values,
            rtol=1e-4,
            atol=1e-9,
            err_msg=f"coefficient {name}",
        )
    assert_allclose(snapshot["seal_leakage"], expected["seal_leakage"], rtol=1e-4)
    if "pert_rcond" in expected:
        assert_allclose(snapshot["pert_rcond"], expected["pert_rcond"], rtol=1e-3)
    for k, expected_dist in enumerate(expected["distributions"]):
        for name, values in expected_dist.items():
            scale = np.max(np.abs(values)) if np.size(values) else 1.0
            assert_allclose(
                snapshot["distributions"][k][name],
                values,
                rtol=1e-4,
                atol=1e-9 * max(scale, 1.0),
                err_msg=f"distribution {name}, frequency index {k}",
            )


@pytest.mark.parametrize("case", list(LABYRINTH_CASES))
def test_labyrinth_golden(golden, case):
    _assert_snapshot_matches(_labyrinth_snapshot(case), golden["labyrinth"][case])


@pytest.mark.parametrize("case", list(HOLEPATTERN_CASES))
def test_holepattern_golden(golden, case):
    _assert_snapshot_matches(_holepattern_snapshot(case), golden["holepattern"][case])


if __name__ == "__main__":
    _generate()
    print(f"Golden data written to {GOLDEN_FILE}")
