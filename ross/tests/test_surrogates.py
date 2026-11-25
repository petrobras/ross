import inspect
import pickle
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import ross.surrogates as sg  # noqa: E402
from ross.surrogates.safe_pickle import loads as safe_loads  # noqa: E402
from ross.units import Q_  # noqa: E402

CLEARANCE = [1e-4, 2e-4]
TEMPERATURE = [300.0, 330.0]
SPEED = [100.0, 200.0]
COEFS = ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy")


def analytic(coef_index, speed, clearance, temperature):
    """Smooth closed-form coefficient used as the exact reference of the metamodels."""
    return (
        (coef_index + 1)
        * 1e6
        * (speed / 100)
        * (1e-4 / clearance)
        * (1 + 0.01 * (temperature - 300) / 30)
    )


class FakeSolver:
    """Stand-in for ross.PlainJournal that returns the analytic coefficients."""

    calls = []

    def __init__(self, speed, radial_clearance, oil_supply_temperature, **kwargs):
        FakeSolver.calls.append(float(speed[0]))
        for i, name in enumerate(COEFS):
            value = analytic(i, speed[0], radial_clearance, oil_supply_temperature)
            setattr(self, name, np.array([value]))


class FakeThrustSolver:
    def __init__(self, speed, axial_load, **kwargs):
        self.kzz = np.array([axial_load * speed[0]])
        self.czz = np.array([axial_load / speed[0]])


class FakeJournalSurrogate(sg.PlainJournalSurrogate):
    _bearing_class = FakeSolver


class FakeThrustSurrogate(sg.ThrustPadSurrogate):
    _bearing_class = FakeThrustSolver


def build_journal(metamodel="KGR", n_samples=30, **kwargs):
    return FakeJournalSurrogate(
        metamodel=metamodel,
        num_inital_samples=n_samples,
        input_variable=["radial_clearance", "oil_supply_temperature"],
        n=2,
        frequency=SPEED,
        radial_clearance=CLEARANCE,
        oil_supply_temperature=TEMPERATURE,
        **kwargs,
    )


@pytest.fixture(scope="module")
def journal():
    return build_journal()


@pytest.mark.parametrize("metamodel", ["KGR", "gaussian"])
def test_predict_matches_analytic_function(metamodel):
    surrogate = build_journal(metamodel=metamodel, n_samples=40)
    speeds = [120.0, 150.0, 180.0]
    clearance, temperature = 1.5e-4, 315.0
    bearing = surrogate.predict(
        frequency=speeds, radial_clearance=clearance, oil_supply_temperature=temperature
    )

    assert bearing.n == 2
    for i, name in enumerate(COEFS):
        expected = [analytic(i, w, clearance, temperature) for w in speeds]
        assert_allclose(getattr(bearing, name), expected, rtol=0.02)


def test_predict_accepts_units(journal):
    in_rpm = journal.predict(
        frequency=Q_(150 * 30 / np.pi, "RPM"),
        radial_clearance=1.5e-4,
        oil_supply_temperature=315.0,
    )
    in_rad = journal.predict(
        frequency=150.0, radial_clearance=1.5e-4, oil_supply_temperature=315.0
    )
    assert_allclose(in_rpm.kxx, in_rad.kxx, rtol=1e-6)


def test_training_data_and_bearings(journal):
    assert len(journal.cylidrical_bearing) == 30
    frequency = np.asarray(journal.inputs_model["frequency"])
    assert frequency.min() >= SPEED[0] and frequency.max() <= SPEED[1]
    clearance = np.asarray(journal.inputs_model["radial_clearance"])
    assert clearance.min() >= CLEARANCE[0] and clearance.max() <= CLEARANCE[1]


def test_save_load_round_trip(journal, tmp_path):
    kwargs = dict(
        frequency=[130.0, 170.0], radial_clearance=1.2e-4, oil_supply_temperature=310.0
    )
    before = journal.predict(**kwargs)
    journal.save(tmp_path / "journal")

    with zipfile.ZipFile(tmp_path / "journal.zip") as archive:
        names = set(archive.namelist())
    assert names == {f"{coef}.pkl" for coef in COEFS} | {"description.lzma"}

    loaded = FakeJournalSurrogate.load(tmp_path / "journal")
    assert type(loaded) is FakeJournalSurrogate
    after = loaded.predict(**kwargs)
    for name in COEFS:
        assert_allclose(getattr(after, name), getattr(before, name), rtol=1e-9)

    # saving must not strip the metamodels of the object that was saved
    again = journal.predict(**kwargs)
    assert_allclose(again.kxx, before.kxx, rtol=1e-12)


@pytest.mark.parametrize("metamodel", ["KGR", "gaussian"])
def test_saved_file_does_not_store_training_matrices(metamodel, tmp_path):
    surrogate = build_journal(metamodel=metamodel, n_samples=60)
    surrogate.save(tmp_path / "s")
    assert (tmp_path / "s.zip").stat().st_size < 200e3


def test_saved_kriging_models_hold_no_training_data(journal, tmp_path):
    journal.save(tmp_path / "s")
    loaded = FakeJournalSurrogate.load(tmp_path / "s")
    for coef in COEFS:
        model = getattr(loaded, f"sm_{coef}")
        assert type(model).__name__ == "Kriging"
        assert model.xt is None and model.yt is None
    assert journal.sm_kxx.xt is not None


@pytest.mark.parametrize("corr", ["squar_exp", "matern52"])
def test_predict_with_other_correlation_functions(corr):
    surrogate = build_journal(n_samples=40, corr=corr)
    bearing = surrogate.predict(
        frequency=150.0, radial_clearance=1.5e-4, oil_supply_temperature=315.0
    )
    for i, name in enumerate(COEFS):
        expected = analytic(i, 150.0, 1.5e-4, 315.0)
        assert_allclose(getattr(bearing, name), [expected], rtol=0.02)


def test_full_factorial_distribution_samples_a_regular_grid():
    surrogate = build_journal(n_samples=27, distribution="FullFactorial")
    for name in ("frequency", "radial_clearance", "oil_supply_temperature"):
        assert len(np.unique(surrogate.inputs_model[name])) == 3
    bearing = surrogate.predict(
        frequency=150.0, radial_clearance=1.5e-4, oil_supply_temperature=315.0
    )
    assert_allclose(bearing.kxx, [analytic(0, 150.0, 1.5e-4, 315.0)], rtol=0.05)


def test_single_speed_range_is_sampled():
    surrogate = FakeJournalSurrogate(
        metamodel="KGR",
        num_inital_samples=20,
        input_variable=["radial_clearance", "oil_supply_temperature"],
        n=2,
        frequency=[150.0, 150.0],
        radial_clearance=CLEARANCE,
        oil_supply_temperature=TEMPERATURE,
    )
    assert_allclose(surrogate.inputs_model["frequency"], 150.0)


def test_save_adds_zip_extension_once(journal, tmp_path):
    journal.save(tmp_path / "a.zip")
    journal.save(tmp_path / "b")
    assert (tmp_path / "a.zip").is_file()
    assert (tmp_path / "b.zip").is_file()
    assert not (tmp_path / "a.zip.zip").exists()


def test_save_sample_resumes_from_csv(tmp_path):
    csv = tmp_path / "samples.csv"
    FakeSolver.calls.clear()
    build_journal(n_samples=12, save_sample=csv)
    assert len(FakeSolver.calls) == 12
    first = pd.read_csv(csv)
    assert len(first) == 12

    FakeSolver.calls.clear()
    resumed = build_journal(n_samples=12, save_sample=csv)
    assert FakeSolver.calls == []
    assert_allclose(resumed.inputs_model["frequency"], first["frequency"])

    FakeSolver.calls.clear()
    larger = build_journal(n_samples=16, save_sample=csv)
    assert len(FakeSolver.calls) == 4
    assert_allclose(larger.inputs_model["frequency"][:12], first["frequency"])


def test_verify_limits_predict(journal):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        speeds, inputs = journal.verify_limits_predict(
            frequency=[150.0, 500.0],
            args={"radial_clearance": 3e-4, "oil_supply_temperature": 310.0},
        )
    assert_allclose(speeds, [500.0])
    assert list(inputs) == ["radial_clearance"]
    assert len(caught) == 2

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        speeds, inputs = journal.verify_limits_predict(
            frequency=150.0,
            args={"radial_clearance": 1.5e-4, "oil_supply_temperature": 310.0},
        )
    assert speeds.size == 0 and inputs == {}


def test_predict_verify_limits_warns(journal):
    with pytest.warns(UserWarning, match="outside the training range"):
        journal.predict(
            frequency=150.0,
            radial_clearance=3e-4,
            oil_supply_temperature=310.0,
            verify_limits=True,
        )


def test_unknown_metamodel_raises():
    with pytest.raises(ValueError, match="metamodel must be"):
        build_journal(metamodel="foo")


def test_thrust_surrogate_returns_axial_coefficients(tmp_path):
    surrogate = FakeThrustSurrogate(
        metamodel="gaussian",
        num_inital_samples=25,
        input_variable=["axial_load"],
        n=1,
        frequency=[80.0, 100.0],
        axial_load=[10e6, 16e6],
    )
    bearing = surrogate.predict(frequency=90.0, axial_load=13e6)
    assert_allclose(bearing.kzz, [13e6 * 90.0], rtol=0.02)
    assert_allclose(bearing.czz, [13e6 / 90.0], rtol=0.02)
    assert_allclose(bearing.kxx, 0)

    surrogate.save(tmp_path / "thrust")
    loaded = FakeThrustSurrogate.load(tmp_path / "thrust")
    assert loaded._get_coef_names() == ["kzz", "czz"]
    assert_allclose(
        loaded.predict(frequency=90.0, axial_load=13e6).kzz, bearing.kzz, rtol=1e-9
    )


@pytest.mark.parametrize(
    "surrogate_class, solver",
    [
        (sg.PlainJournalSurrogate, "ross.bearings.plain_journal.PlainJournal"),
        (sg.TiltingPadSurrogate, "ross.bearings.tilting_pad.TiltingPad"),
        (sg.ThrustPadSurrogate, "ross.bearings.thrust_pad.ThrustPad"),
    ],
)
def test_forwarded_arguments_are_accepted_by_the_solver(surrogate_class, solver):
    bearing_class = surrogate_class._bearing_class
    assert f"{bearing_class.__module__}.{bearing_class.__qualname__}" == solver
    accepted = set()
    for klass in bearing_class.__mro__:
        init = klass.__dict__.get("__init__")
        if init is not None:
            accepted |= set(inspect.signature(init).parameters)
    assert set(surrogate_class._bearing_arguments) <= accepted


def test_safe_pickle_asks_to_retrain_surrogates_saved_with_smt():
    smt_pickle = b"csmt.surrogate_models.krg\nKRG\n)R."
    with pytest.raises(pickle.UnpicklingError, match="Train the surrogate again"):
        safe_loads(smt_pickle)


def test_safe_pickle_rejects_unknown_globals():
    import os

    with pytest.raises(pickle.UnpicklingError):
        safe_loads(pickle.dumps(os.getcwd))


def test_classes_are_lazily_exported():
    for name in sg.__all__:
        assert getattr(sg, name).__name__ == name
    with pytest.raises(AttributeError):
        sg.__getattr__("DoesNotExist")
