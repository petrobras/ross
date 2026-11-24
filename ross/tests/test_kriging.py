import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ross.surrogates.kriging import (
    CORRELATIONS,
    Kriging,
    _componentwise_distance,
    _correlation,
    full_factorial,
    latin_hypercube,
)


def smooth_function(x):
    return np.sin(3 * x[:, 0]) + 0.5 * np.cos(2 * x[:, 1]) + 0.3 * x[:, 0] * x[:, 1]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    x = rng.random((30, 2))
    x_test = rng.random((200, 2))
    return x, smooth_function(x), x_test, smooth_function(x_test)


def trained(corr, x, y, **kwargs):
    model = Kriging(corr=corr, **kwargs)
    model.set_training_values(x, y)
    model.train()
    return model


@pytest.mark.parametrize("corr", CORRELATIONS)
def test_likelihood_gradient_matches_finite_differences(corr, data):
    x, y, _, _ = data
    model = Kriging(corr=corr)
    x_norma = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    y_norma = (y - y.mean()) / y.std(ddof=1)
    distance = _componentwise_distance(x_norma, x_norma, model.power)
    theta = np.array([0.5, 2.0])
    step = 1e-6

    analytic = model._fit(theta, distance, y_norma, gradient=True)["gradient"]
    numeric = []
    for direction in np.eye(2):
        above = model._fit(theta + step * direction, distance, y_norma)["likelihood"]
        below = model._fit(theta - step * direction, distance, y_norma)["likelihood"]
        numeric.append((above - below) / (2 * step))

    # Central differences of a likelihood of order 1e2 with step 1e-6 are accurate
    # to about 1e-8 in absolute value; 1e-5 is far below the gradient itself.
    assert_allclose(analytic, numeric, rtol=1e-5)


@pytest.mark.parametrize(
    "corr, expected",
    [
        ("squar_exp", np.exp(-0.5 * 2.0**2)),
        ("abs_exp", np.exp(-0.5 * 2.0)),
        ("matern32", (1 + np.sqrt(3) * 1.0) * np.exp(-np.sqrt(3) * 1.0)),
        ("matern52", (1 + np.sqrt(5) + 5 / 3) * np.exp(-np.sqrt(5))),
    ],
)
def test_correlation_closed_form(corr, expected):
    # With theta = 0.5 and a distance of 2, theta * distance = 1 in the Matern
    # formulas and in "abs_exp"; "squar_exp" uses theta * distance**2.
    model = Kriging(corr=corr)
    distance = np.array([[[2.0**model.power]]])
    value = _correlation(corr, distance, np.array([0.5]))
    assert_allclose(value, expected, rtol=1e-12)


@pytest.mark.parametrize("corr", CORRELATIONS)
def test_correlation_is_one_at_zero_and_decays(corr):
    distance = np.linspace(0, 5, 20).reshape(-1, 1, 1)
    value = _correlation(corr, distance, np.array([0.7])).ravel()
    assert value[0] == 1.0
    assert np.all(np.diff(value) < 0)
    assert np.all(value > 0)


@pytest.mark.parametrize("corr", CORRELATIONS)
def test_interpolates_the_training_points(corr, data):
    x, y, _, _ = data
    model = trained(corr, x, y)
    # With no noise the predictor passes through the samples up to the nugget
    # (about 2e-14) amplified by the conditioning of the correlation matrix.
    assert_allclose(model.predict_values(x).ravel(), y, atol=1e-5 * np.ptp(y))


@pytest.mark.parametrize(
    "corr, tolerance",
    [
        ("squar_exp", 0.02),
        ("matern52", 0.05),
        ("matern32", 0.10),
        ("pow_exp", 0.05),
    ],
)
def test_recovers_a_smooth_function(corr, tolerance, data):
    # Tolerances are fractions of the range of the function. They grow as the
    # correlation function gets rougher, and they are well below the range, so a
    # wrong trend or a wrong correlation would fail them. "abs_exp" is too rough
    # for a bound of this kind; it is checked in the convergence test below.
    x, y, x_test, y_test = data
    model = trained(corr, x, y)
    error = np.abs(model.predict_values(x_test).ravel() - y_test).max()
    assert error < tolerance * np.ptp(y_test)


@pytest.mark.parametrize("corr", CORRELATIONS)
def test_error_decreases_with_more_training_points(corr):
    rng = np.random.default_rng(8)
    x_test = rng.random((300, 2))
    y_test = smooth_function(x_test)

    errors = []
    for n_samples in (15, 60):
        x = latin_hypercube([[0, 1], [0, 1]], n_samples, seed=1)
        model = trained(corr, x, smooth_function(x))
        errors.append(np.abs(model.predict_values(x_test).ravel() - y_test).max())

    assert errors[1] < 0.5 * errors[0]


def test_predictions_follow_affine_changes_of_the_output(data):
    x, y, x_test, _ = data
    reference = trained("squar_exp", x, y).predict_values(x_test)
    changed = trained("squar_exp", x, 4.0e6 * y - 7.0e5).predict_values(x_test)
    # The standardized problem is the same, but round-off in the standardization
    # moves the stopping point of the hyperparameter optimizer. The effect is below
    # 1e-6 of the range of the output; the bound leaves a factor of 10.
    expected = 4.0e6 * reference - 7.0e5
    assert_allclose(changed, expected, rtol=0, atol=1e-5 * np.ptp(expected))


def test_predictions_do_not_depend_on_the_units_of_the_inputs(data):
    x, y, x_test, _ = data
    factors = np.array([1e3, 1e-4])
    reference = trained("squar_exp", x, y).predict_values(x_test)
    scaled = trained("squar_exp", x * factors, y).predict_values(x_test * factors)
    # Same optimizer stopping effect as above, below 1e-6 of the range.
    assert_allclose(scaled, reference, rtol=0, atol=1e-5 * np.ptp(reference))


def test_training_is_reproducible_with_the_seed(data):
    x, y, _, _ = data
    first = trained("matern52", x, y, seed=5)
    second = trained("matern52", x, y, seed=5)
    assert_array_equal(first.theta, second.theta)
    assert_array_equal(first.gamma, second.gamma)


@pytest.mark.parametrize(
    "corr", ["squar_exp", "abs_exp", "pow_exp", "matern32", "matern52"]
)
def test_matches_smt_at_fixed_hyperparameters(corr, data):
    smt = pytest.importorskip("smt.surrogate_models")
    x, y, x_test, _ = data
    model = trained(corr, x, y)

    reference = smt.KRG(
        theta0=list(model.theta), corr=corr, hyper_opt="NoOp", print_global=False
    )
    reference.set_training_values(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference.train()

    # Same formulation and same hyperparameters: the only difference is round-off,
    # bounded by the condition number of the correlation matrix times the machine
    # epsilon. The correlation matrices of these designs reach condition numbers
    # of 1e8 or more, so the bound is 1e-6 of the range of the function (the largest
    # difference measured is about 1e-8 of it).
    assert_allclose(model.reduced_likelihood, reference.optimal_rlf_value, rtol=1e-9)
    difference = np.abs(
        model.predict_values(x_test).ravel() - reference.predict_values(x_test).ravel()
    )
    assert difference.max() < 1e-6 * np.ptp(y)


@pytest.mark.parametrize("corr", ["squar_exp", "matern52"])
def test_matches_smt_after_optimizing_the_hyperparameters(corr, data):
    smt = pytest.importorskip("smt.surrogate_models")
    x, y, x_test, _ = data
    model = trained(corr, x, y)

    reference = smt.KRG(theta0=[1e-2], corr=corr, print_global=False)
    reference.set_training_values(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference.train()

    # The optimizers stop at slightly different points (stopping tolerance of the
    # Truncated Newton method and different multistart points), so the comparison is
    # on the predictions, relative to the range of the function.
    difference = np.abs(
        model.predict_values(x_test).ravel() - reference.predict_values(x_test).ravel()
    )
    assert difference.max() < 1e-3 * np.ptp(y)


def test_prediction_only_copy_keeps_the_predictions(data):
    x, y, x_test, _ = data
    model = trained("squar_exp", x, y)
    light = model.prediction_only_copy()

    assert light.xt is None and light.yt is None
    assert model.xt is not None and model.yt is not None
    assert_array_equal(light.predict_values(x_test), model.predict_values(x_test))


def test_print_prediction_option(data, capsys):
    x, y, x_test, _ = data
    model = trained("squar_exp", x, y)
    model.predict_values(x_test[:2])
    assert capsys.readouterr().out == ""
    model.options["print_prediction"] = True
    model.predict_values(x_test[:2])
    assert capsys.readouterr().out != ""


def test_one_dimensional_inputs_are_accepted():
    x = np.linspace(0, 1, 10)
    model = Kriging()
    model.set_training_values(x, np.sin(2 * x))
    model.train()
    assert_allclose(model.predict_values([[0.33]]).item(), np.sin(0.66), atol=1e-4)


def test_invalid_arguments_raise(data):
    x, y, _, _ = data
    with pytest.raises(ValueError, match="corr must be one of"):
        Kriging(corr="squar_sin_exp")
    with pytest.raises(ValueError, match="pow_exp_power"):
        Kriging(pow_exp_power=2.5)

    model = Kriging()
    with pytest.raises(ValueError, match="single output"):
        model.set_training_values(x, np.column_stack([y, y]))
    with pytest.raises(ValueError, match="same number"):
        model.set_training_values(x, y[:-1])
    with pytest.raises(ValueError, match="At least 3"):
        model.set_training_values(x[:2], y[:2])
    with pytest.raises(ValueError, match="finite"):
        model.set_training_values(x, np.where(np.arange(len(y)) == 0, np.nan, y))

    model = Kriging(theta0=100.0)
    model.set_training_values(x, y)
    with pytest.raises(ValueError, match="theta0 must be inside"):
        model.train()

    model = trained("squar_exp", x, y)
    with pytest.raises(ValueError, match="inputs"):
        model.predict_values(np.ones((3, 5)))


def test_training_with_a_repeated_sample_reports_the_failure(data):
    x, y, _, _ = data
    model = Kriging(nugget=0.0)
    model.set_training_values(np.vstack([x, x[:1]]), np.append(y, y[0]))
    with pytest.raises(RuntimeError, match="not positive definite"):
        model.train()


def test_latin_hypercube_is_a_centered_latin_hypercube():
    limits = np.array([[10.0, 20.0], [-1.0, 1.0], [5.0, 5.0]])
    n_samples = 9
    samples = latin_hypercube(limits, n_samples)

    assert samples.shape == (n_samples, 3)
    for column, (low, high) in zip(samples.T, limits, strict=True):
        centers = low + (np.arange(n_samples) + 0.5) / n_samples * (high - low)
        assert_allclose(np.sort(column), centers)


def test_latin_hypercube_seed():
    limits = [[0, 1], [0, 1]]
    assert_array_equal(
        latin_hypercube(limits, 12, seed=2), latin_hypercube(limits, 12, seed=2)
    )
    assert not np.array_equal(
        latin_hypercube(limits, 12, seed=2), latin_hypercube(limits, 12, seed=3)
    )


def test_full_factorial_known_grids():
    assert_allclose(
        full_factorial([[0, 1], [0, 2]], 4), [[0, 0], [0, 2], [1, 0], [1, 2]]
    )
    # 5 points need a 3 x 2 grid of 6 nodes; the first 5 are returned.
    assert_allclose(
        full_factorial([[0, 1], [0, 2]], 5),
        [[0, 0], [0, 2], [0.5, 0], [0.5, 2], [1, 0]],
    )


def test_full_factorial_covers_the_limits():
    limits = np.array([[100.0, 200.0], [1e-4, 2e-4], [300.0, 330.0]])
    samples = full_factorial(limits, 27)
    assert samples.shape == (27, 3)
    assert_allclose(samples.min(axis=0), limits[:, 0])
    assert_allclose(samples.max(axis=0), limits[:, 1])
    assert [len(np.unique(column)) for column in samples.T] == [3, 3, 3]
