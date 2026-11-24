"""Kriging metamodel and the sampling plans used to train the bearing surrogates.

``Kriging`` is a local implementation of ordinary Kriging (constant trend) with a
stationary correlation function, written with numpy and scipy only. It follows the
formulation of the ``KRG`` model of the Surrogate Modeling Toolbox (SMT), which
ROSS used before, so the surrogates do not need SMT installed:

- inputs and outputs are standardized with the sample mean and standard deviation;
- the correlation hyperparameters ``theta`` (one per input) are found by maximizing
  the concentrated (reduced) likelihood in ``log10(theta)``, with a bounded
  Truncated Newton optimizer and several starting points;
- the prediction is the best linear unbiased predictor, ``beta + r(x) @ gamma``.

References
----------
.. [1] Bouhlel, M. A., Hwang, J. T., Bartoli, N., Lafage, R., Morlier, J.,
   Martins, J. R. R. A. (2019). A Python surrogate modeling framework with
   derivatives. Advances in Engineering Software, 102662.
   https://doi.org/10.1016/j.advengsoft.2019.03.005

Third-party notice
------------------
The model structure, the correlation functions and ``full_factorial`` are derived
from SMT (https://github.com/SMTorg/smt), distributed under the BSD 3-Clause
License:

    Copyright (c) 2017, SMT developers
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys

import numpy as np
from scipy import linalg, optimize
from scipy.stats import qmc

CORRELATIONS = ("pow_exp", "abs_exp", "squar_exp", "matern52", "matern32")

_FAILED_OBJECTIVE = 1e20
_DEFAULT_NUGGET = 100.0 * np.finfo(np.double).eps


def _distance_power(corr, pow_exp_power):
    """Return the exponent applied to the componentwise distances."""
    if corr == "squar_exp":
        return 2.0
    if corr == "pow_exp":
        return pow_exp_power
    return 1.0


def _correlation(corr, distance, theta):
    """Evaluate the correlation function.

    Parameters
    ----------
    corr : str
        Name of the correlation function.
    distance : numpy.ndarray
        Componentwise distances raised to the exponent of ``corr``, with the input
        index in the last axis.
    theta : numpy.ndarray
        Correlation hyperparameters, one per input.

    Returns
    -------
    numpy.ndarray
        Correlation, with the shape of ``distance`` without its last axis.
    """
    scaled = distance * theta
    if corr == "matern32":
        return np.prod(1.0 + np.sqrt(3.0) * scaled, axis=-1) * np.exp(
            -np.sqrt(3.0) * scaled.sum(axis=-1)
        )
    if corr == "matern52":
        return np.prod(
            1.0 + np.sqrt(5.0) * scaled + 5.0 / 3.0 * scaled**2, axis=-1
        ) * np.exp(-np.sqrt(5.0) * scaled.sum(axis=-1))
    return np.exp(-scaled.sum(axis=-1))


def _correlation_gradient(corr, distance, theta, correlation):
    """Evaluate the derivative of the correlation with respect to each ``theta``.

    Returns
    -------
    numpy.ndarray
        Array shaped like ``distance``; the last axis indexes the hyperparameter.
    """
    correlation = correlation[..., np.newaxis]
    if corr == "matern32":
        scaled = distance * theta
        return -3.0 * theta * distance**2 / (1.0 + np.sqrt(3.0) * scaled) * correlation
    if corr == "matern52":
        scaled = distance * theta
        factor = (
            5.0
            / 3.0
            * theta
            * distance**2
            * (1.0 + np.sqrt(5.0) * scaled)
            / (1.0 + np.sqrt(5.0) * scaled + 5.0 / 3.0 * scaled**2)
        )
        return -factor * correlation
    return -distance * correlation


def _componentwise_distance(x, y, power):
    """Return ``|x_i - y_j|**power`` for every pair, shaped (len(x), len(y), n_inputs)."""
    return np.abs(x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** power


class Kriging:
    """Ordinary Kriging metamodel of a scalar function of several inputs.

    The interface mirrors the one of the SMT ``KRG`` model that the bearing
    surrogates used (``set_training_values``, ``train``, ``predict_values``).

    Parameters
    ----------
    corr : str, optional
        Correlation function: "pow_exp", "abs_exp", "squar_exp", "matern52" or
        "matern32". Default is "squar_exp".
    theta0 : float, array_like, optional
        Initial value of the correlation hyperparameters, one value or one per
        input. Default is 1e-2.
    theta_bounds : array_like, optional
        Lower and upper bounds of the hyperparameters, in the units of the
        standardized inputs. Default is (1e-6, 20).
    nugget : float, optional
        Value added to the diagonal of the correlation matrix for numerical
        stability. Default is 100 times the machine epsilon.
    n_start : int, optional
        Number of Latin Hypercube starting points of the hyperparameter
        optimization, in addition to ``theta0`` and one random point. Default is 10.
    pow_exp_power : float, optional
        Exponent of the "pow_exp" correlation, in (0, 2]. Default is 1.9.
    seed : int, optional
        Seed of the random starting points of the optimization. Default is 41.

    Attributes
    ----------
    theta : numpy.ndarray
        Optimal hyperparameters, available after ``train``.
    reduced_likelihood : float
        Value of the log10 reduced likelihood at ``theta``.
    beta : numpy.ndarray
        Constant trend of the standardized output.
    gamma : numpy.ndarray
        Weights of the correlation with each training point.

    Examples
    --------
    >>> import numpy as np
    >>> from ross.surrogates.kriging import Kriging
    >>> x = np.linspace(0, 1, 8).reshape(-1, 1)
    >>> model = Kriging()
    >>> model.set_training_values(x, np.sin(2 * x))
    >>> model.train()
    >>> y = model.predict_values(np.array([[0.5]]))
    >>> bool(abs(y.item() - np.sin(1.0)) < 1e-3)
    True
    """

    def __init__(
        self,
        corr="squar_exp",
        theta0=1e-2,
        theta_bounds=(1e-6, 20.0),
        nugget=_DEFAULT_NUGGET,
        n_start=10,
        pow_exp_power=1.9,
        seed=41,
    ):
        if corr not in CORRELATIONS:
            raise ValueError(f"corr must be one of {CORRELATIONS}, not {corr!r}")
        if not 0.0 < pow_exp_power <= 2.0:
            raise ValueError("pow_exp_power must be in (0, 2]")
        self.corr = corr
        self.theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
        self.theta_bounds = np.asarray(theta_bounds, dtype=float)
        self.nugget = nugget
        self.n_start = n_start
        self.power = _distance_power(corr, pow_exp_power)
        self.seed = seed
        self.options = {"print_prediction": False}
        self.xt = None
        self.yt = None

    def set_training_values(self, xt, yt):
        """Store the training inputs and outputs.

        Parameters
        ----------
        xt : array_like
            Inputs, shaped (n_samples, n_inputs).
        yt : array_like
            Outputs, shaped (n_samples,) or (n_samples, 1).
        """
        xt = np.asarray(xt, dtype=float)
        yt = np.asarray(yt, dtype=float)
        if xt.ndim == 1:
            xt = xt.reshape(-1, 1)
        if yt.ndim == 2 and yt.shape[1] == 1:
            yt = yt.ravel()
        if yt.ndim != 1:
            raise ValueError("Kriging models a single output; yt must be 1-D")
        if len(xt) != len(yt):
            raise ValueError("xt and yt must have the same number of samples")
        if len(xt) < 3:
            raise ValueError("At least 3 training samples are needed")
        if not (np.isfinite(xt).all() and np.isfinite(yt).all()):
            raise ValueError("The training values must be finite")
        self.xt = xt
        self.yt = yt

    def train(self):
        """Fit the hyperparameters, the trend and the weights to the training data."""
        n_samples, n_inputs = self.xt.shape
        tiny = 100.0 * sys.float_info.epsilon

        self.x_offset = self.xt.mean(axis=0)
        self.x_scale = self.xt.std(axis=0, ddof=1)
        self.x_scale[np.abs(self.x_scale) < tiny] = 1.0
        self.y_mean = self.yt.mean()
        self.y_std = self.yt.std(ddof=1)
        if self.y_std == 0.0:
            self.y_std = 1.0
        self.x_norma = (self.xt - self.x_offset) / self.x_scale
        y_norma = (self.yt - self.y_mean) / self.y_std
        distance = _componentwise_distance(self.x_norma, self.x_norma, self.power)

        theta0 = np.ones(n_inputs) * self.theta0
        low, high = self.theta_bounds
        if np.any(theta0 < low) or np.any(theta0 > high):
            raise ValueError(
                f"theta0 must be inside theta_bounds {tuple(self.theta_bounds)}"
            )

        log_bounds = np.log10(self.theta_bounds)
        rng = np.random.default_rng(self.seed)
        starts = [
            np.log10(theta0),
            log_bounds[0] + rng.random(n_inputs) * np.diff(log_bounds),
        ]
        if self.n_start > 1:
            sampler = qmc.LatinHypercube(d=n_inputs, optimization="random-cd", seed=rng)
            unit = sampler.random(self.n_start)
            starts.extend(log_bounds[0] + unit * np.diff(log_bounds))

        def objective(log_theta):
            fit = self._fit(10.0**log_theta, distance, y_norma, gradient=True)
            if fit is None:
                return _FAILED_OBJECTIVE, np.zeros_like(log_theta)
            return -fit["likelihood"], -fit["gradient"] * 10.0**log_theta

        best = None
        for start in starts:
            result = optimize.minimize(
                objective,
                start,
                jac=True,
                method="TNC",
                bounds=[tuple(log_bounds)] * n_inputs,
                options={"maxfun": max(12 * n_inputs, 50)},
            )
            if best is None or result.fun < best.fun:
                best = result

        fit = self._fit(10.0**best.x, distance, y_norma)
        if fit is None:
            raise RuntimeError(
                "Kriging training failed: the correlation matrix is not positive "
                "definite. Try a larger nugget or remove repeated training points."
            )
        self.theta = 10.0**best.x
        self.reduced_likelihood = fit["likelihood"]
        self.beta = fit["beta"]
        self.gamma = fit["gamma"]

    def _fit(self, theta, distance, y_norma, gradient=False):
        """Evaluate the reduced likelihood and the BLUP coefficients at ``theta``.

        The variance of the standardized output is the residual sum of squares divided
        by the number of samples, as in SMT (which takes no degrees of freedom for the
        trend in this term).

        Returns
        -------
        dict or None
            ``likelihood`` (log10), ``beta``, ``gamma`` and, if requested, the
            ``gradient`` of the likelihood with respect to ``theta``. ``None`` if the
            correlation matrix is not positive definite.
        """
        n_samples = len(y_norma)
        correlation = _correlation(self.corr, distance, theta)
        matrix = correlation + self.nugget * np.eye(n_samples)
        try:
            cholesky = linalg.cholesky(matrix, lower=True)
        except linalg.LinAlgError:
            return None

        ones = linalg.solve_triangular(cholesky, np.ones(n_samples), lower=True)
        scaled_y = linalg.solve_triangular(cholesky, y_norma, lower=True)
        beta = ones @ scaled_y / (ones @ ones)
        residual = scaled_y - ones * beta
        sigma2 = residual @ residual / n_samples
        likelihood = (
            -n_samples * np.log10(sigma2) - 2.0 * np.log10(np.diag(cholesky)).sum()
        )
        gamma = linalg.solve_triangular(cholesky.T, residual)
        fit = {"likelihood": likelihood, "beta": beta, "gamma": gamma}

        if gradient:
            inverse = linalg.cho_solve((cholesky, True), np.eye(n_samples))
            derivative = _correlation_gradient(self.corr, distance, theta, correlation)
            grad = np.empty(len(theta))
            for i in range(len(theta)):
                trace = np.sum(inverse * derivative[:, :, i])
                quadratic = gamma @ derivative[:, :, i] @ gamma
                grad[i] = (
                    n_samples * quadratic / (residual @ residual) - trace
                ) / np.log(10.0)
            fit["gradient"] = grad
        return fit

    def predict_values(self, x):
        """Predict the output at the points ``x``.

        Parameters
        ----------
        x : array_like
            Inputs, shaped (n_points, n_inputs).

        Returns
        -------
        numpy.ndarray
            Predicted values, shaped (n_points, 1).
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != self.x_norma.shape[1]:
            raise ValueError(
                f"x has {x.shape[1]} inputs, the model was trained with "
                f"{self.x_norma.shape[1]}"
            )
        x_norma = (x - self.x_offset) / self.x_scale
        distance = _componentwise_distance(x_norma, self.x_norma, self.power)
        correlation = _correlation(self.corr, distance, self.theta)
        y = self.y_mean + self.y_std * (self.beta + correlation @ self.gamma)
        if self.options.get("print_prediction"):
            print(y)
        return y.reshape(-1, 1)

    def prediction_only_copy(self):
        """Return a copy without the raw training data, enough to call ``predict_values``.

        Returns
        -------
        Kriging
            Shallow copy; the original model is left untouched.
        """
        light = object.__new__(type(self))
        light.__dict__.update(self.__dict__)
        light.xt = None
        light.yt = None
        return light


def latin_hypercube(xlimits, n_samples, seed=None):
    """Sample a centered Latin Hypercube inside a box.

    Parameters
    ----------
    xlimits : array_like
        Lower and upper bound of each variable, shaped (n_variables, 2).
    n_samples : int
        Number of points.
    seed : int, numpy.random.Generator, optional
        Seed of the random permutations. By default the sampling is not reproducible.

    Returns
    -------
    numpy.ndarray
        Samples, shaped (n_samples, n_variables).

    Examples
    --------
    >>> samples = latin_hypercube([[0, 10], [100, 200], [3, 3]], 5, seed=0)
    >>> samples.shape
    (5, 3)
    """
    xlimits = np.asarray(xlimits, dtype=float)
    sampler = qmc.LatinHypercube(d=len(xlimits), scramble=False, seed=seed)
    unit = sampler.random(n_samples)
    return xlimits[:, 0] + unit * (xlimits[:, 1] - xlimits[:, 0])


def full_factorial(xlimits, n_samples):
    """Sample a regular grid inside a box with at least ``n_samples`` nodes.

    The number of nodes of each variable is increased in turn until the grid has
    ``n_samples`` points or more; the first ``n_samples`` nodes are returned.

    Parameters
    ----------
    xlimits : array_like
        Lower and upper bound of each variable, shaped (n_variables, 2).
    n_samples : int
        Number of points.

    Returns
    -------
    numpy.ndarray
        Samples, shaped (n_samples, n_variables).

    Examples
    --------
    >>> full_factorial([[0, 1], [0, 2]], 4)
    array([[0., 0.],
           [0., 2.],
           [1., 0.],
           [1., 2.]])
    """
    xlimits = np.asarray(xlimits, dtype=float)
    n_variables = len(xlimits)
    weights = np.ones(n_variables) / n_variables
    nodes = np.ones(n_variables, dtype=int)
    while np.prod(nodes) < n_samples:
        nodes[np.argmax(weights - nodes / np.sum(nodes))] += 1

    axes = [np.linspace(0.0, 1.0, nodes[i]) for i in range(n_variables)]
    grid = np.meshgrid(*axes, indexing="ij")
    unit = np.column_stack([axis.reshape(-1)[:n_samples] for axis in grid])
    return xlimits[:, 0] + unit * (xlimits[:, 1] - xlimits[:, 0])
