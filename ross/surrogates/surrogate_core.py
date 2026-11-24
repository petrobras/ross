"""Base classes of the surrogate models.

``CoefficientSurrogate`` holds the metamodel training, ``save``/``load`` and ``predict``
logic shared by every bearing surrogate.
"""

import contextlib
import copy
import io
import pickle
import lzma
import time
import warnings
from pathlib import Path
import numpy as np
from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from ross.surrogates.kriging import Kriging, full_factorial, latin_hypercube
from ross.surrogates.safe_pickle import loads as pickle_loads

try:
    from sklearn.metrics import r2_score
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from json_tricks import dumps, loads
    from tqdm import tqdm
except ImportError as error:
    raise ImportError(
        "The surrogate models need optional dependencies that are not installed. "
        'Install them with: pip install "ross-rotordynamics[surrogates]"'
    ) from error
import zipfile
import pandas as pd
import csv
import ast


class GaussianProcessSurrogate:
    """Adapter that exposes the ``Kriging`` interface (``set_training_values``,
    ``train``, ``predict_values``) around scikit-learn's
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor`, so it can be
    used as a drop-in replacement for :class:`~ross.surrogates.kriging.Kriging`
    inside :class:`CoefficientSurrogate`.

    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels.Kernel, optional
        Covariance kernel used by the Gaussian Process. Defaults to a
        constant kernel times an RBF kernel plus a white-noise kernel.
    n_restarts_optimizer : int, optional
        Number of restarts of the optimizer for finding the kernel's
        hyperparameters that maximize the log-marginal likelihood.
    alpha : float, optional
        Value added to the diagonal of the kernel matrix during fitting.
    normalize_y : bool, optional
        Whether to normalize the target values before fitting.
    random_state : int, optional
        Seed used to make the optimizer's outcome deterministic.
    """

    def __init__(
        self,
        kernel=None,
        n_restarts_optimizer=10,
        alpha=1e-10,
        normalize_y=True,
        random_state=0,
    ):
        if kernel is None:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                1.0, (1e-2, 1e2)
            ) + WhiteKernel(1e-3, (1e-10, 1e1))

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
            normalize_y=normalize_y,
            random_state=random_state,
        )
        self.options = {"print_prediction": False}
        self.xt = None
        self.yt = None

    def set_training_values(self, xt, yt):
        """Store training inputs/outputs, mirroring the ``Kriging`` API."""
        self.xt = np.asarray(xt, dtype=float)
        self.yt = np.asarray(yt, dtype=float)

    def train(self):
        """Fit the Gaussian Process on the stored training data."""
        self.x_min = self.xt.min(axis=0)
        span = self.xt.max(axis=0) - self.x_min
        self.x_span = np.where(span > 0, span, 1.0)
        self.model.fit(self._scale(self.xt), self.yt)

    def _scale(self, x):
        """Scale the inputs to [0, 1] with the training range.

        The inputs have very different magnitudes (speed in rad/s, clearance in m),
        which makes the kernel length scales of the Gaussian Process meaningless.
        Models saved before the scaling was introduced have no range and are not
        scaled.
        """
        if getattr(self, "x_min", None) is None:
            return x
        return (x - self.x_min) / self.x_span

    def predict_values(self, x):
        """Predict the mean response for ``x``, shaped like the ``Kriging`` output."""
        x = np.asarray(x, dtype=float)
        y = self.model.predict(self._scale(x))
        if self.options.get("print_prediction"):
            print(y)
        return np.atleast_1d(y).reshape(-1, 1)

    def prediction_only_copy(self):
        """Return a lightweight copy holding only what ``predict_values`` needs.

        ``GaussianProcessRegressor.predict`` (mean only) uses the fitted kernel,
        ``X_train_``, ``alpha_`` and the target normalisation constants. The
        Cholesky factor ``L_`` (n x n), the raw training copies (``xt``/``yt``,
        ``y_train_``) and the cached inverse are only needed for training or
        for predicting variances, and dominate the pickle size. The original
        object is left untouched. ``predict_variances`` is not available on the
        returned copy.
        """
        light_model = copy.copy(self.model)
        for attr in ("L_", "y_train_", "_K_inv", "log_marginal_likelihood_value_"):
            light_model.__dict__.pop(attr, None)

        light = copy.copy(self)
        light.model = light_model
        light.xt = None
        light.yt = None
        return light

    def predict_variances(self, x):
        """Predict the variance of the response for ``x``."""
        x = np.asarray(x, dtype=float)
        _, std = self.model.predict(self._scale(x), return_std=True)
        return np.atleast_1d(std**2).reshape(-1, 1)


class CoefficientSurrogate:
    """Base class of the bearing surrogate models.

    A surrogate samples the inputs of a bearing model, computes the stiffness and
    damping coefficients of each sample with the bearing solver and fits one
    metamodel per coefficient. The trained surrogate predicts the coefficients of a
    new operating point without running the solver again.

    The subclasses define ``_bearing_class`` (the solver), ``_bearing_arguments``
    (the arguments forwarded to it) and fill ``input_parameters``.

    Attributes
    ----------
    cylidrical_bearing : list
        Bearings computed to build the training samples.
    inputs_model : dict
        Inputs of the trained model, with the sampled values of ``frequency`` and
        of each variable in ``input_variable``.
    """

    # Coefficients handled by the metamodels. Subclasses with a different set of
    # outputs (e.g. thrust bearings: kzz, czz) store it in inputs["coef_names"].
    _default_coef_names = ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy")
    coef_names = _default_coef_names

    _kriging_theta0 = [1e-2]

    def _get_coef_names(self):
        inputs = getattr(self, "inputs_model", None) or {}
        return list(inputs.get("coef_names", self._default_coef_names))

    def __init__(self):
        self.cylidrical_bearing = []

    def _set_model(self, inputs=None, coefs=None):
        """Train one metamodel per coefficient.

        Parameters
        ----------
        inputs : dict
            Bearing inputs, with the sampled values.
        coefs : dict
            Coefficients computed for each sample. Invalid samples are removed.
        """
        metamodel = inputs["metamodel"]
        if metamodel == "KGR":

            def build():
                return Kriging(theta0=self._kriging_theta0, corr=inputs["corr"])

        elif metamodel == "gaussian":
            build = GaussianProcessSurrogate
        else:
            raise ValueError(
                f"metamodel must be 'KGR' or 'gaussian', not {metamodel!r}"
            )

        self.inputs_model = inputs
        self.out_coefs = coefs
        xinput, index_error = self._build_training_inputs(inputs, coefs)

        for coef in self._get_coef_names():
            x, y = self._clean_coef_training_data(coef, coefs, index_error, xinput)
            if x is None:
                raise ValueError(f"There are no valid samples to train {coef}.")
            model = build()
            model.set_training_values(x, y)
            model.train()
            setattr(self, f"sm_{coef}", model)

    @staticmethod
    def _build_training_inputs(inputs, coefs):
        """Assembles the training design matrix (frequency + input variables
        shared by every metamodel
        backend, and returns the indices of invalid (``None``) samples.

        Args:
            inputs (dictionary): bearing inputs.
            coefs (dictionary): bearing coefficients.

        Returns:
            tuple: (xinput, index_error)
        """

        xinput = np.array(inputs["frequency"])
        for i in range(len(inputs["input_variable"])):
            xinput = np.vstack((xinput, inputs[inputs["input_variable"][i]]))

        xinput = xinput.T.astype(float)

        _first_coef = inputs.get(
            "coef_names", CoefficientSurrogate._default_coef_names
        )[0]
        index_none = [i for i, v in enumerate(coefs[_first_coef]) if v is None]
        index_error = np.sort(index_none).astype(int)

        if len(index_error) > 0:
            for j in reversed(index_error):
                xinput = np.delete(xinput, j, axis=0)

        return xinput, index_error

    @staticmethod
    def _clean_coef_training_data(coef, coefs, index_error, xinput):
        """Removes invalid samples and coerces a single coefficient's target
        values to a clean float array, filtering out remaining NaNs.

        Args:
            coef (str): coefficient name (e.g. "kxx").
            coefs (dictionary): bearing coefficients (mutated in place).
            index_error (np.ndarray): indices of samples already known invalid.
            xinput (np.ndarray): training design matrix, already filtered by
                ``index_error``.

        Returns:
            tuple: (xinput_filtered, coef_array_filtered) or (None, None) if
            no valid samples remain.
        """

        if len(index_error) > 0:
            for k in reversed(index_error):
                del coefs[coef][k]

        coef_data = []
        for val in coefs[coef]:
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, (list, tuple)):
                        coef_data.append(float(parsed[0]))
                    else:
                        coef_data.append(float(parsed))
                except (ValueError, SyntaxError):
                    val_clean = val.strip().lower()
                    if "nan" in val_clean:
                        coef_data.append(np.nan)
                    else:
                        coef_data.append(float(val))
            else:
                coef_data.append(float(np.ravel(val)[0]))

        coef_array = np.array(coef_data, dtype=float)
        valid_idx = ~np.isnan(coef_array)

        if np.sum(valid_idx) == 0:
            return None, None

        return xinput[valid_idx], coef_array[valid_idx]

    def run(self):
        """Compute the training samples with the bearing solver and train the metamodels.

        Returns
        -------
        bearings : list
            Bearings computed for the training samples.
        """
        parameters = self.input_parameters
        if parameters["metamodel"] not in ("KGR", "gaussian"):
            raise ValueError(
                f"metamodel must be 'KGR' or 'gaussian', not {parameters['metamodel']!r}"
            )

        samples = self._size_samples(
            parameters["num_inital_samples"], parameters.copy()
        )
        variables = ["frequency", *samples["input_variable"]]
        n_samples = len(samples["frequency"])
        coef_names = list(self.coef_names)
        out_coefs = {coef: [] for coef in coef_names}

        n_done = 0
        save_sample = parameters["save_sample"]
        if save_sample is not None and Path(save_sample).is_file():
            saved = pd.read_csv(save_sample).iloc[:n_samples]
            n_done = len(saved)
            for name in variables:
                samples[name][:n_done] = saved[name].to_numpy()

        start_time = time.time()
        sample = samples.copy()
        with tqdm(total=n_samples, initial=n_done, desc="Metamodel training") as bar:
            for i in range(n_done, n_samples):
                for name in variables:
                    sample[name] = samples[name][i]

                with contextlib.redirect_stdout(io.StringIO()):
                    bearing = self._bearing_class(
                        speed=[sample["frequency"]],
                        **{name: sample[name] for name in self._bearing_arguments},
                    )
                self.cylidrical_bearing.append(bearing)

                for coef in coef_names:
                    out_coefs[coef].append(getattr(bearing, coef))

                if parameters["print_result"]:
                    print(bearing)

                if save_sample is not None:
                    self._samples_csv(path=save_sample, inputs=sample, coefs=out_coefs)

                bar.update(1)

        if save_sample is not None:
            saved = pd.read_csv(save_sample).iloc[:n_samples]
            out_coefs = {}
            for coef in coef_names:
                if saved[coef].dtype == "object":
                    out_coefs[coef] = [ast.literal_eval(v)[0] for v in saved[coef]]
                else:
                    out_coefs[coef] = list(saved[coef])

        self._set_model(inputs=samples, coefs=out_coefs)

        if parameters["print_time"]:
            elapsed = (time.time() - start_time) / 60.0
            print(f"Overall time of {type(self).__name__}: {elapsed:.2f} min")

        return self.cylidrical_bearing

    def save(self, file):
        """Save the trained metamodels in a .zip file.

        The file holds one pickle per coefficient and a compressed description of
        the model (inputs and training samples). The metamodels of the object are
        not modified.

        Parameters
        ----------
        file : str, pathlib.Path
            Name of the file. The ``.zip`` extension is added if missing.

        Examples
        --------
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> import ross.surrogates as sg
        >>> file = Path(tempdir) / "tilting_pad_surrogate"
        >>> surrogate = sg.TiltingPadSurrogate(...)  # doctest: +SKIP
        >>> surrogate.save(file)  # doctest: +SKIP
        """
        file = Path(file)
        if file.suffix != ".zip":
            file = file.with_name(file.name + ".zip")

        coef_names = self._get_coef_names()
        description = dict(self.inputs_model)
        for coef in coef_names:
            description[coef] = self.out_coefs[coef]

        with zipfile.ZipFile(file, mode="w") as archive:
            for coef in coef_names:
                model = getattr(self, f"sm_{coef}")
                model = model.prediction_only_copy()
                archive.writestr(f"{coef}.pkl", pickle.dumps(model))

            archive.writestr(
                "description.lzma", lzma.compress(dumps(description).encode("utf-8"))
            )

    @classmethod
    def load(cls, file):
        """Load a surrogate saved with :meth:`save`.

        Parameters
        ----------
        file : str, pathlib.Path
            Name of the .zip file. The ``.zip`` extension is added if missing.

        Returns
        -------
        surrogate : CoefficientSurrogate
            Instance of the class that called ``load``, ready for ``predict``.

        Examples
        --------
        >>> import ross.surrogates as sg
        >>> surrogate = sg.TiltingPadSurrogate.load("tilting_pad_surrogate")  # doctest: +SKIP
        """
        file = Path(file)
        if file.suffix != ".zip":
            file = file.with_name(file.name + ".zip")

        surrogate = cls.__new__(cls)
        CoefficientSurrogate.__init__(surrogate)

        with zipfile.ZipFile(file, mode="r") as archive:
            description = lzma.decompress(archive.read("description.lzma"))
            surrogate.inputs_model = loads(description.decode("utf-8"))
            coef_names = surrogate._get_coef_names()
            surrogate.out_coefs = {
                coef: surrogate.inputs_model[coef]
                for coef in coef_names
                if coef in surrogate.inputs_model
            }
            for coef in coef_names:
                model = pickle_loads(archive.read(f"{coef}.pkl"))
                setattr(surrogate, f"sm_{coef}", model)

        return surrogate

    @check_units
    def predict(
        self,
        frequency=None,
        verify_limits=False,
        print_result=False,
        print_prediction=False,
        **kwargs,
    ):
        """Predict the bearing coefficients with the trained metamodels.

        Parameters
        ----------
        frequency : array, pint.Quantity
            Rotational speeds at which the coefficients are predicted. Default
            unit is rad/s.
        verify_limits : bool, optional
            If True, warn when an input is outside the training range.
        print_result : bool, optional
            If True, print the resulting element and the prediction time.
        print_prediction : bool, optional
            If True, print the raw prediction of each metamodel.
        **kwargs
            Value of each variable listed in ``input_variable`` of the model.

        Returns
        -------
        bearing : ross.BearingElement
            Bearing element holding the predicted coefficients. Thrust bearings
            return only ``kzz`` and ``czz``.

        Examples
        --------
        >>> import ross.surrogates as sg
        >>> from ross.units import Q_
        >>> surrogate = sg.TiltingPadSurrogate.load("tilting_pad_surrogate")  # doctest: +SKIP
        >>> bearing = surrogate.predict(
        ...     frequency=Q_(1000, "RPM"),
        ...     radial_clearance=75e-6,
        ...     oil_supply_temperature=313.15,
        ... )  # doctest: +SKIP
        """
        coef_names = self._get_coef_names()
        for coef in coef_names:
            setattr(self, f"prediction_{coef}", [])

        start_time = time.time()
        frequency = np.atleast_1d(frequency)

        for speed in frequency:
            inputs = np.array([speed])
            for variable in self.inputs_model["input_variable"]:
                inputs = np.append(inputs, kwargs[variable])
            inputs = np.vstack(inputs).T

            for coef in coef_names:
                metamodel = getattr(self, f"sm_{coef}")
                metamodel.options["print_prediction"] = print_prediction
                getattr(self, f"prediction_{coef}").append(
                    metamodel.predict_values(inputs)
                )

        node = self.inputs_model["n"]
        coefficients = {
            coef: [a.item() for a in getattr(self, f"prediction_{coef}")]
            for coef in coef_names
        }

        if self.inputs_model["model"] == "thrustpad":
            result = BearingElement(
                n=node,
                kxx=0,
                cxx=0,
                kzz=coefficients["kzz"],
                czz=coefficients["czz"],
                frequency=frequency,
            )
        else:
            result = BearingElement(n=node, frequency=frequency, **coefficients)

        if print_result:
            print(f"Metamodel used: {self.inputs_model['metamodel']}")
            print(result)
            print(f"Prediction time: {(time.time() - start_time) * 1000:.2f} ms")

        if verify_limits:
            self.verify_limits_predict(frequency, kwargs)

        return result

    @check_units
    def verify_limits_predict(self, frequency=None, args=None):
        """Warn when the inputs of a prediction are outside the training range.

        Parameters
        ----------
        frequency : array, pint.Quantity
            Rotational speeds of the prediction. Default unit is rad/s.
        args : dict
            Value of each variable listed in ``input_variable`` of the model.

        Returns
        -------
        frequency : numpy.ndarray
            Speeds outside the training range.
        inputs : dict
            Input variables outside the training range.

        Examples
        --------
        >>> import ross.surrogates as sg
        >>> from ross.units import Q_
        >>> surrogate = sg.TiltingPadSurrogate.load("tilting_pad_surrogate")  # doctest: +SKIP
        >>> speeds, inputs = surrogate.verify_limits_predict(
        ...     frequency=Q_([900, 5000], "RPM"),
        ...     args={"radial_clearance": 75e-6, "oil_supply_temperature": 313.15},
        ... )  # doctest: +SKIP
        """
        frequency = np.atleast_1d(frequency)
        trained = np.asarray(self.inputs_model["frequency"])
        outside = (frequency < trained.min()) | (frequency > trained.max())
        outside_frequency = frequency[outside]
        if outside_frequency.size:
            warnings.warn(
                f"Frequency {outside_frequency} rad/s outside the training range: "
                f"{trained.min()} to {trained.max()} rad/s."
            )

        outside_inputs = {}
        for variable in self.inputs_model["input_variable"]:
            trained = np.asarray(self.inputs_model[variable])
            value = np.atleast_1d(args[variable])
            if np.any(value < trained.min()) or np.any(value > trained.max()):
                outside_inputs[variable] = args[variable]
                warnings.warn(
                    f"{variable} {args[variable]} outside the training range: "
                    f"{trained.min()} to {trained.max()}."
                )

        return outside_frequency, outside_inputs

    def _size_samples(self, N=5, inputs=None):
        """Generate samples uniformly from past data."""

        if inputs["frequency"].size > 1:
            _limits = np.array([min(inputs["frequency"]), max(inputs["frequency"])])
        else:
            _limits = np.array([inputs["frequency"], inputs["frequency"]])

        for i in range(len(inputs["input_variable"])):
            _limits = np.append(
                _limits,
                np.array(
                    [
                        inputs[inputs["input_variable"][i]][0],
                        inputs[inputs["input_variable"][i]][1],
                    ]
                ),
            )

        _limits = _limits.reshape(len(inputs["input_variable"]) + 1, 2)

        if inputs["distribution"] != "FullFactorial":
            sampling = latin_hypercube(_limits, N)
        else:
            sampling = full_factorial(_limits, N)

        _name_data = ["frequency"]
        _name_data = np.append(_name_data, inputs["input_variable"])
        df = pd.DataFrame(sampling, columns=_name_data)

        inputs["frequency"] = np.array(df["frequency"])
        for i in range(len(inputs["input_variable"])):
            inputs[inputs["input_variable"][i]] = np.array(
                df[inputs["input_variable"][i]]
            )

        return inputs

    @staticmethod
    def or2(out, out_target):
        """Calcultate coefficient of R squared value.

        Parameters:
        -----------
            output (list): output of original model
            predict (list): predict of surrogate model

        Returns:
        --------
        float
            The Coefficient of Determination
        """
        return abs(r2_score(out_target, out, multioutput="variance_weighted"))

    @staticmethod
    def rms_error(out, out_target):
        """Calculates the RMSE.

        Parameters
        ----------
        y : float or numpy.ndarray
            Current data.
        y_target : float or numpy.ndarray
            Target data.

        Returns
        -------
        float or numpy.ndarray
            The error value or array with the values.
        """
        error = np.array(out_target) - np.array(out)
        return np.sqrt(np.sum(error**2) / len(error))

    @staticmethod
    def rms_error_relative(out, out_target):
        """Calculates the RMSE relative.

        Parameters
        ----------
        y : float or numpy.ndarray
            Current data.
        y_target : float or numpy.ndarray
            Target data.

        Returns
        -------
        float or numpy.ndarray
            The error relative value or array with the values.
        """
        error = np.sqrt(np.mean((np.array(out_target) - np.array(out))) ** 2)
        return (error / (len(np.array(out_target)) * np.mean(out_target))) * 100

    def _samples_csv(self, path, inputs, coefs):
        """Save samples in csv file.

        Parameters:
        -----------
        name (str, optional): Name bearning of model. Defaults to "example".
        inputs (dict): Dictionary with all inputs of model.
        """

        id = 0
        if not Path(path).is_file():
            fieldnames = np.array(["index", "frequency"])
            fieldnames = np.append(fieldnames, inputs["input_variable"])
            fieldnames = np.append(
                fieldnames, list(inputs.get("coef_names", self._default_coef_names))
            )
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        df = pd.read_csv(path)
        if not df.empty:
            id = df["index"].max() + 1
        _sample = {}
        _sample["index"] = id
        _sample["frequency"] = inputs["frequency"]
        for j in range(len(inputs["input_variable"])):
            _sample[inputs["input_variable"][j]] = inputs[inputs["input_variable"][j]]
        for k in inputs.get("coef_names", self._default_coef_names):
            _sample[k] = coefs[k][-1]

        fieldnames = list(_sample.keys())
        with open(path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(_sample)
