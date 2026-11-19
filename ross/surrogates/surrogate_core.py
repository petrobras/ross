"""Base classes of the surrogate models.

``CoefficientSurrogate`` holds the metamodel training, ``save``/``load`` and ``predict``
logic shared by every bearing surrogate.
"""

import sys
import os
import copy
import pickle
import lzma
import time
import warnings
from pathlib import Path
import numpy as np
from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from ross.surrogates.safe_pickle import loads as pickle_loads

try:
    from sklearn.metrics import r2_score
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from smt.surrogate_models import KRG
    from smt.sampling_methods import LHS, FullFactorial
    from json_tricks import dumps, loads
    import tqdm  # noqa: F401
except ImportError as error:
    raise ImportError(
        "The surrogate models need optional dependencies that are not installed. "
        'Install them with: pip install "ross-rotordynamics[surrogates]"'
    ) from error
import zipfile
import shutil
import pandas as pd
import csv
import ast


class GaussianProcessSurrogate:
    """Adapter that exposes an SMT-like interface (``set_training_values``,
    ``train``, ``predict_values``) around scikit-learn's
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor`, so it can be
    used as a drop-in replacement for the SMT ``KRG`` model inside
    :class:`CoefficientSurrogate`.

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
        """Store training inputs/outputs, mirroring ``smt``'s API."""
        self.xt = np.asarray(xt, dtype=float)
        self.yt = np.asarray(yt, dtype=float)

    def train(self):
        """Fit the Gaussian Process on the stored training data."""
        self.model.fit(self.xt, self.yt)

    def predict_values(self, x):
        """Predict the mean response for ``x``, shaped like ``smt``'s output."""
        x = np.asarray(x, dtype=float)
        y = self.model.predict(x)
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
        _, std = self.model.predict(x, return_std=True)
        return np.atleast_1d(std**2).reshape(-1, 1)


class CoefficientSurrogate:
    """Class to create a surrogate model based on another model. The class automatically finds the best surrogate structure, i.e., the best combination of method, polynomial function and correlation. This automatization is achieved through the evaluation of error metrics.

    It is possible to:
    ------------------

    Choose the metamodel;
    Find the minimum number of samples;
    Save template;
    Load model;
    Predict coefficients.
    """

    # Coefficients handled by the metamodels. Subclasses with a different set of
    # outputs (e.g. thrust bearings: kzz, czz) store it in inputs["coef_names"].
    _default_coef_names = ("kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy")

    def _get_coef_names(self):
        inputs = getattr(self, "inputs_model", None) or {}
        return list(inputs.get("coef_names", self._default_coef_names))

    def __init__(self):
        self.cylidrical_bearing = []
        self.input_samples = []
        self.output_samples = []
        self.out_kxx = []
        self.out_kxy = []
        self.out_kyx = []
        self.out_kyy = []
        self.out_cxx = []
        self.out_cxy = []
        self.out_cyx = []
        self.out_cyy = []

        self.smt_theta0 = [1e-2]
        self.old_num_min_sample = 5

        self.num_sampling_lhs = 0
        self.lhs_inputs = []

    def _parameterization(self, arg1, arg2):
        qtd_bearing_calculate = len(arg1)

        if qtd_bearing_calculate < len(arg2):
            qtd_bearing_calculate = len(arg2)

            for i in range(qtd_bearing_calculate):
                if i >= len(arg1):
                    arg1 = np.append(arg1, arg1[i - 1])

                if i >= len(arg2):
                    arg2 = np.append(arg2, arg2[i - 1])

        return arg1, arg2

    def _set_model(self, inputs=None, coefs=None):
        """Choose metamodel or AI used.

        Args:
            inputs (dictionary): bearing inputs.
            coefs (dictionary): bearing coefficients.
        """

        self.inputs_model = inputs
        self.out_coefs = coefs
        if inputs["metamodel"] == "KGR":
            self._surrogate_smt_kgr(inputs, coefs)
        elif inputs["metamodel"] == "gaussian":
            self._surrogate_sklearn_gaussian(inputs, coefs)

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
            elif isinstance(val, (list, tuple)):
                coef_data.append(float(val[0]))
            else:
                coef_data.append(float(val))

        coef_array = np.array(coef_data, dtype=float)
        valid_idx = ~np.isnan(coef_array)

        if np.sum(valid_idx) == 0:
            return None, None

        return xinput[valid_idx], coef_array[valid_idx]

    def _surrogate_smt_kgr(self, inputs=None, coefs=None):
        """Performs kriging metamodel training.

        Args:
            inputs (dictionary): bearing inputs.
            coefs (dictionary): bearing coefficients.
        """

        xinput, index_error = self._build_training_inputs(inputs, coefs)

        self.out_coefs = coefs
        _coefs = self._get_coef_names()

        for coef in _coefs:
            xinput_filtered, coef_array_filtered = self._clean_coef_training_data(
                coef, coefs, index_error, xinput
            )
            if xinput_filtered is None:
                continue

            setattr(
                self, f"sm_{coef}", KRG(theta0=self.smt_theta0, corr=inputs["corr"])
            )
            getattr(self, f"sm_{coef}").set_training_values(
                xinput_filtered, coef_array_filtered
            )
            getattr(self, f"sm_{coef}").train()

        print("### Finished training ###")

    def _surrogate_sklearn_gaussian(self, inputs=None, coefs=None):
        """Performs Gaussian Process metamodel training using scikit-learn's
        ``GaussianProcessRegressor`` through the :class:`GaussianProcessSurrogate`
        adapter.

        Args:
            inputs (dictionary): bearing inputs.
            coefs (dictionary): bearing coefficients.
        """

        xinput, index_error = self._build_training_inputs(inputs, coefs)

        self.out_coefs = coefs
        _coefs = self._get_coef_names()

        for coef in _coefs:
            xinput_filtered, coef_array_filtered = self._clean_coef_training_data(
                coef, coefs, index_error, xinput
            )
            if xinput_filtered is None:
                continue

            setattr(self, f"sm_{coef}", GaussianProcessSurrogate())
            getattr(self, f"sm_{coef}").set_training_values(
                xinput_filtered, coef_array_filtered
            )
            getattr(self, f"sm_{coef}").train()

        print("### Finished training (Gaussian Process) ###")

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
                if isinstance(model, GaussianProcessSurrogate):
                    model = model.prediction_only_copy()
                elif isinstance(model, KRG):
                    model = self._stripped_kriging_copy(model)
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

    @staticmethod
    def _stripped_kriging_copy(model):
        """Return a copy of a Kriging model without the internals not needed to predict.

        The training matrices grow with the square of the number of samples and
        dominate the pickle size. They are replaced by empty placeholders while the
        model is copied, so the original model is untouched and the large arrays are
        never duplicated.
        """
        memo = {}
        for name in ("D", "F", "ij", "_thetaMemory"):
            value = getattr(model, name, None)
            if isinstance(value, np.ndarray):
                memo[id(value)] = np.array([])
        memo[id(model.training_points)] = {}
        for name in ("sigma2", "C", "Q", "Ft"):
            value = model.optimal_par.get(name)
            if isinstance(value, np.ndarray):
                memo[id(value)] = np.array([])

        original = model
        model = copy.deepcopy(original, memo)
        model.D = np.array([])
        model.theta0 = np.array([])
        model.optimal_par["sigma2"] = np.array([])
        model.F = np.array([])
        model.optimal_par["C"] = np.array([])
        model.optimal_par["Q"] = np.array([])
        model.optimal_par["Ft"] = np.array([])
        model.F = np.array([])
        model.ij = np.array([])
        model.training_points = {}
        model._correlation_class = {}
        model._train = {}
        model._new_train = {}
        model.corr._abc_impl = {}
        model._abc_impl = {}
        model._check_F = {}
        model._check_param = {}
        model._compute_sigma2 = {}
        model._correct_distances_cat_decreed = {}
        model._final_initialize = {}
        model._initialize = {}
        model._initialize_theta = {}
        model._internal_predict_variance = {}
        model._matrix_data_corr = {}
        model._optimize_hyperparam = {}
        model._post_predict = {}
        model._pre_predict = {}
        model._predict_derivatives = {}
        model._predict_output_derivatives = {}
        model._predict_variance_derivatives = {}
        model._predict_variance_gradient = {}
        model._predict_variances = {}
        model._reduced_likelihood_function = {}
        model._reduced_likelihood_gradient = {}
        model._reduced_likelihood_hessian = {}
        model._thetaMemory = np.array([])
        model.design_space._configs_to_x = {}
        model.design_space._correct_get_acting = {}
        model.design_space._cs_denormalize_x = {}
        model.design_space._cs_denormalize_x_ordered = {}
        model.design_space._design_variables = np.array([])
        model.design_space._get_correct_config = {}
        model.design_space._get_design_variables = {}
        model.design_space._get_n_dim_unfolded = {}
        model.design_space._get_param = {}
        model.design_space._get_param2 = {}
        model.design_space._impute_non_acting = {}
        model.design_space._is_conditionally_acting = {}
        model.design_space._normalize_x = {}
        model.design_space._normalize_x_no_integer = {}
        model.design_space._round_equally_distributed = {}
        model.design_space._sample_valid_x = {}
        model.design_space._to_seed = {}
        return model

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
            sampling = LHS(xlimits=np.array(_limits))(N)
        else:
            sampling = FullFactorial(xlimits=np.array(_limits))(N)

        _name_data = ["frequency"]
        _name_data = np.append(_name_data, inputs["input_variable"])
        df = pd.DataFrame(sampling, columns=_name_data)

        inputs["frequency"] = np.array(df["frequency"])
        for i in range(len(inputs["input_variable"])):
            inputs[inputs["input_variable"][i]] = np.array(
                df[inputs["input_variable"][i]]
            )

        self.old_num_min_sample = N

        return inputs

    def _find_min_samples(self, inputs=None):
        """Finds the minimum amount of samples needed to represent the model."""

        # Latin Hypercube sampling
        xlimits = np.array(
            [
                [
                    min(inputs[inputs["input_variable"][0]]),
                    max(inputs[inputs["input_variable"][0]]),
                ],
                [
                    min(inputs[inputs["input_variable"][1]]),
                    max(inputs[inputs["input_variable"][1]]),
                ],
                [
                    min(inputs["frequency"]),
                    max(inputs["frequency"]),
                ],
            ]
        )

        # if 'gas_composition' in inputs:
        #     _key, _value = next(iter(inputs['gas_composition'].items()))
        #     if isinstance(_value, tuple):
        #         gas_composition = inputs["gas_composition"]
        #         min_max_values = {}

        #         for sample in gas_composition:
        #             for gas, value in sample.items():
        #                 if gas not in min_max_values:
        #                     min_max_values[gas] = {"min": value, "max": value}
        #                 else:
        #                     min_max_values[gas]["min"] = min(min_max_values[gas]["min"], value)
        #                     min_max_values[gas]["max"] = max(min_max_values[gas]["max"], value)

        #         for gas, values in min_max_values.items():
        #             # print(f"{gas}: Mínimo = {values['min']}, Máximo = {values['max']}")
        #             xlimits.append([values['min'], values['max']])

        # --------------------------------- uncoomment init ---------------------------------
        # sampling = LHS(xlimits=xlimits)
        # self.num_sampling_lhs = len(self.out_coefs["kxx"]) - len(self.lhs_inputs)
        # self.lhs_inputs.extend(
        #     sampling(self.num_sampling_lhs)
        # )  # len(self.out_coefs["kxx"]))
        # arg1_LHS = [item[0] for item in self.lhs_inputs]
        # arg2_LHS = [item[1] for item in self.lhs_inputs]
        # arg_freq_LHS = [item[2] for item in self.lhs_inputs]
        # self.predict(
        #     arg1=arg1_LHS, arg2=arg2_LHS, frequency=arg_freq_LHS, print_result=False, check_min_samples=False
        # )

        # _RMSError = self.rms_error(self.prediction_kxx, self.out_coefs["kxx"])
        # _rms_error_relative = self.rms_error_relative(
        #     self.prediction_kxx, self.out_coefs["kxx"]
        # )

        # or2 = self.or2(self.prediction_kxx, self.out_coefs["kxx"])
        # --------------------------------- uncoomment end ---------------------------------
        number_min_sample = inputs["num_inital_samples"]
        # print(f"N. min samples: {number_min_sample}")
        # print(
        #     f"R_square: {or2:.2}"
        # )  # , RMSError: {_RMSError:.2} and RMSError Relative: {_rms_error_relative:.2} ")

        # # if _RMSError < max_RMSError and  _rms_error_relative < max_rms_error_relative:
        # #     return number_min_sample
        # if inputs["coef_R2"] < or2:
        #     return number_min_sample

        # if inputs["step_num_inital_samples"] == None:
        #     number_min_sample = number_min_sample * 2
        # else:
        #     number_min_sample += inputs["step_num_inital_samples"]

        # print(f"New Number min samples: {number_min_sample}")
        # --------------------------------- uncoomment end ---------------------------------

        return number_min_sample

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

    def _blockPrint(self):
        sys.stdout = open(os.devnull, "w")  # Disable prints

    def _enablePrint(self):
        sys.stdout = sys.__stdout__  # Restore prints

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
                f.close()

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
            f.close()
