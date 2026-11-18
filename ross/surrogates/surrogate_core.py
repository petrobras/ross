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

        _first_coef = inputs.get("coef_names", CoefficientSurrogate._default_coef_names)[0]
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
                    if 'nan' in val_clean:
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

    def save(self, name="example"):
        """Save the metamodel in .zip.

        Parameters
        ----------
        name : str or pathlib.Path

        Examples
        --------
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> import ross.surrogates as sg
        >>> file = Path(tempdir) / 'example1'
        >>> _radial_clearance = [1.75e-4, 2.05e-4]
        >>> _journal_temperature = [303.15, 323.15]
        >>> bearing = sg.MaxBrgSurrogate(
        ...        metamodel="KGR",
        ...        name="example1",
        ...        num_inital_samples=30,
        ...        step_num_inital_samples=None,
        ...        coef_R2=0.01,
        ...        input_variable=["radial_clearance", "journal_temperature"],
        ...        n=0,
        ...        frequency=Q_([900, 1100], "RPM"),
        ...        probes=[(1, 80, Q_(3.93701e-2, "in")), (2, 80, Q_(3.93701e-2, "in"))],
        ...        total_ex_film=10,
        ...        total_ey_film=6,
        ...        total_ez_film=4,
        ...        total_ey_pad=4,
        ...        bearing_type="fixed_geometry",
        ...        operating_type="regular_flooded",
        ...        thermal_type="full",
        ...        temp_j_type="averaged_film_temperature",
        ...        deform_type="pad_mechanical_thermal_shaft_shell_thermal",
        ...        equilibrium_type="match_load",
        ...        sump_type="supply_temperature",
        ...        journal_diameter=Q_(15.748, "in"),
        ...        radial_clearance=_radial_clearance,
        ...        pad_thickness=Q_(5.89034, "in"),
        ...        pivot_angle=Q_([90, 270], "deg"),
        ...        pad_arc=Q_([176, 176], "deg"),
        ...        pad_axial_length=Q_([10.36, 10.36], "in"),
        ...        preload=[0, 0],
        ...        offset=[0.5, 0.5],
        ...        lubricant="TEST",
        ...        pivot_type="rocker_back",
        ...        ambient_temperature="supply_temperature",
        ...        pad_convection=Q_(0.00025, "BTU/(s*in²*degF)"),
        ...        pad_E=Q_(30000000, "psi"),
        ...        pad_poisson=0.3,
        ...        pad_conductivity=Q_(0.00067, "BTU/(in*s*degF)"),
        ...        pad_expansion=Q_(6.5e-6, "1/degF"),
        ...        pad_density=Q_(7.322e-4, "lbf*s²/in⁴"),
        ...        journal_expansion=Q_(6.5e-6, "1/degF"),
        ...        shell_expansion=Q_(6.5e-6, "1/degF"),
        ...        weight=Q_(25361.8, "lbf"),
        ...        oil_flow_v=Q_(2.64172e1, "gallon/min"),
        ...        ambient_pressure_1=Q_(14.504, "psi"),
        ...        ambient_pressure_2=Q_(14.504, "psi"),
        ...        oil_supply_temperature=Q_(121.73, "degF"),
        ...        shell_id=Q_(1, "in"),
        ...        shell_od=Q_(2, "in"),
        ...        journal_temperature=_journal_temperature,
        ...        starv_number=2,
        ...        print_result=True,
        ...        print_time=True,)
        >>> bearing.save(file)
        """
        _coefs = self._get_coef_names()
        files = [
            f"{coef}.pkl" for coef in _coefs
        ]  # ["description.lzma", "kxx.pkl", "kxy.pkl", "kyx.pkl", "kyy.pkl", "cxx.pkl", "cxy.pkl", "cyx.pkl", "cyy.pkl"]
        files.append("description.lzma")

        newpath = Path(f"{name}")
        if not newpath.exists():
            newpath.mkdir()
        description_path = newpath / f"{files[-1]}"

        for coef in _coefs:
            self.inputs_model[coef] = self.out_coefs[coef]
            model_obj = getattr(self, f"sm_{coef}")

            if isinstance(model_obj, KRG):
                # Strip large SMT-internal caches before pickling (KRG-specific).
                model_obj.D = np.array([])
                model_obj.theta0 = np.array([])
                model_obj.optimal_par["sigma2"] = np.array([])
                model_obj.F = np.array([])
                model_obj.optimal_par["C"] = np.array([])
                model_obj.optimal_par["Q"] = np.array([])
                model_obj.optimal_par["Ft"] = np.array([])
                model_obj.F = np.array([])
                model_obj.ij = np.array([])
                model_obj.training_points = {}
                model_obj._correlation_class = {}
                model_obj._train = {}
                model_obj._new_train = {}
                model_obj.corr._abc_impl = {}
                model_obj._abc_impl = {}
                model_obj._check_F = {}
                model_obj._check_param = {}
                model_obj._compute_sigma2 = {}
                model_obj._correct_distances_cat_decreed = {}
                model_obj._final_initialize = {}
                model_obj._initialize = {}
                model_obj._initialize_theta = {}
                model_obj._internal_predict_variance = {}
                model_obj._matrix_data_corr = {}
                model_obj._optimize_hyperparam = {}
                model_obj._post_predict = {}
                model_obj._pre_predict = {}
                model_obj._predict_derivatives = {}
                model_obj._predict_output_derivatives = {}
                model_obj._predict_variance_derivatives = {}
                model_obj._predict_variance_gradient = {}
                model_obj._predict_variances = {}
                model_obj._reduced_likelihood_function = {}
                model_obj._reduced_likelihood_gradient = {}
                model_obj._reduced_likelihood_hessian = {}
                model_obj._thetaMemory = np.array([])
                model_obj.design_space._configs_to_x = {}
                model_obj.design_space._correct_get_acting = {}
                model_obj.design_space._cs_denormalize_x = {}
                model_obj.design_space._cs_denormalize_x_ordered = {}
                model_obj.design_space._design_variables = np.array([])
                model_obj.design_space._get_correct_config = {}
                model_obj.design_space._get_design_variables = {}
                model_obj.design_space._get_n_dim_unfolded = {}
                model_obj.design_space._get_param = {}
                model_obj.design_space._get_param2 = {}
                model_obj.design_space._impute_non_acting = {}
                model_obj.design_space._is_conditionally_acting = {}
                model_obj.design_space._normalize_x = {}
                model_obj.design_space._normalize_x_no_integer = {}
                model_obj.design_space._round_equally_distributed = {}
                model_obj.design_space._sample_valid_x = {}
                model_obj.design_space._to_seed = {}

            if isinstance(model_obj, GaussianProcessSurrogate):
                # Keep only the variables required by predict_values().
                model_obj = model_obj.prediction_only_copy()

            filename = newpath / f"{coef}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(model_obj, f)

        with lzma.LZMAFile(description_path, mode="w") as f:
            json = dumps(self.inputs_model)
            f.write(json.encode("utf-8"))

        with zipfile.ZipFile(f"{name}.zip", mode="w") as archive:
            for file_path in files:
                file = newpath / f"{file_path}"
                archive.write(file, arcname=file_path)

        try:
            shutil.rmtree(newpath)
            _metamodel = self.inputs_model["metamodel"]
            print(f"*** Model {name} with Metamodel {_metamodel}: Save ***")
        except OSError as o:
            print(f"Error, {o.strerror}: {newpath}")

    @classmethod
    def load(self, name="example"):
        """Rescues parameters and calculated model.

        Parameters:
        -----------
        name (str): Name bearning of model. Defaults to "example".

        Example
        -------

        >>> import ross.surrogates as sg
        >>> bearing = sg.CoefficientSurrogate()
        >>> model = bearing.load(name="example1")
        """

        try:
            with zipfile.ZipFile(f"{name}.zip", mode="r") as archive:
                with archive.open("description.lzma", mode="r") as description:
                    with lzma.LZMAFile(description, "r") as f:
                        config = f.read()
                        self.inputs_model = loads(config.decode("utf-8"))
            archive.close()

            _coefs = self.inputs_model.get("coef_names", self._default_coef_names)

            for coef in _coefs:
                filename = f"{coef}.pkl"
                with zipfile.ZipFile(f"{name}.zip", mode="r") as archive:
                    with archive.open(filename, mode="r") as pkl_file:
                        setattr(self, f"sm_{coef}", pickle_loads(pkl_file.read()))
                archive.close()

            return self.inputs_model

        except zipfile.BadZipFile as error:
            print(error)

    @check_units
    def predict(
        self,
        name="example",
        print_result=False,
        print_prediction=False,
        frequency=None,
        verify_limits=False,
        **kwargs,
    ):
        """Predict the bearing coefficients with the trained metamodels.

        Parameters
        ----------
        name : str, optional
            Name of the model, used only in the printed output.
        print_result : bool, optional
            If True, print the resulting element and the prediction time.
        print_prediction : bool, optional
            If True, print the raw prediction of each metamodel.
        frequency : array, pint.Quantity
            Rotational speeds at which the coefficients are predicted. Default
            unit is rad/s.
        verify_limits : bool, optional
            If True, warn when an input is outside the training range.
        **kwargs
            Value of each variable listed in ``input_variable`` of the model.

        Returns
        -------
        ross.BearingElement
            Bearing element holding the predicted coefficients. Thrust bearings
            return only ``kzz`` and ``czz``.
        """
        coef_names = self._get_coef_names()
        for coef in coef_names:
            setattr(self, f"prediction_{coef}", [])

        start_time = time.time()

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
            print(f"Time {name}: {(time.time() - start_time) * 1000:.2f} ms")

        if verify_limits:
            self.verify_limits_predict(frequency, kwargs)

        return result

    @check_units
    def verify_limits_predict(self, frequency=None, args=None):
        """Verify if the input variables are within the training range.

        Parameters:
        -----------
        frequency: array, pint.Quantity
            Frequency trained in the metamodel
        kwargs:
            arg1 : Changed argument for template creation.
            arg2 : Changed argument for template creation.

        Returns
        -------
        tuple
            Frequencies and input variables outside the training range.

        Examples
        --------
        >>> import ross.surrogates as sg
        >>> bearing = sg.TiltingPadSurrogate()
        >>> freq, inputs = bearing.verify_limits_predict(
        ...     frequency=Q_([900, 1100], "RPM"),
        ...     args={"radial_clearance": 2.05e-4},
        ... )  # doctest: +SKIP
        """

        _index = []
        for _num in range(len(frequency)):
            if (
                frequency[_num] < self.inputs_model["frequency"].min()
                or frequency[_num] > self.inputs_model["frequency"].max()
            ):
                _index.append(_num)
        if _index:
            frequency = frequency[np.array(_index)]
            log_warning(
                f"Warning: Frequency {frequency[np.array(_index)]} outside the training range: {self.inputs_model['frequency'].min()} to {self.inputs_model['frequency'].max()}"
            )
            
        else:
            frequency = np.zeros(len(frequency))

        _index_inputs = {}
        for _num in range(len(self.inputs_model["input_variable"])):
            if bool(
                args[self.inputs_model["input_variable"][_num]]
                < self.inputs_model[self.inputs_model["input_variable"][_num]].min()
            ) or bool(
                args[self.inputs_model["input_variable"][_num]]
                > self.inputs_model[self.inputs_model["input_variable"][_num]].max()
            ):
                _index_inputs[self.inputs_model["input_variable"][_num]] = args[
                    self.inputs_model["input_variable"][_num]
                ]
                log_warning(
                    f"Warning: {self.inputs_model['input_variable'][_num]} - {args[self.inputs_model['input_variable'][_num]]} outside the training range: {np.min(self.inputs_model[self.inputs_model['input_variable'][_num]])} to {np.max(self.inputs_model[self.inputs_model['input_variable'][_num]])}"
                )

        return frequency, _index_inputs

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

