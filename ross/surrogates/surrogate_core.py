"""Base classes of the surrogate models.

``CoefficientSurrogate`` holds the metamodel training, ``save``/``load``, ``predict`` and
``benchmark`` logic shared by every bearing and seal surrogate.
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
from sklearn.metrics import r2_score
from ross.bearing_seal_element import BearingElement
from ross.units import Q_, check_units
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS, FullFactorial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from json_tricks import dumps, loads
from ross.surrogates.safe_pickle import loads as pickle_loads
import zipfile
import shutil
import plotly.graph_objects as go
from ross.bearing_seal_element import SealElement
import pandas as pd
import csv
import ast
try:
    import ccp
except Exception as e:
    import warnings
    warnings.warn(f"Could not load ccp: {e}. Some functionality may be unavailable.")
    ccp = None


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
    Predict coefficients;
    Benchmarks between samples.
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
        [+ gas composition derived properties]) shared by every metamodel
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

        if "gas_composition" in inputs and isinstance(inputs["gas_composition"], list):
            xinput = np.vstack((xinput, coefs["gas_constant"]))
            xinput = np.vstack((xinput, coefs["ratio_specific_heats"]))
            xinput = np.vstack((xinput, coefs["tz1"]))
            xinput = np.vstack((xinput, coefs["muz1"]))
            xinput = np.vstack((xinput, coefs["tz2"]))
            xinput = np.vstack((xinput, coefs["muz2"]))

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
        check_min_samples=True,
        frequency=None,
        verify_limits=False,
        **kwargs,
    ):
        """Predict coefficients of bearing with surrogate model.

        Parameters:
        -----------
        verify_limits: bool, optional
            Verify if the input variables are within the training range. Defaults to False.
        name: str, optional
            Name bearning of model. Defaults to "example".
        print_result: bool, optional
            Print results in cmd. Defaults to False.
        print_prediction: bool, optional
            Print prediction in cmd. Defaults to False.
        frequency: array, pint.Quantity
            Frequency trained in the metamodel
        kwargs:
            arg1 : Changed argument for template creation.
            arg2 : Changed argument for template creation.

        Example
        -------

        >>> import ross.surrogates as sg
        >>> _radial_clearance = 2.05e-4
        >>> _reference_temperature = 30
        >>> bearing = sg.CoefficientSurrogate()
        >>> result = bearing.predict(
        ...             name="example1",
        ...             frequency=Q_(1000, "RPM"),
        ...             radial_clearance=_radial_clearance,
        ...             reference_temperature=_reference_temperature)
        """

        _coefs = self._get_coef_names()

        for coef in _coefs:
            setattr(self, f"prediction_{coef}", [])

        time_prediction = 0
        _aux_time = time.time()

        if check_min_samples == True:
            # self.load(name)  # load model
            for i in range(len(frequency)):
                _inputs = np.array([frequency[i]])

                for i in range(len(self.inputs_model["input_variable"])):
                    _inputs = np.append(
                        _inputs, kwargs[self.inputs_model["input_variable"][i]]
                    )

                if "gas_composition" in kwargs:
                    inlet_pressure = kwargs["inlet_pressure"]
                    outlet_pressure = kwargs["outlet_pressure"]
                    inlet_temperature = kwargs["inlet_temperature"]
                    gas_composition = kwargs["gas_composition"]

                    state_in = ccp.State.define(
                        p=inlet_pressure, T=inlet_temperature, fluid=gas_composition
                    )
                    state_out = ccp.State.define(
                        p=outlet_pressure, h=state_in.h(), fluid=gas_composition
                    )

                    gas_constant = (state_in.gas_constant() / state_in.molar_mass()).m
                    _inputs = np.append(_inputs, gas_constant)
                    ratio_specific_heats = (state_in.cp() / state_in.cv()).m
                    _inputs = np.append(_inputs, ratio_specific_heats)
                    tz1 = state_in.T().m
                    _inputs = np.append(_inputs, tz1)
                    muz1 = state_in.viscosity().m
                    _inputs = np.append(_inputs, muz1)
                    tz2 = state_out.T().m
                    _inputs = np.append(_inputs, tz2)
                    muz2 = state_out.viscosity().m
                    _inputs = np.append(_inputs, muz2)

                _inputs = np.vstack(_inputs).T
                for coef in _coefs:
                    getattr(self, f"sm_{coef}").options["print_prediction"] = (
                        print_prediction
                    )
                    getattr(self, f"prediction_{coef}").append(
                        getattr(self, f"sm_{coef}").predict_values(_inputs)
                    )

            if self.inputs_model["model"] in ("maxbrg", "tiltingpad", "thrustpad"):
                _node = self.inputs_model["n"]
            elif self.inputs_model["model"] == "ross":
                _node = self.inputs_model["node"]
            elif self.inputs_model["model"] == "laby3":
                _node = self.inputs_model["n"]
            elif self.inputs_model["model"] == "hcomb":
                _node = self.inputs_model["n"]
            elif self.inputs_model["model"] == "labyrinthseal":
                _node = self.inputs_model["n"]

            if (
                self.inputs_model["model"] in "laby3"
                or self.inputs_model["model"] in "hcomb"
                or self.inputs_model["model"] in "labyrinthseal"
            ):
                result = SealElement(
                    n=_node,
                    kxx=[a.item() for a in self.prediction_kxx],
                    cxx=[a.item() for a in self.prediction_cxx],
                    kyy=[a.item() for a in self.prediction_kyy],
                    kxy=[a.item() for a in self.prediction_kxy],
                    kyx=[a.item() for a in self.prediction_kyx],
                    cyy=[a.item() for a in self.prediction_cyy],
                    cxy=[a.item() for a in self.prediction_cxy],
                    cyx=[a.item() for a in self.prediction_cyx],
                    frequency=frequency,
                )
            elif self.inputs_model["model"] == "thrustpad":
                result = BearingElement(
                    n=_node,
                    kxx=0,
                    cxx=0,
                    kzz=[a.item() for a in self.prediction_kzz],
                    czz=[a.item() for a in self.prediction_czz],
                    frequency=frequency,
                )
            else:
                result = BearingElement(
                    n=_node,
                    kxx=[a.item() for a in self.prediction_kxx],
                    cxx=[a.item() for a in self.prediction_cxx],
                    kyy=[a.item() for a in self.prediction_kyy],
                    kxy=[a.item() for a in self.prediction_kxy],
                    kyx=[a.item() for a in self.prediction_kyx],
                    cyy=[a.item() for a in self.prediction_cyy],
                    cxy=[a.item() for a in self.prediction_cxy],
                    cyx=[a.item() for a in self.prediction_cyx],
                    frequency=frequency,
                )

        else:
            arg1 = np.array(kwargs["arg1"])
            arg2 = np.array(kwargs["arg2"])
            _inputs = np.array([arg1, arg2, frequency])

            for i in range(_inputs[0].size):
                for coef in _coefs:
                    getattr(self, f"prediction_{coef}").append(
                        getattr(self, f"sm_{coef}").predict_values(
                            np.array(
                                [
                                    [_inputs[0, i], _inputs[1, i], _inputs[2, i]],
                                    [_inputs[0, i], _inputs[1, i], _inputs[2, i]],
                                ]
                            )
                        )[0]
                    )

            result = {
                "coefs": {
                    "kxx": self.prediction_kxx,
                    "kxy": self.prediction_kxy,
                    "kyx": self.prediction_kyx,
                    "kyy": self.prediction_kyy,
                    "cxx": self.prediction_cxx,
                    "cxy": self.prediction_cxy,
                    "cyx": self.prediction_cyx,
                    "cyy": self.prediction_cyy,
                },
            }

        time_prediction += time.time() - _aux_time

        if print_result:
            _model = self.inputs_model["metamodel"]
            print(f"Metamodel used: {_model}")
            print(result)
            print(f"Time {name}: {time_prediction * 1000:.2f} ms")

        if verify_limits:
            _, _, _ = self.verify_limits_predict(frequency, kwargs)

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

        Returns:
        --------
        frequency: array
            Frequencies outside the training range.
        inputs: dict
            Input variables outside the training range.
        gas_index: dict
            Gas composition outside the training range.

        Example
        -------

        >>> import ross.surrogates as sg
        >>> _radial_clearance = 2.05e-4
        >>> _reference_temperature = 30
        >>> bearing = sg.CoefficientSurrogate()
        >>> freq, inputs, gas_index = bearing.verify_limits_predict(
        ...     inlet_pressure=Q_(150, "bar"),
        ...     outlet_pressure=Q_(4, "bar"),
        ...     inlet_temperature=Q_(40, "degC"),
        ...     frequency=Q_([16000, 6000], "Hz"),
        ...     gas_composition = {
        ...         "METHANE": 99.287,
        ...         "ETHANE": 5.419,
        ...         "PROPANE": 3.219,
        ...         "N-BUTANE": 1.190,
        ...         "I-BUTANE": 0.500,
        ...         "N-PENTANE":0.760,
        ...         "I-PENTANE": 0.280,
        ...         "N-HEXANE": 0.240,
        ...         "N2": 0.380,
        ...         "H2S": 0.017,
        ...         "CO2": 40.309,
        ...         "WATER": 0.400,
        ...         }
        ...     )
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

        gas_index = {}
        if "gas_composition" in args:
            if max(args["gas_composition"].values()) > 1.0:
                args["gas_composition"] = {
                    k: v / 100 for k, v in args["gas_composition"].items()
                }
            for _k in range(len(args["gas_composition"])):
                _max = max(
                    d[list(args["gas_composition"].keys())[_k]]
                    for d in self.inputs_model["gas_composition"]
                )
                _min = min(
                    d[list(args["gas_composition"].keys())[_k]]
                    for d in self.inputs_model["gas_composition"]
                )
                if (
                    args["gas_composition"][list(args["gas_composition"].keys())[_k]]
                    < _min
                    or args["gas_composition"][list(args["gas_composition"].keys())[_k]]
                    > _max
                ):
                    gas_index[list(args["gas_composition"].keys())[_k]] = args[
                        "gas_composition"
                    ][list(args["gas_composition"].keys())[_k]]
            if gas_index:
                log_warning(
                    f"Warning: gas_composition - {gas_index} outside the training range"
                )

        return frequency, _index_inputs, gas_index

    @check_units
    def benchmark(self, name="example", print_result=False, **kwargs):
        """Performs the performance between the model and the predict , extracting results, times and plot.

        Parameters:
        -----------
        name (str, optional): Name bearning of model. Defaults to "example".
        arg1 (list): Changed argument for template creation.
        arg2 (list): Changed argument for template creation.
        print_result (bool, optional): Print results in cmd. Defaults to True.

        Example
        -------

        >>> import ross.surrogates as sg
        >>> _radial_clearance = np.linspace(1.75e-4, 2.05e-4, 10)
        >>> _reference_temperature = np.linspace(30, 50, 100)
        >>> bearings = sg.CoefficientSurrogate()
        >>> bench = bearings.benchmark(
                name="example1",
                arg1=_radial_clearance2,
                arg2=_reference_temperature2)
        """

        _coefs = [
            "kxx",
            "kxy",
            "kyx",
            "kyy",
            "cxx",
            "cxy",
            "cyx",
            "cyy",
        ]

        # self.load(name)  # load model
        # imported here because the subclasses depend on this module
        from ross.surrogates.bearings import (
            MaxBrgSurrogate,
            PlainJournalSurrogate,
            TiltingPadSurrogate,
        )

        arg1 = np.array(kwargs[self.inputs_model["input_variable"][0]])
        arg2 = np.array(kwargs[self.inputs_model["input_variable"][1]])
        _inputs = np.array([arg1, arg2])

        self.inputs_model[self.inputs_model["input_variable"][0]] = arg1
        self.inputs_model[self.inputs_model["input_variable"][1]] = arg2

        if self.inputs_model["model"] == "ross":
            _node = self.inputs_model["node"]
            _frequency = self.inputs_model["speed"]
        else:
            _node = self.inputs_model["n"]
            _frequency = self.inputs_model["frequency"]
            if self.inputs_model["model"] == "hcomb":
                _frequency = self.inputs_model["speed"]

        valid_cylidrical_bearing = []
        self.valid_coefs = []

        for coef in _coefs:
            setattr(self, f"prediction_{coef}", [])
            self.valid_coefs.append([])

        time_model = 0
        time_kriging = 0

        _aux_time = time.time()

        if self.inputs_model["model"] == "ross":
            valid_cylidrical_bearing = PlainJournalSurrogate(
                metamodel=None,
                input_variable=self.inputs_model["input_variable"],
                axial_length=self.inputs_model["axial_length"],
                journal_radius=self.inputs_model["journal_radius"],
                radial_clearance=self.inputs_model["radial_clearance"],
                elements_circumferential=self.inputs_model["elements_circumferential"],
                elements_axial=self.inputs_model["elements_axial"],
                n_y=self.inputs_model["n_y"],
                n_pad=self.inputs_model["n_pad"],
                pad_arc_length=self.inputs_model["pad_arc_length"],
                reference_temperature=self.inputs_model["reference_temperature"],
                reference_viscosity=self.inputs_model["reference_viscosity"],
                speed=self.inputs_model["speed"],
                load_x_direction=self.inputs_model["load_x_direction"],
                load_y_direction=self.inputs_model["load_y_direction"],
                groove_factor=self.inputs_model["groove_factor"],
                lubricant=self.inputs_model["lubricant"],
                node=self.inputs_model["node"],
                sommerfeld_type=self.inputs_model["sommerfeld_type"],
                initial_guess=self.inputs_model["initial_guess"],
                method=self.inputs_model["method"],
                show_coef=self.inputs_model["show_coef"],
                print_result=self.inputs_model["print_result"],
                print_progress=self.inputs_model["print_progress"],
                print_time=self.inputs_model["print_time"],
            )

        elif self.inputs_model["model"] == "maxbrg":
            valid_cylidrical_bearing = MaxBrgSurrogate(
                metamodel=None,
                input_variable=self.inputs_model["input_variable"],
                n=self.inputs_model["n"],
                frequency=self.inputs_model["frequency"],
                probes=self.inputs_model["probes"],
                total_ex_film=self.inputs_model["total_ex_film"],
                total_ey_film=self.inputs_model["total_ey_film"],
                total_ez_film=self.inputs_model["total_ez_film"],
                total_ey_pad=self.inputs_model["total_ey_pad"],
                bearing_type=self.inputs_model["bearing_type"],
                operating_type=self.inputs_model["operating_type"],
                thermal_type=self.inputs_model["thermal_type"],
                temp_j_type=self.inputs_model["temp_j_type"],
                deform_type=self.inputs_model["deform_type"],
                equilibrium_type=self.inputs_model["equilibrium_type"],
                sump_type=self.inputs_model["sump_type"],
                journal_diameter=self.inputs_model["journal_diameter"],
                radial_clearance=self.inputs_model["radial_clearance"],
                pad_thickness=self.inputs_model["pad_thickness"],
                pivot_angle=self.inputs_model["pivot_angle"],
                pad_arc=self.inputs_model["pad_arc"],
                pad_axial_length=self.inputs_model["pad_axial_length"],
                preload=self.inputs_model["preload"],
                offset=self.inputs_model["offset"],
                taper_depth_le=self.inputs_model["taper_depth_le"],
                taper_arc_le=self.inputs_model["taper_arc_le"],
                taper_depth_te=self.inputs_model["taper_depth_te"],
                taper_arc_te=self.inputs_model["taper_arc_te"],
                pad_convection=self.inputs_model["pad_convection"],
                pad_conductivity=self.inputs_model["pad_conductivity"],
                pad_expansion=self.inputs_model["pad_expansion"],
                pad_density=self.inputs_model["pad_density"],
                pad_E=self.inputs_model["pad_E"],
                pad_poisson=self.inputs_model["pad_poisson"],
                track_arc=self.inputs_model["track_arc"],
                track_axial_length=self.inputs_model["track_axial_length"],
                track_depth=self.inputs_model["track_depth"],
                pocket_arc=self.inputs_model["pocket_arc"],
                pocket_axial_length=self.inputs_model["pocket_axial_length"],
                lubricant=self.inputs_model["lubricant"],
                journal_expansion=self.inputs_model["journal_expansion"],
                shell_expansion=self.inputs_model["shell_expansion"],
                house_diameter=self.inputs_model["house_diameter"],
                pivot_type=self.inputs_model["pivot_type"],
                pivot_diameter=self.inputs_model["pivot_diameter"],
                pivot_stiffness=self.inputs_model["pivot_stiffness"],
                sump_convect_area=self.inputs_model["sump_convect_area"],
                environment_temperature=self.inputs_model["environment_temperature"],
                environment_convection=self.inputs_model["environment_convection"],
                weight=self.inputs_model["weight"],
                fxs_load=self.inputs_model["fxs_load"],
                fys_load=self.inputs_model["fys_load"],
                oil_flow_v=self.inputs_model["oil_flow_v"],
                excit_ratios=self.inputs_model["excit_ratios"],
                xj=self.inputs_model["xj"],
                yj=self.inputs_model["yj"],
                ambient_pressure_1=self.inputs_model["ambient_pressure_1"],
                ambient_pressure_2=self.inputs_model["ambient_pressure_2"],
                cavitation_pressure=self.inputs_model["cavitation_pressure"],
                oil_supply_pressure=self.inputs_model["oil_supply_pressure"],
                oil_supply_temperature=self.inputs_model["oil_supply_temperature"],
                reference_temperature=self.inputs_model["reference_temperature"],
                edges_convection=self.inputs_model["edges_convection"],
                crush_fit=self.inputs_model["crush_fit"],
                shell_id=self.inputs_model["shell_id"],
                shell_od=self.inputs_model["shell_od"],
                journal_temperature=self.inputs_model["journal_temperature"],
                ambient_temperature=self.inputs_model["ambient_temperature"],
                starv_number=self.inputs_model["starv_number"],
                hot_oil_lamda=self.inputs_model["hot_oil_lamda"],
                relax_p=self.inputs_model["relax_p"],
                relax_t=self.inputs_model["relax_t"],
                relax_d=self.inputs_model["relax_d"],
                relax_pivot=self.inputs_model["relax_pivot"],
                re_lower=self.inputs_model["re_lower"],
                re_upper=self.inputs_model["re_upper"],
                weight_e=self.inputs_model["weight_e"],
                weight_h=self.inputs_model["weight_h"],
            )

        elif self.inputs_model["model"] == "tiltingpad":
            valid_cylidrical_bearing = TiltingPadSurrogate(
                metamodel=None,
                input_variable=self.inputs_model["input_variable"],
                frequency=self.inputs_model["frequency"],
                **{
                    name: self.inputs_model[name]
                    for name in TiltingPadSurrogate._bearing_arguments
                },
            )

        elif self.inputs_model["model"] == "laby3":
            valid_cylidrical_bearing = Laby3Surrogate(
                metamodel=None,
                input_variable=self.inputs_model["input_variable"],
                n=self.inputs_model["n"],
                inlet_pressure=self.inputs_model["inlet_pressure"],
                outlet_pressure=self.inputs_model["outlet_pressure"],
                inlet_temperature=self.inputs_model["inlet_temperature"],
                inlet_swirl_velocity=self.inputs_model["inlet_swirl_velocity"],
                frequency=self.inputs_model["frequency"],
                n_teeth=self.inputs_model["n_teeth"],
                shaft_radius=self.inputs_model["shaft_radius"],
                radial_clearance=self.inputs_model["radial_clearance"],
                pitch=self.inputs_model["pitch"],
                tooth_height=self.inputs_model["tooth_height"],
                tooth_width=self.inputs_model["tooth_width"],
                seal_type=self.inputs_model["seal_type"],
                gas_composition=self.inputs_model["gas_composition"],
            )

        elif self.inputs_model["model"] == "hcomb":
            valid_cylidrical_bearing = HcombSurrogate(
                metamodel=None,
                input_variable=self.inputs_model["input_variable"],
                num_inital_samples=self.inputs_model["num_inital_samples"],
                step_num_inital_samples=self.inputs_model["step_num_inital_samples"],
                coef_R2=self.inputs_model["coef_R2"],
                n=self.inputs_model["n"],
                length=self.inputs_model["length"],
                radius=self.inputs_model["radius"],
                clearance=self.inputs_model["clearance"],
                roughness=self.inputs_model["roughness"],
                cell_length=self.inputs_model["cell_length"],
                cell_width=self.inputs_model["cell_width"],
                cell_depth=self.inputs_model["cell_depth"],
                inlet_pressure=self.inputs_model["inlet_pressure"],
                outlet_pressure=self.inputs_model["outlet_pressure"],
                inlet_temperature=self.inputs_model["inlet_temperature"],
                speed=self.inputs_model["speed"],
                preswirl=self.inputs_model["preswirl"],
                entr_coef=self.inputs_model["entr_coef"],
                exit_coef=self.inputs_model["exit_coef"],
                gas_composition=self.inputs_model["gas_composition"],
                print_result=self.inputs_model["print_result"],
                print_time=self.inputs_model["print_time"],
            )

        elif self.inputs_model["model"] == "labyrinthseal":
            valid_cylidrical_bearing = LabyrinthSealSurrogate(
                metamodel=None,
                input_variable=self.inputs_model["input_variable"],
                n=self.inputs_model["n"],
                inlet_pressure=self.inputs_model["inlet_pressure"],
                outlet_pressure=self.inputs_model["outlet_pressure"],
                inlet_temperature=self.inputs_model["inlet_temperature"],
                inlet_swirl_velocity=self.inputs_model["inlet_swirl_velocity"],
                frequency=self.inputs_model["frequency"],
                n_teeth=self.inputs_model["n_teeth"],
                shaft_radius=self.inputs_model["shaft_radius"],
                radial_clearance=self.inputs_model["radial_clearance"],
                pitch=self.inputs_model["pitch"],
                tooth_height=self.inputs_model["tooth_height"],
                tooth_width=self.inputs_model["tooth_width"],
                seal_type=self.inputs_model["seal_type"],
                gas_composition=self.inputs_model["gas_composition"],
            )
        time_model = time.time()
        result_predict = []
        _element = "cylidrical_bearing"
        if self.inputs_model["model"] == "laby3":
            _element = "seal_laby3"
        elif self.inputs_model["model"] == "hcomb":
            _element = "seal_hcomb"
        elif self.inputs_model["model"] == "labyrinthseal":
            _element = "labyrinthseal"

        for i in range(_inputs[0].size):
            # getattr(self, f"prediction_{coef}").append(getattr(self,f"sm_{coef}").predict_values(np.array([[_inputs[0,i],_inputs[1,i]],[_inputs[0,i],_inputs[1,i]]]))[0])
            for num, coef in enumerate(_coefs):
                self.valid_coefs[num].append(
                    getattr(
                        getattr(valid_cylidrical_bearing, _element)[i],
                        f"{coef}",
                    )
                )
                # getattr(self, f"prediction_{coef}").append(getattr(self,f"sm_{coef}").predict_values(np.array([[_inputs[0,i],_inputs[1,i]],[_inputs[0,i],_inputs[1,i]]]))[0])
                getattr(self, f"prediction_{coef}").append(
                    getattr(self, f"sm_{coef}").predict_values(
                        np.array([[arg1[i], arg2[i]], [arg1[i], arg2[i]]])
                    )[0][0]
                )

            if (
                self.inputs_model["model"] in "maxbrg"
                or self.inputs_model["model"] in "ross"
                or self.inputs_model["model"] == "tiltingpad"
            ):
                result_predict.append(
                    BearingElement(
                        n=_node,
                        kxx=self.prediction_kxx[0],
                        cxx=self.prediction_cxx[0],
                        kyy=self.prediction_kyy[0],
                        kxy=self.prediction_kxy[0],
                        kyx=self.prediction_kyx[0],
                        cyy=self.prediction_cyy[0],
                        cxy=self.prediction_cxy[0],
                        cyx=self.prediction_cyx[0],
                        frequency=_frequency,
                    )
                )
            else:
                result_predict.append(
                    SealElement(
                        n=_node,
                        kxx=self.prediction_kxx[0],
                        cxx=self.prediction_cxx[0],
                        kyy=self.prediction_kyy[0],
                        kxy=self.prediction_kxy[0],
                        kyx=self.prediction_kyx[0],
                        cyy=self.prediction_cyy[0],
                        cxy=self.prediction_cxy[0],
                        cyx=self.prediction_cyx[0],
                        frequency=_frequency,
                    )
                )

            if print_result == True:
                print("*------------------------------------------------*")
                for num, coef in enumerate(_coefs):
                    _model = self.inputs_model["model"]
                    _coef_metamodel = getattr(
                        getattr(valid_cylidrical_bearing, _element)[i],
                        f"{coef}",
                    )
                    _metamodel = self.inputs_model["metamodel"]
                    _coef_predict = getattr(result_predict[i], coef)
                    print(f"{coef}_{_model} = {_coef_metamodel}")
                    print(f"{coef}_predict_{_metamodel} = {_coef_predict}")
                    print("*------------------------------------------------*")

        time_kriging = time.time()

        for coef in _coefs:
            setattr(self, f"prediction_{coef}", [])
        for i in range(_inputs[0].size):
            for coef in _coefs:
                getattr(self, f"prediction_{coef}").append(
                    getattr(result_predict[i], coef)
                )

        self._plot(self.inputs_model["name"], self.inputs_model["model"])
        _model = self.inputs_model["model"]
        print(f"Time {_model} model: {time_model - _aux_time:.2f} seg")
        print(f"Time kriging: {time_kriging - time_model:.2f} seg")

    def _plot(self, name="example", model="ross"):
        line_styles = [
            {"color": "goldenrod", "width": 5},
            {"color": "magenta", "width": 4},
            {"color": "green", "width": 3},
            {"color": "red", "width": 2},
        ]
        name_coefs = ["xx", "xy", "yx", "yy"]

        _frequency = self.inputs_model["frequency"]
        if (
            self.inputs_model["model"] in "ross"
            or self.inputs_model["model"] in "hcomb"
        ):
            _frequency = self.inputs_model["speed"]

        for f, _freq in enumerate(_frequency):
            fig_k = go.Figure()
            fig_c = go.Figure()

            for i, y_values in enumerate(
                [
                    self.prediction_kxx,
                    self.prediction_kxy,
                    self.prediction_kyx,
                    self.prediction_kyy,
                ]
            ):
                prediction_k_values = [y_values[y][f] for y in range(len(y_values))]
                fig_k.add_trace(
                    go.Scatter(
                        y=prediction_k_values,
                        name=f"k{name_coefs[i]}",
                        legendgrouptitle_text=name,
                        legendgroup=name,
                        line=line_styles[i],
                    )
                )

            for i, y_values in enumerate(self.valid_coefs[:4]):
                k_values = [y_values[y][f] for y in range(len(y_values))]
                fig_k.add_trace(
                    go.Scatter(
                        y=k_values,
                        name=f"k{name_coefs[i]}",
                        legendgrouptitle_text=model,
                        legendgroup=model,
                        line=dict(line_styles[i], dash="dash"),
                    )
                )

            for i, y_values in enumerate(
                [
                    self.prediction_cxx,
                    self.prediction_cxy,
                    self.prediction_cyx,
                    self.prediction_cyy,
                ]
            ):
                prediction_c_values = [y_values[y][f] for y in range(len(y_values))]
                fig_c.add_trace(
                    go.Scatter(
                        y=prediction_c_values,
                        name=f"c{name_coefs[i]}",
                        legendgrouptitle_text=name,
                        legendgroup=name,
                        line=line_styles[i],
                    )
                )

            for i, y_values in enumerate(self.valid_coefs[4:]):
                c_values = [y_values[y][f] for y in range(len(y_values))]
                fig_c.add_trace(
                    go.Scatter(
                        y=c_values,
                        name=f"c{name_coefs[i]}",
                        legendgrouptitle_text=model,
                        legendgroup=model,
                        line=dict(line_styles[i], dash="dash"),
                    )
                )

            freq_rpm = Q_(_freq, "rad/s").to("rpm").m
            _title = f"Frequency: {freq_rpm} rpm"
            fig_k.update_xaxes(title_text="Sample")
            fig_k.update_yaxes(title_text="N/m")
            fig_k.update_layout(title_text=_title)
            fig_c.update_xaxes(title_text="Sample")
            fig_c.update_yaxes(title_text="Ns/m")
            fig_c.update_layout(title_text=_title)

            fig_k.show()
            fig_c.show()

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

        # valid inlet_pressure > outlet_pressure (test olny laby3)
        if "laby3" in inputs["model"] or "labyrinthseal" in inputs["model"]:
            valid_sampling = sampling.copy()
            filter_samples = valid_sampling[valid_sampling[:, 1] > valid_sampling[:, 2]]
            sampling = filter_samples
        # end

        _name_data = ["frequency"]
        _name_data = np.append(_name_data, inputs["input_variable"])
        df = pd.DataFrame(sampling, columns=_name_data)

        inputs["frequency"] = np.array(df["frequency"])
        for i in range(len(inputs["input_variable"])):
            inputs[inputs["input_variable"][i]] = np.array(
                df[inputs["input_variable"][i]]
            )

        if "gas_composition" in inputs:
            _key, _value = next(iter(inputs["gas_composition"].items()))
            if isinstance(_value, tuple):
                self.name_gases = list(inputs["gas_composition"].keys())
                gas_composition = []
                for _ in range(sampling.shape[0]):
                    if inputs["distribution"] != "FullFactorial":
                        sample = {
                            gas: np.random.uniform(low=min_val, high=max_val)
                            for gas, (min_val, max_val) in inputs[
                                "gas_composition"
                            ].items()
                        }
                    else:
                        rng = np.random.default_rng()
                        sample = {
                            gas: rng.uniform(min_val, max_val)
                            for gas, (min_val, max_val) in inputs[
                                "gas_composition"
                            ].items()
                        }
                    total = sum(sample.values())
                    sample = {gas: value / total for gas, value in sample.items()}
                    gas_composition.append(sample)

                inputs["gas_composition"] = gas_composition

                # Cria os vetores de cada gas
                self.gas = {}
                for sample in inputs["gas_composition"]:
                    for gas_name, value in sample.items():
                        if gas_name not in self.gas:
                            self.gas[
                                gas_name
                            ] = []  # Criar uma lista vazia se ainda não existir
                        self.gas[gas_name].append(value)

        if self.old_num_min_sample != 5 and self.old_num_min_sample != N:
            old_arg1 = np.random.uniform(
                inputs[inputs["input_variable"][0]][0],
                inputs[inputs["input_variable"][0]][1],
                self.old_num_min_sample,
            )
            if len(inputs["input_variable"]) > 1:
                old_arg2 = np.random.uniform(
                    inputs[inputs["input_variable"][1]][0],
                    inputs[inputs["input_variable"][1]][1],
                    self.old_num_min_sample,
                )

            for i in range(self.old_num_min_sample):  # remove arguments calculated
                arg1 = np.delete(arg1, np.where(arg1 == old_arg1[i]))
                if len(inputs["input_variable"]) > 1:
                    arg2 = np.delete(arg2, np.where(arg2 == old_arg2[i]))

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
            if "gas_constant" in coefs and coefs["gas_constant"]:
                fieldnames = np.append(
                    fieldnames,
                    [
                        "gas_constant",
                        "ratio_specific_heats",
                        "tz1",
                        "muz1",
                        "tz2",
                        "muz2",
                    ],
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

        if "gas_constant" in coefs and coefs["gas_constant"]:
            for j in [
                "gas_constant",
                "ratio_specific_heats",
                "tz1",
                "muz1",
                "tz2",
                "muz2",
            ]:
                _sample[j] = coefs[j][-1]

        fieldnames = list(_sample.keys())
        with open(path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(_sample)
            f.close()

