"""Surrogate model for plain journal bearings solved with ``ross.PlainJournal``."""

import time
from pathlib import Path
from ross.bearings.plain_journal import PlainJournal
from ross.units import check_units
from tqdm import tqdm
import pandas as pd
import ast
from ross.surrogates.surrogate_core import CoefficientSurrogate


class PlainJournalSurrogate(CoefficientSurrogate):
    """Surrogate of the coefficients of a ``ross.PlainJournal`` bearing.

    The training samples are generated with the ``ross.PlainJournal`` solver and
    the bearing parameters follow its vocabulary (SI units). ``predict()`` returns
    a ``BearingElement`` with the predicted stiffness and damping coefficients.

    Parameters
    ----------
    metamodel : str, optional
        Metamodel used to fit the coefficients, "KGR" (Kriging from SMT) or
        "gaussian" (scikit-learn Gaussian process). Default is "KGR".
    save_sample : str, optional
        Name of the csv file where inputs and outputs of each sample are saved.
    input_variable : list of str
        Variables varied to build the metamodel, at most two. Each one must be
        given as a list [min, max]. Example: ["radial_clearance",
        "oil_supply_temperature"].
    distribution : str, optional
        "LHS" (Latin Hypercube sampling) or "FullFactorial" (regular grid).
        Default is "LHS".
    corr : str, optional
        Correlation function of the Kriging model: "pow_exp", "abs_exp",
        "squar_exp", "matern52" or "matern32". Default is "squar_exp".
    num_inital_samples : int, optional
        Number of samples used to train the metamodel. Default is 100.
    step_num_inital_samples : int, optional
        Increment of samples if the stop requirement is not met.
    coef_R2 : float, optional
        Coefficient of determination used as stop requirement. Default is 0.01.
    n : int, optional
        Node where the bearing is located. Default is 0.
    frequency : list of float, pint.Quantity
        Range [min, max] of rotor speeds, passed to ``PlainJournal`` as ``speed``.
        Default unit is rad/s.
    print_result : bool, optional
        If True, print each bearing after it is computed.
    print_time : bool, optional
        If True, print the total time spent computing the samples.

    The remaining parameters (``pad_axial_length``, ``journal_diameter``,
    ``radial_clearance``, ``n_pads``, ``pad_arc``, ``preload``,
    ``oil_supply_temperature``, ``fxs_load``, ``fys_load``, ``lubricant``,
    ``initial_position``, ``oil_supply_pressure``, ``oil_flow_v``,
    ``pad_thickness``) are those of :class:`ross.bearings.plain_journal.PlainJournal`.
    The ones listed in ``input_variable`` are given as [min, max].

    Examples
    --------
    >>> import ross.surrogates as sg
    >>> from ross.units import Q_
    >>> bearing = sg.PlainJournalSurrogate(
    ...     metamodel="KGR",
    ...     num_inital_samples=10,
    ...     input_variable=["radial_clearance", "oil_supply_temperature"],
    ...     n=3,
    ...     frequency=Q_([900, 1100], "RPM"),
    ...     pad_axial_length=0.263144,
    ...     journal_diameter=0.4,
    ...     radial_clearance=[1.75e-4, 2.05e-4],
    ...     n_pads=2,
    ...     pad_arc=Q_(176, "deg"),
    ...     oil_supply_temperature=Q_([40, 60], "degC"),
    ...     fys_load=-112814.91,
    ...     lubricant="ISOVG32",
    ...     oil_flow_v=Q_(37.86, "l/min"),
    ... )  # doctest: +SKIP
    """

    coef_names = CoefficientSurrogate._default_coef_names

    # Arguments forwarded as-is to ross.PlainJournal (frequency is forwarded as speed)
    _bearing_arguments = (
        "n",
        "pad_axial_length",
        "journal_diameter",
        "radial_clearance",
        "n_pads",
        "pad_arc",
        "preload",
        "oil_supply_temperature",
        "fxs_load",
        "fys_load",
        "lubricant",
        "initial_position",
        "oil_supply_pressure",
        "oil_flow_v",
        "pad_thickness",
    )

    @check_units
    def __init__(
        self,
        metamodel="KGR",
        save_sample=None,
        input_variable=None,
        distribution="LHS",
        corr="squar_exp",
        num_inital_samples=100,
        step_num_inital_samples=None,
        coef_R2=0.01,
        n=0,
        frequency=None,
        pad_axial_length=None,
        journal_diameter=None,
        radial_clearance=None,
        n_pads=None,
        pad_arc=None,
        preload=0.0,
        oil_supply_temperature=None,
        fxs_load=0,
        fys_load=0,
        lubricant=None,
        initial_position=(0.1, -0.1),
        oil_supply_pressure=0,
        oil_flow_v=None,
        pad_thickness=None,
        print_result=False,
        print_time=False,
    ):
        super().__init__()

        _arguments = locals()
        self.input_parameters = {
            "model": "plainjournal",
            "metamodel": metamodel,
            "input_variable": input_variable,
            "save_sample": save_sample,
            "distribution": distribution,
            "corr": corr,
            "num_inital_samples": num_inital_samples,
            "step_num_inital_samples": step_num_inital_samples,
            "coef_R2": coef_R2,
            "frequency": frequency,
            "print_result": print_result,
            "print_time": print_time,
        }
        for name in self._bearing_arguments:
            self.input_parameters[name] = _arguments[name]

        self.cylidrical_bearing = []
        if input_variable is not None:
            self.run()

    def run(self):
        out_coefs = {coef: [] for coef in self.coef_names}

        aux_input_parameters = self.input_parameters.copy()
        if self.input_parameters["metamodel"] != None:
            aux_input_parameters = self._size_samples(
                self.input_parameters["num_inital_samples"], aux_input_parameters
            )
        _aux_input_parameters = aux_input_parameters.copy()

        _init = 0
        if self.input_parameters["save_sample"] is not None:
            if Path(self.input_parameters["save_sample"]).is_file():
                df = pd.read_csv(self.input_parameters["save_sample"])
                if not df.empty:
                    _init = df["index"].max() + 1

        t1 = time.time()
        with tqdm(
            total=len(
                _aux_input_parameters[_aux_input_parameters["input_variable"][0]]
            ),
            desc="Metamodel training",
        ) as progress_bar:
            progress_bar.update(_init - 1)
            for i in range(
                _init,
                len(_aux_input_parameters[_aux_input_parameters["input_variable"][0]]),
            ):
                for j in range(len(_aux_input_parameters["input_variable"])):
                    aux_input_parameters[aux_input_parameters["input_variable"][j]] = (
                        _aux_input_parameters[
                            _aux_input_parameters["input_variable"][j]
                        ][i]
                    )
                    aux_input_parameters["frequency"] = _aux_input_parameters[
                        "frequency"
                    ][i]

                self._blockPrint()  # disable prints

                bearing = PlainJournal(
                    speed=[aux_input_parameters["frequency"]],
                    **{
                        name: aux_input_parameters[name]
                        for name in self._bearing_arguments
                    },
                )
                self.cylidrical_bearing.append(bearing)

                for coef in self.coef_names:
                    out_coefs[coef].append(getattr(self.cylidrical_bearing[i], coef))
                del bearing

                if self.input_parameters["print_result"]:
                    self._enablePrint()  # enable prints
                    print(self.cylidrical_bearing[-1])

                if self.input_parameters["save_sample"] is not None:
                    self._samples_csv(
                        path=self.input_parameters["save_sample"],
                        inputs=aux_input_parameters,
                        coefs=out_coefs,
                    )

                progress_bar.update(1)

        t2 = time.time()
        self._enablePrint()  # enable prints

        if self.input_parameters["save_sample"] is not None:
            out_coefs = {}
            df = pd.read_csv(self.input_parameters["save_sample"])
            for k in self.coef_names:
                if df[k].dtype == "object":
                    out_coefs[k] = [ast.literal_eval(s)[0] for s in df[k]]
                else:
                    out_coefs[k] = list(df[k])

        if self.input_parameters["metamodel"] != None:
            self._set_model(inputs=_aux_input_parameters, coefs=out_coefs)

            new_num_inital_samples = self._find_min_samples(
                inputs=_aux_input_parameters
            )
            if new_num_inital_samples != self.input_parameters["num_inital_samples"]:
                self.input_parameters["num_inital_samples"] = new_num_inital_samples
                self.run()

        if self.input_parameters["print_time"]:
            print(f"Overall Time model PlainJournal: {(t2 - t1) / 60.0:.2f} min")

        return self.cylidrical_bearing
