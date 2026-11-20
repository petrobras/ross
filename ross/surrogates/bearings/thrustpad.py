"""Surrogate model for tilting-pad thrust bearings solved with ``ross.ThrustPad``."""

import time
from pathlib import Path
from ross.bearings.thrust_pad import ThrustPad
from ross.units import check_units
from tqdm import tqdm
import pandas as pd
import ast
from ross.surrogates.surrogate_core import CoefficientSurrogate


class ThrustPadSurrogate(CoefficientSurrogate):
    """Calculate thrust bearing with ross ThrustPad.

    Surrogate of the axial coefficients (``kzz`` and ``czz``) of the
    ``ross.ThrustPad`` solver (ross-rotordynamics >= 3.0). The bearing parameters
    follow the ``ThrustPad`` vocabulary (SI units). ``predict()`` returns a
    ``BearingElement`` with the predicted ``kzz`` and ``czz``.

    Parameters
    ----------
    metamodel : str
        Metamodel used for generate model result. Defaults to "KGR".
    save_sample : str
        Name file for save inputs and outputs in bearning of model (ex.: "example.csv").
    input_variable : list of str
        Name of variables used for bearing variation. Maximum 2 variables.
        Example ["axial_load", "oil_supply_temperature"].
    distribution : str
        LHS - Latin Hypercube sampling (default)
        FullFactorial - Regular grid sampling
    corr : str
        Correlation function used in the metamodel. The options are
        [pow_exp, abs_exp, squar_exp, matern52, matern32]. Default is "squar_exp".
    num_inital_samples : int, optional (Default: 100)
        Number of samples used to create the metamodel.
    step_num_inital_samples : int, optional (Default: None)
        Increment of samples if the stop requirement is not met.
    coef_R2 : float, optional (Default: 0.01)
        Coefficient of determination used as stop requirement.
    frequency : pint.Quantity or list [min, max] values
        Rotor speeds. It is passed to ``ThrustPad`` as ``speed``.

    Bearing Geometry (Parameters)
    -----------------------------
    The remaining parameters are the ``ross.ThrustPad`` arguments with the same name
    (``pad_inner_radius``, ``pad_outer_radius``, ``pad_pivot_radius``, ``pad_arc``,
    ``pivot_angle``, ``lubricant``, ``n_pads``, ``axial_load``,
    ``oil_supply_temperature``...). ``axial_load`` and ``oil_supply_temperature`` may
    be given as a list [min, max] when used in ``input_variable``.

    Example
    -------
    >>> import ross.surrogates as sg
    >>> bearing = sg.ThrustPadSurrogate(
    ...        metamodel="KGR",
    ...        num_inital_samples=500,
    ...        input_variable=["axial_load", "oil_supply_temperature"],
    ...        n=1,
    ...        frequency=Q_([80, 100], "RPM"),
    ...        pad_inner_radius=Q_(1150, "mm"),
    ...        pad_outer_radius=Q_(1725, "mm"),
    ...        pad_pivot_radius=Q_(1442.5, "mm"),
    ...        pad_arc=Q_(26, "deg"),
    ...        pivot_angle=Q_(15, "deg"),
    ...        lubricant="ISOVG68",
    ...        n_pads=12,
    ...        n_theta=10,
    ...        n_radial=10,
    ...        equilibrium_position_mode="calculate",
    ...        axial_load=[10.0e6, 16.0e6],
    ...        oil_supply_temperature=[308.15, 323.15],
    ...        radial_inclination_angle=Q_(-2.75e-04, "rad"),
    ...        circumferential_inclination_angle=Q_(-1.70e-05, "rad"),
    ...        initial_film_thickness=Q_(0.2, "mm"),)
    """

    coef_names = ("kzz", "czz")

    # Arguments forwarded as-is to ross.ThrustPad (frequency is forwarded as speed)
    _bearing_arguments = (
        "n",
        "pad_inner_radius",
        "pad_outer_radius",
        "pad_pivot_radius",
        "pad_arc",
        "pivot_angle",
        "oil_supply_temperature",
        "lubricant",
        "n_pads",
        "n_theta",
        "n_radial",
        "equilibrium_position_mode",
        "radial_inclination_angle",
        "circumferential_inclination_angle",
        "initial_film_thickness",
        "tolerance_force_moment",
        "residual_force_moment",
        "model_type",
        "axial_load",
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
        pad_inner_radius=None,
        pad_outer_radius=None,
        pad_pivot_radius=None,
        pad_arc=None,
        pivot_angle=None,
        oil_supply_temperature=None,
        lubricant=None,
        n_pads=None,
        n_theta=10,
        n_radial=10,
        equilibrium_position_mode="calculate",
        radial_inclination_angle=None,
        circumferential_inclination_angle=None,
        initial_film_thickness=None,
        tolerance_force_moment=0.1,
        residual_force_moment=50,
        model_type="thermo_hydro_dynamic",
        axial_load=None,
        print_result=False,
        print_time=False,
    ):
        super().__init__()

        _arguments = locals()
        self.input_parameters = {
            "model": "thrustpad",
            "metamodel": metamodel,
            "coef_names": list(self.coef_names),
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
        if input_variable != None:
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

                bearing = ThrustPad(
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
            print(f"Overall Time model ThrustPad: {(t2 - t1) / 60.0:.2f} min")

        return self.cylidrical_bearing
