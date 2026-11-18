"""Surrogate model for tilting-pad bearings solved with ``ross.TiltingPad``."""

import time
from pathlib import Path
from ross.bearings.tilting_pad import TiltingPad
from ross.units import check_units
from tqdm import tqdm
import pandas as pd
import ast
from rossml.surrogates.surrogate_core import CoefficientSurrogate


class TiltingPadSurrogate(CoefficientSurrogate):
    """Calculate bearing with ross TiltingPad.

    Surrogate counterpart of :class:`MaxBrgSurrogate` that uses the
    ``ross.TiltingPad`` solver (ross-rotordynamics >= 3.0) to generate the training
    samples. The bearing parameters follow the ``TiltingPad`` vocabulary (SI units).

    Parameters
    ----------
    metamodel : str
        Metamodel used for generate model result. Defaults to "KGR".
    save_sample : str
        Name file for save inputs and outputs in bearning of model (ex.: "example.csv").
    input_variable : list of str
        Name of variables used for bearing variation. Maximum 2 variables.
        Example ["radial_clearance", "journal_temperature"].
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
        Rotor speeds. It is passed to ``TiltingPad`` as ``speed``.

    Bearing Geometry (Parameters)
    -----------------------------
    The remaining parameters are the ``ross.TiltingPad`` arguments with the same name
    (``journal_diameter``, ``radial_clearance``, ``pad_thickness``, ``pivot_angle``,
    ``pad_arc``, ``pad_axial_length``, ``preload``, ``offset``, ``lubricant``,
    ``oil_supply_temperature``, ``oil_flow_v``, ``weight``, ``journal_temperature``...).
    ``radial_clearance`` and ``journal_temperature`` may be given as a list
    [min, max] when used in ``input_variable``.

    Example
    -------
    >>> import rossml as rsml
    >>> bearing = rsml.TiltingPadSurrogate(
    ...        metamodel="KGR",
    ...        num_inital_samples=500,
    ...        input_variable=["radial_clearance", "journal_temperature"],
    ...        n=0,
    ...        frequency=Q_([900, 1100], "RPM"),
    ...        total_ex_film=10,
    ...        total_ey_film=6,
    ...        total_ez_film=4,
    ...        total_ey_pad=4,
    ...        bearing_type="conventional_tilting_pad",
    ...        thermal_type="full",
    ...        deform_type="pad_mechanical_thermal_shaft_shell_thermal",
    ...        equilibrium_type="match_load",
    ...        journal_diameter=Q_(15.748, "in"),
    ...        radial_clearance=[1.75e-4, 2.05e-4],
    ...        pad_thickness=Q_(5.89034, "in"),
    ...        pivot_angle=Q_([90, 270], "deg"),
    ...        pad_arc=Q_([176, 176], "deg"),
    ...        pad_axial_length=Q_([10.36, 10.36], "in"),
    ...        preload=[0, 0],
    ...        offset=[0.5, 0.5],
    ...        lubricant="TEST",
    ...        pivot_type="rocker_back",
    ...        weight=Q_(25361.8, "lbf"),
    ...        oil_flow_v=Q_(2.64172e1, "gallon/min"),
    ...        oil_supply_temperature=Q_(121.73, "degF"),
    ...        journal_temperature=[303.15, 323.15],
    ...        starvation_number=2,)
    """

    # Arguments forwarded as-is to ross.TiltingPad (frequency is forwarded as speed)
    _bearing_arguments = (
        "n",
        "probes",
        "total_ex_film",
        "total_ey_film",
        "total_ez_film",
        "total_ey_pad",
        "bearing_type",
        "operating_type",
        "thermal_type",
        "temp_j_type",
        "deform_type",
        "equilibrium_type",
        "sump_type",
        "journal_diameter",
        "radial_clearance",
        "pad_thickness",
        "pivot_angle",
        "pad_arc",
        "pad_axial_length",
        "preload",
        "offset",
        "taper_depth_le",
        "taper_arc_le",
        "taper_depth_te",
        "taper_arc_te",
        "pad_convection",
        "pad_conductivity",
        "pad_expansion",
        "pad_density",
        "pad_E",
        "pad_poisson",
        "track_arc",
        "track_axial_length",
        "track_depth",
        "pocket_arc",
        "pocket_axial_length",
        "lubricant",
        "journal_expansion",
        "shell_expansion",
        "house_diameter",
        "pivot_type",
        "pivot_diameter",
        "pivot_stiffness",
        "sump_convect_area",
        "environment_temperature",
        "environment_convection",
        "weight",
        "fxs_load",
        "fys_load",
        "oil_flow_v",
        "excitation_ratio",
        "xj",
        "yj",
        "ambient_pressure_1",
        "ambient_pressure_2",
        "cavitation_pressure",
        "oil_supply_pressure",
        "oil_supply_temperature",
        "reference_temperature",
        "edges_convection",
        "crush_fit",
        "shell_id",
        "shell_od",
        "journal_temperature",
        "starvation_number",
        "hot_oil_lambda",
        "relax_pressure",
        "relax_temperature",
        "relax_deformation",
        "relax_pivot",
        "re_laminar",
        "re_turbulent",
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
        probes=None,
        total_ex_film=30,
        total_ey_film=30,
        total_ez_film=30,
        total_ey_pad=16,
        bearing_type="conventional_tilting_pad",
        operating_type="regular_flooded",
        thermal_type="full",
        temp_j_type="averaged_film_temperature",
        deform_type=None,
        equilibrium_type="match_load",
        sump_type="supply_temperature",
        journal_diameter=None,
        radial_clearance=None,
        pad_thickness=None,
        pivot_angle=None,
        pad_arc=None,
        pad_axial_length=None,
        preload=None,
        offset=None,
        taper_depth_le=None,
        taper_arc_le=None,
        taper_depth_te=None,
        taper_arc_te=None,
        pad_convection=735.9030318060636,
        pad_conductivity=50.0,
        pad_expansion=1.17e-05,
        pad_density=7830.0,
        pad_E=206800000000.0,
        pad_poisson=0.3,
        track_arc=None,
        track_axial_length=None,
        track_depth=None,
        pocket_arc=0,
        pocket_axial_length=0,
        lubricant=None,
        journal_expansion=1.17e-05,
        shell_expansion=1.17e-05,
        house_diameter=0,
        pivot_type="user_specified_stiffness",
        pivot_diameter=0,
        pivot_stiffness=0,
        sump_convect_area=0,
        environment_temperature=294.261,
        environment_convection=735.9030318060637,
        weight=0,
        fxs_load=0,
        fys_load=0,
        oil_flow_v=None,
        excitation_ratio=1,
        xj=None,
        yj=None,
        ambient_pressure_1=0,
        ambient_pressure_2=0,
        cavitation_pressure=0,
        oil_supply_pressure=0,
        oil_supply_temperature=None,
        reference_temperature=297.0388888888889,
        edges_convection=73.59030318060636,
        crush_fit=0,
        shell_id=0,
        shell_od=0,
        journal_temperature=None,
        starvation_number=1,
        hot_oil_lambda=0.8,
        relax_pressure=0.5,
        relax_temperature=1,
        relax_deformation=1,
        relax_pivot=1,
        re_laminar=500,
        re_turbulent=1000,
        print_result=False,
        print_time=False,
    ):
        super().__init__()

        try:
            probes = [(i, j, k.to("m").m) for i, j, k in probes]
        except:
            pass

        _arguments = locals()
        self.input_parameters = {
            "model": "tiltingpad",
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
        if input_variable != None:
            self.run()

    def run(self):
        out_coefs = {
            "kxx": [],
            "kxy": [],
            "kyx": [],
            "kyy": [],
            "cxx": [],
            "cxy": [],
            "cyx": [],
            "cyy": [],
        }

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

                bearing = TiltingPad(
                    speed=[aux_input_parameters["frequency"]],
                    **{
                        name: aux_input_parameters[name]
                        for name in self._bearing_arguments
                    },
                )
                self.cylidrical_bearing.append(bearing)

                out_coefs["kxx"].append(self.cylidrical_bearing[i].kxx)
                out_coefs["kxy"].append(self.cylidrical_bearing[i].kxy)
                out_coefs["kyx"].append(self.cylidrical_bearing[i].kyx)
                out_coefs["kyy"].append(self.cylidrical_bearing[i].kyy)
                out_coefs["cxx"].append(self.cylidrical_bearing[i].cxx)
                out_coefs["cxy"].append(self.cylidrical_bearing[i].cxy)
                out_coefs["cyx"].append(self.cylidrical_bearing[i].cyx)
                out_coefs["cyy"].append(self.cylidrical_bearing[i].cyy)
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
            for k in ["kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"]:
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
            print(f"Overall Time model TiltingPad: {(t2 - t1) / 60.0:.2f} min")

        return self.cylidrical_bearing
