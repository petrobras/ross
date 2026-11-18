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
    """This class calculates the pressure and temperature field in oil film of a cylindrical bearing. It is also possible to obtain the stiffness and
    damping coefficients.

    Parameters
        ----------
        save_sample : str
            Name file for save inputs and outputs in bearning of model (ex.: "example.csv").
        model : str
            Metamodel used for generate model result. Defaults to "KGR".
        input_variable : list of str
            Name of variables used for bearing variation. Maximum 2 variables. Example ["radial_clearance","reference_temperature"].
        num_inital_samples: int, optinonal (Default: 100)
            Number that initialize minimum samples in "input_variable" for create metamodel.
        step_num_inital_samples : int, optional (Default: None)
            Increases the number of samples beyond the previous value if the stop requirement is not met. If this variable is declared as "None", the previous value is doubled.
        coef_R2: float, optional (Default: 0.9)
            The coefficient of determination, R squared (R²), is a measure that provides information about the goodness of fit of a model. The value between 0.01 and 0.99 is the variation between the calculated values and accounted for by the metamodel values. The closer to 1, the more similar.

    Bearing Geometry
    ----------------
    Describes the geometric characteristics.
    axial_length : float, pint.Quantity or list [min, max] values
        Bearing length. Default unit is meter.
    journal_radius : float or list [min, max] values
        Rotor radius. The unit is meter.
    radial_clearance : float or list [min, max] values
        Radial clearence between rotor and bearing. The unit is meter.
    n_pad : integer
        Number of pads that compound the bearing surface.
    pad_arc_length : float
        Arc length of each pad. The unit is degree.
    initial_guess : array
        Array with eccentricity ratio and attitude angle
    method : string
        Choose the method to calculate the dynamics coefficients. Options are:
        - 'lund'
        - 'perturbation'
    print_progress : bool
        Set it True to print the score and forces on each iteration.
        False by default.
    print_result : bool
        Set it True to print result at the end.
        False by default.
    print_time : bool
        Set it True to print the time at the end.
        False by default.

    Operation conditions
    --------------------
    Describes the operation conditions of the bearing.
    speed : list of float, pint.Quantity
        Rotor rotational speed. Default unit is rad/s.
    load_x_direction : Float
        Load in X direction. The unit is newton.
    load_y_direction : Float
        Load in Y direction. The unit is newton.

    Fluid propierties
    -----------------
    Describes the fluid characteristics.

    lubricant : str
        Lubricant type. Can be:
        - 'ISOVG46'
    reference_temperature : float
        Oil reference temperature. The unit is celsius.
    reference_viscosity : float
        Oil viscosity at reference temperature. Unit is Pa.s.
    groove_factor : list, numpy array, tuple or float
        Ratio of oil in reservoir temperature that mixes with the circulating oil.
        Is required one factor per segment.


    Turbulence Model
    ----------------
    Turbulence model to improve analysis in higher speed.The model represents
    the turbulence by eddy diffusivities. The basic reference is found in :cite:t:`suganami1979`

    Reyn : Array
        The Reynolds number is a dimensionless number used to calculate the
        fluid flow regime inside the bearing.
    delta_turb : float
        Eddy viscosity scaling factor. Coefficient to assign weight to laminar,
        transitional and turbulent flows to calculate viscosity.

    Mesh discretization
    -------------------
    Describes the discretization of the bearing.
    elements_circumferential : int
        Number of volumes along the direction theta (direction of flow).
    elements_axial : int
        Number of volumes along the Z direction (axial direction).

    Returns
    -------
    A model PlainJournal object.

    Examples
    --------
    >>> import ross.surrogates as sg
    >>> _radial_clearance = [1.75e-4, 2.05e-4]
    >>> _reference_temperature = [30, 50]
    >>> bearing = sg.PlainJournal(
    ...         metamodel="KGR",
    ...         input_variable=["radial_clearance","reference_temperature"],
    ...         num_inital_samples=5,
    ...         step_num_inital_samples=None,
    ...         coef_R2=0.01,
    ...         axial_length=(0.263144),
    ...         journal_radius=(0.2),
    ...         radial_clearance=_radial_clearance,
    ...         elements_circumferential=11,
    ...         elements_axial=3,
    ...         n_pad=2,
    ...         pad_arc_length=176,
    ...         reference_temperature=_reference_temperature,
    ...         reference_viscosity=0.02,
    ...         speed=Q_([900], "RPM"),
    ...         load_x_direction=0,
    ...         load_y_direction=-112814.91,
    ...         groove_factor=[0.52, 0.48],
    ...         lubricant="ISOVG32",
    ...         node=3,
    ...         sommerfeld_type=2,
    ...         initial_guess=[0.1, -0.1],
    ...         method="perturbation",
    ...         injection_pressure=0,
    ...         oil_flow=37.86,
    ...         show_coef=False,
    ...         print_result=True,
    ...         print_progress=False,
    ...         print_time=True,)
    """

    @check_units
    def __init__(
        self,
        metamodel="KGR",
        save_sample=None,
        input_variable=None,
        num_inital_samples=100,
        step_num_inital_samples=None,
        coef_R2=0.01,
        axial_length=None,
        journal_radius=None,
        radial_clearance=None,
        elements_circumferential=None,
        elements_axial=None,
        n_y=None,
        n_pad=None,
        pad_arc_length=None,
        reference_temperature=None,
        reference_viscosity=None,
        speed=None,
        load_x_direction=None,
        load_y_direction=None,
        groove_factor=None,
        lubricant=None,
        node=None,
        sommerfeld_type=2,
        initial_guess=[0.1, -0.1],
        method="perturbation",
        operating_type="flooded",
        injection_pressure=0,
        oil_flow=37.86,
        show_coef=False,
        print_result=False,
        print_progress=False,
        print_time=False,
    ):
        super().__init__()

        self.input_parameters = {
            "model": "ross",
            "metamodel": metamodel,
            "input_variable": input_variable,
            "save_sample": save_sample,
            "num_inital_samples": num_inital_samples,
            "step_num_inital_samples": step_num_inital_samples,
            "coef_R2": coef_R2,
            "axial_length": axial_length,
            "journal_radius": journal_radius,
            "radial_clearance": radial_clearance,
            "elements_circumferential": elements_circumferential,
            "elements_axial": elements_axial,
            "n_y": n_y,
            "n_pad": n_pad,
            "pad_arc_length": pad_arc_length,
            "reference_temperature": reference_temperature,
            "reference_viscosity": reference_viscosity,
            "speed": speed,
            "load_x_direction": load_x_direction,
            "load_y_direction": load_y_direction,
            "lubricant": lubricant,
            "node": node,
            "groove_factor": groove_factor,
            "equilibrium_pos": None,
            "sommerfeld_type": sommerfeld_type,
            "initial_guess": initial_guess,
            "method": method,
            "operating_type": operating_type,
            "injection_pressure": injection_pressure,
            "oil_flow": oil_flow,
            "show_coef": show_coef,
            "print_result": print_result,
            "print_progress": print_progress,
            "print_time": print_time,
        }

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
                aux_input_parameters[aux_input_parameters["input_variable"][0]] = (
                    _aux_input_parameters[_aux_input_parameters["input_variable"][0]][i]
                )
                if len(self.input_parameters["input_variable"]) > 1:
                    aux_input_parameters["frequency"] = _aux_input_parameters[
                        "frequency"
                    ][i]
                    aux_input_parameters[aux_input_parameters["input_variable"][1]] = (
                        _aux_input_parameters[
                            _aux_input_parameters["input_variable"][1]
                        ][i]
                    )

                self._blockPrint()  # disable prints

                bearing = PlainJournal(
                    axial_length=aux_input_parameters["axial_length"],
                    journal_radius=aux_input_parameters["journal_radius"],
                    radial_clearance=aux_input_parameters["radial_clearance"],
                    elements_circumferential=aux_input_parameters[
                        "elements_circumferential"
                    ],
                    elements_axial=aux_input_parameters["elements_axial"],
                    n_pad=aux_input_parameters["n_pad"],
                    pad_arc_length=aux_input_parameters["pad_arc_length"],
                    reference_temperature=aux_input_parameters["reference_temperature"],
                    speed=aux_input_parameters["speed"],
                    load_x_direction=aux_input_parameters["load_x_direction"],
                    load_y_direction=aux_input_parameters["load_y_direction"],
                    groove_factor=aux_input_parameters["groove_factor"],
                    lubricant=aux_input_parameters["lubricant"],
                    node=aux_input_parameters["node"],
                    sommerfeld_type=aux_input_parameters["sommerfeld_type"],
                    initial_guess=aux_input_parameters["initial_guess"],
                    method=aux_input_parameters["method"],
                    operating_type=aux_input_parameters["operating_type"],
                    injection_pressure=aux_input_parameters["injection_pressure"],
                    oil_flow=aux_input_parameters["oil_flow"],
                    show_coef=aux_input_parameters["show_coef"],
                    print_result=aux_input_parameters["print_result"],
                    print_progress=aux_input_parameters["print_progress"],
                    print_time=aux_input_parameters["print_time"],
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
                    self._enablePrint()  # restores prints
                    print(self.cylidrical_bearing[-1])

                if self.input_parameters["save_sample"] is not None:
                    self._samples_csv(
                        path=self.input_parameters["save_sample"],
                        inputs=aux_input_parameters,
                        coefs=out_coefs,
                    )

                progress_bar.update(1)

        t2 = time.time()
        self._enablePrint()  # restores prints

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

            new_n = self._find_min_samples(inputs=_aux_input_parameters)
            if new_n != self.input_parameters["num_inital_samples"]:
                self.input_parameters["num_inital_samples"] = new_n
                self.run()

        if self.input_parameters["print_time"]:
            print(f"Overall Time model THC: {(t2 - t1) / 60.0:.2f} min")

        return self.cylidrical_bearing
