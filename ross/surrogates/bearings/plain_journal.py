"""Surrogate model for plain journal bearings solved with ``ross.PlainJournal``."""

from ross.bearings.plain_journal import PlainJournal
from ross.units import check_units
from ross.surrogates.surrogate_core import CoefficientSurrogate


class PlainJournalSurrogate(CoefficientSurrogate):
    """Surrogate of the coefficients of a plain journal bearing.

    The training samples are computed with :class:`ross.bearings.plain_journal.PlainJournal`.
    ``predict()`` returns a ``BearingElement`` with the predicted stiffness and damping
    coefficients.

    Parameters
    ----------
    metamodel : str, optional
        Metamodel fitted to each coefficient: "KGR" (local Kriging, :class:`~ross.surrogates.kriging.Kriging`) or "gaussian"
        (scikit-learn Gaussian process). Default is "KGR".
    save_sample : str, optional
        Name of a csv file where inputs and coefficients of each sample are saved.
        If it already exists, its samples are reused and only the missing ones are
        computed.
    input_variable : list of str
        Variables varied to build the metamodel, at most two. Each one is given as a
        list [min, max].
    distribution : str, optional
        "LHS" (Latin Hypercube sampling) or "FullFactorial" (regular grid). Default is
        "LHS".
    corr : str, optional
        Correlation function of the Kriging model: "pow_exp", "abs_exp", "squar_exp",
        "matern52" or "matern32". Default is "squar_exp".
    num_inital_samples : int, optional
        Number of samples used to train the metamodel. Default is 100.
    step_num_inital_samples : int, optional
        Not used. Kept for compatibility with ``rossml``.
    coef_R2 : float, optional
        Not used. Kept for compatibility with ``rossml``.
    n : int, optional
        Node where the bearing is located. Default is 0.
    frequency : list of float, pint.Quantity
        Range [min, max] of rotor speeds, passed to the solver as ``speed``. Default
        unit is rad/s.
    print_result : bool, optional
        If True, print each bearing after it is computed. Default is False.
    print_time : bool, optional
        If True, print the total time spent computing the samples. Default is False.

    Other Parameters
    ----------------
    pad_axial_length : float, pint.Quantity
        Axial length of each land. Default unit is m.
    journal_diameter : float, pint.Quantity
        Journal diameter. Default unit is m.
    radial_clearance : float, pint.Quantity
        Radial clearance. Default unit is m.
    n_pads : int
        Number of lands between the axial grooves.
    pad_arc : float, pint.Quantity
        Arc of each land. Default unit is rad.
    preload : float, optional
        Land preload factor. Default is 0.
    oil_supply_temperature : float, pint.Quantity
        Lubricant supply temperature. Default unit is K.
    fxs_load, fys_load : float, pint.Quantity, optional
        Static load in x and y. Default unit is N.
    lubricant : str or dict
        Key of :data:`ross.bearings.lubricants.lubricants_dict` or a dict with the same
        fields.
    initial_position : tuple of float, optional
        Initial journal position as fractions of the radial clearance. Default is
        (0.1, -0.1).
    oil_supply_pressure : float, pint.Quantity, optional
        Oil supply pressure. Default unit is Pa.
    oil_flow_v : float, pint.Quantity, optional
        Supplied oil flow. Default unit is m**3/s.
    pad_thickness : float, pint.Quantity, optional
        Radial thickness of the pad. Default unit is m.

    Any of the bearing parameters may be given as [min, max] when it is listed in
    ``input_variable``.

    See Also
    --------
    ross.bearings.plain_journal.PlainJournal : Solver that generates the samples.

    Examples
    --------
    >>> import ross.surrogates as sg
    >>> from ross.units import Q_
    >>> surrogate = sg.PlainJournalSurrogate(
    ...     metamodel="gaussian",
    ...     num_inital_samples=8,
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
    >>> bearing = surrogate.predict(
    ...     frequency=Q_(1000, "RPM"),
    ...     radial_clearance=1.9e-4,
    ...     oil_supply_temperature=Q_(50, "degC"),
    ... )  # doctest: +SKIP
    """

    _bearing_class = PlainJournal

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

        if input_variable is not None:
            self.run()
