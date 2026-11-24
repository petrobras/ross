"""Surrogate model for tilting-pad thrust bearings solved with ``ross.ThrustPad``."""

from ross.bearings.thrust_pad import ThrustPad
from ross.units import check_units
from ross.surrogates.surrogate_core import CoefficientSurrogate


class ThrustPadSurrogate(CoefficientSurrogate):
    """Surrogate of the axial coefficients of a tilting-pad thrust bearing.

    The training samples are computed with :class:`ross.bearings.thrust_pad.ThrustPad`.
    ``predict()`` returns a ``BearingElement`` with the predicted ``kzz`` and ``czz``.

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
    pad_inner_radius, pad_outer_radius, pad_pivot_radius : float, pint.Quantity
        Inner, outer and pivot radius of the pads. Default unit is m.
    pad_arc : float, pint.Quantity
        Arc of each pad. Default unit is rad.
    pivot_angle : float, pint.Quantity
        Angular position of the pivot inside the pad. Default unit is rad.
    oil_supply_temperature : float, pint.Quantity
        Lubricant supply temperature. Default unit is K.
    lubricant : str or dict
        Key of :data:`ross.bearings.lubricants.lubricants_dict` or a dict with the same
        fields.
    n_pads : int
        Number of pads.
    n_theta, n_radial : int, optional
        Number of volumes in the circumferential and radial directions. Default is 10.
    equilibrium_position_mode : str, optional
        "calculate" (default) or "imposed".
    radial_inclination_angle, circumferential_inclination_angle : float, pint.Quantity
        Initial inclination of the pad. Default unit is rad.
    initial_film_thickness : float, pint.Quantity
        Initial film thickness at the pivot. Default unit is m.
    tolerance_force_moment, residual_force_moment : float, optional
        Convergence tolerances of the equilibrium search. See
        :class:`ross.bearings.thrust_pad.ThrustPad`.
    model_type : str, optional
        Model of the film. Default is "thermo_hydro_dynamic".
    axial_load : float, pint.Quantity
        Axial load. Default unit is N.

    Any of the bearing parameters may be given as [min, max] when it is listed in
    ``input_variable``.

    See Also
    --------
    ross.bearings.thrust_pad.ThrustPad : Solver that generates the samples.

    Examples
    --------
    >>> import ross.surrogates as sg
    >>> from ross.units import Q_
    >>> surrogate = sg.ThrustPadSurrogate(
    ...     metamodel="gaussian",
    ...     num_inital_samples=6,
    ...     input_variable=["axial_load", "oil_supply_temperature"],
    ...     n=1,
    ...     frequency=Q_([80, 100], "RPM"),
    ...     pad_inner_radius=Q_(1150, "mm"),
    ...     pad_outer_radius=Q_(1725, "mm"),
    ...     pad_pivot_radius=Q_(1442.5, "mm"),
    ...     pad_arc=Q_(26, "deg"),
    ...     pivot_angle=Q_(15, "deg"),
    ...     lubricant="ISOVG68",
    ...     n_pads=12,
    ...     axial_load=[10e6, 16e6],
    ...     oil_supply_temperature=[308.15, 323.15],
    ...     radial_inclination_angle=Q_(-2.75e-4, "rad"),
    ...     circumferential_inclination_angle=Q_(-1.7e-5, "rad"),
    ...     initial_film_thickness=Q_(0.2, "mm"),
    ... )  # doctest: +SKIP
    >>> bearing = surrogate.predict(
    ...     frequency=Q_(90, "RPM"),
    ...     axial_load=13e6,
    ...     oil_supply_temperature=313.15,
    ... )  # doctest: +SKIP
    """

    coef_names = ("kzz", "czz")

    _bearing_class = ThrustPad

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

        if input_variable is not None:
            self.run()
