"""Surrogate model for tilting-pad bearings solved with ``ross.TiltingPad``."""

from ross.bearings.tilting_pad import TiltingPad
from ross.units import check_units
from ross.surrogates.surrogate_core import CoefficientSurrogate


class TiltingPadSurrogate(CoefficientSurrogate):
    """Surrogate of the coefficients of a tilting-pad journal bearing.

    The training samples are computed with :class:`ross.bearings.tilting_pad.TiltingPad`.
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
    journal_diameter, radial_clearance, pad_thickness, pad_axial_length,
    oil_supply_temperature, oil_flow_v, ...
        The remaining parameters are those of
        :class:`ross.bearings.tilting_pad.TiltingPad` and of
        :class:`ross.bearings.fluid_film_bearing.FluidFilmBearing`, with the same
        names, meaning and SI units. ``eccentricity``, ``attitude_angle`` and
        ``num_processes`` are not available.

    Any of the bearing parameters may be given as [min, max] when it is listed in
    ``input_variable``.

    See Also
    --------
    ross.bearings.tilting_pad.TiltingPad : Solver that generates the samples.

    Examples
    --------
    >>> import ross.surrogates as sg
    >>> from ross.units import Q_
    >>> surrogate = sg.TiltingPadSurrogate(
    ...     metamodel="gaussian",
    ...     num_inital_samples=6,
    ...     input_variable=["radial_clearance", "oil_supply_temperature"],
    ...     n=1,
    ...     frequency=Q_([3000, 3500], "RPM"),
    ...     equilibrium_type="match_load",
    ...     thermal_type="adiabatic",
    ...     journal_diameter=101.6e-3,
    ...     radial_clearance=[70e-6, 80e-6],
    ...     pad_thickness=12.7e-3,
    ...     pivot_angle=Q_([18, 90, 162, 234, 306], "deg"),
    ...     pad_arc=Q_([60] * 5, "deg"),
    ...     pad_axial_length=[50.8e-3] * 5,
    ...     preload=[0.5] * 5,
    ...     offset=[0.5] * 5,
    ...     lubricant="ISOVG32",
    ...     oil_supply_temperature=[303.15, 323.15],
    ...     oil_flow_v=Q_(10, "l/min"),
    ...     fxs_load=8.8405e02,
    ...     fys_load=-2.6704e03,
    ...     total_ex_film=20,
    ...     total_ez_film=10,
    ...     total_ey_pad=10,
    ...     total_ey_film=10,
    ... )  # doctest: +SKIP
    >>> bearing = surrogate.predict(
    ...     frequency=Q_(3200, "RPM"),
    ...     radial_clearance=75e-6,
    ...     oil_supply_temperature=313.15,
    ... )  # doctest: +SKIP
    """

    _bearing_class = TiltingPad

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
        except (AttributeError, TypeError, ValueError):
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

        if input_variable is not None:
            self.run()
