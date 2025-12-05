from ross import SealElement, HolePatternSeal, LabyrinthSeal
from ross.units import check_units
from prettytable import PrettyTable
from plotly import graph_objects as go
from plotly.subplots import make_subplots

__all__ = ["HybridSeal"]


class HybridSeal(SealElement):
    """Hybrid seal - Compressible flow model with rotordynamic coefficients.

    This class provides a model for hybrid seals that combine a hole-pattern seal
    (damping section) with a labyrinth seal (throttling section). The model iteratively
    determines the interface pressure between the two seal stages by matching their leakage
    rates, then combines their rotordynamic coefficients.

    **Theoretical Approach:**

    1. **Iterative Pressure Matching**:
       - Uses bisection method to find intermediate pressure between seal stages
       - Ensures mass conservation: leakage into hole-pattern = leakage from labyrinth
       - Convergence criterion based on relative leakage difference

    2. **Combined Rotordynamic Coefficients**:
       - Direct and cross-coupled stiffness (K), damping (C), and mass (M) coefficients
       - Series combination: forces from both seals are added
       - Frequency-dependent coefficients for each operating speed

    Parameters
    ----------

    **Common Parameters (both seal stages):**

    n : int
        Node in which the hybrid seal will be located.
    shaft_radius : float, pint.Quantity
        Radius of shaft (m).
    inlet_pressure : float
        Total inlet pressure at labyrinth entrance (Pa).
    outlet_pressure : float
        Final outlet pressure at hole-pattern exit (Pa).
    inlet_temperature : float
        Inlet temperature (K).
    frequency : float, list, pint.Quantity,
        Shaft rotational speed(s) (rad/s).
        Can be a single value or list of frequencies.
    gas_composition : dict, optional
        Gas composition as a dictionary {component: molar_fraction}.
        Example: {"Nitrogen": 0.79, "Oxygen": 0.21} for air.
    molar : float, pint.Quantity, optional
        Molecular mass (kg/kgmol). For Air: molar=28.97 kg/kgmol.
        Required if gas_composition is None. Default is None.
    gamma : float, optional
        Gas constant gamma (Cp/Cv). For Air: gamma=1.4.
        Required if gas_composition is None. Default is None.

    **Hole-Pattern Seal Parameters (downstream damping stage):**

    hole_pattern_parameters : dict
        Other parameters for hole-pattern seal with the following keys:
        - radial_clearance : float, pint.Quantity
            Seal clearance (m).
        - length : float, pint.Quantity
            Length of the seal (m).
        - roughness : float
            E / D (roughness / diameter) of the shaft.
        - cell_length : float, pint.Quantity
            Typical length of a cell in the axial direction (m).
        - cell_width : float, pint.Quantity
            Typical length of a cell in the azimuthal direction (m).
        - cell_depth : float, pint.Quantity
            Depth of a cell (m).
        - b_suther : float, optional
            b coefficient for the Sutherland viscosity model.
            Required if gas_composition is None. Default is None.
        - s_suther : float, optional
            s coefficient for the Sutherland viscosity model.
            Required if gas_composition is None. Default is None.
        - preswirl : float, optional
            Ratio of the circumferential velocity of the gas to the surface velocity of the shaft.
            Default is 0.0.
        - entr_coef : float, optional
            Entrance loss coefficient.
            Default is 0.1.
        - exit_coef : float, optional
            Exit loss coefficient.
            Default is 0.5.
        - whirl_ratio : float, optional
            Ratio of whirl frequency to rotational speed.
            Default is 1.0.
        - nz : int, optional
            Number of discretization points in the axial direction.
            Default is 80.
        - max_iterations : int, optional
            Maximum number of iterations for basic state calculation.
            Default is 180.
        - tolerance : float, optional
            Tolerance of the solution expressed as a percentage of the pressure differential
            across the seal. Default is 0.0001.
        - first_step_size : float, optional
            Initial step for the solution method. It should not be more than 0.01.
            Default is 0.01.
        - rlx_factor : float, optional
            Relaxation factor. Should be smaller than 0.1.
            Default is 0.1.

    **Labyrinth Seal Parameters (upstream throttling stage):**

    labyrinth_parameters : dict
        Other parameters for labyrinth seal with the following keys:
        - radial_clearance : float, pint.Quantity
            Nominal radial clearance (m).
        - n_teeth : int
            Number of teeth (throttlings). Needs to be <= 30.
        - pitch : float, pint.Quantity
            Seal pitch (length of land) or axial cavity length (m).
        - tooth_height : float, pint.Quantity
            Height of seal strip (m).
        - tooth_width : float, pint.Quantity
            Thickness of throttle (tip-width) (m), used in mass flow calculation.
        - seal_type : str
            Indicates where labyrinth teeth are located.
            Specify 'rotor' if teeth are on rotor only.
            Specify 'stator' if teeth are on stator only.
            Specify 'inter' for interlocking type labyrinths.
        - preswirl : float
            Inlet swirl velocity ratio. Positive values for swirl with shaft rotation
            and negative values for swirl against shaft rotations.
        - tz : list of float, optional
            Temperature at states: [T_state1, T_state2] (deg K).
            Required if gas_composition is None.
            Default is None.
        - muz : list of float, optional
            Dynamic viscosity at states: [mu_state1, mu_state2] (kg/(m·s)).
            Required if gas_composition is None.
            Default is None.
        - analz : str, optional
            Indicates what will be analysed.
            Specify "FULL" for rotordynamic calculation and leakage analysis.
            Specify "LEAKAGE" for leakage analysis only.
            Default is "FULL".
        - nprt : int, optional
            Number of parameters to be printed in the output: 1 maximum, 5 minimum.
            Default is 1.
        - iopt1 : int, optional
            Use or no use of tangential momentum parameters introduced by Jenny and Kanki.
            Specify value 0 to not use parameters.
            Specify value 1 to use parameters.
            Default is 0.

    **Hybrid Seal Control Parameters:**

    tolerance : float, optional
        Tolerance for pressure matching. Default is 1e-6.
    max_iterations : int, optional
        Maximum iterations for pressure matching. Default is 1e20.
    color : str, optional
        Color for element visualization. Default is "#787FF6".
    scale_factor : float, optional
        Scale factor for element drawing. Default is 0.75.
    kwargs : optional
        Additional keyword arguments passed to parent SealElement.

    Examples
    --------
    >>> from ross.seals.hybrid_seal import HybridSeal
    >>> from ross.units import Q_
    >>> gas_composition = {
    ...   "Nitrogen": 0.7812,
    ...   "Oxygen": 0.2096,
    ...   "Argon": 0.0092,
    ... }
    >>> holep_params = {
    ...   "radial_clearance": 0.0003,
    ...   "length": 0.04,
    ...   "roughness": 0.0001,
    ...   "cell_length": 0.003,
    ...   "cell_width": 0.003,
    ...   "cell_depth": 0.002,
    ...   "preswirl": 0.8,
    ...   "entr_coef": 0.5,
    ...   "exit_coef": 1.0,
    ... }
    >>> laby_params = {
    ...   "radial_clearance": Q_(0.25, "mm"),
    ...   "n_teeth": 10,
    ...   "pitch": Q_(3, "mm"),
    ...   "tooth_height": Q_(3, "mm"),
    ...   "tooth_width": Q_(0.15, "mm"),
    ...   "seal_type": "inter",
    ...   "preswirl": 0.9,
    ...   "tz": [300.0, 299.5],
    ...   "muz": [1.85e-05, 1.84e-05],
    ... }
    >>> hybrid = HybridSeal(
    ...   n=0,
    ...   shaft_radius=Q_(25, "mm"),
    ...   inlet_pressure=500000,
    ...   outlet_pressure=100000,
    ...   inlet_temperature=300.0,
    ...   frequency=Q_([2000, 3000, 5000], "RPM"),
    ...   gas_composition=gas_composition,
    ...   hole_pattern_parameters=holep_params,
    ...   labyrinth_parameters=laby_params,
    ... )
    >>> hybrid.seal_leakage  # doctest: +ELLIPSIS
    0.0348...
    """

    @check_units
    def __init__(
        self,
        n,
        shaft_radius,
        inlet_pressure,
        outlet_pressure,
        inlet_temperature,
        frequency,
        hole_pattern_parameters,
        labyrinth_parameters,
        gas_composition=None,
        molar=None,
        gamma=None,
        tolerance=1e-6,
        max_iterations=1e20,
        color="#787FF6",
        scale_factor=0.75,
        **kwargs,
    ):
        p_low = outlet_pressure
        p_high = inlet_pressure
        iteration = 0
        convergence_leakage = 1

        self.convergence_history = []
        self.pressure_history = []
        self.leakage_laby_history = []
        self.leakage_hole_history = []

        while convergence_leakage > tolerance and iteration < max_iterations:
            interface_pressure = (p_low + p_high) / 2

            holep = HolePatternSeal(
                n=n,
                inlet_pressure=inlet_pressure,
                outlet_pressure=interface_pressure,
                inlet_temperature=inlet_temperature,
                frequency=frequency,
                shaft_radius=shaft_radius,
                gas_composition=gas_composition,
                molar=molar,
                gamma=gamma,
                **hole_pattern_parameters,
            )

            laby = LabyrinthSeal(
                n=n,
                inlet_pressure=interface_pressure,
                outlet_pressure=outlet_pressure,
                inlet_temperature=inlet_temperature,
                frequency=frequency,
                shaft_radius=shaft_radius,
                gas_composition=gas_composition,
                molar=molar,
                gamma=gamma,
                **labyrinth_parameters,
            )

            hole_leakage = holep.seal_leakage[0]
            laby_leakage = laby.seal_leakage[0]

            convergence_leakage = abs(laby_leakage - hole_leakage) / hole_leakage

            if hole_leakage > laby_leakage:
                p_low = interface_pressure
            else:
                p_high = interface_pressure

            iteration += 1

            self.convergence_history.append(convergence_leakage)
            self.pressure_history.append(interface_pressure)
            self.leakage_laby_history.append(laby_leakage)
            self.leakage_hole_history.append(hole_leakage)

        self.laby = laby
        self.hole_pattern = holep
        self.interface_pressure = interface_pressure
        self.n_iterations = iteration

        coefficients_dict = {
            c: [l + h for l, h in zip(getattr(laby, c), getattr(holep, c))]
            for c in laby._get_coefficient_list()
        }

        super().__init__(
            n,
            frequency=frequency,
            seal_leakage=laby_leakage,
            color=color,
            scale_factor=scale_factor,
            **coefficients_dict,
            **kwargs,
        )

    def summary_results(self):
        """Print summary of hybrid seal analysis results.

        This method displays a comprehensive summary of the hybrid seal calculation,
        including convergence information, seal leakage, interface pressure.

        Returns
        -------
        table : PrettyTable object
            Formatted table of results.
        """
        # Convergence summary
        table = PrettyTable()
        table.field_names = ["Parameter", "Value"]
        table.align["Parameter"] = "l"
        table.align["Value"] = "r"

        table.add_row(["Number of Iterations", f"{self.n_iterations}"])
        table.add_row(["Final Convergence", f"{self.convergence_history[-1]:.6e}"])
        table.add_row(["Interface Pressure (Pa)", f"{self.interface_pressure:.2f}"])
        table.add_row(["Seal Leakage (kg/s)", f"{self.seal_leakage:.6e}"])

        return table

    def plot_convergence(self):
        """Plot convergence history.

        This method creates a unified figure with three subplots showing:
        convergence history, interface pressure evolution, and leakage comparison.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            Plotly figure object with combined subplots.
        """

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Convergence History",
                "Leakage Rate Convergence",
                "Interface Pressure Evolution",
            ),
            vertical_spacing=0.12,
            specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "scatter"}]],
        )

        iterations = list(range(1, len(self.convergence_history) + 1))

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=self.convergence_history,
                mode="lines+markers",
                name="Convergence Error",
                line=dict(width=2),
                marker=dict(size=6),
                legendgroup="group1",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=self.leakage_hole_history,
                mode="lines+markers",
                name="Hole-Pattern Seal",
                line=dict(width=2),
                marker=dict(size=6),
                legendgroup="group2",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=self.leakage_laby_history,
                mode="lines+markers",
                name="Labyrinth Seal",
                line=dict(width=2),
                marker=dict(size=6),
                legendgroup="group2",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=self.pressure_history,
                mode="lines+markers",
                name="Interface Pressure",
                line=dict(width=2),
                marker=dict(size=6),
                legendgroup="group3",
            ),
            row=3,
            col=1,
        )

        fig.update_xaxes(title_text="Iteration", row=3, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)

        fig.update_yaxes(
            title_text="Relative Leakage Error",
            type="log",
            exponentformat="e",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="Leakage Rate (kg/s)",
            exponentformat="e",
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Pressure (Pa)",
            exponentformat="e",
            row=3,
            col=1,
        )

        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Hybrid Seal Analysis Results",
            title_x=0.5,
            title_font=dict(size=18),
        )

        return fig

    def plot_pressure_distribution(
        self, pressure_units="MPa", length_units="m", fig=None, **kwargs
    ):
        """Plot pressure distribution for the hybrid seal.

        Parameters
        ----------
        pressure_units : str, optional
            Units for the pressure.
            Default is "MPa".
        length_units : str, optional
            Units for the length.
            Default is "m".
        fig : go.Figure, optional
            Plotly figure object with the plot.
            Default is None.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            Plotly figure object.
        """

        if fig is None:
            fig = go.Figure()

        hole_data = self.hole_pattern.plot_pressure_distribution(
            pressure_units=pressure_units, length_units=length_units
        ).data[0]
        laby_data = self.laby.plot_pressure_distribution(
            pressure_units=pressure_units, length_units=length_units
        ).data[0]

        laby_data["x"] += hole_data["x"][-1]

        fig.add_trace(hole_data)
        fig.add_trace(laby_data)

        fig.update_xaxes(title_text=f"Axial Position ({length_units})")
        fig.update_yaxes(title_text=f"Pressure ({pressure_units})")

        fig.update_layout(
            title_text="Pressure Distribution - Hybrid Seal",
            title_font=dict(size=18),
            **kwargs,
        )

        return fig
