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

    **Hole-Pattern Seal Parameters (downstream damping stage):**

    hole_pattern_parameters : dict
        Other parameters for hole-pattern seal.

        - length : float, pint.Quantity
            Axial length of hole-pattern seal (m).
        - radial_clearance : float, pint.Quantity
            Radial clearance at hole-pattern seal (m).
        - roughness : float
            Relative surface roughness (roughness/diameter).
        - cell_length : float, pint.Quantity
            Typical axial length of a hole/pocket (m).
        - cell_width : float, pint.Quantity
            Typical circumferential width of a hole/pocket (m).
        - cell_depth : float, pint.Quantity
            Depth of holes/pockets (m).
        - preswirl : float, optional
            Sutherland viscosity coefficient b. Calculated from gas_composition if not provided.
        - s_suther : float, optional
            Sutherland viscosity coefficient S. Calculated from gas_composition if not provided.
        - molar : float, optional
            Molecular mass (kg/kmol). Calculated from gas_composition if not provided.
        - nz : int, optional
            Number of axial discretization points. Default is 80.
        - itrmx : int, optional
            Maximum iterations for base state calculation. Default is 180.
        - tolerance : float, optional
            Convergence tolerance (fraction of pressure differential). Default is 0.0001.
        - first_step_size : float, optional
            Initial step tolerance. Default is 0.01.
        - rlx_factor : float, optional
            Relaxation factor for iterations. Default is 0.1.
        - whirl_ratio : float, optional
            Whirl frequency ratio (whirl_freq/shaft_freq). Default is 1.0.

    **Labyrinth Seal Parameters (upstream throttling stage):**

    labyrinth_parameters : dict
        Other parameters for labyrinth seal with the following keys:
        - n_teeth : int
            Number of labyrinth teeth (throttlings). Must be <= 30.
        - radial_clearance : float, pint.Quantity
            Nominal radial clearance at labyrinth teeth (m).
        - pitch : float, pint.Quantity
            Seal pitch (axial cavity length between teeth) (m).
        - tooth_height : float, pint.Quantity
            Height of labyrinth teeth (m).
        - tooth_width : float, pint.Quantity
            Axial thickness of tooth tips (m).
        - seal_type : str
            Location of labyrinth teeth.
            Options: 'rotor' (teeth on rotor), 'stator' (teeth on stator),
            'inter' (interlocking teeth).
        - preswirl : float
            Inlet tangential velocity ratio at labyrinth entrance.
            Positive for co-rotation, negative for counter-rotation.
        - r : float, optional
            Gas constant (J/(kg·K)). Calculated from gas_composition if not provided.
        - gamma : float, optional
            Ratio of specific heats (Cp/Cv). Calculated from gas_composition if not provided.
        - tz : list of float, optional
            [T1, T2] temperatures for viscosity interpolation (K).
        - muz : list of float, optional
            [mu1, mu2] dynamic viscosities for interpolation (Pa·s).
        - analz : str, optional
            Analysis type. 'FULL' for coefficients + leakage, 'LEAKAGE' for leakage only.
            Default is 'FULL'.
        - nprt : int, optional
            Print verbosity level (1=max, 5=min). Default is 1.
        - iopt1 : int, optional
            Use Jenny-Kanki tangential momentum parameters (0=no, 1=yes). Default is 0.

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
    ...   "b_suther": 1.458e-6,
    ...   "s_suther": 110.4,
    ...   "molar": 29.0,
    ...   "gamma": 1.4,
    ... }
    >>> laby_params = {
    ...   "radial_clearance": Q_(0.25, "mm"),
    ...   "n_teeth": 10,
    ...   "pitch": Q_(3, "mm"),
    ...   "tooth_height": Q_(3, "mm"),
    ...   "tooth_width": Q_(0.15, "mm"),
    ...   "seal_type": "inter",
    ...   "preswirl": 0.9,
    ...   "r": 287.05,
    ...   "gamma": 1.4,
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
    ...   hole_pattern_parameters=holep_params,
    ...   labyrinth_parameters=laby_params,
    ... )
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

        fig.add_trace(laby_data)
        fig.add_trace(hole_data)

        fig.update_xaxes(title_text=f"Axial Position ({length_units})")
        fig.update_yaxes(title_text=f"Pressure ({pressure_units})")

        fig.update_layout(
            title_text="Pressure Distribution - Hybrid Seal",
            title_font=dict(size=18),
            **kwargs,
        )

        return fig
