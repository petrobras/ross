from ross import SealElement, HolePatternSeal, LabyrinthSeal
from ross.units import check_units
from prettytable import PrettyTable
from plotly import graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

__all__ = ["HybridSeal"]


class HybridSeal(SealElement):
    """Hybrid seal - Compressible flow model with rotordynamic coefficients.

    This class provides a model for hybrid seals that combine a labyrinth seal
    (throttling section) with a hole-pattern seal (damping section).
    The model iteratively determines the interface pressure between the two seal stages
    by matching their leakage rates, then combines their rotordynamic coefficients.

    **Theoretical Approach:**

    1. **Iterative Pressure Matching**:
       - Uses bisection method to find intermediate pressure between seal stages
       - Ensures mass conservation: leakage from labyrinth = leakage into hole-pattern
       - Convergence criterion based on relative leakage difference

    2. **Combined Rotordynamic Coefficients**:
       - Direct and cross-coupled stiffness (K), damping (C), and mass (M) coefficients
       - Series combination: forces from both seals are added
       - Frequency-dependent coefficients for each operating speed

    Parameters
    ----------
    n : int
        Node in which the hybrid seal will be located.

    **Common Parameters (both seal stages):**

    inlet_pressure : float
        Total inlet pressure at labyrinth entrance (Pa).
    outlet_pressure : float
        Final outlet pressure at hole-pattern exit (Pa).
    inlet_temperature : float
        Inlet temperature (K).
    frequency : float, pint.Quantity, list
        Shaft rotational speed(s) (rad/s).
        Can be a single value or list of frequencies.
    gas_composition : dict
        Gas composition as a dictionary {component: molar_fraction}.
        Example: {"Nitrogen": 0.79, "Oxygen": 0.21} for air.

    **Labyrinth Seal Parameters (upstream throttling stage):**

    n_teeth : int
        Number of labyrinth teeth (throttlings). Must be <= 30.
    shaft_radius : float, pint.Quantity
        Radius of shaft at labyrinth section (m).
    radial_clearance : float, pint.Quantity
        Nominal radial clearance at labyrinth teeth (m).
    pitch : float, pint.Quantity
        Seal pitch (axial cavity length between teeth) (m).
    tooth_height : float, pint.Quantity
        Height of labyrinth teeth (m).
    tooth_width : float, pint.Quantity
        Axial thickness of tooth tips (m).
    seal_type : str
        Location of labyrinth teeth.
        Options: 'rotor' (teeth on rotor), 'stator' (teeth on stator),
        'inter' (interlocking teeth).
    pre_swirl_ratio : float
        Inlet tangential velocity ratio at labyrinth entrance.
        Positive for co-rotation, negative for counter-rotation.
    r : float, optional
        Gas constant (J/(kg·K)). Calculated from gas_composition if not provided.
    gamma : float, optional
        Ratio of specific heats (Cp/Cv). Calculated from gas_composition if not provided.
    tz : list of float, optional
        [T1, T2] temperatures for viscosity interpolation (K).
    muz : list of float, optional
        [mu1, mu2] dynamic viscosities for interpolation (Pa·s).
    analz : str, optional
        Analysis type. 'FULL' for coefficients + leakage, 'LEAKAGE' for leakage only.
        Default is 'FULL'.
    nprt : int, optional
        Print verbosity level (1=max, 5=min). Default is 1.
    iopt1 : int, optional
        Use Jenny-Kanki tangential momentum parameters (0=no, 1=yes). Default is 0.

    **Hole-Pattern Seal Parameters (downstream damping stage):**

    length : float, pint.Quantity
        Axial length of hole-pattern seal (m).
    radius : float, pint.Quantity
        Radius of shaft at hole-pattern section (m).
    clearance : float, pint.Quantity
        Radial clearance at hole-pattern seal (m).
    roughness : float
        Relative surface roughness (roughness/diameter).
    cell_length : float, pint.Quantity
        Typical axial length of a hole/pocket (m).
    cell_width : float, pint.Quantity
        Typical circumferential width of a hole/pocket (m).
    cell_depth : float, pint.Quantity
        Depth of holes/pockets (m).
    preswirl : float, optional
        Inlet tangential velocity ratio for hole-pattern section.
        Different from pre_swirl_ratio as flow has been throttled through labyrinth.
    entr_coef : float, optional
        Entrance loss coefficient. Default is 0.1.
    exit_coef : float, optional
        Exit loss coefficient. Default is 0.5.
    b_suther : float, optional
        Sutherland viscosity coefficient b. Calculated from gas_composition if not provided.
    s_suther : float, optional
        Sutherland viscosity coefficient S. Calculated from gas_composition if not provided.
    molar : float, optional
        Molecular mass (kg/kmol). Calculated from gas_composition if not provided.
    nz : int, optional
        Number of axial discretization points. Default is 80.
    itrmx : int, optional
        Maximum iterations for base state calculation. Default is 180.
    stopcriterion : float, optional
        Convergence tolerance (fraction of pressure differential). Default is 0.0001.
    toler : float, optional
        Initial step tolerance. Default is 0.01.
    rlx : float, optional
        Relaxation factor for iterations. Default is 0.1.
    whirl_ratio : float, optional
        Whirl frequency ratio (whirl_freq/shaft_freq). Default is 1.0.

    **Hybrid Seal Control Parameters:**

    tolerance : float, optional
        Convergence tolerance for leakage matching between stages. Default is 1e-6.
    max_iterations : int, optional
        Maximum iterations for pressure matching. Default is 1e20.

    **Display Parameters:**

    print_results : bool, optional
        If True, print convergence information. Default is False.
    color : str, optional
        Color for element visualization. Default is "#787FF6".
    scale_factor : float, optional
        Scale factor for element drawing. Default is 0.75.

    **kwargs : dict, optional
        Additional keyword arguments passed to parent SealElement.

    Examples
    --------
    >>> from ross.seals.hybrid_seal import HybridSeal
    >>> from ross.units import Q_
    >>> hybrid = HybridSeal(
    ...     n=0,
    ...     # Common parameters
    ...     inlet_pressure=500000.0,
    ...     outlet_pressure=100000.0,
    ...     inlet_temperature=300.0,
    ...     frequency=Q_([6000, 8000], "RPM"),
    ...     gas_composition={"Nitrogen": 0.79, "Oxygen": 0.21},
    ...     # Labyrinth seal parameters
    ...     n_teeth=12,
    ...     shaft_radius=Q_(75, "mm"),
    ...     radial_clearance=Q_(0.25, "mm"),
    ...     pitch=Q_(3.0, "mm"),
    ...     tooth_height=Q_(3.0, "mm"),
    ...     tooth_width=Q_(0.15, "mm"),
    ...     seal_type="inter",
    ...     pre_swirl_ratio=0.95,
    ...     # Hole-pattern seal parameters
    ...     length=0.050,
    ...     radius=0.075,
    ...     clearance=0.0003,
    ...     roughness=0.0001,
    ...     cell_length=0.003,
    ...     cell_width=0.003,
    ...     cell_depth=0.002,
    ...     preswirl=0.8,
    ... )
    """

    @check_units
    def __init__(
        self,
        n=None,
        # Common parameters (both seals)
        inlet_pressure=None,
        outlet_pressure=None,
        inlet_temperature=None,
        frequency=None,
        gas_composition=None,
        # Labyrinth seal parameters
        n_teeth=None,
        shaft_radius=None,
        radial_clearance=None,
        pitch=None,
        tooth_height=None,
        tooth_width=None,
        seal_type=None,
        pre_swirl_ratio=None,
        r=None,
        gamma=None,
        tz=None,
        muz=None,
        analz="FULL",
        nprt=1,
        iopt1=0,
        # Hole-pattern seal parameters
        length=None,
        radius=None,
        clearance=None,
        roughness=None,
        cell_length=None,
        cell_width=None,
        cell_depth=None,
        preswirl=None,
        entr_coef=None,
        exit_coef=None,
        b_suther=None,
        s_suther=None,
        molar=None,
        nz=80,
        itrmx=180,
        stopcriterion=0.0001,
        toler=0.01,
        rlx=0.1,
        whirl_ratio=1.0,
        # Hybrid seal control parameters
        tolerance=1e-6,
        max_iterations=1e20,
        # Display parameters
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

            laby = LabyrinthSeal(
                n=n,
                inlet_pressure=inlet_pressure,
                outlet_pressure=interface_pressure,
                inlet_temperature=inlet_temperature,
                pre_swirl_ratio=pre_swirl_ratio,
                frequency=frequency,
                n_teeth=n_teeth,
                shaft_radius=shaft_radius,
                radial_clearance=radial_clearance,
                pitch=pitch,
                tooth_height=tooth_height,
                tooth_width=tooth_width,
                seal_type=seal_type,
                gas_composition=gas_composition,
                r=r,
                gamma=gamma,
                tz=tz,
                muz=muz,
                analz=analz,
                nprt=nprt,
                iopt1=iopt1,
            )

            hole = HolePatternSeal(
                n=n,
                inlet_pressure=interface_pressure,
                outlet_pressure=outlet_pressure,
                inlet_temperature=inlet_temperature,
                frequency=frequency,
                length=length,
                radius=radius,
                clearance=clearance,
                roughness=roughness,
                cell_length=cell_length,
                cell_width=cell_width,
                cell_depth=cell_depth,
                gas_composition=gas_composition,
                b_suther=b_suther,
                s_suther=s_suther,
                molar=molar,
                gamma=gamma,
                preswirl=preswirl,
                entr_coef=entr_coef,
                exit_coef=exit_coef,
                nz=nz,
                itrmx=itrmx,
                stopcriterion=stopcriterion,
                toler=toler,
                rlx=rlx,
                whirl_ratio=whirl_ratio,
            )

            convergence_leakage = (
                abs(hole.seal_leakage - laby.seal_leakage) / laby.seal_leakage
            )

            if laby.seal_leakage > hole.seal_leakage:
                p_low = interface_pressure
            else:
                p_high = interface_pressure

            iteration += 1

            self.convergence_history.append(convergence_leakage)
            self.pressure_history.append(interface_pressure)
            self.leakage_laby_history.append(laby.seal_leakage)
            self.leakage_hole_history.append(hole.seal_leakage)

        self.interface_pressure = interface_pressure
        self.n_iterations = iteration

        coefficients_dict = {
            c: [l + h for l, h in zip(getattr(laby, c), getattr(hole, c))]
            for c in laby._get_coefficient_list()
        }

        super().__init__(
            n,
            frequency=frequency,
            seal_leakage=laby.seal_leakage,
            color=color,
            scale_factor=scale_factor,
            **coefficients_dict,
            **kwargs,
        )

    def summary_results(self):
        """
        Print summary of hybrid seal analysis results.

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
        """
        Plot convergence history.

        This method creates a unified figure with three subplots showing:
        convergence history, interface pressure evolution, and leakage comparison.

        Parameters
        ----------

        Returns
        -------
        fig : go.Figure
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
                line=dict(color="royalblue", width=2),
                marker=dict(size=6),
                legendgroup="group1",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=self.leakage_laby_history,
                mode="lines+markers",
                name="Labyrinth Seal",
                line=dict(color="crimson", width=2),
                marker=dict(size=6),
                legendgroup="group2",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=self.leakage_hole_history,
                mode="lines+markers",
                name="Hole-Pattern Seal",
                line=dict(color="darkorange", width=2),
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
                line=dict(color="darkgreen", width=2),
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
            template="plotly_white",
            hovermode="x unified",
            title_text="Hybrid Seal Analysis Results",
            title_x=0.5,
            title_font=dict(size=18),
        )

        return fig
