import warnings
from abc import ABC, abstractmethod

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from prettytable import PrettyTable
from scipy.interpolate import griddata

from ross.plotly_theme import tableau_colors

__all__ = [
    "BearingResults",
    "ThrustPadResults",
    "SqueezeFilmDamperResults",
    "FluidFilmBearingResults",
]


class BearingResults(ABC):
    """Abstract base class for fluid film bearing post-processing results.

    Each bearing class (FluidFilmBearing and its subclasses, ThrustPad,
    SqueezeFilmDamper) creates a ``_results`` attribute of the corresponding
    subclass after the solver runs. The bearing then delegates every ``plot_*`` and ``show_*`` call to that
    object via ``__getattr__``, so the end user never needs to access ``_results``
    directly.

    Subclasses implement bearing-specific visualization while sharing
    common infrastructure (execution-time display and the ``plot_results``
    orchestrator).

    Parameters
    ----------
    frequency : array_like
        Operating frequencies in rad/s (one value per solved point).
    pressure_fields : list of ndarray
        Pressure field arrays, one per frequency.
    temperature_fields : list of ndarray
        Temperature field arrays, one per frequency.
    initial_time : float, optional
        Epoch timestamp recorded at the start of the solver run.
    final_time : float, optional
        Epoch timestamp recorded at the end of the solver run.

    Examples
    --------
    >>> from ross.bearings.tilting_pad import tilting_pad_adiabatic_example
    >>> bearing = tilting_pad_adiabatic_example()
    >>> type(bearing).__name__
    'TiltingPad'
    """

    def __init__(
        self,
        frequency,
        pressure_fields,
        temperature_fields,
        initial_time=None,
        final_time=None,
    ):
        self.frequency = np.atleast_1d(frequency)
        self.pressure_fields = pressure_fields
        self.temperature_fields = temperature_fields
        self.initial_time = initial_time
        self.final_time = final_time

    def show_execution_time(self):
        """Display the total solver execution time.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Prints the elapsed time in seconds to the console.
        """
        if self.initial_time is not None and self.final_time is not None:
            total_time = self.final_time - self.initial_time
            print(f"Execution time: {total_time:.6f} seconds")
        else:
            print("Simulation hasn't been executed yet.")

    def plot_results(self, show_plots=False, freq_index=0):
        """Generate and return all standard bearing result plots.

        Calls the four abstract ``plot_*`` methods and collects their figures
        into a standardized dictionary.  Subclasses may override this method
        to add bearing-specific figures while calling ``super().plot_results()``.

        Parameters
        ----------
        show_plots : bool, optional
            When *True* each figure is displayed immediately via
            ``fig.show()``. Default is False.
        freq_index : int, optional
            Index into the frequency array selecting which solved point to
            visualize.  Default is 0 (first frequency).

        Returns
        -------
        figures : dict
            Dictionary with keys ``"pressure_2d"``, ``"pressure_3d"``,
            ``"temperature_2d"``, and ``"film_temperature_3d"``.  Each value
            is a ``plotly.graph_objects.Figure``.
        """
        figures = {
            "pressure_2d": self.plot_pressure_2d(freq_index=freq_index),
            "pressure_3d": self.plot_pressure_3d(freq_index=freq_index),
            "temperature_2d": self.plot_temperature_2d(freq_index=freq_index),
            "film_temperature_3d": self.plot_film_temperature_3d(freq_index=freq_index),
        }

        if show_plots:
            for fig in figures.values():
                try:
                    fig.show()
                except Exception as e:
                    print(f"Warning: Could not display plot automatically. Error: {e}")

        return figures

    @abstractmethod
    def show_results(self):
        """Print a formatted summary of the bearing analysis results.

        Returns
        -------
        None
        """

    @abstractmethod
    def show_coefficients_comparison(self):
        """Print a table comparing dynamic coefficients across frequencies.

        Returns
        -------
        None
        """

    @abstractmethod
    def plot_pressure_3d(self, freq_index=0, fig=None, **kwargs):
        """Return a 3-D surface plot of the pressure field.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """

    @abstractmethod
    def plot_pressure_2d(self, freq_index=0, fig=None, **kwargs):
        """Return a 2-D contour plot of the pressure field.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """

    @abstractmethod
    def plot_film_temperature_3d(self, freq_index=0, fig=None, **kwargs):
        """Return a 3-D surface plot of the oil film temperature field.

        The plotted quantity is the temperature of the lubricant film.
        Solid (pad) temperatures are not shown here.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """

    def plot_temperature_3d(self, *args, **kwargs):
        """Deprecated alias for :meth:`plot_film_temperature_3d`.

        .. deprecated:: 2.4.0
            ``plot_temperature_3d`` is deprecated and will be removed in a
            future version.  Use ``plot_film_temperature_3d`` instead, which
            states explicitly that the plotted field is the oil film
            temperature.

        Returns
        -------
        fig : go.Figure
        """
        warnings.warn(
            "plot_temperature_3d is deprecated and will be removed in a future "
            "version. Use plot_film_temperature_3d instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.plot_film_temperature_3d(*args, **kwargs)

    @abstractmethod
    def plot_temperature_2d(self, freq_index=0, fig=None, **kwargs):
        """Return a 2-D contour plot of the temperature field.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """


class ThrustPadResults(BearingResults):
    """Post-processing results for a ThrustPad bearing.

    Parameters
    ----------
    frequency : array_like
        Operating frequencies in rad/s.
    pressure_fields : list of ndarray, shape (n_radial + 2, n_theta + 2)
        Pressure fields, one per frequency (Pa).
    temperature_fields : list of ndarray, shape (n_radial + 2, n_theta + 2)
        Temperature fields, one per frequency (°C).
    max_thicknesses : list of float
        Maximum film thickness per frequency (m).
    min_thicknesses : list of float
        Minimum film thickness per frequency (m).
    pivot_film_thicknesses : list of float
        Film thickness at the pivot per frequency (m).
    equilibrium_position_mode : str
        ``"imposed"`` or ``"calculate"``.
    axial_load : float or ndarray
        Applied axial load (N). Scalar when ``equilibrium_position_mode``
        is ``"calculate"``; array when ``"imposed"``.
    kzz : ndarray
        Axial stiffness coefficient (N/m), one value per frequency.
    czz : ndarray
        Axial damping coefficient (N·s/m), one value per frequency.
    n_radial : int
        Number of radial mesh elements.
    n_theta : int
        Number of circumferential mesh elements.
    pad_outer_radius : float
        Pad outer radius (m).
    pad_inner_radius : float
        Pad inner radius (m).
    d_radius : float
        Radial mesh step size (non-dimensional).
    d_theta : float
        Angular mesh step size (non-dimensional).
    pad_arc_length : float
        Pad arc length (rad).
    optimization_history : dict
        Mapping ``{freq_index: [residuals]}``.
    initial_time : float, optional
        Solver start epoch timestamp.
    final_time : float, optional
        Solver end epoch timestamp.
    """

    def __init__(
        self,
        frequency,
        pressure_fields,
        temperature_fields,
        max_thicknesses,
        min_thicknesses,
        pivot_film_thicknesses,
        equilibrium_position_mode,
        axial_load,
        kzz,
        czz,
        n_radial,
        n_theta,
        pad_outer_radius,
        pad_inner_radius,
        d_radius,
        d_theta,
        pad_arc_length,
        optimization_history,
        initial_time=None,
        final_time=None,
    ):
        super().__init__(
            frequency=frequency,
            pressure_fields=pressure_fields,
            temperature_fields=temperature_fields,
            initial_time=initial_time,
            final_time=final_time,
        )
        self.max_thicknesses = max_thicknesses
        self.min_thicknesses = min_thicknesses
        self.pivot_film_thicknesses = pivot_film_thicknesses
        self.equilibrium_position_mode = equilibrium_position_mode
        self.axial_load = axial_load
        self.kzz = np.atleast_1d(kzz)
        self.czz = np.atleast_1d(czz)
        self.n_radial = n_radial
        self.n_theta = n_theta
        self.pad_outer_radius = pad_outer_radius
        self.pad_inner_radius = pad_inner_radius
        self.d_radius = d_radius
        self.d_theta = d_theta
        self.pad_arc_length = pad_arc_length
        self.optimization_history = optimization_history

    def _build_cartesian_coords(self):
        """Compute Cartesian coordinate grids from the polar mesh geometry.

        Returns
        -------
        x_coords : ndarray, shape (n_radial + 2, n_theta + 2)
        y_coords : ndarray, shape (n_radial + 2, n_theta + 2)
        """
        radial_coords = np.zeros(self.n_radial + 2)
        angular_coords = np.zeros(self.n_theta + 2)
        x_coords = np.zeros((self.n_radial + 2, self.n_theta + 2))
        y_coords = np.zeros((self.n_radial + 2, self.n_theta + 2))

        radial_coords[0] = self.pad_outer_radius
        radial_coords[-1] = self.pad_inner_radius
        radial_coords[1 : self.n_radial + 1] = np.arange(
            self.pad_outer_radius - 0.5 * self.d_radius * self.pad_inner_radius,
            self.pad_inner_radius,
            -(self.d_radius * self.pad_inner_radius),
        )

        angular_coords[0] = np.pi / 2 + self.pad_arc_length / 2
        angular_coords[-1] = np.pi / 2 - self.pad_arc_length / 2
        angular_coords[1 : self.n_theta + 1] = np.arange(
            np.pi / 2
            + self.pad_arc_length / 2
            - (0.5 * self.d_theta * self.pad_arc_length),
            np.pi / 2 - self.pad_arc_length / 2,
            -self.d_theta * self.pad_arc_length,
        )

        for i in range(self.n_radial + 2):
            for j in range(self.n_theta + 2):
                x_coords[i, j] = radial_coords[i] * np.cos(angular_coords[j])
                y_coords[i, j] = radial_coords[i] * np.sin(angular_coords[j])

        return x_coords, y_coords

    def _build_interp_grid(self, x_coords, y_coords, z_data, resolution=800):
        """Interpolate field data onto a regular Cartesian grid.

        Parameters
        ----------
        x_coords : ndarray
        y_coords : ndarray
        z_data : ndarray
            Field values on the polar mesh.
        resolution : int, optional
            Grid resolution for interpolation. Default is 800.

        Returns
        -------
        x_grid : ndarray
        y_grid : ndarray
        z_interp : ndarray
        """
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_interp = np.linspace(x_min, x_max, resolution)
        y_interp = np.linspace(y_min, y_max, resolution)
        x_grid, y_grid = np.meshgrid(x_interp, y_interp)

        z_interp = griddata(
            (x_coords.flatten(), y_coords.flatten()),
            z_data.flatten(),
            (x_grid, y_grid),
            method="cubic",
        )
        return x_grid, y_grid, z_interp

    def show_results(self):
        """Print a formatted summary of thrust pad bearing results.

        Iterates over all solved frequencies and prints a PrettyTable with
        operating conditions, field extrema, film thicknesses, axial load,
        and dynamic coefficients.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.frequency.size == 1:
            self._print_single_frequency_results(0)
        else:
            for i in range(self.frequency.size):
                self._print_single_frequency_results(i)

    def _print_single_frequency_results(self, freq_index):
        """Print results table for one frequency index.

        Parameters
        ----------
        freq_index : int
            Index into the frequency array.
        """
        freq = self.frequency[freq_index]

        table = PrettyTable()
        table.field_names = ["Parameter", "Value", "Unit"]

        table.add_row(["Operating Speed", f"{freq * 30 / np.pi:.1f}", "RPM"])
        table.add_row(["Equilibrium Mode", self.equilibrium_position_mode, "-"])

        table.add_row(
            [
                "Maximum Pressure",
                f"{self.pressure_fields[freq_index].max():.4e}",
                "Pa",
            ]
        )
        table.add_row(
            [
                "Maximum Temperature",
                f"{self.temperature_fields[freq_index].max():.1f}",
                "°C",
            ]
        )
        table.add_row(
            [
                "Maximum Film Thickness",
                f"{self.max_thicknesses[freq_index]:.4e}",
                "m",
            ]
        )
        table.add_row(
            [
                "Minimum Film Thickness",
                f"{self.min_thicknesses[freq_index]:.4e}",
                "m",
            ]
        )
        table.add_row(
            [
                "Pivot Film Thickness",
                f"{self.pivot_film_thicknesses[freq_index]:.4e}",
                "m",
            ]
        )

        if self.equilibrium_position_mode == "imposed":
            table.add_row(["Axial Load", f"{self.axial_load.sum():.4e}", "N"])
        elif self.equilibrium_position_mode == "calculate":
            table.add_row(["Axial Load", f"{self.axial_load:.4e}", "N"])

        table.add_row(["kzz (Stiffness)", f"{self.kzz[freq_index]:.4e}", "N/m"])
        table.add_row(["czz (Damping)", f"{self.czz[freq_index]:.4e}", "N*s/m"])

        desired_width = 25
        table.max_width = desired_width
        table.min_width = desired_width

        table_str = table.get_string()
        actual_width = len(table_str.split("\n")[0])

        print("\n" + "=" * actual_width)
        print(
            f"THRUST BEARING RESULTS - {freq * 30 / np.pi:.1f} RPM".center(actual_width)
        )
        print("=" * actual_width)
        print(table)
        print("=" * actual_width)

    def show_coefficients_comparison(self):
        """Print a table comparing axial dynamic coefficients across all frequencies.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        freq_rpm = self.frequency.astype(float) * 30.0 / np.pi

        table = PrettyTable()
        table.field_names = ["Frequency [RPM]", "kzz [N/m]", "czz [N*s/m]"]

        for i in range(len(freq_rpm)):
            table.add_row(
                [
                    f"{freq_rpm[i]:.1f}",
                    f"{self.kzz[i]:.4e}",
                    f"{self.czz[i]:.4e}",
                ]
            )

        desired_width = 25
        table.max_width = desired_width
        table.min_width = desired_width

        table_str = table.get_string()
        actual_width = len(table_str.split("\n")[0])

        print("\n" + "=" * actual_width)
        print("DYNAMIC COEFFICIENTS COMPARISON TABLE".center(actual_width))
        print("=" * actual_width)
        print(table)
        print("=" * actual_width)

    def plot_pressure_3d(self, freq_index=0, fig=None, **kwargs):
        """Return a 3-D surface plot of the pressure field.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """
        if fig is None:
            fig = go.Figure()

        x_coords, y_coords = self._build_cartesian_coords()
        pressure_field = self.pressure_fields[freq_index]

        fig.add_trace(
            go.Surface(
                x=x_coords,
                y=y_coords,
                z=pressure_field,
                colorscale="Viridis",
                colorbar=dict(title="Pressure [Pa]"),
                name="Pressure field",
                hovertemplate="<b>Pressure field</b><br>"
                + "X: %{x:.3f}<br>"
                + "Y: %{y:.3f}<br>"
                + "Pressure [Pa]: %{z:.3f}<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            title="Pressure field",
            scene=dict(
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                zaxis_title="Pressure [Pa]",
                camera=dict(eye=dict(x=-1.5, y=-4, z=1.5), center=dict(x=0, y=0, z=0)),
            ),
            width=800,
            height=600,
            **kwargs,
        )

        return fig

    def plot_pressure_2d(self, freq_index=0, fig=None, **kwargs):
        """Return a 2-D contour plot of the pressure field.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """
        if fig is None:
            fig = go.Figure()

        x_coords, y_coords = self._build_cartesian_coords()
        pressure_field = self.pressure_fields[freq_index]
        x_grid, y_grid, z_interp = self._build_interp_grid(
            x_coords, y_coords, pressure_field
        )

        fig.add_trace(
            go.Contour(
                x=x_grid[0, :],
                y=y_grid[:, 0],
                z=z_interp,
                colorscale="Viridis",
                colorbar=dict(title="Pressure (Pa)"),
                name="Pressure field",
                hovertemplate="<b>Pressure field</b><br>"
                + "X: %{x:.3f}<br>"
                + "Y: %{y:.3f}<br>"
                + "Pressure (Pa): %{z:.3f}<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            title="Pressure field",
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            width=800,
            height=600,
            **kwargs,
        )

        return fig

    def plot_film_temperature_3d(self, freq_index=0, fig=None, **kwargs):
        """Return a 3-D surface plot of the oil film temperature field.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """
        if fig is None:
            fig = go.Figure()

        x_coords, y_coords = self._build_cartesian_coords()
        temperature_field = self.temperature_fields[freq_index]

        fig.add_trace(
            go.Surface(
                x=x_coords,
                y=y_coords,
                z=temperature_field,
                colorscale="Viridis",
                colorbar=dict(title="Temperature [°C]"),
                name="Temperature field",
                hovertemplate="<b>Temperature field</b><br>"
                + "X: %{x:.3f}<br>"
                + "Y: %{y:.3f}<br>"
                + "Temperature [°C]: %{z:.3f}<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            title="Temperature field",
            scene=dict(
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                zaxis_title="Temperature [°C]",
                camera=dict(eye=dict(x=-1.5, y=-4, z=1.5), center=dict(x=0, y=0, z=0)),
            ),
            width=800,
            height=600,
            **kwargs,
        )

        return fig

    def plot_temperature_2d(self, freq_index=0, fig=None, **kwargs):
        """Return a 2-D contour plot of the temperature field.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """
        if fig is None:
            fig = go.Figure()

        x_coords, y_coords = self._build_cartesian_coords()
        temperature_field = self.temperature_fields[freq_index]
        x_grid, y_grid, z_interp = self._build_interp_grid(
            x_coords, y_coords, temperature_field
        )

        fig.add_trace(
            go.Contour(
                x=x_grid[0, :],
                y=y_grid[:, 0],
                z=z_interp,
                colorscale="Viridis",
                colorbar=dict(title="Temperature (°C)"),
                name="Temperature field",
                hovertemplate="<b>Temperature field</b><br>"
                + "X: %{x:.3f}<br>"
                + "Y: %{y:.3f}<br>"
                + "Temperature (°C): %{z:.3f}<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            title="Temperature field",
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            width=800,
            height=600,
            **kwargs,
        )

        return fig

    def show_optimization_convergence(
        self, by: str = "index", show_plots: bool = False
    ) -> None:
        """Display the optimization residuals per iteration for each frequency.

        Parameters
        ----------
        by : str, optional
            ``"index"`` — label frequencies by their array index (default).
            ``"value"`` — label frequencies by their value in rad/s.
        show_plots : bool, optional
            When *True* a convergence plot is shown for each frequency.
            Default is False.

        Returns
        -------
        None
        """
        if not self.optimization_history:
            print("No residual history available. Run the analysis first.")
            return

        for i, res_list in self.optimization_history.items():
            if not res_list:
                continue

            freq = self.frequency[i]
            rpm = freq * 30 / np.pi

            desired_width = 25
            table = PrettyTable()
            table.field_names = ["Iteration", "Residual [N]"]

            for it, res in enumerate(res_list):
                if res is not None:
                    table.add_row([it, f"{res:.6f}"])

            table.max_width = desired_width
            table.min_width = desired_width

            table_str = table.get_string()
            actual_width = len(table_str.split("\n")[0])

            print("\n" + "=" * actual_width)
            print(f"OPTIMIZATION CONVERGENCE - {rpm:.1f} RPM".center(actual_width))
            print("=" * actual_width)
            print(table)
            print("=" * actual_width)

            if show_plots:
                iterations = list(range(1, len(res_list) + 1))
                residuals = [res for res in res_list if res is not None]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=residuals,
                        mode="lines+markers",
                        name=f"Convergence - {rpm:.1f} RPM",
                        line=dict(width=2),
                        marker=dict(size=6),
                    )
                )
                fig.update_layout(
                    title=f"Optimization Convergence - {rpm:.1f} RPM",
                    xaxis_title="Iteration",
                    yaxis_title="Residual [N]",
                    template="ross",
                )
                fig.show()


class SqueezeFilmDamperResults(BearingResults):
    """Post-processing results for a SqueezeFilmDamper bearing.

    The SFD uses closed-form analytical expressions; no numerical pressure or
    temperature fields are solved.  The four abstract field-plot methods
    (``plot_pressure_3d``, ``plot_pressure_2d``, ``plot_film_temperature_3d``,
    ``plot_temperature_2d``) therefore raise ``NotImplementedError``.  Use
    ``plot_coefficients()`` to visualise the computed results.

    Parameters
    ----------
    frequency : array_like
        Operating frequencies in rad/s.
    kxx : array_like
        Stiffness coefficient (N/m), one value per frequency.
    cxx : array_like
        Damping coefficient (N·s/m), one value per frequency.
    theta : array_like
        Pressure angle (rad), one value per frequency.
    p_max : array_like
        Maximum pressure (Pa), one value per frequency.
    axial_length : float
        Bearing axial length (m).
    journal_radius : float
        Journal radius (m).
    radial_clearance : float
        Radial clearance (m).
    eccentricity_ratio : float
        Ratio of journal eccentricity to radial clearance.
    lubricant_viscosity : float
        Dynamic viscosity of the lubricant (Pa·s).
    geometry : str
        Geometry type: ``"groove"``, ``"end_seals"``, or
        ``"groove-end_seals"``.
    cavitation : bool
        Whether cavitation is modelled.
    initial_time : float, optional
        Solver start epoch timestamp.
    final_time : float, optional
        Solver end epoch timestamp.
    """

    def __init__(
        self,
        frequency,
        kxx,
        cxx,
        theta,
        p_max,
        axial_length,
        journal_radius,
        radial_clearance,
        eccentricity_ratio,
        lubricant_viscosity,
        geometry,
        cavitation,
        initial_time=None,
        final_time=None,
    ):
        super().__init__(
            frequency=frequency,
            pressure_fields=[],
            temperature_fields=[],
            initial_time=initial_time,
            final_time=final_time,
        )
        self.kxx = np.atleast_1d(kxx)
        self.cxx = np.atleast_1d(cxx)
        self.theta = np.atleast_1d(theta)
        self.p_max = np.atleast_1d(p_max)
        self.axial_length = axial_length
        self.journal_radius = journal_radius
        self.radial_clearance = radial_clearance
        self.eccentricity_ratio = eccentricity_ratio
        self.lubricant_viscosity = lubricant_viscosity
        self.geometry = geometry
        self.cavitation = cavitation

    def show_results(self):
        """Print a formatted summary of SFD results for all frequencies.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.frequency.size == 1:
            self._print_single_frequency_results(0)
        else:
            for i in range(self.frequency.size):
                self._print_single_frequency_results(i)

    def _print_single_frequency_results(self, freq_index):
        """Print results table for one frequency index.

        Parameters
        ----------
        freq_index : int
            Index into the frequency array.
        """
        freq = self.frequency[freq_index]
        column_width = 20

        table = PrettyTable()
        table.field_names = ["Parameter", "Value", "Unit"]

        for field in table.field_names:
            table.max_width[field] = column_width
            table.min_width[field] = column_width

        table.align["Parameter"] = "l"
        table.align["Value"] = "r"
        table.align["Unit"] = "c"

        table.add_row(["Operating Speed", f"{freq * 30 / np.pi:12.1f}", "RPM"])
        table.add_row(["Geometry Type", f"{self.geometry:>12}", "-"])
        table.add_row(["Cavitation", f"{str(self.cavitation):>12}", "-"])
        table.add_row(["Axial Length", f"{self.axial_length:12.6f}", "m"])
        table.add_row(["Journal Radius", f"{self.journal_radius:12.6f}", "m"])
        table.add_row(["Radial Clearance", f"{self.radial_clearance:12.6e}", "m"])
        table.add_row(["Eccentricity Ratio", f"{self.eccentricity_ratio:12.4f}", "-"])
        table.add_row(
            ["Lubricant Viscosity", f"{self.lubricant_viscosity:12.4e}", "Pa*s"]
        )
        table.add_row(["Damping Coefficient", f"{self.cxx[freq_index]:12.4e}", "N*s/m"])
        table.add_row(["Stiffness Coefficient", f"{self.kxx[freq_index]:12.4e}", "N/m"])
        table.add_row(
            ["Pressure Angle", f"{np.degrees(self.theta[freq_index]):12.2f}", "°"]
        )
        table.add_row(["Pressure Angle", f"{self.theta[freq_index]:12.4f}", "rad"])
        table.add_row(["Maximum Pressure", f"{self.p_max[freq_index]:12.4e}", "Pa"])

        table_str = table.get_string()
        final_width = len(table_str.split("\n")[0])

        print("\n" + "=" * final_width)
        print(
            f"SQUEEZE FILM DAMPER RESULTS - {freq * 30 / np.pi:.1f} RPM".center(
                final_width
            )
        )
        print("=" * final_width)
        print(table)
        print("=" * final_width)

    def show_coefficients_comparison(self):
        """Print a table comparing SFD coefficients across all frequencies.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        freq_rpm = self.frequency.astype(float) * 30.0 / np.pi

        table = PrettyTable()
        table.field_names = [
            "Frequency [RPM]",
            "cxx [N*s/m]",
            "kxx [N/m]",
            "Pressure [Pa]",
            "Angle [°]",
        ]

        for i in range(len(freq_rpm)):
            table.add_row(
                [
                    f"{freq_rpm[i]:.1f}",
                    f"{self.cxx[i]:.4e}",
                    f"{self.kxx[i]:.4e}",
                    f"{self.p_max[i]:.4e}",
                    f"{np.degrees(self.theta[i]):.2f}",
                ]
            )

        desired_width = 20
        table.max_width = desired_width
        table.min_width = desired_width

        table_str = table.get_string()
        actual_width = len(table_str.split("\n")[0])

        print("\n" + "=" * actual_width)
        print("SFD COEFFICIENTS COMPARISON TABLE".center(actual_width))
        print("=" * actual_width)
        print(table)
        print("=" * actual_width)

    def plot_results(self, show_plots=False, freq_index=0):
        """Not available for SqueezeFilmDamper (analytical model).

        The SFD does not solve numerical pressure or temperature fields, so no
        standard field plots are produced.  Use ``show_results()`` or
        ``show_coefficients_comparison()`` to inspect the computed coefficients.

        Parameters
        ----------
        show_plots : bool, optional
            Not used — included for API consistency with the base class.
        freq_index : int, optional
            Not used — included for API consistency with the base class.

        Returns
        -------
        figures : dict
            Empty dictionary.
        """
        print(
            "SqueezeFilmDamper uses analytical formulas — no field plots are "
            "available.  Use show_results() or show_coefficients_comparison() "
            "to inspect the computed coefficients."
        )
        return {}

    def plot_pressure_3d(self, freq_index=0, fig=None, **kwargs):
        """Not available for SqueezeFilmDamper (analytical model).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "SqueezeFilmDamper uses analytical formulas — no 3D pressure field "
            "is computed.  Use plot_coefficients() instead."
        )

    def plot_pressure_2d(self, freq_index=0, fig=None, **kwargs):
        """Not available for SqueezeFilmDamper (analytical model).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "SqueezeFilmDamper uses analytical formulas — no 2D pressure field "
            "is computed.  Use plot_coefficients() instead."
        )

    def plot_film_temperature_3d(self, freq_index=0, fig=None, **kwargs):
        """Not available for SqueezeFilmDamper (analytical model).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "SqueezeFilmDamper uses analytical formulas — no temperature field "
            "is computed."
        )

    def plot_temperature_2d(self, freq_index=0, fig=None, **kwargs):
        """Not available for SqueezeFilmDamper (analytical model).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "SqueezeFilmDamper uses analytical formulas — no temperature field "
            "is computed."
        )


class FluidFilmBearingResults(BearingResults):
    """Post-processing results for :class:`FluidFilmBearing`.

    Field arrays are shaped ``(n_pads, n_circumferential, n_axial)``: one
    grid per pad over the film mesh, with ``theta_grids`` measured from
    each pad's leading edge and ``leading_edge_angles`` placing the pads on
    the bearing circumference.

    Parameters
    ----------
    frequency : array_like
        Operating frequencies, rad/s.
    pressure_fields : list of ndarray
        Film pressure grids (Pa), one per frequency.
    temperature_fields : list of ndarray
        Radially averaged film temperature grids (K), one per frequency.
    film_thickness_fields : list of ndarray
        Film thickness grids (m), one per frequency.
    theta_grids, z_grids : list of ndarray
        Node angular position (rad, from the pad leading edge) and axial
        position (m) grids, one per frequency.
    leading_edge_angles : ndarray
        Per-pad leading edge angular position, rad.
    outputs : list of dict
        The solver's named-output dict of each frequency (eccentricity,
        attitude, power loss, flows, temperatures, ...).
    kxx, kxy, kyx, kyy : array_like
        Stiffness coefficient tables, N/m.
    cxx, cxy, cyx, cyy : array_like
        Damping coefficient tables, N*s/m.
    initial_time, final_time : float, optional
        Epoch timestamps around the solver run.

    Examples
    --------
    >>> from ross.bearings.fluid_film_bearing import fluid_film_bearing_example
    >>> bearing = fluid_film_bearing_example()
    >>> fig = bearing.plot_pressure_2d()
    """

    def __init__(
        self,
        frequency,
        pressure_fields,
        temperature_fields,
        film_thickness_fields,
        theta_grids,
        z_grids,
        leading_edge_angles,
        outputs,
        kxx,
        kxy,
        kyx,
        kyy,
        cxx,
        cxy,
        cyx,
        cyy,
        initial_time=None,
        final_time=None,
    ):
        super().__init__(
            frequency=frequency,
            pressure_fields=pressure_fields,
            temperature_fields=temperature_fields,
            initial_time=initial_time,
            final_time=final_time,
        )
        self.film_thickness_fields = film_thickness_fields
        self.theta_grids = theta_grids
        self.z_grids = z_grids
        self.leading_edge_angles = np.asarray(leading_edge_angles, dtype=float)
        self.outputs = outputs
        self.kxx = kxx
        self.kxy = kxy
        self.kyx = kyx
        self.kyy = kyy
        self.cxx = cxx
        self.cxy = cxy
        self.cyx = cyx
        self.cyy = cyy

    def _pad_indices(self, pad_index):
        n_pads = self.pressure_fields[0].shape[0]
        if pad_index is None:
            return range(n_pads)
        return [pad_index]

    def _surface_plot(self, values, freq_index, pad_index, fig, title, unit, **kwargs):
        if fig is None:
            fig = go.Figure()
        theta = self.theta_grids[freq_index]
        z = self.z_grids[freq_index]
        vmin = min(values[p].min() for p in self._pad_indices(pad_index))
        vmax = max(values[p].max() for p in self._pad_indices(pad_index))
        for p in self._pad_indices(pad_index):
            fig.add_trace(
                go.Surface(
                    x=theta[p] + self.leading_edge_angles[p],
                    y=z[p],
                    z=values[p],
                    colorscale="Viridis",
                    cmin=vmin,
                    cmax=vmax,
                    colorbar=dict(title=f"{title} [{unit}]"),
                    name=f"Pad {p + 1}",
                    hovertemplate=f"<b>Pad {p + 1}</b><br>"
                    + "Theta: %{x:.3f} rad<br>"
                    + "z: %{y:.4f} m<br>"
                    + f"{title}: %{{z:.4g}} {unit}<br>"
                    + "<extra></extra>",
                )
            )
        fig.update_layout(
            scene=dict(
                xaxis_title="Theta [rad]",
                yaxis_title="z [m]",
                zaxis_title=f"{title} [{unit}]",
            ),
            title=f"{title} field (theta vs z)",
            showlegend=False,
            **kwargs,
        )
        return fig

    def _center_plane_plot(self, values, freq_index, fig, title, unit, **kwargs):
        if fig is None:
            fig = go.Figure()
        theta = self.theta_grids[freq_index]
        mid = values.shape[2] // 2
        for p in range(values.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=theta[p][:, mid] + self.leading_edge_angles[p],
                    y=values[p][:, mid],
                    mode="lines",
                    name=f"Pad {p + 1}",
                )
            )
        fig.update_layout(
            xaxis_title="Theta [rad]",
            yaxis_title=f"{title} [{unit}]",
            title=f"{title} at the axial center plane",
            **kwargs,
        )
        return fig

    def plot_pressure_3d(self, freq_index=0, pad_index=None, fig=None, **kwargs):
        """Return a 3-D surface plot of the film pressure field.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index. Default is 0.
        pad_index : int, optional
            Plot a single pad (0-based). Default plots every pad.
        fig : go.Figure, optional
            Existing figure to add the traces to.
        **kwargs : dict
            Additional layout options forwarded to ``fig.update_layout``.

        Returns
        -------
        fig : go.Figure
        """
        return self._surface_plot(
            self.pressure_fields[freq_index],
            freq_index,
            pad_index,
            fig,
            "Pressure",
            "Pa",
            **kwargs,
        )

    def plot_pressure_2d(self, freq_index=0, fig=None, **kwargs):
        """Return the film pressure along the axial center plane, per pad.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index. Default is 0.
        fig : go.Figure, optional
            Existing figure to add the traces to.
        **kwargs : dict
            Additional layout options forwarded to ``fig.update_layout``.

        Returns
        -------
        fig : go.Figure
        """
        return self._center_plane_plot(
            self.pressure_fields[freq_index],
            freq_index,
            fig,
            "Pressure",
            "Pa",
            **kwargs,
        )

    def plot_film_temperature_3d(
        self, freq_index=0, pad_index=None, fig=None, **kwargs
    ):
        """Return a 3-D surface plot of the film temperature field.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index. Default is 0.
        pad_index : int, optional
            Plot a single pad (0-based). Default plots every pad.
        fig : go.Figure, optional
            Existing figure to add the traces to.
        **kwargs : dict
            Additional layout options forwarded to ``fig.update_layout``.

        Returns
        -------
        fig : go.Figure
        """
        return self._surface_plot(
            self.temperature_fields[freq_index],
            freq_index,
            pad_index,
            fig,
            "Temperature",
            "K",
            **kwargs,
        )

    def plot_temperature_2d(self, freq_index=0, fig=None, **kwargs):
        """Return the film temperature along the axial center plane, per pad.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index. Default is 0.
        fig : go.Figure, optional
            Existing figure to add the traces to.
        **kwargs : dict
            Additional layout options forwarded to ``fig.update_layout``.

        Returns
        -------
        fig : go.Figure
        """
        return self._center_plane_plot(
            self.temperature_fields[freq_index],
            freq_index,
            fig,
            "Temperature",
            "K",
            **kwargs,
        )

    def plot_film_thickness_2d(self, freq_index=0, fig=None, **kwargs):
        """Return the film thickness along the axial center plane, per pad.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index. Default is 0.
        fig : go.Figure, optional
            Existing figure to add the traces to.
        **kwargs : dict
            Additional layout options forwarded to ``fig.update_layout``.

        Returns
        -------
        fig : go.Figure
        """
        return self._center_plane_plot(
            self.film_thickness_fields[freq_index],
            freq_index,
            fig,
            "Film thickness",
            "m",
            **kwargs,
        )

    def show_results(self):
        """Print a per-frequency summary of the bearing solution.

        Returns
        -------
        None
        """
        table = PrettyTable()
        table.field_names = [
            "Frequency [RPM]",
            "Eccentricity [-]",
            "Attitude [deg]",
            "Power loss [W]",
            "Max pressure [Pa]",
            "Max temperature [K]",
            "Side flow [m^3/s]",
        ]
        from ross.bearings.fluid_film.driver import ZERO_TEMPERATURE_SENTINEL

        for i, out in enumerate(self.outputs):
            tpad_max = out["tpad_max"][0]
            has_thermal = abs(tpad_max - ZERO_TEMPERATURE_SENTINEL) > 1e-9
            table.add_row(
                [
                    f"{self.frequency[i] * 30.0 / np.pi:.1f}",
                    f"{out['eccentricity'][0]:.4f}",
                    f"{np.degrees(out['attitude'][0]):.1f}",
                    f"{out['power_loss'][0]:.4g}",
                    f"{out['y_max_p'][0]:.4g}",
                    f"{tpad_max:.2f}" if has_thermal else "-",
                    f"{out['differential_flow_rate'][0]:.4g}",
                ]
            )
        actual_width = len(table.get_string().split("\n")[0])
        print("\n" + "=" * actual_width)
        print("FLUID FILM BEARING RESULTS".center(actual_width))
        print("=" * actual_width)
        print(table)
        print("=" * actual_width)

    def show_coefficients_comparison(self):
        """Print a table comparing dynamic coefficients across frequencies.

        Returns
        -------
        None
        """
        freq_rpm = self.frequency.astype(float) * 30.0 / np.pi

        table = PrettyTable()
        table.field_names = [
            "Frequency [RPM]",
            "kxx [N/m]",
            "kxy [N/m]",
            "kyx [N/m]",
            "kyy [N/m]",
            "cxx [N*s/m]",
            "cxy [N*s/m]",
            "cyx [N*s/m]",
            "cyy [N*s/m]",
        ]
        for i in range(len(freq_rpm)):
            table.add_row(
                [
                    f"{freq_rpm[i]:.1f}",
                    f"{self.kxx[i]:.4e}",
                    f"{self.kxy[i]:.4e}",
                    f"{self.kyx[i]:.4e}",
                    f"{self.kyy[i]:.4e}",
                    f"{self.cxx[i]:.4e}",
                    f"{self.cxy[i]:.4e}",
                    f"{self.cyx[i]:.4e}",
                    f"{self.cyy[i]:.4e}",
                ]
            )
        actual_width = len(table.get_string().split("\n")[0])
        print("\n" + "=" * actual_width)
        print("DYNAMIC COEFFICIENTS COMPARISON TABLE".center(actual_width))
        print("=" * actual_width)
        print(table)
        print("=" * actual_width)
