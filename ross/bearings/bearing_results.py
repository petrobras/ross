from abc import ABC, abstractmethod

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from prettytable import PrettyTable

from ross.plotly_theme import tableau_colors

__all__ = [
    "BearingResults",
    "TiltingPadResults",
]


class BearingResults(ABC):
    """Abstract base class for fluid film bearing post-processing results.

    Each bearing class (TiltingPad, PlainJournal, ThrustPad, SqueezeFilmDamper)
    creates a ``_results`` attribute of the corresponding subclass after the solver
    runs. The bearing then delegates every ``plot_*`` and ``show_*`` call to that
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
    >>> from ross.bearings.tilting_pad import tilting_pad_example
    >>> bearing = tilting_pad_example()
    >>> bearing.show_results()           # delegates to TiltingPadResults
    >>> bearing.plot_pressure_3d()       # delegates to TiltingPadResults
    >>> figs = bearing.plot_results()    # dict of go.Figure objects
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
            print(f"Execution time: {total_time:.2f} seconds")
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
            ``"temperature_2d"``, and ``"temperature_3d"``.  Each value is a
            ``plotly.graph_objects.Figure``.
        """
        figures = {
            "pressure_2d": self.plot_pressure_2d(freq_index=freq_index),
            "pressure_3d": self.plot_pressure_3d(freq_index=freq_index),
            "temperature_2d": self.plot_temperature_2d(freq_index=freq_index),
            "temperature_3d": self.plot_temperature_3d(freq_index=freq_index),
        }

        if show_plots:
            for fig in figures.values():
                try:
                    fig.show()
                except Exception as e:
                    print(
                        f"Warning: Could not display plot automatically. Error: {e}"
                    )

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
    def plot_temperature_3d(self, freq_index=0, fig=None, **kwargs):
        """Return a 3-D surface plot of the temperature field.

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


class TiltingPadResults(BearingResults):
    """Post-processing results for a TiltingPad bearing.

    Parameters
    ----------
    frequency : array_like
        Operating frequencies in rad/s.
    pressure_fields : list of ndarray, shape (nz, nx, n_pad)
        Pressure fields, one per frequency.
    temperature_fields : list of ndarray, shape (nz, nx, n_pad)
        Temperature fields, one per frequency.
    maxP_list : list of float
        Maximum pressure per frequency (Pa).
    maxT_list : list of float
        Maximum temperature per frequency (°C).
    minH_list : list of float
        Minimum film thickness per frequency (m).
    ecc_list : list of float
        Journal eccentricity ratio per frequency.
    attitude_angle_list : list of float
        Journal attitude angle per frequency (rad).
    psi_pad_list : list of array_like
        Pad rotation angles per frequency (rad).
    force_x_total_list : list of float
        Total hydrodynamic force in X per frequency (N).
    force_y_total_list : list of float
        Total hydrodynamic force in Y per frequency (N).
    momen_rot_list : list of array_like or None
        Pad moments per frequency (N·m).  ``None`` entries indicate that
        the *determine_eccentricity* equilibrium type was used.
    kxx : ndarray
        Direct stiffness coefficient in XX (N/m), one value per frequency.
    kxy : ndarray
        Cross stiffness coefficient in XY (N/m).
    kyx : ndarray
        Cross stiffness coefficient in YX (N/m).
    kyy : ndarray
        Direct stiffness coefficient in YY (N/m).
    cxx : ndarray
        Direct damping coefficient in XX (N·s/m).
    cxy : ndarray
        Cross damping coefficient in XY (N·s/m).
    cyx : ndarray
        Cross damping coefficient in YX (N·s/m).
    cyy : ndarray
        Direct damping coefficient in YY (N·s/m).
    equilibrium_type : str
        ``"match_eccentricity"`` or ``"determine_eccentricity"``.
    n_pad : int
        Number of pads.
    xtheta : ndarray, shape (nx,)
        Circumferential mesh coordinates (rad, relative to pad centre).
    xz : ndarray, shape (nz,)
        Axial mesh coordinates (non-dimensional, −0.5 to 0.5).
    pivot_angle : ndarray, shape (n_pad,)
        Absolute pivot angle of each pad (rad).
    pad_axial_length : float
        Pad axial length (m).
    nz : int
        Number of axial mesh elements.
    nx : int
        Number of circumferential mesh elements.
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
        maxP_list,
        maxT_list,
        minH_list,
        ecc_list,
        attitude_angle_list,
        psi_pad_list,
        force_x_total_list,
        force_y_total_list,
        momen_rot_list,
        kxx,
        kxy,
        kyx,
        kyy,
        cxx,
        cxy,
        cyx,
        cyy,
        equilibrium_type,
        n_pad,
        xtheta,
        xz,
        pivot_angle,
        pad_axial_length,
        nz,
        nx,
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
        self.maxP_list = maxP_list
        self.maxT_list = maxT_list
        self.minH_list = minH_list
        self.ecc_list = ecc_list
        self.attitude_angle_list = attitude_angle_list
        self.psi_pad_list = psi_pad_list
        self.force_x_total_list = force_x_total_list
        self.force_y_total_list = force_y_total_list
        self.momen_rot_list = momen_rot_list
        self.kxx = np.atleast_1d(kxx)
        self.kxy = np.atleast_1d(kxy)
        self.kyx = np.atleast_1d(kyx)
        self.kyy = np.atleast_1d(kyy)
        self.cxx = np.atleast_1d(cxx)
        self.cxy = np.atleast_1d(cxy)
        self.cyx = np.atleast_1d(cyx)
        self.cyy = np.atleast_1d(cyy)
        self.equilibrium_type = equilibrium_type
        self.n_pad = n_pad
        self.xtheta = xtheta
        self.xz = xz
        self.pivot_angle = pivot_angle
        self.pad_axial_length = pad_axial_length
        self.nz = nz
        self.nx = nx
        self.optimization_history = optimization_history

    def show_results(self):
        """Print a formatted summary of tilting pad bearing results.

        Iterates over all solved frequencies and prints a PrettyTable with
        operating conditions, field extrema, equilibrium data, loads, dynamic
        coefficients, and per-pad rotation angles.

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
        table.add_row(["Equilibrium Type", f"{self.equilibrium_type:>12}", "-"])
        table.add_row(["Number of Pads", f"{self.n_pad:12d}", "-"])

        table.add_row(
            ["Maximum Pressure", f"{self.maxP_list[freq_index]:12.4e}", "Pa"]
        )
        table.add_row(
            [
                "Maximum Temperature",
                f"{self.maxT_list[freq_index]:12.2f}",
                "°C",
            ]
        )
        table.add_row(
            [
                "Minimum Film Thickness",
                f"{self.minH_list[freq_index]:12.4e}",
                "m",
            ]
        )

        table.add_row(
            ["Eccentricity", f"{self.ecc_list[freq_index]:12.4f}", "-"]
        )
        table.add_row(
            [
                "Attitude Angle",
                f"{np.degrees(self.attitude_angle_list[freq_index]):12.2f}",
                "°",
            ]
        )

        table.add_row(
            [
                "Total Force X",
                f"{self.force_x_total_list[freq_index]:12.4e}",
                "N",
            ]
        )
        table.add_row(
            [
                "Total Force Y",
                f"{self.force_y_total_list[freq_index]:12.4e}",
                "N",
            ]
        )

        table.add_row(["kxx (Stiffness)", f"{self.kxx[freq_index]:12.4e}", "N/m"])
        table.add_row(["kxy (Stiffness)", f"{self.kxy[freq_index]:12.4e}", "N/m"])
        table.add_row(["kyx (Stiffness)", f"{self.kyx[freq_index]:12.4e}", "N/m"])
        table.add_row(["kyy (Stiffness)", f"{self.kyy[freq_index]:12.4e}", "N/m"])

        table.add_row(["cxx (Damping)", f"{self.cxx[freq_index]:12.4e}", "N*s/m"])
        table.add_row(["cxy (Damping)", f"{self.cxy[freq_index]:12.4e}", "N*s/m"])
        table.add_row(["cyx (Damping)", f"{self.cyx[freq_index]:12.4e}", "N*s/m"])
        table.add_row(["cyy (Damping)", f"{self.cyy[freq_index]:12.4e}", "N*s/m"])

        pad_table = PrettyTable()
        pad_table.align["Pad #"] = "c"

        if (
            self.momen_rot_list[freq_index] is not None
            and self.equilibrium_type == "match_eccentricity"
        ):
            pad_table.field_names = [
                "Pad #",
                "Moment [N·m]",
                "Angle [rad]",
                "Angle [°]",
            ]
            pad_table.align["Moment [N·m]"] = "r"
            pad_table.align["Angle [rad]"] = "r"
            pad_table.align["Angle [°]"] = "r"

            for i in range(self.n_pad):
                pad_table.add_row(
                    [
                        i + 1,
                        f"{self.momen_rot_list[freq_index][i]:12.4e}",
                        f"{self.psi_pad_list[freq_index][i]:12.4e}",
                        f"{np.degrees(self.psi_pad_list[freq_index][i]):12.4e}",
                    ]
                )
        else:
            pad_table.field_names = ["Pad #", "Angle [rad]", "Angle [°]"]
            pad_table.align["Angle [rad]"] = "r"
            pad_table.align["Angle [°]"] = "r"

            for i in range(self.n_pad):
                pad_table.add_row(
                    [
                        i + 1,
                        f"{self.psi_pad_list[freq_index][i]:12.4e}",
                        f"{np.degrees(self.psi_pad_list[freq_index][i]):12.4e}",
                    ]
                )

        column_width = 14
        for field in pad_table.field_names:
            pad_table.max_width[field] = column_width
            pad_table.min_width[field] = column_width

        table_str = table.get_string()
        final_width = len(table_str.split("\n")[0])

        print("\n" + "=" * final_width)
        print(
            f"TILTING PAD BEARING RESULTS - {freq * 30 / np.pi:.1f} RPM".center(
                final_width
            )
        )
        print("=" * final_width)
        print(table)

        print("\n" + "-" * final_width)
        print("PAD ROTATION ANGLES".center(final_width))
        print("-" * final_width)
        print(pad_table)
        print("=" * final_width)

    def show_coefficients_comparison(self):
        """Print a table comparing dynamic coefficients across all frequencies.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        freq_rpm = self.frequency.astype(float) * 30.0 / np.pi

        table = PrettyTable()
        headers = [
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
        table.field_names = headers

        for i in range(len(freq_rpm)):
            row = [
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
            table.add_row(row)

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

    def plot_pressure_3d(self, freq_index=0, pad_index=None, fig=None, **kwargs):
        """Return a 3-D surface plot of the pressure field for one pad.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        pad_index : int, optional
            Pad index.  When ``None`` the pad with the highest peak pressure
            at the chosen frequency is selected automatically.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """
        if fig is None:
            fig = go.Figure()

        pressure_field = self.pressure_fields[freq_index]

        if pad_index is None:
            pad_index = int(np.argmax(pressure_field.max(axis=(0, 1))))

        XH, YH = np.meshgrid(self.xtheta, self.xz)

        fig.add_trace(
            go.Surface(
                x=XH,
                y=YH,
                z=1e-6 * pressure_field[:, :, pad_index],
                colorscale="Viridis",
                name=f"Pressure field - Pad {pad_index + 1}",
                showscale=True,
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Pressure Distribution - Pad {pad_index + 1}",
                font=dict(size=24),
            ),
            scene=dict(
                xaxis_title=dict(text="X direction [rad]", font=dict(size=14)),
                yaxis_title=dict(text="Z direction [-]", font=dict(size=14)),
                zaxis_title=dict(text="Pressure [MPa]", font=dict(size=14)),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            **kwargs,
        )

        return fig

    def plot_pressure_2d(self, freq_index=0, fig=None, **kwargs):
        """Return a 2-D contour plot of the pressure field for all pads.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add traces to.

        Returns
        -------
        fig : go.Figure
        """
        pressure_field = self.pressure_fields[freq_index]

        d_axial = self.pad_axial_length / self.nz
        axial = np.arange(0, self.pad_axial_length + d_axial, d_axial)
        axial = axial[1:] - np.diff(axial) / 2

        ang = [
            (self.xtheta + self.pivot_angle[k]) * 180 / np.pi
            for k in range(self.n_pad)
        ]

        return self._plot_contour(
            x_data=ang,
            y_data=axial,
            z_data=pressure_field,
            z_title="Pressure (Pa)",
            zmin=0,
            ncontours=15,
            fig=fig,
            **kwargs,
        )

    def plot_temperature_3d(self, freq_index=0, pad_index=None, fig=None, **kwargs):
        """Return a 3-D surface plot of the temperature field for one pad.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        pad_index : int, optional
            Pad index.  When ``None`` the pad with the highest peak temperature
            at the chosen frequency is selected automatically.
        fig : go.Figure, optional
            Existing figure to add the trace to.

        Returns
        -------
        fig : go.Figure
        """
        if fig is None:
            fig = go.Figure()

        temperature_field = self.temperature_fields[freq_index]

        if pad_index is None:
            pad_index = int(np.argmax(temperature_field.max(axis=(0, 1))))

        XH, YH = np.meshgrid(self.xtheta, self.xz)

        fig.add_trace(
            go.Surface(
                x=XH,
                y=YH,
                z=temperature_field[:, :, pad_index],
                colorscale="Viridis",
                name=f"Temperature field - Pad {pad_index + 1}",
                showscale=True,
                colorbar=dict(title="Temperature [°C]"),
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Temperature Distribution - Pad {pad_index + 1}",
                font=dict(size=24),
            ),
            scene=dict(
                xaxis_title=dict(text="X direction [rad]", font=dict(size=14)),
                yaxis_title=dict(text="Z direction [-]", font=dict(size=14)),
                zaxis_title=dict(text="Temperature [°C]", font=dict(size=14)),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            **kwargs,
        )

        return fig

    def plot_temperature_2d(self, freq_index=0, fig=None, **kwargs):
        """Return a 2-D contour plot of the temperature field for all pads.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add traces to.

        Returns
        -------
        fig : go.Figure
        """
        temperature_field = self.temperature_fields[freq_index]

        d_axial = self.pad_axial_length / self.nz
        axial = np.arange(0, self.pad_axial_length + d_axial, d_axial)
        axial = axial[1:] - np.diff(axial) / 2

        ang = [
            (self.xtheta + self.pivot_angle[k]) * 180 / np.pi
            for k in range(self.n_pad)
        ]

        return self._plot_contour(
            x_data=ang,
            y_data=axial,
            z_data=temperature_field,
            z_title="Temperature (ºC)",
            zmin=temperature_field.min(),
            ncontours=25,
            fig=fig,
            **kwargs,
        )

    def plot_pad_pressure(self, freq_index=0, fig=None, **kwargs):
        """Return a scatter plot of the midplane pressure profile per pad.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index.  Default is 0.
        fig : go.Figure, optional
            Existing figure to add traces to.

        Returns
        -------
        fig : go.Figure
        """
        pressure_field = self.pressure_fields[freq_index]

        d_axial = self.pad_axial_length / self.nz
        axial = np.arange(0, self.pad_axial_length + d_axial, d_axial)
        axial = axial[1:] - np.diff(axial) / 2

        ang = [
            (self.xtheta + self.pivot_angle[k]) * 180 / np.pi
            for k in range(self.n_pad)
        ]

        midplane_idx = self.nz // 2

        return self._plot_scatter(
            x_data=ang,
            y_data=pressure_field,
            pos=midplane_idx,
            y_title="Pressure (Pa)",
            fig=fig,
            **kwargs,
        )

    def plot_results(self, show_plots=False, freq_index=0):
        """Return all tilting pad result plots, including pad scatter plots.

        Extends the base ``plot_results`` with two additional entries:
        ``"pressure_scatter"`` and ``"temperature_scatter"``.

        Parameters
        ----------
        show_plots : bool, optional
            When *True* each figure is displayed immediately. Default is False.
        freq_index : int, optional
            Frequency index.  Default is 0.

        Returns
        -------
        figures : dict
            Dictionary with keys ``"pressure_2d"``, ``"pressure_3d"``,
            ``"temperature_2d"``, ``"temperature_3d"``,
            ``"pressure_scatter"``, and ``"temperature_scatter"``.
        """
        figures = super().plot_results(show_plots=False, freq_index=freq_index)

        pressure_field = self.pressure_fields[freq_index]
        temperature_field = self.temperature_fields[freq_index]

        d_axial = self.pad_axial_length / self.nz
        axial = np.arange(0, self.pad_axial_length + d_axial, d_axial)
        axial = axial[1:] - np.diff(axial) / 2

        ang = [
            (self.xtheta + self.pivot_angle[k]) * 180 / np.pi
            for k in range(self.n_pad)
        ]

        midplane_idx = self.nz // 2

        figures["pressure_scatter"] = self._plot_scatter(
            x_data=ang,
            y_data=pressure_field,
            pos=midplane_idx,
            y_title="Pressure (Pa)",
        )
        figures["temperature_scatter"] = self._plot_scatter(
            x_data=ang,
            y_data=temperature_field,
            pos=midplane_idx,
            y_title="Temperature (ºC)",
        )

        if show_plots:
            for fig in figures.values():
                try:
                    fig.show()
                except Exception as e:
                    print(
                        f"Warning: Could not display plot automatically. Error: {e}"
                    )

        return figures

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

            if self.equilibrium_type == "match_eccentricity":
                n_pads = self.n_pad
                total_iters = len(res_list)
                approx_iters_per_pad = total_iters // n_pads

                table = PrettyTable()
                table.field_names = ["Pad", "Iterations", "Final Residual [N]"]

                pad_residuals = []
                for pad_idx in range(n_pads):
                    start_idx = pad_idx * approx_iters_per_pad
                    end_idx = (
                        (pad_idx + 1) * approx_iters_per_pad
                        if pad_idx < n_pads - 1
                        else total_iters
                    )
                    pad_res = [r for r in res_list[start_idx:end_idx] if r is not None]

                    if pad_res:
                        final_res = pad_res[-1]
                        table.add_row([pad_idx + 1, len(pad_res), f"{final_res:.6f}"])
                        pad_residuals.append((pad_idx + 1, pad_res))

                desired_width = 25
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
                    n_rows = (n_pads + 1) // 2
                    fig = make_subplots(
                        rows=n_rows,
                        cols=2,
                        subplot_titles=[f"Pad {p[0]}" for p in pad_residuals],
                    )

                    for idx, (pad_num, pad_res) in enumerate(pad_residuals):
                        row = (idx // 2) + 1
                        col = (idx % 2) + 1

                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(pad_res))),
                                y=pad_res,
                                mode="lines+markers",
                                name=f"Pad {pad_num}",
                                line=dict(width=2),
                                marker=dict(size=4),
                            ),
                            row=row,
                            col=col,
                        )

                        fig.update_xaxes(title_text="Iteration", row=row, col=col)
                        fig.update_yaxes(title_text="Residual [N]", row=row, col=col)

                    fig.update_layout(
                        title=f"Optimization Convergence by Pad - {rpm:.1f} RPM",
                        template="ross",
                        showlegend=False,
                        height=300 * n_rows,
                    )
                    fig.show()

            else:
                table = PrettyTable()
                table.field_names = ["Iteration", "Residual [N]"]

                for it, res in enumerate(res_list):
                    if res is not None:
                        table.add_row([it, f"{res:.6f}"])

                desired_width = 25
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
                    iterations = list(range(len(res_list)))
                    residuals = [res if res is not None else 0 for res in res_list]

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
                        title=f"Global Optimization Convergence - {rpm:.1f} RPM",
                        xaxis_title="Iteration",
                        yaxis_title="Residual [N]",
                        template="ross",
                    )
                    fig.show()

    def _plot_scatter(self, x_data, y_data, pos, y_title, fig=None, **kwargs):
        """Return a scatter plot of field data at a fixed axial position.

        Parameters
        ----------
        x_data : list of ndarray
            Circumferential coordinate for each pad (degrees).
        y_data : ndarray, shape (nz, nx, n_pad)
            Field values.
        pos : int
            Axial index (z-slice) to extract.
        y_title : str
            Y-axis label.
        fig : go.Figure, optional
            Existing figure to add traces to.

        Returns
        -------
        fig : go.Figure
        """
        if fig is None:
            fig = go.Figure()

        for i in range(self.n_pad):
            fig.add_trace(
                go.Scatter(
                    x=x_data[i],
                    y=y_data[pos, :, i],
                    name=f"Pad {i + 1}",
                )
            )

        fig.update_layout(
            plot_bgcolor="white",
            xaxis_range=[
                np.array(x_data).min() * 1.1,
                360 - abs(np.array(x_data).min()),
            ],
            legend=dict(font=dict(family="Times New Roman", size=22, color="black")),
            **kwargs,
        )
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGray",
            tickfont=dict(size=22),
            title=dict(text="Angle (°)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="LightGray",
            tickfont=dict(size=22),
            title=dict(text=y_title, font=dict(family="Times New Roman", size=30)),
        )

        return fig

    def _plot_contour(
        self, x_data, y_data, z_data, z_title, zmin, ncontours, fig=None, **kwargs
    ):
        """Return a 2-D contour plot of field data across all pads.

        Parameters
        ----------
        x_data : list of ndarray
            Circumferential coordinate for each pad (degrees).
        y_data : ndarray
            Axial coordinate array.
        z_data : ndarray, shape (nz, nx, n_pad)
            Field values.
        z_title : str
            Colour bar label.
        zmin : float
            Minimum value of the colour scale.
        ncontours : int
            Number of contour levels.
        fig : go.Figure, optional
            Existing figure to add traces to.

        Returns
        -------
        fig : go.Figure
        """
        if fig is None:
            fig = go.Figure()

        max_val = z_data.max()

        for l in range(self.n_pad):
            fig.add_trace(
                go.Contour(
                    z=z_data[:, :, l],
                    x=x_data[l],
                    y=y_data,
                    zmin=zmin,
                    zmax=max_val,
                    ncontours=ncontours,
                    colorscale="Viridis",
                    colorbar=dict(
                        title=z_title,
                        tickfont=dict(size=22),
                    ),
                )
            )

        fig.update_traces(
            contours_coloring="fill",
            contours_showlabels=True,
            contours_labelfont=dict(size=20),
        )
        fig.update_layout(
            plot_bgcolor="white",
            xaxis_range=[
                np.array(x_data).min() * 1.1,
                360 - abs(np.array(x_data).min()),
            ],
            **kwargs,
        )
        fig.update_xaxes(
            tickfont=dict(size=22),
            title=dict(text="Angle (°)", font=dict(family="Times New Roman", size=30)),
        )
        fig.update_yaxes(
            tickfont=dict(size=22),
            title=dict(
                text="Pad Length (m)", font=dict(family="Times New Roman", size=30)
            ),
        )

        return fig
