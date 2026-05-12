from abc import ABC, abstractmethod

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from prettytable import PrettyTable
from scipy.interpolate import griddata

from ross.plotly_theme import tableau_colors

__all__ = [
    "BearingResults",
    "TiltingPadResults",
    "ThrustPadResults",
    "PlainJournalResults",
    "SqueezeFilmDamperResults",
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

        for pad_idx in range(self.n_pad):
            fig.add_trace(
                go.Contour(
                    z=z_data[:, :, pad_idx],
                    x=x_data[pad_idx],
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
            f"THRUST BEARING RESULTS - {freq * 30 / np.pi:.1f} RPM".center(
                actual_width
            )
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
                camera=dict(
                    eye=dict(x=-1.5, y=-4, z=1.5), center=dict(x=0, y=0, z=0)
                ),
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
                camera=dict(
                    eye=dict(x=-1.5, y=-4, z=1.5), center=dict(x=0, y=0, z=0)
                ),
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


class PlainJournalResults(BearingResults):
    """Post-processing results for a PlainJournal bearing.

    Parameters
    ----------
    frequency : array_like
        Operating frequencies in rad/s.
    pressure_fields : list of ndarray, shape (elements_axial, circumferential_total)
        Dimensional pressure fields at equilibrium, one per frequency (Pa).
    temperature_fields : list of ndarray, shape (elements_axial, circumferential_total)
        Dimensional temperature fields at equilibrium, one per frequency (°C).
    theta_grids : list of ndarray
        Circumferential coordinate meshgrids, one per frequency (rad).
    z_grids : list of ndarray
        Axial coordinate meshgrids, one per frequency (m).
    P_nondim_fields : list of ndarray, shape (elements_axial, elements_circumferential, n_pad)
        Non-dimensional pressure fields at equilibrium, one per frequency.
        Used by ``plot_pressure_distribution``.
    kxx, kxy, kyx, kyy : ndarray
        Stiffness coefficients (N/m), one value per frequency.
    cxx, cxy, cyx, cyy : ndarray
        Damping coefficients (N·s/m), one value per frequency.
    fxs_load : float
        Applied load in the X direction (N).
    fys_load : float
        Applied load in the Y direction (N).
    n_pad : int
        Number of pads.
    betha_s_dg : float
        Pad arc length in degrees.
    dtheta : float
        Circumferential mesh step (rad).
    thetaF : ndarray
        Pad end angles (rad), one per pad.
    elements_axial : int
        Number of axial mesh elements.
    elements_circumferential : int
        Number of circumferential mesh elements per pad.
    equilibrium_pos_by_speed : dict
        Mapping ``{speed_rad_s: equilibrium_position}``.
    opt_results : dict
        Mapping ``{speed_rad_s: scipy.OptimizeResult}``.
    exec_times : dict
        Mapping ``{speed_rad_s: elapsed_seconds}``.
    optimization_history : dict
        Mapping ``{speed_rad_s: [residuals]}``.
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
        theta_grids,
        z_grids,
        P_nondim_fields,
        kxx,
        kxy,
        kyx,
        kyy,
        cxx,
        cxy,
        cyx,
        cyy,
        fxs_load,
        fys_load,
        n_pad,
        betha_s_dg,
        dtheta,
        thetaF,
        elements_axial,
        elements_circumferential,
        equilibrium_pos_by_speed,
        opt_results,
        exec_times,
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
        self.theta_grids = theta_grids
        self.z_grids = z_grids
        self.P_nondim_fields = P_nondim_fields
        self.kxx = np.atleast_1d(kxx)
        self.kxy = np.atleast_1d(kxy)
        self.kyx = np.atleast_1d(kyx)
        self.kyy = np.atleast_1d(kyy)
        self.cxx = np.atleast_1d(cxx)
        self.cxy = np.atleast_1d(cxy)
        self.cyx = np.atleast_1d(cyx)
        self.cyy = np.atleast_1d(cyy)
        self.fxs_load = fxs_load
        self.fys_load = fys_load
        self.n_pad = n_pad
        self.betha_s_dg = betha_s_dg
        self.dtheta = dtheta
        self.thetaF = thetaF
        self.elements_axial = elements_axial
        self.elements_circumferential = elements_circumferential
        self.equilibrium_pos_by_speed = equilibrium_pos_by_speed
        self.opt_results = opt_results
        self.exec_times = exec_times
        self.optimization_history = optimization_history

    def show_results(self):
        """Print a formatted summary of plain journal bearing results.

        Iterates over all solved frequencies and prints a PrettyTable with
        operating conditions, load, equilibrium position, stiffness and
        damping coefficients, and optimization statistics.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        freq_arr = np.atleast_1d(self.frequency)
        for speed_rad in freq_arr:
            self._print_single_frequency_results(float(speed_rad))

    def _print_single_frequency_results(self, speed_rad):
        """Print results table for one speed.

        Parameters
        ----------
        speed_rad : float
            Operating speed in rad/s.
        """
        freq_arr = np.atleast_1d(self.frequency).astype(float)
        idx = int(np.argmin(np.abs(freq_arr - float(speed_rad))))

        k = (self.kxx[idx], self.kxy[idx], self.kyx[idx], self.kyy[idx])
        c = (self.cxx[idx], self.cxy[idx], self.cyx[idx], self.cyy[idx])

        rpm_display = float(speed_rad) * 30.0 / np.pi

        eq = self.equilibrium_pos_by_speed.get(float(speed_rad))
        ecc = float(eq[0]) if eq is not None else None
        attitude_deg = float(eq[1]) * 180.0 / np.pi if eq is not None else None

        table = PrettyTable()
        table.field_names = ["Parameter", "Value", "Unit"]

        table.add_row(["Operating Speed", f"{rpm_display:.1f}", "RPM"])
        if ecc is not None:
            table.add_row(["Eccentricity Ratio", f"{ecc:.4e}", "-"])
        if attitude_deg is not None:
            table.add_row(["Attitude Angle", f"{attitude_deg:.4e}", "deg"])

        try:
            fx = float(getattr(self.fxs_load, "m", self.fxs_load))
        except Exception:
            fx = float(self.fxs_load)
        try:
            fy = float(getattr(self.fys_load, "m", self.fys_load))
        except Exception:
            fy = float(self.fys_load)
        table.add_row(["Load Fx", f"{fx:.4e}", "N"])
        table.add_row(["Load Fy", f"{fy:.4e}", "N"])

        table.add_row(["kxx (Stiffness)", f"{k[0]:.4e}", "N/m"])
        table.add_row(["kxy (Stiffness)", f"{k[1]:.4e}", "N/m"])
        table.add_row(["kyx (Stiffness)", f"{k[2]:.4e}", "N/m"])
        table.add_row(["kyy (Stiffness)", f"{k[3]:.4e}", "N/m"])
        table.add_row(["cxx (Damping)", f"{c[0]:.4e}", "N*s/m"])
        table.add_row(["cxy (Damping)", f"{c[1]:.4e}", "N*s/m"])
        table.add_row(["cyx (Damping)", f"{c[2]:.4e}", "N*s/m"])
        table.add_row(["cyy (Damping)", f"{c[3]:.4e}", "N*s/m"])

        res = self.opt_results.get(float(speed_rad))
        exec_time = self.exec_times.get(float(speed_rad))

        if res is not None:
            table.add_row(["Optimization Success", f"{bool(res.success)}", "-"])
            table.add_row(["Function Value", f"{float(res.fun):.4e}", "-"])
            table.add_row(["Iterations", f"{int(res.nit)}", "-"])
            table.add_row(["Evaluations", f"{int(res.nfev)}", "-"])
        if exec_time is not None:
            table.add_row(["Execution Time", f"{exec_time:.4e}", "s"])

        desired_width = 25
        table.max_width = desired_width
        table.min_width = desired_width

        table_str = table.get_string()
        actual_width = len(table_str.split("\n")[0])

        print("\n" + "=" * actual_width)
        print(f"PLAIN JOURNAL RESULTS - {rpm_display:.1f} RPM".center(actual_width))
        print("=" * actual_width)
        print(table)
        print("=" * actual_width)

    def show_coefficients_comparison(self):
        """Print a table comparing the full 2×2 dynamic coefficient matrix
        across all frequencies.

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
        """Return a 3-D surface plot of the pressure field (theta vs z).

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
        if not self.pressure_fields:
            raise RuntimeError("No field data available.")

        if fig is None:
            fig = go.Figure()

        P = self.pressure_fields[freq_index]
        theta_grid = self.theta_grids[freq_index]
        z_grid = self.z_grids[freq_index]

        fig.add_trace(
            go.Surface(
                x=theta_grid,
                y=z_grid,
                z=P,
                colorscale="Viridis",
                colorbar=dict(title="Pressure [Pa]"),
                hovertemplate="<b>Pressure field</b><br>"
                + "Theta: %{x:.3f} rad<br>"
                + "z: %{y:.3f} m<br>"
                + "Pressure: %{z:.3f} Pa<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="Theta [rad]",
                yaxis_title="z [m]",
                zaxis_title="Pressure [Pa]",
            ),
            title="Pressure field (theta vs z)",
            showlegend=False,
            **kwargs,
        )

        return fig

    def plot_pressure_2d(self, freq_index=0, fig=None, **kwargs):
        """Return a 2-D contour plot of the pressure field (theta vs z).

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
        if not self.pressure_fields:
            raise RuntimeError("No field data available.")

        if fig is None:
            fig = go.Figure()

        P = self.pressure_fields[freq_index]
        theta_grid = self.theta_grids[freq_index]
        z_grid = self.z_grids[freq_index]

        fig.add_trace(
            go.Contour(
                x=theta_grid[0, :],
                y=z_grid[:, 0],
                z=P,
                colorscale="Viridis",
                colorbar=dict(title="Pressure [Pa]"),
                contours=dict(coloring="heatmap"),
                hovertemplate="<b>Pressure field</b><br>"
                + "Theta: %{x:.3f} rad<br>"
                + "z: %{y:.3f} m<br>"
                + "Pressure: %{z:.3f} Pa<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            xaxis_title="Theta [rad]",
            yaxis_title="z [m]",
            title="Pressure field (theta vs z)",
            showlegend=False,
            **kwargs,
        )

        return fig

    def plot_temperature_3d(self, freq_index=0, fig=None, **kwargs):
        """Return a 3-D surface plot of the temperature field (theta vs z).

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
        if not self.temperature_fields:
            raise RuntimeError("No field data available.")

        if fig is None:
            fig = go.Figure()

        T = self.temperature_fields[freq_index]
        theta_grid = self.theta_grids[freq_index]
        z_grid = self.z_grids[freq_index]

        fig.add_trace(
            go.Surface(
                x=theta_grid,
                y=z_grid,
                z=T,
                colorscale="Viridis",
                colorbar=dict(title="Temperature [°C]"),
                hovertemplate="<b>Temperature field</b><br>"
                + "Theta: %{x:.3f} rad<br>"
                + "z: %{y:.3f} m<br>"
                + "Temperature: %{z:.3f} °C<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="Theta [rad]",
                yaxis_title="z [m]",
                zaxis_title="Temperature [°C]",
            ),
            title="Temperature field (theta vs z)",
            showlegend=False,
            **kwargs,
        )

        return fig

    def plot_temperature_2d(self, freq_index=0, fig=None, **kwargs):
        """Return a 2-D contour plot of the temperature field (theta vs z).

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
        if not self.temperature_fields:
            raise RuntimeError("No field data available.")

        if fig is None:
            fig = go.Figure()

        T = self.temperature_fields[freq_index]
        theta_grid = self.theta_grids[freq_index]
        z_grid = self.z_grids[freq_index]

        fig.add_trace(
            go.Contour(
                x=theta_grid[0, :],
                y=z_grid[:, 0],
                z=T,
                colorscale="Viridis",
                colorbar=dict(title="Temperature [°C]"),
                contours=dict(coloring="heatmap"),
                hovertemplate="<b>Temperature field</b><br>"
                + "Theta: %{x:.3f} rad<br>"
                + "z: %{y:.3f} m<br>"
                + "Temperature: %{z:.3f} °C<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            xaxis_title="Theta [rad]",
            yaxis_title="z [m]",
            title="Temperature field (theta vs z)",
            showlegend=False,
            **kwargs,
        )

        return fig

    def plot_bearing_representation(self, fig=None, rotation=90, **kwargs):
        """Return a pie-chart representation of the bearing pad layout with load arrow.

        Parameters
        ----------
        fig : go.Figure, optional
            Existing figure to update.
        rotation : float, optional
            Rotation of the pie chart in degrees.  Default is 90.

        Returns
        -------
        fig : go.Figure
        """
        if fig is None:
            fig = go.Figure()

        groove = (360 / self.n_pad) - self.betha_s_dg
        hG = groove / 2

        pads = [hG, self.betha_s_dg, hG] * self.n_pad
        colors = ["#F5F5DC", "#929591", "#F5F5DC"] * self.n_pad

        fig = go.Figure(data=[go.Pie(values=pads, hole=0.85)])
        fig.update_traces(
            sort=False,
            hoverinfo="label",
            textinfo="none",
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=20)),
            rotation=rotation,
        )

        try:
            fx = float(getattr(self.fxs_load, "m", self.fxs_load))
            fy = float(getattr(self.fys_load, "m", self.fys_load))
        except Exception:
            fx = float(self.fxs_load)
            fy = float(self.fys_load)

        fig.add_annotation(
            x=fx,
            y=fy,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=2.5,
            arrowwidth=3,
            arrowcolor="green",
        )

        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            **kwargs,
        )
        fig.update_xaxes(showline=False)
        fig.update_yaxes(showline=False)
        return fig

    def plot_pressure_distribution(
        self, freq_index=0, axial_element_index=None, fig=None, **kwargs
    ):
        """Return a 2-D radial pressure distribution plot around the bearing.

        Arrows scaled by local pressure magnitude are plotted on the bearing
        circumference, one set of arrows per pad.

        Parameters
        ----------
        freq_index : int, optional
            Frequency index to use for the pressure field.  Default is 0.
        axial_element_index : int, optional
            Axial slice to plot.  Defaults to the middle of the bearing.
        fig : go.Figure, optional
            Existing figure to add traces to.

        Returns
        -------
        fig : go.Figure
        """
        if not self.P_nondim_fields:
            raise RuntimeError("No field data available.")

        if fig is None:
            fig = go.Figure()

        if axial_element_index is None:
            axial_element_index = self.elements_axial // 2

        n_elements = self.elements_circumferential
        P_field = self.P_nondim_fields[freq_index]

        total_points = 1000
        num_points = int(
            self.dtheta * n_elements * total_points / (2 * np.pi)
        )

        thetaI = 0
        theta_p = []
        bearing_plot = []

        for n_p in range(self.n_pad):
            thetaF_pad = self.thetaF[n_p]
            theta_ref = np.sort(
                np.arange(thetaF_pad, thetaI, -self.dtheta)
            )
            theta_p.append((theta_ref[0], theta_ref[-1]))
            thetaI = thetaF_pad

            theta = np.linspace(theta_p[n_p][0], theta_p[n_p][1], num_points)
            x = np.cos(theta)
            y = np.sin(theta)

            bearing_plot.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=tableau_colors["gray"], width=6),
                    hoverinfo="text",
                    text=f"Pad {n_p}",
                    name=f"Pad {n_p} plot",
                )
            )

        P_distribution = P_field[axial_element_index, :, :]
        points = {"x": [], "y": []}
        pressure_plot = []

        for n_p in range(self.n_pad):
            scale = P_distribution[:, n_p] / np.max(np.abs(P_distribution)) * 0.5

            theta = np.arange(
                theta_p[n_p][0] + self.dtheta / 2,
                theta_p[n_p][1],
                self.dtheta,
            )
            x = np.cos(theta)
            y = np.sin(theta)

            for i in range(n_elements):
                x_i = x[i]
                y_i = y[i]
                x_f = x_i + scale[i] * np.cos(theta[i])
                y_f = y_i + scale[i] * np.sin(theta[i])

                angle = theta[i] * 180 / np.pi
                pressure = P_distribution[i, n_p]
                data_info = (
                    f"Pad {n_p}<br>Angle: {angle:.0f} deg<br>"
                    f"Pressure: {pressure:.3e} Pa"
                )
                name = f"Pad {n_p} distribution"

                if abs(np.sqrt(x_f**2 + y_f**2) - 1) > 1e-2:
                    pressure_plot.append(
                        go.Scatter(
                            x=[x_i, x_f],
                            y=[y_i, y_f],
                            mode="lines+markers",
                            line=dict(width=3, color=tableau_colors["orange"]),
                            marker=dict(
                                size=9, symbol="arrow", angleref="previous"
                            ),
                            hoverinfo="text",
                            text=data_info,
                            name=name,
                        )
                    )

                points["x"].append(x_f)
                points["y"].append(y_f)

        points["x"].append(points["x"][0])
        points["y"].append(points["y"][0])

        fig.add_traces(data=[*pressure_plot, *bearing_plot])

        fig.add_trace(
            go.Scatter(
                x=points["x"],
                y=points["y"],
                mode="lines",
                line_shape="spline",
                line=dict(color="black", width=1.5, dash="dash"),
                hoverinfo="none",
                name="Distribution curve",
            )
        )

        P_min = np.min(P_distribution)
        P_max = np.max(P_distribution)
        fig.add_annotation(
            x=1,
            y=1,
            xref="paper",
            yref="paper",
            align="right",
            showarrow=False,
            font=dict(size=16, color="black"),
            text=(
                f"<b>Pressure Distribution</b><br>"
                f"Min: {P_min:.3e} Pa<br>Max: {P_max:.3e} Pa"
            ),
        )

        fig.update_layout(
            title="Plain Journal Bearing",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
            ),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=800,
            showlegend=False,
            **kwargs,
        )

        return fig

    def show_optimization_convergence(
        self, by: str = "value", show_plots: bool = False
    ) -> None:
        """Display the optimization residuals per iteration for each speed.

        Parameters
        ----------
        by : str, optional
            ``"index"`` — sort by frequency array index.
            ``"value"`` — sort by speed value (default).
        show_plots : bool, optional
            When *True* a convergence plot is shown for each speed.
            Default is False.

        Returns
        -------
        None
        """
        if not self.optimization_history:
            print("No residual history available. Run the analysis first.")
            return

        freq_arr = self.frequency.astype(float)

        items = list(self.optimization_history.items())
        if by == "index" and len(freq_arr) > 0:
            items.sort(key=lambda kv: int(np.argmin(np.abs(freq_arr - kv[0]))))
        else:
            items.sort(key=lambda kv: kv[0])

        for speed_key, res_list in items:
            if not res_list:
                continue
            rpm = speed_key * 30.0 / np.pi

            desired_width = 25
            table = PrettyTable()
            table.field_names = ["Iteration", "Residual [N]"]

            for it, res in enumerate(res_list):
                if res is not None:
                    table.add_row([it, f"{res:.4e}"])

            table.max_width = desired_width
            table.min_width = desired_width

            table_str = table.get_string()
            actual_width = len(table_str.split("\n")[0])

            print("\n" + "=" * actual_width)
            print(
                f"OPTIMIZATION CONVERGENCE - {rpm:.1f} RPM".center(
                    actual_width
                ).rstrip()
            )
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
    (``plot_pressure_3d``, ``plot_pressure_2d``, ``plot_temperature_3d``,
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
        table.add_row(
            ["Eccentricity Ratio", f"{self.eccentricity_ratio:12.4f}", "-"]
        )
        table.add_row(
            ["Lubricant Viscosity", f"{self.lubricant_viscosity:12.4e}", "Pa*s"]
        )
        table.add_row(
            ["Damping Coefficient", f"{self.cxx[freq_index]:12.4e}", "N*s/m"]
        )
        table.add_row(
            ["Stiffness Coefficient", f"{self.kxx[freq_index]:12.4e}", "N/m"]
        )
        table.add_row(
            ["Pressure Angle", f"{np.degrees(self.theta[freq_index]):12.2f}", "°"]
        )
        table.add_row(
            ["Pressure Angle", f"{self.theta[freq_index]:12.4f}", "rad"]
        )
        table.add_row(
            ["Maximum Pressure", f"{self.p_max[freq_index]:12.4e}", "Pa"]
        )

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

    def plot_temperature_3d(self, freq_index=0, fig=None, **kwargs):
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
