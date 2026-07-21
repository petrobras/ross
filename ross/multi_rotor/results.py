import numpy as np
from copy import deepcopy as copy
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from ross.units import Q_
from ross.utils import compute_dfft
from ross.results import TimeResponseResults


class BacklashResults(TimeResponseResults):
    def __init__(self, rotor, t, yout, xout):
        super().__init__(rotor, t, yout, xout)

        min_dt = np.diff(self.t).min()

        if min_dt < 1e-4:
            self._step = int(1e-4 / min_dt)
        else:
            self._step = 1

        self.xout = {
            key: np.asarray(copy(value))
            for key, value in rotor.mesh.backlash._data.items()
        }

        self.xout["alfa"] = np.degrees(self.xout["alfa"])

        del rotor.mesh.backlash._data

    def _plot_time_domain(self, key, name, time_range=None, step=None, fig=None):
        """Plots the time domain data of the gear pair.

        Parameters
        ----------
        key : str
            Key name of the data to plot.
        name : str
            Title name of the plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        step : int, optional
            Step size to plot.
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """

        if fig is None:
            fig = go.Figure()

        if step is None:
            step = self._step
        else:
            step = int(step)

        data = self.xout[key]

        if time_range is not None:
            t_min, t_max = time_range
            mask = (self.t >= t_min) & (self.t <= t_max)
            t = self.t[mask]
            data = data[mask]

        fig.add_trace(go.Scattergl(x=t[::step], y=data[::step], name=key))

        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text=name)
        fig.update_layout(title=name)

        return fig

    def _plot_frequency_domain(
        self, key, name, frequency_range=None, step=None, fig=None
    ):
        """Plots the frequency domain data of the gear pair.

        Parameters
        ----------
        key : str
            Key name of the data to plot.
        name : str
            Title name of the plot.
        frequency_range : tuple, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
        step : int, optional
            Step size to plot.
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        if fig is None:
            fig = go.Figure()

        if step is None:
            step = self._step
        else:
            step = int(step)

        data = self.xout[key]

        dt = np.diff(self.t).min()
        freq, amp, _ = compute_dfft(data, dt)

        if frequency_range is not None:
            f_min, f_max = frequency_range
            delta = 0.01 * (f_max - f_min)
            mask = (freq >= f_min - delta) & (freq <= f_max + delta)
            amp = amp[mask]
            freq = freq[mask]

        fig.add_trace(go.Scattergl(x=freq[::step], y=amp[::step], name=f"DFT ({key})"))

        if frequency_range is not None:
            fig.update_xaxes(range=[f_min, f_max])

        fig.update_xaxes(title_text="Frequency (Hz)")
        fig.update_yaxes(title_text="Amplitude (m)")
        fig.update_layout(title=name)

        return fig

    def plot_transmission_error(
        self, domain="time", step=None, time_range=None, frequency_range=None, fig=None
    ):
        """Plots the transmission error.

        Parameters
        ----------
        domain : str, optional
            Domain to plot. Can be "time" or "frequency".
        step : int, optional
            Step size to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        frequency_range : tuple, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "delta"
        name = "Transmission Error"

        if domain == "time":
            fig = self._plot_time_domain(key, name, time_range, step=step, fig=fig)
        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key, name, frequency_range, step=step, fig=fig
            )

        return fig

    def plot_backlash(
        self, domain="time", step=None, time_range=None, frequency_range=None, fig=None
    ):
        """Plot the backlash.

        Parameters
        ----------
        domain : str, optional
            Domain to plot. Can be "time" or "frequency".
        step : int, optional
            Step size to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        frequency_range : tuple, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "bt"
        name = "Backlash"

        if domain == "time":
            fig = self._plot_time_domain(key, name, time_range, step=step, fig=fig)
        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key, name, frequency_range, step=step, fig=fig
            )

        return fig

    def plot_dynamic_mesh_force(
        self, domain="time", step=None, time_range=None, frequency_range=None, fig=None
    ):
        """Plot the dynamic mesh force.

        Parameters
        ----------
        domain : str, optional
            Domain to plot. Can be "time" or "frequency".
        step : int, optional
            Step size to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        frequency_range : tuple, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
        fig : go.Figure
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "Fm"
        name = "Dynamic Mesh Force"

        if domain == "time":
            fig = self._plot_time_domain(key, name, time_range, step=step, fig=fig)
        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key, name, frequency_range, step=step, fig=fig
            )

        return fig

    def plot_mesh_stiffness(
        self, domain="time", step=None, time_range=None, frequency_range=None, fig=None
    ):
        """Plot the mesh stiffness.

        Parameters
        ----------
        domain : str, optional
            Domain to plot. Can be "time" or "frequency".
        step : int, optional
            Step size to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        frequency_range : tuple, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "K_time"
        name = "Mesh Stiffness"

        if domain == "time":
            fig = self._plot_time_domain(key, name, time_range, step=step, fig=fig)
        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key, name, frequency_range, step=step, fig=fig
            )

        return fig

    def plot_center_distance(
        self, domain="time", step=None, time_range=None, frequency_range=None, fig=None
    ):
        """Plot the center distance.

        Parameters
        ----------
        domain : str, optional
            Domain to plot. Can be "time" or "frequency".
        step : int, optional
            Step size to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        frequency_range : tuple, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "d"
        name = "Center Distance"

        if domain == "time":
            fig = self._plot_time_domain(key, name, time_range, step=step, fig=fig)

            Rp1 = self.rotor.mesh.driving_gear.pitch_diameter / 2
            Rp2 = self.rotor.mesh.driven_gear.pitch_diameter / 2

            t = self.t
            d0 = np.full_like(t, Rp1 + Rp2)

            if time_range is not None:
                t_min, t_max = time_range
                mask = (self.t >= t_min) & (self.t <= t_max)
                t = t[mask]
                d0 = d0[mask]

            fig.add_trace(
                go.Scattergl(
                    x=t,
                    y=d0,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    name=f"Nominal {name}",
                )
            )

        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key, name, frequency_range, step=step, fig=fig
            )

        return fig

    def plot_pressure_angle(self, step=None, time_range=None, fig=None):
        """Plot the pressure angle.

        Parameters
        ----------
        step : int, optional
            Step size to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "alfa"
        name = "Pressure Angle"

        fig = self._plot_time_domain(key, name, time_range, step=step, fig=fig)


        t = self.t
        pressure_angle = np.full_like(t, self.rotor.mesh.pressure_angle)
        pressure_angle = np.degrees(pressure_angle)

        if time_range is not None:
            t_min, t_max = time_range
            mask = (t >= t_min) & (t <= t_max)
            t = t[mask]
            pressure_angle = pressure_angle[mask]

        fig.add_trace(
            go.Scattergl(
                x=t,
                y=pressure_angle,
                mode="lines",
                line=dict(color="black", width=2, dash="dot"),
                name=f"Nominal {name}",
            )
        )

        return fig

    def plot_contact_ratio(self, step=None, time_range=None, fig=None):
        """Plot the contact ratio.

        Parameters
        ----------
        step : int, optional
            Step size to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "contact_ratio"
        name = "Contact Ratio"

        fig = self._plot_time_domain(key, name, time_range, step=step, fig=fig)

        return fig

    def plot_dashboard(
        self, step=None, time_range=None, frequency_range=None, fig=None
    ):
        """Plot the dashboard of several results.

        Parameters
        ----------
        step : int, optional
            Number of points to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
        frequency_range : tuple, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """

        subtitles = [
            "Transmission Error (TE) and Backlash (<i>j<sub>t</sub></i>)",
            "DFT Spectrum (TE and <i>j<sub>t</sub></i>)",
            "Dynamic Mesh Force (<i>F<sub>m</sub></i>)",
            "DFT Spectrum (<i>F<sub>m</sub></i>)",
            "Mesh Stiffness (<i>K<sub>m</sub></i>)",
            "DFT Spectrum (<i>K<sub>m</sub></i>)",
            "Center Distance (<i>a</i>)",
            "DFT Spectrum (<i>a</i>)",
            "Pressure Angle (<i>α</i>)",
            "Contact Ratio (CR)"
        ]

        fig = make_subplots(
            rows=5,
            cols=2,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            subplot_titles=subtitles,
        )

        # Transmission Error
        subfig = self.plot_transmission_error(step=step, time_range=time_range)
        for trace in subfig.data:
            fig.add_trace(trace, row=1, col=1)

        subfig = self.plot_transmission_error(
            domain="frequency",
            step=step,
            frequency_range=frequency_range,
        )
        fig.add_trace(subfig.data[0], row=1, col=2)

        # Backlash
        subfig = self.plot_backlash(step=step, time_range=time_range)
        for trace in subfig.data:
            fig.add_trace(trace, row=1, col=1)
        fig.update_yaxes(title_text="Displacement (m)", row=1, col=1)

        subfig = self.plot_backlash(
            domain="frequency", step=step, frequency_range=frequency_range
        )
        fig.add_trace(subfig.data[0], row=1, col=2)
        fig.update_yaxes(title_text="Amplitude (m)", row=1, col=2)

        # Dynamic Mesh Force
        subfig = self.plot_dynamic_mesh_force(step=step, time_range=time_range)
        for trace in subfig.data:
            fig.add_trace(trace, row=2, col=1)
        fig.update_yaxes(title_text="Force (N)", row=2, col=1)

        subfig = self.plot_dynamic_mesh_force(
            domain="frequency", step=step, frequency_range=frequency_range
        )
        fig.add_trace(subfig.data[0], row=2, col=2)
        fig.update_yaxes(title_text="Amplitude (N)", row=2, col=2)

        # Mesh Stiffness
        subfig = self.plot_mesh_stiffness(step=step, time_range=time_range)
        for trace in subfig.data:
            fig.add_trace(trace, row=3, col=1)
        fig.update_yaxes(title_text="Stiffness (N/m)", row=3, col=1)

        subfig = self.plot_mesh_stiffness(
            domain="frequency", step=step, frequency_range=frequency_range
        )
        fig.add_trace(subfig.data[0], row=3, col=2)
        fig.update_yaxes(title_text="Amplitude (N/m)", row=3, col=2)

        # Center Distance
        subfig = self.plot_center_distance(step=step, time_range=time_range)
        for trace in subfig.data:
            fig.add_trace(trace, row=4, col=1)
        fig.update_yaxes(title_text="Distance (m)", row=4, col=1)

        subfig = self.plot_center_distance(
            domain="frequency", step=step, frequency_range=frequency_range
        )
        fig.add_trace(subfig.data[0], row=4, col=2)
        fig.update_yaxes(title_text="Amplitude (m)", row=4, col=2)

        # Pressure Angle
        subfig = self.plot_pressure_angle(step=step, time_range=time_range)
        for trace in subfig.data:
            fig.add_trace(trace, row=5, col=1)
        fig.update_yaxes(title_text="Angle (deg)", row=5, col=1)

        # Contact Ratio
        subfig = self.plot_contact_ratio(step=step, time_range=time_range)
        for trace in subfig.data:
            fig.add_trace(trace, row=5, col=2)
        fig.update_yaxes(title_text="Ratio", row=5, col=2)

        for r in range(1, 5):
            fig.update_xaxes(
                title_text="Frequency (Hz)", range=frequency_range, row=r, col=2
            )

        for r in range(1, 6):
            fig.update_xaxes(title_text="Time (s)", range=time_range, row=r, col=1)
        fig.update_xaxes(title_text="Time (s)", range=time_range, row=5, col=2)

        fig.update_layout(
            title="Gear Pair Dashboard",
            height=1500,
            hovermode="x unified",
            showlegend=False,
        )

        return fig
