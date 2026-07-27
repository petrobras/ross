import numpy as np
from copy import deepcopy as copy
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from ross.units import Q_, check_units
from ross.results import TimeResponseResults

from .utils import compute_dfft


class BacklashResults(TimeResponseResults):
    """Class used to store results and provide plots for system Time Response and
    Backlash analysis.

    This class takes the results from time response analysis and creates a
    plots given a force and a time. It's possible to select through a time response for
    a single DoF, an orbit response for a single node or display orbit response for all
    nodes.
    The plot type options are:
        - 1d: plot time response for given probes.
        - 2d: plot orbit of a selected node of a rotor system.
        - 3d: plot orbits for each node on the rotor system in a 3D view.
        - dfft: plot response in frequency domain for given probes.
        - transmission_error: plot time and frequency response of the transmission error.
        - backlash: plot time and frequency response of the backlash.
        - dynamic_mesh_force: plot time and frequency response of the dynamic mesh force.
        - mesh_stiffness: plot time and frequency response of the mesh stiffness.
        - center_distance: plot time and frequency response of the center distance.
        - pressure_angle: plot time and frequency response of the pressure angle.
        - contact_ratio: plot time and frequency response of the contact ratio.

    Parameters
    ----------
    rotor : Rotor.object
        The Rotor object
    t : array
        Time values for the output.
    yout : array
        System response.
    xout : array
        Time evolution of the state vector.

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    """

    PARAMS = {
        "transmission_error": {
            "name": "Dynamic Transmission Error",
            "measure": "Displacement",
            "units": "m",
            "repr": "DTE",
        },
        "backlash": {
            "name": "Backlash",
            "measure": "Displacement",
            "units": "m",
            "repr": "<i>b<sub>t</sub></i>",
        },
        "mesh_force": {
            "name": "Dynamic Mesh Force",
            "measure": "Force",
            "units": "N",
            "repr": "<i>F<sub>m</sub></i>",
        },
        "mesh_stiffness": {
            "name": "Mesh Stiffness",
            "measure": "Stiffness",
            "units": "N/m",
            "repr": "<i>k<sub>m</sub></i>",
        },
        "center_distance": {
            "name": "Center Distance",
            "measure": "Distance",
            "units": "m",
            "repr": "<i>d</i>",
        },
        "pressure_angle": {
            "name": "Pressure Angle",
            "measure": "Angle",
            "units": "rad",
            "repr": "<i>α</i>",
        },
        "contact_ratio": {
            "name": "Contact Ratio",
            "measure": "Ratio",
            "units": "--",
            "repr": "CR",
        },
    }

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

        del rotor.mesh.backlash._data

    def _get_params(self, key):
        """Gets the parameters for the given key.

        Parameters
        ----------
        key : str
            Key name of the data to get the parameters.

        Returns
        -------
        name : str
            Name of the data.
        measure : str
            Measure of the data.
        units : str
            Units of the data.
        repr_ : str
            Representation of the data.
        """
        name = self.PARAMS[key]["name"]
        repr_ = self.PARAMS[key]["repr"]
        measure = self.PARAMS[key]["measure"]
        units = self.PARAMS[key]["units"]

        return name, repr_, measure, units

    def _plot_time_domain(self, key, time_range=None, step=None, units=None, fig=None):
        """Plots the time domain data of the gear pair.

        Parameters
        ----------
        key : str
            Key name of the data to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        step : int, optional
            Step size to plot.
        units : str, optional
            Units of the data.
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        name, repr_, measure, units_ = self._get_params(key)

        if fig is None:
            fig = go.Figure()

        if step is None:
            step = self._step
        else:
            step = int(step)

        data = self.xout[key]
        t = self.t

        if units is None:
            units = units_
        else:
            data = Q_(data, units_).to(units).m

        if time_range is not None:
            t_min, t_max = time_range
            mask = (t >= t_min) & (t <= t_max)
            t = t[mask]
            data = data[mask]

        fig.add_trace(go.Scattergl(x=t[::step], y=data[::step], name=repr_))

        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text=f"{measure} ({units})")
        fig.update_layout(title=f"{name} ({repr_})")

        return fig

    def _plot_nominal_value(
        self, key, nominal_value, time_range=None, units=None, fig=None
    ):
        """Plots the nominal value of the data.

        Parameters
        ----------
        key : str
            Key name of the data to plot.
        nominal_value : float
            Nominal value of the data.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        units : str, optional
            Units of the data.
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        name, repr_, measure, units_ = self._get_params(key)

        if units is None:
            units = units_
        else:
            nominal_value = Q_(nominal_value, units_).to(units).m

        if fig is None:
            fig = go.Figure()

        if time_range is not None:
            t_min, t_max = time_range
        else:
            t_min, t_max = self.t[0], self.t[-1]

        t = np.linspace(t_min, t_max, 50)
        data = np.full_like(t, nominal_value)

        fig.add_trace(
            go.Scattergl(
                x=t,
                y=data,
                mode="lines",
                line=dict(color="black", width=2, dash="dot"),
                name=f"{repr_}<sub>nom</sub>",
            )
        )

        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text=f"{measure} ({units})")
        fig.update_layout(title=f"Nominal {name} ({repr_}<sub>nom</sub>)")

        return fig

    def _plot_frequency_domain(
        self,
        key,
        frequency_range=None,
        frequency_units="Hz",
        step=None,
        units=None,
        fig=None,
    ):
        """Plots the frequency domain data of the gear pair.

        Parameters
        ----------
        key : str
            Key name of the data to plot.
        frequency_range : tuple, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
        frequency_units : str, optional
            Units of the frequencies.
        step : int, optional
            Step size to plot.
        units : str, optional
            Units of the data.
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        name, repr_, _, units_ = self._get_params(key)

        if fig is None:
            fig = go.Figure()

        if step is None:
            step = self._step
        else:
            step = int(step)

        data = self.xout[key]

        if units is None:
            units = units_
        else:
            data = Q_(data, units_).to(units).m

        dt = np.diff(self.t).min()
        freq, amp = compute_dfft(data, dt)

        if frequency_range is not None:
            f_min, f_max = frequency_range
            delta = 0.01 * (f_max - f_min)
            mask = (freq >= f_min - delta) & (freq <= f_max + delta)
            amp = amp[mask]
            freq = freq[mask]

        freq = Q_(freq, "Hz").to(frequency_units).m

        fig.add_trace(
            go.Scattergl(x=freq[::step], y=amp[::step], name=f"DFT ({repr_})")
        )

        if frequency_range is not None:
            fig.update_xaxes(range=[f_min, f_max])

        fig.update_xaxes(title_text=f"Frequency ({frequency_units})")
        fig.update_yaxes(title_text=f"Amplitude ({units})")
        fig.update_layout(title=f"DFT Spectrum ({repr_})")

        return fig

    @check_units
    def plot_transmission_error(
        self,
        domain="time",
        step=None,
        time_range=None,
        frequency_range=None,
        frequency_units="Hz",
        data_units="m",
        fig=None,
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
        frequency_units : str, optional
            Units of the frequencies for the frequency domain plot.
        data_units : str, optional
            Units of the transmission error data.
            Default is "m".
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "transmission_error"

        if domain == "time":
            fig = self._plot_time_domain(
                key, time_range, step=step, units=data_units, fig=fig
            )
        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key,
                frequency_range,
                frequency_units=frequency_units,
                step=step,
                units=data_units,
                fig=fig,
            )

        return fig

    @check_units
    def plot_backlash(
        self,
        domain="time",
        step=None,
        time_range=None,
        frequency_range=None,
        frequency_units="Hz",
        data_units="m",
        fig=None,
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
        frequency_units : str, optional
            Units of the frequencies for the frequency domain plot.
        data_units : str, optional
            Units of the backlash data.
            Default is "m".
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "backlash"

        if domain == "time":
            fig = self._plot_time_domain(
                key, time_range, step=step, units=data_units, fig=fig
            )
        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key,
                frequency_range,
                frequency_units=frequency_units,
                step=step,
                units=data_units,
                fig=fig,
            )

        return fig

    @check_units
    def plot_dynamic_mesh_force(
        self,
        domain="time",
        step=None,
        time_range=None,
        frequency_range=None,
        frequency_units="Hz",
        data_units="N",
        fig=None,
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
        frequency_units : str, optional
            Units of the frequencies for the frequency domain plot.
        data_units : str, optional
            Units of the dynamic mesh force data.
            Default is "N".
        fig : go.Figure
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "mesh_force"

        if domain == "time":
            fig = self._plot_time_domain(
                key, time_range, step=step, units=data_units, fig=fig
            )
        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key,
                frequency_range,
                frequency_units=frequency_units,
                step=step,
                units=data_units,
                fig=fig,
            )

        return fig

    @check_units
    def plot_mesh_stiffness(
        self,
        domain="time",
        step=None,
        time_range=None,
        frequency_range=None,
        frequency_units="Hz",
        data_units="N/m",
        fig=None,
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
        frequency_units : str, optional
            Units of the frequencies for the frequency domain plot.
        data_units : str, optional
            Units of the mesh stiffness data.
            Default is "N/m".
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "mesh_stiffness"

        if domain == "time":
            fig = self._plot_time_domain(
                key, time_range, step=step, units=data_units, fig=fig
            )
        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key,
                frequency_range,
                frequency_units=frequency_units,
                step=step,
                units=data_units,
                fig=fig,
            )

        return fig

    @check_units
    def plot_center_distance(
        self,
        domain="time",
        step=None,
        time_range=None,
        frequency_range=None,
        frequency_units="Hz",
        data_units="m",
        fig=None,
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
        frequency_units : str, optional
            Units of the frequencies for the frequency domain plot.
        data_units : str, optional
            Units of the center distance data.
            Default is "m".
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "center_distance"

        if domain == "time":
            Rp1 = self.rotor.mesh.driving_gear.pitch_diameter / 2
            Rp2 = self.rotor.mesh.driven_gear.pitch_diameter / 2
            a0 = Rp1 + Rp2
            fig = self._plot_nominal_value(
                key, a0, time_range, units=data_units, fig=fig
            )

            fig = self._plot_time_domain(
                key, time_range, step=step, units=data_units, fig=fig
            )

        elif domain == "frequency":
            fig = self._plot_frequency_domain(
                key,
                frequency_range,
                frequency_units=frequency_units,
                step=step,
                units=data_units,
                fig=fig,
            )

        return fig

    def plot_pressure_angle(
        self, step=None, time_range=None, data_units="deg", fig=None
    ):
        """Plot the pressure angle.

        Parameters
        ----------
        step : int, optional
            Step size to plot.
        time_range : tuple, optional
            Tuple with (min, max) values for the time that will be plotted.
            Time that are not within the range are filtered out and are not plotted.
        data_units : str, optional
            Units of the pressure angle data.
            Default is "deg".
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        key = "pressure_angle"

        alpha0 = self.rotor.mesh.pressure_angle
        fig = self._plot_nominal_value(
            key, alpha0, time_range, units=data_units, fig=fig
        )

        fig = self._plot_time_domain(
            key, time_range, step=step, units=data_units, fig=fig
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

        cr0 = self.rotor.mesh.contact_ratio
        fig = self._plot_nominal_value(key, cr0, time_range, fig=fig)

        fig = self._plot_time_domain(key, time_range, step=step, fig=fig)

        return fig

    def plot_dashboard(
        self,
        step=None,
        time_range=None,
        frequency_range=None,
        frequency_units="Hz",
        fig=None,
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
        frequency_units : str, optional
            Units of the frequencies for the frequency domain plots.
        fig : go.Figure, optional
            Figure object.

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        rows, cols = 5, 2

        subfigs = [[None for j in range(cols)] for i in range(rows)]

        # Transmission Error
        subfigs[0][0] = self.plot_transmission_error(step=step, time_range=time_range)
        text1 = subfigs[0][0].layout.title.text

        subfigs[0][1] = self.plot_transmission_error(
            domain="frequency",
            step=step,
            frequency_range=frequency_range,
            frequency_units=frequency_units,
        )
        text2 = subfigs[0][1].layout.title.text

        # Backlash
        subfigs[0][0] = self.plot_backlash(
            step=step, time_range=time_range, fig=subfigs[0][0]
        )
        text1 += " and " + subfigs[0][0].layout.title.text
        subfigs[0][0].layout.title.text = text1

        subfigs[0][1] = self.plot_backlash(
            domain="frequency",
            step=step,
            frequency_range=frequency_range,
            frequency_units=frequency_units,
            fig=subfigs[0][1],
        )
        text2 = text2.split(")")[0] + " and "
        text2 += subfigs[0][1].layout.title.text.split("(")[1]
        subfigs[0][1].layout.title.text = text2

        # Dynamic Mesh Force
        subfigs[1][0] = self.plot_dynamic_mesh_force(step=step, time_range=time_range)
        subfigs[1][1] = self.plot_dynamic_mesh_force(
            domain="frequency",
            step=step,
            frequency_range=frequency_range,
            frequency_units=frequency_units,
        )

        # Mesh Stiffness
        subfigs[2][0] = self.plot_mesh_stiffness(step=step, time_range=time_range)
        subfigs[2][1] = self.plot_mesh_stiffness(
            domain="frequency",
            step=step,
            frequency_range=frequency_range,
            frequency_units=frequency_units,
        )

        # Center Distance
        subfigs[3][0] = self.plot_center_distance(step=step, time_range=time_range)
        subfigs[3][1] = self.plot_center_distance(
            domain="frequency",
            step=step,
            frequency_range=frequency_range,
            frequency_units=frequency_units,
        )

        # Pressure Angle
        subfigs[4][0] = self.plot_pressure_angle(step=step, time_range=time_range)

        # Contact Ratio
        subfigs[4][1] = self.plot_contact_ratio(step=step, time_range=time_range)

        subtitles = list()
        for i in range(rows):
            for j in range(cols):
                subfig = subfigs[i][j]
                subtitles.append(subfig.layout.title.text)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            subplot_titles=subtitles,
        )

        for i in range(rows):
            for j in range(cols):
                subfig = subfigs[i][j]
                row, col = i + 1, j + 1

                for trace in subfig.data:
                    fig.add_trace(trace, row=row, col=col)

                yaxis = subfig.layout.yaxis.title.text
                xaxis = subfig.layout.xaxis.title.text

                fig.update_yaxes(title_text=yaxis, row=row, col=col)

                x_range = time_range if "Time" in xaxis else frequency_range
                fig.update_xaxes(title_text=xaxis, range=x_range, row=row, col=col)

        fig.update_layout(
            title="Mesh Dynamics | Results Dashboard",
            height=1500,
            hovermode="closest",
            showlegend=False,
        )

        return fig
