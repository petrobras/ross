"""STOCHASTIC ROSS plotting module.

This module returns graphs for each type of analyses in st_rotor_assembly.py.
"""
import numpy as np
from plotly import express as px
from plotly import graph_objects as go
from plotly import io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"

# set Plotly palette of colors
colors1 = px.colors.qualitative.Dark24
colors2 = px.colors.qualitative.Light24


class ST_CampbellResults:
    """Store stochastic results and provide plots for Campbell Diagram.

    It's possible to visualize multiples harmonics in a single plot to check
    other speeds which also excite a specific natural frequency.
    Two options for plooting are available: Matplotlib and Bokeh. The user
    chooses between them using the attribute plot_type. The default is bokeh

    Parameters
    ----------
    speed_range : array
        Array with the speed range in rad/s.
    wd : array
        Array with the damped natural frequencies
    log_dec : array
        Array with the Logarithmic decrement

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with diagrams for frequency and log dec.
    """

    def __init__(self, speed_range, wd, log_dec):
        self.speed_range = speed_range
        self.wd = wd
        self.log_dec = log_dec

    def plot_nat_freq(self, percentile=[], conf_interval=[], harmonics=[1], **kwargs):
        """Plot the damped natural frequencies vs frequency.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()
        x = np.concatenate((self.speed_range, self.speed_range[::-1]))

        for j, h in enumerate(harmonics):
            fig.add_trace(
                go.Scatter(
                    x=self.speed_range,
                    y=self.speed_range * h,
                    opacity=1.0,
                    name="{}x speed".format(h),
                    line=dict(width=3, color=colors1[j], dash="dashdot"),
                    legendgroup="speed{}".format(j),
                    hovertemplate=("Frequency: %{x:.3f}<br>" + "Frequency: %{y:.3f}"),
                    **kwargs,
                )
            )

        for j in range(self.wd.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=self.speed_range,
                    y=np.mean(self.wd[j], axis=1),
                    opacity=1.0,
                    name="Mean - Mode {}".format(j + 1),
                    line=dict(width=3, color=colors1[j]),
                    legendgroup="mean{}".format(j),
                    hovertemplate=("Frequency: %{x:.3f}<br>" + "Frequency: %{y:.3f}"),
                    **kwargs,
                )
            )
            for i, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatter(
                        x=self.speed_range,
                        y=np.percentile(self.wd[j], p, axis=1),
                        opacity=0.6,
                        line=dict(width=2.5, color=colors2[j]),
                        name="percentile: {}%".format(p),
                        legendgroup="percentile{}{}".format(j, i),
                        hovertemplate=(
                            "Frequency: %{x:.3f}<br>" + "Frequency: %{y:.3f}"
                        ),
                        **kwargs,
                    )
                )
            for i, p in enumerate(conf_interval):
                p1 = np.percentile(self.wd[j], 50 + p / 2, axis=1)
                p2 = np.percentile(self.wd[j], 50 - p / 2, axis=1)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.concatenate((p1, p2[::-1])),
                        line=dict(width=1, color=colors1[j]),
                        fill="toself",
                        fillcolor=colors1[j],
                        opacity=0.3,
                        name="confidence interval: {}% - Mode {}".format(p, j + 1),
                        legendgroup="conf{}{}".format(j, i),
                        hovertemplate=(
                            "Frequency: %{x:.3f}<br>" + "Frequency: %{y:.3f}"
                        ),
                        **kwargs,
                    )
                )

        fig.update_xaxes(
            title_text="<b>Rotor speed</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Damped Natural Frequencies</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            width=1200,
            height=900,
            plot_bgcolor="white",
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return fig

    def plot_log_dec(self, percentile=[], conf_interval=[], harmonics=[1], **kwargs):
        """Plot the log_dec vs frequency.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()
        x = np.concatenate((self.speed_range, self.speed_range[::-1]))

        for j in range(self.log_dec.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=self.speed_range,
                    y=np.mean(self.log_dec[j], axis=1),
                    opacity=1.0,
                    name="Mean - Mode {}".format(j + 1),
                    line=dict(width=3, color=colors1[j]),
                    legendgroup="mean{}".format(j),
                    hovertemplate=("Frequency: %{x:.3f}<br>" + "Log Dec: %{y:.3f}"),
                    **kwargs,
                )
            )

            for i, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatter(
                        x=self.speed_range,
                        y=np.percentile(self.log_dec[j], p, axis=1),
                        opacity=0.6,
                        line=dict(width=2.5, color=colors2[j]),
                        name="percentile: {}%".format(p),
                        legendgroup="percentile{}{}".format(j, i),
                        hoverinfo="none",
                        **kwargs,
                    )
                )

            for i, p in enumerate(conf_interval):
                p1 = np.percentile(self.log_dec[j], 50 + p / 2, axis=1)
                p2 = np.percentile(self.log_dec[j], 50 - p / 2, axis=1)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.concatenate((p1, p2[::-1])),
                        line=dict(width=1, color=colors1[j]),
                        fill="toself",
                        fillcolor=colors1[j],
                        opacity=0.3,
                        name="confidence interval: {}% - Mode {}".format(p, j + 1),
                        legendgroup="conf{}{}".format(j, i),
                        hoverinfo="none",
                        **kwargs,
                    )
                )

        fig.update_xaxes(
            title_text="<b>Rotor speed</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Logarithmic decrement</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            plot_bgcolor="white",
            width=1200,
            height=900,
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return fig

    def plot(self, percentile=[], conf_interval=[], *args, **kwargs):
        """Plot Campbell Diagram.

        This method plots Campbell Diagram.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        args: optional
            harmonics : list, optional
                List with the harmonics to be plotted.
                The default is to plot 1x.
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            Plotly figure with diagrams for frequency and log dec.
        """
        fig0 = self.plot_nat_freq(percentile, conf_interval, *args, **kwargs)

        default_values = dict(showlegend=False)
        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig1 = self.plot_log_dec(percentile, conf_interval, *args, **kwargs)

        subplots = make_subplots(rows=1, cols=2)
        for data in fig0["data"]:
            subplots.add_trace(data, 1, 1)
        for data in fig1["data"]:
            subplots.add_trace(data, 1, 2)

        subplots.update_xaxes(fig0.layout.xaxis, row=1, col=1)
        subplots.update_yaxes(fig1.layout.yaxis, row=1, col=1)
        subplots.update_xaxes(fig0.layout.xaxis, row=1, col=2)
        subplots.update_yaxes(fig1.layout.yaxis, row=1, col=2)
        subplots.update_layout(
            plot_bgcolor="white",
            width=1800,
            height=900,
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return subplots


class ST_FrequencyResponseResults:
    """Store stochastic results and provide plots for Frequency Response.

    Parameters
    ----------
    speed_range : array
        Array with the speed range in rad/s.
    magnitude : array
        Array with the frequencies, magnitude (dB) of the frequency
        response for each pair input/output.
    phase : array
        Array with the frequencies, phase of the frequency
        response for each pair input/output.

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with amplitude vs frequency phase angle vs frequency.
    """

    def __init__(self, speed_range, magnitude, phase):
        self.speed_range = speed_range
        self.magnitude = magnitude
        self.phase = phase

    def plot_magnitude(
        self, percentile=[], conf_interval=[], units="mic-pk-pk", **kwargs,
    ):
        """Plot amplitude vs frequency.

        This method plots the frequency response magnitude given an output and
        an input using Plotly.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0% and 100% inclusive.
        units : str, optional
            Unit system
            Default is "mic-pk-pk".
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if units == "m":
            y_axis_label = "<b>Amplitude (m)</b>"
        elif units == "mic-pk-pk":
            y_axis_label = "<b>Amplitude (μ pk-pk)</b>"
        else:
            y_axis_label = "<b>Amplitude (dB)</b>"

        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.speed_range,
                y=np.mean(self.magnitude, axis=1),
                opacity=1.0,
                name="Mean",
                line=dict(width=3, color="black"),
                legendgroup="mean",
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
                **kwargs,
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatter(
                    x=self.speed_range,
                    y=np.percentile(self.magnitude, p, axis=1),
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    legendgroup="percentile{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
                    **kwargs,
                )
            )

        x = np.concatenate((self.speed_range, self.speed_range[::-1]))
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(self.magnitude, 50 + p / 2, axis=1)
            p2 = np.percentile(self.magnitude, 50 - p / 2, axis=1)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.concatenate((p1, p2[::-1])),
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    legendgroup="conf{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
                    **kwargs,
                )
            )

        fig.update_xaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text=y_axis_label,
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            plot_bgcolor="white",
            width=1200,
            height=900,
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return fig

    def plot_phase(self, percentile=[], conf_interval=[], **kwargs):
        """Plot phase angle response.

        This method plots the phase response given an output and an input
        using bokeh.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.speed_range,
                y=np.mean(self.phase, axis=1),
                opacity=1.0,
                name="Mean",
                line=dict(width=3, color="black"),
                legendgroup="mean",
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
                **kwargs,
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatter(
                    x=self.speed_range,
                    y=np.percentile(self.phase, p, axis=1),
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    legendgroup="percentile{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
                    **kwargs,
                )
            )

        x = np.concatenate((self.speed_range, self.speed_range[::-1]))
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(self.phase, 50 + p / 2, axis=1)
            p2 = np.percentile(self.phase, 50 - p / 2, axis=1)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.concatenate((p1, p2[::-1])),
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    legendgroup="conf{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
                    **kwargs,
                )
            )

        fig.update_xaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Phase Angle</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            plot_bgcolor="white",
            width=1200,
            height=900,
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return fig

    def plot_polar_bode(
        self, percentile=[], conf_interval=[], units="mic-pk-pk", **kwargs,
    ):
        """Plot polar forced response using Plotly.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        units : str
            Magnitude unit system.
            Default is "mic-pk-pk"
        polar_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        if units == "m":
            r_axis_label = "<b>Amplitude (m)</b>"
        elif units == "mic-pk-pk":
            r_axis_label = "<b>Amplitude (μ pk-pk)</b>"
        else:
            r_axis_label = "<b>Amplitude (dB)</b>"

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=np.mean(self.magnitude, axis=1),
                theta=np.mean(self.phase, axis=1),
                customdata=self.speed_range,
                thetaunit="radians",
                line=dict(width=3.0, color="black"),
                name="Mean",
                legendgroup="mean",
                hovertemplate=(
                    "<b>Amplitude: %{r:.2e}</b><br>"
                    + "<b>Phase: %{theta:.2f}</b><br>"
                    + "<b>Frequency: %{customdata:.2f}</b>"
                ),
                **kwargs,
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatterpolar(
                    r=np.percentile(self.magnitude, p, axis=1),
                    theta=np.percentile(self.phase, p, axis=1),
                    customdata=self.speed_range,
                    thetaunit="radians",
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    legendgroup="percentile{}".format(i),
                    hovertemplate=(
                        "<b>Amplitude: %{r:.2e}</b><br>"
                        + "<b>Phase: %{theta:.2f}</b><br>"
                        + "<b>Frequency: %{customdata:.2f}</b>"
                    ),
                    **kwargs,
                )
            )
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(self.magnitude, 50 + p / 2, axis=1)
            p2 = np.percentile(self.magnitude, 50 - p / 2, axis=1)
            p3 = np.percentile(self.phase, 50 + p / 2, axis=1)
            p4 = np.percentile(self.phase, 50 - p / 2, axis=1)
            fig.add_trace(
                go.Scatterpolar(
                    r=np.concatenate((p1, p2[::-1])),
                    theta=np.concatenate((p3, p4[::-1])),
                    thetaunit="radians",
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    legendgroup="conf{}".format(i),
                    **kwargs,
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title_text=r_axis_label,
                    title_font=dict(family="Arial", size=14),
                    gridcolor="lightgray",
                    exponentformat="power",
                ),
                angularaxis=dict(
                    tickfont=dict(size=14),
                    gridcolor="lightgray",
                    linecolor="black",
                    linewidth=2.5,
                ),
            ),
        )

        return fig

    def plot(self, percentile=[], conf_interval=[], units="mic-pk-pk", **kwargs):
        """Plot frequency response.

        This method plots the frequency and phase response given an output
        and an input.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        units : str, optional
            Unit system
            Default is "mic-pk-pk"
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            Plotly figure with amplitude vs frequency phase angle vs frequency.
        """
        fig0 = self.plot_magnitude(percentile, conf_interval, units, **kwargs)

        default_values = dict(showlegend=False)
        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig1 = self.plot_phase(percentile, conf_interval, **kwargs)
        fig2 = self.plot_polar_bode(percentile, conf_interval, units, **kwargs)

        subplots = make_subplots(
            rows=2, cols=2, specs=[[{}, {"type": "polar", "rowspan": 2}], [{}, None]]
        )
        for data in fig0["data"]:
            subplots.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            subplots.add_trace(data, row=2, col=1)
        for data in fig2["data"]:
            subplots.add_trace(data, row=1, col=2)

        subplots.update_xaxes(fig0.layout.xaxis, row=1, col=1)
        subplots.update_yaxes(fig0.layout.yaxis, row=1, col=1)
        subplots.update_xaxes(fig1.layout.xaxis, row=2, col=1)
        subplots.update_yaxes(fig1.layout.yaxis, row=2, col=1)
        subplots.update_layout(
            plot_bgcolor="white",
            polar_bgcolor="white",
            width=1800,
            height=900,
            polar=dict(
                radialaxis=fig2.layout.polar.radialaxis,
                angularaxis=fig2.layout.polar.angularaxis,
            ),
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return subplots


class ST_TimeResponseResults:
    """Store stochastic results and provide plots for Time Response and Orbit Response.

    Parameters
    ----------
    time_range : 1-dimensional array
        Time array.
    yout : array
        System response.
    xout : array
        Time evolution of the state vector.
    nodes_list: array
        list with nodes from a rotor model.
    nodes_pos: array
        Rotor nodes axial positions.
    number_dof : int
        Number of degrees of freedom per shaft element's node

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    """

    def __init__(self, time_range, yout, xout, number_dof, nodes_list, nodes_pos):
        self.time_range = time_range
        self.yout = yout
        self.xout = xout
        self.nodes_list = nodes_list
        self.nodes_pos = nodes_pos
        self.number_dof = number_dof

    def _plot_time_response(
        self, dof, percentile=[], conf_interval=[], *args, **kwargs
    ):
        """Plot time response.

        This method plots the time response given.

        Parameters
        ----------
        dof : int
            Degree of freedom that will be observed.
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        args : optional
            Additional plot axes
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if self.number_dof == 4:
            dof_dict = {"0": "x", "1": "y", "2": "α", "3": "β"}

        if self.number_dof == 6:
            dof_dict = {"0": "x", "1": "y", "2": "z", "4": "α", "5": "β", "6": "θ"}

        obs_dof = dof % self.number_dof
        obs_dof = dof_dict[str(obs_dof)]

        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.time_range,
                y=np.mean(self.yout[..., dof], axis=0),
                opacity=1.0,
                name="Mean",
                line=dict(width=3.0, color="black"),
                hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                **kwargs,
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatter(
                    x=self.time_range,
                    y=np.percentile(self.yout[..., dof], p, axis=0),
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                    **kwargs,
                )
            )

        x = np.concatenate((self.time_range, self.time_range[::-1]))
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(self.yout[..., dof], 50 + p / 2, axis=0)
            p2 = np.percentile(self.yout[..., dof], 50 - p / 2, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.concatenate((p1, p2[::-1])),
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                    **kwargs,
                )
            )

        fig.update_xaxes(
            title_text="<b>Time</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Amplitude</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            title="<b>Response for node {} and degree of freedom {}</b>".format(
                dof // 4, obs_dof
            ),
            plot_bgcolor="white",
            width=1200,
            height=900,
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return fig

    def _plot_orbit_2d(self, node, percentile=[], conf_interval=[], *args, **kwargs):
        """Plot orbit response (2D).

        This function plots orbits for a given node on the rotor system in a 2D view.

        Parameters
        ----------
        node : int
            Select the node to display the respective orbit response.
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        args : optional
            Additional plot axes
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        ndof = self.number_dof
        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.mean(self.yout[..., ndof * node], axis=0),
                y=np.mean(self.yout[..., ndof * node + 1], axis=0),
                opacity=1.0,
                name="Mean",
                line=dict(width=3, color="black"),
                hovertemplate=(
                    "X - Amplitude: %{x:.2e}<br>" + "Y - Amplitude: %{y:.2e}"
                ),
                **kwargs,
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatter(
                    x=np.percentile(self.yout[..., ndof * node], p, axis=0),
                    y=np.percentile(self.yout[..., ndof * node + 1], p, axis=0),
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    hovertemplate=(
                        "X - Amplitude: %{x:.2e}<br>" + "Y - Amplitude: %{y:.2e}"
                    ),
                    **kwargs,
                )
            )

        for i, p in enumerate(conf_interval):
            p1 = np.percentile(self.yout[..., ndof * node], 50 + p / 2, axis=0)
            p2 = np.percentile(self.yout[..., ndof * node], 50 - p / 2, axis=0)
            p3 = np.percentile(self.yout[..., ndof * node + 1], 50 + p / 2, axis=0)
            p4 = np.percentile(self.yout[..., ndof * node + 1], 50 - p / 2, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate((p1, p2[::-1])),
                    y=np.concatenate((p3, p4[::-1])),
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    hovertemplate=(
                        "X - Amplitude: %{x:.2e}<br>" + "Y - Amplitude: %{y:.2e}"
                    ),
                    **kwargs,
                )
            )

        fig.update_xaxes(
            title_text="<b>Amplitude</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Amplitude</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            title="<b>Rotor Orbit: node {}</b>".format(node),
            plot_bgcolor="white",
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return fig

    def _plot_orbit_3d(self, percentile=[], conf_interval=[], *args, **kwargs):
        """Plot orbit response (3D).

        This function plots orbits for each node on the rotor system in a 3D view.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        args : optional
            Additional plot axes
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        ndof = self.number_dof
        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        line = np.zeros(len(self.nodes_pos))
        fig.add_trace(
            go.Scatter3d(
                x=self.nodes_pos,
                y=line,
                z=line,
                line=dict(width=2.0, color="black", dash="dashdot"),
                showlegend=False,
                mode="lines",
            )
        )
        for j, n in enumerate(self.nodes_list):
            x = np.ones(self.yout.shape[1]) * self.nodes_pos[n]
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=np.mean(self.yout[..., ndof * n], axis=0),
                    z=np.mean(self.yout[..., ndof * n + 1], axis=0),
                    line=dict(width=5, color="black"),
                    name="Mean",
                    legendgroup="mean",
                    showlegend=True if j == 0 else False,
                    hovertemplate=(
                        "Nodal Position: %{x:.2f}<br>"
                        + "X - Amplitude: %{y:.2e}<br>"
                        + "Y - Amplitude: %{z:.2e}"
                    ),
                    **kwargs,
                )
            )
            for i, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=np.percentile(self.yout[..., ndof * n], p, axis=0),
                        z=np.percentile(self.yout[..., ndof * n + 1], p, axis=0),
                        opacity=1.0,
                        name="percentile: {}%".format(p),
                        line=dict(width=3, color=colors1[i]),
                        legendgroup="perc{}".format(p),
                        showlegend=True if j == 0 else False,
                        hovertemplate=(
                            "Nodal Position: %{x:.2f}<br>"
                            + "X - Amplitude: %{y:.2e}<br>"
                            + "Y - Amplitude: %{z:.2e}"
                        ),
                        **kwargs,
                    )
                )
            for i, p in enumerate(conf_interval):
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=np.percentile(self.yout[..., ndof * n], 50 + p / 2, axis=0),
                        z=np.percentile(
                            self.yout[..., ndof * n + 1], 50 + p / 2, axis=0
                        ),
                        line=dict(width=3.5, color=colors1[i]),
                        opacity=0.6,
                        name="confidence interval: {}%".format(p),
                        legendgroup="conf_interval{}".format(p),
                        showlegend=True if j == 0 else False,
                        hovertemplate=(
                            "Nodal Position: %{x:.2f}<br>"
                            + "X - Amplitude: %{y:.2e}<br>"
                            + "Y - Amplitude: %{z:.2e}"
                        ),
                        **kwargs,
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=np.percentile(self.yout[..., ndof * n], 50 - p / 2, axis=0),
                        z=np.percentile(
                            self.yout[..., ndof * n + 1], 50 - p / 2, axis=0
                        ),
                        line=dict(width=3.5, color=colors1[i]),
                        opacity=0.6,
                        name="confidence interval: {}%".format(p),
                        legendgroup="conf_interval{}".format(p),
                        showlegend=False,
                        hovertemplate=(
                            "Nodal Position: %{x:.2f}<br>"
                            + "X - Amplitude: %{y:.2e}<br>"
                            + "Y - Amplitude: %{z:.2e}"
                        ),
                        **kwargs,
                    )
                )

        fig.update_layout(
            width=1200,
            height=900,
            scene=dict(
                bgcolor="white",
                xaxis=dict(
                    title=dict(text="<b>Rotor Length</b>", font=dict(size=14)),
                    tickfont=dict(size=12),
                    nticks=5,
                    backgroundcolor="lightgray",
                    gridcolor="white",
                    showspikes=False,
                ),
                yaxis=dict(
                    title=dict(text="<b>Amplitude - X</b>", font=dict(size=14)),
                    tickfont=dict(size=12),
                    nticks=5,
                    backgroundcolor="lightgray",
                    gridcolor="white",
                    showspikes=False,
                ),
                zaxis=dict(
                    title=dict(text="<b>Amplitude - Y</b>", font=dict(size=14)),
                    tickfont=dict(size=12),
                    nticks=5,
                    backgroundcolor="lightgray",
                    gridcolor="white",
                    showspikes=False,
                ),
            ),
        )
        return fig

    def plot(
        self,
        plot_type="3d",
        node=None,
        dof=None,
        percentile=[],
        conf_interval=[],
        *args,
        **kwargs,
    ):
        """Plot stochastic time or orbit response.

        This function plots calls the auxiliary methods to plot the stochastic orbit
        response. The plot type options are:
            - 1d: plot time response for a given degree of freedom of a rotor system.
            - 2d: plot orbit of a selected node of a rotor system.
            - 3d: plot orbits for each node on the rotor system in a 3D view.

        If plot_type = "1d": input a dof.
        If plot_type = "2d": input a node.
        if plot_type = "3d": no need to input a dof or node.

        Parameters
        ----------
        plot_type : str, optional
            Defines with type of plot to display.
            Options are: "1d","2d" or "3d"
            Default is "3d".
        node : int, optional
            Select the node to display the respective orbit response.
            Default is None.
        dof : int
            Degree of freedom that will be observed.
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        args : optional
            Additional plot axes
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Raise
        -----
        ValueError
            Error raised if no node is specified when plot_type = '2d'.
        ValueError
            Error raised if an odd string is specified to plot_type

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if plot_type == "1d":
            return self._plot_time_response(
                dof, percentile, conf_interval, *args, **kwargs
            )
        elif plot_type == "2d":
            if node not in self.nodes_list:
                raise ValueError("Please insert a valid node.")
            else:
                return self._plot_orbit_2d(
                    node, percentile, conf_interval, *args, **kwargs
                )
        elif plot_type == "3d":
            return self._plot_orbit_3d(percentile, conf_interval, *args, **kwargs)
        else:
            raise ValueError(
                "plot_type not supported. Choose between '1d', '2d' or '3d'."
            )


class ST_ForcedResponseResults:
    """Store stochastic results and provide plots for Forced Response.

    Parameters
    ----------
    force_resp : array
        Array with the force response for each node for each frequency
    frequency_range : array
        Array with the frequencies
    magnitude : array
        Magnitude of the frequency response for node for each frequency
    phase : array
        Phase of the frequency response for node for each frequency

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with amplitude vs frequency phase angle vs frequency.
    """

    def __init__(self, forced_resp, magnitude, phase, frequency_range):
        self.forced_resp = forced_resp
        self.magnitude = magnitude
        self.phase = phase
        self.frequency_range = frequency_range

    def plot_magnitude(
        self, dof, percentile=[], conf_interval=[], units="mic-pk-pk", **kwargs,
    ):
        """Plot frequency response.

        This method plots the unbalance response magnitude given an  Bokeh.

        Parameters
        ----------
        dof : int
            Degree of freedom to observe the response.
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0% and 100% inclusive.
        units : str, optional
            Unit system
            Default is "mic-pk-pk".
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            Bokeh plot axes with magnitude plot.
        """
        if units == "m":
            y_axis_label = "<b>Amplitude (m)</b>"
        elif units == "mic-pk-pk":
            y_axis_label = "<b>Amplitude (μ pk-pk)</b>"
        else:
            y_axis_label = "<b>Amplitude (dB)</b>"

        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.frequency_range,
                y=np.mean(self.magnitude[..., dof], axis=0),
                opacity=1.0,
                name="Mean",
                line=dict(width=3, color="black"),
                legendgroup="mean",
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
                **kwargs,
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatter(
                    x=self.frequency_range,
                    y=np.percentile(self.magnitude[..., dof], p, axis=0),
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    legendgroup="percentile{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
                    **kwargs,
                )
            )

        x = np.concatenate((self.frequency_range, self.frequency_range[::-1]))
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(self.magnitude[..., dof], 50 + p / 2, axis=0)
            p2 = np.percentile(self.magnitude[..., dof], 50 - p / 2, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.concatenate((p1, p2[::-1])),
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    legendgroup="conf{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
                    **kwargs,
                )
            )

        fig.update_xaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text=y_axis_label,
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            width=1200,
            height=900,
            plot_bgcolor="white",
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return fig

    def plot_phase(self, dof, percentile=[], conf_interval=[], **kwargs):
        """Plot frequency response.

        This method plots the phase response given an output and an input
        using bokeh.

        Parameters
        ----------
        dof : int
            Degree of freedom to observe the response.
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.frequency_range,
                y=np.mean(self.phase[..., dof], axis=0),
                opacity=1.0,
                name="Mean",
                line=dict(width=3, color="black"),
                legendgroup="mean",
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
                **kwargs,
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatter(
                    x=self.frequency_range,
                    y=np.percentile(self.phase[..., dof], p, axis=0),
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    legendgroup="percentile{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
                    **kwargs,
                )
            )

        x = np.concatenate((self.frequency_range, self.frequency_range[::-1]))
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(self.phase[..., dof], 50 + p / 2, axis=0)
            p2 = np.percentile(self.phase[..., dof], 50 - p / 2, axis=0)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.concatenate((p1, p2[::-1])),
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    legendgroup="conf{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
                    **kwargs,
                )
            )

        fig.update_xaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Phase Angle</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            width=1200,
            height=900,
            plot_bgcolor="white",
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return fig

    def plot_polar_bode(
        self, dof, percentile=[], conf_interval=[], units="mic-pk-pk", **kwargs,
    ):
        """Plot polar forced response using Plotly.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        units : str
            Magnitude unit system.
            Default is "mic-pk-pk"
        polar_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        if units == "m":
            r_axis_label = "<b>Amplitude (m)</b>"
        elif units == "mic-pk-pk":
            r_axis_label = "<b>Amplitude (μ pk-pk)</b>"
        else:
            r_axis_label = "<b>Amplitude (dB)</b>"

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=np.mean(self.magnitude[..., dof], axis=1),
                theta=np.mean(self.phase[..., dof], axis=1),
                customdata=self.frequency_range,
                thetaunit="radians",
                line=dict(width=3.0, color="black"),
                name="Mean",
                legendgroup="mean",
                hovertemplate=(
                    "<b>Amplitude: %{r:.2e}</b><br>"
                    + "<b>Phase: %{theta:.2f}</b><br>"
                    + "<b>Frequency: %{customdata:.2f}</b>"
                ),
                **kwargs,
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatterpolar(
                    r=np.percentile(self.magnitude[..., dof], p, axis=1),
                    theta=np.percentile(self.phase[..., dof], p, axis=1),
                    customdata=self.frequency_range,
                    thetaunit="radians",
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    legendgroup="percentile{}".format(i),
                    hovertemplate=(
                        "<b>Amplitude: %{r:.2e}</b><br>"
                        + "<b>Phase: %{theta:.2f}</b><br>"
                        + "<b>Frequency: %{customdata:.2f}</b>"
                    ),
                    **kwargs,
                )
            )
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(self.magnitude[..., dof], 50 + p / 2, axis=1)
            p2 = np.percentile(self.magnitude[..., dof], 50 - p / 2, axis=1)
            p3 = np.percentile(self.phase[..., dof], 50 + p / 2, axis=1)
            p4 = np.percentile(self.phase[..., dof], 50 - p / 2, axis=1)
            fig.add_trace(
                go.Scatterpolar(
                    r=np.concatenate((p1, p2[::-1])),
                    theta=np.concatenate((p3, p4[::-1])),
                    thetaunit="radians",
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    legendgroup="conf{}".format(i),
                    **kwargs,
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title_text=r_axis_label,
                    title_font=dict(family="Arial", size=14),
                    gridcolor="lightgray",
                    exponentformat="power",
                ),
                angularaxis=dict(
                    tickfont=dict(size=14),
                    gridcolor="lightgray",
                    linecolor="black",
                    linewidth=2.5,
                ),
            ),
        )

        return fig

    def plot(
        self, dof, percentile=[], conf_interval=[], units="mic-pk-pk", *args, **kwargs,
    ):
        """Plot frequency response.

        This method plots the frequency and phase response given an output
        and an input.

        Parameters
        ----------
        dof : int
            Degree of freedom to observe the response.
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        units : str, optional
            Unit system
            Default is "mic-pk-pk"
        args : optional
            Additional plot axes
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            Plotly figure with amplitude vs frequency phase angle vs frequency.
        """
        fig0 = self.plot_magnitude(dof, percentile, conf_interval, units, **kwargs)

        default_values = dict(showlegend=False)
        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig1 = self.plot_phase(dof, percentile, conf_interval, **kwargs)
        fig2 = self.plot_polar_bode(dof, percentile, conf_interval, units, **kwargs)

        subplots = make_subplots(
            rows=2, cols=2, specs=[[{}, {"type": "polar", "rowspan": 2}], [{}, None]]
        )
        for data in fig0["data"]:
            subplots.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            subplots.add_trace(data, row=2, col=1)
        for data in fig2["data"]:
            subplots.add_trace(data, row=1, col=2)

        subplots.update_xaxes(fig0.layout.xaxis, row=1, col=1)
        subplots.update_yaxes(fig0.layout.yaxis, row=1, col=1)
        subplots.update_xaxes(fig1.layout.xaxis, row=2, col=1)
        subplots.update_yaxes(fig1.layout.yaxis, row=2, col=1)
        subplots.update_layout(
            plot_bgcolor="white",
            polar_bgcolor="white",
            width=1800,
            height=900,
            polar=dict(
                radialaxis=fig2.layout.polar.radialaxis,
                angularaxis=fig2.layout.polar.angularaxis,
            ),
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
        )

        return subplots
