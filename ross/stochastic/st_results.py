"""STOCHASTIC ROSS plotting module.

This module returns graphs for each type of analyses in st_rotor_assembly.py.
"""
import numpy as np
from plotly import express as px
from plotly import graph_objects as go
from plotly import io as pio
from plotly.subplots import make_subplots

from ross.plotly_theme import tableau_colors

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
        self,
        percentile=[],
        conf_interval=[],
        units="mic-pk-pk",
        **kwargs,
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
        self,
        percentile=[],
        conf_interval=[],
        units="mic-pk-pk",
        **kwargs,
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

    def plot_1d(
        self, probe, percentile=[], conf_interval=[], fig=None, *args, **kwargs
    ):
        """Plot time response.

        This method plots the time response given a tuple of probes with their nodes
        and orientations.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle).
            node : int
                indicate the node where the probe is located.
            orientation : float,
                probe orientation angle about the shaft. The 0 refers to +X direction.
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
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
        if fig is None:
            fig = go.Figure()

        default_values = dict(mode="lines")
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        for i, p in enumerate(probe):
            dofx = p[0] * self.number_dof
            dofy = p[0] * self.number_dof + 1
            angle = p[1]

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )

            probe_resp = np.zeros_like(self.yout[:, :, 0])
            for j, y in enumerate(self.yout):
                _probe_resp = operator @ np.vstack((y[:, dofx], y[:, dofy]))
                probe_resp[j] = (
                    _probe_resp[0] * np.cos(angle) ** 2 +
                    _probe_resp[1] * np.sin(angle) ** 2
                )
            # fmt: on

            fig.add_trace(
                go.Scatter(
                    x=self.time_range,
                    y=np.mean(probe_resp, axis=0),
                    opacity=1.0,
                    name=f"Probe {i + 1} - Mean",
                    line=dict(width=3.0),
                    hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                    **kwargs,
                )
            )
            for j, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatter(
                        x=self.time_range,
                        y=np.percentile(probe_resp, p, axis=0),
                        opacity=0.6,
                        line=dict(width=2.5),
                        name=f"Probe {i + 1} - percentile: {p}%",
                        hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                        **kwargs,
                    )
                )

            x = np.concatenate((self.time_range, self.time_range[::-1]))
            for j, p in enumerate(conf_interval):
                p1 = np.percentile(probe_resp, 50 + p / 2, axis=0)
                p2 = np.percentile(probe_resp, 50 - p / 2, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.concatenate((p1, p2[::-1])),
                        line=dict(width=1),
                        fill="toself",
                        fillcolor=colors1[j],
                        opacity=0.5,
                        name=f"Probe {i + 1} - confidence interval: {p}%",
                        hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                        **kwargs,
                    )
                )

        fig.update_xaxes(title_text="<b>Time (s)</b>")
        fig.update_yaxes(title_text="<b>Amplitude</b>")

        return fig

    def plot_2d(self, node, percentile=[], conf_interval=[], fig=None, *args, **kwargs):
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
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
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

        if fig is None:
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

        fig.update_xaxes(title_text="<b>Amplitude</b>")
        fig.update_yaxes(title_text="<b>Amplitude</b>")
        fig.update_layout(title="<b>Rotor Orbit: node {}</b>".format(node)),

        return fig

    def plot_3d(self, percentile=[], conf_interval=[], fig=None, *args, **kwargs):
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
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
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

        if fig is None:
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
            scene=dict(
                xaxis=dict(title=dict(text="<b>Rotor Length</b>"), showspikes=False),
                yaxis=dict(title=dict(text="<b>Amplitude - X</b>"), showspikes=False),
                zaxis=dict(title=dict(text="<b>Amplitude - Y</b>"), showspikes=False),
            ),
        )
        return fig


class ST_ForcedResponseResults:
    """Store stochastic results and provide plots for Forced Response.

    Parameters
    ----------
    force_resp : array
        Array with the force response for each node for each frequency.
    frequency_range : array
        Array with the frequencies.
    magnitude : array
        Magnitude of the frequency response for node for each frequency.
    phase : array
        Phase of the frequency response for node for each frequency.
    number_dof = int
        Number of degrees of freedom per shaft element's node.

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with amplitude vs frequency phase angle vs frequency.
    """

    def __init__(self, forced_resp, magnitude, phase, frequency_range, number_dof):
        self.forced_resp = forced_resp
        self.magnitude = magnitude
        self.phase = phase
        self.frequency_range = frequency_range
        self.number_dof = number_dof

    def plot_magnitude(
        self,
        probe,
        percentile=[],
        conf_interval=[],
        fig=None,
        units="mic-pk-pk",
        **kwargs,
    ):
        """Plot frequency response.

        This method plots the unbalance response magnitude.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle).
            node : int
                indicate the node where the probe is located.
            orientation : float,
                probe orientation angle about the shaft. The 0 refers to +X direction.
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0% and 100% inclusive.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        units : str, optional
            Unit system
            Default is "mic-pk-pk".
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout
            (e.g. width=800, height=600, ...).
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

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        if fig is None:
            fig = go.Figure()

        color_i = 0
        color_p = 0
        for i, p in enumerate(probe):
            dofx = p[0] * self.number_dof
            dofy = p[0] * self.number_dof + 1
            angle = p[1]

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )

            probe_resp = np.zeros_like(self.magnitude[:, :, 0])
            for j, mag in enumerate(self.magnitude):
                _probe_resp = operator @ np.vstack((mag[:, dofx], mag[:, dofy]))
                probe_resp[i] = np.sqrt((_probe_resp[0] * np.cos(angle)) ** 2 +
                                        (_probe_resp[1] * np.sin(angle)) ** 2)
            # fmt: on

            fig.add_trace(
                go.Scatter(
                    x=self.frequency_range,
                    y=np.mean(probe_resp, axis=0),
                    opacity=1.0,
                    mode="lines",
                    line=dict(width=3, color=list(tableau_colors)[i]),
                    name=f"Probe {i + 1} - Mean",
                    legendgroup=f"Probe {i + 1} - Mean",
                    hovertemplate="Frequency: %{x:.2f}<br>Amplitude: %{y:.2e}",
                )
            )
            for j, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatter(
                        x=self.frequency_range,
                        y=np.percentile(probe_resp, p, axis=0),
                        opacity=0.6,
                        mode="lines",
                        line=dict(width=2.5, color=colors1[color_p]),
                        name=f"Probe {i + 1} - percentile: {p}%",
                        legendgroup=f"Probe {i + 1} - percentile: {p}%",
                        hovertemplate="Frequency: %{x:.2f}<br>Amplitude: %{y:.2e}",
                    )
                )
                color_p += 1

            x = np.concatenate((self.frequency_range, self.frequency_range[::-1]))
            for j, p in enumerate(conf_interval):
                p1 = np.percentile(probe_resp, 50 + p / 2, axis=0)
                p2 = np.percentile(probe_resp, 50 - p / 2, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.concatenate((p1, p2[::-1])),
                        mode="lines",
                        line=dict(width=1, color=colors2[color_i]),
                        fill="toself",
                        fillcolor=colors2[color_i],
                        opacity=0.5,
                        name=f"Probe {i + 1} - confidence interval: {p}%",
                        legendgroup=f"Probe {i + 1} - confidence interval: {p}%",
                        hovertemplate="Frequency: %{x:.2f}<br>Amplitude: %{y:.2e}",
                    )
                )
                color_i += 1

        fig.update_xaxes(title_text="<b>Frequency</b>")
        fig.update_yaxes(title_text=y_axis_label)
        fig.update_layout(**kwargs)

        return fig

    def plot_phase(self, probe, percentile=[], conf_interval=[], fig=None, **kwargs):
        """Plot frequency response.

        This method plots the phase response given a set of probes.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle).
            node : int
                indicate the node where the probe is located.
            orientation : float,
                probe orientation angle about the shaft. The 0 refers to +X direction.
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout
            (e.g. width=800, height=600, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        if fig is None:
            fig = go.Figure()

        color_p = 0
        color_i = 0
        for i, p in enumerate(probe):
            probe_phase = np.zeros_like(self.phase[:, :, 0])
            for j, phs in enumerate(self.phase):
                aux_phase = phs[:, p[0] * self.number_dof]
                probe_phase[i] = np.array(
                    [i + 2 * np.pi if i < 0 else i for i in aux_phase]
                )
                angle = p[1]
                probe_phase[i] = probe_phase[i] - angle

            fig.add_trace(
                go.Scatter(
                    x=self.frequency_range,
                    y=np.mean(probe_phase, axis=0),
                    opacity=1.0,
                    mode="lines",
                    line=dict(width=3, color=list(tableau_colors)[i]),
                    name=f"Probe {i + 1} - Mean",
                    legendgroup=f"Probe {i + 1} - Mean",
                    hovertemplate="Frequency: %{x:.2f}<br>Phase: %{y:.2f}",
                )
            )
            for j, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatter(
                        x=self.frequency_range,
                        y=np.percentile(probe_phase, p, axis=0),
                        opacity=0.6,
                        mode="lines",
                        line=dict(width=2.5, color=colors1[color_p]),
                        name=f"Probe {i + 1} - percentile: {p}%",
                        legendgroup=f"Probe {i + 1} - percentile: {p}%",
                        hovertemplate="Frequency: %{x:.2f}<br>Phase: %{y:.2f}",
                    )
                )
                color_p += 1

            x = np.concatenate((self.frequency_range, self.frequency_range[::-1]))
            for j, p in enumerate(conf_interval):
                p1 = np.percentile(probe_phase, 50 + p / 2, axis=0)
                p2 = np.percentile(probe_phase, 50 - p / 2, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.concatenate((p1, p2[::-1])),
                        mode="lines",
                        line=dict(width=1, color=colors2[color_i]),
                        fill="toself",
                        fillcolor=colors2[color_i],
                        opacity=0.5,
                        name=f"Probe {i + 1} - confidence interval: {p}%",
                        legendgroup=f"Probe {i + 1} - confidence interval: {p}%",
                        hovertemplate="Frequency: %{x:.2f}<br>Phase: %{y:.2f}",
                    )
                )
                color_i += 1

        fig.update_xaxes(title_text="<b>Frequency</b>")
        fig.update_yaxes(title_text="<b>Phase Angle</b>")
        fig.update_layout(**kwargs),

        return fig

    def plot_polar_bode(
        self,
        probe,
        percentile=[],
        conf_interval=[],
        fig=None,
        units="mic-pk-pk",
        **kwargs,
    ):
        """Plot polar forced response using Plotly.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle).
            node : int
                indicate the node where the probe is located.
            orientation : float,
                probe orientation angle about the shaft. The 0 refers to +X direction.
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
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
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        if units == "m":
            r_axis_label = "<b>Amplitude (m)</b>"
        elif units == "mic-pk-pk":
            r_axis_label = "<b>Amplitude (μ pk-pk)</b>"
        else:
            r_axis_label = "<b>Amplitude (dB)</b>"

        if fig is None:
            fig = go.Figure()

        color_p = 0
        color_i = 0
        for i, p in enumerate(probe):
            dofx = p[0] * self.number_dof
            dofy = p[0] * self.number_dof + 1
            angle = p[1]

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )

            probe_resp = np.zeros_like(self.magnitude[:, :, 0])
            for j, mag in enumerate(self.magnitude):
                _probe_resp = operator @ np.vstack((mag[:, dofx], mag[:, dofy]))
                probe_resp[i] = np.sqrt((_probe_resp[0] * np.cos(angle)) ** 2 +
                                        (_probe_resp[1] * np.sin(angle)) ** 2)
            # fmt: on

            probe_phase = np.zeros_like(self.phase[:, :, 0])
            for j, phs in enumerate(self.phase):
                aux_phase = phs[:, p[0] * self.number_dof]
                probe_phase[i] = np.array(
                    [i + 2 * np.pi if i < 0 else i for i in aux_phase]
                )
                angle = p[1]
                probe_phase[i] = probe_phase[i] - angle

            fig.add_trace(
                go.Scatterpolar(
                    r=np.mean(probe_resp, axis=0),
                    theta=np.mean(probe_phase, axis=0),
                    customdata=self.frequency_range,
                    thetaunit="radians",
                    mode="lines",
                    line=dict(width=3.0, color=list(tableau_colors)[i]),
                    name=f"Probe {i + 1} - Mean",
                    legendgroup=f"Probe {i + 1} - Mean",
                    hovertemplate=(
                        "<b>Amplitude: %{r:.2e}</b><br>"
                        + "<b>Phase: %{theta:.2f}</b><br>"
                        + "<b>Frequency: %{customdata:.2f}</b>"
                    ),
                )
            )
            for j, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatterpolar(
                        r=np.percentile(probe_resp, p, axis=0),
                        theta=np.percentile(probe_phase, p, axis=0),
                        customdata=self.frequency_range,
                        thetaunit="radians",
                        opacity=0.6,
                        line=dict(width=2.5, color=colors1[color_p]),
                        name=f"Probe {i + 1} - percentile: {p}%",
                        legendgroup=f"Probe {i + 1} - percentile{p}",
                        hovertemplate=(
                            "<b>Amplitude: %{r:.2e}</b><br>"
                            + "<b>Phase: %{theta:.2f}</b><br>"
                            + "<b>Frequency: %{customdata:.2f}</b>"
                        ),
                    )
                )
                color_p += 1

            for j, p in enumerate(conf_interval):
                p1 = np.percentile(probe_resp, 50 + p / 2, axis=0)
                p2 = np.percentile(probe_resp, 50 - p / 2, axis=0)
                p3 = np.percentile(probe_phase, 50 + p / 2, axis=0)
                p4 = np.percentile(probe_phase, 50 - p / 2, axis=0)
                fig.add_trace(
                    go.Scatterpolar(
                        r=np.concatenate((p1, p2[::-1])),
                        theta=np.concatenate((p3, p4[::-1])),
                        thetaunit="radians",
                        line=dict(width=1, color=colors2[color_i]),
                        fill="toself",
                        fillcolor=colors2[color_i],
                        opacity=0.5,
                        name=f"Probe {i + 1} - confidence interval: {p}%",
                        legendgroup=f"Probe {i + 1} - confidence interval: {p}%",
                    )
                )
                color_i += 1

        fig.update_layout(
            polar=dict(
                radialaxis=dict(title_text=r_axis_label, exponentformat="E"),
                angularaxis=dict(exponentformat="E"),
            ),
            **kwargs,
        )

        return fig

    def plot(
        self,
        probe,
        percentile=[],
        conf_interval=[],
        fig=None,
        units="mic-pk-pk",
        **kwargs,
    ):
        """Plot frequency response.

        This method plots the frequency and phase response given a set of probes.

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
        kwargs : optional
            Additional key word arguments can be passed to change the plot
            (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0, ...)
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            Plotly figure with amplitude vs frequency phase angle vs frequency.
        """
        # fmt: off
        fig0 = self.plot_magnitude(probe, percentile, conf_interval, units=units, **kwargs)
        fig1 = self.plot_phase(probe, percentile, conf_interval, **kwargs)
        fig2 = self.plot_polar_bode(probe, percentile, conf_interval, units=units, **kwargs)

        if fig is None:
            fig = make_subplots(
                rows=2, cols=2, specs=[[{}, {"type": "polar", "rowspan": 2}], [{}, None]]
            )
        # fmt: on

        for data in fig0["data"]:
            data.showlegend = False
            fig.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            data.showlegend = False
            fig.add_trace(data, row=2, col=1)
        for data in fig2["data"]:
            fig.add_trace(data, row=1, col=2)

        fig.update_xaxes(fig0.layout.xaxis, row=1, col=1)
        fig.update_yaxes(fig0.layout.yaxis, row=1, col=1)
        fig.update_xaxes(fig1.layout.xaxis, row=2, col=1)
        fig.update_yaxes(fig1.layout.yaxis, row=2, col=1)
        fig.update_layout(
            polar=dict(
                radialaxis=fig2.layout.polar.radialaxis,
                angularaxis=fig2.layout.polar.angularaxis,
            ),
        )

        return fig
