"""STOCHASTIC ROSS plotting module.

This module returns graphs for each type of analyses in st_rotor_assembly.py.
"""
import bokeh.palettes as bp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure

# set bokeh palette of colors
colors1 = bp.Category10[10]
colors2 = bp.Category20c[20]


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
    ax : matplotlib axes
        Returns the matplotlib axes object with the plot
        if plot_type == "matplotlib"
    bk_ax : bokeh axes
        Returns the bokeh axes object with the plot
        if plot_type == "bokeh"
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
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--').

        Returns
        -------
        fig : Bokeh figure
            The bokeh axes object with the plot.
        """
        default_values = dict(line_width=3.0, line_alpha=1.0)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = figure(
            height=900,
            width=900,
            tools="pan, box_zoom, wheel_zoom, reset, save",
            title="Campbell Diagram",
            y_axis_label="Damped Natural Frequencies",
            x_axis_label="Rotor Speed",
        )
        fig.xaxis.axis_label_text_font_size = "14pt"
        fig.yaxis.axis_label_text_font_size = "14pt"
        fig.axis.major_label_text_font_size = "14pt"
        fig.title.text_font_size = "14pt"

        for j in range(self.wd.shape[0]):
            fig.line(
                x=self.speed_range,
                y=np.mean(self.wd[j], axis=1),
                line_color=colors1[j],
                line_alpha=1.0,
                line_width=3.0,
                muted_color=colors1[j],
                muted_alpha=0.1,
                legend_label="Mean - Mode {}".format(j + 1),
            )
            if len(percentile):
                for i, p in enumerate(percentile):
                    fig.line(
                        x=self.speed_range,
                        y=np.percentile(self.wd[j], p, axis=1),
                        line_color=colors2[i],
                        line_alpha=0.6,
                        line_width=2.5,
                        muted_color=colors2[i],
                        muted_alpha=0.1,
                        legend_label="percentile: {}%".format(p),
                    )
            if len(conf_interval):
                for i, p in enumerate(conf_interval):
                    fig.line(
                        x=self.speed_range,
                        y=np.percentile(self.wd[j], 50 + p / 2, axis=1),
                        line_color=colors1[j],
                        line_alpha=0.6,
                        line_width=2.5,
                        muted_color=colors1[j],
                        muted_alpha=0.1,
                        legend_label="confidence interval: {}% - Mode {}".format(
                            p, j + 1
                        ),
                    )
                    fig.line(
                        x=self.speed_range,
                        y=np.percentile(self.wd[j], 50 - p / 2, axis=1),
                        line_color=colors1[j],
                        line_alpha=0.6,
                        line_width=2.5,
                        muted_color=colors1[j],
                        muted_alpha=0.1,
                        legend_label="confidence interval: {}% - Mode {}".format(
                            p, j + 1
                        ),
                    )

        fig.legend.background_fill_alpha = 0.1
        fig.legend.click_policy = "mute"
        fig.legend.label_text_font_size = "10pt"

        return fig

    def plot_log_dec(self, percentile=[], conf_interval=[], harmonics=[1], **kwargs):
        """Plot the log_dec vs frequency.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--').

        Returns
        -------
        fig : Bokeh figure
            The bokeh axes object with the plot
        """
        default_values = dict(line_width=3.0, line_alpha=1.0)
        percentile = np.sort(percentile)
        conf_interval = np.sort(conf_interval)

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        fig = figure(
            height=900,
            width=900,
            tools="pan, box_zoom, wheel_zoom, reset, save",
            title="Campbell Diagram",
            y_axis_label="Log Dec",
            x_axis_label="Rotor speed",
        )
        fig.xaxis.axis_label_text_font_size = "14pt"
        fig.yaxis.axis_label_text_font_size = "14pt"
        fig.axis.major_label_text_font_size = "14pt"
        fig.title.text_font_size = "14pt"

        for j in range(self.log_dec.shape[0]):
            fig.line(
                x=self.speed_range,
                y=np.mean(self.log_dec[j], axis=1),
                line_color=colors1[j],
                line_alpha=1.0,
                line_width=3.0,
                muted_color=colors1[j],
                muted_alpha=0.1,
                legend_label="Mean - Mode {}".format(j + 1),
            )
            if len(percentile):
                for i, p in enumerate(percentile):
                    fig.line(
                        x=self.speed_range,
                        y=np.percentile(self.log_dec[j], p, axis=1),
                        line_color=colors2[i],
                        line_alpha=0.6,
                        line_width=2.5,
                        muted_color=colors2[i],
                        muted_alpha=0.1,
                        legend_label="percentile {}".format(p),
                    )
            if len(conf_interval):
                for i, p in enumerate(conf_interval):
                    fig.line(
                        x=self.speed_range,
                        y=np.percentile(self.log_dec[j], 50 + p / 2, axis=1),
                        line_color=colors1[j],
                        line_alpha=0.6,
                        line_width=2.5,
                        muted_color=colors1[j],
                        muted_alpha=0.1,
                        legend_label="confidence interval: {}% - Mode {}".format(
                            p, j + 1
                        ),
                    )
                    fig.line(
                        x=self.speed_range,
                        y=np.percentile(self.log_dec[j], 50 - p / 2, axis=1),
                        line_color=colors1[j],
                        line_alpha=0.6,
                        line_width=2.5,
                        muted_color=colors1[j],
                        muted_alpha=0.1,
                        legend_label="confidence interval: {}% - Mode {}".format(
                            p, j + 1
                        ),
                    )

        fig.legend.background_fill_alpha = 0.01
        fig.legend.click_policy = "mute"
        fig.legend.label_text_font_size = "10pt"

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
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--').

        Returns
        -------
        grid_plots : bokeh column
            Bokeh column with diagrams for frequency and log dec.
        """
        fig0 = self.plot_nat_freq(percentile, conf_interval, **kwargs)
        fig1 = self.plot_log_dec(percentile, conf_interval, **kwargs)

        grid_plots = gridplot([[fig0, fig1]])

        return grid_plots


class ST_FrequencyResponseResults:
    """Store stochastic results and provide plots for Frequency Response.

    Parameters
    ----------
    freq_resp : array
        Array with the transfer matrix.
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
    grid_plots : bokeh column
        Bokeh column with amplitude and phase plot.
    """

    def __init__(self, speed_range, magnitude, phase):
        self.speed_range = speed_range
        self.magnitude = magnitude
        self.phase = phase

    def plot_magnitude(
        self, percentile=[], conf_interval=[], units="mic-pk-pk", **kwargs
    ):
        """Plot frequency response.

        This method plots the frequency response magnitude given an output and
        an input using Bokeh.

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
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--').

        Returns
        -------
        fig : bokeh figure
            Bokeh plot axes with magnitude plot.
        """
        if units == "m":
            y_axis_label = "Amplitude (m)"
        elif units == "mic-pk-pk":
            y_axis_label = "Amplitude (μ pk-pk)"
        else:
            y_axis_label = "Amplitude (dB)"

        fig = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=640,
            height=480,
            title="Frequency Response - Magnitude",
            x_axis_label="Frequency",
            y_axis_label=y_axis_label,
        )
        fig.xaxis.axis_label_text_font_size = "14pt"
        fig.yaxis.axis_label_text_font_size = "14pt"
        fig.axis.major_label_text_font_size = "14pt"
        fig.title.text_font_size = "14pt"

        if len(percentile):
            for i, p in enumerate(percentile):
                mag_percentile = np.percentile(self.magnitude, p, axis=1)
                fig.line(
                    x=self.speed_range,
                    y=mag_percentile,
                    line_color=colors1[i],
                    line_alpha=1.0,
                    line_width=3,
                    muted_color=colors1[i],
                    muted_alpha=0.1,
                    legend_label="percentile: {}%".format(p),
                )

        if len(conf_interval):
            for i, p in enumerate(conf_interval):
                mag_conf1 = np.percentile(self.magnitude, 50 + p / 2, axis=1)
                mag_conf2 = np.percentile(self.magnitude, 50 - p / 2, axis=1)
                fig.line(
                    x=self.speed_range,
                    y=mag_conf1,
                    line_color=colors1[i],
                    line_alpha=1.0,
                    line_width=3,
                    muted_color=colors1[i],
                    muted_alpha=0.1,
                    legend_label="confidence interval: {}%".format(p),
                )
                fig.line(
                    x=self.speed_range,
                    y=mag_conf2,
                    line_color=colors1[i],
                    line_alpha=1.0,
                    line_width=3,
                    muted_color=colors1[i],
                    muted_alpha=0.1,
                    legend_label="confidence interval: {}%".format(p),
                )

        mag_mean = np.mean(self.magnitude, axis=1)

        fig.line(
            x=self.speed_range,
            y=mag_mean,
            line_color="black",
            line_alpha=1.0,
            line_width=3,
            muted_color="black",
            muted_alpha=0.1,
            legend_label="Mean",
        )
        fig.legend.background_fill_alpha = 0.1
        fig.legend.click_policy = "mute"
        fig.legend.label_text_font_size = "10pt"

        return fig

    def plot_phase(self, percentile=[], conf_interval=[], **kwargs):
        """Plot frequency response.

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
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--').

        Returns
        -------
        fig : bokeh figure
            Bokeh plot axes with phase plot.
        """
        fig = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=640,
            height=480,
            title="Frequency Response - Phase",
            x_axis_label="Frequency",
            y_axis_label="Phase angle",
        )
        fig.xaxis.axis_label_text_font_size = "14pt"
        fig.yaxis.axis_label_text_font_size = "14pt"
        fig.axis.major_label_text_font_size = "14pt"
        fig.title.text_font_size = "14pt"

        if len(percentile):
            for i, p in enumerate(percentile):
                phs_percentile = np.percentile(self.phase, p, axis=1)
                fig.line(
                    x=self.speed_range,
                    y=phs_percentile,
                    line_color=colors1[i],
                    line_alpha=1.0,
                    line_width=3,
                    muted_color=colors1[i],
                    muted_alpha=0.1,
                    legend_label="percentile: {}%".format(p),
                )

        for i, p in enumerate(conf_interval):
            phs_conf1 = np.percentile(self.phase, 50 + p / 2, axis=1)
            phs_conf2 = np.percentile(self.phase, 50 - p / 2, axis=1)
            fig.line(
                x=self.speed_range,
                y=phs_conf1,
                line_color=colors1[i],
                line_alpha=1.0,
                line_width=3,
                muted_color=colors1[i],
                muted_alpha=0.1,
                legend_label="confidence interval: {}%".format(p),
            )
            fig.line(
                x=self.speed_range,
                y=phs_conf2,
                line_color=colors1[i],
                line_alpha=1.0,
                line_width=3,
                legend_label="confidence interval: {}%".format(p),
            )

        phs_mean = np.mean(self.phase, axis=1)

        fig.line(
            x=self.speed_range,
            y=phs_mean,
            line_color="black",
            line_alpha=1.0,
            line_width=3,
            muted_color="black",
            muted_alpha=0.1,
            legend_label="Mean",
        )
        fig.legend.background_fill_alpha = 0.1
        fig.legend.click_policy = "mute"
        fig.legend.label_text_font_size = "10pt"

        return fig

    def plot(self, percentile=[], conf_interval=[], units="mic-pk-pk", *args, **kwargs):
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
        args : optional
            Additional plot axes
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. line_color="blue").

        Returns
        -------
        grid_plots : bokeh column
            Bokeh column with amplitude and phase plot
        """
        fig0 = self.plot_magnitude(percentile, conf_interval, units, **kwargs)
        fig1 = self.plot_phase(percentile, conf_interval, **kwargs)

        grid_plots = gridplot([[fig0], [fig1]])

        return grid_plots


class ST_TimeResponseResults:
    """Store stochastic results and provide plots for Time Response.

    Parameters
    ----------
    time_range : 1-dimensional array
        Time array.
    yout : array
        System response.
    xout : array
        Time evolution of the state vector.
    dof : int
        Degree of freedom that will be observ

    Returns
    -------
    fig : bokeh figure
        Returns the bokeh axes object with the plot
    """

    def __init__(self, time_range, yout, xout, dof):
        self.time_range = time_range
        self.yout = yout
        self.xout = xout
        self.dof = dof

    def plot(self, percentile=[], conf_interval=[], *args, **kwargs):
        """Plot time response.

        This method plots the time response given.

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
            Additional key word arguments can be passed to change
            the plot (e.g. line_color="blue").

        Returns
        -------
        grid_plots : bokeh figure
            Bokeh figure with time response plot.
        """
        if self.dof % 4 == 0:
            obs_dof = "x"
        elif self.dof % 4 == 1:
            obs_dof = "y"
        elif self.dof % 4 == 2:
            obs_dof = "α"
        elif self.dof % 4 == 3:
            obs_dof = "β"

        fig = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=900,
            height=900,
            title="Response for node {} and degree of freedom {}".format(
                self.dof // 4, obs_dof
            ),
            x_axis_label="Time (s)",
            y_axis_label="Amplitude",
        )
        fig.xaxis.axis_label_text_font_size = "14pt"
        fig.yaxis.axis_label_text_font_size = "14pt"
        fig.axis.major_label_text_font_size = "14pt"
        fig.title.text_font_size = "14pt"

        if len(percentile):
            for i, p in enumerate(percentile):
                phs_percentile = np.percentile(self.yout[..., self.dof], p, axis=0)
                fig.line(
                    x=self.time_range,
                    y=phs_percentile,
                    line_color=colors1[i],
                    line_alpha=1.0,
                    line_width=3,
                    muted_color=colors1[i],
                    muted_alpha=0.1,
                    legend_label="percentile: {}%".format(p),
                )

        for i, p in enumerate(conf_interval):
            conf1 = np.percentile(self.yout[..., self.dof], 50 + p / 2, axis=0)
            conf2 = np.percentile(self.yout[..., self.dof], 50 - p / 2, axis=0)
            fig.line(
                x=self.time_range,
                y=conf1,
                line_color=colors1[i],
                line_alpha=1.0,
                line_width=3,
                muted_color=colors1[i],
                muted_alpha=0.1,
                legend_label="confidence interval: {}%".format(p),
            )
            fig.line(
                x=self.time_range,
                y=conf2,
                line_color=colors1[i],
                line_alpha=1.0,
                line_width=3,
                legend_label="confidence interval: {}%".format(p),
            )

        t_mean = np.mean(self.yout[..., self.dof], axis=0)

        fig.line(
            x=self.time_range,
            y=t_mean,
            line_color="black",
            line_alpha=1.0,
            line_width=3,
            muted_color="black",
            muted_alpha=0.1,
            legend_label="Mean",
        )
        fig.legend.background_fill_alpha = 0.1
        fig.legend.click_policy = "mute"
        fig.legend.label_text_font_size = "10pt"

        return fig


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
    fig : bokeh figure
        Returns the bokeh axes object with the plot
    """

    def __init__(self, forced_resp, magnitude, phase, frequency_range):
        self.forced_resp = forced_resp
        self.magnitude = magnitude
        self.phase = phase
        self.frequency_range = frequency_range

    def plot_magnitude(
        self, dof, percentile=[], conf_interval=[], units="mic-pk-pk", **kwargs
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
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--').

        Returns
        -------
        fig : bokeh figure
            Bokeh plot axes with magnitude plot.
        """
        if units == "m":
            y_axis_label = "Amplitude (m)"
        elif units == "mic-pk-pk":
            y_axis_label = "Amplitude (μ pk-pk)"
        else:
            y_axis_label = "Amplitude (dB)"

        fig = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=640,
            height=480,
            title="Unbalance Response - Magnitude",
            x_axis_label="Frequency",
            y_axis_label=y_axis_label,
        )
        fig.xaxis.axis_label_text_font_size = "14pt"
        fig.yaxis.axis_label_text_font_size = "14pt"
        fig.axis.major_label_text_font_size = "14pt"
        fig.title.text_font_size = "14pt"

        if len(percentile):
            for i, p in enumerate(percentile):
                mag_percentile = np.percentile(self.magnitude[..., dof], p, axis=0)
                fig.line(
                    x=self.frequency_range,
                    y=mag_percentile,
                    line_color=colors1[i],
                    line_alpha=1.0,
                    line_width=3,
                    muted_color=colors1[i],
                    muted_alpha=0.1,
                    legend_label="percentile: {}%".format(p),
                )

        if len(conf_interval):
            for i, p in enumerate(conf_interval):
                mag_conf1 = np.percentile(self.magnitude[..., dof], 50 + p / 2, axis=0)
                mag_conf2 = np.percentile(self.magnitude[..., dof], 50 - p / 2, axis=0)
                fig.line(
                    x=self.frequency_range,
                    y=mag_conf1,
                    line_color=colors1[i],
                    line_alpha=1.0,
                    line_width=3,
                    muted_color=colors1[i],
                    muted_alpha=0.1,
                    legend_label="confidence interval: {}%".format(p),
                )
                fig.line(
                    x=self.frequency_range,
                    y=mag_conf2,
                    line_color=colors1[i],
                    line_alpha=1.0,
                    line_width=3,
                    muted_color=colors1[i],
                    muted_alpha=0.1,
                    legend_label="confidence interval: {}%".format(p),
                )

        mag_mean = np.mean(self.magnitude[..., dof], axis=0)

        fig.line(
            x=self.frequency_range,
            y=mag_mean,
            line_color="black",
            line_alpha=1.0,
            line_width=3,
            muted_color="black",
            muted_alpha=0.1,
            legend_label="Mean",
        )
        fig.legend.background_fill_alpha = 0.1
        fig.legend.click_policy = "mute"
        fig.legend.label_text_font_size = "10pt"

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
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--').

        Returns
        -------
        fig : bokeh figure
            Bokeh plot axes with phase plot.
        """
        fig = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=640,
            height=480,
            title="Unbalance Response - Phase",
            x_axis_label="Frequency",
            y_axis_label="Phase angle",
        )
        fig.xaxis.axis_label_text_font_size = "14pt"
        fig.yaxis.axis_label_text_font_size = "14pt"
        fig.axis.major_label_text_font_size = "14pt"
        fig.title.text_font_size = "14pt"

        if len(percentile):
            for i, p in enumerate(percentile):
                phs_percentile = np.percentile(self.phase[..., dof], p, axis=0)
                fig.line(
                    x=self.frequency_range,
                    y=phs_percentile,
                    line_color=colors1[i],
                    line_alpha=1.0,
                    line_width=3,
                    muted_color=colors1[i],
                    muted_alpha=0.1,
                    legend_label="percentile: {}%".format(p),
                )

        for i, p in enumerate(conf_interval):
            phs_conf1 = np.percentile(self.phase[..., dof], 50 + p / 2, axis=0)
            phs_conf2 = np.percentile(self.phase[..., dof], 50 - p / 2, axis=0)
            fig.line(
                x=self.frequency_range,
                y=phs_conf1,
                line_color=colors1[i],
                line_alpha=1.0,
                line_width=3,
                muted_color=colors1[i],
                muted_alpha=0.1,
                legend_label="confidence interval: {}%".format(p),
            )
            fig.line(
                x=self.frequency_range,
                y=phs_conf2,
                line_color=colors1[i],
                line_alpha=1.0,
                line_width=3,
                legend_label="confidence interval: {}%".format(p),
            )

        phs_mean = np.mean(self.phase[..., dof], axis=0)

        fig.line(
            x=self.frequency_range,
            y=phs_mean,
            line_color="black",
            line_alpha=1.0,
            line_width=3,
            muted_color="black",
            muted_alpha=0.1,
            legend_label="Mean",
        )
        fig.legend.background_fill_alpha = 0.1
        fig.legend.click_policy = "mute"
        fig.legend.label_text_font_size = "10pt"

        return fig

    def plot(
        self, dof, percentile=[], conf_interval=[], units="mic-pk-pk", *args, **kwargs
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
            Additional key word arguments can be passed to change
            the plot (e.g. line_color="blue").

        Returns
        -------
        grid_plots : bokeh column
            Bokeh column with amplitude and phase plot
        """
        fig0 = self.plot_magnitude(dof, percentile, conf_interval, units, **kwargs)
        fig1 = self.plot_phase(dof, percentile, conf_interval, **kwargs)

        grid_plots = gridplot([[fig0], [fig1]])

        return grid_plots
