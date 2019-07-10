import pickle
import numpy as np
from scipy import interpolate

import matplotlib as mpl
import matplotlib.pyplot as plt
import bokeh.palettes as bp
from mpl_toolkits.mplot3d import Axes3D
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap
from bokeh.models import (
        ColumnDataSource,
        ColorBar,
        Arrow,
        NormalHead,
        Label,
        HoverTool
)

# set bokeh palette of colors
bokeh_colors = bp.RdGy[11]


class Results(np.ndarray):
    """Class used to store results and provide plots.
    This class subclasses np.ndarray to provide additional info and a plot
    method to the calculated results from Rotor.
    Metadata about the results should be stored on info as a dictionary to be
    used on plot configurations and so on.
    Additional attributes can be passed as a dictionary in new_attributes kwarg.
    """

    def __new__(cls, input_array, new_attributes=None):
        obj = np.asarray(input_array).view(cls)

        for k, v in new_attributes.items():
            setattr(obj, k, v)

        # save new attributes names to create them on array finalize
        obj._new_attributes = new_attributes

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        try:
            for k, v in obj._new_attributes.items():
                setattr(self, k, getattr(obj, k, v))
        except AttributeError:
            return

    def __reduce__(self):

        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self._new_attributes,)

        return pickled_state[0], pickled_state[1], new_state

    def __setstate__(self, state):
        self._new_attributes = state[-1]
        for k, v in self._new_attributes.items():
            setattr(self, k, v)
        super().__setstate__(state[0:-1])

    def save(self, file):
        with open(file, mode="wb") as f:
            pickle.dump(self, f)

    def plot(self, *args, **kwargs):
        raise NotImplementedError


class CampbellResults:
    def __init__(self, speed_range, wd, log_dec, whirl_values):
        self.speed_range = speed_range
        self.wd = wd
        self.log_dec = log_dec
        self.whirl_values = whirl_values

    def _plot_matplotlib(self, harmonics=[1], fig=None, ax=None, **kwargs):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        wd = self.wd
        num_frequencies = wd.shape[1]
        log_dec = self.log_dec
        whirl = self.whirl_values
        speed_range = np.repeat(
            self.speed_range[:, np.newaxis], num_frequencies, axis=1
        )

        default_values = dict(cmap="RdBu", vmin=0.1, vmax=2.0, s=30, alpha=0.5)
        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        for mark, whirl_dir, legend in zip(
            ["^", "o", "v"], [0.0, 0.5, 1.0], ["Foward", "Mixed", "Backward"]
        ):
            for i in range(num_frequencies):
                w_i = wd[:, i]
                whirl_i = whirl[:, i]
                log_dec_i = log_dec[:, i]
                speed_range_i = speed_range[:, i]

                for harm in harmonics:
                    ax.plot(
                        speed_range[:, 0],
                        harm * speed_range[:, 0],
                        color="k",
                        linewidth=1.5,
                        linestyle="-.",
                        alpha=0.75,
                        label="Rotor speed",
                    )

                    idx = np.argwhere(np.diff(np.sign(w_i - harm*speed_range_i))).flatten()
                    if len(idx) != 0:
                        idx = idx[0]

                        interpolated = interpolate.interp1d(
                            x=[speed_range_i[idx], speed_range_i[idx+1]],
                            y=[w_i[idx], w_i[idx+1]],
                            kind="linear"
                        )
                        xnew = np.linspace(
                            speed_range_i[idx],
                            speed_range_i[idx+1],
                            num=20,
                            endpoint=True,
                        )
                        ynew = interpolated(xnew)
                        idx = np.argwhere(np.diff(np.sign(ynew - harm*xnew))).flatten()

                        ax.scatter(xnew[idx], ynew[idx], marker="X", s=30, c="g")

                whirl_mask = whirl_i == whirl_dir
                if whirl_mask.shape[0] == 0:
                    continue
                else:
                    im = ax.scatter(
                        speed_range_i[whirl_mask],
                        w_i[whirl_mask],
                        c=log_dec_i[whirl_mask],
                        marker=mark,
                        **kwargs,
                    )

        if len(fig.axes) == 1:
            cbar = fig.colorbar(im)
            cbar.ax.set_ylabel("log dec")
            cbar.solids.set_edgecolor("face")

            forward_label = mpl.lines.Line2D(
                [], [], marker="^", lw=0, color="tab:blue", alpha=0.3, label="Forward"
            )
            backward_label = mpl.lines.Line2D(
                [], [], marker="v", lw=0, color="tab:blue", alpha=0.3, label="Backward"
            )
            mixed_label = mpl.lines.Line2D(
                [], [], marker="o", lw=0, color="tab:blue", alpha=0.3, label="Mixed"
            )

            legend = plt.legend(
                handles=[forward_label, backward_label, mixed_label], loc=2
            )

            ax.add_artist(legend)

            ax.set_xlabel("Rotor speed ($rad/s$)")
            ax.set_ylabel("Damped natural frequencies ($rad/s$)")

        return fig, ax

    def _plot_bokeh(self, harmonics=[1], **kwargs):
        wd = self.wd
        num_frequencies = wd.shape[1]
        log_dec = self.log_dec
        whirl = self.whirl_values
        speed_range = np.repeat(
            self.speed_range[:, np.newaxis], num_frequencies, axis=1
        )

        log_dec_map = log_dec.flatten()

        default_values = dict(
            cmap="viridis",
            vmin=min(log_dec_map),
            vmax=max(log_dec_map),
            s=30,
            alpha=1.0,
        )

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        camp = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            title="Campbell Diagram - Damped Natural Frequency Map",
            width=1600,
            height=900,
            x_axis_label="Rotor speed (rad/s)",
            y_axis_label="Damped natural frequencies (rad/s)",
        )
        camp.xaxis.axis_label_text_font_size = "14pt"
        camp.yaxis.axis_label_text_font_size = "14pt"

        color_mapper = linear_cmap(
                field_name="color",
                palette=bp.viridis(256),
                low=min(log_dec_map),
                high=max(log_dec_map),
        )

        for mark, whirl_dir, legend in zip(
            ["^", "o", "v"], [0.0, 0.5, 1.0], ["Foward", "Mixed", "Backward"]
        ):
            num_frequencies = wd.shape[1]
            for i in range(num_frequencies):
                w_i = wd[:, i]
                whirl_i = whirl[:, i]
                log_dec_i = log_dec[:, i]
                speed_range_i = speed_range[:, i]

                for harm in harmonics:
                    camp.line(
                        x=speed_range[:, 0],
                        y=harm * speed_range[:, 0],
                        line_width=3,
                        color=bokeh_colors[0],
                        line_dash="dotdash",
                        line_alpha=0.75,
                        legend="Rotor speed",
                        muted_color=bokeh_colors[0],
                        muted_alpha=0.2,
                    )

                    idx = np.argwhere(np.diff(np.sign(w_i - harm*speed_range_i))).flatten()
                    if len(idx) != 0:
                        idx = idx[0]

                        interpolated = interpolate.interp1d(
                            x=[speed_range_i[idx], speed_range_i[idx+1]],
                            y=[w_i[idx], w_i[idx+1]],
                            kind="linear"
                        )
                        xnew = np.linspace(
                            speed_range_i[idx],
                            speed_range_i[idx+1],
                            num=30,
                            endpoint=True,
                        )
                        ynew = interpolated(xnew)
                        idx = np.argwhere(np.diff(np.sign(ynew - harm*xnew))).flatten()

                        source = ColumnDataSource(
                                dict(xnew=xnew[idx], ynew=ynew[idx])
                        )
                        camp.square(
                                "xnew",
                                "ynew",
                                source=source,
                                size=10,
                                color=bokeh_colors[9],
                                name="critspeed"
                        )
                        hover = HoverTool(names=["critspeed"])
                        hover.tooltips = [
                                ("Frequency :", "@xnew"),
                                ("Critical Speed :", "@ynew")
                        ]
                        hover.mode = "mouse"

                whirl_mask = whirl_i == whirl_dir
                if whirl_mask.shape[0] == 0:
                    continue
                else:
                    source = ColumnDataSource(
                        dict(
                            x=speed_range_i[whirl_mask],
                            y=w_i[whirl_mask],
                            color=log_dec_i[whirl_mask],
                        )
                    )
                    camp.scatter(
                        x="x",
                        y="y",
                        color=color_mapper,
                        marker=mark,
                        fill_alpha=1.0,
                        size=9,
                        muted_color=color_mapper,
                        muted_alpha=0.2,
                        source=source,
                        legend=legend,
                    )

        color_bar = ColorBar(
            color_mapper=color_mapper["transform"],
            width=8,
            location=(0, 0),
            title="log dec",
            title_text_font_style="bold italic",
            title_text_align="center",
            major_label_text_align="left",
        )
        camp.add_tools(hover)
        camp.legend.background_fill_alpha = 0.1
        camp.legend.click_policy = "mute"
        camp.legend.location = "top_left"
        camp.add_layout(color_bar, "right")

        return camp

    def plot(self, *args, plot_type="matplotlib", **kwargs):
        """Plot campbell results.
        Parameters
        ----------
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        plot_type: str
            Matplotlib or bokeh.
            The default is matplotlib
        fig : matplotlib figure, optional
            Figure to insert axes with log_dec colorbar.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        if plot_type == "matplotlib":
            return self._plot_matplotlib(*args, **kwargs)
        elif plot_type == "bokeh":
            return self._plot_bokeh(*args, **kwargs)
        else:
            raise ValueError(f"")


class FrequencyResponseResults:
    def __init__(self, freq_resp, speed_range, magnitude, phase):
        self.freq_resp = freq_resp
        self.speed_range = speed_range
        self.magnitude = magnitude
        self.phase = phase

    def plot_magnitude_matplotlib(self, inp, out, ax=None, units="mic-pk-pk", **kwargs):
        """Plot frequency response.
        This method plots the frequency response magnitude given an output and
        an input.
        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax : matplotlib.axes, optional
            Matplotlib axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')
        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with amplitude plot.
        mag_plot : bokeh plot axes
            Bokeh plot axes with amplitude plot.
        Examples
        --------
        """
        if ax is None:
            ax = plt.gca()

        frequency_range = self.speed_range
        mag = self.magnitude

        ax.plot(frequency_range, mag[inp, out, :], **kwargs)

        ax.set_xlim(0, max(frequency_range))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="upper"))

        ax.set_ylabel("Mag H$(j\omega)$")
        ax.set_xlabel("Frequency (rad/s)")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        return ax

    def plot_magnitude_bokeh(self, inp, out, units="m", **kwargs):
        """Plot frequency response.
        This method plots the frequency response magnitude given an output and
        an input.
        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax : matplotlib.axes, optional
            Matplotlib axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')
        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with amplitude plot.
        mag_plot : bokeh plot axes
            Bokeh plot axes with amplitude plot.
        Examples
        --------
        """
        frequency_range = self.speed_range
        mag = self.magnitude

        if units == "m":
            y_axis_label = "Amplitude (m)"
        elif units == "mic-pk-pk":
            y_axis_label = "Amplitude (\mu pk-pk)"
        else:
            y_axis_label = "Amplitude (dB)"

        # bokeh plot - create a new plot
        mag_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=900,
            height=400,
            title="Frequency Response - Magnitude",
            x_axis_label="Frequency",
            y_axis_label=y_axis_label,
        )
        mag_plot.xaxis.axis_label_text_font_size = "14pt"
        mag_plot.yaxis.axis_label_text_font_size = "14pt"

        source = ColumnDataSource(dict(x=frequency_range, y=mag[inp, out, :]))
        mag_plot.line(
            x="x",
            y="y",
            source=source,
            line_color=bokeh_colors[0],
            line_alpha=1.0,
            line_width=3,
        )

        return mag_plot

    def plot_phase_matplotlib(self, inp, out, ax=None, **kwargs):
        """Plot frequency response.
        This method plots the frequency response phase given an output and
        an input.
        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax : matplotlib.axes, optional
            Matplotlib axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')
        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with phase plot.
        phase_plot : bokeh plot axes
            Bokeh plot axes with phase plot.
        Examples
        --------
        """
        if ax is None:
            ax = plt.gca()

        frequency_range = self.speed_range
        phase = self.phase

        ax.plot(frequency_range, phase[inp, out, :], **kwargs)

        ax.set_xlim(0, max(frequency_range))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="upper"))

        ax.set_ylabel("Phase")
        ax.set_xlabel("Frequency (rad/s)")

        return ax

    def plot_phase_bokeh(self, inp, out, **kwargs):
        """Plot frequency response.
        This method plots the frequency response phase given an output and
        an input.
        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax : matplotlib.axes, optional
            Matplotlib axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')
        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with phase plot.
        phase_plot : bokeh plot axes
            Bokeh plot axes with phase plot.
        Examples
        --------
        """
        frequency_range = self.speed_range
        phase = self.phase

        # bokeh plot - create a new plot
        phase_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=900,
            height=400,
            title="Frequency Response - Phase",
            x_axis_label="Frequency",
            y_axis_label="Phase",
        )
        phase_plot.xaxis.axis_label_text_font_size = "14pt"
        phase_plot.yaxis.axis_label_text_font_size = "14pt"

        source = ColumnDataSource(dict(x=frequency_range, y=phase[inp, out, :]))
        phase_plot.line(
            x="x",
            y="y",
            source=source,
            line_color=bokeh_colors[0],
            line_alpha=1.0,
            line_width=3,
        )

        return phase_plot

    def _plot_matplotlib(self, inp, out, ax0=None, ax1=None, **kwargs):
        """Plot frequency response.
        This method plots the frequency response given
        an output and an input.
        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax0 : matplotlib.axes, bokeh plot axes optional
            Matplotlib and bokeh plot axes where the amplitude will be plotted.
            If None creates a new.
        ax1 : matplotlib.axes, bokeh plot axes optional
            Matplotlib and bokeh plot axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')
        Returns
        -------
        ax0 : matplotlib.axes
            Matplotlib axes with amplitude plot.
        ax1 : matplotlib.axes
            Matplotlib axes with phase plot.
        bk_ax0 : bokeh plot axes
            Bokeh plot axes with amplitude plot
        bk_ax1 : bokeh plot axes
            Bokeh plot axes with phase plot
        Examples
        --------
        """
        if ax0 is None and ax1 is None:
            fig, (ax0, ax1) = plt.subplots(2)

        # matplotlib axes
        ax0 = self.plot_magnitude_matplotlib(inp, out, ax=ax0)
        ax1 = self.plot_phase_matplotlib(inp, out, ax=ax1)

        ax0.set_xlabel("")

        return ax0, ax1

    def _plot_bokeh(self, inp, out, ax0=None, ax1=None, **kwargs):
        """Plot frequency response.
        This method plots the frequency response given
        an output and an input.
        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax0 : matplotlib.axes, bokeh plot axes optional
            Matplotlib and bokeh plot axes where the amplitude will be plotted.
            If None creates a new.
        ax1 : matplotlib.axes, bokeh plot axes optional
            Matplotlib and bokeh plot axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')
        Returns
        -------
        ax0 : matplotlib.axes
            Matplotlib axes with amplitude plot.
        ax1 : matplotlib.axes
            Matplotlib axes with phase plot.
        bk_ax0 : bokeh plot axes
            Bokeh plot axes with amplitude plot
        bk_ax1 : bokeh plot axes
            Bokeh plot axes with phase plot
        Examples
        --------
        """
        # bokeh plot axes
        bk_ax0 = self.plot_magnitude_bokeh(inp, out, ax=ax0)
        bk_ax1 = self.plot_phase_bokeh(inp, out, ax=ax1)

        # show the bokeh plot results
        grid_plots = gridplot([[bk_ax0], [bk_ax1]])
        show(grid_plots)

        return bk_ax0, bk_ax1

    def plot(self, inp, out, *args, plot_type="matplotlib", **kwargs):
        """Plot frequency response.
        This method plots the frequency response given
        an output and an input.
        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        plot_type: str
            Matplotlib or bokeh.
            The default is matplotlib
        ax0 : matplotlib.axes, bokeh plot axes optional
            Matplotlib and bokeh plot axes where the amplitude will be plotted.
            If None creates a new.
        ax1 : matplotlib.axes, bokeh plot axes optional
            Matplotlib and bokeh plot axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')
        Returns
        -------
        ax0 : matplotlib.axes
            Matplotlib axes with amplitude plot.
        ax1 : matplotlib.axes
            Matplotlib axes with phase plot.
        bk_ax0 : bokeh plot axes
            Bokeh plot axes with amplitude plot
        bk_ax1 : bokeh plot axes
            Bokeh plot axes with phase plot
        Examples
        --------
        """
        if plot_type == "matplotlib":
            return self._plot_matplotlib(inp, out, *args, **kwargs)
        elif plot_type == "bokeh":
            return self._plot_bokeh(inp, out, *args, **kwargs)
        else:
            raise ValueError(f"")

    def plot_freq_response_grid(self, outs, inps, ax=None, **kwargs):
        """Plot frequency response.
        This method plots the frequency response given
        an output and an input.
        Parameters
        ----------
        outs : list
            List with the desired outputs.
        inps : list
            List with the desired outputs.
        ax : array with matplotlib.axes, optional
            Matplotlib axes array created with plt.subplots.
            It needs to have a shape of (2*inputs, outputs).
        Returns
        -------
        ax : array with matplotlib.axes, optional
            Matplotlib axes array created with plt.subplots.
        """
        if ax is None:
            fig, ax = plt.subplots(
                len(inps) * 2,
                len(outs),
                sharex=True,
                figsize=(4 * len(outs), 3 * len(inps)),
            )
            fig.subplots_adjust(hspace=0.001, wspace=0.25)

        if len(outs) > 1:
            for i, out in enumerate(outs):
                for j, inp in enumerate(inps):
                    self.plot_magnitude(out, inp, ax=ax[2 * i, j], **kwargs)
                    self.plot_phase(out, inp, ax=ax[2 * i + 1, j], **kwargs)
        else:
            for i, inp in enumerate(inps):
                self.plot_magnitude(outs[0], inp, ax=ax[2 * i], **kwargs)
                self.plot_phase(outs[0], inp, ax=ax[2 * i + 1], **kwargs)

        return ax


class ForcedResponseResults:
    def __init__(self, forced_resp, speed_range, magnitude, phase):
        self.forced_resp = forced_resp
        self.speed_range = speed_range
        self.magnitude = magnitude
        self.phase = phase

    def plot_magnitude_matplotlib(self, dof, ax=None, units="m", **kwargs):
        """Plot frequency response.
        This method plots the frequency response magnitude given an output and
        an input.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        ax : matplotlib.axes, optional
            Matplotlib axes where the phase will be plotted.
            If None creates a new.
        units : str
            Units to plot the magnitude ('m' or 'mic-pk-pk')
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with phase plot.

        Examples
        --------
        """
        if ax is None:
            ax = plt.gca()

        frequency_range = self.speed_range
        mag = self.magnitude

        if units == "m":
            ax.set_ylabel("Amplitude $(m)$")
        elif units == "mic-pk-pk":
            mag = 2 * mag * 1e6
            ax.set_ylabel("Amplitude $(\mu pk-pk)$")

        ax.plot(frequency_range, mag[dof], **kwargs)

        ax.set_xlim(0, max(frequency_range))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="upper"))

        ax.set_xlabel("Frequency (rad/s)")
        ax.legend()

        return ax

    def plot_magnitude_bokeh(self, dof, units="m", **kwargs):
        """Plot frequency response.
        This method plots the frequency response magnitude given an output and
        an input.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        units : str
            Units to plot the magnitude ('m' or 'mic-pk-pk')
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        mag_plot : bokeh axes
            bokeh axes with magnitude plot

        Examples
        --------
        """
        frequency_range = self.speed_range
        mag = self.magnitude

        if units == "m":
            y_axis_label = "Amplitude (m)"
        elif units == "mic-pk-pk":
            mag = 2 * mag * 1e6
            y_axis_label = "Amplitude $(\mu pk-pk)$"

        # bokeh plot - create a new plot
        mag_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=900,
            height=400,
            title="Forced Response - Magnitude",
            x_axis_label="Frequency",
            x_range=[0, max(frequency_range)],
            y_axis_label=y_axis_label,
        )
        mag_plot.xaxis.axis_label_text_font_size = "14pt"
        mag_plot.yaxis.axis_label_text_font_size = "14pt"

        source = ColumnDataSource(dict(x=frequency_range, y=mag[dof]))
        mag_plot.line(
            x="x",
            y="y",
            source=source,
            line_color=bokeh_colors[0],
            line_alpha=1.0,
            line_width=3,
        )

        return mag_plot

    def plot_phase_matplotlib(self, dof, ax=None, **kwargs):
        """Plot frequency response.
        This method plots the frequency response phase given an output and
        an input.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        ax : matplotlib.axes, optional
            Matplotlib axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with phase plot.

        Examples
        --------
        """
        if ax is None:
            ax = plt.gca()

        frequency_range = self.speed_range
        phase = self.phase

        ax.plot(frequency_range, phase[dof], **kwargs)

        ax.set_xlim(0, max(frequency_range))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="upper"))

        ax.set_ylabel("Phase")
        ax.set_xlabel("Frequency (rad/s)")
        ax.legend()

        return ax

    def plot_phase_bokeh(self, dof, **kwargs):
        """Plot frequency response.
        This method plots the frequency response phase given an output and
        an input.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        phase_plot : bokeh axes
            Bokeh axes with phase plot

        Examples
        --------
        """
        frequency_range = self.speed_range
        phase = self.phase

        phase_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=900,
            height=400,
            title="Forced Response - Magnitude",
            x_axis_label="Frequency",
            x_range=[0, max(frequency_range)],
            y_axis_label="Phase",
        )
        source = ColumnDataSource(dict(x=frequency_range, y=phase[dof]))
        phase_plot.line(
            x="x",
            y="y",
            source=source,
            line_color=bokeh_colors[0],
            line_alpha=1.0,
            line_width=3,
        )
        phase_plot.xaxis.axis_label_text_font_size = "14pt"
        phase_plot.yaxis.axis_label_text_font_size = "14pt"

        return phase_plot

    def _plot_matplotlib(self, dof, ax0=None, ax1=None, **kwargs):

        if ax0 is None and ax1 is None:
            fig, (ax0, ax1) = plt.subplots(2)

        ax0 = self.plot_magnitude_matplotlib(dof, ax=ax0, **kwargs)
        # remove label from phase plot
        kwargs.pop("label", None)
        kwargs.pop("units", None)
        ax1 = self.plot_phase_matplotlib(dof, ax=ax1, **kwargs)

        ax0.set_xlabel("")
        ax0.legend()

        return ax0, ax1

    def _plot_bokeh(self, dof, **kwargs):
        # bokeh plot axes
        bk_ax0 = self.plot_magnitude_bokeh(dof, **kwargs)
        bk_ax1 = self.plot_phase_bokeh(dof, **kwargs)

        # show the bokeh plot results
        grid_plots = gridplot([[bk_ax0], [bk_ax1]])
        show(grid_plots)

        return bk_ax0, bk_ax1

    def plot(self, dof, plot_type="matplotlib", **kwargs):
        """Plot frequency response.
        This method plots the frequency response given
        an output and an input.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        plot_type: str
            Matplotlib or bokeh.
            The default is matplotlib
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------

        Examples
        --------
        """
        if plot_type == "matplotlib":
            return self._plot_matplotlib(dof, **kwargs)
        elif plot_type == "bokeh":
            return self._plot_bokeh(dof, **kwargs)
        else:
            raise ValueError(f"{plot_type} is not a valid plot type.")


class ModeShapeResults:
    def __init__(
        self,
        modes,
        ndof,
        nodes,
        nodes_pos,
        elements_length,
        w,
        wd,
        log_dec,
        kappa_modes,
    ):
        self.modes = modes
        self.ndof = ndof
        self.nodes = nodes
        self.nodes_pos = nodes_pos
        self.elements_length = elements_length
        self.w = w
        self.wd = wd
        self.log_dec = log_dec
        self.kappa_modes = kappa_modes

    def plot(self, mode=None, evec=None, fig=None, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection="3d")

        evec0 = self.modes[:, mode]
        nodes = self.nodes
        nodes_pos = self.nodes_pos
        kappa_modes = self.kappa_modes
        elements_length = self.elements_length

        modex = evec0[0::4]
        modey = evec0[1::4]
        xmax, ixmax = max(abs(modex)), np.argmax(abs(modex))
        ymax, iymax = max(abs(modey)), np.argmax(abs(modey))

        if ymax > 0.4 * xmax:
            evec0 /= modey[iymax]
        else:
            evec0 /= modex[ixmax]

        modex = evec0[0::4]
        modey = evec0[1::4]

        num_points = 201
        c = np.linspace(0, 2 * np.pi, num_points)
        circle = np.exp(1j * c)

        x_circles = np.zeros((num_points, len(nodes)))
        y_circles = np.zeros((num_points, len(nodes)))
        z_circles_pos = np.zeros((num_points, len(nodes)))

        kappa_mode = kappa_modes[mode]

        for node in nodes:
            x = modex[node] * circle
            x_circles[:, node] = np.real(x)
            y = modey[node] * circle
            y_circles[:, node] = np.real(y)
            z_circles_pos[:, node] = nodes_pos[node]

        # plot lines
        nn = 21
        zeta = np.linspace(0, 1, nn)
        onn = np.ones_like(zeta)

        zeta = zeta.reshape(nn, 1)
        onn = onn.reshape(nn, 1)

        xn = np.zeros(nn * (len(nodes) - 1))
        yn = np.zeros(nn * (len(nodes) - 1))
        zn = np.zeros(nn * (len(nodes) - 1))

        N1 = onn - 3 * zeta ** 2 + 2 * zeta ** 3
        N2 = zeta - 2 * zeta ** 2 + zeta ** 3
        N3 = 3 * zeta ** 2 - 2 * zeta ** 3
        N4 = -zeta ** 2 + zeta ** 3

        for Le, n in zip(elements_length, nodes):
            node_pos = nodes_pos[n]
            Nx = np.hstack((N1, Le * N2, N3, Le * N4))
            Ny = np.hstack((N1, -Le * N2, N3, -Le * N4))

            xx = [4 * n, 4 * n + 3, 4 * n + 4, 4 * n + 7]
            yy = [4 * n + 1, 4 * n + 2, 4 * n + 5, 4 * n + 6]

            pos0 = nn * n
            pos1 = nn * (n + 1)
            xn[pos0:pos1] = Nx @ evec0[xx].real
            yn[pos0:pos1] = Ny @ evec0[yy].real
            zn[pos0:pos1] = (node_pos * onn + Le * zeta).reshape(nn)

        for node in nodes:
            ax.plot(
                x_circles[10:, node],
                y_circles[10:, node],
                z_circles_pos[10:, node],
                color=kappa_mode[node],
                linewidth=0.5,
                zdir="x",
            )
            ax.scatter(
                x_circles[10, node],
                y_circles[10, node],
                z_circles_pos[10, node],
                s=5,
                color=kappa_mode[node],
                zdir="x",
            )

        ax.plot(xn, yn, zn, "k--", zdir="x")

        # plot center line
        zn_cl0 = -(zn[-1] * 0.1)
        zn_cl1 = zn[-1] * 1.1
        zn_cl = np.linspace(zn_cl0, zn_cl1, 30)
        ax.plot(zn_cl * 0, zn_cl * 0, zn_cl, "k-.", linewidth=0.8, zdir="x")

        ax.set_zlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlim(zn_cl0 - 0.1, zn_cl1 + 0.1)

        ax.set_title(
            f"$speed$ = {self.w:.1f} rad/s\n$"
            f"\omega_d$ = {self.wd[mode]:.1f} rad/s\n"
            f"$log dec$ = {self.log_dec[mode]:.1f}"
        )

        return fig, ax


class StaticResults():
    def __init__(
        self,
        disp_y,
        Vx,
        Bm,
        df_shaft,
        df_disks,
        df_bearings,
        nodes,
        nodes_pos,
        Vx_axis,
    ):

        self.disp_y = disp_y
        self.Vx = Vx
        self.Bm = Bm
        self.df_shaft = df_shaft
        self.df_disks = df_disks
        self.df_bearings = df_bearings
        self.nodes = nodes
        self.nodes_pos = nodes_pos
        self.Vx_axis = Vx_axis

    def plot(self):
        """Plot static analysis graphs.
        This method plots:
            free-body diagram,
            deformed shaft,
            shearing force diagram,
            bending moment diagram.

        Parameters
        ----------

        Returns
        -------
        grid_plots : bokeh.gridplot
        --------
        """
        source = ColumnDataSource(
            data=dict(
                x0=self.nodes_pos,
                y0=self.disp_y * 1000,
                y1=[0] * len(self.nodes_pos)
            )
        )

        TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,hover"
        TOOLTIPS = [
            ("Shaft lenght:", "@x0"),
            ("Underformed:", "@y1"),
            ("Displacement:", "@y0"),
        ]

        # create displacement plot
        disp_graph = figure(
            tools=TOOLS,
            tooltips=TOOLTIPS,
            width=800,
            height=400,
            title="Static Analysis",
            x_axis_label="shaft lenght",
            y_axis_label="lateral displacement",
        )
        disp_graph.xaxis.axis_label_text_font_size = "14pt"
        disp_graph.yaxis.axis_label_text_font_size = "14pt"

        interpolated = interpolate.interp1d(
            source.data["x0"], source.data["y0"], kind="cubic"
        )
        xnew = np.linspace(
            source.data["x0"][0],
            source.data["x0"][-1],
            num=len(self.nodes_pos) * 20,
            endpoint=True,
        )

        ynew = interpolated(xnew)
        auxsource = ColumnDataSource(data=dict(x0=xnew, y0=ynew, y1=[0] * len(xnew)))

        disp_graph.line(
            "x0",
            "y0",
            source=auxsource,
            legend="Deformed shaft",
            line_width=3,
            line_color=bokeh_colors[9],
        )
        disp_graph.circle(
            "x0",
            "y0",
            source=source,
            legend="Deformed shaft",
            size=8,
            fill_color=bokeh_colors[9],
        )
        disp_graph.line(
            "x0",
            "y1",
            source=source,
            legend="underformed shaft",
            line_width=3,
            line_color=bokeh_colors[0],
        )
        disp_graph.circle(
            "x0",
            "y1",
            source=source,
            legend="underformed shaft",
            size=8,
            fill_color=bokeh_colors[0],
        )

        # create a new plot for free body diagram (FDB)
        y_range = []
        sh_weight = sum(self.df_shaft["m"].values) * 9.8065
        y_range.append(sh_weight)
        for i, node in enumerate(self.df_bearings["n"]):
            y_range.append(-self.disp_y[node] * self.df_bearings.loc[i, "kyy"].coefficient[0])

        shaft_end = self.nodes_pos[-1]
        FBD = figure(
            tools=TOOLS,
            width=800,
            height=400,
            title="Free-Body Diagram",
            x_axis_label="shaft lenght",
            y_axis_label="Force",
            x_range=[-0.1 * shaft_end, 1.1 * shaft_end],
            y_range=[-max(y_range) * 1.4, max(y_range) * 1.4],
        )
        FBD.xaxis.axis_label_text_font_size = "14pt"
        FBD.yaxis.axis_label_text_font_size = "14pt"

        FBD.line("x0", "y1", source=source, line_width=5, line_color=bokeh_colors[0])

        # FBD - plot arrows indicating shaft weight distribution
        text = str("%.1f" % sh_weight)
        FBD.line(
            x=self.nodes_pos,
            y=[sh_weight] * len(self.nodes_pos),
            line_width=2,
            line_color=bokeh_colors[0],
        )

        for node in self.nodes_pos:
            FBD.add_layout(
                Arrow(
                    end=NormalHead(
                        fill_color=bokeh_colors[7],
                        fill_alpha=1.0,
                        size=16,
                        line_width=2,
                        line_color=bokeh_colors[0],
                    ),
                    x_start=node,
                    y_start=sh_weight,
                    x_end=node,
                    y_end=0,
                )
            )

        FBD.add_layout(
            Label(
                x=self.nodes_pos[0],
                y=sh_weight,
                text="W = " + text + "N",
                text_font_style="bold",
                text_baseline="top",
                text_align="left",
                y_offset=20,
            )
        )

        # FBD - calculate the reaction force of bearings and plot arrows
        for i, node in enumerate(self.df_bearings["n"]):
            Fb = -self.disp_y[node] * self.df_bearings.loc[i, "kyy"].coefficient[0]
            text = str("%.1f" % Fb)
            FBD.add_layout(
                Arrow(
                    end=NormalHead(
                        fill_color=bokeh_colors[7],
                        fill_alpha=1.0,
                        size=16,
                        line_width=2,
                        line_color=bokeh_colors[0],
                    ),
                    x_start=self.nodes_pos[node],
                    y_start=-Fb,
                    x_end=self.nodes_pos[node],
                    y_end=0,
                )
            )
            FBD.add_layout(
                Label(
                    x=self.nodes_pos[node],
                    y=-Fb,
                    text="Fb = " + text + "N",
                    text_font_style="bold",
                    text_baseline="bottom",
                    text_align="center",
                    y_offset=-20,
                )
            )

        # FBD - plot arrows indicating disk weight
        if len(self.df_disks) != 0:
            for i, node in enumerate(self.df_disks["n"]):
                Fd = self.df_disks.loc[i, "m"] * 9.8065
                text = str("%.1f" % Fd)
                FBD.add_layout(
                    Arrow(
                        end=NormalHead(
                            fill_color=bokeh_colors[7],
                            fill_alpha=1.0,
                            size=16,
                            line_width=2,
                            line_color=bokeh_colors[0],
                        ),
                        x_start=self.nodes_pos[node],
                        y_start=Fd,
                        x_end=self.nodes_pos[node],
                        y_end=0,
                    )
                )
                FBD.add_layout(
                    Label(
                        x=self.nodes_pos[node],
                        y=Fd,
                        text="Fd = " + text + "N",
                        text_font_style="bold",
                        text_baseline="top",
                        text_align="center",
                        y_offset=20,
                    )
                )

        # Shearing Force Diagram plot (SF)
        source_SF = ColumnDataSource(data=dict(x=self.Vx_axis, y=self.Vx))
        TOOLTIPS_SF = [("Shearing Force:", "@y")]
        SF = figure(
            tools=TOOLS,
            tooltips=TOOLTIPS_SF,
            width=800,
            height=400,
            title="Shearing Force Diagram",
            x_axis_label="Shaft lenght",
            y_axis_label="Force",
            x_range=[-0.1 * shaft_end, 1.1 * shaft_end],
        )
        SF.xaxis.axis_label_text_font_size = "14pt"
        SF.yaxis.axis_label_text_font_size = "14pt"

        SF.line("x", "y", source=source_SF, line_width=4, line_color=bokeh_colors[0])
        SF.circle("x", "y", source=source_SF, size=8, fill_color=bokeh_colors[0])

        # SF - plot centerline
        SF.line(
            [-0.1 * shaft_end, 1.1 * shaft_end],
            [0, 0],
            line_width=3,
            line_dash="dotdash",
            line_color=bokeh_colors[0],
        )

        # Bending Moment Diagram plot (BM)
        source_BM = ColumnDataSource(data=dict(x=self.nodes_pos, y=self.Bm))
        TOOLTIPS_BM = [("Bending Moment:", "@y")]
        BM = figure(
            tools=TOOLS,
            tooltips=TOOLTIPS_BM,
            width=800,
            height=400,
            title="Bending Moment Diagram",
            x_axis_label="Shaft lenght",
            y_axis_label="Bending Moment",
            x_range=[-0.1 * shaft_end, 1.1 * shaft_end],
        )
        BM.xaxis.axis_label_text_font_size = "14pt"
        BM.yaxis.axis_label_text_font_size = "14pt"

        i = 0
        while True:
            if i + 3 > len(self.nodes):
                break

            interpolated_BM = interpolate.interp1d(
                self.nodes_pos[i : i + 3], self.Bm[i : i + 3], kind="quadratic"
            )
            xnew_BM = np.linspace(self.nodes_pos[i], self.nodes_pos[i + 2], num=42, endpoint=True)

            ynew_BM = interpolated_BM(xnew_BM)
            auxsource_BM = ColumnDataSource(data=dict(x=xnew_BM, y=ynew_BM))
            BM.line(
                "x", "y", source=auxsource_BM, line_width=4, line_color=bokeh_colors[0]
            )
            i += 2
        BM.circle("x", "y", source=source_BM, size=8, fill_color=bokeh_colors[0])

        # BM - plot centerline
        BM.line(
            [-0.1 * shaft_end, 1.1 * shaft_end],
            [0, 0],
            line_width=3,
            line_dash="dotdash",
            line_color=bokeh_colors[0],
        )

        grid_plots = gridplot([[FBD, SF], [disp_graph, BM]])

        show(grid_plots)


class ConvergenceResults:
    def __init__(self, el_num, eigv_arr, error_arr):
        self.el_num = el_num
        self.eigv_arr = eigv_arr
        self.error_arr = error_arr

    def plot(self):
        """This method plots:
            Natural Frequency vs Number of Elements
            Relative Error vs Number of Elements

        Parameters
        ----------

        Returns
        -------
        plot : bokeh.figure
            Bokeh plot showing the results
        --------
        """
        source = ColumnDataSource(
            data=dict(x0=self.el_num, y0=self.eigv_arr, y1=self.error_arr)
        )

        TOOLS = "pan,wheel_zoom,box_zoom,hover,reset,save,"
        TOOLTIPS1 = [("Frequency:", "@y0"), ("Number of Elements", "@x0")]
        TOOLTIPS2 = [("Relative Error:", "@y1"), ("Number of Elements", "@x0")]

        # create a new plot and add a renderer
        freq_arr = figure(
            tools=TOOLS,
            tooltips=TOOLTIPS1,
            width=800,
            height=600,
            title="Frequency Evaluation",
            x_axis_label="Numer of Elements",
            y_axis_label="Frequency (rad/s)",
        )
        freq_arr.xaxis.axis_label_text_font_size = "14pt"
        freq_arr.yaxis.axis_label_text_font_size = "14pt"

        freq_arr.line("x0", "y0", source=source, line_width=3, line_color="crimson")
        freq_arr.circle("x0", "y0", source=source, size=8, fill_color="crimson")

        # create another new plot and add a renderer
        rel_error = figure(
            tools=TOOLS,
            tooltips=TOOLTIPS2,
            width=800,
            height=600,
            title="Relative Error Evaluation",
            x_axis_label="Number of Elements",
            y_axis_label="Relative Error (%)",
        )
        rel_error.xaxis.axis_label_text_font_size = "14pt"
        rel_error.yaxis.axis_label_text_font_size = "14pt"

        rel_error.line(
            "x0", "y1", source=source, line_width=3, line_color="darkslategray"
        )
        rel_error.circle("x0", "y1", source=source, fill_color="darkslategray", size=8)

        # put the subplots in a gridplot
        plot = gridplot([[freq_arr, rel_error]])
        # show the plots
        show(plot)

        return plot


class TimeResponseResults:
    def __init__(self, t, yout, xout, dof):
        self.t = t
        self.yout = yout
        self.xout = xout
        self.dof = dof

    def _plot_matplotlib(self, ax=None):

        if ax is None:
            ax = plt.gca()

        ax.plot(self.t, self.yout[:, self.dof])

        if self.dof % 4 == 0:
            obs_dof = "x"
            amp = "m"
        elif self.dof % 4 == 1:
            obs_dof = "y"
            amp = "m"
        elif self.dof % 4 == 2:
            obs_dof = "\u03B1"  # unicode for alpha
            amp = "rad"
        elif self.dof % 4 == 3:
            obs_dof = "\u03B2"  # unicode for beta
            amp = "rad"

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (%s)" % amp)
        ax.set_title(
            "Response for node %s and degree of freedom %s" % (self.dof // 4, obs_dof)
        )

    def _plot_bokeh(self):

        if self.dof % 4 == 0:
            obs_dof = "x"
            amp = "m"
        elif self.dof % 4 == 1:
            obs_dof = "y"
            amp = "m"
        elif self.dof % 4 == 2:
            obs_dof = "\u03B1"  # unicode for alpha
            amp = "rad"
        elif self.dof % 4 == 3:
            obs_dof = "\u03B2"  # unicode for beta
            amp = "rad"

        # bokeh plot - create a new plot
        bk_ax = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=1200,
            height=900,
            title="Response for node %s and degree of freedom %s" % (self.dof // 4, obs_dof),
            x_axis_label="Time (s)",
            y_axis_label="Amplitude (%s)" % amp,
        )
        bk_ax.xaxis.axis_label_text_font_size = "14pt"
        bk_ax.yaxis.axis_label_text_font_size = "14pt"

        # bokeh plot - plot shaft centerline
        bk_ax.line(
                self.t,
                self.yout[:, self.dof],
                line_width=3,
                line_color=bokeh_colors[0]
        )

        show(bk_ax)

    def plot(self, plot_type="matplotlib", **kwargs):
        """Plot time response.

        This function will take a rotor object and plot its time response
        given a force and a time.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        plot_type: str
            Matplotlib or bokeh.
            The default is matplotlib
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------

        Examples
        --------
        """
        if plot_type == "matplotlib":
            return self._plot_matplotlib(**kwargs)
        elif plot_type == "bokeh":
            return self._plot_bokeh(**kwargs)
        else:
            raise ValueError(f"{plot_type} is not a valid plot type.")
