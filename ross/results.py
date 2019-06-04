import pickle

import bokeh.palettes as bp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.plotting import figure, output_file, show
from bokeh.transform import linear_cmap

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


class CampbellResults(Results):
    def plot(self, harmonics=[1], wn=False, fig=None, ax=None, **kwargs):
        """Plot campbell results.
        Parameters
        ----------
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        fig : matplotlib figure, optional
            Figure to insert axes with log_dec colorbar.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        """

        # results for campbell is an array with [speed_range, wd/log_dec/whirl]

        if fig is None and ax is None:
            fig, ax = plt.subplots()

        wd = self[..., 0]
        if wn is True:
            wn = self[..., 4]
        log_dec = self[..., 1]
        whirl = self[..., 2]
        speed_range = self[..., 3]

        default_values = dict(cmap="RdBu", vmin=0.1, vmax=2.0, s=20, alpha=0.5)
        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        # bokeh plot - output to static HTML file
        output_file("Campbell_diagram.html")

        log_dec_map = log_dec.flatten()

        # bokeh plot - create a new plot
        camp = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=900,
            height=500,
            title="Campbell Diagram",
            x_axis_label="Rotor speed (rad/s)",
            y_axis_label="Damped natural frequencies (rad/s)",
        )

        for mark, whirl_dir, legend in zip(
            ["^", "o", "v"], [0.0, 0.5, 1.0], ["Foward", "Mixed", "Backward"]
        ):
            num_frequencies = wd.shape[1]
            for i in range(num_frequencies):
                if wn is True:
                    w_i = wn[:, i]
                else:
                    w_i = wd[:, i]
                whirl_i = whirl[:, i]
                log_dec_i = log_dec[:, i]
                speed_range_i = speed_range[:, i]

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

                    # Bokeh plot
                    source = ColumnDataSource(
                        dict(
                            x=speed_range_i[whirl_mask],
                            y=w_i[whirl_mask],
                            color=log_dec_i[whirl_mask],
                        )
                    )
                    color_mapper = linear_cmap(
                        field_name="color",
                        palette=bp.magma(256),
                        low=min(log_dec_map),
                        high=max(log_dec_map),
                    )
                    camp.scatter(
                        x="x",
                        y="y",
                        color=color_mapper,
                        marker=mark,
                        fill_alpha=1.0,
                        size=5,
                        muted_color=color_mapper,
                        muted_alpha=0.2,
                        source=source,
                        legend=legend,
                    )

        ax.plot(
            speed_range[:, 0],
            speed_range[:, 0],
            color="k",
            linewidth=1.5,
            linestyle="-.",
            alpha=0.75,
            label="Rotor speed",
        )

        camp.line(
            x=speed_range[:, 0],
            y=speed_range[:, 0],
            line_width=3,
            color=bokeh_colors[0],
            line_dash="dotdash",
            line_alpha=0.75,
            legend="Rotor speed",
            muted_color=bokeh_colors[0],
            muted_alpha=0.2,
        )

        color_bar = ColorBar(
            color_mapper=color_mapper["transform"], width=8, location=(0, 0)
        )

        camp.legend.click_policy = "mute"
        camp.legend.location = "top_left"
        camp.add_layout(color_bar, "right")
        show(camp)

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


class FrequencyResponseResults(Results):
    def plot_magnitude(self, inp, out, ax=None, units="m", **kwargs):
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

        frequency_range = self.frequency_range
        mag = self.magnitude

        ax.plot(frequency_range, mag[inp, out, :], **kwargs)

        ax.set_xlim(0, max(frequency_range))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="upper"))

        if units == "m":
            ax.set_ylabel("Amplitude $(m)$")
            y_axis_label = "Amplitude (m)"
        elif units == "mic-pk-pk":
            ax.set_ylabel("Amplitude $(\mu pk-pk)$")
            y_axis_label = "Amplitude (\mu pk-pk)"
        else:
            ax.set_ylabel("Amplitude $(dB)$")
            y_axis_label = "Amplitude (dB"

        ax.set_xlabel("Frequency (rad/s)")

        # bokeh plot - create a new plot
        mag_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=900,
            height=400,
            title="Frequency Response - Magnitude",
            x_axis_label="Frequency",
            y_axis_label=y_axis_label,
        )
        source = ColumnDataSource(dict(x=frequency_range, y=mag[inp, out, :]))
        mag_plot.line(
            x="x",
            y="y",
            source=source,
            line_color=bokeh_colors[0],
            line_alpha=1.0,
            line_width=3,
        )

        return ax, mag_plot

    def plot_phase(self, inp, out, ax=None, **kwargs):
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

        frequency_range = self.frequency_range
        phase = self.phase

        ax.plot(frequency_range, phase[inp, out, :], **kwargs)

        ax.set_xlim(0, max(frequency_range))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="upper"))

        ax.set_ylabel("Phase")
        ax.set_xlabel("Frequency (rad/s)")

        # bokeh plot - create a new plot
        phase_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=900,
            height=400,
            title="Frequency Response - Phase",
            x_axis_label="Frequency",
            y_axis_label="Phase",
        )
        source = ColumnDataSource(dict(x=frequency_range, y=phase[inp, out, :]))
        phase_plot.line(
            x="x",
            y="y",
            source=source,
            line_color=bokeh_colors[0],
            line_alpha=1.0,
            line_width=3,
        )

        return ax, phase_plot

    def plot(self, inp, out, ax0=None, ax1=None, **kwargs):
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

        # bokeh plot - output to static HTML file
        output_file("plot_freq_response.html")

        # matplotlib axes
        ax0 = self.plot_magnitude(inp, out, ax=ax0)[0]
        ax1 = self.plot_phase(inp, out, ax=ax1)[0]

        # bokeh plot axes
        bk_ax0 = self.plot_magnitude(inp, out, ax=ax0)[1]
        bk_ax1 = self.plot_phase(inp, out, ax=ax1)[1]

        ax0.set_xlabel("")

        # show the bokeh plot results
        grid_plots = gridplot([[bk_ax0], [bk_ax1]])
        show(grid_plots)

        return ax0, ax1, bk_ax0, bk_ax1

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


class ForcedResponseResults(Results):
    def plot_magnitude(self, dof, ax=None, units="m", **kwargs):
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
        mag_plot : bokeh axes
            bokeh axes with magnitude plot
        Examples
        --------
        """
        if ax is None:
            ax = plt.gca()

        frequency_range = self.frequency_range
        mag = self.magnitude

        if units == "m":
            ax.set_ylabel("Amplitude $(m)$")
            y_axis_label = "Amplitude (m)"
        elif units == "mic-pk-pk":
            mag = 2 * mag * 1e6
            ax.set_ylabel("Amplitude $(\mu pk-pk)$")
            y_axis_label = "Amplitude $(\mu pk-pk)$"

        # matplotlib plotting
        ax.plot(frequency_range, mag[dof], **kwargs)

        ax.set_xlim(0, max(frequency_range))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="upper"))

        ax.set_xlabel("Frequency (rad/s)")
        ax.legend()

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
        source = ColumnDataSource(dict(x=frequency_range, y=mag[dof]))
        mag_plot.line(
            x="x",
            y="y",
            source=source,
            line_color=bokeh_colors[0],
            line_alpha=1.0,
            line_width=3,
        )

        return ax, mag_plot

    def plot_phase(self, dof, ax=None, **kwargs):
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
        phase_plot : bokeh axes
            Bokeh axes with phase plot
        Examples
        --------
        """
        if ax is None:
            ax = plt.gca()

        frequency_range = self.frequency_range
        phase = self.phase

        ax.plot(frequency_range, phase[dof], **kwargs)

        ax.set_xlim(0, max(frequency_range))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="lower"))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune="upper"))

        ax.set_ylabel("Phase")
        ax.set_xlabel("Frequency (rad/s)")
        ax.legend()

        # bokeh plot - create a new plot
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

        return ax, phase_plot

    def plot(self, dof, ax0=None, ax1=None, **kwargs):
        """Plot frequency response.
        This method plots the frequency response given
        an output and an input.
        Parameters
        ----------
        dof : int
            Degree of freedom.
        ax0 : matplotlib.axes, optional
            Matplotlib axes where the amplitude will be plotted.
            If None creates a new.
        ax1 : matplotlib.axes, optional
            Matplotlib axes where the phase will be plotted.
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
        bk_ax0 : bokeh axes
            Bokeh axes with amplitude plot.
        bk_ax1 : bokeh axes
            Bokeh axes with phase plot.
        Examples
        --------
        """
        if ax0 is None and ax1 is None:
            fig, (ax0, ax1) = plt.subplots(2)

        ax0 = self.plot_magnitude(dof, ax=ax0, **kwargs)[0]
        # remove label from phase plot
        kwargs.pop("label", None)
        kwargs.pop("units", None)
        ax1 = self.plot_phase(dof, ax=ax1, **kwargs)[0]

        ax0.set_xlabel("")
        ax0.legend()

        # bokeh plot axes
        bk_ax0 = self.plot_magnitude(dof, ax=ax0, **kwargs)[1]
        bk_ax1 = self.plot_phase(dof, ax=ax1, **kwargs)[1]

        # show the bokeh plot results
        grid_plots = gridplot([[bk_ax0], [bk_ax1]])
        show(grid_plots)

        return ax0, ax1, bk_ax0, bk_ax1


class ModeShapeResults(Results):
    def plot(self, mode=None, evec=None, fig=None, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection="3d")

        evec0 = self[:, mode]
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
            f"\frequency_range_d$ = {self.wd[mode]:.1f} rad/s\n"
            f"$log dec$ = {self.log_dec[mode]:.1f}"
        )

        return fig, ax
