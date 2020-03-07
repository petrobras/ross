# fmt: off
import bokeh.palettes as bp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from bokeh.colors import RGB
from bokeh.layouts import gridplot, widgetbox
from bokeh.models import (Arrow, ColorBar, ColumnDataSource, HoverTool, Label,
                          NormalHead)
from bokeh.models.widgets import (DataTable, NumberFormatter, Panel,
                                  TableColumn, Tabs)
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

# fmt: on

# set bokeh palette of colors
bokeh_colors = bp.RdGy[11]


class ModalResults:
    def __init__(
        self,
        speed,
        evalues,
        evectors,
        wn,
        wd,
        damping_ratio,
        log_dec,
        lti,
        ndof,
        nodes,
        nodes_pos,
        shaft_elements_length,
    ):
        self.speed = speed
        self.evalues = evalues
        self.evectors = evectors
        self.wn = wn
        self.wd = wd
        self.damping_ratio = damping_ratio
        self.log_dec = log_dec
        self.lti = lti
        self.ndof = ndof
        self.nodes = nodes
        self.nodes_pos = nodes_pos
        self.shaft_elements_length = shaft_elements_length
        self.modes = self.evectors[: self.ndof]
        kappa_modes = []
        for mode in range(len(self.wn)):
            kappa_color = []
            kappa_mode = self.kappa_mode(mode)
            for kappa in kappa_mode:
                kappa_color.append("tab:blue" if kappa > 0 else "tab:red")
            kappa_modes.append(kappa_color)
        self.kappa_modes = kappa_modes

    @staticmethod
    def whirl(kappa_mode):
        """Evaluates the whirl of a mode

       Parameters
       ----------
       kappa_mode: list
           A list with the value of kappa for each node related
           to the mode/natural frequency of interest.

       Returns
       -------
       A string indicating the direction of precession related to the kappa_mode

       Example
       -------
       >>> kappa_mode = [-5.06e-13, -3.09e-13, -2.91e-13, 0.011, -4.03e-13, -2.72e-13, -2.72e-13]
       >>> ModalResults.whirl(kappa_mode)
       'Forward'
       """
        if all(kappa >= -1e-3 for kappa in kappa_mode):
            whirldir = "Forward"
        elif all(kappa <= 1e-3 for kappa in kappa_mode):
            whirldir = "Backward"
        else:
            whirldir = "Mixed"
        return whirldir

    @staticmethod
    @np.vectorize
    def whirl_to_cmap(whirl):
        """Maps the whirl to a value

        Parameters
        ----------
        whirl: string
            A string indicating the whirl direction related to the kappa_mode

        Returns
        -------
        An array with reference index for the whirl direction

        Example
        -------
        >>> whirl = 'Backward'
        >>> whirl_to_cmap(whirl)
        array(1.)
        """
        if whirl == "Forward":
            return 0.0
        elif whirl == "Backward":
            return 1.0
        elif whirl == "Mixed":
            return 0.5

    def H_kappa(self, node, w, return_T=False):
        r"""Calculates the H matrix for a given node and natural frequency.

        The matrix H contains information about the whirl direction,
        the orbit minor and major axis and the orbit inclination.
        The matrix is calculated by :math:`H = T.T^T` where the
        matrix T is constructed using the eigenvector corresponding
        to the natural frequency of interest:

        .. math::
           :nowrap:

           \begin{eqnarray}
              \begin{bmatrix}
              u(t)\\
              v(t)
              \end{bmatrix}
              = \mathfrak{R}\Bigg(
              \begin{bmatrix}
              r_u e^{j\eta_u}\\
              r_v e^{j\eta_v}
              \end{bmatrix}\Bigg)
              e^{j\omega_i t}
              =
              \begin{bmatrix}
              r_u cos(\eta_u + \omega_i t)\\
              r_v cos(\eta_v + \omega_i t)
              \end{bmatrix}
              = {\bf T}
              \begin{bmatrix}
              cos(\omega_i t)\\
              sin(\omega_i t)
              \end{bmatrix}
           \end{eqnarray}

        Where :math:`r_u e^{j\eta_u}` e :math:`r_v e^{j\eta_v}` are the
        elements of the *i*\th eigenvector, corresponding to the node and
        natural frequency of interest (mode).

        .. math::

            {\bf T} =
            \begin{bmatrix}
            r_u cos(\eta_u) & -r_u sin(\eta_u)\\
            r_u cos(\eta_u) & -r_v sin(\eta_v)
            \end{bmatrix}

        Parameters
        ----------
        node: int
            Node for which the matrix H will be calculated.
        w: int
            Index corresponding to the natural frequency
            of interest.
        return_T: bool, optional
            If True, returns the H matrix and a dictionary with the
            values for :math:`r_u, r_v, \eta_u, \eta_v`.

            Default is false.

        Returns
        -------
        H: array
            Matrix H.
        Tdic: dict
            Dictionary with values for :math:`r_u, r_v, \eta_u, \eta_v`.

            It will be returned only if return_T is True.
        """
        # get vector of interest based on freqs
        vector = self.evectors[4 * node : 4 * node + 2, w]
        # get translation sdofs for specified node for each mode
        u = vector[0]
        v = vector[1]
        ru = np.absolute(u)
        rv = np.absolute(v)

        nu = np.angle(u)
        nv = np.angle(v)
        # fmt: off
        T = np.array([[ru * np.cos(nu), -ru * np.sin(nu)],
                      [rv * np.cos(nv), -rv * np.sin(nv)]])
        # fmt: on
        H = T @ T.T

        if return_T:
            Tdic = {"ru": ru, "rv": rv, "nu": nu, "nv": nv}
            return H, Tdic

        return H

    def kappa(self, node, w, wd=True):
        r"""Calculates kappa for a given node and natural frequency.

        frequency is the the index of the natural frequency of interest.
        The function calculates the orbit parameter :math:`\kappa`:

        .. math::

            \kappa = \pm \sqrt{\lambda_2 / \lambda_1}

        Where :math:`\sqrt{\lambda_1}` is the length of the semiminor axes
        and :math:`\sqrt{\lambda_2}` is the length of the semimajor axes.

        If :math:`\kappa = \pm 1`, the orbit is circular.

        If :math:`\kappa` is positive we have a forward rotating orbit
        and if it is negative we have a backward rotating orbit.

        Parameters
        ----------
        node: int
            Node for which kappa will be calculated.
        w: int
            Index corresponding to the natural frequency
            of interest.
        wd: bool
            If True, damping natural frequencies are used.

            Default is true.

        Returns
        -------
        kappa: dict
            A dictionary with values for the natural frequency,
            major axis, minor axis and kappa.
        """
        if wd:
            nat_freq = self.wd[w]
        else:
            nat_freq = self.wn[w]

        H, Tvals = self.H_kappa(node, w, return_T=True)
        nu = Tvals["nu"]
        nv = Tvals["nv"]

        lam = la.eig(H)[0]

        # lam is the eigenvalue -> sqrt(lam) is the minor/major axis.
        # kappa encodes the relation between the axis and the precession.
        minor = np.sqrt(lam.min())
        major = np.sqrt(lam.max())
        kappa = minor / major
        diff = nv - nu

        # we need to evaluate if 0 < nv - nu < pi.
        if diff < -np.pi:
            diff += 2 * np.pi
        elif diff > np.pi:
            diff -= 2 * np.pi

        # if nv = nu or nv = nu + pi then the response is a straight line.
        if diff == 0 or diff == np.pi:
            kappa = 0

        # if 0 < nv - nu < pi, then a backward rotating mode exists.
        elif 0 < diff < np.pi:
            kappa *= -1

        k = {
            "Frequency": nat_freq,
            "Minor axes": np.real(minor),
            "Major axes": np.real(major),
            "kappa": np.real(kappa),
        }

        return k

    def kappa_mode(self, w):
        r"""This function evaluates kappa given the index of
        the natural frequency of interest.
        Values of kappa are evaluated for each node of the
        corresponding frequency mode.

        Parameters
        ----------
        w: int
            Index corresponding to the natural frequency
            of interest.

        Returns
        -------
        kappa_mode: list
            A list with the value of kappa for each node related
            to the mode/natural frequency of interest.
        """
        kappa_mode = [self.kappa(node, w)["kappa"] for node in self.nodes]
        return kappa_mode

    def whirl_direction(self):
        """Get the whirl direction for each frequency.

        Parameters
        ----------

        Returns
        -------
        whirl_w : array
            An array of strings indicating the direction of precession related
            to the kappa_mode. Backward, Mixed or Forward depending on values
            of kappa_mode.
        """
        # whirl direction/values are methods because they are expensive.
        whirl_w = [self.whirl(self.kappa_mode(wd)) for wd in range(len(self.wd))]

        return np.array(whirl_w)

    def whirl_values(self):
        """Get the whirl value (0., 0.5, or 1.) for each frequency.

        Parameters
        ----------

        Returns
        -------
        whirl_to_cmap
            0.0 - if the whirl is Forward
            0.5 - if the whirl is Mixed
            1.0 - if the whirl is Backward
        """
        return self.whirl_to_cmap(self.whirl_direction())

    def calc_mode_shape(self, mode=None, evec=None):
        """
        Method that calculate the arrays describing the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
            Default is None
        evec : array
            Array containing the system eigenvectors

        Returns
        -------
        xn : array
            absolut nodal displacement - X direction
        yn : array
            absolut nodal displacement - Y direction
        zn : array
            absolut nodal displacement - Z direction
        x_circles : array
            orbit description - X direction
        y_circles : array
            orbit description - Y direction
        z_circles_pos : array
            axial location of each orbit
        nn : int
            number of points to plot lines
        """
        evec0 = self.modes[:, mode]
        nodes = self.nodes
        nodes_pos = self.nodes_pos
        shaft_elements_length = self.shaft_elements_length

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
        N4 = -(zeta ** 2) + zeta ** 3

        for Le, n in zip(shaft_elements_length, nodes):
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

        return xn, yn, zn, x_circles, y_circles, z_circles_pos, nn

    def plot_mode3D(self, mode=None, evec=None, fig=None, ax=None):
        """
        Method that plots (3D view) the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
            Default is None
        evec : array
            Array containing the system eigenvectors
        fig : matplotlib figure
            The figure object with the plot.
        ax : matplotlib axes
            The axes object with the plot.

        Returns
        -------
        fig : matplotlib figure
            Returns the figure object with the plot.
        ax : matplotlib axes
            Returns the axes object with the plot.
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection="3d")

        nodes = self.nodes
        kappa_mode = self.kappa_modes[mode]
        xn, yn, zn, xc, yc, zc_pos, nn = self.calc_mode_shape(mode=mode, evec=evec)

        for node in nodes:
            ax.plot(
                xc[10:, node],
                yc[10:, node],
                zc_pos[10:, node],
                color=kappa_mode[node],
                linewidth=0.5,
                zdir="x",
            )
            ax.scatter(
                xc[10, node],
                yc[10, node],
                zc_pos[10, node],
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
            f"$mode$ {mode + 1} - $speed$ = {self.speed:.1f} rad/s\n"
            f"$\omega_n$ = {self.wn[mode]:.1f} rad/s\n"
            f"$log dec$ = {self.log_dec[mode]:.1f}\n"
            f"$whirl\_direction$ = {self.whirl_direction()[mode]}",
            fontsize=18,
        )
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.tick_params(axis="both", which="minor", labelsize=18)

        return fig, ax

    def plot_mode2D(self, mode=None, evec=None, fig=None, ax=None):
        """
        Method that plots (2D view) the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
            Default is None
        evec : array
            Array containing the system eigenvectors
        fig : matplotlib figure
            The figure object with the plot.
        ax : matplotlib axes
            The axes object with the plot.

        Returns
        -------
        fig : matplotlib figure
            Returns the figure object with the plot.
        ax : matplotlib axes
            Returns the axes object with the plot.
        """
        xn, yn, zn, xc, yc, zc_pos, nn = self.calc_mode_shape(mode=mode)
        nodes_pos = self.nodes_pos

        vn = np.zeros(len(zn))
        for i in range(len(zn)):
            theta = np.arctan(xn[i] / yn[i])
            vn[i] = xn[i] * np.sin(theta) + yn[i] * np.cos(theta)

        # remove repetitive values from zn and vn
        idx_remove = []
        for i in range(1, len(zn)):
            if zn[i] == zn[i - 1]:
                idx_remove.append(i)
        zn = np.delete(zn, idx_remove)
        vn = np.delete(vn, idx_remove)

        if fig is None and ax is None:
            fig, ax = plt.subplots()

        colors = dict(Backward="firebrick", Mixed="black", Forward="royalblue")
        whirl_dir = colors[self.whirl_direction()[mode]]

        ax.plot(zn, vn, c=whirl_dir)
        label = (
            f"Mode {mode + 1} | {self.whirl_direction()[mode]} | "
            f"wn = {self.wn[mode]:.1f} rad/s | "
            f"log dec = {self.log_dec[mode]:.1f}"
        )

        mode_shape = mpl.lines.Line2D([], [], lw=0, label=label)

        # plot center line
        zn_cl0 = -(zn[-1] * 0.1)
        zn_cl1 = zn[-1] * 1.1
        ax.plot(nodes_pos, np.zeros(len(nodes_pos)), "k-.", linewidth=0.8)

        ax.set_ylim(-1.3, 1.3)
        ax.set_xlim(zn_cl0, zn_cl1)

        ax.set_title("Mode Shape", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=14)

        legend = plt.legend(handles=[mode_shape], loc=0, framealpha=0.1)
        ax.add_artist(legend)
        ax.set_xlabel("Rotor length")
        ax.set_ylabel("Non dimentional rotor deformation")

        return fig, ax


class CampbellResults:
    """Class used to store results and provide plots for Campbell Diagram.

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
    whirl_values : array
        Array with the whirl values (0, 0.5 or 1)

    Returns
    -------
    ax : matplotlib axes
        Returns the matplotlib axes object with the plot
        if plot_type == "matplotlib"
    bk_ax : bokeh axes
        Returns the bokeh axes object with the plot
        if plot_type == "bokeh"
    """

    def __init__(self, speed_range, wd, log_dec, whirl_values):
        self.speed_range = speed_range
        self.wd = wd
        self.log_dec = log_dec
        self.whirl_values = whirl_values

    def _plot_matplotlib(self, harmonics=[1], fig=None, ax=None, **kwargs):
        """
        Method to create Campbell Diagram figure using Matplotlib library.

        Parameters
        ----------
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        fig : matplotlib figure, optional
            Figure in which the plot will be drawn
            Default is None
        ax : matplotlib plotting axes, optional
            Axes which the plot will take to draw.
            Default is None
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        fig : matplotlib figure
            A figure with the Campbell Diagram plot
        ax : matplotlib plotting axes
            The axes from Campbell Diagram plot
        """
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        wd = self.wd
        num_frequencies = wd.shape[1]
        log_dec = self.log_dec
        whirl = self.whirl_values
        speed_range = np.repeat(
            self.speed_range[:, np.newaxis], num_frequencies, axis=1
        )

        default_values = dict(cmap="RdBu", vmin=0.1, vmax=2.0, s=30, alpha=1.0)
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

                for harm in harmonics:
                    idx = np.argwhere(
                        np.diff(np.sign(w_i - harm * speed_range_i))
                    ).flatten()
                    if len(idx) != 0:
                        idx = idx[0]

                        interpolated = interpolate.interp1d(
                            x=[speed_range_i[idx], speed_range_i[idx + 1]],
                            y=[w_i[idx], w_i[idx + 1]],
                            kind="linear",
                        )
                        xnew = np.linspace(
                            speed_range_i[idx],
                            speed_range_i[idx + 1],
                            num=20,
                            endpoint=True,
                        )
                        ynew = interpolated(xnew)
                        idx = np.argwhere(
                            np.diff(np.sign(ynew - harm * xnew))
                        ).flatten()

                        ax.scatter(xnew[idx], ynew[idx], marker="X", s=30, c="g")

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
            crit_marker = mpl.lines.Line2D(
                [], [], marker="X", lw=0, color="g", alpha=0.3, label="Crit. Speed"
            )
            labels = [forward_label, backward_label, mixed_label, crit_marker]

            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]
            for j, harm in enumerate(harmonics):
                harmonic = ax.plot(
                    speed_range[:, 0],
                    harm * speed_range[:, 0],
                    color=colors[j],
                    linewidth=1.5,
                    linestyle="-.",
                    alpha=0.75,
                    label=str(harm) + "x speed",
                )
                labels.append(harmonic[0])

            legend = plt.legend(handles=labels, loc=2, framealpha=0.1)
            ax.add_artist(legend)

            ax.set_xlabel("Rotor speed ($rad/s$)")
            ax.set_ylabel("Damped natural frequencies ($rad/s$)")

        return fig, ax

    def _plot_bokeh(self, harmonics=[1], **kwargs):
        """
        Method to create Campbell Diagram figure using Bokeh library.

        Parameters
        ----------
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        camp : Bokeh axes
            The bokeh axes object with the plot
        """
        wd = self.wd
        num_frequencies = wd.shape[1]
        log_dec = self.log_dec
        whirl = self.whirl_values
        speed_range = np.repeat(
            self.speed_range[:, np.newaxis], num_frequencies, axis=1
        )

        log_dec_map = log_dec.flatten()

        m_coolwarm_rgb = (255 * cm.coolwarm(range(256))).astype("int")
        coolwarm_palette = [RGB(*tuple(rgb)).to_hex() for rgb in m_coolwarm_rgb][::-1]

        default_values = dict(
            vmin=min(log_dec_map), vmax=max(log_dec_map), s=30, alpha=1.0
        )

        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        camp = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            title="Campbell Diagram - Damped Natural Frequency Map",
            width=640,
            height=480,
            x_axis_label="Rotor speed (rad/s)",
            y_axis_label="Damped natural frequencies (rad/s)",
        )
        camp.xaxis.axis_label_text_font_size = "20pt"
        camp.yaxis.axis_label_text_font_size = "20pt"
        camp.axis.major_label_text_font_size = "16pt"
        camp.title.text_font_size = "14pt"
        hover = False

        color_mapper = linear_cmap(
            field_name="color",
            palette=coolwarm_palette,
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
                    idx = np.argwhere(
                        np.diff(np.sign(w_i - harm * speed_range_i))
                    ).flatten()
                    if len(idx) != 0:
                        idx = idx[0]

                        interpolated = interpolate.interp1d(
                            x=[speed_range_i[idx], speed_range_i[idx + 1]],
                            y=[w_i[idx], w_i[idx + 1]],
                            kind="linear",
                        )
                        xnew = np.linspace(
                            speed_range_i[idx],
                            speed_range_i[idx + 1],
                            num=30,
                            endpoint=True,
                        )
                        ynew = interpolated(xnew)
                        idx = np.argwhere(
                            np.diff(np.sign(ynew - harm * xnew))
                        ).flatten()

                        source = ColumnDataSource(dict(xnew=xnew[idx], ynew=ynew[idx]))
                        camp.asterisk(
                            x="xnew",
                            y="ynew",
                            source=source,
                            size=14,
                            fill_alpha=1.0,
                            color=bokeh_colors[9],
                            muted_color=bokeh_colors[9],
                            muted_alpha=0.2,
                            legend_label="Crit. Speed",
                            name="critspeed",
                        )
                        hover = HoverTool(names=["critspeed"])
                        hover.tooltips = [
                            ("Frequency :", "@xnew"),
                            ("Critical Speed :", "@ynew"),
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
                        legend_label=legend,
                    )

        harm_color = bp.Category20[20]
        for j, harm in enumerate(harmonics):
            camp.line(
                x=speed_range[:, 0],
                y=harm * speed_range[:, 0],
                line_width=3,
                color=harm_color[j],
                line_dash="dotdash",
                line_alpha=1.0,
                legend_label=str(harm) + "x speed",
                muted_color=harm_color[j],
                muted_alpha=0.2,
            )

        # turn legend glyphs black
        camp.scatter(0, 0, color="black", size=0, marker="^", legend_label="Foward")
        camp.scatter(0, 0, color="black", size=0, marker="o", legend_label="Mixed")
        camp.scatter(0, 0, color="black", size=0, marker="v", legend_label="Backward")

        color_bar = ColorBar(
            color_mapper=color_mapper["transform"],
            width=8,
            location=(0, 0),
            title="log dec",
            title_text_font_style="bold italic",
            title_text_font_size="16pt",
            title_text_align="center",
            major_label_text_align="left",
            major_label_text_font_size="16pt",
        )
        if hover:
            camp.add_tools(hover)
        camp.legend.background_fill_alpha = 0.1
        camp.legend.click_policy = "mute"
        camp.legend.location = "top_left"
        camp.legend.label_text_font_size = "16pt"
        camp.add_layout(color_bar, "right")

        return camp

    def plot(self, *args, plot_type="bokeh", **kwargs):
        """Plot campbell results.

        Parameters
        ----------
        args: optional
            harmonics : list, optional
                List with the harmonics to be plotted.
                The default is to plot 1x.
        plot_type: str
            Matplotlib or bokeh.
            The default is bokeh
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax : matplotlib axes
            Returns the matplotlib axes object with the plot
            if plot_type == "matplotlib"
        bk_ax : bokeh axes
            Returns the bokeh axes object with the plot
            if plot_type == "bokeh"
        """
        if plot_type == "matplotlib":
            return self._plot_matplotlib(*args, **kwargs)
        elif plot_type == "bokeh":
            return self._plot_bokeh(*args, **kwargs)
        else:
            raise ValueError(f"")


class FrequencyResponseResults:
    """Class used to store results and provide plots for Frequency Response.

    Two options for plooting are available: Matplotlib and Bokeh. The user
    chooses between them using the attribute plot_type. The default is bokeh

    Parameters
    ----------
    freq_resp : array
        Array with the transfer matrix
    speed_range : array
        Array with the speed range in rad/s.
    magnitude : array
        Array with the frequencies, magnitude (dB) of the frequency
        response for each pair input/output
    phase : array
        Array with the frequencies, phase of the frequency
        response for each pair input/output

    Returns
    -------
    ax : matplotlib axes
        Returns the matplotlib axes object with the plot
        if plot_type == "matplotlib"
    bk_ax : bokeh axes
        Returns the bokeh axes object with the plot
        if plot_type == "bokeh"
    """

    def __init__(self, freq_resp, speed_range, magnitude, phase):
        self.freq_resp = freq_resp
        self.speed_range = speed_range
        self.magnitude = magnitude
        self.phase = phase

    def plot_magnitude_matplotlib(self, inp, out, ax=None, units="mic-pk-pk", **kwargs):
        """Plot frequency response.
        This method plots the frequency response magnitude given an output and
        an input using Matplotlib.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax : matplotlib.axes, optional
            Matplotlib axes to plot the magnitude.
            If None creates a new.
        units : str
            Unit system
            Default is "mic-pk-pk"
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with magnitude plot.
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

    def plot_magnitude_bokeh(self, inp, out, units="mic-pk-pk", **kwargs):
        """Plot frequency response.
        This method plots the frequency response magnitude given an output and
        an input using Bokeh.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        units : str
            Unit system
            Default is "mic-pk-pk"
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        mag_plot : bokeh plot axes
            Bokeh plot axes with magnitude plot.
        """
        frequency_range = self.speed_range
        mag = self.magnitude

        if units == "m":
            y_axis_label = "Amplitude (m)"
        elif units == "mic-pk-pk":
            y_axis_label = "Amplitude ($\mu$ pk-pk)"
        else:
            y_axis_label = "Amplitude (dB)"

        # bokeh plot - create a new plot
        mag_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=640,
            height=240,
            title="Frequency Response - Magnitude",
            x_axis_label="Frequency (rad/s)",
            y_axis_label=y_axis_label,
        )
        mag_plot.xaxis.axis_label_text_font_size = "20pt"
        mag_plot.yaxis.axis_label_text_font_size = "20pt"
        mag_plot.axis.major_label_text_font_size = "16pt"
        mag_plot.title.text_font_size = "14pt"

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
        an input using Matplotlib.

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
        an input using bokeh.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        phase_plot : bokeh plot axes
            Bokeh plot axes with phase plot.
        """
        frequency_range = self.speed_range
        phase = self.phase

        # bokeh plot - create a new plot
        phase_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=640,
            height=240,
            title="Frequency Response - Phase",
            x_axis_label="Frequency (rad/s)",
            y_axis_label="Phase",
        )
        phase_plot.xaxis.axis_label_text_font_size = "20pt"
        phase_plot.yaxis.axis_label_text_font_size = "20pt"
        phase_plot.axis.major_label_text_font_size = "16pt"
        phase_plot.title.text_font_size = "14pt"

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
        an output and an input using Matplotib.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax0 : matplotlib.axes, optional
            Matplotlib axes where the magnitude will be plotted.
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
        an output and an input using Bokeh.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        ax0 : bokeh axes, optional
            Bokeh plot axes where the magnitude will be plotted.
            If None creates a new.
        ax1 : bokeh axes, optional
            Bokeh plot axes where the phase will be plotted.
            If None creates a new.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        grid_plots : bokeh column
            Bokeh column with magnitude and phase plots.
        """
        # bokeh plot axes
        bk_ax0 = self.plot_magnitude_bokeh(inp, out, ax=ax0)
        bk_ax1 = self.plot_phase_bokeh(inp, out, ax=ax1)

        # show the bokeh plot results
        grid_plots = gridplot([[bk_ax0], [bk_ax1]])
        grid_plots

        return grid_plots

    def plot(self, inp, out, *args, plot_type="bokeh", **kwargs):
        """Plot frequency response.
        This method plots the frequency response given
        an output and an input.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        args : optional
            Additional bokeh plot axes or matplolib.axes
        plot_type: str
            Matplotlib or bokeh.
            The default is bokeh
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax0 : matplotlib.axes
            Matplotlib axes with amplitude plot.
            if plot_type == "matplotlib"
        ax1 : matplotlib.axes
            Matplotlib axes with phase plot.
            if plot_type == "matplotlib"
        grid_plots : bokeh column
            Bokeh column with amplitude and phase plot
            if plot_type == "bokeh"
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
    """Class used to store results and provide plots for Unbalance and Forced
    Response analysis.

    Two options for plooting are available: Matplotlib and Bokeh. The user
    chooses between them using the attribute plot_type. The default is bokeh

    Parameters
    ----------
    force_resp : array
        Array with the force response for each node for each frequency
    speed_range : array
        Array with the frequencies
    magnitude : array
        Magnitude (dB) of the frequency response for node for each frequency
    phase : array
        Phase of the frequency response for node for each frequency

    Returns
    -------
    ax0 : matplotlib.axes
        Matplotlib axes with magnitude plot.
        if plot_type == "matplotlib"
    ax1 : matplotlib.axes
        Matplotlib axes with phase plot.
        if plot_type == "matplotlib"
    grid_plots : bokeh column
        Bokeh colum with magnitude and phase plot
        if plot_type == "bokeh"
    """

    def __init__(self, forced_resp, speed_range, magnitude, phase):
        self.forced_resp = forced_resp
        self.speed_range = speed_range
        self.magnitude = magnitude
        self.phase = phase

    def plot_magnitude_matplotlib(self, dof, ax=None, units="m", **kwargs):
        """Plot frequency response.
        This method plots the frequency response magnitude given an output and
        an input using Matplotlib.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        ax : matplotlib.axes, optional
            Matplotlib axes where the magnitude will be plotted.
            If None creates a new.
        units : str
            Units to plot the magnitude ('m' or 'mic-pk-pk')
            Default is 'm'
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with magnitude plot.
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
        an input using Bokeh.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        units : str
            Units to plot the magnitude ('m' or 'mic-pk-pk')
            Default is 'm'
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        mag_plot : bokeh axes
            bokeh axes with magnitude plot
        """
        frequency_range = self.speed_range
        mag = self.magnitude

        if units == "m":
            y_axis_label = "Amplitude (m)"
        elif units == "mic-pk-pk":
            mag = 2 * mag * 1e6
            y_axis_label = "Amplitude (μ pk-pk)"

        # bokeh plot - create a new plot
        mag_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=640,
            height=240,
            title="Forced Response - Magnitude",
            x_axis_label="Frequency (rad/s)",
            x_range=[0, max(frequency_range)],
            y_axis_label=y_axis_label,
        )
        mag_plot.xaxis.axis_label_text_font_size = "20pt"
        mag_plot.yaxis.axis_label_text_font_size = "20pt"
        mag_plot.axis.major_label_text_font_size = "16pt"
        mag_plot.title.text_font_size = "14pt"

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
        an input using Matplotlib.

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
        an input using Bokeh.

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
        """
        frequency_range = self.speed_range
        phase = self.phase

        phase_plot = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=640,
            height=240,
            title="Forced Response - Phase",
            x_axis_label="Frequency (rad/s)",
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
        phase_plot.xaxis.axis_label_text_font_size = "20pt"
        phase_plot.yaxis.axis_label_text_font_size = "20pt"
        phase_plot.axis.major_label_text_font_size = "16pt"
        phase_plot.title.text_font_size = "14pt"

        return phase_plot

    def _plot_matplotlib(self, dof, ax0=None, ax1=None, **kwargs):
        """Plot frequency response.
        This method plots the frequency response magnitude and phase given
        an output and an input using Matplotlib.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        ax0 : matplotlib.axes, optional
            Matplotlib axes where the magnitude will be plotted.
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
            Matplotlib axes with magnitude plot.
        ax1 : matplotlib.axes
            Matplotlib axes with phase plot.
        """
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
        """Plot frequency response.
        This method plots the frequency response magnitude and phase given
        an output and an input using Bokeh.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        grid_plots : bokeh column
            Bokeh colum with magnitude and phase plot
        """
        # bokeh plot axes
        bk_ax0 = self.plot_magnitude_bokeh(dof, **kwargs)
        bk_ax1 = self.plot_phase_bokeh(dof, **kwargs)

        # show the bokeh plot results
        grid_plots = gridplot([[bk_ax0], [bk_ax1]])
        grid_plots

        return grid_plots

    def plot(self, dof, plot_type="bokeh", **kwargs):
        """Plot frequency response.
        This method plots the frequency response given an output and an input.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        plot_type: str
            Matplotlib or bokeh.
            The default is bokeh
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax0 : matplotlib.axes
            Matplotlib axes with magnitude plot.
            if plot_type == "matplotlib"
        ax1 : matplotlib.axes
            Matplotlib axes with phase plot.
            if plot_type == "matplotlib"
        grid_plots : bokeh column
            Bokeh colum with magnitude and phase plot
            if plot_type == "bokeh"
        """
        if plot_type == "matplotlib":
            return self._plot_matplotlib(dof, **kwargs)
        elif plot_type == "bokeh":
            return self._plot_bokeh(dof, **kwargs)
        else:
            raise ValueError(f"{plot_type} is not a valid plot type.")


class StaticResults:
    """Class used to store results and provide plots for Static Analysis.

    This class plots free-body diagram, deformed shaft, shearing
    force diagram and bending moment diagram.

    Parameters
    ----------
    disp_y : array
        shaft displacement in y direction
    Vx : array
        shearing force array
    Bm : array
        bending moment array
    df_shaft : dataframe
        shaft dataframe
    df_disks : dataframe
        disks dataframe
    df_bearings : dataframe
        bearing dataframe
    nodes : list
        list of nodes numbers
    nodes_pos : list
        list of nodes positions
    Vx_axis : array
        X axis for displaying shearing force

    Returns
    -------
    fig : bokeh figures
        Bokeh figure with Static Analysis plots depending on which method
        is called.
    """

    def __init__(
        self,
        disp_y,
        Vx,
        Bm,
        w_shaft,
        disk_forces,
        bearing_forces,
        nodes,
        nodes_pos,
        Vx_axis,
    ):

        self.disp_y = disp_y
        self.Vx = Vx
        self.Bm = Bm
        self.w_shaft = w_shaft
        self.disk_forces = disk_forces
        self.bearing_forces = bearing_forces
        self.nodes = nodes
        self.nodes_pos = nodes_pos
        self.Vx_axis = Vx_axis

    def plot_deformation(self):
        """Plot the shaft static deformation.

        This method plots:
            deformed shaft

        Parameters
        ----------

        Returns
        -------
        fig : bokeh figure
            Bokeh figure with static deformation plot
        """
        # create displacement plot
        fig = figure(
            tools="pan, wheel_zoom, box_zoom, reset, save, box_select",
            width=640,
            height=480,
            title="Deformation",
            x_axis_label="Shaft lenght",
            y_axis_label="Lateral displacement",
        )
        fig.xaxis.axis_label_text_font_size = "16pt"
        fig.yaxis.axis_label_text_font_size = "16pt"
        fig.axis.major_label_text_font_size = "16pt"
        fig.title.text_font_size = "14pt"

        count = 0
        for disp_y, Vx, Bm, nodes, nodes_pos, Vx_axis in zip(
            self.disp_y, self.Vx, self.Bm, self.nodes, self.nodes_pos, self.Vx_axis
        ):
            source = ColumnDataSource(data=dict(x=nodes_pos, y=disp_y))

            interpolated = interpolate.interp1d(
                source.data["x"], source.data["y"], kind="cubic"
            )
            xnew = np.linspace(
                source.data["x"][0],
                source.data["x"][-1],
                num=len(nodes_pos) * 20,
                endpoint=True,
            )

            ynew = interpolated(xnew)
            auxsource = ColumnDataSource(data=dict(x=xnew, y=ynew))

            fig.line(
                "x",
                "y",
                source=auxsource,
                legend_label="Deformed - shaft " + str(count),
                line_width=3,
                line_color=bokeh_colors[9 - count],
                muted_alpha=0.1,
                name="def_l",
            )
            fig.circle(
                "x",
                "y",
                source=source,
                legend_label="Deformed - shaft " + str(count),
                size=8,
                fill_color=bokeh_colors[9 - count],
                muted_alpha=0.1,
                name="def_c",
            )
            hover = HoverTool(names=["def_l", "def_c"])
            hover.tooltips = [
                ("Shaft lenght:", "@x"),
                ("Displacement:", "@y"),
            ]
            count += 1
        fig.add_tools(hover)
        fig.legend.background_fill_alpha = 0.1
        fig.legend.click_policy = "mute"

        return fig

    def plot_free_body_diagram(self):
        """Plot the rotor free-body diagram.

        This method plots:
            free-body diagram.

        Parameters
        ----------

        Returns
        -------
        fig : bokeh figure
            Bokeh figure with the free-body diagram plot
        """
        figures = []
        j = 0
        y_start = 5.0
        for nodes_pos, nodes in zip(self.nodes_pos, self.nodes):
            fig = figure(
                tools="pan, wheel_zoom, box_zoom, reset, save, box_select",
                width=640,
                height=480,
                title="Free-Body Diagram - Shaft " + str(j),
                x_axis_label="Shaft lenght",
                x_range=[-0.1 * nodes_pos[-1], 1.1 * nodes_pos[-1]],
                y_range=[-3 * y_start, 3 * y_start],
            )
            fig.yaxis.visible = False
            fig.xaxis.axis_label_text_font_size = "16pt"
            fig.axis.major_label_text_font_size = "16pt"
            fig.title.text_font_size = "14pt"

            fig.line(
                nodes_pos, np.zeros(len(nodes_pos)), line_width=5, color=bokeh_colors[0]
            )

            # fig - plot arrows indicating shaft weight distribution
            text = str("%.1f" % self.w_shaft[j])
            fig.line(
                x=nodes_pos,
                y=[y_start] * len(nodes_pos),
                line_width=2,
                line_color=bokeh_colors[0],
            )

            ini = nodes_pos[0]
            fin = nodes_pos[-1]
            arrows_list = np.arange(ini, 1.01 * fin, (fin - ini) / 5.0)
            for node in arrows_list:
                fig.add_layout(
                    Arrow(
                        end=NormalHead(
                            fill_color=bokeh_colors[2],
                            fill_alpha=1.0,
                            size=14,
                            line_width=2,
                            line_color=bokeh_colors[0],
                        ),
                        x_start=node,
                        y_start=y_start,
                        x_end=node,
                        y_end=0,
                    )
                )

            fig.add_layout(
                Label(
                    x=nodes_pos[0],
                    y=y_start,
                    text="Weight = " + text + "N",
                    text_font_style="bold",
                    text_font_size="10pt",
                    text_baseline="top",
                    text_align="left",
                    y_offset=20,
                )
            )

            # fig - calculate the reaction force of bearings and plot arrows
            for k, v in self.bearing_forces.items():
                _, node = k.split("_")
                node = int(node)
                if node in nodes:
                    text = str(v)
                    var = 1 if v < 0 else -1
                    fig.add_layout(
                        Arrow(
                            end=NormalHead(
                                fill_color=bokeh_colors[6],
                                fill_alpha=1.0,
                                size=14,
                                line_width=2,
                                line_color=bokeh_colors[0],
                            ),
                            x_start=nodes_pos[nodes.index(node)],
                            y_start=var * 2 * y_start,
                            x_end=nodes_pos[nodes.index(node)],
                            y_end=0,
                        )
                    )
                    fig.add_layout(
                        Label(
                            x=nodes_pos[nodes.index(node)],
                            y=var * 2 * y_start,
                            angle=np.pi / 2,
                            text="Fb = " + text + "N",
                            text_font_style="bold",
                            text_font_size="10pt",
                            text_baseline="top",
                            text_align="center",
                            x_offset=2,
                        )
                    )

            # fig - plot arrows indicating disk weight
            for k, v in self.disk_forces.items():
                _, node = k.split("_")
                node = int(node)
                if node in nodes:
                    text = str(v)
                    fig.add_layout(
                        Arrow(
                            end=NormalHead(
                                fill_color=bokeh_colors[9],
                                fill_alpha=1.0,
                                size=14,
                                line_width=2,
                                line_color=bokeh_colors[0],
                            ),
                            x_start=nodes_pos[nodes.index(node)],
                            y_start=2 * y_start,
                            x_end=nodes_pos[nodes.index(node)],
                            y_end=0,
                        )
                    )
                    fig.add_layout(
                        Label(
                            x=nodes_pos[nodes.index(node)],
                            y=2 * y_start,
                            angle=np.pi / 2,
                            text="Fd = " + text + "N",
                            text_font_style="bold",
                            text_font_size="10pt",
                            text_baseline="top",
                            text_align="center",
                            x_offset=2,
                        )
                    )
            figures.append(fig)
            j += 1
        grid_plots = gridplot([figures])

        return grid_plots

    def plot_shearing_force(self):
        """Plot the rotor shearing force diagram.

        This method plots:
            shearing force diagram.

        Parameters
        ----------

        Returns
        -------
        fig : bokeh figure
            Bokeh figure with the shearing force diagram plot
        """
        shaft_end = max([sublist[-1] for sublist in self.nodes_pos])
        fig = figure(
            tools="pan, wheel_zoom, box_zoom, reset, save, box_select",
            width=640,
            height=480,
            title="Shearing Force Diagram",
            x_axis_label="Shaft lenght",
            y_axis_label="Force",
            x_range=[-0.1 * shaft_end, 1.1 * shaft_end],
        )
        fig.xaxis.axis_label_text_font_size = "16pt"
        fig.yaxis.axis_label_text_font_size = "16pt"
        fig.axis.major_label_text_font_size = "16pt"
        fig.title.text_font_size = "14pt"

        # fig - plot centerline
        fig.line(
            [-0.1 * shaft_end, 1.1 * shaft_end],
            [0, 0],
            line_width=3,
            line_dash="dotdash",
            line_color=bokeh_colors[0],
        )

        j = 0
        for Vx, Vx_axis in zip(self.Vx, self.Vx_axis):
            source_SF = ColumnDataSource(data=dict(x=Vx_axis, y=Vx))

            fig.line(
                "x",
                "y",
                source=source_SF,
                line_width=4,
                line_color=bokeh_colors[9 - j],
                line_alpha=1.0,
                muted_alpha=0.1,
                legend_label="Shaft " + str(j),
                name="line",
            )
            fig.circle(
                "x",
                "y",
                source=source_SF,
                size=8,
                fill_color=bokeh_colors[9 - j],
                fill_alpha=1.0,
                muted_alpha=0.1,
                legend_label="Shaft " + str(j),
                name="circle",
            )
            hover = HoverTool(names=["line", "circle"])
            hover.tooltips = [("Shear Force:", "@y")]
            j += 1
        fig.add_tools(hover)
        fig.legend.background_fill_alpha = 0.1
        fig.legend.click_policy = "mute"

        return fig

    def plot_bending_moment(self):
        """Plot the rotor bending moment diagram.

        This method plots:
            bending moment diagram.

        Parameters
        ----------

        Returns
        -------
        fig : bokeh figure
            Bokeh figure with the bending moment diagram plot
        """
        shaft_end = max([sublist[-1] for sublist in self.nodes_pos])
        fig = figure(
            tools="pan, wheel_zoom, box_zoom, reset, save, box_select",
            width=640,
            height=480,
            title="Bending Moment Diagram",
            x_axis_label="Shaft lenght",
            y_axis_label="Bending Moment",
            x_range=[-0.1 * shaft_end, 1.1 * shaft_end],
        )
        fig.xaxis.axis_label_text_font_size = "20pt"
        fig.yaxis.axis_label_text_font_size = "20pt"
        fig.axis.major_label_text_font_size = "16pt"
        fig.title.text_font_size = "14pt"

        # fig - plot centerline
        fig.line(
            [-0.1 * shaft_end, 1.1 * shaft_end],
            [0, 0],
            line_width=3,
            line_dash="dotdash",
            line_color=bokeh_colors[0],
        )

        j = 0
        for Bm, nodes_pos, nodes in zip(self.Bm, self.nodes_pos, self.nodes):
            source_BM = ColumnDataSource(data=dict(x=nodes_pos, y=Bm))
            i = 0
            while True:
                if i + 3 > len(nodes):
                    break

                interpolated_BM = interpolate.interp1d(
                    nodes_pos[i : i + 3], Bm[i : i + 3], kind="quadratic"
                )
                xnew_BM = np.linspace(
                    nodes_pos[i], nodes_pos[i + 2], num=42, endpoint=True
                )

                ynew_BM = interpolated_BM(xnew_BM)
                auxsource_BM = ColumnDataSource(data=dict(x=xnew_BM, y=ynew_BM))
                fig.line(
                    "x",
                    "y",
                    source=auxsource_BM,
                    line_width=4,
                    line_color=bokeh_colors[9 - j],
                    line_alpha=1.0,
                    muted_alpha=0.1,
                    legend_label="Shaft " + str(j),
                    name="line",
                )
                i += 2

            fig.circle(
                "x",
                "y",
                source=source_BM,
                size=8,
                fill_color=bokeh_colors[9 - j],
                fill_alpha=1.0,
                muted_alpha=0.1,
                legend_label="Shaft " + str(j),
                name="circle",
            )
            hover = HoverTool(names=["line", "circle"])
            hover.tooltips = [("Beinding Moment:", "@y")]
            j += 1

        fig.add_tools(hover)
        fig.legend.background_fill_alpha = 0.1
        fig.legend.click_policy = "mute"

        return fig


class SummaryResults:
    """Class used to store results and provide plots rotor summary.

    This class aims to present a summary of the main parameters and attributes
    from a rotor model. The data is presented in a table format.

    Parameters
    ----------
    df_shaft: dataframe
        shaft dataframe
    df_disks: dataframe
        disks dataframe
    df_bearings: dataframe
        bearings dataframe
    brg_forces: list
        list of reaction forces on bearings
    nodes_pos:  list
        list of nodes axial position
    CG: float
        rotor center of gravity
    Ip: float
        rotor total moment of inertia around the center line
    tag: str
        rotor's tag

    Returns
    -------
    table : bokeh WidgetBox
        Bokeh WidgetBox with the summary table plot
    """

    def __init__(
        self, df_shaft, df_disks, df_bearings, nodes_pos, brg_forces, CG, Ip, tag
    ):
        self.df_shaft = df_shaft
        self.df_disks = df_disks
        self.df_bearings = df_bearings
        self.brg_forces = brg_forces
        self.nodes_pos = np.array(nodes_pos)
        self.CG = CG
        self.Ip = Ip
        self.tag = tag

    def plot(self):
        """Plot the summary table.

        This method plots:
            Table with summary of rotor parameters and attributes

        Parameters
        ----------

        Returns
        -------
        tabs : bokeh WidgetBox
            Bokeh WidgetBox with the summary table plot
        """
        materials = [mat.name for mat in self.df_shaft["material"]]

        shaft_data = dict(
            tags=self.df_shaft["tag"],
            sh_number=self.df_shaft["shaft_number"],
            lft_stn=self.df_shaft["n_l"],
            rgt_stn=self.df_shaft["n_r"],
            elem_no=self.df_shaft["_n"],
            beam_left_loc=self.df_shaft["nodes_pos_l"],
            elem_len=self.df_shaft["L"],
            beam_cg=self.df_shaft["beam_cg"],
            axial_cg_pos=self.df_shaft["axial_cg_pos"],
            beam_right_loc=self.df_shaft["nodes_pos_r"],
            material=materials,
            mass=self.df_shaft["m"],
            inertia=self.df_shaft["Im"],
        )

        rotor_data = dict(
            tag=[self.tag],
            starting_node=[self.df_shaft["n_l"].iloc[0]],
            ending_node=[self.df_shaft["n_r"].iloc[-1]],
            starting_point=[self.df_shaft["nodes_pos_r"].iloc[0]],
            total_lenght=[self.df_shaft["nodes_pos_r"].iloc[-1]],
            CG=[self.CG],
            Ip=[self.Ip],
            total_mass=[np.sum(self.df_shaft["m"])],
        )

        disk_data = dict(
            tags=self.df_disks["tag"],
            sh_number=self.df_disks["shaft_number"],
            disk_node=self.df_disks["n"],
            disk_pos=self.nodes_pos[self.df_bearings["n"]],
            disk_mass=self.df_disks["m"],
            disk_Ip=self.df_disks["Ip"],
        )

        bearing_data = dict(
            tags=self.df_bearings["tag"],
            sh_number=self.df_bearings["shaft_number"],
            brg_node=self.df_bearings["n"],
            brg_pos=self.nodes_pos[self.df_bearings["n"]],
            brg_force=list(self.brg_forces.values()),
        )

        shaft_source = ColumnDataSource(shaft_data)
        rotor_source = ColumnDataSource(rotor_data)
        disk_source = ColumnDataSource(disk_data)
        bearing_source = ColumnDataSource(bearing_data)

        shaft_titles = [
            "Element Tag",
            "Shaft Number",
            "Left Station",
            "Right Station",
            "Element Number",
            "Elem. Left Location",
            "Elem. Lenght",
            "Element CG",
            "Axial CG Location",
            "Elem. Right Location",
            "Material",
            "Elem. Mass",
            "Inertia",
        ]

        rotor_titles = [
            "Tag",
            "First Station",
            "Last Station",
            "Starting Pos.",
            "Total Lenght",
            "C.G. Locantion",
            "Total Ip about C.L.",
        ]

        disk_titles = [
            "Tag",
            "Shaft Number",
            "Disk Station",
            "C.G. Locantion",
            "Disk Mass",
            "Total Ip about C.L.",
        ]

        bearing_titles = [
            "Tag",
            "Shaft Number",
            "Bearing Station",
            "Bearing Locantion",
            "Static Reaction Force",
        ]

        shaft_formatters = [
            None,
            None,
            None,
            None,
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
            None,
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.0000000"),
        ]

        rotor_formatters = [
            None,
            None,
            None,
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
        ]

        disk_formatters = [
            None,
            None,
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
        ]

        bearing_formatters = [
            None,
            None,
            NumberFormatter(format="0.000"),
            NumberFormatter(format="0.000"),
        ]

        shaft_columns = [
            TableColumn(field=str(field), title=title, formatter=form)
            for field, title, form in zip(
                shaft_data.keys(), shaft_titles, shaft_formatters
            )
        ]

        rotor_columns = [
            TableColumn(field=str(field), title=title, formatter=form)
            for field, title, form in zip(
                rotor_data.keys(), rotor_titles, rotor_formatters
            )
        ]

        disk_columns = [
            TableColumn(field=str(field), title=title, formatter=form)
            for field, title, form in zip(
                disk_data.keys(), disk_titles, disk_formatters
            )
        ]

        bearing_columns = [
            TableColumn(field=str(field), title=title, formatter=form)
            for field, title, form in zip(
                bearing_data.keys(), bearing_titles, bearing_formatters
            )
        ]

        shaft_data_table = DataTable(
            source=shaft_source, columns=shaft_columns, width=1600
        )
        rotor_data_table = DataTable(
            source=rotor_source, columns=rotor_columns, width=1600
        )
        disk_data_table = DataTable(
            source=disk_source, columns=disk_columns, width=1600
        )
        bearing_data_table = DataTable(
            source=bearing_source, columns=bearing_columns, width=1600
        )

        rotor_table = widgetbox(rotor_data_table)
        tab1 = Panel(child=rotor_table, title="Rotor Summary")

        shaft_table = widgetbox(shaft_data_table)
        tab2 = Panel(child=shaft_table, title="Shaft Summary")

        disk_table = widgetbox(disk_data_table)
        tab3 = Panel(child=disk_table, title="Disk Summary")

        bearing_table = widgetbox(bearing_data_table)
        tab4 = Panel(child=bearing_table, title="Bearing Summary")

        tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])

        return tabs


class ConvergenceResults:
    """Class used to store results and provide plots for Convergence Analysis.

    This class plots:
        Natural Frequency vs Number of Elements
        Relative Error vs Number of Elements

    Parameters
    ----------
    el_num : array
        Array with number of elements in each iteraction
    eigv_arr : array
        Array with the n'th natural frequency in each iteraction
    error_arr : array
        Array with the relative error in each iteraction

    Returns
    -------
    plot : bokeh.gridplot
        Bokeh column with Convergence Analysis plots
    """

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
        plot : bokeh.gridplot
            Bokeh column with Convergence Analysis plots
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
            width=640,
            height=480,
            title="Frequency Evaluation",
            x_axis_label="Numer of Elements",
            y_axis_label="Frequency (rad/s)",
        )
        freq_arr.xaxis.axis_label_text_font_size = "20pt"
        freq_arr.yaxis.axis_label_text_font_size = "20pt"
        freq_arr.axis.major_label_text_font_size = "16pt"
        freq_arr.title.text_font_size = "14pt"

        freq_arr.line("x0", "y0", source=source, line_width=3, line_color="crimson")
        freq_arr.circle("x0", "y0", source=source, size=8, fill_color="crimson")

        # create another new plot and add a renderer
        rel_error = figure(
            tools=TOOLS,
            tooltips=TOOLTIPS2,
            width=640,
            height=480,
            title="Relative Error Evaluation",
            x_axis_label="Number of Elements",
            y_axis_label="Relative Error (%)",
        )
        rel_error.xaxis.axis_label_text_font_size = "20pt"
        rel_error.yaxis.axis_label_text_font_size = "20pt"
        rel_error.axis.major_label_text_font_size = "16pt"
        rel_error.title.text_font_size = "14pt"

        rel_error.line(
            "x0", "y1", source=source, line_width=3, line_color="darkslategray"
        )
        rel_error.circle("x0", "y1", source=source, fill_color="darkslategray", size=8)

        # put the subplots in a gridplot
        plot = gridplot([[freq_arr, rel_error]])

        return plot


class TimeResponseResults:
    """Class used to store results and provide plots for Time Response
    Analysis.

    This class takes the results from time response analysis and creates a
    plot given a force and a time.

    Parameters
    ----------
    t : array
        Time values for the output.
    yout : array
        System response.
    xout : array
        Time evolution of the state vector.
    dof : int
        Degree of freedom

    Returns
    -------
    ax : matplotlib.axes
        Matplotlib axes with time response plot.
        if plot_type == "matplotlib"
    bk_ax : bokeh axes
        Bokeh axes with time response plot
        if plot_type == "bokeh"
    """

    def __init__(self, t, yout, xout, dof):
        self.t = t
        self.yout = yout
        self.xout = xout
        self.dof = dof

    def _plot_matplotlib(self, ax=None):
        """Plot time response.

        This function will take a rotor object and plot its time response
        using Matplotlib

        Parameters
        ----------
        ax : matplotlib.axes
            Matplotlib axes where time response will be plotted.
            if None, creates a new one

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with time response plot.
        """
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
        """Plot time response.

        This function will take a rotor object and plot its time response
        using Bokeh

        Parameters
        ----------

        Returns
        -------
        bk_ax : bokeh axes
            Bokeh axes with time response plot
            if plot_type == "bokeh"
        """
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
            width=640,
            height=480,
            title="Response for node %s and degree of freedom %s"
            % (self.dof // 4, obs_dof),
            x_axis_label="Time (s)",
            y_axis_label="Amplitude (%s)" % amp,
        )
        bk_ax.xaxis.axis_label_text_font_size = "20pt"
        bk_ax.yaxis.axis_label_text_font_size = "20pt"
        bk_ax.axis.major_label_text_font_size = "16pt"
        bk_ax.title.text_font_size = "14pt"

        bk_ax.line(
            self.t, self.yout[:, self.dof], line_width=3, line_color=bokeh_colors[0]
        )

        return bk_ax

    def plot(self, plot_type="bokeh", **kwargs):
        """Plot time response.

        This function will take a rotor object and plot its time response

        Parameters
        ----------
        plot_type: str
            Matplotlib or bokeh.
            The default is bokeh
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with time response plot.
            if plot_type == "matplotlib"
        bk_ax : bokeh axes
            Bokeh axes with time response plot
            if plot_type == "bokeh"
        """
        if plot_type == "matplotlib":
            return self._plot_matplotlib(**kwargs)
        elif plot_type == "bokeh":
            return self._plot_bokeh(**kwargs)
        else:
            raise ValueError(f"{plot_type} is not a valid plot type.")


class OrbitResponseResults:
    """Class used to store results and provide plots for Orbit Response
    Analysis.

    This class takes the results from orbit response analysis and creates a
    plot (2D or 3D) given a force array and a time array.

    Parameters
    ----------
    t: array
        Time values for the output.
    yout: array
        System response.
    xout: array
        Time evolution of the state vector.
    nodes_list: array
        list with nodes from a rotor model
    nodes_pos: array
        Rotor nodes axial positions

    Returns
    -------
    ax : matplotlib.axes
        Matplotlib axes with orbit response plot.
        if plot_type == "3d"
    bk_ax : bokeh axes
        Bokeh axes with orbit response plot
        if plot_type == "2d"
    """

    def __init__(self, t, yout, xout, nodes_list, nodes_pos):
        self.t = t
        self.yout = yout
        self.xout = xout
        self.nodes_pos = nodes_pos
        self.nodes_list = nodes_list

    def _plot3d(self, fig=None, ax=None):
        """Plot orbit response.

        This function will take a rotor object and plot its orbit response
        using Matplotlib

        Parameters
        ----------
        fig : matplotlib figure
            The figure object with the plot.
            if None, creates a new one
        ax : matplotlib.axes
            Matplotlib axes where orbit response will be plotted.
            if None, creates a new one

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with orbit response plot.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection="3d")

        for n in self.nodes_list:
            z_pos = np.ones(self.yout.shape[0]) * self.nodes_pos[n]
            ax.plot(
                self.yout[200:, 4 * n],
                self.yout[200:, 4 * n + 1],
                z_pos[200:],
                zdir="x",
                color="k",
            )

        # plot center line
        line = np.zeros(len(self.nodes_pos))
        ax.plot(line, line, self.nodes_pos, "k-.", linewidth=1.5, zdir="x")

        ax.set_xlabel("Rotor length (m)", labelpad=20, fontsize=18)
        ax.set_ylabel("Amplitude - X direction (m)", labelpad=20, fontsize=18)
        ax.set_zlabel("Amplitude - Y direction (m)", labelpad=20, fontsize=18)
        ax.set_title("Rotor Orbits", fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.tick_params(axis="both", which="minor", labelsize=18)

        return ax

    def _plot2d(self, node):
        """Plot orbit response.

        This function will take a rotor object and plot its orbit response
        using Bokeh

        Parameters
        ----------
        node: int, optional
            Selected node to plot orbit.

        Returns
        -------
        bk_ax : bokeh axes
            Bokeh axes with orbit response plot
            if plot_type == "bokeh"
        """
        # bokeh plot - create a new plot
        bk_ax = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=640,
            height=480,
            title="Response for node %s" % (node),
            x_axis_label="Amplitude - X direction (m)",
            y_axis_label="Amplitude - Y direction (m)",
        )
        bk_ax.xaxis.axis_label_text_font_size = "20pt"
        bk_ax.yaxis.axis_label_text_font_size = "20pt"
        bk_ax.title.text_font_size = "14pt"

        bk_ax.line(
            self.yout[:, 4 * node],
            self.yout[:, 4 * node + 1],
            line_width=3,
            line_color=bokeh_colors[0],
        )

        return bk_ax

    def plot(self, plot_type="3d", node=None, **kwargs):
        """Plot orbit response.

        This function will take a rotor object and plot its orbit response

        Parameters
        ----------
        plot_type: str
            3d or 2d.
            Choose between plotting orbit for all nodes (3d plot) and
            plotting orbit for a single node (2d plot).
            Default is 3d.
        node: int, optional
            Selected node to plot orbit.
            Fill this attribute only when selection plot_type = "2d".
            Detault is None
        kwargs : optional
            Additional key word arguments can be passed to change
            the plot (e.g. linestyle='--')

        Returns
        -------
        ax : matplotlib.axes
            Matplotlib axes with time response plot.
            if plot_type == "3d"
        bk_ax : bokeh axes
            Bokeh axes with time response plot
            if plot_type == "2d"
        """
        if plot_type == "3d":
            return self._plot3d(**kwargs)
        elif plot_type == "2d":
            if node is None:
                raise Exception("Select a node to plot orbit when plotting 2D")
            elif node not in self.nodes_list:
                raise Exception("Select a valid node to plot 2D orbit")
            return self._plot2d(node=node, **kwargs)
        else:
            raise ValueError(f"{plot_type} is not a valid plot type.")
