"""ROSS plotting module.

This module returns graphs for each type of analyses in rotor_assembly.py.
"""
import copy

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import linalg as la

from ross.plotly_theme import tableau_colors
from ross.units import Q_
from ross.utils import intersection


class CriticalSpeedResults:
    """Class used to store results from run_critical_speed() method.

    Parameters
    ----------
    wn : array
        Undamped critical speeds array.
    wd : array
        Undamped critical speeds array.
    log_dec : array
        Logarithmic decrement for each critical speed.
    damping_ratio : array
        Damping ratio for each critical speed.
    """

    def __init__(self, wn, wd, log_dec, damping_ratio):
        self.wn = wn
        self.wd = wd
        self.log_dec = log_dec
        self.damping_ratio = damping_ratio


class ModalResults:
    """Class used to store results and provide plots for Modal Analysis.

    Two options for plottting are available: plot_mode3D (mode shape 3D view)
    and plot_mode2D (mode shape 2D view). The user chooses between them using
    the respective methods.

    Parameters
    ----------
    speed : float
        Rotor speed.
    evalues : array
        Eigenvalues array.
    evectors : array
        Eigenvectors array.
    wn : array
        Undamped natural frequencies array.
    wd : array
        Damped natural frequencies array.
    log_dec : array
        Logarithmic decrement for each mode.
    damping_ratio : array
        Damping ratio for each mode.
    lti : StateSpaceContinuous
        Space State Continuos with A, B, C and D matrices.
    ndof : int
        Number of degrees of freedom.
    nodes : list
        List of nodes number.
    nodes_pos : list
        List of nodes positions.
    shaft_elements_length : list
        List with Rotor shaft elements lengths.
    """

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
                kappa_color.append("blue" if kappa > 0 else "red")
            kappa_modes.append(kappa_color)
        self.kappa_modes = kappa_modes

    @staticmethod
    def whirl(kappa_mode):
        """Evaluate the whirl of a mode.

        Parameters
        ----------
        kappa_mode : list
            A list with the value of kappa for each node related
            to the mode/natural frequency of interest.

        Returns
        -------
        whirldir : str
            A string indicating the direction of precession related to the
            kappa_mode.

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
        """Map the whirl to a value.

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
        r"""Calculate the H matrix for a given node and natural frequency.

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
        r"""Calculate kappa for a given node and natural frequency.

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
        r"""Evaluate kappa values.

        This function evaluates kappa given the index of the natural frequency
        of interest.
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
        r"""Get the whirl direction for each frequency.

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
        r"""Get the whirl value (0., 0.5, or 1.) for each frequency.

        Returns
        -------
        whirl_to_cmap
            0.0 - if the whirl is Forward
            0.5 - if the whirl is Mixed
            1.0 - if the whirl is Backward
        """
        return self.whirl_to_cmap(self.whirl_direction())

    def calc_mode_shape(self, mode=None, evec=None):
        r"""Calculate the arrays describing the mode shapes.

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
        if evec is None:
            evec = self.modes[:, mode]
        nodes = self.nodes
        nodes_pos = self.nodes_pos
        shaft_elements_length = self.shaft_elements_length

        modex = evec[0::4]
        modey = evec[1::4]

        xmax, ixmax = max(abs(modex)), np.argmax(abs(modex))
        ymax, iymax = max(abs(modey)), np.argmax(abs(modey))

        if ymax > 0.4 * xmax:
            evec /= modey[iymax]
        else:
            evec /= modex[ixmax]

        modex = evec[0::4]
        modey = evec[1::4]

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

            xn[pos0:pos1] = Nx @ evec[xx].real
            yn[pos0:pos1] = Ny @ evec[yy].real
            zn[pos0:pos1] = (node_pos * onn + Le * zeta).reshape(nn)

        return xn, yn, zn, x_circles, y_circles, z_circles_pos, nn

    def plot_mode_3d(
        self, mode=None, evec=None, fig=None, frequency_units="rad/s", **kwargs
    ):
        """Plot (3D view) the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
            Default is None
        evec : array
            Array containing the system eigenvectors
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        frequency_units : str, optional
            Frequency units that will be used in the plot title.
            Default is rad/s.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if fig is None:
            fig = go.Figure()

        nodes = self.nodes
        kappa_mode = self.kappa_modes[mode]
        xn, yn, zn, xc, yc, zc_pos, nn = self.calc_mode_shape(mode=mode, evec=evec)

        for node in nodes:
            fig.add_trace(
                go.Scatter3d(
                    x=zc_pos[10:, node],
                    y=xc[10:, node],
                    z=yc[10:, node],
                    mode="lines",
                    line=dict(color=kappa_mode[node]),
                    name="node {}".format(node),
                    showlegend=False,
                    hovertemplate=(
                        "Nodal Position: %{x:.2f}<br>"
                        + "X - Relative Displacement: %{y:.2f}<br>"
                        + "Y - Relative Displacement: %{z:.2f}"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[zc_pos[10, node]],
                    y=[xc[10, node]],
                    z=[yc[10, node]],
                    mode="markers",
                    marker=dict(color=kappa_mode[node]),
                    name="node {}".format(node),
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=zn,
                y=xn,
                z=yn,
                mode="lines",
                line=dict(color="black", dash="dash"),
                name="mode shape",
                showlegend=False,
            )
        )

        # plot center line
        zn_cl0 = -(zn[-1] * 0.1)
        zn_cl1 = zn[-1] * 1.1
        zn_cl = np.linspace(zn_cl0, zn_cl1, 30)
        fig.add_trace(
            go.Scatter3d(
                x=zn_cl,
                y=zn_cl * 0,
                z=zn_cl * 0,
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                hoverinfo="none",
                showlegend=False,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=dict(text="Rotor Length"), autorange="reversed", nticks=5
                ),
                yaxis=dict(
                    title=dict(text="Relative Displacement"), range=[-2, 2], nticks=5
                ),
                zaxis=dict(
                    title=dict(text="Relative Displacement"), range=[-2, 2], nticks=5
                ),
            ),
            title=dict(
                text=(
                    f"Mode {mode + 1} | "
                    f"whirl: {self.whirl_direction()[mode]} | "
                    f"ω<sub>n</sub> = {Q_(self.wn[mode], 'rad/s').to(frequency_units).m:.1f} {frequency_units} | "
                    f"log dec = {self.log_dec[mode]:.1f}"
                )
            ),
            **kwargs,
        )

        return fig

    def plot_mode_2d(
        self, mode=None, evec=None, fig=None, frequency_units="rad/s", **kwargs
    ):
        """Plot (2D view) the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
            Default is None
        evec : array
            Array containing the system eigenvectors
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        frequency_units : str, optional
            Frequency units that will be used in the plot title.
            Default is rad/s.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        xn, yn, zn, xc, yc, zc_pos, nn = self.calc_mode_shape(mode=mode, evec=evec)
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

        if fig is None:
            fig = go.Figure()

        colors = dict(Backward="red", Mixed="black", Forward="blue")
        whirl_dir = colors[self.whirl_direction()[mode]]

        fig.add_trace(
            go.Scatter(
                x=zn,
                y=vn,
                mode="lines",
                line=dict(color=whirl_dir),
                name="mode shape",
                showlegend=False,
            )
        )
        # plot center line
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=np.zeros(len(nodes_pos)),
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                name="centerline",
                hoverinfo="none",
                showlegend=False,
            )
        )

        fig.update_xaxes(title_text="Rotor Length")
        fig.update_yaxes(title_text="Relative Displacement")
        fig.update_layout(
            title=dict(
                text=(
                    f"Mode {mode + 1} | "
                    f"whirl: {self.whirl_direction()[mode]} | "
                    f"ωn = {Q_(self.wn[mode], 'rad/s').to(frequency_units).m:.1f} {frequency_units} | "
                    f"log dec = {self.log_dec[mode]:.1f}"
                )
            ),
            **kwargs,
        )

        return fig


class CampbellResults:
    """Class used to store results and provide plots for Campbell Diagram.

    It's possible to visualize multiples harmonics in a single plot to check
    other speeds which also excite a specific natural frequency.

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
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    """

    def __init__(self, speed_range, wd, log_dec, whirl_values):
        self.speed_range = speed_range
        self.wd = wd
        self.log_dec = log_dec
        self.whirl_values = whirl_values

    def plot(self, harmonics=[1], frequency_units="rad/s", fig=None, **kwargs):
        """Create Campbell Diagram figure using Plotly.

        Parameters
        ----------
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        wd = Q_(self.wd, "rad/s").to(frequency_units).m
        num_frequencies = wd.shape[1]
        log_dec = self.log_dec
        whirl = self.whirl_values
        speed_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        if fig is None:
            fig = go.Figure()

        default_values = dict(
            coloraxis_cmin=0.0,
            coloraxis_cmax=1.0,
            coloraxis_colorscale="rdbu",
            coloraxis_colorbar=dict(title=dict(text="<b>Log Dec</b>", side="right")),
        )
        for k, v in default_values.items():
            kwargs.setdefault(k, v)

        scatter_marker = ["triangle-up", "circle", "triangle-down"]
        for mark, whirl_dir, legend in zip(
            scatter_marker, [0.0, 0.5, 1.0], ["Foward", "Mixed", "Backward"]
        ):
            for i in range(num_frequencies):
                w_i = wd[:, i]
                whirl_i = whirl[:, i]
                log_dec_i = log_dec[:, i]

                for harm in harmonics:
                    x1 = speed_range
                    y1 = w_i
                    x2 = speed_range
                    y2 = harm * speed_range

                    x, y = intersection(x1, y1, x2, y2)

                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="markers",
                            marker=dict(symbol="x", color="black"),
                            name="Crit. Speed",
                            legendgroup="Crit. Speed",
                            showlegend=False,
                            hovertemplate=(
                                f"Frequency ({frequency_units}): %{{x:.2f}}<br>Critical Speed ({frequency_units}): %{{y:.2f}}"
                            ),
                        )
                    )

                whirl_mask = whirl_i == whirl_dir
                if whirl_mask.shape[0] == 0:
                    continue
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=speed_range[whirl_mask],
                            y=w_i[whirl_mask],
                            marker=dict(
                                symbol=mark,
                                color=log_dec_i[whirl_mask],
                                coloraxis="coloraxis",
                            ),
                            mode="markers",
                            name=legend,
                            legendgroup=legend,
                            showlegend=False,
                            hoverinfo="none",
                        )
                    )

        for j, h in enumerate(harmonics):
            fig.add_trace(
                go.Scatter(
                    x=speed_range,
                    y=h * speed_range,
                    mode="lines",
                    line=dict(dash="dashdot", color=list(tableau_colors)[j]),
                    name="{}x speed".format(h),
                    hoverinfo="none",
                )
            )
        # turn legend glyphs black
        scatter_marker = ["triangle-up", "circle", "triangle-down", "x"]
        legends = ["Foward", "Mixed", "Backward", "Crit. Speed"]
        for mark, legend in zip(scatter_marker, legends):
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    name=legend,
                    legendgroup=legend,
                    marker=dict(symbol=mark, color="black"),
                )
            )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(speed_range), np.max(speed_range)],
            exponentformat="none",
        )
        fig.update_yaxes(
            title_text=f"Natural Frequencies ({frequency_units})",
            range=[0, 1.1 * np.max(wd)],
        )
        fig.update_layout(
            legend=dict(
                itemsizing="constant",
                orientation="h",
                xanchor="center",
                x=0.5,
                yanchor="bottom",
                y=-0.3,
            ),
            **kwargs,
        )

        return fig


class FrequencyResponseResults:
    """Class used to store results and provide plots for Frequency Response.

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
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with Amplitude vs Frequency and Phase vs Frequency plots.
    """

    def __init__(self, freq_resp, speed_range, magnitude, phase):
        self.freq_resp = freq_resp
        self.speed_range = speed_range
        self.magnitude = magnitude
        self.phase = phase

    def plot_magnitude(
        self,
        inp,
        out,
        frequency_units="rad/s",
        amplitude_units="m/N",
        fig=None,
        **mag_kwargs,
    ):
        """Plot frequency response (magnitude) using Plotly.

        This method plots the frequency response magnitude given an output and
        an input using Plotly.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Default is "m/N"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        mag_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        frequency_range = self.speed_range
        mag = self.magnitude

        frequency_range = Q_(frequency_range, "rad/s").to(frequency_units).m
        mag = Q_(mag, "m/N").to(amplitude_units).m

        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=frequency_range,
                y=mag[inp, out, :],
                mode="lines",
                line=dict(color=tableau_colors["blue"]),
                name="Amplitude",
                legendgroup="Amplitude",
                showlegend=False,
                hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
            )
        )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(frequency_range), np.max(frequency_range)],
        )
        fig.update_yaxes(title_text=f"Amplitude ({amplitude_units})")
        fig.update_layout(**mag_kwargs)

        return fig

    def plot_phase(
        self,
        inp,
        out,
        frequency_units="rad/s",
        phase_units="rad",
        fig=None,
        **phase_kwargs,
    ):
        """Plot frequency response (phase) using Plotly.

        This method plots the frequency response phase given an output and
        an input using Plotly.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        phase_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        frequency_range = self.speed_range
        phase = self.phase[inp, out, :]

        frequency_range = Q_(frequency_range, "rad/s").to(frequency_units).m
        phase = Q_(phase, "rad").to(phase_units).m

        if phase_units in ["rad", "radian", "radians"]:
            phase = [i + 2 * np.pi if i < 0 else i for i in phase]
        else:
            phase = [i + 360 if i < 0 else i for i in phase]

        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=frequency_range,
                y=phase,
                mode="lines",
                line=dict(color=tableau_colors["blue"]),
                name="Phase",
                legendgroup="Phase",
                showlegend=False,
                hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Phase: %{{y:.2e}}",
            )
        )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(frequency_range), np.max(frequency_range)],
        )
        fig.update_yaxes(title_text=f"Phase ({phase_units})")
        fig.update_layout(**phase_kwargs)

        return fig

    def plot_polar_bode(
        self,
        inp,
        out,
        frequency_units="rad/s",
        amplitude_units="m/N",
        phase_units="rad",
        fig=None,
        **polar_kwargs,
    ):
        """Plot frequency response (polar) using Plotly.

        This method plots the frequency response (polar graph) given an output and
        an input using Plotly.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Default is "m/N"
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        polar_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        frequency_range = self.speed_range
        mag = self.magnitude[inp, out, :]
        phase = self.phase[inp, out, :]

        frequency_range = Q_(frequency_range, "rad/s").to(frequency_units).m
        mag = Q_(mag, "m/N").to(amplitude_units).m
        phase = Q_(phase, "rad").to(phase_units).m

        if fig is None:
            fig = go.Figure()

        if phase_units in ["rad", "radian", "radians"]:
            polar_theta_unit = "radians"
            phase = [i + 2 * np.pi if i < 0 else i for i in phase]
        else:
            polar_theta_unit = "degrees"
            phase = [i + 360 if i < 0 else i for i in phase]

        fig.add_trace(
            go.Scatterpolar(
                r=mag,
                theta=phase,
                customdata=frequency_range,
                thetaunit=polar_theta_unit,
                mode="lines+markers",
                marker=dict(color=tableau_colors["blue"]),
                line=dict(color=tableau_colors["blue"]),
                name="Polar_plot",
                legendgroup="Polar",
                showlegend=False,
                hovertemplate=f"Amplitude ({amplitude_units}): %{{r:.2e}}<br>Phase: %{{theta:.2f}}<br>Frequency ({frequency_units}): %{{customdata:.2f}}",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title=dict(text=f"Amplitude ({amplitude_units})"),
                    exponentformat="power",
                ),
                angularaxis=dict(thetaunit=polar_theta_unit),
            ),
            **polar_kwargs,
        )

        return fig

    def plot(
        self,
        inp,
        out,
        frequency_units="rad/s",
        amplitude_units="m/N",
        phase_units="rad",
        mag_kwargs=None,
        phase_kwargs=None,
        polar_kwargs=None,
        subplot_kwargs=None,
    ):
        """Plot frequency response.

        This method plots the frequency response given an output and an input
        using Plotly.

        This method returns a subplot with:
            - Frequency vs Amplitude;
            - Frequency vs Phase Angle;
            - Polar plot Amplitude vs Phase Angle;

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Default is "m/N"
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        mag_kwargs : optional
            Additional key word arguments can be passed to change the magnitude plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        phase_kwargs : optional
            Additional key word arguments can be passed to change the phase plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        polar_kwargs : optional
            Additional key word arguments can be passed to change the polar plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        subplot_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...). This kwargs override "mag_kwargs",
            "phase_kwargs" and "polar_kwargs" dictionaries.
            *See Plotly Python make_subplots Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            Plotly figure with Amplitude vs Frequency and Phase vs Frequency and
            polar Amplitude vs Phase plots.
        """
        mag_kwargs = {} if mag_kwargs is None else copy.copy(mag_kwargs)
        phase_kwargs = {} if phase_kwargs is None else copy.copy(phase_kwargs)
        polar_kwargs = {} if polar_kwargs is None else copy.copy(polar_kwargs)
        subplot_kwargs = {} if subplot_kwargs is None else copy.copy(subplot_kwargs)

        fig0 = self.plot_magnitude(
            inp, out, frequency_units, amplitude_units, **mag_kwargs
        )
        fig1 = self.plot_phase(inp, out, frequency_units, phase_units, **phase_kwargs)
        fig2 = self.plot_polar_bode(
            inp, out, frequency_units, amplitude_units, phase_units, **polar_kwargs
        )

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
            polar=dict(
                radialaxis=fig2.layout.polar.radialaxis,
                angularaxis=fig2.layout.polar.angularaxis,
            ),
            **subplot_kwargs,
        )

        return subplots


class ForcedResponseResults:
    """Store results and provide plots for Unbalance and Forced Response analysis.

    Parameters
    ----------
    rotor : ross.Rotor object
        The Rotor object
    force_resp : array
        Array with the force response for each node for each frequency.
    speed_range : array
        Array with the frequencies.
    magnitude : array
        Magnitude (dB) of the frequency response for node for each frequency.
    phase : array
        Phase of the frequency response for node for each frequency.
    unbalance : array, optional
        Array with the unbalance data (node, magnitude and phase) to be plotted
        with deflected shape. This argument is set only if running an unbalance
        response analysis.
        Default is None.

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with Amplitude vs Frequency and Phase vs Frequency plots.
    """

    def __init__(
        self, rotor, forced_resp, speed_range, magnitude, phase, unbalance=None
    ):
        self.rotor = rotor
        self.forced_resp = forced_resp
        self.speed_range = speed_range
        self.magnitude = magnitude
        self.phase = phase
        self.unbalance = unbalance

    def plot_magnitude(
        self,
        probe,
        probe_units="rad",
        frequency_units="rad/s",
        amplitude_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot forced response (magnitude) using Plotly.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle).
            node : int
                indicate the node where the probe is located.
            orientation : float,
                probe orientation angle about the shaft. The 0 refers to +X direction.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Default is "m"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        frequency_range = Q_(self.speed_range, "rad/s").to(frequency_units).m
        magnitude = self.magnitude
        number_dof = self.rotor.number_dof

        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            dofx = p[0] * number_dof
            dofy = p[0] * number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )

            probe_resp = operator @ np.vstack((magnitude[dofx], magnitude[dofy]))
            z = np.sqrt((probe_resp[0] * np.cos(angle)) ** 2 +
                        (probe_resp[1] * np.sin(angle)) ** 2)
            # fmt: on

            z = Q_(z, "m").to(amplitude_units).m

            fig.add_trace(
                go.Scatter(
                    x=frequency_range,
                    y=z,
                    mode="lines",
                    line=dict(color=list(tableau_colors)[i]),
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(frequency_range), np.max(frequency_range)],
        )
        fig.update_yaxes(
            title_text=f"Amplitude ({amplitude_units})", exponentformat="power"
        )
        fig.update_layout(**kwargs)

        return fig

    def plot_phase(
        self,
        probe,
        probe_units="rad",
        frequency_units="rad/s",
        phase_units="rad",
        fig=None,
        **kwargs,
    ):
        """Plot forced response (phase) using Plotly.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle).
            node : int
                indicate the node where the probe is located.
            orientation : float,
                probe orientation angle about the shaft. The 0 refers to +X direction.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        frequency_range = Q_(self.speed_range, "rad/s").to(frequency_units).m
        number_dof = self.rotor.number_dof

        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            probe_phase = Q_(self.phase[p[0] * number_dof], "rad").to(phase_units).m

            if phase_units in ["rad", "radian", "radians"]:
                probe_phase = np.array(
                    [i + 2 * np.pi if i < 0 else i for i in probe_phase]
                )
            else:
                probe_phase = np.array([i + 360 if i < 0 else i for i in probe_phase])

            angle = Q_(p[1], probe_units).to(phase_units).m
            probe_phase = probe_phase - angle

            fig.add_trace(
                go.Scatter(
                    x=frequency_range,
                    y=probe_phase,
                    mode="lines",
                    line=dict(color=list(tableau_colors)[i]),
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Phase ({phase_units}): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(frequency_range), np.max(frequency_range)],
        )
        fig.update_yaxes(title_text=f"Phase ({phase_units})")
        fig.update_layout(**kwargs)

        return fig

    def plot_polar_bode(
        self,
        probe,
        probe_units="rad",
        frequency_units="rad/s",
        amplitude_units="m",
        phase_units="rad",
        fig=None,
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
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Default is "m"
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        frequency_range = Q_(self.speed_range, "rad/s").to(frequency_units).m
        number_dof = self.rotor.number_dof

        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            dofx = p[0] * number_dof
            dofy = p[0] * number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array([
                [np.cos(angle), - np.sin(angle)],
                [np.cos(angle), + np.sin(angle)],
            ])
            # fmt: on

            resp = operator @ np.vstack((self.magnitude[dofx], self.magnitude[dofy]))
            probe_mag = (
                (resp[0] * np.cos(angle)) ** 2 + (resp[1] * np.sin(angle)) ** 2
            ) ** 0.5

            probe_mag = Q_(probe_mag, "m").to(amplitude_units).m

            phase = Q_(self.phase[p[0] * number_dof], "rad").to(phase_units).m

            if phase_units in ["rad", "radian", "radians"]:
                polar_theta_unit = "radians"
                phase = np.array([i + 2 * np.pi if i < 0 else i for i in phase])
            else:
                phase = np.array([i + 360 if i < 0 else i for i in phase])
                polar_theta_unit = "degrees"

            angle = Q_(p[1], probe_units).to(phase_units).m
            phase = phase - angle

            fig.add_trace(
                go.Scatterpolar(
                    r=probe_mag,
                    theta=phase,
                    customdata=frequency_range,
                    thetaunit=polar_theta_unit,
                    mode="lines+markers",
                    marker=dict(color=list(tableau_colors)[i]),
                    line=dict(color=list(tableau_colors)[i]),
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Amplitude ({amplitude_units}): %{{r:.2e}}<br>Phase: %{{theta:.2f}}<br>Frequency ({frequency_units}): %{{customdata:.2f}}",
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title=dict(text=f"Amplitude ({amplitude_units})"),
                    exponentformat="power",
                ),
                angularaxis=dict(thetaunit=polar_theta_unit),
            ),
            **kwargs,
        )

        return fig

    def plot(
        self,
        probe,
        probe_units="rad",
        frequency_units="rad/s",
        amplitude_units="m",
        phase_units="rad",
        mag_kwargs=None,
        phase_kwargs=None,
        polar_kwargs=None,
        subplot_kwargs=None,
    ):
        """Plot forced response.

        This method returns a subplot with:
            - Frequency vs Amplitude;
            - Frequency vs Phase Angle;
            - Polar plot Amplitude vs Phase Angle;

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle).
            node : int
                indicate the node where the probe is located.
            orientation : float,
                probe orientation angle about the shaft. The 0 refers to +X direction.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        amplitude_units : str, optional
            Amplitude units.
            Default is "m/N"
        phase_units : str, optional
            Phase units.
            Default is "rad"
        mag_kwargs : optional
            Additional key word arguments can be passed to change the magnitude plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        phase_kwargs : optional
            Additional key word arguments can be passed to change the phase plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        polar_kwargs : optional
            Additional key word arguments can be passed to change the polar plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        subplot_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...). This kwargs override "mag_kwargs" and
            "phase_kwargs" dictionaries.
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            Plotly figure with Amplitude vs Frequency and Phase vs Frequency and
            polar Amplitude vs Phase plots.
        """
        mag_kwargs = {} if mag_kwargs is None else copy.copy(mag_kwargs)
        phase_kwargs = {} if phase_kwargs is None else copy.copy(phase_kwargs)
        polar_kwargs = {} if polar_kwargs is None else copy.copy(polar_kwargs)
        subplot_kwargs = {} if subplot_kwargs is None else copy.copy(subplot_kwargs)

        # fmt: off
        fig0 = self.plot_magnitude(
            probe, probe_units, frequency_units, amplitude_units, **mag_kwargs
        )
        fig1 = self.plot_phase(
            probe, probe_units, frequency_units, phase_units, **phase_kwargs
        )
        fig2 = self.plot_polar_bode(
            probe, probe_units, frequency_units, amplitude_units, phase_units, **polar_kwargs
        )
        # fmt: on

        subplots = make_subplots(
            rows=2, cols=2, specs=[[{}, {"type": "polar", "rowspan": 2}], [{}, None]]
        )
        for data in fig0["data"]:
            data.showlegend = False
            subplots.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            data.showlegend = False
            subplots.add_trace(data, row=2, col=1)
        for data in fig2["data"]:
            subplots.add_trace(data, row=1, col=2)

        subplots.update_xaxes(fig0.layout.xaxis, row=1, col=1)
        subplots.update_yaxes(fig0.layout.yaxis, row=1, col=1)
        subplots.update_xaxes(fig1.layout.xaxis, row=2, col=1)
        subplots.update_yaxes(fig1.layout.yaxis, row=2, col=1)
        subplots.update_layout(
            polar=dict(
                radialaxis=fig2.layout.polar.radialaxis,
                angularaxis=fig2.layout.polar.angularaxis,
            ),
            **subplot_kwargs,
        )

        return subplots

    def _calculate_major_axis(self, speed, frequency_units="rad/s"):
        """Calculate the major axis for each nodal orbit.

        Parameters
        ----------
        speed : optional, float
            The rotor rotation speed.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"

        Returns
        -------
        major_axis_vector : np.ndarray
            major_axis_vector[0, :] = foward vector
            major_axis_vector[1, :] = backward vector
            major_axis_vector[2, :] = major axis angle
            major_axis_vector[3, :] = major axis vector for the maximum major axis angle
            major_axis_vector[4, :] = absolute values for major axes vectors
        """
        forced_resp = self.forced_resp
        nodes = self.rotor.nodes
        number_dof = self.rotor.number_dof

        speed = Q_(speed, frequency_units).to("rad/s").m
        major_axis_vector = np.zeros((5, len(nodes)), dtype=complex)
        idx = np.where(np.isclose(self.speed_range, speed, atol=1e-6))[0][0]

        for i, n in enumerate(nodes):
            dofx = number_dof * n
            dofy = number_dof * n + 1

            # Relative angle between probes (90°)
            Rel_ang = np.exp(1j * np.pi / 2)

            # Foward and Backward vectors
            fow = forced_resp[dofx, idx] / 2 + Rel_ang * forced_resp[dofy, idx] / 2
            back = (
                np.conj(forced_resp[dofx, idx]) / 2
                + Rel_ang * np.conj(forced_resp[dofy, idx]) / 2
            )

            ang_fow = np.angle(fow)
            if ang_fow < 0:
                ang_fow += 2 * np.pi

            ang_back = np.angle(back)
            if ang_back < 0:
                ang_back += 2 * np.pi

            # Major axis angles
            ang_maj_ax = (ang_back - ang_fow) / 2

            # Adjusting points to the same quadrant
            if ang_maj_ax > np.pi:
                ang_maj_ax -= np.pi
            if ang_maj_ax > np.pi / 2:
                ang_maj_ax -= np.pi / 2

            major_axis_vector[0, i] = fow
            major_axis_vector[1, i] = back
            major_axis_vector[2, i] = ang_maj_ax

        max_major_axis_angle = np.max(major_axis_vector[2])

        # fmt: off
        major_axis_vector[3] = (
            major_axis_vector[0] * np.exp(1j * max_major_axis_angle) +
            major_axis_vector[1] * np.exp(-1j * max_major_axis_angle)
        )
        major_axis_vector[4] = np.abs(major_axis_vector[3])
        # fmt: on

        return major_axis_vector

    def _calculate_bending_moment(self, speed, frequency_units="rad/s"):
        """Calculate the bending moment in X and Y directions.

        This method calculate forces and moments on nodal positions for a deflected
        shape configuration.

        Parameters
        ----------
        speed : float
            The rotor rotation speed.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"

        Returns
        -------
        Mx : array
            Bending Moment on X directon.
        My : array
            Bending Moment on Y directon.
        """
        speed = Q_(speed, frequency_units).to("rad/s").m
        idx = np.where(np.isclose(self.speed_range, speed, atol=1e-6))[0][0]
        mag = self.magnitude[:, idx]
        phase = self.phase[:, idx]
        number_dof = self.rotor.number_dof
        ndof = self.rotor.ndof

        disp = np.zeros(ndof)
        for i in range(number_dof):
            disp[i::number_dof] = mag[i::number_dof] * np.cos(-phase[i::number_dof])

        nodal_forces = self.rotor.K(speed) @ disp

        Mx = np.cumsum(nodal_forces[2::number_dof])
        My = np.cumsum(nodal_forces[3::number_dof])

        return Mx, My

    def plot_deflected_shape_2d(
        self,
        speed,
        frequency_units="rad/s",
        displacement_units="m",
        rotor_length_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot the 2D deflected shape diagram.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        displacement_units : str, optional
            Displacement units.
            Default is 'm'.
        rotor_length_units : str, optional
            Displacement units.
            Default is 'm'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the deflected shape
            plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        speed = Q_(speed, frequency_units).to("rad/s").m
        if not any(np.isclose(self.speed_range, speed, atol=1e-6)):
            raise ValueError("No data available for this speed value.")

        nodes_pos = Q_(self.rotor.nodes_pos, "m").to(rotor_length_units).m
        maj_vect = self._calculate_major_axis(speed=speed)
        maj_vect = Q_(maj_vect[4].real, "m").to(displacement_units).m

        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=maj_vect,
                mode="lines",
                name="Major Axis",
                legendgroup="Major_Axis_2d",
                showlegend=False,
                hovertemplate=f"Nodal Position ({rotor_length_units}): %{{x:.2f}}<br>Amplitude ({displacement_units}): %{{y:.2e}}",
            )
        )
        # plot center line
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=np.zeros(len(nodes_pos)),
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        fig.update_xaxes(title_text=f"Rotor Length ({rotor_length_units})")
        fig.update_yaxes(
            title_text=f"Major Axis Absolute Amplitude ({displacement_units})"
        )
        fig.update_layout(**kwargs)

        return fig

    def plot_deflected_shape_3d(
        self,
        speed,
        samples=101,
        frequency_units="rad/s",
        displacement_units="m",
        rotor_length_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot the 3D deflected shape diagram.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class.
        samples : int, optional
            Number of samples to generate the orbit for each node.
            Default is 101.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        displacement_units : str, optional
            Displacement units.
            Default is 'm'.
        rotor_length_units : str, optional
            Rotor Length units.
            Default is 'm'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the deflected shape
            plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        speed = Q_(speed, frequency_units).to("rad/s").m
        if not any(np.isclose(self.speed_range, speed, atol=1e-6)):
            raise ValueError("No data available for this speed value.")

        mag = self.magnitude
        phase = self.phase
        ub = self.unbalance
        nodes = self.rotor.nodes
        nodes_pos = Q_(self.rotor.nodes_pos, "m").to(rotor_length_units).m
        number_dof = self.rotor.number_dof
        idx = np.where(np.isclose(self.speed_range, speed, atol=1e-6))[0][0]

        # orbit of a single revolution
        t = np.linspace(0, 2 * np.pi / speed, samples)
        x_pos = np.repeat(nodes_pos, t.size).reshape(len(nodes_pos), t.size)

        if fig is None:
            fig = go.Figure()

        for i, n in enumerate(nodes):
            dofx = number_dof * n
            dofy = number_dof * n + 1

            y = mag[dofx, idx] * np.cos(speed * t - phase[dofx, idx])
            z = mag[dofy, idx] * np.cos(speed * t - phase[dofy, idx])

            # plot nodal orbit
            fig.add_trace(
                go.Scatter3d(
                    x=x_pos[n],
                    y=Q_(y, "m").to(displacement_units).m,
                    z=Q_(z, "m").to(displacement_units).m,
                    mode="lines",
                    line=dict(color="royalblue"),
                    name="Orbit",
                    legendgroup="Orbit",
                    showlegend=False,
                    hovertemplate=(
                        f"Position ({rotor_length_units}): %{{x:.2f}}<br>X - Amplitude ({displacement_units}): %{{y:.2e}}<br>Y - Amplitude ({displacement_units}): %{{z:.2e}}"
                    ),
                )
            )

        # plot major axis
        maj_vect = self._calculate_major_axis(speed)

        fig.add_trace(
            go.Scatter3d(
                x=x_pos[:, 0],
                y=Q_(np.real(maj_vect[3]), "m").to(displacement_units).m,
                z=Q_(np.imag(maj_vect[3]), "m").to(displacement_units).m,
                mode="lines+markers",
                marker=dict(color="black"),
                line=dict(color="black", dash="dashdot"),
                name="Major Axis",
                legendgroup="Major_Axis",
                showlegend=True,
                hovertemplate=(
                    f"Position ({rotor_length_units}): %{{x:.2f}}<br>X - Amplitude ({displacement_units}): %{{y:.2e}}<br>Y - Amplitude ({displacement_units}): %{{z:.2e}}"
                ),
            )
        )

        # plot center line
        line = np.zeros(len(nodes_pos))
        fig.add_trace(
            go.Scatter3d(
                x=nodes_pos,
                y=line,
                z=line,
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        # plot unbalance markers
        i = 0
        for n, m, p in zip(ub[0], ub[1], ub[2]):
            fig.add_trace(
                go.Scatter3d(
                    x=[x_pos[int(n), 0], x_pos[int(n), 0]],
                    y=Q_([0, np.amax(np.abs(maj_vect[4])) / 2 * np.cos(p)], "m")
                    .to(displacement_units)
                    .m,
                    z=Q_([0, np.amax(np.abs(maj_vect[4])) / 2 * np.sin(p)], "m")
                    .to(displacement_units)
                    .m,
                    mode="lines",
                    line=dict(color="firebrick"),
                    legendgroup="Unbalance",
                    hoverinfo="none",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[x_pos[int(n), 0]],
                    y=Q_([np.amax(np.abs(maj_vect[4])) / 2 * np.cos(p)], "m")
                    .to(displacement_units)
                    .m,
                    z=Q_([np.amax(np.abs(maj_vect[4])) / 2 * np.sin(p)], "m")
                    .to(displacement_units)
                    .m,
                    mode="markers",
                    marker=dict(symbol="diamond", color="firebrick"),
                    name="Unbalance",
                    legendgroup="Unbalance",
                    showlegend=True if i == 0 else False,
                    hovertemplate=(
                        "Node: {}<br>" + "Magnitude: {:.2e}<br>" + "Phase: {:.2f}"
                    ).format(int(n), m, p),
                )
            )
            i += 1

        speed_str = Q_(speed, "rad/s").to(frequency_units).m
        fig.update_layout(
            scene=dict(
                xaxis=dict(title=dict(text=f"Rotor Length ({rotor_length_units})")),
                yaxis=dict(title=dict(text=f"Amplitude - X ({displacement_units})")),
                zaxis=dict(title=dict(text=f"Amplitude - Y ({displacement_units})")),
            ),
            title=dict(
                text=f"Deflected Shape<br>Speed = {speed_str} {frequency_units}"
            ),
            **kwargs,
        )

        return fig

    def plot_bending_moment(
        self,
        speed,
        frequency_units="rad/s",
        moment_units="N*m",
        rotor_length_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot the bending moment diagram.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        moment_units : str, optional
            Moment units.
            Default is 'N*m'.
        rotor_length_units : str
            Rotor Length units.
            Default is m.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the deflected shape
            plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        speed = Q_(speed, frequency_units).to("rad/s").m
        if not any(np.isclose(self.speed_range, speed, atol=1e-6)):
            raise ValueError("No data available for this speed value.")

        Mx, My = self._calculate_bending_moment(speed=speed)
        Mx = Q_(Mx, "N*m").to(moment_units).m
        My = Q_(My, "N*m").to(moment_units).m
        Mr = np.sqrt(Mx ** 2 + My ** 2)

        nodes_pos = Q_(self.rotor.nodes_pos, "m").to(rotor_length_units).m

        if fig is None:
            fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=Mx,
                mode="lines",
                name=f"Bending Moment (X dir.) ({moment_units})",
                legendgroup="Mx",
                showlegend=True,
                hovertemplate=f"Nodal Position: %{{x:.2f}}<br>Mx ({moment_units}): %{{y:.2e}}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=My,
                mode="lines",
                name=f"Bending Moment (Y dir.) ({moment_units})",
                legendgroup="My",
                showlegend=True,
                hovertemplate=f"Nodal Position: %{{x:.2f}}<br>My ({moment_units}): %{{y:.2e}}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=Mr,
                mode="lines",
                name="Bending Moment (abs)",
                legendgroup="Mr",
                showlegend=True,
                hovertemplate=f"Nodal Position: %{{x:.2f}}<br>Mr ({moment_units}): %{{y:.2e}}",
            )
        )

        # plot center line
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=np.zeros_like(nodes_pos),
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        fig.update_xaxes(title_text=f"Rotor Length ({rotor_length_units})")
        fig.update_yaxes(title_text=f"Bending Moment ({moment_units})")
        fig.update_layout(**kwargs)

        return fig

    def plot_deflected_shape(
        self,
        speed,
        samples=101,
        frequency_units="rad/s",
        displacement_units="m",
        rotor_length_units="m",
        moment_units="N*m",
        shape2d_kwargs=None,
        shape3d_kwargs=None,
        bm_kwargs=None,
        subplot_kwargs=None,
    ):
        """Plot deflected shape diagrams.

        This method returns a subplot with:
            - 3D view deflected shape;
            - 2D view deflected shape - Major Axis;
            - Bending Moment Diagram;

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class.
        samples : int, optional
            Number of samples to generate the orbit for each node.
            Default is 101.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        displacement_units : str, optional
            Displacement units.
            Default is 'm'.
        rotor_length_units : str, optional
            Rotor length units.
            Default is 'm'.
        moment_units : str
            Moment units.
            Default is 'N*m'
        shape2d_kwargs : optional
            Additional key word arguments can be passed to change the 2D deflected shape
            plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        shape3d_kwargs : optional
            Additional key word arguments can be passed to change the 3D deflected shape
            plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        bm_kwargs : optional
            Additional key word arguments can be passed to change the bending moment
            diagram plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        subplot_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...). This kwargs override "mag_kwargs" and
            "phase_kwargs" dictionaries.
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            Plotly figure with Amplitude vs Frequency and Phase vs Frequency and
            polar Amplitude vs Phase plots.
        """
        shape2d_kwargs = {} if shape2d_kwargs is None else copy.copy(shape2d_kwargs)
        shape3d_kwargs = {} if shape3d_kwargs is None else copy.copy(shape3d_kwargs)
        bm_kwargs = {} if bm_kwargs is None else copy.copy(bm_kwargs)
        subplot_kwargs = {} if subplot_kwargs is None else copy.copy(subplot_kwargs)

        # fmt: off
        fig0 = self.plot_deflected_shape_2d(
            speed, frequency_units, displacement_units, rotor_length_units, **shape2d_kwargs
        )
        fig1 = self.plot_deflected_shape_3d(
            speed, samples, frequency_units, displacement_units, rotor_length_units, **shape3d_kwargs
        )
        fig2 = self.plot_bending_moment(
            speed, frequency_units, moment_units, rotor_length_units, **bm_kwargs
        )
        # fmt: on

        subplots = make_subplots(
            rows=2, cols=2, specs=[[{}, {"type": "scene", "rowspan": 2}], [{}, None]]
        )
        for data in fig0["data"]:
            subplots.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            subplots.add_trace(data, row=1, col=2)
        for data in fig2["data"]:
            subplots.add_trace(data, row=2, col=1)

        subplots.update_xaxes(fig0.layout.xaxis, row=1, col=1)
        subplots.update_yaxes(fig0.layout.yaxis, row=1, col=1)
        subplots.update_xaxes(fig2.layout.xaxis, row=2, col=1)
        subplots.update_yaxes(fig2.layout.yaxis, row=2, col=1)
        subplots.update_layout(
            scene=dict(
                bgcolor=fig1.layout.scene.bgcolor,
                xaxis=fig1.layout.scene.xaxis,
                yaxis=fig1.layout.scene.yaxis,
                zaxis=fig1.layout.scene.zaxis,
            ),
            **subplot_kwargs,
        )

        return subplots


class StaticResults:
    """Class used to store results and provide plots for Static Analysis.

    This class plots free-body diagram, deformed shaft, shearing
    force diagram and bending moment diagram.

    Parameters
    ----------
    deformation : array
        shaft displacement in y direction.
    Vx : array
        shearing force array.
    Bm : array
        bending moment array.
    w_shaft : dataframe
        shaft dataframe
    disk_forces : dict
        Indicates the force exerted by each disk.
    bearing_forces : dict
        Relates the static force at each node due to the bearing reaction forces.
    nodes : list
        list of nodes numbers.
    nodes_pos : list
        list of nodes positions.
    Vx_axis : array
        X axis for displaying shearing force and bending moment.

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        Plotly figure with Static Analysis plots depending on which method
        is called.
    """

    def __init__(
        self,
        deformation,
        Vx,
        Bm,
        w_shaft,
        disk_forces,
        bearing_forces,
        nodes,
        nodes_pos,
        Vx_axis,
    ):

        self.deformation = deformation
        self.Vx = Vx
        self.Bm = Bm
        self.w_shaft = w_shaft
        self.disk_forces = disk_forces
        self.bearing_forces = bearing_forces
        self.nodes = nodes
        self.nodes_pos = nodes_pos
        self.Vx_axis = Vx_axis

    def plot_deformation(
        self, deformation_units="m", rotor_length_units="m", fig=None, **kwargs
    ):
        """Plot the shaft static deformation.

        This method plots:
            deformed shaft

        Parameters
        ----------
        deformation_units : str
            Deformation units.
            Default is 'm'.
        rotor_length_units : str
            Rotor Length units.
            Default is 'm'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if fig is None:
            fig = go.Figure()

        shaft_end = max([sublist[-1] for sublist in self.nodes_pos])
        shaft_end = Q_(shaft_end, "m").to(rotor_length_units).m

        # fig - plot centerline
        fig.add_trace(
            go.Scatter(
                x=[-0.01 * shaft_end, 1.01 * shaft_end],
                y=[0, 0],
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        count = 0
        for deformation, Vx, Bm, nodes, nodes_pos, Vx_axis in zip(
            self.deformation, self.Vx, self.Bm, self.nodes, self.nodes_pos, self.Vx_axis
        ):

            fig.add_trace(
                go.Scatter(
                    x=Q_(nodes_pos, "m").to(rotor_length_units).m,
                    y=Q_(deformation, "m").to(deformation_units).m,
                    mode="lines",
                    line_shape="spline",
                    line_smoothing=1.0,
                    name=f"Shaft {count}",
                    showlegend=True,
                    hovertemplate=(
                        f"Rotor Length ({rotor_length_units}): %{{x:.2f}}<br>Displacement ({deformation_units}): %{{y:.2e}}"
                    ),
                )
            )
            count += 1

        fig.update_xaxes(title_text=f"Rotor Length ({rotor_length_units})")
        fig.update_yaxes(title_text=f"Deformation ({deformation_units})")
        fig.update_layout(title=dict(text="Static Deformation"), **kwargs)

        return fig

    def plot_free_body_diagram(
        self, force_units="N", rotor_length_units="m", fig=None, **kwargs
    ):
        """Plot the rotor free-body diagram.

        Parameters
        ----------
        force_units : str
            Force units.
            Default is 'N'.
        rotor_length_units : str
            Rotor Length units.
            Default is 'm'.
        subplots : Plotly graph_objects.make_subplots()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. plot_bgcolor="white", ...).
            *See Plotly Python make_subplot Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            The figure object with the plot.
        """
        cols = 1 if len(self.nodes_pos) < 2 else 2
        rows = len(self.nodes_pos) // 2 + len(self.nodes_pos) % 2
        if fig is None:
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[
                    "Free-Body Diagram - Shaft {}".format(j)
                    for j in range(len(self.nodes_pos))
                ],
            )
        j = 0
        y_start = 5.0
        for nodes_pos, nodes in zip(self.nodes_pos, self.nodes):
            col = j % 2 + 1
            row = j // 2 + 1

            fig.add_trace(
                go.Scatter(
                    x=Q_(nodes_pos, "m").to(rotor_length_units).m,
                    y=np.zeros(len(nodes_pos)),
                    mode="lines",
                    line=dict(color="black"),
                    hoverinfo="none",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=Q_(nodes_pos, "m").to(rotor_length_units).m,
                    y=[y_start] * len(nodes_pos),
                    mode="lines",
                    line=dict(color="black"),
                    hoverinfo="none",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # fig - plot arrows indicating shaft weight distribution
            text = "{:.1f}".format(Q_(self.w_shaft[j], "N").to(force_units).m)
            ini = nodes_pos[0]
            fin = nodes_pos[-1]
            arrows_list = np.arange(ini, 1.01 * fin, (fin - ini) / 5.0)
            for node in arrows_list:
                fig.add_annotation(
                    x=Q_(node, "m").to(rotor_length_units).m,
                    y=0,
                    axref="x{}".format(j + 1),
                    ayref="y{}".format(j + 1),
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=5,
                    arrowcolor="DimGray",
                    ax=Q_(node, "m").to(rotor_length_units).m,
                    ay=y_start * 1.08,
                    row=row,
                    col=col,
                )
            fig.add_annotation(
                x=Q_(nodes_pos[0], "m").to(rotor_length_units).m,
                y=y_start,
                xref="x{}".format(j + 1),
                yref="y{}".format(j + 1),
                xshift=125,
                yshift=20,
                text=f"Shaft weight = {text}{force_units}",
                align="right",
                showarrow=False,
            )

            # plot bearing reaction forces
            for k, v in self.bearing_forces.items():
                _, node = k.split("_")
                node = int(node)
                if node in nodes:
                    text = f"{Q_(v, 'N').to(force_units).m:.2f}"
                    var = 1 if v < 0 else -1
                    fig.add_annotation(
                        x=Q_(nodes_pos[nodes.index(node)], "m")
                        .to(rotor_length_units)
                        .m,
                        y=0,
                        axref="x{}".format(j + 1),
                        ayref="y{}".format(j + 1),
                        text=f"Fb = {text}{force_units}",
                        textangle=90,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=5,
                        arrowcolor="DarkSalmon",
                        ax=Q_(nodes_pos[nodes.index(node)], "m")
                        .to(rotor_length_units)
                        .m,
                        ay=var * 2.5 * y_start,
                        row=row,
                        col=col,
                    )

            # plot disk forces
            for k, v in self.disk_forces.items():
                _, node = k.split("_")
                node = int(node)
                if node in nodes:
                    text = f"{-Q_(v, 'N').to(force_units).m:.2f}"
                    fig.add_annotation(
                        x=Q_(nodes_pos[nodes.index(node)], "m")
                        .to(rotor_length_units)
                        .m,
                        y=0,
                        axref="x{}".format(j + 1),
                        ayref="y{}".format(j + 1),
                        text=f"Fd = {text}{force_units}",
                        textangle=270,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=5,
                        arrowcolor="FireBrick",
                        ax=Q_(nodes_pos[nodes.index(node)], "m")
                        .to(rotor_length_units)
                        .m,
                        ay=2.5 * y_start,
                        row=row,
                        col=col,
                    )

            fig.update_xaxes(
                title_text=f"Rotor Length ({rotor_length_units})", row=row, col=col
            )
            fig.update_yaxes(
                visible=False, gridcolor="lightgray", showline=False, row=row, col=col
            )
            j += 1

        fig.update_layout(**kwargs)

        return fig

    def plot_shearing_force(
        self, force_units="N", rotor_length_units="m", fig=None, **kwargs
    ):
        """Plot the rotor shearing force diagram.

        This method plots:
            shearing force diagram.

        Parameters
        ----------
        force_units : str
            Force units.
            Default is 'N'.
        rotor_length_units : str
            Rotor Length units.
            Default is 'm'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if fig is None:
            fig = go.Figure()

        shaft_end = (
            Q_(max([sublist[-1] for sublist in self.nodes_pos]), "m")
            .to(rotor_length_units)
            .m
        )

        # fig - plot centerline
        fig.add_trace(
            go.Scatter(
                x=[-0.1 * shaft_end, 1.1 * shaft_end],
                y=[0, 0],
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        j = 0
        for Vx, Vx_axis in zip(self.Vx, self.Vx_axis):
            fig.add_trace(
                go.Scatter(
                    x=Q_(Vx_axis, "m").to(rotor_length_units).m,
                    y=Q_(Vx, "N").to(force_units).m,
                    mode="lines",
                    name=f"Shaft {j}",
                    legendgroup=f"Shaft {j}",
                    showlegend=True,
                    hovertemplate=(
                        f"Rotor Length ({rotor_length_units}): %{{x:.2f}}<br>Shearing Force ({force_units}): %{{y:.2f}}"
                    ),
                )
            )
            j += 1

        fig.update_xaxes(
            title_text=f"Rotor Length ({rotor_length_units})",
            range=[-0.1 * shaft_end, 1.1 * shaft_end],
        )
        fig.update_yaxes(title_text=f"Force ({force_units})")
        fig.update_layout(title=dict(text="Shearing Force Diagram"), **kwargs)

        return fig

    def plot_bending_moment(
        self, moment_units="N*m", rotor_length_units="m", fig=None, **kwargs
    ):
        """Plot the rotor bending moment diagram.

        This method plots:
            bending moment diagram.

        Parameters
        ----------
        moment_units : str, optional
            Moment units.
            Default is 'N*m'.
        rotor_length_units : str
            Rotor Length units.
            Default is 'm'.
        fig : Plotly graph_objects.Figure()
            Plotly figure with the bending moment diagram plot

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            Plotly figure with the bending moment diagram plot
        """
        if fig is None:
            fig = go.Figure()

        shaft_end = (
            Q_(max([sublist[-1] for sublist in self.nodes_pos]), "m")
            .to(rotor_length_units)
            .m
        )

        # fig - plot centerline
        fig.add_trace(
            go.Scatter(
                x=[-0.1 * shaft_end, 1.1 * shaft_end],
                y=[0, 0],
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        j = 0
        for Bm, nodes_pos in zip(self.Bm, self.Vx_axis):
            fig.add_trace(
                go.Scatter(
                    x=Q_(nodes_pos, "m").to(rotor_length_units).m,
                    y=Q_(Bm, "N*m").to(moment_units).m,
                    mode="lines",
                    line_shape="spline",
                    line_smoothing=1.0,
                    name=f"Shaft {j}",
                    legendgroup=f"Shaft {j}",
                    showlegend=True,
                    hovertemplate=(
                        f"Rotor Length ({rotor_length_units}): %{{x:.2f}}<br>Bending Moment ({moment_units}): %{{y:.2f}}"
                    ),
                )
            )
            j += 1

        fig.update_xaxes(title_text=f"Rotor Length ({rotor_length_units})")
        fig.update_yaxes(title_text=f"Bending Moment ({moment_units})")
        fig.update_layout(title=dict(text="Bending Moment Diagram"), **kwargs)

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
    fig : Plotly graph_objects.make_subplots()
        The figure object with the tables plot.
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

        Returns
        -------
        fig : Plotly graph_objects.make_subplots()
            The figure object with the tables plot.
        """
        materials = [mat.name for mat in self.df_shaft["material"]]

        shaft_data = {
            "Shaft number": self.df_shaft["shaft_number"],
            "Left station": self.df_shaft["n_l"],
            "Right station": self.df_shaft["n_r"],
            "Elem number": self.df_shaft["_n"],
            "Beam left loc": self.df_shaft["nodes_pos_l"],
            "Length": self.df_shaft["L"],
            "Axial CG Pos": self.df_shaft["axial_cg_pos"],
            "Beam right loc": self.df_shaft["nodes_pos_r"],
            "Material": materials,
            "Mass": self.df_shaft["m"].map("{:.3f}".format),
            "Inertia": self.df_shaft["Im"].map("{:.2e}".format),
        }

        rotor_data = {
            "Tag": [self.tag],
            "Starting node": [self.df_shaft["n_l"].iloc[0]],
            "Ending node": [self.df_shaft["n_r"].iloc[-1]],
            "Starting point": [self.df_shaft["nodes_pos_l"].iloc[0]],
            "Total lenght": [self.df_shaft["nodes_pos_r"].iloc[-1]],
            "CG": ["{:.3f}".format(self.CG)],
            "Ip": ["{:.3e}".format(self.Ip)],
            "Rotor Mass": [
                "{:.3f}".format(np.sum(self.df_shaft["m"]) + np.sum(self.df_disks["m"]))
            ],
        }

        disk_data = {
            "Tag": self.df_disks["tag"],
            "Shaft number": self.df_disks["shaft_number"],
            "Node": self.df_disks["n"],
            "Nodal Position": self.nodes_pos[self.df_bearings["n"]],
            "Mass": self.df_disks["m"].map("{:.3f}".format),
            "Ip": self.df_disks["Ip"].map("{:.3e}".format),
        }

        bearing_data = {
            "Tag": self.df_bearings["tag"],
            "Shaft number": self.df_bearings["shaft_number"],
            "Node": self.df_bearings["n"],
            "N_link": self.df_bearings["n_link"],
            "Nodal Position": self.nodes_pos[self.df_bearings["n"]],
            "Bearing force": list(self.brg_forces.values()),
        }

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "table"}, {"type": "table"}],
                [{"type": "table"}, {"type": "table"}],
            ],
            subplot_titles=[
                "Rotor data",
                "Shaft Element data",
                "Disk Element data",
                "Bearing Element data",
            ],
        )
        colors = ["#ffffff", "#c4d9ed"]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["{}".format(k) for k in rotor_data.keys()],
                    font=dict(size=12, color="white"),
                    line=dict(color="#1f4060", width=1.5),
                    fill=dict(color="#1f4060"),
                    align="center",
                ),
                cells=dict(
                    values=list(rotor_data.values()),
                    font=dict(size=12),
                    line=dict(color="#1f4060"),
                    fill=dict(color="white"),
                    align="center",
                    height=25,
                ),
            ),
            row=1,
            col=1,
        )

        cell_colors = [colors[i % 2] for i in range(len(materials))]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["{}".format(k) for k in shaft_data.keys()],
                    font=dict(family="Verdana", size=12, color="white"),
                    line=dict(color="#1e4162", width=1.5),
                    fill=dict(color="#1e4162"),
                    align="center",
                ),
                cells=dict(
                    values=list(shaft_data.values()),
                    font=dict(family="Verdana", size=12, color="#12263b"),
                    line=dict(color="#c4d9ed", width=1.5),
                    fill=dict(color=[cell_colors * len(shaft_data)]),
                    align="center",
                    height=25,
                ),
            ),
            row=1,
            col=2,
        )

        cell_colors = [colors[i % 2] for i in range(len(self.df_disks["tag"]))]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["{}".format(k) for k in disk_data.keys()],
                    font=dict(family="Verdana", size=12, color="white"),
                    line=dict(color="#1e4162", width=1.5),
                    fill=dict(color="#1e4162"),
                    align="center",
                ),
                cells=dict(
                    values=list(disk_data.values()),
                    font=dict(family="Verdana", size=12, color="#12263b"),
                    line=dict(color="#c4d9ed", width=1.5),
                    fill=dict(color=[cell_colors * len(shaft_data)]),
                    align="center",
                    height=25,
                ),
            ),
            row=2,
            col=1,
        )

        cell_colors = [colors[i % 2] for i in range(len(self.df_bearings["tag"]))]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["{}".format(k) for k in bearing_data.keys()],
                    font=dict(family="Verdana", size=12, color="white"),
                    line=dict(color="#1e4162", width=1.5),
                    fill=dict(color="#1e4162"),
                    align="center",
                ),
                cells=dict(
                    values=list(bearing_data.values()),
                    font=dict(family="Verdana", size=12, color="#12263b"),
                    line=dict(color="#c4d9ed", width=1.5),
                    fill=dict(color=[cell_colors * len(shaft_data)]),
                    align="center",
                    height=25,
                ),
            ),
            row=2,
            col=2,
        )
        return fig


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
    fig : Plotly graph_objects.make_subplots()
        The figure object with the plot.
    """

    def __init__(self, el_num, eigv_arr, error_arr):
        self.el_num = el_num
        self.eigv_arr = eigv_arr
        self.error_arr = error_arr

    def plot(self, fig=None, **kwargs):
        """Plot convergence results.

        This method plots:
            Natural Frequency vs Number of Elements
            Relative Error vs Number of Elements

        Parameters
        ----------
        fig : Plotly graph_objects.make_subplots()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.make_subplots()
            The figure object with the plot.
        """
        if fig is None:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Frequency Evaluation", "Relative Error Evaluation"],
            )

        # plot Frequency vs number of elements
        fig.add_trace(
            go.Scatter(
                x=self.el_num,
                y=self.eigv_arr,
                mode="lines+markers",
                hovertemplate=(
                    "Number of Elements: %{x:.2f}<br>" + "Frequency: %{y:.0f}"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Number of Elements", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)

        # plot Error vs number of elements
        fig.add_trace(
            go.Scatter(
                x=self.el_num,
                y=self.error_arr,
                mode="lines+markers",
                hovertemplate=(
                    "Number of Elements: %{x:.2f}<br>" + "Relative Error: %{y:.0f}"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Number of Elements", row=1, col=2)
        fig.update_yaxes(title_text="Relative Error (%)", row=1, col=2)

        fig.update_layout(**kwargs)

        return fig


class TimeResponseResults:
    """Class used to store results and provide plots for Time Response Analysis.

    This class takes the results from time response analysis and creates a
    plots given a force and a time. It's possible to select through a time response for
    a single DoF, an orbit response for a single node or display orbit response for all
    nodes.
    The plot type options are:
        - 1d: plot time response for given probes.
        - 2d: plot orbit of a selected node of a rotor system.
        - 3d: plot orbits for each node on the rotor system in a 3D view.

    plot_1d: input probes.
    plot_2d: input a node.
    plot_3d: no need to input probes or node.

    Parameters
    ----------
    t : array
        Time values for the output.
    yout : array
        System response.
    xout : array
        Time evolution of the state vector.
    nodes_list : array
        list with nodes from a rotor model.
    nodes_pos : array
        Rotor nodes axial positions.
    number_dof : int
        Number of degrees of freedom per shaft element's node

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    """

    def __init__(self, t, yout, xout, nodes_list, nodes_pos, number_dof):
        self.t = t
        self.yout = yout
        self.xout = xout
        self.nodes_list = nodes_list
        self.nodes_pos = nodes_pos
        self.number_dof = number_dof

    def plot_1d(
        self,
        probe,
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        """Plot time response for a single DoF using Plotly.

        This function will take a rotor object and plot its time response using Plotly.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle).
            node : int
                indicate the node where the probe is located.
            orientation : float,
                probe orientation angle about the shaft. The 0 refers to +X direction.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        displacement_units : str, optional
            Displacement units.
            Default is 'm'.
        time_units : str
            Time units.
            Default is 's'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            dofx = p[0] * self.number_dof
            dofy = p[0] * self.number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )

            _probe_resp = operator @ np.vstack((self.yout[:, dofx], self.yout[:, dofy]))
            probe_resp = (
                _probe_resp[0] * np.cos(angle) ** 2  +
                _probe_resp[1] * np.sin(angle) ** 2
            )
            # fmt: on

            probe_resp = Q_(probe_resp, "m").to(displacement_units).m

            fig.add_trace(
                go.Scatter(
                    x=Q_(self.t, "s").to(time_units).m,
                    y=Q_(probe_resp, "m").to(displacement_units).m,
                    mode="lines",
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Time ({time_units}): %{{x:.2f}}<br>Amplitude ({displacement_units}): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(title_text=f"Time ({time_units})")
        fig.update_yaxes(title_text=f"Amplitude ({displacement_units})")
        fig.update_layout(**kwargs)

        return fig

    def plot_2d(self, node, displacement_units="m", fig=None, **kwargs):
        """Plot orbit response (2D).

        This function will take a rotor object and plot its orbit response using Plotly.

        Parameters
        ----------
        node: int, optional
            Selected node to plot orbit.
        displacement_units : str, optional
            Displacement units.
            Default is 'm'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=Q_(self.yout[:, self.number_dof * node], "m")
                .to(displacement_units)
                .m,
                y=Q_(self.yout[:, self.number_dof * node + 1], "m")
                .to(displacement_units)
                .m,
                mode="lines",
                name="Phase",
                legendgroup="Phase",
                showlegend=False,
                hovertemplate=(
                    f"X - Amplitude ({displacement_units}): %{{x:.2e}}<br>Y - Amplitude ({displacement_units}): %{{y:.2e}}"
                ),
            )
        )

        fig.update_xaxes(title_text=f"Amplitude ({displacement_units}) - X direction")
        fig.update_yaxes(title_text=f"Amplitude ({displacement_units}) - Y direction")
        fig.update_layout(
            title=dict(text="Response for node {}".format(node)), **kwargs
        )

        return fig

    def plot_3d(
        self, displacement_units="m", rotor_length_units="m", fig=None, **kwargs
    ):
        """Plot orbit response (3D).

        This function will take a rotor object and plot its orbit response using Plotly.

        Parameters
        ----------
        displacement_units : str
            Displacement units.
            Default is 'm'.
        rotor_length_units : str
            Rotor Length units.
            Default is 'm'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. hoverlabel_align="center", ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if fig is None:
            fig = go.Figure()

        for n in self.nodes_list:
            x_pos = np.ones(self.yout.shape[0]) * self.nodes_pos[n]
            fig.add_trace(
                go.Scatter3d(
                    x=Q_(x_pos, "m").to(rotor_length_units).m,
                    y=Q_(self.yout[:, self.number_dof * n], "m")
                    .to(displacement_units)
                    .m,
                    z=Q_(self.yout[:, self.number_dof * n + 1], "m")
                    .to(displacement_units)
                    .m,
                    mode="lines",
                    line=dict(color=tableau_colors["blue"]),
                    name="Mean",
                    legendgroup="mean",
                    showlegend=False,
                    hovertemplate=(
                        f"Nodal Position ({rotor_length_units}): %{{x:.2f}}<br>X - Amplitude ({displacement_units}): %{{y:.2e}}<br>Y - Amplitude ({displacement_units}): %{{z:.2e}}"
                    ),
                    **kwargs,
                )
            )

        # plot center line
        line = np.zeros(len(self.nodes_pos))

        fig.add_trace(
            go.Scatter3d(
                x=Q_(self.nodes_pos, "m").to(rotor_length_units).m,
                y=line,
                z=line,
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                showlegend=False,
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(title=dict(text=f"Rotor Length ({rotor_length_units})")),
                yaxis=dict(title=dict(text=f"Amplitude - X ({displacement_units})")),
                zaxis=dict(title=dict(text=f"Amplitude - Y ({displacement_units})")),
            ),
            **kwargs,
        )

        return fig
