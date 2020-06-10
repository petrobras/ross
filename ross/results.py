"""ROSS plotting module.

This module returns graphs for each type of analyses in rotor_assembly.py.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scipy.linalg as la
from plotly.subplots import make_subplots
from scipy import interpolate

pio.renderers.default = "browser"

# set Plotly palette of colors
colors1 = px.colors.qualitative.Dark24


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
        Logarithmic decrement for each .
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

    def plot_mode3D(self, mode=None, evec=None, **kwargs):
        """Plot (3D view) the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
            Default is None
        evec : array
            Array containing the system eigenvectors
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
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
                    mode="markers+lines",
                    line=dict(width=4.0, color=kappa_mode[node]),
                    marker=dict(size=5, color=kappa_mode[node]),
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
                line=dict(width=3.0, color="black", dash="dash"),
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
                line=dict(width=2.0, color="black", dash="dashdot"),
                hoverinfo="none",
                showlegend=False,
            )
        )
        fig.update_layout(
            width=1200,
            height=900,
            scene=dict(
                bgcolor="white",
                xaxis=dict(
                    title=dict(text="<b>Rotor Length</b>", font=dict(size=14)),
                    tickfont=dict(size=16),
                    range=[zn_cl0 - 0.1, zn_cl1 + 0.1],
                    nticks=5,
                    backgroundcolor="lightgray",
                    gridcolor="white",
                    showspikes=False,
                ),
                yaxis=dict(
                    title=dict(
                        text="<b>Dimensionless deformation</b>", font=dict(size=14),
                    ),
                    tickfont=dict(size=16),
                    range=[-2, 2],
                    nticks=5,
                    backgroundcolor="lightgray",
                    gridcolor="white",
                    showspikes=False,
                ),
                zaxis=dict(
                    title=dict(
                        text="<b>Dimensionless deformation</b>", font=dict(size=14),
                    ),
                    tickfont=dict(size=16),
                    range=[-2, 2],
                    nticks=5,
                    backgroundcolor="lightgray",
                    gridcolor="white",
                    showspikes=False,
                ),
            ),
            title=dict(
                text=(
                    f"<b>Mode</b> {mode + 1}<br>"
                    f"<b>whirl</b>: {self.whirl_direction()[mode]}<br>"
                    f"<b>ωn</b> = {self.wn[mode]:.1f} rad/s<br>"
                    f"<b>log dec</b> = {self.log_dec[mode]:.1f}"
                ),
                font=dict(size=14),
            ),
            **kwargs,
        )

        return fig

    def plot_mode2D(self, mode=None, evec=None, **kwargs):
        """Plot (2D view) the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
            Default is None
        evec : array
            Array containing the system eigenvectors
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

        fig = go.Figure()

        colors = dict(Backward="firebrick", Mixed="black", Forward="royalblue")
        whirl_dir = colors[self.whirl_direction()[mode]]

        fig.add_trace(
            go.Scatter(
                x=zn,
                y=vn,
                mode="lines",
                line=dict(width=2.5, color=whirl_dir),
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
                line=dict(width=1.0, color="black", dash="dashdot"),
                name="centerline",
                hoverinfo="none",
                showlegend=False,
            )
        )

        fig.update_xaxes(
            title_text="<b>Rotor Length</b>",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Non dimensional deformation</b>",
            title_font=dict(size=16),
            tickfont=dict(size=14),
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
            title=dict(
                text=(
                    f"<b>Mode</b> {mode + 1} | "
                    f"<b>whirl</b>: {self.whirl_direction()[mode]} | "
                    f"<b>ωn</b> = {self.wn[mode]:.1f} rad/s | "
                    f"<b>log dec</b> = {self.log_dec[mode]:.1f}"
                ),
                font=dict(size=16),
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

    def plot(self, harmonics=[1], **kwargs):
        """Create Campbell Diagram figure using Plotly.

        Parameters
        ----------
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        wd = self.wd
        num_frequencies = wd.shape[1]
        log_dec = self.log_dec
        whirl = self.whirl_values
        speed_range = self.speed_range
        log_dec_map = log_dec.flatten()

        fig = go.Figure()

        scatter_marker = ["triangle-up", "circle", "triangle-down"]
        for mark, whirl_dir, legend in zip(
            scatter_marker, [0.0, 0.5, 1.0], ["Foward", "Mixed", "Backward"]
        ):
            num_frequencies = wd.shape[1]
            for i in range(num_frequencies):
                w_i = wd[:, i]
                whirl_i = whirl[:, i]
                log_dec_i = log_dec[:, i]

                for harm in harmonics:
                    idx = np.argwhere(np.diff(np.sign(w_i - harm * speed_range)))
                    idx = idx.flatten()
                    if len(idx) != 0:
                        idx = idx[0]

                        interpolated = interpolate.interp1d(
                            x=[speed_range[idx], speed_range[idx + 1]],
                            y=[w_i[idx], w_i[idx + 1]],
                            kind="linear",
                        )
                        xnew = np.linspace(
                            speed_range[idx],
                            speed_range[idx + 1],
                            num=30,
                            endpoint=True,
                        )
                        ynew = interpolated(xnew)
                        idx = np.argwhere(
                            np.diff(np.sign(ynew - harm * xnew))
                        ).flatten()

                        fig.add_trace(
                            go.Scatter(
                                x=xnew[idx],
                                y=ynew[idx],
                                mode="markers",
                                marker=dict(symbol="x", size=10, color="black"),
                                name="Crit. Speed",
                                legendgroup="Crit. Speed",
                                showlegend=False,
                                hovertemplate=(
                                    "Frequency: %{x:.2f}<br>"
                                    + "Critical Speed: %{y:.2f}"
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
                                size=16,
                                cmax=max(log_dec_map),
                                cmin=min(log_dec_map),
                                color=log_dec_i[whirl_mask],
                                coloraxis="coloraxis",
                                colorscale="rdbu",
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
                    line=dict(width=2.5, color=colors1[j], dash="dashdot"),
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
                    x=[-1000],
                    y=[-1000],
                    mode="markers",
                    name=legend,
                    legendgroup=legend,
                    marker=dict(symbol=mark, size=16, color="black"),
                )
            )

        fig.update_xaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            range=[0, np.max(speed_range)],
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
            range=[0, 1.1 * np.max(wd)],
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
            hoverlabel_align="right",
            coloraxis=dict(
                cmin=min(log_dec_map),
                cmax=max(log_dec_map),
                colorscale="rdbu",
                colorbar=dict(
                    title=dict(text="<b>Log Dec</b>", side="right", font=dict(size=20)),
                    tickfont=dict(size=16),
                ),
            ),
            legend=dict(
                itemsizing="constant",
                bgcolor="white",
                borderwidth=2,
                font=dict(size=14),
                orientation="h",
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

    def plot_magnitude(self, inp, out, units="mic-pk-pk", **mag_kwargs):
        """Plot frequency response (magnitude) using Plotly.

        This method plots the frequency response magnitude given an output and
        an input using Plotly.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        units : str
            Unit system
            Default is "mic-pk-pk"
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

        if units == "m":
            y_axis_label = "<b>Amplitude (m)</b>"
        elif units == "mic-pk-pk":
            y_axis_label = "<b>Amplitude (μ pk-pk)</b>"
        else:
            y_axis_label = "<b>Amplitude (dB)</b>"

        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            mag_kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=frequency_range,
                y=mag[inp, out, :],
                mode="lines",
                line=dict(width=3.0, color="royalblue"),
                name="Amplitude",
                legendgroup="Amplitude",
                showlegend=False,
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
            )
        )

        fig.update_xaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            range=[0, np.max(frequency_range)],
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
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **mag_kwargs,
        )

        return fig

    def plot_phase(self, inp, out, **phase_kwargs):
        """Plot frequency response (phase) using Plotly.

        This method plots the frequency response phase given an output and
        an input using Plotly.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
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
        phase = self.phase

        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            phase_kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=frequency_range,
                y=phase[inp, out, :],
                mode="lines",
                line=dict(width=3.0, color="royalblue"),
                name="Phase",
                legendgroup="Phase",
                showlegend=False,
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
            )
        )

        fig.update_xaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            range=[0, np.max(frequency_range)],
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Phase</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **phase_kwargs,
        )

        return fig

    def plot_polar_bode(self, inp, out, units="mic-pk-pk", **polar_kwargs):
        """Plot frequency response (polar) using Plotly.

        This method plots the frequency response (polar graph) given an output and
        an input using Plotly.

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
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
        frequency_range = self.speed_range
        mag = self.magnitude
        phase = self.phase

        if units == "m":
            r_axis_label = "<b>Amplitude (m)</b>"
        elif units == "mic-pk-pk":
            r_axis_label = "<b>Amplitude (μ pk-pk)</b>"
        else:
            r_axis_label = "<b>Amplitude (dB)</b>"

        kwargs_default_values = dict(
            width=1200, height=900, polar_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            polar_kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=mag[inp, out, :],
                theta=phase[inp, out, :],
                customdata=frequency_range,
                thetaunit="radians",
                mode="lines+markers",
                marker=dict(size=8, color="royalblue"),
                line=dict(width=3.0, color="royalblue"),
                name="Polar_plot",
                legendgroup="Polar",
                showlegend=False,
                hovertemplate=(
                    "<b>Amplitude: %{r:.2e}</b><br>"
                    + "<b>Phase: %{theta:.2f}</b><br>"
                    + "<b>Frequency: %{customdata:.2f}</b>"
                ),
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title=dict(text=r_axis_label, font=dict(size=14)),
                    tickfont=dict(size=14),
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
            **polar_kwargs,
        )

        return fig

    def plot(
        self,
        inp,
        out,
        units="mic-pk-pk",
        mag_kwargs={},
        phase_kwargs={},
        polar_kwargs={},
        subplot_kwargs={},
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
        units : str, optional
            Magnitude unit system.
            Options:
                - "m" : meters
                - "mic-pk-pk" : microns peak to peak
                - "db" : decibels
            Default is "mic-pk-pk".
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
        kwargs_default_values = dict(
            width=1800,
            height=900,
            polar_bgcolor="white",
            plot_bgcolor="white",
            hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            subplot_kwargs.setdefault(k, v)

        fig0 = self.plot_magnitude(inp, out, units, **mag_kwargs)
        fig1 = self.plot_phase(inp, out, **phase_kwargs)
        fig2 = self.plot_polar_bode(inp, out, units, **polar_kwargs)

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
        self, rotor, forced_resp, speed_range, magnitude, phase, unbalance=None,
    ):
        self.rotor = rotor
        self.forced_resp = forced_resp
        self.speed_range = speed_range
        self.magnitude = magnitude
        self.phase = phase
        self.unbalance = unbalance

    def plot_magnitude(self, dof, units="m", **kwargs):
        """Plot forced response (magnitude) using Plotly.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        units : str
            Units to plot the magnitude ('m' or 'mic-pk-pk')
            Default is 'm'
        kwargs : optional
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

        if units == "m":
            y_axis_label = "<b>Amplitude (m)</b>"
        elif units == "mic-pk-pk":
            mag = 2 * mag * 1e6
            y_axis_label = "<b>Amplitude (μ pk-pk)</b>"

        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=frequency_range,
                y=mag[dof],
                mode="lines",
                line=dict(width=3.0, color="royalblue"),
                name="Amplitude",
                legendgroup="Amplitude",
                showlegend=False,
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
            )
        )

        fig.update_xaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            range=[0, np.max(frequency_range)],
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
            exponentformat="power",
            mirror=True,
        )
        fig.update_layout(
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **kwargs,
        )

        return fig

    def plot_phase(self, dof, **kwargs):
        """Plot forced response (phase) using Plotly.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        frequency_range = self.speed_range
        phase = self.phase

        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=frequency_range,
                y=phase[dof],
                mode="lines",
                line=dict(width=3.0, color="royalblue"),
                name="Phase",
                legendgroup="Phase",
                showlegend=False,
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
            )
        )

        fig.update_xaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            range=[0, np.max(frequency_range)],
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
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **kwargs,
        )

        return fig

    def plot_polar_bode(self, dof, units="mic-pk-pk", **kwargs):
        """Plot polar forced response using Plotly.

        Parameters
        ----------
        dof : int
            Degree of freedom.
        units : str
            Magnitude unit system.
            Default is "mic-pk-pk"
        kwargs : optional
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
        phase = self.phase

        if units == "m":
            r_axis_label = "<b>Amplitude (m)</b>"
        elif units == "mic-pk-pk":
            r_axis_label = "<b>Amplitude (μ pk-pk)</b>"
        else:
            r_axis_label = "<b>Amplitude (dB)</b>"

        kwargs_default_values = dict(
            width=1200, height=900, polar_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=mag[dof],
                theta=phase[dof],
                customdata=frequency_range,
                thetaunit="radians",
                mode="lines+markers",
                marker=dict(size=8, color="royalblue"),
                line=dict(width=3.0, color="royalblue"),
                name="Polar_plot",
                legendgroup="Polar",
                showlegend=False,
                hovertemplate=(
                    "<b>Amplitude: %{r:.2e}</b><br>"
                    + "<b>Phase: %{theta:.2f}</b><br>"
                    + "<b>Frequency: %{customdata:.2f}</b>"
                ),
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
            **kwargs,
        )

        return fig

    def plot(
        self,
        dof,
        units="mic-pk-pk",
        mag_kwargs={},
        phase_kwargs={},
        polar_kwargs={},
        subplot_kwargs={},
    ):
        """Plot forced response.

        This method returns a subplot with:
            - Frequency vs Amplitude;
            - Frequency vs Phase Angle;
            - Polar plot Amplitude vs Phase Angle;

        Parameters
        ----------
        dof : int
            Degree of freedom.
        units : str, optional
            Magnitude unit system.
            Options:
                - "m" : meters
                - "mic-pk-pk" : microns peak to peak
                - "db" : decibels
            Default is "mic-pk-pk".
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
        kwargs_default_values = dict(
            width=1800,
            height=900,
            polar_bgcolor="white",
            plot_bgcolor="white",
            hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            subplot_kwargs.setdefault(k, v)

        fig0 = self.plot_magnitude(dof, **mag_kwargs)
        fig1 = self.plot_phase(dof, **phase_kwargs)
        fig2 = self.plot_polar_bode(dof, **polar_kwargs)

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

    def _calculate_major_axis(self, speed):
        """Calculate the major axis for each nodal orbit.

        Parameters
        ----------
        speed : optional, float
            The rotor rotation speed.

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
        major_axis_vector[4] = (
            np.real(major_axis_vector[3]) ** 2 +
            np.imag(major_axis_vector[3]) ** 2
        ) ** 0.5
        # fmt: on

        return major_axis_vector

    def _calculate_bending_moment(self, speed):
        """Calculate the bending moment in X and Y directions.

        This method calculate forces and moments on nodal positions for a deflected
        shape configuration.

        Parameters
        ----------
        speed : float
            The rotor rotation speed.

        Returns
        -------
        Mx : array
            Bending Moment on X directon.
        My : array
            Bending Moment on Y directon.
        """
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

    def plot_deflected_shape_2d(self, speed, units="mic-pk-pk", **kwargs):
        """Plot the 2D deflected shape diagram.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class.
        units : str, optional
            Magnitude unit system.
            Options:
                - "m" : meters
                - "mic-pk-pk" : microns peak to peak
                - "db" : decibels
            Default is "mic-pk-pk".
        kwargs : optional
            Additional key word arguments can be passed to change the deflected shape
            plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if not any(np.isclose(self.speed_range, speed, atol=1e-6)):
            raise ValueError("No data available for this speed value.")

        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        nodes_pos = self.rotor.nodes_pos
        maj_vect = self._calculate_major_axis(speed=speed)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=maj_vect[4].real,
                mode="lines",
                line=dict(width=6.0, color="royalblue"),
                name="Major Axis",
                legendgroup="Major_Axis_2d",
                showlegend=False,
                hovertemplate=(
                    "<b>Nodal Position: %{x:.2f}</b><br>" + "<b>Amplitude: %{y:.2e}</b>"
                ),
            )
        )
        # plot center line
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=np.zeros(len(nodes_pos)),
                mode="lines",
                line=dict(width=2.0, color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        fig.update_xaxes(
            title_text="<b>Rotor Length</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Major Axis Absolute Amplitude</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **kwargs,
        )

        return fig

    def plot_deflected_shape_3d(self, speed, samples=101, units="mic-pk-pk", **kwargs):
        """Plot the 3D deflected shape diagram.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class.
        samples : int, optional
            Number of samples to generate the orbit for each node.
            Default is 101.
        units : str, optional
            Magnitude unit system.
            Options:
                - "m" : meters
                - "mic-pk-pk" : microns peak to peak
                - "db" : decibels
            Default is "mic-pk-pk".
        kwargs : optional
            Additional key word arguments can be passed to change the deflected shape
            plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if not any(np.isclose(self.speed_range, speed, atol=1e-6)):
            raise ValueError("No data available for this speed value.")

        kwargs_default_values = dict(hoverlabel_align="right")
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        mag = self.magnitude
        phase = self.phase
        ub = self.unbalance
        nodes = self.rotor.nodes
        nodes_pos = self.rotor.nodes_pos
        number_dof = self.rotor.number_dof
        idx = np.where(np.isclose(self.speed_range, speed, atol=1e-6))[0][0]

        # orbit of a single revolution
        t = np.linspace(0, 2 * np.pi / speed, samples)

        x_pos = np.repeat(nodes_pos, t.size).reshape(len(nodes_pos), t.size)

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
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(width=6.0, color="royalblue"),
                    name="Orbit",
                    legendgroup="Orbit",
                    showlegend=False,
                    hovertemplate=(
                        "<b>Nodal Position: %{x:.2f}</b><br>"
                        + "<b>X - Amplitude: %{y:.2e}</b><br>"
                        + "<b>Y - Amplitude: %{z:.2e}</b>"
                    ),
                )
            )

        # plot major axis
        maj_vect = self._calculate_major_axis(speed=speed)

        fig.add_trace(
            go.Scatter3d(
                x=x_pos[:, 0],
                y=np.real(maj_vect[3]),
                z=np.imag(maj_vect[3]),
                mode="lines+markers",
                marker=dict(size=4.0, color="black"),
                line=dict(width=6.0, color="black", dash="dashdot"),
                name="Major Axis",
                legendgroup="Major_Axis",
                showlegend=True,
                hovertemplate=(
                    "Position: %{x:.2f}<br>"
                    + "X - Amplitude: %{y:.2e}<br>"
                    + "Y - Amplitude: %{z:.2e}"
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
                line=dict(width=2.0, color="black", dash="dashdot"),
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
                    y=[0, np.amax(np.real(maj_vect[4])) / 2 * np.cos(p)],
                    z=[0, np.amax(np.real(maj_vect[4])) / 2 * np.sin(p)],
                    mode="lines",
                    line=dict(width=6.0, color="firebrick"),
                    legendgroup="Unbalance",
                    hoverinfo="none",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[x_pos[int(n), 0]],
                    y=[np.amax(np.real(maj_vect[4])) / 2 * np.cos(p)],
                    z=[np.amax(np.real(maj_vect[4])) / 2 * np.sin(p)],
                    mode="markers",
                    marker=dict(symbol="diamond", size=5.0, color="firebrick"),
                    name="Unbalance",
                    legendgroup="Unbalance",
                    showlegend=True if i == 0 else False,
                    hovertemplate=(
                        "Node: {}<br>" + "Magnitude: {:.2e}<br>" + "Phase: {:.2f}"
                    ).format(int(n), m, p),
                )
            )
            i += 1

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
            title=dict(
                text=(f"<b>Deflected Shape</b><br>" f"<b>Speed = {speed}</b>"),
                font=dict(size=18),
            ),
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **kwargs,
        )

        return fig

    def plot_bending_moment(self, speed, units="mic-pk-pk", **kwargs):
        """Plot the bending moment diagram.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class.
        units : str, optional
            Magnitude unit system.
            Options:
                - "m" : meters
                - "mic-pk-pk" : microns peak to peak
                - "db" : decibels
            Default is "mic-pk-pk".
        kwargs : optional
            Additional key word arguments can be passed to change the deflected shape
            plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if not any(np.isclose(self.speed_range, speed, atol=1e-6)):
            raise ValueError("No data available for this speed value.")

        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        Mx, My = self._calculate_bending_moment(speed=speed)
        Mr = np.sqrt(Mx ** 2 + My ** 2)

        nodes_pos = self.rotor.nodes_pos

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=Mx,
                mode="lines",
                line=dict(width=6.0, color=colors1[2]),
                name="Bending Moment (X dir.)",
                legendgroup="Mx",
                showlegend=True,
                hovertemplate=(
                    "<b>Nodal Position: %{x:.2f}</b><br>" + "<b>Mx: %{y:.2e}</b>"
                ),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=My,
                mode="lines",
                line=dict(width=6.0, color=colors1[6]),
                name="Bending Moment (Y dir.)",
                legendgroup="My",
                showlegend=True,
                hovertemplate=(
                    "<b>Nodal Position: %{x:.2f}</b><br>" + "<b>My: %{y:.2e}</b>"
                ),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=Mr,
                mode="lines",
                line=dict(width=6.0, color=colors1[7]),
                name="Bending Moment (abs)",
                legendgroup="Mr",
                showlegend=True,
                hovertemplate=(
                    "<b>Nodal Position: %{x:.2f}</b><br>" + "<b>Mr: %{y:.2e}</b>"
                ),
            ),
        )

        # plot center line
        fig.add_trace(
            go.Scatter(
                x=nodes_pos,
                y=np.zeros_like(nodes_pos),
                mode="lines",
                line=dict(width=3.0, color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        fig.update_xaxes(
            title_text="<b>Rotor Length</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Bending Moment</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **kwargs,
        )

        return fig

    def plot_deflected_shape(
        self,
        speed,
        samples=101,
        units="mic-pk-pk",
        shape2d_kwargs={},
        shape3d_kwargs={},
        bm_kwargs={},
        subplot_kwargs={},
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
        units : str, optional
            Magnitude unit system.
            Options:
                - "m" : meters
                - "mic-pk-pk" : microns peak to peak
                - "db" : decibels
            Default is "mic-pk-pk".
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
        kwargs_default_values = dict(
            width=1800, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            subplot_kwargs.setdefault(k, v)

        fig0 = self.plot_deflected_shape_2d(speed, units, **shape2d_kwargs)
        fig1 = self.plot_deflected_shape_3d(speed, samples, units, **shape3d_kwargs)
        fig2 = self.plot_bending_moment(speed, units, **bm_kwargs)

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
    fig : Plotly graph_objects.Figure()
        Plotly figure with Static Analysis plots depending on which method
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

    def plot_deformation(self, **kwargs):
        """Plot the shaft static deformation.

        This method plots:
            deformed shaft

        Parameters
        ----------
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        shaft_end = max([sublist[-1] for sublist in self.nodes_pos])

        # fig - plot centerline
        fig.add_trace(
            go.Scatter(
                x=[-0.01 * shaft_end, 1.01 * shaft_end],
                y=[0, 0],
                mode="lines",
                line=dict(width=3.0, color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        count = 0
        for disp_y, Vx, Bm, nodes, nodes_pos, Vx_axis in zip(
            self.disp_y, self.Vx, self.Bm, self.nodes, self.nodes_pos, self.Vx_axis
        ):
            interpolated = interpolate.interp1d(nodes_pos, disp_y, kind="cubic")
            xnew = np.linspace(
                nodes_pos[0], nodes_pos[-1], num=len(nodes_pos) * 20, endpoint=True,
            )

            ynew = interpolated(xnew)

            fig.add_trace(
                go.Scatter(
                    x=xnew,
                    y=ynew,
                    mode="lines",
                    line=dict(width=5.0, color=colors1[count]),
                    name="Shaft {}".format(count),
                    showlegend=True,
                    hovertemplate=(
                        "Shaft lengh: %{x:.2f}<br>" + "Displacement: %{y:.2e}"
                    ),
                )
            )
            count += 1

        fig.update_xaxes(
            title_text="<b>Shaft Length</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Deformation</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            title=dict(text="<b>Static Deformation</b>", font=dict(size=16)),
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **kwargs,
        )

        return fig

    def plot_free_body_diagram(self, **kwargs):
        """Plot the rotor free-body diagram.

        Parameters
        ----------
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. plot_bgcolor="white", ...).
            *See Plotly Python make_subplot Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            The figure object with the plot.
        """
        kwargs_default_values = dict(width=1200, height=900, plot_bgcolor="white")
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        cols = 1 if len(self.nodes_pos) < 2 else 2
        rows = len(self.nodes_pos) // 2 + len(self.nodes_pos) % 2
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                "<b>Free-Body Diagram - Shaft {}</b>".format(j)
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
                    x=nodes_pos,
                    y=np.zeros(len(nodes_pos)),
                    mode="lines",
                    line=dict(width=10.0, color="black"),
                    hoverinfo="none",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=nodes_pos,
                    y=[y_start] * len(nodes_pos),
                    mode="lines",
                    line=dict(width=3.0, color="black"),
                    hoverinfo="none",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # fig - plot arrows indicating shaft weight distribution
            text = "{:.1f}".format(self.w_shaft[j])
            ini = nodes_pos[0]
            fin = nodes_pos[-1]
            arrows_list = np.arange(ini, 1.01 * fin, (fin - ini) / 5.0)
            for node in arrows_list:
                fig.add_annotation(
                    x=node,
                    y=0,
                    axref="x{}".format(j + 1),
                    ayref="y{}".format(j + 1),
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=5,
                    arrowcolor="DimGray",
                    ax=node,
                    ay=y_start * 1.08,
                    row=row,
                    col=col,
                )
            fig.add_annotation(
                x=nodes_pos[0],
                y=y_start,
                xref="x{}".format(j + 1),
                yref="y{}".format(j + 1),
                xshift=125,
                yshift=20,
                text="<b>Shaft weight = {}N</b>".format(text),
                font=dict(size=20),
                align="right",
                showarrow=False,
            )

            # plot bearing reaction forces
            for k, v in self.bearing_forces.items():
                _, node = k.split("_")
                node = int(node)
                if node in nodes:
                    text = str(v)
                    var = 1 if v < 0 else -1
                    fig.add_annotation(
                        x=nodes_pos[nodes.index(node)],
                        y=0,
                        axref="x{}".format(j + 1),
                        ayref="y{}".format(j + 1),
                        text="<b>Fb = {}N</b>".format(text),
                        textangle=90,
                        font=dict(size=20),
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=5,
                        arrowcolor="DarkSalmon",
                        ax=nodes_pos[nodes.index(node)],
                        ay=var * 2.5 * y_start,
                        row=row,
                        col=col,
                    )

            # plot disk forces
            for k, v in self.disk_forces.items():
                _, node = k.split("_")
                node = int(node)
                if node in nodes:
                    text = str(-v)
                    fig.add_annotation(
                        x=nodes_pos[nodes.index(node)],
                        y=0,
                        axref="x{}".format(j + 1),
                        ayref="y{}".format(j + 1),
                        text="<b>Fd = {}N</b>".format(text),
                        textangle=270,
                        font=dict(size=20),
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=5,
                        arrowcolor="FireBrick",
                        ax=nodes_pos[nodes.index(node)],
                        ay=2.5 * y_start,
                        row=row,
                        col=col,
                    )

            fig.update_xaxes(
                title_text="<b>Shaft Length</b>",
                title_font=dict(family="Arial", size=20),
                tickfont=dict(size=16),
                showgrid=False,
                showline=True,
                linewidth=2.5,
                linecolor="black",
                row=row,
                col=col,
            )
            fig.update_yaxes(
                visible=False, gridcolor="lightgray", showline=False, row=row, col=col,
            )
            j += 1

        fig.update_layout(**kwargs)

        return fig

    def plot_shearing_force(self, **kwargs):
        """Plot the rotor shearing force diagram.

        This method plots:
            shearing force diagram.

        Parameters
        ----------
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        shaft_end = max([sublist[-1] for sublist in self.nodes_pos])

        # fig - plot centerline
        fig.add_trace(
            go.Scatter(
                x=[-0.1 * shaft_end, 1.1 * shaft_end],
                y=[0, 0],
                mode="lines",
                line=dict(width=3.0, color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        j = 0
        for Vx, Vx_axis in zip(self.Vx, self.Vx_axis):
            fig.add_trace(
                go.Scatter(
                    x=Vx_axis,
                    y=Vx,
                    mode="lines",
                    line=dict(width=5.0, color=colors1[j]),
                    name="Shaft {}".format(j),
                    legendgroup="Shaft {}".format(j),
                    showlegend=True,
                    hovertemplate=(
                        "Shaft lengh: %{x:.2f}<br>" + "Shearing Force: %{y:.2f}"
                    ),
                )
            )
            j += 1

        fig.update_xaxes(
            title_text="<b>Shaft Length</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            range=[-0.1 * shaft_end, 1.1 * shaft_end],
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Force</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            title=dict(text="<b>Shearing Force Diagram</b>", font=dict(size=16)),
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **kwargs,
        )

        return fig

    def plot_bending_moment(self, **kwargs):
        """Plot the rotor bending moment diagram.

        This method plots:
            bending moment diagram.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            Plotly figure with the bending moment diagram plot
        """
        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        shaft_end = max([sublist[-1] for sublist in self.nodes_pos])

        # fig - plot centerline
        fig.add_trace(
            go.Scatter(
                x=[-0.1 * shaft_end, 1.1 * shaft_end],
                y=[0, 0],
                mode="lines",
                line=dict(width=3.0, color="black", dash="dashdot"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        j = 0
        for Bm, nodes_pos, nodes in zip(self.Bm, self.nodes_pos, self.nodes):
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
                fig.add_trace(
                    go.Scatter(
                        x=xnew_BM,
                        y=ynew_BM,
                        mode="lines",
                        line=dict(width=5.0, color=colors1[j]),
                        name="Shaft {}".format(j),
                        legendgroup="Shaft {}".format(j),
                        showlegend=True if i == 0 else False,
                        hovertemplate=(
                            "Shaft lengh: %{x:.2f}<br>" + "Bending Moment: %{y:.2f}"
                        ),
                    )
                )
                i += 2
            j += 1

        fig.update_xaxes(
            title_text="<b>Shaft Length</b>",
            title_font=dict(family="Arial", size=20),
            range=[-0.1 * shaft_end, 1.1 * shaft_end],
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Bending Moment</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            title=dict(text="<b>Bending Moment Diagram</b>", font=dict(size=16)),
            legend=dict(
                font=dict(family="sans-serif", size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=2,
            ),
            **kwargs,
        )

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
                "<b>Rotor data</b>",
                "<b>Shaft Element data</b>",
                "<b>Disk Element data</b>",
                "<b>Bearing Element data</b>",
            ],
        )
        colors = ["#ffffff", "#c4d9ed"]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>{}</b>".format(k) for k in rotor_data.keys()],
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
                    values=["<b>{}</b>".format(k) for k in shaft_data.keys()],
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
                    values=["<b>{}</b>".format(k) for k in disk_data.keys()],
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
                    values=["<b>{}</b>".format(k) for k in bearing_data.keys()],
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

    def plot(self, **kwargs):
        """Plot convergence results.

        This method plots:
            Natural Frequency vs Number of Elements
            Relative Error vs Number of Elements

        Parameters
        ----------
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.make_subplots()
            The figure object with the plot.
        """
        kwargs_default_values = dict(plot_bgcolor="white", hoverlabel_align="right")

        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "<b>Frequency Evaluation</b>",
                "<b>Relative Error Evaluation</b>",
            ],
        )

        # plot Frequency vs number of elements
        fig.add_trace(
            go.Scatter(
                x=self.el_num,
                y=self.eigv_arr,
                mode="lines+markers",
                line=dict(width=3.0, color="FireBrick"),
                marker=dict(size=10, color="FireBrick"),
                hovertemplate=(
                    "Number of Elements: %{x:.2f}<br>" + "Frequency: %{y:.0f}"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="<b>Number of Elements</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="<b>Frequency</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
            row=1,
            col=1,
        )

        # plot Error vs number of elements
        fig.add_trace(
            go.Scatter(
                x=self.el_num,
                y=self.error_arr,
                mode="lines+markers",
                line=dict(width=3.0, color="FireBrick"),
                marker=dict(size=10, color="FireBrick"),
                hovertemplate=(
                    "Number of Elements: %{x:.2f}<br>" + "Relative Error: %{y:.0f}"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(
            title_text="<b>Number of Elements</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
            row=1,
            col=2,
        )
        fig.update_yaxes(
            title_text="<b>Relative Error (%)</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            exponentformat="power",
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
            row=1,
            col=2,
        )

        fig.update_layout(**kwargs)

        return fig


class TimeResponseResults:
    """Class used to store results and provide plots for Time Response Analysis.

    This class takes the results from time response analysis and creates a
    plots given a force and a time. It's possible to select through a time response for
    a single DoF, an orbit response for a single node or display orbit response for all
    nodes.
    The plot type options are:
        - 1d: plot time response for a given degree of freedom of a rotor system.
        - 2d: plot orbit of a selected node of a rotor system.
        - 3d: plot orbits for each node on the rotor system in a 3D view.

    If plot_type = "1d": input a dof.
    If plot_type = "2d": input a node.
    if plot_type = "3d": no need to input a dof or node.

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

    def _plot1d(self, dof, **kwargs):
        """Plot time response for a single DoF using Plotly.

        This function will take a rotor object and plot its time response using Plotly.

        Parameters
        ----------
        dof : int
            Degree of freedom that will be observed.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
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

        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.t,
                y=self.yout[:, dof],
                mode="lines",
                line=dict(width=3.0, color="royalblue"),
                name="Phase",
                legendgroup="Phase",
                showlegend=False,
                hovertemplate=("Time: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
            )
        )

        fig.update_xaxes(
            title_text="<b>Time</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            range=[0, self.t[-1]],
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
            title=dict(
                text="<b>Response for node {} - DoF {}</b>".format(dof // 4, obs_dof),
                font=dict(size=20),
            ),
            **kwargs,
        )

        return fig

    def _plot2d(self, node, **kwargs):
        """Plot orbit response (2D).

        This function will take a rotor object and plot its orbit response using Plotly.

        Parameters
        ----------
        node: int, optional
            Selected node to plot orbit.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        kwargs_default_values = dict(
            width=1200, height=900, plot_bgcolor="white", hoverlabel_align="right",
        )
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.yout[:, self.number_dof * node],
                y=self.yout[:, self.number_dof * node + 1],
                mode="lines",
                line=dict(width=3.0, color="royalblue"),
                name="Phase",
                legendgroup="Phase",
                showlegend=False,
                hovertemplate=(
                    "X - Amplitude: %{x:.2e}<br>" + "Y - Amplitude: %{y:.2e}"
                ),
            )
        )

        fig.update_xaxes(
            title_text="<b>Amplitude - X direction</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<b>Amplitude - Y direction</b>",
            title_font=dict(family="Arial", size=20),
            tickfont=dict(size=16),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            title=dict(
                text="<b>Response for node {}</b>".format(node), font=dict(size=20)
            ),
            **kwargs,
        )

        return fig

    def _plot3d(self, **kwargs):
        """Plot orbit response (3D).

        This function will take a rotor object and plot its orbit response using Plotly.

        Parameters
        ----------
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. hoverlabel_align="center", ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        kwargs_default_values = dict(hoverlabel_align="right")
        for k, v in kwargs_default_values.items():
            kwargs.setdefault(k, v)

        fig = go.Figure()

        for n in self.nodes_list:
            x_pos = np.ones(self.yout.shape[0]) * self.nodes_pos[n]
            fig.add_trace(
                go.Scatter3d(
                    x=x_pos,
                    y=self.yout[:, self.number_dof * n],
                    z=self.yout[:, self.number_dof * n + 1],
                    mode="lines",
                    line=dict(width=3.0, color="royalblue"),
                    name="Mean",
                    legendgroup="mean",
                    showlegend=False,
                    hovertemplate=(
                        "Nodal Position: %{x:.2f}<br>"
                        + "X - Amplitude: %{y:.2e}<br>"
                        + "Y - Amplitude: %{z:.2e}"
                    ),
                    **kwargs,
                )
            )

        # plot center line
        line = np.zeros(len(self.nodes_pos))

        fig.add_trace(
            go.Scatter3d(
                x=self.nodes_pos,
                y=line,
                z=line,
                mode="lines",
                line=dict(width=2.0, color="black", dash="dashdot"),
                showlegend=False,
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
            **kwargs,
        )

        return fig

    def plot(self, plot_type="3d", dof=None, node=None, **kwargs):
        """Plot time response.

        The plot type options are:
            - 1d: plot time response for a given degree of freedom of a rotor system.
            - 2d: plot orbit of a selected node of a rotor system.
            - 3d: plot orbits for each node on the rotor system in a 3D view.

        Parameters
        ----------
        plot_type: str
            String to select the plot type.
            - "1d": plot time response for a given degree of freedom of a rotor system.
            - "2d": plot orbit of a selected node of a rotor system.
            - "3d": plot orbits for each node on the rotor system in a 3D view.
            Default is 3d.
        dof : int
            Degree of freedom that will be observed.
            Fill this attribute only when selection plot_type = "1d".
            Default is None.
        node: int, optional
            Selected node to plot orbit.
            Fill this attribute only when selection plot_type = "2d".
            Default is None
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Raises
        ------
        ValueError
            Error raised if a non valid string is passed to plot_type.
        ValueError
            Error raised if no node is specified or an odd node is passed
            when plot_type = "2d".
        ValueError
            Error raised if no dof is specified or an odd dof is passed
            when plot_type = "1d".

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        if plot_type == "3d":
            return self._plot3d(**kwargs)
        elif plot_type == "2d":
            if node is None:
                raise Exception("Select a node to plot orbit when plot_type '2d'")
            elif node not in self.nodes_list:
                raise Exception("Select a valid node to plot 2D orbit")
            return self._plot2d(node=node, **kwargs)
        elif plot_type == "1d":
            if dof is None:
                raise Exception("Select a dof to plot orbit when plot_type == '1d'")
            return self._plot1d(dof=dof, **kwargs)
        else:
            raise ValueError(f"{plot_type} is not a valid plot type.")
