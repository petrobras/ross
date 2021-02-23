"""ROSS plotting module.

This module returns graphs for each type of analyses in rotor_assembly.py.
"""
import copy
import inspect
from abc import ABC
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import toml
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import linalg as la

from ross.plotly_theme import tableau_colors
from ross.units import Q_
from ross.utils import intersection

__all__ = [
    "CriticalSpeedResults",
    "ModalResults",
    "CampbellResults",
    "FrequencyResponseResults",
    "ForcedResponseResults",
    "StaticResults",
    "SummaryResults",
    "ConvergenceResults",
    "TimeResponseResults",
]


class Results(ABC):
    """Results class.

    This class is a general abstract class to be implemented in other classes
    for post-processing results, in order to add saving and loading data options.
    """

    def save(self, file):
        """Save results in a .toml file.

        This function will save the simulation results to a .toml file.
        The file will have all the argument's names and values that are needed to
        reinstantiate the class.

        Parameters
        ----------
        file : str, pathlib.Path
            The name of the file the results will be saved in.

        Examples
        --------
        >>> # Example running a unbalance response
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> import ross as rs

        >>> # Running an example
        >>> rotor = rs.rotor_example()
        >>> speed = np.linspace(0, 1000, 101)
        >>> response = rotor.run_unbalance_response(node=3,
        ...                                         unbalance_magnitude=0.001,
        ...                                         unbalance_phase=0.0,
        ...                                         frequency=speed)

        >>> # create path for a temporary file
        >>> file = Path(tempdir) / 'unb_resp.toml'
        >>> response.save(file)
        """
        # get __init__ arguments
        signature = inspect.signature(self.__init__)
        args_list = list(signature.parameters)
        args = {arg: getattr(self, arg) for arg in args_list}
        try:
            data = toml.load(file)
        except FileNotFoundError:
            data = {}

        data[f"{self.__class__.__name__}"] = args
        with open(file, "w") as f:
            toml.dump(data, f, encoder=toml.TomlNumpyEncoder())

        if "rotor" in args.keys():
            aux_file = str(file)[:-5] + "_rotor" + str(file)[-5:]
            args["rotor"].save(aux_file)

    @classmethod
    def read_toml_data(cls, data):
        """Read and parse data stored in a .toml file.

        The data passed to this method needs to be according to the
        format saved in the .toml file by the .save() method.

        Parameters
        ----------
        data : dict
            Dictionary obtained from toml.load().

        Returns
        -------
        The result object.
        """
        return cls(**data)

    @classmethod
    def load(cls, file):
        """Load results from a .toml file.

        This function will load the simulation results from a .toml file.
        The file must have all the argument's names and values that are needed to
        reinstantiate the class.

        Parameters
        ----------
        file : str, pathlib.Path
            The name of the file the results will be loaded from.

        Examples
        --------
        >>> # Example running a stochastic unbalance response
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> import ross as rs

        >>> # Running an example
        >>> rotor = rs.rotor_example()
        >>> freq_range = np.linspace(0, 500, 31)
        >>> n = 3
        >>> m = 0.01
        >>> p = 0.0
        >>> results = rotor.run_unbalance_response(n, m, p, freq_range)

        >>> # create path for a temporary file
        >>> file = Path(tempdir) / 'unb_resp.toml'
        >>> results.save(file)

        >>> # Loading file
        >>> results2 = rs.ForcedResponseResults.load(file)
        >>> abs(results2.forced_resp).all() == abs(results.forced_resp).all()
        True
        """
        str_type = [np.dtype(f"<U4{i}") for i in range(10)]

        data = toml.load(file)
        data = list(data.values())[0]
        for key, value in data.items():
            if key == "rotor":
                aux_file = str(file)[:-5] + "_rotor" + str(file)[-5:]
                from ross.rotor_assembly import Rotor

                data[key] = Rotor.load(aux_file)

            elif isinstance(value, Iterable):
                data[key] = np.array(value)
                if data[key].dtype in str_type:
                    data[key] = np.array(value).astype(np.complex128)

        return cls.read_toml_data(data)


class CriticalSpeedResults(Results):
    """Class used to store results from run_critical_speed() method.

    Parameters
    ----------
    _wn : array
        Undamped critical speeds array.
    _wd : array
        Undamped critical speeds array.
    log_dec : array
        Logarithmic decrement for each critical speed.
    damping_ratio : array
        Damping ratio for each critical speed.
    whirl_direction : array
        Whirl direction for each critical speed. Can be forward, backward or mixed.
    """

    def __init__(self, _wn, _wd, log_dec, damping_ratio, whirl_direction):
        self._wn = _wn
        self._wd = _wd
        self.log_dec = log_dec
        self.damping_ratio = damping_ratio
        self.whirl_direction = whirl_direction

    def wn(self, frequency_units="rad/s"):
        """Convert units for undamped critical speeds.

        Parameters
        ----------
        frequency_units : str, optional
            Critical speeds units.
            Default is rad/s

        Returns
        -------
        wn : array
            Undamped critical speeds array.
        """
        return Q_(self.__dict__["_wn"], "rad/s").to(frequency_units).m

    def wd(self, frequency_units="rad/s"):
        """Convert units for damped critical speeds.

        Parameters
        ----------
        frequency_units : str, optional
            Critical speeds units.
            Default is rad/s

        Returns
        -------
        wd : array
            Undamped critical speeds array.
        """
        return Q_(self.__dict__["_wd"], "rad/s").to(frequency_units).m


class ModalResults(Results):
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
        self,
        mode=None,
        evec=None,
        fig=None,
        frequency_type="wd",
        title=None,
        length_units="m",
        frequency_units="rad/s",
        **kwargs,
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
        frequency_type : str, optional
            "wd" calculates de map for the damped natural frequencies.
            "wn" calculates de map for the undamped natural frequencies.
            Defaults is "wd".
        title : str, optional
            A brief title to the mode shape plot, it will be displayed above other
            relevant data in the plot area. It does not modify the figure layout from
            Plotly.
        length_units : str, optional
            length units.
            Default is 'm'.
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

        # fmt: off
        frequency = {
            "wd":  f"ω<sub>d</sub> = {Q_(self.wd[mode], 'rad/s').to(frequency_units).m:.1f}",
            "wn":  f"ω<sub>n</sub> = {Q_(self.wn[mode], 'rad/s').to(frequency_units).m:.1f}",
            "speed": f"Speed = {Q_(self.speed, 'rad/s').to(frequency_units).m:.1f}",
        }
        # fmt: on

        for node in nodes:
            fig.add_trace(
                go.Scatter3d(
                    x=Q_(zc_pos[10:, node], "m").to(length_units).m,
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
                    x=Q_([zc_pos[10, node]], "m").to(length_units).m,
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
                x=Q_(zn, "m").to(length_units).m,
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
                x=Q_(zn_cl, "m").to(length_units).m,
                y=zn_cl * 0,
                z=zn_cl * 0,
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                hoverinfo="none",
                showlegend=False,
            )
        )

        if title is None:
            title = ""

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=dict(text=f"Rotor Length ({length_units})"),
                    autorange="reversed",
                    nticks=5,
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
                    f"{title}<br>"
                    f"Mode {mode + 1} | "
                    f"{frequency['speed']} {frequency_units} | "
                    f"whirl: {self.whirl_direction()[mode]} | "
                    f"{frequency[frequency_type]} {frequency_units} | "
                    f"Log. Dec. = {self.log_dec[mode]:.1f} | "
                    f"Damping ratio = {self.damping_ratio[mode]:.2f}"
                )
            ),
            **kwargs,
        )

        return fig

    def plot_mode_2d(
        self,
        mode=None,
        evec=None,
        fig=None,
        frequency_type="wd",
        title=None,
        length_units="m",
        frequency_units="rad/s",
        **kwargs,
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
        frequency_type : str, optional
            "wd" calculates de map for the damped natural frequencies.
            "wn" calculates de map for the undamped natural frequencies.
            Defaults is "wd".
        title : str, optional
            A brief title to the mode shape plot, it will be displayed above other
            relevant data in the plot area. It does not modify the figure layout from
            Plotly.
        length_units : str, optional
            length units.
            Default is 'm'.
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
        nodes_pos = Q_(self.nodes_pos, "m").to(length_units).m

        theta = np.arctan(xn[0] / yn[0])
        vn = xn * np.sin(theta) + yn * np.cos(theta)

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

        # fmt: off
        frequency = {
            "wd":  f"ω<sub>d</sub> = {Q_(self.wd[mode], 'rad/s').to(frequency_units).m:.1f}",
            "wn":  f"ω<sub>n</sub> = {Q_(self.wn[mode], 'rad/s').to(frequency_units).m:.1f}",
            "speed": f"Speed = {Q_(self.speed, 'rad/s').to(frequency_units).m:.1f}",
        }
        # fmt: on
        whirl_dir = colors[self.whirl_direction()[mode]]

        fig.add_trace(
            go.Scatter(
                x=Q_(zn, "m").to(length_units).m,
                y=vn / vn[np.argmax(np.abs(vn))],
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

        if title is None:
            title = ""

        fig.update_xaxes(title_text=f"Rotor Length ({length_units})")
        fig.update_yaxes(title_text="Relative Displacement")
        fig.update_layout(
            title=dict(
                text=(
                    f"{title}<br>"
                    f"Mode {mode + 1} | "
                    f"{frequency['speed']} {frequency_units} | "
                    f"whirl: {self.whirl_direction()[mode]} | "
                    f"{frequency[frequency_type]} {frequency_units} | "
                    f"Log. Dec. = {self.log_dec[mode]:.1f} | "
                    f"Damping ratio = {self.damping_ratio[mode]:.2f}"
                )
            ),
            **kwargs,
        )

        return fig


class CampbellResults(Results):
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

        crit_x = []
        crit_y = []
        for i in range(num_frequencies):
            w_i = wd[:, i]

            for harm in harmonics:
                x1 = speed_range
                y1 = w_i
                x2 = speed_range
                y2 = harm * speed_range

                x, y = intersection(x1, y1, x2, y2)
                crit_x.extend(x)
                crit_y.extend(y)

        if len(crit_x) and len(crit_y):
            fig.add_trace(
                go.Scatter(
                    x=crit_x,
                    y=crit_y,
                    mode="markers",
                    marker=dict(symbol="x", color="black"),
                    name="Crit. Speed",
                    legendgroup="Crit. Speed",
                    showlegend=True,
                    hovertemplate=(
                        f"Frequency ({frequency_units}): %{{x:.2f}}<br>Critical Speed ({frequency_units}): %{{y:.2f}}"
                    ),
                )
            )

        scatter_marker = ["triangle-up", "circle", "triangle-down"]
        for mark, whirl_dir, legend in zip(
            scatter_marker, [0.0, 0.5, 1.0], ["Foward", "Mixed", "Backward"]
        ):
            for i in range(num_frequencies):
                w_i = wd[:, i]
                whirl_i = whirl[:, i]
                log_dec_i = log_dec[:, i]

                whirl_mask = whirl_i == whirl_dir
                if any(check for check in whirl_mask):
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
                else:
                    continue

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
        scatter_marker = ["triangle-up", "circle", "triangle-down"]
        legends = ["Foward", "Mixed", "Backward"]
        for mark, legend in zip(scatter_marker, legends):
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    name=legend,
                    legendgroup=legend,
                    marker=dict(symbol=mark, color="black"),
                    hoverinfo="none",
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


class FrequencyResponseResults(Results):
    """Class used to store results and provide plots for Frequency Response.

    Parameters
    ----------
    freq_resp : array
        Array with the frequency response (displacement).
    velc_resp : array
        Array with the frequency response (velocity).
    accl_resp : array
        Array with the frequency response (acceleration).
    speed_range : array
        Array with the speed range in rad/s.
    number_dof : int
        Number of degrees of freedom per node.

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with Amplitude vs Frequency and Phase vs Frequency plots.
    """

    def __init__(self, freq_resp, velc_resp, accl_resp, speed_range, number_dof):
        self.freq_resp = freq_resp
        self.velc_resp = velc_resp
        self.accl_resp = accl_resp
        self.speed_range = speed_range
        self.number_dof = number_dof

        if self.number_dof == 4:
            self.dof_dict = {"0": "x", "1": "y", "2": "α", "3": "β"}
        elif self.number_dof == 6:
            self.dof_dict = {"0": "x", "1": "y", "2": "z", "4": "α", "5": "β", "6": "θ"}

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
        It is possible to plot displacement, velocity and accelaration responses,
        depending on the unit entered in 'amplitude_units'. If '[length]/[force]',
        it displays the displacement; If '[speed]/[force]', it displays the velocity;
        If '[acceleration]/[force]', it displays the acceleration.

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
            Acceptable units are:
                '[length]/[force]' - Displays the displacement;
                '[speed]/[force]' - Displays the velocity;
                '[acceleration]/[force]' - Displays the acceleration.
            Default is "m/N" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m/N)
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
        inpn = inp // self.number_dof
        idof = self.dof_dict[str(inp % self.number_dof)]
        outn = out // self.number_dof
        odof = self.dof_dict[str(out % self.number_dof)]

        frequency_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        dummy_var = Q_(1, amplitude_units)
        if dummy_var.check("[length]/[force]"):
            mag = np.abs(self.freq_resp)
            mag = Q_(mag, "m/N").to(amplitude_units).m
            y_label = "Displacement"
        elif dummy_var.check("[speed]/[force]"):
            mag = np.abs(self.velc_resp)
            mag = Q_(mag, "m/s/N").to(amplitude_units).m
            y_label = "Velocity"
        elif dummy_var.check("[acceleration]/[force]"):
            mag = np.abs(self.accl_resp)
            mag = Q_(mag, "m/s**2/N").to(amplitude_units).m
            y_label = "Acceleration"
        else:
            raise ValueError(
                "Not supported unit. Options are '[length]/[force]', '[speed]/[force]', '[acceleration]/[force]'"
            )

        if fig is None:
            fig = go.Figure()
        idx = len(fig.data)

        fig.add_trace(
            go.Scatter(
                x=frequency_range,
                y=mag[inp, out, :],
                mode="lines",
                line=dict(color=list(tableau_colors)[idx]),
                name=f"inp: node {inpn} | dof: {idof}<br>out: node {outn} | dof: {odof}",
                legendgroup=f"inp: node {inpn} | dof: {idof}<br>out: node {outn} | dof: {odof}",
                showlegend=True,
                hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
            )
        )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(frequency_range), np.max(frequency_range)],
        )
        fig.update_yaxes(title_text=f"{y_label} ({amplitude_units})")
        fig.update_layout(**mag_kwargs)

        return fig

    def plot_phase(
        self,
        inp,
        out,
        frequency_units="rad/s",
        amplitude_units="m/N",
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
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units are:
                '[length]/[force]' - Displays the displacement;
                '[speed]/[force]' - Displays the velocity;
                '[acceleration]/[force]' - Displays the acceleration.
            Default is "m/N" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m/N)
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
        inpn = inp // self.number_dof
        idof = self.dof_dict[str(inp % self.number_dof)]
        outn = out // self.number_dof
        odof = self.dof_dict[str(out % self.number_dof)]

        frequency_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        dummy_var = Q_(1, amplitude_units)
        if dummy_var.check("[length]/[force]"):
            phase = np.angle(self.freq_resp[inp, out, :])
        elif dummy_var.check("[speed]/[force]"):
            phase = np.angle(self.velc_resp[inp, out, :])
        elif dummy_var.check("[acceleration]/[force]"):
            phase = np.angle(self.accl_resp[inp, out, :])
        else:
            raise ValueError(
                "Not supported unit. Options are '[length]/[force]', '[speed]/[force]', '[acceleration]/[force]'"
            )

        phase = Q_(phase, "rad").to(phase_units).m

        if phase_units in ["rad", "radian", "radians"]:
            phase = [i + 2 * np.pi if i < 0 else i for i in phase]
        else:
            phase = [i + 360 if i < 0 else i for i in phase]

        if fig is None:
            fig = go.Figure()
        idx = len(fig.data)

        fig.add_trace(
            go.Scatter(
                x=frequency_range,
                y=phase,
                mode="lines",
                line=dict(color=list(tableau_colors)[idx]),
                name=f"inp: node {inpn} | dof: {idof}<br>out: node {outn} | dof: {odof}",
                legendgroup=f"inp: node {inpn} | dof: {idof}<br>out: node {outn} | dof: {odof}",
                showlegend=True,
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
        It is possible to plot displacement, velocity and accelaration responses,
        depending on the unit entered in 'amplitude_units'. If '[length]/[force]',
        it displays the displacement; If '[speed]/[force]', it displays the velocity;
        If '[acceleration]/[force]', it displays the acceleration.

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
            Acceptable units are:
                '[length]/[force]' - Displays the displacement;
                '[speed]/[force]' - Displays the velocity;
                '[acceleration]/[force]' - Displays the acceleration.
            Default is "m/N" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m/N)
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
        inpn = inp // self.number_dof
        idof = self.dof_dict[str(inp % self.number_dof)]
        outn = out // self.number_dof
        odof = self.dof_dict[str(out % self.number_dof)]

        frequency_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        dummy_var = Q_(1, amplitude_units)
        if dummy_var.check("[length]/[force]"):
            mag = np.abs(self.freq_resp[inp, out, :])
            mag = Q_(mag, "m/N").to(amplitude_units).m
            phase = np.angle(self.freq_resp[inp, out, :])
            y_label = "Displacement"
        elif dummy_var.check("[speed]/[force]"):
            mag = np.abs(self.velc_resp[inp, out, :])
            mag = Q_(mag, "m/s/N").to(amplitude_units).m
            phase = np.angle(self.velc_resp[inp, out, :])
            y_label = "Velocity"
        elif dummy_var.check("[acceleration]/[force]"):
            mag = np.abs(self.accl_resp[inp, out, :])
            mag = Q_(mag, "m/s**2/N").to(amplitude_units).m
            phase = np.angle(self.accl_resp[inp, out, :])
            y_label = "Acceleration"
        else:
            raise ValueError(
                "Not supported unit. Options are '[length]/[force]', '[speed]/[force]', '[acceleration]/[force]'"
            )

        phase = Q_(phase, "rad").to(phase_units).m

        if phase_units in ["rad", "radian", "radians"]:
            polar_theta_unit = "radians"
            phase = [i + 2 * np.pi if i < 0 else i for i in phase]
        else:
            polar_theta_unit = "degrees"
            phase = [i + 360 if i < 0 else i for i in phase]

        if fig is None:
            fig = go.Figure()
        idx = len(fig.data)

        fig.add_trace(
            go.Scatterpolar(
                r=mag,
                theta=phase,
                customdata=frequency_range,
                thetaunit=polar_theta_unit,
                mode="lines+markers",
                marker=dict(color=list(tableau_colors)[idx]),
                line=dict(color=list(tableau_colors)[idx]),
                name=f"inp: node {inpn} | dof: {idof}<br>out: node {outn} | dof: {odof}",
                legendgroup=f"inp: node {inpn} | dof: {idof}<br>out: node {outn} | dof: {odof}",
                showlegend=True,
                hovertemplate=f"Amplitude ({amplitude_units}): %{{r:.2e}}<br>Phase: %{{theta:.2f}}<br>Frequency ({frequency_units}): %{{customdata:.2f}}",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title=dict(text=f"{y_label} ({amplitude_units})"),
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
        fig=None,
        mag_kwargs=None,
        phase_kwargs=None,
        polar_kwargs=None,
        fig_kwargs=None,
    ):
        """Plot frequency response.

        This method plots the frequency response given an output and an input
        using Plotly.

        This method returns a subplot with:
            - Frequency vs Amplitude;
            - Frequency vs Phase Angle;
            - Polar plot Amplitude vs Phase Angle;

        Amplitude can be displacement, velocity or accelaration responses,
        depending on the unit entered in 'amplitude_units'. If '[length]/[force]',
        it displays the displacement; If '[speed]/[force]', it displays the velocity;
        If '[acceleration]/[force]', it displays the acceleration.

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
            Acceptable units are:
                '[length]/[force]' - Displays the displacement;
                '[speed]/[force]' - Displays the velocity;
                '[acceleration]/[force]' - Displays the acceleration.
            Default is "m/N" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m/N)
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
        fig_kwargs : optional
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
        fig_kwargs = {} if fig_kwargs is None else copy.copy(fig_kwargs)

        fig0 = self.plot_magnitude(
            inp, out, frequency_units, amplitude_units, None, **mag_kwargs
        )
        fig1 = self.plot_phase(
            inp,
            out,
            frequency_units,
            amplitude_units,
            phase_units,
            None,
            **phase_kwargs,
        )
        fig2 = self.plot_polar_bode(
            inp,
            out,
            frequency_units,
            amplitude_units,
            phase_units,
            None,
            **polar_kwargs,
        )

        if fig is None:
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {"type": "polar", "rowspan": 2}], [{}, None]],
            )

        for data in fig0["data"]:
            fig.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            data.showlegend = False
            fig.add_trace(data, row=2, col=1)
        for data in fig2["data"]:
            data.showlegend = False
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
            **fig_kwargs,
        )

        return fig


class ForcedResponseResults(Results):
    """Class used to store results and provide plots for Forced Response analysis.

    Parameters
    ----------
    rotor : ross.Rotor object
        The Rotor object
    force_resp : array
        Array with the forced response (displacement) for each node for each frequency.
    velc_resp : array
        Array with the forced response (velocity) for each node for each frequency.
    accl_resp : array
        Array with the forced response (acceleration) for each node for each frequency.
    speed_range : array
        Array with the frequencies.
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
        self, rotor, forced_resp, velc_resp, accl_resp, speed_range, unbalance=None
    ):
        self.rotor = rotor
        self.forced_resp = forced_resp
        self.velc_resp = velc_resp
        self.accl_resp = accl_resp
        self.speed_range = speed_range
        self.unbalance = unbalance

        self.default_units = {
            "[length]": ["m", "forced_resp"],
            "[length] / [time]": ["m/s", "velc_resp"],
            "[length] / [time] ** 2": ["m/s**2", "accl_resp"],
        }

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
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
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

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            base_unit = self.default_units[unit_type][0]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            angle = Q_(p[1], probe_units).to("rad").m
            vector = self._calculate_major_axis_per_node(
                node=p[0], angle=angle, amplitude_units=amplitude_units
            )[3]
            try:
                probe_tag = p[2]
            except IndexError:
                probe_tag = f"Probe {i+1} - Node {p[0]}"

            fig.add_trace(
                go.Scatter(
                    x=frequency_range,
                    y=Q_(np.abs(vector), base_unit).to(amplitude_units).m,
                    mode="lines",
                    line=dict(color=list(tableau_colors)[i]),
                    name=probe_tag,
                    legendgroup=probe_tag,
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
        amplitude_units="m",
        phase_units="rad",
        fig=None,
        **kwargs,
    ):
        """Plot forced response (phase) using Plotly.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
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

        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            angle = Q_(p[1], probe_units).to("rad").m
            vector = self._calculate_major_axis_per_node(
                node=p[0], angle=angle, amplitude_units=amplitude_units
            )[4]

            probe_phase = np.real(vector)
            probe_phase = np.array([i + 2 * np.pi if i < 0 else i for i in probe_phase])
            probe_phase = Q_(probe_phase, "rad").to(phase_units).m

            try:
                probe_tag = p[2]
            except IndexError:
                probe_tag = f"Probe {i+1} - Node {p[0]}"

            fig.add_trace(
                go.Scatter(
                    x=frequency_range,
                    y=probe_phase,
                    mode="lines",
                    line=dict(color=list(tableau_colors)[i]),
                    name=probe_tag,
                    legendgroup=probe_tag,
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
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
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

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            base_unit = self.default_units[unit_type][0]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            angle = Q_(p[1], probe_units).to("rad").m
            vector = self._calculate_major_axis_per_node(
                node=p[0], angle=angle, amplitude_units=amplitude_units
            )

            probe_phase = np.real(vector[4])
            probe_phase = np.array([i + 2 * np.pi if i < 0 else i for i in probe_phase])
            probe_phase = Q_(probe_phase, "rad").to(phase_units).m

            if phase_units in ["rad", "radian", "radians"]:
                polar_theta_unit = "radians"
            elif phase_units in ["degree", "degrees", "deg"]:
                polar_theta_unit = "degrees"

            try:
                probe_tag = p[2]
            except IndexError:
                probe_tag = f"Probe {i+1} - Node {p[0]}"

            fig.add_trace(
                go.Scatterpolar(
                    r=Q_(np.abs(vector[3]), base_unit).to(amplitude_units).m,
                    theta=probe_phase,
                    customdata=frequency_range,
                    thetaunit=polar_theta_unit,
                    mode="lines+markers",
                    marker=dict(color=list(tableau_colors)[i]),
                    line=dict(color=list(tableau_colors)[i]),
                    name=probe_tag,
                    legendgroup=probe_tag,
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
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
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
            probe, probe_units, frequency_units, amplitude_units, phase_units, **phase_kwargs
        )
        fig2 = self.plot_polar_bode(
            probe, probe_units, frequency_units, amplitude_units, phase_units, **polar_kwargs
        )
        # fmt: on

        subplots = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {"type": "polar", "rowspan": 2}], [{}, None]],
            shared_xaxes=True,
            vertical_spacing=0.02,
        )
        for data in fig0["data"]:
            data.showlegend = False
            subplots.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            data.showlegend = False
            subplots.add_trace(data, row=2, col=1)
        for data in fig2["data"]:
            subplots.add_trace(data, row=1, col=2)

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

    def _calculate_major_axis_per_node(self, node, angle, amplitude_units="m"):
        """Calculate the major axis for a node for each frequency.

        Parameters
        ----------
        node : float
            A node from the rotor model.
        angle : float, str
            The orientation angle of the axis.
            Options are:
                float : angle in rad capture the response in a probe orientation;
                str : "major" to capture the response for the major axis;
                str : "minor" to capture the response for the minor axis.
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)

        Returns
        -------
        major_axis_vector : np.ndarray
            major_axis_vector[0, :] = foward vector
            major_axis_vector[1, :] = backward vector
            major_axis_vector[2, :] = axis angle
            major_axis_vector[3, :] = axis vector response for the input angle
            major_axis_vector[4, :] = phase response for the input angle
        """
        ndof = self.rotor.number_dof
        nodes = self.rotor.nodes
        link_nodes = self.rotor.link_nodes

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            response = self.__dict__[self.default_units[unit_type][1]]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        major_axis_vector = np.zeros((5, len(self.speed_range)), dtype=complex)

        fix_dof = (node - nodes[-1] - 1) * ndof // 2 if node in link_nodes else 0
        dofx = ndof * node - fix_dof
        dofy = ndof * node + 1 - fix_dof

        # Relative angle between probes (90°)
        Rel_ang = np.exp(1j * np.pi / 2)

        for i, f in enumerate(self.speed_range):

            # Foward and Backward vectors
            fow = response[dofx, i] / 2 + Rel_ang * response[dofy, i] / 2
            back = (
                np.conj(response[dofx, i]) / 2
                + Rel_ang * np.conj(response[dofy, i]) / 2
            )

            ang_fow = np.angle(fow)
            if ang_fow < 0:
                ang_fow += 2 * np.pi

            ang_back = np.angle(back)
            if ang_back < 0:
                ang_back += 2 * np.pi

            if angle == "major":
                # Major axis angle
                axis_angle = (ang_back - ang_fow) / 2
                if axis_angle > np.pi:
                    axis_angle -= np.pi

            elif angle == "minor":
                # Minor axis angle
                axis_angle = (ang_back - ang_fow + np.pi) / 2
                if axis_angle > np.pi:
                    axis_angle -= np.pi

            else:
                axis_angle = angle

            major_axis_vector[0, i] = fow
            major_axis_vector[1, i] = back
            major_axis_vector[2, i] = axis_angle
            major_axis_vector[3, i] = np.abs(
                fow * np.exp(1j * axis_angle) + back * np.exp(-1j * axis_angle)
            )
            major_axis_vector[4, i] = np.angle(
                fow * np.exp(1j * axis_angle) + back * np.exp(-1j * axis_angle)
            )

        return major_axis_vector

    def _calculate_major_axis_per_speed(self, speed, amplitude_units="m"):
        """Calculate the major axis for each nodal orbit.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class (rad/s).
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)

        Returns
        -------
        major_axis_vector : np.ndarray
            major_axis_vector[0, :] = foward vector
            major_axis_vector[1, :] = backward vector
            major_axis_vector[2, :] = major axis angle
            major_axis_vector[3, :] = major axis vector for the maximum major axis angle
            major_axis_vector[4, :] = absolute values for major axes vectors
        """
        nodes = self.rotor.nodes
        ndof = self.rotor.number_dof

        major_axis_vector = np.zeros((5, len(nodes)), dtype=complex)
        idx = np.where(np.isclose(self.speed_range, speed, atol=1e-6))[0][0]

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            response = self.__dict__[self.default_units[unit_type][1]]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        for i, n in enumerate(nodes):
            dofx = ndof * n
            dofy = ndof * n + 1

            # Relative angle between probes (90°)
            Rel_ang = np.exp(1j * np.pi / 2)

            # Foward and Backward vectors
            fow = response[dofx, idx] / 2 + Rel_ang * response[dofy, idx] / 2
            back = (
                np.conj(response[dofx, idx]) / 2
                + Rel_ang * np.conj(response[dofy, idx]) / 2
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

            major_axis_vector[0, i] = fow
            major_axis_vector[1, i] = back
            major_axis_vector[2, i] = ang_maj_ax

        max_major_axis_angle = np.max(major_axis_vector[2])

        # fmt: off
        major_axis_vector[3] = (
            major_axis_vector[0] * np.exp(1j * max_major_axis_angle) +
            major_axis_vector[1] * np.exp(-1j * max_major_axis_angle)
        )
        major_axis_vector[4] = np.abs(
            major_axis_vector[0] * np.exp(1j * major_axis_vector[2]) +
            major_axis_vector[1] * np.exp(-1j * major_axis_vector[2])
        )
        # fmt: on

        return major_axis_vector

    def _calculate_bending_moment(self, speed):
        """Calculate the bending moment in X and Y directions.

        This method calculate forces and moments on nodal positions for a deflected
        shape configuration.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class (rad/s).

        Returns
        -------
        Mx : array
            Bending Moment on X directon.
        My : array
            Bending Moment on Y directon.
        """
        idx = np.where(np.isclose(self.speed_range, speed, atol=1e-6))[0][0]
        mag = np.abs(self.forced_resp[:, idx])
        phase = np.angle(self.forced_resp[:, idx])
        number_dof = self.rotor.number_dof
        nodes = self.rotor.nodes

        Mx = np.zeros_like(nodes, dtype=np.float64)
        My = np.zeros_like(nodes, dtype=np.float64)
        mag = mag * np.cos(-phase)

        # fmt: off
        for i, el in enumerate(self.rotor.shaft_elements):
            x = (-el.material.E * el.Ie / el.L ** 2) * np.array([
                [-6, +6, -4 * el.L, -2 * el.L],
                [+6, -6, +2 * el.L, +4 * el.L],
            ])
            response_x = np.array([
                [mag[number_dof * el.n_l + 0]],
                [mag[number_dof * el.n_r + 0]],
                [mag[number_dof * el.n_l + 3]],
                [mag[number_dof * el.n_r + 3]],
            ])

            Mx[[el.n_l, el.n_r]] += (x @ response_x).flatten()

            y = (-el.material.E * el.Ie / el.L ** 2) * np.array([
                [-6, +6, +4 * el.L, +2 * el.L],
                [+6, -6, -2 * el.L, -4 * el.L],
            ])
            response_y = np.array([
                [mag[number_dof * el.n_l + 1]],
                [mag[number_dof * el.n_r + 1]],
                [mag[number_dof * el.n_l + 2]],
                [mag[number_dof * el.n_r + 2]],
            ])
            My[[el.n_l, el.n_r]] += (y @ response_y).flatten()
        # fmt: on

        return Mx, My

    def plot_deflected_shape_2d(
        self,
        speed,
        frequency_units="rad/s",
        amplitude_units="m",
        rotor_length_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot the 2D deflected shape diagram.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class (rad/s).
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
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
        if not any(np.isclose(self.speed_range, speed, atol=1e-6)):
            raise ValueError("No data available for this speed value.")

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            base_unit = self.default_units[unit_type][0]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        nodes_pos = Q_(self.rotor.nodes_pos, "m").to(rotor_length_units).m
        maj_vect = self._calculate_major_axis_per_speed(speed, amplitude_units)
        maj_vect = Q_(maj_vect[4].real, base_unit).to(amplitude_units).m

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
                hovertemplate=f"Nodal Position ({rotor_length_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
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
            title_text=f"Major Axis Abs Amplitude ({amplitude_units})",
            title_font=dict(size=12),
        )
        fig.update_layout(**kwargs)

        return fig

    def plot_deflected_shape_3d(
        self,
        speed,
        samples=101,
        frequency_units="rad/s",
        amplitude_units="m",
        rotor_length_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot the 3D deflected shape diagram.

        Parameters
        ----------
        speed : float
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class (rad/s).
        samples : int, optional
            Number of samples to generate the orbit for each node.
            Default is 101.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
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
        if not any(np.isclose(self.speed_range, speed, atol=1e-6)):
            raise ValueError("No data available for this speed value.")

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            base_unit = self.default_units[unit_type][0]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        mag = np.abs(self.__dict__[self.default_units[unit_type][1]])
        phase = np.angle(self.__dict__[self.default_units[unit_type][1]])
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
                    y=Q_(y, base_unit).to(amplitude_units).m,
                    z=Q_(z, base_unit).to(amplitude_units).m,
                    mode="lines",
                    line=dict(color="royalblue"),
                    name="Orbit",
                    legendgroup="Orbit",
                    showlegend=False,
                    hovertemplate=(
                        f"Position ({rotor_length_units}): %{{x:.2f}}<br>X - Amplitude ({amplitude_units}): %{{y:.2e}}<br>Y - Amplitude ({amplitude_units}): %{{z:.2e}}"
                    ),
                )
            )

        # plot major axis
        maj_vect = self._calculate_major_axis_per_speed(speed, amplitude_units)

        fig.add_trace(
            go.Scatter3d(
                x=x_pos[:, 0],
                y=Q_(np.real(maj_vect[3]), base_unit).to(amplitude_units).m,
                z=Q_(np.imag(maj_vect[3]), base_unit).to(amplitude_units).m,
                mode="lines+markers",
                marker=dict(color="black"),
                line=dict(color="black", dash="dashdot"),
                name="Major Axis",
                legendgroup="Major_Axis",
                showlegend=True,
                hovertemplate=(
                    f"Position ({rotor_length_units}): %{{x:.2f}}<br>X - Amplitude ({amplitude_units}): %{{y:.2e}}<br>Y - Amplitude ({amplitude_units}): %{{z:.2e}}"
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
                    y=Q_([0, np.amax(np.abs(maj_vect[4])) / 2 * np.cos(p)], base_unit)
                    .to(amplitude_units)
                    .m,
                    z=Q_([0, np.amax(np.abs(maj_vect[4])) / 2 * np.sin(p)], base_unit)
                    .to(amplitude_units)
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
                    y=Q_([np.amax(np.abs(maj_vect[4])) / 2 * np.cos(p)], base_unit)
                    .to(amplitude_units)
                    .m,
                    z=Q_([np.amax(np.abs(maj_vect[4])) / 2 * np.sin(p)], base_unit)
                    .to(amplitude_units)
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
                yaxis=dict(title=dict(text=f"Amplitude - X ({amplitude_units})")),
                zaxis=dict(title=dict(text=f"Amplitude - Y ({amplitude_units})")),
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
            passed to the class (rad/s).
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
                name=f"Bending Moment (abs) ({moment_units})",
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
        fig.update_yaxes(
            title_text=f"Bending Moment ({moment_units})",
            title_font=dict(size=12),
        )
        fig.update_layout(**kwargs)

        return fig

    def plot_deflected_shape(
        self,
        speed,
        samples=101,
        frequency_units="rad/s",
        amplitude_units="m",
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
            passed to the class (rad/s).
        samples : int, optional
            Number of samples to generate the orbit for each node.
            Default is 101.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
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
        speed_str = Q_(speed, "rad/s").to(frequency_units).m

        # fmt: off
        fig0 = self.plot_deflected_shape_2d(
            speed, frequency_units, amplitude_units, rotor_length_units, **shape2d_kwargs
        )
        fig1 = self.plot_deflected_shape_3d(
            speed, samples, frequency_units, amplitude_units, rotor_length_units, **shape3d_kwargs
        )
        fig2 = self.plot_bending_moment(
            speed, frequency_units, moment_units, rotor_length_units, **bm_kwargs
        )
        # fmt: on

        subplots = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {"type": "scene", "rowspan": 2}], [{}, None]],
            shared_xaxes=True,
            vertical_spacing=0.02,
        )
        for data in fig0["data"]:
            subplots.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            subplots.add_trace(data, row=1, col=2)
        for data in fig2["data"]:
            subplots.add_trace(data, row=2, col=1)

        subplots.update_yaxes(fig0.layout.yaxis, row=1, col=1)
        subplots.update_xaxes(fig2.layout.xaxis, row=2, col=1)
        subplots.update_yaxes(fig2.layout.yaxis, row=2, col=1)
        subplots.update_layout(
            scene=dict(
                bgcolor=fig1.layout.scene.bgcolor,
                xaxis=fig1.layout.scene.xaxis,
                yaxis=fig1.layout.scene.yaxis,
                zaxis=fig1.layout.scene.zaxis,
                domain=dict(x=[0.47, 1]),
            ),
            title=dict(
                text=f"Deflected Shape<br>Speed = {speed_str} {frequency_units}",
            ),
            legend=dict(
                orientation="h",
                xanchor="center",
                yanchor="bottom",
                x=0.5,
                y=-0.3,
            ),
            **subplot_kwargs,
        )

        return subplots


class StaticResults(Results):
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


class SummaryResults(Results):
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


class ConvergenceResults(Results):
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


class TimeResponseResults(Results):
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

    def __init__(self, rotor, t, yout, xout):
        self.t = t
        self.yout = yout
        self.xout = xout
        self.rotor = rotor

    def plot_1d(
        self,
        probe,
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        """Plot time response.

        This method plots the time response given a tuple of probes with their nodes
        and orientations.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
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
        nodes = self.rotor.nodes
        link_nodes = self.rotor.link_nodes
        ndof = self.rotor.number_dof

        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            fix_dof = (p[0] - nodes[-1] - 1) * ndof // 2 if p[0] in link_nodes else 0
            dofx = ndof * p[0] - fix_dof
            dofy = ndof * p[0] + 1 - fix_dof

            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )

            _probe_resp = operator @ np.vstack((self.yout[:, dofx], self.yout[:, dofy]))
            probe_resp = (
                _probe_resp[0] * np.cos(angle) ** 2 +
                _probe_resp[1] * np.sin(angle) ** 2
            )
            # fmt: on

            probe_resp = Q_(probe_resp, "m").to(displacement_units).m

            try:
                probe_tag = p[2]
            except IndexError:
                probe_tag = f"Probe {i+1} - Node {p[0]}"

            fig.add_trace(
                go.Scatter(
                    x=Q_(self.t, "s").to(time_units).m,
                    y=Q_(probe_resp, "m").to(displacement_units).m,
                    mode="lines",
                    name=probe_tag,
                    legendgroup=probe_tag,
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
        nodes = self.rotor.nodes
        link_nodes = self.rotor.link_nodes
        ndof = self.rotor.number_dof

        fix_dof = (node - nodes[-1] - 1) * ndof // 2 if node in link_nodes else 0
        dofx = ndof * node - fix_dof
        dofy = ndof * node + 1 - fix_dof

        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=Q_(self.yout[:, dofx], "m").to(displacement_units).m,
                y=Q_(self.yout[:, dofy], "m").to(displacement_units).m,
                mode="lines",
                name="Orbit",
                legendgroup="Orbit",
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
        nodes_pos = self.rotor.nodes_pos
        nodes = self.rotor.nodes
        ndof = self.rotor.number_dof

        if fig is None:
            fig = go.Figure()

        for n in nodes:
            x_pos = np.ones(self.yout.shape[0]) * nodes_pos[n]
            fig.add_trace(
                go.Scatter3d(
                    x=Q_(x_pos, "m").to(rotor_length_units).m,
                    y=Q_(self.yout[:, ndof * n], "m").to(displacement_units).m,
                    z=Q_(self.yout[:, ndof * n + 1], "m").to(displacement_units).m,
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
        line = np.zeros(len(nodes_pos))

        fig.add_trace(
            go.Scatter3d(
                x=Q_(nodes_pos, "m").to(rotor_length_units).m,
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


class UCSResults(Results):
    """Class used to store results and provide plots for UCS Analysis.

    Parameters
    ----------
    stiffness_range : tuple, optional
        Tuple with (start, end) for stiffness range.
    stiffness_log : tuple, optional
        Evenly numbers spaced evenly on a log scale to create a better visualization
        (see np.logspace).
    wn : array
        Undamped natural frequencies array.
    bearing : ross.BearingElement
        Bearing used in the calculation.
    intersection_points : array
        Points where there is a intersection between undamped natural frequency and
        the bearing stiffness.
    """

    def __init__(
        self, stiffness_range, stiffness_log, wn, bearing, intersection_points
    ):
        self.stiffness_range = stiffness_range
        self.stiffness_log = stiffness_log
        self.wn = wn
        self.bearing = bearing
        self.intersection_points = intersection_points

    def plot(
        self,
        fig=None,
        stiffness_units="N/m",
        frequency_units="rad/s",
        **kwargs,
    ):
        """Plot undamped critical speed map.

        This method will plot the undamped critical speed map for a given range
        of stiffness values. If the range is not provided, the bearing
        stiffness at rated speed will be used to create a range.

        Parameters
        ----------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        stiffness_units : str, optional
            Units for the x axis.
            Default is N/m.
        frequency_units : str, optional
            Units for th y axis.
            Default is rad/s
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """

        stiffness_log = self.stiffness_log
        rotor_wn = self.wn
        bearing0 = self.bearing
        intersection_points = self.intersection_points

        if fig is None:
            fig = go.Figure()

        # convert to desired units
        stiffness_log = Q_(stiffness_log, "N/m").to(stiffness_units).m
        rotor_wn = Q_(rotor_wn, "rad/s").to(frequency_units).m
        intersection_points["x"] = (
            Q_(intersection_points["x"], "N/m").to(stiffness_units).m
        )
        intersection_points["y"] = (
            Q_(intersection_points["y"], "rad/s").to(frequency_units).m
        )
        bearing_kxx_stiffness = (
            Q_(bearing0.kxx.interpolated(bearing0.frequency), "N/m")
            .to(stiffness_units)
            .m
        )
        bearing_kyy_stiffness = (
            Q_(bearing0.kyy.interpolated(bearing0.frequency), "N/m")
            .to(stiffness_units)
            .m
        )
        bearing_frequency = Q_(bearing0.frequency, "rad/s").to(frequency_units).m

        for j in range(rotor_wn.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=stiffness_log,
                    y=rotor_wn[j],
                    mode="lines",
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=intersection_points["x"],
                y=intersection_points["y"],
                mode="markers",
                marker=dict(symbol="circle-open-dot", color="red", size=8),
                hovertemplate=f"Stiffness ({stiffness_units}): %{{x:.2e}}<br>Frequency ({frequency_units}): %{{y:.2f}}",
                showlegend=False,
                name="",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=bearing_kxx_stiffness,
                y=bearing_frequency,
                mode="lines",
                line=dict(dash="dashdot"),
                hoverinfo="none",
                name="Kxx",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bearing_kyy_stiffness,
                y=bearing_frequency,
                mode="lines",
                line=dict(dash="dashdot"),
                hoverinfo="none",
                name="Kyy",
            )
        )

        fig.update_xaxes(
            title_text=f"Bearing Stiffness ({stiffness_units})",
            type="log",
            exponentformat="power",
        )
        fig.update_yaxes(
            title_text=f"Critical Speed ({frequency_units})",
            type="log",
            exponentformat="power",
        )
        fig.update_layout(title=dict(text="Undamped Critical Speed Map"), **kwargs)

        return fig


class Level1Results(Results):
    """Class used to store results and provide plots for Level 1 Stability Analysis.

    Parameters
    ----------
    stiffness_range : array
        Stiffness array used in the calculation.
    log_dec : array
        Calculated log dec array for each cross coupling.
    """

    def __init__(self, stiffness_range, log_dec):
        self.stiffness_range = stiffness_range
        self.log_dec = log_dec

    def plot(self, fig=None, **kwargs):
        """Plot level 1 stability analysis.

        This method will plot the stability 1 analysis for a
        given stiffness range.

        Parameters
        ----------
        fig : Plotly graph_objects.Figure
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

        stiffness = self.stiffness_range
        log_dec = self.log_dec

        fig.add_trace(
            go.Scatter(
                x=stiffness,
                y=log_dec,
                mode="lines",
                line=dict(width=3),
                showlegend=False,
                hovertemplate=("Stiffness: %{x:.2e}<br>" + "Log Dec: %{y:.2f}"),
            )
        )

        fig.update_xaxes(
            title_text="Applied Cross Coupled Stiffness", exponentformat="power"
        )
        fig.update_yaxes(title_text="Log Dec", exponentformat="power")
        fig.update_layout(title=dict(text="Level 1 stability analysis"), **kwargs)
