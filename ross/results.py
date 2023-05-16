"""ROSS plotting module.

This module returns graphs for each type of analyses in rotor_assembly.py.
"""
import copy
import inspect
from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import toml
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import linalg as la

from ross.plotly_theme import tableau_colors
from ross.units import Q_, check_units
from ross.utils import intersection

__all__ = [
    "Orbit",
    "Shape",
    "CriticalSpeedResults",
    "ModalResults",
    "CampbellResults",
    "FrequencyResponseResults",
    "ForcedResponseResults",
    "StaticResults",
    "SummaryResults",
    "ConvergenceResults",
    "TimeResponseResults",
    "UCSResults",
    "Level1Results",
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
                try:
                    data[key] = np.array(value)
                    if data[key].dtype in str_type:
                        data[key] = np.array(value).astype(np.complex128)
                except:
                    data[key] = value

        return cls.read_toml_data(data)


class Orbit(Results):
    r"""Class used to construct orbits for a node in a mode or deflected shape.

    The matrix H contains information about the whirl direction,
    the orbit minor and major axis and the orbit inclination.
    The matrix is calculated by :math:`H = T.T^T` where the
    matrix T is constructed using the eigenvector corresponding
    to the natural frequency of interest:

    .. math::

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
    node : int
        Orbit node in the rotor.
    ru_e : complex
        Element in the vector corresponding to the x direction.
    rv_e : complex
        Element in the vector corresponding to the y direction.
    """

    def __init__(self, *, node, node_pos, ru_e, rv_e):
        self.node = node
        self.node_pos = node_pos
        self.ru_e = ru_e
        self.rv_e = rv_e

        # data for plotting
        num_points = 360
        c = np.linspace(0, 2 * np.pi, num_points)
        circle = np.exp(1j * c)

        self.x_circle = np.real(ru_e * circle)
        self.y_circle = np.real(rv_e * circle)
        angle = np.arctan2(self.y_circle, self.x_circle)
        angle[angle < 0] = angle[angle < 0] + 2 * np.pi
        self.angle = angle

        # find major axis index looking at the first half circle
        self.major_index = np.argmax(
            np.sqrt(self.x_circle[:180] ** 2 + self.y_circle[:180] ** 2)
        )
        self.major_x = self.x_circle[self.major_index]
        self.major_y = self.y_circle[self.major_index]
        self.major_angle = self.angle[self.major_index]
        self.minor_angle = self.major_angle + np.pi / 2

        # calculate T matrix
        ru = np.absolute(ru_e)
        rv = np.absolute(rv_e)

        nu = np.angle(ru_e)
        nv = np.angle(rv_e)
        self.nu = nu
        self.nv = nv
        # fmt: off
        T = np.array([[ru * np.cos(nu), -ru * np.sin(nu)],
                      [rv * np.cos(nv), -rv * np.sin(nv)]])
        # fmt: on
        H = T @ T.T

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

        self.minor_axis = np.real(minor)
        self.major_axis = np.real(major)
        self.kappa = np.real(kappa)
        self.whirl = "Forward" if self.kappa > 0 else "Backward"
        self.color = (
            tableau_colors["blue"] if self.whirl == "Forward" else tableau_colors["red"]
        )

    @check_units
    def calculate_amplitude(self, angle):
        """Calculates the amplitude for a given angle of the orbit.

        Parameters
        ----------
        angle : float, str, pint.Quantity

        Returns
        -------
        amplitude, phase : tuple
            Tuple with (amplitude, phase) value.
            The amplitude units are the same as the ru_e and rv_e used to create the orbit.
        """
        # find closest angle index
        if angle == "major":
            return self.major_axis, self.major_angle
        elif angle == "minor":
            return self.minor_axis, self.minor_angle

        idx = (np.abs(self.angle - angle)).argmin()
        amplitude = np.sqrt(self.x_circle[idx] ** 2 + self.y_circle[idx] ** 2)
        phase = self.angle[0] + angle
        if phase > 2 * np.pi:
            phase -= 2 * np.pi

        return amplitude, phase

    def plot_orbit(self, fig=None):
        if fig is None:
            fig = go.Figure()

        xc = self.x_circle
        yc = self.y_circle

        fig.add_trace(
            go.Scatter(
                x=xc[:-10],
                y=yc[:-10],
                mode="lines",
                line=dict(color=self.color),
                name=f"node {self.node}<br>{self.whirl}",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[xc[0]],
                y=[yc[0]],
                mode="markers",
                marker=dict(color=self.color),
                name="node {}".format(self.node),
                showlegend=False,
            )
        )

        return fig


class Shape(Results):
    """Class used to construct a mode or a deflected shape from a eigen or response vector.

    Parameters
    ----------
    vector : np.array
        Complex array with the eigenvector or response vector from a forced response analysis.
        Array shape should be equal to the number of degrees of freedom of
        the complete rotor (ndof * number_of_nodes).
    nodes : list
        List of nodes number.
    nodes_pos : list
        List of nodes positions.
    shaft_elements_length : list
        List with Rotor shaft elements lengths.
    normalize : bool, optional
        If True the vector is normalized.
        Default is False.
    """

    def __init__(
        self, vector, nodes, nodes_pos, shaft_elements_length, normalize=False
    ):
        self.vector = vector
        self.nodes = nodes
        self.nodes_pos = nodes_pos
        self.shaft_elements_length = shaft_elements_length
        self.normalize = normalize
        evec = np.copy(vector)

        if self.normalize:
            modex = evec[0::4]
            modey = evec[1::4]
            xmax, ixmax = max(abs(modex)), np.argmax(abs(modex))
            ymax, iymax = max(abs(modey)), np.argmax(abs(modey))

            if ymax > xmax:
                evec /= modey[iymax]
            else:
                evec /= modex[ixmax]

        self._evec = evec
        self.orbits = None
        self.whirl = None
        self.color = None
        self.xn = None
        self.yn = None
        self.zn = None
        self.major_axis = None
        self._calculate()

    def _calculate_orbits(self):
        orbits = []
        whirl = []
        for node, node_pos in zip(self.nodes, self.nodes_pos):
            ru_e, rv_e = self._evec[4 * node : 4 * node + 2]
            orbit = Orbit(node=node, node_pos=node_pos, ru_e=ru_e, rv_e=rv_e)
            orbits.append(orbit)
            whirl.append(orbit.whirl)

        self.orbits = orbits
        # check shape whirl
        if all(w == "Forward" for w in whirl):
            self.whirl = "Forward"
            self.color = tableau_colors["blue"]
        elif all(w == "Backward" for w in whirl):
            self.whirl = "Backward"
            self.color = tableau_colors["red"]
        else:
            self.whirl = "Mixed"
            self.color = tableau_colors["gray"]

    def _calculate(self):
        evec = self._evec
        nodes = self.nodes
        shaft_elements_length = self.shaft_elements_length
        nodes_pos = self.nodes_pos

        # calculate each orbit
        self._calculate_orbits()

        # plot lines
        nn = 5  # number of points in each line between nodes
        zeta = np.linspace(0, 1, nn)
        onn = np.ones_like(zeta)

        zeta = zeta.reshape(nn, 1)
        onn = onn.reshape(nn, 1)

        xn = np.zeros(nn * (len(nodes) - 1))
        yn = np.zeros(nn * (len(nodes) - 1))
        xn_complex = np.zeros(nn * (len(nodes) - 1), dtype=np.complex128)
        yn_complex = np.zeros(nn * (len(nodes) - 1), dtype=np.complex128)
        zn = np.zeros(nn * (len(nodes) - 1))
        major = np.zeros(nn * (len(nodes) - 1))
        major_x = np.zeros(nn * (len(nodes) - 1))
        major_y = np.zeros(nn * (len(nodes) - 1))
        major_angle = np.zeros(nn * (len(nodes) - 1))

        N1 = onn - 3 * zeta**2 + 2 * zeta**3
        N2 = zeta - 2 * zeta**2 + zeta**3
        N3 = 3 * zeta**2 - 2 * zeta**3
        N4 = -(zeta**2) + zeta**3

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

            # major axes calculation
            xn_complex[pos0:pos1] = Nx @ evec[xx]
            yn_complex[pos0:pos1] = Ny @ evec[yy]
            for i in range(pos0, pos1):
                orb = Orbit(node=0, node_pos=0, ru_e=xn_complex[i], rv_e=yn_complex[i])
                major[i] = orb.major_axis
                major_x[i] = orb.major_x
                major_y[i] = orb.major_y
                major_angle[i] = orb.major_angle

        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.major_axis = major
        self.major_x = major_x
        self.major_y = major_y
        self.major_angle = major_angle

    def plot_orbit(self, nodes, fig=None):
        """Plot orbits.

        Parameters
        ----------
        nodes : list
            List with nodes for which the orbits will be plotted.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        # only perform calculation if necessary
        if fig is None:
            fig = go.Figure()

        selected_orbits = [orbit for orbit in self.orbits if orbit.node in nodes]

        for orbit in selected_orbits:
            fig = orbit.plot_orbit(fig=fig)

        return fig

    def plot_2d(
        self, orientation="major", length_units="m", phase_units="rad", fig=None
    ):
        """Rotor shape 2d plot.

        Parameters
        ----------
        orientation : str, optional
            Orientation can be 'major', 'x' or 'y'.
            Default is 'major' to display the major axis.
        length_units : str, optional
            length units.
            Default is 'm'.
        phase_units : str, optional
            Phase units.
            Default is "rad"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        xn = self.major_x.copy()
        yn = self.major_y.copy()
        zn = self.zn.copy()
        nodes_pos = Q_(self.nodes_pos, "m").to(length_units).m

        if fig is None:
            fig = go.Figure()

        if orientation == "major":
            values = self.major_axis.copy()
        elif orientation == "x":
            values = xn
        elif orientation == "y":
            values = yn
        else:
            raise ValueError(f"Invalid orientation {orientation}.")

        fig.add_trace(
            go.Scatter(
                x=Q_(zn, "m").to(length_units).m,
                y=values,
                mode="lines",
                line=dict(color=self.color),
                name=f"{orientation}",
                showlegend=False,
                customdata=Q_(self.major_angle, "rad").to(phase_units).m,
                hovertemplate=(
                    f"Displacement: %{{y:.2f}}<br>"
                    + f"Angle {phase_units}: %{{customdata:.2f}}"
                ),
            ),
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

        fig.update_xaxes(title_text=f"Rotor Length ({length_units})")

        return fig

    def plot_3d(
        self,
        mode=None,
        orientation="major",
        title=None,
        length_units="m",
        phase_units="rad",
        fig=None,
        **kwargs,
    ):
        if fig is None:
            fig = go.Figure()

        xn = self.xn.copy()
        yn = self.yn.copy()
        zn = self.zn.copy()

        # plot orbits
        first_orbit = True
        for orbit in self.orbits:
            zc_pos = (
                Q_(np.repeat(orbit.node_pos, len(orbit.x_circle)), "m")
                .to(length_units)
                .m
            )
            fig.add_trace(
                go.Scatter3d(
                    x=zc_pos[:-10],
                    y=orbit.x_circle[:-10],
                    z=orbit.y_circle[:-10],
                    mode="lines",
                    line=dict(color=orbit.color),
                    name="node {}".format(orbit.node),
                    showlegend=False,
                    hovertemplate=(
                        "Nodal Position: %{x:.2f}<br>"
                        + "X - Displacement: %{y:.2f}<br>"
                        + "Y - Displacement: %{z:.2f}"
                    ),
                )
            )
            # add orbit start
            fig.add_trace(
                go.Scatter3d(
                    x=[zc_pos[0]],
                    y=[orbit.x_circle[0]],
                    z=[orbit.y_circle[0]],
                    mode="markers",
                    marker=dict(color=orbit.color),
                    name="node {}".format(orbit.node),
                    showlegend=False,
                    hovertemplate=(
                        "Nodal Position: %{x:.2f}<br>"
                        + "X - Displacement: %{y:.2f}<br>"
                        + "Y - Displacement: %{z:.2f}"
                    ),
                )
            )
            # add orbit major axis marker
            fig.add_trace(
                go.Scatter3d(
                    x=[zc_pos[0]],
                    y=[orbit.major_x],
                    z=[orbit.major_y],
                    mode="markers",
                    marker=dict(color="black", symbol="cross", size=4, line_width=2),
                    name="Major axis",
                    showlegend=True if first_orbit else False,
                    legendgroup="major_axis",
                    customdata=np.array(
                        [
                            orbit.major_axis,
                            Q_(orbit.major_angle, "rad").to(phase_units).m,
                        ]
                    ).reshape(1, 2),
                    hovertemplate=(
                        "Nodal Position: %{x:.2f}<br>"
                        + "Major axis: %{customdata[0]:.2f}<br>"
                        + "Angle: %{customdata[1]:.2f}"
                    ),
                )
            )
            first_orbit = False

        # plot line connecting orbits starting points
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

        # plot major axis line
        fig.add_trace(
            go.Scatter3d(
                x=Q_(zn, "m").to(length_units).m,
                y=self.major_x,
                z=self.major_y,
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                hoverinfo="none",
                legendgroup="major_axis",
                showlegend=False,
            )
        )

        fig.update_layout(
            legend=dict(itemsizing="constant"),
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=2.5, y=1, z=1),
            ),
        )

        return fig


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
        self.shapes = []
        kappa_modes = []
        for mode in range(len(self.wn)):
            self.shapes.append(
                Shape(
                    vector=self.modes[:, mode],
                    nodes=self.nodes,
                    nodes_pos=self.nodes_pos,
                    shaft_elements_length=self.shaft_elements_length,
                    normalize=True,
                )
            )
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

        k = {
            "Frequency": nat_freq,
            "Minor axis": self.shapes[w].orbits[node].minor_axis,
            "Major axis": self.shapes[w].orbits[node].major_axis,
            "kappa": self.shapes[w].orbits[node].kappa,
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
        kappa_mode = [orb.kappa for orb in self.shapes[w].orbits]
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

    def plot_mode_3d(
        self,
        mode=None,
        frequency_type="wd",
        title=None,
        length_units="m",
        phase_units="rad",
        frequency_units="rad/s",
        damping_parameter="log_dec",
        fig=None,
        **kwargs,
    ):
        """Plot (3D view) the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
            Default is None
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
        phase_units : str, optional
            Phase units.
            Default is "rad"
        frequency_units : str, optional
            Frequency units that will be used in the plot title.
            Default is rad/s.
        damping_parameter : str, optional
            Define which value to show for damping. We can use "log_dec" or "damping_ratio".
            Default is "log_dec".
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

        damping_name = "Log. Dec."
        damping_value = self.log_dec[mode]
        if damping_parameter == "damping_ratio":
            damping_name = "Damping ratio"
            damping_value = self.damping_ratio[mode]

        frequency = {
            "wd": f"ω<sub>d</sub> = {Q_(self.wd[mode], 'rad/s').to(frequency_units).m:.2f}",
            "wn": f"ω<sub>n</sub> = {Q_(self.wn[mode], 'rad/s').to(frequency_units).m:.2f}",
            "speed": f"Speed = {Q_(self.speed, 'rad/s').to(frequency_units).m:.2f}",
        }

        shape = self.shapes[mode]
        fig = shape.plot_3d(length_units=length_units, phase_units=phase_units, fig=fig)

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
                aspectmode="manual",
                aspectratio=dict(x=2.5, y=1, z=1),
            ),
            title=dict(
                text=(
                    f"{title}<br>"
                    f"Mode {mode} | "
                    f"{frequency['speed']} {frequency_units} | "
                    f"whirl: {self.whirl_direction()[mode]} | "
                    f"{frequency[frequency_type]} {frequency_units} | "
                    f"{damping_name} = {damping_value:.2f}"
                ),
                x=0.5,
                xanchor="center",
            ),
            **kwargs,
        )

        return fig

    def plot_mode_2d(
        self,
        mode=None,
        fig=None,
        orientation="major",
        frequency_type="wd",
        title=None,
        length_units="m",
        frequency_units="rad/s",
        damping_parameter="log_dec",
        **kwargs,
    ):
        """Plot (2D view) the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        orientation : str, optional
            Orientation can be 'major', 'x' or 'y'.
            Default is 'major' to display the major axis.
        frequency_type : str, optional
            "wd" calculates the damped natural frequencies.
            "wn" calculates the undamped natural frequencies.
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
        damping_parameter : str, optional
            Define which value to show for damping. We can use "log_dec" or "damping_ratio".
            Default is "log_dec".
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        damping_name = "Log. Dec."
        damping_value = self.log_dec[mode]
        if damping_parameter == "damping_ratio":
            damping_name = "Damping ratio"
            damping_value = self.damping_ratio[mode]

        if fig is None:
            fig = go.Figure()

        frequency = {
            "wd": f"ω<sub>d</sub> = {Q_(self.wd[mode], 'rad/s').to(frequency_units).m:.2f}",
            "wn": f"ω<sub>n</sub> = {Q_(self.wn[mode], 'rad/s').to(frequency_units).m:.2f}",
            "speed": f"Speed = {Q_(self.speed, 'rad/s').to(frequency_units).m:.2f}",
        }

        shape = self.shapes[mode]
        fig = shape.plot_2d(fig=fig, orientation=orientation)

        if title is None:
            title = ""

        fig.update_xaxes(title_text=f"Rotor Length ({length_units})")
        fig.update_yaxes(title_text="Relative Displacement")
        fig.update_layout(
            title=dict(
                text=(
                    f"{title}<br>"
                    f"Mode {mode} | "
                    f"{frequency['speed']} {frequency_units} | "
                    f"whirl: {self.whirl_direction()[mode]} | "
                    f"{frequency[frequency_type]} {frequency_units} | "
                    f"{damping_name} = {damping_value:.2f}"
                ),
                x=0.5,
                xanchor="center",
            ),
            **kwargs,
        )

        return fig

    def plot_orbit(
        self,
        mode=None,
        nodes=None,
        fig=None,
        frequency_type="wd",
        title=None,
        frequency_units="rad/s",
        **kwargs,
    ):
        """Plot (2D view) the mode shapes.

        Parameters
        ----------
        mode : int
            The n'th vibration mode
            Default is None
        nodes : int, list(ints)
            Int or list of ints with the nodes selected to be plotted.
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

        # case where an int is given
        if not isinstance(nodes, Iterable):
            nodes = [nodes]

        shape = self.shapes[mode]
        fig = shape.plot_orbit(nodes, fig=fig)

        fig.update_layout(
            autosize=False,
            width=500,
            height=500,
            xaxis_range=[-1, 1],
            yaxis_range=[-1, 1],
            title={
                "text": f"Mode {mode} - Nodes {nodes}",
                "x": 0.5,
                "xanchor": "center",
            },
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

    def __init__(self, speed_range, wd, log_dec, damping_ratio, whirl_values):
        self.speed_range = speed_range
        self.wd = wd
        self.log_dec = log_dec
        self.damping_ratio = damping_ratio
        self.whirl_values = whirl_values

    @check_units
    def plot(
        self,
        harmonics=[1],
        frequency_units="rad/s",
        damping_parameter="log_dec",
        frequency_range=None,
        damping_range=None,
        fig=None,
        **kwargs,
    ):
        """Create Campbell Diagram figure using Plotly.

        Parameters
        ----------
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        damping_parameter : str, optional
            Define which value to show for damping. We can use "log_dec" or "damping_ratio".
            Default is "log_dec".
        frequency_range : tuple, pint.Quantity(tuple), optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
            Default is None (no filter).
        damping_range : tuple, optional
            Tuple with (min, max) values for the damping parameter that will be plotted.
            Damping values that are not within the range are filtered out and are not plotted.
            Default is None (no filter).
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

        Examples
        --------
        >>> import ross as rs
        >>> import numpy as np
        >>> Q_ = rs.Q_
        >>> rotor = rs.rotor_example()
        >>> speed = np.linspace(0, 400, 101)
        >>> camp = rotor.run_campbell(speed)
        >>> fig = camp.plot(
        ...     harmonics=[1, 2],
        ...     damping_parameter="damping_ratio",
        ...     frequency_range=Q_((2000, 10000), "RPM"),
        ...     damping_range=(-0.1, 100),
        ...     frequency_units="RPM",
        ... )
        """
        if damping_parameter == "log_dec":
            damping_values = self.log_dec
            title_text = "<b>Log Dec</b>"
        elif damping_parameter == "damping_ratio":
            damping_values = self.damping_ratio
            title_text = "<b>Damping Ratio</b>"
        else:
            raise ValueError(
                f"damping_parameter can be 'log_dec' or 'damping_ratio'. {damping_parameter} is not valid"
            )

        wd = self.wd
        num_frequencies = wd.shape[1]

        whirl = self.whirl_values
        speed_range = self.speed_range

        if fig is None:
            fig = go.Figure()

        default_values = dict(
            coloraxis_cmin=0.0,
            coloraxis_cmax=1.0,
            coloraxis_colorscale="rdbu",
            coloraxis_colorbar=dict(title=dict(text=title_text, side="right")),
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

        # filter frequency range
        if frequency_range is not None:
            crit_x_filtered = []
            crit_y_filtered = []
            for x, y in zip(crit_x, crit_y):
                if frequency_range[0] < y < frequency_range[1]:
                    crit_x_filtered.append(x)
                    crit_y_filtered.append(y)
            crit_x = crit_x_filtered
            crit_y = crit_y_filtered

        if len(crit_x) and len(crit_y):
            fig.add_trace(
                go.Scatter(
                    x=Q_(crit_x, "rad/s").to(frequency_units).m,
                    y=Q_(crit_y, "rad/s").to(frequency_units).m,
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
            scatter_marker, [0.0, 0.5, 1.0], ["Forward", "Mixed", "Backward"]
        ):
            for i in range(num_frequencies):
                w_i = wd[:, i]
                whirl_i = whirl[:, i]
                damping_values_i = damping_values[:, i]

                whirl_mask = whirl_i == whirl_dir
                mask = whirl_mask
                if frequency_range is not None:
                    frequency_mask = (w_i > frequency_range[0]) & (
                        w_i < frequency_range[1]
                    )
                    mask = mask & frequency_mask
                if damping_range is not None:
                    damping_mask = (damping_values_i > damping_range[0]) & (
                        damping_values_i < damping_range[1]
                    )
                    mask = mask & damping_mask

                if any(check for check in mask):
                    fig.add_trace(
                        go.Scatter(
                            x=Q_(speed_range[mask], "rad/s").to(frequency_units).m,
                            y=Q_(w_i[mask], "rad/s").to(frequency_units).m,
                            marker=dict(
                                symbol=mark,
                                color=damping_values_i[mask],
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
                    x=Q_(speed_range, "rad/s").to(frequency_units).m,
                    y=h * Q_(speed_range, "rad/s").to(frequency_units).m,
                    mode="lines",
                    line=dict(dash="dashdot", color=list(tableau_colors)[j]),
                    name="{}x speed".format(h),
                    hoverinfo="none",
                )
            )
        # turn legend glyphs black
        scatter_marker = ["triangle-up", "circle", "triangle-down"]
        legends = ["Forward", "Mixed", "Backward"]
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
            range=[
                np.min(Q_(speed_range, "rad/s").to(frequency_units).m),
                np.max(Q_(speed_range, "rad/s").to(frequency_units).m),
            ],
            exponentformat="none",
        )
        fig.update_yaxes(
            title_text=f"Natural Frequencies ({frequency_units})",
            range=[0, 1.1 * np.max(Q_(wd, "rad/s").to(frequency_units).m)],
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
        It is possible to plot the magnitude with different units, depending on the unit entered in 'amplitude_units'. If '[length]/[force]', it displays the displacement unit (m); If '[speed]/[force]', it displays the velocity unit (m/s); If '[acceleration]/[force]', it displays the acceleration  unit (m/s**2).

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
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the magnitude with units (m/N);

            '[speed]' - Displays the magnitude with units (m/s/N);

            '[acceleration]' - Displays the magnitude with units (m/s**2/N).

            Default is "m/N" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm/N pkpk')
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
        y_label = "Magnitude"
        if dummy_var.check("[length]/[force]"):
            mag = np.abs(self.freq_resp)
            mag = Q_(mag, "m/N").to(amplitude_units).m
        elif dummy_var.check("[speed]/[force]"):
            mag = np.abs(self.velc_resp)
            mag = Q_(mag, "m/s/N").to(amplitude_units).m
        elif dummy_var.check("[acceleration]/[force]"):
            mag = np.abs(self.accl_resp)
            mag = Q_(mag, "m/s**2/N").to(amplitude_units).m
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
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the magnitude with units (m/N);

            '[speed]' - Displays the magnitude with units (m/s/N);

            '[acceleration]' - Displays the magnitude with units (m/s**2/N).

            Default is "m/N" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm/N pkpk')
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
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m/N" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm/N pkpk')
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

        Amplitude magnitude unit can be displacement, velocity or accelaration responses,
        depending on the unit entered in 'amplitude_units'. If '[length]/[force]', it displays the displacement unit (m); If '[speed]/[force]', it displays the velocity unit (m/s); If '[acceleration]/[force]', it displays the acceleration  unit (m/s**2).

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
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the magnitude with units (m/N);

            '[speed]' - Displays the magnitude with units (m/s/N);

            '[acceleration]' - Displays the magnitude with units (m/s**2/N).

            Default is "m/N" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm/N pkpk')
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

    def data_magnitude(
        self,
        probe,
        probe_units="rad",
        frequency_units="rad/s",
        amplitude_units="m",
    ):
        """Return the forced response (magnitude) in DataFrame format.

        Parameters
        ----------
        probe : list
            List with rs.Probe objects.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the frequency range.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')

        Returns
        -------
        df : pd.DataFrame
            DataFrame storing magnitude data arrays. The columns are set based on the
            probe's tag.
        """
        frequency_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            base_unit = self.default_units[unit_type][0]
            response = getattr(self, self.default_units[unit_type][1])
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        data = {}
        data["frequency"] = frequency_range

        for i, p in enumerate(probe):
            amplitude = []
            for speed_idx in range(len(self.speed_range)):
                # first try to get the angle from the probe object
                try:
                    angle = p.angle
                    node = p.node
                # if it is a tuple, warn the user that the use of tuples is deprecated
                except AttributeError:
                    try:
                        angle = Q_(p[1], probe_units).to("rad").m
                        warn(
                            "The use of tuples in the probe argument is deprecated. Use the Probe class instead.",
                            DeprecationWarning,
                        )
                        node = p[0]
                    except TypeError:
                        angle = p[1]
                        node = p[0]

                ru_e, rv_e = response[:, speed_idx][
                    self.rotor.number_dof * node : self.rotor.number_dof * node + 2
                ]
                orbit = Orbit(
                    node=node, node_pos=self.rotor.nodes_pos[node], ru_e=ru_e, rv_e=rv_e
                )
                amp, phase = orbit.calculate_amplitude(angle=angle)
                amplitude.append(amp)

            try:
                probe_tag = p.tag
            except AttributeError:
                try:
                    probe_tag = p[2]
                except IndexError:
                    probe_tag = f"Probe {i+1} - Node {p[0]}"

            data[probe_tag] = Q_(amplitude, base_unit).to(amplitude_units).m

        df = pd.DataFrame(data)

        return df

    def data_phase(
        self,
        probe,
        probe_units="rad",
        frequency_units="rad/s",
        amplitude_units="m",
        phase_units="rad",
    ):
        """Return the forced response (phase) in DataFrame format.

        Parameters
        ----------
        probe : list
            List with rs.Probe objects.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"

        Returns
        -------
        df : pd.DataFrame
            DataFrame storing phase data arrays. They columns are set based on the
            probe's tag.
        """
        frequency_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            base_unit = self.default_units[unit_type][0]
            response = getattr(self, self.default_units[unit_type][1])
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        data = {}
        data["frequency"] = frequency_range

        for i, p in enumerate(probe):
            phase_values = []
            for speed_idx in range(len(self.speed_range)):
                # first try to get the angle from the probe object
                try:
                    angle = p.angle
                    node = p.node
                # if it is a tuple, warn the user that the use of tuples is deprecated
                except AttributeError:
                    try:
                        angle = Q_(p[1], probe_units).to("rad").m
                        warn(
                            "The use of tuples in the probe argument is deprecated. Use the Probe class instead.",
                            DeprecationWarning,
                        )
                        node = p[0]
                    except TypeError:
                        angle = p[1]
                        node = p[0]

                ru_e, rv_e = response[:, speed_idx][
                    self.rotor.number_dof * node : self.rotor.number_dof * node + 2
                ]
                orbit = Orbit(
                    node=node, node_pos=self.rotor.nodes_pos[node], ru_e=ru_e, rv_e=rv_e
                )
                amp, phase = orbit.calculate_amplitude(angle=angle)
                phase_values.append(phase)

            try:
                probe_tag = p.tag
            except AttributeError:
                try:
                    probe_tag = p[2]
                except IndexError:
                    probe_tag = f"Probe {i+1} - Node {p[0]}"

            data[probe_tag] = Q_(phase_values, "rad").to(phase_units).m

        df = pd.DataFrame(data)

        return df

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
        probe : list
            List with rs.Probe objects.
        probe_units : str, optional
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')
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
        df = self.data_magnitude(probe, probe_units, frequency_units, amplitude_units)

        if fig is None:
            fig = go.Figure()

        for i, column in enumerate(df.columns[1:]):
            fig.add_trace(
                go.Scatter(
                    x=df["frequency"],
                    y=df[column],
                    mode="lines",
                    line=dict(color=list(tableau_colors)[i]),
                    name=column,
                    legendgroup=column,
                    showlegend=True,
                    hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(df["frequency"]), np.max(df["frequency"])],
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
        probe : list
            List with rs.Probe objects.
        probe_units : str, optional
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')
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
        df = self.data_phase(
            probe, probe_units, frequency_units, amplitude_units, phase_units
        )

        if fig is None:
            fig = go.Figure()

        for i, column in enumerate(df.columns[1:]):
            fig.add_trace(
                go.Scatter(
                    x=df["frequency"],
                    y=df[column],
                    mode="lines",
                    line=dict(color=list(tableau_colors)[i]),
                    name=column,
                    legendgroup=column,
                    showlegend=True,
                    hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Phase ({phase_units}): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(df["frequency"]), np.max(df["frequency"])],
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
        probe : list
            List with rs.Probe objects.
        probe_units : str, optional
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')
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
        df_m = self.data_magnitude(probe, probe_units, frequency_units, amplitude_units)
        df_p = self.data_phase(
            probe, probe_units, frequency_units, amplitude_units, phase_units
        )

        if fig is None:
            fig = go.Figure()

        if phase_units in ["rad", "radian", "radians"]:
            polar_theta_unit = "radians"
        elif phase_units in ["degree", "degrees", "deg"]:
            polar_theta_unit = "degrees"

        for i, column in enumerate(df_m.columns[1:]):
            fig.add_trace(
                go.Scatterpolar(
                    r=df_m[column],
                    theta=df_p[column],
                    customdata=df_m["frequency"],
                    thetaunit=polar_theta_unit,
                    mode="lines+markers",
                    marker=dict(color=list(tableau_colors)[i]),
                    line=dict(color=list(tableau_colors)[i]),
                    name=column,
                    legendgroup=column,
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
        probe : list
            List with rs.Probe objects.
        probe_units : str, optional
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')
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
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')

        Returns
        -------
        major_axis_vector : np.ndarray
            major_axis_vector[0, :] = forward vector
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
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')

        Returns
        -------
        major_axis_vector : np.ndarray
            major_axis_vector[0, :] = forward vector
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

    @check_units
    def plot_deflected_shape_2d(
        self,
        speed,
        amplitude_units="m",
        phase_units="rad",
        rotor_length_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot the 2D deflected shape diagram.

        Parameters
        ----------
        speed : float, pint.Quantity
            The rotor rotation speed. Must be an element from the speed_range argument
            passed to the class.
            Default unit is rad/s.
        amplitude_units : str, optional
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')
        phase_units : str, optional
            Phase units.
            Default is "rad"
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

        # get response with the right displacement units and speed
        response = self.__dict__[self.default_units[unit_type][1]]
        idx = np.where(np.isclose(self.speed_range, speed, atol=1e-6))[0][0]
        response = Q_(response[:, idx], base_unit).to(amplitude_units).m

        shape = Shape(
            vector=response,
            nodes=self.rotor.nodes,
            nodes_pos=self.rotor.nodes_pos,
            shaft_elements_length=self.rotor.shaft_elements_length,
        )

        if fig is None:
            fig = go.Figure()
        fig = shape.plot_2d(
            phase_units=phase_units, length_units=rotor_length_units, fig=fig
        )

        # customize hovertemplate
        fig.update_traces(
            selector=dict(name="major"),
            hovertemplate=(
                f"Amplitude ({amplitude_units}): %{{y:.2e}}<br>"
                + f"Phase ({phase_units}): %{{customdata:.2f}}<br>"
                + f"Nodal Position ({rotor_length_units}): %{{x:.2f}}"
            ),
        )
        fig.update_yaxes(
            title_text=f"Major Axis Amplitude ({amplitude_units})",
        )

        return fig

    def plot_deflected_shape_3d(
        self,
        speed,
        amplitude_units="m",
        phase_units="rad",
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
        amplitude_units : str, optional
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm pkpk')
        phase_units : str, optional
            Phase units.
            Default is "rad"
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

        if not any(np.isclose(self.speed_range, speed, atol=1e-6)):
            raise ValueError("No data available for this speed value.")

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            base_unit = self.default_units[unit_type][0]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        # get response with the right displacement units and speed
        response = self.__dict__[self.default_units[unit_type][1]]
        idx = np.where(np.isclose(self.speed_range, speed, atol=1e-6))[0][0]
        response = Q_(response[:, idx], base_unit).to(amplitude_units).m
        unbalance = self.unbalance

        shape = Shape(
            vector=response,
            nodes=self.rotor.nodes,
            nodes_pos=self.rotor.nodes_pos,
            shaft_elements_length=self.rotor.shaft_elements_length,
        )

        if fig is None:
            fig = go.Figure()

        fig = shape.plot_3d(
            phase_units=phase_units, length_units=rotor_length_units, fig=fig
        )

        # plot unbalance markers
        for i, n, amplitude, phase in zip(
            range(unbalance.shape[1]), unbalance[0], unbalance[1], unbalance[2]
        ):
            # scale unbalance marker to half the maximum major axis
            n = int(n)
            scaled_amplitude = np.max(shape.major_axis) / 2
            x = scaled_amplitude * np.cos(phase)
            y = scaled_amplitude * np.sin(phase)
            z_pos = Q_(shape.nodes_pos[n], "m").to(rotor_length_units).m

            fig.add_trace(
                go.Scatter3d(
                    x=[z_pos, z_pos],
                    y=[0, Q_(x, "m").to(amplitude_units).m],
                    z=[0, Q_(y, "m").to(amplitude_units).m],
                    mode="lines",
                    line=dict(color=tableau_colors["red"]),
                    legendgroup="Unbalance",
                    hoverinfo="none",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[z_pos],
                    y=[Q_(x, "m").to(amplitude_units).m],
                    z=[Q_(y, "m").to(amplitude_units).m],
                    mode="markers",
                    marker=dict(color=tableau_colors["red"], symbol="diamond"),
                    name="Unbalance",
                    legendgroup="Unbalance",
                    showlegend=True if i == 0 else False,
                    hovertemplate=(
                        f"Node: {n}<br>"
                        + f"Magnitude: {amplitude:.2e}<br>"
                        + f"Phase: {phase:.2f}"
                    ),
                )
            )

        # customize hovertemplate
        fig.update_traces(
            selector=dict(name="Major axis"),
            hovertemplate=(
                "Nodal Position: %{x:.2f}<br>"
                + "Major axis: %{customdata[0]:.2e}<br>"
                + "Angle: %{customdata[1]:.2f}"
            ),
        )

        plot_range = Q_(np.max(shape.major_axis) * 1.5, "m").to(amplitude_units).m
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=dict(text=f"Rotor Length ({rotor_length_units})"),
                    autorange="reversed",
                    nticks=5,
                ),
                yaxis=dict(
                    title=dict(text=f"Amplitude x ({amplitude_units})"),
                    range=[-plot_range, plot_range],
                    nticks=5,
                ),
                zaxis=dict(
                    title=dict(text=f"Amplitude y ({amplitude_units})"),
                    range=[-plot_range, plot_range],
                    nticks=5,
                ),
            ),
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
        Mr = np.sqrt(Mx**2 + My**2)

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
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the displacement;

            '[speed]' - Displays the velocity;

            '[acceleration]' - Displays the acceleration.

            Default is "m/N" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm/N pkpk')
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

        fig0 = self.plot_deflected_shape_2d(
            speed,
            frequency_units=frequency_units,
            amplitude_units=amplitude_units,
            rotor_length_units=rotor_length_units,
            **shape2d_kwargs,
        )
        fig1 = self.plot_deflected_shape_3d(
            speed,
            frequency_units=frequency_units,
            amplitude_units=amplitude_units,
            rotor_length_units=rotor_length_units,
            **shape3d_kwargs,
        )
        fig2 = self.plot_bending_moment(
            speed,
            frequency_units=frequency_units,
            moment_units=moment_units,
            rotor_length_units=rotor_length_units,
            **bm_kwargs,
        )

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
                aspectmode=fig1.layout.scene.aspectmode,
                aspectratio=fig1.layout.scene.aspectratio,
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

        shaft_end = self.nodes_pos[-1]
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

        fig.add_trace(
            go.Scatter(
                x=Q_(self.nodes_pos, "m").to(rotor_length_units).m,
                y=Q_(self.deformation, "m").to(deformation_units).m,
                mode="lines",
                line_shape="spline",
                line_smoothing=1.0,
                name=f"Shaft",
                showlegend=True,
                hovertemplate=(
                    f"Rotor Length ({rotor_length_units}): %{{x:.2f}}<br>Displacement ({deformation_units}): %{{y:.2e}}"
                ),
            )
        )

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
        col = cols = 1
        row = rows = 1
        if fig is None:
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=["Free-Body Diagram"],
            )

        y_start = 5.0
        nodes_pos = self.nodes_pos
        nodes = self.nodes

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
        text = "{:.2f}".format(Q_(self.w_shaft, "N").to(force_units).m)
        ini = nodes_pos[0]
        fin = nodes_pos[-1]
        arrows_list = np.arange(ini, 1.01 * fin, (fin - ini) / 5.0)
        for node in arrows_list:
            fig.add_annotation(
                x=Q_(node, "m").to(rotor_length_units).m,
                y=0,
                axref="x{}".format(1),
                ayref="y{}".format(1),
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
            xref="x{}".format(1),
            yref="y{}".format(1),
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
                    x=Q_(nodes_pos[nodes.index(node)], "m").to(rotor_length_units).m,
                    y=0,
                    axref="x{}".format(1),
                    ayref="y{}".format(1),
                    text=f"Fb = {text}{force_units}",
                    textangle=90,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=5,
                    arrowcolor="DarkSalmon",
                    ax=Q_(nodes_pos[nodes.index(node)], "m").to(rotor_length_units).m,
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
                        axref="x{}".format(1),
                        ayref="y{}".format(1),
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
            # j += 1

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

        shaft_end = Q_(self.nodes_pos[-1], "m").to(rotor_length_units).m

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

        Vx, Vx_axis = self.Vx, self.Vx_axis
        fig.add_trace(
            go.Scatter(
                x=Q_(Vx_axis, "m").to(rotor_length_units).m,
                y=Q_(Vx, "N").to(force_units).m,
                mode="lines",
                name=f"Shaft",
                legendgroup=f"Shaft",
                showlegend=True,
                hovertemplate=(
                    f"Rotor Length ({rotor_length_units}): %{{x:.2f}}<br>Shearing Force ({force_units}): %{{y:.2f}}"
                ),
            )
        )

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

        shaft_end = Q_(self.nodes_pos[-1], "m").to(rotor_length_units).m

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

        Bm, nodes_pos = self.Bm, self.Vx_axis
        fig.add_trace(
            go.Scatter(
                x=Q_(nodes_pos, "m").to(rotor_length_units).m,
                y=Q_(Bm, "N*m").to(moment_units).m,
                mode="lines",
                line_shape="spline",
                line_smoothing=1.0,
                name=f"Shaft",
                legendgroup=f"Shaft",
                showlegend=True,
                hovertemplate=(
                    f"Rotor Length ({rotor_length_units}): %{{x:.2f}}<br>Bending Moment ({moment_units}): %{{y:.2f}}"
                ),
            )
        )

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
        probe : list
            List with rs.Probe objects.
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
                aspectmode="manual",
                aspectratio=dict(x=2.5, y=1, z=1),
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
    bearing_frequency_range : tuple, optional
        The bearing frequency range used to calculate the intersection points.
        In some cases bearing coefficients will have to be extrapolated.
        The default is None. In this case the bearing frequency attribute is used.
    wn : array
        Undamped natural frequencies array.
    bearing : ross.BearingElement
        Bearing used in the calculation.
    intersection_points : array
        Points where there is a intersection between undamped natural frequency and
        the bearing stiffness.
    critical_points_modal : list
        List with modal results for each critical (intersection) point.
    """

    def __init__(
        self,
        stiffness_range,
        stiffness_log,
        bearing_frequency_range,
        wn,
        bearing,
        intersection_points,
        critical_points_modal,
    ):
        self.stiffness_range = stiffness_range
        self.stiffness_log = stiffness_log
        self.bearing_frequency_range = bearing_frequency_range
        self.wn = wn
        self.critical_points_modal = critical_points_modal
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
        intersection_points = copy.copy(self.intersection_points)
        bearing_frequency_range = self.bearing_frequency_range

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
            Q_(bearing0.kxx_interpolated(bearing_frequency_range), "N/m")
            .to(stiffness_units)
            .m
        )
        bearing_kyy_stiffness = (
            Q_(bearing0.kyy_interpolated(bearing_frequency_range), "N/m")
            .to(stiffness_units)
            .m
        )
        bearing_frequency = Q_(bearing_frequency_range, "rad/s").to(frequency_units).m

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

    def plot_mode_2d(
        self,
        critical_mode,
        fig=None,
        frequency_type="wd",
        title=None,
        length_units="m",
        frequency_units="rad/s",
        **kwargs,
    ):
        """Plot (2D view) the mode shape.

        Parameters
        ----------
        critical_mode : int
            The n'th critical mode.
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
        modal_critical = self.critical_points_modal[critical_mode]
        # select nearest forward
        forward_frequencies = modal_critical.wd[
            modal_critical.whirl_direction() == "Forward"
        ]
        idx_forward = (np.abs(forward_frequencies - modal_critical.speed)).argmin()
        forward_frequency = forward_frequencies[idx_forward]
        idx = (np.abs(modal_critical.wd - forward_frequency)).argmin()
        fig = modal_critical.plot_mode_2d(
            idx,
            fig=fig,
            frequency_type=frequency_type,
            title=title,
            length_units=length_units,
            frequency_units=frequency_units,
            **kwargs,
        )

        return fig

    def plot_mode_3d(
        self,
        critical_mode,
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
        critical_mode : int
            The n'th critical mode.
            Default is None
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
        modal_critical = self.critical_points_modal[critical_mode]
        # select nearest forward
        forward_frequencies = modal_critical.wd[
            modal_critical.whirl_direction() == "Forward"
        ]
        idx_forward = (np.abs(forward_frequencies - modal_critical.speed)).argmin()
        forward_frequency = forward_frequencies[idx_forward]
        idx = (np.abs(modal_critical.wd - forward_frequency)).argmin()
        fig = modal_critical.plot_mode_3d(
            idx,
            fig=fig,
            frequency_type=frequency_type,
            title=title,
            length_units=length_units,
            frequency_units=frequency_units,
            **kwargs,
        )

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

        return fig
