import os
import shutil
import warnings
from collections import Iterable
from copy import copy, deepcopy
from pathlib import Path

import bokeh.palettes as bp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.linalg as la
import scipy.signal as signal
import scipy.sparse.linalg as las
import toml
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Arrow, NormalHead, Label
from bokeh.models.glyphs import Text
from bokeh.plotting import figure, output_file, show
from cycler import cycler
from scipy import interpolate

import ross
from ross.bearing_seal_element import BearingElement
from ross.disk_element import DiskElement
from ross.materials import steel
from ross.results import (
    CampbellResults,
    FrequencyResponseResults,
    ForcedResponseResults,
    ModeShapeResults,
)
from ross.shaft_element import ShaftElement

__all__ = ["Rotor", "rotor_example"]

# set style and colors
plt.style.use("seaborn-white")
plt.style.use(
    {
        "lines.linewidth": 2.5,
        "axes.grid": True,
        "axes.linewidth": 0.1,
        "grid.color": ".9",
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.2,
    }
)

# set bokeh palette of colors
bokeh_colors = bp.RdGy[11]

_orig_rc_params = mpl.rcParams.copy()

seaborn_colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd"]


class Rotor(object):
    r"""A rotor object.

    This class will create a rotor with the shaft,
    disk, bearing and seal elements provided.

    Parameters
    ----------
    shaft_elements : list
        List with the shaft elements
    disk_elements : list
        List with the disk elements
    bearing_seal_elements : list
        List with the bearing elements
    w : float, optional
        Rotor speed. Defaults to 0.
    sparse : bool, optional
        If sparse, eigenvalues will be calculated with arpack.
        Default is True.
    n_eigen : int, optional
        Number of eigenvalues calculated by arpack.
        Default is 12.

    Returns
    -------
    A rotor object.

    Attributes
    ----------
    evalues : array
        Rotor's eigenvalues.
    evectors : array
        Rotor's eigenvectors.
    wn : array
        Rotor's natural frequencies in rad/s.
    wd : array
        Rotor's damped natural frequencies in rad/s.

    Examples
    --------
    >>> #  Rotor without damping with 2 shaft elements 1 disk and 2 bearings
    >>> import ross as rs
    >>> steel = rs.materials.steel
    >>> z = 0
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> tim0 = rs.ShaftElement(le, i_d, o_d, steel,
    ...                        shear_effects=True,
    ...                        rotary_inertia=True,
    ...                        gyroscopic=True)
    >>> tim1 = rs.ShaftElement(le, i_d, o_d, steel,
    ...                        shear_effects=True,
    ...                        rotary_inertia=True,
    ...                        gyroscopic=True)
    >>> shaft_elm = [tim0, tim1]
    >>> disk0 = rs.DiskElement.from_geometry(1, steel, 0.07, 0.05, 0.28)
    >>> stf = 1e6
    >>> bearing0 = rs.BearingElement(0, kxx=stf, cxx=0)
    >>> bearing1 = rs.BearingElement(2, kxx=stf, cxx=0)
    >>> rotor = rs.Rotor(shaft_elm, [disk0], [bearing0, bearing1])
    >>> rotor.run_modal()
    >>> rotor.wd[0] # doctest: +ELLIPSIS
    215.3707...
    """

    def __init__(
        self,
        shaft_elements,
        disk_elements=None,
        bearing_seal_elements=None,
        w=0,
        sparse=True,
        n_eigen=12,
        min_w=None,
        max_w=None,
        rated_w=None,
    ):

        self.parameters = {
            "w": w,
            "sparse": True,
            "n_eigen": n_eigen,
            "min_w": min_w,
            "max_w": max_w,
            "rated_w": rated_w,
        }
        self._w = w

        ####################################################
        # Config attributes
        ####################################################

        self.sparse = sparse
        self.n_eigen = n_eigen
        # operational speeds
        self.min_w = min_w
        self.max_w = max_w
        self.rated_w = rated_w

        ####################################################

        # flatten shaft_elements
        def flatten(l):
            for el in l:
                if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                    yield from flatten(el)
                else:
                    yield el

        # flatten and make a copy for shaft elements to avoid altering
        # attributes for elements that might be used in different rotors
        # e.g. altering shaft_element.n
        shaft_elements = [copy(el) for el in flatten(shaft_elements)]

        # set n for each shaft element
        for i, sh in enumerate(shaft_elements):
            if sh.n is None:
                sh.n = i

        if disk_elements is None:
            disk_elements = []
        if bearing_seal_elements is None:
            bearing_seal_elements = []

        self.shaft_elements = shaft_elements
        self.bearing_seal_elements = bearing_seal_elements
        self.disk_elements = disk_elements
        self.elements = [
            el
            for el in flatten(
                [self.shaft_elements, self.disk_elements, self.bearing_seal_elements]
            )
        ]

        ####################################################
        # Rotor summary
        ####################################################
        columns = [
            "type",
            "n",
            "L",
            "node_pos",
            "node_pos_r",
            "i_d",
            "o_d",
            "i_d_r",
            "o_d_r",
            "material",
            "rho",
            "volume",
            "m",
        ]

        df_shaft = pd.DataFrame([el.summary() for el in self.shaft_elements])
        df_disks = pd.DataFrame([el.summary() for el in self.disk_elements])
        df_bearings = pd.DataFrame([el.summary() for el in self.bearing_seal_elements])

        nodes_pos_l = np.zeros(len(df_shaft.n_l))
        nodes_pos_r = np.zeros(len(df_shaft.n_l))

        for i in range(len(df_shaft)):
            if i == 0:
                nodes_pos_r[i] = nodes_pos_r[i] + df_shaft.loc[i, "L"]
                continue
            if df_shaft.loc[i, "n_l"] == df_shaft.loc[i - 1, "n_l"]:
                nodes_pos_l[i] = nodes_pos_l[i - 1]
                nodes_pos_r[i] = nodes_pos_r[i - 1]
            else:
                nodes_pos_l[i] = nodes_pos_r[i - 1]
                nodes_pos_r[i] = nodes_pos_l[i] + df_shaft.loc[i, "L"]

        df_shaft["nodes_pos_l"] = nodes_pos_l
        df_shaft["nodes_pos_r"] = nodes_pos_r
        # bearings

        df = pd.concat([df_shaft, df_disks, df_bearings], sort=True)
        df = df.sort_values(by="n_l")
        df = df.reset_index(drop=True)

        self.df_disks = df_disks
        self.df_bearings = df_bearings
        self.df_shaft = df_shaft

        # check consistence for disks and bearings location
        if df.n_l.max() > df[df.type == "ShaftElement"].n_r.max():
            raise ValueError("Trying to set disk or bearing outside shaft")

        self.df = df

        # nodes axial position and diameter
        nodes_pos = list(df_shaft.groupby("n_l")["nodes_pos_l"].max())
        nodes_pos.append(df_shaft["nodes_pos_r"].iloc[-1])
        self.nodes_pos = nodes_pos

        nodes_i_d = list(df_shaft.groupby("n_l")["i_d"].min())
        nodes_i_d.append(df_shaft["i_d"].iloc[-1])
        self.nodes_i_d = nodes_i_d

        nodes_o_d = list(df_shaft.groupby("n_l")["o_d"].min())
        nodes_o_d.append(df_shaft["o_d"].iloc[-1])
        self.nodes_o_d = nodes_o_d

        nodes_le = list(df_shaft.groupby("n_l")["L"].min())
        nodes_le.append(df_shaft["L"].iloc[-1])
        self.nodes_le = nodes_le

        self.nodes = list(range(len(self.nodes_pos)))
        self.elements_length = [sh_el.L for sh_el in self.shaft_elements]
        self.L = nodes_pos[-1]

        # rotor mass can also be calculated with self.M()[::4, ::4].sum()
        self.m_disks = np.sum([disk.m for disk in self.disk_elements])
        self.m_shaft = np.sum([sh_el.m for sh_el in self.shaft_elements])
        self.m = self.m_disks + self.m_shaft

        # values for evalues and evectors will be calculated by self._calc_system
        self.evalues = None
        self.evectors = None
        self.wn = None
        self.wd = None
        self.lti = None

        self._v0 = None  # used to call eigs

        # number of dofs
        self.ndof = 4 * max([el.n for el in shaft_elements]) + 8

        #  values for static analysis will be calculated by def static
        self.Vx = None
        self.Bm = None
        self.disp_y = None

        #  diameter at node position
        self.run_modal()

    def __eq__(self, other):
        if self.elements == other.elements and self.parameters == other.parameters:
            return True
        else:
            return False

    def run_modal(self):
        self.evalues, self.evectors = self._eigen(self.w)
        wn_len = len(self.evalues) // 2
        self.wn = (np.absolute(self.evalues))[:wn_len]
        self.wd = (np.imag(self.evalues))[:wn_len]
        self.damping_ratio = (-np.real(self.evalues) / np.absolute(self.evalues))[
            :wn_len
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.log_dec = (
                2 * np.pi * self.damping_ratio / np.sqrt(1 - self.damping_ratio ** 2)
            )
        self.lti = self._lti()

    def convergence(self, n_eigval=0, err_max=1e-02, output_html=False):
        """
        Function to analyze the eigenvalues convergence through the number of
        shaft elements. Every new run doubles the number os shaft elements.

        Parameters
        ----------
        n_eigval : int
            The nth eigenvalue which the convergence analysis will run.
            Default is 0 (the first eigenvalue).
        err_max : float
            Maximum allowable convergence error.
            Default is 1e-02
        output_html : Boolean, optional
            outputs a html file.
            Default is False

        Returns
        -------
        p : bokeh.figure
            Bokeh plot showing the results.

        Example
        -------
        >>> import ross as rs
        >>> i_d = 0
        >>> o_d = 0.05
        >>> n = 6
        >>> L = [0.25 for _ in range(n)]
        ...
        >>> shaft_elem = [rs.ShaftElement(l, i_d, o_d, steel,
        ... shear_effects=True, rotary_inertia=True, gyroscopic=True) for l in L]
        >>> disk0 = DiskElement.from_geometry(2, steel, 0.07, 0.05, 0.28)
        >>> disk1 = DiskElement.from_geometry(4, steel, 0.07, 0.05, 0.35)
        >>> bearing0 = BearingElement(0, kxx=1e6, kyy=8e5, cxx=2e3)
        >>> bearing1 = BearingElement(6, kxx=1e6, kyy=8e5, cxx=2e3)
        >>> rotor0 = Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])
        >>> len(rotor0.shaft_elements)
        6
        >>> convergence = rotor0.convergence(n_eigval=0, err_max=1e-08)
        >>> len(rotor0.shaft_elements)
        96
        """
        el_num = np.array([len(self.shaft_elements)])
        eigv_arr = np.array([])
        error_arr = np.array([0])

        self.run_modal()
        eigv_arr = np.append(eigv_arr, self.wn[n_eigval])

        # this value is up to start the loop while
        error = 1
        nel_r = 2

        while error > err_max:
            shaft_elem = []
            disk_elem = []
            brgs_elem = []

            for i, leng in enumerate(self.shaft_elements):
                le = self.shaft_elements[i].L / nel_r
                o_ds = self.shaft_elements[i].o_d
                i_ds = self.shaft_elements[i].i_d

                # loop to double the number of element
                for j in range(nel_r):
                    shaft_elem.append(
                        ShaftElement(
                            le,
                            i_ds,
                            o_ds,
                            material=self.shaft_elements[i].material,
                            shear_effects=True,
                            rotary_inertia=True,
                            gyroscopic=True,
                        )
                    )

            for DiskEl in self.disk_elements:
                aux_DiskEl = deepcopy(DiskEl)
                aux_DiskEl.n = nel_r * DiskEl.n
                disk_elem.append(aux_DiskEl)

            for Brg_SealEl in self.bearing_seal_elements:
                aux_Brg_SealEl = deepcopy(Brg_SealEl)
                aux_Brg_SealEl.n = nel_r * Brg_SealEl.n
                brgs_elem.append(aux_Brg_SealEl)

            rotor = Rotor(
                shaft_elem, disk_elem, brgs_elem, w=self.w, n_eigen=self.n_eigen
            )
            rotor.run_modal()

            eigv_arr = np.append(eigv_arr, rotor.wn[n_eigval])
            el_num = np.append(el_num, len(shaft_elem))

            error = min(eigv_arr[-1], eigv_arr[-2]) / max(eigv_arr[-1], eigv_arr[-2])
            error = 1 - error

            error_arr = np.append(error_arr, 100 * error)
            nel_r *= 2

        self.__dict__ = rotor.__dict__
        self.error_arr = error_arr
        if output_html:
            output_file("convergence.html")
        source = ColumnDataSource(
            data=dict(x0=el_num[1:], y0=eigv_arr[1:], x1=el_num[1:], y1=error_arr[1:])
        )

        TOOLS = "pan,wheel_zoom,box_zoom,hover,reset,save,"
        TOOLTIPS1 = [("Frequency:", "@y0 Hz"), ("Number of Elements", "@x0")]
        TOOLTIPS2 = [("Relative Error:", "@y1"), ("Number of Elements", "@x1")]
        # create a new plot and add a renderer
        freq_arr = figure(
            tools=TOOLS,
            tooltips=TOOLTIPS1,
            width=500,
            height=500,
            title="Frequency Evaluation",
            x_axis_label="Numer of Elements",
            y_axis_label="Frequency (Hz)",
        )
        freq_arr.line("x0", "y0", source=source, line_width=3, line_color="crimson")
        freq_arr.circle("x0", "y0", source=source, fill_color="crimson", size=8)

        # create another new plot and add a renderer
        rel_error = figure(
            tools=TOOLS,
            tooltips=TOOLTIPS2,
            width=500,
            height=500,
            title="Relative Error Evaluation",
            x_axis_label="Number of Elements",
            y_axis_label="Relative Rrror",
        )
        rel_error.line(
            "x1", "y1", source=source, line_width=3, line_color="darkslategray"
        )
        rel_error.circle("x1", "y1", source=source, fill_color="darkslategray", size=8)

        # put the subplots in a gridplot
        p = gridplot([[freq_arr, rel_error]])

        return p

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value
        self.run_modal()

    def _dofs(self, element):

        """The first and last dof for a given element"""
        node = element.n
        n1 = 4 * node

        if isinstance(element, ShaftElement):
            n2 = n1 + 8
        if isinstance(element, DiskElement):
            n2 = n1 + 4
        if isinstance(element, BearingElement):
            n2 = n1 + 2

        return n1, n2

    def M(self):
        r"""Mass matrix for an instance of a rotor.

        Returns
        -------
        Mass matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.M()[:4, :4]
        array([[ 1.42050794,  0.        ,  0.        ,  0.04931719],
               [ 0.        ,  1.42050794, -0.04931719,  0.        ],
               [ 0.        , -0.04931719,  0.00231392,  0.        ],
               [ 0.04931719,  0.        ,  0.        ,  0.00231392]])
        """
        #  Create the matrices
        M0 = np.zeros((self.ndof, self.ndof))

        for elm in self.shaft_elements:
            n1, n2 = self._dofs(elm)
            M0[n1:n2, n1:n2] += elm.M()

        for elm in self.disk_elements:
            n1, n2 = self._dofs(elm)
            M0[n1:n2, n1:n2] += elm.M()

        return M0

    def K(self, w=None):
        """Stiffness matrix for an instance of a rotor.

        Returns
        -------
        Stiffness matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> np.round(rotor.K()[:4, :4]/1e6)
        array([[47.,  0.,  0.,  6.],
               [ 0., 46., -6.,  0.],
               [ 0., -6.,  1.,  0.],
               [ 6.,  0.,  0.,  1.]])
        """
        if w is None:
            w = self.w
        #  Create the matrices
        K0 = np.zeros((self.ndof, self.ndof))

        for elm in self.shaft_elements:
            n1, n2 = self._dofs(elm)
            K0[n1:n2, n1:n2] += elm.K()

        for elm in self.bearing_seal_elements:
            n1, n2 = self._dofs(elm)
            K0[n1:n2, n1:n2] += elm.K(w)
        #  Skew-symmetric speed dependent contribution to element stiffness matrix
        #  from the internal damping.

        return K0

    def C(self, w=None):
        """Damping matrix for an instance of a rotor.

        Returns
        -------
        Damping matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.C()[:4, :4]
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """
        if w is None:
            w = self.w
        #  Create the matrices
        C0 = np.zeros((self.ndof, self.ndof))

        for elm in self.bearing_seal_elements:
            n1, n2 = self._dofs(elm)
            C0[n1:n2, n1:n2] += elm.C(w)

        return C0

    def G(self):
        """Gyroscopic matrix for an instance of a rotor.

        Returns
        -------
        Gyroscopic matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> rotor.G()[:4, :4]
        array([[ 0.        ,  0.01943344, -0.00022681,  0.        ],
               [-0.01943344,  0.        ,  0.        , -0.00022681],
               [ 0.00022681,  0.        ,  0.        ,  0.0001524 ],
               [ 0.        ,  0.00022681, -0.0001524 ,  0.        ]])
        """
        #  Create the matrices
        G0 = np.zeros((self.ndof, self.ndof))

        for elm in self.shaft_elements:
            n1, n2 = self._dofs(elm)
            G0[n1:n2, n1:n2] += elm.G()

        for elm in self.disk_elements:
            n1, n2 = self._dofs(elm)
            G0[n1:n2, n1:n2] += elm.G()

        return G0

    def A(self, w=None):
        """State space matrix for an instance of a rotor.

        Returns
        -------
        State space matrix for the rotor.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> np.round(rotor.A()[50:56, :2])
        array([[     0.,  10927.],
               [-10924.,     -0.],
               [  -174.,      0.],
               [    -0.,   -174.],
               [    -0.,  10723.],
               [-10719.,     -0.]])
        """
        if w is None:
            w = self.w

        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)

        # fmt: off
        A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-self.M(), self.K(w)), la.solve(-self.M(), (self.C(w) + self.G() * w))])])
        # fmt: on

        return A

    @staticmethod
    def _index(eigenvalues):
        r"""Function used to generate an index that will sort
        eigenvalues and eigenvectors based on the imaginary (wd)
        part of the eigenvalues. Positive eigenvalues will be
        positioned at the first half of the array.

        Parameters
        ----------
        eigenvalues: array
            Array with the eigenvalues.

        Returns
        -------
        idx:
            An array with indices that will sort the
            eigenvalues and eigenvectors.

        Examples
        --------
        >>> rotor = rotor_example()
        >>> evalues, evectors = rotor._eigen(0, sorted_=True)
        >>> idx = rotor._index(evalues)
        >>> idx[:6] # doctest: +ELLIPSIS
        array([0, 1, 2, 3, 4, ...
        """
        # avoid float point errors when sorting
        evals_truncated = np.around(eigenvalues, decimals=10)
        a = np.imag(evals_truncated)  # First column
        b = np.absolute(evals_truncated)  # Second column
        ind = np.lexsort((b, a))  # Sort by imag, then by absolute
        # Positive eigenvalues first
        positive = [i for i in ind[len(a) // 2 :]]
        negative = [i for i in ind[: len(a) // 2]]

        idx = np.array([positive, negative]).flatten()

        return idx

    def _eigen(self, w=None, sorted_=True, A=None):
        r"""This method will return the eigenvalues and eigenvectors of the
        state space matrix A, sorted by the index method which considers
        the imaginary part (wd) of the eigenvalues for sorting.
        To avoid sorting use sorted_=False

        Parameters
        ----------
        w: float
            Rotor speed.

        Returns
        -------
        evalues: array
            An array with the eigenvalues
        evectors array
            An array with the eigenvectors

        Examples
        --------
        >>> rotor = rotor_example()
        >>> evalues, evectors = rotor._eigen(0)
        >>> evalues[0].imag # doctest: +ELLIPSIS
        91.796...
        """
        if w is None:
            w = self.w
        if A is None:
            A = self.A(w)

        if self.sparse is True:
            try:
                evalues, evectors = las.eigs(
                    A,
                    k=self.n_eigen,
                    sigma=0,
                    ncv=2 * self.n_eigen,
                    which="LM",
                    v0=self._v0,
                )
                # store v0 as a linear combination of the previously
                # calculated eigenvectors to use in the next call to eigs
                self._v0 = np.real(sum(evectors.T))
            except las.ArpackError:
                evalues, evectors = la.eig(A)
        else:
            evalues, evectors = la.eig(A)

        if sorted_ is False:
            return evalues, evectors

        idx = self._index(evalues)

        return evalues[idx], evectors[:, idx]

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

        Examples
        --------
        >>> rotor = rotor_example()
        >>> # H matrix for the 0th node
        >>> h_kappa = rotor.H_kappa(0, 0)
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

        w is the the index of the natural frequency of interest.
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

        Examples
        --------
        >>> rotor = rotor_example()
        >>> # kappa for each node of the first natural frequency
        >>> # Major axes for node 0 and natural frequency (mode) 0.
        >>> rotor.kappa(0, 0)['Major axes'] # doctest: +ELLIPSIS
        0.00141...
        >>> # kappa for node 2 and natural frequency (mode) 3.
        >>> rotor.kappa(2, 3)['kappa'].round(2) # doctest: +ELLIPSIS
        -0.0
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

        Examples
        --------
        >>> rotor = rotor_example()
        >>> # kappa for each node of the first natural frequency
        >>> rotor.kappa_mode(0) # doctest: +ELLIPSIS
        [...]
        """
        kappa_mode = [self.kappa(node, w)["kappa"] for node in self.nodes]
        return kappa_mode

    def whirl_direction(self):
        """Get the whirl direction for each frequency."""
        # whirl direction/values are methods because they are expensive.
        whirl_w = [whirl(self.kappa_mode(wd)) for wd in range(len(self.wd))]

        return np.array(whirl_w)

    def whirl_values(self):
        """Get the whirl value (0., 0.5, or 1.) for each frequency."""
        return whirl_to_cmap(self.whirl_direction())

    def orbit(self):
        pass

    def _lti(self):
        """Continuous-time linear time invariant system.

        This method is used to create a Continuous-time linear
        time invariant system for the mdof system.
        From this system we can obtain poles, impulse response,
        generate a bode, etc.
        """
        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)

        # x' = Ax + Bu
        B2 = I
        A = self.A()
        # fmt: off
        B = np.vstack([Z,
                       la.solve(self.M(), B2)])
        # fmt: on

        # y = Cx + Du
        # Observation matrices
        Cd = I
        Cv = Z
        Ca = Z

        # fmt: off
        C = np.hstack((Cd - Ca @ la.solve(self.M(), self.K()), Cv - Ca @ la.solve(self.M(), self.C())))
        # fmt: on
        D = Ca @ la.solve(self.M(), B2)

        sys = signal.lti(A, B, C, D)

        return sys

    def transfer_matrix(self, w=None, modes=None):
        B = self.lti.B
        C = self.lti.C
        D = self.lti.D

        # calculate eigenvalues and eigenvectors using la.eig to get
        # left and right eigenvectors.

        evals, psi, = la.eig(self.A(w))

        psi_inv = la.inv(psi)

        if modes is not None:
            n = self.ndof  # n dof -> number of modes
            m = len(modes)  # -> number of desired modes
            # idx to get each evalue/evector and its conjugate
            idx = np.zeros((2 * m), int)
            idx[0:m] = modes  # modes
            idx[m:] = range(2 * n)[-m:]  # conjugates (see how evalues are ordered)

            evals = evals[np.ix_(idx)]
            psi = psi[np.ix_(range(2 * n), idx)]
            psi_inv = psi_inv[np.ix_(idx, range(2 * n))]

        diag = np.diag([1 / (1j * w - lam) for lam in evals])

        H = C @ psi @ diag @ psi_inv @ B + D

        return H

    def run_freq_response(self, frequency_range=None, modes=None):
        """Frequency response for a mdof system.

        This method returns the frequency response for a mdof system
        given a range of frequencies and the modes that will be used.

        Parameters
        ----------
        force : array, optional
            Force array (needs to have the same length as frequencies array).
            If not given the impulse response is calculated.
        omega : array, optional
            Array with the desired range of frequencies (the default
             is 0 to 1.5 x highest damped natural frequency.
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).

        Returns
        -------
        omega : array
            Array with the frequencies
        magdb : array
            Magnitude (dB) of the frequency response for each pair input/output.
            The order of the array is: [output, input, magnitude]
        phase : array
            Phase of the frequency response for each pair input/output.
            The order of the array is: [output, input, phase]

        Examples
        --------
        """
        if frequency_range is None:
            frequency_range = np.linspace(0, max(self.evalues.imag) * 1.5, 1000)

        freq_resp = np.empty(
            (self.lti.inputs, self.lti.outputs, len(frequency_range)), dtype=np.complex
        )

        for i, w in enumerate(frequency_range):
            H = self.transfer_matrix(w=w, modes=modes)
            freq_resp[..., i] = H

        results = FrequencyResponseResults(
            freq_resp,
            new_attributes={
                "frequency_range": frequency_range,
                "magnitude": abs(freq_resp),
                "phase": np.angle(freq_resp),
            },
        )

        return results

    def run_forced_response(self, force=None, frequency_range=None, modes=None):
        freq_resp = self.run_freq_response(frequency_range=frequency_range, modes=modes)

        forced_resp = np.zeros(
            (self.ndof, len(freq_resp.frequency_range)), dtype=np.complex
        )

        for i in range(len(freq_resp.frequency_range)):
            forced_resp[:, i] = freq_resp[..., i] @ force[..., i]

        forced_resp = ForcedResponseResults(
            forced_resp,
            new_attributes={
                "frequency_range": frequency_range,
                "magnitude": abs(forced_resp),
                "phase": np.angle(forced_resp),
            },
        )

        return forced_resp

    def _unbalance_force(self, node, magnitude, phase, omega):
        """Function to calculate unbalance force"""

        F0 = np.zeros((self.ndof, len(omega)), dtype=np.complex128)
        me = magnitude
        delta = phase
        b0 = np.array(
            [
                me * np.exp(1j * delta),
                -1j * me * np.exp(1j * delta),
                0,  # 1j*(Id - Ip)*beta*np.exp(1j*gamma),
                0,
            ]
        )  # (Id - Ip)*beta*np.exp(1j*gamma)])

        n0 = 4 * node
        n1 = n0 + 4
        for i, w in enumerate(omega):
            F0[n0:n1, i] += w ** 2 * b0

        return F0

    def unbalance_response(self, node, magnitude, phase, frequency_range=None):
        """frequency response for a mdof system.

        This method returns the frequency response for a mdof system
        given a range of frequencies and the modes that will be used.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        magnitude : list, float
            Unbalance magnitude (kg.m)
        phase : list, float
            Unbalance phase (rad)

        Returns
        -------
        frequency_range : array
            Array with the frequencies
        magdb : array
            Magnitude (dB) of the frequency response for each pair input/output.
            The order of the array is: [output, input, magnitude]
        phase : array
            Phase of the frequency response for each pair input/output.
            The order of the array is: [output, input, phase]

        Examples
        --------
        """
        force = np.zeros((self.ndof, len(frequency_range)), dtype=np.complex)

        try:
            for n, m, p in zip(node, magnitude, phase):
                force += self._unbalance_force(n, m, p, frequency_range)
        except TypeError:
            force = self._unbalance_force(node, magnitude, phase, frequency_range)

        forced_response = self.run_forced_response(force, frequency_range)

        return forced_response

    def time_response(self, F, t, ic=None):
        """Time response for a rotor.

        This method returns the time response for a rotor
        given a force, time and initial conditions.

        Parameters
        ----------
        F : array
            Force array (needs to have the same length as time array).
        t : array
            Time array.
        ic : array, optional
            The initial conditions on the state vector (zero by default).

        Returns
        -------
        t : array
            Time values for the output.
        yout : array
            System response.
        xout : array
            Time evolution of the state vector.


        Examples
        --------
        """
        return signal.lsim(self.lti, F, t, X0=ic)

    def plot_rotor(self, nodes=1, ax=None, output_html=False, bk_ax=None):
        """Plots a rotor object.

        This function will take a rotor object and plot its shaft,
        disks and bearing elements

        Parameters
        ----------
        nodes : int, optional
            Increment that will be used to plot nodes label.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        bk_ax : bokeh plotting axes, optional
            Axes in which the plot will be drawn.
        output_html : Boolean, optional
            outputs a html file.
            Default is False

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        bk_ax : bokeh plotting axes
            Returns the axes object with the plot.

        Examples:
        """

        # check slenderness ratio of beam elements
        SR = np.array([])
        for shaft in self.shaft_elements:
            if shaft.slenderness_ratio < 1.6:
                SR = np.append(SR, shaft.n)
        if len(SR) != 0:
            warnings.warn(
                "The beam elements "
                + str(SR)
                + " have slenderness ratio (G*A*L^2 / EI) of less than 1.6."
                + " Results may not converge correctly"
            )

        if ax is None:
            ax = plt.gca()

        #  plot shaft centerline
        shaft_end = self.nodes_pos[-1]
        ax.plot([-0.2 * shaft_end, 1.2 * shaft_end], [0, 0], "k-.")

        try:
            max_diameter = max([disk.o_d for disk in self.disk_elements])
        except (ValueError, AttributeError):
            max_diameter = max([shaft.o_d for shaft in self.shaft_elements])

        # matplotlib
        ax.set_ylim(-1.2 * max_diameter, 1.2 * max_diameter)
        ax.axis("equal")
        ax.set_xlabel("Axial location (m)")
        ax.set_ylabel("Shaft radius (m)")

        # bokeh plot - output to static HTML file
        if output_html:
            output_file("rotor.html")

        # bokeh plot - create a new plot
        bk_ax = figure(
            tools="pan, wheel_zoom, reset, save",
            width=1800,
            height=900,
            x_range=[-0.1 * shaft_end, 1.1 * shaft_end],
            y_range=[-0.3 * shaft_end, 0.3 * shaft_end],
            title="Rotor model",
            x_axis_label="Axial location (m)",
            y_axis_label="Shaft radius (m)",
            match_aspect=True,
        )

        # bokeh plot - plot shaft centerline
        bk_ax.line(
            [-0.2 * shaft_end, 1.2 * shaft_end],
            [0, 0],
            line_width=3,
            line_dash="dotdash",
            line_color=bokeh_colors[0],
        )

        # plot nodes
        text = []
        x_pos = []
        for node, position in enumerate(self.nodes_pos[::nodes]):
            # bokeh plot
            text.append(str(node))
            x_pos.append(position)

            # matplotlib
            ax.plot(
                position,
                0,
                zorder=2,
                ls="",
                marker="D",
                color="#6caed6",
                markersize=10,
                alpha=0.6,
            )
            ax.text(
                position,
                0,
                f"{node*nodes}",
                size="smaller",
                horizontalalignment="center",
                verticalalignment="center",
            )

        # bokeh plot - plot nodes
        y_pos = np.linspace(0, 0, len(self.nodes_pos))

        source = ColumnDataSource(dict(x=x_pos, y=y_pos, text=text))

        bk_ax.circle(
            x=x_pos, y=y_pos, size=30, fill_alpha=0.8, fill_color=bokeh_colors[6]
        )

        glyph = Text(
            x="x",
            y="y",
            text="text",
            text_font_style="bold",
            text_baseline="middle",
            text_align="center",
            text_alpha=1.0,
            text_color=bokeh_colors[0],
        )
        bk_ax.add_glyph(source, glyph)

        # plot shaft elements
        for sh_elm in self.shaft_elements:
            position = self.nodes_pos[sh_elm.n]
            sh_elm.patch(position, SR, ax, bk_ax)

        # plot disk elements
        for disk in self.disk_elements:
            position = (self.nodes_pos[disk.n], self.nodes_o_d[disk.n] / 2)
            length = min(self.nodes_le)
            disk.patch(position, length, ax, bk_ax)

        # plot bearings
        for bearing in self.bearing_seal_elements:
            position = (self.nodes_pos[bearing.n], -self.nodes_o_d[bearing.n] / 2)
            length = min(self.nodes_le)
            bearing.patch(position, length, ax, bk_ax)

        show(bk_ax)

        return bk_ax, ax

    def run_campbell(self, speed_range, frequencies=6, frequency_type="wd"):
        """Calculates the Campbell diagram.

        This function will calculate the damped natural frequencies
        for a speed range.

        Parameters
        ----------
        speed_range : array
            Array with the speed range in rad/s.
        frequencies : int, optional
            Number of frequencies that will be calculated.
            Default is 6.

        Returns
        -------
        results : array
            Array with the natural frequencies corresponding to each speed
            of the speed_rad array. It will be returned if plot=False.

        Examples
        --------
        >>> rotor1 = rotor_example()
        >>> speed = np.linspace(0, 400, 101)
        >>> camp = rotor1.run_campbell(speed)
        >>> camp.plot() # doctest: +ELLIPSIS
        (<Figure ...
        """
        rotor_current_speed = self.w

        # store in results [speeds(x axis), frequencies[0] or logdec[1] or
        # whirl[2](y axis), 3]
        results = np.zeros([len(speed_range), frequencies, 5])

        for i, w in enumerate(speed_range):
            self.w = w

            if frequency_type == "wd":
                results[i, :, 0] = self.wd[:frequencies]
                results[i, :, 1] = self.log_dec[:frequencies]
                results[i, :, 2] = self.whirl_values()[:frequencies]
            else:
                idx = self.wn.argsort()
                results[i, :, 0] = self.wn[idx][:frequencies]
                results[i, :, 1] = self.log_dec[idx][:frequencies]
                results[i, :, 2] = self.whirl_values()[idx][:frequencies]

            results[i, :, 3] = w
            results[i, :, 4] = self.wn[:frequencies]

        results = CampbellResults(
            results,
            new_attributes={
                "speed_range": speed_range,
                "wd": results[..., 0],
                "log_dec": results[..., 1],
                "whirl_values": results[..., 2],
            },
        )

        self.w = rotor_current_speed

        return results

    def run_mode_shapes(self):

        kappa_modes = []
        for mode in range(len(self.wn)):
            kappa_color = []
            kappa_mode = self.kappa_mode(mode)
            for kappa in kappa_mode:
                kappa_color.append("tab:blue" if kappa > 0 else "tab:red")
            kappa_modes.append(kappa_color)

        mode_shapes = ModeShapeResults(
            self.evectors[: self.ndof],
            new_attributes={
                "ndof": self.ndof,
                "nodes": self.nodes,
                "nodes_pos": self.nodes_pos,
                "elements_length": self.elements_length,
                "w": self.w,
                "wd": self.wd,
                "log_dec": self.log_dec,
                "kappa_modes": kappa_modes,
            },
        )

        return mode_shapes

    def plot_ucs(self, stiffness_range=None, num=20, ax=None, output_html=False):
        """Plot undamped critical speed map.

        This method will plot the undamped critical speed map for a given range
        of stiffness values. If the range is not provided, the bearing
        stiffness at rated speed will be used to create a range.

        Parameters
        ----------
        stiffness_range : tuple, optional
            Tuple with (start, end) for stiffness range.
        num : int
            Number of steps in the range.
            Default is 20.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        output_html : Boolean, optional
            outputs a html file.
            Default is False

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        bk_ax : bokeh plot axes
            Returns the axes object with the plot. 
       """
        if ax is None:
            ax = plt.gca()

        if stiffness_range is None:
            if self.rated_w is not None:
                bearing = self.bearing_seal_elements[0]
                k = bearing.kxx.interpolated(self.rated_w)
                k = int(np.log10(k))
                stiffness_range = (k - 3, k + 3)
            else:
                stiffness_range = (6, 11)

        stiffness_log = np.logspace(*stiffness_range, num=num)
        rotor_wn = np.zeros((4, len(stiffness_log)))

        bearings_elements = []  # exclude the seals
        for bearing in self.bearing_seal_elements:
            if type(bearing) == BearingElement:
                bearings_elements.append(bearing)

        for i, k in enumerate(stiffness_log):
            bearings = [BearingElement(b.n, kxx=k, cxx=0) for b in bearings_elements]
            rotor = self.__class__(
                self.shaft_elements, self.disk_elements, bearings, n_eigen=16
            )
            rotor.run()
            rotor_wn[:, i] = rotor.wn[:8:2]

        ax.set_prop_cycle(cycler("color", seaborn_colors))
        ax.loglog(stiffness_log, rotor_wn.T)
        ax.set_xlabel("Bearing Stiffness (N/m)")
        ax.set_ylabel("Critical Speed (rad/s)")

        bearing0 = bearings_elements[0]

        ax.plot(
            bearing0.kxx.interpolated(bearing0.w),
            bearing0.w,
            marker="o",
            color="k",
            alpha=0.25,
            markersize=5,
            lw=0,
            label="kxx",
        )
        ax.plot(
            bearing0.kyy.interpolated(bearing0.w),
            bearing0.w,
            marker="s",
            color="k",
            alpha=0.25,
            markersize=5,
            lw=0,
            label="kyy",
        )
        ax.legend()

        # bokeh plot - output to static HTML file
        if output_html:
            output_file("Plot_UCS.html")

        # bokeh plot - create a new plot
        bk_ax = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=1200,
            height=900,
            title="Undamped critical speed map",
            x_axis_label="Bearing Stiffness (N/m)",
            y_axis_label="Critical Speed (rad/s)",
            x_axis_type="log",
            y_axis_type="log",
        )

        # bokeh plot - plot shaft centerline
        bk_ax.circle(
            bearing0.kxx.interpolated(bearing0.w),
            bearing0.w,
            size=5,
            fill_alpha=0.5,
            fill_color=bokeh_colors[0],
            legend="Kxx",
        )
        bk_ax.square(
            bearing0.kyy.interpolated(bearing0.w),
            bearing0.w,
            size=5,
            fill_alpha=0.5,
            fill_color=bokeh_colors[0],
            legend="Kyy",
        )
        for j in range(rotor_wn.T.shape[1]):
            bk_ax.line(
                stiffness_log,
                np.transpose(rotor_wn.T)[j],
                line_width=3,
                line_color=bokeh_colors[-j + 1],
            )
        show(bk_ax)

        return ax

    def plot_level1(
        self, n=None, stiffness_range=None, num=5, ax=None, output_html=False, **kwargs
    ):
        """Plot level 1 stability analysis.

        This method will plot the stability 1 analysis for a
        given stiffness range.

        Parameters
        ----------
        stiffness_range : tuple, optional
            Tuple with (start, end) for stiffness range.
        num : int
            Number of steps in the range.
            Default is 5.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        output_html : Boolean, optional
            outputs a html file.
            Default is False

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        bk_ax : bokeh plot axes
            Returns the axes object with the plot.
        """
        if ax is None:
            ax = plt.gca()

        stiffness = np.linspace(*stiffness_range, num)

        log_dec = np.zeros(len(stiffness))

        # set rotor speed to mcs
        speed = self.rated_w

        for i, Q in enumerate(stiffness):
            bearings = [copy(b) for b in self.bearing_seal_elements]
            cross_coupling = BearingElement(n=n, kxx=0, cxx=0, kxy=Q, kyx=-Q)
            bearings.append(cross_coupling)

            rotor = self.__class__(
                self.shaft_elements, self.disk_elements, bearings, w=speed
            )

            non_backward = rotor.whirl_direction() != "Backward"
            log_dec[i] = rotor.log_dec[non_backward][0]

        ax.plot(stiffness, log_dec, "--", **kwargs)
        ax.set_xlabel("Applied Cross Coupled Stiffness, Q (N/m)")
        ax.set_ylabel("Log Dec")

        # bokeh plot - output to static HTML file
        if output_html:
            output_file("Plot_level1.html")

        # bokeh plot - create a new plot
        bk_ax = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=1200,
            height=900,
            title="Level 1 stability analysis",
            x_axis_label="Applied Cross Coupled Stiffness, Q (N/m)",
            y_axis_label="Log Dec",
        )

        # bokeh plot - plot shaft centerline
        bk_ax.line(stiffness, log_dec, line_width=3, line_color=bokeh_colors[0])

        show(bk_ax)

        return ax, bk_ax

    def plot_time_response(self, F, t, dof, ax=None, output_html=False):
        """Plot the time response.

        This function will take a rotor object and plot its time response
        given a force and a time.

        Parameters
        ----------
        F : array
            Force array (needs to have the same number of rows as time array).
            Each column corresponds to a dof and each row to a time.
        t : array
            Time array.
        dof : int
            Degree of freedom that will be observed.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.
        output_html : Boolean, optional
            outputs a html file.
            Default is False

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.
        bk_ax : bokeh plot axes
            Returns the axes object with the plot

        Examples:
        ---------
        """
        t_, yout, xout = self.time_response(F, t)

        if ax is None:
            ax = plt.gca()

        ax.plot(t, yout[:, dof])

        if dof % 4 == 0:
            obs_dof = "$x$"
            amp = "m"
        elif dof + 1 % 4 == 0:
            obs_dof = "$y$"
            amp = "m"
        elif dof + 2 % 4 == 0:
            obs_dof = "$\alpha$"
            amp = "rad"
        else:
            obs_dof = "$\beta$"
            amp = "rad"

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (%s)" % amp)
        ax.set_title(
            "Response for node %s and degree of freedom %s" % (dof // 4, obs_dof)
        )

        # bokeh plot - output to static HTML file
        if output_html:
            output_file("time_response.html")

        # bokeh plot - create a new plot
        bk_ax = figure(
            tools="pan, box_zoom, wheel_zoom, reset, save",
            width=1200,
            height=900,
            title="Response for node %s and degree of freedom %s" % (dof // 4, obs_dof),
            x_axis_label="Time (s)",
            y_axis_label="Amplitude (%s)" % amp,
        )

        # bokeh plot - plot shaft centerline
        bk_ax.line(t, yout[:, dof], line_width=3, line_color=bokeh_colors[0])

        show(bk_ax)

        return ax, bk_ax

    def save_mat(self, file_name):
        """Save matrices and rotor model to a .mat file."""
        dic = {
            "M": self.M(),
            "K": self.K(),
            "C": self.C(),
            "G": self.G(),
            "nodes": self.nodes_pos,
        }

        sio.savemat("%s/%s.mat" % (os.getcwd(), file_name), dic)

    def save(self, file_name):
        """Save rotor to toml file.

        Parameters
        ----------
        file_name : str
        """
        main_path = os.path.dirname(ross.__file__)
        path = Path(main_path)
        path_rotors = path / "rotors"

        if os.path.isdir(path_rotors / file_name):
            if int(
                input(
                    "There is a rotor with this file_name, do you want to overwrite it? (1 for yes and 0 for no)"
                )
            ):
                shutil.rmtree(path_rotors / file_name)
            else:
                return "The rotor was not saved."

        os.chdir(path_rotors)
        current = Path(".")

        os.mkdir(file_name)
        os.chdir(current / file_name)

        with open("properties.toml", "w") as f:
            toml.dump({"parameters": self.parameters}, f)
        os.mkdir("results")
        os.mkdir("elements")
        current = Path(".")

        os.chdir(current / "elements")

        for element in self.elements:
            element.save(type(element).__name__ + ".toml")

        os.chdir(main_path)

    @staticmethod
    def load(file_name):
        """Load rotor from toml file.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        rotor : ross.rotor.Rotor
        """
        main_path = os.path.dirname(ross.__file__)
        rotor_path = Path(main_path) / "rotors" / file_name
        try:
            os.chdir(rotor_path / "elements")
        except FileNotFoundError:
            return "A rotor with this name does not exist, check the rotors folder."

        os.chdir(rotor_path / "elements")
        shaft_elements = ShaftElement.load()
        os.chdir(rotor_path / "elements")
        disk_elements = DiskElement.load()
        bearing_elements = BearingElement.load()
        seal_elements = []

        os.chdir(rotor_path)
        with open("properties.toml", "r") as f:
            parameters = toml.load(f)["parameters"]

        os.chdir(main_path)
        return Rotor(
            shaft_elements=shaft_elements,
            bearing_seal_elements=bearing_elements + seal_elements,
            disk_elements=disk_elements,
            **parameters,
        )

    @staticmethod
    def available_rotors():
        return [x for x in os.listdir(Path(os.path.dirname(ross.__file__)) / "rotors")]

    @staticmethod
    def remove(rotor_name):
        shutil.rmtree(Path(os.path.dirname(ross.__file__)) / "rotors" / rotor_name)

    def run_static(self, output_html=False):
        """
        Static analysis calculates free-body diagram, deformed shaft, shearing
        force diagram and bending moment diagram.

        Parameters
        ----------
        output_html : Boolean, optional
            outputs a html file.
            Default is False

        Returns
        -------
            grid_plots : bokeh.gridplot

        """
        # gravity aceleration vector
        grav = np.zeros((len(self.M()), 1))

        # place gravity effect on shaft and disks nodes
        for node_y in range(int(len(self.M()) / 4)):
            grav[4 * node_y + 1] = -9.8065

        # calculates x, for [K]*(x) = [M]*(g)
        disp = (la.solve(self.K(0), self.M() @ grav)).flatten()

        # calculates displacement values in gravity's direction
        # dof = degree of freedom
        disp_y = np.array([])
        for node_dof in range(int(len(disp) / 4)):
            disp_y = np.append(disp_y, disp[4 * node_dof + 1])
        self.disp_y = disp_y

        # Shearing Force
        BrgForce = [0] * len(self.nodes_pos)
        DskForce = [0] * len(self.nodes_pos)
        SchForce = [0] * len(self.nodes_pos)

        for i, node in enumerate(self.df_bearings["n"]):
            BrgForce[node] = (
                -disp_y[node] * self.df_bearings.loc[i, "kyy"].coefficient[0]
            )

        for i, node in enumerate(self.df_disks["n"]):
            DskForce[node] = self.df_disks.loc[i, "m"] * -9.8065

        for i, node in enumerate(self.df_shaft["_n"]):
            SchForce[node + 1] = self.df_shaft.loc[i, "m"] * -9.8065

        # Shearing Force vector
        Vx = [0] * (len(self.nodes_pos))
        Vx_axis = []
        for i in range(int(len(self.nodes))):
            Vx_axis.append(self.nodes_pos[i])
            Vx[i] = Vx[i - 1] + BrgForce[i] + DskForce[i] + SchForce[i]

        for i in range(len(Vx) + len(self.df_disks) + len(self.df_bearings)):
            if DskForce[i] != 0:
                Vx.insert(i, Vx[i - 1] + SchForce[i])
                DskForce.insert(i + 1, 0)
                SchForce.insert(i + 1, 0)
                BrgForce.insert(i + 1, 0)
                Vx_axis.insert(i, Vx_axis[i])

            if BrgForce[i] != 0:
                Vx.insert(i, Vx[i - 1] + SchForce[i])
                BrgForce.insert(i + 1, 0)
                DskForce.insert(i + 1, 0)
                SchForce.insert(i + 1, 0)
                Vx_axis.insert(i, Vx_axis[i])
        self.Vx = [x * -1 for x in Vx]
        Vx = self.Vx

        # Bending Moment vector
        Mx = []
        for i in range(len(Vx) - 1):
            if Vx_axis[i] == Vx_axis[i + 1]:
                pass
            else:
                Mx.append(
                    (
                        (Vx_axis[i + 1] * Vx[i + 1])
                        + (Vx_axis[i + 1] * Vx[i])
                        - (Vx_axis[i] * Vx[i + 1])
                        - (Vx_axis[i] * Vx[i])
                    )
                    / 2
                )
        Bm = [0]
        for i in range(len(Mx)):
            Bm.append(Bm[i] + Mx[i])
        self.Bm = Bm
        if output_html:
            output_file("static.html")
        source = ColumnDataSource(
            data=dict(x0=self.nodes_pos, y0=disp_y * 1000, y1=[0] * len(self.nodes_pos))
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
        for i, node in enumerate(self.df_bearings["n"]):
            y_range.append(
                -disp_y[node] * self.df_bearings.loc[i, "kyy"].coefficient[0]
            )

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

        FBD.line("x0", "y1", source=source, line_width=5, line_color=bokeh_colors[0])

        # FBD - plot arrows indicating shaft weight distribution
        sh_weight = sum(self.df_shaft["m"].values) * 9.8065
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
            Fb = -disp_y[node] * self.df_bearings.loc[i, "kyy"].coefficient[0]
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
        source_SF = ColumnDataSource(data=dict(x=Vx_axis, y=Vx))
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
        source_BM = ColumnDataSource(data=dict(x=self.nodes_pos, y=Bm))
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
        i = 0
        while True:
            if i + 3 > len(self.nodes):
                break

            interpolated_BM = interpolate.interp1d(
                self.nodes_pos[i : i + 3], Bm[i : i + 3], kind="quadratic"
            )
            xnew_BM = np.linspace(
                self.nodes_pos[i], self.nodes_pos[i + 2], num=42, endpoint=True
            )

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

        return grid_plots

    @classmethod
    def from_section(
        cls,
        leng_data,
        o_ds_data,
        i_ds_data,
        disk_data=None,
        brg_seal_data=None,
        sparse=True,
        min_w=None,
        max_w=None,
        n_eigen=12,
        w=0,
        nel_r=1,
    ):

        """This class is an alternative to build rotors from separated
        sections. Each section has the same number (n) of shaft elements.

        Parameters
        ----------
        leng_data : list
            List with the lengths of rotor regions.
        o_d_data : list
            List with the outer diameters of rotor regions.
        i_d_data : list
            List with the inner diameters of rotor regions.
        disk_data : dict, optional
            Dict holding disks datas.
            Example : disk_data=DiskElement.from_geometry(n=2,
                                                          material=steel,
                                                          width=0.07,
                                                          i_d=0,
                                                          o_d=0.28
                                                          )
            ***See 'disk_element.py' docstring for more information***
        brg_seal_data : dict, optional
            Dict holding lists of bearings and seals datas.
            Example : brg_seal_data=BearingElement(n=1, kxx=1e6, cxx=0,
                                                   kyy=1e6, cyy=0, kxy=0,
                                                   cxy=0, kyx=0, cyx=0)
            ***See 'bearing_seal_element.py' docstring for more information***
        w : float, optional
            Rotor speed.
        nel_r : int, optional
            Number or elements per shaft region.
            Default is 1
        n_eigen : int, optional
            Number of eigenvalues calculated by arpack.
            Default is 12.

        Example
        -------

        >>> rotor = Rotor.from_section(leng_data=[0.5,0.5,0.5],
        ...             o_ds_data=[0.05,0.05,0.05],
        ...             i_ds_data=[0,0,0],
        ...             disk_data=[DiskElement.from_geometry(n=1, material=steel, width=0.07, i_d=0, o_d=0.28),
        ...                        DiskElement.from_geometry(n=2, material=steel, width=0.07, i_d=0, o_d=0.35)],
        ...             brg_seal_data=[BearingElement(n=0, kxx=1e6, cxx=0, kyy=1e6, cyy=0, kxy=0, cxy=0, kyx=0, cyx=0),
        ...                            BearingElement(n=3, kxx=1e6, cxx=0, kyy=1e6, cyy=0, kxy=0, cxy=0, kyx=0, cyx=0)],
        ...             w=0, nel_r=1)
        >>> rotor.run_modal()
        >>> rotor.wn.round(4)
        array([ 85.7634,  85.7634, 271.9326, 271.9326, 718.58  , 718.58  ])

        """

        if len(leng_data) != len(o_ds_data) or len(leng_data) != len(i_ds_data):
            raise ValueError("The matrices lenght do not match.")

        def rotor_regions(nel_r):

            regions = []
            shaft_elements = []
            disk_elements = []
            bearing_seal_elements = []
            # nel_r = initial number of elements per regions

            # loop through rotor regions
            for i, leng in enumerate(leng_data):

                le = leng / nel_r
                o_ds = o_ds_data[i]
                i_ds = i_ds_data[i]

                # loop to generate n elements per region
                for j in range(nel_r):
                    shaft_elements.append(
                        ShaftElement(
                            le,
                            i_ds,
                            o_ds,
                            material=steel,
                            shear_effects=True,
                            rotary_inertia=True,
                            gyroscopic=True,
                        )
                    )

            regions.extend([shaft_elements])

            for DiskEl in disk_data:
                aux_DiskEl = deepcopy(DiskEl)
                aux_DiskEl.n = nel_r * DiskEl.n
                disk_elements.append(aux_DiskEl)

            for Brg_SealEl in brg_seal_data:
                aux_Brg_SealEl = deepcopy(Brg_SealEl)
                aux_Brg_SealEl.n = nel_r * Brg_SealEl.n
                bearing_seal_elements.append(aux_Brg_SealEl)

            regions.append(disk_elements)
            regions.append(bearing_seal_elements)

            return regions

        regions = rotor_regions(nel_r)
        shaft_elements = regions[0]
        disk_elements = regions[1]
        bearing_seal_elements = regions[2]

        return cls(
            shaft_elements,
            disk_elements,
            bearing_seal_elements,
            w=w,
            sparse=sparse,
            n_eigen=n_eigen,
            min_w=min_w,
            max_w=max_w,
            rated_w=None,
        )


def rotor_example():
    """This function returns an instance of a simple rotor with
    two shaft elements, one disk and two simple bearings.
    The purpose of this is to make available a simple model
    so that doctest can be written using this.

    Parameters
    ----------

    Returns
    -------
    An instance of a rotor object.

    Examples
    --------
    >>> rotor = rotor_example()
    >>> np.round(rotor.wd[:4])
    array([ 92.,  96., 275., 297.])
    """
    #  Rotor without damping with 6 shaft elements 2 disks and 2 bearings
    i_d = 0
    o_d = 0.05
    n = 6
    L = [0.25 for _ in range(n)]

    shaft_elem = [
        ShaftElement(
            l, i_d, o_d, steel, shear_effects=True, rotary_inertia=True, gyroscopic=True
        )
        for l in L
    ]

    disk0 = DiskElement.from_geometry(
        n=2, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )
    disk1 = DiskElement.from_geometry(
        n=4, material=steel, width=0.07, i_d=0.05, o_d=0.28
    )

    stfx = 1e6
    stfy = 0.8e6
    bearing0 = BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    bearing1 = BearingElement(6, kxx=stfx, kyy=stfy, cxx=0)

    return Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])


def MAC(u, v):
    """MAC for two vectors"""
    H = lambda a: a.T.conj()
    return np.absolute((H(u) @ v) ** 2 / ((H(u) @ u) * (H(v) @ v)))


def MAC_modes(U, V, n=None, plot=True):
    """MAC for multiple vectors"""
    # n is the number of modes to be evaluated
    if n is None:
        n = U.shape[1]
    macs = np.zeros((n, n))
    for u in enumerate(U.T[:n]):
        for v in enumerate(V.T[:n]):
            macs[u[0], v[0]] = MAC(u[1], v[1])

    if not plot:
        return macs

    xpos, ypos = np.meshgrid(range(n), range(n))
    xpos, ypos = 0.5 + xpos.flatten(), 0.5 + ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = 0.75 * np.ones_like(xpos)
    dy = 0.75 * np.ones_like(xpos)
    dz = macs.T.flatten()

    fig = plt.figure(figsize=(12, 8))
    # fig.suptitle('MAC - %s vs %s' % (U.name, V.name), fontsize=12)
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        xpos, ypos, zpos, dx, dy, dz, color=plt.cm.viridis(dz), alpha=0.7, zsort="max"
    )
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_zlim(0, 1)
    # ax.set_xlabel('%s  modes' % U.name)
    # ax.set_ylabel('%s  modes' % V.name)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    # fake up the array of the scalar mappable
    sm._A = []
    cbar = fig.colorbar(sm, shrink=0.5, aspect=10)
    cbar.set_label("MAC")

    return macs


def whirl(kappa_mode):
    """Evaluates the whirl of a mode"""
    if all(kappa >= -1e-3 for kappa in kappa_mode):
        whirldir = "Forward"
    elif all(kappa <= 1e-3 for kappa in kappa_mode):
        whirldir = "Backward"
    else:
        whirldir = "Mixed"
    return whirldir


@np.vectorize
def whirl_to_cmap(whirl):
    """Maps the whirl to a value"""
    if whirl == "Forward":
        return 0.
    elif whirl == "Backward":
        return 1.
    elif whirl == "Mixed":
        return 0.5
