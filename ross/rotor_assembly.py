import os
import warnings
import pickle
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse.linalg as las
import scipy.signal as signal
import scipy.io as sio
from copy import copy
from collections import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from ross.bearing_seal_element import BearingElement
from ross.disk_element import DiskElement
from ross.shaft_element import ShaftElement
from ross.materials import steel
from ross.results import (
    CampbellResults,
    FrequencyResponseResults,
    ForcedResponseResults,
    ModeShapeResults,
)


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
    >>> from ross.materials import steel
    >>> z = 0
    >>> le = 0.25
    >>> i_d = 0
    >>> o_d = 0.05
    >>> tim0 = ShaftElement(le, i_d, o_d, steel,
    ...                    shear_effects=True,
    ...                    rotary_inertia=True,
    ...                    gyroscopic=True)
    >>> tim1 = ShaftElement(le, i_d, o_d, steel,
    ...                    shear_effects=True,
    ...                    rotary_inertia=True,
    ...                    gyroscopic=True)
    >>> shaft_elm = [tim0, tim1]
    >>> disk0 = DiskElement(1, steel, 0.07, 0.05, 0.28)
    >>> stf = 1e6
    >>> bearing0 = BearingElement(0, kxx=stf, cxx=0)
    >>> bearing1 = BearingElement(2, kxx=stf, cxx=0)
    >>> rotor = Rotor(shaft_elm, [disk0], [bearing0, bearing1])
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

        df = pd.concat([df_shaft, df_disks, df_bearings])
        df = df.sort_values(by="n_l")
        df = df.reset_index(drop=True)

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

        self.nodes = list(range(len(self.nodes_pos)))
        self.elements_length = self.df.groupby("n_l")["L"].max()
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

        #  diameter at node position

        # call self._calc_system() to calculate current evalues and evectors
        self._calc_system()

    def _calc_system(self):
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

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value
        self._calc_system()

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
        array([[ 47.,   0.,   0.,   6.],
               [  0.,  46.,  -6.,   0.],
               [  0.,  -6.,   1.,   0.],
               [  6.,   0.,   0.,   1.]])
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
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]])
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
        array([[     0.,  11110.],
               [-11106.,     -0.],
               [  -169.,     -0.],
               [    -0.,   -169.],
               [    -0.,  10511.],
               [-10507.,     -0.]])
        """
        if w is None:
            w = self.w

        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)

        # fmt: off
        A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-self.M(), self.K(w)), la.solve(-self.M(), (self.C(w) + self.G()*w))])])
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
        >>> evalues, evectors = rotor._eigen(0, sorted_=False)
        >>> idx = rotor._index(evalues)
        >>> idx[:6] # doctest: +ELLIPSIS
        array([ 1,  3,  5,  7,  9, 11]...
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
        82.653...
        """
        if w is None:
            w = self.w
        if A is None:
            A = self.A(w)

        if self.sparse is True:
            try:
                evalues, evectors = las.eigs(
                    A, k=self.n_eigen, sigma=0, ncv=24, which="LM", v0=self._v0
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
        >>> rotor.H_kappa(0, 0) # doctest: +ELLIPSIS
        array([[  8.78547006e-30,  -4.30647963e-18],
               [ -4.30647963e-18,   2.11429917e-06]])


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
        0.00145...
        >>> # kappa for node 2 and natural frequency (mode) 3.
        >>> rotor.kappa(2, 3)['kappa'] # doctest: +ELLIPSIS
        8.539...e-14
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
        [-0.0, -0.0, -0.0, -0.0, -1.153...e-08, -0.0, -1.239...e-08]
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

    def freq_response(self, frequency_range=None, modes=None):
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

    def forced_response(self, force=None, frequency_range=None, modes=None):
        freq_resp = self.freq_response(frequency_range=frequency_range, modes=modes)

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

        forced_response = self.forced_response(force, frequency_range)

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

    def plot_rotor(self, nodes=1, ax=None):
        """Plots a rotor object.

        This function will take a rotor object and plot its shaft,
        disks and bearing elements

        Parameters
        ----------
        nodes : int, optional
            Increment that will be used to plot nodes label.
        ax : matplotlib axes, optional
            Axes in which the plot will be drawn.

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.

        Examples:
        """
        if ax is None:
            ax = plt.gca()

        #  plot shaft centerline
        shaft_end = self.nodes_pos[-1]
        ax.plot([-0.2 * shaft_end, 1.2 * shaft_end], [0, 0], "k-.")
        try:
            max_diameter = max([disk.o_d for disk in self.disk_elements])
        except (ValueError, AttributeError):
            max_diameter = max([shaft.o_d for shaft in self.shaft_elements])

        ax.set_ylim(-1.2 * max_diameter, 1.2 * max_diameter)
        ax.axis("equal")
        ax.set_xlabel("Axial location (m)")
        ax.set_ylabel("Shaft radius (m)")

        #  plot nodes
        for node, position in enumerate(self.nodes_pos[::nodes]):
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

        # plot shaft elements
        for sh_elm in self.shaft_elements:
            position = self.nodes_pos[sh_elm.n]
            sh_elm.patch(ax, position)

        # plot disk elements
        for disk in self.disk_elements:
            position = (self.nodes_pos[disk.n], self.nodes_o_d[disk.n])
            disk.patch(ax, position)

        # plot bearings
        for bearing in self.bearing_seal_elements:
            position = (self.nodes_pos[bearing.n], -self.nodes_o_d[bearing.n])
            bearing.patch(ax, position)

        return ax

    def campbell(self, speed_range, frequencies=6, frequency_type="wd"):
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
        >>> camp = rotor1.campbell(speed)
        >>> np.round(camp[:, 0], 1) #  damped natural frequencies at the first rotor speed (0 rad/s)
        array([  82.7,   86.7,  254.5,  274.3,  679.5,  716.8])
        >>> np.round(camp[:, 10], 1) # damped natural frequencies at 40 rad/s
        array([  82.6,   86.7,  254.3,  274.5,  676.5,  719.7])
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

    def mode_shapes(self):

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

    def plot_ucs(self, stiffness_range=None, num=20, ax=None):
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

        Returns
        -------
        ax : matplotlib axes
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

        return ax

    def plot_level1(self, n=None, stiffness_range=None, num=5, ax=None, **kwargs):
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

        Returns
        -------
        ax : matplotlib axes
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

        return ax

    def plot_time_response(self, F, t, dof, ax=None):
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

        Returns
        -------
        ax : matplotlib axes
            Returns the axes object with the plot.

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

        return ax

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
        """Save rotor to binary file.

        Parameters
        ----------
        file_name : str
        """
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        """Load rotor from binary file.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        rotor : ross.rotor.Rotor
        """
        with open(file_name, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_section(
        cls,
        leng_data=list,
        o_ds_data=list,
        i_ds_data=list,
        disk_data=None,
        brg_seal_data=None,
        w=0,
        nel_r=1,
        n_eigval=1,
        err_max=1e-02,
    ):

        """This class is an alternative to build rotors from separated
        sections. Each section has the same number (n) of shaft elements.

        This class will verify the eigenvalues calculation
        and check its convergence to minimize the numerical errors.

        Parameters
        ----------
        leng_data : list
            List with the lengths of rotor regions.
        o_d_data : list
            List with the outer diameters of rotor regions.
        i_d_data : list
            List with the inner diameters of rotor regions.
        disk_data : list, optional
            List holding lists of disks datas.
            Example : disk_data = [[n, material, width, i_d, o_d], [n, ...]]
            ***See 'disk_element.py' docstring for more information***
        brg_seal_data : list, optional
            list holding lists of bearings and seals datas.
            Example : brg_seal_data=[[n, kxx, cxx, kyy=None, kxy=0, kyx=0,
                                      cyy=None, cxy=0, cyx=0, w=None],
                                     [n, ...]]
            ***See 'bearing_seal_element.py' docstring for more information***
        w : float, optional
            Rotor speed.
        nel_r : int
            Initial number or elements per shaft region.
            Default is 1
        eigval : int
            Indicates which eingenvalue convergence to check.
            default is 1 (1st eigenvalue).
        err_max : float, optional
            maximum allowed for eigenvalues calculation.
            default is 0.01 (or 1%).

        Example
        -------
        >>> rotor = Rotor.from_section(leng_data=[0.5,0.5,0.5],
        ...             o_ds_data=[0.05,0.05,0.05],
        ...             i_ds_data=[0,0,0],
        ...             disk_data=[[1, steel, 0.07, 0, 0.28],
        ...                        [2, steel, 0.07, 0, 0.35]],
        ...             brg_seal_data=[[0, 1e6, 0, 1e6, 0,0,0,0,0,None],
        ...                            [3, 1e6, 0, 1e6,0,0,0,0,0,None]],
        ...             w=0, nel_r=1, n_eigval=1, err_max=1e-07)
        >>> rotor.wn[:]
        array([ 85.76222593,  85.76222594, 271.86711771, 271.86711774,
               716.27524675, 716.27524696])
        """

        if len(leng_data) != len(o_ds_data) or len(leng_data) != len(i_ds_data):
            raise ValueError("The matrices lenght do not match.")

        def rotor_regions(nel_r=1):

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

            for i, leng in enumerate(leng_data):
                for j, disk in enumerate(disk_data):
                    if disk_data is not None and len(disk) == 5 and i == disk[0]:
                        disk_elements.append(
                            DiskElement.from_geometry(
                                n=nel_r * disk[0],
                                material=disk[1],
                                width=disk[2],
                                i_d=disk[3],
                                o_d=disk[4],
                            )
                        )

            for i, leng in enumerate(leng_data):
                for j, disk in enumerate(disk_data):
                    if disk_data is not None and len(disk) == 4 and i == disk[0]:
                        disk_elements.append(
                            DiskElement(
                                n=nel_r * disk[0], m=disk[1], Id=disk[2], Ip=disk[3]
                            )
                        )

            for i in range(len(leng_data) + 1):
                for j, brg in enumerate(brg_seal_data):
                    if brg_seal_data is not None and i == brg[0]:
                        bearing_seal_elements.append(
                            BearingElement(
                                n=i * nel_r,
                                kxx=brg[1],
                                cxx=brg[2],
                                kyy=brg[3],
                                kxy=brg[4],
                                kyx=brg[5],
                                cyy=brg[6],
                                cxy=brg[7],
                                cyx=brg[8],
                                w=brg[9],
                            )
                        )

            regions.append(disk_elements)
            regions.append(bearing_seal_elements)

            return regions

        el_num = np.array([nel_r * len(leng_data)])
        eigv_arr = np.array([])
        error_arr = np.array([0])

        regions0 = rotor_regions(nel_r)
        rotor0 = Rotor(regions0[0], regions0[1], regions0[2], w=w, n_eigen=12)

        eigv_arr = np.append(eigv_arr, rotor0.wn[n_eigval])
        # this value is up to start the loop while
        error = 1
        nel_r = nel_r * 2

        while error > err_max:

            regions = rotor_regions(nel_r)
            rotor = Rotor(regions[0], regions[1], regions[2], w=w, n_eigen=12)

            eigv_arr = np.append(eigv_arr, rotor.wn[n_eigval])
            el_num = np.append(el_num, nel_r * len(leng_data))

            error = min(eigv_arr[-1], eigv_arr[-2]) / max(eigv_arr[-1], eigv_arr[-2])
            error = 1 - error
            error_arr = np.append(error_arr, 100 * error)

            nel_r *= 2

        shaft_elements = regions[0]
        disk_elements = regions[1]
        bearing_seal_elements = regions[2]

        return cls(
            shaft_elements,
            disk_elements,
            bearing_seal_elements,
            w=0,
            sparse=True,
            n_eigen=12,
            min_w=None,
            max_w=None,
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
    array([  83.,   87.,  255.,  274.])
    """
    #  Rotor without damping with 2 shaft elements 1 disk and 2 bearings
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

    disk0 = DiskElement(2, steel, 0.07, 0.05, 0.28)
    disk1 = DiskElement(4, steel, 0.07, 0.05, 0.35)

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
        return 0
    elif whirl == "Backward":
        return 1
    else:
        return 0.5
