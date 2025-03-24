from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid as integrate

import ross as rs
from ross.units import Q_, check_units

from .fault import Fault

__all__ = [
    "Crack",
]


class Crack(Fault):
    """Models a crack based on Linear Fracture Mechanics in a given shaft element
    of a rotor system.

    Contains a :cite:`gasch1993survey` and :cite:`mayes1984analysis` transversal
    crack models for applications on finite element models of rotative machinery.
    The reference coordenate system is:
        - x-axis and y-axis in the sensors' planes;
        - z-axis throught the shaft center.

    Parameters
    ----------
    rotor : ross.Rotor
        Rotor object.
    n_crack : float
        Number of shaft element where crack is located.
    depth_ratio : float
        Crack depth ratio related to the diameter of the crack container element.
        A depth value of 0.1 is equal to 10%, 0.2 equal to 20%, and so on.
        This parameter is restricted to up to 50% within the implemented approach,
        as discussed in :cite `papadopoulos2004some`.
    crack_model : string, optional
        Name of the chosen crack model. The avaible types are: "Mayes" and "Gasch".
        Default is "Mayes".

    Returns
    -------
    A crack object.

    Attributes
    ----------
    shaft_element : ross.ShaftElement
        A 6 degrees of freedom shaft element object where crack is located.
    K_elem : np.ndarray
        Stiffness matrix of the shaft element without crack.
    Ko : np.ndarray
        Stiffness of the shaft with the crack closed (equivalent to the shaft without crack).
    Kc : np.ndarray
        Stiffness of the shaft including compliance coefficients according to the crack depth.
    forces : np.ndarray
        Force matrix of shape `(ndof, len(t))` for the crack.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> rotor = rs.rotor_example_with_damping()
    >>> fault = Crack(rotor, n_crack=18, depth_ratio=0.2, crack_model="Gasch")
    >>> fault.shaft_element
    ShaftElement()
    """

    @check_units
    def __init__(
        self,
        rotor,
        n_crack,
        depth_ratio,
        crack_model="Mayes",
    ):
        self.rotor = rotor
        self.n_crack = n_crack

        if depth_ratio <= 0.5:
            self.depth_ratio = depth_ratio
        else:
            raise ValueError(
                """
                The implemented approach is based on Linear Fracture Mechanics.
                For cracks deeper than 50% of diameter, this approach has a singularity and cannot be used.
                This is discussed in Papadopoulos (2004).
                """
            )

        if crack_model is None or crack_model == "Mayes":
            self._crack_model = self.mayes
        elif crack_model == "Gasch":
            self._crack_model = self.gasch
        else:
            raise Exception("Check the crack model!")

        dir_path = Path(__file__).parents[0] / "data/PAPADOPOULOS.csv"
        self.coefficient_data = pd.read_csv(dir_path)

        # Shaft element with crack
        self.shaft_element = [
            elm for elm in rotor.shaft_elements if elm.n == self.n_crack
        ][0]

        self.K_elem = self.shaft_element.K()
        self.dof_crack = list(self.shaft_element.dof_global_index.values())

        L = self.shaft_element.L
        E = self.shaft_element.material.E
        Ie = self.shaft_element.Ie
        phi = self.shaft_element.phi

        co1 = L**3 * (1 + phi / 4) / 3
        co2 = L**2 / 2
        co3 = L

        # fmt: off
        Co = np.array([
            [co1,   0,     0, co2],
            [  0,  co1, -co2,   0],
            [  0, -co2,  co3,   0],
            [co2,    0,    0, co3],
        ]) / (E * Ie)
        # fmt: on

        if self.depth_ratio == 0:
            Cc = Co
        else:
            c44 = self._get_coefficient("c44")
            c55 = self._get_coefficient("c55")
            c45 = self._get_coefficient("c45")

            Cc = Co + np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, c55, c45], [0, 0, c45, c44]]
            )

        self.Ko = np.linalg.pinv(Co)
        self.Kc = np.linalg.pinv(Cc)

    def _get_coefficient(self, coeff):
        """Terms os the compliance matrix.

        Parameters
        -----------
        coeff : string
            Name of the coefficient according to the corresponding direction.

        Returns
        -------
        c : np.ndarray
            Compliance coefficient according to the crack depth.
        """

        Poisson = self.shaft_element.material.Poisson
        E = self.shaft_element.material.E
        radius = self.shaft_element.odl / 2

        c = np.array(pd.eval(self.coefficient_data[coeff]))
        ind = np.where(c[:, 1] >= self.depth_ratio * 2)[0]

        c = c[ind[0], 0] * (1 - Poisson**2) / (E * radius**3)

        return c

    def cracked_element_stiffness(self, ap):
        """Stiffness matrix of the shaft element with crack in inertial coordinates.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of the cracked element.
        """

        L = self.shaft_element.L

        Kmodel = self._crack_model(ap)

        Toxy = np.array([[-1, 0], [-L, -1], [1, 0], [0, 1]])
        kxy = np.array([[Kmodel[0, 0], self.Ko[0, 3]], [self.Ko[3, 0], self.Ko[3, 3]]])
        Koxy = Toxy @ kxy @ Toxy.T

        Toyz = np.array([[-1, 0], [L, -1], [1, 0], [0, 1]])
        kyz = np.array([[Kmodel[1, 1], self.Ko[1, 2]], [self.Ko[2, 1], self.Ko[2, 2]]])
        Koyz = Toyz @ kyz @ Toyz.T

        # fmt: off
        K = np.array([
            [Koxy[0,0],         0,   0,         0, Koxy[0,1],   0, Koxy[0,2],         0,   0,         0, Koxy[0,3],   0],
            [        0, Koyz[0,0],   0, Koyz[0,1],         0,   0,         0, Koyz[0,2],   0, Koyz[0,3],         0,   0],
            [        0,         0,   0,         0,         0,   0,         0,         0,   0,         0,         0,   0],
            [        0, Koyz[1,0],   0, Koyz[1,1],         0,   0,         0, Koyz[1,2],   0, Koyz[1,3],         0,   0],
            [Koxy[1,0],         0,   0,         0, Koxy[1,1],   0, Koxy[1,2],         0,   0,         0, Koxy[1,3],   0],
            [        0,         0,   0,         0,         0,   0,         0,         0,   0,         0,         0,   0],
            [Koxy[2,0],         0,   0,         0, Koxy[2,1],   0, Koxy[2,2],         0,   0,         0, Koxy[2,3],   0],
            [        0, Koyz[2,0],   0, Koyz[2,1],         0,   0,         0, Koyz[2,2],   0, Koyz[2,3],         0,   0],
            [        0,         0,   0,         0,         0,   0,         0,         0,   0,         0,         0,   0],
            [        0, Koyz[3,0],   0, Koyz[3,1],         0,   0,         0, Koyz[3,2],   0, Koyz[3,3],         0,   0],
            [Koxy[3,0],         0,   0,         0, Koxy[3,1],   0, Koxy[3,2],         0,   0,         0, Koxy[3,3],   0],
            [        0,         0,   0,         0,         0,   0,         0,         0,   0,         0,         0,   0]
        ])
        # fmt: on

        return K

    def gasch(self, ang_pos):
        """Stiffness matrix of the shaft element with crack in rotating coordinates
        according to the breathing model of Gasch.

        Paramenters
        -----------
        ang_pos : float
            Angular position of the shaft.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of the cracked element.
        """

        # Gasch
        ko = self.Ko[0, 0]
        kcx = self.Kc[0, 0]
        kcz = self.Kc[1, 1]

        kme = (ko + kcx) / 2
        kmn = (ko + kcz) / 2
        kde = (ko - kcx) / 2
        kdn = (ko - kcz) / 2

        size = 18
        cosine_sum = np.sum(
            [
                (-1) ** i * np.cos((2 * i + 1) * ang_pos) / (2 * i + 1)
                for i in range(size)
            ]
        )

        ke = kme + (4 / np.pi) * kde * cosine_sum
        kn = kmn + (4 / np.pi) * kdn * cosine_sum

        T_matrix = np.array(
            [
                [np.cos(ang_pos), np.sin(ang_pos)],
                [-np.sin(ang_pos), np.cos(ang_pos)],
            ]
        )

        K = T_matrix.T @ np.array([[ke, 0], [0, kn]]) @ T_matrix

        return K

    def mayes(self, ang_pos):
        """Stiffness matrix of the shaft element with crack in rotating coordinates
        according to the breathing model of Mayes.

        Paramenters
        -----------
        ang_pos : float
            Angular position of the shaft.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of the cracked element.
        """

        # Mayes
        ko = self.Ko[0, 0]
        kcx = self.Kc[0, 0]
        kcz = self.Kc[1, 1]

        ke = 0.5 * (ko + kcx) + 0.5 * (ko - kcx) * np.cos(ang_pos)
        kn = 0.5 * (ko + kcz) + 0.5 * (ko - kcz) * np.cos(ang_pos)

        T_matrix = np.array(
            [
                [np.cos(ang_pos), np.sin(ang_pos)],
                [-np.sin(ang_pos), np.cos(ang_pos)],
            ]
        )

        K = T_matrix.T @ np.array([[ke, 0], [0, kn]]) @ T_matrix

        return K

    def _force_in_time(self, step, disp_resp, ang_pos):
        """Calculates the dynamic force related on given time step.

        Paramenters
        -----------
        step : int
            Current time step index.
        disp_resp : np.ndarray
            Displacement response of the system at the current time step.
        ang_pos : float
            Angular position of the shaft at the current time step.

        Returns
        -------
        F_crack : np.ndarray
            Force matrix related to the open crack in the current time step `t[step]`.
        """

        K_crack = self.cracked_element_stiffness(ang_pos)

        F_crack = np.zeros(self.rotor.ndof)
        F_crack[self.dof_crack] = (self.K_elem - K_crack) @ disp_resp[self.dof_crack]
        self.forces[:, step] = F_crack

        return F_crack

    def run(self, node, unb_magnitude, unb_phase, speed, t, **kwargs):
        """Run analysis for the system with crack given an unbalance force.

        System time response is simulated considering weight force.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        unb_magnitude : list, float
            Unbalance magnitude (kg.m).
        unb_phase : list, float
            Unbalance phase (rad).
        speed : float or array_like, pint.Quantity
            Rotor speed.
        t : array
            Time array.
        **kwargs : optional
            Additional keyword arguments can be passed to define the parameters
            of the Newmark method if it is used (e.g. gamma, beta, tol, ...).
            See `ross.utils.newmark` for more details.
            Other keyword arguments can also be passed to be used in numerical
            integration (e.g. num_modes).
            See `Rotor.integrate_system` for more details.

        Returns
        -------
        results : ross.TimeResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.TimeResponseResults`
        """

        rotor = self.rotor

        self.forces = np.zeros((rotor.ndof, len(t)))

        # Unbalance force
        F, ang_pos, _, _ = rotor._unbalance_force_in_time(
            node, unb_magnitude, unb_phase, speed, t
        )

        # Weight force
        g = np.zeros(rotor.ndof)
        g[1::6] = -9.81
        M = rotor.M()

        for i in range(len(t)):
            F[:, i] += M @ g

        force_crack = lambda step, **state: self._force_in_time(
            step, state.get("disp_resp"), ang_pos[step]
        )

        results = rotor.run_time_response(
            speed=speed,
            F=F.T,
            t=t,
            method="newmark",
            add_to_RHS=force_crack,
            **kwargs,
        )

        return results


def crack_example():
    """Create an example to evaluate the influence of transverse cracks in a rotating shaft.

    This function returns time response results of a transversal crack fault. The purpose is
    to make available a simple example so that a doctest can be written using it.

    Returns
    -------
    results : ross.TimeResponseResults
        Results for a shaft with crack.

    Examples
    --------
    >>> results = crack_example()
    """

    rotor = rs.rotor_example_with_damping()

    n1 = rotor.disk_elements[0].n
    n2 = rotor.disk_elements[1].n

    results = rotor.run_crack(
        n_crack=18,
        depth_ratio=0.2,
        node=[n1, n2],
        unbalance_magnitude=[5e-4, 0],
        unbalance_phase=[-np.pi / 2, 0],
        crack_model="Mayes",
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
    )

    return results
