from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg as la

import ross as rs
from ross.units import Q_, check_units

from .abs_fault import Fault

__all__ = [
    "Crack",
]


class Crack(Fault):
    """Contains a :cite:`gasch1993survey` and :cite:`mayes1984analysis` transversal crack models for applications on
    finite element models of rotative machinery.
    The reference coordenates system is: z-axis throught the shaft center; x-axis and y-axis in the sensors' planes
    Calculates the dynamic forces of a crack on a given shaft element.

    Parameters
    ----------
    dt : float
        Time step
    tI : float
        Initial time
    tF : float
        Final time
    depth_ratio : float
        Crack depth ratio related to the diameter of the crack container element. A depth value of 0.1 is equal to 10%,
        0.2 equal to 20%, and so on. This parameter is restricted to up to 50% within the implemented approach,
        as discussed in :cite `papadopoulos2004some`.
    n_crack : float
        Element where the crack is located
    speed : float, pint.Quantity
        Operational speed of the machine. Default unit is rad/s.
    unbalance_magnitude : array
        Array with the unbalance magnitude. The unit is kg.m.
    unbalance_phase : array
        Array with the unbalance phase. The unit is rad.
    crack_type : string
        String containing type of crack model chosed. The avaible types are: Mayes and Gasch.
    print_progress : bool
        Set it True, to print the time iterations and the total time spent, by default False.

    Returns
    -------
    A force to be applied on the shaft.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> from ross.probe import Probe
    >>> from ross.faults.crack import crack_example
    >>> probe1 = Probe(14, 0)
    >>> probe2 = Probe(22, 0)
    >>> response = crack_example()
    >>> results = response.run_time_response()
    >>> fig = response.plot_dfft(probe=[probe1, probe2], range_freq=[0, 100], yaxis_type="log")
    >>> # fig.show()
    """

    @check_units
    def __init__(
        self,
        rotor,
        dt,
        tI,
        tF,
        depth_ratio,
        n_crack,
        speed,
        unbalance_magnitude,
        unbalance_phase,
        crack_type="Mayes",
        print_progress=False,
    ):
        self.dt = dt
        self.tI = tI
        self.tF = tF
        self.n_crack = n_crack
        self.speed = speed
        self.speedI = speed
        self.speedF = speed
        self.unbalance_magnitude = unbalance_magnitude
        self.unbalance_phase = unbalance_phase
        self.print_progress = print_progress

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

        if crack_type is None or crack_type == "Mayes":
            self.crack_model = self._mayes
        elif crack_type == "Gasch":
            self.crack_model = self._gasch
        else:
            raise Exception("Check the crack model!")

        if len(self.unbalance_magnitude) != len(self.unbalance_phase):
            raise Exception(
                "The unbalance magnitude vector and phase must have the same size!"
            )

        dir_path = Path(__file__).parents[0] / "data/PAPADOPOULOS.csv"
        self.coefficient_data = pd.read_csv(dir_path)

        self.rotor = rotor

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

    def run(self):
        """Calculates the shaft angular position and the unbalance forces at X / Y directions."""

        #####################

        self.n_disk = len(self.rotor.disk_elements)
        if self.n_disk != len(self.unbalance_magnitude):
            raise Exception("The number of discs and unbalances must agree!")

        self.ndofd = np.zeros(len(self.rotor.disk_elements))
        for ii in range(self.n_disk):
            self.ndofd[ii] = (self.rotor.disk_elements[ii].n) * 6

        # parameters for the time integration
        self.lambdat = 0.00001

        # pre-processing of auxilary variuables for the time integration
        self.sA = (
            self.speedI * np.exp(-self.lambdat * self.tF)
            - self.speedF * np.exp(-self.lambdat * self.tI)
        ) / (np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI))
        self.sB = (self.speedF - self.speedI) / (
            np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI)
        )

        t = np.arange(self.tI, self.tF + self.dt, self.dt)

        self.angular_position = (
            self.sA * t
            - (self.sB / self.lambdat) * np.exp(-self.lambdat * t)
            + (self.sB / self.lambdat)
        )

        self.Omega = self.sA + self.sB * np.exp(-self.lambdat * t)
        self.AccelV = -self.lambdat * self.sB * np.exp(-self.lambdat * t)

        self.tetaUNB = np.zeros((len(self.unbalance_phase), len(self.angular_position)))
        unbx = np.zeros(len(self.angular_position))
        unby = np.zeros(len(self.angular_position))

        FFunb = np.zeros((self.rotor.ndof, len(t)))
        self.forces = np.zeros((self.rotor.ndof, len(t)))

        # Unbalance force
        for ii in range(self.n_disk):
            self.tetaUNB[ii, :] = (
                self.angular_position + self.unbalance_phase[ii] + np.pi / 2
            )

            unbx = self.unbalance_magnitude[ii] * (self.AccelV) * (
                np.cos(self.tetaUNB[ii, :])
            ) - self.unbalance_magnitude[ii] * (self.Omega**2) * (
                np.sin(self.tetaUNB[ii, :])
            )

            unby = -self.unbalance_magnitude[ii] * (self.AccelV) * (
                np.sin(self.tetaUNB[ii, :])
            ) - self.unbalance_magnitude[ii] * (self.Omega**2) * (
                np.cos(self.tetaUNB[ii, :])
            )

            FFunb[int(self.ndofd[ii]), :] += unbx
            FFunb[int(self.ndofd[ii] + 1), :] += unby

        # Weight force
        g = np.zeros(self.rotor.ndof)
        g[1::6] = -9.81
        M = self.rotor.M()

        for i in range(FFunb.shape[1]):
            FFunb[:, i] += M @ g

        crack_force = lambda step, **state: self._force_in_time(
            step, state.get("disp_resp")
        )

        results = self.rotor.run_time_response(
            speed=self.Omega,
            F=FFunb.T,
            t=t,
            method="newmark",
            add_to_RHS=crack_force,
            # num_modes=12,
            # **kwargs,
        )

        return results

    def _force_in_time(self, step, disp_resp):
        K_crack = self._cracked_element_stiffness(self.angular_position[step])

        F_crack = np.zeros(self.rotor.ndof)
        F_crack[self.dof_crack] = (self.K_elem - K_crack) @ disp_resp[self.dof_crack]
        self.forces[:, step] = F_crack

        return F_crack

    def _cracked_element_stiffness(self, ap):
        """Stiffness matrix of the shaft element with crack in inertial coordinates.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of the cracked element.
        """

        L = self.shaft_element.L

        Kmodel = self.crack_model(ap)

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

    def _gasch(self, ap):
        """Stiffness matrix of the shaft element with crack in rotating coordinates
        according to the Gasch model.

        Paramenters
        -----------
        ap : float
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
            [(-1) ** i * np.cos((2 * i + 1) * ap) / (2 * i + 1) for i in range(size)]
        )

        ke = kme + (4 / np.pi) * kde * cosine_sum
        kn = kmn + (4 / np.pi) * kdn * cosine_sum

        T_matrix = np.array(
            [
                [np.cos(ap), np.sin(ap)],
                [-np.sin(ap), np.cos(ap)],
            ]
        )

        K = T_matrix.T @ np.array([[ke, 0], [0, kn]]) @ T_matrix

        return K

    def _mayes(self, ap):
        """Stiffness matrix of the shaft element with crack in rotating coordinates
        according to the Mayes model.

        Paramenters
        -----------
        ap : float
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

        ke = 0.5 * (ko + kcx) + 0.5 * (ko - kcx) * np.cos(ap)
        kn = 0.5 * (ko + kcz) + 0.5 * (ko - kcz) * np.cos(ap)

        T_matrix = np.array(
            [
                [np.cos(ap), np.sin(ap)],
                [-np.sin(ap), np.cos(ap)],
            ]
        )

        K = T_matrix.T @ np.array([[ke, 0], [0, kn]]) @ T_matrix

        return K

    def _get_coefficient(self, coeff):
        """Terms os the compliance matrix.

        Paramenters
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

        return c[ind[0], 0] * (1 - Poisson**2) / (E * radius**3)


def crack_example():
    """Create an example to evaluate the influence of transverse cracks in a rotating shaft.

    This function returns an instance of a transversal crack
    fault. The purpose is to make available a simple model so that a
    doctest can be written using it.

    Returns
    -------
    crack : ross.Crack Object
        An instance of a crack model object.

    Examples
    --------
    >>> crack = crack_example()
    >>> crack.speed
    125.66370614359172
    """

    rotor = rs.rotor_example_with_damping()

    crack = rotor.run_crack(
        dt=0.0001,
        tI=0,
        tF=0.5,
        depth_ratio=0.2,
        n_crack=18,
        speed=Q_(1200, "RPM"),
        unbalance_magnitude=np.array([5e-4, 0]),
        unbalance_phase=np.array([-np.pi / 2, 0]),
        crack_type="Mayes",
        print_progress=False,
    )

    return crack
