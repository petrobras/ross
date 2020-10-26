import time
from abc import ABC, abstractmethod

import numpy as np
import plotly.graph_objects as go
import scipy as sp
import scipy.integrate
import scipy.linalg

from ross.results import TimeResponseResults
from ross.units import Q_

from .abs_defect import Defect
from .integrate_solver import Integrator

__all__ = [
    "Rubbing",
]


class Rubbing(Defect):
    """Contains a rubbing model for applications on finite element models of rotative machinery.
    The reference coordenates system is: z-axis throught the shaft center; x-axis and y-axis in the sensors' planes 

    Parameters
    ----------
    ShaftRad : numpy.ndarray
        Vector containing the radius of each element.
    Nele : int
        Number of elements. 
    yfuture : numpy.ndarray
        Displacement vector of each element.
    yptfuture : numpy.ndarray
        Velocity vector of each element.
    Omega : float
        Angular velocity.
    deltaRUB : float
        Distance between the housing and shaft surface.
    kRUB : float
        Contact stiffness.
    cRUB : float
        Contact damping.
    miRUB : float
        Friction coefficient.
    posRUB : int
        Node where the rubbing is ocurring.
    torque : bool, optional
        Set it as True to consider the torque provided by the rubbing, by default False.

    References
    ----------
    .. [1] Yamamoto, T., Ishida, Y., &Kirk, R.(2002). Linear and Nonlinear Rotordynamics: A Modern Treatment with Applications, pp. 215-222.

    Examples
    --------
    >>> from rubbing import Rubbing
    >>> import numpy as np
    >>> Nele = 33
    >>> posRUB = 5
    >>> deltaRUB = 2.3e-7
    >>> kRUB = 2e6
    >>> cRUB = 4e2
    >>> miRUB = 0.2
    >>> ShaftRad = np.ones(33)
    >>> ShaftRad *= 0.0095
    >>> yfuture = np.loadtxt("yf.txt")
    >>> yptfuture = np.loadtxt("yptf.txt")
    >>> Omega = 94.2478
    >>> Rubbing = Rubbing(ShaftRad, Nele, yfuture, yptfuture, Omega, deltaRUB, kRUB, cRUB, miRUB, posRUB)
    >>> Force = Rubbing.forces
    """

    def __init__(
        self,
        dt,
        tI,
        tF,
        # ShaftRad,  # passado do rotor
        deltaRUB,
        kRUB,
        cRUB,
        miRUB,
        posRUB,
        speed,
        massunb,
        phaseunb,
        torque=False,
    ):

        self.dt = dt
        self.tI = tI
        self.tF = tF
        self.deltaRUB = deltaRUB
        self.kRUB = kRUB
        self.cRUB = cRUB
        self.miRUB = miRUB
        self.posRUB = posRUB
        self.speedI = speed
        self.speedF = speed
        self.DoF = np.arange((self.posRUB * 6), (self.posRUB * 6 + 6))
        self.torque = torque
        self.MassUnb1 = massunb[0]
        self.MassUnb2 = massunb[1]
        self.PhaseUnb1 = phaseunb[0]
        self.PhaseUnb2 = phaseunb[1]

    def run(self, rotor):

        self.rotor = rotor
        self.ndof = rotor.ndof
        self.iteration = 0
        self.radius = rotor.df_shaft.iloc[self.posRUB].o_d / 2

        self.ndofd1 = (self.rotor.disk_elements[0].n) * 6
        self.ndofd2 = (self.rotor.disk_elements[1].n) * 6

        warI = self.speedI * np.pi / 30
        warF = self.speedF * np.pi / 30

        # self.FFunb = np.zeros(self.ndof)
        self.lambdat = 0.00001
        # Faxial = 0
        # TorqueI = 0
        # TorqueF = 0

        self.sA = (
            warI * np.exp(-self.lambdat * self.tF)
            - warF * np.exp(-self.lambdat * self.tI)
        ) / (np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI))
        self.sB = (warF - warI) / (
            np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI)
        )

        # sAT = (
        #     TorqueI * np.exp(-lambdat * self.tF) - TorqueF * np.exp(-lambdat * self.tI)
        # ) / (np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI))
        # sBT = (TorqueF - TorqueI) / (
        #     np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI)
        # )

        # self.SpeedV = sA + sB * np.exp(-lambdat * t)
        # self.TorqueV = sAT + sBT * np.exp(-lambdat * t)
        # self.AccelV = -lambdat * sB * np.exp(-lambdat * t)

        # Determining the modal matrix
        self.K = self.rotor.K(self.speedI * np.pi / 30)
        self.C = self.rotor.C(self.speedI * np.pi / 30)
        self.G = self.rotor.G()
        self.M = self.rotor.M()
        self.Kst = self.rotor.Kst()

        V1, ModMat = scipy.linalg.eigh(
            self.K,
            self.M,
            # lower=False,
            # type=1,
            driver="gvd",
        )

        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(self.M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(self.C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(self.G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(self.K)).dot(ModMat)
        self.Kstmodal = ((ModMat.T).dot(self.Kst)).dot(ModMat)

        y0 = np.zeros(24)
        t_eval = np.arange(self.tI, self.tF + self.dt, self.dt)
        T = t_eval

        self.angular_position = (
            self.sA * T
            - (self.sB / self.lambdat) * np.exp(-self.lambdat * T)
            + (self.sB / self.lambdat)
        )

        self.Omega = self.sA + self.sB * np.exp(-self.lambdat * T)
        self.AccelV = -self.lambdat * self.sB * np.exp(-self.lambdat * T)

        self.tetaUNB1 = self.angular_position + self.PhaseUnb1 + np.pi / 2
        self.tetaUNB2 = self.angular_position + self.PhaseUnb2 + np.pi / 2

        unb1x = self.MassUnb1 * (
            (self.AccelV) * (np.cos(self.tetaUNB1))
        ) - self.MassUnb1 * ((self.Omega ** 2)) * (np.sin(self.tetaUNB1))

        unb1y = -self.MassUnb1 * (self.AccelV) * (
            np.sin(self.tetaUNB1)
        ) - self.MassUnb1 * (self.Omega ** 2) * (np.cos(self.tetaUNB1))

        unb2x = self.MassUnb2 * (self.AccelV) * (
            np.cos(self.tetaUNB2)
        ) - self.MassUnb2 * (self.Omega ** 2) * (np.sin(self.tetaUNB2))

        unb2y = -self.MassUnb2 * (self.AccelV) * (
            np.sin(self.tetaUNB2)
        ) - self.MassUnb2 * (self.Omega ** 2) * (np.cos(self.tetaUNB2))

        FFunb = np.zeros((self.ndof, len(t_eval)))

        FFunb[self.ndofd1, :] += unb1x
        FFunb[self.ndofd1 + 1, :] += unb1y
        FFunb[self.ndofd2, :] += unb2x
        FFunb[self.ndofd2 + 1, :] += unb2y

        self.Funbmodal = (self.ModMat.T).dot(FFunb)

        self.inv_Mmodal = np.linalg.pinv(self.Mmodal)
        t1 = time.time()

        x = Integrator(self.tI, y0, self.tF, self.dt, self._equation_of_movement)
        x = x.rk4()
        t2 = time.time()
        print(f"Time spent: {t2-t1} s")

        self.displacement = x[:12, :]
        self.velocity = x[12:, :]
        self.time_vector = t_eval
        self.response = self.ModMat.dot(self.displacement)

    def _equation_of_movement(self, T, Y, i):

        positions = Y[:12]
        velocity = Y[12:]  # velocity in space state

        positionsFis = self.ModMat.dot(positions)
        velocityFis = self.ModMat.dot(velocity)

        Frub, ft = self._rub(positionsFis, velocityFis, self.Omega[i])
        ftmodal = (self.ModMat.T).dot(ft)

        # proper equation of movement to be integrated in time
        new_V_dot = (
            ftmodal
            + self.Funbmodal[:, i]
            - ((self.Cmodal + self.Gmodal * self.Omega[i])).dot(velocity)
            - ((self.Kmodal + self.Kstmodal * self.AccelV[i]).dot(positions))
        ).dot(self.inv_Mmodal)

        new_X_dot = velocity

        new_Y = np.zeros(24)
        new_Y[:12] = new_X_dot
        new_Y[12:] = new_V_dot

        return new_Y

    def _rub(self, positionsFis, velocityFis, ang):
        self.F_k = np.zeros(self.ndof)
        self.F_c = np.zeros(self.ndof)
        self.F_f = np.zeros(self.ndof)

        self.y = np.concatenate((positionsFis, velocityFis))

        ii = 0 + 6 * self.posRUB  # rubbing position

        self.radial_displ_node = np.sqrt(
            self.y[ii] ** 2 + self.y[ii + 1] ** 2
        )  # radial displacement
        self.radial_displ_vel_node = np.sqrt(
            self.y[ii + self.ndof] ** 2 + self.y[ii + 1 + self.ndof] ** 2
        )  # velocity
        self.phi_angle = np.arctan2(self.y[ii + 1], self.y[ii])

        if self.radial_displ_node >= self.deltaRUB:
            self.F_k[ii] = self._stiffness_force(self.y[ii])
            self.F_k[ii + 1] = self._stiffness_force(self.y[ii + 1])
            self.F_c[ii] = self._damping_force(self.y[ii + self.ndof])
            self.F_c[ii + 1] = self._damping_force(self.y[ii + 1 + self.ndof])

            Vt = -self.y[ii + self.ndof + 1] * sp.sin(self.phi_angle) + self.y[
                ii + self.ndof
            ] * sp.cos(self.phi_angle)

            if Vt + ang * self.radius > 0:
                self.F_f[ii] = -self._tangential_force(self.F_k[ii], self.F_c[ii])
                self.F_f[ii + 1] = self._tangential_force(
                    self.F_k[ii + 1], self.F_c[ii + 1]
                )

                if self.torque:
                    self.F_f[ii + 5] = self._torque_force(
                        self.F_f[ii], self.F_f[ii + 1], self.y[ii]
                    )
            elif Vt + ang * self.radius < 0:
                self.F_f[ii] = self._tangential_force(self.F_k[ii], self.F_c[ii])
                self.F_f[ii + 1] = -self._tangential_force(
                    self.F_k[ii + 1], self.F_c[ii + 1]
                )

                if self.torque:
                    self.F_f[ii + 5] = self._torque_force(
                        self.F_f[ii], self.F_f[ii + 1], self.y[ii]
                    )

        return self._combine_forces(self.F_k, self.F_c, self.F_f)

    def _stiffness_force(self, y):
        """Calculates the stiffness force

        Parameters
        ----------
        y : float
            Displacement value.

        Returns
        -------
        force : numpy.float64
            Force magnitude.
        """
        force = (
            -self.kRUB
            * (self.radial_displ_node - self.deltaRUB)
            * y
            / abs(self.radial_displ_node)
        )
        return force

    def _damping_force(self, y):
        """Calculates the damping force

        Parameters
        ----------
        y : float
            Displacement value.

        Returns
        -------
        force : numpy.float64
            Force magnitude.
        """
        force = (
            -self.cRUB
            * (self.radial_displ_vel_node)
            * y
            / abs(self.radial_displ_vel_node)
        )
        return force

    def _tangential_force(self, F_k, F_c):
        """Calculates the tangential force

        Parameters
        ----------
        y : float
            Displacement value.

        Returns
        -------
        force : numpy.float64
            Force magnitude.
        """
        force = self.miRUB * (abs(F_k + F_c))
        return force

    def _torque_force(self, F_f, F_fp, y):
        """Calculates the torque force

        Parameters
        ----------
        y : float
            Displacement value.

        Returns
        -------
        force : numpy.float64
            Force magnitude.
        """
        force = self.radius * (
            np.sqrt(F_f ** 2 + F_fp ** 2) * y / abs(self.radial_displ_node)
        )
        return force

    def _combine_forces(self, F_k, F_c, F_f):
        """Mounts the final force vector.

        Parameters
        ----------
        F_k : numpy.ndarray
            Stiffness force vector.
        F_c : numpy.ndarray
            Damping force vector.
        F_f : numpy.ndarray
            Tangential force vector.

        Returns
        -------
        Frub : numpy.ndarray
            Final force vector for each degree of freedom.
        FFrub : numpy.ndarray
            Final force vector.
        """
        Frub = F_k[self.DoF] + F_c[self.DoF] + F_f[self.DoF]
        FFrub = F_k + F_c + F_f

        return Frub, FFrub

    @property
    def forces(self):
        pass
