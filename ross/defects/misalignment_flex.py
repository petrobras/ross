from abc import ABC, abstractmethod
from .abs_defect import Defect
import numpy as np

__all__ = [
    "MisalignmentFlexParallel",
    "MisalignmentFlexAngular",
    "MisalignmentFlexCombined",
]


class MisalignmentFlex(Defect, ABC):
    """Calculates the dynamic reaction force of hexangular flexible coupling induced by 6DOF's rotor parallel and angular misalignment.

    Parameters
    ----------
    Radius : float
        Radius of the shaft
    TetaV :  numpy.ndarray 
        Angular position of the shaft
    kd : float
        Radial stiffness of flexible coupling
    ks : float
        Bending stiffness of flexible coupling
    eCOUPx : float
        Parallel misalignment offset between driving rotor and driven rotor along X direction
    eCOUPz : float
        Parallel misalignment offset between driving rotor and driven rotor along Y direction
    misalignment_angle : float
        Angle of the angular misaligned 
    TD : float
        Driving torque
    TL : float
        Driven torque

    References
    ----------
    .. [1] 'Xia, Y., Pang, J., Yang, L., Zhao, Q., & Yang, X. (2019). Study on vibration response and orbits of misaligned rigid rotors 
    connected by hexangular flexible coupling. Applied Acoustics, 155, 286-296..
    """

    def __init__(
        self,
        dt,
        tI,
        tF,
        kd,
        ks,
        eCOUPx,
        eCOUPy,
        Radius,
        misalignment_angle,
        TD,
        TL,
        n1,
        n2,
        speedI,
        speedF=None,
    ):
        self.n1 = n1
        self.n2 = n2

        #

        t = np.arange(tI, tF + dt, dt)

        if speedF is None:
            speedF = speedI

        warI = speedI * np.pi / 30
        warF = speedF * np.pi / 30

        tI = t[0]
        tF = t[-1]

        lambdat = 0.00001
        Faxial = 0
        TorqueI = 0
        TorqueF = 0

        sA = (warI * np.exp(-lambdat * tF) - warF * np.exp(-lambdat * tI)) / (
            np.exp(-lambdat * tF) - np.exp(-lambdat * tI)
        )
        sB = (warF - warI) / (np.exp(-lambdat * tF) - np.exp(-lambdat * tI))

        sAT = (TorqueI * np.exp(-lambdat * tF) - TorqueF * np.exp(-lambdat * tI)) / (
            np.exp(-lambdat * tF) - np.exp(-lambdat * tI)
        )
        sBT = (TorqueF - TorqueI) / (np.exp(-lambdat * tF) - np.exp(-lambdat * tI))

        SpeedV = sA + sB * np.exp(-lambdat * t)
        TorqueV = sAT + sBT * np.exp(-lambdat * t)
        AccelV = -lambdat * sB * np.exp(-lambdat * t)

        TetaV = sA * t - (sB / lambdat) * np.exp(-lambdat * t) + (sB / lambdat)

        angular_position = TetaV[1:]
        self.angular_position = angular_position

        # Desalinhamento Paralelo
        fib = np.arctan(eCOUPy / eCOUPx)
        self.mi_y = (
            (
                np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.sin(fib + angular_position)
                )
                - Radius
            )
            * np.cos(angular_position)
            + (
                np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + angular_position)
                )
                - Radius
            )
            * np.cos(2 * np.pi / 3 + angular_position)
            + (
                Radius
                - np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    - 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + angular_position)
                )
            )
            * np.cos(4 * np.pi / 3 + angular_position)
        )

        self.mi_x = (
            (
                np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.sin(fib + angular_position)
                )
                - Radius
            )
            * np.sin(angular_position)
            + (
                np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + angular_position)
                )
                - Radius
            )
            * np.sin(2 * np.pi / 3 + angular_position)
            + (
                Radius
                - np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    - 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + angular_position)
                )
            )
            * np.sin(4 * np.pi / 3 + angular_position)
        )
        self.C = ks * Radius * np.sqrt(2 - 2 * np.cos(misalignment_angle))

        self.kd = kd
        self.TD = TD
        self.TL = TL
        self.misalignment_angle = misalignment_angle

    def _parallel(self):
        """Reaction forces of parallel misalignment
        
        Returns
        -------
        F_mis_p(12,n) : numpy.ndarray
            Excitation force caused by the parallel misalignment for a 6DOFs system with 'n' values of angular position  
        """

        F_mis_p = np.zeros((12, len(self.angular_position) + 1))

        Fpy = self.kd * self.mi_y

        Fpx = self.kd * self.mi_x

        F_mis_p[0, 1:] = -Fpx
        F_mis_p[1, 1:] = Fpy
        F_mis_p[5, 1:] = self.TD
        F_mis_p[6, 1:] = Fpx
        F_mis_p[7, 1:] = -Fpy
        F_mis_p[11, 1:] = self.TL

        return F_mis_p.T

    def _angular(self):
        """Reaction forces of angular misalignment
        
        Returns
        -------
        F_mis_a(12,n) : numpy.ndarray
            Excitation force caused by the angular misalignment for a 6DOFs system with 'n' values of angular position 
        """
        F_mis_a = np.zeros((12, len(self.angular_position) + 1))

        # Desalinhamento Angular

        Fay = (
            np.abs(
                self.C * np.sin(self.angular_position) * np.sin(self.misalignment_angle)
            )
            * np.sin(self.angular_position + np.pi)
            + np.abs(
                self.C
                * np.sin(self.angular_position + 2 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.sin(self.angular_position + np.pi + 2 * np.pi / 3)
            + np.abs(
                self.C
                * np.sin(self.angular_position + 4 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.sin(self.angular_position + np.pi + 4 * np.pi / 3)
        )

        Fax = (
            np.abs(
                self.C * np.sin(self.angular_position) * np.sin(self.misalignment_angle)
            )
            * np.cos(self.angular_position + np.pi)
            + np.abs(
                self.C
                * np.sin(self.angular_position + 2 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.cos(self.angular_position + np.pi + 2 * np.pi / 3)
            + np.abs(
                self.C
                * np.sin(self.angular_position + 4 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.cos(self.angular_position + np.pi + 4 * np.pi / 3)
        )

        F_mis_a[0, 1:] = -Fax
        F_mis_a[1, 1:] = Fay
        F_mis_a[5, 1:] = self.TD
        F_mis_a[6, 1:] = Fax
        F_mis_a[7, 1:] = -Fay
        F_mis_a[11, 1:] = self.TL

        return F_mis_a.T

    def _combined(self):
        """Reaction forces of combined (parallel and angular) misalignment
        
        Returns
        -------
        F_misalign(12,n) : numpy.ndarray
            Excitation force caused by the combined misalignment for a 6DOFs system with 'n' values of angular position 
        """
        F_misalign = self._parallel() + self._angular()
        return F_misalign

    @abstractmethod
    def force(self):
        pass


class MisalignmentFlexParallel(MisalignmentFlex):
    @property
    def force(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return self._parallel()


class MisalignmentFlexAngular(MisalignmentFlex):
    @property
    def force(self):
        return self._angular()


class MisalignmentFlexCombined(MisalignmentFlex):
    @property
    def force(self):
        return self._combined()
