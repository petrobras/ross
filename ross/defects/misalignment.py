from abc import ABC, abstractmethod

import numpy as np

from .abs_defect import Defect

__all__ = [
    "MisalignmentFlexParallel",
    "MisalignmentFlexAngular",
    "MisalignmentFlexCombined",
    "MisalignmentRigid",
    "Rubbing",
    "CrackGasch",
    "CrackMayes",
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
        self, dt, tI, tF, kd, ks, eCOUPx, eCOUPy, misalignment_angle, TD, TL, n1, speed,
    ):
        self.dt = dt
        self.tI = tI
        self.tF = tF
        self.kd = kd
        self.ks = ks
        self.eCOUPx = eCOUPx
        self.eCOUPy = eCOUPy
        self.misalignment_angle = misalignment_angle
        self.TD = TD
        self.TL = TL
        self.n1 = n1
        self.n2 = n1 + 1
        self.speed = speed

        self.speedI = speed
        self.speedF = speed

    def run(self, Radius, ndof):
        #
        self.ndof = ndof

        self.t = np.arange(self.tI, self.tF + self.dt, self.dt)

        warI = self.speedI * np.pi / 30
        warF = self.speedF * np.pi / 30

        self.tI = self.t[0]
        self.tF = self.t[-1]

        lambdat = 0.00001
        Faxial = 0
        TorqueI = 0
        TorqueF = 0

        sA = (warI * np.exp(-lambdat * self.tF) - warF * np.exp(-lambdat * self.tI)) / (
            np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI)
        )
        sB = (warF - warI) / (np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI))

        sAT = (
            TorqueI * np.exp(-lambdat * self.tF) - TorqueF * np.exp(-lambdat * self.tI)
        ) / (np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI))
        sBT = (TorqueF - TorqueI) / (
            np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI)
        )

        SpeedV = sA + sB * np.exp(-lambdat * self.t)
        TorqueV = sAT + sBT * np.exp(-lambdat * self.t)
        AccelV = -lambdat * sB * np.exp(-lambdat * self.t)

        TetaV = (
            sA * self.t - (sB / lambdat) * np.exp(-lambdat * self.t) + (sB / lambdat)
        )

        angular_position = TetaV[1:]
        self.angular_position = angular_position

        fib = np.arctan(self.eCOUPy / self.eCOUPx)
        self.mi_y = (
            (
                np.sqrt(
                    Radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(fib + angular_position)
                )
                - Radius
            )
            * np.cos(angular_position)
            + (
                np.sqrt(
                    Radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + angular_position)
                )
                - Radius
            )
            * np.cos(2 * np.pi / 3 + angular_position)
            + (
                Radius
                - np.sqrt(
                    Radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    - 2
                    * Radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + angular_position)
                )
            )
            * np.cos(4 * np.pi / 3 + angular_position)
        )

        self.mi_x = (
            (
                np.sqrt(
                    Radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(fib + angular_position)
                )
                - Radius
            )
            * np.sin(angular_position)
            + (
                np.sqrt(
                    Radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + angular_position)
                )
                - Radius
            )
            * np.sin(2 * np.pi / 3 + angular_position)
            + (
                Radius
                - np.sqrt(
                    Radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    - 2
                    * Radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + angular_position)
                )
            )
            * np.sin(4 * np.pi / 3 + angular_position)
        )
        self.C = self.ks * Radius * np.sqrt(2 - 2 * np.cos(self.misalignment_angle))

        return self.force()

    def _parallel(self):
        """Reaction forces of parallel misalignment
        
        Returns
        -------
        F_mis_p(12,n) : numpy.ndarray
            Excitation force caused by the parallel misalignment for a 6DOFs system with 'n' values of angular position  
        """

        F_mis_p = np.zeros((len(self.angular_position) + 1, self.ndof))

        Fpy = self.kd * self.mi_y

        Fpx = self.kd * self.mi_x

        F_mis_p[1:, 0 + 6 * self.n1] = -Fpx
        F_mis_p[1:, 1 + 6 * self.n1] = Fpy
        F_mis_p[1:, 5 + 6 * self.n1] = self.TD
        F_mis_p[1:, 0 + 6 * self.n2] = Fpx
        F_mis_p[1:, 1 + 6 * self.n2] = -Fpy
        F_mis_p[1:, 5 + 6 * self.n2] = self.TL

        return F_mis_p

    def _angular(self):
        """Reaction forces of angular misalignment
        
        Returns
        -------
        F_mis_a(12,n) : numpy.ndarray
            Excitation force caused by the angular misalignment for a 6DOFs system with 'n' values of angular position 
        """
        F_mis_a = np.zeros((len(self.angular_position) + 1, self.ndof))

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

        F_mis_a[1:, 0 + 6 * self.n1] = -Fax
        F_mis_a[1:, 1 + 6 * self.n1] = Fay
        F_mis_a[1:, 5 + 6 * self.n1] = self.TD
        F_mis_a[1:, 0 + 6 * self.n2] = Fax
        F_mis_a[1:, 1 + 6 * self.n2] = -Fay
        F_mis_a[1:, 5 + 6 * self.n2] = self.TL

        return F_mis_a

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
    def force(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        return self._parallel()


class MisalignmentFlexAngular(MisalignmentFlex):
    def force(self):
        return self._angular()


class MisalignmentFlexCombined(MisalignmentFlex):
    def force(self):
        return self._combined()


class MisalignmentRigid(Defect, ABC):
    def __init__(
        self,
        dt,
        tI,
        tF,
        Kcoup_auxI,
        Kcoup_auxF,
        kCOUP,
        eCOUP,
        angANG,
        angPAR,
        yfuture,
        TD,
        TL,
        n1,
        speed,
    ):
        self.self.dt = dt
        self.tI = tI
        self.tF = tF
        self.self.Kcoup_auxI = Kcoup_auxI
        self.self.Kcoup_auxF = Kcoup_auxF
        self.kCOUP = kCOUP
        self.eCOUP = eCOUP
        self.angANG = angANG
        self.angPAR = angPAR
        self.self.yfuture = yfuture
        self.TD = TD
        self.TL = TL
        self.n1 = n1
        self.n2 = n1 + 1
        self.speedI = speed
        self.speedF = speed

    def run(self, ndof):
        self.ndof = ndof
        t = np.arange(self.tI, self.tF + self.dt, self.dt)

        warI = self.speedI * np.pi / 30
        warF = self.speedF * np.pi / 30

        self.tI = t[0]
        self.tF = t[-1]

        lambdat = 0.00001
        Faxial = 0
        TorqueI = 0
        TorqueF = 0

        sA = (warI * np.exp(-lambdat * self.tF) - warF * np.exp(-lambdat * self.tI)) / (
            np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI)
        )
        sB = (warF - warI) / (np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI))

        sAT = (
            TorqueI * np.exp(-lambdat * self.tF) - TorqueF * np.exp(-lambdat * self.tI)
        ) / (np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI))
        sBT = (TorqueF - TorqueI) / (
            np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI)
        )

        SpeedV = sA + sB * np.exp(-lambdat * t)
        TorqueV = sAT + sBT * np.exp(-lambdat * t)
        AccelV = -lambdat * sB * np.exp(-lambdat * t)

        TetaV = sA * t - (sB / lambdat) * np.exp(-lambdat * t) + (sB / lambdat)

        angular_position = TetaV[1:]
        self.angular_position = angular_position

        self.k0 = kCOUP
        self.delta1 = eCOUP
        self.fir = angANG
        self.teta = angPAR

        self.beta = 0

        k_misalignbeta1 = np.array(
            [
                self.k0 * self.Kcoup_auxI * self.delta1 * np.sin(self.beta - self.fir),
                -self.k0 * self.Kcoup_auxI * self.delta1 * np.cos(self.beta - self.fir),
                0,
                0,
                0,
                0,
                -self.k0 * self.Kcoup_auxF * self.delta1 * np.sin(self.beta - self.fir),
                self.k0 * self.Kcoup_auxF * self.delta1 * np.cos(self.beta - self.fir),
                0,
                0,
                0,
                0,
            ]
        )

        K_mis_matrix = np.zeros((12, 12))
        K_mis_matrix[4, :] = k_misalignbeta1
        K_mis_matrix[10, :] = -k_misalignbeta1

        self.DoF = [
            (self.n1 * 6 - 5) - 1,
            (self.n1 * 6 - 3) - 1,
            (self.n1 * 6 - 4) - 1,
            (self.n1 * 6 - 2) - 1,
            (self.n1 * 6 - 0) - 1,
            (self.n1 * 6 - 1) - 1,
            (self.n2 * 6 - 5) - 1,
            (self.n2 * 6 - 3) - 1,
            (self.n2 * 6 - 4) - 1,
            (self.n2 * 6 - 2) - 1,
            (self.n2 * 6 - 0) - 1,
            (self.n2 * 6 - 1) - 1,
        ]
        self.Force_kkmis = K_mis_matrix * self.yfuture[self.DoF]

        self.TD = TD
        self.TL = TL

        return self.force()

    def _parallel(self):
        F_misalign = np.array(
            [
                (
                    -self.k0 * self.delta1 * np.cos(self.beta - self.fir)
                    + self.k0 * self.delta1
                ),
                -self.k0 * self.delta1 * np.sin(self.beta - self.fir),
                0,
                0,
                0,
                self.TD - self.TL,
                (
                    self.k0 * self.delta1 * np.cos(self.beta - self.fir)
                    - self.k0 * self.delta1
                ),
                self.k0 * self.delta1 * np.sin(self.beta - self.fir),
                0,
                0,
                0,
                -(self.TD - self.TL),
            ]
        )

        Fmis = self.Force_kkmis + F_misalign
        FFmis = np.zeros(self.ndof)
        FFmis[self.DoF] = Fmis

        return Fmis, FFmis

    def force(self):
        return self._parallel()
