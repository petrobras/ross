from abc import ABC, abstractmethod
from .abs_defect import Defect
import numpy as np

__all__ = [
    "MisalignmentRigidParallel",
]


class MisalignmentRigid(Defect, ABC):
    def __init__(
        self,
        dt,
        tI,
        tF,
        Nele,
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
        n2,
        speedI,
        speedF=None,
    ):
        self.n1 = n1
        self.n2 = n2
        self.N_GDL = 6 * (Nele + 1)
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

        self.k0 = kCOUP
        self.delta1 = eCOUP
        self.fir = angANG
        self.teta = angPAR

        self.beta = 0

        k_misalignbeta1 = np.array(
            [
                self.k0 * Kcoup_auxI * self.delta1 * np.sin(self.beta - self.fir),
                -self.k0 * Kcoup_auxI * self.delta1 * np.cos(self.beta - self.fir),
                0,
                0,
                0,
                0,
                -self.k0 * Kcoup_auxF * self.delta1 * np.sin(self.beta - self.fir),
                self.k0 * Kcoup_auxF * self.delta1 * np.cos(self.beta - self.fir),
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
            (self.n1 * 6) - 1,
            (self.n1 * 6 - 1) - 1,
            (self.n2 * 6 - 5) - 1,
            (self.n2 * 6 - 3) - 1,
            (self.n2 * 6 - 4) - 1,
            (self.n2 * 6 - 2) - 1,
            (self.n2 * 6) - 1,
            (self.n2 * 6 - 1) - 1,
        ]
        self.Force_kkmis = K_mis_matrix * yfuture[self.DoF]

        self.TD = TD
        self.TL = TL

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
        FFmis = np.zeros(self.N_GDL)
        FFmis[self.DoF] = Fmis

        return Fmis, FFmis

    @abstractmethod
    def force(self):
        pass


class MisalignmentRigidParallel(MisalignmentRigid):
    @property
    def force(self):
        return self._parallel()

