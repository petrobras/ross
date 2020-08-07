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
    alpha : float
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
        self, TetaV, kd, ks, eCOUPx, eCOUPy, Radius, alpha, TD, TL,
    ):

        beta = TetaV[1:]
        self.beta = beta
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
                    * np.sin(fib + beta)
                )
                - Radius
            )
            * np.cos(beta)
            + (
                np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + beta)
                )
                - Radius
            )
            * np.cos(2 * np.pi / 3 + beta)
            + (
                Radius
                - np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    - 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + beta)
                )
            )
            * np.cos(4 * np.pi / 3 + beta)
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
                    * np.sin(fib + beta)
                )
                - Radius
            )
            * np.sin(beta)
            + (
                np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    + 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + beta)
                )
                - Radius
            )
            * np.sin(2 * np.pi / 3 + beta)
            + (
                Radius
                - np.sqrt(
                    Radius ** 2
                    + eCOUPx ** 2
                    + eCOUPy ** 2
                    - 2
                    * Radius
                    * np.sqrt(eCOUPx ** 2 + eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + beta)
                )
            )
            * np.sin(4 * np.pi / 3 + beta)
        )
        self.C = ks * Radius * np.sqrt(2 - 2 * np.cos(alpha))

        self.kd = kd
        self.TD = TD
        self.TL = TL
        self.alpha = alpha

    def _parallel(self):
        """Reaction forces of parallel misalignment
        
        Returns
        -------
        F_mis_p(12,n) : numpy.ndarray
            Excitation force caused by the parallel misalignment for a 6DOFs system with 'n' values of angular position  
        """

        F_mis_p = np.zeros((12, len(self.beta) + 1))

        Fpy = self.kd * self.mi_y

        Fpx = self.kd * self.mi_x

        F_mis_p[0, 1:] = -Fpx
        F_mis_p[1, 1:] = Fpy
        F_mis_p[5, 1:] = self.TD
        F_mis_p[6, 1:] = Fpx
        F_mis_p[7, 1:] = -Fpy
        F_mis_p[11, 1:] = self.TL

        return F_mis_p

    def _angular(self):
        """Reaction forces of angular misalignment
        
        Returns
        -------
        F_mis_a(12,n) : numpy.ndarray
            Excitation force caused by the angular misalignment for a 6DOFs system with 'n' values of angular position 
        """
        F_mis_a = np.zeros((12, len(self.beta) + 1))

        # Desalinhamento Angular

        Fay = (
            np.abs(self.C * np.sin(self.beta) * np.sin(self.alpha))
            * np.sin(self.beta + np.pi)
            + np.abs(self.C * np.sin(self.beta + 2 * np.pi / 3) * np.sin(self.alpha))
            * np.sin(self.beta + np.pi + 2 * np.pi / 3)
            + np.abs(self.C * np.sin(self.beta + 4 * np.pi / 3) * np.sin(self.alpha))
            * np.sin(self.beta + np.pi + 4 * np.pi / 3)
        )

        Fax = (
            np.abs(self.C * np.sin(self.beta) * np.sin(self.alpha))
            * np.cos(self.beta + np.pi)
            + np.abs(self.C * np.sin(self.beta + 2 * np.pi / 3) * np.sin(self.alpha))
            * np.cos(self.beta + np.pi + 2 * np.pi / 3)
            + np.abs(self.C * np.sin(self.beta + 4 * np.pi / 3) * np.sin(self.alpha))
            * np.cos(self.beta + np.pi + 4 * np.pi / 3)
        )

        F_mis_a[0, 1:] = -Fax
        F_mis_a[1, 1:] = Fay
        F_mis_a[5, 1:] = self.TD
        F_mis_a[6, 1:] = Fax
        F_mis_a[7, 1:] = -Fay
        F_mis_a[11, 1:] = self.TL

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
        return self._parallel()


class MisalignmentFlexAngular(MisalignmentFlex):
    def force(self):
        return self._angular()


class MisalignmentFlexCombined(MisalignmentFlex):
    def force(self):
        return self._combined()


if __name__ == "__main__":

    dt = 0.0001

    time = np.arange(0, 10 + dt, dt)
    speedI = 1200
    speedF = 1200
    lambdat = 0.00001

    warI = speedI * np.pi / 30
    warF = speedF * np.pi / 30

    tI = time[0]
    tF = time[-1]

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

    SpeedV = sA + sB * np.exp(-lambdat * time)
    TorqueV = sAT + sBT * np.exp(-lambdat * time)
    AccelV = -lambdat * sB * np.exp(-lambdat * time)

    TetaV = sA * time - (sB / lambdat) * np.exp(-lambdat * time) + (sB / lambdat)
    # TetaV = np.loadtxt("data/angular_position.txt")

    Radius = (1 / 2) * 19 * 1 * 10 ** (-3)
    coup = 1  # posicao do acoplamento - para correcao na matriz de rigidez
    kCOUP = 5e5  # k3 - rigidez no acoplamento
    nodeI = 1  # no inicial do acoplamento
    nodeF = 2  # no final do acoplamento
    eCOUPx = 2 * 10 ** (-4)  # Distancia de desalinhamento entre os eixos - direcao x
    eCOUPy = 2 * 10 ** (-4)  # Distancia de desalinhamento entre os eixos - direcao z
    kd = 40 * 10 ** (3)  # Rigidez radial do acoplamento flexivel
    ks = 38 * 10 ** (3)  # Rigidez de flexão do acoplamento flexivel
    alpha = 5 * np.pi / 180  # Angulo do desalinhamento angular (rad)
    fib = np.arctan2(eCOUPy, eCOUPx)  # Angulo de rotacao em torno de y;
    TD = 0  # Torque antes do acoplamento
    TL = 0  # Torque dopois do acoplamento
    Nele = 0

    teste1 = MisalignmentFlexParallel(
        TetaV, kd, ks, eCOUPx, eCOUPy, Radius, alpha, TD, TL
    )
    teste2 = MisalignmentFlexAngular(
        TetaV, kd, ks, eCOUPx, eCOUPy, Radius, alpha, TD, TL
    )
    teste3 = MisalignmentFlexCombined(
        TetaV, kd, ks, eCOUPx, eCOUPy, Radius, alpha, TD, TL
    )

    # time = np.arange(0, 10 + 0.0001, 0.0001)

    # matlab = np.loadtxt("test_data/combined_forces5.txt")
    # dt = time[1] - time[0]
    plt.figure
    plt.subplot(311)
    plt.plot(time, teste1.force()[0, ::])
    plt.title("Fx")
    plt.xlim([9, 10])

    plt.subplot(312)
    plt.plot(time, teste2.force()[0, ::])
    plt.title("Fx")
    plt.xlim([9, 10])
    plt.subplot(313)
    plt.plot(time, teste3.force()[0, ::])
    plt.title("Fx")
    plt.xlim([9, 10])
    plt.show()
    print(teste1.force())
