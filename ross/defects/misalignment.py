from abc import ABC, abstractmethod

import numpy as np
import scipy.integrate
import scipy.linalg
import time

from .abs_defect import Defect

__all__ = [
    "MisalignmentFlexParallel",
    "MisalignmentFlexAngular",
    "MisalignmentFlexCombined",
    "MisalignmentRigid",
    # "Rubbing",
    # "CrackGasch",
    # "CrackMayes",
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
        self, tI, tF, Kcoup_auxI, Kcoup_auxF, kCOUP, eCOUP, TD, TL, n1, speed,
    ):
        self.tI = tI
        self.tF = tF
        self.Kcoup_auxI = Kcoup_auxI
        self.Kcoup_auxF = Kcoup_auxF
        self.kCOUP = kCOUP
        self.eCOUP = eCOUP
        self.TD = TD
        self.TL = TL
        self.n1 = n1
        self.n2 = n1 + 1
        self.speedI = speed
        self.speedF = speed

    def run(self, rotor):
        self.rotor = rotor
        self.ndof = rotor.ndof
        self.iteration = 0

        warI = self.speedI * np.pi / 30
        warF = self.speedF * np.pi / 30

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
        K = self.rotor.K(self.speedI * np.pi / 30)
        C = self.rotor.C(self.speedI * np.pi / 30)
        G = self.rotor.G()
        M = self.rotor.M()
        Kst = self.rotor.Kst()

        _, ModMat = scipy.linalg.eigh(K, M)
        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(K)).dot(ModMat)
        self.Kstmodal = ((ModMat.T).dot(Kst)).dot(ModMat)
        self.angANG = -np.pi / 180

        FFmis = np.zeros(self.ndof)

        self.ModalTransformation(FFmis)

        y0 = np.zeros(24)
        # x = scipy.integrate.RK45(
        #    self.EquationOfMovement,
        #    0,
        #    y0,
        #    100,
        #    # max_step=0.01,
        #    rtol=0.001,
        #    atol=1e-06,
        #    vectorized=False,
        #    first_step=None,
        # )
        dt = 0.0001
        # time = np.arange(dt, self.tF, dt)
        self.inv_Mmodal = np.linalg.inv(self.Mmodal)
        t1 = time.time()
        x = scipy.integrate.solve_ivp(
            self.EquationOfMovement,
            (self.tI, self.tF),
            y0,
            method="RK45",
            # t_eval=time,
            # dense_output=True,
            # atol=1e-03,
            # rtol=0.1,
        )
        t2 = time.time()
        print(f"spend time: {t2-t1} s")

        displacement = x.y[:12, :]
        velocity = x.y[12:, :]
        response = self.ModMat.dot(displacement)
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x.t, y=response[84, :]))
        fig.show()

    # '''
    #         function Y=movimento(T,Y)
    #             %controla a for�a
    #             if T>tf
    #                 ft=0;
    #             else
    #                 ft=f*sin(OM*T);
    #             end

    #             y0=Y(2,1); %posi��o (x)
    #             y1=Y(1,1); %velocidade (v, x�)

    #             Y(1,1)=ft/M-(C/M)*y1-(K/M)*y0; %equa��o do movimento for�ado com
    #             %amortecimento

    #             Y(2,1)=y1; %velocidade do sistema
    #         end
    # '''

    def EquationOfMovement(self, T, Y):
        self.iteration += 1
        if self.iteration % 100 == 0:
            print(f"iteration: {self.iteration} \n time: {T}")

        X_dot = Y[:12]
        V_dot = Y[12:]  # velocity ign space state

        kcoup_auxt = self.rotor.K(self.speedI)[5 + 6 * self.n2, 5 + 6 * self.n2] / (
            self.rotor.K(self.speedI)[5 + 6 * self.n1, 5 + 6 * self.n1]
            + self.rotor.K(self.speedI)[5 + 6 * self.n2, 5 + 6 * self.n2]
        )

        self.calculate_angular_position(T)

        self.angANG = (
            self.Kcoup_auxI * self.angular_position
            + self.Kcoup_auxF * self.angular_position
            + self.kCOUP
            * kcoup_auxt
            * self.eCOUP
            * (
                -X_dot[0 + 6 * self.n1] * np.sin(self.angANG)
                + X_dot[0 + 6 * self.n2] * np.sin(self.angANG)
                - X_dot[1 + 6 * self.n1] * np.cos(self.angANG)
                + X_dot[1 + 6 * self.n2] * np.cos(self.angANG)
            )
        )
        positionsMod = self.ModMat.dot(X_dot)
        Fmis, ft = self._parallel(positionsMod, self.angANG)
        ftmodal = self.ModalTransformation(ft)

        Omega = self.speedI

        new_Y = np.zeros(len(Y))
        new_V_dot = (
            ftmodal
            - ((self.Cmodal + self.Gmodal * Omega)).dot(V_dot)
            - ((self.Kmodal + self.Kstmodal * Omega).dot(X_dot))
        ).dot(
            self.inv_Mmodal
        )  # proper equation of movement to be integrated in time

        # aux[12:] = (
        #    ftmodal.dot(self.inv_Mmodal)
        #    - ((self.Cmodal + self.Gmodal * Omega).dot(self.inv_Mmodal)).dot(y1)
        #    - ((self.Kmodal + self.Kstmodal * Omega).dot(self.inv_Mmodal)).dot(y0)
        # )  # proper equation of movement to be integrated in time
        new_X_dot = V_dot

        new_Y[:12] = new_X_dot
        new_Y[12:] = new_V_dot

        # Y[12:] = (
        #     ftmodal.dot(np.linalg.inv(self.Mmodal))
        #     - ((self.Cmodal + self.Gmodal * Omega).dot(np.linalg.inv(self.Mmodal))).dot(
        #         y1
        #     )
        #     - (
        #         (self.Kmodal + self.Kstmodal * Omega).dot(np.linalg.inv(self.Mmodal))
        #     ).dot(y0)
        # )  # proper equation of movement to be integrated in time
        # Y[:12] = y1  # velocity of system
        return new_Y

    def calculate_angular_position(self, t):

        self.angular_position = (
            self.sA * t
            - (self.sB / self.lambdat) * np.exp(-self.lambdat * t)
            + (self.sB / self.lambdat)
        )

    def ModalTransformation(self, ft):

        return (self.ModMat.T).dot(ft)

    def _parallel(self, positions, fir):

        self.k0 = self.kCOUP
        self.delta1 = self.eCOUP

        self.beta = 0

        k_misalignbeta1 = np.array(
            [
                self.k0 * self.Kcoup_auxI * self.delta1 * np.sin(self.beta - fir),
                -self.k0 * self.Kcoup_auxI * self.delta1 * np.cos(self.beta - fir),
                0,
                0,
                0,
                0,
                -self.k0 * self.Kcoup_auxF * self.delta1 * np.sin(self.beta - fir),
                self.k0 * self.Kcoup_auxF * self.delta1 * np.cos(self.beta - fir),
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
            (self.n1 * 6 - 5) + 5,
            (self.n1 * 6 - 3) + 5,
            (self.n1 * 6 - 4) + 5,
            (self.n1 * 6 - 2) + 5,
            (self.n1 * 6 - 0) + 5,
            (self.n1 * 6 - 1) + 5,
            (self.n2 * 6 - 5) + 5,
            (self.n2 * 6 - 3) + 5,
            (self.n2 * 6 - 4) + 5,
            (self.n2 * 6 - 2) + 5,
            (self.n2 * 6 - 0) + 5,
            (self.n2 * 6 - 1) + 5,
        ]
        self.Force_kkmis = K_mis_matrix.dot(positions[self.DoF])

        F_misalign = np.array(
            [
                (
                    -self.k0 * self.delta1 * np.cos(self.beta - fir)
                    + self.k0 * self.delta1
                ),
                -self.k0 * self.delta1 * np.sin(self.beta - fir),
                0,
                0,
                0,
                self.TD - self.TL,
                (
                    self.k0 * self.delta1 * np.cos(self.beta - fir)
                    - self.k0 * self.delta1
                ),
                self.k0 * self.delta1 * np.sin(self.beta - fir),
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
