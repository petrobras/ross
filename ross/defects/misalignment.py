from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
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
        self.DoF = np.arange((self.n1 * 6), (self.n2 * 6 + 6))

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
        self.K = self.rotor.K(self.speedI * np.pi / 30)
        self.C = self.rotor.C(self.speedI * np.pi / 30)
        self.G = self.rotor.G()
        self.M = self.rotor.M()
        self.Kst = self.rotor.Kst()

        _, ModMat = scipy.linalg.eigh(self.K, self.M)
        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(self.M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(self.C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(self.G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(self.K)).dot(ModMat)
        self.Kstmodal = ((ModMat.T).dot(self.Kst)).dot(ModMat)
        self.angANG = -np.pi / 180

        FFmis = np.zeros(self.ndof)

        y0 = np.zeros(24)
        dt = 0.0001
        tt = np.arange(dt, self.tF, dt)
        self.inv_Mmodal = np.linalg.inv(self.Mmodal)
        t1 = time.time()
        # x = scipy.integrate.RK45(
        #     self.EquationOfMovement,
        #     0,
        #     y0,
        #     1,
        #     # max_step=0.01,
        #     rtol=0.001,
        #     atol=1e-06,
        #     vectorized=False,
        #     first_step=None,
        # )
        x = scipy.integrate.solve_ivp(
            self.EquationOfMovement,
            (self.tI, self.tF),
            y0,
            method="RK45",
            t_eval=tt,
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

        rows, cols = response.shape
        AA = RotorSignalAnalysis(
            response[84, int(cols / 2) :], response[84, int(cols / 2) :], dt
        )
        amp = AA.x_amp
        freq = AA.freq

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x.t, y=response[84, :]))
        fig.show()

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=freq, y=amp))
        fig1.update_layout(yaxis_type="log")
        fig1.show()

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
        if self.iteration % 10000 == 0:
            print(f"iteration: {self.iteration} \n time: {T}")

        positions = Y[:12]
        velocity = Y[12:]  # velocity ign space state

        kcoup_auxt = self.K[5 + 6 * self.n2, 5 + 6 * self.n2] / (
            self.K[5 + 6 * self.n1, 5 + 6 * self.n1]
            + self.K[5 + 6 * self.n2, 5 + 6 * self.n2]
        )

        angular_position = (
            self.sA * T
            - (self.sB / self.lambdat) * np.exp(-self.lambdat * T)
            + (self.sB / self.lambdat)
        )
        # SpeedV = sA + sB * np.exp(-lambdat * )

        self.angANG = (
            self.Kcoup_auxI * angular_position
            + self.Kcoup_auxF * angular_position
            + self.kCOUP
            * kcoup_auxt
            * self.eCOUP
            * (
                -positions[0 + 6 * self.n1] * np.sin(self.angANG)
                + positions[0 + 6 * self.n2] * np.sin(self.angANG)
                - positions[1 + 6 * self.n1] * np.cos(self.angANG)
                + positions[1 + 6 * self.n2] * np.cos(self.angANG)
            )
        )
        positionsMod = self.ModMat.dot(positions)
        Fmis, ft = self._parallel(positionsMod, self.angANG)
        ftmodal = (self.ModMat.T).dot(ft)

        Omega = self.speedI * np.pi / 30

        new_V_dot = (
            ftmodal
            - ((self.Cmodal + self.Gmodal * Omega)).dot(velocity)
            - ((self.Kmodal + self.Kstmodal * Omega).dot(positions))
        ).dot(
            self.inv_Mmodal
        )  # proper equation of movement to be integrated in time

        # aux[12:] = (
        #    ftmodal.dot(self.inv_Mmodal)
        #    - ((self.Cmodal + self.Gmodal * Omega).dot(self.inv_Mmodal)).dot(y1)
        #    - ((self.Kmodal + self.Kstmodal * Omega).dot(self.inv_Mmodal)).dot(y0)
        # )  # proper equation of movement to be integrated in time
        new_X_dot = velocity

        new_Y = np.zeros(24)
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

    def _parallel(self, positions, fir):

        k0 = self.kCOUP
        delta1 = self.eCOUP

        beta = 0

        k_misalignbeta1 = np.array(
            [
                -k0 * self.Kcoup_auxI * delta1 * np.sin(beta - fir),
                -k0 * self.Kcoup_auxI * delta1 * np.cos(beta - fir),
                0,
                0,
                0,
                0,
                +k0 * self.Kcoup_auxF * delta1 * np.sin(beta - fir),
                k0 * self.Kcoup_auxF * delta1 * np.cos(beta - fir),
                0,
                0,
                0,
                0,
            ]
        )

        K_mis_matrix = np.zeros((12, 12))
        K_mis_matrix[5, :] = k_misalignbeta1
        K_mis_matrix[11, :] = -k_misalignbeta1

        Force_kkmis = K_mis_matrix.dot(positions[self.DoF])

        F_misalign = np.array(
            [
                -(-k0 * delta1 * np.cos(beta - fir) + k0 * delta1),
                -k0 * delta1 * np.sin(beta - fir),
                0,
                0,
                0,
                self.TD - self.TL,
                -(k0 * delta1 * np.cos(beta - fir) - k0 * delta1),
                k0 * delta1 * np.sin(beta - fir),
                0,
                0,
                0,
                -(self.TD - self.TL),
            ]
        )

        Fmis = Force_kkmis + F_misalign
        FFmis = np.zeros(self.ndof)
        FFmis[self.DoF] = Fmis

        return Fmis, FFmis

    def force(self):
        return self._parallel()


class RotorSignalAnalysis:
    """ Contains signal analysis' methods, using vibration signals from perpendicular sensors, e.g. x and y directions. 
    The reference coordenates system is: y-axis throught the shaft center; x-axis and z-axis in the sensors' planes
    
    Parameters
    ----------
    x : numpy.ndarray
        Vibration signal in the time domain in x-axis.
    z : numpy.ndarray
        Vibration signal in the time domain in z-axis.
    dt : float
        Time step.
    
    References
    ----------
    .. [1] 'Application of full spectrum to rotating machinery diagnostics' by Ph.D. Paul Goldman ...
    & Ph.D. Agnes Muszynska. Avaible at http://amconsulting.intellibit.com/499agnes.shtml.

    Examples
    --------
    >>> from rotordynamics import RotorSignalAnalysis
    >>> import numpy as np
    >>> t = np.arange(0, 10, 0.01)
    >>> dt = t[1] - t[0]
    >>> w = 2
    >>> x = np.sin(2 * np.pi * w * t) + np.random.rand(len(t)) / 10
    >>> z = np.cos(2 * np.pi * w * t) + np.random.rand(len(t)) / 10
    >>> r_analysis = RotorSignalAnalysis(x, z, dt)
    >>> freq, amp = r_analysis.full_fft(plot=True)
    """

    def __init__(self, x, z, dt: float):

        window = sp.signal.windows.boxcar(len(x))

        b = np.floor(len(x) / 2)
        c = len(x)
        df = 1 / (c * dt)

        x_amp = sp.fft(x * window)[: int(b)]
        x_amp = x_amp * 2 / c
        self.x_phase = np.angle(x_amp)
        self.x_amp = np.abs(x_amp)

        b = np.floor(len(z) / 2)
        c = len(z)
        z_amp = sp.fft(z * window)[: int(b)]
        z_amp = z_amp * 2 / c
        self.z_phase = np.angle(z_amp)
        self.z_amp = np.abs(z_amp)

        freq = np.arange(0, df * b, df)
        freq = freq[: int(b)]  # Frequency vector

        self.freq = freq

    def forward(self, plot=False, log=False):
        """Calculates the forward spectrum fourier transform.
        
        Parameters
        ----------
        plot : bool, optional
            Set it as True to plot the forward spectrum, by default it is False
        log : bool, optional
            Set it as True to plot using 'log' scale in the y-axis

        Returns
        -------
        freq : numpy.ndarray
            Frequency vector
        f_amp : numpy.ndarray
            Forward amplitude vector
            
        Examples
        --------

        >>> from rotordynamics import RotorSignalAnalysis
        >>> import numpy as np
        >>> t = np.arange(0, 10, 0.01)
        >>> dt = t[1] - t[0]
        >>> w = 2
        >>> x = np.sin(2 * np.pi * w * t) + np.random.rand(len(t)) / 10
        >>> z = np.cos(2 * np.pi * w * t) + np.random.rand(len(t)) / 10
        >>> r_analysis = RotorSignalAnalysis(x, z, dt)
        >>> freq, f_amp = r_analysis.forward(plot=True)
        """
        f_amp = np.abs(
            (
                self.x_amp ** 2
                + self.z_amp ** 2
                - 2 * self.x_amp * self.z_amp * np.sin(self.x_phase - self.z_phase)
            )
            ** 0.5  # Forward amplitude
        )
        if plot:
            self.__plot(
                self.freq, f_amp, title="Forward Spectrum Fourier Transform", log=log
            )
        return self.freq, f_amp

    def backward(self, plot=False, log=False):
        """Calculates the backward spectrum fourier transform.
        
        Parameters
        ----------
        plot : bool, optional
            Set it as True to plot the backward spectrum, by default it is False
        log : bool, optional
            Set it as True to plot using 'log' scale in the y-axis
        
        Returns
        -------
        freq : numpy.ndarray
            Frequency vector
        b_amp : numpy.ndarray
            Backward amplitude vector

        Examples
        --------
        >>> from rotordynamics import RotorSignalAnalysis
        >>> import numpy as np
        >>> t = np.arange(0, 10, 0.01)
        >>> dt = t[1] - t[0]
        >>> w = 2
        >>> x = np.sin(2 * np.pi * w * t) + np.random.rand(len(t)) / 10
        >>> z = np.cos(2 * np.pi * w * t) + np.random.rand(len(t)) / 10
        >>> r_analysis = RotorSignalAnalysis(x, z, dt)
        >>> freq, b_amp = r_analysis.backward(plot=True)
        """
        b_amp = np.abs(
            (
                self.x_amp ** 2
                + self.z_amp ** 2
                + 2 * self.x_amp * self.z_amp * np.sin(self.x_phase - self.z_phase)
            )
            ** 0.5  # Backward amplitude
        )
        if plot:
            self.__plot(
                self.freq, b_amp, title="Backward Spectrum Fourier Transform", log=log
            )
        return self.freq, b_amp

    def full_fft(self, plot=False, log=False):
        """Creates the full spectrum fourier transform vector.
        
        Parameters
        ----------
        plot : bool, optional
            Set it as True to plot the full spectrum, by default it is False
        log : bool, optional
            Set it as True to plot using 'log' scale in the y-axis

        Returns
        -------
        full_freq : numpy.ndarray
            Frequency vector
        full_amp : numpy.ndarray
            Full amplitude vector

        Examples
        --------
        >>> from rotordynamics import RotorSignalAnalysis
        >>> import numpy as np
        >>> t = np.arange(0, 10, 0.01)
        >>> dt = t[1] - t[0]
        >>> w = 2
        >>> x = np.sin(2 * np.pi * w * t) + np.random.rand(len(t)) / 10
        >>> z = np.cos(2 * np.pi * w * t) + np.random.rand(len(t)) / 10
        >>> r_analysis = RotorSignalAnalysis(x, z, dt)
        >>> freq, amp = r_analysis.full_fft(plot=True)
        """
        full_freq = np.concatenate(
            (-self.freq[-1:0:-1], self.freq)
        )  # full spectrum frequency -> [-n..0..n]
        f_amp = self.forward()[1]
        b_amp = self.backward()[1]
        full_amp = np.concatenate(
            (b_amp[-1:0:-1], f_amp)
        )  # full spectrum amplitude -> [backward forward]

        if plot:
            self.__plot(
                full_freq, full_amp, title="Full Spectrum Fourier Transform", log=log
            )

        return full_freq, full_amp

    def __plot(self, x, y, title="", log=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines"))
        fig.update_layout(
            title=title, xaxis_title="Frequency [Hz]", yaxis_title="Amplitude [um]",
        )
        if log:
            fig.update_layout(yaxis_type="log")
        fig.show()
