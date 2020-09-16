"""Cracks module.

This module defines the Defect classes for cracks on the shaft. There 
are a number of options, for the formulation of 6 DoFs (degrees of freedom).
"""
from abc import ABC, abstractmethod

import ross
import numpy as np
import scipy as sp
import scipy.integrate
import scipy.linalg
import time
from ross.units import Q_

import plotly.graph_objects as go

from .abs_defect import Defect
from .integrate_solver import Integrator

__all__ = [
    "Crack",
]


class Crack:
    """A Gasch and Mayes crack models for a shaft.
    
    Calculates the dynamic forces of a crack on a given shaft element.

    Parameters
    ----------
    dt : float
        Time step
    tI : float
        Initial time
    tF : float
        Final time
    kd : float
        Radial stiffness of flexible coupling
    ks : float
        Bending stiffness of flexible coupling
    eCOUPx : float
        Parallel misalignment offset between driving rotor and driven rotor along X direction
    eCOUPy : float
        Parallel misalignment offset between driving rotor and driven rotor along Y direction
    misalignment_angle : float
        Angle of the angular misaligned 
    TD : float
        Driving torque
    TL : float
        Driven torque
    n1 : float
        Node where the misalignment is ocurring
    speed : float
        Operational speed of the machine

    Returns
    -------
    A force to be applied on the shaft.

    References
    ----------
    .. [1] 'Xia, Y., Pang, J., Yang, L., Zhao, Q., & Yang, X. (2019). Study on vibration response 
    and orbits of misaligned rigid rotors connected by hexangular flexible coupling. Applied 
    Acoustics, 155, 286-296..

    Examples
    --------
    AQUI AINDA TEM QUE SER ATUALIZADO, ABAIXO SEGUE SOMENTE UM EXEMPLO PARA A "SHAFT ELEMENT"
    >>> from ross.materials import steel
    >>> Timoshenko_Element = ShaftElement(
    ...                         material=steel, L=0.5, idl=0.05, odl=0.1,
    ...                         rotary_inertia=True,
    ...                         shear_effects=True)
    >>> Timoshenko_Element.phi
    0.1571268472906404
    """

    def __init__(
        self, dt, tI, tF, cd, n_crack, speed, massunb, phaseunb, crack_type="Mayes"
    ):

        self.dt = dt
        self.tI = tI
        self.tF = tF
        self.cd = cd
        self.n_crack = n_crack
        self.speed = speed
        self.speedI = speed
        self.speedF = speed
        self.MassUnb1 = massunb[0]
        self.MassUnb2 = massunb[1]
        self.PhaseUnb1 = phaseunb[0]
        self.PhaseUnb2 = phaseunb[1]

        if crack_type is None or crack_type == "Mayes":
            self.crack_model = self._mayes
        elif crack_type == "Gasch":
            self.crack_model = self._gasch
        else:
            raise Exception("Check the crack model!")

    def run(self, rotor):
        """Calculates the shaft angular position and the misalignment amount at X / Y directions.

        Parameters
        ----------
        radius : float
                Radius of shaft in node of misalignment
        ndof : float
                Total number of degrees of freedom

        Returns
        -------
        
                
        """
        self.rotor = rotor
        self.ndof = rotor.ndof
        self.L = rotor.elements[self.n_crack].L
        self.KK = rotor.elements[self.n_crack].K()
        self.radius = rotor.elements[self.n_crack].odl / 2
        self.Poisson = rotor.elements[self.n_crack].material.Poisson
        self.E = rotor.elements[self.n_crack].material.E

        self.ndofd1 = (self.rotor.disk_elements[0].n) * 6
        self.ndofd2 = (self.rotor.disk_elements[1].n) * 6

        G_s = rotor.elements[self.n_crack].material.G_s
        odr = rotor.elements[self.n_crack].odr
        odl = rotor.elements[self.n_crack].odl
        idr = rotor.elements[self.n_crack].idr
        idl = rotor.elements[self.n_crack].idl

        self.dof_crack = np.arange((self.n_crack * 6), (self.n_crack * 6 + 12))
        tempS = np.pi * (
            ((odr / 2) ** 2 + (odl / 2) ** 2) / 2
            - ((idr / 2) ** 2 + (idl / 2) ** 2) / 2
        )
        tempI = (
            np.pi
            / 4
            * (
                ((odr / 2) ** 4 + (odl / 2) ** 4) / 2
                - ((idr / 2) ** 4 + (idl / 2) ** 4) / 2
            )
        )

        r = ((idl + idr) / 2) / ((odl + odr) / 2)
        r2 = r * r
        r12 = (1 + r2) ** 2

        kappa = (
            6
            * r12
            * (
                (1 + self.Poisson)
                / (
                    (
                        r12 * (7 + 12 * self.Poisson + 4 * self.Poisson ** 2)
                        + 4 * r2 * (5 + 6 * self.Poisson + 2 * self.Poisson ** 2)
                    )
                )
            )
        )

        A = 12 * self.E * tempI / (G_s * kappa * tempS * (self.L ** 2))

        # fmt = off
        Coxy = np.array(
            [
                [
                    (self.L ** 3) * (1 + A / 4) / (3 * self.E * tempI),
                    -(self.L ** 2) / (2 * self.E * tempI),
                ],
                [-(self.L ** 2) / (2 * self.E * tempI), self.L / (self.E * tempI)],
            ]
        )

        Coyz = np.array(
            [
                [
                    (self.L ** 3) * (1 + A / 4) / (3 * self.E * tempI),
                    (self.L ** 2) / (2 * self.E * tempI),
                ],
                [(self.L ** 2) / (2 * self.E * tempI), self.L / (self.E * tempI)],
            ]
        )

        Co = np.array(
            [
                [Coxy[0, 0], 0, 0, Coxy[0, 1]],
                [0, Coyz[0, 0], Coyz[0, 1], 0],
                [0, Coyz[1, 0], Coyz[1, 1], 0],
                [Coxy[1, 0], 0, 0, Coxy[1, 1]],
            ]
        )
        # fmt = on

        c44 = self._get_coefs("c44")
        c55 = self._get_coefs("c55")
        c45 = self._get_coefs("c45")

        if self.cd == 0:
            Cc = Co
        else:
            Cc = Co + np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, c55, c45], [0, 0, c45, c44]]
            )

        self.Kele = np.linalg.pinv(Co)
        self.ko = self.Kele[0, 0]

        self.kc = np.linalg.pinv(Cc)
        self.kcx = self.kc[0, 0]
        self.kcz = self.kc[1, 1]
        self.fcrack = np.zeros(self.ndof)

        self.iteration = 0

        warI = self.speedI * np.pi / 30
        warF = self.speedF * np.pi / 30

        # parameters for the time integration
        self.lambdat = 0.00001
        Faxial = 0
        TorqueI = 0
        TorqueF = 0

        # pre-processing of auxilary variuables for the time integration
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

        # SpeedV = sA + sB * np.exp(-lambdat * self.t)
        # TorqueV = sAT + sBT * np.exp(-lambdat * self.t)
        # AccelV = -lambdat * sB * np.exp(-lambdat * self.t)

        # Determining the modal matrix
        self.K = self.rotor.K(self.speedI * np.pi / 30)
        self.C = self.rotor.C(self.speedI * np.pi / 30)
        self.G = self.rotor.G()
        self.M = self.rotor.M()
        self.Kst = self.rotor.Kst()

        _, ModMat = scipy.linalg.eigh(self.K, self.M, type=1, turbo=False,)
        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(self.M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(self.C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(self.G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(self.K)).dot(ModMat)
        self.Kstmodal = ((ModMat.T).dot(self.Kst)).dot(ModMat)

        y0 = np.zeros(24)
        self.dt = 0.0001
        t_eval = np.arange(self.dt, self.tF, self.dt)
        self.inv_Mmodal = np.linalg.pinv(self.Mmodal)
        t1 = time.time()

        x = Integrator(0, y0, self.tF, self.dt, self._equation_of_movement)
        x = x.rk4()
        t2 = time.time()
        print(f"spend time: {t2-t1} s")

        self.displacement = x[:12, :]
        self.velocity = x[12:, :]
        self.time_vector = t_eval
        self.response = self.ModMat.dot(self.displacement)

    def _equation_of_movement(self, T, Y):
        self.iteration += 1
        if self.iteration % 10000 == 0:
            print(f"iteration: {self.iteration} \n time: {T}")

        positions = Y[:12]
        velocity = Y[12:]  # velocity in space state

        self.angular_position = (
            self.sA * T
            - (self.sB / self.lambdat) * np.exp(-self.lambdat * T)
            + (self.sB / self.lambdat)
        )

        self.positionsFis = self.ModMat.dot(positions)
        self.velocityFis = self.ModMat.dot(velocity)

        self.tetaUNB1 = self.angular_position + self.PhaseUnb1
        self.tetaUNB2 = self.angular_position + self.PhaseUnb2

        # Omega = self.speedI * np.pi / 30
        self.Omega = self.sA + self.sB * np.exp(-self.lambdat * T)
        self.AccelV = -self.lambdat * self.sB * np.exp(-self.lambdat * T)

        unb1x = self.MassUnb1 * self.AccelV * np.cos(self.tetaUNB1) - self.MassUnb1 * (
            self.Omega ** 2
        ) * np.sin(self.tetaUNB1)

        unb1y = -self.MassUnb1 * self.AccelV * np.sin(self.tetaUNB1) - self.MassUnb1 * (
            self.Omega ** 2
        ) * np.cos(self.tetaUNB1)

        unb2x = self.MassUnb2 * self.AccelV * np.cos(self.tetaUNB2) - self.MassUnb2 * (
            self.Omega ** 2
        ) * np.sin(self.tetaUNB2)
        unb2y = -self.MassUnb2 * self.AccelV * np.sin(self.tetaUNB2) - self.MassUnb2 * (
            self.Omega ** 2
        ) * np.cos(self.tetaUNB2)
        FFunb = np.zeros(self.ndof)

        FFunb[self.ndofd1] += unb1x
        FFunb[self.ndofd1 + 1] += unb1y
        FFunb[self.ndofd2] += unb2x
        FFunb[self.ndofd2 + 1] += unb2y

        Funbmodal = (self.ModMat.T).dot(FFunb)

        FF_CRACK, ft = self._crack(self.crack_model)
        ftmodal = (self.ModMat.T).dot(ft)

        # Omega = self.speedI * np.pi / 30
        Omega = self.sA + self.sB * np.exp(-self.lambdat * T)
        AccelV = -self.lambdat * self.sB * np.exp(-self.lambdat * T)

        # equation of movement to be integrated in time
        new_V_dot = (
            ftmodal
            + Funbmodal
            - ((self.Cmodal + self.Gmodal * Omega)).dot(velocity)
            - ((self.Kmodal + self.Kstmodal * AccelV).dot(positions))
        ).dot(self.inv_Mmodal)

        new_X_dot = velocity

        new_Y = np.zeros(24)
        new_Y[:12] = new_X_dot
        new_Y[12:] = new_V_dot

        return new_Y

    def _crack(self, func):
        """Reaction forces of cracked element
        
        Returns
        -------
        F_mis_p(12,n) : numpy.ndarray
            Excitation force caused by the parallel misalignment for a 6DOFs system with 'n' values of angular position  
        """

        self.T_matrix = np.array(
            [
                [np.cos(self.angular_position), np.sin(self.angular_position)],
                [-np.sin(self.angular_position), np.cos(self.angular_position)],
            ]
        )

        K = func()

        k11 = K[0, 0]
        k12 = K[0, 1]
        k22 = K[1, 1]

        # Stiffness matrix of the cracked element
        Toxy = np.array([[-1, 0], [self.L, -1], [1, 0], [0, 1]])  # OXY

        kxy = np.array(
            [[self.Kele[0, 0], self.Kele[0, 3]], [self.Kele[3, 0], self.Kele[3, 3]]]
        )

        kxy[0, 0] = k11
        Koxy = ((Toxy).dot(kxy)).dot(Toxy.T)

        Toyz = np.array([[-1, 0], [-self.L, -1], [1, 0], [0, 1]])  # OYZ

        kyz = np.array(
            [[self.Kele[1, 1], self.Kele[1, 2]], [self.Kele[2, 1], self.Kele[2, 2]]]
        )

        kyz[0, 0] = k22
        Koyz = ((Toyz).dot(kyz)).dot(Toyz.T)

        # fmt: off
        KK_crack = np.array([[Koxy[0,0]	,0  , 0         , 0	        , 0	, Koxy[0,1] , Koxy[0,2] , 0 ,         0 , 0         , 0	, Koxy[0,3]],
                             [0	        ,0  , 0         , 0	        , 0 ,         0 ,         0 , 0 ,         0 , 0         , 0 ,         0],
                             [0	        ,0	, Koyz[0,0] , Koyz[0,1]	, 0 ,         0 ,         0 , 0 , Koyz[0,2] , Koyz[0,3] , 0 ,         0],
                             [0	        ,0	, Koyz[1,0] , Koyz[1,1] , 0 ,         0 ,         0 , 0 , Koyz[1,2] , Koyz[1,3] , 0 ,         0],
                             [0	        ,0  , 0         , 0	        , 0 ,         0 ,         0 , 0 ,         0 , 0         , 0 ,         0],
                             [Koxy[1,0]	,0  , 0         , 0	        , 0	, Koxy[1,1] , Koxy[1,2] , 0 ,         0 , 0         , 0	, Koxy[1,3]],
                             [Koxy[2,0]	,0  , 0         , 0	        , 0	, Koxy[2,1] , Koxy[2,2] , 0 ,         0 , 0         , 0	, Koxy[2,3]],
                             [0	        ,0  , 0         , 0	        , 0 ,         0 ,         0 , 0 ,         0 , 0         , 0 ,         0],
                             [0	        ,0	, Koyz[2,0] , Koyz[2,1]	, 0 ,         0 ,         0 , 0 , Koyz[2,2] , Koyz[2,3] , 0 ,         0],
                             [0	        ,0	, Koyz[3,0] , Koyz[3,1]	, 0 ,         0 ,         0 , 0 , Koyz[3,2] , Koyz[3,3] , 0 ,         0],
                             [0	        ,0  , 0         , 0	        , 0 ,         0 ,         0 , 0 ,         0 , 0         , 0 ,         0],
                             [Koxy[3,0]	,0  , 0         , 0	        , 0	, Koxy[3,1] , Koxy[3,2] , 0 ,         0 , 0         , 0	, Koxy[3,3]]])
        # fmt: on
        F_CRACK = np.zeros(self.ndof)

        KK_CRACK = self.KK - KK_crack
        FF_CRACK = (KK_CRACK).dot(self.positionsFis[self.dof_crack])
        F_CRACK[self.dof_crack] = FF_CRACK

        return FF_CRACK, F_CRACK

    def _gasch(self):
        # Gasch
        kme = (self.ko + self.kcx) / 2
        kmn = (self.ko + self.kcz) / 2
        kde = (self.ko - self.kcx) / 2
        kdn = (self.ko - self.kcz) / 2

        kee = kme + (4 / np.pi) * kde * (
            np.cos(self.angular_position)
            - np.cos(3 * self.angular_position) / 3
            + np.cos(5 * self.angular_position) / 5
            - np.cos(7 * self.angular_position) / 7
            + np.cos(9 * self.angular_position) / 9
            - np.cos(11 * self.angular_position) / 11
            + np.cos(13 * self.angular_position) / 13
            - np.cos(15 * self.angular_position) / 15
            + np.cos(17 * self.angular_position) / 17
            - np.cos(19 * self.angular_position) / 19
            + np.cos(21 * self.angular_position) / 21
            - np.cos(23 * self.angular_position) / 23
            + np.cos(25 * self.angular_position) / 25
            - np.cos(27 * self.angular_position) / 27
            + np.cos(29 * self.angular_position) / 29
            - np.cos(31 * self.angular_position) / 31
            + np.cos(33 * self.angular_position) / 33
            - np.cos(35 * self.angular_position) / 35
        )

        knn = kmn + (4 / pi) * kdn * (
            np.cos(self.angular_position)
            - np.cos(3 * self.angular_position) / 3
            + np.cos(5 * self.angular_position) / 5
            - np.cos(7 * self.angular_position) / 7
            + np.cos(9 * self.angular_position) / 9
            - np.cos(11 * self.angular_position) / 11
            + np.cos(13 * self.angular_position) / 13
            - np.cos(15 * self.angular_position) / 15
            + np.cos(17 * self.angular_position) / 17
            - np.cos(19 * self.angular_position) / 19
            + np.cos(21 * self.angular_position) / 21
            - np.cos(23 * self.angular_position) / 23
            + np.cos(25 * self.angular_position) / 25
            - np.cos(27 * self.angular_position) / 27
            + np.cos(29 * self.angular_position) / 29
            - np.cos(31 * self.angular_position) / 31
            + np.cos(33 * self.angular_position) / 33
            - np.cos(35 * self.angular_position) / 35
        )

        aux = np.array([[kee, 0], [0, knn]])

        K = ((self.T_matrix.T).dot(aux)).dot(self.T_matrix)

        return K

    def _mayes(self):
        # Mayes

        kee = 0.5 * (self.ko + self.kcx) + 0.5 * (self.ko - self.kcx) * np.cos(
            self.angular_position
        )

        knn = 0.5 * (self.ko + self.kcz) + 0.5 * (self.ko - self.kcz) * np.cos(
            self.angular_position
        )

        aux = np.array([[kee, 0], [0, knn]])

        K = ((self.T_matrix.T).dot(aux)).dot(self.T_matrix)

        return K

    def _get_coefs(self, coef):
        x = scipy.io.loadmat(
            "/home/izabela/Documents/Projeto EDGE Petro/ross/PAPADOPOULOS_c"
        )

        c = x[coef]
        aux = np.where(c[:, 1] >= self.cd * 2)[0]
        c = c[aux[0], 0] * (1 - self.Poisson ** 2) / (self.E * (self.radius ** 3))

        return c

    def plot_time_response(
        self,
        probe,
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        """
        """
        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            dofx = p[0] * self.rotor.number_dof
            dofy = p[0] * self.rotor.number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )

            _probe_resp = operator @ np.vstack((self.response[dofx,:], self.response[dofy,:]))
            probe_resp = (
                _probe_resp[0] * np.cos(angle) ** 2  +
                _probe_resp[1] * np.sin(angle) ** 2
            )
            # fmt: on

            probe_resp = Q_(probe_resp, "m").to(displacement_units).m

            fig.add_trace(
                go.Scatter(
                    x=Q_(self.time_vector, "s").to(time_units).m,
                    y=Q_(probe_resp, "m").to(displacement_units).m,
                    mode="lines",
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Time ({time_units}): %{{x:.2f}}<br>Amplitude ({displacement_units}): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(title_text=f"Time ({time_units})")
        fig.update_yaxes(title_text=f"Amplitude ({displacement_units})")
        fig.update_layout(**kwargs)

        return fig

    def plot_dfft(
        self, probe, probe_units="rad", fig=None, log=False, **kwargs,
    ):
        """
        """
        if fig is None:
            fig = go.Figure()

        for i, p in enumerate(probe):
            dofx = p[0] * self.rotor.number_dof
            dofy = p[0] * self.rotor.number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), - np.sin(angle)],
                 [np.cos(angle), + np.sin(angle)]]
            )
            row, cols = self.response.shape
            _probe_resp = operator @ np.vstack((self.response[dofx,int(2*cols/3):], self.response[dofy,int(2*cols/3):]))
            probe_resp = (
                _probe_resp[0] * np.cos(angle) ** 2  +
                _probe_resp[1] * np.sin(angle) ** 2
            )
            # _probe_resp = operator @ np.vstack((self.response[dofx,200000:], self.response[dofy,200000:]))
            # probe_resp = (
            #     _probe_resp[0] * np.cos(angle) ** 2  +
            #     _probe_resp[1] * np.sin(angle) ** 2
            # )
            # fmt: on

            amp, freq = self._dfft(probe_resp, self.dt)

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=amp,
                    mode="lines",
                    name=f"Probe {i + 1}",
                    legendgroup=f"Probe {i + 1}",
                    showlegend=True,
                    hovertemplate=f"Frequency (Hz): %{{x:.2f}}<br>Amplitude (m): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(title_text=f"Frequency (Hz)")
        fig.update_yaxes(title_text=f"Amplitude (m)")
        fig.update_layout(**kwargs)

        if log:
            fig.update_layout(yaxis_type="log")

        return fig

    def _dfft(self, x, dt):
        b = np.floor(len(x) / 2)
        c = len(x)
        df = 1 / (c * dt)

        x_amp = sp.fft(x)[: int(b)]
        x_amp = x_amp * 2 / c
        x_phase = np.angle(x_amp)
        x_amp = np.abs(x_amp)

        freq = np.arange(0, df * b, df)
        freq = freq[: int(b)]  # Frequency vector

        return x_amp, freq
