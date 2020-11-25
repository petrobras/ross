import time
from abc import ABC, abstractmethod

import numpy as np
import plotly.graph_objects as go
import scipy as sp
import scipy.integrate
import scipy.linalg

import ross
from ross.results import TimeResponseResults
from ross.units import Q_

from .abs_defect import Defect
from .integrate_solver import Integrator

__all__ = [
    "Crack",
]


class Crack(Defect):
    """Contains a Gasch and Mayes transversal crack models for applications on finite element models of rotative machinery.
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
    cd : float
        Crack depth
    n_crack : float
        Element where the crack is located
    speed : float
        Operational speed of the machine
    massunb : array
        Array with the unbalance magnitude. The unit is kg.m.
    phaseunb : array
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
    .. [1] Mayes, I. W., & Davies, W. G. R. (1984). Analysis of the response of a multi-rotor-bearing system
           containing a transverse crack in a rotor;
       [2] Gasch, R. (1993). A survey of the dynamic behaviour of a simple rotating shaft with a transverse
           crack. Journal of sound and vibration, 160(2), 313-332;
       [3] Papadopoulos, C. A., & Dimarogonas, A. D. (1987). Coupled longitudinal and bending vibrations
           of a rotating shaft with an open crack. Journal of sound and vibration, 117(1), 81-93...

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
        self,
        dt,
        tI,
        tF,
        cd,
        n_crack,
        speed,
        massunb,
        phaseunb,
        crack_type="Mayes",
        print_progress=False,
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
        self.print_progress = print_progress

        if crack_type is None or crack_type == "Mayes":
            self.crack_model = self._mayes
        elif crack_type == "Gasch":
            self.crack_model = self._gasch
        else:
            raise Exception("Check the crack model!")

    def run(self, rotor):
        """Calculates the shaft angular position and the unbalance forces at X / Y directions.

        Parameters
        ----------
        rotor : ross.Rotor Object
             6 DoF rotor model.
                
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

        kappa = (6 * (1 + self.Poisson) ** 2) / (
            7 + 12 * self.Poisson + 4 * self.Poisson ** 2
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
        t_eval = np.arange(self.tI, self.tF + self.dt, self.dt)
        # t_eval = np.arange(self.dt, self.tF, self.dt)
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
        self.forces_crack = np.zeros((self.ndof, len(t_eval)))

        FFunb[self.ndofd1, :] += unb1x
        FFunb[self.ndofd1 + 1, :] += unb1y
        FFunb[self.ndofd2, :] += unb2x
        FFunb[self.ndofd2 + 1, :] += unb2y

        self.Funbmodal = (self.ModMat.T).dot(FFunb)

        self.inv_Mmodal = np.linalg.pinv(self.Mmodal)
        t1 = time.time()

        x = Integrator(
            self.tI,
            y0,
            self.tF,
            self.dt,
            self._equation_of_movement,
            self.print_progress,
        )
        x = x.rk45()
        t2 = time.time()
        if self.print_progress:
            print(f"Time spent: {t2-t1} s")

        self.displacement = x[:12, :]
        self.velocity = x[12:, :]
        self.time_vector = t_eval
        self.response = self.ModMat.dot(self.displacement)

    def _equation_of_movement(self, T, Y, i):
        """ Calculates the displacement and velocity using state-space representation in the modal domain.

        Parameters
        ----------
        T : float
            Iteration time.
        Y : array
            Array of displacement and velocity, in the modal domain.
        i : int
            Iteration step.

        Returns
        -------
        new_Y :  array
            Array of the new displacement and velocity, in the modal domain.
        """

        positions = Y[:12]
        velocity = Y[12:]  # velocity in space state

        self.positionsFis = self.ModMat.dot(positions)
        self.velocityFis = self.ModMat.dot(velocity)
        self.T_matrix = np.array(
            [
                [np.cos(self.angular_position[i]), np.sin(self.angular_position[i])],
                [-np.sin(self.angular_position[i]), np.cos(self.angular_position[i])],
            ]
        )
        self.tp = self.crack_model(self.angular_position[i])

        FF_CRACK, ft = self._crack(self.tp, self.angular_position[i])
        self.forces_crack[:, i] = ft
        ftmodal = (self.ModMat.T).dot(ft)

        # equation of movement to be integrated in time
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

    def _crack(self, func, ap):
        """Reaction forces of cracked element
        
        Returns
        -------
        F_mis_p(12,n) : numpy.ndarray
            Excitation force caused by the parallel misalignment for a 6DOFs system with 'n' values of angular position  
        """

        # self.T_matrix = np.array([[np.cos(ap), np.sin(ap)], [-np.sin(ap), np.cos(ap)],])

        K = func

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
        KK_crack = np.array([[Koxy[0,0]	,0           ,0         ,0	        ,Koxy[0,1]	,0	        ,Koxy[0,2]	        ,0                  ,0      ,0                  ,Koxy[0,3]	,0],
                             [0	        ,Koyz[0,0]   ,0         ,Koyz[0,1]	,0          ,0          ,0                  ,Koyz[0,2]	        ,0      ,Koyz[0,3]	        ,0          ,0],
                             [0	        ,0	         ,0         ,0	        ,0          ,0          ,0                  ,0                  ,0      ,0                  ,0          ,0],
                             [0	        ,Koyz[1,0]	 ,0         ,Koyz[1,1]  ,0          ,0          ,0                  ,Koyz[1,2]          ,0      ,Koyz[1,3]          ,0          ,0],
                             [Koxy[1,0]	,0           ,0         ,0	        ,Koxy[1,1]  ,0          ,Koxy[1,2]          ,0                  ,0      ,0                  ,Koxy[1,3]  ,0],
                             [0	        ,0           ,0         ,0	        ,0	        ,0          ,0                  ,0                  ,0      ,0                  ,0	        ,0],
                             [Koxy[2,0]	,0           ,0         ,0	        ,Koxy[2,1]	,0          ,Koxy[2,2]          ,0                  ,0      ,0                  ,Koxy[2,3]	,0],
                             [0	        ,Koyz[2,0]   ,0         ,Koyz[2,1]	,0 ,         0          ,0                  ,Koyz[2,2]          ,0      ,Koyz[2,3]          ,0          ,0],
                             [0	        ,0	         ,0         ,0	        ,0 ,         0          ,0                  ,0                  ,0      ,0                  ,0          ,0],
                             [0	        ,Koyz[3,0]	 ,0         ,Koyz[3,1]	,0 ,         0          ,0                  ,Koyz[3,2]          ,0      ,Koyz[3,3]          ,0          ,0],
                             [Koxy[3,0]	,0           ,0         , 0	        ,Koxy[3,1]  ,0          ,Koxy[3,2]          ,0                  ,0      ,0                  ,Koxy[3,3]  ,0],
                             [0	        ,0           ,0         , 0	        ,0	        ,0          ,0                  ,0                  ,0      ,0                  ,0	        ,0]])
        # fmt: on
        F_CRACK = np.zeros(self.ndof)

        KK_CRACK = self.KK - KK_crack
        FF_CRACK = (KK_CRACK).dot(self.positionsFis[self.dof_crack])
        F_CRACK[self.dof_crack] = FF_CRACK

        return FF_CRACK, F_CRACK

    def _gasch(self, ap):
        # Gasch
        kme = (self.ko + self.kcx) / 2
        kmn = (self.ko + self.kcz) / 2
        kde = (self.ko - self.kcx) / 2
        kdn = (self.ko - self.kcz) / 2

        kee = kme + (4 / np.pi) * kde * (
            np.cos(ap)
            - np.cos(3 * ap) / 3
            + np.cos(5 * ap) / 5
            - np.cos(7 * ap) / 7
            + np.cos(9 * ap) / 9
            - np.cos(11 * ap) / 11
            + np.cos(13 * ap) / 13
            - np.cos(15 * ap) / 15
            + np.cos(17 * ap) / 17
            - np.cos(19 * ap) / 19
            + np.cos(21 * ap) / 21
            - np.cos(23 * ap) / 23
            + np.cos(25 * ap) / 25
            - np.cos(27 * ap) / 27
            + np.cos(29 * ap) / 29
            - np.cos(31 * ap) / 31
            + np.cos(33 * ap) / 33
            - np.cos(35 * ap) / 35
        )

        knn = kmn + (4 / np.pi) * kdn * (
            np.cos(ap)
            - np.cos(3 * ap) / 3
            + np.cos(5 * ap) / 5
            - np.cos(7 * ap) / 7
            + np.cos(9 * ap) / 9
            - np.cos(11 * ap) / 11
            + np.cos(13 * ap) / 13
            - np.cos(15 * ap) / 15
            + np.cos(17 * ap) / 17
            - np.cos(19 * ap) / 19
            + np.cos(21 * ap) / 21
            - np.cos(23 * ap) / 23
            + np.cos(25 * ap) / 25
            - np.cos(27 * ap) / 27
            + np.cos(29 * ap) / 29
            - np.cos(31 * ap) / 31
            + np.cos(33 * ap) / 33
            - np.cos(35 * ap) / 35
        )

        aux = np.array([[kee, 0], [0, knn]])

        K = ((self.T_matrix.T).dot(aux)).dot(self.T_matrix)

        return K

    def _mayes(self, ap):
        # Mayes

        kee = 0.5 * (self.ko + self.kcx) + 0.5 * (self.ko - self.kcx) * np.cos(ap)

        knn = 0.5 * (self.ko + self.kcz) + 0.5 * (self.ko - self.kcz) * np.cos(ap)

        aux = np.array([[kee, 0], [0, knn]])

        K = ((self.T_matrix.T).dot(aux)).dot(self.T_matrix)

        return K

    def _get_coefs(self, coef):
        x = scipy.io.loadmat(
            "/home/izabela/Documents/Projeto EDGE Petro/ross/tools/data/PAPADOPOULOS_c"
        )

        c = x[coef]
        aux = np.where(c[:, 1] >= self.cd * 2)[0]
        c = c[aux[0], 0] * (1 - self.Poisson ** 2) / (self.E * (self.radius ** 3))

        return c
