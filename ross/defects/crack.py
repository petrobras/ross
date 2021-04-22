import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.linalg

import ross
from ross.units import Q_, check_units

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
    depth_ratio : float
        Crack depth ratio related to the diameter of the crack container element. A depth value of 0.1 is equal to 10%, 0.2 equal to 20%, and so on.
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
    .. [1] Mayes, I. W., & Davies, W. G. R. (1984). Analysis of the response of a multi-rotor-bearing system
           containing a transverse crack in a rotor;
       [2] Gasch, R. (1993). A survey of the dynamic behaviour of a simple rotating shaft with a transverse
           crack. Journal of sound and vibration, 160(2), 313-332;
       [3] Papadopoulos, C. A., & Dimarogonas, A. D. (1987). Coupled longitudinal and bending vibrations
           of a rotating shaft with an open crack. Journal of sound and vibration, 117(1), 81-93...

    Examples
    --------
    >>> from ross.defects.crack import crack_example
    >>> probe1 = (14, 0)
    >>> probe2 = (22, 0)
    >>> response = crack_example()
    >>> results = response.run_time_response()
    >>> fig = response.plot_dfft(probe=[probe1, probe2], range_freq=[0, 100], yaxis_type="log")
    >>> # fig.show()
    """

    @check_units
    def __init__(
        self,
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
        self.depth_ratio = depth_ratio
        self.n_crack = n_crack
        self.speed = speed
        self.speedI = speed
        self.speedF = speed
        self.unbalance_magnitude = unbalance_magnitude
        self.unbalance_phase = unbalance_phase
        self.print_progress = print_progress

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

        dir_path = Path(__file__).parents[2] / "tools/data/PAPADOPOULOS.csv"
        self.data_coefs = pd.read_csv(dir_path)

    def run(self, rotor):
        """Calculates the shaft angular position and the unbalance forces at X / Y directions.

        Parameters
        ----------
        rotor : ross.Rotor Object
             6 DoF rotor model.

        """

        self.rotor = rotor
        self.n_disk = len(self.rotor.disk_elements)
        if self.n_disk != len(self.unbalance_magnitude):
            raise Exception("The number of discs and unbalances must agree!")

        self.ndof = rotor.ndof
        self.L = rotor.elements[self.n_crack].L
        self.KK = rotor.elements[self.n_crack].K()
        self.radius = rotor.elements[self.n_crack].odl / 2
        self.Poisson = rotor.elements[self.n_crack].material.Poisson
        self.E = rotor.elements[self.n_crack].material.E
        self.ndofd = np.zeros(len(self.rotor.disk_elements))

        for ii in range(self.n_disk):
            self.ndofd[ii] = (self.rotor.disk_elements[ii].n) * 6

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

        if self.depth_ratio == 0:
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

        # parameters for the time integration
        self.lambdat = 0.00001
        Faxial = 0
        TorqueI = 0
        TorqueF = 0

        # pre-processing of auxilary variuables for the time integration
        self.sA = (
            self.speedI * np.exp(-self.lambdat * self.tF)
            - self.speedF * np.exp(-self.lambdat * self.tI)
        ) / (np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI))
        self.sB = (self.speedF - self.speedI) / (
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
        self.K = self.rotor.K(self.speed)
        self.C = self.rotor.C(self.speed)
        self.G = self.rotor.G()
        self.M = self.rotor.M()
        self.Kst = self.rotor.Kst()

        _, ModMat = scipy.linalg.eigh(
            self.K,
            self.M,
            type=1,
            turbo=False,
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
        # t_eval = np.arange(self.dt, self.tF, self.dt)
        T = t_eval

        self.angular_position = (
            self.sA * T
            - (self.sB / self.lambdat) * np.exp(-self.lambdat * T)
            + (self.sB / self.lambdat)
        )

        self.Omega = self.sA + self.sB * np.exp(-self.lambdat * T)
        self.AccelV = -self.lambdat * self.sB * np.exp(-self.lambdat * T)

        self.tetaUNB = np.zeros((len(self.unbalance_phase), len(self.angular_position)))
        unbx = np.zeros(len(self.angular_position))
        unby = np.zeros(len(self.angular_position))

        FFunb = np.zeros((self.ndof, len(t_eval)))
        self.forces_crack = np.zeros((self.ndof, len(t_eval)))

        for ii in range(self.n_disk):
            self.tetaUNB[ii, :] = (
                self.angular_position + self.unbalance_phase[ii] + np.pi / 2
            )

            unbx = self.unbalance_magnitude[ii] * (self.AccelV) * (
                np.cos(self.tetaUNB[ii, :])
            ) - self.unbalance_magnitude[ii] * ((self.Omega ** 2)) * (
                np.sin(self.tetaUNB[ii, :])
            )

            unby = -self.unbalance_magnitude[ii] * (self.AccelV) * (
                np.sin(self.tetaUNB[ii, :])
            ) - self.unbalance_magnitude[ii] * (self.Omega ** 2) * (
                np.cos(self.tetaUNB[ii, :])
            )

            FFunb[int(self.ndofd[ii]), :] += unbx
            FFunb[int(self.ndofd[ii] + 1), :] += unby

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
        """Calculates the displacement and velocity using state-space representation in the modal domain.

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
        F_CRACK : array
            Excitation force caused by the parallel misalignment on the node of application.
        FF_CRACK : array
            Excitation force caused by the parallel misalignment on the entire system.
        """

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
        self.KK_CRACK = KK_CRACK

        return FF_CRACK, F_CRACK

    def _gasch(self, ap):
        """Stiffness matrix of the cracked element according to the Gasch model.

        Paramenters
        -----------
        ap : float
            Angular position of the shaft.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of cracked element.
        """

        # Gasch
        kme = (self.ko + self.kcx) / 2
        kmn = (self.ko + self.kcz) / 2
        kde = (self.ko - self.kcx) / 2
        kdn = (self.ko - self.kcz) / 2

        size = 18
        cosine_sum = np.sum(
            [(-1) ** i * np.cos((2 * i + 1) * ap) / (2 * i + 1) for i in range(size)]
        )

        kee = kme + (4 / np.pi) * kde * cosine_sum
        knn = kmn + (4 / np.pi) * kdn * cosine_sum

        aux = np.array([[kee, 0], [0, knn]])

        K = ((self.T_matrix.T).dot(aux)).dot(self.T_matrix)

        return K

    def _mayes(self, ap):
        """Stiffness matrix of the cracked element according to the Mayes model.

        Paramenters
        -----------
        ap : float
            Angular position of the shaft.

        Returns
        -------
        K : np.ndarray
            Stiffness matrix of cracked element.
        """
        # Mayes

        kee = 0.5 * (self.ko + self.kcx) + 0.5 * (self.ko - self.kcx) * np.cos(ap)

        knn = 0.5 * (self.ko + self.kcz) + 0.5 * (self.ko - self.kcz) * np.cos(ap)

        aux = np.array([[kee, 0], [0, knn]])

        K = ((self.T_matrix.T).dot(aux)).dot(self.T_matrix)

        return K

    def _get_coefs(self, coef):
        """Terms os the compliance matrix.

        Paramenters
        -----------
        coef : string
            Name of the Coefficient according to the corresponding direction.

        Returns
        -------
        c : np.ndarray
            Compliance coefficient according to the crack depth.
        """

        c = np.array(pd.eval(self.data_coefs[coef]))
        aux = np.where(c[:, 1] >= self.depth_ratio * 2)[0]
        c = c[aux[0], 0] * (1 - self.Poisson ** 2) / (self.E * (self.radius ** 3))

        return c


def base_rotor_example():
    """Internal routine that create an example of a rotor, to be used in
    the associated crack problems as a prerequisite.

    This function returns an instance of a 6 DoF rotor, with a number of
    components attached. As this is not the focus of the example here, but
    only a requisite, see the example in "rotor assembly" for additional
    information on the rotor object.

    Returns
    -------
    rotor : ross.Rotor Object
        An instance of a flexible 6 DoF rotor object.

    Examples
    --------
    >>> rotor = base_rotor_example()
    >>> rotor.Ip
    0.015118294226367068
    """
    steel2 = ross.Material(name="Steel", rho=7850, E=2.17e11, G_s=81.2e9)
    #  Rotor with 6 DoFs, with internal damping, with 10 shaft elements, 2 disks and 2 bearings.
    i_d = 0
    o_d = 0.019
    n = 33

    # fmt: off
    L = np.array(
            [0  ,  25,  64, 104, 124, 143, 175, 207, 239, 271,
            303, 335, 345, 355, 380, 408, 436, 466, 496, 526,
            556, 586, 614, 647, 657, 667, 702, 737, 772, 807,
            842, 862, 881, 914]
            )/ 1000
    # fmt: on

    L = [L[i] - L[i - 1] for i in range(1, len(L))]

    shaft_elem = [
        ross.ShaftElement6DoF(
            material=steel2,
            L=l,
            idl=i_d,
            odl=o_d,
            idr=i_d,
            odr=o_d,
            alpha=8.0501,
            beta=1.0e-5,
            rotary_inertia=True,
            shear_effects=True,
        )
        for l in L
    ]

    Id = 0.003844540885417
    Ip = 0.007513248437500

    disk0 = ross.DiskElement6DoF(n=12, m=2.6375, Id=Id, Ip=Ip)
    disk1 = ross.DiskElement6DoF(n=24, m=2.6375, Id=Id, Ip=Ip)

    kxx1 = 4.40e5
    kyy1 = 4.6114e5
    kzz = 0
    cxx1 = 27.4
    cyy1 = 2.505
    czz = 0
    kxx2 = 9.50e5
    kyy2 = 1.09e8
    cxx2 = 50.4
    cyy2 = 100.4553

    bearing0 = ross.BearingElement6DoF(
        n=4, kxx=kxx1, kyy=kyy1, cxx=cxx1, cyy=cyy1, kzz=kzz, czz=czz
    )
    bearing1 = ross.BearingElement6DoF(
        n=31, kxx=kxx2, kyy=kyy2, cxx=cxx2, cyy=cyy2, kzz=kzz, czz=czz
    )

    rotor = ross.Rotor(shaft_elem, [disk0, disk1], [bearing0, bearing1])

    return rotor


def crack_example():
    """Create an example to evaluate the influence of transverse cracks in a rotating shaft.

    This function returns an instance of a transversal crack
    defect. The purpose is to make available a simple model so that a
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

    rotor = base_rotor_example()

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
