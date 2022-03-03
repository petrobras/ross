"""Misalignment module.

This module defines misalignments of various types on the shaft coupling. There are 
a number of options, for the formulation of 6 DoFs (degrees of freedom).
"""
import time

import numpy as np
import scipy as sp
import scipy.integrate
import scipy.linalg

import ross
from ross.units import Q_, check_units

from .abs_defect import Defect
from .integrate_solver import Integrator

__all__ = ["MisalignmentFlex", "MisalignmentRigid"]


class MisalignmentFlex(Defect):
    """A flexible coupling with misalignment of some kind.

    Calculates the dynamic reaction force of hexangular flexible coupling
    induced by 6DOF's rotor parallel and angular misalignment.

    Parameters
    ----------
    dt : float
        Time step.
    tI : float
        Initial time.
    tF : float
        Final time.
    kd : float
        Radial stiffness of flexible coupling.
    ks : float
        Bending stiffness of flexible coupling.
    eCOUPx : float
        Parallel misalignment offset between driving rotor and driven rotor along X direction.
    eCOUPy : float
        Parallel misalignment offset between driving rotor and driven rotor along Y direction.
    misalignment_angle : float
        Angular misalignment angle.
    TD : float
        Driving torque.
    TL : float
        Driven torque.
    n1 : float
        Node where the misalignment is ocurring.
    speed : float, pint.Quantity
        Operational speed of the machine. Default unit is rad/s.
    unbalance_magnitude : array
        Array with the unbalance magnitude. The unit is kg.m.
    unbalance_phase : array
        Array with the unbalance phase. The unit is rad.
    mis_type: string
        String containing the misalignment type choosed. The avaible types are: parallel, by default; angular; combined.
    print_progress : bool
        Set it True, to print the time iterations and the total time spent.
        False by default.

    Returns
    -------
    A force to be applied on the shaft.

    References
    ----------
    .. [1] 'Xia, Y., Pang, J., Yang, L., Zhao, Q., & Yang, X. (2019). Study on vibration response
    and orbits of misaligned rigid rotors connected by hexangular flexible coupling. Applied
    Acoustics, 155, 286-296 ..

    Examples
    --------
    >>> from ross.defects.misalignment import misalignment_flex_parallel_example
    >>> probe1 = (14, 0)
    >>> probe2 = (22, 0)
    >>> response = misalignment_flex_parallel_example()
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
        kd,
        ks,
        eCOUPx,
        eCOUPy,
        misalignment_angle,
        TD,
        TL,
        n1,
        speed,
        unbalance_magnitude,
        unbalance_phase,
        mis_type,
        print_progress=False,
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
        self.unbalance_magnitude = unbalance_magnitude
        self.unbalance_phase = unbalance_phase

        self.speedI = speed
        self.speedF = speed

        self.mis_type = mis_type
        self.print_progress = print_progress

        if self.mis_type is None or self.mis_type == "parallel":
            self._force = self._parallel
        elif self.mis_type == "angular":
            self._force = self._angular
        elif self.mis_type == "combined":
            self._force = self._combined
        else:
            raise Exception("Check the misalignment type!")

        if len(self.unbalance_magnitude) != len(self.unbalance_phase):
            raise Exception(
                "The unbalance magnitude vector and phase must have the same size!"
            )

    def run(self, rotor):
        """Calculates the shaft angular position and the misalignment amount at X / Y directions.

        Parameters
        ----------
        rotor : ross.Rotor Object
             6 DoF rotor model.

        """
        self.rotor = rotor
        self.n_disk = len(self.rotor.disk_elements)
        if self.n_disk != len(self.unbalance_magnitude):
            raise Exception("The number of discs and unbalances must agree!")

        self.radius = rotor.elements[self.n1].odl / 2
        self.ndof = rotor.ndof
        self.ndofd = np.zeros(len(self.rotor.disk_elements))

        for ii in range(self.n_disk):
            self.ndofd[ii] = (self.rotor.disk_elements[ii].n) * 6

        self.Cte = (
            self.ks * self.radius * np.sqrt(2 - 2 * np.cos(self.misalignment_angle))
        )

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

        # This code below here is used for acceleration and torque application to the rotor. As of
        # september/2020 it is unused, but might be implemented in future releases. These would be
        # run-up and run-down operations and variations of operating conditions.
        #
        # sAT = (
        #     TorqueI * np.exp(-lambdat * self.tF) - TorqueF * np.exp(-lambdat * self.tI)
        # ) / (np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI))
        # sBT = (TorqueF - TorqueI) / (
        #     np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI)
        # )
        #
        # SpeedV = sA + sB * np.exp(-lambdat * self.t)
        # TorqueV = sAT + sBT * np.exp(-lambdat * self.t)
        # AccelV = -lambdat * sB * np.exp(-lambdat * self.t)

        # Determining the modal matrix
        self.K = self.rotor.K(self.speed)
        self.C = self.rotor.C(self.speed)
        self.G = self.rotor.G()
        self.M = self.rotor.M()
        self.Kst = self.rotor.Kst()

        _, ModMat = scipy.linalg.eigh(self.K, self.M, type=1, turbo=False)
        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(self.M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(self.C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(self.G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(self.K)).dot(ModMat)
        self.Kstmodal = ((ModMat.T).dot(self.Kst)).dot(ModMat)

        # Omega = self.speedI * np.pi / 30

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

        self.tetaUNB = np.zeros((len(self.unbalance_phase), len(self.angular_position)))
        unbx = np.zeros(len(self.angular_position))
        unby = np.zeros(len(self.angular_position))

        FFunb = np.zeros((self.ndof, len(t_eval)))

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
        self.forces = self._force(self.angular_position)

        self.ft_modal = (self.ModMat.T).dot(self.forces).T

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
        velocity = Y[12:]  # velocity ign space state

        ftmodal = self.ft_modal[i]

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

    def _parallel(self, angular_position):
        """Reaction forces of parallel misalignment.

        angular_position : float
                        Angular position of the shaft.

        Returns
        -------
        F_mis_p : array
               Excitation force caused by the parallel misalignment on the entire system.
        """

        F_mis_p = np.zeros((self.ndof, len(angular_position)))

        fib = np.arctan(self.eCOUPy / self.eCOUPx)

        self.mi_y = (
            (
                np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(fib + angular_position)
                )
                - self.radius
            )
            * np.cos(angular_position)
            + (
                np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + angular_position)
                )
                - self.radius
            )
            * np.cos(2 * np.pi / 3 + angular_position)
            + (
                self.radius
                - np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    - 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + angular_position)
                )
            )
            * np.cos(4 * np.pi / 3 + angular_position)
        )

        self.mi_x = (
            (
                np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(fib + angular_position)
                )
                - self.radius
            )
            * np.sin(angular_position)
            + (
                np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    + 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.cos(np.pi / 6 + fib + angular_position)
                )
                - self.radius
            )
            * np.sin(2 * np.pi / 3 + angular_position)
            + (
                self.radius
                - np.sqrt(
                    self.radius ** 2
                    + self.eCOUPx ** 2
                    + self.eCOUPy ** 2
                    - 2
                    * self.radius
                    * np.sqrt(self.eCOUPx ** 2 + self.eCOUPy ** 2)
                    * np.sin(np.pi / 3 + fib + angular_position)
                )
            )
            * np.sin(4 * np.pi / 3 + angular_position)
        )

        Fpy = self.kd * self.mi_y

        Fpx = self.kd * self.mi_x

        F_mis_p[0 + 6 * self.n1] = Fpx
        F_mis_p[1 + 6 * self.n1] = Fpy
        F_mis_p[5 + 6 * self.n1] = self.TD
        F_mis_p[0 + 6 * self.n2] = -Fpx
        F_mis_p[1 + 6 * self.n2] = -Fpy
        F_mis_p[5 + 6 * self.n2] = self.TL

        return F_mis_p

    def _angular(self, angular_position):
        """Reaction forces of angular misalignment.

        angular_position : float
                Angular position of the shaft.

        Returns
        -------
        F_mis_a : array
            Excitation force caused by the parallel misalignment on the entire system.
        """
        F_mis_a = np.zeros((self.ndof, len(angular_position)))

        Fay = (
            np.abs(
                self.Cte * np.sin(angular_position) * np.sin(self.misalignment_angle)
            )
            * np.sin(angular_position + np.pi)
            + np.abs(
                self.Cte
                * np.sin(angular_position + 2 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.sin(angular_position + np.pi + 2 * np.pi / 3)
            + np.abs(
                self.Cte
                * np.sin(angular_position + 4 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.sin(angular_position + np.pi + 4 * np.pi / 3)
        )

        Fax = (
            np.abs(
                self.Cte * np.sin(angular_position) * np.sin(self.misalignment_angle)
            )
            * np.cos(angular_position + np.pi)
            + np.abs(
                self.Cte
                * np.sin(angular_position + 2 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.cos(angular_position + np.pi + 2 * np.pi / 3)
            + np.abs(
                self.Cte
                * np.sin(angular_position + 4 * np.pi / 3)
                * np.sin(self.misalignment_angle)
            )
            * np.cos(angular_position + np.pi + 4 * np.pi / 3)
        )

        F_mis_a[0 + 6 * self.n1] = Fax
        F_mis_a[1 + 6 * self.n1] = Fay
        F_mis_a[5 + 6 * self.n1] = self.TD
        F_mis_a[0 + 6 * self.n2] = -Fax
        F_mis_a[1 + 6 * self.n2] = -Fay
        F_mis_a[5 + 6 * self.n2] = self.TL

        return F_mis_a

    def _combined(self, angular_position):
        """Reaction forces of combined (parallel and angular) misalignment.

        angular_position : float
                Angular position of the shaft.

        Returns
        -------
        F_misalign : array
            Excitation force caused by the parallel misalignment on the entire system.
        """
        F_misalign = self._parallel(angular_position) + self._angular(angular_position)
        return F_misalign


class MisalignmentRigid(Defect):
    """A rigid coupling with parallel misalignment.

    Calculates the dynamic reaction force of hexangular rigid coupling
    induced by 6DOF's rotor parallel misalignment.

    Parameters
    ----------
    dt : float
        Time step.
    tI : float
        Initial time.
    tF : float
        Final time.
    eCOUP : float
        Parallel misalignment offset between driving rotor and driven rotor along X direction.
    TD : float
        Driving torque.
    TL : float
        Driven torque.
    n1 : float
        Node where the misalignment is ocurring.
    speed : float, pint.Quantity
        Operational speed of the machine. Default unit is rad/s.
    unbalance_magnitude : array
        Array with the unbalance magnitude. The unit is kg.m.
    unbalance_phase : array
        Array with the unbalance phase. The unit is rad.
    print_progress : bool
        Set it True, to print the time iterations and the total time spent.
        False by default.

    Returns
    -------
    A force to be applied on the shaft.

    References
    ----------

    .. [1] 'Al-Hussain, K. M., & Redmond, I. (2002). Dynamic response of two rotors connected by rigid mechanical coupling with parallel misalignment. Journal of Sound and vibration, 249(3), 483-498..

    Examples
    --------
    >>> from ross.defects.misalignment import misalignment_rigid_example
    >>> probe1 = (14, 0)
    >>> probe2 = (22, 0)
    >>> response = misalignment_rigid_example()
    >>> results = response.run_time_response()
    >>> fig = response.plot_dfft(probe=[probe1, probe2], range_freq=[0, 100], yaxis_type="log")
    >>> # fig.show
    """

    @check_units
    def __init__(
        self,
        dt,
        tI,
        tF,
        eCOUP,
        TD,
        TL,
        n1,
        speed,
        unbalance_magnitude,
        unbalance_phase,
        print_progress=False,
    ):
        self.dt = dt
        self.tI = tI
        self.tF = tF
        self.eCOUP = eCOUP
        self.TD = TD
        self.TL = TL
        self.n1 = n1
        self.n2 = n1 + 1
        self.speed = speed
        self.speedI = speed
        self.speedF = speed
        self.unbalance_magnitude = unbalance_magnitude
        self.unbalance_phase = unbalance_phase
        self.DoF = np.arange((self.n1 * 6), (self.n2 * 6 + 6))
        self.print_progress = print_progress

        if len(self.unbalance_magnitude) != len(self.unbalance_phase):
            raise Exception(
                "The unbalance magnitude vector and phase must have the same size!"
            )

    def run(self, rotor):
        """Calculates the shaft angular position and the misalignment amount at X directions.

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
        self.ndofd = np.zeros(len(self.rotor.disk_elements))

        for ii in range(self.n_disk):
            self.ndofd[ii] = (self.rotor.disk_elements[ii].n) * 6

        self.lambdat = 0.00001
        # Faxial = 0
        # TorqueI = 0
        # TorqueF = 0

        # pre-processing of auxilary variuables for the time integration
        self.sA = (
            self.speedI * np.exp(-self.lambdat * self.tF)
            - self.speedF * np.exp(-self.lambdat * self.tI)
        ) / (np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI))
        self.sB = (self.speedF - self.speedI) / (
            np.exp(-self.lambdat * self.tF) - np.exp(-self.lambdat * self.tI)
        )

        # This code below here is used for acceleration and torque application to the rotor. As of
        # september/2020 it is unused, but might be implemented in future releases. These would be
        # run-up and run-down operations and variations of operating conditions.
        #
        # sAT = (
        #     TorqueI * np.exp(-lambdat * self.tF) - TorqueF * np.exp(-lambdat * self.tI)
        # ) / (np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI))
        # sBT = (TorqueF - TorqueI) / (
        #     np.exp(-lambdat * self.tF) - np.exp(-lambdat * self.tI)
        # )
        #
        # self.SpeedV = sA + sB * np.exp(-lambdat * t)
        # self.TorqueV = sAT + sBT * np.exp(-lambdat * t)
        # self.AccelV = -lambdat * sB * np.exp(-lambdat * t)

        # Determining the modal matrix
        self.K = self.rotor.K(self.speed)
        self.C = self.rotor.C(self.speed)
        self.G = self.rotor.G()
        self.M = self.rotor.M()
        self.Kst = self.rotor.Kst()

        _, ModMat = scipy.linalg.eigh(self.K, self.M, type=1, turbo=False)
        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(self.M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(self.C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(self.G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(self.K)).dot(ModMat)
        self.Kstmodal = ((ModMat.T).dot(self.Kst)).dot(ModMat)

        self.angANG = -np.pi / 180

        self.kcoup_auxt = 1 / (
            self.K[5 + 6 * self.n1, 5 + 6 * self.n1]
            + self.K[5 + 6 * self.n2, 5 + 6 * self.n2]
        )

        self.kCOUP = (
            self.K[6 * self.n1, 6 * self.n1] * self.K[6 * self.n2, 6 * self.n2]
        ) / (self.K[6 * self.n1, 6 * self.n1] + self.K[6 * self.n2, 6 * self.n2])

        self.Kcoup_auxI = self.K[5 + 6 * self.n1, 5 + 6 * self.n1] / (
            self.K[5 + 6 * self.n1, 5 + 6 * self.n1]
            + self.K[5 + 6 * self.n2, 5 + 6 * self.n2]
        )

        self.Kcoup_auxF = self.K[5 + 6 * self.n2, 5 + 6 * self.n2] / (
            self.K[5 + 6 * self.n1, 5 + 6 * self.n1]
            + self.K[5 + 6 * self.n2, 5 + 6 * self.n2]
        )

        FFmis = np.zeros(self.ndof)

        # Omega = self.speedI * np.pi / 30

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

        self.tetaUNB = np.zeros((len(self.unbalance_phase), len(self.angular_position)))
        unbx = np.zeros(len(self.angular_position))
        unby = np.zeros(len(self.angular_position))

        FFunb = np.zeros((self.ndof, len(t_eval)))

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
        self.forces = np.zeros((self.ndof, len(t_eval)))

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
        velocity = Y[12:]  # velocity ign space state

        positionsFis = self.ModMat.dot(positions)

        self.angANG = (
            self.Kcoup_auxI * self.angular_position[i]
            + self.Kcoup_auxF * self.angular_position[i]
            + self.kCOUP
            * self.kcoup_auxt
            * self.eCOUP
            * (
                -positionsFis[0 + 6 * self.n1] * np.sin(self.angANG)
                + positionsFis[0 + 6 * self.n2] * np.sin(self.angANG)
                + positionsFis[1 + 6 * self.n1] * np.cos(self.angANG)
                - positionsFis[1 + 6 * self.n2] * np.cos(self.angANG)
            )
        )

        Fmis, ft = self._parallel(positionsFis, self.angANG)
        self.forces[:, i] = ft
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

    def _parallel(self, positions, fir):
        """Reaction forces of parallel misalignment.

        Returns
        -------
        Fmis : array
            Excitation force caused by the parallel misalignment on the node of application.
        FFmis : array
            Excitation force caused by the parallel misalignment on the entire system.
        """
        k0 = self.kCOUP
        delta1 = self.eCOUP

        betam = 0

        k_misalignbeta1 = np.array(
            [
                k0 * self.Kcoup_auxI * delta1 * np.sin(betam + fir),
                -k0 * self.Kcoup_auxI * delta1 * np.cos(betam + fir),
                0,
                0,
                0,
                0,
                k0 * self.Kcoup_auxF * delta1 * np.sin(betam + fir),
                -k0 * self.Kcoup_auxF * delta1 * np.cos(betam + fir),
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
                (-k0 * delta1 * np.cos(betam + fir) + k0 * delta1),
                -k0 * delta1 * np.sin(betam + fir),
                0,
                0,
                0,
                self.TD - self.TL,
                (k0 * delta1 * np.cos(betam + fir) - k0 * delta1),
                k0 * delta1 * np.sin(betam + fir),
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


def base_rotor_example():
    """Internal routine that create an example of a rotor, to be used in
    the associated misalignment problems as a prerequisite.

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


def misalignment_flex_parallel_example():
    """Create an example of a flexible parallel misalignment defect.

    This function returns an instance of a flexible parallel misalignment
    defect. The purpose is to make available a simple model so that a
    doctest can be written using it.

    Returns
    -------
    misalignment : ross.MisalignmentFlex Object
        An instance of a flexible parallel misalignment model object.

    Examples
    --------
    >>> misalignment = misalignment_flex_parallel_example()
    >>> misalignment.speed
    125.66370614359172
    """

    rotor = base_rotor_example()

    misalignment = rotor.run_misalignment(
        coupling="flex",
        dt=0.0001,
        tI=0,
        tF=0.5,
        kd=40 * 10 ** (3),
        ks=38 * 10 ** (3),
        eCOUPx=2 * 10 ** (-4),
        eCOUPy=2 * 10 ** (-4),
        misalignment_angle=5 * np.pi / 180,
        TD=0,
        TL=0,
        n1=0,
        speed=Q_(1200, "RPM"),
        unbalance_magnitude=np.array([5e-4, 0]),
        unbalance_phase=np.array([-np.pi / 2, 0]),
        mis_type="parallel",
        print_progress=False,
    )

    return misalignment


def misalignment_flex_angular_example():
    """Create an example of a flexible angular misalignment defect.

    This function returns an instance of a flexible angular misalignment
    defect. The purpose is to make available a simple model so that a
    doctest can be written using it.

    Returns
    -------
    misalignment : ross.MisalignmentFlex Object
        An instance of a flexible Angular misalignment model object.

    Examples
    --------
    >>> misalignment = misalignment_flex_angular_example()
    >>> misalignment.speed
    125.66370614359172
    """

    rotor = base_rotor_example()

    misalignment = rotor.run_misalignment(
        coupling="flex",
        dt=0.0001,
        tI=0,
        tF=0.5,
        kd=40 * 10 ** (3),
        ks=38 * 10 ** (3),
        eCOUPx=2 * 10 ** (-4),
        eCOUPy=2 * 10 ** (-4),
        misalignment_angle=5 * np.pi / 180,
        TD=0,
        TL=0,
        n1=0,
        speed=Q_(1200, "RPM"),
        unbalance_magnitude=np.array([5e-4, 0]),
        unbalance_phase=np.array([-np.pi / 2, 0]),
        mis_type="angular",
        print_progress=False,
    )

    return misalignment


def misalignment_flex_combined_example():
    """Create an example of a flexible combined misalignment defect.

    This function returns an instance of a flexible combined misalignment
    defect. The purpose is to make available a simple model so that a
    doctest can be written using it.

    Returns
    -------
    misalignment : ross.MisalignmentFlex Object
        An instance of a flexible combined misalignment model object.

    Examples
    --------
    >>> misalignment = misalignment_flex_combined_example()
    >>> misalignment.speed
    125.66370614359172
    """

    rotor = base_rotor_example()

    misalignment = rotor.run_misalignment(
        coupling="flex",
        dt=0.0001,
        tI=0,
        tF=0.5,
        kd=40 * 10 ** (3),
        ks=38 * 10 ** (3),
        eCOUPx=2 * 10 ** (-4),
        eCOUPy=2 * 10 ** (-4),
        misalignment_angle=5 * np.pi / 180,
        TD=0,
        TL=0,
        n1=0,
        speed=Q_(1200, "RPM"),
        unbalance_magnitude=np.array([5e-4, 0]),
        unbalance_phase=np.array([-np.pi / 2, 0]),
        mis_type="combined",
        print_progress=False,
    )

    return misalignment


def misalignment_rigid_example():
    """Create an example of a rigid misalignment defect.

    This function returns an instance of a rigid misalignment
    defect. The purpose is to make available a simple model so that a
    doctest can be written using it.

    Returns
    -------
    misalignment : ross.MisalignmentRigid Object
        An instance of a rigid misalignment model object.

    Examples
    --------
    >>> misalignment = misalignment_rigid_example()
    >>> misalignment.speed
    125.66370614359172
    """

    rotor = base_rotor_example()

    misalignment = rotor.run_misalignment(
        coupling="rigid",
        dt=0.0001,
        tI=0,
        tF=0.5,
        eCOUP=2e-4,
        TD=0,
        TL=0,
        n1=0,
        speed=Q_(1200, "RPM"),
        unbalance_magnitude=np.array([5e-4, 0]),
        unbalance_phase=np.array([-np.pi / 2, 0]),
        print_progress=False,
    )

    return misalignment
