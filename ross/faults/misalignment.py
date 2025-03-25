"""Misalignment module.

This module defines misalignments of various types on the shaft coupling. There are
a number of options, for the formulation of 6 DoFs (degrees of freedom).
"""

import time

import numpy as np
from scipy import linalg as la

import ross as rs
from ross.units import Q_, check_units

from .fault import Fault
from .integrate_solver import Integrator

__all__ = ["MisalignmentFlex", "MisalignmentRigid"]


class MisalignmentFlex(Fault):
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
    mis_angle : float
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
    >>> from ross.probe import Probe
    >>> from ross.faults.misalignment import misalignment_flex_parallel_example
    >>> probe1 = Probe(14, 0)
    >>> probe2 = Probe(22, 0)
    >>> response = misalignment_flex_parallel_example()
    >>> results = response.run_time_response()
    >>> fig = response.plot_dfft(probe=[probe1, probe2], range_freq=[0, 100], yaxis_type="log")
    >>> # fig.show()
    """

    @check_units
    def __init__(
        self,
        rotor,
        n1,
        TD,
        TL,
        eCOUPx,
        eCOUPy,
        kd,
        ks,
        mis_angle,
        mis_type="parallel",
    ):
        self.rotor = rotor

        self.TD = TD
        self.TL = TL

        self.eCOUPx = eCOUPx
        self.eCOUPy = eCOUPy

        self.kd = kd
        self.ks = ks

        self.mis_angle = mis_angle

        if mis_type == "parallel":
            self._compute_reaction_forces = self._parallel
        elif mis_type == "angular":
            self._compute_reaction_forces = self._angular
        elif mis_type == "combined":
            self._compute_reaction_forces = self._combined
        else:
            raise Exception("Check the misalignment type!")

        # Shaft element with misalignment
        self.shaft_elem = [elm for elm in rotor.shaft_elements if elm.n == n1][0]

        self.dofs = list(self.shaft_elem.dof_global_index.values())
        self.radius = self.shaft_elem.odl / 2

    def _parallel(self, ang_pos):
        """Reaction forces of parallel misalignment.

        ang_pos : array_like
            Angular position of the shaft. Each value corresponds to a time.

        Returns
        -------
        F : np.ndarray
            Excitation force caused by the parallel misalignment on the entire system.
            Each row corresponds to a dof and each column to a time.
        """

        F = np.zeros((self.rotor.ndof, len(ang_pos)))

        fib = np.arctan(self.eCOUPx / self.eCOUPy)

        aux1 = self.radius**2 + self.eCOUPx**2 + self.eCOUPy**2
        aux2 = 2 * self.radius * np.sqrt(self.eCOUPx**2 + self.eCOUPy**2)

        Fpy = (
            (np.sqrt(aux1 + aux2 * np.sin(fib + ang_pos)) - self.radius)
            * np.cos(ang_pos)
            + (np.sqrt(aux1 + aux2 * np.cos(np.pi / 6 + fib + ang_pos)) - self.radius)
            * np.cos(2 * np.pi / 3 + ang_pos)
            + (self.radius - np.sqrt(aux1 - aux2 * np.sin(np.pi / 3 + fib + ang_pos)))
            * np.cos(4 * np.pi / 3 + ang_pos)
        ) * self.kd

        Fpx = (
            (np.sqrt(aux1 + aux2 * np.sin(fib + ang_pos)) - self.radius)
            * np.sin(ang_pos)
            + (np.sqrt(aux1 + aux2 * np.cos(np.pi / 6 + fib + ang_pos)) - self.radius)
            * np.sin(2 * np.pi / 3 + ang_pos)
            + (self.radius - np.sqrt(aux1 - aux2 * np.sin(np.pi / 3 + fib + ang_pos)))
            * np.sin(4 * np.pi / 3 + ang_pos)
        ) * self.kd

        F[self.dofs[0]] = Fpx
        F[self.dofs[1]] = Fpy

        F[self.dofs[6]] = -Fpx
        F[self.dofs[7]] = -Fpy

        F[self.dofs[5]] = self.TD
        F[self.dofs[11]] = self.TL

        return F

    def _angular(self, ang_pos):
        """Reaction forces of angular misalignment.

        ang_pos : array_like
            Angular position of the shaft. Each value corresponds to a time.

        Returns
        -------
        F : np.ndarray
            Excitation force caused by the angular misalignment on the entire system.
            Each row corresponds to a dof and each column to a time.
        """

        F = np.zeros((self.rotor.ndof, len(ang_pos)))

        cte = self.ks * self.radius * np.sqrt(2 - 2 * np.cos(self.mis_angle))

        Fay = (
            np.abs(cte * np.sin(ang_pos) * np.sin(self.mis_angle))
            * np.sin(ang_pos + np.pi)
            + np.abs(cte * np.sin(ang_pos + 2 * np.pi / 3) * np.sin(self.mis_angle))
            * np.sin(ang_pos + np.pi + 2 * np.pi / 3)
            + np.abs(cte * np.sin(ang_pos + 4 * np.pi / 3) * np.sin(self.mis_angle))
            * np.sin(ang_pos + np.pi + 4 * np.pi / 3)
        )

        Fax = (
            np.abs(cte * np.sin(ang_pos) * np.sin(self.mis_angle))
            * np.cos(ang_pos + np.pi)
            + np.abs(cte * np.sin(ang_pos + 2 * np.pi / 3) * np.sin(self.mis_angle))
            * np.cos(ang_pos + np.pi + 2 * np.pi / 3)
            + np.abs(cte * np.sin(ang_pos + 4 * np.pi / 3) * np.sin(self.mis_angle))
            * np.cos(ang_pos + np.pi + 4 * np.pi / 3)
        )

        F[self.dofs[0]] = Fax
        F[self.dofs[1]] = Fay

        F[self.dofs[6]] = -Fax
        F[self.dofs[7]] = -Fay

        F[self.dofs[5]] = self.TD
        F[self.dofs[11]] = self.TL

        return F

    def _combined(self, ang_pos):
        """Reaction forces of combined (parallel and angular) misalignment.

        ang_pos : array_like
            Angular position of the shaft. Each value corresponds to a time.

        Returns
        -------
        F : np.ndarray
            Excitation force caused by the combined misalignment on the entire system.
            Each row corresponds to a dof and each column to a time.
        """

        F = self._parallel(ang_pos) + self._angular(ang_pos)
        return F

    def run(self, node, unb_magnitude, unb_phase, speed, t, **kwargs):
        """Run analysis for the system with misalignment given an unbalance force.

        System time response is simulated.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        unb_magnitude : list, float
            Unbalance magnitude (kg.m).
        unb_phase : list, float
            Unbalance phase (rad).
        speed : float or array_like, pint.Quantity
            Rotor speed.
        t : array
            Time array.
        **kwargs : optional
            Additional keyword arguments can be passed to define the parameters
            of the Newmark method if it is used (e.g. gamma, beta, tol, ...).
            See `ross.utils.newmark` for more details.
            Other keyword arguments can also be passed to be used in numerical
            integration (e.g. num_modes).
            See `Rotor.integrate_system` for more details.

        Returns
        -------
        results : ross.TimeResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.TimeResponseResults`
        """

        rotor = self.rotor

        # Unbalance force
        F, ang_pos, _, _ = rotor._unbalance_force_over_time(
            node, unb_magnitude, unb_phase, speed, t
        )

        self.forces = self._compute_reaction_forces(ang_pos)

        F += self.forces

        results = rotor.run_time_response(
            speed=speed,
            F=F.T,
            t=t,
            method="newmark",
            **kwargs,
        )

        return results


class MisalignmentRigid(Fault):
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
    >>> from ross.probe import Probe
    >>> from ross.faults.misalignment import misalignment_rigid_example
    >>> probe1 = Probe(14, 0)
    >>> probe2 = Probe(22, 0)
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
        self.M = self.rotor.M(self.speed)
        self.Ksdt = self.rotor.Ksdt()

        _, ModMat = la.eigh(self.K, self.M)
        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(self.M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(self.C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(self.G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(self.K)).dot(ModMat)
        self.Ksdtmodal = ((ModMat.T).dot(self.Ksdt)).dot(ModMat)

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
            ) - self.unbalance_magnitude[ii] * (self.Omega**2) * (
                np.sin(self.tetaUNB[ii, :])
            )

            unby = -self.unbalance_magnitude[ii] * (self.AccelV) * (
                np.sin(self.tetaUNB[ii, :])
            ) - self.unbalance_magnitude[ii] * (self.Omega**2) * (
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
            - (self.Cmodal + self.Gmodal * self.Omega[i]).dot(velocity)
            - ((self.Kmodal + self.Ksdtmodal * self.AccelV[i]).dot(positions))
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


def misalignment_flex_parallel_example():
    """Create an example of a flexible parallel misalignment fault.

    This function returns an instance of a flexible parallel misalignment
    fault. The purpose is to make available a simple model so that a
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

    rotor = rs.rotor_example_with_damping()

    misalignment = rotor.run_misalignment(
        coupling="flex",
        dt=0.0001,
        tI=0,
        tF=0.5,
        kd=40 * 10 ** (3),
        ks=38 * 10 ** (3),
        eCOUPx=2 * 10 ** (-4),
        eCOUPy=2 * 10 ** (-4),
        mis_angle=5 * np.pi / 180,
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
    """Create an example of a flexible angular misalignment fault.

    This function returns an instance of a flexible angular misalignment
    fault. The purpose is to make available a simple model so that a
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

    rotor = rs.rotor_example_with_damping()

    misalignment = rotor.run_misalignment(
        coupling="flex",
        dt=0.0001,
        tI=0,
        tF=0.5,
        kd=40 * 10 ** (3),
        ks=38 * 10 ** (3),
        eCOUPx=2 * 10 ** (-4),
        eCOUPy=2 * 10 ** (-4),
        mis_angle=5 * np.pi / 180,
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
    """Create an example of a flexible combined misalignment fault.

    This function returns an instance of a flexible combined misalignment
    fault. The purpose is to make available a simple model so that a
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

    rotor = rs.rotor_example_with_damping()

    misalignment = rotor.run_misalignment(
        coupling="flex",
        dt=0.0001,
        tI=0,
        tF=0.5,
        kd=40 * 10 ** (3),
        ks=38 * 10 ** (3),
        eCOUPx=2 * 10 ** (-4),
        eCOUPy=2 * 10 ** (-4),
        mis_angle=5 * np.pi / 180,
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
    """Create an example of a rigid misalignment fault.

    This function returns an instance of a rigid misalignment
    fault. The purpose is to make available a simple model so that a
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

    rotor = rs.rotor_example_with_damping()

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
