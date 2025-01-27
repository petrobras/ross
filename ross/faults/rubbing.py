import time

import numpy as np
from scipy import linalg as la

import ross as rs
from ross.units import Q_, check_units

from .abs_fault import Fault
from .integrate_solver import Integrator

__all__ = [
    "Rubbing",
]


class Rubbing(Fault):
    """Contains a rubbing model for applications on finite element models of rotative machinery.
    The reference coordenates system is: z-axis throught the shaft center; x-axis and y-axis in the sensors' planes

    Parameters
    ----------
    dt : float
        Time step.
    tI : float
        Initial time.
    tF : float
        Final time.
    deltaRUB : float
        Distance between the housing and shaft surface.
    kRUB : float
        Contact stiffness.
    cRUB : float
        Contact damping.
    miRUB : float
        Friction coefficient.
    posRUB : int
        Node where the rubbing is ocurring.
    speed : float, pint.Quantity
        Operational speed of the machine. Default unit is rad/s.
    unbalance_magnitude : array
        Array with the unbalance magnitude. The unit is kg.m.
    unbalance_phase : array
        Array with the unbalance phase. The unit is rad.
    torque : bool
        Set it as True to consider the torque provided by the rubbing, by default False.
    print_progress : bool
        Set it True, to print the time iterations and the total time spent, by default False.

    Returns
    -------
    A force to be applied on the shaft.

    References
    ----------
    .. [1] Yamamoto, T., Ishida, Y., &Kirk, R.(2002). Linear and Nonlinear Rotordynamics: A Modern Treatment with Applications, pp. 215-222 ..

    Examples
    --------
    >>> from ross.probe import Probe
    >>> from ross.faults.rubbing import rubbing_example
    >>> probe1 = Probe(14, 0)
    >>> probe2 = Probe(22, 0)
    >>> response = rubbing_example()
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
        deltaRUB,
        kRUB,
        cRUB,
        miRUB,
        posRUB,
        speed,
        unbalance_magnitude,
        unbalance_phase,
        torque=False,
        print_progress=False,
    ):
        self.dt = dt
        self.tI = tI
        self.tF = tF
        self.deltaRUB = deltaRUB
        self.kRUB = kRUB
        self.cRUB = cRUB
        self.miRUB = miRUB
        self.posRUB = posRUB
        self.speed = speed
        self.speedI = speed
        self.speedF = speed
        self.DoF = np.arange((self.posRUB * 6), (self.posRUB * 6 + 6))
        self.torque = torque
        self.unbalance_magnitude = unbalance_magnitude
        self.unbalance_phase = unbalance_phase
        self.print_progress = print_progress

        if len(self.unbalance_magnitude) != len(self.unbalance_phase):
            raise Exception(
                "The unbalance magnitude vector and phase must have the same size!"
            )

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
        self.iteration = 0
        self.radius = rotor.df_shaft.iloc[self.posRUB].o_d / 2
        self.ndofd = np.zeros(len(self.rotor.disk_elements))

        for ii in range(self.n_disk):
            self.ndofd[ii] = (self.rotor.disk_elements[ii].n) * 6

        self.lambdat = 0.00001
        # Faxial = 0
        # TorqueI = 0
        # TorqueF = 0

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

        # self.SpeedV = sA + sB * np.exp(-lambdat * t)
        # self.TorqueV = sAT + sBT * np.exp(-lambdat * t)
        # self.AccelV = -lambdat * sB * np.exp(-lambdat * t)

        # Determining the modal matrix
        self.K = self.rotor.K(self.speed)
        self.C = self.rotor.C(self.speed)
        self.G = self.rotor.G()
        self.M = self.rotor.M(self.speed)
        self.Ksdt = self.rotor.Ksdt()

        V1, ModMat = la.eigh(self.K, self.M)

        ModMat = ModMat[:, :12]
        self.ModMat = ModMat

        # Modal transformations
        self.Mmodal = ((ModMat.T).dot(self.M)).dot(ModMat)
        self.Cmodal = ((ModMat.T).dot(self.C)).dot(ModMat)
        self.Gmodal = ((ModMat.T).dot(self.G)).dot(ModMat)
        self.Kmodal = ((ModMat.T).dot(self.K)).dot(ModMat)
        self.Ksdtmodal = ((ModMat.T).dot(self.Ksdt)).dot(ModMat)

        y0 = np.zeros(24)
        t_eval = np.arange(self.tI, self.tF + self.dt, self.dt)
        # t_eval = np.arange(self.tI, self.tF, self.dt)
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
        self.forces_rub = np.zeros((self.ndof, len(t_eval)))

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
        x = x.rk4()
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

        positionsFis = self.ModMat.dot(positions)
        velocityFis = self.ModMat.dot(velocity)

        Frub, ft = self._rub(positionsFis, velocityFis, self.Omega[i])
        self.forces_rub[:, i] = ft
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

    def _rub(self, positionsFis, velocityFis, ang):
        self.F_k = np.zeros(self.ndof)
        self.F_c = np.zeros(self.ndof)
        self.F_f = np.zeros(self.ndof)

        self.y = np.concatenate((positionsFis, velocityFis))

        ii = 0 + 6 * self.posRUB  # rubbing position

        self.radial_displ_node = np.sqrt(
            self.y[ii] ** 2 + self.y[ii + 1] ** 2
        )  # radial displacement
        self.radial_displ_vel_node = np.sqrt(
            self.y[ii + self.ndof] ** 2 + self.y[ii + 1 + self.ndof] ** 2
        )  # velocity
        self.phi_angle = np.arctan2(self.y[ii + 1], self.y[ii])

        if self.radial_displ_node >= self.deltaRUB:
            self.F_k[ii] = self._stiffness_force(self.y[ii])
            self.F_k[ii + 1] = self._stiffness_force(self.y[ii + 1])
            self.F_c[ii] = self._damping_force(self.y[ii + self.ndof])
            self.F_c[ii + 1] = self._damping_force(self.y[ii + 1 + self.ndof])

            Vt = -self.y[ii + self.ndof + 1] * np.sin(self.phi_angle) + self.y[
                ii + self.ndof
            ] * np.cos(self.phi_angle)

            if Vt + ang * self.radius > 0:
                self.F_f[ii] = -self._tangential_force(self.F_k[ii], self.F_c[ii])
                self.F_f[ii + 1] = self._tangential_force(
                    self.F_k[ii + 1], self.F_c[ii + 1]
                )

                if self.torque:
                    self.F_f[ii + 5] = self._torque_force(
                        self.F_f[ii], self.F_f[ii + 1], self.y[ii]
                    )
            elif Vt + ang * self.radius < 0:
                self.F_f[ii] = self._tangential_force(self.F_k[ii], self.F_c[ii])
                self.F_f[ii + 1] = -self._tangential_force(
                    self.F_k[ii + 1], self.F_c[ii + 1]
                )

                if self.torque:
                    self.F_f[ii + 5] = self._torque_force(
                        self.F_f[ii], self.F_f[ii + 1], self.y[ii]
                    )

        return self._combine_forces(self.F_k, self.F_c, self.F_f)

    def _stiffness_force(self, y):
        """Calculates the stiffness force

        Parameters
        ----------
        y : float
            Displacement value.

        Returns
        -------
        force : numpy.float64
            Force magnitude.
        """
        force = (
            -self.kRUB
            * (self.radial_displ_node - self.deltaRUB)
            * y
            / abs(self.radial_displ_node)
        )
        return force

    def _damping_force(self, y):
        """Calculates the damping force

        Parameters
        ----------
        y : float
            Displacement value.

        Returns
        -------
        force : numpy.float64
            Force magnitude.
        """
        force = (
            -self.cRUB
            * (self.radial_displ_vel_node)
            * y
            / abs(self.radial_displ_vel_node)
        )
        return force

    def _tangential_force(self, F_k, F_c):
        """Calculates the tangential force

        Parameters
        ----------
        y : float
            Displacement value.

        Returns
        -------
        force : numpy.float64
            Force magnitude.
        """
        force = self.miRUB * (abs(F_k + F_c))
        return force

    def _torque_force(self, F_f, F_fp, y):
        """Calculates the torque force

        Parameters
        ----------
        y : float
            Displacement value.

        Returns
        -------
        force : numpy.float64
            Force magnitude.
        """
        force = self.radius * (
            np.sqrt(F_f**2 + F_fp**2) * y / abs(self.radial_displ_node)
        )
        return force

    def _combine_forces(self, F_k, F_c, F_f):
        """Mounts the final force vector.

        Parameters
        ----------
        F_k : numpy.ndarray
            Stiffness force vector.
        F_c : numpy.ndarray
            Damping force vector.
        F_f : numpy.ndarray
            Tangential force vector.

        Returns
        -------
        Frub : numpy.ndarray
            Final force vector for each degree of freedom.
        FFrub : numpy.ndarray
            Final force vector.
        """
        Frub = F_k[self.DoF] + F_c[self.DoF] + F_f[self.DoF]
        FFrub = F_k + F_c + F_f

        return Frub, FFrub

    @property
    def forces(self):
        pass


def rubbing_example():
    """Create an example of a rubbing fault.

    This function returns an instance of a rubbing fault. The purpose is to make
    available a simple model so that a doctest can be written using it.

    Returns
    -------
    rubbing : ross.Rubbing Object
        An instance of a rubbing model object.

    Examples
    --------
    >>> rubbing = rubbing_example()
    >>> rubbing.speed
    125.66370614359172
    """

    rotor = rs.rotor_example_with_damping()

    rubbing = rotor.run_rubbing(
        dt=0.0001,
        tI=0,
        tF=0.5,
        deltaRUB=7.95e-5,
        kRUB=1.1e6,
        cRUB=40,
        miRUB=0.3,
        posRUB=12,
        speed=Q_(1200, "RPM"),
        unbalance_magnitude=np.array([5e-4, 0]),
        unbalance_phase=np.array([-np.pi / 2, 0]),
        torque=False,
        print_progress=False,
    )

    return rubbing
