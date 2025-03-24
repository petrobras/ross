import numpy as np

import ross as rs
from ross.units import Q_, check_units

from .fault import Fault

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
        rotor,
        deltaRUB,
        kRUB,
        cRUB,
        miRUB,
        posRUB,
        torque=False,
    ):
        self.rotor = rotor
        self.deltaRUB = deltaRUB
        self.kRUB = kRUB
        self.cRUB = cRUB
        self.miRUB = miRUB
        self.posRUB = posRUB
        self.dof_rubbing = np.arange((self.posRUB * 6), (self.posRUB * 6 + 6))
        self.torque = torque

        # Shaft element with rubbing
        self.shaft_element = [
            elm for elm in rotor.shaft_elements if elm.n == self.posRUB
        ][0]

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
        radius = self.shaft_element.odl / 2

        force = radius * (np.sqrt(F_f**2 + F_fp**2) * y / abs(self.radial_displ_node))
        return force

    def _rub(self, positionsFis, velocityFis, ang):
        ndof = self.rotor.ndof
        radius = self.shaft_element.odl / 2

        self.F_k = np.zeros(ndof)
        self.F_c = np.zeros(ndof)
        self.F_f = np.zeros(ndof)

        self.y = np.concatenate((positionsFis, velocityFis))

        ii = 0 + 6 * self.posRUB  # rubbing position

        self.radial_displ_node = np.sqrt(
            self.y[ii] ** 2 + self.y[ii + 1] ** 2
        )  # radial displacement
        self.radial_displ_vel_node = np.sqrt(
            self.y[ii + ndof] ** 2 + self.y[ii + 1 + ndof] ** 2
        )  # velocity
        self.phi_angle = np.arctan2(self.y[ii + 1], self.y[ii])

        if self.radial_displ_node >= self.deltaRUB:
            self.F_k[ii] = self._stiffness_force(self.y[ii])
            self.F_k[ii + 1] = self._stiffness_force(self.y[ii + 1])
            self.F_c[ii] = self._damping_force(self.y[ii + ndof])
            self.F_c[ii + 1] = self._damping_force(self.y[ii + 1 + ndof])

            Vt = -self.y[ii + ndof + 1] * np.sin(self.phi_angle) + self.y[
                ii + ndof
            ] * np.cos(self.phi_angle)

            if Vt + ang * radius > 0:
                self.F_f[ii] = -self._tangential_force(self.F_k[ii], self.F_c[ii])
                self.F_f[ii + 1] = self._tangential_force(
                    self.F_k[ii + 1], self.F_c[ii + 1]
                )

                if self.torque:
                    self.F_f[ii + 5] = self._torque_force(
                        self.F_f[ii], self.F_f[ii + 1], self.y[ii]
                    )
            elif Vt + ang * radius < 0:
                self.F_f[ii] = self._tangential_force(self.F_k[ii], self.F_c[ii])
                self.F_f[ii + 1] = -self._tangential_force(
                    self.F_k[ii + 1], self.F_c[ii + 1]
                )

                if self.torque:
                    self.F_f[ii + 5] = self._torque_force(
                        self.F_f[ii], self.F_f[ii + 1], self.y[ii]
                    )

        return self.F_k + self.F_c + self.F_f

    def _force_in_time(self, step, disp_resp, velc_resp, speed):
        """Calculates the dynamic force on given time step.

        Paramenters
        -----------
        step : int
            Current time step index.
        disp_resp : np.ndarray
            Displacement response of the system at the current time step.
        velc_resp : np.ndarray
            Velocity response of the system at the current time step.
        speed : float
            Rotation speed of the shaft at the current time step.

        Returns
        -------
        F_rubbing : np.ndarray
            Force matrix related to rubbing in the current time step `t[step]`.
        """

        F_rubbing = self._rub(disp_resp, velc_resp, speed)

        self.forces[:, step] = F_rubbing

        return F_rubbing

    def run(self, node, unb_magnitude, unb_phase, speed, t, **kwargs):
        """Run analysis for the system with rubbing given an unbalance force.

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

        self.forces = np.zeros((rotor.ndof, len(t)))

        # Unbalance force
        F, _, speed, _ = rotor._unbalance_force_in_time(
            node, unb_magnitude, unb_phase, speed, t
        )

        force_rubbing = lambda step, **state: self._force_in_time(
            step, state.get("disp_resp"), state.get("velc_resp"), speed[step]
        )

        results = rotor.run_time_response(
            speed=speed,
            F=F.T,
            t=t,
            method="newmark",
            add_to_RHS=force_rubbing,
            **kwargs,
        )

        return results


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
