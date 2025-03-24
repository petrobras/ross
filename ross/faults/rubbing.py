import numpy as np

import ross as rs
from ross.units import Q_, check_units

from .fault import Fault

__all__ = [
    "Rubbing",
]


class Rubbing(Fault):
    """Models rubbing based on Finite Element Method on a given shaft element of a
    rotor system.

    Contains a rubbing model :cite:`yamamoto2002linear`. The reference coordenate system is:
        - x-axis and y-axis in the sensors' planes;
        - z-axis throught the shaft center.

    Parameters
    ----------
    n_rub : int
        Number of shaft element where rubbing is ocurring.
    delta_rub : float
        Distance between the housing and shaft surface.
    contact_stiffness : float
        Contact stiffness.
    contact_damping : float
        Contact damping.
    friction_coeff : float
        Friction coefficient.
    torque : bool, optional
        If True a torque is considered by rubbing.
        Default is False.

    Returns
    -------
    A rubbing object.

    Attributes
    ----------
    shaft_element : ross.ShaftElement
        A 6 degrees of freedom shaft element object where rubbing is ocurring.
    forces : np.ndarray
        Force matrix of shape `(ndof, len(t))` for the rubbing.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> rotor = rs.rotor_example_with_damping()
    >>> fault = Rubbing(
    ...     rotor,
    ...     n_rub=12,
    ...     delta_rub=7.95e-5,
    ...     contact_stiffness=1.1e6,
    ...     contact_damping=40,
    ...     friction_coeff=0.3
    ... )
    >>> fault.shaft_element
    """

    @check_units
    def __init__(
        self,
        rotor,
        n_rub,
        delta_rub,
        contact_stiffness,
        contact_damping,
        friction_coeff,
        torque=False,
    ):
        self.rotor = rotor
        self.delta_rub = delta_rub
        self.contact_stiffness = contact_stiffness
        self.contact_damping = contact_damping
        self.friction_coeff = friction_coeff
        self.n_rub = n_rub
        self.torque = torque

        # Shaft element with rubbing
        self.shaft_element = [
            elm for elm in rotor.shaft_elements if elm.n == self.n_rub
        ][0]

        self.dof_rub = list(self.shaft_element.dof_global_index.values())

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
            -self.contact_stiffness
            * (self.radial_displ_node - self.delta_rub)
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
            -self.contact_damping
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
        force = self.friction_coeff * (abs(F_k + F_c))
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

        ii = 0 + 6 * self.n_rub  # rubbing position

        self.radial_displ_node = np.sqrt(
            self.y[ii] ** 2 + self.y[ii + 1] ** 2
        )  # radial displacement

        self.radial_displ_vel_node = np.sqrt(
            self.y[ii + ndof] ** 2 + self.y[ii + 1 + ndof] ** 2
        )  # velocity

        self.phi_angle = np.arctan2(self.y[ii + 1], self.y[ii])

        if self.radial_displ_node >= self.delta_rub:
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
    results : ross.TimeResponseResults
        Results for a shaft with rubbing.

    Examples
    --------
    >>> rubbing = rubbing_example()
    """

    rotor = rs.rotor_example_with_damping()

    n1 = rotor.disk_elements[0].n
    n2 = rotor.disk_elements[1].n

    results = rotor.run_rubbing(
        n_rub=12,
        delta_rub=7.95e-5,
        contact_stiffness=1.1e6,
        contact_damping=40,
        friction_coeff=0.3,
        torque=False,
        node=[n1, n2],
        unbalance_magnitude=[5e-4, 0],
        unbalance_phase=[-np.pi / 2, 0],
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
    )

    return results
