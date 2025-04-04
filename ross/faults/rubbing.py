from abc import ABC

import numpy as np

import ross as rs
from ross.units import Q_, check_units

__all__ = [
    "Rubbing",
]


class Rubbing(ABC):
    """Model rubbing based on Finite Element Method on a given shaft element of a
    rotor system.

    Contains a rubbing model :cite:`yamamoto2002linear`.
    The reference coordenate system is:
        - x-axis and y-axis in the sensors' planes;
        - z-axis throught the shaft center.

    Parameters
    ----------
    rotor : ross.Rotor
        Rotor object.
    n : int
        Number of shaft element where rubbing is ocurring.
    distance : float, pint.Quantity
        Distance between the housing and shaft surface.
    contact_stiffness : float, pint.Quantity
        Contact stiffness.
    contact_damping : float, pint.Quantity
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
    shaft_elem : ross.ShaftElement
        A 6 degrees of freedom shaft element object where rubbing is ocurring.
    forces : np.ndarray
        Force matrix due to rubbing. Each row corresponds to a dof and each column to a time.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> rotor = rs.rotor_example_with_damping()
    >>> fault = Rubbing(
    ...     rotor,
    ...     n=12,
    ...     distance=7.95e-5,
    ...     contact_stiffness=1.1e6,
    ...     contact_damping=40,
    ...     friction_coeff=0.3
    ... )
    >>> fault.shaft_elem
    ShaftElement(L=0.01, idl=0.0, idr=0.0, odl=0.019,  odr=0.019, material='Steel', n=12)
    """

    @check_units
    def __init__(
        self,
        rotor,
        n,
        distance,
        contact_stiffness,
        contact_damping,
        friction_coeff,
        torque=False,
    ):
        self.rotor = rotor
        self.delta = distance
        self.contact_stiffness = contact_stiffness
        self.contact_damping = contact_damping
        self.friction_coeff = friction_coeff
        self.torque = torque

        # Shaft element with rubbing
        self.shaft_elem = [elm for elm in rotor.shaft_elements if elm.n == n][0]

        self.dofs = list(self.shaft_elem.dof_global_index.values())

    def compute_rubbing_force(self, disp_resp, velc_resp, ang_speed):
        """Calculate the force on the shaft element with rubbing.

        Parameters
        ----------
        disp_resp : np.ndarray
            Displacement response of the element.
        velc_resp : np.ndarray
            Velocity response of the element.
        ang_speed : float
            Angular speed of the element.

        Returns
        -------
        F : np.ndarray
            Force matrix of the element due to rubbing.
        """
        radius = self.shaft_elem.odl / 2

        F_k = np.zeros_like(disp_resp)
        F_c = np.zeros_like(disp_resp)
        F_f = np.zeros_like(disp_resp)

        radial_disp = np.sqrt(disp_resp[0] ** 2 + disp_resp[1] ** 2)
        radial_velc = np.sqrt(velc_resp[0] ** 2 + velc_resp[1] ** 2)

        if radial_disp >= self.delta:
            F_k[0:2] = (
                -self.contact_stiffness
                * disp_resp[0:2]
                * (radial_disp - self.delta)
                / abs(radial_disp)
            )
            F_c[0:2] = (
                -self.contact_damping * velc_resp[0:2] * radial_velc / abs(radial_velc)
            )

            F_f[0:2] = self.friction_coeff * abs(F_k[0:2] + F_c[0:2])

            phi = np.arctan2(disp_resp[1], disp_resp[0])
            velc_t = velc_resp[0] * np.cos(phi) - velc_resp[1] * np.sin(phi)
            velc = velc_t + ang_speed * radius

            if velc > 0:
                F_f[0] = -F_f[0]
            else:
                F_f[1] = -F_f[1]

            if self.torque:
                F_f[5] = (
                    radius
                    * np.sqrt(F_f[0] ** 2 + F_f[1] ** 2)
                    * disp_resp[0]
                    / abs(radial_disp)
                )

        F = F_k + F_c + F_f

        return F

    def _get_force_over_time(self, step, disp_resp, velc_resp, speed):
        """Calculate the dynamic force on given time step.

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
        F : np.ndarray
            Force matrix due to rubbing in the current time step `t[step]`.
        """

        F = np.zeros(self.rotor.ndof)
        F[self.dofs] = self.compute_rubbing_force(
            disp_resp[self.dofs], velc_resp[self.dofs], speed
        )
        self.forces[:, step] = F

        return F

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

        # Unbalance force
        F, _, speed, _ = rotor._unbalance_force_over_time(
            node, unb_magnitude, unb_phase, speed, t
        )

        self.forces = np.zeros((rotor.ndof, len(t)))

        force_rubbing = lambda step, **state: self._get_force_over_time(
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

    This function returns time response results of a rubbing fault.
    The purpose is to make available a simple model so that a doctest
    can be written using it.

    Returns
    -------
    results : ross.TimeResponseResults
        Results for a shaft with rubbing.

    Examples
    --------
    >>> from ross.faults.rubbing import rubbing_example
    >>> from ross.probe import Probe
    >>> results = rubbing_example()
    Running direct method
    >>> probe1 = Probe(14, 0)
    >>> probe2 = Probe(22, 0)
    >>> fig = results.plot_1d([probe1, probe2])
    """

    rotor = rs.rotor_example_with_damping()

    n1 = rotor.disk_elements[0].n
    n2 = rotor.disk_elements[1].n

    results = rotor.run_rubbing(
        n=12,
        distance=7.95e-5,
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
