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
            self.compute_reaction_force = self._parallel
        elif mis_type == "angular":
            self.compute_reaction_force = self._angular
        elif mis_type == "combined":
            self.compute_reaction_force = self._combined
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

        self.forces = self.compute_reaction_force(ang_pos)

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
        rotor,
        n1,
        TD,
        TL,
        eCOUP,
    ):
        self.rotor = rotor

        self.eCOUP = eCOUP

        self.TD = TD
        self.TL = TL

        # Shaft element with misalignment
        self.shaft_elem = [elm for elm in rotor.shaft_elements if elm.n == n1][0]

        self.dofs = list(self.shaft_elem.dof_global_index.values())

    def _initialize_parameters(self, frequency):
        K = self.rotor.K(frequency)

        self.kcoup_auxt = 1 / (
            K[self.dofs[5], self.dofs[5]] + K[self.dofs[11], self.dofs[11]]
        )

        self.kCOUP = (K[self.dofs[0], self.dofs[0]] * K[self.dofs[6], self.dofs[6]]) / (
            K[self.dofs[0], self.dofs[0]] + K[self.dofs[6], self.dofs[6]]
        )

        self.Kcoup_auxI = K[self.dofs[5], self.dofs[5]] / (
            K[self.dofs[5], self.dofs[5]] + K[self.dofs[11], self.dofs[11]]
        )

        self.Kcoup_auxF = K[self.dofs[11], self.dofs[11]] / (
            K[self.dofs[5], self.dofs[5]] + K[self.dofs[11], self.dofs[11]]
        )

        self.fir = -np.pi / 180

    def compute_reaction_force(self, y, ap):
        """Calculate reaction forces of parallel misalignment.

        Parameters
        ----------
        y : np.ndarray
            Displacement response of the element.
        ap : float
            Angular position of the element.

        Returns
        -------
        F : np.ndarray
            Force matrix of the element due to misalignment.
        """

        self.fir = (
            self.Kcoup_auxI * ap
            + self.Kcoup_auxF * ap
            + self.kCOUP
            * self.kcoup_auxt
            * self.eCOUP
            * (
                (y[self.dofs[6]] - y[self.dofs[0]]) * np.sin(self.fir)
                - (y[self.dofs[7]] - y[self.dofs[1]]) * np.cos(self.fir)
            )
        )

        k0 = self.kCOUP
        delta = self.eCOUP
        beta = 0

        k_beta = np.array(
            [
                k0 * self.Kcoup_auxI * delta * np.sin(beta + self.fir),
                -k0 * self.Kcoup_auxI * delta * np.cos(beta + self.fir),
                0,
                0,
                0,
                0,
                k0 * self.Kcoup_auxF * delta * np.sin(beta + self.fir),
                -k0 * self.Kcoup_auxF * delta * np.cos(beta + self.fir),
                0,
                0,
                0,
                0,
            ]
        )

        K_mis = np.zeros((12, 12))
        K_mis[5, :] = k_beta
        K_mis[11, :] = -k_beta

        F_mis = np.array(
            [
                (-k0 * delta * np.cos(beta + self.fir) + k0 * delta),
                -k0 * delta * np.sin(beta + self.fir),
                0,
                0,
                0,
                self.TD - self.TL,
                (k0 * delta * np.cos(beta + self.fir) - k0 * delta),
                k0 * delta * np.sin(beta + self.fir),
                0,
                0,
                0,
                -(self.TD - self.TL),
            ]
        )

        F = K_mis @ y[self.dofs] + F_mis

        return F

    def _get_force_over_time(self, step, disp_resp, ang_pos):
        """Calculate the dynamic force on given time step.

        Paramenters
        -----------
        step : int
            Current time step index.
        disp_resp : np.ndarray
            Displacement response of the system at the current time step.
        ang_pos : float
            Angular position of the shaft at the current time step.

        Returns
        -------
        F : np.ndarray
            Force matrix due to rubbing in the current time step `t[step]`.
        """

        F = np.zeros(self.rotor.ndof)
        F[self.dofs] = self.compute_reaction_force(disp_resp, ang_pos)
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

        F, ang_pos, speed, _ = rotor._unbalance_force_over_time(
            node, unb_magnitude, unb_phase, speed, t
        )

        self._initialize_parameters(np.mean(speed))

        self.forces = np.zeros((rotor.ndof, len(t)))

        force_mis = lambda step, **state: self._get_force_over_time(
            step, state.get("disp_resp"), ang_pos[step]
        )

        results = rotor.run_time_response(
            speed=speed,
            F=F.T,
            t=t,
            method="newmark",
            add_to_RHS=force_mis,
            num_modes=12,
            **kwargs,
        )

        return results


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
