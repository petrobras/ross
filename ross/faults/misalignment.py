"""Misalignment module.

This module defines misalignments of various types on the shaft coupling. There are
a number of options, for the formulation of 6 DoFs (degrees of freedom).
"""

from abc import ABC

import numpy as np

import ross as rs
from ross.units import Q_, check_units


__all__ = ["MisalignmentFlex", "MisalignmentRigid"]


class MisalignmentFlex(ABC):
    """Model misalignment on a given flexible coupling element of a rotor system.

    Calculates the dynamic reaction force of hexangular flexible coupling
    induced by rotor misalignment of some kind based on :cite:`xia2019study`.

    Parameters
    ----------
    rotor : ross.Rotor
        Rotor object.
    n_mis : float
        Number of shaft element where the misalignment is ocurring.
    delta_x : float
        Parallel misalignment offset between driving rotor and driven rotor along X direction.
    delta_y : float
        Parallel misalignment offset between driving rotor and driven rotor along Y direction.
    radial_stiffness : float
        Radial stiffness of flexible coupling.
    bending_stifness : float
        Bending stiffness of flexible coupling. Provide if mis_type is "angular" or "combined".
    mis_angle : float
        Angular misalignment angle.
    mis_type: string
        Name of the chosen misalignment type.
        The avaible types are: "parallel", "angular" and "combined". Default is "parallel".
    input_torque : float
        Driving torque. Default is 0.
    load_torque : float
        Driven torque. Default is 0.

    Returns
    -------
        A MisalignmentFlex object.

    Attributes
    ----------
    shaft_elem : ross.ShaftElement
        A 6 degrees of freedom shaft element object where misalignment is ocurring.
    forces : np.ndarray
        Force matrix due to misalignment. Each row corresponds to a dof and each column to a time.

    References
    ----------
    .. bibliography::
    :filter: docname in docnames

    Examples
    --------
    >>> rotor = rs.rotor_example_with_damping()
    >>> fault = MisalignmentFlex(rotor, n_mis=0, delta_x=2e-4, delta_y=2e-4,
    ... radial_stiffness=40e3, bending_stiffness=38e3, mis_angle=5 * np.pi / 180,
    ... mis_type="combined", input_torque=0, load_torque=0)
    >>> fault.shaft_elem
    ShaftElement(L=0.025, idl=0.0, idr=0.0, odl=0.019,  odr=0.019, material='Steel', n=0)
    """

    @check_units
    def __init__(
        self,
        rotor,
        n_mis,
        delta_x,
        delta_y,
        radial_stiffness,
        bending_stiffness,
        mis_angle,
        mis_type="parallel",
        input_torque=0,
        load_torque=0,
    ):
        self.rotor = rotor

        self.input_torque = input_torque
        self.load_torque = load_torque

        self.delta_x = delta_x
        self.delta_y = delta_y

        self.radial_stiffness = radial_stiffness
        self.bending_stiffness = bending_stiffness

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
        self.shaft_elem = [elm for elm in rotor.shaft_elements if elm.n == n_mis][0]

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

        aux1 = self.radius**2 + self.delta_x**2 + self.delta_y**2
        aux2 = 2 * self.radius * np.sqrt(self.delta_x**2 + self.delta_y**2)

        phi = np.arctan(self.delta_x / self.delta_y) + ang_pos

        Fp_ = lambda f, a: np.sqrt(aux1 + aux2 * f(phi + a)) - self.radius
        Fpx_ = lambda f, a: Fp_(f, a) * np.sin(ang_pos + 4 * a)
        Fpy_ = lambda f, a: Fp_(f, a) * np.cos(ang_pos + 4 * a)

        Fpx = (
            Fpx_(np.sin, 0)
            + Fpx_(np.cos, np.pi / 6)
            - Fpx_(lambda x: -np.sin(x), np.pi / 3)
        ) * self.radial_stiffness

        Fpy = (
            Fpy_(np.sin, 0)
            + Fpy_(np.cos, np.pi / 6)
            - Fpy_(lambda x: -np.sin(x), np.pi / 3)
        ) * self.radial_stiffness

        F = np.zeros((self.rotor.ndof, len(ang_pos)))

        F[self.dofs[0]] = Fpx
        F[self.dofs[6]] = -Fpx

        F[self.dofs[1]] = Fpy
        F[self.dofs[7]] = -Fpy

        F[self.dofs[5]] = self.input_torque
        F[self.dofs[11]] = self.load_torque

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

        aux = (
            self.bending_stiffness
            * self.radius
            * np.sqrt(2 - 2 * np.cos(self.mis_angle))
            * np.sin(self.mis_angle)
        )

        Fa_ = lambda a: np.abs(aux * np.sin(ang_pos + 4 * a))
        Fax_ = lambda a: Fa_(a) * np.cos(ang_pos + np.pi + 4 * a)
        Fay_ = lambda a: Fa_(a) * np.sin(ang_pos + np.pi + 4 * a)

        Fax = Fax_(0) + Fax_(np.pi / 6) + Fax_(np.pi / 3)
        Fay = Fay_(0) + Fay_(np.pi / 6) + Fay_(np.pi / 3)

        F = np.zeros((self.rotor.ndof, len(ang_pos)))

        F[self.dofs[0]] = Fax
        F[self.dofs[6]] = -Fax

        F[self.dofs[1]] = Fay
        F[self.dofs[7]] = -Fay

        F[self.dofs[5]] = self.input_torque
        F[self.dofs[11]] = self.load_torque

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


class MisalignmentRigid(ABC):
    """Model misalignment on a given rigid coupling element of a rotor system.

    Calculates the dynamic reaction force of hexangular rigid coupling
    induced by rotor parallel misalignment based on :cite:`hussain2002dynamic`.

    Parameters
    ----------
    n_mis : float
        Number of shaft element where the misalignment is ocurring.
    delta : float
        Parallel misalignment offset between driving rotor and driven rotor.
    input_torque : float
        Driving torque. Default is 0.
    load_torque : float
        Driven torque. Default is 0.

    Returns
    -------
    A MisalignmentRigid object.

    Attributes
    ----------
    shaft_elem : ross.ShaftElement
        A 6 degrees of freedom shaft element object where misalignment is ocurring.
    kl1 : float
        Stiffness of the x-direction degree of freedom at the left node of the shaft element.
    kl2 : float
        Stiffness of the x-direction degree of freedom at the right node of the shaft element.
    kt1 : float
        Stiffness of the torsional degree of freedom at the left node of the shaft element.
    kt2 : float
        Stiffness of the torsional degree of freedom at the right node of the shaft element.
    phi : float
        Coupling angular position.
    forces : np.ndarray
        Force matrix due to misalignment. Each row corresponds to a dof and each column to a time.

    References
    ----------
    .. bibliography::
        :filter: docname in docnames

    Examples
    --------
    >>> rotor = rs.rotor_example_with_damping()
    >>> fault = MisalignmentRigid(rotor, n_mis=0, delta=2e-4,
    ... input_torque=0, load_torque=0)
    >>> fault.shaft_elem
    ShaftElement(L=0.025, idl=0.0, idr=0.0, odl=0.019,  odr=0.019, material='Steel', n=0)
    """

    @check_units
    def __init__(
        self,
        rotor,
        n_mis,
        delta,
        input_torque=0,
        load_torque=0,
    ):
        self.rotor = rotor

        self.delta = delta

        self.input_torque = input_torque
        self.load_torque = load_torque

        # Shaft element with misalignment
        self.shaft_elem = [elm for elm in rotor.shaft_elements if elm.n == n_mis][0]

        self.dofs = list(self.shaft_elem.dof_global_index.values())

        self._initialize_params()

    def _initialize_params(self, speed=0):
        K = self.rotor.K(speed)

        self.kl1 = K[self.dofs[0], self.dofs[0]]
        self.kl2 = K[self.dofs[6], self.dofs[6]]

        self.kt1 = K[self.dofs[5], self.dofs[5]]
        self.kt2 = K[self.dofs[11], self.dofs[11]]

        self.phi = -np.pi / 180

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

        kte = 1 / (self.kt1 + self.kt2)
        kt1 = self.kt1 * kte
        kt2 = self.kt2 * kte
        kle = self.kl1 * self.kl2 / (self.kl1 + self.kl2)

        x1 = y[self.dofs[0]]
        x2 = y[self.dofs[6]]

        y1 = y[self.dofs[1]]
        y2 = y[self.dofs[7]]

        # fmt: off
        self.phi = kt1 * ap + kt2 * ap + (
            kle * kte * self.delta * (
                (x2 - x1) * np.sin(self.phi) - (y2 - y1) * np.cos(self.phi)
            )
        )
        # fmt: on

        beta = 0
        sin = np.sin(beta + self.phi)
        cos = np.cos(beta + self.phi)

        k_beta = np.array(
            [
                kle * self.delta * (kt1 * sin),
                -kle * self.delta * (kt1 * cos),
                0,
                0,
                0,
                0,
                kle * self.delta * (kt2 * sin),
                -kle * self.delta * (kt2 * cos),
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
                -kle * self.delta * (cos - 1),
                -kle * self.delta * sin,
                0,
                0,
                0,
                self.input_torque - self.load_torque,
                kle * self.delta * (cos - 1),
                kle * self.delta * sin,
                0,
                0,
                0,
                -(self.input_torque - self.load_torque),
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
            Force matrix due to misalignment in the current time step `t[step]`.
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

        self.forces = np.zeros((rotor.ndof, len(t)))

        force_mis = lambda step, **state: self._get_force_over_time(
            step, state.get("disp_resp"), ang_pos[step]
        )

        self._initialize_params(np.mean(speed))

        results = rotor.run_time_response(
            speed=speed,
            F=F.T,
            t=t,
            method="newmark",
            add_to_RHS=force_mis,
            **kwargs,
        )

        return results


def misalignment_flex_example(mis_type="parallel"):
    """Create an example of a flexible combined misalignment fault.

    This function returns time response results of a flexible misalignment
    fault. The purpose is to make available a simple model so that a
    doctest can be written using it.

    mis_type: string
        Name of the chosen misalignment type.
        The avaible types are: "parallel", "angular" and "combined".
        Default is "parallel".

    Returns
    -------
    results : ross.TimeResponseResults
        Results for a shaft with misalignment.

    Examples
    --------
    >>> from ross.faults.misalignment import misalignment_flex_example
    >>> from ross.probe import Probe
    >>> results = misalignment_flex_example("combined")
    Running direct method
    >>> probe1 = Probe(14, 0)
    >>> probe2 = Probe(22, 0)
    >>> fig = results.plot_1d([probe1, probe2])
    """

    rotor = rs.rotor_example_with_damping()

    n1 = rotor.disk_elements[0].n
    n2 = rotor.disk_elements[1].n

    misalignment = rotor.run_misalignment(
        coupling="flex",
        n_mis=0,
        radial_stiffness=40e3,
        bending_stiffness=38e3,
        delta_x=2e-4,
        delta_y=2e-4,
        mis_angle=5 * np.pi / 180,
        mis_type=mis_type,
        input_torque=0,
        load_torque=0,
        node=[n1, n2],
        unbalance_magnitude=[5e-4, 0],
        unbalance_phase=[-np.pi / 2, 0],
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
    )

    return misalignment


def misalignment_rigid_example():
    """Create an example of a rigid misalignment fault.

    This function returns time response results of a rigid misalignment
    fault. The purpose is to make available a simple model so that a
    doctest can be written using it.

    Returns
    -------
    misalignment : ross.MisalignmentRigid Object
        An instance of a rigid misalignment model object.

    Examples
    --------
    >>> from ross.faults.misalignment import misalignment_rigid_example
    >>> from ross.probe import Probe
    >>> results = misalignment_rigid_example()
    Running direct method
    >>> probe1 = Probe(14, 0)
    >>> probe2 = Probe(22, 0)
    >>> fig = results.plot_1d([probe1, probe2])
    """

    rotor = rs.rotor_example_with_damping()

    n1 = rotor.disk_elements[0].n
    n2 = rotor.disk_elements[1].n

    misalignment = rotor.run_misalignment(
        coupling="rigid",
        n_mis=0,
        delta=2e-4,
        input_torque=0,
        load_torque=0,
        node=[n1, n2],
        unbalance_magnitude=[5e-4, 0],
        unbalance_phase=[-np.pi / 2, 0],
        speed=Q_(1200, "RPM"),
        t=np.arange(0, 0.5, 0.0001),
    )

    return misalignment
