import numpy as np
import ross as rs

from copy import copy

from ross.rotor_assembly import Rotor
from ross.results import ForcedResponseResults
from ross.units import check_units
from ross.utils import make_speed_array

__all__ = ["CoAxialRotor", "coaxrotor_example"]


class CoAxialRotor(Rotor):
    """A rotor object.

    This class will create a system of co-axial rotors with the shaft,
    disk, bearing and seal elements provided.

    Parameters
    ----------
    shafts : list of lists
        Each list of shaft elements builds a different shaft. The number of
        lists sets the number of shafts.
    disk_elements : list
        List with the disk elements
    bearing_elements : list
        List with the bearing elements
    point_mass_elements: list
        List with the point mass elements
    speed_ratio: float, optional
        Speed ratio of the outer rotor with respect to the inner rotor
    modal_damping_ratio: list, optional
        List of modal damping ratio(s) for the first modes
    default_damping_ratio: float, optional
        Default modal damping ratio for the remaining modes.
        Default is zero.
    alpha : float, optional
        Mass proportional damping factor.
        Default is zero.
    beta : float, optional
        Stiffness proportional damping factor.
        Default is zero.
    tag : str
        A tag for the rotor

    Returns
    -------
    A rotor object.

    Attributes
    ----------
    nodes : list
        List of the model's nodes.
    nodes_pos : list
        List with nodal spatial location.
    CG : float
        Center of gravity

    Examples
    --------
    >>> import ross as rs
    >>> steel = rs.materials.steel
    >>> i_d = 0
    >>> o_d = 0.05
    >>> n = 10
    >>> L = [0.25 for _ in range(n)]
    >>> axial_shaft = [rs.ShaftElement(l, i_d, o_d, material=steel) for l in L]
    >>> i_d = 0.15
    >>> o_d = 0.20
    >>> n = 6
    >>> L = [0.25 for _ in range(n)]
    >>> coaxial_shaft = [rs.ShaftElement(l, i_d, o_d, material=steel) for l in L]
    >>> shaft = [axial_shaft, coaxial_shaft]
    >>> disk0 = rs.DiskElement.from_geometry(n=1,
    ...                                     material=steel,
    ...                                     width=0.07,
    ...                                     i_d=0.05,
    ...                                     o_d=0.28)
    >>> disk1 = rs.DiskElement.from_geometry(n=9,
    ...                                     material=steel,
    ...                                     width=0.07,
    ...                                     i_d=0.05,
    ...                                     o_d=0.28)
    >>> disk2 = rs.DiskElement.from_geometry(n=13,
    ...                                      material=steel,
    ...                                      width=0.07,
    ...                                      i_d=0.20,
    ...                                      o_d=0.48)
    >>> disk3 = rs.DiskElement.from_geometry(n=15,
    ...                                      material=steel,
    ...                                      width=0.07,
    ...                                      i_d=0.20,
    ...                                      o_d=0.48)
    >>> disks = [disk0, disk1, disk2, disk3]
    >>> stfx = 1e6
    >>> stfy = 0.8e6
    >>> bearing0 = rs.BearingElement(0, kxx=stfx, kyy=stfy, cxx=0)
    >>> bearing1 = rs.BearingElement(10, kxx=stfx, kyy=stfy, cxx=0)
    >>> bearing2 = rs.BearingElement(11, kxx=stfx, kyy=stfy, cxx=0)
    >>> bearing3 = rs.BearingElement(8, n_link=17, kxx=stfx, kyy=stfy, cxx=0)
    >>> bearings = [bearing0, bearing1, bearing2, bearing3]
    >>> rotor = rs.CoAxialRotor(shaft, disks, bearings)
    """

    def __init__(
        self,
        shafts,
        disk_elements=None,
        bearing_elements=None,
        point_mass_elements=None,
        speed_ratio=1,
        min_w=None,
        max_w=None,
        rated_w=None,
        modal_damping_ratio=None,
        default_damping_ratio=0.0,
        alpha=0.0,
        beta=0.0,
        tag=None,
    ):
        self.speed_ratio = speed_ratio

        # copy shaft elements to avoid altering attributes for elements
        # that might be used in different rotors, e.g. altering shaft_element.n
        shafts = [[copy(sh) for sh in shaft] for shaft in shafts]

        # number each shaft right after the previous one
        aux_n = 0
        for shaft in shafts:
            for i, sh in enumerate(shaft):
                if sh.n is None:
                    sh.n = i + aux_n
            aux_n = shaft[-1].n_r + 1

        self.shafts_nodes = [
            sorted({n for sh in shaft for n in (sh.n, sh.n_r)}) for shaft in shafts
        ]

        super().__init__(
            shafts,
            disk_elements=disk_elements,
            bearing_elements=bearing_elements,
            point_mass_elements=point_mass_elements,
            min_w=min_w,
            max_w=max_w,
            rated_w=rated_w,
            modal_damping_ratio=modal_damping_ratio,
            default_damping_ratio=default_damping_ratio,
            alpha=alpha,
            beta=beta,
            tag=tag,
        )

        self.outer_dofs = self._get_outer_global_dofs(self.shaft_elements)

        # Fill the shaft_number column of the rotor dataframes
        shaft_numbers = {el.tag: self._shaft_number(el.n) for el in self.elements}

        for df in (
            self.df,
            self.df_shaft,
            self.df_disks,
            self.df_bearings,
            self.df_seals,
            self.df_point_mass,
        ):
            if len(df):
                df["shaft_number"] = df["tag"].map(shaft_numbers).astype(float)

        # Draw bearings between shafts up to the outer shaft inner surface
        for brg in self.bearing_elements:
            if brg.n_link in self.nodes:
                outer_node = brg.n if brg.n in self.shafts_nodes[1] else brg.n_link
                sh_at_node = self.df_shaft[
                    (self.df_shaft.n_l == outer_node)
                    | (self.df_shaft.n_r == outer_node)
                ]
                self.df.loc[self.df.tag == brg.tag, "y_pos_sup"] = (
                    sh_at_node.i_d.min() / 2
                )

    def _shaft_number(self, node):
        """Return the index of the shaft the node belongs to.

        Nodes outside the shafts (e.g. bearing housings) belong to the shaft
        of the bearing they are linked to.

        Parameters
        ----------
        node : int
            Node number.

        Returns
        -------
        shaft_number : int
            Index of the shaft in ``shafts``.
        """
        if node in self.link_nodes:
            node = self._find_linked_bearing_node(node)

        return next(
            j for j, shaft_nodes in enumerate(self.shafts_nodes) if node in shaft_nodes
        )

    def _fix_nodes_pos(self, index, node, nodes_pos_l):
        """Adjust node positions of the outer rotor"""
        if node == self.shafts_nodes[1][0]:
            for n_outer in self.shafts_nodes[1]:
                n_inner = self._find_linked_bearing_node(n_outer)

                if n_inner is not None:
                    i = next(
                        i for i, sh in enumerate(self.shaft_elements) if sh.n == n_inner
                    )
                    L_outer_shaft = sum(
                        sh.L for sh in self._get_outer_elements(self.shaft_elements)
                    )

                    nodes_pos_l[index] = nodes_pos_l[i]
                    if node != n_outer:
                        nodes_pos_l[index] -= L_outer_shaft

                    return

    def _set_nodes(self, df_shaft):
        """Set nodes and nodes_pos lists"""
        nodes_pos = {}

        for n, pos in zip(df_shaft.n_l, df_shaft.nodes_pos_l, strict=True):
            nodes_pos[int(n)] = max(nodes_pos.get(int(n), pos), pos)

        for n, pos in zip(df_shaft.n_r, df_shaft.nodes_pos_r, strict=True):
            nodes_pos.setdefault(int(n), pos)

        self.nodes = [n for sh_n in self.shafts_nodes for n in sh_n]
        self.nodes_pos = [nodes_pos[n] for n in self.nodes]
        self.center_line_pos = [0] * len(self.nodes)

    def _get_outer_elements(self, elements=None):
        elements = elements or self.elements

        return [el for el in elements if el.n in self.shafts_nodes[1]]

    def _get_outer_global_dofs(self, elements=None):
        if elements is None:
            return self.outer_dofs
        else:
            return sorted(
                {
                    dof
                    for el in self._get_outer_elements(elements)
                    for dof in el.dof_global_index.values()
                }
            )

    def G(self):
        G0 = self.G0.copy()
        dofs = self.outer_dofs

        G0[np.ix_(dofs, dofs)] *= self.speed_ratio

        return G0

    def _node_speed_ratio(self, node):
        """Return the speed ratio of the shaft the node belongs to.

        Parameters
        ----------
        node : int
            Node index.

        Returns
        -------
        ratio : float
            ``speed_ratio`` for a node on the outer shaft, 1 otherwise.
        """
        return self.speed_ratio if node in self.shafts_nodes[1] else 1

    def _unbalance_force(self, node, magnitude, phase, omega):
        """Calculate unbalance forces at the node's own excitation frequency.

        The force rotates with the shaft the node belongs to, so its
        frequency is ``abs(ratio) * omega``. For a counter-rotating shaft the
        force whirls backward, which flips the phase and the sign of the y
        component.

        Parameters
        ----------
        node : int
            Node where the unbalance is applied.
        magnitude : float
            Unbalance magnitude (kg.m).
        phase : float
            Unbalance phase (rad).
        omega : list, float
            Inner shaft speeds (rad/s).

        Returns
        -------
        F0 : np.ndarray
            Unbalance force in each degree of freedom for each value in omega.
        """
        ratio = self._node_speed_ratio(node)
        frequency = abs(ratio) * np.asarray(omega)

        if ratio >= 0:
            return super()._unbalance_force(node, magnitude, phase, frequency)
        else:
            F0 = super()._unbalance_force(node, magnitude, -phase, frequency)
            F0[node * self.number_dof + 1] *= -1

            return F0

    @check_units
    def run_unbalance_response(
        self,
        node,
        unbalance_magnitude,
        unbalance_phase,
        speed_range=None,
        modes=None,
    ):
        """Unbalanced response for a coaxial rotor.

        Each unbalance excites at the speed of its own shaft, so an unbalance
        on the outer shaft excites at ``abs(speed_ratio)`` times the inner
        shaft speed. The response is evaluated at that frequency, with the
        rotor at the inner shaft speed.

        Parameters
        ----------
        node : list, int
            Node where the unbalance is applied.
        unbalance_magnitude : list, float, pint.Quantity
            Unbalance magnitude (kg.m).
        unbalance_phase : list, float, pint.Quantity
            Unbalance phase (rad).
        speed_range : list, pint.Quantity
            Inner shaft speeds (rad/s).
            Default is 0 to 1.5 x highest damped natural frequency.
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).

        Returns
        -------
        results : ross.ForcedResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.ForcedResponseResults`

        Raises
        ------
        ValueError
            If there are unbalances on both shafts and ``abs(speed_ratio)``
            is not 1. The response then has two harmonics, which a single
            result cannot hold, so each shaft must be run separately.

        Examples
        --------
        >>> rotor = coaxrotor_example()
        >>> speed = np.linspace(0, 150, 31)
        >>> response = rotor.run_unbalance_response(node=[3, 13],
        ...                                         unbalance_magnitude=[1e-4, 1e-4],
        ...                                         unbalance_phase=[0, 0],
        ...                                         speed_range=speed)
        """
        if speed_range is None:
            modal = self.run_modal(0)
            speed_range = np.linspace(0, max(modal.evalues.imag) * 1.5, 1000)

        speed_range = np.asarray(speed_range)

        node = np.atleast_1d(node)
        unbalance_magnitude = np.atleast_1d(unbalance_magnitude)
        unbalance_phase = np.atleast_1d(unbalance_phase)

        frequency_ratios = {abs(self._node_speed_ratio(n)) for n in node}
        if len(frequency_ratios) > 1:
            raise ValueError(
                "Unbalances on both shafts excite at different frequencies "
                f"(abs(speed_ratio) = {abs(self.speed_ratio)}). Run the unbalance "
                "response for each shaft separately."
            )

        frequency_range = frequency_ratios.pop() * speed_range

        self._check_coefficient_axes(speed=speed_range, frequency=frequency_range)

        force = np.zeros((self.ndof, len(speed_range)), dtype=complex)
        for n, m, p in zip(node, unbalance_magnitude, unbalance_phase, strict=True):
            force += self._unbalance_force(n, m, p, speed_range)

        forced_resp = np.zeros((self.ndof, len(speed_range)), dtype=complex)
        for i, (speed, frequency) in enumerate(
            zip(speed_range, frequency_range, strict=True)
        ):
            H = self.transfer_matrix(speed=speed, frequency=frequency, modes=modes)
            forced_resp[:, i] = H @ force[:, i]

        return ForcedResponseResults(
            rotor=self,
            forced_resp=forced_resp,
            velc_resp=1j * frequency_range * forced_resp,
            accl_resp=-(frequency_range**2) * forced_resp,
            speed_range=speed_range,
            unbalance=np.vstack((node, unbalance_magnitude, unbalance_phase)),
        )

    def unbalance_force_over_time(
        self, node, magnitude, phase, omega, t, return_all=False
    ):
        """Calculate unbalance forces for each time step.

        This auxiliary function calculates the unbalanced forces by taking
        into account the magnitude and phase of the force. It generates an
        array of force values at each degree of freedom for the specified
        nodes at each time step, while also considering a range of
        frequencies.

        Parameters
        ----------
        node : list, int
            Nodes where the unbalance is applied.
        magnitude : list, float
            Unbalance magnitude (kg.m) for each node.
        phase : list, float
            Unbalance phase (rad) for each node.
        omega : float, np.darray
            Constant velocity or desired range of velocities (rad/s).
        t : np.darray
            Time array (s).
        return_all : bool, optional
            If True, returns F0, theta, omega, and alpha.
            If False, returns only F0.
            Default is False.

        Returns
        -------
        F0 : np.ndarray
            Unbalance force at each degree of freedom for each time step.
        theta : np.ndarray
            Angular positions for each time step.
        omega : np.ndarray
            Angular velocities for each time step.
        alpha : np.ndarray
            Angular accelerations for each time step.
        """

        omega, theta, alpha = make_speed_array(omega, t)

        F0 = np.zeros((self.ndof, len(t)))

        for i, n in enumerate(node):
            phi = phase[i] + theta * self._node_speed_ratio(n)
            w = omega * self._node_speed_ratio(n)
            a = alpha * self._node_speed_ratio(n)

            Fx = magnitude[i] * ((w**2) * np.cos(phi) + a * np.sin(phi))
            Fy = magnitude[i] * ((w**2) * np.sin(phi) - a * np.cos(phi))

            F0[n * self.number_dof + 0, :] += Fx
            F0[n * self.number_dof + 1, :] += Fy

        if return_all:
            return F0, theta, omega, alpha
        else:
            return F0


def coaxrotor_example():
    """Create a coaxial rotor as example.

    This function returns an instance of a coaxial rotor with
    2 shafts, 4 disk and 4 bearings.

    Returns
    -------
    An instance of a rotor object.

    Examples
    --------
    >>> rotor = coaxrotor_example()
    >>> modal = rotor.run_modal(speed=0)
    >>> np.round(modal.wd[:4])
    array([39., 39., 99., 99.])
    """
    steel = rs.materials.steel

    i_d = 0
    o_d = 0.05
    n = 10
    L = [0.25 for _ in range(n)]

    axial_shaft = [rs.ShaftElement(l, i_d, o_d, material=steel) for l in L]

    i_d = 0.25
    o_d = 0.30
    n = 6
    L = [0.25 for _ in range(n)]

    coaxial_shaft = [rs.ShaftElement(l, i_d, o_d, material=steel) for l in L]

    disk0 = rs.DiskElement.from_geometry(
        n=1, material=steel, width=0.07, i_d=0.05, o_d=0.28, scale_factor=0.8
    )
    disk1 = rs.DiskElement.from_geometry(
        n=9, material=steel, width=0.07, i_d=0.05, o_d=0.28, scale_factor=0.8
    )
    disk2 = rs.DiskElement.from_geometry(
        n=13, material=steel, width=0.07, i_d=0.20, o_d=0.48, scale_factor=0.8
    )
    disk3 = rs.DiskElement.from_geometry(
        n=15, material=steel, width=0.07, i_d=0.20, o_d=0.48, scale_factor=0.8
    )

    shaft = [axial_shaft, coaxial_shaft]
    disks = [disk0, disk1, disk2, disk3]

    stfx = 1e6
    stfy = 1e6
    bearing0 = rs.BearingElement(
        0, n_link=18, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4
    )
    bearing1 = rs.BearingElement(
        10, n_link=19, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4
    )
    bearing2 = rs.BearingElement(11, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4)
    bearing3 = rs.BearingElement(
        8, n_link=17, kxx=stfx, kyy=stfy, cxx=0, scale_factor=0.4
    )

    base0 = rs.BearingElement(18, kxx=1e8, kyy=1e8, cxx=0, scale_factor=0.4)
    base1 = rs.BearingElement(19, kxx=1e8, kyy=1e8, cxx=0, scale_factor=0.4)

    pointmass0 = rs.PointMass(n=18, m=20)
    pointmass1 = rs.PointMass(n=19, m=20)

    bearings = [bearing0, bearing1, bearing2, bearing3, base0, base1]
    pointmasses = [pointmass0, pointmass1]

    return CoAxialRotor(shaft, disks, bearings, pointmasses)
