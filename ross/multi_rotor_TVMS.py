import numpy as np
from re import search
from copy import deepcopy as copy

import ross as rs
from ross.rotor_assembly import Rotor
from ross.gear_mesh_TVMS import GearElementTVMS, Mesh
import time
from ross.units import Q_
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import os

import plotly.io as pio
#pio.renderers.default = "vscode"


__all__ = ["MultiRotorTVMS"]


class MultiRotorTVMS(Rotor):
    """A class representing a multi-rotor system.

    This class creates a system comprising multiple rotors, with the specified
    driving rotor and driven rotor. For systems with more than two rotors,
    multiple multi-rotors can be nested.

    Parameters
    ----------
    driving_rotor : rs.Rotor
        The driving rotor object.
    driven_rotor : rs.Rotor
        The driven rotor object.
    coupled_nodes : tuple of int
        Tuple specifying the coupled nodes, where the first node corresponds to
        the driving rotor and the second node corresponds to the driven rotor.
    gear_ratio : float
        The gear ratio between the rotors.
    gear_mesh_stiffness : float
        The stiffness of the gear mesh.
    orientation_angle : float, pint.Quantity, optional
        The angle between the line of gear centers and x-axis. Default is 0.0 rad.
    position : {'above', 'below'}, optional
        The relative position of the driven rotor with respect to the driving rotor
        when plotting the multi-rotor. Default is 'above'.
    tag : str, optional
        A tag to identify the multi-rotor. Default is None.

    Returns
    -------
    rotor : rs.Rotor
        The created multi-rotor object.

    Examples
    --------
    >>> import ross as rs
    >>> steel = rs.materials.steel
    >>> # Rotor 1:
    >>> L1 = [0.1, 4.24, 1.16, 0.3]
    >>> d1 = [0.3, 0.3, 0.22, 0.22]
    >>> shaft1 = [
    ...     rs.ShaftElement(
    ...         L=L1[i],
    ...         idl=0.0,
    ...         odl=d1[i],
    ...         material=steel,
    ...     )
    ...     for i in range(len(L1))
    ... ]
    >>> generator = rs.DiskElement(n=1, m=525.7, Id=16.1, Ip=32.2)
    >>> disk = rs.DiskElement(n=2, m=116.04, Id=3.115, Ip=6.23)
    >>> gear1 = rs.GearElement(
    ...     n=4, m=726.4, Id=56.95, Ip=113.9,
    ...     pitch_diameter=1.1, pressure_angle=rs.Q_(22.5, 'deg'),
    ... )
    >>> bearing1 = rs.BearingElement(n=0, kxx=183.9e6, kyy=200.4e6, cxx=3e3)
    >>> bearing2 = rs.BearingElement(n=3, kxx=183.9e6, kyy=200.4e6, cxx=3e3)
    >>> rotor1 = rs.Rotor(shaft1, [generator, disk, gear1], [bearing1, bearing2],)

    >>> # Rotor 2:
    >>> L2 = [0.3, 5, 0.1]
    >>> d2 = [0.15, 0.15, 0.15]
    >>> shaft2 = [
    ...     rs.ShaftElement(
    ...         L=L2[i],
    ...         idl=0.0,
    ...         odl=d2[i],
    ...         material=steel,
    ...     )
    ...     for i in range(len(L2))
    ... ]
    >>> gear2 = rs.GearElement(
    ...     n=0, m=5, Id=0.002, Ip=0.004,
    ...     pitch_diameter=0.077, pressure_angle=rs.Q_(22.5, 'deg'),
    ... )
    >>> turbine = rs.DiskElement(n=2, m=7.45, Id=0.0745, Ip=0.149)
    >>> bearing3 = rs.BearingElement(n=1, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    >>> bearing4 = rs.BearingElement(n=3, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    >>> rotor2 = rs.Rotor(shaft2, [gear2, turbine], [bearing3, bearing4],)

    >>> # Multi rotor:
    >>> multi_rotor = rs.MultiRotor(
    ...     rotor1,
    ...     rotor2,
    ...     coupled_nodes=(4, 0),
    ...     gear_ratio=328 / 23,
    ...     gear_mesh_stiffness=1e8,
    ...     orientation_angle=0.0,
    ...     position="below"
    ... )
    >>> modal = multi_rotor.run_modal(speed=0)
    >>> modal.wd[0] # doctest: +ELLIPSIS
    74.160244...
    """

    def __init__(
        self,
        driving_rotor,
        driven_rotor,
        coupled_nodes,
        orientation_angle=0.0,
        only_max_stiffness = False,
        interpolation = False,
        user_defined_stiffness = None | float,
        position="above",
        tag=None,
    ):
        self.rotors = [driving_rotor, driven_rotor]

        self.orientation_angle = float(orientation_angle)

        R1 = copy(driving_rotor)
        R2 = copy(driven_rotor)

        gear_1 = [
            elm
            for elm in R1.disk_elements
            if elm.n == coupled_nodes[0] and type(elm) == GearElementTVMS
        ]
        gear_2 = [
            elm
            for elm in R2.disk_elements
            if elm.n == coupled_nodes[1] and type(elm) == GearElementTVMS
        ]
        if len(gear_1) == 0 or len(gear_2) == 0:
            raise TypeError("Each rotor needs a GearElement in the coupled nodes!")
        else:
            gear_1 = gear_1[0]
            gear_2 = gear_2[0]

        self.gears = [gear_1, gear_2]

        self.gear_mesh = Mesh(*self.gears, interpolation=interpolation, only_max_stiffness=only_max_stiffness, user_defined_stiffness=user_defined_stiffness)
        self.gear_ratio = self.gear_mesh.eta

        gear1_plot = next(
            (
                elm
                for elm in R1.plot_rotor().data
                if elm["legendgroup"] == "gears"
                and int(search(r"Gear Node: (\d+)", elm.text).group(1)) == gear_1.n
            ),
            None,
        )

        gear2_plot = next(
            (
                elm
                for elm in R2.plot_rotor().data
                if elm["legendgroup"] == "gears"
                and int(search(r"Gear Node: (\d+)", elm.text).group(1)) == gear_2.n
            ),
            None,
        )

        if position == "above":
            ymax = max(y for y in gear1_plot["y"] if y is not None)
            ymin = min(y for y in gear2_plot["y"] if y is not None)
            self.dy_pos = +abs(ymax - ymin)
        else:
            ymax = max(y for y in gear2_plot["y"] if y is not None)
            ymin = min(y for y in gear1_plot["y"] if y is not None)
            self.dy_pos = -abs(ymax - ymin)

        idx1 = R1.nodes.index(gear_1.n)
        idx2 = R2.nodes.index(gear_2.n)
        self.dz_pos = R1.nodes_pos[idx1] - R2.nodes_pos[idx2]

        R1_max_node = max([*R1.nodes, *R1.link_nodes])
        R2_min_node = min([*R2.nodes, *R2.link_nodes])
        d_node = 0
        if R1_max_node >= R2_min_node:
            d_node = R1_max_node + 1
            for elm in R2.elements:
                elm.n += d_node
                try:
                    elm.n_link += d_node
                except:
                    pass

        self.R2_nodes = [n + d_node for n in R2.nodes]

        shaft_elements = [*R1.shaft_elements, *R2.shaft_elements]
        disk_elements = [*R1.disk_elements, *R2.disk_elements]
        bearing_elements = [*R1.bearing_elements, *R2.bearing_elements]
        point_mass_elements = [*R1.point_mass_elements, *R2.point_mass_elements]

        super().__init__(
            shaft_elements,
            disk_elements,
            bearing_elements,
            point_mass_elements,
            tag=tag,
        )

    def _fix_nodes_pos(self, index, node, nodes_pos_l):
        if node < self.R2_nodes[0]:
            nodes_pos_l[index] = self.rotors[0].nodes_pos[
                self.rotors[0].nodes.index(node)
            ]
        elif node == self.R2_nodes[0]:
            nodes_pos_l[index] = self.rotors[1].nodes_pos[0] + self.dz_pos

    def _fix_nodes(self):
        self.nodes = [*self.rotors[0].nodes, *self.R2_nodes]

        R2_nodes_pos = [pos + self.dz_pos for pos in self.rotors[1].nodes_pos]
        self.nodes_pos = [*self.rotors[0].nodes_pos, *R2_nodes_pos]

        R2_center_line = [pos + self.dy_pos for pos in self.rotors[1].center_line_pos]
        self.center_line_pos = [*self.rotors[0].center_line_pos, *R2_center_line]

    def _join_matrices(self, driving_matrix, driven_matrix):
        """Join matrices from the driving rotor and driven rotor to form the matrix of
        the coupled system.

        Parameters
        ----------
        driving_matrix : np.ndarray
            The matrix from the driving rotor.
        driven_matrix : np.ndarray
            The matrix from the driven rotor.

        Returns
        -------
        global_matrix : np.ndarray
            The combined matrix of the coupled system.
        """

        global_matrix = np.zeros((self.ndof, self.ndof))

        first_ndof = self.rotors[0].ndof
        global_matrix[:first_ndof, :first_ndof] = driving_matrix
        global_matrix[first_ndof:, first_ndof:] = driven_matrix

        return global_matrix

    def _unbalance_force(self, node, magnitude, phase, omega):
        """Calculate unbalance forces.

        This is an auxiliary function the calculate unbalance forces. It takes the
        force magnitude and phase and generate an array with complex values of forces
        on each degree degree of freedom of the given node.

        Parameters
        ----------
        node : int
            Node where the unbalance is applied.
        magnitude : float
            Unbalance magnitude (kg.m)
        phase : float
            Unbalance phase (rad)
        omega : list, float
            Array with the desired range of frequencies

        Returns
        -------
        F0 : list
            Unbalance force in each degree of freedom for each value in omega
        """
        speed = self.check_speed(node, omega)

        return super()._unbalance_force(node, magnitude, phase, speed)

    def check_speed(self, node, omega):
        """Adjusts the speed for the specified node based on the rotor configuration.

        This method checks if the given node belongs to the driven rotor.
        If so, the rotation speed is multiplied by the gear ratio.

        Parameters
        ----------
        node : int
            The node index where the speed check is being applied.
        omega : float or np.ndarray
            The original rotation speed of the driving rotor in rad/s.

        Returns
        -------
        speed : float or np.ndarray
            The adjusted rotation speed for the specified node.
        """

        speed = omega
        rotor = self.rotors[0]

        if node in self.R2_nodes:
            speed = -1/self.gear_ratio * omega
            rotor = self.rotors[1]

        if isinstance(rotor, MultiRotorTVMS):
            return rotor.check_speed(node, speed)

        return speed

    def coupling_matrix(self):
        """Coupling matrix of two coupled gears.

        Returns
        -------
        coupling_matrix : np.ndarray
            Dimensionless coupling matrix of two coupled gears

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.coupling_matrix()[:4, :4]
        array([[0.14644661, 0.35355339, 0.        , 0.        ],
               [0.35355339, 0.85355339, 0.        , 0.        ],
               [0.        , 0.        , 0.        , 0.        ],
               [0.        , 0.        , 0.        , 0.        ]])
        """
        r1 = self.gears[0].geometry_dict['r_b']

        r2 = self.gears[1].geometry_dict['r_b']

        S = np.sin(self.gears[0].pressure_angle - self.orientation_angle)
        C = np.cos(self.gears[0].pressure_angle - self.orientation_angle)

        # fmt: off
        coupling_matrix = np.array([
            [   S**2,  S * C, 0, 0, 0,  r1 * S,   -S**2,  -S * C, 0, 0, 0,  r2 * S],
            [  S * C,   C**2, 0, 0, 0,  r1 * C,  -S * C,   -C**2, 0, 0, 0,  r2 * C],
            [      0,      0, 0, 0, 0,       0,       0,       0, 0, 0, 0,       0],
            [      0,      0, 0, 0, 0,       0,       0,       0, 0, 0, 0,       0],
            [      0,      0, 0, 0, 0,       0,       0,       0, 0, 0, 0,       0],
            [ r1 * S, r1 * C, 0, 0, 0,   r1**2, -r1 * S, -r1 * C, 0, 0, 0, r1 * r2],
            [  -S**2, -S * C, 0, 0, 0, -r1 * S,    S**2,   S * C, 0, 0, 0, -r2 * S],
            [ -S * C,  -C**2, 0, 0, 0, -r1 * C,   S * C,    C**2, 0, 0, 0, -r2 * C],
            [      0,      0, 0, 0, 0,       0,       0,       0, 0, 0, 0,       0],
            [      0,      0, 0, 0, 0,       0,       0,       0, 0, 0, 0,       0],
            [      0,      0, 0, 0, 0,       0,       0,       0, 0, 0, 0,       0],
            [ r2 * S, r2 * C, 0, 0, 0, r1 * r2, -r2 * S, -r2 * C, 0, 0, 0,   r2**2],
        ])
        # fmt: on

        return coupling_matrix

    def M(self, frequency=None, synchronous=False):
        """Mass matrix for a multi-rotor.

        Parameters
        ----------
        synchronous : bool, optional
            If True a synchronous analysis is carried out.
            Default is False.

        Returns
        -------
        M0 : np.ndarray
            Mass matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.M(0)[:4, :4]
        array([[18.55298224,  0.        ,  0.        ,  0.        ],
               [ 0.        , 18.55298224,  0.        , -0.16179571],
               [ 0.        ,  0.        , 18.37831702,  0.        ],
               [ 0.        , -0.16179571,  0.        ,  0.10074262]])
        """

        if frequency is None:
            return self._join_matrices(
                self.rotors[0].M(synchronous=synchronous),
                self.rotors[1].M(synchronous=synchronous),
            )
        else:
            return self._join_matrices(
                self.rotors[0].M(frequency, synchronous),
                self.rotors[1].M(-frequency / self.gear_ratio, synchronous),
            )

    def K(self, frequency, ignore=[], **kwargs):
        """Stiffness matrix for a multi-rotor.

        Parameters
        ----------
        frequency : float, optional
            Excitation frequency.
        ignore : list, optional
            List of elements to leave out of the matrix.

        Returns
        -------
        K0 : np.ndarray
            Stiffness matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.K(0)[:4, :4] / 1e10
        array([[ 4.7609372 ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  4.7625872 ,  0.        , -0.23712736],
               [ 0.        ,  0.        , 14.63196778,  0.        ],
               [ 0.        , -0.23712736,  0.        ,  0.09416119]])
        """

        K0 = self._join_matrices(
            self.rotors[0].K(frequency, ignore),
            self.rotors[1].K(-frequency / self.gear_ratio, ignore),
        )

        dofs_1 = self.gears[0].dof_global_index.values()
        dofs_2 = self.gears[1].dof_global_index.values()
        dofs = [*dofs_1, *dofs_2]

        # If TVMS or Maximum Stiffness, it's time dependent and therefore must have time as parameter.
        if hasattr(self.gear_mesh, '_user_defined_stiffness'):
            k_eq = self.gear_mesh._user_defined_stiffness

        # If it is a user defined stiffness, it's not time dependent.
        else:
            t = kwargs['t']
            k_eq = self.gear_mesh.mesh(frequency, t)[0]

        K0[np.ix_(dofs, dofs)] += self.coupling_matrix() * k_eq

        return K0

    def Ksdt(self):
        """Dynamic stiffness matrix for a multi-rotor.

        Stiffness matrix associated with the transient motion of the
        shaft and disks. For time-dependent analyses, this matrix needs to be
        multiplied by the angular acceleration. Therefore, the stiffness matrix
        of the driven rotor is scaled by the gear ratio before being combined
        with the driving rotor matrix.

        Returns
        -------
        Ksdt0 : np.ndarray
            Dynamic stiffness matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.Ksdt()[:6, :4]
        array([[  0.        , -74.43218395,   0.        ,   0.6202682 ],
               [  0.        ,   0.        ,   0.        ,   0.        ],
               [  0.        ,   0.        ,   0.        ,   0.        ],
               [  0.        ,   0.        ,   0.        ,   0.        ],
               [  0.        ,  -0.6202682 ,   0.        ,   0.08270243],
               [  0.        ,   0.        ,   0.        ,   0.        ]])
        """

        return self._join_matrices(
            self.rotors[0].Ksdt(), -1/self.gear_ratio * self.rotors[1].Ksdt()
        )

    def C(self, frequency, ignore=[]):
        """Damping matrix for a multi-rotor rotor.

        Parameters
        ----------
        frequency : float
            Excitation frequency.
        ignore : list, optional
            List of elements to leave out of the matrix.

        Returns
        -------
        C0 : np.ndarray
            Damping matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.C(0)[:4, :4] / 1e3
        array([[3., 0., 0., 0.],
               [0., 3., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        """

        return self._join_matrices(
            self.rotors[0].C(frequency, ignore),
            self.rotors[1].C(-frequency / self.gear_ratio, ignore),
        )

    def G(self):
        """Gyroscopic matrix for a multi-rotor.

        For time-dependent analyses, this matrix needs to be multiplied by the
        rotor speed. Therefore, the gyroscopic matrix of the driven rotor is
        scaled by the gear ratio before being combined with the driving rotor matrix.

        Returns
        -------
        G0 : np.ndarray
            Gyroscopic matrix for the multi-rotor.

        Examples
        --------
        >>> multi_rotor = two_shaft_rotor_example()
        >>> multi_rotor.G()[:4, :4]
        array([[ 0.        ,  0.17162125,  0.        ,  0.1403395 ],
               [-0.17162125,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [-0.1403395 ,  0.        ,  0.        ,  0.        ]])
        """

        return self._join_matrices(
            self.rotors[0].G(), -1/self.gear_ratio * self.rotors[1].G()
        )


def two_shaft_rotor_example(run_type: str):
    """Create a multi-rotor as example.

    This function returns an instance of two-shaft rotor system from Rao et al.
    This typical example is a turbo-alternator rotor system, which consists of
    a generator rotor, a turbine rotor and a spur gear pair connecting two rotors.
    Each rotor is supported by a pair of bearing two shaft elements, one disk and
    two simple bearings.

    The purpose of this is to make available a simple model so that doctest can
    be written using this.

    Returns
    -------
    An instance of a rotor object.

    References
    ----------
    Rao, J. S., Shiau, T. N., chang, J. R. (1998). Theoretical analysis of lateral
    response due to torsional excitation of geared rotors. Mechanism and Machine Theory,
    33 (6), 761-783. doi: 10.1016/S0094-114X(97)00056-6

    Examples
    --------
    >>> multi_rotor = two_shaft_rotor_example()
    >>> modal = multi_rotor.run_modal(speed=0)
    >>> np.round(modal.wd[:4])
    array([ 74.,  77., 112., 113.])
    """
    # A spur geared two-shaft rotor system.
    material = rs.Material(name="mat_steel", rho=7800, E=207e9, G_s=79.5e9)

    # Rotor 1
    L1 = [0.1, 4.24, 1.16, 0.3]
    d1 = [0.3, 0.3, 0.22, 0.22]
    shaft1 = [
        rs.ShaftElement(
            L=L1[i],
            idl=0.0,
            odl=d1[i],
            material=material,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(len(L1))
    ]

    generator = rs.DiskElement(
        n=1,
        m=525.7,
        Id=16.1,
        Ip=32.2,
    )
    disk = rs.DiskElement(
        n=2,
        m=116.04,
        Id=3.115,
        Ip=6.23,
    )
    
    gear1 = GearElementTVMS(n=4, m=5, module=Q_(2,'mm'), width=Q_(2, 'cm'), n_tooth=40, hub_bore_radius=Q_(4, 'cm'))

    bearing1 = rs.BearingElement(n=0, kxx=183.9e6, kyy=200.4e6, cxx=3e3)
    bearing2 = rs.BearingElement(n=3, kxx=183.9e6, kyy=200.4e6, cxx=3e3)

    rotor1 = rs.Rotor(
        shaft1,
        [generator, disk, gear1],
        [bearing1, bearing2],
    )

    # Rotor 2
    L2 = [0.3, 5, 0.1]
    d2 = [0.15, 0.15, 0.15]
    shaft2 = [
        rs.ShaftElement(
            L=L2[i],
            idl=0.0,
            odl=d2[i],
            material=material,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for i in range(len(L2))
    ]
    
    gear2 = GearElementTVMS(n=0, m=Q_(6, 'kg'), module=Q_(2, 'mm'), width=Q_(2,'cm'), n_tooth=75, hub_bore_radius=Q_(7.5,'cm'))

    turbine = rs.DiskElement(n=2, m=7.45, Id=0.0745, Ip=0.149)

    bearing3 = rs.BearingElement(n=1, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    bearing4 = rs.BearingElement(n=3, kxx=10.1e6, kyy=41.6e6, cxx=3e3)

    rotor2 = rs.Rotor(
        shaft2,
        [gear2, turbine],
        [bearing3, bearing4],
    )

    if run_type == 'interpolation':
        return MultiRotorTVMS(
            rotor1,
            rotor2,
            coupled_nodes=(4, 0),
            orientation_angle=0.0,
            position="below",
            interpolation=True,
            only_max_stiffness=False,
            user_defined_stiffness=None
        )

    if run_type == 'max_stiffness':
        return MultiRotorTVMS(
            rotor1,
            rotor2,
            coupled_nodes=(4, 0),
            orientation_angle=0.0,
            position="below",
            interpolation=False,
            only_max_stiffness=True,
            user_defined_stiffness=None
        )

    if run_type == 'TVMS':
        return MultiRotorTVMS(
            rotor1,
            rotor2,
            coupled_nodes=(4, 0),
            orientation_angle=0.0,
            position="below",
            interpolation=False,
            only_max_stiffness=False,
            user_defined_stiffness=None
        )

    if run_type == 'user_defined':
        return MultiRotorTVMS(
            rotor1,
            rotor2,
            coupled_nodes=(4, 0),
            orientation_angle=0.0,
            position="below",
            interpolation=False,
            only_max_stiffness=False,
            user_defined_stiffness=4e8
        )


def main_example(id, t=10, speed=50, run_type = 'interpolation', dt = 1e-5, unb_mag = [35e-4, 40e-4]) -> None:

    process_pid = os.getpid()
    print(f"[PID {process_pid}] Iniciando Simulação: {id} (Speed: {speed} Hz, run_type: {run_type}, dt: {dt}, t_final: {t}), unb: {unb_mag}")
    
    run_type = run_type
    rotor = two_shaft_rotor_example(run_type=run_type)

    nodes = [2, 7]
    unb_mag = unb_mag
    unb_phase = [0, 0]

    dt = dt
    t = np.arange(0, t, dt)
    speed1 = speed*2*np.pi  # Generator rotor speed

    num_dof = rotor.number_dof

    F = np.zeros((len(t), rotor.ndof))

    for i, node in enumerate(nodes):
        speed = rotor.check_speed(node, speed1)
        phi = speed * t + unb_phase[i]

        dofx = num_dof * node + 0
        dofy = num_dof * node + 1
        F[:, dofx] += unb_mag[i] * (speed**2) * np.cos(phi)
        F[:, dofy] += unb_mag[i] * (speed**2) * np.sin(phi)

    start_time=time.time()
    tr = rotor.run_time_response(speed1, F, t, method='newmark', progress_interval=0.01)


    end_time = time.time()
    print(f'Time to run:{end_time - start_time}')

    node = 3
    dof_node_x = node * rotor.number_dof + 5
    dof_node_y = dof_node_x + 1
    
    x = tr.yout[:,dof_node_x]
    y = tr.yout[:,dof_node_y]
    
    arr = np.column_stack([t,x, y])
    arr = pd.DataFrame(arr, columns=['time','x', 'y'])
    arr.to_csv(f'~/Desktop/IC/gear_simulation/w_{speed1/np.pi/2:.2f}hz_dt_{dt:.2e}s_t_{np.max(t):.2f}s_40_75_{run_type}_unb0_{unb_mag[0]}_unb1_{unb_mag[1]}.csv')

if __name__ == "__main__":

    lista_argumentos = [
        ('A1', 4, 60, 'interpolation', 1e-5, [50e-4, 70e-4]),
        ('A2', 4, 60, 'max_stiffness', 1e-5, [50e-4, 70e-4]),
        ('B1', 4, 70, 'interpolation', 1e-5, [80e-4, 100e-4]),
        ('B1', 4, 70, 'max_stiffness', 1e-5, [80e-4, 100e-4]),      
        ('C1', 4, 20, 'interpolation', 1e-5, [100e-4, 100e-4]),
        ('C2', 4, 20, 'max_stiffness', 1e-5, [100e-4, 100e-4]),
    ]

    overall_start_time = time.time()

    # O 'with' garante que o pool de processos seja fechado corretamente ao final.
    with multiprocessing.Pool(processes=6) as pool:
        # 'starmap' é usado porque cada conjunto de argumentos em 'lista_argumentos_simulacoes'
        # é uma tupla que precisa ser desempacotada ("starred") para a função 'run_single_simulation'.
        pool.starmap(main_example, lista_argumentos)
    
    overall_end_time = time.time()
    total_parallel_time = overall_end_time - overall_start_time

    print("\n--- Todas as Simulações Concluídas ---")
    print(f"Tempo total de execução paralela: {total_parallel_time:.2f} segundos.")
    print("Resumo dos resultados:")
