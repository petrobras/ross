import numpy as np
from re import search
from copy import deepcopy as copy

import ross as rs
from ross.gear_element import GearElement
from ross.rotor_assembly import Rotor

__all__ = ["MultiRotor"]


class MultiRotor(Rotor):
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
        gear_mesh_stiffness=None,
        orientation_angle=0.0,
        position="above",
        tag=None,
    ):
        self.rotors = [driving_rotor, driven_rotor]
        self.gear_mesh_stiffness = gear_mesh_stiffness
        self.orientation_angle = float(orientation_angle)

        R1 = copy(driving_rotor)
        R2 = copy(driven_rotor)

        gear_1 = [
            elm
            for elm in R1.disk_elements
            if elm.n == coupled_nodes[0] and type(elm) == GearElement
        ]
        gear_2 = [
            elm
            for elm in R2.disk_elements
            if elm.n == coupled_nodes[1] and type(elm) == GearElement
        ]
        if len(gear_1) == 0 or len(gear_2) == 0:
            raise TypeError("Each rotor needs a GearElement in the coupled nodes!")
        else:
            gear_1 = gear_1[0]
            gear_2 = gear_2[0]

        self.gears = [gear_1, gear_2]

        self.gear_ratio = gear_2.n_teeth/gear_1.n_teeth

        self.K_coupled_mesh_stiffness = None

        # Contact ratio
    
        module_1 = gear_1.pitch_diameter/gear_1.n_teeth
        addendum_1 = 1*module_1
        radii_ad_1 = gear_1.pitch_diameter + addendum_1
        radii_base_1 = gear_1.pitch_diameter*np.cos(gear_1.pressure_angle)

        module_2 = gear_2.pitch_diameter/gear_2.n_teeth
        addendum_2 = 1*module_2
        radii_ad_2 = gear_2.pitch_diameter + addendum_2
        radii_base_2 = gear_2.pitch_diameter*np.cos(gear_2.pressure_angle)

        if round(module_1,1) != round(module_2,1): 
            raise ValueError("The gear module must be the same for both gears in order to mesh properly.")
        
        if gear_1.pressure_angle != gear_2.pressure_angle: 
            raise ValueError("The gear width must be the same for both gears in order to mesh properly.")

        center_distance = (
            gear_1.pitch_diameter + gear_2.pitch_diameter
        )

        contact_length = (
            np.sqrt(
                radii_ad_1 ** 2
                - radii_base_1 ** 2
            )
            + np.sqrt(
                radii_ad_2 ** 2
                - radii_base_2 ** 2
            )
            - center_distance * np.sin(gear_1.pressure_angle)
        )
        base_pitch = np.pi * module_1 * np.cos(gear_1.pressure_angle)
        self.contact_ratio = contact_length / base_pitch

        if gear_1.width != gear_2.width: 
            raise ValueError("The gear width must be the same for both gears in order to mesh properly.")
    
        # If mesh stiffneess is already not defined
        if gear_mesh_stiffness is None:
            c = self.contact_ratio
            w = gear_1.width
            E1 = gear_1.material.E
            E2 = gear_2.material.E
            gear_mesh_stiffness = (c * w * E1 * E2) / (9 * (E1 + E2))

        self.gear_mesh_stiffness = gear_mesh_stiffness

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
        self.dz_pos = float(R1.nodes_pos[idx1] - R2.nodes_pos[idx2])

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

        self.R2_nodes = [int(n + d_node) for n in R2.nodes]

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
            speed = -(1/(self.gear_ratio)) * omega
            rotor = self.rotors[1]

        if isinstance(rotor, MultiRotor):
            return rotor.check_speed(node, speed)

        return speed

    def coupling_matrix(self):
        """Coupling matrix of two coupled gears.

        coupling matrix according to:
        STRINGER, D. B. Geared Rotor Dynamic Methodologies for Advancing Prognostic Modeling
        Capabilities in Rotary-Wing Transmission Systems. Tese (Dissertation) — University of Virginia, Charlottesville, VA, 2008

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

        #Note:  Pressure angle is the normal pressure angle (not transverse)

        #Angles start in degrees

        Pitch_Radius_1 = self.gears[0].pitch_diameter/2 # driving gear
        Pitch_Radius_2 = self.gears[1].pitch_diameter/2 # driven gear
        Pressure_Angle = self.gears[0].pressure_angle
        Helical_Angle = self.gears[0].helix_angle - 180*np.pi/180 # correction done for adjusting with the formulation
        Orientation_Angle = self.orientation_angle
        
        
        #Direction Cosine angles
        cx=np.cos(Pressure_Angle)*np.cos(Helical_Angle)
        cy=np.sin(Pressure_Angle)
        cz=np.sin(Pressure_Angle)*np.sin(Helical_Angle)
    
        cp=np.cos(Orientation_Angle)
        sp=np.sin(Orientation_Angle)

        #Submatrices

        # --- Matriz Kii ---
        Kii = np.zeros((6, 6))

        Kii[0, 0] = (sp * cx + cp * cy)**2
        Kii[1, 0] = (sp * cx + cp * cy) * (sp * cy - cp * cx)  
        Kii[2, 0] = cz * (sp * cx + cp * cy)
        Kii[3, 0] = sp * cz * Pitch_Radius_1 * (sp * cx + cp * cy)
        Kii[4, 0] = -1 * cp * cz * Pitch_Radius_1 * (sp * cx + cp * cy)
        Kii[5, 0] = -1 * cx * Pitch_Radius_1 * (cp**2 + sp**2) * (sp * cx + cp * cy) 

        Kii[0, 1] = Kii[1, 0]
        Kii[1, 1] = (sp * cy - cp * cx)**2
        Kii[2, 1] = cz * (sp * cy - cp * cx)
        Kii[3, 1] = sp * cz * Pitch_Radius_1 * (sp * cy - cp * cx)
        Kii[4, 1] = -1 * cp * cz * Pitch_Radius_1 * (sp * cy - cp * cx)
        Kii[5, 1] = -1 * cx * Pitch_Radius_1 * (cp**2 + sp**2) * (sp * cy - cp * cx)

        Kii[0, 2] = Kii[2, 0]
        Kii[1, 2] = Kii[2, 1]
        Kii[2, 2] = cz**2
        Kii[3, 2] = sp * (cz**2) * Pitch_Radius_1
        Kii[4, 2] = -1 * cp * (cz**2) * Pitch_Radius_1
        Kii[5, 2] = -1 * cx * cz * Pitch_Radius_1 * (cp**2 + sp**2)

        Kii[0, 3] = Kii[3, 0]
        Kii[1, 3] = Kii[3, 1]
        Kii[2, 3] = Kii[3, 2]
        Kii[3, 3] = (sp * cz * Pitch_Radius_1)**2
        Kii[4, 3] = -1 * cp * sp * (cz * Pitch_Radius_1)**2
        Kii[5, 3] = -1 * sp * cx * cz * (Pitch_Radius_1**2) * (cp**2 + sp**2)

        Kii[0, 4] = Kii[4, 0]
        Kii[1, 4] = Kii[4, 1]
        Kii[2, 4] = Kii[4, 2]
        Kii[3, 4] = Kii[4, 3]
        Kii[4, 4] = (cp * cz * Pitch_Radius_1)**2
        Kii[5, 4] = cp * cx * cz * (Pitch_Radius_1**2) * (cp**2 + sp**2)

        Kii[0, 5] = Kii[5, 0]
        Kii[1, 5] = Kii[5, 1]
        Kii[2, 5] = Kii[5, 2]
        Kii[3, 5] = Kii[5, 3]
        Kii[4, 5] = Kii[5, 4]
        Kii[5, 5] = ((cx * Pitch_Radius_1)**2) * (cp**4 + 2 * (cp * sp)**2 + sp**4)

        # --- Matriz Kji ---
        Kji = np.zeros((6, 6))

        Kji[0, 0] = -1 * (sp * cx + cp * cy)**2
        Kji[1, 0] = -1 * (sp * cx + cp * cy) * (sp * cy - cp * cx) 
        Kji[2, 0] = -1 * cz * (sp * cx + cp * cy)
        Kji[3, 0] = -1 * sp * cz * Pitch_Radius_2 * (sp * cx + cp * cy)
        Kji[4, 0] = cp * cz * Pitch_Radius_2 * (sp * cx + cp * cy)
        Kji[5, 0] = cx * Pitch_Radius_2 * (cp**2 + sp**2) * (sp * cx + cp * cy) 

        Kji[0, 1] = Kji[1, 0]
        Kji[1, 1] = -1 * (sp * cy - cp * cx)**2
        Kji[2, 1] = -1 * cz * (sp * cy - cp * cx)
        Kji[3, 1] = -1 * sp * cz * Pitch_Radius_2 * (sp * cy - cp * cx)
        Kji[4, 1] = cp * cz * Pitch_Radius_2 * (sp * cy - cp * cx)
        Kji[5, 1] = cx * Pitch_Radius_2 * (cp**2 + sp**2) * (sp * cy - cp * cx)  

        Kji[0, 2] = Kji[2, 0]
        Kji[1, 2] = Kji[2, 1]
        Kji[2, 2] = -1 * (cz**2)
        Kji[3, 2] = -1 * sp * (cz**2) * Pitch_Radius_2
        Kji[4, 2] = cp * (cz**2) * Pitch_Radius_2
        Kji[5, 2] = cx * cz * Pitch_Radius_2 * (cp**2 + sp**2)

        Kji[0, 3] = -1 * sp * cz * Pitch_Radius_1 * (sp * cx + cp * cy)
        Kji[1, 3] = -1 * sp * cz * Pitch_Radius_1 * (sp * cy - cp * cx)
        Kji[2, 3] = -1 * sp * (cz**2) * Pitch_Radius_1
        Kji[3, 3] = -1 * ((sp * cz)**2) * Pitch_Radius_1 * Pitch_Radius_2
        Kji[4, 3] = cp * sp * (cz**2) * Pitch_Radius_1 * Pitch_Radius_2
        Kji[5, 3] = sp * cx * cz * Pitch_Radius_1 * Pitch_Radius_2 * (cp**2 + sp**2)

        Kji[0, 4] = cp * cz * Pitch_Radius_1 * (sp * cx + cp * cy)
        Kji[1, 4] = cp * cz * Pitch_Radius_1 * (sp * cy - cp * cx)
        Kji[2, 4] = cp * (cz**2) * Pitch_Radius_1
        Kji[3, 4] = cp * sp * (cz**2) * Pitch_Radius_1 * Pitch_Radius_2
        Kji[4, 4] = -1 * ((cp * cz)**2) * Pitch_Radius_1 * Pitch_Radius_2
        Kji[5, 4] = -1 * cp * cx * cz * Pitch_Radius_1 * Pitch_Radius_2 * (cp**2 + sp**2)

        Kji[0, 5] = cx * Pitch_Radius_1 * (cp**2 + sp**2) * (sp * cx + cp * cy)  
        Kji[1, 5] = cx * Pitch_Radius_1 * (cp**2 + sp**2) * (sp * cy - cp * cx)
        Kji[2, 5] = cx * cz * Pitch_Radius_1 * (cp**2 + sp**2)
        Kji[3, 5] = sp * cx * cz * Pitch_Radius_1 * Pitch_Radius_2 * (cp**2 + sp**2)
        Kji[4, 5] = -1 * cp * cx * cz * Pitch_Radius_1 * Pitch_Radius_2 * (cp**2 + sp**2)
        Kji[5, 5] = -1 * (cx**2) * Pitch_Radius_1 * Pitch_Radius_2 * (cp**4 + 2 * (cp * sp)**2 + sp**4) 

        # --- Matriz Kij ---
        Kij = Kji.T # Transposição em NumPy

        # --- Matriz Kjj ---
        Kjj = np.zeros((6, 6))

        Kjj[0, 0] = (sp * cx + cp * cy)**2
        Kjj[1, 0] = (sp * cx + cp * cy) * (sp * cy - cp * cx) 
        Kjj[2, 0] = cz * (sp * cx + cp * cy)
        Kjj[3, 0] = sp * cz * Pitch_Radius_2 * (sp * cx + cp * cy)
        Kjj[4, 0] = -1 * cp * cz * Pitch_Radius_2 * (sp * cx + cp * cy)
        Kjj[5, 0] = -1 * cx * Pitch_Radius_2 * (cp**2 + sp**2) * (sp * cx + cp * cy)

        Kjj[0, 1] = Kjj[1, 0]
        Kjj[1, 1] = (sp * cy - cp * cx)**2
        Kjj[2, 1] = cz * (sp * cy - cp * cx)
        Kjj[3, 1] = sp * cz * Pitch_Radius_2 * (sp * cy - cp * cx)
        Kjj[4, 1] = -1 * cp * cz * Pitch_Radius_2 * (sp * cy - cp * cx)
        Kjj[5, 1] = -1 * cx * Pitch_Radius_2 * (cp**2 + sp**2) * (sp * cy - cp * cx) 

        Kjj[0, 2] = Kjj[2, 0]
        Kjj[1, 2] = Kjj[2, 1]
        Kjj[2, 2] = cz**2
        Kjj[3, 2] = sp * (cz**2) * Pitch_Radius_2
        Kjj[4, 2] = -1 * cp * (cz**2) * Pitch_Radius_2
        Kjj[5, 2] = -1 * cx * cz * Pitch_Radius_2 * (cp**2 + sp**2)

        Kjj[0, 3] = Kjj[3, 0]
        Kjj[1, 3] = Kjj[3, 1]
        Kjj[2, 3] = Kjj[3, 2]
        Kjj[3, 3] = (sp * cz * Pitch_Radius_2)**2
        Kjj[4, 3] = -1 * cp * sp * (cz * Pitch_Radius_2)**2
        Kjj[5, 3] = -1 * sp * cx * cz * (Pitch_Radius_2**2) * (cp**2 + sp**2)

        Kjj[0, 4] = Kjj[4, 0]
        Kjj[1, 4] = Kjj[4, 1]
        Kjj[2, 4] = Kjj[4, 2]
        Kjj[3, 4] = Kjj[4, 3]
        Kjj[4, 4] = (cp * cz * Pitch_Radius_2)**2
        Kjj[5, 4] = cp * cx * cz * (Pitch_Radius_2**2) * (cp**2 + sp**2)

        Kjj[0, 5] = Kjj[5, 0]
        Kjj[1, 5] = Kjj[5, 1]
        Kjj[2, 5] = Kjj[5, 2]
        Kjj[3, 5] = Kjj[5, 3]
        Kjj[4, 5] = Kjj[5, 4]
        Kjj[5, 5] = ((cx * Pitch_Radius_2)**2) * (cp**4 + 2 * (cp * sp)**2 + sp**4)


        # --- Montagem da Matriz de Rigidez Completa ---
        coupling_matrix = np.block([[Kii, Kij],
                                    [Kji, Kjj]])

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
                self.rotors[1].M(frequency * (1/(self.gear_ratio)), synchronous),
            )

    def K(self, frequency):
        """Stiffness matrix for a multi-rotor.

        Parameters
        ----------
        frequency : float, optional
            Excitation frequency.

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
            self.rotors[0].K(frequency),
            self.rotors[1].K(frequency * (1/(self.gear_ratio))),
        )

        dofs_1 = self.gears[0].dof_global_index.values()
        dofs_2 = self.gears[1].dof_global_index.values()
        dofs = [*dofs_1, *dofs_2]

        self.K_coupled_mesh_stiffness = self.coupling_matrix() * self.gear_mesh_stiffness

        K0[np.ix_(dofs, dofs)] += self.K_coupled_mesh_stiffness

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
            self.rotors[0].Ksdt(), -(1/(self.gear_ratio)) * self.rotors[1].Ksdt()
        )

    def C(self, frequency):
        """Damping matrix for a multi-rotor rotor.

        Parameters
        ----------
        frequency : float
            Excitation frequency.

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
            self.rotors[0].C(frequency),
            self.rotors[1].C(frequency * (1/(self.gear_ratio))),
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
            self.rotors[0].G(), -(1/(self.gear_ratio)) * self.rotors[1].G()
        )


def two_shaft_rotor_example():
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

    pressure_angle = rs.Q_(22.5, "deg")
    base_radius = 0.5086
    pitch_diameter = 2 * base_radius / np.cos(pressure_angle)

    N1 = 328  # Number of teeth of gear 1
    m=726.4
    Id=56.95
    Ip=113.9
    width = (4*m)/(material.rho*np.pi*(pitch_diameter**2-d1[-1]*2))
    gear1 = rs.GearElement(
        n=4,
        m=m,
        Id=Id,
        Ip=Ip,
        width=width,
        n_teeth=N1,
        pitch_diameter=pitch_diameter,
        pressure_angle=pressure_angle,
        material=material,
        helix_angle=0,
    )

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

    base_radius = 0.03567
    pitch_diameter = 2 * base_radius / np.cos(pressure_angle)

    N2 = 23  # Number of teeth of gear 2
    m=5
    Id=0.002
    Ip=0.004
    width = (4*m)/(material.rho*np.pi*(pitch_diameter**2-d2[0]*2))
    gear2 = rs.GearElement(
        n=0,
        m=m,
        Id=Id,
        Ip=Ip,
        width=width,
        n_teeth=N2,
        pitch_diameter=pitch_diameter,
        pressure_angle=pressure_angle,
        material=material,
        helix_angle=0,
    )

    turbine = rs.DiskElement(n=2, m=7.45, Id=0.0745, Ip=0.149)

    bearing3 = rs.BearingElement(n=1, kxx=10.1e6, kyy=41.6e6, cxx=3e3)
    bearing4 = rs.BearingElement(n=3, kxx=10.1e6, kyy=41.6e6, cxx=3e3)

    rotor2 = rs.Rotor(
        shaft2,
        [gear2, turbine],
        [bearing3, bearing4],
    )


    return rs.MultiRotor(
        rotor1,
        rotor2,
        coupled_nodes=(4, 0),
        orientation_angle=0.0,
        position="below",
    )
