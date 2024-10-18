import numpy as np
from re import search
from copy import deepcopy as copy
from plotly import graph_objects as go

from ross.disk_element import DiskElement6DoF
from ross.rotor_assembly import Rotor
from ross.units import Q_

__all__ = ["GearElement", "MultiRotor"]


class GearElement(DiskElement6DoF):
    """A gear element.

    This class creates a gear element from input data of inertia and mass.

    Parameters
    ----------
    n: int
        Node in which the gear will be inserted.
    m : float
        Mass of the gear element.
    Id : float
        Diametral moment of inertia.
    Ip : float
        Polar moment of inertia
    pitch_diameter : float
        Pitch diameter of the gear.
    pressure_angle : float, optional
        The pressure angle of the gear in radians. Default is 20° (converted to radians).
    tag : str, optional
        A tag to name the element.
        Default is None.
    scale_factor: float or str, optional
        The scale factor is used to scale the gear drawing.
        For gears it is also possible to provide 'mass' as the scale factor.
        In this case the code will calculate scale factors for each gear based
        on the gear with the higher mass. Notice that in this case you have to
        create all gears with the scale_factor='mass'.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is 'Goldenrod'.

    Examples
    --------
    """

    def __init__(
        self,
        n,
        m,
        Id,
        Ip,
        pitch_diameter,
        pressure_angle=Q_(20, "deg"),
        tag=None,
        scale_factor=1.0,
        color="Goldenrod",
    ):

        self.base_radius = pitch_diameter * np.cos(pressure_angle) / 2
        self.pressure_angle = pressure_angle

        super().__init__(n, m, Id, Ip, tag, scale_factor, color)

    @classmethod
    def from_geometry(
        cls,
        n,
        material,
        width,
        i_d,
        o_d,
        pressure_angle=Q_(20, "deg"),
        tag=None,
        scale_factor=1.0,
        color="Goldenrod",
    ):
        """Create a gear element from geometry properties.

        This class method will create a gear element from geometry data.
        Properties are calculated as per :cite:`friswell2010dynamics`, appendix 1
        for a hollow cylinder:

        Mass:

        :math:`m = \\rho \\pi w (d_o^2 - d_i^2) / 4`

        Polar moment of inertia:

        :math:`I_p = m (d_o^2 + d_i^2) / 8`

        Diametral moment of inertia:

        :math:`I_d = \\frac{1}{2} I_p + \\frac{1}{12} m w^2`

        Where :math:`\\rho` is the material density, :math:`w` is the gear width,
        :math:`d_o` is the outer diameter and :math:`d_i` is the inner diameter.

        Parameters
        ----------
        n : int
            Node in which the gear will be inserted.
        material: ross.Material
            Gear material.
        width : float
            The face width of the gear (considering that the gear body has the same thickness).
        i_d : float
            Inner diameter (the diameter of the shaft on which the gear is mounted).
        o_d : float
            Outer pitch diameter.
        pressure_angle : float, optional
            The pressure angle of the gear in radians. Default is 20° (converted to radians).
        tag : str, optional
            A tag to name the element
            Default is None
        scale_factor: float, optional
            The scale factor is used to scale the disk drawing.
            Default is 1.
        color : str, optional
            A color to be used when the element is represented.
            Default is 'Goldenrod'.

        Attributes
        ----------
        m : float
            Mass of the gear element.
        Id : float
            Diametral moment of inertia.
        Ip : float
            Polar moment of inertia

        Examples
        --------
        """
        m = material.rho * np.pi * width * (o_d**2 - i_d**2) / 4
        Ip = m * (o_d**2 + i_d**2) / 8
        Id = 1 / 2 * Ip + 1 / 12 * m * width**2

        return cls(n, m, Id, Ip, o_d, pressure_angle, tag, scale_factor, color)

    def _patch(self, position, fig):
        """Gear element patch.

        Patch that will be used to draw the gear element using plotly library.

        Parameters
        ----------
        position : float
            Position in which the patch will be drawn.
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        """

        zpos, ypos, yc_pos, scale_factor = position
        scale_factor *= 1.3
        radius = self.base_radius * 1.1 + 0.05

        z_upper = [
            zpos + scale_factor / 25,
            zpos + scale_factor / 25,
            zpos - scale_factor / 25,
            zpos - scale_factor / 25,
        ]
        y_upper = [ypos, ypos + radius, ypos + radius, ypos]

        z_lower = [
            zpos + scale_factor / 25,
            zpos + scale_factor / 25,
            zpos - scale_factor / 25,
            zpos - scale_factor / 25,
        ]
        y_lower = [-ypos, -ypos - radius, -ypos - radius, -ypos]

        z_pos = z_upper
        z_pos.append(None)
        z_pos.extend(z_lower)

        y_pos = y_upper
        y_upper.append(None)
        y_pos.extend(y_lower)

        customdata = [self.n, self.Ip, self.Id, self.m, self.base_radius * 2]
        hovertemplate = (
            f"Gear Node: {customdata[0]}<br>"
            + f"Polar Inertia: {customdata[1]:.3e}<br>"
            + f"Diametral Inertia: {customdata[2]:.3e}<br>"
            + f"Gear Mass: {customdata[3]:.3f}<br>"
            + f"Gear Base Diam.: {customdata[4]:.3f}<br>"
        )

        fig.add_trace(
            go.Scatter(
                x=z_pos,
                y=[y + yc_pos if y is not None else None for y in y_pos],
                customdata=[customdata] * len(z_pos),
                text=hovertemplate,
                mode="lines",
                fill="toself",
                fillcolor=self.color,
                fillpattern=dict(
                    shape="/", fgcolor="rgba(0, 0, 0, 0.2)", bgcolor=self.color
                ),
                opacity=0.8,
                line=dict(width=2.0, color="rgba(0, 0, 0, 0.2)"),
                showlegend=False,
                name=self.tag,
                legendgroup="gears",
                hoveron="points+fills",
                hoverinfo="text",
                hovertemplate=hovertemplate,
                hoverlabel=dict(bgcolor=self.color),
            )
        )

        return fig


class MultiRotor(Rotor):
    """A class representing a multi-rotor system.

    This class creates a system comprising multiple rotors, with the specified driving rotor and driven rotor.
    For systems with more than two rotors, multiple multi-rotors can be nested.

    Parameters
    ----------
    driving_rotor : rs.Rotor
        The driving rotor object.
    driven_rotor : rs.Rotor
        The driven rotor object.
    coupled_nodes : tuple of int
        Tuple specifying the coupled nodes, where the first node corresponds to the driving rotor and
        the second node corresponds to the driven rotor.
    gear_ratio : float
        The gear ratio between the rotors.
    gear_mesh_stiffness : float
        The stiffness of the gear mesh.
    orientation_angle : float, optional
        The angle between the line of gear centers and x axis. Default is 0.0 rad.
    position : {'above', 'below'}, optional
        The relative position of the driven rotor with respect to the driving rotor when plotting
        the multi-rotor. Default is 'above'.
    tag : str, optional
        A tag to identify the multi-rotor. Default is None.

    Returns
    -------
    rotor : rs.Rotor
        The created multi-rotor object.

    Examples
    --------
    """

    def __init__(
        self,
        driving_rotor,
        driven_rotor,
        coupled_nodes,
        gear_ratio,
        gear_mesh_stiffness,
        orientation_angle=0.0,
        position="above",
        tag=None,
    ):

        self.rotors = [driving_rotor, driven_rotor]
        self.gear_ratio = gear_ratio
        self.gear_mesh_stiffness = gear_mesh_stiffness
        self.orientation_angle = orientation_angle

        if driving_rotor.number_dof != 6 or driven_rotor.number_dof != 6:
            raise TypeError("Rotors must be modeled with 6 degrees of freedom!")

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

        Examples
        --------
        """

        global_matrix = np.zeros((self.ndof, self.ndof))

        first_ndof = self.rotors[0].ndof
        global_matrix[:first_ndof, :first_ndof] = driving_matrix
        global_matrix[first_ndof:, first_ndof:] = driven_matrix

        return global_matrix

    def _check_speed(self, node, omega):
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

        Examples
        --------
        """

        speed = omega
        rotor = self.rotors[0]

        if node in self.R2_nodes:
            speed = -omega * self.gear_ratio
            rotor = self.rotors[1]

        if isinstance(rotor, MultiRotor):
            return rotor._check_speed(node, speed)

        return speed

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

        Examples
        --------
        """
        speed = self._check_speed(node, omega)

        return super()._unbalance_force(node, magnitude, phase, speed)

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
        """

        if frequency is None:
            return self._join_matrices(
                self.rotors[0].M(synchronous=synchronous),
                self.rotors[1].M(synchronous=synchronous),
            )
        else:
            return self._join_matrices(
                self.rotors[0].M(frequency, synchronous),
                self.rotors[1].M(frequency * self.gear_ratio, synchronous),
            )

    def K(self, frequency, ignore=[]):
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
        """

        K0 = self._join_matrices(
            self.rotors[0].K(frequency, ignore),
            self.rotors[1].K(frequency * self.gear_ratio, ignore),
        )

        # Coupling
        phi = self.orientation_angle
        k_g = self.gear_mesh_stiffness

        beta = self.gears[0].pressure_angle
        r1 = self.gears[0].base_radius
        r2 = self.gears[1].base_radius

        S = np.sin(beta - phi)
        C = np.cos(beta - phi)

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
        ]) * k_g
        # fmt: on

        dofs_1 = self.gears[0].dof_global_index.values()
        dofs_2 = self.gears[1].dof_global_index.values()
        dofs = [*dofs_1, *dofs_2]

        K0[np.ix_(dofs, dofs)] += coupling_matrix

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
        """

        return self._join_matrices(
            self.rotors[0].Ksdt(), self.rotors[1].Ksdt() * self.gear_ratio
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
        """

        return self._join_matrices(
            self.rotors[0].C(frequency, ignore),
            self.rotors[1].C(frequency * self.gear_ratio, ignore),
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
        """

        return self._join_matrices(
            self.rotors[0].G(), -self.rotors[1].G() * self.gear_ratio
        )
