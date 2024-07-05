from copy import deepcopy as copy
from plotly import graph_objects as go

from ross.disk_element import DiskElement6DoF
from ross.rotor_assembly import Rotor
from ross.units import Q_

__all__ = ["GearElement", "MultiRotor"]


class GearElement(DiskElement6DoF):

    def __init__(
        self, n, m, Id, Ip, radius, tag=None, scale_factor=1.0, color="Goldenrod"
    ):

        self.radius = radius  # Base circle radius

        super().__init__(n, m, Id, Ip, tag, scale_factor, color)

    def _patch(self, position, fig):

        zpos, ypos, yc_pos, scale_factor = position
        scale_factor *= 1.3
        radius = self.radius * 1.1 + 0.05

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

        customdata = [self.n, self.Ip, self.Id, self.m, self.radius]
        hovertemplate = (
            f"Gear Node: {customdata[0]}<br>"
            + f"Polar Inertia: {customdata[1]:.3e}<br>"
            + f"Diametral Inertia: {customdata[2]:.3e}<br>"
            + f"Gear Mass: {customdata[3]:.3f}<br>"
            + f"Gear Radius: {customdata[4]:.3f}<br>"
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

    def __init__(
        self,
        rotor_1,
        rotor_2,
        coupled_nodes,
        gear_ratio,
        gear_mesh_stiffness,
        pressure_angle=Q_(25, "deg"),
    ):

        self.rotors = [rotor_1, rotor_2]
        self.gear_ratio = gear_ratio
        self.gear_mesh_stiffness = gear_mesh_stiffness
        self.pressure_angle = pressure_angle  # 20° -- 25°

        if rotor_1.number_dof != 6 or rotor_2.number_dof != 6:
            raise TypeError("Rotors must be modeled with 6 degrees of freedom!")

        R1 = copy(rotor_1)
        R2 = copy(rotor_2)

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

        idx1 = R1.nodes.index(gear_1.n)
        idx2 = R2.nodes.index(gear_2.n)
        self.dz_pos = R1.nodes_pos[idx1] - R1.nodes_pos[idx2]

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
            shaft_elements, disk_elements, bearing_elements, point_mass_elements
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
