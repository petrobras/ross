from copy import deepcopy as copy

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

        shaft_elements = [*R1.shaft_elements, *R2.shaft_elements]
        disk_elements = [*R1.disk_elements, *R2.disk_elements]
        bearing_elements = [*R1.bearing_elements, *R2.bearing_elements]
        point_mass_elements = [*R1.point_mass_elements, *R2.point_mass_elements]

        super().__init__(
            shaft_elements, disk_elements, bearing_elements, point_mass_elements
        )
