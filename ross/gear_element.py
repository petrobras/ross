"""Gear Element module.

This module defines the GearElement classes which can be used to represent
gears or gearboxes used to couple different shafts in the MultiRotor class.
"""

import numpy as np
from plotly import graph_objects as go
from ross.units import Q_

from ross.disk_element import DiskElement6DoF


__all__ = ["GearElement"]


class GearElement(DiskElement6DoF):
    """A gear element.

    This class creates a gear element from input data of inertia and mass.

    Parameters
    ----------
    n: int
        Node in which the gear will be inserted.
    m : float, pint.Quantity
        Mass of the gear element (kg).
    Id : float, pint.Quantity
        Diametral moment of inertia.
    Ip : float, pint.Quantity
        Polar moment of inertia.
    m  : float
        Gear module (mm).
    n_tooth: int
        Tooth number of the gear. 
    pressure_angle : float, pint.Quantity, optional
        The pressure angle of the gear (rad).
        Default is 20 deg (converted to rad).
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
    >>> gear = GearElement(
    ...        n=0, m=4.67, Id=0.015, Ip=0.030,
    ...        pitch_diameter=0.187,
    ...        pressure_angle=Q_(22.5, "deg")
    ... )
    >>> gear.pressure_angle # doctest: +ELLIPSIS
    0.392699...
    """

    def __init__(
        self,
        n,
        m,
        Id,
        Ip,
        module,
        n_tooth,
        pressure_angle=20 * np.pi / 180,
        addendum_coeff=1,
        clearance_coeff=0.25,
        tag=None,
        scale_factor=1.0,
        color="Goldenrod",
    ):
        if pressure_angle is None:
            pressure_angle = Q_(20, "deg")

        self.module = module
        self.n_tooth = n_tooth
        self.pitch_diameter = self.module * self.n_tooth
        self.pressure_angle = float(pressure_angle)
        self.base_radius = float(self.pitch_diameter) * np.cos(self.pressure_angle) / 2

        super().__init__(n, m, Id, Ip, tag, scale_factor, color)
        
    @classmethod
    def from_geometry(
        cls,
        n,
        material,
        width,
        i_d,
        o_d,
        pressure_angle=None,
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
        width : float, pint.Quantity
            The face width of the gear considering that the gear body has the
            same thickness (m).
        i_d : float, pint.Quantity
            Inner diameter (the diameter of the shaft on which the gear is mounted).
        o_d : float, pint.Quantity
            Outer pitch diameter (m).
        pressure_angle : float, pint.Quantity, optional
            The pressure angle of the gear (rad).
            Default is 20 deg (converted to rad).
        tag : str, optional
            A tag to name the element
            Default is None
        scale_factor: float, optional
            The scale factor is used to scale the gear drawing.
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
        >>> from ross.materials import steel
        >>> gear = GearElement.from_geometry(0, steel, 0.07, 0.05, 0.28)
        >>> gear.base_radius # doctest: +ELLIPSIS
        0.131556...
        >>>
        """
        m = material.rho * np.pi * width * (o_d**2 - i_d**2) / 4
        Ip = m * (o_d**2 + i_d**2) / 8
        Id = 1 / 2 * Ip + 1 / 12 * m * width**2

        if pressure_angle is None:
            pressure_angle = Q_(20, "deg")

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

class GearGeometry:

    def __init__(self, gear: GearElement):
        self.gear: GearElement = gear
        self.geometry: dict[str, float] = self._initialize_geometry()
    
    def _initialize_geometry(self) -> dict[str, float]:
        angles: dict[str, float] = self._notable_angles()
        radii: dict[str, float] = self._notable_radii()
        geo_const: dict[str, float] = self._geometric_constants()
        tau_const: dict[str, float] = self._tau_constants()

        geometry: dict[str, float] = angles | radii | geo_const | tau_const

        return geometry

    
    def _notable_angles(self) -> dict[str, float]:
        pass

    def _notable_radii(self) -> dict[str, float]:
        gear = self.gear

        r_b: float = 1 / 2 * gear.module * gear.n_tooth * np.cos(gear.pressure_angle) # radius of base circle [m] MAOK
        r_p: float = r_b / np.cos(gear.pressure_angle)                # radius of pitch circle [m] MAOK
        r_a: float = r_p + self.ha_ * self.m                 # radius of addendum circle [m] 
        r_c: float = np.sqrt( np.square(self.r_b * np.tan(self.alpha)  - self.ha_ * self.m /  np.sin(self.alpha) ) + np.square(self.r_b) ) # radii of the involute starting point [m] MAOK
        r_f: float = 1 / 2 * self.m * self.N - (self.ha_ + self.c_) * self.m   # radius of root circle [m] MAOK
        r_rho: float = self.c_ * self.m / (1 - np.sin(self.alpha) )            # radius of cutter tip round corner [m] MAOK
        r_rho_: float = self.r_rho / self.m

        dict_place = {
            'r_b': r_b, 
            'r_p': r_p,
            'r_a': r_a,
            'r_c': r_c,
            'r_f': r_f,
            'r_rho': r_rho,
            'r_rho_': r_rho_
        }

        return dict_place
    def _geometric_constants(self) -> dict[str, float]:
        pass

    def _tau_constants(self) -> dict[str, float]:
        pass

    @staticmethod
    def _involute(angle: float) -> float:
        """Involute function

        Calculates the involute function for a given angle.

        The involute function is used to describe the contact region of the gear profile.

        :math:`inv(angle) = tan(angle) - angle`
        
        This functions computes the value of the involute for an input angle in radians.

        Parameters
        ----------
        angle : float
            Input angle in radians.

        Returns 
        ---------
        float
            The inv(angle)
        
        Notes
        --------
        - Ensure that the angle is in radians before passing it to this function.

        Examples
        --------
        >>> Gear.involute(20 / 180 * np.pi)
        0.014904383867336446

        >>> Gear.involute(15 / 180 * np.pi)
        0.006149804631973288
        """

        return np.tan(angle) - angle
    
    def _to_tau(self, alpha_i: float) -> float:
        pass
    



