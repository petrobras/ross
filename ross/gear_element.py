"""Gear Element module.

This module defines the GearElement classes which can be used to represent
gears or gearboxes used to couple different shafts in the MultiRotor class.
"""

import numpy as np
from plotly import graph_objects as go
from ross.units import Q_

from ross.disk_element import DiskElement


__all__ = ["GearElement"]


class GearElement(DiskElement):
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
        Gear module (m).
    n_tooth: int
        Tooth number of the gear. 
    pressure_angle : float, pint.Quantity, optional
        The pressure angle of the gear (rad).
        Default is 20 deg (converted to rad).
    width: float
        The width of the spur gear (m)
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
        width,
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
        self.pressure_angle = float(pressure_angle)
        self.width = width
        self.ha_ = addendum_coeff
        self.c_ = clearance_coeff
        self.pitch_diameter = module * n_tooth

        super().__init__(n, m, Id, Ip, tag, scale_factor, color)

        self.geometry: GearGeometry = GearGeometry(self)
        
        
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
        self._initialize_geometry()
    
    def _initialize_geometry(self) -> None:
        """
        Initialize the geometry dictionary for gear tooth stiffness analysis.

        This method populates ``self.geometryDict`` with various geometric parameters 
        computed by helper functions:

        - Radii constants (from :meth:`_notable_radii`)
        - Angular constants (from :meth:`_notable_angles`)
        - Geometric constants (from :meth:`_geometric_constants`)
        - Tau constants (from :meth:`_tau_constants`)

        Parameters:
        ----
            None

        Returns:
        -----
            None
        """
        # Initialize the geometryDict
        self.geometryDict: dict[str, float] = {}

        # Add radii constants
        radii:      dict[str, float] = self._notable_radii()
        self.geometryDict |= radii 

        # Add angular constants of the involute profile
        angles:     dict[str, float] = self._notable_angles()
        self.geometryDict |= angles

        # Add geometric constants of the tooth profile
        geo_const:  dict[str, float] = self._geometric_constants()
        self.geometryDict |= geo_const

        # Add geometric constants of tau values of the tooth profile
        tau_const:  dict[str, float] = self._tau_constants()
        self.geometryDict |= tau_const

    def _notable_angles(self) -> dict[str, float]:
        """Computes and returns key angles from the gear tooth's geometric profile. 
        
        These angles are typically used in integration methods and geometric analyses of the gear.
        
        Parameters
        -----
        None

        Returns
        --- 
        dict[str, float]: Dictionary containing angles for geometry evaluations.

            `alpha_a` (float): Pressure angle at the addendum circle [rad].

            `alpha_c` (float): Pressure angle at the involute "C" point [rad].

            `theta_f` (float): Angle between the tooth center line and the junction with the root circle [rad].

            `theta_b` (float): Reference angle combining the half-tooth pitch and the base involute angle [rad].

        """
        gear: GearElement           = self.gear
        geometry: dict[str, float]  = self.geometryDict

        # pressure angle when the contact point is on the addendum circle [rad] MAOK
        alpha_a: float = np.arccos(geometry['r_b'] / geometry['r_a']) 

        # pressure angle when the contact point is on the C point [rad] MAOK
        alpha_c: float = np.arccos(geometry['r_b'] / geometry['r_c'])       
        
        # The angle between the tooth center-line and de junction with the root circle [radians]
        theta_f: float = (                                                          
            1 / gear.n_tooth * ( 
                np.pi / 2 
                + 2 * np.tan(gear.pressure_angle) * (gear.ha_ - geometry['r_rho_'])
                + 2 * geometry['r_rho_'] / np.cos(gear.pressure_angle)
            ) 
        )

        theta_b: float = np.pi / (2*gear.n_tooth) + GearGeometry._involute(gear.pressure_angle)

        # Placeholder dictionary
        dict_place: dict[str, float] = {
            'alpha_a': alpha_a,
            'alpha_c': alpha_c,
            'theta_f': theta_f,
            'theta_b': theta_b
        }

        return dict_place
    
    def _notable_radii(self) -> dict[str, float]:
        """Compute notable gear tooth radii used in integration and geometric analysis.

        Calculates key radii from the gear tooth geometric profile required for
        subsequent integration methods and geometric calculations. All radii are
        returned in meters (m).

        Parameters:
        -------
        None

        Returns:
        ------
        dict_place: dict[str, float]

            - `r_b` (float): 
                Base circle radius [m]  

            - `r_p` (float): 
                Pitch circle radius [m]  

            - `r_a` (float): 
                Addendum circle radius [m]  

            - `r_c` (float): 
                Involute starting point radius [m]  

            - `r_f` (float): 
                Root circle radius [m]  

            - `r_rho` (float): 
                Cutter tip round corner radius [m]  

            - `r_rho_` (float): 
                Normalized cutter tip radius [-]  

        Note:
        ------
            All calculations assume standard gear geometry parameters stored in the
            class instance's `gear` and `geometryDict` attributes.
        """

        # Initializing as different variables for simplicity
        gear = self.gear

        # radius of base circle [m] MAOK
        r_b: float = 1 / 2 * gear.module * gear.n_tooth * np.cos(gear.pressure_angle)

        # radius of pitch circle [m] MAOK
        r_p: float = r_b / np.cos(gear.pressure_angle)

        # radius of addendum circle [m]                                  
        r_a: float = r_p + gear.ha_ * gear.module                                        
        
        # radii of the involute starting point [m] MAOK
        r_c: float = (                                                                 
            np.sqrt( 
                np.square(r_b * np.tan(gear.pressure_angle) 
                - gear.ha_ * gear.module /  np.sin(gear.pressure_angle) ) 
                + np.square(r_b) 
            ) 
        )

        # radius of root circle [m] MAOK
        r_f:    float = 1 / 2 * gear.module * gear.n_tooth - (gear.ha_ + gear.c_) * gear.module

        # radius of cutter tip round corner [m] MAOK
        r_rho:  float = gear.c_ * gear.module / (1 - np.sin(gear.pressure_angle) )

        # normalized by the module radius of cutter tip round corner [m]                  
        r_rho_: float = r_rho / gear.module                                                         

        # placeholder dictionary
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
        """Calculate geometric constants for gear tooth root analysis.

        Computes fundamental geometric parameters required for stress analysis
        and tooth root dimension verification. These constants combine multiple
        geometric properties of the gear system.

        Parameters:
        ---
            None

        Returns:
        ----
            dict[str, float]: Dictionary containing geometric constants.

            - 'a1': (float) 
                Radial clearance constant [m].  

            - 'b1': (float) 
                Base tangent length [m].  

        Note:
        ---
            - Requires precomputed geometryDict values from _notable_radii().
            - All values calculated in meters (m).
            - MAOK refers to mechanical engineering verification markers.
        """
    
        gear = self.gear
        geometry = self.geometryDict

        a1: float = (gear.ha_ + gear.c_) * gear.module - geometry['r_rho']  # MAOK
        b1: float = (                                                       # MAOK
            np.pi * gear.module / 4 + gear.ha_ * gear.module * np.tan(gear.pressure_angle) 
            + geometry['r_rho'] * np.cos(gear.pressure_angle) 
        )

        dict_place: dict[str, float] = {
            'a1': a1,
            'b1': b1
        }

        return dict_place

    def _tau_constants(self) -> dict[str, float]:
        geometry = self.geometryDict

        tau_c: float = self._to_tau(geometry['alpha_c'])
        tau_a: float = self._to_tau(geometry['alpha_a'])  

        dict_place: dict[str, float] = {
            'tau_c': tau_c,
            'tau_a': tau_a
        }      

        return dict_place
    
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
        >>> GearGeometry._involute(20 / 180 * np.pi)
        0.014904383867336446

        >>> GearGeometry._involute(15 / 180 * np.pi)
        0.006149804631973288
        """

        return np.tan(angle) - angle
    
    def _to_tau(self, alpha_i: float) -> float:
        """
        Transforms the alpha angle, used to build the involute profile, into the integration variable tau.

        :math:`tau(alpha_i) = alpha_i - self.theta_b + self.involute(alpha_i)`
        
        Args
        ----------
        alpha_i : float
            An angle within the involute profile.

        Returns 
        ---------
        float
            tau_i
    
        Examples
        ---------
        >>> self._to_tau(31 * np.pi / 180)
        0.5573963019457713

        References
        --------
        Ma, H., Pang, X., Song, R., & Yang, J. (2014). 基于改进能量法的直齿轮时变啮合刚度计算 
        [Time-varying mesh stiffness calculation of spur gears based on improved energy method].
        Journal of Northeastern University (Natural Science), 35(6), 863–867. https://doi.org/10.3969/j.issn.1005-3026.2014.06.023
        """

        geometry = self.geometryDict

        return alpha_i - geometry['theta_b'] + GearGeometry._involute(alpha_i)
    
def gearGeometryExample() -> None:
    gear1 = GearElement(21, 12, 12, 12, 2e-3, 55, 2e-2)
    print(gear1.geometry.geometryDict)
    pass

if __name__ == '__main__':
    gearGeometryExample()


