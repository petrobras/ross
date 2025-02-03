"""Gear Element module.

This module defines the GearElement classes which can be used to represent
gears or gearboxes used to couple different shafts in the MultiRotor class.
"""

import numpy as np
from plotly import graph_objects as go
from ross.units import Q_
from abc import ABC, abstractmethod

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

        self.geometry: GearGeometry = GearGeometry(self)


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

        # Curves
        self.transition = TransitionCurve(self.gear)
        self.involute = InvoluteCurve(self.gear)

        # Initialize the geometryDict
        self.geometryDict: dict[str, float] = {}
        self.geometryDict:  dict[str, float] = self._initialize_geometry(self.geometryDict)

        pass
    
    def _initialize_geometry(self, dict_geo: dict) -> dict[str, float]:
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
        geometryDict = dict_geo
        # Add radii constants
        radii:      dict[str, float] = self._notable_radii()
        geometryDict |= radii 

        # Add angular constants of the involute profile
        angles:     dict[str, float] = self._notable_angles()
        geometryDict |= angles

        # Add geometric constants of the tooth profile
        geo_const:  dict[str, float] = self._geometric_constants()
        geometryDict |= geo_const

        # Add geometric constants of tau values of the tooth profile
        tau_const:  dict[str, float] = self._tau_constants()
        geometryDict |= tau_const

        return geometryDict

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


        # pressure angle when the contact point is on the addendum circle [rad] MAOK
        alpha_a: float = np.arccos(self.geometryDict['r_b'] / self.geometryDict['r_a']) 

        # pressure angle when the contact point is on the C point [rad] MAOK
        alpha_c: float = np.arccos(self.geometryDict['r_b'] / self.geometryDict['r_c'])       
        
        # The angle between the tooth center-line and de junction with the root circle [radians]
        theta_f: float = (                                                          
            1 / gear.n_tooth * ( 
                np.pi / 2 
                + 2 * np.tan(gear.pressure_angle) * (gear.ha_ - self.geometryDict['r_rho_'])
                + 2 * self.geometryDict['r_rho_'] / np.cos(gear.pressure_angle)
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
        r_f: float = 1 / 2 * gear.module * gear.n_tooth - (gear.ha_ + gear.c_) * gear.module

        # radius of cutter tip round corner [m] MAOK
        r_rho: float = gear.c_ * gear.module / (1 - np.sin(gear.pressure_angle) )

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

        a1: float = (self.gear.ha_ + self.gear.c_) * self.gear.module - self.geometryDict['r_rho']  # MAOK
        b1: float = (                                                       # MAOK
            np.pi * self.gear.module / 4 + self.gear.ha_ * self.gear.module * np.tan(self.gear.pressure_angle) 
            + self.geometryDict['r_rho'] * np.cos(self.gear.pressure_angle) 
        )

        dict_place: dict[str, float] = {
            'a_1': a1,
            'b_1': b1
        }

        return dict_place

    def _tau_constants(self) -> dict[str, float]:

        tau_c: float = self._to_tau(self.geometryDict['alpha_c'])
        tau_a: float = self._to_tau(self.geometryDict['alpha_a'])  

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

        return alpha_i - self.geometryDict['theta_b'] + GearGeometry._involute(alpha_i)

    def plot_tooth_geometry(self) -> None:
        
        """
        Plot the geometry of the tooth profile.
        """

        gear: GearElement = self.gear
        geometry: dict[str, float] = gear.geometry

        # Generate the transition geometry
        transition: np.array = np.linspace(np.pi/2, gear.pressure_angle, 200)
        transition_vectorized: np.vectorize = np.vectorize(self.transition._compute_transition_curve)
        y_t, x_t, _, _ = transition_vectorized(transition)

        # Generate the involute geometry
        involute = np.linspace(self.geometryDict['alpha_c'], self.geometryDict['alpha_a'], 200)
        tau_vectorize = np.vectorize(self._to_tau)
        tau_values = tau_vectorize(involute)

        involute_vectorize = np.vectorize(self.involute._compute_involute_curve)
        y_i, x_i, _, _ = involute_vectorize(tau_values)

        # Create the plot
        fig = go.Figure()

        # Add the transition curve
        fig.add_trace(go.Scatter(x=y_t[-1::-1], y=x_t[-1::-1], mode='lines', name='Transition Curve'))

        # Add the involute curve
        fig.add_trace(go.Scatter(x=y_i, y=x_i, mode='lines', name='Involute Curve'))

        # Update layout with gridlines
        fig.update_layout(
            title='Tooth Profile Geometry',
            xaxis_title='Y-axis [m]',
            yaxis_title='X-axis [m]',
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )

        # Show the plot
        fig.show()
        pass

class GeometryProfile(ABC):
    
    @abstractmethod
    def _y(*args, **kwargs):
        pass

    @abstractmethod
    def _x(*args, **kwargs):
        pass

    @abstractmethod
    def _area(*args, **kwargs):
        pass
    
    @abstractmethod
    def _I_y(*args, **kwargs):
        pass

class TransitionCurve(GeometryProfile):
    """
    Transition curve class.

    Used to compute the geometric transition curve profile for a single gamma.

    Based on:
    Ma, H., Song, R., Pang, X., & Wen, B. (2014). Time-varying mesh stiffness calculation of cracked spur gears. 
    Engineering Failure Analysis, 44, 179–194. https://doi.org/10.1016/j.engfailanal.2014.05.018

    Parameters
    ----------
    -`gear` : GearElement

    Methods 
    -------
    -`_x` -> float
        x-coordinates of the transition curve.
    -`_y` -> float
        y-coordinates of the transition curve.
    -`_A_y` -> float
        Area values based on x_1.
    -`_I_y` -> float
        Moment of inertia values based on x_1.
    """
    def __init__(self, gear: GearElement):
        self.gear: GearElement = gear

    def _compute_transition_curve(self, gamma: float) -> tuple[float, float, float, float]:
        """Compute the y, x, area and I_y of the transition curve given a gamma angle.

        Args:
            gamma (float): The angle indicating the position on the transition curve profile.

        Returns:
            - y (float): The y-coordinate of the transition curve at gamma.
            - x (float): The x-coordinate of the transition curve at gamma.
            - area (float): The cumulative area under the transition curve up to x.
            - I_y (float): The second moment of area (moment of inertia) about the y-axis at x.
        """
        self.geometryDict: dict[str, float] = self.gear.geometry.geometryDict
        phi: float = self._phi(gamma)
        y: float = self._y(gamma, phi)
        x: float = self._x(gamma, phi)
        area: float = self._area(x)
        I_y: float = self._I_y(x) 

        return y, x, area, I_y

    def _phi(self, gamma: float) -> float:
        
        phi: float = (
            (
                self.geometryDict['a_1'] 
                / np.tan(gamma) 
                + self.geometryDict['b_1']
            ) 
            / self.geometryDict['r_p']
        )

        return phi
        
    def _y(self, gamma: float, phi: float) -> float:
        
        y: float = (
            self.geometryDict['r_p'] 
            * np.cos(phi) 
            - (
                self.geometryDict['a_1'] 
                / np.sin(gamma) 
                + self.geometryDict['r_rho']
            ) 
            * np.sin(gamma-phi)
        )

        return y 
    
    def _x(self, gamma: float, phi: float) -> float:

        x: float = (
            self.geometryDict['r_p']
            * np.sin(phi)
            - (
                self.geometryDict['a_1']
                / np.sin(gamma)
                + self.geometryDict['r_rho']
            )
            * np.cos(gamma - phi)
        )

        return x
    
    def _area(self, x_gamma: float) -> float:
        """Evaluate the transition curve area given a gamma angle. 

        Args:
            gamma (float): _description_

        Returns:
            float: _description_
        """

        area: float = 2 * x_gamma * self.gear.width

        return area
    
    def _I_y(self, x_gamma: float) -> float:
        """Evaluate the transition curve area moment of inertia for the y-axis given a gamma angle. 

        Args:
            gamma (float): positional angle

        Returns:
            float: Area moment of inertia for the y-axis.
        """

        I_y: float = 2/3 * x_gamma**3 * self.gear.width

        return I_y

class InvoluteCurve(GeometryProfile):
    """
    Class to compute the geometric involute curve profile.

    Parameters
    --------
        -`gear`: GearElement

    Methods 
    -------
        -`_x` -> float
            x-coordinates of the involute curve.
        -`_y` -> float
            y-coordinates of the involute curve.
        -`_area` -> float
            Area values based on _x.
        -`_I_y` -> float
            Moment of inertia values based on _x.
    
    Example
    --------
    >>> self.involute_curve(0.3405551128775112)
    (0.0014447795342141163, 0.055344111158185265, 5.7791181368564653e-05, 4.02108709529994e-11)

    References
    ---------
    From Ma, H., Song, R., Pang, X., & Wen, B. (2014). Time-varying mesh stiffness calculation of cracked spur gears. 
    Engineering Failure Analysis, 44, 179–194. https://doi.org/10.1016/j.engfailanal.2014.05.018

    """
    
    def __init__(self, gear: GearElement):
        self.gear = gear

    def _compute_involute_curve(self, tau_i: float) -> tuple[float, float, float, float]:
        """Compute the y, x, area and I_y of the involute curve given a tau angle.

        Args:
            tau (float): The angle indicating the position on the involute curve profile.

        Returns:
            - y (float): The y-coordinate of the transition curve at tau_i.
            - x (float): The x-coordinate of the transition curve at tau_i.
            - area (float): The area under the transition curve up to x.
            - I_y (float): The second moment of area (moment of inertia) about the y-axis at x.
        """
        self.geometryDict: dict[str, float] = self.gear.geometry.geometryDict
        
        y: float = self._y(tau_i)
        x: float = self._x(tau_i)
        area: float = self._area(x)
        I_y: float = self._I_y(x) 

        return y, x, area, I_y

    def _y(self, tau_i: float) -> float:
        
        y: float = (
            self.geometryDict['r_b']
            * (
                (
                    tau_i + self.geometryDict['theta_b']
                )
            * np.sin(tau_i)
            + np.cos(tau_i)
            )
        ) 

        return y

    def _x(self, tau_i: float) -> float:

        x: float = (
            self.geometryDict['r_b']
            * (
                (
                    tau_i + self.geometryDict['theta_b']
                )
            * np.cos(tau_i)    
            - np.sin(tau_i)            
            )
        )

        return x

    def _area(self, x: float) -> float:

        area: float = 2 * x * self.gear.width

        return area
    
    def _I_y(self, x: float) -> float:


        I_y: float = 2/3 * x**3 * self.gear.width

        return I_y
    
def gearGeometryExample() -> None:
    gear1 = GearElement(21, 12, 12, 12, 2e-3, 55, 2e-2)
    print(gear1.geometry.geometryDict)
    pass

def gearInvoluteCurve() -> None:
    gear1 = GearElement(21, 12, 12, 12, 2e-3, 55, 2e-2)
    gear1.geometry.plot_tooth_geometry()

if __name__ == '__main__':
    gearInvoluteCurve()


