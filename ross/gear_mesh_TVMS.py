"""Gear Element module.

This module defines the GearElementTVMS classes which can be used to represent
gears or gearboxes used to couple different shafts in the MultiRotor class.
"""

import numpy as np
from plotly import graph_objects as go
from ross.units import Q_
from ross.materials import Material
from abc import ABC, abstractmethod
import pandas as pd
import scipy as sp

from ross.disk_element import DiskElement


__all__ = ["GearElementTVMS", "Mesh"]


class GearElementTVMS(DiskElement):
    """A gear element.

    This class creates a gear element from input data of inertia and mass.

    Parameters
    ----------
    n: int
        Node in which the gear will be inserted.
    m : float, pint.Quantity
        Mass of the gear element (kg).
    module  : float
        Gear module (m).
    n_tooth : int
        Tooth number of the gear.
    width   : float
        Width of the gear (m).
    hub_bore_radius     : float
        The hub bore radius (m).
    material    : Material
        Gear's construction material.
    pressure_angle : float, optional
        The pressure angle of the gear (rad).
        Default is 20 deg (converted to rad).
    addendum_coeff  : float | optional
        Addendum coefficient of the gear.
        Default is 1.
    clearance_coeff : float | optional
        Gear's clearance coefficient.
        Default is 0.25.
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
    >>> gear = GearElementTVMS(
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
        module,
        n_tooth,
        width,
        hub_bore_radius,
        material: Material = Material(name="Steel", rho=Q_(7.81, 'g/cm**3'), E=211e9, G_s=81.2e9), 
        pressure_angle=20 * np.pi / 180,
        addendum_coeff=1,
        clearance_coeff=0.25,
        tag=None,
        scale_factor=1.0,
        color="Goldenrod",
    ):
        if pressure_angle is None:
            pressure_angle = Q_(20, "deg")

        self.r_shaft: float = hub_bore_radius # [m]
        self.module: float = module # [m]
        self.n_tooth: float = n_tooth # [-]
        self.pressure_angle: float = float(pressure_angle) #[rad]
        self.width: float = width # [m]
        self.ha_: float = addendum_coeff #[-]
        self.c_: float = clearance_coeff #[-]
        self.pitch_diameter: float = module * n_tooth #[m]

        self.Ip = np.pi/2 * ((self.pitch_diameter/2)**4 - self.r_shaft**4)
        self.Id = np.pi/4 * ((self.pitch_diameter/2)**4 - self.r_shaft**4)

        self.material: Material = material
        
        # Initialize the geometryDict
        self.geometryDict: dict[str, float] = {}
        self.geometryDict:  dict[str, float] = self._initialize_geometry(self.geometryDict)

        self.stiffness: GearStiffness = GearStiffness(self)

        super().__init__(n, m, self.Id, self.Ip, tag, scale_factor, color)

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
        >>> gear = GearElementTVMS.from_geometry(0, steel, 0.07, 0.05, 0.28)
        >>> gear.r_b # doctest: +ELLIPSIS
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
        radius = self.geometryDict['r_b'] * 1.1 + 0.05

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

        customdata = [self.n, self.Ip, self.Id, self.m, self.geometryDict['r_b'] * 2]
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

# para conseguir acessar o tempo naquele instante


        pressure_angle = self.gearInput.pressure_angle

        
        Keq, _, _ = self.mesh(self, time)

        Kxx, _, _ = np.cos(pressure_angle) * self.mesh(self, time)
        Kyy, _, _ = np.sin(pressure_angle) * self.mesh(self, time)

        K = np.array([  [Kxx, 0., 0., 0., 0., 0.],
                        [0., Kyy, 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0.]])
        
        return K

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

        # pressure angle when the contact point is on the addendum circle [rad] MAOK
        alpha_a: float = np.arccos(self.geometryDict['r_b'] / self.geometryDict['r_a']) 

        # pressure angle when the contact point is on the C point [rad] MAOK
        alpha_c: float = np.arccos(self.geometryDict['r_b'] / self.geometryDict['r_c'])       
        
        # The angle between the tooth center-line and de junction with the root circle [radians]
        theta_f: float = (                                                          
            1 / self.n_tooth * ( 
                np.pi / 2 
                + 2 * np.tan(self.pressure_angle) * (self.ha_ - self.geometryDict['r_rho_'])
                + 2 * self.geometryDict['r_rho_'] / np.cos(self.pressure_angle)
            ) 
        )

        theta_b: float = np.pi / (2*self.n_tooth) + self._involute(self.pressure_angle)

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

        # radius of base circle [m] MAOK
        r_b: float = 1 / 2 * self.module * self.n_tooth * np.cos(self.pressure_angle)

        # radius of pitch circle [m] MAOK
        r_p: float = r_b / np.cos(self.pressure_angle)

        # radius of addendum circle [m]                                  
        r_a: float = r_p + self.ha_ * self.module                                        
        
        # radii of the involute starting point [m] MAOK
        r_c: float = (                                                                 
            np.sqrt( 
                np.square(r_b * np.tan(self.pressure_angle) 
                - self.ha_ * self.module /  np.sin(self.pressure_angle) ) 
                + np.square(r_b) 
            ) 
        )

        # radius of root circle [m] MAOK
        r_f: float = 1 / 2 * self.module * self.n_tooth - (self.ha_ + self.c_) * self.module

        # radius of cutter tip round corner [m] MAOK
        r_rho: float = self.c_ * self.module / (1 - np.sin(self.pressure_angle) )

        # normalized by the module radius of cutter tip round corner [m]                  
        r_rho_: float = r_rho / self.module                                                         

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

        a1: float = (self.ha_ + self.c_) * self.module - self.geometryDict['r_rho']  # MAOK
        b1: float = (                                                       # MAOK
            np.pi * self.module / 4 + self.ha_ * self.module * np.tan(self.pressure_angle) 
            + self.geometryDict['r_rho'] * np.cos(self.pressure_angle) 
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
        phi: float = (
            (
                self.geometryDict['a_1'] 
                / np.tan(gamma) 
                + self.geometryDict['b_1']
            ) 
            / self.geometryDict['r_p']
        )

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

        area: float = 2 * x * self.width
        I_y: float = 2/3 * x**3 * self.width

        return y, x, area, I_y

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
        self.geometryDict: dict[str, float] = self.geometryDict
        
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

        area: float = 2 * x * self.width
        I_y: float = 2/3 * x**3 * self.width

        return y, x, area, I_y

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

        return alpha_i - self.geometryDict['theta_b'] + self._involute(alpha_i)

    def plot_tooth_geometry(self) -> None:
        
        """
        Plot the geometry of the tooth profile.
        """

        geometry: dict[str, float] = self.geometry

        # Generate the transition geometry
        transition: np.array = np.linspace(np.pi/2, self.pressure_angle, 200)
        transition_vectorized: np.vectorize = np.vectorize(self._compute_transition_curve)
        y_t, x_t, _, _ = transition_vectorized(transition)

        # Generate the involute geometry
        involute = np.linspace(self.geometryDict['alpha_c'], self.geometryDict['alpha_a'], 200)
        tau_vectorize = np.vectorize(self._to_tau)
        tau_values = tau_vectorize(involute)

        involute_vectorize = np.vectorize(self._compute_involute_curve)
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

class GearStiffness:
    """
    A class which evaluates the stiffness of a single gear.

    Parameters:
    --------
    gear: Gear object

    """

    def __init__(self, gear: GearElementTVMS):
        self.gear = gear 
        self.geometryDict = gear.geometryDict
        self.material = gear.material

    def diff_tau(self, tau_i) -> float:
        """
        Method for evaluating the stiffness commonly found in the integrative functions of the involute region on a specified angle tau_i.

        Parameters
        ----------
        tau_i : float
            Operational tau_i angle.

        Returns
        ----------
        float
            The value of diff_tau(tau_i)
        """
        return self.geometryDict['r_b'] * (tau_i + self.geometryDict['theta_b']) * np.cos(tau_i) # OK

    def diff_gamma(self, gamma) -> float:
        """
        Method used in evaluating the stiffness commonly found in the integrative functions of the transition region.

        Parameters
        ----------
        gamma : float
            Value of the gamma angle used to describre the profile.

        Returns
        ----------
        float
            The value of diff_gamma(gamma)
        """
        a1 = self.geometryDict['a_1']
        b1 = self.geometryDict['b_1']
        r_p = self.geometryDict['r_p']
        r_rho = self.geometryDict['r_rho']

        term_1 = (
                ( 
                    a1 * np.sin( (a1/np.tan(gamma) + b1) / r_p ) 
                    * (1 + np.square( np.tan(gamma) ) )
                ) 
                / np.square( np.tan(gamma) )
            )
        
        term_2 =  (
            a1 * np.cos( gamma ) 
            / np.square( np.sin( gamma ) ) 
            * np.sin( 
                gamma - (a1 / np.tan( gamma ) + b1)
                / r_p 
                )
            )
        
        term_3 = (
                - ( a1 / np.sin( gamma ) + r_rho ) 
                * np.cos( gamma - ( a1 / np.tan( gamma ) + b1 ) / r_p )  
                * (
                    1 + a1 * ( 1 + np.square( np.tan( gamma ) ) ) 
                    / (r_p * np.square( np.tan ( gamma ) ) )
                )
            )

        return term_1 + term_2 + term_3 # OK 

    def compute_stiffness(self, alpha_op) -> tuple[float, float, float, float]:
        """
        Computes the stiffness in the direction of the applied force on the gear (line of action), according to the involute profile. 
        It evaluates each of them separetly, and those values are returned as 1/stiffness for an approach in accordance with Ma, H. et. al. (2014).
        
        Parameters
        ----------
        tau_op : float
            The tau operational angle, e.g. the angle formed by the normal of the contact involute curves and the x axis. 
        
        Returns
        ----------
        tuple of float
            A tuple containing the computed stiffness components as 1/stiffness values. The elements represent:
            - ka : float
                Stiffness related to axial stresses.
            - kb : float
                Stiffness related to bending stresses.
            - kf : float 
                Stiffness related to body of the gear.
            - ks : float
                Stiffness related to shear stresses.
        """
        tau_op = self.gear._to_tau(alpha_op)

        inv_kf = self.inv_kf(tau_op)
        inv_ka = self.inv_ka(tau_op)
        inv_kb = self.inv_kb(tau_op)
        inv_ks = self.inv_ks(tau_op)

        if (np.isnan(inv_kf)):
            return inv_ka, inv_kb, inv_ks
        
        return  1/inv_ka, 1/inv_kb, 1/inv_kf, 1/inv_ks

    def _gear_body_polynominal(self) -> pd.DataFrame | str:
        """
        This method uses the approach described by Sainsot et al. (2004) to calculate the stiffness factor (kf) 
        contributing to tooth deflections. If the parameters fall outside the experimental range used to derive 
        the analytical formula, the method returns `'oo'` to indicate an infinite stiffness approximation.

        Returns
        ---------
        float
            The calculated stiffness factor (kf) for the gear base. 
        
        str
            Return 'oo' if it doesn't match the criteria for the experimental range where this method was built.
        """

        h = self.geometryDict['r_f'] / self.gear.r_shaft

        poly = pd.DataFrame()
        poly['var'] = ['L'          , 'M',            'P'           , 'Q']
        poly['A_i'] = [-5.574e-5    , 60.111e-5     , -50.952e-5    , -6.2042e-5]
        poly['B_i'] = [-1.9986e-3   , 28.100e-3     , 185.50e-3     , 9.0889e-3]
        poly['C_i'] = [-2.3015e-4   , -83.431e-4    , 0.0538e-4     , -4.0964e-4]
        poly['D_i'] = [4.7702e-3    , -9.9256e-3    , 53.300e-3     , 7.8297e-3]
        poly['E_i'] = [0.0271       ,  0.1624       , 0.2895        , -0.1472]
        poly['F_i'] = [6.8045       , 0.9086        , 0.9236        , 0.6904]

        calculate_x_i = lambda row: (
            row['A_i'] / (self.geometryDict['theta_f'] ** 2)
            + row['B_i'] * h ** 2
            + row['C_i'] * h / self.geometryDict['theta_f']
            + row['D_i'] / self.geometryDict['theta_f']
            + row['E_i'] * h
            + row['F_i']
        ) # OK

        poly['X_i'] = poly.apply(lambda row: calculate_x_i(row), axis=1)
        
        limits = {
            'L': (6.82, 6.94),
            'M': (1.08, 3.29),
            'P': (2.56, 13.47),
            'Q': (0.141, 0.62)
        }

        for index, row in poly.iterrows():
            var_name = row['var']
            X_i = row['X_i']
            lower_limit, upper_limit = limits[var_name]

            # if (not lower_limit <= X_i <= upper_limit) :#or (not 1.4 <= h <= 7) or (not 0.01 <= gear.theta_f <= 0.12):
            #     # for the stiffness on the base of the tooth to match the model, it has to match those criteria above. If not, kf -> oo.
            #     global contador
            #     contador+=1 
            #     return 'oo'


        return poly.loc[:,['var','X_i']]
    
    def inv_kf(self, tau_op) -> float:
        """
        Sainsot, P., Velex, P., & Duverger, O. (2004). Contribution of gear body to tooth deflections - A new bidimensional analytical 
        formula. Journal of Mechanical Design, 126(4), 748–752. https://doi.org/10.1115/1.1758252

        Calculate the stiffness contribution from the gear base, given a point on the involute curve.

        Parameters
        ---------
        tau_op (float): 
            The operational pressure angle (tau) in radians. This angle is used to determine the stiffness characteristics 
            based on the gear geometry and material properties. It's the current angle of the contact point between the meshing gears.

        Returns
        ---------
        float: 
            The calculated 1/kf for the gear base.
    
        """

        # obtain a dataframe of polynomials coefficients
        poly = self._gear_body_polynominal()

        # Extrapolating the range of interpolation described by Sainsot et. al. (2014).
        # if type(poly) == str:
        #     return 0

        L_poly, M_poly, P_poly, Q_poly =  poly['X_i']

        y,_, _, _ = self.gear._compute_involute_curve(tau_op)

        Sf = 2 * self.geometryDict['r_f'] * self.geometryDict['theta_f']
        u = y - self.geometryDict['r_f']

        kf = (
            ( np.cos(tau_op)**2 / (self.material.E * self.gear.width) )
            * ( 
                L_poly * ( u /Sf)**2 
                + M_poly * u / Sf 
                + P_poly * (1 + Q_poly * np.tan(tau_op)**2 )
            ) 
        )

        return kf

    def inv_ks(self, tau_op) -> float:
        """
        Calculate the stiffness contribution from the gear resistance from shear stresses, given the tau operational angle.

        Parameters
        ----------
        tau_op
            Operational tau angle.
        
        Returns
        ---------
        float
            The shear stiffness in the form of 1/ks.
        """        
        
        f_transiction = lambda gamma: (
            1.2 * np.cos(tau_op)**2 
            / (
                self.material.G_s 
                * self.gear._compute_transition_curve(gamma)[2]
            ) 
            * self.diff_gamma(gamma)
        ) 

        k_transiction, _ = sp.integrate.quad(
            f_transiction, 
            np.pi/2, 
            self.gear.pressure_angle
        ) # OK

        f_involute = lambda tau: (
            1.2 * (
                np.cos(tau_op)**2
                / (self.material.G_s*self.gear._compute_involute_curve(tau)[2]) # verificar se esse 2 é a área msm
                * self.diff_tau(tau)
            )
        ) 

        k_involute, _ = sp.integrate.quad(
            f_involute, 
            self.gear._to_tau(self.geometryDict['alpha_c']), 
            tau_op
        ) 

        return (k_transiction + k_involute)

    def inv_kb(self, tau_op) -> float:
        """
        Calculate the stiffness contribution from the gear resistance from bending stresses, given the tau operational angle.

        Parameters
        ----------
        tau_op
            Operational tau angle.
        
        Returns
        ---------
        float
            The bending stiffness in the form of 1/kb.
        """        
        y_op, x_op, _, _ = self.gear._compute_involute_curve(tau_op)

        f_transiction = lambda gamma: (
            (
                np.cos(tau_op) 
                * (
                    y_op - self.gear._compute_transition_curve(gamma)[0]
                )
                - x_op * np.sin(tau_op)
            )**2 
            / (
                self.material.E 
                * self.gear._compute_transition_curve(gamma)[3]
            )
            * self.diff_gamma(gamma)
        ) 

        k_transiction, _ = sp.integrate.quad(f_transiction, np.pi/2, self.gear.pressure_angle)

        f_involute = lambda tau: (
            (
                np.cos(tau_op) 
                * (y_op - self.gear._compute_involute_curve(tau)[0])
                - x_op * np.sin(tau_op)
            )**2
            / (self.material.E * self.gear._compute_involute_curve(tau)[3])
            * self.diff_tau(tau)
        ) 

        k_involute, _ = sp.integrate.quad(
            f_involute, 
            self.gear._to_tau(self.geometryDict['alpha_c']), 
            tau_op
        )

        return (k_transiction + k_involute)

    def inv_ka(self, tau_op: float) -> float:
        """
        Calculate the stiffness contribution from the gear resistance from axial stresses, given the tau operational angle.

        Parameters
        ----------
        tau_op
            Operational tau angle.
        
        Returns
        ---------
        float
            The axial stiffness in the form of 1/ka.
        """        

        f_transiction = lambda gamma: (
            np.sin(tau_op)**2
            / (self.material.E * self.gear._compute_transition_curve(gamma)[2])
            * self.diff_gamma(gamma)
        ) 

        k_transiction, _ = sp.integrate.quad(f_transiction, np.pi/2, self.gear.pressure_angle)

        f_involute = lambda tau: (
            np.sin(tau_op)**2 
            / (self.material.E * self.gear._compute_involute_curve(tau)[2])
            * self.diff_tau(tau)
        ) 

        k_involute, _ = sp.integrate.quad(
            f_involute, 
            self.gear._to_tau(self.geometryDict['alpha_c']), 
            tau_op
        )

        return (k_transiction + k_involute)

    @staticmethod
    def kh(gear1: GearElementTVMS, gear2: GearElementTVMS) -> float:
        """
        Evaluates the contact hertzian stiffness considering that both elasticity modulus are equal.

        Parameters
        ----------
        gear1
            Gear object.
        
        gear2
            Gear object.

        Returns
        ----------
        float
            The hertzian contact stiffness.

        Notes
        ----------
        - It returns kh, not 1/kh.
        """

        return np.pi * gear1.material.E * gear1.width / 4 / (1 - gear1.material.Poisson**2)

class Mesh:
    """
    Represents the meshing behavior between two gears, typically a pinion and a crown gear, 
    including stiffness and contact ratio calculations.

    Parameters:
    -----------
    pinion : Gear
        The pinion gear object used in the gear pair (driver).
    gear : Gear
        The crown gear object used in the gear pair (driven).
    pinion_w : float
        The rotational speed [rad/sec] of the pinion gear in radians per second.
        
    Attributes:
    -----------
    gear : Gear
        The gear wheel object, which contains information about the geometry and properties of the wheel gear.
    pinion : Gear
        The pinion gear object, which contains information about the geometry and properties of the pinion gear.
    tm : float
        The meshing period, calculated based on the rotational speed and the number of teeth of the pinion.
    kh : float
        Hertzian stiffness of 2 tooth in contact (same Elasticity Modulus).
    k_mesh : float
        The equivalent stiffness of the gear mesh, combining the stiffness of the pinion and crown.
    cr : float
        The contact ratio, representing the average number of tooth in contact during meshing.

    """

    def __init__(self, gearInput: GearElementTVMS, gearOutput: GearElementTVMS):
        self.gearInput = gearInput
        self.gearOutput = gearOutput


        self.time = 0

        self.eta = gearOutput.n_tooth / gearInput.n_tooth # Gear ratio 
        
        self.kh = GearStiffness.kh(gearInput, gearOutput)
        self.cr = self.contact_ratio(self.gearInput, self.gearOutput)

    @staticmethod
    def contact_ratio(gearInput: GearElementTVMS, gearOutput: GearElementTVMS) -> float:
        """
        Parameters:
        ---------
        pinion : Gear
            The pinion object.
        gear : Gear
            The wheel object.
        
        Returns
        -------
        CR : float
            The contact ratio of the gear pair.

        Example
        -------
        >>> Mesh.contact_ratio(pinion, Gear)
        1.7939883590132295

        Reference
        ----------
        Understanding the contact ratio for spur gears with some comments on ways to read a textbook.
        Retrieved from: https://www.myweb.ttu.edu/amosedal/articles.html
        """

        pb = np.pi * 2 * gearInput.geometryDict['r_b'] / gearInput.n_tooth # base pitch
        C = gearInput.geometryDict['r_p'] + gearOutput.geometryDict['r_p'] # center distance (not the operating one)

        lc = ( # length of contact (not the operating one) # OK
            np.sqrt(gearInput.geometryDict['r_a']**2 - gearInput.geometryDict['r_b']**2) 
            + np.sqrt(gearOutput.geometryDict['r_a']**2 - gearOutput.geometryDict['r_b']**2) 
            - C * np.sin(gearInput.pressure_angle)
        )

        CR = lc / pb # contact ratio
    
        return CR

    def time_equivalent_stiffness(self, t: float, gearInputSpeed: float) -> float:
        """
        Parameters
        ---------
        gearInput : Gear
            The gearInput object.
        gear: Gear
            The gear object.
        t: float
            The time of meshing [0, self.tm]
        
        Returns
        -------
        float
            The equivalent stiffness of the contact based on the time [0, self.tm] of mesh.
        
        Example
        --------
        >>> self.time_equivalent_stiffness()
        167970095.70859054
        """

        gearOutputSpeed = - gearInputSpeed / self.eta


        # Angular displacements 
        alphaGearInput = t * gearInputSpeed + self.gearInput.geometryDict['alpha_c'] # angle variation of the input pinion [rad]
        alphaGearOutput   = t * gearOutputSpeed + self.gearOutput.geometryDict['alpha_a']   # angle variation of the output gear   [rad]
        
        # Tau displacementes
        dTauGearInput = self.gearInput._to_tau(alphaGearInput)     # angle variation of the pinion in tau [rad]
        dTauGearOutput   = self.gearOutput._to_tau(alphaGearOutput)     # angle variation of the pinion in tau [rad]

        # Contact stiffness according to tau angles
        ka_1, kb_1, kf_1, ks_1 = self.gearInput.stiffness.compute_stiffness(dTauGearInput)
        ka_2, kb_2, kf_2, ks_2 = self.gearOutput.stiffness.compute_stiffness(dTauGearOutput)

        # Evaluating the equivalate meshing stiffness. 
        k_t = 1 / (1/self.kh + 1/ka_1 + 1/kb_1 + 1/kf_1 + 1/ks_1 + 1/ka_2 + 1/kb_2 + 1/kf_2 + 1/ks_2)

        return k_t, dTauGearInput, dTauGearOutput
    


    def mesh(self, t, gearInputSpeed: float,  interpolation: bool = False):
        """
        Calculate the time-varying meshing stiffness of a gear pair.

        This method computes the equivalent stiffness of a gear mesh at a given time `t`, taking into
        account the periodic nature of the meshing process and the contact ratio (`cr`) of the gear pair.
        The computation considers whether one or two pairs of teeth are in contact during the meshing cycle.

        Parameters
        ----------
        pinion : Gear
            The pinion gear object containing properties needed for stiffness calculation.
        gear : Gear
            The gear object containing properties needed for stiffness calculation.
        t : float
            Time instant for which the meshing stiffness is calculated.

        Returns
        -------
        tuple
            - float: The total equivalent meshing stiffness at time `t`.
            - float: The meshing stiffness of the first tooth pair in contact (returns `np.nan` if not applicable).
            - float: The meshing stiffness of the second tooth pair in contact.

        Notes
        -----
        - The calculation considers the periodic nature of meshing and the contact ratio (cr) of the gear pair.
        - The stiffness contribution varies depending on whether one or two pairs of teeth are in contact.
        - For the correct evaluation of the TVMS it's important that `dt = (2 * np.pi) / (20 * n_toth_gear_x * speed_gear_x)`
        """


        if interpolation == True: # Runs the time dependency for one period of double-single mesh

            if hasattr(self, 'already_evaluated') == False: # Case 1: If it had never evaluated the stiffness

                dt = 1e-5
                t_interpol, double_contact, single_contact = self._time_stiffness(gearInputSpeed, dt)

                mask_double_contact= double_contact > 0 
                self.double_contact = double_contact[mask_double_contact]
                self.t_interpol_double = t_interpol[mask_double_contact]

                mask_single_contact = single_contact > 0
                self.single_contact = single_contact[mask_single_contact]
                self.t_interpol_single = t_interpol[mask_single_contact]

                self.already_evaluated = True
        
            # Case 2: If the stiffness is already known

            tm = 2 * np.pi / (gearInputSpeed * self.gearInput.n_tooth) # Gearmesh period [seconds/engagement]
            ctm = self.cr * tm # [seconds/tooth] how much time each tooth remains in contact

            t = t - t // tm * tm
            
            if t <= (self.cr-1) * tm:
                return np.interp(t, self.t_interpol_double, self.double_contact), None, None

            elif t > (self.cr-1) * tm:
                return np.interp(t, self.t_interpol_single, self.single_contact), None, None

        else: # If it needs to re-evaluate every stiffness integration every step

            tm = 2 * np.pi / (gearInputSpeed * self.gearInput.n_tooth) # Gearmesh period [seconds/engagement]
            ctm = self.cr * tm # [seconds/tooth] how much time each tooth remains in contact

            t = t - t // tm * tm
            
            if t <= (self.cr-1) * tm:
                stiffnessMesh1, d_tau_pinion1, d_tau_gear1 = self.time_equivalent_stiffness(t, gearInputSpeed)
                stiffnessMesh0,  d_tau_pinion0, d_tau_gear0 = self.time_equivalent_stiffness(tm + t, gearInputSpeed)

                return stiffnessMesh0 + stiffnessMesh1, stiffnessMesh0, stiffnessMesh1
            
            elif t > (self.cr-1) * tm:
                stiffnessMesh1, d_tau_pinion1, d_tau_gear1 = self.time_equivalent_stiffness(t, gearInputSpeed)
                
                return stiffnessMesh1, np.nan, stiffnessMesh1
    
    def _time_stiffness(self, gearInputSpeed: float, dt: float) -> float:

        tm = 2 * np.pi / (gearInputSpeed * self.gearInput.n_tooth) # Gearmesh period [seconds/engagement]
        ctm = self.cr * tm # [seconds/tooth] how much time each tooth remains in contact

        t_interpol = np.arange(0, tm+dt, dt)
        double_contact = np.zeros(np.shape(t_interpol))
        single_contact = np.zeros(np.shape(double_contact))

        for i, t in enumerate(t_interpol):
            
            t = t - t // tm * tm

            if t <= (self.cr-1) * tm:
                stiffnessMesh1, _, _ = self.time_equivalent_stiffness(t, gearInputSpeed)
                stiffnessMesh0,  _, _ = self.time_equivalent_stiffness(tm + t, gearInputSpeed)

                double_contact[i] = stiffnessMesh0 + stiffnessMesh1
            
            elif t > (self.cr-1) * tm:
                stiffnessMesh1, _, _ = self.time_equivalent_stiffness(t, gearInputSpeed)
                
                single_contact[i] = stiffnessMesh1
        
        return t_interpol, double_contact, single_contact

def gearGeometryExample() -> None:
    gear1 = GearElementTVMS(21, 12, 12, 12, 2e-3, 55, 2e-2)
    print(gear1.geometryDict)
    pass

def gearMeshStiffnessExample() -> None:
    gear1 = GearElementTVMS(n=21, m=12, module=2e-3, width=2e-2, n_tooth=55, hub_bore_radius=17.5e-3)
    gear2 = GearElementTVMS(n=21, m=12, module=2e-3, width=2e-2, n_tooth=75, hub_bore_radius=17.5e-3)

    gear1Speed = 80 * 2 * np.pi

    meshing = Mesh(gear1, gear2)   
    
    dt = 2 * np.pi / (20 * gear1Speed * gear1.n_tooth)

    nTm = 3
    time_range = np.arange(0, 2 * np.pi / (gear1Speed * gear1.n_tooth), dt)

    speed_range = gear1Speed * np.ones(np.shape(time_range))

    angle_range = time_range * gear1Speed

    stiffness = np.zeros(np.shape(time_range))
    k0_stiffness = np.zeros(np.shape(time_range))
    k1_stiffness = np.zeros(np.shape(time_range))

    for i, time in enumerate(time_range):
        stiffness[i], k0_stiffness[i], k1_stiffness[i] = meshing.mesh(time, speed_range[i])

    # Calculate limits and yticks
    x_lim = time_range[-1]
    # yticks = np.arange(3.8e8, int(4.4e8), int(0.1e8))

    # Create figure
    fig = go.Figure()

    # Add the main plot lines
    fig.add_trace(go.Scatter(
        x=time_range,
        y=stiffness,
        mode='markers',
        line=dict(color='blue', width=1),
        name='Stiffness'
    ))

    fig.add_trace(go.Scatter(
        x=time_range,
        y=k1_stiffness,
        mode='lines',
        line=dict(color='red', dash='solid'),
        name='K1 Stiffness'
    ))

    fig.add_trace(go.Scatter(
        x=time_range,
        y=k0_stiffness,
        mode='lines',
        line=dict(color='black', dash='dot'),
        name='K0 Stiffness'
    ))

    # Update layout
    fig.update_layout(
        title='Stiffness x Angular Displacement',
        xaxis=dict(
            title='Time [s]',
            range=[0, x_lim],
        ),
        yaxis=dict(
            title='Stiffness [N/m]',
            autorange=True,
            tickformat=".1e",  # Use scientific notation for y-axis labels
        ),
        showlegend=True
    )

    fig.add_annotation(
        x=1,  # x-coordinate of the annotation (relative to the plot area, 1 = right edge)
        y=1,  # y-coordinate of the annotation (relative to the plot area, 1 = top edge)
        xref="paper",  # Use relative coordinates for x (0 to 1)
        yref="paper",  # Use relative coordinates for y (0 to 1)
        text=f"gear_1: {gear1.n_tooth} tooth,    gear_2: {gear2.n_tooth} tooth,    gear_1_speed={gear1Speed/np.pi/2:.2f} Hz,    dt = {dt:.3e}",  # The text to display
        showarrow=False,  # Do not show an arrow pointing to the annotation
        align="right",  # Align text to the right
        xanchor="right",  # Anchor point for x (right-aligned)
        yanchor="top",  # Anchor point for y (top-aligned)
        font=dict(size=12, color="black")  # Customize font size and color
    )
    fig.show()

    # gear1.plot_tooth_geometry()

def gearStiffnessExample():
    gear1 = GearElementTVMS(    n=21, m=12, module=2e-3, width=2e-2, n_tooth=55, hub_bore_radius=17.5e-3)
    computeStiffness = np.vectorize(gear1.stiffness.compute_stiffness)

    angle_range = np.linspace(gear1.geometryDict['alpha_c'], gear1.geometryDict['alpha_a'], 200)
    ka, kb, kf, ks = computeStiffness(angle_range)
    
    # Create the figure
    fig = go.Figure()

    stiffness_dict = {"ka": ka, "kb": kb, "kf": kf, "ks": ks}

    # Add traces for each stiffness component
    for name, values in stiffness_dict.items():
        fig.add_trace(go.Scatter(x=angle_range*180/np.pi, y=values, mode='lines', name=name))

    # Customize layout
    fig.update_layout(
        title="Stiffness Variation vs. Angle",
        xaxis_title="Angle (radians or degrees)",  # Adjust accordingly
        yaxis_title="Stiffness",
        template="plotly_dark",  # Optional: Choose from 'plotly', 'plotly_dark', etc.
        legend_title="Stiffness Components",
        yaxis_tickformat='.2e',
    )

    # Show plot
    fig.show()

if __name__ == '__main__':
    gearGeometryExample()
