"""Gear Element module.

This module defines the GearElement classes which can be used to represent
gears or gearboxes used to couple different shafts in the MultiRotor class.
"""

import math
import numpy as np
import scipy as sp
from plotly import graph_objects as go
from warnings import warn

from ross.units import Q_, check_units
from ross.materials import steel
from ross.disk_element import DiskElement
from ross.materials import steel


__all__ = ["GearElement", "GearElementTVMS", "Mesh"]


def mod(val, max_val):
    mod = np.mod(val, max_val)
    return np.where((np.isclose(mod, 0)) & (val != 0), max_val, mod)


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
    n_teeth : int
        Number of teeth.
    base_diameter : float, pint.Quantity
        Base diameter (m).
        If given `pitch_diameter` is not necessary.
    pitch_diameter : float, pint.Quantity
        Pitch diameter (m).
        If given `base_diameter` is not necessary.
    pr_angle : float, pint.Quantity, optional
        The normal pressure angle (rad).
        Default is 20 deg (converted to rad).
    helix_angle: float, pint.Quantity, optional
        Helix angle for helical gears (rad).
        Default is 0, representing spur gear.
    bore_diameter : float, pint.Quantity, optional
        Inner diameter, the radius of the shaft on which the gear is mounted (m).
        If provided, it could be used to calculate gear mesh stiffness.
        Default is None.
    material : ross.Material, optional
        Gear's construction material. If provided, it could be used to calculate
        gear mesh stiffness.
        Default is steel.
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
    ...        pitch_diameter=0.187, n_teeth=26,
    ...        pr_angle=Q_(22.5, "deg"), helix_angle=0,
    ... )
    >>> gear.base_radius # doctest: +ELLIPSIS
    0.086382...
    """

    @check_units
    def __init__(
        self,
        n,
        m,
        Id,
        Ip,
        n_teeth,
        pitch_diameter=None,
        base_diameter=None,
        pr_angle=None,
        helix_angle=0,
        bore_diameter=None,
        material=steel,
        tag=None,
        scale_factor=1.0,
        color="Goldenrod",
    ):
        self.n_teeth = n_teeth

        if pr_angle is None:
            pr_angle = Q_(20, "deg").to_base_units().m

        self.pr_angle = pr_angle
        self.helix_angle = helix_angle

        if base_diameter:
            self.base_radius = base_diameter / 2
            self.pitch_diameter = 2 * self.base_radius / np.cos(self.pr_angle)
        elif pitch_diameter:
            self.pitch_diameter = pitch_diameter
            self.base_radius = self.pitch_diameter * np.cos(self.pr_angle) / 2
        else:
            raise TypeError(
                "At least one of the following must be informed for GearElement: base_diameter or pitch_diameter"
            )

        self.module = self.pitch_diameter / self.n_teeth

        self.material = material
        self.bore_diameter = bore_diameter
        self.width = None

        if self.bore_diameter:
            self.width = self.calculate_width(
                material.rho, m, self.bore_diameter, self.pitch_diameter
            )

        super().__init__(n, m, Id, Ip, tag, scale_factor, color)

    @classmethod
    def from_geometry(
        cls,
        n,
        material,
        width,
        i_d,
        o_d,
        n_teeth,
        pr_angle=None,
        helix_angle=0,
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
        material : ross.Material
            Gear's construction material.
        width : float, pint.Quantity
            The face width of the gear considering that the gear body has the
            same thickness (m).
        i_d : float, pint.Quantity
            Bore, inner diameter, the diameter of the shaft on which the gear
            is mounted (m).
        o_d : float, pint.Quantity
            Pitch diameter (m).
        n_teeth : int
            Number of teeth.
        pr_angle : float, pint.Quantity, optional
            The normal pressure angle (rad).
            Default is 20 deg (converted to rad).
        helix_angle: float, pint.Quantity, optional
            Helix angle for helical gears (rad).
            Default is 0, representing spur gear.
        tag : str, optional
            A tag to name the element.
            Default is None.
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
            Polar moment of inertia.

        Examples
        --------
        >>> from ross.materials import steel
        >>> gear = GearElement.from_geometry(0, steel, 0.07, 0.05, 0.28, 50)
        >>> gear.base_radius # doctest: +ELLIPSIS
        0.131556...
        >>>
        """
        m = cls.calculate_mass(material.rho, width, i_d, o_d)
        Ip = cls.calculate_Ip(m, i_d, o_d)
        Id = cls.calculate_Id(Ip, m, width)

        return cls(
            n,
            m,
            Id,
            Ip,
            n_teeth=n_teeth,
            pitch_diameter=o_d,
            pr_angle=pr_angle,
            helix_angle=helix_angle,
            bore_diameter=i_d,
            material=material,
            tag=tag,
            scale_factor=scale_factor,
            color=color,
        )

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
        scale_factor *= 2
        radius = min(self.base_radius * 1.1 + 0.05, 1)

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


class GearElementTVMS(GearElement):
    """A gear element with Time-Varying Mesh Stiffness (TVMS) modeling.

    This class defines a gear element used in rotor dynamics simulations, incorporating
    time-varying mesh stiffness to represent gear interactions more accurately. It uses
    geometric, material, and mass properties to model the gear.

    Parameters
    ----------
    n : int
        Node in which the gear will be inserted.
    material : ross.Material
        Gear's construction material.
    width : float, pint.Quantity
        Tooth width (m).
    bore_diameter : float, pint.Quantity
        Inner diameter, diameter of the shaft on which the gear is mounted (m).
    module : float, pint.Quantity
        Gear module (m).
    n_teeth : int
        Number of teeth.
    pr_angle : float, pint.Quantity, optional
        The normal pressure angle (rad).
        Default is 20 deg (converted to rad).
    helix_angle: float, pint.Quantity, optional
        Helix angle for helical gears (rad).
        Default is 0, representing spur gear.
    addendum_coeff : float, optional
        Addendum coefficient.
        Default is 1.
    tip_clearance_coeff : float, optional
        Gear's clearance coefficient.
        Default is 0.25.
    tag : str, optional
        A tag to name the element.
        Default is None.
    scale_factor : float or str, optional
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
    >>> from ross.materials import steel
    >>> gear = GearElementTVMS(
    ...    n=0,
    ...    material=steel,
    ...    width=0.02,
    ...    bore_diameter=0.035 * 2,
    ...    module=0.002,
    ...    n_teeth=62,
    ...    pr_angle=0.349066
    ... )
    >>> gear.base_radius # doctest : +ELLIPSIS
    0.058260...
    """

    @check_units
    def __init__(
        self,
        n,
        material,
        width,
        bore_diameter,
        module,
        n_teeth,
        pr_angle=None,
        helix_angle=0,
        addendum_coeff=1,
        tip_clearance_coeff=0.25,
        tag=None,
        scale_factor=1.0,
        color="Goldenrod",
    ):
        pitch_diameter = float(module) * n_teeth

        m = self.calculate_mass(material.rho, width, bore_diameter, pitch_diameter)
        Ip = self.calculate_Ip(m, bore_diameter, pitch_diameter)
        Id = self.calculate_Id(Ip, m, width)

        super().__init__(
            n,
            m,
            Id,
            Ip,
            n_teeth=n_teeth,
            pitch_diameter=pitch_diameter,
            pr_angle=pr_angle,
            helix_angle=helix_angle,
            tag=tag,
            scale_factor=scale_factor,
            color=color,
        )

        self.material = material
        self.width = width
        self.bore_diameter = bore_diameter

        self.addendum_coeff = float(addendum_coeff)
        self.tip_clearance_coeff = float(tip_clearance_coeff)

        # Initialize geometry related dictionaries
        a_coeff_mod = self.addendum_coeff * self.module
        c_coeff_mod = self.tip_clearance_coeff * self.module

        p_ang = self.pr_angle

        # Add radii constants
        r_b = self.base_radius
        r_p = r_b / np.cos(p_ang)
        r_a = r_p + a_coeff_mod
        r_c = np.sqrt((r_b * np.tan(p_ang) - a_coeff_mod / np.sin(p_ang)) ** 2 + r_b**2)
        r_f = r_p - (a_coeff_mod + c_coeff_mod)
        r_rho = c_coeff_mod / (1 - np.sin(p_ang))
        r_rho_ = r_rho / self.module

        # Add angular constants of the involute profile
        alpha_a = np.arccos(r_b / r_a)
        alpha_c = np.arccos(r_b / r_c)

        # The angle between the tooth center-line and de junction with the root circle [rad]
        theta_f = (
            np.pi / 2
            + 2 * np.tan(p_ang) * (self.addendum_coeff - r_rho_)
            + 2 * r_rho_ / np.cos(p_ang)
        ) / self.n_teeth

        theta_b = np.pi / (2 * self.n_teeth) + self._involute(p_ang)

        # Add geometric constants of the tooth profile
        a1 = (a_coeff_mod + c_coeff_mod) - r_rho

        b1 = (
            np.pi * self.module / 4
            + a_coeff_mod * np.tan(p_ang)
            + r_rho * np.cos(p_ang)
        )

        self.radii_dict = {
            "base": r_b,
            "pitch": r_p,
            "addendum": r_a,
            "root": r_f,
            "cutter_tip": r_rho,
        }

        self.tooth_dict = {
            "root_angle": theta_f,
            "base_angle": theta_b,
            "a": a1,
            "b": b1,
        }

        self.pr_angles_dict = {
            "pitch": p_ang,
            "addendum": alpha_a,
            "start_point": alpha_c,
        }

        self.tau_c = self._to_tau(alpha_c)

    @staticmethod
    def _involute(angle):
        """Involute function

        Calculates the involute function for a given angle. This function is
        used to describe the contact region of the gear profile.
        """
        return np.tan(angle) - float(angle)

    def _to_tau(self, pr_angle):
        """Transforms the pressure angle, used to build the involute profile,
        into the integration variable tau.

        Parameters
        ----------
        pr_angle : float
            The pressure angle (rad) at which to perform the transformation.

        Returns
        -------
        tau : float
            Corresponding tau angle (rad).
        """
        tau = pr_angle + self._involute(pr_angle) - self.tooth_dict["base_angle"]
        return tau

    def _diff_tau(self, tau):
        """Method for evaluating the stiffness commonly found in the
        integrative functions of the involute region on a specified angle.

        Parameters
        ----------
        tau : float
            Operational tau angle.
        """
        return self.base_radius * (tau + self.tooth_dict["base_angle"]) * np.cos(tau)

    def _diff_gamma(self, gamma):
        """Method used in evaluating the stiffness commonly found in the
        integrative functions of the transition region.

        Parameters
        ----------
        gamma : float
            Value of the gamma angle used to describe the profile.
        """
        a1 = self.tooth_dict["a"]
        b1 = self.tooth_dict["b"]
        r_p = self.radii_dict["pitch"]
        r_rho = self.radii_dict["cutter_tip"]

        term_1 = (
            a1 * np.sin((a1 / np.tan(gamma) + b1) / r_p) * (1 + np.tan(gamma) ** 2)
        ) / np.tan(gamma) ** 2

        term_2 = (
            a1
            * np.cos(gamma)
            / np.sin(gamma) ** 2
            * np.sin(gamma - (a1 / np.tan(gamma) + b1) / r_p)
        )

        term_3 = (
            -(a1 / np.sin(gamma) + r_rho)
            * np.cos(gamma - (a1 / np.tan(gamma) + b1) / r_p)
            * (1 + a1 * (1 + np.tan(gamma) ** 2) / (r_p * np.tan(gamma) ** 2))
        )

        return term_1 + term_2 + term_3

    def _compute_involute_curve(self, tau):
        """Compute the y, x, area and I_y of the involute curve given a tau
        angle.

        Parameters
        ----------
        tau : float
            The angle indicating the position on the involute curve profile.

        Returns
        -------
        y : float
            The y-coordinate of the transition curve at tau_i.
        x : float
            The x-coordinate of the transition curve at tau_i.
        area : float
            The area under the transition curve up to x.
        I_y : float
            The second moment of area (moment of inertia) about the y-axis at x.
        """
        theta_b = self.tooth_dict["base_angle"]
        r_b = self.base_radius

        x = r_b * ((tau + theta_b) * np.cos(tau) - np.sin(tau))
        y = r_b * ((tau + theta_b) * np.sin(tau) + np.cos(tau))

        area = 2 * x * self.width
        I_y = 2 / 3 * x**3 * self.width

        return y, x, area, I_y

    def _compute_transition_curve(self, gamma):
        """Compute the y, x, area and I_y of the transition curve given a gamma
        angle.

        Parameters
        ----------
        gamma : float
            The angle indicating the position on the transition curve profile.

        Returns
        -------
        y : float
            The y-coordinate of the transition curve at gamma.
        x : float
            The x-coordinate of the transition curve at gamma.
        area : float
            The cumulative area under the transition curve up to x.
        I_y : float
            The second moment of area (moment of inertia) about the y-axis at x.
        """
        a1 = self.tooth_dict["a"]
        b1 = self.tooth_dict["b"]
        r_p = self.radii_dict["pitch"]
        r_rho = self.radii_dict["cutter_tip"]

        phi = (a1 / np.tan(gamma) + b1) / r_p

        x = r_p * np.sin(phi) - (a1 / np.sin(gamma) + r_rho) * np.cos(gamma - phi)
        y = r_p * np.cos(phi) - (a1 / np.sin(gamma) + r_rho) * np.sin(gamma - phi)

        area = 2 * x * self.width
        I_y = 2 / 3 * x**3 * self.width

        return y, x, area, I_y

    @check_units
    def _compute_stiffness(self, angle):
        """Computes the stiffness in the direction of the applied force on the
        gear (line of action), according to the involute profile. Ma, H. et. al. (2014).

        Parameters
        ----------
        angle : float
            The angle formed by the normal of the contact involute curves and the x axis.

        Returns
        -------
        k : float
            The sum of the computed stiffness components.
        """
        beta = self._to_tau(angle)

        inv_kf = self._inv_kf(beta)
        inv_ka = self._inv_ka(beta)
        inv_kb = self._inv_kb(beta)
        inv_ks = self._inv_ks(beta)

        k = 1 / (inv_kf + inv_ka + inv_kb + inv_ks)

        return k

    def _inv_kf(self, beta):
        """Calculate the stiffness contribution from the gear base, given a
        point on the involute curve.

        Sainsot, P., Velex, P., & Duverger, O. (2004). Contribution of gear
        body to tooth deflections - A new bidimensional analytical
        formula. Journal of Mechanical Design, 126(4), 748–752.
        https://doi.org/10.1115/1.1758252

        Parameters
        ----------
        beta : float
            The operating pressure angle in radians. This angle is used
            to determine the stiffness characteristics based on the gear
            geometry and material properties. It's the current angle of the
            contact point between the meshing gears.

        Returns
        -------
        inv_kf : float
            The inverse of kf for the gear base.
        """

        r_f = self.radii_dict["root"]
        theta_f = self.tooth_dict["root_angle"]
        h = r_f / (0.5 * self.bore_diameter)

        # curve-fitted by polynomial functions
        poly_func = lambda A, B, C, D, E, F: (
            A / (theta_f**2) + B * h**2 + C * h / theta_f + D / theta_f + E * h + F
        )

        poly_coeffs = (
            (-5.574e-5, -1.9986e-3, -2.3015e-4, 4.7702e-3, 0.0271, 6.8045),
            (60.111e-5, 28.100e-3, -83.431e-4, -9.9256e-3, 0.1624, 0.9086),
            (-50.952e-5, 185.50e-3, 0.0538e-4, 53.300e-3, 0.2895, 0.9236),
            (-6.2042e-5, 9.0889e-3, -4.0964e-4, 7.8297e-3, -0.1472, 0.6904),
        )

        poly_limits = (
            (6.82, 6.94),
            (1.08, 3.29),
            (2.56, 13.47),
            (0.141, 0.62),
        )

        L = poly_func(*poly_coeffs[0])
        M = poly_func(*poly_coeffs[1])
        P = poly_func(*poly_coeffs[2])
        Q = poly_func(*poly_coeffs[3])

        for i, value in enumerate((L, M, P, Q)):
            if value < min(poly_limits[i]) or value > max(poly_limits[i]):
                warn(
                    "Extrapolating gear body coefficients described by Sainsot et. al. (2014). Be careful when post-processing the results."
                )
                break

        y, _, _, _ = self._compute_involute_curve(beta)

        Sf = 2 * r_f * theta_f
        u = y - r_f

        inv_kf = (np.cos(beta) ** 2 / (self.material.E * self.width)) * (
            L * (u / Sf) ** 2 + M * u / Sf + P * (1 + Q * np.tan(beta) ** 2)
        )

        return inv_kf

    def _inv_ks(self, beta):
        """Calculate the stiffness contribution from the gear resistance from
        shear stresses, given the operating pressure angle.

        Parameters
        ----------
        beta : float
            Operating pressure angle.

        Returns
        -------
        inv_ks : float
            The inverse of shear stiffness, 1/ks.
        """
        func_ks = lambda angle, compute_curve, diff: (
            1.2
            * np.cos(beta) ** 2
            / (self.material.G_s * compute_curve(angle)[2])
            * diff(angle)
        )

        inv_ks_t = self._integrate_transiction_term(func_ks)
        inv_ks_i = self._integrate_invol_term(func_ks, beta)

        inv_ks = inv_ks_t + inv_ks_i

        return inv_ks

    def _inv_kb(self, beta):
        """Calculate the stiffness contribution from the gear resistance from
        bending stresses, given the operating pressure angle.

        Parameters
        ----------
        beta : float
            Operating pressure angle.

        Returns
        -------
        inv_kb : float
            The inverse of bending stiffness, 1/kb.
        """
        y_op, x_op, _, _ = self._compute_involute_curve(beta)

        func_kb = lambda angle, compute_curve, diff: (
            (np.cos(beta) * (y_op - compute_curve(angle)[0]) - x_op * np.sin(beta)) ** 2
            / (self.material.E * compute_curve(angle)[3])
            * diff(angle)
        )

        inv_kb_t = self._integrate_transiction_term(func_kb)
        inv_kb_i = self._integrate_invol_term(func_kb, beta)

        inv_kb = inv_kb_t + inv_kb_i

        return inv_kb

    def _inv_ka(self, beta):
        """Calculate the stiffness contribution from the gear resistance from
        axial stresses, given the operating pressure angle.

        Parameters
        ----------
        beta : float
            Operating pressure angle.

        Returns
        -------
        inv_ka : float
            The inverse of axial stiffness, 1/ka.
        """
        func_ka = lambda angle, compute_curve, diff: (
            np.sin(beta) ** 2
            / (self.material.E * compute_curve(angle)[2])
            * diff(angle)
        )

        inv_ka_t = self._integrate_transiction_term(func_ka)
        inv_ka_i = self._integrate_invol_term(func_ka, beta)

        inv_ka = inv_ka_t + inv_ka_i

        return inv_ka

    def _integrate_transiction_term(self, func):
        inv_k_t, _ = sp.integrate.quad(
            lambda gamma: func(gamma, self._compute_transition_curve, self._diff_gamma),
            np.pi / 2,
            self.pr_angle,
        )
        return inv_k_t

    def _integrate_invol_term(self, func, beta):
        tau_c = self.tau_c
        inv_k_i, error = sp.integrate.quad(
            lambda tau: func(tau, self._compute_involute_curve, self._diff_tau),
            tau_c,
            beta,
        )
        return inv_k_i

    def plot_tooth_geometry(self):
        """Plot the geometry of the tooth profile."""
        # Generate the transition geometry
        transition = np.linspace(np.pi / 2, self.pr_angle, 200)
        transition_vectorized = np.vectorize(self._compute_transition_curve)
        y_t, x_t, _, _ = transition_vectorized(transition)

        # Generate the involute geometry
        involute = np.linspace(
            self.pr_angles_dict["start_point"], self.pr_angles_dict["addendum"], 200
        )
        tau_vectorize = np.vectorize(self._to_tau)
        tau_values = tau_vectorize(involute)

        involute_vectorize = np.vectorize(self._compute_involute_curve)
        y_i, x_i, _, _ = involute_vectorize(tau_values)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=y_t[-1::-1],
                y=x_t[-1::-1],
                mode="lines",
                line=dict(dash="dash"),
                name="Transition Curve",
            )
        )

        fig.add_trace(go.Scatter(x=y_i, y=x_i, mode="lines", name="Involute Curve"))

        fig.update_layout(
            title="Tooth Profile Geometry",
            xaxis_title="Y-axis (m)",
            yaxis_title="X-axis (m)",
        )

        return fig


class Mesh:
    """Represents the meshing behavior between two gears in contact
    including stiffness and contact ratio calculations.

    Parameters:
    -----------
    driving_gear : GearElement
        The driving gear object used in the gear pair.
    driven_gear : GearElement
        The driven gear object used in the gear pair.
    gear_mesh_stiffness : float, optional
        Directly specify the stiffness of the gear mesh.
        If not provided, it can be calculated automatically
        when using `GearElementTVMS` instead of `GearElement`.

    Attributes:
    -----------
    driving_gear : GearElement
        The driving_gear object, which contains information about the
        geometry and properties of the driving gear.
    driven_gear : GearElement
        The driven gear object, which contains information about the
        geometry and properties of the wheel gear.
    gear_ratio : float
        The transamission ratio, defined as the ratio of the radii between the
        driving and driven gears.
    pressure_angle : float
        The pressure angle of the gear mesh (rad).

    Examples
    --------
    >>> from ross.materials import steel
    >>> driving = GearElementTVMS(
    ...    n=0,
    ...    material=steel,
    ...    width=0.02,
    ...    bore_diameter=0.0175 * 2,
    ...    module=0.002,
    ...    n_teeth=62,
    ...    pr_angle=0.349066
    ... )
    >>> driven = GearElementTVMS(
    ...    n=2,
    ...    material=steel,
    ...    width=0.02,
    ...    bore_diameter=0.0175 * 2,
    ...    module=0.002,
    ...    n_teeth=62,
    ...    pr_angle=0.349066
    ... )
    >>> mesh = Mesh(driving, driven)
    >>> mesh.stiffness # doctest : +ELLIPSIS
    429787128.434...
    """

    def __init__(
        self,
        driving_gear,
        driven_gear,
        gear_mesh_stiffness=None,
    ):
        self.driving_gear = driving_gear
        self.driven_gear = driven_gear
        self.gear_ratio = (
            driving_gear.n_teeth / driven_gear.n_teeth
        )  # Shigley Machine Elements
        self.pressure_angle = driving_gear.pr_angle
        self.helix_angle = driving_gear.helix_angle

        if not math.isclose(driving_gear.module, driven_gear.module, rel_tol=0.05):
            warn(
                "Gear modules must match for proper meshing | "
                f"Driving gear: {driving_gear.module:.4f}, Driven gear: {driven_gear.module:.4f}"
            )

        if gear_mesh_stiffness is None:
            if (
                type(driving_gear) == GearElementTVMS
                and type(driven_gear) == GearElementTVMS
            ):
                ra1 = driving_gear.radii_dict["addendum"]
                ra2 = driven_gear.radii_dict["addendum"]
                self.contact_ratio = self._calculate_contact_ratio(ra1, ra2)

                self.hertzian_stiffness = (
                    np.pi
                    * driving_gear.material.E
                    * driving_gear.width
                    / (4 * (1 - driving_gear.material.Poisson**2))
                )

                theta_range, stiffness_range = self.get_stiffness_for_mesh_period()

                self.theta_range = theta_range
                self.stiffness_range = stiffness_range
                self.stiffness = max(stiffness_range)

            else:
                if driving_gear.width:
                    ra1 = driving_gear.pitch_diameter * (0.5 + 1 / driving_gear.n_teeth)
                    ra2 = driven_gear.pitch_diameter * (0.5 + 1 / driven_gear.n_teeth)
                    cr = self._calculate_contact_ratio(ra1, ra2)

                    E1 = driving_gear.material.E
                    E2 = driven_gear.material.E

                    self.contact_ratio = cr
                    self.stiffness = (cr * driving_gear.width * E1 * E2) / (
                        9 * (E1 + E2)
                    )

                else:
                    raise TypeError(
                        "Missing 'gear_mesh_stiffness'. You have two options if you don't set this value:\n"
                        "1) Provide 'material' and 'bore_diameter' for 'GearElement'\n"
                        "2) Use 'GearElementTVMS' instead"
                    )

        else:
            self.stiffness = gear_mesh_stiffness

    def _calculate_contact_ratio(self, driving_addendum_radius, driven_addendum_radius):
        rb1 = self.driving_gear.base_radius
        rb2 = self.driven_gear.base_radius

        center_distance = (
            self.driving_gear.pitch_diameter + self.driven_gear.pitch_diameter
        ) / 2

        contact_length = (
            np.sqrt(driving_addendum_radius**2 - rb1**2)
            + np.sqrt(driven_addendum_radius**2 - rb2**2)
            - center_distance * np.sin(self.pressure_angle)
        )

        base_pitch = 2 * np.pi * rb1 / self.driving_gear.n_teeth

        contact_ratio = contact_length / base_pitch

        return contact_ratio

    def _angular_equivalent_stiffness(self, d_alpha):
        """
        Parameters
        ---------
        d_alpha : float
            The angular displacement of the driving gear in radians.

        Returns
        -------
        k : float
            The angular equivalent stiffness of mesh contact.
        """
        # Angular displacements
        alpha_1 = self.driving_gear.pr_angles_dict["start_point"] + d_alpha
        alpha_2 = (
            self.driven_gear.pr_angles_dict["addendum"] - self.gear_ratio * d_alpha
        )

        # Contact stiffness
        k1 = self.driving_gear._compute_stiffness(alpha_1)
        k2 = self.driven_gear._compute_stiffness(alpha_2)

        # Evaluating the equivalent stiffness
        kh = self.hertzian_stiffness
        k = 1 / (1 / kh + 1 / k1 + 1 / k2)

        return k

    def get_variable_stiffness(self, angular_position):
        """Calculate the variable stiffness of a gear pair.

        This method computes the equivalent stiffness of a gear mesh at a given
        angular position, taking into account the periodic nature of the meshing
        process and the contact ratio of the gear pair.

        Parameters
        ----------
        angular_position : float
            Gear angular position for which the meshing stiffness is calculated.

        Returns
        -------
        stiffness : float
            The total equivalent meshing stiffness at the given angular position.
        """
        cr = self.contact_ratio
        alpha_c = self.driving_gear.pr_angles_dict["start_point"]
        alpha_a = self.driving_gear.pr_angles_dict["addendum"]

        tm_om = 2 * np.pi / self.driven_gear.n_teeth
        theta = mod(angular_position, tm_om)

        d_meshing = (alpha_a - alpha_c) / cr
        d_alpha = d_meshing / tm_om * theta

        stiffness = self._angular_equivalent_stiffness(d_alpha)

        if d_alpha <= d_meshing * (cr - 1):
            stiffness += self._angular_equivalent_stiffness(d_alpha + d_meshing)

        return stiffness

    def get_stiffness_for_mesh_period(self, n_mesh_period=1, n_points=1000):
        """Computes the mesh stiffness profile over a specified number of gear
        mesh periods.

        Parameters
        ----------
        n_mesh_period : int, optional
            Number of mesh periods to evaluate. Default is 1.
        n_points : int, optional
            Number of angular sample points to compute within the total range.
            Default is 1000.

        Returns
        -------
        theta_range : np.ndarray
            Array of angular positions (rad) spanning the specified mesh
            periods.
        stiffness_range : list of float
            List of stiffness values corresponding to each angular position.
        """
        theta_end = 2 * np.pi / self.driving_gear.n_teeth * n_mesh_period
        theta_range = np.linspace(0, theta_end, n_points)

        stiffness_range = [self.get_variable_stiffness(theta) for theta in theta_range]

        return theta_range, stiffness_range

    def interpolate_stiffness(self, angular_position):
        """Interpolates the mesh stiffness value at a given angular position.

        Parameters
        ----------
        angular_position : float or array-like
            Angular position(s) at which to evaluate the stiffness. Should be in
            the radians.

        Returns
        -------
        stiffness : float or np.ndarray
            Interpolated stiffness value(s) in N/m.
        """
        theta = mod(angular_position, max(self.theta_range))
        stiffness = np.interp(theta, self.theta_range, self.stiffness_range)

        return stiffness

    def plot_stiffness_profile(
        self,
        n_mesh_period=1,
        n_points=1000,
        angle_units="rad",
        stiffness_units="N/m",
        **kwargs,
    ):
        """Plots the gear mesh stiffness profile over one or more meshing periods.

        Parameters
        ----------
        n_mesh_period : int, optional
            Number of mesh periods to plot. Default is 1.
        n_points : int, optional
            Number of data points to evaluate for the stiffness profile. Default is 1000.
        angle_units : str, optional
            Units for the angular position axis. Default is 'rad'.
        stiffness_units : str, optional
            Units for the stiffness axis. Default is 'N/m'.
        **kwargs : dict, optional
            Additional keyword arguments passed to `plotly.graph_objects.Figure.update_layout`
            for customizing the figure (e.g., title, font, size, legend settings, etc.).
        """
        fig = go.Figure()

        if n_mesh_period != 1 or n_points != 1000:
            theta_range, stiffness_range = self.get_stiffness_for_mesh_period(
                n_mesh_period, n_points
            )
        else:
            theta_range = self.theta_range
            stiffness_range = self.stiffness_range

        fig.add_trace(
            go.Scatter(
                x=Q_(theta_range, "rad").to(angle_units).m,
                y=Q_(stiffness_range, "N/m").to(stiffness_units).m,
                mode="lines",
                line=dict(color="black", width=3),
            )
        )

        fig.update_layout(
            xaxis=dict(
                title=f"Angular position ({angle_units})",
            ),
            yaxis=dict(
                title=f"Stiffness ({stiffness_units})",
                tickformat=".1e",
            ),
            **kwargs,
        )

        return fig
