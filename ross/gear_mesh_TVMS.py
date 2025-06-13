"""Gear Element module.

This module defines the GearElementTVMS classes which can be used to represent
gears or gearboxes used to couple different shafts in the MultiRotor class.
"""

import numpy as np
import scipy as sp
import pandas as pd
from plotly import graph_objects as go

from ross.units import check_units
from ross.materials import steel
from ross.gear_element import GearElement

__all__ = ["GearElementTVMS", "Mesh"]


class GearElementTVMS(GearElement):
    """A gear element.

    This class creates a gear element from input data of inertia and mass.

    Parameters
    ----------
    n : int
        Node in which the gear will be inserted.
    m : float, pint.Quantity
        Mass of the gear element (kg).
    module : float
        Gear module (m).
    n_tooth : int
        Tooth number of the gear.
    width : float
        Width of the gear (m).
    hub_bore_radius : float
        The hub bore radius (m).
    material : ross.Material
        Gear's construction material.
    pr_angle : float, optional
        The pressure angle of the gear (rad).
        Default is 20 deg (converted to rad).
    addendum_coeff : float, optional
        Addendum coefficient of the gear.
        Default is 1.
    clearance_coeff : float, optional
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
    >>> gear = GearElementTVMS(
    ...        n=0, m=4.67, Id=0.015, Ip=0.030,
    ...        pitch_diameter=0.187,
    ...        pr_angle=Q_(22.5, "deg")
    ... )
    >>> gear.pr_angle # doctest : +ELLIPSIS
    0.392699...
    """

    @check_units
    def __init__(
        self,
        n,
        m,
        module,
        n_tooth,
        width,
        hub_bore_radius,
        material=steel,
        pr_angle=None,
        addendum_coeff=1,
        clearance_coeff=0.25,
        tag=None,
        scale_factor=1.0,
        color="Goldenrod",
    ):
        pitch_diameter = module * n_tooth

        Ip = np.pi / 2 * ((pitch_diameter / 2) ** 4 - (hub_bore_radius / 2) ** 4)

        Id = np.pi / 4 * ((pitch_diameter / 2) ** 4 - (hub_bore_radius / 2) ** 4)

        super().__init__(
            n,
            m,
            Id,
            Ip,
            pitch_diameter=pitch_diameter,
            pr_angle=pr_angle,
            tag=tag,
            scale_factor=scale_factor,
            color=color,
        )

        self.hub_bore_radius = float(hub_bore_radius)
        self.module = float(module)
        self.n_tooth = float(n_tooth)
        self.width = float(width)
        self.addendum_coeff = float(addendum_coeff)
        self.clearance_coeff = float(clearance_coeff)

        self.material = material

        # Initialize the geometry_dict
        self._initialize_geometry()

        self._ka_transiction = False
        self._kb_transiction = False
        self._ks_transiction = False

    def _initialize_geometry(self):
        """Initialize the geometry dictionary for gear tooth stiffness analysis.

        This method populates ``geometry_dict`` with various geometric
        parameters:

        NEED TO COMPLETE...


        Returns
        -------
        geometry_dict : dict
            A dictionary populated with computed geometric parameters.

        """
        self.geometry_dict = {}

        a_coeff_mod = self.addendum_coeff * self.module
        c_coeff_mod = self.clearance_coeff * self.module

        p_ang = self.pr_angle

        # Add radii constants
        # radius of base circle [m]
        r_b = 1 / 2 * self.module * self.n_tooth * np.cos(p_ang)
        # radius of pitch circle [m]
        r_p = r_b / np.cos(p_ang)
        # radius of addendum circle [m]
        r_a = r_p + a_coeff_mod
        # radii of the involute starting point [m]
        r_c = np.sqrt(
            np.square(r_b * np.tan(p_ang) - a_coeff_mod / np.sin(p_ang))
            + np.square(r_b)
        )
        # radius of root circle [m]
        r_f = 1 / 2 * self.module * self.n_tooth - (a_coeff_mod + c_coeff_mod)
        # radius of cutter tip round corner [m]
        r_rho = c_coeff_mod / (1 - np.sin(p_ang))
        r_rho_ = r_rho / self.module

        self.geometry_dict.update(
            {
                "r_b": r_b,
                "r_p": r_p,
                "r_a": r_a,
                "r_c": r_c,
                "r_f": r_f,
                "r_rho": r_rho,
                "r_rho_": r_rho_,
            }
        )

        # Add angular constants of the involute profile
        # pressure angle when the contact point is on the addendum circle [rad]
        alpha_a = np.arccos(r_b / r_a)
        # pressure angle when the contact point is on the C point [rad]
        alpha_c = np.arccos(r_b / r_c)

        # The angle between the tooth center-line and de junction with the root circle [rad]
        theta_f = (
            np.pi / 2
            + 2 * np.tan(p_ang) * (self.addendum_coeff - r_rho_)
            + 2 * r_rho_ / np.cos(p_ang)
        ) / self.n_tooth

        theta_b = np.pi / (2 * self.n_tooth) + self._involute(p_ang)

        self.geometry_dict.update(
            {
                "alpha_a": alpha_a,
                "alpha_c": alpha_c,
                "theta_f": theta_f,
                "theta_b": theta_b,
            }
        )

        # Add geometric constants of the tooth profile
        a1 = (a_coeff_mod + c_coeff_mod) - r_rho

        b1 = (
            np.pi * self.module / 4
            + a_coeff_mod * np.tan(p_ang)
            + r_rho * np.cos(p_ang)
        )

        tau_c = self._to_tau(alpha_c)
        tau_a = self._to_tau(alpha_a)

        self.geometry_dict.update(
            {"a_1": a1, "b_1": b1, "tau_c": tau_c, "tau_a": tau_a}
        )

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
        phi = (
            self.geometry_dict["a_1"] / np.tan(gamma) + self.geometry_dict["b_1"]
        ) / self.geometry_dict["r_p"]

        y = self.geometry_dict["r_p"] * np.cos(phi) - (
            self.geometry_dict["a_1"] / np.sin(gamma) + self.geometry_dict["r_rho"]
        ) * np.sin(gamma - phi)

        x = self.geometry_dict["r_p"] * np.sin(phi) - (
            self.geometry_dict["a_1"] / np.sin(gamma) + self.geometry_dict["r_rho"]
        ) * np.cos(gamma - phi)

        area = 2 * x * self.width
        I_y = 2 / 3 * x**3 * self.width

        return y, x, area, I_y

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
        y = self.geometry_dict["r_b"] * (
            (tau + self.geometry_dict["theta_b"]) * np.sin(tau) + np.cos(tau)
        )

        x = self.geometry_dict["r_b"] * (
            (tau + self.geometry_dict["theta_b"]) * np.cos(tau) - np.sin(tau)
        )

        area = 2 * x * self.width
        I_y = 2 / 3 * x**3 * self.width

        return y, x, area, I_y

    @staticmethod
    @check_units
    def _involute(angle):
        """Involute function

        Calculates the involute function for a given angle. This function is
        used to describe the contact region of the gear profile.

        Parameters
        ----------
        angle : float
            Input angle in radians.

        Returns
        -------
        float
            The inv(angle)

        Examples
        --------
        >>> GearGeometry._involute(20 / 180 * np.pi)
        0.014904383867336446
        """
        angle = float(angle)
        return np.tan(angle) - angle

    def _to_tau(self, alpha):
        """Transforms the alpha angle, used to build the involute profile, into
        the integration variable tau.

        Parameters
        ----------
        alpha : float
            An angle within the involute profile.

        Returns
        -------
        tau : float

        Examples
        ---------
        >>> self._to_tau(31 * np.pi / 180)
        0.5573963019457713

        References
        ----------
        Ma, H., Pang, X., Song, R., & Yang, J. (2014). Time-varying mesh
        stiffness calculation of spur gears based on improved energy method.
        Journal of Northeastern University (Natural Science), 35(6), 863–867.
        https://doi.org/10.3969/j.issn.1005-3026.2014.06.023
        """

        return alpha - self.geometry_dict["theta_b"] + self._involute(alpha)

    def plot_tooth_geometry(self):
        """Plot the geometry of the tooth profile."""
        # Generate the transition geometry
        transition = np.linspace(np.pi / 2, self.pr_angle, 200)
        transition_vectorized = np.vectorize(self._compute_transition_curve)
        y_t, x_t, _, _ = transition_vectorized(transition)

        # Generate the involute geometry
        involute = np.linspace(
            self.geometry_dict["alpha_c"], self.geometry_dict["alpha_a"], 200
        )
        tau_vectorize = np.vectorize(self._to_tau)
        tau_values = tau_vectorize(involute)

        involute_vectorize = np.vectorize(self._compute_involute_curve)
        y_i, x_i, _, _ = involute_vectorize(tau_values)

        # Create the plot
        fig = go.Figure()

        # Add the transition curve
        fig.add_trace(
            go.Scatter(
                x=y_t[-1::-1], y=x_t[-1::-1], mode="lines", name="Transition Curve"
            )
        )

        # Add the involute curve
        fig.add_trace(go.Scatter(x=y_i, y=x_i, mode="lines", name="Involute Curve"))

        # Update layout with gridlines
        fig.update_layout(
            title="Tooth Profile Geometry",
            xaxis_title="Y-axis [m]",
            yaxis_title="X-axis [m]",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )

        # Show the plot
        fig.show()
        pass

    def _diff_tau(self, tau):
        """Method for evaluating the stiffness commonly found in the
        integrative functions of the involute region on a specified angle.

        Parameters
        ----------
        tau : float
            Operational tau angle.

        Returns
        -------
        float
            The value of _diff_tau(tau)
        """
        return (
            self.geometry_dict["r_b"]
            * (tau + self.geometry_dict["theta_b"])
            * np.cos(tau)
        )

    def _diff_gamma(self, gamma):
        """Method used in evaluating the stiffness commonly found in the
        integrative functions of the transition region.

        Parameters
        ----------
        gamma : float
            Value of the gamma angle used to describre the profile.

        Returns
        -------
        float
            The value of _diff_gamma(gamma)
        """
        a1 = self.geometry_dict["a_1"]
        b1 = self.geometry_dict["b_1"]
        r_p = self.geometry_dict["r_p"]
        r_rho = self.geometry_dict["r_rho"]

        term_1 = (
            a1
            * np.sin((a1 / np.tan(gamma) + b1) / r_p)
            * (1 + np.square(np.tan(gamma)))
        ) / np.square(np.tan(gamma))

        term_2 = (
            a1
            * np.cos(gamma)
            / np.square(np.sin(gamma))
            * np.sin(gamma - (a1 / np.tan(gamma) + b1) / r_p)
        )

        term_3 = (
            -(a1 / np.sin(gamma) + r_rho)
            * np.cos(gamma - (a1 / np.tan(gamma) + b1) / r_p)
            * (
                1
                + a1 * (1 + np.square(np.tan(gamma))) / (r_p * np.square(np.tan(gamma)))
            )
        )

        return term_1 + term_2 + term_3

    @check_units
    def _compute_stiffness(self, alpha_op_angle):
        """Computes the stiffness in the direction of the applied force on the
        gear (line of action), according to the involute profile.

        It evaluates each of them separetly, and those values are returned as 1/stiffness for an approach in accordance with Ma, H. et. al. (2014).

        Parameters
        ----------
        tau_op : float
            The tau operational angle, e.g. the angle formed by the normal of
            the contact involute curves and the x axis.

        Returns
        -------
        A tuple containing the computed stiffness components as 1/stiffness
        values. The elements represent:
            ka : float
                Stiffness related to axial stresses.
            kb : float
                Stiffness related to bending stresses.
            kf : float
                Stiffness related to body of the gear.
            ks : float
                Stiffness related to shear stresses.
        """
        alpha_op = float(alpha_op_angle)
        tau_op = self._to_tau(alpha_op)

        _inv_kf = self._inv_kf(tau_op)
        _inv_ka = self._inv_ka(tau_op)
        _inv_kb = self._inv_kb(tau_op)
        _inv_ks = self._inv_ks(tau_op)

        if np.isnan(_inv_kf):
            return _inv_ka, _inv_kb, _inv_ks

        return 1 / _inv_ka, 1 / _inv_kb, 1 / _inv_kf, 1 / _inv_ks

    def _gear_body_polynominal(self):
        """This method uses the approach described by Sainsot et al. (2004) to
        calculate the stiffness factor (kf) contributing to tooth deflections.
        If the parameters fall outside the experimental range used to derive
        the analytical formula, the method returns `'oo'` to indicate an
        infinite stiffness approximation.

        Returns
        -------
        float
            The calculated stiffness factor (kf) for the gear base.

        str
            Return 'oo' if it doesn't match the criteria for the experimental
            range where this method was built.
        """

        h = self.geometry_dict["r_f"] / self.hub_bore_radius

        poly = pd.DataFrame()
        poly["var"] = ["L", "M", "P", "Q"]
        poly["A_i"] = [-5.574e-5, 60.111e-5, -50.952e-5, -6.2042e-5]
        poly["B_i"] = [-1.9986e-3, 28.100e-3, 185.50e-3, 9.0889e-3]
        poly["C_i"] = [-2.3015e-4, -83.431e-4, 0.0538e-4, -4.0964e-4]
        poly["D_i"] = [4.7702e-3, -9.9256e-3, 53.300e-3, 7.8297e-3]
        poly["E_i"] = [0.0271, 0.1624, 0.2895, -0.1472]
        poly["F_i"] = [6.8045, 0.9086, 0.9236, 0.6904]

        calculate_x_i = lambda row: (
            row["A_i"] / (self.geometry_dict["theta_f"] ** 2)
            + row["B_i"] * h**2
            + row["C_i"] * h / self.geometry_dict["theta_f"]
            + row["D_i"] / self.geometry_dict["theta_f"]
            + row["E_i"] * h
            + row["F_i"]
        )  # OK

        poly["X_i"] = poly.apply(lambda row: calculate_x_i(row), axis=1)

        limits = {
            "L": (6.82, 6.94),
            "M": (1.08, 3.29),
            "P": (2.56, 13.47),
            "Q": (0.141, 0.62),
        }

        for index, row in poly.iterrows():
            var_name = row["var"]
            X_i = row["X_i"]
            lower_limit, upper_limit = limits[var_name]

            # if (not lower_limit <= X_i <= upper_limit) :#or (not 1.4 <= h <= 7) or (not 0.01 <= gear.theta_f <= 0.12):
            #     # for the stiffness on the base of the tooth to match the model, it has to match those criteria above. If not, kf -> oo.
            #     global contador
            #     contador+=1
            #     return 'oo'

        return poly.loc[:, ["var", "X_i"]]

    def _inv_kf(self, tau_op):
        """Calculate the stiffness contribution from the gear base, given a
        point on the involute curve.

        Sainsot, P., Velex, P., & Duverger, O. (2004). Contribution of gear
        body to tooth deflections - A new bidimensional analytical
        formula. Journal of Mechanical Design, 126(4), 748–752.
        https://doi.org/10.1115/1.1758252

        Parameters
        ----------
        tau_op : float
            The operational pressure angle (tau) in radians. This angle is used
            to determine the stiffness characteristics based on the gear
            geometry and material properties. It's the current angle of the
            contact point between the meshing gears.

        Returns
        -------
        kf : float
            The calculated 1/kf for the gear base.
        """

        # obtain a dataframe of polynomials coefficients
        poly = self._gear_body_polynominal()

        # Extrapolating the range of interpolation described by Sainsot et. al. (2014).
        # if type(poly) == str:
        #     return 0

        L_poly, M_poly, P_poly, Q_poly = poly["X_i"]

        y, _, _, _ = self._compute_involute_curve(tau_op)

        Sf = 2 * self.geometry_dict["r_f"] * self.geometry_dict["theta_f"]
        u = y - self.geometry_dict["r_f"]

        kf = (np.cos(tau_op) ** 2 / (self.material.E * self.width)) * (
            L_poly * (u / Sf) ** 2
            + M_poly * u / Sf
            + P_poly * (1 + Q_poly * np.tan(tau_op) ** 2)
        )

        return kf

    def _inv_ks(self, tau_op):
        """Calculate the stiffness contribution from the gear resistance from
        shear stresses, given the tau operational angle.

        Parameters
        ----------
        tau_op : float
            Operational tau angle.

        Returns
        -------
        float
            The shear stiffness in the form of 1/ks.
        """

        # it's constant, if it doesn't have any crack.
        if not self._ks_transiction:
            f_transiction = lambda gamma: (
                1.2
                * np.cos(tau_op) ** 2
                / (self.material.G_s * self._compute_transition_curve(gamma)[2])
                * self._diff_gamma(gamma)
            )

            self._ks_transiction, _ = sp.integrate.quad(
                f_transiction, np.pi / 2, self.pr_angle
            )  # OK

        f_involute = lambda tau: (
            1.2
            * (
                np.cos(tau_op) ** 2
                / (
                    self.material.G_s * self._compute_involute_curve(tau)[2]
                )  # verificar se esse 2 é a área msm
                * self._diff_tau(tau)
            )
        )

        k_involute, _ = sp.integrate.quad(
            f_involute, self._to_tau(self.geometry_dict["alpha_c"]), tau_op
        )

        return self._ks_transiction + k_involute

    def _inv_kb(self, tau_op):
        """Calculate the stiffness contribution from the gear resistance from
        bending stresses, given the tau operational angle.

        Parameters
        ----------
        tau_op : float
            Operational tau angle.

        Returns
        -------
        float
            The bending stiffness in the form of 1/kb.
        """
        y_op, x_op, _, _ = self._compute_involute_curve(tau_op)

        if not self._kb_transiction:
            f_transiction = lambda gamma: (
                (
                    np.cos(tau_op) * (y_op - self._compute_transition_curve(gamma)[0])
                    - x_op * np.sin(tau_op)
                )
                ** 2
                / (self.material.E * self._compute_transition_curve(gamma)[3])
                * self._diff_gamma(gamma)
            )

            self._kb_transiction, _ = sp.integrate.quad(
                f_transiction, np.pi / 2, self.pr_angle
            )

        f_involute = lambda tau: (
            (
                np.cos(tau_op) * (y_op - self._compute_involute_curve(tau)[0])
                - x_op * np.sin(tau_op)
            )
            ** 2
            / (self.material.E * self._compute_involute_curve(tau)[3])
            * self._diff_tau(tau)
        )

        k_involute, _ = sp.integrate.quad(
            f_involute, self._to_tau(self.geometry_dict["alpha_c"]), tau_op
        )

        return self._kb_transiction + k_involute

    def _inv_ka(self, tau_op):
        """Calculate the stiffness contribution from the gear resistance from
        axial stresses, given the tau operational angle.

        Parameters
        ----------
        tau_op : float
            Operational tau angle.

        Returns
        -------
        float
            The axial stiffness in the form of 1/ka.
        """
        if not self._ka_transiction:
            f_transiction = lambda gamma: (
                np.sin(tau_op) ** 2
                / (self.material.E * self._compute_transition_curve(gamma)[2])
                * self._diff_gamma(gamma)
            )

            self._ka_transiction, _ = sp.integrate.quad(
                f_transiction, np.pi / 2, self.pr_angle
            )

        f_involute = lambda tau: (
            np.sin(tau_op) ** 2
            / (self.material.E * self._compute_involute_curve(tau)[2])
            * self._diff_tau(tau)
        )

        k_involute, _ = sp.integrate.quad(
            f_involute, self._to_tau(self.geometry_dict["alpha_c"]), tau_op
        )

        return self._ka_transiction + k_involute

    @staticmethod
    def _kh(gear1, gear2):
        """Evaluates the contact hertzian stiffness considering that both
        elasticity modulus are equal.

        Parameters
        ----------
        gear1 : GearElementTVMS
            gear_input object.

        gear2 : GearElementTVMS
            Gear object.

        Returns
        -------
        float
            The hertzian contact stiffness.
        """

        return (
            np.pi * gear1.material.E * gear1.width / 4 / (1 - gear1.material.Poisson**2)
        )


class Mesh:
    """Represents the meshing behavior between two gears, typically a
    gear_input and a crown gear, including stiffness and contact ratio
    calculations.

    Parameters:
    -----------
    gear_input : GearElementTVMS
        The gear_input gear object used in the gear pair (driver).
    gear_output : GearElementTVMS
        The crown gear object used in the gear pair (driven).
    gear_input_w : float
        The rotational speed [rad/sec] of the gear_input gear in rad/s.
    tvms : bool
        If True, it will run the TVMS once and interpolate after (increased
        performance).
    max_stiffness : bool
        If True, return only the max stiffness of the meshing.


    Attributes:
    -----------
    gear_input : GearElementTVMS
        The gear_input gear object, which contains information about the
        geometry and properties of the gear_input gear.
    gear_output : GearElementTVMS
        The gear wheel object, which contains information about the geometry
        and properties of the wheel gear.
    tm : float
        The meshing period, calculated based on the rotational speed and the
        number of teeth of the gear_input.
    cr : float
        The contact ratio, representing the average number of tooth in contact
        during meshing.
    eta : float
        The transamission ratio, defined as the ratio of the radii between the
        driven and driving gears.
    """

    def __init__(
        self,
        gear_input,
        gear_output,
        tvms=False,
        only_max_stiffness=False,
        user_defined_stiffness=None,
    ):
        self._user_defined_stiffness = user_defined_stiffness

        self.gear_input = gear_input
        self.gear_output = gear_output

        self.time = 0

        self.eta = gear_output.n_tooth / gear_input.n_tooth  # Gear ratio

        self._kh = GearElementTVMS._kh(gear_input, gear_output)
        self.cr = self.contact_ratio(self.gear_input, self.gear_output)

        self.tvms = tvms
        self.only_max_stiffness = only_max_stiffness
        self.already_evaluated_max = False
        self.already_interpolated = False

    @staticmethod
    def contact_ratio(gear_input, gear_output):
        """
        Parameters:
        ---------
        gear_input : GearElementTVMS
            The gear_input object.
        gear_output : GearElementTVMS
            The gear_output object.

        Returns
        -------
        CR : float
            The contact ratio of the gear pair.

        Example
        -------
        >>> Mesh.contact_ratio(gear_input, Gear)
        1.7939883590132295

        Reference
        ----------
        Understanding the contact ratio for spur gears with some comments on
        ways to read a textbook.
        Retrieved from : https://www.myweb.ttu.edu/amosedal/articles.html
        """

        pb = (
            np.pi * 2 * gear_input.geometry_dict["r_b"] / gear_input.n_tooth
        )  # base pitch
        C = (
            gear_input.geometry_dict["r_p"] + gear_output.geometry_dict["r_p"]
        )  # center distance (not the operating one)

        lc = (  # length of contact (not the operating one) # OK
            np.sqrt(
                gear_input.geometry_dict["r_a"] ** 2
                - gear_input.geometry_dict["r_b"] ** 2
            )
            + np.sqrt(
                gear_output.geometry_dict["r_a"] ** 2
                - gear_output.geometry_dict["r_b"] ** 2
            )
            - C * np.sin(gear_input.pr_angle)
        )

        CR = lc / pb  # contact ratio

        return CR

    def _time_equivalent_stiffness(self, t, gear_input_speed):
        """
        Parameters
        ---------
        gear_input : GearElementTVMS
            The gear_input object.
        t : float
            The time of meshing [0, self.tm]

        Returns
        -------
        A containing the following elements:
            k_t : float
                The time equivalent stiffness of mesh contact.
            d_tau_gear_input : float

        Example
        --------
        >>> self._time_equivalent_stiffness()
        167970095.70859054
        """

        gear_output_speed = -gear_input_speed / self.eta

        # Angular displacements
        alphagear_input = (
            t * gear_input_speed + self.gear_input.geometry_dict["alpha_c"]
        )  # angle variation of the input gear_input [rad]
        alphagear_output = (
            t * gear_output_speed + self.gear_output.geometry_dict["alpha_a"]
        )  # angle variation of the output gear   [rad]

        # Tau displacementes
        d_tau_gear_input = self.gear_input._to_tau(
            alphagear_input
        )  # angle variation of the gear_input in tau [rad]
        d_tau_gear_output = self.gear_output._to_tau(
            alphagear_output
        )  # angle variation of the gear_input in tau [rad]

        # Contact stiffness according to tau angles
        ka_1, kb_1, kf_1, ks_1 = self.gear_input._compute_stiffness(d_tau_gear_input)
        ka_2, kb_2, kf_2, ks_2 = self.gear_output._compute_stiffness(d_tau_gear_output)

        # Evaluating the equivalate meshing stiffness.
        k_t = 1 / (
            1 / self._kh
            + 1 / ka_1
            + 1 / kb_1
            + 1 / kf_1
            + 1 / ks_1
            + 1 / ka_2
            + 1 / kb_2
            + 1 / kf_2
            + 1 / ks_2
        )

        return k_t, d_tau_gear_input, d_tau_gear_output

    @check_units
    def mesh(self, gear_input_speed, t=None):
        """Calculate the time-varying meshing stiffness of a gear pair.

        This method computes the equivalent stiffness of a gear mesh at a given
        time `t`, taking into account the periodic nature of the meshing
        process and the contact ratio (`cr`) of the gear pair.
        The computation considers whether one or two pairs of teeth are in
        contact during the meshing cycle.

        Parameters
        ----------
        gear_input_speed : GearElementTVMS
            The gear_input object.
        t : float
            Time instant for which the meshing stiffness is calculated.

        Returns
        -------
        A tuple containing the following elements:
            total_stiffness : float
                The total equivalent meshing stiffness at time `t`.
            stiffness_tooth_pair_1 : float
                The meshing stiffness of the first tooth pair in contact.
                If no first tooth pair is in contact, this value is `np.nan`.
            stiffness_tooth_pair_2 : float
                The meshing stiffness of the second tooth pair in contact.

        Notes
        -----
        - The calculation considers the periodic nature of meshing and the
        contact ratio (cr) of the gear pair.
        - The stiffness contribution varies depending on whether one or two
        pairs of teeth are in contact.
        - For the correct evaluation of the TVMS it's important that
        `dt = (2 * np.pi) / (K * n_toth_gear_x * speed_gear_x)`
            Where K is the discretization ratio of a double gear-mesh contact,
            recommended K >= 20.
        """
        # OPTION 0 : RETURN ONLY THE USER DEFINED STIFFNESS
        if isinstance(self._user_defined_stiffness, (float, int)):
            return self._user_defined_stiffness, None, None

        t = float(t)
        gear_input_speed = float(gear_input_speed)

        tm = (
            2 * np.pi / (gear_input_speed * self.gear_input.n_tooth)
        )  # Gearmesh period [seconds/engagement]
        ctm = (
            self.cr * tm
        )  # [seconds/tooth] how much time each tooth remains in contact

        # OPTION 1 : RETURN ONLY MAX STIFFNESS
        if self.only_max_stiffness:
            if not self.already_evaluated_max:
                dt = (2 * np.pi) / (20 * self.gear_input.n_tooth * gear_input_speed)
                self.max_stiffness = self._max_gear_stiff(gear_input_speed, dt)
                self.already_evaluated_max = True
                return self.max_stiffness, None, None
            else:
                return self.max_stiffness, None, None

        # OPTION 2 : RUN INTERPOLATION ONLY
        if self.tvms:  # Runs the time dependency for one period of double-single mesh
            if (
                not self.already_interpolated
            ):  # Case 1 : If it had never evaluated the stiffness
                if (2 * np.pi) / (
                    100 * self.gear_input.n_tooth * gear_input_speed
                ) < 1e-5:
                    dt = (2 * np.pi) / (
                        100 * self.gear_input.n_tooth * gear_input_speed
                    )

                dt = 1e-5
                t_interpol, double_contact, single_contact = self._time_stiffness(
                    gear_input_speed, dt
                )

                mask_double_contact = double_contact > 0
                self.double_contact = double_contact[mask_double_contact]
                self.t_interpol_double = t_interpol[mask_double_contact]

                mask_single_contact = single_contact > 0
                self.single_contact = single_contact[mask_single_contact]
                self.t_interpol_single = t_interpol[mask_single_contact]

                self.already_interpolated = True

            # Case 2 : If the stiffness is already known

            t = t - t // tm * tm

            if t <= (self.cr - 1) * tm:
                return (
                    np.interp(t, self.t_interpol_double, self.double_contact),
                    None,
                    None,
                )

            elif t > (self.cr - 1) * tm:
                return (
                    np.interp(t, self.t_interpol_single, self.single_contact),
                    None,
                    None,
                )

        # OPTION 3 : RUN THE TVMS EVERY STEP
        else:  # If it needs to re-evaluate every stiffness integration every step
            t = t - t // tm * tm

            if t <= (self.cr - 1) * tm:
                stiffnes_mesh_1, d_tau_gear_input_1, d_tau_gear_1 = (
                    self._time_equivalent_stiffness(t, gear_input_speed)
                )
                stiffnes_mesh_0, d_tau_gear_input_0, d_tau_gear0 = (
                    self._time_equivalent_stiffness(tm + t, gear_input_speed)
                )

                return (
                    stiffnes_mesh_0 + stiffnes_mesh_1,
                    stiffnes_mesh_0,
                    stiffnes_mesh_1,
                )

            elif t > (self.cr - 1) * tm:
                stiffnes_mesh_1, d_tau_gear_input_1, d_tau_gear_1 = (
                    self._time_equivalent_stiffness(t, gear_input_speed)
                )

                return stiffnes_mesh_1, np.nan, stiffnes_mesh_1

    def _time_stiffness(self, gear_input_speed, dt):
        """Calculate the time-varying meshing stiffness of a gear pair in ONE
        time-mesh period.

        - This method is used for interpolation, since the TVMS is constant for
        gear-pairs without deffects.
        - This method is also used for evaluating the _max_stiffness, since in
        one time-mesh period it's possible to evaluate the maximum stiffness.

        Parameters
        ----------
        gear_input_speed : GearElementTVMS
            The gear_input object.
        t : float
            Time instant for which the meshing stiffness is calculated.

        Returns
        -------
        A tuple containing the following elements:
            total_stiffness : float
                The total equivalent meshing stiffness at time `t`.
            stiffness_tooth_pair_1 : float
                The meshing stiffness of the first tooth pair in contact.
                If no first tooth pair is in contact, this value is `np.nan`.
            stiffness_tooth_pair_2 : float
                The meshing stiffness of the second tooth pair in contact.
        """
        tm = (
            2 * np.pi / (gear_input_speed * self.gear_input.n_tooth)
        )  # Gearmesh period [seconds/engagement]
        ctm = (
            self.cr * tm
        )  # [seconds/tooth] how much time each tooth remains in contact

        t_interpol = np.arange(0, tm + dt, dt)
        double_contact = np.zeros(np.shape(t_interpol))
        single_contact = np.zeros(np.shape(double_contact))

        for i, t in enumerate(t_interpol):
            t = t - t // tm * tm

            if t <= (self.cr - 1) * tm:
                stiffnes_mesh_1, _, _ = self._time_equivalent_stiffness(
                    t, gear_input_speed
                )
                stiffnes_mesh_0, _, _ = self._time_equivalent_stiffness(
                    tm + t, gear_input_speed
                )

                double_contact[i] = stiffnes_mesh_0 + stiffnes_mesh_1

            elif t > (self.cr - 1) * tm:
                stiffnes_mesh_1, _, _ = self._time_equivalent_stiffness(
                    t, gear_input_speed
                )

                single_contact[i] = stiffnes_mesh_1

        return t_interpol, double_contact, single_contact

    def _max_gear_stiff(self, gear_input_speed, dt):
        """Evaluate the maximum meshing stiffness from one time-mesh period.

        Parameters
        ----------
        gear_input_speed : GearElementTVMS
            The gear_input object.
        t : float
            Time instant for which the meshing stiffness is calculated.

        Returns
        -------
        np.max(double_contact) : float
            The maximum stiffness [N/m]
        """

        _, double_contact, _ = self._time_stiffness(gear_input_speed, dt)

        return np.max(double_contact)
