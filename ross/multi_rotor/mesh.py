import math
import numpy as np
from plotly import graph_objects as go
from warnings import warn

from ross.units import Q_

from .gear_element import GearElementTVMS
from .utils import involute, mod


__all__ = ["Mesh"]


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
    square_varying_stiffness: boll, optional
        Set the square shape time varying mesh stiffness
    square_stiffness_amplitude_ratio: float, optional
        Ratio of stiffness amplitude based on the mean value of stiffness.
    orientation_angle : float, pint.Quantity, optional
        The angle between the line of gear centers and x-axis. Default is 0.0 rad.

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
    419603831.338...
    """

    def __init__(
        self,
        driving_gear,
        driven_gear,
        gear_mesh_stiffness=None,
        square_varying_stiffness={"enable": False, "amplitude_ratio": 0},
        backlash={
            "enable": False,
            "initial_value": 0.0,
            "error_amp": 0.0,
            "smooth_operator": False,
            "sigma": 1e4,
        },
        damping_ratio=0.07,
        orientation_angle=0,
    ):

        if not math.isclose(driving_gear.module, driven_gear.module, rel_tol=0.05):
            warn(
                "Gear modules must match for proper meshing | "
                f"Driving gear: {driving_gear.module:.4f}, Driven gear: {driven_gear.module:.4f}"
            )

        if driving_gear.width and driven_gear.width:
            if not math.isclose(driving_gear.width, driven_gear.width, rel_tol=0.05):
                warn(
                    "Gear widths must match for proper meshing | "
                    f"Driving gear: {driving_gear.width:.4f}, Driven gear: {driven_gear.width:.4f}"
                )

        self.driving_gear = driving_gear
        self.driven_gear = driven_gear
        self.gear_ratio = (
            driving_gear.n_teeth / driven_gear.n_teeth
        )  # Shigley Machine Elements
        self.pressure_angle = driving_gear.pr_angle
        self.helix_angle = driving_gear.helix_angle

        self.square_stiffness_amplitude_ratio = square_varying_stiffness[
            "amplitude_ratio"
        ]
        self.orientation_angle = orientation_angle
        self.module = driving_gear.module
        self.damping_ratio = damping_ratio
        self.contact_ratio = self.calculate_contact_ratio()

        stiffness_type = "constant"

        if gear_mesh_stiffness is None:
            if isinstance(driving_gear, GearElementTVMS) and isinstance(
                driven_gear, GearElementTVMS
            ):
                stiffness_type = "equivalent"

                w1 = driving_gear.width
                poisson = driving_gear.material.Poisson
                E1 = driving_gear.material.E
                E2 = driven_gear.material.E

                self.stiffness = (self.contact_ratio * w1 * E1 * E2) / (9 * (E1 + E2))
                self.hertzian_stiffness = np.pi * w1 * E1 / (4 * (1 - poisson**2))

            else:
                if driving_gear.width:
                    w1 = driving_gear.width
                    E1 = driving_gear.material.E
                    E2 = driven_gear.material.E

                    self.stiffness = (self.contact_ratio * w1 * E1 * E2) / (
                        9 * (E1 + E2)
                    )

                else:
                    raise TypeError(
                        "Missing 'gear_mesh_stiffness'. You have two options if you don't set this value:\n"
                        "1) Provide 'material' and 'bore_diameter' for 'GearElement', or\n"
                        "2) Use 'GearElementTVMS' instead"
                    )

            if square_varying_stiffness["enable"]:
                stiffness_type = "square"

        else:
            self.stiffness = gear_mesh_stiffness

        self.theta_range, self.stiffness_range = self.get_stiffness_for_mesh_period(
            stiffness_type=stiffness_type
        )

        if backlash["enable"]:
            theta_range, contact_ratio_range, stiffness_table = (
                self.generate_stiffness_table(stiffness_type=stiffness_type)
            )
            self.backlash = Backlash(
                pressure_angle=self.pressure_angle,
                orientation_angle=self.orientation_angle,
                helix_angle=self.helix_angle,
                damping_ratio=self.damping_ratio,
                module=self.module,
                driving_gear=self.driving_gear,
                driven_gear=self.driven_gear,
                theta_range=theta_range,
                contact_ratio_range=contact_ratio_range,
                stiffness_table=stiffness_table,
                initial_value=backlash["initial_value"],
                error_amp=backlash["error_amp"],
                smooth_operator=backlash["smooth_operator"],
                sigma=backlash["sigma"],
            )
        else:
            self.backlash = None

    def calculate_contact_ratio(self):
        """Calculates the contact ratio of the gear pair.

        Returns
        -------
        contact_ratio : float
            The calculated contact ratio.
        """
        Ra1 = self.driving_gear.addendum_radius
        Ra2 = self.driven_gear.addendum_radius

        Rb1 = self.driving_gear.base_radius
        Rb2 = self.driven_gear.base_radius

        Rp1 = self.driving_gear.pitch_diameter / 2
        Rp2 = self.driven_gear.pitch_diameter / 2
        center_distance = Rp1 + Rp2

        contact_length = (
            np.sqrt(Ra1**2 - Rb1**2)
            + np.sqrt(Ra2**2 - Rb2**2)
            - center_distance * np.sin(self.pressure_angle)
        )

        base_pitch = np.pi * self.module * np.cos(self.pressure_angle)

        contact_ratio = contact_length / base_pitch

        return contact_ratio

    def _angular_equivalent_stiffness(self, d_alpha):
        """Calculate the angular equivalent stiffness of a gear pair.

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

    def get_variable_equivalent_stiffness(self, angular_position, contact_ratio):
        """Calculate the variable equivalent stiffness of a gear pair.

        This method computes the equivalent stiffness of a gear mesh at a given
        angular position, taking into account the periodic nature of the meshing
        process and the contact ratio of the gear pair. It is assumed constant
        rotor speed.

        Parameters
        ----------
        angular_position : float
            Gear angular position for which the meshing stiffness is calculated (rad).
        contact_ratio : float
            The contact ratio of the gear pair.

        Returns
        -------
        stiffness : float
            The total equivalent meshing stiffness at the given angular position.
        """
        cr = contact_ratio
        alpha_c = self.driving_gear.pr_angles_dict["start_point"]
        alpha_a = self.driving_gear.pr_angles_dict["addendum"]

        tm_om = 2 * np.pi / self.driving_gear.n_teeth
        theta = mod(angular_position, tm_om)

        d_meshing = (alpha_a - alpha_c) / cr
        d_alpha = d_meshing / tm_om * theta

        stiffness = self._angular_equivalent_stiffness(d_alpha)

        if d_alpha <= d_meshing * (cr - 1):
            stiffness += self._angular_equivalent_stiffness(d_alpha + d_meshing)

        return stiffness

    def get_square_varying_stiffness(self, theta_range, contact_ratio):
        """Calculate the square varying stiffness of a gear pair.

        Parameters
        ----------
        theta_range : array-like
            Angular positions at which to calculate the stiffness (rad).
        contact_ratio : float
            The contact ratio of the gear pair.

        Returns
        -------
        stiffness_range : array-like
            Stiffness values at the given angular positions (N/m).
        """
        n_terms = 100  # number of terms in the Fourier series expansion

        cr = contact_ratio
        phase = self.orientation_angle
        Kg = self.stiffness
        Ka = Kg * self.square_stiffness_amplitude_ratio
        n_teeth = self.driving_gear.n_teeth

        Kv_unit = []
        stiffness_range = []

        for angular_position in theta_range:
            # Fourier series coefficients
            A = [0]
            B = [0]
            Kv = 0

            for s in range(1, n_terms + 2):
                A.append(
                    (-2 / (s * np.pi))
                    * np.sin(s * np.pi * (cr - 2 * phase))
                    * np.sin(s * np.pi * cr)
                )
                B.append(
                    (-2 / (s * np.pi))
                    * np.cos(s * np.pi * (cr - 2 * phase))
                    * np.sin(s * np.pi * cr)
                )

                Kv = (
                    Kv
                    + A[s] * np.sin(s * n_teeth * (angular_position))
                    + B[s] * np.cos(s * n_teeth * (angular_position))
                )

            Kv_unit.append(Kv)

        mean_kv = np.mean(Kv_unit)

        Kv_aux = []
        minus_multiplier = []
        maximus_multiplier = []

        for i in range(len(Kv_unit)):
            if Kv_unit[i] < mean_kv:
                Kv_aux.append(-1)
                minus_multiplier.append(Kv_unit[i] / -1)
            elif Kv_unit[i] > mean_kv:
                Kv_aux.append(1)
                maximus_multiplier.append(Kv_unit[i] / 1)
            else:
                Kv_aux.append(0)

        minus_multiplier_value = np.median(sorted(minus_multiplier))
        maximus_multiplier_value = np.median(sorted(maximus_multiplier))

        Kv_aux = np.array(Kv_aux, dtype=float)

        Kv_aux[Kv_aux == -1] *= minus_multiplier_value
        Kv_aux[Kv_aux == 1] *= maximus_multiplier_value

        for i in range(len(Kv_aux)):
            stiffness_range.append(Kg - 2 * Ka * Kv_aux[i])

        return np.array(stiffness_range)

    def get_stiffness_for_mesh_period(
        self, stiffness_type="constant", n_mesh_period=1, n_points=1000
    ):
        """Computes the mesh stiffness profile over a specified number of gear
        mesh periods.

        Parameters
        ----------
        stiffness_type : str
            Type of stiffness to compute. Available options are:
            - "square": square varying stiffness
            - "equivalent": variable equivalent stiffness
            otherwise, constant stiffness is computed.
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
        cr = self.contact_ratio

        if stiffness_type == "equivalent":
            stiffness_range = np.vectorize(self.get_variable_equivalent_stiffness)(
                theta_range, cr
            )
        elif stiffness_type == "square":
            stiffness_range = self.get_square_varying_stiffness(theta_range, cr)
        else:
            stiffness_range = np.full(n_points, self.stiffness)

        return theta_range, stiffness_range

    def interpolate_stiffness(self, angular_position):
        """Interpolates the mesh stiffness value at a given angular position.

        Parameters
        ----------
        angular_position : float or array-like
            Angular position(s) at which to evaluate the stiffness (rad).

        Returns
        -------
        stiffness : float or np.ndarray
            Interpolated stiffness value(s) in N/m.
        """
        theta = mod(angular_position, max(self.theta_range))
        stiffness = np.interp(theta, self.theta_range, self.stiffness_range)

        return stiffness

    def generate_stiffness_table(self, stiffness_type="square", n_points=200):
        """Generates a table of stiffness values for a gear pair.

        Parameters
        ----------
        stiffness_type : str, optional
            Type of stiffness to compute. Available options are:
            - "square": square varying stiffness
            - "equivalent": variable equivalent stiffness
        n_points : int, optional
            Number of data points to evaluate for the stiffness profile.
            Default is 200.

        Returns
        -------
        theta_range : np.ndarray
            Array of angular positions (rad).
        contact_ratio_range : np.ndarray
            Array of contact ratios.
        stiffness_table : np.ndarray
            Array of stiffness values corresponding to each angular position and contact ratio.
        """
        theta_end = 2 * np.pi / self.driving_gear.n_teeth
        theta_range = np.linspace(0, theta_end, n_points)
        cr_range = np.linspace(0.8, 2.5, n_points)

        if stiffness_type == "equivalent":
            stiffness_table = np.vectorize(self.get_variable_equivalent_stiffness)(
                theta_range[:, None], cr_range[None, :]
            )
        else:
            stiffness_table = np.array(
                [self.get_square_varying_stiffness(theta_range, cr) for cr in cr_range]
            ).T

        return theta_range, cr_range, stiffness_table

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


class Backlash:
    """Backlash model for a gear pair.

    The implementation was based on the work of Yi et al. (2019) and equations were
    expanded using the works of Kubur et al. (2004) and Mo et al. (2025).

    Parameters
    ----------
    pressure_angle : float
        Pressure angle of the gear pair.
    orientation_angle : float
        Orientation angle of the gear pair.
    helix_angle : float
        Helix angle of the gear pair.
    damping_ratio : float
        Damping ratio of the gear pair.
    module : float
        Module of the gear pair.
    driving_gear : Gear
        Driving gear of the gear pair.
    driven_gear : Gear
        Driven gear of the gear pair.
    theta_range : array-like
        Angular positions at which the stiffness table is defined (rad).
    contact_ratio_range : array-like
        Contact ratios at which the stiffness table is defined.
    stiffness_table : array-like
        Stiffness values at the given angular positions and contact ratios.
        It is a 2D array of shape `(len(theta_range), len(contact_ratio_range))`.
    initial_value : float, optional
        Initial backlash of the gear pair.
        Default is 0.0.
    error_amp : float, optional
        Error amplitude used to calculate the backlash force model.
        Default is 0.0.
    smooth_operator : bool, optional
        Whether to use a smooth operator.
        Default is False.
    sigma : float, optional
        Parameter related to the regularization of the smooth approach.
        Default is 1e4.

    References
    ----------
    KUBUR, M.; KAHRAMAN, A.; ZINI, D. M.; KIENZLE, K. Dynamic analysis of a multi-shaft helical
    gear transmission by finite elements: Model and experiment. Journal of Vibration and Acoustics,
    American Society of Mechanical Engineers, v. 126, n. 3, p. 398–406, 2004.

    MO, G.; LIU, C.; LIU, G.; LIU, F. Improved nonlinear dynamic model of helical gears considering
    frictional excitation and fractal effects in backlash. Machines, MDPI, v. 13, n. 4, p. 262, 2025.

    YI, Y.; HUANG, K.; XIONG, Y.; SANG, M. Nonlinear dynamic modelling and analysis for a spur gear
    system with time-varying pressure angle and gear backlash. Mechanical Systems and Signal Processing,
    Elsevier, v. 132, p. 18–34, 2019.
    """

    def __init__(
        self,
        pressure_angle,
        orientation_angle,
        helix_angle,
        damping_ratio,
        module,
        driving_gear,
        driven_gear,
        theta_range,
        contact_ratio_range,
        stiffness_table,
        initial_value=0.0,
        error_amp=0.0,
        smooth_operator=False,
        sigma=1e4,
    ):

        self.initial_value = initial_value
        self.error_amp = error_amp

        self.sigma = sigma
        if smooth_operator:
            self.apply_penalty_function = self.smooth_approach
        else:
            self.apply_penalty_function = self.rigid_approach

        self.pressure_angle = pressure_angle
        self.orientation_angle = orientation_angle
        self.helix_angle = helix_angle
        self.damping_ratio = damping_ratio
        self.module = module

        self.n_teeth = driving_gear.n_teeth

        self.driving_gear_base_radius = driving_gear.base_radius
        self.driven_gear_base_radius = driven_gear.base_radius
        self.driving_gear_pitch_radius = driving_gear.pitch_diameter / 2
        self.driven_gear_pitch_radius = driven_gear.pitch_diameter / 2
        self.driving_gear_addendum_radius = driving_gear.addendum_radius
        self.driven_gear_addendum_radius = driven_gear.addendum_radius

        self.driving_gear_dofs = list(driving_gear.dof_global_index.values())
        self.driven_gear_dofs = list(driven_gear.dof_global_index.values())

        Ip1 = driving_gear.Ip
        Ip2 = driven_gear.Ip
        Rb1 = self.driving_gear_base_radius
        Rb2 = self.driven_gear_base_radius
        self.M_eq = (Ip1 * Ip2) / (Ip2 * (Rb1**2) + Ip1 * (Rb2**2))

        self.theta_range = theta_range
        self.contact_ratio_range = contact_ratio_range
        self.stiffness_table = stiffness_table

        data_keys = [
            "time",
            "delta",
            "bt",
            "Fm",
            "K_time",
            "d",
            "alfa",
            "contact_ratio",
        ]
        self._data = {key: list() for key in data_keys}

    def calculate_force(self, step, disp_resp, velc_resp, time, angular_pos, speed):
        """Calculates the backlash force to be used in time response integration with Newmark method.

        Parameters
        ----------
        step : int
            Step number.
        disp_resp : array-like
            Displacement response.
        velc_resp : array-like
            Velocity response.
        time : float
            Time (s).
        angular_pos : float
            Angular position of rotor system.
        speed : float
            Speed of the rotor system.

        Returns
        -------
        backlash_force : array-like
            Backlash force.
        """

        alpha0 = self.pressure_angle
        orientation_angle = self.orientation_angle
        helix_angle = self.helix_angle
        n_teeth = self.n_teeth
        Rp1 = self.driving_gear_pitch_radius
        Rp2 = self.driven_gear_pitch_radius
        Rb1 = self.driving_gear_base_radius
        Rb2 = self.driven_gear_base_radius
        damping_ratio = self.damping_ratio
        dofs1 = self.driving_gear_dofs
        dofs2 = self.driven_gear_dofs
        b0 = self.initial_value

        x1, y1, z1, rx1, ry1, t1 = disp_resp[dofs1]
        vx1, vy1, vz1, vrx1, vry1, vt1 = velc_resp[dofs1]

        x2, y2, z2, rx2, ry2, t2 = disp_resp[dofs2]
        vx2, vy2, vz2, vrx2, vry2, vt2 = velc_resp[dofs2]

        error = self.error_amp * np.sin(n_teeth * angular_pos)
        error_dot = self.error_amp * (n_teeth * speed) * np.cos(n_teeth * angular_pos)

        # Calculation of the non-linear kinematics in the transverse plane
        d0 = Rp1 + Rp2
        x2_abs = x2 + d0 * np.cos(orientation_angle)
        y2_abs = y2 + d0 * np.sin(orientation_angle)

        dx = x2_abs - x1
        dy = y2_abs - y1
        beta = np.arctan2(dy, dx)

        d_inst = max(np.sqrt(dx**2 + dy**2), 1e-12)

        cos_alpha_val = np.clip((Rb1 + Rb2) / d_inst, -1.0, 1.0)
        alpha = np.arccos(cos_alpha_val)

        psi = alpha - beta
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        sin_beta_h = np.sin(helix_angle)
        cos_beta_h = np.cos(helix_angle)

        # Basic Geometric Derivatives
        d_pow2 = d_inst**2
        term_in_sqrt = max(d_pow2 - (Rb1 + Rb2) ** 2, 1e-12)
        term_sqrt = np.sqrt(term_in_sqrt)

        alpha_1 = -((Rb1 + Rb2) * np.array([dx, dy])) / (d_pow2 * term_sqrt)
        beta_1 = np.array([dy, -dx]) / d_pow2
        alpha_2 = -alpha_1
        beta_2 = -beta_1

        # Full DTE 3D Equation
        delta = (
            ((x1 - x2) * sin_psi + (y1 - y2) * cos_psi + Rb1 * t1 + Rb2 * t2)
            * cos_beta_h
            + (
                (-z1 + z2)
                + (Rb1 * rx1 + Rb2 * rx2) * sin_psi
                + (Rb1 * ry1 + Rb2 * ry2) * cos_psi
            )
            * sin_beta_h
            - error
        )

        # Dynamic backlash calculation (bt)
        delta_b = (Rb1 + Rb2) * (involute(alpha) - involute(alpha0))
        bt = b0 + delta_b * cos_beta_h

        # Derivatives of delta with respect to the DOFs
        geo_tr = (x1 - x2) * cos_psi - (y1 - y2) * sin_psi
        geo_rt = (Rb1 * rx1 + Rb2 * rx2) * cos_psi - (Rb1 * ry1 + Rb2 * ry2) * sin_psi

        psi_1 = alpha_1 - beta_1
        psi_2 = alpha_2 - beta_2

        tri_psi = np.array([sin_psi, cos_psi])
        d_delta_dt1 = np.append(
            (tri_psi + geo_tr * psi_1) * cos_beta_h + (geo_rt * psi_1) * sin_beta_h,
            -sin_beta_h,
        )
        d_delta_dr1 = Rb1 * np.append(tri_psi * sin_beta_h, cos_beta_h)
        d_delta_d1 = np.append(d_delta_dt1, d_delta_dr1)

        d_delta_dt2 = np.append(
            (-tri_psi + geo_tr * psi_2) * cos_beta_h + (geo_rt * psi_2) * sin_beta_h,
            sin_beta_h,
        )
        d_delta_dr2 = Rb2 * np.append(tri_psi * sin_beta_h, cos_beta_h)
        d_delta_d2 = np.append(d_delta_dt2, d_delta_dr2)

        # Derivatives of backlash with respect to the DOFs
        tan2_alpha = np.tan(alpha) ** 2
        bt_1 = (Rb1 + Rb2) * tan2_alpha * alpha_1 * cos_beta_h
        bt_2 = (Rb1 + Rb2) * tan2_alpha * alpha_2 * cos_beta_h

        delta_dot = (
            np.dot(d_delta_d1, np.array([vx1, vy1, vz1, vrx1, vry1, vt1]))
            + np.dot(d_delta_d2, np.array([vx2, vy2, vz2, vrx2, vry2, vt2]))
            - error_dot
        )

        # Derivative of alpha with respect to the DOFs
        # Only the translations affect alpha, so alpha_dot depends on vx and vy
        alpha_dot = np.dot(alpha_1, np.array([vx1, vy1])) + np.dot(
            alpha_2, np.array([vx2, vy2])
        )
        bt_dot = (Rb1 + Rb2) * tan2_alpha * alpha_dot * cos_beta_h

        # Penalty function application
        f_val, f1_val, f1, f2 = self.apply_penalty_function(
            delta, delta_dot, d_delta_d1, d_delta_d2, bt, bt_dot, bt_1, bt_2
        )

        contact_ratio = self.calculate_contact_ratio(d_inst, alpha, alpha0)

        k_m = self.interpolate2d_stiffness(angular_pos, contact_ratio)
        c_m = 2.0 * damping_ratio * np.sqrt(k_m * self.M_eq)

        # Total normal force in the action line
        Fm = k_m * f_val + c_m * f1_val

        # Force decomposition: Q_i = -Fm * f_(q_i)
        backlash_force = np.zeros(len(disp_resp))
        backlash_force[dofs1] = -Fm * f1
        backlash_force[dofs2] = -Fm * f2

        # ATTENTION: Check this!!
        results = {
            "time": time,
            "delta": delta,
            "bt": bt,
            "Fm": Fm,
            "K_time": k_m,
            "d": d_inst,
            "alfa": alpha,
            "contact_ratio": contact_ratio,
        }
        self._save_time_results(step, results)

        return backlash_force

    def calculate_contact_ratio(self, center_distance, pr_angle_op, pr_angle_nom):
        """Calculates the contact ratio of the gear pair.

        Parameters
        ----------
        center_distance : float
            Distance between the gears (m).
        pr_angle_op : float
            Operating pressure angle (rad).
        pr_angle_nom : float
            Nominal pressure angle (rad).

        Returns
        -------
        contact_ratio : float
            Contact ratio of the gear pair.
        """
        Ra1 = self.driving_gear_addendum_radius
        Ra2 = self.driven_gear_addendum_radius

        Rb1 = self.driving_gear_base_radius
        Rb2 = self.driven_gear_base_radius

        contact_length = (
            np.sqrt(Ra1**2 - Rb1**2)
            + np.sqrt(Ra2**2 - Rb2**2)
            - center_distance * np.sin(pr_angle_op)
        )

        base_pitch = np.pi * self.module * np.cos(pr_angle_nom)

        contact_ratio = contact_length / base_pitch

        return contact_ratio

    def rigid_approach(
        self, delta, delta_dot, d_delta_d1, d_delta_d2, bt, bt_dot, bt_1, bt_2
    ):
        """Rigid approach for the backlash force model.

        Parameters
        ----------
        delta : float
            Transmission error.
        delta_dot : float
            Transmission error derivative.
        d_delta_d1 : array-like
            Derivative of the transmission error with respect to the first gear DOFs.
        d_delta_d2 : array-like
            Derivative of the transmission error with respect to the second gear DOFs.
        bt : float
            Backlash.
        bt_dot : float
            Backlash derivative.
        bt_1 : array-like
            Derivative of the backlash with respect to the first gear DOFs.
        bt_2 : array-like
            Derivative of the backlash with respect to the second gear DOFs.

        Returns
        -------
        f_val : float
            Backlash force.
        f1_val : float
            Backlash force derivative with respect to the first gear DOFs.
        f1 : array-like
            Backlash force with respect to the first gear DOFs.
        f2 : array-like
            Backlash force with respect to the second gear DOFs.
        """
        # Original rigid approach (Discrete conditional)
        # if bt = 0
        if delta > bt:
            f_val = delta - bt
            f1_val = delta_dot - bt_dot
            sgn = 1.0
        elif delta < -bt:
            f_val = delta + bt
            f1_val = delta_dot + bt_dot
            sgn = -1.0
        else:
            return 0.0, 0.0, np.zeros(6), np.zeros(6)

        f1 = d_delta_d1
        f1[:2] -= sgn * bt_1

        f2 = d_delta_d2
        f2[:2] -= sgn * bt_2

        return f_val, f1_val, f1, f2

    def smooth_approach(
        self, delta, delta_dot, d_delta_d1, d_delta_d2, bt, bt_dot, bt_1, bt_2
    ):
        """Smooth approach for the backlash force model.

        Parameters
        ----------
        delta : float
            Transmission error.
        delta_dot : float
            Transmission error derivative.
        d_delta_d1 : array-like
            Derivative of the transmission error with respect to the first gear DOFs.
        d_delta_d2 : array-like
            Derivative of the transmission error with respect to the second gear DOFs.
        bt : float
            Backlash.
        bt_dot : float
            Backlash derivative.
        bt_1 : array-like
            Derivative of the backlash with respect to the first gear DOFs.
        bt_2 : array-like
            Derivative of the backlash with respect to the second gear DOFs.

        References
        ----------
        WALHA, L.; FAKHFAKH, T.; HADDAR, M. Backlash effect on dynamic analysis of
        a two-stage spur gear system. Journal of Failure Analysis and Prevention,
        ASM International, v. 6, p. 60-68, 2006.

        Returns
        -------
        f_val : float
            Backlash force.
        f1_val : float
            Backlash force derivative with respect to the first gear DOFs.
        f1 : array-like
            Backlash force with respect to the first gear DOFs.
        f2 : array-like
            Backlash force with respect to the second gear DOFs.
        """
        x1_val = delta - bt
        x2_val = delta + bt

        tanh_x1 = np.tanh(self.sigma * x1_val)
        tanh_x2 = np.tanh(self.sigma * x2_val)

        g1 = x1_val * tanh_x1
        g2 = x2_val * tanh_x2

        f_val = delta + 0.5 * (g1 - g2)

        gp1 = tanh_x1 + self.sigma * x1_val * (1.0 - tanh_x1**2)
        gp2 = tanh_x2 + self.sigma * x2_val * (1.0 - tanh_x2**2)

        df_ddelta = 1.0 + 0.5 * (gp1 - gp2)
        df_dbt = 0.5 * (-gp1 - gp2)

        f1_val = delta_dot * df_ddelta + bt_dot * df_dbt

        f1 = d_delta_d1 * df_ddelta
        f1[:2] += bt_1 * df_dbt

        f2 = d_delta_d2 * df_ddelta
        f2[:2] += bt_2 * df_dbt

        return f_val, f1_val, f1, f2

    def interpolate2d_stiffness(self, angular_position, contact_ratio):
        """Interpolates the mesh stiffness value at a given angular position
        and contact ratio.

        Parameters
        ----------
        angular_position : float
            Angular position at which to evaluate the stiffness (rad).
        contact_ratio : float
            Contact ratio at which to evaluate the stiffness.

        Returns
        -------
        stiffness : float or np.ndarray
            Interpolated stiffness value(s) in N/m.
        """
        theta = angular_position % max(self.theta_range)
        cr = contact_ratio

        i = np.searchsorted(self.theta_range, theta) - 1
        j = np.searchsorted(self.contact_ratio_range, cr) - 1

        # Prevent "Index Out of Bounds"
        i = np.clip(i, 0, len(self.theta_range) - 2)
        j = np.clip(j, 0, len(self.contact_ratio_range) - 2)

        t1, t2 = self.theta_range[i : i + 2]
        c1, c2 = self.contact_ratio_range[j : j + 2]

        wt = (theta - t1) / (t2 - t1) if t2 != t1 else 0.0
        wc = (cr - c1) / (c2 - c1) if c2 != c1 else 0.0

        k0, k1 = (
            self.stiffness_table[i, j : j + 2] * (1 - wt)
            + self.stiffness_table[i + 1, j : j + 2] * wt
        )
        stiffness = k0 * (1 - wc) + k1 * wc

        return stiffness

    def _save_time_results(self, step, results):
        """Save time results in data at each step.

        Parameters
        ----------
        step : int
            Step number.
        results : dict
            Results to save.
        """
        for key, value in results.items():
            lst = self._data[key]
            if step < len(lst):
                lst[step] = value
            else:
                lst.append(value)
