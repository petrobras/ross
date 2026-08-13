"""Mesh module.

This module defines the Mesh and Backlash classes, which are used to model
the meshing behavior (stiffness, contact ratio and, optionally, backlash)
between the two gears coupled in a MultiRotor.
"""

import math
import numpy as np
from plotly import graph_objects as go
from warnings import warn
from numba import njit

from ross.units import Q_

from .gear_element import GearElementTVMS
from .utils import involute, mod, interpolate2d, compute_contact_ratio


__all__ = ["Mesh"]


class Mesh:
    """A class representing the meshing behavior between two gears in
    contact, including stiffness and contact ratio calculations.

    Parameters
    ----------
    driving_gear : GearElement
        The driving gear object used in the gear pair.
    driven_gear : GearElement
        The driven gear object used in the gear pair.
    gear_mesh_stiffness : float, optional
        Directly specify the stiffness of the gear mesh.
        If not provided, it can be calculated automatically
        when using `GearElementTVMS` instead of `GearElement`.
        Default is None.
    square_varying_stiffness : dict, optional
        Dictionary to enable and configure a square-shaped time-varying
        mesh stiffness. Keys are:

        - enable : bool
            If True, a square-shaped time-varying mesh stiffness is used.
            Default is False.
        - amplitude_ratio : float
            Ratio of the stiffness amplitude based on the mean value of the
            mesh stiffness.

        Default is `{"enable": False, "amplitude_ratio": 0}`.
    backlash : dict, optional
        Dictionary to enable and configure the backlash model between the
        coupled gears. Keys are:

        - enable : bool
            If True, the backlash model is used. Default is False.
        - initial_value : float
            Initial backlash of the gear pair (m). Default is 0.0.
        - error_amp : float
            Error amplitude used in the backlash force model. Default is 0.0.
        - smooth_operator : bool
            If True, a smooth (hyperbolic tangent) approximation is used for
            the backlash force. Default is False.
        - sigma : float
            Parameter related to the regularization of the smooth approach.
            Default is 1e4.

        Default is `{"enable": False, "initial_value": 0.0, "error_amp": 0.0,
        "smooth_operator": False, "sigma": 1e4}`.
    damping_ratio : float, optional
        Damping ratio used to compute the mesh damping when the backlash
        model is enabled. Default is 0.07.
    orientation_angle : float, pint.Quantity, optional
        The angle between the line of gear centers and x-axis. Default is 0.0 rad.

    Attributes
    ----------
    driving_gear : GearElement
        The driving_gear object, which contains information about the
        geometry and properties of the driving gear.
    driven_gear : GearElement
        The driven gear object, which contains information about the
        geometry and properties of the wheel gear.
    gear_ratio : float
        The transmission ratio, defined as the ratio of the number of teeth
        between the driving and driven gears.
    pressure_angle : float
        The pressure angle of the gear mesh (rad).
    contact_ratio : float
        The contact ratio of the gear pair.
    stiffness : float
        The (constant or mean) mesh stiffness of the gear pair (N/m).
    backlash : Backlash or None
        The backlash model of the gear pair, if enabled. None otherwise.

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
    >>> mesh.stiffness # doctest: +ELLIPSIS
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

        self.Ksq_ratio = square_varying_stiffness["amplitude_ratio"]
        self.orientation_angle = orientation_angle
        self.module = driving_gear.module
        self.damping_ratio = damping_ratio
        self.contact_ratio = self.compute_contact_ratio()

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

        self.stiffness_type = stiffness_type

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

    def compute_contact_ratio(self):
        """Calculate the contact ratio of the gear pair.

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

        contact_ratio = compute_contact_ratio(
            center_distance,
            self.pressure_angle,
            self.pressure_angle,
            Ra1,
            Ra2,
            Rb1,
            Rb2,
            self.module,
        )

        return contact_ratio

    def _angular_equivalent_stiffness(self, d_alpha):
        """Calculate the angular equivalent stiffness of a gear pair.

        Parameters
        ----------
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
        Ka = Kg * self.Ksq_ratio
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
        """Compute the mesh stiffness profile over a specified number of gear
        mesh periods.

        Parameters
        ----------
        stiffness_type : str, optional
            Type of stiffness to compute. Available options are:
            - "square": square varying stiffness
            - "equivalent": variable equivalent stiffness
            otherwise, a constant stiffness is used.
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
        """Interpolate the mesh stiffness value at a given angular position.

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

    def generate_stiffness_table(self, stiffness_type=None, n_points=200):
        """Generate a table of stiffness values for a gear pair.

        Parameters
        ----------
        stiffness_type : str, optional
            Type of stiffness to compute. Available options are:
            - "square": square varying stiffness
            - "equivalent": variable equivalent stiffness
            Default is None, which uses the stiffness type defined in the Mesh object.
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

        if stiffness_type is None:
            stiffness_type = self.stiffness_type

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
        stiffness_type=None,
        **kwargs,
    ):
        """Plot the gear mesh stiffness profile over one or more meshing periods.

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
        stiffness_type : str, optional
            Type of stiffness to compute. Available options are:
            - "constant": constant stiffness
            - "square": square varying stiffness
            - "equivalent": variable equivalent stiffness
            Default is None, which uses the stiffness type defined in the Mesh object.
        **kwargs : dict, optional
            Additional keyword arguments passed to `plotly.graph_objects.Figure.update_layout`
            for customizing the figure (e.g., title, font, size, legend settings, etc.).

        Returns
        -------
        fig : go.Figure
            Figure object.
        """
        fig = go.Figure()

        if stiffness_type is None:
            stiffness_type = self.stiffness_type

        if n_mesh_period != 1 or n_points != 1000:
            theta_range, stiffness_range = self.get_stiffness_for_mesh_period(
                stiffness_type, n_mesh_period, n_points
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

    The implementation constitutes a core part of the work by Sousa (2026). It adapts the
    model from Yi et al. (2019) and extends its mathematical foundation using the equations
    established by Kubur et al. (2004) and Mo et al. (2025).

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
    SOUSA, M. A. B., Implementation of a dynamic backlash model in gears for rotational dynamics
    analysis. 2026. 92 p. Master Dissertation, Federal University of Uberlândia, Uberlândia.

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
        self.smooth_operator = smooth_operator

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

        # Pre-allocated arrays for the backlash force model
        self._d_delta_d1 = np.zeros(6)
        self._d_delta_d2 = np.zeros(6)
        self._f1 = np.zeros(6)
        self._f2 = np.zeros(6)

        data_keys = [
            "transmission_error",
            "backlash",
            "mesh_force",
            "mesh_stiffness",
            "center_distance",
            "pressure_angle",
            "contact_ratio",
        ]
        self._data = {key: list() for key in data_keys}

    def interpolate_stiffness(self, angular_position, contact_ratio):
        """Interpolate the mesh stiffness value at a given angular position
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

        return interpolate2d(
            theta,
            contact_ratio,
            self.theta_range,
            self.contact_ratio_range,
            self.stiffness_table,
        )

    def compute_force(self, step, disp_resp, velc_resp, time, angular_pos, speed):
        """Calculate the backlash force to be used in time response integration
        with the Newmark method.

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
        dofs1 = self.driving_gear_dofs
        dofs2 = self.driven_gear_dofs

        disp1 = disp_resp[dofs1]
        velc1 = velc_resp[dofs1]
        disp2 = disp_resp[dofs2]
        velc2 = velc_resp[dofs2]

        Fm, delta, bt, k_m, d_inst, alpha, contact_ratio = _compute_backlash_force(
            disp1,
            velc1,
            disp2,
            velc2,
            angular_pos,
            speed,
            self.pressure_angle,
            self.orientation_angle,
            self.helix_angle,
            self.n_teeth,
            self.driving_gear_pitch_radius,
            self.driven_gear_pitch_radius,
            self.driving_gear_base_radius,
            self.driven_gear_base_radius,
            self.driving_gear_addendum_radius,
            self.driven_gear_addendum_radius,
            self.damping_ratio,
            self.module,
            self.M_eq,
            self.initial_value,
            self.error_amp,
            self.smooth_operator,
            self.sigma,
            self.theta_range,
            self.contact_ratio_range,
            self.stiffness_table,
            self._d_delta_d1,
            self._d_delta_d2,
            self._f1,
            self._f2,
        )

        # Force decomposition: Q_i = -Fm * f_(q_i)
        backlash_force = np.zeros(len(disp_resp))
        backlash_force[dofs1] = -Fm * self._f1
        backlash_force[dofs2] = -Fm * self._f2

        results = {
            "transmission_error": delta,
            "backlash": bt,
            "mesh_force": Fm,
            "mesh_stiffness": k_m,
            "center_distance": d_inst,
            "pressure_angle": alpha,
            "contact_ratio": contact_ratio,
        }
        self._save_time_results(step, results)

        return backlash_force

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


@njit(fastmath=True)
def _rigid_approach(
    delta, delta_dot, d_delta_d1, d_delta_d2, bt, bt_dot, bt_1, bt_2, f1, f2
):
    """Compute the backlash force using a rigid (non-smooth) approach.

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
    f1 : array-like
        Pre-allocated array that is updated in place with the backlash force
        derivative with respect to the first gear DOFs.
    f2 : array-like
        Pre-allocated array that is updated in place with the backlash force
        derivative with respect to the second gear DOFs.

    Returns
    -------
    f_val : float
        Backlash force.
    f1_val : float
        Backlash force derivative with respect to time.
    """
    if delta > bt:
        f_val = delta - bt
        f1_val = delta_dot - bt_dot
        sgn = 1.0
    elif delta < -bt:
        f_val = delta + bt
        f1_val = delta_dot + bt_dot
        sgn = -1.0
    else:
        f1[:] = 0.0
        f2[:] = 0.0
        return 0.0, 0.0

    f1[:] = d_delta_d1[:]
    f1[:2] -= sgn * bt_1

    f2[:] = d_delta_d2[:]
    f2[:2] -= sgn * bt_2

    return f_val, f1_val


@njit(fastmath=True)
def _smooth_approach(
    delta, delta_dot, d_delta_d1, d_delta_d2, bt, bt_dot, bt_1, bt_2, sigma, f1, f2
):
    """Compute the backlash force using a smooth (hyperbolic tangent) approach.

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
    sigma : float
        Parameter related to the regularization of the smooth approach.
    f1 : array-like
        Pre-allocated array that is updated in place with the backlash force
        derivative with respect to the first gear DOFs.
    f2 : array-like
        Pre-allocated array that is updated in place with the backlash force
        derivative with respect to the second gear DOFs.

    Returns
    -------
    f_val : float
        Backlash force.
    f1_val : float
        Backlash force derivative with respect to time.

    References
    ----------
    Walh, L., Fakhfakh, T., & Haddar, M. (2006). Backlash effect on dynamic analysis of
    a two-stage spur gear system. Journal of Failure Analysis and Prevention, 6, 60-68.
    """
    x1_val = delta - bt
    x2_val = delta + bt

    tanh_x1 = np.tanh(sigma * x1_val)
    tanh_x2 = np.tanh(sigma * x2_val)

    g1 = x1_val * tanh_x1
    g2 = x2_val * tanh_x2

    f_val = delta + 0.5 * (g1 - g2)

    gp1 = tanh_x1 + sigma * x1_val * (1.0 - tanh_x1**2)
    gp2 = tanh_x2 + sigma * x2_val * (1.0 - tanh_x2**2)

    df_ddelta = 1.0 + 0.5 * (gp1 - gp2)
    df_dbt = 0.5 * (-gp1 - gp2)

    f1_val = delta_dot * df_ddelta + bt_dot * df_dbt

    f1[:] = d_delta_d1 * df_ddelta
    f1[:2] += bt_1 * df_dbt

    f2[:] = d_delta_d2 * df_ddelta
    f2[:2] += bt_2 * df_dbt

    return f_val, f1_val


@njit(fastmath=True)
def _compute_backlash_force(
    disp1,
    velc1,
    disp2,
    velc2,
    angular_pos,
    speed,
    alpha_0,
    orientation_angle,
    helix_angle,
    n_teeth,
    Rp1,
    Rp2,
    Rb1,
    Rb2,
    Ra1,
    Ra2,
    damping_ratio,
    module,
    M_eq,
    b0,
    error_amp,
    smooth_operator,
    sigma,
    theta_range,
    contact_ratio_range,
    stiffness_table,
    d_delta_d1,
    d_delta_d2,
    f1,
    f2,
):
    """Compute the backlash force.

    Parameters
    ----------
    disp1 : array-like
        Displacement of the first gear.
    velc1 : array-like
        Velocity of the first gear.
    disp2 : array-like
        Displacement of the second gear.
    velc2 : array-like
        Velocity of the second gear.
    angular_pos : float
        Angular position of the gear pair.
    speed : float
        Speed of the gear pair.
    alpha_0 : float
        Nominal pressure angle.
    orientation_angle : float
        Orientation angle of the gear pair.
    helix_angle : float
        Helix angle of the gear pair.
    n_teeth : int
        Number of teeth of the gears.
    Rp1 : float
        Pitch radius of the first gear.
    Rp2 : float
        Pitch radius of the second gear.
    Rb1 : float
        Base radius of the first gear.
    Rb2 : float
        Base radius of the second gear.
    Ra1 : float
        Addendum radius of the first gear.
    Ra2 : float
        Addendum radius of the second gear.
    damping_ratio : float
        Damping ratio of the gear pair.
    module : float
        Module of the gears.
    M_eq : float
        Equivalent mass of the gear pair.
    b0 : float
        Initial backlash.
    error_amp : float
        Error amplitude.
    smooth_operator : bool
        Whether to use a smooth operator.
    sigma : float
        Sigma of the smooth operator.
    theta_range : array-like
        Theta range of the stiffness table.
    contact_ratio_range : array-like
        Contact ratio range of the stiffness table.
    stiffness_table : array-like
        Stiffness table.
    d_delta_d1 : array-like
        Pre-allocated array that is updated in place with the derivative of
        the transmission error with respect to the first gear DOFs.
    d_delta_d2 : array-like
        Pre-allocated array that is updated in place with the derivative of
        the transmission error with respect to the second gear DOFs.
    f1 : array-like
        Pre-allocated array that is updated in place with the backlash force
        derivative with respect to the first gear DOFs.
    f2 : array-like
        Pre-allocated array that is updated in place with the backlash force
        derivative with respect to the second gear DOFs.

    Returns
    -------
    Fm : float
        Backlash force.
    delta : float
        Transmission error.
    bt : float
        Backlash.
    k_m : float
        Stiffness of the gear pair.
    d_inst : float
        Instantaneous center distance.
    alpha : float
        Pressure angle.
    contact_ratio : float
        Contact ratio of the gear pair.
    """

    x1, y1, z1, rx1, ry1, t1 = disp1
    vx1, vy1, vz1, vrx1, vry1, vt1 = velc1
    x2, y2, z2, rx2, ry2, t2 = disp2
    vx2, vy2, vz2, vrx2, vry2, vt2 = velc2

    error = error_amp * np.sin(n_teeth * angular_pos)
    error_dot = error_amp * (n_teeth * speed) * np.cos(n_teeth * angular_pos)

    # Calculation of the non-linear kinematics in the transverse plane
    d0 = Rp1 + Rp2
    x2_abs = x2 + d0 * np.cos(orientation_angle)
    y2_abs = y2 + d0 * np.sin(orientation_angle)

    dx = x2_abs - x1
    dy = y2_abs - y1
    beta = np.arctan2(dy, dx)

    d_inst = max(np.sqrt(dx**2 + dy**2), 1e-12)

    cos_alpha_val = min(max((Rb1 + Rb2) / d_inst, -1.0), 1.0)
    alpha = np.arccos(cos_alpha_val)

    psi = alpha - beta
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    sin_hlx = np.sin(helix_angle)
    cos_hlx = np.cos(helix_angle)

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
        ((x1 - x2) * sin_psi + (y1 - y2) * cos_psi + Rb1 * t1 + Rb2 * t2) * cos_hlx
        + (
            (-z1 + z2)
            + (Rb1 * rx1 + Rb2 * rx2) * sin_psi
            + (Rb1 * ry1 + Rb2 * ry2) * cos_psi
        )
        * sin_hlx
        - error
    )

    # Dynamic backlash calculation (bt)
    delta_b = (Rb1 + Rb2) * (involute(alpha) - involute(alpha_0))
    bt = b0 + delta_b * cos_hlx

    # Derivatives of delta with respect to the DOFs
    geo_tr = (x1 - x2) * cos_psi - (y1 - y2) * sin_psi
    geo_rt = (Rb1 * rx1 + Rb2 * rx2) * cos_psi - (Rb1 * ry1 + Rb2 * ry2) * sin_psi

    psi_1 = alpha_1 - beta_1
    psi_2 = alpha_2 - beta_2

    tri_psi = np.array([sin_psi, cos_psi])
    d_delta_d1[0:2] = (tri_psi + geo_tr * psi_1) * cos_hlx + (geo_rt * psi_1) * sin_hlx
    d_delta_d1[2] = -sin_hlx
    d_delta_d1[3:5] = Rb1 * tri_psi * sin_hlx
    d_delta_d1[5] = Rb1 * cos_hlx

    d_delta_d2[0:2] = (-tri_psi + geo_tr * psi_2) * cos_hlx + (geo_rt * psi_2) * sin_hlx
    d_delta_d2[2] = sin_hlx
    d_delta_d2[3:5] = Rb2 * tri_psi * sin_hlx
    d_delta_d2[5] = Rb2 * cos_hlx

    # Derivatives of backlash with respect to the DOFs
    tan2_alpha = np.tan(alpha) ** 2
    bt_1 = (Rb1 + Rb2) * tan2_alpha * alpha_1 * cos_hlx
    bt_2 = (Rb1 + Rb2) * tan2_alpha * alpha_2 * cos_hlx

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
    bt_dot = (Rb1 + Rb2) * tan2_alpha * alpha_dot * cos_hlx

    # Penalty function application
    if smooth_operator:
        f_val, f1_val = _smooth_approach(
            delta,
            delta_dot,
            d_delta_d1,
            d_delta_d2,
            bt,
            bt_dot,
            bt_1,
            bt_2,
            sigma,
            f1,
            f2,
        )
    else:
        f_val, f1_val = _rigid_approach(
            delta, delta_dot, d_delta_d1, d_delta_d2, bt, bt_dot, bt_1, bt_2, f1, f2
        )

    contact_ratio = compute_contact_ratio(
        d_inst, alpha, alpha_0, Ra1, Ra2, Rb1, Rb2, module
    )

    theta = angular_pos % max(theta_range)
    k_m = interpolate2d(
        theta, contact_ratio, theta_range, contact_ratio_range, stiffness_table
    )

    c_m = 2.0 * damping_ratio * np.sqrt(k_m * M_eq)

    # Total normal force in the action line
    Fm = k_m * f_val + c_m * f1_val

    return Fm, delta, bt, k_m, d_inst, alpha, contact_ratio
