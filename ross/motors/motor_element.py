"""Electric Motor Element module.

This module defines the MotorElement class which represents a 3-phase Induction Electric Motor
simulated using a 4th-order Runge-Kutta method, considering magnetic fluxes and currents.
"""

import base64
from pathlib import Path

import numpy as np
from numba import njit
from inspect import signature
from plotly import graph_objects as go

from ross.element import Element
from ross.units import Q_, check_units

from .sources import SourceAC
from .inverters import InverterVF
from .results import MotorResponseResults
from .utils import phase_to_line, clarke_transform, park_transform, rk4_step

__all__ = ["MotorElement", "motor_example"]

_MOTOR_SVG_PATH = Path(__file__).parent / "electric_motor.svg"


class MotorElement(Element):
    """Create a 3-phase induction motor element for rotordynamic analysis.

    This class represents a Three-Phase Induction Motor (TPIM) using a synchronous
    reference frame for rotor flux orientation.

    Parameters
    ----------
    n : int
        Node at which the motor is coupled.
    power_nom : float, pint.Quantity
        Nominal power [W].
    voltage_nom : float
        Nominal voltage [V].
    speed_nom : float, pint.Quantity
        Nominal machine speed [rad/s].
    stator_resistance : float
        Stator resistance [Ohm].
    rotor_resistance : float
        Rotor resistance [Ohm].
    stator_reactance : float
        Stator self-reactance at frequency [Ohm].
    rotor_reactance : float
        Rotor self-reactance at frequency [Ohm].
    mutual_reactance : float
        Mutual reactance at frequency [Ohm].
    Ip_motor : float, pint.Quantity
        Polar Moment of inertia related to motor axis [kg.m²].
    frequency_nom : float, optional, pint.Quantity
        Nominal frequency [rad/s].
        Default is 60 Hz.
    n_poles: int, optional
        Number of machine's poles.
        Default is 4.
    viscosity_coeff : float, optional, pint.Quantity
        Viscosity coefficient [Pa.s].
        Default is 0.
    Ip_load : float, optional, pint.Quantity
        Polar Moment of inertia related to load [kg.m²].
        Default is 0.
    voltage_net : float, optional
        Electrical tension of Power Supply [V].
        Default is None, which adopts the motor's nominal voltage (`voltage`).
    frequency_net : float, optional, pint.Quantity
        Electrical frequency of Power Supply [rad/s].
        Default is None, which adopts the motor's nominal frequency (`frequency`).
    initial_angle_net : float, optional, pint.Quantity
        Initial angular phase frequency of Power Supply [rad].
        Default is 20 degrees.
    short_circuit_ratio_net : float, optional
        Short-Circuit Ratio at common coupling point with power supply.
        Default is 50.
    XR_ratio_net : float, optional
        Reactance (X) / Resistance (R) Ratio at coupling point with power supply.
        Default is 80.
    tag : str, optional
        Tag to name the element.
        Default is None.
    scale_factor : float, optional
        Scale factor used to draw the motor in the rotor plot.
        Default is 1.
    color : str, optional
        Color used when the element is represented in the rotor plot.
        It needs to be in hex format. Default is "#4682c9".

    Examples
    --------
    >>> from ross import MotorElement
    >>> from ross.units import Q_

    >>> motor = MotorElement(
    ...     n=0,
    ...     tag=None,
    ...     power_nom=Q_(1.5, "cv"),
    ...     voltage_nom=127,
    ...     speed_nom=Q_(1725, "RPM"),
    ...     frequency_nom=Q_(60.0, "Hz"),
    ...     n_poles=4,
    ...     stator_resistance=2.5,
    ...     rotor_resistance=1.8,
    ...     stator_reactance=1.3,
    ...     rotor_reactance=1.3,
    ...     mutual_reactance=43.08,
    ...     Ip_motor=0.0372,
    ...     viscosity_coeff=0.0,
    ...     Ip_load=0.0,
    ...     voltage_net=127,
    ...     frequency_net=Q_(60.0, "Hz"),
    ... )

    >>> motor.power_nom
    1103.248125
    """

    @check_units
    def __init__(
        self,
        n,
        power_nom,
        voltage_nom,
        speed_nom,
        stator_resistance,
        rotor_resistance,
        stator_reactance,
        rotor_reactance,
        mutual_reactance,
        Ip_motor,
        frequency_nom=None,
        n_poles=4,
        viscosity_coeff=0.0,
        Ip_load=0.0,
        voltage_net=None,
        frequency_net=None,
        initial_angle_net=None,
        short_circuit_ratio_net=50.0,
        XR_ratio_net=80.0,
        tag=None,
        scale_factor=1,
        color="#4682c9",
    ):

        # Numerical Validation of NOMP entries
        self.power_nom = float(power_nom)
        self.voltage_nom = float(voltage_nom)
        self.speed_nom = float(speed_nom)
        self.n_poles = int(n_poles)
        self.stator_resistance = float(stator_resistance)
        self.rotor_resistance = float(rotor_resistance)
        self.stator_reactance = float(stator_reactance)
        self.rotor_reactance = float(rotor_reactance)
        self.mutual_reactance = float(mutual_reactance)
        self.Ip_motor = float(Ip_motor)
        self.viscosity_coeff = float(viscosity_coeff)
        self.Ip_load = float(Ip_load)

        if frequency_nom is None:
            self.frequency_nom = Q_(60.0, "Hz").to("rad/s").m
        else:
            self.frequency_nom = float(frequency_nom)

        self.frequency_ref = self.frequency_nom

        # Numerical Validation of SCIP entries
        self.short_circuit_ratio_net = float(short_circuit_ratio_net)
        self.XR_ratio_net = float(XR_ratio_net)

        if voltage_net is None:
            self.voltage_net = self.voltage_nom
        else:
            self.voltage_net = float(voltage_net)

        if frequency_net is None:
            self.frequency_net = self.frequency_nom
        else:
            self.frequency_net = float(frequency_net)

        if initial_angle_net is None:
            self.initial_angle_net = Q_(20.0, "deg").to("rad").m
        else:
            self.initial_angle_net = float(initial_angle_net)

        # Internal model inductances parameters derived from CEMP
        Lls = self.stator_reactance / self.frequency_nom
        Llr = self.rotor_reactance / self.frequency_nom
        self.Lm = self.mutual_reactance / self.frequency_nom
        self.Lss = Lls + self.Lm
        self.Lrr = Llr + self.Lm

        # Internal Electric Motor constants derived from NOMP and CEMP
        snom = (1 - self.speed_nom * self.n_poles / (2 * self.frequency_nom)) * 100
        wnom = (self.frequency_nom * (1 - snom / 100)) / (self.n_poles / 2)
        sigma = 1 - self.Lm**2 / (self.Lss * self.Lrr)
        self.Tnom = self.power_nom / wnom
        self.a = 1 / (sigma * self.Lss)
        self.b = 1 / (sigma * self.Lrr)
        self.c = self.Lm / (sigma * self.Lss * self.Lrr)

        # Short-Circuit Power and Impedances parameters derived from SCIP
        SCC_net = self.short_circuit_ratio_net * self.power_nom
        Zsc = self.voltage_net**2 / SCC_net
        Xsc = Zsc * self.XR_ratio_net / np.sqrt(1 + self.XR_ratio_net**2)
        self.short_circuit_resistance = Xsc / self.XR_ratio_net

        self.n = n
        self.n_l = n
        self.n_r = n
        self.tag = tag
        self.scale_factor = scale_factor
        self.color = color
        self.dof_global_index = None

    def __repr__(self):
        """Return a string representation of a motor element.

        Returns
        -------
        str
            A string representation of a motor element object.

        Examples
        --------
        >>> motor = motor_example()
        >>> motor # doctest: +ELLIPSIS
        MotorElement(n=0, tag=None...
        """
        return (
            f"{self.__class__.__name__}"
            f"(n={self.n}, tag={self.tag!r}, "
            f"power_nom={self.power_nom:{0}.{5}}, "
            f"voltage_nom={self.voltage_nom:{0}.{5}}, "
            f"speed_nom={self.speed_nom:{0}.{5}}, "
            f"frequency_nom={self.frequency_nom:{0}.{5}}, "
            f"n_poles={self.n_poles}, "
            f"stator_resistance={self.stator_resistance:{0}.{5}}, "
            f"rotor_resistance={self.rotor_resistance:{0}.{5}}, "
            f"stator_reactance={self.stator_reactance:{0}.{5}}, "
            f"rotor_reactance={self.rotor_reactance:{0}.{5}}, "
            f"mutual_reactance={self.mutual_reactance:{0}.{5}}, "
            f"Ip_motor={self.Ip_motor:{0}.{5}}, "
            f"viscosity_coeff={self.viscosity_coeff:{0}.{5}}, "
            f"Ip_load={self.Ip_load:{0}.{5}}, "
            f"voltage_net={self.voltage_net:{0}.{5}}, "
            f"frequency_net={self.frequency_net:{0}.{5}}, "
            f"initial_angle_net={self.initial_angle_net:{0}.{5}}, "
            f"short_circuit_ratio_net={self.short_circuit_ratio_net:{0}.{5}}, "
            f"XR_ratio_net={self.XR_ratio_net:{0}.{5}}, "
            f"scale_factor={self.scale_factor}, "
            f"color={self.color!r})"
        )

    def __str__(self):
        """Convert object into string.

        Returns
        -------
        str
            The object's parameters translated to strings.

        Examples
        --------
        >>> print(motor_example())  # doctest: +ELLIPSIS
        Tag:                                None
        Node:                               0
        -------- Nominal Parameters (NOMP) -------
        Nominal Power (W):                  1103.2
        Nominal Voltage (V):                127.0
        Nominal Rotation (rad/s):           179.07
        Nominal Frequency (Hz):             60.0
        Number of Poles:                    4
        ...
        """
        return (
            f"Tag:                                {self.tag}"
            f"\nNode:                               {self.n}"
            f"\n-------- Nominal Parameters (NOMP) -------"
            f"\nNominal Power (W):                  {self.power_nom:{2}.{5}}"
            f"\nNominal Voltage (V):                {self.voltage_nom:{2}.{5}}"
            f"\nNominal Rotation (rad/s):           {self.speed_nom:{2}.{5}}"
            f"\nNominal Frequency (Hz):             {Q_(self.frequency_nom, 'rad/s').to('Hz').m:{2}.{5}}"
            f"\nNumber of Poles:                    {self.n_poles}"
            f"\n-------- Circuit Parameters (CEMP) -------"
            f"\nStator Resistance (Ohm):            {self.stator_resistance:{2}.{5}}"
            f"\nRotor Resistance (Ohm):             {self.rotor_resistance:{2}.{5}}"
            f"\nStator Reactance (Ohm):             {self.stator_reactance:{2}.{5}}"
            f"\nRotor Reactance (Ohm):              {self.rotor_reactance:{2}.{5}}"
            f"\nMutual Reactance (Ohm):             {self.mutual_reactance:{2}.{5}}"
            f"\nMotor Inertia (kg.m²):              {self.Ip_motor:{2}.{5}}"
            f"\nViscosity Coefficient (Pa.s):       {self.viscosity_coeff:{2}.{5}}"
            f"\nLoad Inertia (kg.m²):               {self.Ip_load:{2}.{5}}"
            f"\n----- Power Supply Parameters (SCIP) -----"
            f"\nSupply Voltage (V):                 {self.voltage_net:{2}.{5}}"
            f"\nSupply Frequency (Hz):              {Q_(self.frequency_net, 'rad/s').to('Hz').m:{2}.{5}}"
            f"\nInitial Phase Angle (deg):          {Q_(self.initial_angle_net, 'rad').to('deg').m:{2}.{5}}"
            f"\nShort-Circuit Ratio (ad):           {self.short_circuit_ratio_net:{2}.{5}}"
            f"\nX/R Ratio (ad):                     {self.XR_ratio_net:{2}.{5}}"
        )

    def __eq__(self, other):
        """Equality method for comparisons.

        Node and tag are not considered when comparing motor elements.

        Parameters
        ----------
        other : object
            The second object to be compared with.

        Returns
        -------
        bool
            True if the comparison is true; False otherwise.

        Examples
        --------
        >>> motor1 = motor_example()
        >>> motor2 = motor_example()
        >>> motor1 == motor2
        True
        """
        if not isinstance(other, self.__class__):
            return False

        compared_attributes = set(signature(self.__init__).parameters) - {
            "self",
            "n",
            "tag",
        }
        compared_attributes = compared_attributes.intersection(self.__dict__.keys())

        for attr in compared_attributes:
            self_attr = getattr(self, attr)
            other_attr = getattr(other, attr)

            try:
                if not np.allclose(self_attr, other_attr):
                    return False
            except TypeError:
                if self_attr != other_attr:
                    return False

        return True

    def __hash__(self):
        """Return the hash value of the motor element.

        Returns
        -------
        int
            Hash value based on the element tag.
        """
        return hash(self.tag)

    def dof_mapping(self):
        pass

    def M(self):
        pass

    def K(self):
        pass

    def C(self):
        pass

    def G(self):
        pass

    def _hover_info(self):
        """Return hover information for the motor patch.

        Returns
        -------
        customdata : list
            Values displayed in the hover tooltip.
        hovertemplate : str
            Plotly hover template string.
        """
        frequency_hz = Q_(self.frequency_nom, "rad/s").to("Hz").m
        customdata = [
            self.n,
            self.power_nom,
            self.voltage_nom,
            self.speed_nom,
            frequency_hz,
            self.n_poles,
            self.Ip_motor,
        ]
        hovertemplate = (
            f"Motor Node: {customdata[0]}<br>"
            f"Nominal Power (W): {customdata[1]:.3f}<br>"
            f"Nominal Voltage (V): {customdata[2]:.3f}<br>"
            f"Nominal Speed (rad/s): {customdata[3]:.3f}<br>"
            f"Nominal Frequency (Hz): {customdata[4]:.3f}<br>"
            f"Number of Poles: {customdata[5]}<br>"
            f"Motor Inertia (kg.m²): {customdata[6]:.3e}<br>"
        )
        return customdata, hovertemplate

    def _patch(self, position, fig):
        """Motor element patch.

        Patch used to draw the motor element in the rotor plot using Plotly.
        The motor icon is rendered from ``electric_motor.svg``.

        Parameters
        ----------
        position : tuple
            Position ``(zpos, ypos, yc_pos, scale_factor, side)`` in which the
            patch will be drawn. ``side`` must be ``"left"`` or ``"right"`` and
            defines the direction in which the motor extends from the node.
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure object which traces are added on.
        """
        zpos, ypos, yc_pos, scale_factor, side = position

        yc_pos += 4e-3

        customdata, hovertemplate = self._hover_info()

        image_height = max(ypos * 8.0, scale_factor * 4.0)
        image_width = image_height * 0.9
        marker_size = image_height * 350

        svg = _MOTOR_SVG_PATH.read_text(encoding="utf-8")

        if side == "right":
            x_anchor = "left"
            z_hover = zpos + image_width / 2
        else:
            x_anchor = "right"
            z_hover = zpos - image_width / 2
            svg = svg.replace(
                "<svg ",
                '<svg transform="scale(-1,1)" ',
                1,
            )

        svg = svg.replace("#61809A", f"{self.color}")

        encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        src_svg = f"data:image/svg+xml;base64,{encoded}"

        fig.add_layout_image(
            dict(
                source=src_svg,
                xref="x",
                yref="y",
                x=zpos,
                y=yc_pos,
                sizex=image_width,
                sizey=image_height,
                xanchor=x_anchor,
                yanchor="middle",
                sizing="stretch",
                layer="below",
                opacity=0.8,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[z_hover],
                y=[yc_pos],
                mode="markers",
                marker=dict(size=marker_size, color=self.color, opacity=0),
                customdata=[customdata],
                hovertemplate=hovertemplate,
                hoverinfo="text",
                name=self.tag,
                legendgroup="motors",
                showlegend=False,
                hoverlabel=dict(bgcolor=self.color),
            )
        )

        return fig

    def calculate_electric_torque(self, Lds, Lqs, Ldr, Lqr):
        """Calculate the electric torque generated by the motor.

        Parameters
        ----------
        Lds : float
            d-axis stator inductance [H].
        Lqs : float
            q-axis stator inductance [H].
        Ldr : float
            d-axis rotor inductance [H].
        Lqr : float
            q-axis rotor inductance [H].

        Returns
        -------
        float
            Electric torque [N.m].
        """
        return calculate_electric_torque(Lds, Lqs, Ldr, Lqr, self.c, self.n_poles)

    def run(
        self,
        t,
        time_step=None,
        load_torque_entrance_time=None,
        load_torque_ratio=1.0,
        element=None,
    ):
        """Run motor generical simulation for specified time points.

        Parameters
        ----------
        t : array-like
            Time points to store the computed solution [s].
        time_step : float, optional
            Time step for the simulation [s].
            Default is None, which uses the minimum time step between the time points.
        load_torque_entrance_time : float, optional
            Time when load torque is applied to the motor shaft [s].
            Default is half the simulation time.
        load_torque_ratio : float, optional
            Load torque ratio applied at the entrance time. This is a multiplier
            for the nominal load torque, e.g., a value of 1.0 applies 100% of the
            nominal torque at entrance time. Default is 1.0.

        Returns
        -------
        results : ross.MotorResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.MotorResponseResults`
        """

        if element is None:
            raise ValueError(
                "An element providing the 'get_phase_voltages(t)' method must be provided for simulation."
            )

        # Initial values
        wr0 = 0.0  # Rotor's angular speed
        thetar0 = 0.0  # Rotor's angular position
        flux_angle_0 = self.initial_angle_net - np.pi / 2  # Flux initial angle

        # Initial alpha-beta and dq currents (based in nulled instantaneous phase currents)
        ias, ibs, ics = 0, 0, 0
        i_alpha, i_beta = clarke_transform(ias, ibs, ics)
        ids, iqs = park_transform(i_alpha, i_beta, flux_angle_0)
        idr, iqr = 0, 0

        # Initial rotor and stator's inductances
        Lds0 = self.Lss * ids + self.Lm * idr
        Lqs0 = self.Lss * iqs + self.Lm * iqr
        Ldr0 = self.Lrr * idr + self.Lm * ids
        Lqr0 = self.Lrr * iqr + self.Lm * iqs

        # Get solution in time
        t_eval = np.array(t)

        n_sub = 1
        dt = t_eval[1] - t_eval[0]

        if time_step is not None and time_step < dt:
            n_sub = int(np.round(dt / time_step))
            dt = dt / n_sub

        nt = int((t_eval[-1] - t_eval[0]) / dt) + 1
        t_simul = np.linspace(t_eval[0], t_eval[-1], nt)

        # Load torque
        if load_torque_entrance_time is None:
            load_torque_entrance_time = t_simul[nt // 2]

        Tl_full = self.Tnom * load_torque_ratio

        w_shaft, vas, vbs, vcs = np.vectorize(element.get_operating_state)(t_simul)

        Rs = self.stator_resistance + self.short_circuit_resistance
        Ip = self.Ip_motor + self.Ip_load

        outputs = run_motor_time_loop(
            t_simul,
            dt,
            flux_angle_0,
            Lds0,
            Lqs0,
            Ldr0,
            Lqr0,
            wr0,
            load_torque_entrance_time,
            Tl_full,
            w_shaft,
            vas,
            vbs,
            vcs,
            self.a,
            self.b,
            self.c,
            self.n_poles,
            Rs,
            self.rotor_resistance,
            self.viscosity_coeff,
            Ip,
        )

        # Compute outputs
        speed, Lds, Lqs, Ldr, Lqr, load_torque, flux_angle, electric_torque = outputs

        angular_position = thetar0 + (speed * self.n_poles / 2) * t_simul

        voltages = dict()
        voltages["a"] = vas
        voltages["b"] = vbs
        voltages["c"] = vcs

        ids = self.a * Lds - self.c * Ldr
        iqs = self.a * Lqs - self.c * Lqr
        i_alpha = ids * np.cos(flux_angle) - iqs * np.sin(flux_angle)
        i_beta = ids * np.sin(flux_angle) + iqs * np.cos(flux_angle)
        ias = i_alpha
        ibs = -i_alpha / 2 + np.sqrt(3) * i_beta / 2
        ics = -i_alpha / 2 - np.sqrt(3) * i_beta / 2

        currents = dict()
        currents["a"] = ias
        currents["b"] = ibs
        currents["c"] = ics
        currents["alpha"] = i_alpha
        currents["beta"] = i_beta
        currents["d"] = ids
        currents["q"] = iqs

        results = MotorResponseResults(
            t_simul,
            electric_torque,
            load_torque,
            speed,
            currents,
            voltages,
            t_eval=t_eval,
        )

        return results

    @check_units
    def run_with_AC_source(
        self,
        t,
        time_step=None,
        load_torque_entrance_time=None,
        load_torque_ratio=1.0,
        voltage_net=None,
        frequency_net=None,
        initial_phase_angle=0.0,
        harmonics=None,
        unbalances=None,
    ):
        """Run motor simulation with AC source for specified time points.

        Parameters
        ----------
        t : array-like
            Time points to store the computed solution [s].
        time_step : float, optional
            Time step for the simulation [s].
            Default is None, which uses the minimum time step between the time points.
        load_torque_entrance_time : float, optional
            Time when load torque is applied to the motor shaft [s].
            Default is half the simulation time.
        load_torque_ratio : float, optional
            Load torque ratio applied at the entrance time. This is a multiplier
            for the nominal load torque, e.g., a value of 1.0 applies 100% of the
            nominal torque at entrance time. Default is 1.0.
        voltage_net : float, pint.Quantity, optional
            Power supply voltage [V]. If None, uses motor nominal voltage.
        frequency_net : float, pint.Quantity, optional
            Power supply frequency [rad/s]. If None, uses motor nominal frequency.
        initial_phase_angle : float, pint.Quantity, optional
            Initial power supply phase angle [rad]. Default is 0.
        harmonics : dict, optional
            Harmonic configuration with keys:

            - "orders" : list of int
                Harmonic orders (e.g., [5, 7])
            - "amplitudes" : list of float
                Harmonic amplitudes as % of nominal voltage
            - "enable" : bool
                Enable harmonics

        unbalances : dict, optional
            Unbalance configuration with keys:

            - "voltage_percent" : list of float
                Voltage magnitude unbalance per phase [%]
            - "angle_deviation" : list of float, pint.Quantity
                Angle deviation per phase [rad]
            - "enable" : bool
                Enable unbalances

        Returns
        -------
        results : ross.MotorResponseResults
            For more information on attributes and methods available see:
            :py:class:`ross.MotorResponseResults`

        Examples
        --------
        >>> from ross.units import Q_
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> t = np.arange(0, tf + dt, dt)

        >>> results = motor.run_with_AC_source(
        ...     t,
        ...     load_torque_entrance_time=3.0,
        ...     load_torque_ratio=1.5,
        ...     initial_phase_angle=0.0,
        ...     harmonics={
        ...         "enable": True,
        ...         "orders": [5, 7],
        ...         "amplitudes": [3, 2],
        ...     },
        ...     unbalances={
        ...         "enable": True,
        ...         "voltage_percent": [-1, 2, 3],
        ...         "angle_deviation": Q_([1, 0, -2], "deg"),
        ...     },
        ... )

        Time domain plots
        >>> fig1 = results.plot_torque()
        >>> fig2 = results.plot_speed()
        >>> fig3 = results.plot_phase_currents(reference_frame="a-b-c")
        >>> fig4 = results.plot_phase_voltages()

        Frequency domain plots
        >>> fig5 = results.plot_torque(domain="frequency")
        >>> fig6 = results.plot_speed(domain="frequency")
        >>> fig7 = results.plot_phase_currents(domain="frequency")
        >>> fig8 = results.plot_phase_voltages(domain="frequency")
        """
        # Creating AC source instance
        if harmonics:
            harmonic_orders = harmonics.get("orders")
            harmonic_amplitudes = harmonics.get("amplitudes")
        else:
            harmonic_orders, harmonic_amplitudes = None, None

        if unbalances:
            unbalance_voltage_percent = unbalances.get("voltage_percent")
            unbalance_angle_deviation = unbalances.get("angle_deviation")
        else:
            unbalance_voltage_percent, unbalance_angle_deviation = None, None

        source = SourceAC(
            voltage_net=voltage_net or self.voltage_nom,
            frequency_net=frequency_net or self.frequency_nom,
            initial_phase_angle=initial_phase_angle,
            harmonic_orders=harmonic_orders,
            harmonic_amplitudes=harmonic_amplitudes,
            harmonic_enable=harmonics.get("enable") if harmonics else False,
            unbalance_voltage_percent=unbalance_voltage_percent,
            unbalance_angle_deviation=unbalance_angle_deviation,
            unbalance_enable=unbalances.get("enable") if unbalances else False,
        )

        results = self.run(
            t,
            time_step=time_step,
            load_torque_entrance_time=load_torque_entrance_time,
            load_torque_ratio=load_torque_ratio,
            element=source,
        )

        return results

    @check_units
    def run_with_inverter(
        self,
        t,
        time_step=None,
        load_torque_entrance_time=None,
        load_torque_ratio=1.0,
        frequency_s=None,
        time_ramp=1.0,
        frequency_ref=None,
    ):
        """Simulate motor with variable frequency inverter control.

        Runs motor simulation with a three-phase voltage source inverter employing
        Space Vector PWM modulation and scalar V/f speed control. The inverter
        generates three-phase voltages based on the reference frequency.

        Parameters
        ----------
        t : array-like
            Time points to store the computed solution [s].
        time_step : float, optional
            Time step for the simulation [s].
            Default is None, which uses the minimum time step between the time points.
        load_torque_entrance_time : float, optional
            Time when load torque is applied to the motor shaft [s].
            Default is None (no load applied).
        load_torque_ratio : float, optional
            Load torque ratio applied at the entrance time. This is a multiplier
            for the nominal load torque, e.g., a value of 1 applies 100% of the
            nominal torque. Default is 1.
        frequency_s : float or pint.Quantity
            IGBT switching frequency [rad/s].
        time_ramp : float, optional
            Acceleration ramp time [s] for frequency ramping. Default is 1.
        frequency_ref : float or pint.Quantity, optional
            Reference frequency for V/f control [rad/s]. If None, uses half the
            motor nominal frequency. Default is None.

        Returns
        -------
        results : ross.MotorResponseResults
            Motor simulation results containing time-domain response data.
            For more information on attributes and methods available see:
            :py:class:`ross.MotorResponseResults`

        Examples
        --------
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> size = int(tf / dt) + 1
        >>> t = np.linspace(0, tf, size)

        >>> results = motor.run_with_inverter(
        ...     t, # Evaluation time vector
        ...     time_step=1e-4, # Simulation time step
        ...     load_torque_entrance_time=2.5,
        ...     load_torque_ratio=1.0,
        ...     frequency_s=Q_(5000, "Hz"),
        ...     time_ramp=0.6667,
        ...     frequency_ref=Q_(30.0, "Hz")
        ... )

        Time domain plots
        >>> fig1 = results.plot_torque()
        >>> fig2 = results.plot_speed()
        >>> fig3 = results.plot_phase_currents(reference_frame="a-b-c")
        >>> fig4 = results.plot_phase_voltages()

        Frequency domain plots
        >>> fig5 = results.plot_torque(domain="frequency")
        >>> fig6 = results.plot_speed(domain="frequency")
        >>> fig7 = results.plot_phase_currents(domain="frequency")
        >>> fig8 = results.plot_phase_voltages(domain="frequency")
        """

        Vnl = phase_to_line(self.voltage_nom)
        line_to_dc_bus = 1.35

        inverter = InverterVF(
            voltage_dc=line_to_dc_bus * Vnl,
            frequency_s=frequency_s,
            voltage_nom=Vnl,
            frequency_nom=self.frequency_nom,
            time_ramp=time_ramp,
            frequency_ref=frequency_ref or self.frequency_nom / 2,
        )

        results = self.run(
            t,
            time_step=time_step or inverter.Ts / 200,
            load_torque_entrance_time=load_torque_entrance_time,
            load_torque_ratio=load_torque_ratio,
            element=inverter,
        )

        return results


@njit
def calculate_electric_torque(Lds, Lqs, Ldr, Lqr, c, n_poles):
    """Compute electric torque from flux linkage."""
    return 1.5 * c * (Lqs * Ldr - Lds * Lqr) * n_poles / 2


@njit
def motor_ode_system(y, args):
    """State derivatives for the motor ODE system"""
    Lds, Lqs, Ldr, Lqr, wr = y
    (
        vds,
        vqs,
        vdr,
        vqr,
        w_shaft,
        Tl,
        c,
        n_poles,
        Rs,
        a,
        b,
        Rr,
        viscosity_coeff,
        Ip_total,
    ) = args

    Te = calculate_electric_torque(Lds, Lqs, Ldr, Lqr, c, n_poles)

    dLds_dt = vds - Rs * a * Lds + Rs * c * Ldr + w_shaft * Lqs
    dLqs_dt = vqs - Rs * a * Lqs + Rs * c * Lqr - w_shaft * Lds

    w = w_shaft - wr * n_poles / 2

    dLdr_dt = vdr - Rr * b * Ldr + Rr * c * Lds + w * Lqr
    dLqr_dt = vqr - Rr * b * Lqr + Rr * c * Lqs - w * Ldr

    dwr_dt = (Te - viscosity_coeff * wr - Tl) / Ip_total

    return np.array([dLds_dt, dLqs_dt, dLdr_dt, dLqr_dt, dwr_dt])


@njit
def run_motor_time_loop(
    t_simul,
    dt,
    flux_angle_0,
    Lds0,
    Lqs0,
    Ldr0,
    Lqr0,
    wr0,
    load_torque_entrance_time,
    Tl_full,
    w_shaft_arr,
    vas_arr,
    vbs_arr,
    vcs_arr,
    a,
    b,
    c,
    n_poles,
    Rs,
    Rr,
    viscosity_coeff,
    Ip,
):
    """Run the motor time-stepping loop."""
    nt = len(t_simul)
    speed = np.zeros(nt)
    Lds_hist = np.zeros(nt)
    Lqs_hist = np.zeros(nt)
    Ldr_hist = np.zeros(nt)
    Lqr_hist = np.zeros(nt)
    load_torque = np.zeros(nt)
    flux_angle = np.zeros(nt)
    electric_torque = np.zeros(nt)

    Lds, Lqs, Ldr, Lqr, wr = Lds0, Lqs0, Ldr0, Lqr0, wr0
    f_ang = flux_angle_0
    vdr, vqr = 0.0, 0.0
    load_applied = False
    Tl = 0.0

    for step in range(1, nt):
        t = t_simul[step]

        if not load_applied and t >= load_torque_entrance_time:
            load_applied = True
            Tl = Tl_full

        w_shaft = w_shaft_arr[step]
        vas = vas_arr[step]
        vbs = vbs_arr[step]
        vcs = vcs_arr[step]

        v_alpha, v_beta = clarke_transform(vas, vbs, vcs)

        f_ang += w_shaft * dt
        vds, vqs = park_transform(v_alpha, v_beta, f_ang)

        Lds, Lqs, Ldr, Lqr, wr = rk4_step(
            motor_ode_system,
            dt,
            [Lds, Lqs, Ldr, Lqr, wr],
            [
                vds,
                vqs,
                vdr,
                vqr,
                w_shaft,
                Tl,
                c,
                n_poles,
                Rs,
                a,
                b,
                Rr,
                viscosity_coeff,
                Ip,
            ],
        )

        speed[step] = wr
        Lds_hist[step] = Lds
        Lqs_hist[step] = Lqs
        Ldr_hist[step] = Ldr
        Lqr_hist[step] = Lqr
        load_torque[step] = Tl
        flux_angle[step] = f_ang
        electric_torque[step] = calculate_electric_torque(
            Lds, Lqs, Ldr, Lqr, c, n_poles
        )

    return (
        speed,
        Lds_hist,
        Lqs_hist,
        Ldr_hist,
        Lqr_hist,
        load_torque,
        flux_angle,
        electric_torque,
    )


def motor_example():
    """Create an example of motor element.

    This function returns an instance of a simple electric motor. The purpose is
    to make available a simple model so that doctest can be written using it.

    Returns
    -------
    motor : ross.MotorElement
        An instance of a motor object.

    Examples
    --------
    >>> motor = motor_example()
    >>> motor.frequency_nom  # doctest: +ELLIPSIS
    376.991...
    """

    return MotorElement(
        n=0,
        tag=None,
        power_nom=Q_(1.5, "cv"),
        voltage_nom=127,
        speed_nom=Q_(1710, "RPM"),
        frequency_nom=Q_(60.0, "Hz"),
        n_poles=4,
        stator_resistance=2.5,
        rotor_resistance=1.8,
        stator_reactance=1.3,
        rotor_reactance=1.3,
        mutual_reactance=43.08,
        Ip_motor=0.0372,
        viscosity_coeff=0.0,
        Ip_load=0.0,
        voltage_net=127,
        frequency_net=Q_(60.0, "Hz"),
    )
