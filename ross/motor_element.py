"""Electric Motor Element module.

This module defines the MotorElement class which represents a 3-phase Induction Electric Motor
simulated using a 4th-order Runge-Kutta method, considering magnetic fluxes and currents.
"""

import numpy as np
import plotly.graph_objects as go

from ross.element import Element
from ross.units import Q_, check_units

from .motor_sourceAC import SourceAC

__all__ = ["MotorElement"]


class MotorElement(Element):
    """A 3-phase Induction Motor element.

    This class creates a electric Three-Phase Induction Motor (TPIM) assuming
    rotor flux as sinchronous reference

    Parameters
    ----------
    n: int
        Node in which the motor will be coupled.
    power : float, pint.Quantity
        Nominal power [W].
    voltage : float, pint.Quantity
        Nominal voltage [V].
    speed : float, pint.Quantity
        Nominal machine rotation [rad/s].
    stator_resistance : float, pint.Quantity
        Stator resistance [Ohm].
    rotor_resistance : float, pint.Quantity
        Rotor resistance [Ohm].
    stator_reactance : float
        Stator self-reactance at frequency [Ohm].
    rotor_reactance : float
        Rotor self-reactance at frequency [Ohm].
    mutual_reactance : float
        Mutual reactance at frequency [Ohm].
    Ip_motor : float, pint.Quantity
        Polar Moment of inertia related to motor axis [kg*m²].
    frequency : float, optional, pint.Quantity
        Nominal frequency [rad/s].
        Default is 60.0 Hz.
    n_poles: int, optional
        Number of machine's poles.
        Default is 4.
    viscosity_coeff : float, optional, pint.Quantity
        Viscosity coefficient [Pa*s].
        Default is 0.0.
    Ip_load : float, optional, pint.Quantity
        Polar Moment of inertia related to load [kg*m²].
        Default is 0.0.
    voltage_net : float, optional, pint.Quantity
        Electrical tension of Power Supply [V].
        Default is None, which adopts the motor's nominal voltage (`voltage`).
    frequency_net : float, optional, pint.Quantity
        Electrical frequency of Power Supply [rad/s].
        Default is None, which adopts the motor's nominal frequency (`frequency`).
    initial_angle_net : float, optional, pint.Quantity
        Initial angular phase frequency of Power Supply [rad].
        Default is 20.0 degrees.
    short_circuit_ratio_net : float, optional
        Short-Circuit Ratio in Common Coupling Point with Power Supply.
        Default is 50.0.
    XR_ratio_net : float, optional
        Reactance (X) / Resistance (R) Ratio in Coupling Point with Power Supply.
        Default is 80.0.
    tag : str, optional
        A tag to name the element.
        Default is None.
    """

    @check_units
    def __init__(
        self,
        n,
        power,
        voltage,
        speed,
        stator_resistance,
        rotor_resistance,
        stator_reactance,
        rotor_reactance,
        mutual_reactance,
        Ip_motor,
        frequency=60.0,
        n_poles=4,
        viscosity_coeff=0.0,
        Ip_load=0.0,
        voltage_net=None,
        frequency_net=None,
        initial_angle_net=None,
        short_circuit_ratio_net=50.0,
        XR_ratio_net=80.0,
        tag=None,
    ):

        # Numerical Validation of NOMP entries
        self.power = float(power)
        self.voltage = float(voltage)
        self.speed = float(speed)
        self.n_poles = int(n_poles)
        self.stator_resistance = float(stator_resistance)
        self.rotor_resistance = float(rotor_resistance)
        self.stator_reactance = float(stator_reactance)
        self.rotor_reactance = float(rotor_reactance)
        self.mutual_reactance = float(mutual_reactance)
        self.Ip_motor = float(Ip_motor)
        self.viscosity_coeff = float(viscosity_coeff)
        self.Ip_load = float(Ip_load)

        if frequency is None:
            self.frequency = Q_(60.0, "Hz").to("rad/s").m
        else:
            self.frequency = float(frequency)

        # Numerical Validation of SCIP entries
        self.voltage_net = self.voltage if voltage_net is None else float(voltage_net)
        self.frequency_net = (
            self.frequency if frequency_net is None else float(frequency_net)
        )
        self.short_circuit_ratio_net = float(short_circuit_ratio_net)
        self.XR_ratio_net = float(XR_ratio_net)

        if initial_angle_net is None:
            self.initial_angle_net = Q_(20.0, "deg").to("rad").m
        else:
            self.initial_angle_net = float(initial_angle_net)

        # Internal model inductances parameters derived from CEMP
        Lls = self.stator_reactance / self.frequency
        Llr = self.rotor_reactance / self.frequency
        self.Lm = self.mutual_reactance / self.frequency
        self.Lss = Lls + self.Lm
        self.Lrr = Llr + self.Lm

        # Internal Electric Motor constants derived from NOMP and CEMP
        snom = (1 - self.speed * self.n_poles / (2 * self.frequency)) * 100
        wnom = (self.frequency * (1 - snom / 100)) / (self.n_poles / 2)
        sigma = 1 - self.Lm**2 / (self.Lss * self.Lrr)
        self.Tnom = self.power / wnom
        self.a = 1 / (sigma * self.Lss)
        self.b = 1 / (sigma * self.Lrr)
        self.c = self.Lm / (sigma * self.Lss * self.Lrr)

        # Short-Circuit Power and Impedances parameters derived from SCIP
        SCC_net = self.short_circuit_ratio_net * self.power
        Zsc = self.voltage_net**2 / SCC_net
        Xsc = Zsc * self.XR_ratio_net / np.sqrt(1 + self.XR_ratio_net**2)
        self.short_circuit_resistance = Xsc / self.XR_ratio_net

        self.n = n
        self.tag = tag

        # Initial values of Rotor speed, Flux angle and Electrial Torque
        # Obs: a possible new feature is to insert non-null initial values user's parameters
        self.wr = 0.0  # Rotor's angular speed in rad*s
        self.thetar = 0.0  # Rotor's angle in rad
        self.thetai = self.initial_angle_net - np.pi / 2
        self.ro = self.thetai  # Flux's initial angle in rad
        self.Te = 0.0  # Electrical Torque in N*m

        # Initial alpha-beta and dq currents (based in nulled instantaneous phase currents)
        ias, ibs, ics = 0, 0, 0
        i_alpha = 2 / 3 * (ias - ibs / 2 - ics / 2)
        i_beta = 2 / 3 * (ibs - ics) * np.sqrt(3) / 2
        ids = i_alpha * np.cos(self.ro) + i_beta * np.sin(self.ro)
        iqs = -i_alpha * np.sin(self.ro) + i_beta * np.cos(self.ro)

        # Initial rotor and stator's inductances
        self.Lds = self.Lss * ids + self.Lm * 0
        self.Lqs = self.Lss * iqs + self.Lm * 0
        self.Ldr = self.Lrr * 0 + self.Lm * ids
        self.Lqr = self.Lrr * 0 + self.Lm * iqs

        # Motor AC Source instance
        self.sourceAC = SourceAC(
            voltage_net=self.voltage,
            frequency_net=Q_(self.frequency, "rad/s").to("Hz").m,
        )

        # Initial simulation parameters scheme
        self.tI = 0.0  # Initial time of simulation (tI)
        self.tF = 5.0  # Final time of simulation (tF)
        self.step = 1e-4  # Resolution  (s)
        self.npts = int(
            (self.tF - self.tI) / self.step
        )  # Number of points in simulation
        self.tTL = (self.tF - self.tI) / 2  # TLoad entrance time
        self.rTL = (
            1.0  # TLoad ratio related Tnom at entrance time tTL (1.0 ->100% Tnom)
        )

        # Time vector and Load Torque vector for the simulation, considering the TLoad entrance time
        self.t_vector, self.dt = np.linspace(self.tI, self.tF, self.npts, retstep=True)
        lenT = int(len(self.t_vector))
        self.TLoad_vector = np.ones(lenT) * self.Tnom * self.rTL
        arr = np.array(self.t_vector)
        itTL = np.abs(
            arr - self.tTL
        ).argmin()  # Catching the near index to time do TLoad entrance
        self.TLoad_vector[0:itTL] = 0.0

    def __str__(self):
        """Convert object into string.

        Returns
        -------
        The object's parameters translated to strings.
        """
        return (
            f"Tag:                                {self.tag}"
            f"\nNode:                               {self.n}"
            f"\n--- Nominal Parameters (NOMP) ---"
            f"\nNominal Power (W):                  {self.power}"
            f"\nNominal Voltage (V):                {self.voltage}"
            f"\nNominal Rotation (rad/s):           {self.speed}"
            f"\nNominal Frequency (Hz):             {Q_(self.frequency, 'rad/s').to('Hz').m}"
            f"\nNumber of Poles:                    {self.n_poles}"
            f"\n--- Circuit Parameters (CEMP) ---"
            f"\nStator Resistance (Ohm):            {self.stator_resistance}"
            f"\nRotor Resistance (Ohm):             {self.rotor_resistance}"
            f"\nStator Reactance (Ohm):             {self.stator_reactance}"
            f"\nRotor Reactance (Ohm):              {self.rotor_reactance}"
            f"\nMutual Reactance (Ohm):             {self.mutual_reactance}"
            f"\nMotor Inertia (kg*m2):              {self.Ip_motor}"
            f"\nViscosity Coefficient (Pa*s):       {self.viscosity_coeff}"
            f"\nLoad Inertia (kg*m2):               {self.Ip_load}"
            f"\n--- Power Supply Parameters (SCIP) ---"
            f"\nSupply Voltage (V):                 {self.voltage_net}"
            f"\nSupply Frequency (Hz):              {self.frequency_net}"
            f"\nInitial Phase Angle (deg):          {self.initial_angle_net}"
            f"\nShort-Circuit Ratio (ad):           {self.short_circuit_ratio_net}"
            f"\nX/R Ratio (ad):                     {self.XR_ratio_net}"
        )

    def __repr__(self):
        pass

    def __eq__(self, other):
        pass

    def __hash__(self):
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

    def _perform_single_step(self, h, t, Tload):
        """Perform a single iteration calculation for the motor dynamics.

        This method calculates the state of the motor for a specific time point 't',
        given the input voltages and load torque. It uses a 4th-order Runge-Kutta
        integration step.

        Parameters
        ----------
        h : float, optional
            Simulation time step (step size) for the Runge-Kutta integration.
            Default is 1e-4.

        t : float
            Current simulation time [s].

        Tload : float
            Load torque applied to the shaft [N.m].


        Returns
        -------
        results : dict
            A dictionary containing the calculated values for the current step:
            - time: Time [s]
            - Vas, Vbs, Vcs: Phase voltage [V]
            - Ias, Ibs, Ics: Phase currents [A]
            - Ialfas, Ibetas: Alpha-Beta currents [A]
            - Ids, Iqs: d-q axis currents [A]
            - TE: Electromagnetic Torque [N.m]
            - TC: Load Torque [N.m]
        """
        # Determine step size h based on current time or use fixed internal h
        # Note: The original logic relies on a fixed h for the RK coefficients.
        # We assume the user calls this sequentially or we rely on the internal h.
        self.h = float(h)

        # Electrical 3-phase tensions
        vas, vbs, vcs = self.sourceAC(t)

        # Updating angles
        w_axis = self.frequency
        self.ro += w_axis * h
        self.thetar += (self.wr * self.n_poles / 2) * h

        # Clarke & Park Transforms for Voltages
        valfas = 2 / 3 * (vas - vbs / 2 - vcs / 2)
        vbetas = 2 / 3 * (vbs - vcs) * np.sqrt(3) / 2
        vds = valfas * np.cos(self.ro) + vbetas * np.sin(self.ro)
        vqs = -valfas * np.sin(self.ro) + vbetas * np.cos(self.ro)

        vdr, vqr = 0, 0

        # Constants for readability in RK4
        Rs, Rsc = self.stator_resistance, self.short_circuit_resistance
        Rr = self.rotor_resistance
        Lds, Lqs = self.Lds, self.Lqs
        Ldr, Lqr = self.Ldr, self.Lqr
        a, b, c = self.a, self.b, self.c
        wr = self.wr
        n_poles = self.n_poles
        Te = self.Te
        Jm, Jl = self.Ip_motor, self.Ip_load
        Bm = self.viscosity_coeff

        # --- Runge-Kutta 4th Order Step ---

        # Step 1
        k11 = h * (vds - (Rs + Rsc) * a * Lds + (Rs + Rsc) * c * Ldr + w_axis * Lqs)
        k21 = h * (vqs - (Rs + Rsc) * a * Lqs + (Rs + Rsc) * c * Lqr - w_axis * Lds)
        k31 = h * (
            vdr - Rr * b * Ldr + Rr * c * Lds + (w_axis - wr * n_poles / 2) * Lqr
        )
        k41 = h * (
            vqr - Rr * b * Lqr + Rr * c * Lqs - (w_axis - wr * n_poles / 2) * Ldr
        )
        k51 = h * (Te / (Jm + Jl) - Bm * wr / (Jm + Jl) - Tload / (Jm + Jl))

        Te_rk = (
            1.5
            * c
            * ((Lqs + k21 / 2) * (Ldr + k31 / 2) - (Lds + k11 / 2) * (Lqr + k41 / 2))
            * n_poles
            / 2
        )

        # Step 2
        k12 = h * (
            vds
            - (Rs + Rsc) * a * (Lds + k11 / 2)
            + (Rs + Rsc) * c * (Ldr + k31 / 2)
            + w_axis * (Lqs + k21 / 2)
        )
        k22 = h * (
            vqs
            - (Rs + Rsc) * a * (Lqs + k21 / 2)
            + (Rs + Rsc) * c * (Lqr + k41 / 2)
            - w_axis * (Lds + k11 / 2)
        )
        k32 = h * (
            vdr
            - Rr * b * (Ldr + k31 / 2)
            + Rr * c * (Lds + k11 / 2)
            + (w_axis - (wr + k51 / 2) * n_poles / 2) * (Lqr + k41 / 2)
        )
        k42 = h * (
            vqr
            - Rr * b * (Lqr + k41 / 2)
            + Rr * c * (Lqs + k21 / 2)
            - (w_axis - (wr + k51 / 2) * n_poles / 2) * (Ldr + k31 / 2)
        )
        k52 = h * (
            Te_rk / (Jm + Jl) - Bm * (wr + k51 / 2) / (Jm + Jl) - Tload / (Jm + Jl)
        )

        Te_rk = (
            1.5
            * c
            * ((Lqs + k22 / 2) * (Ldr + k32 / 2) - (Lds + k12 / 2) * (Lqr + k42 / 2))
            * n_poles
            / 2
        )

        # Step 3
        k13 = h * (
            vds
            - (Rs + Rsc) * a * (Lds + k12 / 2)
            + (Rs + Rsc) * c * (Ldr + k32 / 2)
            + w_axis * (Lqs + k22 / 2)
        )
        k23 = h * (
            vqs
            - (Rs + Rsc) * a * (Lqs + k22 / 2)
            + (Rs + Rsc) * c * (Lqr + k42 / 2)
            - w_axis * (Lds + k12 / 2)
        )
        k33 = h * (
            vdr
            - Rr * b * (Ldr + k32 / 2)
            + Rr * c * (Lds + k12 / 2)
            + (w_axis - (wr + k52 / 2) * n_poles / 2) * (Lqr + k42 / 2)
        )
        k43 = h * (
            vqr
            - Rr * b * (Lqr + k42 / 2)
            + Rr * c * (Lqs + k22 / 2)
            - (w_axis - (wr + k52 / 2) * n_poles / 2) * (Ldr + k32 / 2)
        )
        k53 = h * (
            Te_rk / (Jm + Jl) - Bm * (wr + k52 / 2) / (Jm + Jl) - Tload / (Jm + Jl)
        )

        Te_rk = (
            1.5
            * c
            * ((Lqs + k23) * (Ldr + k33) - (Lds + k13) * (Lqr + k43))
            * n_poles
            / 2
        )

        # Step 4
        k14 = h * (
            vds
            - (Rs + Rsc) * a * (Lds + k13)
            + (Rs + Rsc) * c * (Ldr + k33)
            + w_axis * (Lqs + k23)
        )
        k24 = h * (
            vqs
            - (Rs + Rsc) * a * (Lqs + k23)
            + (Rs + Rsc) * c * (Lqr + k43)
            - w_axis * (Lds + k13)
        )
        k34 = h * (
            vdr
            - Rr * b * (Ldr + k33)
            + Rr * c * (Lds + k13)
            + (w_axis - (wr + k53) * n_poles / 2) * (Lqr + k43)
        )
        k44 = h * (
            vqr
            - Rr * b * (Lqr + k43)
            + Rr * c * (Lqs + k23)
            - (w_axis - (wr + k53) * n_poles / 2) * (Ldr + k33)
        )
        k54 = h * (Te_rk / (Jm + Jl) - Bm * (wr + k53) / (Jm + Jl) - Tload / (Jm + Jl))

        # Update State Variables
        self.Lds += (k11 + 2 * k12 + 2 * k13 + k14) / 6
        self.Lqs += (k21 + 2 * k22 + 2 * k23 + k24) / 6
        self.Ldr += (k31 + 2 * k32 + 2 * k33 + k34) / 6
        self.Lqr += (k41 + 2 * k42 + 2 * k43 + k44) / 6
        self.wr += (k51 + 2 * k52 + 2 * k53 + k54) / 6

        # Calculate Outputs
        ids = a * self.Lds - c * self.Ldr
        iqs = a * self.Lqs - c * self.Lqr
        self.Te = 1.5 * c * (self.Lqs * self.Ldr - self.Lds * self.Lqr) * n_poles / 2

        i_alpha = ids * np.cos(self.ro) - iqs * np.sin(self.ro)
        i_beta = ids * np.sin(self.ro) + iqs * np.cos(self.ro)
        ias = i_alpha
        ibs = -i_alpha / 2 + np.sqrt(3) * i_beta / 2
        ics = -i_alpha / 2 - np.sqrt(3) * i_beta / 2

        self.current_time = t

        return {
            "time": t,
            "Vas": vas,
            "Vbs": vbs,
            "Vcs": vcs,
            "Ias": ias,
            "Ibs": ibs,
            "Ics": ics,
            "Ialfas": i_alpha,
            "Ibetas": i_beta,
            "Ids": ids,
            "Iqs": iqs,
            "TE": self.Te,
            "Tl": Tload,
            "wr": self.wr,
            "RPM": self.wr * 30 / np.pi,
        }

    def run(self):
        """Run the simulation for a series of time steps.

        Parameters
        ----------

        Returns
        -------
        results : dict
            A dictionary containing lists of results for the entire simulation:
            - tempo, Ias, Ibs, Ics, Ialfas, Ibetas, Ids, Iqs, TE, TC.
        """
        results = {
            "time": [],
            "Vas": [],
            "Vbs": [],
            "Vcs": [],
            "Ias": [],
            "Ibs": [],
            "Ics": [],
            "Ialfas": [],
            "Ibetas": [],
            "Ids": [],
            "Iqs": [],
            "TE": [],
            "Tl": [],
            "wr": [],
            "RPM": [],
        }

        # Ensure inputs are iterable/arrays
        time_vector = np.array(self.t_vector)
        Tload_vector = np.array(self.TLoad_vector)

        for i, t in enumerate(time_vector):
            # Run single step calculation
            step_result = self._perform_single_step(self.dt, t, Tload_vector[i])

            # Append results
            for key in results:
                results[key].append(step_result[key])

        return results

    def plot(self, results):
        """Plot the simulation results (Torque and Speed) in separate figures.

        Parameters
        ----------
        results : dict
            Dictionary returned by the 'run' method containing lists of results.

        Returns
        -------
        fig_torque, fig_speed, fig_currents, fig_voltages: tuple of plotly.graph_objects.Figure
            Four separate figures for Torque, Speed, Electric Current and Tension.
        """

        # Figure 1: Torques
        fig_torque = go.Figure()
        fig_torque.add_trace(
            go.Scatter(
                x=results["time"],
                y=results["TE"],
                name="Electromagnetic Torque(N.m)",
                line=dict(color="blue"),
            )
        )
        fig_torque.add_trace(
            go.Scatter(
                x=results["time"],
                y=results["Tl"],
                name="Load Torque (N.m)",
                line=dict(color="red"),
            )
        )
        fig_torque.update_layout(
            title="Motor operation: Electromagnetic Torque and Load Torque",
            xaxis_title="Time (s)",
            yaxis_title="Torque (N.m)",
        )

        # Figure 2: Shaft Motor Speed
        fig_speed = go.Figure()
        fig_speed.add_trace(
            go.Scatter(
                x=results["time"],
                y=results["RPM"],
                name="Rotação (RPM)",
                line=dict(color="red"),
            )
        )
        fig_speed.update_layout(
            title="Motor operation: Shaft Speed",
            xaxis_title="Time (s)",
            yaxis_title="Motor speed (RPM)",
        )

        # Figure 3: Phase Currents
        fig_currents = go.Figure()
        fig_currents.add_trace(
            go.Scatter(
                x=results["time"],
                y=results["Ias"],
                name="Ia (A)",
                line=dict(color="blue"),
            )
        )
        fig_currents.add_trace(
            go.Scatter(
                x=results["time"],
                y=results["Ibs"],
                name="Ib (A)",
                line=dict(color="black"),
            )
        )
        fig_currents.add_trace(
            go.Scatter(
                x=results["time"],
                y=results["Ics"],
                name="Ic (A)",
                line=dict(color="red"),
            )
        )
        fig_currents.update_layout(
            title="Motor operation: Stator Currents",
            xaxis_title="Time (s)",
            yaxis_title="Currents (A)",
        )

        # Figure 4: Phase Tensions
        fig_voltages = go.Figure()
        fig_voltages.add_trace(
            go.Scatter(
                x=results["time"],
                y=results["Vas"],
                name="Va (V)",
                line=dict(color="blue"),
            )
        )
        fig_voltages.add_trace(
            go.Scatter(
                x=results["time"],
                y=results["Vbs"],
                name="Vb (V)",
                line=dict(color="black"),
            )
        )
        fig_voltages.add_trace(
            go.Scatter(
                x=results["time"],
                y=results["Vcs"],
                name="Vc (V)",
                line=dict(color="red"),
            )
        )
        fig_voltages.update_layout(
            title="Motor operation: Stator Voltages",
            xaxis_title="Time (s)",
            yaxis_title="Stator Voltages (V)",
        )
        return fig_torque, fig_speed, fig_currents, fig_voltages

    # def simulparams(self, tI, tF, tTL, rTL, npts):
    #     """Simulation Parameters control

    #     Parameters
    #     ----------
    #     time_vector : array_like
    #         Array of time steps.

    #     Tload_vector : array_like
    #         Array of load torques.

    #     Returns
    #     -------
    #     results : dict
    #         A dictionary containing lists of results for the entire simulation:
    #         - tempo, Ias, Ibs, Ics, Ialfas, Ibetas, Ids, Iqs, TE, TC.
    #     """
    def simulparams(self, tI=None, tF=None, step=None, npts=None, tTL=None, rTL=None):
        # Checks if no arguments were passed to trigger the report (Requirement 3)
        no_args = all(v is None for v in [tI, tF, step, npts, tTL, rTL])

        # Simulation parameters setup
        self.tI = tI if tI is not None else 0.0
        self.tF = tF if tF is not None else 5.0

        # Logic for step vs npts
        if step is not None:
            if step == 0:
                raise ValueError("Parameter 'step' cannot be zero.")
            self.step = step
            self.npts = int((self.tF - self.tI) / self.step)

            if self.npts > 10**6:
                raise ValueError("Number of points (npts) cannot be greater than 10E6.")

        elif npts is not None:
            self.npts = int(npts)
            self.step = (self.tF - self.tI) / self.npts

        else:  # If neither is provided, assume defaults
            self.step = 1e-4
            self.npts = int((self.tF - self.tI) / self.step)

        # Sets tTL and rTL based on the resolved tI and tF values above
        self.tTL = tTL if tTL is not None else (self.tF - self.tI) / 2
        self.rTL = rTL if rTL is not None else 1.0

        # Report if no values were provided
        if no_args:
            print("=== Default Parameters Report ===")
            print(f"tI   = {self.tI}")
            print(f"tF   = {self.tF}")
            print(f"step = {self.step}")
            print(f"tTL  = {self.tTL}")
            print(f"rTL  = {self.rTL}")
            print("=================================\n")

        # Vector creation
        # Time vector and deltaTime
        self.t_vector, self.dt = np.linspace(self.tI, self.tF, self.npts, retstep=True)

        lenT = len(self.t_vector)

        # Creating TLoad vector
        self.TLoad_vector = np.ones(lenT) * self.Tnom * self.rTL

        arr = np.array(self.t_vector)

        # Catching the near index to the time of TLoad entrance
        itTL = np.abs(arr - self.tTL).argmin()

        # Setting TLoad vector
        self.TLoad_vector[0:itTL] = 0.0

    # return dt, time_vector, Tload_vector


def motor_example():
    """Create an example of notor element.

    This function returns an instance of a simple electric motor. The purpose is to make available
    a simple model so that doctest can be written using it.

    Returns
    -------
    motor : ross.MotorElement
        An instance of a motor object.

    Examples
    --------

    """

    motor = MotorElement(
        n=0,
        tag=None,
        power=Q_(1.5 * 735.499, "W"),  # Direct conversion cv --> W
        voltage=127,  # Volts
        speed=Q_(1725, "RPM"),  # RPM
        frequency=Q_(60.0, "Hz"),  # Hz
        n_poles=4,  # Stator's poles
        stator_resistance=2.5,  # Ohm
        rotor_resistance=1.8,  # Ohm
        stator_reactance=1.3,  # Ohm
        rotor_reactance=1.3,  # Ohm
        mutual_reactance=43.08,  # Ohm
        Ip_motor=0.0372,  # kg*m2
        viscosity_coeff=0.0,  # kg*m*s2
        Ip_load=0.0,  # kg*m2
        voltage_net=127,  # Volts
        frequency_net=Q_(60.0, "Hz"),  # Hz
        # npts=1000
        # initial_angle_net=Q_(20.0, 'deg'),
        # short_circuit_ratio_net=50.0,
        # XR_ratio_net=80.0
    )
    # Adjusting simulation parameters

    motor.sourceAC.harmonics(fHO=[5, 7], aHO=[5, 5])
    motor.sourceAC.voltage_net = 90.0
    motor.simulparams(tTL=3.0, rTL=1.5)
    motor.sourceAC.harmonics("enable")
    motor.sourceAC.unbalances("disable")

    dataResults = motor.run()

    fig_torque, fig_speed, fig_currents, fig_voltages = motor.plot(dataResults)

    return motor, fig_torque, fig_speed, fig_currents, fig_voltages
