"""Motor results plotting module.

This module returns graphs for each type of analyses in motors.
"""

import plotly.graph_objects as go

from ross.results import Results
from ross.units import Q_


class MotorResponseResults(Results):
    """Store and plot motor time response results.

    Attributes
    ----------
    t : array
        Time vector [s].
    electric_torque : array
        Electrical torque at each time step [N·m].
    load_torque : array
        Load torque at each time step [N·m].
    speed : array
        Rotor speed at each time step [rad/s].
    currents : dict
        Phase currents with keys 'a', 'b', 'c' (3-phase), 'alpha', 'beta'
        (Clarke), and 'd', 'q' (Park) [A].
    voltages : dict
        Phase voltages with keys 'a', 'b', 'c' [V].
    """

    def __init__(self, t, electric_torque, load_torque, speed, currents, voltages):
        """Initialize motor response results.

        Parameters
        ----------
        t : array_like
            Time vector [s].
        electric_torque : array_like
            Electrical torque at each time step [N·m].
        load_torque : array_like
            Load torque at each time step [N·m].
        speed : array_like
            Rotor speed at each time step [rad/s].
        currents : dict
            Phase currents with keys for different reference frames.
        voltages : dict
            Phase voltages with keys for each phase.
        """
        self.t = t
        self.electric_torque = electric_torque
        self.load_torque = load_torque
        self.speed = speed
        self.currents = currents
        self.voltages = voltages

    def plot_torque(self, torque_unit="N*m", fig=None, **kwargs):
        """Plot electromagnetic and load torques over time.

        Parameters
        ----------
        torque_unit : str, optional
            Unit for torque display. Default is "N*m".
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with electromagnetic and load torque traces.
        """
        if fig is None:
            fig = go.Figure()

        fig.add_traces(
            [
                go.Scatter(
                    x=self.t,
                    y=Q_(self.electric_torque, "N*m").to(torque_unit).m,
                    name="Electromagnetic Torque",
                ),
                go.Scatter(
                    x=self.t,
                    y=Q_(self.load_torque, "N*m").to(torque_unit).m,
                    name="Load Torque",
                ),
            ]
        )

        fig.update_layout(
            title="Motor operation: Electromagnetic Torque and Load Torque",
            xaxis_title="Time (s)",
            yaxis_title=f"Torque ({torque_unit})",
        )

        fig.update_layout(**kwargs)

        return fig

    def plot_speed(self, speed_unit="RPM", fig=None, **kwargs):
        """Plot rotor speed over time.

        Parameters
        ----------
        speed_unit : str, optional
            Unit for speed display. Default is "RPM".
        fig : plotly.graph_objects.Figure, optional
            Figure to add trace to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with rotor speed trace.
        """
        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.t,
                y=Q_(self.speed, "rad/s").to(speed_unit).m,
                name="Shaft Speed",
            )
        )

        fig.update_layout(
            title="Motor operation: Shaft Speed",
            xaxis_title="Time (s)",
            yaxis_title=f"Motor speed ({speed_unit})",
        )

        fig.update_layout(**kwargs)

        return fig

    def plot_phase_currents(self, reference_frame="a-b-c", fig=None, **kwargs):
        """Plot phase currents in selected reference frame.

        Parameters
        ----------
        reference_frame : str, optional
            Reference frame for current display. Options: 'a-b-c' (3-phase),
            'alpha-beta' (Clarke), 'd-q' (Park). Default is 'a-b-c'.
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with current traces in selected reference frame.
        """
        if kwargs.get("title") is None:
            kwargs["title"] = "Motor operation: Stator Currents"

        current = CurrentTimeResults(self.t, self.currents)
        fig = current.plot(reference_frame=reference_frame, fig=fig, **kwargs)

        return fig

    def plot_phase_voltages(self, fig=None, **kwargs):
        """Plot 3-phase voltages over time.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with phase voltage traces.
        """
        if kwargs.get("title") is None:
            kwargs["title"] = "Motor operation: Stator Voltages"

        voltage = VoltageTimeResults(self.t, self.voltages)
        fig = voltage.plot(fig=fig, **kwargs)

        return fig


class CurrentTimeResults(Results):
    """Store and plot time-domain motor current data.

    Supports multiple reference frames: 3-phase (a-b-c), Clarke (alpha-beta),
    and Park (d-q) transforms.
    """

    _REFERENCE_MAP = {
        "a": "a",
        "b": "b",
        "c": "c",
        "alpha": "α",
        "beta": "β",
        "d": "d",
        "q": "q",
    }

    def __init__(self, t, currents):
        """Initialize current results.

        Parameters
        ----------
        t : array_like
            Time vector [s].
        currents : dict
            Currents with keys for different reference frames (a, b, c, alpha,
            beta, d, q) [A].
        """
        self.t = t
        self.currents = currents

    def plot(self, reference_frame="a-b-c", fig=None, **kwargs):
        """Plot currents in selected reference frame.

        Parameters
        ----------
        reference_frame : str, optional
            Reference frame for current display. Options: 'a-b-c' (3-phase),
            'alpha-beta' (Clarke), 'd-q' (Park). Default is 'a-b-c'.
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with current traces in selected reference frame.
        """
        if fig is None:
            fig = go.Figure()

        reference_frame = reference_frame.split("-")

        for axis in reference_frame:
            try:
                fig.add_trace(
                    go.Scatter(
                        x=self.t,
                        y=self.currents[axis],
                        name=f"I<sub>{self._REFERENCE_MAP[axis]}</sub>",
                    )
                )
            except KeyError:
                raise ValueError(
                    f"Invalid reference frame axis '{axis}'. "
                    f"Valid options: {set(self.currents.keys()).intersection(set(self._REFERENCE_MAP.keys()))}"
                )

        fig.update_layout(
            title="Phase Currents",
            xaxis_title="Time (s)",
            yaxis_title="Current (A)",
        )

        fig.update_layout(**kwargs)

        return fig


class VoltageTimeResults(Results):
    """Store and plot time-domain motor voltage data."""

    def __init__(self, t, voltages):
        """Initialize voltage results.

        Parameters
        ----------
        t : array_like
            Time vector [s].
        voltages : dict
            3-phase voltages with keys 'a', 'b', 'c' [V].
        """
        self.t = t
        self.voltages = voltages

    def plot(self, fig=None, **kwargs):
        """Plot 3-phase voltages over time.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with 3-phase voltage traces.
        """
        if fig is None:
            fig = go.Figure()

        for axis in self.voltages.keys():
            fig.add_trace(
                go.Scatter(
                    x=self.t,
                    y=self.voltages[axis],
                    name=f"V<sub>{axis}</sub>",
                )
            )

        fig.update_layout(
            title="3-phase Source Voltage",
            xaxis_title="Time (s)",
            yaxis_title="Voltage (V)",
        )

        fig.update_layout(**kwargs)

        return fig
