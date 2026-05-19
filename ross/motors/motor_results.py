"""Motor results plotting module.

This module returns graphs for each type of analyses in motors.
"""

import plotly.graph_objects as go

from ross.results import Results
from ross.units import Q_


class MotorTimeResponseResults(Results):
    """Class for time response results."""

    def __init__(self, t, electric_torque, load_torque, speed, currents, voltages):
        self.t = t
        self.electric_torque = electric_torque
        self.load_torque = load_torque
        self.speed = speed
        self.currents = currents
        self.voltages = voltages

    def plot_torque(self, torque_unit="N*m", fig=None):
        """Plot the torque results."""
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

        return fig

    def plot_speed(self, speed_unit="RPM", fig=None):
        """Plot the shaft motor speed."""
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

        return fig

    def plot_phase_currents(self, reference_frame="a-b-c", fig=None):
        """Plot the phase currents.

        Parameters
        ----------
        reference_frame : str, optional
            Reference frame for current. Options available: 'a-b-c', 'alpha-beta', 'd-q'.
            Default is 'a-b-c'.
        """
        if fig is None:
            fig = go.Figure()

        reference_frame = reference_frame.split("-")

        greek_map = {
            "alpha": "α",
            "beta": "β",
            "a": "a",
            "b": "b",
            "c": "c",
            "d": "d",
            "q": "q",
        }

        for axis in reference_frame:
            fig.add_trace(
                go.Scatter(
                    x=self.t,
                    y=self.currents[axis],
                    name=f"I<sub>{greek_map[axis]}</sub>",
                )
            )

        fig.update_layout(
            title="Motor operation: Stator Currents",
            xaxis_title="Time (s)",
            yaxis_title="Current (A)",
        )

        return fig

    def plot_phase_tensions(self, fig=None):
        """Plot the phase tensions."""
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
            title="Motor operation: Stator Voltages",
            xaxis_title="Time (s)",
            yaxis_title="Voltage (V)",
        )

        return fig
