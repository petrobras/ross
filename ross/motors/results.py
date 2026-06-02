"""Motor results plotting module.

This module returns graphs for each type of analyses in motors.
"""

import plotly.graph_objects as go
import numpy as np

from ross.results import Results
from ross.units import Q_, check_units
from .utils import windowed_dfft


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

        self.line_voltages = dict()
        self.line_voltages["ab"] = self.voltages["a"] - self.voltages["b"]
        self.line_voltages["bc"] = self.voltages["b"] - self.voltages["c"]
        self.line_voltages["ca"] = self.voltages["c"] - self.voltages["a"]

    @check_units
    def _plot_dfft(
        self,
        result_dict,
        title,
        yaxis_title,
        fig,
        frequency_units="Hz",
        frequency_range=None,
        **kwargs,
    ):

        if frequency_range is not None:
            min_freq, max_freq = frequency_range
            frequency_range = Q_(frequency_range, "rad/s").to("Hz").m

        dt = self.t[1] - self.t[0]

        for name, signal in result_dict.items():
            freq, mag = windowed_dfft(signal, dt)

            if frequency_range is not None:
                delta = 0.01 * (frequency_range[1] - frequency_range[0])
                mask = (freq >= frequency_range[0] - delta) & (
                    freq <= frequency_range[1] + delta
                )
                mag = mag[mask]
                freq = freq[mask]

            fig.add_trace(
                go.Scatter(
                    x=Q_(freq, "Hz").to(frequency_units).m,
                    y=mag,
                    name=name,
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=f"Frequency ({frequency_units})",
            yaxis_title=yaxis_title,
        )

        if frequency_range is not None:
            fig.update_xaxes(
                range=[min_freq, max_freq], rangeslider=dict(visible=False)
            )

        fig.update_layout(**kwargs)

        return fig

    def _plot_time(self, result_dict, title, yaxis_title, fig, **kwargs):

        for name, signal in result_dict.items():
            fig.add_trace(
                go.Scatter(
                    x=self.t,
                    y=signal,
                    name=name,
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title=yaxis_title,
        )

        fig.update_layout(**kwargs)

        return fig

    @check_units
    def plot_torque(
        self,
        domain="time",
        torque_units="N*m",
        frequency_units="Hz",
        frequency_range=None,
        fig=None,
        **kwargs,
    ):
        """Plot electromagnetic and load torques in time domain or frequency domain.

        Parameters
        ----------
        domain : str, optional
            Domain for plotting. Options are "time" or "frequency".
            Default is "time".
        torque_units : str, optional
            Unit for torque display. Default is "N*m".
        frequency_units : str, optional
            Unit for frequency display in frequency domain. Default is "Hz".
        frequency_range : tuple, pint.Quantity(tuple), optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
            Default is None (no filter).
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

        main_inputs = dict(
            result_dict={
                "Electromagnetic Torque": Q_(self.electric_torque, "N*m")
                .to(torque_units)
                .m,
                "Load Torque": Q_(self.load_torque, "N*m").to(torque_units).m,
            },
            title="Motor operation: Electromagnetic Torque and Load Torque",
            yaxis_title=f"Torque ({torque_units})",
            fig=fig,
        )

        if domain == "frequency":
            fig = self._plot_dfft(
                **main_inputs,
                frequency_units=frequency_units,
                frequency_range=frequency_range,
                **kwargs,
            )

        else:
            fig = self._plot_time(
                **main_inputs,
                **kwargs,
            )

        return fig

    @check_units
    def plot_speed(
        self,
        domain="time",
        speed_units="RPM",
        frequency_units="Hz",
        frequency_range=None,
        fig=None,
        **kwargs,
    ):
        """Plot rotor speed in time domain or frequency domain.

        Parameters
        ----------
        domain : str, optional
            Domain for plotting. Options are "time" or "frequency".
            Default is "time".
        speed_units : str, optional
            Unit for speed display. Default is "RPM".
        frequency_units : str, optional
            Unit for frequency display in frequency domain. Default is "Hz".
        frequency_range : tuple, pint.Quantity(tuple), optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
            Default is None (no filter).
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

        main_inputs = dict(
            result_dict={
                "Shaft Speed": Q_(self.speed, "rad/s").to(speed_units).m,
            },
            title="Motor operation: Shaft Speed",
            yaxis_title=f"Motor speed ({speed_units})",
            fig=fig,
        )

        if domain == "frequency":
            fig = self._plot_dfft(
                **main_inputs,
                frequency_units=frequency_units,
                frequency_range=frequency_range,
                **kwargs,
            )
        else:
            fig = self._plot_time(
                **main_inputs,
                **kwargs,
            )

        return fig

    def plot_phase_currents(
        self,
        domain="time",
        reference_frame="a-b-c",
        frequency_units="Hz",
        frequency_range=None,
        fig=None,
        **kwargs,
    ):
        """Plot phase currents in selected reference frame.

        Parameters
        ----------
        domain : str, optional
            Domain for plotting. Options are "time" or "frequency".
            Default is "time".
        reference_frame : str, optional
            Reference frame for current display. Options: 'a-b-c' (3-phase),
            'alpha-beta' (Clarke), 'd-q' (Park). Default is 'a-b-c'.
        frequency_units : str, optional
            Unit for frequency display in frequency domain. Default is "Hz".
        frequency_range : tuple, pint.Quantity, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
            Default is None (no filter).
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

        current = PhaseResults(self.t, self.currents, "I")

        if domain == "frequency":
            fig = current.plot_dfft(
                reference_frame=reference_frame,
                fig=fig,
                frequency_units=frequency_units,
                frequency_range=frequency_range,
                **kwargs,
            )
        else:
            fig = current.plot(reference_frame=reference_frame, fig=fig, **kwargs)

        return fig

    def plot_phase_voltages(
        self,
        domain="time",
        frequency_units="Hz",
        frequency_range=None,
        fig=None,
        **kwargs,
    ):
        """Plot 3-phase voltages in time domain or frequency domain.

        Parameters
        ----------
        domain : str, optional
            Domain for plotting. Options are "time" or "frequency".
            Default is "time".
        frequency_units : str, optional
            Unit for frequency display in frequency domain. Default is "Hz".
        frequency_range : tuple, pint.Quantity, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
            Default is None (no filter).
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
            kwargs["title"] = "Motor operation: Stator Phase Voltages"

        voltage = PhaseResults(self.t, self.voltages, "V")

        if domain == "frequency":
            fig = voltage.plot_dfft(
                fig=fig,
                frequency_units=frequency_units,
                frequency_range=frequency_range,
                **kwargs,
            )
        else:
            fig = voltage.plot(fig=fig, **kwargs)

        return fig

    def plot_line_voltages(
        self,
        domain="time",
        frequency_units="Hz",
        frequency_range=None,
        fig=None,
        **kwargs,
    ):
        """Plot line voltages in time domain or frequency domain.

        Parameters
        ----------
        domain : str, optional
            Domain for plotting. Options are "time" or "frequency".
            Default is "time".
        frequency_units : str, optional
            Unit for frequency display in frequency domain. Default is "Hz".
        frequency_range : tuple, pint.Quantity, optional
            Tuple with (min, max) values for the frequencies that will be plotted.
            Frequencies that are not within the range are filtered out and are not plotted.
            It is possible to use a pint Quantity (e.g. Q_((2000, 1000), "RPM")).
            Default is None (no filter).
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with rotor speed trace.
        """
        if fig is None:
            fig = go.Figure()

        main_inputs = dict(
            result_dict={
                f"V<sub>{key}</sub>": value for key, value in self.line_voltages.items()
            },
            title="Motor operation: Stator Line Voltages",
            yaxis_title="Voltage (V)",
            fig=fig,
        )

        if domain == "frequency":
            fig = self._plot_dfft(
                **main_inputs,
                frequency_units=frequency_units,
                frequency_range=frequency_range,
                **kwargs,
            )

        else:
            fig = self._plot_time(
                **main_inputs,
                **kwargs,
            )

        return fig


class PhaseResults(Results):
    """Store and plot phase data.

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

    _DATA_TYPE_MAP = {
        "I": {"units": "A", "name": "Current"},
        "V": {"units": "V", "name": "Voltage"},
    }

    def __init__(self, t, data, data_type):
        """Initialize results.

        Parameters
        ----------
        t : array_like
            Time vector [s].
        data : dict
            Data with keys for different reference frames
            (a, b, c, alpha, beta, d, q).
        data_type : str
            Type of data to store. Options: 'I' (Current), 'V' (Voltage).
        """
        self.t = t
        self.data = data
        self.data_type = data_type

        if data_type not in self._DATA_TYPE_MAP:
            raise ValueError(
                f"Invalid data type '{data_type}'. Valid options: {set(self._DATA_TYPE_MAP.keys())}"
            )

        self.name = self._DATA_TYPE_MAP[data_type]["name"]
        self.units = self._DATA_TYPE_MAP[data_type]["units"]

    def plot(self, reference_frame="a-b-c", fig=None, **kwargs):
        """Plot data over time in selected reference frame.

        Parameters
        ----------
        reference_frame : str, optional
            Reference frame for data display. Options: 'a-b-c' (3-phase),
            'alpha-beta' (Clarke), 'd-q' (Park). Default is 'a-b-c'.
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with data traces in selected reference frame.
        """
        if fig is None:
            fig = go.Figure()

        reference_frame = reference_frame.split("-")

        for axis in reference_frame:
            try:
                fig.add_trace(
                    go.Scatter(
                        x=self.t,
                        y=self.data[axis],
                        name=f"{self.data_type}<sub>{self._REFERENCE_MAP[axis]}</sub>",
                    )
                )
            except KeyError:
                raise ValueError(
                    f"Invalid reference frame axis '{axis}'. "
                    f"Valid options: {set(self.data.keys()).intersection(set(self._REFERENCE_MAP.keys()))}"
                )

        fig.update_layout(
            title=f"Phase {self.name}s",
            xaxis_title="Time (s)",
            yaxis_title=f"{self.name} ({self.units})",
        )

        fig.update_layout(**kwargs)

        return fig

    @check_units
    def plot_dfft(
        self,
        reference_frame="a-b-c",
        fig=None,
        frequency_units="Hz",
        frequency_range=None,
        **kwargs,
    ):
        """Plot data in frequency domain.

        Parameters
        ----------
        reference_frame : str, optional
            Reference frame for data display. Options: 'a-b-c' (3-phase),
            'alpha-beta' (Clarke), 'd-q' (Park). Default is 'a-b-c'.
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        frequency_units : str, optional
            Units for frequency axis. Default is 'Hz'.
        frequency_range : tuple, pint.Quantity, optional
            Frequency range to display. Default is None.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with data traces in selected reference frame.
        """
        if fig is None:
            fig = go.Figure()

        reference_frame = reference_frame.split("-")

        if frequency_range is not None:
            min_freq, max_freq = frequency_range
            frequency_range = Q_(frequency_range, "rad/s").to("Hz").m

        dt = self.t[1] - self.t[0]

        for axis in reference_frame:
            freq, mag = windowed_dfft(self.data[axis], dt)

            if frequency_range is not None:
                delta = 0.01 * (frequency_range[1] - frequency_range[0])
                mask = (freq >= frequency_range[0] - delta) & (
                    freq <= frequency_range[1] + delta
                )
                mag = mag[mask]
                freq = freq[mask]

            try:
                fig.add_trace(
                    go.Scatter(
                        x=Q_(freq, "Hz").to(frequency_units).m,
                        y=mag,
                        name=f"{self.data_type}<sub>{self._REFERENCE_MAP[axis]}</sub>",
                    )
                )
            except KeyError:
                raise ValueError(
                    f"Invalid reference frame axis '{axis}'. "
                    f"Valid options: {set(self.data.keys()).intersection(set(self._REFERENCE_MAP.keys()))}"
                )

        fig.update_layout(
            title=f"Phase {self.name}s",
            xaxis_title=f"Frequency ({frequency_units})",
            yaxis_title=f"{self.name} ({self.units})",
        )

        if frequency_range is not None:
            fig.update_xaxes(
                range=[min_freq, max_freq], rangeslider=dict(visible=False)
            )

        fig.update_layout(**kwargs)

        return fig
