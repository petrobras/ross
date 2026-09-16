"""Motor results plotting module.

This module returns graphs for each type of analyses in motors.
"""

import plotly.graph_objects as go
from scipy.interpolate import interp1d
import numpy as np

from plotly_resampler import FigureResampler

from ross.results import Results
from ross.units import Q_, check_units
from .utils import windowed_dfft


class PhaseResults(Results):
    """Store and plot phase data.

    Supports multiple reference frames: 3-phase (a-b-c), Clarke (alpha-beta),
    and Park (d-q) transforms.

    Examples
    --------
    >>> from ross import motor_example
    >>> motor = motor_example()
    >>> dt = 1e-3
    >>> tf = 1.0
    >>> t = np.arange(0, tf + dt, dt)
    >>> results = motor.run_direct_on_line(t)
    >>> current = PhaseResults(results.t, results.currents, "I", results.t_eval)
    >>> current.units
    'A'
    >>> fig = current.plot(reference_frame="alpha-beta")
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

    def __init__(self, t, data, data_type, t_eval=None):
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
        t_eval : array, optional
            Time points at which the simulation results are returned [s].
            All values must lie within the range of `t`. The spacing between
            consecutive points may be larger than the time step used in `t`,
            allowing the results to be sampled at a lower temporal resolution.
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

        min_dt = 1e-4
        dt = np.diff(self.t).min()
        self.n_shown_samples = int(len(self.t) * dt / min_dt) if dt < min_dt else None

        self.t_eval = t_eval

    def plot(self, reference_frame="a-b-c", fig=None, n_shown_samples=None, **kwargs):
        """Plot data over time in selected reference frame.

        Parameters
        ----------
        reference_frame : str, optional
            Reference frame for data display. Options: 'a-b-c' (3-phase),
            'alpha-beta' (Clarke), 'd-q' (Park). Default is 'a-b-c'.
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        n_shown_samples : int, optional
            Number of samples to display in the plot. If None, all samples are displayed.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with data traces in selected reference frame.

        Examples
        --------
        >>> from ross import motor_example
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> t = np.arange(0, tf + dt, dt)
        >>> results = motor.run_direct_on_line(t)
        >>> voltage = PhaseResults(results.t, results.voltages, "V", results.t_eval)
        >>> fig = voltage.plot(reference_frame="a-b-c")
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

        n_samples = n_shown_samples or self.n_shown_samples
        if n_samples is not None:
            fig = FigureResampler(fig, default_n_shown_samples=n_samples)

        return fig

    @check_units
    def plot_dfft(
        self,
        reference_frame="a-b-c",
        fig=None,
        frequency_units="Hz",
        frequency_range=None,
        n_shown_samples=None,
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
        n_shown_samples : int, optional
            Number of samples to display in the plot. If None, all samples are displayed.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with data traces in selected reference frame.

        Examples
        --------
        >>> from ross import motor_example
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> t = np.arange(0, tf + dt, dt)
        >>> results = motor.run_direct_on_line(t)
        >>> current = PhaseResults(results.t, results.currents, "I", results.t_eval)
        >>> fig = current.plot_dfft(reference_frame="a-b-c")
        >>> fig = current.plot_dfft(reference_frame="d-q", frequency_units="Hz")
        """
        if fig is None:
            fig = go.Figure()

        reference_frame = reference_frame.split("-")

        if frequency_range is not None:
            min_freq, max_freq = Q_(frequency_range, "rad/s").to(frequency_units).m
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

        n_samples = n_shown_samples or self.n_shown_samples
        if n_samples is not None and n_samples < len(fig.data[0].x):
            fig = FigureResampler(fig, default_n_shown_samples=n_samples)

        return fig


class MotorResponseResults(Results):
    """Store and plot motor time response results.

    Parameters
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
    t_eval : array, optional
        Time points at which the simulation results are returned [s].
        All values must lie within the range of `t`. The spacing between
        consecutive points may be larger than the time step used in `t`,
        allowing the results to be sampled at a lower temporal resolution.

    Examples
    --------
    >>> from ross import motor_example
    >>> motor = motor_example()
    >>> dt = 1e-3
    >>> tf = 1.0
    >>> t = np.arange(0, tf + dt, dt)
    >>> results = motor.run_direct_on_line(t)
    >>> results.speed.shape == t.shape
    True
    >>> sorted(results.currents.keys())
    ['a', 'alpha', 'b', 'beta', 'c', 'd', 'q']
    >>> sorted(results.line_voltages.keys())
    ['ab', 'bc', 'ca']
    """

    def __init__(
        self, t, electric_torque, load_torque, speed, currents, voltages, t_eval=None
    ):
        """Initialize motor response results."""
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

        min_dt = 1e-4
        dt = np.diff(self.t).min()
        self.n_shown_samples = int(len(self.t) * dt / min_dt) if dt < min_dt else None

        self.t_eval = t_eval

    def sample_at(self, attr, t_eval=None):
        """Sample a time-domain signal at specified time instants using
        cubic interpolation.

        Parameters
        ----------
        attr : str
            Name of the attribute containing the signal. Must refer to a
            time-series stored in the object. Common options include:
            - electric_torque
            - load_torque
            - speed
            - currents
            - voltages
            - line_voltages
        t_eval : array_like, optional
            Time instants at which to sample the signal. If None, the
            evaluation grid specified during initialization is used.

        Returns
        -------
        array_like or dict of ndarray
            Interpolated signal evaluated at `t_eval`.

        Examples
        --------
        >>> from ross import motor_example
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> t = np.arange(0, tf + dt, dt)
        >>> results = motor.run_direct_on_line(t)
        >>> t_sample = np.array([0.5])
        >>> speed = results.sample_at("speed", t_eval=t_sample)
        >>> speed  # doctest: +ELLIPSIS
        array([187.1460...])
        >>> currents = results.sample_at("currents", t_eval=t_sample)
        >>> currents['a']  # doctest: +ELLIPSIS
        array([0.866...])
        """
        if t_eval is None:
            t_eval = self.t_eval

        def interp(y):
            f = interp1d(self.t, y, kind="cubic")
            y_eval = f(t_eval)
            return y_eval

        data = getattr(self, attr)

        if isinstance(data, dict):
            return {key: interp(value) for key, value in data.items()}

        return interp(data)

    def _plot_dfft(
        self,
        result_dict,
        title,
        yaxis_title,
        fig,
        frequency_units="Hz",
        frequency_range=None,
        n_shown_samples=None,
        **kwargs,
    ):

        if frequency_range is not None:
            min_freq, max_freq = Q_(frequency_range, "rad/s").to(frequency_units).m
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

        n_samples = n_shown_samples or self.n_shown_samples
        if n_samples is not None and n_samples < len(fig.data[0].x):
            fig = FigureResampler(fig, default_n_shown_samples=n_samples)

        return fig

    def _plot_time(
        self, result_dict, title, yaxis_title, fig, n_shown_samples=None, **kwargs
    ):

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

        n_samples = n_shown_samples or self.n_shown_samples
        if n_samples is not None:
            fig = FigureResampler(fig, default_n_shown_samples=n_samples)

        return fig

    @check_units
    def plot_torque(
        self,
        domain="time",
        torque_units="N*m",
        frequency_units="Hz",
        frequency_range=None,
        n_shown_samples=None,
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
        n_shown_samples : int, optional
            Number of samples to display in the plot. If None, uses the default value.
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with electromagnetic and load torque traces.

        Examples
        --------
        >>> from ross import motor_example
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> t = np.arange(0, tf + dt, dt)
        >>> results = motor.run_direct_on_line(t)
        >>> fig = results.plot_torque()
        >>> fig = results.plot_torque(domain="frequency")
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
            n_shown_samples=n_shown_samples,
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
        n_shown_samples=None,
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
        n_shown_samples : int, optional
            Number of samples to display in the plot. If None, uses the default value.
        fig : plotly.graph_objects.Figure, optional
            Figure to add trace to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with rotor speed trace.

        Examples
        --------
        >>> from ross import motor_example
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> t = np.arange(0, tf + dt, dt)
        >>> results = motor.run_direct_on_line(t)
        >>> fig = results.plot_speed()
        >>> fig = results.plot_speed(domain="frequency", speed_units="rad/s")
        """
        if fig is None:
            fig = go.Figure()

        main_inputs = dict(
            result_dict={
                "Shaft Speed": Q_(self.speed, "rad/s").to(speed_units).m,
            },
            title="Motor operation: Shaft Speed",
            yaxis_title=f"Speed ({speed_units})",
            fig=fig,
            n_shown_samples=n_shown_samples,
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
        n_shown_samples=None,
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
        n_shown_samples : int, optional
            Number of samples to display in the plot. If None, uses the default value.
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with current traces in selected reference frame.

        Examples
        --------
        >>> from ross import motor_example
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> t = np.arange(0, tf + dt, dt)
        >>> results = motor.run_direct_on_line(t)
        >>> fig = results.plot_phase_currents(reference_frame="a-b-c")
        >>> fig = results.plot_phase_currents(reference_frame="d-q", domain="frequency")
        """
        if kwargs.get("title") is None:
            kwargs["title"] = "Motor operation: Stator Currents"

        current = PhaseResults(self.t, self.currents, "I", self.t_eval)

        if domain == "frequency":
            fig = current.plot_dfft(
                reference_frame=reference_frame,
                fig=fig,
                frequency_units=frequency_units,
                frequency_range=frequency_range,
                n_shown_samples=n_shown_samples,
                **kwargs,
            )
        else:
            fig = current.plot(
                reference_frame=reference_frame,
                fig=fig,
                n_shown_samples=n_shown_samples,
                **kwargs,
            )

        return fig

    def plot_phase_voltages(
        self,
        domain="time",
        frequency_units="Hz",
        frequency_range=None,
        n_shown_samples=None,
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
        n_shown_samples : int, optional
            Number of samples to display in the plot. If None, uses the default value.
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with phase voltage traces.

        Examples
        --------
        >>> from ross import motor_example
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> t = np.arange(0, tf + dt, dt)
        >>> results = motor.run_direct_on_line(t)
        >>> fig = results.plot_phase_voltages()
        >>> fig = results.plot_phase_voltages(domain="frequency")
        """
        if kwargs.get("title") is None:
            kwargs["title"] = "Motor operation: Stator Phase Voltages"

        voltage = PhaseResults(self.t, self.voltages, "V", self.t_eval)

        if domain == "frequency":
            fig = voltage.plot_dfft(
                fig=fig,
                frequency_units=frequency_units,
                frequency_range=frequency_range,
                n_shown_samples=n_shown_samples,
                **kwargs,
            )
        else:
            fig = voltage.plot(fig=fig, n_shown_samples=n_shown_samples, **kwargs)

        return fig

    def plot_line_voltages(
        self,
        domain="time",
        frequency_units="Hz",
        frequency_range=None,
        n_shown_samples=None,
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
        n_shown_samples : int, optional
            Number of samples to display in the plot. If None, uses the default value.
        fig : plotly.graph_objects.Figure, optional
            Figure to add traces to. If None, creates new figure.
        **kwargs
            Additional keyword arguments passed to figure layout.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure with line voltage traces.

        Examples
        --------
        >>> from ross import motor_example
        >>> motor = motor_example()
        >>> dt = 1e-3
        >>> tf = 1.0
        >>> t = np.arange(0, tf + dt, dt)
        >>> results = motor.run_direct_on_line(t)
        >>> fig = results.plot_line_voltages()
        >>> fig = results.plot_line_voltages(domain="frequency")
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
            n_shown_samples=n_shown_samples,
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
