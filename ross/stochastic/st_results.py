"""STOCHASTIC ROSS plotting module.

This module returns graphs for each type of analyses in st_rotor_assembly.py.
"""

import copy
import inspect
from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from warnings import warn

import numpy as np
import toml
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from ross.plotly_theme import tableau_colors
from ross.units import Q_

# set Plotly palette of colors
colors1 = px.colors.qualitative.Dark24
colors2 = px.colors.qualitative.Light24

__all__ = [
    "ST_CampbellResults",
    "ST_FrequencyResponseResults",
    "ST_TimeResponseResults",
    "ST_ForcedResponseResults",
    "ST_Results",
]


class ST_Results(ABC):
    """Results class.

    This class is a general abstract class to be implemented in other classes
    for post-processing results, in order to add saving and loading data options.
    """

    def save(self, file):
        """Save results in a .toml file.

        This function will save the simulation results to a .toml file.
        The file will have all the argument's names and values that are needed to
        reinstantiate the class.

        Parameters
        ----------
        file : str, pathlib.Path
            The name of the file the results will be saved in.

        Examples
        --------
        >>> # Example running a stochastic unbalance response
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> import ross.stochastic as srs

        >>> # Running an example
        >>> rotors = srs.st_rotor_example()
        >>> freq_range = np.linspace(0, 500, 31)
        >>> n = 3
        >>> m = np.random.uniform(0.001, 0.002, 10)
        >>> p = 0.0
        >>> results = rotors.run_unbalance_response(n, m, p, freq_range)

        >>> # create path for a temporary file
        >>> file = Path(tempdir) / 'results.toml'
        >>> results.save(file)
        """
        # get __init__ arguments
        signature = inspect.signature(self.__init__)
        args_list = list(signature.parameters)
        args = {arg: getattr(self, arg) for arg in args_list}
        try:
            data = toml.load(file)
        except FileNotFoundError:
            data = {}

        data[f"{self.__class__.__name__}"] = args
        with open(file, "w") as f:
            toml.dump(data, f, encoder=toml.TomlNumpyEncoder())

    @classmethod
    def read_toml_data(cls, data):
        """Read and parse data stored in a .toml file.

        The data passed to this method needs to be according to the
        format saved in the .toml file by the .save() method.

        Parameters
        ----------
        data : dict
            Dictionary obtained from toml.load().

        Returns
        -------
        The result object.
        """
        return cls(**data)

    @classmethod
    def load(cls, file):
        """Load results from a .toml file.

        This function will load the simulation results from a .toml file.
        The file must have all the argument's names and values that are needed to
        reinstantiate the class.

        Parameters
        ----------
        file : str, pathlib.Path
            The name of the file the results will be loaded from.

        Examples
        --------
        >>> # Example running a stochastic unbalance response
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> import ross.stochastic as srs

        >>> # Running an example
        >>> rotors = srs.st_rotor_example()
        >>> freq_range = np.linspace(0, 500, 31)
        >>> n = 3
        >>> m = np.random.uniform(0.001, 0.002, 10)
        >>> p = 0.0
        >>> results = rotors.run_unbalance_response(n, m, p, freq_range)

        >>> # create path for a temporary file
        >>> file = Path(tempdir) / 'results.toml'
        >>> results.save(file)

        >>> # Loading file
        >>> results2 = srs.ST_ForcedResponseResults.load(file)
        >>> results2.forced_resp.all() == results.forced_resp.all()
        True
        """
        data = toml.load(file)
        # extract single dictionary in the data
        data = list(data.values())[0]
        for key, value in data.items():
            if isinstance(value, Iterable):
                data[key] = np.array(value)
                if data[key].dtype == np.dtype("<U49"):
                    data[key] = np.array(value).astype(np.complex128)
        return cls.read_toml_data(data)


class ST_CampbellResults(ST_Results):
    """Store stochastic results and provide plots for Campbell Diagram.

    It's possible to visualize multiples harmonics in a single plot to check
    other speeds which also excite a specific natural frequency.
    Two options for plooting are available: Matplotlib and Bokeh. The user
    chooses between them using the attribute plot_type. The default is bokeh

    Parameters
    ----------
    speed_range : array
        Array with the speed range in rad/s.
    wd : array
        Array with the damped natural frequencies
    log_dec : array
        Array with the Logarithmic decrement

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with diagrams for frequency and log dec.
    """

    def __init__(self, speed_range, wd, log_dec, mode_type):
        self.speed_range = speed_range
        self.wd = wd
        self.log_dec = log_dec
        self.mode_type = mode_type

    def plot_nat_freq(
        self,
        percentile=[],
        conf_interval=[],
        harmonics=[1],
        frequency_units="rad/s",
        **kwargs,
    ):
        """Plot the damped natural frequencies vs frequency.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        wd = Q_(self.wd, "rad/s").to(frequency_units).m
        speed_range = Q_(self.speed_range, "rad/s").to(frequency_units).m
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        fig = go.Figure()
        x = np.concatenate((speed_range, speed_range[::-1]))

        for j, h in enumerate(harmonics):
            fig.add_trace(
                go.Scatter(
                    x=speed_range,
                    y=speed_range * h,
                    mode="lines",
                    name="{}x speed".format(h),
                    line=dict(width=3, color=colors1[j], dash="dashdot"),
                    legendgroup="speed{}".format(j),
                    hovertemplate=("Frequency: %{x:.3f}<br>" + "Frequency: %{y:.3f}"),
                )
            )

        for mode in ["Lateral", "Torsional", "Axial"]:
            mode_num = np.unique(np.where(self.mode_type == mode)[0])

            for n, j in enumerate(mode_num):
                fig.add_trace(
                    go.Scatter(
                        x=speed_range,
                        y=np.mean(wd[j], axis=1),
                        name=f"Mean - Mode {n + 1} ({mode})",
                        mode="lines",
                        line=dict(width=3, color=colors1[j]),
                        legendgroup=f"mean{j}",
                        hovertemplate=(
                            "Frequency: %{x:.3f}<br>" + "Frequency: %{y:.3f}"
                        ),
                    )
                )
                for i, p in enumerate(percentile):
                    fig.add_trace(
                        go.Scatter(
                            x=speed_range,
                            y=np.percentile(wd[j], p, axis=1),
                            opacity=0.6,
                            mode="lines",
                            line=dict(width=2.5, color=colors2[j]),
                            name=f"percentile: {p}%",
                            legendgroup=f"percentile{j}{i}",
                            hovertemplate=(
                                "Frequency: %{x:.3f}<br>" + "Frequency: %{y:.3f}"
                            ),
                        )
                    )
                for i, p in enumerate(conf_interval):
                    p1 = np.percentile(wd[j], 50 + p / 2, axis=1)
                    p2 = np.percentile(wd[j], 50 - p / 2, axis=1)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=np.concatenate((p1, p2[::-1])),
                            mode="lines",
                            line=dict(width=1, color=colors1[j]),
                            fill="toself",
                            fillcolor=colors1[j],
                            opacity=0.3,
                            name=f"confidence interval: {p}% - Mode {n + 1} ({mode})",
                            legendgroup="conf{}{}".format(j, i),
                            hovertemplate=(
                                "Frequency: %{x:.3f}<br>" + "Frequency: %{y:.3f}"
                            ),
                        )
                    )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(speed_range), np.max(speed_range)],
            exponentformat="none",
        )
        fig.update_yaxes(
            title_text=f"Natural Frequencies ({frequency_units})",
            range=[0, 1.1 * np.max(wd)],
        )
        fig.update_layout(**kwargs)

        return fig

    def plot_log_dec(
        self,
        percentile=[],
        conf_interval=[],
        frequency_units="rad/s",
        **kwargs,
    ):
        """Plot the log. decrement vs frequency.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)
        speed_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        fig = go.Figure()
        x = np.concatenate((speed_range, speed_range[::-1]))

        for mode in ["Lateral", "Torsional", "Axial"]:
            mode_num = np.unique(np.where(self.mode_type == mode)[0])

            for n, j in enumerate(mode_num):
                fig.add_trace(
                    go.Scatter(
                        x=speed_range,
                        y=np.mean(self.log_dec[j], axis=1),
                        opacity=1.0,
                        name=f"Mean - Mode {n + 1} ({mode})",
                        line=dict(width=3, color=colors1[j]),
                        legendgroup=f"mean{j}",
                        hovertemplate=("Frequency: %{x:.3f}<br>" + "Log Dec: %{y:.3f}"),
                    )
                )

                for i, p in enumerate(percentile):
                    fig.add_trace(
                        go.Scatter(
                            x=speed_range,
                            y=np.percentile(self.log_dec[j], p, axis=1),
                            opacity=0.6,
                            line=dict(width=2.5, color=colors2[j]),
                            name=f"percentile: {p}%",
                            legendgroup=f"percentile{j}{i}",
                            hoverinfo="none",
                        )
                    )

                for i, p in enumerate(conf_interval):
                    p1 = np.percentile(self.log_dec[j], 50 + p / 2, axis=1)
                    p2 = np.percentile(self.log_dec[j], 50 - p / 2, axis=1)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=np.concatenate((p1, p2[::-1])),
                            line=dict(width=1, color=colors1[j]),
                            fill="toself",
                            fillcolor=colors1[j],
                            opacity=0.3,
                            name=f"confidence interval: {p}% - Mode {n + 1} ({mode})",
                            legendgroup=f"conf{j}{i}",
                            hoverinfo="none",
                        )
                    )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(speed_range), np.max(speed_range)],
            exponentformat="none",
        )
        fig.update_yaxes(
            title_text="Logarithmic decrement",
        )
        fig.update_layout(**kwargs)

        return fig

    def plot(
        self,
        percentile=[],
        conf_interval=[],
        harmonics=[1],
        frequency_units="rad/s",
        freq_kwargs=None,
        logdec_kwargs=None,
        fig_kwargs=None,
    ):
        """Plot Campbell Diagram.

        This method plots Campbell Diagram.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        harmonics: list, optional
            List withe the harmonics to be plotted.
            The default is to plot 1x.
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        freq_kwargs : dict, optional
            Additional key word arguments can be passed to change the natural frequency
            vs frequency plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        logdec_kwargs : dict, optional
            Additional key word arguments can be passed to change the log. decrement
            vs frequency plot layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        fig_kwargs : dict, optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...). This kwargs override "freq_kwargs",
            "logdec_kwargs" dictionaries.
            *See Plotly Python make_subplots Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            Plotly figure with diagrams for frequency and log dec.
        """
        freq_kwargs = {} if freq_kwargs is None else copy.copy(freq_kwargs)
        logdec_kwargs = {} if logdec_kwargs is None else copy.copy(logdec_kwargs)
        fig_kwargs = {} if fig_kwargs is None else copy.copy(fig_kwargs)

        fig0 = self.plot_nat_freq(
            percentile, conf_interval, harmonics, frequency_units, **freq_kwargs
        )
        fig1 = self.plot_log_dec(
            percentile, conf_interval, frequency_units, **logdec_kwargs
        )

        fig = make_subplots(rows=1, cols=2)
        for data in fig0["data"]:
            fig.add_trace(data, 1, 1)
        for data in fig1["data"]:
            data.showlegend = False
            fig.add_trace(data, 1, 2)

        fig.update_xaxes(fig0.layout.xaxis, row=1, col=1)
        fig.update_yaxes(fig0.layout.yaxis, row=1, col=1)
        fig.update_xaxes(fig1.layout.xaxis, row=1, col=2)
        fig.update_yaxes(fig1.layout.yaxis, row=1, col=2)
        fig.update_layout(**fig_kwargs)

        return fig


class ST_FrequencyResponseResults(ST_Results):
    """Store stochastic results and provide plots for Frequency Response.

    Parameters
    ----------
    speed_range : array
        Array with the speed range in rad/s.
    magnitude : array
        Array with the frequencies, magnitude (dB) of the frequency
        response for each pair input/output.
    phase : array
        Array with the frequencies, phase of the frequency
        response for each pair input/output.

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with amplitude vs frequency phase angle vs frequency.
    """

    def __init__(self, speed_range, freq_resp, velc_resp, accl_resp):
        self.speed_range = speed_range
        self.freq_resp = freq_resp
        self.velc_resp = velc_resp
        self.accl_resp = accl_resp

    def plot_magnitude(
        self,
        percentile=[],
        conf_interval=[],
        frequency_units="rad/s",
        amplitude_units="m/N",
        fig=None,
        **kwargs,
    ):
        """Plot stochastic frequency response (magnitude) using Plotly.

        This method plots the frequency response magnitude given an output and
        an input using Plotly.
        It is possible to plot displacement, velocity and accelaration responses,
        depending on the unit entered in 'amplitude_units'. If '[length]/[force]',
        it displays the displacement; If '[speed]/[force]', it displays the velocity;
        If '[acceleration]/[force]', it displays the acceleration.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0% and 100% inclusive.
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units are:
                '[length]/[force]' - Displays the displacement;
                '[speed]/[force]' - Displays the velocity;
                '[acceleration]/[force]' - Displays the acceleration.
            Default is "m/N" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m/N)
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        speed_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        dummy_var = Q_(1, amplitude_units)
        if dummy_var.check("[length]/[force]"):
            mag = np.abs(self.freq_resp)
            mag = Q_(mag, "m/N").to(amplitude_units).m
            y_label = "Displacement"
        elif dummy_var.check("[speed]/[force]"):
            mag = np.abs(self.velc_resp)
            mag = Q_(mag, "m/s/N").to(amplitude_units).m
            y_label = "Velocity"
        elif dummy_var.check("[acceleration]/[force]"):
            mag = np.abs(self.accl_resp)
            mag = Q_(mag, "m/s**2/N").to(amplitude_units).m
            y_label = "Acceleration"
        else:
            raise ValueError(
                "Not supported unit. Options are '[length]/[force]', '[speed]/[force]', '[acceleration]/[force]'"
            )

        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=speed_range,
                y=np.mean(mag, axis=1),
                mode="lines",
                name="Mean",
                line=dict(width=3, color="black"),
                legendgroup="mean",
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatter(
                    x=speed_range,
                    y=np.percentile(mag, p, axis=1),
                    mode="lines",
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    legendgroup="percentile{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
                )
            )

        x = np.concatenate((speed_range, speed_range[::-1]))
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(mag, 50 + p / 2, axis=1)
            p2 = np.percentile(mag, 50 - p / 2, axis=1)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.concatenate((p1, p2[::-1])),
                    mode="lines",
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    legendgroup="conf{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
                )
            )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(speed_range), np.max(speed_range)],
        )
        fig.update_yaxes(title_text=f"{y_label} ({amplitude_units})")
        fig.update_layout(**kwargs)

        return fig

    def plot_phase(
        self,
        percentile=[],
        conf_interval=[],
        frequency_units="rad/s",
        amplitude_units="m/N",
        phase_units="rad",
        fig=None,
        **kwargs,
    ):
        """Plot stochastic frequency response (phase) using Plotly.

        This method plots the phase response given an output and an input
        using Plotly.
        It is possible to plot displacement, velocity and accelaration responses,
        depending on the unit entered in 'amplitude_units'. If '[length]/[force]',
        it displays the displacement; If '[speed]/[force]', it displays the velocity;
        If '[acceleration]/[force]', it displays the acceleration.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units are:
                '[length]/[force]' - Displays the displacement;
                '[speed]/[force]' - Displays the velocity;
                '[acceleration]/[force]' - Displays the acceleration.
            Default is "m/N" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m/N)
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        speed_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        dummy_var = Q_(1, amplitude_units)
        if dummy_var.check("[length]/[force]"):
            phase = np.angle(self.freq_resp)
        elif dummy_var.check("[speed]/[force]"):
            phase = np.angle(self.velc_resp)
        elif dummy_var.check("[acceleration]/[force]"):
            phase = np.angle(self.accl_resp)
        else:
            raise ValueError(
                "Not supported unit. Options are '[length]/[force]', '[speed]/[force]', '[acceleration]/[force]'"
            )

        phase = Q_(phase, "rad").to(phase_units).m

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=speed_range,
                y=np.mean(phase, axis=1),
                opacity=1.0,
                mode="lines",
                name="Mean",
                line=dict(width=3, color="black"),
                legendgroup="mean",
                hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatter(
                    x=speed_range,
                    y=np.percentile(phase, p, axis=1),
                    mode="lines",
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    legendgroup="percentile{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
                )
            )

        x = np.concatenate((speed_range, speed_range[::-1]))
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(phase, 50 + p / 2, axis=1)
            p2 = np.percentile(phase, 50 - p / 2, axis=1)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.concatenate((p1, p2[::-1])),
                    mode="lines",
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    legendgroup="conf{}".format(i),
                    hovertemplate=("Frequency: %{x:.2f}<br>" + "Phase: %{y:.2f}"),
                )
            )

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(speed_range), np.max(speed_range)],
        )
        fig.update_yaxes(title_text=f"Phase ({phase_units})")
        fig.update_layout(**kwargs)

        return fig

    def plot_polar_bode(
        self,
        percentile=[],
        conf_interval=[],
        frequency_units="rad/s",
        amplitude_units="m/N",
        phase_units="rad",
        fig=None,
        **kwargs,
    ):
        """Plot stochastic frequency response (polar) using Plotly.

        This method plots the frequency response (polar graph) given an output and
        an input using Plotly.
        It is possible to plot displacement, velocity and accelaration responses,
        depending on the unit entered in 'amplitude_units'. If '[length]/[force]',
        it displays the displacement; If '[speed]/[force]', it displays the velocity;
        If '[acceleration]/[force]', it displays the acceleration.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units are:
                '[length]/[force]' - Displays the displacement;
                '[speed]/[force]' - Displays the velocity;
                '[acceleration]/[force]' - Displays the acceleration.
            Default is "m/N" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m/N)
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        speed_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        dummy_var = Q_(1, amplitude_units)
        if dummy_var.check("[length]/[force]"):
            mag = np.abs(self.freq_resp)
            mag = Q_(mag, "m/N").to(amplitude_units).m
            phase = np.angle(self.freq_resp)
            y_label = "Displacement"
        elif dummy_var.check("[speed]/[force]"):
            mag = np.abs(self.velc_resp)
            mag = Q_(mag, "m/s/N").to(amplitude_units).m
            phase = np.angle(self.velc_resp)
            y_label = "Velocity"
        elif dummy_var.check("[acceleration]/[force]"):
            mag = np.abs(self.accl_resp)
            mag = Q_(mag, "m/s**2/N").to(amplitude_units).m
            phase = np.angle(self.accl_resp)
            y_label = "Acceleration"
        else:
            raise ValueError(
                "Not supported unit. Options are '[length]/[force]', '[speed]/[force]', '[acceleration]/[force]'"
            )

        phase = Q_(phase, "rad").to(phase_units).m

        if phase_units in ["rad", "radian", "radians"]:
            polar_theta_unit = "radians"
        else:
            polar_theta_unit = "degrees"

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=np.mean(mag, axis=1),
                theta=np.mean(phase, axis=1),
                customdata=speed_range,
                thetaunit="radians",
                line=dict(width=3.0, color="black"),
                name="Mean",
                legendgroup="mean",
                hovertemplate=(
                    "<b>Amplitude: %{r:.2e}</b><br>"
                    + "<b>Phase: %{theta:.2f}</b><br>"
                    + "<b>Frequency: %{customdata:.2f}</b>"
                ),
                **kwargs,
            )
        )
        for i, p in enumerate(percentile):
            fig.add_trace(
                go.Scatterpolar(
                    r=np.percentile(mag, p, axis=1),
                    theta=np.percentile(phase, p, axis=1),
                    customdata=speed_range,
                    thetaunit="radians",
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    legendgroup="percentile{}".format(i),
                    hovertemplate=(
                        "<b>Amplitude: %{r:.2e}</b><br>"
                        + "<b>Phase: %{theta:.2f}</b><br>"
                        + "<b>Frequency: %{customdata:.2f}</b>"
                    ),
                    **kwargs,
                )
            )
        for i, p in enumerate(conf_interval):
            p1 = np.percentile(mag, 50 + p / 2, axis=1)
            p2 = np.percentile(mag, 50 - p / 2, axis=1)
            p3 = np.percentile(phase, 50 + p / 2, axis=1)
            p4 = np.percentile(phase, 50 - p / 2, axis=1)
            fig.add_trace(
                go.Scatterpolar(
                    r=np.concatenate((p1, p2[::-1])),
                    theta=np.concatenate((p3, p4[::-1])),
                    thetaunit="radians",
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    legendgroup="conf{}".format(i),
                    **kwargs,
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title=dict(text=f"{y_label} ({amplitude_units})"),
                    exponentformat="power",
                ),
                angularaxis=dict(thetaunit=polar_theta_unit),
            ),
            **kwargs,
        )

        return fig

    def plot(
        self,
        percentile=[],
        conf_interval=[],
        frequency_units="rad/s",
        amplitude_units="m/N",
        phase_units="rad",
        fig=None,
        mag_kwargs=None,
        phase_kwargs=None,
        polar_kwargs=None,
        fig_kwargs=None,
    ):
        """Plot frequency response.

        This method plots the frequency and phase response given an output
        and an input.

        This method returns a subplot with:
            - Frequency vs Amplitude;
            - Frequency vs Phase Angle;
            - Polar plot Amplitude vs Phase Angle;

        Amplitude can be displacement, velocity or accelaration responses,
        depending on the unit entered in 'amplitude_units'. If '[length]/[force]',
        it displays the displacement; If '[speed]/[force]', it displays the velocity;
        If '[acceleration]/[force]', it displays the acceleration.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units are:
                '[length]/[force]' - Displays the displacement;
                '[speed]/[force]' - Displays the velocity;
                '[acceleration]/[force]' - Displays the acceleration.
            Default is "m/N" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m/N)
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        mag_kwargs : optional
            Additional key word arguments can be passed to change the magnitude plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        phase_kwargs : optional
            Additional key word arguments can be passed to change the phase plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        polar_kwargs : optional
            Additional key word arguments can be passed to change the polar plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        fig_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...). This kwargs override "mag_kwargs",
            "phase_kwargs" and "polar_kwargs" dictionaries.
            *See Plotly Python make_subplots Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.make_subplots()
            Plotly figure with amplitude vs frequency phase angle vs frequency.
        """
        mag_kwargs = {} if mag_kwargs is None else copy.copy(mag_kwargs)
        phase_kwargs = {} if phase_kwargs is None else copy.copy(phase_kwargs)
        polar_kwargs = {} if polar_kwargs is None else copy.copy(polar_kwargs)
        fig_kwargs = {} if fig_kwargs is None else copy.copy(fig_kwargs)

        fig0 = self.plot_magnitude(
            percentile,
            conf_interval,
            frequency_units,
            amplitude_units,
            None,
            **mag_kwargs,
        )
        fig1 = self.plot_phase(
            percentile,
            conf_interval,
            frequency_units,
            amplitude_units,
            phase_units,
            None,
            **phase_kwargs,
        )
        fig2 = self.plot_polar_bode(
            percentile,
            conf_interval,
            frequency_units,
            amplitude_units,
            phase_units,
            None,
            **polar_kwargs,
        )

        if fig is None:
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {"type": "polar", "rowspan": 2}], [{}, None]],
            )

        for data in fig0["data"]:
            fig.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            data.showlegend = False
            fig.add_trace(data, row=2, col=1)
        for data in fig2["data"]:
            data.showlegend = False
            fig.add_trace(data, row=1, col=2)

        fig.update_xaxes(fig0.layout.xaxis, row=1, col=1)
        fig.update_yaxes(fig0.layout.yaxis, row=1, col=1)
        fig.update_xaxes(fig1.layout.xaxis, row=2, col=1)
        fig.update_yaxes(fig1.layout.yaxis, row=2, col=1)
        fig.update_layout(
            polar=dict(
                radialaxis=fig2.layout.polar.radialaxis,
                angularaxis=fig2.layout.polar.angularaxis,
            ),
            **fig_kwargs,
        )

        return fig


class ST_TimeResponseResults(ST_Results):
    """Store stochastic results and provide plots for Time Response and Orbit Response.

    Parameters
    ----------
    t : 1-dimensional array
        Time array.
    yout : array
        System response.
    xout : array
        Time evolution of the state vector.
    nodes: array
        list with nodes from a rotor model.
    link_nodes: array
        list with nodes created with "n_link" from a rotor model.
    nodes_pos: array
        Rotor nodes axial positions.
    number_dof : int
        Number of degrees of freedom per shaft element's node

    Returns
    -------
    fig : Plotly graph_objects.Figure()
        The figure object with the plot.
    """

    def __init__(self, t, yout, xout, number_dof, nodes, link_nodes, nodes_pos):
        self.t = t
        self.yout = yout
        self.xout = xout
        self.nodes = nodes
        self.link_nodes = link_nodes
        self.nodes_pos = nodes_pos
        self.number_dof = number_dof

    def plot_1d(
        self,
        probe,
        percentile=[],
        conf_interval=[],
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        """Plot stochastic time response.

        This method plots the time response given a tuple of probes with their nodes
        and orientations.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        displacement_units : str, optional
            Displacement units.
            Default is 'm'.
        time_units : str
            Time units.
            Default is 's'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        nodes = self.nodes
        link_nodes = self.link_nodes
        ndof = self.number_dof

        if fig is None:
            fig = go.Figure()

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        for i, p in enumerate(probe):
            try:
                node = p.node
                angle = p.angle
                probe_tag = p.tag or p.get_label(i + 1)
                if p.direction == "axial":
                    continue
            except AttributeError:
                node = p[0]
                warn(
                    "The use of tuples in the probe argument is deprecated. Use the Probe class instead.",
                    DeprecationWarning,
                )
                try:
                    angle = Q_(p[1], probe_units).to("rad").m
                except TypeError:
                    angle = p[1]
                try:
                    probe_tag = p[2]
                except IndexError:
                    probe_tag = f"Probe {i + 1} - Node {p[0]}"

            fix_dof = (node - nodes[-1] - 1) * ndof // 2 if node in link_nodes else 0
            dofx = ndof * node - fix_dof
            dofy = ndof * node + 1 - fix_dof

            # fmt: off
            operator = np.array(
                [[np.cos(angle), np.sin(angle)],
                 [-np.sin(angle), np.cos(angle)]]
            )

            probe_resp = np.zeros_like(self.yout[:, :, 0])
            for j, y in enumerate(self.yout):
                _probe_resp = operator @ np.vstack((y[:, dofx], y[:, dofy]))
                probe_resp[j] = _probe_resp[0,:]
            # fmt: on

            fig.add_trace(
                go.Scatter(
                    x=Q_(self.t, "s").to(time_units).m,
                    y=Q_(np.mean(probe_resp, axis=0), "m").to(displacement_units).m,
                    mode="lines",
                    opacity=1.0,
                    name=f"{probe_tag} - Mean",
                    line=dict(width=3.0),
                    hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                )
            )
            for j, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatter(
                        x=Q_(self.t, "s").to(time_units).m,
                        y=Q_(np.percentile(probe_resp, p, axis=0), "m")
                        .to(displacement_units)
                        .m,
                        # y=np.percentile(probe_resp, p, axis=0),
                        mode="lines",
                        opacity=0.6,
                        line=dict(width=2.5),
                        name=f"{probe_tag} - percentile: {p}%",
                        hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                    )
                )

            x = np.concatenate((self.t, self.t[::-1]))
            for j, p in enumerate(conf_interval):
                p1 = np.percentile(probe_resp, 50 + p / 2, axis=0)
                p2 = np.percentile(probe_resp, 50 - p / 2, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=Q_(x, "s").to(time_units).m,
                        y=Q_(np.concatenate((p1, p2[::-1])), "m")
                        .to(displacement_units)
                        .m,
                        mode="lines",
                        line=dict(width=1, color=colors1[j]),
                        fill="toself",
                        fillcolor=colors1[j],
                        opacity=0.5,
                        name=f"{probe_tag} - confidence interval: {p}%",
                        hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                    )
                )

        fig.update_xaxes(title_text=f"Time ({time_units})")
        fig.update_yaxes(title_text=f"Amplitude ({displacement_units})")
        fig.update_layout(**kwargs)

        return fig

    def plot_2d(
        self,
        node,
        percentile=[],
        conf_interval=[],
        displacement_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot orbit response (2D).

        This function plots orbits for a given node on the rotor system in a 2D view.

        Parameters
        ----------
        node : int
            Select the node to display the respective orbit response.
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        displacement_units : str, optional
            Displacement units.
            Default is 'm'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        nodes = self.nodes
        link_nodes = self.link_nodes
        ndof = self.number_dof

        fix_dof = (node - nodes[-1] - 1) * ndof // 2 if node in link_nodes else 0
        dofx = ndof * node - fix_dof
        dofy = ndof * node + 1 - fix_dof

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        if fig is None:
            fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=Q_(np.mean(self.yout[..., dofx], axis=0), "m")
                .to(displacement_units)
                .m,
                y=Q_(np.mean(self.yout[..., dofy], axis=0), "m")
                .to(displacement_units)
                .m,
                mode="lines",
                opacity=1.0,
                name="Mean",
                line=dict(width=3, color="black"),
                hovertemplate="X - Amplitude: %{x:.2e}<br>" + "Y - Amplitude: %{y:.2e}",
            )
        )

        for i, p in enumerate(percentile):
            p1 = np.percentile(self.yout[..., dofx], p, axis=0)
            p2 = np.percentile(self.yout[..., dofy], p, axis=0)

            fig.add_trace(
                go.Scatter(
                    x=Q_(p1, "m").to(displacement_units).m,
                    y=Q_(p2, "m").to(displacement_units).m,
                    mode="lines",
                    opacity=0.6,
                    line=dict(width=2.5, color=colors2[i]),
                    name="percentile: {}%".format(p),
                    hovertemplate="X - Amplitude: %{x:.2e}<br>"
                    + "Y - Amplitude: %{y:.2e}",
                )
            )

        for i, p in enumerate(conf_interval):
            p1 = np.percentile(self.yout[..., dofx], 50 + p / 2, axis=0)
            p2 = np.percentile(self.yout[..., dofx], 50 - p / 2, axis=0)
            p3 = np.percentile(self.yout[..., dofy], 50 + p / 2, axis=0)
            p4 = np.percentile(self.yout[..., dofy], 50 - p / 2, axis=0)

            fig.add_trace(
                go.Scatter(
                    x=Q_(np.concatenate((p1, p2[::-1])), "m").to(displacement_units).m,
                    y=Q_(np.concatenate((p3, p4[::-1])), "m").to(displacement_units).m,
                    mode="lines",
                    line=dict(width=1, color=colors1[i]),
                    fill="toself",
                    fillcolor=colors1[i],
                    opacity=0.5,
                    name="confidence interval: {}%".format(p),
                    hovertemplate="X - Amplitude: %{x:.2e}<br>"
                    + "Y - Amplitude: %{y:.2e}",
                )
            )

        fig.update_xaxes(title_text=f"Amplitude ({displacement_units}) - X direction")
        fig.update_yaxes(title_text=f"Amplitude ({displacement_units}) - Y direction")
        fig.update_layout(
            title=dict(text="Response for node {}".format(node)), **kwargs
        )

        return fig

    def plot_3d(
        self,
        percentile=[],
        conf_interval=[],
        displacement_units="m",
        rotor_length_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot orbit response (3D).

        This function plots orbits for each node on the rotor system in a 3D view.

        Parameters
        ----------
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        displacement_units : str
            Displacement units.
            Default is 'm'.
        rotor_length_units : str
            Rotor Length units.
            Default is 'm'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. hoverlabel_align="center", ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        nodes_pos = self.nodes_pos
        nodes = self.nodes
        ndof = self.number_dof

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        if fig is None:
            fig = go.Figure()

        line = np.zeros(len(self.nodes_pos))
        fig.add_trace(
            go.Scatter3d(
                x=Q_(nodes_pos, "m").to(rotor_length_units).m,
                y=line,
                z=line,
                mode="lines",
                line=dict(width=2.0, color="black", dash="dashdot"),
                showlegend=False,
            )
        )
        for j, n in enumerate(nodes):
            dofx = ndof * n
            dofy = ndof * n + 1

            x = np.ones(self.yout.shape[1]) * self.nodes_pos[n]
            fig.add_trace(
                go.Scatter3d(
                    x=Q_(x, "m").to(rotor_length_units).m,
                    y=Q_(np.mean(self.yout[..., dofx], axis=0), "m")
                    .to(displacement_units)
                    .m,
                    z=Q_(np.mean(self.yout[..., dofy], axis=0), "m")
                    .to(displacement_units)
                    .m,
                    mode="lines",
                    line=dict(width=5, color="black"),
                    name="Mean",
                    legendgroup="mean",
                    showlegend=True if j == 0 else False,
                    hovertemplate=(
                        "Nodal Position: %{x:.2f}<br>"
                        + "X - Amplitude: %{y:.2e}<br>"
                        + "Y - Amplitude: %{z:.2e}"
                    ),
                )
            )
            for i, p in enumerate(percentile):
                p1 = np.percentile(self.yout[..., dofx], p, axis=0)
                p2 = np.percentile(self.yout[..., dofy], p, axis=0)
                fig.add_trace(
                    go.Scatter3d(
                        x=Q_(x, "m").to(rotor_length_units).m,
                        y=Q_(p1, "m").to(displacement_units).m,
                        z=Q_(p2, "m").to(displacement_units).m,
                        mode="lines",
                        opacity=1.0,
                        name="percentile: {}%".format(p),
                        line=dict(width=3, color=colors2[i]),
                        legendgroup="perc{}".format(p),
                        showlegend=True if j == 0 else False,
                        hovertemplate=(
                            "Nodal Position: %{x:.2f}<br>"
                            + "X - Amplitude: %{y:.2e}<br>"
                            + "Y - Amplitude: %{z:.2e}"
                        ),
                    )
                )

            for i, p in enumerate(conf_interval):
                p1 = np.percentile(self.yout[..., dofx], 50 + p / 2, axis=0)
                p2 = np.percentile(self.yout[..., dofx], 50 - p / 2, axis=0)
                p3 = np.percentile(self.yout[..., dofy], 50 + p / 2, axis=0)
                p4 = np.percentile(self.yout[..., dofy], 50 - p / 2, axis=0)
                fig.add_trace(
                    go.Scatter3d(
                        x=Q_(x, "m").to(rotor_length_units).m,
                        y=Q_(p1, "m").to(displacement_units).m,
                        z=Q_(p3, "m").to(displacement_units).m,
                        mode="lines",
                        line=dict(width=3.5, color=colors1[i]),
                        opacity=0.6,
                        name="confidence interval: {}%".format(p),
                        legendgroup="conf_interval{}".format(p),
                        showlegend=False,
                        hovertemplate=(
                            "Nodal Position: %{x:.2f}<br>"
                            + "X - Amplitude: %{y:.2e}<br>"
                            + "Y - Amplitude: %{z:.2e}"
                        ),
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=Q_(p2, "m").to(displacement_units).m,
                        z=Q_(p4, "m").to(displacement_units).m,
                        mode="lines",
                        line=dict(width=3.5, color=colors1[i]),
                        opacity=0.6,
                        name="confidence interval: {}%".format(p),
                        legendgroup="conf_interval{}".format(p),
                        showlegend=False,
                        hovertemplate=(
                            "Nodal Position: %{x:.2f}<br>"
                            + "X - Amplitude: %{y:.2e}<br>"
                            + "Y - Amplitude: %{z:.2e}"
                        ),
                    )
                )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title=dict(text=f"Rotor Length ({rotor_length_units})"),
                    showspikes=False,
                ),
                yaxis=dict(
                    title=dict(text=f"Amplitude - X ({displacement_units})"),
                    showspikes=False,
                ),
                zaxis=dict(
                    title=dict(text=f"Amplitude - Y ({displacement_units})"),
                    showspikes=False,
                ),
            ),
            **kwargs,
        )

        return fig


class ST_ForcedResponseResults(ST_Results):
    """Store stochastic results and provide plots for Forced Response.

    Parameters
    ----------
    force_resp : array
        Array with the force response for each node for each frequency.
    frequency_range : array
        Array with the frequencies.
    velc_resp : array
        Array with the forced response (velocity) for each node for each frequency.
    accl_resp : array
        Array with the forced response (acceleration) for each node for each frequency.
    number_dof = int
        Number of degrees of freedom per shaft element's node.
    nodes : list
        List of shaft nodes.
    link_nodes : list
        List of n_link nodes.

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with amplitude vs frequency phase angle vs frequency.
    """

    def __init__(
        self,
        forced_resp,
        velc_resp,
        accl_resp,
        frequency_range,
        number_dof,
        nodes,
        link_nodes,
    ):
        self.forced_resp = forced_resp
        self.velc_resp = velc_resp
        self.accl_resp = accl_resp
        self.frequency_range = frequency_range
        self.number_dof = number_dof
        self.nodes = nodes
        self.link_nodes = link_nodes

        self.default_units = {
            "[length]": ["m", "forced_resp"],
            "[length] / [time]": ["m/s", "velc_resp"],
            "[length] / [time] ** 2": ["m/s**2", "accl_resp"],
        }

    def _calculate_major_axis_per_node(self, node, angle, amplitude_units="m"):
        """Calculate the major axis for a node for each frequency.

        Parameters
        ----------
        node : float
            A node from the rotor model.
        angle : float, str
            The orientation angle of the axis.
            Options are:
                float : angle in rad capture the response in a probe orientation;
                str : "major" to capture the response for the major axis;
                str : "minor" to capture the response for the minor axis.
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)

        Returns
        -------
        major_axis_vector : np.ndarray
            major_axis_vector[:, 0, :] = axis angle
            major_axis_vector[:, 1, :] = axis vector response for the input angle
            major_axis_vector[:, 2, :] = phase response for the input angle
        """
        ndof = self.number_dof
        nodes = self.nodes
        link_nodes = self.link_nodes

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            response = self.__dict__[self.default_units[unit_type][1]]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        size = response.shape[0]
        major_axis_vector = np.zeros(
            (size, 3, len(self.frequency_range)), dtype=complex
        )

        fix_dof = (node - nodes[-1] - 1) * ndof // 2 if node in link_nodes else 0
        dofx = ndof * node - fix_dof
        dofy = ndof * node + 1 - fix_dof

        # Relative angle between probes (90°)
        Rel_ang = np.exp(1j * np.pi / 2)

        for j in range(size):
            for i, f in enumerate(self.frequency_range):
                # Foward and Backward vectors
                fow = response[j, dofx, i] / 2 + Rel_ang * response[j, dofy, i] / 2
                back = (
                    np.conj(response[j, dofx, i]) / 2
                    + Rel_ang * np.conj(response[j, dofy, i]) / 2
                )

                ang_fow = np.angle(fow)
                if ang_fow < 0:
                    ang_fow += 2 * np.pi

                ang_back = np.angle(back)
                if ang_back < 0:
                    ang_back += 2 * np.pi

                if angle == "major":
                    # Major axis angle
                    axis_angle = (ang_back - ang_fow) / 2
                    if axis_angle > np.pi:
                        axis_angle -= np.pi

                elif angle == "minor":
                    # Minor axis angle
                    axis_angle = (ang_back - ang_fow + np.pi) / 2
                    if axis_angle > np.pi:
                        axis_angle -= np.pi

                else:
                    axis_angle = angle

                major_axis_vector[j, 0, i] = axis_angle
                major_axis_vector[j, 1, i] = np.abs(
                    fow * np.exp(1j * axis_angle) + back * np.exp(-1j * axis_angle)
                )
                major_axis_vector[j, 2, i] = np.angle(
                    fow * np.exp(1j * axis_angle) + back * np.exp(-1j * axis_angle)
                )

        return major_axis_vector

    def plot_magnitude(
        self,
        probe,
        percentile=[],
        conf_interval=[],
        probe_units="rad",
        frequency_units="rad/s",
        amplitude_units="m",
        fig=None,
        **kwargs,
    ):
        """Plot stochastic frequency response.

        This method plots the unbalance response magnitude.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0% and 100% inclusive.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            Bokeh plot axes with magnitude plot.
        """
        frequency_range = Q_(self.frequency_range, "rad/s").to(frequency_units).m

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            base_unit = self.default_units[unit_type][0]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        if fig is None:
            fig = go.Figure()

        color_i = 0
        color_p = 0

        for i, p in enumerate(probe):
            try:
                node = p.node
                angle = p.angle
                probe_tag = p.tag or p.get_label(i + 1)
                if p.direction == "axial":
                    continue
            except AttributeError:
                node = p[0]
                warn(
                    "The use of tuples in the probe argument is deprecated. Use the Probe class instead.",
                    DeprecationWarning,
                )
                try:
                    angle = Q_(p[1], probe_units).to("rad").m
                except TypeError:
                    angle = p[1]
                try:
                    probe_tag = p[2]
                except IndexError:
                    probe_tag = f"Probe {i + 1} - Node {p[0]}"

            vector = self._calculate_major_axis_per_node(
                node=node, angle=angle, amplitude_units=amplitude_units
            )[:, 1, :]

            fig.add_trace(
                go.Scatter(
                    x=frequency_range,
                    y=Q_(np.mean(np.abs(vector), axis=0), base_unit)
                    .to(amplitude_units)
                    .m,
                    opacity=1.0,
                    mode="lines",
                    line=dict(width=3, color=list(tableau_colors)[i]),
                    name=f"{probe_tag} - Mean",
                    legendgroup=f"{probe_tag} - Mean",
                    hovertemplate="Frequency: %{x:.2f}<br>Amplitude: %{y:.2e}",
                )
            )
            for j, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatter(
                        x=frequency_range,
                        y=Q_(np.percentile(np.abs(vector), p, axis=0), base_unit)
                        .to(amplitude_units)
                        .m,
                        opacity=0.6,
                        mode="lines",
                        line=dict(width=2.5, color=colors1[color_p]),
                        name=f"{probe_tag} - percentile: {p}%",
                        legendgroup=f"{probe_tag} - percentile: {p}%",
                        hovertemplate="Frequency: %{x:.2f}<br>Amplitude: %{y:.2e}",
                    )
                )
                color_p += 1

            x = np.concatenate((frequency_range, frequency_range[::-1]))
            for j, p in enumerate(conf_interval):
                p1 = np.percentile(np.abs(vector), 50 + p / 2, axis=0)
                p2 = np.percentile(np.abs(vector), 50 - p / 2, axis=0)

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=Q_(np.concatenate((p1, p2[::-1])), base_unit)
                        .to(amplitude_units)
                        .m,
                        mode="lines",
                        line=dict(width=1, color=colors2[color_i]),
                        fill="toself",
                        fillcolor=colors2[color_i],
                        opacity=0.5,
                        name=f"{probe_tag} - confidence interval: {p}%",
                        legendgroup=f"{probe_tag} - confidence interval: {p}%",
                        hovertemplate="Frequency: %{x:.2f}<br>Amplitude: %{y:.2e}",
                    )
                )
                color_i += 1

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(frequency_range), np.max(frequency_range)],
        )
        fig.update_yaxes(
            title_text=f"Amplitude ({amplitude_units})", exponentformat="power"
        )
        fig.update_layout(**kwargs)

        return fig

    def plot_phase(
        self,
        probe,
        percentile=[],
        conf_interval=[],
        probe_units="rad",
        frequency_units="rad/s",
        amplitude_units="m",
        phase_units="rad",
        fig=None,
        **kwargs,
    ):
        """Plot stochastic frequency response.

        This method plots the phase response given a set of probes.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        frequency_range = Q_(self.frequency_range, "rad/s").to(frequency_units).m

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        if fig is None:
            fig = go.Figure()

        color_p = 0
        color_i = 0

        x = np.concatenate((frequency_range, frequency_range[::-1]))
        for i, p in enumerate(probe):
            try:
                node = p.node
                angle = p.angle
                probe_tag = p.tag or p.get_label(i + 1)
                if p.direction == "axial":
                    continue
            except AttributeError:
                node = p[0]
                warn(
                    "The use of tuples in the probe argument is deprecated. Use the Probe class instead.",
                    DeprecationWarning,
                )
                try:
                    angle = Q_(p[1], probe_units).to("rad").m
                except TypeError:
                    angle = p[1]
                try:
                    probe_tag = p[2]
                except IndexError:
                    probe_tag = f"Probe {i + 1} - Node {p[0]}"

            vector = self._calculate_major_axis_per_node(
                node=node, angle=angle, amplitude_units=amplitude_units
            )[:, 2, :]

            probe_phase = np.real(vector)
            probe_phase = Q_(probe_phase, "rad").to(phase_units).m

            fig.add_trace(
                go.Scatter(
                    x=frequency_range,
                    y=np.mean(probe_phase, axis=0),
                    opacity=1.0,
                    mode="lines",
                    line=dict(width=3, color=list(tableau_colors)[i]),
                    name=f"{probe_tag} - Mean",
                    legendgroup=f"{probe_tag} - Mean",
                    hovertemplate="Frequency: %{x:.2f}<br>Phase: %{y:.2f}",
                )
            )
            for j, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatter(
                        x=frequency_range,
                        y=np.percentile(probe_phase, p, axis=0),
                        opacity=0.6,
                        mode="lines",
                        line=dict(width=2.5, color=colors1[color_p]),
                        name=f"{probe_tag} - percentile: {p}%",
                        legendgroup=f"{probe_tag} - percentile: {p}%",
                        hovertemplate="Frequency: %{x:.2f}<br>Phase: %{y:.2f}",
                    )
                )
                color_p += 1

            for j, p in enumerate(conf_interval):
                p1 = np.percentile(probe_phase, 50 + p / 2, axis=0)
                p2 = np.percentile(probe_phase, 50 - p / 2, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.concatenate((p1, p2[::-1])),
                        mode="lines",
                        line=dict(width=1, color=colors2[color_i]),
                        fill="toself",
                        fillcolor=colors2[color_i],
                        opacity=0.5,
                        name=f"{probe_tag} - confidence interval: {p}%",
                        legendgroup=f"{probe_tag} - confidence interval: {p}%",
                        hovertemplate="Frequency: %{x:.2f}<br>Phase: %{y:.2f}",
                    )
                )
                color_i += 1

        fig.update_xaxes(
            title_text=f"Frequency ({frequency_units})",
            range=[np.min(frequency_range), np.max(frequency_range)],
        )
        fig.update_yaxes(title_text=f"Phase ({phase_units})")
        fig.update_layout(**kwargs)

        return fig

    def plot_polar_bode(
        self,
        probe,
        percentile=[],
        conf_interval=[],
        probe_units="rad",
        frequency_units="rad/s",
        amplitude_units="m",
        phase_units="rad",
        fig=None,
        **kwargs,
    ):
        """Plot polar forced response using Plotly.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
        percentile : list, optional
            Sequence of percentiles to compute, which must be between
            0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be between
            0 and 100 inclusive.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
        phase_units : str, optional
            Units for the x axis.
            Default is "rad"
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """
        frequency_range = Q_(self.frequency_range, "rad/s").to(frequency_units).m

        conf_interval = np.sort(conf_interval)
        percentile = np.sort(percentile)

        unit_type = str(Q_(1, amplitude_units).dimensionality)
        try:
            base_unit = self.default_units[unit_type][0]
        except KeyError:
            raise ValueError(
                "Not supported unit. Dimensionality options are '[length]', '[speed]', '[acceleration]'"
            )

        if phase_units in ["rad", "radian", "radians"]:
            polar_theta_unit = "radians"
        else:
            polar_theta_unit = "degrees"

        if fig is None:
            fig = go.Figure()

        color_p = 0
        color_i = 0

        for i, p in enumerate(probe):
            try:
                node = p.node
                angle = p.angle
                probe_tag = p.tag or p.get_label(i + 1)
                if p.direction == "axial":
                    continue
            except AttributeError:
                node = p[0]
                warn(
                    "The use of tuples in the probe argument is deprecated. Use the Probe class instead.",
                    DeprecationWarning,
                )
                try:
                    angle = Q_(p[1], probe_units).to("rad").m
                except TypeError:
                    angle = p[1]
                try:
                    probe_tag = p[2]
                except IndexError:
                    probe_tag = f"Probe {i + 1} - Node {p[0]}"

            mag = self._calculate_major_axis_per_node(
                node=node, angle=angle, amplitude_units=amplitude_units
            )[:, 1, :]
            probe_phase = self._calculate_major_axis_per_node(
                node=node, angle=angle, amplitude_units=amplitude_units
            )[:, 2, :]

            probe_phase = np.real(probe_phase)
            probe_phase = Q_(probe_phase, "rad").to(phase_units).m

            fig.add_trace(
                go.Scatterpolar(
                    r=Q_(np.mean(np.abs(mag), axis=0), base_unit).to(amplitude_units).m,
                    theta=np.mean(probe_phase, axis=0),
                    customdata=frequency_range,
                    thetaunit=polar_theta_unit,
                    mode="lines",
                    line=dict(width=3.0, color=list(tableau_colors)[i]),
                    name=f"{probe_tag} - Mean",
                    legendgroup=f"{probe_tag} - Mean",
                    hovertemplate=(
                        "<b>Amplitude: %{r:.2e}</b><br>"
                        + "<b>Phase: %{theta:.2f}</b><br>"
                        + "<b>Frequency: %{customdata:.2f}</b>"
                    ),
                )
            )
            for j, p in enumerate(percentile):
                fig.add_trace(
                    go.Scatterpolar(
                        r=Q_(np.percentile(np.abs(mag), p, axis=0), base_unit)
                        .to(amplitude_units)
                        .m,
                        theta=np.percentile(probe_phase, p, axis=0),
                        customdata=frequency_range,
                        thetaunit=polar_theta_unit,
                        opacity=0.6,
                        mode="lines",
                        line=dict(width=2.5, color=colors1[color_p]),
                        name=f"{probe_tag} - percentile: {p}%",
                        legendgroup=f"{probe_tag} - percentile{p}",
                        hovertemplate=(
                            "<b>Amplitude: %{r:.2e}</b><br>"
                            + "<b>Phase: %{theta:.2f}</b><br>"
                            + "<b>Frequency: %{customdata:.2f}</b>"
                        ),
                    )
                )
                color_p += 1

            for j, p in enumerate(conf_interval):
                # fmt: off
                p1 = Q_(np.percentile(np.abs(mag), 50 + p / 2, axis=0), base_unit).to(amplitude_units).m
                p2 = Q_(np.percentile(np.abs(mag), 50 - p / 2, axis=0), base_unit).to(amplitude_units).m
                p3 = np.percentile(probe_phase, 50 + p / 2, axis=0)
                p4 = np.percentile(probe_phase, 50 - p / 2, axis=0)
                # fmt: on
                fig.add_trace(
                    go.Scatterpolar(
                        r=np.concatenate((p1, p2[::-1])),
                        theta=np.concatenate((p3, p4[::-1])),
                        thetaunit=polar_theta_unit,
                        mode="lines",
                        line=dict(width=1, color=colors2[color_i]),
                        fill="toself",
                        fillcolor=colors2[color_i],
                        opacity=0.5,
                        name=f"Probe {i + 1} - confidence interval: {p}%",
                        legendgroup=f"Probe {i + 1} - confidence interval: {p}%",
                    )
                )
                color_i += 1

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title=dict(text=f"Amplitude ({amplitude_units})"),
                ),
                angularaxis=dict(thetaunit=polar_theta_unit),
            ),
            **kwargs,
        )

        return fig

    def plot(
        self,
        probe,
        percentile=[],
        conf_interval=[],
        probe_units="rad",
        frequency_units="rad/s",
        amplitude_units="m",
        phase_units="rad",
        mag_kwargs=None,
        phase_kwargs=None,
        polar_kwargs=None,
        subplot_kwargs=None,
    ):
        """Plot stochastic forced response using Plotly.

        This method plots the forced response given a set of probes.

        Parameters
        ----------
        probe : list of tuples
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
        percentile : list, optional
            Sequence of percentiles to compute, which must be
            between 0 and 100 inclusive.
        conf_interval : list, optional
            Sequence of confidence intervals to compute, which must be
            between 0 and 100 inclusive.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        frequency_units : str, optional
            Frequency units.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the y axis.
            Acceptable units dimensionality are:
                '[length]' - Displays the displacement;
                '[speed]' - Displays the velocity;
                '[acceleration]' - Displays the acceleration.
            Default is "m" 0 to peak.
            To use peak to peak use the prefix 'pkpk_' (e.g. pkpk_m)
        phase_units : str, optional
            Phase units.
            Default is "rad"
        mag_kwargs : optional
            Additional key word arguments can be passed to change the magnitude plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        phase_kwargs : optional
            Additional key word arguments can be passed to change the phase plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        polar_kwargs : optional
            Additional key word arguments can be passed to change the polar plot
            layout only (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.
        subplot_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...). This kwargs override "mag_kwargs" and
            "phase_kwargs" dictionaries.
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        subplots : Plotly graph_objects.make_subplots()
            Plotly figure with amplitude vs frequency phase angle vs frequency.
        """
        mag_kwargs = {} if mag_kwargs is None else copy.copy(mag_kwargs)
        phase_kwargs = {} if phase_kwargs is None else copy.copy(phase_kwargs)
        polar_kwargs = {} if polar_kwargs is None else copy.copy(polar_kwargs)
        subplot_kwargs = {} if subplot_kwargs is None else copy.copy(subplot_kwargs)

        # fmt: off
        fig0 = self.plot_magnitude(
            probe, percentile, conf_interval, probe_units, frequency_units, amplitude_units, None, **mag_kwargs
        )
        fig1 = self.plot_phase(
            probe, percentile, conf_interval, probe_units, frequency_units, amplitude_units, phase_units, None, **phase_kwargs
        )
        fig2 = self.plot_polar_bode(
            probe, percentile, conf_interval, probe_units, frequency_units, amplitude_units, phase_units, None, **polar_kwargs,
        )
        # fmt: on

        fig = make_subplots(
            rows=2, cols=2, specs=[[{}, {"type": "polar", "rowspan": 2}], [{}, None]]
        )

        for data in fig0["data"]:
            data.showlegend = False
            fig.add_trace(data, row=1, col=1)
        for data in fig1["data"]:
            data.showlegend = False
            fig.add_trace(data, row=2, col=1)
        for data in fig2["data"]:
            fig.add_trace(data, row=1, col=2)

        fig.update_xaxes(fig0.layout.xaxis, row=1, col=1)
        fig.update_yaxes(fig0.layout.yaxis, row=1, col=1)
        fig.update_xaxes(fig1.layout.xaxis, row=2, col=1)
        fig.update_yaxes(fig1.layout.yaxis, row=2, col=1)
        fig.update_layout(
            polar=dict(
                radialaxis=fig2.layout.polar.radialaxis,
                angularaxis=fig2.layout.polar.angularaxis,
            ),
        )

        return fig
