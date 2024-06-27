from abc import ABC, abstractmethod

import numpy as np
import plotly.graph_objects as go
import scipy as sp
from warnings import warn

from ross.results import TimeResponseResults
from ross.units import Q_

__all__ = ["Fault"]


class Fault(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def run_time_response(self):
        results = TimeResponseResults(
            rotor=self.rotor,
            t=self.time_vector,
            yout=self.response.T,
            xout=[],
        )
        return results

    def plot_dfft(self, probe, probe_units="rad", range_freq=None, fig=None, **kwargs):
        """Plot response in frequency domain (dFFT - discrete Fourier Transform) using Plotly.

        Parameters
        ----------
        probe : list
            List with rs.Probe objects.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        range_freq : list, optional
            Units for the x axis.
            Default is "Hz"
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
        if fig is None:
            fig = go.Figure()

        num_dof = self.rotor.number_dof

        for i, p in enumerate(probe):
            probe_direction = "radial"
            try:
                node = p.node
                angle = p.angle
                probe_tag = p.tag or p.get_label(i + 1)
                if p.direction == "axial":
                    if num_dof == 6:
                        probe_direction = p.direction
                    else:
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
                    probe_tag = f"Probe {i+1} - Node {p[0]}"

            row, cols = self.response.shape
            init_step = int(2 * cols / 3)

            if probe_direction == "radial":
                dofx = num_dof * node
                dofy = num_dof * node + 1

                # fmt: off
                operator = np.array(
                    [[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]]
                )

                _probe_resp = operator @ np.vstack((self.response[dofx, init_step:], self.response[dofy, init_step:]))
                probe_resp = _probe_resp[0,:]
                # fmt: on
            else:
                dofz = num_dof * node + 2
                probe_resp = self.response[dofz, init_step:]

            amp, freq = self._dfft(probe_resp, self.dt)

            if range_freq is not None:
                amp = amp[(freq >= range_freq[0]) & (freq <= range_freq[1])]
                freq = freq[(freq >= range_freq[0]) & (freq <= range_freq[1])]

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=amp,
                    mode="lines",
                    name=probe_tag,
                    legendgroup=probe_tag,
                    showlegend=True,
                    hovertemplate=f"Frequency (Hz): %{{x:.2f}}<br>Amplitude (m): %{{y:.2e}}",
                )
            )

        fig.update_xaxes(title_text=f"Frequency (Hz)")
        fig.update_yaxes(title_text=f"Amplitude (m)")
        fig.update_layout(**kwargs)

        return fig

    def _dfft(self, x, dt):
        """Calculate dFFT - discrete Fourier Transform.

        Parameters
        ----------
        x : np.array
            Magnitude of the response in time domain.
            Default is "m".
        dt : int
            Time step.
            Default is "s".

        Returns
        -------
        x_amp : np.array
            Amplitude of the response in frequency domain.
            Default is "m".
        freq : np.array
            Frequency range.
            Default is "Hz".
        """
        b = np.floor(len(x) / 2)
        c = len(x)
        df = 1 / (c * dt)

        x_amp = sp.fft.fft(x)[: int(b)]
        x_amp = x_amp * 2 / c
        x_phase = np.angle(x_amp)
        x_amp = np.abs(x_amp)

        freq = np.arange(0, df * b, df)
        freq = freq[: int(b)]  # Frequency vector

        return x_amp, freq
