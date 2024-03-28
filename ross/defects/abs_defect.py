from abc import ABC, abstractmethod

import numpy as np
import plotly.graph_objects as go
import scipy as sp

from ross.results import TimeResponseResults
from ross.units import Q_

__all__ = ["Defect"]


class Defect(ABC):
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
        probe : list of tuples
            List with tuples (node, orientation angle, tag).
            node : int
                indicate the node where the probe is located.
            orientation : float
                probe orientation angle about the shaft. The 0 refers to +X direction.
            tag : str, optional
                probe tag to be displayed at the legend.
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

        for i, p in enumerate(probe):
            dofx = p[0] * self.rotor.number_dof
            dofy = p[0] * self.rotor.number_dof + 1
            angle = Q_(p[1], probe_units).to("rad").m

            # fmt: off
            operator = np.array(
                [[np.cos(angle), np.sin(angle)],
                 [-np.sin(angle), np.cos(angle)]]
            )
            row, cols = self.response.shape
            _probe_resp = operator @ np.vstack((self.response[dofx,int(2*cols/3):], self.response[dofy,int(2*cols/3):]))
            probe_resp = _probe_resp[0,:]
            # fmt: on

            amp, freq = self._dfft(probe_resp, self.dt)

            if range_freq is not None:
                amp = amp[(freq >= range_freq[0]) & (freq <= range_freq[1])]

                freq = freq[(freq >= range_freq[0]) & (freq <= range_freq[1])]

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=amp,
                    mode="lines",
                    name=f"Probe {i + 1} - Node {p[0]}",
                    legendgroup=f"Probe {i + 1} - Node {p[0]}",
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
