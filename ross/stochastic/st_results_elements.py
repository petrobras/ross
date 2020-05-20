"""Plotting module for elements.

This modules provides functions to plot the elements statistic data.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

pio.renderers.default = "browser"


def plot_histogram(
    attribute_dict, label={}, var_list=[], histogram_kwargs={}, plot_kwargs={}
):
    """Plot histogram and the PDF.

    This function creates a histogram to display the random variable distribution.

    Parameters
    ----------
    attribute_dict : dict
        Dictionary with element parameters.
    label : dict
        Dictionary with labels for each element parameter. Labels are displayed
        on plotly figure.
    var_list : list, optional
        List of random variables, in string format, to plot.
    histogram_kwargs : dict, optional
        Additional key word arguments can be passed to change
        the plotly.go.histogram (e.g. histnorm="probability density", nbinsx=20...).
        *See Plotly API to more information.
    plot_kwargs : dict, optional
        Additional key word arguments can be passed to change the plotly go.figure
        (e.g. line=dict(width=4.0, color="royalblue"), opacity=1.0...).
        *See Plotly API to more information.

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        A figure with the histogram plots.
    """
    hist_default_values = dict(
        histnorm="probability density",
        cumulative_enabled=False,
        nbinsx=20,
        marker_color="red",
        opacity=1.0,
    )
    for k, v in hist_default_values.items():
        histogram_kwargs.setdefault(k, v)

    plot_default_values = dict(line=dict(width=4.0, color="royalblue"), opacity=1.0)
    for k, v in plot_default_values.items():
        plot_kwargs.setdefault(k, v)

    rows = 1 if len(var_list) < 2 else 2
    cols = len(var_list) // 2 + len(var_list) % 2
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=["<b>Histogram - {}<b>".format(var) for var in var_list],
    )

    for i, var in enumerate(var_list):
        row = i % 2 + 1
        col = i // 2 + 1
        if histogram_kwargs["histnorm"] == "probability density":
            if histogram_kwargs["cumulative_enabled"] is True:
                y_label = "CDF"
            else:
                y_label = "PDF"
        else:
            y_label = "Frequency"

        fig.add_trace(
            go.Histogram(x=attribute_dict[var], name="Histogram", **histogram_kwargs),
            row=row,
            col=col,
        )
        if y_label == "PDF":
            x = np.linspace(
                min(attribute_dict[var]),
                max(attribute_dict[var]),
                len(attribute_dict[var]),
            )
            kernel = gaussian_kde(attribute_dict[var])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=kernel(x),
                    mode="lines",
                    name="PDF Estimation",
                    **plot_kwargs,
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(
            title_text="{}".format(label[var]),
            title_font=dict(family="Arial", size=20),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="{}".format(y_label),
            title_font=dict(family="Arial", size=20),
            gridcolor="lightgray",
            showline=True,
            linewidth=2.5,
            linecolor="black",
            mirror=True,
            row=row,
            col=col,
        )
    fig.update_layout(bargroupgap=0.1, plot_bgcolor="white")

    return fig
