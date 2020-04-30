"""Plotting module for elements.

This modules provides functions to plot the elements statistic data.
"""
import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from scipy.stats import gaussian_kde


def plot_histogram(attribute_dict, label={}, var_list=[], **kwargs):
    """Plot histogram and the PDF.

    This function creates a histogram to display the random variable
    distribution.

    Parameters
    ----------
    attribute_dict : dict
        Dictionary with element parameters.
    label : dict
        Dictionary with labels for each element parameter. Labels are displayed
        on bokeh figure.
    var_list : list, optional
        List of random variables, in string format, to plot.
    **kwargs : optional
        Additional key word arguments can be passed to change
        the numpy.histogram (e.g. density=True, bins=11, ...)

    Returns
    -------
    grid_plot : bokeh row
        A row with the histogram plots.
    """
    default_values = dict(density=True, bins=21)
    for k, v in default_values.items():
        kwargs.setdefault(k, v)

    figures = []

    for var in var_list:
        hist, edges = np.histogram(attribute_dict[var], **kwargs)
        if kwargs["density"] is True:
            y_label = "PDF"
        else:
            y_label = "Frequency"

        fig = figure(
            width=640,
            height=480,
            title="Histogram - {}".format(var),
            x_axis_label="{}".format(label[var]),
            y_axis_label="{}".format(y_label),
        )
        fig.xaxis.axis_label_text_font_size = "14pt"
        fig.yaxis.axis_label_text_font_size = "14pt"
        fig.axis.major_label_text_font_size = "14pt"
        fig.title.text_font_size = "14pt"

        fig.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="red",
            line_color="white",
            alpha=1.0,
        )
        if y_label == "PDF":
            x = np.linspace(
                min(attribute_dict[var]),
                max(attribute_dict[var]),
                len(attribute_dict[var]),
            )
            kernel = gaussian_kde(attribute_dict[var])
            fig.line(
                x,
                kernel(x),
                line_alpha=1.0,
                line_width=3.0,
                line_color="royalblue",
                legend_label="pdf estimation",
            )

        figures.append(fig)

    grid_plot = gridplot([figures], toolbar_location="right")

    return grid_plot
