"""Plotly theme to ROSS - Rotordynamic Open Source Sofware."""
from plotly import graph_objects as go
from plotly import io as pio

# tableau colors
tableau_colors = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}
pio.templates["ross"] = go.layout.Template(
    layout={
        "annotationdefaults": {
            "arrowcolor": "#2a3f5f",
            "arrowhead": 0,
            "arrowwidth": 1,
        },
        "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
        "colorscale": {
            "diverging": [
                [0, "#8e0152"],
                [0.1, "#c51b7d"],
                [0.2, "#de77ae"],
                [0.3, "#f1b6da"],
                [0.4, "#fde0ef"],
                [0.5, "#f7f7f7"],
                [0.6, "#e6f5d0"],
                [0.7, "#b8e186"],
                [0.8, "#7fbc41"],
                [0.9, "#4d9221"],
                [1, "#276419"],
            ],
            "sequential": [
                [0.0, "#0d0887"],
                [0.1111111111111111, "#46039f"],
                [0.2222222222222222, "#7201a8"],
                [0.3333333333333333, "#9c179e"],
                [0.4444444444444444, "#bd3786"],
                [0.5555555555555556, "#d8576b"],
                [0.6666666666666666, "#ed7953"],
                [0.7777777777777778, "#fb9f3a"],
                [0.8888888888888888, "#fdca26"],
                [1.0, "#f0f921"],
            ],
            "sequentialminus": [
                [0.0, "#0d0887"],
                [0.1111111111111111, "#46039f"],
                [0.2222222222222222, "#7201a8"],
                [0.3333333333333333, "#9c179e"],
                [0.4444444444444444, "#bd3786"],
                [0.5555555555555556, "#d8576b"],
                [0.6666666666666666, "#ed7953"],
                [0.7777777777777778, "#fb9f3a"],
                [0.8888888888888888, "#fdca26"],
                [1.0, "#f0f921"],
            ],
        },
        "colorway": list(tableau_colors.values()),
        "font": {"color": "#2a3f5f"},
        "geo": {
            "bgcolor": "white",
            "lakecolor": "white",
            "landcolor": "white",
            "showlakes": True,
            "showland": True,
            "subunitcolor": "#C8D4E3",
        },
        "hoverlabel": {"align": "left"},
        "hovermode": "closest",
        "mapbox": {"style": "light"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "polar": {
            "angularaxis": {
                "gridcolor": "#EBF0F8",
                "linecolor": "#EBF0F8",
                "ticks": "",
            },
            "bgcolor": "white",
            "radialaxis": {"gridcolor": "#EBF0F8", "linecolor": "#EBF0F8", "ticks": ""},
        },
        "scene": {
            "xaxis": {
                "backgroundcolor": "white",
                "gridcolor": "#DFE8F3",
                "gridwidth": 2,
                "linecolor": "#EBF0F8",
                "showbackground": True,
                "ticks": "",
                "zerolinecolor": "#EBF0F8",
                "showspikes": False,
            },
            "yaxis": {
                "backgroundcolor": "white",
                "gridcolor": "#DFE8F3",
                "gridwidth": 2,
                "linecolor": "#EBF0F8",
                "showbackground": True,
                "ticks": "",
                "zerolinecolor": "#EBF0F8",
                "showspikes": False,
            },
            "zaxis": {
                "backgroundcolor": "white",
                "gridcolor": "#DFE8F3",
                "gridwidth": 2,
                "linecolor": "#EBF0F8",
                "showbackground": True,
                "ticks": "",
                "zerolinecolor": "#EBF0F8",
                "showspikes": False,
            },
            "camera": {
                "up": {"x": 0, "y": 0, "z": 1},
                "center": {"x": 0, "y": 0, "z": 0},
                "eye": {"x": 2.0, "y": 2.0, "z": 2.0},
            },
        },
        "shapedefaults": {"line": {"color": "#2a3f5f"}},
        "ternary": {
            "aaxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""},
            "baxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""},
            "bgcolor": "white",
            "caxis": {"gridcolor": "#DFE8F3", "linecolor": "#A2B1C6", "ticks": ""},
        },
        "title": {"x": 0.05},
        "xaxis": {
            "automargin": True,
            "gridcolor": "#EBF0F8",
            "showline": True,
            "linecolor": "black",
            "linewidth": 2.0,
            "mirror": True,
            "ticks": "",
            "title": {"standoff": 15},
            "zeroline": False,
            "zerolinecolor": "#EBF0F8",
            "zerolinewidth": 2,
        },
        "yaxis": {
            "automargin": True,
            "gridcolor": "#EBF0F8",
            "showline": True,
            "linecolor": "black",
            "linewidth": 2.0,
            "mirror": True,
            "ticks": "",
            "title": {"standoff": 15},
            "zeroline": False,
            "zerolinecolor": "#EBF0F8",
            "zerolinewidth": 2,
        },
    },
    data={
        "bar": [
            {
                "error_x": {"color": "#2a3f5f"},
                "error_y": {"color": "#2a3f5f"},
                "marker": {"line": {"color": "white", "width": 0.5}},
                "type": "bar",
            }
        ],
        "barpolar": [
            {"marker": {"line": {"color": "white", "width": 0.5}}, "type": "barpolar"}
        ],
        "carpet": [
            {
                "aaxis": {
                    "endlinecolor": "#2a3f5f",
                    "gridcolor": "#C8D4E3",
                    "linecolor": "#C8D4E3",
                    "minorgridcolor": "#C8D4E3",
                    "startlinecolor": "#2a3f5f",
                },
                "baxis": {
                    "endlinecolor": "#2a3f5f",
                    "gridcolor": "#C8D4E3",
                    "linecolor": "#C8D4E3",
                    "minorgridcolor": "#C8D4E3",
                    "startlinecolor": "#2a3f5f",
                },
                "type": "carpet",
            }
        ],
        "choropleth": [
            {"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}
        ],
        "contour": [
            {
                "colorbar": {"outlinewidth": 0, "ticks": ""},
                "colorscale": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
                "type": "contour",
            }
        ],
        "contourcarpet": [
            {"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}
        ],
        "heatmap": [
            {
                "colorbar": {"outlinewidth": 0, "ticks": ""},
                "colorscale": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
                "type": "heatmap",
            }
        ],
        "heatmapgl": [
            {
                "colorbar": {"outlinewidth": 0, "ticks": ""},
                "colorscale": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
                "type": "heatmapgl",
            }
        ],
        "histogram": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "histogram",
            }
        ],
        "histogram2d": [
            {
                "colorbar": {"outlinewidth": 0, "ticks": ""},
                "colorscale": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
                "type": "histogram2d",
            }
        ],
        "histogram2dcontour": [
            {
                "colorbar": {"outlinewidth": 0, "ticks": ""},
                "colorscale": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
                "type": "histogram2dcontour",
            }
        ],
        "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}],
        "parcoords": [
            {
                "line": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "parcoords",
            }
        ],
        "pie": [{"automargin": True, "type": "pie"}],
        "scatter": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "scatter",
            }
        ],
        "scatter3d": [
            {
                "line": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "scatter3d",
            }
        ],
        "scattercarpet": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "scattercarpet",
            }
        ],
        "scattergeo": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "scattergeo",
            }
        ],
        "scattergl": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "scattergl",
            }
        ],
        "scattermapbox": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "scattermapbox",
            }
        ],
        "scatterpolar": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "scatterpolar",
            }
        ],
        "scatterpolargl": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "scatterpolargl",
            }
        ],
        "scatterternary": [
            {
                "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}},
                "type": "scatterternary",
            }
        ],
        "surface": [
            {
                "colorbar": {"outlinewidth": 0, "ticks": ""},
                "colorscale": [
                    [0.0, "#0d0887"],
                    [0.1111111111111111, "#46039f"],
                    [0.2222222222222222, "#7201a8"],
                    [0.3333333333333333, "#9c179e"],
                    [0.4444444444444444, "#bd3786"],
                    [0.5555555555555556, "#d8576b"],
                    [0.6666666666666666, "#ed7953"],
                    [0.7777777777777778, "#fb9f3a"],
                    [0.8888888888888888, "#fdca26"],
                    [1.0, "#f0f921"],
                ],
                "type": "surface",
            }
        ],
        "table": [
            {
                "cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}},
                "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}},
                "type": "table",
            }
        ],
    },
)
