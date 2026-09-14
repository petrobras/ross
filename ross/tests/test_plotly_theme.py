import pytest
from plotly import io as pio

import ross  # noqa: F401
from plotly import graph_objects as go

from ross.plotly_theme import (
    DEFAULT_COLOR,
    ROSS_FONT_FAMILY,
    axes_indicator_2d,
    axes_indicator_3d,
    color_shades,
    parse_color,
)


@pytest.mark.parametrize(
    "color, expected",
    [
        ("#b22222", (178, 34, 34)),
        ("#B22222", (178, 34, 34)),
        ("#abc", (170, 187, 204)),
        ("rgb(178, 34, 34)", (178, 34, 34)),
        ("rgba(178, 34, 34, 0.3)", (178, 34, 34)),
        ("Firebrick", (178, 34, 34)),
        ("darkslategray", (47, 79, 79)),
        (DEFAULT_COLOR, (82, 82, 82)),
    ],
)
def test_parse_color(color, expected):
    assert parse_color(color) == expected


@pytest.mark.parametrize(
    "color", ["", "not a color", "#12345", "#gggggg", "rgb(1, 2)", None, 42]
)
def test_parse_color_fallback(color):
    assert parse_color(color) == parse_color(DEFAULT_COLOR)


def test_color_shades():
    shades = color_shades("#355d7a")

    assert shades["base"] == "#355d7a"
    assert shades["tint"] == "rgba(53,93,122,0.30)"
    assert shades["section"] == "#90a6b6"
    assert shades["edge"] == "#223c4f"
    assert shades["dark"] == "#2d4f68"


def test_color_shades_are_ordered_by_lightness():
    def lightness(color):
        return sum(parse_color(color))

    shades = color_shades("Firebrick")

    assert lightness(shades["section"]) > lightness(shades["base"])
    assert lightness(shades["base"]) > lightness(shades["dark"])
    assert lightness(shades["dark"]) > lightness(shades["edge"])


def test_color_shades_of_black_and_white():
    assert color_shades("black")["edge"] == "#000000"
    assert color_shades("white")["section"] == "#ffffff"


def test_templates_are_registered():
    assert "ross" in pio.templates
    assert "ross_dark" in pio.templates
    assert pio.templates.default == "ross"


def test_templates_use_ibm_plex_with_fallbacks():
    assert ROSS_FONT_FAMILY.startswith("IBM Plex Sans")
    assert "sans-serif" in ROSS_FONT_FAMILY
    assert pio.templates["ross"].layout.font.family == ROSS_FONT_FAMILY
    assert pio.templates["ross_dark"].layout.font.family == ROSS_FONT_FAMILY


def test_updatemenus_follow_the_templates():
    light = pio.templates["ross"].layout.updatemenudefaults
    dark = pio.templates["ross_dark"].layout.updatemenudefaults

    assert light.bgcolor == "white"
    assert dark.bgcolor == "#122839"
    assert dark.bordercolor != light.bordercolor
    assert dark.font.color == "#dfe8f3"


def test_dark_template_differs_from_light():
    light = pio.templates["ross"].layout
    dark = pio.templates["ross_dark"].layout

    assert light.paper_bgcolor == "white"
    assert light.plot_bgcolor == "white"
    assert dark.paper_bgcolor == "#0b1826"
    assert dark.plot_bgcolor == "#0b1826"
    assert dark.font.color == "#dfe8f3"
    assert dark.font.color != light.font.color
    assert dark.xaxis.gridcolor != light.xaxis.gridcolor
    assert dark.yaxis.linecolor != light.yaxis.linecolor
    assert dark.colorway != light.colorway
    assert len(dark.colorway) == len(light.colorway)


def test_axes_indicator_2d_zy_plane():
    fig = go.Figure()
    show, hide = axes_indicator_2d(fig, plane="zy", visible=False)

    # x points into the page: a circle crossed by two diagonal lines
    assert [shape.type for shape in fig.layout.shapes].count("line") == 2
    assert all(shape.visible is False for shape in fig.layout.shapes)
    assert all(shape.xsizemode == "pixel" for shape in fig.layout.shapes)
    labels = {a.text for a in fig.layout.annotations if a.text}
    assert labels == {"<i>x</i>", "<i>y</i>", "<i>z</i>", "<i>ω</i>"}

    assert set(show) == set(hide)
    assert all(show.values()) and not any(hide.values())
    fig.plotly_relayout(dict(show))
    assert all(shape.visible for shape in fig.layout.shapes)
    assert all(annotation.visible for annotation in fig.layout.annotations)


def test_axes_indicator_2d_xy_plane():
    fig = go.Figure()
    axes_indicator_2d(fig, plane="xy")

    # z points out of the page: a circle with a filled center dot, no cross
    assert [shape.type for shape in fig.layout.shapes].count("line") == 0
    assert [shape.type for shape in fig.layout.shapes].count("circle") == 2
    assert all(shape.visible for shape in fig.layout.shapes)

    with pytest.raises(ValueError):
        axes_indicator_2d(go.Figure(), plane="xz")


def test_axes_indicator_2d_appends_to_existing_items():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1)
    fig.add_annotation(text="kept", x=0, y=0)
    show, hide = axes_indicator_2d(fig)

    assert "shapes[0].visible" not in show
    assert "annotations[0].visible" not in show
    assert fig.layout.shapes[0].type == "rect"
    assert fig.layout.annotations[0].text == "kept"


def test_axes_indicator_3d_maps_display_arms_onto_scene_axes():
    fig = go.Figure()
    fig = axes_indicator_3d(
        fig,
        origin=dict(x=1.0, y=2.0, z=3.0),
        size=0.5,
        scales=dict(x=10.0, y=4.0, z=4.0),
        scene_axes={"x": "y", "y": "z", "z": "x"},
    )

    assert [trace.type for trace in fig.data] == ["scatter3d", "scatter3d"]
    lines, text = fig.data
    # one legend entry toggles the whole triad
    assert [trace.showlegend for trace in fig.data] == [True, False]
    assert {trace.legendgroup for trace in fig.data} == {"axes"}
    points = {
        (round(x, 9), round(y, 9), round(z, 9))
        for x, y, z in zip(lines.x, lines.y, lines.z)
        if x is not None
    }
    # rotor x arm runs along the scene y axis: half a display unit is 2 data units
    assert (1.0, 4.0, 3.0) in points
    # rotor y arm along the scene z axis
    assert (1.0, 2.0, 5.0) in points
    # rotor z arm along the scene x axis: half a display unit is 5 data units
    assert (6.0, 2.0, 3.0) in points
    assert list(text.text) == ["x", "y", "z", "ω"]


def test_axes_indicator_3d_without_z_arm():
    fig = axes_indicator_3d(
        go.Figure(),
        origin=dict(x=0.0, y=0.0, z=0.0),
        size=1.0,
        scales=dict(x=1.0, y=1.0, z=1.0),
        arms=("x", "y"),
    )
    lines, text = fig.data
    assert list(text.text) == ["x", "y", "ω"]
    points = {(x, y, z) for x, y, z in zip(lines.x, lines.y, lines.z) if x is not None}
    assert (0.0, 0.0, 1.0) not in points
    # the spin label sits in the x-y plane, where the ring turns about the origin
    assert text.z[2] == 0
