import pytest
from plotly import io as pio

import ross  # noqa: F401
from ross.plotly_theme import (
    DEFAULT_COLOR,
    ROSS_FONT_FAMILY,
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
