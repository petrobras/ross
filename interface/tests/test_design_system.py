# -*- coding: utf-8 -*-
"""The interface wears the ROSS design system, and wears the same one as the docs.

The tokens -- fonts, colour ramps, semantic aliases, spacing, radii, elevation,
motion, and the dark theme -- live in `docs/_static/ross-tokens.css`, where the
documentation loads them. The interface cannot reach that folder: it runs from
its own `frontend/` (and, packaged, from a bundle with no docs in it), so it
carries a copy under `frontend/design/`. A copy drifts. These guards are what
keep it from drifting, the same way `test_packaging.py` keeps the installed CI
workflow equal to the one kept here.

The rest of the file holds the interface to the rule that makes two themes
possible at all: every colour on screen comes from a token. A hex value in the
stylesheet, the page or a JS-built fragment is a colour that does not change
with the theme, and it shows up as a light patch on a dark screen -- or the
reverse -- on exactly the widget nobody looked at in the other theme."""

import glob
import io
import os
import re

import pytest

from frontend_source import FRONTEND, ROOT, modules, relative

DOCS_STATIC = os.path.join(os.path.dirname(ROOT), "docs", "_static")
DESIGN = os.path.join(FRONTEND, "design")
TOKENS = os.path.join(DESIGN, "ross-tokens.css")


def _read(path):
    with io.open(path, encoding="utf-8") as handle:
        return handle.read()


def _index():
    return _read(os.path.join(FRONTEND, "index.html"))


def _style():
    return _read(os.path.join(FRONTEND, "style.css"))


# --- the copy is the original ------------------------------------------------


def test_the_tokens_are_the_documentations_tokens():
    """One design system, not two that started out equal.

    Text and not bytes, with universal newlines: the two copies live under
    different line-ending rules (see `test_the_installed_workflow_is_the_one_we_keep`).
    """
    original = os.path.join(DOCS_STATIC, "ross-tokens.css")
    if not os.path.exists(original):
        return  # the interface checked out alone, or packaged: nothing to compare
    assert _read(TOKENS) == _read(original), (
        "frontend/design/ross-tokens.css and docs/_static/ross-tokens.css have "
        "drifted apart: copy the documentation's file over ours"
    )


def test_the_fonts_are_the_documentations_fonts():
    """The woff2 files the tokens name, byte for byte, licence included.

    The licence is text and is compared as text, with universal newlines: the
    two copies live under different line-ending rules (ours is pinned to LF by
    this folder's `.gitattributes`, the documentation's follows `text=auto`),
    so on a Windows checkout the documentation's copy arrives in CRLF and a
    byte comparison went red on a machine where nobody touched anything. Same
    reasoning as `test_the_tokens_are_the_documentations_tokens`.
    """
    original = os.path.join(DOCS_STATIC, "fonts")
    if not os.path.isdir(original):
        return
    theirs = sorted(os.listdir(original))
    ours = sorted(os.listdir(os.path.join(DESIGN, "fonts")))
    assert ours == theirs, "the font folders differ: %s vs %s" % (ours, theirs)
    for name in theirs:
        if name.endswith(".txt"):
            there = _read(os.path.join(original, name))
            here = _read(os.path.join(DESIGN, "fonts", name))
        else:
            with io.open(os.path.join(original, name), "rb") as handle:
                there = handle.read()
            with io.open(os.path.join(DESIGN, "fonts", name), "rb") as handle:
                here = handle.read()
        assert here == there, "%s differs from the documentation's copy" % name


def test_every_font_the_tokens_name_is_shipped():
    """A `src: url(...)` pointing at a missing file falls back in silence."""
    named = re.findall(r'url\("([^"]+\.woff2)"\)', _read(TOKENS))
    assert named, "the tokens no longer declare any font"
    missing = [name for name in named if not os.path.exists(os.path.join(DESIGN, name))]
    assert missing == [], "font named by the tokens and not shipped: %s" % missing


# --- the page loads the system -------------------------------------------------


def test_the_page_loads_the_tokens_before_the_stylesheet():
    """`style.css` is written in `var(--...)`: without the tokens first, it is blank."""
    html = _index()
    tokens = html.index('href="design/ross-tokens.css"')
    style = html.index('href="style.css"')
    assert tokens < style, "the tokens are loaded after the stylesheet that uses them"


def test_the_head_script_and_the_module_read_the_same_key():
    """The theme is applied twice: before the first paint, and by `core/theme.js`.

    The head script exists so that a dark page is not born white. If it read a
    different key from the module, the page would flash the wrong theme on every
    load -- which is the one defect the script is there to prevent."""
    theme_js = _read(os.path.join(FRONTEND, "core", "theme.js"))
    key = re.search(r"export const THEME_KEY = '([^']+)'", theme_js).group(1)
    html = _index()
    assert "localStorage.getItem('%s')" % key in html, (
        "index.html reads a different theme key from core/theme.js"
    )
    head = html[: html.index("</head>")]
    assert "data-theme" in head, "the theme is not applied in the head"
    opening = head.rindex("<script", 0, head.index("data-theme"))
    assert "type=" not in head[opening : opening + 40], (
        "the theme script became a module: it would run after the first paint"
    )


def test_the_dark_theme_is_scoped_the_way_the_documentation_scopes_it():
    """`html[data-theme="dark"]` is the attribute sphinx-book-theme sets.

    The tokens re-point their aliases under it; the interface has to set the
    same attribute, on the same element, or the dark values never apply."""
    assert 'html[data-theme="dark"]' in _read(TOKENS)
    theme_js = _read(os.path.join(FRONTEND, "core", "theme.js"))
    assert "document.documentElement" in theme_js
    assert "setAttribute('data-theme'" in theme_js


def test_the_theme_button_is_on_every_screen():
    """Like the language selector: changing it only on the Hub would force leaving."""
    html = _index()
    screens = html.count('class="screen')
    assert html.count('class="btn-theme"') == screens
    assert html.count('onclick="toggleTheme()"') == screens


# --- every colour is a token -----------------------------------------------------

HEX_COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b")

# The busy spinner masks with pure black and white: they are mask values, not
# colours anyone sees, and they must not follow the theme.
MASK_ONLY = ("core/dom.js",)


def test_the_stylesheet_has_no_colour_of_its_own():
    found = HEX_COLOUR.findall(_style())
    assert found == [], "hex colours in style.css: %s" % sorted(set(found))


def test_the_page_has_no_colour_of_its_own():
    found = HEX_COLOUR.findall(_index())
    assert found == [], "hex colours in index.html: %s" % sorted(set(found))


@pytest.mark.parametrize(
    "module", [relative(p) for p in modules() if relative(p) not in MASK_ONLY]
)
def test_the_javascript_writes_no_colour_of_its_own(module):
    text = _read(os.path.join(FRONTEND, module))
    found = HEX_COLOUR.findall(text) + re.findall(
        r"color:\s*(?:red|blue|green)\b", text
    )
    assert found == [], "%s paints with %s" % (module, sorted(set(found)))


def test_every_variable_used_is_defined():
    """`var(--x)` with no `--x` is not an error: it is a transparent widget."""
    defined = set(re.findall(r"(--[a-zA-Z][\w-]*)\s*:", _read(TOKENS) + _style()))
    used = set()
    texts = [_style(), _index()] + [_read(os.path.join(FRONTEND, m)) for m in modules()]
    for text in texts:
        used.update(re.findall(r"var\((--[a-zA-Z][\w-]*)\)", text))
    assert used, "no var() found: the sweep stopped working"
    undefined = sorted(used - defined)
    assert undefined == [], "variables used and never defined: %s" % undefined


def test_the_stylesheet_is_written_in_tokens():
    """Control for the test above: if `var()` fell out of use, it would pass empty."""
    assert _style().count("var(--") > 300


# --- the figures follow the theme -----------------------------------------------


def test_every_figure_is_drawn_through_the_theme():
    """A `Plotly.newPlot` that skips `themedLayout` draws a white chart on a dark card."""
    culprits = []
    for path in modules():
        text = _read(path)
        for found in re.finditer(r"Plotly\.newPlot\(", text):
            call = text[found.start() : found.start() + 400]
            if "themedLayout(" not in call:
                line = text.count("\n", 0, found.start()) + 1
                culprits.append("%s:%d" % (relative(path), line))
    assert culprits == [], "figures drawn outside the theme: %s" % culprits


def test_the_theme_paints_from_the_tokens_and_not_from_a_second_list():
    """The dark values Plotly gets are read from the stylesheet at run time.

    `docs/_static/plotly-theme-sync.js` writes them out by hand and says so;
    here the tokens are on the page, so there is no reason for a second copy."""
    theme_js = _read(os.path.join(FRONTEND, "core", "theme.js"))
    assert "getComputedStyle" in theme_js
    assert "getPropertyValue('--" in theme_js or "getPropertyValue(name)" in theme_js
    assert HEX_COLOUR.findall(theme_js) == []


def test_the_static_files_include_the_design_folder():
    """The bundle takes `frontend/` whole; a design folder outside it would be left behind."""
    assert DESIGN.startswith(FRONTEND)
    assert glob.glob(os.path.join(DESIGN, "fonts", "*.woff2"))
