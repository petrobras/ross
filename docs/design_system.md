# Design system

ROSS has more than one surface: this documentation, the figures the library
draws, and the graphical interface under `interface/`. They should look like
one product, and the way they do is by sharing one set of **design tokens**:
named values for type, colour, spacing, radius, elevation and motion, with a
light and a dark theme, that every surface reads instead of choosing its own.

## Where it lives

| What | Where | Role |
|------|-------|------|
| Tokens | `docs/_static/ross-tokens.css` | **Source of truth.** Fonts, colour ramps, semantic aliases, spacing, radii, shadows, motion, and the dark theme. |
| Fonts | `docs/_static/fonts/` | IBM Plex Sans and Mono, Latin-1 subsets, self-hosted so that nothing is fetched from a third party. |
| Documentation theme | `docs/_static/theme-ross.css` | Maps the tokens onto sphinx-book-theme. |
| Documentation figures | `docs/_static/plotly-theme-sync.js` | Repaints embedded Plotly figures when the reader toggles the theme. |
| Plotly templates | `ross/plotly_theme.py` | The `ross` (light, default) and `ross_dark` templates every ROSS figure is drawn with. |
| Interface | `interface/frontend/design/`, `interface/frontend/style.css`, `interface/frontend/core/theme.js` | A verbatim copy of the tokens and fonts, a stylesheet written only in tokens, and the theme switch. |

The mirrors cannot import the source: Python does not read a stylesheet at
import time, and the interface runs from its own folder (and, packaged, from a
bundle that carries no docs). So they are copies, and copies are kept honest by
tests rather than by discipline:

* `ross/tests/test_plotly_theme.py` reads the tokens and checks that the light
  and dark colorways and the dark surfaces in `plotly_theme.py` are the same
  values.
* `interface/tests/test_design_system.py` checks that the interface's tokens and
  fonts are byte-for-byte the documentation's, that no hex colour exists in the
  interface outside the tokens, and that every figure is drawn through the theme.

To change a colour, change the token here, then copy the file into
`interface/frontend/design/` and update `plotly_theme.py`; the tests name what
was missed.

## The tokens

### Typography

IBM Plex Sans for text and Plex Mono for code and numbers, with system
fallbacks (`--font-sans`, `--font-mono`). The scale is a set of roles rather
than sizes:

| Role | Token | Use |
|------|-------|-----|
| Display | `--type-display` | Landing titles. |
| Headings | `--type-h1`, `--type-h2`, `--type-h3` | Semibold, slightly tightened (`--ls-heading`). |
| Body | `--type-body` | Paragraphs, 1.65 line height. |
| Lead | `--type-lead` | Introductions. |
| Code | `--type-code` | Code blocks and inline code. |
| Eyebrow | `--type-eyebrow` | Small uppercase labels: form labels, table headers, section captions. Letter-spaced by `--ls-eyebrow`. |

Weights are `--fw-regular`, `--fw-medium`, `--fw-semibold` and `--fw-bold`.

### Colour

Two ramps carry the brand. **Ink** is a cool blue-gray anchored on the navy
ROSS already used in its results, `--ink-900` (darkest) to `--ink-050`
(lightest). **Blue** is tableau blue, the first colour of every ROSS plot and
the de-facto accent, `--blue-700` to `--blue-100`.

The plot palette is the tableau colorway in colorway order, `--plot-blue`,
`--plot-orange`, `--plot-green`, `--plot-red`, `--plot-purple`,
`--plot-brown`, `--plot-pink`, `--plot-gray`, `--plot-olive`, `--plot-cyan`.
The semantic hues are drawn from it: `--success-*` (green), `--warning-*`
(orange), `--danger-*` (red) and `--info-*` (cyan), each with a `500` for
fills, a `700` for text and a `100` for tints.

Surfaces and text are named by role, and the roles are what a stylesheet
should use:

| Alias | Light | Meaning |
|-------|-------|---------|
| `--surface-page` | white | The page. |
| `--surface-sunken` | `--ink-050` | A recessed area: a workspace behind cards, a code block. |
| `--surface-card` | white | A raised card. |
| `--surface-tint` | `--blue-100` | The selected item. |
| `--surface-hover` | `--ink-050` | Hover on a neutral control. |
| `--text-strong`, `--text-body`, `--text-muted`, `--text-subtle` | ink ramp | Headings, prose, secondary text, labels. |
| `--text-link`, `--text-link-hover` | blue | Links. |
| `--border-subtle`, `--border-default`, `--border-strong` | ink ramp | Rules and outlines, by weight. |
| `--border-focus`, `--shadow-focus` | blue | The focus ring. |
| `--accent`, `--accent-hover`, `--accent-active`, `--accent-soft` | blue | The one action colour. |
| `--grid-line`, `--grid-line-strong` | ink ramp | Plot grids. |

### Spacing, radius, elevation, motion

Spacing is a 4 px grid, `--space-1` (4 px) to `--space-32` (128 px). Radii
run from `--radius-xs` (3 px) to `--radius-xl` (16 px), plus `--radius-pill`
and `--radius-circle`. Shadows are cool-tinted and low: `--shadow-xs` on
resting cards, `--shadow-md` on hover, `--shadow-lg` on modals. Motion uses
`--dur-fast`, `--dur-base`, `--dur-slow` with `--ease-standard`, and all
durations collapse to zero under `prefers-reduced-motion`.

## The dark theme

The dark theme is not a second palette. Under `html[data-theme="dark"]` the
tokens re-point only the semantic aliases -- surfaces, text, borders, accent,
grid lines, the semantic tints -- and lift the plot colours for contrast on
dark paper. The base ramps keep their meaning. A stylesheet written in the
aliases needs no dark rules of its own; a stylesheet with a hex value in it
has a bug that only shows in the other theme.

`data-theme` on `<html>` is the attribute sphinx-book-theme sets for its own
toggle, which is why the tokens are scoped to it and why every other surface
sets the same one. The interface remembers the choice in the browser under
`ross-theme` and follows the operating system until a choice is made.

## Figures

Every ROSS figure is drawn with the `ross` template. The dark counterpart is
`ross_dark`, opt-in from Python:

```python
import plotly.io as pio

pio.templates.default = "ross_dark"      # every figure from here on
fig = rotor.run_campbell(speed_range).plot(template="ross_dark")  # or one figure
```

In a page, a figure is not redrawn to change theme -- it is repainted.
The documentation and the interface both keep the light template baked into
the figure and, in the dark theme, `Plotly.relayout` the paper, the fonts, the
grids, the colorway, the hover labels and the menus from the dark tokens;
back in the light theme the same keys are set to `null` and the template
shows through again. The interface reads the values from the tokens on the
page at run time; the documentation, whose figures are static HTML, writes
them out in `plotly-theme-sync.js`, and that copy is one of the mirrors the
tests watch.

## Using it in another ROSS surface

A new page, dashboard or app that should look like ROSS needs three things:

1. Copy `ross-tokens.css` and the `fonts/` folder next to it, and load the
   tokens before your own stylesheet.
2. Write the stylesheet in the semantic aliases, never in hex.
3. Set `data-theme="light"` or `"dark"` on `<html>`, before the first paint,
   from the user's choice or from `prefers-color-scheme`.

Plotly figures then take the `ross` template as they are, and the same
relayout the interface performs in `core/theme.js` turns them dark.
