// The light and dark themes, and the figures that follow them.
//
// The colours live in `design/ross-tokens.css`, the same file the ROSS
// documentation loads: light values on `:root`, dark values re-pointed under
// `html[data-theme="dark"]`. This module only decides which of the two the page
// is in, and passes the answer on to Plotly, which does not read CSS.
//
// The choice is remembered under `ross-theme`, next to `ross-language`. With
// nothing remembered the page follows the system, and keeps following it until
// the user presses the button. A classic script in `index.html` applies the
// same rule before the first paint, so a dark page is never born white; the two
// read the same key, and `tests/test_design_system.py` keeps them equal.

export const THEME_KEY = 'ross-theme';

const THEMES = ['light', 'dark'];

const DARK_QUERY = '(prefers-color-scheme: dark)';

function systemTheme() {
    return typeof window.matchMedia === 'function' && window.matchMedia(DARK_QUERY).matches
        ? 'dark'
        : 'light';
}

export function preferredTheme() {
    try {
        const stored = localStorage.getItem(THEME_KEY);
        return THEMES.includes(stored) ? stored : null;
    } catch (e) {
        return null;   // storage blocked: the page follows the system
    }
}

export function currentTheme() {
    return preferredTheme() || systemTheme();
}

function rememberTheme(theme) {
    try {
        localStorage.setItem(THEME_KEY, theme);
    } catch (e) { /* a preference is a convenience; it cannot break the page */ }
}

// Applies a theme to the page: the attribute the tokens are scoped by, the icon
// of every theme button, and the figures already drawn.
export function applyTheme(theme) {
    const root = document.documentElement;
    if (root) root.setAttribute('data-theme', theme);
    document.querySelectorAll('.btn-theme i').forEach(icon => {
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    });
    restyleFigures();
}

export function toggleTheme() {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    rememberTheme(next);
    applyTheme(next);
}

// Run once at startup. Until the user chooses, the page follows the system,
// including a change of the system's mind while the page is open.
export function startTheme() {
    applyTheme(currentTheme());
    if (typeof window.matchMedia !== 'function') return;
    const query = window.matchMedia(DARK_QUERY);
    const follow = () => { if (!preferredTheme()) applyTheme(systemTheme()); };
    if (typeof query.addEventListener === 'function') query.addEventListener('change', follow);
}

// --- Figures ----------------------------------------------------------------
//
// Plotly paints from the template ROSS bakes into each figure -- `ross`, the
// light one. In the dark theme the figure is repainted from the tokens, read
// from the stylesheet and not written here a second time, so the chart and the
// card it sits on come from the same values. Back in the light theme the same
// keys go to null and the template shows through again, which is exactly what
// the documentation does with its own figures (`plotly-theme-sync.js`).
//
// The paper is transparent in both themes: a figure sits on a card, and the
// card already has the right surface.

const TRANSPARENT = 'rgba(0,0,0,0)';

const SERIES = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'];

// The keys each part of a figure is repainted by. The list exists apart from
// the values so that the light theme can unset exactly what the dark one set.
const KEYS = {
    layout: ['font.color', 'colorway', 'hoverlabel.bgcolor', 'hoverlabel.bordercolor',
             'hoverlabel.font.color', 'legend.bgcolor', 'legend.bordercolor',
             'modebar.color', 'modebar.activecolor'],
    axis: ['gridcolor', 'zerolinecolor', 'linecolor'],
    sceneAxis: ['backgroundcolor', 'gridcolor', 'linecolor', 'zerolinecolor'],
    polarAxis: ['gridcolor', 'linecolor'],
    menu: ['bgcolor', 'bordercolor', 'font.color'],
};

function darkPaint() {
    if (typeof window.getComputedStyle !== 'function' || !document.documentElement) return null;
    const css = window.getComputedStyle(document.documentElement);
    const token = name => css.getPropertyValue(name).trim();
    const text = token('--text-body');
    const grid = token('--grid-line');
    const gridStrong = token('--grid-line-strong');
    const card = token('--surface-card');
    if (!text || !grid || !card) return null;   // the tokens did not load
    return {
        layout: {
            'font.color': text,
            colorway: SERIES.map(name => token('--plot-' + name)),
            'hoverlabel.bgcolor': card,
            'hoverlabel.bordercolor': gridStrong,
            'hoverlabel.font.color': text,
            'legend.bgcolor': TRANSPARENT,
            'legend.bordercolor': gridStrong,
            'modebar.color': token('--text-muted'),
            'modebar.activecolor': text,
        },
        axis: { gridcolor: grid, zerolinecolor: grid, linecolor: token('--border-strong') },
        sceneAxis: { backgroundcolor: card, gridcolor: gridStrong, linecolor: grid, zerolinecolor: grid },
        polarAxis: { gridcolor: grid, linecolor: grid },
        menu: { bgcolor: card, bordercolor: gridStrong, 'font.color': text },
    };
}

// The flat `Plotly.relayout` update that paints `layout` in `theme`. The keys
// depend on the figure: only the axes, scenes and menus it has are touched.
function paintUpdate(layout, theme) {
    const paint = theme === 'dark' ? darkPaint() : null;
    const update = { paper_bgcolor: TRANSPARENT, plot_bgcolor: TRANSPARENT };
    const put = (prefix, group) => KEYS[group].forEach(key => {
        update[prefix + key] = paint ? paint[group][key] : null;
    });
    put('', 'layout');
    Object.keys(layout || {}).forEach(name => {
        if (/^[xy]axis\d*$/.test(name)) {
            put(name + '.', 'axis');
        } else if (/^scene\d*$/.test(name)) {
            ['xaxis', 'yaxis', 'zaxis'].forEach(axis => put(name + '.' + axis + '.', 'sceneAxis'));
        } else if (/^polar\d*$/.test(name)) {
            update[name + '.bgcolor'] = paint ? TRANSPARENT : null;
            ['angularaxis', 'radialaxis'].forEach(axis => put(name + '.' + axis + '.', 'polarAxis'));
        }
    });
    ((layout || {}).updatemenus || []).forEach((menu, i) => put('updatemenus[' + i + '].', 'menu'));
    return update;
}

// `a.b[2].c` written into a nested object, the way relayout reads it.
function assignPath(target, path, value) {
    const steps = path.replace(/\[(\d+)\]/g, '.$1').split('.');
    let cursor = target;
    steps.slice(0, -1).forEach((step, i) => {
        const next = /^\d+$/.test(steps[i + 1]) ? [] : {};
        if (cursor[step] === null || typeof cursor[step] !== 'object') cursor[step] = next;
        cursor = cursor[step];
    });
    cursor[steps[steps.length - 1]] = value;
}

// The layout to hand to `Plotly.newPlot`, painted for the current theme.
//
// A copy, not the object given: the figure the server sent is what the analysis
// store keeps and what a saved file carries, and a theme is not part of a
// result. Painting it in place would save tonight's dark colours into a file
// that opens, tomorrow, on a light screen.
export function themedLayout(layout) {
    const painted = JSON.parse(JSON.stringify(layout || {}));
    const update = paintUpdate(painted, currentTheme());
    Object.keys(update).forEach(key => {
        if (update[key] !== null) assignPath(painted, key, update[key]);
    });
    return painted;
}

// Repaints every figure on the page. Called on a change of theme; what is drawn
// after the change comes through `themedLayout` already painted.
function restyleFigures() {
    if (typeof Plotly === 'undefined' || typeof Plotly.relayout !== 'function') return;
    document.querySelectorAll('.js-plotly-plot').forEach(figure => {
        if (!figure.layout) return;
        Promise.resolve(Plotly.relayout(figure, paintUpdate(figure.layout, currentTheme())))
            .catch(error => console.error('theme: could not repaint a figure', error));
    });
}
