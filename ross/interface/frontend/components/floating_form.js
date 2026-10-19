// The element form as a window of its own, over the modeling screen.
//
// WHY A WINDOW AND NOT A BOX IN THE LIST. The form used to open inside the
// element list, under the row it edited. Three things came from living there:
//
//   * a long form -- a bearing's coefficient tables, a seal's -- pushed the
//     list out of sight, in a panel 360 px wide;
//   * every redraw of the list had to lift the form out first, or emptying the
//     list deleted it from the page (the "lists vanished" of an earlier week),
//     and Sortable counted it as a row until told not to;
//   * the list could not stay hidden while editing, so editing from the figure
//     always paid 360 px of it.
//
// Out of the list, all three go: the list keeps only its rows, the form scrolls
// on its own, and the list stays as the person left it.
//
// WHY NOT MODAL. The figure beside the form is part of editing: a click on
// another part in 3D opens that one, the edited part stays marked, the Delete
// key removes it. A scrim over the screen would hide exactly that. So the
// window floats, is dragged by its bar off whatever it covers, and opens where
// it was last put.
//
// Where it may go is a pure function, `keepInside`, so the rule is checked
// apart from a browser.

const PLACE_KEY = 'ross-form-window';
const MARGIN = 16;
// What has to stay on screen under the window's top: the bar and the first
// fields. Its height is capped to the room below it, and the rest scrolls
// inside the window.
const MIN_SHOWN = 160;

// A place for a window `width` wide in a `viewport`: all of its width on
// screen, and its bar never above the top or below the last `MIN_SHOWN` px.
//
// This keeps a window reachable; it is not a layout that changes with the
// window's width (`test_the_layout_does_not_reflow_with_the_window`). Nothing
// on the page changes shape: a window dragged near an edge, or left there by a
// smaller window, is only brought back into view.
export function keepInside(place, width, viewport) {
    const maxLeft = Math.max(MARGIN, viewport.width - width - MARGIN);
    const maxTop = Math.max(MARGIN, viewport.height - MIN_SHOWN);
    return {
        left: Math.round(Math.min(Math.max(place.left, MARGIN), maxLeft)),
        top: Math.round(Math.min(Math.max(place.top, MARGIN), maxTop)),
    };
}

// The first place, before the person has put it anywhere: beside the list and
// under the bar with the rotor's mass -- the corner of the figure nearest to
// where the form used to open.
export function firstPlace(figurePanel, infoBar) {
    const top = infoBar && infoBar.bottom > figurePanel.top ? infoBar.bottom : figurePanel.top;
    return { left: figurePanel.left + MARGIN, top: top + MARGIN / 2 };
}

function viewport() {
    const page = document.documentElement;
    return { width: page.clientWidth, height: page.clientHeight };
}

function remembered() {
    try {
        const place = JSON.parse(localStorage.getItem(PLACE_KEY));
        if (place && isFinite(place.left) && isFinite(place.top)) return place;
    } catch (e) { /* storage blocked or damaged: the first place */ }
    return null;
}

function remember(place) {
    try {
        localStorage.setItem(PLACE_KEY, JSON.stringify(place));
    } catch (e) { /* a preference: without storage it lasts until the page closes */ }
}

function placeOf(box) {
    return { left: parseFloat(box.style.left) || 0, top: parseFloat(box.style.top) || 0 };
}

function put(box, place) {
    const room = viewport();
    const at = keepInside(place, box.offsetWidth || 0, room);
    box.style.left = at.left + 'px';
    box.style.top = at.top + 'px';
    box.style.maxHeight = (room.height - at.top - MARGIN) + 'px';
    return at;
}

function isOpen(box) {
    return box.style.display === 'block';
}

// Called as the form opens. A window already open stays where it is: moving
// from one element to the next is not a reason for it to jump.
export function placeFormWindow() {
    const box = document.getElementById('insertion-form');
    if (!box || isOpen(box)) return;
    const panel = document.querySelector('#screen-modeling .plot-panel');
    const info = document.getElementById('rotor-info');
    const start = remembered()
        || firstPlace(panel.getBoundingClientRect(), info && info.getBoundingClientRect());
    // Shown before it is measured: a hidden box has no width to keep inside.
    box.style.visibility = 'hidden';
    box.style.display = 'block';
    put(box, start);
    box.style.visibility = '';
}

// Dragging by the bar, and back into view when the browser window shrinks.
export function startFormWindow() {
    const box = document.getElementById('insertion-form');
    const bar = box && box.querySelector('.form-window-bar');
    if (!bar) return;
    let grab = null;
    bar.addEventListener('pointerdown', event => {
        if (event.button !== 0 || event.target.closest('button')) return;
        const at = box.getBoundingClientRect();
        grab = { id: event.pointerId, dx: event.clientX - at.left, dy: event.clientY - at.top };
        bar.setPointerCapture(event.pointerId);
        event.preventDefault();   // no text selected on the way
    });
    bar.addEventListener('pointermove', event => {
        if (!grab || event.pointerId !== grab.id) return;
        put(box, { left: event.clientX - grab.dx, top: event.clientY - grab.dy });
    });
    const drop = event => {
        if (!grab || event.pointerId !== grab.id) return;
        grab = null;
        remember(placeOf(box));
    };
    bar.addEventListener('pointerup', drop);
    bar.addEventListener('pointercancel', drop);
    window.addEventListener('resize', () => {
        if (isOpen(box)) put(box, placeOf(box));
    });
}
