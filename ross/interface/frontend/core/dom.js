// Two utilities with no owner: escaping text that goes inside HTML, and
// handing a file to the user.
export function escapeHtml(value) {
    return String(value === null || value === undefined ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

export function downloadTextFile(name, text, type) {
    const blob = new Blob([text], { type: type });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
}

// The busy indicator: the ROSS logo with its journal whirling in the bearing.
//
// The logo draws a journal sitting off-centre in its bearing, as it does when it
// runs. While the app computes, the bore orbits the bearing centre on that same
// eccentricity, so the picture at rest is the logo itself. It is one SVG with a
// SMIL rotation: no timer, no library, no GIF. It takes its colour from
// `currentColor` and its size in em, like the icon it replaced, so the CSS
// around it did not have to change.
//
// WHY A MASK AND NOT A PATH. The logo is a single path with the hole cut out of
// it. Moving the hole would mean rewriting the path every frame; a mask moves
// only the circle that cuts it. Each spinner gets its own mask id: two spinners
// sharing one would both point at whichever came first in the page, and the
// survivor would turn into a full disc when that one was removed.
//
// WHY IT MAY STAND STILL. A user who asked the system for less motion gets the
// logo without the orbit. The words beside it already say what is happening.

const BEARING = { x: 25, y: 23.7246, r: 19.4896 };
const JOURNAL = { x: 27.7402, y: 26.9844, r: 13.0717 };
let spinnersDrawn = 0;

export function busySpinner(sizeInEm) {
    const size = sizeInEm || 1;
    const id = 'ross-bore-' + (++spinnersDrawn);
    const whirl = prefersStillness()
        ? ''
        : `<animateTransform attributeName="transform" type="rotate"`
            + ` from="0 ${BEARING.x} ${BEARING.y}" to="-360 ${BEARING.x} ${BEARING.y}"`
            + ` dur="2s" repeatCount="indefinite"/>`;
    return `<svg class="ross-spinner" viewBox="0 0 50 47.45"`
        + ` style="width:${size}em; height:${size}em;" aria-hidden="true">`
        + `<mask id="${id}"><rect width="50" height="47.45" fill="#fff"/>`
        + `<g>${whirl}<circle cx="${JOURNAL.x}" cy="${JOURNAL.y}" r="${JOURNAL.r}" fill="#000"/></g></mask>`
        + `<circle cx="${BEARING.x}" cy="${BEARING.y}" r="${BEARING.r}" fill="currentColor" mask="url(#${id})"/>`
        + `</svg>`;
}

function prefersStillness() {
    return typeof window.matchMedia === 'function'
        && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}
