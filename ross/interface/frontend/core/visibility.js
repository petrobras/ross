// What the 3D view is asked not to draw: whole categories, from the legend over
// the view, and single elements, from the eye on their row of the list.
//
// A view setting, like the legend of ROSS's 2D figure: it changes nothing sent
// to ROSS and nothing saved in the project. It lasts while the page is open.
//
// WHY ELEMENTS ARE HELD BY OBJECT AND NOT BY POSITION. Elements have no
// identity of their own (see core/selection.js), and positions are what every
// deletion, copy and drag changes. A hidden set of positions would, after a
// drag, hide the element that took the place -- one element vanishing and
// another coming back, which reads as the view being wrong. Held by the object
// in the project, the mark goes where the element goes. It is lost when the
// object is replaced: saving its form, undoing, loading. That shows it again,
// which is the safe way to be wrong: nothing stays out of sight unnoticed.
// A `WeakSet`, so a hidden element that is deleted takes its mark with it.
//
// This module also knows whether the 3D view is the one on screen, because the
// list's eyes are drawn only then: in the 2D figure (ROSS's own, in Plotly)
// hiding one element is not possible, and an eye that does nothing is worse
// than none.

// The categories the legend offers, in the order it shows them. `nodes` is the
// rings on the shaft line, which ROSS's 2D figure also labels.
export const LEGEND_CATEGORIES = ['shafts', 'disks', 'gears', 'couplings', 'bearings', 'seals', 'pointmasses', 'probes', 'nodes'];

const hiddenElements = new WeakSet();
const hiddenCategories = new Set();
let threeD = false;

export function setThreeDShown(shown) {
    threeD = !!shown;
}

export function threeDShown() {
    return threeD;
}

export function elementHidden(element) {
    return !!element && typeof element === 'object' && hiddenElements.has(element);
}

export function toggleElementHidden(element) {
    if (!element || typeof element !== 'object') return false;
    if (hiddenElements.has(element)) hiddenElements.delete(element);
    else hiddenElements.add(element);
    return hiddenElements.has(element);
}

export function categoryHidden(category) {
    return hiddenCategories.has(category);
}

// One click on the legend: this category in or out.
export function toggleCategory(category) {
    if (hiddenCategories.has(category)) hiddenCategories.delete(category);
    else hiddenCategories.add(category);
}

// A double click, as in Plotly's legend: only this category -- or, when it is
// already the only one shown, all of them again. `present` is the categories
// the rotor on screen has; the others are not counted.
export function isolateCategory(category, present) {
    const others = present.filter(c => c !== category);
    const alreadyAlone = !hiddenCategories.has(category) && others.every(c => hiddenCategories.has(c));
    hiddenCategories.clear();
    if (!alreadyAlone) others.forEach(c => hiddenCategories.add(c));
}

// Whether a part is drawn: neither its category nor itself is hidden.
export function drawn(category, element) {
    return !hiddenCategories.has(category) && !elementHidden(element);
}

// What a click on a legend chip does, as Plotly's legend: the first click of
// a double click toggles the chip, and its second click (`detail` 2) arrives
// after that. The toggle is taken back before isolating, or "is it already
// alone?" would be asked of a state the person never chose -- and a second
// double click would never bring the others back (measured in a browser).
export function legendClick(category, detail, present) {
    if (detail >= 2) {
        toggleCategory(category);
        isolateCategory(category, present);
    } else {
        toggleCategory(category);
    }
}
