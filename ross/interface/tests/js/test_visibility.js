// Hiding parts of the 3D view: whole categories from its legend, one element
// from the eye on its row, and the "move" button.
//
// WHY THIS BATTERY EXISTS. Hiding is a view setting, and the ways it goes
// wrong look like the model going wrong:
//
//   * an element hidden by its *position* would, after a drag, leave another
//     element hidden in its place -- one part vanishing, another coming back;
//   * a double click on the legend arrives as a click (which toggles) and then
//     a second click: judged after the first, "is it already alone?" answers
//     a state nobody chose, and the second double click never brings the
//     others back (measured in a browser before this was written);
//   * an eye drawn under ROSS's 2D figure, which cannot hide one element,
//     would be a button that does nothing.
import { check, node, registerSelector, schemaResponse, shutDown } from './fake_dom.js';

globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    return { ok: true, status: 200, json: async () => ({ status: 'success', mass: 1, ip: 1, scene: null }) };
};

const V = await import('../../frontend/core/visibility.js');
const { state } = await import('../../frontend/core/state.js');
const { renderList } = await import('../../frontend/components/list.js');
const { MODELING_ACTIONS } = await import('../../frontend/features/modeling_actions.js');
const { setRotorView } = await import('../../frontend/features/modeling.js');

// --- categories ------------------------------------------------------------------------
console.log('\nThe legend: categories');

const present = ['shafts', 'disks', 'bearings', 'nodes'];
V.toggleCategory('disks');
check('a click hides a category', V.categoryHidden('disks') && !V.drawn('disks', {}));
V.toggleCategory('disks');
check('a second click shows it again', !V.categoryHidden('disks') && V.drawn('disks', {}));

V.isolateCategory('bearings', present);
check('a double click leaves only that category',
    !V.categoryHidden('bearings') && ['shafts', 'disks', 'nodes'].every(V.categoryHidden));
V.isolateCategory('bearings', present);
check('and another double click on it brings all back', present.every(c => !V.categoryHidden(c)));

// The clicks a browser sends for a double click: detail 1, then detail 2.
const doubleClick = category => { V.legendClick(category, 1, present); V.legendClick(category, 2, present); };
doubleClick('bearings');
check('a double click, as a browser sends it, isolates',
    !V.categoryHidden('bearings') && ['shafts', 'disks', 'nodes'].every(V.categoryHidden));
doubleClick('bearings');
check('and a second one brings all back -- judged on the state before its first click',
    present.every(c => !V.categoryHidden(c)));
V.legendClick('disks', 1, present);
check('a single click only toggles', V.categoryHidden('disks') && !V.categoryHidden('shafts'));
V.legendClick('disks', 1, present);

// Through the action, as the chip calls it.
MODELING_ACTIONS['hide-category']({ dataset: { category: 'seals' } }, { detail: 1 });
check('the chip\'s action toggles its category', V.categoryHidden('seals'));
MODELING_ACTIONS['hide-category']({ dataset: { category: 'seals' } }, { detail: 1 });

// --- elements --------------------------------------------------------------------------
console.log('\nThe eye: one element, held by the object and not by its place');

const a = { n: '1', m: '10', tag: 'A' };
const b = { n: '2', m: '5', tag: 'B' };
const disks = [a, b];
V.toggleElementHidden(a);
check('an element hidden', V.elementHidden(a) && !V.drawn('disks', a));
check('its neighbour is not', !V.elementHidden(b) && V.drawn('disks', b));
disks.reverse();
check('after a drag, the hidden one is still the one hidden', V.elementHidden(disks[1]) && !V.elementHidden(disks[0]));
const copy = JSON.parse(JSON.stringify(a));
check('a copy of it, or the element saved by its form, is shown', !V.elementHidden(copy));
check('nothing that is not an element counts as hidden', !V.elementHidden(null) && !V.elementHidden('disks'));
V.toggleElementHidden(a);
check('and the eye brings it back', !V.elementHidden(a));

// --- the list ---------------------------------------------------------------------------
console.log('\nThe eyes in the list, only under the 3D view');

state.projectData = { name: 'R', materials: [{ name: 'Steel' }], shafts: [{ n: '0', L: '1' }],
    disks: [{ n: '0', m: '1', tag: 'D0' }, { n: '1', m: '2', tag: 'D1' }], gears: [], couplings: [],
    seals: [], bearings: [], pointmasses: [] };
state.rotorLibrary = [state.projectData];
state.activeRotorIndex = 0;
state.currentTab = 'disks';
state.editingIndex = -1;

// The rows on screen: the fake DOM's `innerHTML = ''` does not drop children,
// so the last redraw is the tail of the list.
const rows = () => node('element-list').children.slice(-state.projectData[state.currentTab].length);
setRotorView('2d');
renderList();
check('with the 2D figure: no eye on any row', rows().every(r => !/data-action="hide-element"/.test(r.innerHTML)));

setRotorView('3d');
check('switching to 3D redraws the list with an eye on each row',
    rows().length === 2 && rows().every(r => /data-action="hide-element"/.test(r.innerHTML)));

const second = state.projectData.disks[1];
MODELING_ACTIONS['hide-element']({ dataset: { index: '1' } });
check('the eye hides the element of its own row', V.elementHidden(second) && !V.elementHidden(state.projectData.disks[0]));
check('the row says so: crossed eye, pressed, faded',
    /fa-eye-slash/.test(rows()[1].innerHTML) && /aria-pressed="true"/.test(rows()[1].innerHTML)
    && rows()[1].className.includes('is-out-of-view') && !rows()[0].className.includes('is-out-of-view'));

V.toggleCategory('disks');
renderList();
check('a category hidden from the legend fades all its rows',
    rows().every(r => r.className.includes('is-out-of-view')));
V.toggleCategory('disks');

state.currentTab = 'materials';
renderList();
check('materials are not drawn: no eye there', rows().every(r => !/hide-element/.test(r.innerHTML)));
state.currentTab = 'disks';

setRotorView('2d');
check('back in 2D the eyes go', rows().every(r => !/hide-element/.test(r.innerHTML)) && !V.threeDShown());

// --- the move button --------------------------------------------------------------------
console.log('\nThe move button');

const panButton = node('pan-button');
registerSelector('[data-action="toggle-pan"]', [panButton]);
MODELING_ACTIONS['toggle-pan']();
check('pressed before the 3D view ever opened, it says so', panButton.getAttribute('aria-pressed') === 'true');
MODELING_ACTIONS['toggle-pan']();
check('and pressed again, it is off', panButton.getAttribute('aria-pressed') === 'false');

shutDown();
