// Hiding the element list by clicking the open tab again.
//
// WHY THIS BATTERY EXISTS. The gesture is one line of CSS; what can go wrong is
// *who* toggles. `openTab` is called from three places -- the sidebar button,
// `switchScreen` on the way back from the analyses, and the node hub -- and
// only the first is a person asking for the list to go away. A toggle placed in
// `openTab` would hide the list every other time the modeling screen came back,
// and a form opened from the figure would open into a panel nobody can see.
//
// So what is checked here is mostly what must **not** hide or reset the panel.
import { check, node, registerSelector, schemaResponse, shutDown } from './fake_dom.js';

// The fake DOM's `classList` answers "no" to everything, which would make the
// panel look open forever. This one keeps what it is told.
function realClassList() {
    const names = new Set();
    return {
        add: n => names.add(n),
        remove: n => names.delete(n),
        toggle: n => (names.has(n) ? names.delete(n) : names.add(n)),
        contains: n => names.has(n),
    };
}
const panel = node('list-panel');
panel.classList = realClassList();
const hidden = () => panel.classList.contains('collapsed');

// The buttons, found the way the application finds them: by `data-tab`.
function tabButton(category, key) {
    const b = node('tab:' + category);
    b.classList = realClassList();
    b.dataset.tab = category;
    b.dataset.i18n = key;
    return b;
}
registerSelector('.tab-btn', [tabButton('shafts', 'catShaft'), tabButton('disks', 'catDisk')]);

// What the figure hears. Plotly follows the window, so a panel that changes
// width without a `resize` leaves the figure the width it was.
let resizes = 0;
globalThis.dispatchEvent = event => { if (event.type === 'resize') resizes++; };
const settle = () => new Promise(done => setTimeout(done, 350));

globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    return { ok: true, status: 200, json: async () => ({
        status: 'success', plot_json: JSON.stringify({ data: [], layout: {} }),
    }) };
};

const { editItem, pickTab } = await import('../../frontend/features/modeling.js');
const { switchScreen } = await import('../../frontend/features/screens.js');
const { openProjectHistory, state } = await import('../../frontend/core/state.js');

state.rotorLibrary = [{
    name: 'Compressor', uid: 'uid_c', savedAnalyses: [], materials: [],
    shafts: [{ element_type: 'BASIC', n: '0', L: '250', odl: '50' }],
    disks: [{ element_type: 'BASIC', n: '1', m: '10' }],
    gears: [], couplings: [], seals: [], bearings: [], pointmasses: [],
}];
state.activeRotorIndex = 0;
state.projectData = state.rotorLibrary[0];
state.currentTab = null;
openProjectHistory(state.projectData);

// --- the gesture ------------------------------------------------------------------

pickTab('shafts');
check('the first click opens the tab', state.currentTab === 'shafts');
check('with the list showing', !hidden());
check('and its button lit', node('tab:shafts').classList.contains('active'));
// The lookup that moved from the `onclick` text to `data-tab`: a heading that
// came out as the internal key would mean the button was not found.
check('and the heading names the category', /Shaft/.test(node('tab-title').innerHTML));

resizes = 0;
pickTab('shafts');
check('clicking the open tab again hides the list', hidden());
check('without leaving the tab', state.currentTab === 'shafts');
await settle();
check('and the figure is told its width changed', resizes === 1);

// --- coming back ---------------------------------------------------------------------

// Something under way in the form when the list was hidden.
state.editingIndex = 0;
pickTab('shafts');
check('clicking it once more shows the list again', !hidden());
// `openTab` closes the form. Reopening the same tab must not go through it, or
// hiding the list would cost whatever was being typed.
check('with the form that was open still open', state.editingIndex === 0);

pickTab('shafts');
pickTab('disks');
check('another tab, from hidden, shows the list', !hidden());
check('and opens that tab', state.currentTab === 'disks');
check('with its own button lit', node('tab:disks').classList.contains('active')
    && !node('tab:shafts').classList.contains('active'));

// Control on the other direction: a different tab with the list showing is an
// ordinary tab change, not a hide.
pickTab('shafts');
check('changing tabs with the list showing keeps it showing', !hidden());

// --- what is not the person asking ------------------------------------------------------

pickTab('shafts');
check('setting the scene: the list is hidden', hidden());

// The return from the analyses calls `openTab` with the tab that is already
// open. If that toggled, the list would come back on every second visit.
switchScreen('screen-modeling');
check('coming back to the modeling screen leaves it hidden', hidden());
switchScreen('screen-modeling');
check('every time', hidden());

// A form opened from the figure lives in the panel.
editItem(0);
await new Promise(done => setTimeout(done, 50));
check('editing an element from elsewhere brings the list back', !hidden());

shutDown();
