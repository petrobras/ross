// The element form as a window of its own: where it opens, where it may go,
// and what its heading says.
//
// WHY THIS BATTERY EXISTS. The form left the element list for a floating
// window (components/floating_form.js). Two things can go wrong with that and
// look like nothing at all:
//
//   * the window ends up where it cannot be reached -- dragged past an edge,
//     or remembered from a bigger screen -- and the form seems not to open;
//   * the window says nothing about which element it is. Under its row, the
//     form's place said it; in a window, only the heading and the marked row
//     can, and a heading that goes stale after a drag or a copy would put one
//     element's name over another's form.
//
// The drag itself is measured in a browser; here are the rules it follows.
import { disk, check, node, registerSelector, schemaResponse, shutDown } from './fake_dom.js';

globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    return { ok: true, status: 200, json: async () => ({ status: 'success', mass: 1, ip: 1, scene: null }) };
};

// The browser window's own listeners (the resize).
const windowListeners = {};
globalThis.addEventListener = (type, handler) => { (windowListeners[type] ||= []).push(handler); };

const { firstPlace, keepInside, placeFormWindow, startFormWindow } = await import('../../frontend/components/floating_form.js');
const { markEditedRow } = await import('../../frontend/components/list.js');
const { state } = await import('../../frontend/core/state.js');
const { schemaReady } = await import('../../frontend/core/schema.js');
const { closeForm, editItem, openForm, openTab } = await import('../../frontend/features/modeling.js');

await schemaReady();
const settle = () => new Promise(done => setTimeout(done, 20));
const screen = { width: 1366, height: 800 };

// --- where it may go ----------------------------------------------------------------
console.log('\nWhere the window may go');

const same = (a, b) => a.left === b.left && a.top === b.top;
check('a place already on screen is kept', same(keepInside({ left: 600, top: 130 }, 400, screen), { left: 600, top: 130 }));
check('all of its width stays on screen', keepInside({ left: 1200, top: 130 }, 400, screen).left === 1366 - 400 - 16);
check('never past the left edge or above the top', same(keepInside({ left: -50, top: -50 }, 400, screen), { left: 16, top: 16 }));
// The height is capped to the room under the top, so only the top is kept:
// low enough for the bar and the first fields to show.
check('its bar never below the last 160 px', keepInside({ left: 600, top: 790 }, 400, screen).top === 800 - 160);
check('on a screen narrower than the window, its left edge -- bar and heading -- wins',
    keepInside({ left: 300, top: 100 }, 400, { width: 380, height: 800 }).left === 16);

check('the first place is beside the list, under the mass bar',
    same(firstPlace({ left: 580, top: 52 }, { bottom: 121 }), { left: 596, top: 129 }));
check('and with no mass bar, at the top of the figure',
    same(firstPlace({ left: 220, top: 52 }, null), { left: 236, top: 60 }));

// --- where it opens -----------------------------------------------------------------
console.log('\nWhere it opens');

const box = node('insertion-form');
box.offsetWidth = 400;
node('sel:#screen-modeling .plot-panel').getBoundingClientRect = () => ({ left: 580, top: 52, right: 1366, bottom: 800 });
node('rotor-info').getBoundingClientRect = () => ({ left: 600, top: 72, right: 1346, bottom: 121 });

function reopen() {
    box.style.display = 'none';
    placeFormWindow();
    return { left: parseFloat(box.style.left), top: parseFloat(box.style.top) };
}

delete disk.content['ross-form-window'];
check('the first time, in its first place', same(reopen(), { left: 596, top: 129 }));
check('shown by being placed', box.style.display === 'block' && box.style.visibility === '');
check('its height capped to the room under it', box.style.maxHeight === (800 - 129 - 16) + 'px');

disk.content['ross-form-window'] = JSON.stringify({ left: 900, top: 200 });
check('after that, where it was last put', same(reopen(), { left: 900, top: 200 }));

// Put on a big monitor, opened on the laptop.
disk.content['ross-form-window'] = JSON.stringify({ left: 3000, top: 2000 });
check('a place remembered off this screen is brought back into view', same(reopen(), { left: 950, top: 640 }));
check('without forgetting it, for when the big screen is back',
    disk.content['ross-form-window'] === JSON.stringify({ left: 3000, top: 2000 }));

disk.content['ross-form-window'] = '{not json';
check('a damaged memory falls back to the first place', same(reopen(), { left: 596, top: 129 }));

box.style.display = 'block';
box.style.left = '700px';
placeFormWindow();
check('an open window stays put when another element is opened in it', box.style.left === '700px');

// --- dragging ------------------------------------------------------------------------
console.log('\nDragging by the bar');

const bar = node('insertion-form:q');
const barListeners = {};
bar.addEventListener = (type, handler) => { barListeners[type] = handler; };
bar.setPointerCapture = () => {};
startFormWindow();

box.style.left = '596px';
box.style.top = '129px';
box.getBoundingClientRect = () => ({ left: 596, top: 129 });
const onBar = { closest: () => null };
const press = (x, y, target = onBar) =>
    barListeners.pointerdown({ button: 0, pointerId: 1, clientX: x, clientY: y, target, preventDefault() {} });
const move = (x, y) => barListeners.pointermove({ pointerId: 1, clientX: x, clientY: y });
const release = () => barListeners.pointerup({ pointerId: 1 });

delete disk.content['ross-form-window'];
press(620, 140);
move(820, 240);
check('the window follows the pointer, held where it was grabbed', box.style.left === '796px' && box.style.top === '229px');
move(2000, 240);
check('and stops at the edge of the screen', box.style.left === '950px');
release();
check('where it is dropped is remembered', disk.content['ross-form-window'] === JSON.stringify({ left: 950, top: 229 }));
move(700, 300);
check('after the drop, the pointer no longer moves it', box.style.left === '950px');

press(620, 140, { closest: () => ({}) });   // on the close button, inside the bar
move(700, 300);
check('a press on the close button is not a drag', box.style.left === '950px');
release();

document.documentElement.clientWidth = 1024;
windowListeners.resize.forEach(handler => handler());
check('a narrower browser window brings it back into view', box.style.left === (1024 - 400 - 16) + 'px');
document.documentElement.clientWidth = 1366;

// --- what the heading says -------------------------------------------------------------
console.log('\nThe heading');

function tabButton(category, key) {
    const b = node('tab:' + category);
    b.dataset.tab = category;
    b.dataset.i18n = key;
    return b;
}
registerSelector('.tab-btn', [tabButton('disks', 'catDisk'), tabButton('gears', 'catGear')]);

function line(name) {
    return { name, materials: [], shafts: [{ n: '0', L: '250' }, { n: '1', L: '250' }],
             disks: [{ n: '1', m: '10' }, { n: '2', m: '5' }], gears: [{ n: '1' }],
             couplings: [], seals: [], bearings: [], pointmasses: [] };
}
state.projectData = line('R');
state.rotorLibrary = [state.projectData];
state.activeRotorIndex = 0;
box.style.display = 'none';

openTab('disks');
await openForm(true);
check('a new element: the kind being added', node('form-window-title').textContent === 'New element: Disk');
editItem(1);
await settle();
check('an element being edited: its row, in the row\'s words', node('form-window-title').textContent === 'DISK #2 (Node 2)');
closeForm();
check('closed: no heading left behind', node('form-window-title').textContent === '');

// On a MultiRotor, the list says which line it shows. With the list hidden,
// only the window can -- and the two lines number their elements alike.
state.projectData = { name: 'M', isMultiRotor: true, driving_rotor: line('A'), driven_rotor: line('B'),
                      multi_params: { coupled_nodes: '1, 1' } };
state.rotorLibrary = [state.projectData];
state.multiRotorEditTarget = 'driven';
openTab('gears');
editItem(0);
await settle();
check('on a MultiRotor, the line it belongs to', node('form-window-title').textContent === 'GEAR #1 (Node 1) · Driven Rotor');
state.multiRotorEditTarget = 'driving';
markEditedRow();
check('and the other line, named as such', node('form-window-title').textContent === 'GEAR #1 (Node 1) · Driving Rotor');

shutDown();
