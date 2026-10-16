// Editing from the 3D view: a click opens the part's row, Delete removes it,
// a double click opens the node hub.
//
// WHY THIS BATTERY EXISTS. The 3D view edits through the paths the element list
// already uses -- `editItem`, `deleteItem`, the node hub -- so that undo, the
// rules of each form and the question about a material in use hold there
// without a line of their own. The one thing it adds is getting to the row:
// the right tab and, on a MultiRotor, the right half. A row reached in the
// wrong half would edit, or delete, an element of the other shaft line -- with
// no error, and with the screen showing the list of the half it went to. So
// the claims are about which row of which list each gesture reaches.
//
// The canvas gestures themselves (a click that is not a drag, a double click
// that is not two clicks) need a browser; they were measured in one. Here are
// the calls those gestures make.
import { clearDom, node, schemaResponse } from './fake_dom.js';

globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    return { ok: true, status: 200, json: async () => ({ status: 'success', mass: 1, ip: 1, scene: null }) };
};

const { state } = await import('../../frontend/core/state.js');
const { schemaReady } = await import('../../frontend/core/schema.js');
const { addFrom3d, deleteFrom3d, editFrom3d } = await import('../../frontend/features/modeling.js');
const { hitPoint, nearestNode } = await import('../../frontend/core/rotor3d_layout.js');

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}
const settle = ms => new Promise(done => setTimeout(done, ms));

await schemaReady();

function line(gearTag) {
    return {
        name: gearTag, materials: [], shafts: [{ n: '0', L: '250' }, { n: '1', L: '250' }],
        disks: [{ n: '1', m: '10', tag: gearTag + '-disk' }], gears: [{ n: '1', tag: gearTag + '-gear' }],
        couplings: [], seals: [], bearings: [], pointmasses: [],
    };
}

// --- one rotor --------------------------------------------------------------------
console.log('\nOne rotor');

clearDom();
state.projectData = line('only');
state.rotorLibrary = [state.projectData];
state.activeRotorIndex = 0;
state.currentTab = 'shafts';
state.editingIndex = -1;

editFrom3d('disks', 0, null);
await settle(20);
check('a click on a disk opens the disks tab', state.currentTab === 'disks');
check('and the form of that row', state.editingIndex === 0 && node('insertion-form').style.display === 'block');

editFrom3d('gears', 0, null);
await settle(20);
check('a click on a gear moves to the gears tab and its row', state.currentTab === 'gears' && state.editingIndex === 0);

deleteFrom3d('disks', 0, null);
await settle(20);
check('Delete on a disk removes that disk', state.projectData.disks.length === 0);
check('and nothing else', state.projectData.gears.length === 1 && state.projectData.shafts.length === 2);

// The list hidden to give the figure room stays hidden: the form is a window
// of its own and is seen without it.
const panelClasses = new Set(['collapsed']);
node('list-panel').classList = {
    add: n => panelClasses.add(n), remove: n => panelClasses.delete(n),
    toggle: n => (panelClasses.has(n) ? panelClasses.delete(n) : panelClasses.add(n)),
    contains: n => panelClasses.has(n),
};
editFrom3d('gears', 0, null);
await settle(20);
check('a click with the list hidden opens the form and leaves the list hidden',
    state.editingIndex === 0 && node('insertion-form').style.display === 'block' && panelClasses.has('collapsed'));

addFrom3d(1, null);
check('a double click near node 1 opens the node hub on node 1',
    node('node-hub-overlay').style.display === 'flex' && String(node('node-hub-target').innerText) === '1');

// --- a MultiRotor -----------------------------------------------------------------
console.log('\nA MultiRotor: the half the part belongs to');

clearDom();
state.projectData = {
    name: 'M', isMultiRotor: true, driving_rotor: line('driving'), driven_rotor: line('driven'),
    multi_params: { coupled_nodes: '1, 1' },
};
state.rotorLibrary = [state.projectData];
state.multiRotorEditTarget = 'driving';
state.currentTab = 'gears';
state.editingIndex = -1;

editFrom3d('gears', 0, 'driven');
await settle(20);
check('a click on the driven line\'s gear switches to the driven half',
    state.multiRotorEditTarget === 'driven' && state.currentTab === 'gears' && state.editingIndex === 0);
// Same tab, other half: the list and its half selector have to be redrawn,
// or the screen shows the driving line's list while editing the driven one.
check('and the list is redrawn for that half, even on the same tab',
    /value="driven" selected/.test(node('tab-title').innerHTML));

deleteFrom3d('disks', 0, 'driven');
await settle(20);
check('Delete there removes the driven line\'s disk', state.projectData.driven_rotor.disks.length === 0);
check('and leaves the driving line\'s alone', state.projectData.driving_rotor.disks.length === 1);

editFrom3d('disks', 0, 'driving');
await settle(20);
check('a click on the driving line goes back to the driving half',
    state.multiRotorEditTarget === 'driving' && state.currentTab === 'disks' && state.editingIndex === 0);

addFrom3d(2, 'driven');
check('the node hub opens on the driven line\'s node, with the driven half chosen',
    state.multiRotorEditTarget === 'driven' && String(node('node-hub-target').innerText) === '2');

// --- which node a double click means ---------------------------------------------
console.log('\nThe node under the pointer');

const rings = [
    { half: 'driving', n: 0, z: 0 }, { half: 'driving', n: 1, z: 0.25 }, { half: 'driving', n: 2, z: 0.5 },
    { half: 'driven', n: 0, z: 0.25 }, { half: 'driven', n: 1, z: 0.35 },
];
check('the node nearest the point, on its own line', nearestNode(rings, 'driving', 0.3) === 1);
check('not the other line\'s, even when that one is nearer', nearestNode(rings, 'driven', 0.3) === 0);
check('no node on a line that has none', nearestNode(rings, null, 0.3) === null);
const at = hitPoint([0, 1, 0.3], [0, -1, 0], 0.95);
check('the point the pointer is on is where the ray meets the part', at[0] === 0 && Math.abs(at[1] - 0.05) < 1e-12 && at[2] === 0.3);

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
