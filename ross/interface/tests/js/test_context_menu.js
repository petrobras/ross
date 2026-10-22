// The 3D view's right-button menu: what it offers on each part, where a
// "split here" cuts, and that each entry reaches the row of the part it was
// opened on -- in the right tab and, on a MultiRotor, the right half.
//
// WHY THIS BATTERY EXISTS. The menu adds no action of its own: split, mesh,
// edit, copy, hide, add and delete are the list's, reached from the part
// (features/modeling.js), so undo and the questions each one asks hold without
// a line here. What it adds is (1) the distance of a cut read off the pointer
// -- the one number a wrong unit or a wrong face would silently turn into a
// cut somewhere else -- and (2) a menu whose entries run the part it was drawn
// for, and nothing once it is closed. The gestures (a right click that is not
// a drag, Shift+F10) need a browser; here are the calls they make.
import { check, clearDom, node, schemaResponse, shutDown } from './fake_dom.js';

let sent = [];
globalThis.fetch = async (path, options = {}) => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    if (String(path).includes('split_shaft')) {
        sent.push(JSON.parse(options.body));
        const project = JSON.parse(options.body).project;
        project.shafts.splice(1, 0, { n: '', L: '50' });
        return { ok: true, status: 200, json: async () => ({ status: 'success', projectData: project }) };
    }
    return { ok: true, status: 200, json: async () => ({ status: 'success', mass: 1, ip: 1, scene: null }) };
};

const { state } = await import('../../frontend/core/state.js');
const { schemaReady } = await import('../../frontend/core/schema.js');
const { t } = await import('../../frontend/core/i18n.js');
const { blockBrowserMenu, closeContextMenu, contextMenuOpen, openContextMenu, runContextItem } = await import('../../frontend/components/context_menu.js');
const { cutHere, menuEntries } = await import('../../frontend/features/rotor3d.js');
const { DIALOG_ACTIONS } = await import('../../frontend/features/dialog_actions.js');
const settle = ms => new Promise(done => setTimeout(done, ms));

await schemaReady();

// --- the menu itself ------------------------------------------------------------------
console.log('\nThe menu');

clearDom();
let ran = [];
openContextMenu(5000, 5000, [
    { header: 'Shaft #1' },
    { label: 'First', icon: 'fa-pen', run: () => ran.push('first') },
    null,
    { label: 'Off', icon: 'fa-cut', disabled: true, run: () => ran.push('off') },
    { label: 'Gone <b>', icon: 'fa-trash', danger: true, run: () => ran.push('gone') },
]);
const html = node('context-menu').innerHTML;
check('it opens, with a title, the entries, a separator', contextMenuOpen()
    && /context-header">Shaft #1/.test(html) && /role="separator"/.test(html) && (html.match(/data-action="context-item"/g) || []).length === 3);
check('each entry carries only its position; its function stays in the menu', /data-index="1"/.test(html) && !/run/.test(html));
check('a disabled one is drawn disabled, a dangerous one marked', /data-index="3" disabled/.test(html) && /is-danger/.test(html));
check('what an entry says is escaped', html.includes('Gone &lt;b&gt;'));
check('it is kept inside the view', parseInt(node('context-menu').style.left, 10) < 1366 && parseInt(node('context-menu').style.top, 10) < 800);
DIALOG_ACTIONS['context-item']({ dataset: { index: '1' } });
check('an entry runs its own function, and the menu closes', ran.join() === 'first' && !contextMenuOpen());
check('once closed, no position runs anything', runContextItem(1) === undefined && ran.length === 1);
openContextMenu(10, 10, [{ label: 'Off', disabled: true, run: () => ran.push('off') }]);
runContextItem(0);
check('a disabled entry runs nothing', ran.length === 1);

// Windows asks for the browser's menu after the release that opened ours, and
// from ours, under the pointer: both opened (Leonardo, in Chrome).
const asked = () => { const event = { prevented: false, preventDefault() { this.prevented = true; } }; blockBrowserMenu(event); return event.prevented; };
openContextMenu(10, 10, [{ label: 'One', run: () => {} }]);
check('while ours is open, the browser\'s menu is refused', asked() === true);
closeContextMenu();
check('once ours is closed, the browser\'s menu is the page\'s again', asked() === false);

// --- where a split lands ------------------------------------------------------------------
console.log('\nSplit here: the distance from the pointer');

const shaftPart = { kind: 'shaft', category: 'shafts', index: 0, half: null, z0: 0.25, z1: 0.5, offset: { x: 0, y: 0, z: 0 } };
const at = z => [0, 0, z];
check('50 mm from the left face, in the mm the length is typed in', JSON.stringify(cutHere(shaftPart, at(0.3), { L: '250' })) === JSON.stringify({ shown: '50.0 mm', value: '50' }));
check('in metres, when the length is typed in metres', cutHere(shaftPart, at(0.3), { L: '0.25', L_unit: 'm' }).value === '0.05');
check('in inches, when in inches', Math.abs(Number(cutHere(shaftPart, at(0.3), { L: '9.8', L_unit: 'in' }).value) - 1.9685) < 1e-4);
check('measured from the left face whichever way the element is stored',
    cutHere({ ...shaftPart, z0: 0.5, z1: 0.25 }, at(0.3), {}).value === '50');
check('on a MultiRotor\'s driven line, from its own start', cutHere({ ...shaftPart, offset: { x: 0.2, y: 0, z: 1.0 } }, at(1.3), {}).value === '50');
check('none on a node, where the cut would land on one that exists',
    cutHere(shaftPart, at(0.2501), {}) === null && cutHere(shaftPart, at(0.4999), {}) === null);
check('none in a unit it cannot read, rather than a cut at a guess', cutHere(shaftPart, at(0.3), { L_unit: 'furlong' }) === null);
check('none on anything but a shaft', cutHere({ ...shaftPart, kind: 'disk' }, at(0.3), {}) === null);

// --- the entries, and the rows they reach --------------------------------------------------
console.log('\nWhat each part offers, and the row it reaches');

function rotor() {
    return {
        name: 'R', materials: [], shafts: [{ L: '250', tag: 'Inlet' }, { L: '250' }],
        disks: [{ n: '1', m: '10' }], gears: [], couplings: [], seals: [], bearings: [], pointmasses: [],
    };
}
clearDom();
state.projectData = rotor();
state.rotorLibrary = [state.projectData];
state.activeRotorIndex = 0;
state.currentTab = 'disks';
state.editingIndex = -1;

const labels = entries => entries.filter(Boolean).map(e => e.header || e.label);
const onShaft = menuEntries({ part: shaftPart, point: at(0.3) });
check('a shaft: its name, the three cuts, then edit, copy, hide, delete', JSON.stringify(labels(onShaft)) === JSON.stringify([
    `${t('catShaft')} #1 · Inlet`, t('ctxSplitHere').replace('%1', '50.0 mm'), t('ctxSplitAsk'), t('ctxMesh'),
    t('ctxEdit'), t('ctxCopy'), t('rotor3dHideElement'), t('ctxDelete')]));
check('delete is the last, and marked', onShaft[onShaft.length - 1].danger === true);
const nearNode = menuEntries({ part: shaftPart, point: at(0.25001) });
check('too close to a node, "split here" is offered disabled and says why', nearNode[1].disabled === true && nearNode[1].label === t('ctxSplitHereNo'));
const diskPart = { kind: 'disk', category: 'disks', index: 0, half: null, z0: 0.24, z1: 0.26, offset: { x: 0, y: 0, z: 0 } };
check('a disk: no cut to offer', !labels(menuEntries({ part: diskPart, point: at(0.25) })).some(l => l.includes(t('ctxMesh'))));
check('the empty view: the rotor as a whole', JSON.stringify(labels(menuEntries(null))) === JSON.stringify([t('ctxMeshAll'), t('ctxMeasure'), t('rotor3dFrame')]));

sent = [];
const splitHere = onShaft[1];
await splitHere.run();
await settle(20);
check('"split here" goes to the shafts tab and cuts that shaft at that distance, asking nothing',
    state.currentTab === 'shafts' && sent.length === 1 && sent[0].index === 0 && sent[0].offset === '50');
check('and the project takes the cut', state.projectData.shafts.length === 3);

state.currentTab = 'disks';
state.projectData.disks = [{ n: '1', m: '10', tag: 'D' }];
const copy = menuEntries({ part: diskPart, point: at(0.25) }).find(e => e && e.label === t('ctxCopy'));
copy.run();
await settle(20);
check('copy reaches the disks tab and copies that disk', state.currentTab === 'disks' && state.projectData.disks.length === 2);

// A MultiRotor: the entry of a part of the driven line reaches the driven line.
state.projectData = { name: 'MR', isMultiRotor: true, driving_rotor: rotor(), driven_rotor: rotor() };
state.rotorLibrary = [state.projectData];
state.multiRotorEditTarget = 'driving';
state.currentTab = 'disks';
const drivenDisk = { ...diskPart, half: 'driven' };
menuEntries({ part: drivenDisk, point: at(0.25) }).find(e => e && e.label === t('ctxCopy')).run();
await settle(20);
check('on a MultiRotor it reaches the half the part is on, and only it',
    state.multiRotorEditTarget === 'driven' && state.projectData.driven_rotor.disks.length === 2
    && state.projectData.driving_rotor.disks.length === 1);

shutDown();
