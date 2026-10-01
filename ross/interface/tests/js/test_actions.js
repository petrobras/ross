// What the buttons mean, by name -- and that every name means something.
//
// WHY THIS BATTERY EXISTS. The page moved off the `window` bridge
// (core/actions.js, slices 12 to 15): a button says `data-action="save-rotor"`
// and one listener looks the name up. The failure this trades away was silent -- a
// function missing from `window` left its button dead. The failure it could
// introduce is the same one under a new name: a `data-action` nobody defined.
// So the table is read against the HTML in both directions, every action is
// pressed once, and the listener's rules are checked on plain objects.
import { check, node, schemaResponse, shutDown } from './fake_dom.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    return { ok: true, status: 200, json: async () => schema || { status: 'success' } };
};
// Screens and side panels announce a resize; Node's global is not an EventTarget.
globalThis.dispatchEvent = () => true;

const { actionNames, defineActions, handleEvent, runAction, startActions } =
    await import('../../frontend/core/actions.js');
await import('../../frontend/main.js');

const FRONTEND = fileURLToPath(new URL('../../frontend/', import.meta.url));
const PAGE = fs.readFileSync(path.join(FRONTEND, 'index.html'), 'utf8');

function sources(folder) {
    return fs.readdirSync(folder, { withFileTypes: true }).flatMap(entry => {
        const full = path.join(folder, entry.name);
        if (entry.isDirectory()) return ['lib', 'vendor'].includes(entry.name) ? [] : sources(full);
        return entry.name.endsWith('.js') ? [fs.readFileSync(full, 'utf8')] : [];
    });
}
const TEXTS = [PAGE, ...sources(FRONTEND)];
const named = new Set();
for (const text of TEXTS) for (const m of text.matchAll(/data-action="([a-z-]+)"/g)) named.add(m[1]);

// --- the table and the page agree -------------------------------------------------

check('the page names actions at all', named.size > 10);
const undefinedNames = [...named].filter(name => !actionNames().includes(name));
check('every action the page names is defined: ' + undefinedNames.join(', '), undefinedNames.length === 0);
const unusedNames = actionNames().filter(name => !named.has(name));
check('and every defined action is named somewhere: ' + unusedNames.join(', '), unusedNames.length === 0);

// The attributes the adapters read have to point at something that exists.
const ids = new Set([...PAGE.matchAll(/\sid="([\w-]+)"/g)].map(m => m[1]));
const inputs = [...PAGE.matchAll(/data-action="choose-file" data-input="([\w-]+)"/g)].map(m => m[1]);
check('every file button names its input', inputs.length === 4);
check('and each of those inputs is on the page', inputs.every(id => ids.has(id)));
const screens = [...PAGE.matchAll(/data-action="show-screen" data-screen="([\w-]+)"/g)].map(m => m[1]);
check('every screen button names a screen that exists', screens.length > 0 && screens.every(id => ids.has(id)));

// Every tab and every node-hub button carries the category it acts on.
const CATEGORIES = ['materials', 'shafts', 'disks', 'gears', 'couplings', 'seals', 'bearings', 'pointmasses'];
const tabButtons = (PAGE.match(/data-action="pick-tab"/g) || []).length;
const tabsNamed = [...PAGE.matchAll(/data-action="pick-tab" data-tab="(\w+)"/g)].map(m => m[1]);
check('every category tab names its category', tabButtons === 8
    && tabsNamed.length === 8 && tabsNamed.every(c => CATEGORIES.includes(c)));
const hubButtons = (PAGE.match(/data-action="add-from-node-hub"/g) || []).length;
const hubNamed = [...PAGE.matchAll(/data-action="add-from-node-hub" data-category="(\w+)"/g)].map(m => m[1]);
check('and so does every node-hub button', hubButtons === 7
    && hubNamed.length === 7 && hubNamed.every(c => CATEGORIES.includes(c)));

// Every element that names an action carries what that action reads. An
// action whose attribute is missing does not fail on the click: it runs with
// `undefined` -- a card deleted under the wrong id, an analysis recalculated as
// no analysis at all -- so the attributes are checked where they are written.
const REQUIRES = {
    'run-card': ['data-card', 'data-type'],
    'delete-card': ['data-card'],
    'toggle-card': ['data-card'],
    'card-help': ['data-type'],
    'check-deps': ['data-card'],
    'add-row': ['data-list', 'data-card', 'data-field', 'data-type'],
    'open-rotor': ['data-index', 'data-screen'],
    'rename-rotor': ['data-index'], 'copy-rotor': ['data-index'], 'delete-rotor': ['data-index'],
    'save-rotor-file': ['data-index'], 'export-rotor-python': ['data-index'],
    'edit-element': ['data-index'], 'copy-element': ['data-index'], 'delete-element': ['data-index'],
    'split-element': ['data-index'], 'pick-element': ['data-index'],
    'pick-subtype': ['data-subtype'], 'section-help': ['data-category'],
    'choose-file': ['data-input'], 'show-screen': ['data-screen'],
};
const missing = [];
for (const text of TEXTS) {
    for (const tag of text.matchAll(/<[a-z]+\b[^<>]*data-action="([a-z-]+)"[^<>]*>/g)) {
        for (const attribute of REQUIRES[tag[1]] || []) {
            if (!tag[0].includes(attribute + '=')) missing.push(tag[1] + ' without ' + attribute);
        }
    }
}
check('every element carries what its action reads: ' + [...new Set(missing)].join(', '), missing.length === 0);
// Control: the sweep does find the tags it checks.
check('control: the attribute sweep sees the generated cards',
    TEXTS.some(text => /<button[^<>]*data-action="run-card"[^<>]*>/.test(text)));

// --- the listener's rules ------------------------------------------------------------

let pressed = [];
defineActions({ 'battery-probe': (element, event) => { pressed.push([element, event]); return 'done'; } });

function element(tagName, action, extra) {
    const el = Object.assign({ tagName, dataset: { action }, disabled: false }, extra || {});
    el.closest = () => el;
    return el;
}

const button = element('BUTTON', 'battery-probe');
handleEvent({ type: 'click', target: button });
check('a click on a button runs its action', pressed.length === 1 && pressed[0][0] === button);

// The icon inside a button is what the pointer usually lands on.
const icon = { closest: () => button };
handleEvent({ type: 'click', target: icon });
check('and so does a click on the icon inside it', pressed.length === 2);

handleEvent({ type: 'click', target: element('BUTTON', 'battery-probe', { disabled: true }) });
check('a disabled button does nothing', pressed.length === 2);

// A hidden file input is opened by `input.click()`; that click must not run the
// input's own action before any file was chosen.
handleEvent({ type: 'click', target: element('INPUT', 'battery-probe') });
check('a click on a form field does not run its action', pressed.length === 2);
handleEvent({ type: 'change', target: element('SELECT', 'battery-probe') });
check('its change does', pressed.length === 3);
handleEvent({ type: 'change', target: element('BUTTON', 'battery-probe') });
check('and a change never runs a button', pressed.length === 3);

handleEvent({ type: 'click', target: { closest: () => null } });
check('a click on nothing in particular is nothing', pressed.length === 3);

let thrown = null;
try { runAction(element('BUTTON', 'no-such-thing')); } catch (error) { thrown = error; }
// The whole point: the dead-button failure now has a name, and the error
// notice shows it.
check('a name nobody defined fails, and says which', thrown && /no-such-thing/.test(thrown.message));

thrown = null;
try { defineActions({ 'battery-probe': () => {} }); } catch (error) { thrown = error; }
check('a name defined twice is refused', thrown && /twice/.test(thrown.message));

// The listener itself: one for clicks and one for changes, on the document.
// Without the second, the language selects and the file inputs go dead.
const listening = {};
const realAdd = document.addEventListener;
document.addEventListener = (type, handler) => { listening[type] = handler; };
startActions();
document.addEventListener = realAdd;
check('the document listens for clicks', listening.click === handleEvent);
check('and for changes', listening.change === handleEvent);

// --- what a few of them do -------------------------------------------------------------

let opened = null;
node('upload-rotor-hub').click = () => { opened = 'upload-rotor-hub'; };
runAction(element('BUTTON', 'choose-file', { dataset: { action: 'choose-file', input: 'upload-rotor-hub' } }));
check('a file button opens the input it names', opened === 'upload-rotor-hub');

// --- every action, pressed once ----------------------------------------------------------
//
// The promise the old smoke battery made for the bridge, now for the table:
// nothing throws on the way in. `exit` is left out: it shuts the server down
// and closes the window.
const LEFT_OUT = ['exit', 'battery-probe'];
const event = { preventDefault() {}, stopPropagation() {}, target: { files: [], value: '' } };
// Whatever an action reads from its element: a file input, a screen, a row, a
// category, a subtype. One set serves them all.
const attributes = { input: 'upload-rotor-hub', screen: 'screen-modeling', index: '0',
                     tab: 'shafts', category: 'shafts', subtype: 'BASIC',
                     // an analysis card and one of its row editors
                     card: 'x1', type: 'modes', list: 'probe_list', field: 'probes' };
const failures = [];
// Some actions start work they do not return (`editItem` opens the form and
// hands back nothing); a failure there would end the battery with no name on
// it. Collected instead.
process.on('unhandledrejection', error => failures.push('(not returned) ' + (error && error.message)));

// A rotor open on the shafts tab, afresh before every action, so the modelling
// actions have a row to act on -- and `delete-element` cannot take away the row
// `edit-element` is about to open.
const { openProjectHistory, state } = await import('../../frontend/core/state.js');
function openRotor() {
    state.rotorLibrary = [{
        name: 'R', uid: 'uid_r', savedAnalyses: [], materials: [],
        shafts: [{ element_type: 'BASIC', n: '0', L: '250', odl: '50' },
                 { element_type: 'BASIC', n: '1', L: '250', odl: '50' }],
        disks: [], gears: [], couplings: [], seals: [], bearings: [], pointmasses: [],
    }];
    state.activeRotorIndex = 0;
    state.projectData = state.rotorLibrary[0];
    state.currentTab = 'shafts';
    state.editingIndex = -1;
    openProjectHistory(state.projectData);
}
for (const name of actionNames().filter(n => !LEFT_OUT.includes(n))) {
    const el = element(name === 'change-language' ? 'SELECT' : 'BUTTON', name,
        // `toggle-advanced` opens the block right after its button.
        { dataset: Object.assign({ action: name }, attributes), value: 'en',
          nextElementSibling: { style: { display: 'none' } },
          // `remove-row` removes the row its button sits in.
          remove() {} });
    openRotor();
    try {
        const answer = runAction(el, event);
        // Not awaited: several of them open a dialog and wait for an answer
        // that never comes here. Their failures are still collected.
        if (answer && typeof answer.then === 'function') answer.catch(error => failures.push(name + ': ' + error.message));
    } catch (error) {
        failures.push(name + ': ' + error.message);
    }
    // Let the ones that open the form finish before the rotor is replaced.
    await new Promise(done => setTimeout(done, 20));
}
await new Promise(done => setTimeout(done, 200));
check('every action runs without throwing: ' + failures.join(' | '), failures.length === 0);

// --- a row's position is a number -------------------------------------------------------
//
// A dataset value is always a string, and `copyItem` inserts at `index + 1`:
// with "0" that is "01", which `splice` reads as 1 by luck -- but with "1" it
// is "11", and the copy lands at the end of the list instead of under its
// original. So the position is pressed through the real table, on a row that
// is not the first.
openRotor();
state.projectData.shafts.forEach((shaft, i) => { shaft.tag = 'AB'[i]; });
state.projectData.shafts.push({ element_type: 'BASIC', n: '2', L: '250', odl: '50', tag: 'C' });
runAction(element('BUTTON', 'copy-element', { dataset: { action: 'copy-element', index: '1' } }), event);
check('a row\'s button acts on that row, read as a number',
    state.projectData.shafts.map(s => s.tag).join(',') === 'A,B,B_1,C');
runAction(element('BUTTON', 'delete-element', { dataset: { action: 'delete-element', index: '0' } }), event);
check('and each row button on its own row', state.projectData.shafts.map(s => s.tag).join(',') === 'B,B_1,C');

// A tab button opens the tab it names, and nothing else it carries.
openRotor();
runAction(element('BUTTON', 'pick-tab', { dataset: { action: 'pick-tab', tab: 'disks' } }), event);
check('a tab button opens its own tab', state.currentTab === 'disks');

// --- the analysis cards -----------------------------------------------------------------
//
// The row editors used to be picked through a variable and written into the
// handler, which is why the bridge battery had to list four names by hand. The
// button now names its kind of list, and the table picks the editor.
const containers = { probe_list: 'probe', force_list: 'force', unbalance_list: 'unb',
                     angle_probe_list: 'angle-probe' };
for (const [list, prefix] of Object.entries(containers)) {
    const box = node(prefix + '-container-f1-c1');
    box.innerHTML = '';
    runAction(element('BUTTON', 'add-row', { dataset: { action: 'add-row', list, card: 'c1', field: 'f1', type: 'modes' } }), event);
    check('the add button of a ' + list + ' adds to its own list', /probe-row/.test(box.innerHTML));
}
thrown = null;
try {
    runAction(element('BUTTON', 'add-row', { dataset: { action: 'add-row', list: 'mystery_list', card: 'c1', field: 'f1' } }), event);
} catch (error) { thrown = error; }
check('a kind of list with no editor fails with its name', thrown && /mystery_list/.test(thrown.message));

// A row's remove button removes the row it sits in -- not itself, and not the
// named element the listener found (which is the button).
let removed = null;
const row = { remove() { removed = 'row'; } };
const removeButton = element('BUTTON', 'remove-row');
removeButton.closest = selector => (selector === '.probe-row' ? row : removeButton);
runAction(removeButton, event);
check('a row\'s remove button removes its row', removed === 'row');

// --- the rotor cards of the hub ------------------------------------------------------------

openRotor();
state.rotorLibrary.push(JSON.parse(JSON.stringify(state.rotorLibrary[0])));
state.rotorLibrary[1].name = 'Second';
runAction(element('BUTTON', 'open-rotor', { dataset: { action: 'open-rotor', index: '1', screen: 'screen-modeling' } }), event);
check('a rotor card opens its own rotor', state.projectData === state.rotorLibrary[1]);
// ... on the screen it names: only the modelling screen lights the sidebar.
node('sel:.sidebar').style.opacity = '0';
runAction(element('BUTTON', 'open-rotor', { dataset: { action: 'open-rotor', index: '0', screen: 'screen-modeling' } }), event);
check('and on the screen the card names', node('sel:.sidebar').style.opacity === '1');

// Deleting a card asks first, then removes that card and forgets its analysis.
const { closeCustomConfirm } = await import('../../frontend/components/modals.js');
const { registerAnalysis, analysesToSave } = await import('../../frontend/core/analysis_store.js');
registerAnalysis('c9', 'modes', 'Modes', '');
let cardRemoved = false;
node('card-c9').remove = () => { cardRemoved = true; };
const deleting = runAction(element('BUTTON', 'delete-card', { dataset: { action: 'delete-card', card: 'c9' } }), event);
await new Promise(done => setTimeout(done, 0));
closeCustomConfirm(true);
await deleting;
check('a card\'s delete button removes that card', cardRemoved);
check('and forgets its analysis', !analysesToSave().some(a => a.title === 'Modes'));

// --- the dialogs ----------------------------------------------------------------------------
//
// The answers used to be written into the attribute -- `closeCustomConfirm(true)`
// -- and are names now. What a mix-up would cost is a confirmation read as a
// refusal or the other way round (a rotor deleted on "No"), so each answer is
// pressed and what the waiting code receives is read back.
const { openCustomAlert, openCustomConfirm, openCustomPrompt } =
    await import('../../frontend/components/modals.js');
// A button that answers the wrong dialog leaves this one waiting forever; the
// race turns that into a failed check with a name instead of a battery that
// never ends.
async function answerWith(open, action) {
    const waiting = open();
    runAction(element('BUTTON', action), event);
    return Promise.race([waiting, new Promise(done => setTimeout(() => done('no answer'), 50))]);
}
check('"Yes" confirms', await answerWith(() => openCustomConfirm('?'), 'confirm-yes') === true);
check('"No" refuses', await answerWith(() => openCustomConfirm('?'), 'confirm-no') === false);
node('custom-prompt-input').value = 'typed';
check('the prompt\'s OK gives back what was typed',
    await answerWith(() => openCustomPrompt('?', 'typed'), 'prompt-ok') === 'typed');
check('and its cancel gives back nothing',
    await answerWith(() => openCustomPrompt('?', 'typed'), 'prompt-cancel') === null);
check('the alert\'s OK lets the waiting code go on',
    await answerWith(() => openCustomAlert('!').then(() => 'closed'), 'alert-ok') === 'closed');

// ... and in the page, each answer sits on the button that says it. The
// check above cannot see a "Yes" button carrying `confirm-no`.
function buttonsOf(overlay) {
    const start = PAGE.indexOf('id="' + overlay + '"');
    const box = PAGE.slice(start, PAGE.indexOf('<div id="', start + 1));
    return [...box.matchAll(/<button class="(btn-[a-z-]+)"[^>]*data-action="([a-z-]+)"/g)]
        .map(m => m[1] + ' ' + m[2]);
}
check('the confirm dialog: X and No refuse, Yes confirms',
    buttonsOf('custom-confirm-overlay').join(' | ')
        === 'btn-close-modal confirm-no | btn-confirm confirm-yes | btn-cancel confirm-no');
check('the prompt: X and Cancel give nothing back, OK the text',
    buttonsOf('custom-prompt-overlay').join(' | ')
        === 'btn-close-modal prompt-cancel | btn-confirm prompt-ok | btn-cancel prompt-cancel');
check('the alert: both close it',
    buttonsOf('custom-alert-overlay').join(' | ') === 'btn-close-modal alert-ok | btn-confirm alert-ok');

// The rotor selects of the two hub modals are fields: their sentence is
// rewritten on change. With two different rotors picked, it says something.
openRotor();
state.rotorLibrary.push(Object.assign(JSON.parse(JSON.stringify(state.rotorLibrary[0])), { name: 'Other', uid: 'uid_o' }));
node('mr-driving').value = '0'; node('mr-driven').value = '1'; node('mr-hint').innerHTML = '';
handleEvent({ type: 'change', target: element('SELECT', 'describe-coupling') });
check('changing a MultiRotor select describes the coupling', /Other/.test(node('mr-hint').innerHTML));
node('cc-first').value = '0'; node('cc-second').value = '1'; node('cc-hint').innerHTML = '';
handleEvent({ type: 'change', target: element('SELECT', 'describe-joint') });
check('changing a concatenation select describes the joint', /Other/.test(node('cc-hint').innerHTML));
runAction(element('BUTTON', 'swap-concatenation'), event);
check('and the swap button swaps the two', node('cc-first').value === '1' && node('cc-second').value === '0');
const selects = [...PAGE.matchAll(/<select id="(mr-driving|mr-driven|cc-first|cc-second)" data-action="(describe-[a-z]+)"/g)]
    .map(m => m[1] + ' ' + m[2]);
check('the four selects carry their sentence in the page', selects.join(' | ')
    === 'mr-driving describe-coupling | mr-driven describe-coupling | cc-first describe-joint | cc-second describe-joint');

// --- nothing on the window -------------------------------------------------------------------
//
// The bridge is gone. A name hung on `window` again would be a function the
// page could reach without the table -- the thing slices 12 to 15 undid.
// Comment lines out: main.js tells the story of the bridge in its header.
const MAIN = fs.readFileSync(path.join(FRONTEND, 'main.js'), 'utf8')
    .split('\n').filter(line => !line.trim().startsWith('//')).join('\n');
check('main.js publishes nothing on the window', !MAIN.includes('Object.assign(window'));

// --- the ratchet -------------------------------------------------------------------------
//
// Inline handlers in index.html: none since slice 15, and a new `onclick=`
// in the page fails here. The five `onerror` on the logo images call nothing
// (they hide a missing picture) and are counted apart.
const inline = (PAGE.match(/\son(?!error)[a-z]+="/g) || []).length;
check('inline handlers left in index.html: ' + inline + ' (none)', inline === 0);

// The same count for the HTML the JavaScript writes (the hub's rotor cards and
// the analysis cards, today). Comment lines are left out: a comment that
// quotes an old handler is not a handler -- and the bridge sweep once counted
// one as a live call, which kept `saveRotor` on the bridge for a whole slice.
const generated = sources(FRONTEND)
    .map(text => text.split('\n').filter(line => !line.trim().startsWith('//')).join('\n'))
    .reduce((total, text) => total + (text.match(/\bon[a-z]+=\\?["']/g) || []).length, 0);
// None left since slice 14: from here on it can only stay at zero.
check('inline handlers written by the JavaScript: ' + generated + ' (none)', generated === 0);

shutDown();
