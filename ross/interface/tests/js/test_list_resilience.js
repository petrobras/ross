// The element list, when something goes wrong -- and the form, when the list
// moves under it.
//
// WHY THIS BATTERY EXISTS. Leonardo reported that the element lists vanished
// during a modelling session, and nobody could say why. The mechanism was in
// the code: `renderList` emptied the list *before* reading the project, so any
// exception in between left an empty panel, and with no error handler anywhere
// nothing said what had happened. The trigger was never reproduced. So what is
// checked here is the class, not the trigger: whatever throws, the list that was
// on screen stays there, and the failure is shown.
//
// And the same slice closed the other way an edit could land in the wrong
// place: the form remembers its element by position, and copying or dragging
// moved positions without telling it.
import { check, draggable, node, shutDown } from './fake_dom.js';

// Node's global is not an EventTarget; the notice listens on `window`.
const windowListeners = {};
globalThis.addEventListener = (type, handler) => {
    (windowListeners[type] = windowListeners[type] || []).push(handler);
};
const fire = (type, event) => (windowListeners[type] || []).forEach(handler => handler(event));
globalThis.dispatchEvent = event => fire(event.type, event);

globalThis.fetch = async () => ({
    ok: true, status: 200,
    json: async () => ({ status: 'error', message: 'could not read <b>abc</b>' }),
});

const { afterInsertion, afterMove, afterRemoval } = await import('../../frontend/core/editing.js');
const { onReorder, renderList } = await import('../../frontend/components/list.js');
const { buildRotorLive, copyItem, deleteItem } = await import('../../frontend/features/modeling.js');
const { openProjectHistory, state } = await import('../../frontend/core/state.js');
const { describeError, isNoise, startErrorNotice } = await import('../../frontend/features/error_notice.js');
// Wired like main.js wires it.
onReorder(buildRotorLive);

function openRotor(tags) {
    state.rotorLibrary = [{
        name: 'Compressor', uid: 'uid_c', savedAnalyses: [], materials: [],
        shafts: tags.map(tag => ({ element_type: 'BASIC', n: '', L: '250', odl: '50', tag })),
        disks: [], gears: [], couplings: [], seals: [], bearings: [], pointmasses: [],
    }];
    state.activeRotorIndex = 0;
    state.projectData = state.rotorLibrary[0];
    state.currentTab = 'shafts';
    state.editingIndex = -1;
    openProjectHistory(state.projectData);
}

const list = node('element-list');
const form = node('insertion-form');

// --- a failure leaves the list where it was ---------------------------------------

openRotor(['A', 'B', 'C']);
renderList();
const before = list.innerHTML;
check('setting the scene: three rows on screen', (before.match(/list-item|SHAFT #/g) || []).length >= 3);

// A `null` in a category -- a hand-edited file, a half-applied change. Reading
// `null.element_type` throws in the middle of building the rows.
state.projectData.shafts[1] = null;
let thrown = null;
try { renderList(); } catch (error) { thrown = error; }
check('building the rows still fails, and says so', thrown !== null);
check('but the list that was on screen is still there', list.innerHTML === before);

// A category that is not in the project at all.
state.projectData.shafts = ['A', 'B', 'C'].map(tag => ({ element_type: 'BASIC', L: '250', tag }));
delete state.projectData.gears;
state.currentTab = 'gears';
thrown = null;
try { renderList(); } catch (error) { thrown = error; }
check('a missing category fails too', thrown !== null);
check('and leaves the previous list on screen as well', list.innerHTML === before);
state.projectData.gears = [];
state.currentTab = 'shafts';

// The form is part of what survives. It lives inside the list while an element
// is being edited; the old code moved it out before failing, and it stayed out.
openRotor(['A', 'B', 'C']);
renderList();
list.appendChild(form);
state.editingIndex = 1;
state.projectData.shafts[2] = null;
try { renderList(); } catch (error) { /* shown by the notice */ }
// `parentElement`, not `contains`: the fake DOM's `appendChild` does not take
// a node out of its old parent, so `contains` stayed true after the form had
// left -- and the first version of this check passed with the defect in place.
check('an open form stays inside the list when rendering fails', form.parentElement === list);

// Control: rendering that succeeds still replaces the list.
openRotor(['X', 'Y']);
renderList();
check('a render that succeeds does replace the list', list.innerHTML !== before);

// --- the form follows its element ----------------------------------------------------

check('an insertion above moves the edited element down', afterInsertion(2, 1) === 3);
check('an insertion right on it moves it too', afterInsertion(2, 2) === 3);
check('an insertion below leaves it', afterInsertion(2, 3) === 2);
check('with no form open nothing moves', afterInsertion(-1, 0) === -1);
check('removing the edited element leaves nothing to point at', afterRemoval(2, 2) === null);
check('removing one above moves it up', afterRemoval(2, 0) === 1);
check('dragging the edited element takes it along', afterMove(1, 1, 3) === 3);
check('dragging one from above to below it moves it up', afterMove(2, 0, 3) === 1);
check('dragging one from below to above it moves it down', afterMove(1, 3, 0) === 2);
check('dragging entirely below it leaves it', afterMove(0, 2, 3) === 0);
check('dragging one from above onto its place moves it up', afterMove(3, 0, 3) === 2);

// The bug, through the screen. Editing C (position 2), copying A (position 0):
// A's copy goes in at 1 and C moves to 3. The form used to stay at 2 -- on B --
// and "Save" wrote C's form over B.
openRotor(['A', 'B', 'C']);
renderList();
list.appendChild(form);
state.editingIndex = 2;
copyItem(0);
check('copying above the edited element keeps the form on it',
    state.projectData.shafts[state.editingIndex].tag === 'C');
check('and the form is put back into the list, under its element', form.parentElement === list);

copyItem(state.editingIndex);
check('copying the edited element itself keeps the form on the original',
    state.projectData.shafts[state.editingIndex].tag === 'C');

// The rule was already right in `deleteItem`; it now goes through the same
// helper, and this says it still is.
openRotor(['A', 'B', 'C']);
renderList();
state.editingIndex = 2;
deleteItem(0);
check('deleting above the edited element keeps the form on it',
    state.projectData.shafts[state.editingIndex].tag === 'C');

// Dragging, which did not follow at all.
openRotor(['A', 'B', 'C', 'D']);
renderList();
state.editingIndex = 1;                                  // B
draggable().options.onEnd({ oldDraggableIndex: 3, newDraggableIndex: 0 }); // D to the top
check('dragging another element above keeps the form on its own',
    state.projectData.shafts[state.editingIndex].tag === 'B');
draggable().options.onEnd({ oldDraggableIndex: state.editingIndex, newDraggableIndex: 3 });
check('dragging the edited element takes the form with it',
    state.projectData.shafts[state.editingIndex].tag === 'B');

// What Leonardo hit, in the browser: with the form open, Sortable counted it as
// a row, so the indices it reported were one too high. Dragging the last of
// four shafts to the top asked for element 4 of a four-element list, and
// `splice` answered by inserting `undefined` -- saved as `null`, and no render
// of that list succeeded afterwards.
check('only rows can be dragged, so the form is never counted as one',
    draggable().options.draggable === '.list-item');
openRotor(['A', 'B', 'C', 'D']);
renderList();
let refused = null;
try { draggable().options.onEnd({ oldDraggableIndex: 4, newDraggableIndex: 0 }); } catch (error) { refused = error; }
check('a position outside the list is refused, loudly', refused !== null);
check('and nothing was inserted or lost',
    state.projectData.shafts.map(s => s && s.tag).join(',') === 'A,B,C,D');

// --- leaving with the form open ------------------------------------------------------------
//
// The other half of what Leonardo saw, and very likely the original "lists
// vanished": `openRotorWorkspace` emptied the list with the form still inside
// it, which deletes the form from the page. Every later `closeForm` threw, and
// `openTab` stopped before drawing the list.
const { openRotorWorkspace } = await import('../../frontend/features/hub.js');
openRotor(['A', 'B']);
renderList();
list.appendChild(form);
state.editingIndex = 1;
openRotorWorkspace(0, 'screen-analysis');
check('opening a rotor takes the form out of the list before emptying it',
    form.parentElement === node('list-area'));
check('and closes it', state.editingIndex === -1);

// --- the notice -------------------------------------------------------------------------

check('the message says where', describeError(new Error('boom'), 'http://127.0.0.1:5001/components/list.js?v=3', 173)
    === 'boom (list.js:173)');
check('and still says something without a place', describeError('just text') === 'just text');
check('a cancelled request is not an error', isNoise({ name: 'AbortError' }) === true);
check('neither is the ResizeObserver warning',
    isNoise(new Error('ResizeObserver loop completed with undelivered notifications.')) === true);
check('a real error is not noise', isNoise(new TypeError('x is null')) === false);

let closeHandler = null;
node('error-notice-close').addEventListener = (type, handler) => { closeHandler = handler; };
const notice = node('error-notice');
notice.style.display = 'none';
startErrorNotice();

fire('error', { error: new AbortError(), filename: 'api.js', lineno: 1 });
check('a cancelled request shows nothing', notice.style.display === 'none');

fire('error', { error: new TypeError("Cannot read properties of null (reading 'element_type')"), filename: 'list.js', lineno: 139 });
check('an error in a handler is shown', notice.style.display === 'flex');
check('with its message and its place', /element_type.*list\.js:139/.test(node('error-notice-text').textContent));

fire('unhandledrejection', { reason: new Error('saved project is not a list') });
check('a rejected promise is shown too', /not a list/.test(node('error-notice-text').textContent));
check('and the notice counts them', /2 errors so far/.test(node('error-notice-text').textContent));

closeHandler();
check('closing hides it', notice.style.display === 'none');
fire('error', { error: new Error('again') });
check('and the count starts over', !/errors so far/.test(node('error-notice-text').textContent));

// --- the modelling error is text, not markup ----------------------------------------------

openRotor(['A']);
buildRotorLive();
await new Promise(done => setTimeout(done, 700));
const figure = node('plot-rotor').innerHTML;
check('the server\'s message is shown', /could not read/.test(figure));
check('with what it quotes escaped, not drawn', figure.includes('&lt;b&gt;abc&lt;/b&gt;') && !figure.includes('<b>abc</b>'));

// --- the hub, while here ----------------------------------------------------------------

const { renderRotorHub } = await import('../../frontend/features/hub.js');
state.rotorLibrary = [{
    name: 'Gearbox', uid: 'g', savedAnalyses: [], materials: [{}, {}],
    shafts: [{}, {}], disks: [{}], gears: [{}], couplings: [{}], seals: [{}], bearings: [{}, {}], pointmasses: [{}],
}];
renderRotorHub();
// It used to add up shafts, disks and bearings only: this rotor was "5 elements".
check('the hub counts every element of every kind', /\(9 elements\)/.test(node('rotor-hub-list').innerHTML));

const { readFileSync } = await import('fs');
const page = readFileSync(new URL('../../frontend/index.html', import.meta.url), 'utf8');
check('the hub\'s load button carries its translation key',
    /data-action="choose-file" data-input="upload-rotor-hub"><i class="fas fa-upload"><\/i> <span data-i18n="homeLoadRotor">/.test(page));

shutDown();

function AbortError() { this.name = 'AbortError'; this.message = 'aborted'; }
