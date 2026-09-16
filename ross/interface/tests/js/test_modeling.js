// The modeling screen: opening a tab, seeing the list, opening the form.
//
// This battery was born from a defect that got past every check of slice 3 and
// reached the user: the add-element button did not respond and no tab showed
// its list. Cause -- the two accessors that moved to `core/state.js` were left
// reading a loose `projectData`, and `openTab` threw a ReferenceError on its
// first line.
//
// Why nothing caught it: the sweep for loose names discarded, from the set of
// used names, every name that appeared as a **key** anywhere in the file -- and
// `projectData:` is a key of the state object, in the same file. And the DOM
// harnesses that existed exercised the analysis screen, which was precisely the
// one that worked.
import { node, looseNode, clearDom, draggable, schemaResponse,
         ELEMENT_SCHEMA_FIXTURE } from './fake_dom.js';

const CATEGORIES = Object.keys(ELEMENT_SCHEMA_FIXTURE.categories);

let requests = [];
globalThis.fetch = async path => {
    requests.push(String(path));
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    return { ok: true, status: 200, json: async () => ({
        status: 'success',
        plot_json: JSON.stringify({ data: [], layout: {} }),
        script: '# script', image: '', message: 'ok',
    }) };
};

import { state } from '../../frontend/core/state.js';
import { schemaReady } from '../../frontend/core/schema.js';
import { renderList } from '../../frontend/components/list.js';
import { changeLanguage } from '../../frontend/features/modeling.js';
import '../../frontend/main.js';

const { openTab, openForm, editItem, selectSubType } = window;

let ok = 0, failed = 0;
// `check(d, error === null || !console.log(error.message))` was the earlier
// idiom, and `!console.log(...)` is **always** true: the check printed the
// error and passed all the same. A test that cannot fail is worse than no test,
// because it takes the place of one.
function withoutError(description, error) {
    if (error) console.log('          ' + error.message);
    check(description, !error);
}

function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}

function sampleProject() {
    return {
        materials: [{ name: 'Steel' }],
        shafts: [{ n: '0', L: '500' }, { n: '1', L: '500' }],
        disks: [{ n: '1', m: '32' }],
        gears: [], couplings: [], seals: [],
        bearings: [{ n: '0' }, { n: '2' }],
        pointmasses: [],
    };
}

function prepare() {
    clearDom();
    state.projectData = sampleProject();
    state.currentTab = null;
    state.editingIndex = -1;
    requests = [];
}

await schemaReady();

// --- opening each tab -----------------------------------------------------------
console.log('\nOpening an element tab');

for (const category of CATEGORIES) {
    prepare();
    let error = null;
    try { openTab(category); } catch (e) { error = e; }
    withoutError('openTab(' + category + ') does not throw', error);
    if (error) continue;
    check('  and the tab is registered', state.currentTab === category);
    const items = node('element-list').children.length;
    check('  the list shows ' + state.projectData[category].length + ' item(s)',
              items === state.projectData[category].length);
}

// --- what the list writes -------------------------------------------------------
console.log('\nWhat the list writes');

prepare();
openTab('shafts');
const text = node('element-list').innerHTML;
check('the shaft shows up with a node number', /SHAFT #1 \(Node 0\)/.test(text));
check('and the second with the next node', /SHAFT #2 \(Node 1\)/.test(text));
check('with the edit, copy and delete buttons',
          /editItem\(0\)/.test(text) && /copyItem\(1\)/.test(text) && /deleteItem\(1\)/.test(text));

prepare();
openTab('materials');
check('a material shows up by name, with no node',
          /MATERIAL #1 - Steel/.test(node('element-list').innerHTML));

// --- the add button -----------------------------------------------------------
console.log('\nThe add-element button');

prepare();
openTab('shafts');
let errorOnOpen = null;
try { await openForm(true); } catch (e) { errorOnOpen = e; }
withoutError('openForm(true) does not throw', errorOnOpen);
check('the form becomes visible', node('insertion-form').style.display === 'block');
check('and starts with no item being edited', state.editingIndex === -1);
// With BASIC in the schema the form offers BASIC or LIST before the fields.
check('the model choice comes first',
          node('form-fields').innerHTML.includes('Select Model')
          && node('form-fields').innerHTML.includes("selectSubType('LIST')"));

selectSubType('BASIC');
check('with the model chosen, the schema fields come',
          node('form-fields').innerHTML.includes('Length'));
check('and the subtype is registered', state.currentSubType === 'BASIC');

// `editItem` calls `openForm` without awaiting; the test awaits, otherwise a
// rejection inside it passes as success -- which is what happened in the first
// version of this battery, hiding a `positionFormBox` that threw.
prepare();
openTab('shafts');
let errorOnEdit = null;
try { editItem(1); await new Promise(ready => setTimeout(ready, 0)); }
catch (e) { errorOnEdit = e; }
withoutError('editing an item does not throw', errorOnEdit);
check('editing an item records which one', state.editingIndex === 1);
check('and the form goes next to the item',
          node('element-list').children.includes(node('insertion-form')));

// --- changing the language ----------------------------------------------------
// This function was broken in another way by the same renaming: the local that
// held the form values was also called `state`, and started shadowing the
// shared one -- including in a read before the declaration, which is always a
// ReferenceError.
console.log('\nChanging the language');

prepare();
openTab('shafts');
// `changeLanguage` is not on the bridge: there is no language selector in the
// HTML yet. The test imports it directly, so that the defect it had (a read
// before the declaration) does not come back silently while it waits for the
// selector.
let languageError = null;
try { await changeLanguage('pt'); } catch (e) { languageError = e; }
withoutError('changeLanguage does not throw with the form closed', languageError);

prepare();
openTab('shafts');
await openForm(true);
selectSubType('BASIC');
languageError = null;
try { await changeLanguage('en'); } catch (e) { languageError = e; }
withoutError('nor with the form open', languageError);
check('and the tab stays the same after the change', state.currentTab === 'shafts');

// --- dragging to reorder ------------------------------------------------------
// The list announces that the order changed; whoever rebuilds the rotor
// subscribes to the hook. In the slice 3 delivery the hook existed and nobody
// subscribed: dragging reordered the list and the figure stayed as it was,
// with no error at all.
console.log('\nDragging an element in the list');

prepare();
state.projectData.shafts = [{ n: '0', L: '100' }, { n: '1', L: '200' }, { n: '2', L: '300' }];
openTab('shafts');
const list = draggable();
check('the list became draggable', !!list && typeof list.options.onEnd === 'function');

requests = [];
let errorOnDrag = null;
try { list.options.onEnd({ oldIndex: 0, newIndex: 2 }); }
catch (e) { errorOnDrag = e; }
withoutError('dragging does not throw', errorOnDrag);
check('the order changed in the project',
          state.projectData.shafts.map(s => s.L).join(',') === '200,300,100');
// `buildRotorLive` waits 600 ms before talking to the server: typing in a
// field must not fire one request per keystroke.
await new Promise(ready => setTimeout(ready, 700));
check('and the rotor was told to rebuild',
          requests.some(p => String(p).includes('/build_rotor')));

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
