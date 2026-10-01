// A material's name, the way ROSS takes it -- and the elements that use it.
//
// `rs.Material` refuses a space in the name, and the form let a person type
// "Stainless Steel": the rotor failed to build. Now the name is written with
// underscores when the material is saved (core/material_names.js), the server
// applies the same rule to older projects (domain/material_names.py), and both
// are held to tests/golden/material_names.json.
//
// Two things came along because they live on the same field: an edited name
// takes the shafts that used it along (before, they kept naming a material
// that no longer existed, and the server answers that with the first material
// of the list), and the names are escaped where the shaft form lists them.
import { check, node, schemaResponse, shutDown } from './fake_dom.js';
import { readFileSync } from 'fs';

globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    return { ok: true, status: 200, json: async () => schema || {
        status: 'success', plot_json: JSON.stringify({ data: [], layout: {} }) } };
};
globalThis.dispatchEvent = () => true;

const { elementsUsing, materialKey, renameMaterial, rossMaterialName } =
    await import('../../frontend/core/material_names.js');
const { state, openProjectHistory } = await import('../../frontend/core/state.js');
const { schemaReady } = await import('../../frontend/core/schema.js');
await import('../../frontend/main.js');
const { saveItem, selectSubType } = await import('../../frontend/features/modeling.js');
const { renderRotorHub } = await import('../../frontend/features/hub.js');
await schemaReady();

// --- the rule, on the cases the server is held to ---------------------------------

const { cases } = JSON.parse(readFileSync(new URL('../golden/material_names.json', import.meta.url), 'utf8'));
check('the golden file has cases', cases.length > 5);
const wrong = cases.filter(([typed, expected]) => rossMaterialName(typed) !== expected);
check('the screen writes every name the way the server does: '
    + wrong.map(c => JSON.stringify(c[0])).join(', '), wrong.length === 0);
check('and matches without case', materialKey('Stainless Steel') === materialKey('stainless_STEEL'));

// --- saving from the form ---------------------------------------------------------------

function openRotor(materials, shafts) {
    const rotor = { name: 'R', uid: 'u', savedAnalyses: [], materials, shafts,
                    disks: [], gears: [], couplings: [], seals: [], bearings: [], pointmasses: [] };
    state.rotorLibrary = [rotor];
    state.activeRotorIndex = 0;
    state.projectData = rotor;
    state.currentTab = 'materials';
    state.currentSubType = 'BASIC';
    state.editingIndex = -1;
    openProjectHistory(rotor);
    return rotor;
}
// What the form holds: the fake DOM does not read selectors, so the form's
// fields are declared.
function typeIntoForm(fields) {
    const inputs = Object.entries(fields).map(([key, value]) => ({ id: 'inp-' + key, value, tagName: 'INPUT' }));
    node('form-fields').querySelectorAll = () => inputs;
}

let rotor = openRotor([], []);
typeIntoForm({ name: 'Stainless Steel', rho: '8000' });
saveItem();
check('a new material typed with a space is saved the way ROSS takes it',
    rotor.materials.length === 1 && rotor.materials[0].name === 'Stainless_Steel');

rotor = openRotor([], []);
state.currentSubType = 'LIST';
typeIntoForm({ name: 'Aço Inox, Low Carbon', rho: '8000, 7850' });
saveItem();
check('and so is each one of a list',
    rotor.materials.map(m => m.name).join('|') === 'Aço_Inox|Low_Carbon');

// Editing: the shafts that named the old name follow it -- whichever spelling
// they used -- and a shaft of another material stays where it was.
rotor = openRotor([{ element_type: 'BASIC', name: 'Old Alloy' }, { element_type: 'BASIC', name: 'Other' }],
    [{ material: 'Old Alloy' }, { material: 'old_alloy' }, { material: 'Other' }, { L: '1' }]);
rotor.gears = [{ material: 'Old Alloy' }];
state.editingIndex = 0;
typeIntoForm({ name: 'New Alloy' });
saveItem();
check('an edited material takes its new name', rotor.materials[0].name === 'New_Alloy');
check('the shafts that used it follow it, in either spelling',
    rotor.shafts.map(s => s.material).join('|') === 'New_Alloy|New_Alloy|Other|');
check('and so do the gears', rotor.gears[0].material === 'New_Alloy');

// Saving an old material again only underscores it; its shafts follow.
rotor = openRotor([{ element_type: 'BASIC', name: 'Stainless Steel' }], [{ material: 'Stainless Steel' }]);
state.editingIndex = 0;
typeIntoForm({ name: 'Stainless Steel' });
saveItem();
check('a material saved before the rule is underscored when it is saved again, with its shafts',
    rotor.materials[0].name === 'Stainless_Steel' && rotor.shafts[0].material === 'Stainless_Steel');

// The rename on its own.
const project = { materials: [{ name: 'A' }], shafts: [{ material: 'A' }, { material: 'B' }], savedAnalyses: [{}] };
check('renaming says how many elements moved', renameMaterial(project, 'A', 'C') === 1);
check('and a name that did not change moves nothing', renameMaterial(project, 'C', 'C') === 0);
check('and neither does a material that had no name', renameMaterial(project, undefined, 'D') === 0);
check('counting the elements of a material walks every category',
    elementsUsing({ materials: [{ name: 'A' }], shafts: [{ material: 'A' }, { material: 'a' }],
                    gears: [{ material: 'A' }], disks: [{ material: 'B' }] }, 'A') === 3);
check('and a material nothing is made of counts zero',
    elementsUsing({ materials: [{ name: 'A' }], shafts: [{ material: 'B' }] }, 'A') === 0);

// --- deleting a material that elements still use ---------------------------------------------
//
// The server refuses to build a rotor whose shaft names a material it does not
// have (domain/material_names.py). That refusal arrives later and names
// elements the person is no longer looking at, so the screen asks first -- and
// says how many elements are made of it.
const { closeCustomConfirm } = await import('../../frontend/components/modals.js');
const { deleteItem, deleteSelected } = await import('../../frontend/features/modeling.js');
const { pick } = await import('../../frontend/core/selection.js');
const { listContext } = await import('../../frontend/core/state.js');

function asking() {
    return node('custom-confirm-overlay').style.display === 'flex';
}
async function answer(deleting, yes) {
    await new Promise(done => setTimeout(done, 0));
    closeCustomConfirm(yes);
    await deleting;
}

rotor = openRotor([{ name: 'Used' }, { name: 'Spare' }], [{ material: 'Used' }, { material: 'used' }]);
node('custom-confirm-message').innerText = '';
let deleting = deleteItem(0);
await new Promise(done => setTimeout(done, 0));
check('deleting a material in use asks first', asking());
check('and the question counts the elements', /2/.test(node('custom-confirm-message').innerText));
await answer(deleting, false);
check('answered no, the material stays', rotor.materials.map(m => m.name).join('|') === 'Used|Spare');

deleting = deleteItem(0);
await answer(deleting, true);
check('answered yes, it goes', rotor.materials.map(m => m.name).join('|') === 'Spare');

// One element, one sentence -- and the unused material goes with no question.
rotor = openRotor([{ name: 'Used' }, { name: 'Spare' }], [{ material: 'Used' }]);
deleting = deleteItem(0);
await new Promise(done => setTimeout(done, 0));
check('one element asks in the singular', /1 element\b/.test(node('custom-confirm-message').innerText));
await answer(deleting, false);

node('custom-confirm-overlay').style.display = 'none';
rotor = openRotor([{ name: 'Used' }, { name: 'Spare' }], [{ material: 'Used' }]);
await deleteItem(1);
check('an unused material goes with no question',
    !asking() && rotor.materials.map(m => m.name).join('|') === 'Used');

// The same for the bulk delete, which is a different path.
rotor = openRotor([{ name: 'Used' }, { name: 'Spare' }], [{ material: 'Used' }, { material: 'Used' }]);
pick(listContext(), 0);
pick(listContext(), 1);
deleting = deleteSelected();
await new Promise(done => setTimeout(done, 0));
check('deleting several materials asks once, for all of them', asking());
await answer(deleting, false);
check('and "no" keeps every one of them', rotor.materials.length === 2);

// --- the names in the shaft form are escaped ------------------------------------------------
//
// A material's name is whatever was typed into it; written raw into an
// `<option>`, a quote closed the attribute and the rest became markup.
rotor = openRotor([{ element_type: 'BASIC', name: 'x"><b>bold</b>' }], []);
state.currentTab = 'shafts';
const materialSelect = { innerHTML: '' };
node('form-fields').querySelectorAll = selector => (selector === 'select#inp-material' ? [materialSelect] : []);
selectSubType('BASIC');
check('a material name is escaped in the shaft form',
    materialSelect.innerHTML.includes('x&quot;&gt;&lt;b&gt;bold&lt;/b&gt;')
    && !materialSelect.innerHTML.includes('<b>'));

// --- the hub counts in the singular too ----------------------------------------------------

function hubBadge(shafts) {
    openRotor([{ name: 'Steel' }], Array.from({ length: shafts }, () => ({ L: '1' })));
    node('rotor-hub-list').innerHTML = '';
    renderRotorHub();
    return node('rotor-hub-list').innerHTML;
}
check('one element is "1 element"', /\(1 element\)/.test(hubBadge(1)));
check('two are "2 elements"', /\(2 elements\)/.test(hubBadge(2)));

shutDown();
