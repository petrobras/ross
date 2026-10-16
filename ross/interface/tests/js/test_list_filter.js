// The filter of the element list: by material, model, node range, tag, and a
// condition on a field's value.
//
// WHY THIS BATTERY EXISTS. A filter that is wrong does not look wrong: it
// shows a shorter list, which is what a filter is for. What is checked here is
// each way it could quietly lie --
//
//   * a value compared in the wrong unit. The first version compared 0.3 m
//     with "260" as if both were millimetres, because the condition's unit was
//     left empty until someone picked one (measured in a browser);
//   * a row ticked and then filtered out of sight, still deleted by "delete
//     selected";
//   * a row left out of the list instead of hidden, shifting every position
//     after it -- and the form, the 3D view and the drag all count positions.
import { check, draggable, node, schemaResponse, shutDown } from './fake_dom.js';

// A schema with what the filter reads: the fields' controls, units and
// materials, as the real /api/schema/elements gives them.
const text = (name, unit, options) => ({ name, label: name, control: 'text', unit: unit || null, unit_options: options || [] });
const SCHEMA = {
    language: 'en', unit_alternatives: {},
    categories: {
        materials: { BASIC: { fields: [text('name')] } },
        shafts: { BASIC: { fields: [text('n'), text('L', 'mm', ['mm', 'm', 'in']), { name: 'material', label: 'Material', control: 'material_ref' }, text('tag')] } },
        disks: {
            BASIC: { fields: [text('n'), text('m', 'kg', ['kg', 'g', 'lb']), text('tag')] },
            Geometry: { fields: [text('n'), text('width', 'mm', ['mm', 'm']), { name: 'material', label: 'Material', control: 'material_ref' }, text('tag')] },
        },
        gears: { BASIC: { fields: [text('n')] } }, couplings: { BASIC: { fields: [text('n')] } },
        seals: { BASIC: { fields: [text('n')] } }, pointmasses: { BASIC: { fields: [text('n')] } },
        bearings: { BASIC: { fields: [text('n'), text('kxx', 'N/m', ['N/m'])] } },
    },
};

const asked = [];
globalThis.fetch = async (path, options) => {
    if (String(path).includes('/api/schema/elements')) return { ok: true, status: 200, json: async () => SCHEMA };
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    if (String(path).includes('/api/units/convert')) {
        const { items } = JSON.parse(options.body);
        asked.push(items);
        const factor = { mm: 1e-3, m: 1, in: 0.0254, kg: 1, lb: 0.45359237, g: 1e-3 };
        const values = items.map(({ value, unit, to }) => {
            const v = Number(value);
            if (!Number.isFinite(v)) return null;
            return unit && to ? v * factor[unit] / factor[to] : v;
        });
        return { ok: true, status: 200, json: async () => ({ status: 'success', values }) };
    }
    return { ok: true, status: 200, json: async () => ({ status: 'success', mass: 1, ip: 1, scene: null }) };
};

const F = await import('../../frontend/core/list_filter.js');
const { schemaReady } = await import('../../frontend/core/schema.js');
const { state, listContext } = await import('../../frontend/core/state.js');
const { pick, isPicked, pickAll, picked } = await import('../../frontend/core/selection.js');
const { getTabNodes, renderList } = await import('../../frontend/components/list.js');
const { MODELING_ACTIONS } = await import('../../frontend/features/modeling_actions.js');
const { readNumbers } = await import('../../frontend/features/list_filter.js');
await schemaReady();

const settle = () => new Promise(done => setTimeout(done, 20));
const shafts = [
    { n: '0', L: '250', material: 'Steel' },
    { n: '1', L: '0.3', L_unit: 'm', material: 'Titanium' },
    { n: '2', L: '100' },
    { n: '3', L: '10', L_unit: 'in', material: 'Titanium', tag: 'Rear bearing seat' },
];
const nodes = [0, 1, 2, 3];
const shownOf = () => F.shownIndexes('shafts', shafts, nodes);

// --- the criteria ---------------------------------------------------------------------
console.log('\nWhat each criterion lets through');

F.setCriterion('shafts', 'material', 'Titanium');
check('by material', shownOf().join() === '1,3');
F.setCriterion('shafts', 'material', F.DEFAULT_MATERIAL);
check('a shaft with no material chosen is built with the default steel', shownOf().join() === '2');
F.clearCriteria('shafts');

const disks = [{ n: '1', m: '32' }, { element_type: 'Geometry', n: '2', width: '40' }];
F.setCriterion('disks', 'material', F.DEFAULT_MATERIAL);
check('a form with no material is not "default steel": it has none',
    F.shownIndexes('disks', disks, [1, 2]).join() === '1');
F.clearCriteria('disks');
F.setCriterion('disks', 'model', 'Geometry');
check('by model', F.shownIndexes('disks', disks, [1, 2]).join() === '1');
F.clearCriteria('disks');

F.setCriterion('shafts', 'nodeFrom', '1');
F.setCriterion('shafts', 'nodeTo', '2');
check('by node range, both ends included', shownOf().join() === '1,2');
F.clearCriteria('shafts');
F.setCriterion('shafts', 'text', 'rear');
check('by text in the tag, whatever its case', shownOf().join() === '3');
F.clearCriteria('shafts');

// --- a value, in its units ------------------------------------------------------------
console.log('\nA value, compared in one unit');

F.setCriterion('shafts', 'field', 'L');
F.setCriterion('shafts', 'op', '>');
F.setCriterion('shafts', 'value', '260');
const missing = F.missingNumbers('shafts', shafts);
check('what is typed in the field\'s own unit is compared here, with no server',
    missing.every(m => m.value !== '250' && m.value !== '100'));
check('what is typed in another unit is asked for in the field\'s unit -- not in none',
    missing.length === 2 && missing.every(m => m.to === 'mm') && missing.some(m => m.unit === 'm') && missing.some(m => m.unit === 'in'));
check('until the answer, those rows are shown', F.rowMatches('shafts', shafts[1], 1) === null && shownOf().join() === '1,3');

await readNumbers(missing);
check('asked once, together', asked.length === 1 && asked[0].length === 2);
check('then 0.3 m is 300 mm, and passes "> 260 mm"', shownOf().join() === '1');
F.setCriterion('shafts', 'op', '<');
check('and "< 260 mm": 250 mm, 100 mm, and 10 in (254 mm)', shownOf().join() === '0,2,3');
F.setCriterion('shafts', 'unit', 'm');
F.setCriterion('shafts', 'value', '0.26');
await readNumbers(F.missingNumbers('shafts', shafts));
check('the same, written in metres', shownOf().join() === '0,2,3');
F.setCriterion('shafts', 'op', '=');
F.setCriterion('shafts', 'value', '0.30000000000000004');
check('"=" allows for the rounding of a conversion', shownOf().join() === '1');
F.setCriterion('shafts', 'unit', '');
F.setCriterion('shafts', 'op', '>');
F.setCriterion('shafts', 'value', '250');
check('"> 250" leaves out what is 250', !shownOf().includes(0));
F.setCriterion('shafts', 'op', '>=');
check('and ">= 250" keeps it', shownOf().includes(0));
F.clearCriteria('shafts');

const bearings = [{ n: '0', kxx: '[1e6, 2e6]' }, { n: '1', kxx: '1e6' }];
F.setCriterion('bearings', 'field', 'kxx');
F.setCriterion('bearings', 'op', '>=');
F.setCriterion('bearings', 'value', '1');
await readNumbers(F.missingNumbers('bearings', bearings));
check('a list of values is not one number: left out, not guessed at',
    F.shownIndexes('bearings', bearings, [0, 1]).join() === '1');
check('a field left empty is left out too', F.rowMatches('bearings', { n: '2' }, 2) === false);
F.clearCriteria('bearings');

// --- the badge, the selection ---------------------------------------------------------
console.log('\nThe selection under a filter');

check('no criteria, no badge', F.activeCount(F.criteriaFor('shafts')) === 0);
F.setCriterion('shafts', 'nodeFrom', '1');
F.setCriterion('shafts', 'nodeTo', '3');
F.setCriterion('shafts', 'text', 'x');
check('a node range counts once', F.activeCount(F.criteriaFor('shafts')) === 2);
F.setCriterion('shafts', 'field', 'L');
check('a field with no value to compare is not a criterion yet', F.activeCount(F.criteriaFor('shafts')) === 2);
F.clearCriteria('shafts');

state.projectData = { name: 'R', materials: [{ name: 'Steel' }, { name: 'Titanium' }],
    shafts: shafts.map(s => Object.assign({}, s)), disks: [], gears: [], couplings: [], seals: [], bearings: [], pointmasses: [] };
state.rotorLibrary = [state.projectData];
state.activeRotorIndex = 0;
state.currentTab = 'shafts';
state.editingIndex = -1;

pick(listContext(), 0);
F.setCriterion('shafts', 'material', 'Titanium');
check('a change of filter drops the ticks: they were made on another list', !isPicked(listContext(), 0));
pickAll(listContext(), F.shownIndexes('shafts', state.projectData.shafts, nodes));
check('"select all" ticks the rows shown, and only them', picked(listContext()).join() === '1,3');
F.clearCriteria('shafts');

// --- the list on screen --------------------------------------------------------------
console.log('\nThe rows on screen');

const rows = () => node('element-list').children.slice(-state.projectData.shafts.length);
MODELING_ACTIONS['toggle-list-filter']();
MODELING_ACTIONS['filter-by']({ dataset: { key: 'material' }, value: 'Titanium' });
check('rows out of the filter are hidden, and kept in their places',
    rows().length === 4 && rows().map(r => r.className.includes('is-filtered-out')).join() === 'true,false,true,false');
check('each keeps its own position', /data-index="3"/.test(rows()[3].innerHTML));
check('the bar says how many are shown', /Showing 2 of 4/.test(node('selection-bar').innerHTML));
check('the shafts\' bar offers to discretize them (features/mesh.js)',
    /data-action="mesh-shafts"/.test(node('selection-bar').innerHTML) && typeof MODELING_ACTIONS['mesh-shafts'] === 'function');
check('the panel is open, and says it too', node('list-filter').style.display === 'block' && /Showing 2 of 4/.test(node('list-filter').innerHTML));
check('no dragging through a filter', draggable().options.disabled === true);

MODELING_ACTIONS['filter-by']({ dataset: { key: 'field' }, value: 'L' });
MODELING_ACTIONS['filter-by']({ dataset: { key: 'unit' }, value: 'in' });
MODELING_ACTIONS['filter-by']({ dataset: { key: 'field' }, value: 'L' });
check('choosing a field starts its unit over at the field\'s own', F.criteriaFor('shafts').unit === '');
MODELING_ACTIONS['filter-by']({ dataset: { key: 'value' }, value: '1000' });
await settle();
check('a condition nothing passes hides every row', rows().every(r => r.className.includes('is-filtered-out')));
MODELING_ACTIONS['clear-list-filter']();
check('clearing brings them all back, and dragging with them',
    rows().every(r => !r.className.includes('is-filtered-out')) && draggable().options.disabled === false);

F.setCriterion('disks', 'model', 'Geometry');
check('each category keeps its own filter', F.activeCount(F.criteriaFor('shafts')) === 0 && F.activeCount(F.criteriaFor('disks')) === 1);
F.forgetFilters();
check('and another rotor starts with none', F.activeCount(F.criteriaFor('disks')) === 0);

// --- couplings, numbered as the builder numbers them ------------------------------
console.log('\nCouplings');

// A blank coupling sits on its place in the list (`node_resolver.listed_nodes`),
// not on the lowest free node: after one pinned on 2, the blank one is on 1,
// where the lowest-free rule would have put it on 0.
const couplings = [{ n: '2' }, {}];
check('a blank coupling is on its place in the list', getTabNodes('couplings', couplings).join() === '2,1');
check('every other tab keeps the lowest free node', getTabNodes('disks', couplings).join() === '2,0');
state.projectData.couplings = couplings;
state.currentTab = 'couplings';
F.setCriterion('couplings', 'nodeFrom', '1');
F.setCriterion('couplings', 'nodeTo', '1');
renderList();
const couplingRows = node('element-list').children.slice(-2);
check('the filter by node finds it on the node it is built on',
    couplingRows.map(r => r.className.includes('is-filtered-out')).join() === 'true,false');
F.clearCriteria('couplings');
state.currentTab = 'shafts';

let refused = null;
try { F.setCriterion('shafts', 'colour', 'red'); } catch (error) { refused = error; }
check('a criterion that does not exist is refused by name', refused && /colour/.test(refused.message));

shutDown();
