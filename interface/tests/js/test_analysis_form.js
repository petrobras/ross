// Is the form built from the server catalogue the same as before?
//
// Slice 4 moved 147 fields from the frontend to the backend. A port like that
// is checked by comparing the two outputs, not by reading the diff: the same
// function that builds the panel gets both catalogues and the HTML has to come
// out identical.
//
// The old catalogue is frozen in `tests/golden/analysis_dashboards.json`, taken
// from the real object by evaluating the module -- not by a parser of mine.
import { node, clearDom, schemaResponse } from './fake_dom.js';
import { readFileSync } from 'fs';

globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    return { ok: true, status: 200, json: async () => ({ status: 'success' }) };
};

import { schemaReady, analysisFieldsFor } from '../../frontend/core/schema.js';
import { buildDashboardHTML } from '../../frontend/features/analysis.js';

const OLD = JSON.parse(readFileSync(
    new URL('../golden/analysis_dashboards.json', import.meta.url), 'utf8'));

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}

await schemaReady();

console.log('\nThe catalogue arrived from the server');
check('the twelve analyses came',
          Object.keys(OLD).every(type => (analysisFieldsFor(type) || []).length > 0));
check('and an analysis that does not exist gives back nothing',
          analysisFieldsFor('does_not_exist') === null);

// The only deliberate difference since then is slice 5's `data-deps-de`, which
// says which field each conditional visibility depends on (FE-07). It is taken
// out of the comparison here and enforced just below -- removing it without
// enforcing would let its absence pass as equality.
const withoutDepsSource = html => html.replace(/ data-deps-de="[^"]*"/g, '');

// The other kind of exception, and this one is not free. Where the port had to
// reproduce the old panel, `clearance` is a panel we deliberately rebuilt: a
// single `node` box beside list-valued magnitude and phase could never carry
// more than one value, and the one value it did carry reached ROSS as a
// one-element array where a number was expected -- tolerated by numpy until
// 2.5, an error after it. The three fields became one unbalance table.
//
// Regenerating the frozen file would have been easier and would have destroyed
// the guarantee for the other eleven at the same time, because a golden
// regenerated from the code it guards agrees with anything. So the file stays
// frozen, the exception is named, and the check below fails the day it stops
// being an exception.
// `ucs` joins `clearance` here: the bearing frequency range came back to the
// form as a pair of fields after ROSS fixed the line that refused it.
const CHANGED_ON_PURPOSE = ['clearance', 'ucs'];

// What each one has to show, so that the exception is checked and not merely
// declared. `clearance` traded three fields for an unbalance table; `ucs` got
// the bearing frequency range back as a pair, after ROSS fixed the line that
// refused any value for it.
const REBUILT = {
    clearance: html =>
        html.includes('addUnbalanceRow') && !html.includes('unbalance_magnitude'),
    ucs: html =>
        html.includes('bearing_freq_min') && html.includes('bearing_freq_max'),
};

console.log('\nThe panel comes out the same as before');
for (const type of Object.keys(OLD)) {
    if (CHANGED_ON_PURPOSE.includes(type)) continue;
    clearDom();
    const fromServer = buildDashboardHTML('id1', type);
    clearDom();
    const fromOld = buildDashboardHTML('id1', type, JSON.parse(JSON.stringify(OLD[type])));
    check(type + ' (' + OLD[type].length + ' fields)',
              withoutDepsSource(fromServer) === withoutDepsSource(fromOld));
    if (fromServer !== fromOld) {
        for (let i = 0; i < Math.max(fromServer.length, fromOld.length); i++) {
            if (fromServer[i] !== fromOld[i]) {
                console.log('          differs at ' + i + ':');
                console.log('          server: ' + fromServer.slice(i - 40, i + 60));
                console.log('          old:    ' + fromOld.slice(i - 40, i + 60));
                break;
            }
        }
    }
}

console.log('\nThe rebuilt panel really is rebuilt');
for (const type of CHANGED_ON_PURPOSE) {
    clearDom();
    const fromServer = buildDashboardHTML('id1', type);
    clearDom();
    const fromOld = buildDashboardHTML('id1', type, JSON.parse(JSON.stringify(OLD[type])));
    check(type + ' no longer matches the frozen panel',
              withoutDepsSource(fromServer) !== withoutDepsSource(fromOld));
    // And it differs in the way the exception claims. Without this, any
    // difference at all would satisfy the entry above -- including one nobody
    // meant, which is how a deliberate change becomes cover for an accident.
    check(type + ' differs where its reason says it does', REBUILT[type](fromServer));
}

console.log('\nAnd the deliberate difference is there');
clearDom();
const withDeps = buildDashboardHTML('id1', 'misalignment');
check('the conditional field says what it depends on',
          withDeps.includes('data-deps-de="coupling"'));
check('and it still says which values show it',
          withDeps.includes('data-deps="flex"'));
check('no conditional field was left without its source',
          (withDeps.match(/data-deps=/g) || []).length
          === (withDeps.match(/data-deps-de=/g) || []).length);

// Control: if `buildDashboardHTML` ignored the catalogue and always gave back
// the same thing, the twelve comparisons above would pass.
console.log('\nControl');
clearDom();
const ucs = buildDashboardHTML('id1', 'ucs');
clearDom();
const modes = buildDashboardHTML('id1', 'modes');
check('different analyses give different panels', ucs !== modes);
check('and the panel carries the fields of the analysis', ucs.includes('k_min') && modes.includes('num_modes'));

// The catalogue labels reach the screen -- this is where the translation comes from.
check('the label from the catalogue shows up in the panel',
          ucs.includes(analysisFieldsFor('ucs')[0].label));

// --- FE-07: which field each visibility depends on ---------------------------
//
// `checkDeps` read **every** select on the card and showed the field if any of
// them held one of the allowed values. It worked because no dependency value
// appeared in two selects of the same analysis -- a measured coincidence, not a
// guarantee. Now it reads the field the catalogue names.
import { registerSelector } from './fake_dom.js';
import { checkDeps } from '../../frontend/features/analysis.js';

console.log('\nConditional visibility');

clearDom();
const soFlex = node('field-flex');
soFlex.dataset = { deps: 'flex', depsDe: 'coupling' };
const soRigid = node('field-rigid');
soRigid.dataset = { deps: 'rigid', depsDe: 'coupling' };
registerSelector('.dash-dep-x1', [soFlex, soRigid]);

node('input-coupling-x1').value = 'flex';
checkDeps('x1');
check('the field of the flexible coupling shows up', soFlex.style.display === 'flex');
check('and the rigid one disappears', soRigid.style.display === 'none');

node('input-coupling-x1').value = 'rigid';
checkDeps('x1');
check('changing the coupling changes which one shows',
          soFlex.style.display === 'none' && soRigid.style.display === 'flex');

// The FE-07 control: another select on the same card carrying the value of the
// other branch. In the old code this showed both fields at once.
node('input-coupling-x1').value = 'flex';
node('input-plot_type-x1').value = 'rigid';
checkDeps('x1');
check('another select with the same value does not interfere',
          soFlex.style.display === 'flex' && soRigid.style.display === 'none');

// And a `deps_de` for an absent field hides, instead of showing by mistake.
const orphan = node('field-orphan');
orphan.dataset = { deps: 'flex', depsDe: 'field_that_does_not_exist' };
registerSelector('.dash-dep-x2', [orphan]);
checkDeps('x2');
check('a dependency on a field that does not exist hides it', orphan.style.display === 'none');

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
