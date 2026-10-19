// The probes of the model, offered by the analyses' probe tables
// (core/probes.js, features/analysis.js), and the ids they are named by.
//
// WHY THIS BATTERY EXISTS. A row that names a probe of the model is read from
// the model on every run (domain/probe_refs.py), by the probe's id. So the
// claims are about the id and about what the row says: every probe has an id
// and keeps it, a copy is another probe, the row offers the probes the model
// has now and keeps the one it named, and a probe the model lost is shown as
// lost rather than quietly read at node 0.
import { readFileSync } from 'fs';
import { check, schemaResponse, shutDown } from './fake_dom.js';

globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    return { ok: true, status: 200, json: async () => ({ status: 'success' }) };
};

const { newProbeId, probeChoices, withProbeIds } = await import('../../frontend/core/probes.js');
const { withEveryCategory } = await import('../../frontend/core/project_file.js');
const { state } = await import('../../frontend/core/state.js');
const { schemaReady, analysisFieldsFor } = await import('../../frontend/core/schema.js');
const { buildDashboardHTML } = await import('../../frontend/features/analysis.js');
const { freshCopy } = await import('../../frontend/features/modeling.js');

// --- the ids ------------------------------------------------------------------------
console.log('\nThe ids');

const a = newProbeId();
const b = newProbeId();
check('a new id each time', a !== b && /^probe-/.test(a));

const rotor = {
    probes: [
        { n: '2', tag: 'DE X', direction: 'radial', angle: '45' },
        { n: '4', tag: '', direction: 'axial', angle: '' },
        { n: '1', tag: 'NDE Y', direction: 'radial', angle: '1.5', angle_unit: 'rad', id: 'kept' },
    ],
};
withProbeIds(rotor);
check('every probe gets one', rotor.probes.every(p => typeof p.id === 'string' && p.id.length > 0));
check('one it already had is kept', rotor.probes[2].id === 'kept');
const first = rotor.probes[0].id;
withProbeIds(rotor);
check('and given once: asking again changes nothing', rotor.probes[0].id === first);

const train = { isMultiRotor: true, driving_rotor: { probes: [{ n: '1', angle: '0' }] }, driven_rotor: {} };
withEveryCategory(train);
check('a MultiRotor loaded from a file: the probes of each half get one',
    !!train.driving_rotor.probes[0].id && Array.isArray(train.driven_rotor.probes));

const copy = freshCopy(rotor.probes[0], rotor.probes);
check('a copy is another probe, with its own id', copy.id && copy.id !== first && copy.n === '2');

// --- what the analyses offer -----------------------------------------------------------
console.log('\nWhat the analyses offer');

const choices = probeChoices(rotor);
check('one choice per probe, in the order of the list', choices.map(c => c.id).join() === rotor.probes.map(p => p.id).join());
check('named by its name, its node and its angle in degrees', choices[0].label === 'DE X · node 2 · 45°');
check('an axial one says so, and an unnamed one is named by its place', choices[1].label === 'Probe #2 · node 4 · axial');
check('an angle typed in radians says radians', choices[2].label === 'NDE Y · node 1 · 1.5 rad');
const halves = probeChoices({
    isMultiRotor: true,
    driving_rotor: { probes: [{ n: '3', tag: 'A', angle: '0', id: 'one' }] },
    driven_rotor: { probes: [{ n: '1', tag: 'B', direction: 'axial', id: 'two' }] },
});
check('on a MultiRotor, the line it is on', halves.map(c => c.label).join(' | ') === 'A · Driving Rotor · node 3 · 0° | B · Driven Rotor · node 1 · axial');

await schemaReady();
const FORMS = JSON.parse(readFileSync(new URL('../golden/analysis_dashboards.json', import.meta.url), 'utf8'));
const withProbes = Object.keys(FORMS).find(type => (analysisFieldsFor(type) || []).some(f => f.type === 'angle_probe_list'));
const fields = () => JSON.parse(JSON.stringify(analysisFieldsFor(withProbes)));
const card = rows => {
    const config = fields();
    config.forEach(item => { if (item.type === 'angle_probe_list') item.val = rows; });
    return buildDashboardHTML('card', withProbes, config);
};

state.projectData = { probes: [] };
const plain = card([{ node: 3, angle: 0.5 }]);
check('with no probes in the model, the row is typed, as it always was', !/probe-ref/.test(plain) && /value="3"/.test(plain));

state.projectData = rotor;
const offered = card([{ node: 3, angle: 0.5 }]);
check('with probes, a row offers them', /class="probe-ref"/.test(offered)
    && rotor.probes.every(p => offered.includes(`value="${p.id}"`)));
const picker = html => (html.match(/<select class="probe-ref"[\s\S]*?<\/select>/) || [''])[0];
check('a typed row keeps its node and angle, and nothing chosen', /<option value="">Typed: node and angle<\/option>/.test(offered)
    && !/ selected/.test(picker(offered)) && /value="3"/.test(offered) && !/probe-typed" hidden/.test(offered));

const named = card([{ node: 0, angle: 0, probe: first }]);
check('a row that named a probe shows it chosen', named.includes(`value="${first}" selected`));
check('and its typed node and angle put away', /class="probe-typed" hidden/.test(named));

const lost = card([{ node: 0, angle: 0, probe: 'gone' }]);
check('a probe the model no longer has is shown as missing, still chosen',
    /value="gone" selected>A probe no longer in the model</.test(lost));
state.projectData = { probes: [] };
check('even with no probe left in the model at all', /value="gone" selected/.test(card([{ node: 0, angle: 0, probe: 'gone' }])));

// --- the outputs of the frequency response -----------------------------------------
console.log('\nThe outputs of the frequency response');

// An output is where the machine has its probes; an input is a force on a
// degree of freedom, not a measurement, and stays typed.
state.projectData = rotor;
const frf = fields_ => buildDashboardHTML('frf', 'freq_response', fields_);
const frfConfig = JSON.parse(JSON.stringify(analysisFieldsFor('freq_response')));
frfConfig.forEach(item => {
    if (item.id === 'inps') item.val = [{ node: 1, dof: 0 }];
    if (item.id === 'outs') item.val = [{ node: 3, dof: 1 }, { node: 0, dof: 0, probe: first }];
});
const frfHtml = frf(frfConfig);
const inputsBlock = (frfHtml.match(/id="probe-container-inps-frf"[\s\S]*?id="probe-container-outs-frf"/) || [''])[0];
const outputsBlock = frfHtml.slice(frfHtml.indexOf('id="probe-container-outs-frf"'));
check('the outputs offer the probes of the model', (outputsBlock.match(/class="probe-ref"/g) || []).length === 2);
check('an output that named one shows it chosen, its node and dof put away',
    outputsBlock.includes(`value="${first}" selected`) && /class="probe-typed" hidden/.test(outputsBlock));
check('a typed output keeps its node and its degree of freedom', /value="3"/.test(outputsBlock) && /value="1" selected>y/.test(outputsBlock));
check('the inputs do not: a force is not a probe', inputsBlock.length > 0 && !/probe-ref/.test(inputsBlock));

shutDown();
