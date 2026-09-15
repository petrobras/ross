// What goes into the exported Python script, and under which rotor model.
//
// Two defects lived here. Exporting from the Hub sent an empty analysis list
// -- the remains of a decision that made sense when the only source of
// analyses was the DOM of the open project. And the degree-of-freedom
// conversion came from the screen's selector, the same value for every card,
// including the ones computed under another conversion.
import { node, clearDom, schemaResponse } from './fake_dom.js';

let requests = [];
let downloaded = [];
globalThis.fetch = async (path, options) => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    requests.push({ path, body: JSON.parse(options.body) });
    return { ok: true, status: 200, json: async () => ({ status: 'success', script: '# script' }) };
};
URL.createObjectURL = blob => { downloaded.push(blob); return 'blob:test'; };
URL.revokeObjectURL = () => {};

import {
    recordResult, forgetAllAnalyses, registerAnalysis,
} from '../../frontend/core/analysis_store.js';
import { state } from '../../frontend/core/state.js';
import '../../frontend/main.js';

const { generatePythonFile, generatePythonFromHub, closeCustomAlert } = window;

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}
function clearAll() {
    clearDom(); forgetAllAnalyses();
    requests = []; downloaded = [];
    state.rotorLibrary = []; state.projectData = { shafts: [] };
}

// The mixed-conversions warning is the interface's own modal: it only resolves
// when someone closes it. Whoever exports waits for it, so the test closes it
// -- exercising the real modal instead of a stand-in for it.
async function exportIt(call) {
    const promise = call();
    await Promise.resolve();
    closeCustomAlert();
    return promise;
}

const ANALYSIS = params => ({ type: 'modes', params, conversion: '' });

// --- the Hub exports the analyses of its own rotor --------------------------
console.log('\nExport from the Hub');

clearAll();
state.rotorLibrary = [{
    name: 'A', shafts: [{ L: '0.5' }],
    savedAnalyses: [{ title: 'Modal', type: 'modes', params: { num_modes: '4' }, conversion: '' }],
}];
await exportIt(() => generatePythonFromHub(0));

check('the Hub called the export route',
          requests.length === 1 && requests[0].path.includes('/api/export/python'));
check('and sent the analysis of that rotor', requests[0].body.analyses.length === 1);
check('with the right type', requests[0].body.analyses[0].type === 'modes');
check('and with the parameters', requests[0].body.analyses[0].params.num_modes === '4');
check('the file was delivered', downloaded.length === 1);

clearAll();
state.rotorLibrary = [{ name: 'B', shafts: [], savedAnalyses: [] }];
await exportIt(() => generatePythonFromHub(0));
check('a rotor with no analysis still exports the rotor',
          requests.length === 1 && requests[0].body.analyses.length === 0);

// An analysis that never computed has no parameters; it must not go into the
// script as an empty block.
clearAll();
state.rotorLibrary = [{ name: 'C', shafts: [], savedAnalyses: [
    { title: 'Modal', type: 'modes', params: {}, conversion: '' },
    { title: 'UCS', type: 'ucs', params: { num_modes: '2' }, conversion: '' },
] }];
await exportIt(() => generatePythonFromHub(0));
check('an analysis with no parameters stays out',
          requests[0].body.analyses.length === 1 && requests[0].body.analyses[0].type === 'ucs');

// --- the conversion comes from the analyses, not from the screen ------------
console.log('\nWhere the conversion of the script comes from');

clearAll();
node('rotor-conversion-type').value = 'torsional';       // the screen says one thing...
state.rotorLibrary = [{ name: 'D', shafts: [], savedAnalyses: [
    { title: 'Modal', type: 'modes', params: { n: '1' }, conversion: '4dof' },
] }];
await exportIt(() => generatePythonFromHub(0));
check('the conversion comes from the analysis, not from the selector',
          requests[0].body.conversion_type === '4dof');

clearAll();
state.rotorLibrary = [{ name: 'E', shafts: [], savedAnalyses: [
    { title: 'Modal', type: 'modes', params: { n: '1' }, conversion: '4dof' },
    { title: 'UCS', type: 'ucs', params: { n: '1' }, conversion: 'torsional' },
] }];
await exportIt(() => generatePythonFromHub(0));
check('mixed conversions generate no script', requests.length === 0);
check('and nothing is downloaded', downloaded.length === 0);

// --- export from the analysis screen ----------------------------------------
console.log('\nExport from the analysis screen');

clearAll();
state.projectData = { name: 'open', shafts: [{ L: '0.5' }] };
registerAnalysis('a1', 'modes', 'Modal', '');
recordResult('a1', { num_modes: '4' }, '', null);   // restored: with no figure
await exportIt(() => generatePythonFile());

check('the restored analysis, with no chart, goes into the script',
          requests.length === 1 && requests[0].body.analyses.length === 1);
check('with the parameters it keeps',
          requests[0].body.analyses[0].params.num_modes === '4');

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
