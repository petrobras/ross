// Calls every function on the bridge once and sees whether any throws on its
// first line.
//
// It does not replace behaviour tests: it does not check what each one does. It
// covers something else -- the class of defect slice 3 produced that got past
// every check and reached the user: a name the modularisation left behind
// (`projectData` loose in `core/state.js`) which only throws when the line
// runs. The analysis screen had a harness; the modeling screen did not, and it
// was exactly the one that broke.
//
// The table below has to cover the whole bridge. A new name on the bridge with
// no entry here fails the test -- the decision not to exercise something is
// written down, with its reason, instead of being an oversight.
import { node, clearDom, schemaResponse } from './fake_dom.js';

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
import { registerAnalysis, recordResult, forgetAllAnalyses }
    from '../../frontend/core/analysis_store.js';
import '../../frontend/main.js';

// --- what each name receives -------------------------------------------------
const event = () => ({ preventDefault() {}, stopPropagation() {}, target: { files: [], value: '' } });
const button = () => { const b = node('button'); b.nextElementSibling = node('advanced'); return b; };

const CALLS = {
    // Hub
    openRotorHub: [], returnToHub: [],
    createNewRotorInHub: [], copyRotorInHub: [0], deleteRotorInHub: [0],
    editRotorName: [0], saveRotorFromHub: [0], generatePythonFromHub: [0],
    openRotorWorkspace: [0, 'screen-modeling'],
    // screens and panels
    switchScreen: ['screen-modeling'], toggleSidebar: [], toggleAnalysisSidebar: [],
    openTab: ['shafts'], toggleAdvanced: [button], changeLanguage: ['pt'],
    // element form
    openForm: [true], selectSubType: ['BASIC'], closeForm: [], fillDefault: [],
    editItem: [0], copyItem: [0], deleteItem: [0], saveItem: [],
    handleUnitChange: [() => { const s = node('unit'); s.value = 'meter'; s.id = 'inp-L_unit'; return s; }],
    // rotor
    saveRotor: [event], loadRotor: [event],
    // analyses
    addAnalysis: [event], runCardAnalysis: ['x1', 'modes'], toggleAnalysis: ['x1'],
    deleteAnalysis: [event, 'card-x1'], saveAnalysis: [event],
    loadAnalysis: [event], loadAnalysisDirect: [event],
    generatePythonFile: [], toggleDashAdv: [button], checkDeps: ['x1'],
    addProbeRow: ['x1', 'probes', 'modes'], addForceRow: ['x1', 'forces', 'time_response'],
    addUnbalanceRow: ['x1', 'unbalances', 'unbalance'],
    addAngleProbeRow: ['x1', 'probes', 'unbalance'],
    // help and modals
    openGeneralHelp: [], openAnalysisHelp: [], openSectionHelp: ['shafts'],
    openAnalysisCardHelp: [event, 'modes'], closeHelpModal: [],
    closeCustomAlert: [], closeCustomConfirm: [false], closeCustomPrompt: [null],
    confirmCustomPrompt: [],
    // multirotor and node
    openMultiRotorModal: [], closeMultiRotorModal: [], saveMultiRotor: [],
    switchMultiRotorTarget: ['driving'],
    addElementFromNodeHub: ['shafts'], closeNodeHub: [],
};

// Not called, and why. The list is deliberately short.
const OUTSIDE = {
    exitApplication: 'shuts the server down and closes the window',
};

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}

// The bridge, read from the main.js block itself.
const fs = await import('fs');
const MAIN = fs.readFileSync(new URL('../../frontend/main.js', import.meta.url), 'utf8');
const block = MAIN.slice(MAIN.indexOf('Object.assign(window, {'));
const BRIDGE = block.slice(block.indexOf('{') + 1, block.indexOf('}'))
    .split(',').map(s => s.trim()).filter(Boolean);

console.log('\nThe table covers the bridge');
const withoutEntry = BRIDGE.filter(n => !(n in CALLS) && !(n in OUTSIDE)).sort();
check('every name on the bridge is in the table: ' + withoutEntry.join(', '), withoutEntry.length === 0);
const leftOver = [...Object.keys(CALLS), ...Object.keys(OUTSIDE)]
    .filter(n => !BRIDGE.includes(n)).sort();
check('and the table has no name that left the bridge: ' + leftOver.join(', '),
          leftOver.length === 0);

await schemaReady();

function prepare() {
    clearDom();
    forgetAllAnalyses();
    state.rotorLibrary = [
        { name: 'A', uid: 'u1', materials: [], shafts: [{ n: '0', L: '500' }], disks: [],
          gears: [], couplings: [], seals: [], bearings: [{ n: '0' }, { n: '1' }],
          pointmasses: [], savedAnalyses: [] },
        { name: 'B', uid: 'u2', materials: [], shafts: [{ n: '0', L: '500' }], disks: [],
          gears: [], couplings: [], seals: [], bearings: [{ n: '0' }, { n: '1' }],
          pointmasses: [], savedAnalyses: [] },
    ];
    state.activeRotorIndex = 0;
    state.projectData = JSON.parse(JSON.stringify(state.rotorLibrary[0]));
    state.currentTab = 'shafts';
    state.currentSubType = 'BASIC';
    state.editingIndex = 0;
    registerAnalysis('x1', 'modes', 'Modal Analysis', '');
    recordResult('x1', { num_modes: '4' }, '', null);
}

console.log('\nNone throws on the first line');
for (const name of BRIDGE) {
    if (name in OUTSIDE) { console.log('  --      ' + name + '  (' + OUTSIDE[name] + ')'); continue; }
    prepare();
    const args = CALLS[name].map(a => (typeof a === 'function' ? a() : a));
    let error = null;
    try {
        // The interface's own modals only resolve when someone closes them; we close
        // on the next tick so that no call is left hanging.
        const r = window[name](...args);
        if (r && typeof r.then === 'function') {
            await Promise.resolve();
            window.closeCustomAlert(); window.closeCustomConfirm(false); window.closeCustomPrompt(null);
            await r;
        }
    } catch (e) {
        error = e;
    }
    check(name + (error ? '  -> ' + e_msg(error) : ''), error === null);
}

function e_msg(e) { return e.constructor.name + ': ' + e.message + ' | ' + (e.stack || '').split('\n')[1]; }

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
