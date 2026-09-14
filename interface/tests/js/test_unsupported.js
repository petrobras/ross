// A combination ROSS does not support warns before computing.
//
// The user converted the rotor to torsional, ran an analysis and got the raw
// error from the library. Worse than that: in some combinations there is no
// error at all -- the conversion swaps the rotor's matrix methods, and an
// analysis that builds its own gives back a chart badged "4 DoF" carrying the
// numbers of the full model.
//
// The table lives in the backend, next to the catalogue, and travels with the
// schema: the screen warns by reading exactly what the route will enforce.
import { node, clearDom, schemaResponse } from './fake_dom.js';

let requests = [];
globalThis.fetch = async path => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    requests.push(String(path));
    return { ok: true, status: 200, json: async () => ({
        status: 'success',
        plot_json: JSON.stringify({ data: [{ x: [1], y: [1] }], layout: {} }),
    }) };
};

import { state } from '../../frontend/core/state.js';
import { schemaReady, analysisUnsupported } from '../../frontend/core/schema.js';
import { forgetAllAnalyses, registerAnalysis } from '../../frontend/core/analysis_store.js';
import { addAnalysis, runCardAnalysis } from '../../frontend/features/analysis.js';

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}

await schemaReady();

function clearAll() {
    clearDom();
    forgetAllAnalyses();
    requests = [];
    state.projectData = { shafts: [{ L: '500' }], bearings: [], materials: [],
                           disks: [], gears: [], seals: [], couplings: [], pointmasses: [] };
}

console.log('\nThe table arrived from the server');
check('the torsional UCS is in the table',
          analysisUnsupported('ucs', 'torsional') !== null);
check('and the reason is real text',
          (analysisUnsupported('ucs', 'torsional').reason || '').length > 40);
check('and it carries the failure kind',
          analysisUnsupported('ucs', 'torsional').failure === 'ignored');
check('the 6 DoF UCS is not', analysisUnsupported('ucs', '') === null);
check('nor an analysis the table does not mention',
          analysisUnsupported('campbell', 'torsional') === null);

// Crack, misalignment and rubbing in 4 DoF run without error on ross 2.3.0.
// They are in the table by domain decision: the fault models were built for
// the 6 DoF rotor, and running is not the same as being validated.
check('the three fault models in 4 DoF are in the table',
          ['crack', 'misalignment', 'rubbing']
              .every(t => analysisUnsupported(t, '4dof') !== null));

console.log('\nThe card warns instead of computing');

clearAll();
registerAnalysis('c1', 'ucs', 'UCS Diagram', 'torsional');
await runCardAnalysis('c1', 'ucs');
const card = node('plot-c1').innerHTML;
check('the card shows the warning', /analysis-unsupported/.test(card));
check('with the reason coming from the server',
          card.includes(analysisUnsupported('ucs', 'torsional').reason.slice(0, 40)));
check('and it did not call the server',
          !requests.some(p => p.includes('/run_analysis')));

console.log('\nA supported combination still computes');

clearAll();
registerAnalysis('c2', 'ucs', 'UCS Diagram', '');
await runCardAnalysis('c2', 'ucs');
check('the 6 DoF UCS goes to the server',
          requests.some(p => p.includes('/run_analysis')));
check('and shows no warning', !/analysis-unsupported/.test(node('plot-c2').innerHTML));

clearAll();
registerAnalysis('c3', 'campbell', 'Campbell', 'torsional');
await runCardAnalysis('c3', 'campbell');
check('the torsional Campbell computes too',
          requests.some(p => p.includes('/run_analysis')));

clearAll();
registerAnalysis('c4', 'crack', 'Crack Response', '4dof');
await runCardAnalysis('c4', 'crack');
check('the 4 DoF crack warns instead of computing too',
          /analysis-unsupported/.test(node('plot-c4').innerHTML)
          && !requests.some(p => p.includes('/run_analysis')));

// Control: if `analysisUnsupported` always gave back null, the
// "still computes" checks would pass just the same.
console.log('\nControl');
check('the table is not empty',
          ['ucs', 'unbalance'].every(t => analysisUnsupported(t, 'torsional') !== null));
check('and it tells the failure kinds apart',
          new Set(['ucs+torsional', 'crack+4dof', 'unbalance+torsional']
              .map(k => k.split('+'))
              .map(([t, cv]) => analysisUnsupported(t, cv).failure)).size >= 2);

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
