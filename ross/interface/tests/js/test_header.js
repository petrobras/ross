// Does the badge actually reach the header of the card?
//
// `test_conversion.js` checks the function that builds the badge, and the guard
// in `test_fase3.py` checks that the paths call it. Neither looks at the HTML
// that ends up on screen -- and that is exactly where the user saw no badge at
// all. Testing the unit and testing the call site are not the same thing.
//
// Here the paths that build a card really run, against the fake DOM, and what
// is checked is the header that comes out of them.
import { node, clearDom, schemaResponse } from './fake_dom.js';

// What the interface would ask the server, answered with no server.
let requests = [];
globalThis.fetch = async (path, options) => {
    const schema = schemaResponse(path);
    if (schema) return { ok: true, status: 200, json: async () => schema };
    requests.push({ path, body: options && options.body ? JSON.parse(options.body) : null });
    return {
        ok: true, status: 200,
        json: async () => ({
            status: 'success',
            plot_json: JSON.stringify({ data: [{ x: [1], y: [1] }], layout: {} }),
        }),
    };
};

import { forgetAllAnalyses } from '../../frontend/core/analysis_store.js';
import { addAnalysis, restoreAnalysesFromMemory } from '../../frontend/features/analysis.js';

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}

function list() { return node('analysis-list'); }
function clearAll() {
    clearDom();
    forgetAllAnalyses();
    requests = [];
}

// Only the header, so that "the badge is on the card" cannot be satisfied by a
// badge hidden in the body of the card.
function headers() {
    return (list().innerHTML.match(/<span class="analysis-title">[\s\S]*?<\/span>/g) || []);
}

async function create(conversion) {
    clearAll();
    node('analysis-type').value = 'modes';
    node('rotor-conversion-type').value = conversion;
    await addAnalysis(null);
    return headers()[0] || '';
}

console.log('\nA freshly created card');

const six = await create('');
check('the 6 DoF card is born with a badge', /badge-6dof/.test(six) && /6 DoF/.test(six));

const four = await create('4dof');
check('the 4 DoF card is born with a badge', /badge-4dof/.test(four) && /4 DoF/.test(four));

const torc = await create('torsional');
check('the torsional card is born with a badge', /badge-torsional/.test(torc));

check('the title stays in the header', /Modal Analysis/i.test(six));
check('the created card already asks the server to compute',
          requests.length === 1 && requests[0].path.includes('/run_analysis'));
check('and asks under the model of the card',
          requests[0].body.conversion_type === 'torsional');

console.log('\nCards restored from memory');

clearAll();
await restoreAnalysesFromMemory([
    { title: 'Modal Analysis', type: 'modes', params: { n: '1' }, conversion: '',
      data: [{ x: [1] }], layout: {}, frames: [] },
    { title: 'Modal Analysis', type: 'modes', params: { n: '1' }, conversion: '4dof',
      data: [{ x: [1] }], layout: {}, frames: [] },
    { title: 'Modal Analysis', type: 'modes', params: { n: '1' }, conversion: 'torsional',
      data: [{ x: [1] }], layout: {}, frames: [] },
]);

const restored = headers();
check('three headers came back', restored.length === 3);
check('one of them carries the 6 DoF badge', restored.some(h => /badge-6dof/.test(h)));
check('one of them carries the 4 DoF badge', restored.some(h => /badge-4dof/.test(h)));
check('one of them carries the torsional badge', restored.some(h => /badge-torsional/.test(h)));
check('the three badges differ from one another',
          new Set(restored.map(h => (h.match(/badge-\w+/g) || []).join(','))).size === 3);
check('control: the headers are not empty',
          restored.every(h => h.length > 40));
check('restoring does not call the server by itself', requests.length === 0);

// `loadAnalysis` lives inside a FileReader callback and does not run here; its
// header is checked by reading, so that all three paths stay covered.
console.log('\nCard read from a file');
const fs = await import('fs');
const source = fs.readFileSync(
    new URL('../../frontend/features/analysis.js', import.meta.url), 'utf8');
const line = source.split('\n').find(l => l.includes("t('loadedSuffix')")
                                            && l.includes('analysis-title'));
check('the file header stamps the badge too',
          !!line && line.includes('conversionBadge(an.conversion)'));

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
