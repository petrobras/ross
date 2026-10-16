// Discretizing the shafts from the list: the dialog of the three ways, what it
// sends, what it shows before changing anything, and that nothing changes
// until "yes".
//
// WHY THIS BATTERY EXISTS. Discretizing can turn three rows into sixty and
// renumber every disk and bearing. The screen's part is to ask for exactly
// what the way chosen needs, to say before it happens what will be cut -- and
// for the convergence, which meshes were analysed and which one was chosen --
// and to leave the project alone when the person says no. The cut itself is
// Python's (tests/test_meshing.py, held to ROSS's refine_mesh and
// refine_mesh_by_convergence).
import { check, node, registerSelector, shutDown } from './fake_dom.js';

const { convergenceLines, meshProject, meshSummary } = await import('../../frontend/features/mesh.js');
const { chosenMethod, showMethodFields } = await import('../../frontend/components/mesh_dialog.js');
const { closeCustomAlert, closeCustomConfirm } = await import('../../frontend/components/modals.js');
const { DIALOG_ACTIONS } = await import('../../frontend/features/dialog_actions.js');
const { setSchemaLanguage, t } = await import('../../frontend/core/i18n.js');

// The dialog as index.html has it: three ways, the fields of each, the buttons.
const radios = ['ratio', 'parts', 'convergence'].map(value => Object.assign(node('radio-' + value), { value, checked: value === 'ratio' }));
registerSelector('input[name="mesh-method"]', radios);
const blocks = ['ratio', 'parts', 'convergence'].map(method => {
    const block = node('fields-' + method);
    block.dataset.method = method;
    return block;
});
registerSelector('.mesh-fields', blocks);
const controls = [node('mesh-go-button'), node('mesh-cancel-button'), ...radios];
registerSelector('#mesh-overlay button, #mesh-overlay input', controls);
const DEFAULTS = { 'mesh-max-ld': '0.5', 'mesh-parts': '2', 'mesh-modes': '6', 'mesh-rtol': '0.1', 'mesh-speed': '0' };
function resetDialog() {
    Object.entries(DEFAULTS).forEach(([id, value]) => { node(id).value = value; });
    choose('ratio');
}
function choose(method) {
    radios.forEach(radio => { radio.checked = radio.value === method; });
    DIALOG_ACTIONS['mesh-method']();
}

function project() {
    return {
        name: 'Rig', uid: 'uid_rig', savedAnalyses: [{ figure: 'heavy' }],
        materials: [{ name: 'Steel' }],
        shafts: [
            { L: '400', odl: '200', tag: 'Inlet' },
            { L: '10', odl: '200' },
            { L: '300', odl: '200', tag: 'Outlet' },
        ],
        disks: [{ n: '2', m: '10' }], gears: [], couplings: [], seals: [],
        bearings: [{ n: '0' }, { n: '3' }], pointmasses: [],
    };
}

const REPORT = {
    cut: [{ rows: [0], parts: 4, ratio: 2 }, { rows: [2], parts: 3, ratio: 1.5 }],
    short: [{ rows: [1], ratio: 0.05 }],
    kept: [],
    before: 3,
    after: 8,
};
const CONVERGENCE = {
    rows: [
        { max_ld: null, elements: 3, change: 8.88, wn: [30.7] },
        { max_ld: 2, elements: 8, change: 0.0125, wn: [30.69] },
        { max_ld: 1, elements: 14, change: null, wn: [30.69] },
    ],
    chosen: 1, n_modes: 6, rtol: 0.1,
};
const CUT = {
    shafts: new Array(8).fill(0).map((_, i) => ({ L: '100', odl: '200', tag: 'piece ' + i })),
    disks: [{ n: '5', m: '10' }],
    bearings: [{ n: '0' }, { n: '8' }],
};

let sent = null;
let busyWhenSent = null;
function serverAnswers(ok, payload) {
    globalThis.fetch = async (path, options) => {
        sent = { path, body: JSON.parse(options.body) };
        busyWhenSent = { line: node('mesh-busy').hidden === false, text: node('mesh-busy').innerText, locked: controls.every(c => c.disabled) };
        return { ok, json: async () => payload };
    };
}
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

// Opens the dialog, lets `setUp` fill it, presses Compute (or Cancel when
// `press` is 'cancel'), then answers the next dialog: `yes` true/false for a
// confirm, 'alert' to close an alert.
async function run(data, indexes, setUp, yes, press = 'go') {
    resetDialog();
    const finished = meshProject(data, indexes);
    await tick();
    const opened = node('mesh-overlay').style.display;
    const scope = node('mesh-scope').innerText;
    if (setUp) setUp();
    DIALOG_ACTIONS[press === 'go' ? 'mesh-go' : 'mesh-cancel']();
    await tick(); await tick(); await tick();
    const closed = node('mesh-overlay').style.display;
    const confirmed = node('custom-confirm-message').innerText;
    const alerted = node('custom-alert-message').innerText;
    if (yes === 'alert') closeCustomAlert(); else if (yes !== undefined) closeCustomConfirm(yes);
    const result = await finished;
    node('custom-confirm-message').innerText = '';
    node('custom-alert-message').innerText = '';
    return { opened, scope, closed, confirmed, alerted, result };
}

// --- the dialog --------------------------------------------------------------------
console.log('\nThe dialog: three ways, the fields of the one chosen');

resetDialog();
check('it opens on the L/D ratio, with its field alone', chosenMethod() === 'ratio'
    && blocks.map(b => b.hidden).join() === 'false,true,true');
choose('parts');
check('choosing the parts shows the number of parts alone', blocks.map(b => b.hidden).join() === 'true,false,true');
choose('convergence');
check('choosing the convergence shows its frequencies, tolerance and speed', blocks.map(b => b.hidden).join() === 'true,true,false');
radios.forEach(radio => { radio.checked = false; });
showMethodFields();
check('nothing checked is read as the ratio, never as nothing', chosenMethod() === 'ratio');

// --- the plan in words ---------------------------------------------------------------
console.log('\nThe plan, before anything changes');

const words = meshSummary(REPORT, project().shafts);
check('says how many elements the shafts go from and to', words.includes('3') && words.includes('8'));
check('names each shaft cut by its position and its name', words.includes('#1 Inlet') && words.includes('#3 Outlet'));
check('with its L/D and the parts it becomes', words.includes('2.00') && words.includes('4'));
check('and what cutting cannot fix, by the API 684 floor', words.includes(t('meshShortTitle')) && words.includes('#2') && words.includes('0.05'));
const many = { ...REPORT, cut: new Array(20).fill(0).map((_, i) => ({ rows: [i], parts: 2, ratio: 1 })) };
check('a long plan ends in "..." instead of a dialog taller than the screen', meshSummary(many, []).split('\n').length < 15);
check('a span kept whole beside a coupling is said',
    meshSummary({ ...REPORT, kept: [{ rows: [1], why: 'coupling' }] }, project().shafts).includes('#2'));
const table = convergenceLines(CONVERGENCE);
check('the convergence lists every mesh analysed, the rotor as it is first',
    table.length === 4 && table[1].startsWith(t('meshConvAsIs')) && table[2].startsWith('L/D 2') && table[3].startsWith('L/D 1'));
check('with its elements and how much refining it changed the frequencies',
    table[1].includes('3') && table[1].includes('8.880') && table[2].includes('0.013'));
check('and the one chosen marked, only it', table.filter(line => line.includes(t('meshConvChosen'))).length === 1
    && table[2].includes(t('meshConvChosen')));
check('the last mesh has no change of its own to show', !table[3].includes('%'));
check('none converged is said', convergenceLines({ ...CONVERGENCE, chosen: null }).includes(t('meshConvNone')));

// --- the trip ---------------------------------------------------------------------------------
console.log('\nAsking, sending, taking the answer on');

const data = project();
serverAnswers(true, { status: 'success', projectData: CUT, report: REPORT });
const accepted = await run(data, [], () => { node('mesh-max-ld').value = '0,5'; }, true);
check('the ruler opens the dialog, over every shaft when none is ticked', accepted.opened === 'flex' && accepted.scope === t('meshScopeAll'));
check('the route is the one the server serves', sent.path === '/api/rotor/mesh_shafts');
check('the way and its number go up as text, a decimal comma read as a point',
    sent.body.method === 'ratio' && sent.body.max_ld === '0.5');
check('nothing ticked asks for every shaft', Array.isArray(sent.body.indexes) && sent.body.indexes.length === 0);
check('the saved analyses stay behind', sent.body.project.savedAnalyses === undefined);
check('while the server works the dialog says so and nothing can be pressed',
    busyWhenSent.line && busyWhenSent.text === t('meshBusy') && busyWhenSent.locked);
check('the dialog closes with the answer, and the plan is shown before the project is taken on',
    accepted.closed === 'none' && accepted.confirmed.includes('#1 Inlet'));
check('yes takes the project on', accepted.result === true && data.shafts.length === 8 && data.disks[0].n === '5');
check('and keeps the saved analyses and the identity', data.savedAnalyses.length === 1 && data.uid === 'uid_rig');
check('the dialog can be used again', controls.every(c => !c.disabled));

const declined = project();
serverAnswers(true, { status: 'success', projectData: CUT, report: REPORT });
const no = await run(declined, [], null, false);
check('no leaves the project exactly as it was', no.result === false && JSON.stringify(declined) === JSON.stringify(project()));

const ticked = project();
serverAnswers(true, { status: 'success', projectData: CUT, report: REPORT });
const inParts = await run(ticked, [0, 2], () => { choose('parts'); node('mesh-parts').value = ' 3 '; }, false);
check('the dialog says which shafts, when some are ticked', inParts.scope.includes('2'));
check('in parts: the number of parts goes up, and the shafts ticked', sent.body.method === 'parts'
    && sent.body.parts === '3' && JSON.stringify(sent.body.indexes) === '[0,2]');

const converging = project();
serverAnswers(true, { status: 'success', projectData: CUT, report: { ...REPORT, convergence: CONVERGENCE } });
const conv = await run(converging, [1], () => {
    choose('convergence');
    node('mesh-modes').value = '4'; node('mesh-rtol').value = '0,2'; node('mesh-speed').value = '3600';
}, true);
check('the convergence sends its frequencies, tolerance and speed', sent.body.method === 'convergence'
    && sent.body.n_modes === '4' && sent.body.rtol === '0.2' && sent.body.speed === '3600');
check('and always the whole rotor, whatever is ticked', sent.body.indexes.length === 0);
check('it says it is computing each mesh', busyWhenSent.text === t('meshBusyConvergence'));
check('its table comes before the plan', conv.confirmed.indexOf(t('meshConvChosen')) >= 0
    && conv.confirmed.indexOf(t('meshConvChosen')) < conv.confirmed.indexOf('#1 Inlet'));
check('and yes takes the converged project on', conv.result === true && converging.shafts.length === 8);

// --- the ways out ---------------------------------------------------------------------------
console.log('\nThe ways out');

sent = null;
const cancelled = await run(project(), [], null, undefined, 'cancel');
check('Cancel sends nothing, closes the dialog and changes nothing',
    sent === null && cancelled.result === false && cancelled.closed === 'none');
check('a rotor with no shaft opens nothing', (await meshProject({ shafts: [] }, [])) === false);

const refused = project();
serverAnswers(false, { status: 'error', message: "The number of parts has to be a whole number of 1 or more, not 'many'." });
const refusal = await run(refused, [], () => { choose('parts'); node('mesh-parts').value = 'many'; }, 'alert');
check('a refusal shows the server\'s sentence', /not 'many'/.test(refusal.alerted));
check('and changes nothing', refusal.result === false && refused.shafts.length === 3);

const already = project();
serverAnswers(true, { status: 'success', projectData: project(), report: { cut: [], short: REPORT.short, kept: [], before: 3, after: 3, convergence: { ...CONVERGENCE, chosen: 0 } } });
const nothing = await run(already, [], () => choose('convergence'), 'alert');
check('nothing to cut says so, with no question to answer', nothing.alerted.includes(t('meshNothing')) && nothing.confirmed === '');
check('and still shows the convergence and what is too short', nothing.alerted.includes(t('meshConvChosen')) && nothing.alerted.includes('#2'));
check('and changes nothing', nothing.result === false);

globalThis.fetch = async () => { throw new Error('offline'); };
const offline = await run(project(), [], null, 'alert');
check('a server that does not answer closes the dialog and says so', offline.closed === 'none' && offline.alerted === t('meshFailed'));

setSchemaLanguage('pt');
check('the dialog is translated', t('meshByConvergence') === 'Pela convergência das frequências naturais');
setSchemaLanguage('en');

shutDown();
