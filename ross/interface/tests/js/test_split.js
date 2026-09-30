// Splitting a shaft from the modelling list: what it asks, what it sends, and
// what it does with the answer.
//
// WHY THIS BATTERY EXISTS. The screen's whole contribution to this feature is a
// question -- "shaft #2 (Body) is 300 mm long, how far from its left face?" --
// and everything that makes that question answerable is in the wording: which
// element, how long it is, and which face the distance is measured from. A
// dialog that drops any of the three sends the user to guess, and guessing is
// what `add_nodes` punishes by doing nothing at all.
//
// The refusals are the other half. They are the server's sentences, not ours,
// and the only way they reach the person is if this module shows `body.message`
// instead of a phrase of its own.
import { check, node, shutDown } from './fake_dom.js';

const { applyProject, middleOf, splitPrompt, splitProject } =
    await import('../../frontend/features/split.js');
const { setSchemaLanguage } = await import('../../frontend/core/i18n.js');

// Millimetres, like the form: `domain/units.py` maps a ShaftElement's `L` and
// `odl` to mm.
function project() {
    return {
        name: 'Compressor',
        uid: 'uid_compressor',
        savedAnalyses: [{ type: 'campbell', figure: 'heavy' }],
        materials: [{ name: 'Steel' }],
        shafts: [
            { element_type: 'BASIC', L: '400', odl: '100', tag: 'Inlet' },
            { element_type: 'BASIC', L: '300', odl: '200' },
        ],
        disks: [{ element_type: 'BASIC', n: '2', m: '10' }],
        gears: [],
        couplings: [],
        seals: [],
        bearings: [{ element_type: 'BASIC', n: '0', kxx: '1e6' }],
        pointmasses: [],
    };
}

// --- the default the box opens with -------------------------------------------
//
// Halfway is the one distance inside an element that can never be refused, so
// pressing Enter is always a legal answer.

check('halfway along 400 is 200', middleOf('400') === '200');
check('a blank length offers nothing', middleOf('') === '');
check('and neither does a length of zero', middleOf('0') === '');
check('nor one that is not a number', middleOf('a bit') === '');

// --- the question -------------------------------------------------------------

const named = splitPrompt({ L: '400', tag: 'Inlet' }, 0);
check('the question names the element by its position', named.includes('#1'));
check('and by its name when it has one', named.includes('Inlet'));
check('it says how long the element is', named.includes('400'));
check(
    'and which face the distance is measured from, which is the whole ambiguity',
    /left face/.test(named),
);

const plain = splitPrompt({ L: '300' }, 1);
check('an unnamed element is still named by position', plain.includes('#2'));
check('and its length still travels', plain.includes('300'));
check('with no empty parentheses where the name would be', !/\(\s*\)/.test(plain));
// The same assertion as on the named branch, and it is here because it was
// missing: a mutation that took "left face" out of `splitAsk` passed the whole
// battery, because the only wording check read `splitNamed`. Two sentences, two
// checks -- a guard on one branch is an assertion that the other does not exist.
check('and it says which face the distance is measured from', /left face/.test(plain));

check('a length with no unit is in the form\'s millimetres', named.includes('400 mm'));
const inMetres = splitPrompt({ L: '0.4', L_unit: 'm', tag: 'Inlet' }, 0);
check('a length typed in metres is asked about in metres', inMetres.includes('0.4 m long'));
check('and not in millimetres', !inMetres.includes('mm'));
check(
    'an unnamed element names its unit too',
    splitPrompt({ L: '12', L_unit: 'in' }, 1).includes('12 in long'),
);

setSchemaLanguage('pt');
check(
    'the unit survives the translation',
    splitPrompt({ L: '0.4', L_unit: 'm' }, 0).includes('0.4 m.'),
);
const portuguese = splitPrompt({ L: '400', tag: 'Inlet' }, 0);
check('the question is translated', portuguese !== named);
check('and it is really Portuguese', /face esquerda/.test(portuguese));
check('the numbers survive the translation', portuguese.includes('400'));
setSchemaLanguage('en');

// --- copying the answer back --------------------------------------------------

const target = { name: 'Compressor', uid: 'x', savedAnalyses: [1, 2], shafts: ['old'] };
applyProject(target, { shafts: ['a', 'b'], disks: [], name: 'Something Else' });
check('every list the server sent is copied back', target.shafts.length === 2);
check('an empty list is copied too, because emptying is a change', Array.isArray(target.disks));
check('what is not a list is left alone', target.name === 'Compressor');
check('the saved analyses are not touched', target.savedAnalyses.length === 2);
check('and neither is the identity', target.uid === 'x');
check('an answer with nothing in it does not throw', applyProject({}, undefined) !== null);

// --- the trip to the server ---------------------------------------------------

let sent = null;

const { closeCustomPrompt } = await import('../../frontend/components/modals.js');
const { closeCustomAlert } = await import('../../frontend/components/modals.js');

// The real dialogs, not doubles: module exports are read only, and using the
// real ones is better anyway -- what gets checked is the text in
// `#custom-prompt-message` and `#custom-alert-message`, which is what the
// person actually reads.
async function answering(typed, start) {
    const finished = start();
    await new Promise(resolve => setTimeout(resolve, 0));
    const asked = node('custom-prompt-message').innerText;
    closeCustomPrompt(typed);
    const result = await finished;
    return { asked, result };
}

async function warningFrom(typed, start) {
    const finished = start();
    await new Promise(resolve => setTimeout(resolve, 0));
    closeCustomPrompt(typed);
    await new Promise(resolve => setTimeout(resolve, 0));
    const shown = node('custom-alert-message').innerText;
    closeCustomAlert();
    await finished;
    return shown;
}

function serverAnswers(ok, payload) {
    globalThis.fetch = async (path, options) => {
        sent = { path, body: JSON.parse(options.body) };
        return { ok, json: async () => payload };
    };
}

const data = project();
serverAnswers(true, {
    status: 'success',
    projectData: {
        materials: [{ name: 'Steel' }],
        shafts: [
            { L: '200', odl: '100', tag: 'Inlet' },
            { L: '200', odl: '100', tag: 'Inlet (2)' },
            { L: '300', odl: '200' },
        ],
        disks: [{ n: '3', m: '10' }],
        gears: [],
        couplings: [],
        seals: [],
        bearings: [{ n: '0', kxx: '1e6' }],
        pointmasses: [],
    },
});

const trip = await answering('200', () => splitProject(data, 0));

check('the dialog asked about the element that was clicked', trip.asked.includes('Inlet'));
check('the route was the one the server serves', sent.path === '/api/rotor/split_shaft');
check('the element index went up as a number', sent.body.index === 0);
check('the distance went up as text, so the domain can refuse it', typeof sent.body.offset === 'string');
check('and it is what was typed', sent.body.offset === '200');
check('the project travelled', sent.body.project.shafts.length === 2);
// The heavy half of the project is the saved charts, and splitting does not
// read them.
check('the saved analyses stayed behind', sent.body.project.savedAnalyses === undefined);

check('the call reports that the project changed', trip.result === true);
check('the shaft became two', data.shafts.length === 3);
check('both halves carry the name', data.shafts[0].tag === 'Inlet' && data.shafts[1].tag === 'Inlet (2)');
check('the untouched element is still there', data.shafts[2].L === '300');
check('the disk was renumbered by the server', data.disks[0].n === '3');
// Splitting an element is an edit like any other on this screen, and editing has
// never thrown the computed charts away.
check('the saved analyses survived the split', data.savedAnalyses.length === 1);
check('and the rotor is still the same rotor', data.uid === 'uid_compressor');

// --- the ways out -------------------------------------------------------------

const cancelled = project();
sent = null;
const escaped = await answering(null, () => splitProject(cancelled, 0));
check('Escape sends nothing to the server', sent === null);
check('and reports that nothing changed', escaped.result === false);
check('the project is untouched', cancelled.shafts.length === 2);

const blank = project();
sent = null;
const emptied = await answering('   ', () => splitProject(blank, 0));
check('an empty box is not a distance either', sent === null);
check('and it is not an error', emptied.result === false);

const missing = project();
sent = null;
check('an element that does not exist returns before asking', (await splitProject(missing, 9)) === false);
check('and asks nothing', sent === null);

// --- a refusal shows the server's sentence ------------------------------------

const refused = project();
serverAnswers(false, {
    status: 'error',
    message: 'The split has to fall inside shaft #1, which is 400 mm long: 400 mm would land on node 1, which already exists.',
});
const sentence = await warningFrom('400', () => splitProject(refused, 0));

check('the refusal reached the screen the user reads', /already exists/.test(sentence));
check('naming the node, which is what tells them what to type instead', /node 1/.test(sentence));
check('nothing was changed', refused.shafts.length === 2);

// Control: with no message in the body, the screen still says something rather
// than an empty dialog.
serverAnswers(false, { status: 'error' });
const fallback = await warningFrom('400', () => splitProject(project(), 0));
check('an answer with no message still warns', fallback.length > 0);
check('and it does not show the word undefined', !/undefined/.test(fallback));

shutDown();
