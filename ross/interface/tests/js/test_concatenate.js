// The concatenation modal: what it says before the button is pressed.
//
// WHY THIS BATTERY EXISTS. The hub now has two buttons that read alike and do
// different things -- linking a MultiRotor couples two shafts through a gear
// mesh, each at its own speed; concatenating welds them into one shaft line at
// one speed -- and `Rotor.concatenate` shares the joint node, which is the part
// a user does not expect. Both facts are only in the screen's words. A screen
// that does not distinguish two states asserts that they are the same, so what
// is checked here is the sentence, not the plumbing.
import { check, node, shutDown } from './fake_dom.js';

const { describeJoint, modelOfRotor, openConcatenateModal, saveConcatenation,
    structuralNodes, swapConcatenationOrder } =
    await import('../../frontend/features/concatenate.js');
const { state } = await import('../../frontend/core/state.js');
const { setSchemaLanguage } = await import('../../frontend/core/i18n.js');

function rotor(name, shafts, analyses) {
    return {
        name,
        uid: 'uid_' + name,
        savedAnalyses: analyses || [],
        materials: [],
        // Millimetres, like the form: `domain/units.py` maps a ShaftElement's
        // `L` and `odl` to mm. Nothing here reads them -- the count of shafts is
        // what matters -- but a fixture that reads as metres teaches the next
        // person the wrong unit, which is how the Python fixture got it wrong.
        shafts: Array.from({ length: shafts }, () => ({ L: '250', odl: '50' })),
        disks: [],
        gears: [],
        couplings: [],
        seals: [],
        bearings: [],
        pointmasses: [],
    };
}

// --- how many nodes a rotor occupies ------------------------------------------

check('two shafts span three nodes', structuralNodes(rotor('A', 2)) === 2);
check('one shaft spans two', structuralNodes(rotor('B', 1)) === 1);
check('no shaft at all is zero, and does not throw', structuralNodes(rotor('C', 0)) === 0);
// Four shafts on 0, 1, 3, 4 around a coupling on 2, then a coupling with a
// blank node: the builder puts it on 1, its place in the list, and not on the
// lowest free node. Nodes 0..5, so the rotor ends on 5.
const coupled = rotor('D', 4);
coupled.shafts[2].n = '3';
coupled.shafts[3].n = '4';
coupled.couplings = [{ n: '2' }, {}];
check('a coupling counts, and by its place in the list', structuralNodes(coupled) === 5);
check(
    'blank couplings are placed by the list, past the one shaft',
    structuralNodes({ shafts: [{}], couplings: [{ n: '0' }, {}, {}, {}, {}] }) === 5,
);

// --- the model of a rotor, as the modal shows it ------------------------------
//
// Three answers, and the difference between the first two is the whole reason
// this is not `unanimousConversion`: a rotor nobody has analysed has no model
// and joins anything, while a rotor whose analyses were all 6 DoF has one.

check('a rotor with no analyses has no model', modelOfRotor(rotor('A', 2)) === undefined);
check(
    'analyses that agree give that model',
    modelOfRotor(rotor('A', 2, [{ conversion: '4dof' }, { conversion: '4dof' }])) === '4dof',
);
check(
    'analyses that disagree give none',
    modelOfRotor(rotor('A', 2, [{ conversion: '4dof' }, { conversion: '' }])) === null,
);

// --- the sentence -------------------------------------------------------------

state.rotorLibrary = [rotor('Compressor', 2), rotor('Turbine', 3)];
await openConcatenateModal();

check('the modal opened', node('concatenate-modal-overlay').style.display === 'flex');
check('both rotors are offered as the first', node('cc-first').innerHTML.includes('Compressor'));
check('and as the second', node('cc-second').innerHTML.includes('Turbine'));

const said = () => node('cc-hint').innerHTML;

check('the sentence names where the first ends', said().includes('<b>2</b>'));
check(
    'and says the joint is shared rather than leaving it to be guessed',
    /shared|welded/.test(said()),
);
check('it counts the nodes of the result', said().includes('<b>6</b>'));
check('it names both rotors', said().includes('Compressor') && said().includes('Turbine'));
check('and it says neither has been analysed yet', said().includes('no analyses yet'));

// Control: the sentence is computed and not a fixed string. Two rotors of two
// and three shafts give 3 + 4 = 7 nodes before the joint is shared, 6 after --
// and a rotor of one shaft has to give a different number.
state.rotorLibrary = [rotor('Compressor', 2), rotor('Stub', 1)];
await openConcatenateModal();
check('a different pair gives a different count', said().includes('<b>4</b>'));

// --- the order ----------------------------------------------------------------
//
// A + B is not B + A: the second rotor is the one that moves right. Swapping
// has to change the sentence, or the button is decoration.

state.rotorLibrary = [rotor('Compressor', 2), rotor('Turbine', 3)];
await openConcatenateModal();
const before = said();
swapConcatenationOrder();
check('swapping changes which rotor ends the first half', said() !== before);
check('and the new first is the one that was second', said().indexOf('Turbine') < said().indexOf('Compressor'));

// --- a rotor with itself ------------------------------------------------------

node('cc-second').value = node('cc-first').value;
describeJoint();
check('the same rotor twice is refused in the sentence, before the button', /cannot be concatenated with itself/.test(said()));

// --- both languages -----------------------------------------------------------

state.rotorLibrary = [rotor('Compressor', 2), rotor('Turbine', 3)];
await openConcatenateModal();
const english = said();
setSchemaLanguage('pt');
describeJoint();
const portuguese = said();
check('the sentence is translated', portuguese !== english);
check('and it is really Portuguese', /compartilhada|emendados/.test(portuguese));
check('the numbers survive the translation', portuguese.includes('<b>6</b>'));
setSchemaLanguage('en');

// --- a name is not required ---------------------------------------------------

await openConcatenateModal();
check('the name field starts empty, so the default can be used', node('cc-name').value === '');


// --- the button, and what it sends --------------------------------------------
//
// This half of the module had no battery at all: the sentence was covered, the
// trip to the server was covered only by the Python route tests and by somebody
// remembering to click. The `fetch` double is the one `test_requests.js` already
// uses, and it turns "I tried it and it worked" into something that stays true.

let sent = null;

// The alert is not doubled. `openCustomAlert` is a module export and module
// exports are read only -- and using the real one is better anyway: it writes
// into `#custom-alert-message`, which is the element the person actually reads,
// so what is checked is the sentence on screen and not a call that was made.
//
// It returns a promise that only settles when the dialog is closed, so the call
// is started, the microtasks are let through, the message is read, and the
// dialog is closed to let the caller finish.
const { closeCustomAlert } = await import('../../frontend/components/modals.js');

async function warningFrom(start) {
    const finished = start();
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

state.rotorLibrary = [rotor('Compressor', 2), rotor('Turbine', 3)];
await openConcatenateModal();
serverAnswers(true, {
    status: 'success',
    projectData: { materials: [], shafts: [{ n: '0' }, { n: '1' }, { n: '2' }],
                   disks: [], gears: [], couplings: [], seals: [], bearings: [],
                   pointmasses: [] },
});
await saveConcatenation();

check('the route was the one the server serves', sent.path === '/api/rotor/concatenate');
check('the first rotor went up', sent.body.first.shafts.length === 2);
check('and the second', sent.body.second.shafts.length === 3);
check('the conversions travelled as lists', Array.isArray(sent.body.first_conversions));
check('and they are empty when nothing was analysed', sent.body.first_conversions.length === 0);
// The analyses of the sources do not go up: they were computed on a rotor that
// no longer describes this one, and they are the heavy half of the payload.
check('the saved analyses stayed behind', sent.body.first.savedAnalyses === undefined);

check('the library grew by one', state.rotorLibrary.length === 3);
check('the two sources are still there', state.rotorLibrary[0].name === 'Compressor' && state.rotorLibrary[1].name === 'Turbine');
const born = state.rotorLibrary[2];
check('the new rotor is named after both', born.name === 'Compressor + Turbine');
check('it carries the project the server sent', born.shafts.length === 3);
check('it starts with no analyses of its own', born.savedAnalyses.length === 0);
check('and it has an identity, like every other rotor', typeof born.uid === 'string' && born.uid.length > 0);

// A name typed by hand wins over the default.
await openConcatenateModal();
node('cc-name').value = 'Train';
await saveConcatenation();
check('a typed name is kept', state.rotorLibrary[3].name === 'Train');

// --- a refusal shows the server's sentence ------------------------------------
//
// Not a phrase of ours. "These two rotors were analysed under different rotor
// models" is what tells the person what to do; a generic failure is a guess.

const sizeBefore = state.rotorLibrary.length;
await openConcatenateModal();
serverAnswers(false, { status: 'error', message: 'these two rotors were analysed under different rotor models: 4 DoF and 6 DoF.' });
const refusal = await warningFrom(saveConcatenation);

check('nothing was added to the library', state.rotorLibrary.length === sizeBefore);
check('the refusal reached the screen the user reads', /different rotor models/.test(refusal));
check('with both models named', /4 DoF/.test(refusal) && /6 DoF/.test(refusal));

// Control: with no message in the body, the screen still says something rather
// than an empty dialog.
await openConcatenateModal();
serverAnswers(false, { status: 'error' });
const fallback = await warningFrom(saveConcatenation);
check('an answer with no message still warns', fallback.length > 0);
check('and it does not show the word undefined', !/undefined/.test(fallback));

shutDown();
