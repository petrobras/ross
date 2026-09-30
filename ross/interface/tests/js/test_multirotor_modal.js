// The MultiRotor modal: which two rotors, and through which gears.
//
// WHY THIS BATTERY EXISTS. The modal had none. It opened with
// `selectedIndex = 1`, which the fake DOM does not model, so any battery would
// have seen both selects on the first rotor and only ever tested "the same
// rotor twice". And measured against ROSS, `rs.MultiRotor` couples the two
// rotors through a gear on each coupled node and refuses anything else -- while
// the modal offered "0, 0", said nothing about gears, and saved whatever it was
// given. The refusal arrived later, on the modelling screen, far from the
// fields that caused it.
import { check, node, shutDown } from './fake_dom.js';

const { describeCoupling, gearNodes, openMultiRotorModal, saveMultiRotor } =
    await import('../../frontend/features/multirotor.js');
const { state } = await import('../../frontend/core/state.js');
const { closeCustomAlert } = await import('../../frontend/components/modals.js');

function rotor(name, gears) {
    return {
        name, uid: 'uid_' + name, savedAnalyses: [{ type: 'modal' }], materials: [],
        shafts: [{ L: '250', odl: '50' }, { L: '250', odl: '50' }],
        disks: [], gears: gears || [], couplings: [], seals: [], bearings: [], pointmasses: [],
    };
}

// The alert is the real one, read where the person reads it.
async function warningFrom(start) {
    const finished = start();
    await new Promise(resolve => setTimeout(resolve, 0));
    const shown = node('custom-alert-message').innerText;
    closeCustomAlert();
    await finished;
    return shown;
}

let sent = [];
function serverAnswers(payload) {
    globalThis.fetch = async (path, options) => {
        sent.push({ path: String(path), body: JSON.parse(options.body) });
        return { ok: payload.status === 'success', json: async () => payload };
    };
}

// --- where the gears are ----------------------------------------------------------

check('a rotor with no gears has none', gearNodes(rotor('A')).length === 0);
check('a typed node is the node', gearNodes(rotor('A', [{ n: '2' }])).join() === '2');
// The same numbering the list uses: a gear with no node takes the next free one.
check('an untyped node is numbered like the list numbers it',
    gearNodes(rotor('A', [{ n: '0' }, { n: '' }])).join() === '0,1');
check('a MultiRotor half is not guessed at', gearNodes({ isMultiRotor: true }) === null);

// --- opening --------------------------------------------------------------------------

state.rotorLibrary = [rotor('Motor <A>', [{ n: '2' }]), rotor('Pump', [{ n: '1' }])];
await openMultiRotorModal();

check('the first rotor drives', node('mr-driving').value === '0');
// What `selectedIndex` hid: the second select really starts on the second rotor.
check('and the second is driven', node('mr-driven').value === '1');
check('a rotor name is text, not markup',
    node('mr-driving').innerHTML.includes('Motor &lt;A&gt;') && !node('mr-driving').innerHTML.includes('<A>'));
check('the modal says where each rotor\'s gear is',
    /has gears at node 2/.test(node('mr-hint').innerHTML) && /has gears at node 1/.test(node('mr-hint').innerHTML));
// One gear on each side: the only coupling ROSS will accept, typed in.
check('and fills in the coupled nodes when there is one way to', node('mr-coupled-nodes').value === '2, 1');

// A rotor with no gear is named as the problem, before anything is saved.
state.rotorLibrary.push(rotor('Plain'));
node('mr-driven').value = '2';
node('mr-coupled-nodes').value = '2, 1';
describeCoupling();
check('a rotor with no gear is called out', /Plain<\/b> has no gear/.test(node('mr-hint').innerHTML));
check('and nothing is filled in for it', node('mr-coupled-nodes').value === '2, 1');

// Two gears on one side: the choice belongs to the person.
state.rotorLibrary.push(rotor('Gearbox', [{ n: '1' }, { n: '3' }]));
node('mr-driven').value = '3';
node('mr-coupled-nodes').value = '9, 9';
describeCoupling();
check('with two gears to pick from, both are listed', /node 1, 3/.test(node('mr-hint').innerHTML));
check('and the field is left to the person', node('mr-coupled-nodes').value === '9, 9');

// A name is filled in as it was typed. `String.replace` reads `$&` in a
// replacement string as "the text matched", and `%2` in a name would be filled
// with the gear nodes if the name went in first.
state.rotorLibrary.push(rotor('R$&%2', [{ n: '4' }]));
node('mr-driven').value = '4';
describeCoupling();
check('a name with $& and %2 in it is shown as typed',
    node('mr-hint').innerHTML.includes('<b>R$&amp;%2</b> has gears at node 4'));
state.rotorLibrary.push(rotor('No$&gear'));
node('mr-driven').value = '5';
describeCoupling();
check('and so is one on the sentence for a rotor with no gear',
    node('mr-hint').innerHTML.includes('<b>No$&amp;gear</b> has no gear'));

// Control: the same rotor on both sides says nothing about gears.
node('mr-driven').value = '0';
describeCoupling();
check('the same rotor twice has no gear sentence', node('mr-hint').innerHTML === '');

// --- saving ------------------------------------------------------------------------------

sent = [];
const same = await warningFrom(saveMultiRotor);
check('the same rotor twice is refused before asking the server',
    /cannot be the same/.test(same) && sent.length === 0);

// ROSS refuses: its sentence is shown, and nothing joins the library.
state.rotorLibrary = [rotor('Motor', [{ n: '2' }]), rotor('Pump')];
await openMultiRotorModal();
serverAnswers({ status: 'error', message: 'Unexpected error (TypeError): Each rotor needs a GearElement in the coupled nodes!' });
sent = [];
const refused = await warningFrom(saveMultiRotor);
check('the MultiRotor is built by the server before it is kept',
    sent.length === 1 && sent[0].path === '/build_rotor' && sent[0].body.project.isMultiRotor === true);
check('without the analyses of the two rotors',
    sent[0].body.project.driving_rotor.savedAnalyses === undefined);
check('ROSS\'s own sentence reaches the person', /needs a GearElement/.test(refused));
check('and the library does not grow', state.rotorLibrary.length === 2);
check('the modal stays open on the fields that caused it',
    node('multirotor-modal-overlay').style.display === 'flex');

// ROSS builds it: it joins the library, linked to its two parents.
state.rotorLibrary = [rotor('Motor', [{ n: '2' }]), rotor('Pump', [{ n: '1' }])];
await openMultiRotorModal();
serverAnswers({ status: 'success', plot_json: '{}' });
await saveMultiRotor();
const made = state.rotorLibrary[2];
check('a MultiRotor ROSS builds joins the library', state.rotorLibrary.length === 3 && made.isMultiRotor === true);
check('coupled through the gears the modal found', made.multi_params.coupled_nodes === '2, 1');
check('and linked to its two parents', made.driving_uid === 'uid_Motor' && made.driven_uid === 'uid_Pump');
check('and the modal closes', node('multirotor-modal-overlay').style.display === 'none');

// No server at all: said so, nothing kept.
globalThis.fetch = async () => { throw new TypeError('Failed to fetch'); };
state.rotorLibrary = [rotor('Motor', [{ n: '2' }]), rotor('Pump', [{ n: '1' }])];
await openMultiRotorModal();
const offline = await warningFrom(saveMultiRotor);
check('with no server, it says it could not check', /could not be checked/.test(offline));
check('and keeps nothing', state.rotorLibrary.length === 2);

shutDown();
