// The work has to survive closing the tab (FE-09) -- and what gets stored has
// to be the model, not the charts: a Plotly figure of a Campbell is over 15 KB
// and would blow the localStorage quota in a handful of projects.
import { disk, clearDisk } from './fake_dom.js';
import {
    ANALYSES, analysesToSave, recordResult, forgetAllAnalyses,
    registerAnalysis,
} from '../../frontend/core/analysis_store.js';
import { state } from '../../frontend/core/state.js';
import {
    STATE_KEY, REFUSED_KEY, STATE_VERSION,
    stateFromDisk, stateToDisk, refuseState, restoreState, saveState,
    persistenceIsOff, resumePersistence,
} from '../../frontend/core/persistence.js';

console.warn = () => {}; console.error = () => {}; console.info = () => {};

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}

function clearAll() {
    clearDisk();
    resumePersistence();
    state.rotorLibrary = [];
    state.activeRotorIndex = -1;
    ANALYSES.clear();
}

const BIG_FIGURE = {
    data: [{ x: Array.from({ length: 600 }, (_, i) => i), y: Array.from({ length: 600 }, () => 1) }],
    layout: { title: 'x'.repeat(400) },
    frames: [{ n: 1 }],
};

// 1) what goes to disk is the model, without the charts
clearAll();
state.rotorLibrary = [{
    name: 'Compressor A',
    shafts: [{ L: '100' }],
    savedAnalyses: [Object.assign({ title: 'Campbell', type: 'campbell',
                                    params: { speed_max: '4000' }, conversion: '4dof' },
                                   BIG_FIGURE)],
}];
check('it wrote', saveState() === true);

const stored = JSON.parse(disk.content[STATE_KEY]);
const storedAnalysis = stored.library[0].savedAnalyses[0];
check('the analysis configuration survives',
    storedAnalysis.type === 'campbell' && storedAnalysis.params.speed_max === '4000');
check('the conversion too', storedAnalysis.conversion === '4dof');
check('the chart does not go along',
    storedAnalysis.data === undefined && storedAnalysis.layout === undefined
    && storedAnalysis.frames === undefined);
check('the rotor itself goes whole', stored.library[0].shafts[0].L === '100');
check('and the result is small', disk.content[STATE_KEY].length < 1000);

// 2) the open rotor uses the live analyses, not the stale savedAnalyses
clearAll();
state.rotorLibrary = [{ name: 'A', savedAnalyses: [{ title: 'stale', type: 'ucs', params: {} }] }];
state.activeRotorIndex = 0;
registerAnalysis('fresh', 'campbell', 'Campbell', '');
recordResult('fresh', { speed_max: '9999' }, '', BIG_FIGURE);
saveState();
const withOpenCards = JSON.parse(disk.content[STATE_KEY]).library[0].savedAnalyses;
check('the open rotor writes what is on screen, not what was stored',
    withOpenCards.length === 1 && withOpenCards[0].type === 'campbell'
    && withOpenCards[0].params.speed_max === '9999');

// 3) a MultiRotor has charts in both children
clearAll();
state.rotorLibrary = [{
    name: 'MR', isMultiRotor: true,
    savedAnalyses: [],
    driving_rotor: { name: 'A', savedAnalyses: [Object.assign({ title: 'x', type: 'modes', params: {} }, BIG_FIGURE)] },
    driven_rotor: { name: 'B', savedAnalyses: [Object.assign({ title: 'y', type: 'modes', params: {} }, BIG_FIGURE)] },
}];
saveState();
const mr = JSON.parse(disk.content[STATE_KEY]).library[0];
check('the charts come out of both rotors of a MultiRotor too',
    mr.driving_rotor.savedAnalyses[0].data === undefined
    && mr.driven_rotor.savedAnalyses[0].data === undefined);

// 4) it does not rewrite what did not change
clearAll();
state.rotorLibrary = [{ name: 'A', savedAnalyses: [] }];
check('the first write happens', saveState() === true);
check('the second, with nothing changed, does not', saveState() === false);
state.rotorLibrary[0].name = 'B';
check('but after a change, it does', saveState() === true);

// 5) an exceeded quota switches off instead of retrying forever
clearAll();
state.rotorLibrary = [{ name: 'A', savedAnalyses: [] }];
disk.quotaExceeded = true;
check('a write that exceeds the quota gives back false', saveState() === false);
check('and switches persistence off', persistenceIsOff() === true);
disk.quotaExceeded = false;
check('with no further attempt afterwards', saveState() === false);

// 6) restoring
clearAll();
state.rotorLibrary = [{ name: 'Compressor A', shafts: [{ L: '100' }], savedAnalyses: [] }];
saveState();
state.rotorLibrary = [];

check('it restored the library', restoreState() === 1);
check('with the right content', state.rotorLibrary[0].name === 'Compressor A');
check('stays on the Hub, reopening no rotor by itself', state.activeRotorIndex === -1);
// `restoreState` gives back how many rotors came back and draws nothing. A
// core module telling the screen to redraw was the only cycle the boundary
// measurement found between core and feature; the line above is already the
// check on the returned value.

// 6b) work saved under the old key still comes back
//
// The key was renamed from `..._estado_v1` when the codebase went to English.
// Without the fallback read, an absent new key looks exactly like a first run,
// and the library of anyone who had already saved would be gone with no error
// and no warning.
clearAll();
state.rotorLibrary = [{ name: 'Old Rotor', shafts: [{ L: '100' }], savedAnalyses: [] }];
saveState();
disk.content['ross_interface_estado_v1'] = disk.content[STATE_KEY];
delete disk.content[STATE_KEY];
state.rotorLibrary = [];
check('state written under the Portuguese key still restores', restoreState() === 1);
check('with its content intact', state.rotorLibrary[0].name === 'Old Rotor');

// Control: without the fallback the check above would pass for the wrong
// reason if the new key were still there.
check('control: the new key really was empty', !(STATE_KEY in disk.content));

// 7) nothing stored is not an error
clearAll();
check('with no stored state, restoring does nothing', restoreState() === 0);

// 8) a corrupt state does not bring things down, and does not vanish
clearAll();
disk.content[STATE_KEY] = '{this is not json';
check('unreadable state does not bring it down', restoreState() === 0);
check('and it is kept for rescue',
    disk.content[REFUSED_KEY] === '{this is not json');
check('leaving the normal key', disk.content[STATE_KEY] === undefined);

// 9) a different version is refused just the same
clearAll();
disk.content[STATE_KEY] = JSON.stringify({ version: 99, library: [{ name: 'X' }] });
check('an unknown version is refused', restoreState() === 0);
check('and kept as well', disk.content[REFUSED_KEY] !== undefined);

// 10) an unexpected shape (no library) too
clearAll();
disk.content[STATE_KEY] = JSON.stringify({ version: 1, library: 'not a list' });
check('a library that is not a list is refused', restoreState() === 0);

console.log(`\n${ok} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
