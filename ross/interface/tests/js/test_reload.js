// Reloading the page: the state comes back from memory and the screen shows it.
//
// Slice 3 inverted the dependency -- `restoreState` gives back how many rotors
// came back, instead of telling the Hub to redraw, because a core module does
// not tell the screen to draw. The inversion was right and the other end was
// left loose: the bootstrap did not call `renderRotorHub()`. The rotors came
// back into memory and the Hub stayed empty.
//
// It is the second time a dependency inversion loses its other end in this
// slice (the first was `onReorder`). That is why this battery exists.
import { node, clearDom, disk, clearDisk } from './fake_dom.js';

globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({
    language: 'en', categories: {}, unit_map: {}, unit_alternatives: {} }) });
globalThis.addEventListener = () => {};
globalThis.setInterval = () => 0;

// The bootstrap registers itself on DOMContentLoaded; we keep it to fire by hand.
let onLoad = null;
globalThis.document.addEventListener = (event, fn) => {
    if (event === 'DOMContentLoaded') onLoad = fn;
};

const STORED = JSON.stringify({
    version: 1,
    library: [
        { name: 'Compressor A', shafts: [{ L: '500' }], savedAnalyses: [
            { title: 'Modal Analysis', type: 'modes', params: { num_modes: '4' }, conversion: '4dof' },
        ] },
        { name: 'Turbina B', shafts: [{ L: '300' }], savedAnalyses: [] },
    ],
});
disk.content['ross_interface_state_v1'] = STORED;

const { state } = await import('../../frontend/core/state.js');
await import('../../frontend/main.js');

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}

console.log('\nOpening the page with stored state');

check('the bootstrap registered itself on DOMContentLoaded', typeof onLoad === 'function');
check('and nothing was restored before it ran', state.rotorLibrary.length === 0);

onLoad();
await new Promise(ready => setTimeout(ready, 10));

check('both rotors came back', state.rotorLibrary.length === 2);
check('with the right names',
          state.rotorLibrary.map(r => r.name).join(',') === 'Compressor A,Turbina B');
check('the saved analyses came along',
          state.rotorLibrary[0].savedAnalyses.length === 1
          && state.rotorLibrary[0].savedAnalyses[0].conversion === '4dof');

const hubList = node('rotor-hub-list').innerHTML || '';
check('the Hub was drawn', hubList.length > 0);
check('and it shows both rotors',
          hubList.includes('Compressor A') && hubList.includes('Turbina B'));

// Reopening a rotor by itself would fire a computation on the server with
// nobody having asked.
check('stays on the Hub, opening no rotor', state.activeRotorIndex === -1);

console.log('\nWith no stored state');

clearDom();
clearDisk();
state.rotorLibrary = [];
state.activeRotorIndex = -1;
let error = null;
try { onLoad(); } catch (e) { error = e; }
if (error) console.log('          ' + error.message);
check('opening with nothing stored does not throw', !error);
check('and the Hub invents no rotor', state.rotorLibrary.length === 0);

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
