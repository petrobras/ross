// The bridge between the HTML and the modules.
//
// An ES module has its own scope: nothing is global unless someone puts it
// there. index.html and the HTML the cards generate call ~55 functions by name,
// in `onclick`/`onchange` attributes. If a name leaves the
// `Object.assign(window,...)` of main.js, the matching button stops responding
// -- silently, with no console error until someone clicks.
//
// This test enforces both sides: every name called in a handler is on the
// bridge, and nothing on the bridge stopped being called. The second side
// matters as much as the first: the bridge only shrinks of its own accord if
// someone is measuring.
import { node } from './fake_dom.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

globalThis.fetch = async () => ({ json: async () => ({ status: 'success' }) });

// `fileURLToPath`, and not `.pathname`: on Windows the pathname of a file URL
// comes with a leading slash (`/C:/Users/...`), and the following `path.join`
// produced `C:\C:\Users\...` -- a duplicated drive, ENOENT. It was the only
// defect that showed up when the suite finally ran outside Linux.
const FRONTEND = fileURLToPath(new URL('../../frontend/', import.meta.url));

function files(folder) {
    return fs.readdirSync(folder)
        .filter(n => n.endsWith('.js'))
        .map(n => path.join(folder, n));
}

const SOURCES = [
    path.join(FRONTEND, 'index.html'),
    path.join(FRONTEND, 'main.js'),
    ...files(path.join(FRONTEND, 'core')),
    ...files(path.join(FRONTEND, 'components')),
    ...files(path.join(FRONTEND, 'features')),
].map(file => fs.readFileSync(file, 'utf8'));

// Names called from inside an event attribute. A method (`e.stopPropagation`)
// does not count: what needs the bridge is the function called by bare name.
function calledInHandlers(text) {
    const names = new Set();
    for (const handler of text.matchAll(/\bon[a-z]+=\\?(["'])(.*?)\1/g)) {
        for (const call of handler[2].matchAll(/(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(/g)) {
            names.add(call[1]);
        }
    }
    return names;
}

// Four names no sweep finds: `buildDashboardHTML` picks the function through a
// variable and interpolates it into the onclick
// (`onclick="${btnFunc}(...)"`). They are declared here on purpose -- and the
// test below checks that all four still exist.
const BY_VARIABLE = ['addProbeRow', 'addForceRow', 'addUnbalanceRow', 'addAngleProbeRow'];

const called = new Set(BY_VARIABLE);
for (const source of SOURCES) for (const n of calledInHandlers(source)) called.add(n);

// What the bridge publishes, read from the block itself.
const MAIN = fs.readFileSync(path.join(FRONTEND, 'main.js'), 'utf8');
const block = MAIN.slice(MAIN.indexOf('Object.assign(window, {'));
const published = new Set(
    block.slice(block.indexOf('{') + 1, block.indexOf('}'))
        .split(',').map(s => s.trim()).filter(Boolean));

await import('../../frontend/main.js');

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}

console.log('\nThe bridge covers the HTML');

const notOnTheBridge = [...called].filter(n => typeof window[n] !== 'function').sort();
check('every name called in a handler reached the window: ' + notOnTheBridge.join(', '),
          notOnTheBridge.length === 0);

const outsideTheBlock = [...called].filter(n => !published.has(n)).sort();
check('and all of them come from the bridge block, not from leftovers elsewhere: '
          + outsideTheBlock.join(', '), outsideTheBlock.length === 0);

check('the four picked by variable really exist',
          BY_VARIABLE.every(n => typeof window[n] === 'function'));

console.log('\nThe bridge publishes no more than needed');

const unused = [...published].filter(n => !called.has(n)).sort();
check('nothing on the bridge stopped being called: ' + unused.join(', '),
          unused.length === 0);

check('the bridge is the size the HTML asks for',
          published.size === called.size);

// Control: if `calledInHandlers` stopped finding anything, both sides above
// would pass empty.
check('control: the sweep really found handlers', called.size > 40);

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
