// 2D or 3D: which figure the rotor panel shows, and what it asks the server for.
//
// WHY THIS BATTERY EXISTS. The 3D view saves the server its slowest step --
// ROSS's 2D figure, about a second on a large rotor -- by not asking for it.
// That makes the 2D figure able to fall behind the project, and the one thing
// that must never happen is the panel showing a 2D rotor that is not the one in
// the lists. So the claims are about the requests: 3D asks for no figure,
// coming back to 2D after an edit asks for one there and then, and the choice
// survives a reload.
//
// Node has no WebGL and no three.js, which is itself a case worth pinning: the
// 3D view has to say so on screen instead of throwing.
import { clearDom, disk, node, registerSelector } from './fake_dom.js';

// The 3D view logs why it could not draw; kept here to check it did.
const logged = [];
console.error = (...args) => logged.push(args.join(' '));

let bodies = [];
let answer = () => ({ status: 'success', mass: 1, ip: 1, scene: { kind: 'rotor', nodes: [], shafts: [] } });
// A build that takes its time, as ROSS solving a tilting pad does: held until
// `release()`, and cancelled, as a real fetch is, when its signal aborts.
let holding = false;
let releases = [];
const release = () => { releases.forEach(go => go()); releases = []; };
globalThis.fetch = async (path, options = {}) => {
    const body = options.body ? JSON.parse(options.body) : null;
    if (String(path).includes('/build_rotor')) bodies.push(body);
    if (holding && String(path).includes('/build_rotor')) {
        await new Promise((go, stop) => {
            releases.push(go);
            if (options.signal) {
                options.signal.addEventListener('abort', () => {
                    const error = new Error('aborted');
                    error.name = 'AbortError';
                    stop(error);
                });
            }
        });
    }
    const reply = answer(body);
    if (body && body.figure && reply.status === 'success') reply.plot_json = JSON.stringify({ data: [], layout: {} });
    return { ok: true, status: 200, json: async () => reply };
};

const { state } = await import('../../frontend/core/state.js');
const { t } = await import('../../frontend/core/i18n.js');
const { buildRotorLive, setRotorView, startRotorViewToggle } = await import('../../frontend/features/modeling.js');

let ok = 0, failed = 0;
function check(description, condition) {
    if (condition) { ok++; console.log('  ok      ' + description); }
    else { failed++; console.log('  FAILED  ' + description); }
}
const settle = ms => new Promise(done => setTimeout(done, ms));

function prepare() {
    clearDom();
    const buttons = ['2d', '3d'].map(view => {
        const button = node('button-' + view);
        button.dataset.view = view;
        return button;
    });
    registerSelector('[data-action="set-rotor-view"]', buttons);
    state.projectData = {
        materials: [], shafts: [{ n: '0', L: '500', odl: '50' }], disks: [], gears: [],
        couplings: [], seals: [], bearings: [], pointmasses: [],
    };
    bodies = [];
    return buttons;
}
const pressed = buttons => buttons.map(b => b.getAttribute('aria-pressed')).join(',');

// --- the default is ROSS's figure ------------------------------------------------
console.log('\nThe 2D figure, as before');

disk.content = {};
let buttons = prepare();
startRotorViewToggle();
check('with nothing remembered the panel opens on the 2D figure',
    node('plot-rotor').hidden === false && node('rotor-3d').hidden === true && pressed(buttons) === 'true,false');
buildRotorLive();
await settle(700);
check('and a build asks for the figure', bodies.length === 1 && bodies[0].figure === true);

// --- 3D ------------------------------------------------------------------------------
console.log('\nThe 3D view');

setRotorView('3d');
await settle(50);
check('the 3D view takes the panel, and the vertical scale goes with the 2D figure',
    node('plot-rotor').hidden === true && node('rotor-3d').hidden === false && node('rotor-scale').hidden === true);
check('the buttons say which one is on', pressed(buttons) === 'false,true');
check('the choice is remembered', disk.content['ross-rotor-view'] === '3d');
check('without WebGL or three.js the panel says so instead of throwing',
    node('rotor-3d-message').hidden === false && node('rotor-3d-message').textContent === t('rotor3dUnavailable'));
check('and the reason goes to the console for whoever looks', logged.some(line => line.startsWith('3D view:')));

bodies = [];
buildRotorLive();
await settle(700);
check('an edit with the 3D view up asks for the scene and not the figure',
    bodies.length === 1 && bodies[0].figure === false);

// --- back to 2D --------------------------------------------------------------------
console.log('\nBack to the 2D figure');

bodies = [];
setRotorView('2d');
await settle(50);
check('the figure built while in 3D is asked for at once, not left behind the project',
    bodies.length === 1 && bodies[0].figure === true);
bodies = [];
setRotorView('2d');
await settle(50);
check('and not again when it is already current', bodies.length === 0);

// --- remembered across a reload ----------------------------------------------------------
console.log('\nA reload');

disk.content['ross-rotor-view'] = '3d';
buttons = prepare();
startRotorViewToggle();
check('opens on the view the person left', node('rotor-3d').hidden === false && pressed(buttons) === 'false,true');
disk.content['ross-rotor-view'] = 'sideways';
buttons = prepare();
startRotorViewToggle();
check('and on 2D when what is stored is not a view', node('plot-rotor').hidden === false);

// --- a build that takes its time ------------------------------------------------------
console.log('\nA long build');

// What the person sees of a build: dimmed and closed to the pointer, and the
// card that says it is computing.
const closed = id => node(id).style.opacity === '0.4' && node(id).style.pointerEvents === 'none';
const open = id => node(id).style.opacity === '1' && node(id).style.pointerEvents === 'auto';

disk.content['ross-rotor-view'] = '3d';
buttons = prepare();
startRotorViewToggle();
holding = true;
buildRotorLive();
await settle(650);
check('the 3D view is closed to the pointer while ROSS builds', node('rotor-3d').style.pointerEvents === 'none');
await settle(550);
check('and after half a second says it is computing, as the 2D figure does',
    node('rotor-3d-busy').hidden === false && node('rotor-3d-busy').innerHTML.includes(t('computingElement')));
release();
await settle(50);
check('the answer takes the card down and opens the view again', node('rotor-3d-busy').hidden === true && open('rotor-3d'));

// Leonardo's tilting pad: long enough to switch views in the middle. The
// build started in 3D is overtaken by the one the 2D figure asks for, and
// returns without a word -- it used to leave the 3D view closed for good.
buildRotorLive();
await settle(1200);
check('(a build under way in 3D)', closed('rotor-3d') || node('rotor-3d-busy').hidden === false);
setRotorView('2d');
await settle(50);
check('switching to 2D in the middle takes the 3D view\'s card down and opens it again',
    node('rotor-3d-busy').hidden === true && open('rotor-3d'));
check('while the 2D figure is the one waiting now', closed('plot-rotor'));
setRotorView('3d');
await settle(50);
check('and back in 3D, the view answers the pointer', open('rotor-3d') && node('rotor-3d-busy').hidden === true);
release();
await settle(50);
check('the 2D build answered, nothing is left closed', open('plot-rotor') && open('rotor-3d'));

// A rotor ROSS refuses after a long wait: the card goes with the message.
buildRotorLive();
await settle(1200);
answer = () => ({ status: 'error', message: 'no film' });
release();
await settle(50);
check('a refusal takes the card down too', node('rotor-3d-busy').hidden === true && open('rotor-3d'));
holding = false;
answer = () => ({ status: 'success', mass: 1, ip: 1, scene: { kind: 'rotor', nodes: [], shafts: [] } });

// --- what the 3D view says when there is nothing to draw ---------------------------------
console.log('\nNothing to draw');

disk.content['ross-rotor-view'] = '3d';
buttons = prepare();
startRotorViewToggle();
answer = () => ({ status: 'error', message: 'Add at least one Shaft!' });
buildRotorLive();
await settle(700);
check('a rotor ROSS refuses is said in the 3D view too',
    node('rotor-3d-message').hidden === false && node('rotor-3d-message').textContent.includes('Add at least one Shaft!'));
state.projectData.shafts = [];
buildRotorLive();
await settle(700);
check('and so is a project with no shaft yet', node('rotor-3d-message').textContent === t('addOneShaft'));

console.log('\n' + ok + ' checks ok, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
