// The rotor figure following the width without losing its height.
//
// WHY THIS BATTERY EXISTS. The figure used to follow the window through
// Plotly's `responsive: true`. Measured in a browser while building the list
// toggle, that answers every resize with `relayout({autosize: true})`, which
// throws away the layout's height: ROSS's 332 px became the container's 676,
// and a 5× stretch lost the height it had computed. Every check of the vertical
// scale passed, because none of them resized anything.
//
// Hiding the list is a resize, so is hiding the sidebar, zooming the page or
// dragging the window. What is checked is the layout that reaches Plotly after
// one of them.
import { check, node, shutDown } from './fake_dom.js';

const drawn = [];
globalThis.Plotly = {
    newPlot: async (id, data, layout, config) => { drawn.push({ layout, config }); },
    Plots: { resize() {} },
};
// Node's global is not an EventTarget; the application only ever adds one
// listener and the battery fires it.
const listeners = [];
globalThis.addEventListener = (type, handler) => { if (type === 'resize') listeners.push(handler); };
const resize = () => listeners.forEach(handler => handler());
const wait = ms => new Promise(done => setTimeout(done, ms));

// What ROSS answers: height 332, margins t:100 b:102, 1:1 on the y axis.
let answer = 'figure';
globalThis.fetch = async () => ({
    ok: true, status: 200,
    json: async () => (answer === 'figure'
        ? { status: 'success', mass: 1, ip: 1, plot_json: JSON.stringify({ data: [], layout: {
            height: 332, margin: { l: 70, r: 25, t: 100, b: 102 },
            yaxis: { scaleanchor: 'x', scaleratio: 1, title: { text: 'Shaft radius (m)' } },
            updatemenus: [{ y: -0.4615 }],
        } }) }
        : { status: 'error', message: 'no' }),
});

const { buildRotorLive, setVerticalScale, startRotorFigureFollowsWidth } =
    await import('../../frontend/features/modeling.js');
const { state } = await import('../../frontend/core/state.js');

state.projectData = {
    materials: [], shafts: [{ L: '250', odl: '50' }], disks: [], gears: [],
    couplings: [], seals: [], bearings: [], pointmasses: [],
};
node('plot-rotor').offsetWidth = 800;
startRotorFigureFollowsWidth();

buildRotorLive();
await wait(700);
const last = () => drawn[drawn.length - 1];
check('the figure reached the screen', drawn.length === 1);
// The one that says Plotly's own resizing is off: with it on, the next resize
// would relayout the figure behind this module's back.
check('without handing its size to Plotly', last().config.responsive === false);

// --- a resize ----------------------------------------------------------------------

resize(); resize(); resize();
await wait(200);
check('a burst of resizes redraws once', drawn.length === 2);
check('with ROSS\'s height', last().layout.height === 332);
check('and ROSS\'s buttons where it put them', last().layout.updatemenus[0].y === -0.4615);

setVerticalScale(5);
resize();
await wait(200);
check('a stretched figure keeps its stretch across a resize',
    last().layout.height === 130 * 5 + 202 && last().layout.yaxis.scaleratio === 5);
setVerticalScale(1);

// --- not drawing what is not there -------------------------------------------------

// On another screen the div has no width, and Plotly would draw it at 700 px.
const before = drawn.length;
node('plot-rotor').offsetWidth = 0;
resize();
await wait(200);
check('a hidden figure is not redrawn', drawn.length === before);
node('plot-rotor').offsetWidth = 800;

// Every shaft deleted: the note takes the figure's place, and a resize must not
// bring back a rotor that no longer exists.
state.projectData.shafts = [];
buildRotorLive();
await wait(700);
const afterNote = drawn.length;
resize();
await wait(200);
check('after the last shaft goes, a resize draws nothing', drawn.length === afterNote);
setVerticalScale(2);
check('and neither does the scale picker', drawn.length === afterNote);
setVerticalScale(1);

// The same for a model the server refused.
state.projectData.shafts = [{ L: '250', odl: '50' }];
buildRotorLive();
await wait(700);
answer = 'error';
buildRotorLive();
await wait(700);
const afterError = drawn.length;
resize();
await wait(200);
check('nor after an error took its place', drawn.length === afterError);

shutDown();
