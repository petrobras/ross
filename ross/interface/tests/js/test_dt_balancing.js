// Step 4 of the Digital Twin on screen: the node pickers and the sensor cards.
//
// WHY THIS BATTERY EXISTS. The node pickers are filled from the rotor the step
// before builds, and they were twice on screen and twice invisible: first
// because nothing built the model until a button was pressed, then because the
// panel they sit in is a flex column 360px wide, and a tall panel is shrunk by
// the flexbox until its content is clipped away. Neither failure shows in a
// Python test: the routes answered the right nodes both times. So the claims
// here are about the screen -- the options the `<select>` ends up with, and
// where the sensor cards are put.
import { check, clearDom, node, shutDown } from './fake_dom.js';
import { registerExtensionImports } from './extension_imports.mjs';

registerExtensionImports();

// The fake DOM has no layout, so `dispatchEvent` -- which `switchScreen` fires
// to let the figures resize -- has to exist for the screen change to finish.
globalThis.dispatchEvent = () => {};
// Plotly draws into a real element, which the fake DOM has none of. What is
// checked here is which boxes it was asked to draw in.
const drawn = [];
globalThis.window = globalThis.window || {};
globalThis.window.Plotly = { newPlot: (id, data) => drawn.push([String(id), data.length]) };
globalThis.Event = class { constructor(type) { this.type = type; } };

const ROTOR_NODES = 7;

let planes = { measuring: [], balancing: [] };
let ran = null;
let applied = new Set();
let parameters = null;
const asked = [];

globalThis.fetch = async (path, options = {}) => {
    const route = String(path);
    const body = options.body ? JSON.parse(options.body) : null;
    asked.push(route);
    return {
        ok: true,
        status: 200,
        json: async () => {
            if (route.includes('/api/dt/session')) return { id: 'a'.repeat(16), name: 'S' };
            if (route.includes('/api/dt/planes')) {
                planes = { measuring: body.measuring, balancing: body.balancing };
                return step();
            }
            if (route.includes('/api/dt/sensor')) {
                applied.add(body.node + ' ' + body.plane);
                return step();
            }
            if (route.includes('/api/dt/balancing-parameters')) {
                parameters = body.content ? { name: body.filename, source: 'file', keys: ['indirect'] } : null;
                return analysis(false);
            }
            if (route.includes('/api/dt/balancing-polar')) {
                asked.polar = route;
                return {
                    view: 'A',
                    sensor: 'Sensor #1',
                    charts: planes.balancing.map(plane => ({
                        node: plane.node,
                        figure: { data: [{ type: 'scatterpolar', r: [1], theta: [10] }], layout: {} },
                    })),
                };
            }
            if (route.includes('/api/dt/balancing-settings')) return analysis(false);
            if (route.includes('/api/dt/balancing')) {
                ran = body;
                // A real run takes seconds to minutes; this one takes long
                // enough for the loader to be looked at.
                await new Promise(done => setTimeout(done, 10));
                return analysis(true);
            }
            if (route.includes('/api/dt/user-model')) {
                return {
                    session: { equipment: body.name, model: 'user_rotor', rotation: body.rotation, has_rotor: true },
                    rotor: { nodes: ROTOR_NODES, mass: 88.2, bearings: 2, disks: 2, bearing_nodes: [0, 6] },
                    components: { bearings: 'b', labyrinths: '' },
                    nan_warning: false, settings: null, messages: [],
                    measurement: step(), figure: { data: [], layout: {} },
                };
            }
            return {};
        },
    };
};

// What the server answers about the analysis. `ready` is its decision, and the
// button follows it rather than deciding for itself.
function analysis(done) {
    return {
        available: true,
        message: '',
        parameters,
        parameters_warning: parameters ? '' : 'will not match the Digital Twin app',
        ready: pending().length === 0 && planes.balancing.length > 0,
        refusal: pending().length
            ? 'Apply the signal of every sensor.'
            : (planes.balancing.length ? '' : 'Choose at least one balancing plane.'),
        balancing: done ? {
            planes: [{
                node: 3, radius: 120.0, unbalance: 60.0,
                unbalance_phase: 10.0, mass: 0.5, phase: 190.0,
            }],
            loss: 0.1,
            speed: [840.0, 'rpm'],
            measuring: planes.measuring,
            phases: { 'Sensor #1': [10.0], 'Sensor #2': [100.0] },
            sensors: [
                { node: 2, plane: 'X', key: 'Sensor #1' },
                { node: 2, plane: 'Y', key: 'Sensor #2' },
                { node: 4, plane: 'X', key: 'Sensor #3' },
                { node: 4, plane: 'Y', key: 'Sensor #4' },
            ],
            views: ['A', 'B'],
        } : null,
    };
}

// The sensors still without a signal, as the server counts them.
function pending() {
    const waiting = [];
    if (planes.measuring.length < 2) return ['none applied'];
    planes.measuring.forEach(node => ['X', 'Y'].forEach(plane => {
        if (!applied.has(node + ' ' + plane)) waiting.push(node + ' ' + plane);
    }));
    return waiting;
}

function step() {
    const sensors = {};
    applied.forEach(key => {
        const [node, plane] = key.split(' ');
        sensors[node] = sensors[node] || {};
        sensors[node][plane] = { probes_angle: 0, filename: 's.csv', rows: 3 };
    });
    return {
        nodes: ROTOR_NODES,
        measuring: planes.measuring,
        balancing: planes.balancing,
        sensors,
        pending: pending(),
        complete: pending().length === 0,
    };
}

const FRONTEND = new URL('../../extensions/digital_twin/frontend/', import.meta.url);
const { state } = await import('../../frontend/core/state.js');
const { addTexts, t } = await import('../../frontend/core/i18n.js');
const { switchScreen } = await import('../../frontend/features/screens.js');
const { TEXTS } = await import(new URL('texts.js', FRONTEND));
const { MEASUREMENT_ACTIONS, startMeasurement } = await import(new URL('features/measurement.js', FRONTEND));
const { startDigitalTwin } = await import(new URL('features/digital_twin.js', FRONTEND));

addTexts(TEXTS);

const settle = () => new Promise(done => setTimeout(done, 20));
const options = id => (document.getElementById(id).innerHTML.match(/value="(\d+)"/g) || [])
    .map(found => found.slice(7, -1));

function prepare() {
    clearDom();
    node('screen-dt');
    planes = { measuring: [], balancing: [] };
    ran = null;
    applied = new Set();
    parameters = null;
    asked.length = 0;
    state.rotorLibrary = [{
        name: 'Teste', savedAnalyses: [],
        materials: [], shafts: [], disks: [], gears: [], couplings: [],
        seals: [], bearings: [], pointmasses: [], probes: [],
    }];
    state.activeRotorIndex = -1;
    startMeasurement();
    startDigitalTwin();
}

// --- The pickers are filled by opening the screen, with no button pressed ----

prepare();
check('before the screen opens, the balancing panel is hidden',
      document.getElementById('dt-planes-form').style.display === 'none');

switchScreen('screen-dt');
await settle();

check('opening the screen builds the chosen rotor',
      asked.some(route => route.includes('/api/dt/user-model')));
check('the balancing panel is shown',
      document.getElementById('dt-planes-form').style.display === '');
check('the measuring picker offers every node of the rotor',
      options('dt-measuring-node').join(',') === '0,1,2,3,4,5,6');
check('the balancing picker offers every node of the rotor',
      options('dt-balancing-node').join(',') === '0,1,2,3,4,5,6');
check('with no plane chosen, the sensors card is not shown',
      document.getElementById('dt-sensors-card').style.display === 'none');

// --- The sensor cards appear in the wide panel, not in the 360px sidebar -----
// A sensor is offered for every plane chosen, from the first one: waiting for
// the second plane left the file inputs nowhere to be seen.

document.getElementById('dt-measuring-node').value = '2';
await MEASUREMENT_ACTIONS['dt-plane-add']({ dataset: { kind: 'measuring' } });
await settle();
check('the first measuring plane already offers its two sensors',
      document.getElementById('dt-sensors-card').style.display === ''
      && (document.getElementById('dt-sensors').innerHTML.match(/dt-sensor-\d+-[XY]-apply/g) || []).join(' ')
         === 'dt-sensor-2-X-apply dt-sensor-2-Y-apply');
check('with one plane the status asks for the second',
      document.getElementById('dt-sensors-status').textContent === t('dtNeedTwoPlanes'));

document.getElementById('dt-measuring-node').value = '4';
await MEASUREMENT_ACTIONS['dt-plane-add']({ dataset: { kind: 'measuring' } });
await settle();
check('two measuring planes keep the sensors card',
      document.getElementById('dt-sensors-card').style.display === '');
check('a card per plane and per sensor, in the wide panel',
      (document.getElementById('dt-sensors').innerHTML.match(/dt-sensor-\d+-[XY]-apply/g) || []).join(' ')
      === 'dt-sensor-2-X-apply dt-sensor-2-Y-apply dt-sensor-4-X-apply dt-sensor-4-Y-apply');
check('the Y sensor is 90 degrees after the X sensor and cannot be edited',
      document.getElementById('dt-sensors').innerHTML.includes('value="90" disabled'));
check('the chosen planes are listed',
      document.getElementById('dt-measuring-list').textContent.includes('2, 4'));

// --- A balancing plane carries the radius, and the analysis is the server's --

check('before the planes and the sensors, the button is off and says why',
      document.getElementById('dt-run').disabled === true
      && document.getElementById('dt-run-refusal').textContent !== '');

document.getElementById('dt-balancing-node').value = '3';
document.getElementById('dt-balancing-radius').value = '120';
await MEASUREMENT_ACTIONS['dt-plane-add']({ dataset: { kind: 'balancing' } });
await settle();
check('a balancing plane is sent as its node and its radius',
      JSON.stringify(planes.balancing) === '[{"node":3,"radius":120}]');
check('the balancing list shows the radius of each plane',
      document.getElementById('dt-balancing-list').textContent.includes('3 (120.0 mm)'));
check('with the planes chosen, the refusal moves on to the sensors',
      document.getElementById('dt-run-refusal').textContent.includes('sensor'));
check('no result before the analysis is run',
      document.getElementById('dt-balancing-card').style.display === 'none');

// --- The button follows the sensors, with no screen change in between -------
// WHY: applying the last sensor left the button disabled, because nothing
// asked the server again. The planes were chosen, every signal was in, and
// "Run analysis" could not be clicked until the screen was reopened.
check('with sensors missing, the analysis is still refused',
      document.getElementById('dt-run').disabled === true);

for (const node of [2, 4]) {
    for (const plane of ['X', 'Y']) {
        document.getElementById(`dt-sensor-${node}-${plane}-file`).files = [];
        await MEASUREMENT_ACTIONS['dt-sensor-apply']({ dataset: { node: String(node), plane } });
        await settle();
    }
}
check('applying the last sensor enables the analysis, with no reload',
      document.getElementById('dt-run').disabled === false
      && document.getElementById('dt-run-refusal').textContent === '');

// --- The machine's parameters, which the estimate is not the app's without --

check('without the machine parameters, the panel warns that the estimate differs',
      document.getElementById('dt-parameters-status').textContent.includes('will not match'));

document.getElementById('dt-parameters-file').files = [{
    name: 'input_parameters.json',
    text: async () => '{"indirect": {"P_factor": [[1, 9], [-10, 10]]}}',
}];
await MEASUREMENT_ACTIONS['dt-parameters-apply']({ dataset: {} });
await settle();
check('once applied, the panel names the file instead of warning',
      document.getElementById('dt-parameters-status').textContent.includes('input_parameters.json')
      && !document.getElementById('dt-parameters-status').textContent.includes('will not match'));

document.getElementById('dt-iterations').value = '2';
document.getElementById('dt-population').value = '3';
document.getElementById('dt-optimize').value = 'yes';
// The card appears right away with the spinner over it, and the result takes
// its place when the run ends. The fake DOM has no layout, so what is checked
// is the dimming the loader puts on the body it covers.
const running = MEASUREMENT_ACTIONS['dt-run-balancing']({ dataset: {} });
check('while the analysis runs, the card is shown and dimmed under the spinner',
      document.getElementById('dt-balancing-card').style.display === ''
      && document.getElementById('dt-balancing-body').style.opacity === '0.4');
await running;
await settle();
check('when the run ends, the dimming is lifted',
      document.getElementById('dt-balancing-body').style.opacity === '1');
check('the run carries the chosen iterations, population and optimization',
      ran && ran.iterations === 2 && ran.population === 3 && ran.optimize === true);
check('the correction mass and its phase are shown per plane',
      document.getElementById('dt-balancing-card').style.display === ''
      && document.getElementById('dt-balancing-result').innerHTML.includes('0.500')
      && document.getElementById('dt-balancing-result').innerHTML.includes('190.0'));
check('the summary says at which speed the signals were measured',
      document.getElementById('dt-balancing-summary').textContent.includes('840 rpm'));

// --- The polar charts: one per balancing plane, for the sensor chosen -------

check('the reference sensor picker offers every sensor of the measuring planes',
      document.getElementById('dt-polar-sensor').innerHTML.includes('Sensor #1')
      && document.getElementById('dt-polar-sensor').innerHTML.includes('Sensor #4'));
check('the plan view picker offers both views',
      (document.getElementById('dt-polar-view').innerHTML.match(/value="[AB]"/g) || []).length === 2);
// The rotor figure of step 3 is drawn through the same Plotly, so only the
// polar boxes are counted here.
const polarDrawn = () => drawn.filter(([id]) => id.startsWith('dt-polar-chart-'));
check('a polar chart is drawn for each balancing plane',
      polarDrawn().length === 1 && polarDrawn()[0][1] === 1);

// In a browser, filling a `<select>` selects its first option; the fake DOM
// does not, so the choice is made here as the user would.
document.getElementById('dt-polar-sensor').value = 'Sensor #2';
document.getElementById('dt-polar-view').value = 'B';
drawn.length = 0;
await MEASUREMENT_ACTIONS['dt-polar-change']({ dataset: {} });
await settle();
check('the chart of the chosen sensor and view is the one asked for',
      asked.polar.includes('sensor=Sensor%20%232') && asked.polar.includes('view=B'));
check('the phase in the table follows the chosen sensor',
      document.getElementById('dt-balancing-result').innerHTML.includes('280.0'));
check('the chart is drawn again for the new choice',
      polarDrawn().length === 1);

// --- Choosing another rotor builds it again, so the nodes follow it ----------

state.rotorLibrary.push({
    name: 'Outro', savedAnalyses: [],
    materials: [], shafts: [], disks: [], gears: [], couplings: [],
    seals: [], bearings: [], pointmasses: [], probes: [],
});
switchScreen('screen-dt');
await settle();
check('reopening the screen lists the rotors added meanwhile',
      options('dt-equipment').join(',') === '0,1');

shutDown();
