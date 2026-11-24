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
// The analysis parameters as the server answers them: one field per value the
// estimate uses, every one empty, with the rossml default it falls back to.
const PARAMETER_FIELDS = [
    { key: 'number_points', group: 'signal', default: 4096, whole: true },
    { key: 'w_f0', group: 'kalman', default: 0.65, whole: false },
    { key: 'direct_P', group: 'direct', default: null, whole: false },
    { key: 'indirect_P_mantissa_min', group: 'indirect', default: 1, whole: true },
];
const PARAMETER_VALUES = { number_points: null, w_f0: null, direct_P: null, indirect_P_mantissa_min: null };
const startingParameters = () => ({ values: { ...PARAMETER_VALUES }, fields: PARAMETER_FIELDS });
let parameters = startingParameters();
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
            if (route.includes('/api/dt/balancing-figure')) {
                asked.figure = body;
                // The polar chart comes one per balancing plane, every other
                // chart one for the plane asked.
                const nodes = body.chart === 'polar' ? planes.balancing.map(plane => plane.node)
                    : [body.node === undefined ? null : body.node];
                return {
                    chart: body.chart,
                    charts: nodes.map(node => ({
                        node,
                        figure: { data: [{ type: 'scatter', x: [0], y: [1] }], layout: {} },
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
                    nan_warning: false, settings: null,
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
                { node: 2, plane: 'X', key: 'Sensor #1', index: 0 },
                { node: 2, plane: 'Y', key: 'Sensor #2', index: 1 },
                { node: 4, plane: 'X', key: 'Sensor #3', index: 2 },
                { node: 4, plane: 'Y', key: 'Sensor #4', index: 3 },
            ],
            vs_nodes: [3, 2, 4],
            charts: ['polar', 'decomposed', 'forces_1d', 'forces_2d', 'vs_1d', 'vs_2d', 'vs_3d'],
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
    parameters = startingParameters();
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

// --- The analysis parameters: empty fields, without a file or a machine -----

const fieldsBox = () => document.getElementById('dt-parameters-fields').innerHTML;
check('each field is drawn in its group, empty at the rossml defaults',
      fieldsBox().includes('Number of points') && fieldsBox().includes('Filter factors (direct)')
      && fieldsBox().includes('id="dt-param-w_f0"') && !fieldsBox().includes('value="0.65"')
      && !fieldsBox().includes('value="4096"'));
check('the rossml default is shown in the empty box, so it can be changed',
      /id="dt-param-number_points"[^>]*placeholder="4096"/.test(fieldsBox())
      && /id="dt-param-w_f0"[^>]*placeholder="0.65"/.test(fieldsBox())
      && /id="dt-param-indirect_P_mantissa_min"[^>]*placeholder="1"/.test(fieldsBox()));
check('a factor rossml searches for says so instead of showing a default',
      /id="dt-param-direct_P"[^>]*placeholder="searched"/.test(fieldsBox()));
check('no machine-parameters picker is wired up any more',
      MEASUREMENT_ACTIONS['dt-parameters-preset'] === undefined);
check('a whole-number field steps by one, the others take any value',
      /id="dt-param-number_points" type="number" step="1"/.test(fieldsBox())
      && /id="dt-param-w_f0" type="number" step="any"/.test(fieldsBox()));
check('no file is asked for any more',
      !asked.some(route => route.includes('/api/dt/balancing-parameters')));

// The fake DOM keeps an element's value apart from the HTML it was drawn
// with, so the values the run reads are set on the elements themselves.
node('dt-param-number_points').value = '2048';
node('dt-param-w_f0').value = '0.4';
node('dt-param-direct_P').value = '';
node('dt-param-indirect_P_mantissa_min').value = ' ';

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
check('the run sends every field, an empty one as null',
      ran.preset === undefined && ran.parameters.number_points === 2048
      && ran.parameters.w_f0 === 0.4
      && ran.parameters.direct_P === null
      && ran.parameters.indirect_P_mantissa_min === null);
check('the correction mass and its phase are shown per plane',
      document.getElementById('dt-balancing-card').style.display === ''
      && document.getElementById('dt-balancing-result').innerHTML.includes('0.500')
      && document.getElementById('dt-balancing-result').innerHTML.includes('190.0'));
check('the summary says at which speed the signals were measured',
      document.getElementById('dt-balancing-summary').textContent.includes('840 rpm'));

// --- The charts of the run: polar first, then the others on request ---------

check('the reference sensor picker offers every sensor of the measuring planes',
      document.getElementById('dt-polar-sensor').innerHTML.includes('Sensor #1')
      && document.getElementById('dt-polar-sensor').innerHTML.includes('Sensor #4'));
check('the chart picker offers the polar diagram, the forces and the virtual sensing',
      ['polar', 'decomposed', 'forces_1d', 'forces_2d', 'vs_1d', 'vs_2d', 'vs_3d'].every(kind =>
          document.getElementById('dt-chart-kind').innerHTML.includes(`value="${kind}"`)));
// The rotor figure of step 3 is drawn through the same Plotly, so only the
// boxes of the result card are counted here.
const chartsDrawn = () => drawn.filter(([id]) => id.startsWith('dt-chart-'));
check('the polar diagram is drawn first, one chart per balancing plane',
      asked.figure && asked.figure.chart === 'polar' && chartsDrawn().length === 1);
check('the polar diagram needs no plane, so the plane picker is hidden',
      document.getElementById('dt-chart-node-group').style.display === 'none');
check('the reference sensor picker is shown with the polar diagram',
      document.getElementById('dt-polar-sensor-group').style.display === '');

// In a browser, filling a `<select>` selects its first option; the fake DOM
// does not, so the choice is made here as the user would.
document.getElementById('dt-polar-sensor').value = 'Sensor #2';
drawn.length = 0;
await MEASUREMENT_ACTIONS['dt-polar-change']({ dataset: {} });
await settle();
check('the polar diagram of the chosen sensor is the one asked for',
      asked.figure.chart === 'polar' && asked.figure.sensor === 1);
check('the phase in the table follows the chosen sensor',
      document.getElementById('dt-balancing-result').innerHTML.includes('280.0'));
check('the chart is drawn again for the new choice',
      chartsDrawn().length === 1);

// The virtual sensing is offered on the balancing planes and the measuring
// planes.
document.getElementById('dt-chart-kind').value = 'vs_2d';
drawn.length = 0;
await MEASUREMENT_ACTIONS['dt-chart-change']({ dataset: {} });
await settle();
check('the virtual sensing offers the balancing and the measuring planes',
      options('dt-chart-node').join(',') === '3,2,4'
      && document.getElementById('dt-chart-node-group').style.display === '');
check('the reference sensor picker is hidden for any chart but the polar diagram',
      document.getElementById('dt-polar-sensor-group').style.display === 'none');
document.getElementById('dt-chart-node').value = '4';
await MEASUREMENT_ACTIONS['dt-chart-change']({ dataset: {} });
await settle();
check('the virtual sensing of the chosen plane is the one asked for',
      asked.figure.chart === 'vs_2d' && asked.figure.node === 4 && chartsDrawn().length >= 1);

document.getElementById('dt-chart-kind').value = 'forces_1d';
await MEASUREMENT_ACTIONS['dt-chart-change']({ dataset: {} });
await settle();
check('the estimated forces are offered on the balancing planes only',
      options('dt-chart-node').join(',') === '3' && asked.figure.chart === 'forces_1d');

// The decomposition waits for its positions: a chart drawn before the angles
// and radii are typed would be one the user did not ask for.
document.getElementById('dt-chart-kind').value = 'decomposed';
const before = asked.figure;
await MEASUREMENT_ACTIONS['dt-chart-change']({ dataset: {} });
await settle();
check('choosing the decomposition shows its positions and draws nothing yet',
      document.getElementById('dt-decomposition').style.display === ''
      && asked.figure === before);
check('the positions start at the radius of the balancing plane',
      document.getElementById('dt-decomp-radius-a').value === '120'
      && document.getElementById('dt-decomp-radius-b').value === '120');
document.getElementById('dt-decomp-angle-a').value = '30';
document.getElementById('dt-decomp-angle-b').value = '120';
await MEASUREMENT_ACTIONS['dt-chart-draw']({ dataset: {} });
await settle();
check('splitting sends the angles and radii of both positions',
      asked.figure.chart === 'decomposed'
      && JSON.stringify(asked.figure.positions) === '[30,120]'
      && JSON.stringify(asked.figure.radii) === '[120,120]'
      && asked.figure.node === 3 && asked.figure.sensor === 1);

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
