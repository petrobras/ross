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
globalThis.Event = class { constructor(type) { this.type = type; } };

const ROTOR_NODES = 7;

let planes = { measuring: [], balancing: [] };
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

function step() {
    const pending = [];
    planes.measuring.forEach(n => ['X', 'Y'].forEach(p => pending.push(n + ' ' + p)));
    return {
        nodes: ROTOR_NODES,
        measuring: planes.measuring,
        balancing: planes.balancing,
        sensors: {},
        pending,
        complete: false,
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
