// The coupled nodes of a MultiRotor follow its gears when a half is cut,
// meshed or added to (core/coupled_nodes.js, features/split.js `applyProject`).
//
// WHY THIS BATTERY EXISTS. `coupled_nodes` is typed once, when the MultiRotor
// is made, as two node numbers. Splitting the driving rotor's first shaft below
// its gear moved the gear from node 2 to node 3 and left "2, 1" behind: the
// MultiRotor then refused to build ("Each rotor needs a GearElement in the
// coupled nodes!"), on a screen where nobody had touched the coupling. The
// numbers below are that case: the `multirotor` golden of
// tests/golden/rotor_scenes.json, its first driving shaft cut at 50 mm by
// domain/splitting.py, which rewrites the gear's `n` from "2" to "3".
import fs from 'node:fs';
import { check, shutDown } from './fake_dom.js';

const { followCoupling, followedNode, parseCoupledNodes } = await import('../../frontend/core/coupled_nodes.js');
const { applyProject } = await import('../../frontend/features/split.js');
const { state } = await import('../../frontend/core/state.js');

const CASES = JSON.parse(fs.readFileSync(new URL('../golden/rotor_scenes.json', import.meta.url)));
const copy = value => JSON.parse(JSON.stringify(value));

// --- reading the field ----------------------------------------------------------

check('"2, 1" is the two nodes', JSON.stringify(parseCoupledNodes('2, 1')) === '[2,1]');
check('without the space too', JSON.stringify(parseCoupledNodes('2,1')) === '[2,1]');
check('one number is not a pair', parseCoupledNodes('2') === null);
check('three are not either', parseCoupledNodes('1, 2, 3') === null);
check('nor a word', parseCoupledNodes('two, 1') === null);
check('nor nothing', parseCoupledNodes(undefined) === null && parseCoupledNodes('') === null);

// --- the gear's node, followed -------------------------------------------------

check('the gear on node 2 is now on 3', followedNode([2], [3], 2) === 3);
check('the second gear of a half is followed, not the first', followedNode([0, 5], [0, 7], 5) === 7);
check('a node with no gear on it is left alone', followedNode([2], [3], 4) === 4);
check('as it is when the rows are not the same ones', followedNode([2], [3, 4], 2) === 2);

check('the driving node follows', followCoupling('2, 1', 0, [2], [3]) === '3, 1');
check('the driven node follows', followCoupling('2, 1', 1, [1], [4]) === '2, 4');
check('nothing moved: the text the person typed is kept, spelling and all',
    followCoupling('2 ,1', 0, [2], [2]) === '2 ,1');
check('a field that cannot be read is not rewritten', followCoupling('two, 1', 0, [2], [3]) === 'two, 1');
check('nor is it for a half that is not one of the two', followCoupling('2, 1', 2, [2], [3]) === '2, 1');

// --- where it happens: the answer of the server copied back --------------------

const golden = CASES.multirotor.project;
check('the golden is the case: gear on node 2 of the driving rotor, coupled "2, 1"',
    golden.multi_params.coupled_nodes === '2, 1'
    && golden.driving_rotor.gears.map(g => g.n).join() === '2');

function cutBelowTheGear(half) {
    // What domain/splitting.py answers for the first shaft cut at 50 mm: one
    // more shaft row, and every node above the cut one higher.
    const answered = copy(half);
    answered.shafts.splice(0, 1, Object.assign({}, half.shafts[0], { L: '50' }), Object.assign({}, half.shafts[0]));
    answered.gears = answered.gears.map(g => Object.assign({}, g, { n: String(Number(g.n) + 1) }));
    return answered;
}

state.projectData = copy(golden);
const driving = state.projectData.driving_rotor;
applyProject(driving, cutBelowTheGear(driving));
check('cut below its gear, the driving rotor couples on the gear\'s new node',
    state.projectData.multi_params.coupled_nodes === '3, 1');
check('and the half has the answer', driving.shafts.length === golden.driving_rotor.shafts.length + 1);

state.projectData = copy(golden);
const driven = state.projectData.driven_rotor;
applyProject(driven, cutBelowTheGear(driven));
check('the same cut on the driven rotor moves the second number',
    state.projectData.multi_params.coupled_nodes === '2, ' + (Number(golden.driven_rotor.gears[0].n) + 1));

state.projectData = copy(golden);
const cut = cutBelowTheGear(state.projectData.driving_rotor);
cut.gears = copy(golden.driving_rotor.gears);
applyProject(state.projectData.driving_rotor, cut);
check('a change that leaves the gear where it was leaves the field as typed',
    state.projectData.multi_params.coupled_nodes === '2, 1');

// A rotor that is not a MultiRotor, and a copy of a half that is not the open
// project's: nothing to follow, and nothing written.
state.projectData = copy(golden.driving_rotor);
applyProject(state.projectData, cutBelowTheGear(state.projectData));
check('a single rotor has no coupling to write', state.projectData.multi_params === undefined);

state.projectData = copy(golden);
const stray = copy(golden.driving_rotor);
applyProject(stray, cutBelowTheGear(stray));
check('a half that is not the open MultiRotor\'s does not touch its coupling',
    state.projectData.multi_params.coupled_nodes === '2, 1');

shutDown();
