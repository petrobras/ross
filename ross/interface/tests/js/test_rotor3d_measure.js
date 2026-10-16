// The measuring tape of the 3D view (core/rotor3d_measure.js,
// components/rotor3d_measure.js).
//
// WHY THIS BATTERY EXISTS. A distance on screen is a number someone will
// write down. The claims are the ones a wrong one would break silently: the
// pointer moves an end to the point of the axis it is over, from any angle;
// an end stops at a node unless Shift is held, and never leaves the shaft
// line; the reading is the distance between the ends and names their nodes
// only when both are on one; the tape clears every part drawn and stays under
// the dimensions; an end on a node stays on it when the rotor is rebuilt; and
// the handles are where the ends are.
import fs from 'node:fs';
import { check, shutDown } from './fake_dom.js';

const THREE = await import('../../frontend/vendor/three/three.module.js');
const { annotationLayout, layoutScene } = await import('../../frontend/core/rotor3d_layout.js');
const M = await import('../../frontend/core/rotor3d_measure.js');
const { buildMeasure } = await import('../../frontend/components/rotor3d_measure.js');

const CASES = JSON.parse(fs.readFileSync(new URL('../golden/rotor_scenes.json', import.meta.url)));
const near = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));

// --- the line it runs on, on every scene ROSS built ------------------------------------
console.log('\nThe line, on every scene ROSS built');

for (const [name, { scene }] of Object.entries(CASES)) {
    const layout = layoutScene(scene);
    const plan = annotationLayout(layout);
    const halves = scene.kind === 'multirotor' ? ['driving', 'driven'] : [null];
    const lines = halves.map(half => M.measureLine(layout, plan, half));
    lines.forEach((line, k) => {
        const label = `${name}${halves[k] ? ' ' + halves[k] : ''}`;
        const dims = plan.lines.find(l => l.half === halves[k]);
        check(`${label}: a line to measure`, line !== null);
        check(`${label}: from the first node to the last`, near(line.z0, dims.total.z0) && near(line.z1, dims.total.z1));
        check(`${label}: the tape clears every part drawn, and stays under the dimensions`,
            line.tape > dims.total.y - 2 * dims.gap && M.measureTop(line) < dims.total.y - dims.gap);
        const [a, b] = M.startingEnds(line);
        check(`${label}: it starts from end to end, and reads the total length`,
            near(M.measured([a, b]).distance, dims.total.length) && a.node === line.rings[0].n && b.node === line.rings[line.rings.length - 1].n);
    });
    if (lines.length === 2) {
        check(`${name}: each line of a MultiRotor has its own tape`, lines[0].x !== lines[1].x || lines[0].y !== lines[1].y);
    }
}
check('a line that is not there has no tape',
    M.measureLine(layoutScene(CASES.coupling.scene), annotationLayout(layoutScene(CASES.coupling.scene)), 'driving') === null);

// --- the pointer, from any angle ------------------------------------------------------
console.log('\nThe pointer');

const layout = layoutScene(CASES.compressor_example.scene);
const line = M.measureLine(layout, annotationLayout(layout), null);
const zAt = k => line.rings[k].z;
const middle = (zAt(10) + zAt(11)) / 2;

check('from straight above, the end goes under the pointer',
    near(M.alongAxis([line.x, line.tape + 1, middle], [0, -1, 0], line), middle));
const eye = [line.x + 1.3, line.tape + 0.7, middle - 0.4];
const toward = [line.x - eye[0], line.tape - eye[1], middle - eye[2]];
check('from an angle, to the point of the axis the ray passes through', near(M.alongAxis(eye, toward, line), middle));
const passing = [toward[0], toward[1] + 0.05, toward[2]];
const z = M.alongAxis(eye, passing, line);
check('a ray that passes beside it, to the nearest point of the axis', Math.abs(z - middle) < 0.02 && z !== middle);
check('looking down the axis, no point is nearer than another', M.alongAxis([line.x, line.tape, -1], [0, 0, 1], line) === null);

// --- where an end stops -----------------------------------------------------------------
console.log('\nWhere an end stops');

const nearer = zAt(11) - 0.2 * (zAt(11) - zAt(10));
check('at the nearest node', M.placeEnd(nearer, line, false).z === zAt(11) && M.placeEnd(nearer, line, false).node === line.rings[11].n);
check('anywhere with Shift, and then it names no node', near(M.placeEnd(nearer, line, true).z, nearer) && M.placeEnd(nearer, line, true).node === null);
check('with Shift, exactly on a node, it names that node', M.placeEnd(zAt(7), line, true).node === line.rings[7].n);
check('never past the first node', M.placeEnd(line.z0 - 1, line, true).z === line.z0 && M.placeEnd(line.z0 - 1, line, false).node === line.rings[0].n);
check('nor past the last', M.placeEnd(line.z1 + 1, line, true).z === line.z1);

// --- what it reads ----------------------------------------------------------------------
console.log('\nThe reading');

const onNodes = [{ z: zAt(9), node: line.rings[9].n }, { z: zAt(3), node: line.rings[3].n }];
const reading = M.measured(onNodes);
check('the distance between the ends, whichever is on the left', near(reading.distance, zAt(9) - zAt(3)));
check('in millimetres', reading.text === `${((zAt(9) - zAt(3)) * 1000).toFixed(1)} mm`);
check('and the two nodes, lowest first', JSON.stringify(reading.nodes) === JSON.stringify([line.rings[3].n, line.rings[9].n]));
check('no nodes when an end is between two', M.measured([onNodes[0], { z: nearer, node: null }]).nodes === null);

// --- from a node, and through a rebuild ------------------------------------------------
console.log('\nFrom a node, and through a rebuild');

const fromSeven = M.startingEnds(line, line.rings[7].n);
check('from a node, to the last one', fromSeven[0].node === line.rings[7].n && fromSeven[1].node === line.rings[line.rings.length - 1].n);
const fromLast = M.startingEnds(line, line.rings[line.rings.length - 1].n);
check('from the last node, back to the first', fromLast[1].node === line.rings[0].n);
check('from a node the line does not have, end to end', M.startingEnds(line, 9999)[0].node === line.rings[0].n);

// The rotor rebuilt with every node 10 mm further along: an end on a node
// follows it, an end between nodes keeps its place.
const moved = { ...line, rings: line.rings.map(r => ({ ...r, z: r.z + 0.01 })), z0: line.z0 + 0.01, z1: line.z1 + 0.01 };
const kept = M.keptEnds([onNodes[0], { z: nearer, node: null }], moved);
check('an end on a node stays on it', near(kept[0].z, zAt(9) + 0.01) && kept[0].node === line.rings[9].n);
check('an end between nodes keeps its place', near(kept[1].z, nearer) && kept[1].node === null);
const fewer = { ...line, rings: line.rings.slice(0, 5), z1: line.rings[4].z };
check('an end whose node is gone stays inside the line', M.keptEnds([onNodes[0]], fewer)[0].z === fewer.z1);

// --- what is drawn -----------------------------------------------------------------------
console.log('\nWhat is drawn');

const points = M.measureSegments(line, onNodes);
check('the tape runs between the ends, at its height',
    near(points[0][2], zAt(3)) && near(points[1][2], zAt(9)) && points[0][1] === line.tape && points[1][1] === line.tape);
check('a line down from each end to the shaft',
    near(points[3][1], M.surfaceAt(line, zAt(9))) && near(points[5][1], M.surfaceAt(line, zAt(3))));
check('down to the shaft, not into it', M.surfaceAt(line, zAt(9)) === line.y + line.rings[9].radius);
check('a tick at every node between the ends', (points.length - 6) / 2 === 5);

const measure = buildMeasure(THREE, line, onNodes, 'steelblue');
const handles = measure.object.children.filter(c => c.isMesh);
check('two handles, on the ends', handles.length === 2 && near(handles[0].position.z, zAt(9)) && near(handles[1].position.z, zAt(3)));
check('drawn over the parts, so an end can always be grabbed', handles.every(h => h.material.depthTest === false));
measure.update(line, [onNodes[0], { z: nearer, node: null }]);
check('dragged, a handle follows its end', near(handles[1].position.z, nearer));
measure.restyle('firebrick');
check('it takes the theme', handles[0].material.color.getHexString() === new THREE.Color('firebrick').getHexString());
measure.dispose();

// --- which end the pointer is on ---------------------------------------------------------
console.log('\nGrabbing an end');

const screen = [[100, 50, 0.5], [300, 50, 0.5]];
check('the end under the pointer', M.grabbedEnd({ x: 106, y: 54 }, screen) === 0);
check('the nearer of two', M.grabbedEnd({ x: 296, y: 50 }, [[290, 50, 0.5], [298, 50, 0.5]]) === 1);
check('none when the pointer is away from both', M.grabbedEnd({ x: 200, y: 50 }, screen) === -1);
check('none behind the camera', M.grabbedEnd({ x: 100, y: 50 }, [[100, 50, 1.2]]) === -1);

shutDown();
