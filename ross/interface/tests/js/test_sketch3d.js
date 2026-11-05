// The sketch and the extrusion of a shaft element in the 3D view
// (core/sketch3d.js, components/sketch3d_parts.js, features/sketch3d.js).
//
// WHY THIS BATTERY EXISTS. The sketch turns pointer positions into the numbers
// of a shaft row, and every step can be wrong without looking wrong:
//
//   * the diameter is twice the distance from the axis, in millimetres -- off
//     by two or by a thousand, the circle still looks like a circle;
//   * of two circles the larger is the outside, whatever order they came in;
//   * the length is measured from the face in the sense the element grows;
//   * the row has to be the one the form would have written: millimetres, no
//     `_unit`, the material by name;
//   * what is drawn has the size of those numbers, and is freed after.
import { check, node, shutDown } from './fake_dom.js';

const THREE = await import('../../frontend/vendor/three/three.module.js');
const S = await import('../../frontend/core/sketch3d.js');
const { buildCircle, buildExtrusion, buildSketchPlane } = await import('../../frontend/components/sketch3d_parts.js');
const { SKETCH_ACTIONS } = await import('../../frontend/features/sketch3d.js');
const { addShaftFromSketch } = await import('../../frontend/features/modeling.js');
const { openProjectHistory, state } = await import('../../frontend/core/state.js');
const { canUndo } = await import('../../frontend/core/history.js');

const near = (a, b, tolerance = 1e-9) => Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(b));

// --- rounding ------------------------------------------------------------------------
console.log('\nRounding what the pointer gives');

check('47.3 mm moves by 1 mm', S.niceStep(47.3) === 1);
check('4.73 mm by 0.1 mm', near(S.niceStep(4.73), 0.1));
check('473 mm by 10 mm', S.niceStep(473) === 10);
check('nothing sensible: 0.1', S.niceStep(0) === 0.1 && S.niceStep(-3) === 0.1 && S.niceStep(NaN) === 0.1);
check('no float noise in a rounded value', S.rounded(0.30000000000000004, 0.1) === 0.3 && S.rounded(47.6, 1) === 48);
check('a value written as the form writes it', S.shown(100) === '100' && S.shown(0.1 + 0.2) === '0.3' && S.shown(12.5) === '12.5');
check('a typed length: a decimal comma reads as a point', S.typedLength('12,5') === 12.5 && S.typedLength(' 40 ') === 40);
check('and nothing else is one', [null, undefined, '', 'abc', '0', '-5'].every(v => S.typedLength(v) === null));

// --- the circles ------------------------------------------------------------------------
console.log('\nThe circles');

// The plane point under the pointer, in metres; the diameter in millimetres.
check('the diameter is twice the distance from the axis, in mm', S.pointerDiameter(0.05, 0) === 100);
check('in any direction around it', S.pointerDiameter(0.03, 0.04) === 100);
check('rounded to the step of its size', S.pointerDiameter(0.02371, 0) === 47 && S.pointerDiameter(0.0023, 0) === 4.6);
check('never zero: the smallest step', S.pointerDiameter(0, 0) > 0);

check('no circle, no section', S.sectionOf([]) === null && S.sectionOf(null) === null);
const solid = S.sectionOf([100]);
check('one circle: a solid shaft', solid.odl === 100 && solid.idl === 0);
check('two: the larger outside, the smaller the bore', JSON.stringify(S.sectionOf([40, 100])) === JSON.stringify({ odl: 100, idl: 40 })
    && JSON.stringify(S.sectionOf([100, 40])) === JSON.stringify({ odl: 100, idl: 40 }));

check('a diameter that is no number is refused', S.circleProblem(null, []) === 'sketchNeedsDiameter' && S.circleProblem(0, []) === 'sketchNeedsDiameter');
check('a third circle is refused', S.circleProblem(30, [100, 40]) === 'sketchTwoCircles');
check('so is a second one of the same diameter', S.circleProblem(100, [100]) === 'sketchSameCircle');
check('a good one is not', S.circleProblem(40, [100]) === null);
check('a circle retyped is not compared with itself', S.circleProblem(100, [100, 40], 0) === null
    && S.circleProblem(40, [100, 40], 0) === 'sketchSameCircle');
check('nor refused for being the second of two', S.circleProblem(60, [100, 40], 1) === null);

// --- the extrusion ----------------------------------------------------------------------
console.log('\nThe extrusion');

check('dragged 0.25 m past a face at z = 0, growing +z: 250 mm', S.draggedLength(0.25, 0, 1) === 250);
check('from a face at z = 0.5 growing -z, the same pull the other way', S.draggedLength(0.25, 0.5, -1) === 250);
check('rounded to its step, unless free', S.draggedLength(0.2537, 0, 1) === 250 && S.draggedLength(0.2537, 0, 1, true) === 253.7);
check('dragged behind the face: one step, never nothing', S.draggedLength(-0.1, 0, 1) > 0);
check('it starts as long as it is wide (L/D 1, the most API RP 684 takes)', S.startingLength({ odl: 100, idl: 0 }) === 100);

const ratio = S.slenderness(150, { odl: 100, idl: 0 });
check('L/D', near(ratio.ratio, 1.5) && ratio.tooLong && ratio.overPreferred);
check('at L/D 1 it is not too long, but past the preferred 0.5', !S.slenderness(100, { odl: 100 }).tooLong && S.slenderness(100, { odl: 100 }).overPreferred);
check('at 0.5, neither', !S.slenderness(50, { odl: 100 }).tooLong && !S.slenderness(50, { odl: 100 }).overPreferred);

const row = S.shaftRow(250, { odl: 100, idl: 40 }, 'Steel');
check('the row the form would write: mm, no unit, the material by name',
    JSON.stringify(row) === JSON.stringify({ element_type: 'BASIC', L: '250', idl: '40', odl: '100', material: 'Steel' }));
check('with no material, the default steel', S.shaftRow(250, { odl: 100, idl: 0 }).material === 'Default (Steel)');
check('a typed 12,5 mm length is 12.5 in the row', S.shaftRow(S.typedLength('12,5'), { odl: 10, idl: 0 }).L === '12.5');

const firstOne = S.firstFace();
check('the first shaft starts on z = 0 and grows +z, sketched from the front view', firstOne.z === 0 && firstOne.sense === 1
    && firstOne.diameter === 0 && JSON.stringify(firstOne.look) === JSON.stringify([0, 0, -1]));
check('the sketch spans three radii of the face, 150 mm with none', near(S.sketchExtent(100), 0.15) && near(S.sketchExtent(0), 0.15));
check('a grid of about thirty round steps', [0.15, 0.03, 0.9].every(e => {
    const step = S.gridSpacing(e) * 1000;
    const lines = 2 * e * 1000 / step;
    const leading = step / Math.pow(10, Math.floor(Math.log10(step)));
    return lines > 14 && lines <= 31 && [1, 2, 5].some(k => near(leading, k));
}));
const box = S.extrusionBounds({ z: 0.5, sense: -1 }, 200, { odl: 100, idl: 0 });
check('the box of the extrusion, growing -z from z = 0.5', near(box.min[2], 0.3) && near(box.max[2], 0.5) && near(box.max[0], 0.05));

// --- the faces of a shaft ----------------------------------------------------------------
console.log('\nThe faces a sketch can be drawn on');

// Two shafts of a line, 0..0.25 and 0.25..0.5, the second tapered 60 -> 40 mm
// with a 20 mm bore, and a coupling 0.5..0.6 after it.
const first = { kind: 'shaft', half: null, index: 0, z0: 0, z1: 0.25, entry: { n: 0 },
    profile: { odl: 0.05, odr: 0.05, idl: 0, idr: 0 } };
const second = { kind: 'shaft', half: null, index: 1, z0: 0.25, z1: 0.5, entry: { n: 1 },
    profile: { odl: 0.06, odr: 0.04, idl: 0.02, idr: 0.02 } };
const coupling = { kind: 'coupling', half: null, index: 0, z0: 0.5, z1: 0.6 };
const disk = { kind: 'disk', half: null, index: 0, z0: 0.49, z1: 0.51 };
const line = [first, second, coupling, disk];
const startOfFirst = S.shaftFace(first, 'start', line);
check('the start face of the first shaft: z = 0, growing -z, node 0, free',
    startOfFirst.z === 0 && startOfFirst.sense === -1 && startOfFirst.node === 0 && startOfFirst.free && startOfFirst.side === 'start');
check('seen from outside it, the side it grows into', JSON.stringify(startOfFirst.look) === JSON.stringify([0, 0, -1]));
const endOfFirst = S.shaftFace(first, 'end', line);
check('its end face: z = 0.25, growing +z, node 1 -- the next shaft goes on from it',
    endOfFirst.z === 0.25 && endOfFirst.sense === 1 && endOfFirst.node === 1 && !endOfFirst.free);
const startOfSecond = S.shaftFace(second, 'start', line);
check('the start face of the second: the diameters of its left end, in mm',
    near(startOfSecond.diameter, 60) && near(startOfSecond.inner, 20) && !startOfSecond.free && startOfSecond.index === 1);
const endOfSecond = S.shaftFace(second, 'end', line);
check('its end face: those of its right end', near(endOfSecond.diameter, 40) && endOfSecond.node === 2);
check('a coupling going on from it is the line going on; a disk is not', !endOfSecond.free
    && S.shaftFace(second, 'end', [first, second, disk]).free);
check('the other half of a MultiRotor is another line', S.shaftFace(second, 'end', [first, second, { ...coupling, half: 'driven' }]).free);
check('not the first shaft of a project', !endOfSecond.first && S.firstFace().first);

// --- what is drawn ----------------------------------------------------------------------
console.log('\nWhat is drawn');

const colours = { accent: 'steelblue', strong: 'black', grid: 'gray', x: 'firebrick', y: 'seagreen', arrow: 'steelblue', arrowHover: 'black' };
const geometryOf = object => {
    const box3 = new THREE.Box3().setFromObject(object);
    return { min: box3.min.toArray(), max: box3.max.toArray() };
};

const plane = buildSketchPlane(THREE, { z: 0.2, extent: 0.15, spacing: S.gridSpacing(0.15), colours });
const planeBox = geometryOf(plane.object);
check('the plane lies at its z', near(planeBox.min[2], 0.2) && near(planeBox.max[2], 0.2));
// Float32 positions: a millionth is the precision of what WebGL is given.
check('and reaches the extent on each side', near(planeBox.max[0], plane.edge, 1e-6) && plane.edge >= 0.15 - 1e-9 && near(planeBox.min[1], -plane.edge, 1e-6));
let overlay = true;
plane.object.traverse(o => { if (o.material && o.material.depthTest !== false) overlay = false; });
check('drawn over what is there: no flicker on the face it lies on', overlay);

const circle = buildCircle(THREE, 0.2, 'black');
circle.setDiameter(100);
const circleBox = geometryOf(circle.object);
check('a circle of 100 mm is 0.05 m from the axis', near(circleBox.max[0], 0.05, 1e-6) && near(circleBox.min[1], -0.05, 1e-6));
check('on the plane', near(circleBox.min[2], 0.2) && near(circleBox.max[2], 0.2));

const extrusion = buildExtrusion(THREE, colours);
extrusion.set({ z: 0.1, sense: 1 }, 250, { odl: 100, idl: 40 });
const body = extrusion.object.children[0];
body.geometry.computeBoundingBox();
const bodyBox = body.geometry.boundingBox;
check('the element from the face, 250 mm along +z', near(bodyBox.min.z + 0.1, 0.1, 1e-6) && near(bodyBox.max.z + 0.1, 0.35, 1e-6));
check('as wide as its outside', near(bodyBox.max.x, 0.05, 1e-3));
// The bore: no vertex of the body is closer to the axis than its radius.
const positions = body.geometry.attributes.position.array;
let innermost = Infinity;
for (let i = 0; i < positions.length; i += 3) innermost = Math.min(innermost, Math.hypot(positions[i], positions[i + 1]));
check('hollow with its bore', near(innermost, 0.02, 1e-6));
extrusion.object.updateMatrixWorld(true);
const tip = new THREE.Vector3();
extrusion.arrow.getWorldPosition(tip);
check('the arrow on the far face, on the axis', near(tip.z, 0.35, 1e-9) && near(tip.x, 0) && near(tip.y, 0));
const pointing = new THREE.Vector3(0, 1, 0).applyQuaternion(extrusion.arrow.quaternion);
check('pointing the way the element grows', near(pointing.z, 1, 1e-9));
extrusion.set({ z: 0.5, sense: -1 }, 200, { odl: 100, idl: 0 });
body.geometry.computeBoundingBox();
check('growing -z, the other way from its face', near(body.geometry.boundingBox.min.z + 0.5, 0.3, 1e-6)
    && near(body.geometry.boundingBox.max.z + 0.5, 0.5, 1e-6));
const back = new THREE.Vector3(0, 1, 0).applyQuaternion(extrusion.arrow.quaternion);
check('and the arrow with it', near(back.z, -1, 1e-9));

// Everything made is freed.
const freed = new Set();
const watch = object => object.traverse(o => {
    if (o.geometry) o.geometry.addEventListener('dispose', () => freed.add(o.geometry));
});
const all = [plane, circle, extrusion];
let made = 0;
all.forEach(item => item.object.traverse(o => { if (o.geometry) made += 1; }));
all.forEach(item => watch(item.object));
all.forEach(item => item.dispose());
check('every geometry made is freed', freed.size === made && made > 0);

// --- into the list -----------------------------------------------------------------------
console.log('\nThe row joins the list');

state.rotorLibrary = [{ name: 'R', uid: 'uid_r', savedAnalyses: [], materials: [], shafts: [],
    disks: [], gears: [], couplings: [], seals: [], bearings: [], pointmasses: [] }];
state.activeRotorIndex = 0;
state.projectData = state.rotorLibrary[0];
state.currentTab = 'shafts';
openProjectHistory(state.projectData);
node('element-list');
addShaftFromSketch(S.shaftRow(250, { odl: 100, idl: 40 }, 'Default (Steel)'));
check('the sketched shaft is the first of the list', state.projectData.shafts.length === 1 && state.projectData.shafts[0].L === '250');
check('and can be undone, like a shaft from the form', canUndo());

// --- the buttons, with no sketch open ----------------------------------------------------
console.log('\nThe buttons with nothing to act on');

let failed = '';
for (const [name, action] of Object.entries(SKETCH_ACTIONS)) {
    if (name === 'sketch-first-shaft') continue;   // needs WebGL; the actions battery presses it
    try {
        const answer = action({ dataset: { index: '0' }, value: 'Steel' }, {});
        if (answer && typeof answer.then === 'function') await answer;
    } catch (error) {
        failed += `${name}: ${error.message}; `;
    }
}
check('none fails with no sketch open: ' + failed, failed === '');

shutDown();
