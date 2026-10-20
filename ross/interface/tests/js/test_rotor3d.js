// The 3D view's two halves that run without a browser: where every part goes
// (core/rotor3d_layout.js) and the solids built for it
// (components/rotor3d_parts.js), with the real three.js from vendor/three.
//
// WHY THIS BATTERY EXISTS. The scenes below are ROSS's, not made up here:
// tests/golden/rotor_scenes.json holds the projects and the scenes the server
// answered for them, and tests/test_rotor_scene.py keeps the file equal to
// what ROSS builds today. Every claim is about agreement -- a part drawn where
// the pointer finds it, a camera that frames what is drawn, a rebuild that
// frees what it replaces -- because a view that agrees with itself and not
// with the rotor is the prototype's defect over again.
import fs from 'node:fs';
import { check, shutDown } from './fake_dom.js';

const THREE = await import('../../frontend/vendor/three/three.module.js');
const { PROBE_COLOR, SYMBOL, benchLayout, drivenPlacement, framing, layoutScene, pickPart, shaftRadiusAt, viewDirection } =
    await import('../../frontend/core/rotor3d_layout.js');
const { MAX_DRAWN_TEETH, buildBench, buildRotorModel, mergeColoured } =
    await import('../../frontend/components/rotor3d_parts.js');
const { motorPieces } = await import('../../frontend/components/rotor3d_motor.js');

const CASES = JSON.parse(fs.readFileSync(new URL('../golden/rotor_scenes.json', import.meta.url)));
const near = (a, b, tolerance = 1e-9) => Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(a), Math.abs(b));
const copy = value => JSON.parse(JSON.stringify(value));

// --- every element becomes exactly one part -------------------------------------
console.log('\nOne part per element, at the element\'s place');

const CATEGORIES = ['shafts', 'couplings', 'disks', 'gears', 'bearings', 'seals', 'pointmasses', 'probes'];
for (const [name, { scene }] of Object.entries(CASES)) {
    const lines = scene.kind === 'multirotor' ? [['driving', scene.driving], ['driven', scene.driven]] : [[null, scene]];
    const { parts } = layoutScene(scene);
    const expected = lines.reduce((sum, [, line]) => sum + CATEGORIES.reduce((n, c) => n + (line[c] || []).length, 0), 0);
    check(`${name}: ${expected} elements, ${parts.length} parts`, parts.length === expected);
    const keys = new Set(parts.map(p => p.key));
    check(`${name}: no two parts share a key`, keys.size === parts.length);
    const everyRowOnce = lines.every(([half, line]) => CATEGORIES.every(c => (line[c] || []).every(e =>
        parts.filter(p => p.half === half && p.category === c && p.index === e.index && p.entry === e).length === 1)));
    check(`${name}: each part is its own row, by category and index`, everyRowOnce);
}

const compressor = CASES.compressor_example.scene;
const compressorParts = layoutScene(compressor).parts;
const shaftParts = compressorParts.filter(p => p.kind === 'shaft');
check('a shaft part spans the nodes the server put it between',
    shaftParts.every(p => p.z0 === p.entry.z0 && p.z1 === p.entry.z1));
check('and is as thick as its larger end',
    shaftParts.every(p => near(p.radius, Math.max(p.entry.odl, p.entry.odr) / 2)));

// --- sizes -------------------------------------------------------------------------
console.log('\nSizes: the scene\'s when it has one, the shaft\'s when it has none');

const every = CASES.every_element.scene;
const everyParts = layoutScene(every).parts;
const disksDrawn = everyParts.filter(p => p.kind === 'disk');
check('a disk from its dimensions is drawn at its outer radius',
    disksDrawn.some(p => p.entry.shape.exact && near(p.radius, p.entry.shape.outer_radius)));
check('an equivalent disk at the equivalent radius and width',
    disksDrawn.some(p => !p.entry.shape.exact && near(p.radius, p.entry.shape.outer_radius)
        && near(Math.abs(p.z1 - p.z0), p.entry.shape.width)));

const noShape = copy(every);
noShape.disks[0].shape = null;
const ring = layoutScene(noShape).parts.find(p => p.key === 'disks:0');
const radiusThere = shaftRadiusAt(noShape.shafts, noShape.disks[0].z, 0);
check('a disk with no equivalent is a thin ring in proportion to its shaft, and says so',
    ring.symbolic === true && near(ring.radius, SYMBOL.disk.radius * radiusThere));

const bearing = everyParts.find(p => p.kind === 'bearing' && !p.hanging);
check('a bearing is sized by the shaft it holds',
    near(bearing.radius, SYMBOL.bearing.radius * shaftRadiusAt(every.shafts, bearing.entry.z, 0)));

// every_element: a point mass on link node 5, which bearing 0 (node 0) links to.
const hanging = everyParts.find(p => p.kind === 'pointmass');
check('a part on a link node hangs under the node that links to it',
    hanging.entry.link === true && hanging.hanging === 1
    && near(hanging.offset.y, -SYMBOL.link * hanging.shaftRadius) && hanging.entry.z === every.bearings[0].z);

// A tapered element, read halfway along; a sleeve over a shaft, read outside.
const tapered = [{ z0: 0, z1: 1, odl: 0.1, odr: 0.2 }];
check('the radius of a tapered element halfway along is halfway between its ends',
    near(shaftRadiusAt(tapered, 0.5, 0), 0.075));
const layered = [{ z0: 0, z1: 1, odl: 0.1, odr: 0.1 }, { z0: 0, z1: 1, odl: 0.14, odr: 0.14 }];
check('under a sleeve, the radius is the sleeve\'s', near(shaftRadiusAt(layered, 0.3, 0), 0.07));

// --- the MultiRotor -------------------------------------------------------------------
console.log('\nThe second shaft line of a MultiRotor');

const multi = CASES.multirotor.scene;
const placed = drivenPlacement(multi);
const pitch = (line, node) => line.gears.find(g => g.n === node).pitch_radius;
const apart = pitch(multi.driving, multi.coupled_nodes[0]) + pitch(multi.driven, multi.coupled_nodes[1]);
check('the two axes are the sum of the pitch radii apart',
    near(Math.hypot(placed.x, placed.y), apart));
check('along the line of centres at ROSS\'s orientation angle',
    near(Math.atan2(placed.y, placed.x), multi.orientation_angle));
check('and shifted along the axis by ROSS\'s own dz_pos', near(placed.z, multi.driven_offset));
const multiParts = layoutScene(multi).parts;
check('the driven line\'s parts are keyed apart and carry the shift',
    multiParts.filter(p => p.half === 'driven').every(p => p.key.startsWith('driven:')
        && (p.hanging || (near(p.offset.x, placed.x) && near(p.offset.y, placed.y) && near(p.offset.z, placed.z)))));

// ROSS's own MultiRotor example: the pinion's pitch diameter (77 mm) is smaller
// than its 150 mm shaft. Drawn as given, its teeth were buried in the shaft
// and the two gears were drawn inside each other -- what Leonardo saw.
const example = CASES.ross_multirotor_example.scene;
const exampleParts = layoutScene(example).parts;
const pinion = exampleParts.find(p => p.half === 'driven' && p.kind === 'gear');
const wheel = exampleParts.find(p => p.half === 'driving' && p.kind === 'gear');
check('a pitch circle inside its own shaft is drawn as teeth cut on the shaft',
    pinion.gear.onShaft === true && near(pinion.gear.root, pinion.shaftRadius)
    && pinion.gear.pitch > pinion.shaftRadius && near(pinion.gear.rossPitch, pinion.entry.pitch_radius));
check('a gear bigger than its shaft keeps ROSS\'s pitch circle',
    wheel.gear.onShaft === false && near(wheel.gear.pitch, wheel.entry.pitch_radius));
const exampleOffset = drivenPlacement(example);
const exampleDistance = Math.hypot(exampleOffset.x, exampleOffset.y);
check('the two gears are drawn in mesh: the axes are the drawn pitch radii apart',
    near(exampleDistance, wheel.gear.pitch + pinion.gear.pitch));
check('and neither shaft runs into the other gear',
    exampleDistance - pinion.shaftRadius >= wheel.gear.tip && exampleDistance - wheel.shaftRadius >= pinion.gear.tip);
check('a gear with no width in ROSS is drawn ten modules wide, and says so',
    pinion.entry.width === null && pinion.gear.widthAssumed && near(pinion.faceWidth, 10 * pinion.gear.module));

// --- shafts and couplings --------------------------------------------------------
console.log('\nShaft ends and couplings');

const everyShafts = everyParts.filter(p => p.kind === 'shaft').sort((a, b) => a.z0 - b.z0);
check('a free end of the shaft line is chamfered', everyShafts[0].chamfer[0] === true);
check('an end that meets a thicker element is not', everyShafts[0].chamfer[1] === false);
check('the thicker element\'s end, standing proud of the thinner one, is', everyShafts[1].chamfer[0] === true);

// A coupling is drawn at its own length, mid-span: hubs at the two ends of
// its body, a slender spacer between them, and the shaft showing on each side
// as it runs into a hub. (Leonardo: on node 0 it used to swallow the whole
// first element up to the bearing.)
const couplingCase = CASES.coupling.scene;
const couplingPart = layoutScene(couplingCase).parts.find(p => p.kind === 'coupling');
const [body0, body1] = couplingPart.body;
const span = couplingPart.z1 - couplingPart.z0;
check('the coupling is drawn shorter than its span, in the middle of it',
    body1 - body0 <= 0.6 * span + 1e-12 && body1 - body0 <= 2.6 * couplingPart.radius + 1e-12
    && near((body0 + body1) / 2, (couplingPart.z0 + couplingPart.z1) / 2));
const longSpan = copy(couplingCase);
longSpan.couplings[0].z1 = longSpan.couplings[0].z0 + 2.0;
const longPart = layoutScene(longSpan).parts.find(p => p.kind === 'coupling');
check('on a long span it keeps its own size, not the span\'s',
    near(longPart.body[1] - longPart.body[0], 2.6 * longPart.radius));
const couplingModel = buildRotorModel(THREE, { parts: [couplingPart], rings: [] });
const cp = couplingModel.object.getObjectByName('metal').geometry.attributes.position.array;
const bodyLength = body1 - body0;
let hubAtLeft = false;
let hubAtRight = false;
let bulkInMiddle = false;
let shaftLeft = false;
let shaftRight = false;
for (let i = 0; i < cp.length; i += 3) {
    const radius = Math.hypot(cp[i], cp[i + 1]);
    const z = cp[i + 2];
    if (radius > 0.95 * couplingPart.radius && z < body0 + 0.35 * bodyLength) hubAtLeft = true;
    if (radius > 0.95 * couplingPart.radius && z > body1 - 0.35 * bodyLength) hubAtRight = true;
    if (radius > 0.6 * couplingPart.radius && Math.abs(z - (body0 + body1) / 2) < 0.1 * bodyLength) bulkInMiddle = true;
    if (radius > 1e-6 && radius <= couplingPart.bores[0] * 1.0001 && z < body0 - 1e-6) shaftLeft = true;
    if (radius > 1e-6 && radius <= couplingPart.bores[1] * 1.0001 && z > body1 + 1e-6) shaftRight = true;
}
check('a coupling is a flanged hub at each end of its body', hubAtLeft && hubAtRight);
check('and not a drum across the span', !bulkInMiddle);
check('the shaft shows on each side of it, running into the hubs', shaftLeft && shaftRight);
check('it carries ROSS\'s two lumped masses for the tooltip',
    couplingPart.entry.m_l === 2 && couplingPart.entry.m_r === 3);
couplingModel.dispose();

// The shaft line is separated at the coupling (Leonardo: "como se os dois eixos
// se separassem pelo acoplamento"): it runs into each hub and stops. When a
// shaft element also spans the coupling's nodes -- a second, parallel
// stiffness in ROSS -- it is cut the same way, and the tooltip says so.
const gapOf = part => [part.body[0] + part.hub, part.body[1] - part.hub];
const shaftVerticesIn = (model, [g0, g1]) => {
    const p = model.object.getObjectByName('metal').geometry.attributes.position.array;
    let inside = 0;
    for (let i = 0; i < p.length; i += 3) if (p[i + 2] > g0 + 1e-6 && p[i + 2] < g1 - 1e-6) inside += 1;
    return inside;
};
const shaftsOnly = parts => ({ parts: parts.filter(p => p.kind === 'shaft'), rings: [] });
const plainParts = layoutScene(couplingCase).parts;
const plainCoupling = plainParts.find(p => p.kind === 'coupling');
check('with no shaft element across it, the coupling carries the shaft ends in its hubs',
    plainCoupling.stubs[0] && plainCoupling.stubs[1] && plainCoupling.overlapsShaft === false);

const overlapping = copy(couplingCase);
const joint = overlapping.couplings[0];
overlapping.shafts.push({ ...overlapping.shafts[0], index: 3, n: joint.n, z0: joint.z0, z1: joint.z1, tag: 'over' });
const overParts = layoutScene(overlapping).parts;
const overCoupling = overParts.find(p => p.kind === 'coupling');
const overShaft = overParts.find(p => p.kind === 'shaft' && p.entry.tag === 'over');
check('a shaft element across the coupling is cut between the hubs',
    overShaft.cuts.length === 1 && near(overShaft.cuts[0][0], gapOf(overCoupling)[0]) && near(overShaft.cuts[0][1], gapOf(overCoupling)[1]));
const overModel = buildRotorModel(THREE, shaftsOnly(overParts));
check('and nothing of it is drawn there', shaftVerticesIn(overModel, gapOf(overCoupling)) === 0);
check('while both its ends are, running into the hubs',
    shaftVerticesIn(overModel, [overCoupling.z0, gapOf(overCoupling)[0]]) > 0
    && shaftVerticesIn(overModel, [gapOf(overCoupling)[1], overCoupling.z1]) > 0);
overModel.dispose();
check('its ends in the hubs are its own, so the coupling adds no stubs, and says a shaft is there too',
    !overCoupling.stubs[0] && !overCoupling.stubs[1] && overCoupling.overlapsShaft === true);
const others = overParts.filter(p => p.kind === 'shaft' && p.entry.tag !== 'over');
check('shaft elements outside the coupling are not cut', others.every(p => p.cuts.length === 0));

// --- the bench ---------------------------------------------------------------------
console.log('\nThe test bench');

for (const [name, { scene }] of Object.entries(CASES)) {
    const layout = layoutScene(scene);
    const bench = benchLayout(layout);
    const bearingsHere = layout.parts.filter(p => p.kind === 'bearing');
    const motor = bench.motor;
    check(`${name}: the bench top is below every part of the rotor, and the motor stands on it`,
        bench.plate.top < layout.bounds.min[1] && bench.plate.top <= motor.feet + 1e-12);
    check(`${name}: the plate is under the whole rotor`,
        bench.plate.x0 < layout.bounds.min[0] && bench.plate.x1 > layout.bounds.max[0]
        && bench.plate.z0 < layout.bounds.min[2] && bench.plate.z1 > layout.bounds.max[2]);
    const underBearings = bench.pedestals.filter(p => p.key !== 'motor');
    const underMotor = bench.pedestals.filter(p => p.key === 'motor');
    check(`${name}: a pedestal under every bearing, from the plate to the bearing's base`,
        underBearings.length === bearingsHere.length && underBearings.every(p => {
            const b = bearingsHere.find(x => x.key === p.key);
            return near(p.y0, bench.plate.top) && near(p.y1, b.offset.y - SYMBOL.bearing.base * b.shaftRadius) && p.y1 > p.y0;
        }));
    check(`${name}: short legs, each on an isolator reaching the floor`,
        bench.legs.every(l => near(bench.plate.bottom - bench.legHeight - l.mount, bench.floor))
        && bench.legHeight < 0.5 * (bench.plate.x1 - bench.plate.x0) + 0.04 * (layout.bounds.max[2] - layout.bounds.min[2]));
    check(`${name}: legs from the plate to the floor, and the camera takes the whole bench in`,
        bench.legs.length >= 4 && bench.floor < bench.plate.bottom
        && bench.bounds.min[1] === bench.floor && bench.bounds.min[0] <= layout.bounds.min[0]
        && bench.bounds.max[1] === Math.max(layout.bounds.max[1], motor.top));
    // The motor: at node 0's end of the line it drives, before all that is
    // drawn, on that line's axis, its shaft reaching the first shaft element
    // and the coupling in between; on the plate, or on a pedestal when the
    // plate is lower than its feet.
    const driven = layout.parts.filter(p => p.kind === 'shaft' && p.half !== 'driven');
    const start = Math.min(...driven.map(p => p.offset.z + Math.min(p.z0, p.z1)));
    check(`${name}: a motor at the start of the rotor, on its axis, joined to its first shaft element`,
        motor.z0 < motor.z1 && motor.z1 < layout.bounds.min[2] && near(motor.shaftTo, start)
        && motor.x === driven[0].offset.x && motor.y === driven[0].offset.y
        && motor.coupling[0] > motor.z1 && motor.coupling[1] < Math.min(layout.bounds.min[2], start)
        && motor.radius > motor.shaftRadius);
    check(`${name}: the motor on the plate, or on a pedestal reaching it`,
        underMotor.length === 0 ? near(motor.feet, bench.plate.top)
            : underMotor.length === 1 && near(underMotor[0].y0, bench.plate.top) && near(underMotor[0].y1, motor.feet));
    check(`${name}: and on the bench, in the box the camera frames`,
        bench.plate.z0 < motor.z0 && bench.bounds.min[2] <= motor.z0 && bench.plate.x0 < motor.x - motor.across
        && bench.plate.x1 > motor.x + motor.across);
    const motorBox = new THREE.Box3();
    motorPieces(THREE, motor, []).forEach(({ geometry }) => { geometry.computeBoundingBox(); motorBox.union(geometry.boundingBox); geometry.dispose(); });
    const give = 1e-6;   // the geometry is kept in float32
    check(`${name}: the motor drawn is the size the layout gives it`,
        motorBox.min.x >= motor.x - motor.across - give && motorBox.max.x <= motor.x + motor.across + give
        && motorBox.min.y >= motor.feet - give && motorBox.max.y <= motor.top + give
        && motorBox.min.z >= motor.z0 - give && motorBox.max.z <= motor.shaftTo + give
        && Math.abs(motorBox.min.y - motor.feet) < give && Math.abs(motorBox.max.z - motor.shaftTo) < give);
    const built = buildBench(THREE, bench);
    const benchBox = new THREE.Box3().setFromObject(built.object);
    check(`${name}: the bench drawn fits the box the camera frames`,
        benchBox.min.y >= bench.bounds.min[1] - 1e-6 && benchBox.min.x >= bench.bounds.min[0] - 1e-6
        && benchBox.max.x <= bench.bounds.max[0] + 1e-6 && benchBox.max.z <= bench.bounds.max[2] + 1e-6
        && benchBox.min.z >= bench.bounds.min[2] - 1e-6 && benchBox.max.y <= bench.bounds.max[1] + 1e-6);
    const held = [];
    built.object.traverse(o => { if (o.geometry) held.push(o.geometry, o.material); });
    const gone = new Set();
    held.forEach(o => o.addEventListener('dispose', () => gone.add(o)));
    built.dispose();
    check(`${name}: taking the bench away frees it -- painted, bare metal, outlines`,
        held.length === 6 && held.every(o => gone.has(o)));
}

// --- the pointer ----------------------------------------------------------------------
console.log('\nThe pointer finds the part it is on');

const gearPart = everyParts.find(p => p.kind === 'gear');
const gz = gearPart.entry.z;
const down = pickPart(everyParts, [0, 1, gz], [0, -1, 0]);
check('straight down onto a gear, it is the gear', down && down.part === gearPart);
check('at the gear\'s tip, not at the shaft inside it', down && near(down.distance, 1 - gearPart.radius));

const shaftOnly = everyParts.find(p => p.kind === 'shaft' && p.index === 2);
const zShaft = (shaftOnly.z0 + shaftOnly.z1) / 2;
const onShaft = pickPart(everyParts, [0, 1, zShaft], [0, -1, 0]);
check('between the parts, the shaft', onShaft && onShaft.part.kind === 'shaft' && onShaft.part.index === 2);
check('beside the rotor, nothing', pickPart(everyParts, [1, 1, zShaft], [0, -1, 0]) === null);
check('along the axis from past the end, the last part first',
    (() => {
        const hit = pickPart(everyParts, [0, 0, 5], [0, 0, -1]);
        const zs = everyParts.filter(p => !p.hanging && p.offset.y === 0).map(p => Math.max(p.z0, p.z1));
        return hit && near(5 - hit.distance, Math.max(...zs));
    })());
const drivenGear = multiParts.find(p => p.half === 'driven' && p.kind === 'gear');
const drivenHit = pickPart(multiParts, [placed.x, placed.y + 1, drivenGear.entry.z + placed.z], [0, -1, 0]);
check('on the driven line, the driven line\'s gear', drivenHit && drivenHit.part === drivenGear);

// --- the camera -----------------------------------------------------------------------
console.log('\nThe camera frames what is drawn, closely');

for (const [name, { scene }] of Object.entries(CASES)) {
    const { bounds } = layoutScene(scene);
    const direction = viewDirection(scene);
    const fov = 35;
    const aspect = 1.6;
    const view = framing(bounds, fov, aspect, direction);
    // Independently: each corner, seen from the camera, inside the half-angles.
    const back = view.direction;
    const eye = view.center.map((c, i) => c + back[i] * view.distance);
    const forward = back.map(v => -v);
    const r = [-forward[2], 0, forward[0]];
    const rl = Math.hypot(...r);
    const rightAxis = r.map(v => v / rl);
    const upAxis = [
        rightAxis[1] * forward[2] - rightAxis[2] * forward[1],
        rightAxis[2] * forward[0] - rightAxis[0] * forward[2],
        rightAxis[0] * forward[1] - rightAxis[1] * forward[0],
    ];
    const tanV = Math.tan(fov * Math.PI / 360);
    const tanH = tanV * aspect;
    let worst = 0;
    for (const x of [bounds.min[0], bounds.max[0]]) for (const y of [bounds.min[1], bounds.max[1]]) for (const z of [bounds.min[2], bounds.max[2]]) {
        const p = [x - eye[0], y - eye[1], z - eye[2]];
        const depth = p[0] * forward[0] + p[1] * forward[1] + p[2] * forward[2];
        const h = Math.abs(p[0] * rightAxis[0] + p[1] * rightAxis[1] + p[2] * rightAxis[2]) / (depth * tanH);
        const v = Math.abs(p[0] * upAxis[0] + p[1] * upAxis[1] + p[2] * upAxis[2]) / (depth * tanV);
        worst = Math.max(worst, h, v);
    }
    check(`${name}: every corner is in view`, worst <= 1 + 1e-9);
    check(`${name}: and the rotor fills most of it (${(worst * 100).toFixed(0)}% of the frame)`, worst >= 0.85);
}

// --- the solids -------------------------------------------------------------------------
console.log('\nThe solids: a mesh per finish, ROSS\'s colours, in the layout\'s box, freed on rebuild');

// Every triangle of a mesh faces the way its normals say. `revolve` writes its
// normals itself; a triangle wound the other way is lit from behind, which is
// what a rotor that looks black on the side facing the light is.
function facingOut(geometry) {
    const p = geometry.attributes.position.array;
    const n = geometry.attributes.normal.array;
    let good = 0;
    let counted = 0;
    for (let i = 0; i < p.length; i += 9) {
        const e1 = [p[i + 3] - p[i], p[i + 4] - p[i + 1], p[i + 5] - p[i + 2]];
        const e2 = [p[i + 6] - p[i], p[i + 7] - p[i + 1], p[i + 8] - p[i + 2]];
        const c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        const area = Math.hypot(...c);
        if (area < 1e-12) continue;
        const normal = [0, 1, 2].map(k => n[i + k] + n[i + 3 + k] + n[i + 6 + k]);
        counted += 1;
        if (c[0] * normal[0] + c[1] * normal[1] + c[2] * normal[2] > 0) good += 1;
    }
    return counted ? good / counted : 0;
}

for (const [name, { scene }] of Object.entries(CASES)) {
    const layout = layoutScene(scene);
    const model = buildRotorModel(THREE, layout);
    const meshes = model.object.children.filter(o => o.isMesh);
    const names = model.object.children.map(o => o.name).sort();
    check(`${name}: a mesh per finish, the outlines and the node rings, nothing else`,
        meshes.length >= 1 && names.every(n => ['metal', 'paint', 'outlines', 'nodes'].includes(n))
        && model.object.getObjectByName('nodes').isLineSegments
        && model.object.getObjectByName('outlines').isLineSegments);
    check(`${name}: coloured by vertex`, meshes.every(m => m.material.vertexColors === true && !!m.geometry.attributes.color));
    check(`${name}: every face turned the way it is lit`, meshes.every(m => facingOut(m.geometry) > 0.999));
    const box = new THREE.Box3();
    meshes.forEach(m => { m.geometry.computeBoundingBox(); box.union(m.geometry.boundingBox); });
    const size = Math.max(...[0, 1, 2].map(i => layout.bounds.max[i] - layout.bounds.min[i]));
    const inside = ['x', 'y', 'z'].every((axis, i) => box.min[axis] >= layout.bounds.min[i] - 1e-3 * size
        && box.max[axis] <= layout.bounds.max[i] + 1e-3 * size);
    const filling = ['x', 'y', 'z'].every((axis, i) => box.min[axis] <= layout.bounds.min[i] + 2e-2 * size
        && box.max[axis] >= layout.bounds.max[i] - 2e-2 * size);
    check(`${name}: what is drawn stays inside the box the camera frames`, inside);
    check(`${name}: and reaches its sides`, filling);
    model.dispose();
}

// Colours: ROSS's names, not the prototype's fixed palette.
const oneDisk = { parts: layoutScene(every).parts.filter(p => p.kind === 'disk').slice(0, 1), rings: [], bounds: null };
const diskModel = buildRotorModel(THREE, oneDisk);
const colour = diskModel.object.getObjectByName('metal').geometry.attributes.color.array;
const firebrick = new THREE.Color().setHex(THREE.Color.NAMES.firebrick);
check('a disk is ROSS\'s Firebrick', near(colour[0], firebrick.r, 1e-6) && near(colour[1], firebrick.g, 1e-6)
    && near(colour[2], firebrick.b, 1e-6));
diskModel.dispose();
const unknown = { ...oneDisk, parts: [{ ...oneDisk.parts[0], color: 'not-a-colour' }] };
const greyModel = buildRotorModel(THREE, unknown);
const grey = greyModel.object.getObjectByName('metal').geometry.attributes.color.array;
check('a colour three.js does not know is the shaft grey, not white',
    near(grey[0], new THREE.Color(0x525252).r, 1e-6) && grey[0] < 0.5);
greyModel.dispose();

// Teeth: the gear's own count, up to the cap.
const gearOnly = teeth => {
    const found = layoutScene(every).parts.find(p => p.kind === 'gear');
    const part = { ...found, gear: { ...found.gear, teeth } };
    const model = buildRotorModel(THREE, { parts: [part], rings: [] });
    const count = model.vertices;
    model.dispose();
    return count;
};
check('more teeth, more vertices: the count is the gear\'s', gearOnly(40) > gearOnly(20));

// The teeth are as wide as the gear's face; only the hub stands proud of them.
const gearModel = buildRotorModel(THREE, { parts: [gearPart], rings: [] });
const positions = gearModel.object.getObjectByName('metal').geometry.attributes.position.array;
let toothLo = Infinity;
let toothHi = -Infinity;
for (let i = 0; i < positions.length; i += 3) {
    if (Math.hypot(positions[i], positions[i + 1]) > gearPart.gear.pitch) {
        toothLo = Math.min(toothLo, positions[i + 2]);
        toothHi = Math.max(toothHi, positions[i + 2]);
    }
}
check('the teeth span the face width ROSS gives the gear',
    near(toothHi - toothLo, gearPart.entry.width, 1e-5) && near((toothHi + toothLo) / 2, gearPart.entry.z, 1e-5));
gearModel.dispose();
check('past the cap, no more', gearOnly(100000) === gearOnly(MAX_DRAWN_TEETH));

// Freeing: every geometry and material the model holds, and the pieces merged
// into it as soon as they are copied.
const freed = new Set();
const watch = object => object.addEventListener('dispose', () => freed.add(object));
const model = buildRotorModel(THREE, layoutScene(compressor));
const held = [];
model.object.traverse(o => { if (o.geometry) held.push(o.geometry, o.material); });
held.forEach(watch);
model.dispose();
check('dispose frees every geometry and material the model holds', held.every(o => freed.has(o)));
check('and leaves nothing hanging from the group', model.object.children.length === 0);

// Indexed pieces (as three.js makes them) and one already flat, which is
// copied as it is and has to be freed all the same.
const pieces = [0, 1, 2].map(() => new THREE.CylinderGeometry(1, 1, 1, 8));
pieces.push(new THREE.CylinderGeometry(1, 1, 1, 8).toNonIndexed());
pieces.forEach(watch);
const merged = mergeColoured(THREE, pieces.map(geometry => ({ geometry, color: new THREE.Color(1, 0, 0) })));
check('merging frees the pieces it copied, indexed or not', pieces.every(p => freed.has(p)));
check('and keeps all their vertices',
    merged.attributes.position.count === 4 * new THREE.CylinderGeometry(1, 1, 1, 8).toNonIndexed().attributes.position.count);
merged.dispose();


// --- probes ---------------------------------------------------------------------------
console.log('\nProbes: where the analyses read the response');

// The probes are not in ROSS's rotor (domain/rotor_builder.py); the scene
// places them by their node, with their direction and angle.
const probeParts = everyParts.filter(p => p.kind === 'probe');
const radialProbe = probeParts.find(p => !p.axial);
const axialProbe = probeParts.find(p => p.axial);
check('a part per probe', probeParts.length === every.probes.length && !!radialProbe && !!axialProbe);
// A disk sits on node 1, where the radial probe reads: the probe is drawn
// just beside it, on the outer side, and says so. On a bare node it is drawn
// on the node.
const diskOnOne = everyParts.find(p => p.kind === 'disk' && p.z0 <= radialProbe.entry.z && p.z1 >= radialProbe.entry.z);
check('a probe on a node a disk sits on is drawn beside the disk, on the outer side',
    radialProbe.beside === true && radialProbe.outward === -1 && radialProbe.z1 <= diskOnOne.z0 + 1e-12);
const bare = layoutScene({ ...every, disks: [], seals: [] }).parts.find(p => p.kind === 'probe' && !p.axial);
check('on a bare node, on the node itself', !bare.beside && near((bare.z0 + bare.z1) / 2, bare.entry.z));
check('reaching from the axis past the shaft, by the tip, body and cable',
    near(radialProbe.radius, (1 + SYMBOL.probe.gap + SYMBOL.probe.length) * radialProbe.shaftRadius));
check('in the probe colour, which the legend shows', radialProbe.color === PROBE_COLOR);

// Drawn alone, its solid lies where its angle says: from x toward y, as ROSS
// reads x cos(angle) + y sin(angle).
const vertexMean = model => {
    const mesh = model.object.children.filter(c => c.isMesh);
    let sx = 0; let sy = 0; let count = 0;
    mesh.forEach(m => {
        const a = m.geometry.attributes.position;
        for (let i = 0; i < a.count; i++) { sx += a.getX(i); sy += a.getY(i); count++; }
    });
    return [sx / count, sy / count];
};
const radialModel = buildRotorModel(THREE, { parts: [radialProbe], rings: [] });
const [mx, my] = vertexMean(radialModel);
check(`the radial probe points from its angle (${(Math.atan2(my, mx) * 180 / Math.PI).toFixed(1)}° for 45°)`,
    Math.abs(Math.atan2(my, mx) - radialProbe.entry.angle) < 1e-6);
check('and stays outside the shaft', Math.hypot(mx, my) > radialProbe.shaftRadius);
radialModel.dispose();
const turned = layoutScene({ ...every, probes: [{ ...every.probes[0], angle: -2.5 }] }).parts.find(p => p.kind === 'probe');
const turnedModel = buildRotorModel(THREE, { parts: [turned], rings: [] });
const [tx, ty] = vertexMean(turnedModel);
check('at any angle, below the axis too', Math.abs(Math.atan2(ty, tx) - -2.5) < 1e-6);
turnedModel.dispose();

// An axial probe lies along the shaft, on the outer side of its node: the
// one in the scene is on node 3 of 5, past the middle, so it points to +z.
check('an axial probe lies along the shaft, outward of its node',
    axialProbe.outward === 1 && axialProbe.z0 > axialProbe.entry.z && axialProbe.z1 > axialProbe.z0);
const diskOnThree = everyParts.find(p => p.kind === 'disk' && p.z0 <= axialProbe.entry.z && p.z1 >= axialProbe.entry.z);
check('facing the face of the disk on its node, a gap off it', axialProbe.beside === true
    && near(axialProbe.z0, diskOnThree.z1 + SYMBOL.probe.gap * axialProbe.shaftRadius));
check('over the shaft, on a node in the middle of the line', axialProbe.lift > axialProbe.shaftRadius);
const atTheEnd = layoutScene({ ...every, probes: [{ ...every.probes[1], n: 0, z: every.nodes[0].z }] }).parts.find(p => p.kind === 'probe');
check('on the axis, facing the end face, at an end of the line', atTheEnd.lift === 0 && atTheEnd.outward === -1
    && atTheEnd.z1 < every.nodes[0].z);
const axialModel = buildRotorModel(THREE, { parts: [axialProbe], rings: [] });
const box = new THREE.Box3().setFromObject(axialModel.object);
check('and its solid is where its part says it is',
    box.min.z >= axialProbe.z0 - 1e-6 && box.max.z <= axialProbe.z1 + 1e-6 && box.min.y > axialProbe.shaftRadius);
axialModel.dispose();
check('a MultiRotor line keeps its own probes', layoutScene(CASES.multirotor.scene).parts
    .filter(p => p.kind === 'probe').every(p => p.half === 'driving'));

shutDown();
