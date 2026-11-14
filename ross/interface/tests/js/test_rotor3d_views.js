// The standard views of the 3D view and the camera's flight between them
// (core/rotor3d_views.js, features/rotor3d.js).
//
// WHY THIS BATTERY EXISTS. A view can look right and be wrong:
//
//   * a view that mirrors the rotor (node 0 on the right) contradicts the 2D
//     figure beside it, as the first camera of the 3D view did;
//   * a view whose framing leaves a corner of the rotor off the screen;
//   * a flight that cuts through the rotor or under the floor on its way, or
//     does not land exactly where the view is.
import fs from 'node:fs';
import { check, shutDown } from './fake_dom.js';

const THREE = await import('../../frontend/vendor/three/three.module.js');
const { framing, layoutScene, viewDirection } = await import('../../frontend/core/rotor3d_layout.js');
const { FLIGHT_MS, VIEWS, cameraBetween, easeInOut, viewTowards } = await import('../../frontend/core/rotor3d_views.js');
const { showView, viewEntries } = await import('../../frontend/features/rotor3d.js');
const { setSchemaLanguage, t } = await import('../../frontend/core/i18n.js');

const CASES = JSON.parse(fs.readFileSync(new URL('../golden/rotor_scenes.json', import.meta.url)));
const close = (a, b, tolerance = 1e-9) => a.every((v, i) => Math.abs(v - b[i]) <= tolerance);
const length = v => Math.hypot(...v);

function cameraAt(position, target, aspect = 1.6) {
    const camera = new THREE.PerspectiveCamera(35, aspect, 1e-4, 1e4);
    camera.position.set(...position);
    camera.lookAt(...target);
    camera.updateMatrixWorld();
    return camera;
}

function framed(scene, name, aspect = 1.6) {
    const { bounds } = layoutScene(scene);
    const view = framing(bounds, 35, aspect, viewTowards(name, scene));
    const position = view.center.map((c, i) => c + view.direction[i] * view.distance);
    return { bounds, view, camera: cameraAt(position, view.center, aspect), position, target: view.center };
}

const onScreen = (camera, point) => new THREE.Vector3(...point).project(camera);

// --- the views ---------------------------------------------------------------------------
console.log('\nThe views');

check('five views, the opening one first', VIEWS.join() === 'home,iso,front,side,top');
for (const [name, { scene }] of Object.entries(CASES)) {
    check(`${name}: the opening view is the one the page opens on`,
        close(viewTowards('home', scene), viewDirection(scene)));
}
const iso = viewTowards('iso', null);
check('isometric: the three axes at equal angles', close(iso.map(Math.abs), [1, 1, 1].map(v => v / Math.sqrt(3)), 1e-12));
check('and from above', iso[1] > 0);
check('front: straight along the rotor axis, from the side of node 0', close(viewTowards('front', null), [0, 0, -1]));
check('side: square to the axis and level', close(viewTowards('side', null), [-1, 0, 0]));
const top = viewTowards('top', null);
check('top: from above, to within a tenth of a degree', top[1] > Math.cos(0.1 * Math.PI / 180));
check('every view is a unit direction', VIEWS.every(v => Math.abs(length(viewTowards(v, CASES.compressor_example.scene)) - 1) < 1e-12));
check('an unknown view is the opening one', close(viewTowards('nonsense', CASES.compressor_example.scene),
    viewDirection(CASES.compressor_example.scene)));

for (const [name, { scene }] of Object.entries(CASES)) {
    const { rings } = layoutScene(scene);
    const line = rings.filter(r => !r.half || r.half === 'driving');
    const first = line.reduce((a, b) => (b.n < a.n ? b : a));
    const last = line.reduce((a, b) => (b.n > a.n ? b : a));
    for (const view of VIEWS) {
        const { bounds, camera } = framed(scene, view);
        let inside = true;
        for (const x of [bounds.min[0], bounds.max[0]]) {
            for (const y of [bounds.min[1], bounds.max[1]]) {
                for (const z of [bounds.min[2], bounds.max[2]]) {
                    const p = onScreen(camera, [x, y, z]);
                    if (Math.abs(p.x) > 1 + 1e-9 || Math.abs(p.y) > 1 + 1e-9) inside = false;
                }
            }
        }
        check(`${name}, ${view}: every corner of the rotor on the screen`, inside);
        if (view === 'front') continue;
        const a = onScreen(camera, [0, 0, first.z]);
        const b = onScreen(camera, [0, 0, last.z]);
        check(`${name}, ${view}: node ${first.n} left of node ${last.n}, as in the 2D figure`, a.x < b.x);
    }
    // Along the axis, the shaft line is seen end on: its two ends fall on the
    // same point of the screen, the middle of it.
    if (scene.kind !== 'multirotor') {
        const { camera } = framed(scene, 'front');
        const a = onScreen(camera, [0, 0, first.z]);
        const b = onScreen(camera, [0, 0, last.z]);
        check(`${name}, front: the shaft line seen end on, in the middle`,
            Math.hypot(a.x - b.x, a.y - b.y) < 1e-9 && Math.abs(a.x) < 1e-9 && Math.abs(a.y) < 0.5);
    }
}

// --- the flight ----------------------------------------------------------------------------
console.log('\nThe flight between two views');

check('half a second', FLIGHT_MS === 500);
check('eased: starts and ends at rest', easeInOut(0) === 0 && easeInOut(1) === 1 && Math.abs(easeInOut(0.5) - 0.5) < 1e-12);
let monotone = true;
for (let k = 1; k <= 100; k += 1) if (easeInOut(k / 100) < easeInOut((k - 1) / 100)) monotone = false;
check('and never goes back', monotone);
check('clamped outside 0..1', easeInOut(-1) === 0 && easeInOut(2) === 1);

const scene = CASES.compressor_example.scene;
const from = framed(scene, 'home');
const to = framed(scene, 'top');
const start = cameraBetween(from, to, 0);
const end = cameraBetween(from, to, 1);
check('leaves from where the camera is', close(start.position, from.position, 1e-9) && close(start.target, from.target, 1e-12));
check('and lands exactly on the view', close(end.position, to.position, 1e-9) && close(end.target, to.target, 1e-12));

// Around the point looked at, along the great circle, at a distance changing
// by the same ratio every step.
const r0 = length(from.position.map((v, i) => v - from.target[i]));
const r1 = length(to.position.map((v, i) => v - to.target[i]));
let onCircle = true;
let ratio = true;
let previous = -1;
let turning = true;
let steady = true;
const a0 = from.position.map((v, i) => (v - from.target[i]) / r0);
const b1 = to.position.map((v, i) => (v - to.target[i]) / r1);
const whole = Math.acos(a0[0] * b1[0] + a0[1] * b1[1] + a0[2] * b1[2]);
for (let k = 0; k <= 20; k += 1) {
    const at = cameraBetween(from, to, k / 20);
    const offset = at.position.map((v, i) => v - at.target[i]);
    const r = length(offset);
    if (Math.abs(r - r0 * Math.pow(r1 / r0, k / 20)) > 1e-9 * r0) ratio = false;
    const unit = offset.map(v => v / r);
    const angle = Math.acos(Math.max(-1, Math.min(1, unit[0] * a0[0] + unit[1] * a0[1] + unit[2] * a0[2])));
    if (angle < previous - 1e-12) turning = false;
    if (Math.abs(angle - whole * k / 20) > 1e-7) steady = false;
    previous = angle;
    // On the plane of the two directions: a great circle.
    const b0 = to.position.map((v, i) => (v - to.target[i]) / r1);
    const normal = [a0[1] * b0[2] - a0[2] * b0[1], a0[2] * b0[0] - a0[0] * b0[2], a0[0] * b0[1] - a0[1] * b0[0]];
    if (Math.abs(normal[0] * unit[0] + normal[1] * unit[1] + normal[2] * unit[2]) > 1e-9) onCircle = false;
}
check('turns along the great circle between the two directions', onCircle);
check('always further round, never back', turning);
check('at a steady angular pace: the easing alone shapes the speed', steady);
check('the distance changes by the same ratio every step', ratio);

// The way round between opposite directions goes over the top, never under
// the floor -- the front view and the one straight behind it.
const ahead = { position: [0, 0, -5], target: [0, 0, 0] };
const behind = { position: [0, 0, 5], target: [0, 0, 0] };
let over = true;
for (let k = 0; k <= 20; k += 1) {
    const at = cameraBetween(ahead, behind, k / 20);
    if (at.position[1] < -1e-9) over = false;
    if (Math.abs(length(at.position) - 5) > 1e-9) over = false;
}
check('opposite directions: over the top, at the same distance', over
    && cameraBetween(ahead, behind, 0.5).position[1] > 4.99);
const above = { position: [0, 5, 0], target: [0, 0, 0] };
const below = { position: [0, -5, 0], target: [0, 0, 0] };
check('and from straight above to straight below, by the side', Math.abs(length(cameraBetween(above, below, 0.5).position) - 5) < 1e-9);

// --- the menu ------------------------------------------------------------------------------
console.log('\nThe views menu');

for (const language of ['en', 'pt']) {
    setSchemaLanguage(language);
    const entries = viewEntries();
    check(`${language}: a title, then the five views`, entries[0].header === t('rotor3dViewsTitle') && entries.length === 6);
    check(`${language}: every view named in the language`,
        entries.slice(1).every(e => typeof e.label === 'string' && e.label && !e.label.startsWith('rotor3d')));
    check(`${language}: every view with its icon`, entries.slice(1).every(e => /^fa-/.test(e.icon)));
}
setSchemaLanguage('en');
check('before the 3D view is open, choosing a view does nothing it cannot', (() => {
    try { showView('iso'); showView('nonsense'); return true; } catch (e) { return false; }
})());

shutDown();
