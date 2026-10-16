// The 3D view's finish: which way the rotor reads, the axis triad, and the
// theme's colours following a change of theme.
//
// WHY THIS BATTERY EXISTS. Each of the three fails silently, with a picture
// that still looks fine:
//
//   * the camera said, in a comment, that the shaft reads left to right as in
//     ROSS's 2D figure -- and looked from the side that puts node 0 on the
//     right. Switching between 2D and 3D mirrored the rotor, and nothing
//     checked the claim until the triad drew z pointing left;
//   * a triad that turns the wrong way round is worse than none: it points at
//     the wrong axis with confidence;
//   * WebGL keeps the colours it was given. A view drawn in the light theme
//     stayed light in the dark one, with nothing to say it had not heard.
import fs from 'node:fs';
import { check, shutDown } from './fake_dom.js';

const THREE = await import('../../frontend/vendor/three/three.module.js');
const { framing, layoutScene, viewDirection } = await import('../../frontend/core/rotor3d_layout.js');
const { buildRotorModel } = await import('../../frontend/components/rotor3d_parts.js');
const { AXES, LETTER_STROKES, buildTriad } = await import('../../frontend/components/rotor3d_triad.js');
const { applyTheme, onThemeChanged } = await import('../../frontend/core/theme.js');
const { restyleRotor3d } = await import('../../frontend/features/rotor3d.js');

const CASES = JSON.parse(fs.readFileSync(new URL('../golden/rotor_scenes.json', import.meta.url)));

// A camera where the view puts it, looking at the middle of what is drawn.
function cameraFor(scene) {
    const { bounds } = layoutScene(scene);
    const view = framing(bounds, 35, 1.6, viewDirection(scene));
    const camera = new THREE.PerspectiveCamera(35, 1.6, view.distance / 200, view.distance * 20);
    const [cx, cy, cz] = view.center;
    const [dx, dy, dz] = view.direction;
    camera.position.set(cx + dx * view.distance, cy + dy * view.distance, cz + dz * view.distance);
    camera.lookAt(cx, cy, cz);
    camera.updateMatrixWorld();
    return camera;
}

const onScreen = (camera, point) => new THREE.Vector3(...point).project(camera);

// --- which way the rotor reads ---------------------------------------------------------
console.log('\nThe rotor reads left to right, node 0 first, as in the 2D figure');

for (const [name, { scene }] of Object.entries(CASES)) {
    const camera = cameraFor(scene);
    const { rings } = layoutScene(scene);
    const line = rings.filter(r => !r.half || r.half === 'driving');
    const first = line.reduce((a, b) => (b.n < a.n ? b : a));
    const last = line.reduce((a, b) => (b.n > a.n ? b : a));
    const a = onScreen(camera, [0, 0, first.z]);
    const b = onScreen(camera, [0, 0, last.z]);
    check(`${name}: node ${first.n} left of node ${last.n}`, a.x < b.x);
}
const single = viewDirection(CASES.compressor_example.scene);
check('one rotor: seen from a little above', single[1] > 0.2);
check('and from a little in front of node 0', single[2] < 0);

// Every orientation of a MultiRotor's second line, not only the two on file.
let leftToRight = true;
let clearOfTheLine = true;
let notFromBelow = true;
for (let degrees = -180; degrees < 180; degrees += 5) {
    const angle = degrees * Math.PI / 180;
    const d = viewDirection({ kind: 'multirotor', orientation_angle: angle });
    if (!(d[0] < 0)) leftToRight = false;
    // Around the rotor axis, the view keeps off the line between the axes:
    // looking along it, one line hides the other.
    const across = Math.abs(Math.cos(angle) * d[0] + Math.sin(angle) * d[1]) / Math.hypot(d[0], d[1]);
    if (across > Math.cos(50 * Math.PI / 180) + 1e-9) clearOfTheLine = false;
    if (d[1] < -0.2) notFromBelow = false;
}
check('a MultiRotor at any orientation still reads left to right', leftToRight);
check('seen at least 50 degrees off the line between its two axes', clearOfTheLine);
check('and never from far below', notFromBelow);

// --- the triad -----------------------------------------------------------------------------
console.log('\nThe axis triad');

const colours = { x: 'firebrick', y: 'seagreen', z: 'steelblue' };
const triad = buildTriad(THREE, colours);
const drawn = [];
triad.scene.traverse(o => { if (o.isMesh || o.isLineSegments) drawn.push(o); });
check('two draw calls: the arrows, the letters', drawn.length === 2
    && drawn.filter(o => o.isMesh).length === 1 && drawn.filter(o => o.isLineSegments).length === 1);
check('drawn in the colours given, not tone-mapped like the scene',
    drawn.every(o => o.material.toneMapped === false));

const letters = drawn.find(o => o.isLineSegments);
// The centre of one axis's letter on the triad camera's screen (of its box:
// the strokes of a Y are not spread evenly around its middle).
function letterAt(name) {
    let before = 0;
    for (const axis of AXES) {
        if (axis.name === name) break;
        before += LETTER_STROKES[axis.name].length;
    }
    const count = LETTER_STROKES[name].length * 2;
    const p = letters.geometry.attributes.position;
    const xs = [], ys = [];
    for (let i = 0; i < count; i++) { xs.push(p.getX(2 * before + i)); ys.push(p.getY(2 * before + i)); }
    return { x: (Math.min(...xs) + Math.max(...xs)) / 2, y: (Math.min(...ys) + Math.max(...ys)) / 2 };
}

// Looking down -z, as the triad's own camera does: x right, y up, z at you.
triad.orient(new THREE.Quaternion());
check('unturned: x to the right', letterAt('x').x > 1 && Math.abs(letterAt('x').y) < 1e-6);
check('y up', letterAt('y').y > 1 && Math.abs(letterAt('y').x) < 1e-6);
check('z straight at the viewer', Math.hypot(letterAt('z').x, letterAt('z').y) < 1e-6);

// As the view's camera sits: z to the right, like the rotor; y up.
const viewCamera = cameraFor(CASES.compressor_example.scene);
triad.orient(viewCamera.quaternion);
check('turned with the view: z to the right, along the rotor', letterAt('z').x > 0.5);
check('y still up', letterAt('y').y > 0.5);
const zOnScreen = onScreen(viewCamera, [0, 0, 1]).x - onScreen(viewCamera, [0, 0, 0]).x;
check('the same way round as the rotor drawn by that camera', Math.sign(zOnScreen) === Math.sign(letterAt('z').x));

// From above, looking down: y points at the viewer, x and z lie flat.
const above = new THREE.PerspectiveCamera();
above.position.set(0, 10, 0);
above.up.set(0, 0, -1);
above.lookAt(0, 0, 0);
above.updateMatrixWorld();
triad.orient(above.quaternion);
check('seen from above, y points at the viewer', Math.hypot(letterAt('y').x, letterAt('y').y) < 1e-6);

const colourOf = (attribute, i) => '#' + new THREE.Color(attribute.getX(i), attribute.getY(i), attribute.getZ(i)).getHexString();
const arrows = drawn.find(o => o.isMesh);
const hexes = name => '#' + new THREE.Color().setStyle(name).getHexString();
const arrowHexes = new Set();
for (let i = 0; i < arrows.geometry.attributes.color.count; i++) arrowHexes.add(colourOf(arrows.geometry.attributes.color, i));
check('each arrow in its axis colour, and nothing else',
    arrowHexes.size === 3 && ['x', 'y', 'z'].every(n => arrowHexes.has(hexes(colours[n]))));
triad.restyle({ x: 'orange', y: 'purple', z: 'teal' });
const after = new Set();
for (let i = 0; i < arrows.geometry.attributes.color.count; i++) after.add(colourOf(arrows.geometry.attributes.color, i));
check('repainted in place for another theme', after.has(hexes('orange')) && after.has(hexes('purple')) && after.has(hexes('teal')) && after.size === 3);
check('the letters too', colourOf(letters.geometry.attributes.color, 0) === hexes('orange'));

let freed = 0;
drawn.forEach(o => { o.geometry.addEventListener('dispose', () => freed++); o.material.addEventListener('dispose', () => freed++); });
triad.dispose();
check('dispose frees what it made', freed === 4);

// --- the theme ----------------------------------------------------------------------------
console.log('\nA change of theme reaches what WebGL keeps');

const model = buildRotorModel(THREE, layoutScene(CASES.every_element.scene), { ring: 'steelblue', outline: 'black' });
const named = name => model.object.children.find(o => o.name === name);
model.restyle({ ring: 'orange', outline: 'white' });
check('the node rings repainted in place', named('nodes').material.color.getHexString() === new THREE.Color('orange').getHexString());
check('and the outlines', named('outlines').material.color.getHexString() === new THREE.Color('white').getHexString());
model.dispose();

const heard = [];
onThemeChanged(theme => heard.push('first:' + theme));
onThemeChanged(theme => heard.push('second:' + theme));
applyTheme('dark');
check('applying a theme tells every subscriber, not only the last',
    heard.join(',') === 'first:dark,second:dark');

let quiet = null;
try { restyleRotor3d(); } catch (error) { quiet = error; }
check('a theme change before the 3D view was ever opened is nothing to do', quiet === null);

shutDown();
