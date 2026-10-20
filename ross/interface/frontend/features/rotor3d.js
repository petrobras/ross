// The 3D view of the rotor, beside ROSS's 2D figure.
//
// It is a view and nothing else: it draws the scene the server answered with
// the rotor (`domain/rotor_scene.py`), and has no model of its own to fall out
// of step with the element lists. Editing from it goes through the same forms
// and paths the lists use (`editFrom3d` and its siblings in modeling.js).
//
// What the prototype cost, and what this does instead (measured; see the
// project's 3D evaluation):
//
// * three.js is **loaded the first time the 3D view is opened**, with
//   `import()`: whoever never opens it never downloads it;
// * a frame is drawn **when something changed** -- the camera moved, the
//   rotor was rebuilt, the pointer moved -- and not sixty times a second while
//   the page sits there;
// * the pointer is resolved **once per frame**, against the cylinder around
//   each part (`pickPart`), not once per mouse event against every triangle;
// * a rebuild **frees what it replaces** (`components/rotor3d_parts.js`);
// * the renderer is made once and kept: one WebGL context for the page's
//   lifetime, where browsers allow only a handful before they drop the oldest.
import { escapeHtml } from '../core/dom.js';
import { t } from '../core/i18n.js';
import {
    annotationLayout, benchLayout, framing, hitPoint, layoutScene, nearestNode, pickPart, viewDirection,
} from '../core/rotor3d_layout.js';
import { arrangeLabels, buildAnnotations, fillLabels } from '../components/rotor3d_annotations.js';
import {
    alongAxis, grabbedEnd, keptEnds, measureLine, measureTop, measured, placeEnd, startingEnds,
} from '../core/rotor3d_measure.js';
import { buildMeasure } from '../components/rotor3d_measure.js';
import { state } from '../core/state.js';
import {
    LEGEND_CATEGORIES, categoryHidden, drawn, elementHidden, legendClick, toggleElementHidden,
} from '../core/visibility.js';
import { renderList } from '../components/list.js';
import { buildBench, buildRotorModel } from '../components/rotor3d_parts.js';
import { buildTriad } from '../components/rotor3d_triad.js';
import { SHAPE_FIELD, drawnShape, shapeOf } from '../core/shapes3d.js';
import { openContextMenu } from '../components/context_menu.js';
import { addFrom3d, copyFrom3d, deleteFrom3d, editFrom3d, meshFrom3d, splitFrom3d } from './modeling.js';

const FOV = 35;

let library = null;
let loadingLibrary = null;
let stage = null;
let model = null;
let layout = null;
let direction = null;
let framedFor = null;
let framedFrom = null;
let bench = null;
let benchPlan = null;
let frameAsked = false;
// The parts drawn: the layout's, less what the legend and the list's eyes hid
// (core/visibility.js). The pointer finds only these.
let shownParts = [];
// Left-drag moves the view instead of turning it (the "move" button).
let panning = false;

// The test bench under the rotor (`benchLayout`), shown or not as the person
// left it -- like the language, the theme and the 2D/3D choice.
const BENCH_KEY = 'ross-rotor-bench';
let benchShown = rememberedBench();

function rememberedBench() {
    try {
        return localStorage.getItem(BENCH_KEY) === 'on';
    } catch (e) {
        return false;   // storage blocked: no bench until asked for
    }
}

// The node numbers and the dimensions (`annotationLayout`), shown or not as
// the person left them, like the bench.
const DIMENSIONS_KEY = 'ross-rotor-dimensions';
let dimensionsShown = rememberedDimensions();
let annotations = null;
let annotationPlan = null;
let labelSpans = [];

function rememberedDimensions() {
    try {
        return localStorage.getItem(DIMENSIONS_KEY) === 'on';
    } catch (e) {
        return false;
    }
}

// The measuring tape (core/rotor3d_measure.js): on or off, the line it runs
// on, its two ends, and the end being dragged (-1 for none). Not remembered:
// a measurement is taken and put away, it is not a way of looking at rotors.
let measuring = false;
let measure = null;
let tapeLine = null;
let tapeEnds = null;
let dragging = -1;

let pointer = null;

// Editing from the 3D view (slice 3). `chosen` is the key of the part last
// clicked -- its row's form is open -- and `listPointed` the part of the row
// the pointer is on in the element list.
let chosen = null;
let listPointed = null;
let clickTimer = null;

// How far a pointer may move between press and release and still be a click:
// more is the camera being turned.
const CLICK_SLOP = 5;
// A click waits this long to see whether it is the first of a double click.
const DOUBLE_CLICK = 250;

// `import()` of the bare name 'three' goes through the import map in
// index.html, which points it at `vendor/three/`. The OrbitControls addon
// imports 'three' by that same name, so both see one copy of the library.
function loadLibrary() {
    if (library) return Promise.resolve(library);
    if (!loadingLibrary) {
        loadingLibrary = Promise.all([
            import('three'),
            import('three/addons/controls/OrbitControls.js'),
            import('three/addons/environments/RoomEnvironment.js'),
        ]).then(([THREE, controls, room]) => {
            library = { THREE, OrbitControls: controls.OrbitControls, RoomEnvironment: room.RoomEnvironment };
            return library;
        }).finally(() => { loadingLibrary = null; });
    }
    return loadingLibrary;
}

// A colour of the page's theme, for what the 3D view draws in the interface's
// colours rather than ROSS's: the node rings, the outlines, the highlight and
// the axes.
function themeColor(name, fallback) {
    const value = window.getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

function rotorLook() {
    return { ring: themeColor('--accent', 'steelblue'), outline: themeColor('--text-strong', 'black') };
}

// The axes in the colours the figures give x, y and z (the plots' series).
function axisColours() {
    return { x: themeColor('--plot-red', 'firebrick'), y: themeColor('--plot-green', 'seagreen'), z: themeColor('--plot-blue', 'steelblue') };
}

// The theme changed (core/theme.js, subscribed in main.js). What wears its
// colours is repainted where it is -- WebGL keeps the colours it was given, and
// a view built in the light theme stayed light in the dark one until the next
// rebuild. ROSS's own colours, most of what is drawn, are not the page's.
export function restyleRotor3d() {
    if (!stage) return;
    const accent = themeColor('--accent', 'steelblue');
    stage.sleeve.material.color.set(accent);
    stage.mark.material.color.set(accent);
    stage.triad.restyle(axisColours());
    if (model) model.restyle(rotorLook());
    if (annotations) annotations.restyle(themeColor('--text-muted', 'gray'));
    if (measure) measure.restyle(accent);
    requestFrame();
}

// The axis triad (components/rotor3d_triad.js), drawn over the bottom-left
// corner by the same renderer after the rotor: a square of this many CSS
// pixels, this far from the edges.
const TRIAD_SIZE = 100;
const TRIAD_MARGIN = 8;

function inTriad(x, y, height) {
    return x >= TRIAD_MARGIN && x <= TRIAD_MARGIN + TRIAD_SIZE
        && y >= height - TRIAD_MARGIN - TRIAD_SIZE && y <= height - TRIAD_MARGIN;
}

function drawTriad() {
    const { renderer, camera, triad } = stage;
    const size = renderer.getSize(new library.THREE.Vector2());
    triad.orient(camera.quaternion);
    renderer.autoClear = false;
    renderer.setScissorTest(true);
    // Measured from the bottom, as WebGL counts.
    renderer.setViewport(TRIAD_MARGIN, TRIAD_MARGIN, TRIAD_SIZE, TRIAD_SIZE);
    renderer.setScissor(TRIAD_MARGIN, TRIAD_MARGIN, TRIAD_SIZE, TRIAD_SIZE);
    renderer.clearDepth();
    renderer.render(triad.scene, triad.camera);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, size.x, size.y);
    renderer.autoClear = true;
}

// How the rotor is lit. Metal looks like metal by what it reflects, so the
// scene is lit by a room -- three.js's RoomEnvironment, made once into an
// environment map -- with the tones mapped as a camera would. A light from
// above casts the rotor's shadow on a floor under it: without it, a rotor
// floats, and its supports read as being nowhere.
function light(stageScene, renderer) {
    const { THREE, RoomEnvironment } = library;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    // The sun stands still and so does the rotor while the camera turns: the
    // shadow is redrawn when what casts it changes (`placeFloor`), not on every
    // frame. Measured on the compressor: half of each frame's triangles were
    // the shadow's, drawn again for a picture that had not changed.
    renderer.shadowMap.autoUpdate = false;
    lightByRoom(stageScene, renderer);

    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    stageScene.add(sun, sun.target);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({ opacity: 0.22 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    stageScene.add(floor);
    return { sun, floor };
}

// The room the metal reflects, made into an environment map on the GPU.
function lightByRoom(stageScene, renderer) {
    const { THREE, RoomEnvironment } = library;
    if (stageScene.environment) stageScene.environment.dispose();
    const makeEnvironment = new THREE.PMREMGenerator(renderer);
    const room = new RoomEnvironment();
    stageScene.environment = makeEnvironment.fromScene(room, 0.04).texture;
    room.dispose();
    makeEnvironment.dispose();
}

// The floor under the rotor's box, and the light's shadow camera around it.
function placeFloor(current, bounds) {
    const { sun, floor } = current;
    const [x0, y0, z0] = bounds.min;
    const [x1, y1, z1] = bounds.max;
    const cx = (x0 + x1) / 2;
    const cz = (z0 + z1) / 2;
    const size = Math.max(x1 - x0, y1 - y0, z1 - z0);
    floor.position.set(cx, y0 - 0.02 * size, cz);
    floor.scale.set((x1 - x0) + size, (z1 - z0) + size, 1);
    sun.position.set(cx + 0.3 * size, y1 + 2 * size, cz + 0.2 * size);
    sun.target.position.set(cx, y0, cz);
    const shadow = sun.shadow.camera;
    const half = 0.75 * size;
    shadow.left = -half; shadow.right = half; shadow.top = half; shadow.bottom = -half;
    shadow.near = 0.1 * size; shadow.far = 5 * size;
    shadow.updateProjectionMatrix();
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.002 * size;
    // Called whenever what casts the shadow changed: the rotor was rebuilt,
    // the bench came or went.
    current.renderer.shadowMap.needsUpdate = true;
}

function makeStage(container) {
    const { THREE, OrbitControls } = library;
    // Throws where WebGL is off or missing; the caller says so on screen.
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.className = 'rotor3d-canvas';

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(FOV, 1, 0.001, 100);
    scene.add(camera);
    // A little light that rides with the camera, so the side you look at is
    // never in the dark.
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(0.5, 1, 0.3);
    camera.add(fill);
    const { sun, floor } = light(scene, renderer);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.addEventListener('change', requestFrame);
    // Moving the view: OrbitControls already pans with the right button and
    // with Shift, Ctrl or Cmd held on the left one; the arrow keys pan once it
    // listens to them. The "move" button (`togglePan`) swaps the left and right
    // buttons for whoever does not know the gestures, or has no right button.
    controls.screenSpacePanning = true;
    controls.listenToKeyEvents(renderer.domElement);
    setButtons(controls);

    // The highlight: a translucent sleeve over the part under the pointer. One
    // mesh, moved and scaled; it is never rebuilt.
    const sleeveGeometry = new THREE.CylinderGeometry(1, 1, 1, 48, 1, true);
    sleeveGeometry.rotateX(Math.PI / 2);
    const sleeve = new THREE.Mesh(sleeveGeometry, new THREE.MeshBasicMaterial({
        color: themeColor('--accent', 'steelblue'), transparent: true, opacity: 0.3,
        depthWrite: false, side: THREE.DoubleSide,
    }));
    sleeve.visible = false;
    scene.add(sleeve);
    // The part whose form is open wears a fainter one.
    const mark = new THREE.Mesh(sleeveGeometry, new THREE.MeshBasicMaterial({
        color: themeColor('--accent', 'steelblue'), transparent: true, opacity: 0.16,
        depthWrite: false, side: THREE.DoubleSide,
    }));
    mark.visible = false;
    scene.add(mark);

    const tip = document.createElement('div');
    tip.className = 'rotor3d-tip';
    tip.hidden = true;

    // The legend: one chip per category the rotor has, as in the 2D figure.
    const legend = document.createElement('div');
    legend.className = 'rotor3d-legend';
    legend.setAttribute('role', 'group');

    // The node numbers and dimensions, as text over the canvas
    // (components/rotor3d_annotations.js).
    const labels = document.createElement('div');
    labels.className = 'rotor3d-labels';
    labels.setAttribute('aria-hidden', 'true');

    // What the tape measures, over its middle.
    const measureLabel = document.createElement('div');
    measureLabel.className = 'rotor3d-measure';
    measureLabel.hidden = true;

    container.appendChild(renderer.domElement);
    container.appendChild(labels);
    container.appendChild(measureLabel);
    container.appendChild(tip);
    container.appendChild(legend);

    // The tape's ends are grabbed before OrbitControls sees the press: these
    // listen in the capture phase, which on the canvas itself runs ahead of
    // the controls' own listeners, and a press on a handle goes no further --
    // not to the controls, which would turn the view, nor to the click that
    // would open a part's form.
    renderer.domElement.addEventListener('pointerdown', event => {
        if (!measuring || event.button !== 0) return;
        const end = endUnder(event);
        if (end < 0) return;
        dragging = end;
        event.stopImmediatePropagation();
        event.preventDefault();
        try { renderer.domElement.setPointerCapture(event.pointerId); } catch (e) { /* not a live pointer */ }
        showGrab();
    }, { capture: true });
    renderer.domElement.addEventListener('pointermove', event => {
        if (dragging < 0) return;
        event.stopImmediatePropagation();
        dragEnd(event);
    }, { capture: true });
    const letGo = event => {
        if (dragging < 0) return;
        event.stopImmediatePropagation();
        dragging = -1;
        try { renderer.domElement.releasePointerCapture(event.pointerId); } catch (e) { /* already released */ }
        showGrab(event);
        requestFrame();
    };
    renderer.domElement.addEventListener('pointerup', letGo, { capture: true });
    renderer.domElement.addEventListener('pointercancel', letGo, { capture: true });

    renderer.domElement.addEventListener('pointermove', event => {
        showGrab(event);
        const box = renderer.domElement.getBoundingClientRect();
        pointer = { x: event.clientX - box.left, y: event.clientY - box.top, width: box.width, height: box.height };
        requestFrame();
    });
    renderer.domElement.addEventListener('pointerleave', () => {
        pointer = null;
        requestFrame();
    });

    // A click edits the part (its row's form, as the list's pencil opens it);
    // a double click adds an element at the nearest node (the node hub, as the
    // 2D figure's "+" opens it); the Delete key removes the part last clicked.
    // The canvas takes the keyboard focus on a click, so Delete typed in the
    // form is never read here.
    renderer.domElement.tabIndex = 0;
    let pressed = null;
    // The right button: released where it was pressed, it opens the menu of
    // what is under it (`openMenu`); dragged, it moves the view, as before --
    // the same slop that tells a click from a turn with the left one.
    let rightPressed = null;
    renderer.domElement.addEventListener('pointerdown', event => {
        pressed = event.button === 0 ? { x: event.clientX, y: event.clientY } : null;
        rightPressed = event.button === 2 ? { x: event.clientX, y: event.clientY } : null;
    });
    renderer.domElement.addEventListener('contextmenu', event => event.preventDefault());
    renderer.domElement.addEventListener('pointerup', event => {
        if (event.button === 2) {
            const still = rightPressed && Math.hypot(event.clientX - rightPressed.x, event.clientY - rightPressed.y) <= CLICK_SLOP;
            rightPressed = null;
            if (still) openMenu(pickAt(event), event.clientX, event.clientY);
            return;
        }
        if (!pressed || Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y) > CLICK_SLOP) return;
        const hit = pickAt(event);
        clearTimeout(clickTimer);
        clickTimer = setTimeout(() => { if (hit) editPart(hit.part); }, DOUBLE_CLICK);
    });
    renderer.domElement.addEventListener('dblclick', event => {
        clearTimeout(clickTimer);
        const hit = pickAt(event);
        if (!hit) return;
        const node = nodeUnder(hit);
        if (node !== null) addFrom3d(node, hit.part.half);
    });
    renderer.domElement.addEventListener('keydown', event => {
        // The menu from the keyboard, for the part last clicked, at the
        // middle of the view.
        if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) {
            event.preventDefault();
            const part = chosenPart();
            const box = renderer.domElement.getBoundingClientRect();
            const point = part ? [part.offset.x, part.offset.y, part.offset.z + (part.z0 + part.z1) / 2] : null;
            openMenu(part ? { part, point } : null, box.left + box.width / 2, box.top + box.height / 2);
            return;
        }
        if (event.key === 'Escape' && measuring) {
            event.preventDefault();
            toggleMeasure();
            return;
        }
        if (event.key !== 'Delete') return;
        const part = chosenPart();
        if (!part) return;
        event.preventDefault();
        chosen = null;
        deleteFrom3d(part.category, part.index, part.half);
        requestFrame();
    });

    // The browser may take the WebGL context away -- the graphics driver
    // reset, the GPU's memory wanted elsewhere -- and give it back. three.js
    // uploads the geometries again by itself; what was drawn on the GPU, the
    // environment map and the shadow map, has to be made again, and a view
    // that draws only when something changes has to be told to draw. Without
    // this the canvas stayed as the loss left it until the pointer moved over
    // it, and dull where it used to shine.
    renderer.domElement.addEventListener('webglcontextrestored', () => {
        lightByRoom(scene, renderer);
        renderer.shadowMap.needsUpdate = true;
        requestFrame();
    });

    const fresh = {
        container, renderer, scene, camera, controls, sleeve, mark, tip, sun, floor, legend, labels, measureLabel,
        triad: buildTriad(THREE, axisColours()),
        raycaster: new THREE.Raycaster(), hovered: null,
    };
    if (typeof ResizeObserver === 'function') {
        new ResizeObserver(() => { fitCanvas(fresh); requestFrame(); }).observe(container);
    }
    return fresh;
}

function fitCanvas(current) {
    const width = current.container.clientWidth;
    const height = current.container.clientHeight;
    if (!width || !height) return false;
    current.renderer.setSize(width, height, false);
    current.renderer.domElement.style.width = '100%';
    current.renderer.domElement.style.height = '100%';
    current.camera.aspect = width / height;
    current.camera.updateProjectionMatrix();
    return true;
}

// The camera is put back only when the rotor changed size noticeably: editing
// one disk should not undo the angle the person was looking from.
// What the camera and the floor have to take in: the rotor, and the bench
// when it is shown.
// And the dimensions over it, when they are shown.
// And the tape, while measuring.
function shownBounds() {
    const bounds = benchShown && benchPlan ? benchPlan.bounds : layout.bounds;
    let top = bounds.max[1];
    if (dimensionsShown && annotationPlan) top = Math.max(top, annotationPlan.top);
    if (measuring && tapeLine) top = Math.max(top, measureTop(tapeLine));
    if (!(top > bounds.max[1])) return bounds;
    return { min: bounds.min, max: [bounds.max[0], top, bounds.max[2]] };
}

function needsFraming(bounds) {
    if (!framedFor) return true;
    const size = b => [0, 1, 2].map(i => b.max[i] - b.min[i]);
    const before = size(framedFor);
    const after = size(bounds);
    return after.some((value, i) => Math.abs(value - before[i]) > 0.25 * Math.max(before[i], 1e-6));
}

export function frameRotor() {
    if (!stage || !layout) return;
    const { camera, controls } = stage;
    const view = framing(shownBounds(), FOV, camera.aspect, direction);
    const [cx, cy, cz] = view.center;
    const [dx, dy, dz] = view.direction;
    const d = view.distance;
    camera.position.set(cx + dx * d, cy + dy * d, cz + dz * d);
    camera.near = Math.max(view.distance / 200, 1e-4);
    camera.far = view.distance * 20;
    camera.updateProjectionMatrix();
    controls.target.set(cx, cy, cz);
    controls.update();
    framedFor = shownBounds();
    framedFrom = String(direction);
    requestFrame();
}

// Draw `scene` in `container`. The first call loads three.js; a browser
// without WebGL rejects, and the caller shows why.
export async function showRotor3d(container, scene) {
    await loadLibrary();
    if (!stage || stage.container !== container) {
        stage = makeStage(container);
        framedFor = null;
    }
    fitCanvas(stage);
    layout = layoutScene(scene, shapeForElement);
    direction = viewDirection(scene);
    buildModel();
    placeAnnotations();
    placeMeasure();
    placeBench();
    stage.hovered = null;
    if (needsFraming(shownBounds()) || framedFrom !== String(direction)) frameRotor();
    requestFrame();
}

// The model of what is shown: the layout's parts less the hidden ones. The
// camera, the floor and the bench keep to the whole rotor, so hiding a
// category does not make the view jump.
function buildModel() {
    if (model) {
        stage.scene.remove(model.object);
        model.dispose();
        model = null;
    }
    shownParts = layout.parts.filter(part => drawn(part.category, elementOf(part)));
    const rings = categoryHidden('nodes') ? [] : layout.rings;
    model = buildRotorModel(library.THREE, { parts: shownParts, rings, bounds: layout.bounds }, rotorLook());
    stage.scene.add(model.object);
    // A hidden part takes its shadow with it.
    stage.renderer.shadowMap.needsUpdate = true;
    renderLegend();
}

// The shape of the geometry bank an element is drawn with: the one given in
// its form, or what its model says (`entry`, the scene's), or null.
function shapeForElement(half, category, index, entry) {
    const element = elementOf({ half, category, index });
    return drawnShape(category, element && element[SHAPE_FIELD], entry);
}

// The project's element a part was drawn from.
function elementOf(part) {
    const project = state.projectData;
    if (!project) return null;
    const line = project.isMultiRotor ? project[`${part.half}_rotor`] : project;
    return line && line[part.category] ? line[part.category][part.index] || null : null;
}

function presentCategories() {
    if (!layout) return [];
    return LEGEND_CATEGORIES.filter(c => (c === 'nodes' ? layout.rings.length > 0 : layout.parts.some(p => p.category === c)));
}

function renderLegend() {
    const chips = presentCategories().map(category => {
        const parts = layout.parts.filter(p => p.category === category);
        const shown = parts.filter(p => !elementHidden(elementOf(p))).length;
        const count = category === 'nodes' ? ''
            : `<span class="rotor3d-chip-count">${shown === parts.length ? shown : `${shown}/${parts.length}`}</span>`;
        const swatch = category === 'nodes' ? 'var(--accent)' : escapeHtml(String(parts[0].color || 'gray'));
        return `<button type="button" class="rotor3d-chip" data-action="hide-category" data-category="${category}"`
            + ` aria-pressed="${String(!categoryHidden(category))}" title="${escapeHtml(t('rotor3dLegendHint'))}">`
            + `<span class="rotor3d-swatch" style="background:${swatch}"></span>`
            + `${escapeHtml(legendName(category))}${count}</button>`;
    });
    stage.legend.innerHTML = chips.join('');
}

// A change of language: the legend's names are written here, not in the HTML.
export function relabelRotor3d() {
    if (stage && layout) renderLegend();
}

function legendName(category) {
    return category === 'nodes' ? t('rotor3dNodes') : (CATEGORY_NAMES[category] || (() => category))();
}

function showWhatIsHidden() {
    if (!stage || !layout) return;
    buildModel();
    stage.hovered = null;
    requestFrame();
}

// The legend's chip: a click hides or shows the category, a double click
// shows it alone (or all again), as in Plotly's legend (`legendClick`).
export function toggleCategory3d(category, event) {
    legendClick(category, event ? event.detail : 1, presentCategories());
    showWhatIsHidden();
    redrawOpenList();
}

// The eye on a row of the list: that one element, in or out of the 3D view.
export function toggleElement3d(element) {
    toggleElementHidden(element);
    showWhatIsHidden();
    redrawOpenList();
}

// The list shows which rows are out of view; with no tab open there is none.
function redrawOpenList() {
    if (state.currentTab && state.projectData) renderList();
}

// The "move" button: which mouse button turns the view and which moves it.
function setButtons(controls) {
    const { MOUSE } = library.THREE;
    controls.mouseButtons = panning
        ? { LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE }
        : { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };
    showPanButton();
}

function showPanButton() {
    document.querySelectorAll('[data-action="toggle-pan"]').forEach(button => {
        button.setAttribute('aria-pressed', String(panning));
    });
}

export function togglePan() {
    panning = !panning;
    if (stage) setButtons(stage.controls);
    else showPanButton();
}

// The bench follows the rotor: rebuilt with it, or taken away.
function placeBench() {
    if (bench) {
        stage.scene.remove(bench.object);
        bench.dispose();
        bench = null;
    }
    benchPlan = benchLayout(layout);
    if (benchShown) {
        bench = buildBench(library.THREE, benchPlan);
        stage.scene.add(bench.object);
    }
    placeFloor(stage, shownBounds());
    showBenchButton();
}

function showBenchButton() {
    document.querySelectorAll('[data-action="toggle-bench"]').forEach(button => {
        button.setAttribute('aria-pressed', String(benchShown));
    });
}

// The bench button: on or off, remembered, and the camera takes in the new
// whole -- the bench adds a table and its legs under the rotor.
export function toggleBench() {
    benchShown = !benchShown;
    try {
        localStorage.setItem(BENCH_KEY, benchShown ? 'on' : 'off');
    } catch (e) { /* a preference: without storage it lasts until the page closes */ }
    showBenchButton();
    if (!stage || !layout) return;
    placeBench();
    frameRotor();
}

// While the 2D figure is on screen the 3D view keeps its renderer and its
// rotor, and only stops reacting to a pointer it can no longer see.
export function hideRotor3d() {
    pointer = null;
    dragging = -1;
    markRow(null);
    if (stage) {
        stage.tip.hidden = true;
        stage.sleeve.visible = false;
    }
}

// The part under a pointer event, with where the ray met it.
function pickAt(event) {
    if (!stage || !layout) return null;
    const box = stage.renderer.domElement.getBoundingClientRect();
    if (!box.width || !box.height) return null;
    // The triad's corner is not a window onto the rotor.
    if (inTriad(event.clientX - box.left, event.clientY - box.top, box.height)) return null;
    const ndc = {
        x: ((event.clientX - box.left) / box.width) * 2 - 1,
        y: -((event.clientY - box.top) / box.height) * 2 + 1,
    };
    stage.raycaster.setFromCamera(ndc, stage.camera);
    const { origin, direction } = stage.raycaster.ray;
    const from = [origin.x, origin.y, origin.z];
    const toward = [direction.x, direction.y, direction.z];
    const hit = pickPart(shownParts, from, toward);
    return hit ? { ...hit, point: hitPoint(from, toward, hit.distance) } : null;
}

// The node nearest to where the pointer is on a part, in that part's line.
function nodeUnder(hit) {
    return nearestNode(layout.rings, hit.part.half, hit.point[2]);
}

// The part last clicked, while its row's form is still the one open: once
// the form is saved or closed, or another row is edited, nothing is chosen.
function chosenPart() {
    const part = chosen && layout ? layout.parts.find(p => p.key === chosen) : null;
    const halfOpen = !(state.projectData && state.projectData.isMultiRotor) || state.multiRotorEditTarget === (part && part.half);
    if (part && state.currentTab === part.category && state.editingIndex === part.index && halfOpen) return part;
    chosen = null;
    return null;
}

function editPart(part) {
    chosen = part.key;
    editFrom3d(part.category, part.index, part.half);
    requestFrame();
}

// --- the menu of the right button ----------------------------------------------------
//
// What the list does to a row, done to the part under the pointer -- and for a
// shaft, what only the pointer can say: where to cut it. `hit` is null on the
// empty view, which offers what concerns the whole rotor.

function fill(template, ...values) {
    // One pass over the template: a value that itself holds "%2" is shown as
    // typed, rather than filled in by the next placeholder.
    return template.replace(/%(\d)/g, (marker, digit) =>
        (digit >= 1 && digit <= values.length ? String(values[digit - 1]) : marker));
}

// Metres in the units a length can be typed in; anything else is not offered
// a cut "here", rather than cut at a guess.
const METRES_IN = { mm: 1000, cm: 100, m: 1, in: 1 / 0.0254, inch: 1 / 0.0254, ft: 1 / 0.3048 };

// How far along a shaft the pointer is, from its left face: shown in mm, and
// sent in the unit its length is typed in (`splitProjectAt`). None within 1 %
// of either face, where a cut would land on the node that is already there.
export function cutHere(part, point, element) {
    if (!part || part.kind !== 'shaft' || !point) return null;
    const from = Math.min(part.z0, part.z1);
    const length = Math.abs(part.z1 - part.z0);
    const along = point[2] - part.offset.z - from;
    if (!(length > 0) || along < 0.01 * length || along > 0.99 * length) return null;
    const unit = String((element && element.L_unit) || 'mm').trim();
    const factor = METRES_IN[unit];
    if (!factor) return null;
    return { shown: `${(along * 1000).toFixed(1)} mm`, value: String(Number((along * factor).toPrecision(6))) };
}

export function menuEntries(hit) {
    if (!hit) {
        return [
            { label: t('ctxMeshAll'), icon: 'fa-ruler-horizontal', run: () => meshFrom3d(null, currentHalf()) },
            { label: measuring ? t('ctxMeasureStop') : t('ctxMeasure'), icon: 'fa-ruler', run: () => toggleMeasure() },
            { label: t('rotor3dFrame'), icon: 'fa-expand', run: () => frameRotor() },
        ];
    }
    const { part } = hit;
    const element = elementOf(part);
    const name = (CATEGORY_NAMES[part.category] || (() => part.category))();
    const tag = element && String(element.tag || '').trim();
    const entries = [{ header: `${name} #${part.index + 1}${tag ? ' · ' + tag : ''}` }];
    if (part.kind === 'shaft') {
        const here = cutHere(part, hit.point, element);
        entries.push(here
            ? { label: fill(t('ctxSplitHere'), here.shown), icon: 'fa-scissors', run: () => splitFrom3d(part.index, part.half, here.value) }
            : { label: t('ctxSplitHereNo'), icon: 'fa-scissors', disabled: true });
        entries.push({ label: t('ctxSplitAsk'), icon: 'fa-scissors', run: () => splitFrom3d(part.index, part.half) });
        entries.push({ label: t('ctxMesh'), icon: 'fa-ruler-horizontal', run: () => meshFrom3d(part.index, part.half) });
        entries.push(null);
    }
    entries.push({ label: t('ctxEdit'), icon: 'fa-pen', run: () => editPart(part) });
    entries.push({ label: t('ctxCopy'), icon: 'fa-copy', run: () => copyFrom3d(part.category, part.index, part.half) });
    entries.push({ label: t('rotor3dHideElement'), icon: 'fa-eye-slash', disabled: !element, run: () => toggleElement3d(element) });
    const node = hit.point && layout ? nodeUnder(hit) : null;
    if (node !== null) entries.push({ label: fill(t('ctxAddAtNode'), node), icon: 'fa-plus', run: () => addFrom3d(node, part.half) });
    if (node !== null && !part.hanging) entries.push({ label: fill(t('ctxMeasureFrom'), node), icon: 'fa-ruler', run: () => measureFrom(node, part.half) });
    entries.push(null);
    entries.push({ label: t('ctxDelete'), icon: 'fa-trash', danger: true, run: () => deleteFrom3d(part.category, part.index, part.half) });
    return entries;
}

// The line being edited, on a MultiRotor; none on a rotor.
function currentHalf() {
    return state.projectData && state.projectData.isMultiRotor ? state.multiRotorEditTarget : null;
}

function openMenu(hit, x, y) {
    if (!layout) return;
    clearTimeout(clickTimer);
    openContextMenu(x, y, menuEntries(hit));
}

// The row of a part, when the list is showing it: same tab, same MultiRotor half.
function rowOf(part) {
    if (!part || state.currentTab !== part.category) return null;
    if (state.projectData && state.projectData.isMultiRotor && state.multiRotorEditTarget !== part.half) return null;
    const rows = document.querySelectorAll('#element-list .list-item');
    return rows[part.index] || null;
}

let pointedRow = null;

// The list row of the part under the pointer is marked in the list.
function markRow(part) {
    const row = rowOf(part);
    if (row === pointedRow) return;
    if (pointedRow) pointedRow.classList.remove('is-pointed');
    if (row) row.classList.add('is-pointed');
    pointedRow = row;
}

// The other way: the pointer on a row of the element list marks its part in
// the 3D view. Delegated from the document, since the list is rebuilt on
// every change.
export function startRotor3dListLink() {
    document.addEventListener('mouseover', event => {
        const row = event.target && event.target.closest ? event.target.closest('#element-list .list-item') : null;
        const found = row ? row.querySelector('[data-index]') : null;
        let key = null;
        if (found && state.currentTab) {
            const half = state.projectData && state.projectData.isMultiRotor ? state.multiRotorEditTarget : null;
            key = (half ? `${half}:` : '') + `${state.currentTab}:${found.dataset.index}`;
        }
        if (key !== listPointed) {
            listPointed = key;
            requestFrame();
        }
    });
}

function placeSleeve(sleeve, part, grow) {
    const o = part.offset;
    const length = Math.abs(part.z1 - part.z0);
    sleeve.position.set(o.x, o.y, o.z + (part.z0 + part.z1) / 2);
    sleeve.scale.set(part.radius * grow, part.radius * grow, length + part.radius * 0.08);
    sleeve.visible = true;
}

function requestFrame() {
    if (frameAsked || !stage) return;
    frameAsked = true;
    requestAnimationFrame(drawFrame);
}

function drawFrame() {
    frameAsked = false;
    if (!stage) return;
    showWhatIsUnderThePointer();
    stage.renderer.render(stage.scene, stage.camera);
    drawTriad();
    placeLabels();
    placeMeasureLabel();
}

// The numbers follow the view: projected every frame, those that would print
// over each other left out (`arrangeLabels`).
function placeLabels() {
    if (!annotations || !labelSpans.length) return;
    const { THREE } = library;
    const box = stage.renderer.domElement.getBoundingClientRect();
    const width = box.width;
    const height = box.height;
    const point = new THREE.Vector3();
    const project = ([x, y, z]) => {
        point.set(x, y, z).project(stage.camera);
        return [(point.x + 1) / 2 * width, (1 - point.y) / 2 * height, point.z];
    };
    arrangeLabels(annotations.labels, project, width, height).forEach((place, k) => {
        const span = labelSpans[k];
        span.hidden = !place.shown;
        if (place.shown) span.style.transform = `translate(${place.x.toFixed(1)}px, ${place.y.toFixed(1)}px)`;
    });
}

// Built with the rotor, or taken away.
function placeAnnotations() {
    if (annotations) {
        stage.scene.remove(annotations.object);
        annotations.dispose();
        annotations = null;
    }
    annotationPlan = annotationLayout(layout);
    labelSpans = [];
    stage.labels.innerHTML = '';
    if (dimensionsShown) {
        annotations = buildAnnotations(library.THREE, annotationPlan, themeColor('--text-muted', 'gray'));
        stage.scene.add(annotations.object);
        labelSpans = fillLabels(stage.labels, annotations.labels);
    }
    showDimensionsButton();
}

function showDimensionsButton() {
    document.querySelectorAll('[data-action="toggle-dimensions"]').forEach(button => {
        button.setAttribute('aria-pressed', String(dimensionsShown));
    });
}

// The button: on or off, remembered, and the view takes in the dimensions
// over the rotor.
export function toggleDimensions() {
    dimensionsShown = !dimensionsShown;
    try {
        localStorage.setItem(DIMENSIONS_KEY, dimensionsShown ? 'on' : 'off');
    } catch (e) { /* a preference: without storage it lasts until the page closes */ }
    showDimensionsButton();
    if (!stage || !layout) return;
    placeAnnotations();
    frameRotor();
}

// --- the measuring tape ------------------------------------------------------------
//
// Where it goes is core/rotor3d_measure.js's; here it is drawn, dragged and
// labelled. It measures the line being edited (the driving or the driven one
// on a MultiRotor), or the one it was started on from the menu.

let tapeHalf = null;

// Built with the rotor, or taken away. The ends stay on their nodes through a
// rebuild -- an element added or edited -- and start over on another line.
function placeMeasure() {
    if (measure) {
        stage.scene.remove(measure.object);
        measure.dispose();
        measure = null;
    }
    dragging = -1;
    tapeLine = measuring && layout ? measureLine(layout, annotationPlan, tapeHalf) : null;
    if (measuring && !tapeLine) measuring = false;
    if (tapeLine) {
        const ends = tapeEnds && tapeEnds.half === tapeHalf ? keptEnds(tapeEnds.ends, tapeLine) : startingEnds(tapeLine);
        tapeEnds = { half: tapeHalf, ends };
        measure = buildMeasure(library.THREE, tapeLine, tapeEnds.ends, themeColor('--accent', 'steelblue'));
        stage.scene.add(measure.object);
    } else {
        tapeEnds = null;
    }
    if (stage.measureLabel) stage.measureLabel.hidden = !tapeLine;
    showMeasureButton();
}

function showMeasureButton() {
    document.querySelectorAll('[data-action="toggle-measure"]').forEach(button => {
        button.setAttribute('aria-pressed', String(measuring));
    });
}

// The button: the tape from end to end of the line being edited, or put away.
export function toggleMeasure() {
    measuring = !measuring;
    tapeHalf = currentHalf();
    tapeEnds = null;
    showMeasureButton();
    if (!stage || !layout) return;
    placeMeasure();
    requestFrame();
}

// The menu's "measure from node n": the tape from that node, on its line.
export function measureFrom(node, half = null) {
    measuring = true;
    tapeHalf = half;
    tapeEnds = null;
    if (!stage || !layout) {
        showMeasureButton();
        return;
    }
    placeMeasure();
    if (tapeLine) {
        tapeEnds = { half, ends: startingEnds(tapeLine, node) };
        measure.update(tapeLine, tapeEnds.ends);
    }
    requestFrame();
}

// The ray under a pointer event, in the scene.
function rayAt(event) {
    const box = stage.renderer.domElement.getBoundingClientRect();
    if (!box.width || !box.height) return null;
    const ndc = {
        x: ((event.clientX - box.left) / box.width) * 2 - 1,
        y: -((event.clientY - box.top) / box.height) * 2 + 1,
    };
    stage.raycaster.setFromCamera(ndc, stage.camera);
    const { origin, direction } = stage.raycaster.ray;
    return { from: [origin.x, origin.y, origin.z], toward: [direction.x, direction.y, direction.z] };
}

// Each end on screen: [x, y, depth] in CSS pixels.
function endsOnScreen() {
    if (!tapeLine || !tapeEnds) return [];
    const box = stage.renderer.domElement.getBoundingClientRect();
    const point = new library.THREE.Vector3();
    return tapeEnds.ends.map(end => {
        point.set(tapeLine.x, tapeLine.tape, end.z).project(stage.camera);
        return [(point.x + 1) / 2 * box.width, (1 - point.y) / 2 * box.height, point.z];
    });
}

function endUnder(event) {
    if (!measuring || !tapeLine) return -1;
    const box = stage.renderer.domElement.getBoundingClientRect();
    return grabbedEnd({ x: event.clientX - box.left, y: event.clientY - box.top }, endsOnScreen());
}

function overHandle() {
    return measuring && !!tapeLine && !!pointer && (dragging >= 0 || grabbedEnd(pointer, endsOnScreen()) >= 0);
}

function dragEnd(event) {
    const ray = rayAt(event);
    const z = ray ? alongAxis(ray.from, ray.toward, tapeLine) : null;
    if (z === null) return;
    tapeEnds.ends[dragging] = placeEnd(z, tapeLine, event.shiftKey);
    measure.update(tapeLine, tapeEnds.ends);
    requestFrame();
}

// The cursor says an end can be grabbed, and that it is.
function showGrab(event) {
    const canvas = stage && stage.renderer.domElement;
    if (!canvas) return;
    const over = dragging < 0 && event ? endUnder(event) >= 0 : false;
    canvas.classList.toggle('is-grab', over);
    canvas.classList.toggle('is-grabbing', dragging >= 0);
}

// The distance over the middle of the tape, and the nodes when both ends are
// on one. Text of the page, as the dimensions are.
function placeMeasureLabel() {
    const label = stage.measureLabel;
    if (!label) return;
    if (!tapeLine || !tapeEnds) {
        label.hidden = true;
        return;
    }
    const { text, nodes } = measured(tapeEnds.ends);
    const content = nodes ? `${text} · ${fill(t('measureNodes'), nodes[0], nodes[1])}` : text;
    if (label.textContent !== content) label.textContent = content;
    const box = stage.renderer.domElement.getBoundingClientRect();
    const middle = (tapeEnds.ends[0].z + tapeEnds.ends[1].z) / 2;
    const point = new library.THREE.Vector3(tapeLine.x, tapeLine.tape, middle).project(stage.camera);
    const inView = point.z > -1 && point.z < 1;
    label.hidden = !inView;
    if (inView) {
        const x = (point.x + 1) / 2 * box.width;
        const y = (1 - point.y) / 2 * box.height;
        label.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    }
}

function showWhatIsUnderThePointer() {
    const { sleeve, tip, raycaster, camera } = stage;
    const overTriad = !!pointer && inTriad(pointer.x, pointer.y, pointer.height);
    let hit = null;
    if (pointer && !overTriad && layout && pointer.width && pointer.height) {
        const ndc = { x: (pointer.x / pointer.width) * 2 - 1, y: -(pointer.y / pointer.height) * 2 + 1 };
        raycaster.setFromCamera(ndc, camera);
        const { origin, direction } = raycaster.ray;
        hit = pickPart(shownParts, [origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z]);
    }
    const part = hit ? hit.part : null;
    const find = key => (key && layout ? layout.parts.find(p => p.key === key) : null);
    const marked = chosenPart();
    if (marked) placeSleeve(stage.mark, marked, 1.14); else stage.mark.visible = false;
    if (overTriad) {
        showAxesTip();
        return;
    }
    markRow(part);
    // A handle of the tape is over whatever part is behind it.
    if (!part || overHandle()) {
        tip.hidden = true;
        stage.hovered = null;
        if (part) {
            sleeve.visible = false;
            markRow(null);
            return;
        }
        // Nothing under the pointer here: the part of the list row under it,
        // if any, is the one lit.
        const fromList = find(listPointed);
        if (fromList) placeSleeve(sleeve, fromList, 1.08); else sleeve.visible = false;
        return;
    }
    placeSleeve(sleeve, part, 1.08);
    const { origin, direction } = raycaster.ray;
    const at = hitPoint([origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z], hit.distance);
    const node = nodeUnder({ part, point: at });
    const tipKey = `${part.key}@${node}`;
    if (stage.hovered !== tipKey) {
        tip.innerHTML = describePart(part)
            + `<div class="rotor3d-tip-hint">${escapeHtml(t('rotor3dHint').replace('%1', () => node))}</div>`;
        stage.hovered = tipKey;
    }
    placeTip();
}

// Over the triad: what the axes are, and no part.
function showAxesTip() {
    const { sleeve, tip } = stage;
    sleeve.visible = false;
    markRow(null);
    if (stage.hovered !== 'axes') {
        tip.innerHTML = `<div class="rotor3d-tip-title">${escapeHtml(t('rotor3dAxesTitle'))}</div>`
            + `<div>${escapeHtml(t('rotor3dAxes'))}</div>`;
        stage.hovered = 'axes';
    }
    placeTip();
}

// Beside the pointer, and turned back inside when it would leave the view.
function placeTip() {
    const { tip } = stage;
    tip.hidden = false;
    const left = pointer.x + 16 + tip.offsetWidth > pointer.width ? pointer.x - 16 - tip.offsetWidth : pointer.x + 16;
    const top = Math.min(pointer.y + 16, Math.max(0, pointer.height - tip.offsetHeight - 4));
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${Math.max(0, top)}px`;
}

// --- what the tooltip says -----------------------------------------------------

const CATEGORY_NAMES = {
    shafts: () => t('catShaft'),
    disks: () => t('catDisk'),
    gears: () => t('catGear'),
    couplings: () => t('catCoupling'),
    bearings: () => t('catBearing'),
    seals: () => t('catSeal'),
    pointmasses: () => t('catPointMassLong'),
    probes: () => t('catProbe'),
};

const mm = metres => `${(metres * 1000).toFixed(1)} mm`;
const number = value => (Math.abs(value) >= 1000 || Math.abs(value) < 0.01 ? value.toExponential(3) : value.toPrecision(4));

function line(label, value) {
    return `<div><span class="rotor3d-tip-label">${escapeHtml(label)}</span> ${escapeHtml(value)}</div>`;
}

function note(text) {
    return `<div class="rotor3d-tip-note">${escapeHtml(text)}</div>`;
}

function halfName(part) {
    if (part.half === 'driving') return t('multiDriving');
    if (part.half === 'driven') return t('multiDriven');
    return '';
}

// The row's own words: its position in the list (from 1, as the list numbers
// it), its tag, its node -- and the few numbers that say what was drawn and
// why it is that size.
function describePart(part) {
    const e = part.entry;
    const name = (CATEGORY_NAMES[part.category] || (() => part.category))();
    const title = `${name} #${part.index + 1}` + (e.tag ? ` · ${e.tag}` : '');
    const rows = [];
    const half = halfName(part);
    if (half) rows.push(line(t('rotor3dLine'), half));
    rows.push(line(t('rotor3dNode'), String(e.n)));
    if (part.kind === 'shaft') {
        rows.push(line(t('rotor3dLength'), mm(Math.abs(part.z1 - part.z0))));
        const { odl, odr, idl, idr } = part.profile;
        rows.push(line(t('rotor3dOuter'), odl === odr ? mm(odl) : `${mm(odl)} → ${mm(odr)}`));
        if (idl || idr) rows.push(line(t('rotor3dInner'), idl === idr ? mm(idl) : `${mm(idl)} → ${mm(idr)}`));
    } else if (part.kind === 'disk') {
        rows.push(line(t('rotor3dMass'), `${number(e.m)} kg`));
        rows.push(line('Ip', `${number(e.Ip)} kg·m²`));
        if (e.shape) {
            rows.push(line(t('rotor3dOuter'), mm(2 * e.shape.outer_radius)));
            rows.push(line(t('rotor3dWidth'), mm(e.shape.width)));
        }
        rows.push(`<div class="rotor3d-tip-note">${escapeHtml(
            !e.shape ? t('rotor3dNoSize') : e.shape.exact ? t('rotor3dExactDisk') : t('rotor3dEquivalentDisk'))}</div>`);
    } else if (part.kind === 'gear') {
        rows.push(line(t('rotor3dMass'), `${number(e.m)} kg`));
        if (e.teeth) rows.push(line(t('rotor3dTeeth'), String(e.teeth)));
        if (e.pitch_radius) rows.push(line(t('rotor3dPitch'), mm(2 * e.pitch_radius)));
        if (e.width) rows.push(line(t('rotor3dWidth'), mm(e.width)));
        if (!e.width) rows.push(note(t('rotor3dAssumedWidth')));
        else if (e.width_is_equivalent) rows.push(note(t('rotor3dEquivalentWidth')));
        if (part.gear && part.gear.onShaft) rows.push(note(t('rotor3dPinionOnShaft')));
    } else if (part.kind === 'coupling') {
        rows.push(line(t('rotor3dLength'), mm(Math.abs(part.z1 - part.z0))));
        if (e.outer_diameter) rows.push(line(t('rotor3dOuter'), mm(e.outer_diameter)));
        if (e.m_l != null) rows.push(line(t('rotor3dMassLeft'), `${number(e.m_l)} kg`));
        if (e.m_r != null) rows.push(line(t('rotor3dMassRight'), `${number(e.m_r)} kg`));
        rows.push(note(t('rotor3dCouplingHubs')));
        if (part.overlapsShaft) rows.push(note(t('rotor3dCouplingOverShaft')));
    } else if (part.kind === 'probe') {
        rows.push(line(t('rotor3dDirection'), e.direction === 'axial' ? t('rotor3dProbeAxial')
            : fill(t('rotor3dProbeRadial'), ((e.angle || 0) * 180 / Math.PI).toFixed(0))));
        if (part.beside) rows.push(note(t('rotor3dProbeBeside')));
        rows.push(note(t('rotor3dProbeNote')));
    } else if (part.kind === 'pointmass') {
        rows.push(line(t('rotor3dMass'), `${number(e.m)} kg`));
        rows.push(`<div class="rotor3d-tip-note">${escapeHtml(t('rotor3dNoSize'))}</div>`);
    } else {
        rows.push(line(t('rotor3dModel'), e.kind || ''));
        rows.push(`<div class="rotor3d-tip-note">${escapeHtml(t('rotor3dNoSize'))}</div>`);
    }
    // A shape from the geometry bank: what it is, whether the form chose it or
    // the model says it, and that it is only drawn. The category's own
    // drawing, when the model says it, goes without saying.
    const drawnAs = shapeOf(part.category, part.shape);
    const element = elementOf(part);
    const chosen = !!shapeOf(part.category, element && element[SHAPE_FIELD]);
    if (drawnAs && (chosen || !drawnAs.plain)) {
        rows.push(line(t('rotor3dShape'), chosen ? drawnAs.name() : `${drawnAs.name()} ${t('rotor3dShapeByModel')}`));
        if (part.helix) rows.push(line(t('rotor3dHelix'), `${(Math.abs(part.helix) * 180 / Math.PI).toFixed(1)}°`));
        if (part.helixAssumed) rows.push(note(t('rotor3dHelixAssumed')));
        rows.push(note(t('rotor3dShapeOnlyPicture')));
    }
    if (part.hanging) rows.push(`<div class="rotor3d-tip-note">${escapeHtml(t('rotor3dLinkNode'))}</div>`);
    return `<div class="rotor3d-tip-title">${escapeHtml(title)}</div>${rows.join('')}`;
}
