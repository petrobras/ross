// A shaft element drawn in the 3D view: a sketch of one or two concentric
// circles on the plane of a cross-section, then an extrusion along the rotor
// axis (Leonardo's proposal, after the way a CAD program builds a solid).
//
// It is a second way into the element list, not a model of its own: what it
// makes is a row of the shaft list, added the way the form adds one, and the
// rotor is then built by ROSS from the list as always. The numbers come from
// core/sketch3d.js; what is drawn, from components/sketch3d_parts.js; the view
// lends the stage, the camera and the pointer (`toolKit`, features/rotor3d.js).
//
// The two steps:
//
//   1. the sketch. The camera flies to the front view, square to the plane,
//      and stops turning (it still zooms and moves). "Circle" puts a circle
//      on the axis that follows the pointer, its diameter rounded; a click
//      fixes it and asks for the diameter, and the value typed is the one
//      kept. Two circles at most: the larger is the outside, the smaller the
//      bore. A circle is retyped or removed from the panel;
//   2. the extrusion. Leaving the sketch, the camera flies back out and the
//      element appears translucent, as long as it is wide, with an arrow on
//      its far face. Dragging the arrow (rounded; free with Shift held) or
//      typing the length sets it; the panel says its L/D against API RP 684
//      and takes the material. "Create" adds the row.
//
// Esc steps back: out of the diameter box, out of the circle being drawn, out
// of the tool.
import { escapeHtml } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { state } from '../core/state.js';
import {
    MAX_CIRCLES, circleProblem, draggedLength, extrusionBounds, firstFace, gridSpacing, pointerDiameter,
    sectionOf, shaftRow, shown, sketchExtent, slenderness, startingLength, typedLength,
} from '../core/sketch3d.js';
import { alongAxis } from '../core/rotor3d_measure.js';
import { buildCircle, buildExtrusion, buildSketchPlane } from '../components/sketch3d_parts.js';
import { flyToNextRotor, showEmptyStage, toolKit, useTool } from './rotor3d.js';
import { addShaftFromSketch, insertShaftFromSketch, refreshRotorFigure } from './modeling.js';

const DEFAULT_MATERIAL = 'Default (Steel)';

// Why a diameter was refused (core/sketch3d.js, `circleProblem`), in words.
const PROBLEMS = {
    sketchNeedsDiameter: () => t('sketchNeedsDiameter'),
    sketchTwoCircles: () => t('sketchTwoCircles'),
    sketchSameCircle: () => t('sketchSameCircle'),
};

let session = null;

// --- starting ---------------------------------------------------------------

// The button of the empty 3D view: the first shaft of the project.
export async function sketchFirstShaft() {
    const box = document.getElementById('rotor-3d');
    if (!box || session) return;
    try {
        await showEmptyStage(box);
    } catch (error) {
        // No WebGL, or three.js would not load: the view already says so.
        console.error('sketch:', error);
        return;
    }
    const kit = toolKit();
    if (!kit) return;
    const note = document.getElementById('rotor-3d-message');
    if (note) note.hidden = true;
    const button = document.getElementById('rotor-3d-first-shaft');
    if (button) button.hidden = true;
    const face = firstFace();
    const extent = sketchExtent(face.diameter);
    // From an isometric look at the empty plane, so the flight to it shows.
    kit.place(kit.framed(planeBounds(face, extent), 'iso'));
    openSketch(kit, face);
}

// The menu of a shaft in the 3D view (features/rotor3d.js): a sketch on its
// start or end face (core/sketch3d.js, `shaftFace`), over the rotor as it is.
export function sketchOnFace(face) {
    if (session || !face) return;
    const kit = toolKit();
    if (!kit) return;
    openSketch(kit, face);
}

function planeBounds(face, extent) {
    return { min: [-extent, -extent, face.z], max: [extent, extent, face.z] };
}

function colours(kit) {
    return {
        accent: kit.colour('--accent', 'steelblue'),
        strong: kit.colour('--text-strong', 'black'),
        grid: kit.colour('--text-muted', 'gray'),
        x: kit.colour('--plot-red', 'firebrick'),
        y: kit.colour('--plot-green', 'seagreen'),
        arrow: kit.colour('--accent', 'steelblue'),
        arrowHover: kit.colour('--text-strong', 'black'),
    };
}

// --- step 1, the sketch ---------------------------------------------------------

function openSketch(kit, face) {
    const extent = sketchExtent(face.diameter);
    const palette = colours(kit);
    const plane = buildSketchPlane(kit.THREE, { z: face.z, extent, spacing: gridSpacing(extent), colours: palette });
    kit.scene.add(plane.object);
    // The face's own circles, faint, to draw to.
    const references = [face.diameter, face.inner].filter(d => d > 0).map(d => {
        const circle = buildCircle(kit.THREE, face.z, palette.grid);
        circle.setDiameter(d);
        kit.scene.add(circle.object);
        return circle;
    });
    const panel = document.createElement('div');
    panel.className = 'sketch-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', t('sketchTitle'));
    const labels = document.createElement('div');
    labels.className = 'sketch-labels';
    labels.setAttribute('aria-hidden', 'true');
    kit.container.appendChild(labels);
    kit.container.appendChild(panel);
    session = {
        kit, face, extent, palette, plane, panel, labels, references,
        // Where the camera was: it goes back there when the sketch is left
        // without making anything, or once the element is made.
        returnTo: kit.here(),
        phase: 'sketch',
        circles: [],
        drawn: [],
        // The circle following the pointer: { d, object, fixed }.
        drawing: null,
        // The position of the circle the diameter box is for (-1: a new one).
        editing: -1,
        box: null,
        extrusion: null,
        length: 0,
        material: DEFAULT_MATERIAL,
        dragging: false,
        message: '',
    };
    useTool({
        locksRotation: true,
        pointermove: sketchMove,
        pointerdown: sketchDown,
        keydown: sketchKey,
        frame: placeLabels,
        // Torn down only: the rotor drawn is what the view now shows.
        onRotor: () => closeSketch(true),
    });
    renderSketchPanel();
    kit.fly(kit.framed(planeBounds(face, extent), face.look || 'front'));
}

function renderSketchPanel() {
    const { panel, circles, drawing } = session;
    const full = circles.length >= MAX_CIRCLES;
    const rows = circles.map((d, i) => `<li class="sketch-circle-row">`
        + `<span>${escapeHtml(circleName(i))}: Ø ${escapeHtml(shown(d))} mm</span>`
        + `<button type="button" class="sketch-icon" data-action="sketch-edit-circle" data-index="${i}" title="${escapeHtml(t('sketchRetype'))}"><i class="fas fa-pen"></i></button>`
        + `<button type="button" class="sketch-icon" data-action="sketch-remove-circle" data-index="${i}" title="${escapeHtml(t('sketchRemove'))}"><i class="fas fa-trash"></i></button>`
        + `</li>`).join('');
    panel.innerHTML = `<div class="sketch-title">${escapeHtml(t('sketchTitle'))}</div>`
        + `<p class="sketch-hint">${escapeHtml(drawing ? t('sketchDrawingHint') : t('sketchHint'))}</p>`
        + `<button type="button" class="sketch-tool" data-action="sketch-circle" aria-pressed="${String(!!drawing)}"`
        + `${full ? ' disabled' : ''} title="${escapeHtml(full ? t('sketchTwoCircles') : t('sketchCircle'))}">`
        + `<i class="far fa-circle"></i> ${escapeHtml(t('sketchCircle'))}</button>`
        + (rows ? `<ul class="sketch-circles">${rows}</ul>` : '')
        + (session.message ? `<p class="sketch-error" role="alert">${escapeHtml(session.message)}</p>` : '')
        + `<div class="sketch-actions">`
        + `<button type="button" class="btn-secondary" data-action="sketch-cancel">${escapeHtml(t('cancel'))}</button>`
        + `<button type="button" class="btn-primary" data-action="sketch-finish"${circles.length ? '' : ' disabled'}>`
        + `${escapeHtml(t('sketchFinish'))}</button></div>`;
}

// With two circles the larger is the outside and the smaller the bore; the
// names follow the values, not the order they were drawn in.
function circleName(i) {
    const { circles } = session;
    if (circles.length < 2) return t('sketchOuter');
    return circles[i] === Math.max(...circles) ? t('sketchOuter') : t('sketchInner');
}

// "Circle": a circle on the axis that follows the pointer.
export function beginCircle() {
    if (!session || session.phase !== 'sketch' || session.drawing || session.box) return;
    if (session.circles.length >= MAX_CIRCLES) return;
    const object = buildCircle(session.kit.THREE, session.face.z, session.palette.accent);
    const d = startingDiameter();
    object.setDiameter(d);
    session.kit.scene.add(object.object);
    session.drawing = { d, object, fixed: false };
    session.message = '';
    renderSketchPanel();
    session.kit.canvas.focus();
    session.kit.requestFrame();
}

// Before the pointer reaches the canvas: a third of the plane, or between the
// two circles there are, so the new one is not drawn on top of another.
// On a face, the first circle starts as the face's own diameter.
function startingDiameter() {
    const { circles, extent, face } = session;
    if (circles.length) return Math.max(1, Math.round(circles[0] / 2));
    if (face.diameter > 0) return face.diameter;
    return Math.round((extent * 1000 * 2) / 3);
}

function onPlane(event) {
    const ray = session.kit.ray(event);
    if (!ray) return null;
    const [ox, oy, oz] = ray.origin;
    const [dx, dy, dz] = ray.direction;
    if (Math.abs(dz) < 1e-9) return null;
    const along = (session.face.z - oz) / dz;
    if (along < 0) return null;
    return [ox + dx * along, oy + dy * along];
}

function sketchMove(event) {
    const { drawing } = session;
    if (!drawing || drawing.fixed) return false;
    const point = onPlane(event);
    if (!point) return false;
    drawing.d = pointerDiameter(point[0], point[1]);
    drawing.object.setDiameter(drawing.d);
    session.kit.requestFrame();
    return false;
}

function sketchDown(event) {
    const { drawing } = session;
    if (event.button !== 0 || !drawing || drawing.fixed) return false;
    const point = onPlane(event);
    if (point) {
        drawing.d = pointerDiameter(point[0], point[1]);
        drawing.object.setDiameter(drawing.d);
    }
    drawing.fixed = true;
    const box = session.kit.container.getBoundingClientRect();
    openDimensionBox(-1, drawing.d, event.clientX - box.left, event.clientY - box.top);
    return true;
}

function sketchKey(event) {
    if (event.key !== 'Escape') return false;
    if (session.box) closeDimensionBox(false);
    else if (session.drawing) dropDrawing();
    else closeSketch();
    return true;
}

function dropDrawing() {
    const { drawing, kit } = session;
    if (!drawing) return;
    kit.scene.remove(drawing.object.object);
    drawing.object.dispose();
    session.drawing = null;
    renderSketchPanel();
    kit.requestFrame();
}

// The box that asks for the diameter: beside the click for a new circle, by
// the panel for one being retyped. What is typed redraws the circle as it is
// typed, and Enter (or OK) keeps it.
function openDimensionBox(editing, value, x, y) {
    closeDimensionBox(false, true);
    session.editing = editing;
    const box = document.createElement('div');
    box.className = 'sketch-dimension';
    box.setAttribute('role', 'dialog');
    box.innerHTML = `<label>${escapeHtml(t('sketchDiameter'))} <input type="text" inputmode="decimal" `
        + `class="sketch-dimension-input" value="${escapeHtml(shown(value))}"> mm</label>`
        + `<button type="button" class="btn-primary" data-action="sketch-dimension-ok">${escapeHtml(t('sketchOk'))}</button>`
        + `<p class="sketch-error" hidden></p>`;
    const area = session.kit.container.getBoundingClientRect();
    box.style.left = `${Math.max(8, Math.min(x + 12, area.width - 260))}px`;
    box.style.top = `${Math.max(8, Math.min(y + 12, area.height - 90))}px`;
    session.kit.container.appendChild(box);
    session.box = box;
    const input = box.querySelector('input');
    input.addEventListener('input', () => {
        const d = typedLength(input.value);
        if (d) previewDiameter(d);
    });
    input.addEventListener('keydown', event => {
        if (event.key === 'Enter') {
            event.preventDefault();
            confirmDimension();
        } else if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            closeDimensionBox(false);
        }
    });
    input.focus();
    input.select();
}

function previewDiameter(d) {
    const { drawing, editing, drawn } = session;
    if (editing >= 0 && drawn[editing]) drawn[editing].setDiameter(d);
    else if (drawing) drawing.object.setDiameter(d);
    session.kit.requestFrame();
}

// OK in the box: the value typed is the circle's.
export function confirmDimension() {
    if (!session || !session.box) return;
    const input = session.box.querySelector('input');
    const d = typedLength(input.value);
    const problem = circleProblem(d, session.circles, session.editing);
    if (problem) {
        const line = session.box.querySelector('.sketch-error');
        line.textContent = PROBLEMS[problem]();
        line.hidden = false;
        input.focus();
        return;
    }
    if (session.editing >= 0) {
        session.circles[session.editing] = d;
        session.drawn[session.editing].setDiameter(d);
    } else if (session.drawing) {
        const { object } = session.drawing;
        object.setDiameter(d);
        object.setColour(session.palette.strong);
        session.circles.push(d);
        session.drawn.push(object);
        session.drawing = null;
    }
    closeDimensionBox(true);
}

// Closing the box without a value: a new circle goes back to following the
// pointer; a retyped one keeps what it had.
function closeDimensionBox(kept, quietly = false) {
    if (!session || !session.box) return;
    session.box.remove();
    session.box = null;
    if (!kept) {
        if (session.editing >= 0) session.drawn[session.editing].setDiameter(session.circles[session.editing]);
        else if (session.drawing) session.drawing.fixed = false;
    }
    session.editing = -1;
    if (!quietly) {
        renderSketchPanel();
        session.kit.canvas.focus();
        session.kit.requestFrame();
    }
}

export function retypeCircle(index) {
    if (!session || session.phase !== 'sketch' || !(index >= 0 && index < session.circles.length)) return;
    if (session.drawing) dropDrawing();
    const row = session.panel.getBoundingClientRect();
    const area = session.kit.container.getBoundingClientRect();
    openDimensionBox(index, session.circles[index], row.right - area.left, row.top - area.top);
}

export function removeCircle(index) {
    if (!session || session.phase !== 'sketch' || !(index >= 0 && index < session.circles.length)) return;
    closeDimensionBox(false, true);
    const [object] = session.drawn.splice(index, 1);
    session.circles.splice(index, 1);
    session.kit.scene.remove(object.object);
    object.dispose();
    renderSketchPanel();
    session.kit.requestFrame();
}

// The diameters over the circles, at their top, as the view turns or zooms.
function placeLabels() {
    if (!session) return;
    const { kit, labels, face } = session;
    if (session.phase !== 'sketch') {
        labels.innerHTML = '';
        return;
    }
    const items = session.circles.map(d => ({ d, live: false }));
    if (session.drawing && !session.box) items.push({ d: session.drawing.d, live: true });
    labels.innerHTML = items.map(({ d, live }) => {
        const [x, y] = kit.project([0, d / 2000, face.z]);
        return `<span class="sketch-label${live ? ' is-live' : ''}" style="transform: translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)">`
            + `Ø ${escapeHtml(shown(d))}</span>`;
    }).join('');
}

// --- step 2, the extrusion --------------------------------------------------------

// "Leave the sketch": the circles become a section, and the element it makes
// appears to be pulled to length.
export function finishSketch() {
    if (!session || session.phase !== 'sketch') return;
    const section = sectionOf(session.circles);
    if (!section) return;
    closeDimensionBox(false, true);
    if (session.drawing) dropDrawing();
    const { kit } = session;
    clearSketchDrawing();
    session.phase = 'extrude';
    session.section = section;
    session.length = startingLength(section);
    session.extrusion = buildExtrusion(kit.THREE, session.palette);
    session.extrusion.set(session.face, session.length, section);
    kit.scene.add(session.extrusion.object);
    useTool({
        locksRotation: false,
        pointermove: extrudeMove,
        pointerdown: extrudeDown,
        pointerup: extrudeUp,
        keydown: extrudeKey,
        frame: placeLabels,
        // Torn down only: the rotor drawn is what the view now shows.
        onRotor: () => closeSketch(true),
    });
    renderExtrudePanel();
    flyToExtrusion();
}

// `from`: the direction to look from. Unless given: the isometric view for the
// first shaft, and over a rotor the isometric view from the side the element
// grows into, so the rest of the rotor is behind it rather than in front.
function flyToExtrusion(from = null) {
    const { kit, face, section } = session;
    if (from === null) from = face.first ? 'iso' : [-1, 1, face.sense];
    // Over a rotor, room round the face for what is drawn on it.
    const bounds = extrusionBounds(face, Math.max(session.length, section.odl) * 1.6, section, face.first ? 0 : 4);
    kit.fly(kit.framed(bounds, from));
}

function clearSketchDrawing() {
    const { kit } = session;
    kit.scene.remove(session.plane.object);
    session.plane.dispose();
    session.references.forEach(circle => {
        kit.scene.remove(circle.object);
        circle.dispose();
    });
    session.references = [];
    session.drawn.forEach(object => {
        kit.scene.remove(object.object);
        object.dispose();
    });
    session.drawn = [];
    session.labels.innerHTML = '';
}

function materialOptions() {
    const project = state.projectData || {};
    const names = (project.materials || []).map(m => String(m.name || '')).filter(Boolean);
    return [`<option value="${DEFAULT_MATERIAL}">${escapeHtml(t('defaultSteel'))}</option>`]
        .concat(names.map(name => `<option value="${escapeHtml(name)}"${name === session.material ? ' selected' : ''}>${escapeHtml(name)}</option>`))
        .join('');
}

function renderExtrudePanel() {
    const { panel, section } = session;
    const bore = section.idl > 0 ? ` · ${escapeHtml(t('sketchInner'))} Ø ${escapeHtml(shown(section.idl))} mm` : '';
    panel.innerHTML = `<div class="sketch-title">${escapeHtml(t('extrudeTitle'))}</div>`
        + `<p class="sketch-hint">${escapeHtml(t('extrudeHint'))}</p>`
        + `<p class="sketch-section">${escapeHtml(t('sketchOuter'))} Ø ${escapeHtml(shown(section.odl))} mm${bore}</p>`
        + (session.face.first || session.face.free ? ''
            : `<p class="sketch-hint is-warning">${escapeHtml(t('extrudeShifts'))}</p>`)
        + `<label class="sketch-field">${escapeHtml(t('extrudeLength'))} <input type="text" inputmode="decimal" `
        + `class="extrude-length" value="${escapeHtml(shown(session.length))}"> mm</label>`
        + `<p class="sketch-ratio" aria-live="polite"></p>`
        + `<label class="sketch-field">${escapeHtml(t('extrudeMaterial'))} `
        + `<select class="extrude-material" data-action="extrude-material">${materialOptions()}</select></label>`
        + `<div class="sketch-actions">`
        + `<button type="button" class="btn-secondary" data-action="sketch-cancel">${escapeHtml(t('cancel'))}</button>`
        + `<button type="button" class="btn-primary" data-action="extrude-confirm">${escapeHtml(t('extrudeCreate'))}</button></div>`;
    const input = panel.querySelector('.extrude-length');
    input.addEventListener('input', () => {
        const value = typedLength(input.value);
        if (value) setLength(value, false);
    });
    input.addEventListener('keydown', event => {
        if (event.key === 'Enter') {
            event.preventDefault();
            confirmExtrusion();
        }
    });
    showRatio();
}

function showRatio() {
    const line = session.panel.querySelector('.sketch-ratio');
    if (!line) return;
    const ratio = slenderness(session.length, session.section);
    if (!ratio) {
        line.textContent = '';
        return;
    }
    const value = `L/D = ${shown(Number(ratio.ratio.toPrecision(3)))}`;
    line.classList.toggle('is-warning', ratio.tooLong);
    line.textContent = ratio.tooLong ? `${value} · ${t('extrudeTooLong')}`
        : ratio.overPreferred ? `${value} · ${t('extrudeOverPreferred')}` : value;
}

// The length, from the arrow (`fromArrow`: the box is rewritten to match) or
// from the box.
function setLength(value, fromArrow) {
    session.length = value;
    session.extrusion.set(session.face, value, session.section);
    if (fromArrow) {
        const input = session.panel.querySelector('.extrude-length');
        if (input) input.value = shown(value);
    }
    showRatio();
    session.kit.requestFrame();
}

function arrowUnder(event) {
    const ray = session.kit.ray(event);
    if (!ray) return false;
    return ray.raycaster.intersectObject(session.extrusion.arrow, true).length > 0;
}

function extrudeMove(event) {
    const { kit } = session;
    if (session.dragging) {
        const ray = kit.ray(event);
        const z = ray ? alongAxis(ray.origin, ray.direction, { x: 0, tape: 0 }) : null;
        if (z !== null) setLength(draggedLength(z, session.face.z, session.face.sense, event.shiftKey), true);
        return true;
    }
    const over = arrowUnder(event);
    session.extrusion.setHover(over);
    kit.canvas.classList.toggle('is-grab', over);
    kit.requestFrame();
    return false;
}

function extrudeDown(event) {
    if (event.button !== 0 || !arrowUnder(event)) return false;
    session.dragging = true;
    // The press is kept from the camera, and with it the focus a click gives:
    // given here, so Enter and Esc reach the extrusion after the drag.
    session.kit.canvas.focus();
    try { session.kit.canvas.setPointerCapture(event.pointerId); } catch (e) { /* not a live pointer */ }
    session.kit.canvas.classList.add('is-grabbing');
    return true;
}

function extrudeUp(event) {
    if (!session.dragging) return false;
    session.dragging = false;
    try { session.kit.canvas.releasePointerCapture(event.pointerId); } catch (e) { /* already released */ }
    session.kit.canvas.classList.remove('is-grabbing');
    keepInView();
    return true;
}

// Pulled past the edge of the view, the element is framed again once the
// arrow is let go -- not while it is held, which would move the arrow away
// from the pointer.
function keepInView() {
    const { kit, face, section, length } = session;
    const box = kit.canvas.getBoundingClientRect();
    const far = face.z + face.sense * (length / 1000);
    const r = section.odl / 2000;
    const outside = [[0, r, far], [0, -r, far], [r, 0, far], [-r, 0, far]].some(point => {
        const [x, y, depth] = kit.project(point);
        return depth > 1 || x < 0 || y < 0 || x > box.width || y > box.height;
    });
    if (!outside) return;
    // From where the person was looking, not back to the isometric view.
    const { position, target } = kit.here();
    flyToExtrusion(position.map((v, i) => v - target[i]));
}

function extrudeKey(event) {
    if (event.key === 'Escape') {
        closeSketch();
        return true;
    }
    if (event.key === 'Enter') {
        confirmExtrusion();
        return true;
    }
    return false;
}

export function chooseMaterial(value) {
    if (session) session.material = value || DEFAULT_MATERIAL;
}

// "Create": the row joins the shaft list, and ROSS builds the rotor from it.
export function confirmExtrusion() {
    if (!session || session.phase !== 'extrude') return;
    const input = session.panel.querySelector('.extrude-length');
    const typed = input ? typedLength(input.value) : session.length;
    if (!typed) {
        input.focus();
        return;
    }
    const row = shaftRow(typed, session.section, session.material);
    const { face, kit, returnTo } = session;
    closeSketch(true);
    flyToNextRotor();
    if (face.first) {
        addShaftFromSketch(row);
        return;
    }
    kit.fly(returnTo);
    insertShaftFromSketch(face.index, face.side, row);
}

// --- leaving ------------------------------------------------------------------------

// Takes down everything the tool put up. `made`: a row was added, and the
// rotor about to be built replaces what is drawn; otherwise the view goes back
// to what it was -- the message and the button of an empty project, or the
// camera where it was over the rotor.
export function closeSketch(made = false) {
    if (!session) return;
    const { kit, face, returnTo } = session;
    closeDimensionBox(false, true);
    if (session.drawing) {
        kit.scene.remove(session.drawing.object.object);
        session.drawing.object.dispose();
        session.drawing = null;
    }
    if (session.phase === 'sketch') clearSketchDrawing();
    if (session.extrusion) {
        kit.scene.remove(session.extrusion.object);
        session.extrusion.dispose();
    }
    session.panel.remove();
    session.labels.remove();
    kit.canvas.classList.remove('is-grab', 'is-grabbing');
    session = null;
    useTool(null);
    if (made) return;
    if (face.first) refreshRotorFigure();
    else kit.fly(returnTo);
}

export const SKETCH_ACTIONS = {
    'sketch-first-shaft': () => sketchFirstShaft(),
    'sketch-circle': () => beginCircle(),
    'sketch-edit-circle': element => retypeCircle(Number(element.dataset.index)),
    'sketch-remove-circle': element => removeCircle(Number(element.dataset.index)),
    'sketch-dimension-ok': () => confirmDimension(),
    'sketch-finish': () => finishSketch(),
    'sketch-cancel': () => closeSketch(),
    'extrude-material': element => chooseMaterial(element.value),
    'extrude-confirm': () => confirmExtrusion(),
};
