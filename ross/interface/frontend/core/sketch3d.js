// The sketch and the extrusion of a shaft element in the 3D view
// (features/sketch3d.js), without three.js or the page, so node can test it.
//
// A ROSS shaft element is a cylinder: its length `L`, its outer diameter
// `odl` and its inner one `idl`. The sketch gives the two diameters -- one or
// two concentric circles on the plane of a cross-section -- and the extrusion
// the length. Nothing else is asked for here: a tapered element (`idr`,
// `odr`) is typed in the form.
//
// What the mouse gives is only a preview, rounded to a step that suits the
// size (`niceStep`); the value typed in the box beside it is what the element
// gets (Leonardo: "o valor digitado manda").
//
// Lengths are in metres in the scene and in millimetres for the person: the
// shaft form's unit (domain/units.py), so the row needs no `_unit`.

export const MAX_CIRCLES = 2;

// The rounding step for a value of about `value` (any unit): a tenth of its
// order of magnitude -- 47.3 mm moves by 1 mm, 4.73 mm by 0.1 mm, 473 mm by 10.
export function niceStep(value) {
    if (!(value > 0) || !Number.isFinite(value)) return 0.1;
    return Math.pow(10, Math.floor(Math.log10(value)) - 1);
}

// `value` to the nearest multiple of `step`, without the float's noise
// (0.1 + 0.2 must read 0.3).
export function rounded(value, step) {
    if (!(step > 0)) return value;
    return Number((Math.round(value / step) * step).toPrecision(12));
}

// A number as the form would show it: no trailing zeros, no float noise.
export function shown(value) {
    return String(Number(Number(value).toPrecision(10)));
}

// What was typed in a dimension box, as a number of millimetres: a decimal
// comma reads as the point it means. Null when it is not a positive number.
export function typedLength(text) {
    const value = Number(String(text === undefined || text === null ? '' : text).trim().replace(',', '.'));
    return Number.isFinite(value) && value > 0 ? value : null;
}

// The diameter the pointer draws: twice its distance from the axis, in mm,
// rounded. `x`, `y` are the point of the sketch plane under the pointer, in
// metres; the circles are centred on the rotor axis.
export function pointerDiameter(x, y) {
    const millimetres = 2000 * Math.hypot(x, y);
    return rounded(millimetres, niceStep(millimetres)) || niceStep(millimetres);
}

// The section the circles make: the larger is the outer diameter, the
// smaller (if there are two) the bore. Null without a circle.
export function sectionOf(circles) {
    const valid = (circles || []).filter(d => d > 0);
    if (!valid.length) return null;
    const odl = Math.max(...valid);
    const idl = valid.length > 1 ? Math.min(...valid) : 0;
    return { odl, idl };
}

// Why a circle of diameter `d` (mm) cannot join `circles`, or null when it
// can. `replacing`: the position of the circle being retyped, which is not
// compared with itself. The reasons are i18n keys.
export function circleProblem(d, circles, replacing = -1) {
    if (!(d > 0)) return 'sketchNeedsDiameter';
    const others = (circles || []).filter((_, i) => i !== replacing);
    if (replacing < 0 && others.length >= MAX_CIRCLES) return 'sketchTwoCircles';
    if (others.some(o => Math.abs(o - d) <= 1e-9 * Math.max(o, d))) return 'sketchSameCircle';
    return null;
}

// The extrusion's length (mm) for the point `z` of the rotor axis the arrow
// was dragged to (metres): measured from the sketch plane `z0` in the sense
// the element grows (+1 after the face, -1 before it), rounded unless `free`,
// and never less than one step.
export function draggedLength(z, z0, sense, free = false) {
    const millimetres = 1000 * (z - z0) * (sense < 0 ? -1 : 1);
    const step = niceStep(Math.max(Math.abs(millimetres), 1));
    if (!(millimetres > 0)) return step;
    const value = free ? Number(millimetres.toPrecision(6)) : rounded(millimetres, step);
    return Math.max(value, step);
}

// The length the extrusion starts at: as long as the section is wide, the
// most an element should be by API RP 684 (1.5.2.1: L/D at most 1.0).
export function startingLength(section) {
    return section ? rounded(section.odl, niceStep(section.odl)) : 100;
}

// A tapered element: the section of its far face, `{ od, id }` in mm, beside
// the sketched one (`section`, the near face). Null is a straight element.
// Turned on, the far face starts as the near one.
export function farOf(section) {
    return { od: section.odl, id: section.idl };
}

export function isTapered(section, far) {
    return !!far && (Math.abs(far.od - section.odl) > 1e-12 || Math.abs(far.id - section.idl) > 1e-12);
}

// Why the far face cannot be made, or null: it needs an outside, wider than
// its bore. The reasons are i18n keys.
export function taperProblem(far) {
    if (!far) return null;
    if (!(far.od > 0)) return 'extrudeFarNeedsOuter';
    if (!(far.id >= 0) || far.id >= far.od) return 'extrudeFarBore';
    return null;
}

// The far face's outside from the handle on its rim: twice the distance of
// the pointer from the axis on the far face's plane, rounded unless `free`,
// and kept a step wider than the bore.
export function handleDiameter(x, y, bore, free = false) {
    const millimetres = 2000 * Math.hypot(x, y);
    const step = niceStep(Math.max(millimetres, 1));
    const value = free ? Number(millimetres.toPrecision(6)) : rounded(millimetres, step);
    return Math.max(value, (bore || 0) + step, step);
}

// The diameter API RP 684's L/D is taken with: the larger end of a taper, as
// ROSS's `refine_mesh` takes it.
function widest(section, far) {
    return Math.max(section.odl, far && far.od > 0 ? far.od : 0);
}

// L/D, and whether it is past what API RP 684 accepts for an element (1.0)
// or prefers (0.5) -- the rule the discretization dialog cuts by.
export function slenderness(length, section, far = null) {
    if (!section || !(section.odl > 0) || !(length > 0)) return null;
    const ratio = length / widest(section, far);
    return { ratio, tooLong: ratio > 1 + 1e-9, overPreferred: ratio > 0.5 + 1e-9 };
}

// The row of the element list the sketch and the extrusion make. A taper
// writes the right end too (`odr`, `idr`): ROSS's left end is the lower z, so
// the sketched face is the left one when the element grows +z (`sense` 1) and
// the right one when it grows -z, from the start face of a shaft.
export function shaftRow(length, section, material, far = null, sense = 1) {
    const row = { element_type: 'BASIC', L: shown(length) };
    if (isTapered(section, far)) {
        const near = { od: section.odl, id: section.idl };
        const [left, right] = sense < 0 ? [far, near] : [near, far];
        Object.assign(row, {
            idl: shown(left.id), odl: shown(left.od), idr: shown(right.id), odr: shown(right.od),
        });
    } else {
        Object.assign(row, { idl: shown(section.idl), odl: shown(section.odl) });
    }
    row.material = material || 'Default (Steel)';
    return row;
}

// How far around the axis the sketch plane is shown, in metres: three times
// the radius of the face it is drawn on, or 150 mm with nothing to go by.
export function sketchExtent(referenceDiameterMm) {
    const radius = referenceDiameterMm > 0 ? referenceDiameterMm / 2000 : 0.05;
    return 3 * radius;
}

// The grid of the sketch plane: its half-size `extent` (metres) cut in about
// thirty steps of a round number of millimetres; every fifth line is heavier.
export function gridSpacing(extent) {
    const millimetres = (2 * extent * 1000) / 30;
    const power = Math.pow(10, Math.floor(Math.log10(millimetres)));
    const k = [1, 2, 5, 10].find(m => m * power >= millimetres * (1 - 1e-12));
    return Number(((k * power) / 1000).toPrecision(12));
}

// Where a face lets a new element start: its plane `z` (metres), the sense it
// grows (+1 after the face, -1 before it), and the diameter the face has (mm),
// for the size of the sketch. The first shaft of a rotor starts on z = 0,
// sketched from the front view (`look`, the direction the camera looks from).
export function firstFace() {
    return { z: 0, sense: 1, diameter: 0, first: true, look: [0, 0, -1] };
}

// The start or the end face of a shaft part of the 3D view's layout
// (core/rotor3d_layout.js), as a place to sketch on:
//
//   * `side`, `index`: what domain/inserting.py is told -- the new element goes
//     after the shaft (`end`) or before it (`start`);
//   * `node`: the node the face is on, for the menu to name it;
//   * `look`: the sketch is seen from outside the face, the side the element
//     grows into;
//   * `free`: nothing of the line goes on from the face. When something does,
//     it is still the place to sketch -- the neighbour hidden or not -- but
//     what is past the face moves along by the new length, and the panel says
//     so.
// `diameter` and `inner` are the face's (mm): the size of the sketch, and the
// circles it shows for reference.
export function shaftFace(part, side, parts) {
    const end = side === 'end';
    const z = end ? part.z1 : part.z0;
    const profile = part.profile || {};
    const outer = end ? profile.odr : profile.odl;
    const inner = end ? profile.idr : profile.idl;
    const span = Math.abs(part.z1 - part.z0) || 1;
    const touching = (parts || []).some(p => p !== part && p.half === part.half
        && (p.kind === 'shaft' || p.kind === 'coupling')
        && Math.abs((end ? Math.min(p.z0, p.z1) : Math.max(p.z0, p.z1)) - z) <= 1e-9 * span);
    const start = part.entry && Number.isFinite(Number(part.entry.n)) ? Number(part.entry.n) : null;
    return {
        z,
        sense: end ? 1 : -1,
        diameter: (outer || 0) * 1000,
        inner: (inner || 0) * 1000,
        first: false,
        look: [0, 0, end ? 1 : -1],
        index: part.index,
        side: end ? 'end' : 'start',
        node: start === null ? null : start + (end ? 1 : 0),
        free: !touching,
    };
}

// The box the extrusion takes up, for the camera to frame it. `margin`, in
// radii of the element, widens it all round: on the face of a rotor the camera
// has to stand clear of the bearing or the disk that sits there, which are
// drawn a few shaft radii wide.
export function extrusionBounds(face, length, section, margin = 0, far = null) {
    const r = (section ? widest(section, far) : 100) / 2000;
    const z1 = face.z + face.sense * (length / 1000);
    const out = r * (1 + margin);
    const along = r * margin;
    return { min: [-out, -out, Math.min(face.z, z1) - along], max: [out, out, Math.max(face.z, z1) + along] };
}
