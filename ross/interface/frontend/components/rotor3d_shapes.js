// The shapes of the geometry bank (core/shapes3d.js), each built in the
// envelope of the element it stands for: its outer radius `part.radius`, its
// width `z0..z1` and its bore. Only pictures -- ROSS computes with the
// element's own mass and inertias -- so what matters is that each reads as
// what it is, and that nothing is drawn outside the part: the pointer finds
// parts by that envelope and the camera frames it.
//
// three.js is handed in, not imported (see rotor3d_parts.js). Each builder
// returns pieces as rotor3d_solids.js describes them.
import {
    SEGMENTS, bearingHousing, boltCircle, boltHead, couplingShaftEnds, faceted, loft, own, revolve, ring, steel,
} from './rotor3d_solids.js';
import { SEAL_FLANGE } from '../core/rotor3d_layout.js';

// The frame every shape is built in: z along the shaft, the part's middle at
// `mid`.
function frame(part) {
    const w = Math.abs(part.z1 - part.z0);
    const R = part.radius;
    return { R, w, mid: (part.z0 + part.z1) / 2, bore: Math.min(part.bore || 0, 0.8 * R) };
}

// A point at radius r, angle a round the shaft, height z.
const at = (r, a, z) => [r * Math.cos(a), r * Math.sin(a), z];

// --- centrifugal compressor impeller -------------------------------------------
//
// The eye faces node 0 (-z), the back plate the other way; the hub flares
// from the eye to the full radius, and the blades run over it from the inlet
// -- leaning into the flow -- to the tip, swept back. Half of them are
// splitters, starting part-way along.
function impeller(THREE, part, edges) {
    const { R, w, mid, bore } = frame(part);
    const front = mid - w / 2;
    const back = mid + w / 2;
    const plate = 0.1 * w;
    const run = w - plate;
    const nose = Math.max(bore * 1.3, 0.28 * R);
    const hubAt = th => [R - (R - nose) * Math.cos(th), front + run * Math.sin(th)];
    const inletTip = nose + 0.55 * (R - nose);
    const exitHeight = 0.2 * run;
    const shroudAt = th => [R - (R - inletTip) * Math.cos(th), front + (run - exitHeight) * Math.sin(th)];

    const profile = [[bore, front], [nose, front]];
    for (let i = 1; i <= 14; i++) profile.push(hubAt((i / 14) * (Math.PI / 2)));
    profile.push([R, back], [bore, back], [bore, front]);
    const pieces = [own(revolve(THREE, profile, SEGMENTS.part, edges))];

    const blades = 7;
    const thick = 0.02 * R;
    const samples = 20;
    const blade = (start, phase) => {
        const sections = [];
        for (const s of [0, 0.5, 1]) {
            const sideA = [];
            const sideB = [];
            for (let i = 0; i <= samples; i++) {
                const t = start + (1 - start) * (i / samples);
                const th = t * (Math.PI / 2);
                const [rh, zh] = hubAt(th);
                const [rs, zs] = shroudAt(th);
                const r = rh + (rs - rh) * s;
                const z = zh + (zs - zh) * s;
                const lead = (1 - t) * (1 - t);
                const a = phase + 0.6 * lead - 0.35 * t * t + 0.2 * s * lead;
                const half = (0.5 * thick) / Math.max(r, 1e-9);
                sideA.push(at(r, a + half, z));
                sideB.unshift(at(r, a - half, z));
            }
            sections.push(sideA.concat(sideB));
        }
        return loft(THREE, sections, edges);
    };
    for (let i = 0; i < blades; i++) {
        const phase = (i / blades) * Math.PI * 2;
        pieces.push(own(blade(0, phase)));
        pieces.push(own(blade(0.45, phase + Math.PI / blades)));
    }
    return pieces;
}

// A profile in (r, z), turned to go counter-clockwise, as `revolve` needs:
// the shoelace sign says which way it goes.
function counterClockwise(profile) {
    let area = 0;
    for (let i = 0; i + 1 < profile.length; i++) {
        area += profile[i][0] * profile[i + 1][1] - profile[i + 1][0] * profile[i][1];
    }
    return area >= 0 ? profile : profile.slice().reverse();
}

// --- closed (shrouded) impeller ------------------------------------------------------
//
// The impeller of a multistage centrifugal compressor: the blades run between
// the hub and a cover disk, the shroud, so what shows from outside is the
// shroud, the eye in it with its seal teeth, and the blades in the passages at
// the rim. The eye faces node 0 (-z), as the open impeller's does.
function closedImpeller(THREE, part, edges) {
    const { R, w, mid, bore } = frame(part);
    const front = mid - w / 2;
    const back = mid + w / 2;
    const plate = 0.1 * w;
    const recess = 0.12 * w;
    const nose = Math.max(bore * 1.3, 0.3 * R);
    const eye = nose + 0.5 * (R - nose);
    const lip = 0.16 * w;
    const exit = 0.2 * (w - plate - recess);
    const cover = 0.045 * R;
    const exitZ = back - plate - exit;
    const hubAt = th => [R - (R - nose) * Math.cos(th), front + recess + (back - plate - front - recess) * Math.sin(th)];
    const shroudAt = th => [R - (R - eye) * Math.cos(th), front + lip + (exitZ - front - lip) * Math.sin(th)];

    // The hub and its back plate.
    const hub = [[bore, front + recess], [nose, front + recess]];
    for (let i = 1; i <= 14; i++) hub.push(hubAt((i / 14) * (Math.PI / 2)));
    hub.push([R, back], [bore, back], [bore, front + recess]);
    const pieces = [own(revolve(THREE, counterClockwise(hub), SEGMENTS.part, edges))];

    // The shroud: its inner face from the eye to the rim, and its outer face
    // `cover` further out along the normal, back to the eye.
    const inner = [[eye, front]];
    const outer = [[eye + cover, front]];
    const samples = 14;
    for (let i = 0; i <= samples; i++) {
        const th = (i / samples) * (Math.PI / 2);
        const [r, z] = shroudAt(th);
        const tr = (R - eye) * Math.sin(th);
        const tz = (exitZ - front - lip) * Math.cos(th);
        const length = Math.hypot(tr, tz) || 1;
        inner.push([r, z]);
        outer.push([Math.min(r + (cover * tz) / length, R), z - (cover * tr) / length]);
    }
    const shroud = inner.concat(outer.reverse());
    shroud.push(inner[0]);
    pieces.push(own(revolve(THREE, counterClockwise(shroud), SEGMENTS.part, edges)));
    // The eye seal's teeth round the shroud's lip.
    for (let k = 0; k < 3; k++) {
        const z = front + (0.2 + 0.3 * k) * lip;
        pieces.push(own(ring(THREE, eye + 0.9 * cover, eye + cover + 0.035 * R, z, z + 0.12 * lip, edges)));
    }

    // Full-length blades, swept back, reaching a little into the hub and the
    // shroud so no seam shows between them.
    const blades = 11;
    const thick = 0.022 * R;
    const along = 20;
    const blade = phase => {
        const sections = [];
        for (const s of [-0.03, 0.5, 1.03]) {
            const sideA = [];
            const sideB = [];
            for (let i = 0; i <= along; i++) {
                const t = i / along;
                const th = t * (Math.PI / 2);
                const [rh, zh] = hubAt(th);
                const [rs, zs] = shroudAt(th);
                const r = Math.min(rh + (rs - rh) * s, R);
                const z = zh + (zs - zh) * s;
                const lead = (1 - t) * (1 - t);
                const a = phase + 0.45 * lead - 0.45 * t * t;
                const half = (0.5 * thick) / Math.max(r, 1e-9);
                sideA.push(at(r, a + half, z));
                sideB.unshift(at(r, a - half, z));
            }
            sections.push(sideA.concat(sideB));
        }
        return loft(THREE, sections, edges);
    };
    for (let i = 0; i < blades; i++) pieces.push(own(blade((i / blades) * Math.PI * 2)));
    return pieces;
}

// --- bladed disks: axial compressor stage, turbine wheel, fan --------------------
//
// A blade is an airfoil section -- a cambered, thickened line -- set at a
// stagger angle from the shaft axis that grows from root to tip (the twist),
// and lofted over the span.
function bladeRow(THREE, edges, { count, root, tip, centre, chord, stagger, camber, thickness, spans = 4 }) {
    const pieces = [];
    const samples = 8;
    for (let b = 0; b < count; b++) {
        const a = (b / count) * Math.PI * 2;
        const radial = [Math.cos(a), Math.sin(a), 0];
        const across = [-Math.sin(a), Math.cos(a), 0];
        const sections = [];
        for (let k = 0; k <= spans; k++) {
            const f = k / spans;
            const reach = root + (tip - root) * f;
            const c = chord(f);
            const z = stagger(f);
            const along = [Math.sin(z) * across[0], Math.sin(z) * across[1], Math.cos(z)];
            const normal = [Math.cos(z) * across[0], Math.cos(z) * across[1], -Math.sin(z)];
            const outline = [];
            for (let i = 0; i <= samples; i++) {
                const u = (i / samples - 0.5) * c;
                const x = 1 - Math.pow((2 * u) / c, 2);
                const bend = camber * c * x;
                const half = 0.5 * thickness * c * Math.max(Math.sqrt(Math.max(x, 0)), 0.15);
                outline.push([u, bend + half], [u, bend - half]);
            }
            // The section leans across the radius, so its corners stand further
            // out than its middle: the middle is brought in until the farthest
            // corner is at `reach`, inside the part's envelope.
            const sideways = Math.max(...outline.map(([u, v]) => Math.abs(u * Math.sin(z) + v * Math.cos(z))));
            const r = Math.sqrt(Math.max(reach * reach - sideways * sideways, 0.25 * reach * reach));
            const point = ([u, v]) => [0, 1, 2].map(i => r * radial[i] + u * along[i] + v * normal[i] + (i === 2 ? centre : 0));
            const upper = outline.filter((_, i) => i % 2 === 0).map(point);
            const lower = outline.filter((_, i) => i % 2 === 1).map(point).reverse();
            sections.push(upper.concat(lower));
        }
        pieces.push(own(loft(THREE, sections, edges)));
    }
    return pieces;
}

// The disk under the blades: a hub the full width, a thinner web, and the
// platform the blades stand on.
function bladedDisk(THREE, part, edges, platformAt, platformWidth) {
    const { R, w, mid, bore } = frame(part);
    const h = w / 2;
    const platform = platformAt * R;
    const lip = 0.07 * R;
    const hub = bore + 0.25 * (platform - lip - bore);
    const pw = platformWidth * h;
    const profile = [
        [bore, mid - h], [hub, mid - h], [hub, mid - 0.36 * h], [platform - lip, mid - 0.24 * h],
        [platform - lip, mid - pw], [platform, mid - pw], [platform, mid + pw], [platform - lip, mid + pw],
        [platform - lip, mid + 0.24 * h], [hub, mid + 0.36 * h], [hub, mid + h], [bore, mid + h], [bore, mid - h],
    ];
    return own(revolve(THREE, profile, SEGMENTS.part, edges));
}

function axialStage(THREE, part, edges) {
    const { R, w, mid } = frame(part);
    const root = 0.64 * R;
    const chord = 0.8 * w / Math.cos((30 * Math.PI) / 180);
    const count = Math.max(16, Math.min(56, Math.round((2 * Math.PI * (root + R) / 2) / (0.75 * chord))));
    return [bladedDisk(THREE, part, edges, 0.66, 0.84)].concat(bladeRow(THREE, edges, {
        count, root, tip: R, centre: mid, chord: () => chord,
        stagger: f => ((30 + 25 * f) * Math.PI) / 180, camber: 0.06, thickness: 0.08,
    }));
}

// Fewer, more curved blades than a compressor stage, under a shroud band.
function turbine(THREE, part, edges) {
    const { R, w, mid } = frame(part);
    const root = 0.6 * R;
    const band = 0.07 * R;
    const chord = 0.8 * w / Math.cos((22 * Math.PI) / 180);
    const count = Math.max(12, Math.min(40, Math.round((2 * Math.PI * (root + R) / 2) / (1.05 * chord))));
    return [bladedDisk(THREE, part, edges, 0.62, 0.9)]
        .concat(bladeRow(THREE, edges, {
            count, root, tip: R - band * 0.9, centre: mid, chord: () => chord,
            stagger: f => ((22 + 18 * f) * Math.PI) / 180, camber: 0.2, thickness: 0.14,
        }))
        .concat([own(ring(THREE, R - band, R, mid - 0.42 * w, mid + 0.42 * w, edges, SEGMENTS.part, 0.2 * band))]);
}

// A few broad, strongly twisted blades on a hub with a spinner.
function fan(THREE, part, edges) {
    const { R, w, mid, bore } = frame(part);
    const hub = Math.max(bore * 1.4, 0.3 * R);
    const front = mid - w / 2;
    const noseLength = 0.35 * w;
    const profile = [[0, front]];
    for (let i = 1; i <= 8; i++) {
        const th = (i / 8) * (Math.PI / 2);
        profile.push([hub * Math.sin(th), front + noseLength * (1 - Math.cos(th))]);
    }
    profile.push([hub, mid + w / 2], [0, mid + w / 2], [0, front]);
    const pieces = [own(revolve(THREE, profile, SEGMENTS.part, edges))];
    const count = 7;
    const room = w - noseLength;
    const centre = front + noseLength + room / 2;
    const rootStagger = (35 * Math.PI) / 180;
    const tipStagger = (65 * Math.PI) / 180;
    const rootChord = (0.9 * room) / Math.cos(rootStagger);
    const tipChord = Math.min(1.35 * rootChord, (0.9 * room) / Math.cos(tipStagger), (0.8 * 2 * Math.PI * R) / count);
    return pieces.concat(bladeRow(THREE, edges, {
        count, root: hub * 0.98, tip: R, centre, spans: 5,
        chord: f => rootChord + (tipChord - rootChord) * f,
        stagger: f => rootStagger + (tipStagger - rootStagger) * f, camber: 0.08, thickness: 0.06,
    }));
}

// --- flywheel and pulley -------------------------------------------------------------

// A heavy rim on spokes.
function flywheel(THREE, part, edges) {
    const { R, w, mid, bore } = frame(part);
    const rimIn = 0.74 * R;
    const hub = Math.max(bore * 1.9, 0.22 * R);
    const pieces = [
        own(ring(THREE, rimIn, R, mid - w / 2, mid + w / 2, edges, SEGMENTS.part, 0.04 * R)),
        own(ring(THREE, bore, hub, mid - w / 2, mid + w / 2, edges, SEGMENTS.part, 0.03 * R)),
    ];
    const spokes = 6;
    const thick = Math.min(0.35 * w, 0.07 * R);
    const length = rimIn - hub + 0.04 * R;
    for (let i = 0; i < spokes; i++) {
        const a = (i / spokes) * Math.PI * 2;
        const spoke = new THREE.CylinderGeometry(thick, thick * 1.25, length, 16);
        spoke.translate(0, (hub + rimIn) / 2, 0);
        spoke.rotateZ(a - Math.PI / 2);
        spoke.translate(0, 0, mid);
        pieces.push(own(faceted(THREE, spoke, edges)));
    }
    return pieces;
}

// A V-belt pulley: grooves round the rim, a web, the hub.
function pulley(THREE, part, edges) {
    const { R, w, mid, bore } = frame(part);
    const h = w / 2;
    const hub = Math.max(bore * 1.6, 0.25 * R);
    const rimIn = 0.8 * R;
    const grooves = Math.max(1, Math.min(6, Math.floor(w / (0.16 * R))));
    const pitch = w / grooves;
    const top = 0.8 * pitch;
    const depth = Math.min(0.1 * R, 0.8 * (R - rimIn));
    const bottom = Math.max(top - 2 * depth * Math.tan((19 * Math.PI) / 180), 0.12 * pitch);
    const profile = [
        [bore, mid - h], [hub, mid - h], [hub, mid - 0.12 * w], [rimIn, mid - 0.12 * w], [rimIn, mid - h], [R, mid - h],
    ];
    for (let g = 0; g < grooves; g++) {
        const c = mid - h + pitch * (g + 0.5);
        profile.push([R, c - top / 2], [R - depth, c - bottom / 2], [R - depth, c + bottom / 2], [R, c + top / 2]);
    }
    profile.push([R, mid + h], [rimIn, mid + h], [rimIn, mid + 0.12 * w], [hub, mid + 0.12 * w], [hub, mid + h],
        [bore, mid + h], [bore, mid - h]);
    return [own(revolve(THREE, profile, SEGMENTS.part, edges))];
}

// --- supports: bearings --------------------------------------------------------------
//
// Materials, as numbers like the other ones ROSS does not colour (see
// rotor3d_parts.js): the bearing's white metal, a magnet's copper windings.
const WHITE_METAL = 0xc9c2a8;
const COPPER = 0xb8733d;
const RACE = 0xc9ced4;
const BRISTLE = 0x9b8f78;
const ELASTOMER = 0x2f3236;
const HOLE = 0x16191c;

// The frame of a part that sits on a node: the shaft's radius there, the
// part's own, its width and middle.
function onNode(part) {
    const w = Math.abs(part.z1 - part.z0);
    return { r: part.shaftRadius, R: part.radius, w, mid: (part.z0 + part.z1) / 2 };
}

// An annular sector -- a pad, a jaw, a weight -- from angle a0 to a1, between
// two radii, `depth` long along the shaft and centred at z.
function sector(THREE, inner, outer, a0, a1, depth, z, edges) {
    const shape = new THREE.Shape();
    shape.absarc(0, 0, outer, a0, a1, false);
    shape.absarc(0, 0, inner, a1, a0, true);
    shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false, curveSegments: 16 });
    geometry.translate(0, 0, z - depth / 2);
    return faceted(THREE, geometry, edges);
}

// A box from `inner` outwards along the direction at angle a, `across` wide
// and `depth` long along the shaft: a magnet's pole, a tuft of bristles. `lay`
// leans it off the radius about its inner end, as a brush seal's bristles
// lean with the rotation; leaning, it reaches less far than `outer`.
function radialBox(THREE, inner, outer, a, across, depth, z, edges, lay = 0) {
    const box = new THREE.BoxGeometry(across, outer - inner, depth);
    box.translate(0, (outer - inner) / 2, 0);
    if (lay) box.rotateZ(lay);
    box.translate(0, inner, 0);
    box.rotateZ(a - Math.PI / 2);
    box.translate(0, 0, z);
    return faceted(THREE, box, edges);
}

// A double-row ball bearing in a round housing: inner race on the shaft, the
// balls, the outer race.
function rollingBearing(THREE, part, edges) {
    const { r, w, mid } = onNode(part);
    const { pieces } = bearingHousing(THREE, part, 1.42 * r, edges);
    pieces.push({ geometry: ring(THREE, 1.34 * r, 1.42 * r, mid - 0.4 * w, mid + 0.4 * w, edges), color: RACE, finish: 'metal' });
    pieces.push({ geometry: ring(THREE, 1.02 * r, 1.16 * r, mid - 0.42 * w, mid + 0.42 * w, edges), color: RACE, finish: 'metal' });
    const ball = 0.085 * r;
    const pitch = 1.25 * r;
    const count = Math.max(8, Math.floor((2 * Math.PI * pitch) / (2.4 * ball)));
    // Each row just inside a face of the races, where the balls show between
    // them -- set deeper, they were a dark ring.
    const row = 0.4 * w - 1.3 * ball;
    for (const side of [-1, 1]) {
        for (let i = 0; i < count; i++) {
            const a = (i / count) * Math.PI * 2 + (side > 0 ? Math.PI / count : 0);
            const sphere = new THREE.SphereGeometry(ball, 12, 8);
            sphere.translate(pitch * Math.cos(a), pitch * Math.sin(a), mid + side * row);
            pieces.push(steel(faceted(THREE, sphere, null)));
        }
    }
    return pieces;
}

// A tilting-pad journal bearing: five pads on their pivots, in the casing.
function tiltingPad(THREE, part, edges) {
    const { r, w, mid } = onNode(part);
    const { pieces } = bearingHousing(THREE, part, 1.4 * r, edges);
    const pads = 5;
    const arc = (56 * Math.PI) / 180;
    for (let i = 0; i < pads; i++) {
        // Load between pads: the bottom of the bearing is a gap, not a pad.
        const centre = -Math.PI / 2 + Math.PI / pads + (i / pads) * Math.PI * 2;
        pieces.push({ geometry: sector(THREE, 1.04 * r, 1.26 * r, centre - arc / 2, centre + arc / 2, 0.7 * w, mid, edges),
            color: WHITE_METAL, finish: 'metal' });
        pieces.push(steel(radialBox(THREE, 1.25 * r, 1.41 * r, centre, 0.12 * r, 0.2 * w, mid, edges)));
    }
    return pieces;
}

// An active magnetic bearing: the stator's poles and their windings round a
// laminated sleeve on the shaft.
function magneticBearing(THREE, part, edges) {
    const { r, w, mid } = onNode(part);
    const { pieces } = bearingHousing(THREE, part, 1.46 * r, edges);
    pieces.push(steel(ring(THREE, 1.3 * r, 1.46 * r, mid - 0.36 * w, mid + 0.36 * w, edges)));
    pieces.push({ geometry: ring(THREE, 1.0 * r, 1.07 * r, mid - 0.32 * w, mid + 0.32 * w, edges), color: RACE, finish: 'metal' });
    const poles = 8;
    for (let i = 0; i < poles; i++) {
        const a = (i / poles) * Math.PI * 2 + Math.PI / poles;
        pieces.push(steel(radialBox(THREE, 1.1 * r, 1.31 * r, a, 0.2 * r, 0.6 * w, mid, edges)));
        pieces.push({ geometry: radialBox(THREE, 1.14 * r, 1.28 * r, a, 0.34 * r, 0.7 * w, mid, edges), color: COPPER, finish: 'metal' });
    }
    return pieces;
}

// --- seals --------------------------------------------------------------------------------

// A brush seal: a pack of bristles laid at an angle to the shaft, between a
// front plate and a backing plate.
function brushSeal(THREE, part, edges) {
    const { r, R, w, mid } = onNode(part);
    const pieces = [
        own(ring(THREE, 1.08 * r, R, mid - w / 2, mid - 0.3 * w, edges, SEGMENTS.part, 0.03 * R)),
        own(ring(THREE, 1.16 * r, R, mid + 0.1 * w, mid + w / 2, edges, SEGMENTS.part, 0.03 * R)),
    ];
    const bristles = 96;
    for (let i = 0; i < bristles; i++) {
        const a = (i / bristles) * Math.PI * 2;
        pieces.push({ geometry: radialBox(THREE, 1.005 * r, 0.97 * R, a, 0.035 * r, 0.4 * w, mid - 0.1 * w, null, Math.PI / 4.5),
            color: BRISTLE, finish: 'metal' });
    }
    return pieces;
}

// Hole-pattern and honeycomb seals: what makes them what they are is on the
// bore, facing the shaft across a clearance of a few per cent of its radius,
// where no view of the assembled rotor reaches -- measured: through a window
// cut in the stator, the shaft hides the bore. They are drawn as a drawing
// shows them, cut away: a quarter of the stator is left out, the one facing
// the camera as the view opens (x < 0, y > 0; `viewDirection` in
// core/rotor3d_layout.js), and the two faces of the cut show the section --
// the holes as dark pockets, the honeycomb as the comb of its cell walls. A
// flange on the face towards the rotor's end, as the labyrinth's, bolted.
const CUT = { from: Math.PI, to: 2.5 * Math.PI };

// Marks on both faces of the cut, one per z in `zs`: a thin plate in the
// face, `radial` long from `from`, `along` wide along the shaft, standing
// just proud of the face so it shows on it. Local x runs against the angle
// (see `radialBox`), so the side proud of each face is the window's.
function onCutFaces(THREE, from, radial, along, zs, thickness) {
    const marks = [];
    for (const [a, side] of [[CUT.from, 1], [CUT.to, -1]]) {
        for (const z of zs) {
            const mark = new THREE.BoxGeometry(thickness, radial, along);
            mark.translate((side * thickness) / 2, from + radial / 2, 0);
            mark.rotateZ(a - Math.PI / 2);
            mark.translate(0, 0, z);
            marks.push(faceted(THREE, mark, null));
        }
    }
    return marks;
}

// The rows of cells, along the shaft (`cellCentres`).
function cellRows(centres) {
    return Array.from(new Set(centres.map(([, z]) => z)));
}

function cutAwayStator(THREE, part, edges, inner) {
    const { r, R, w, mid } = onNode(part);
    const flangeFrom = mid + 0.3 * w;
    const pieces = [
        own(sector(THREE, inner, R, CUT.from, CUT.to, flangeFrom - (mid - w / 2), (mid - w / 2 + flangeFrom) / 2, edges)),
        own(sector(THREE, 1.02 * r, SEAL_FLANGE * R, CUT.from, CUT.to, mid + w / 2 - flangeFrom, (flangeFrom + mid + w / 2) / 2, edges)),
    ];
    const bolts = 6;
    for (let i = 0; i < bolts; i++) {
        const a = CUT.from + ((i + 0.5) / bolts) * (CUT.to - CUT.from);
        const rb = ((1 + SEAL_FLANGE) / 2) * R;
        pieces.push(steel(boltHead(THREE, 0.06 * R, 0.05 * R, 'z', rb * Math.cos(a), rb * Math.sin(a), mid + w / 2 + 0.025 * R, edges)));
    }
    return pieces;
}

// Where the cells go: round the bore within the cut (`CUT`, less a margin at
// each cut face) and along the stator short of its flange -- `pitch` apart,
// every other row moved by half a pitch and the rows sin 60 degrees of a pitch
// apart, so each cell has six neighbours at the same distance.
function cellCentres(part, radius, pitch) {
    const { w, mid } = onNode(part);
    const z0 = mid - w / 2 + 0.6 * pitch;
    const z1 = mid + 0.3 * w - 0.6 * pitch;
    const spacing = (Math.sqrt(3) / 2) * pitch;
    const rows = Math.max(1, Math.floor((z1 - z0) / spacing) + 1);
    const first = (z0 + z1) / 2 - ((rows - 1) * spacing) / 2;
    const step = pitch / radius;
    const margin = 0.6 * step;
    const centres = [];
    for (let k = 0; k < rows; k++) {
        const z = first + k * spacing;
        for (let a = CUT.from + margin + (k % 2 ? step / 2 : 0); a <= CUT.to - margin; a += step) centres.push([a, z]);
    }
    return centres;
}

// A hole-pattern (damper) seal: a smooth stator whose bore is drilled with a
// close pattern of round holes. Each hole is a dark disk on the bore.
function holePatternSeal(THREE, part, edges) {
    const { r } = onNode(part);
    const bore = 1.04 * r;
    const pieces = cutAwayStator(THREE, part, edges, bore);
    const hole = 0.045 * r;
    const centres = cellCentres(part, bore, 0.14 * r);
    for (const [a, z] of centres) {
        const disk = new THREE.CylinderGeometry(hole, hole, 0.004 * r, 12);
        disk.translate(0, bore - 0.003 * r, 0);
        disk.rotateZ(a - Math.PI / 2);
        disk.translate(0, 0, z);
        pieces.push({ geometry: faceted(THREE, disk, null), color: HOLE, finish: 'paint' });
    }
    // In section, each row a pocket drilled from the bore.
    onCutFaces(THREE, bore, 0.09 * r, 2 * hole, cellRows(centres), 0.004 * r)
        .forEach(geometry => pieces.push({ geometry, color: HOLE, finish: 'paint' }));
    return pieces;
}

// A honeycomb seal: the stator's bore lined with hexagonal cells of thin
// foil, their walls standing from the backing to the bore.
function honeycombSeal(THREE, part, edges) {
    const { r } = onNode(part);
    const bore = 1.04 * r;
    const depth = 0.1 * r;
    const pieces = cutAwayStator(THREE, part, edges, bore + depth);
    const size = 0.075 * r;
    const foil = 0.01 * r;
    const middle = bore + depth / 2;
    // Three walls of each cell, the ones towards its neighbours at 0, 60 and
    // 120 degrees: every wall inside the pattern is drawn once.
    const walls = [0, 1, 2].map(k => (k * Math.PI) / 3);
    const centres = cellCentres(part, middle, Math.sqrt(3) * size);
    for (const [a, z] of centres) {
        for (const toward of walls) {
            const mx = Math.cos(toward) * size * 0.866;
            const mz = Math.sin(toward) * size * 0.866;
            const wall = new THREE.BoxGeometry(size, depth, foil);
            wall.rotateY(-(toward + Math.PI / 2));
            wall.translate(mx, middle, mz);
            wall.rotateZ(a - Math.PI / 2);
            wall.translate(0, 0, z);
            pieces.push(steel(faceted(THREE, wall, null)));
        }
    }
    // In section: the cell layer is open between the bore and the backing,
    // the walls crossing it a comb, a wall between each two rows and one on
    // each row.
    const rows = cellRows(centres);
    const between = rows.slice(1).map((z, k) => (z + rows[k]) / 2);
    onCutFaces(THREE, bore, depth, 2 * foil, rows.concat(between), 0.03 * r)
        .forEach(geometry => pieces.push(steel(geometry)));
    return pieces;
}

// --- point masses --------------------------------------------------------------------------

// A balance weight: a block on one side of the shaft, on a clamping band.
function balanceWeight(THREE, part, edges) {
    const { r, R, w, mid } = onNode(part);
    const pieces = [
        own(ring(THREE, 1.02 * r, 1.12 * r, mid - 0.45 * w, mid + 0.45 * w, edges), 'paint'),
        own(sector(THREE, 1.1 * r, R, Math.PI / 2 - 0.9, Math.PI / 2 + 0.9, 0.9 * w, mid, edges), 'paint'),
    ];
    for (const a of [Math.PI / 2 - 0.55, Math.PI / 2 + 0.55]) {
        const head = boltHead(THREE, 0.12 * r, 0.2 * r, 'y', 0, 0, 0, edges);
        head.translate(0, R + 0.05 * r, 0);
        head.rotateZ(a - Math.PI / 2);
        head.translate(0, 0, mid);
        pieces.push(steel(head));
    }
    return pieces;
}

// A lock nut and its washer.
function lockNut(THREE, part, edges) {
    const { r, R, w, mid } = onNode(part);
    const hex = new THREE.Shape();
    for (let i = 0; i <= 6; i++) {
        const a = (i / 6) * Math.PI * 2 + Math.PI / 6;
        if (i === 0) hex.moveTo(0.99 * R * Math.cos(a), 0.99 * R * Math.sin(a));
        else hex.lineTo(0.99 * R * Math.cos(a), 0.99 * R * Math.sin(a));
    }
    const hole = new THREE.Path();
    hole.absarc(0, 0, 1.02 * r, 0, Math.PI * 2, true);
    hex.holes.push(hole);
    const depth = 0.72 * w;
    const nut = new THREE.ExtrudeGeometry(hex, { depth, bevelEnabled: false, curveSegments: 24 });
    nut.translate(0, 0, mid - w / 2);
    // The nut in the element's colour, as the legend shows it; the washer steel.
    return [
        own(faceted(THREE, nut, edges)),
        steel(ring(THREE, 1.02 * r, 0.92 * R, mid - w / 2 + depth, mid + w / 2, edges)),
    ];
}

// --- couplings ------------------------------------------------------------------------------
//
// Drawn in the coupling's own length (`part.body`), with the shaft ends it
// clamps, as the disc-pack coupling is (rotor3d_parts.js).
function couplingFrame(part) {
    const [z0, z1] = part.body || [Math.min(part.z0, part.z1), Math.max(part.z0, part.z1)];
    const [bore0, bore1] = part.bores || [part.bore, part.bore];
    return { z0, z1, span: z1 - z0, R: part.radius, bore0, bore1, hub: part.hub };
}

// A gear coupling: a sleeve with a bolted flange in the middle, closed at the
// ends round the hubs.
function gearCoupling(THREE, part, edges) {
    const { z0, z1, span, R, bore0, bore1, hub } = couplingFrame(part);
    const mid = (z0 + z1) / 2;
    const flange = 0.14 * span;
    const pieces = couplingShaftEnds(THREE, part, edges);
    pieces.push(own(ring(THREE, Math.max(bore0 * 1.4, 0.55 * R), 0.62 * R, z0, z0 + 0.35 * hub, edges)));
    pieces.push(own(ring(THREE, Math.max(bore1 * 1.4, 0.55 * R), 0.62 * R, z1 - 0.35 * hub, z1, edges)));
    pieces.push(own(ring(THREE, 0.6 * R, 0.84 * R, z0 + 0.3 * hub, mid - flange / 2, edges, SEGMENTS.part, 0.03 * R)));
    pieces.push(own(ring(THREE, 0.6 * R, 0.84 * R, mid + flange / 2, z1 - 0.3 * hub, edges, SEGMENTS.part, 0.03 * R)));
    pieces.push(own(ring(THREE, 0.6 * R, R, mid - flange / 2, mid + flange / 2, edges, SEGMENTS.part, 0.03 * R)));
    for (const side of [-1, 1]) {
        boltCircle(THREE, 8, 0.92 * R, 0.05 * R, 0.04 * R, mid + side * (flange / 2 + 0.02 * R), edges)
            .forEach(g => pieces.push(steel(g)));
    }
    return pieces;
}

// A jaw coupling: two hubs whose jaws interleave, with the elastomer spider
// between them.
function jawCoupling(THREE, part, edges) {
    const { z0, z1, span, R, bore0, bore1 } = couplingFrame(part);
    const mid = (z0 + z1) / 2;
    const hubLength = 0.38 * span;
    const jawLength = 0.3 * span;
    const pieces = couplingShaftEnds(THREE, part, edges);
    pieces.push(own(ring(THREE, bore0, 0.9 * R, z0, z0 + hubLength, edges, SEGMENTS.part, 0.04 * R)));
    pieces.push(own(ring(THREE, bore1, 0.9 * R, z1 - hubLength, z1, edges, SEGMENTS.part, 0.04 * R)));
    const jaws = 3;
    const arc = (44 * Math.PI) / 180;
    for (let i = 0; i < jaws; i++) {
        const a = (i / jaws) * Math.PI * 2;
        pieces.push(own(sector(THREE, 0.42 * R, 0.9 * R, a - arc / 2, a + arc / 2, jawLength, z0 + hubLength + jawLength / 2 - 0.02 * span, edges)));
        const b = a + Math.PI / jaws;
        pieces.push(own(sector(THREE, 0.42 * R, 0.9 * R, b - arc / 2, b + arc / 2, jawLength, z1 - hubLength - jawLength / 2 + 0.02 * span, edges)));
    }
    pieces.push({ geometry: ring(THREE, 0.34 * R, 0.86 * R, mid - 0.12 * span, mid + 0.12 * span, edges), color: ELASTOMER, finish: 'paint' });
    return pieces;
}

// A rigid flange coupling: two flanged hubs bolted face to face.
function rigidCoupling(THREE, part, edges) {
    const { z0, z1, span, R, bore0, bore1 } = couplingFrame(part);
    const mid = (z0 + z1) / 2;
    const flange = 0.16 * span;
    const pieces = couplingShaftEnds(THREE, part, edges);
    pieces.push(own(ring(THREE, bore0, Math.max(bore0 * 1.5, 0.55 * R), z0, mid - flange, edges, SEGMENTS.part, 0.04 * R)));
    pieces.push(own(ring(THREE, bore1, Math.max(bore1 * 1.5, 0.55 * R), mid + flange, z1, edges, SEGMENTS.part, 0.04 * R)));
    pieces.push(own(ring(THREE, Math.min(bore0, bore1), R, mid - flange, mid, edges, SEGMENTS.part, 0.03 * R)));
    pieces.push(own(ring(THREE, Math.min(bore0, bore1), R, mid, mid + flange, edges, SEGMENTS.part, 0.03 * R)));
    for (const side of [-1, 1]) {
        boltCircle(THREE, 6, 0.8 * R, 0.06 * R, 0.045 * R, mid + side * (flange + 0.022 * R), edges, Math.PI / 6)
            .forEach(g => pieces.push(steel(g)));
    }
    return pieces;
}

// By key, as core/shapes3d.js names them. The keys are unique across
// categories; the layout only hands a part a key its own category offers.
export const SHAPE_BUILDERS = {
    impeller,
    closed_impeller: closedImpeller,
    axial: axialStage,
    turbine,
    fan,
    flywheel,
    pulley,
    rolling: rollingBearing,
    tilting_pad: tiltingPad,
    magnetic: magneticBearing,
    brush: brushSeal,
    hole_pattern: holePatternSeal,
    honeycomb: honeycombSeal,
    balance_weight: balanceWeight,
    nut: lockNut,
    gear_coupling: gearCoupling,
    jaw: jawCoupling,
    rigid: rigidCoupling,
};
