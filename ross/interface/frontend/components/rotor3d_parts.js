// The solids of the 3D view, built from the layout (`core/rotor3d_layout.js`).
//
// three.js is handed in, not imported: this module is loaded by node in the
// tests with the real library, and by the page only once the 3D view is
// opened, so that nobody who never opens it pays for 2 MB of it.
//
// What the prototype taught, measured (see the project's 3D evaluation):
//
// * **few geometries for the whole rotor.** Every part is merged, with its
//   colour in the vertices, into one mesh per finish (bare metal, painted) and
//   one set of outlines -- a handful of draw calls whatever the number of
//   elements, where the prototype spent about five per shaft element and
//   seventeen per bearing;
// * **everything built here is disposed here.** The prototype removed the old
//   meshes from the scene and never freed them: eleven times the geometries on
//   the GPU after twenty edits. `dispose()` below frees every geometry and
//   material this module created, and nothing it did not create;
// * **detail stays**, as decided with Leonardo, and after the first look at it
//   he asked for more: chamfered shafts, disks with hub, tapered web, rim and a
//   bolt circle, gears with involute teeth, lightening holes and a keyway,
//   pillow blocks with their bolts and grease nipple, ribbed seal glands,
//   couplings as a flanged hub on each node with a disc pack and a spacer.
//
// Solids of revolution are built by `revolve` (rotor3d_solids.js) rather than three.js's
// LatheGeometry: Lathe smooths the normal across every corner of the profile,
// so a shaft's end face and its side shaded as one rounded surface. Here each
// segment of the profile keeps its own normal -- smooth around the axis, sharp
// at the corners -- and the corners are where the outlines are drawn.

import { SEAL_FLANGE, SYMBOL } from '../core/rotor3d_layout.js';
import {
    EDGE_ANGLE, SEGMENTS, bearingHousing, boltCircle, boltHead, couplingShaftEnds, faceted, own, revolve, ring, steel,
} from './rotor3d_solids.js';
import { motorPieces } from './rotor3d_motor.js';
import { SHAPE_BUILDERS } from './rotor3d_shapes.js';

// A gear with more teeth than this is drawn with this many: past it the teeth
// are finer than a pixel at any sensible zoom, and each one costs vertices.
export const MAX_DRAWN_TEETH = 160;

// Colours of what ROSS does not colour: bolts, the bearing's bushing, a
// coupling's disc pack. As numbers: the page's own colours are CSS tokens, and
// these are materials, not interface.
const BRASS = 0xc39a45;
const DISC_PACK = 0x8e98a2;

// --- one builder per kind -------------------------------------------------------
//
// Each returns a list of { geometry, color, finish } (rotor3d_solids.js). A
// part given a shape from the geometry bank (core/shapes3d.js) is drawn as
// that shape instead, by rotor3d_shapes.js.

// One stretch of shaft from za to zb, radii read off the element's taper, its
// ends chamfered where `chamfered` says.
function shaftStretch(THREE, part, za, zb, chamfered, edges) {
    const { odl, odr, idl, idr } = part.profile;
    const along = z => (part.z1 === part.z0 ? 0 : (z - part.z0) / (part.z1 - part.z0));
    const outer = z => (odl + (odr - odl) * along(z)) / 2;
    const inner = z => (idl + (idr - idl) * along(z)) / 2;
    const ro0 = outer(za);
    const ro1 = outer(zb);
    const ri0 = inner(za);
    const ri1 = inner(zb);
    // A chamfer of 6 % of the radius, never more than a fifth of the length.
    const c0 = chamfered[0] ? Math.min(0.06 * ro0, (zb - za) / 5) : 0;
    const c1 = chamfered[1] ? Math.min(0.06 * ro1, (zb - za) / 5) : 0;
    const profile = [[ri0, za], [ro0 - c0, za]];
    if (c0) profile.push([ro0, za + c0]);
    if (c1) profile.push([ro1, zb - c1], [ro1 - c1, zb]); else profile.push([ro1, zb]);
    profile.push([ri1, zb], [ri0, za]);
    return own(revolve(THREE, profile, SEGMENTS.shaft, edges));
}

// A shaft element, less where a coupling separates the shaft line
// (`part.cuts`, from the layout): the line stops inside the coupling's hub and
// starts again inside the other, as two shaft ends joined by the coupling.
// An end that stands proud is chamfered (the layout says which), and so is
// every end a cut makes.
function shaftPieces(THREE, part, edges) {
    let stretches = [[part.z0, part.z1, !!(part.chamfer && part.chamfer[0]), !!(part.chamfer && part.chamfer[1])]];
    for (const [g0, g1] of part.cuts || []) {
        const next = [];
        for (const [a, b, ca, cb] of stretches) {
            if (g1 <= a || g0 >= b) { next.push([a, b, ca, cb]); continue; }
            if (g0 > a) next.push([a, g0, ca, true]);
            if (g1 < b) next.push([g1, b, true, cb]);
        }
        stretches = next;
    }
    return stretches
        .filter(([a, b]) => b - a > 1e-9)
        .map(([a, b, ca, cb]) => shaftStretch(THREE, part, a, b, [ca, cb], edges));
}

// A wheel: hub the full width, a web that thins from hub to rim, the rim, and
// a circle of bolts on both faces of the web. A symbolic disk (no size in
// ROSS) is a plain thin ring.
//
// A disk given a shape from the geometry bank (core/shapes3d.js) is drawn as
// that shape instead, in the same envelope (rotor3d_shapes.js).
function diskPieces(THREE, part, edges) {
    const shaped = !part.symbolic && part.shape && SHAPE_BUILDERS[part.shape];
    if (shaped) return shaped(THREE, part, edges);
    const z = (part.z0 + part.z1) / 2;
    const w = Math.abs(part.z1 - part.z0);
    const ri = Math.min(part.bore, part.radius * 0.9);
    const ro = part.radius;
    if (part.symbolic || ro - ri < 1e-6) return [own(ring(THREE, ri, ro, z - w / 2, z + w / 2, edges))];
    const dr = ro - ri;
    const rHub = ri + dr * 0.26;
    const rRim = ro - dr * 0.2;
    const hub = w / 2;
    const webIn = w * 0.24;
    const webOut = w * 0.15;
    const rim = w * 0.44;
    const ch = Math.min(dr, w) * 0.06;
    const profile = [
        [ri, z - hub], [rHub - ch, z - hub], [rHub, z - hub + ch], [rHub, z - webIn],
        [rRim, z - webOut], [rRim, z - rim + ch], [rRim + ch, z - rim], [ro - ch, z - rim], [ro, z - rim + ch],
        [ro, z + rim - ch], [ro - ch, z + rim], [rRim + ch, z + rim], [rRim, z + rim - ch], [rRim, z + webOut],
        [rHub, z + webIn], [rHub, z + hub - ch], [rHub - ch, z + hub], [ri, z + hub], [ri, z - hub],
    ];
    const pieces = [own(revolve(THREE, profile, SEGMENTS.part, edges))];
    const rBolts = rHub + (rRim - rHub) * 0.3;
    const size = Math.min(dr * 0.045, w * 0.3);
    // The web's face at the bolt circle, read off its taper.
    const t = (rBolts - rHub) / (rRim - rHub);
    const face = webIn + (webOut - webIn) * t;
    for (const side of [-1, 1]) {
        boltCircle(THREE, 8, rBolts, size, size * 0.7, z + side * (face + size * 0.3), edges)
            .forEach(g => pieces.push(steel(g)));
    }
    return pieces;
}

// One tooth space after another: an involute flank up from the root (radial
// below the base circle), across the tip, and down the other flank. The
// thickness at the pitch circle is half the circular pitch.
function toothOutline(THREE, g) {
    const teeth = Math.min(Math.max(Math.round(g.teeth) || 24, 6), MAX_DRAWN_TEETH);
    const pressure = (20 * Math.PI) / 180;
    const base = g.pitch * Math.cos(pressure);
    const involute = a => Math.tan(a) - a;
    const halfAtPitch = Math.PI / (2 * teeth);
    const halfAt = r => {
        if (r <= base) return halfAtPitch + involute(pressure);
        return halfAtPitch + involute(pressure) - involute(Math.acos(base / r));
    };
    const samples = 5;
    const radii = [];
    for (let s = 0; s <= samples; s++) radii.push(g.root + ((g.tip - g.root) * s) / samples);
    const shape = new THREE.Shape();
    const step = (2 * Math.PI) / teeth;
    let first = true;
    const at = (r, a) => {
        const x = Math.cos(a) * r;
        const y = Math.sin(a) * r;
        if (first) { shape.moveTo(x, y); first = false; } else shape.lineTo(x, y);
    };
    for (let i = 0; i < teeth; i++) {
        const centre = i * step;
        // The tooth may not be wider than its share of the circle at the root.
        const limit = step * 0.48;
        radii.forEach(r => at(r, centre - Math.min(halfAt(r), limit)));
        [...radii].reverse().forEach(r => at(r, centre + Math.min(halfAt(r), limit)));
        // Along the root to the next tooth.
        at(g.root, centre + step / 2);
    }
    shape.closePath();
    return shape;
}

// A bore with a keyway, as a hole path (clockwise).
function boreWithKeyway(THREE, radius) {
    const path = new THREE.Path();
    const half = radius * 0.22;
    const depth = radius * 0.2;
    const a = Math.asin(Math.min(half / radius, 0.9));
    path.moveTo(Math.cos(Math.PI / 2 - a) * radius, Math.sin(Math.PI / 2 - a) * radius);
    path.absarc(0, 0, radius, Math.PI / 2 - a, Math.PI / 2 + a - 2 * Math.PI, true);
    path.lineTo(-half, radius + depth);
    path.lineTo(half, radius + depth);
    path.closePath();
    return path;
}

// The toothed piece of a helical gear: each section along the face turned
// about the axis by as much as the helix advances there -- tan(helix) / pitch
// radius per unit of length -- so a tooth winds round the gear. The extrusion
// is cut in `steps` along the face for the turn to have somewhere to bend;
// enough that no step turns a tooth by more than a few hundredths of a
// radian. The turn fades in from the piece's inner edge `inner` to the root of
// the teeth, so a bore and its keyway stay straight.
function helicalSteps(part) {
    const turn = Math.abs(part.faceWidth * Math.tan(part.helix) / part.gear.pitch);
    return Math.min(Math.max(Math.ceil(turn / 0.03), 2), 16);
}

function twist(geometry, part, centre, inner) {
    const rate = Math.tan(part.helix) / part.gear.pitch;
    const root = part.gear.root;
    const p = geometry.attributes.position;
    for (let i = 0; i < p.count; i++) {
        const x = p.getX(i);
        const y = p.getY(i);
        const reach = root > inner ? Math.min(Math.max((Math.hypot(x, y) - inner) / (root - inner), 0), 1) : 1;
        const a = (p.getZ(i) - centre) * rate * reach;
        p.setXY(i, x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a));
    }
    return geometry;
}

// A spur gear -- or, when the layout gives it a helix angle (the geometry
// bank's helical gear), a helical one: the same gear with its teeth wound
// (`twist`). A small one is solid; a large one is a toothed rim on a thin
// web with lightening holes, around a hub that stands proud of the faces. A
// pinion cut on its shaft (see `gearDrawing`) has no bore of its own.
function gearPieces(THREE, part, edges) {
    const g = part.gear;
    const z = (part.z0 + part.z1) / 2;
    const w = part.faceWidth;
    const hubHalf = Math.abs(part.z1 - part.z0) / 2;
    // `inner`: for a toothed piece, where its teeth's turn starts (`twist`).
    const extrude = (shape, depth, at, inner = null) => {
        const helical = inner !== null && part.helix;
        const geometry = new THREE.ExtrudeGeometry(shape, {
            depth, bevelEnabled: false, curveSegments: 20, steps: helical ? helicalSteps(part) : 1,
        });
        geometry.translate(0, 0, at - depth / 2);
        return faceted(THREE, helical ? twist(geometry, part, at, inner) : geometry, edges);
    };
    const pieces = [];
    if (g.onShaft) {
        // Teeth only, on the shaft's surface.
        const shape = toothOutline(THREE, g);
        const hole = new THREE.Path();
        hole.absarc(0, 0, g.root * 0.999, 0, Math.PI * 2, true);
        shape.holes.push(hole);
        pieces.push(own(extrude(shape, w, z, g.root * 0.999)));
        return pieces;
    }
    const rHub = Math.min(g.bore + (g.root - g.bore) * 0.32, g.bore * 1.8);
    const rimInner = g.root - Math.max((g.root - g.bore) * 0.14, 2.2 * g.module);
    const webbed = rimInner - rHub > 0.9 * w && rimInner - rHub > 3 * g.module;
    const outline = toothOutline(THREE, g);
    if (!webbed) {
        outline.holes.push(boreWithKeyway(THREE, g.bore));
        pieces.push(own(extrude(outline, w, z, g.bore)));
        pieces.push(own(ring(THREE, g.bore, Math.max(rHub, g.bore * 1.25), z - hubHalf, z - w / 2, edges)));
        pieces.push(own(ring(THREE, g.bore, Math.max(rHub, g.bore * 1.25), z + w / 2, z + hubHalf, edges)));
        return pieces;
    }
    // The rim: the teeth around a ring.
    const inner = new THREE.Path();
    inner.absarc(0, 0, rimInner, 0, Math.PI * 2, true);
    outline.holes.push(inner);
    pieces.push(own(extrude(outline, w, z, rimInner)));
    // The web, thinner, with lightening holes.
    const web = new THREE.Shape();
    web.absarc(0, 0, rimInner * 1.001, 0, Math.PI * 2, false);
    const webHole = new THREE.Path();
    webHole.absarc(0, 0, rHub * 0.999, 0, Math.PI * 2, true);
    web.holes.push(webHole);
    const holes = rimInner / rHub > 3 ? 6 : 4;
    const rHoles = (rHub + rimInner) / 2;
    const holeRadius = Math.min((rimInner - rHub) * 0.3, (Math.PI * rHoles) / holes * 0.32);
    for (let i = 0; i < holes; i++) {
        const a = (i / holes) * Math.PI * 2 + Math.PI / holes;
        const hole = new THREE.Path();
        hole.absarc(Math.cos(a) * rHoles, Math.sin(a) * rHoles, holeRadius, 0, Math.PI * 2, true);
        web.holes.push(hole);
    }
    pieces.push(own(extrude(web, w * 0.3, z)));
    // The hub, with its keyway.
    const hubShape = new THREE.Shape();
    hubShape.absarc(0, 0, rHub, 0, Math.PI * 2, false);
    hubShape.holes.push(boreWithKeyway(THREE, g.bore));
    pieces.push(own(extrude(hubShape, 2 * hubHalf, z)));
    return pieces;
}

// Drawn to the side at x = ±offset: a half the coupling, as a hub on its node
// and a flange facing the span, bolted, with a disc pack against it.
function couplingPieces(THREE, part, edges) {
    const shaped = part.shape && SHAPE_BUILDERS[part.shape];
    if (shaped) return shaped(THREE, part, edges);
    // The coupling's own length (`couplingBody`), in the middle of its span.
    const [z0, z1] = part.body || [Math.min(part.z0, part.z1), Math.max(part.z0, part.z1)];
    const span = z1 - z0;
    const R = part.radius;
    const [bore0, bore1] = part.bores || [part.bore, part.bore];
    const hubLength = part.hub;
    const flange = Math.min(0.35 * hubLength, 0.25 * R);
    const pack = Math.min(0.05 * span, 0.08 * R);
    const pieces = [];
    const half = (bore, from, sign) => {
        const hubRadius = Math.max(bore * 1.45, 0.6 * R);
        const flangeAt = from + sign * (hubLength - flange);
        const end = from + sign * hubLength;
        const lo = Math.min(from, flangeAt);
        const hi = Math.max(from, flangeAt);
        pieces.push(own(ring(THREE, bore, hubRadius, lo, hi, edges, SEGMENTS.part, 0.08 * hubRadius)));
        pieces.push(own(ring(THREE, bore, R, Math.min(flangeAt, end), Math.max(flangeAt, end), edges, SEGMENTS.part, 0.04 * R)));
        const packFrom = end;
        const packTo = end + sign * pack;
        pieces.push({ geometry: ring(THREE, bore * 1.1, R * 0.86, Math.min(packFrom, packTo), Math.max(packFrom, packTo), edges),
            color: DISC_PACK, finish: 'metal' });
        boltCircle(THREE, 6, R * 0.78, R * 0.07, R * 0.05, flangeAt - sign * R * 0.025, edges, Math.PI / 6)
            .forEach(g => pieces.push(steel(g)));
    };
    half(bore0, z0, 1);
    half(bore1, z1, -1);
    pieces.push(...couplingShaftEnds(THREE, part, edges));
    // The spacer between the two disc packs: a slender tube.
    const spacerFrom = z0 + hubLength + pack;
    const spacerTo = z1 - hubLength - pack;
    if (spacerTo > spacerFrom) {
        const spacer = Math.max(Math.max(bore0, bore1) * 1.25, 0.45 * R);
        pieces.push(own(ring(THREE, spacer * 0.7, spacer, spacerFrom, spacerTo, edges)));
    }
    return pieces;
}

// A pillow block, seen from the front: a round cap over the shaft, a body that
// widens to two feet, a base; extruded along the shaft (`bearingHousing`). A
// brass bushing shows in the bore; hex bolts hold the cap and the feet; a
// grease nipple on top.
function bearingPieces(THREE, part, edges) {
    const shaped = part.shape && SHAPE_BUILDERS[part.shape];
    if (shaped) return shaped(THREE, part, edges);
    const r = part.shaftRadius;
    const z = (part.z0 + part.z1) / 2;
    const w = Math.abs(part.z1 - part.z0);
    const { top } = SYMBOL.bearing;
    const { pieces } = bearingHousing(THREE, part, 1.25 * r, edges);
    // The bushing, proud of the housing on both faces.
    pieces.push({ geometry: ring(THREE, r * 1.01, r * 1.25, z - w / 2, z + w / 2, edges), color: BRASS, finish: 'metal' });
    // Cap bolts, vertical, where the cap meets the body.
    for (const x of [-1.2 * r, 1.2 * r]) {
        const y = Math.sqrt(Math.max(part.radius * part.radius - x * x, 0));
        pieces.push(steel(boltHead(THREE, 0.16 * r, 0.3 * r, 'y', x, y + 0.1 * r, z, edges)));
    }
    // The grease nipple, from inside the cap up to the top SYMBOL.bearing gives.
    const stemFrom = part.radius - 0.05 * r;
    const stem = new THREE.CylinderGeometry(0.07 * r, 0.1 * r, top * r - stemFrom, 10);
    stem.translate(0, (top * r + stemFrom) / 2, z);
    pieces.push({ geometry: faceted(THREE, stem, null), color: BRASS, finish: 'metal' });
    return pieces;
}

// A labyrinth gland: a ribbed ring round the shaft, and a flange with its
// bolts on one face.
function sealPieces(THREE, part, edges) {
    const shaped = part.shape && SHAPE_BUILDERS[part.shape];
    if (shaped) return shaped(THREE, part, edges);
    const r = part.shaftRadius;
    const z = (part.z0 + part.z1) / 2;
    const w = Math.abs(part.z1 - part.z0);
    const R = part.radius;
    const ribs = 4;
    const bodyFrom = z - w / 2;
    const bodyTo = z + w * 0.28;
    const pitch = (bodyTo - bodyFrom) / ribs;
    const groove = R - 0.06 * (R - r) - 0.04 * r;
    const profile = [[r * 1.02, bodyFrom], [R, bodyFrom]];
    for (let i = 0; i < ribs; i++) {
        const a = bodyFrom + i * pitch;
        profile.push([R, a + pitch * 0.6], [groove, a + pitch * 0.6], [groove, a + pitch]);
        if (i < ribs - 1) profile.push([R, a + pitch]);
    }
    profile.push([groove, bodyTo], [r * 1.02, bodyTo], [r * 1.02, bodyFrom]);
    const pieces = [own(revolve(THREE, profile, SEGMENTS.part, edges))];
    pieces.push(own(ring(THREE, r * 1.02, R * SEAL_FLANGE, bodyTo, z + w / 2, edges, SEGMENTS.part, 0.03 * R)));
    boltCircle(THREE, 6, R * (1 + SEAL_FLANGE) / 2, 0.06 * R, 0.05 * R, z + w / 2 + 0.025 * R, edges)
        .forEach(g => pieces.push(steel(g)));
    return pieces;
}

// A clamped collar: the ring, its lug on top and the clamp screw.
function pointMassPieces(THREE, part, edges) {
    const shaped = part.shape && SHAPE_BUILDERS[part.shape];
    if (shaped) return shaped(THREE, part, edges);
    const r = part.shaftRadius;
    const z = (part.z0 + part.z1) / 2;
    const w = Math.abs(part.z1 - part.z0);
    const lug = new THREE.BoxGeometry(r * 0.5, r * 0.55, w * 0.7);
    lug.translate(0, part.radius + r * 0.2, z);
    return [
        own(ring(THREE, r * 1.02, part.radius, z - w / 2, z + w / 2, edges, SEGMENTS.part, 0.05 * part.radius), 'paint'),
        own(faceted(THREE, lug, edges), 'paint'),
        steel(boltHead(THREE, 0.14 * r, 0.6 * r, 'x', r * 0.32, part.radius + r * 0.22, z, edges)),
    ];
}

// A proximity probe, built pointing up (+y) from its tip and then turned to
// where it reads: its tip, the threaded body with two lock nuts, and the cable,
// in the probe's colour. Sizes in shaft radii (`SYMBOL.probe`).
const PROBE_TIP = 0x2b2b2b;

function probePieces(THREE, part, edges) {
    const r = part.shaftRadius;
    const { gap } = SYMBOL.probe;
    const tip = 0.3 * r;
    const body = 1.5 * r;
    const cable = 0.5 * r;
    const along = (radius, height, centre, sides = SEGMENTS.small) => {
        const geometry = new THREE.CylinderGeometry(radius, radius, height, sides);
        geometry.translate(0, centre, 0);
        return geometry;
    };
    const start = part.axial ? gap * r : (1 + gap) * r;
    const pieces = [
        { geometry: along(0.2 * r, tip, start + tip / 2), color: PROBE_TIP, finish: 'paint' },
        steel(along(0.13 * r, body, start + tip + body / 2)),
        steel(faceted(THREE, along(0.27 * r, 0.14 * r, start + tip + 0.35 * body, 6), edges)),
        steel(faceted(THREE, along(0.27 * r, 0.14 * r, start + tip + 0.75 * body, 6), edges)),
        own(along(0.07 * r, cable, start + tip + body + cable / 2), 'paint'),
    ];
    const z = (part.z0 + part.z1) / 2;
    for (const piece of pieces) {
        if (part.axial) {
            // +y turned to the probe's side of the node along the axis, lifted
            // over the shaft, and moved to the node's plane.
            piece.geometry.rotateX(part.outward * Math.PI / 2);
            piece.geometry.translate(0, part.lift, z - part.outward * ((part.z1 - part.z0) / 2 + gap * r));
        } else {
            // +y is 90 degrees: turned to the probe's angle from x.
            piece.geometry.rotateZ(part.angle - Math.PI / 2);
            piece.geometry.translate(0, 0, z);
        }
    }
    return pieces;
}

const BUILDERS = {
    shaft: shaftPieces,
    disk: diskPieces,
    gear: gearPieces,
    coupling: couplingPieces,
    bearing: bearingPieces,
    seal: sealPieces,
    pointmass: pointMassPieces,
    probe: probePieces,
};

// A colour ROSS names ('Firebrick', or a hex code) as three.js linear RGB. An
// unknown name, or none, is ROSS's shaft grey (0x525252) rather than white.
function colorOf(THREE, cache, name) {
    const text = name || '';
    if (!cache.has(text)) {
        const color = new THREE.Color(0x525252);
        const named = THREE.Color.NAMES[text.toLowerCase()];
        if (named !== undefined) color.setHex(named);
        else if (/^#[0-9a-f]{3}([0-9a-f]{3})?$/i.test(text)) color.setStyle(text);
        cache.set(text, color);
    }
    return cache.get(text);
}

// Merge plain geometries into one, with a colour per vertex. Every piece is
// made non-indexed first, so position, normal and colour line up one to one;
// the pieces themselves are freed as soon as they are copied.
export function mergeColoured(THREE, pieces) {
    let count = 0;
    const flat = pieces.map(({ geometry, color }) => {
        const plain = geometry.index ? geometry.toNonIndexed() : geometry;
        if (plain !== geometry) geometry.dispose();
        if (!plain.attributes.normal) plain.computeVertexNormals();
        count += plain.attributes.position.count;
        return { plain, color };
    });
    const position = new Float32Array(count * 3);
    const normal = new Float32Array(count * 3);
    const colour = new Float32Array(count * 3);
    let at = 0;
    for (const { plain, color } of flat) {
        const n = plain.attributes.position.count;
        position.set(plain.attributes.position.array, at * 3);
        normal.set(plain.attributes.normal.array, at * 3);
        for (let i = 0; i < n; i++) {
            colour[(at + i) * 3] = color.r;
            colour[(at + i) * 3 + 1] = color.g;
            colour[(at + i) * 3 + 2] = color.b;
        }
        at += n;
        plain.dispose();
    }
    const merged = new THREE.BufferGeometry();
    merged.setAttribute('position', new THREE.BufferAttribute(position, 3));
    merged.setAttribute('normal', new THREE.BufferAttribute(normal, 3));
    merged.setAttribute('color', new THREE.BufferAttribute(colour, 3));
    merged.computeBoundingSphere();
    return merged;
}

// The outlines: the circles `revolve` marked at its corners, moved like their
// part, and the sharp edges of every faceted piece -- all in one set of lines.
function outlineLines(THREE, marks) {
    const chunks = [];
    let count = 0;
    for (const mark of marks) {
        if (mark.circle) {
            const [r, z] = mark.circle;
            const n = mark.segments;
            const points = new Float32Array(n * 6);
            for (let i = 0; i < n; i++) {
                const a = (i / n) * Math.PI * 2;
                const b = ((i + 1) / n) * Math.PI * 2;
                points.set([Math.cos(a) * r + mark.dx, Math.sin(a) * r + mark.dy, z + mark.dz,
                    Math.cos(b) * r + mark.dx, Math.sin(b) * r + mark.dy, z + mark.dz], i * 6);
            }
            chunks.push(points);
            count += points.length;
        } else {
            const lines = new THREE.EdgesGeometry(mark.geometry, EDGE_ANGLE);
            const points = lines.attributes.position.array;
            chunks.push(points);
            count += points.length;
            lines.dispose();
        }
    }
    const all = new Float32Array(count);
    let at = 0;
    for (const chunk of chunks) { all.set(chunk, at); at += chunk.length; }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(all, 3));
    return geometry;
}

// The circles that mark the nodes on the shaft surface, as one set of lines.
function ringLines(THREE, rings, segments = 40) {
    const points = new Float32Array(rings.length * segments * 6);
    let k = 0;
    for (const node of rings) {
        const r = node.radius * 1.004;
        for (let i = 0; i < segments; i++) {
            const a = (i / segments) * Math.PI * 2;
            const b = ((i + 1) / segments) * Math.PI * 2;
            points.set([node.x + Math.cos(a) * r, node.y + Math.sin(a) * r, node.z,
                node.x + Math.cos(b) * r, node.y + Math.sin(b) * r, node.z], k);
            k += 6;
        }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(points, 3));
    return geometry;
}

// The test bench (`benchLayout`): the bed plate with its T-slots, the legs,
// a pedestal with a foot plate under each bearing, and the motor that drives
// the rotor (rotor3d_motor.js). Painted, in the greys of a workshop, but for
// the motor's shaft and coupling; built and freed apart from the rotor, since
// it comes and goes with a button.
const BENCH = { plate: 0x7d8894, slot: 0x2c343d, leg: 0x3b4450, pedestal: 0x5a6571, rubber: 0x1c1f23, mount: 0x9aa4ae };

export function buildBench(THREE, bench) {
    const colors = new Map();
    const tone = hex => colorOf(THREE, colors, `#${hex.toString(16).padStart(6, '0')}`);
    const pieces = [];
    const edges = [];
    const block = (sx, sy, sz, x, y, z, hex) => {
        const geometry = new THREE.BoxGeometry(sx, sy, sz);
        geometry.translate(x, y, z);
        pieces.push({ geometry: faceted(THREE, geometry, edges), color: tone(hex), finish: 'paint' });
    };
    const { plate } = bench;
    const plateWidth = plate.x1 - plate.x0;
    const plateLength = plate.z1 - plate.z0;
    const cx = (plate.x0 + plate.x1) / 2;
    const cz = (plate.z0 + plate.z1) / 2;
    const thickness = plate.top - plate.bottom;
    block(plateWidth, thickness, plateLength, cx, (plate.top + plate.bottom) / 2, cz, BENCH.plate);
    // Three T-slots along the plate.
    for (const f of [-0.28, 0, 0.28]) {
        block(0.035 * plateWidth, 0.004 * plateWidth, 0.97 * plateLength, cx + f * plateWidth, plate.top + 0.002 * plateWidth, cz, BENCH.slot);
    }
    // A round piece of the isolator: a cylinder with its axis vertical.
    const disc = (radius, height, x, y, z, hex) => {
        const geometry = new THREE.CylinderGeometry(radius, radius, height, 28);
        geometry.translate(x, y, z);
        pieces.push({ geometry: faceted(THREE, geometry, edges), color: tone(hex), finish: 'paint' });
    };
    for (const leg of bench.legs) {
        const foot = plate.bottom - bench.legHeight;
        block(leg.size, bench.legHeight, leg.size, leg.x, plate.bottom - bench.legHeight / 2, leg.z, BENCH.leg);
        // The isolator under the leg: a square top plate, the rubber, a round base.
        const m = leg.mount;
        block(1.5 * leg.size, 0.15 * m, 1.5 * leg.size, leg.x, foot - 0.075 * m, leg.z, BENCH.mount);
        disc(0.7 * leg.size, 0.55 * m, leg.x, foot - 0.15 * m - 0.275 * m, leg.z, BENCH.rubber);
        disc(1.0 * leg.size, 0.3 * m, leg.x, foot - 0.7 * m - 0.15 * m, leg.z, BENCH.mount);
    }
    for (const p of bench.pedestals) {
        const height = p.y1 - p.y0;
        if (height > 1e-6) block(0.75 * p.width, height, 0.85 * p.depth, p.x, p.y0 + height / 2, p.z, BENCH.pedestal);
        // The foot it is bolted down with.
        const foot = Math.max(0.012 * plateWidth, 0.1 * p.width);
        block(1.05 * p.width, foot, 1.2 * p.depth, p.x, p.y0 + foot / 2, p.z, BENCH.pedestal);
    }
    if (bench.motor) {
        // Its circles are marked about the motor's axis; they move with it.
        const motorEdges = [];
        for (const piece of motorPieces(THREE, bench.motor, motorEdges)) pieces.push({ ...piece, color: tone(piece.color) });
        motorEdges.forEach(mark => edges.push(mark.circle ? { ...mark, dx: bench.motor.x, dy: bench.motor.y, dz: 0 } : mark));
    }
    // Before merging, which frees the pieces the outlines are read from.
    const outline = outlineLines(THREE, edges);
    const group = new THREE.Group();
    const made = [];
    for (const finish of ['paint', 'metal']) {
        const ofFinish = pieces.filter(piece => piece.finish === finish);
        if (!ofFinish.length) continue;
        const geometry = mergeColoured(THREE, ofFinish);
        const material = new THREE.MeshStandardMaterial({ vertexColors: true, ...FINISHES[finish] });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = finish === 'paint' ? 'bench' : 'bench-metal';
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
        made.push(geometry, material);
    }
    const lineMaterial = new THREE.LineBasicMaterial({ color: 'black', transparent: true, opacity: 0.3, depthWrite: false });
    const lines = new THREE.LineSegments(outline, lineMaterial);
    lines.name = 'bench-outlines';
    group.add(lines);
    made.push(outline, lineMaterial);
    return {
        object: group,
        dispose: () => {
            made.forEach(thing => thing.dispose());
            group.clear();
        },
    };
}

// How each finish looks. Bare metal takes its look from the environment the
// view lights the scene with; paint is duller.
const FINISHES = {
    metal: { metalness: 0.78, roughness: 0.3 },
    paint: { metalness: 0.15, roughness: 0.55 },
};

// The whole rotor as three.js objects: a mesh per finish, the outlines, and
// the node rings. `dispose` frees exactly what was made here.
//
// `look.ring` and `look.outline` are colours read from the page's theme by the
// caller: this module does not touch the DOM.
export function buildRotorModel(THREE, layout, look = {}) {
    const colors = new Map();
    const byFinish = { metal: [], paint: [] };
    const marks = [];
    for (const part of layout.parts) {
        const build = BUILDERS[part.kind];
        if (!build) continue;
        const partColor = colorOf(THREE, colors, part.color);
        const edges = [];
        const o = part.offset;
        for (const piece of build(THREE, part, edges)) {
            if (o.x || o.y || o.z) piece.geometry.translate(o.x, o.y, o.z);
            // null: the element's colour; a name: a colour ROSS gave something
            // else (a shaft end in a coupling); a number: a fixed material.
            const color = piece.color === null ? partColor
                : piece.named ? colorOf(THREE, colors, piece.color)
                    : colorOf(THREE, colors, `#${piece.color.toString(16).padStart(6, '0')}`);
            byFinish[piece.finish].push({ geometry: piece.geometry, color });
        }
        // Outlines of faceted pieces are read after the move; circles move here.
        for (const mark of edges) {
            if (mark.circle) marks.push({ ...mark, dx: o.x, dy: o.y, dz: o.z });
            else marks.push(mark);
        }
    }

    const group = new THREE.Group();
    const made = { geometries: [], materials: [] };
    const keep = (object, geometry, material) => {
        group.add(object);
        made.geometries.push(geometry);
        made.materials.push(material);
    };
    // Before merging, which frees the pieces the faceted outlines are read from.
    const outlines = marks.length ? outlineLines(THREE, marks) : null;
    for (const finish of ['metal', 'paint']) {
        if (!byFinish[finish].length) continue;
        const geometry = mergeColoured(THREE, byFinish[finish]);
        const material = new THREE.MeshStandardMaterial({ vertexColors: true, ...FINISHES[finish] });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = finish;
        mesh.castShadow = true;
        keep(mesh, geometry, material);
    }
    // The two that wear the theme's colours, kept to be repainted in place.
    const themed = { outline: null, ring: null };
    if (outlines) {
        const material = new THREE.LineBasicMaterial({
            color: look.outline || 'black', transparent: true, opacity: 0.35, depthWrite: false,
        });
        const lines = new THREE.LineSegments(outlines, material);
        lines.name = 'outlines';
        keep(lines, outlines, material);
        themed.outline = material;
    }
    if (layout.rings.length) {
        const geometry = ringLines(THREE, layout.rings);
        const material = new THREE.LineBasicMaterial({ color: look.ring || 'steelblue' });
        const lines = new THREE.LineSegments(geometry, material);
        lines.name = 'nodes';
        keep(lines, geometry, material);
        themed.ring = material;
    }

    const release = () => {
        made.geometries.forEach(g => g.dispose());
        made.materials.forEach(m => m.dispose());
        made.geometries = [];
        made.materials = [];
        group.clear();
    };
    // A change of theme repaints the outlines and the rings where they are:
    // rebuilding the rotor for two colours would cost a rebuild, and ROSS's
    // colours, which are most of what is drawn, do not change with the page.
    const restyle = colours => {
        if (themed.outline && colours.outline) themed.outline.color.set(colours.outline);
        if (themed.ring && colours.ring) themed.ring.color.set(colours.ring);
    };
    return {
        object: group,
        vertices: made.geometries.reduce((sum, g) => sum + g.attributes.position.count, 0),
        restyle,
        dispose: release,
    };
}
