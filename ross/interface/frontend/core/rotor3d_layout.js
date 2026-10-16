// Where each part of the 3D view goes, worked out from the scene the server
// sends (`domain/rotor_scene.py`), with no three.js in it.
//
// The scene says what ROSS built, in SI: where every node is along the axis,
// the diameters of every shaft element, the real or equivalent size of every
// disk. What it does not say is how to draw what has no size in ROSS -- a
// bearing, a seal, a point mass -- nor where the second shaft line of a
// MultiRotor sits. Both are decided here, as numbers, so they can be checked
// in node without a GPU, and so the drawing (`components/rotor3d_parts.js`)
// and the pointer (`pickPart`) read the same answer.
//
// Coordinates are ROSS's: z along the shaft, x and y across it, y up. A part is
// a solid of revolution around its own axis, which is the shaft line's axis
// moved by `offset` -- the MultiRotor's driven line, or a support hanging under
// the node that links to it.

// How big the parts ROSS gives no size to are drawn, in shaft radii at their
// node. The proportions are the prototype's, which were chosen to read as a
// bearing housing, a seal gland and a clamped collar at any rotor size.
export const SYMBOL = {
    // A pillow block: the cap is `radius`, the feet reach `feet` to each side
    // and the base sits `base` below the axis; the grease nipple tops it.
    bearing: { radius: 1.6, width: 2.6, feet: 2.6, base: 1.9, top: 2.0 },
    seal: { radius: 1.25, width: 1.0 },
    pointmass: { radius: 1.35, width: 1.0 },
    // A disk whose mass or inertia is not positive has no equivalent size:
    // a thin ring says "something is here" without inventing one.
    disk: { radius: 1.3, width: 0.4 },
    // How far below its parent a part on a link node hangs.
    link: 3.4,
    // A proximity probe: its tip `gap` off the shaft, `length` long from tip
    // to cable, and `across` either side of its node along the axis.
    probe: { gap: 0.15, length: 2.3, across: 0.3 },
};

// A probe has no colour in ROSS, which does not draw probes; this one is the
// cable's, and the legend's.
export const PROBE_COLOR = 'darkorange';

// How far a gear's hub reaches along the axis, each way, in face widths; and
// how much wider than its gland a seal's flange is.
export const GEAR_HUB = 0.62;
export const SEAL_FLANGE = 1.2;

// How far a coupling's hub reaches into its span from each node: under a third
// of the span, and not much longer than the coupling is wide.
export function couplingHub(span, radius) {
    return Math.min(0.3 * span, 1.1 * radius);
}

// How long a coupling is drawn: a real one is about two and a half diameters'
// worth of hubs and spacer, whatever the length of the span ROSS gives it. On a
// long span it sits in the middle, and the shaft shows on each side, running
// into a hub (Leonardo: the coupling on node 0 used to swallow the whole first
// element up to the bearing). It never takes more than 60 % of the span.
export function couplingBody(z0, z1, radius) {
    const length = Math.min(0.6 * (z1 - z0), 2.6 * radius);
    const middle = (z0 + z1) / 2;
    return [middle - length / 2, middle + length / 2];
}

// The face width a gear is drawn with when ROSS has none, in modules: gears
// are commonly 8 to 12 modules wide.
export const FACE_WIDTH_IN_MODULES = 10;

// How a gear is drawn, from what ROSS keeps of it: the pitch radius, the tip
// (addendum) radius, the number of teeth, the bore and the width.
//
// The module is the pitch diameter over the teeth, the dedendum 1.25 modules.
// One case needs a decision: a pitch circle **inside the gear's own shaft**.
// ROSS takes it -- the pinion of its MultiRotor example has a 77 mm pitch
// diameter on a 150 mm shaft -- because the mesh only needs the radius. Drawn
// as given, the teeth would be buried in the shaft and the gear would look like
// a ring stuck to it. It is drawn as what such a pinion is made as, teeth cut
// on the shaft: the root on the shaft's surface, the pitch circle 1.25 modules
// out. `rossPitch` keeps ROSS's number for the tooltip, and `onShaft` says so.
export function gearDrawing(g, shaftRadius) {
    const teeth = g.teeth > 0 ? g.teeth : 24;
    const rossPitch = g.pitch_radius || g.outer_radius || 2 * shaftRadius;
    const module = (2 * rossPitch) / teeth;
    const addendum = g.outer_radius > rossPitch ? g.outer_radius - rossPitch : module;
    const hub = Math.max(g.bore_radius || 0, shaftRadius);
    const onShaft = rossPitch <= hub;
    const pitch = onShaft ? hub + 1.25 * addendum : rossPitch;
    const root = Math.max(pitch - 1.25 * addendum, hub);
    const width = g.width || FACE_WIDTH_IN_MODULES * module;
    return {
        teeth, pitch, rossPitch, root, tip: pitch + addendum, bore: g.bore_radius || shaftRadius,
        width, widthAssumed: !g.width, onShaft, module,
    };
}

// The helix angle a helical gear is drawn with: ROSS's (the scene gives it in
// radians), or HELIX_DRAWN for a gear ROSS has as spur and the form asks to be
// drawn helical -- `assumed` says so, for the tooltip. Past HELIX_MAX the teeth
// would read as a screw. The driven line of a MultiRotor turns the other way:
// two external gears in mesh have opposite hands, and so their teeth meet.
export const HELIX_DRAWN = (20 * Math.PI) / 180;
export const HELIX_MAX = (45 * Math.PI) / 180;

function helixDrawn(g, half) {
    const own = Number(g.helix_angle) || 0;
    const assumed = Math.abs(own) <= 1e-6;
    const size = Math.min(assumed ? HELIX_DRAWN : Math.abs(own), HELIX_MAX);
    const hand = (own < 0 ? -1 : 1) * (half === 'driven' ? -1 : 1);
    return { angle: hand * size, assumed };
}

// When a node has no shaft element at all (a link node, or an empty rotor),
// symbols need some radius: the median of the shaft radii keeps them in scale
// with the rest of the rotor.
function typicalRadius(shafts) {
    const radii = shafts.map(s => Math.max(s.odl || 0, s.odr || 0) / 2).filter(r => r > 0).sort((a, b) => a - b);
    return radii.length ? radii[Math.floor(radii.length / 2)] : 0.025;
}

// The outer radius of the shaft line at `z`: the largest of the elements that
// reach it, tapered ones read at that point. Layered elements (a sleeve over a
// shaft, as in ROSS's compressor example) count with their outer diameter.
export function shaftRadiusAt(shafts, z, fallback) {
    let best = 0;
    for (const s of shafts) {
        if (s.z0 == null || s.z1 == null) continue;
        const lo = Math.min(s.z0, s.z1) - 1e-9;
        const hi = Math.max(s.z0, s.z1) + 1e-9;
        if (z < lo || z > hi) continue;
        const span = s.z1 - s.z0;
        const along = span ? (z - s.z0) / span : 0;
        const outer = ((s.odl || 0) + ((s.odr ?? s.odl ?? 0) - (s.odl || 0)) * along) / 2;
        best = Math.max(best, outer);
    }
    return best || fallback;
}

// The parts of one shaft line, in its own coordinates, moved by `offset`.
function partsOfLine(line, half, offset, shapeFor) {
    const shafts = line.shafts || [];
    const typical = typicalRadius(shafts);
    const radiusAt = z => shaftRadiusAt(shafts, z, typical);
    const parts = [];
    const key = (category, index) => (half ? `${half}:${category}:${index}` : `${category}:${index}`);
    const base = { half, offset };
    // The key of the geometry-bank shape an element is drawn with, '' for its
    // category's own drawing.
    const shapeKey = (category, index, entry) => ((shapeFor && shapeFor(half, category, index, entry)) || {}).key || '';

    // A coupling occupies its span like a shaft element (ROSS lumps its two
    // halves at the two nodes, m_l and m_r). It is drawn at a coupling's own
    // length in the middle of the span (`couplingBody`): two flanged hubs and a
    // slender spacer, and the shaft line separated there -- from each node it
    // runs into a hub and stops, whether or not a shaft element of the project
    // also spans the coupling's nodes. Its bore is the shaft it clamps on each
    // side.
    const withEnds = shafts.filter(s => s.z0 != null && s.z1 != null);
    const reaching = z => withEnds.filter(s => Math.abs(s.z0 - z) < 1e-9 || Math.abs(s.z1 - z) < 1e-9);
    const spanning = (a, b) => withEnds.filter(s => Math.min(s.z0, s.z1) < b - 1e-9 && Math.max(s.z0, s.z1) > a + 1e-9);
    const couplings = [];
    for (const c of line.couplings || []) {
        if (c.z0 == null || c.z1 == null) continue;
        const z0 = Math.min(c.z0, c.z1);
        const z1 = Math.max(c.z0, c.z1);
        const across = spanning(z0, z1);
        const sideRadius = z => {
            const touching = reaching(z).concat(across);
            return touching.length ? shaftRadiusAt(touching, z, typical) : typical;
        };
        const bores = [sideRadius(z0), sideRadius(z1)];
        const r = Math.max(...bores);
        const radius = c.outer_diameter ? Math.max(c.outer_diameter / 2, 1.3 * r) : 2.2 * r;
        const body = couplingBody(z0, z1, radius);
        const hub = couplingHub(body[1] - body[0], radius);
        // The hubs need a shaft end in them: one the project draws there (an
        // element spanning the coupling, which the cut below shortens), or a
        // stub of the shaft that ends on that node.
        const covered = z => across.some(s => Math.min(s.z0, s.z1) <= z + 1e-9 && Math.max(s.z0, s.z1) >= z - 1e-9);
        const shaftColor = (reaching(z0).concat(reaching(z1), across)[0] || {}).color;
        couplings.push({
            ...base, key: key('couplings', c.index), category: 'couplings', index: c.index, kind: 'coupling',
            entry: c, color: c.color, z0: c.z0, z1: c.z1, radius, bore: r, bores, hub,
            body,
            gap: [body[0] + hub, body[1] - hub],
            stubs: [!covered((z0 + body[0]) / 2), !covered((body[1] + z1) / 2)],
            stubColor: shaftColor,
            shape: shapeKey('couplings', c.index, c),
            // A shaft element over the same span is a second, parallel stiffness
            // in ROSS; the tooltip says so.
            overlapsShaft: across.length > 0,
        });
    }

    // An end of a shaft element is chamfered where it stands proud: at a free
    // end, or at a step down to a thinner neighbour. Between two elements of the
    // same diameter the surface runs on, as on a real shaft.
    const exposed = (s, z, radius) => !shafts.some(o => o !== s && o.z0 != null && o.z1 != null
        && (Math.abs(o.z0 - z) < 1e-9 || Math.abs(o.z1 - z) < 1e-9)
        && shaftRadiusAt([o], z, 0) >= radius * 0.999);
    for (const s of shafts) {
        if (s.z0 == null || s.z1 == null) continue;
        const profile = { odl: s.odl, odr: s.odr ?? s.odl, idl: s.idl || 0, idr: s.idr ?? s.idl ?? 0 };
        parts.push({
            ...base, key: key('shafts', s.index), category: 'shafts', index: s.index, kind: 'shaft',
            entry: s, color: s.color, z0: s.z0, z1: s.z1,
            radius: Math.max(s.odl || 0, s.odr || 0) / 2,
            profile,
            chamfer: [exposed(s, s.z0, profile.odl / 2), exposed(s, s.z1, profile.odr / 2)],
            cuts: couplings.map(c => c.gap).filter(([g0, g1]) => g1 > g0
                && Math.min(s.z0, s.z1) < g1 && Math.max(s.z0, s.z1) > g0),
        });
    }
    parts.push(...couplings);


    for (const d of line.disks || []) {
        if (d.z == null) continue;
        const r = radiusAt(d.z);
        const shape = d.shape;
        const outer = shape ? shape.outer_radius : SYMBOL.disk.radius * r;
        let width = shape && shape.width ? shape.width : SYMBOL.disk.width * r;
        // A shape from the bank, drawn in the disk's envelope -- widened to the
        // depth the shape needs to read as itself. A disk with no size in ROSS
        // stays the plain ring that says so.
        const drawnAs = shape && shapeFor ? shapeFor(half, 'disks', d.index, d) : null;
        if (drawnAs) width = Math.max(width, drawnAs.minWidth * outer);
        parts.push({
            ...base, key: key('disks', d.index), category: 'disks', index: d.index, kind: 'disk',
            entry: d, color: d.color, z0: d.z - width / 2, z1: d.z + width / 2,
            radius: outer, bore: shape ? shape.inner_radius : r, symbolic: !shape,
            shape: drawnAs ? drawnAs.key : '', shaftRadius: r,
        });
    }

    for (const g of line.gears || []) {
        if (g.z == null) continue;
        const drawn = gearDrawing(g, radiusAt(g.z));
        const drawnAs = shapeFor ? shapeFor(half, 'gears', g.index, g) : null;
        const helix = drawnAs && drawnAs.key === 'helical' ? helixDrawn(g, half) : null;
        // The hub stands proud of the teeth on both faces (GEAR_HUB of the width).
        parts.push({
            ...base, key: key('gears', g.index), category: 'gears', index: g.index, kind: 'gear',
            entry: g, color: g.color, z0: g.z - GEAR_HUB * drawn.width, z1: g.z + GEAR_HUB * drawn.width,
            radius: drawn.tip, bore: drawn.bore, faceWidth: drawn.width, shaftRadius: radiusAt(g.z),
            gear: drawn,
            shape: drawnAs ? drawnAs.key : '',
            helix: helix ? helix.angle : 0,
            helixAssumed: !!helix && helix.assumed,
        });
    }

    // What sits on a node with no size of its own. A part on a link node hangs
    // under the part that links to it, one level per link, as ROSS's 2D
    // figure draws a support under its bearing.
    const depth = linkDepths(line);
    const symbolic = [
        ['bearings', 'bearing'],
        ['seals', 'seal'],
        ['pointmasses', 'pointmass'],
    ];
    for (const [category, kind] of symbolic) {
        for (const e of line[category] || []) {
            if (e.z == null) continue;
            const r = radiusAt(e.z);
            const levels = e.link ? (depth.get(e.n) || 1) : 0;
            const size = SYMBOL[kind];
            parts.push({
                ...base, key: key(category, e.index), category, index: e.index, kind,
                entry: e, color: e.color, z0: e.z - (size.width * r) / 2, z1: e.z + (size.width * r) / 2,
                radius: size.radius * r, bore: r, shaftRadius: r,
                offset: { x: offset.x, y: offset.y - levels * SYMBOL.link * r, z: offset.z },
                hanging: levels,
                shape: shapeKey(category, e.index, e),
            });
        }
    }

    // The probes, which ROSS's rotor does not have (domain/rotor_builder.py
    // `build_probes`) and the analyses read the response at. A radial probe
    // points at the shaft from its angle -- from x toward y, ROSS's own
    // convention -- its tip a gap off the surface. An axial one lies along
    // the shaft and faces its node's plane, from the outer side of the rotor's
    // middle: on the axis at either end of the shaft line, where it reads the
    // end face, and over the shaft anywhere else.
    //
    // A probe reads at its node, and on its node there is often a disk or a
    // bearing that would swallow it. So it is drawn just beside what sits
    // there, on the outer side, as a probe is mounted beside a bearing on the
    // machine -- and the tooltip says so (`beside`).
    const zs = (line.nodes || []).map(node => node.z);
    const middle = zs.length ? (Math.min(...zs) + Math.max(...zs)) / 2 : 0;
    const ends = zs.length ? [Math.min(...zs), Math.max(...zs)] : [];
    const onNode = parts.filter(p => !p.hanging && ['disk', 'gear', 'bearing', 'seal', 'pointmass'].includes(p.kind));
    const occupied = (z, outward) => {
        const there = onNode.filter(p => Math.min(p.z0, p.z1) <= z + 1e-12 && Math.max(p.z0, p.z1) >= z - 1e-12);
        if (!there.length) return null;
        return Math.max(...there.map(p => (outward > 0 ? Math.max(p.z0, p.z1) - z : z - Math.min(p.z0, p.z1))));
    };
    for (const p of line.probes || []) {
        if (p.z == null) continue;
        const r = radiusAt(p.z);
        const { gap, length, across } = SYMBOL.probe;
        const outward = p.z < middle ? -1 : 1;
        const width = occupied(p.z, outward);
        const common = {
            ...base, key: key('probes', p.index), category: 'probes', index: p.index, kind: 'probe',
            entry: p, color: PROBE_COLOR, shaftRadius: r, outward, beside: width !== null,
        };
        if (p.direction === 'axial') {
            const atEnd = ends.some(z => Math.abs(z - p.z) < 1e-9);
            const lift = atEnd ? 0 : (1 + gap + across) * r;
            const face = p.z + outward * (width || 0);
            const from = face + outward * gap * r;
            const to = face + outward * (gap + length) * r;
            parts.push({
                ...common, axial: true, lift,
                z0: Math.min(from, to), z1: Math.max(from, to),
                radius: lift + across * r,
            });
        } else {
            const z = width === null ? p.z : p.z + outward * (width + 1.3 * across * r);
            parts.push({
                ...common, axial: false, angle: p.angle || 0,
                z0: z - across * r, z1: z + across * r,
                radius: (1 + gap + length) * r,
            });
        }
    }
    return parts;
}

// How many links down each link node is: 1 under a structural node, 2 under a
// link node, and so on. A cycle stops at the number of links there are.
function linkDepths(line) {
    const parent = new Map();
    for (const category of ['bearings', 'seals']) {
        for (const e of line[category] || []) {
            if (e.n_link != null && !parent.has(e.n_link)) parent.set(e.n_link, e.n);
        }
    }
    const structural = new Set((line.nodes || []).map(n => n.n));
    const depth = new Map();
    for (const link of parent.keys()) {
        let levels = 0;
        let node = link;
        while (!structural.has(node) && parent.has(node) && levels <= parent.size) {
            node = parent.get(node);
            levels += 1;
        }
        depth.set(link, Math.max(1, levels));
    }
    return depth;
}

// Where the driven line of a MultiRotor goes. Along the axis, ROSS says it:
// `driven_offset` is the `dz_pos` that lines the two coupled gears up. Across
// it, the two axes are the sum of the (drawn) pitch radii apart, on the line of gear
// centres at `orientation_angle` from x -- ROSS's own definition of that angle.
// (`position`, above or below, is how ROSS's 2D figure stacks the two lines on
// paper; in space the angle is what places them.)
export function drivenPlacement(scene) {
    const [drivingNode, drivenNode] = scene.coupled_nodes || [];
    // The drawn pitch radius (gearDrawing), which is ROSS's but for a pinion
    // whose pitch circle is inside its own shaft: the two gears are drawn in
    // mesh, and do not overlap.
    const radiusOf = (line, node) => {
        const gear = (line.gears || []).find(g => g.n === node);
        if (!gear) return 0;
        const shafts = line.shafts || [];
        return gearDrawing(gear, shaftRadiusAt(shafts, gear.z, typicalRadius(shafts))).pitch;
    };
    let distance = radiusOf(scene.driving, drivingNode) + radiusOf(scene.driven, drivenNode);
    if (!(distance > 0)) {
        distance = 4 * Math.max(typicalRadius(scene.driving.shafts || []), typicalRadius(scene.driven.shafts || []));
    }
    const angle = scene.orientation_angle || 0;
    return { x: distance * Math.cos(angle), y: distance * Math.sin(angle), z: scene.driven_offset || 0 };
}

// Every part of the scene, in world coordinates, plus the node rings and the
// box that holds them. This is the one reading of the scene the 3D view makes.
// `shapeFor(half, category, index, entry)`, when given, answers the shape of
// the geometry bank (core/shapes3d.js) an element is drawn with -- `{ key,
// minWidth }`, or null for its category's own drawing. `entry` is the scene's
// entry for the element, which says its ROSS class and helix angle; the choice
// made in the form is the project's, which the scene knows nothing of. Handed
// in, so this module stays a function of what it is given.
export function layoutScene(scene, shapeFor) {
    const lines = [];
    if (scene && scene.kind === 'multirotor') {
        lines.push(['driving', scene.driving, { x: 0, y: 0, z: 0 }]);
        lines.push(['driven', scene.driven, drivenPlacement(scene)]);
    } else if (scene) {
        lines.push([null, scene, { x: 0, y: 0, z: 0 }]);
    }

    const parts = [];
    const rings = [];
    for (const [half, line, offset] of lines) {
        const ofThisLine = partsOfLine(line, half, offset, shapeFor);
        parts.push(...ofThisLine);
        const shafts = line.shafts || [];
        const typical = typicalRadius(shafts);
        for (const node of line.nodes || []) {
            rings.push({ half, n: node.n, z: node.z + offset.z, x: offset.x, y: offset.y,
                radius: shaftRadiusAt(shafts, node.z, typical) });
        }
    }
    return { parts, rings, bounds: boundsOf(parts) };
}

// How far a part reaches above its own axis: a bearing's grease nipple, a
// point mass's lug, a seal's flange, or else its radius.
function heightAbove(p) {
    if (p.kind === 'pointmass') return p.radius + 0.55 * p.shaftRadius;
    if (p.kind === 'bearing') return SYMBOL.bearing.top * p.shaftRadius;
    if (p.kind === 'seal') return SEAL_FLANGE * p.radius;
    return p.radius;
}

function boundsOf(parts) {
    if (!parts.length) return { min: [-0.1, -0.1, 0], max: [0.1, 0.1, 0.1] };
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const p of parts) {
        const o = p.offset;
        const r = p.radius;
        // What reaches past the cylinder of a part (components/rotor3d_parts.js):
        // a bearing's feet, base and grease nipple (SYMBOL.bearing), a seal's
        // flange, and a point mass's clamp lug on top.
        const flange = p.kind === 'seal' ? SEAL_FLANGE * r : r;
        const bearing = SYMBOL.bearing;
        const across = p.kind === 'bearing' ? bearing.feet * p.shaftRadius : flange;
        const below = p.kind === 'bearing' ? bearing.base * p.shaftRadius : flange;
        const above = heightAbove(p);
        min[0] = Math.min(min[0], o.x - across); max[0] = Math.max(max[0], o.x + across);
        min[1] = Math.min(min[1], o.y - below); max[1] = Math.max(max[1], o.y + above);
        min[2] = Math.min(min[2], o.z + Math.min(p.z0, p.z1)); max[2] = Math.max(max[2], o.z + Math.max(p.z0, p.z1));
    }
    return { min, max };
}

// The motor that drives the rotor, shown with the bench and, like it, only a
// picture (Leonardo: "totalmente estético, assim como a bancada"). It stands
// at the start of the rotor -- node 0's end, the left of the view -- on the
// axis of the line it drives (a MultiRotor's driving line), before everything
// drawn, and a shaft of its own runs through a coupling to the rotor's first
// shaft element.
//
// Its size follows the shaft it drives: `radius` in radii of that shaft end,
// `length` of the frame in motor radii, `gap` between its face and the rotor
// in shaft radii (room for the coupling), `drop` from the axis to the bottom
// of its feet in motor radii.
export const MOTOR = { radius: 3.2, length: 2.6, gap: 3.2, drop: 1.15 };

export function motorLayout(layout) {
    const shafts = layout.parts.filter(p => p.kind === 'shaft' && p.half !== 'driven' && p.profile);
    if (!shafts.length) return null;
    const startOf = p => p.offset.z + Math.min(p.z0, p.z1);
    const first = shafts.reduce((best, p) => (startOf(p) < startOf(best) ? p : best));
    const start = startOf(first);
    const end = first.z0 <= first.z1 ? first.profile.odl : first.profile.odr;
    const r = Math.max(end / 2, 1e-4);
    const radius = MOTOR.radius * r;
    const face = Math.min(layout.bounds.min[2], start) - MOTOR.gap * r;
    const length = MOTOR.length * radius;
    return {
        x: first.offset.x, y: first.offset.y, shaftRadius: r, radius,
        // The frame from its back (the fan cover) to its face; the shaft from
        // the face to the rotor; the coupling in the gap, clear of both.
        z0: face - length, z1: face, shaftTo: start,
        coupling: [face + 0.35 * r, face + (MOTOR.gap - 0.9) * r],
        feet: first.offset.y - MOTOR.drop * radius,
        top: first.offset.y + 1.3 * radius,
        across: radius,
    };
}

// The node numbers and the dimensions, shown with a button as ROSS's 2D figure
// numbers its nodes (a suggestion Leonardo brought back from people using
// it). Per shaft line -- a MultiRotor has two:
//
//   * a number over each node, at the top of the shaft there;
//   * above the rotor, two tiers of dimension lines: the nearer one from end
//     to bearing to bearing to end -- the overhangs and the spans, the lengths
//     that move the critical speeds most -- and over it the total length;
//   * extension lines down from the outer tier to the shaft at both ends, and
//     from the inner tier to each bearing.
//
// Above and not below, where the bench is. The tiers clear everything drawn on
// that line by `gap`, a fraction of its largest radius or of its length.
// Bearings that hang under a link node are not on the shaft line and do not
// cut it into spans.
export function annotationLayout(layout) {
    const halves = Array.from(new Set(layout.rings.map(ring => ring.half)));
    const lines = [];
    const nodes = [];
    let top = -Infinity;
    for (const half of halves) {
        const rings = layout.rings.filter(ring => ring.half === half).sort((a, b) => a.z - b.z);
        const parts = layout.parts.filter(part => part.half === half);
        if (!rings.length) continue;
        rings.forEach(ring => nodes.push({ half, n: ring.n, position: [ring.x, ring.y + ring.radius, ring.z] }));
        const first = rings[0];
        const last = rings[rings.length - 1];
        const length = last.z - first.z;
        if (!(length > 0)) continue;
        const { x, y } = first;
        const highest = Math.max(...parts.map(p => p.offset.y + heightAbove(p)), y + Math.max(...rings.map(r => r.radius)));
        const largest = Math.max(...parts.map(p => p.radius), ...rings.map(r => r.radius));
        const gap = Math.max(0.6 * largest, 0.04 * length);
        const supports = Array.from(new Set(parts
            .filter(p => p.kind === 'bearing' && !p.hanging)
            .map(p => p.offset.z + (p.z0 + p.z1) / 2)))
            .filter(z => z > first.z + 1e-9 * length && z < last.z - 1e-9 * length)
            .sort((a, b) => a - b);
        const bearingsAt = parts.filter(p => p.kind === 'bearing' && !p.hanging)
            .map(p => ({ z: p.offset.z + (p.z0 + p.z1) / 2, y: p.offset.y + heightAbove(p) }));
        const inner = highest + gap;
        const outer = highest + 2 * gap;
        const stops = [first.z, ...supports, last.z];
        // With no bearing inside the ends, the one span would be the total.
        const spans = stops.length > 2
            ? stops.slice(1).map((z, k) => ({ z0: stops[k], z1: z, y: inner, length: z - stops[k] }))
            : [];
        const extensions = [
            { z: first.z, from: first.y + first.radius, to: outer },
            { z: last.z, from: last.y + last.radius, to: outer },
            ...bearingsAt.map(b => ({ z: b.z, from: b.y, to: inner })),
        ];
        lines.push({
            half, x, gap, length,
            total: { z0: first.z, z1: last.z, y: outer, length },
            spans, extensions,
        });
        top = Math.max(top, outer + 0.5 * gap);
    }
    return { lines, nodes, top };
}

// The test bench the rotor can be shown on, as in Leonardo's prototype: a
// slotted bed plate on legs, and a pedestal under every bearing reaching down
// to it -- and the motor (`motorLayout`), on a pedestal of its own when the
// plate is further down than its feet. It is not in the model -- ROSS knows no
// bench -- and it is drawn only when asked for. Every size is in proportion to
// the rotor, so a 1 m rotor and an 11 m one both stand on a bench that fits
// them.
//
// The plate's top sits below the lowest point of the rotor and of the motor's
// feet, so a large disk clears it; the pedestals make up the difference under
// each bearing.
export function benchLayout(layout) {
    const motor = motorLayout(layout);
    const min = layout.bounds.min.slice();
    const max = layout.bounds.max.slice();
    const length = max[2] - min[2];
    const height = max[1] - min[1];
    if (motor) {
        min[0] = Math.min(min[0], motor.x - motor.across);
        max[0] = Math.max(max[0], motor.x + motor.across);
        max[1] = Math.max(max[1], motor.top);
        min[2] = Math.min(min[2], motor.z0);
    }
    const width = max[0] - min[0];
    const margin = Math.max(0.06 * length, 0.25 * width);
    const thickness = Math.min(Math.max(0.1 * width, 0.015 * length), 0.03 * length);
    const top = Math.min(min[1] - Math.max(0.08 * height, 0.01 * length), motor ? motor.feet : Infinity);
    const plate = {
        x0: min[0] - margin, x1: max[0] + margin, z0: min[2] - margin, z1: max[2] + margin,
        top, bottom: top - thickness,
    };
    // Short legs, close under the plate (Leonardo asked for them shorter), each
    // standing on a vibration isolator: a rubber mount between two plates.
    const legHeight = Math.max(0.3 * (plate.x1 - plate.x0), 0.035 * length);
    const legSize = 0.08 * (plate.x1 - plate.x0);
    const mount = Math.max(0.9 * legSize, 0.12 * legHeight);
    const floor = plate.bottom - legHeight - mount;
    const pedestals = layout.parts.filter(p => p.kind === 'bearing').map(p => {
        const r = p.shaftRadius;
        return {
            key: p.key, x: p.offset.x, z: p.offset.z + (p.z0 + p.z1) / 2,
            y0: top, y1: p.offset.y - SYMBOL.bearing.base * r,
            width: 1.5 * SYMBOL.bearing.feet * r, depth: 0.8 * Math.abs(p.z1 - p.z0),
        };
    });
    if (motor && motor.feet > top + 1e-9) {
        pedestals.push({
            key: 'motor', x: motor.x, z: (motor.z0 + motor.z1) / 2, y0: top, y1: motor.feet,
            width: 2.4 * motor.radius, depth: 0.95 * (motor.z1 - motor.z0),
        });
    }
    // Legs at the corners, and more along a long bench, about three plate
    // widths apart.
    const along = Math.max(2, Math.ceil((plate.z1 - plate.z0) / (3 * (plate.x1 - plate.x0))) + 1);
    const legs = [];
    for (let i = 0; i < along; i++) {
        const z = plate.z0 + legSize + ((plate.z1 - plate.z0 - 2 * legSize) * i) / (along - 1);
        for (const x of [plate.x0 + legSize, plate.x1 - legSize]) legs.push({ x, z, size: legSize, mount });
    }
    return {
        plate, pedestals, legs, floor, legHeight, motor,
        bounds: {
            min: [Math.min(min[0], plate.x0), floor, Math.min(min[2], plate.z0)],
            max: [Math.max(max[0], plate.x1), max[1], Math.max(max[2], plate.z1)],
        },
    };
}

// Where to look from, for the whole rotor in view. The camera stands on the
// line through the centre of the box along `direction`, as close as it can
// while every corner of the box stays inside the field of view -- the box is
// projected, not wrapped in a sphere: a shaft is long and thin, and the
// sphere around it would leave it a sliver in the middle of the panel.
export function framing(bounds, fovDegrees, aspect, direction = [1, 0.45, 0.3]) {
    const center = [0, 1, 2].map(i => (bounds.min[i] + bounds.max[i]) / 2);
    const back = normalized(direction);
    const view = back.map(v => -v);
    let right = normalized(cross(view, [0, 1, 0]));
    if (!right.some(v => Math.abs(v) > 1e-9)) right = [1, 0, 0];
    const up = cross(right, view);
    const tanV = Math.tan((fovDegrees * Math.PI) / 360);
    const tanH = tanV * Math.max(aspect || 1, 1e-3);
    let distance = 0;
    for (const x of [bounds.min[0], bounds.max[0]]) {
        for (const y of [bounds.min[1], bounds.max[1]]) {
            for (const z of [bounds.min[2], bounds.max[2]]) {
                const p = [x - center[0], y - center[1], z - center[2]];
                const toward = dot(p, back);
                distance = Math.max(distance,
                    toward + Math.abs(dot(p, right)) / tanH,
                    toward + Math.abs(dot(p, up)) / tanV);
            }
        }
    }
    const size = [0, 1, 2].map(i => bounds.max[i] - bounds.min[i]);
    const radius = Math.max(1e-3, Math.hypot(...size) / 2);
    return { center, radius, distance: Math.max(distance * 1.08, radius * 1.05), direction: back };
}

// Where the camera looks from: the side, a little above and a little in front
// of node 0, so the shaft reads left to right as in ROSS's 2D figure -- node 0
// on the left, z growing to the right.
//
// It used to say that and do the opposite: from +x, which puts node 0 on the
// right, so switching between the 2D figure and the 3D view mirrored the rotor.
// Nothing checked the claim until the axis triad drew z pointing left. What
// decides it is the sign of x alone: seen from `d`, the screen's right is
// (d.z, 0, -d.x), so z grows to the right exactly when d.x < 0.
//
// For a MultiRotor the second line must not hide behind the first, so the
// camera keeps away from the line between the two axes (`orientation_angle`,
// from x): the direction around the rotor axis nearest to the one-rotor view
// that is at least `MIN_ACROSS` degrees off that line, among those that still
// read left to right.
const SIDE_VIEW = [-1, 0.45, -0.3];
const MIN_ACROSS = 50;

export function viewDirection(scene) {
    if (!(scene && scene.kind === 'multirotor')) return normalized(SIDE_VIEW);
    const around = Math.hypot(SIDE_VIEW[0], SIDE_VIEW[1]);
    const preferred = Math.atan2(SIDE_VIEW[1], SIDE_VIEW[0]) * 180 / Math.PI;
    const line = ((scene.orientation_angle || 0) * 180 / Math.PI) % 180;
    const offLine = theta => {
        const d = Math.abs(((theta - line) % 180 + 180) % 180);
        return Math.min(d, 180 - d);
    };
    // Reading left to right is x < 0: angles strictly between 90 and 270. From
    // above or level first (up to 180); from below only when the line between
    // the axes leaves nothing else -- when it runs at 45 degrees down from x.
    const nearest = (from, to) => {
        let found = null;
        for (let theta = from; theta <= to; theta += 1) {
            if (offLine(theta) < MIN_ACROSS) continue;
            if (found === null || Math.abs(theta - preferred) < Math.abs(found - preferred)) found = theta;
        }
        return found;
    };
    const best = nearest(95, 180) ?? nearest(181, 265) ?? preferred;
    const angle = best * Math.PI / 180;
    return normalized([around * Math.cos(angle), around * Math.sin(angle), SIDE_VIEW[2]]);
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalized(v) {
    const length = Math.hypot(...v);
    return length > 0 ? v.map(x => x / length) : [0, 0, 1];
}

// The node of a shaft line closest to `z` (world coordinates), for adding an
// element there from the 3D view. `half` is the MultiRotor line, null for a
// single rotor. Null when the line has no node.
export function nearestNode(rings, half, z) {
    let best = null;
    for (const ring of rings) {
        if (ring.half !== half) continue;
        if (best === null || Math.abs(ring.z - z) < Math.abs(best.z - z)) best = ring;
    }
    return best ? best.n : null;
}

// Where a ray meets the part it picked: the point the pointer is on.
export function hitPoint(origin, direction, distance) {
    return [0, 1, 2].map(i => origin[i] + direction[i] * distance);
}

// The part under a ray, or null. Every part is picked as the cylinder that
// holds it -- the prototype cast its ray at every tooth of every gear, which
// cost 80 ms per mouse move on a large rotor. A coaxial cylinder is a few
// multiplications, and a pointer does not need the teeth to know it is on a
// gear. Returns the nearest hit.
export function pickPart(parts, origin, direction) {
    let best = null;
    for (const part of parts) {
        const t = rayCylinder(part, origin, direction);
        if (t !== null && (best === null || t < best.distance)) best = { part, distance: t };
    }
    return best;
}

function rayCylinder(part, origin, direction) {
    const o = part.offset;
    const ox = origin[0] - o.x;
    const oy = origin[1] - o.y;
    const oz = origin[2] - o.z;
    const [dx, dy, dz] = direction;
    const r = part.radius;
    const zLo = Math.min(part.z0, part.z1);
    const zHi = Math.max(part.z0, part.z1);
    let nearest = null;
    const keep = t => { if (t >= 0 && (nearest === null || t < nearest)) nearest = t; };

    // The side: (ox + t dx)^2 + (oy + t dy)^2 = r^2.
    const a = dx * dx + dy * dy;
    if (a > 1e-12) {
        const b = 2 * (ox * dx + oy * dy);
        const c = ox * ox + oy * oy - r * r;
        const disc = b * b - 4 * a * c;
        if (disc >= 0) {
            const root = Math.sqrt(disc);
            for (const t of [(-b - root) / (2 * a), (-b + root) / (2 * a)]) {
                const z = oz + t * dz;
                if (z >= zLo && z <= zHi) keep(t);
            }
        }
    }
    // The two ends.
    if (Math.abs(dz) > 1e-12) {
        for (const zEnd of [zLo, zHi]) {
            const t = (zEnd - oz) / dz;
            const x = ox + t * dx;
            const y = oy + t * dy;
            if (x * x + y * y <= r * r) keep(t);
        }
    }
    return nearest;
}
