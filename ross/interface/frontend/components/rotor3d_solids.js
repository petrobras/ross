// The building blocks every solid of the 3D view is made of: solids of
// revolution, rings, faceted pieces, bolt heads, and blades lofted from their
// sections. Shared by the element builders (rotor3d_parts.js) and the shapes
// of the geometry bank (rotor3d_shapes.js), which cannot import each other.
//
// three.js is handed in, not imported (see rotor3d_parts.js).
import { SYMBOL } from '../core/rotor3d_layout.js';

export const SEGMENTS = { shaft: 40, part: 56, small: 12 };

// Bolts and the like: a material, not interface (see rotor3d_parts.js).
export const STEEL = 0xb4bcc4;

// Where the outline is drawn: between two faces more than this apart.
export const EDGE_ANGLE = 28;

// A solid of revolution around the z axis, from an (r, z) profile. The profile
// goes around the section counter-clockwise (r across, z up), as every profile
// in this file does, so the normals point out of the solid.
export function revolve(THREE, profile, segments, edges) {
    const bands = profile.length - 1;
    const position = new Float32Array(bands * segments * 18);
    const normal = new Float32Array(bands * segments * 18);
    const cos = [];
    const sin = [];
    for (let j = 0; j <= segments; j++) {
        const a = (j / segments) * Math.PI * 2;
        cos.push(Math.cos(a));
        sin.push(Math.sin(a));
    }
    const normals = [];
    let k = 0;
    for (let i = 0; i < bands; i++) {
        const [ra, za] = profile[i];
        const [rb, zb] = profile[i + 1];
        const length = Math.hypot(rb - ra, zb - za) || 1;
        const nr = (zb - za) / length;
        const nz = -(rb - ra) / length;
        normals.push([nr, nz]);
        for (let j = 0; j < segments; j++) {
            // Counter-clockwise seen from where the normal points: along the
            // profile t and around the axis s, t x s points into the solid, so
            // each triangle goes the other way round.
            const quad = [[ra, za, j], [rb, zb, j + 1], [rb, zb, j], [ra, za, j], [ra, za, j + 1], [rb, zb, j + 1]];
            for (const [r, z, m] of quad) {
                position[k] = r * cos[m]; position[k + 1] = r * sin[m]; position[k + 2] = z;
                normal[k] = nr * cos[m]; normal[k + 1] = nr * sin[m]; normal[k + 2] = nz;
                k += 3;
            }
        }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(normal, 3));
    if (edges) {
        // A circle wherever two neighbouring segments of the profile meet at
        // an angle, and at the two ends of an open profile.
        const limit = Math.cos((EDGE_ANGLE * Math.PI) / 180);
        for (let i = 0; i < profile.length; i++) {
            const before = normals[i - 1];
            const after = normals[i];
            const corner = !before || !after || before[0] * after[0] + before[1] * after[1] < limit;
            if (corner && profile[i][0] > 1e-9) edges.push({ circle: profile[i], segments });
        }
    }
    return geometry;
}

// A closed ring (or disk, with rIn 0) between two radii, from z0 to z1, with
// its edges broken by a chamfer `c` on the outside.
export function ring(THREE, rIn, rOut, z0, z1, edges, segments = SEGMENTS.part, c = 0) {
    const ch = Math.min(c, (rOut - rIn) / 3, Math.abs(z1 - z0) / 3);
    const profile = [[rIn, z0], [rOut - ch, z0]];
    if (ch > 0) profile.push([rOut, z0 + ch], [rOut, z1 - ch]); else profile.push([rOut, z1]);
    profile.push([rOut - ch, z1]);
    if (ch <= 0) profile.pop();
    profile.push([rIn, z1], [rIn, z0]);
    return revolve(THREE, profile, segments, edges);
}

// Plain three.js geometry, made flat-shaded and outline-ready. Everything that
// is not a solid of revolution goes through here.
export function faceted(THREE, geometry, edges) {
    const flat = geometry.index ? geometry.toNonIndexed() : geometry;
    if (flat !== geometry) geometry.dispose();
    flat.deleteAttribute('uv');
    flat.computeVertexNormals();
    if (edges) edges.push({ geometry: flat });
    return flat;
}

// A hex bolt head, its axis along `axis` ('x', 'y' or 'z'), centred at (x, y, z).
export function boltHead(THREE, size, height, axis, x, y, z, edges) {
    const head = new THREE.CylinderGeometry(size, size, height, 6);
    if (axis === 'z') head.rotateX(Math.PI / 2);
    if (axis === 'x') head.rotateZ(Math.PI / 2);
    head.translate(x, y, z);
    return faceted(THREE, head, edges);
}

// Bolt heads on a circle of radius `radius` around the axis, on the face at z.
export function boltCircle(THREE, count, radius, size, height, z, edges, phase = 0) {
    const heads = [];
    for (let i = 0; i < count; i++) {
        const a = phase + (i / count) * Math.PI * 2;
        heads.push(boltHead(THREE, size, height, 'z', Math.cos(a) * radius, Math.sin(a) * radius, z, edges));
    }
    return heads;
}

// What every builder returns: a list of { geometry, color, finish } in the
// part's own frame, with `color` null for "the element's own colour".

export const own = (geometry, finish = 'metal') => ({ geometry, color: null, finish });
export const steel = geometry => ({ geometry, color: STEEL, finish: 'metal' });

// A closed solid skinned over `sections`: each a closed loop of [x, y, z]
// points, all with the same count, one after the other along the solid (a
// blade from root to tip, say). The two end sections are capped.
//
// The skin is shaded smooth -- a blade is a curved, twisted surface, and the
// flat shading of `faceted` broke it into the zig-zag of its triangles -- and
// the caps flat. Winding is settled by the signed volume, so the loops may go
// either way round.
export function loft(THREE, sections, edges) {
    const count = sections[0].length;
    const points = sections.flat();
    const sides = [];
    for (let k = 0; k + 1 < sections.length; k++) {
        for (let i = 0; i < count; i++) {
            const j = (i + 1) % count;
            const a = k * count;
            const b = (k + 1) * count;
            sides.push([a + i, b + i, b + j], [a + i, b + j, a + j]);
        }
    }
    // Each cap runs round its section against the sides that meet it, so the
    // whole surface faces one way.
    const caps = [];
    for (const [end, flip] of [[sections[0], false], [sections[sections.length - 1], true]]) {
        const centre = [0, 1, 2].map(c => end.reduce((sum, p) => sum + p[c], 0) / count);
        for (let i = 0; i < count; i++) {
            const j = (i + 1) % count;
            caps.push(flip ? [centre, end[j], end[i]] : [centre, end[i], end[j]]);
        }
    }
    // Six times the signed volume: negative when the triangles face inwards.
    const triple = (p, q, r) => p[0] * (q[1] * r[2] - q[2] * r[1]) - p[1] * (q[0] * r[2] - q[2] * r[0]) + p[2] * (q[0] * r[1] - q[1] * r[0]);
    let volume = 0;
    for (const [i, j, k] of sides) volume += triple(points[i], points[j], points[k]);
    for (const [p, q, r] of caps) volume += triple(p, q, r);
    const inwards = volume < 0;

    const skin = new THREE.BufferGeometry();
    skin.setAttribute('position', new THREE.BufferAttribute(new Float32Array(points.flat()), 3));
    skin.setIndex(sides.flatMap(([i, j, k]) => (inwards ? [i, k, j] : [i, j, k])));
    skin.computeVertexNormals();
    const smooth = skin.toNonIndexed();
    skin.dispose();

    const lids = new THREE.BufferGeometry();
    lids.setAttribute('position', new THREE.BufferAttribute(new Float32Array(
        caps.flatMap(([p, q, r]) => (inwards ? [p, r, q] : [p, q, r]).flat())), 3));
    lids.computeVertexNormals();

    const count2 = smooth.attributes.position.count + lids.attributes.position.count;
    const position = new Float32Array(count2 * 3);
    const normal = new Float32Array(count2 * 3);
    position.set(smooth.attributes.position.array, 0);
    normal.set(smooth.attributes.normal.array, 0);
    position.set(lids.attributes.position.array, smooth.attributes.position.count * 3);
    normal.set(lids.attributes.normal.array, smooth.attributes.position.count * 3);
    smooth.dispose();
    lids.dispose();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(normal, 3));
    if (edges) edges.push({ geometry });
    return geometry;
}

// The shaft ends a coupling clamps, whatever it is drawn as. Where no shaft
// element of the project reaches in (the layout says which side), the shaft
// is drawn from the node into the hub: a coupling clamps a shaft end, it does
// not float in its span.
export function couplingShaftEnds(THREE, part, edges) {
    const [z0, z1] = part.body || [Math.min(part.z0, part.z1), Math.max(part.z0, part.z1)];
    const [bore0, bore1] = part.bores || [part.bore, part.bore];
    const hubLength = part.hub;
    const stubs = part.stubs || [false, false];
    const stubColor = part.stubColor === undefined ? null : part.stubColor;
    const node0 = Math.min(part.z0, part.z1);
    const node1 = Math.max(part.z0, part.z1);
    const pieces = [];
    if (stubs[0]) {
        pieces.push({ geometry: ring(THREE, 0, bore0, node0, z0 + 0.9 * hubLength, edges, SEGMENTS.shaft, 0.06 * bore0),
            color: stubColor, finish: 'metal', named: true });
    }
    if (stubs[1]) {
        pieces.push({ geometry: ring(THREE, 0, bore1, z1 - 0.9 * hubLength, node1, edges, SEGMENTS.shaft, 0.06 * bore1),
            color: stubColor, finish: 'metal', named: true });
    }
    return pieces;
}

// The housing a bearing stands in, seen from the front: a round cap over the
// shaft whose sides run on straight down from it, tangent, to a base with a
// foot to each side, where SYMBOL.bearing puts them -- extruded along the
// shaft, its edges bevelled, with a bore of radius `bore`, and a bolt in each
// foot. The pillow block's (rotor3d_parts.js), and every bearing's of the
// geometry bank (rotor3d_shapes.js): Leonardo asked for theirs to meet the
// base as the pillow block's does, and one outline for all of them is what
// keeps them standing alike on the bench's pedestals. `footTop` is where the
// feet end, for what else sits on them.
export function bearingHousing(THREE, part, bore, edges) {
    const r = part.shaftRadius;
    const z = (part.z0 + part.z1) / 2;
    const w = Math.abs(part.z1 - part.z0);
    const { feet, base } = SYMBOL.bearing;
    const depth = w * 0.86;
    // The bevel grows the outline by its own size: the outline is drawn that
    // much inside, so the block ends where SYMBOL.bearing says it does.
    const bevel = Math.min(0.06 * r, depth * 0.08);
    const cap = part.radius - bevel;
    const edge = feet * r - bevel;
    const bottom = -base * r + bevel;
    const footTop = (-base + 0.4) * r;
    const shape = new THREE.Shape();
    shape.moveTo(-edge, bottom);
    shape.lineTo(edge, bottom);
    shape.lineTo(edge, footTop);
    shape.lineTo(cap * 1.02, footTop);
    shape.lineTo(cap, 0);
    shape.absarc(0, 0, cap, 0, Math.PI, false);
    shape.lineTo(-cap * 1.02, footTop);
    shape.lineTo(-edge, footTop);
    shape.closePath();
    const hole = new THREE.Path();
    hole.absarc(0, 0, bore + bevel, 0, Math.PI * 2, true);
    shape.holes.push(hole);
    const body = new THREE.ExtrudeGeometry(shape, {
        depth: depth - 2 * bevel, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel,
        bevelSegments: 1, curveSegments: 28,
    });
    body.translate(0, 0, z - depth / 2 + bevel);
    const pieces = [own(faceted(THREE, body, edges), 'paint')];
    for (const x of [-(feet - 0.45) * r, (feet - 0.45) * r]) {
        pieces.push(steel(boltHead(THREE, 0.16 * r, 0.25 * r, 'y', x, footTop + 0.12 * r, z, edges)));
    }
    return { pieces, footTop, depth };
}
