// The measuring tape of the 3D view: two ends dragged along a shaft line, and
// the distance between them -- as Google Maps measures a distance, but along
// the one direction a rotor has.
//
// WHY ALONG THE AXIS ONLY. A rotor is drawn around its axis, and every length
// that means something in it -- a span, where a disk sits, how far a seal is
// from its bearing -- is measured along that axis. An end that could go
// anywhere in space would measure diagonals nobody asked for, and would have
// to be placed in depth with a pointer that has none. So each end lives on a
// line parallel to the axis, just above what is drawn on it, and the pointer
// moves it to the point of that line nearest to the ray under the cursor
// (`alongAxis`), which works from any angle the view is turned to.
//
// WHY THE ENDS STOP AT NODES. A node is where ROSS has a number: an element
// sits on one, a probe reads one, a span is between two. So an end stops at
// the nearest node unless Shift is held, and then it goes anywhere along the
// line. Either way it never leaves the line between its first and last node.
//
// This module is a function of what it is given -- the layout and the
// dimensions' plan (core/rotor3d_layout.js) -- and knows nothing of three.js
// or of the page: components/rotor3d_measure.js draws it, features/rotor3d.js
// moves it.

const mm = metres => `${(metres * 1000).toFixed(1)} mm`;

// The line the tape runs on, for the shaft line `half` (null on a rotor): the
// axis's x, its y, the height of the tape -- half a gap above the tallest part
// drawn, under the dimensions' first tier -- and the first and last node.
// Null when that line has no length to measure.
export function measureLine(layout, plan, half) {
    const line = (plan && plan.lines || []).find(entry => entry.half === half);
    if (!line) return null;
    const rings = layout.rings.filter(ring => ring.half === half).sort((a, b) => a.z - b.z);
    if (rings.length < 2) return null;
    // `annotationLayout` puts its tiers at highest + gap and highest + 2 gap.
    const highest = line.total.y - 2 * line.gap;
    return {
        half, x: line.x, y: rings[0].y, tape: highest + 0.5 * line.gap, gap: line.gap,
        z0: line.total.z0, z1: line.total.z1, rings,
    };
}

// The size of an end's handle, in the scene's units.
export function handleRadius(line) {
    return 0.16 * line.gap;
}

// How high the tape reaches, for the camera and the floor.
export function measureTop(line) {
    return line.tape + handleRadius(line);
}

// Where along the tape the ray from `origin` toward `direction` passes
// nearest: the `z` of the closest point of the tape's line, or null when the
// ray runs along it (the view looking down the axis), where every z is as
// near as any other.
export function alongAxis(origin, direction, line) {
    const [dx, dy, dz] = direction;
    const w = [origin[0] - line.x, origin[1] - line.tape, origin[2]];
    const a = dx * dx + dy * dy + dz * dz;
    const across = a - dz * dz;
    if (!(across > 1e-9 * a)) return null;
    const d = dx * w[0] + dy * w[1] + dz * w[2];
    return (a * w[2] - dz * d) / across;
}

// An end at `z`: kept inside the line, and stopped at the nearest node unless
// it is `free`. A free end still names a node it lands exactly on.
export function placeEnd(z, line, free) {
    const inside = Math.min(line.z1, Math.max(line.z0, z));
    let nearest = line.rings[0];
    for (const ring of line.rings) {
        if (Math.abs(ring.z - inside) < Math.abs(nearest.z - inside)) nearest = ring;
    }
    if (!free) return { z: nearest.z, node: nearest.n };
    const on = Math.abs(nearest.z - inside) <= 1e-9 * Math.max(1, line.z1 - line.z0);
    return { z: inside, node: on ? nearest.n : null };
}

// Where the tape starts: from end to end of the line, or, from a node, to the
// last node -- to the first when that node is the last one.
export function startingEnds(line, fromNode = null) {
    const first = line.rings[0];
    const last = line.rings[line.rings.length - 1];
    const from = fromNode === null ? null : line.rings.find(ring => ring.n === fromNode);
    if (!from) return [{ z: first.z, node: first.n }, { z: last.z, node: last.n }];
    const other = from === last ? first : last;
    return [{ z: from.z, node: from.n }, { z: other.z, node: other.n }];
}

// The ends after the rotor was rebuilt: an end on a node stays on that node,
// wherever it went; an end between nodes keeps its place, inside the new line.
export function keptEnds(ends, line) {
    return ends.map(end => {
        const ring = end.node === null ? null : line.rings.find(r => r.n === end.node);
        return ring ? { z: ring.z, node: ring.n } : placeEnd(end.z, line, true);
    });
}

// The shaft's surface under the tape at `z`: its radius, from the node rings
// on either side.
export function surfaceAt(line, z) {
    const { rings } = line;
    let k = 0;
    while (k < rings.length - 2 && rings[k + 1].z < z) k++;
    const a = rings[k];
    const b = rings[k + 1];
    const t = b.z > a.z ? Math.min(1, Math.max(0, (z - a.z) / (b.z - a.z))) : 0;
    return line.y + a.radius + t * (b.radius - a.radius);
}

// What is drawn, as segments in the plane of the line: the tape between the
// ends, a line down from each end to the shaft, and a short tick at every
// node the tape passes over, as a tape measure is graduated.
export function measureSegments(line, ends) {
    const { x, tape, gap } = line;
    const [lo, hi] = [Math.min(ends[0].z, ends[1].z), Math.max(ends[0].z, ends[1].z)];
    const points = [[x, tape, lo], [x, tape, hi]];
    ends.forEach(end => points.push([x, tape, end.z], [x, surfaceAt(line, end.z), end.z]));
    line.rings
        .filter(ring => ring.z > lo && ring.z < hi)
        .forEach(ring => points.push([x, tape, ring.z], [x, tape - 0.2 * gap, ring.z]));
    return points;
}

// What the label says: the distance, and the nodes when both ends are on one.
export function measured(ends) {
    const [a, b] = ends;
    const nodes = a.node !== null && b.node !== null
        ? [Math.min(a.node, b.node), Math.max(a.node, b.node)]
        : null;
    return { distance: Math.abs(b.z - a.z), text: mm(Math.abs(b.z - a.z)), nodes };
}

// The end under the pointer: the nearest whose handle, on screen, is within
// `slop` pixels of it -- or -1. `screen` is each end projected, [x, y, depth].
export function grabbedEnd(pointer, screen, slop = 12) {
    let best = -1;
    let bestDistance = slop;
    screen.forEach((place, k) => {
        const [sx, sy, depth] = place;
        if (!(depth > -1 && depth < 1)) return;
        const distance = Math.hypot(sx - pointer.x, sy - pointer.y);
        if (distance <= bestDistance) {
            best = k;
            bestDistance = distance;
        }
    });
    return best;
}
