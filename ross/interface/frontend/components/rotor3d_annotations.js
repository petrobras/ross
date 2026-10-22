// The node numbers and the dimensions of the 3D view, from
// `annotationLayout` (core/rotor3d_layout.js): the dimension lines, their
// arrows and extension lines in the scene, and every number as text of the
// page laid over the canvas.
//
// WHY THE NUMBERS ARE TEXT OF THE PAGE. Drawn into the scene, a number is a
// texture: blurred close up, a speck from afar, in the colours of the theme
// it was drawn in. As text it stays sharp at every zoom, takes the theme and
// the font of the page, and costs a few hundred positions a frame. What it
// cannot do is hide behind a part -- which a dimension never should anyway.
//
// Numbers that would print over each other are left out, as ROSS's 2D figure
// does with its node labels. Each has a rank: the total length and the first
// and last node of each line always show; the spans next, where they fit;
// the other nodes in what room is left. Turning or zooming the view brings
// back what was left out. The total reads above its line and each span below
// its own, toward the rotor: seen from afar the two tiers are a few pixels
// apart, and two labels over two lines that close printed over each other
// (measured on ROSS's MultiRotor example).
//
// three.js is handed in, not imported (see rotor3d_parts.js).

const mm = metres => `${(metres * 1000).toFixed(1)} mm`;

export function annotationLabels(plan) {
    const labels = [];
    for (const line of plan.lines) {
        const { x, total } = line;
        labels.push({ kind: 'dimension', text: `L = ${mm(total.length)}`, position: [x, total.y, (total.z0 + total.z1) / 2], rank: 2, below: false });
        line.spans.forEach(span => {
            labels.push({ kind: 'dimension', text: mm(span.length), position: [x, span.y, (span.z0 + span.z1) / 2], rank: 1, below: true });
        });
    }
    const byHalf = new Map();
    plan.nodes.forEach(node => {
        if (!byHalf.has(node.half)) byHalf.set(node.half, []);
        byHalf.get(node.half).push(node);
    });
    byHalf.forEach(nodes => {
        nodes.sort((a, b) => a.position[2] - b.position[2]);
        nodes.forEach((node, k) => labels.push({
            kind: 'node', text: String(node.n), position: node.position, rank: k === 0 || k === nodes.length - 1 ? 2 : 0, below: false,
        }));
    });
    return labels;
}

// The lines: each dimension with an open arrow at both ends, and the
// extension lines, in one set of segments in the plane of its shaft line.
export function annotationSegments(plan) {
    const points = [];
    const segment = (x, a, b) => points.push([x, a[0], a[1]], [x, b[0], b[1]]);
    for (const line of plan.lines) {
        const { x, gap } = line;
        const dimension = ({ z0, z1, y }) => {
            segment(x, [y, z0], [y, z1]);
            const arrow = Math.min(0.35 * gap, 0.2 * (z1 - z0));
            for (const [tip, inward] of [[z0, 1], [z1, -1]]) {
                segment(x, [y, tip], [y + 0.4 * arrow, tip + inward * arrow]);
                segment(x, [y, tip], [y - 0.4 * arrow, tip + inward * arrow]);
            }
        };
        dimension(line.total);
        line.spans.forEach(dimension);
        line.extensions.forEach(({ z, from, to }) => segment(x, [from + 0.1 * gap, z], [to + 0.25 * gap, z]));
    }
    return points;
}

export function buildAnnotations(THREE, plan, colour) {
    const points = annotationSegments(plan);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(points.flat()), 3));
    const material = new THREE.LineBasicMaterial({ color: colour || 'gray' });
    const object = new THREE.LineSegments(geometry, material);
    object.name = 'dimensions';
    return {
        object,
        labels: annotationLabels(plan),
        restyle: next => { if (next) material.color.set(next); },
        dispose: () => {
            geometry.dispose();
            material.dispose();
        },
    };
}

// The labels into `layer`, one element each, kept in the same order.
export function fillLabels(layer, labels) {
    // The class written out whole, so that a search for it finds this line.
    layer.innerHTML = labels.map(label => {
        if (label.kind !== 'dimension') return '<span class="rotor3d-label"></span>';
        return label.below ? '<span class="rotor3d-label is-dimension is-below"></span>' : '<span class="rotor3d-label is-dimension"></span>';
    }).join('');
    const spans = Array.from(layer.children);
    spans.forEach((span, k) => { span.textContent = labels[k].text; });
    return spans;
}

// Where each label is on screen, and which are shown: by rank, and in order
// within it -- rank 2 always, the others where they overlap none shown.
// `project` takes [x, y, z] to [sx, sy, depth] in CSS pixels, depth outside
// -1..1 when it is behind the camera or past the far plane.
export function arrangeLabels(labels, project, width, height) {
    const placed = [];
    const boxes = [];
    const fits = box => boxes.every(other => box.x1 <= other.x0 || box.x0 >= other.x1 || box.y1 <= other.y0 || box.y0 >= other.y1);
    const order = labels.map((label, k) => k).sort((a, b) => labels[b].rank - labels[a].rank || a - b);
    for (const k of order) {
        const label = labels[k];
        const [sx, sy, depth] = project(label.position);
        const inView = depth > -1 && depth < 1 && sx >= 0 && sx <= width && sy >= 0 && sy <= height;
        const w = 7 * label.text.length + 8;
        const h = 16;
        const box = label.below
            ? { x0: sx - w / 2, x1: sx + w / 2, y0: sy + 4, y1: sy + 4 + h }
            : { x0: sx - w / 2, x1: sx + w / 2, y0: sy - h - 4, y1: sy - 4 };
        const shown = inView && (label.rank === 2 || fits(box));
        if (shown) boxes.push(box);
        placed[k] = { x: sx, y: sy, shown };
    }
    return placed;
}
