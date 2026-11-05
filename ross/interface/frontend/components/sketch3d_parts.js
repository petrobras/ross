// What the sketch and the extrusion draw in the 3D view (features/sketch3d.js):
// the sketch plane with its grid, the circles, the translucent element being
// extruded and the arrow that pulls it. three.js comes in as an argument, like
// the rotor's parts (components/rotor3d_parts.js), and every builder answers a
// `dispose` that frees what it made.
//
// The sketch is drawn over what is already there (`depthTest: false`): on the
// face of an element the plane lies exactly on the face, and depth-tested lines
// would flicker in and out of it.

const SEGMENTS = 128;
const OVERLAY = 10;

function overlayLine(THREE, geometry, colour, opacity = 1) {
    const material = new THREE.LineBasicMaterial({
        color: colour, transparent: opacity < 1, opacity, depthTest: false, depthWrite: false, toneMapped: false,
    });
    const line = new THREE.LineSegments(geometry, material);
    line.renderOrder = OVERLAY;
    return line;
}

// The plane z = `z`: a square grid of `spacing` metres out to `extent` on each
// side of the axis, every fifth line heavier, and the x and y axes through it.
export function buildSketchPlane(THREE, { z, extent, spacing, colours }) {
    const group = new THREE.Group();
    const minor = [];
    const major = [];
    const count = Math.max(1, Math.round(extent / spacing));
    const edge = count * spacing;
    for (let k = -count; k <= count; k += 1) {
        const v = k * spacing;
        const target = k % 5 === 0 ? major : minor;
        if (k === 0) continue;
        target.push(-edge, v, 0, edge, v, 0, v, -edge, 0, v, edge, 0);
    }
    const lines = (points, colour, opacity) => {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
        return overlayLine(THREE, geometry, colour, opacity);
    };
    if (minor.length) group.add(lines(minor, colours.grid, 0.35));
    if (major.length) group.add(lines(major, colours.grid, 0.7));
    group.add(lines([-edge, 0, 0, edge, 0, 0], colours.x, 0.9));
    group.add(lines([0, -edge, 0, 0, edge, 0], colours.y, 0.9));
    group.add(lines([-edge, -edge, 0, edge, -edge, 0, edge, -edge, 0, edge, edge, 0,
        edge, edge, 0, -edge, edge, 0, -edge, edge, 0, -edge, -edge, 0], colours.grid, 0.9));
    group.position.z = z;
    return {
        object: group,
        edge,
        dispose: () => {
            group.traverse(o => {
                if (o.geometry) o.geometry.dispose();
                if (o.material) o.material.dispose();
            });
        },
    };
}

// A circle of the sketch, centred on the axis in the plane z = `z`. One unit
// circle, scaled: drawing a circle while the pointer moves changes a number,
// not a geometry.
export function buildCircle(THREE, z, colour) {
    const points = [];
    for (let k = 0; k < SEGMENTS; k += 1) {
        const a0 = (2 * Math.PI * k) / SEGMENTS;
        const a1 = (2 * Math.PI * (k + 1)) / SEGMENTS;
        points.push(Math.cos(a0), Math.sin(a0), 0, Math.cos(a1), Math.sin(a1), 0);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    const line = overlayLine(THREE, geometry, colour);
    line.position.z = z;
    return {
        object: line,
        setDiameter: (millimetres) => {
            const r = Math.max(millimetres, 1e-6) / 2000;
            line.scale.set(r, r, 1);
        },
        setColour: (value) => {
            line.material.color.set(value);
        },
        dispose: () => {
            geometry.dispose();
            line.material.dispose();
        },
    };
}

// The element being extruded: translucent, hollow when it has a bore, from the
// face to `length` along the axis, with an arrow on its far face to pull it.
export function buildExtrusion(THREE, colours) {
    const group = new THREE.Group();
    // Drawn over the rotor, like the sketch: an element under construction is
    // seen through the bearing or the disk on the face it starts from.
    const body = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshStandardMaterial({
        color: colours.accent, transparent: true, opacity: 0.38, depthWrite: false, depthTest: false,
        metalness: 0.2, roughness: 0.5, side: THREE.DoubleSide,
    }));
    const edges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({
        color: colours.accent, depthTest: false,
    }));
    body.renderOrder = OVERLAY;
    edges.renderOrder = OVERLAY + 1;
    group.add(body, edges);

    // The arrow: a shaft and a head, pointing +y before it is turned onto the axis.
    const arrow = new THREE.Group();
    const arrowMaterial = new THREE.MeshBasicMaterial({ color: colours.arrow, toneMapped: false, depthTest: false });
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.7, 16), arrowMaterial);
    stem.position.y = 0.35;
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.3, 24), arrowMaterial);
    head.position.y = 0.85;
    stem.renderOrder = OVERLAY;
    head.renderOrder = OVERLAY;
    arrow.add(stem, head);
    group.add(arrow);

    // The handle of a taper: a bead on the rim of the far face, pulled away
    // from the axis or toward it. Shown only while the element is tapered.
    const handleMaterial = new THREE.MeshBasicMaterial({ color: colours.arrow, toneMapped: false, depthTest: false });
    const handle = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 12), handleMaterial);
    handle.renderOrder = OVERLAY;
    handle.visible = false;
    group.add(handle);

    // `far`: the far face of a taper, `{ od, id }` in mm; null for a straight
    // element, whose far face is the sketched one.
    function set(face, length, section, far = null) {
        const outer = section.odl / 2000;
        const inner = section.idl / 2000;
        const farOuter = (far ? far.od : section.odl) / 2000;
        const farInner = (far ? far.id : section.idl) / 2000;
        const span = length / 1000;
        // The profile turned about the axis: from the bore to the outside at
        // the sketched face, to those of the far face, closed at both ends.
        const profile = [
            new THREE.Vector2(inner, 0), new THREE.Vector2(outer, 0),
            new THREE.Vector2(farOuter, span), new THREE.Vector2(farInner, span), new THREE.Vector2(inner, 0),
        ];
        const geometry = new THREE.LatheGeometry(profile, 64);
        geometry.rotateX(face.sense < 0 ? -Math.PI / 2 : Math.PI / 2);
        body.geometry.dispose();
        body.geometry = geometry;
        edges.geometry.dispose();
        edges.geometry = new THREE.EdgesGeometry(geometry, 30);
        group.position.set(0, 0, face.z);
        // The arrow stands on the far face, on the axis, as long as the section is wide.
        const size = Math.max(outer, farOuter, 0.005) * 1.2;
        arrow.scale.setScalar(size);
        arrow.position.set(0, 0, face.sense * span);
        arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, face.sense < 0 ? -1 : 1));
        handle.visible = !!far;
        handle.scale.setScalar(size * 0.12);
        handle.position.set(0, farOuter, face.sense * span);
    }

    return {
        object: group,
        arrow,
        handle,
        set,
        setHover: (on) => {
            arrowMaterial.color.set(on ? colours.arrowHover : colours.arrow);
        },
        setHandleHover: (on) => {
            handleMaterial.color.set(on ? colours.arrowHover : colours.arrow);
        },
        dispose: () => {
            body.geometry.dispose();
            body.material.dispose();
            edges.geometry.dispose();
            edges.material.dispose();
            stem.geometry.dispose();
            head.geometry.dispose();
            arrowMaterial.dispose();
            handle.geometry.dispose();
            handleMaterial.dispose();
        },
    };
}
