// The axis triad in a corner of the 3D view: ROSS's x, y and z, turned as the
// camera turns.
//
// WHY. ROSS gives its results in its own axes -- orbits and mode shapes in x
// and y, the rotor along z, y up -- and the scene is laid out in the same ones
// (core/rotor3d_layout.js). A view that can be turned any way loses them after
// the first drag; the triad keeps them in sight.
//
// It is drawn by the view's own renderer, in a second pass over a corner
// (`drawTriad` in features/rotor3d.js). A renderer of its own would be a second
// WebGL context for three arrows, out of the handful a page gets before the
// browser drops the oldest -- and the oldest would be the rotor's.
//
// The letters are strokes, not text: no canvas, no font, no DOM -- this module,
// like rotor3d_parts.js, gets three.js injected and touches nothing else -- and
// they stay sharp at any pixel ratio. Two draw calls: the arrows in one mesh,
// the letters in one set of lines, each vertex coloured by its axis.
import { mergeColoured } from './rotor3d_parts.js';

// The letters, as segments [x0, y0, x1, y1] in a unit box centred on the origin.
export const LETTER_STROKES = {
    x: [[-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5]],
    y: [[-0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, -0.5]],
    z: [[-0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, -0.5]],
};

export const AXES = [
    { name: 'x', direction: [1, 0, 0] },
    { name: 'y', direction: [0, 1, 0] },
    { name: 'z', direction: [0, 0, 1] },
];

const SHAFT = { radius: 0.045, length: 0.7 };
const HEAD = { radius: 0.12, length: 0.3 };
// Where the letters sit along each axis, and how big they are.
const LABEL_AT = 1.3;
const LABEL_SIZE = 0.34;
// Half the side of what the triad's camera sees: the arrows and the letters
// in every orientation.
export const TRIAD_REACH = LABEL_AT + LABEL_SIZE;

// An arrow along +y from the origin, then turned onto its axis.
function arrow(THREE, direction) {
    const shaft = new THREE.CylinderGeometry(SHAFT.radius, SHAFT.radius, SHAFT.length, 12);
    shaft.translate(0, SHAFT.length / 2, 0);
    const head = new THREE.ConeGeometry(HEAD.radius, HEAD.length, 16);
    head.translate(0, SHAFT.length + HEAD.length / 2, 0);
    const turn = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0), new THREE.Vector3(...direction));
    return [shaft, head].map(piece => {
        piece.applyQuaternion(turn);
        const plain = piece.toNonIndexed();
        piece.dispose();
        return plain;
    });
}

// Where each letter goes on the screen of the triad's camera: its axis's tip,
// turned by `turn` (the inverse of the view camera's rotation).
export function letterCentres(THREE, turn) {
    return AXES.map(axis => new THREE.Vector3(...axis.direction).multiplyScalar(LABEL_AT).applyQuaternion(turn));
}

export function buildTriad(THREE, colours) {
    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-TRIAD_REACH, TRIAD_REACH, TRIAD_REACH, -TRIAD_REACH, 0.1, 10);
    camera.position.set(0, 0, 5);

    // The arrows: one mesh, the vertices of each arrow in the colour of its axis.
    const axisOfVertex = [];
    const pieces = [];
    AXES.forEach((axis, a) => {
        for (const geometry of arrow(THREE, axis.direction)) {
            for (let i = 0; i < geometry.attributes.position.count; i++) axisOfVertex.push(a);
            pieces.push({ geometry, color: new THREE.Color() });
        }
    });
    const arrows = new THREE.Mesh(mergeColoured(THREE, pieces),
        new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false }));

    // The letters: one set of lines, written again each frame where the tips are.
    const strokes = [];
    AXES.forEach((axis, a) => LETTER_STROKES[axis.name].forEach(s => strokes.push({ a, s })));
    const letterGeometry = new THREE.BufferGeometry();
    letterGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(strokes.length * 6), 3));
    letterGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(strokes.length * 6), 3));
    const letters = new THREE.LineSegments(letterGeometry,
        new THREE.LineBasicMaterial({ vertexColors: true, toneMapped: false, depthTest: false }));
    letters.renderOrder = 1;
    letters.frustumCulled = false;

    const axes = new THREE.Group();
    axes.add(arrows);
    scene.add(axes, letters);

    // The colours of the axes, from the page's theme (read by the caller).
    const restyle = next => {
        const tone = AXES.map(axis => new THREE.Color().setStyle(next[axis.name] || 'gray'));
        const arrowColour = arrows.geometry.attributes.color;
        axisOfVertex.forEach((a, i) => arrowColour.setXYZ(i, tone[a].r, tone[a].g, tone[a].b));
        arrowColour.needsUpdate = true;
        const letterColour = letterGeometry.attributes.color;
        strokes.forEach(({ a }, i) => {
            letterColour.setXYZ(2 * i, tone[a].r, tone[a].g, tone[a].b);
            letterColour.setXYZ(2 * i + 1, tone[a].r, tone[a].g, tone[a].b);
        });
        letterColour.needsUpdate = true;
    };
    restyle(colours);

    // Turned as the view is: the arrows by the inverse of the view camera's
    // rotation, the letters moved to the tips and kept upright on the screen.
    const orient = viewQuaternion => {
        axes.quaternion.copy(viewQuaternion).invert();
        const centres = letterCentres(THREE, axes.quaternion);
        const position = letterGeometry.attributes.position;
        strokes.forEach(({ a, s }, i) => {
            const c = centres[a];
            position.setXYZ(2 * i, c.x + LABEL_SIZE * s[0], c.y + LABEL_SIZE * s[1], 1);
            position.setXYZ(2 * i + 1, c.x + LABEL_SIZE * s[2], c.y + LABEL_SIZE * s[3], 1);
        });
        position.needsUpdate = true;
    };
    orient(new THREE.Quaternion());

    return {
        scene,
        camera,
        restyle,
        orient,
        dispose: () => {
            arrows.geometry.dispose();
            arrows.material.dispose();
            letterGeometry.dispose();
            letters.material.dispose();
            scene.clear();
        },
    };
}
