// The measuring tape drawn in the 3D view (core/rotor3d_measure.js says where
// it goes): the tape, its graduations and the lines down to the shaft, and a
// handle at each end. The handles are drawn over everything, so an end is
// always there to be grabbed, even with a part between it and the camera.
//
// three.js is handed in, not imported (see rotor3d_parts.js).
import { handleRadius, measureSegments } from '../core/rotor3d_measure.js';

export function buildMeasure(THREE, line, ends, colour) {
    const object = new THREE.Group();
    object.name = 'measure';
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial({ color: colour || 'steelblue', depthTest: false, transparent: true });
    const tape = new THREE.LineSegments(geometry, material);
    tape.renderOrder = 10;
    const ball = new THREE.SphereGeometry(1, 20, 12);
    const handleMaterial = new THREE.MeshBasicMaterial({ color: colour || 'steelblue', depthTest: false, transparent: true });
    const handles = [0, 1].map(() => {
        const handle = new THREE.Mesh(ball, handleMaterial);
        handle.renderOrder = 11;
        object.add(handle);
        return handle;
    });
    object.add(tape);

    const update = (nextLine, nextEnds) => {
        const points = measureSegments(nextLine, nextEnds);
        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(points.flat()), 3));
        geometry.computeBoundingSphere();
        const size = handleRadius(nextLine);
        nextEnds.forEach((end, k) => {
            handles[k].position.set(nextLine.x, nextLine.tape, end.z);
            handles[k].scale.setScalar(size);
        });
    };
    update(line, ends);

    return {
        object,
        update,
        restyle: next => {
            if (!next) return;
            material.color.set(next);
            handleMaterial.color.set(next);
        },
        dispose: () => {
            geometry.dispose();
            material.dispose();
            ball.dispose();
            handleMaterial.dispose();
        },
    };
}
