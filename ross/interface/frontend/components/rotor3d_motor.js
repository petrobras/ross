// The motor drawn with the bench (`motorLayout` in core/rotor3d_layout.js): a
// totally enclosed, fan-cooled induction motor, as rotor rigs are driven by.
// Only a picture, like the bench it comes with.
//
// Seen from its back to the rotor: the fan cover with its grille, the end
// bell, the finned frame with the terminal box on top and the feet under it,
// the drive-end bell, its shaft, and a jaw coupling to the rotor's first shaft
// element. Built along z like everything else, and handed back as pieces
// ({ geometry, color, finish }, rotor3d_solids.js) with fixed colours: ROSS
// colours no motor.
//
// three.js is handed in, not imported (see rotor3d_parts.js).
import { SEGMENTS, STEEL, boltCircle, boltHead, faceted, ring } from './rotor3d_solids.js';

// Materials, as numbers like the bench's: the frame's paint, the grille and
// the coupling's spider, the nameplate.
const MOTOR_PAINT = 0x46627d;
const GRILLE = 0x2c343d;
const SPIDER = 0x1c1f23;
const PLATE = 0xd9dde2;
const HUB = 0x8e98a2;

export function motorPieces(THREE, motor, edges) {
    const { radius: R, shaftRadius: r, z0, z1 } = motor;
    const length = z1 - z0;
    const pieces = [];
    const paint = geometry => pieces.push({ geometry, color: MOTOR_PAINT, finish: 'paint' });
    const metal = (geometry, color = STEEL) => pieces.push({ geometry, color, finish: 'metal' });
    const box = (sx, sy, sz, x, y, z) => {
        const geometry = new THREE.BoxGeometry(sx, sy, sz);
        geometry.translate(x, y, z);
        return faceted(THREE, geometry, edges);
    };

    // Along the frame, from the back: fan cover, end bell, finned frame, end bell.
    const cowl = 0.2 * length;
    const bell = 0.09 * length;
    const frameFrom = z0 + cowl + bell;
    const frameTo = z1 - bell;
    const core = 0.8 * R;

    // The fan cover: a drum a little smaller than the fins, closed by a grille
    // of rings on its back.
    paint(ring(THREE, 0.55 * R, 0.9 * R, z0 + 0.02 * length, z0 + cowl, edges, SEGMENTS.part, 0.04 * R));
    for (let i = 0; i < 4; i++) {
        const inner = 0.12 * R + i * 0.11 * R;
        metal(ring(THREE, inner, inner + 0.06 * R, z0, z0 + 0.03 * length, edges), GRILLE);
    }
    metal(ring(THREE, 0, 0.9 * R, z0 + 0.03 * length, z0 + 0.035 * length, null), GRILLE);

    // The end bells, each with the boss of its bearing, and the frame.
    paint(ring(THREE, 0.2 * R, 0.86 * R, z0 + cowl, frameFrom, edges, SEGMENTS.part, 0.05 * R));
    paint(ring(THREE, 0.2 * R, 0.86 * R, frameTo, z1 - 0.2 * bell, edges, SEGMENTS.part, 0.05 * R));
    paint(ring(THREE, 1.02 * r, 0.34 * R, z1 - 0.2 * bell, z1, edges, SEGMENTS.part, 0.04 * R));
    paint(ring(THREE, 0, core, frameFrom, frameTo, edges));
    boltCircle(THREE, 4, 0.7 * R, 0.05 * R, 0.04 * R, z1 - 0.2 * bell + 0.02 * R, edges, Math.PI / 4)
        .forEach(g => metal(g));

    // Cooling fins round the frame, a fin every 12 degrees to the full
    // radius, but where the feet, the terminal box and the nameplate are.
    const fins = 30;
    for (let i = 0; i < fins; i++) {
        const a = (i / fins) * Math.PI * 2;
        const feet = Math.sin(a) < -0.65;
        const terminal = Math.sin(a) > 0.95;
        const nameplate = Math.cos(a) < -0.99;
        if (feet || terminal || nameplate) continue;
        const fin = new THREE.BoxGeometry(0.035 * R, R - 0.97 * core, frameTo - frameFrom);
        fin.translate(0, 0.97 * core + (R - 0.97 * core) / 2, 0);
        fin.rotateZ(a - Math.PI / 2);
        fin.translate(0, 0, (frameFrom + frameTo) / 2);
        paint(faceted(THREE, fin, edges));
    }

    // The terminal box on top, its lid, and a cable gland on the side facing
    // the viewer.
    const middle = (frameFrom + frameTo) / 2;
    paint(box(0.62 * R, 0.4 * R, 0.5 * length, 0, core + 0.2 * R, middle));
    paint(box(0.68 * R, 0.06 * R, 0.56 * length, 0, core + 0.43 * R, middle));
    const gland = new THREE.CylinderGeometry(0.07 * R, 0.07 * R, 0.14 * R, 12);
    gland.rotateZ(Math.PI / 2);
    gland.translate(-0.38 * R, core + 0.2 * R, middle);
    metal(faceted(THREE, gland, edges));

    // The nameplate, on the side towards the viewer.
    metal(box(0.02 * R, 0.22 * R, 0.3 * length, -1.01 * core, 0, middle), PLATE);

    // The feet: under each corner of the frame, a post down to where the
    // layout puts the bottom, on a flange bolted to what it stands on.
    const bottom = motor.feet - motor.y;
    const footTop = -0.6 * R;
    const flange = 0.12 * R;
    for (const x of [-0.7 * R, 0.7 * R]) {
        for (const z of [frameFrom + 0.12 * length, frameTo - 0.12 * length]) {
            paint(box(0.3 * R, footTop - bottom, 0.2 * length, x, (footTop + bottom) / 2, z));
            paint(box(0.56 * R, flange, 0.24 * length, x + Math.sign(x) * 0.02 * R, bottom + flange / 2, z));
            metal(boltHead(THREE, 0.05 * R, 0.07 * R, 'y', x + Math.sign(x) * 0.2 * R, bottom + flange + 0.035 * R, z, edges));
        }
    }
    // A rib under the frame joins the feet, so the frame stands on them.
    paint(box(1.2 * R, 0.3 * R, frameTo - frameFrom, 0, -0.68 * R, middle));

    // The shaft, from the drive end to the rotor's first shaft element.
    metal(ring(THREE, 0, r, z1 - 0.05 * length, motor.shaftTo, edges, SEGMENTS.shaft));

    // A jaw coupling in the gap: two hubs and the spider between them.
    const [c0, c1] = motor.coupling;
    const span = c1 - c0;
    const hubR = 1.9 * r;
    metal(ring(THREE, r, hubR, c0, c0 + 0.42 * span, edges, SEGMENTS.part, 0.1 * r), HUB);
    metal(ring(THREE, r, hubR, c1 - 0.42 * span, c1, edges, SEGMENTS.part, 0.1 * r), HUB);
    pieces.push({ geometry: ring(THREE, 1.05 * r, 0.92 * hubR, c0 + 0.42 * span, c1 - 0.42 * span, edges),
        color: SPIDER, finish: 'paint' });

    // In place: on the axis of the line it drives.
    for (const piece of pieces) piece.geometry.translate(motor.x, motor.y, 0);
    return pieces;
}
