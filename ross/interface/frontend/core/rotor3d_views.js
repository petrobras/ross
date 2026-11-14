// The standard views of the 3D view, and the camera's flight between two of
// them (features/rotor3d.js). No three.js here, so node can test it.
//
// The axes are ROSS's: z along the rotor, y up, x across (the triad's). A view
// is the direction from the middle of the rotor to the camera; `framing` then
// sets the distance that takes the whole rotor in.
//
//   * home: the opening view (`viewDirection`): the side, a little above and
//     in front of node 0; for a MultiRotor, clear of the line between axes;
//   * iso: the isometric view, the three axes at equal angles to the screen;
//   * front: along the rotor axis, from the side of node 0 -- the plane of the
//     cross-sections, where a shaft is two circles;
//   * side: square to the axis and level, as ROSS's 2D figure is drawn;
//   * top: from above.
//
// Every view but `front` reads left to right as the 2D figure does, node 0 on
// the left: seen from `d`, the screen's right is (d.z, 0, -d.x), and z grows
// to the right exactly when d.x < 0 (core/rotor3d_layout.js says why). The top
// view leans that way by a hair: straight down, the direction around the
// vertical is undefined, and OrbitControls would choose it.
import { viewDirection } from './rotor3d_layout.js';

export const VIEWS = ['home', 'iso', 'front', 'side', 'top'];

const DIRECTIONS = {
    iso: [-1, 1, -1],
    front: [0, 0, -1],
    side: [-1, 0, 0],
    top: [-1e-3, 1, 0],
};

// Long enough to see where the camera went, short enough not to be waited
// for the tenth time (the person's choice: about half a second).
export const FLIGHT_MS = 500;

export function viewTowards(name, scene) {
    const direction = DIRECTIONS[name];
    return direction ? normalized(direction) : viewDirection(scene);
}

export function easeInOut(t) {
    const u = Math.min(1, Math.max(0, t));
    return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
}

// The camera at fraction `t` (0 to 1, already eased) of the way from `from` to
// `to`, each { position, target }. The point looked at moves in a straight
// line; the camera turns around it along the great circle between the two
// directions, and its distance changes by the same ratio every step, so a
// zoom from far to near does not rush at the end.
export function cameraBetween(from, to, t) {
    const target = from.target.map((value, i) => value + (to.target[i] - value) * t);
    const start = sub(from.position, from.target);
    const end = sub(to.position, to.target);
    const r0 = Math.hypot(...start);
    const r1 = Math.hypot(...end);
    const distance = r0 > 0 && r1 > 0 ? r0 * Math.pow(r1 / r0, t) : r0 + (r1 - r0) * t;
    const direction = turn(normalized(start), normalized(end), t);
    return { target, position: target.map((value, i) => value + direction[i] * distance) };
}

// Along the great circle from `a` to `b`. Opposite directions have no single
// great circle; the way round goes over the top (or, from straight above, by
// x), so the flight never passes under the floor.
function turn(a, b, t) {
    const cosine = Math.max(-1, Math.min(1, dot(a, b)));
    if (cosine > 1 - 1e-12) return b;
    if (cosine < -1 + 1e-9) {
        let across = cross(a, [0, 1, 0]);
        if (Math.hypot(...across) < 1e-9) across = cross(a, [1, 0, 0]);
        const over = normalized(cross(normalized(across), a));
        const middle = over[1] >= 0 ? over : over.map(v => -v);
        return t < 0.5 ? turn(a, middle, 2 * t) : turn(middle, b, 2 * t - 1);
    }
    const angle = Math.acos(cosine);
    const s = Math.sin(angle);
    const wa = Math.sin((1 - t) * angle) / s;
    const wb = Math.sin(t * angle) / s;
    return normalized(a.map((value, i) => value * wa + b[i] * wb));
}

function sub(a, b) {
    return a.map((value, i) => value - b[i]);
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
