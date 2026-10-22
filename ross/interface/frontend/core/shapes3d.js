// The geometry bank: other shapes an element can be drawn with in the 3D view
// -- a compressor impeller, a turbine wheel, a fan -- instead of its default.
//
// ONLY A PICTURE. Leonardo's decision: ROSS goes on computing with the mass
// and inertias of the element's form, and the shape chosen here is drawn in
// the envelope the element already has (its outer radius, its width). It is
// saved with the element, as `shape3d`, so a project opens as it was left;
// the server never passes it to ROSS nor writes it into the exported script
// (`VIEW_FIELDS` in domain/rotor_builder.py).
//
// The catalogue is data, here; how each shape is built is
// components/rotor3d_shapes.js, and the picker in the form is
// components/shape_picker.js. A key this catalogue does not know -- a project
// from a later version, a typo in a hand-edited file -- is drawn as if
// nothing had been chosen, never as an error.
import { t } from './i18n.js';

// The element field the choice is kept in.
export const SHAPE_FIELD = 'shape3d';

// `minWidth`: some shapes need depth to read as what they are -- an impeller
// is a cone of blades, not a plate. Their width is at least this fraction of
// the outer radius (the layout widens the part, so what the pointer finds and
// what the camera frames stay what is drawn).
//
// `plain`: the category's own drawing (rotor3d_parts.js), named so it can be
// chosen over what the model would say -- a pillow block for a ball bearing.
// It has no builder in rotor3d_shapes.js.
//
// THE EMPTY KEY. Where the element's model says what it is -- a ROSS
// `BallBearingElement` is a rolling bearing, a `HolePatternSeal` a hole-pattern
// seal, a gear with a helix angle a helical gear -- nothing chosen means "as
// the model says" (`automaticShape`); Leonardo asked for bearings to follow
// their model. Where it says nothing (a disk, a point mass, a coupling), it
// means the category's own drawing, as before.
const CATALOGUE = {
    disks: [
        { key: '', name: () => t('shapeDiskPlain'), minWidth: 0 },
        { key: 'impeller', name: () => t('shapeImpeller'), minWidth: 0.42 },
        { key: 'closed_impeller', name: () => t('shapeClosedImpeller'), minWidth: 0.42 },
        { key: 'axial', name: () => t('shapeAxialStage'), minWidth: 0.16 },
        { key: 'turbine', name: () => t('shapeTurbine'), minWidth: 0.16 },
        { key: 'fan', name: () => t('shapeFan'), minWidth: 0.22 },
        { key: 'flywheel', name: () => t('shapeFlywheel'), minWidth: 0 },
        { key: 'pulley', name: () => t('shapePulley'), minWidth: 0.2 },
    ],
    gears: [
        { key: '', name: () => t('shapeByModel'), minWidth: 0, automatic: true },
        { key: 'spur', name: () => t('shapeSpurGear'), minWidth: 0, plain: true },
        { key: 'helical', name: () => t('shapeHelicalGear'), minWidth: 0 },
    ],
    // Every support stands on the same base as the pillow block, so the
    // bench's pedestal meets it whatever it is drawn as.
    bearings: [
        { key: '', name: () => t('shapeByModel'), minWidth: 0, automatic: true },
        { key: 'pillow', name: () => t('shapePillowBlock'), minWidth: 0, plain: true },
        { key: 'rolling', name: () => t('shapeRollingBearing'), minWidth: 0 },
        { key: 'tilting_pad', name: () => t('shapeTiltingPad'), minWidth: 0 },
        { key: 'magnetic', name: () => t('shapeMagneticBearing'), minWidth: 0 },
    ],
    seals: [
        { key: '', name: () => t('shapeByModel'), minWidth: 0, automatic: true },
        { key: 'labyrinth', name: () => t('shapeLabyrinth'), minWidth: 0, plain: true },
        { key: 'hole_pattern', name: () => t('shapeHolePattern'), minWidth: 0 },
        { key: 'honeycomb', name: () => t('shapeHoneycomb'), minWidth: 0 },
        { key: 'brush', name: () => t('shapeBrushSeal'), minWidth: 0 },
    ],
    pointmasses: [
        { key: '', name: () => t('shapeCollar'), minWidth: 0 },
        { key: 'balance_weight', name: () => t('shapeBalanceWeight'), minWidth: 0 },
        { key: 'nut', name: () => t('shapeLockNut'), minWidth: 0 },
    ],
    couplings: [
        { key: '', name: () => t('shapeDiscPack'), minWidth: 0 },
        { key: 'gear_coupling', name: () => t('shapeGearCoupling'), minWidth: 0 },
        { key: 'jaw', name: () => t('shapeJawCoupling'), minWidth: 0 },
        { key: 'rigid', name: () => t('shapeRigidCoupling'), minWidth: 0 },
    ],
};

// What each ROSS class is drawn as, by the name the scene gives it (`kind`,
// domain/rotor_scene.py). A class not listed -- a plain `BearingElement`, a
// fluid-film bearing, a `LabyrinthSeal` -- is the category's own drawing.
const BY_CLASS = {
    bearings: {
        BallBearingElement: 'rolling',
        RollerBearingElement: 'rolling',
        TiltingPad: 'tilting_pad',
        MagneticBearingElement: 'magnetic',
    },
    seals: {
        HolePatternSeal: 'hole_pattern',
    },
};

// Below this a helix angle is a spur gear (radians, as the scene gives it).
const NO_HELIX = 1e-6;

// The shapes offered for a category, the first one what nothing chosen means.
// None for a category the bank does not cover yet.
export function shapesFor(category) {
    return CATALOGUE[category] || [];
}

// The shape a stored key means, or null for nothing chosen (and for a key the
// catalogue does not know).
export function shapeOf(category, key) {
    if (!key) return null;
    return shapesFor(category).find(shape => shape.key === key && !shape.automatic) || null;
}

// The shape an element's model says it is, from the scene's entry for it; null
// where the model says nothing.
export function automaticShape(category, entry) {
    const shapes = shapesFor(category);
    if (!entry || !shapes.length || !shapes[0].automatic) return null;
    const key = category === 'gears'
        ? (Math.abs(entry.helix_angle || 0) > NO_HELIX ? 'helical' : 'spur')
        : (BY_CLASS[category] || {})[entry.kind] || shapes.find(shape => shape.plain).key;
    return shapeOf(category, key);
}

// What an element is drawn as: the shape chosen in its form, or else what its
// model says, or else null for the category's own drawing.
export function drawnShape(category, key, entry) {
    return shapeOf(category, key) || automaticShape(category, entry);
}
