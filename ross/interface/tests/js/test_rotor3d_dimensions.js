// The node numbers and the dimensions of the 3D view, shown with a button
// (core/rotor3d_layout.js `annotationLayout`, components/rotor3d_annotations.js).
//
// WHY THIS BATTERY EXISTS. A dimension is a number someone will read and
// trust. The claims are the ones a wrong one would break without anything
// else noticing: the total is the length of the shaft line, the spans run
// end -- bearing -- bearing -- end and add up to it, a bearing hanging under a
// link node does not cut the line, every tier clears what is drawn, each line
// of a MultiRotor has its own, and numbers that would print over each other
// are left out the way ROSS's 2D figure leaves them out -- never the first
// and last node, never a dimension.
import fs from 'node:fs';
import { check, disk, node, registerSelector, shutDown } from './fake_dom.js';

const THREE = await import('../../frontend/vendor/three/three.module.js');
const { annotationLayout, layoutScene } = await import('../../frontend/core/rotor3d_layout.js');
const { annotationLabels, annotationSegments, arrangeLabels, buildAnnotations, fillLabels } =
    await import('../../frontend/components/rotor3d_annotations.js');

const CASES = JSON.parse(fs.readFileSync(new URL('../golden/rotor_scenes.json', import.meta.url)));
const near = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));

// --- the dimensions -------------------------------------------------------------------
console.log('\nThe dimensions, on every scene ROSS built');

for (const [name, { scene }] of Object.entries(CASES)) {
    const layout = layoutScene(scene);
    const plan = annotationLayout(layout);
    const lines = scene.kind === 'multirotor' ? [scene.driving, scene.driven] : [scene];
    check(`${name}: a set of dimensions per shaft line`, plan.lines.length === lines.length);
    plan.lines.forEach((line, k) => {
        const rings = layout.rings.filter(r => r.half === line.half);
        const zs = rings.map(r => r.z);
        const parts = layout.parts.filter(p => p.half === line.half);
        check(`${name} ${line.half || ''}: the total is the shaft line, first node to last`,
            near(line.total.z0, Math.min(...zs)) && near(line.total.z1, Math.max(...zs))
            && near(line.total.length, lines[k].nodes[lines[k].nodes.length - 1].z - lines[k].nodes[0].z));
        const inside = lines[k].bearings.filter(b => !b.link)
            .map(b => b.z + (layout.parts.find(p => p.half === line.half)?.offset.z || 0))
            .filter(z => z > line.total.z0 + 1e-9 && z < line.total.z1 - 1e-9);
        if (inside.length) {
            check(`${name} ${line.half || ''}: end -- bearing -- bearing -- end, adding up to the total`,
                line.spans.length === inside.length + 1
                && near(line.spans.reduce((sum, s) => sum + s.length, 0), line.total.length)
                && line.spans.slice(1).every((s, i) => near(s.z0, line.spans[i].z1)));
        } else {
            check(`${name} ${line.half || ''}: no bearing inside the ends, no second tier repeating the total`, line.spans.length === 0);
        }
        const highest = Math.max(...parts.map(p => p.offset.y + p.radius));
        check(`${name} ${line.half || ''}: both tiers above everything drawn on the line, the total outermost`,
            line.total.y > highest && line.spans.every(s => s.y > highest && s.y < line.total.y));
        check(`${name} ${line.half || ''}: in the plane of its own shaft line`, line.x === rings[0].x);
    });
    check(`${name}: a number over every node`, plan.nodes.length === layout.rings.length
        && plan.nodes.every(n => layout.rings.some(r => r.n === n.n && r.half === n.half)));
    check(`${name}: the box the camera takes in reaches over the dimensions`, plan.top > Math.max(...plan.lines.map(l => l.total.y)));
}

const every = annotationLayout(layoutScene(CASES.every_element.scene));
check('a bearing hanging under a link node does not cut the line', every.lines[0].spans.length === 0
    && CASES.every_element.scene.bearings.some(b => b.link));
const compressor = annotationLayout(layoutScene(CASES.compressor_example.scene)).lines[0];
check('the compressor: two overhangs and the span between its bearings',
    compressor.spans.length === 3 && near(compressor.spans[0].length, 0.2355) && near(compressor.spans[1].length, 1.19));

// --- the labels --------------------------------------------------------------------------
console.log('\nThe numbers');

const labels = annotationLabels(annotationLayout(layoutScene(CASES.compressor_example.scene)));
const dims = labels.filter(l => l.kind === 'dimension');
check('the total reads as one, in mm, above its line, always shown', dims[0].text === 'L = 1653.3 mm' && dims[0].rank === 2 && !dims[0].below);
check('each span its own length, below its line, before any node', dims.slice(1).map(d => d.text).join() === '235.5 mm,1190.0 mm,227.8 mm'
    && dims.slice(1).every(d => d.below && d.rank === 1));
const nodes = labels.filter(l => l.kind === 'node');
check('a number per node, the first and the last always kept', nodes.length === CASES.compressor_example.scene.nodes.length
    && nodes[0].rank === 2 && nodes[nodes.length - 1].rank === 2 && nodes.filter(n => n.rank === 2).length === 2);

// Labels on a line across the screen, 6 px apart: far closer than they are wide.
const crowd = [
    { kind: 'dimension', text: 'L = 1000.0 mm', position: [0, 0, 50], rank: 2, below: false },
    ...Array.from({ length: 30 }, (_, k) => ({ kind: 'node', text: String(k), position: [0, 100, k * 6], rank: k === 0 || k === 29 ? 2 : 0, below: false })),
];
const flat = ([x, y, z]) => [z + 20, y + 50, 0];
const placed = arrangeLabels(crowd, flat, 800, 600);
const shown = placed.filter(p => p.shown).length;
check('numbers that would print over each other are left out', shown > 3 && shown < 31);
check('never the first and last node, never the total', placed[0].shown && placed[1].shown && placed[30].shown);

// Two tiers a few pixels apart, seen from afar: the total above its line,
// the span below its own -- they no longer print over each other.
const tiers = [
    { kind: 'dimension', text: 'L = 5400.0 mm', position: [0, 10, 0], rank: 2, below: false },
    { kind: 'dimension', text: '5100.0 mm', position: [0, 16, 0], rank: 1, below: true },
];
const twoTiers = arrangeLabels(tiers, ([x, y, z]) => [400, y, 0], 800, 600);
check('the total above its line and the span below its own both show, though their lines are 6 px apart',
    twoTiers.every(p => p.shown));
const clash = arrangeLabels([tiers[0], { ...tiers[1], below: false }], ([x, y, z]) => [400, y, 0], 800, 600);
check('(both above, they would clash, and the span gives way)', clash[0].shown && !clash[1].shown);
const boxes = crowd.map((l, k) => ({ k, x: placed[k].x, w: 7 * l.text.length + 8 })).filter(b => placed[b.k].shown && crowd[b.k].kind === 'node');
check('and no two of the numbers shown overlap',
    boxes.every((a, i) => boxes.every((b, j) => j <= i || a.x + a.w / 2 <= b.x - b.w / 2 || b.x + b.w / 2 <= a.x - a.w / 2)));
const zoomed = arrangeLabels(crowd, ([x, y, z]) => [z * 10 + 20, y + 50, 0], 2000, 600);
check('zooming in brings them back', zoomed.filter(p => p.shown).length > shown);
const behind = arrangeLabels(crowd, ([x, y, z]) => [z + 20, y + 50, 1.5], 800, 600);
check('behind the camera, or out of the view, nothing is shown -- not even what is kept',
    behind.every(p => !p.shown) && arrangeLabels(crowd, ([x, y, z]) => [z + 2000, 50, 0], 800, 600).every(p => !p.shown));

// --- the lines ------------------------------------------------------------------------------
console.log('\nThe lines in the scene');

const plan = annotationLayout(layoutScene(CASES.ross_multirotor_example.scene));
const points = annotationSegments(plan);
check('segments come in pairs, each in the plane of its own line',
    points.length % 2 === 0 && points.every(p => plan.lines.some(l => near(p[0], l.x))));
const built = buildAnnotations(THREE, plan, 'gray');
check('one set of segments for all of it', built.object.isLineSegments && built.object.geometry.attributes.position.count === points.length);
let freed = 0;
built.object.geometry.addEventListener('dispose', () => { freed += 1; });
built.object.material.addEventListener('dispose', () => { freed += 1; });
built.restyle('red');
check('a theme change repaints it where it is', built.object.material.color.getHexString() === 'ff0000');
built.dispose();
check('taking it away frees it', freed === 2);

const layer = node('labels-layer');
const children = [];
layer.innerHTML = '';
Object.defineProperty(layer, 'children', { get: () => children, configurable: true });
children.push(...built.labels.map(() => ({ textContent: '', style: {} })));
const spans = fillLabels(layer, built.labels);
check('each label its own element, its text set as text, not markup',
    spans.length === built.labels.length && spans.every((s, k) => s.textContent === built.labels[k].text) && !/L = /.test(layer.innerHTML));

// --- the button -----------------------------------------------------------------------------
console.log('\nThe button');

const button = node('dimensions-button');
registerSelector('[data-action="toggle-dimensions"]', [button]);
const { toggleDimensions } = await import('../../frontend/features/rotor3d.js');
const { MODELING_ACTIONS } = await import('../../frontend/features/modeling_actions.js');
toggleDimensions();
check('it turns them on, says so, and is remembered', button.getAttribute('aria-pressed') === 'true' && disk.content['ross-rotor-dimensions'] === 'on');
MODELING_ACTIONS['toggle-dimensions']();
check('and off again', button.getAttribute('aria-pressed') === 'false' && disk.content['ross-rotor-dimensions'] === 'off');

shutDown();
