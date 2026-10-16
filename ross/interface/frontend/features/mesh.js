// Discretizing the shafts: the ruler button of the shafts' list. The person
// draws the shaft by its sections -- one row per step of diameter -- and this
// cuts them into the elements the model needs, one of three ways
// (components/mesh_dialog.js): by the L/D rule of API RP 684, in a number of
// parts per element, or by the convergence of the natural frequencies.
//
// The server answers the cut project AND the plan (domain/meshing.py), and the
// plan is shown before the project is taken on: how many elements each shaft
// becomes, what is too short to fix, what a coupling keeps whole -- and for
// the convergence, the meshes it analysed and the one it chose. Nothing
// changes until the person says yes; after, Undo takes it back like any other
// edit.
import { openCustomAlert, openCustomConfirm } from '../components/modals.js';
import { closeMeshDialog, openMeshDialog, setMeshBusy } from '../components/mesh_dialog.js';
import { apiFetch } from '../core/api.js';
import { t } from '../core/i18n.js';
import { applyProject, withoutAnalyses } from './split.js';

// A plan with more lines than this ends in "..." -- a rotor of fifty shafts
// cut would otherwise be a dialog taller than the screen.
const MOST_LINES = 8;

function fill(template, ...values) {
    // One pass over the template: a value that itself holds "%2" is shown as
    // typed, rather than filled in by the next placeholder.
    return template.replace(/%(\d)/g, (marker, digit) =>
        (digit >= 1 && digit <= values.length ? String(values[digit - 1]) : marker));
}

// "#2 Body" for the rows of a span, as the list numbers them (from 1).
function rowsName(rows, shafts) {
    return rows.map(index => {
        const tag = String((shafts[index] || {}).tag || '').trim();
        return tag ? `#${index + 1} ${tag}` : `#${index + 1}`;
    }).join(' + ');
}

const decimal = (value, digits) => (typeof value === 'number' && isFinite(value) ? value.toFixed(digits) : '?');

// The meshes the convergence analysed, the one it chose marked.
export function convergenceLines(convergence) {
    const lines = [fill(t('meshConvTitle'), convergence.n_modes, convergence.rtol)];
    convergence.rows.forEach((row, position) => {
        const name = row.max_ld === null ? t('meshConvAsIs') : `L/D ${decimal(row.max_ld, 3).replace(/0+$/, '').replace(/\.$/, '')}`;
        const change = row.change === null ? '' : fill(t('meshConvChange'), decimal(row.change, 3));
        const mark = position === convergence.chosen ? t('meshConvChosen') : '';
        lines.push(`${name}: ${fill(t('meshConvElements'), row.elements)}${change}${mark}`);
    });
    if (convergence.chosen === null) lines.push(t('meshConvNone'));
    return lines;
}

// The plan in words: the convergence first when there was one, then one line
// per span cut, then what could not be done.
export function meshSummary(report, shafts) {
    const lines = report.convergence ? convergenceLines(report.convergence).concat(['']) : [];
    lines.push(fill(t('meshSummary'), report.before, report.after));
    const cut = report.cut || [];
    cut.slice(0, MOST_LINES).forEach(step => {
        lines.push(fill(t('meshCutLine'), rowsName(step.rows, shafts), decimal(step.ratio, 2), step.parts));
    });
    if (cut.length > MOST_LINES) lines.push('...');
    const short = report.short || [];
    if (short.length) {
        lines.push('', t('meshShortTitle'));
        short.slice(0, MOST_LINES).forEach(step => {
            lines.push(fill(t('meshShortLine'), rowsName(step.rows, shafts), decimal(step.ratio, 2)));
        });
        if (short.length > MOST_LINES) lines.push('...');
    }
    const kept = (report.kept || []).filter(step => step.why === 'coupling');
    if (kept.length) lines.push('', fill(t('meshKeptCoupling'), kept.map(step => rowsName(step.rows, shafts)).join(', ')));
    return lines.join('\n');
}

// Answers whether the project changed. `indexes`: the shaft rows ticked in
// the list, or none for every shaft.
export async function meshProject(data, indexes) {
    const shafts = (data && data.shafts) || [];
    if (!shafts.length) return false;
    const ticked = indexes || [];

    const choice = await openMeshDialog(ticked.length);
    if (!choice) return false;
    const converging = choice.method === 'convergence';
    setMeshBusy(true, converging ? t('meshBusyConvergence') : t('meshBusy'));

    let answer;
    let body;
    try {
        answer = await apiFetch('/api/rotor/mesh_shafts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.assign({ project: withoutAnalyses(data), indexes: converging ? [] : ticked }, choice)),
        });
        body = await answer.json();
    } catch (error) {
        closeMeshDialog();
        await openCustomAlert(t('meshFailed'));
        return false;
    }
    closeMeshDialog();
    if (!answer.ok) {
        await openCustomAlert(body.message || t('meshFailed'));
        return false;
    }

    const report = body.report || {};
    if (!(report.cut || []).length) {
        // Nothing to cut -- the rotor had converged, or every element is
        // within the ratio. What was found is still worth reading.
        const words = meshSummary(report, shafts).split('\n');
        const summary = fill(t('meshSummary'), report.before, report.after);
        await openCustomAlert([t('meshNothing')].concat(words.filter(line => line !== summary)).join('\n').trim());
        return false;
    }
    if (!(await openCustomConfirm(meshSummary(report, shafts)))) return false;
    applyProject(data, body.projectData);
    return true;
}
