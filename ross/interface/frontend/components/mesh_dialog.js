// The dialog of the ruler button (features/mesh.js): the three ways of
// discretizing the shafts, and what each needs.
//
//   * by the L/D ratio (API RP 684, 1.5.2.1): no element longer than so many
//     diameters -- 0.5 preferred, 1.0 at most;
//   * in a number of equal parts per element, whatever its ratio;
//   * by the convergence of the lowest natural frequencies: the coarsest mesh
//     that refining no longer changes (ROSS's `refine_mesh_by_convergence`).
//
// The first two act on the shafts ticked in the list, or on all of them; the
// convergence always on the whole rotor -- it is the whole model's frequencies
// that converge or not -- and the dialog says so.
//
// Opening it answers a Promise with what was asked, or null. The dialog stays
// open, and says it is computing, until the caller closes it: a convergence
// takes a modal analysis per mesh, and a dialog that vanished on "Compute"
// would leave the person looking at nothing for those seconds.
import { t } from '../core/i18n.js';

export const MESH_METHODS = ['ratio', 'parts', 'convergence'];

let answer = null;
let busy = false;

function byId(id) {
    return document.getElementById(id);
}

function fill(template, ...values) {
    // One pass over the template: a value that itself holds "%2" is shown as
    // typed, rather than filled in by the next placeholder.
    return template.replace(/%(\d)/g, (marker, digit) =>
        (digit >= 1 && digit <= values.length ? String(values[digit - 1]) : marker));
}

export function chosenMethod() {
    const checked = Array.from(document.querySelectorAll('input[name="mesh-method"]')).find(radio => radio.checked);
    return checked && MESH_METHODS.includes(checked.value) ? checked.value : 'ratio';
}

// Only the fields of the way chosen are shown.
export function showMethodFields() {
    const method = chosenMethod();
    document.querySelectorAll('.mesh-fields').forEach(block => {
        block.hidden = block.dataset.method !== method;
    });
}

// `ticked`: how many shafts are ticked in the list, 0 for none.
export function openMeshDialog(ticked) {
    return new Promise(resolve => {
        answer = resolve;
        byId('mesh-scope').innerText = ticked ? fill(t('meshScopePicked'), ticked) : t('meshScopeAll');
        setMeshBusy(false);
        showMethodFields();
        byId('mesh-overlay').style.display = 'flex';
    });
}

// What was typed, as the server reads it: text, with a decimal comma read as
// the point it means.
export function meshDialogValues() {
    const read = id => String(byId(id).value || '').trim().replace(',', '.');
    return {
        method: chosenMethod(),
        max_ld: read('mesh-max-ld'),
        parts: read('mesh-parts'),
        n_modes: read('mesh-modes'),
        rtol: read('mesh-rtol'),
        speed: read('mesh-speed'),
    };
}

export function submitMeshDialog() {
    if (busy || !answer) return;
    const resolve = answer;
    answer = null;
    resolve(meshDialogValues());
}

export function cancelMeshDialog() {
    if (busy) return;
    closeMeshDialog();
    if (answer) {
        const resolve = answer;
        answer = null;
        resolve(null);
    }
}

// While the server computes: the line that says so, and nothing to press.
export function setMeshBusy(on, text = '') {
    busy = !!on;
    const line = byId('mesh-busy');
    line.innerText = on ? text : '';
    line.hidden = !on;
    document.querySelectorAll('#mesh-overlay button, #mesh-overlay input').forEach(control => {
        control.disabled = busy;
    });
}

export function closeMeshDialog() {
    setMeshBusy(false);
    byId('mesh-overlay').style.display = 'none';
}
