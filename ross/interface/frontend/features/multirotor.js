// Coupled rotors: the modal that picks the two of them, and switching which one
// is being edited.
import { getEffectiveNodes, renderList } from '../components/list.js';
import { openCustomAlert } from '../components/modals.js';
import { apiFetch, projectForServer } from '../core/api.js';
import { escapeHtml } from '../core/dom.js';
import { ensureUIDs, state } from '../core/state.js';
import { renderRotorHub } from './hub.js';
import { closeForm } from './modeling.js';
import { t } from '../core/i18n.js';
// Toggles editing between the Driving and Driven rotors
export const switchMultiRotorTarget = function(target) {
    state.multiRotorEditTarget = target;
    closeForm();
    renderList();
};

// ==========================================
// MULTIROTOR LOGIC
// ==========================================

// --- the gears, which are what a MultiRotor is coupled through -----------------
//
// `rs.MultiRotor` couples the two rotors through a **gear on each coupled
// node**, and refuses anything else: measured, with no gear, or with gears on
// nodes other than the ones typed, it raises "Each rotor needs a GearElement in
// the coupled nodes!". The modal used to offer "0, 0" and say nothing, so the
// MultiRotor was saved without complaint and only failed later, on the
// modelling screen.
//
// So the modal now says where each rotor's gears are, and fills the coupled
// nodes in when there is only one way to fill them. The *rule* is not copied
// here: `saveMultiRotor` asks the server to build the MultiRotor before keeping
// it, and ROSS is the one that answers.

// The nodes the gears of a rotor sit on, numbered the way the list numbers them
// -- a gear with no node typed takes the next free one, like everything else.
// `null` for a MultiRotor chosen as one of the two: its nodes are the merged
// numbering of two rotors, which this sentence cannot state honestly.
export function gearNodes(rotor) {
    if (!rotor || rotor.isMultiRotor) return null;
    return getEffectiveNodes(Array.isArray(rotor.gears) ? rotor.gears : []);
}

function chosenRotor(id) {
    const node = document.getElementById(id);
    return node ? state.rotorLibrary[Number(node.value)] : undefined;
}

function gearSentence(rotor) {
    const name = `<b>${escapeHtml(rotor.name || '')}</b>`;
    const nodes = gearNodes(rotor);
    if (nodes === null) return escapeHtml(t('multiGearsUnknown')).replace('%1', () => name);
    if (!nodes.length) return escapeHtml(t('multiNoGear')).replace('%1', () => name);
    // `%2` first: a rotor named `R%2` would otherwise have its name filled in.
    return escapeHtml(t('multiGearNodes'))
        .replace('%2', () => nodes.join(', '))
        .replace('%1', () => name);
}

// Runs when the modal opens and on every change of either select.
export function describeCoupling() {
    const hint = document.getElementById('mr-hint');
    if (!hint) return;
    const driving = chosenRotor('mr-driving');
    const driven = chosenRotor('mr-driven');
    if (!driving || !driven || driving === driven) {
        hint.innerHTML = '';
        return;
    }
    hint.innerHTML = gearSentence(driving) + '<br>' + gearSentence(driven);

    // One gear on each side: there is exactly one coupling ROSS will accept, so
    // it is typed in. Anything else is the person's choice to make.
    const first = gearNodes(driving);
    const second = gearNodes(driven);
    if (first && second && first.length === 1 && second.length === 1) {
        document.getElementById('mr-coupled-nodes').value = first[0] + ', ' + second[0];
    }
}

export async function openMultiRotorModal() {
    if (state.rotorLibrary.length < 2) {
        await openCustomAlert(t('needTwoRotors'));
        return;
    }
    let options = '';
    state.rotorLibrary.forEach((r, i) => {
        // Escaped: a rotor's name is whatever was typed into it.
        options += `<option value="${i}">${escapeHtml(r.name || '')}</option>`;
    });
    document.getElementById('mr-driving').innerHTML = options;
    document.getElementById('mr-driven').innerHTML = options;

    // `value` and not `selectedIndex`, as in the concatenation modal: the two
    // are the same in a browser, and only `value` is something the node
    // batteries can see. With `selectedIndex` both selects read as the first
    // rotor there, and the "the same rotor twice" path was the only one tested.
    document.getElementById('mr-driving').value = '0';
    document.getElementById('mr-driven').value = '1';
    document.getElementById('mr-coupled-nodes').value = '0, 0';
    describeCoupling();

    document.getElementById('multirotor-modal-overlay').style.display = 'flex';
}

export function closeMultiRotorModal() {
    document.getElementById('multirotor-modal-overlay').style.display = 'none';
}

export async function saveMultiRotor() {
    let drivingIdx = document.getElementById('mr-driving').value;
    let drivenIdx = document.getElementById('mr-driven').value;
    
    if (drivingIdx === drivenIdx) {
        await openCustomAlert(t('sameRotorTwice'));
        return;
    }
    
    ensureUIDs();
    let drvRotor = state.rotorLibrary[drivingIdx];
    let drvnRotor = state.rotorLibrary[drivenIdx];
    
    let mrData = {
        name: document.getElementById('mr-name').value || "MultiRotor",
        isMultiRotor: true,
        uid: 'rotor_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9),
        driving_uid: drvRotor.uid,
        driven_uid: drvnRotor.uid,
        savedAnalyses: [],
        driving_rotor: JSON.parse(JSON.stringify(drvRotor)),
        driven_rotor: JSON.parse(JSON.stringify(drvnRotor)),
        multi_params: {
            coupled_nodes: document.getElementById('mr-coupled-nodes').value,
            gear_mesh_stiffness: document.getElementById('mr-stiffness').value,
            update_mesh_stiffness: document.getElementById('mr-update-stiffness').value,
            square_varying_stiffness: {
                enable: document.getElementById('mr-square-stiffness-enable').value === 'true',
                amplitude_ratio: parseFloat(document.getElementById('mr-square-stiffness-ratio').value) || 0
            },
            backlash: {
                enable: document.getElementById('mr-backlash-enable').value === 'true',
                initial_value: parseFloat(document.getElementById('mr-backlash-initial').value) || 0.0,
                error_amp: parseFloat(document.getElementById('mr-backlash-error').value) || 0.0,
                smooth_operator: document.getElementById('mr-backlash-smooth').value === 'true',
                sigma: parseFloat(document.getElementById('mr-backlash-sigma').value) || 1e4
            },
            orientation_angle: document.getElementById('mr-angle').value,
            position: document.getElementById('mr-position').value
        },
        materials: [], shafts: [], disks: [], gears: [], couplings: [], seals: [], bearings: [], pointmasses: []
    };
    
    // Built once by the server before it is kept. A MultiRotor that ROSS will
    // not build -- no gear on a coupled node, a mesh stiffness it cannot work
    // out -- used to be saved anyway and fail on the modelling screen, far from
    // the fields that caused it. Now ROSS's own sentence is shown here, with
    // the modal still open on those fields.
    let answer;
    try {
        answer = await apiFetch('/build_rotor', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project: projectForServer(mrData) }),
        });
    } catch (error) {
        await openCustomAlert(t('multiCheckFailed'));
        return;
    }
    const body = await answer.json();
    if (body.status !== 'success') {
        await openCustomAlert(body.message || t('multiCheckFailed'));
        return;
    }

    state.rotorLibrary.push(mrData);
    closeMultiRotorModal();
    renderRotorHub();
}
