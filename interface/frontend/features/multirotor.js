// Coupled rotors: the modal that picks the two of them, and switching which one
// is being edited.
import { renderList } from '../components/list.js';
import { openCustomAlert } from '../components/modals.js';
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

export async function openMultiRotorModal() {
    if (state.rotorLibrary.length < 2) {
        await openCustomAlert(t('needTwoRotors'));
        return;
    }
    let options = '';
    state.rotorLibrary.forEach((r, i) => {
        options += `<option value="${i}">${r.name}</option>`;
    });
    document.getElementById('mr-driving').innerHTML = options;
    document.getElementById('mr-driven').innerHTML = options;
    
    if (state.rotorLibrary.length > 1) {
        document.getElementById('mr-driven').selectedIndex = 1;
    }
    
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
    
    state.rotorLibrary.push(mrData);
    closeMultiRotorModal();
    renderRotorHub();
}
