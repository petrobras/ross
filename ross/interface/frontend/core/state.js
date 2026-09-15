// The six values that cross module boundaries, in one named object.
//
// This is not a state framework and does not want to be: it is the minimum set
// of what several parts read **and write**. They live in an object because
// `export let` is read-only for importers -- and because this way a mutation
// has an address and a search finds it. The goal of the coming slices is for
// this object to shrink.
export const state = {
    rotorLibrary: [],
    activeRotorIndex: -1,
    projectData: { materials: [], shafts: [], disks: [], gears: [], couplings: [],
                   seals: [], bearings: [], pointmasses: [] },
    currentTab: null,
    editingIndex: -1,
    currentSubType: 'BASIC',
    // Which of a MultiRotor's two rotors is being edited. It used to live in
    // `state.multiRotorEditTarget`, read and written by three modules -- a seventh
    // shared value, only without an address.
    multiRotorEditTarget: 'driving',
};


// Returns data for the active rotor on the screen (whether a simple rotor or the selected half of a multi-rotor)
export function getActiveData() {
    if (state.projectData.isMultiRotor) {
        return state.multiRotorEditTarget === 'driven' ? state.projectData.driven_rotor : state.projectData.driving_rotor;
    }
    return state.projectData;
}

// Synchronizes any changes made to the MultiRotor back to the Hub's parent (original) rotors
export function syncBackToLibrary() {
    if (state.projectData.isMultiRotor) {
        let drvLib = state.rotorLibrary.find(r => r.uid === state.projectData.driving_uid);
        if (drvLib) Object.assign(drvLib, JSON.parse(JSON.stringify(state.projectData.driving_rotor)));
        
        let drvnLib = state.rotorLibrary.find(r => r.uid === state.projectData.driven_uid);
        if (drvnLib) Object.assign(drvnLib, JSON.parse(JSON.stringify(state.projectData.driven_rotor)));
    }
}

// Operations on the rotor library, used by the Hub and by the MultiRotor.
// They lived in the MultiRotor module by accident of history: they touch
// `rotorLibrary` and nothing else.
export function ensureUIDs() {
    state.rotorLibrary.forEach(r => {
        if (!r.uid) r.uid = 'rotor_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    });
}

export function syncMultiRotors() {
    state.rotorLibrary.forEach(mr => {
        if (mr.isMultiRotor) {
            let drv = state.rotorLibrary.find(r => r.uid === mr.driving_uid);
            if (drv) mr.driving_rotor = JSON.parse(JSON.stringify(drv));
            
            let drvn = state.rotorLibrary.find(r => r.uid === mr.driven_uid);
            if (drvn) mr.driven_rotor = JSON.parse(JSON.stringify(drvn));
        }
    });
}
