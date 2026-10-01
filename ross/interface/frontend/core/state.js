import { recordChange, resetHistory, structuralSnapshot } from './history.js';
import { clearSelection } from './selection.js';

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

// Called by whoever changes the open project. It has one subscriber -- the two
// history buttons, which have to grey out the moment there is nothing left to
// undo -- and it is a hook rather than a direct call because this module has no
// business touching the DOM. Same shape as `onReorder` in components/list.js.
//
// A **list** of subscribers, not one.
//
// The first version held a single function, and that is a trap with a delay on
// it: subscribing twice silently threw the first one away, and the symptom
// would have appeared in whichever feature happened to subscribe first. There
// are two now -- the history buttons and the multiple selection -- and there is
// no reason for a third to be harder than the second.
//
// No throw when nobody subscribes, unlike `onReorder`: this fires on every
// mutation, including inside the node batteries, where nobody has any reason to.
let changeHandlers = [];

export function onProjectChanged(fn) {
    changeHandlers.push(fn);
}

// The project changed. Everything that stops being true when it does gets
// dealt with here, and nothing anywhere else.
//
// The selection is dropped **in this function** rather than subscribed to the
// hook, and the difference matters. The hook exists for the things that touch
// the page -- core/state.js has no business doing that, so it announces and the
// features listen. A set of positions is not one of those: it is state, like
// the undo stack, and `recordChange` is called straight from the funnel for the
// same reason.
//
// It was a subscriber first, and the guard that keeps hooks connected
// (`test_every_hook_a_component_offers_is_registered`) would not have noticed
// it going missing, because it only asks whether a hook has *a* subscriber --
// and the history buttons are already one. A selection that outlived a deletion
// would then delete the wrong elements and leave a shorter list, which looks
// exactly like a list shortened on purpose. Here, forgetting is not available.
export function projectChanged() {
    clearSelection();
    changeHandlers.forEach(handler => handler());
}

// Which list the modelling screen is showing: the tab, and for a MultiRotor
// which of the two rotors. The multiple selection carries this around so that a
// set of indices taken on one list cannot be read as a set of indices on
// another -- see core/selection.js.
export function listContext() {
    const half = state.projectData && state.projectData.isMultiRotor
        ? state.multiRotorEditTarget
        : '';
    return String(state.currentTab) + '/' + half;
}

// Writes the open project back to the library, records the step for undo, and
// tells whoever is listening.
//
// The write-back only does something for a MultiRotor: for a plain rotor
// `state.projectData` **is** the library entry (`openRotorWorkspace` assigns
// the object itself, not a copy), so there is nothing to copy across.
//
// The history is recorded here because this is the one call every mutation of
// the modelling screen already makes -- see the comment at the top of
// core/history.js for why that matters more than it looks.
export function syncBackToLibrary() {
    writeBackToLibrary();
    recordChange(structuralSnapshot(state.projectData));
    projectChanged();
}

// A rotor was opened: its history starts empty, holding this model as the state
// the first change will come back to.
//
// It lives here, beside `syncBackToLibrary`, and not in the Hub calling
// `resetHistory` directly -- which is what it did, and it was a bug the node
// battery could not see. Emptying the stacks is a change to the history exactly
// like recording a step is, and the buttons have to hear about both; the Hub
// emptied them and told nobody, so the buttons kept the state they had under
// the *previous* rotor and undo arrived alive in a model with no history.
//
// Putting both behind this module is what makes that impossible rather than
// merely fixed: there is no way to change the history without announcing it.
export function openProjectHistory(project) {
    resetHistory(structuralSnapshot(project));
    projectChanged();
}

// The same write-back with no history and no notification, for undo and redo:
// they put a snapshot on screen, and a restore that recorded itself would push
// what it just undid back onto the stack.
export function writeBackToLibrary() {
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
