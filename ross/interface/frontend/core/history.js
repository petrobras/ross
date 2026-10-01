// Undo and redo for the modelling screen: two stacks and a present.
//
// WHY THIS MODULE KNOWS NOTHING ABOUT THE SCREEN OR THE STATE. It holds three
// values and four verbs, touches no DOM and imports nothing. That is what lets
// the whole of it be checked without a browser -- and the part worth checking
// is not "does the button work", it is the bookkeeping: that redo dies the
// moment you do something new, that undoing twice and redoing twice lands
// exactly where you started, that a bounded stack drops the oldest and not the
// newest.
//
// WHY THE SNAPSHOT IS TAKEN AT `syncBackToLibrary` AND NOT AT EACH MUTATION.
// Five places on the modelling screen change the project -- copy, delete, save,
// split, and dragging in the list -- and all five already call
// `syncBackToLibrary`, because without it the library goes stale, which is a
// bug you can see. Hooking the history there means the sixth mutation somebody
// writes next year joins the undo on its own, and the one way to leave it out
// is to write a mutation that is already broken for another reason.
//
// The call arrives *after* the change, so what it is handed is the new state.
// That is why `present` exists: the stack is fed the state as it was before
// this change, which is the one this module was already holding.

// Fifty steps of a model that runs about a kilobyte. Deep enough that nobody
// reaches the end by accident, shallow enough that the whole history is smaller
// than a single Campbell chart.
export const HISTORY_LIMIT = 50;

// Keys that describe the screen's copy of a rotor rather than the rotor. The
// analyses are out by decision: they are the heavy half (one Campbell figure
// runs past 15 KB) and they are *already* stale after any edit, so restoring
// yesterday's chart onto today's model would be worse than leaving it alone.
const NOT_THE_MODEL = ["name", "uid", "savedAnalyses"];

// The two halves of a MultiRotor are projects in their own right, so the same
// rule has to reach inside them.
const NESTED = ["driving_rotor", "driven_rotor"];

let past = [];
let future = [];
let present = null;

export function structuralSnapshot(project) {
    if (!project || typeof project !== "object") return {};
    const copy = {};
    Object.keys(project).forEach(key => {
        if (NOT_THE_MODEL.indexOf(key) !== -1) return;
        copy[key] = JSON.parse(JSON.stringify(project[key]));
    });
    NESTED.forEach(key => {
        if (copy[key]) copy[key] = structuralSnapshot(project[key]);
    });
    return copy;
}

// Write a snapshot back onto a live project. What the snapshot does not carry
// -- the name, the identity, the computed charts -- is left exactly as it is.
export function applySnapshot(target, snapshot) {
    Object.keys(snapshot || {}).forEach(key => {
        target[key] = JSON.parse(JSON.stringify(snapshot[key]));
    });
    return target;
}

// The project changed, and `snapshot` is what it looks like now.
export function recordChange(snapshot) {
    if (present !== null) {
        past.push(present);
        // The oldest goes, never the newest: a stack that dropped the most
        // recent step would make undo do nothing right after a change, which
        // reads as a broken button rather than as a full history.
        if (past.length > HISTORY_LIMIT) past.shift();
    }
    present = snapshot;
    // Doing something new makes the abandoned branch unreachable. Keeping it
    // would let redo paste a model that never followed from what is on screen.
    future = [];
}

// Entering a rotor, or starting up. The history belongs to one open model:
// undoing across that boundary would restore one rotor over another.
export function resetHistory(snapshot) {
    past = [];
    future = [];
    present = snapshot === undefined ? null : snapshot;
}

export function canUndo() {
    return past.length > 0;
}

export function canRedo() {
    return future.length > 0;
}

// Both return the snapshot to put on screen, or `null` when there is nowhere to
// go. `null` and not a throw: pressing Ctrl+Z one time too many is the most
// ordinary thing a person does.
export function undo() {
    if (!past.length) return null;
    future.push(present);
    present = past.pop();
    return present;
}

export function redo() {
    if (!future.length) return null;
    past.push(present);
    present = future.pop();
    return present;
}
