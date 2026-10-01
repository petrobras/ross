// Undo and redo: the bookkeeping, and the one snapshot rule.
//
// WHY THIS BATTERY EXISTS. An undo that mostly works is worse than none: the
// person stops checking, and the one time the stack is wrong they lose a model
// they thought was safe. What can go wrong here is not the button, it is the
// bookkeeping -- a redo that survives a new change and pastes a model that
// never followed from what is on screen; a bounded stack that drops the newest
// step instead of the oldest; a restore that records itself and so undoes the
// same step forever.
//
// `core/history.js` imports nothing and touches no DOM precisely so that all of
// that can be checked here, exactly.
import { check, node, shutDown } from './fake_dom.js';

const { HISTORY_LIMIT, applySnapshot, canRedo, canUndo, recordChange,
    redo, resetHistory, structuralSnapshot, undo } =
    await import('../../frontend/core/history.js');
const { onProjectChanged, state, syncBackToLibrary } =
    await import('../../frontend/core/state.js');

function rotor(shafts) {
    return {
        name: 'Compressor',
        uid: 'uid_compressor',
        savedAnalyses: [{ type: 'campbell', figure: 'a heavy plotly figure' }],
        materials: [{ name: 'Steel' }],
        shafts: shafts.map(L => ({ element_type: 'BASIC', L: String(L), odl: '50' })),
        disks: [],
        gears: [],
        couplings: [],
        seals: [],
        bearings: [{ element_type: 'BASIC', n: '0', kxx: '1e6' }],
        pointmasses: [],
    };
}

// --- what a snapshot is, and is not -------------------------------------------

const snap = structuralSnapshot(rotor([100, 200]));
check('the model is in the snapshot', snap.shafts.length === 2);
check('and so are the materials', snap.materials.length === 1);
// The analyses are the heavy half and they are stale after any edit anyway:
// restoring yesterday's chart onto today's model would be worse than leaving it.
check('the computed charts are not', snap.savedAnalyses === undefined);
check('nor the name', snap.name === undefined);
check('nor the identity', snap.uid === undefined);

const source = rotor([100]);
const copied = structuralSnapshot(source);
copied.shafts[0].L = '999';
check('the snapshot is a copy, not a view', source.shafts[0].L === '100');

// A MultiRotor is two projects, and the rule has to reach inside both.
const pair = { isMultiRotor: true, name: 'Train', driving_rotor: rotor([100]), driven_rotor: rotor([200]) };
const both = structuralSnapshot(pair);
check('both halves of a multirotor are snapshotted', both.driving_rotor.shafts.length === 1 && both.driven_rotor.shafts.length === 1);
check('and the charts are dropped inside them too', both.driving_rotor.savedAnalyses === undefined);

// Restoring leaves alone what the snapshot does not carry.
const live = rotor([100, 200, 300]);
applySnapshot(live, structuralSnapshot(rotor([50])));
check('the model was replaced', live.shafts.length === 1 && live.shafts[0].L === '50');
check('the charts were left where they were', live.savedAnalyses.length === 1);
check('and so was the name', live.name === 'Compressor');

// --- the stack ------------------------------------------------------------------

resetHistory(structuralSnapshot(rotor([1])));
check('a fresh history has nowhere to go back to', canUndo() === false);
check('and nowhere to go forward to', canRedo() === false);

recordChange(structuralSnapshot(rotor([1, 2])));
check('one change makes undo possible', canUndo() === true);
check('but not redo', canRedo() === false);

const back = undo();
check('undo gives back the state before the change', back.shafts.length === 1);
check('and now there is somewhere to go forward to', canRedo() === true);
check('and nowhere further back', canUndo() === false);

const forward = redo();
check('redo gives back the state after it', forward.shafts.length === 2);
check('and undo is possible again', canUndo() === true);

// Control: two steps out and two steps back land exactly where they started.
resetHistory(structuralSnapshot(rotor([1])));
recordChange(structuralSnapshot(rotor([1, 2])));
recordChange(structuralSnapshot(rotor([1, 2, 3])));
undo();
undo();
redo();
const there = redo();
check('two out and two back is where it started', JSON.stringify(there) === JSON.stringify(structuralSnapshot(rotor([1, 2, 3]))));

// --- redo dies when a new branch starts ----------------------------------------
//
// The one that would be silently wrong. After undoing, doing something new
// makes the abandoned branch unreachable; a redo that survived would paste a
// model that never followed from what is on screen.

resetHistory(structuralSnapshot(rotor([1])));
recordChange(structuralSnapshot(rotor([1, 2])));
undo();
check('after an undo there is a redo waiting', canRedo() === true);
recordChange(structuralSnapshot(rotor([1, 9])));
check('and doing something new throws it away', canRedo() === false);
check('while the new step is still undoable', canUndo() === true);

// --- too many out --------------------------------------------------------------

resetHistory(structuralSnapshot(rotor([1])));
check('undoing an empty history answers null rather than throwing', undo() === null);
check('and so does redoing one', redo() === null);

// --- the bound ------------------------------------------------------------------

resetHistory(structuralSnapshot(rotor([0])));
for (let step = 1; step <= HISTORY_LIMIT + 10; step += 1) {
    recordChange(structuralSnapshot(rotor(Array.from({ length: step }, (_, i) => i))));
}
// Walked rather than read off a counter: how many steps back the person can
// actually take is the thing that matters, and a counter can be right while the
// walk is wrong.
//
// Which end gets dropped is the whole question. A stack that lost the newest
// step would make undo do nothing right after a change, which reads as a broken
// button rather than as a full history -- so the first step back has to be the
// one just before the last change, not something from the beginning.
const firstBack = undo();
check('the first step back is the state before the last change',
    firstBack.shafts.length === HISTORY_LIMIT + 9);

let steps = 1;
while (undo() !== null) steps += 1;
check('and the walk stops at the limit', steps === HISTORY_LIMIT);

// --- the funnel -----------------------------------------------------------------
//
// The reason the history hangs off `syncBackToLibrary` and not off each of the
// five mutation sites: every one of them already calls it, and one that forgot
// would be broken for another reason first. This checks the wiring, not the
// stack.

let announced = 0;
onProjectChanged(() => { announced += 1; });

state.rotorLibrary = [rotor([100, 200])];
state.activeRotorIndex = 0;
state.projectData = state.rotorLibrary[0];
resetHistory(structuralSnapshot(state.projectData));

state.projectData.shafts.push({ element_type: 'BASIC', L: '300', odl: '50' });
syncBackToLibrary();

check('a change that syncs is a change the history saw', canUndo() === true);
check('and it told whoever is listening', announced === 1);

const before = undo();
check('undoing it gives back the two-shaft model', before.shafts.length === 2);
// The restore itself must not record: a restore that recorded would push back
// onto the stack the very step it just took off, and undo would never finish.
check('and the undo did not record itself as a step', canUndo() === false);

// --- the buttons -----------------------------------------------------------------

const { refreshHistoryButtons } = await import('../../frontend/features/modeling.js');

resetHistory(structuralSnapshot(rotor([1])));
refreshHistoryButtons();
check('with nothing to undo the back button is dead', node('btn-undo').disabled === true);
check('and so is the forward one', node('btn-redo').disabled === true);

recordChange(structuralSnapshot(rotor([1, 2])));
refreshHistoryButtons();
check('one change brings the back button to life', node('btn-undo').disabled === false);
check('and leaves the forward one dead', node('btn-redo').disabled === true);

undo();
refreshHistoryButtons();
check('undoing brings the forward one to life', node('btn-redo').disabled === false);
check('and puts the back one to sleep', node('btn-undo').disabled === true);

// --- the whole action, and not only the stack ------------------------------------
//
// Everything above drives `core/history.js` directly. That left one thing
// unchecked, and a mutation found it: `restore` in features/modeling.js must
// call `writeBackToLibrary` and not `syncBackToLibrary`, because the second one
// records -- and a restore that records pushes back onto the stack the very
// step it just took off, which wipes the redo and makes undo walk in place.
//
// The tell is simple and it is what the person sees: after an undo, redo has to
// be possible.
const { undoModelling } = await import('../../frontend/features/modeling.js');

globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ status: 'success', figure: { data: [], layout: {} } }),
});

state.rotorLibrary = [rotor([100, 200])];
state.activeRotorIndex = 0;
state.projectData = state.rotorLibrary[0];
state.currentTab = 'shafts';
resetHistory(structuralSnapshot(state.projectData));

state.projectData.shafts.push({ element_type: 'BASIC', L: '300', odl: '50' });
syncBackToLibrary();
check('the change is on the stack', canUndo() === true);

undoModelling();
check('the action put the old model on screen', state.projectData.shafts.length === 2);
check('and redo is possible, which is what says the restore did not record itself',
    canRedo() === true);
check('and there is nothing further back', canUndo() === false);

// --- opening a rotor ---------------------------------------------------------------
//
// The bug this exists for, found by hand and not by this battery: entering
// another rotor emptied the stacks and told nobody, so the buttons kept the
// state they had under the previous model -- undo arrived alive in a rotor with
// no history to walk.
//
// The check is deliberately *not* calling `refreshHistoryButtons` itself. That
// is what the first version did everywhere above, and it is exactly why the
// defect was invisible: a battery that refreshes the buttons by hand cannot
// notice that nobody else does.
const { openProjectHistory } = await import('../../frontend/core/state.js');

// What main.js does at boot. The section above replaced the subscriber with a
// counter, and the first run of this test failed for that reason rather than
// for the defect -- which is the right way round: the battery has to be wired
// like the application before it can say anything about the application.
onProjectChanged(refreshHistoryButtons);

resetHistory(structuralSnapshot(rotor([1])));
recordChange(structuralSnapshot(rotor([1, 2])));
refreshHistoryButtons();
check('setting the scene: undo is alive in the rotor being left',
    node('btn-undo').disabled === false);

openProjectHistory(rotor([500]));

check('entering another rotor puts the back button to sleep, with nobody asking',
    node('btn-undo').disabled === true);
check('and the forward one too', node('btn-redo').disabled === true);
check('and the history really is empty, not just the buttons', canUndo() === false);

// --- the shortcut ----------------------------------------------------------------
//
// Read as a pure function of the event, so the three guards around it -- a
// focused field, an open dialog, another screen -- can be argued about
// separately from which keys mean what.

const { historyShortcut } = await import('../../frontend/features/shortcuts.js');

check('Ctrl+Z undoes', historyShortcut({ ctrlKey: true, key: 'z' }) === 'undo');
check('Ctrl+Y redoes', historyShortcut({ ctrlKey: true, key: 'y' }) === 'redo');
check('Ctrl+Shift+Z redoes too, because half the editors spell it that way',
    historyShortcut({ ctrlKey: true, shiftKey: true, key: 'z' }) === 'redo');
check('Cmd+Z works for a Mac', historyShortcut({ metaKey: true, key: 'z' }) === 'undo');
check('an upper-case Z is the same key', historyShortcut({ ctrlKey: true, key: 'Z' }) === 'undo');
// Controls: a plain keypress is not a shortcut, and neither is another chord.
check('Z on its own is just a letter', historyShortcut({ key: 'z' }) === null);
check('Ctrl+S is not our business', historyShortcut({ ctrlKey: true, key: 's' }) === null);
check('an event with no key does not throw', historyShortcut({ ctrlKey: true }) === null);

shutDown();
