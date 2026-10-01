// Which elements of the modelling list are ticked.
//
// WHY A SELECTION CARRIES THE ADDRESS OF ITS LIST. An element of this project
// has no identity: rotors have a `uid`, shafts and disks and bearings have
// nothing stable -- a `tag` is optional and need not be unique. So a selection
// can only be a set of **positions**, and a position means nothing without
// saying in which list.
//
// The list on screen changes in three places (`openTab`,
// `switchMultiRotorTarget`, and entering a rotor) and its contents change in
// five more. Clearing the selection at each of those is eight places to
// remember, and this project has just paid for exactly that shape of mistake:
// the undo buttons kept the state of the previous rotor because one path
// changed something and told nobody.
//
// So the set is stamped with the list it was taken on, and reading it from
// another list gives nothing rather than gives the wrong thing. Forgetting is
// no longer possible; it is not merely discouraged.
//
// WHY LEAVING THE LIST THROWS THE SELECTION AWAY. The stamp alone makes the
// ticks invisible elsewhere but keeps them waiting, so coming back to the tab
// finds them again. That was a consequence of the design rather than a decision
// about it, and it was decided the other way: a selection belongs to what is on
// screen. `nowShowing` is what enforces it, and it is called by `renderList` --
// the one function that always runs when the list being shown changes.
//
// The reads stay pure. A read with a side effect would mean that merely drawing
// the screen could empty a selection depending on the order things happened in;
// here exactly one function drops it, and it is the one whose whole job is to
// say which list is up.
//
// That handles *which* list. Positions shifting **inside** one list -- an
// element deleted, copied, split, dragged -- is the other half, and it is
// handled inside `projectChanged` in core/state.js, which every change goes
// through: after one, the numbers no longer point at what was ticked. It is a
// call there rather than a subscriber on the hook because a set of positions is
// state and not something on the page -- the reasoning is written beside it.

let stamp = null;
let picks = new Set();

function align(context) {
    if (stamp !== context) {
        stamp = context;
        picks = new Set();
    }
}

// The list on screen is this one now. If it is not the one the ticks were made
// on, they go.
export function nowShowing(context) {
    align(context);
}

export function pick(context, index) {
    align(context);
    if (picks.has(index)) picks.delete(index);
    else picks.add(index);
    return picks.has(index);
}

export function isPicked(context, index) {
    if (stamp !== context) return false;
    return picks.has(index);
}

// Ascending, always. Callers that delete or copy walk it backwards precisely
// because they know it is sorted; handing back insertion order would make that
// a coincidence.
export function picked(context) {
    if (stamp !== context) return [];
    return Array.from(picks).sort((a, b) => a - b);
}

export function pickedCount(context) {
    return picked(context).length;
}

// Tick everything, or untick everything, by the rule the person expects from a
// header checkbox: if it is not already all of them, it becomes all of them.
export function pickAll(context, count) {
    align(context);
    if (picks.size >= count) picks = new Set();
    else picks = new Set(Array.from({ length: count }, (_, index) => index));
    return picks.size;
}

export function allPicked(context, count) {
    return count > 0 && pickedCount(context) === count;
}

export function clearSelection() {
    stamp = null;
    picks = new Set();
}
