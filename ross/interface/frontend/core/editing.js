// Where the element being edited is, after the list changes under it.
//
// The form remembers its element by **position** (`state.editingIndex`) --
// elements have no identity of their own. So every change that moves positions
// owes the form an update, or saving writes the form into whichever element now
// sits where the edited one used to be. With no message: the list simply shows
// the wrong element changed.
//
// `deleteItem` and `splitItem` did this arithmetic by hand; `copyItem` and
// dragging did not do it at all. Copying an element above the one being edited
// pushed it down by one, and "Save" then overwrote its neighbour. The three
// rules live here, pure, so each caller states *what* happened and none of them
// repeats *how* positions follow.

// An element was inserted at `at`; everything from there down moved one place.
export function afterInsertion(editing, at) {
    if (editing < 0) return editing;
    return editing >= at ? editing + 1 : editing;
}

// The element at `at` was removed. `null` means it was the one being edited,
// and the form has nothing left to point at -- the caller closes it.
export function afterRemoval(editing, at) {
    if (editing < 0) return editing;
    if (editing === at) return null;
    return editing > at ? editing - 1 : editing;
}

// The element at `from` was dragged to `to`.
export function afterMove(editing, from, to) {
    if (editing < 0) return editing;
    if (editing === from) return to;
    if (from < editing && editing <= to) return editing - 1;
    if (to <= editing && editing < from) return editing + 1;
    return editing;
}
