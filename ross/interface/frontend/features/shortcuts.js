// Ctrl+Z and Ctrl+Y on the modelling screen.
//
// This module exists because `main.js` is allowed to wire and not to decide --
// `tests/test_frontend_structure.py::test_the_entry_point_holds_no_logic` says
// so, and it caught the first version of this, which put the whole keyboard map
// in the entry point. It was right: a keyboard map is a decision, and a
// decision has a home.
//
// WHY THE SHORTCUT IS GUARDED AND NOT GLOBAL. Three cases, each of which would
// otherwise be a bug the person blames on themselves:
//
//   * a field is focused -- then Ctrl+Z means the browser's own undo, because
//     what they are undoing is their typing, not their model;
//   * a dialog is open -- then the model behind it is not what is being edited;
//   * another screen is showing -- then there is no history to walk.
import { redoModelling, undoModelling } from './modeling.js';

const FIELDS = ['INPUT', 'SELECT', 'TEXTAREA'];

function typingSomewhere(target) {
    if (!target || !target.tagName) return false;
    return FIELDS.indexOf(target.tagName) !== -1 || target.isContentEditable === true;
}

function aDialogIsOpen() {
    return Array.prototype.some.call(
        document.querySelectorAll('.modal-overlay'),
        overlay => overlay.style.display === 'flex',
    );
}

// Which of the two a keystroke means, or null. A pure function of the event, so
// that which keys mean what can be checked apart from when they are allowed to
// fire. Ctrl+Shift+Z is here because half the world's editors spell redo that
// way and the other half spell it Ctrl+Y.
export function historyShortcut(event) {
    if (!(event.ctrlKey || event.metaKey)) return null;
    const key = String(event.key || '').toLowerCase();
    if (key === 'z' && event.shiftKey) return 'redo';
    if (key === 'z') return 'undo';
    if (key === 'y') return 'redo';
    return null;
}

export function shortcutsAreLive(target) {
    const screen = document.getElementById('screen-modeling');
    if (!screen || !screen.classList.contains('active')) return false;
    return !typingSomewhere(target) && !aDialogIsOpen();
}

export function startHistoryShortcuts() {
    document.addEventListener('keydown', event => {
        const wanted = historyShortcut(event);
        if (!wanted) return;
        if (!shortcutsAreLive(event.target)) return;
        event.preventDefault();
        if (wanted === 'undo') undoModelling();
        else redoModelling();
    });
}
