// What a button means, by name -- the replacement for the `window` bridge.
//
// WHY. The HTML used to call functions by name, in `onclick` attributes, and
// for that to work every such function had to be hung on `window` by
// main.js: 73 of them. A name that fell off the bridge left its button dead
// with nothing on screen and nothing in the console until somebody clicked.
//
// Now an element says what it *is* -- `data-action="save-rotor"` -- and one
// listener on the document looks the name up here. What changes:
//
//   * a name nobody defined is an error **with the name in it**, raised on the
//     click, and the error notice (features/error_notice.js) shows it;
//   * the functions stay inside their modules; nothing is global;
//   * the whole vocabulary of the page is one table the batteries can read
//     against the HTML, in both directions.
//
// This module holds the table and the listener, and nothing about any screen.
// The tables themselves live with the features that own the buttons.

const ACTIONS = new Map();

// Adds a table of `{ 'action-name': (element, event) => ... }`.
//
// A name defined twice is refused rather than overwritten: two modules
// claiming the same button would otherwise be settled by import order, which
// is nobody's decision.
export function defineActions(table) {
    Object.keys(table).forEach(name => {
        if (ACTIONS.has(name)) throw new Error('action "' + name + '" is defined twice');
        if (typeof table[name] !== 'function') throw new Error('action "' + name + '" is not a function');
        ACTIONS.set(name, table[name]);
    });
}

// The vocabulary, for tests/js/test_actions.js to read against the page.
export function actionNames() {
    return Array.from(ACTIONS.keys());
}

// Runs the action an element names. Exported so a battery can press a button
// without a browser.
export function runAction(element, event) {
    const name = element.dataset.action;
    const action = ACTIONS.get(name);
    if (!action) throw new Error('no action is called "' + name + '"');
    return action(element, event);
}

// Form fields act when their value changes; everything else when clicked.
//
// Without the split, a hidden file input -- which a button opens by calling
// its `click()` -- would run its own action on that very click, before any
// file had been chosen.
const FIELDS = ['SELECT', 'INPUT', 'TEXTAREA'];

function isField(element) {
    return FIELDS.includes(String(element.tagName).toUpperCase());
}

function actionTarget(event) {
    const start = event.target;
    return start && typeof start.closest === 'function' ? start.closest('[data-action]') : null;
}

// Exported for the battery, which drives it with plain objects.
export function handleEvent(event) {
    const element = actionTarget(event);
    if (!element || element.disabled) return undefined;
    if ((event.type === 'change') !== isField(element)) return undefined;
    return runAction(element, event);
}

// One listener per kind of event, on the document: buttons drawn later (the
// cards of the analysis screen, the rows of a list) need nothing wired.
export function startActions() {
    document.addEventListener('click', handleEvent);
    document.addEventListener('change', handleEvent);
}
