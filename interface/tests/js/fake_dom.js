// A fake DOM, installed by the act of importing this module.
//
// The order matters: ES modules are evaluated in the order they appear in the
// `import` list, and some frontend modules touch `document` on load already
// (`components/help.js` wires the click outside the modal). That is why this
// file is always a test's first import, and installs by side effect.
//
// Until Phase 3 slice 3 the JS tests read `app.js` as text and `eval`ed a cut
// between two markers. With modules, they import the real file -- and the class
// of defect where a guard measured another excerpt because a comment moved
// somewhere else disappears.

const nodes = new Map();

let loose = 0;

export function looseNode(tag) {
    return node('loose:' + tag + ':' + (++loose));
}

export function node(id) {
    if (nodes.has(id)) return nodes.get(id);
    const n = {
        id,
        value: '',
        innerHTML: '',
        innerText: '',
        checked: false,
        style: {},
        dataset: {},
        classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
        parentElement: null,
        _children: [],
        // `children` is what the real code uses; `_children` is the actual list.
        // Without this alias `positionFormBox` blew up -- and blew up inside a promise
        // nobody awaited, so the test carried on believing it had passed.
        get children() { return this._children; },
        insertAdjacentElement(where, element) {
            const parent = this.parentElement || this;
            if (!parent._children.includes(element)) parent._children.push(element);
            element.parentElement = parent;
            return element;
        },
        addEventListener() {},
        removeEventListener() {},
        // A real appendChild: the element list is built this way, and a no-op here
        // would make the modeling-screen test measure an empty list and call that
        // success.
        appendChild(child) {
            this._children.push(child);
            child.parentElement = this;
            this.innerHTML += child.innerHTML;
            return child;
        },
        removeChild(child) {
            this._children = this._children.filter(f => f !== child);
            return child;
        },
        contains(child) { return this._children.includes(child); },
        remove() {},
        focus() {}, select() {}, click() {}, on() {},
        // Attributes used to be discarded (`setAttribute` was a no-op). `applyLanguage()`
        // writes `title` and `placeholder` through here: without keeping them, the
        // language battery would have no way to see the translation arrive.
        attributes: {},
        setAttribute(name, value) { this.attributes[name] = value; },
        getAttribute(name) {
            return name in this.attributes ? this.attributes[name] : null;
        },
        getBoundingClientRect() { return { top: 0, left: 0, width: 0, height: 0 }; },
        scrollIntoView() {},
        insertAdjacentHTML(where, html) {
            this.innerHTML = where === 'afterbegin' ? html + this.innerHTML : this.innerHTML + html;
        },
        querySelector() { return node(id + ':q'); },
        querySelectorAll() { return []; },
    };
    nodes.set(id, n);
    return n;
}

// What `querySelectorAll` gives back for a selector. The fake DOM does not
// interpret CSS selectors; the test declares the answer, which is more honest
// than an approximate match that happens to be right.
const bySelector = new Map();

export function registerSelector(selector, list) {
    bySelector.set(selector, list);
}

export function clearDom() {
    nodes.clear();
    bySelector.clear();
}

globalThis.document = {
    getElementById: id => node(id),
    querySelector: sel => node('sel:' + sel),
    querySelectorAll: sel => bySelector.get(sel) || [],
    createElement: tag => looseNode(tag),
    addEventListener() {},
    body: node('body'),
};
globalThis.window = globalThis;
globalThis.Plotly = { newPlot: async () => {}, Plots: { resize() {} } };
// Sortable keeps the options: it is through `onEnd` that the test simulates a drag.
let lastDraggable = null;

export function draggable() {
    return lastDraggable;
}

globalThis.Sortable = function (container, options) {
    lastDraggable = { container, options };
    return { destroy() { } };
};
globalThis.alert = () => {};

// --- localStorage -----------------------------------------------------------
export const disk = { content: {}, quotaExceeded: false };

export function clearDisk() {
    disk.content = {};
    disk.quotaExceeded = false;
}

globalThis.localStorage = {
    getItem: key => (key in disk.content ? disk.content[key] : null),
    setItem: (key, value) => {
        if (disk.quotaExceeded) {
            const error = new Error('QuotaExceededError');
            error.name = 'QuotaExceededError';
            throw error;
        }
        disk.content[key] = String(value);
    },
    removeItem: key => { delete disk.content[key]; },
};

// --- the schema answers ------------------------------------------------------
//
// The analysis catalogue started coming from the server in slice 4. The
// stand-ins answer with the **real** catalogue, read from the golden file a
// Python test keeps equal to `domain/analysis_catalog.py` -- a catalogue
// invented here would make the tests agree with themselves.
import { readFileSync } from 'fs';

const CATALOGUE = JSON.parse(readFileSync(
    new URL('../golden/analysis_schema.json', import.meta.url), 'utf8'));

const CATEGORIES = ['materials', 'shafts', 'disks', 'gears', 'couplings',
                    'seals', 'bearings', 'pointmasses'];

export const ELEMENT_SCHEMA_FIXTURE = {
    language: 'en',
    categories: Object.fromEntries(CATEGORIES.map(c => [c, { BASIC: {
        ross_class: 'X', label: c,
        fields: [{ id: 'n', label: 'Node', type: 'number', val: '0' },
                 { id: 'L', label: 'Length', type: 'number', val: '1', unit: 'meter' }],
    } }])),
    unit_map: {}, unit_alternatives: {},
};

/** The schema answer for `path`, or null if the route is not a schema route. */
export function schemaResponse(path) {
    const route = String(path);
    if (route.includes('/api/schema/elements')) {
        // The real route echoes the language asked for (`build_schema` gives back
        // `language`), and it is from that echo that `loadElementSchema` takes the
        // current language. The fake always gave back 'en': `changeLanguage('pt')`
        // ended up in English here and in Portuguese on the real screen.
        const request = (route.match(/[?&]lang=(\w+)/) || [])[1];
        return Object.assign({}, ELEMENT_SCHEMA_FIXTURE,
                             { language: request || ELEMENT_SCHEMA_FIXTURE.language });
    }
    if (route.includes('/api/schema/analyses')) return CATALOGUE;   // { fields, unsupported }
    return null;
}

// --- the scoreboard ----------------------------------------------------------
export const scoreboard = { ok: 0, failed: 0 };

export function check(description, condition) {
    if (condition) { scoreboard.ok++; console.log('  ok      ' + description); }
    else { scoreboard.failed++; console.log('  FAILED  ' + description); }
}

export function shutDown() {
    console.log('\n' + scoreboard.ok + ' checks ok, ' + scoreboard.failed + ' failed');
    process.exit(scoreboard.failed ? 1 : 0);
}
