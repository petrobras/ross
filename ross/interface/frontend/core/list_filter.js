// The filter of the element list: which rows are shown, by material, model,
// node range, text in the tag, and a condition on a field's value.
//
// A way of looking at the list and nothing else: it changes no element, and
// nothing about it is saved. Each category keeps its own criteria while the
// page is open -- a material chosen for the shafts means nothing to the
// bearings.
//
// WHY THE ROWS ARE HIDDEN AND NOT LEFT OUT. Every row carries its position in
// the project (`data-index`), and so does everything that reaches a row by
// counting: the form's mark, the 3D view's pointer, the drag. A row left out
// would shift every position after it. Hidden, the rows keep their places.
//
// WHY A VALUE IS COMPARED THROUGH THE SERVER. A field holds what was typed --
// `250`, `0.25`, `2*pi`, a list -- in the unit picked beside it, and the units
// include whatever someone typed into "Others". Only pint, on the server,
// knows all of that (`domain/conversion.to_unit`). The numbers it answers are
// kept here by what they were converted from, so a redraw of the list asks
// only for what it has not seen. Until an answer arrives the row is shown:
// hiding a row on a number nobody has read yet would be a guess.
import { schemaFor } from './schema.js';

export const OPERATORS = ['>', '>=', '<', '<=', '='];

const EMPTY = { material: '', model: '', nodeFrom: '', nodeTo: '', text: '', field: '', op: '>', value: '', unit: '' };

// The default material of every form that has one, as the form writes it.
export const DEFAULT_MATERIAL = 'Default (Steel)';

const criteriaBy = new Map();
let panelOpen = false;
const numbers = new Map();

export function criteriaFor(category) {
    return Object.assign({}, EMPTY, criteriaBy.get(category) || {});
}

export function setCriterion(category, key, value) {
    if (!(key in EMPTY)) throw new Error('no filter criterion is called "' + key + '"');
    const next = criteriaFor(category);
    next[key] = String(value == null ? '' : value);
    criteriaBy.set(category, next);
}

export function clearCriteria(category) {
    criteriaBy.delete(category);
}

// Leaving a rotor forgets every filter: the next rotor's materials and
// fields are not this one's.
export function forgetFilters() {
    criteriaBy.clear();
}

export function filterPanelOpen() {
    return panelOpen;
}

export function toggleFilterPanel() {
    panelOpen = !panelOpen;
    return panelOpen;
}

function finite(text) {
    const value = Number(String(text).trim());
    return String(text).trim() !== '' && Number.isFinite(value) ? value : null;
}

function conditionActive(c) {
    return !!c.field && finite(c.value) !== null;
}

// How many criteria are in force: the badge on the filter button.
export function activeCount(c) {
    return [!!c.material, !!c.model, finite(c.nodeFrom) !== null || finite(c.nodeTo) !== null,
        c.text.trim() !== '', conditionActive(c)].filter(Boolean).length;
}

// What makes two filters the same: the selection is stamped with it, so a
// selection made under one filter is not acted on under another (core/state.js).
export function filterSignature(category) {
    const c = criteriaFor(category);
    return activeCount(c) ? JSON.stringify(c) : '';
}

// The material an element is built with, or null when its form has none.
export function materialOf(category, item) {
    const form = schemaFor(category, item.element_type || 'BASIC');
    if (!form || !form.fields.some(f => f.control === 'material_ref')) return null;
    const name = String(item.material || '').trim();
    return name === '' || name.toLowerCase() === DEFAULT_MATERIAL.toLowerCase() ? DEFAULT_MATERIAL : name;
}

// The unit a field of this form is typed in when nobody changed it.
function formUnit(category, item, name) {
    const form = schemaFor(category, item.element_type || 'BASIC');
    const field = form && form.fields.find(f => f.name === name);
    return (field && field.unit) || '';
}

// The unit a field of this element is typed in: its own selector, or the
// form's default.
function unitOf(category, item, name) {
    return item[name + '_unit'] || formUnit(category, item, name);
}

// The unit the condition is written in: the one picked in the panel, or the
// field's own until one is picked. Leaving it empty compared 0.3 m with
// "260" as if both were millimetres -- measured in a browser.
function conditionUnit(category, item, c) {
    return c.unit || formUnit(category, item, c.field);
}

export function conversionKey(value, from, to) {
    return JSON.stringify([String(value).trim(), from || '', to || '']);
}

// A plain number with no conversion to do needs no server.
function localNumber(value, from, to) {
    return (!from || !to || from === to) ? finite(value) : null;
}

export function rememberNumbers(entries) {
    entries.forEach(({ key, number }) => numbers.set(key, typeof number === 'number' ? number : null));
}

// What the condition needs converted and has not been told yet.
export function missingNumbers(category, items) {
    const c = criteriaFor(category);
    if (!conditionActive(c)) return [];
    const missing = new Map();
    for (const item of items) {
        const raw = item && item[c.field];
        if (raw === undefined || raw === null || String(raw).trim() === '') continue;
        const from = unitOf(category, item, c.field);
        const to = conditionUnit(category, item, c);
        if (localNumber(raw, from, to) !== null) continue;
        const key = conversionKey(raw, from, to);
        if (!numbers.has(key)) missing.set(key, { key, value: String(raw), unit: from, to });
    }
    return Array.from(missing.values());
}

function compare(number, op, limit) {
    const close = Math.abs(number - limit) <= 1e-9 * Math.max(1, Math.abs(number), Math.abs(limit));
    if (op === '=') return close;
    if (op === '>') return number > limit && !close;
    if (op === '>=') return number > limit || close;
    if (op === '<') return number < limit && !close;
    return number < limit || close;
}

// Whether a row passes: true, false, or null when its value is still being
// read by the server (and it is shown meanwhile). `node` is the row's
// effective node, as the list numbers it.
export function rowMatches(category, item, node, criteria) {
    const c = criteria || criteriaFor(category);
    if (!item) return false;
    if (c.material && materialOf(category, item) !== c.material) return false;
    if (c.model && (item.element_type || 'BASIC') !== c.model) return false;
    const from = finite(c.nodeFrom);
    const to = finite(c.nodeTo);
    if (from !== null && !(node >= from)) return false;
    if (to !== null && !(node <= to)) return false;
    if (c.text.trim() && !String(item.tag || '').toLowerCase().includes(c.text.trim().toLowerCase())) return false;
    if (!conditionActive(c)) return true;
    const raw = item[c.field];
    if (raw === undefined || raw === null || String(raw).trim() === '') return false;
    const unit = unitOf(category, item, c.field);
    const target = conditionUnit(category, item, c);
    let number = localNumber(raw, unit, target);
    if (number === null) {
        const key = conversionKey(raw, unit, target);
        if (!numbers.has(key)) return null;
        number = numbers.get(key);
    }
    return number !== null && compare(number, c.op, finite(c.value));
}

// The positions shown, for "select all": a row still being read counts as
// shown, as it is on screen.
export function shownIndexes(category, items, nodes) {
    const c = criteriaFor(category);
    return items.map((item, index) => index).filter(index => rowMatches(category, items[index], nodes[index], c) !== false);
}

// The list asks for what it is missing; features/list_filter.js answers from
// the server and redraws. A hook, because a component does not call the
// server. It throws until someone subscribes, like `onReorder`: a filter that
// silently never converts would show every row and call it filtered.
let numbersNeeded = () => {
    throw new Error('onNumbersNeeded: nobody subscribed to the conversions the list filter needs');
};

export function onNumbersNeeded(fn) {
    numbersNeeded = fn;
}

export function askForNumbers(missing) {
    if (missing.length) numbersNeeded(missing);
}
