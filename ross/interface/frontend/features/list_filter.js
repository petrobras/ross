// The filter of the element list, on the screen: the panel's controls, and
// the values its condition needs read by the server.
//
// What is filtered, and how, is core/list_filter.js; the panel is
// components/list_filter_panel.js. This module is the part that talks to the
// server -- a component does not -- and redraws the list when it answers.
import { renderList } from '../components/list.js';
import { apiFetch } from '../core/api.js';
import { clearCriteria, rememberNumbers, setCriterion, toggleFilterPanel } from '../core/list_filter.js';
import { state } from '../core/state.js';

// Conversions asked and not answered yet: a redraw while waiting does not ask
// again.
const asking = new Set();

// Subscribed in main.js to `onNumbersNeeded`.
export async function readNumbers(missing) {
    const fresh = missing.filter(m => !asking.has(m.key));
    if (!fresh.length) return;
    fresh.forEach(m => asking.add(m.key));
    let values = null;
    try {
        const answer = await apiFetch('/api/units/convert', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ items: fresh.map(({ value, unit, to }) => ({ value, unit, to })) }),
        });
        const body = await answer.json();
        if (!answer.ok) throw new Error(body.message || 'HTTP ' + answer.status);
        values = body.values;
    } finally {
        // An answer, or a failure: either way these are no longer being asked.
        // After a failure they stay unknown, and the rows stay shown -- the
        // error notice says what happened.
        fresh.forEach(m => asking.delete(m.key));
    }
    rememberNumbers(fresh.map((m, i) => ({ key: m.key, number: values[i] })));
    if (state.currentTab && state.projectData) renderList();
}

// A control of the panel changed. Choosing another field starts its unit
// over at the field's own: a length's "mm" means nothing to a mass.
export function filterBy(key, value) {
    if (!key || !state.currentTab) return;
    setCriterion(state.currentTab, key, value);
    if (key === 'field') setCriterion(state.currentTab, 'unit', '');
    renderList();
}

export function clearListFilter() {
    clearCriteria(state.currentTab);
    renderList();
}

export function toggleListFilter() {
    toggleFilterPanel();
    renderList();
}
