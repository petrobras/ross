// The filter panel above the element list (core/list_filter.js holds what it
// filters by, and why the rows are hidden and not left out).
//
// Every control is one action, `filter-by`, naming its criterion in
// `data-key`: the panel is redrawn from the criteria each time the list is,
// so what it shows is never other than what is applied.
import { escapeHtml } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { DEFAULT_MATERIAL, OPERATORS, criteriaFor, filterPanelOpen } from '../core/list_filter.js';
import { formSubtypes, schemaFor } from '../core/schema.js';
import { getActiveData } from '../core/state.js';

// The fields a value can be compared on: typed as text, and not the node, the
// tag or the colour, which have their own criteria or none.
const NOT_VALUES = ['n', 'tag', 'color', 'lubricant'];

// The fields of every model of this category, once each, with the label and
// units of the first model that has them.
export function valueFields(category) {
    const seen = new Map();
    for (const subtype of formSubtypes(category)) {
        const form = schemaFor(category, subtype);
        if (!form) continue;
        for (const field of form.fields) {
            if (field.control !== 'text' || NOT_VALUES.includes(field.name) || seen.has(field.name)) continue;
            seen.set(field.name, { name: field.name, label: field.label, unit: field.unit || '', units: field.unit_options || [] });
        }
    }
    return Array.from(seen.values());
}

function hasMaterial(category) {
    return formSubtypes(category).some(subtype => {
        const form = schemaFor(category, subtype);
        return form && form.fields.some(f => f.control === 'material_ref');
    });
}

function options(values, chosen, label = v => v) {
    return values.map(v => `<option value="${escapeHtml(v)}" ${v === chosen ? 'selected' : ''}>${escapeHtml(label(v))}</option>`).join('');
}

function control(key, html) {
    return html.replace('<select', `<select data-action="filter-by" data-key="${key}"`)
        .replace('<input', `<input data-action="filter-by" data-key="${key}"`);
}

export function renderFilterPanel(category, shown, total) {
    const panel = document.getElementById('list-filter');
    if (!panel) return;
    if (!filterPanelOpen() || !category) {
        panel.style.display = 'none';
        panel.innerHTML = '';
        return;
    }
    const c = criteriaFor(category);
    const any = `<option value="">${escapeHtml(t('filterAny'))}</option>`;
    const rows = [];

    if (hasMaterial(category)) {
        const names = [DEFAULT_MATERIAL].concat((getActiveData().materials || []).map(m => m.name).filter(Boolean));
        rows.push(`<label>${escapeHtml(t('filterMaterial'))}</label>`
            + control('material', `<select>${any}${options(names, c.material)}</select>`));
    }
    const models = formSubtypes(category).filter(type => type !== 'LIST');
    if (models.length > 1) {
        rows.push(`<label>${escapeHtml(t('filterModel'))}</label>`
            + control('model', `<select>${any}${options(models, c.model)}</select>`));
    }
    rows.push(`<label>${escapeHtml(t('filterNodes'))}</label><span class="filter-pair">`
        + control('nodeFrom', `<input type="text" inputmode="numeric" value="${escapeHtml(c.nodeFrom)}" placeholder="${escapeHtml(t('filterFrom'))}">`)
        + control('nodeTo', `<input type="text" inputmode="numeric" value="${escapeHtml(c.nodeTo)}" placeholder="${escapeHtml(t('filterTo'))}">`)
        + '</span>');
    rows.push(`<label>${escapeHtml(t('filterTag'))}</label>`
        + control('text', `<input type="text" value="${escapeHtml(c.text)}">`));

    const fields = valueFields(category);
    if (fields.length) {
        const field = fields.find(f => f.name === c.field);
        const fieldSelect = control('field', `<select>${any}${options(fields.map(f => f.name), c.field,
            name => { const f = fields.find(x => x.name === name); return f.unit ? `${f.label} (${f.unit})` : f.label; })}</select>`);
        let condition = '';
        if (field) {
            const units = field.units.length ? field.units : (field.unit ? [field.unit] : []);
            condition = '<span class="filter-condition">'
                + control('op', `<select>${options(OPERATORS, c.op, op => ({ '>=': '≥', '<=': '≤' }[op] || op))}</select>`)
                + control('value', `<input type="text" inputmode="decimal" value="${escapeHtml(c.value)}">`)
                + (units.length ? control('unit', `<select>${options(units, c.unit || field.unit)}</select>`) : '')
                + '</span>';
        }
        rows.push(`<label>${escapeHtml(t('filterField'))}</label><span class="filter-field">${fieldSelect}${condition}</span>`);
    }

    panel.style.display = 'block';
    panel.innerHTML = `<div class="filter-grid">${rows.join('')}</div>
        <div class="filter-foot">
            <span class="filter-count">${escapeHtml(t('filterShowing')).replace('%1', () => shown).replace('%2', () => total)}</span>
            <button type="button" class="btn-cancel filter-clear" data-action="clear-list-filter">${escapeHtml(t('filterClear'))}</button>
        </div>`;
}
