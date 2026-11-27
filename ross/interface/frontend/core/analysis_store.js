// The analysis registry: what each card is, with which parameters and under
// which rotor model. Not in the DOM: deleting a card must not delete the only
// copy of its configuration.
import { escapeHtml } from './dom.js';
import { t } from './i18n.js';
// Templates

// --- The analysis state -----------------------------------------------------
//
// Each analysis is a record here; the card on screen is its drawing, never the
// other way round. With the DOM standing in for a database, saving a rotor
// would sweep the cards and read the charts' divs: an analysis whose card had
// not rendered -- because the request failed, say -- would vanish from what is
// saved, with no warning.

export const ANALYSES = new Map();   // id -> record

export function registerAnalysis(id, type, title, conversion) {
    ANALYSES.set(id, {
        id: id,
        type: type,
        title: title,
        conversion: conversion || '',
        params: {},
        figure: null      // { data, layout, frames } after the first computation
    });
    return ANALYSES.get(id);
}

export function recordResult(id, params, conversion, figure) {
    const record = ANALYSES.get(id);
    if (!record) return null;
    record.params = Object.assign({}, params);
    record.conversion = conversion || '';
    // A figure with no traces is not a figure. Storing it as though it were would
    // make `hasChart` lie, and a restored card would look computed.
    if (figure && figure.data && figure.data.length > 0) {
        record.figure = {
            data: figure.data,
            layout: figure.layout || {},
            frames: figure.frames || []
        };
    }
    return record;
}

export function hasChart(record) {
    return !!(record && record.figure);
}

// --- The conversion belongs to the analysis, not to the screen ---------------
//
// The "Rotor Model" selector sits beside the button that creates a card: it
// picks the model of the card about to be born. From then on the model belongs
// to the card, and the badge in the header is what announces which one it is.
//
// Three paths disagreed with that. `runCardAnalysis` re-read the selector on
// every computation: changing the selector and pressing Update recomputed under
// another model without changing the badge, and after a page reload -- with the
// selector back at its default -- the 4 DoF and torsional cards all recomputed
// as 6 DoF. And cards restored from memory or from a file were rebuilt with no
// badge at all, because the badge only existed in the HTML `addAnalysis` wrote.
//
// 6 DoF got a badge too. The absence of one was ambiguous: there was no telling
// "full model" from "badge lost along the way" -- which was exactly what was
// happening.
const CONVERSION_BADGES = {
    '':          { text: '6 DoF',     cssClass: 'badge-6dof',      tooltip: 'conv6dof' },
    '4dof':      { text: '4 DoF',     cssClass: 'badge-4dof',      tooltip: 'conv4dof' },
    'torsional': { text: 'Torsional', cssClass: 'badge-torsional', tooltip: 'convTorsional' },
};

export function conversionBadge(conversion) {
    const key = conversion || '';
    const badge = CONVERSION_BADGES[key];
    // A value the map does not know shows up as it is, instead of vanishing:
    // with no badge the card would say "6 DoF", and say it wrongly.
    if (!badge) {
        return `<span class="badge-conversion" title="${escapeHtml(key)}">`
             + `${escapeHtml(key)}</span>`;
    }
    return `<span class="badge-conversion ${badge.cssClass}" title="${escapeHtml(t(badge.tooltip))}">`
         + `${badge.text}</span>`;
}

// The model's name, without the badge's HTML. Stripping the badge's tags with a
// regular expression would work and would be a lie: whoever wants the text asks
// for the text.
export function conversionName(conversion) {
    const badge = CONVERSION_BADGES[conversion || ''];
    return badge ? badge.text : (conversion || '');
}

// The record decides. The selector only answers for a card that does not exist yet.
export function cardConversion(uniqueId) {
    const record = ANALYSES.get(uniqueId);
    if (record) return record.conversion || '';
    const node = document.getElementById('rotor-conversion-type');
    return node ? node.value : '';
}

export function forgetAnalysis(id) {
    ANALYSES.delete(id);
}

// Cards go in with `afterbegin`, so the screen shows newest to oldest. The Map
// keeps creation order; this function returns them in the order they appear,
// which is the order they are saved and restored in.
export function analysesInScreenOrder() {
    return Array.from(ANALYSES.values()).reverse();
}

// Only the ones that already computed: an analysis without a figure has no
// chart to save, but it keeps its configuration.
export function analysesToSave() {
    return analysesInScreenOrder().map(record => ({
        title: record.title,
        type: record.type,
        params: record.params,
        conversion: record.conversion,
        data: (record.figure && record.figure.data) || [],
        layout: (record.figure && record.figure.layout) || {},
        frames: (record.figure && record.figure.frames) || []
    }));
}

export function forgetAllAnalyses() {
    ANALYSES.clear();
}


// Format the kwargs

// The Python script comes from the backend, at /api/export/python.
//
// Assembled here, the frontend would need its own copy of three things the
// backend already knows: node numbering, the unit map and the ROSS class names.
// One of them drifting would be enough for the exported file to stop reproducing
// the on-screen chart, with no error at all. The same code that builds the real
// rotor writes the script, and the backend runs ast.parse on it before handing
// it back.

export function collectActiveAnalyses() {
    // The criterion is having configuration, not having a chart. A card restored
    // from an earlier session lost its figure but kept its parameters, and the
    // Python script does not need the figure -- it needs the parameters. Filtering
    // by chart would leave those analyses out of the exported file.
    return analysesInScreenOrder()
        .filter(record => Object.keys(record.params || {}).length > 0)
        .map(record => ({
            type: record.type,
            params: record.params,
            conversion: record.conversion || ''
        }));
}

// The project and the analyses come in explicitly, not read off the screen:
// exporting a rotor from the hub must not carry along the analysis cards of the
// open project, which speak of another rotor and other nodes.
// A script carries **one** rotor, and a rotor has **one** degree-of-freedom
// conversion. If the chosen analyses were computed under different conversions,
// no script reproduces them all: one would have to be picked and the others
// would come out with numbers different from those on screen.
// Return the unanimous conversion, or null when there is a conflict.

export function unanimousConversion(analyses) {
    const conversions = new Set(analyses.map(a => a.conversion || ''));
    return conversions.size <= 1 ? (conversions.values().next().value || '') : null;
}
