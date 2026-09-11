// The element schema comes from the installed ROSS, via /api/schema/elements.
// Nothing here is a hand-kept copy: a parameter renamed in the library
// disappears from the form on its own, instead of becoming a TypeError at
// build time.
import { apiFetch } from './api.js';
import { setSchemaLanguage, preferredLanguage } from './i18n.js';
// --- Element schema and form building ---------------------------------------
// The forms used to be ~450 lines of literal HTML in FormTemplates, with copies
// of UNITS_MAPPING and UNIT_ALTERNATIVES beside them. They now come from
// /api/schema/elements, which the backend derives from the installed ROSS. A
// parameter renamed in the library disappears from the form on its own, instead
// of becoming a TypeError at build time.

let ELEMENT_SCHEMA = null;
let ANALYSIS_SCHEMA = null;
let ANALYSIS_UNSUPPORTED = {};
let ANALYSIS_TITLES = {};

let SCHEMA_PROMISE = null;

function lookUp(route, language) {
    return apiFetch(route + '?lang=' + encodeURIComponent(language)).then(response => {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
    });
}

// Both schemas arrive together. They depend on the same language, and letting
// one land after the other would open a window in which the analysis screen
// builds a card with no fields at all.
export function loadElementSchema(language) {
    SCHEMA_PROMISE = Promise.all([
        lookUp('/api/schema/elements', language),
        lookUp('/api/schema/analyses', language),
    ]).then(([elements, analyses]) => {
        ELEMENT_SCHEMA = elements;
        ANALYSIS_SCHEMA = analyses.fields;
        ANALYSIS_UNSUPPORTED = analyses.unsupported || {};
        ANALYSIS_TITLES = analyses.titles || {};
        setSchemaLanguage(elements.language);
        UNIT_MAP_CACHE = null;
        return elements;
    });
    return SCHEMA_PROMISE;
}

// An analysis's fields, always as a copy: whoever builds a card writes `val`
// to fill the form with what was saved, and writing into the catalog would
// corrupt the defaults of every card created afterwards -- which is what the
// `JSON.parse(JSON.stringify(...))` scattered across the three paths avoided.
// Why this analysis does not run on the converted rotor, or null when it does.
//
// The table comes from the server -- from the same place the `/run_analysis`
// route refuses from. The screen warns before computing by reading exactly what
// the server will enforce; a second table here would drift, and the user would
// see "allowed" and get "not allowed".
export function analysisUnsupported(type, conversion) {
    const byConversion = ANALYSIS_UNSUPPORTED[type];
    return (byConversion && byConversion[conversion || '']) || null;
}

// An analysis's title, in the schema's language. Until the i18n slice the
// twelve names were written twice -- in the <select> options of index.html and
// in a `typeNames` in the JS.
export function analysisTitle(type) {
    return ANALYSIS_TITLES[type] || String(type || '').toUpperCase();
}

export function analysisTitles() {
    return Object.assign({}, ANALYSIS_TITLES);
}

export function analysisFieldsFor(type) {
    const fields = (ANALYSIS_SCHEMA || {})[type];
    return fields ? JSON.parse(JSON.stringify(fields)) : null;
}


export function schemaReady() {
    return SCHEMA_PROMISE || loadElementSchema(preferredLanguage());
}

// Unit alternatives and the parameter->unit map also come from the schema:
// they were the duplicated UNIT_ALTERNATIVES and UNITS_MAPPING. The map now
// arrives with inherited units already resolved (a BallBearing's cxx, say).
let UNIT_MAP_CACHE = null;

export function unitAlternativesFor(unit) {
    const map = ELEMENT_SCHEMA && ELEMENT_SCHEMA.unit_alternatives;
    return (map && map[unit]) || null;
}
export function schemaFor(category, subtype) {
    const subtypes = ELEMENT_SCHEMA && ELEMENT_SCHEMA.categories[category];
    return subtypes ? subtypes[subtype] : null;
}

// Subtypes come from the schema; LIST is a BASIC variation made in the interface.
export function formSubtypes(category) {
    const subtypes = ELEMENT_SCHEMA && ELEMENT_SCHEMA.categories[category];
    if (!subtypes) return [];
    const types = Object.keys(subtypes);
    if (types.includes('BASIC')) types.push('LIST');
    return types;
}