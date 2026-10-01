// The buttons of the analysis screen: the cards, their forms and the row
// editors (probes, forces, unbalances). Phase 5, slice 14 -- the third stage
// of moving the page off the `window` bridge (see core/actions.js).
//
// A card's buttons carry the card as `data-card` and the analysis as
// `data-type`. The row editors add with `data-list` (which kind of list) and
// `data-field` (which field of the form holds it); a row is removed by its own
// button, which finds the row it sits in.

import { openSectionHelp } from '../components/help.js';
import {
    addAnalysis, addAngleProbeRow, addForceRow, addProbeRow, addUnbalanceRow, checkDeps,
    deleteAnalysis, runCardAnalysis, toggleAnalysis, toggleDashAdv,
} from './analysis.js';

// Which editor adds a row to which kind of list. The names are the catalogue's
// field types; a type missing here is an error with its name, not a dead button.
const ADD_ROW = {
    probe_list: addProbeRow,
    force_list: addForceRow,
    unbalance_list: addUnbalanceRow,
    angle_probe_list: addAngleProbeRow,
};

function addRow(element) {
    const add = ADD_ROW[element.dataset.list];
    if (!add) throw new Error('no row editor for a list of type "' + element.dataset.list + '"');
    add(element.dataset.card, element.dataset.field, element.dataset.type);
}

export const ANALYSIS_ACTIONS = {
    'add-analysis': (element, event) => addAnalysis(event),

    // a card
    'toggle-card': element => toggleAnalysis(element.dataset.card),
    'run-card': element => runCardAnalysis(element.dataset.card, element.dataset.type),
    'card-help': element => openSectionHelp(element.dataset.type),
    'delete-card': element => deleteAnalysis('card-' + element.dataset.card),

    // its form
    'check-deps': element => checkDeps(element.dataset.card),
    'toggle-dash-advanced': element => toggleDashAdv(element),
    'add-row': element => addRow(element),
    'remove-row': element => element.closest('.probe-row').remove(),
};
