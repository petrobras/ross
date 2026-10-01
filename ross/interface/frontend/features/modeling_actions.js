// The buttons of the modelling screen: the category tabs, the element list,
// the form, the node hub and what sits around the rotor figure. Phase 5,
// slice 13 -- the second stage of moving the page off the `window` bridge
// (see core/actions.js).
//
// Rows of a list carry their position as `data-index` instead of inside a
// call written into the attribute (editItem with a 3 in it); categories and
// subtypes as `data-category` and `data-subtype`. `position` reads it as a
// number, because a dataset value is always a string, and `"3" + 1` is `"31"`.

import { fillDefault, handleUnitChange, toggleAdvanced } from '../components/form.js';
import { openSectionHelp } from '../components/help.js';
import {
    addElementFromNodeHub, closeForm, closeNodeHub, copyItem, copySelected, deleteItem,
    deleteSelected, editItem, openForm, pickTab, saveItem, selectSubType, setVerticalScale,
    splitItem, toggleSelectAll, toggleSelected,
} from './modeling.js';
import { switchMultiRotorTarget } from './multirotor.js';

function position(element) {
    return Number(element.dataset.index);
}

export const MODELING_ACTIONS = {
    // the category tabs
    'pick-tab': element => pickTab(element.dataset.tab),
    'section-help': element => openSectionHelp(element.dataset.category),
    'switch-half': element => switchMultiRotorTarget(element.value),

    // the element list
    'pick-element': element => toggleSelected(position(element)),
    'pick-all': () => toggleSelectAll(),
    'copy-picked': () => copySelected(),
    'delete-picked': () => deleteSelected(),
    'edit-element': element => editItem(position(element)),
    'copy-element': element => copyItem(position(element)),
    'delete-element': element => deleteItem(position(element)),
    'split-element': element => splitItem(position(element)),

    // the form
    'add-element': () => openForm(true),
    'save-element': () => saveItem(),
    'close-form': () => closeForm(),
    'pick-subtype': element => selectSubType(element.dataset.subtype),
    'fill-default': () => fillDefault(),
    'toggle-advanced': element => toggleAdvanced(element),
    'change-unit': element => handleUnitChange(element),

    // the node hub, opened from the figure
    'add-from-node-hub': element => addElementFromNodeHub(element.dataset.category),
    'close-node-hub': () => closeNodeHub(),

    // the figure
    'set-vertical-scale': element => setVerticalScale(element.value),
};
