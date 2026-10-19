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
import { pickShape } from '../components/shape_picker.js';
import {
    addElementFromNodeHub, closeForm, closeNodeHub, copyItem, copySelected, deleteItem,
    deleteSelected, editItem, openForm, pickTab, saveItem, selectSubType, setRotorView,
    meshItems, setVerticalScale, splitItem, toggleSelectAll, toggleSelected,
} from './modeling.js';
import { switchMultiRotorTarget } from './multirotor.js';
import { clearListFilter, filterBy, toggleListFilter } from './list_filter.js';
import {
    frameRotor, toggleBench, toggleCategory3d, toggleDimensions, toggleElement3d, toggleMeasure, togglePan,
} from './rotor3d.js';
import { getActiveData, state } from '../core/state.js';

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
    'mesh-shafts': () => meshItems(),

    // the list's filter (features/list_filter.js)
    'toggle-list-filter': () => toggleListFilter(),
    'filter-by': element => filterBy(element.dataset.key, element.value),
    'clear-list-filter': () => clearListFilter(),

    // the form
    'add-element': () => openForm(true),
    'save-element': () => saveItem(),
    'close-form': () => closeForm(),
    'pick-subtype': element => selectSubType(element.dataset.subtype),
    'fill-default': () => fillDefault(),
    'pick-shape': element => pickShape(element.dataset.shape),
    'toggle-advanced': element => toggleAdvanced(element),
    'change-unit': element => handleUnitChange(element),

    // the node hub, opened from the figure
    'add-from-node-hub': element => addElementFromNodeHub(element.dataset.category),
    'close-node-hub': () => closeNodeHub(),

    // the figure
    'set-vertical-scale': element => setVerticalScale(element.value),
    'set-rotor-view': element => setRotorView(element.dataset.view),
    'frame-rotor': () => frameRotor(),
    'toggle-bench': () => toggleBench(),
    'toggle-dimensions': () => toggleDimensions(),
    'toggle-measure': () => toggleMeasure(),
    'toggle-pan': () => togglePan(),
    'hide-category': (element, event) => toggleCategory3d(element.dataset.category, event),
    'hide-element': element => toggleElement3d(getActiveData()[state.currentTab][position(element)]),
};
