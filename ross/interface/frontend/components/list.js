// The element list of the modeling screen: effective node numbering, the row
// whose form is open, and reordering by dragging.
import { escapeHtml } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { listContext, state, getActiveData, syncBackToLibrary } from '../core/state.js';
import { allPicked, isPicked, nowShowing, pickedCount } from '../core/selection.js';
import { afterMove } from '../core/editing.js';
import { categoryHidden, elementHidden, threeDShown } from '../core/visibility.js';
import {
    activeCount, askForNumbers, criteriaFor, filterPanelOpen, missingNumbers, rowMatches,
} from '../core/list_filter.js';
import { renderFilterPanel } from './list_filter_panel.js';
// The list does not know the rotor. Whoever builds the rotor subscribes here;
// before, `renderList` called `buildRotorLive` directly, and measuring the
// boundaries showed that as the only path from a component to a feature.
// The default **throws**, and that is not carelessness. Slice 3 created this hook
// and forgot to connect it; with a `() => {}` in its place, dragging an element
// stopped updating the figure with nothing to show for it, and only the user
// noticed. A hook with no subscriber has to hurt on first use.
let reorderHandler = () => {
    throw new Error('onReorder: ninguem se inscreveu no reordenamento');
};

export function onReorder(fn) {
    reorderHandler = fn;
}

// Node numbering -- mirror of domain/node_resolver.effective_nodes.
//
// Here the function only labels the list on screen; what decides the nodes of the
// figure and of the exported script is the backend. But the user compares the
// two, so the rule has to be identical. That is why the regex exists:
// JavaScript's Number() accepts '0x10' and Python's float() accepts '1_0'; each
// side turned a different piece of garbage into a node. tests/test_fase1_fatia4.py
// compares the two implementations case by case.
const NODE_SYNTAX = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

function explicitNode(item) {
    const raw = (item.n === undefined || item.n === null) ? '' : String(item.n).trim();
    if (!NODE_SYNTAX.test(raw)) return null;
    const value = Number(raw);
    if (!isFinite(value)) return null;
    return Math.trunc(value);
}

// The builder's rule for couplings (`node_resolver.listed_nodes`): the node the
// user pinned, else the coupling's place in the list.
export const getListedNodes = (arr) => arr.map((item, index) => {
    const node = explicitNode(item);
    return node === null ? index : node;
});

// The node each row of a tab sits on, by the rule the builder uses for that
// category: couplings by `getListedNodes`, every other one by `getEffectiveNodes`.
// The list, its filter and its "select all" read nodes through here, so a
// coupling with a blank `n` is labelled with the node it is built on.
export const getTabNodes = (tab, arr) =>
    (tab === 'couplings' ? getListedNodes(arr) : getEffectiveNodes(arr));

export const getEffectiveNodes = (arr) => {
    const fixed = new Set();
    for (const item of arr) {
        const node = explicitNode(item);
        if (node !== null) fixed.add(node);
    }

    const eff = [];
    let next = 0;
    for (const item of arr) {
        const node = explicitNode(item);
        if (node !== null) { eff.push(node); continue; }
        while (fixed.has(next)) next++;
        eff.push(next);
        fixed.add(next);
        next++;
    }
    return eff;
};

// The row whose form is open, marked, and the heading of the form window.
//
// The form used to open under its row, and that was how the person knew which
// element it was. In a window of its own (components/floating_form.js) the row is
// marked instead, and the window's bar says it in words -- the words of the
// row, so the two cannot disagree. Called on every redraw of the list, because
// a drag or a copy above it moves the edited row and renumbers it.
export function markEditedRow() {
    const box = document.getElementById('insertion-form');
    const open = !!box && box.style.display === 'block';
    const editing = open ? state.editingIndex : -1;
    Array.from(document.getElementById('element-list').children).forEach((row, index) => {
        if (index === editing) row.classList.add('is-editing');
        else row.classList.remove('is-editing');
    });
    const title = document.getElementById('form-window-title');
    if (title) title.textContent = open ? formTitle(editing) : '';
}

function formTitle(editing) {
    const items = getActiveData()[state.currentTab] || [];
    let title = editing >= 0 && items[editing]
        ? rowTitle(items[editing], editing, getTabNodes(state.currentTab, items)[editing])
        : t('formWindowNew').replace('%1', () => categoryName(state.currentTab));
    // On a MultiRotor the list says which line it is showing; with the list
    // hidden, only the window can.
    if (state.projectData.isMultiRotor) {
        title += ' · ' + t(state.multiRotorEditTarget === 'driven' ? 'multiDriven' : 'multiDriving');
    }
    return title;
}

// The translated name of a category comes from the sidebar button that opens it:
// that button is the one carrying the `data-i18n`. A second `category -> key`
// table here could only diverge from the one already in `index.html`.
export function categoryName(category, button) {
    const target = button || tabButton(category);
    const key = target && target.dataset && target.dataset.i18n;
    return key ? t(key) : category;
}

// The sidebar button of a category, found by what it *is* (`data-tab`) rather
// than by what it happens to call. It used to be found by reading `openTab('x')`
// out of the `onclick` text -- which is how this slice would have broken the
// title and the highlight the moment the buttons started calling `pickTab`.
export function tabButton(category) {
    return Array.from(document.querySelectorAll('.tab-btn'))
        .find(b => b.dataset && b.dataset.tab === category);
}

// The bar above the list: tick everything, how many are ticked, and the two
// things worth doing to several at once.
//
// It is shown whenever the list has anything in it, and not only once something
// is ticked. A bar that appears with the first tick would hide the *only* way of
// ticking everything, which is the one case where selecting several by hand is
// most tedious. The two action buttons are dead until something is ticked, which
// says the same thing without hiding it.
//
// The filter button sits here too (core/list_filter.js), with the number of
// criteria in force; with a filter on and nothing ticked, the count says how
// many of the rows are shown. "Select all" ticks the rows shown, not the ones
// filtered out of sight.
function renderSelectionBar(count, shown, filtering) {
    const bar = document.getElementById('selection-bar');
    if (!bar) return;
    if (!count) {
        bar.style.display = 'none';
        bar.innerHTML = '';
        return;
    }
    const chosen = pickedCount(listContext());
    const dead = chosen ? '' : 'disabled';
    const counted = chosen ? escapeHtml(t('selectedCount')).replace('%1', () => chosen)
        : filtering ? escapeHtml(t('filterShowing')).replace('%1', () => shown.length).replace('%2', () => count) : '';
    const badge = filtering ? `<span class="filter-badge">${filtering}</span>` : '';
    bar.style.display = 'flex';
    bar.innerHTML = `
        <label class="pick-all">
            <input type="checkbox" data-action="pick-all" ${allPicked(listContext(), shown) ? 'checked' : ''}>
            <span>${escapeHtml(t('selectAll'))}</span>
        </label>
        <span class="pick-count">${counted}</span>
        <button class="btn-action filter${filtering ? ' is-on' : ''}" data-action="toggle-list-filter" aria-pressed="${String(filterPanelOpen())}" title="${escapeHtml(t('filterTitle'))}"><i class="fas fa-filter"></i>${badge}</button>
        ${meshButton()}
        <button class="btn-action copy" data-action="copy-picked" ${dead} title="${escapeHtml(t('copySelected'))}"><i class="fas fa-copy"></i></button>
        <button class="btn-action delete" data-action="delete-picked" ${dead} title="${escapeHtml(t('deleteSelected'))}"><i class="fas fa-trash"></i></button>
    `;
}

// Discretizing the shafts (features/mesh.js): in the shafts' bar only, and
// live whether or not anything is ticked -- nothing ticked is every shaft.
function meshButton() {
    if (state.currentTab !== 'shafts') return '';
    return `<button class="btn-action mesh" data-action="mesh-shafts" title="${escapeHtml(t('meshTitle'))}"><i class="fas fa-ruler-horizontal"></i></button>`;
}

// The split button, and only on shafts.
//
// It is an element action rather than a screen-level field because what
// `add_nodes` needs -- a position along the whole shaft line, in metres -- is
// something the user would have to work out by adding up lengths. The row
// already knows which element it is, so the question shrinks to "how far along
// *this* one", in the millimetres the form already speaks.
//
// Every other category is on a node, not between two: a disk has nothing to
// split. So the button appears for `shafts` and for nothing else.
// The eye: this element in or out of the 3D view (core/visibility.js). Only
// while the 3D view is the one on screen -- ROSS's 2D figure cannot hide one
// element -- and not on materials, which are not drawn.
function eyeButton(item, index) {
    if (!threeDShown() || state.currentTab === 'materials') return '';
    const hidden = elementHidden(item);
    const title = hidden ? t('rotor3dShowElement') : t('rotor3dHideElement');
    return `<button class="btn-action eye${hidden ? ' is-off' : ''}" data-action="hide-element" data-index="${index}" aria-pressed="${String(hidden)}" title="${escapeHtml(title)}"><i class="fas ${hidden ? 'fa-eye-slash' : 'fa-eye'}"></i></button>`;
}

function splitButton(index) {
    if (state.currentTab !== 'shafts') return '';
    return `<button class="btn-action split" data-action="split-element" data-index="${index}" title="${escapeHtml(t('splitTitle'))}"><i class="fas fa-scissors"></i></button>`;
}

let sortableInstance = null;

function rowTitle(item, index, effectiveNode) {
    if (state.currentTab === 'materials') {
        return `MATERIAL #${index + 1}` + (item.name ? ` - ${item.name}` : '');
    }
    const kind = (item.element_type && item.element_type !== 'BASIC') ? ` [${item.element_type}]` : '';
    if (state.currentTab === 'couplings') return `COUPLING #${index + 1}` + kind;
    const singularName = (state.currentTab === 'pointmasses') ? 'POINTMASS' : state.currentTab.slice(0, -1).toUpperCase();
    return `${singularName} #${index + 1} (Node ${effectiveNode})` + kind;
}

// The rows of the list, detached. Nothing here touches the page.
function buildRows(currentArray, effNodes, passes) {
    return currentArray.map((item, index) => {
        const div = document.createElement('div');
        // Faded when the 3D view on screen does not draw it -- hidden by its
        // eye or with its whole category from the legend.
        const outOfView = threeDShown() && state.currentTab !== 'materials'
            && (elementHidden(item) || categoryHidden(state.currentTab));
        // Out of the filter: hidden, but still in its place (core/list_filter.js).
        div.className = 'list-item' + (outOfView ? ' is-out-of-view' : '') + (passes[index] === false ? ' is-filtered-out' : '');
        div.innerHTML = `
            <div style="display:flex; align-items:center; flex:1; overflow:hidden;">
                <input type="checkbox" class="item-pick" data-action="pick-element" data-index="${index}" ${isPicked(listContext(), index) ? 'checked' : ''} title="${escapeHtml(t('select'))}">
                <i class="fas fa-grip-vertical item-drag"></i>
                <span class="item-text">${escapeHtml(rowTitle(item, index, effNodes[index]))}</span>
            </div>
            <div class="item-actions">
                ${eyeButton(item, index)}
                ${splitButton(index)}
                <button class="btn-action edit" data-action="edit-element" data-index="${index}" title="${escapeHtml(t('edit'))}"><i class="fas fa-pen"></i></button>
                <button class="btn-action copy" data-action="copy-element" data-index="${index}" title="${escapeHtml(t('copy'))}"><i class="fas fa-copy"></i></button>
                <button class="btn-action delete" data-action="delete-element" data-index="${index}" title="${escapeHtml(t('delete'))}"><i class="fas fa-trash"></i></button>
            </div>
        `;
        return div;
    });
}

// The list is **built first and swapped in at the end**.
//
// It used to be emptied first -- `container.innerHTML = ''` -- and then built
// from the project. Anything that threw in between (a category missing from the
// project, a `null` in a list, a half of a MultiRotor that is not there) left
// the panel empty, and with no error handler anywhere, nothing on screen or in
// the console said why. That is what Leonardo saw as "the lists vanished".
//
// Now every row is made before the container is touched. If making them fails,
// the error still propagates -- the notice in `features/error_notice.js` shows
// it -- but the list that was on screen stays there, and the form with it.
export function renderList() {
    const container = document.getElementById('element-list');

    const activeData = getActiveData();
    const currentArray = activeData[state.currentTab];

    // Before anything is read: if this is not the list the ticks were made on,
    // they are gone. Drawing is the moment the list on screen changes, and it
    // is the only moment, which is why this lives here and not in the three
    // places that cause it.
    nowShowing(listContext());

    const effNodes = getTabNodes(state.currentTab, currentArray);
    const criteria = criteriaFor(state.currentTab);
    const filtering = activeCount(criteria);
    const passes = currentArray.map((item, index) =>
        (filtering ? rowMatches(state.currentTab, item, effNodes[index], criteria) : true));
    const rows = buildRows(currentArray, effNodes, passes);
    const shown = passes.map((pass, index) => (pass === false ? -1 : index)).filter(index => index >= 0);

    // Nothing below can fail on the project's content: from here on the old
    // list is replaced by one that is already complete.
    container.innerHTML = '';
    rows.forEach(row => container.appendChild(row));
    if (filtering) container.classList.add('is-filtering');
    else container.classList.remove('is-filtering');
    renderSelectionBar(currentArray.length, shown, filtering);
    renderFilterPanel(state.currentTab, shown.length, currentArray.length);
    markEditedRow();
    // Values the filter's condition needs read by the server; the answer
    // redraws the list (features/list_filter.js).
    askForNumbers(missingNumbers(state.currentTab, currentArray));

    // One instance per container: before, every render created another one without
    // destroying the previous, piling listeners onto the same list.
    if (sortableInstance) sortableInstance.destroy();
    sortableInstance = new Sortable(container, {
        handle: '.item-drag',
        // Only the rows move, and only rows are counted.
        //
        // The open form used to be a child of this same container, under the
        // element it edited (it is a window of its own now, but anything else
        // put in the list would do the same). With Sortable's defaults it
        // counted as an item:
        // `oldIndex` and `newIndex` are positions among **all** the children,
        // so with the form above the dragged row every index was one too high.
        // Dragging the last of four shafts to the top with a form open asked
        // for element 4 of a four-element list: nothing was removed, and
        // `undefined` was inserted at the top -- saved as `null`, and every
        // later render of that list failed. Measured in a browser, not guessed.
        //
        // `draggable` keeps anything that is not a row from being picked up,
        // and the `...DraggableIndex` pair counts rows only.
        draggable: '.list-item',
        // No dragging through a filter: rows out of sight would be dragged
        // past without being seen, and the order they land in would surprise.
        disabled: filtering > 0,
        animation: 150,
        onEnd: function (evt) {
            const oldIdx = evt.oldDraggableIndex;
            const newIdx = evt.newDraggableIndex;
            if(oldIdx === newIdx) return;
            
            const actData = getActiveData();
            const length = actData[state.currentTab].length;
            // A position that is not a row of this list is refused rather than
            // spliced: `splice` does not complain, it inserts `undefined`.
            if (!(oldIdx >= 0 && oldIdx < length && newIdx >= 0 && newIdx < length)) {
                renderList();
                throw new Error('drag: position ' + oldIdx + ' -> ' + newIdx + ' is not in a list of ' + length);
            }
            const item = actData[state.currentTab][oldIdx];
            
            actData[state.currentTab].splice(oldIdx, 1);
            actData[state.currentTab].splice(newIdx, 0, item);
            // An open form follows its element to where it was dropped.
            state.editingIndex = afterMove(state.editingIndex, oldIdx, newIdx);            
            
            syncBackToLibrary();
            renderList();
            reorderHandler();
        }
    });
}
