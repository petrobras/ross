// The element list of the modeling screen: effective node numbering, the form box
// anchored to the item, and reordering by dragging.
import { escapeHtml } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { listContext, state, getActiveData, syncBackToLibrary } from '../core/state.js';
import { allPicked, isPicked, nowShowing, pickedCount } from '../core/selection.js';
import { afterMove } from '../core/editing.js';
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

// Takes the form out of the list before any redraw.
// Without this, `container.innerHTML = ''` deletes #insertion-form from the
// document and the interface is left with no form at all until the page is
// reloaded.
function moveFormBoxOutOfList() {
    const formBox = document.getElementById('insertion-form');
    const container = document.getElementById('element-list');
    if (formBox && container && container.contains(formBox)) {
        document.getElementById('list-area').appendChild(formBox);
        return true;
    }
    return false;
}

// Puts the form back right under the item being edited (or at the end of the area).
export function positionFormBox(index) {
    const formBox = document.getElementById('insertion-form');
    if (!formBox) return;
    const items = document.getElementById('element-list').children;
    if (index >= 0 && items[index]) items[index].insertAdjacentElement('afterend', formBox);
    else document.getElementById('list-area').appendChild(formBox);
}

// The bar above the list: tick everything, how many are ticked, and the two
// things worth doing to several at once.
//
// It is shown whenever the list has anything in it, and not only once something
// is ticked. A bar that appears with the first tick would hide the *only* way of
// ticking everything, which is the one case where selecting several by hand is
// most tedious. The two action buttons are dead until something is ticked, which
// says the same thing without hiding it.
function renderSelectionBar(count) {
    const bar = document.getElementById('selection-bar');
    if (!bar) return;
    if (!count) {
        bar.style.display = 'none';
        bar.innerHTML = '';
        return;
    }
    const chosen = pickedCount(listContext());
    const dead = chosen ? '' : 'disabled';
    bar.style.display = 'flex';
    bar.innerHTML = `
        <label class="pick-all">
            <input type="checkbox" data-action="pick-all" ${allPicked(listContext(), count) ? 'checked' : ''}>
            <span>${escapeHtml(t('selectAll'))}</span>
        </label>
        <span class="pick-count">${chosen ? escapeHtml(t('selectedCount')).replace('%1', () => chosen) : ''}</span>
        <button class="btn-action copy" data-action="copy-picked" ${dead} title="${escapeHtml(t('copySelected'))}"><i class="fas fa-copy"></i></button>
        <button class="btn-action delete" data-action="delete-picked" ${dead} title="${escapeHtml(t('deleteSelected'))}"><i class="fas fa-trash"></i></button>
    `;
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
function buildRows(currentArray) {
    const effNodes = getEffectiveNodes(currentArray);
    return currentArray.map((item, index) => {
        const div = document.createElement('div');
        div.className = 'list-item';
        div.innerHTML = `
            <div style="display:flex; align-items:center; flex:1; overflow:hidden;">
                <input type="checkbox" class="item-pick" data-action="pick-element" data-index="${index}" ${isPicked(listContext(), index) ? 'checked' : ''} title="${escapeHtml(t('select'))}">
                <i class="fas fa-grip-vertical item-drag"></i>
                <span class="item-text">${escapeHtml(rowTitle(item, index, effNodes[index]))}</span>
            </div>
            <div class="item-actions">
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

    const rows = buildRows(currentArray);

    // Nothing below can fail on the project's content: from here on the old
    // list is replaced by one that is already complete.
    const formWasInTheList = moveFormBoxOutOfList();
    container.innerHTML = '';
    rows.forEach(row => container.appendChild(row));
    renderSelectionBar(currentArray.length);
    // The form goes back under the element it is editing. Before, a copy or a
    // delete elsewhere in the list left it stranded at the bottom, under
    // nothing in particular.
    if (formWasInTheList) positionFormBox(state.editingIndex);

    // One instance per container: before, every render created another one without
    // destroying the previous, piling listeners onto the same list.
    if (sortableInstance) sortableInstance.destroy();
    sortableInstance = new Sortable(container, {
        handle: '.item-drag',
        // Only the rows move, and only rows are counted.
        //
        // The open form is a child of this same container, sitting under the
        // element it edits. With Sortable's defaults it counted as an item:
        // `oldIndex` and `newIndex` are positions among **all** the children,
        // so with the form above the dragged row every index was one too high.
        // Dragging the last of four shafts to the top with a form open asked
        // for element 4 of a four-element list: nothing was removed, and
        // `undefined` was inserted at the top -- saved as `null`, and every
        // later render of that list failed. Measured in a browser, not guessed.
        //
        // `draggable` keeps the form from being picked up, and the
        // `...DraggableIndex` pair counts rows only.
        draggable: '.list-item',
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
