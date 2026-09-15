// The element list of the modeling screen: effective node numbering, the form box
// anchored to the item, and reordering by dragging.
import { escapeHtml } from '../core/dom.js';
import { t } from '../core/i18n.js';
import { state, getActiveData, syncBackToLibrary } from '../core/state.js';
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

let sortableInstance = null;

export function renderList() {
    const container = document.getElementById('element-list');
    moveFormBoxOutOfList();
    container.innerHTML = '';
    
    const activeData = getActiveData();
    const currentArray = activeData[state.currentTab];
    
    const effNodes = getEffectiveNodes(currentArray);
    currentArray.forEach((item, index) => {
        let titleStr = "";        
        if (state.currentTab === 'materials') {
            titleStr = `MATERIAL #${index + 1}`;
            if (item.name) titleStr += ` - ${item.name}`;
        } else if (state.currentTab === 'couplings') {
            titleStr = `COUPLING #${index + 1}`;
            if (item.element_type && item.element_type !== 'BASIC') titleStr += ` [${item.element_type}]`;
        } else {
            let singularName = (state.currentTab === 'pointmasses') ? 'POINTMASS' : state.currentTab.slice(0, -1).toUpperCase();
            let effectiveN = effNodes[index];
            titleStr = `${singularName} #${index + 1} (Node ${effectiveN})`;
            if (item.element_type && item.element_type !== 'BASIC') titleStr += ` [${item.element_type}]`;
        }
        const div = document.createElement('div');
        div.className = 'list-item';
        div.innerHTML = `
            <div style="display:flex; align-items:center; flex:1; overflow:hidden;">
                <i class="fas fa-grip-vertical item-drag"></i>
                <span class="item-text">${escapeHtml(titleStr)}</span>
            </div>
            <div class="item-actions">
                <button class="btn-action edit" onclick="editItem(${index})" title="${escapeHtml(t('edit'))}"><i class="fas fa-pen"></i></button>
                <button class="btn-action copy" onclick="copyItem(${index})" title="${escapeHtml(t('copy'))}"><i class="fas fa-copy"></i></button>
                <button class="btn-action delete" onclick="deleteItem(${index})" title="${escapeHtml(t('delete'))}"><i class="fas fa-trash"></i></button>
            </div>
        `;
        container.appendChild(div);
    });
    // One instance per container: before, every render created another one without
    // destroying the previous, piling listeners onto the same list.
    if (sortableInstance) sortableInstance.destroy();
    sortableInstance = new Sortable(container, {
        handle: '.item-drag',
        animation: 150,
        onEnd: function (evt) {
            const oldIdx = evt.oldIndex;
            const newIdx = evt.newIndex;
            if(oldIdx === newIdx) return;
            
            const actData = getActiveData();
            const item = actData[state.currentTab][oldIdx];
            
            actData[state.currentTab].splice(oldIdx, 1);
            actData[state.currentTab].splice(newIdx, 0, item);            
            
            syncBackToLibrary();
            renderList();
            reorderHandler();
        }
    });
}
