// A menu at the pointer: what can be done to the thing under it. The 3D
// view's (features/rotor3d.js) opens it with the right button released where
// it was pressed -- dragging with it still moves the view -- or with
// Shift+F10 / the menu key on the part last clicked.
//
// Entries are { label, icon, run, danger?, disabled? }, { header } for a title
// line, or null for a separator. A button carries only its position; the
// functions stay here, with the menu that was opened, so a menu drawn for one
// part can never run another's.
import { escapeHtml } from '../core/dom.js';

let entries = [];

function menu() {
    return document.getElementById('context-menu');
}

export function contextMenuOpen() {
    const box = menu();
    return !!box && !box.hidden;
}

function entryHTML(entry, position) {
    if (!entry) return '<div class="context-separator" role="separator"></div>';
    if (entry.header) return `<div class="context-header">${escapeHtml(entry.header)}</div>`;
    return `<button type="button" role="menuitem" class="context-item${entry.danger ? ' is-danger' : ''}"`
        + ` data-action="context-item" data-index="${position}"${entry.disabled ? ' disabled' : ''}>`
        + `<i class="fas ${escapeHtml(entry.icon || 'fa-circle')}" aria-hidden="true"></i>`
        + `<span>${escapeHtml(entry.label)}</span></button>`;
}

// At (x, y) in the viewport, kept inside it.
export function openContextMenu(x, y, list) {
    const box = menu();
    if (!box) return;
    entries = list.slice();
    box.innerHTML = entries.map(entryHTML).join('');
    box.hidden = false;
    const width = document.documentElement.clientWidth;
    const height = document.documentElement.clientHeight;
    box.style.left = `${Math.max(4, Math.min(x, width - (box.offsetWidth || 0) - 4))}px`;
    box.style.top = `${Math.max(4, Math.min(y, height - (box.offsetHeight || 0) - 4))}px`;
    const first = box.querySelector('.context-item:not([disabled])');
    if (first && typeof first.focus === 'function') first.focus();
}

export function closeContextMenu() {
    const box = menu();
    if (box) {
        box.hidden = true;
        box.innerHTML = '';
    }
    entries = [];
}

// Closed first: what the entry does may open a dialog of its own.
export function runContextItem(position) {
    const entry = entries[position];
    closeContextMenu();
    if (entry && !entry.header && !entry.disabled && typeof entry.run === 'function') return entry.run();
    return undefined;
}

// The browser's own menu, while ours is open. On Windows the browser asks
// for its menu AFTER the button is released -- after the 3D view has opened
// ours under the pointer -- so the question comes from our menu, not from the
// canvas that refuses it, and both menus opened (Leonardo, in Chrome). Linux
// and macOS ask on the press, before ours exists, and the canvas refuses it
// there. Once ours is closed -- a right click anywhere else closes it on the
// press -- the browser's menu is the page's again.
export function blockBrowserMenu(event) {
    if (contextMenuOpen()) event.preventDefault();
}

// A press anywhere else, Escape, or the page scrolling away from it closes it.
export function startContextMenu() {
    document.addEventListener('contextmenu', blockBrowserMenu, true);
    document.addEventListener('pointerdown', event => {
        const box = menu();
        if (contextMenuOpen() && !(box.contains && box.contains(event.target))) closeContextMenu();
    }, true);
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && contextMenuOpen()) closeContextMenu();
    });
    document.addEventListener('scroll', () => { if (contextMenuOpen()) closeContextMenu(); }, true);
}
