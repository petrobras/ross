// The line under the side panel that says what is happening. Errors do not go
// here: they open the alert dialog, as in the reference interface.
import { escapeHtml } from '/core/dom.js';

const ICONS = {
    busy: 'fas fa-sync fa-spin',
    ok: 'fas fa-check-circle',
};

export function setStatus(id, text, kind) {
    const node = document.getElementById(id);
    if (!node) return;
    if (!text) {
        node.className = 'panel-status';
        node.innerHTML = '';
        return;
    }
    node.className = 'panel-status ' + (kind || 'ok');
    node.innerHTML = `<i class="${ICONS[kind] || ICONS.ok}"></i> <span>${escapeHtml(text)}</span>`;
}
