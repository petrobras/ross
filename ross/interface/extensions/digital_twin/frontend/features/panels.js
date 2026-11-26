// The side panels of the Digital Twin screens: the step buttons that show one
// panel of a group, the sidebars, and the cards that fold. The ROSS interface's
// `toggleSidebar` acts on its own modelling sidebar, so these screens keep
// theirs, addressed by id.
import { t } from '/core/i18n.js';

export function openPanel(group, name) {
    document.querySelectorAll(`.tab-btn[data-tab-group="${group}"]`).forEach(button => {
        button.classList.toggle('active', button.dataset.tab === name);
    });
    document.querySelectorAll(`[data-panel-group="${group}"]`).forEach(panel => {
        panel.style.display = panel.dataset.panel === name ? '' : 'none';
    });
}

// The step buttons of a group are numbered in the order they are shown, so a
// step that is hidden (the process data, without the pre-established models;
// the balancing, without rossml) leaves no gap. The panel titles take the
// number of their button. Called again after `applyLanguage`, which writes
// the texts back without it.
export function numberSteps(group) {
    let number = 0;
    document.querySelectorAll(`.tab-btn[data-tab-group="${group}"]`).forEach(button => {
        const title = document.querySelector(
            `[data-panel-group="${group}"][data-panel="${button.dataset.tab}"] .panel-title`);
        if (button.style.display === 'none') return;
        number += 1;
        [button, title].forEach(element => {
            if (element) element.textContent = `${number}. ${t(element.dataset.i18n)}`;
        });
    });
}

export function toggleSidebarById(sidebarId) {
    const sidebar = document.getElementById(sidebarId);
    if (!sidebar) return;
    sidebar.classList.toggle('collapsed');
    setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
}

export function toggleCard(bodyId) {
    const body = document.getElementById(bodyId);
    if (!body) return;
    body.style.display = body.style.display === 'none' ? '' : 'none';
    setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
}
